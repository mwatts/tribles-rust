//! Content-addressed witnesses for exact collection support.
//!
//! A [`CollectionData`](super::CollectionData) digest identifies only the
//! representation bytes. [`CollectionElement`] additionally binds those bytes
//! to one collection descriptor and one exact set of signed commits. The
//! support set is committed by a canonical [`Blake3Merkle`] PATCH root over
//! each commit's full intrinsic hash, so construction order, duplicates, and
//! PATCH edit history do not affect the identity.
//!
//! These types are structural. Decoding an element or support root does not
//! prove that the named commits exist, are authorized, or join to the named
//! data. That semantic validation remains the collection resolver's job.

use crate::inline::encodings::hash::Blake3;
use crate::inline::Inline;
use crate::patch::{Blake3Merkle, IdentitySchema, PATCH};

use super::{CollectionCommit, CollectionData, CollectionHandle};

/// Exact byte length of a canonical [`CollectionElement`] body.
pub const COLLECTION_ELEMENT_BYTES_LEN: usize = 3 * 32;

/// Canonical Merkle commitment to a nonempty exact set of signed commits.
///
/// Leaves are the complete 32-byte intrinsic hashes of canonical
/// [`CollectionCommit`] records, not their compact 128-bit entity ids. The
/// root uses PATCH's versioned BLAKE3 leaf and branch framing. An empty support
/// set has no root and is represented by `None` from [`Self::from_commits`].
/// A root is a commitment, not a materialized set: two roots cannot be joined
/// without the canonical PATCH nodes or commit hashes beneath them.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SupportRoot([u8; 32]);

impl SupportRoot {
    /// Commit to the exact set of supplied commit records.
    ///
    /// Input order and byte-identical repeats are ignored. This operation is
    /// structural and deliberately does not verify signatures or authority.
    pub fn from_commits<'a>(
        commits: impl IntoIterator<Item = &'a CollectionCommit>,
    ) -> Option<Self> {
        let support = PATCH::<32, IdentitySchema, (), Blake3Merkle>::from_keys(
            commits.into_iter().map(CollectionCommit::intrinsic_hash),
        );
        support.merkle_root().map(Self)
    }

    /// Decode an opaque root digest received from storage or a protocol.
    ///
    /// Every 32-byte digest is structurally representable. Trust requires
    /// separately obtaining and validating the canonical support tree.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the canonical root bytes.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Borrow the canonical root bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// One addressable collection value with its exact authority support.
///
/// Its canonical body is `collection || support || data`, three consecutive
/// 32-byte fields. The BLAKE3 digest of those 96 bytes is the element's content
/// address. Construction and decoding establish only that byte identity;
/// they do not establish semantic validity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CollectionElement {
    collection: CollectionHandle,
    support: SupportRoot,
    data: CollectionData,
}

impl CollectionElement {
    /// Bind one collection data identity to an exact commit-support root.
    pub const fn new(
        collection: CollectionHandle,
        support: SupportRoot,
        data: CollectionData,
    ) -> Self {
        Self {
            collection,
            support,
            data,
        }
    }

    /// Decode the exact fixed-width body.
    pub fn from_bytes(bytes: [u8; COLLECTION_ELEMENT_BYTES_LEN]) -> Self {
        Self::new(
            Inline::new(field(&bytes, 0)),
            SupportRoot::from_bytes(field(&bytes, 1)),
            Inline::new(field(&bytes, 2)),
        )
    }

    /// Collection whose lattice contains this value.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Exact signed-commit support for this value.
    pub const fn support(&self) -> SupportRoot {
        self.support
    }

    /// Content identity of the collection representation.
    pub const fn data(&self) -> CollectionData {
        self.data
    }

    /// Encode `collection || support || data` without relying on Rust layout.
    pub fn to_bytes(&self) -> [u8; COLLECTION_ELEMENT_BYTES_LEN] {
        let mut bytes = [0; COLLECTION_ELEMENT_BYTES_LEN];
        bytes[..32].copy_from_slice(&self.collection.raw);
        bytes[32..64].copy_from_slice(self.support.as_bytes());
        bytes[64..].copy_from_slice(&self.data.raw);
        bytes
    }

    /// BLAKE3 content address of the exact 96-byte body.
    pub fn content_hash(&self) -> [u8; 32] {
        Blake3::digest(&self.to_bytes())
    }
}

fn field(bytes: &[u8; COLLECTION_ELEMENT_BYTES_LEN], index: usize) -> [u8; 32] {
    bytes[index * 32..(index + 1) * 32]
        .try_into()
        .expect("collection element fields have exact width")
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use hex_literal::hex;

    use super::*;
    use crate::collection::empty_metadata_handle;

    type SupportPatch = PATCH<32, IdentitySchema, (), Blake3Merkle>;

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn commit(key: u8, collection: u8, data: u8) -> CollectionCommit {
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            self::collection(collection),
            self::data(data),
            empty_metadata_handle(),
        )
    }

    fn support_patch(commits: &[CollectionCommit]) -> SupportPatch {
        SupportPatch::from_keys(commits.iter().map(CollectionCommit::intrinsic_hash))
    }

    fn root(patch: &SupportPatch) -> SupportRoot {
        SupportRoot::from_bytes(patch.merkle_root().expect("nonempty support"))
    }

    #[test]
    fn support_root_is_a_canonical_set_commitment() {
        let a = commit(1, 1, 10);
        let b = commit(2, 1, 11);
        let c = commit(3, 1, 12);

        assert_eq!(SupportRoot::from_commits([]), None);
        assert_eq!(
            SupportRoot::from_commits([&a, &b, &c, &a]),
            SupportRoot::from_commits([&c, &b, &a]),
        );

        let mut idempotent = support_patch(&[a]);
        idempotent.union(support_patch(&[a]));
        assert_eq!(root(&idempotent), root(&support_patch(&[a])));

        let mut ab = support_patch(&[a]);
        ab.union(support_patch(&[b]));
        let mut ba = support_patch(&[b]);
        ba.union(support_patch(&[a]));
        assert_eq!(root(&ab), root(&ba));

        let mut left = ab;
        left.union(support_patch(&[c]));
        let mut bc = support_patch(&[b]);
        bc.union(support_patch(&[c]));
        let mut right = support_patch(&[a]);
        right.union(bc);
        assert_eq!(root(&left), root(&right));
        assert_eq!(
            root(&left),
            SupportRoot::from_commits([&a, &b, &c]).unwrap(),
        );
    }

    #[test]
    fn distinct_authors_of_identical_data_remain_distinct_support() {
        let first = commit(1, 1, 10);
        let second = commit(2, 1, 10);
        assert_ne!(first.intrinsic_hash(), second.intrinsic_hash());

        let combined = support_patch(&[first, second]);
        assert_eq!(combined.len(), 2);
        assert_ne!(root(&combined), root(&support_patch(&[first])));
    }

    #[test]
    fn singleton_support_and_collection_element_are_golden() {
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            collection(1),
            data(2),
            Inline::new([3; 32]),
        );
        let support = SupportRoot::from_commits([&commit]).unwrap();
        assert_eq!(
            support.into_bytes(),
            hex!("98692252BFAB005E4059B196165DDDBE2B92334E6A284CD2E8EED0508F71315A")
        );

        let element = CollectionElement::new(collection(1), support, data(2));
        let bytes = element.to_bytes();
        assert_eq!(bytes.len(), COLLECTION_ELEMENT_BYTES_LEN);
        assert_eq!(&bytes[..32], &[1; 32]);
        assert_eq!(&bytes[32..64], support.as_bytes());
        assert_eq!(&bytes[64..], &[2; 32]);
        assert_eq!(CollectionElement::from_bytes(bytes), element);
        assert_eq!(
            element.content_hash(),
            hex!("A31ED3F3D6D4ED1886043A9B0324FB58B6A9AA4D0BA9CC21AB711DDB4C1EF578")
        );

        assert_ne!(
            CollectionElement::new(collection(9), support, data(2)).content_hash(),
            element.content_hash(),
        );
        assert_ne!(
            CollectionElement::new(collection(1), SupportRoot::from_bytes([9; 32]), data(2))
                .content_hash(),
            element.content_hash(),
        );
        assert_ne!(
            CollectionElement::new(collection(1), support, data(9)).content_hash(),
            element.content_hash(),
        );
    }
}
