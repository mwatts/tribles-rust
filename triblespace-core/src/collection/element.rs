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

use std::collections::HashSet;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::Blake3MerkleNode;
use crate::blob::{Blob, BlobEncoding, Bytes, TryFromBlob};
use crate::id::{id_hex, ExclusiveId, Id};
use crate::inline::encodings::hash::{Blake3, Handle};
use crate::inline::{Encodes, Inline};
use crate::macros::entity;
use crate::metadata::{self, MetaDescribe};
use crate::patch::{
    Blake3Merkle, Blake3MerkleNodeBlob, Blake3MerkleNodeDecodeError, Entry, IdentitySchema,
    PATCHMerkleNode, PATCH,
};
use crate::repo::{BlobStoreGet, BlobStorePut};
use crate::trible::Fragment;

use super::{CollectionCommit, CollectionData, CollectionHandle};

/// Exact byte length of a canonical [`CollectionElement`] body.
pub const COLLECTION_ELEMENT_BYTES_LEN: usize = 3 * 32;

type SupportPatch = PATCH<32, IdentitySchema, (), Blake3Merkle>;

/// Blob encoding for the exact fixed-width body of a [`CollectionElement`].
pub struct CollectionElementBlob;

/// Content-addressed identity of one exact [`CollectionElement`] body.
pub type CollectionElementHandle = Inline<Handle<CollectionElementBlob>>;

impl BlobEncoding for CollectionElementBlob {}

impl MetaDescribe for CollectionElementBlob {
    fn describe() -> Fragment {
        // Minted with `trible genid` on 2026-08-27.
        let id: Id = id_hex!("6452157382999C15AEFB759EE537D3F2");
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "collection-element",
                metadata::description: "Exact 96-byte collection element body: aligned collection descriptor handle, exact commit-support root, and collection data handle. Its BLAKE3 handle is the element identity.",
                metadata::tag: metadata::KIND_BLOB_ENCODING,
        }
    }
}

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
        SupportSet::from_commits(commits).root()
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

    /// Reinterpret this root as the typed handle of its canonical PATCH node.
    pub fn node_handle(self) -> Inline<Handle<Blake3MerkleNode>> {
        Inline::new(self.0)
    }

    /// Strictly load and validate the complete support tree rooted here.
    ///
    /// Every node is fetched through the generic blob-store interface. The
    /// loader independently verifies each content hash, the canonical v3 node
    /// encoding, and every immediate prefix/edge/count relationship before it
    /// reconstructs the exact key-only PATCH.
    pub fn load<R>(
        self,
        reader: &R,
    ) -> Result<SupportSet, SupportLoadError<R::GetError<Infallible>>>
    where
        R: BlobStoreGet,
    {
        let mut state = SupportLoadState::default();
        load_support_node(reader, self.0, None, &mut state)?;

        let patch = SupportPatch::from_keys(state.keys);
        let actual = patch
            .merkle_root()
            .expect("a successfully loaded support root contains a leaf");
        if actual != self.0 {
            return Err(SupportLoadError::ReconstructedRootMismatch {
                expected: self.0,
                actual,
            });
        }
        Ok(SupportSet { patch })
    }
}

/// A non-authoritative, exact set of commit hashes and its canonical support tree.
///
/// This is deliberately a thin semantic wrapper around the existing key-only
/// PATCH. It adds no codec or storage state: building and union are PATCH set
/// operations, [`Self::root`] is `None` for the empty set, and materialization
/// stores the existing canonical v3 nodes verbatim.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct SupportSet {
    patch: SupportPatch,
}

impl fmt::Debug for SupportSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SupportSet")
            .field("len", &self.len())
            .field("root", &self.root())
            .finish()
    }
}

impl SupportSet {
    /// Construct the empty support set. Its root is [`None`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Build exact support from signed commit identities.
    pub fn from_commits<'a>(commits: impl IntoIterator<Item = &'a CollectionCommit>) -> Self {
        Self {
            patch: SupportPatch::from_keys(
                commits.into_iter().map(CollectionCommit::intrinsic_hash),
            ),
        }
    }

    /// Insert one signed commit identity.
    pub fn insert(&mut self, commit: &CollectionCommit) {
        self.patch.insert(&Entry::<32, (), Blake3Merkle>::new(
            &commit.intrinsic_hash(),
        ));
    }

    /// Join another exact support set into this one.
    pub fn union(&mut self, other: Self) {
        self.patch.union(other.patch);
    }

    /// Number of distinct signed commit identities.
    pub fn len(&self) -> u64 {
        self.patch.len()
    }

    /// Whether this support set has no commits.
    pub fn is_empty(&self) -> bool {
        self.patch.is_empty()
    }

    /// Canonical support root, or `None` for the empty set.
    pub fn root(&self) -> Option<SupportRoot> {
        self.patch.merkle_root().map(SupportRoot)
    }

    /// Iterate over commit hashes in canonical byte order.
    pub fn commit_hashes(&self) -> impl ExactSizeIterator<Item = &[u8; 32]> + '_ {
        self.patch.iter_ordered()
    }

    /// Store every logical support node in postorder through `BlobStorePut`.
    ///
    /// The return value is the existing root, not a second manifest. Empty
    /// support stores nothing and returns `None`.
    pub fn materialize<S>(&self, store: &mut S) -> Result<Option<SupportRoot>, S::PutError>
    where
        S: BlobStorePut,
    {
        let Some(root) = self.patch.merkle_node(&[]) else {
            return Ok(None);
        };
        materialize_support_node(root, store)?;
        Ok(self.root())
    }
}

#[derive(Default)]
struct SupportLoadState {
    active: HashSet<[u8; 32]>,
    complete: HashSet<[u8; 32]>,
    keys: Vec<[u8; 32]>,
}

#[derive(Clone, Copy)]
struct ExpectedChild<'a> {
    parent: [u8; 32],
    parent_prefix: &'a [u8],
    edge: u8,
    leaf_count: u64,
}

/// Failure while loading one exact support tree from ordinary blobs.
#[derive(Debug)]
pub enum SupportLoadError<E> {
    /// A node blob could not be retrieved.
    Get { digest: [u8; 32], source: E },
    /// Retrieved bytes did not have the digest by which the parent named them.
    DigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// A node did not have the one canonical version-3 encoding.
    InvalidNode {
        digest: [u8; 32],
        source: Blake3MerkleNodeDecodeError,
    },
    /// A child prefix was not a strict extension of its parent's prefix.
    ChildPrefixMismatch { parent: [u8; 32], child: [u8; 32] },
    /// The parent descriptor's edge did not select the child's prefix.
    ChildEdgeMismatch {
        parent: [u8; 32],
        child: [u8; 32],
        declared: u8,
        actual: u8,
    },
    /// The parent descriptor's child count did not match the child node.
    ChildLeafCountMismatch {
        parent: [u8; 32],
        child: [u8; 32],
        declared: u64,
        actual: u64,
    },
    /// Hostile input attempted to recurse back into a node still being loaded.
    Cycle { digest: [u8; 32] },
    /// Hostile input referenced one logical node from multiple tree positions.
    RepeatedNode { digest: [u8; 32] },
    /// Rebuilding from the validated leaves did not reproduce the named root.
    ReconstructedRootMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
}

impl<E: fmt::Display> fmt::Display for SupportLoadError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get { source, .. } => {
                write!(formatter, "support node could not be read: {source}")
            }
            Self::DigestMismatch { .. } => {
                formatter.write_str("support node bytes do not match their BLAKE3 address")
            }
            Self::InvalidNode { source, .. } => write!(formatter, "invalid support node: {source}"),
            Self::ChildPrefixMismatch { .. } => {
                formatter.write_str("support child prefix does not extend its parent")
            }
            Self::ChildEdgeMismatch { .. } => {
                formatter.write_str("support child prefix does not match its parent edge")
            }
            Self::ChildLeafCountMismatch { .. } => {
                formatter.write_str("support child leaf count does not match its parent descriptor")
            }
            Self::Cycle { .. } => formatter.write_str("support tree contains a cycle"),
            Self::RepeatedNode { .. } => {
                formatter.write_str("support tree references one node more than once")
            }
            Self::ReconstructedRootMismatch { .. } => {
                formatter.write_str("validated support leaves reconstruct to a different root")
            }
        }
    }
}

impl<E> Error for SupportLoadError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Get { source, .. } => Some(source),
            Self::InvalidNode { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn materialize_support_node<S>(
    node: PATCHMerkleNode<'_, 32, ()>,
    store: &mut S,
) -> Result<(), S::PutError>
where
    S: BlobStorePut,
{
    for (_, child) in node.children() {
        materialize_support_node(child, store)?;
    }

    let digest = node.digest();
    let blob = Blob::<Blake3MerkleNode>::with_handle(
        Bytes::from_source(node.canonical_blob_bytes()),
        Inline::new(digest),
    );
    let stored = store.put::<Blake3MerkleNode, _>(blob)?;
    debug_assert_eq!(stored.raw, digest, "BlobStorePut returned the wrong handle");
    Ok(())
}

fn load_support_node<R>(
    reader: &R,
    digest: [u8; 32],
    expected: Option<ExpectedChild<'_>>,
    state: &mut SupportLoadState,
) -> Result<(), SupportLoadError<R::GetError<Infallible>>>
where
    R: BlobStoreGet,
{
    if state.complete.contains(&digest) {
        return Err(SupportLoadError::RepeatedNode { digest });
    }
    if !state.active.insert(digest) {
        return Err(SupportLoadError::Cycle { digest });
    }

    let blob: Blob<Blake3MerkleNode> = reader
        .get(Inline::new(digest))
        .map_err(|source| SupportLoadError::Get { digest, source })?;
    let actual = Blake3::digest(blob.bytes.as_ref());
    if actual != digest {
        return Err(SupportLoadError::DigestMismatch {
            expected: digest,
            actual,
        });
    }
    let node = Blake3MerkleNodeBlob::<32>::decode(blob.bytes.as_ref())
        .map_err(|source| SupportLoadError::InvalidNode { digest, source })?;

    if let Some(expected) = expected {
        let prefix = node.prefix();
        if prefix.len() <= expected.parent_prefix.len()
            || !prefix.starts_with(expected.parent_prefix)
        {
            return Err(SupportLoadError::ChildPrefixMismatch {
                parent: expected.parent,
                child: digest,
            });
        }
        let actual_edge = prefix[expected.parent_prefix.len()];
        if actual_edge != expected.edge {
            return Err(SupportLoadError::ChildEdgeMismatch {
                parent: expected.parent,
                child: digest,
                declared: expected.edge,
                actual: actual_edge,
            });
        }
        if node.leaf_count() != expected.leaf_count {
            return Err(SupportLoadError::ChildLeafCountMismatch {
                parent: expected.parent,
                child: digest,
                declared: expected.leaf_count,
                actual: node.leaf_count(),
            });
        }
    }

    if let Some(key) = node.key() {
        state.keys.push(*key);
    } else {
        for child in node.children() {
            load_support_node(
                reader,
                child.digest(),
                Some(ExpectedChild {
                    parent: digest,
                    parent_prefix: node.prefix(),
                    edge: child.edge(),
                    leaf_count: child.leaf_count(),
                }),
                state,
            )?;
        }
    }

    let removed = state.active.remove(&digest);
    debug_assert!(removed, "the loaded support node was active");
    let inserted = state.complete.insert(digest);
    debug_assert!(inserted, "the loaded support node was not complete");
    Ok(())
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

    /// Typed content address of the exact 96-byte body.
    pub fn handle(&self) -> CollectionElementHandle {
        Inline::new(self.content_hash())
    }
}

impl Encodes<CollectionElement> for CollectionElementBlob {
    type Output = Blob<CollectionElementBlob>;

    fn encode(source: CollectionElement) -> Self::Output {
        Blob::new(Bytes::from_source(source.to_bytes().to_vec()))
    }
}

impl Encodes<&CollectionElement> for CollectionElementBlob {
    type Output = Blob<CollectionElementBlob>;

    fn encode(source: &CollectionElement) -> Self::Output {
        CollectionElementBlob::encode(*source)
    }
}

/// Structural failure while decoding a typed collection-element blob.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionElementDecodeError {
    /// The body was not exactly three aligned 32-byte fields.
    InvalidLength { actual: usize },
    /// The bytes did not match the content address supplied by the store.
    DigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
}

impl fmt::Display for CollectionElementDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { actual } => write!(
                formatter,
                "collection element must be exactly {COLLECTION_ELEMENT_BYTES_LEN} bytes, got {actual}"
            ),
            Self::DigestMismatch { .. } => {
                formatter.write_str("collection element bytes do not match their BLAKE3 address")
            }
        }
    }
}

impl Error for CollectionElementDecodeError {}

impl TryFromBlob<CollectionElementBlob> for CollectionElement {
    type Error = CollectionElementDecodeError;

    fn try_from_blob(blob: Blob<CollectionElementBlob>) -> Result<Self, Self::Error> {
        let expected = blob.get_handle().raw;
        let actual = Blake3::digest(blob.bytes.as_ref());
        if actual != expected {
            return Err(CollectionElementDecodeError::DigestMismatch { expected, actual });
        }
        let bytes: [u8; COLLECTION_ELEMENT_BYTES_LEN] =
            blob.bytes.as_ref().try_into().map_err(|_| {
                CollectionElementDecodeError::InvalidLength {
                    actual: blob.bytes.len(),
                }
            })?;
        Ok(CollectionElement::from_bytes(bytes))
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
    use crate::blob::encodings::rawbytes::RawBytes;
    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::blob::encodings::UnknownBlob;
    use crate::blob::{IntoBlob, MemoryBlobStore};
    use crate::collection::empty_metadata_handle;
    use crate::repo::pile::{Pile, WantRewritePolicy};
    use crate::repo::{BlobChildren, BlobStore, RetentionRoots};
    use crate::trible::TribleSet;

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

    fn support_set(commits: &[CollectionCommit]) -> SupportSet {
        SupportSet::from_commits(commits)
    }

    fn root(support: &SupportSet) -> SupportRoot {
        support.root().expect("nonempty support")
    }

    fn materialized_support(
        commits: &[CollectionCommit],
    ) -> (SupportSet, SupportRoot, MemoryBlobStore) {
        let support = support_set(commits);
        let expected = root(&support);
        let mut store = MemoryBlobStore::new();
        assert_eq!(support.materialize(&mut store).unwrap(), Some(expected));
        (support, expected, store)
    }

    fn replace_root(
        entries: &[(Inline<Handle<UnknownBlob>>, Blob<UnknownBlob>)],
        old_root: SupportRoot,
        bytes: Vec<u8>,
    ) -> (SupportRoot, MemoryBlobStore) {
        let blob = Blob::<Blake3MerkleNode>::new(Bytes::from_source(bytes));
        let root = SupportRoot::from_bytes(blob.get_handle().raw);
        let handle = blob.get_handle().transmute();
        let mut entries = entries
            .iter()
            .cloned()
            .filter(|(handle, _)| handle.raw != old_root.into_bytes())
            .collect::<Vec<_>>();
        entries.push((handle, blob.transmute()));
        (root, entries.into_iter().collect())
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

        let mut idempotent = support_set(&[a]);
        idempotent.union(support_set(&[a]));
        assert_eq!(root(&idempotent), root(&support_set(&[a])));

        let mut ab = support_set(&[a]);
        ab.union(support_set(&[b]));
        let mut ba = support_set(&[b]);
        ba.union(support_set(&[a]));
        assert_eq!(root(&ab), root(&ba));

        let mut left = ab;
        left.union(support_set(&[c]));
        let mut bc = support_set(&[b]);
        bc.union(support_set(&[c]));
        let mut right = support_set(&[a]);
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

        let combined = support_set(&[first, second]);
        assert_eq!(combined.len(), 2);
        assert_ne!(root(&combined), root(&support_set(&[first])));
    }

    #[test]
    fn support_tree_materializes_and_loads_exactly() {
        let commits = [commit(1, 1, 10), commit(2, 1, 11), commit(3, 1, 12)];
        let (support, root, mut store) = materialized_support(&commits);
        let materialized_len = store.len();

        assert_eq!(support.materialize(&mut store).unwrap(), Some(root));
        assert_eq!(store.len(), materialized_len, "node puts are idempotent");

        let reader = store.reader().unwrap();
        let loaded = root.load(&reader).unwrap();
        assert_eq!(loaded, support);
        assert_eq!(loaded.root(), Some(root));
        assert_eq!(
            loaded.commit_hashes().copied().collect::<Vec<_>>(),
            support.commit_hashes().copied().collect::<Vec<_>>()
        );

        let empty = SupportSet::new();
        assert_eq!(empty.materialize(&mut store).unwrap(), None);
        assert_eq!(store.len(), materialized_len);
    }

    #[test]
    fn support_loader_rejects_missing_and_corrupt_children() {
        let commits = [commit(1, 1, 10), commit(2, 1, 11), commit(3, 1, 12)];
        let (_, root, mut store) = materialized_support(&commits);
        let reader = store.reader().unwrap();
        let root_blob: Blob<Blake3MerkleNode> = reader.get(root.node_handle()).unwrap();
        let root_node = Blake3MerkleNodeBlob::<32>::decode(root_blob.bytes.as_ref()).unwrap();
        let child = root_node
            .children()
            .next()
            .expect("three support leaves have a branch root")
            .digest();

        let mut missing: MemoryBlobStore = reader
            .clone()
            .into_iter()
            .filter(|(handle, _)| handle.raw != child)
            .collect();
        assert!(matches!(
            root.load(&missing.reader().unwrap()),
            Err(SupportLoadError::Get { digest, .. }) if digest == child
        ));

        let mut corrupt: MemoryBlobStore = reader
            .into_iter()
            .map(|(handle, blob)| {
                if handle.raw != child {
                    return (handle, blob);
                }
                let mut bytes = blob.bytes.as_ref().to_vec();
                bytes[0] ^= 1;
                (handle, Blob::<UnknownBlob>::new(Bytes::from_source(bytes)))
            })
            .collect();
        assert!(matches!(
            root.load(&corrupt.reader().unwrap()),
            Err(SupportLoadError::DigestMismatch { expected, .. }) if expected == child
        ));
    }

    #[test]
    fn support_loader_rejects_malformed_relationships_and_repeated_nodes() {
        let commits = [commit(1, 1, 10), commit(2, 1, 11), commit(3, 1, 12)];
        let (_, root, mut store) = materialized_support(&commits);
        let reader = store.reader().unwrap();
        let root_blob: Blob<Blake3MerkleNode> = reader.get(root.node_handle()).unwrap();
        let root_node = Blake3MerkleNodeBlob::<32>::decode(root_blob.bytes.as_ref()).unwrap();
        let children = root_node.children().collect::<Vec<_>>();
        assert!(children.len() < 256);
        let entries = reader.clone().into_iter().collect::<Vec<_>>();

        let prefix_len = root_node.prefix().len();
        let children_start = 64 + (prefix_len + 31) / 32 * 32;
        let edges = children
            .iter()
            .map(|child| child.edge())
            .collect::<Vec<_>>();
        let (changed_index, changed_edge) = edges
            .iter()
            .enumerate()
            .find_map(|(index, edge)| {
                let low = index
                    .checked_sub(1)
                    .map_or(0, |previous| u16::from(edges[previous]) + 1);
                let high = edges
                    .get(index + 1)
                    .map_or(255, |next| u16::from(*next) - 1);
                (low..=high)
                    .find(|candidate| *candidate != u16::from(*edge))
                    .map(|candidate| (index, candidate as u8))
            })
            .expect("a non-full branch has another ordered edge spelling");

        let mut malformed_bytes = root_blob.bytes.as_ref().to_vec();
        malformed_bytes[children_start + changed_index * 64] = changed_edge;
        let (malformed_root, mut malformed_store) = replace_root(&entries, root, malformed_bytes);
        assert!(matches!(
            malformed_root.load(&malformed_store.reader().unwrap()),
            Err(SupportLoadError::ChildEdgeMismatch { .. })
        ));

        let mut wrong_count_bytes = root_blob.bytes.as_ref().to_vec();
        let first_count_start = children_start + 8;
        wrong_count_bytes[first_count_start..first_count_start + 8]
            .copy_from_slice(&(children[0].leaf_count() + 1).to_le_bytes());
        wrong_count_bytes[40..48].copy_from_slice(&(root_node.leaf_count() + 1).to_le_bytes());
        let (wrong_count_root, mut wrong_count_store) =
            replace_root(&entries, root, wrong_count_bytes);
        assert!(matches!(
            wrong_count_root.load(&wrong_count_store.reader().unwrap()),
            Err(SupportLoadError::ChildLeafCountMismatch { .. })
        ));

        let mut repeated_bytes = root_blob.bytes.as_ref().to_vec();
        let first_digest = children[0].digest();
        let second_digest_start = children_start + 64 + 32;
        repeated_bytes[second_digest_start..second_digest_start + 32]
            .copy_from_slice(&first_digest);
        let (repeated_root, mut repeated_store) = replace_root(&entries, root, repeated_bytes);
        assert!(matches!(
            repeated_root.load(&repeated_store.reader().unwrap()),
            Err(SupportLoadError::RepeatedNode { digest }) if digest == first_digest
        ));
    }

    #[test]
    fn typed_element_blob_round_trips_and_checks_shape_and_digest() {
        let support = SupportRoot::from_commits([&commit(1, 1, 10)]).unwrap();
        let element = CollectionElement::new(collection(1), support, data(2));
        let mut store = MemoryBlobStore::new();
        let handle = store.put::<CollectionElementBlob, _>(element).unwrap();
        assert_eq!(handle, element.handle());
        assert_eq!(handle.raw, element.content_hash());

        let reader = store.reader().unwrap();
        assert_eq!(
            reader
                .get::<CollectionElement, CollectionElementBlob>(handle)
                .unwrap(),
            element
        );

        let short = Blob::<CollectionElementBlob>::new(Bytes::from_source(vec![0u8; 95]));
        assert_eq!(
            <CollectionElement as TryFromBlob<CollectionElementBlob>>::try_from_blob(short),
            Err(CollectionElementDecodeError::InvalidLength { actual: 95 })
        );

        let expected = Inline::<Handle<CollectionElementBlob>>::new(element.content_hash());
        let mut corrupt = element.to_bytes();
        corrupt[0] ^= 1;
        let corrupt = Blob::<CollectionElementBlob>::with_handle(
            Bytes::from_source(corrupt.to_vec()),
            expected,
        );
        assert!(matches!(
            <CollectionElement as TryFromBlob<CollectionElementBlob>>::try_from_blob(corrupt),
            Err(CollectionElementDecodeError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn recursive_element_retention_follows_generic_aligned_blob_children() {
        let mut store = MemoryBlobStore::new();
        let descriptor = store
            .put::<SimpleArchive, _>(IntoBlob::<SimpleArchive>::to_blob(TribleSet::new()))
            .unwrap();
        let data_handle = store
            .put::<RawBytes, _>(Bytes::from_source(b"collection data".to_vec()))
            .unwrap();
        let data = Inline::new(data_handle.raw);
        let commits = [
            CollectionCommit::sign(
                &SigningKey::from_bytes(&[1; 32]),
                descriptor,
                data,
                empty_metadata_handle(),
            ),
            CollectionCommit::sign(
                &SigningKey::from_bytes(&[2; 32]),
                descriptor,
                data,
                empty_metadata_handle(),
            ),
            CollectionCommit::sign(
                &SigningKey::from_bytes(&[3; 32]),
                descriptor,
                data,
                empty_metadata_handle(),
            ),
        ];
        let support = SupportSet::from_commits(&commits);
        let support_root = support.materialize(&mut store).unwrap().unwrap();
        let element = CollectionElement::new(descriptor, support_root, data);
        let element_handle = store.put::<CollectionElementBlob, _>(element).unwrap();
        let orphan = store
            .put::<RawBytes, _>(Bytes::from_source(b"unrelated".to_vec()))
            .unwrap();

        let reader = store.reader().unwrap();
        let element_children = reader.children(element_handle.transmute());
        assert!(element_children
            .iter()
            .any(|child| child.raw == descriptor.raw));
        assert!(element_children
            .iter()
            .any(|child| child.raw == support_root.into_bytes()));
        assert!(element_children.iter().any(|child| child.raw == data.raw));

        let mut roots = RetentionRoots::new();
        roots.retain_recursive(element_handle);
        let keep = roots.expanded(&reader);
        assert!(keep.iter().any(|handle| handle.raw == element_handle.raw));
        assert!(keep.iter().any(|handle| handle.raw == descriptor.raw));
        assert!(keep.iter().any(|handle| handle.raw == data.raw));
        assert!(keep
            .iter()
            .any(|handle| handle.raw == support_root.into_bytes()));
        assert!(!keep.iter().any(|handle| handle.raw == orphan.raw));
        for commit in &commits {
            assert!(
                !keep
                    .iter()
                    .any(|handle| handle.raw == commit.intrinsic_hash()),
                "commit hashes are support keys, not duplicate commit blobs"
            );
        }

        store.keep(keep);
        let kept = store.reader().unwrap();
        assert!(kept
            .get::<CollectionElement, CollectionElementBlob>(element_handle)
            .is_ok());
        assert!(support_root.load(&kept).is_ok());
        assert!(kept
            .get::<Blob<SimpleArchive>, SimpleArchive>(descriptor)
            .is_ok());
        assert!(kept.get::<Blob<RawBytes>, RawBytes>(data_handle).is_ok());
        assert!(kept.get::<Blob<RawBytes>, RawBytes>(orphan).is_err());
    }

    #[test]
    fn pile_rewrite_retains_complete_element_closure_and_collects_orphans() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("element-source.pile");
        let destination_path = directory.path().join("element-destination.pile");
        std::fs::File::create(&source_path).unwrap();
        std::fs::File::create(&destination_path).unwrap();
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();

        let descriptor = source
            .put::<SimpleArchive, _>(IntoBlob::<SimpleArchive>::to_blob(TribleSet::new()))
            .unwrap();
        let data_handle = source
            .put::<RawBytes, _>(Bytes::from_source(b"collection data".to_vec()))
            .unwrap();
        let data = Inline::new(data_handle.raw);
        let commits = [
            CollectionCommit::sign(
                &SigningKey::from_bytes(&[1; 32]),
                descriptor,
                data,
                empty_metadata_handle(),
            ),
            CollectionCommit::sign(
                &SigningKey::from_bytes(&[2; 32]),
                descriptor,
                data,
                empty_metadata_handle(),
            ),
            CollectionCommit::sign(
                &SigningKey::from_bytes(&[3; 32]),
                descriptor,
                data,
                empty_metadata_handle(),
            ),
        ];
        let support = SupportSet::from_commits(&commits);
        let support_root = support.materialize(&mut source).unwrap().unwrap();
        let element = CollectionElement::new(descriptor, support_root, data);
        let element_handle = source.put::<CollectionElementBlob, _>(element).unwrap();
        let orphan = source
            .put::<RawBytes, _>(Bytes::from_source(b"unrelated".to_vec()))
            .unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_recursive(element_handle);
        source
            .rewrite_retained_into(&mut destination, &roots, WantRewritePolicy::Drop)
            .unwrap();

        let reader = destination.reader().unwrap();
        assert_eq!(
            reader
                .get::<CollectionElement, CollectionElementBlob>(element_handle)
                .unwrap(),
            element
        );
        assert_eq!(support_root.load(&reader).unwrap(), support);
        assert!(reader
            .get::<Blob<SimpleArchive>, SimpleArchive>(descriptor)
            .is_ok());
        assert!(reader.get::<Blob<RawBytes>, RawBytes>(data_handle).is_ok());
        assert!(reader.get::<Blob<RawBytes>, RawBytes>(orphan).is_err());

        destination.close().unwrap();
        source.close().unwrap();
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
            hex!("C56A0E9B5573102E62E40210B90C5F69C83E040F8636C3C59408E3B479ECEDB2")
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
            hex!("15039C7B101782B35D30B77EC47B9FD7F60D51B0D574BD8DCBBA256D11D483E1")
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
