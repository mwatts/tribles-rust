//! Orthogonal, monotone publication grants for collections.
//!
//! A [`CollectionGossip`] is deliberately not a collection-calculus record:
//! it does not add a member, merge elements, derive a representation, or keep
//! any blob alive. It is low-level store metadata beside blob wants. Its sole
//! meaning is that one author permanently permits redistribution of that
//! author's strictly verified [`CollectionCommit`](super::CollectionCommit)s
//! in one collection.
//!
//! Grants are signed because pile concatenation is itself merge. An unsigned
//! collection-wide bit copied from an untrusted pile could otherwise opt a
//! later holder into publishing private commits. Author scoping also means a
//! grant never publishes another author's commits merely because both authors
//! wrote to the same collection.
//!
//! The store algebra is a grow-only set. There is intentionally no
//! `ungossip`: publication cannot be undone after another peer has observed
//! it. A node's decision to run a gossip service is runtime policy, not
//! durable truth. Material that must remain private belongs in a different
//! collection identity and must never receive a grant.

use std::error::Error;
use std::fmt;

use ed25519::signature::Signer;
use ed25519::Signature;
use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::id::{id_hex, Id};
use crate::inline::encodings::ed25519::{ED25519PublicKey, ED25519RComponent, ED25519SComponent};
use crate::inline::Inline;

use super::CollectionHandle;

/// Stable semantic kind of a signed collection-gossip grant.
///
/// Minted with `trible genid` on 2026-08-12:
/// `9BB5B1F4D6FD8FB850B494C2CF51B5CA`.
pub const KIND_COLLECTION_GOSSIP: Id = id_hex!("9BB5B1F4D6FD8FB850B494C2CF51B5CA");

/// Version of the signed gossip-grant transcript.
pub const GOSSIP_TRANSCRIPT_VERSION: u32 = 1;

/// Domain prefix of the signed gossip-grant transcript.
pub const GOSSIP_TRANSCRIPT_DOMAIN: &[u8] = b"triblespace.collection.gossip.transcript";

/// Canonical byte length of one collection-gossip witness.
///
/// The fixed layout is `collection || author || signature_r || signature_s`,
/// with four 32-byte fields and no padding. Decoding these bytes is
/// structural only; call [`CollectionGossip::verify_strict`] before treating
/// the result as a publication grant.
pub const COLLECTION_GOSSIP_BYTES_LEN: usize = 128;

/// Number of bytes in a version-1 gossip-grant transcript.
pub const GOSSIP_TRANSCRIPT_LEN: usize = GOSSIP_TRANSCRIPT_DOMAIN.len()
    + 16 // kind id
    + 4 // version
    + 32 // author public key
    + 32; // collection descriptor handle

/// Permanent permission to redistribute one author's valid commits in a
/// collection.
///
/// This value is structural evidence. Decoding it does not make it trusted;
/// consumers must call [`verify_strict`](Self::verify_strict) before treating
/// it as permission. Strict verification proves only authorship of the grant.
/// Collection-commit admission remains the caller's separate authorization
/// policy.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CollectionGossip {
    collection: CollectionHandle,
    public_key: Inline<ED25519PublicKey>,
    signature_r: Inline<ED25519RComponent>,
    signature_s: Inline<ED25519SComponent>,
}

impl CollectionGossip {
    /// Sign a permanent redistribution grant for `collection`.
    pub fn sign(signing_key: &SigningKey, collection: CollectionHandle) -> Self {
        let public_key = Inline::new(signing_key.verifying_key().to_bytes());
        let transcript = gossip_transcript(public_key, collection);
        let signature: Signature = signing_key.sign(&transcript);
        Self::from_parts(
            collection,
            public_key,
            Inline::new(*signature.r_bytes()),
            Inline::new(*signature.s_bytes()),
        )
    }

    /// Reconstruct stored structural evidence without trusting its signature.
    pub(crate) fn from_parts(
        collection: CollectionHandle,
        public_key: Inline<ED25519PublicKey>,
        signature_r: Inline<ED25519RComponent>,
        signature_s: Inline<ED25519SComponent>,
    ) -> Self {
        Self {
            collection,
            public_key,
            signature_r,
            signature_s,
        }
    }

    /// Decode one exact canonical 128-byte witness without trusting it.
    ///
    /// Every byte string of this fixed size has a structural representation;
    /// invalid public keys and signatures remain available as evidence and
    /// are rejected by [`verify_strict`](Self::verify_strict).
    pub fn from_bytes(bytes: [u8; COLLECTION_GOSSIP_BYTES_LEN]) -> Self {
        Self::from_parts(
            Inline::new(bytes[0..32].try_into().expect("fixed collection field")),
            Inline::new(bytes[32..64].try_into().expect("fixed author field")),
            Inline::new(bytes[64..96].try_into().expect("fixed signature field")),
            Inline::new(bytes[96..128].try_into().expect("fixed signature field")),
        )
    }

    /// Encode this witness into its exact canonical 128-byte layout.
    pub fn to_bytes(&self) -> [u8; COLLECTION_GOSSIP_BYTES_LEN] {
        let mut bytes = [0u8; COLLECTION_GOSSIP_BYTES_LEN];
        bytes[0..32].copy_from_slice(&self.collection.raw);
        bytes[32..64].copy_from_slice(&self.public_key.raw);
        bytes[64..96].copy_from_slice(&self.signature_r.raw);
        bytes[96..128].copy_from_slice(&self.signature_s.raw);
        bytes
    }

    /// Strictly verify the Ed25519 signature over the canonical transcript.
    pub fn verify_strict(&self) -> Result<(), GossipVerificationError> {
        let public_key = VerifyingKey::from_bytes(&self.public_key.raw)
            .map_err(|_| GossipVerificationError::InvalidPublicKey)?;
        let signature = Signature::from_components(self.signature_r.raw, self.signature_s.raw);
        public_key
            .verify_strict(&self.signing_transcript(), &signature)
            .map_err(|_| GossipVerificationError::InvalidSignature)
    }

    /// Exact bytes attested by this grant's signature.
    pub fn signing_transcript(&self) -> Vec<u8> {
        gossip_transcript(self.public_key, self.collection)
    }

    /// Collection whose commits may be redistributed.
    pub fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Author whose commits are covered by the grant.
    pub fn public_key(&self) -> Inline<ED25519PublicKey> {
        self.public_key
    }

    /// Raw signature components.
    pub fn signature(&self) -> (Inline<ED25519RComponent>, Inline<ED25519SComponent>) {
        (self.signature_r, self.signature_s)
    }
}

/// Semantic verification failure for a signed gossip grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GossipVerificationError {
    /// The author bytes do not encode an Ed25519 verifying key.
    InvalidPublicKey,
    /// Strict Ed25519 verification rejected the transcript/signature pair.
    InvalidSignature,
}

impl fmt::Display for GossipVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKey => write!(formatter, "collection gossip has an invalid key"),
            Self::InvalidSignature => {
                write!(formatter, "collection gossip signature is invalid")
            }
        }
    }
}

impl Error for GossipVerificationError {}

/// Grow-only storage for signed collection-publication grants.
///
/// Re-insertion is idempotent. Implementations enumerate their currently
/// observed set in deterministic value order. There is deliberately no
/// removal operation: once permission has escaped, no local mutation can
/// retract it from the world.
pub trait CollectionGossipStore {
    /// Failure while enumerating grants.
    type GossipsError: Error + fmt::Debug + Send + Sync + 'static;
    /// Failure while admitting structural grant evidence.
    type GossipError: Error + fmt::Debug + Send + Sync + 'static;

    /// Borrowing iterator over one deterministic view of known grants.
    type GossipIter<'a>: Iterator<Item = Result<CollectionGossip, Self::GossipsError>>
    where
        Self: 'a;

    /// Enumerate currently known grants in deterministic value order.
    fn gossips<'a>(&'a mut self) -> Result<Self::GossipIter<'a>, Self::GossipsError>;

    /// Insert one immutable grant.
    ///
    /// Storage preserves structural evidence; consumers still perform strict
    /// signature verification before granting publication permission.
    fn gossip(&mut self, grant: CollectionGossip) -> Result<(), Self::GossipError>;
}

impl<S> CollectionGossipStore for &mut S
where
    S: CollectionGossipStore + ?Sized,
{
    type GossipsError = S::GossipsError;
    type GossipError = S::GossipError;
    type GossipIter<'a>
        = S::GossipIter<'a>
    where
        Self: 'a;

    fn gossips<'a>(&'a mut self) -> Result<Self::GossipIter<'a>, Self::GossipsError> {
        (**self).gossips()
    }

    fn gossip(&mut self, grant: CollectionGossip) -> Result<(), Self::GossipError> {
        (**self).gossip(grant)
    }
}

fn gossip_transcript(public_key: Inline<ED25519PublicKey>, collection: CollectionHandle) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(GOSSIP_TRANSCRIPT_LEN);
    transcript.extend_from_slice(GOSSIP_TRANSCRIPT_DOMAIN);
    transcript.extend_from_slice(&KIND_COLLECTION_GOSSIP.raw());
    transcript.extend_from_slice(&GOSSIP_TRANSCRIPT_VERSION.to_be_bytes());
    transcript.extend_from_slice(&public_key.raw);
    transcript.extend_from_slice(&collection.raw);
    debug_assert_eq!(transcript.len(), GOSSIP_TRANSCRIPT_LEN);
    transcript
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    #[test]
    fn signed_grant_roundtrips_and_is_deterministic() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let first = CollectionGossip::sign(&key, collection(3));
        let second = CollectionGossip::sign(&key, collection(3));

        assert_eq!(first, second);
        assert_eq!(first.collection(), collection(3));
        assert_eq!(first.signing_transcript().len(), GOSSIP_TRANSCRIPT_LEN);
        assert_eq!(
            first.signing_transcript(),
            hex_literal::hex!(
                "747269626C6573706163652E636F6C6C656374696F6E2E676F737369702E7472616E736372697074
                 9BB5B1F4D6FD8FB850B494C2CF51B5CA
                 00000001
                 EA4A6C63E29C520ABEF5507B132EC5F9954776AEBEBE7B92421EEA691446D22C
                 0303030303030303030303030303030303030303030303030303030303030303"
            )
            .to_vec()
        );
        assert_eq!(
            first.signature_r.raw,
            hex_literal::hex!("283AC5E4CF477F452A8C099F217FAC91EA67843A484A311C2026D529A91981A1")
        );
        assert_eq!(
            first.signature_s.raw,
            hex_literal::hex!("E6D1675C7608CF2F6480218716E99899E9649171809E094B60FCA3B8669CFD00")
        );
        first.verify_strict().unwrap();
        assert_eq!(CollectionGossip::from_bytes(first.to_bytes()), first);
        assert_eq!(
            &first.to_bytes()[0..64],
            &[collection(3).raw, first.public_key().raw].concat()
        );
    }

    #[test]
    fn byte_decode_is_structural_and_strict_verification_stays_explicit() {
        let key = SigningKey::from_bytes(&[19; 32]);
        let valid = CollectionGossip::sign(&key, collection(8));
        let mut bytes = valid.to_bytes();
        bytes[127] ^= 1;

        let structural = CollectionGossip::from_bytes(bytes);
        assert_eq!(structural.collection(), valid.collection());
        assert_eq!(structural.public_key(), valid.public_key());
        assert_eq!(
            structural.verify_strict(),
            Err(GossipVerificationError::InvalidSignature)
        );
    }

    #[test]
    fn signature_covers_author_and_collection() {
        let key = SigningKey::from_bytes(&[11; 32]);
        let valid = CollectionGossip::sign(&key, collection(4));
        let (r, s) = valid.signature();

        let other_collection =
            CollectionGossip::from_parts(collection(5), valid.public_key(), r, s);
        assert_eq!(
            other_collection.verify_strict(),
            Err(GossipVerificationError::InvalidSignature)
        );

        let other_key = SigningKey::from_bytes(&[12; 32]);
        let other_author = CollectionGossip::from_parts(
            valid.collection(),
            Inline::new(other_key.verifying_key().to_bytes()),
            r,
            s,
        );
        assert_eq!(
            other_author.verify_strict(),
            Err(GossipVerificationError::InvalidSignature)
        );
    }
}
