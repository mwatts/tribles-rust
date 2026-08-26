//! Native storage for monotone peer-routing evidence.
//!
//! A [`PeerEvidence`] value says only that one peer public key has been
//! observed in association with one team trust-root public key. It is a
//! grow-only routing hint. Presence grants no authority and proves neither
//! liveness, reachability, residency, nor continued membership.

use std::error::Error;
use std::fmt;
use std::fmt::Debug;

use ed25519_dalek::VerifyingKey;

use crate::id::{id_hex, Id};
use crate::inline::encodings::hash::{Blake3, Hash};
use crate::inline::Inline;

/// Stable semantic kind of canonical `PEER(team_public_key, peer_public_key)`
/// evidence.
///
/// Minted with `trible genid` on 2026-08-26. The pile record description is
/// rooted at this same anchor.
pub const KIND_PEER_EVIDENCE: Id = id_hex!("E25B4427F30DCE7B36F3F80BB38E375A");

/// Exact byte length of canonical peer evidence.
pub const PEER_EVIDENCE_BYTES_LEN: usize = 64;

/// Version of the peer-evidence content identity.
pub const PEER_EVIDENCE_ID_VERSION: u32 = 1;

/// Domain prefix of peer-evidence content identity.
pub const PEER_EVIDENCE_ID_DOMAIN: &[u8] = b"triblespace.peer.evidence.id";

/// BLAKE3 inventory identity of one exact canonical [`PeerEvidence`] body.
///
/// The dense body itself remains the storage key. This fixed-width digest is
/// the portable selector used when an inventory must compare heterogeneous
/// record kinds without carrying their full bodies.
pub type PeerEvidenceId = Inline<Hash<Blake3>>;

/// Positive routing evidence associating one peer with one team.
///
/// The canonical dense representation is exactly the team Ed25519 public key
/// followed by the peer Ed25519 public key. Both compressed points are
/// validated at construction. There is intentionally no inverse or retraction
/// value: stores form a grow-only set under union.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PeerEvidence([u8; PEER_EVIDENCE_BYTES_LEN]);

impl PeerEvidence {
    /// Construct evidence from validated Ed25519 public keys.
    pub fn new(team: VerifyingKey, peer: VerifyingKey) -> Self {
        let mut bytes = [0u8; PEER_EVIDENCE_BYTES_LEN];
        bytes[..32].copy_from_slice(team.as_bytes());
        bytes[32..].copy_from_slice(peer.as_bytes());
        Self(bytes)
    }

    /// Decode and validate one canonical dense body.
    pub fn from_bytes(
        bytes: [u8; PEER_EVIDENCE_BYTES_LEN],
    ) -> Result<Self, PeerEvidenceDecodeError> {
        let mut team = [0u8; 32];
        let mut peer = [0u8; 32];
        team.copy_from_slice(&bytes[..32]);
        peer.copy_from_slice(&bytes[32..]);
        VerifyingKey::from_bytes(&team).map_err(PeerEvidenceDecodeError::InvalidTeamKey)?;
        VerifyingKey::from_bytes(&peer).map_err(PeerEvidenceDecodeError::InvalidPeerKey)?;
        Ok(Self(bytes))
    }

    /// Return the exact canonical dense body.
    pub const fn to_bytes(self) -> [u8; PEER_EVIDENCE_BYTES_LEN] {
        self.0
    }

    /// Borrow the exact canonical dense body.
    pub const fn as_bytes(&self) -> &[u8; PEER_EVIDENCE_BYTES_LEN] {
        &self.0
    }

    /// Team trust-root public key named by this evidence.
    pub fn team(self) -> VerifyingKey {
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&self.0[..32]);
        VerifyingKey::from_bytes(&bytes)
            .expect("PeerEvidence construction validates its team public key")
    }

    /// Peer public key named by this evidence.
    pub fn peer(self) -> VerifyingKey {
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&self.0[32..]);
        VerifyingKey::from_bytes(&bytes)
            .expect("PeerEvidence construction validates its peer public key")
    }

    /// Content identity of this exact canonical body.
    pub fn id(self) -> PeerEvidenceId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(PEER_EVIDENCE_ID_DOMAIN);
        hasher.update(&PEER_EVIDENCE_ID_VERSION.to_be_bytes());
        hasher.update(&KIND_PEER_EVIDENCE.raw());
        hasher.update(&self.0);
        Inline::new(*hasher.finalize().as_bytes())
    }
}

/// Failure while decoding canonical peer evidence.
#[derive(Debug)]
pub enum PeerEvidenceDecodeError {
    /// The team field is not a valid compressed Ed25519 public key.
    InvalidTeamKey(ed25519_dalek::SignatureError),
    /// The peer field is not a valid compressed Ed25519 public key.
    InvalidPeerKey(ed25519_dalek::SignatureError),
}

impl fmt::Display for PeerEvidenceDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTeamKey(error) => write!(f, "invalid team public key: {error}"),
            Self::InvalidPeerKey(error) => write!(f, "invalid peer public key: {error}"),
        }
    }
}

impl Error for PeerEvidenceDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidTeamKey(error) | Self::InvalidPeerKey(error) => Some(error),
        }
    }
}

/// Storage surface for positive peer-routing evidence.
///
/// Insertion is idempotent and enumeration is deterministic. The interface
/// deliberately has no removal operation and carries no authorization or
/// availability semantics.
pub trait PeerStore {
    /// Failure while enumerating peer evidence.
    type PeersError: Error + Debug + Send + Sync + 'static;
    /// Failure while inserting peer evidence.
    type InsertError: Error + Debug + Send + Sync + 'static;

    /// Borrowing iterator over one deterministic view of known evidence.
    type PeerIter<'a>: Iterator<Item = Result<PeerEvidence, Self::PeersError>>
    where
        Self: 'a;

    /// Enumerate currently known evidence in canonical byte order.
    fn peers<'a>(&'a mut self) -> Result<Self::PeerIter<'a>, Self::PeersError>;

    /// Insert one positive routing fact.
    fn insert_peer(&mut self, evidence: PeerEvidence) -> Result<(), Self::InsertError>;
}

impl<S> PeerStore for &mut S
where
    S: PeerStore + ?Sized,
{
    type PeersError = S::PeersError;
    type InsertError = S::InsertError;
    type PeerIter<'a>
        = S::PeerIter<'a>
    where
        Self: 'a;

    fn peers<'a>(&'a mut self) -> Result<Self::PeerIter<'a>, Self::PeersError> {
        (**self).peers()
    }

    fn insert_peer(&mut self, evidence: PeerEvidence) -> Result<(), Self::InsertError> {
        (**self).insert_peer(evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    use crate::repo::memoryrepo::MemoryRepo;

    fn key(byte: u8) -> VerifyingKey {
        SigningKey::from_bytes(&[byte; 32]).verifying_key()
    }

    #[test]
    fn dense_codec_round_trips_keys() {
        let evidence = PeerEvidence::new(key(1), key(2));
        assert_eq!(evidence.team(), key(1));
        assert_eq!(evidence.peer(), key(2));
        assert_eq!(
            PeerEvidence::from_bytes(evidence.to_bytes()).unwrap(),
            evidence
        );
    }

    #[test]
    fn identity_is_stable_and_field_sensitive() {
        let evidence = PeerEvidence::new(key(1), key(2));
        assert_eq!(evidence.id(), PeerEvidence::new(key(1), key(2)).id());
        assert_ne!(evidence.id(), PeerEvidence::new(key(1), key(3)).id());
        assert_ne!(evidence.id(), PeerEvidence::new(key(3), key(2)).id());
    }

    #[test]
    fn dense_order_is_team_then_peer() {
        let low_peer = PeerEvidence::new(key(1), key(2));
        let high_peer = PeerEvidence::new(key(1), key(3));
        let high_team = PeerEvidence::new(key(4), key(1));
        assert_eq!(
            [high_team, high_peer, low_peer]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>(),
            [low_peer, high_peer, high_team]
        );
    }

    #[test]
    fn memory_store_is_an_idempotent_deterministic_set() {
        let low = PeerEvidence::new(key(1), key(2));
        let high = PeerEvidence::new(key(4), key(1));
        let mut repo = MemoryRepo::default();

        repo.insert_peer(high).unwrap();
        repo.insert_peer(low).unwrap();
        repo.insert_peer(high).unwrap();

        let peers = repo
            .peers()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut expected = vec![low, high];
        expected.sort();
        assert_eq!(peers, expected);
    }
}
