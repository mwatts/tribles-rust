//! Opaque, collection-scoped wakeups over stock `iroh-gossip`.
//!
//! A collection handle is both the exact gossip topic id and the discovery
//! capability for that topic. Joining therefore has no separate authorization
//! exchange. The only application payload is a fixed-width signed wake: it
//! says that one endpoint has some anti-entropy state under an opaque root.
//! Records, counts, blobs, proofs, and the anti-entropy protocol itself stay
//! outside this plane.

use std::fmt;
use std::sync::Arc;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use futures::StreamExt as _;
use iroh_base::EndpointId;
use iroh_gossip::api::{ApiError, Event as GossipEvent, GossipReceiver, GossipSender, Message};
use iroh_gossip::proto::DeliveryScope;
use iroh_gossip::{Gossip, TopicId};
use triblespace_core::collection::CollectionHandle;

/// Domain prefix for the signed wake transcript.
pub const COLLECTION_WAKE_TRANSCRIPT_DOMAIN: &[u8] = b"triblespace.collection.wake";

/// Current dense wake-envelope and signature-transcript version.
pub const COLLECTION_WAKE_VERSION: u8 = 1;

/// Exact number of bytes in a collection wake.
pub const COLLECTION_WAKE_WIRE_LEN: usize = 1 + 32 + 32 + 64;

const COLLECTION_WAKE_TRANSCRIPT_LEN: usize =
    COLLECTION_WAKE_TRANSCRIPT_DOMAIN.len() + 1 + 32 + 32 + 32;

/// An opaque root naming the state an origin can reconcile for one collection.
///
/// The wake plane neither interprets this value nor uses it as authority. Its
/// meaning belongs to the separately authorized anti-entropy protocol.
/// A collection implementation may derive it from a domain-separated product
/// of several summaries; no individual PATCH root or leaf count is privileged
/// by this wire format. After a wake, the receiver opens READ(C)-authorized
/// repair against the signed [`CollectionWake::origin`] and obtains the current
/// component summaries there.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CollectionWakeRoot([u8; 32]);

impl CollectionWakeRoot {
    /// Wrap one opaque anti-entropy root.
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the root bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl From<[u8; 32]> for CollectionWakeRoot {
    fn from(bytes: [u8; 32]) -> Self {
        Self::new(bytes)
    }
}

/// A fixed-width collection wake signed by its claimed endpoint origin.
///
/// The collection is deliberately absent from the wire envelope because the
/// enclosing gossip topic already names it. It is nevertheless part of the
/// signature transcript, so replaying identical bytes on another collection
/// topic fails strict verification.
#[derive(Clone, Eq, PartialEq)]
pub struct CollectionWake {
    origin: EndpointId,
    root: CollectionWakeRoot,
    signature: [u8; 64],
}

impl fmt::Debug for CollectionWake {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CollectionWake")
            .field("origin", &self.origin)
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl CollectionWake {
    /// Sign a wake with the same Ed25519 key used by the local iroh endpoint.
    pub fn sign(
        collection: CollectionHandle,
        root: CollectionWakeRoot,
        signing_key: &SigningKey,
    ) -> Self {
        let origin = EndpointId::from_bytes(signing_key.verifying_key().as_bytes())
            .expect("an Ed25519 verifying key is a valid endpoint id");
        let transcript = wake_transcript(collection, origin, root);
        let signature = signing_key.sign(&transcript).to_bytes();
        Self {
            origin,
            root,
            signature,
        }
    }

    /// Decode exactly one dense envelope and strictly verify it for `collection`.
    pub fn decode_and_verify(
        collection: CollectionHandle,
        bytes: &[u8],
    ) -> Result<Self, CollectionWakeError> {
        if bytes.len() != COLLECTION_WAKE_WIRE_LEN {
            return Err(CollectionWakeError::InvalidLength {
                expected: COLLECTION_WAKE_WIRE_LEN,
                actual: bytes.len(),
            });
        }
        if bytes[0] != COLLECTION_WAKE_VERSION {
            return Err(CollectionWakeError::UnsupportedVersion(bytes[0]));
        }

        let mut origin_bytes = [0; 32];
        origin_bytes.copy_from_slice(&bytes[1..33]);
        let origin = EndpointId::from_bytes(&origin_bytes)
            .map_err(|_| CollectionWakeError::InvalidOrigin)?;

        let mut root = [0; 32];
        root.copy_from_slice(&bytes[33..65]);

        let mut signature = [0; 64];
        signature.copy_from_slice(&bytes[65..]);
        let wake = Self {
            origin,
            root: CollectionWakeRoot::new(root),
            signature,
        };
        wake.verify(collection)?;
        Ok(wake)
    }

    /// Strictly verify this origin signature for the exact collection topic.
    pub fn verify(&self, collection: CollectionHandle) -> Result<(), CollectionWakeError> {
        let verifying_key = VerifyingKey::from_bytes(self.origin.as_bytes())
            .map_err(|_| CollectionWakeError::InvalidOrigin)?;
        let transcript = wake_transcript(collection, self.origin, self.root);
        verifying_key
            .verify_strict(&transcript, &Signature::from_bytes(&self.signature))
            .map_err(|_| CollectionWakeError::InvalidSignature)
    }

    /// Claimed origin, authenticated by the envelope signature.
    pub const fn origin(&self) -> EndpointId {
        self.origin
    }

    /// Opaque per-collection anti-entropy root.
    pub const fn root(&self) -> CollectionWakeRoot {
        self.root
    }

    /// Encode this wake in its exact dense wire form.
    pub fn to_bytes(&self) -> [u8; COLLECTION_WAKE_WIRE_LEN] {
        let mut bytes = [0; COLLECTION_WAKE_WIRE_LEN];
        bytes[0] = COLLECTION_WAKE_VERSION;
        bytes[1..33].copy_from_slice(self.origin.as_bytes());
        bytes[33..65].copy_from_slice(self.root.as_bytes());
        bytes[65..].copy_from_slice(&self.signature);
        bytes
    }

    /// Exact stock Plumtree message identity for this envelope.
    ///
    /// `iroh-gossip` 0.101 identifies messages by the BLAKE3 hash of their
    /// complete content. Ed25519 signatures are deterministic, so equal
    /// `(collection, origin, root)` wakes have equal identities. Different
    /// roots remain distinct; the wake plane performs no latest-root
    /// coalescing. Upstream's bounded message-id retention is transport-level
    /// duplicate suppression, not a durable checkpoint.
    pub fn dedup_id(&self) -> [u8; 32] {
        *blake3::hash(&self.to_bytes()).as_bytes()
    }
}

/// Structural or cryptographic rejection of a collection wake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionWakeError {
    /// The envelope was not exactly [`COLLECTION_WAKE_WIRE_LEN`] bytes.
    InvalidLength { expected: usize, actual: usize },
    /// The envelope used an unsupported version byte.
    UnsupportedVersion(u8),
    /// The claimed origin bytes were not a valid Ed25519 endpoint key.
    InvalidOrigin,
    /// Strict Ed25519 verification failed for this collection and root.
    InvalidSignature,
}

impl fmt::Display for CollectionWakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(
                    formatter,
                    "invalid collection wake length: expected {expected}, got {actual}"
                )
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported collection wake version {version}")
            }
            Self::InvalidOrigin => formatter.write_str("invalid collection wake origin"),
            Self::InvalidSignature => formatter.write_str("invalid collection wake signature"),
        }
    }
}

impl std::error::Error for CollectionWakeError {}

/// The stock gossip lifecycle and signing capability for collection wakes.
///
/// This type intentionally does not expose the underlying untyped gossip API.
/// Knowing a collection handle is sufficient to subscribe; authorization is
/// checked only when a receiver follows a wake into the separate anti-entropy
/// protocol.
#[derive(Clone)]
pub struct CollectionWakePlane {
    gossip: Gossip,
    signing_key: Arc<SigningKey>,
}

impl fmt::Debug for CollectionWakePlane {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CollectionWakePlane")
            .field("origin", &self.origin())
            .finish_non_exhaustive()
    }
}

impl CollectionWakePlane {
    pub(crate) fn spawn(endpoint: &iroh::Endpoint) -> Self {
        let signing_key = SigningKey::from_bytes(&endpoint.secret_key().to_bytes());
        debug_assert_eq!(
            signing_key.verifying_key().as_bytes(),
            endpoint.id().as_bytes()
        );
        let gossip = Gossip::builder()
            // Stock gossip enforces a 512-byte minimum. The typed API below
            // emits only the 129-byte envelope and rejects every other size.
            .max_message_size(iroh_gossip::proto::MIN_MAX_MESSAGE_SIZE)
            .spawn(endpoint.clone());
        Self {
            gossip,
            signing_key: Arc::new(signing_key),
        }
    }

    pub(crate) fn protocol_handler(&self) -> Gossip {
        self.gossip.clone()
    }

    /// Local endpoint identity used to sign wakes.
    pub fn origin(&self) -> EndpointId {
        EndpointId::from_bytes(self.signing_key.verifying_key().as_bytes())
            .expect("an Ed25519 verifying key is a valid endpoint id")
    }

    /// Derive the gossip topic exactly from the collection handle bytes.
    pub const fn topic_id(collection: CollectionHandle) -> TopicId {
        TopicId::from_bytes(collection.raw)
    }

    /// Join the collection mesh through zero or more stock gossip peers.
    ///
    /// No topic authorization is performed: possession of `collection` is
    /// intentionally the discovery capability.
    pub async fn subscribe(
        &self,
        collection: CollectionHandle,
        bootstrap: Vec<EndpointId>,
    ) -> Result<CollectionWakeTopic, ApiError> {
        let topic = self
            .gossip
            .subscribe(Self::topic_id(collection), bootstrap)
            .await?;
        let (sender, receiver) = topic.split();
        Ok(CollectionWakeTopic {
            collection,
            signing_key: self.signing_key.clone(),
            sender,
            receiver,
        })
    }
}

/// One subscription to the stock gossip mesh for an exact collection.
pub struct CollectionWakeTopic {
    collection: CollectionHandle,
    signing_key: Arc<SigningKey>,
    sender: GossipSender,
    receiver: GossipReceiver,
}

impl fmt::Debug for CollectionWakeTopic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CollectionWakeTopic")
            .field("collection", &self.collection)
            .field("neighbors", &self.receiver.neighbors().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl CollectionWakeTopic {
    /// Collection whose exact bytes form this topic id.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Current direct stock-gossip neighbors.
    pub fn neighbors(&self) -> impl Iterator<Item = EndpointId> + '_ {
        self.receiver.neighbors()
    }

    /// Whether at least one direct gossip neighbor is present.
    pub fn is_joined(&self) -> bool {
        self.receiver.is_joined()
    }

    /// Wait until at least one direct stock-gossip neighbor is present.
    pub async fn joined(&mut self) -> Result<(), ApiError> {
        self.receiver.joined().await
    }

    /// Ask stock gossip to join additional bootstrap peers.
    pub async fn join_peers(&self, peers: Vec<EndpointId>) -> Result<(), ApiError> {
        self.sender.join_peers(peers).await
    }

    /// Sign and gossip only an opaque root wake for this collection.
    pub async fn broadcast(&self, root: CollectionWakeRoot) -> Result<CollectionWake, ApiError> {
        let wake = CollectionWake::sign(self.collection, root, &self.signing_key);
        self.sender
            .broadcast(wake.to_bytes().to_vec().into())
            .await?;
        Ok(wake)
    }

    /// Receive the next typed topology, wake, rejection, or lag event.
    pub async fn next_event(&mut self) -> Result<Option<CollectionWakeEvent>, ApiError> {
        let Some(event) = self.receiver.next().await else {
            return Ok(None);
        };
        let event = event?;
        Ok(Some(match event {
            GossipEvent::NeighborUp(peer) => CollectionWakeEvent::NeighborUp(peer),
            GossipEvent::NeighborDown(peer) => CollectionWakeEvent::NeighborDown(peer),
            GossipEvent::Lagged => CollectionWakeEvent::Lagged,
            GossipEvent::Received(message) => decode_received(self.collection, message),
        }))
    }
}

/// A typed event from one collection wake mesh.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionWakeEvent {
    /// A direct stock-gossip neighbor joined this collection mesh.
    NeighborUp(EndpointId),
    /// A direct stock-gossip neighbor left this collection mesh.
    NeighborDown(EndpointId),
    /// A strictly verified signed wake arrived.
    Received(ReceivedCollectionWake),
    /// An invalid application envelope arrived and was not exposed as a wake.
    Rejected {
        /// Immediate gossip hop that delivered the invalid bytes.
        delivered_from: EndpointId,
        /// Stock gossip delivery scope.
        scope: DeliveryScope,
        /// Reason the envelope was rejected.
        error: CollectionWakeError,
    },
    /// The bounded upstream subscription queue lagged.
    Lagged,
}

/// A verified wake plus transport delivery metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceivedCollectionWake {
    /// Signed envelope. [`CollectionWake::origin`] is its authenticated author.
    pub wake: CollectionWake,
    /// Immediate gossip hop, which may be a relay and is not an author claim.
    pub delivered_from: EndpointId,
    /// Whether stock gossip delivered this directly or through the swarm.
    pub scope: DeliveryScope,
}

fn decode_received(collection: CollectionHandle, message: Message) -> CollectionWakeEvent {
    match CollectionWake::decode_and_verify(collection, &message.content) {
        Ok(wake) => CollectionWakeEvent::Received(ReceivedCollectionWake {
            wake,
            delivered_from: message.delivered_from,
            scope: message.scope,
        }),
        Err(error) => CollectionWakeEvent::Rejected {
            delivered_from: message.delivered_from,
            scope: message.scope,
            error,
        },
    }
}

fn wake_transcript(
    collection: CollectionHandle,
    origin: EndpointId,
    root: CollectionWakeRoot,
) -> [u8; COLLECTION_WAKE_TRANSCRIPT_LEN] {
    let mut transcript = [0; COLLECTION_WAKE_TRANSCRIPT_LEN];
    let mut offset = 0;
    transcript[..COLLECTION_WAKE_TRANSCRIPT_DOMAIN.len()]
        .copy_from_slice(COLLECTION_WAKE_TRANSCRIPT_DOMAIN);
    offset += COLLECTION_WAKE_TRANSCRIPT_DOMAIN.len();
    transcript[offset] = COLLECTION_WAKE_VERSION;
    offset += 1;
    transcript[offset..offset + 32].copy_from_slice(&collection.raw);
    offset += 32;
    transcript[offset..offset + 32].copy_from_slice(origin.as_bytes());
    offset += 32;
    transcript[offset..].copy_from_slice(root.as_bytes());
    transcript
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collection(byte: u8) -> CollectionHandle {
        CollectionHandle::new([byte; 32])
    }

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    #[test]
    fn topic_id_is_exact_collection_handle() {
        let collection = collection(0xA3);
        assert_eq!(
            CollectionWakePlane::topic_id(collection).as_bytes(),
            &collection.raw
        );
    }

    #[test]
    fn exact_codec_fixture_and_signature_round_trip() {
        let collection = collection(0x11);
        let wake = CollectionWake::sign(collection, CollectionWakeRoot::new([0x22; 32]), &key(7));
        let encoded = wake.to_bytes();
        assert_eq!(encoded.len(), COLLECTION_WAKE_WIRE_LEN);
        assert_eq!(
            hex::encode(encoded),
            "01ea4a6c63e29c520abef5507b132ec5f9954776aebebe7b92421eea691446d22c22222222222222222222222222222222222222222222222222222222222222220df3a4469d65f8497f975757fd87f664a5a8375b3590613f0ea710d25ba88df2bf80ad4e3dab8456edc03d13cd80e005c1cf6289282473d2384310a7e067a30c"
        );
        assert_eq!(
            CollectionWake::decode_and_verify(collection, &encoded),
            Ok(wake)
        );
    }

    #[test]
    fn strict_verification_binds_collection_origin_and_root() {
        let expected = collection(0x31);
        let wake = CollectionWake::sign(expected, CollectionWakeRoot::new([0x41; 32]), &key(9));
        let encoded = wake.to_bytes();

        assert_eq!(
            CollectionWake::decode_and_verify(collection(0x32), &encoded),
            Err(CollectionWakeError::InvalidSignature)
        );

        for index in [1, 33, 65, COLLECTION_WAKE_WIRE_LEN - 1] {
            let mut tampered = encoded;
            tampered[index] ^= 1;
            assert!(
                CollectionWake::decode_and_verify(expected, &tampered).is_err(),
                "tampering byte {index} must fail"
            );
        }

        let mut wrong_version = encoded;
        wrong_version[0] = COLLECTION_WAKE_VERSION + 1;
        assert_eq!(
            CollectionWake::decode_and_verify(expected, &wrong_version),
            Err(CollectionWakeError::UnsupportedVersion(2))
        );
        assert_eq!(
            CollectionWake::decode_and_verify(expected, &encoded[..encoded.len() - 1]),
            Err(CollectionWakeError::InvalidLength {
                expected: COLLECTION_WAKE_WIRE_LEN,
                actual: COLLECTION_WAKE_WIRE_LEN - 1,
            })
        );
    }

    #[test]
    fn dedup_is_deterministic_exact_wire_identity_not_root_coalescing() {
        let collection = collection(0x51);
        let signing_key = key(0x61);
        let first = CollectionWake::sign(
            collection,
            CollectionWakeRoot::new([0x71; 32]),
            &signing_key,
        );
        let same = CollectionWake::sign(
            collection,
            CollectionWakeRoot::new([0x71; 32]),
            &signing_key,
        );
        let next_root = CollectionWake::sign(
            collection,
            CollectionWakeRoot::new([0x72; 32]),
            &signing_key,
        );
        let next_origin =
            CollectionWake::sign(collection, CollectionWakeRoot::new([0x71; 32]), &key(0x62));

        assert_eq!(first.to_bytes(), same.to_bytes());
        assert_eq!(first.dedup_id(), same.dedup_id());
        assert_ne!(first.dedup_id(), next_root.dedup_id());
        assert_ne!(first.dedup_id(), next_origin.dedup_id());
        assert_eq!(
            first.dedup_id(),
            *blake3::hash(&first.to_bytes()).as_bytes()
        );
    }

    #[test]
    fn signed_origin_is_independent_of_immediate_delivery_hop() {
        let collection = collection(0x81);
        let origin_key = key(0x82);
        let wake =
            CollectionWake::sign(collection, CollectionWakeRoot::new([0x83; 32]), &origin_key);
        let relay = EndpointId::from_bytes(key(0x84).verifying_key().as_bytes()).unwrap();
        let event = decode_received(
            collection,
            Message {
                content: wake.to_bytes().to_vec().into(),
                scope: DeliveryScope::Swarm(1u16.into()),
                delivered_from: relay,
            },
        );

        let CollectionWakeEvent::Received(received) = event else {
            panic!("valid signed wake was rejected");
        };
        assert_eq!(received.wake.origin(), wake.origin());
        assert_ne!(received.wake.origin(), received.delivered_from);
        assert_eq!(received.delivered_from, relay);
        assert!(!received.scope.is_direct());
    }
}
