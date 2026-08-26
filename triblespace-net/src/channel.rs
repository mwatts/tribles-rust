//! Messages crossing the synchronous store / asynchronous host boundary.
//!
//! Inventory admission is monotone. The host streams authenticated leaves to
//! the store side, where one refresh drain inserts the whole available batch
//! and crosses a single durability barrier. Gossip never carries semantic
//! records; it only wakes the authenticated anti-entropy scheduler.

use anybytes::Bytes;
use triblespace_core::capability::CapabilityProof;
use triblespace_core::collection::CollectionRecord;
use triblespace_core::repo::peer::PeerEvidence;

use crate::inventory::InventoryGeneration;
use crate::protocol::RawHash;
use crate::transport::PeerId;

/// A newly installed immutable local observation.
///
/// The snapshot slot is replaced before this command is sent. Consequently a
/// generation wake can never race ahead of the bytes that
/// the direct protocol will serve.
pub(crate) struct SnapshotNotice {
    pub(crate) generation: InventoryGeneration,
    pub(crate) peers: Vec<PeerId>,
}

/// Commands sent from [`crate::peer::Peer`] to the host runtime.
pub(crate) enum NetCommand {
    SnapshotInstalled(SnapshotNotice),
}

/// Authenticated, structurally canonical inventory items returned by a walk.
///
/// These values remain inert evidence. In particular, a proof is not used as
/// ambient authority and PEER is only a routing hint.
#[derive(Debug)]
pub(crate) enum NetEvent {
    Peer(PeerEvidence),
    CollectionRecord(CollectionRecord),
    CapabilityProof(CapabilityProof),
    Blob { hash: RawHash, bytes: Bytes },
}
