//! Messages crossing the synchronous store / asynchronous host boundary.
//!
//! Inventory admission is monotone. The host streams authenticated leaves to
//! the store side in bounded batches, where one refresh drain inserts all
//! available batches and crosses a single durability barrier.

use anybytes::Bytes;
use triblespace_core::capability::CapabilityProof;
use triblespace_core::collection::{
    COLLECTION_COMMIT_BYTES_LEN, COLLECTION_DERIVE_BYTES_LEN, COLLECTION_MERGE_BYTES_LEN,
    CollectionRecord,
};
use triblespace_core::repo::ArtifactOfferSnapshot;
use triblespace_core::repo::peer::PeerEvidence;

use crate::protocol::RawHash;
use crate::transport::PeerId;

/// A newly installed immutable local observation.
///
/// The snapshot slot is replaced before this command is sent. The host uses
/// the first notice to start anti-entropy immediately and later notices only
/// to learn newly stored PEER evidence; periodic scheduling remains bounded.
pub(crate) struct SnapshotNotice {
    pub(crate) peers: Vec<PeerId>,
}

/// Commands sent from [`crate::peer::Peer`] to the host runtime.
pub(crate) enum NetCommand {
    SnapshotInstalled(SnapshotNotice),
    /// Replace the host's local publication policy observation.
    ///
    /// Offers are operational service intent, not another synchronized
    /// inventory component. The host intersects this set with its current
    /// immutable Blob serving snapshot before publishing any provider hint.
    ArtifactOffersUpdated(ArtifactOfferSnapshot),
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

impl NetEvent {
    fn admission_bytes(&self) -> usize {
        match self {
            Self::Peer(_) => 64,
            Self::CollectionRecord(CollectionRecord::Commit(_)) => 1 + COLLECTION_COMMIT_BYTES_LEN,
            Self::CollectionRecord(CollectionRecord::Merge(_)) => 1 + COLLECTION_MERGE_BYTES_LEN,
            Self::CollectionRecord(CollectionRecord::Derive(_)) => 1 + COLLECTION_DERIVE_BYTES_LEN,
            Self::CapabilityProof(proof) => proof.as_bytes().len(),
            Self::Blob { bytes, .. } => 32usize.saturating_add(bytes.len()),
        }
    }
}

/// Maximum number of independently authenticated items carried by one
/// host-to-store message.
pub(crate) const MAX_ADMISSION_BATCH_ITEMS: usize = 256;
/// Soft byte ceiling for one host-to-store message.
///
/// Blob values are indivisible at this boundary. One blob larger than this
/// ceiling is therefore carried alone; `Bytes` keeps the file-backed receive
/// mapping shared instead of copying it into the channel.
pub(crate) const MAX_ADMISSION_BATCH_BYTES: usize = 512 * 1024;
/// Maximum number of batches buffered across the async/synchronous bridge and
/// consumed by one refresh drain.
pub(crate) const MAX_ADMISSION_BRIDGE_BATCHES: usize = 16;

/// One bounded unit of monotone store admission.
#[derive(Debug, Default)]
pub(crate) struct NetEventBatch {
    events: Vec<NetEvent>,
    bytes: usize,
}

impl NetEventBatch {
    pub(crate) fn singleton(event: NetEvent) -> Self {
        let bytes = event.admission_bytes();
        Self {
            events: vec![event],
            bytes,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }

    pub(crate) fn into_events(self) -> impl Iterator<Item = NetEvent> {
        self.events.into_iter()
    }

    /// Append `event`, or return it unchanged when the nonempty batch has
    /// reached either bound. An indivisible oversized event is accepted only
    /// into an empty batch and immediately makes it ready to send.
    pub(crate) fn try_push(&mut self, event: NetEvent) -> Result<(), NetEvent> {
        let event_bytes = event.admission_bytes();
        let exceeds_count = self.events.len() >= MAX_ADMISSION_BATCH_ITEMS;
        let exceeds_bytes = self
            .bytes
            .checked_add(event_bytes)
            .is_none_or(|bytes| bytes > MAX_ADMISSION_BATCH_BYTES);
        if !self.events.is_empty() && (exceeds_count || exceeds_bytes) {
            return Err(event);
        }
        self.bytes = self.bytes.saturating_add(event_bytes);
        self.events.push(event);
        Ok(())
    }

    pub(crate) fn is_full(&self) -> bool {
        self.events.len() >= MAX_ADMISSION_BATCH_ITEMS || self.bytes >= MAX_ADMISSION_BATCH_BYTES
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use triblespace_core::repo::peer::PeerEvidence;

    use super::*;

    fn peer(byte: u8) -> NetEvent {
        NetEvent::Peer(PeerEvidence::new(
            SigningKey::from_bytes(&[0xA5; 32]).verifying_key(),
            SigningKey::from_bytes(&[byte; 32]).verifying_key(),
        ))
    }

    #[test]
    fn admission_batches_enforce_count_and_byte_bounds() {
        let mut count_bounded = NetEventBatch::default();
        for byte in 0..MAX_ADMISSION_BATCH_ITEMS {
            count_bounded.try_push(peer(byte as u8)).unwrap();
        }
        assert!(count_bounded.is_full());
        assert!(count_bounded.try_push(peer(0xFF)).is_err());

        let mut byte_bounded = NetEventBatch::default();
        let almost_half = vec![0x11; MAX_ADMISSION_BATCH_BYTES / 2];
        byte_bounded
            .try_push(NetEvent::Blob {
                hash: [0x11; 32],
                bytes: Bytes::from_source(almost_half.clone()),
            })
            .unwrap();
        let rejected = NetEvent::Blob {
            hash: [0x22; 32],
            bytes: Bytes::from_source(almost_half),
        };
        assert!(byte_bounded.try_push(rejected).is_err());
    }

    #[test]
    fn indivisible_oversized_blob_is_a_singleton() {
        let mut batch = NetEventBatch::default();
        batch
            .try_push(NetEvent::Blob {
                hash: [0x33; 32],
                bytes: Bytes::from_source(vec![0x33; MAX_ADMISSION_BATCH_BYTES + 1]),
            })
            .unwrap();
        assert_eq!(batch.len(), 1);
        assert!(batch.is_full());
        assert!(batch.try_push(peer(0x44)).is_err());
    }
}
