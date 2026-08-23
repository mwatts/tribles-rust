//! Channel types bridging the async network thread and the sync store layer.
//!
//! `NetCommand`: outgoing effects sent from a [`Peer`](crate::peer::Peer)
//! into the network thread. Collection discovery is gossip-driven; exact
//! reconciliation requests use one bounded reply channel.
//! `NetEvent`: incoming data sent back from the network thread to be
//! applied into the wrapped store.
//!
use std::sync::mpsc;
use triblespace_core::collection::CollectionHandle;

use crate::protocol::RawHash;
use crate::transport::PeerId;
use triblespace_core::collection::CollectionCommit;

/// Commands sent to the network thread.
///
/// The surface is minimal by design. Immutable collection evidence floods the
/// team topic, while content remains independently addressable through the
/// DHT-routed `OP_GET_BLOB` path or exact collection RPCs.
pub enum NetCommand {
    /// Announce a blob hash to the DHT (fire-and-forget). Local
    /// puts trigger this; new providers improve the swarm's
    /// content-distribution fan-out.
    Announce(RawHash),
    /// Gossip one strictly verified collection commit.
    ///
    /// This is immutable ledger evidence, not an admission decision. The
    /// receiving side must not treat the mesh carrier as the commit author or
    /// infer local author trust from transport delivery.
    GossipCollectionEvidence { evidence: CollectionCommit },
    /// Fetch one exact collection's signed sparse evidence from a
    /// specific authenticated peer. The host runtime executes the async
    /// transport work; the synchronous `Peer` side owns admission policy.
    FetchCollectionEvidence {
        peer: PeerId,
        collection: CollectionHandle,
        reply: mpsc::Sender<anyhow::Result<Vec<CollectionCommit>>>,
    },
    // The swarm-addressed read-miss fetch is no longer a command: it
    // runs inline via `NetSender::fetch_blob` / `host::NetCapability`,
    // so there is no `FetchBlob` round-trip through this loop.
}

/// Events received from the network thread.
#[derive(Debug)]
pub enum NetEvent {
    /// Strictly verified immutable collection evidence learned via gossip.
    ///
    /// Deliberately carries no transport publisher: the relaying neighbor is
    /// not necessarily the author, and author identity is already signed into
    /// the record. Admission remains a synchronous store-side policy.
    CollectionEvidence(CollectionCommit),
}
