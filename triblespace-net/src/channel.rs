//! Channel types bridging the async network thread and the sync store layer.
//!
//! `NetCommand`: outgoing effects sent from a [`Peer`](crate::peer::Peer)
//! into the network thread. All fire-and-forget — there are no RPC
//! variants because collection discovery is gossip-driven, not
//! peer-targeted.
//! `NetEvent`: incoming data sent back from the network thread to be
//! applied into the wrapped store.
//!
//! Byte payloads use [`anybytes::Bytes`] rather than `Vec<u8>`:
//! Bytes is Arc-refcounted, so cloning across the channel boundary
//! is a refcount bump instead of a full byte-copy. The same payload
//! can flow into multiple onward sinks (wire write + local store)
//! without re-materialising the buffer.

use anybytes::Bytes;
use std::sync::mpsc;
use triblespace_core::collection::CollectionHandle;

use crate::protocol::RawHash;
use crate::transport::PeerId;
use triblespace_core::collection::CollectionCommit;

/// A 32-byte public key identifying a publisher.
pub type PublisherKey = [u8; 32];

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
    /// Gossip one strictly verified author grant + collection commit pair.
    ///
    /// This is immutable ledger evidence, not an admission decision. The
    /// receiving side must not treat the mesh carrier as the commit author or
    /// infer local author trust from transport delivery.
    GossipCollectionEvidence { evidence: CollectionCommit },
    /// Dispatch a freshly-signed cap+sig pair to `subject` via the
    /// auth-handshake ALPN. Used by the renewal daemon (push-based
    /// renewal) and by the `team approve` subcommand (response to a
    /// pending request). The network thread opens a connection to
    /// the subject's pubkey, sends `OP_DELIVER_CAP`, and closes.
    ///
    /// Delivery is best-effort fire-and-forget at this layer.
    /// Confirmation happens later, when the subject actually
    /// authenticates against our pile-sync ALPN presenting the
    /// delivered cap — see `NetEvent::CapDeliveryConfirmed`. The
    /// renewal daemon redispatches entries that haven't been
    /// confirmed yet (per-entry cooldown to avoid hammering an
    /// unreachable peer).
    DeliverCap {
        subject: PublisherKey,
        cap_bytes: Bytes,
        sig_bytes: Bytes,
    },
    /// Fetch one exact collection's grant-backed sparse evidence from a
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
    /// A blob was fetched from the network.
    Blob(Bytes),
    /// Strictly verified immutable collection evidence learned via gossip.
    ///
    /// Deliberately carries no transport publisher: the relaying neighbor is
    /// not necessarily the author, and author identity is already signed into
    /// both records. Admission remains a synchronous store-side policy.
    CollectionEvidence(CollectionCommit),
    /// A peer asked us to issue them a capability. The partial cap
    /// blob carries the subject they're requesting for (must match
    /// `requester` — verified at connection time via iroh's TLS),
    /// the scope they're asking for, and their preferred expiry
    /// interval. The local policy collection decides whether
    /// to auto-approve, queue for human review, or reject.
    CapRequest {
        requester: PublisherKey,
        partial_cap_bytes: Bytes,
    },
    /// A peer issued us a capability — either in response to a prior
    /// `CapRequest` we made, or as an unsolicited renewal push. The
    /// cap+sig bytes are content-verified before being recorded in the
    /// local policy collection.
    CapDelivered {
        issuer: PublisherKey,
        cap_bytes: Bytes,
        sig_bytes: Bytes,
    },
    /// `subject` successfully authenticated against our pile-sync
    /// `OP_AUTH` stream by presenting signature handle `sig_handle`.
    /// This is the unambiguous "the subject has the cap and uses
    /// it" signal — the wire-level STATUS_OK on `OP_DELIVER_CAP`
    /// only tells us the bytes landed; auth tells us the subject
    /// can both load AND verify the chain. The Peer side uses this
    /// to mark the matching renewal-policy entry as delivered so
    /// the daemon's next tick skips it from the redispatch set.
    ///
    /// Field is the *signature* handle, not the cap blob handle —
    /// OP_AUTH wires the sig blob since that's the credential the
    /// dialer needs to prove possession of. Match against
    /// `PolicyEntry::latest_sig` (not `latest_cap`) when looking up
    /// the corresponding renewal-policy entry.
    CapDeliveryConfirmed {
        subject: PublisherKey,
        sig_handle: RawHash,
    },
}
