//! Collection-scoped anti-entropy for triblespace.
//!
//! [`Peer<S>`](peer::Peer) wraps one store. Periodic per-request authorized
//! PATCH walks converge one explicitly active collection's records and
//! portable WRITE evidence. A separate stock-gossip
//! wake plane carries only a signed endpoint origin and opaque per-collection
//! anti-entropy root; knowing the collection handle is its discovery
//! capability, while every useful byte remains capability-gated. Exact content
//! reads retain durable-WANT
//! semantics and use authenticated DHT provider lookup independently of broad
//! inventory mirroring. [`collection_sync`] lets the core collection
//! resolver select an exact physical cover from speculative remote artifacts,
//! so callers can fetch only useful materializations without creating WANTs.
//! A bounded, global opaque provider directory can locate peers for an
//! already-known immutable artifact handle. Each selected artifact publishes
//! one exact full-width derived-key lease at its K closest DHT nodes; directory
//! requests never carry the bearer handle, and unrelated keys do not collapse
//! into fixed prefix hotspots. Publication policy remains separate because a
//! derived key alone cannot protect low-entropy plaintext from dictionary
//! confirmation.
//!
//! All store traits stay sync. Async is jailed inside the network thread.

mod channel;
pub mod collection_activation;
pub mod collection_delta;
pub(crate) mod collection_session;
pub(crate) mod collection_wire;

/// Base backoff for failed WANT fulfillment in [`reconcile::Reconciler`];
/// doubles per attempt up to
/// [`RETRY_BACKOFF_CAP`]. Values chosen so a transient fault (peer
/// restarting, partition healing) is retried promptly while a
/// persistently-dead source costs at most one attempt per cap period.
pub(crate) const RETRY_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_secs(1);
/// Upper bound the exponential retry backoff saturates at.
pub(crate) const RETRY_BACKOFF_CAP: std::time::Duration = std::time::Duration::from_secs(60);
pub mod clock;
pub mod collection_sync;
pub mod host;
pub mod identity;
pub mod inventory;
pub mod patch_repair;
pub mod peer;
pub mod protocol;
pub mod provider;
pub mod reconcile;
pub(crate) mod routing;
pub mod transport;
pub mod wake;
