//! Collection-scoped anti-entropy for triblespace.
//!
//! [`Peer<S>`](peer::Peer) wraps one store. Periodic per-request authorized
//! PATCH walks converge one explicitly active collection's records and
//! portable WRITE evidence. A separate stock-gossip
//! wake plane carries only a signed endpoint origin and opaque per-collection
//! anti-entropy root; knowing the collection handle is its discovery
//! capability, while every useful byte remains capability-gated. Exact content
//! reads use collection-granular provider discovery under a domain-separated
//! derived key of the collection handle. Before the requester reveals an exact
//! blob handle, the selected endpoint must prove current READ authority for
//! that collection. The handle itself then authorizes the exact bytes. Bare
//! durable blob WANTs remain local retention intent; network discovery needs
//! the collection route explicitly.
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
