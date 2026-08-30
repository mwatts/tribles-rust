//! Authorized team-inventory synchronization for triblespace.
//!
//! [`Peer<S>`](peer::Peer) wraps one single-team store. Periodic authenticated
//! PATCH walks converge peer-routing evidence, collection records, complete
//! capability proofs, and optionally resident blobs. A bounded rotating
//! scheduler makes those pairwise set unions the epidemic exchange itself;
//! there is no broadcast wake plane. Exact content reads retain durable-WANT
//! semantics and use authenticated DHT provider lookup independently of broad
//! inventory mirroring. [`collection_sync`] lets the core collection
//! resolver select an exact physical cover from speculative remote artifacts,
//! so callers can fetch only useful materializations without creating WANTs.
//! A bounded, global opaque provider directory can locate peers for an
//! already-known immutable artifact handle. Providers publish at most 256
//! canonical prefix-cover leases rather than one lease per artifact; exact
//! membership checks remain bounded soft routing state, not an enumeration or
//! artifact-discovery plane.
//!
//! All store traits stay sync. Async is jailed inside the network thread.

mod channel;

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
pub(crate) mod inventory_reconcile;
pub(crate) mod inventory_wire;
pub mod peer;
pub mod protocol;
pub mod provider;
pub mod reconcile;
pub(crate) mod routing;
pub mod transport;
