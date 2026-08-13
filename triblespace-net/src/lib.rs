//! Distributed sync for triblespace.
//!
//! The main type is [`Peer<S>`](peer::Peer): a store wrapper that owns an
//! iroh network thread internally and exposes the standard storage traits.
//! Reads auto-drain immutable collection evidence; writes announce blobs to
//! the DHT and publish grant-backed commits to the team gossip topic. Local
//! pins remain only as one input to capability-scoped blob serving.
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
pub mod collection_wire;
pub mod dht;
pub mod handshake;
pub mod host;
pub mod identity;
pub mod peer;
pub mod policy;
pub mod protocol;
pub mod reconcile;
pub mod transport;
