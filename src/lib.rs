//! Distributed sync for triblespace.
//!
//! Three composable types, one async boundary:
//!
//! - **Host**: network thread (identity, endpoint, gossip, DHT)
//! - **Leader\<S\>**: store wrapper, outgoing effects (announce, gossip)
//! - **Follower\<S\>**: store wrapper, incoming sync (poll-driven)
//!
//! `Follower<Leader<Pile>>` = peer.
//!
//! All store traits (`BlobStore`, `BranchStore`) stay sync.
//! Async is jailed inside the Host thread.

pub mod channel;
pub mod host;
pub mod leader;
pub mod follower;
pub mod protocol;
pub mod identity;
pub mod tracking;

