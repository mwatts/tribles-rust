//! Distributed sync protocol for triblespace piles.
//!
//! Provides a blob-centric sync protocol over iroh QUIC connections.
//! The protocol has three operations:
//!
//! - **LIST**: enumerate branches (id + head hash)
//! - **GET_BLOB**: fetch a blob by hash
//! - **SYNC**: batch reference diff (client sends HAVE set, server sends complement)
//!
//! The protocol is generic over storage: both the server and the remote
//! client implement triblespace's `BlobStoreGet` + `BranchStore` traits,
//! so any storage backend (Pile, S3, in-memory) can serve or consume.

pub mod protocol;
pub mod remote;
pub mod server;
pub mod sync;
pub mod identity;

pub use triblespace_core::repo::pile::Pile;
pub use triblespace_core::value::schemas::hash::Blake3;
