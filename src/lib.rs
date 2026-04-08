//! Distributed sync protocol for triblespace.
//!
//! Storage-agnostic: all operations are expressed in terms of
//! triblespace's `BlobStore`, `BlobStorePut`, and `BranchStore` traits.
//! Any backend (Pile, S3, in-memory) works as both local and remote
//! storage — the `RemoteStore` is just another `BlobStore` that happens
//! to do network I/O.
//!
//! The protocol has three operations:
//!
//! - **LIST**: enumerate branches (id + head hash)
//! - **GET_BLOB**: fetch a blob by hash
//! - **SYNC**: batch reference diff (client sends HAVE set, server sends complement)
//!
//! Names are resolved externally. The protocol only sees 16-byte branch
//! IDs and 32-byte blob hashes.

pub mod protocol;
pub mod remote;
pub mod server;
pub mod sync;
pub mod identity;
