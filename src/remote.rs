//! Remote blob store backed by a network connection.
//!
//! `RemoteStore` implements `BlobStore + BranchStore` over the sync
//! protocol, making a remote node look like a local storage backend.
//!
//! This is the key abstraction: sync becomes "given two stores, converge"
//! regardless of whether either store is local or remote.
//!
//! ## Async bridging
//!
//! The triblespace traits are sync. The network is async. `RemoteStore`
//! bridges this by holding a tokio runtime handle and blocking on each
//! I/O operation internally. Callers see a normal sync trait impl.
//!
//! This means `RemoteStore` must NOT be used from inside an async context
//! (nested block_on panics). It's designed for the sync-over-async pattern
//! where the outermost layer is async (the CLI's Runtime::block_on) but
//! the library internals are trait-based and sync.

// TODO: implement BlobStore<Blake3> + BranchStore<Blake3> for RemoteStore
//
// struct RemoteStore {
//     runtime: tokio::runtime::Handle,
//     connection: iroh::endpoint::Connection,
// }
//
// impl BranchStore<Blake3> for RemoteStore {
//     fn branches(&mut self) -> Iter<Id> {
//         self.runtime.block_on(list_branches(&self.connection))
//     }
//     fn head(&mut self, id: Id) -> Option<Handle> {
//         self.runtime.block_on(remote_head(&self.connection, id))
//     }
// }
//
// impl BlobStore<Blake3> for RemoteStore {
//     type Reader = RemoteReader;
//     fn reader(&mut self) -> RemoteReader { ... }
// }
//
// The reader caches blob data fetched over the wire, so repeated
// reads don't re-fetch. The cache is per-reader-instance and dropped
// when the reader is dropped.
