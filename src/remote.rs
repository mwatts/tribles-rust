//! Remote blob store backed by a network connection.
//!
//! `RemoteStore` implements read-only blob access over the sync protocol,
//! making a remote pile look like a local `BlobStoreGet`.

// TODO: implement BlobStoreGet<Blake3> for RemoteStore
// This requires bridging async (network I/O) with the sync trait.
// Options:
// 1. Hold a tokio runtime handle + connection, block_on each get()
// 2. Pre-fetch blobs in batch, serve from local cache
// 3. Define async versions of the traits in this crate
//
// For now, the sync logic in sync.rs uses the raw protocol directly
// rather than going through traits. The trait wrapper comes later
// when the API stabilizes.
