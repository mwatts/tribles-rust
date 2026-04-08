//! `Follower<S>`: keeps a local store in sync with a remote peer.
//!
//! Wraps a local store, pulls blobs from a remote connection.
//! Implements `BlobStore`, `BlobStorePut`, and `BranchStore` by
//! delegating to the inner store — Follower IS a store.
//!
//! Branch management and merging are NOT the Follower's job.
//! Use `Repository<Follower<Pile>>` for that — regular merge on
//! a store that happens to sync from a remote.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use iroh::endpoint::Connection;
use triblespace_core::blob::{BlobSchema, ToBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut, BranchStore, PushResult};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::protocol::*;

/// Keeps a local store in sync with a remote peer.
///
/// Owns the store, pulls blobs into it. Delegates all trait methods
/// to the inner store, so `Repository<Follower<Pile>>` just works.
pub struct Follower<S> {
    store: S,
    conn: Connection,
}

impl<S> Follower<S> {
    pub fn new(store: S, conn: Connection) -> Self {
        Self { store, conn }
    }

    /// Access the local store.
    pub fn store(&self) -> &S { &self.store }

    /// Access the local store mutably.
    pub fn store_mut(&mut self) -> &mut S { &mut self.store }

    /// Consume the follower, return the store.
    pub fn into_store(self) -> S { self.store }
}

impl<S> Follower<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3>,
{
    /// Pull all blobs reachable from a root hash into the local store.
    /// BFS using CHILDREN for discovery, GET_BLOB for data.
    pub async fn pull_reachable(&mut self, root: &RawHash) -> Result<SyncResult> {
        let mut fetched = 0usize;
        let mut fetched_bytes = 0usize;
        let mut seen: HashSet<RawHash> = HashSet::new();

        // Fetch root if missing.
        if !self.has_blob(root) {
            if let Some(data) = op_get_blob(&self.conn, root).await? {
                fetched += 1;
                fetched_bytes += data.len();
                self.put_blob(data)?;
            }
        }
        seen.insert(*root);

        // BFS level by level.
        let mut current_level = vec![*root];
        while !current_level.is_empty() {
            let mut next_level: Vec<RawHash> = Vec::new();

            for parent in &current_level {
                let children = op_children(&self.conn, parent).await?;
                for hash in children {
                    if !seen.insert(hash) { continue; }
                    if self.has_blob(&hash) { continue; }
                    if let Some(data) = op_get_blob(&self.conn, &hash).await? {
                        fetched += 1;
                        fetched_bytes += data.len();
                        self.put_blob(data)?;
                        next_level.push(hash);
                    }
                }
            }
            current_level = next_level;
        }

        Ok(SyncResult { blobs_fetched: fetched, bytes_fetched: fetched_bytes })
    }

    fn has_blob(&mut self, hash: &RawHash) -> bool {
        let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
        let Ok(reader) = self.store.reader() else { return false; };
        reader.get::<Bytes, UnknownBlob>(handle).is_ok()
    }

    fn put_blob(&mut self, data: Vec<u8>) -> Result<()> {
        let bytes: Bytes = data.into();
        let _: Value<Handle<Blake3, UnknownBlob>> = self.store.put::<UnknownBlob, Bytes>(bytes)
            .map_err(|e| anyhow!("put: {e:?}"))?;
        Ok(())
    }
}

/// Result of a pull operation.
pub struct SyncResult {
    pub blobs_fetched: usize,
    pub bytes_fetched: usize,
}

impl std::fmt::Display for SyncResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} blobs ({}B)", self.blobs_fetched, self.bytes_fetched)
    }
}

// ── Trait delegations: Follower IS a store ───────────────────────────

impl<S> BlobStorePut<Blake3> for Follower<S>
where
    S: BlobStorePut<Blake3>,
{
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> std::result::Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        self.store.put(item)
    }
}

impl<S> BlobStore<Blake3> for Follower<S>
where
    S: BlobStore<Blake3>,
{
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> std::result::Result<Self::Reader, Self::ReaderError> {
        self.store.reader()
    }
}

impl<S> BranchStore<Blake3> for Follower<S>
where
    S: BranchStore<Blake3>,
{
    type BranchesError = S::BranchesError;
    type HeadError = S::HeadError;
    type UpdateError = S::UpdateError;
    type ListIter<'a> = S::ListIter<'a> where S: 'a;

    fn branches<'a>(&'a mut self) -> std::result::Result<Self::ListIter<'a>, Self::BranchesError> {
        self.store.branches()
    }

    fn head(&mut self, id: Id) -> std::result::Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        self.store.head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> std::result::Result<PushResult<Blake3>, Self::UpdateError> {
        self.store.update(id, old, new)
    }
}
