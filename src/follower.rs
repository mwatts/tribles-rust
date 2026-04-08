//! `Follower<S>`: keeps a local store in sync with a remote peer.
//!
//! Owns the local store, actively pulls blobs on demand or on gossip.
//! The local store IS the API — read from it directly, always up to date.
//! No trait implementations on Follower itself — it's a sync driver,
//! not a store.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use iroh::endpoint::Connection;
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut, BranchStore, Repository};
use triblespace_core::trible::TribleSet;
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::protocol::*;

/// Keeps a local store in sync with a remote peer.
pub struct Follower<S> {
    store: Option<S>,  // Option for temporary ownership transfer to Repository
    conn: Connection,
    rt: tokio::runtime::Handle,
}

impl<S> Follower<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    /// Create a follower that syncs into the given store.
    pub fn new(store: S, conn: Connection, rt: tokio::runtime::Handle) -> Self {
        Self { store: Some(store), conn, rt }
    }

    /// Access the local store for reading.
    pub fn store(&self) -> &S { self.store.as_ref().unwrap() }

    /// Access the local store mutably.
    pub fn store_mut(&mut self) -> &mut S { self.store.as_mut().unwrap() }

    /// Consume the follower, return the store.
    pub fn into_store(self) -> S { self.store.unwrap() }

    /// Pull all blobs reachable from a remote branch head into the local store,
    /// then merge the branch.
    pub async fn sync_branch(
        &mut self,
        signing_key: &ed25519_dalek::SigningKey,
        remote_branch_id: Id,
        local_branch_id: Id,
    ) -> Result<SyncResult> {
        // Get remote head.
        let id_bytes: [u8; 16] = remote_branch_id.into();
        let remote_head = match op_head(&self.conn, &id_bytes).await? {
            Some(h) => h,
            None => return Ok(SyncResult { blobs_fetched: 0, bytes_fetched: 0 }),
        };

        // Pull all reachable blobs.
        let stats = self.pull_reachable(&remote_head).await?;

        // Merge: read commit hash from branch metadata, update local branch.
        let branch_meta_handle = Value::<Handle<Blake3, SimpleArchive>>::new(remote_head);
        let reader = self.store_mut().reader().map_err(|e| anyhow!("reader: {e:?}"))?;
        let branch_meta: TribleSet = reader.get(branch_meta_handle)
            .map_err(|e| anyhow!("read branch meta: {e:?}"))?;
        use triblespace_core::macros::{find, pattern};
        let commit_handle: Value<Handle<Blake3, SimpleArchive>> = find!(
            h: Value<Handle<Blake3, SimpleArchive>>,
            pattern!(&branch_meta, [{ _?e @ triblespace_core::repo::head: ?h }])
        ).next().ok_or_else(|| anyhow!("no head commit in remote branch metadata"))?;
        drop(reader);

        // Repository takes ownership — move store out, merge, put back.
        let store = self.store.take().unwrap();
        let mut repo = Repository::new(store, signing_key.clone(), TribleSet::new())
            .map_err(|e| anyhow!("repo: {e:?}"))?;
        let mut ws = repo.pull(local_branch_id).map_err(|e| anyhow!("pull: {e:?}"))?;
        ws.merge_commit(commit_handle).map_err(|e| anyhow!("merge: {e:?}"))?;
        repo.push(&mut ws).map_err(|_| anyhow!("push failed"))?;
        self.store = Some(repo.into_storage());

        Ok(stats)
    }

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
        let store = self.store_mut();
        let Ok(reader) = store.reader() else { return false; };
        reader.get::<Bytes, UnknownBlob>(handle).is_ok()
    }

    fn put_blob(&mut self, data: Vec<u8>) -> Result<()> {
        let bytes: Bytes = data.into();
        let _: Value<Handle<Blake3, UnknownBlob>> = self.store_mut().put::<UnknownBlob, Bytes>(bytes)
            .map_err(|e| anyhow!("put: {e:?}"))?;
        Ok(())
    }
}

/// Result of a sync operation.
pub struct SyncResult {
    pub blobs_fetched: usize,
    pub bytes_fetched: usize,
}

impl std::fmt::Display for SyncResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} blobs ({}B)", self.blobs_fetched, self.bytes_fetched)
    }
}
