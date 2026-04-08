//! Blob-centric sync: BFS over the CAS graph, fetch missing blobs, merge.
//!
//! All operations use triblespace traits — no Pile dependency.
//! Branch identification is by ID. Name resolution is a separate step.
//! Each protocol operation gets its own QUIC stream — no multiplexing.

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

const VALUE_LEN: usize = 32;

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

/// Resolve a branch name and sync in one call.
///
/// Resolves the name on both sides:
/// - Remote: LIST + metadata fetch → remote branch ID
/// - Local: `ensure_branch` → local branch ID (creates if missing)
///
/// Then syncs using the resolved IDs.
pub async fn resolve_and_sync<S>(
    conn: &Connection,
    local: S,
    signing_key: &ed25519_dalek::SigningKey,
    branch_name: &str,
) -> Result<(S, SyncResult)>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    let (remote_id, _head) = resolve_branch_name(conn, branch_name).await?
        .ok_or_else(|| anyhow!("branch '{branch_name}' not found on remote"))?;

    let mut repo = Repository::new(local, signing_key.clone(), TribleSet::new())
        .map_err(|e| anyhow!("repo: {e:?}"))?;
    let local_id = repo.ensure_branch(branch_name, None)
        .map_err(|_| anyhow!("ensure branch failed"))?;
    let local = repo.into_storage();

    sync_branch(conn, local, signing_key, remote_id, local_id).await
}

/// Resolve a branch name to its ID by fetching metadata from a remote.
pub async fn resolve_branch_name(
    conn: &Connection,
    name: &str,
) -> Result<Option<(Id, RawHash)>> {
    let branches = op_list(conn).await?;

    for (id_bytes, head) in &branches {
        let Some(id) = Id::new(*id_bytes) else { continue; };

        // Fetch branch metadata blob.
        let Some(meta_data) = op_get_blob(conn, head).await? else { continue; };
        let blob: triblespace_core::blob::Blob<SimpleArchive> =
            triblespace_core::blob::Blob::new(meta_data.into());
        let Ok(meta) = <TribleSet as triblespace_core::blob::TryFromBlob<SimpleArchive>>::try_from_blob(blob) else { continue; };

        use triblespace_core::macros::{find, pattern};
        for name_handle in find!(
            h: Value<Handle<Blake3, triblespace_core::blob::schemas::longstring::LongString>>,
            pattern!(&meta, [{ _?e @ triblespace_core::metadata::name: ?h }])
        ) {
            // Fetch the name string blob.
            let Some(name_data) = op_get_blob(conn, &name_handle.raw).await? else { continue; };
            if let Ok(n) = std::str::from_utf8(&name_data) {
                if n == name {
                    return Ok(Some((id, *head)));
                }
            }
        }
    }
    Ok(None)
}

/// List remote branches: returns (branch_id, head_hash) pairs.
pub async fn list_branches(conn: &Connection) -> Result<Vec<(Id, RawHash)>> {
    let raw = op_list(conn).await?;
    Ok(raw.into_iter()
        .filter_map(|(id_bytes, head)| Id::new(id_bytes).map(|id| (id, head)))
        .collect())
}

/// Query a remote peer's head for a specific branch.
pub async fn remote_head(conn: &Connection, branch_id: Id) -> Result<Option<RawHash>> {
    let id_bytes: [u8; 16] = branch_id.into();
    op_head(conn, &id_bytes).await
}

/// Sync a branch by ID, consuming the store and returning it.
///
/// `remote_branch_id`: which branch to read from the remote.
/// `local_branch_id`: which branch to merge into locally.
///
/// Each protocol operation gets its own QUIC stream.
pub async fn sync_branch<S>(
    conn: &Connection,
    mut local: S,
    signing_key: &ed25519_dalek::SigningKey,
    remote_branch_id: Id,
    local_branch_id: Id,
) -> Result<(S, SyncResult)>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    // Get remote head.
    let id_bytes: [u8; 16] = remote_branch_id.into();
    let remote_head = match op_head(conn, &id_bytes).await? {
        Some(h) => h,
        None => return Ok((local, SyncResult { blobs_fetched: 0, bytes_fetched: 0 })),
    };

    // Already up to date?
    if has_blob(&mut local, &remote_head) {
        return Ok((local, SyncResult { blobs_fetched: 0, bytes_fetched: 0 }));
    }

    let mut fetched = 0usize;
    let mut fetched_bytes = 0usize;

    // Fetch head blob.
    if let Some(data) = op_get_blob(conn, &remote_head).await? {
        fetched += 1;
        fetched_bytes += data.len();
        put_blob(&mut local, data)?;
    }

    // BFS level by level using CHILDREN.
    let mut current_level = vec![remote_head];
    let mut seen: HashSet<RawHash> = HashSet::new();
    seen.insert(remote_head);

    while !current_level.is_empty() {
        let mut next_level: Vec<RawHash> = Vec::new();

        for parent_hash in &current_level {
            // Build HAVE set: references we already have locally.
            let mut have: Vec<RawHash> = Vec::new();
            if let Some(data) = get_blob(&mut local, parent_hash) {
                for chunk in data.chunks(VALUE_LEN) {
                    if chunk.len() == VALUE_LEN {
                        let mut candidate = [0u8; 32];
                        candidate.copy_from_slice(chunk);
                        if seen.insert(candidate) && has_blob(&mut local, &candidate) {
                            have.push(candidate);
                        }
                    }
                }
            }

            // CHILDREN: get missing child hashes, then fetch each blob.
            let child_hashes = op_children(conn, parent_hash, &have).await?;
            for hash in child_hashes {
                if let Some(data) = op_get_blob(conn, &hash).await? {
                    fetched += 1;
                    fetched_bytes += data.len();
                    put_blob(&mut local, data)?;
                    next_level.push(hash);
                }
            }
        }
        current_level = next_level;
    }

    // Merge: extract commit hash from branch metadata, adopt into local branch.
    let branch_meta_handle = Value::<Handle<Blake3, SimpleArchive>>::new(remote_head);
    {
        let reader = local.reader().map_err(|e| anyhow!("reader: {e:?}"))?;
        let branch_meta: TribleSet = reader.get(branch_meta_handle)
            .map_err(|e| anyhow!("read branch meta: {e:?}"))?;
        use triblespace_core::macros::{find, pattern};
        let commit_handle: Value<Handle<Blake3, SimpleArchive>> = find!(
            h: Value<Handle<Blake3, SimpleArchive>>,
            pattern!(&branch_meta, [{ _?e @ triblespace_core::repo::head: ?h }])
        ).next().ok_or_else(|| anyhow!("no head commit in remote branch metadata"))?;

        let mut repo = Repository::new(local, signing_key.clone(), TribleSet::new())
            .map_err(|e| anyhow!("repo: {e:?}"))?;
        let mut ws = repo.pull(local_branch_id).map_err(|e| anyhow!("pull: {e:?}"))?;
        ws.merge_commit(commit_handle).map_err(|e| anyhow!("merge: {e:?}"))?;
        repo.push(&mut ws).map_err(|_| anyhow!("push failed"))?;
        local = repo.into_storage();
    }

    Ok((local, SyncResult { blobs_fetched: fetched, bytes_fetched: fetched_bytes }))
}

fn has_blob<S: BlobStore<Blake3>>(store: &mut S, hash: &RawHash) -> bool {
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let Ok(reader) = store.reader() else { return false; };
    reader.get::<Bytes, UnknownBlob>(handle).is_ok()
}

fn get_blob<S: BlobStore<Blake3>>(store: &mut S, hash: &RawHash) -> Option<Vec<u8>> {
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let reader = store.reader().ok()?;
    reader.get::<Bytes, UnknownBlob>(handle).ok().map(|b| b.to_vec())
}

fn put_blob<S: BlobStorePut<Blake3>>(store: &mut S, data: Vec<u8>) -> Result<()> {
    let bytes: Bytes = data.into();
    let _: Value<Handle<Blake3, UnknownBlob>> = store.put::<UnknownBlob, Bytes>(bytes)
        .map_err(|e| anyhow!("put: {e:?}"))?;
    Ok(())
}
