//! Blob-centric sync: BFS over the CAS graph, fetch missing blobs, merge.
//!
//! All operations use triblespace traits — no Pile dependency.
//! Branch identification is by ID. Name resolution is a separate step.

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
    // Remote: resolve name → id.
    let (remote_id, _head) = resolve_branch_name(conn, branch_name).await?
        .ok_or_else(|| anyhow!("branch '{branch_name}' not found on remote"))?;

    // Local: ensure branch exists, get its id.
    let mut repo = Repository::new(local, signing_key.clone(), TribleSet::new())
        .map_err(|e| anyhow!("repo: {e:?}"))?;
    let local_id = repo.ensure_branch(branch_name, None)
        .map_err(|_| anyhow!("ensure branch failed"))?;
    let local = repo.into_storage();

    sync_branch(conn, local, signing_key, remote_id, local_id).await
}

/// Resolve a branch name to its ID by fetching metadata from a remote.
///
/// Lists branches, fetches each branch's metadata blob, reads the name,
/// and returns the first match.
pub async fn resolve_branch_name(
    conn: &Connection,
    name: &str,
) -> Result<Option<(Id, RawHash)>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;

    send_u8(&mut send, REQ_LIST).await?;
    let mut branches: Vec<(Id, RawHash)> = Vec::new();
    loop {
        let rsp = recv_u8(&mut recv).await?;
        match rsp {
            RSP_LIST_ENTRY => {
                let id_bytes = recv_branch_id(&mut recv).await?;
                let head = recv_hash(&mut recv).await?;
                if let Some(id) = Id::new(id_bytes) {
                    branches.push((id, head));
                }
            }
            RSP_END_LIST => break,
            _ => return Err(anyhow!("unexpected list response: {rsp}")),
        }
    }

    for (id, head) in &branches {
        send_u8(&mut send, REQ_GET_BLOB).await?;
        send_hash(&mut send, head).await?;
        let rsp = recv_u8(&mut recv).await?;
        if rsp == RSP_BLOB {
            let (_hash, data) = recv_blob_data(&mut recv).await?;
            let blob: triblespace_core::blob::Blob<SimpleArchive> =
                triblespace_core::blob::Blob::new(data.into());
            if let Ok(meta) = <TribleSet as triblespace_core::blob::TryFromBlob<SimpleArchive>>::try_from_blob(blob) {
                use triblespace_core::macros::{find, pattern};
                for name_handle in find!(
                    h: Value<Handle<Blake3, triblespace_core::blob::schemas::longstring::LongString>>,
                    pattern!(&meta, [{ _?e @ triblespace_core::metadata::name: ?h }])
                ) {
                    send_u8(&mut send, REQ_GET_BLOB).await?;
                    send_hash(&mut send, &name_handle.raw).await?;
                    let rsp2 = recv_u8(&mut recv).await?;
                    if rsp2 == RSP_BLOB {
                        let (_h2, name_data) = recv_blob_data(&mut recv).await?;
                        if let Ok(n) = std::str::from_utf8(&name_data) {
                            if n == name {
                                send_u8(&mut send, REQ_DONE).await?;
                                send.finish().map_err(|e| anyhow!("finish: {e}"))?;
                                return Ok(Some((*id, *head)));
                            }
                        }
                    }
                }
            }
        }
    }

    send_u8(&mut send, REQ_DONE).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;
    Ok(None)
}

/// List remote branches: returns (branch_id, head_hash) pairs.
pub async fn list_branches(conn: &Connection) -> Result<Vec<(Id, RawHash)>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, REQ_LIST).await?;

    let mut branches = Vec::new();
    loop {
        let rsp = recv_u8(&mut recv).await?;
        match rsp {
            RSP_LIST_ENTRY => {
                let id_bytes = recv_branch_id(&mut recv).await?;
                let head = recv_hash(&mut recv).await?;
                if let Some(id) = Id::new(id_bytes) {
                    branches.push((id, head));
                }
            }
            RSP_END_LIST => break,
            _ => return Err(anyhow!("unexpected list response: {rsp}")),
        }
    }
    send_u8(&mut send, REQ_DONE).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;
    Ok(branches)
}

/// Query a remote peer's head for a specific branch.
pub async fn remote_head(conn: &Connection, branch_id: Id) -> Result<Option<RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, REQ_HEAD).await?;
    let id_bytes: [u8; 16] = branch_id.into();
    send_branch_id(&mut send, &id_bytes).await?;

    let rsp = recv_u8(&mut recv).await?;
    let result = match rsp {
        RSP_HEAD_OK => Some(recv_hash(&mut recv).await?),
        RSP_NONE => None,
        _ => return Err(anyhow!("unexpected head response: {rsp}")),
    };

    send_u8(&mut send, REQ_DONE).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;
    Ok(result)
}

/// Sync a branch by ID, consuming the store and returning it.
///
/// `remote_branch_id`: which branch to read from the remote.
/// `local_branch_id`: which branch to merge into locally.
///
/// These may be different IDs — branch IDs are local to each pile.
/// The caller resolves names to IDs on each side before calling this.
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
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;

    // Get remote head for this branch.
    send_u8(&mut send, REQ_HEAD).await?;
    let id_bytes: [u8; 16] = remote_branch_id.into();
    send_branch_id(&mut send, &id_bytes).await?;

    let rsp = recv_u8(&mut recv).await?;
    let remote_head = match rsp {
        RSP_HEAD_OK => recv_hash(&mut recv).await?,
        RSP_NONE => {
            send_u8(&mut send, REQ_DONE).await?;
            send.finish().map_err(|e| anyhow!("finish: {e}"))?;
            return Ok((local, SyncResult { blobs_fetched: 0, bytes_fetched: 0 }));
        }
        _ => return Err(anyhow!("unexpected head response: {rsp}")),
    };

    // Already up to date?
    if has_blob(&mut local, &remote_head) {
        send_u8(&mut send, REQ_DONE).await?;
        send.finish().map_err(|e| anyhow!("finish: {e}"))?;
        return Ok((local, SyncResult { blobs_fetched: 0, bytes_fetched: 0 }));
    }

    let mut fetched = 0usize;
    let mut fetched_bytes = 0usize;

    // Fetch head blob.
    send_u8(&mut send, REQ_GET_BLOB).await?;
    send_hash(&mut send, &remote_head).await?;
    let rsp = recv_u8(&mut recv).await?;
    if rsp == RSP_BLOB {
        let (_hash, data) = recv_blob_data(&mut recv).await?;
        fetched += 1;
        fetched_bytes += data.len();
        put_blob(&mut local, data)?;
    }

    // BFS level by level with SYNC batches.
    let mut current_level = vec![remote_head];
    let mut seen: HashSet<RawHash> = HashSet::new();
    seen.insert(remote_head);

    while !current_level.is_empty() {
        let mut next_level: Vec<RawHash> = Vec::new();

        for parent_hash in &current_level {
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

            send_u8(&mut send, REQ_CHILDREN).await?;
            send_hash(&mut send, parent_hash).await?;
            send_u32_be(&mut send, have.len() as u32).await?;
            for h in &have {
                send_hash(&mut send, h).await?;
            }

            loop {
                let rsp = recv_u8(&mut recv).await?;
                match rsp {
                    RSP_BLOB => {
                        let (hash, data) = recv_blob_data(&mut recv).await?;
                        fetched += 1;
                        fetched_bytes += data.len();
                        put_blob(&mut local, data)?;
                        next_level.push(hash);
                    }
                    RSP_END_CHILDREN => break,
                    _ => return Err(anyhow!("unexpected sync response: {rsp}")),
                }
            }
        }
        current_level = next_level;
    }

    // Merge: read the remote branch metadata to find the actual commit hash.
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

    send_u8(&mut send, REQ_DONE).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

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
