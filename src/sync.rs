//! Blob-centric sync: BFS over the CAS graph, fetch missing blobs, merge.

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use iroh::endpoint::Connection;
use triblespace_core::blob::TryFromBlob;
use triblespace_core::blob::Blob;
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut, BranchStore, Repository};
use triblespace_core::repo::pile::Pile;
use triblespace_core::trible::TribleSet;
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};
use ed25519_dalek::SigningKey;

use crate::protocol::*;

const VALUE_LEN: usize = 32;

/// Result of a sync operation.
pub struct SyncResult {
    pub blobs_fetched: usize,
    pub bytes_fetched: usize,
    pub branch_name: String,
}

impl std::fmt::Display for SyncResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "'{}': {} blobs ({}B), merged",
            self.branch_name, self.blobs_fetched, self.bytes_fetched)
    }
}

/// Sync a branch from a remote peer into a local pile.
pub async fn sync_branch(
    conn: &Connection,
    pile_path: &PathBuf,
    signing_key: &SigningKey,
    branch_name: &str,
) -> Result<SyncResult> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;

    // LIST remote branches, match by name via metadata.
    send_u8(&mut send, REQ_LIST).await?;
    let mut remote_branches: Vec<([u8; 16], RawHash)> = Vec::new();
    loop {
        let rsp = recv_u8(&mut recv).await?;
        match rsp {
            RSP_LIST_ENTRY => {
                let id = recv_branch_id(&mut recv).await?;
                let head = recv_hash(&mut recv).await?;
                remote_branches.push((id, head));
            }
            RSP_END_LIST => break,
            _ => return Err(anyhow!("unexpected list response: {rsp}")),
        }
    }

    // Resolve branch name by fetching metadata blobs.
    let mut remote_head: Option<RawHash> = None;
    for (_id, head) in &remote_branches {
        send_u8(&mut send, REQ_GET_BLOB).await?;
        send_hash(&mut send, head).await?;
        let rsp = recv_u8(&mut recv).await?;
        if rsp == RSP_BLOB {
            let (_hash, data) = recv_blob_data(&mut recv).await?;
            let blob: Blob<SimpleArchive> = Blob::new(data.into());
            if let Ok(meta) = TribleSet::try_from_blob(blob) {
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
                        if let Ok(name) = std::str::from_utf8(&name_data) {
                            if name == branch_name {
                                remote_head = Some(*head);
                                break;
                            }
                        }
                    }
                }
            }
            if remote_head.is_some() { break; }
        }
    }
    let remote_head = match remote_head {
        Some(h) => h,
        None => return Ok(SyncResult {
            blobs_fetched: 0, bytes_fetched: 0,
            branch_name: format!("{branch_name}: not found"),
        }),
    };

    // BFS: fetch head blob if missing.
    let mut pile = crate::open_pile(pile_path)?;
    let mut fetched = 0usize;
    let mut fetched_bytes = 0usize;

    if !has_blob(&mut pile, &remote_head) {
        send_u8(&mut send, REQ_GET_BLOB).await?;
        send_hash(&mut send, &remote_head).await?;
        let rsp = recv_u8(&mut recv).await?;
        if rsp == RSP_BLOB {
            let (_hash, data) = recv_blob_data(&mut recv).await?;
            fetched += 1;
            fetched_bytes += data.len();
            let bytes: Bytes = data.into();
            let _: Value<Handle<Blake3, UnknownBlob>> = pile.put::<UnknownBlob, Bytes>(bytes)
                .map_err(|e| anyhow!("put: {e:?}"))?;
        }
    }

    // BFS level by level with SYNC batches.
    let mut current_level = vec![remote_head];
    let mut seen: HashSet<RawHash> = HashSet::new();
    seen.insert(remote_head);

    while !current_level.is_empty() {
        let mut next_level: Vec<RawHash> = Vec::new();

        for parent_hash in &current_level {
            let mut have: Vec<RawHash> = Vec::new();
            if let Some(data) = get_blob(&mut pile, parent_hash) {
                for chunk in data.chunks(VALUE_LEN) {
                    if chunk.len() == VALUE_LEN {
                        let mut candidate = [0u8; 32];
                        candidate.copy_from_slice(chunk);
                        if seen.insert(candidate) && has_blob(&mut pile, &candidate) {
                            have.push(candidate);
                        }
                    }
                }
            }

            send_u8(&mut send, REQ_SYNC).await?;
            send_hash(&mut send, parent_hash).await?;
            send_u32_be(&mut send, have.len() as u32).await?;
            for h in &have { send_hash(&mut send, h).await?; }

            loop {
                let rsp = recv_u8(&mut recv).await?;
                match rsp {
                    RSP_BLOB => {
                        let (hash, data) = recv_blob_data(&mut recv).await?;
                        fetched += 1;
                        fetched_bytes += data.len();
                        let bytes: Bytes = data.into();
                        let _: Value<Handle<Blake3, UnknownBlob>> = pile.put::<UnknownBlob, Bytes>(bytes)
                            .map_err(|e| anyhow!("put: {e:?}"))?;
                        next_level.push(hash);
                    }
                    RSP_END_SYNC => break,
                    _ => return Err(anyhow!("unexpected: {rsp}")),
                }
            }
        }
        current_level = next_level;
    }

    // CAS merge.
    let remote_branch_meta = Value::<Handle<Blake3, SimpleArchive>>::new(remote_head);
    let reader = BlobStore::<Blake3>::reader(&mut pile).map_err(|e| anyhow!("reader: {e:?}"))?;
    let branch_meta: TribleSet = reader.get(remote_branch_meta)
        .map_err(|e| anyhow!("read meta: {e:?}"))?;

    use triblespace_core::macros::{find, pattern};
    let remote_commit: Value<Handle<Blake3, SimpleArchive>> = find!(
        h: Value<Handle<Blake3, SimpleArchive>>,
        pattern!(&branch_meta, [{ _?e @ triblespace_core::repo::head: ?h }])
    ).next().ok_or_else(|| anyhow!("no head in branch meta"))?;
    drop(reader);

    let mut repo = Repository::new(pile, signing_key.clone(), TribleSet::new())
        .map_err(|e| anyhow!("repo: {e:?}"))?;
    let branch_id = repo.ensure_branch(branch_name, None)
        .map_err(|e| anyhow!("ensure branch: {e:?}"))?;
    let mut ws = repo.pull(branch_id).map_err(|e| anyhow!("pull: {e:?}"))?;
    ws.merge_commit(remote_commit).map_err(|e| anyhow!("merge: {e:?}"))?;
    repo.push(&mut ws).map_err(|e| anyhow!("push: {e:?}"))?;
    let _ = repo.close();

    send_u8(&mut send, REQ_DONE).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    Ok(SyncResult { blobs_fetched: fetched, bytes_fetched: fetched_bytes, branch_name: branch_name.to_string() })
}

fn has_blob(pile: &mut Pile<Blake3>, hash: &RawHash) -> bool {
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let Ok(reader) = BlobStore::<Blake3>::reader(pile) else { return false; };
    reader.get::<Bytes, UnknownBlob>(handle).is_ok()
}

fn get_blob(pile: &mut Pile<Blake3>, hash: &RawHash) -> Option<Vec<u8>> {
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let reader = BlobStore::<Blake3>::reader(pile).ok()?;
    reader.get::<Bytes, UnknownBlob>(handle).ok().map(|b| b.to_vec())
}
