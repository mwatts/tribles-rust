//! Protocol handler: dispatches one operation per incoming QUIC stream.
//!
//! Generic over `BlobStore + BranchStore` — any storage backend works.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BranchStore};
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::protocol::*;

const VALUE_LEN: usize = 32;

/// Handle one incoming stream: read the operation byte, dispatch.
pub async fn handle_stream<S>(
    store: &mut S,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<()>
where
    S: BlobStore<Blake3> + BranchStore<Blake3>,
{
    let op = recv_u8(recv).await?;

    match op {
        OP_LIST => {
            let branch_ids: Vec<Id> = store.branches()
                .map_err(|e| anyhow!("branches: {e:?}"))?
                .filter_map(|r| r.ok())
                .collect();
            for id in branch_ids {
                if let Ok(Some(head)) = store.head(id) {
                    let id_bytes: [u8; 16] = id.into();
                    send_u8(send, RSP_BLOB).await?; // reuse as "entry present" marker
                    send_branch_id(send, &id_bytes).await?;
                    send_hash(send, &head.raw).await?;
                }
            }
            send_u8(send, RSP_END).await?;
        }

        OP_HEAD => {
            let id_bytes = recv_branch_id(recv).await?;
            let Some(branch_id) = Id::new(id_bytes) else {
                send_u8(send, RSP_NONE).await?;
                return Ok(());
            };
            match store.head(branch_id) {
                Ok(Some(head)) => {
                    send_u8(send, RSP_HEAD_OK).await?;
                    send_hash(send, &head.raw).await?;
                }
                _ => send_u8(send, RSP_NONE).await?,
            }
        }

        OP_GET_BLOB => {
            let hash = recv_hash(recv).await?;
            match get_blob(store, &hash) {
                Some(data) => {
                    send_u8(send, RSP_BLOB).await?;
                    send_u32_be(send, data.len() as u32).await?;
                    send.write_all(&data).await.map_err(|e| anyhow!("send: {e}"))?;
                }
                None => send_u8(send, RSP_MISSING).await?,
            }
        }

        OP_CHILDREN => {
            let parent_hash = recv_hash(recv).await?;
            let have_count = recv_u32_be(recv).await? as usize;
            let mut have_set: HashSet<RawHash> = HashSet::with_capacity(have_count);
            for _ in 0..have_count {
                have_set.insert(recv_hash(recv).await?);
            }
            // Stream child hashes, nil hash as sentinel.
            if let Some(parent_data) = get_blob(store, &parent_hash) {
                for chunk in parent_data.chunks(VALUE_LEN) {
                    if chunk.len() == VALUE_LEN {
                        let mut candidate = [0u8; 32];
                        candidate.copy_from_slice(chunk);
                        if !have_set.contains(&candidate) && has_blob(store, &candidate) {
                            send_hash(send, &candidate).await?;
                        }
                    }
                }
            }
            send_hash(send, &[0u8; 32]).await?;
        }

        _ => {} // Unknown op — ignore, stream closes.
    }
    Ok(())
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
