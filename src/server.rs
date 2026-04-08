//! Protocol handler that serves GET_BLOB, SYNC, LIST, and HEAD
//! from any `BlobStore + BranchStore` implementation.

use std::collections::HashSet;

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BranchStore};
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::protocol::*;

const VALUE_LEN: usize = 32;

/// Serve requests on a bidirectional stream from any `BlobStore + BranchStore`.
///
/// The caller is responsible for opening the storage and accepting the
/// connection. This function handles the protocol loop.
pub async fn serve<S>(
    store: &mut S,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<()>
where
    S: BlobStore<Blake3> + BranchStore<Blake3>,
{
    loop {
        let msg_type = match recv_u8(recv).await {
            Ok(t) => t,
            Err(_) => break,
        };

        match msg_type {
            REQ_DONE => break,

            REQ_LIST => {
                let branch_ids: Vec<Id> = store.branches()
                    .map_err(|e| anyhow!("branches: {e:?}"))?
                    .filter_map(|r| r.ok())
                    .collect();
                for id in branch_ids {
                    if let Ok(Some(head)) = store.head(id) {
                        let id_bytes: [u8; 16] = id.into();
                        send_u8(send, RSP_LIST_ENTRY).await?;
                        send_branch_id(send, &id_bytes).await?;
                        send_hash(send, &head.raw).await?;
                    }
                }
                send_u8(send, RSP_END_LIST).await?;
            }

            REQ_GET_BLOB => {
                let hash = recv_hash(recv).await?;
                match get_blob(store, &hash) {
                    Some(data) => send_blob(send, &hash, &data).await?,
                    None => send_u8(send, RSP_MISSING).await?,
                }
            }

            REQ_SYNC => {
                let parent_hash = recv_hash(recv).await?;
                let have_count = recv_u32_be(recv).await? as usize;
                let mut have_set: HashSet<RawHash> = HashSet::with_capacity(have_count);
                for _ in 0..have_count {
                    have_set.insert(recv_hash(recv).await?);
                }
                if let Some(parent_data) = get_blob(store, &parent_hash) {
                    for chunk in parent_data.chunks(VALUE_LEN) {
                        if chunk.len() == VALUE_LEN {
                            let mut candidate = [0u8; 32];
                            candidate.copy_from_slice(chunk);
                            if !have_set.contains(&candidate) {
                                if let Some(data) = get_blob(store, &candidate) {
                                    send_blob(send, &candidate, &data).await?;
                                }
                            }
                        }
                    }
                }
                send_u8(send, RSP_END_SYNC).await?;
            }

            REQ_HEAD => {
                let branch_id_bytes = recv_branch_id(recv).await?;
                let Some(branch_id) = Id::new(branch_id_bytes) else {
                    send_u8(send, RSP_NONE).await?;
                    continue;
                };
                match store.head(branch_id) {
                    Ok(Some(head)) => {
                        send_u8(send, RSP_HEAD_OK).await?;
                        send_hash(send, &head.raw).await?;
                    }
                    _ => send_u8(send, RSP_NONE).await?,
                }
            }

            _ => break,
        }
    }
    Ok(())
}

fn get_blob<S: BlobStore<Blake3>>(store: &mut S, hash: &RawHash) -> Option<Vec<u8>> {
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let reader = store.reader().ok()?;
    reader.get::<Bytes, UnknownBlob>(handle).ok().map(|b| b.to_vec())
}
