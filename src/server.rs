//! Protocol handler that serves GET_BLOB, SYNC, LIST, and HEAD
//! from a pile file.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BranchStore};
use triblespace_core::repo::pile::Pile;
use triblespace_core::value::schemas::hash::Blake3 as Blake3Hash;
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::protocol::*;

const VALUE_LEN: usize = 32;

/// Protocol handler that serves blob requests from a pile.
#[derive(Debug, Clone)]
pub struct PileBlobServer {
    pub pile_path: PathBuf,
}

impl ProtocolHandler for PileBlobServer {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let pile_path = self.pile_path.clone();

        let result: Result<()> = async {
            let (mut send, mut recv) = connection.accept_bi().await
                .map_err(|e| anyhow!("accept_bi: {e}"))?;

            let mut pile = Pile::<Blake3>::open(&pile_path).map_err(|e| anyhow!("open: {e:?}"))?;
            serve_pile(&mut pile, &mut send, &mut recv).await?;
            send.finish().map_err(|e| anyhow!("finish: {e}"))?;
            pile.close().map_err(|e| anyhow!("close: {e:?}"))?;
            Ok(())
        }.await;

        if let Err(e) = &result {
            let peer = connection.remote_id().fmt_short();
            tracing::warn!("handler error [{peer}]: {e}");
        }
        connection.closed().await;
        Ok(())
    }
}

/// Serve requests from a pile on a bidirectional stream.
pub async fn serve_pile(
    pile: &mut Pile<Blake3>,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<()> {
    loop {
        let msg_type = match recv_u8(recv).await {
            Ok(t) => t,
            Err(_) => break,
        };

        match msg_type {
            REQ_DONE => break,
            REQ_LIST => {
                let iter = pile.branches().map_err(|e| anyhow!("branches: {e:?}"))?;
                for branch_result in iter {
                    let id = branch_result.map_err(|e| anyhow!("iter: {e:?}"))?;
                    if let Ok(Some(head)) = pile.head(id) {
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
                match get_blob(pile, &hash) {
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
                if let Some(parent_data) = get_blob(pile, &parent_hash) {
                    for chunk in parent_data.chunks(VALUE_LEN) {
                        if chunk.len() == VALUE_LEN {
                            let mut candidate = [0u8; 32];
                            candidate.copy_from_slice(chunk);
                            if !have_set.contains(&candidate) {
                                if let Some(data) = get_blob(pile, &candidate) {
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
                let Some(branch_id) = triblespace_core::id::Id::new(branch_id_bytes) else {
                    send_u8(send, RSP_NONE).await?;
                    continue;
                };
                match pile.head(branch_id) {
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

fn get_blob(pile: &mut Pile<Blake3>, hash: &RawHash) -> Option<Vec<u8>> {
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let reader = BlobStore::<Blake3>::reader(pile).ok()?;
    reader.get::<Bytes, UnknownBlob>(handle).ok().map(|b| b.to_vec())
}
