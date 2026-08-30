//! READ-authorized, stream-pinned repair of one collection activation overlay.

use std::sync::Arc;

use anyhow::{Result, bail};
use ed25519_dalek::VerifyingKey;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{CapabilityProofBundle, CapabilityProofId};
use triblespace_core::collection::{CollectionRecord, collection_reader_is_admitted_by_policy_at};

use crate::collection_activation::{
    CollectionActivationOverlay, decode_write_evidence_bundle, encode_write_evidence_bundle,
};
use crate::collection_delta::{decode_record, encode_record};
use crate::collection_wire::{
    CollectionRepairAdmission, CollectionRepairCommand, CollectionRepairComponent,
    CollectionRepairHello, CollectionRepairManifest, recv_repair_admission, recv_repair_command,
    recv_repair_hello, recv_repair_node_response, send_repair_admission, send_repair_done,
    send_repair_hello, send_repair_node_request, send_repair_node_response,
};
use crate::patch_repair::{
    PatchNodeResponse, PatchRepairRequest, PatchRepairWalker, PatchSummary, patch_node_response,
    validate_patch_node,
};
use crate::transport::Conn;

/// Evidence missing from the caller's immutable local observation.
#[derive(Clone, Debug)]
pub(crate) struct CollectionRepairDelta {
    pub(crate) records: Vec<CollectionRecord>,
    pub(crate) write_evidence: Vec<CapabilityProofBundle>,
}

/// Serve the body of one collection-repair operation after its operation byte
/// has already been consumed.
///
/// `lookup` must return an immutable overlay. Its lifetime is the stream's
/// snapshot lease: every manifest and node response comes from the exact same
/// two PATCH roots, so no historical-root cache is needed.
pub(crate) async fn serve_collection_repair<R, W>(
    recv: &mut R,
    send: &mut W,
    remote: VerifyingKey,
    lookup: impl FnOnce(
        triblespace_core::collection::CollectionHandle,
    ) -> Option<Arc<CollectionActivationOverlay>>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let hello = recv_repair_hello(recv).await?;
    let Some(overlay) = lookup(hello.collection) else {
        send_repair_admission(send, CollectionRepairAdmission::Unavailable).await?;
        send.shutdown().await?;
        return Ok(());
    };
    let admitted = collection_reader_is_admitted_by_policy_at(
        hello.collection,
        overlay.policy(),
        remote,
        &hello.read_evidence,
        crate::clock::epoch_now(),
    );
    if !admitted {
        send_repair_admission(send, CollectionRepairAdmission::Rejected).await?;
        send.shutdown().await?;
        return Ok(());
    }

    let manifest = manifest(&overlay);
    send_repair_admission(send, CollectionRepairAdmission::Admitted(manifest)).await?;
    loop {
        match recv_repair_command(recv).await? {
            CollectionRepairCommand::Done => {
                require_eof(recv).await?;
                send.shutdown().await?;
                return Ok(());
            }
            CollectionRepairCommand::Node {
                component,
                prefix,
                expected_digest,
            } => {
                let summary = manifest.component(component);
                let Some(root) = summary.root() else {
                    bail!("client requested a node from an empty collection PATCH");
                };
                let request = PatchRepairRequest::new(
                    component,
                    summary,
                    component.key_len(),
                    prefix,
                    expected_digest,
                )?;
                if request.prefix().is_empty() && request.expected_digest() != root {
                    bail!("collection repair request does not pin the manifest root");
                }
                let response = node_response(&overlay, component, request.prefix())?;
                send_repair_node_response(send, &response, component).await?;
            }
        }
    }
}

fn manifest(overlay: &CollectionActivationOverlay) -> CollectionRepairManifest {
    CollectionRepairManifest {
        wake_root: overlay.wake_root(),
        records: overlay.records().summary(),
        write_evidence: overlay.write_evidence().summary(),
    }
}

fn node_response(
    overlay: &CollectionActivationOverlay,
    component: CollectionRepairComponent,
    prefix: &[u8],
) -> Result<PatchNodeResponse<Vec<u8>>> {
    match component {
        CollectionRepairComponent::Record => {
            patch_node_response(overlay.records().patch(), &[], prefix, |key, record| {
                if record.id().raw() != key {
                    bail!("collection record id does not match its PATCH leaf key");
                }
                encode_record(overlay.collection(), *record).map_err(anyhow::Error::new)
            })
        }
        CollectionRepairComponent::WriteEvidence => patch_node_response(
            overlay.write_evidence().patch(),
            &[],
            prefix,
            |key, bundle| {
                if bundle.proof().id().raw != key {
                    bail!("WRITE proof id does not match its PATCH leaf key");
                }
                encode_write_evidence_bundle(overlay.collection(), overlay.policy().write(), bundle)
                    .map_err(anyhow::Error::new)
            },
        ),
    }
}

/// Pull one exact collection overlay over an already authenticated iroh
/// connection. TLS binds `conn.remote_id()`; READ(C) binds the local endpoint
/// through the supplied portable proof forest.
pub(crate) async fn pull_collection<C: Conn>(
    conn: &C,
    local: &CollectionActivationOverlay,
    read_evidence: Vec<CapabilityProofBundle>,
) -> Result<CollectionRepairDelta> {
    let (mut send, mut recv) = conn.open_bi().await?;
    pull_collection_stream(&mut send, &mut recv, local, read_evidence).await
}

async fn pull_collection_stream<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &CollectionActivationOverlay,
    read_evidence: Vec<CapabilityProofBundle>,
) -> Result<CollectionRepairDelta>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    send_repair_hello(
        send,
        &CollectionRepairHello {
            collection: local.collection(),
            read_evidence,
        },
    )
    .await?;
    let remote = match recv_repair_admission(recv).await? {
        CollectionRepairAdmission::Admitted(manifest) => manifest,
        CollectionRepairAdmission::Rejected => bail!("remote rejected READ(C) evidence"),
        CollectionRepairAdmission::Unavailable => {
            bail!("remote does not retain the requested collection")
        }
    };

    let records = pull_record_patch(send, recv, local, remote.records).await?;
    let write_evidence =
        pull_write_evidence_patch(send, recv, local, remote.write_evidence).await?;
    send_repair_done(send).await?;
    send.shutdown().await?;
    require_eof(recv).await?;
    Ok(CollectionRepairDelta {
        records,
        write_evidence,
    })
}

async fn pull_record_patch<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &CollectionActivationOverlay,
    remote: PatchSummary,
) -> Result<Vec<CollectionRecord>>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let component = CollectionRepairComponent::Record;
    let mut walker = PatchRepairWalker::new(component, remote, component.key_len())?;
    let mut missing = Vec::new();
    loop {
        let request = walker.next_request(|_, prefix| {
            local.records().patch().merkle_node(prefix).map(|node| {
                PatchSummary::new(Some(node.digest()), node.leaf_count())
                    .expect("a PATCH node is nonempty")
            })
        })?;
        let Some(request) = request else {
            break;
        };
        send_repair_node_request(send, &request, component).await?;
        let response = recv_repair_node_response(recv, component).await?;
        validate_response(&request, component, &response, |key, bytes| {
            let record = decode_record(local.collection(), bytes)?;
            if record.id().raw().as_slice() != key {
                bail!("collection record body does not match its PATCH leaf key");
            }
            Ok(())
        })?;
        if let Some(leaf) = walker.accept(&request, response, |_, key| {
            let Ok(key) = <[u8; 16]>::try_from(key) else {
                return false;
            };
            triblespace_core::id::Id::new(key).is_some_and(|id| local.records().get(id).is_some())
        })? {
            missing.push(decode_record(local.collection(), &leaf.value)?);
        }
    }
    walker.finish()?;
    Ok(missing)
}

async fn pull_write_evidence_patch<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &CollectionActivationOverlay,
    remote: PatchSummary,
) -> Result<Vec<CapabilityProofBundle>>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let component = CollectionRepairComponent::WriteEvidence;
    let mut walker = PatchRepairWalker::new(component, remote, component.key_len())?;
    let mut missing = Vec::new();
    loop {
        let request = walker.next_request(|_, prefix| {
            local
                .write_evidence()
                .patch()
                .merkle_node(prefix)
                .map(|node| {
                    PatchSummary::new(Some(node.digest()), node.leaf_count())
                        .expect("a PATCH node is nonempty")
                })
        })?;
        let Some(request) = request else {
            break;
        };
        send_repair_node_request(send, &request, component).await?;
        let response = recv_repair_node_response(recv, component).await?;
        validate_response(&request, component, &response, |key, bytes| {
            let bundle =
                decode_write_evidence_bundle(local.collection(), local.policy().write(), bytes)?;
            if bundle.proof().id().raw.as_slice() != key {
                bail!("WRITE proof body does not match its PATCH leaf key");
            }
            Ok(())
        })?;
        if let Some(leaf) = walker.accept(&request, response, |_, key| {
            let Ok(key) = <[u8; 32]>::try_from(key) else {
                return false;
            };
            local
                .write_evidence()
                .get(CapabilityProofId::new(key))
                .is_some()
        })? {
            missing.push(decode_write_evidence_bundle(
                local.collection(),
                local.policy().write(),
                &leaf.value,
            )?);
        }
    }
    walker.finish()?;
    Ok(missing)
}

fn validate_response<S>(
    request: &PatchRepairRequest<S>,
    component: CollectionRepairComponent,
    response: &PatchNodeResponse<Vec<u8>>,
    validate_leaf: impl FnOnce(&[u8], &[u8]) -> Result<()>,
) -> Result<()> {
    match response {
        PatchNodeResponse::Found(node) => {
            validate_patch_node(request, component.key_len(), &[], node, |key, bytes| {
                validate_leaf(key, bytes)
            })
        }
        PatchNodeResponse::PrefixAbsent => {
            bail!("remote omitted an authenticated collection PATCH prefix")
        }
        PatchNodeResponse::SnapshotUnavailable => {
            bail!("remote lost a stream-pinned collection PATCH")
        }
    }
}

async fn require_eof<R: AsyncRead + Unpin>(recv: &mut R) -> Result<()> {
    let mut trailing = [0u8; 1];
    if recv.read(&mut trailing).await? != 0 {
        bail!("collection repair stream contains trailing bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use triblespace_core::collection::{
        AdmissionPolicy, CollectionCommit, CollectionData, CollectionPolicy, CollectionRecord,
        CollectionStore, CollectionStoreExt, empty_metadata_handle,
    };
    use triblespace_core::repo::SnapshotSource;
    use triblespace_core::repo::memoryrepo::MemoryRepo;

    use crate::collection_activation::collection_activation_overlay;
    use crate::protocol::recv_u8;

    use super::*;

    #[tokio::test]
    async fn one_stream_repairs_records_without_global_inventory() {
        let policy = CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open);
        let mut server_store = MemoryRepo::default();
        let server_collection = server_store.collection("shared", policy.clone()).unwrap();
        server_store
            .insert(CollectionRecord::Commit(CollectionCommit::sign(
                &SigningKey::from_bytes(&[7; 32]),
                server_collection.handle(),
                CollectionData::new([9; 32]),
                empty_metadata_handle(),
            )))
            .unwrap();
        let server_snapshot = server_store.snapshot().unwrap();
        let server = Arc::new(
            collection_activation_overlay(&server_snapshot, server_collection.handle()).unwrap(),
        );
        let mut client_store = MemoryRepo::default();
        let client_collection = client_store.collection("shared", policy).unwrap();
        assert_eq!(client_collection.handle(), server_collection.handle());
        let client_snapshot = client_store.snapshot().unwrap();
        let client =
            collection_activation_overlay(&client_snapshot, client_collection.handle()).unwrap();

        let (server_io, client_io) = tokio::io::duplex(1 << 20);
        let (mut server_recv, mut server_send) = tokio::io::split(server_io);
        let (mut client_recv, mut client_send) = tokio::io::split(client_io);
        let server_task = tokio::spawn(async move {
            assert_eq!(
                recv_u8(&mut server_recv).await.unwrap(),
                crate::collection_wire::OP_COLLECTION_REPAIR
            );
            serve_collection_repair(
                &mut server_recv,
                &mut server_send,
                SigningKey::from_bytes(&[8; 32]).verifying_key(),
                |collection| (collection == server.collection()).then_some(server),
            )
            .await
            .unwrap();
        });

        let delta = pull_collection_stream(&mut client_send, &mut client_recv, &client, vec![])
            .await
            .unwrap();
        assert_eq!(delta.records.len(), 1);
        assert!(delta.write_evidence.is_empty());
        server_task.await.unwrap();
    }
}
