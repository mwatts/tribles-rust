//! READ-authorized, stream-pinned repair of one collection activation overlay.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anybytes::Bytes;
use anyhow::{Result, bail};
use ed25519_dalek::VerifyingKey;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{CapabilityProofBundle, CapabilityProofId};
use triblespace_core::collection::{
    CollectionHandle, CollectionRecord, collection_reader_is_admitted_by_policy_at,
    collection_writer_is_admitted_by_policy_at,
};
use triblespace_core::patch::{Blake3Merkle, IdentitySchema, PATCH};

use crate::collection_activation::{
    CollectionActivationOverlay, decode_write_evidence_bundle, encode_write_evidence_bundle,
};
use crate::collection_delta::{decode_record, encode_record};
use crate::collection_wire::{
    CollectionRepairAdmission, CollectionRepairCommand, CollectionRepairComponent,
    CollectionRepairManifest, recv_repair_admission, recv_repair_blob_response,
    recv_repair_collection, recv_repair_command, recv_repair_hello, recv_repair_node_response,
    send_repair_admission, send_repair_blob_request, send_repair_blob_response, send_repair_done,
    send_repair_evidence, send_repair_node_request, send_repair_node_response,
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
    pub(crate) blobs: Vec<([u8; 32], Bytes)>,
    pub(crate) full_cursor: Option<FullReplicaCursor>,
    pub(crate) more: bool,
}

pub(crate) type DisclosureForestPatch = PATCH<80, IdentitySchema, (), Blake3Merkle>;

#[derive(Clone, Debug)]
pub(crate) struct FullReplicaState {
    pub(crate) forest: DisclosureForestPatch,
    pub(crate) direct_roots: HashSet<[u8; 32]>,
}

#[derive(Clone, Debug)]
pub(crate) struct FullReplicaCursor {
    pub(crate) remote_semantic: [u8; 32],
    pub(crate) remote_forest: PatchSummary,
    pub(crate) seen: DisclosureForestPatch,
}

const MAX_REPAIR_RECORD_ITEMS: usize = 4_096;
const MAX_REPAIR_WRITE_EVIDENCE_ITEMS: usize = 16;
const MAX_REPAIR_NODE_REQUESTS: usize = 16_384;
const MAX_SERVER_REPAIR_COMMANDS: usize = 512;
const MAX_SERVER_NODE_RESPONSE_BYTES: usize = 64 << 20;

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
    ) -> Option<(
        Arc<CollectionActivationOverlay>,
        Arc<[CapabilityProofBundle]>,
        Arc<FullReplicaState>,
    )>,
    disclosed_blob: impl Fn(CollectionHandle, [u8; 32]) -> Option<Bytes>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let collection = recv_repair_collection(recv).await?;
    let Some((overlay, _local_read_evidence, full)) = lookup(collection) else {
        send_repair_admission(send, CollectionRepairAdmission::Unavailable).await?;
        send.shutdown().await?;
        return Ok(());
    };
    let hello = recv_repair_hello(recv).await?;
    let admitted = collection_reader_is_admitted_by_policy_at(
        collection,
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

    let manifest = manifest(&overlay, &full.forest);
    send_repair_admission(send, CollectionRepairAdmission::Admitted(manifest)).await?;
    let mut commands = 0_usize;
    let mut response_bytes = 0_usize;
    loop {
        if commands == MAX_SERVER_REPAIR_COMMANDS {
            bail!("collection repair command budget exhausted");
        }
        commands += 1;
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
                let response = node_response(&overlay, &full.forest, component, request.prefix())?;
                if response_bytes >= MAX_SERVER_NODE_RESPONSE_BYTES {
                    bail!("collection repair response budget exhausted");
                }
                response_bytes = response_bytes.saturating_add(node_response_wire_len(&response));
                send_repair_node_response(send, &response, component).await?;
            }
            CollectionRepairCommand::Blob(handle) => {
                if response_bytes >= MAX_SERVER_NODE_RESPONSE_BYTES {
                    bail!("collection repair response budget exhausted");
                }
                let bytes = disclosed_blob(collection, handle);
                response_bytes = response_bytes
                    .saturating_add(bytes.as_ref().map_or(8, |bytes| 8 + bytes.len()));
                send_repair_blob_response(send, bytes.as_deref()).await?;
            }
        }
    }
}

fn node_response_wire_len(response: &PatchNodeResponse<Vec<u8>>) -> usize {
    match response {
        PatchNodeResponse::SnapshotUnavailable | PatchNodeResponse::PrefixAbsent => 1,
        PatchNodeResponse::Found(crate::patch_repair::PatchNode::Leaf { leaf, .. }) => {
            1 + 1 + 32 + leaf.key.len() + 4 + leaf.value.len()
        }
        PatchNodeResponse::Found(crate::patch_repair::PatchNode::Branch { branch, .. }) => {
            1 + 1 + 32 + 8 + branch.representative.len() + 1 + 4 + branch.children.len() * 41
        }
    }
}

fn manifest(
    overlay: &CollectionActivationOverlay,
    forest: &DisclosureForestPatch,
) -> CollectionRepairManifest {
    CollectionRepairManifest {
        wake_root: overlay.wake_root(),
        records: overlay.records().summary(),
        write_evidence: overlay.write_evidence().summary(),
        resident: PatchSummary::from_patch(forest),
    }
}

fn node_response(
    overlay: &CollectionActivationOverlay,
    forest: &DisclosureForestPatch,
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
        CollectionRepairComponent::Resident => {
            patch_node_response(forest, &[], prefix, |_, ()| Ok(Vec::new()))
        }
    }
}

/// Pull one exact collection overlay over an already authenticated iroh
/// connection. TLS binds `conn.remote_id()`; READ(C) binds the local endpoint
/// through the supplied portable proof forest.
pub(crate) async fn pull_collection<C: Conn>(
    conn: &C,
    local: &CollectionActivationOverlay,
    read_evidence: Vec<CapabilityProofBundle>,
    full_state: &FullReplicaState,
    prior_cursor: Option<&FullReplicaCursor>,
    parent_blob: impl Fn([u8; 32]) -> Option<Bytes>,
    full: bool,
) -> Result<CollectionRepairDelta> {
    let (mut send, mut recv) = conn.open_bi().await?;
    pull_collection_stream(
        &mut send,
        &mut recv,
        local,
        read_evidence,
        full_state,
        prior_cursor,
        parent_blob,
        full,
    )
    .await
}

async fn pull_collection_stream<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &CollectionActivationOverlay,
    read_evidence: Vec<CapabilityProofBundle>,
    full_state: &FullReplicaState,
    prior_cursor: Option<&FullReplicaCursor>,
    parent_blob: impl Fn([u8; 32]) -> Option<Bytes>,
    full: bool,
) -> Result<CollectionRepairDelta>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    crate::protocol::send_u8(send, crate::collection_wire::OP_COLLECTION_REPAIR).await?;
    crate::protocol::send_hash(send, &local.collection().raw).await?;
    send_repair_evidence(send, &read_evidence).await?;
    let remote = match recv_repair_admission(recv).await? {
        CollectionRepairAdmission::Admitted(manifest) => manifest,
        CollectionRepairAdmission::Rejected => bail!("remote rejected READ(C) evidence"),
        CollectionRepairAdmission::Unavailable => {
            bail!("remote does not retain the requested collection")
        }
    };

    let mut remaining_requests = MAX_SERVER_REPAIR_COMMANDS - 1;
    let mut response_bytes = 0_usize;
    let (write_evidence, write_more) = pull_write_evidence_patch(
        send,
        recv,
        local,
        remote.write_evidence,
        &mut remaining_requests,
        &mut response_bytes,
    )
    .await?;
    let mut complete_write_evidence = local
        .write_evidence()
        .bundles()
        .cloned()
        .collect::<Vec<_>>();
    complete_write_evidence.extend(write_evidence.iter().cloned());
    let (records, record_more) = pull_record_patch(
        send,
        recv,
        local,
        remote.records,
        &complete_write_evidence,
        &mut remaining_requests,
        &mut response_bytes,
    )
    .await?;
    let semantic_changed = !write_evidence.is_empty() || !records.is_empty();
    let matching_cursor = prior_cursor.filter(|cursor| {
        cursor.remote_semantic == remote.wake_root && cursor.remote_forest == remote.resident
    });
    let (blobs, seen, forest_more) = if full && !write_more && !record_more && !semantic_changed {
        pull_resident_patch(
            send,
            recv,
            full_state,
            matching_cursor.map(|cursor| &cursor.seen),
            parent_blob,
            remote.resident,
            &mut remaining_requests,
            &mut response_bytes,
        )
        .await?
    } else {
        (
            Vec::new(),
            DisclosureForestPatch::new(),
            full && (write_more || record_more || semantic_changed),
        )
    };
    send_repair_done(send).await?;
    send.shutdown().await?;
    require_eof(recv).await?;
    Ok(CollectionRepairDelta {
        records,
        write_evidence,
        blobs,
        full_cursor: full.then(|| FullReplicaCursor {
            remote_semantic: remote.wake_root,
            remote_forest: remote.resident,
            seen,
        }),
        more: write_more || record_more || forest_more,
    })
}

async fn pull_resident_patch<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &FullReplicaState,
    prior_seen: Option<&DisclosureForestPatch>,
    parent_blob: impl Fn([u8; 32]) -> Option<Bytes>,
    remote: PatchSummary,
    remaining_requests: &mut usize,
    response_bytes: &mut usize,
) -> Result<(Vec<([u8; 32], Bytes)>, DisclosureForestPatch, bool)>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let component = CollectionRepairComponent::Resident;
    let mut walker = PatchRepairWalker::new(component, remote, component.key_len())?;
    let mut blobs = Vec::new();
    let mut seen = prior_seen.cloned().unwrap_or_default();
    let mut known = local.forest.clone();
    known.union(seen.clone());
    let mut complete = false;
    let mut trusted_depth = known
        .iter_ordered()
        .map(|key| {
            (
                <[u8; 32]>::try_from(&key[48..]).unwrap(),
                u64::from_be_bytes(key[..8].try_into().unwrap()),
            )
        })
        .collect::<HashMap<_, _>>();
    for root in &local.direct_roots {
        trusted_depth.entry(*root).or_insert(0);
    }
    loop {
        if *remaining_requests < 2 || *response_bytes >= MAX_SERVER_NODE_RESPONSE_BYTES {
            break;
        }
        let request = walker.next_request(|_, prefix| {
            known.merkle_node(prefix).map(|node| {
                PatchSummary::new(Some(node.digest()), node.leaf_count())
                    .expect("a PATCH node is nonempty")
            })
        })?;
        let Some(request) = request else {
            walker.finish()?;
            complete = true;
            break;
        };
        *remaining_requests -= 1;
        send_repair_node_request(send, &request, component).await?;
        let response = recv_repair_node_response(recv, component).await?;
        *response_bytes = response_bytes.saturating_add(node_response_wire_len(&response));
        validate_response(&request, component, &response, |_key, value| {
            if !value.is_empty() {
                bail!("disclosure forest PATCH leaf value must be empty");
            }
            Ok(())
        })?;
        if let Some(leaf) = walker.accept(&request, response, |_, key| {
            <[u8; 80]>::try_from(key).is_ok_and(|key| known.get(&key).is_some())
        })? {
            let key = <[u8; 80]>::try_from(leaf.key.as_slice())
                .map_err(|_| anyhow::anyhow!("disclosure forest key has wrong width"))?;
            let depth = u64::from_be_bytes(key[..8].try_into().unwrap());
            let parent: [u8; 32] = key[8..40].try_into().unwrap();
            let index = u64::from_be_bytes(key[40..48].try_into().unwrap());
            let handle: [u8; 32] = key[48..].try_into().unwrap();
            let trusted = if depth == 0 {
                parent == handle && index == u64::MAX && local.direct_roots.contains(&handle)
            } else {
                let resident_parent = parent_blob(parent);
                let parent_bytes = blobs
                    .iter()
                    .find_map(|(hash, bytes)| (*hash == parent).then_some(bytes))
                    .or(resident_parent.as_ref());
                trusted_depth
                    .get(&parent)
                    .is_some_and(|parent_depth| parent_depth.checked_add(1) == Some(depth))
                    && parent_bytes.is_some_and(|bytes| {
                        usize::try_from(index)
                            .ok()
                            .and_then(|index| bytes.chunks_exact(32).nth(index))
                            == Some(handle.as_slice())
                    })
            };
            if !trusted {
                continue;
            }
            seen.insert(&triblespace_core::patch::Entry::new(&key));
            known.insert(&triblespace_core::patch::Entry::new(&key));
            trusted_depth.insert(handle, depth);
            if parent_blob(handle).is_some() {
                continue;
            }
            *remaining_requests -= 1;
            send_repair_blob_request(send, handle).await?;
            let Some(bytes) = recv_repair_blob_response(recv).await? else {
                bail!("authenticated disclosure-forest handle is not resident on provider");
            };
            if *blake3::hash(&bytes).as_bytes() != handle {
                bail!("resident blob response does not match requested handle");
            }
            blobs.push((handle, bytes));
            *response_bytes =
                response_bytes.saturating_add(8 + blobs.last().expect("just pushed").1.len());
        }
    }
    Ok((blobs, seen, !complete))
}

async fn pull_record_patch<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &CollectionActivationOverlay,
    remote: PatchSummary,
    write_evidence: &[CapabilityProofBundle],
    remaining_requests: &mut usize,
    response_bytes: &mut usize,
) -> Result<(Vec<CollectionRecord>, bool)>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let component = CollectionRepairComponent::Record;
    let mut walker = PatchRepairWalker::new(component, remote, component.key_len())?;
    let mut missing = Vec::new();
    let mut admission_by_writer = HashMap::new();
    let mut requests = 0;
    let mut complete = false;
    loop {
        if requests >= MAX_REPAIR_NODE_REQUESTS
            || *remaining_requests == 0
            || *response_bytes >= MAX_SERVER_NODE_RESPONSE_BYTES
            || missing.len() >= MAX_REPAIR_RECORD_ITEMS
        {
            break;
        }
        let request = walker.next_request(|_, prefix| {
            local.records().patch().merkle_node(prefix).map(|node| {
                PatchSummary::new(Some(node.digest()), node.leaf_count())
                    .expect("a PATCH node is nonempty")
            })
        })?;
        let Some(request) = request else {
            complete = true;
            break;
        };
        requests += 1;
        *remaining_requests -= 1;
        send_repair_node_request(send, &request, component).await?;
        let response = recv_repair_node_response(recv, component).await?;
        *response_bytes = response_bytes.saturating_add(node_response_wire_len(&response));
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
            let record = decode_record(local.collection(), &leaf.value)?;
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            let Ok(writer) = VerifyingKey::from_bytes(&commit.public_key().raw) else {
                continue;
            };
            let admitted = *admission_by_writer
                .entry(writer.to_bytes())
                .or_insert_with(|| {
                    collection_writer_is_admitted_by_policy_at(
                        local.collection(),
                        local.policy(),
                        writer,
                        write_evidence,
                        crate::clock::epoch_now(),
                    )
                });
            if admitted {
                missing.push(CollectionRecord::Commit(commit));
            }
        }
    }
    if complete {
        walker.finish()?;
    }
    Ok((missing, !complete))
}

async fn pull_write_evidence_patch<W, R>(
    send: &mut W,
    recv: &mut R,
    local: &CollectionActivationOverlay,
    remote: PatchSummary,
    remaining_requests: &mut usize,
    response_bytes: &mut usize,
) -> Result<(Vec<CapabilityProofBundle>, bool)>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let component = CollectionRepairComponent::WriteEvidence;
    let mut walker = PatchRepairWalker::new(component, remote, component.key_len())?;
    let mut missing = Vec::new();
    let mut requests = 0;
    let mut complete = false;
    loop {
        if requests >= MAX_REPAIR_NODE_REQUESTS
            || *remaining_requests == 0
            || *response_bytes >= MAX_SERVER_NODE_RESPONSE_BYTES
            || missing.len() >= MAX_REPAIR_WRITE_EVIDENCE_ITEMS
        {
            break;
        }
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
            complete = true;
            break;
        };
        requests += 1;
        *remaining_requests -= 1;
        send_repair_node_request(send, &request, component).await?;
        let response = recv_repair_node_response(recv, component).await?;
        *response_bytes = response_bytes.saturating_add(node_response_wire_len(&response));
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
    if complete {
        walker.finish()?;
    }
    Ok((missing, !complete))
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
        let empty_full = FullReplicaState {
            forest: DisclosureForestPatch::new(),
            direct_roots: HashSet::new(),
        };

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
                |collection| {
                    (collection == server.collection()).then_some((
                        server,
                        Arc::<[CapabilityProofBundle]>::from([]),
                        Arc::new(empty_full),
                    ))
                },
                |_, _| None,
            )
            .await
            .unwrap();
        });

        let delta = pull_collection_stream(
            &mut client_send,
            &mut client_recv,
            &client,
            vec![],
            &FullReplicaState {
                forest: DisclosureForestPatch::new(),
                direct_roots: HashSet::new(),
            },
            None,
            |_| None,
            false,
        )
        .await
        .unwrap();
        assert_eq!(delta.records.len(), 1);
        assert!(delta.write_evidence.is_empty());
        assert!(delta.blobs.is_empty());
        server_task.await.unwrap();
    }
}
