//! Bounded wire codec for authorized prefix-Merkle reconciliation.
//!
//! CONNECT is still the mandatory first stream on a connection. A successful
//! [`OP_INVENTORY_AUTH`] then exchanges the two subject-bound SYNC_TEAM proofs
//! and installs exactly one server-selected inventory team session for that
//! connection. Manifest, node, exact `GET_BLOB`, provider, and mirror range
//! operations are rejected until that mutual second authorization succeeds.
//! The proofs are not repeated on every request.
//!
//! Every node and range request pins an exact component Merkle root. If that
//! immutable snapshot is no longer cached for the authorized team, the server
//! returns an explicit unavailable response; it never serves bytes from its
//! current snapshot as a fallback.

use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{
    CapabilityMode, CapabilityProof, CapabilityProofBundle, CapabilityRequest,
    MAX_CAPABILITY_PROOF_STEPS, VerifiedCapability,
};
use triblespace_core::collection::{COLLECTION_COMMIT_BYTES_LEN, CollectionRecord};
use triblespace_core::patch::{Blake3Merkle, IdentitySchema, PATCH};
use triblespace_core::repo::peer::{PEER_EVIDENCE_BYTES_LEN, PeerEvidence};

use crate::inventory::{
    AuthorizedInventorySession, ComponentManifest, InventoryComponent, InventoryGeneration,
    InventoryManifest, sync_team_capability_atom,
};
use crate::patch_repair::{
    PatchBranch, PatchChild, PatchLeaf, PatchNode, PatchNodeResponse, PatchRepairRequest,
    PatchSummary, patch_node_response, validate_patch_node,
};
use crate::protocol::{
    recv_capability_proof_bundle, recv_hash, recv_proof_response, recv_u8, recv_u32_be,
    recv_u64_be, send_capability_proof_bundle, send_hash, send_u8, send_u32_be, send_u64_be,
};
use crate::transport::Conn;

/// Authorize exactly one server-selected team inventory session on this
/// CONNECT-authenticated connection.
pub(crate) const OP_INVENTORY_AUTH: u8 = 0x08;
/// Fetch the four immutable component roots.
pub(crate) const OP_INVENTORY_MANIFEST: u8 = 0x09;
/// Fetch one authenticated PATCH node under a protocol-relative prefix.
pub(crate) const OP_INVENTORY_NODE: u8 = 0x0A;
/// Fetch one bounded range of a blob present in a pinned Blob inventory.
pub(crate) const OP_INVENTORY_BLOB_RANGE: u8 = 0x0B;

pub(crate) const INVENTORY_AUTH_OK: u8 = crate::protocol::AUTH_OK;
pub(crate) const INVENTORY_AUTH_REJECTED: u8 = crate::protocol::AUTH_REJECTED;

const NODE_FOUND: u8 = 0x00;
const NODE_SNAPSHOT_UNAVAILABLE: u8 = 0x01;
const NODE_PREFIX_ABSENT: u8 = 0x02;
const NODE_LEAF: u8 = 0x00;
const NODE_BRANCH: u8 = 0x01;

const BLOB_RANGE_FOUND: u8 = 0x00;
const BLOB_RANGE_SNAPSHOT_UNAVAILABLE: u8 = 0x01;
const BLOB_RANGE_NOT_IN_SNAPSHOT: u8 = 0x02;

const RECORD_MAX_BYTES: usize = 1 + COLLECTION_COMMIT_BYTES_LEN;
const PROOF_MAX_BYTES: usize = 32 + MAX_CAPABILITY_PROOF_STEPS * 128;

/// Maximum payload of one resumable mirror range operation.
pub(crate) const BLOB_TRANSFER_CHUNK_BYTES: usize = 1024 * 1024;

pub(crate) async fn send_u16_be<W: AsyncWrite + Unpin>(send: &mut W, value: u16) -> Result<()> {
    send.write_all(&value.to_be_bytes())
        .await
        .map_err(|error| anyhow!("send u16: {error}"))
}

pub(crate) async fn recv_u16_be<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u16> {
    let mut bytes = [0; 2];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv u16: {error}"))?;
    Ok(u16::from_be_bytes(bytes))
}

pub(crate) async fn require_eof<R: AsyncRead + Unpin>(recv: &mut R) -> Result<()> {
    let mut trailing = [0; 1];
    if recv
        .read(&mut trailing)
        .await
        .map_err(|error| anyhow!("read request terminator: {error}"))?
        != 0
    {
        bail!("inventory frame contains trailing bytes");
    }
    Ok(())
}

async fn recv_exact_vec<R: AsyncRead + Unpin>(
    recv: &mut R,
    length: usize,
    what: &str,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| anyhow!("cannot allocate {what}: {error}"))?;
    bytes.resize(length, 0);
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv {what}: {error}"))?;
    Ok(bytes)
}

async fn recv_bounded_u16_frame<R: AsyncRead + Unpin>(
    recv: &mut R,
    maximum: usize,
    what: &str,
) -> Result<Vec<u8>> {
    let length = recv_u16_be(recv).await? as usize;
    if length > maximum {
        bail!("{what} frame is {length} bytes; limit is {maximum}");
    }
    recv_exact_vec(recv, length, what).await
}

/// Client-side reciprocal second authorization exchange.
pub(crate) async fn op_inventory_auth<C: Conn>(
    connection: &C,
    proof: &CapabilityProofBundle,
    team: ed25519_dalek::VerifyingKey,
    expected_remote: [u8; 32],
) -> Result<VerifiedCapability> {
    if connection.remote_id() != expected_remote {
        bail!("inventory connection identity does not match the requested peer");
    }
    let remote = ed25519_dalek::VerifyingKey::from_bytes(&expected_remote)
        .map_err(|error| anyhow!("invalid remote endpoint identity: {error}"))?;
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory auth stream: {error}"))?;
    send_u8(&mut send, OP_INVENTORY_AUTH).await?;
    send_capability_proof_bundle(&mut send, proof).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory auth request: {error}"))?;

    recv_proof_response(
        &mut recv,
        team,
        remote,
        CapabilityRequest::new(sync_team_capability_atom(team), CapabilityMode::Invoke),
        crate::clock::epoch_now(),
        "SYNC_TEAM",
    )
    .await
}

/// Decode the one proof bundle carried by [`OP_INVENTORY_AUTH`].
pub(crate) async fn recv_inventory_auth_request<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CapabilityProofBundle> {
    let proof = recv_capability_proof_bundle(recv).await?;
    require_eof(recv).await?;
    Ok(proof)
}

/// Send a successful second-authorization response.
pub(crate) async fn send_inventory_auth_ok<W: AsyncWrite + Unpin>(
    send: &mut W,
    proof: &CapabilityProofBundle,
) -> Result<()> {
    send_u8(send, INVENTORY_AUTH_OK).await?;
    send_capability_proof_bundle(send, proof).await
}

/// Send a rejected second-authorization response without leaking details.
pub(crate) async fn send_inventory_auth_rejected<W: AsyncWrite + Unpin>(
    send: &mut W,
) -> Result<()> {
    send_u8(send, INVENTORY_AUTH_REJECTED).await
}

/// Decode the empty body of a manifest request. The connection session fixes
/// the only possible team inventory.
pub(crate) async fn recv_manifest_request<R: AsyncRead + Unpin>(recv: &mut R) -> Result<()> {
    require_eof(recv).await
}

/// Encode one exact four-component manifest.
pub(crate) async fn send_manifest<W: AsyncWrite + Unpin>(
    send: &mut W,
    manifest: &InventoryManifest,
) -> Result<()> {
    send_hash(send, &manifest.generation().into_bytes()).await?;
    for component in manifest.components() {
        send_u8(send, component.component() as u8).await?;
        send_u64_be(send, component.summary().leaf_count()).await?;
        match component.summary().root() {
            None => send_u8(send, 0).await?,
            Some(root) => {
                send_u8(send, 1).await?;
                send_hash(send, &root).await?;
            }
        }
    }
    Ok(())
}

async fn recv_manifest<R: AsyncRead + Unpin>(
    recv: &mut R,
    team: ed25519_dalek::VerifyingKey,
) -> Result<InventoryManifest> {
    let generation = InventoryGeneration::from_bytes(recv_hash(recv).await?);
    let mut components = Vec::with_capacity(InventoryComponent::ALL.len());
    for expected_component in InventoryComponent::ALL {
        let component = InventoryComponent::from_byte(recv_u8(recv).await?)?;
        if component != expected_component {
            bail!("manifest components are out of canonical order");
        }
        let leaf_count = recv_u64_be(recv).await?;
        let root = match recv_u8(recv).await? {
            0 => None,
            1 => Some(recv_hash(recv).await?),
            marker => bail!("invalid inventory root marker {marker:#x}"),
        };
        let summary = PatchSummary::new(root, leaf_count)?;
        components.push(ComponentManifest::from_wire(component, summary));
    }
    require_eof(recv).await?;
    let components: [ComponentManifest; 4] = components
        .try_into()
        .expect("the canonical component loop has exactly four entries");
    InventoryManifest::from_wire(team, generation, components)
}

/// Fetch an authenticated root manifest for the connection's authorized team.
pub(crate) async fn op_inventory_manifest<C: Conn>(
    connection: &C,
    team: ed25519_dalek::VerifyingKey,
) -> Result<InventoryManifest> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory manifest stream: {error}"))?;
    send_u8(&mut send, OP_INVENTORY_MANIFEST).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory manifest request: {error}"))?;
    recv_manifest(&mut recv, team).await
}

async fn send_node_request<W: AsyncWrite + Unpin>(
    send: &mut W,
    request: &PatchRepairRequest<InventoryComponent>,
) -> Result<()> {
    send_u8(send, OP_INVENTORY_NODE).await?;
    send_u8(send, *request.scope() as u8).await?;
    send_hash(
        send,
        &request
            .summary()
            .root()
            .expect("a repair request has a nonempty summary"),
    )
    .await?;
    send_u64_be(send, request.summary().leaf_count()).await?;
    send_u8(
        send,
        u8::try_from(request.prefix().len()).expect("inventory keys are at most 64 bytes"),
    )
    .await?;
    send.write_all(request.prefix())
        .await
        .map_err(|error| anyhow!("send inventory prefix: {error}"))?;
    send_hash(send, &request.expected_digest()).await
}

/// Decode one node request relative to the connection's authorized base.
pub(crate) async fn recv_node_request<R: AsyncRead + Unpin>(
    recv: &mut R,
    authorized: AuthorizedInventorySession,
) -> Result<PatchRepairRequest<InventoryComponent>> {
    let component = InventoryComponent::from_byte(recv_u8(recv).await?)?;
    let root = recv_hash(recv).await?;
    let leaf_count = recv_u64_be(recv).await?;
    let summary = PatchSummary::new(Some(root), leaf_count)?;
    let prefix_length = recv_u8(recv).await? as usize;
    let maximum = component.relative_key_len(authorized.base_prefix(component));
    if prefix_length > maximum {
        bail!(
            "inventory prefix is {prefix_length} bytes; component-relative key is {maximum} bytes"
        );
    }
    let prefix = recv_exact_vec(recv, prefix_length, "inventory prefix").await?;
    let expected_digest = recv_hash(recv).await?;
    require_eof(recv).await?;
    PatchRepairRequest::new(component, summary, maximum, prefix, expected_digest)
}

/// Value carried by one inventory leaf. Keys remain the only hashed input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InventoryLeafValue {
    /// PEER's relative key is the peer public key; the fixed team base
    /// reconstructs the complete evidence body.
    Peer,
    CollectionRecord(CollectionRecord),
    CapabilityProof(CapabilityProof),
    /// The blob handle is the complete authenticated leaf value. Bytes are
    /// fetched and BLAKE3-verified only through a range request.
    Blob,
}

impl InventoryLeafValue {
    fn component(&self) -> InventoryComponent {
        match self {
            Self::Peer => InventoryComponent::Peer,
            Self::CollectionRecord(_) => InventoryComponent::CollectionRecord,
            Self::CapabilityProof(_) => InventoryComponent::CapabilityProof,
            Self::Blob => InventoryComponent::Blob,
        }
    }
}

pub(crate) fn inventory_node_response<const KEY_LEN: usize, V>(
    inventory: &PATCH<KEY_LEN, IdentitySchema, V, Blake3Merkle>,
    team: ed25519_dalek::VerifyingKey,
    component: InventoryComponent,
    relative_prefix: &[u8],
    resolve: impl FnOnce([u8; KEY_LEN], &V) -> Result<InventoryLeafValue>,
) -> Result<PatchNodeResponse<InventoryLeafValue>> {
    let base = component.base_prefix(team);
    patch_node_response(inventory, base.as_bytes(), relative_prefix, resolve)
}

fn validate_node(
    team: ed25519_dalek::VerifyingKey,
    request: &PatchRepairRequest<InventoryComponent>,
    node: &PatchNode<InventoryLeafValue>,
) -> Result<()> {
    let component = *request.scope();
    let base = component.base_prefix(team);
    validate_patch_node(
        request,
        component.key_len(),
        base.as_bytes(),
        node,
        |absolute, value| {
            if value.component() != component {
                bail!("inventory leaf value does not match requested component");
            }
            match value {
                InventoryLeafValue::Peer => {
                    let bytes: [u8; PEER_EVIDENCE_BYTES_LEN] = absolute
                        .try_into()
                        .expect("validated PEER absolute key has fixed length");
                    PeerEvidence::from_bytes(bytes).context("decode PEER inventory key")?;
                }
                InventoryLeafValue::CollectionRecord(record) => {
                    if record.id().raw().as_slice() != absolute {
                        bail!("collection record body does not match its inventory key");
                    }
                }
                InventoryLeafValue::CapabilityProof(proof) => {
                    if proof.id().raw.as_slice() != absolute {
                        bail!("capability proof body does not match its inventory key");
                    }
                }
                InventoryLeafValue::Blob => {}
            }
            Ok(())
        },
    )
}

async fn send_leaf_value<W: AsyncWrite + Unpin>(
    send: &mut W,
    value: &InventoryLeafValue,
) -> Result<()> {
    match value {
        InventoryLeafValue::Peer => Ok(()),
        InventoryLeafValue::CollectionRecord(record) => {
            let bytes = record.to_bytes();
            send_u16_be(
                send,
                u16::try_from(bytes.len()).expect("collection record fits u16"),
            )
            .await?;
            send.write_all(&bytes)
                .await
                .map_err(|error| anyhow!("send collection record: {error}"))
        }
        InventoryLeafValue::CapabilityProof(proof) => {
            let bytes = proof.as_bytes();
            send_u16_be(
                send,
                u16::try_from(bytes.len()).expect("bounded proof fits u16"),
            )
            .await?;
            send.write_all(bytes)
                .await
                .map_err(|error| anyhow!("send capability proof: {error}"))
        }
        InventoryLeafValue::Blob => Ok(()),
    }
}

async fn recv_leaf_value<R: AsyncRead + Unpin>(
    recv: &mut R,
    component: InventoryComponent,
) -> Result<InventoryLeafValue> {
    match component {
        InventoryComponent::Peer => Ok(InventoryLeafValue::Peer),
        InventoryComponent::CollectionRecord => {
            let bytes = recv_bounded_u16_frame(recv, RECORD_MAX_BYTES, "collection record").await?;
            let record = CollectionRecord::from_bytes(&bytes)
                .context("decode canonical collection record")?;
            if record.to_bytes() != bytes {
                bail!("collection record did not round-trip canonically");
            }
            Ok(InventoryLeafValue::CollectionRecord(record))
        }
        InventoryComponent::CapabilityProof => {
            let bytes = recv_bounded_u16_frame(recv, PROOF_MAX_BYTES, "capability proof").await?;
            let proof =
                CapabilityProof::from_bytes(&bytes).context("decode canonical capability proof")?;
            if proof.as_bytes() != bytes {
                bail!("capability proof did not round-trip canonically");
            }
            Ok(InventoryLeafValue::CapabilityProof(proof))
        }
        InventoryComponent::Blob => Ok(InventoryLeafValue::Blob),
    }
}

/// Send a node, unavailable-snapshot response, or absent-prefix response.
pub(crate) async fn send_node_response<W: AsyncWrite + Unpin>(
    send: &mut W,
    team: ed25519_dalek::VerifyingKey,
    request: &PatchRepairRequest<InventoryComponent>,
    response: &PatchNodeResponse<InventoryLeafValue>,
) -> Result<()> {
    let node = match response {
        PatchNodeResponse::SnapshotUnavailable => {
            return send_u8(send, NODE_SNAPSHOT_UNAVAILABLE).await;
        }
        PatchNodeResponse::PrefixAbsent => {
            return send_u8(send, NODE_PREFIX_ABSENT).await;
        }
        PatchNodeResponse::Found(node) => node,
    };
    validate_node(team, request, node)?;
    send_u8(send, NODE_FOUND).await?;
    match node {
        PatchNode::Leaf { digest, leaf } => {
            send_u8(send, NODE_LEAF).await?;
            send_hash(send, digest).await?;
            send_u64_be(send, 1).await?;
            send.write_all(&leaf.key)
                .await
                .map_err(|error| anyhow!("send inventory leaf key: {error}"))?;
            send_leaf_value(send, &leaf.value).await
        }
        PatchNode::Branch {
            digest,
            leaf_count,
            branch,
        } => {
            send_u8(send, NODE_BRANCH).await?;
            send_hash(send, digest).await?;
            send_u64_be(send, *leaf_count).await?;
            send.write_all(&branch.representative)
                .await
                .map_err(|error| anyhow!("send inventory representative: {error}"))?;
            send_u8(send, branch.end_depth).await?;
            send_u16_be(
                send,
                u16::try_from(branch.children.len()).expect("PATCH fanout is at most 256"),
            )
            .await?;
            for child in &branch.children {
                send_u8(send, child.edge).await?;
                send_hash(send, &child.digest).await?;
                send_u64_be(send, child.leaf_count).await?;
            }
            Ok(())
        }
    }
}

async fn recv_node_response<R: AsyncRead + Unpin>(
    recv: &mut R,
    team: ed25519_dalek::VerifyingKey,
    request: &PatchRepairRequest<InventoryComponent>,
) -> Result<PatchNodeResponse<InventoryLeafValue>> {
    let kind = match recv_u8(recv).await? {
        NODE_SNAPSHOT_UNAVAILABLE => {
            require_eof(recv).await?;
            return Ok(PatchNodeResponse::SnapshotUnavailable);
        }
        NODE_PREFIX_ABSENT => {
            require_eof(recv).await?;
            return Ok(PatchNodeResponse::PrefixAbsent);
        }
        NODE_FOUND => recv_u8(recv).await?,
        status => bail!("unknown inventory node response {status:#x}"),
    };
    let digest = recv_hash(recv).await?;
    let leaf_count = recv_u64_be(recv).await?;
    let component = *request.scope();
    let relative_key_len = component.relative_key_len(component.base_prefix(team));
    let node = match kind {
        NODE_LEAF => {
            if leaf_count != 1 {
                bail!("inventory leaf advertises {leaf_count} leaves");
            }
            let key = recv_exact_vec(recv, relative_key_len, "inventory leaf key").await?;
            let value = recv_leaf_value(recv, component).await?;
            PatchNode::Leaf {
                digest,
                leaf: PatchLeaf { key, value },
            }
        }
        NODE_BRANCH => {
            let representative =
                recv_exact_vec(recv, relative_key_len, "inventory representative").await?;
            let end_depth = recv_u8(recv).await?;
            let child_count = recv_u16_be(recv).await? as usize;
            if !(2..=256).contains(&child_count) {
                bail!("inventory branch fanout is not canonical");
            }
            let mut children = Vec::new();
            children
                .try_reserve_exact(child_count)
                .map_err(|error| anyhow!("cannot allocate inventory children: {error}"))?;
            for _ in 0..child_count {
                children.push(PatchChild {
                    edge: recv_u8(recv).await?,
                    digest: recv_hash(recv).await?,
                    leaf_count: recv_u64_be(recv).await?,
                });
            }
            PatchNode::Branch {
                digest,
                leaf_count,
                branch: PatchBranch {
                    representative,
                    end_depth,
                    children,
                },
            }
        }
        other => bail!("unknown inventory node kind {other:#x}"),
    };
    require_eof(recv).await?;
    validate_node(team, request, &node)?;
    Ok(PatchNodeResponse::Found(node))
}

/// Fetch one node from an exact immutable component snapshot.
pub(crate) async fn op_inventory_node<C: Conn>(
    connection: &C,
    team: ed25519_dalek::VerifyingKey,
    request: &PatchRepairRequest<InventoryComponent>,
) -> Result<PatchNodeResponse<InventoryLeafValue>> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory node stream: {error}"))?;
    send_node_request(&mut send, request).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory node request: {error}"))?;
    recv_node_response(&mut recv, team, request).await
}

/// Exact request for one bounded range of a pinned inventory blob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InventoryBlobRangeRequest {
    /// Exact non-empty Blob component root advertised by the manifest.
    pub(crate) root: [u8; 32],
    /// Manifest count committed by `root`.
    pub(crate) leaf_count: u64,
    pub(crate) hash: [u8; 32],
    pub(crate) offset: u64,
    pub(crate) maximum: u32,
}

impl InventoryBlobRangeRequest {
    pub(crate) fn new(
        root: [u8; 32],
        leaf_count: u64,
        hash: [u8; 32],
        offset: u64,
        maximum: u32,
    ) -> Result<Self> {
        if leaf_count == 0 {
            bail!("a blob range request cannot pin an empty component");
        }
        if maximum == 0 || maximum as usize > BLOB_TRANSFER_CHUNK_BYTES {
            bail!(
                "inventory blob range limit is {maximum}; expected 1..={BLOB_TRANSFER_CHUNK_BYTES}"
            );
        }
        Ok(Self {
            root,
            leaf_count,
            hash,
            offset,
            maximum,
        })
    }
}

async fn send_blob_range_request<W: AsyncWrite + Unpin>(
    send: &mut W,
    request: InventoryBlobRangeRequest,
) -> Result<()> {
    send_u8(send, OP_INVENTORY_BLOB_RANGE).await?;
    send_hash(send, &request.root).await?;
    send_u64_be(send, request.leaf_count).await?;
    send_hash(send, &request.hash).await?;
    send_u64_be(send, request.offset).await?;
    send_u32_be(send, request.maximum).await
}

/// Decode a bounded blob range request for the authorized team session.
pub(crate) async fn recv_blob_range_request<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<InventoryBlobRangeRequest> {
    let root = recv_hash(recv).await?;
    let leaf_count = recv_u64_be(recv).await?;
    let hash = recv_hash(recv).await?;
    let offset = recv_u64_be(recv).await?;
    let maximum = recv_u32_be(recv).await?;
    require_eof(recv).await?;
    InventoryBlobRangeRequest::new(root, leaf_count, hash, offset, maximum)
}

/// Result of one bounded mirror range request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InventoryBlobRangeResponse {
    Chunk { total_length: u64, bytes: Vec<u8> },
    SnapshotUnavailable,
    NotInSnapshot,
}

/// Serve a bounded range from bytes already verified as a member of the pinned
/// Blob component.
pub(crate) async fn send_blob_range<W: AsyncWrite + Unpin>(
    send: &mut W,
    request: InventoryBlobRangeRequest,
    bytes: &[u8],
) -> Result<()> {
    let total_length = bytes.len() as u64;
    if request.offset > total_length {
        bail!("inventory blob range starts past the end of the blob");
    }
    let start = usize::try_from(request.offset)
        .map_err(|_| anyhow!("inventory blob offset does not fit this address space"))?;
    let end = start
        .saturating_add(request.maximum as usize)
        .min(bytes.len());
    let chunk = &bytes[start..end];
    send_u8(send, BLOB_RANGE_FOUND).await?;
    send_u64_be(send, total_length).await?;
    send_u32_be(
        send,
        u32::try_from(chunk.len()).expect("bounded blob range fits u32"),
    )
    .await?;
    send.write_all(chunk)
        .await
        .map_err(|error| anyhow!("send inventory blob range: {error}"))
}

pub(crate) async fn send_blob_snapshot_unavailable<W: AsyncWrite + Unpin>(
    send: &mut W,
) -> Result<()> {
    send_u8(send, BLOB_RANGE_SNAPSHOT_UNAVAILABLE).await
}

pub(crate) async fn send_blob_not_in_snapshot<W: AsyncWrite + Unpin>(send: &mut W) -> Result<()> {
    send_u8(send, BLOB_RANGE_NOT_IN_SNAPSHOT).await
}

async fn recv_blob_range<R: AsyncRead + Unpin>(
    recv: &mut R,
    request: InventoryBlobRangeRequest,
) -> Result<InventoryBlobRangeResponse> {
    match recv_u8(recv).await? {
        BLOB_RANGE_SNAPSHOT_UNAVAILABLE => {
            require_eof(recv).await?;
            Ok(InventoryBlobRangeResponse::SnapshotUnavailable)
        }
        BLOB_RANGE_NOT_IN_SNAPSHOT => {
            require_eof(recv).await?;
            Ok(InventoryBlobRangeResponse::NotInSnapshot)
        }
        BLOB_RANGE_FOUND => {
            let total_length = recv_u64_be(recv).await?;
            let length = recv_u32_be(recv).await? as usize;
            if length > request.maximum as usize || length > BLOB_TRANSFER_CHUNK_BYTES {
                bail!("inventory blob response exceeds the requested range bound");
            }
            if request.offset > total_length
                || request.offset.saturating_add(length as u64) > total_length
            {
                bail!("inventory blob response range exceeds its total length");
            }
            let bytes = recv_exact_vec(recv, length, "inventory blob range").await?;
            require_eof(recv).await?;
            Ok(InventoryBlobRangeResponse::Chunk {
                total_length,
                bytes,
            })
        }
        status => bail!("unknown inventory blob range response {status:#x}"),
    }
}

/// Fetch one bounded range from a blob in an exact pinned inventory.
pub(crate) async fn op_inventory_blob_range<C: Conn>(
    connection: &C,
    request: InventoryBlobRangeRequest,
) -> Result<InventoryBlobRangeResponse> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory blob stream: {error}"))?;
    send_blob_range_request(&mut send, request).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory blob request: {error}"))?;
    recv_blob_range(&mut recv, request).await
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use tokio::io::{AsyncWriteExt, duplex};
    use triblespace_core::patch::PatchHash;

    use super::*;

    fn team() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    fn request(
        team: ed25519_dalek::VerifyingKey,
        component: InventoryComponent,
        root: [u8; 32],
        leaf_count: u64,
        prefix: Vec<u8>,
        expected_digest: [u8; 32],
    ) -> PatchRepairRequest<InventoryComponent> {
        let base = component.base_prefix(team);
        PatchRepairRequest::new(
            component,
            PatchSummary::new(Some(root), leaf_count).unwrap(),
            component.relative_key_len(base),
            prefix,
            expected_digest,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn manifest_codec_binds_team_roots_counts_and_generation() {
        let team = team();
        let components = InventoryComponent::ALL.map(|component| {
            let count = component as u64;
            ComponentManifest::new(
                component,
                PatchSummary::new(Some([component as u8; 32]), count).unwrap(),
            )
        });
        let manifest = InventoryManifest::new(team, components);
        let (mut writer, mut reader) = duplex(4096);
        send_manifest(&mut writer, &manifest).await.unwrap();
        writer.shutdown().await.unwrap();
        let decoded = recv_manifest(&mut reader, team).await.unwrap();
        assert_eq!(decoded, manifest);
    }

    #[tokio::test]
    async fn peer_leaf_uses_team_base_and_relative_peer_key() {
        let team = team();
        let peer = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let absolute = PeerEvidence::new(team, peer).to_bytes();
        let digest = <Blake3Merkle as PatchHash>::leaf(&absolute);
        let request = request(
            team,
            InventoryComponent::Peer,
            digest,
            1,
            Vec::new(),
            digest,
        );
        let response = PatchNodeResponse::Found(PatchNode::Leaf {
            digest,
            leaf: PatchLeaf {
                key: peer.to_bytes().to_vec(),
                value: InventoryLeafValue::Peer,
            },
        });
        let (mut writer, mut reader) = duplex(4096);
        send_node_response(&mut writer, team, &request, &response)
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        assert_eq!(
            recv_node_response(&mut reader, team, &request)
                .await
                .unwrap(),
            response
        );
    }

    #[tokio::test]
    async fn branch_codec_recomputes_authenticated_shape() {
        let team = team();
        let first_key = [0x10; 32];
        let second_key = [0x20; 32];
        let first_digest = <Blake3Merkle as PatchHash>::leaf(&first_key);
        let second_digest = <Blake3Merkle as PatchHash>::leaf(&second_key);
        let children = vec![
            PatchChild {
                edge: 0x10,
                digest: first_digest,
                leaf_count: 1,
            },
            PatchChild {
                edge: 0x20,
                digest: second_digest,
                leaf_count: 1,
            },
        ];
        let tree_to_key: Vec<usize> = (0..first_key.len()).collect();
        let mut state =
            <Blake3Merkle as PatchHash>::begin_branch(&first_key, &tree_to_key, 0, 2, 2);
        for child in &children {
            <Blake3Merkle as PatchHash>::push_child(
                &mut state,
                child.edge,
                child.leaf_count,
                child.digest,
            );
        }
        let digest = <Blake3Merkle as PatchHash>::finish_branch(state);
        let request = request(
            team,
            InventoryComponent::Blob,
            digest,
            2,
            Vec::new(),
            digest,
        );
        let response = PatchNodeResponse::Found(PatchNode::Branch {
            digest,
            leaf_count: 2,
            branch: PatchBranch {
                representative: first_key.to_vec(),
                end_depth: 0,
                children,
            },
        });
        let (mut writer, mut reader) = duplex(4096);
        send_node_response(&mut writer, team, &request, &response)
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        assert_eq!(
            recv_node_response(&mut reader, team, &request)
                .await
                .unwrap(),
            response
        );
    }

    #[tokio::test]
    async fn malformed_branch_counts_are_rejected_before_send() {
        let team = team();
        let request = request(
            team,
            InventoryComponent::Blob,
            [9; 32],
            3,
            Vec::new(),
            [9; 32],
        );
        let response = PatchNodeResponse::Found(PatchNode::Branch {
            digest: [9; 32],
            leaf_count: 3,
            branch: PatchBranch {
                representative: vec![0; 32],
                end_depth: 0,
                children: vec![
                    PatchChild {
                        edge: 0,
                        digest: [1; 32],
                        leaf_count: 1,
                    },
                    PatchChild {
                        edge: 1,
                        digest: [2; 32],
                        leaf_count: 1,
                    },
                ],
            },
        });
        let (mut writer, _reader) = duplex(4096);
        assert!(
            send_node_response(&mut writer, team, &request, &response)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn redistributed_child_counts_break_the_authenticated_descriptor() {
        let team = team();
        let representative = [0x10; 32];
        let mut children = vec![
            PatchChild {
                edge: 0x10,
                digest: [1; 32],
                leaf_count: 3,
            },
            PatchChild {
                edge: 0x20,
                digest: [2; 32],
                leaf_count: 1,
            },
        ];
        let tree_to_key: Vec<usize> = (0..representative.len()).collect();
        let mut state = <Blake3Merkle as PatchHash>::begin_branch(
            &representative,
            &tree_to_key,
            0,
            children.len(),
            4,
        );
        for child in &children {
            <Blake3Merkle as PatchHash>::push_child(
                &mut state,
                child.edge,
                child.leaf_count,
                child.digest,
            );
        }
        let digest = <Blake3Merkle as PatchHash>::finish_branch(state);
        let request = request(
            team,
            InventoryComponent::Blob,
            digest,
            4,
            Vec::new(),
            digest,
        );

        children[0].leaf_count = 2;
        children[1].leaf_count = 2;
        let forged = PatchNodeResponse::Found(PatchNode::Branch {
            digest,
            leaf_count: 4,
            branch: PatchBranch {
                representative: representative.to_vec(),
                end_depth: 0,
                children,
            },
        });
        let (mut writer, _reader) = duplex(4096);
        assert!(
            send_node_response(&mut writer, team, &request, &forged)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn blob_ranges_are_bounded_and_report_total_length() {
        let request = InventoryBlobRangeRequest::new([5; 32], 1, [6; 32], 3, 4).unwrap();
        let (mut writer, mut reader) = duplex(4096);
        send_blob_range(&mut writer, request, b"abcdefghij")
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        assert_eq!(
            recv_blob_range(&mut reader, request).await.unwrap(),
            InventoryBlobRangeResponse::Chunk {
                total_length: 10,
                bytes: b"defg".to_vec(),
            }
        );
        assert!(
            InventoryBlobRangeRequest::new(
                [5; 32],
                1,
                [6; 32],
                0,
                BLOB_TRANSFER_CHUNK_BYTES as u32 + 1,
            )
            .is_err()
        );
    }
}
