//! Bounded wire codec for authorized prefix-Merkle reconciliation.
//!
//! CONNECT is still the mandatory first stream on a connection. A successful
//! [`OP_INVENTORY_AUTH`] then installs exactly one server-selected inventory
//! view for that connection. Manifest, node, exact `GET_BLOB`, and mirror
//! range operations are rejected by the host until that second authorization
//! succeeds. The proof is not repeated on every request.
//!
//! Every node and range request pins a view-scoped component token. If that
//! immutable snapshot is no longer cached, the server returns an explicit
//! unavailable response; it never serves bytes from its current snapshot as a
//! fallback.

use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{
    CapabilityProof, CapabilityProofBundle, MAX_CAPABILITY_PROOF_STEPS,
};
use triblespace_core::collection::{COLLECTION_COMMIT_BYTES_LEN, CollectionRecord};
use triblespace_core::patch::{Blake3Merkle, PatchHash};
use triblespace_core::repo::peer::{PEER_EVIDENCE_BYTES_LEN, PeerEvidence};

use crate::inventory::{
    AuthorizedInventoryView, AuthorizedViewId, ComponentManifest, InventoryComponent,
    InventoryGeneration, InventoryManifest, InventoryProjection, InventoryToken,
};
use crate::protocol::{
    recv_capability_proof_bundle, recv_hash, recv_u8, recv_u32_be, recv_u64_be,
    send_capability_proof_bundle, send_hash, send_u8, send_u32_be, send_u64_be,
};
use crate::transport::Conn;

/// Authorize exactly one server-selected team inventory view on this
/// CONNECT-authenticated connection.
pub(crate) const OP_INVENTORY_AUTH: u8 = 0x08;
/// Fetch the four component roots and immutable per-component tokens.
pub(crate) const OP_INVENTORY_MANIFEST: u8 = 0x09;
/// Fetch one authenticated PATCH node under a protocol-relative prefix.
pub(crate) const OP_INVENTORY_NODE: u8 = 0x0A;
/// Fetch one bounded range of a blob present in a pinned Blob inventory.
pub(crate) const OP_INVENTORY_BLOB_RANGE: u8 = 0x0B;

pub(crate) const INVENTORY_AUTH_OK: u8 = 0x00;
pub(crate) const INVENTORY_AUTH_REJECTED: u8 = 0x01;

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

/// Client-side second authorization exchange.
pub(crate) async fn op_inventory_auth<C: Conn>(
    connection: &C,
    team: ed25519_dalek::VerifyingKey,
    proof: &CapabilityProofBundle,
) -> Result<AuthorizedInventoryView> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory auth stream: {error}"))?;
    send_u8(&mut send, OP_INVENTORY_AUTH).await?;
    send_capability_proof_bundle(&mut send, proof).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory auth request: {error}"))?;

    match recv_u8(&mut recv).await? {
        INVENTORY_AUTH_OK => {
            let projection = InventoryProjection::from_wire_tag(recv_u8(&mut recv).await?)?;
            let id = AuthorizedViewId::from_bytes(recv_hash(&mut recv).await?);
            require_eof(&mut recv).await?;
            let expected = AuthorizedInventoryView::full_team(team);
            if projection != expected.projection() || id != expected.id() {
                bail!("server selected an unexpected inventory projection");
            }
            Ok(expected)
        }
        INVENTORY_AUTH_REJECTED => {
            require_eof(&mut recv).await?;
            bail!("server rejected SYNC_TEAM proof bundle")
        }
        status => bail!("unknown inventory auth response {status:#x}"),
    }
}

/// Decode the one proof bundle carried by [`OP_INVENTORY_AUTH`].
pub(crate) async fn recv_inventory_auth_request<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CapabilityProofBundle> {
    let proof = recv_capability_proof_bundle(recv).await?;
    require_eof(recv).await?;
    Ok(proof)
}

/// Send a successful second-authorization response and the selected view.
pub(crate) async fn send_inventory_auth_ok<W: AsyncWrite + Unpin>(
    send: &mut W,
    view: AuthorizedInventoryView,
) -> Result<()> {
    send_u8(send, INVENTORY_AUTH_OK).await?;
    send_u8(send, view.projection().wire_tag()).await?;
    send_hash(send, &view.id().into_bytes()).await
}

/// Send a rejected second-authorization response without leaking details.
pub(crate) async fn send_inventory_auth_rejected<W: AsyncWrite + Unpin>(
    send: &mut W,
) -> Result<()> {
    send_u8(send, INVENTORY_AUTH_REJECTED).await
}

async fn send_view_request<W: AsyncWrite + Unpin>(
    send: &mut W,
    operation: u8,
    view: AuthorizedViewId,
) -> Result<()> {
    send_u8(send, operation).await?;
    send_hash(send, &view.into_bytes()).await
}

/// Decode and verify the selected view on a manifest request.
pub(crate) async fn recv_manifest_request<R: AsyncRead + Unpin>(
    recv: &mut R,
    authorized: AuthorizedInventoryView,
) -> Result<()> {
    let requested = AuthorizedViewId::from_bytes(recv_hash(recv).await?);
    require_eof(recv).await?;
    if requested != authorized.id() {
        bail!("manifest request does not name the authorized inventory view");
    }
    Ok(())
}

/// Encode one exact four-component manifest.
pub(crate) async fn send_manifest<W: AsyncWrite + Unpin>(
    send: &mut W,
    manifest: &InventoryManifest,
) -> Result<()> {
    send_hash(send, &manifest.view().into_bytes()).await?;
    send_hash(send, &manifest.generation().into_bytes()).await?;
    for component in manifest.components() {
        send_u8(send, component.component() as u8).await?;
        send_hash(send, &component.token().into_bytes()).await?;
        send_u64_be(send, component.leaf_count()).await?;
        match component.root() {
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
    expected_view: AuthorizedViewId,
) -> Result<InventoryManifest> {
    let view = AuthorizedViewId::from_bytes(recv_hash(recv).await?);
    if view != expected_view {
        bail!("manifest response does not match the authorized view");
    }
    let generation = InventoryGeneration::from_bytes(recv_hash(recv).await?);
    let mut components = Vec::with_capacity(InventoryComponent::ALL.len());
    for expected_component in InventoryComponent::ALL {
        let component = InventoryComponent::from_byte(recv_u8(recv).await?)?;
        if component != expected_component {
            bail!("manifest components are out of canonical order");
        }
        let token = InventoryToken::from_bytes(recv_hash(recv).await?);
        let leaf_count = recv_u64_be(recv).await?;
        let root = match recv_u8(recv).await? {
            0 => None,
            1 => Some(recv_hash(recv).await?),
            marker => bail!("invalid inventory root marker {marker:#x}"),
        };
        if root.is_none() != (leaf_count == 0) {
            bail!("empty inventory root and leaf count disagree");
        }
        components.push(ComponentManifest::from_wire(
            view, component, leaf_count, root, token,
        )?);
    }
    require_eof(recv).await?;
    let components: [ComponentManifest; 4] = components
        .try_into()
        .expect("the canonical component loop has exactly four entries");
    InventoryManifest::from_wire(view, generation, components)
}

/// Fetch an authenticated root manifest for the connection's selected view.
pub(crate) async fn op_inventory_manifest<C: Conn>(
    connection: &C,
    view: AuthorizedInventoryView,
) -> Result<InventoryManifest> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory manifest stream: {error}"))?;
    send_view_request(&mut send, OP_INVENTORY_MANIFEST, view.id()).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory manifest request: {error}"))?;
    recv_manifest(&mut recv, view.id()).await
}

/// Exact request for one node in a pinned component tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InventoryNodeRequest {
    pub(crate) view: AuthorizedViewId,
    pub(crate) component: InventoryComponent,
    pub(crate) token: InventoryToken,
    /// Prefix relative to the authorized component base.
    pub(crate) prefix: Vec<u8>,
    /// Digest the caller expects at this exact prefix.
    pub(crate) expected_digest: [u8; 32],
}

impl InventoryNodeRequest {
    pub(crate) fn new(
        view: AuthorizedInventoryView,
        component: InventoryComponent,
        token: InventoryToken,
        prefix: Vec<u8>,
        expected_digest: [u8; 32],
    ) -> Result<Self> {
        let maximum = component.relative_key_len(view.base_prefix(component));
        if prefix.len() > maximum {
            bail!(
                "inventory prefix is {} bytes; component-relative key is {maximum} bytes",
                prefix.len()
            );
        }
        Ok(Self {
            view: view.id(),
            component,
            token,
            prefix,
            expected_digest,
        })
    }
}

async fn send_node_request<W: AsyncWrite + Unpin>(
    send: &mut W,
    request: &InventoryNodeRequest,
) -> Result<()> {
    send_view_request(send, OP_INVENTORY_NODE, request.view).await?;
    send_u8(send, request.component as u8).await?;
    send_hash(send, &request.token.into_bytes()).await?;
    send_u8(
        send,
        u8::try_from(request.prefix.len()).expect("inventory keys are at most 64 bytes"),
    )
    .await?;
    send.write_all(&request.prefix)
        .await
        .map_err(|error| anyhow!("send inventory prefix: {error}"))?;
    send_hash(send, &request.expected_digest).await
}

/// Decode one node request relative to the connection's authorized base.
pub(crate) async fn recv_node_request<R: AsyncRead + Unpin>(
    recv: &mut R,
    authorized: AuthorizedInventoryView,
) -> Result<InventoryNodeRequest> {
    let view = AuthorizedViewId::from_bytes(recv_hash(recv).await?);
    if view != authorized.id() {
        bail!("node request does not name the authorized inventory view");
    }
    let component = InventoryComponent::from_byte(recv_u8(recv).await?)?;
    let token = InventoryToken::from_bytes(recv_hash(recv).await?);
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
    Ok(InventoryNodeRequest {
        view,
        component,
        token,
        prefix,
        expected_digest,
    })
}

/// Value carried by one inventory leaf. Keys remain the only hashed input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InventoryLeafValue {
    /// PEER's relative key is the peer public key; the fixed team base
    /// reconstructs the complete evidence body.
    Peer,
    CollectionRecord(CollectionRecord),
    CapabilityProof(CapabilityProof),
    /// Storage-observed length hint. Full bytes and BLAKE3 are still verified
    /// before admission.
    BlobLength(u64),
}

impl InventoryLeafValue {
    fn component(&self) -> InventoryComponent {
        match self {
            Self::Peer => InventoryComponent::Peer,
            Self::CollectionRecord(_) => InventoryComponent::CollectionRecord,
            Self::CapabilityProof(_) => InventoryComponent::CapabilityProof,
            Self::BlobLength(_) => InventoryComponent::Blob,
        }
    }
}

/// One canonical PATCH leaf.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InventoryLeaf {
    /// Complete key relative to the authorized component base.
    pub(crate) key: Vec<u8>,
    pub(crate) value: InventoryLeafValue,
}

/// Authenticated summary of one child edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InventoryChild {
    pub(crate) edge: u8,
    pub(crate) digest: [u8; 32],
    pub(crate) leaf_count: u64,
}

/// One canonical PATCH branch. Depth and representative are relative to the
/// authorized component base; child edges are strictly ascending.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InventoryBranch {
    pub(crate) representative: Vec<u8>,
    pub(crate) end_depth: u8,
    pub(crate) children: Vec<InventoryChild>,
}

/// One authenticated PATCH node returned by the server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InventoryNode {
    Leaf {
        digest: [u8; 32],
        leaf: InventoryLeaf,
    },
    Branch {
        digest: [u8; 32],
        leaf_count: u64,
        branch: InventoryBranch,
    },
}

impl InventoryNode {
    pub(crate) const fn digest(&self) -> [u8; 32] {
        match self {
            Self::Leaf { digest, .. } | Self::Branch { digest, .. } => *digest,
        }
    }

    pub(crate) const fn leaf_count(&self) -> u64 {
        match self {
            Self::Leaf { .. } => 1,
            Self::Branch { leaf_count, .. } => *leaf_count,
        }
    }
}

/// Result of one pinned node lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InventoryNodeResponse {
    Found(InventoryNode),
    /// The exact `(view, component, token)` snapshot is no longer cached.
    SnapshotUnavailable,
    /// No node exists at the requested prefix in the pinned snapshot.
    PrefixAbsent,
}

fn validate_node(
    view: AuthorizedInventoryView,
    request: &InventoryNodeRequest,
    node: &InventoryNode,
) -> Result<()> {
    if request.view != view.id() {
        bail!("node request view does not match the authorized view");
    }
    if node.digest() != request.expected_digest {
        bail!("inventory node digest does not match the requested digest");
    }
    let component = request.component;
    let base = view.base_prefix(component);
    let relative_key_len = component.relative_key_len(base);
    match node {
        InventoryNode::Leaf { digest, leaf } => {
            if leaf.value.component() != component {
                bail!("inventory leaf value does not match requested component");
            }
            if leaf.key.len() != relative_key_len || !leaf.key.starts_with(&request.prefix) {
                bail!("inventory leaf key is outside the requested relative prefix");
            }
            let absolute = base.absolute_key(component, &leaf.key)?;
            let expected = <Blake3Merkle as PatchHash>::leaf(&absolute);
            if digest != &expected {
                bail!("inventory leaf digest does not bind its full PATCH key");
            }
            match &leaf.value {
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
                InventoryLeafValue::BlobLength(_) => {}
            }
        }
        InventoryNode::Branch {
            digest,
            leaf_count,
            branch,
        } => {
            let end_depth = branch.end_depth as usize;
            if branch.representative.len() != relative_key_len
                || !branch.representative.starts_with(&request.prefix)
            {
                bail!("inventory branch representative is outside the requested prefix");
            }
            if end_depth < request.prefix.len() || end_depth >= relative_key_len {
                bail!("inventory branch end depth is outside its relative key");
            }
            if !(2..=256).contains(&branch.children.len()) {
                bail!("inventory branch fanout is not canonical");
            }
            if branch.children[0].edge != branch.representative[end_depth] {
                bail!("inventory branch representative is not in its first child");
            }
            let mut previous = None;
            let mut summed_count = 0u64;
            for child in &branch.children {
                if previous.is_some_and(|edge| child.edge <= edge) {
                    bail!("inventory branch child edges are not strictly ascending");
                }
                if child.leaf_count == 0 {
                    bail!("inventory branch child has zero leaves");
                }
                previous = Some(child.edge);
                summed_count = summed_count
                    .checked_add(child.leaf_count)
                    .ok_or_else(|| anyhow!("inventory branch leaf count overflow"))?;
            }
            if summed_count != *leaf_count {
                bail!("inventory branch child counts do not match its leaf count");
            }
            let absolute_end_depth = base.as_bytes().len() + end_depth;
            let mut state = <Blake3Merkle as PatchHash>::begin_branch(
                component.key_len(),
                absolute_end_depth,
                branch.children.len(),
                *leaf_count,
            );
            for child in &branch.children {
                <Blake3Merkle as PatchHash>::push_child(&mut state, child.edge, child.digest);
            }
            let expected = <Blake3Merkle as PatchHash>::finish_branch(state);
            if digest != &expected {
                bail!("inventory branch digest does not bind its canonical child summaries");
            }
        }
    }
    Ok(())
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
        InventoryLeafValue::BlobLength(length) => send_u64_be(send, *length).await,
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
        InventoryComponent::Blob => Ok(InventoryLeafValue::BlobLength(recv_u64_be(recv).await?)),
    }
}

/// Send a node, stale-token response, or absent-prefix response.
pub(crate) async fn send_node_response<W: AsyncWrite + Unpin>(
    send: &mut W,
    view: AuthorizedInventoryView,
    request: &InventoryNodeRequest,
    response: &InventoryNodeResponse,
) -> Result<()> {
    let node = match response {
        InventoryNodeResponse::SnapshotUnavailable => {
            return send_u8(send, NODE_SNAPSHOT_UNAVAILABLE).await;
        }
        InventoryNodeResponse::PrefixAbsent => {
            return send_u8(send, NODE_PREFIX_ABSENT).await;
        }
        InventoryNodeResponse::Found(node) => node,
    };
    validate_node(view, request, node)?;
    send_u8(send, NODE_FOUND).await?;
    match node {
        InventoryNode::Leaf { digest, leaf } => {
            send_u8(send, NODE_LEAF).await?;
            send_hash(send, digest).await?;
            send_u64_be(send, 1).await?;
            send.write_all(&leaf.key)
                .await
                .map_err(|error| anyhow!("send inventory leaf key: {error}"))?;
            send_leaf_value(send, &leaf.value).await
        }
        InventoryNode::Branch {
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
    view: AuthorizedInventoryView,
    request: &InventoryNodeRequest,
) -> Result<InventoryNodeResponse> {
    let kind = match recv_u8(recv).await? {
        NODE_SNAPSHOT_UNAVAILABLE => {
            require_eof(recv).await?;
            return Ok(InventoryNodeResponse::SnapshotUnavailable);
        }
        NODE_PREFIX_ABSENT => {
            require_eof(recv).await?;
            return Ok(InventoryNodeResponse::PrefixAbsent);
        }
        NODE_FOUND => recv_u8(recv).await?,
        status => bail!("unknown inventory node response {status:#x}"),
    };
    let digest = recv_hash(recv).await?;
    let leaf_count = recv_u64_be(recv).await?;
    let relative_key_len = request
        .component
        .relative_key_len(view.base_prefix(request.component));
    let node = match kind {
        NODE_LEAF => {
            if leaf_count != 1 {
                bail!("inventory leaf advertises {leaf_count} leaves");
            }
            let key = recv_exact_vec(recv, relative_key_len, "inventory leaf key").await?;
            let value = recv_leaf_value(recv, request.component).await?;
            InventoryNode::Leaf {
                digest,
                leaf: InventoryLeaf { key, value },
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
                children.push(InventoryChild {
                    edge: recv_u8(recv).await?,
                    digest: recv_hash(recv).await?,
                    leaf_count: recv_u64_be(recv).await?,
                });
            }
            InventoryNode::Branch {
                digest,
                leaf_count,
                branch: InventoryBranch {
                    representative,
                    end_depth,
                    children,
                },
            }
        }
        other => bail!("unknown inventory node kind {other:#x}"),
    };
    require_eof(recv).await?;
    validate_node(view, request, &node)?;
    Ok(InventoryNodeResponse::Found(node))
}

/// Fetch one node from an exact immutable component snapshot.
pub(crate) async fn op_inventory_node<C: Conn>(
    connection: &C,
    view: AuthorizedInventoryView,
    request: &InventoryNodeRequest,
) -> Result<InventoryNodeResponse> {
    if request.view != view.id() {
        bail!("node request does not name the connection's authorized view");
    }
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open inventory node stream: {error}"))?;
    send_node_request(&mut send, request).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish inventory node request: {error}"))?;
    recv_node_response(&mut recv, view, request).await
}

/// Exact request for one bounded range of a pinned inventory blob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InventoryBlobRangeRequest {
    pub(crate) view: AuthorizedViewId,
    pub(crate) token: InventoryToken,
    pub(crate) hash: [u8; 32],
    pub(crate) offset: u64,
    pub(crate) maximum: u32,
}

impl InventoryBlobRangeRequest {
    pub(crate) fn new(
        view: AuthorizedInventoryView,
        token: InventoryToken,
        hash: [u8; 32],
        offset: u64,
        maximum: u32,
    ) -> Result<Self> {
        if maximum == 0 || maximum as usize > BLOB_TRANSFER_CHUNK_BYTES {
            bail!(
                "inventory blob range limit is {maximum}; expected 1..={BLOB_TRANSFER_CHUNK_BYTES}"
            );
        }
        Ok(Self {
            view: view.id(),
            token,
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
    send_view_request(send, OP_INVENTORY_BLOB_RANGE, request.view).await?;
    send_hash(send, &request.token.into_bytes()).await?;
    send_hash(send, &request.hash).await?;
    send_u64_be(send, request.offset).await?;
    send_u32_be(send, request.maximum).await
}

/// Decode a bounded blob range request for the authorized view.
pub(crate) async fn recv_blob_range_request<R: AsyncRead + Unpin>(
    recv: &mut R,
    authorized: AuthorizedInventoryView,
) -> Result<InventoryBlobRangeRequest> {
    let view = AuthorizedViewId::from_bytes(recv_hash(recv).await?);
    if view != authorized.id() {
        bail!("blob range request does not name the authorized inventory view");
    }
    let token = InventoryToken::from_bytes(recv_hash(recv).await?);
    let hash = recv_hash(recv).await?;
    let offset = recv_u64_be(recv).await?;
    let maximum = recv_u32_be(recv).await?;
    require_eof(recv).await?;
    InventoryBlobRangeRequest::new(authorized, token, hash, offset, maximum)
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
    view: AuthorizedInventoryView,
    request: InventoryBlobRangeRequest,
) -> Result<InventoryBlobRangeResponse> {
    if request.view != view.id() {
        bail!("blob range request does not name the connection's authorized view");
    }
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

    use super::*;

    fn view() -> AuthorizedInventoryView {
        AuthorizedInventoryView::full_team(SigningKey::from_bytes(&[1; 32]).verifying_key())
    }

    fn token(byte: u8) -> InventoryToken {
        InventoryToken::from_bytes([byte; 32])
    }

    #[tokio::test]
    async fn manifest_codec_binds_view_roots_counts_and_generation() {
        let view = view();
        let components = InventoryComponent::ALL.map(|component| {
            let count = component as u64;
            ComponentManifest::new(view.id(), component, count, Some([component as u8; 32]))
        });
        let manifest = InventoryManifest::new(view.id(), components);
        let (mut writer, mut reader) = duplex(4096);
        send_manifest(&mut writer, &manifest).await.unwrap();
        writer.shutdown().await.unwrap();
        let decoded = recv_manifest(&mut reader, view.id()).await.unwrap();
        assert_eq!(decoded, manifest);
    }

    #[tokio::test]
    async fn peer_leaf_uses_team_base_and_relative_peer_key() {
        let view = view();
        let peer = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let absolute = PeerEvidence::new(view.team(), peer).to_bytes();
        let digest = <Blake3Merkle as PatchHash>::leaf(&absolute);
        let request =
            InventoryNodeRequest::new(view, InventoryComponent::Peer, token(3), Vec::new(), digest)
                .unwrap();
        let response = InventoryNodeResponse::Found(InventoryNode::Leaf {
            digest,
            leaf: InventoryLeaf {
                key: peer.to_bytes().to_vec(),
                value: InventoryLeafValue::Peer,
            },
        });
        let (mut writer, mut reader) = duplex(4096);
        send_node_response(&mut writer, view, &request, &response)
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        assert_eq!(
            recv_node_response(&mut reader, view, &request)
                .await
                .unwrap(),
            response
        );
    }

    #[tokio::test]
    async fn branch_codec_recomputes_authenticated_shape() {
        let view = view();
        let first_key = [0x10; 32];
        let second_key = [0x20; 32];
        let first_digest = <Blake3Merkle as PatchHash>::leaf(&first_key);
        let second_digest = <Blake3Merkle as PatchHash>::leaf(&second_key);
        let children = vec![
            InventoryChild {
                edge: 0x10,
                digest: first_digest,
                leaf_count: 1,
            },
            InventoryChild {
                edge: 0x20,
                digest: second_digest,
                leaf_count: 1,
            },
        ];
        let mut state = <Blake3Merkle as PatchHash>::begin_branch(32, 0, 2, 2);
        for child in &children {
            <Blake3Merkle as PatchHash>::push_child(&mut state, child.edge, child.digest);
        }
        let digest = <Blake3Merkle as PatchHash>::finish_branch(state);
        let request =
            InventoryNodeRequest::new(view, InventoryComponent::Blob, token(4), Vec::new(), digest)
                .unwrap();
        let response = InventoryNodeResponse::Found(InventoryNode::Branch {
            digest,
            leaf_count: 2,
            branch: InventoryBranch {
                representative: first_key.to_vec(),
                end_depth: 0,
                children,
            },
        });
        let (mut writer, mut reader) = duplex(4096);
        send_node_response(&mut writer, view, &request, &response)
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        assert_eq!(
            recv_node_response(&mut reader, view, &request)
                .await
                .unwrap(),
            response
        );
    }

    #[tokio::test]
    async fn malformed_branch_counts_are_rejected_before_send() {
        let view = view();
        let request = InventoryNodeRequest::new(
            view,
            InventoryComponent::Blob,
            token(4),
            Vec::new(),
            [9; 32],
        )
        .unwrap();
        let response = InventoryNodeResponse::Found(InventoryNode::Branch {
            digest: [9; 32],
            leaf_count: 3,
            branch: InventoryBranch {
                representative: vec![0; 32],
                end_depth: 0,
                children: vec![
                    InventoryChild {
                        edge: 0,
                        digest: [1; 32],
                        leaf_count: 1,
                    },
                    InventoryChild {
                        edge: 1,
                        digest: [2; 32],
                        leaf_count: 1,
                    },
                ],
            },
        });
        let (mut writer, _reader) = duplex(4096);
        assert!(
            send_node_response(&mut writer, view, &request, &response)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn blob_ranges_are_bounded_and_report_total_length() {
        let view = view();
        let request = InventoryBlobRangeRequest::new(view, token(5), [6; 32], 3, 4).unwrap();
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
                view,
                token(5),
                [6; 32],
                0,
                BLOB_TRANSFER_CHUNK_BYTES as u32 + 1,
            )
            .is_err()
        );
    }
}
