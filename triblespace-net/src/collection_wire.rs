//! Dense wire frames for one READ-authorized collection repair session.
//!
//! One bidirectional stream pins one [`CollectionRepairManifest`]. The server
//! first presents its bounded READ(C) witness; only an admitting client sends
//! its own witness, then walks the record and WRITE-evidence PATCHes
//! interactively beneath those exact roots. The same
//! admitted stream may request an exact handle from that collection's resident
//! Full-replica disclosure forest. No global inventory, historical-root cache, repeated
//! authorization exchange, or blob list participates in this protocol.

use anybytes::Bytes;
use anyhow::{Result, anyhow, bail};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{CapabilityProofBundle, MAX_CAPABILITY_PROOF_BUNDLE_BYTES};
use triblespace_core::collection::CollectionHandle;

use crate::patch_repair::{
    PatchBranch, PatchChild, PatchLeaf, PatchNode, PatchNodeResponse, PatchRepairRequest,
    PatchSummary,
};
use crate::protocol::{
    recv_capability_proof_bundle, recv_hash, recv_u8, recv_u32_be, recv_u64_be,
    send_capability_proof_bundle, send_hash, send_u8, send_u32_be, send_u64_be,
};

/// Direct-RPC operation which opens one collection repair session.
///
/// `0x0D` was a pre-v17 provider-cover operation. The ALPN generation change
/// deliberately frees the byte for this clean-slate meaning.
pub(crate) const OP_COLLECTION_REPAIR: u8 = 0x0D;
pub(crate) const OP_COLLECTION_BLOB: u8 = 0x0E;

/// Maximum portable READ proof branches accepted at one session boundary.
pub(crate) const MAX_COLLECTION_READ_BUNDLES: usize = 16;
/// Aggregate bound across the length-prefixed READ proof frames.
pub(crate) const MAX_COLLECTION_READ_EVIDENCE_BYTES: usize =
    MAX_COLLECTION_READ_BUNDLES * MAX_CAPABILITY_PROOF_BUNDLE_BYTES;
/// Largest value transported by one authenticated PATCH leaf.
pub(crate) const MAX_COLLECTION_LEAF_BYTES: usize = MAX_CAPABILITY_PROOF_BUNDLE_BYTES;

const REPAIR_ADMITTED: u8 = 0x00;
const REPAIR_REJECTED: u8 = 0x01;
const REPAIR_UNAVAILABLE: u8 = 0x02;
const REPAIR_CHALLENGE: u8 = 0x03;

const REQUEST_NODE: u8 = 0x01;
const REQUEST_BLOB: u8 = 0x02;
const REQUEST_DONE: u8 = 0xFF;

const NODE_FOUND: u8 = 0x00;
const NODE_PREFIX_ABSENT: u8 = 0x01;
const NODE_LEAF: u8 = 0x00;
const NODE_BRANCH: u8 = 0x01;

/// One of the two grow-only PATCHes which determine collection activation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CollectionRepairComponent {
    Record,
    WriteEvidence,
    Resident,
}

impl CollectionRepairComponent {
    pub(crate) const fn key_len(self) -> usize {
        match self {
            Self::Record => 16,
            Self::WriteEvidence => 32,
            Self::Resident => 80,
        }
    }

    const fn wire(self) -> u8 {
        match self {
            Self::Record => 0,
            Self::WriteEvidence => 1,
            Self::Resident => 2,
        }
    }

    fn from_wire(byte: u8) -> Result<Self> {
        match byte {
            0 => Ok(Self::Record),
            1 => Ok(Self::WriteEvidence),
            2 => Ok(Self::Resident),
            other => bail!("unknown collection repair component {other:#x}"),
        }
    }
}

/// Client identity-bearing response after the server has proved READ(C).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CollectionRepairHello {
    pub(crate) read_evidence: Vec<CapabilityProofBundle>,
}

/// Exact activation state pinned for one accepted stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CollectionRepairManifest {
    pub(crate) wake_root: [u8; 32],
    pub(crate) records: PatchSummary,
    pub(crate) write_evidence: PatchSummary,
    pub(crate) resident: PatchSummary,
}

impl CollectionRepairManifest {
    pub(crate) const fn component(self, component: CollectionRepairComponent) -> PatchSummary {
        match component {
            CollectionRepairComponent::Record => self.records,
            CollectionRepairComponent::WriteEvidence => self.write_evidence,
            CollectionRepairComponent::Resident => self.resident,
        }
    }
}

/// Server decision at the READ(C) boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CollectionRepairAdmission {
    Admitted(CollectionRepairManifest),
    Rejected,
    Unavailable,
}

/// One client command after admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CollectionRepairCommand {
    Node {
        component: CollectionRepairComponent,
        prefix: Vec<u8>,
        expected_digest: [u8; 32],
    },
    Blob([u8; 32]),
    Done,
}

pub(crate) async fn send_repair_evidence<W: AsyncWrite + Unpin>(
    send: &mut W,
    read_evidence: &[CapabilityProofBundle],
) -> Result<()> {
    if read_evidence.len() > MAX_COLLECTION_READ_BUNDLES {
        bail!(
            "collection READ proof forest has {} bundles; limit is {}",
            read_evidence.len(),
            MAX_COLLECTION_READ_BUNDLES
        );
    }
    send_u32_be(
        send,
        u32::try_from(read_evidence.len()).expect("proof count bound fits u32"),
    )
    .await?;
    let mut aggregate = 0usize;
    for bundle in read_evidence {
        let length = bundle.to_bytes()?.len();
        aggregate = aggregate
            .checked_add(length)
            .ok_or_else(|| anyhow!("collection READ evidence length overflow"))?;
        if aggregate > MAX_COLLECTION_READ_EVIDENCE_BYTES {
            bail!(
                "collection READ evidence is {aggregate} bytes; limit is {MAX_COLLECTION_READ_EVIDENCE_BYTES}"
            );
        }
        send_capability_proof_bundle(send, bundle).await?;
    }
    Ok(())
}

/// Decode the body after the caller has already consumed
/// [`OP_COLLECTION_REPAIR`].
pub(crate) async fn recv_repair_hello<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CollectionRepairHello> {
    Ok(CollectionRepairHello {
        read_evidence: recv_repair_evidence(recv).await?,
    })
}

pub(crate) async fn recv_repair_collection<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CollectionHandle> {
    Ok(CollectionHandle::new(recv_hash(recv).await?))
}

pub(crate) async fn recv_repair_evidence<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<Vec<CapabilityProofBundle>> {
    let count = recv_u32_be(recv).await? as usize;
    if count > MAX_COLLECTION_READ_BUNDLES {
        bail!(
            "collection READ proof forest has {count} bundles; limit is {MAX_COLLECTION_READ_BUNDLES}"
        );
    }
    let mut read_evidence = Vec::new();
    read_evidence
        .try_reserve_exact(count)
        .map_err(|error| anyhow!("cannot allocate collection READ proof forest: {error}"))?;
    let mut aggregate = 0usize;
    for _ in 0..count {
        let bundle = recv_capability_proof_bundle(recv).await?;
        let length = bundle.to_bytes()?.len();
        aggregate = aggregate
            .checked_add(length)
            .ok_or_else(|| anyhow!("collection READ evidence length overflow"))?;
        if aggregate > MAX_COLLECTION_READ_EVIDENCE_BYTES {
            bail!(
                "collection READ evidence is {aggregate} bytes; limit is {MAX_COLLECTION_READ_EVIDENCE_BYTES}"
            );
        }
        read_evidence.push(bundle);
    }
    Ok(read_evidence)
}

pub(crate) async fn send_repair_challenge<W: AsyncWrite + Unpin>(
    send: &mut W,
    read_evidence: Option<&[CapabilityProofBundle]>,
) -> Result<()> {
    match read_evidence {
        Some(evidence) => {
            send_u8(send, REPAIR_CHALLENGE).await?;
            send_repair_evidence(send, evidence).await
        }
        None => send_u8(send, REPAIR_UNAVAILABLE).await,
    }
}

pub(crate) async fn recv_repair_challenge<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<Option<Vec<CapabilityProofBundle>>> {
    match recv_u8(recv).await? {
        REPAIR_CHALLENGE => Ok(Some(recv_repair_evidence(recv).await?)),
        REPAIR_UNAVAILABLE => Ok(None),
        other => bail!("unknown collection repair challenge {other:#x}"),
    }
}

pub(crate) async fn send_repair_admission<W: AsyncWrite + Unpin>(
    send: &mut W,
    admission: CollectionRepairAdmission,
) -> Result<()> {
    match admission {
        CollectionRepairAdmission::Admitted(manifest) => {
            send_u8(send, REPAIR_ADMITTED).await?;
            send_hash(send, &manifest.wake_root).await?;
            send_summary(send, manifest.records).await?;
            send_summary(send, manifest.write_evidence).await?;
            send_summary(send, manifest.resident).await?;
        }
        CollectionRepairAdmission::Rejected => send_u8(send, REPAIR_REJECTED).await?,
        CollectionRepairAdmission::Unavailable => send_u8(send, REPAIR_UNAVAILABLE).await?,
    }
    Ok(())
}

pub(crate) async fn recv_repair_admission<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CollectionRepairAdmission> {
    Ok(match recv_u8(recv).await? {
        REPAIR_ADMITTED => CollectionRepairAdmission::Admitted(CollectionRepairManifest {
            wake_root: recv_hash(recv).await?,
            records: recv_summary(recv).await?,
            write_evidence: recv_summary(recv).await?,
            resident: recv_summary(recv).await?,
        }),
        REPAIR_REJECTED => CollectionRepairAdmission::Rejected,
        REPAIR_UNAVAILABLE => CollectionRepairAdmission::Unavailable,
        other => bail!("unknown collection repair admission {other:#x}"),
    })
}

async fn send_summary<W: AsyncWrite + Unpin>(send: &mut W, summary: PatchSummary) -> Result<()> {
    match summary.root() {
        Some(root) => {
            send_u8(send, 1).await?;
            send_hash(send, &root).await?;
        }
        None => {
            send_u8(send, 0).await?;
            send_hash(send, &[0; 32]).await?;
        }
    }
    send_u64_be(send, summary.leaf_count()).await
}

async fn recv_summary<R: AsyncRead + Unpin>(recv: &mut R) -> Result<PatchSummary> {
    let present = recv_u8(recv).await?;
    let raw = recv_hash(recv).await?;
    let count = recv_u64_be(recv).await?;
    let root = match present {
        0 => {
            if raw != [0; 32] {
                bail!("empty PATCH summary carries a nonzero root field");
            }
            None
        }
        1 => Some(raw),
        other => bail!("invalid PATCH summary root tag {other:#x}"),
    };
    PatchSummary::new(root, count)
}

pub(crate) async fn send_repair_node_request<W: AsyncWrite + Unpin, S>(
    send: &mut W,
    request: &PatchRepairRequest<S>,
    component: CollectionRepairComponent,
) -> Result<()> {
    if request.prefix().len() > component.key_len() {
        bail!("collection PATCH request prefix exceeds component key length");
    }
    send_u8(send, REQUEST_NODE).await?;
    send_u8(send, component.wire()).await?;
    send_u8(
        send,
        u8::try_from(request.prefix().len()).expect("collection PATCH keys fit u8"),
    )
    .await?;
    send.write_all(request.prefix())
        .await
        .map_err(|error| anyhow!("send collection PATCH prefix: {error}"))?;
    send_hash(send, &request.expected_digest()).await
}

pub(crate) async fn send_repair_done<W: AsyncWrite + Unpin>(send: &mut W) -> Result<()> {
    send_u8(send, REQUEST_DONE).await
}

pub(crate) async fn send_repair_blob_request<W: AsyncWrite + Unpin>(
    send: &mut W,
    handle: [u8; 32],
) -> Result<()> {
    send_u8(send, REQUEST_BLOB).await?;
    send_hash(send, &handle).await
}

pub(crate) async fn recv_repair_command<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CollectionRepairCommand> {
    match recv_u8(recv).await? {
        REQUEST_DONE => Ok(CollectionRepairCommand::Done),
        REQUEST_NODE => {
            let component = CollectionRepairComponent::from_wire(recv_u8(recv).await?)?;
            let prefix_len = recv_u8(recv).await? as usize;
            if prefix_len > component.key_len() {
                bail!("collection PATCH request prefix exceeds component key length");
            }
            let mut prefix = vec![0; prefix_len];
            recv.read_exact(&mut prefix)
                .await
                .map_err(|error| anyhow!("receive collection PATCH prefix: {error}"))?;
            let expected_digest = recv_hash(recv).await?;
            Ok(CollectionRepairCommand::Node {
                component,
                prefix,
                expected_digest,
            })
        }
        REQUEST_BLOB => Ok(CollectionRepairCommand::Blob(recv_hash(recv).await?)),
        other => bail!("unknown collection repair command {other:#x}"),
    }
}

/// Return one exact bearer blob without ending the authenticated session.
/// `u64::MAX` means no bytes are available for the requested exact handle in
/// this snapshot.
pub(crate) async fn send_repair_blob_response<W: AsyncWrite + Unpin>(
    send: &mut W,
    bytes: Option<&[u8]>,
) -> Result<()> {
    match bytes {
        Some(bytes) => {
            send_u64_be(
                send,
                u64::try_from(bytes.len()).expect("an addressable blob length fits u64"),
            )
            .await?;
            send.write_all(bytes)
                .await
                .map_err(|error| anyhow!("send collection blob: {error}"))?;
        }
        None => send_u64_be(send, u64::MAX).await?,
    }
    Ok(())
}

pub(crate) async fn recv_repair_blob_response<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<Option<Bytes>> {
    let length = recv_u64_be(recv).await?;
    if length == u64::MAX {
        return Ok(None);
    }
    if length > crate::protocol::MAX_EXACT_BLOB_BYTES {
        bail!(
            "collection blob response exceeds the {}-byte transport bound",
            crate::protocol::MAX_EXACT_BLOB_BYTES
        );
    }
    let length = usize::try_from(length)
        .map_err(|_| anyhow!("collection blob length does not fit this address space"))?;
    Ok(Some(
        crate::protocol::recv_exact_blob_body(recv, length).await?,
    ))
}

pub(crate) async fn send_repair_node_response<W: AsyncWrite + Unpin>(
    send: &mut W,
    response: &PatchNodeResponse<Vec<u8>>,
    component: CollectionRepairComponent,
) -> Result<()> {
    match response {
        PatchNodeResponse::SnapshotUnavailable => {
            bail!("a stream-pinned collection snapshot became unavailable")
        }
        PatchNodeResponse::PrefixAbsent => send_u8(send, NODE_PREFIX_ABSENT).await?,
        PatchNodeResponse::Found(node) => {
            send_u8(send, NODE_FOUND).await?;
            match node {
                PatchNode::Leaf { digest, leaf } => {
                    if leaf.key.len() != component.key_len() {
                        bail!("collection PATCH leaf key has the wrong length");
                    }
                    if leaf.value.len() > MAX_COLLECTION_LEAF_BYTES {
                        bail!(
                            "collection PATCH leaf is {} bytes; limit is {MAX_COLLECTION_LEAF_BYTES}",
                            leaf.value.len()
                        );
                    }
                    send_u8(send, NODE_LEAF).await?;
                    send_hash(send, digest).await?;
                    send.write_all(&leaf.key)
                        .await
                        .map_err(|error| anyhow!("send collection PATCH leaf key: {error}"))?;
                    send_u32_be(
                        send,
                        u32::try_from(leaf.value.len())
                            .expect("collection leaf byte bound fits u32"),
                    )
                    .await?;
                    send.write_all(&leaf.value)
                        .await
                        .map_err(|error| anyhow!("send collection PATCH leaf value: {error}"))?;
                }
                PatchNode::Branch {
                    digest,
                    leaf_count,
                    branch,
                } => {
                    if branch.representative.len() != component.key_len() {
                        bail!("collection PATCH branch representative has the wrong length");
                    }
                    if !(2..=256).contains(&branch.children.len()) {
                        bail!("collection PATCH branch fanout is not canonical");
                    }
                    send_u8(send, NODE_BRANCH).await?;
                    send_hash(send, digest).await?;
                    send_u64_be(send, *leaf_count).await?;
                    send.write_all(&branch.representative)
                        .await
                        .map_err(|error| {
                            anyhow!("send collection PATCH branch representative: {error}")
                        })?;
                    send_u8(send, branch.end_depth).await?;
                    send_u32_be(
                        send,
                        u32::try_from(branch.children.len())
                            .expect("canonical PATCH fanout fits u32"),
                    )
                    .await?;
                    for child in &branch.children {
                        send_u8(send, child.edge).await?;
                        send_hash(send, &child.digest).await?;
                        send_u64_be(send, child.leaf_count).await?;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) async fn recv_repair_node_response<R: AsyncRead + Unpin>(
    recv: &mut R,
    component: CollectionRepairComponent,
) -> Result<PatchNodeResponse<Vec<u8>>> {
    match recv_u8(recv).await? {
        NODE_PREFIX_ABSENT => Ok(PatchNodeResponse::PrefixAbsent),
        NODE_FOUND => {
            let kind = recv_u8(recv).await?;
            let digest = recv_hash(recv).await?;
            match kind {
                NODE_LEAF => {
                    let mut key = vec![0; component.key_len()];
                    recv.read_exact(&mut key)
                        .await
                        .map_err(|error| anyhow!("receive collection PATCH leaf key: {error}"))?;
                    let length = recv_u32_be(recv).await? as usize;
                    if length > MAX_COLLECTION_LEAF_BYTES {
                        bail!(
                            "collection PATCH leaf is {length} bytes; limit is {MAX_COLLECTION_LEAF_BYTES}"
                        );
                    }
                    let mut value = Vec::new();
                    value.try_reserve_exact(length).map_err(|error| {
                        anyhow!("cannot allocate collection PATCH leaf: {error}")
                    })?;
                    value.resize(length, 0);
                    recv.read_exact(&mut value)
                        .await
                        .map_err(|error| anyhow!("receive collection PATCH leaf value: {error}"))?;
                    Ok(PatchNodeResponse::Found(PatchNode::Leaf {
                        digest,
                        leaf: PatchLeaf { key, value },
                    }))
                }
                NODE_BRANCH => {
                    let leaf_count = recv_u64_be(recv).await?;
                    let mut representative = vec![0; component.key_len()];
                    recv.read_exact(&mut representative)
                        .await
                        .map_err(|error| {
                            anyhow!("receive collection PATCH branch representative: {error}")
                        })?;
                    let end_depth = recv_u8(recv).await?;
                    let child_count = recv_u32_be(recv).await? as usize;
                    if !(2..=256).contains(&child_count) {
                        bail!("collection PATCH branch fanout is {child_count}");
                    }
                    let mut children = Vec::with_capacity(child_count);
                    for _ in 0..child_count {
                        children.push(PatchChild {
                            edge: recv_u8(recv).await?,
                            digest: recv_hash(recv).await?,
                            leaf_count: recv_u64_be(recv).await?,
                        });
                    }
                    Ok(PatchNodeResponse::Found(PatchNode::Branch {
                        digest,
                        leaf_count,
                        branch: PatchBranch {
                            representative,
                            end_depth,
                            children,
                        },
                    }))
                }
                other => bail!("unknown collection PATCH node kind {other:#x}"),
            }
        }
        other => bail!("unknown collection PATCH response {other:#x}"),
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use triblespace_core::capability::{
        CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    };

    use super::*;

    fn bundle() -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            &SigningKey::from_bytes(&[1; 32]),
            CapabilityClaim::root(
                CapabilityAtom::new(
                    triblespace_core::capability::CapabilityAction::new(
                        triblespace_core::collection::ACTION_READ,
                    ),
                    triblespace_core::capability::CapabilityResource::new([2; 32]),
                ),
                CapabilityMode::Invoke,
                None,
            ),
            SigningKey::from_bytes(&[3; 32]).verifying_key(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn hello_and_manifest_roundtrip_without_ending_the_stream() {
        let (mut left, mut right) = tokio::io::duplex(1 << 20);
        let hello = CollectionRepairHello {
            read_evidence: vec![bundle()],
        };
        let collection = CollectionHandle::new([2; 32]);
        let manifest = CollectionRepairManifest {
            wake_root: [4; 32],
            records: PatchSummary::new(Some([5; 32]), 7).unwrap(),
            write_evidence: PatchSummary::new(None, 0).unwrap(),
            resident: PatchSummary::new(Some([6; 32]), 9).unwrap(),
        };
        let sent_hello = hello.clone();
        let writer = tokio::spawn(async move {
            send_u8(&mut left, OP_COLLECTION_REPAIR).await.unwrap();
            send_hash(&mut left, &collection.raw).await.unwrap();
            send_repair_evidence(&mut left, &sent_hello.read_evidence)
                .await
                .unwrap();
            assert_eq!(recv_u8(&mut left).await.unwrap(), 0xA5);
            send_repair_admission(&mut left, CollectionRepairAdmission::Admitted(manifest))
                .await
                .unwrap();
        });

        assert_eq!(recv_u8(&mut right).await.unwrap(), OP_COLLECTION_REPAIR);
        assert_eq!(
            recv_repair_collection(&mut right).await.unwrap(),
            collection
        );
        assert_eq!(recv_repair_hello(&mut right).await.unwrap(), hello);
        send_u8(&mut right, 0xA5).await.unwrap();
        assert_eq!(
            recv_repair_admission(&mut right).await.unwrap(),
            CollectionRepairAdmission::Admitted(manifest)
        );
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn node_commands_and_leaf_values_roundtrip() {
        let component = CollectionRepairComponent::Record;
        let summary = PatchSummary::new(Some([7; 32]), 1).unwrap();
        let request = PatchRepairRequest::new(component, summary, 16, vec![], [7; 32]).unwrap();
        let response = PatchNodeResponse::Found(PatchNode::Leaf {
            digest: [7; 32],
            leaf: PatchLeaf {
                key: vec![8; 16],
                value: vec![9; 192],
            },
        });
        let expected_response = response.clone();
        let (mut left, mut right) = tokio::io::duplex(1 << 20);
        let writer = tokio::spawn(async move {
            send_repair_node_request(&mut left, &request, component)
                .await
                .unwrap();
            send_repair_done(&mut left).await.unwrap();
            send_repair_node_response(&mut left, &response, component)
                .await
                .unwrap();
        });

        assert!(matches!(
            recv_repair_command(&mut right).await.unwrap(),
            CollectionRepairCommand::Node {
                component: CollectionRepairComponent::Record,
                ref prefix,
                expected_digest,
            } if prefix.is_empty() && expected_digest == [7; 32]
        ));
        assert_eq!(
            recv_repair_command(&mut right).await.unwrap(),
            CollectionRepairCommand::Done
        );
        assert_eq!(
            recv_repair_node_response(&mut right, component)
                .await
                .unwrap(),
            expected_response
        );
        writer.await.unwrap();
    }
}
