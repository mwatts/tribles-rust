//! Bounded wire codec for proof-gated custody replication.
//!
//! This protocol is intentionally separate from public evidence gossip. Every
//! request repeats the exact replica-set resource and its portable proof
//! bundle; CONNECT only admits the transport connection.

use anybytes::Bytes;
use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{
    CapabilityProof, CapabilityProofBundle, MAX_CAPABILITY_PROOF_STEPS,
};
use triblespace_core::collection::{COLLECTION_COMMIT_BYTES_LEN, CollectionRecord};

use crate::protocol::{
    recv_capability_proof_bundle, recv_hash, recv_u8, recv_u64_be, send_capability_proof_bundle,
    send_hash, send_u8, send_u64_be,
};
use crate::replica::{
    ReplicaBucketSummary, ReplicaComponent, ReplicaItem, ReplicaItemId, ReplicaSetId,
    ReplicaSummary,
};
use crate::transport::Conn;

/// Fetch the fixed summary of the complete semantic product.
pub(crate) const OP_REPLICA_SUMMARY: u8 = 0x08;
/// Fetch one bounded, sorted first-byte bucket page.
pub(crate) const OP_REPLICA_PAGE: u8 = 0x09;
/// Stream one exact resident blob without a contiguous heap allocation.
pub(crate) const OP_REPLICA_BLOB: u8 = 0x0A;

#[cfg(test)]
const SUMMARY_BUCKET_BYTES: usize = 8 + 8 + 32;
#[cfg(test)]
const SUMMARY_BYTES: usize = 3 * 256 * SUMMARY_BUCKET_BYTES;
const RECORD_MAX_BYTES: usize = 1 + COLLECTION_COMMIT_BYTES_LEN;
const PROOF_MAX_BYTES: usize = 32 + MAX_CAPABILITY_PROOF_STEPS * 128;
/// Maximum payload of one resumable blob range operation.
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

async fn request_prefix<W: AsyncWrite + Unpin>(
    send: &mut W,
    operation: u8,
    replica_set: ReplicaSetId,
    proof: &CapabilityProofBundle,
) -> Result<()> {
    send_u8(send, operation).await?;
    send_hash(send, &replica_set.into_bytes()).await?;
    send_capability_proof_bundle(send, proof).await
}

pub(crate) async fn recv_request_prefix<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<(ReplicaSetId, CapabilityProofBundle)> {
    let replica_set = ReplicaSetId::new(recv_hash(recv).await?);
    let proof = recv_capability_proof_bundle(recv).await?;
    Ok((replica_set, proof))
}

pub(crate) async fn require_eof<R: AsyncRead + Unpin>(recv: &mut R) -> Result<()> {
    let mut trailing = [0; 1];
    if recv
        .read(&mut trailing)
        .await
        .map_err(|error| anyhow!("read request terminator: {error}"))?
        != 0
    {
        bail!("custody-replica request contains trailing bytes");
    }
    Ok(())
}

pub(crate) async fn send_summary<W: AsyncWrite + Unpin>(
    send: &mut W,
    summary: &ReplicaSummary,
) -> Result<()> {
    for component in ReplicaComponent::ALL {
        for bucket in summary.buckets(component) {
            send_u64_be(send, bucket.count).await?;
            send_u64_be(send, bucket.bytes).await?;
            send.write_all(&bucket.digest)
                .await
                .map_err(|error| anyhow!("send replica summary digest: {error}"))?;
        }
    }
    Ok(())
}

async fn recv_summary<R: AsyncRead + Unpin>(recv: &mut R) -> Result<ReplicaSummary> {
    let mut buckets = [[ReplicaBucketSummary::default(); 256]; 3];
    for component in ReplicaComponent::ALL {
        for prefix in 0..=u8::MAX {
            let count = recv_u64_be(recv).await?;
            let bytes = recv_u64_be(recv).await?;
            let mut digest = [0; 32];
            recv.read_exact(&mut digest)
                .await
                .map_err(|error| anyhow!("recv replica summary digest: {error}"))?;
            buckets[component as usize - 1][prefix as usize] = ReplicaBucketSummary {
                count,
                bytes,
                digest,
            };
        }
    }
    require_eof(recv).await?;
    Ok(ReplicaSummary::from_buckets(buckets))
}

pub(crate) async fn op_replica_summary<C: Conn>(
    connection: &C,
    replica_set: ReplicaSetId,
    proof: &CapabilityProofBundle,
) -> Result<ReplicaSummary> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open replica summary stream: {error}"))?;
    request_prefix(&mut send, OP_REPLICA_SUMMARY, replica_set, proof).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish replica summary request: {error}"))?;
    recv_summary(&mut recv).await
}

pub(crate) async fn recv_page_request<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<(ReplicaComponent, u8, Option<ReplicaItemId>)> {
    let component = ReplicaComponent::from_byte(recv_u8(recv).await?)?;
    let prefix = recv_u8(recv).await?;
    let after = match recv_u8(recv).await? {
        0 => None,
        1 => Some(ReplicaItemId(recv_hash(recv).await?)),
        other => bail!("invalid replica page cursor marker {other:#x}"),
    };
    if let Some(after) = after {
        if after.prefix() != prefix {
            bail!("replica page cursor is outside requested prefix bucket");
        }
    }
    require_eof(recv).await?;
    Ok((component, prefix, after))
}

pub(crate) async fn send_page<W: AsyncWrite + Unpin>(
    send: &mut W,
    component: ReplicaComponent,
    items: &[ReplicaItem],
    done: bool,
) -> Result<()> {
    if items.len() > component.page_limit() {
        bail!("replica page exceeds component limit");
    }
    send_u16_be(
        send,
        u16::try_from(items.len()).expect("replica page limits fit u16"),
    )
    .await?;
    send_u8(send, u8::from(done)).await?;
    for item in items {
        let item_component = match item {
            ReplicaItem::Blob(_) => ReplicaComponent::Blobs,
            ReplicaItem::CollectionRecord(_) => ReplicaComponent::CollectionRecords,
            ReplicaItem::CapabilityProof(_) => ReplicaComponent::CapabilityProofs,
        };
        if item_component != component {
            bail!("replica page item does not match requested component");
        }
        send_hash(send, &item.id().0).await?;
        match item {
            ReplicaItem::Blob(info) => send_u64_be(send, info.length).await?,
            ReplicaItem::CollectionRecord(record) => {
                let bytes = record.to_bytes();
                send_u16_be(
                    send,
                    u16::try_from(bytes.len()).expect("collection record fits u16"),
                )
                .await?;
                send.write_all(&bytes)
                    .await
                    .map_err(|error| anyhow!("send collection record: {error}"))?;
            }
            ReplicaItem::CapabilityProof(proof) => {
                let bytes = proof.as_bytes();
                send_u16_be(
                    send,
                    u16::try_from(bytes.len()).expect("bounded capability proof fits u16"),
                )
                .await?;
                send.write_all(bytes)
                    .await
                    .map_err(|error| anyhow!("send capability proof: {error}"))?;
            }
        }
    }
    Ok(())
}

async fn recv_bounded_frame<R: AsyncRead + Unpin>(
    recv: &mut R,
    maximum: usize,
    what: &str,
) -> Result<Vec<u8>> {
    let len = recv_u16_be(recv).await? as usize;
    if len > maximum {
        bail!("{what} frame is {len} bytes; limit is {maximum}");
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|error| anyhow!("cannot allocate {what} frame: {error}"))?;
    bytes.resize(len, 0);
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv {what}: {error}"))?;
    Ok(bytes)
}

async fn recv_page<R: AsyncRead + Unpin>(
    recv: &mut R,
    component: ReplicaComponent,
    prefix: u8,
    after: Option<ReplicaItemId>,
) -> Result<(Vec<ReplicaItem>, bool)> {
    let count = recv_u16_be(recv).await? as usize;
    if count > component.page_limit() {
        bail!(
            "replica page count {count} exceeds {} limit {}",
            component as u8,
            component.page_limit()
        );
    }
    let done = match recv_u8(recv).await? {
        0 => false,
        1 => true,
        other => bail!("invalid replica page done marker {other:#x}"),
    };
    if !done && count == 0 {
        bail!("nonterminal replica page is empty");
    }

    let mut items = Vec::new();
    items
        .try_reserve_exact(count)
        .map_err(|error| anyhow!("cannot allocate replica page: {error}"))?;
    let mut previous = after;
    for _ in 0..count {
        let wire_id = ReplicaItemId(recv_hash(recv).await?);
        if wire_id.prefix() != prefix {
            bail!("replica page item is outside requested prefix bucket");
        }
        if previous.is_some_and(|previous| wire_id <= previous) {
            bail!("replica page is not strictly ordered after its cursor");
        }
        let item = match component {
            ReplicaComponent::Blobs => ReplicaItem::Blob(triblespace_core::repo::BlobInfo {
                handle: triblespace_core::inline::Inline::new(wire_id.0),
                length: recv_u64_be(recv).await?,
            }),
            ReplicaComponent::CollectionRecords => {
                let bytes = recv_bounded_frame(recv, RECORD_MAX_BYTES, "collection record").await?;
                let record = CollectionRecord::from_bytes(&bytes)
                    .context("decode canonical collection record")?;
                if record.to_bytes() != bytes {
                    bail!("collection record did not round-trip canonically");
                }
                if ReplicaItemId::collection(record.id()) != wire_id {
                    bail!("collection record identity does not match page key");
                }
                ReplicaItem::CollectionRecord(record)
            }
            ReplicaComponent::CapabilityProofs => {
                let bytes = recv_bounded_frame(recv, PROOF_MAX_BYTES, "capability proof").await?;
                let proof = CapabilityProof::from_bytes(&bytes)
                    .context("decode canonical capability proof")?;
                if proof.as_bytes() != bytes {
                    bail!("capability proof did not round-trip canonically");
                }
                if proof.id().raw != wire_id.0 {
                    bail!("capability proof identity does not match page key");
                }
                ReplicaItem::CapabilityProof(proof)
            }
        };
        previous = Some(wire_id);
        items.push(item);
    }
    require_eof(recv).await?;
    Ok((items, done))
}

pub(crate) async fn op_replica_page<C: Conn>(
    connection: &C,
    replica_set: ReplicaSetId,
    proof: &CapabilityProofBundle,
    component: ReplicaComponent,
    prefix: u8,
    after: Option<ReplicaItemId>,
) -> Result<(Vec<ReplicaItem>, bool)> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open replica page stream: {error}"))?;
    request_prefix(&mut send, OP_REPLICA_PAGE, replica_set, proof).await?;
    send_u8(&mut send, component as u8).await?;
    send_u8(&mut send, prefix).await?;
    send_u8(&mut send, u8::from(after.is_some())).await?;
    if let Some(after) = after {
        if after.prefix() != prefix {
            bail!("replica page cursor is outside requested prefix bucket");
        }
        send_hash(&mut send, &after.0).await?;
    }
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish replica page request: {error}"))?;
    recv_page(&mut recv, component, prefix, after).await
}

pub(crate) async fn recv_blob_request<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<(ReplicaItemId, u64, u32)> {
    let id = ReplicaItemId(recv_hash(recv).await?);
    let offset = recv_u64_be(recv).await?;
    let maximum = crate::protocol::recv_u32_be(recv).await?;
    if maximum == 0 || maximum as usize > BLOB_TRANSFER_CHUNK_BYTES {
        bail!("replica blob range limit is {maximum}; expected 1..={BLOB_TRANSFER_CHUNK_BYTES}");
    }
    require_eof(recv).await?;
    Ok((id, offset, maximum))
}

pub(crate) async fn send_blob_range<W: AsyncWrite + Unpin>(
    send: &mut W,
    bytes: Option<&Bytes>,
    offset: u64,
    maximum: u32,
) -> Result<()> {
    let Some(bytes) = bytes else {
        return send_u64_be(send, u64::MAX).await;
    };
    let total = u64::try_from(bytes.len()).context("blob length does not fit u64")?;
    if offset > total {
        bail!("replica blob range starts past the end of the blob");
    }
    let start = usize::try_from(offset).context("blob offset does not fit address space")?;
    let end = start.saturating_add(maximum as usize).min(bytes.len());
    let chunk = &bytes[start..end];
    send_u64_be(send, total).await?;
    crate::protocol::send_u32_be(
        send,
        u32::try_from(chunk.len()).expect("bounded blob chunk fits u32"),
    )
    .await?;
    send.write_all(chunk)
        .await
        .map_err(|error| anyhow!("send replica blob range: {error}"))?;
    Ok(())
}

async fn recv_blob_range<R: AsyncRead + Unpin>(
    recv: &mut R,
    expected_len: u64,
    offset: u64,
    target: &mut [u8],
) -> Result<Option<usize>> {
    let declared_len = recv_u64_be(recv).await?;
    if declared_len == u64::MAX {
        require_eof(recv).await?;
        return Ok(None);
    }
    if declared_len != expected_len {
        bail!(
            "replica blob length changed between inventory ({expected_len}) and transfer ({declared_len})"
        );
    }
    let chunk_len = crate::protocol::recv_u32_be(recv).await? as usize;
    if chunk_len > target.len() || chunk_len > BLOB_TRANSFER_CHUNK_BYTES {
        bail!("replica blob range exceeds requested bounded target");
    }
    let remaining = expected_len
        .checked_sub(offset)
        .ok_or_else(|| anyhow!("replica blob range offset exceeds expected length"))?;
    let expected_chunk = usize::try_from(remaining.min(target.len() as u64))
        .expect("bounded target length fits usize");
    if chunk_len != expected_chunk {
        bail!("replica blob range returned {chunk_len} bytes; expected {expected_chunk}");
    }
    recv.read_exact(&mut target[..chunk_len])
        .await
        .map_err(|error| anyhow!("recv replica blob range: {error}"))?;
    require_eof(recv).await?;
    Ok(Some(chunk_len))
}

pub(crate) async fn op_replica_blob_range<C: Conn>(
    connection: &C,
    replica_set: ReplicaSetId,
    proof: &CapabilityProofBundle,
    requested: ReplicaItemId,
    expected_len: u64,
    offset: u64,
    target: &mut [u8],
) -> Result<Option<usize>> {
    if target.is_empty() || target.len() > BLOB_TRANSFER_CHUNK_BYTES {
        bail!("replica blob range target must be nonempty and bounded");
    }
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| anyhow!("open replica blob stream: {error}"))?;
    request_prefix(&mut send, OP_REPLICA_BLOB, replica_set, proof).await?;
    send_hash(&mut send, &requested.0).await?;
    send_u64_be(&mut send, offset).await?;
    crate::protocol::send_u32_be(
        &mut send,
        u32::try_from(target.len()).expect("bounded blob target fits u32"),
    )
    .await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish replica blob request: {error}"))?;
    recv_blob_range(&mut recv, expected_len, offset, target).await
}

#[cfg(test)]
mod tests {
    use anybytes::Bytes;
    use tokio::io::{AsyncWriteExt, duplex};
    use triblespace_core::collection::{CollectionData, CollectionMerge};

    use super::*;

    #[tokio::test]
    async fn summary_round_trips_exact_fixed_frame() {
        let mut buckets = [[ReplicaBucketSummary::default(); 256]; 3];
        buckets[1][7] = ReplicaBucketSummary {
            count: 3,
            bytes: 901,
            digest: [4; 32],
        };
        let summary = ReplicaSummary::from_buckets(buckets);
        let (mut writer, mut reader) = duplex(SUMMARY_BYTES + 1);
        let send = tokio::spawn(async move {
            send_summary(&mut writer, &summary).await.unwrap();
            writer.shutdown().await.unwrap();
        });
        let decoded = recv_summary(&mut reader).await.unwrap();
        send.await.unwrap();
        assert_eq!(
            decoded.bucket(ReplicaComponent::CollectionRecords, 7),
            ReplicaBucketSummary {
                count: 3,
                bytes: 901,
                digest: [4; 32],
            }
        );
    }

    #[tokio::test]
    async fn page_round_trips_canonical_record() {
        let record = CollectionRecord::Merge(CollectionMerge::new(
            triblespace_core::inline::Inline::new([1; 32]),
            CollectionData::new([2; 32]),
            CollectionData::new([3; 32]),
            CollectionData::new([4; 32]),
        ));
        let prefix = record.id().raw()[0];
        let (mut writer, mut reader) = duplex(1024);
        let send = tokio::spawn(async move {
            send_page(
                &mut writer,
                ReplicaComponent::CollectionRecords,
                &[ReplicaItem::CollectionRecord(record)],
                true,
            )
            .await
            .unwrap();
            writer.shutdown().await.unwrap();
        });
        let (page, done) = recv_page(
            &mut reader,
            ReplicaComponent::CollectionRecords,
            prefix,
            None,
        )
        .await
        .unwrap();
        send.await.unwrap();
        assert!(done);
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].id(), ReplicaItemId::collection(record.id()));
    }

    #[tokio::test]
    async fn page_rejects_wrong_prefix_and_nonadvancing_cursor() {
        let wrong_prefix = ReplicaItem::Blob(triblespace_core::repo::BlobInfo {
            handle: triblespace_core::inline::Inline::new([0x44; 32]),
            length: 1,
        });
        let (mut writer, mut reader) = duplex(1024);
        let send = tokio::spawn(async move {
            send_page(&mut writer, ReplicaComponent::Blobs, &[wrong_prefix], true)
                .await
                .unwrap();
            writer.shutdown().await.unwrap();
        });
        assert!(
            recv_page(&mut reader, ReplicaComponent::Blobs, 0x43, None)
                .await
                .is_err()
        );
        send.await.unwrap();

        let id = ReplicaItemId([0x45; 32]);
        let repeated = ReplicaItem::Blob(triblespace_core::repo::BlobInfo {
            handle: triblespace_core::inline::Inline::new(id.0),
            length: 1,
        });
        let (mut writer, mut reader) = duplex(1024);
        let send = tokio::spawn(async move {
            send_page(&mut writer, ReplicaComponent::Blobs, &[repeated], true)
                .await
                .unwrap();
            writer.shutdown().await.unwrap();
        });
        assert!(
            recv_page(&mut reader, ReplicaComponent::Blobs, 0x45, Some(id))
                .await
                .is_err()
        );
        send.await.unwrap();
    }

    #[tokio::test]
    async fn blob_range_is_bounded_and_resumable() {
        let payload = vec![9; BLOB_TRANSFER_CHUNK_BYTES + 17];
        let expected_len = payload.len() as u64;
        let bytes = Bytes::from(payload.clone());
        let (mut writer, mut reader) = duplex(64 * 1024);
        let send = tokio::spawn(async move {
            send_blob_range(
                &mut writer,
                Some(&bytes),
                BLOB_TRANSFER_CHUNK_BYTES as u64,
                17,
            )
            .await
            .unwrap();
            writer.shutdown().await.unwrap();
        });
        let mut received = [0; 17];
        let count = recv_blob_range(
            &mut reader,
            expected_len,
            BLOB_TRANSFER_CHUNK_BYTES as u64,
            &mut received,
        )
        .await
        .unwrap()
        .unwrap();
        send.await.unwrap();
        assert_eq!(count, 17);
        assert_eq!(&received, &payload[BLOB_TRANSFER_CHUNK_BYTES..]);
    }
}
