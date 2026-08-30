//! Binary wire protocol primitives.
//!
//! One QUIC stream carries one operation. Establishing the TLS connection
//! grants no collection authority: `COLLECTION_REPAIR` carries READ(C)
//! evidence in its own request. Exact GET by a known handle and DHT
//! FIND/PROVIDER operations are intentionally bearer/public mechanisms.

use anyhow::{Result, anyhow};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use triblespace_core::capability::{CapabilityProofBundle, MAX_CAPABILITY_PROOF_BUNDLE_BYTES};

use crate::transport::Conn;

/// Clean collection-scoped protocol generation.
pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/17";

// Operation types — first byte on each stream.
// 0x01 was branch-list; 0x03 was blob-children; 0x04 was branch-head;
// 0x05 was connection AUTH. None are accepted in v17.
pub const OP_GET_BLOB: u8 = 0x02;
pub const OP_PROVIDER_PUT: u8 = 0x06;
pub const OP_PROVIDER_GET: u8 = 0x07;
pub const OP_FIND_NODE: u8 = 0x0C;
// 0x0D is OP_COLLECTION_REPAIR, owned by collection_wire.

pub const PROVIDER_PUT_OK: u8 = 0x00;
pub const PROVIDER_PUT_FULL: u8 = 0x01;

pub type RawHash = [u8; 32];

pub async fn send_u8<W: AsyncWrite + Unpin>(send: &mut W, value: u8) -> Result<()> {
    send.write_all(&[value])
        .await
        .map_err(|error| anyhow!("send: {error}"))
}

pub async fn send_hash<W: AsyncWrite + Unpin>(send: &mut W, hash: &RawHash) -> Result<()> {
    send.write_all(hash)
        .await
        .map_err(|error| anyhow!("send: {error}"))
}

pub async fn send_u32_be<W: AsyncWrite + Unpin>(send: &mut W, value: u32) -> Result<()> {
    send.write_all(&value.to_be_bytes())
        .await
        .map_err(|error| anyhow!("send: {error}"))
}

pub async fn send_u64_be<W: AsyncWrite + Unpin>(send: &mut W, value: u64) -> Result<()> {
    send.write_all(&value.to_be_bytes())
        .await
        .map_err(|error| anyhow!("send: {error}"))
}

pub async fn recv_u8<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u8> {
    let mut bytes = [0; 1];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv: {error}"))?;
    Ok(bytes[0])
}

pub async fn recv_hash<R: AsyncRead + Unpin>(recv: &mut R) -> Result<RawHash> {
    let mut bytes = [0; 32];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv: {error}"))?;
    Ok(bytes)
}

pub async fn recv_u32_be<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u32> {
    let mut bytes = [0; 4];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv: {error}"))?;
    Ok(u32::from_be_bytes(bytes))
}

pub async fn recv_u64_be<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u64> {
    let mut bytes = [0; 8];
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv: {error}"))?;
    Ok(u64::from_be_bytes(bytes))
}

/// Write one bounded, canonical, length-prefixed proof bundle.
pub async fn send_capability_proof_bundle<W: AsyncWrite + Unpin>(
    send: &mut W,
    bundle: &CapabilityProofBundle,
) -> Result<()> {
    let bytes = bundle.to_bytes()?;
    debug_assert!(bytes.len() <= MAX_CAPABILITY_PROOF_BUNDLE_BYTES);
    send_u32_be(
        send,
        u32::try_from(bytes.len()).expect("the static bundle bound fits u32"),
    )
    .await?;
    send.write_all(&bytes)
        .await
        .map_err(|error| anyhow!("send capability proof bundle: {error}"))
}

/// Read one bounded, canonical, length-prefixed proof bundle.
pub async fn recv_capability_proof_bundle<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<CapabilityProofBundle> {
    let frame_len = recv_u32_be(recv).await? as usize;
    if frame_len > MAX_CAPABILITY_PROOF_BUNDLE_BYTES {
        return Err(anyhow!(
            "capability proof bundle frame is {frame_len} bytes; limit is {MAX_CAPABILITY_PROOF_BUNDLE_BYTES}"
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(frame_len)
        .map_err(|error| anyhow!("cannot allocate capability proof bundle: {error}"))?;
    bytes.resize(frame_len, 0);
    recv.read_exact(&mut bytes)
        .await
        .map_err(|error| anyhow!("recv capability proof bundle: {error}"))?;
    Ok(CapabilityProofBundle::from_bytes(&bytes)?)
}

/// Bearer exact GET by an already-known content handle.
pub async fn op_get_blob<C: Conn>(conn: &C, hash: &RawHash) -> Result<Option<Vec<u8>>> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|error| anyhow!("open_bi: {error}"))?;
    send_u8(&mut send, OP_GET_BLOB).await?;
    send_hash(&mut send, hash).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish: {error}"))?;
    recv_blob_response(&mut recv).await
}

/// Install or renew one opaque provider key for the TLS-authenticated caller.
pub(crate) async fn op_provider_put<C: Conn>(conn: &C, key: &RawHash) -> Result<bool> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|error| anyhow!("open_bi: {error}"))?;
    send_u8(&mut send, OP_PROVIDER_PUT).await?;
    send_hash(&mut send, key).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish: {error}"))?;
    let stored = match recv_u8(&mut recv).await? {
        PROVIDER_PUT_OK => true,
        PROVIDER_PUT_FULL => false,
        other => return Err(anyhow!("unknown provider-put response: {other:#x}")),
    };
    require_response_eof(&mut recv).await?;
    Ok(stored)
}

/// Return bounded provider hints for one derived rendezvous key.
pub async fn op_provider_get<C: Conn>(conn: &C, key: &RawHash) -> Result<Vec<RawHash>> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|error| anyhow!("open_bi: {error}"))?;
    send_u8(&mut send, OP_PROVIDER_GET).await?;
    send_hash(&mut send, key).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish: {error}"))?;
    let count = recv_u8(&mut recv).await? as usize;
    if count > crate::provider::MAX_PROVIDERS_PER_KEY {
        return Err(anyhow!(
            "provider-get response has {count} entries; limit is {}",
            crate::provider::MAX_PROVIDERS_PER_KEY
        ));
    }
    let mut providers = Vec::with_capacity(count);
    for _ in 0..count {
        providers.push(recv_hash(&mut recv).await?);
    }
    require_response_eof(&mut recv).await?;
    Ok(providers)
}

/// Return at most K verified routes nearest an arbitrary XOR target.
pub async fn op_find_node<C: Conn>(
    conn: &C,
    target: &crate::routing::RoutingKey,
) -> Result<Vec<crate::transport::PeerId>> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|error| anyhow!("open_bi: {error}"))?;
    send_u8(&mut send, OP_FIND_NODE).await?;
    send_hash(&mut send, target).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow!("finish: {error}"))?;
    recv_find_node_response(&mut recv).await
}

pub(crate) async fn recv_find_node_response<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<Vec<crate::transport::PeerId>> {
    let count = recv_u8(recv).await? as usize;
    if count > crate::routing::K {
        return Err(anyhow!(
            "FIND_NODE response has {count} entries; limit is {}",
            crate::routing::K
        ));
    }
    let mut peers = Vec::with_capacity(count);
    for _ in 0..count {
        peers.push(recv_hash(recv).await?);
    }
    require_response_eof(recv).await?;
    Ok(peers)
}

async fn require_response_eof<R: AsyncRead + Unpin>(recv: &mut R) -> Result<()> {
    let mut trailing = [0; 1];
    if recv.read(&mut trailing).await? != 0 {
        return Err(anyhow!("response contains trailing bytes"));
    }
    Ok(())
}

async fn recv_blob_response<R: AsyncRead + Unpin>(recv: &mut R) -> Result<Option<Vec<u8>>> {
    let len = recv_u64_be(recv).await?;
    if len == u64::MAX {
        return Ok(None);
    }
    let len = usize::try_from(len)
        .map_err(|_| anyhow!("blob response length does not fit this address space"))?;
    const READ_CHUNK_BYTES: usize = 64 * 1024;
    let mut data = Vec::new();
    while data.len() < len {
        let chunk_len = (len - data.len()).min(READ_CHUNK_BYTES);
        data.try_reserve_exact(chunk_len)
            .map_err(|error| anyhow!("cannot allocate blob response: {error}"))?;
        let start = data.len();
        data.resize(start + chunk_len, 0);
        recv.read_exact(&mut data[start..])
            .await
            .map_err(|error| anyhow!("recv: {error}"))?;
    }
    require_response_eof(recv).await?;
    Ok(Some(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn proof_frame_bound_is_checked_before_body_allocation() {
        let bytes = ((MAX_CAPABILITY_PROOF_BUNDLE_BYTES + 1) as u32).to_be_bytes();
        assert!(
            recv_capability_proof_bundle(&mut bytes.as_slice())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn find_node_enforces_count_and_exact_eof() {
        let oversized = [u8::try_from(crate::routing::K + 1).unwrap()];
        assert!(
            recv_find_node_response(&mut oversized.as_slice())
                .await
                .is_err()
        );
        assert!(
            recv_find_node_response(&mut [0, 1].as_slice())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn exact_get_accepts_empty_content_and_rejects_trailing_bytes() {
        assert_eq!(
            recv_blob_response(&mut [0; 8].as_slice()).await.unwrap(),
            Some(Vec::new())
        );
        let mut trailing = [0; 9];
        trailing[8] = 1;
        assert!(recv_blob_response(&mut trailing.as_slice()).await.is_err());
    }
}
