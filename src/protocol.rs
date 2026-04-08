//! Binary wire protocol types and helpers.
//!
//! All messages are fixed-width headers + optional payload.
//! No text encoding, no newlines — raw bytes throughout.

pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/1";

// Request types (client → server)
pub const REQ_DONE: u8 = 0x00;
pub const REQ_LIST: u8 = 0x01;
pub const REQ_GET_BLOB: u8 = 0x02;
pub const REQ_SYNC: u8 = 0x03;
pub const REQ_HEAD: u8 = 0x04;
pub const REQ_CAS_PUSH: u8 = 0x05;

// Response types (server → client)
pub const RSP_LIST_ENTRY: u8 = 0x01;
pub const RSP_END_LIST: u8 = 0x02;
pub const RSP_BLOB: u8 = 0x03;
pub const RSP_MISSING: u8 = 0x04;
pub const RSP_END_SYNC: u8 = 0x05;
pub const RSP_HEAD_OK: u8 = 0x06;
pub const RSP_NONE: u8 = 0x07;
pub const RSP_CAS_OK: u8 = 0x08;
pub const RSP_CAS_CONFLICT: u8 = 0x09;

pub type RawHash = [u8; 32];
pub type RawBranchId = [u8; 16];

// ── Send helpers ─────────────────────────────────────────────────────

use anyhow::{Result, anyhow};
use iroh::endpoint::{SendStream, RecvStream};

pub async fn send_u8(send: &mut SendStream, v: u8) -> Result<()> {
    send.write_all(&[v]).await.map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_hash(send: &mut SendStream, hash: &RawHash) -> Result<()> {
    send.write_all(hash).await.map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_branch_id(send: &mut SendStream, id: &RawBranchId) -> Result<()> {
    send.write_all(id).await.map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_u32_be(send: &mut SendStream, v: u32) -> Result<()> {
    send.write_all(&v.to_be_bytes()).await.map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_blob(send: &mut SendStream, hash: &RawHash, data: &[u8]) -> Result<()> {
    send_u8(send, RSP_BLOB).await?;
    send_hash(send, hash).await?;
    send_u32_be(send, data.len() as u32).await?;
    send.write_all(data).await.map_err(|e| anyhow!("send: {e}"))
}

// ── Recv helpers ─────────────────────────────────────────────────────

pub async fn recv_u8(recv: &mut RecvStream) -> Result<u8> {
    let mut buf = [0u8; 1];
    recv.read_exact(&mut buf).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok(buf[0])
}

pub async fn recv_hash(recv: &mut RecvStream) -> Result<RawHash> {
    let mut buf = [0u8; 32];
    recv.read_exact(&mut buf).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok(buf)
}

pub async fn recv_branch_id(recv: &mut RecvStream) -> Result<RawBranchId> {
    let mut buf = [0u8; 16];
    recv.read_exact(&mut buf).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok(buf)
}

pub async fn recv_u32_be(recv: &mut RecvStream) -> Result<u32> {
    let mut buf = [0u8; 4];
    recv.read_exact(&mut buf).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok(u32::from_be_bytes(buf))
}

pub async fn recv_blob_data(recv: &mut RecvStream) -> Result<(RawHash, Vec<u8>)> {
    let hash = recv_hash(recv).await?;
    let len = recv_u32_be(recv).await? as usize;
    let mut data = vec![0u8; len];
    recv.read_exact(&mut data).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok((hash, data))
}
