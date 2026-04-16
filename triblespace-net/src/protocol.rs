//! Binary wire protocol types and helpers.
//!
//! One QUIC stream per operation. The first byte identifies the operation,
//! followed by the request payload. The response follows on the same stream.
//! Stream FIN signals completion — no explicit DONE framing needed.
//!
//! Nil sentinels: nil id ([0u8; 16]) and nil hash ([0u8; 32]) terminate
//! sequences. P(collision) = 2^(-128) / 2^(-256). Content-addressed systems
//! already assume hash uniqueness — nil sentinels are the same assumption.
//!
//! Operations:
//!   LIST       → (id:16 head:32)* nil_id:16         (48-byte aligned entries)
//!   HEAD       id:16 → hash:32                      (nil = no head)
//!   GET_BLOB   hash:32 → len:u64 data                (u64::MAX = missing)
//!   CHILDREN   parent:32 → hash* nil                  (nil = end)
//!   (protocol is read-only — no remote writes)

pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/3";

// Operation types — first byte on each stream.
pub const OP_LIST: u8 = 0x01;
pub const OP_GET_BLOB: u8 = 0x02;
pub const OP_CHILDREN: u8 = 0x03;
pub const OP_HEAD: u8 = 0x04;
// CAS_PUSH removed: the data model is monotonic (set union), merge
// always succeeds, and each node manages its own branches locally.
// No remote writes needed — the protocol is read-only.

pub const NIL_HASH: RawHash = [0u8; 32];
pub const NIL_BRANCH_ID: RawBranchId = [0u8; 16];

pub type RawHash = [u8; 32];
pub type RawBranchId = [u8; 16];

// ── Send/Recv helpers ────────────────────────────────────────────────

use anyhow::{Result, anyhow};
use iroh::endpoint::{SendStream, RecvStream, Connection};

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

pub async fn send_u64_be(send: &mut SendStream, v: u64) -> Result<()> {
    send.write_all(&v.to_be_bytes()).await.map_err(|e| anyhow!("send: {e}"))
}

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

pub async fn recv_u64_be(recv: &mut RecvStream) -> Result<u64> {
    let mut buf = [0u8; 8];
    recv.read_exact(&mut buf).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok(u64::from_be_bytes(buf))
}

// ── Single-stream operations (client side) ───────────────────────────

/// LIST: get all (branch_id, head_hash) pairs. Nil branch_id terminates.
pub async fn op_list(conn: &Connection) -> Result<Vec<(RawBranchId, RawHash)>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_LIST).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let mut branches = Vec::new();
    loop {
        let id = recv_branch_id(&mut recv).await?;
        if id == NIL_BRANCH_ID { break; }
        let head = recv_hash(&mut recv).await?;
        branches.push((id, head));
    }
    Ok(branches)
}

/// HEAD: query head hash for a specific branch. Nil hash = no head.
pub async fn op_head(conn: &Connection, branch_id: &RawBranchId) -> Result<Option<RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_HEAD).await?;
    send_branch_id(&mut send, branch_id).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let hash = recv_hash(&mut recv).await?;
    if hash == NIL_HASH { Ok(None) } else { Ok(Some(hash)) }
}

/// GET_BLOB: fetch a single blob by hash.
/// Response: len:u64 + data. len=u64::MAX means missing.
/// Supports empty blobs (len=0) and blobs up to 2^64-2 bytes.
pub async fn op_get_blob(conn: &Connection, hash: &RawHash) -> Result<Option<Vec<u8>>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_GET_BLOB).await?;
    send_hash(&mut send, hash).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let len = recv_u64_be(&mut recv).await?;
    if len == u64::MAX { return Ok(None); }
    let mut data = vec![0u8; len as usize];
    recv.read_exact(&mut data).await.map_err(|e| anyhow!("recv: {e}"))?;
    Ok(Some(data))
}

/// CHILDREN: get child hashes of a parent blob. Nil hash terminates.
pub async fn op_children(
    conn: &Connection,
    parent: &RawHash,
) -> Result<Vec<RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_CHILDREN).await?;
    send_hash(&mut send, parent).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let mut children = Vec::new();
    loop {
        let hash = recv_hash(&mut recv).await?;
        if hash == NIL_HASH { break; }
        children.push(hash);
    }
    Ok(children)
}
