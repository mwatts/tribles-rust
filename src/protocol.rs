//! Binary wire protocol types and helpers.
//!
//! One QUIC stream per operation. The first byte identifies the operation,
//! followed by the request payload. The response follows on the same stream.
//! Stream FIN signals completion — no explicit DONE framing needed.
//!
//! Operations:
//!   LIST       → (branch_id:16, head:32)*  END
//!   HEAD       branch_id:16 → hash:32 | NONE
//!   GET_BLOB   hash:32 → len:32 data | MISSING
//!   CHILDREN   parent:32 have_count:32 have* → (hash:32 len:32 data)* END
//!   CAS_PUSH   (reserved)

pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/2";

// Operation types — first byte on each stream.
pub const OP_LIST: u8 = 0x01;
pub const OP_GET_BLOB: u8 = 0x02;
pub const OP_CHILDREN: u8 = 0x03;
pub const OP_HEAD: u8 = 0x04;
pub const OP_CAS_PUSH: u8 = 0x05;

// Response markers (inline in the stream).
pub const RSP_BLOB: u8 = 0x01;
pub const RSP_MISSING: u8 = 0x02;
pub const RSP_HEAD_OK: u8 = 0x03;
pub const RSP_NONE: u8 = 0x04;
pub const RSP_END: u8 = 0x00;       // end of list/children sequence
pub const RSP_CAS_OK: u8 = 0x05;    // CAS_PUSH succeeded
pub const RSP_CAS_CONFLICT: u8 = 0x06; // CAS_PUSH conflict, followed by current head:32

pub type RawHash = [u8; 32];
pub type RawBranchId = [u8; 16];

// ── Send helpers ─────────────────────────────────────────────────────

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

// ── Single-stream operations (client side) ───────────────────────────

/// LIST: open stream, get all (branch_id, head_hash) pairs.
pub async fn op_list(conn: &Connection) -> Result<Vec<(RawBranchId, RawHash)>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_LIST).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let mut branches = Vec::new();
    loop {
        let marker = recv_u8(&mut recv).await?;
        if marker == RSP_END { break; }
        let id = recv_branch_id(&mut recv).await?;
        let head = recv_hash(&mut recv).await?;
        branches.push((id, head));
    }
    Ok(branches)
}

/// HEAD: query head hash for a specific branch.
pub async fn op_head(conn: &Connection, branch_id: &RawBranchId) -> Result<Option<RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_HEAD).await?;
    send_branch_id(&mut send, branch_id).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let rsp = recv_u8(&mut recv).await?;
    match rsp {
        RSP_HEAD_OK => Ok(Some(recv_hash(&mut recv).await?)),
        RSP_NONE => Ok(None),
        _ => Err(anyhow!("unexpected head response: {rsp}")),
    }
}

/// GET_BLOB: fetch a single blob by hash.
pub async fn op_get_blob(conn: &Connection, hash: &RawHash) -> Result<Option<Vec<u8>>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_GET_BLOB).await?;
    send_hash(&mut send, hash).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let rsp = recv_u8(&mut recv).await?;
    match rsp {
        RSP_BLOB => {
            let len = recv_u32_be(&mut recv).await? as usize;
            let mut data = vec![0u8; len];
            recv.read_exact(&mut data).await.map_err(|e| anyhow!("recv: {e}"))?;
            Ok(Some(data))
        }
        RSP_MISSING => Ok(None),
        _ => Err(anyhow!("unexpected blob response: {rsp}")),
    }
}

/// CAS_PUSH: compare-and-swap a branch head on the remote.
/// Returns Ok(true) on success, Ok(false) + current head on conflict.
pub async fn op_cas_push(
    conn: &Connection,
    branch_id: &RawBranchId,
    old: Option<&RawHash>,
    new: &RawHash,
) -> Result<std::result::Result<(), RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_CAS_PUSH).await?;
    send_branch_id(&mut send, branch_id).await?;
    // old: 32 bytes, zero = "create new branch"
    match old {
        Some(h) => send_hash(&mut send, h).await?,
        None => send_hash(&mut send, &[0u8; 32]).await?,
    }
    send_hash(&mut send, new).await?;
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let rsp = recv_u8(&mut recv).await?;
    match rsp {
        RSP_CAS_OK => Ok(Ok(())),
        RSP_CAS_CONFLICT => {
            let current = recv_hash(&mut recv).await?;
            Ok(Err(current))
        }
        _ => Err(anyhow!("unexpected cas_push response: {rsp}")),
    }
}

/// CHILDREN: given a parent blob and a HAVE set, receive child hashes
/// that the remote has but aren't in the HAVE set.
/// Returns hashes only — blob data is fetched separately via GET_BLOB.
pub async fn op_children(
    conn: &Connection,
    parent: &RawHash,
    have: &[RawHash],
) -> Result<Vec<RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_CHILDREN).await?;
    send_hash(&mut send, parent).await?;
    send_u32_be(&mut send, have.len() as u32).await?;
    for h in have {
        send_hash(&mut send, h).await?;
    }
    send.finish().map_err(|e| anyhow!("finish: {e}"))?;

    let mut children = Vec::new();
    loop {
        let marker = recv_u8(&mut recv).await?;
        if marker == RSP_END { break; }
        children.push(recv_hash(&mut recv).await?);
    }
    Ok(children)
}
