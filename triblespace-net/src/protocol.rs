//! Binary wire protocol types and helpers.
//!
//! One QUIC stream per operation. The first byte identifies the operation,
//! followed by the request payload. The response follows on the same stream.
//! Stream FIN signals completion — no explicit DONE framing needed.
//!
//! Auth: the first stream on an ordinary connection must be
//! `OP_AUTH(cap_handle)`. The only pre-auth exception is the one-shot,
//! bounded `OP_CAPABILITY_PROOF(head)` bootstrap operation; that connection is
//! closed after its response. The server walks an auth chain back to the
//! configured team root and caches the verified permissions for later streams,
//! rechecking expiry on every operation.
//!
//! Nil sentinels: nil id ([0u8; 16]) and nil hash ([0u8; 32]) terminate
//! sequences. P(collision) = 2^(-128) / 2^(-256). Content-addressed systems
//! already assume hash uniqueness — nil sentinels are the same assumption.
//!
//! Operations:
//!   AUTH       cap_handle:32 → resp:u8                (0x00 = OK, 0x01 = REJECTED)
//!   GET_BLOB   hash:32 → len:u64 data                (u64::MAX = missing)
//!   CHILDREN   parent:32 → hash* nil                  (nil = end)
//!   COLLECTION_EVIDENCE collection:32 → count:u32 evidence[count]
//!                  (`u32::MAX` = read capability required)
//!   COLLECTION_OPERATION_RECEIPTS request:97 → count:u32 receipt[count]
//!                  (`u32::MAX` = rejected; each receipt is 128 bytes)
//!   CAPABILITY_PROOF head:32 known_count:u32 (hash:32 len:u32 data)*
//!                  → count:u32 (hash:32 len:u32 data)*
//!                  (`u32::MAX` = rejected; bounded, pre-auth, one-shot)
//!   (protocol is read-only — no remote writes)
//!
//! Branch-state operations are retired. Immutable grant-backed collection
//! commits are discovered through the team gossip mesh; content and exact
//! receipts remain explicitly fetched through this read-only protocol.

pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/5";

// Operation types — first byte on each stream.
// 0x01 was the retired branch-list operation.
pub const OP_GET_BLOB: u8 = 0x02;
pub const OP_CHILDREN: u8 = 0x03;
// 0x04 was the retired branch-head operation.
/// First stream on every ordinary connection. Body: cap_handle:32. Response:
/// u8 status (`AUTH_OK` or `AUTH_REJECTED`). Connection state caches the
/// verified permissions; subsequent ops on the same connection inherit them.
pub const OP_AUTH: u8 = 0x05;
/// Enumerate grant-backed signed commits for one exact 32-byte collection
/// descriptor handle. The response framing and strict evidence codec live in
/// [`crate::collection_wire`].
pub const OP_COLLECTION_EVIDENCE: u8 = 0x06;
/// Ask for every locally known exact `MERGE` or `DERIVE` receipt answering one
/// canonical 97-byte [`triblespace_core::repo::WantRequest`]. Responses carry
/// full untagged 128-byte records; the request kind supplies their type.
pub const OP_COLLECTION_OPERATION_RECEIPTS: u8 = 0x07;
/// Obtain the complete, bounded capability proof rooted at one exact sig
/// handle. This is the sole operation permitted before authentication besides
/// `OP_AUTH` itself, breaking the otherwise-circular bootstrap between two
/// members that each hold only their own private credential chain.
pub const OP_CAPABILITY_PROOF: u8 = 0x08;
// CAS_PUSH removed: the data model is monotonic (set union), and immutable
// collection records travel as evidence rather than remote mutable-head
// writes. The request/response protocol is read-only.

/// Auth response: capability verified, all subsequent ops on this
/// connection are scope-gated by the verified cap.
pub const AUTH_OK: u8 = 0x00;
/// Auth response: capability did not verify (chain malformed, signature
/// failed, expired, scope-not-subset, fetch failed for any link, etc.).
/// The connection should be closed by the client.
pub const AUTH_REJECTED: u8 = 0x01;

/// `OP_COLLECTION_EVIDENCE` response sentinel: the authenticated capability
/// lacks read permission.
pub const COLLECTION_EVIDENCE_REJECTED: u32 = u32::MAX;
/// `OP_COLLECTION_OPERATION_RECEIPTS` response sentinel. This covers both
/// authorization rejection and structurally invalid requests; callers learn
/// no collection evidence from either case.
pub const COLLECTION_OPERATION_RECEIPTS_REJECTED: u32 = u32::MAX;
/// `OP_CAPABILITY_PROOF` response sentinel: the requested handle is not a
/// complete, currently valid capability proof rooted in this team.
pub const CAPABILITY_PROOF_REJECTED: u32 = u32::MAX;

/// Capability proof blobs are canonical SimpleArchives produced by the team
/// machinery. This bound applies only to pre-auth proof transfer, never to
/// ordinary content blobs.
pub const MAX_CAPABILITY_PROOF_BLOB_BYTES: usize = 16 * 1024;
/// One leaf sig blob plus at most one cap per verifier depth.
pub const MAX_CAPABILITY_PROOF_ITEMS: usize =
    triblespace_core::repo::capability::MAX_CHAIN_DEPTH + 2;

pub const NIL_HASH: RawHash = [0u8; 32];

pub type RawHash = [u8; 32];

// ── Send/Recv helpers ────────────────────────────────────────────────
//
// Generic over `tokio::io::{AsyncRead, AsyncWrite}` so the same wire
// code runs over iroh QUIC streams (production) and in-memory duplex
// pipes (deterministic simulation). `SendStream::finish()` from the
// pre-seam code maps to `AsyncWriteExt::shutdown()` — iroh's QUIC
// send-stream implements `poll_shutdown` as finish.

use anyhow::{Result, anyhow};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::transport::Conn;

pub async fn send_u8<W: AsyncWrite + Unpin>(send: &mut W, v: u8) -> Result<()> {
    send.write_all(&[v]).await.map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_hash<W: AsyncWrite + Unpin>(send: &mut W, hash: &RawHash) -> Result<()> {
    send.write_all(hash).await.map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_u32_be<W: AsyncWrite + Unpin>(send: &mut W, v: u32) -> Result<()> {
    send.write_all(&v.to_be_bytes())
        .await
        .map_err(|e| anyhow!("send: {e}"))
}

pub async fn send_u64_be<W: AsyncWrite + Unpin>(send: &mut W, v: u64) -> Result<()> {
    send.write_all(&v.to_be_bytes())
        .await
        .map_err(|e| anyhow!("send: {e}"))
}

pub async fn recv_u8<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| anyhow!("recv: {e}"))?;
    Ok(buf[0])
}

pub async fn recv_hash<R: AsyncRead + Unpin>(recv: &mut R) -> Result<RawHash> {
    let mut buf = [0u8; 32];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| anyhow!("recv: {e}"))?;
    Ok(buf)
}

pub async fn recv_u32_be<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| anyhow!("recv: {e}"))?;
    Ok(u32::from_be_bytes(buf))
}

pub async fn recv_u64_be<R: AsyncRead + Unpin>(recv: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    recv.read_exact(&mut buf)
        .await
        .map_err(|e| anyhow!("recv: {e}"))?;
    Ok(u64::from_be_bytes(buf))
}

// ── Single-stream operations (client side) ───────────────────────────

/// AUTH: present a capability handle. Must be the first stream opened
/// on every new connection. Returns `Ok(())` if the server accepted the
/// capability and the connection is authorised for subsequent ops.
pub async fn op_auth<C: Conn>(conn: &C, cap_handle: &RawHash) -> Result<()> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_AUTH).await?;
    send_hash(&mut send, cap_handle).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;
    let resp = recv_u8(&mut recv).await?;
    match resp {
        AUTH_OK => Ok(()),
        AUTH_REJECTED => Err(anyhow!("server rejected capability")),
        other => Err(anyhow!("unknown auth response: {other:#x}")),
    }
}

/// GET_BLOB: fetch a single blob by hash.
/// Response: len:u64 + data. len=u64::MAX means missing.
/// Supports empty blobs (len=0) and every blob representable in this process's
/// address space. Receive storage grows fallibly in transferred-size chunks;
/// the pre-auth proof path has its own strict framing bound.
pub async fn op_get_blob<C: Conn>(conn: &C, hash: &RawHash) -> Result<Option<Vec<u8>>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_GET_BLOB).await?;
    send_hash(&mut send, hash).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;

    recv_blob_response(&mut recv).await
}

async fn recv_blob_response<R: AsyncRead + Unpin>(recv: &mut R) -> Result<Option<Vec<u8>>> {
    let len = recv_u64_be(recv).await?;
    if len == u64::MAX {
        return Ok(None);
    }
    let len = usize::try_from(len)
        .map_err(|_| anyhow!("blob response length does not fit this address space"))?;
    // Grow only as bytes are about to be read. A hostile peer can announce a
    // huge u64 length, but it cannot force one huge eager allocation without
    // actually transferring that much content. `try_reserve_exact` turns
    // allocator refusal into an ordinary protocol error rather than aborting.
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
            .map_err(|e| anyhow!("recv: {e}"))?;
    }
    Ok(Some(data))
}

/// CHILDREN: get child hashes of a parent blob. Nil hash terminates.
pub async fn op_children<C: Conn>(conn: &C, parent: &RawHash) -> Result<Vec<RawHash>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_CHILDREN).await?;
    send_hash(&mut send, parent).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;

    recv_children_response(&mut recv).await
}

async fn recv_children_response<R: AsyncRead + Unpin>(recv: &mut R) -> Result<Vec<RawHash>> {
    let mut children = Vec::new();
    loop {
        let hash = recv_hash(recv).await?;
        if hash == NIL_HASH {
            break;
        }
        children
            .try_reserve(1)
            .map_err(|error| anyhow!("cannot allocate child response: {error}"))?;
        children.push(hash);
    }
    Ok(children)
}

/// CAPABILITY_PROOF: fetch one complete, bounded credential proof without
/// first authenticating this connection.
///
/// The server validates the full chain before releasing it and closes the
/// one-shot connection afterwards. The client independently checks framing,
/// ordering, content hashes, and ultimately the chain itself in `verify_chain`.
pub async fn op_capability_proof<C: Conn>(
    conn: &C,
    head: &RawHash,
    known: &std::collections::BTreeMap<RawHash, Vec<u8>>,
) -> Result<Option<std::collections::BTreeMap<RawHash, Vec<u8>>>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_CAPABILITY_PROOF).await?;
    send_hash(&mut send, head).await?;
    send_capability_proof_items(&mut send, known).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;

    recv_capability_proof_response(&mut recv).await
}

async fn send_capability_proof_items<W: AsyncWrite + Unpin>(
    send: &mut W,
    items: &std::collections::BTreeMap<RawHash, Vec<u8>>,
) -> Result<()> {
    if items.len() > MAX_CAPABILITY_PROOF_ITEMS {
        return Err(anyhow!("too many capability proof items"));
    }
    if items.iter().any(|(hash, bytes)| {
        bytes.len() > MAX_CAPABILITY_PROOF_BLOB_BYTES || *blake3::hash(bytes).as_bytes() != *hash
    }) {
        return Err(anyhow!("invalid locally supplied capability proof item"));
    }
    send_u32_be(
        send,
        u32::try_from(items.len()).expect("proof item count is statically bounded"),
    )
    .await?;
    for (hash, bytes) in items {
        send_hash(send, hash).await?;
        send_u32_be(
            send,
            u32::try_from(bytes.len()).expect("proof blob length is statically bounded"),
        )
        .await?;
        send.write_all(bytes)
            .await
            .map_err(|error| anyhow!("send capability proof item: {error}"))?;
    }
    Ok(())
}

pub(crate) async fn recv_capability_proof_request<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<std::collections::BTreeMap<RawHash, Vec<u8>>> {
    let count = recv_u32_be(recv).await?;
    recv_capability_proof_items(recv, count as usize, true).await
}

async fn recv_capability_proof_response<R: AsyncRead + Unpin>(
    recv: &mut R,
) -> Result<Option<std::collections::BTreeMap<RawHash, Vec<u8>>>> {
    let count = recv_u32_be(recv).await?;
    if count == CAPABILITY_PROOF_REJECTED {
        return Ok(None);
    }
    let count = usize::try_from(count).map_err(|_| anyhow!("proof count does not fit usize"))?;
    Ok(Some(recv_capability_proof_items(recv, count, false).await?))
}

async fn recv_capability_proof_items<R: AsyncRead + Unpin>(
    recv: &mut R,
    count: usize,
    allow_empty: bool,
) -> Result<std::collections::BTreeMap<RawHash, Vec<u8>>> {
    if (!allow_empty && count == 0) || count > MAX_CAPABILITY_PROOF_ITEMS {
        return Err(anyhow!(
            "capability proof item count {count} is outside the valid bound"
        ));
    }

    let mut proof = std::collections::BTreeMap::new();
    let mut previous = None;
    for _ in 0..count {
        let hash = recv_hash(recv).await?;
        if previous.is_some_and(|previous| hash <= previous) {
            return Err(anyhow!("capability proof handles are not strictly ordered"));
        }
        previous = Some(hash);

        let len = recv_u32_be(recv).await? as usize;
        if len > MAX_CAPABILITY_PROOF_BLOB_BYTES {
            return Err(anyhow!(
                "capability proof blob length {len} exceeds {MAX_CAPABILITY_PROOF_BLOB_BYTES}"
            ));
        }
        let mut bytes = vec![0u8; len];
        recv.read_exact(&mut bytes)
            .await
            .map_err(|e| anyhow!("recv: {e}"))?;
        if blake3::hash(&bytes).as_bytes() != &hash {
            return Err(anyhow!("capability proof blob content hash mismatch"));
        }
        proof.insert(hash, bytes);
    }
    Ok(proof)
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    #[tokio::test]
    async fn capability_proof_rejects_oversized_item_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&[7u8; 32]);
        bytes.extend_from_slice(&((MAX_CAPABILITY_PROOF_BLOB_BYTES + 1) as u32).to_be_bytes());
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_response(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn ordinary_blob_response_is_not_subject_to_auth_proof_bound() {
        let payload = vec![7u8; 65];
        let mut bytes = Vec::with_capacity(8 + payload.len());
        bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        bytes.extend_from_slice(&payload);
        let mut input = bytes.as_slice();
        assert_eq!(recv_blob_response(&mut input).await.unwrap(), Some(payload));
    }

    #[tokio::test]
    async fn hostile_huge_blob_length_does_not_trigger_eager_allocation() {
        let bytes = (u64::MAX - 1).to_be_bytes();
        let mut input = bytes.as_slice();
        assert!(recv_blob_response(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn capability_proof_response_checks_content_hash() {
        let payload = b"not the named content";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&[7u8; 32]);
        bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        bytes.extend_from_slice(payload);
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_response(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn capability_proof_request_rejects_oversized_known_item() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&[7u8; 32]);
        bytes.extend_from_slice(&((MAX_CAPABILITY_PROOF_BLOB_BYTES + 1) as u32).to_be_bytes());
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_request(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn capability_proof_request_rejects_unordered_known_items() {
        let first = b"first";
        let second = b"second";
        let mut items = [
            (*blake3::hash(first).as_bytes(), first.as_slice()),
            (*blake3::hash(second).as_bytes(), second.as_slice()),
        ];
        items.sort_by_key(|(hash, _)| std::cmp::Reverse(*hash));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_be_bytes());
        for (hash, payload) in items {
            bytes.extend_from_slice(&hash);
            bytes.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            bytes.extend_from_slice(payload);
        }
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_request(&mut input).await.is_err());
    }
}
