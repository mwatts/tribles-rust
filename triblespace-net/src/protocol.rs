//! Binary wire protocol types and helpers.
//!
//! One QUIC stream per operation. The first byte identifies the operation,
//! followed by the request payload. The response follows on the same stream.
//! Stream FIN signals completion — no explicit DONE framing needed.
//!
//! Auth: the first stream on every connection must be `OP_AUTH(proof)`. The
//! proof is a complete, bounded, root-to-leaf [`CapabilityProof`] for
//! [`ACTION_CONNECT`] on the configured 32-byte CONNECT resource. It is
//! verified entirely from the bytes on that stream; authentication never
//! fetches or persists ambient state.
//!
//! Nil sentinels: nil id ([0u8; 16]) and nil hash ([0u8; 32]) terminate
//! sequences. P(collision) = 2^(-128) / 2^(-256). Content-addressed systems
//! already assume hash uniqueness — nil sentinels are the same assumption.
//!
//! Operations:
//!   AUTH       proof_len:u32 proof:bytes → resp:u8    (0x00 = OK, 0x01 = REJECTED)
//!   GET_BLOB   hash:32 → len:u64 data                (u64::MAX = missing)
//!   CHILDREN   parent:32 → hash* nil                  (nil = end)
//!   COLLECTION_EVIDENCE collection:32 → count:u32 evidence[count]
//!   COLLECTION_OPERATION_RECEIPTS request:97 → count:u32 receipt[count]
//!                  (each receipt is 128 bytes)
//!   (protocol is read-only — no remote writes)
//!
//! Branch-state operations are retired. Immutable signed collection
//! commits are discovered through the configured gossip mesh; content and exact
//! receipts remain explicitly fetched through this read-only protocol.

pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/7";

use triblespace_core::id::{Id, id_hex};

/// Permission to establish an authenticated direct-RPC connection.
///
/// Minted on 2026-08-23 CEST with the exact command `trible genid`, whose
/// output was `9685583C6ADD2A5F5309F9504F46ABC3`.
///
/// Its resource is the exact 32-byte identity selected by the application
/// (for teams, the trust-root public key bytes). It grants no collection write
/// admission, disclosure, gossip, custody, or retention authority.
pub const ACTION_CONNECT: Id = id_hex!("9685583C6ADD2A5F5309F9504F46ABC3");

/// Exact capability atom required for direct RPC under `connect_root`.
///
/// Authorization and gossip rendezvous are deliberately independent: this
/// resource is always the trust-root public key bytes, regardless of which
/// (if any) gossip topic a peer joins.
pub fn connect_capability_atom(
    connect_root: ed25519_dalek::VerifyingKey,
) -> triblespace_core::capability::CapabilityAtom {
    triblespace_core::capability::CapabilityAtom::new(
        ACTION_CONNECT.into(),
        triblespace_core::capability::CapabilityResource::new(connect_root.to_bytes()),
    )
}

// Operation types — first byte on each stream.
// 0x01 was the retired branch-list operation.
pub const OP_GET_BLOB: u8 = 0x02;
pub const OP_CHILDREN: u8 = 0x03;
// 0x04 was the retired branch-head operation.
/// First stream on every connection. Body: one length-prefixed canonical
/// capability proof. Response: u8 status (`AUTH_OK` or `AUTH_REJECTED`).
pub const OP_AUTH: u8 = 0x05;
/// Enumerate signed commits for one exact 32-byte collection
/// descriptor handle. The response framing and strict evidence codec live in
/// [`crate::collection_wire`].
pub const OP_COLLECTION_EVIDENCE: u8 = 0x06;
/// Ask for every locally known exact `MERGE` or `DERIVE` receipt answering one
/// canonical 97-byte [`triblespace_core::repo::WantRequest`]. Responses carry
/// full untagged 128-byte records; the request kind supplies their type.
pub const OP_COLLECTION_OPERATION_RECEIPTS: u8 = 0x07;
// CAS_PUSH removed: the data model is monotonic (set union), and immutable
// collection records travel as evidence rather than remote mutable-head
// writes. The request/response protocol is read-only.

/// Auth response: CONNECT capability verified. Subsequent direct RPCs on this
/// connection may proceed.
pub const AUTH_OK: u8 = 0x00;
/// Auth response: the inline proof was malformed or did not authorize the TLS
/// peer to CONNECT. The connection should be closed by the client.
pub const AUTH_REJECTED: u8 = 0x01;

/// Version of the standalone capability-proof byte codec.
pub const CAPABILITY_PROOF_WIRE_VERSION: u8 = 2;
/// Transport-local bound on delegation depth, derived directly from the
/// one-byte step count in this protocol version. The capability algebra itself
/// remains unbounded.
pub const MAX_CAPABILITY_PROOF_STEPS: usize = u8::MAX as usize;
/// Largest closed canonical capability claim: seven tribles.
pub const MAX_CAPABILITY_CLAIM_BYTES: usize = 7 * triblespace_core::trible::TRIBLE_LEN;
/// Exact closed canonical capability signature: five tribles.
pub const MAX_CAPABILITY_SIGNATURE_BYTES: usize = 5 * triblespace_core::trible::TRIBLE_LEN;
const CAPABILITY_PROOF_HEADER_BYTES: usize = 2;
const CAPABILITY_PROOF_STEP_HEADER_BYTES: usize = 4;
/// Largest complete proof frame accepted by this transport.
pub const MAX_CAPABILITY_PROOF_BYTES: usize = CAPABILITY_PROOF_HEADER_BYTES
    + MAX_CAPABILITY_PROOF_STEPS
        * (CAPABILITY_PROOF_STEP_HEADER_BYTES
            + MAX_CAPABILITY_CLAIM_BYTES
            + MAX_CAPABILITY_SIGNATURE_BYTES);

pub const NIL_HASH: RawHash = [0u8; 32];

pub type RawHash = [u8; 32];

use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::capability::{CapabilityProof, CapabilityProofStep};

/// Structural failure in the canonical bounded capability-proof codec.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CapabilityProofWireError {
    #[error("capability proof uses wire version {actual}, expected {expected}")]
    WrongVersion { expected: u8, actual: u8 },
    #[error("capability proof must contain at least one step")]
    Empty,
    #[error("capability proof has {count} steps; limit is {limit}")]
    TooManySteps { count: usize, limit: usize },
    #[error("capability proof step {step} claim is {bytes} bytes; limit is {limit}")]
    ClaimTooLarge {
        step: usize,
        bytes: usize,
        limit: usize,
    },
    #[error("capability proof step {step} signature is {bytes} bytes; limit is {limit}")]
    SignatureTooLarge {
        step: usize,
        bytes: usize,
        limit: usize,
    },
    #[error("capability proof frame is {bytes} bytes; limit is {limit}")]
    FrameTooLarge { bytes: usize, limit: usize },
    #[error("capability proof is truncated at byte {offset}")]
    Truncated { offset: usize },
    #[error("capability proof contains {bytes} trailing bytes")]
    TrailingBytes { bytes: usize },
    #[error("capability proof length arithmetic overflowed")]
    LengthOverflow,
}

/// Encode one capability proof into the canonical standalone byte format.
///
/// The format is `version:u8, count:u8, (claim_len:u16,
/// signature_len:u16, claim[claim_len], signature[signature_len])*`. It is
/// independent of QUIC and can be embedded unchanged in invite bundles.
/// Semantic trust remains exclusively in [`CapabilityProof::verify_claim`].
pub fn encode_capability_proof(
    proof: &CapabilityProof,
) -> Result<Vec<u8>, CapabilityProofWireError> {
    let count = proof.steps().len();
    validate_proof_count(count)?;

    let mut encoded_len = CAPABILITY_PROOF_HEADER_BYTES;
    for (step, item) in proof.steps().iter().enumerate() {
        let claim_len = item.claim().bytes.len();
        let signature_len = item.signature().bytes.len();
        validate_claim_len(step, claim_len)?;
        validate_signature_len(step, signature_len)?;
        encoded_len = encoded_len
            .checked_add(CAPABILITY_PROOF_STEP_HEADER_BYTES)
            .and_then(|len| len.checked_add(claim_len))
            .and_then(|len| len.checked_add(signature_len))
            .ok_or(CapabilityProofWireError::LengthOverflow)?;
    }
    validate_frame_len(encoded_len)?;

    let mut bytes = Vec::with_capacity(encoded_len);
    bytes.push(CAPABILITY_PROOF_WIRE_VERSION);
    bytes.push(count as u8);
    for item in proof.steps() {
        let claim = item.claim().bytes.as_ref();
        let signature = item.signature().bytes.as_ref();
        bytes.extend_from_slice(&(claim.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&(signature.len() as u16).to_be_bytes());
        bytes.extend_from_slice(claim);
        bytes.extend_from_slice(signature);
    }
    debug_assert_eq!(bytes.len(), encoded_len);
    Ok(bytes)
}

/// Decode one complete canonical bounded capability-proof byte string.
///
/// Framing is validated in a full first pass before the step vector or any
/// per-step blob is allocated. Cryptographic and authority semantics are
/// deliberately left to [`CapabilityProof::verify_claim`].
pub fn decode_capability_proof(bytes: &[u8]) -> Result<CapabilityProof, CapabilityProofWireError> {
    validate_frame_len(bytes.len())?;
    if bytes.len() < CAPABILITY_PROOF_HEADER_BYTES {
        return Err(CapabilityProofWireError::Truncated {
            offset: bytes.len(),
        });
    }
    if bytes[0] != CAPABILITY_PROOF_WIRE_VERSION {
        return Err(CapabilityProofWireError::WrongVersion {
            expected: CAPABILITY_PROOF_WIRE_VERSION,
            actual: bytes[0],
        });
    }
    let count = bytes[1] as usize;
    validate_proof_count(count)?;

    // First pass: validate every declared boundary before allocating output.
    let mut offset = CAPABILITY_PROOF_HEADER_BYTES;
    for step in 0..count {
        let header_end = offset
            .checked_add(CAPABILITY_PROOF_STEP_HEADER_BYTES)
            .ok_or(CapabilityProofWireError::LengthOverflow)?;
        if header_end > bytes.len() {
            return Err(CapabilityProofWireError::Truncated { offset });
        }
        let claim_len = u16::from_be_bytes([bytes[offset], bytes[offset + 1]]) as usize;
        let signature_len = u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]) as usize;
        validate_claim_len(step, claim_len)?;
        validate_signature_len(step, signature_len)?;
        let step_end = header_end
            .checked_add(claim_len)
            .and_then(|end| end.checked_add(signature_len))
            .ok_or(CapabilityProofWireError::LengthOverflow)?;
        if step_end > bytes.len() {
            return Err(CapabilityProofWireError::Truncated { offset: header_end });
        }
        offset = step_end;
    }
    if offset != bytes.len() {
        return Err(CapabilityProofWireError::TrailingBytes {
            bytes: bytes.len() - offset,
        });
    }

    // Second pass: exact framing has been proven bounded and complete.
    let mut steps = Vec::with_capacity(count);
    let mut offset = CAPABILITY_PROOF_HEADER_BYTES;
    for _ in 0..count {
        let claim_len = u16::from_be_bytes([bytes[offset], bytes[offset + 1]]) as usize;
        let signature_len = u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]) as usize;
        offset += CAPABILITY_PROOF_STEP_HEADER_BYTES;
        let claim_end = offset + claim_len;
        let claim = Blob::<SimpleArchive>::new(anybytes::Bytes::from_source(
            bytes[offset..claim_end].to_vec(),
        ));
        offset = claim_end;
        let signature_end = offset + signature_len;
        let signature = Blob::<SimpleArchive>::new(anybytes::Bytes::from_source(
            bytes[offset..signature_end].to_vec(),
        ));
        offset = signature_end;
        steps.push(CapabilityProofStep::new(claim, signature));
    }
    Ok(CapabilityProof::new(steps))
}

fn validate_proof_count(count: usize) -> Result<(), CapabilityProofWireError> {
    if count == 0 {
        return Err(CapabilityProofWireError::Empty);
    }
    if count > MAX_CAPABILITY_PROOF_STEPS {
        return Err(CapabilityProofWireError::TooManySteps {
            count,
            limit: MAX_CAPABILITY_PROOF_STEPS,
        });
    }
    Ok(())
}

fn validate_claim_len(step: usize, bytes: usize) -> Result<(), CapabilityProofWireError> {
    if bytes > MAX_CAPABILITY_CLAIM_BYTES {
        return Err(CapabilityProofWireError::ClaimTooLarge {
            step,
            bytes,
            limit: MAX_CAPABILITY_CLAIM_BYTES,
        });
    }
    Ok(())
}

fn validate_signature_len(step: usize, bytes: usize) -> Result<(), CapabilityProofWireError> {
    if bytes > MAX_CAPABILITY_SIGNATURE_BYTES {
        return Err(CapabilityProofWireError::SignatureTooLarge {
            step,
            bytes,
            limit: MAX_CAPABILITY_SIGNATURE_BYTES,
        });
    }
    Ok(())
}

fn validate_frame_len(bytes: usize) -> Result<(), CapabilityProofWireError> {
    if bytes > MAX_CAPABILITY_PROOF_BYTES {
        return Err(CapabilityProofWireError::FrameTooLarge {
            bytes,
            limit: MAX_CAPABILITY_PROOF_BYTES,
        });
    }
    Ok(())
}

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

/// Write one length-prefixed capability proof.
pub async fn send_capability_proof<W: AsyncWrite + Unpin>(
    send: &mut W,
    proof: &CapabilityProof,
) -> Result<()> {
    let bytes = encode_capability_proof(proof)?;
    send_u32_be(
        send,
        u32::try_from(bytes.len()).expect("proof frame has a static usize bound below u32::MAX"),
    )
    .await?;
    send.write_all(&bytes)
        .await
        .map_err(|error| anyhow!("send capability proof: {error}"))
}

/// Read one length-prefixed capability proof with count and byte bounds checked
/// before allocating its frame buffer.
pub async fn recv_capability_proof<R: AsyncRead + Unpin>(recv: &mut R) -> Result<CapabilityProof> {
    let frame_len = recv_u32_be(recv).await? as usize;
    validate_frame_len(frame_len)?;
    if frame_len < CAPABILITY_PROOF_HEADER_BYTES {
        return Err(CapabilityProofWireError::Truncated { offset: frame_len }.into());
    }

    let mut header = [0u8; CAPABILITY_PROOF_HEADER_BYTES];
    recv.read_exact(&mut header)
        .await
        .map_err(|error| anyhow!("recv capability proof header: {error}"))?;
    if header[0] != CAPABILITY_PROOF_WIRE_VERSION {
        return Err(CapabilityProofWireError::WrongVersion {
            expected: CAPABILITY_PROOF_WIRE_VERSION,
            actual: header[0],
        }
        .into());
    }
    let count = header[1] as usize;
    validate_proof_count(count)?;
    let minimum = CAPABILITY_PROOF_HEADER_BYTES
        + count
            .checked_mul(CAPABILITY_PROOF_STEP_HEADER_BYTES)
            .ok_or(CapabilityProofWireError::LengthOverflow)?;
    let maximum = CAPABILITY_PROOF_HEADER_BYTES
        + count
            .checked_mul(
                CAPABILITY_PROOF_STEP_HEADER_BYTES
                    + MAX_CAPABILITY_CLAIM_BYTES
                    + MAX_CAPABILITY_SIGNATURE_BYTES,
            )
            .ok_or(CapabilityProofWireError::LengthOverflow)?;
    if frame_len < minimum {
        return Err(CapabilityProofWireError::Truncated { offset: frame_len }.into());
    }
    if frame_len > maximum {
        return Err(CapabilityProofWireError::FrameTooLarge {
            bytes: frame_len,
            limit: maximum,
        }
        .into());
    }

    let mut bytes = Vec::with_capacity(frame_len);
    bytes.extend_from_slice(&header);
    bytes.resize(frame_len, 0);
    recv.read_exact(&mut bytes[CAPABILITY_PROOF_HEADER_BYTES..])
        .await
        .map_err(|error| anyhow!("recv capability proof body: {error}"))?;
    Ok(decode_capability_proof(&bytes)?)
}

/// AUTH: present a complete CONNECT proof. Must be the first stream opened on
/// every new connection.
pub async fn op_auth<C: Conn>(conn: &C, proof: &CapabilityProof) -> Result<()> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_AUTH).await?;
    send_capability_proof(&mut send, proof).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;
    let resp = recv_u8(&mut recv).await?;
    match resp {
        AUTH_OK => Ok(()),
        AUTH_REJECTED => Err(anyhow!("server rejected CONNECT proof")),
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

#[cfg(test)]
mod bounds_tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hifitime::Epoch;
    use triblespace_core::capability::{
        CapabilityAtom, CapabilityClaim, CapabilityGrant, CapabilityMode, CapabilityResource,
    };

    fn connect_proof() -> (SigningKey, SigningKey, CapabilityProof) {
        let root = SigningKey::from_bytes(&[0xA1; 32]);
        let delegate = SigningKey::from_bytes(&[0xA2; 32]);
        let peer = SigningKey::from_bytes(&[0xA3; 32]);
        let atom = CapabilityAtom::new(
            ACTION_CONNECT.into(),
            CapabilityResource::new(root.verifying_key().to_bytes()),
        );

        let parent = CapabilityProofStep::issue(
            &root,
            CapabilityGrant::root(
                delegate.verifying_key(),
                atom,
                CapabilityMode::InvokeAndDelegate,
                None,
            ),
        );
        let leaf = CapabilityProofStep::issue(
            &delegate,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                peer.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
        );

        (root, peer, CapabilityProof::new(vec![parent, leaf]))
    }

    #[test]
    fn capability_proof_codec_roundtrips_a_verified_delegation() {
        let (root, peer, proof) = connect_proof();
        let bytes = encode_capability_proof(&proof).unwrap();
        let decoded = decode_capability_proof(&bytes).unwrap();
        let atom = CapabilityAtom::new(
            ACTION_CONNECT.into(),
            CapabilityResource::new(root.verifying_key().to_bytes()),
        );

        assert_eq!(decoded, proof);
        let leaf = decoded
            .verify_claim(
                root.verifying_key(),
                Epoch::from_tai_seconds(0.0),
                CapabilityClaim::new(peer.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .unwrap();
        assert_eq!(leaf.grant().subject(), peer.verifying_key());
        assert_eq!(leaf.grant().atom(), atom);
        assert!(leaf.grant().mode().satisfies(CapabilityMode::Invoke));
    }

    #[tokio::test]
    async fn capability_proof_rejects_oversized_frame_before_reading_a_body() {
        let bytes = ((MAX_CAPABILITY_PROOF_BYTES + 1) as u32).to_be_bytes();
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn capability_proof_rejects_invalid_count_before_reading_a_body() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(CAPABILITY_PROOF_HEADER_BYTES as u32).to_be_bytes());
        bytes.extend_from_slice(&[CAPABILITY_PROOF_WIRE_VERSION, 0]);
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof(&mut input).await.is_err());
    }

    #[test]
    fn capability_proof_rejects_noncanonical_step_width_and_trailing_bytes() {
        let (_, _, proof) = connect_proof();
        let mut bytes = encode_capability_proof(&proof).unwrap();
        bytes.push(0);
        assert!(matches!(
            decode_capability_proof(&bytes),
            Err(CapabilityProofWireError::TrailingBytes { bytes: 1 })
        ));

        let mut oversized = vec![CAPABILITY_PROOF_WIRE_VERSION, 1];
        oversized.extend_from_slice(&((MAX_CAPABILITY_CLAIM_BYTES + 1) as u16).to_be_bytes());
        oversized.extend_from_slice(&0_u16.to_be_bytes());
        assert!(matches!(
            decode_capability_proof(&oversized),
            Err(CapabilityProofWireError::ClaimTooLarge { step: 0, .. })
        ));
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
}
