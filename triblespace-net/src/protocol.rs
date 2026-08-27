//! Binary wire protocol types and helpers.
//!
//! One QUIC stream per operation. The first byte identifies the operation,
//! followed by the request payload. The response follows on the same stream.
//! Stream FIN signals completion — no explicit DONE framing needed.
//!
//! Auth: the first stream on every connection must be `OP_AUTH(bundle)`. The
//! bundle is a complete, bounded, root-to-leaf [`CapabilityProofBundle`] for
//! [`ACTION_CONNECT`] on the configured 32-byte CONNECT resource. It is
//! verified entirely from the bytes on that stream; authentication never
//! fetches or persists ambient state.
//!
//! Operations:
//!   AUTH       bundle_len:u32 bundle:bytes → resp:u8 server_bundle
//!   GET_BLOB   hash:32 → len:u64 data                (u64::MAX = missing)
//!   PROVIDER_PUT artifact:32 → resp:u8               (caller identity is provider)
//!   PROVIDER_GET artifact:32 → count:u8 provider:[32; count]
//!   FIND_NODE target:32 → count:u8 peer:[32; count]
//!   INVENTORY_AUTH, MANIFEST, NODE, and BLOB_RANGE are defined by
//!   `inventory_wire`; they form the bounded SYNC_TEAM-authorized Merkle walk.
//!
//! Both authorization exchanges are mutual: an accepted server appends its own
//! subject-bound proof to the success response, and the client verifies that
//! proof before pooling the connection or sending the next phase. The initiating
//! CONNECT bundle necessarily crosses before the server proves its capability;
//! it is non-bearer evidence bound to the caller key and sent over TLS to the
//! exact identity the client intended to dial, but the proof itself is not
//! cryptographically bound to that receiver key. It is not a confidential or
//! zero-knowledge credential. No SYNC proof, artifact ID, query, or data request
//! is sent until reciprocal CONNECT verification succeeds.
//! Content and inventory operations additionally wait for reciprocal SYNC_TEAM.
//! `PROVIDER_PUT` is the one soft-state write: it binds the TLS-authenticated
//! caller to a receiver-local lease and never accepts a claimed provider identity.

pub const PILE_SYNC_ALPN: &[u8] = b"/triblespace/pile-sync/13";

use ed25519_dalek::VerifyingKey;
use hifitime::Epoch;
use triblespace_core::capability::{
    CapabilityProofBundle, CapabilityRequest, MAX_CAPABILITY_PROOF_BUNDLE_BYTES, VerifiedCapability,
};
use triblespace_core::id::{Id, id_hex};

/// Permission to establish an authenticated direct-RPC connection.
///
/// Minted on 2026-08-23 CEST with the exact command `trible genid`, whose
/// output was `9685583C6ADD2A5F5309F9504F46ABC3`.
///
/// Its resource is the exact team trust-root public key bytes. It admits the
/// transport connection only; disclosure additionally requires SYNC_TEAM.
pub const ACTION_CONNECT: Id = id_hex!("9685583C6ADD2A5F5309F9504F46ABC3");

/// Exact capability atom required for direct RPC under `connect_root`.
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
// 0x03 was the retired blob-children operation.
// 0x04 was the retired branch-head operation.
/// First stream on every connection. Body: one length-prefixed canonical
/// capability proof. A successful response carries the server's own bounded
/// CONNECT proof immediately after `AUTH_OK`.
pub const OP_AUTH: u8 = 0x05;
/// Renew the authenticated caller's receiver-local provider lease.
pub const OP_PROVIDER_PUT: u8 = 0x06;
/// Query live provider leases for one already-known, team-scoped artifact key.
pub const OP_PROVIDER_GET: u8 = 0x07;
/// Query up to K directly verified routes nearest one arbitrary XOR key.
pub const OP_FIND_NODE: u8 = 0x0C;

/// Auth response: CONNECT capability verified. Subsequent direct RPCs on this
/// connection may proceed.
pub const AUTH_OK: u8 = 0x00;
/// Auth response: the inline proof was malformed or did not authorize the TLS
/// peer to CONNECT. The connection should be closed by the client.
pub const AUTH_REJECTED: u8 = 0x01;

/// Provider hint was stored or renewed.
pub const PROVIDER_PUT_OK: u8 = 0x00;
/// Receiver-local directory capacity was exhausted.
pub const PROVIDER_PUT_FULL: u8 = 0x01;

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

/// Write one length-prefixed canonical capability proof bundle.
pub async fn send_capability_proof_bundle<W: AsyncWrite + Unpin>(
    send: &mut W,
    bundle: &CapabilityProofBundle,
) -> Result<()> {
    let bytes = bundle.to_bytes()?;
    debug_assert!(bytes.len() <= MAX_CAPABILITY_PROOF_BUNDLE_BYTES);
    send_u32_be(
        send,
        u32::try_from(bytes.len()).expect("bundle has a static usize bound below u32::MAX"),
    )
    .await?;
    send.write_all(&bytes)
        .await
        .map_err(|error| anyhow!("send capability proof bundle: {error}"))
}

/// Read one length-prefixed canonical capability proof bundle.
///
/// The core bundle maximum is enforced before any frame allocation. Exact
/// inner framing and canonical claim/proof decoding remain owned by core.
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

/// AUTH: present a complete CONNECT proof bundle. Must be the first stream
/// opened on every new connection.
pub async fn op_auth<C: Conn>(
    conn: &C,
    bundle: &CapabilityProofBundle,
    team: VerifyingKey,
    expected_remote: [u8; 32],
) -> Result<VerifiedCapability> {
    if conn.remote_id() != expected_remote {
        return Err(anyhow!(
            "dialed endpoint identity does not match the requested peer"
        ));
    }
    let remote = VerifyingKey::from_bytes(&expected_remote)
        .map_err(|error| anyhow!("invalid remote endpoint identity: {error}"))?;
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_AUTH).await?;
    send_capability_proof_bundle(&mut send, bundle).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;
    recv_proof_response(
        &mut recv,
        team,
        remote,
        CapabilityRequest::new(
            connect_capability_atom(team),
            triblespace_core::capability::CapabilityMode::Invoke,
        ),
        crate::clock::epoch_now(),
        "CONNECT",
    )
    .await
}

/// Verify one subject-bound endpoint proof under an explicit request and instant.
///
/// This is shared by startup validation and the two mutual-authentication
/// responses so the local and remote boundaries cannot drift semantically.
pub(crate) fn verify_endpoint_proof(
    bundle: &CapabilityProofBundle,
    team: VerifyingKey,
    endpoint: VerifyingKey,
    request: CapabilityRequest,
    now: Epoch,
    label: &'static str,
) -> Result<VerifiedCapability> {
    bundle
        .verify(team, now, endpoint, request)
        .map_err(|error| anyhow!("verify {label} endpoint proof: {error}"))
}

/// Decode and verify the proof carried by a successful authorization response.
/// The length is rejected before allocation and trailing bytes are forbidden.
pub(crate) async fn recv_proof_response<R: AsyncRead + Unpin>(
    recv: &mut R,
    team: VerifyingKey,
    expected_remote: VerifyingKey,
    request: CapabilityRequest,
    now: Epoch,
    label: &'static str,
) -> Result<VerifiedCapability> {
    match recv_u8(recv).await? {
        AUTH_OK => {
            let proof = recv_capability_proof_bundle(recv).await?;
            require_response_eof(recv).await?;
            verify_endpoint_proof(&proof, team, expected_remote, request, now, label)
        }
        AUTH_REJECTED => {
            require_response_eof(recv).await?;
            Err(anyhow!("server rejected {label} proof bundle"))
        }
        other => Err(anyhow!("unknown {label} auth response: {other:#x}")),
    }
}

/// GET_BLOB: fetch a single blob by hash after connection-local SYNC_TEAM auth.
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

/// PROVIDER_PUT: renew this connection's authenticated endpoint as provider.
/// The request intentionally contains no provider identity to forge.
pub async fn op_provider_put<C: Conn>(conn: &C, artifact: &RawHash) -> Result<bool> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_PROVIDER_PUT).await?;
    send_hash(&mut send, artifact).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;

    let stored = match recv_u8(&mut recv).await? {
        PROVIDER_PUT_OK => true,
        PROVIDER_PUT_FULL => false,
        other => return Err(anyhow!("unknown provider-put response: {other:#x}")),
    };
    require_response_eof(&mut recv).await?;
    Ok(stored)
}

/// PROVIDER_GET: return the receiver's bounded live provider hints.
pub async fn op_provider_get<C: Conn>(conn: &C, artifact: &RawHash) -> Result<Vec<[u8; 32]>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_PROVIDER_GET).await?;
    send_hash(&mut send, artifact).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;

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

/// FIND_NODE: return at most K directly authenticated routing identities
/// nearest an arbitrary 256-bit XOR target.
pub async fn op_find_node<C: Conn>(
    conn: &C,
    target: &crate::routing::RoutingKey,
) -> Result<Vec<crate::transport::PeerId>> {
    let (mut send, mut recv) = conn.open_bi().await.map_err(|e| anyhow!("open_bi: {e}"))?;
    send_u8(&mut send, OP_FIND_NODE).await?;
    send_hash(&mut send, target).await?;
    send.shutdown().await.map_err(|e| anyhow!("finish: {e}"))?;
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
    let mut trailing = [0u8; 1];
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

#[cfg(test)]
mod bounds_tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hifitime::Epoch;
    use triblespace_core::capability::{
        CAPABILITY_PROOF_BUNDLE_VERSION, CapabilityAtom, CapabilityClaim, CapabilityMode,
        CapabilityRequest, CapabilityResource, CapabilityValidity,
    };

    fn connect_proof_bundle() -> (SigningKey, SigningKey, CapabilityProofBundle) {
        let root = SigningKey::from_bytes(&[0xA1; 32]);
        let delegate = SigningKey::from_bytes(&[0xA2; 32]);
        let peer = SigningKey::from_bytes(&[0xA3; 32]);
        let atom = CapabilityAtom::new(
            ACTION_CONNECT.into(),
            CapabilityResource::new(root.verifying_key().to_bytes()),
        );

        let parent_bundle = CapabilityProofBundle::issue_root(
            &root,
            CapabilityClaim::root(atom, CapabilityMode::InvokeAndDelegate, None),
            delegate.verifying_key(),
        )
        .unwrap();
        let parent = parent_bundle
            .verify(
                root.verifying_key(),
                Epoch::from_tai_seconds(0.0),
                delegate.verifying_key(),
                CapabilityRequest::new(atom, CapabilityMode::InvokeAndDelegate),
            )
            .unwrap();
        let leaf_bundle = parent
            .delegate(
                &delegate,
                CapabilityClaim::delegated(
                    parent.claim_handle(),
                    atom,
                    CapabilityMode::Invoke,
                    None,
                ),
                peer.verifying_key(),
            )
            .unwrap();

        (root, peer, leaf_bundle)
    }

    fn one_step_bundle(
        root: &SigningKey,
        peer: &SigningKey,
        atom: CapabilityAtom,
        validity: Option<CapabilityValidity>,
    ) -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            root,
            CapabilityClaim::root(atom, CapabilityMode::Invoke, validity),
            peer.verifying_key(),
        )
        .unwrap()
    }

    fn successful_response(bundle: &CapabilityProofBundle) -> Vec<u8> {
        let proof = bundle.to_bytes().unwrap();
        let mut bytes = Vec::with_capacity(1 + 4 + proof.len());
        bytes.push(AUTH_OK);
        bytes.extend_from_slice(&(proof.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&proof);
        bytes
    }

    async fn verify_connect_response(
        bytes: &[u8],
        team: VerifyingKey,
        peer: VerifyingKey,
        now: Epoch,
    ) -> Result<VerifiedCapability> {
        let mut input = bytes;
        recv_proof_response(
            &mut input,
            team,
            peer,
            CapabilityRequest::new(connect_capability_atom(team), CapabilityMode::Invoke),
            now,
            "CONNECT",
        )
        .await
    }

    #[test]
    fn capability_proof_bundle_roundtrips_a_verified_delegation() {
        let (root, peer, bundle) = connect_proof_bundle();
        let bytes = bundle.to_bytes().unwrap();
        let decoded = CapabilityProofBundle::from_bytes(&bytes).unwrap();
        let atom = CapabilityAtom::new(
            ACTION_CONNECT.into(),
            CapabilityResource::new(root.verifying_key().to_bytes()),
        );

        assert_eq!(decoded, bundle);
        let verified = decoded
            .verify(
                root.verifying_key(),
                Epoch::from_tai_seconds(0.0),
                peer.verifying_key(),
                CapabilityRequest::new(atom, CapabilityMode::Invoke),
            )
            .unwrap();
        assert_eq!(verified.subject(), peer.verifying_key());
        assert_eq!(verified.effective_atom(), atom);
        assert!(verified.effective_mode().satisfies(CapabilityMode::Invoke));
    }

    #[tokio::test]
    async fn capability_proof_bundle_rejects_oversized_frame_before_reading_a_body() {
        let bytes = ((MAX_CAPABILITY_PROOF_BUNDLE_BYTES + 1) as u32).to_be_bytes();
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_bundle(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn capability_proof_bundle_rejects_an_empty_inner_bundle() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2_u32.to_be_bytes());
        bytes.extend_from_slice(&[CAPABILITY_PROOF_BUNDLE_VERSION, 0]);
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_bundle(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn capability_proof_bundle_rejects_trailing_inner_bytes() {
        let (_, _, bundle) = connect_proof_bundle();
        let mut body = bundle.to_bytes().unwrap();
        body.push(0);
        let mut bytes = Vec::with_capacity(4 + body.len());
        bytes.extend_from_slice(&(body.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&body);
        let mut input = bytes.as_slice();
        assert!(recv_capability_proof_bundle(&mut input).await.is_err());
    }

    #[tokio::test]
    async fn successful_auth_response_requires_the_exact_server_subject_team_and_action() {
        let team = SigningKey::from_bytes(&[0xB1; 32]);
        let server = SigningKey::from_bytes(&[0xB2; 32]);
        let other_server = SigningKey::from_bytes(&[0xB3; 32]);
        let other_team = SigningKey::from_bytes(&[0xB4; 32]);
        let now = Epoch::from_tai_seconds(10.0);
        let atom = connect_capability_atom(team.verifying_key());

        let correct = one_step_bundle(&team, &server, atom, None);
        assert!(
            verify_connect_response(
                &successful_response(&correct),
                team.verifying_key(),
                server.verifying_key(),
                now,
            )
            .await
            .is_ok()
        );

        let wrong_subject = one_step_bundle(&team, &other_server, atom, None);
        assert!(
            verify_connect_response(
                &successful_response(&wrong_subject),
                team.verifying_key(),
                server.verifying_key(),
                now,
            )
            .await
            .is_err()
        );

        let wrong_team = one_step_bundle(&other_team, &server, atom, None);
        assert!(
            verify_connect_response(
                &successful_response(&wrong_team),
                team.verifying_key(),
                server.verifying_key(),
                now,
            )
            .await
            .is_err()
        );

        let wrong_action = one_step_bundle(
            &team,
            &server,
            crate::inventory::sync_team_capability_atom(team.verifying_key()),
            None,
        );
        assert!(
            verify_connect_response(
                &successful_response(&wrong_action),
                team.verifying_key(),
                server.verifying_key(),
                now,
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn successful_auth_response_rejects_an_expired_server_proof() {
        let team = SigningKey::from_bytes(&[0xC1; 32]);
        let server = SigningKey::from_bytes(&[0xC2; 32]);
        let validity =
            CapabilityValidity::new(Epoch::from_tai_seconds(1.0), Epoch::from_tai_seconds(2.0))
                .unwrap();
        let proof = one_step_bundle(
            &team,
            &server,
            connect_capability_atom(team.verifying_key()),
            Some(validity),
        );
        assert!(
            verify_connect_response(
                &successful_response(&proof),
                team.verifying_key(),
                server.verifying_key(),
                Epoch::from_tai_seconds(3.0),
            )
            .await
            .is_err()
        );
    }

    #[tokio::test]
    async fn auth_response_rejects_malformed_and_oversized_server_bundles() {
        let team = SigningKey::from_bytes(&[0xD1; 32]);
        let server = SigningKey::from_bytes(&[0xD2; 32]);
        let now = Epoch::from_tai_seconds(0.0);

        let mut malformed = vec![AUTH_OK];
        malformed.extend_from_slice(&2_u32.to_be_bytes());
        malformed.extend_from_slice(&[CAPABILITY_PROOF_BUNDLE_VERSION, 0]);
        assert!(
            verify_connect_response(
                &malformed,
                team.verifying_key(),
                server.verifying_key(),
                now,
            )
            .await
            .is_err()
        );

        let mut oversized = vec![AUTH_OK];
        oversized.extend_from_slice(
            &u32::try_from(MAX_CAPABILITY_PROOF_BUNDLE_BYTES + 1)
                .unwrap()
                .to_be_bytes(),
        );
        assert!(
            verify_connect_response(
                &oversized,
                team.verifying_key(),
                server.verifying_key(),
                now,
            )
            .await
            .is_err()
        );
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
    async fn find_node_response_enforces_count_and_exact_eof() {
        let oversized = [u8::try_from(crate::routing::K + 1).unwrap()];
        assert!(
            recv_find_node_response(&mut oversized.as_slice())
                .await
                .is_err()
        );

        let trailing = [0, 1];
        assert!(
            recv_find_node_response(&mut trailing.as_slice())
                .await
                .is_err()
        );

        let mut exact = vec![u8::try_from(crate::routing::K).unwrap()];
        for byte in 0..crate::routing::K as u8 {
            exact.extend_from_slice(&[byte; 32]);
        }
        let peers = recv_find_node_response(&mut exact.as_slice())
            .await
            .unwrap();
        assert_eq!(peers.len(), crate::routing::K);
    }
}
