//! Network thread: spawns iroh endpoint, gossip, DHT, protocol server.
//!
//! Private implementation detail of [`crate::peer::Peer`] — `spawn()`
//! returns the [`NetSender`] / [`NetReceiver`] pair the Peer uses to
//! communicate with the async world (commands + snapshot updates one
//! way, events the other).
//!
//! Async is jailed inside the spawned thread.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use ed25519_dalek::SigningKey;
use futures::stream::{FuturesUnordered, StreamExt};
use iroh_base::{EndpointAddr, EndpointId};
use tracing::{Instrument, debug, debug_span, error, info, info_span, instrument, trace, warn};
use triblespace_core::collection::{
    CollectionGossip, CollectionGossipStore, CollectionId, CollectionRecord, CollectionStore,
};
use triblespace_core::repo::{WANT_REQUEST_BYTES_LEN, WantRequest};

use crate::channel::{NetCommand, NetEvent, PublisherKey};
use crate::collection_wire::{
    COLLECTION_COMMIT_EVIDENCE_LEN, CollectionCommitEvidence, CollectionOperationReceiptResponse,
    collection_operation_receipts, decode_collection_operation_request,
    encode_collection_operation_receipts, grant_backed_commits, op_collection_operation_receipts,
};
use crate::identity::iroh_secret;
use crate::protocol::*;
use crate::transport::{Conn, GossipEvent, GossipSink, Harness, PeerId, Transport};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Team-topic domain tag for immutable collection commit evidence.
const GOSSIP_COLLECTION_EVIDENCE: u8 = 0x03;
/// tag(1) + grant-backed evidence(320) + anti-dedupe nonce(8).
const COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN: usize = 1 + COLLECTION_COMMIT_EVIDENCE_LEN + 8;

/// Encode one immutable collection-evidence gossip frame.
///
/// The nonce is intentionally outside the signed evidence. It only gives
/// periodic republishes distinct mesh message ids, so a late joiner can learn
/// unchanged evidence without changing its canonical 320-byte identity.
fn collection_evidence_gossip_frame(evidence: CollectionCommitEvidence, nonce: u64) -> Vec<u8> {
    let mut frame = Vec::with_capacity(COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN);
    frame.push(GOSSIP_COLLECTION_EVIDENCE);
    frame.extend_from_slice(&evidence.encode());
    frame.extend_from_slice(&nonce.to_be_bytes());
    debug_assert_eq!(frame.len(), COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN);
    frame
}

/// Decode and strictly verify one complete collection-evidence gossip frame.
///
/// The transport carrier and anti-dedupe nonce are deliberately discarded:
/// neither participates in author identity or local admission policy.
fn decode_collection_evidence_gossip_frame(bytes: &[u8]) -> Option<CollectionCommitEvidence> {
    if bytes.len() != COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN
        || bytes.first().copied() != Some(GOSSIP_COLLECTION_EVIDENCE)
    {
        return None;
    }
    CollectionCommitEvidence::decode(&bytes[1..1 + COLLECTION_COMMIT_EVIDENCE_LEN]).ok()
}

fn op_name(op: u8) -> &'static str {
    match op {
        OP_AUTH => "AUTH",
        OP_GET_BLOB => "GET_BLOB",
        OP_CHILDREN => "CHILDREN",
        OP_COLLECTION_EVIDENCE => "COLLECTION_EVIDENCE",
        OP_COLLECTION_OPERATION_RECEIPTS => "COLLECTION_OPERATION_RECEIPTS",
        _ => "UNKNOWN",
    }
}

/// Builds a [`RelayMap`] mirroring iroh's prod default but with
/// trailing dots stripped from each relay's hostname.
///
/// Iroh's `iroh::defaults::prod` ships FQDN-absolute hostnames
/// (e.g. `"euc1-1.relay.n0.iroh-canary.iroh.link."` — note the
/// trailing dot, which is the DNS-absolute marker). When iroh
/// constructs HTTPS probe URLs via `Url::parse(...)`, the dot
/// rides through into reqwest's `Host` header. WAFs that treat
/// trailing-dot Host as a known bypass-attempt signature
/// (Anthropic's web-sandbox egress proxy is one) reject those
/// requests with synthetic 503s, which permanently jams iroh's
/// `net_report` cycle and prevents any relay session — and,
/// in iroh's current connect-path design, prevents direct-dial
/// attempts that would otherwise honor a ticket's pre-known
/// addresses.
///
/// Stripping the trailing dot before iroh constructs its
/// `RelayUrl`s produces an HTTP-canonical Host header that the
/// WAFs pass through unmolested. Resolves to the same upstream
/// relay (DNS resolution doesn't care about the absolute/relative
/// distinction); just a different on-the-wire request shape.
///
/// We transform the upstream default rather than hardcoding
/// hostnames, so we stay in sync with whatever n0 ships in
/// `iroh::defaults::prod::default_relay_map()`.
pub(crate) fn dot_stripped_default_relay_map() -> iroh::RelayMap {
    let original = iroh::defaults::prod::default_relay_map();
    let stripped_urls: Vec<String> = original
        .urls::<Vec<_>>()
        .into_iter()
        .map(|relay_url| {
            let mut url: url::Url = relay_url.into();
            if let Some(host) = url.host_str() {
                if let Some(trimmed) = host.strip_suffix('.') {
                    // `set_host` re-validates; on failure (which
                    // shouldn't happen for a valid relay URL with
                    // a trimmable host) we keep the original.
                    let trimmed = trimmed.to_string();
                    let _ = url.set_host(Some(&trimmed));
                }
            }
            url.to_string()
        })
        .collect();
    iroh::RelayMap::try_from_iter(stripped_urls.iter().map(|s| s.as_str()))
        .expect("stripped relay URLs are valid (transformed from valid input)")
}

/// Configuration for [`Peer::new`](crate::peer::Peer::new). No
/// `Default` impl — auth is mandatory in protocol v4 so every peer
/// construction site must explicitly choose a team root. For solo
/// workflows the convention is `team_root = signing_key.verifying_key()`
/// (the user is the team root and the founder of a team-of-one);
/// see the `Peer` struct's doctest for the full pattern.
pub struct PeerConfig {
    /// Bootstrap peers — for both the gossip mesh and the DHT.
    ///
    /// An address-less entry delegates route selection to iroh's standard
    /// discovery.  When direct addresses are present, outbound protocol dials
    /// use the full address verbatim.  This is how a deployment selects a
    /// dedicated fabric instead of accidentally falling back to a management
    /// interface or relay.
    pub peers: Vec<EndpointAddr>,
    /// Whether to subscribe to live collection-evidence gossip. The topic id
    /// is the team root pubkey's 32 bytes — every team has exactly one gossip
    /// mesh, derived from its identity. `false` = serve-/pull-only (no
    /// subscription, no broadcasts).
    pub gossip: bool,
    /// The team root public key — verifies all incoming capability
    /// chains. Every connection's first stream must present a cap that
    /// chains back to this key. See `triblespace_core::repo::capability`.
    /// When `gossip = true`, also serves as the gossip topic id.
    pub team_root: ed25519_dalek::VerifyingKey,
    /// This node's own capability sig handle. Presented to remote peers
    /// as the first stream on every outgoing connection so they can
    /// authorise us. Required — protocol v4 has mandatory auth on both
    /// directions of a connection.
    pub self_cap: RawHash,
    /// Direction of participation in the team swarm. Controls whether this
    /// node publishes collection evidence (write side) and/or admits incoming
    /// collection evidence (read side). Default is `Bidirectional`. Use
    /// [`SyncDirection::ReadOnly`] for follower/catch-up workflows; use
    /// [`SyncDirection::WriteOnly`] for pure-publisher workflows where the
    /// local node has nothing to learn from the swarm.
    pub direction: SyncDirection,
}

/// Which directions of the team swarm this node participates in.
///
/// The wire protocol is symmetric — every peer runs the same code path
/// — but locally we can choose to suppress one side of the data flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncDirection {
    /// Subscribe to gossip and publish our own collection evidence. Default
    /// behaviour.
    #[default]
    Bidirectional,
    /// Subscribe to gossip, but suppress local collection-evidence publishes.
    /// Useful for follower/leecher workflows where the local node is catching
    /// up to the swarm and has no canonical state to contribute.
    ReadOnly,
    /// Publish local collection evidence to gossip, but ignore incoming
    /// evidence from peers. Useful for pure-publisher workflows (for example,
    /// an importer feeding the swarm) where the local node has nothing to
    /// learn from the swarm.
    WriteOnly,
}

// No `Default` impl: every PeerConfig must specify a team root because
// auth is mandatory in protocol v4. For a single-user OSS deployment
// the convention is `team_root = signing_key.verifying_key()` (the user
// is the team root and the founder of a team-of-one).

/// Snapshot of store state for serving protocol requests.
pub struct StoreSnapshot<R> {
    pub reader: R,
    pub pin_heads: triblespace_core::repo::PinSnapshot,
    collection_records: Vec<CollectionRecord>,
    collection_gossips: Vec<CollectionGossip>,
}

impl StoreSnapshot<()> {
    pub fn from_store<S>(store: &mut S) -> Option<StoreSnapshot<S::Reader>>
    where
        S: triblespace_core::repo::BlobStore
            + triblespace_core::repo::PinSnapshotSource
            + CollectionStore
            + CollectionGossipStore,
    {
        // Collection evidence is an additive serving capability. Failure to
        // enumerate it must never suppress the pre-existing blob/branch
        // snapshot (notably, legacy model piles remain directly readable).
        // Invalid structural evidence is inert and simply absent from this
        // serving view.
        let collection_records = store
            .records()
            .map(|records| records.filter_map(Result::ok).collect())
            .unwrap_or_default();
        let collection_gossips = store
            .gossips()
            .map(|gossips| gossips.filter_map(Result::ok).collect())
            .unwrap_or_default();
        let pin_heads = store.snapshot_pin_heads().ok()?;
        let reader = store.reader().ok()?;
        Some(StoreSnapshot {
            reader,
            pin_heads,
            collection_records,
            collection_gossips,
        })
    }
}

/// Type-erased snapshot for the host thread.
///
/// Carries just enough of the pile for the network thread to serve
/// peer requests: per-hash blob fetch, branch head listing, and a
/// quick presence check.
pub trait AnySnapshot: Send + 'static {
    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>>;
    fn has_blob(&self, hash: &RawHash) -> bool;
    fn pin_heads(&self) -> &triblespace_core::repo::PinSnapshot;
    /// Strict grant-backed commits for one exact descriptor handle, in
    /// deterministic intrinsic-record order.
    fn collection_evidence(&self, collection: CollectionId) -> Vec<CollectionCommitEvidence>;
    /// Every strict grant-backed commit in deterministic commit-id order.
    /// Used only to periodically republish the current store truth for late
    /// gossip joiners; the host does not maintain a second ledger mirror.
    fn all_collection_evidence(&self) -> Vec<CollectionCommitEvidence>;
    /// Exact unsigned merge/derive receipts answering one durable question,
    /// in deterministic intrinsic-record order. Conflicting answers remain
    /// distinct records.
    fn collection_operation_receipts(&self, request: WantRequest) -> Vec<CollectionRecord>;
}

impl<R> AnySnapshot for StoreSnapshot<R>
where
    R: triblespace_core::repo::BlobStoreGet + Send + 'static,
{
    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>> {
        use triblespace_core::blob::encodings::UnknownBlob;
        use triblespace_core::inline::Inline;
        use triblespace_core::inline::encodings::hash::Handle;
        let handle = Inline::<Handle<UnknownBlob>>::new(*hash);
        self.reader
            .get::<anybytes::Bytes, UnknownBlob>(handle)
            .ok()
            .map(|b| b.to_vec())
    }

    fn has_blob(&self, hash: &RawHash) -> bool {
        self.get_blob(hash).is_some()
    }

    fn pin_heads(&self) -> &triblespace_core::repo::PinSnapshot {
        &self.pin_heads
    }

    fn collection_evidence(&self, collection: CollectionId) -> Vec<CollectionCommitEvidence> {
        grant_backed_commits(
            &self.collection_records,
            &self.collection_gossips,
            collection,
        )
    }

    fn all_collection_evidence(&self) -> Vec<CollectionCommitEvidence> {
        let grants: std::collections::BTreeMap<([u8; 32], [u8; 32]), CollectionGossip> = self
            .collection_gossips
            .iter()
            .copied()
            .filter(|grant| grant.verify_strict().is_ok())
            .map(|grant| ((grant.collection().raw, grant.public_key().raw), grant))
            .collect();

        self.collection_records
            .iter()
            .filter_map(|record| match record {
                CollectionRecord::Commit(commit) => grants
                    .get(&(commit.collection().raw, commit.public_key().raw))
                    .and_then(|grant| CollectionCommitEvidence::new(*grant, *commit).ok()),
                CollectionRecord::Merge(_) | CollectionRecord::Derive(_) => None,
            })
            .map(|evidence| (evidence.commit().id(), evidence))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect()
    }

    fn collection_operation_receipts(&self, request: WantRequest) -> Vec<CollectionRecord> {
        collection_operation_receipts(request, self.collection_records.iter().copied())
            .unwrap_or_default()
    }
}

/// The network capability a `Peer` invokes **inline** for
/// request/response work — currently the lazy read-miss swarm fetch.
///
/// This is what replaces the old `FetchBlob` command round-trip: rather
/// than ship a command to the host loop and await a reply channel, the
/// Peer method awaits this directly and the fetch runs in its own task.
/// Type-erased over the transport so `Peer` stays transport-agnostic;
/// published through a readiness slot ([`NetSender::fetch_blob`]) once
/// the transport binds, which is how the inline path handles the
/// construction-ordering the command channel used to paper over.
pub(crate) trait NetCapability: Send + Sync {
    /// Swarm-addressed fetch of `hash` (DHT-routed, content-verified).
    /// `None` is Unavailable.
    fn fetch_blob(&self, hash: RawHash) -> futures::future::BoxFuture<'static, Option<Vec<u8>>>;
    /// Start one independent probe per configured peer. The caller consumes
    /// completed probes until its own end-to-end deadline, preserving healthy
    /// answers even when another peer stalls.
    fn collection_operation_receipt_probes(
        &self,
        request: WantRequest,
    ) -> FuturesUnordered<futures::future::BoxFuture<'static, CollectionOperationPeerProbe>>;
}

pub enum CollectionOperationPeerProbe {
    /// The peer answered the exact request, possibly with no records.
    Complete(Vec<CollectionRecord>),
    /// Dial, authentication, framing, or the per-peer deadline failed.
    Incomplete,
}

/// Outcome of one bounded sweep over every configured peer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CollectionOperationProbe {
    /// Deterministic union of every exact receipt returned before the budget.
    pub receipts: Vec<CollectionRecord>,
    /// `true` only when every configured peer completed its probe. Partial
    /// receipts remain useful evidence, but must not quiesce a durable want.
    pub complete: bool,
}

/// Dialable team peers, most-recent live gossip neighbor first. Explicitly
/// configured peers seed the list; neighbor events keep it current. These are
/// routing candidates, not claims that a peer holds any particular blob.
/// `Vec` preserves deterministic simulation replay order.
type RoutingCandidates = Arc<Mutex<Vec<PeerId>>>;

/// Cap on the remembered routing list. Team meshes are small; eight recent
/// peers bounds the worst-case dial fan-out of a DHT-independent miss (each
/// attempt is also bounded by the caller's overall fetch budget).
const ROUTING_CANDIDATE_CAP: usize = 8;

/// Move `peer` to the front of the routing-candidate list (dedup + cap).
fn note_routing_candidate(candidates: &RoutingCandidates, peer: PeerId) {
    let mut list = candidates.lock().unwrap();
    if let Some(pos) = list.iter().position(|p| *p == peer) {
        list.remove(pos);
    }
    list.insert(0, peer);
    list.truncate(ROUTING_CANDIDATE_CAP);
}

/// Remove a peer that the gossip mesh reports as no longer adjacent. A
/// configured peer may re-enter on a later `NeighborUp`; DHT discovery remains
/// available independently.
fn remove_routing_candidate(candidates: &RoutingCandidates, peer: PeerId) {
    candidates
        .lock()
        .unwrap()
        .retain(|candidate| *candidate != peer);
}

fn remove_transient_routing_candidate(
    candidates: &RoutingCandidates,
    configured: &HashSet<PeerId>,
    peer: PeerId,
) {
    if !configured.contains(&peer) {
        remove_routing_candidate(candidates, peer);
    }
}

fn collection_evidence_for_rebroadcast(
    snapshot: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    publishes: bool,
) -> Vec<CollectionCommitEvidence> {
    if !publishes {
        return Vec::new();
    }
    snapshot
        .lock()
        .unwrap()
        .as_ref()
        .map(|snapshot| snapshot.all_collection_evidence())
        .unwrap_or_default()
}

/// Transport-bound implementation of [`NetCapability`]. Holds exactly
/// what the fetch needs; built in the host once the transport exists.
struct NetCap<T: Transport> {
    transport: T,
    pool: SharedPool<T::Conn>,
    self_cap: RawHash,
    my_id: PeerId,
    /// Configured peers and live gossip neighbors — consulted before the DHT
    /// on every on-demand fetch. Membership is only a routing hint; callers
    /// still verify content hashes and fall back to DHT providers.
    candidates: RoutingCandidates,
    /// Explicit peers are the discovery boundary for input-only operation
    /// questions: their unknown result hashes cannot be looked up in the DHT.
    configured_peers: Vec<PeerId>,
}

impl<T: Transport> NetCapability for NetCap<T> {
    fn fetch_blob(&self, hash: RawHash) -> futures::future::BoxFuture<'static, Option<Vec<u8>>> {
        let t = self.transport.clone();
        let pool = self.pool.clone();
        let self_cap = self.self_cap;
        let my_id = self.my_id;
        // Snapshot the routing list now (sync lock, most-recent-first, self
        // excluded) so the future is self-contained.
        let known: Vec<PeerId> = self
            .candidates
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|p| *p != my_id)
            .collect();
        Box::pin(async move {
            // Route-first: try known dialable team peers before the DHT. They
            // are not presumed holders; an ordinary miss simply falls through.
            let mut data = if known.is_empty() {
                None
            } else {
                fetch_from_providers(&t, &hash, &pool, &known, &self_cap).await
            };
            // DHT fallback: no publisher known, or none of them held it.
            if data.is_none() {
                data = fetch_one(&t, &hash, &pool, my_id, &self_cap).await;
            }
            data.filter(|data| blake3::hash(data).as_bytes() == &hash)
        })
    }

    fn collection_operation_receipt_probes(
        &self,
        request: WantRequest,
    ) -> FuturesUnordered<futures::future::BoxFuture<'static, CollectionOperationPeerProbe>> {
        let transport = self.transport.clone();
        let pool = self.pool.clone();
        let self_cap = self.self_cap;
        let peers = self.configured_peers.clone();
        peers
            .into_iter()
            .map(|peer| {
                let transport = transport.clone();
                let pool = pool.clone();
                Box::pin(async move {
                    let Some(connection) = pool_get(&transport, &pool, peer, &self_cap).await
                    else {
                        return CollectionOperationPeerProbe::Incomplete;
                    };
                    match tokio::time::timeout(
                        OP_DEADLINE,
                        op_collection_operation_receipts(&connection, request),
                    )
                    .await
                    {
                        Ok(Ok(CollectionOperationReceiptResponse::Receipts(receipts))) => {
                            CollectionOperationPeerProbe::Complete(receipts)
                        }
                        // Authorization refusal is not a broken pooled
                        // connection, but it also cannot establish that this
                        // peer has no receipt. The sweep remains incomplete.
                        Ok(Ok(CollectionOperationReceiptResponse::Rejected)) => {
                            CollectionOperationPeerProbe::Incomplete
                        }
                        Ok(Err(error)) => {
                            debug!(
                                peer = %hex::encode(&peer[..4]),
                                %error,
                                "collection receipt probe failed"
                            );
                            pool_evict(&pool, peer).await;
                            CollectionOperationPeerProbe::Incomplete
                        }
                        Err(_) => {
                            debug!(
                                peer = %hex::encode(&peer[..4]),
                                "collection receipt probe deadline exceeded"
                            );
                            pool_evict(&pool, peer).await;
                            CollectionOperationPeerProbe::Incomplete
                        }
                    }
                })
                    as futures::future::BoxFuture<'static, CollectionOperationPeerProbe>
            })
            .collect()
    }
}

// ── Outgoing half ────────────────────────────────────────────────────

/// Default overall budget for an **interactive** on-demand blob fetch
/// (a lazy read a caller is actively waiting on). Bounds the WHOLE
/// resolution — capability readiness, DHT lookup, every per-provider
/// dial + op — where the per-stage deadlines alone (`DIAL_DEADLINE`,
/// `OP_DEADLINE`) could stack up to 40s+ across a provider list.
/// Background work (the want-reconciler) passes its own, more generous
/// budget; the want stays durably recorded either way, so an expired
/// budget only defers the fetch, never loses the demand.
pub const INTERACTIVE_FETCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Send fire-and-forget commands to the host loop, refresh the serving
/// snapshot, and invoke inline request/response capabilities (the swarm
/// fetch). `update_snapshot` is a pure snapshot refresh; `fetch_blob`
/// awaits the inline capability rather than the command loop.
#[derive(Clone)]
pub struct NetSender {
    cmd_tx: mpsc::Sender<NetCommand>,
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    /// Readiness slot for the inline fetch capability, published by the
    /// host once its transport binds. `None` until then.
    cap: tokio::sync::watch::Receiver<Option<Arc<dyn NetCapability>>>,
    id: EndpointId,
}

impl NetSender {
    pub fn id(&self) -> EndpointId {
        self.id
    }

    pub fn announce(&self, hash: RawHash) {
        let _ = self.cmd_tx.send(NetCommand::Announce(hash));
    }

    /// Publish one exact, already-verified collection evidence pair.
    ///
    /// This announces immutable evidence only. Receiving it says nothing
    /// about whether the local view authorizes the signed author.
    pub fn gossip_collection_evidence(&self, evidence: CollectionCommitEvidence) {
        let _ = self
            .cmd_tx
            .send(NetCommand::GossipCollectionEvidence { evidence });
    }

    /// Dispatch a freshly-signed (cap, sig) blob pair to `subject`.
    /// Fire-and-forget — the network thread handles the dial,
    /// `OP_DELIVER_CAP`, and connection teardown. Used by the
    /// renewal daemon and `team approve`.
    pub fn deliver_cap(
        &self,
        subject: PublisherKey,
        cap_bytes: anybytes::Bytes,
        sig_bytes: anybytes::Bytes,
    ) {
        let _ = self.cmd_tx.send(NetCommand::DeliverCap {
            subject,
            cap_bytes,
            sig_bytes,
        });
    }

    pub fn update_snapshot(&self, snapshot: impl AnySnapshot) {
        let boxed: Box<dyn AnySnapshot> = Box::new(snapshot);
        *self.snapshot.lock().unwrap() = Some(boxed);
    }

    /// Remove the serving view immediately. Every authenticated data operation
    /// treats an absent snapshot as unavailable/out of scope, so this is the
    /// fail-closed transition when a current store view cannot be produced.
    fn clear_snapshot(&self) {
        *self.snapshot.lock().unwrap() = None;
    }

    /// Replace the host's serving view with the store's current snapshot.
    ///
    /// Snapshot construction is replacement semantics, not best-effort cache
    /// refresh: after a prior success, retaining that old view on failure could
    /// keep a revoked pin head authorized indefinitely. Failure therefore
    /// clears the slot, and a later successful refresh restores service.
    pub(crate) fn refresh_store_snapshot<S>(&self, store: &mut S) -> bool
    where
        S: triblespace_core::repo::BlobStore
            + triblespace_core::repo::PinSnapshotSource
            + CollectionStore
            + CollectionGossipStore,
    {
        match StoreSnapshot::from_store(store) {
            Some(snapshot) => {
                self.update_snapshot(snapshot);
                true
            }
            None => {
                warn!("store snapshot unavailable; clearing serving view");
                self.clear_snapshot();
                false
            }
        }
    }

    /// Swarm-addressed on-demand blob fetch (lazy read-miss) — run
    /// **inline**, not via the command loop. Awaits the network
    /// capability becoming ready (published once the host's transport
    /// binds), then runs the fetch in this task. `None` is Unavailable:
    /// no provider served it, the host never came up, or `budget`
    /// expired.
    ///
    /// `budget` is the END-TO-END deadline over the whole resolution
    /// (capability readiness + DHT lookup + every provider attempt).
    /// Interactive callers pass [`INTERACTIVE_FETCH_DEADLINE`];
    /// background reconcile ticks pass a longer one. Expiry has the
    /// same semantics as any other Unavailable — a recorded want stays
    /// recorded.
    pub async fn fetch_blob(&self, hash: RawHash, budget: std::time::Duration) -> Option<Vec<u8>> {
        match tokio::time::timeout(budget, self.fetch_blob_unbounded(hash)).await {
            Ok(result) => result,
            Err(_) => {
                debug!(
                    hash = %hex::encode(&hash[..4]),
                    budget = ?budget,
                    "fetch_blob: overall budget exceeded; Unavailable"
                );
                None
            }
        }
    }

    /// Probe configured authenticated peers for exact collection-operation
    /// receipts. Empty means unavailable: no configured peer answered with a
    /// matching receipt. No DHT lookup is possible because the result hash is
    /// precisely what the question asks us to discover.
    pub(crate) async fn fetch_collection_operation_receipts(
        &self,
        request: WantRequest,
        budget: std::time::Duration,
    ) -> CollectionOperationProbe {
        let mut rx = self.cap.clone();
        let deadline = tokio::time::Instant::now() + budget;
        let cap = match tokio::time::timeout_at(deadline, rx.wait_for(|cap| cap.is_some())).await {
            Ok(Ok(guard)) => guard.clone(),
            Ok(Err(_)) => return CollectionOperationProbe::default(),
            Err(_) => {
                debug!(
                    ?request,
                    ?budget,
                    "collection receipt probe budget exceeded before network readiness"
                );
                return CollectionOperationProbe::default();
            }
        };
        let Some(cap) = cap else {
            return CollectionOperationProbe::default();
        };
        let mut probes = cap.collection_operation_receipt_probes(request);
        let mut union = std::collections::BTreeMap::new();
        let mut complete = true;
        loop {
            match tokio::time::timeout_at(deadline, probes.next()).await {
                Ok(Some(CollectionOperationPeerProbe::Complete(receipts))) => {
                    for receipt in receipts {
                        union.entry(receipt.id()).or_insert(receipt);
                    }
                }
                Ok(Some(CollectionOperationPeerProbe::Incomplete)) => complete = false,
                Ok(None) => break,
                Err(_) => {
                    complete = false;
                    debug!(
                        ?request,
                        ?budget,
                        completed = union.len(),
                        pending_peers = probes.len(),
                        "collection receipt probe budget exceeded; preserving completed answers"
                    );
                    break;
                }
            }
        }
        CollectionOperationProbe {
            receipts: union.into_values().collect(),
            complete,
        }
    }

    /// Ask the jailed host runtime to fetch inert sparse collection evidence.
    ///
    /// There is deliberately no public deadline knob. Existing transport
    /// stage deadlines bound the operation; channel failure reports that the
    /// host stopped. This method never mutates a store.
    pub fn fetch_collection_evidence(
        &self,
        peer: PeerId,
        collection: CollectionId,
    ) -> anyhow::Result<Vec<CollectionCommitEvidence>> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.cmd_tx
            .send(NetCommand::FetchCollectionEvidence {
                peer,
                collection,
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("network host stopped before collection fetch"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("network host stopped during collection fetch"))?
    }

    /// The unbounded fetch [`fetch_blob`](Self::fetch_blob) wraps in its
    /// overall budget. Kept private: every public path must carry an
    /// end-to-end deadline (per-stage deadlines alone can stack to 40s+
    /// across a provider list).
    async fn fetch_blob_unbounded(&self, hash: RawHash) -> Option<Vec<u8>> {
        let mut rx = self.cap.clone();
        // Resolve the capability — immediate if already published, else
        // park until the transport binds. `Err` means the host dropped
        // its sender (gone) → Unavailable.
        let cap = match rx.wait_for(|c| c.is_some()).await {
            Ok(guard) => guard.clone(),
            Err(_) => return None,
        };
        match cap {
            Some(cap) => cap.fetch_blob(hash).await,
            None => None,
        }
    }
}

// ── Incoming half ────────────────────────────────────────────────────

/// Receive events from the network thread.
pub struct NetReceiver {
    evt_rx: mpsc::Receiver<NetEvent>,
}

impl NetReceiver {
    pub fn try_recv(&self) -> Option<NetEvent> {
        self.evt_rx.try_recv().ok()
    }
}

// ── Spawn ────────────────────────────────────────────────────────────

/// The host loop's end of the Peer↔host channel pair, plus the shared
/// serving-snapshot slot. Produced by [`wire`]; consumed by
/// [`run_host`]. Exists so the loop can run either on its own thread
/// + runtime (production, [`spawn`]) or as a task on a caller-owned
/// runtime (deterministic simulation, where every node shares one
/// paused current-thread runtime).
pub struct HostWiring {
    pub(crate) cmd_rx: mpsc::Receiver<NetCommand>,
    pub(crate) evt_tx: mpsc::Sender<NetEvent>,
    pub(crate) snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    /// Publish half of the inline-fetch capability slot; the host fills
    /// it once its transport binds.
    pub(crate) cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
}

/// Build the Peer↔host channel pair for a node with identity `id`.
/// The `(NetSender, NetReceiver)` half goes to the Peer; the
/// [`HostWiring`] half goes to [`run_host`].
pub fn wire(id: EndpointId) -> (NetSender, NetReceiver, HostWiring) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<NetCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<NetEvent>();
    let snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> = Arc::new(Mutex::new(None));
    let (cap_tx, cap_rx) = tokio::sync::watch::channel::<Option<Arc<dyn NetCapability>>>(None);

    let sender = NetSender {
        cmd_tx,
        snapshot: snapshot.clone(),
        cap: cap_rx,
        id,
    };
    let receiver = NetReceiver { evt_rx };
    let wiring = HostWiring {
        cmd_rx,
        evt_tx,
        snapshot,
        cap_tx,
    };
    (sender, receiver, wiring)
}

/// Run the host loop over an already-constructed transport harness.
/// This is the transport-generic entry point: production wraps it in
/// a dedicated thread ([`spawn`]); the simulator spawns it as a local
/// task per node on one shared deterministic runtime.
pub async fn run_host<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    host_loop(
        harness,
        config,
        wiring.cmd_rx,
        wiring.evt_tx,
        wiring.snapshot,
        wiring.cap_tx,
    )
    .await;
}

/// Spawn the network thread. Returns the outgoing/incoming channel halves
/// — used internally by [`Peer::new`](crate::peer::Peer::new).
pub fn spawn(key: SigningKey, config: PeerConfig) -> (NetSender, NetReceiver) {
    let secret = iroh_secret(&key);
    let id: EndpointId = secret.public().into();

    let (sender, receiver, wiring) = wire(id);

    let _thread = thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async move {
            let Some(harness) = crate::transport::iroh::bind(secret, &config).await else {
                // bind already logged the failure; net thread exits.
                return;
            };
            run_host(harness, config, wiring).await;
        });
    });

    (sender, receiver)
}

// ── Network thread event loop ────────────────────────────────────────

/// Deadline for establishing + authenticating a connection (the
/// `pool_get` init future: dial + OP_AUTH round trip). A connection
/// attempt that exceeds this counts as failed: the pool's
/// singleflight cell resets so the next walk re-dials, instead of
/// every later fetch to that peer queueing forever behind one
/// stalled handshake. Generous relative to real-world QUIC + relay
/// setup times; deterministic under simulated virtual time.
const DIAL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// Deadline for a single protocol op (OP_CHILDREN / OP_GET_BLOB
/// request + full response) on an established connection. On expiry
/// the op reports an error and the caller's existing
/// evict-and-try-next-provider path takes over. Total-op rather than
/// progress-based: at the 1 MiB max blob size even slow links finish
/// well inside this; revisit with idle-deadlines if blob sizes grow.
const OP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Connect to a peer over the pile-sync ALPN and immediately present
/// our capability so subsequent ops are authorised. Protocol v4 makes
/// this mandatory — the server rejects any op until the connection
/// completes auth.
#[instrument(level = "info", skip(t, self_cap), fields(peer = %hex::encode(&peer[..4])))]
async fn connect_authed<T: Transport>(
    t: &T,
    peer: PeerId,
    self_cap: &RawHash,
) -> anyhow::Result<T::Conn> {
    let conn = t.dial(peer, PILE_SYNC_ALPN).await.map_err(|e| {
        warn!(error = %e, "connect failed");
        anyhow::anyhow!("connect: {e}")
    })?;
    debug!(self_cap = %hex::encode(&self_cap[..4]), "connected; sending OP_AUTH");
    op_auth(&conn, self_cap).await.map_err(|e| {
        warn!(error = %e, "auth handshake failed");
        anyhow::anyhow!("auth: {e}")
    })?;
    info!("auth ok");
    Ok(conn)
}

async fn host_loop<T: Transport>(
    harness: Harness<T>,
    config: PeerConfig,
    commands: mpsc::Receiver<NetCommand>,
    events: mpsc::Sender<NetEvent>,
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
) {
    let Harness {
        transport,
        incoming,
        gossip,
    } = harness;

    let my_id: PeerId = transport.local_id();
    let self_cap: RawHash = config.self_cap;
    let publishes_collection_evidence = config.direction != SyncDirection::ReadOnly;
    let mut configured_peers: Vec<PeerId> = config
        .peers
        .iter()
        .map(|address| *address.id.as_bytes())
        .filter(|peer| *peer != my_id)
        .collect();
    configured_peers.sort_unstable();
    configured_peers.dedup();

    // Host-wide singleflight connection pool — one authed connection per
    // remote peer, reused across direct blob and capability-chain fetches. See
    // `SharedPool` docs for the OnceCell-based dial deduplication.
    let conn_pool: SharedPool<T::Conn> = new_shared_pool();

    // Configured peers seed routing even before gossip neighbor discovery.
    // Live NeighborUp events move active routes to the front; collection
    // evidence carriers do not imply possession of referenced blobs.
    let routing_candidates: RoutingCandidates = Arc::new(Mutex::new(configured_peers.clone()));
    let configured_peer_set: Arc<HashSet<PeerId>> =
        Arc::new(configured_peers.iter().copied().collect());

    // Publish the inline-fetch capability now that the transport exists.
    // `Peer::fetch_blob` parks on this slot until it's filled, which is
    // how the inline read path handles the construction-ordering the old
    // `FetchBlob` command channel used to buffer past.
    let _ = cap_tx.send(Some(Arc::new(NetCap {
        transport: transport.clone(),
        pool: conn_pool.clone(),
        self_cap,
        my_id,
        candidates: routing_candidates.clone(),
        configured_peers,
    }) as Arc<dyn NetCapability>));

    // Our own pubkey — the expected `cap_subject` of any cap
    // delivered to us via OP_DELIVER_CAP.
    let our_pubkey = ed25519_dalek::VerifyingKey::from_bytes(&my_id)
        .expect("transport local id is an ed25519 pubkey");

    // ── Inbound connections: dispatch by ALPN to the protocol
    // handlers. Each connection gets its own task; each handler
    // accepts sequential bi-streams until the peer closes.
    let snapshot_handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        team_root: config.team_root,
        transport: transport.clone(),
        self_cap,
        events: events.clone(),
        pool: conn_pool.clone(),
    };
    let handshake_handler = HandshakeHandler {
        events: events.clone(),
        team_root: config.team_root,
        our_pubkey,
        snapshot: snapshot.clone(),
        transport: transport.clone(),
        pool: conn_pool.clone(),
    };
    let mut incoming = incoming;
    tokio::spawn(async move {
        while let Some(inc) = incoming.recv().await {
            if inc.alpn == PILE_SYNC_ALPN {
                let h = snapshot_handler.clone();
                tokio::spawn(async move { h.handle(inc.conn).await });
            } else if inc.alpn == crate::handshake::AUTH_HANDSHAKE_ALPN {
                let h = handshake_handler.clone();
                tokio::spawn(async move { h.handle(inc.conn).await });
            } else {
                debug!(alpn = %String::from_utf8_lossy(inc.alpn), "incoming conn on unknown alpn; dropping");
            }
        }
    });

    // ── Gossip: consume immutable collection evidence and maintain
    // transport-level routing hints from neighbor liveness events.
    let mut gossip_sender: Option<T::Gossip> = None;
    if let Some((sender, mut gossip_events)) = gossip {
        gossip_sender = Some(sender);
        let events_tx = events.clone();
        let candidates_for_gossip = routing_candidates.clone();
        let configured_for_gossip = configured_peer_set.clone();
        tokio::spawn(async move {
            while let Some(event) = gossip_events.recv().await {
                match event {
                    GossipEvent::Received {
                        bytes,
                        delivered_from,
                    } => {
                        if let Some(evidence) = decode_collection_evidence_gossip_frame(&bytes) {
                            // The carrier is only a mesh hop. Do not attach it
                            // to the evidence or use it as author authority;
                            // both author identities are signed inside the
                            // strictly decoded pair.
                            let _ = events_tx.send(NetEvent::CollectionEvidence(evidence));
                        } else if bytes.first().copied() == Some(GOSSIP_COLLECTION_EVIDENCE) {
                            debug!(
                                delivered_from = %hex::encode(&delivered_from[..4]),
                                length = bytes.len(),
                                "discarding malformed collection evidence gossip frame"
                            );
                        }
                    }
                    GossipEvent::NeighborUp(peer) => {
                        note_routing_candidate(&candidates_for_gossip, peer);
                        info!(peer = %hex::encode(&peer[..4]), "gossip neighbor up");
                    }
                    GossipEvent::NeighborDown(peer) => {
                        remove_transient_routing_candidate(
                            &candidates_for_gossip,
                            &configured_for_gossip,
                            peer,
                        );
                        info!(peer = %hex::encode(&peer[..4]), "gossip neighbor down");
                    }
                }
            }
        });
    }

    let rebroadcast_period = std::time::Duration::from_secs(30);
    // Read through crate::clock (not std Instant) so the rebroadcast
    // tick advances under simulated virtual time.
    let mut last_rebroadcast = crate::clock::mono_now();

    // Command loop.
    loop {
        while let Ok(cmd) = commands.try_recv() {
            match cmd {
                NetCommand::Announce(hash) => {
                    let t = transport.clone();
                    tokio::spawn(async move {
                        t.dht_announce(hash).await;
                    });
                }
                NetCommand::GossipCollectionEvidence { evidence } => {
                    if let Some(sender) = &gossip_sender {
                        let msg = collection_evidence_gossip_frame(
                            evidence,
                            crate::clock::mono_now().as_nanos(),
                        );
                        let sender = sender.clone();
                        tokio::spawn(async move {
                            let _ = sender.broadcast(msg).await;
                        });
                    }
                }
                NetCommand::DeliverCap {
                    subject,
                    cap_bytes,
                    sig_bytes,
                } => {
                    // Open a fresh connection on the auth-handshake
                    // ALPN, send OP_DELIVER_CAP, close. On STATUS_OK
                    // ack we emit `NetEvent::CapDeliveryConfirmed`
                    // so the Peer can mark the matching
                    // renewal-policy entry as delivered; on any
                    // failure (connect/send/non-OK) the entry stays
                    // in the undelivered set and the next renewal
                    // tick attempts redispatch.
                    let t_for_deliver = transport.clone();
                    tokio::spawn(async move {
                        let conn = match t_for_deliver
                            .dial(subject, crate::handshake::AUTH_HANDSHAKE_ALPN)
                            .await
                        {
                            Ok(c) => c,
                            Err(e) => {
                                debug!(
                                    subject = %hex::encode(&subject[..4]),
                                    error = %e,
                                    "DeliverCap: connect failed"
                                );
                                return;
                            }
                        };
                        match crate::handshake::send_deliver_cap(&conn, &cap_bytes, &sig_bytes)
                            .await
                        {
                            Ok(status) if status == crate::handshake::STATUS_OK => {
                                debug!(
                                    subject = %hex::encode(&subject[..4]),
                                    "DeliverCap: recipient ack OK (wire-level — absorb \
                                     happens asynchronously on recipient; \
                                     CapDeliveryConfirmed is emitted later from the OP_AUTH \
                                     path when the subject actually authenticates with the cap)"
                                );
                            }
                            Ok(status) => {
                                debug!(
                                    subject = %hex::encode(&subject[..4]),
                                    status,
                                    "DeliverCap: recipient returned non-OK status"
                                );
                            }
                            Err(e) => {
                                debug!(
                                    subject = %hex::encode(&subject[..4]),
                                    error = %e,
                                    "DeliverCap: send failed"
                                );
                            }
                        }
                        conn.close(0, b"ok");
                    });
                }
                NetCommand::FetchCollectionEvidence {
                    peer,
                    collection,
                    reply,
                } => {
                    let transport = transport.clone();
                    let pool = conn_pool.clone();
                    tokio::spawn(async move {
                        let result = async {
                            if peer == my_id {
                                anyhow::bail!("collection reconciliation peer is the local node");
                            }
                            let Some(connection) =
                                pool_get(&transport, &pool, peer, &self_cap).await
                            else {
                                anyhow::bail!(
                                    "could not establish an authenticated collection connection to {}",
                                    hex::encode(peer),
                                );
                            };
                            match crate::collection_wire::op_collection_evidence(
                                &connection,
                                collection,
                            )
                            .await
                            {
                                Ok(evidence) => Ok(evidence),
                                Err(error) => {
                                    pool_evict(&pool, peer).await;
                                    Err(error)
                                }
                            }
                        }
                        .await;
                        let _ = reply.send(result);
                    });
                }
            }
        }

        if crate::clock::mono_now().duration_since(last_rebroadcast) >= rebroadcast_period {
            let collection_evidence =
                collection_evidence_for_rebroadcast(&snapshot, publishes_collection_evidence);
            trace!(
                collection_evidence = collection_evidence.len(),
                "rebroadcast tick: replaying published evidence"
            );
            if let Some(sender) = &gossip_sender {
                for evidence in collection_evidence {
                    let msg = collection_evidence_gossip_frame(
                        evidence,
                        crate::clock::mono_now().as_nanos(),
                    );
                    let sender = sender.clone();
                    tokio::spawn(async move {
                        let _ = sender.broadcast(msg).await;
                    });
                }
            }
            last_rebroadcast = crate::clock::mono_now();
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Resolve providers for a hash. When `preferred_peer` is not self, use it as
/// the exact first candidate (cap-chain transfer knows the requesting peer is
/// the source). Otherwise query the DHT.
///
/// Self is filtered out — `find_providers` will list us as a
/// provider for any blob we've announced, and trying to dial
/// ourselves trips iroh's "Connecting to ourself is not supported"
/// error. Self is never useful for a missing local hash.
async fn providers_for<T: Transport>(t: &T, hash: &RawHash, preferred_peer: PeerId) -> Vec<PeerId> {
    let my_id = t.local_id();
    if preferred_peer != my_id {
        return vec![preferred_peer];
    }
    trace!(hash = %hex::encode(&hash[..4]), "providers_for: DHT find_providers awaiting");
    let mut providers: Vec<PeerId> =
        match tokio::time::timeout(std::time::Duration::from_secs(3), t.dht_providers(*hash)).await
        {
            Ok(p) => p,
            Err(_) => {
                warn!(
                    hash = %hex::encode(&hash[..4]),
                    "dht_providers timed out; no provider candidates"
                );
                Vec::new()
            }
        };
    trace!(hash = %hex::encode(&hash[..4]), n = providers.len(), "providers_for: DHT find_providers returned");
    providers.retain(|id| *id != my_id);
    providers
}

/// Host-wide connection pool: one authed `iroh::endpoint::Connection`
/// per remote peer, shared across all direct blob and capability-chain fetches.
///
/// `OnceCell` per peer provides automatic singleflight: the first
/// task to encounter a missing entry runs the dial; concurrent tasks
/// await the same `OnceCell` and reuse the resulting connection. No
/// dial-storm when several reads target the same peer concurrently.
///
/// iroh QUIC multiplexes streams cheaply on a single connection; our
/// `serve_stream` accepts unbounded sequential bi-streams per
/// connection (auth state set on the first OP_AUTH stream, reused on
/// every subsequent stream). So one connection per peer is enough.
pub(crate) type SharedPool<C> =
    Arc<tokio::sync::Mutex<HashMap<PeerId, Arc<tokio::sync::OnceCell<C>>>>>;

fn new_shared_pool<C>() -> SharedPool<C> {
    Arc::new(tokio::sync::Mutex::new(HashMap::new()))
}

/// Get-or-dial an authed connection to `provider` from the shared
/// pool. `OnceCell::get_or_try_init` runs the dial exactly once even
/// if many tasks race here concurrently; the rest await the same
/// initialization. Returns `None` if the dial fails (the cell stays
/// uninitialized so a later call can retry).
async fn pool_get<T: Transport>(
    t: &T,
    pool: &SharedPool<T::Conn>,
    provider: PeerId,
    self_cap: &RawHash,
) -> Option<T::Conn> {
    let cell = {
        let mut guard = pool.lock().await;
        guard
            .entry(provider)
            .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
            .clone()
    };
    let init = || async {
        match tokio::time::timeout(DIAL_DEADLINE, connect_authed(t, provider, self_cap)).await {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!(
                "connection setup deadline ({DIAL_DEADLINE:?}) exceeded"
            )),
        }
    };
    match cell.get_or_try_init(init).await {
        Ok(conn) => Some(conn.clone()),
        Err(e) => {
            debug!(error = %e, provider = %hex::encode(&provider[..4]), "pool dial failed");
            // Drop the cell so the next caller can retry. Use a fresh
            // entry: if anyone awaited the original cell while we were
            // in get_or_try_init, they all got the same Err — they'll
            // retry through their own entries below.
            let mut guard = pool.lock().await;
            if let Some(existing) = guard.get(&provider) {
                if std::ptr::eq(Arc::as_ptr(existing), Arc::as_ptr(&cell)) {
                    guard.remove(&provider);
                }
            }
            None
        }
    }
}

/// Evict a connection from the pool. Called when an op on the pooled
/// connection errors (peer may have closed, network changed, etc.)
/// so the next access re-dials.
async fn pool_evict<C: Conn>(pool: &SharedPool<C>, provider: PeerId) {
    let removed = {
        let mut guard = pool.lock().await;
        guard.remove(&provider)
    };
    if let Some(cell) = removed {
        if let Some(conn) = cell.get() {
            conn.close(0, b"pool evict");
        }
    }
}

/// Fetch a single blob via the swarm — DHT-resolved providers
/// first, publisher as fallback. Returns the first successful
/// fetch's bytes (caller verifies hash).
async fn fetch_one<T: Transport>(
    t: &T,
    hash: &RawHash,
    pool: &SharedPool<T::Conn>,
    publisher_id: PeerId,
    self_cap: &RawHash,
) -> Option<Vec<u8>> {
    let providers = providers_for(t, hash, publisher_id).await;
    fetch_from_providers(t, hash, pool, &providers, self_cap).await
}

/// Try `providers` in order for a single blob: pooled authed connection,
/// OP_GET_BLOB with the per-op deadline, evict-and-try-next on
/// connection errors. First success wins; the caller verifies the hash.
/// The provider-iteration tail of [`fetch_one`], split out so the
/// route-first on-demand path ([`NetCap::fetch_blob`]) can drive it with known
/// dialable candidates without a DHT round-trip.
async fn fetch_from_providers<T: Transport>(
    t: &T,
    hash: &RawHash,
    pool: &SharedPool<T::Conn>,
    providers: &[PeerId],
    self_cap: &RawHash,
) -> Option<Vec<u8>> {
    for &provider in providers {
        let Some(conn) = pool_get(t, pool, provider, self_cap).await else {
            continue;
        };
        let op = tokio::time::timeout(OP_DEADLINE, op_get_blob(&conn, hash))
            .await
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "OP_GET_BLOB deadline ({OP_DEADLINE:?}) exceeded"
                ))
            });
        match op {
            Ok(Some(data)) => return Some(data),
            Ok(None) => {
                debug!(hash = %hex::encode(&hash[..4]), provider = %hex::encode(&provider[..4]), "blob miss");
                continue;
            }
            Err(e) => {
                debug!(error = %e, hash = %hex::encode(&hash[..4]), provider = %hex::encode(&provider[..4]), "op_get_blob errored, evicting and trying next provider");
                // Connection-level error: pooled connection may be
                // dead. Evict so subsequent ops to this peer re-dial.
                pool_evict(pool, provider).await;
                continue;
            }
        }
    }
    None
}

/// Swarm-fetch the closure rooted at `head` (a cap sig handle, in the
/// OP_AUTH context) and return it as a `BTreeMap<RawHash, Vec<u8>>`
/// (ordered, so draining it into NetEvent::Blob emissions is
/// deterministic for simulation replay).
/// Uses a two-phase walk (OP_CHILDREN discovery, then OP_GET_BLOB in
/// reverse-BFS order) and writes results to a map. The caller decides whether
/// to cache the bytes into the local store after using them.
async fn swarm_fetch_chain<T: Transport>(
    t: &T,
    publisher: PeerId,
    head: &RawHash,
    self_cap: &RawHash,
    pool: &SharedPool<T::Conn>,
) -> std::collections::BTreeMap<RawHash, Vec<u8>> {
    let mut fetched: std::collections::BTreeMap<RawHash, Vec<u8>> =
        std::collections::BTreeMap::new();
    let publisher_id = publisher;

    // Ensure we have an authed connection to the publisher (the
    // peer that just sent us the cap_handle via OP_AUTH). pool_get
    // is singleflight, so concurrent swarm_fetch_chain calls in
    // the parallel-OP_AUTH-burst case share one dial + one OP_AUTH.
    // The whole recursion bottoms out at the publisher for typical
    // two-level chains.
    if pool_get(t, pool, publisher_id, self_cap).await.is_none() {
        // Couldn't even auth to the dialer. Give up — there's no
        // realistic path to fetch the chain without them.
        return fetched;
    }

    // Phase 1: discovery via OP_CHILDREN. BFS order; stop when
    // every frontier blob is either no-children (root cap) or
    // unreachable.
    let mut seen: HashSet<RawHash> = HashSet::new();
    let mut to_fetch: Vec<RawHash> = Vec::new();
    let mut frontier: Vec<RawHash> = vec![*head];
    seen.insert(*head);
    to_fetch.push(*head);

    while !frontier.is_empty() {
        let mut next: Vec<RawHash> = Vec::new();
        for parent in &frontier {
            let children = match children_one(t, parent, pool, publisher_id, self_cap).await {
                Some(c) => c,
                None => continue,
            };
            for hash in children {
                if !seen.insert(hash) {
                    continue;
                }
                to_fetch.push(hash);
                next.push(hash);
            }
        }
        frontier = next;
    }

    // Phase 2: deepest-first fetch. Order matters for the caller's
    // cache-write step: emitting children before parents keeps the
    // bottom-up insertion invariant when the events get drained.
    for hash in to_fetch.iter().rev() {
        let Some(data) = fetch_one(t, hash, pool, publisher_id, self_cap).await else {
            continue;
        };
        if blake3::hash(&data).as_bytes() != hash {
            warn!(hash = %hex::encode(&hash[..4]), "hash mismatch on swarm-fetched cap blob");
            continue;
        }
        fetched.insert(*hash, data);
    }

    fetched
}

/// Walk children of a parent blob via the swarm.
async fn children_one<T: Transport>(
    t: &T,
    parent: &RawHash,
    pool: &SharedPool<T::Conn>,
    publisher_id: PeerId,
    self_cap: &RawHash,
) -> Option<Vec<RawHash>> {
    trace!(parent = %hex::encode(&parent[..4]), "children_one: providers_for awaiting");
    let providers = providers_for(t, parent, publisher_id).await;
    trace!(parent = %hex::encode(&parent[..4]), n = providers.len(), "children_one: providers_for returned");
    for provider in &providers {
        trace!(parent = %hex::encode(&parent[..4]), provider = %hex::encode(&provider[..4]), "children_one: pool_get awaiting");
        let Some(conn) = pool_get(t, pool, *provider, self_cap).await else {
            trace!(parent = %hex::encode(&parent[..4]), provider = %hex::encode(&provider[..4]), "children_one: pool_get returned None");
            continue;
        };
        trace!(parent = %hex::encode(&parent[..4]), provider = %hex::encode(&provider[..4]), "children_one: op_children awaiting");
        let op = tokio::time::timeout(OP_DEADLINE, op_children(&conn, parent))
            .await
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "OP_CHILDREN deadline ({OP_DEADLINE:?}) exceeded"
                ))
            });
        match op {
            Ok(c) => return Some(c),
            Err(e) => {
                debug!(error = %e, parent = %hex::encode(&parent[..4]), provider = %hex::encode(&provider[..4]), "op_children errored, evicting and trying next provider");
                pool_evict(pool, *provider).await;
                continue;
            }
        }
    }
    None
}

// ── Protocol handler ─────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotHandler<T: Transport> {
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    /// Verifies all incoming capability chains. Required — protocol v4
    /// has mandatory auth.
    team_root: ed25519_dalek::VerifyingKey,
    /// Transport for outbound connections + DHT provider lookup
    /// during the swarm-fetch fallback in OP_AUTH (when an incoming
    /// cap chain references blobs we don't have locally).
    transport: T,
    /// Our own cap handle, presented at OP_AUTH when we dial peers
    /// to fetch missing cap chain blobs.
    self_cap: RawHash,
    /// Channel back to the Peer for caching fetched cap blobs. After
    /// a successful swarm-fetch + verify_chain, we emit NetEvent::Blob
    /// for each fetched cap so the Peer puts them in the local store —
    /// next OP_AUTH involving the same chain hits local instead of
    /// re-walking the swarm.
    events: mpsc::Sender<NetEvent>,
    /// Host-wide connection pool. Shared with the gossip-arrival
    /// fetch path. The OP_AUTH swarm-fetch and the gossip-driven
    /// fetch end up using the same authed connection per peer.
    pool: SharedPool<T::Conn>,
}

/// Protocol handler for `/triblespace/auth-handshake/1`. Accepts
/// incoming `OP_REQUEST_CAP` and `OP_DELIVER_CAP` streams and
/// forwards their payloads to the Peer's event channel. All policy
/// (approve / queue / reject; verify / pin / drop) lives in the
/// receiving Peer, not here — this handler just bridges the wire to
/// the local event queue.
#[derive(Clone)]
struct HandshakeHandler<T: Transport> {
    events: mpsc::Sender<NetEvent>,
    /// Team root pubkey — verifies the delivered cap's chain at
    /// `OP_DELIVER_CAP` time so STATUS_OK means "we'd accept this".
    team_root: ed25519_dalek::VerifyingKey,
    /// Our own pubkey — the expected `cap_subject` of any cap
    /// delivered to us.
    our_pubkey: ed25519_dalek::VerifyingKey,
    /// Snapshot for local-pile blob lookup during verify.
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    /// Transport + pool are the swarm-fetch substrate. When the
    /// local-pile verify fails with `Fetch`, we open `pile-sync/4`
    /// to providers of the missing blobs (DHT providers first,
    /// dialer as fallback) and walk the chain via `OP_CHILDREN` +
    /// `OP_GET_BLOB` until we have everything verify needs. The
    /// swarm-fetch credential is the just-delivered sig handle
    /// itself (see the OP_DELIVER_CAP arm), so no self_cap here.
    transport: T,
    pool: SharedPool<T::Conn>,
}

impl<T: Transport> HandshakeHandler<T> {
    async fn handle(&self, connection: T::Conn) {
        // PublisherKey is just the 32-byte pubkey representation;
        // the transport's remote id is the TLS-verified ed25519
        // pubkey of the dialer (matched against the type alias in
        // channel.rs).
        let peer_pubkey_bytes: PublisherKey = connection.remote_id();
        let events = self.events.clone();
        let team_root = self.team_root;
        let our_pubkey = self.our_pubkey;
        let snapshot = self.snapshot.clone();
        let transport = self.transport.clone();
        let pool = self.pool.clone();
        let span = info_span!(
            "auth-handshake",
            peer = %hex::encode(&peer_pubkey_bytes[..4]),
        );
        async move {
            // Each connection can carry multiple bi-streams (e.g. a
            // request followed by a deliver). Loop until the peer
            // closes the connection.
            loop {
                let Some((mut send, mut recv)) = connection.accept_bi().await else {
                    debug!("accept_bi ended; handshake connection closing");
                    break;
                };
                match crate::handshake::read_incoming(&mut recv).await {
                    Ok(Some(crate::handshake::IncomingOp::Request {
                        partial_cap_bytes,
                    })) => {
                        let _ = events.send(NetEvent::CapRequest {
                            requester: peer_pubkey_bytes,
                            partial_cap_bytes,
                        });
                        let _ = crate::handshake::respond(
                            &mut send,
                            crate::handshake::STATUS_OK,
                        )
                        .await;
                    }
                    Ok(Some(crate::handshake::IncomingOp::Deliver {
                        cap_bytes,
                        sig_bytes,
                    })) => {
                        use triblespace_core::blob::{Blob, TryFromBlob};
                        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
                        use triblespace_core::inline::Inline;
                        use triblespace_core::inline::encodings::hash::Handle;
                        use triblespace_core::trible::TribleSet;
                        use triblespace_core::macros::{find, pattern};

                        let cap_blob: Blob<SimpleArchive> = Blob::new(cap_bytes.clone());
                        let sig_blob: Blob<SimpleArchive> = Blob::new(sig_bytes.clone());
                        let cap_hash: RawHash = *blake3::hash(&cap_bytes).as_bytes();
                        let sig_hash: RawHash = *blake3::hash(&sig_bytes).as_bytes();
                        let sig_handle: Inline<Handle<SimpleArchive>> =
                            Inline::new(sig_hash);

                        // Cheap DoS guard before any swarm work: the
                        // cap's declared `cap_issuer` must equal the
                        // TLS-verified pubkey of whoever just dialed
                        // us. The auth-handshake ALPN is open to
                        // unauthenticated peers, so without this gate
                        // a stranger could ship a cap with our subject
                        // + a `cap_parent` pointing at random hashes,
                        // and we'd burn DHT lookups chasing chain
                        // blobs that will never verify. The check
                        // costs one `find!` against the leaf cap
                        // blob.
                        let declared_issuer = if let Ok(cap_set) =
                            TribleSet::try_from_blob(cap_blob.clone())
                        {
                            find!(
                                (issuer: ed25519_dalek::VerifyingKey),
                                pattern!(&cap_set, [{
                                    triblespace_core::repo::capability::cap_issuer: ?issuer,
                                }])
                            )
                            .next()
                            .map(|(k,)| k)
                        } else {
                            None
                        };
                        match declared_issuer {
                            Some(issuer) if issuer.to_bytes() == peer_pubkey_bytes => {}
                            Some(issuer) => {
                                warn!(
                                    declared_issuer = %hex::encode(&issuer.to_bytes()[..4]),
                                    dialer = %hex::encode(&peer_pubkey_bytes[..4]),
                                    "OP_DELIVER_CAP: cap_issuer doesn't match TLS dialer; rejecting",
                                );
                                let _ = crate::handshake::respond(
                                    &mut send,
                                    crate::handshake::STATUS_REJECTED,
                                )
                                .await;
                                continue;
                            }
                            None => {
                                warn!("OP_DELIVER_CAP: cap blob malformed or missing cap_issuer; rejecting");
                                let _ = crate::handshake::respond(
                                    &mut send,
                                    crate::handshake::STATUS_MALFORMED,
                                )
                                .await;
                                continue;
                            }
                        }

                        // Verify-with-swarm-fetch: try local first, then
                        // pull missing chain blobs via the same
                        // DHT-routed pool path OP_AUTH uses. The dialer
                        // is the immediate issuer and almost certainly
                        // has the parent cap, but for 3+ hop chains the
                        // intermediate cap might live elsewhere — DHT
                        // provider lookup finds them either way.
                        let verify_once = |fetched: &std::collections::BTreeMap<RawHash, Vec<u8>>| {
                            let snap_for_fetch = snapshot.clone();
                            let fetched_for_lookup = fetched.clone();
                            let cap_blob_for_fetch = cap_blob.clone();
                            let sig_blob_for_fetch = sig_blob.clone();
                            triblespace_core::repo::capability::verify_chain(
                                team_root,
                                sig_handle,
                                our_pubkey,
                                move |h: Inline<Handle<SimpleArchive>>| -> Option<Blob<SimpleArchive>> {
                                    if h.raw == cap_hash {
                                        return Some(cap_blob_for_fetch.clone());
                                    }
                                    if h.raw == sig_hash {
                                        return Some(sig_blob_for_fetch.clone());
                                    }
                                    if let Some(bytes) = snap_for_fetch
                                        .lock()
                                        .unwrap()
                                        .as_ref()
                                        .and_then(|s| s.get_blob(&h.raw))
                                    {
                                        return Some(Blob::new(anybytes::Bytes::from_source(bytes)));
                                    }
                                    let bytes = fetched_for_lookup.get(&h.raw)?.clone();
                                    Some(Blob::new(anybytes::Bytes::from_source(bytes)))
                                },
                            )
                        };

                        let mut fetched: std::collections::BTreeMap<RawHash, Vec<u8>> =
                            std::collections::BTreeMap::new();
                        let mut result = verify_once(&fetched);

                        if matches!(
                            result,
                            Err(triblespace_core::repo::capability::VerifyError::Fetch),
                        ) {
                            debug!(
                                sig = %hex::encode(&sig_hash[..4]),
                                "OP_DELIVER_CAP: chain incomplete locally, swarm-fetching",
                            );

                            // Use the just-received `sig_hash` as the
                            // OP_AUTH credential for the swarm-fetch
                            // — for both first-time delivery and
                            // renewals. The new cap is by definition
                            // the one we're going to be using going
                            // forward; the prior `self_cap` is at
                            // best redundant and at worst
                            // already-expired. The dialer-equals-
                            // issuer precheck above already
                            // established that the cap was actually
                            // signed by this dialer, so they trivially
                            // accept it on AUTH (they have its
                            // chain), and the remote's own OP_AUTH
                            // path validates against team_root for
                            // anyone deeper.
                            fetched = swarm_fetch_chain(
                                &transport, peer_pubkey_bytes, &sig_hash,
                                &sig_hash, &pool,
                            )
                            .await;
                            debug!(blobs = fetched.len(), "swarm-fetched chain blobs");
                            result = verify_once(&fetched);
                        }

                        match result {
                            Ok(_verified) => {
                                debug!(
                                    sig = %hex::encode(&sig_hash[..4]),
                                    issuer = %hex::encode(&peer_pubkey_bytes[..4]),
                                    "OP_DELIVER_CAP: chain verified; absorbing",
                                );
                                // Emit Blob events for everything the
                                // verify needed — the in-band leaf
                                // pair + every swarm-fetched parent.
                                // mpsc preserves order so the Peer
                                // thread sees these before the
                                // CapDelivered marker that triggers
                                // pinning.
                                let _ = events.send(NetEvent::Blob(cap_bytes.clone()));
                                let _ = events.send(NetEvent::Blob(sig_bytes.clone()));
                                for (_, bytes) in std::mem::take(&mut fetched) {
                                    let _ = events.send(NetEvent::Blob(
                                        anybytes::Bytes::from_source(bytes),
                                    ));
                                }
                                let _ = events.send(NetEvent::CapDelivered {
                                    issuer: peer_pubkey_bytes,
                                    cap_bytes,
                                    sig_bytes,
                                });
                                let _ = crate::handshake::respond(
                                    &mut send,
                                    crate::handshake::STATUS_OK,
                                )
                                .await;
                            }
                            Err(e) => {
                                warn!(
                                    error = ?e,
                                    sig = %hex::encode(&sig_hash[..4]),
                                    "OP_DELIVER_CAP: chain verify failed; rejecting",
                                );
                                let _ = crate::handshake::respond(
                                    &mut send,
                                    crate::handshake::STATUS_REJECTED,
                                )
                                .await;
                            }
                        }
                    }
                    Ok(None) => {
                        let _ = crate::handshake::respond(
                            &mut send,
                            crate::handshake::STATUS_MALFORMED,
                        )
                        .await;
                    }
                    Err(e) => {
                        debug!(error = %e, "handshake decode error; rejecting");
                        let _ = crate::handshake::respond(
                            &mut send,
                            crate::handshake::STATUS_MALFORMED,
                        )
                        .await;
                    }
                }
            }
        }
        .instrument(span)
        .await;
    }
}

impl<T: Transport> SnapshotHandler<T> {
    async fn handle(&self, connection: T::Conn) {
        let snap = self.snapshot.clone();
        let team_root = self.team_root;
        let transport = self.transport.clone();
        let self_cap = self.self_cap;
        let events = self.events.clone();
        let pool = self.pool.clone();

        let peer_id: PeerId = connection.remote_id();
        let span = info_span!(
            "connection",
            peer = %hex::encode(&peer_id[..4]),
            alpn = %String::from_utf8_lossy(PILE_SYNC_ALPN),
        );

        async move {
            info!("connection accepted");

            // The connecting peer's verified ed25519 identity from
            // the transport's TLS layer.
            let peer_pubkey = match ed25519_dalek::VerifyingKey::from_bytes(&peer_id) {
                Ok(k) => k,
                Err(e) => {
                    warn!(error = %e, "peer pubkey parse failed; closing");
                    return;
                }
            };

            // Per-connection auth state. Set by the first `OP_AUTH`
            // stream; read by every subsequent stream to gate access.
            let auth_state: Arc<
                tokio::sync::RwLock<Option<triblespace_core::repo::capability::VerifiedCapability>>,
            > = Arc::new(tokio::sync::RwLock::new(None));

            loop {
                let Some((mut send, mut recv)) = connection.accept_bi().await else {
                    debug!("accept_bi ended; connection closing");
                    break;
                };
                let snap = snap.clone();
                let auth_state = auth_state.clone();
                let transport = transport.clone();
                let events = events.clone();
                let pool = pool.clone();
                tokio::spawn(
                    async move {
                        if let Err(e) = serve_stream(
                            &snap,
                            team_root,
                            peer_pubkey,
                            auth_state,
                            &transport,
                            &self_cap,
                            &events,
                            &pool,
                            &mut send,
                            &mut recv,
                        )
                        .await
                        {
                            error!(error = %e, "stream handler error");
                        }
                        let _ = send.shutdown().await;
                    }
                    .in_current_span(),
                );
            }
        }
        .instrument(span)
        .await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn serve_stream<T: Transport>(
    snap_arc: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    team_root: ed25519_dalek::VerifyingKey,
    peer_pubkey: ed25519_dalek::VerifyingKey,
    auth_state: Arc<
        tokio::sync::RwLock<Option<triblespace_core::repo::capability::VerifiedCapability>>,
    >,
    t: &T,
    self_cap: &RawHash,
    events: &mpsc::Sender<NetEvent>,
    pool: &SharedPool<T::Conn>,
    send: &mut <T::Conn as Conn>::SendHalf,
    recv: &mut <T::Conn as Conn>::RecvHalf,
) -> anyhow::Result<()> {
    use triblespace_core::blob::Blob;
    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace_core::inline::Inline;
    use triblespace_core::inline::encodings::hash::Handle;

    let op = recv_u8(recv).await?;
    let span = debug_span!("stream", op = op_name(op));
    let _enter = span.enter();

    if op == OP_AUTH {
        let cap_handle_raw = recv_hash(recv).await?;
        debug!(cap_handle = %hex::encode(&cap_handle_raw[..4]), "auth: cap handle received");
        let cap_handle: Inline<Handle<SimpleArchive>> = Inline::new(cap_handle_raw);

        // Brief sync read inside async — guard is dropped before any
        // .await runs so this never blocks an async worker.
        // First-pass verify with local-only lookup. The common case is
        // "we already have the whole chain"; only retry with a swarm
        // fetch on the specific "missing blob" failure mode.
        let verify_once = |fetched: &std::collections::BTreeMap<RawHash, Vec<u8>>| {
            let snap_for_fetch = snap_arc.clone();
            let fetched_for_lookup = fetched.clone();
            triblespace_core::repo::capability::verify_chain(
                team_root,
                cap_handle,
                peer_pubkey,
                move |h: Inline<Handle<SimpleArchive>>| -> Option<Blob<SimpleArchive>> {
                    if let Some(bytes) = snap_for_fetch
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|s| s.get_blob(&h.raw))
                    {
                        return Some(Blob::new(anybytes::Bytes::from_source(bytes)));
                    }
                    let bytes = fetched_for_lookup.get(&h.raw)?.clone();
                    Some(Blob::new(anybytes::Bytes::from_source(bytes)))
                },
            )
        };

        let mut fetched: std::collections::BTreeMap<RawHash, Vec<u8>> =
            std::collections::BTreeMap::new();
        let mut result = verify_once(&fetched);

        // Swarm fetch + retry on missing-blob. Capability blobs are not
        // collection evidence, so on first auth from a peer whose chain we
        // have not cached, walk it via OP_CHILDREN and pull the blobs into a
        // local map. Sending peers verify our chain when we dial them (mutual
        // recursion that terminates because issued capabilities are held by
        // members of the team).
        if matches!(
            result,
            Err(triblespace_core::repo::capability::VerifyError::Fetch),
        ) {
            debug!(
                cap_handle = %hex::encode(&cap_handle_raw[..4]),
                "auth: chain incomplete locally, swarm-fetching",
            );
            let publisher: PeerId = peer_pubkey.to_bytes();
            fetched = swarm_fetch_chain(t, publisher, &cap_handle_raw, self_cap, pool).await;
            debug!(blobs = fetched.len(), "swarm-fetched chain blobs");
            result = verify_once(&fetched);
        }

        match result {
            Ok(verified) => {
                let granted = verified.granted_branches().map(|s| s.len()).unwrap_or(0);
                let unrestricted = verified.granted_branches().is_none();
                info!(branches = granted, unrestricted = unrestricted, "auth ok");
                // Cache the swarm-fetched blobs into the local store so
                // the next AUTH involving the same chain finds them
                // locally. mpsc preserves order; child-before-parent
                // ordering doesn't matter here because the chain is
                // already self-consistent (every parent referenced by
                // every fetched cap is also in `fetched`).
                for (_, bytes) in std::mem::take(&mut fetched) {
                    let _ = events.send(NetEvent::Blob(anybytes::Bytes::from_source(bytes)));
                }
                // Tell the Peer thread that this remote authed with
                // `cap_handle_raw`. If the Peer issued a cap to this
                // subject and `cap_handle_raw` matches the policy
                // entry's `latest_sig`, the Peer marks the entry as
                // delivered (the subject has the cap and can use it).
                let _ = events.send(NetEvent::CapDeliveryConfirmed {
                    subject: peer_pubkey.to_bytes(),
                    sig_handle: cap_handle_raw,
                });
                *auth_state.write().await = Some(verified);
                send_u8(send, AUTH_OK).await?;
            }
            Err(e) => {
                warn!(error = ?e, "auth rejected");
                send_u8(send, AUTH_REJECTED).await?;
            }
        }
        return Ok(());
    }

    // All other ops require a verified cap on the connection. Snapshot
    // the auth state once so the scope gate sees a stable view of the
    // verified cap for the rest of this stream's lifetime.
    let verified = match auth_state.read().await.clone() {
        Some(v) => v,
        None => {
            // Not authenticated. Close the stream silently — the client
            // should have presented OP_AUTH first.
            debug!("op without prior OP_AUTH on connection; closing stream");
            return Ok(());
        }
    };
    // Blob-level scope gate: `OP_GET_BLOB` and `OP_CHILDREN` are filtered by
    //    blob-graph reachability from the allowed heads. A peer with a
    //    cap restricted to branch X cannot fetch blobs that only branch
    //    Y reaches, even if they probe by raw hash. Unrestricted caps
    //    (`granted_branches() == None`) skip the reachability filter.
    //
    // Reachability is recomputed per OP_GET_BLOB / OP_CHILDREN call for
    // simplicity; for chain-walk-heavy workloads, a per-stream cache
    // would be the obvious next optimisation.

    match op {
        OP_GET_BLOB => {
            let hash = recv_hash(recv).await?;
            let in_scope_flag;
            let data = {
                let guard = snap_arc.lock().unwrap();
                let scope_ok = guard
                    .as_ref()
                    .map(|snap| blob_in_scope(snap.as_ref(), &verified, &hash))
                    .unwrap_or(false);
                in_scope_flag = scope_ok;
                guard.as_ref().and_then(|snap| {
                    if !scope_ok {
                        return None;
                    }
                    snap.get_blob(&hash)
                })
            };
            match data {
                Some(data) => {
                    debug!(hash = %hex::encode(&hash[..4]), bytes = data.len(), "OP_GET_BLOB served");
                    send_u64_be(send, data.len() as u64).await?;
                    send.write_all(&data)
                        .await
                        .map_err(|e| anyhow::anyhow!("send: {e}"))?;
                }
                None => {
                    if !in_scope_flag {
                        warn!(hash = %hex::encode(&hash[..4]), "OP_GET_BLOB denied: out of scope");
                    } else {
                        debug!(hash = %hex::encode(&hash[..4]), "OP_GET_BLOB miss: blob not present");
                    }
                    send_u64_be(send, u64::MAX).await?;
                }
            }
        }

        OP_CHILDREN => {
            let parent_hash = recv_hash(recv).await?;
            let mut parent_in_scope = true;
            let mut total_chunks = 0usize;
            let children: Vec<RawHash> = {
                let guard = snap_arc.lock().unwrap();
                match guard.as_ref() {
                    None => Vec::new(),
                    Some(snap) => {
                        // Compute the reachable set once for this op
                        // and check membership against it for every
                        // candidate — avoids the previous O(K×N) BFS
                        // re-walk per child.
                        let reachable = reachable_set_for(snap.as_ref(), &verified);
                        let in_scope = |hash: &RawHash| -> bool {
                            if !snap.has_blob(hash) {
                                return false;
                            }
                            match &reachable {
                                None => verified.grants_read(),
                                Some(set) => set.contains(hash),
                            }
                        };
                        if !in_scope(&parent_hash) {
                            parent_in_scope = false;
                            Vec::new()
                        } else {
                            match snap.get_blob(&parent_hash) {
                                None => Vec::new(),
                                Some(parent_data) => {
                                    let mut result = Vec::new();
                                    for chunk in parent_data.chunks(32) {
                                        if chunk.len() == 32 {
                                            total_chunks += 1;
                                            let mut candidate = [0u8; 32];
                                            candidate.copy_from_slice(chunk);
                                            if in_scope(&candidate) {
                                                result.push(candidate);
                                            }
                                        }
                                    }
                                    result
                                }
                            }
                        }
                    }
                }
            };
            if !parent_in_scope {
                warn!(parent = %hex::encode(&parent_hash[..4]), "OP_CHILDREN denied: parent out of scope");
            } else {
                debug!(
                    parent = %hex::encode(&parent_hash[..4]),
                    candidates = total_chunks,
                    in_scope = children.len(),
                    "OP_CHILDREN served"
                );
            }
            for hash in &children {
                send_hash(send, hash).await?;
            }
            send_hash(send, &NIL_HASH).await?;
        }

        OP_COLLECTION_EVIDENCE => {
            let collection = CollectionId::new(recv_hash(recv).await?);
            // Branch restrictions have no principled interpretation for
            // descriptor-addressed collections. Until capabilities gain a
            // collection scope, expose this operation only to read-equivalent
            // caps with no resource restriction.
            if !verified.grants_read() || verified.granted_branches().is_some() {
                warn!(
                    collection = %hex::encode(&collection.raw[..4]),
                    "OP_COLLECTION_EVIDENCE denied: unrestricted read required"
                );
                send_u32_be(send, COLLECTION_EVIDENCE_REJECTED).await?;
            } else {
                let evidence = {
                    let guard = snap_arc.lock().unwrap();
                    guard
                        .as_ref()
                        .map(|snapshot| snapshot.collection_evidence(collection))
                        .unwrap_or_default()
                };
                let count = u32::try_from(evidence.len())
                    .map_err(|_| anyhow::anyhow!("too many collection evidence records"))?;
                // `u32::MAX` is reserved for the authorization sentinel.
                if count == COLLECTION_EVIDENCE_REJECTED {
                    return Err(anyhow::anyhow!("too many collection evidence records"));
                }
                send_u32_be(send, count).await?;
                for item in evidence {
                    send.write_all(&item.encode())
                        .await
                        .map_err(|error| anyhow::anyhow!("send collection evidence: {error}"))?;
                }
                debug!(
                    collection = %hex::encode(&collection.raw[..4]),
                    count,
                    "OP_COLLECTION_EVIDENCE served"
                );
            }
        }

        OP_COLLECTION_OPERATION_RECEIPTS => {
            let mut request_bytes = [0u8; WANT_REQUEST_BYTES_LEN];
            let request = match recv.read_exact(&mut request_bytes).await {
                Ok(_) => {
                    let mut trailing = [0u8; 1];
                    match recv.read(&mut trailing).await {
                        Ok(0) => decode_collection_operation_request(request_bytes).ok(),
                        Ok(_) => {
                            debug!("collection operation request contains trailing bytes");
                            None
                        }
                        Err(error) => {
                            debug!(%error, "failed to finish collection operation request");
                            None
                        }
                    }
                }
                Err(error) => {
                    debug!(%error, "truncated collection operation request");
                    None
                }
            };
            // Like collection evidence, operation receipts are addressed by
            // collection descriptors rather than legacy branch ids. Until
            // capabilities carry collection scope, only an unrestricted read
            // grant has a principled meaning here.
            if !verified.grants_read() || verified.granted_branches().is_some() || request.is_none()
            {
                warn!(
                    valid_request = request.is_some(),
                    "OP_COLLECTION_OPERATION_RECEIPTS denied or malformed"
                );
                send_u32_be(send, COLLECTION_OPERATION_RECEIPTS_REJECTED).await?;
            } else {
                let request = request.expect("checked operation request");
                let receipts = {
                    let guard = snap_arc.lock().unwrap();
                    guard
                        .as_ref()
                        .map(|snapshot| snapshot.collection_operation_receipts(request))
                        .unwrap_or_default()
                };
                let response = encode_collection_operation_receipts(request, receipts)?;
                send.write_all(&response).await.map_err(|error| {
                    anyhow::anyhow!("send collection operation receipts: {error}")
                })?;
                debug!(
                    ?request,
                    count = (response.len() - 4)
                        / crate::collection_wire::COLLECTION_OPERATION_RECEIPT_BYTES_LEN,
                    "OP_COLLECTION_OPERATION_RECEIPTS served"
                );
            }
        }

        _ => {}
    }
    Ok(())
}

/// Build the reachable set for the given verified cap once. Returns
/// `None` if the cap is unrestricted (i.e. every present blob is in
/// scope — caller short-circuits to `snap.has_blob` checks).
/// Returns `Some(set)` for branch-restricted caps; the BFS walks
/// from each allowed branch's head following 32-byte child chunks
/// in blob bytes, just like the OP_CHILDREN handler does.
///
/// This is a per-op O(reachable subgraph) computation. Previously
/// `blob_in_scope` re-did this BFS for every blob a single
/// `OP_CHILDREN` response had to test (parent + every candidate
/// child) — worst case `O(K × N)` for K children and N reachable
/// blobs. Computing the set once amortises the BFS across the
/// whole response.
fn reachable_set_for(
    snap: &dyn AnySnapshot,
    verified: &triblespace_core::repo::capability::VerifiedCapability,
) -> Option<HashSet<RawHash>> {
    if verified.granted_branches().is_none() {
        // Unrestricted cap: every blob present in the snapshot is in
        // scope. The cap may still lack read permission entirely; in
        // that case `grants_read()` is false and the branch-level
        // gate would have filtered every head — caller cross-checks
        // via `verified.grants_read()` before consulting this set.
        return None;
    }

    let pin_heads = snap.pin_heads();
    let mut frontier: Vec<RawHash> = pin_heads
        .iter()
        .filter_map(|bid| {
            triblespace_core::id::Id::new(*bid)
                .filter(|id| verified.grants_read_on(id))
                .and_then(|_| pin_heads.get(bid).map(|h| h.raw))
        })
        .collect();
    let mut reachable: HashSet<RawHash> = HashSet::new();
    while let Some(h) = frontier.pop() {
        if !reachable.insert(h) {
            continue;
        }
        if let Some(data) = snap.get_blob(&h) {
            for chunk in data.chunks(32) {
                if chunk.len() == 32 {
                    let mut child = [0u8; 32];
                    child.copy_from_slice(chunk);
                    if snap.has_blob(&child) && !reachable.contains(&child) {
                        frontier.push(child);
                    }
                }
            }
        }
    }
    Some(reachable)
}

/// Returns `true` if `hash` is reachable (transitively, via 32-byte-chunk
/// children references) from at least one branch head the `verified` cap
/// grants read access on. Unrestricted caps short-circuit to `true` for
/// every hash present in the snapshot.
///
/// Convenience wrapper over [`reachable_set_for`] for callers that only
/// need to test a single hash. Multi-hash callers (e.g. `OP_CHILDREN`)
/// should compute the set once and check membership directly to avoid
/// recomputing the BFS per candidate.
fn blob_in_scope(
    snap: &dyn AnySnapshot,
    verified: &triblespace_core::repo::capability::VerifiedCapability,
    hash: &RawHash,
) -> bool {
    if !snap.has_blob(hash) {
        return false;
    }
    match reachable_set_for(snap, verified) {
        None => verified.grants_read(),
        Some(set) => set.contains(hash),
    }
}

#[cfg(test)]
mod collection_evidence_gossip_tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    use ed25519_dalek::SigningKey;
    use triblespace_core::collection::{
        CollectionCommit, CollectionGossip, CollectionId, CollectionRecord, empty_metadata_handle,
        simplearchive_union,
    };
    use triblespace_core::id::Id;
    use triblespace_core::inline::Inline;
    use triblespace_core::repo::{PinSnapshot, WantRequest};

    use super::{
        AnySnapshot, COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN, GOSSIP_COLLECTION_EVIDENCE,
        RoutingCandidates, collection_evidence_for_rebroadcast, collection_evidence_gossip_frame,
        decode_collection_evidence_gossip_frame, note_routing_candidate,
        remove_transient_routing_candidate,
    };
    use crate::collection_wire::{COLLECTION_COMMIT_EVIDENCE_LEN, CollectionCommitEvidence};

    fn evidence() -> CollectionCommitEvidence {
        let author = SigningKey::from_bytes(&[0xA7; 32]);
        let descriptor = simplearchive_union::descriptor(Id::new([0x3C; 16]).unwrap());
        let commit = CollectionCommit::sign(
            &author,
            descriptor.handle(),
            Inline::new([0xD4; 32]),
            empty_metadata_handle(),
        );
        CollectionCommitEvidence::new(CollectionGossip::sign(&author, descriptor.handle()), commit)
            .unwrap()
    }

    struct EvidenceSnapshot {
        pin_heads: PinSnapshot,
        evidence: CollectionCommitEvidence,
    }

    impl AnySnapshot for EvidenceSnapshot {
        fn get_blob(&self, _: &[u8; 32]) -> Option<Vec<u8>> {
            None
        }

        fn has_blob(&self, _: &[u8; 32]) -> bool {
            false
        }

        fn pin_heads(&self) -> &PinSnapshot {
            &self.pin_heads
        }

        fn collection_evidence(&self, collection: CollectionId) -> Vec<CollectionCommitEvidence> {
            (self.evidence.commit().collection() == collection)
                .then_some(self.evidence)
                .into_iter()
                .collect()
        }

        fn all_collection_evidence(&self) -> Vec<CollectionCommitEvidence> {
            vec![self.evidence]
        }

        fn collection_operation_receipts(&self, _: WantRequest) -> Vec<CollectionRecord> {
            Vec::new()
        }
    }

    #[test]
    fn collection_evidence_gossip_frame_is_exact_and_roundtrips_strictly() {
        let evidence = evidence();
        let nonce = 0x0102_0304_0506_0708;
        let frame = collection_evidence_gossip_frame(evidence, nonce);

        assert_eq!(COLLECTION_COMMIT_EVIDENCE_LEN, 320);
        assert_eq!(COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN, 329);
        assert_eq!(frame.len(), COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN);
        assert_eq!(frame[0], GOSSIP_COLLECTION_EVIDENCE);
        assert_eq!(
            &frame[1..1 + COLLECTION_COMMIT_EVIDENCE_LEN],
            &evidence.encode()
        );
        assert_eq!(
            &frame[1 + COLLECTION_COMMIT_EVIDENCE_LEN..],
            &nonce.to_be_bytes()
        );
        assert_eq!(
            decode_collection_evidence_gossip_frame(&frame),
            Some(evidence)
        );
    }

    #[test]
    fn collection_evidence_gossip_nonce_is_nonsemantic_but_signed_bytes_are_strict() {
        let evidence = evidence();
        let first = collection_evidence_gossip_frame(evidence, 1);
        let second = collection_evidence_gossip_frame(evidence, 2);
        assert_ne!(first, second);
        assert_eq!(
            decode_collection_evidence_gossip_frame(&second),
            Some(evidence)
        );

        let mut tampered = first.clone();
        tampered[1] ^= 0x80;
        assert_eq!(decode_collection_evidence_gossip_frame(&tampered), None);

        let mut wrong_tag = first.clone();
        wrong_tag[0] ^= 0x80;
        assert_eq!(decode_collection_evidence_gossip_frame(&wrong_tag), None);
        assert_eq!(decode_collection_evidence_gossip_frame(&first[..328]), None);
    }

    #[test]
    fn read_only_host_never_selects_periodic_evidence_for_publication() {
        let evidence = evidence();
        let snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
            Arc::new(Mutex::new(Some(Box::new(EvidenceSnapshot {
                pin_heads: PinSnapshot::default(),
                evidence,
            }))));

        assert!(collection_evidence_for_rebroadcast(&snapshot, false).is_empty());
        assert_eq!(
            collection_evidence_for_rebroadcast(&snapshot, true),
            vec![evidence]
        );
    }

    #[test]
    fn routing_candidates_follow_neighbor_liveness_without_dropping_configuration() {
        let configured_peer = [1; 32];
        let transient_peer = [2; 32];
        let candidates: RoutingCandidates = Arc::new(Mutex::new(vec![configured_peer]));
        let configured = HashSet::from([configured_peer]);

        note_routing_candidate(&candidates, transient_peer);
        assert_eq!(
            *candidates.lock().unwrap(),
            vec![transient_peer, configured_peer]
        );

        remove_transient_routing_candidate(&candidates, &configured, transient_peer);
        remove_transient_routing_candidate(&candidates, &configured, configured_peer);
        assert_eq!(*candidates.lock().unwrap(), vec![configured_peer]);
    }
}

#[cfg(test)]
mod serving_snapshot_tests {
    use std::convert::Infallible;
    use std::fmt;

    use ed25519_dalek::SigningKey;
    use iroh_base::EndpointId;
    use triblespace_core::blob::{BlobEncoding, IntoBlob};
    use triblespace_core::collection::{
        CollectionGossip, CollectionGossipStore, CollectionRecord, CollectionStore,
    };
    use triblespace_core::id::Id;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::inline::{Inline, InlineEncoding};
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{
        BlobStore, BlobStorePut, PinSnapshot, PinSnapshotSource, PinStore,
    };

    use super::wire;

    #[derive(Debug)]
    struct SnapshotUnavailable;

    impl fmt::Display for SnapshotUnavailable {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("snapshot unavailable")
        }
    }

    impl std::error::Error for SnapshotUnavailable {}

    /// A normal in-memory store behind a deliberately fallible observational
    /// pin-snapshot boundary. It does not implement PinStore itself, proving
    /// the host needs only the narrow source capability.
    struct FallibleSnapshotStore {
        inner: MemoryRepo,
        fail_snapshot: bool,
    }

    impl BlobStorePut for FallibleSnapshotStore {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
        where
            S: BlobEncoding + 'static,
            T: IntoBlob<S>,
            Handle<S>: InlineEncoding,
        {
            self.inner.put(item)
        }
    }

    impl BlobStore for FallibleSnapshotStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.inner.reader()
        }
    }

    impl CollectionStore for FallibleSnapshotStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = <MemoryRepo as CollectionStore>::RecordIter<'a>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.inner.records()
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.inner.insert(record)
        }
    }

    impl CollectionGossipStore for FallibleSnapshotStore {
        type GossipsError = Infallible;
        type GossipError = Infallible;
        type GossipIter<'a> = <MemoryRepo as CollectionGossipStore>::GossipIter<'a>;

        fn gossips<'a>(&'a mut self) -> Result<Self::GossipIter<'a>, Self::GossipsError> {
            self.inner.gossips()
        }

        fn gossip(&mut self, grant: CollectionGossip) -> Result<(), Self::GossipError> {
            self.inner.gossip(grant)
        }
    }

    impl PinSnapshotSource for FallibleSnapshotStore {
        type PinSnapshotError = SnapshotUnavailable;

        fn snapshot_pin_heads(&mut self) -> Result<PinSnapshot, Self::PinSnapshotError> {
            if self.fail_snapshot {
                Err(SnapshotUnavailable)
            } else {
                Ok(PinStore::pin_snapshot(&mut self.inner)
                    .expect("MemoryRepo pin snapshots are infallible"))
            }
        }
    }

    #[test]
    fn failed_refresh_clears_a_previously_authorizing_snapshot() {
        let branch = Id::new([0x31; 16]).unwrap();
        let head = Inline::new([0x42; 32]);
        let mut inner = MemoryRepo::default();
        assert!(matches!(
            inner.update(branch, None, Some(head)).unwrap(),
            triblespace_core::repo::PushResult::Success()
        ));
        let mut store = FallibleSnapshotStore {
            inner,
            fail_snapshot: false,
        };

        let key = SigningKey::from_bytes(&[0x71; 32]);
        let endpoint = EndpointId::from_bytes(&key.verifying_key().to_bytes()).unwrap();
        let (sender, _receiver, wiring) = wire(endpoint);

        assert!(sender.refresh_store_snapshot(&mut store));
        {
            let guard = wiring.snapshot.lock().unwrap();
            let snapshot = guard.as_ref().expect("first snapshot succeeds");
            assert_eq!(
                snapshot.pin_heads().get(&branch.into()),
                Some(&head),
                "the good snapshot authorizes reachability from this pin head"
            );
        }

        store.fail_snapshot = true;
        assert!(!sender.refresh_store_snapshot(&mut store));
        assert!(
            wiring.snapshot.lock().unwrap().is_none(),
            "construction failure must clear the prior roots; request handlers deny when no snapshot is installed"
        );

        store.fail_snapshot = false;
        assert!(sender.refresh_store_snapshot(&mut store));
        assert!(
            wiring.snapshot.lock().unwrap().is_some(),
            "a later successful refresh restores service"
        );
    }
}
