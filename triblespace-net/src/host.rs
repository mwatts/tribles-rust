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
use tracing::{Instrument, debug, debug_span, info, info_span, instrument, trace, warn};
use triblespace_core::collection::{
    COLLECTION_COMMIT_BYTES_LEN, CollectionCommit, CollectionHandle, CollectionRecord,
    CollectionStore,
};
use triblespace_core::repo::{WANT_REQUEST_BYTES_LEN, WantRequest};

use crate::channel::{NetCommand, NetEvent};
use crate::collection_wire::{
    collection_operation_receipts, decode_collection_operation_request,
    encode_collection_operation_receipts, op_collection_operation_receipts, relayable_commits,
    relayable_commits_for,
};
use crate::identity::iroh_secret;
use crate::protocol::*;
use crate::transport::{Conn, GossipEvent, GossipSink, Harness, PeerId, Transport};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Team-topic domain tag for immutable collection commit evidence.
const GOSSIP_COLLECTION_EVIDENCE: u8 = 0x03;
/// tag(1) + signed commit(192) + anti-dedupe nonce(8).
const COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN: usize = 1 + COLLECTION_COMMIT_BYTES_LEN + 8;

/// Encode one immutable collection-evidence gossip frame.
///
/// The nonce is intentionally outside the signed evidence. It only gives
/// periodic republishes distinct mesh message ids, so a late joiner can learn
/// unchanged evidence without changing its canonical 192-byte identity.
fn collection_evidence_gossip_frame(evidence: CollectionCommit, nonce: u64) -> Vec<u8> {
    let mut frame = Vec::with_capacity(COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN);
    frame.push(GOSSIP_COLLECTION_EVIDENCE);
    frame.extend_from_slice(&evidence.to_bytes());
    frame.extend_from_slice(&nonce.to_be_bytes());
    debug_assert_eq!(frame.len(), COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN);
    frame
}

/// Decode and strictly verify one complete collection-evidence gossip frame.
///
/// The transport carrier and anti-dedupe nonce are deliberately discarded:
/// neither participates in author identity or local admission policy.
fn decode_collection_evidence_gossip_frame(bytes: &[u8]) -> Option<CollectionCommit> {
    if bytes.len() != COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN
        || bytes.first().copied() != Some(GOSSIP_COLLECTION_EVIDENCE)
    {
        return None;
    }
    bytes[1..1 + COLLECTION_COMMIT_BYTES_LEN]
        .try_into()
        .ok()
        .map(CollectionCommit::from_bytes)
        .filter(|commit: &CollectionCommit| commit.verify_strict().is_ok())
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
/// `Default` impl — auth is mandatory in protocol v7 so every peer
/// construction site must explicitly choose a CONNECT trust root. For solo
/// workflows the convention is `connect_root = signing_key.verifying_key()`
/// (the user is their own trust root);
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
    /// Explicit collection-evidence gossip topic. `None` is serve-/pull-only
    /// (no subscription, no broadcasts). This rendezvous choice is orthogonal
    /// to CONNECT authorization and is never derived from `connect_root`.
    pub gossip_topic: Option<[u8; 32]>,
    /// External trust-root public key for direct RPC. Every connection's first
    /// stream must carry a complete capability proof rooted here whose leaf
    /// invokes exact CONNECT on these same 32 public-key bytes.
    pub connect_root: ed25519_dalek::VerifyingKey,
    /// Complete, prebuilt root-to-leaf proof authorizing this node's TLS key
    /// to invoke [`crate::protocol::ACTION_CONNECT`] on the exact
    /// `connect_root` resource. Outgoing dials send these bytes inline; the
    /// transport never constructs or fetches proof state implicitly.
    pub connect_proof: triblespace_core::capability::CapabilityProof,
    /// Direction of participation in the evidence swarm. Controls whether this
    /// node publishes collection evidence (write side) and/or admits incoming
    /// collection evidence (read side). Default is `Bidirectional`. Use
    /// [`SyncDirection::ReadOnly`] for follower/catch-up workflows; use
    /// [`SyncDirection::WriteOnly`] for pure-publisher workflows where the
    /// local node has nothing to learn from the swarm.
    pub direction: SyncDirection,
}

/// Which directions of the evidence swarm this node participates in.
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

// No `Default` impl: every PeerConfig must specify a trust root because auth
// is mandatory in protocol v7. For a single-user OSS deployment the convention
// is `connect_root = signing_key.verifying_key()`.

/// Snapshot of store state for serving protocol requests.
pub struct StoreSnapshot<R> {
    pub reader: R,
    collection_records: Vec<CollectionRecord>,
}

impl StoreSnapshot<()> {
    pub fn from_store<S>(store: &mut S) -> Option<StoreSnapshot<S::Reader>>
    where
        S: triblespace_core::repo::BlobStore + CollectionStore,
    {
        // Collection evidence is an additive serving capability. Failure to
        // enumerate it must never suppress the blob snapshot (notably,
        // legacy model piles remain directly readable).
        // Invalid structural evidence is inert and simply absent from this
        // serving view.
        let collection_records = store
            .records()
            .map(|records| records.filter_map(Result::ok).collect())
            .unwrap_or_default();
        let reader = store.reader().ok()?;
        Some(StoreSnapshot {
            reader,
            collection_records,
        })
    }
}

/// Type-erased snapshot for the host thread.
///
/// Carries just enough of the pile for the network thread to serve peer
/// requests: per-hash blob access and canonical collection evidence.
pub trait AnySnapshot: Send + 'static {
    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>>;
    fn has_blob(&self, hash: &RawHash) -> bool;
    /// Strictly verified commits for one exact descriptor handle whose own
    /// descriptor permits relay, in deterministic intrinsic-record order.
    fn collection_evidence(&self, collection: CollectionHandle) -> Vec<CollectionCommit>;
    /// Every relayable commit in deterministic commit-id order.
    /// Used only to periodically republish the current store truth for late
    /// gossip joiners; the host does not maintain a second ledger mirror.
    fn all_collection_evidence(&self) -> Vec<CollectionCommit>;
    /// Exact unsigned merge/derive receipts answering one durable question,
    /// in deterministic intrinsic-record order. Conflicting answers remain
    /// distinct records.
    fn collection_operation_receipts(&self, request: WantRequest) -> Vec<CollectionRecord>;
}

impl<R> StoreSnapshot<R>
where
    R: triblespace_core::repo::BlobStoreGet,
{
    /// Resolve one collection descriptor out of this snapshot's reader.
    ///
    /// `None` means the descriptor is not resident here, which relay
    /// selection reads as a refusal: a node that cannot see a collection's
    /// permission does not have it.
    fn descriptor(
        &self,
        collection: CollectionHandle,
    ) -> Option<triblespace_core::trible::TribleSet> {
        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
        self.reader
            .get::<triblespace_core::trible::TribleSet, SimpleArchive>(collection)
            .ok()
    }
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

    fn collection_evidence(&self, collection: CollectionHandle) -> Vec<CollectionCommit> {
        relayable_commits_for(
            &self.collection_records,
            |handle| self.descriptor(handle),
            collection,
        )
    }

    fn all_collection_evidence(&self) -> Vec<CollectionCommit> {
        relayable_commits(&self.collection_records, |handle| self.descriptor(handle))
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

/// Dialable peers, most-recent live gossip neighbor first. Explicitly
/// configured peers seed the list; neighbor events keep it current. These are
/// routing candidates, not claims that a peer holds any particular blob.
/// `Vec` preserves deterministic simulation replay order.
type RoutingCandidates = Arc<Mutex<Vec<PeerId>>>;

/// Cap on the remembered routing list. Gossip meshes are expected to be small;
/// eight recent
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
) -> Vec<CollectionCommit> {
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
    connect_proof: triblespace_core::capability::CapabilityProof,
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
        let connect_proof = self.connect_proof.clone();
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
            // Route-first: try known dialable peers before the DHT. They
            // are not presumed holders; an ordinary miss simply falls through.
            let mut data = if known.is_empty() {
                None
            } else {
                fetch_from_providers(&t, &hash, &pool, &known, &connect_proof).await
            };
            // DHT fallback: no publisher known, or none of them held it.
            if data.is_none() {
                data = fetch_one(&t, &hash, &pool, my_id, &connect_proof).await;
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
        let connect_proof = self.connect_proof.clone();
        let peers = self.configured_peers.clone();
        peers
            .into_iter()
            .map(|peer| {
                let transport = transport.clone();
                let pool = pool.clone();
                let connect_proof = connect_proof.clone();
                Box::pin(async move {
                    let Some(connection) = pool_get(&transport, &pool, peer, &connect_proof).await
                    else {
                        return CollectionOperationPeerProbe::Incomplete;
                    };
                    match tokio::time::timeout(
                        OP_DEADLINE,
                        op_collection_operation_receipts(&connection, request),
                    )
                    .await
                    {
                        Ok(Ok(receipts)) => CollectionOperationPeerProbe::Complete(receipts),
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
    pub fn gossip_collection_evidence(&self, evidence: CollectionCommit) {
        let _ = self
            .cmd_tx
            .send(NetCommand::GossipCollectionEvidence { evidence });
    }

    pub fn update_snapshot(&self, snapshot: impl AnySnapshot) {
        let boxed: Box<dyn AnySnapshot> = Box::new(snapshot);
        *self.snapshot.lock().unwrap() = Some(boxed);
    }

    /// Remove the serving view immediately. Every authenticated data operation
    /// treats an absent snapshot as unavailable, so this is the fail-closed
    /// transition when a current store view cannot be produced.
    fn clear_snapshot(&self) {
        *self.snapshot.lock().unwrap() = None;
    }

    /// Replace the host's serving view with the store's current snapshot.
    ///
    /// Snapshot construction is replacement semantics, not best-effort cache
    /// refresh. Failure clears the slot rather than serving stale bytes or
    /// collection evidence; a later successful refresh restores service.
    pub(crate) fn refresh_store_snapshot<S>(&self, store: &mut S) -> bool
    where
        S: triblespace_core::repo::BlobStore + CollectionStore,
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
        collection: CollectionHandle,
    ) -> anyhow::Result<Vec<CollectionCommit>> {
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

/// Deadline for establishing + authenticating a connection (the `pool_get`
/// init future: dial + inline CONNECT proof round trip). A connection
/// attempt that exceeds this counts as failed: the pool's
/// singleflight cell resets so the next walk re-dials, instead of
/// every later fetch to that peer queueing forever behind one
/// stalled authentication exchange. Generous relative to real-world QUIC + relay
/// setup times; deterministic under simulated virtual time.
const DIAL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// Deadline for a single protocol op (OP_CHILDREN / OP_GET_BLOB request + full
/// response) on an established connection. On expiry the op reports an error
/// and the caller's existing evict-and-try-next-provider path takes over.
/// Total-op rather than progress-based; large-content streaming may eventually
/// warrant an idle deadline instead.
const OP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Connect to a peer over the pile-sync ALPN and immediately present
/// our complete CONNECT proof so subsequent direct RPCs are admitted.
#[instrument(level = "info", skip(t, connect_proof), fields(peer = %hex::encode(&peer[..4])))]
async fn connect_authed<T: Transport>(
    t: &T,
    peer: PeerId,
    connect_proof: &triblespace_core::capability::CapabilityProof,
) -> anyhow::Result<T::Conn> {
    let conn = t.dial(peer, PILE_SYNC_ALPN).await.map_err(|e| {
        warn!(error = %e, "connect failed");
        anyhow::anyhow!("connect: {e}")
    })?;
    debug!(
        steps = connect_proof.steps().len(),
        "connected; sending OP_AUTH"
    );
    op_auth(&conn, connect_proof).await.map_err(|e| {
        warn!(error = %e, "CONNECT authentication failed");
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
    let connect_proof = config.connect_proof.clone();
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
    // remote peer, reused across direct blob and collection fetches. See
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
        connect_proof: connect_proof.clone(),
        my_id,
        candidates: routing_candidates.clone(),
        configured_peers,
    }) as Arc<dyn NetCapability>));

    // ── Inbound connections. Each connection gets its own task and
    // accepts sequential bi-streams until the peer closes.
    let snapshot_handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        connect_root: config.connect_root,
    };
    let mut incoming = incoming;
    tokio::spawn(async move {
        while let Some(inc) = incoming.recv().await {
            if inc.alpn == PILE_SYNC_ALPN {
                let h = snapshot_handler.clone();
                tokio::spawn(async move { h.handle::<T>(inc.conn).await });
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
                NetCommand::FetchCollectionEvidence {
                    peer,
                    collection,
                    reply,
                } => {
                    let transport = transport.clone();
                    let pool = conn_pool.clone();
                    let connect_proof = connect_proof.clone();
                    tokio::spawn(async move {
                        let result = async {
                            if peer == my_id {
                                anyhow::bail!("collection reconciliation peer is the local node");
                            }
                            let Some(connection) =
                                pool_get(&transport, &pool, peer, &connect_proof).await
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
/// the exact first candidate. Otherwise query the DHT.
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
/// per remote peer, shared across all direct blob and collection fetches.
///
/// `OnceCell` per peer provides automatic singleflight: the first
/// task to encounter a missing entry runs the dial; concurrent tasks
/// await the same `OnceCell` and reuse the resulting connection. No
/// dial-storm when several reads target the same peer concurrently.
///
/// iroh QUIC multiplexes streams cheaply on a single connection; our
/// `serve_stream` accepts unbounded sequential bi-streams per
/// connection. The handler admits request streams only after the first
/// OP_AUTH stream succeeds. So one connection per peer is enough.
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
    connect_proof: &triblespace_core::capability::CapabilityProof,
) -> Option<T::Conn> {
    let cell = {
        let mut guard = pool.lock().await;
        guard
            .entry(provider)
            .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
            .clone()
    };
    let init = || async {
        match tokio::time::timeout(DIAL_DEADLINE, connect_authed(t, provider, connect_proof)).await
        {
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
    connect_proof: &triblespace_core::capability::CapabilityProof,
) -> Option<Vec<u8>> {
    let providers = providers_for(t, hash, publisher_id).await;
    fetch_from_providers(t, hash, pool, &providers, connect_proof).await
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
    connect_proof: &triblespace_core::capability::CapabilityProof,
) -> Option<Vec<u8>> {
    for &provider in providers {
        let Some(conn) = pool_get(t, pool, provider, connect_proof).await else {
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

// ── Protocol handler ─────────────────────────────────────────────────

/// Wait until the first clock reading strictly after an inclusive capability
/// upper bound. Long intervals are rechecked daily so converting hifitime's
/// i128 range into Tokio's timer range cannot overflow.
fn capability_expired(expires: Option<hifitime::Epoch>) -> bool {
    expires.is_some_and(|upper| crate::clock::epoch_now() > upper)
}

async fn wait_until_after(expires: Option<hifitime::Epoch>) {
    let Some(upper) = expires else {
        std::future::pending::<()>().await;
        return;
    };
    let upper_ns = upper.to_tai_duration().total_nanoseconds();
    const MAX_SLEEP_NS: u64 = 24 * 60 * 60 * 1_000_000_000;

    loop {
        let now_ns = crate::clock::epoch_now()
            .to_tai_duration()
            .total_nanoseconds();
        if now_ns > upper_ns {
            return;
        }
        let remaining_ns = upper_ns.saturating_sub(now_ns).saturating_add(1);
        let sleep_ns = u64::try_from(remaining_ns)
            .unwrap_or(u64::MAX)
            .min(MAX_SLEEP_NS);
        tokio::time::sleep(std::time::Duration::from_nanos(sleep_ns)).await;
    }
}

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    connect_root: ed25519_dalek::VerifyingKey,
}

impl SnapshotHandler {
    async fn handle<T: Transport>(&self, connection: T::Conn) {
        let snapshot = self.snapshot.clone();
        let connect_root = self.connect_root;
        let peer_id = connection.remote_id();
        let span = info_span!(
            "connection",
            peer = %hex::encode(&peer_id[..4]),
            alpn = %String::from_utf8_lossy(PILE_SYNC_ALPN),
        );

        async move {
            info!("connection accepted");
            let peer = match ed25519_dalek::VerifyingKey::from_bytes(&peer_id) {
                Ok(peer) => peer,
                Err(error) => {
                    warn!(%error, "peer public key is malformed; closing");
                    connection.close(1, b"invalid peer identity");
                    return;
                }
            };

            // Authentication is deliberately structural control flow rather
            // than mutable per-connection state: the first stream proves the
            // exact CONNECT claim, then and only then can request streams run.
            let Some((mut send, mut recv)) = connection.accept_bi().await else {
                debug!("connection ended before OP_AUTH");
                return;
            };
            let authenticated =
                authenticate_connection::<T::Conn>(connect_root, peer, &mut send, &mut recv).await;
            let _ = send.shutdown().await;
            let expires = match authenticated {
                Ok(Some(verified)) => verified
                    .effective_validity()
                    .map(|validity| validity.bounds().1),
                Ok(None) => {
                    connection.close(1, b"CONNECT capability required");
                    return;
                }
                Err(error) => {
                    warn!(%error, "authentication stream failed");
                    connection.close(1, b"malformed authentication");
                    return;
                }
            };

            let expiry = wait_until_after(expires);
            tokio::pin!(expiry);

            loop {
                let stream = tokio::select! {
                    stream = connection.accept_bi() => stream,
                    () = &mut expiry => {
                        info!("CONNECT capability expired; closing connection");
                        connection.close(1, b"CONNECT capability expired");
                        return;
                    }
                };
                let Some((mut send, mut recv)) = stream else {
                    debug!("accept_bi ended; connection closing");
                    break;
                };
                // Recheck at the admission boundary as well as using the idle
                // timer above. This closes the one scheduler race where the
                // wall clock advances just before the expiry task is polled.
                if capability_expired(expires) {
                    connection.close(1, b"CONNECT capability expired");
                    return;
                }
                let snapshot = snapshot.clone();
                tokio::spawn(
                    async move {
                        if let Err(error) =
                            serve_stream::<T::Conn>(&snapshot, &mut send, &mut recv).await
                        {
                            debug!(%error, "direct RPC stream failed");
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

async fn authenticate_connection<C: Conn>(
    connect_root: ed25519_dalek::VerifyingKey,
    peer: ed25519_dalek::VerifyingKey,
    send: &mut C::SendHalf,
    recv: &mut C::RecvHalf,
) -> anyhow::Result<Option<triblespace_core::capability::VerifiedCapability>> {
    use triblespace_core::capability::{CapabilityClaim, CapabilityMode};

    let verdict = async {
        let op = recv_u8(recv).await?;
        if op != OP_AUTH {
            anyhow::bail!(
                "first stream operation is {}, expected OP_AUTH",
                op_name(op)
            );
        }
        let proof = recv_capability_proof(recv).await?;
        let mut trailing = [0u8; 1];
        if recv.read(&mut trailing).await? != 0 {
            anyhow::bail!("OP_AUTH contains trailing bytes");
        }
        let verified = proof.verify_claim(
            connect_root,
            crate::clock::epoch_now(),
            CapabilityClaim::new(
                peer,
                connect_capability_atom(connect_root),
                CapabilityMode::Invoke,
            ),
        )?;
        anyhow::Ok(verified)
    }
    .await;

    match verdict {
        Ok(verified) => {
            info!("CONNECT capability verified");
            send_u8(send, AUTH_OK).await?;
            Ok(Some(verified))
        }
        Err(error) => {
            warn!(%error, "CONNECT capability rejected");
            send_u8(send, AUTH_REJECTED).await?;
            Ok(None)
        }
    }
}

async fn serve_stream<C: Conn>(
    snapshot: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    send: &mut C::SendHalf,
    recv: &mut C::RecvHalf,
) -> anyhow::Result<()> {
    let op = recv_u8(recv).await?;
    let span = debug_span!("stream", op = op_name(op));
    let _enter = span.enter();

    match op {
        OP_GET_BLOB => {
            let hash = recv_hash(recv).await?;
            let data = snapshot
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|snapshot| snapshot.get_blob(&hash));
            match data {
                Some(data) => {
                    debug!(hash = %hex::encode(&hash[..4]), bytes = data.len(), "OP_GET_BLOB served");
                    send_u64_be(send, data.len() as u64).await?;
                    send.write_all(&data)
                        .await
                        .map_err(|error| anyhow::anyhow!("send blob: {error}"))?;
                }
                None => {
                    debug!(hash = %hex::encode(&hash[..4]), "OP_GET_BLOB miss");
                    send_u64_be(send, u64::MAX).await?;
                }
            }
        }
        OP_CHILDREN => {
            let parent_hash = recv_hash(recv).await?;
            let mut total_chunks = 0usize;
            let children: Vec<RawHash> = {
                let guard = snapshot.lock().unwrap();
                match guard.as_ref() {
                    None => Vec::new(),
                    Some(snapshot) => match snapshot.get_blob(&parent_hash) {
                        None => Vec::new(),
                        Some(parent_data) => {
                            let mut result = Vec::new();
                            for chunk in parent_data.chunks(32) {
                                if chunk.len() == 32 {
                                    total_chunks += 1;
                                    let mut candidate = [0u8; 32];
                                    candidate.copy_from_slice(chunk);
                                    if snapshot.has_blob(&candidate) {
                                        result.push(candidate);
                                    }
                                }
                            }
                            result
                        }
                    },
                }
            };
            debug!(
                parent = %hex::encode(&parent_hash[..4]),
                candidates = total_chunks,
                in_scope = children.len(),
                "OP_CHILDREN served"
            );
            for hash in children {
                send_hash(send, &hash).await?;
            }
            send_hash(send, &NIL_HASH).await?;
        }
        OP_COLLECTION_EVIDENCE => {
            let collection = CollectionHandle::new(recv_hash(recv).await?);
            let evidence = snapshot
                .lock()
                .unwrap()
                .as_ref()
                .map(|snapshot| snapshot.collection_evidence(collection))
                .unwrap_or_default();
            let count = u32::try_from(evidence.len())
                .map_err(|_| anyhow::anyhow!("too many collection evidence records"))?;
            send_u32_be(send, count).await?;
            for item in evidence {
                send.write_all(&item.to_bytes())
                    .await
                    .map_err(|error| anyhow::anyhow!("send collection evidence: {error}"))?;
            }
            debug!(
                collection = %hex::encode(&collection.raw[..4]),
                count,
                "OP_COLLECTION_EVIDENCE served"
            );
        }
        OP_COLLECTION_OPERATION_RECEIPTS => {
            let mut request_bytes = [0u8; WANT_REQUEST_BYTES_LEN];
            recv.read_exact(&mut request_bytes).await.map_err(|error| {
                anyhow::anyhow!("truncated collection operation request: {error}")
            })?;
            let mut trailing = [0u8; 1];
            if recv.read(&mut trailing).await? != 0 {
                anyhow::bail!("collection operation request contains trailing bytes");
            }
            let request = decode_collection_operation_request(request_bytes)?;
            let receipts = snapshot
                .lock()
                .unwrap()
                .as_ref()
                .map(|snapshot| snapshot.collection_operation_receipts(request))
                .unwrap_or_default();
            let response = encode_collection_operation_receipts(request, receipts)?;
            send.write_all(&response)
                .await
                .map_err(|error| anyhow::anyhow!("send collection operation receipts: {error}"))?;
            debug!(
                ?request,
                count = (response.len() - 4)
                    / crate::collection_wire::COLLECTION_OPERATION_RECEIPT_BYTES_LEN,
                "OP_COLLECTION_OPERATION_RECEIPTS served"
            );
        }
        OP_AUTH => anyhow::bail!("OP_AUTH may only appear on the first stream"),
        _ => anyhow::bail!("unknown direct RPC operation {op:#x}"),
    }
    Ok(())
}
#[cfg(test)]
mod collection_evidence_gossip_tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use triblespace_core::collection::reach;

    use ed25519_dalek::SigningKey;
    use triblespace_core::collection::{
        CollectionHandle, CollectionRecord, empty_metadata_handle, simplearchive_union,
    };
    use triblespace_core::inline::Inline;
    use triblespace_core::repo::WantRequest;

    use super::{
        AnySnapshot, COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN, GOSSIP_COLLECTION_EVIDENCE,
        RoutingCandidates, collection_evidence_for_rebroadcast, collection_evidence_gossip_frame,
        decode_collection_evidence_gossip_frame, note_routing_candidate,
        remove_transient_routing_candidate,
    };
    use triblespace_core::collection::{COLLECTION_COMMIT_BYTES_LEN, CollectionCommit};

    fn evidence() -> CollectionCommit {
        use triblespace_core::blob::IntoBlob;
        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
        use triblespace_core::collection::records::CollectionName;

        let author = SigningKey::from_bytes(&[0xA7; 32]);
        let descriptor = simplearchive_union::descriptor(
            &CollectionName::new("gossiped").unwrap(),
            author.verifying_key(),
            Some(author.verifying_key()),
            reach::private(),
        );
        // Gossip only ever carries the identity; nothing here stores the
        // descriptor it names -- these frame tests are about framing, and the
        // descriptor check lives in selection, one layer up.
        let collection = IntoBlob::<SimpleArchive>::to_blob(descriptor.into_facts()).get_handle();
        CollectionCommit::sign(
            &author,
            collection,
            Inline::new([0xD4; 32]),
            empty_metadata_handle(),
        )
    }

    struct EvidenceSnapshot {
        evidence: CollectionCommit,
    }

    impl AnySnapshot for EvidenceSnapshot {
        fn get_blob(&self, _: &[u8; 32]) -> Option<Vec<u8>> {
            None
        }

        fn has_blob(&self, _: &[u8; 32]) -> bool {
            false
        }

        fn collection_evidence(&self, collection: CollectionHandle) -> Vec<CollectionCommit> {
            (self.evidence.collection() == collection)
                .then_some(self.evidence)
                .into_iter()
                .collect()
        }

        fn all_collection_evidence(&self) -> Vec<CollectionCommit> {
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

        assert_eq!(COLLECTION_COMMIT_BYTES_LEN, 192);
        assert_eq!(COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN, 201);
        assert_eq!(frame.len(), COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN);
        assert_eq!(frame[0], GOSSIP_COLLECTION_EVIDENCE);
        assert_eq!(
            &frame[1..1 + COLLECTION_COMMIT_BYTES_LEN],
            &evidence.to_bytes()
        );
        assert_eq!(
            &frame[1 + COLLECTION_COMMIT_BYTES_LEN..],
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
        assert_eq!(
            decode_collection_evidence_gossip_frame(
                &first[..COLLECTION_EVIDENCE_GOSSIP_FRAME_LEN - 1]
            ),
            None
        );
    }

    #[test]
    fn read_only_host_never_selects_periodic_evidence_for_publication() {
        let evidence = evidence();
        let snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
            Arc::new(Mutex::new(Some(Box::new(EvidenceSnapshot { evidence }))));

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
