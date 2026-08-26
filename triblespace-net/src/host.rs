//! Network thread: spawns iroh endpoint, gossip, DHT, protocol server.
//!
//! Private implementation detail of [`crate::peer::Peer`] — `spawn()`
//! returns the [`NetSender`] / [`NetReceiver`] pair the Peer uses to
//! communicate with the async world (commands + snapshot updates one
//! way, events the other).
//!
//! Async is jailed inside the spawned thread.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
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
use crate::replica::{
    ReplicaBlobFetchError, ReplicaComponent, ReplicaGeneration, ReplicaItem, ReplicaItemId,
    ReplicaServerConfig, ReplicaSummary,
};
use crate::replica_wire::{
    OP_REPLICA_BLOB, OP_REPLICA_PAGE, OP_REPLICA_SUMMARY, recv_blob_request, recv_page_request,
    recv_request_prefix, send_blob_range, send_page, send_summary,
};
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
        OP_REPLICA_SUMMARY => "REPLICA_SUMMARY",
        OP_REPLICA_PAGE => "REPLICA_PAGE",
        OP_REPLICA_BLOB => "REPLICA_BLOB",
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
/// `Default` impl — auth is mandatory in protocol v8 so every peer
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
    /// stream must carry a complete capability proof bundle rooted here whose
    /// final key invokes exact CONNECT on these same 32 public-key bytes.
    pub connect_root: ed25519_dalek::VerifyingKey,
    /// Complete, prebuilt root-to-leaf proof bundle authorizing this node's
    /// TLS key to invoke [`crate::protocol::ACTION_CONNECT`] on the exact
    /// `connect_root` resource. Outgoing dials send these bytes inline; the
    /// transport never constructs or fetches claim state implicitly.
    pub connect_proof: triblespace_core::capability::CapabilityProofBundle,
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
// is mandatory in protocol v8. For a single-user OSS deployment the convention
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

/// Type-erased immutable view used only by explicit custody replication.
///
/// This is deliberately separate from [`AnySnapshot`]: constructing the full
/// resident inventory is expensive and must never become part of ordinary
/// [`Peer`](crate::peer::Peer) refreshes.
pub(crate) trait AnyReplicaSnapshot: Send + 'static {
    fn summary(&self) -> ReplicaSummary;
    fn page(
        &self,
        component: ReplicaComponent,
        prefix: u8,
        after: Option<ReplicaItemId>,
    ) -> (Vec<ReplicaItem>, bool);
    fn blob_bytes(&self, id: ReplicaItemId) -> Option<anybytes::Bytes>;
}

/// One immutable custody generation shared by the current serving slot and
/// any walks that already observed its summary.
type SharedReplicaSnapshot = Arc<Mutex<Box<dyn AnyReplicaSnapshot>>>;
type ReplicaSnapshotSlot = Arc<Mutex<Option<SharedReplicaSnapshot>>>;

/// Bounded reconnect cache for immutable custody generations.
///
/// A generation token is content-derived, and current REPLICATE authority
/// grants the whole replica set. It therefore identifies a cached snapshot
/// without being bound to the peer that first received its summary. Sharing
/// by generation collapses identical concurrent walks onto one allocation.
#[derive(Default)]
struct ReplicaSnapshotCache {
    snapshots: HashMap<ReplicaGeneration, SharedReplicaSnapshot>,
    least_to_most_recent: VecDeque<ReplicaGeneration>,
}

impl ReplicaSnapshotCache {
    fn insert(&mut self, generation: ReplicaGeneration, snapshot: SharedReplicaSnapshot) {
        if !self.snapshots.contains_key(&generation) {
            self.snapshots.insert(generation, snapshot);
        }
        self.touch(generation);
        while self.snapshots.len() > MAX_INBOUND_CONNECTIONS_GLOBAL {
            let oldest = self
                .least_to_most_recent
                .pop_front()
                .expect("nonempty generation cache has an LRU entry");
            self.snapshots.remove(&oldest);
        }
    }

    fn get(&mut self, generation: ReplicaGeneration) -> Option<SharedReplicaSnapshot> {
        let snapshot = self.snapshots.get(&generation)?.clone();
        self.touch(generation);
        Some(snapshot)
    }

    fn touch(&mut self, generation: ReplicaGeneration) {
        if let Some(position) = self
            .least_to_most_recent
            .iter()
            .position(|candidate| *candidate == generation)
        {
            self.least_to_most_recent.remove(position);
        }
        self.least_to_most_recent.push_back(generation);
    }
}

type ReplicaSnapshotGenerations = Arc<Mutex<ReplicaSnapshotCache>>;

fn shared_replica_snapshot(snapshot: impl AnyReplicaSnapshot) -> SharedReplicaSnapshot {
    Arc::new(Mutex::new(Box::new(snapshot)))
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
    /// Current process-local custody neighbors.
    ///
    /// Explicit bootstrap peers seed this set. A remote identity joins it only
    /// after presenting a valid REPLICATE proof for this node's exact replica
    /// set, so one bootstrap edge is enough for a connected team to converge
    /// without publishing a global roster.
    fn replica_peers(&self) -> Vec<PeerId>;
    /// Fetch one exact custody summary from one known process-local neighbor.
    fn replica_summary(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<ReplicaSummary>>;
    /// Fetch one bounded custody inventory page from a known process-local
    /// neighbor. Reachability uses normal iroh route selection; custody never
    /// consults collection gossip or the content-provider DHT.
    fn replica_page(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
        generation: ReplicaGeneration,
        component: ReplicaComponent,
        prefix: u8,
        after: Option<ReplicaItemId>,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<(Vec<ReplicaItem>, bool)>>;
    /// Fetch one exact custody blob through resumable bounded ranges.
    fn replica_blob(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
        generation: ReplicaGeneration,
        id: ReplicaItemId,
        expected_len: u64,
        receive_temp_dir: std::path::PathBuf,
    ) -> futures::future::BoxFuture<'static, Result<Option<anybytes::Bytes>, ReplicaBlobFetchError>>;
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

/// Why one identity is currently a custody routing neighbor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CustodyPeerAuthority {
    /// Explicit operator-selected bootstrap route.
    Bootstrap,
    /// Learned from an inbound REPLICATE proof. `None` is an unbounded proof;
    /// otherwise the TAI nanosecond is the inclusive effective upper bound.
    Learned(Option<i128>),
}

/// Process-local custody topology. This is routing state, not a durable roster:
/// configured bootstrap peers seed it and successful inbound REPLICATE
/// authorization adds the presenter only for the proof's effective lifetime.
type CustodyPeers = Arc<Mutex<BTreeMap<PeerId, CustodyPeerAuthority>>>;

fn epoch_tai_nanoseconds() -> i128 {
    crate::clock::epoch_now()
        .to_tai_duration()
        .total_nanoseconds()
}

fn custody_peer_is_current(authority: CustodyPeerAuthority, now: i128) -> bool {
    match authority {
        CustodyPeerAuthority::Bootstrap | CustodyPeerAuthority::Learned(None) => true,
        CustodyPeerAuthority::Learned(Some(upper)) => now <= upper,
    }
}

fn custody_peer_is_known(peers: &CustodyPeers, peer: PeerId) -> bool {
    let now = epoch_tai_nanoseconds();
    peers
        .lock()
        .unwrap()
        .get(&peer)
        .copied()
        .is_some_and(|authority| custody_peer_is_current(authority, now))
}

fn custody_peer_snapshot(peers: &CustodyPeers) -> Vec<PeerId> {
    let now = epoch_tai_nanoseconds();
    let mut peers = peers.lock().unwrap();
    peers.retain(|_, authority| custody_peer_is_current(*authority, now));
    peers.keys().copied().collect()
}

/// Intersect inclusive upper bounds. `None` denotes an unbounded interval.
fn intersect_upper_bounds(left: Option<i128>, right: Option<i128>) -> Option<i128> {
    match (left, right) {
        (None, None) => None,
        (Some(upper), None) | (None, Some(upper)) => Some(upper),
        (Some(left), Some(right)) => Some(left.min(right)),
    }
}

/// Union already-valid authority intervals. Successful introductions happen
/// at the current instant, so their intervals overlap there and maxing their
/// inclusive upper bounds is an exact union. `None` denotes unbounded.
fn union_upper_bounds(left: Option<i128>, right: Option<i128>) -> Option<i128> {
    match (left, right) {
        (None, _) | (_, None) => None,
        (Some(left), Some(right)) => Some(left.max(right)),
    }
}

fn note_authorized_custody_peer(
    peers: &CustodyPeers,
    peer: PeerId,
    connect_upper: Option<hifitime::Epoch>,
    replicate: &triblespace_core::capability::VerifiedCapability,
) {
    let connect_upper = connect_upper.map(|upper| upper.to_tai_duration().total_nanoseconds());
    let replicate_upper = replicate
        .effective_validity()
        .map(|validity| validity.bounds().1.to_tai_duration().total_nanoseconds());
    // The introduction is authorized only by the intersection of the
    // connection and operation grants. `None` means unbounded on that side.
    let learned = intersect_upper_bounds(connect_upper, replicate_upper);
    let mut peers = peers.lock().unwrap();
    match peers.entry(peer) {
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(CustodyPeerAuthority::Learned(learned));
        }
        std::collections::btree_map::Entry::Occupied(mut entry) => match *entry.get() {
            CustodyPeerAuthority::Bootstrap => {}
            CustodyPeerAuthority::Learned(existing) => {
                let union = union_upper_bounds(existing, learned);
                entry.insert(CustodyPeerAuthority::Learned(union));
            }
        },
    }
}

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
    connect_proof: triblespace_core::capability::CapabilityProofBundle,
    my_id: PeerId,
    /// Configured peers and live gossip neighbors — consulted before the DHT
    /// on every on-demand fetch. Membership is only a routing hint; callers
    /// still verify content hashes and fall back to DHT providers.
    candidates: RoutingCandidates,
    /// Explicit peers are the discovery boundary for input-only operation
    /// questions: their unknown result hashes cannot be looked up in the DHT.
    configured_peers: Vec<PeerId>,
    /// Bootstrap peers plus identities that authenticated for this exact
    /// custody replica set during this process lifetime.
    custody_peers: CustodyPeers,
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

    fn replica_peers(&self) -> Vec<PeerId> {
        custody_peer_snapshot(&self.custody_peers)
    }

    fn replica_summary(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<ReplicaSummary>> {
        let transport = self.transport.clone();
        let pool = self.pool.clone();
        let connect_proof = self.connect_proof.clone();
        let known = custody_peer_is_known(&self.custody_peers, peer) && peer != self.my_id;
        Box::pin(async move {
            if !known {
                anyhow::bail!("custody peer is not a known replication neighbor");
            }
            let mut last_error = None;
            for _ in 0..2 {
                let Some(connection) = pool_get(&transport, &pool, peer, &connect_proof).await
                else {
                    last_error = Some(anyhow::anyhow!("custody peer is unavailable"));
                    continue;
                };
                match tokio::time::timeout(
                    OP_DEADLINE,
                    crate::replica_wire::op_replica_summary(&connection, replica_set, &proof),
                )
                .await
                {
                    Ok(Ok(summary)) => return Ok(summary),
                    Ok(Err(error)) => last_error = Some(error),
                    Err(_) => {
                        last_error = Some(anyhow::anyhow!("custody summary deadline exceeded"));
                    }
                }
                pool_evict(&pool, peer).await;
            }
            Err(last_error.unwrap_or_else(|| anyhow::anyhow!("custody summary failed")))
        })
    }

    fn replica_page(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
        generation: ReplicaGeneration,
        component: ReplicaComponent,
        prefix: u8,
        after: Option<ReplicaItemId>,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<(Vec<ReplicaItem>, bool)>> {
        let transport = self.transport.clone();
        let pool = self.pool.clone();
        let connect_proof = self.connect_proof.clone();
        let known = custody_peer_is_known(&self.custody_peers, peer) && peer != self.my_id;
        Box::pin(async move {
            if !known {
                anyhow::bail!("custody peer is not a known replication neighbor");
            }
            let mut last_error = None;
            for _ in 0..2 {
                let Some(connection) = pool_get(&transport, &pool, peer, &connect_proof).await
                else {
                    last_error = Some(anyhow::anyhow!("custody peer is unavailable"));
                    continue;
                };
                match tokio::time::timeout(
                    OP_DEADLINE,
                    crate::replica_wire::op_replica_page(
                        &connection,
                        replica_set,
                        &proof,
                        generation,
                        component,
                        prefix,
                        after,
                    ),
                )
                .await
                {
                    Ok(Ok(page)) => return Ok(page),
                    Ok(Err(error)) => last_error = Some(error),
                    Err(_) => {
                        last_error = Some(anyhow::anyhow!("custody page deadline exceeded"));
                    }
                }
                pool_evict(&pool, peer).await;
            }
            Err(last_error.unwrap_or_else(|| anyhow::anyhow!("custody page failed")))
        })
    }

    fn replica_blob(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
        generation: ReplicaGeneration,
        id: ReplicaItemId,
        expected_len: u64,
        receive_temp_dir: std::path::PathBuf,
    ) -> futures::future::BoxFuture<'static, Result<Option<anybytes::Bytes>, ReplicaBlobFetchError>>
    {
        const RANGE_ATTEMPTS: usize = 3;

        let transport = self.transport.clone();
        let pool = self.pool.clone();
        let connect_proof = self.connect_proof.clone();
        let known = custody_peer_is_known(&self.custody_peers, peer) && peer != self.my_id;
        Box::pin(async move {
            if !known {
                return Err(ReplicaBlobFetchError::local(anyhow::anyhow!(
                    "custody peer is not a known replication neighbor"
                )));
            }
            let len = usize::try_from(expected_len).map_err(|_| {
                ReplicaBlobFetchError::local(anyhow::anyhow!(
                    "replica blob does not fit this address space"
                ))
            })?;
            if len == 0 {
                let bytes = anybytes::Bytes::empty();
                if blake3::hash(&bytes).as_bytes() != &id.0 {
                    return Err(ReplicaBlobFetchError::remote(anyhow::anyhow!(
                        "empty blob inventory carries a nonempty content hash"
                    )));
                }
                return Ok(Some(bytes));
            }

            let mut temporary = tempfile::tempfile_in(&receive_temp_dir).map_err(|error| {
                ReplicaBlobFetchError::local(anyhow::anyhow!(
                    "create replica receive file in {}: {error}",
                    receive_temp_dir.display()
                ))
            })?;
            let mut hasher = blake3::Hasher::new();
            let mut offset = 0usize;
            let mut chunk = Vec::new();
            chunk
                .try_reserve_exact(crate::replica_wire::BLOB_TRANSFER_CHUNK_BYTES)
                .map_err(|error| {
                    ReplicaBlobFetchError::local(anyhow::anyhow!(
                        "allocate replica range buffer: {error}"
                    ))
                })?;
            chunk.resize(crate::replica_wire::BLOB_TRANSFER_CHUNK_BYTES, 0);
            while offset < len {
                let chunk_len = (len - offset).min(crate::replica_wire::BLOB_TRANSFER_CHUNK_BYTES);
                let target = &mut chunk[..chunk_len];
                let mut received = None;
                let mut last_error = None;
                for _ in 0..RANGE_ATTEMPTS {
                    let Some(connection) = pool_get(&transport, &pool, peer, &connect_proof).await
                    else {
                        last_error = Some(anyhow::anyhow!("custody peer is unavailable"));
                        continue;
                    };
                    let operation = crate::replica_wire::op_replica_blob_range(
                        &connection,
                        replica_set,
                        &proof,
                        generation,
                        id,
                        expected_len,
                        offset as u64,
                        target,
                    );
                    match tokio::time::timeout(OP_DEADLINE, operation).await {
                        Ok(Ok(Some(count))) => {
                            received = Some(count);
                            break;
                        }
                        Ok(Ok(None)) => return Ok(None),
                        Ok(Err(error)) => last_error = Some(error),
                        Err(_) => {
                            last_error =
                                Some(anyhow::anyhow!("custody blob range deadline exceeded"));
                        }
                    }
                    pool_evict(&pool, peer).await;
                }
                let count = received
                    .ok_or_else(|| {
                        last_error.unwrap_or_else(|| anyhow::anyhow!("custody blob range failed"))
                    })
                    .map_err(ReplicaBlobFetchError::remote)?;
                if count != chunk_len {
                    return Err(ReplicaBlobFetchError::remote(anyhow::anyhow!(
                        "custody blob range returned a short chunk"
                    )));
                }
                hasher.update(target);
                std::io::Write::write_all(&mut temporary, target).map_err(|error| {
                    ReplicaBlobFetchError::local(anyhow::anyhow!(
                        "write replica receive file: {error}"
                    ))
                })?;
                offset += count;
            }
            if hasher.finalize().as_bytes() != &id.0 {
                return Err(ReplicaBlobFetchError::remote(anyhow::anyhow!(
                    "replica blob content does not match requested handle"
                )));
            }
            let file = temporary;
            let actual_len = file
                .metadata()
                .map_err(|error| {
                    ReplicaBlobFetchError::local(anyhow::anyhow!(
                        "inspect replica receive file: {error}"
                    ))
                })?
                .len();
            if actual_len != expected_len {
                return Err(ReplicaBlobFetchError::local(anyhow::anyhow!(
                    "replica receive file length is {actual_len}; expected {expected_len}"
                )));
            }
            // Mapping does not make data durable; durability remains the
            // destination store's explicit flush policy after insertion.
            let mapping = unsafe {
                memmap2::MmapOptions::new()
                    .len(len)
                    .map(&file)
                    .map_err(|error| {
                        ReplicaBlobFetchError::local(anyhow::anyhow!(
                            "map replica receive file: {error}"
                        ))
                    })?
            };
            Ok(Some(anybytes::Bytes::from_source(mapping)))
        })
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
    replica_snapshot: ReplicaSnapshotSlot,
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

    /// Replace the last explicitly built custody inventory.
    ///
    /// Ordinary Peer refreshes never call this path. A custody driver retains
    /// the last complete immutable view until a later explicit rebuild
    /// succeeds. Peers that already fetched a summary remain pinned to the
    /// older immutable generation until their next summary request.
    pub(crate) fn update_replica_snapshot(&self, snapshot: impl AnyReplicaSnapshot) {
        *self.replica_snapshot.lock().unwrap() = Some(shared_replica_snapshot(snapshot));
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

    async fn ready_capability(&self) -> anyhow::Result<Arc<dyn NetCapability>> {
        let mut receiver = self.cap.clone();
        let guard = receiver
            .wait_for(|capability| capability.is_some())
            .await
            .map_err(|_| anyhow::anyhow!("network host stopped before becoming ready"))?;
        guard
            .clone()
            .ok_or_else(|| anyhow::anyhow!("network host did not publish its capability"))
    }

    /// Snapshot the currently known custody neighbors in deterministic order.
    pub(crate) async fn replica_peers(&self) -> anyhow::Result<Vec<PeerId>> {
        Ok(self.ready_capability().await?.replica_peers())
    }

    pub(crate) async fn replica_summary(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
    ) -> anyhow::Result<ReplicaSummary> {
        self.ready_capability()
            .await?
            .replica_summary(peer, replica_set, proof)
            .await
    }

    pub(crate) async fn replica_page(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
        generation: ReplicaGeneration,
        component: ReplicaComponent,
        prefix: u8,
        after: Option<ReplicaItemId>,
    ) -> anyhow::Result<(Vec<ReplicaItem>, bool)> {
        self.ready_capability()
            .await?
            .replica_page(
                peer,
                replica_set,
                proof,
                generation,
                component,
                prefix,
                after,
            )
            .await
    }

    pub(crate) async fn replica_blob(
        &self,
        peer: PeerId,
        replica_set: crate::replica::ReplicaSetId,
        proof: triblespace_core::capability::CapabilityProofBundle,
        generation: ReplicaGeneration,
        id: ReplicaItemId,
        expected_len: u64,
        receive_temp_dir: std::path::PathBuf,
    ) -> Result<Option<anybytes::Bytes>, ReplicaBlobFetchError> {
        let capability = self
            .ready_capability()
            .await
            .map_err(ReplicaBlobFetchError::local)?;
        capability
            .replica_blob(
                peer,
                replica_set,
                proof,
                generation,
                id,
                expected_len,
                receive_temp_dir,
            )
            .await
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
    pub(crate) replica_snapshot: ReplicaSnapshotSlot,
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
    let replica_snapshot: ReplicaSnapshotSlot = Arc::new(Mutex::new(None));
    let (cap_tx, cap_rx) = tokio::sync::watch::channel::<Option<Arc<dyn NetCapability>>>(None);

    let sender = NetSender {
        cmd_tx,
        snapshot: snapshot.clone(),
        replica_snapshot: replica_snapshot.clone(),
        cap: cap_rx,
        id,
    };
    let receiver = NetReceiver { evt_rx };
    let wiring = HostWiring {
        cmd_rx,
        evt_tx,
        snapshot,
        replica_snapshot,
        cap_tx,
    };
    (sender, receiver, wiring)
}

/// Run the host loop over an already-constructed transport harness.
/// This is the transport-generic entry point: production wraps it in
/// a dedicated thread ([`spawn`]); the simulator spawns it as a local
/// task per node on one shared deterministic runtime.
pub async fn run_host<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    run_host_with_replica(harness, config, wiring, None).await;
}

/// Run the ordinary host with an additional, independently authorized custody
/// service. Passing `None` is exactly the public-gossip host behavior.
pub(crate) async fn run_host_with_replica<T: Transport>(
    harness: Harness<T>,
    config: PeerConfig,
    wiring: HostWiring,
    replica_server: Option<ReplicaServerConfig>,
) {
    host_loop(
        harness,
        config,
        wiring.cmd_rx,
        wiring.evt_tx,
        wiring.snapshot,
        wiring.replica_snapshot,
        wiring.cap_tx,
        replica_server,
    )
    .await;
}

/// Run the proof-gated custody service over a caller-supplied deterministic
/// transport harness.
///
/// This exists only with the `sim` feature; production uses the ordinary iroh
/// reachability binder owned by [`crate::replica::CustodyReplica::new`].
#[cfg(feature = "sim")]
pub async fn run_custody_host<T: Transport>(
    harness: Harness<T>,
    config: PeerConfig,
    wiring: HostWiring,
    replica_server: ReplicaServerConfig,
) {
    run_host_with_replica(harness, config, wiring, Some(replica_server)).await;
}

/// Spawn the network thread. Returns the outgoing/incoming channel halves
/// — used internally by [`Peer::new`](crate::peer::Peer::new).
pub fn spawn(key: SigningKey, config: PeerConfig) -> (NetSender, NetReceiver) {
    spawn_with_replica(key, config, None)
}

/// Owned production host thread for a custody replica.
///
/// The custody facade retains this handle so it can stop serving immutable
/// readers and join the runtime before returning the underlying store.
pub(crate) struct CustodyHostThread {
    join: Option<thread::JoinHandle<()>>,
}

impl CustodyHostThread {
    pub(crate) fn join(mut self) -> anyhow::Result<()> {
        let Some(join) = self.join.take() else {
            return Ok(());
        };
        join.join()
            .map_err(|_| anyhow::anyhow!("custody host thread panicked"))
    }
}

/// Spawn a host whose custody operations are gated by a second exact action.
pub(crate) fn spawn_with_replica(
    key: SigningKey,
    config: PeerConfig,
    replica_server: Option<ReplicaServerConfig>,
) -> (NetSender, NetReceiver) {
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
            run_host_with_replica(harness, config, wiring, replica_server).await;
        });
    });

    (sender, receiver)
}

/// Spawn custody's pile-sync-only protocol over ordinary iroh reachability.
pub(crate) fn spawn_custody(
    key: SigningKey,
    config: PeerConfig,
    replica_server: ReplicaServerConfig,
) -> anyhow::Result<(NetSender, NetReceiver, CustodyHostThread, EndpointAddr)> {
    let secret = iroh_secret(&key);
    let id: EndpointId = secret.public().into();
    let (sender, receiver, wiring) = wire(id);
    let (ready_tx, ready_rx) = mpsc::sync_channel::<anyhow::Result<EndpointAddr>>(1);
    let join = thread::Builder::new()
        .name("triblespace-custody".to_owned())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready_tx.send(Err(anyhow::Error::new(error)));
                    return;
                }
            };
            let harness =
                match runtime.block_on(crate::transport::iroh::bind_custody(secret, &config)) {
                    Ok(harness) => harness,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
            // bind_custody returns after the endpoint and pile-sync router are
            // installed. Relay selection and address discovery may continue to
            // enrich these route hints after startup.
            let endpoint_addr = harness.transport.endpoint_addr();
            if ready_tx.send(Ok(endpoint_addr)).is_err() {
                return;
            }
            runtime.block_on(run_host_with_replica(
                harness,
                config,
                wiring,
                Some(replica_server),
            ));
        })
        .map_err(|error| anyhow::anyhow!("spawn custody host thread: {error}"))?;

    match ready_rx.recv() {
        Ok(Ok(endpoint_addr)) => Ok((
            sender,
            receiver,
            CustodyHostThread { join: Some(join) },
            endpoint_addr,
        )),
        Ok(Err(error)) => {
            let _ = join.join();
            Err(error)
        }
        Err(error) => {
            let _ = join.join();
            Err(anyhow::anyhow!(
                "custody host exited before startup acknowledgement: {error}"
            ))
        }
    }
}

// ── Network thread event loop ────────────────────────────────────────

/// Deadline for establishing + authenticating a connection (the `pool_get`
/// init future: dial + inline CONNECT proof-bundle round trip). A connection
/// attempt that exceeds this counts as failed: the pool's
/// singleflight cell resets so the next walk re-dials, instead of
/// every later fetch to that peer queueing forever behind one
/// stalled authentication exchange. Generous relative to real-world QUIC + relay
/// setup times; deterministic under simulated virtual time.
const DIAL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// Deadline for an accepted connection to open its first stream and complete
/// the whole OP_AUTH exchange. This bounds unauthenticated connection state
/// even when a peer opens no stream or dribbles an incomplete proof bundle.
const INBOUND_AUTH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
/// Total accepted connections whose authentication/serve loop may exist at
/// once. Admission happens before production spawns a connection task.
pub(crate) const MAX_INBOUND_CONNECTIONS_GLOBAL: usize = 64;
/// An authenticated connection that opens no new request stream for this long
/// is closed. This is independent of transport keepalive.
const INBOUND_CONNECTION_IDLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);

/// Maximum post-CONNECT request streams one connection may execute at once.
/// A peer exceeding it is disconnected rather than accumulating application
/// tasks behind transport-level stream queues.
pub(crate) const MAX_INBOUND_REQUESTS_PER_CONNECTION: usize = 16;
/// Node-wide post-CONNECT request bound, shared by every connection handler.
/// The per-connection limit prevents one authorized key from monopolizing it.
const MAX_INBOUND_REQUESTS_GLOBAL: usize = 64;
/// End-to-end inbound request budget: operation byte, complete request/proof,
/// snapshot work, response body, and response shutdown.
const INBOUND_REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Deadline for a single protocol op (OP_CHILDREN / OP_GET_BLOB request + full
/// response) on an established connection. On expiry the op reports an error
/// and the caller's existing evict-and-try-next-provider path takes over.
/// Total-op rather than progress-based; large-content streaming may eventually
/// warrant an idle deadline instead.
const OP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Connect to a peer over the pile-sync ALPN and immediately present
/// our complete CONNECT proof bundle so subsequent direct RPCs are admitted.
#[instrument(level = "info", skip(t, connect_proof), fields(peer = %hex::encode(&peer[..4])))]
async fn connect_authed<T: Transport>(
    t: &T,
    peer: PeerId,
    connect_proof: &triblespace_core::capability::CapabilityProofBundle,
) -> anyhow::Result<T::Conn> {
    let conn = t.dial(peer, PILE_SYNC_ALPN).await.map_err(|e| {
        warn!(error = %e, "connect failed");
        anyhow::anyhow!("connect: {e}")
    })?;
    debug!(
        steps = connect_proof.proof().step_count(),
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
    replica_snapshot: ReplicaSnapshotSlot,
    cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
    replica_server: Option<ReplicaServerConfig>,
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
    let custody_peers: CustodyPeers = Arc::new(Mutex::new(
        configured_peers
            .iter()
            .copied()
            .map(|peer| (peer, CustodyPeerAuthority::Bootstrap))
            .collect(),
    ));

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
        custody_peers: custody_peers.clone(),
    }) as Arc<dyn NetCapability>));

    // ── Inbound connections. Each connection gets its own task and
    // accepts sequential bi-streams until the peer closes.
    let snapshot_handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        replica_snapshot,
        replica_snapshot_generations: Arc::new(Mutex::new(ReplicaSnapshotCache::default())),
        custody_peers,
        connect_root: config.connect_root,
        replica_server,
        inbound_connections: Arc::new(tokio::sync::Semaphore::new(MAX_INBOUND_CONNECTIONS_GLOBAL)),
        inbound_requests: Arc::new(tokio::sync::Semaphore::new(MAX_INBOUND_REQUESTS_GLOBAL)),
    };
    let mut incoming = incoming;
    tokio::spawn(async move {
        while let Some(inc) = incoming.recv().await {
            if inc.alpn == PILE_SYNC_ALPN {
                let h = snapshot_handler.clone();
                let Some(permit) = h.try_admit_connection() else {
                    warn!(
                        limit = MAX_INBOUND_CONNECTIONS_GLOBAL,
                        "inbound connection limit exceeded; rejecting connection"
                    );
                    inc.conn.close(1, b"inbound connection limit exceeded");
                    continue;
                };
                tokio::spawn(async move { h.handle_admitted::<T>(inc.conn, permit).await });
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
        let commands_disconnected = loop {
            let cmd = match commands.try_recv() {
                Ok(cmd) => cmd,
                Err(mpsc::TryRecvError::Empty) => break false,
                Err(mpsc::TryRecvError::Disconnected) => break true,
            };
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
        };
        if commands_disconnected {
            // All owning senders are gone. Production custody shutdown joins
            // this loop before closing the store, so no serving snapshot can
            // outlive its storage owner.
            let _ = cap_tx.send(None);
            break;
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

    transport.shutdown().await;
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
/// The handler admits a bounded number of request streams only after the
/// first OP_AUTH stream succeeds. So one connection per peer is enough.
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
    connect_proof: &triblespace_core::capability::CapabilityProofBundle,
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
    connect_proof: &triblespace_core::capability::CapabilityProofBundle,
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
    connect_proof: &triblespace_core::capability::CapabilityProofBundle,
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
    replica_snapshot: ReplicaSnapshotSlot,
    /// Recently summarized immutable generations. This is a bounded retry
    /// cache: eviction makes a walker request a fresh summary, never splice
    /// two snapshots together.
    replica_snapshot_generations: ReplicaSnapshotGenerations,
    /// Process-local bootstrap graph, extended only after an exact REPLICATE
    /// operation has been authorized and completely framed.
    custody_peers: CustodyPeers,
    connect_root: ed25519_dalek::VerifyingKey,
    replica_server: Option<ReplicaServerConfig>,
    /// Shared before-spawn connection admission budget for this node.
    inbound_connections: Arc<tokio::sync::Semaphore>,
    /// Shared across every cloned connection handler for this node.
    inbound_requests: Arc<tokio::sync::Semaphore>,
}

impl SnapshotHandler {
    fn try_admit_connection(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.inbound_connections.clone().try_acquire_owned().ok()
    }

    fn try_admit_request(
        &self,
        connection_requests: &Arc<tokio::sync::Semaphore>,
    ) -> Option<(
        tokio::sync::OwnedSemaphorePermit,
        tokio::sync::OwnedSemaphorePermit,
    )> {
        let connection = connection_requests.clone().try_acquire_owned().ok()?;
        let global = match self.inbound_requests.clone().try_acquire_owned() {
            Ok(global) => global,
            Err(_) => {
                drop(connection);
                return None;
            }
        };
        Some((connection, global))
    }

    #[cfg(all(test, feature = "sim"))]
    async fn handle<T: Transport>(&self, connection: T::Conn) {
        let Some(permit) = self.try_admit_connection() else {
            connection.close(1, b"inbound connection limit exceeded");
            return;
        };
        self.handle_admitted::<T>(connection, permit).await;
    }

    async fn handle_admitted<T: Transport>(
        &self,
        connection: T::Conn,
        _connection_permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let snapshot = self.snapshot.clone();
        let replica_snapshot = self.replica_snapshot.clone();
        let replica_snapshot_generations = self.replica_snapshot_generations.clone();
        let custody_peers = self.custody_peers.clone();
        let connect_root = self.connect_root;
        let replica_server = self.replica_server;
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
            // One deadline covers both an absent first stream and a partial
            // auth body, including its required EOF/trailing-byte check.
            let authenticated = tokio::time::timeout(INBOUND_AUTH_DEADLINE, async {
                let Some((mut send, mut recv)) = connection.accept_bi().await else {
                    return None;
                };
                let authenticated =
                    authenticate_connection::<T::Conn>(connect_root, peer, &mut send, &mut recv)
                        .await;
                let _ = send.shutdown().await;
                Some(authenticated)
            })
            .await;
            let expires = match authenticated {
                Err(_) => {
                    warn!(
                        deadline = ?INBOUND_AUTH_DEADLINE,
                        "inbound authentication deadline exceeded"
                    );
                    connection.close(1, b"authentication deadline exceeded");
                    return;
                }
                Ok(None) => {
                    debug!("connection ended before OP_AUTH");
                    return;
                }
                Ok(Some(Ok(Some(verified)))) => verified
                    .effective_validity()
                    .map(|validity| validity.bounds().1),
                Ok(Some(Ok(None))) => {
                    connection.close(1, b"CONNECT capability required");
                    return;
                }
                Ok(Some(Err(error))) => {
                    warn!(%error, "authentication stream failed");
                    connection.close(1, b"malformed authentication");
                    return;
                }
            };

            let expiry = wait_until_after(expires);
            tokio::pin!(expiry);
            let connection_requests = Arc::new(tokio::sync::Semaphore::new(
                MAX_INBOUND_REQUESTS_PER_CONNECTION,
            ));

            loop {
                let idle = tokio::time::sleep(INBOUND_CONNECTION_IDLE_DEADLINE);
                tokio::pin!(idle);
                let stream = tokio::select! {
                    stream = connection.accept_bi() => stream,
                    () = &mut expiry => {
                        info!("CONNECT capability expired; closing connection");
                        connection.close(1, b"CONNECT capability expired");
                        return;
                    }
                    () = &mut idle => {
                        info!(
                            deadline = ?INBOUND_CONNECTION_IDLE_DEADLINE,
                            "authenticated connection idle deadline exceeded"
                        );
                        connection.close(0, b"authenticated connection idle timeout");
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
                let Some((_connection_permit, _global_permit)) =
                    self.try_admit_request(&connection_requests)
                else {
                    warn!(
                        per_connection = MAX_INBOUND_REQUESTS_PER_CONNECTION,
                        global = MAX_INBOUND_REQUESTS_GLOBAL,
                        "inbound request concurrency limit exceeded; closing connection"
                    );
                    connection.close(1, b"inbound request concurrency limit exceeded");
                    return;
                };
                let snapshot = snapshot.clone();
                let replica_snapshot = replica_snapshot.clone();
                let replica_snapshot_generations = replica_snapshot_generations.clone();
                let custody_peers = custody_peers.clone();
                tokio::spawn(
                    async move {
                        let operation = tokio::time::timeout(INBOUND_REQUEST_DEADLINE, async {
                            let result = serve_stream::<T::Conn>(
                                &snapshot,
                                &replica_snapshot,
                                &replica_snapshot_generations,
                                &custody_peers,
                                replica_server,
                                expires,
                                peer,
                                &mut send,
                                &mut recv,
                            )
                            .await;
                            let shutdown = send.shutdown().await.map_err(|error| {
                                anyhow::anyhow!("finish response stream: {error}")
                            });
                            result.and(shutdown)
                        })
                        .await;
                        match operation {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => debug!(%error, "direct RPC stream failed"),
                            Err(_) => warn!(
                                deadline = ?INBOUND_REQUEST_DEADLINE,
                                "inbound request deadline exceeded"
                            ),
                        }
                        drop((_connection_permit, _global_permit));
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
    use triblespace_core::capability::{CapabilityMode, CapabilityRequest};

    let verdict = async {
        let op = recv_u8(recv).await?;
        if op != OP_AUTH {
            anyhow::bail!(
                "first stream operation is {}, expected OP_AUTH",
                op_name(op)
            );
        }
        let bundle = recv_capability_proof_bundle(recv).await?;
        let mut trailing = [0u8; 1];
        if recv.read(&mut trailing).await? != 0 {
            anyhow::bail!("OP_AUTH contains trailing bytes");
        }
        let verified = bundle.verify(
            connect_root,
            crate::clock::epoch_now(),
            peer,
            CapabilityRequest::new(
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
    replica_snapshot: &ReplicaSnapshotSlot,
    replica_snapshot_generations: &ReplicaSnapshotGenerations,
    custody_peers: &CustodyPeers,
    replica_server: Option<ReplicaServerConfig>,
    connect_upper: Option<hifitime::Epoch>,
    peer: ed25519_dalek::VerifyingKey,
    send: &mut C::SendHalf,
    recv: &mut C::RecvHalf,
) -> anyhow::Result<()> {
    let op = recv_u8(recv).await?;
    if replica_server.is_some()
        && !matches!(op, OP_REPLICA_SUMMARY | OP_REPLICA_PAGE | OP_REPLICA_BLOB)
    {
        anyhow::bail!(
            "ordinary {} is disabled on a custody-only endpoint",
            op_name(op)
        );
    }
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
        OP_REPLICA_SUMMARY => {
            let (replica_set, proof) = recv_request_prefix(recv).await?;
            let verified = authorize_replica_operation(replica_server, replica_set, &proof, peer)?;
            crate::replica_wire::require_eof(recv).await?;
            // A valid summary request is the authenticated introduction. This
            // routing hint is deliberately process-local: authority stays in
            // the proof and no globally enumerable membership ledger appears.
            note_authorized_custody_peer(custody_peers, peer.to_bytes(), connect_upper, &verified);
            let pinned = replica_snapshot
                .lock()
                .unwrap()
                .as_ref()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("custody snapshot is unavailable"))?;
            let summary = pinned.lock().unwrap().summary();
            let generation = summary.generation();
            replica_snapshot_generations
                .lock()
                .unwrap()
                .insert(generation, pinned);
            send_summary(send, &summary).await?;
        }
        OP_REPLICA_PAGE => {
            let (replica_set, proof) = recv_request_prefix(recv).await?;
            authorize_replica_operation(replica_server, replica_set, &proof, peer)?;
            let (generation, component, prefix, after) = recv_page_request(recv).await?;
            let pinned = replica_snapshot_for_generation(replica_snapshot_generations, generation)?;
            let (page, done) = pinned.lock().unwrap().page(component, prefix, after);
            drop(pinned);
            send_page(send, component, &page, done).await?;
        }
        OP_REPLICA_BLOB => {
            let (replica_set, proof) = recv_request_prefix(recv).await?;
            authorize_replica_operation(replica_server, replica_set, &proof, peer)?;
            let (generation, id, offset, maximum) = recv_blob_request(recv).await?;
            let pinned = replica_snapshot_for_generation(replica_snapshot_generations, generation)?;
            let bytes = pinned.lock().unwrap().blob_bytes(id);
            drop(pinned);
            send_blob_range(send, bytes.as_ref(), offset, maximum).await?;
        }
        OP_AUTH => anyhow::bail!("OP_AUTH may only appear on the first stream"),
        _ => anyhow::bail!("unknown direct RPC operation {op:#x}"),
    }
    Ok(())
}

fn replica_snapshot_for_generation(
    generations: &ReplicaSnapshotGenerations,
    requested: ReplicaGeneration,
) -> anyhow::Result<SharedReplicaSnapshot> {
    generations.lock().unwrap().get(requested).ok_or_else(|| {
        anyhow::anyhow!(
            "custody snapshot generation is stale or unavailable; request a new summary"
        )
    })
}

fn authorize_replica_operation(
    server: Option<ReplicaServerConfig>,
    requested_set: crate::replica::ReplicaSetId,
    proof: &triblespace_core::capability::CapabilityProofBundle,
    peer: ed25519_dalek::VerifyingKey,
) -> anyhow::Result<triblespace_core::capability::VerifiedCapability> {
    use triblespace_core::capability::{CapabilityMode, CapabilityRequest};

    let server = server.ok_or_else(|| anyhow::anyhow!("custody replication is disabled"))?;
    if requested_set != server.replica_set {
        anyhow::bail!("custody replica-set resource does not match this endpoint");
    }
    let verified = proof.verify(
        server.trust_root,
        crate::clock::epoch_now(),
        peer,
        CapabilityRequest::new(
            crate::replica::replicate_capability_atom(requested_set),
            CapabilityMode::Invoke,
        ),
    )?;
    if verified.effective_mode() != CapabilityMode::Invoke {
        anyhow::bail!("custody operation proof must end in exact invoke-only authority");
    }
    Ok(verified)
}

#[cfg(test)]
mod custody_state_tests {
    use super::{
        AnyReplicaSnapshot, MAX_INBOUND_CONNECTIONS_GLOBAL, ReplicaComponent, ReplicaGeneration,
        ReplicaItem, ReplicaItemId, ReplicaSnapshotCache, ReplicaSummary, intersect_upper_bounds,
        shared_replica_snapshot, union_upper_bounds,
    };

    struct EmptyReplicaSnapshot;

    impl AnyReplicaSnapshot for EmptyReplicaSnapshot {
        fn summary(&self) -> ReplicaSummary {
            ReplicaSummary::from_buckets([[Default::default(); 256]; 3])
        }

        fn page(
            &self,
            _component: ReplicaComponent,
            _prefix: u8,
            _after: Option<ReplicaItemId>,
        ) -> (Vec<ReplicaItem>, bool) {
            (Vec::new(), true)
        }

        fn blob_bytes(&self, _id: ReplicaItemId) -> Option<anybytes::Bytes> {
            None
        }
    }

    fn generation(byte: u8) -> ReplicaGeneration {
        ReplicaGeneration::new([byte; 32])
    }

    #[test]
    fn custody_authority_bounds_intersect_then_union() {
        assert_eq!(intersect_upper_bounds(None, None), None);
        assert_eq!(intersect_upper_bounds(Some(7), None), Some(7));
        assert_eq!(intersect_upper_bounds(None, Some(9)), Some(9));
        assert_eq!(intersect_upper_bounds(Some(7), Some(9)), Some(7));

        assert_eq!(union_upper_bounds(Some(7), Some(9)), Some(9));
        assert_eq!(union_upper_bounds(None, Some(9)), None);
        assert_eq!(union_upper_bounds(Some(7), None), None);
    }

    #[test]
    fn generation_cache_deduplicates_and_evicts_least_recently_used() {
        assert!(MAX_INBOUND_CONNECTIONS_GLOBAL < u8::MAX as usize);
        let snapshot = shared_replica_snapshot(EmptyReplicaSnapshot);
        let mut cache = ReplicaSnapshotCache::default();
        for index in 0..MAX_INBOUND_CONNECTIONS_GLOBAL {
            cache.insert(generation(index as u8), snapshot.clone());
        }
        assert_eq!(cache.snapshots.len(), MAX_INBOUND_CONNECTIONS_GLOBAL);

        // Touch the oldest generation, making generation 1 the eviction
        // candidate, then introduce one distinct generation beyond the cap.
        assert!(cache.get(generation(0)).is_some());
        let newest = generation(MAX_INBOUND_CONNECTIONS_GLOBAL as u8);
        cache.insert(newest, snapshot.clone());
        assert_eq!(cache.snapshots.len(), MAX_INBOUND_CONNECTIONS_GLOBAL);
        assert!(cache.snapshots.contains_key(&generation(0)));
        assert!(!cache.snapshots.contains_key(&generation(1)));
        assert!(cache.snapshots.contains_key(&newest));

        // Re-summarizing one generation refreshes it without retaining a
        // second allocation or consuming another cache slot.
        cache.insert(newest, snapshot);
        assert_eq!(cache.snapshots.len(), MAX_INBOUND_CONNECTIONS_GLOBAL);
        assert_eq!(
            cache
                .least_to_most_recent
                .iter()
                .filter(|candidate| **candidate == newest)
                .count(),
            1
        );
    }
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

#[cfg(all(test, feature = "sim"))]
mod inbound_auth_deadline_tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anybytes::Bytes;
    use ed25519_dalek::SigningKey;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use triblespace_core::blob::{Blob, encodings::UnknownBlob};
    use triblespace_core::capability::{CapabilityClaim, CapabilityMode, CapabilityProofBundle};
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::repo::memoryrepo::MemoryRepo;

    use super::{
        AnySnapshot, INBOUND_AUTH_DEADLINE, INBOUND_CONNECTION_IDLE_DEADLINE,
        INBOUND_REQUEST_DEADLINE, MAX_INBOUND_CONNECTIONS_GLOBAL, MAX_INBOUND_REQUESTS_GLOBAL,
        MAX_INBOUND_REQUESTS_PER_CONNECTION, SnapshotHandler, shared_replica_snapshot,
    };
    use crate::protocol::{
        OP_GET_BLOB, PILE_SYNC_ALPN, connect_capability_atom, op_auth, op_get_blob, send_u8,
    };
    use crate::replica::{
        ReplicaComponent, ReplicaGeneration, ReplicaItemId, ReplicaServerConfig, ReplicaSetId,
        replicate_capability_atom, snapshot_from_store,
    };
    use crate::replica_wire::{op_replica_blob_range, op_replica_page, op_replica_summary};
    use crate::transport::sim::{SimConfig, SimNet, SimTransport};
    use crate::transport::{Conn, Transport};

    fn connect_proof(root: &SigningKey, leaf: &SigningKey) -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            root,
            CapabilityClaim::root(
                connect_capability_atom(root.verifying_key()),
                CapabilityMode::Invoke,
                None,
            ),
            leaf.verifying_key(),
        )
        .unwrap()
    }

    fn replica_proof(
        root: &SigningKey,
        leaf: &SigningKey,
        replica_set: ReplicaSetId,
    ) -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            root,
            CapabilityClaim::root(
                replicate_capability_atom(replica_set),
                CapabilityMode::Invoke,
                None,
            ),
            leaf.verifying_key(),
        )
        .unwrap()
    }

    fn blob_payloads_with_prefix(prefix: u8, count: usize, mut nonce: u64) -> Vec<Vec<u8>> {
        let mut payloads = Vec::with_capacity(count);
        while payloads.len() < count {
            let payload = nonce.to_be_bytes().to_vec();
            let blob = Blob::<UnknownBlob>::new(Bytes::from(payload.clone()));
            if blob.get_handle().raw[0] == prefix {
                payloads.push(payload);
            }
            nonce = nonce.checked_add(1).expect("test blob nonce overflow");
        }
        payloads
    }

    fn handler(root: &SigningKey) -> SnapshotHandler {
        let snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> = Arc::new(Mutex::new(None));
        let replica_snapshot = Arc::new(Mutex::new(None));
        SnapshotHandler {
            snapshot,
            replica_snapshot,
            replica_snapshot_generations: Arc::new(Mutex::new(Default::default())),
            custody_peers: Arc::new(Mutex::new(Default::default())),
            connect_root: root.verifying_key(),
            replica_server: None,
            inbound_connections: Arc::new(tokio::sync::Semaphore::new(
                MAX_INBOUND_CONNECTIONS_GLOBAL,
            )),
            inbound_requests: Arc::new(tokio::sync::Semaphore::new(MAX_INBOUND_REQUESTS_GLOBAL)),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn inbound_connection_that_opens_no_auth_stream_hits_deadline() {
        let config = SimConfig {
            latency: Duration::ZERO..Duration::from_nanos(1),
            ..SimConfig::default()
        };
        let net = SimNet::new(0xA117, config);
        let server_key = SigningKey::from_bytes(&[0xA1; 32]);
        let client_key = SigningKey::from_bytes(&[0xA2; 32]);
        let root_key = SigningKey::from_bytes(&[0xA3; 32]);
        let server_id = server_key.verifying_key().to_bytes();
        let client_id = client_key.verifying_key().to_bytes();
        let mut server = net.join(server_id, None);
        let client = net.join(client_id, None);

        let client_connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        assert_eq!(incoming.alpn, PILE_SYNC_ALPN);

        let handler = handler(&root_key);
        let task = tokio::spawn(async move {
            handler.handle::<SimTransport>(incoming.conn).await;
        });
        tokio::task::yield_now().await;

        tokio::time::advance(INBOUND_AUTH_DEADLINE - Duration::from_nanos(1)).await;
        tokio::task::yield_now().await;
        assert!(
            !task.is_finished(),
            "an idle inbound connection remains admitted until its auth deadline"
        );

        tokio::time::advance(Duration::from_nanos(1)).await;
        task.await.unwrap();
        assert!(
            client_connection.open_bi().await.is_err(),
            "the auth deadline closes the unauthenticated connection"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn partial_post_connect_request_is_dropped_at_end_to_end_deadline() {
        let net = SimNet::new(
            0xA118,
            SimConfig {
                latency: Duration::ZERO..Duration::from_nanos(1),
                ..SimConfig::default()
            },
        );
        let server_key = SigningKey::from_bytes(&[0xB1; 32]);
        let client_key = SigningKey::from_bytes(&[0xB2; 32]);
        let root_key = SigningKey::from_bytes(&[0xB3; 32]);
        let server_id = server_key.verifying_key().to_bytes();
        let client_id = client_key.verifying_key().to_bytes();
        let mut server = net.join(server_id, None);
        let client = net.join(client_id, None);
        let connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        let server_handler = handler(&root_key);
        let task = tokio::spawn(async move {
            server_handler.handle::<SimTransport>(incoming.conn).await;
        });
        op_auth(&connection, &connect_proof(&root_key, &client_key))
            .await
            .unwrap();

        let (mut send, mut recv) = connection.open_bi().await.unwrap();
        send_u8(&mut send, OP_GET_BLOB).await.unwrap();
        tokio::task::yield_now().await;
        tokio::time::advance(INBOUND_REQUEST_DEADLINE - Duration::from_nanos(1)).await;
        tokio::task::yield_now().await;
        let mut byte = [0u8; 1];
        let mut read = Box::pin(recv.read(&mut byte));
        assert!(
            matches!(futures::poll!(read.as_mut()), std::task::Poll::Pending),
            "partial request ended before its deadline"
        );
        drop(read);

        tokio::time::advance(Duration::from_nanos(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(recv.read(&mut byte).await.unwrap(), 0);
        assert_eq!(
            op_get_blob(&connection, &[0x77; 32]).await.unwrap(),
            None,
            "timing out one stream must release its permits without killing the connection"
        );
        connection.close(0, b"test complete");
        task.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn per_connection_request_cap_closes_an_overcommitting_peer() {
        let net = SimNet::new(
            0xA119,
            SimConfig {
                latency: Duration::ZERO..Duration::from_nanos(1),
                ..SimConfig::default()
            },
        );
        let server_key = SigningKey::from_bytes(&[0xC1; 32]);
        let client_key = SigningKey::from_bytes(&[0xC2; 32]);
        let root_key = SigningKey::from_bytes(&[0xC3; 32]);
        let server_id = server_key.verifying_key().to_bytes();
        let client_id = client_key.verifying_key().to_bytes();
        let mut server = net.join(server_id, None);
        let client = net.join(client_id, None);
        let connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        let server_handler = handler(&root_key);
        let task = tokio::spawn(async move {
            server_handler.handle::<SimTransport>(incoming.conn).await;
        });
        op_auth(&connection, &connect_proof(&root_key, &client_key))
            .await
            .unwrap();

        let mut held = Vec::new();
        for _ in 0..MAX_INBOUND_REQUESTS_PER_CONNECTION {
            let (mut send, recv) = connection.open_bi().await.unwrap();
            send_u8(&mut send, OP_GET_BLOB).await.unwrap();
            held.push((send, recv));
            tokio::task::yield_now().await;
        }
        let (mut excess_send, mut excess_recv) = connection.open_bi().await.unwrap();
        send_u8(&mut excess_send, OP_GET_BLOB).await.unwrap();
        excess_send.write_all(&[0x55; 32]).await.unwrap();
        excess_send.shutdown().await.unwrap();
        tokio::task::yield_now().await;
        let mut byte = [0u8; 1];
        assert_eq!(
            excess_recv.read(&mut byte).await.unwrap(),
            0,
            "an overcommitting connection was not closed"
        );
        assert!(connection.open_bi().await.is_err());
        drop(held);
        task.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn authenticated_idle_connection_is_closed_and_releases_its_slot() {
        let net = SimNet::new(
            0xA11A,
            SimConfig {
                latency: Duration::ZERO..Duration::from_nanos(1),
                ..SimConfig::default()
            },
        );
        let server_key = SigningKey::from_bytes(&[0xD1; 32]);
        let client_key = SigningKey::from_bytes(&[0xD2; 32]);
        let root_key = SigningKey::from_bytes(&[0xD3; 32]);
        let server_id = server_key.verifying_key().to_bytes();
        let client_id = client_key.verifying_key().to_bytes();
        let mut server = net.join(server_id, None);
        let client = net.join(client_id, None);
        let connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        let server_handler = handler(&root_key);
        let task = tokio::spawn(async move {
            server_handler.handle::<SimTransport>(incoming.conn).await;
        });
        op_auth(&connection, &connect_proof(&root_key, &client_key))
            .await
            .unwrap();
        tokio::task::yield_now().await;

        tokio::time::advance(INBOUND_CONNECTION_IDLE_DEADLINE - Duration::from_nanos(1)).await;
        tokio::task::yield_now().await;
        assert!(
            !task.is_finished(),
            "an authenticated connection ended before its idle deadline"
        );

        tokio::time::advance(Duration::from_nanos(1)).await;
        task.await.unwrap();
        assert!(
            connection.open_bi().await.is_err(),
            "the authenticated idle deadline did not close the connection"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn custody_endpoint_rejects_ordinary_blob_rpc_after_connect() {
        let net = SimNet::new(
            0xA11B,
            SimConfig {
                latency: Duration::ZERO..Duration::from_nanos(1),
                ..SimConfig::default()
            },
        );
        let server_key = SigningKey::from_bytes(&[0xE1; 32]);
        let client_key = SigningKey::from_bytes(&[0xE2; 32]);
        let connect_root = SigningKey::from_bytes(&[0xE3; 32]);
        let replica_root = SigningKey::from_bytes(&[0xE4; 32]);
        let server_id = server_key.verifying_key().to_bytes();
        let client_id = client_key.verifying_key().to_bytes();
        let mut server = net.join(server_id, None);
        let client = net.join(client_id, None);
        let connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        let mut server_handler = handler(&connect_root);
        server_handler.replica_server = Some(ReplicaServerConfig {
            trust_root: replica_root.verifying_key(),
            replica_set: ReplicaSetId::new([0xE5; 32]),
        });
        let task = tokio::spawn(async move {
            server_handler.handle::<SimTransport>(incoming.conn).await;
        });
        op_auth(&connection, &connect_proof(&connect_root, &client_key))
            .await
            .unwrap();

        let error = op_get_blob(&connection, &[0xE6; 32]).await.unwrap_err();
        assert!(
            error.to_string().contains("eof"),
            "custody endpoint unexpectedly served an ordinary RPC: {error}"
        );
        connection.close(0, b"test complete");
        task.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn custody_walk_stays_on_its_summary_generation_across_publish_and_reconnect() {
        let net = SimNet::new(
            0xA11C,
            SimConfig {
                latency: Duration::ZERO..Duration::from_nanos(1),
                ..SimConfig::default()
            },
        );
        let server_key = SigningKey::from_bytes(&[0xF1; 32]);
        let client_key = SigningKey::from_bytes(&[0xF2; 32]);
        let connect_root = SigningKey::from_bytes(&[0xF3; 32]);
        let replica_root = SigningKey::from_bytes(&[0xF4; 32]);
        let replica_set = ReplicaSetId::new([0xF5; 32]);
        let server_id = server_key.verifying_key().to_bytes();
        let client_id = client_key.verifying_key().to_bytes();
        let mut server = net.join(server_id, None);
        let client = net.join(client_id, None);

        let mut payloads =
            blob_payloads_with_prefix(0x42, ReplicaComponent::Blobs.page_limit() + 2, 0);
        // Make the concurrent append sort after generation A's first-page
        // cursor, so serving generation B's final page necessarily reproduces
        // the production count-overrun failure.
        payloads.sort_unstable_by_key(|payload| *blake3::hash(payload).as_bytes());
        let added_payload = payloads.pop().unwrap();
        let mut generation_a = MemoryRepo::default();
        for payload in payloads {
            generation_a
                .put::<UnknownBlob, _>(Bytes::from(payload))
                .unwrap();
        }
        let mut generation_b = generation_a.clone();
        let added_len = added_payload.len() as u64;
        let added_handle = generation_b
            .put::<UnknownBlob, _>(Bytes::from(added_payload.clone()))
            .unwrap();
        let added_id = ReplicaItemId(added_handle.raw);
        let snapshot_a = snapshot_from_store(&mut generation_a).unwrap();
        let snapshot_b = snapshot_from_store(&mut generation_b).unwrap();

        let mut server_handler = handler(&connect_root);
        *server_handler.replica_snapshot.lock().unwrap() =
            Some(shared_replica_snapshot(snapshot_a));
        let serving_snapshot = server_handler.replica_snapshot.clone();
        let snapshot_generations = server_handler.replica_snapshot_generations.clone();
        server_handler.replica_server = Some(ReplicaServerConfig {
            trust_root: replica_root.verifying_key(),
            replica_set,
        });

        let connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        let first_handler = server_handler.clone();
        let first_task = tokio::spawn(async move {
            first_handler.handle::<SimTransport>(incoming.conn).await;
        });
        op_auth(&connection, &connect_proof(&connect_root, &client_key))
            .await
            .unwrap();
        let proof = replica_proof(&replica_root, &client_key, replica_set);

        // Data requests cannot select the mutable current snapshot directly:
        // a successful summary must first cache the exact generation token.
        let unknown_generation = ReplicaGeneration::new([0xF6; 32]);
        assert!(
            op_replica_page(
                &connection,
                replica_set,
                &proof,
                unknown_generation,
                ReplicaComponent::Blobs,
                0x42,
                None,
            )
            .await
            .is_err()
        );
        let mut before_summary_target = vec![0; added_payload.len()];
        assert!(
            op_replica_blob_range(
                &connection,
                replica_set,
                &proof,
                unknown_generation,
                added_id,
                added_len,
                0,
                &mut before_summary_target,
            )
            .await
            .is_err()
        );

        let summary = op_replica_summary(&connection, replica_set, &proof)
            .await
            .unwrap();
        let generation_a_token = summary.generation();
        assert_eq!(
            summary.bucket(ReplicaComponent::Blobs, 0x42).count,
            (ReplicaComponent::Blobs.page_limit() + 1) as u64
        );

        let (first_page, done) = op_replica_page(
            &connection,
            replica_set,
            &proof,
            generation_a_token,
            ReplicaComponent::Blobs,
            0x42,
            None,
        )
        .await
        .unwrap();
        assert!(!done);
        assert_eq!(first_page.len(), ReplicaComponent::Blobs.page_limit());
        let cursor = first_page.last().unwrap().id();

        // This is the production race: a concurrent local sweep publishes a
        // newly learned semantic item after this peer received generation A's
        // summary and first page but before it asks for the final page.
        *serving_snapshot.lock().unwrap() = Some(shared_replica_snapshot(snapshot_b));

        // The newly published blob is absent from A even though it is now in
        // the current serving slot.
        let mut generation_a_target = vec![0; added_payload.len()];
        assert_eq!(
            op_replica_blob_range(
                &connection,
                replica_set,
                &proof,
                generation_a_token,
                added_id,
                added_len,
                0,
                &mut generation_a_target,
            )
            .await
            .unwrap(),
            None
        );

        // The generation cache is host-wide, not connection-local: a pooled-
        // connection eviction and redial can resume the same immutable walk.
        connection.close(0, b"exercise custody reconnect");
        first_task.await.unwrap();
        let connection = client
            .transport
            .dial(server_id, PILE_SYNC_ALPN)
            .await
            .unwrap();
        let incoming = server.incoming.recv().await.unwrap();
        let second_handler = server_handler.clone();
        let second_task = tokio::spawn(async move {
            second_handler.handle::<SimTransport>(incoming.conn).await;
        });
        op_auth(&connection, &connect_proof(&connect_root, &client_key))
            .await
            .unwrap();
        let (page, done) = op_replica_page(
            &connection,
            replica_set,
            &proof,
            generation_a_token,
            ReplicaComponent::Blobs,
            0x42,
            Some(cursor),
        )
        .await
        .unwrap();
        assert!(done);
        assert_eq!(page.len(), 1);
        assert_eq!(
            first_page.len() + page.len(),
            summary.bucket(ReplicaComponent::Blobs, 0x42).count as usize
        );

        // The next summary exposes the concurrently published item on a new
        // walk. The bounded cache retains A as an independent generation, so
        // another in-flight or reconnecting walker is not invalidated merely
        // because this identity also observed B.
        let next_summary = op_replica_summary(&connection, replica_set, &proof)
            .await
            .unwrap();
        let generation_b_token = next_summary.generation();
        assert_ne!(generation_a_token, generation_b_token);
        assert_eq!(
            next_summary.bucket(ReplicaComponent::Blobs, 0x42).count,
            (ReplicaComponent::Blobs.page_limit() + 2) as u64
        );

        let (old_first_page, old_done) = op_replica_page(
            &connection,
            replica_set,
            &proof,
            generation_a_token,
            ReplicaComponent::Blobs,
            0x42,
            None,
        )
        .await
        .unwrap();
        assert!(!old_done);
        assert_eq!(
            old_first_page
                .iter()
                .map(|item| item.id())
                .collect::<Vec<_>>(),
            first_page.iter().map(|item| item.id()).collect::<Vec<_>>()
        );
        let mut old_blob_target = vec![0; added_payload.len()];
        assert_eq!(
            op_replica_blob_range(
                &connection,
                replica_set,
                &proof,
                generation_a_token,
                added_id,
                added_len,
                0,
                &mut old_blob_target,
            )
            .await
            .unwrap(),
            None
        );
        let (next_first_page, done) = op_replica_page(
            &connection,
            replica_set,
            &proof,
            generation_b_token,
            ReplicaComponent::Blobs,
            0x42,
            None,
        )
        .await
        .unwrap();
        assert!(!done);
        let (next_final_page, done) = op_replica_page(
            &connection,
            replica_set,
            &proof,
            generation_b_token,
            ReplicaComponent::Blobs,
            0x42,
            Some(next_first_page.last().unwrap().id()),
        )
        .await
        .unwrap();
        assert!(done);
        assert_eq!(next_final_page.len(), 2);

        let mut generation_b_target = vec![0; added_payload.len()];
        assert_eq!(
            op_replica_blob_range(
                &connection,
                replica_set,
                &proof,
                generation_b_token,
                added_id,
                added_len,
                0,
                &mut generation_b_target,
            )
            .await
            .unwrap(),
            Some(added_payload.len())
        );
        assert_eq!(generation_b_target, added_payload);
        let generations = snapshot_generations.lock().unwrap();
        assert_eq!(generations.snapshots.len(), 2);
        assert!(generations.snapshots.contains_key(&generation_a_token));
        assert!(generations.snapshots.contains_key(&generation_b_token));
        drop(generations);

        connection.close(0, b"test complete");
        second_task.await.unwrap();
    }

    #[test]
    fn node_global_connection_budget_is_shared_across_handler_clones() {
        let root = SigningKey::from_bytes(&[0xF3; 32]);
        let mut handler = handler(&root);
        handler.inbound_connections = Arc::new(tokio::sync::Semaphore::new(2));
        let clone = handler.clone();
        let first = handler.try_admit_connection().unwrap();
        let second = clone.try_admit_connection().unwrap();
        assert!(
            handler.try_admit_connection().is_none(),
            "a clone escaped the node-global connection budget"
        );
        drop(first);
        assert!(clone.try_admit_connection().is_some());
        drop(second);
    }

    #[test]
    fn node_global_request_budget_is_shared_across_handler_clones() {
        let root = SigningKey::from_bytes(&[0xD3; 32]);
        let mut handler = handler(&root);
        handler.inbound_requests = Arc::new(tokio::sync::Semaphore::new(2));
        let clone = handler.clone();
        let connection_a = Arc::new(tokio::sync::Semaphore::new(2));
        let connection_b = Arc::new(tokio::sync::Semaphore::new(2));
        let first = handler.try_admit_request(&connection_a).unwrap();
        let second = clone.try_admit_request(&connection_b).unwrap();
        assert!(
            handler.try_admit_request(&connection_a).is_none(),
            "a clone escaped the node-global request budget"
        );
        drop(first);
        assert!(handler.try_admit_request(&connection_a).is_some());
        drop(second);
    }
}
