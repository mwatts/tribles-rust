//! Async network host for one authorized team inventory.
//!
//! The synchronous [`crate::peer::Peer`] owns the store. This module owns
//! transport, connection authentication, immutable serving snapshots, and the
//! periodic anti-entropy scheduler. Gossip is deliberately only a wake-up
//! hint: every useful byte arrives through a CONNECT- and SYNC_TEAM-authorized
//! direct connection and is checked against a pinned PATCH root.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use anybytes::Bytes;
use ed25519_dalek::{SigningKey, VerifyingKey};
use futures::{StreamExt, stream::FuturesUnordered};
use iroh_base::{EndpointAddr, EndpointId};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{Instrument, debug, debug_span, info_span, instrument, trace, warn};
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::capability::CapabilityProofBundle;
use triblespace_core::collection::CollectionStore;
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{Entry as PatchEntry, IdentitySchema, PATCH};
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::peer::PeerEvidence;
use triblespace_core::repo::{
    BlobStore, BlobStoreGet, BlobStoreList, CapabilityProofStore, PeerStore,
};

use crate::channel::{
    MAX_ADMISSION_BRIDGE_BATCHES, NetCommand, NetEvent, NetEventBatch, SnapshotNotice,
};
use crate::identity::iroh_secret;
use crate::inventory::{
    AuthorizedInventorySession, InventoryComponent, InventoryGeneration, InventoryManifest,
    InventoryServerConfig, InventorySnapshot, ReconcileQos,
};
use crate::inventory_reconcile::InventoryWalker;
use crate::inventory_wire::{
    BLOB_TRANSFER_CHUNK_BYTES, InventoryBlobRangeRequest, InventoryBlobRangeResponse,
    InventoryLeaf, InventoryLeafValue, InventoryNodeRequest, InventoryNodeResponse,
    OP_INVENTORY_AUTH, OP_INVENTORY_BLOB_RANGE, OP_INVENTORY_MANIFEST, OP_INVENTORY_NODE,
    op_inventory_auth, op_inventory_blob_range, op_inventory_manifest, op_inventory_node,
    recv_blob_range_request, recv_inventory_auth_request, recv_manifest_request, recv_node_request,
    send_blob_not_in_snapshot, send_blob_range, send_blob_snapshot_unavailable,
    send_inventory_auth_ok, send_inventory_auth_rejected, send_manifest, send_node_response,
    snapshot_node,
};
use crate::protocol::*;
use crate::transport::{Conn, GossipEvent, GossipSink, Harness, PeerId, Transport};

/// Configuration for a peer attached to one single-team store.
///
/// `team` is simultaneously the capability trust root, inventory scope, and
/// gossip rendezvous. The backing store must not mix team-unscoped records,
/// proofs, or blobs from another team.
#[derive(Clone)]
pub struct PeerConfig {
    /// Bootstrap routes. Stored PEER evidence can add routing candidates but
    /// never grants authority.
    pub peers: Vec<EndpointAddr>,
    /// Team trust root, inventory scope, and gossip topic.
    pub team: VerifyingKey,
    /// Complete proof authorizing this endpoint key for exact CONNECT on
    /// `team`.
    pub connect_proof: CapabilityProofBundle,
    /// Complete proof authorizing this endpoint key for exact SYNC_TEAM on
    /// `team`.
    pub sync_proof: CapabilityProofBundle,
    /// Local scheduling and blob-residency choices. Never sent as authority.
    pub qos: ReconcileQos,
}

/// Team-derived gossip topic. Keeping the derivation explicit prevents a
/// caller from accidentally rendezvousing an authorized store on another
/// team's mesh.
pub fn team_gossip_topic(team: VerifyingKey) -> [u8; 32] {
    team.to_bytes()
}

/// Snapshot of the complete single-team store observation served by the host.
pub(crate) struct StoreSnapshot<R> {
    inventory: InventorySnapshot<R>,
    inventory_bodies: MemoryRepo,
    routing_peers: Vec<PeerId>,
}

impl StoreSnapshot<()> {
    pub(crate) fn from_store<S>(
        store: &mut S,
        team: VerifyingKey,
    ) -> anyhow::Result<StoreSnapshot<S::Reader>>
    where
        S: BlobStore + CollectionStore + CapabilityProofStore + PeerStore,
    {
        // Pile::reader performs external-append reobservation. Do it before
        // enumerating native evidence so one refresh sees every component.
        let reader = store.reader().map_err(anyhow::Error::new)?;
        let peers = store
            .peers()
            .map_err(anyhow::Error::new)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;
        let records = store
            .records()
            .map_err(anyhow::Error::new)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;
        let proofs = store
            .proofs()
            .map_err(anyhow::Error::new)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;
        let inventory = InventorySnapshot::from_observation(
            team,
            reader,
            peers.iter().copied(),
            records.iter().copied(),
            proofs.iter().cloned(),
        )?;

        // Record/proof bodies are frozen beside their key-only PATCHes. A
        // later store mutation can therefore never splice a body into an old
        // root walk.
        let mut inventory_bodies = MemoryRepo::default();
        for record in records {
            inventory_bodies
                .insert(record)
                .map_err(anyhow::Error::new)?;
        }
        for proof in proofs {
            inventory_bodies
                .insert_proof(proof)
                .map_err(anyhow::Error::new)?;
        }
        let mut routing_peers: Vec<_> = peers
            .into_iter()
            .filter(|evidence| evidence.team() == team)
            .map(|evidence| evidence.peer().to_bytes())
            .collect();
        routing_peers.sort_unstable();
        routing_peers.dedup();
        Ok(StoreSnapshot {
            inventory,
            inventory_bodies,
            routing_peers,
        })
    }
}

pub(crate) enum PinnedBlob {
    SnapshotUnavailable,
    NotInSnapshot,
    Found(Bytes),
}

/// Type-erased immutable view shared between the store and async host.
pub(crate) trait AnySnapshot: Send + 'static {
    fn team(&self) -> VerifyingKey;
    fn manifest(&self) -> InventoryManifest;
    fn routing_peers(&self) -> Vec<PeerId>;
    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>>;
    fn node_summary(
        &self,
        component: InventoryComponent,
        relative_prefix: &[u8],
    ) -> Option<([u8; 32], u64)>;
    fn contains_relative_key(&self, component: InventoryComponent, relative_key: &[u8]) -> bool;
    fn inventory_node(
        &mut self,
        request: &InventoryNodeRequest,
    ) -> anyhow::Result<InventoryNodeResponse>;
    fn inventory_blob(&self, request: InventoryBlobRangeRequest) -> PinnedBlob;
}

impl<R> AnySnapshot for StoreSnapshot<R>
where
    R: BlobStoreGet + BlobStoreList + Send + 'static,
{
    fn team(&self) -> VerifyingKey {
        self.inventory.team()
    }

    fn manifest(&self) -> InventoryManifest {
        self.inventory.manifest().clone()
    }

    fn routing_peers(&self) -> Vec<PeerId> {
        self.routing_peers.clone()
    }

    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>> {
        self.inventory
            .reader()
            .get::<Bytes, UnknownBlob>(Inline::<Handle<UnknownBlob>>::new(*hash))
            .ok()
            .map(|bytes| bytes.to_vec())
    }

    fn node_summary(
        &self,
        component: InventoryComponent,
        relative_prefix: &[u8],
    ) -> Option<([u8; 32], u64)> {
        self.inventory.node_summary(component, relative_prefix)
    }

    fn contains_relative_key(&self, component: InventoryComponent, relative_key: &[u8]) -> bool {
        self.inventory
            .contains_relative_key(component, relative_key)
    }

    fn inventory_node(
        &mut self,
        request: &InventoryNodeRequest,
    ) -> anyhow::Result<InventoryNodeResponse> {
        snapshot_node(&self.inventory, &mut self.inventory_bodies, request)
    }

    fn inventory_blob(&self, request: InventoryBlobRangeRequest) -> PinnedBlob {
        let advertised = self
            .inventory
            .manifest()
            .component(InventoryComponent::Blob);
        if advertised.root() != Some(request.root) || advertised.leaf_count() != request.leaf_count
        {
            return PinnedBlob::SnapshotUnavailable;
        }
        if !self
            .inventory
            .contains_relative_key(InventoryComponent::Blob, &request.hash)
        {
            return PinnedBlob::NotInSnapshot;
        }
        self.inventory
            .blob_bytes(request.hash)
            .map(PinnedBlob::Found)
            .unwrap_or(PinnedBlob::SnapshotUnavailable)
    }
}

type SharedSnapshot = Arc<Mutex<Box<dyn AnySnapshot>>>;
type SnapshotSlot = Arc<Mutex<Option<SharedSnapshot>>>;

fn shared_snapshot(snapshot: impl AnySnapshot) -> SharedSnapshot {
    Arc::new(Mutex::new(Box::new(snapshot)))
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct InventoryCacheKey {
    team: [u8; 32],
    component: InventoryComponent,
    root: [u8; 32],
}

/// A manifest pins at most four entries, so this retains at least eight
/// complete recent manifests globally while bounding churn-driven memory use.
const MAX_PINNED_INVENTORY_ROOTS: usize = 32;

#[derive(Default)]
struct InventorySnapshotCache {
    snapshots: HashMap<InventoryCacheKey, SharedSnapshot>,
    least_to_most_recent: VecDeque<InventoryCacheKey>,
}

impl InventorySnapshotCache {
    fn pin_manifest(
        &mut self,
        team: VerifyingKey,
        manifest: &InventoryManifest,
        snapshot: SharedSnapshot,
    ) {
        for entry in manifest.components() {
            let Some(root) = entry.root() else {
                continue;
            };
            let key = InventoryCacheKey {
                team: team.to_bytes(),
                component: entry.component(),
                root,
            };
            self.snapshots
                .entry(key)
                .or_insert_with(|| snapshot.clone());
            self.touch(key);
        }
        while self.snapshots.len() > MAX_PINNED_INVENTORY_ROOTS {
            let oldest = self
                .least_to_most_recent
                .pop_front()
                .expect("nonempty inventory cache has an LRU entry");
            self.snapshots.remove(&oldest);
        }
    }

    fn get(
        &mut self,
        team: VerifyingKey,
        component: InventoryComponent,
        root: [u8; 32],
    ) -> Option<SharedSnapshot> {
        let key = InventoryCacheKey {
            team: team.to_bytes(),
            component,
            root,
        };
        let snapshot = self.snapshots.get(&key)?.clone();
        self.touch(key);
        Some(snapshot)
    }

    fn touch(&mut self, key: InventoryCacheKey) {
        if let Some(position) = self
            .least_to_most_recent
            .iter()
            .position(|candidate| *candidate == key)
        {
            self.least_to_most_recent.remove(position);
        }
        self.least_to_most_recent.push_back(key);
    }
}

type InventorySnapshots = Arc<Mutex<InventorySnapshotCache>>;

const GOSSIP_INVENTORY_WAKE: u8 = 0x04;
const GOSSIP_INVENTORY_WAKE_VERSION: u32 = 1;
const GOSSIP_INVENTORY_WAKE_LEN: usize = 1 + 4 + 32 + 32;

fn inventory_wake_frame(team: VerifyingKey, generation: InventoryGeneration) -> Vec<u8> {
    let mut frame = Vec::with_capacity(GOSSIP_INVENTORY_WAKE_LEN);
    frame.push(GOSSIP_INVENTORY_WAKE);
    frame.extend_from_slice(&GOSSIP_INVENTORY_WAKE_VERSION.to_be_bytes());
    frame.extend_from_slice(team.as_bytes());
    frame.extend_from_slice(&generation.into_bytes());
    frame
}

fn decode_inventory_wake_frame(bytes: &[u8], team: VerifyingKey) -> Option<InventoryGeneration> {
    if bytes.len() != GOSSIP_INVENTORY_WAKE_LEN
        || bytes[0] != GOSSIP_INVENTORY_WAKE
        || bytes[1..5] != GOSSIP_INVENTORY_WAKE_VERSION.to_be_bytes()
        || bytes[5..37] != team.to_bytes()
    {
        return None;
    }
    let generation = bytes[37..].try_into().ok()?;
    Some(InventoryGeneration::from_bytes(generation))
}

/// The async capability cloned into lazy readers. Exact GET_BLOB and broad
/// inventory enumeration both require the connection-local SYNC_TEAM session;
/// the content hash narrows the request but is not itself disclosure authority.
pub(crate) trait NetCapability: Send + Sync {
    fn fetch_blob(&self, hash: RawHash) -> futures::future::BoxFuture<'static, Option<Vec<u8>>>;
}

struct RoutingTable {
    configured: PATCH<32, IdentitySchema>,
    learned: PATCH<32, IdentitySchema>,
}

impl RoutingTable {
    fn new(configured: Vec<PeerId>) -> Self {
        let mut configured_index = PATCH::new();
        for peer in configured {
            configured_index.insert(&PatchEntry::new(&peer));
        }
        Self {
            configured: configured_index,
            learned: PATCH::new(),
        }
    }

    fn note(&mut self, peer: PeerId) {
        if self.configured.get(&peer).is_some() {
            return;
        }
        self.learned.insert(&PatchEntry::new(&peer));
    }

    fn replace_learned(&mut self, peers: Vec<PeerId>, self_id: PeerId) {
        for peer in peers.into_iter().filter(|peer| *peer != self_id) {
            self.note(peer);
        }
    }

    fn candidates(&self, self_id: PeerId) -> Vec<PeerId> {
        let mut all = self.configured.clone();
        all.union(self.learned.clone());
        all.into_iter_ordered()
            .filter(|peer| *peer != self_id)
            .collect()
    }
}

type RoutingCandidates = Arc<Mutex<RoutingTable>>;

struct PoolEntry<C> {
    connection: tokio::sync::OnceCell<C>,
    inventory_auth: tokio::sync::OnceCell<()>,
}

impl<C> Default for PoolEntry<C> {
    fn default() -> Self {
        Self {
            connection: tokio::sync::OnceCell::new(),
            inventory_auth: tokio::sync::OnceCell::new(),
        }
    }
}

type SharedPool<C> = Arc<tokio::sync::Mutex<HashMap<PeerId, Arc<PoolEntry<C>>>>>;

fn new_shared_pool<C>() -> SharedPool<C> {
    Arc::new(tokio::sync::Mutex::new(HashMap::new()))
}

struct NetCap<T: Transport> {
    transport: T,
    pool: SharedPool<T::Conn>,
    connect_proof: CapabilityProofBundle,
    sync_proof: CapabilityProofBundle,
    can_fetch: bool,
    my_id: PeerId,
    candidates: RoutingCandidates,
}

impl<T: Transport> NetCapability for NetCap<T> {
    fn fetch_blob(&self, hash: RawHash) -> futures::future::BoxFuture<'static, Option<Vec<u8>>> {
        let transport = self.transport.clone();
        let pool = self.pool.clone();
        let connect_proof = self.connect_proof.clone();
        let sync_proof = self.sync_proof.clone();
        let can_fetch = self.can_fetch;
        let my_id = self.my_id;
        let known = self.candidates.lock().unwrap().candidates(my_id);
        Box::pin(async move {
            if !can_fetch {
                return None;
            }
            let bytes = fetch_from_providers(
                &transport,
                &hash,
                &pool,
                &known,
                &connect_proof,
                &sync_proof,
            )
            .await;
            bytes.filter(|bytes| blake3::hash(bytes).as_bytes() == &hash)
        })
    }
}

/// Default end-to-end budget for an interactive exact blob read.
pub const INTERACTIVE_FETCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// Commands and snapshot updates sent to the host runtime.
#[derive(Clone)]
pub struct NetSender {
    cmd_tx: mpsc::Sender<NetCommand>,
    snapshot: SnapshotSlot,
    installed_generation: Arc<Mutex<Option<InventoryGeneration>>>,
    cap: tokio::sync::watch::Receiver<Option<Arc<dyn NetCapability>>>,
    id: EndpointId,
}

impl NetSender {
    pub fn id(&self) -> EndpointId {
        self.id
    }

    pub(crate) fn update_snapshot(&self, snapshot: impl AnySnapshot) {
        let manifest = snapshot.manifest();
        let notice = SnapshotNotice {
            generation: manifest.generation(),
            peers: snapshot.routing_peers(),
        };
        *self.snapshot.lock().unwrap() = Some(shared_snapshot(snapshot));

        let mut installed = self.installed_generation.lock().unwrap();
        if *installed != Some(notice.generation) {
            *installed = Some(notice.generation);
            let _ = self.cmd_tx.send(NetCommand::SnapshotInstalled(notice));
        }
    }

    pub fn clear_snapshot(&self) {
        *self.snapshot.lock().unwrap() = None;
        *self.installed_generation.lock().unwrap() = None;
    }

    pub(crate) fn refresh_store_snapshot<S>(&self, store: &mut S, team: VerifyingKey) -> bool
    where
        S: BlobStore + CollectionStore + CapabilityProofStore + PeerStore,
    {
        match StoreSnapshot::from_store(store, team) {
            Ok(snapshot) => {
                self.update_snapshot(snapshot);
                true
            }
            Err(error) => {
                warn!(%error, "store inventory snapshot unavailable; clearing serving view");
                self.clear_snapshot();
                false
            }
        }
    }

    async fn ready_capability(&self) -> anyhow::Result<Arc<dyn NetCapability>> {
        let mut slot = self.cap.clone();
        loop {
            if let Some(capability) = slot.borrow().clone() {
                return Ok(capability);
            }
            slot.changed()
                .await
                .map_err(|_| anyhow::anyhow!("network host stopped before becoming ready"))?;
        }
    }

    pub async fn fetch_blob(&self, hash: RawHash, budget: std::time::Duration) -> Option<Vec<u8>> {
        tokio::time::timeout(budget, async {
            self.ready_capability().await.ok()?.fetch_blob(hash).await
        })
        .await
        .ok()
        .flatten()
    }
}

/// Incoming inventory leaves for the synchronous store side.
pub struct NetReceiver {
    evt_rx: tokio::sync::mpsc::Receiver<NetEventBatch>,
}

impl NetReceiver {
    pub(crate) fn try_recv(&mut self) -> Option<NetEventBatch> {
        self.evt_rx.try_recv().ok()
    }
}

/// Host half of [`wire`].
pub struct HostWiring {
    cmd_rx: mpsc::Receiver<NetCommand>,
    evt_tx: tokio::sync::mpsc::Sender<NetEventBatch>,
    snapshot: SnapshotSlot,
    cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
}

/// Construct the synchronous/asynchronous boundary for a caller-owned host.
pub fn wire(id: EndpointId) -> (NetSender, NetReceiver, HostWiring) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    // Batches are independently bounded by item and byte count. Sixteen
    // queued batches keep a first sync moving between synchronous refreshes
    // without turning this bridge into an unbounded second store.
    let (evt_tx, evt_rx) = tokio::sync::mpsc::channel(MAX_ADMISSION_BRIDGE_BATCHES);
    let snapshot = Arc::new(Mutex::new(None));
    let installed_generation = Arc::new(Mutex::new(None));
    let (cap_tx, cap_rx) = tokio::sync::watch::channel(None);
    (
        NetSender {
            cmd_tx,
            snapshot: snapshot.clone(),
            installed_generation,
            cap: cap_rx,
            id,
        },
        NetReceiver { evt_rx },
        HostWiring {
            cmd_rx,
            evt_tx,
            snapshot,
            cap_tx,
        },
    )
}

/// Run the host over a caller-provided transport harness.
pub async fn run_host<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    host_loop(harness, config, wiring).await;
}

/// Spawn the production iroh host thread and wait until its endpoint is bound.
///
/// The startup rendezvous is deliberately synchronous: returning a sender for
/// a thread that already failed would make a dead peer indistinguishable from
/// a temporarily quiet one.
pub fn spawn(key: SigningKey, config: PeerConfig) -> anyhow::Result<(NetSender, NetReceiver)> {
    let secret = iroh_secret(&key);
    let id: EndpointId = secret.public().into();
    let (sender, receiver, wiring) = wire(id);
    let (startup_tx, startup_rx) = mpsc::sync_channel(1);
    let _thread = thread::Builder::new()
        .name("triblespace-net".to_owned())
        .spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = startup_tx
                        .send(Err(anyhow::Error::new(error)
                            .context("create triblespace-net tokio runtime")));
                    return;
                }
            };
            runtime.block_on(async move {
                let harness = match crate::transport::iroh::bind(secret, &config).await {
                    Ok(harness) => harness,
                    Err(error) => {
                        let _ = startup_tx.send(Err(error.context("bind iroh network host")));
                        return;
                    }
                };
                if startup_tx.send(Ok(())).is_err() {
                    return;
                }
                run_host(harness, config, wiring).await;
            });
        })
        .map_err(|error| anyhow::Error::new(error).context("spawn triblespace-net thread"))?;
    startup_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("network host stopped during startup"))??;
    Ok((sender, receiver))
}

const DIAL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
const OP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
const INVENTORY_SWEEP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(300);
const INVENTORY_SWEEP_PERIOD: std::time::Duration = std::time::Duration::from_secs(30);
/// Untrusted gossip can accelerate periodic correctness, but cannot schedule
/// more than one extra sweep per interval regardless of sender volume.
const MIN_GOSSIP_WAKE_PERIOD: std::time::Duration = std::time::Duration::from_secs(1);
const HOST_POLL_PERIOD: std::time::Duration = std::time::Duration::from_millis(10);
const INBOUND_AUTH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
const INBOUND_CONNECTION_IDLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);
const INBOUND_REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
pub(crate) const MAX_INBOUND_CONNECTIONS_GLOBAL: usize = 64;
pub(crate) const MAX_INBOUND_REQUESTS_PER_CONNECTION: usize = 16;
const MAX_INBOUND_REQUESTS_GLOBAL: usize = 64;
const MAX_CONCURRENT_SWEEPS: usize = 8;

struct SweepOutcome {
    peer: PeerId,
    success: bool,
}

async fn host_loop<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    let Harness {
        transport,
        mut incoming,
        gossip,
    } = harness;
    let HostWiringParts {
        commands,
        events,
        snapshot,
        cap_tx,
    } = HostWiringParts::from(wiring);
    let my_id = transport.local_id();
    let mut configured: Vec<_> = config
        .peers
        .iter()
        .map(|address| *address.id.as_bytes())
        .filter(|peer| *peer != my_id)
        .collect();
    configured.sort_unstable();
    configured.dedup();
    let candidates = Arc::new(Mutex::new(RoutingTable::new(configured)));
    let pool = new_shared_pool();
    let _ = cap_tx.send(Some(Arc::new(NetCap {
        transport: transport.clone(),
        pool: pool.clone(),
        connect_proof: config.connect_proof.clone(),
        sync_proof: config.sync_proof.clone(),
        can_fetch: config.qos.direction.pulls(),
        my_id,
        candidates: candidates.clone(),
    }) as Arc<dyn NetCapability>));

    let handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        snapshots: Arc::new(Mutex::new(InventorySnapshotCache::default())),
        server: InventoryServerConfig::full_team(config.team),
        events: events.clone(),
        candidates: candidates.clone(),
        serve_inventory: config.qos.direction.serves(),
        admit_inbound_peer: config.qos.direction.admits_inbound_peer(),
        inbound_connections: Arc::new(tokio::sync::Semaphore::new(MAX_INBOUND_CONNECTIONS_GLOBAL)),
        inbound_requests: Arc::new(tokio::sync::Semaphore::new(MAX_INBOUND_REQUESTS_GLOBAL)),
    };
    tokio::spawn(async move {
        while let Some(accepted) = incoming.recv().await {
            if accepted.alpn != PILE_SYNC_ALPN {
                accepted.conn.close(1, b"unknown protocol");
                continue;
            }
            let Some(permit) = handler.try_admit_connection() else {
                accepted.conn.close(1, b"inbound connection limit exceeded");
                continue;
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                handler.handle_admitted::<T>(accepted.conn, permit).await;
            });
        }
    });

    let wake_pending = Arc::new(AtomicBool::new(false));
    let mut gossip_sender = None;
    if let Some((sender, mut gossip_events)) = gossip {
        gossip_sender = Some(sender);
        let team = config.team;
        let wake_pending = wake_pending.clone();
        tokio::spawn(async move {
            let mut last_generation = None;
            while let Some(event) = gossip_events.recv().await {
                match event {
                    GossipEvent::Received { bytes, .. } => {
                        if let Some(generation) = decode_inventory_wake_frame(&bytes, team)
                            && last_generation != Some(generation)
                        {
                            last_generation = Some(generation);
                            wake_pending.store(true, Ordering::Release);
                        }
                    }
                    GossipEvent::NeighborUp(peer) => {
                        trace!(peer = %hex::encode(&peer[..4]), "team gossip neighbor up");
                    }
                    GossipEvent::NeighborDown(peer) => {
                        trace!(peer = %hex::encode(&peer[..4]), "team gossip neighbor down");
                    }
                }
            }
        });
    }

    let (sweep_tx, mut sweep_rx) = tokio::sync::mpsc::unbounded_channel::<SweepOutcome>();
    let mut in_flight = HashSet::new();
    let mut pending_sweeps = VecDeque::new();
    let mut failures: HashMap<PeerId, (u32, crate::clock::Mono)> = HashMap::new();
    let mut next_sweep = crate::clock::mono_now();
    let mut next_gossip_wake = crate::clock::mono_now();
    let mut current_generation = None;
    let commands = commands;

    loop {
        let mut disconnected = false;
        loop {
            match commands.try_recv() {
                Ok(NetCommand::SnapshotInstalled(notice)) => {
                    current_generation = Some(notice.generation);
                    candidates
                        .lock()
                        .unwrap()
                        .replace_learned(notice.peers, my_id);
                    if config.qos.direction.publishes()
                        && let Some(sender) = gossip_sender.as_ref()
                    {
                        let sender = sender.clone();
                        let frame = inventory_wake_frame(config.team, notice.generation);
                        tokio::spawn(async move {
                            if let Err(error) = sender.broadcast(frame).await {
                                debug!(%error, "inventory wake broadcast failed");
                            }
                        });
                    }
                    // The slot was installed before this command. Pull now so
                    // startup and every newly admitted generation converge
                    // without waiting for gossip or the periodic interval.
                    next_sweep = crate::clock::mono_now();
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        if disconnected {
            transport.shutdown().await;
            return;
        }

        let now = crate::clock::mono_now();
        if now >= next_gossip_wake && wake_pending.swap(false, Ordering::AcqRel) {
            next_sweep = now;
            next_gossip_wake = now + MIN_GOSSIP_WAKE_PERIOD;
        }
        while let Ok(outcome) = sweep_rx.try_recv() {
            in_flight.remove(&outcome.peer);
            if outcome.success {
                failures.remove(&outcome.peer);
            } else {
                let attempts = failures
                    .get(&outcome.peer)
                    .map_or(1, |(attempts, _)| attempts.saturating_add(1));
                let shift = attempts.saturating_sub(1).min(31);
                let multiplier = 1u32 << shift;
                let backoff = crate::RETRY_BACKOFF_BASE
                    .saturating_mul(multiplier)
                    .min(crate::RETRY_BACKOFF_CAP);
                failures.insert(outcome.peer, (attempts, crate::clock::mono_now() + backoff));
            }
        }

        let now = crate::clock::mono_now();
        if config.qos.direction.pulls() && current_generation.is_some() && now >= next_sweep {
            let queued: HashSet<_> = pending_sweeps.iter().copied().collect();
            for peer in candidates.lock().unwrap().candidates(my_id) {
                if !in_flight.contains(&peer) && !queued.contains(&peer) {
                    pending_sweeps.push_back(peer);
                }
            }
            next_sweep = now + INVENTORY_SWEEP_PERIOD;
        }

        let local = snapshot.lock().unwrap().as_ref().cloned();
        if config.qos.direction.pulls()
            && let Some(local) = local
        {
            let mut deferred = VecDeque::new();
            while in_flight.len() < MAX_CONCURRENT_SWEEPS {
                let Some(peer) = pending_sweeps.pop_front() else {
                    break;
                };
                if in_flight.contains(&peer) {
                    continue;
                }
                if failures
                    .get(&peer)
                    .is_some_and(|(_, retry_at)| now < *retry_at)
                {
                    deferred.push_back(peer);
                    continue;
                }
                in_flight.insert(peer);
                let transport = transport.clone();
                let pool = pool.clone();
                let connect_proof = config.connect_proof.clone();
                let sync_proof = config.sync_proof.clone();
                let events = events.clone();
                let candidates = candidates.clone();
                let sweep_tx = sweep_tx.clone();
                let team = config.team;
                let qos = config.qos;
                let local = local.clone();
                tokio::spawn(async move {
                    let result = tokio::time::timeout(
                        INVENTORY_SWEEP_DEADLINE,
                        reconcile_inventory_peer(
                            &transport,
                            &pool,
                            peer,
                            team,
                            &connect_proof,
                            &sync_proof,
                            qos,
                            local,
                            &events,
                            &candidates,
                        ),
                    )
                    .await;
                    let success = match result {
                        Ok(Ok(())) => true,
                        Ok(Err(error)) => {
                            debug!(peer = %hex::encode(&peer[..4]), %error, "inventory sweep failed");
                            pool_evict(&pool, peer).await;
                            false
                        }
                        Err(_) => {
                            debug!(peer = %hex::encode(&peer[..4]), "inventory sweep deadline exceeded");
                            pool_evict(&pool, peer).await;
                            false
                        }
                    };
                    let _ = sweep_tx.send(SweepOutcome { peer, success });
                });
            }
            pending_sweeps.extend(deferred);
        }

        tokio::time::sleep(HOST_POLL_PERIOD).await;
    }
}

struct HostWiringParts {
    commands: mpsc::Receiver<NetCommand>,
    events: tokio::sync::mpsc::Sender<NetEventBatch>,
    snapshot: SnapshotSlot,
    cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
}

impl From<HostWiring> for HostWiringParts {
    fn from(wiring: HostWiring) -> Self {
        Self {
            commands: wiring.cmd_rx,
            events: wiring.evt_tx,
            snapshot: wiring.snapshot,
            cap_tx: wiring.cap_tx,
        }
    }
}

/// Use half the server's per-connection request bound so exact demand reads can
/// share the authenticated pool while a sweep is active. QUIC streams provide
/// the correlation envelope, so this fixed window overlaps independent node
/// lookups without adding another wire operation or request identifier.
const INVENTORY_NODE_WINDOW: usize = MAX_INBOUND_REQUESTS_PER_CONNECTION / 2;

struct AdmissionBatcher {
    events: tokio::sync::mpsc::Sender<NetEventBatch>,
    pending: NetEventBatch,
}

impl AdmissionBatcher {
    fn new(events: &tokio::sync::mpsc::Sender<NetEventBatch>) -> Self {
        Self {
            events: events.clone(),
            pending: NetEventBatch::default(),
        }
    }

    async fn push(&mut self, event: NetEvent) -> anyhow::Result<()> {
        if let Err(event) = self.pending.try_push(event) {
            self.flush().await?;
            self.pending
                .try_push(event)
                .expect("an empty admission batch accepts one indivisible event");
        }
        if self.pending.is_full() {
            self.flush().await?;
        }
        Ok(())
    }

    async fn flush(&mut self) -> anyhow::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.events
            .send(std::mem::take(&mut self.pending))
            .await
            .map_err(|_| anyhow::anyhow!("store side stopped during inventory admission"))
    }
}

async fn reconcile_inventory_peer<T: Transport>(
    transport: &T,
    pool: &SharedPool<T::Conn>,
    peer: PeerId,
    team: VerifyingKey,
    connect_proof: &CapabilityProofBundle,
    sync_proof: &CapabilityProofBundle,
    qos: ReconcileQos,
    local: SharedSnapshot,
    events: &tokio::sync::mpsc::Sender<NetEventBatch>,
    candidates: &RoutingCandidates,
) -> anyhow::Result<()> {
    let connection = inventory_pool_get(transport, pool, peer, connect_proof, sync_proof).await?;
    let remote_key = VerifyingKey::from_bytes(&peer)?;
    candidates.lock().unwrap().note(peer);
    let mut admissions = AdmissionBatcher::new(events);
    admissions
        .push(NetEvent::Peer(PeerEvidence::new(team, remote_key)))
        .await?;

    let manifest = tokio::time::timeout(OP_DEADLINE, op_inventory_manifest(&connection, team))
        .await
        .map_err(|_| anyhow::anyhow!("inventory manifest deadline exceeded"))??;
    for component in InventoryComponent::ALL {
        if !qos.traverses(component) {
            continue;
        }
        let advertised = manifest.component(component);
        let mut walker = InventoryWalker::new(team, advertised)?;
        let mut in_flight = FuturesUnordered::new();
        loop {
            while in_flight.len() < INVENTORY_NODE_WINDOW {
                let request = walker.next_request(|component, prefix| {
                    local.lock().unwrap().node_summary(component, prefix)
                })?;
                let Some(request) = request else {
                    break;
                };
                let request_connection = connection.clone();
                in_flight.push(async move {
                    let response = tokio::time::timeout(
                        OP_DEADLINE,
                        op_inventory_node(&request_connection, team, &request),
                    )
                    .await
                    .map_err(|_| anyhow::anyhow!("inventory node deadline exceeded"))??;
                    Ok::<_, anyhow::Error>((request, response))
                });
            }
            let Some(result) = in_flight.next().await else {
                break;
            };
            let (request, response) = result?;
            let missing = walker.accept(&request, response, |component, key| {
                local.lock().unwrap().contains_relative_key(component, key)
            })?;
            if let Some(leaf) = missing {
                let event = remote_leaf_event(&connection, team, advertised, leaf).await?;
                admissions.push(event).await?;
            }
        }
        walker.finish()?;
        // Preserve valid progress if a later component fails. Full batches
        // already stream as they fill; this publishes the bounded tail.
        admissions.flush().await?;
    }
    admissions.flush().await?;
    Ok(())
}

async fn remote_leaf_event<C: Conn>(
    connection: &C,
    team: VerifyingKey,
    advertised: crate::inventory::ComponentManifest,
    leaf: InventoryLeaf,
) -> anyhow::Result<NetEvent> {
    let event = match leaf.value {
        InventoryLeafValue::Peer => {
            let peer: [u8; 32] = leaf
                .key
                .as_slice()
                .try_into()
                .map_err(|_| anyhow::anyhow!("PEER inventory key has the wrong length"))?;
            NetEvent::Peer(PeerEvidence::new(team, VerifyingKey::from_bytes(&peer)?))
        }
        InventoryLeafValue::CollectionRecord(record) => NetEvent::CollectionRecord(record),
        InventoryLeafValue::CapabilityProof(proof) => NetEvent::CapabilityProof(proof),
        InventoryLeafValue::Blob => {
            let hash: RawHash = leaf
                .key
                .as_slice()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Blob inventory key has the wrong length"))?;
            let root = advertised
                .root()
                .ok_or_else(|| anyhow::anyhow!("a missing blob came from an empty manifest"))?;
            let bytes =
                fetch_inventory_blob(connection, root, advertised.leaf_count(), hash).await?;
            NetEvent::Blob { hash, bytes }
        }
    };
    Ok(event)
}

async fn fetch_inventory_blob<C: Conn>(
    connection: &C,
    root: [u8; 32],
    leaf_count: u64,
    hash: RawHash,
) -> anyhow::Result<Bytes> {
    let mut file = tempfile::tempfile()
        .map_err(|error| anyhow::anyhow!("create mirror receive file: {error}"))?;
    let mut hasher = blake3::Hasher::new();
    let mut offset = 0u64;
    let mut total = None;
    loop {
        let request = InventoryBlobRangeRequest::new(
            root,
            leaf_count,
            hash,
            offset,
            BLOB_TRANSFER_CHUNK_BYTES as u32,
        )?;
        let response =
            tokio::time::timeout(OP_DEADLINE, op_inventory_blob_range(connection, request))
                .await
                .map_err(|_| anyhow::anyhow!("inventory blob range deadline exceeded"))??;
        let InventoryBlobRangeResponse::Chunk {
            total_length,
            bytes,
        } = response
        else {
            anyhow::bail!("pinned inventory blob became unavailable")
        };
        if let Some(expected) = total {
            if expected != total_length {
                anyhow::bail!("inventory blob total length changed between ranges");
            }
        } else {
            usize::try_from(total_length)
                .map_err(|_| anyhow::anyhow!("inventory blob does not fit this address space"))?;
            total = Some(total_length);
        }
        if bytes.is_empty() && offset < total_length {
            anyhow::bail!("inventory blob returned an empty nonterminal range");
        }
        file.write_all(&bytes)
            .map_err(|error| anyhow::anyhow!("write mirror receive file: {error}"))?;
        hasher.update(&bytes);
        offset = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("inventory blob offset overflow"))?;
        if offset == total_length {
            break;
        }
        if offset > total_length {
            anyhow::bail!("inventory blob range exceeded advertised total length");
        }
    }
    if hasher.finalize().as_bytes() != &hash {
        anyhow::bail!("inventory blob bytes do not match their authenticated handle");
    }
    let length = usize::try_from(total.unwrap_or(0))?;
    if length == 0 {
        return Ok(Bytes::empty());
    }
    let mapping = unsafe {
        memmap2::MmapOptions::new()
            .len(length)
            .map(&file)
            .map_err(|error| anyhow::anyhow!("map mirror receive file: {error}"))?
    };
    Ok(Bytes::from_source(mapping))
}

#[instrument(level = "info", skip(transport, connect_proof), fields(peer = %hex::encode(&peer[..4])))]
async fn connect_authed<T: Transport>(
    transport: &T,
    peer: PeerId,
    connect_proof: &CapabilityProofBundle,
) -> anyhow::Result<T::Conn> {
    let connection = transport.dial(peer, PILE_SYNC_ALPN).await?;
    op_auth(&connection, connect_proof).await?;
    Ok(connection)
}

async fn pool_entry<C>(pool: &SharedPool<C>, peer: PeerId) -> Arc<PoolEntry<C>> {
    pool.lock()
        .await
        .entry(peer)
        .or_insert_with(|| Arc::new(PoolEntry::default()))
        .clone()
}

async fn pool_get<T: Transport>(
    transport: &T,
    pool: &SharedPool<T::Conn>,
    peer: PeerId,
    connect_proof: &CapabilityProofBundle,
) -> Option<(Arc<PoolEntry<T::Conn>>, T::Conn)> {
    let entry = pool_entry(pool, peer).await;
    let initialized = entry
        .connection
        .get_or_try_init(|| async {
            tokio::time::timeout(
                DIAL_DEADLINE,
                connect_authed(transport, peer, connect_proof),
            )
            .await
            .map_err(|_| anyhow::anyhow!("connection setup deadline exceeded"))?
        })
        .await;
    match initialized {
        Ok(connection) => Some((entry.clone(), connection.clone())),
        Err(error) => {
            debug!(peer = %hex::encode(&peer[..4]), %error, "authenticated dial failed");
            pool_remove_if(pool, peer, &entry).await;
            None
        }
    }
}

async fn inventory_pool_get<T: Transport>(
    transport: &T,
    pool: &SharedPool<T::Conn>,
    peer: PeerId,
    connect_proof: &CapabilityProofBundle,
    sync_proof: &CapabilityProofBundle,
) -> anyhow::Result<T::Conn> {
    let (entry, connection) = pool_get(transport, pool, peer, connect_proof)
        .await
        .ok_or_else(|| anyhow::anyhow!("peer is unavailable"))?;
    let authorized = entry
        .inventory_auth
        .get_or_try_init(|| async {
            tokio::time::timeout(OP_DEADLINE, op_inventory_auth(&connection, sync_proof))
                .await
                .map_err(|_| anyhow::anyhow!("inventory authorization deadline exceeded"))?
        })
        .await;
    if let Err(error) = authorized {
        pool_remove_if(pool, peer, &entry).await;
        return Err(error);
    }
    Ok(connection)
}

async fn pool_remove_if<C: Conn>(pool: &SharedPool<C>, peer: PeerId, expected: &Arc<PoolEntry<C>>) {
    let removed = {
        let mut guard = pool.lock().await;
        if guard
            .get(&peer)
            .is_some_and(|current| Arc::ptr_eq(current, expected))
        {
            guard.remove(&peer)
        } else {
            None
        }
    };
    if let Some(entry) = removed
        && let Some(connection) = entry.connection.get()
    {
        connection.close(0, b"pool evict");
    }
}

async fn pool_evict<C: Conn>(pool: &SharedPool<C>, peer: PeerId) {
    let entry = pool.lock().await.get(&peer).cloned();
    if let Some(entry) = entry {
        pool_remove_if(pool, peer, &entry).await;
    }
}

async fn fetch_from_providers<T: Transport>(
    transport: &T,
    hash: &RawHash,
    pool: &SharedPool<T::Conn>,
    providers: &[PeerId],
    connect_proof: &CapabilityProofBundle,
    sync_proof: &CapabilityProofBundle,
) -> Option<Vec<u8>> {
    for peer in providers.iter().copied() {
        let connection =
            match inventory_pool_get(transport, pool, peer, connect_proof, sync_proof).await {
                Ok(connection) => connection,
                Err(_) => continue,
            };
        match tokio::time::timeout(OP_DEADLINE, op_get_blob(&connection, hash)).await {
            Ok(Ok(Some(bytes))) if blake3::hash(&bytes).as_bytes() == hash => return Some(bytes),
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                debug!(peer = %hex::encode(&peer[..4]), %error, "GET_BLOB failed");
                pool_evict(pool, peer).await;
            }
            Err(_) => {
                pool_evict(pool, peer).await;
            }
        }
    }
    None
}

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: SnapshotSlot,
    snapshots: InventorySnapshots,
    server: InventoryServerConfig,
    events: tokio::sync::mpsc::Sender<NetEventBatch>,
    candidates: RoutingCandidates,
    serve_inventory: bool,
    admit_inbound_peer: bool,
    inbound_connections: Arc<tokio::sync::Semaphore>,
    inbound_requests: Arc<tokio::sync::Semaphore>,
}

#[derive(Clone, Copy)]
enum InventoryAuthorization {
    Unattempted,
    Rejected,
    Authorized(AuthorizedInventorySession),
}

impl SnapshotHandler {
    fn try_admit_connection(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.inbound_connections.clone().try_acquire_owned().ok()
    }

    async fn handle_admitted<T: Transport>(
        &self,
        connection: T::Conn,
        _connection_permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let peer_id = connection.remote_id();
        let span = info_span!("connection", peer = %hex::encode(&peer_id[..4]));
        async move {
            let peer = match VerifyingKey::from_bytes(&peer_id) {
                Ok(peer) => peer,
                Err(error) => {
                    warn!(%error, "invalid transport peer key");
                    connection.close(1, b"invalid peer identity");
                    return;
                }
            };
            let authentication = tokio::time::timeout(INBOUND_AUTH_DEADLINE, async {
                let (mut send, mut recv) = connection.accept_bi().await?;
                let result = authenticate_connection::<T::Conn>(
                    self.server.team(),
                    peer,
                    &mut send,
                    &mut recv,
                )
                .await;
                let _ = send.shutdown().await;
                Some(result)
            })
            .await;
            let connect_expires = match authentication {
                Ok(Some(Ok(Some(verified)))) => verified
                    .effective_validity()
                    .map(|validity| validity.bounds().1),
                Ok(Some(Ok(None))) => {
                    connection.close(1, b"CONNECT capability required");
                    return;
                }
                Ok(Some(Err(error))) => {
                    debug!(%error, "malformed CONNECT stream");
                    connection.close(1, b"malformed authentication");
                    return;
                }
                Ok(None) => return,
                Err(_) => {
                    connection.close(1, b"authentication deadline exceeded");
                    return;
                }
            };

            let authorization = Arc::new(Mutex::new(InventoryAuthorization::Unattempted));
            let per_connection = Arc::new(tokio::sync::Semaphore::new(
                MAX_INBOUND_REQUESTS_PER_CONNECTION,
            ));
            loop {
                let accepted = tokio::select! {
                    stream = connection.accept_bi() => stream,
                    () = wait_until_after(connect_expires) => {
                        connection.close(1, b"CONNECT capability expired");
                        return;
                    }
                    () = tokio::time::sleep(INBOUND_CONNECTION_IDLE_DEADLINE) => {
                        connection.close(0, b"authenticated connection idle timeout");
                        return;
                    }
                };
                let Some((mut send, mut recv)) = accepted else {
                    return;
                };
                if capability_expired(connect_expires) {
                    connection.close(1, b"CONNECT capability expired");
                    return;
                }
                let Ok(connection_permit) = per_connection.clone().try_acquire_owned() else {
                    connection.close(1, b"request concurrency exceeded");
                    return;
                };
                let Ok(global_permit) = self.inbound_requests.clone().try_acquire_owned() else {
                    connection.close(1, b"global request concurrency exceeded");
                    return;
                };
                let handler = self.clone();
                let authorization = authorization.clone();
                tokio::spawn(
                    async move {
                        let operation = tokio::time::timeout(
                            INBOUND_REQUEST_DEADLINE,
                            handler.serve_stream::<T::Conn>(
                                peer,
                                authorization,
                                &mut send,
                                &mut recv,
                            ),
                        )
                        .await;
                        match operation {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => debug!(%error, "direct RPC stream failed"),
                            Err(_) => warn!("direct RPC stream deadline exceeded"),
                        }
                        let _ = send.shutdown().await;
                        drop((connection_permit, global_permit));
                    }
                    .in_current_span(),
                );
            }
        }
        .instrument(span)
        .await;
    }

    async fn serve_stream<C: Conn>(
        &self,
        peer: VerifyingKey,
        authorization: Arc<Mutex<InventoryAuthorization>>,
        send: &mut C::SendHalf,
        recv: &mut C::RecvHalf,
    ) -> anyhow::Result<()> {
        let op = recv_u8(recv).await?;
        let span = debug_span!("stream", op = op_name(op));
        let _entered = span.enter();
        match op {
            OP_GET_BLOB => {
                if !self.serve_inventory {
                    anyhow::bail!("local direction policy does not serve data");
                }
                let _session = current_inventory_session(&authorization)?;
                let hash = recv_hash(recv).await?;
                require_stream_eof(recv).await?;
                let current = self.snapshot.lock().unwrap().as_ref().cloned();
                let bytes = current.and_then(|snapshot| snapshot.lock().unwrap().get_blob(&hash));
                if let Some(bytes) = bytes {
                    send_u64_be(send, bytes.len() as u64).await?;
                    send.write_all(&bytes).await?;
                } else {
                    send_u64_be(send, u64::MAX).await?;
                }
            }
            OP_INVENTORY_AUTH => {
                let proof = recv_inventory_auth_request(recv).await?;
                let already_attempted = {
                    let mut state = authorization.lock().unwrap();
                    let attempted = !matches!(*state, InventoryAuthorization::Unattempted);
                    if !attempted {
                        // Mark attempted before verification so two concurrent
                        // auth streams cannot both install a session.
                        *state = InventoryAuthorization::Rejected;
                    }
                    attempted
                };
                if already_attempted {
                    send_inventory_auth_rejected(send).await?;
                    return Ok(());
                }
                if !self.serve_inventory {
                    send_inventory_auth_rejected(send).await?;
                    return Ok(());
                }
                match self
                    .server
                    .authorize(peer, &proof, crate::clock::epoch_now())
                {
                    Ok(session) => {
                        *authorization.lock().unwrap() =
                            InventoryAuthorization::Authorized(session);
                        if self.admit_inbound_peer {
                            self.candidates.lock().unwrap().note(peer.to_bytes());
                            let _ =
                                self.events
                                    .send(NetEventBatch::singleton(NetEvent::Peer(
                                        PeerEvidence::new(self.server.team(), peer),
                                    )))
                                    .await;
                        }
                        send_inventory_auth_ok(send).await?;
                    }
                    Err(error) => {
                        debug!(%error, "SYNC_TEAM capability rejected");
                        send_inventory_auth_rejected(send).await?;
                    }
                }
            }
            OP_INVENTORY_MANIFEST => {
                if !self.serve_inventory {
                    anyhow::bail!("local direction policy does not serve inventories");
                }
                recv_manifest_request(recv).await?;
                let session = current_inventory_session(&authorization)?;
                let pinned = self
                    .snapshot
                    .lock()
                    .unwrap()
                    .as_ref()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("inventory snapshot unavailable"))?;
                let manifest = {
                    let snapshot = pinned.lock().unwrap();
                    if snapshot.team() != session.team() {
                        anyhow::bail!("current snapshot belongs to another team");
                    }
                    snapshot.manifest()
                };
                self.snapshots
                    .lock()
                    .unwrap()
                    .pin_manifest(session.team(), &manifest, pinned);
                send_manifest(send, &manifest).await?;
            }
            OP_INVENTORY_NODE => {
                if !self.serve_inventory {
                    anyhow::bail!("local direction policy does not serve inventories");
                }
                let session = current_inventory_session(&authorization)?;
                let request = recv_node_request(recv, session).await?;
                let pinned = self.snapshots.lock().unwrap().get(
                    session.team(),
                    request.component,
                    request.root,
                );
                let response = match pinned {
                    None => InventoryNodeResponse::SnapshotUnavailable,
                    Some(pinned) => pinned.lock().unwrap().inventory_node(&request)?,
                };
                send_node_response(send, session.team(), &request, &response).await?;
            }
            OP_INVENTORY_BLOB_RANGE => {
                if !self.serve_inventory {
                    anyhow::bail!("local direction policy does not serve inventories");
                }
                let session = current_inventory_session(&authorization)?;
                let request = recv_blob_range_request(recv).await?;
                let pinned = self.snapshots.lock().unwrap().get(
                    session.team(),
                    InventoryComponent::Blob,
                    request.root,
                );
                let response = match pinned {
                    None => PinnedBlob::SnapshotUnavailable,
                    Some(pinned) => pinned.lock().unwrap().inventory_blob(request),
                };
                match response {
                    PinnedBlob::SnapshotUnavailable => send_blob_snapshot_unavailable(send).await?,
                    PinnedBlob::NotInSnapshot => send_blob_not_in_snapshot(send).await?,
                    PinnedBlob::Found(bytes) => send_blob_range(send, request, &bytes).await?,
                }
            }
            OP_AUTH => anyhow::bail!("OP_AUTH may only appear on the first stream"),
            _ => anyhow::bail!("unknown direct RPC operation {op:#x}"),
        }
        Ok(())
    }
}

fn current_inventory_session(
    authorization: &Mutex<InventoryAuthorization>,
) -> anyhow::Result<AuthorizedInventorySession> {
    let mut state = authorization.lock().unwrap();
    let InventoryAuthorization::Authorized(session) = *state else {
        anyhow::bail!("SYNC_TEAM authorization is required")
    };
    if !session.is_current_at(crate::clock::epoch_now()) {
        *state = InventoryAuthorization::Rejected;
        anyhow::bail!("SYNC_TEAM capability expired")
    }
    Ok(session)
}

async fn authenticate_connection<C: Conn>(
    team: VerifyingKey,
    peer: VerifyingKey,
    send: &mut C::SendHalf,
    recv: &mut C::RecvHalf,
) -> anyhow::Result<Option<triblespace_core::capability::VerifiedCapability>> {
    use triblespace_core::capability::{CapabilityMode, CapabilityRequest};

    let verdict = async {
        let op = recv_u8(recv).await?;
        if op != OP_AUTH {
            anyhow::bail!("first stream must be OP_AUTH");
        }
        let proof = recv_capability_proof_bundle(recv).await?;
        require_stream_eof(recv).await?;
        Ok(proof.verify(
            team,
            crate::clock::epoch_now(),
            peer,
            CapabilityRequest::new(connect_capability_atom(team), CapabilityMode::Invoke),
        )?)
    }
    .await;
    match verdict {
        Ok(verified) => {
            send_u8(send, AUTH_OK).await?;
            Ok(Some(verified))
        }
        Err(error) => {
            debug!(%error, "CONNECT capability rejected");
            send_u8(send, AUTH_REJECTED).await?;
            Ok(None)
        }
    }
}

async fn require_stream_eof<R: tokio::io::AsyncRead + Unpin>(recv: &mut R) -> anyhow::Result<()> {
    let mut trailing = [0u8; 1];
    if recv.read(&mut trailing).await? != 0 {
        anyhow::bail!("request contains trailing bytes");
    }
    Ok(())
}

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
        let remaining = upper_ns.saturating_sub(now_ns).saturating_add(1);
        let nanos = u64::try_from(remaining)
            .unwrap_or(u64::MAX)
            .min(MAX_SLEEP_NS);
        tokio::time::sleep(std::time::Duration::from_nanos(nanos)).await;
    }
}

fn op_name(op: u8) -> &'static str {
    match op {
        OP_AUTH => "AUTH",
        OP_GET_BLOB => "GET_BLOB",
        OP_INVENTORY_AUTH => "INVENTORY_AUTH",
        OP_INVENTORY_MANIFEST => "INVENTORY_MANIFEST",
        OP_INVENTORY_NODE => "INVENTORY_NODE",
        OP_INVENTORY_BLOB_RANGE => "INVENTORY_BLOB_RANGE",
        _ => "UNKNOWN",
    }
}

/// Production relay defaults with trailing-dot hostnames normalized for HTTP
/// intermediaries that reject absolute-FQDN Host headers.
pub(crate) fn dot_stripped_default_relay_map() -> iroh::RelayMap {
    let original = iroh::defaults::prod::default_relay_map();
    let urls: Vec<String> = original
        .urls::<Vec<_>>()
        .into_iter()
        .map(|relay| {
            let mut url: url::Url = relay.into();
            if let Some(host) = url.host_str().and_then(|host| host.strip_suffix('.')) {
                let host = host.to_owned();
                let _ = url.set_host(Some(&host));
            }
            url.to_string()
        })
        .collect();
    iroh::RelayMap::try_from_iter(urls.iter().map(String::as_str))
        .expect("normalized default relay URLs remain valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> VerifyingKey {
        SigningKey::from_bytes(&[byte; 32]).verifying_key()
    }

    #[test]
    fn wake_is_exactly_team_scoped_and_versioned() {
        let team = key(1);
        let generation = InventoryGeneration::from_bytes([7; 32]);
        let frame = inventory_wake_frame(team, generation);
        assert_eq!(decode_inventory_wake_frame(&frame, team), Some(generation));
        assert_eq!(decode_inventory_wake_frame(&frame, key(2)), None);

        let mut malformed = frame;
        malformed[4] ^= 1;
        assert_eq!(decode_inventory_wake_frame(&malformed, team), None);
    }

    #[test]
    fn routing_evidence_never_replaces_bootstraps() {
        let mut routes = RoutingTable::new(vec![[1; 32], [2; 32]]);
        routes.note([3; 32]);
        routes.note([1; 32]);
        assert_eq!(routes.candidates([9; 32]), vec![[1; 32], [2; 32], [3; 32]]);
    }

    #[test]
    fn routing_evidence_is_not_truncated_at_an_arbitrary_peer_count() {
        let mut routes = RoutingTable::new(vec![[1; 32]]);
        for byte in 2..=100 {
            routes.note([byte; 32]);
        }
        assert_eq!(routes.candidates([0; 32]).len(), 100);
    }
}
