//! Collection-scoped network host.
//!
//! TLS authenticates endpoint identities, but establishing a transport
//! connection grants no team or collection authority. Each semantic repair is
//! one stream whose request carries complete READ(C) evidence. DHT routing,
//! provider-directory operations discover collection participants through an
//! opaque KDF(C). Exact bytes stay inside the admitted collection stream.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use anybytes::Bytes;
use ed25519_dalek::{SigningKey, VerifyingKey};
use futures::{StreamExt as _, stream::FuturesUnordered};
use iroh_base::{EndpointAddr, EndpointId};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tracing::{Instrument as _, debug, debug_span, info_span, warn};
use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::{UnknownBlob, simplearchive::SimpleArchive};
use triblespace_core::capability::CapabilityProofBundle;
use triblespace_core::collection::{CollectionHandle, CollectionPolicy, CollectionRecord};
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{Entry as PatchEntry, IdentitySchema, PATCH};
use triblespace_core::repo::{BlobChildren, BlobStoreGet, StoreChanges, StoreRead};

use crate::channel::{NetCommand, NetEvent, NetEventBatch, SnapshotNotice};
use crate::collection_activation::{
    CollectionActivationOverlay, CollectionActivationOverlayError, CollectionReadEvidenceError,
    collection_activation_overlay_at, collection_read_evidence_bundles_at,
};
use crate::collection_session::{
    DisclosureForestPatch, FullReplicaCursor, FullReplicaState,
    fetch_collection_blob as fetch_collection_blob_session, pull_collection, serve_collection_blob,
    serve_collection_repair,
};
use crate::collection_wire::{
    MAX_COLLECTION_READ_BUNDLES, OP_COLLECTION_BLOB, OP_COLLECTION_REPAIR,
};
use crate::identity::iroh_secret;
use crate::inventory::{BlobReplication, ReconcileQos};
use crate::patch_repair::PatchSummary;
use crate::protocol::{
    OP_FIND_NODE, OP_GET_BLOB, OP_PROVIDER_GET, OP_PROVIDER_PUT, PILE_SYNC_ALPN, PROVIDER_PUT_FULL,
    PROVIDER_PUT_OK, RawHash, op_find_node, op_get_blob, op_provider_get, op_provider_put,
    recv_hash, recv_u8, send_hash, send_u8, send_u64_be,
};
use crate::provider::{
    PROVIDER_LEASE_LIFETIME, ProviderDirectory, ProviderKey, ProviderObservation, ProviderToken,
    collection_provider_key, collection_provider_token,
};
use crate::routing::{ALPHA, IterativeLookup, K, RoutingKey, RoutingTable};
use crate::transport::{Conn, Harness, PeerId, Transport};
use crate::wake::{
    CollectionWakeEvent, CollectionWakeNetwork, CollectionWakePlane, CollectionWakeRoot,
    CollectionWakeSubscription, ReceivedCollectionWake,
};

/// Ephemeral local collection interest. It is deliberately not a durable
/// marker or ambient registry.
pub(crate) type ActiveCollections = PATCH<32, IdentitySchema>;

/// Transport and local scheduling configuration.
///
/// Collection authority is intentionally absent. A connection is ordinary
/// mutually authenticated TLS; READ authority is supplied on each collection
/// repair stream.
#[derive(Clone)]
pub struct PeerConfig {
    /// Bootstrap endpoint routes.
    pub peers: Vec<EndpointAddr>,
    /// Local pull/serve scheduling choices. Never sent as authority.
    pub qos: ReconcileQos,
}

trait BlobSnapshotReader: Send + Sync + 'static {
    fn get_blob(&self, hash: RawHash) -> Option<Bytes>;
}

struct CloneableBlobSnapshotReader<R>(Mutex<R>);

impl<R> BlobSnapshotReader for CloneableBlobSnapshotReader<R>
where
    R: BlobStoreGet + Clone + Send + 'static,
{
    fn get_blob(&self, hash: RawHash) -> Option<Bytes> {
        let reader = self.0.lock().unwrap().clone();
        reader
            .get::<Bytes, UnknownBlob>(Inline::<Handle<UnknownBlob>>::new(hash))
            .ok()
    }
}

/// One collection's immutable server overlay plus the request evidence this
/// endpoint will present when it pulls the same collection.
pub(crate) struct CollectionSnapshot {
    activation: Arc<CollectionActivationOverlay>,
    read_evidence: Arc<[CapabilityProofBundle]>,
    full: Arc<FullReplicaState>,
    blobs: Arc<dyn BlobSnapshotReader>,
}

impl CollectionSnapshot {
    pub(crate) fn collection(&self) -> CollectionHandle {
        self.activation.collection()
    }

    fn wake_root(&self) -> [u8; 32] {
        self.activation.wake_root()
    }
}

type CollectionSnapshotIndex = PATCH<32, IdentitySchema, Arc<CollectionSnapshot>>;

fn build_full_replica_state<R>(
    snapshot: &R,
    activation: &CollectionActivationOverlay,
) -> FullReplicaState
where
    R: StoreRead,
{
    let mut direct_roots = HashSet::new();
    direct_roots.insert(activation.collection().raw);
    for record in activation.records().records() {
        let CollectionRecord::Commit(commit) = record else {
            continue;
        };
        direct_roots.insert(commit.data().raw);
        direct_roots.insert(commit.metadata().raw);
    }
    let mut forest = DisclosureForestPatch::new();
    let mut visited = HashSet::new();
    let mut level = direct_roots.iter().copied().collect::<Vec<_>>();
    level.sort_unstable();
    level.retain(|handle| {
        snapshot
            .contains_blob(Inline::<Handle<UnknownBlob>>::new(*handle))
            .unwrap_or(false)
    });
    for handle in &level {
        let mut key = [0; 80];
        key[8..40].copy_from_slice(handle);
        key[40..48].copy_from_slice(&u64::MAX.to_be_bytes());
        key[48..].copy_from_slice(handle);
        forest.insert(&PatchEntry::new(&key));
        visited.insert(*handle);
    }
    let mut depth = 0_u64;
    while !level.is_empty() {
        let mut next = BTreeMap::<[u8; 32], ([u8; 32], u64)>::new();
        for parent in &level {
            let Ok(bytes) = snapshot.get::<Bytes, UnknownBlob>(Inline::new(*parent)) else {
                continue;
            };
            for (index, chunk) in bytes.chunks_exact(32).enumerate() {
                let child: [u8; 32] = chunk.try_into().expect("fixed-width chunk");
                if visited.contains(&child)
                    || !snapshot
                        .contains_blob(Inline::<Handle<UnknownBlob>>::new(child))
                        .unwrap_or(false)
                {
                    continue;
                }
                next.entry(child).or_insert((*parent, index as u64));
            }
        }
        let Some(next_depth) = depth.checked_add(1) else {
            break;
        };
        depth = next_depth;
        level = Vec::with_capacity(next.len());
        for (child, (parent, index)) in next {
            let mut key = [0; 80];
            key[..8].copy_from_slice(&depth.to_be_bytes());
            key[8..40].copy_from_slice(&parent);
            key[40..48].copy_from_slice(&index.to_be_bytes());
            key[48..].copy_from_slice(&child);
            forest.insert(&PatchEntry::new(&key));
            visited.insert(child);
            level.push(child);
        }
    }
    FullReplicaState {
        forest,
        direct_roots,
    }
}

/// Immutable host observation indexed exactly by active collection handle.
///
/// The semantic state of each value is the product constructed by
/// `CollectionActivationOverlay`: record PATCH × portable WRITE-evidence
/// PATCH. No global team inventory, proof list, or blob manifest is retained.
pub(crate) struct StoreSnapshot {
    collections: CollectionSnapshotIndex,
    blobs: Arc<dyn BlobSnapshotReader>,
    observed_at: hifitime::Epoch,
    next_authorization_change: Option<hifitime::Epoch>,
}

impl StoreSnapshot {
    pub(crate) fn from_store_changes<R>(
        snapshot: R,
        active: &ActiveCollections,
        local: VerifyingKey,
        previous: Option<&Self>,
        changes: StoreChanges,
        authorization_changed: bool,
        full_replication: bool,
        next_authorization_change: Option<hifitime::Epoch>,
        instant: hifitime::Epoch,
    ) -> anyhow::Result<Self>
    where
        R: StoreRead + BlobChildren + Clone,
    {
        let mut collections = CollectionSnapshotIndex::new();
        let blob_reader: Arc<dyn BlobSnapshotReader> =
            Arc::new(CloneableBlobSnapshotReader(Mutex::new(snapshot.clone())));
        let activation_inputs_changed = changes.contains(StoreChanges::BLOBS)
            || changes.contains(StoreChanges::COLLECTION_RECORDS)
            || changes.contains(StoreChanges::CAPABILITY_PROOFS)
            || authorization_changed;
        for raw in active.iter_ordered() {
            let collection = CollectionHandle::new(*raw);
            if snapshot
                .get::<Blob<SimpleArchive>, SimpleArchive>(Inline::new(collection.raw))
                .is_err()
            {
                warn!(collection = %hex::encode(&collection.raw[..4]), "active collection descriptor unavailable; isolating pending collection");
                continue;
            }
            let prior = previous.and_then(|prior| prior.collection(collection));
            let activation_result = if !activation_inputs_changed {
                prior
                    .as_ref()
                    .map(|prior| prior.activation.clone())
                    .map_or_else(
                        || {
                            collection_activation_overlay_at(&snapshot, collection, instant)
                                .map(Arc::new)
                        },
                        Ok,
                    )
            } else {
                collection_activation_overlay_at(&snapshot, collection, instant).map(|fresh| {
                    prior
                        .as_ref()
                        .filter(|prior| prior.wake_root() == fresh.wake_root())
                        .map_or_else(|| Arc::new(fresh), |prior| prior.activation.clone())
                })
            };
            let activation = match activation_result {
                Ok(activation) => activation,
                Err(CollectionActivationOverlayError::Descriptor(error)) => {
                    warn!(collection = %hex::encode(&collection.raw[..4]), %error, "active collection descriptor is unavailable or invalid; isolating collection");
                    continue;
                }
                Err(error) => return Err(anyhow::Error::new(error)),
            };
            let read_evidence = if !activation_inputs_changed
                && !authorization_changed
                && prior.is_some()
            {
                prior.as_ref().unwrap().read_evidence.clone()
            } else {
                match collection_read_evidence_bundles_at(
                    &snapshot,
                    collection,
                    local,
                    MAX_COLLECTION_READ_BUNDLES,
                    instant,
                ) {
                    Ok(evidence) => evidence.into(),
                    Err(CollectionReadEvidenceError::TooMany { count, limit }) => {
                        warn!(
                            collection = %hex::encode(&collection.raw[..4]),
                            count,
                            limit,
                            "collection READ witness exceeds network bound; collection remains locally active but cannot be presented remotely"
                        );
                        Arc::from([])
                    }
                    Err(error) => return Err(anyhow::Error::new(error)),
                }
            };
            let full_inputs_changed = changes.contains(StoreChanges::BLOBS)
                || changes.contains(StoreChanges::COLLECTION_RECORDS)
                || changes.contains(StoreChanges::CAPABILITY_PROOFS)
                || authorization_changed;
            let full = if !full_replication {
                Arc::new(FullReplicaState {
                    forest: DisclosureForestPatch::new(),
                    direct_roots: HashSet::new(),
                })
            } else if !full_inputs_changed && prior.is_some() {
                prior.as_ref().unwrap().full.clone()
            } else {
                Arc::new(build_full_replica_state(&snapshot, &activation))
            };
            let value = Arc::new(CollectionSnapshot {
                activation,
                read_evidence,
                full,
                blobs: blob_reader.clone(),
            });
            collections.insert(&PatchEntry::with_value(raw, value));
        }
        Ok(Self {
            collections,
            blobs: blob_reader,
            observed_at: instant,
            next_authorization_change,
        })
    }

    fn time_valid(&self) -> bool {
        let now = crate::clock::epoch_now();
        now >= self.observed_at
            && !self
                .next_authorization_change
                .is_some_and(|boundary| now >= boundary)
    }

    fn collection(&self, collection: CollectionHandle) -> Option<Arc<CollectionSnapshot>> {
        self.time_valid()
            .then(|| self.collections.get(&collection.raw).cloned())
            .flatten()
    }

    pub(crate) fn collections(&self) -> impl Iterator<Item = Arc<CollectionSnapshot>> + '_ {
        let valid = self.time_valid();
        self.collections
            .iter_ordered()
            .filter_map(move |key| valid.then(|| self.collections.get(key).cloned()).flatten())
    }

    fn notices(&self) -> Vec<(CollectionHandle, [u8; 32], [u8; 32])> {
        self.collections()
            .map(|collection| {
                (
                    collection.collection(),
                    collection.wake_root(),
                    PatchSummary::from_patch(&collection.full.forest)
                        .root()
                        .unwrap_or([0; 32]),
                )
            })
            .collect()
    }

    fn get_blob(&self, hash: &RawHash) -> Option<Bytes> {
        self.blobs.get_blob(*hash)
    }

    fn get_bearer_blob(&self, hash: RawHash) -> Option<Bytes> {
        self.get_blob(&hash)
    }
}

type SharedSnapshot = Arc<StoreSnapshot>;
type SnapshotSlot = Arc<Mutex<Option<SharedSnapshot>>>;
type OperationalBlobSlot = Arc<Mutex<Option<Arc<dyn BlobSnapshotReader>>>>;

/// The async capability cloned into lazy readers.
pub(crate) trait NetCapability: Send + Sync {
    fn fetch_collection_blob(
        &self,
        collection: CollectionHandle,
        policy: CollectionPolicy,
        hash: RawHash,
    ) -> futures::future::BoxFuture<'static, Option<Bytes>>;
}

type RoutingCandidates = Arc<Mutex<RoutingTable>>;
type CollectionParticipants = Arc<Mutex<HashMap<[u8; 32], HashMap<PeerId, crate::clock::Mono>>>>;

const MAX_COLLECTION_PARTICIPANTS: usize = 128;
const COLLECTION_PARTICIPANT_LEASE: std::time::Duration = std::time::Duration::from_secs(5 * 60);
const PERIODIC_REPAIR_SAMPLE: usize = 8;

fn observe_participant(
    participants: &mut HashMap<[u8; 32], HashMap<PeerId, crate::clock::Mono>>,
    collection: [u8; 32],
    peer: PeerId,
    now: crate::clock::Mono,
) {
    let peers = participants.entry(collection).or_default();
    peers.retain(|_, seen| now.duration_since(*seen) <= COLLECTION_PARTICIPANT_LEASE);
    if !peers.contains_key(&peer)
        && peers.len() >= MAX_COLLECTION_PARTICIPANTS
        && let Some(oldest) = peers
            .iter()
            .min_by_key(|(peer, seen)| (**seen, **peer))
            .map(|(peer, _)| *peer)
    {
        peers.remove(&oldest);
    }
    peers.insert(peer, now);
}

fn live_participants(
    participants: &mut HashMap<[u8; 32], HashMap<PeerId, crate::clock::Mono>>,
    collection: [u8; 32],
    now: crate::clock::Mono,
) -> Vec<PeerId> {
    let Some(peers) = participants.get_mut(&collection) else {
        return Vec::new();
    };
    peers.retain(|_, seen| now.duration_since(*seen) <= COLLECTION_PARTICIPANT_LEASE);
    let mut live = peers.keys().copied().collect::<Vec<_>>();
    live.sort_unstable();
    live
}

struct PoolEntry<C> {
    connection: tokio::sync::OnceCell<Result<C, Arc<anyhow::Error>>>,
}

impl<C> Default for PoolEntry<C> {
    fn default() -> Self {
        Self {
            connection: tokio::sync::OnceCell::new(),
        }
    }
}

#[derive(Clone)]
struct PooledConnection<C> {
    entry: Arc<PoolEntry<C>>,
    connection: C,
}

impl<C> PooledConnection<C> {
    fn conn(&self) -> &C {
        &self.connection
    }
}

struct ConnectionPool<C> {
    entries: HashMap<PeerId, Arc<PoolEntry<C>>>,
    least_to_most_recent: VecDeque<PeerId>,
}

impl<C> ConnectionPool<C> {
    fn entry(&mut self, peer: PeerId) -> Arc<PoolEntry<C>> {
        self.entries
            .entry(peer)
            .or_insert_with(|| Arc::new(PoolEntry::default()))
            .clone()
    }

    fn admit(&mut self, peer: PeerId, expected: &Arc<PoolEntry<C>>) -> Option<Arc<PoolEntry<C>>> {
        if !self
            .entries
            .get(&peer)
            .is_some_and(|current| Arc::ptr_eq(current, expected))
        {
            return None;
        }
        self.least_to_most_recent
            .retain(|candidate| *candidate != peer);
        self.least_to_most_recent.push_back(peer);
        if self.least_to_most_recent.len() <= MAX_CONNECTIONS {
            return None;
        }
        let oldest = self.least_to_most_recent.pop_front().unwrap();
        self.entries.remove(&oldest)
    }

    fn remove_if(
        &mut self,
        peer: PeerId,
        expected: &Arc<PoolEntry<C>>,
    ) -> Option<Arc<PoolEntry<C>>> {
        if !self
            .entries
            .get(&peer)
            .is_some_and(|current| Arc::ptr_eq(current, expected))
        {
            return None;
        }
        self.least_to_most_recent
            .retain(|candidate| *candidate != peer);
        self.entries.remove(&peer)
    }
}

type SharedPool<C> = Arc<Mutex<ConnectionPool<C>>>;

fn new_shared_pool<C>() -> SharedPool<C> {
    Arc::new(Mutex::new(ConnectionPool {
        entries: HashMap::new(),
        least_to_most_recent: VecDeque::new(),
    }))
}

async fn pool_get<T: Transport>(
    transport: &T,
    pool: &SharedPool<T::Conn>,
    peer: PeerId,
) -> anyhow::Result<PooledConnection<T::Conn>> {
    let entry = pool.lock().unwrap().entry(peer);
    let initialized = entry
        .connection
        .get_or_init(|| async {
            tokio::time::timeout(DIAL_DEADLINE, transport.dial(peer, PILE_SYNC_ALPN))
                .await
                .map_err(|_| anyhow::anyhow!("connection setup deadline exceeded"))
                .and_then(|result| result)
                .and_then(|connection| {
                    if connection.remote_id() != peer {
                        anyhow::bail!("dialed endpoint identity does not match requested peer")
                    }
                    Ok(connection)
                })
                .map_err(Arc::new)
        })
        .await;
    let connection = match initialized {
        Ok(connection) => connection.clone(),
        Err(error) => {
            pool.lock().unwrap().remove_if(peer, &entry);
            return Err(anyhow::anyhow!(error.to_string()));
        }
    };
    drop(pool.lock().unwrap().admit(peer, &entry));
    Ok(PooledConnection { entry, connection })
}

fn pool_invalidate<C: Conn>(pool: &SharedPool<C>, peer: PeerId, entry: &Arc<PoolEntry<C>>) {
    let removed = pool.lock().unwrap().remove_if(peer, entry);
    if removed.is_some()
        && let Some(Ok(connection)) = entry.connection.get()
    {
        connection.close(0, b"pool evict");
    }
}

#[derive(Clone)]
struct ProviderClient<T: Transport> {
    transport: T,
    pool: SharedPool<T::Conn>,
    providers: Arc<Mutex<ProviderDirectory>>,
    candidates: RoutingCandidates,
    participants: CollectionParticipants,
    my_id: PeerId,
}

struct NetCap<T: Transport> {
    client: ProviderClient<T>,
    can_fetch: bool,
}

impl<T: Transport> NetCapability for NetCap<T> {
    fn fetch_collection_blob(
        &self,
        collection: CollectionHandle,
        policy: CollectionPolicy,
        hash: RawHash,
    ) -> futures::future::BoxFuture<'static, Option<Bytes>> {
        let client = self.client.clone();
        let can_fetch = self.can_fetch;
        Box::pin(async move {
            if !can_fetch {
                return None;
            }
            client.fetch_collection_blob(collection, policy, hash).await
        })
    }
}

/// Default end-to-end budget for an interactive exact blob read.
pub const INTERACTIVE_FETCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Clone)]
pub struct NetSender {
    cmd_tx: mpsc::Sender<NetCommand>,
    snapshot: SnapshotSlot,
    operational_blobs: OperationalBlobSlot,
    cap: tokio::sync::watch::Receiver<Option<Arc<dyn NetCapability>>>,
    id: EndpointId,
}

impl NetSender {
    pub fn id(&self) -> EndpointId {
        self.id
    }

    pub(crate) fn current_snapshot(&self) -> Option<SharedSnapshot> {
        self.snapshot.lock().unwrap().clone()
    }

    pub(crate) fn update_operational_blobs<R>(&self, snapshot: R)
    where
        R: BlobStoreGet + Clone + Send + 'static,
    {
        *self.operational_blobs.lock().unwrap() =
            Some(Arc::new(CloneableBlobSnapshotReader(Mutex::new(snapshot))));
    }

    pub(crate) fn update_snapshot(&self, snapshot: StoreSnapshot, active: &ActiveCollections) {
        let mut notices = snapshot.notices();
        for raw in active.iter_ordered() {
            if !notices
                .iter()
                .any(|(collection, _, _)| collection.raw == *raw)
            {
                notices.push((CollectionHandle::new(*raw), [0; 32], [0; 32]));
            }
        }
        let retired = self.snapshot.lock().unwrap().replace(Arc::new(snapshot));
        drop(retired);
        let _ = self
            .cmd_tx
            .send(NetCommand::SnapshotChanged(SnapshotNotice {
                collections: notices,
                installed: true,
            }));
    }

    pub(crate) fn update_providers(&self, providers: ProviderObservation) {
        let _ = self.cmd_tx.send(NetCommand::ProvidersUpdated(providers));
    }

    pub fn clear_snapshot(&self) {
        let had_snapshot = self.snapshot.lock().unwrap().take().is_some();
        if had_snapshot {
            let _ = self
                .cmd_tx
                .send(NetCommand::SnapshotChanged(SnapshotNotice {
                    collections: Vec::new(),
                    installed: false,
                }));
        }
        self.update_providers(ProviderObservation::default());
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

    pub async fn fetch_collection_blob(
        &self,
        collection: CollectionHandle,
        hash: RawHash,
        budget: std::time::Duration,
    ) -> Option<Bytes> {
        let policy = self
            .current_snapshot()?
            .collection(collection)?
            .activation
            .policy()
            .clone();
        tokio::time::timeout(budget, async {
            self.ready_capability()
                .await
                .ok()?
                .fetch_collection_blob(collection, policy, hash)
                .await
        })
        .await
        .ok()
        .flatten()
    }
}

pub struct NetReceiver {
    evt_rx: tokio::sync::mpsc::Receiver<NetEventBatch>,
}

impl NetReceiver {
    pub(crate) fn try_recv(&mut self) -> Option<NetEventBatch> {
        self.evt_rx.try_recv().ok()
    }
}

pub struct HostWiring {
    cmd_rx: mpsc::Receiver<NetCommand>,
    evt_tx: tokio::sync::mpsc::Sender<NetEventBatch>,
    snapshot: SnapshotSlot,
    operational_blobs: OperationalBlobSlot,
    cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
}

pub fn wire(id: EndpointId) -> (NetSender, NetReceiver, HostWiring) {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (evt_tx, evt_rx) = tokio::sync::mpsc::channel(crate::channel::MAX_ADMISSION_BRIDGE_BATCHES);
    let snapshot = Arc::new(Mutex::new(None));
    let operational_blobs = Arc::new(Mutex::new(None));
    let (cap_tx, cap_rx) = tokio::sync::watch::channel(None);
    (
        NetSender {
            cmd_tx,
            snapshot: snapshot.clone(),
            operational_blobs: operational_blobs.clone(),
            cap: cap_rx,
            id,
        },
        NetReceiver { evt_rx },
        HostWiring {
            cmd_rx,
            evt_tx,
            snapshot,
            operational_blobs,
            cap_tx,
        },
    )
}

pub async fn run_host<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    host_loop(harness, config, wiring).await;
}

pub fn spawn(
    key: SigningKey,
    config: PeerConfig,
) -> anyhow::Result<(NetSender, NetReceiver, CollectionWakePlane)> {
    let secret = iroh_secret(&key);
    let id: EndpointId = secret.public().into();
    let (sender, receiver, wiring) = wire(id);
    let (startup_tx, startup_rx) = mpsc::sync_channel(1);
    thread::Builder::new()
        .name("triblespace-net".to_owned())
        .spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = startup_tx.send(Err(anyhow::Error::new(error)));
                    return;
                }
            };
            runtime.block_on(async move {
                let harness = match crate::transport::iroh::bind(secret, &config).await {
                    Ok(harness) => harness,
                    Err(error) => {
                        let _ = startup_tx.send(Err(error));
                        return;
                    }
                };
                let wake_plane = harness.transport.wake_plane();
                if startup_tx.send(Ok(wake_plane)).is_ok() {
                    run_host(harness, config, wiring).await;
                }
            });
        })?;
    let wake_plane = startup_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("network host stopped during startup"))??;
    Ok((sender, receiver, wake_plane))
}

const DIAL_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
const OP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
const REPAIR_DEADLINE: std::time::Duration = std::time::Duration::from_secs(300);
const REPAIR_PERIOD: std::time::Duration = std::time::Duration::from_secs(30);
const HOST_POLL_PERIOD: std::time::Duration = std::time::Duration::from_millis(10);
const CONNECTION_IDLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);
const REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(300);
const MAX_CONNECTIONS: usize = 64;
const MAX_REQUESTS_PER_CONNECTION: usize = 16;
const MAX_REQUESTS_GLOBAL: usize = 16;
const MAX_CONCURRENT_REPAIRS: usize = 8;
const MAX_PENDING_REPAIRS: usize = 512;
const PROVIDER_RENEWAL_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(PROVIDER_LEASE_LIFETIME.as_secs() / 2);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RepairTarget {
    collection: CollectionHandle,
    peer: PeerId,
}

struct RepairOutcome {
    target: RepairTarget,
    success: bool,
    more: bool,
    full_cursor: Option<FullReplicaCursor>,
}

fn enqueue_repair(
    queue: &mut VecDeque<RepairTarget>,
    pending: &mut HashSet<RepairTarget>,
    target: RepairTarget,
) {
    if pending.len() < MAX_PENDING_REPAIRS && pending.insert(target) {
        queue.push_back(target);
    }
}

enum WakeCommand {
    Observe(CollectionWakeRoot),
    Join(Vec<EndpointId>),
    Shutdown,
}

enum WakeNotice {
    Received {
        collection: CollectionHandle,
        received: ReceivedCollectionWake,
    },
    Lagged,
}

fn spawn_wake_topic<P: CollectionWakeNetwork>(
    plane: P,
    collection: CollectionHandle,
    bootstrap: Vec<EndpointId>,
    notices: tokio::sync::mpsc::Sender<WakeNotice>,
) -> tokio::sync::mpsc::UnboundedSender<WakeCommand> {
    let (commands, mut command_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut current_root = None;
        loop {
            let mut topic = loop {
                match plane.subscribe_network(collection, bootstrap.clone()).await {
                    Ok(topic) => break topic,
                    Err(error) => {
                        debug!(%error, "collection gossip subscription failed; retrying");
                        tokio::select! {
                            command = command_rx.recv() => match command {
                                Some(WakeCommand::Observe(root)) => current_root = Some(root),
                                Some(WakeCommand::Join(_)) => {}
                                Some(WakeCommand::Shutdown) | None => return,
                            },
                            () = tokio::time::sleep(crate::RETRY_BACKOFF_BASE) => {}
                        }
                    }
                }
            };
            if let Some(root) = current_root
                && let Err(error) = topic.broadcast_wake(root).await
            {
                debug!(%error, "collection wake broadcast after subscribe failed");
            }
            'events: loop {
                tokio::select! {
                    command = command_rx.recv() => match command {
                        Some(WakeCommand::Observe(root)) => {
                            current_root = Some(root);
                            if let Err(error) = topic.broadcast_wake(root).await {
                                debug!(%error, "collection wake broadcast failed");
                            }
                        }
                        Some(WakeCommand::Join(peers)) => {
                            if let Err(error) = topic.join_wake_peers(peers).await {
                                debug!(%error, "joining DHT-discovered collection wake peers failed");
                            }
                        }
                        Some(WakeCommand::Shutdown) | None => return,
                    },
                    event = topic.next_wake_event() => match event {
                        Ok(Some(CollectionWakeEvent::Received(received))) => {
                        // Wakes are repeatable hints. Dropping one under load
                        // preserves correctness while keeping a nonce flood
                        // behind a hard process-wide memory bound.
                        let _ = notices.try_send(WakeNotice::Received { collection, received });
                        }
                        Ok(Some(CollectionWakeEvent::Lagged)) => {
                            let _ = notices.try_send(WakeNotice::Lagged);
                            if let Some(root) = current_root {
                                let _ = topic.broadcast_wake(root).await;
                            }
                        }
                        Ok(Some(CollectionWakeEvent::Rejected { error, .. })) => {
                            debug!(%error, "rejected invalid collection wake");
                        }
                        Ok(Some(CollectionWakeEvent::NeighborUp(_))) => {
                            if let Some(root) = current_root
                                && let Err(error) = topic.broadcast_wake(root).await
                            {
                                debug!(%error, "collection wake rebroadcast failed");
                            }
                        }
                        Ok(Some(CollectionWakeEvent::NeighborDown(_))) => {}
                        Ok(None) => {
                            debug!("collection wake subscription ended; retrying");
                            break 'events;
                        },
                        Err(error) => {
                            debug!(%error, "collection wake subscription failed; retrying");
                            break 'events;
                        }
                    }
                }
            }
        }
    });
    commands
}

async fn host_loop<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    let Harness {
        transport,
        mut incoming,
    } = harness;
    let my_id = transport.local_id();
    let configured: Vec<_> = config
        .peers
        .iter()
        .map(|address| *address.id.as_bytes())
        .filter(|peer| *peer != my_id)
        .collect();
    let candidates = Arc::new(Mutex::new(RoutingTable::new(
        my_id,
        configured.iter().copied(),
    )));
    let participants = Arc::new(Mutex::new(HashMap::new()));
    let pool = new_shared_pool();
    let providers = Arc::new(Mutex::new(ProviderDirectory::default()));
    let provider_client = ProviderClient {
        transport: transport.clone(),
        pool: pool.clone(),
        providers: providers.clone(),
        candidates: candidates.clone(),
        participants: participants.clone(),
        my_id,
    };
    let cap = Arc::new(NetCap {
        client: provider_client.clone(),
        can_fetch: config.qos.direction.pulls(),
    });
    let _ = wiring.cap_tx.send(Some(cap as Arc<dyn NetCapability>));

    let handler = SnapshotHandler {
        snapshot: wiring.snapshot.clone(),
        candidates: candidates.clone(),
        providers: providers.clone(),
        serve_data: config.qos.direction.serves(),
        inbound_connections: Arc::new(tokio::sync::Semaphore::new(MAX_CONNECTIONS)),
        inbound_requests: Arc::new(tokio::sync::Semaphore::new(MAX_REQUESTS_GLOBAL)),
    };
    tokio::spawn(async move {
        while let Some(accepted) = incoming.recv().await {
            if accepted.alpn != PILE_SYNC_ALPN {
                accepted.conn.close(1, b"unknown protocol");
                continue;
            }
            let Ok(permit) = handler.inbound_connections.clone().try_acquire_owned() else {
                accepted.conn.close(1, b"inbound connection limit exceeded");
                continue;
            };
            let handler = handler.clone();
            tokio::spawn(async move {
                handler.handle::<T>(accepted.conn, permit).await;
            });
        }
    });

    let wake_plane = transport.collection_wake_plane();
    let bootstrap_ids = config.peers.iter().map(|peer| peer.id).collect::<Vec<_>>();
    let (wake_tx, mut wake_rx) = tokio::sync::mpsc::channel::<WakeNotice>(256);
    let mut wake_topics: HashMap<[u8; 32], tokio::sync::mpsc::UnboundedSender<WakeCommand>> =
        HashMap::new();
    let (repair_tx, mut repair_rx) = tokio::sync::mpsc::unbounded_channel::<RepairOutcome>();
    let (discovery_tx, mut discovery_rx) =
        tokio::sync::mpsc::channel::<(CollectionHandle, Vec<PeerId>)>(64);
    let mut immediate = VecDeque::new();
    let mut pending = HashSet::new();
    let mut in_flight = HashSet::new();
    let mut failures: HashMap<RepairTarget, (u32, crate::clock::Mono)> = HashMap::new();
    let mut full_cursors: HashMap<RepairTarget, FullReplicaCursor> = HashMap::new();
    let mut current_roots: HashMap<[u8; 32], ([u8; 32], [u8; 32])> = HashMap::new();
    let mut next_period = crate::clock::mono_now();
    let mut provider_due: HashMap<ProviderKey, (ProviderToken, crate::clock::Mono)> =
        HashMap::new();

    loop {
        let mut disconnected = false;
        loop {
            match wiring.cmd_rx.try_recv() {
                Ok(NetCommand::SnapshotChanged(notice)) => {
                    if !notice.installed {
                        current_roots.clear();
                        participants.lock().unwrap().clear();
                        immediate.clear();
                        pending.clear();
                        failures.clear();
                        for (_, topic) in wake_topics.drain() {
                            let _ = topic.send(WakeCommand::Shutdown);
                        }
                        continue;
                    }
                    let mut observed = HashSet::new();
                    for (collection, semantic_root, payload_root) in notice.collections {
                        observed.insert(collection.raw);
                        let roots = (semantic_root, payload_root);
                        let changed = current_roots.insert(collection.raw, roots) != Some(roots);
                        let topic = wake_topics.entry(collection.raw).or_insert_with(|| {
                            spawn_wake_topic(
                                wake_plane.clone(),
                                collection,
                                bootstrap_ids.clone(),
                                wake_tx.clone(),
                            )
                        });
                        if changed && semantic_root != [0; 32] {
                            let _ = topic.send(WakeCommand::Observe(
                                CollectionWakeRoot::with_payload(semantic_root, payload_root),
                            ));
                        }
                    }
                    current_roots.retain(|collection, _| observed.contains(collection));
                    participants
                        .lock()
                        .unwrap()
                        .retain(|collection, _| observed.contains(collection));
                    let stale = wake_topics
                        .keys()
                        .filter(|collection| !observed.contains(*collection))
                        .copied()
                        .collect::<Vec<_>>();
                    for collection in stale {
                        if let Some(topic) = wake_topics.remove(&collection) {
                            let _ = topic.send(WakeCommand::Shutdown);
                        }
                    }
                }
                Ok(NetCommand::ProvidersUpdated(observation)) => {
                    let set = observation.into_set();
                    let now = crate::clock::mono_now();
                    provider_due.retain(|key, _| set.contains(key));
                    for (key, token) in set.iter() {
                        provider_due.entry(key).or_insert((token, now));
                    }
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

        while let Ok(notice) = wake_rx.try_recv() {
            let WakeNotice::Received {
                collection,
                received,
            } = notice
            else {
                next_period = crate::clock::mono_now();
                continue;
            };
            let wake = received.wake;
            if wake.origin().as_bytes() == &my_id {
                continue;
            }
            if !current_roots.contains_key(&collection.raw) {
                continue;
            }
            observe_participant(
                &mut participants.lock().unwrap(),
                collection.raw,
                *wake.origin().as_bytes(),
                crate::clock::mono_now(),
            );
            if current_roots
                .get(&collection.raw)
                .is_some_and(|(semantic, payload)| {
                    *semantic != [0; 32]
                        && (semantic != wake.root().as_bytes()
                            || (matches!(config.qos.blobs, BlobReplication::Full)
                                && payload != wake.root().payload_bytes()))
                })
            {
                enqueue_repair(
                    &mut immediate,
                    &mut pending,
                    RepairTarget {
                        collection,
                        peer: *wake.origin().as_bytes(),
                    },
                );
            }
        }
        while let Ok(outcome) = repair_rx.try_recv() {
            in_flight.remove(&outcome.target);
            if outcome.success {
                failures.remove(&outcome.target);
                if outcome.more {
                    if let Some(cursor) = outcome.full_cursor {
                        full_cursors.insert(outcome.target, cursor);
                    }
                } else {
                    full_cursors.remove(&outcome.target);
                }
                if outcome.more {
                    enqueue_repair(&mut immediate, &mut pending, outcome.target);
                }
            } else {
                let attempts = failures
                    .get(&outcome.target)
                    .map_or(1, |(attempts, _)| attempts.saturating_add(1));
                let shift = attempts.saturating_sub(1).min(6);
                if failures.len() < MAX_PENDING_REPAIRS || failures.contains_key(&outcome.target) {
                    failures.insert(
                        outcome.target,
                        (
                            attempts,
                            crate::clock::mono_now()
                                + crate::RETRY_BACKOFF_BASE.saturating_mul(1u32 << shift),
                        ),
                    );
                }
            }
        }
        while let Ok((collection, peers)) = discovery_rx.try_recv() {
            if let Some(topic) = wake_topics.get(&collection.raw) {
                let joined = peers
                    .iter()
                    .filter_map(|peer| EndpointId::from_bytes(peer).ok())
                    .collect();
                let _ = topic.send(WakeCommand::Join(joined));
            }
            let descriptor_missing = wiring
                .snapshot
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|snapshot| snapshot.get_blob(&collection.raw).is_none());
            if descriptor_missing && !peers.is_empty() {
                let client = provider_client.clone();
                let events = wiring.evt_tx.clone();
                let descriptor_peers = peers.clone();
                tokio::spawn(async move {
                    if let Some(bytes) = client
                        .fetch_from_providers(collection.raw, descriptor_peers)
                        .await
                    {
                        let mut batch = NetEventBatch::default();
                        let _ = batch.try_push(NetEvent::Blob {
                            expected: collection.raw,
                            bytes,
                        });
                        let _ = events.send(batch).await;
                    }
                });
            }
            for peer in peers {
                observe_participant(
                    &mut participants.lock().unwrap(),
                    collection.raw,
                    peer,
                    crate::clock::mono_now(),
                );
                enqueue_repair(
                    &mut immediate,
                    &mut pending,
                    RepairTarget { collection, peer },
                );
            }
        }

        let now = crate::clock::mono_now();
        if now >= next_period {
            next_period = now + REPAIR_PERIOD;
            for raw in current_roots.keys() {
                if config.qos.direction.pulls() {
                    let collection = CollectionHandle::new(*raw);
                    let client = provider_client.clone();
                    let discovery_tx = discovery_tx.clone();
                    tokio::spawn(async move {
                        let peers = client
                            .find_key(
                                collection_provider_key(collection),
                                collection_provider_token,
                                collection.raw,
                            )
                            .await;
                        let _ = discovery_tx.try_send((collection, peers));
                    });
                    let mut peers = live_participants(&mut participants.lock().unwrap(), *raw, now);
                    if !peers.is_empty() {
                        let rotation = (now.as_nanos() as usize
                            / REPAIR_PERIOD.as_nanos() as usize)
                            % peers.len();
                        peers.rotate_left(rotation);
                        peers.truncate(PERIODIC_REPAIR_SAMPLE);
                    }
                    for peer in peers {
                        enqueue_repair(
                            &mut immediate,
                            &mut pending,
                            RepairTarget {
                                collection: CollectionHandle::new(*raw),
                                peer,
                            },
                        );
                    }
                }
            }
        }

        if config.qos.direction.pulls() {
            while in_flight.len() < MAX_CONCURRENT_REPAIRS {
                let Some(target) = immediate.pop_front() else {
                    break;
                };
                pending.remove(&target);
                if in_flight.contains(&target)
                    || failures
                        .get(&target)
                        .is_some_and(|(_, retry_at)| now < *retry_at)
                {
                    continue;
                }
                let Some(local) = wiring
                    .snapshot
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|snapshot| snapshot.collection(target.collection))
                else {
                    continue;
                };
                in_flight.insert(target);
                let transport = transport.clone();
                let pool = pool.clone();
                let events = wiring.evt_tx.clone();
                let operational_blobs = wiring.operational_blobs.clone();
                let repair_tx = repair_tx.clone();
                let full = matches!(config.qos.blobs, BlobReplication::Full);
                let full_cursor = full_cursors.get(&target).cloned();
                tokio::spawn(async move {
                    let result = tokio::time::timeout(
                        REPAIR_DEADLINE,
                        reconcile_collection_peer(
                            &transport,
                            &pool,
                            target,
                            local,
                            full_cursor,
                            &operational_blobs,
                            &events,
                            full,
                        ),
                    )
                    .await;
                    let (success, more, full_cursor) = match result {
                        Ok(Ok((more, cursor))) => (true, more, cursor),
                        Ok(Err(error)) => {
                            debug!(%error, "collection repair failed");
                            (false, false, None)
                        }
                        Err(_) => (false, false, None),
                    };
                    let _ = repair_tx.send(RepairOutcome {
                        target,
                        success,
                        more,
                        full_cursor,
                    });
                });
            }
        }

        let due = provider_due
            .iter()
            .filter_map(|(key, (token, due))| (*due <= now).then_some((*key, *token)))
            .take(ALPHA)
            .collect::<Vec<_>>();
        for (key, token) in due {
            provider_due.insert(key, (token, now + PROVIDER_RENEWAL_INTERVAL));
            let client = provider_client.clone();
            tokio::spawn(async move {
                client.announce_key(key, token).await;
            });
        }

        tokio::time::sleep(HOST_POLL_PERIOD).await;
    }
}

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
            .map_err(|_| anyhow::anyhow!("store side stopped during collection admission"))
    }
}

async fn reconcile_collection_peer<T: Transport>(
    transport: &T,
    pool: &SharedPool<T::Conn>,
    target: RepairTarget,
    local: Arc<CollectionSnapshot>,
    prior_cursor: Option<FullReplicaCursor>,
    operational_blobs: &OperationalBlobSlot,
    events: &tokio::sync::mpsc::Sender<NetEventBatch>,
    full: bool,
) -> anyhow::Result<(bool, Option<FullReplicaCursor>)> {
    let connection = pool_get(transport, pool, target.peer).await?;
    let reader = operational_blobs
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| local.blobs.clone());
    let delta = match pull_collection(
        connection.conn(),
        &local.activation,
        local.read_evidence.iter().cloned().collect(),
        &local.full,
        prior_cursor.as_ref(),
        |hash| reader.get_blob(hash),
        full,
    )
    .await
    {
        Ok(delta) => delta,
        Err(error) => {
            pool_invalidate(pool, target.peer, &connection.entry);
            return Err(error);
        }
    };
    let mut admissions = AdmissionBatcher::new(events);
    for bundle in delta.write_evidence {
        admissions
            .push(NetEvent::CapabilityProofBundle(bundle))
            .await?;
    }
    for record in delta.records {
        admissions.push(NetEvent::CollectionRecord(record)).await?;
    }
    if !delta.blobs.is_empty() || (full && !delta.more) {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        admissions
            .push(NetEvent::FullPage {
                blobs: delta.blobs,
                final_page: !delta.more,
                ack: ack_tx,
            })
            .await?;
        admissions.flush().await?;
        ack_rx
            .await
            .map_err(|_| anyhow::anyhow!("store side dropped Full page before durability"))?;
        return Ok((delta.more, delta.full_cursor));
    }
    admissions.flush().await?;
    Ok((delta.more, delta.full_cursor))
}

impl<T: Transport> ProviderClient<T> {
    async fn find_node(&self, peer: PeerId, target: RoutingKey) -> anyhow::Result<Vec<PeerId>> {
        let connection = pool_get(&self.transport, &self.pool, peer).await?;
        let response = tokio::time::timeout(OP_DEADLINE, op_find_node(connection.conn(), &target))
            .await
            .map_err(|_| anyhow::anyhow!("FIND_NODE deadline exceeded"))?;
        match response {
            Ok(peers) => {
                self.candidates.lock().unwrap().promote_authenticated(peer);
                Ok(peers)
            }
            Err(error) => {
                pool_invalidate(&self.pool, peer, &connection.entry);
                Err(error)
            }
        }
    }

    async fn lookup_replicas(&self, target: RoutingKey) -> Vec<PeerId> {
        let seeds = self.candidates.lock().unwrap().closest(target, K);
        let mut lookup = IterativeLookup::new(self.my_id, target, seeds);
        let mut pending: FuturesUnordered<
            futures::future::BoxFuture<'_, (PeerId, anyhow::Result<Vec<PeerId>>)>,
        > = FuturesUnordered::new();
        let completed = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                for peer in lookup.next_batch() {
                    pending.push(Box::pin(async move {
                        let reply = self.find_node(peer, target).await;
                        (peer, reply)
                    }));
                }
                let Some((peer, reply)) = pending.next().await else {
                    break;
                };
                match reply {
                    Ok(peers) => {
                        let valid = peers
                            .into_iter()
                            .filter(|candidate| EndpointId::from_bytes(candidate).is_ok());
                        lookup.record_authenticated_response(
                            peer,
                            valid,
                            &mut self.candidates.lock().unwrap(),
                        );
                    }
                    Err(_) => {
                        lookup.record_failure(peer, &mut self.candidates.lock().unwrap());
                    }
                }
                if lookup.is_finished() && pending.is_empty() {
                    break;
                }
            }
        })
        .await;
        if completed.is_err() {
            drop(pending);
        }
        let mut replicas = lookup.closest_authenticated_responders().to_vec();
        replicas.push(self.my_id);
        replicas.sort_unstable_by(|a, b| crate::routing::distance_cmp(target, *a, *b));
        replicas.dedup();
        replicas.truncate(K);
        replicas
    }

    async fn put(&self, peer: PeerId, key: ProviderKey, token: ProviderToken) -> bool {
        if peer == self.my_id {
            return self.providers.lock().unwrap().put(
                key,
                self.my_id,
                token,
                crate::clock::mono_now(),
            );
        }
        let Ok(connection) = pool_get(&self.transport, &self.pool, peer).await else {
            return false;
        };
        match tokio::time::timeout(
            OP_DEADLINE,
            op_provider_put(connection.conn(), &key, &token),
        )
        .await
        {
            Ok(Ok(stored)) => {
                self.candidates.lock().unwrap().promote_authenticated(peer);
                stored
            }
            Ok(Err(_)) | Err(_) => {
                pool_invalidate(&self.pool, peer, &connection.entry);
                false
            }
        }
    }

    async fn announce_key(&self, key: ProviderKey, token: ProviderToken) {
        let targets = self.lookup_replicas(key).await;
        futures::stream::iter(targets)
            .for_each_concurrent(ALPHA, |peer| async move {
                self.put(peer, key, token).await;
            })
            .await;
    }

    async fn get(&self, peer: PeerId, key: ProviderKey) -> Vec<(PeerId, ProviderToken)> {
        if peer == self.my_id {
            return self
                .providers
                .lock()
                .unwrap()
                .get(key, crate::clock::mono_now());
        }
        let Ok(connection) = pool_get(&self.transport, &self.pool, peer).await else {
            return Vec::new();
        };
        match tokio::time::timeout(OP_DEADLINE, op_provider_get(connection.conn(), &key)).await {
            Ok(Ok(providers)) => {
                self.candidates.lock().unwrap().promote_authenticated(peer);
                providers
            }
            Ok(Err(_)) | Err(_) => {
                pool_invalidate(&self.pool, peer, &connection.entry);
                Vec::new()
            }
        }
    }

    async fn find_key(
        &self,
        key: ProviderKey,
        token_for: fn([u8; 32], PeerId) -> ProviderToken,
        identity: [u8; 32],
    ) -> Vec<PeerId> {
        let replicas = self.lookup_replicas(key).await;
        let mut replies = futures::stream::iter(replicas)
            .map(|peer| async move { self.get(peer, key).await })
            .buffer_unordered(ALPHA);
        let mut providers = Vec::new();
        while let Some(reply) = replies.next().await {
            for (provider, token) in reply {
                if token_for(identity, provider) == token && !providers.contains(&provider) {
                    providers.push(provider);
                }
            }
        }
        providers
    }

    async fn fetch_from_providers(&self, hash: RawHash, providers: Vec<PeerId>) -> Option<Bytes> {
        let mut attempts = futures::stream::iter(providers)
            .map(|peer| async move {
                let connection = pool_get(&self.transport, &self.pool, peer).await.ok()?;
                let response =
                    tokio::time::timeout(OP_DEADLINE, op_get_blob(connection.conn(), &hash)).await;
                match response {
                    Ok(Ok(Some(bytes))) if blake3::hash(&bytes).as_bytes() == &hash => {
                        self.candidates.lock().unwrap().promote_authenticated(peer);
                        Some(bytes)
                    }
                    Ok(Ok(None)) => None,
                    Ok(Ok(Some(_))) | Ok(Err(_)) | Err(_) => {
                        pool_invalidate(&self.pool, peer, &connection.entry);
                        None
                    }
                }
            })
            .buffer_unordered(ALPHA);
        while let Some(result) = attempts.next().await {
            if result.is_some() {
                return result;
            }
        }
        None
    }

    /// Ask known collection participants directly, but reveal `hash` only
    /// after the remote endpoint admits our READ(C) proof forest.
    async fn fetch_collection_blob(
        &self,
        collection: CollectionHandle,
        policy: CollectionPolicy,
        hash: RawHash,
    ) -> Option<Bytes> {
        let peers = live_participants(
            &mut self.participants.lock().unwrap(),
            collection.raw,
            crate::clock::mono_now(),
        );
        let mut peers = peers;
        for peer in self
            .find_key(
                collection_provider_key(collection),
                collection_provider_token,
                collection.raw,
            )
            .await
        {
            if !peers.contains(&peer) {
                peers.push(peer);
            }
        }
        let mut attempts = futures::stream::iter(peers)
            .filter(|peer| futures::future::ready(*peer != self.my_id))
            .map(|peer| {
                let policy = policy.clone();
                async move {
                    let connection = pool_get(&self.transport, &self.pool, peer).await.ok()?;
                    let response =
                        fetch_collection_blob_session(connection.conn(), collection, policy, hash)
                            .await;
                    match response {
                        Ok(Some(bytes)) if blake3::hash(&bytes).as_bytes() == &hash => Some(bytes),
                        Ok(None) => None,
                        Ok(Some(_)) | Err(_) => {
                            pool_invalidate(&self.pool, peer, &connection.entry);
                            None
                        }
                    }
                }
            })
            .buffer_unordered(ALPHA);
        while let Some(result) = attempts.next().await {
            if result.is_some() {
                return result;
            }
        }
        None
    }
}

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: SnapshotSlot,
    candidates: RoutingCandidates,
    providers: Arc<Mutex<ProviderDirectory>>,
    serve_data: bool,
    inbound_connections: Arc<tokio::sync::Semaphore>,
    inbound_requests: Arc<tokio::sync::Semaphore>,
}

impl SnapshotHandler {
    async fn handle<T: Transport>(
        &self,
        connection: T::Conn,
        _permit: tokio::sync::OwnedSemaphorePermit,
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
            let per_connection = Arc::new(tokio::sync::Semaphore::new(MAX_REQUESTS_PER_CONNECTION));
            loop {
                let accepted = tokio::select! {
                    stream = connection.accept_bi() => stream,
                    () = tokio::time::sleep(CONNECTION_IDLE_DEADLINE) => {
                        connection.close(0, b"connection idle timeout");
                        return;
                    }
                };
                let Some((mut send, mut recv)) = accepted else {
                    return;
                };
                let Ok(connection_permit) = per_connection.clone().try_acquire_owned() else {
                    connection.close(1, b"request concurrency exceeded");
                    return;
                };
                let Ok(global_permit) = self.inbound_requests.clone().try_acquire_owned() else {
                    connection.close(1, b"global request concurrency exceeded");
                    return;
                };
                let handler = self.clone();
                tokio::spawn(
                    async move {
                        let operation = tokio::time::timeout(
                            REQUEST_DEADLINE,
                            handler.serve_stream::<T::Conn>(peer, &mut send, &mut recv),
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
        send: &mut C::SendHalf,
        recv: &mut C::RecvHalf,
    ) -> anyhow::Result<()> {
        let op = recv_u8(recv).await?;
        let span = debug_span!("stream", op = op_name(op));
        let _entered = span.enter();
        match op {
            OP_COLLECTION_BLOB => {
                let snapshot = self.snapshot.lock().unwrap().clone();
                let blob_snapshot = snapshot.clone();
                serve_collection_blob(
                    recv,
                    send,
                    move |collection| {
                        snapshot
                            .as_ref()
                            .and_then(|snapshot| snapshot.collection(collection))
                            .map(|collection| {
                                (
                                    collection.activation.clone(),
                                    collection.read_evidence.clone(),
                                )
                            })
                    },
                    move |_collection, hash| {
                        self.serve_data
                            .then(|| {
                                blob_snapshot
                                    .as_ref()
                                    .and_then(|snapshot| snapshot.get_bearer_blob(hash))
                            })
                            .flatten()
                    },
                )
                .await?;
            }
            OP_COLLECTION_REPAIR => {
                if !self.serve_data {
                    serve_collection_repair(recv, send, peer, |_| None, |_, _| None).await?;
                } else {
                    let snapshot = self.snapshot.lock().unwrap().clone();
                    let blob_snapshot = snapshot.clone();
                    serve_collection_repair(
                        recv,
                        send,
                        peer,
                        move |collection| {
                            snapshot
                                .as_ref()
                                .and_then(|snapshot| snapshot.collection(collection))
                                .map(|collection| {
                                    (
                                        collection.activation.clone(),
                                        collection.read_evidence.clone(),
                                        collection.full.clone(),
                                    )
                                })
                        },
                        move |_collection, hash| {
                            blob_snapshot
                                .as_ref()
                                .and_then(|snapshot| snapshot.get_bearer_blob(hash))
                        },
                    )
                    .await?;
                }
            }
            OP_GET_BLOB => {
                if !self.serve_data {
                    anyhow::bail!("local direction policy does not serve data");
                }
                let hash = recv_hash(recv).await?;
                require_stream_eof(recv).await?;
                let bytes = self
                    .snapshot
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(|snapshot| snapshot.get_bearer_blob(hash));
                if let Some(bytes) = bytes {
                    send_u64_be(send, bytes.len() as u64).await?;
                    send.write_all(&bytes).await?;
                } else {
                    send_u64_be(send, u64::MAX).await?;
                }
            }
            OP_PROVIDER_PUT => {
                let key = recv_hash(recv).await?;
                let token = recv_hash(recv).await?;
                require_stream_eof(recv).await?;
                let stored = self.providers.lock().unwrap().put(
                    key,
                    peer.to_bytes(),
                    token,
                    crate::clock::mono_now(),
                );
                send_u8(
                    send,
                    if stored {
                        PROVIDER_PUT_OK
                    } else {
                        PROVIDER_PUT_FULL
                    },
                )
                .await?;
            }
            OP_PROVIDER_GET => {
                let key = recv_exact_key(recv).await?;
                let providers = self
                    .providers
                    .lock()
                    .unwrap()
                    .get(key, crate::clock::mono_now());
                send_u8(send, providers.len() as u8).await?;
                for (provider, token) in providers {
                    send_hash(send, &provider).await?;
                    send_hash(send, &token).await?;
                }
            }
            OP_FIND_NODE => {
                let target = recv_exact_key(recv).await?;
                let mut peers = self.candidates.lock().unwrap().closest_verified(target, K);
                peers.retain(|candidate| *candidate != peer.to_bytes());
                send_u8(send, peers.len() as u8).await?;
                for peer in peers {
                    send_hash(send, &peer).await?;
                }
            }
            _ => anyhow::bail!("unknown direct RPC operation {op:#x}"),
        }
        self.candidates
            .lock()
            .unwrap()
            .promote_authenticated(peer.to_bytes());
        Ok(())
    }
}

async fn recv_exact_key<R: tokio::io::AsyncRead + Unpin>(recv: &mut R) -> anyhow::Result<[u8; 32]> {
    let key = recv_hash(recv).await?;
    require_stream_eof(recv).await?;
    Ok(key)
}

async fn require_stream_eof<R: tokio::io::AsyncRead + Unpin>(recv: &mut R) -> anyhow::Result<()> {
    let mut trailing = [0u8; 1];
    if recv.read(&mut trailing).await? != 0 {
        anyhow::bail!("request contains trailing bytes");
    }
    Ok(())
}

fn op_name(op: u8) -> &'static str {
    match op {
        OP_GET_BLOB => "GET_BLOB",
        OP_PROVIDER_PUT => "PROVIDER_PUT",
        OP_PROVIDER_GET => "PROVIDER_GET",
        OP_FIND_NODE => "FIND_NODE",
        OP_COLLECTION_REPAIR => "COLLECTION_REPAIR",
        OP_COLLECTION_BLOB => "COLLECTION_BLOB",
        _ => "UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        MAX_COLLECTION_PARTICIPANTS, MAX_PENDING_REPAIRS, RepairTarget, enqueue_repair,
        live_participants, observe_participant,
    };

    #[test]
    fn collection_participant_hints_are_bounded_under_signed_wake_flood() {
        let collection = [0x41; 32];
        let now = crate::clock::mono_now();
        let mut participants = HashMap::new();
        for index in 0..(MAX_COLLECTION_PARTICIPANTS * 2) {
            let mut peer = [0u8; 32];
            peer[..8].copy_from_slice(&(index as u64).to_be_bytes());
            observe_participant(&mut participants, collection, peer, now);
        }
        let live = live_participants(&mut participants, collection, now);
        assert_eq!(live.len(), MAX_COLLECTION_PARTICIPANTS);
        assert!(live.contains(&{
            let mut newest = [0u8; 32];
            newest[..8]
                .copy_from_slice(&((MAX_COLLECTION_PARTICIPANTS * 2 - 1) as u64).to_be_bytes());
            newest
        }));
    }

    #[test]
    fn pending_repairs_are_coalesced_and_bounded_under_wake_flood() {
        let mut queue = std::collections::VecDeque::new();
        let mut pending = std::collections::HashSet::new();
        for index in 0..(MAX_PENDING_REPAIRS * 2) {
            let mut peer = [0u8; 32];
            peer[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let target = RepairTarget {
                collection: triblespace_core::collection::CollectionHandle::new([0x51; 32]),
                peer,
            };
            enqueue_repair(&mut queue, &mut pending, target);
            enqueue_repair(&mut queue, &mut pending, target);
        }
        assert_eq!(queue.len(), MAX_PENDING_REPAIRS);
        assert_eq!(pending.len(), MAX_PENDING_REPAIRS);
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
        .expect("default relay URLs remain valid after hostname normalization")
}
