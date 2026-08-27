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
use triblespace_core::capability::{CapabilityProofBundle, CapabilityProofId};
use triblespace_core::collection::CollectionStore;
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{Entry as PatchEntry, IdentitySchema, PATCH};
use triblespace_core::repo::peer::PeerEvidence;
use triblespace_core::repo::{BlobStore, BlobStoreGet, CapabilityProofStore, PeerStore};

use crate::channel::{
    MAX_ADMISSION_BRIDGE_BATCHES, NetCommand, NetEvent, NetEventBatch, SnapshotNotice,
};
use crate::identity::iroh_secret;
use crate::inventory::{
    AuthorizedInventorySession, BlobInventory, CapabilityProofInventory, CollectionRecordInventory,
    InventoryComponent, InventoryGeneration, InventoryManifest, InventoryServerConfig,
    InventorySnapshot, PeerInventory, ReconcileQos,
};
use crate::inventory_reconcile::InventoryWalker;
use crate::inventory_wire::{
    BLOB_TRANSFER_CHUNK_BYTES, InventoryBlobRangeRequest, InventoryBlobRangeResponse,
    InventoryLeaf, InventoryLeafValue, InventoryNodeRequest, InventoryNodeResponse,
    OP_INVENTORY_AUTH, OP_INVENTORY_BLOB_RANGE, OP_INVENTORY_MANIFEST, OP_INVENTORY_NODE,
    key_node_response, op_inventory_auth, op_inventory_blob_range, op_inventory_manifest,
    op_inventory_node, recv_blob_range_request, recv_inventory_auth_request, recv_manifest_request,
    recv_node_request, send_blob_not_in_snapshot, send_blob_range, send_blob_snapshot_unavailable,
    send_inventory_auth_ok, send_inventory_auth_rejected, send_manifest, send_node_response,
};
use crate::protocol::*;
use crate::provider::{ElementId, ProviderDirectory, ProviderKey, provider_key};
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

trait BlobSnapshotReader: Send + Sync + 'static {
    fn get_blob(&self, hash: RawHash) -> Option<Bytes>;
}

struct CloneableBlobSnapshotReader<R>(Mutex<R>);

impl<R> BlobSnapshotReader for CloneableBlobSnapshotReader<R>
where
    R: BlobStoreGet + Clone + Send + 'static,
{
    fn get_blob(&self, hash: RawHash) -> Option<Bytes> {
        // BlobStore's public Reader contract promises Clone + Send but not
        // Sync. Clone the immutable handle under this narrow lock and perform
        // payload lookup and validation after releasing it.
        let reader = self.0.lock().unwrap().clone();
        reader
            .get::<Bytes, UnknownBlob>(Inline::<Handle<UnknownBlob>>::new(hash))
            .ok()
    }
}

enum InventoryComponentData {
    Peer(PeerInventory),
    CollectionRecord(CollectionRecordInventory),
    CapabilityProof(CapabilityProofInventory),
    Blob {
        inventory: Arc<BlobInventory>,
        reader: Arc<dyn BlobSnapshotReader>,
    },
}

/// One immutable component of a store observation.
///
/// Historical root pinning retains this object, never the aggregate store
/// snapshot. A record-only change can therefore keep exactly one old record
/// tree without also retaining old blob readers and the other three trees.
struct InventoryComponentSnapshot {
    team: VerifyingKey,
    manifest: crate::inventory::ComponentManifest,
    data: InventoryComponentData,
}

impl InventoryComponentSnapshot {
    fn node_summary(&self, relative_prefix: &[u8]) -> Option<([u8; 32], u64)> {
        fn summary<const KEY_LEN: usize, V>(
            inventory: &PATCH<KEY_LEN, IdentitySchema, V, triblespace_core::patch::Blake3Merkle>,
            prefix: &[u8],
        ) -> Option<([u8; 32], u64)> {
            inventory
                .merkle_node(prefix)
                .map(|node| (node.digest(), node.leaf_count()))
        }

        let component = self.manifest.component();
        let base = component.base_prefix(self.team);
        if relative_prefix.len() > component.relative_key_len(base) {
            return None;
        }
        let mut absolute = Vec::with_capacity(base.as_bytes().len() + relative_prefix.len());
        absolute.extend_from_slice(base.as_bytes());
        absolute.extend_from_slice(relative_prefix);
        match &self.data {
            InventoryComponentData::Peer(inventory) => summary(inventory, &absolute),
            InventoryComponentData::CollectionRecord(inventory) => summary(inventory, &absolute),
            InventoryComponentData::CapabilityProof(inventory) => summary(inventory, &absolute),
            InventoryComponentData::Blob { inventory, .. } => summary(inventory, &absolute),
        }
    }

    fn contains_relative_key(&self, relative_key: &[u8]) -> bool {
        let component = self.manifest.component();
        let base = component.base_prefix(self.team);
        let Ok(absolute) = base.absolute_key(component, relative_key) else {
            return false;
        };
        match &self.data {
            InventoryComponentData::Peer(inventory) => absolute
                .as_slice()
                .try_into()
                .ok()
                .is_some_and(|key| inventory.get(key).is_some()),
            InventoryComponentData::CollectionRecord(inventory) => absolute
                .as_slice()
                .try_into()
                .ok()
                .is_some_and(|key| inventory.get(key).is_some()),
            InventoryComponentData::CapabilityProof(inventory) => absolute
                .as_slice()
                .try_into()
                .ok()
                .is_some_and(|key| inventory.get(key).is_some()),
            InventoryComponentData::Blob { inventory, .. } => absolute
                .as_slice()
                .try_into()
                .ok()
                .is_some_and(|key| inventory.get(key).is_some()),
        }
    }

    fn inventory_node(
        &self,
        request: &InventoryNodeRequest,
    ) -> anyhow::Result<InventoryNodeResponse> {
        if request.component != self.manifest.component()
            || self.manifest.root() != Some(request.root)
            || self.manifest.leaf_count() != request.leaf_count
        {
            return Ok(InventoryNodeResponse::SnapshotUnavailable);
        }
        let component = self.manifest.component();
        match &self.data {
            InventoryComponentData::Peer(inventory) => {
                key_node_response(inventory, self.team, component, &request.prefix, |_, ()| {
                    Ok(InventoryLeafValue::Peer)
                })
            }
            InventoryComponentData::CollectionRecord(inventory) => key_node_response(
                inventory,
                self.team,
                component,
                &request.prefix,
                |key, record| {
                    let id = Id::new(key).ok_or_else(|| {
                        anyhow::anyhow!("inventory contains the reserved nil record id")
                    })?;
                    if record.id() != id {
                        anyhow::bail!(
                            "inventory record body does not match its authenticated leaf key"
                        );
                    }
                    Ok(InventoryLeafValue::CollectionRecord(*record))
                },
            ),
            InventoryComponentData::CapabilityProof(inventory) => key_node_response(
                inventory,
                self.team,
                component,
                &request.prefix,
                |key, proof| {
                    if proof.id() != CapabilityProofId::new(key) {
                        anyhow::bail!(
                            "inventory proof body does not match its authenticated leaf key"
                        );
                    }
                    Ok(InventoryLeafValue::CapabilityProof(proof.clone()))
                },
            ),
            InventoryComponentData::Blob { inventory, .. } => {
                key_node_response(inventory, self.team, component, &request.prefix, |_, ()| {
                    Ok(InventoryLeafValue::Blob)
                })
            }
        }
    }

    fn get_blob(&self, hash: RawHash) -> Option<Bytes> {
        match &self.data {
            InventoryComponentData::Blob { reader, .. } => reader.get_blob(hash),
            _ => None,
        }
    }

    fn inventory_blob(&self, request: InventoryBlobRangeRequest) -> PinnedBlob {
        let InventoryComponentData::Blob { inventory, reader } = &self.data else {
            return PinnedBlob::SnapshotUnavailable;
        };
        if self.manifest.root() != Some(request.root)
            || self.manifest.leaf_count() != request.leaf_count
        {
            return PinnedBlob::SnapshotUnavailable;
        }
        if inventory.get(&request.hash).is_none() {
            return PinnedBlob::NotInSnapshot;
        }
        reader
            .get_blob(request.hash)
            .map(PinnedBlob::Found)
            .unwrap_or(PinnedBlob::SnapshotUnavailable)
    }

    fn reusable_with(&self, other: &Self) -> bool {
        self.team == other.team && self.manifest == other.manifest
    }

    fn refreshed_blob_with_tree_from(&self, previous: &Self) -> Self {
        let (
            InventoryComponentData::Blob {
                reader,
                inventory: _,
            },
            InventoryComponentData::Blob {
                inventory: previous_inventory,
                reader: _,
            },
        ) = (&self.data, &previous.data)
        else {
            unreachable!("the Blob component slot always contains blob snapshots");
        };
        Self {
            team: self.team,
            manifest: self.manifest,
            data: InventoryComponentData::Blob {
                inventory: previous_inventory.clone(),
                reader: reader.clone(),
            },
        }
    }
}

type SharedComponentSnapshot = Arc<InventoryComponentSnapshot>;

/// Snapshot of the complete single-team store observation served by the host.
pub(crate) struct StoreSnapshot {
    team: VerifyingKey,
    manifest: InventoryManifest,
    components: [SharedComponentSnapshot; 4],
    routing_peers: Vec<PeerId>,
}

impl StoreSnapshot {
    pub(crate) fn from_store<S>(store: &mut S, team: VerifyingKey) -> anyhow::Result<Self>
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
        let mut routing_peers: Vec<_> = peers
            .iter()
            .filter(|evidence| evidence.team() == team)
            .map(|evidence| evidence.peer().to_bytes())
            .collect();
        routing_peers.sort_unstable();
        routing_peers.dedup();

        let inventory = InventorySnapshot::from_observation(team, reader, peers, records, proofs)?;
        let components = [
            Arc::new(InventoryComponentSnapshot {
                team,
                manifest: inventory.manifest().component(InventoryComponent::Peer),
                data: InventoryComponentData::Peer(inventory.peers().clone()),
            }),
            Arc::new(InventoryComponentSnapshot {
                team,
                manifest: inventory
                    .manifest()
                    .component(InventoryComponent::CollectionRecord),
                data: InventoryComponentData::CollectionRecord(inventory.records().clone()),
            }),
            Arc::new(InventoryComponentSnapshot {
                team,
                manifest: inventory
                    .manifest()
                    .component(InventoryComponent::CapabilityProof),
                data: InventoryComponentData::CapabilityProof(inventory.proofs().clone()),
            }),
            Arc::new(InventoryComponentSnapshot {
                team,
                manifest: inventory.manifest().component(InventoryComponent::Blob),
                data: InventoryComponentData::Blob {
                    inventory: Arc::new(inventory.blobs().clone()),
                    reader: Arc::new(CloneableBlobSnapshotReader(Mutex::new(
                        inventory.reader().clone(),
                    ))),
                },
            }),
        ];

        Ok(StoreSnapshot {
            team,
            manifest: inventory.manifest().clone(),
            components,
            routing_peers,
        })
    }

    fn component(&self, component: InventoryComponent) -> &SharedComponentSnapshot {
        &self.components[component.index()]
    }

    /// Reuse immutable component structures and return superseded trees so
    /// their potentially large recursive drops can happen outside pointer
    /// locks held by the caller.
    fn reuse_unchanged_components(&mut self, previous: &Self) -> Vec<SharedComponentSnapshot> {
        let mut retired = Vec::new();
        for component in InventoryComponent::ALL {
            let index = component.index();
            if self.components[index].reusable_with(&previous.components[index]) {
                let replacement = if component == InventoryComponent::Blob {
                    Arc::new(
                        self.components[index]
                            .refreshed_blob_with_tree_from(&previous.components[index]),
                    )
                } else {
                    previous.components[index].clone()
                };
                retired.push(std::mem::replace(&mut self.components[index], replacement));
            }
        }
        retired
    }

    fn team(&self) -> VerifyingKey {
        self.team
    }

    fn manifest(&self) -> InventoryManifest {
        self.manifest.clone()
    }

    fn routing_peers(&self) -> Vec<PeerId> {
        self.routing_peers.clone()
    }

    fn get_blob(&self, hash: &RawHash) -> Option<Bytes> {
        self.component(InventoryComponent::Blob).get_blob(*hash)
    }

    fn node_summary(
        &self,
        component: InventoryComponent,
        relative_prefix: &[u8],
    ) -> Option<([u8; 32], u64)> {
        self.component(component).node_summary(relative_prefix)
    }

    fn contains_relative_key(&self, component: InventoryComponent, relative_key: &[u8]) -> bool {
        self.component(component)
            .contains_relative_key(relative_key)
    }
}

pub(crate) enum PinnedBlob {
    SnapshotUnavailable,
    NotInSnapshot,
    Found(Bytes),
}

type SharedSnapshot = Arc<StoreSnapshot>;
type SnapshotSlot = Arc<Mutex<Option<SharedSnapshot>>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct InventoryCacheKey {
    team: [u8; 32],
    component: InventoryComponent,
    root: [u8; 32],
}

/// Bound history independently for each component. A hot Blob inventory may
/// therefore retain at most eight reader/tree generations instead of
/// consuming 29 of a global 32-slot cache while three cold roots remain.
/// Active requests own an Arc lease and safely outlive LRU eviction.
const MAX_PINNED_ROOTS_PER_COMPONENT: usize = 8;

#[derive(Default)]
struct InventorySnapshotCache {
    snapshots: HashMap<InventoryCacheKey, SharedComponentSnapshot>,
    least_to_most_recent: VecDeque<InventoryCacheKey>,
}

impl InventorySnapshotCache {
    /// Refresh the backend lease of an already-pinned Blob root without
    /// implicitly pinning a root that no client requested.
    ///
    /// The Blob key set can remain stable across backend compaction. Updating
    /// the current store snapshot must still retire the old mmap/generation;
    /// waiting for another manifest request could otherwise retain it forever.
    fn refresh_pinned_blob_reader(
        &mut self,
        snapshot: &StoreSnapshot,
    ) -> Option<SharedComponentSnapshot> {
        let component = InventoryComponent::Blob;
        let current = snapshot.component(component);
        let Some(root) = current.manifest.root() else {
            return None;
        };
        let key = InventoryCacheKey {
            team: snapshot.team().to_bytes(),
            component,
            root,
        };
        let Some(pinned) = self.snapshots.get_mut(&key) else {
            return None;
        };
        debug_assert!(current.reusable_with(pinned));
        Some(std::mem::replace(pinned, current.clone()))
    }

    fn pin_manifest(
        &mut self,
        team: VerifyingKey,
        manifest: &InventoryManifest,
        snapshot: &StoreSnapshot,
    ) -> Vec<SharedComponentSnapshot> {
        let mut retired = Vec::new();
        for entry in manifest.components() {
            let Some(root) = entry.root() else {
                continue;
            };
            let key = InventoryCacheKey {
                team: team.to_bytes(),
                component: entry.component(),
                root,
            };
            let current = snapshot.component(entry.component());
            match self.snapshots.entry(key) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(current.clone());
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    debug_assert!(current.reusable_with(slot.get()));
                    // A blob component also owns one backend reader lease.
                    // Replace that lease with the newest equivalent snapshot
                    // instead of keeping a compacted mmap/generation alive
                    // forever while the blob key set stays unchanged. Active
                    // requests hold their own Arc and finish safely.
                    if entry.component() == InventoryComponent::Blob {
                        retired.push(slot.insert(current.clone()));
                    }
                }
            }
            self.touch(key);
            while self
                .snapshots
                .keys()
                .filter(|candidate| candidate.component == entry.component())
                .count()
                > MAX_PINNED_ROOTS_PER_COMPONENT
            {
                let position = self
                    .least_to_most_recent
                    .iter()
                    .position(|candidate| candidate.component == entry.component())
                    .expect("an overfull component cache has an LRU entry");
                let oldest = self
                    .least_to_most_recent
                    .remove(position)
                    .expect("located component LRU entry remains present");
                if let Some(snapshot) = self.snapshots.remove(&oldest) {
                    retired.push(snapshot);
                }
            }
        }
        retired
    }

    fn get(
        &mut self,
        team: VerifyingKey,
        component: InventoryComponent,
        root: [u8; 32],
    ) -> Option<SharedComponentSnapshot> {
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

fn pinned_snapshot_if_serving(
    current: &SnapshotSlot,
    snapshots: &InventorySnapshots,
    team: VerifyingKey,
    component: InventoryComponent,
    root: [u8; 32],
) -> Option<SharedComponentSnapshot> {
    // Keep the current-slot lock through the cache lookup so clearing the
    // serving view is the linearization point after which no old pinned root
    // can begin another response.
    let current = current.lock().unwrap();
    current.as_ref()?;
    snapshots.lock().unwrap().get(team, component, root)
}

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
    fn announce_element(&self, element: ElementId) -> futures::future::BoxFuture<'static, usize>;
    fn find_element_providers(
        &self,
        element: ElementId,
    ) -> futures::future::BoxFuture<'static, Vec<PeerId>>;
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

    /// Deterministic rendezvous replicas among the currently known peers.
    ///
    /// Temporarily divergent PEER views can choose different replicas and
    /// therefore yield a temporary miss. This soft directory intentionally
    /// does not grow a second node-discovery protocol: ordinary PEER/gossip
    /// convergence plus lease renewal repairs the overlap.
    fn closest(&self, target: ProviderKey, self_id: PeerId, limit: usize) -> Vec<PeerId> {
        let mut peers = self.candidates(self_id);
        peers.push(self_id);
        peers.sort_unstable_by(|a, b| {
            a.iter()
                .zip(b)
                .zip(target)
                .map(|((&a, &b), target)| (a ^ target).cmp(&(b ^ target)))
                .find(|ordering| !ordering.is_eq())
                .unwrap_or_else(|| a.cmp(b))
        });
        peers.truncate(limit);
        peers
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
    can_announce: bool,
    my_id: PeerId,
    team: VerifyingKey,
    candidates: RoutingCandidates,
    providers: Arc<Mutex<ProviderDirectory>>,
}

struct ProviderClient<T: Transport> {
    transport: T,
    pool: SharedPool<T::Conn>,
    connect_proof: CapabilityProofBundle,
    sync_proof: CapabilityProofBundle,
    providers: Arc<Mutex<ProviderDirectory>>,
    my_id: PeerId,
}

impl<T: Transport> NetCap<T> {
    fn provider_client(&self) -> ProviderClient<T> {
        ProviderClient {
            transport: self.transport.clone(),
            pool: self.pool.clone(),
            connect_proof: self.connect_proof.clone(),
            sync_proof: self.sync_proof.clone(),
            providers: self.providers.clone(),
            my_id: self.my_id,
        }
    }
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

    fn announce_element(&self, element: ElementId) -> futures::future::BoxFuture<'static, usize> {
        let can_announce = self.can_announce;
        let key = provider_key(self.team, element);
        let targets =
            self.candidates
                .lock()
                .unwrap()
                .closest(key, self.my_id, PROVIDER_REPLICA_COUNT);
        let client = self.provider_client();
        Box::pin(async move {
            if !can_announce {
                return 0;
            }
            client.announce(key, element, targets).await
        })
    }

    fn find_element_providers(
        &self,
        element: ElementId,
    ) -> futures::future::BoxFuture<'static, Vec<PeerId>> {
        let can_find = self.can_fetch;
        let key = provider_key(self.team, element);
        let targets =
            self.candidates
                .lock()
                .unwrap()
                .closest(key, self.my_id, PROVIDER_REPLICA_COUNT);
        let client = self.provider_client();
        Box::pin(async move {
            if !can_find {
                return Vec::new();
            }
            client.find(key, element, targets).await
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
    snapshots: InventorySnapshots,
    installed_generation: Arc<Mutex<Option<InventoryGeneration>>>,
    cap: tokio::sync::watch::Receiver<Option<Arc<dyn NetCapability>>>,
    id: EndpointId,
}

impl NetSender {
    pub fn id(&self) -> EndpointId {
        self.id
    }

    pub(crate) fn update_snapshot(&self, mut snapshot: StoreSnapshot) {
        let mut slot = self.snapshot.lock().unwrap();
        let retired_components = slot.as_ref().map_or_else(Vec::new, |previous| {
            snapshot.reuse_unchanged_components(previous)
        });
        let manifest = snapshot.manifest();
        let notice = SnapshotNotice {
            generation: manifest.generation(),
            peers: snapshot.routing_peers(),
        };
        let snapshot = Arc::new(snapshot);
        let retired_blob_reader = self
            .snapshots
            .lock()
            .unwrap()
            .refresh_pinned_blob_reader(&snapshot);
        let retired_snapshot = slot.replace(snapshot);
        drop(slot);
        drop(retired_blob_reader);
        drop(retired_snapshot);
        drop(retired_components);

        let mut installed = self.installed_generation.lock().unwrap();
        if *installed != Some(notice.generation) {
            *installed = Some(notice.generation);
            let _ = self.cmd_tx.send(NetCommand::SnapshotInstalled(notice));
        }
    }

    pub fn clear_snapshot(&self) {
        let retired = self.snapshot.lock().unwrap().take();
        drop(retired);
        *self.installed_generation.lock().unwrap() = None;
    }

    #[cfg(test)]
    pub(crate) fn snapshot_available(&self) -> bool {
        self.snapshot.lock().unwrap().is_some()
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

    /// Best-effort lease renewal for an already-known element identity.
    /// Returns the number of receiver-local replicas which accepted the hint.
    pub async fn announce_element(&self, element: ElementId, budget: std::time::Duration) -> usize {
        tokio::time::timeout(budget, async {
            let capability = self.ready_capability().await.ok()?;
            Some(capability.announce_element(element).await)
        })
        .await
        .ok()
        .flatten()
        .unwrap_or(0)
    }

    /// Find soft provider hints for an already-known element identity.
    /// This cannot enumerate or discover element IDs.
    pub async fn find_element_providers(
        &self,
        element: ElementId,
        budget: std::time::Duration,
    ) -> Vec<PeerId> {
        tokio::time::timeout(budget, async {
            let capability = self.ready_capability().await.ok()?;
            Some(capability.find_element_providers(element).await)
        })
        .await
        .ok()
        .flatten()
        .unwrap_or_default()
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
    snapshots: InventorySnapshots,
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
    let snapshots = Arc::new(Mutex::new(InventorySnapshotCache::default()));
    let installed_generation = Arc::new(Mutex::new(None));
    let (cap_tx, cap_rx) = tokio::sync::watch::channel(None);
    (
        NetSender {
            cmd_tx,
            snapshot: snapshot.clone(),
            snapshots: snapshots.clone(),
            installed_generation,
            cap: cap_rx,
            id,
        },
        NetReceiver { evt_rx },
        HostWiring {
            cmd_rx,
            evt_tx,
            snapshot,
            snapshots,
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
/// Rendezvous replication factor over the currently known PEER set.
const PROVIDER_REPLICA_COUNT: usize = 20;
/// Bounded parallelism for the small provider-directory RPC fan-out.
const MAX_CONCURRENT_PROVIDER_RPCS: usize = 3;

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
        snapshots,
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
    let providers = Arc::new(Mutex::new(ProviderDirectory::default()));
    let _ = cap_tx.send(Some(Arc::new(NetCap {
        transport: transport.clone(),
        pool: pool.clone(),
        connect_proof: config.connect_proof.clone(),
        sync_proof: config.sync_proof.clone(),
        can_fetch: config.qos.direction.pulls(),
        can_announce: config.qos.direction.serves(),
        my_id,
        team: config.team,
        candidates: candidates.clone(),
        providers: providers.clone(),
    }) as Arc<dyn NetCapability>));

    let handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        snapshots,
        server: InventoryServerConfig::full_team(config.team),
        events: events.clone(),
        candidates: candidates.clone(),
        providers,
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
    snapshots: InventorySnapshots,
    cap_tx: tokio::sync::watch::Sender<Option<Arc<dyn NetCapability>>>,
}

impl From<HostWiring> for HostWiringParts {
    fn from(wiring: HostWiring) -> Self {
        Self {
            commands: wiring.cmd_rx,
            events: wiring.evt_tx,
            snapshot: wiring.snapshot,
            snapshots: wiring.snapshots,
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
                let request = walker
                    .next_request(|component, prefix| local.node_summary(component, prefix))?;
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
                local.contains_relative_key(component, key)
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

impl<T: Transport> ProviderClient<T> {
    async fn announce(&self, key: ProviderKey, element: ElementId, targets: Vec<PeerId>) -> usize {
        futures::stream::iter(targets)
            .map(|target| self.put(target, key, element))
            .buffer_unordered(MAX_CONCURRENT_PROVIDER_RPCS)
            .filter(|stored| futures::future::ready(*stored))
            .count()
            .await
    }

    async fn put(&self, target: PeerId, key: ProviderKey, element: ElementId) -> bool {
        if target == self.my_id {
            return self
                .providers
                .lock()
                .unwrap()
                .put(key, self.my_id, crate::clock::mono_now());
        }
        let connection = match inventory_pool_get(
            &self.transport,
            &self.pool,
            target,
            &self.connect_proof,
            &self.sync_proof,
        )
        .await
        {
            Ok(connection) => connection,
            Err(error) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider PUT setup failed");
                return false;
            }
        };
        match tokio::time::timeout(OP_DEADLINE, op_provider_put(&connection, &element)).await {
            Ok(Ok(stored)) => stored,
            Ok(Err(error)) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider PUT failed");
                pool_evict(&self.pool, target).await;
                false
            }
            Err(_) => {
                pool_evict(&self.pool, target).await;
                false
            }
        }
    }

    async fn find(
        &self,
        key: ProviderKey,
        element: ElementId,
        targets: Vec<PeerId>,
    ) -> Vec<PeerId> {
        let mut found: Vec<_> = futures::stream::iter(targets)
            .map(|target| self.get(target, key, element))
            .buffer_unordered(MAX_CONCURRENT_PROVIDER_RPCS)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .flatten()
            .collect();
        // Directory answers are routing hints, not authority. Drop malformed
        // endpoint identities and let exact transfer authenticate survivors.
        found.retain(|provider| EndpointId::from_bytes(provider).is_ok());
        found.sort_unstable();
        found.dedup();
        found
    }

    async fn get(&self, target: PeerId, key: ProviderKey, element: ElementId) -> Vec<PeerId> {
        if target == self.my_id {
            return self
                .providers
                .lock()
                .unwrap()
                .get(key, crate::clock::mono_now());
        }
        let connection = match inventory_pool_get(
            &self.transport,
            &self.pool,
            target,
            &self.connect_proof,
            &self.sync_proof,
        )
        .await
        {
            Ok(connection) => connection,
            Err(error) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider GET setup failed");
                return Vec::new();
            }
        };
        match tokio::time::timeout(OP_DEADLINE, op_provider_get(&connection, &element)).await {
            Ok(Ok(providers)) => providers,
            Ok(Err(error)) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider GET failed");
                pool_evict(&self.pool, target).await;
                Vec::new()
            }
            Err(_) => {
                pool_evict(&self.pool, target).await;
                Vec::new()
            }
        }
    }
}

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: SnapshotSlot,
    snapshots: InventorySnapshots,
    server: InventoryServerConfig,
    events: tokio::sync::mpsc::Sender<NetEventBatch>,
    candidates: RoutingCandidates,
    providers: Arc<Mutex<ProviderDirectory>>,
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
                let bytes = current.and_then(|snapshot| snapshot.get_blob(&hash));
                if let Some(bytes) = bytes {
                    send_u64_be(send, bytes.len() as u64).await?;
                    send.write_all(&bytes).await?;
                } else {
                    send_u64_be(send, u64::MAX).await?;
                }
            }
            OP_PROVIDER_PUT => {
                let session = current_inventory_session(&authorization)?;
                // This exact request shape is the forged-provider defense:
                // any appended claimed identity is rejected, and the value
                // stored below comes only from the authenticated connection.
                let element = recv_provider_element(recv).await?;
                let key = provider_key(session.team(), element);
                let stored = self.providers.lock().unwrap().put(
                    key,
                    peer.to_bytes(),
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
                let session = current_inventory_session(&authorization)?;
                let element = recv_provider_element(recv).await?;
                let key = provider_key(session.team(), element);
                let providers = self
                    .providers
                    .lock()
                    .unwrap()
                    .get(key, crate::clock::mono_now());
                debug_assert!(providers.len() <= crate::provider::MAX_PROVIDERS_PER_KEY);
                send_u8(
                    send,
                    u8::try_from(providers.len())
                        .expect("provider fan-out is statically bounded below u8::MAX"),
                )
                .await?;
                for provider in providers {
                    send_hash(send, &provider).await?;
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
                // SYNC_TEAM establishes a connection-local team session. A
                // read-only node may use that session for the soft provider
                // directory while its local direction policy still rejects
                // every inventory/blob disclosure operation below.
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
                if pinned.team() != session.team() {
                    anyhow::bail!("current snapshot belongs to another team");
                }
                let manifest = pinned.manifest();
                let retired = {
                    let mut snapshots = self.snapshots.lock().unwrap();
                    snapshots.pin_manifest(session.team(), &manifest, &pinned)
                };
                drop(retired);
                send_manifest(send, &manifest).await?;
            }
            OP_INVENTORY_NODE => {
                if !self.serve_inventory {
                    anyhow::bail!("local direction policy does not serve inventories");
                }
                let session = current_inventory_session(&authorization)?;
                let request = recv_node_request(recv, session).await?;
                let pinned = pinned_snapshot_if_serving(
                    &self.snapshot,
                    &self.snapshots,
                    session.team(),
                    request.component,
                    request.root,
                );
                let response = match pinned {
                    None => InventoryNodeResponse::SnapshotUnavailable,
                    Some(pinned) => pinned.inventory_node(&request)?,
                };
                send_node_response(send, session.team(), &request, &response).await?;
            }
            OP_INVENTORY_BLOB_RANGE => {
                if !self.serve_inventory {
                    anyhow::bail!("local direction policy does not serve inventories");
                }
                let session = current_inventory_session(&authorization)?;
                let request = recv_blob_range_request(recv).await?;
                let pinned = pinned_snapshot_if_serving(
                    &self.snapshot,
                    &self.snapshots,
                    session.team(),
                    InventoryComponent::Blob,
                    request.root,
                );
                let response = match pinned {
                    None => PinnedBlob::SnapshotUnavailable,
                    Some(pinned) => pinned.inventory_blob(request),
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

async fn recv_provider_element<R: tokio::io::AsyncRead + Unpin>(
    recv: &mut R,
) -> anyhow::Result<ElementId> {
    let element = recv_hash(recv).await?;
    require_stream_eof(recv).await?;
    Ok(element)
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
        OP_PROVIDER_PUT => "PROVIDER_PUT",
        OP_PROVIDER_GET => "PROVIDER_GET",
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
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use triblespace_core::collection::{
        CollectionCommit, CollectionData, CollectionRecord, CollectionStore,
    };
    use triblespace_core::inline::Inline;
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::repo::memoryrepo::MemoryRepo;

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

    #[test]
    fn provider_replica_selection_is_deterministic_and_bounded() {
        let mut routes = RoutingTable::new(Vec::new());
        for byte in 1..=100 {
            routes.note([byte; 32]);
        }
        let selected = routes.closest([0; 32], [0; 32], 20);
        assert_eq!(selected.len(), 20);
        assert_eq!(selected, (0..20).map(|byte| [byte; 32]).collect::<Vec<_>>());
        assert_eq!(selected, routes.closest([0; 32], [0; 32], 20));
    }

    #[tokio::test]
    async fn provider_request_has_no_forgeable_provider_field() {
        let element = [7; 32];
        let (mut send, mut recv) = tokio::io::duplex(64);
        send.write_all(&element).await.unwrap();
        send.shutdown().await.unwrap();
        assert_eq!(recv_provider_element(&mut recv).await.unwrap(), element);

        let (mut send, mut recv) = tokio::io::duplex(64);
        send.write_all(&element).await.unwrap();
        send.write_all(&[8; 32]).await.unwrap();
        send.shutdown().await.unwrap();
        assert!(
            recv_provider_element(&mut recv).await.is_err(),
            "an appended claimed provider identity must invalidate the request"
        );
    }

    #[test]
    fn clearing_current_snapshot_also_gates_old_pinned_roots() {
        let team = key(3);
        let mut store = MemoryRepo::default();
        store.insert_peer(PeerEvidence::new(team, key(4))).unwrap();
        let snapshot = Arc::new(StoreSnapshot::from_store(&mut store, team).unwrap());
        let manifest = snapshot.manifest();
        let component = InventoryComponent::Peer;
        let root = manifest
            .component(component)
            .root()
            .expect("nonempty peer inventory has a root");
        let current = Arc::new(Mutex::new(Some(snapshot.clone())));
        let snapshots = Arc::new(Mutex::new(InventorySnapshotCache::default()));
        drop(
            snapshots
                .lock()
                .unwrap()
                .pin_manifest(team, &manifest, &snapshot),
        );

        assert!(pinned_snapshot_if_serving(&current, &snapshots, team, component, root).is_some());
        *current.lock().unwrap() = None;
        assert!(
            pinned_snapshot_if_serving(&current, &snapshots, team, component, root).is_none(),
            "withdrawal must gate immutable roots retained in the cache"
        );
    }

    fn commit(author: &SigningKey, byte: u8) -> CollectionRecord {
        CollectionRecord::Commit(CollectionCommit::sign(
            author,
            Inline::new([0x31; 32]),
            CollectionData::new([byte; 32]),
            Inline::new([0x32; 32]),
        ))
    }

    #[test]
    fn production_component_serves_the_body_frozen_with_its_value_leaf() {
        let team = key(1);
        let record = commit(&SigningKey::from_bytes(&[3; 32]), 7);
        let mut store = MemoryRepo::default();
        CollectionStore::insert(&mut store, record).unwrap();
        let snapshot = StoreSnapshot::from_store(&mut store, team).unwrap();
        let component = snapshot
            .manifest()
            .component(InventoryComponent::CollectionRecord);
        let root = component.root().unwrap();
        let request = InventoryNodeRequest::new(
            team,
            InventoryComponent::CollectionRecord,
            root,
            component.leaf_count(),
            Vec::new(),
            root,
        )
        .unwrap();

        let response = snapshot
            .component(InventoryComponent::CollectionRecord)
            .inventory_node(&request)
            .unwrap();
        let InventoryNodeResponse::Found(crate::inventory_wire::InventoryNode::Leaf {
            leaf, ..
        }) = response
        else {
            panic!("single-record production component must resolve to one leaf")
        };
        assert_eq!(leaf.key, record.id().raw());
        assert_eq!(leaf.value, InventoryLeafValue::CollectionRecord(record));

        let unavailable = InventoryNodeRequest::new(
            team,
            InventoryComponent::CollectionRecord,
            [9; 32],
            component.leaf_count(),
            Vec::new(),
            [9; 32],
        )
        .unwrap();
        assert_eq!(
            snapshot
                .component(InventoryComponent::CollectionRecord)
                .inventory_node(&unavailable)
                .unwrap(),
            InventoryNodeResponse::SnapshotUnavailable
        );
    }

    #[test]
    fn record_churn_reuses_the_blob_tree_but_refreshes_its_reader_component() {
        let team = key(1);
        let author = SigningKey::from_bytes(&[2; 32]);
        let mut store = MemoryRepo::default();
        store
            .put::<UnknownBlob, _>(Bytes::from_source(vec![7; 1024]))
            .unwrap();

        let first = StoreSnapshot::from_store(&mut store, team).unwrap();
        let first_blob = first.component(InventoryComponent::Blob).clone();
        let InventoryComponentData::Blob {
            inventory: first_blob_tree,
            ..
        } = &first_blob.data
        else {
            unreachable!()
        };
        let first_blob_tree = first_blob_tree.clone();
        let first_records = first
            .component(InventoryComponent::CollectionRecord)
            .clone();
        let first_peers = first.component(InventoryComponent::Peer).clone();
        let first_proofs = first.component(InventoryComponent::CapabilityProof).clone();

        CollectionStore::insert(&mut store, commit(&author, 1)).unwrap();
        let mut second = StoreSnapshot::from_store(&mut store, team).unwrap();
        drop(second.reuse_unchanged_components(&first));

        assert!(!Arc::ptr_eq(
            &first_blob,
            second.component(InventoryComponent::Blob)
        ));
        let InventoryComponentData::Blob {
            inventory: second_blob_tree,
            ..
        } = &second.component(InventoryComponent::Blob).data
        else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(&first_blob_tree, second_blob_tree));
        assert!(!Arc::ptr_eq(
            &first_records,
            second.component(InventoryComponent::CollectionRecord)
        ));
        assert!(Arc::ptr_eq(
            &first_peers,
            second.component(InventoryComponent::Peer)
        ));
        assert!(Arc::ptr_eq(
            &first_proofs,
            second.component(InventoryComponent::CapabilityProof)
        ));
    }

    #[test]
    fn pinned_components_do_not_retain_the_aggregate_snapshot() {
        let team = key(1);
        let author = SigningKey::from_bytes(&[2; 32]);
        let mut store = MemoryRepo::default();
        CollectionStore::insert(&mut store, commit(&author, 1)).unwrap();
        let snapshot = Arc::new(StoreSnapshot::from_store(&mut store, team).unwrap());
        let weak = Arc::downgrade(&snapshot);
        let manifest = snapshot.manifest();
        let record_root = manifest
            .component(InventoryComponent::CollectionRecord)
            .root()
            .unwrap();
        let mut cache = InventorySnapshotCache::default();
        drop(cache.pin_manifest(team, &manifest, &snapshot));

        drop(snapshot);
        assert!(weak.upgrade().is_none());
        assert!(
            cache
                .get(team, InventoryComponent::CollectionRecord, record_root)
                .is_some()
        );
    }

    struct DropBlobProbe {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropBlobProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    impl BlobSnapshotReader for DropBlobProbe {
        fn get_blob(&self, _hash: RawHash) -> Option<Bytes> {
            None
        }
    }

    fn replace_blob_reader(snapshot: &mut StoreSnapshot, reader: Arc<dyn BlobSnapshotReader>) {
        let index = InventoryComponent::Blob.index();
        let previous = &snapshot.components[index];
        let InventoryComponentData::Blob { inventory, .. } = &previous.data else {
            unreachable!()
        };
        snapshot.components[index] = Arc::new(InventoryComponentSnapshot {
            team: previous.team,
            manifest: previous.manifest,
            data: InventoryComponentData::Blob {
                inventory: inventory.clone(),
                reader,
            },
        });
    }

    #[test]
    fn installing_same_root_snapshot_refreshes_an_already_pinned_blob_reader() {
        let team = key(1);
        let endpoint = EndpointId::from_bytes(team.as_bytes()).unwrap();
        let (sender, _receiver, wiring) = wire(endpoint);
        let mut store = MemoryRepo::default();
        store
            .put::<UnknownBlob, _>(Bytes::from_source(vec![7; 1024]))
            .unwrap();
        let old_drops = Arc::new(AtomicUsize::new(0));
        let new_drops = Arc::new(AtomicUsize::new(0));

        let mut old = StoreSnapshot::from_store(&mut store, team).unwrap();
        replace_blob_reader(
            &mut old,
            Arc::new(DropBlobProbe {
                drops: old_drops.clone(),
            }),
        );
        let old_blob = Arc::downgrade(old.component(InventoryComponent::Blob));
        let manifest = old.manifest();
        let blob_root = manifest.component(InventoryComponent::Blob).root().unwrap();
        sender.update_snapshot(old);
        {
            let current = sender.snapshot.lock().unwrap().as_ref().unwrap().clone();
            let retired = {
                let mut snapshots = wiring.snapshots.lock().unwrap();
                snapshots.pin_manifest(team, &manifest, &current)
            };
            drop(retired);
        }

        let mut new = StoreSnapshot::from_store(&mut store, team).unwrap();
        replace_blob_reader(
            &mut new,
            Arc::new(DropBlobProbe {
                drops: new_drops.clone(),
            }),
        );
        // No second manifest request occurs. Installing the equivalent Blob
        // root itself must update the cache's backend reader lease.
        sender.update_snapshot(new);

        assert!(old_blob.upgrade().is_none());
        assert_eq!(old_drops.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(new_drops.load(AtomicOrdering::SeqCst), 0);
        assert!(
            wiring
                .snapshots
                .lock()
                .unwrap()
                .get(team, InventoryComponent::Blob, blob_root)
                .is_some()
        );

        sender.clear_snapshot();
        drop(sender);
        drop(wiring);
        assert_eq!(new_drops.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn blob_only_churn_retains_at_most_eight_component_generations() {
        let team = key(1);
        let mut store = MemoryRepo::default();
        let mut cache = InventorySnapshotCache::default();
        let mut generations = Vec::new();

        for byte in 0..12 {
            store
                .put::<UnknownBlob, _>(Bytes::from_source(vec![byte; 257]))
                .unwrap();
            let snapshot = Arc::new(StoreSnapshot::from_store(&mut store, team).unwrap());
            drop(cache.pin_manifest(team, &snapshot.manifest(), &snapshot));
            generations.push(Arc::downgrade(snapshot.component(InventoryComponent::Blob)));
        }

        let retained = cache
            .snapshots
            .keys()
            .filter(|key| key.component == InventoryComponent::Blob)
            .count();
        assert_eq!(retained, MAX_PINNED_ROOTS_PER_COMPONENT);
        assert!(generations[..4].iter().all(|weak| weak.upgrade().is_none()));
        assert!(generations[4..].iter().all(|weak| weak.upgrade().is_some()));
    }

    struct ConcurrentBlobState {
        entered: AtomicUsize,
        active: AtomicUsize,
        maximum: AtomicUsize,
        gate: Mutex<()>,
        wake: Condvar,
    }

    struct ConcurrentBlobProbe(Arc<ConcurrentBlobState>);

    impl BlobSnapshotReader for ConcurrentBlobProbe {
        fn get_blob(&self, _hash: RawHash) -> Option<Bytes> {
            let active = self.0.active.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            self.0.maximum.fetch_max(active, AtomicOrdering::SeqCst);
            self.0.entered.fetch_add(1, AtomicOrdering::SeqCst);
            self.0.wake.notify_all();
            let gate = self.0.gate.lock().unwrap();
            let _ = self
                .0
                .wake
                .wait_timeout_while(gate, Duration::from_secs(1), |_| {
                    self.0.entered.load(AtomicOrdering::SeqCst) < 2
                })
                .unwrap();
            self.0.active.fetch_sub(1, AtomicOrdering::SeqCst);
            Some(Bytes::from_source(vec![1]))
        }
    }

    #[test]
    fn immutable_component_reads_are_not_serialized_by_a_snapshot_mutex() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StoreSnapshot>();

        let team = key(1);
        let hash = [9; 32];
        let inventory = BlobInventory::from_keys([hash]);
        let state = Arc::new(ConcurrentBlobState {
            entered: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            maximum: AtomicUsize::new(0),
            gate: Mutex::new(()),
            wake: Condvar::new(),
        });
        let component = Arc::new(InventoryComponentSnapshot {
            team,
            manifest: crate::inventory::ComponentManifest::new(
                InventoryComponent::Blob,
                1,
                inventory.merkle_root(),
            ),
            data: InventoryComponentData::Blob {
                inventory: Arc::new(inventory),
                reader: Arc::new(ConcurrentBlobProbe(state.clone())),
            },
        });

        let left = component.clone();
        let right = component.clone();
        let left = std::thread::spawn(move || left.get_blob(hash).unwrap());
        let right = std::thread::spawn(move || right.get_blob(hash).unwrap());
        let _ = left.join().unwrap();
        let _ = right.join().unwrap();

        // The probe can only release both readers if their calls overlap.
        assert_eq!(state.maximum.load(AtomicOrdering::SeqCst), 2);
    }
}
