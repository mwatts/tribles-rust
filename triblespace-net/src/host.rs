//! Async network host for one authorized team inventory.
//!
//! The synchronous [`crate::peer::Peer`] owns the store. This module owns
//! transport, connection authentication, immutable serving snapshots, and the
//! bounded periodic anti-entropy scheduler. Every useful byte arrives through
//! a CONNECT- and SYNC_TEAM-authorized direct connection and is checked against
//! a pinned PATCH root.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::io::Write as _;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use anybytes::Bytes;
use ed25519_dalek::{SigningKey, VerifyingKey};
use futures::{StreamExt, stream::FuturesUnordered};
use iroh_base::{EndpointAddr, EndpointId};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{Instrument, debug, debug_span, info_span, instrument, warn};
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::capability::{
    CapabilityMode, CapabilityProofBundle, CapabilityProofId, CapabilityRequest, CapabilityValidity,
};
use triblespace_core::collection::CollectionStore;
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{IdentitySchema, PATCH};
use triblespace_core::repo::peer::PeerEvidence;
use triblespace_core::repo::{
    ArtifactOfferSnapshot, BlobStore, BlobStoreGet, CapabilityProofStore, PeerStore,
};

use crate::channel::{
    MAX_ADMISSION_BRIDGE_BATCHES, NetCommand, NetEvent, NetEventBatch, SnapshotNotice,
};
use crate::identity::iroh_secret;
use crate::inventory::{
    AuthorizedInventorySession, BlobInventory, CapabilityProofInventory, CollectionRecordInventory,
    InventoryComponent, InventoryGeneration, InventoryManifest, InventoryServerConfig,
    InventorySnapshot, PeerInventory, ReconcileQos, sync_team_capability_atom,
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
use crate::provider::{
    ArtifactId, PROVIDER_LEASE_LIFETIME, ProviderCover, ProviderCoverBuild, ProviderDirectory,
    ProviderKey, ProviderProbe, ProviderShard, ProviderShardCandidate, provider_key,
    provider_prefix_key,
};
use crate::routing::{ALPHA, IterativeLookup, K, RoutingKey, RoutingTable};
use crate::transport::{Conn, Harness, PeerId, Transport};

/// Configuration for a peer attached to one single-team store.
///
/// `team` is simultaneously the capability trust root and inventory scope.
/// The backing store must not mix team-unscoped records, proofs, or blobs from
/// another team.
#[derive(Clone)]
pub struct PeerConfig {
    /// Bootstrap routes. Stored PEER evidence can add routing candidates but
    /// never grants authority.
    pub peers: Vec<EndpointAddr>,
    /// Team trust root and inventory scope.
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

fn validate_local_authorizations(
    config: &PeerConfig,
    endpoint: VerifyingKey,
    now: hifitime::Epoch,
) -> anyhow::Result<()> {
    crate::protocol::verify_endpoint_proof(
        &config.connect_proof,
        config.team,
        endpoint,
        CapabilityRequest::new(connect_capability_atom(config.team), CapabilityMode::Invoke),
        now,
        "local CONNECT",
    )?;
    crate::protocol::verify_endpoint_proof(
        &config.sync_proof,
        config.team,
        endpoint,
        CapabilityRequest::new(
            sync_team_capability_atom(config.team),
            CapabilityMode::Invoke,
        ),
        now,
        "local SYNC_TEAM",
    )?;
    Ok(())
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

/// The async capability cloned into lazy readers. Exact GET_BLOB and broad
/// inventory enumeration both require the connection-local SYNC_TEAM session;
/// the content hash narrows the request but is not itself disclosure authority.
pub(crate) trait NetCapability: Send + Sync {
    fn fetch_blob(&self, hash: RawHash) -> futures::future::BoxFuture<'static, Option<Vec<u8>>>;
    fn find_artifact_providers(
        &self,
        artifact: ArtifactId,
    ) -> futures::future::BoxFuture<'static, Vec<PeerId>>;
}

type RoutingCandidates = Arc<Mutex<RoutingTable>>;

struct PoolEntry<C> {
    connection: tokio::sync::OnceCell<AuthenticatedConnection<C>>,
    inventory_auth: tokio::sync::OnceCell<RemoteAuthorization>,
}

#[derive(Clone)]
struct AuthenticatedConnection<C> {
    connection: C,
    validity: Option<CapabilityValidity>,
}

impl<C> AuthenticatedConnection<C> {
    fn is_current_at(&self, now: hifitime::Epoch) -> bool {
        self.validity.is_none_or(|validity| validity.contains(now))
    }
}

#[derive(Clone, Copy)]
struct RemoteAuthorization {
    validity: Option<CapabilityValidity>,
}

impl RemoteAuthorization {
    fn is_current_at(self, now: hifitime::Epoch) -> bool {
        self.validity.is_none_or(|validity| validity.contains(now))
    }
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
    team: VerifyingKey,
    candidates: RoutingCandidates,
    providers: Arc<Mutex<ProviderDirectory>>,
}

#[derive(Clone)]
struct ProviderClient<T: Transport> {
    transport: T,
    pool: SharedPool<T::Conn>,
    connect_proof: CapabilityProofBundle,
    sync_proof: CapabilityProofBundle,
    providers: Arc<Mutex<ProviderDirectory>>,
    candidates: RoutingCandidates,
    my_id: PeerId,
    team: VerifyingKey,
}

impl<T: Transport> NetCap<T> {
    fn provider_client(&self) -> ProviderClient<T> {
        ProviderClient {
            transport: self.transport.clone(),
            pool: self.pool.clone(),
            connect_proof: self.connect_proof.clone(),
            sync_proof: self.sync_proof.clone(),
            providers: self.providers.clone(),
            candidates: self.candidates.clone(),
            my_id: self.my_id,
            team: self.team,
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
        let team = self.team;
        let candidates = self.candidates.clone();
        let client = self.provider_client();
        Box::pin(async move {
            if !can_fetch {
                return None;
            }

            let providers = client.find_artifact(hash).await;
            fetch_from_providers(
                &transport,
                &hash,
                &pool,
                &providers,
                team,
                &connect_proof,
                &sync_proof,
                &candidates,
            )
            .await
        })
    }

    fn find_artifact_providers(
        &self,
        artifact: ArtifactId,
    ) -> futures::future::BoxFuture<'static, Vec<PeerId>> {
        let can_find = self.can_fetch;
        let client = self.provider_client();
        Box::pin(async move {
            if !can_find {
                return Vec::new();
            }
            client.find_artifact(artifact).await
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
            peers: snapshot.routing_peers(),
            blob: Some(manifest.component(InventoryComponent::Blob)),
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
        if *installed != Some(manifest.generation()) {
            *installed = Some(manifest.generation());
            let _ = self.cmd_tx.send(NetCommand::SnapshotChanged(notice));
        }
    }

    pub(crate) fn update_artifact_offers(&self, offers: ArtifactOfferSnapshot) {
        let _ = self.cmd_tx.send(NetCommand::ArtifactOffersUpdated(offers));
    }

    pub fn clear_snapshot(&self) {
        let retired = self.snapshot.lock().unwrap().take();
        let had_snapshot = retired.is_some();
        drop(retired);
        *self.installed_generation.lock().unwrap() = None;
        if had_snapshot {
            let _ = self
                .cmd_tx
                .send(NetCommand::SnapshotChanged(SnapshotNotice {
                    peers: Vec::new(),
                    blob: None,
                }));
        }
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

    /// Find soft provider hints for an already-known physical artifact.
    /// This cannot enumerate or discover artifact IDs.
    pub async fn find_artifact_providers(
        &self,
        artifact: ArtifactId,
        budget: std::time::Duration,
    ) -> Vec<PeerId> {
        tokio::time::timeout(budget, async {
            let capability = self.ready_capability().await.ok()?;
            Some(capability.find_artifact_providers(artifact).await)
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
    let endpoint = VerifyingKey::from_bytes(id.as_bytes())
        .map_err(|error| anyhow::anyhow!("invalid local endpoint identity: {error}"))?;
    validate_local_authorizations(&config, endpoint, crate::clock::epoch_now())?;
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
const HOST_POLL_PERIOD: std::time::Duration = std::time::Duration::from_millis(10);
const INBOUND_AUTH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);
const INBOUND_CONNECTION_IDLE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);
const INBOUND_REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
pub(crate) const MAX_INBOUND_CONNECTIONS_GLOBAL: usize = 64;
pub(crate) const MAX_INBOUND_REQUESTS_PER_CONNECTION: usize = 16;
const MAX_INBOUND_REQUESTS_GLOBAL: usize = 64;
const MAX_CONCURRENT_SWEEPS: usize = 8;
/// Maximum number of new pairwise reconciliations admitted by one period.
///
/// Reusing the DHT replication width gives the scheduler one fixed natural
/// budget instead of a second deployment knob. The cursor carries fairness
/// across periods when the stored PEER set is larger than this bound.
const SWEEPS_PER_PERIOD: usize = K;
/// Bounded parallelism for provider-directory shard fan-out.
const MAX_CONCURRENT_PROVIDER_RPCS: usize = ALPHA;
/// Bound whole-prefix publication independently of the alpha-bounded RPC
/// fan-out inside one DHT announcement. Work therefore never scales with the
/// number of offered artifacts.
const MAX_CONCURRENT_PREFIX_ANNOUNCEMENTS: usize = ALPHA;
/// Renew with half the receiver-selected lease still remaining. Scheduling
/// from successful completion absorbs lookup latency without accumulating
/// catch-up bursts after a delayed host iteration.
const PROVIDER_RENEWAL_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(PROVIDER_LEASE_LIFETIME.as_secs() / 2);
/// A missed lease is actionable, but one overloaded scheduler must not fill
/// logs every 10ms while it works through the same backlog.
const OFFER_BACKLOG_WARNING_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);
/// One lookup must fit comfortably inside the default interactive operation.
const DHT_LOOKUP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(3);
/// Deadline for small provider control RPCs and authenticated connection setup.
/// Exact artifact bodies deliberately are not capped by this deadline.
const PROVIDER_CONTROL_RPC_DEADLINE: std::time::Duration = std::time::Duration::from_secs(1);
/// Leave at least half of the public operation budget for exact blob bytes.
const PROVIDER_CONTROL_PHASE_DEADLINE: std::time::Duration = std::time::Duration::from_secs(3);
/// Changed shard bodies are bounded but intentionally much larger than DHT
/// control frames.
const PROVIDER_SHARD_PUBLICATION_DEADLINE: std::time::Duration = std::time::Duration::from_secs(45);
/// Admit one bounded second alpha batch without cancelling already progressing
/// bodies. A full second avoids multiplying ordinary medium-sized transfers.
const PROVIDER_FETCH_HEDGE_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

struct SweepOutcome {
    peer: PeerId,
    success: bool,
}

/// Bounded fair admission for periodic pairwise anti-entropy.
///
/// `pending` contains only peers admitted by the current period, never every
/// known peer and never a backoff-delayed peer. It is refilled only after it is
/// empty and a period is due, so slow sweeps cannot accumulate catch-up work.
/// The lexicographic cursor is an identity rather than a vector index, keeping
/// progress well-defined when grow-only peer evidence inserts around it.
struct SweepScheduler {
    next_period: Option<crate::clock::Mono>,
    cursor: Option<PeerId>,
    pending: VecDeque<PeerId>,
}

impl SweepScheduler {
    fn new() -> Self {
        Self {
            next_period: None,
            cursor: None,
            pending: VecDeque::with_capacity(SWEEPS_PER_PERIOD),
        }
    }

    /// Arm the first period when the serving snapshot exists. Further
    /// generations deliberately do not move the deadline.
    fn observe_snapshot(&mut self, now: crate::clock::Mono) {
        self.next_period.get_or_insert(now);
    }

    fn period_is_due(&self, now: crate::clock::Mono) -> bool {
        self.pending.is_empty()
            && self
                .next_period
                .is_some_and(|next_period| now >= next_period)
    }

    fn admit_period(
        &mut self,
        now: crate::clock::Mono,
        candidates: &[PeerId],
        in_flight: &HashSet<PeerId>,
        failures: &HashMap<PeerId, (u32, crate::clock::Mono)>,
    ) -> usize {
        let Some(next_period) = self.next_period else {
            return 0;
        };
        if now < next_period || !self.pending.is_empty() {
            return 0;
        }
        // Never replay missed periods in a burst. One delayed host iteration
        // admits one budget and starts the next interval from its observation.
        self.next_period = Some(now + INVENTORY_SWEEP_PERIOD);
        if candidates.is_empty() {
            return 0;
        }

        let start = self
            .cursor
            .map_or(0, |cursor| match candidates.binary_search(&cursor) {
                Ok(index) => (index + 1) % candidates.len(),
                Err(index) if index < candidates.len() => index,
                Err(_) => 0,
            });
        let mut admitted = 0;
        for offset in 0..candidates.len() {
            let peer = candidates[(start + offset) % candidates.len()];
            self.cursor = Some(peer);
            if in_flight.contains(&peer)
                || failures
                    .get(&peer)
                    .is_some_and(|(_, retry_at)| now < *retry_at)
            {
                continue;
            }
            self.pending.push_back(peer);
            admitted += 1;
            if admitted == SWEEPS_PER_PERIOD {
                break;
            }
        }
        admitted
    }

    fn pop(&mut self) -> Option<PeerId> {
        self.pending.pop_front()
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

struct ProviderShardAnnouncementOutcome {
    prefix: u8,
    digest: [u8; 32],
    attempted_at: crate::clock::Mono,
    publication: ProviderPublication,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ProviderPublication {
    local_accepted: bool,
    remote_expected: bool,
    remote_accepted: usize,
}

impl ProviderPublication {
    fn succeeded(self) -> bool {
        self.remote_accepted > 0 || (!self.remote_expected && self.local_accepted)
    }
}

/// Bounded fair scheduling over changed or due provider-cover prefixes.
///
/// A cover may contain any number of artifacts, but has at most 256 schedule
/// units. Unchanged roots retain their deadlines; a changed root is published
/// immediately and a prefix omitted from the next cover simply stops renewing.
struct ProviderCoverScheduler {
    cover: ProviderCover,
    due: BTreeSet<(crate::clock::Mono, u8)>,
    due_by_prefix: HashMap<u8, crate::clock::Mono>,
    lease_deadlines: BTreeSet<(crate::clock::Mono, u8)>,
    lease_deadline_by_prefix: HashMap<u8, crate::clock::Mono>,
    in_flight: HashSet<u8>,
    failures: HashMap<u8, u32>,
    next_backlog_warning: Option<crate::clock::Mono>,
}

impl ProviderCoverScheduler {
    fn new() -> Self {
        Self {
            cover: ProviderCover::default(),
            due: BTreeSet::new(),
            due_by_prefix: HashMap::new(),
            lease_deadlines: BTreeSet::new(),
            lease_deadline_by_prefix: HashMap::new(),
            in_flight: HashSet::new(),
            failures: HashMap::new(),
            next_backlog_warning: None,
        }
    }

    fn observe_cover(&mut self, cover: ProviderCover, now: crate::clock::Mono) {
        if self.cover.same_membership(&cover) {
            self.cover = cover;
            return;
        }
        let removed: Vec<_> = self
            .cover
            .iter()
            .filter_map(|(&prefix, _)| cover.get(prefix).is_none().then_some(prefix))
            .collect();
        for prefix in removed {
            self.unschedule(prefix);
            self.remove_lease_deadline(prefix);
            self.failures.remove(&prefix);
        }

        let changed: Vec<_> = cover
            .iter()
            .filter_map(|(&prefix, shard)| {
                let unchanged = self.cover.get(prefix).is_some_and(|old| {
                    old.digest() == shard.digest() && old.count() == shard.count()
                });
                (!unchanged).then_some(prefix)
            })
            .collect();
        self.cover = cover;
        for prefix in changed {
            self.failures.remove(&prefix);
            if !self.in_flight.contains(&prefix) {
                self.schedule(prefix, now);
            }
        }
    }

    fn schedule(&mut self, prefix: u8, due: crate::clock::Mono) {
        if let Some(previous) = self.due_by_prefix.insert(prefix, due) {
            self.due.remove(&(previous, prefix));
        }
        self.due.insert((due, prefix));
    }

    fn unschedule(&mut self, prefix: u8) {
        if let Some(previous) = self.due_by_prefix.remove(&prefix) {
            self.due.remove(&(previous, prefix));
        }
    }

    fn set_lease_deadline(&mut self, prefix: u8, deadline: crate::clock::Mono) {
        if let Some(previous) = self.lease_deadline_by_prefix.insert(prefix, deadline) {
            self.lease_deadlines.remove(&(previous, prefix));
        }
        self.lease_deadlines.insert((deadline, prefix));
    }

    fn remove_lease_deadline(&mut self, prefix: u8) {
        if let Some(previous) = self.lease_deadline_by_prefix.remove(&prefix) {
            self.lease_deadlines.remove(&(previous, prefix));
        }
    }

    fn pop_due(&mut self, now: crate::clock::Mono) -> Option<ProviderShard> {
        loop {
            let (due, prefix) = self.due.first().copied()?;
            if due > now {
                return None;
            }
            self.due.remove(&(due, prefix));
            self.due_by_prefix.remove(&prefix);
            let Some(shard) = self.cover.get(prefix).cloned() else {
                continue;
            };
            if self.in_flight.insert(prefix) {
                return Some(shard);
            }
        }
    }

    fn complete(&mut self, outcome: ProviderShardAnnouncementOutcome, now: crate::clock::Mono) {
        self.in_flight.remove(&outcome.prefix);
        let Some(current) = self.cover.get(outcome.prefix) else {
            return;
        };
        if current.digest() != outcome.digest {
            self.schedule(outcome.prefix, now);
            return;
        }

        let delay = if outcome.publication.succeeded() {
            self.failures.remove(&outcome.prefix);
            // The receiver installs its lease during the attempt. Starting at
            // launch is a conservative lower bound on every accepted remote
            // lease deadline; completion time could overstate it by a whole
            // control-phase timeout.
            self.set_lease_deadline(
                outcome.prefix,
                outcome.attempted_at + PROVIDER_LEASE_LIFETIME,
            );
            PROVIDER_RENEWAL_INTERVAL
        } else {
            let attempts = self
                .failures
                .get(&outcome.prefix)
                .copied()
                .unwrap_or(0)
                .saturating_add(1);
            self.failures.insert(outcome.prefix, attempts);
            let shift = attempts.saturating_sub(1).min(31);
            crate::RETRY_BACKOFF_BASE
                .saturating_mul(1u32 << shift)
                .min(crate::RETRY_BACKOFF_CAP)
        };
        self.schedule(outcome.prefix, now + delay);
    }

    /// Report a definite lease miss at most once per warning interval. The
    /// ordered deadline set makes the normal not-overdue check O(1), without
    /// truncating or separately planning the fair work queue.
    fn warnable_expired_lease(
        &mut self,
        now: crate::clock::Mono,
    ) -> Option<(u8, crate::clock::Mono)> {
        let (deadline, prefix) = self.lease_deadlines.first().copied()?;
        if deadline > now
            || self
                .next_backlog_warning
                .is_some_and(|next_warning| next_warning > now)
        {
            return None;
        }
        self.next_backlog_warning = Some(now + OFFER_BACKLOG_WARNING_INTERVAL);
        Some((prefix, deadline))
    }

    fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    #[cfg(test)]
    fn active_len(&self) -> usize {
        self.cover.shard_count()
    }

    #[cfg(test)]
    fn next_due(&self, prefix: u8) -> Option<crate::clock::Mono> {
        self.due_by_prefix.get(&prefix).copied()
    }

    #[cfg(test)]
    fn lease_deadline(&self, prefix: u8) -> Option<crate::clock::Mono> {
        self.lease_deadline_by_prefix.get(&prefix).copied()
    }
}

fn active_provider_cover(
    offers: &ArtifactOfferSnapshot,
    snapshot: &SnapshotSlot,
    serves: bool,
    team: VerifyingKey,
) -> ProviderCoverBuild {
    if !serves {
        return ProviderCoverBuild::default();
    }
    let current = snapshot.lock().unwrap().as_ref().cloned();
    let Some(current) = current else {
        return ProviderCoverBuild::default();
    };
    ProviderCover::from_artifacts(
        team,
        offers.iter().filter_map(|handle| {
            current
                .contains_relative_key(InventoryComponent::Blob, &handle.raw)
                .then_some(handle.raw)
        }),
    )
}

async fn host_loop<T: Transport>(harness: Harness<T>, config: PeerConfig, wiring: HostWiring) {
    let Harness {
        transport,
        mut incoming,
    } = harness;
    let HostWiringParts {
        commands,
        events,
        snapshot,
        snapshots,
        cap_tx,
    } = HostWiringParts::from(wiring);
    let my_id = transport.local_id();
    let endpoint = match VerifyingKey::from_bytes(&my_id) {
        Ok(endpoint) => endpoint,
        Err(error) => {
            warn!(%error, "invalid local transport identity");
            transport.shutdown().await;
            return;
        }
    };
    if let Err(error) = validate_local_authorizations(&config, endpoint, crate::clock::epoch_now())
    {
        warn!(%error, "local outbound capability proofs do not authorize this endpoint");
        transport.shutdown().await;
        return;
    }
    let mut configured: Vec<_> = config
        .peers
        .iter()
        .map(|address| *address.id.as_bytes())
        .filter(|peer| *peer != my_id)
        .collect();
    configured.sort_unstable();
    configured.dedup();
    let candidates = Arc::new(Mutex::new(RoutingTable::new(my_id, configured)));
    let pool = new_shared_pool();
    let providers = Arc::new(Mutex::new(ProviderDirectory::default()));
    let net_cap = Arc::new(NetCap {
        transport: transport.clone(),
        pool: pool.clone(),
        connect_proof: config.connect_proof.clone(),
        sync_proof: config.sync_proof.clone(),
        can_fetch: config.qos.direction.pulls(),
        my_id,
        team: config.team,
        candidates: candidates.clone(),
        providers: providers.clone(),
    });
    let publication_client = net_cap.provider_client();
    let _ = cap_tx.send(Some(net_cap as Arc<dyn NetCapability>));

    let handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        snapshots,
        server: InventoryServerConfig::full_team(config.team),
        local: endpoint,
        connect_proof: config.connect_proof.clone(),
        sync_proof: config.sync_proof.clone(),
        events: events.clone(),
        candidates: candidates.clone(),
        providers: providers.clone(),
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

    let (sweep_tx, mut sweep_rx) = tokio::sync::mpsc::unbounded_channel::<SweepOutcome>();
    let mut in_flight = HashSet::new();
    let mut failures: HashMap<PeerId, (u32, crate::clock::Mono)> = HashMap::new();
    let mut sweeps = SweepScheduler::new();
    let (offer_tx, mut offer_rx) =
        tokio::sync::mpsc::unbounded_channel::<ProviderShardAnnouncementOutcome>();
    let mut artifact_offers = ArtifactOfferSnapshot::default();
    let mut offer_scheduler = ProviderCoverScheduler::new();
    let mut publication_blob = None;
    let commands = commands;

    loop {
        let mut disconnected = false;
        let mut publication_inputs_changed = false;
        loop {
            match commands.try_recv() {
                Ok(NetCommand::SnapshotChanged(notice)) => {
                    let mut routes = candidates.lock().unwrap();
                    for peer in notice.peers {
                        routes.note_sync_candidate(peer);
                    }
                    drop(routes);
                    // The first immutable serving view starts anti-entropy
                    // immediately. Later local generations update stored-peer
                    // evidence but cannot reset or multiply the fixed periodic
                    // work budget.
                    if notice.blob.is_some() {
                        sweeps.observe_snapshot(crate::clock::mono_now());
                    }
                    if publication_blob != notice.blob {
                        publication_blob = notice.blob;
                        publication_inputs_changed = true;
                    }
                }
                Ok(NetCommand::ArtifactOffersUpdated(offers)) => {
                    artifact_offers = offers;
                    publication_inputs_changed = true;
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
        if publication_inputs_changed {
            let build = active_provider_cover(
                &artifact_offers,
                &snapshot,
                config.qos.direction.serves(),
                config.team,
            );
            for omitted in &build.omitted {
                warn!(
                    prefix = format_args!("{:#04x}", omitted.prefix),
                    count = omitted.count,
                    "active provider-cover prefix exceeds the bounded publication body"
                );
            }
            offer_scheduler.observe_cover(build.cover, crate::clock::mono_now());
        }

        while let Ok(outcome) = sweep_rx.try_recv() {
            in_flight.remove(&outcome.peer);
            if outcome.success {
                failures.remove(&outcome.peer);
            } else {
                candidates.lock().unwrap().remove(outcome.peer);
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
        while let Ok(outcome) = offer_rx.try_recv() {
            offer_scheduler.complete(outcome, crate::clock::mono_now());
        }

        let now = crate::clock::mono_now();
        if let Some((prefix, deadline)) = offer_scheduler.warnable_expired_lease(now) {
            warn!(
                prefix,
                lag_ms = now.duration_since(deadline).as_millis(),
                "provider-cover backlog missed a shard lease renewal deadline"
            );
        }
        if config.qos.direction.pulls() && sweeps.period_is_due(now) {
            let candidates = candidates.lock().unwrap().sync_candidates();
            sweeps.admit_period(now, &candidates, &in_flight, &failures);
        }

        let local = snapshot.lock().unwrap().as_ref().cloned();
        if config.qos.direction.pulls()
            && let Some(local) = local
        {
            while in_flight.len() < MAX_CONCURRENT_SWEEPS {
                let Some(peer) = sweeps.pop() else {
                    break;
                };
                if in_flight.contains(&peer) {
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
        }

        if config.qos.direction.serves() && snapshot.lock().unwrap().is_some() {
            while offer_scheduler.in_flight_len() < MAX_CONCURRENT_PREFIX_ANNOUNCEMENTS {
                let Some(shard) = offer_scheduler.pop_due(now) else {
                    break;
                };
                let client = publication_client.clone();
                let offer_tx = offer_tx.clone();
                let attempted_at = now;
                let prefix = shard.prefix();
                let digest = shard.digest();
                tokio::spawn(async move {
                    let remote_expected = client.expects_remote();
                    let publication = tokio::time::timeout(
                        PROVIDER_SHARD_PUBLICATION_DEADLINE,
                        client.announce_shard(&shard, remote_expected),
                    )
                    .await
                    .unwrap_or(ProviderPublication {
                        remote_expected,
                        ..ProviderPublication::default()
                    });
                    let _ = offer_tx.send(ProviderShardAnnouncementOutcome {
                        prefix,
                        digest,
                        attempted_at,
                        publication,
                    });
                });
            }
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
    let connection =
        inventory_pool_get(transport, pool, peer, team, connect_proof, sync_proof).await?;
    let remote_key = VerifyingKey::from_bytes(&peer)?;
    candidates.lock().unwrap().promote_authenticated(peer);
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
    team: VerifyingKey,
    connect_proof: &CapabilityProofBundle,
) -> anyhow::Result<AuthenticatedConnection<T::Conn>> {
    let connection = transport.dial(peer, PILE_SYNC_ALPN).await?;
    let verified = op_auth(&connection, connect_proof, team, peer).await?;
    Ok(AuthenticatedConnection {
        connection,
        validity: verified.effective_validity(),
    })
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
    team: VerifyingKey,
    connect_proof: &CapabilityProofBundle,
) -> Option<(Arc<PoolEntry<T::Conn>>, T::Conn)> {
    let entry = pool_entry(pool, peer).await;
    let initialized = entry
        .connection
        .get_or_try_init(|| async {
            tokio::time::timeout(
                DIAL_DEADLINE,
                connect_authed(transport, peer, team, connect_proof),
            )
            .await
            .map_err(|_| anyhow::anyhow!("connection setup deadline exceeded"))?
        })
        .await;
    match initialized {
        Ok(connection) if connection.is_current_at(crate::clock::epoch_now()) => {
            Some((entry.clone(), connection.connection.clone()))
        }
        Ok(_) => {
            debug!(peer = %hex::encode(&peer[..4]), "remote CONNECT capability expired");
            pool_remove_if(pool, peer, &entry).await;
            None
        }
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
    team: VerifyingKey,
    connect_proof: &CapabilityProofBundle,
    sync_proof: &CapabilityProofBundle,
) -> anyhow::Result<T::Conn> {
    let (entry, connection) = pool_get(transport, pool, peer, team, connect_proof)
        .await
        .ok_or_else(|| anyhow::anyhow!("peer is unavailable"))?;
    let authorized = entry
        .inventory_auth
        .get_or_try_init(|| async {
            let verified = tokio::time::timeout(
                OP_DEADLINE,
                op_inventory_auth(&connection, sync_proof, team, peer),
            )
            .await
            .map_err(|_| anyhow::anyhow!("inventory authorization deadline exceeded"))??;
            Ok::<_, anyhow::Error>(RemoteAuthorization {
                validity: verified.effective_validity(),
            })
        })
        .await;
    match authorized {
        Ok(authorization) if authorization.is_current_at(crate::clock::epoch_now()) => {}
        Ok(_) => {
            pool_remove_if(pool, peer, &entry).await;
            anyhow::bail!("remote SYNC_TEAM capability expired");
        }
        Err(error) => {
            pool_remove_if(pool, peer, &entry).await;
            return Err(error);
        }
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
        connection.connection.close(0, b"pool evict");
    }
}

async fn pool_evict<C: Conn>(pool: &SharedPool<C>, peer: PeerId) {
    let entry = pool.lock().await.get(&peer).cloned();
    if let Some(entry) = entry {
        pool_remove_if(pool, peer, &entry).await;
    }
}

async fn hedged_find_map<'a, T, R>(
    attempts: impl IntoIterator<Item = futures::future::BoxFuture<'a, T>>,
    mut accept: impl FnMut(T) -> Option<R>,
) -> Option<R> {
    let mut queued: VecDeque<_> = attempts.into_iter().collect();
    let mut pending = FuturesUnordered::new();
    while pending.len() < ALPHA {
        let Some(attempt) = queued.pop_front() else {
            break;
        };
        pending.push(attempt);
    }
    let hedge = tokio::time::sleep(PROVIDER_FETCH_HEDGE_DELAY);
    tokio::pin!(hedge);
    let mut hedged = false;

    while !pending.is_empty() {
        tokio::select! {
            result = pending.next() => {
                let result = result.expect("pending was checked as non-empty");
                if let Some(found) = accept(result) {
                    return Some(found);
                }
                let active_limit = if hedged { 2 * ALPHA } else { ALPHA };
                while pending.len() < active_limit {
                    let Some(attempt) = queued.pop_front() else {
                        break;
                    };
                    pending.push(attempt);
                }
            }
            () = &mut hedge, if !hedged && !queued.is_empty() => {
                hedged = true;
                while pending.len() < 2 * ALPHA {
                    let Some(attempt) = queued.pop_front() else {
                        break;
                    };
                    pending.push(attempt);
                }
            }
        }
    }
    None
}

async fn fetch_from_providers<T: Transport>(
    transport: &T,
    hash: &RawHash,
    pool: &SharedPool<T::Conn>,
    providers: &[PeerId],
    team: VerifyingKey,
    connect_proof: &CapabilityProofBundle,
    sync_proof: &CapabilityProofBundle,
    candidates: &RoutingCandidates,
) -> Option<Vec<u8>> {
    let attempts = providers.iter().copied().map(|peer| {
        Box::pin(async move {
            let connection = tokio::time::timeout(
                PROVIDER_CONTROL_RPC_DEADLINE,
                inventory_pool_get(transport, pool, peer, team, connect_proof, sync_proof),
            )
            .await;
            let result = match connection {
                Ok(Ok(connection)) => op_get_blob(&connection, hash).await,
                Ok(Err(error)) => Err(error),
                Err(_) => Err(anyhow::anyhow!("provider connection deadline exceeded")),
            };
            match result {
                Ok(Some(bytes)) if blake3::hash(&bytes).as_bytes() == hash => {
                    candidates.lock().unwrap().promote_authenticated(peer);
                    Some(bytes)
                }
                Ok(None) => {
                    candidates.lock().unwrap().promote_authenticated(peer);
                    None
                }
                Ok(Some(_)) | Err(_) => {
                    candidates.lock().unwrap().remove(peer);
                    pool_evict(pool, peer).await;
                    None
                }
            }
        }) as futures::future::BoxFuture<'_, _>
    });
    hedged_find_map(attempts, |result| result).await
}

impl<T: Transport> ProviderClient<T> {
    async fn lookup_replicas(&self, target: RoutingKey) -> Vec<PeerId> {
        let seeds = self.candidates.lock().unwrap().closest(target, K);
        let mut lookup = IterativeLookup::new(self.my_id, target, seeds);
        let mut pending: FuturesUnordered<
            futures::future::BoxFuture<'_, (PeerId, anyhow::Result<Vec<PeerId>>)>,
        > = FuturesUnordered::new();
        let mut active = HashSet::new();
        let completed = tokio::time::timeout(DHT_LOOKUP_DEADLINE, async {
            loop {
                for peer in lookup.next_batch() {
                    active.insert(peer);
                    pending.push(Box::pin(async move {
                        let reply = tokio::time::timeout(
                            PROVIDER_CONTROL_RPC_DEADLINE,
                            self.find_node(peer, target),
                        )
                        .await
                        .map_err(|_| anyhow::anyhow!("FIND_NODE control deadline exceeded"))
                        .and_then(|reply| reply);
                        (peer, reply)
                    }));
                }
                let Some((peer, reply)) = pending.next().await else {
                    break;
                };
                active.remove(&peer);
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
                    Err(error) => {
                        debug!(peer = %hex::encode(&peer[..4]), %error, "FIND_NODE failed");
                        lookup.record_failure(peer, &mut self.candidates.lock().unwrap());
                        pool_evict(&self.pool, peer).await;
                    }
                }
                if lookup.is_finished() && pending.is_empty() {
                    break;
                }
            }
        })
        .await;
        if completed.is_err() {
            debug!("iterative FIND_NODE deadline exceeded");
            drop(pending);
            for peer in active {
                lookup.record_failure(peer, &mut self.candidates.lock().unwrap());
                pool_evict(&self.pool, peer).await;
            }
        }

        let mut replicas = lookup.closest_authenticated_responders().to_vec();
        replicas.push(self.my_id);
        replicas.sort_unstable_by(|a, b| crate::routing::distance_cmp(target, *a, *b));
        replicas.dedup();
        replicas.truncate(K);
        replicas
    }

    async fn find_node(&self, peer: PeerId, target: RoutingKey) -> anyhow::Result<Vec<PeerId>> {
        let connection = inventory_pool_get(
            &self.transport,
            &self.pool,
            peer,
            self.team,
            &self.connect_proof,
            &self.sync_proof,
        )
        .await?;
        tokio::time::timeout(OP_DEADLINE, op_find_node(&connection, &target))
            .await
            .map_err(|_| anyhow::anyhow!("FIND_NODE deadline exceeded"))?
    }

    fn expects_remote(&self) -> bool {
        self.candidates.lock().unwrap().expects_remote()
    }

    async fn announce_shard(
        &self,
        shard: &ProviderShard,
        remote_expected: bool,
    ) -> ProviderPublication {
        let target = provider_prefix_key(self.team, shard.prefix());
        let targets = self.lookup_replicas(target).await;
        self.announce(shard, targets, remote_expected).await
    }

    async fn announce(
        &self,
        shard: &ProviderShard,
        targets: Vec<PeerId>,
        remote_expected: bool,
    ) -> ProviderPublication {
        let mut replies = futures::stream::iter(targets)
            .map(|target| async move {
                (
                    target,
                    tokio::time::timeout(OP_DEADLINE, self.put(target, shard)).await,
                )
            })
            .buffer_unordered(MAX_CONCURRENT_PROVIDER_RPCS);
        let mut publication = ProviderPublication {
            remote_expected,
            ..ProviderPublication::default()
        };
        let _ = tokio::time::timeout(PROVIDER_SHARD_PUBLICATION_DEADLINE, async {
            while let Some((target, reply)) = replies.next().await {
                match reply {
                    Ok(true) if target == self.my_id => publication.local_accepted = true,
                    Ok(true) => publication.remote_accepted += 1,
                    Ok(false) => {}
                    Err(_) => {
                        self.candidates.lock().unwrap().remove(target);
                        pool_evict(&self.pool, target).await;
                    }
                }
            }
        })
        .await;
        publication
    }

    async fn put(&self, target: PeerId, shard: &ProviderShard) -> bool {
        if target == self.my_id {
            let now = crate::clock::mono_now();
            let probe = self.providers.lock().unwrap().probe(
                shard.prefix(),
                shard.digest(),
                shard.count(),
                self.my_id,
                now,
            );
            return match probe {
                ProviderProbe::Known => true,
                ProviderProbe::Full => false,
                ProviderProbe::Need => ProviderShardCandidate::validate(
                    shard.prefix(),
                    shard.digest(),
                    shard.count(),
                    shard.keys().to_vec(),
                )
                .is_ok_and(|candidate| {
                    self.providers
                        .lock()
                        .unwrap()
                        .install(candidate, self.my_id, now)
                }),
            };
        }
        let connection = match inventory_pool_get(
            &self.transport,
            &self.pool,
            target,
            self.team,
            &self.connect_proof,
            &self.sync_proof,
        )
        .await
        {
            Ok(connection) => connection,
            Err(error) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider shard setup failed");
                self.candidates.lock().unwrap().remove(target);
                return false;
            }
        };
        self.candidates
            .lock()
            .unwrap()
            .promote_authenticated(target);
        let probe = match tokio::time::timeout(
            PROVIDER_CONTROL_RPC_DEADLINE,
            op_provider_probe(&connection, shard.prefix(), &shard.digest(), shard.count()),
        )
        .await
        {
            Ok(Ok(probe)) => probe,
            Ok(Err(error)) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider PROBE failed");
                self.candidates.lock().unwrap().remove(target);
                pool_evict(&self.pool, target).await;
                return false;
            }
            Err(_) => {
                self.candidates.lock().unwrap().remove(target);
                pool_evict(&self.pool, target).await;
                return false;
            }
        };
        match probe {
            ProviderProbe::Known => true,
            ProviderProbe::Full => false,
            ProviderProbe::Need => {
                match tokio::time::timeout(
                    OP_DEADLINE,
                    op_provider_body(&connection, shard.prefix(), &shard.digest(), shard.keys()),
                )
                .await
                {
                    Ok(Ok(stored)) => stored,
                    Ok(Err(error)) => {
                        debug!(peer = %hex::encode(&target[..4]), %error, "provider BODY failed");
                        self.candidates.lock().unwrap().remove(target);
                        pool_evict(&self.pool, target).await;
                        false
                    }
                    Err(_) => {
                        self.candidates.lock().unwrap().remove(target);
                        pool_evict(&self.pool, target).await;
                        false
                    }
                }
            }
        }
    }

    async fn find_artifact(&self, artifact: ArtifactId) -> Vec<PeerId> {
        let key = provider_key(self.team, artifact);
        let targets = self
            .lookup_replicas(provider_prefix_key(self.team, key[0]))
            .await;
        self.find(key, artifact, targets).await
    }

    async fn find(
        &self,
        key: ProviderKey,
        artifact: ArtifactId,
        targets: Vec<PeerId>,
    ) -> Vec<PeerId> {
        let mut replies = futures::stream::iter(targets)
            .map(|target| async move {
                (
                    target,
                    tokio::time::timeout(
                        PROVIDER_CONTROL_RPC_DEADLINE,
                        self.get(target, key, artifact),
                    )
                    .await,
                )
            })
            .buffer_unordered(ALPHA);
        let mut replies_by_replica = Vec::new();
        let _ = tokio::time::timeout(PROVIDER_CONTROL_PHASE_DEADLINE, async {
            while let Some((target, reply)) = replies.next().await {
                match reply {
                    Ok(providers) => replies_by_replica.push((target, providers)),
                    Err(_) => {
                        self.candidates.lock().unwrap().remove(target);
                        pool_evict(&self.pool, target).await;
                    }
                }
            }
        })
        .await;
        interleave_provider_replies(replies_by_replica)
    }

    async fn get(&self, target: PeerId, key: ProviderKey, artifact: ArtifactId) -> Vec<PeerId> {
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
            self.team,
            &self.connect_proof,
            &self.sync_proof,
        )
        .await
        {
            Ok(connection) => connection,
            Err(error) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider GET setup failed");
                self.candidates.lock().unwrap().remove(target);
                return Vec::new();
            }
        };
        self.candidates
            .lock()
            .unwrap()
            .promote_authenticated(target);
        match tokio::time::timeout(OP_DEADLINE, op_provider_get(&connection, &artifact)).await {
            Ok(Ok(providers)) => providers,
            Ok(Err(error)) => {
                debug!(peer = %hex::encode(&target[..4]), %error, "provider GET failed");
                self.candidates.lock().unwrap().remove(target);
                pool_evict(&self.pool, target).await;
                Vec::new()
            }
            Err(_) => {
                self.candidates.lock().unwrap().remove(target);
                pool_evict(&self.pool, target).await;
                Vec::new()
            }
        }
    }
}

fn interleave_provider_replies(mut replies: Vec<(PeerId, Vec<PeerId>)>) -> Vec<PeerId> {
    // Directory answers are routing hints, not authority. Replica order is
    // canonicalized before round-robin interleaving so one replica's low
    // endpoint ids cannot permanently crowd every hint from another replica.
    // Exact transfer authenticates every survivor.
    replies.sort_unstable_by_key(|(replica, _)| *replica);
    for (_, providers) in &mut replies {
        providers.retain(|provider| EndpointId::from_bytes(provider).is_ok());
    }

    let mut found = Vec::new();
    let mut seen = BTreeSet::new();
    let rounds = replies
        .iter()
        .map(|(_, providers)| providers.len())
        .max()
        .unwrap_or(0);
    for round in 0..rounds {
        for (_, providers) in &replies {
            let Some(provider) = providers.get(round).copied() else {
                continue;
            };
            if seen.insert(provider) {
                found.push(provider);
                if found.len() == crate::provider::MAX_PROVIDERS_PER_KEY {
                    return found;
                }
            }
        }
    }
    found
}

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: SnapshotSlot,
    snapshots: InventorySnapshots,
    server: InventoryServerConfig,
    local: VerifyingKey,
    connect_proof: CapabilityProofBundle,
    sync_proof: CapabilityProofBundle,
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
                    self.local,
                    &self.connect_proof,
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
            OP_PROVIDER_PROBE => {
                let _session = current_inventory_session(&authorization)?;
                let (prefix, digest, count) = recv_provider_probe(recv).await?;
                let response = self.providers.lock().unwrap().probe(
                    prefix,
                    digest,
                    count,
                    peer.to_bytes(),
                    crate::clock::mono_now(),
                );
                let response = match response {
                    ProviderProbe::Known => PROVIDER_PROBE_KNOWN,
                    ProviderProbe::Need => PROVIDER_PROBE_NEED,
                    ProviderProbe::Full => PROVIDER_PROBE_FULL,
                };
                send_u8(send, response).await?;
            }
            OP_PROVIDER_BODY => {
                let _session = current_inventory_session(&authorization)?;
                // The provider identity is solely the authenticated transport
                // peer. The body contains only team-scoped rendezvous keys.
                let candidate = recv_provider_body(recv).await?;
                let stored = self.providers.lock().unwrap().install(
                    candidate,
                    peer.to_bytes(),
                    crate::clock::mono_now(),
                );
                send_u8(
                    send,
                    if stored {
                        PROVIDER_BODY_OK
                    } else {
                        PROVIDER_BODY_FULL
                    },
                )
                .await?;
            }
            OP_PROVIDER_GET => {
                let session = current_inventory_session(&authorization)?;
                let artifact = recv_provider_artifact(recv).await?;
                let key = provider_key(session.team(), artifact);
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
            OP_FIND_NODE => {
                let _session = current_inventory_session(&authorization)?;
                let target = recv_routing_key(recv).await?;
                let mut peers = self.candidates.lock().unwrap().closest_verified(target, K);
                peers.retain(|candidate| *candidate != peer.to_bytes());
                send_u8(
                    send,
                    u8::try_from(peers.len())
                        .expect("FIND_NODE fan-out is statically bounded below u8::MAX"),
                )
                .await?;
                for peer in peers {
                    send_hash(send, &peer).await?;
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
                let now = crate::clock::epoch_now();
                let caller = self.server.authorize(peer, &proof, now);
                let local = self.server.authorize(self.local, &self.sync_proof, now);
                match (caller, local) {
                    (Ok(session), Ok(_)) => {
                        *authorization.lock().unwrap() =
                            InventoryAuthorization::Authorized(session);
                        self.candidates
                            .lock()
                            .unwrap()
                            .promote_authenticated(peer.to_bytes());
                        if self.admit_inbound_peer {
                            let _ =
                                self.events
                                    .send(NetEventBatch::singleton(NetEvent::Peer(
                                        PeerEvidence::new(self.server.team(), peer),
                                    )))
                                    .await;
                        }
                        send_inventory_auth_ok(send, &self.sync_proof).await?;
                    }
                    (Err(error), _) => {
                        debug!(%error, "caller SYNC_TEAM capability rejected");
                        send_inventory_auth_rejected(send).await?;
                    }
                    (_, Err(error)) => {
                        debug!(%error, "local SYNC_TEAM capability is no longer current");
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
    local: VerifyingKey,
    local_proof: &CapabilityProofBundle,
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
        let now = crate::clock::epoch_now();
        let caller = proof.verify(
            team,
            now,
            peer,
            CapabilityRequest::new(connect_capability_atom(team), CapabilityMode::Invoke),
        )?;
        crate::protocol::verify_endpoint_proof(
            local_proof,
            team,
            local,
            CapabilityRequest::new(connect_capability_atom(team), CapabilityMode::Invoke),
            now,
            "local CONNECT",
        )?;
        Ok(caller)
    }
    .await;
    match verdict {
        Ok(verified) => {
            send_u8(send, AUTH_OK).await?;
            send_capability_proof_bundle(send, local_proof).await?;
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

async fn recv_provider_artifact<R: tokio::io::AsyncRead + Unpin>(
    recv: &mut R,
) -> anyhow::Result<ArtifactId> {
    let artifact = recv_hash(recv).await?;
    require_stream_eof(recv).await?;
    Ok(artifact)
}

async fn recv_provider_probe<R: tokio::io::AsyncRead + Unpin>(
    recv: &mut R,
) -> anyhow::Result<(u8, [u8; 32], u32)> {
    let prefix = recv_u8(recv).await?;
    let digest = recv_hash(recv).await?;
    let count = recv_u32_be(recv).await?;
    require_stream_eof(recv).await?;
    Ok((prefix, digest, count))
}

async fn recv_provider_body<R: tokio::io::AsyncRead + Unpin>(
    recv: &mut R,
) -> anyhow::Result<ProviderShardCandidate> {
    let prefix = recv_u8(recv).await?;
    let digest = recv_hash(recv).await?;
    let count = recv_u32_be(recv).await?;
    let count_usize = usize::try_from(count).expect("u32 fits usize on supported platforms");
    if count_usize == 0 || count_usize > crate::provider::MAX_PROVIDER_SHARD_MEMBERS {
        anyhow::bail!("provider-cover body count is outside the supported bounds");
    }
    let mut keys = Vec::new();
    keys.try_reserve_exact(count_usize)
        .map_err(|error| anyhow::anyhow!("cannot allocate provider-cover body: {error}"))?;
    for _ in 0..count {
        keys.push(recv_hash(recv).await?);
    }
    require_stream_eof(recv).await?;
    ProviderShardCandidate::validate(prefix, digest, count, keys)
}

async fn recv_routing_key<R: tokio::io::AsyncRead + Unpin>(
    recv: &mut R,
) -> anyhow::Result<RoutingKey> {
    let target = recv_hash(recv).await?;
    require_stream_eof(recv).await?;
    Ok(target)
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
        OP_PROVIDER_PROBE => "PROVIDER_PROBE",
        OP_PROVIDER_BODY => "PROVIDER_BODY",
        OP_PROVIDER_GET => "PROVIDER_GET",
        OP_FIND_NODE => "FIND_NODE",
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
    use std::collections::BTreeMap;
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Duration;

    use triblespace_core::capability::{CapabilityClaim, CapabilityMode, CapabilityProofBundle};
    use triblespace_core::collection::{
        CollectionCommit, CollectionData, CollectionRecord, CollectionStore,
    };
    use triblespace_core::inline::Inline;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{ArtifactHandle, ArtifactOfferStore, BlobStorePut};

    use super::*;

    fn key(byte: u8) -> VerifyingKey {
        SigningKey::from_bytes(&[byte; 32]).verifying_key()
    }

    fn artifact(index: u64) -> ArtifactId {
        let mut artifact = [0; 32];
        artifact[24..].copy_from_slice(&index.to_be_bytes());
        artifact
    }

    fn cover_with_prefixes(team: VerifyingKey, prefixes: usize) -> ProviderCover {
        let mut artifacts = Vec::new();
        for index in 0..u64::MAX {
            artifacts.push(artifact(index));
            let cover = ProviderCover::from_artifacts(team, artifacts.iter().copied()).cover;
            if cover.shard_count() == prefixes {
                return cover;
            }
        }
        unreachable!("provider-key derivation did not populate enough prefixes")
    }

    fn endpoint_proof(
        root: &SigningKey,
        endpoint: &SigningKey,
        atom: triblespace_core::capability::CapabilityAtom,
    ) -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            root,
            CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
            endpoint.verifying_key(),
        )
        .unwrap()
    }

    #[test]
    fn startup_rejects_outbound_proofs_for_another_local_endpoint() {
        let team = SigningKey::from_bytes(&[0x71; 32]);
        let endpoint = SigningKey::from_bytes(&[0x72; 32]);
        let other = SigningKey::from_bytes(&[0x73; 32]);
        let config = PeerConfig {
            peers: Vec::new(),
            team: team.verifying_key(),
            connect_proof: endpoint_proof(
                &team,
                &other,
                connect_capability_atom(team.verifying_key()),
            ),
            sync_proof: endpoint_proof(
                &team,
                &endpoint,
                sync_team_capability_atom(team.verifying_key()),
            ),
            qos: ReconcileQos::default(),
        };

        assert!(
            validate_local_authorizations(
                &config,
                endpoint.verifying_key(),
                hifitime::Epoch::from_tai_seconds(0.0),
            )
            .is_err()
        );
    }

    #[test]
    fn routing_evidence_never_replaces_bootstraps() {
        let local = [0; 32];
        let configured = [[1; 32], [2; 32]];
        let mut routes = RoutingTable::new(local, configured);
        for byte in 3..=100 {
            routes.note_candidate([byte; 32]);
        }
        for seed in configured {
            assert!(routes.closest(seed, usize::MAX).contains(&seed));
        }
    }

    fn drain_sweep_period(scheduler: &mut SweepScheduler) -> Vec<PeerId> {
        let mut selected = Vec::new();
        while let Some(peer) = scheduler.pop() {
            selected.push(peer);
        }
        selected
    }

    #[test]
    fn sweep_scheduler_is_bounded_and_fair_across_peer_set_changes() {
        let mut candidates: Vec<_> = (1..=(2 * SWEEPS_PER_PERIOD + 7))
            .map(|index| [u8::try_from(index).unwrap(); 32])
            .collect();
        let now = crate::clock::mono_now();
        let mut scheduler = SweepScheduler::new();
        scheduler.observe_snapshot(now);
        let in_flight = HashSet::new();
        let failures = HashMap::new();
        let mut selected = HashSet::new();

        assert_eq!(
            scheduler.admit_period(now, &candidates, &in_flight, &failures),
            SWEEPS_PER_PERIOD
        );
        assert_eq!(scheduler.pending_len(), SWEEPS_PER_PERIOD);
        selected.extend(drain_sweep_period(&mut scheduler));

        // Remove the exact cursor and insert identities on both sides of its
        // former position. The identity-based insertion-point cursor must not
        // strand either new or surviving evidence.
        let old_cursor = scheduler.cursor.unwrap();
        candidates.retain(|peer| *peer != old_cursor && *peer != [35; 32]);
        candidates.push([0; 32]);
        candidates.push([250; 32]);
        candidates.sort_unstable();

        let mut tick = now + INVENTORY_SWEEP_PERIOD;
        for _ in 0..=candidates.len().div_ceil(SWEEPS_PER_PERIOD) {
            let admitted = scheduler.admit_period(tick, &candidates, &in_flight, &failures);
            assert!(admitted <= SWEEPS_PER_PERIOD);
            assert!(scheduler.pending_len() <= SWEEPS_PER_PERIOD);
            selected.extend(drain_sweep_period(&mut scheduler));
            tick = tick + INVENTORY_SWEEP_PERIOD;
        }

        assert!(
            candidates.iter().all(|peer| selected.contains(peer)),
            "every surviving or inserted peer is eventually selected"
        );
    }

    #[test]
    fn sweep_scheduler_checks_backoff_only_once_per_period() {
        let candidates = vec![[1; 32], [2; 32], [3; 32]];
        let now = crate::clock::mono_now();
        let retry_at = now + INVENTORY_SWEEP_PERIOD + INVENTORY_SWEEP_PERIOD;
        let failures = HashMap::from([([2; 32], (1, retry_at))]);
        let in_flight = HashSet::new();
        let mut scheduler = SweepScheduler::new();
        assert!(!scheduler.period_is_due(now));
        scheduler.observe_snapshot(now);
        assert!(scheduler.period_is_due(now));

        assert_eq!(
            scheduler.admit_period(now, &candidates, &in_flight, &failures),
            2
        );
        assert_eq!(drain_sweep_period(&mut scheduler), vec![[1; 32], [3; 32]]);
        assert!(!scheduler.period_is_due(now + Duration::from_millis(10)));
        assert_eq!(
            scheduler.admit_period(
                now + Duration::from_millis(10),
                &candidates,
                &in_flight,
                &failures,
            ),
            0,
            "an empty queue does not trigger a host-poll-rate rescan"
        );

        let next = now + INVENTORY_SWEEP_PERIOD;
        assert!(scheduler.period_is_due(next));
        scheduler.admit_period(next, &candidates, &in_flight, &failures);
        assert!(!drain_sweep_period(&mut scheduler).contains(&[2; 32]));
        scheduler.admit_period(
            next + INVENTORY_SWEEP_PERIOD,
            &candidates,
            &in_flight,
            &failures,
        );
        assert!(drain_sweep_period(&mut scheduler).contains(&[2; 32]));
    }

    #[test]
    fn repeated_snapshots_do_not_reset_or_amplify_the_period_budget() {
        let candidates: Vec<_> = (1..=SWEEPS_PER_PERIOD + 1)
            .map(|index| [u8::try_from(index).unwrap(); 32])
            .collect();
        let now = crate::clock::mono_now();
        let mut scheduler = SweepScheduler::new();
        scheduler.observe_snapshot(now);
        scheduler.observe_snapshot(now + Duration::from_secs(10));

        assert_eq!(
            scheduler.admit_period(now, &candidates, &HashSet::new(), &HashMap::new()),
            SWEEPS_PER_PERIOD,
            "the first snapshot starts immediately"
        );
        drain_sweep_period(&mut scheduler);

        for millis in [10, 20, 1_000, 29_999] {
            scheduler.observe_snapshot(now + Duration::from_millis(millis));
            assert_eq!(
                scheduler.admit_period(
                    now + Duration::from_millis(millis),
                    &candidates,
                    &HashSet::new(),
                    &HashMap::new(),
                ),
                0
            );
        }
        assert_eq!(
            scheduler.admit_period(
                now + INVENTORY_SWEEP_PERIOD,
                &candidates,
                &HashSet::new(),
                &HashMap::new(),
            ),
            SWEEPS_PER_PERIOD
        );
    }

    #[test]
    fn provider_cover_scheduler_is_bounded_and_fair_beyond_concurrency() {
        let now = crate::clock::mono_now();
        let count = 3 * MAX_CONCURRENT_PREFIX_ANNOUNCEMENTS + 2;
        let cover = cover_with_prefixes(key(9), count);
        let mut scheduler = ProviderCoverScheduler::new();
        scheduler.observe_cover(cover, now);
        assert_eq!(scheduler.active_len(), count);

        let mut seen = BTreeSet::new();
        while seen.len() < count {
            let mut batch = Vec::new();
            while scheduler.in_flight_len() < MAX_CONCURRENT_PREFIX_ANNOUNCEMENTS {
                let Some(shard) = scheduler.pop_due(now) else {
                    break;
                };
                assert!(
                    seen.insert(shard.prefix()),
                    "an active prefix launches once per due time"
                );
                batch.push(shard);
            }
            assert!(!batch.is_empty());
            assert!(batch.len() <= MAX_CONCURRENT_PREFIX_ANNOUNCEMENTS);
            for shard in batch {
                scheduler.complete(
                    ProviderShardAnnouncementOutcome {
                        prefix: shard.prefix(),
                        digest: shard.digest(),
                        attempted_at: now,
                        publication: ProviderPublication {
                            remote_expected: true,
                            remote_accepted: 1,
                            ..ProviderPublication::default()
                        },
                    },
                    now,
                );
            }
        }
        assert_eq!(seen.len(), count);
    }

    #[test]
    fn provider_cover_scheduler_renews_at_half_life_and_retries_with_backoff() {
        let now = crate::clock::mono_now();
        let cover = cover_with_prefixes(key(10), 2);
        let prefixes: Vec<_> = cover.iter().map(|(&prefix, _)| prefix).collect();
        let success = prefixes[0];
        let retry = prefixes[1];
        let mut scheduler = ProviderCoverScheduler::new();
        scheduler.observe_cover(cover, now);

        let success_shard = scheduler.pop_due(now).unwrap();
        assert_eq!(success_shard.prefix(), success);
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix: success,
                digest: success_shard.digest(),
                attempted_at: now,
                publication: ProviderPublication {
                    local_accepted: true,
                    ..ProviderPublication::default()
                },
            },
            now,
        );
        assert_eq!(
            scheduler.next_due(success),
            Some(now + PROVIDER_RENEWAL_INTERVAL)
        );

        let retry_shard = scheduler.pop_due(now).unwrap();
        assert_eq!(retry_shard.prefix(), retry);
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix: retry,
                digest: retry_shard.digest(),
                attempted_at: now,
                publication: ProviderPublication {
                    local_accepted: true,
                    remote_expected: true,
                    remote_accepted: 0,
                },
            },
            now,
        );
        assert_eq!(
            scheduler.next_due(retry),
            Some(now + crate::RETRY_BACKOFF_BASE)
        );

        let retry_at = now + crate::RETRY_BACKOFF_BASE;
        let retry_shard = scheduler.pop_due(retry_at).unwrap();
        assert_eq!(retry_shard.prefix(), retry);
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix: retry,
                digest: retry_shard.digest(),
                attempted_at: retry_at,
                publication: ProviderPublication {
                    local_accepted: true,
                    remote_expected: true,
                    remote_accepted: 0,
                },
            },
            retry_at,
        );
        assert_eq!(
            scheduler.next_due(retry),
            Some(retry_at + (crate::RETRY_BACKOFF_BASE * 2))
        );
        assert_eq!(
            scheduler.lease_deadline(success),
            Some(now + PROVIDER_LEASE_LIFETIME)
        );
        assert_eq!(scheduler.lease_deadline(retry), None);
    }

    #[test]
    fn provider_cover_scheduler_preserves_deadlines_across_equivalent_snapshots() {
        let now = crate::clock::mono_now();
        let cover = cover_with_prefixes(key(11), 1);
        let prefix = *cover.iter().next().unwrap().0;
        let mut scheduler = ProviderCoverScheduler::new();
        scheduler.observe_cover(cover.clone(), now);
        let shard = scheduler.pop_due(now).unwrap();
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix,
                digest: shard.digest(),
                attempted_at: now,
                publication: ProviderPublication {
                    local_accepted: true,
                    ..ProviderPublication::default()
                },
            },
            now,
        );
        let renewal = scheduler.next_due(prefix);
        scheduler.observe_cover(cover, now + Duration::from_secs(1));
        assert_eq!(scheduler.next_due(prefix), renewal);

        scheduler.observe_cover(ProviderCover::default(), now + Duration::from_secs(2));
        assert_eq!(scheduler.active_len(), 0);
        assert_eq!(scheduler.next_due(prefix), None);
        assert_eq!(scheduler.lease_deadline(prefix), None);
    }

    #[test]
    fn changed_prefix_retains_old_deadline_and_collapses_history_by_current_root() {
        let now = crate::clock::mono_now();
        let team = key(12);
        let original = cover_with_prefixes(team, 1);
        let prefix = *original.iter().next().unwrap().0;
        let original_shard = original.get(prefix).unwrap().clone();
        let mut artifacts = vec![artifact(0)];
        let mut candidate_index = 1_u64;
        let changed = loop {
            artifacts.push(artifact(candidate_index));
            candidate_index += 1;
            let candidate = ProviderCover::from_artifacts(team, artifacts.iter().copied()).cover;
            if candidate.shard_count() == 1
                && candidate
                    .get(prefix)
                    .is_some_and(|shard| shard.digest() != original_shard.digest())
            {
                break candidate;
            }
            if candidate.shard_count() > 1 {
                artifacts.pop();
            }
        };
        let mut scheduler = ProviderCoverScheduler::new();
        scheduler.observe_cover(original.clone(), now);
        let first = scheduler.pop_due(now).unwrap();
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix,
                digest: first.digest(),
                attempted_at: now,
                publication: ProviderPublication {
                    local_accepted: true,
                    ..ProviderPublication::default()
                },
            },
            now,
        );
        let old_deadline = now + PROVIDER_LEASE_LIFETIME;
        let renewal = now + PROVIDER_RENEWAL_INTERVAL;
        let renewing = scheduler.pop_due(renewal).unwrap();
        scheduler.observe_cover(changed, renewal);
        assert_eq!(
            scheduler.lease_deadline(prefix),
            Some(old_deadline),
            "a failed changed-root publication can still miss the old lease"
        );
        scheduler.observe_cover(original, renewal);
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix,
                digest: renewing.digest(),
                attempted_at: renewal,
                publication: ProviderPublication {
                    local_accepted: true,
                    ..ProviderPublication::default()
                },
            },
            renewal,
        );
        assert_eq!(
            scheduler.next_due(prefix),
            Some(renewal + PROVIDER_RENEWAL_INTERVAL),
            "A→B→A while A is in flight already realizes the current state"
        );
        assert_eq!(
            scheduler.lease_deadline(prefix),
            Some(renewal + PROVIDER_LEASE_LIFETIME)
        );
    }

    #[test]
    fn oversized_prefix_expires_while_neighboring_exact_prefix_stays_scheduled() {
        let now = crate::clock::mono_now();
        let team = key(13);
        let mut by_prefix = BTreeMap::<u8, Vec<ArtifactId>>::new();
        let (oversized_prefix, pair) = (0..u64::MAX)
            .find_map(|index| {
                let artifact = artifact(index);
                let prefix = provider_key(team, artifact)[0];
                let artifacts = by_prefix.entry(prefix).or_default();
                artifacts.push(artifact);
                (artifacts.len() == 2).then(|| (prefix, artifacts.clone()))
            })
            .expect("two provider keys eventually share a prefix");
        let neighbor = (0..u64::MAX)
            .map(artifact)
            .find(|artifact| provider_key(team, *artifact)[0] != oversized_prefix)
            .expect("a provider key eventually lands in another prefix");
        let healthy_prefix = provider_key(team, neighbor)[0];
        let artifacts = [pair[0], pair[1], neighbor];

        let initial = ProviderCover::from_artifacts_with_shard_limit(team, artifacts, 2);
        assert!(initial.omitted.is_empty());
        let mut scheduler = ProviderCoverScheduler::new();
        scheduler.observe_cover(initial.cover, now);
        while let Some(shard) = scheduler.pop_due(now) {
            scheduler.complete(
                ProviderShardAnnouncementOutcome {
                    prefix: shard.prefix(),
                    digest: shard.digest(),
                    attempted_at: now,
                    publication: ProviderPublication {
                        local_accepted: true,
                        ..ProviderPublication::default()
                    },
                },
                now,
            );
        }
        let healthy_due = scheduler.next_due(healthy_prefix);
        let healthy_deadline = scheduler.lease_deadline(healthy_prefix);
        assert!(scheduler.lease_deadline(oversized_prefix).is_some());

        let partial = ProviderCover::from_artifacts_with_shard_limit(team, artifacts, 1);
        assert_eq!(
            partial.omitted,
            vec![crate::provider::OmittedProviderPrefix {
                prefix: oversized_prefix,
                count: 2,
            }]
        );
        assert!(partial.cover.get(oversized_prefix).is_none());
        assert!(partial.cover.get(healthy_prefix).is_some());
        scheduler.observe_cover(partial.cover, now + Duration::from_secs(1));

        assert_eq!(scheduler.active_len(), 1);
        assert_eq!(scheduler.next_due(oversized_prefix), None);
        assert_eq!(scheduler.lease_deadline(oversized_prefix), None);
        assert_eq!(scheduler.next_due(healthy_prefix), healthy_due);
        assert_eq!(scheduler.lease_deadline(healthy_prefix), healthy_deadline);
    }

    #[test]
    fn provider_cover_scheduler_rate_limits_definite_lease_miss_warnings() {
        let now = crate::clock::mono_now();
        let cover = cover_with_prefixes(key(13), 1);
        let prefix = *cover.iter().next().unwrap().0;
        let mut scheduler = ProviderCoverScheduler::new();
        scheduler.observe_cover(cover, now);
        let shard = scheduler.pop_due(now).unwrap();
        scheduler.complete(
            ProviderShardAnnouncementOutcome {
                prefix,
                digest: shard.digest(),
                attempted_at: now,
                publication: ProviderPublication {
                    local_accepted: true,
                    ..ProviderPublication::default()
                },
            },
            now,
        );

        let deadline = now + PROVIDER_LEASE_LIFETIME;
        assert_eq!(
            scheduler.warnable_expired_lease(deadline),
            Some((prefix, deadline))
        );
        assert_eq!(scheduler.warnable_expired_lease(deadline), None);
        assert_eq!(
            scheduler.warnable_expired_lease(deadline + OFFER_BACKLOG_WARNING_INTERVAL),
            Some((prefix, deadline))
        );
    }

    #[test]
    fn active_offers_are_exactly_policy_intersect_resident_snapshot() {
        let team = key(1);
        let mut store = MemoryRepo::default();
        let resident = store
            .put::<UnknownBlob, _>(Bytes::from_source(vec![7; 257]))
            .unwrap()
            .raw;
        let absent = [9; 32];
        store
            .offer_all([ArtifactHandle::new(resident), ArtifactHandle::new(absent)])
            .unwrap();
        let offers = store.offers_snapshot().unwrap();
        let snapshot = Arc::new(StoreSnapshot::from_store(&mut store, team).unwrap());
        let slot = Arc::new(Mutex::new(Some(snapshot)));

        let active = active_provider_cover(&offers, &slot, true, team).cover;
        let expected = ProviderCover::from_artifacts(team, [resident]).cover;
        assert!(active.same_membership(&expected));
        assert!(
            active_provider_cover(&offers, &slot, false, team)
                .cover
                .same_membership(&ProviderCover::default())
        );
        slot.lock().unwrap().take();
        assert!(
            active_provider_cover(&offers, &slot, true, team)
                .cover
                .same_membership(&ProviderCover::default())
        );
    }

    #[test]
    fn aggregate_provider_replies_are_deduplicated_and_bounded() {
        let mut providers: Vec<_> = (1..=100)
            .map(|byte| {
                SigningKey::from_bytes(&[byte; 32])
                    .verifying_key()
                    .to_bytes()
            })
            .collect();
        providers.extend_from_within(..32);

        let normalized = interleave_provider_replies(vec![([1; 32], providers)]);
        assert_eq!(normalized.len(), crate::provider::MAX_PROVIDERS_PER_KEY);
        assert_eq!(
            normalized.iter().copied().collect::<BTreeSet<_>>().len(),
            normalized.len()
        );
    }

    #[test]
    fn aggregate_provider_replies_do_not_let_one_replica_crowd_out_another() {
        let crowded: Vec<_> = (1..=64)
            .map(|byte| {
                SigningKey::from_bytes(&[byte; 32])
                    .verifying_key()
                    .to_bytes()
            })
            .collect();
        let only_healthy = SigningKey::from_bytes(&[100; 32])
            .verifying_key()
            .to_bytes();

        let normalized =
            interleave_provider_replies(vec![([1; 32], crowded), ([2; 32], vec![only_healthy])]);
        assert_eq!(normalized.len(), crate::provider::MAX_PROVIDERS_PER_KEY);
        assert!(normalized.contains(&only_healthy));
    }

    #[test]
    fn aggregate_provider_replies_preserve_replica_rotation_across_calls() {
        let left: Vec<_> = (1..=64)
            .map(|byte| {
                SigningKey::from_bytes(&[byte; 32])
                    .verifying_key()
                    .to_bytes()
            })
            .collect();
        let right: Vec<_> = (65..=128)
            .map(|byte| {
                SigningKey::from_bytes(&[byte; 32])
                    .verifying_key()
                    .to_bytes()
            })
            .collect();
        let mut seen = BTreeSet::new();
        for rotation in 0..64 {
            let mut rotated_left = left.clone();
            let mut rotated_right = right.clone();
            rotated_left.rotate_left(rotation);
            rotated_right.rotate_left(rotation);
            seen.extend(interleave_provider_replies(vec![
                ([1; 32], rotated_left),
                ([2; 32], rotated_right),
            ]));
        }

        assert_eq!(seen.len(), 128);
    }

    #[tokio::test(start_paused = true)]
    async fn exact_fetch_hedges_once_beyond_alpha_without_cancelling_slow_bodies() {
        let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut attempts: VecDeque<futures::future::BoxFuture<'_, Option<u8>>> = VecDeque::new();
        for _ in 0..ALPHA {
            let started = started.clone();
            attempts.push_back(Box::pin(async move {
                started.fetch_add(1, AtomicOrdering::SeqCst);
                futures::future::pending().await
            }));
        }
        let healthy_started = started.clone();
        attempts.push_back(Box::pin(async move {
            healthy_started.fetch_add(1, AtomicOrdering::SeqCst);
            Some(7)
        }));

        let found = hedged_find_map(attempts, |result| result).await;
        assert_eq!(found, Some(7));
        assert_eq!(started.load(AtomicOrdering::SeqCst), ALPHA + 1);
    }

    #[tokio::test(start_paused = true)]
    async fn exact_fetch_hedging_never_exceeds_two_alpha_transfers() {
        let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let attempts = (0..crate::provider::MAX_PROVIDERS_PER_KEY).map(|_| {
            let started = started.clone();
            Box::pin(async move {
                started.fetch_add(1, AtomicOrdering::SeqCst);
                futures::future::pending::<Option<()>>().await
            }) as futures::future::BoxFuture<'_, Option<()>>
        });

        let deadline = PROVIDER_FETCH_HEDGE_DELAY + std::time::Duration::from_millis(1);
        assert!(
            tokio::time::timeout(deadline, hedged_find_map(attempts, |result| result))
                .await
                .is_err()
        );
        assert_eq!(started.load(AtomicOrdering::SeqCst), 2 * ALPHA);
    }

    #[tokio::test]
    async fn provider_request_has_no_forgeable_provider_field() {
        let artifact = [7; 32];
        let (mut send, mut recv) = tokio::io::duplex(64);
        send.write_all(&artifact).await.unwrap();
        send.shutdown().await.unwrap();
        assert_eq!(recv_provider_artifact(&mut recv).await.unwrap(), artifact);

        let (mut send, mut recv) = tokio::io::duplex(64);
        send.write_all(&artifact).await.unwrap();
        send.write_all(&[8; 32]).await.unwrap();
        send.shutdown().await.unwrap();
        assert!(
            recv_provider_artifact(&mut recv).await.is_err(),
            "an appended claimed provider identity must invalidate the request"
        );
    }

    #[tokio::test]
    async fn provider_body_rejects_corruption_truncation_and_trailing_bytes() {
        let mut first = [0; 32];
        first[0] = 7;
        first[31] = 1;
        let mut second = first;
        second[31] = 2;
        let keys = [first, second];
        let digest =
            PATCH::<32, IdentitySchema, (), triblespace_core::patch::Blake3Merkle>::from_keys(keys)
                .merkle_root()
                .unwrap();
        let mut body = Vec::new();
        body.push(7);
        body.extend_from_slice(&digest);
        body.extend_from_slice(&2_u32.to_be_bytes());
        for key in keys {
            body.extend_from_slice(&key);
        }
        assert!(recv_provider_body(&mut body.as_slice()).await.is_ok());

        let mut corrupt = body.clone();
        corrupt[1] ^= 1;
        assert!(recv_provider_body(&mut corrupt.as_slice()).await.is_err());

        let mut truncated = body.clone();
        truncated.pop();
        assert!(recv_provider_body(&mut truncated.as_slice()).await.is_err());

        let mut trailing = body;
        trailing.push(0);
        assert!(recv_provider_body(&mut trailing.as_slice()).await.is_err());
    }

    #[tokio::test]
    async fn find_node_request_is_exactly_one_routing_key() {
        let target = [9; 32];
        let (mut send, mut recv) = tokio::io::duplex(64);
        send.write_all(&target).await.unwrap();
        send.shutdown().await.unwrap();
        assert_eq!(recv_routing_key(&mut recv).await.unwrap(), target);

        let (mut send, mut recv) = tokio::io::duplex(64);
        send.write_all(&target).await.unwrap();
        send.write_all(&[1]).await.unwrap();
        send.shutdown().await.unwrap();
        assert!(recv_routing_key(&mut recv).await.is_err());
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
