//! Proof-gated custody replication for complete semantic stores.
//!
//! A custody replica converges the product join-semilattice
//!
//! ```text
//! BlobStore x CollectionStore x CapabilityProofStore
//! ```
//!
//! by componentwise set union. It deliberately does not copy WANTs, legacy
//! pins, append timestamps, padding, routing state, or opaque future records.
//! Public collection gossip remains sparse and reach-governed; this module is
//! a separate, explicitly authorized full-residency path between static peers.

use std::array;
use std::collections::{BTreeMap, BTreeSet};

use triblespace_core::capability::{
    CapabilityAtom, CapabilityMode, CapabilityProof, CapabilityProofBundle, CapabilityRequest,
    CapabilityResource,
};
use triblespace_core::collection::{CollectionRecord, CollectionStore};
use triblespace_core::id::{Id, id_hex};
use triblespace_core::repo::{
    BlobInfo, BlobStore, BlobStoreGet, BlobStoreList, CapabilityProofStore, StorageFlush,
};

/// Permission to union one complete resident semantic store into another.
///
/// Minted on 2026-08-26 CEST with `trible genid`, which returned
/// `D8453B974E15F5DF17B1B67A338B3EBD`.
///
/// CONNECT remains an independent transport-admission action. A CONNECT proof
/// never satisfies this action and every custody operation presents its own
/// exact proof bundle.
pub const ACTION_REPLICATE_STORE: Id = id_hex!("D8453B974E15F5DF17B1B67A338B3EBD");

/// Exact 256-bit identity of one custody-replica set.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReplicaSetId([u8; 32]);

impl ReplicaSetId {
    /// Construct an exact replica-set identity.
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the exact resource bytes used by capability verification.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl From<[u8; 32]> for ReplicaSetId {
    fn from(bytes: [u8; 32]) -> Self {
        Self::new(bytes)
    }
}

/// Exact capability atom required for one custody-replica operation.
pub fn replicate_capability_atom(replica_set: ReplicaSetId) -> CapabilityAtom {
    CapabilityAtom::new(
        ACTION_REPLICATE_STORE.into(),
        CapabilityResource::new(replica_set.into_bytes()),
    )
}

/// Server-side authority for custody operations.
///
/// This value deliberately carries no presenter proof. The server chooses the
/// external root and exact resource; each caller supplies its proof anew on
/// every operation.
#[derive(Clone, Copy, Debug)]
pub struct ReplicaServerConfig {
    /// External trust root against which operation proofs are checked.
    pub trust_root: ed25519_dalek::VerifyingKey,
    /// Exact custody set this endpoint serves.
    pub replica_set: ReplicaSetId,
}

/// Complete client and server configuration for a private custody node.
#[derive(Clone)]
pub struct CustodyReplicaConfig {
    /// Static peers. Custody never expands this set from gossip or the DHT.
    pub peers: Vec<iroh_base::EndpointAddr>,
    /// External trust root for the ordinary CONNECT handshake.
    pub connect_root: ed25519_dalek::VerifyingKey,
    /// Exact CONNECT proof presented when dialing a configured peer.
    pub connect_proof: CapabilityProofBundle,
    /// External trust root for the independent REPLICATE action.
    pub replica_root: ed25519_dalek::VerifyingKey,
    /// Exact custody set being converged.
    pub replica_set: ReplicaSetId,
    /// Invoke-only proof presented afresh on every custody operation.
    pub replica_proof: CapabilityProofBundle,
    /// Restart-stable socket bound only on the private fabric (for example a
    /// ZeroTier `10.242.x.y` address).
    pub bind_addr: std::net::SocketAddr,
    /// Directory for incomplete large-blob receive files. Production callers
    /// should place this beside the destination pile, not in the OS temp dir.
    pub receive_temp_dir: std::path::PathBuf,
}

/// One component of the semantic custody product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ReplicaComponent {
    Blobs = 1,
    CollectionRecords = 2,
    CapabilityProofs = 3,
}

impl ReplicaComponent {
    pub(crate) const ALL: [Self; 3] =
        [Self::Blobs, Self::CollectionRecords, Self::CapabilityProofs];

    pub(crate) fn from_byte(byte: u8) -> anyhow::Result<Self> {
        match byte {
            1 => Ok(Self::Blobs),
            2 => Ok(Self::CollectionRecords),
            3 => Ok(Self::CapabilityProofs),
            _ => anyhow::bail!("unknown custody-replica component {byte:#x}"),
        }
    }

    const fn index(self) -> usize {
        self as usize - 1
    }
}

/// Canonical 32-byte page key. Collection-record IDs occupy the first sixteen
/// bytes and leave the suffix zero; blob and proof IDs use all 32 bytes.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct ReplicaItemId(pub(crate) [u8; 32]);

impl ReplicaItemId {
    pub(crate) fn collection(id: Id) -> Self {
        let mut bytes = [0; 32];
        bytes[..16].copy_from_slice(&id.raw());
        Self(bytes)
    }

    pub(crate) const fn prefix(self) -> u8 {
        self.0[0]
    }
}

/// One item returned by a bounded inventory page.
#[derive(Clone, Debug)]
pub(crate) enum ReplicaItem {
    Blob(BlobInfo),
    CollectionRecord(CollectionRecord),
    CapabilityProof(CapabilityProof),
}

impl ReplicaItem {
    pub(crate) fn id(&self) -> ReplicaItemId {
        match self {
            Self::Blob(info) => ReplicaItemId(info.handle.raw),
            Self::CollectionRecord(record) => ReplicaItemId::collection(record.id()),
            Self::CapabilityProof(proof) => ReplicaItemId(proof.id().raw),
        }
    }

    pub(crate) fn encoded_len(&self) -> u64 {
        match self {
            Self::Blob(info) => info.length,
            Self::CollectionRecord(record) => record.to_bytes().len() as u64,
            Self::CapabilityProof(proof) => proof.as_bytes().len() as u64,
        }
    }
}

/// Digest of one stable first-byte inventory bucket.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplicaBucketSummary {
    pub(crate) count: u64,
    pub(crate) bytes: u64,
    pub(crate) digest: [u8; 32],
}

/// Fixed-size summary of all three custody components.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReplicaSummary {
    buckets: [[ReplicaBucketSummary; 256]; 3],
}

impl ReplicaSummary {
    pub(crate) fn bucket(&self, component: ReplicaComponent, prefix: u8) -> ReplicaBucketSummary {
        self.buckets[component.index()][prefix as usize]
    }

    pub(crate) fn buckets(&self, component: ReplicaComponent) -> &[ReplicaBucketSummary; 256] {
        &self.buckets[component.index()]
    }

    pub(crate) fn from_buckets(buckets: [[ReplicaBucketSummary; 256]; 3]) -> Self {
        Self { buckets }
    }
}

const SUMMARY_DOMAIN: &[u8] = b"triblespace.custody-replica-bucket\0";
const SUMMARY_VERSION: u32 = 1;
/// Maximum amount of newly admitted blob payload left between durability
/// barriers. One oversized blob is flushed immediately after admission.
const BLOB_DURABILITY_BATCH_BYTES: u64 = 64 * 1024 * 1024;

/// Immutable, sorted serving inventory for one observed store prefix.
pub(crate) struct ReplicaSnapshotData {
    blobs: [Vec<BlobInfo>; 256],
    records: [Vec<CollectionRecord>; 256],
    proofs: [Vec<CapabilityProof>; 256],
    summary: ReplicaSummary,
}

impl ReplicaSnapshotData {
    /// Build an exact inventory from already-observed component snapshots.
    ///
    /// Backend enumeration order is not trusted. Duplicate identities collapse;
    /// conflicting bytes under one identity fail instead of becoming an
    /// order-dependent page.
    pub(crate) fn new<R>(
        reader: &R,
        records: impl IntoIterator<Item = CollectionRecord>,
        proofs: impl IntoIterator<Item = CapabilityProof>,
    ) -> anyhow::Result<Self>
    where
        R: BlobStoreList + BlobStoreGet,
    {
        let mut blob_map = BTreeMap::<[u8; 32], BlobInfo>::new();
        for info in reader.blobs() {
            let info = info?;
            // Listing metadata is explicitly unvalidated. Custody inventory is
            // a claim of resident semantic bytes, so only values accepted by
            // BlobStoreGet under the recorded content handle participate. A
            // corrupt local occurrence is omitted and can therefore be healed
            // by a good peer's later append.
            use triblespace_core::blob::encodings::UnknownBlob;
            let Ok(bytes) = reader.get::<anybytes::Bytes, UnknownBlob>(info.handle) else {
                continue;
            };
            if bytes.len() as u64 != info.length {
                continue;
            }
            match blob_map.entry(info.handle.raw) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(info);
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get().length == info.length => {}
                std::collections::btree_map::Entry::Occupied(entry) => {
                    anyhow::bail!(
                        "blob {} has conflicting resident lengths {} and {}",
                        hex::encode(info.handle.raw),
                        entry.get().length,
                        info.length
                    );
                }
            }
        }

        let mut record_map = BTreeMap::<Id, CollectionRecord>::new();
        for record in records {
            match record_map.entry(record.id()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(record);
                }
                std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &record => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    anyhow::bail!(
                        "collection record id {} names conflicting canonical bytes",
                        hex::encode(record.id().raw())
                    );
                }
            }
        }

        let mut proof_map = BTreeMap::<[u8; 32], CapabilityProof>::new();
        for proof in proofs {
            let id = proof.id().raw;
            match proof_map.entry(id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(proof);
                }
                std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &proof => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    anyhow::bail!(
                        "capability proof id {} names conflicting canonical bytes",
                        hex::encode(id)
                    );
                }
            }
        }

        let mut blobs: [Vec<BlobInfo>; 256] = array::from_fn(|_| Vec::new());
        for (_, info) in blob_map {
            blobs[info.handle.raw[0] as usize].push(info);
        }
        let mut record_buckets: [Vec<CollectionRecord>; 256] = array::from_fn(|_| Vec::new());
        for (_, record) in record_map {
            record_buckets[record.id().raw()[0] as usize].push(record);
        }
        let mut proof_buckets: [Vec<CapabilityProof>; 256] = array::from_fn(|_| Vec::new());
        for (_, proof) in proof_map {
            proof_buckets[proof.id().raw[0] as usize].push(proof);
        }

        let mut summary = [[ReplicaBucketSummary::default(); 256]; 3];
        for prefix in 0..=u8::MAX {
            summary[ReplicaComponent::Blobs.index()][prefix as usize] = summarize(
                ReplicaComponent::Blobs,
                prefix,
                blobs[prefix as usize]
                    .iter()
                    .copied()
                    .map(ReplicaItem::Blob),
            )?;
            summary[ReplicaComponent::CollectionRecords.index()][prefix as usize] = summarize(
                ReplicaComponent::CollectionRecords,
                prefix,
                record_buckets[prefix as usize]
                    .iter()
                    .copied()
                    .map(ReplicaItem::CollectionRecord),
            )?;
            summary[ReplicaComponent::CapabilityProofs.index()][prefix as usize] = summarize(
                ReplicaComponent::CapabilityProofs,
                prefix,
                proof_buckets[prefix as usize]
                    .iter()
                    .cloned()
                    .map(ReplicaItem::CapabilityProof),
            )?;
        }

        Ok(Self {
            blobs,
            records: record_buckets,
            proofs: proof_buckets,
            summary: ReplicaSummary::from_buckets(summary),
        })
    }

    pub(crate) fn summary(&self) -> &ReplicaSummary {
        &self.summary
    }

    /// Return one strictly bounded, exclusive-cursor page.
    pub(crate) fn page(
        &self,
        component: ReplicaComponent,
        prefix: u8,
        after: Option<ReplicaItemId>,
    ) -> (Vec<ReplicaItem>, bool) {
        let limit = component.page_limit();
        match component {
            ReplicaComponent::Blobs => page_slice(
                &self.blobs[prefix as usize],
                after,
                limit,
                |info| ReplicaItemId(info.handle.raw),
                |info| ReplicaItem::Blob(*info),
            ),
            ReplicaComponent::CollectionRecords => page_slice(
                &self.records[prefix as usize],
                after,
                limit,
                |record| ReplicaItemId::collection(record.id()),
                |record| ReplicaItem::CollectionRecord(*record),
            ),
            ReplicaComponent::CapabilityProofs => page_slice(
                &self.proofs[prefix as usize],
                after,
                limit,
                |proof| ReplicaItemId(proof.id().raw),
                |proof| ReplicaItem::CapabilityProof(proof.clone()),
            ),
        }
    }

    pub(crate) fn blob_bytes<R>(reader: &R, id: ReplicaItemId) -> Option<anybytes::Bytes>
    where
        R: triblespace_core::repo::BlobStoreGet,
    {
        use triblespace_core::blob::encodings::UnknownBlob;
        use triblespace_core::inline::Inline;
        use triblespace_core::inline::encodings::hash::Handle;
        reader
            .get::<anybytes::Bytes, UnknownBlob>(Inline::<Handle<UnknownBlob>>::new(id.0))
            .ok()
    }
}

fn page_slice<T>(
    bucket: &[T],
    after: Option<ReplicaItemId>,
    limit: usize,
    id: impl Fn(&T) -> ReplicaItemId,
    item: impl Fn(&T) -> ReplicaItem,
) -> (Vec<ReplicaItem>, bool) {
    let start = after
        .map(|cursor| bucket.partition_point(|value| id(value) <= cursor))
        .unwrap_or(0);
    let end = start.saturating_add(limit).min(bucket.len());
    (
        bucket[start..end].iter().map(item).collect(),
        end == bucket.len(),
    )
}

impl ReplicaComponent {
    pub(crate) const fn page_limit(self) -> usize {
        match self {
            Self::Blobs => 512,
            Self::CollectionRecords => 256,
            // A proof can be ~32 KiB at the 255-edge carrier limit. Sixteen
            // keeps one response below roughly 512 KiB.
            Self::CapabilityProofs => 16,
        }
    }
}

fn summarize(
    component: ReplicaComponent,
    prefix: u8,
    items: impl IntoIterator<Item = ReplicaItem>,
) -> anyhow::Result<ReplicaBucketSummary> {
    let mut accumulator = ReplicaBucketAccumulator::new(component, prefix);
    for item in items {
        accumulator.observe(&item)?;
    }
    Ok(accumulator.finish())
}

struct ReplicaBucketAccumulator {
    hasher: blake3::Hasher,
    count: u64,
    bytes: u64,
}

impl ReplicaBucketAccumulator {
    fn new(component: ReplicaComponent, prefix: u8) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SUMMARY_DOMAIN);
        hasher.update(&SUMMARY_VERSION.to_be_bytes());
        hasher.update(&[component as u8, prefix]);
        Self {
            hasher,
            count: 0,
            bytes: 0,
        }
    }

    fn observe(&mut self, item: &ReplicaItem) -> anyhow::Result<()> {
        let id = item.id();
        let len = item.encoded_len();
        self.hasher.update(&id.0);
        self.hasher.update(&len.to_be_bytes());
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("custody bucket item count overflow"))?;
        self.bytes = self
            .bytes
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("custody bucket byte count overflow"))?;
        Ok(())
    }

    fn finish(self) -> ReplicaBucketSummary {
        ReplicaBucketSummary {
            count: self.count,
            bytes: self.bytes,
            digest: *self.hasher.finalize().as_bytes(),
        }
    }
}

fn validate_completed_bucket(
    received: ReplicaBucketAccumulator,
    advertised: ReplicaBucketSummary,
) -> anyhow::Result<()> {
    if received.finish() != advertised {
        anyhow::bail!(
            "custody peer's completed inventory bucket does not match its advertised summary"
        );
    }
    Ok(())
}

fn validate_partial_bucket(
    received: &ReplicaBucketAccumulator,
    advertised: ReplicaBucketSummary,
) -> anyhow::Result<()> {
    if received.count > advertised.count || received.bytes > advertised.bytes {
        anyhow::bail!("custody peer inventory exceeds its advertised bucket count or byte total");
    }
    Ok(())
}

/// Complete immutable serving view retained between explicit custody sweeps.
pub(crate) struct CustodyStoreSnapshot<R> {
    reader: R,
    inventory: ReplicaSnapshotData,
}

#[derive(Default)]
struct ReplicaKnownIds {
    blobs: BTreeSet<[u8; 32]>,
    records: BTreeSet<Id>,
    proofs: BTreeSet<[u8; 32]>,
}

impl ReplicaSnapshotData {
    fn known_ids(&self) -> ReplicaKnownIds {
        ReplicaKnownIds {
            blobs: self
                .blobs
                .iter()
                .flatten()
                .map(|info| info.handle.raw)
                .collect(),
            records: self
                .records
                .iter()
                .flatten()
                .map(CollectionRecord::id)
                .collect(),
            proofs: self
                .proofs
                .iter()
                .flatten()
                .map(|proof| proof.id().raw)
                .collect(),
        }
    }
}

impl<R> CustodyStoreSnapshot<R> {
    fn summary_clone(&self) -> ReplicaSummary {
        self.inventory.summary().clone()
    }

    fn known_ids(&self) -> ReplicaKnownIds {
        self.inventory.known_ids()
    }
}

impl<R> crate::host::AnyReplicaSnapshot for CustodyStoreSnapshot<R>
where
    R: BlobStoreGet + Send + 'static,
{
    fn summary(&self) -> ReplicaSummary {
        self.inventory.summary().clone()
    }

    fn page(
        &self,
        component: ReplicaComponent,
        prefix: u8,
        after: Option<ReplicaItemId>,
    ) -> (Vec<ReplicaItem>, bool) {
        self.inventory.page(component, prefix, after)
    }

    fn blob_bytes(&self, id: ReplicaItemId) -> Option<anybytes::Bytes> {
        ReplicaSnapshotData::blob_bytes(&self.reader, id)
    }
}

/// Build the complete immutable custody snapshot for one known store prefix.
///
/// This is intentionally called only by the custody driver, never by ordinary
/// [`Peer`](crate::peer::Peer) refresh.
pub(crate) fn snapshot_from_store<S>(
    store: &mut S,
) -> anyhow::Result<CustodyStoreSnapshot<S::Reader>>
where
    S: BlobStore + CollectionStore + CapabilityProofStore,
{
    let records: Vec<_> = store.records()?.collect::<Result<_, _>>()?;
    let proofs: Vec<_> = store.proofs()?.collect::<Result<_, _>>()?;
    let reader = store.reader()?;
    let data = ReplicaSnapshotData::new(&reader, records.iter().copied(), proofs)?;
    Ok(CustodyStoreSnapshot {
        reader,
        inventory: data,
    })
}

/// Observable result of one bounded custody anti-entropy sweep.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CustodyReconcileOutcome {
    /// Static peers whose summary was requested.
    pub peers_attempted: usize,
    /// Peers that completed blobs, records, and proofs during this sweep.
    pub peers_completed: usize,
    /// Peers that were unavailable or failed one component. Healthy peers
    /// continue independently.
    pub peer_errors: Vec<String>,
    /// Newly admitted resident blobs.
    pub blobs_added: u64,
    /// Payload bytes in newly admitted blobs.
    pub blob_bytes_added: u64,
    /// Newly observed native collection records.
    pub collection_records_added: u64,
    /// Newly observed native capability proofs.
    pub capability_proofs_added: u64,
    /// Remote inventory pages consumed.
    pub pages_read: u64,
}

/// Exact local inventory observed during a no-bind custody preflight.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CustodyInventoryStats {
    pub blobs: u64,
    pub blob_bytes: u64,
    pub collection_records: u64,
    pub capability_proofs: u64,
    pub build_elapsed: std::time::Duration,
}

impl ReplicaSummary {
    fn inventory_stats(&self, build_elapsed: std::time::Duration) -> CustodyInventoryStats {
        let total = |component| {
            self.buckets(component)
                .iter()
                .fold((0u64, 0u64), |(count, bytes), bucket| {
                    (
                        count.saturating_add(bucket.count),
                        bytes.saturating_add(bucket.bytes),
                    )
                })
        };
        let (blobs, blob_bytes) = total(ReplicaComponent::Blobs);
        let (collection_records, _) = total(ReplicaComponent::CollectionRecords);
        let (capability_proofs, _) = total(ReplicaComponent::CapabilityProofs);
        CustodyInventoryStats {
            blobs,
            blob_bytes,
            collection_records,
            capability_proofs,
            build_elapsed,
        }
    }
}

struct RemoteInventory {
    peer: crate::transport::PeerId,
    summary: ReplicaSummary,
}

/// Failure while receiving one remote blob, classified by the side that must
/// change before a retry can make progress.
pub(crate) enum ReplicaBlobFetchError {
    Remote(anyhow::Error),
    Local(anyhow::Error),
}

impl ReplicaBlobFetchError {
    pub(crate) fn remote(error: impl Into<anyhow::Error>) -> Self {
        Self::Remote(error.into())
    }

    pub(crate) fn local(error: impl Into<anyhow::Error>) -> Self {
        Self::Local(error.into())
    }
}

enum UnionComponentError {
    Remote(anyhow::Error),
    Local(anyhow::Error),
}

impl UnionComponentError {
    fn remote(error: impl Into<anyhow::Error>) -> Self {
        Self::Remote(error.into())
    }

    fn local(error: impl Into<anyhow::Error>) -> Self {
        Self::Local(error.into())
    }
}

/// Startup failure that preserves ownership of the unopened/unserved store.
///
/// A pile-backed caller can recover the store with [`Self::into_parts`] and
/// close it explicitly instead of triggering a dropped-without-close warning.
pub struct CustodyReplicaStartError<S> {
    store: S,
    error: anyhow::Error,
}

impl<S> CustodyReplicaStartError<S> {
    fn new(store: S, error: impl Into<anyhow::Error>) -> Self {
        Self {
            store,
            error: error.into(),
        }
    }

    /// Recover the store together with the startup cause.
    pub fn into_parts(self) -> (S, anyhow::Error) {
        (self.store, self.error)
    }
}

impl<S> std::fmt::Debug for CustodyReplicaStartError<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CustodyReplicaStartError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<S> std::fmt::Display for CustodyReplicaStartError<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "custody replica startup failed: {:#}",
            self.error
        )
    }
}

impl<S> std::error::Error for CustodyReplicaStartError<S> {}

/// Host-join failure that still returns ownership of the local store.
pub struct CustodyReplicaShutdownError<S> {
    store: S,
    error: anyhow::Error,
}

impl<S> CustodyReplicaShutdownError<S> {
    /// Recover the store so a caller can still flush and close it explicitly.
    pub fn into_parts(self) -> (S, anyhow::Error) {
        (self.store, self.error)
    }
}

impl<S> std::fmt::Debug for CustodyReplicaShutdownError<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CustodyReplicaShutdownError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<S> std::fmt::Display for CustodyReplicaShutdownError<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "custody replica shutdown failed: {:#}",
            self.error
        )
    }
}

impl<S> std::error::Error for CustodyReplicaShutdownError<S> {}

/// Explicit, proof-gated full-residency replica over static private peers.
///
/// Unlike [`Peer`](crate::peer::Peer), this type never joins public gossip,
/// announces content, records wants, or performs DHT discovery. The owned store
/// grows by componentwise union of all resident blobs, all native collection
/// records, and all native capability proofs.
pub struct CustodyReplica<S> {
    sender: crate::host::NetSender,
    _receiver: crate::host::NetReceiver,
    host: Option<crate::host::CustodyHostThread>,
    store: S,
    config: CustodyReplicaConfig,
    peers: Vec<crate::transport::PeerId>,
}

impl<S> CustodyReplica<S>
where
    S: BlobStore + CollectionStore + CapabilityProofStore + StorageFlush + Send + 'static,
{
    /// Validate all production configuration and build a complete, hash-
    /// validated local inventory without binding a socket.
    ///
    /// Pile-specific callers must additionally reject opaque future records;
    /// that property intentionally is not part of the generic store traits.
    pub fn preflight(
        store: &mut S,
        presenter: ed25519_dalek::VerifyingKey,
        config: &CustodyReplicaConfig,
    ) -> anyhow::Result<CustodyInventoryStats> {
        let started = std::time::Instant::now();
        validate_transport_config(presenter, config)?;
        validate_presenter_proofs(presenter, config)?;
        let snapshot = snapshot_from_store(store)?;
        Ok(snapshot
            .inventory
            .summary()
            .inventory_stats(started.elapsed()))
    }

    /// Start a production custody node on its fixed private-fabric socket.
    pub fn new(
        mut store: S,
        key: ed25519_dalek::SigningKey,
        config: CustodyReplicaConfig,
    ) -> Result<Self, CustodyReplicaStartError<S>> {
        if let Err(error) = validate_transport_config(key.verifying_key(), &config) {
            return Err(CustodyReplicaStartError::new(store, error));
        }
        if let Err(error) = validate_presenter_proofs(key.verifying_key(), &config) {
            return Err(CustodyReplicaStartError::new(store, error));
        }
        let snapshot = match snapshot_from_store(&mut store) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(CustodyReplicaStartError::new(store, error)),
        };
        let peer_config = crate::host::PeerConfig {
            peers: config.peers.clone(),
            gossip_topic: None,
            connect_root: config.connect_root,
            connect_proof: config.connect_proof.clone(),
            direction: crate::host::SyncDirection::ReadOnly,
        };
        let server = ReplicaServerConfig {
            trust_root: config.replica_root,
            replica_set: config.replica_set,
        };
        let (sender, receiver, host) =
            match crate::host::spawn_custody(key, peer_config, server, config.bind_addr) {
                Ok(host) => host,
                Err(error) => {
                    drop(snapshot);
                    return Err(CustodyReplicaStartError::new(store, error));
                }
            };
        Ok(Self::assemble(
            store,
            sender,
            receiver,
            Some(host),
            config,
            snapshot,
        ))
    }

    /// Assemble a custody node over caller-provided transport wiring.
    ///
    /// This is the deterministic simulation seam. Production callers use
    /// [`Self::new`], whose binder is fixed-address and explicit-only.
    pub fn with_wiring(
        mut store: S,
        presenter: ed25519_dalek::VerifyingKey,
        sender: crate::host::NetSender,
        receiver: crate::host::NetReceiver,
        config: CustodyReplicaConfig,
    ) -> Result<Self, CustodyReplicaStartError<S>> {
        if let Err(error) = validate_presenter_proofs(presenter, &config) {
            return Err(CustodyReplicaStartError::new(store, error));
        }
        let snapshot = match snapshot_from_store(&mut store) {
            Ok(snapshot) => snapshot,
            Err(error) => return Err(CustodyReplicaStartError::new(store, error)),
        };
        Ok(Self::assemble(
            store, sender, receiver, None, config, snapshot,
        ))
    }

    fn assemble(
        store: S,
        sender: crate::host::NetSender,
        receiver: crate::host::NetReceiver,
        host: Option<crate::host::CustodyHostThread>,
        config: CustodyReplicaConfig,
        snapshot: CustodyStoreSnapshot<S::Reader>,
    ) -> Self {
        let mut peers: Vec<_> = config
            .peers
            .iter()
            .map(|address| *address.id.as_bytes())
            .filter(|peer| *peer != *sender.id().as_bytes())
            .collect();
        peers.sort_unstable();
        peers.dedup();

        sender.update_replica_snapshot(snapshot);
        Self {
            sender,
            _receiver: receiver,
            host,
            store,
            config,
            peers,
        }
    }

    /// This node's stable transport identity.
    pub fn id(&self) -> iroh_base::EndpointId {
        self.sender.id()
    }

    /// Exact fixed direct address published after production bind succeeds.
    pub fn endpoint_addr(&self) -> iroh_base::EndpointAddr {
        iroh_base::EndpointAddr::from_parts(
            self.id(),
            [iroh_base::TransportAddr::Ip(self.config.bind_addr)],
        )
    }

    /// Borrow the underlying semantic store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Mutably borrow the underlying semantic store.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Stop and join the production host before returning the owned store.
    ///
    /// This does not add a storage flush; callers retain explicit control over
    /// the final durability barrier and close operation.
    pub fn shutdown(self) -> Result<S, CustodyReplicaShutdownError<S>> {
        let Self {
            sender,
            _receiver,
            host,
            store,
            ..
        } = self;
        drop(sender);
        drop(_receiver);
        if let Some(host) = host {
            if let Err(error) = host.join() {
                return Err(CustodyReplicaShutdownError { store, error });
            }
        }
        Ok(store)
    }

    fn publish_current_snapshot(&mut self) -> anyhow::Result<(ReplicaSummary, ReplicaKnownIds)> {
        let snapshot = snapshot_from_store(&mut self.store)?;
        let summary = snapshot.summary_clone();
        let known = snapshot.known_ids();
        self.sender.update_replica_snapshot(snapshot);
        Ok((summary, known))
    }

    /// Run one anti-entropy union sweep.
    ///
    /// Every successful peer completes its blob phase before any of its record
    /// or proof evidence is admitted. A failed peer is skipped for later
    /// phases, while other peers continue; a later sweep repairs it after the
    /// partition heals.
    pub async fn reconcile_once(&mut self) -> anyhow::Result<CustodyReconcileOutcome> {
        let (local_summary, mut known) = self.publish_current_snapshot()?;
        let mut outcome = CustodyReconcileOutcome {
            peers_attempted: self.peers.len(),
            ..CustodyReconcileOutcome::default()
        };
        let mut remotes = Vec::new();
        let mut failed = BTreeSet::new();
        for peer in self.peers.iter().copied() {
            match self
                .sender
                .replica_summary(
                    peer,
                    self.config.replica_set,
                    self.config.replica_proof.clone(),
                )
                .await
            {
                Ok(summary) => remotes.push(RemoteInventory { peer, summary }),
                Err(error) => {
                    failed.insert(peer);
                    outcome
                        .peer_errors
                        .push(format!("{} summary: {error:#}", hex::encode(&peer[..4])));
                }
            }
        }

        // Phase 1: every healthy peer's complete resident blob inventory.
        for remote in &remotes {
            match self
                .union_component(
                    remote,
                    ReplicaComponent::Blobs,
                    &local_summary,
                    &mut known,
                    &mut outcome,
                )
                .await
            {
                Ok(()) => {}
                Err(UnionComponentError::Remote(error)) => {
                    failed.insert(remote.peer);
                    outcome.peer_errors.push(format!(
                        "{} blobs: {error:#}",
                        hex::encode(&remote.peer[..4])
                    ));
                }
                Err(UnionComponentError::Local(error)) => {
                    return Err(error.context("admit or flush custody blob component"));
                }
            }
        }

        // Phases 2 and 3 only advance peers whose blob inventory completed.
        for component in [
            ReplicaComponent::CollectionRecords,
            ReplicaComponent::CapabilityProofs,
        ] {
            for remote in &remotes {
                if failed.contains(&remote.peer) {
                    continue;
                }
                match self
                    .union_component(remote, component, &local_summary, &mut known, &mut outcome)
                    .await
                {
                    Ok(()) => {}
                    Err(UnionComponentError::Remote(error)) => {
                        failed.insert(remote.peer);
                        outcome.peer_errors.push(format!(
                            "{} component {}: {error:#}",
                            hex::encode(&remote.peer[..4]),
                            component as u8
                        ));
                    }
                    Err(UnionComponentError::Local(error)) => {
                        return Err(error.context(format!(
                            "admit or flush custody component {}",
                            component as u8
                        )));
                    }
                }
            }
        }

        if outcome.blobs_added != 0
            || outcome.collection_records_added != 0
            || outcome.capability_proofs_added != 0
        {
            // Every component method has already crossed its final durability
            // barrier. Rebuild only the immutable serving view; an idle sweep
            // is entirely read-only and never calls flush.
            self.publish_current_snapshot()?;
        }
        outcome.peers_completed = remotes
            .iter()
            .filter(|remote| !failed.contains(&remote.peer))
            .count();
        Ok(outcome)
    }

    async fn union_component(
        &mut self,
        remote: &RemoteInventory,
        component: ReplicaComponent,
        local_summary: &ReplicaSummary,
        known: &mut ReplicaKnownIds,
        outcome: &mut CustodyReconcileOutcome,
    ) -> Result<(), UnionComponentError> {
        let mut dirty = false;
        let mut unflushed_blob_bytes = 0u64;
        let work: Result<(), UnionComponentError> = async {
            for prefix in 0..=u8::MAX {
                if local_summary.bucket(component, prefix)
                    == remote.summary.bucket(component, prefix)
                {
                    continue;
                }
                let mut after = None;
                let mut received = ReplicaBucketAccumulator::new(component, prefix);
                loop {
                    let (page, done) = self
                        .sender
                        .replica_page(
                            remote.peer,
                            self.config.replica_set,
                            self.config.replica_proof.clone(),
                            component,
                            prefix,
                            after,
                        )
                        .await
                        .map_err(UnionComponentError::remote)?;
                    outcome.pages_read = outcome.pages_read.saturating_add(1);
                    if page.is_empty() && !done {
                        return Err(UnionComponentError::remote(anyhow::anyhow!(
                            "custody peer returned an empty non-final inventory page"
                        )));
                    }
                    for item in &page {
                        received
                            .observe(item)
                            .map_err(UnionComponentError::remote)?;
                    }
                    let advertised = remote.summary.bucket(component, prefix);
                    validate_partial_bucket(&received, advertised)
                        .map_err(UnionComponentError::remote)?;
                    if !done && received.count == advertised.count {
                        return Err(UnionComponentError::remote(anyhow::anyhow!(
                            "custody peer marked a complete advertised item count as non-final"
                        )));
                    }
                    let last = page.last().map(ReplicaItem::id);
                    let mut page_dirty = false;
                    for item in page {
                        match item {
                            ReplicaItem::Blob(info) => {
                                if known.blobs.contains(&info.handle.raw) {
                                    continue;
                                }
                                let id = ReplicaItemId(info.handle.raw);
                                let bytes = self
                                    .sender
                                    .replica_blob(
                                        remote.peer,
                                        self.config.replica_set,
                                        self.config.replica_proof.clone(),
                                        id,
                                        info.length,
                                        self.config.receive_temp_dir.clone(),
                                    )
                                    .await
                                    .map_err(|error| match error {
                                        ReplicaBlobFetchError::Remote(error) => {
                                            UnionComponentError::Remote(error)
                                        }
                                        ReplicaBlobFetchError::Local(error) => {
                                            UnionComponentError::Local(error)
                                        }
                                    })?
                                    .ok_or_else(|| {
                                        UnionComponentError::remote(anyhow::anyhow!(
                                            "advertised blob {} became unavailable",
                                            hex::encode(info.handle.raw)
                                        ))
                                    })?;
                                if bytes.len() as u64 != info.length {
                                    return Err(UnionComponentError::remote(anyhow::anyhow!(
                                        "downloaded blob failed final length validation"
                                    )));
                                }
                                use triblespace_core::blob::{Blob, encodings::UnknownBlob};
                                use triblespace_core::inline::Inline;
                                use triblespace_core::inline::encodings::hash::Handle;
                                // replica_blob verified the complete stream hash. Preserve
                                // that cached handle so admission does not hash a multi-GB
                                // mmap twice more.
                                let blob = Blob::<UnknownBlob>::with_handle(
                                    bytes,
                                    Inline::<Handle<UnknownBlob>>::new(info.handle.raw),
                                );
                                let admitted =
                                    self.store.put::<UnknownBlob, _>(blob).map_err(|error| {
                                        UnionComponentError::local(anyhow::Error::new(error))
                                    })?;
                                if admitted.raw != info.handle.raw {
                                    return Err(UnionComponentError::local(anyhow::anyhow!(
                                        "destination returned a different blob handle"
                                    )));
                                }
                                known.blobs.insert(info.handle.raw);
                                outcome.blobs_added = outcome.blobs_added.saturating_add(1);
                                outcome.blob_bytes_added =
                                    outcome.blob_bytes_added.saturating_add(info.length);
                                unflushed_blob_bytes =
                                    unflushed_blob_bytes.saturating_add(info.length);
                                dirty = true;
                                if unflushed_blob_bytes >= BLOB_DURABILITY_BATCH_BYTES {
                                    self.store.flush().map_err(|error| {
                                        UnionComponentError::local(anyhow::Error::new(error))
                                    })?;
                                    dirty = false;
                                    unflushed_blob_bytes = 0;
                                }
                            }
                            ReplicaItem::CollectionRecord(record) => {
                                if known.records.contains(&record.id()) {
                                    continue;
                                }
                                self.store.insert(record).map_err(|error| {
                                    UnionComponentError::local(anyhow::Error::new(error))
                                })?;
                                known.records.insert(record.id());
                                outcome.collection_records_added =
                                    outcome.collection_records_added.saturating_add(1);
                                dirty = true;
                                page_dirty = true;
                            }
                            ReplicaItem::CapabilityProof(proof) => {
                                let proof_id = proof.id().raw;
                                if known.proofs.contains(&proof_id) {
                                    continue;
                                }
                                self.store.insert_proof(proof).map_err(|error| {
                                    UnionComponentError::local(anyhow::Error::new(error))
                                })?;
                                known.proofs.insert(proof_id);
                                outcome.capability_proofs_added =
                                    outcome.capability_proofs_added.saturating_add(1);
                                dirty = true;
                                page_dirty = true;
                            }
                        }
                    }
                    if component != ReplicaComponent::Blobs && page_dirty {
                        // Record/proof response bytes are bounded by their page
                        // limits, so a page is the durability batch.
                        self.store.flush().map_err(|error| {
                            UnionComponentError::local(anyhow::Error::new(error))
                        })?;
                        dirty = false;
                    }
                    if done {
                        validate_completed_bucket(received, advertised)
                            .map_err(UnionComponentError::remote)?;
                        break;
                    }
                    after = last;
                }
            }
            Ok(())
        }
        .await;

        // This is both the successful component barrier and the cleanup path
        // after a mid-page network error. Thus records/proofs never begin while
        // admitted blob writes remain unflushed.
        let flush: Result<(), UnionComponentError> = if dirty {
            self.store
                .flush()
                .map_err(|error| UnionComponentError::local(anyhow::Error::new(error)))
        } else {
            Ok(())
        };
        match (work, flush) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(UnionComponentError::Remote(work)), Err(UnionComponentError::Local(flush)))
            | (Err(UnionComponentError::Local(work)), Err(UnionComponentError::Local(flush))) => {
                Err(UnionComponentError::local(anyhow::anyhow!(
                    "{work:#}; additionally failed to flush admitted custody data: {flush:#}"
                )))
            }
            // `flush` is constructed only from a local storage operation.
            (Err(work), Err(UnionComponentError::Remote(_))) => Err(work),
        }
    }
}

fn validate_transport_config(
    presenter: ed25519_dalek::VerifyingKey,
    config: &CustodyReplicaConfig,
) -> anyhow::Result<()> {
    validate_direct_socket(config.bind_addr, "custody bind")?;
    let metadata = std::fs::metadata(&config.receive_temp_dir).map_err(|error| {
        anyhow::anyhow!(
            "inspect custody receive directory {}: {error}",
            config.receive_temp_dir.display()
        )
    })?;
    if !metadata.is_dir() {
        anyhow::bail!(
            "custody receive path {} is not a directory",
            config.receive_temp_dir.display()
        );
    }
    drop(
        tempfile::tempfile_in(&config.receive_temp_dir).map_err(|error| {
            anyhow::anyhow!(
                "custody receive directory {} is not writable: {error}",
                config.receive_temp_dir.display()
            )
        })?,
    );

    let mut peer_ids = BTreeSet::new();
    for peer in &config.peers {
        if peer.id.as_bytes() == presenter.as_bytes() {
            anyhow::bail!("custody peer list contains this node's own endpoint id");
        }
        if !peer_ids.insert(*peer.id.as_bytes()) {
            anyhow::bail!("custody peer list repeats one endpoint id");
        }
        if peer.addrs.len() != 1 {
            anyhow::bail!(
                "custody peer {} must name exactly one explicit private-fabric IP socket",
                peer.id,
            );
        }
        let Some(iroh_base::TransportAddr::Ip(socket)) = peer.addrs.first() else {
            anyhow::bail!(
                "custody peer {} contains a non-IP route; relay/custom discovery is forbidden",
                peer.id
            );
        };
        validate_direct_socket(*socket, "custody peer")?;
    }
    Ok(())
}

fn validate_direct_socket(socket: std::net::SocketAddr, label: &str) -> anyhow::Result<()> {
    if socket.port() == 0 {
        anyhow::bail!("{label} port must be restart-stable and nonzero");
    }
    if socket.ip().is_unspecified() || socket.ip().is_multicast() {
        anyhow::bail!("{label} address {socket} is not one explicit unicast interface");
    }
    if matches!(socket.ip(), std::net::IpAddr::V4(ip) if ip.is_broadcast()) {
        anyhow::bail!("{label} address {socket} is broadcast");
    }
    Ok(())
}

fn validate_presenter_proofs(
    presenter: ed25519_dalek::VerifyingKey,
    config: &CustodyReplicaConfig,
) -> anyhow::Result<()> {
    config.connect_proof.verify(
        config.connect_root,
        crate::clock::epoch_now(),
        presenter,
        CapabilityRequest::new(
            crate::protocol::connect_capability_atom(config.connect_root),
            CapabilityMode::Invoke,
        ),
    )?;
    let replica = config.replica_proof.verify(
        config.replica_root,
        crate::clock::epoch_now(),
        presenter,
        CapabilityRequest::new(
            replicate_capability_atom(config.replica_set),
            CapabilityMode::Invoke,
        ),
    )?;
    if replica.effective_mode() != CapabilityMode::Invoke {
        anyhow::bail!("custody presenter proof must end in exact invoke-only authority");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anybytes::Bytes;
    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::encodings::UnknownBlob;
    use triblespace_core::capability::{CapabilityClaim, CapabilityMode};
    use triblespace_core::collection::{CollectionData, CollectionMerge, empty_metadata_handle};
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::repo::memoryrepo::MemoryRepo;

    use super::*;

    #[test]
    fn replica_resource_is_exact_and_independent_of_connect() {
        let id = ReplicaSetId::new([7; 32]);
        let atom = replicate_capability_atom(id);
        assert_eq!(atom.action().id(), ACTION_REPLICATE_STORE);
        assert_eq!(atom.resource().into_bytes(), [7; 32]);
        assert_ne!(ACTION_REPLICATE_STORE, crate::protocol::ACTION_CONNECT);
    }

    #[test]
    fn snapshot_sorts_components_and_pages_with_an_exclusive_cursor() {
        let signer = SigningKey::from_bytes(&[9; 32]);
        let mut repo = MemoryRepo::default();
        let a = repo.put::<UnknownBlob, _>(Bytes::from(vec![1])).unwrap();
        let b = repo.put::<UnknownBlob, _>(Bytes::from(vec![2])).unwrap();
        let merge = CollectionRecord::Merge(CollectionMerge::new(
            triblespace_core::inline::Inline::new([3; 32]),
            CollectionData::new([4; 32]),
            CollectionData::new([5; 32]),
            CollectionData::new([6; 32]),
        ));
        repo.insert(merge).unwrap();
        let claim = CapabilityClaim::root(
            replicate_capability_atom(ReplicaSetId::new([7; 32])),
            CapabilityMode::Invoke,
            None,
        );
        let proof = CapabilityProofBundle::issue_root(&signer, claim, signer.verifying_key())
            .unwrap()
            .proof()
            .clone();
        repo.insert_proof(proof.clone()).unwrap();

        let reader = repo.reader().unwrap();
        let snapshot = ReplicaSnapshotData::new(&reader, [merge], [proof]).unwrap();
        for handle in [a, b] {
            let prefix = handle.raw[0];
            let (page, done) = snapshot.page(ReplicaComponent::Blobs, prefix, None);
            assert!(done);
            assert!(page.iter().any(|item| item.id().0 == handle.raw));
        }
        let prefix = merge.id().raw()[0];
        let (page, done) = snapshot.page(ReplicaComponent::CollectionRecords, prefix, None);
        assert!(done);
        assert_eq!(page.len(), 1);
        let cursor = page[0].id();
        let (empty, done) =
            snapshot.page(ReplicaComponent::CollectionRecords, prefix, Some(cursor));
        assert!(done);
        assert!(empty.is_empty());
    }

    #[test]
    fn strict_commit_fixture_is_available_for_wire_validation() {
        let signer = SigningKey::from_bytes(&[11; 32]);
        let commit = triblespace_core::collection::CollectionCommit::sign(
            &signer,
            triblespace_core::inline::Inline::new([1; 32]),
            CollectionData::new([2; 32]),
            empty_metadata_handle(),
        );
        commit.verify_strict().unwrap();
    }

    #[test]
    fn completed_bucket_rejects_a_truncated_page_sequence() {
        use triblespace_core::inline::Inline;
        use triblespace_core::inline::encodings::hash::Handle;

        let first = ReplicaItem::Blob(BlobInfo {
            handle: Inline::<Handle<UnknownBlob>>::new([0x42; 32]),
            length: 7,
        });
        let mut second_id = [0x43; 32];
        second_id[0] = 0x42;
        let second = ReplicaItem::Blob(BlobInfo {
            handle: Inline::<Handle<UnknownBlob>>::new(second_id),
            length: 11,
        });
        let advertised = summarize(ReplicaComponent::Blobs, 0x42, [first.clone(), second]).unwrap();
        let mut truncated = ReplicaBucketAccumulator::new(ReplicaComponent::Blobs, 0x42);
        truncated.observe(&first).unwrap();
        assert!(validate_completed_bucket(truncated, advertised).is_err());

        let mut excess = ReplicaBucketAccumulator::new(ReplicaComponent::Blobs, 0x42);
        excess.observe(&first).unwrap();
        excess
            .observe(&ReplicaItem::Blob(BlobInfo {
                handle: Inline::<Handle<UnknownBlob>>::new(second_id),
                length: 11,
            }))
            .unwrap();
        assert!(
            validate_partial_bucket(
                &excess,
                ReplicaBucketSummary {
                    count: 1,
                    bytes: 7,
                    digest: [0; 32],
                }
            )
            .is_err()
        );
    }
}
