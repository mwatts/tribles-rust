//! A synchronous store wrapped in collection-scoped anti-entropy.
//!
//! The host runtime repairs immutable per-collection activation overlays. This
//! side owns the only mutable store boundary: authenticated leaves are deduplicated,
//! inserted monotonically, flushed once per drain, and only then exposed in a
//! replacement serving snapshot. Exact blob reads keep their durable-WANT
//! semantics independently of broad inventory mirroring.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{BlobEncoding, IntoBlob, TryFromBlob};
use triblespace_core::collection::{
    CollectionHandle, CollectionRead, CollectionStore, DisclosureSnapshot,
};
use triblespace_core::inline::Inline;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{Entry as PatchEntry, PATCH};
use triblespace_core::repo::lazy::WantRecordError;
use triblespace_core::repo::{
    BlobChildren, BlobStore, BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut,
    CapabilityProofRead, CapabilityProofStore, PeerRead, SnapshotSource, StorageFlush,
    StoreChanges, StoreRead, StoreSnapshot as CoreStoreSnapshot, WantRequest, WantStore,
};

use crate::channel::{MAX_ADMISSION_BRIDGE_BATCHES, NetEvent};
use crate::host::{self, ActiveCollections, NetReceiver, NetSender, StoreSnapshot};
use crate::protocol::RawHash;
use crate::provider::{ArtifactId, ProviderObservation};
use crate::wake::CollectionWakePlane;

pub use crate::host::PeerConfig;
pub use crate::inventory::{ReconcileDirection, ReconcileQos};

/// Failure while starting a production network host.
#[derive(Debug)]
pub enum PeerOpenError {
    /// The production network thread, runtime, or iroh endpoint could not start.
    HostStartup(anyhow::Error),
}

impl fmt::Display for PeerOpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostStartup(error) => write!(f, "cannot start network host: {error}"),
        }
    }
}

impl Error for PeerOpenError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::HostStartup(error) => Some(error.as_ref()),
        }
    }
}

/// Failure while freezing the local observation behind a [`Peer`].
#[derive(Debug)]
pub enum PeerSnapshotError<SnapshotError> {
    /// The backing store could not freeze its coherent observation.
    Store(SnapshotError),
    /// An active collection could not be projected from the coherent store
    /// observation. The previous serving view is withdrawn.
    Overlay(anyhow::Error),
}

impl<SnapshotError> fmt::Display for PeerSnapshotError<SnapshotError>
where
    SnapshotError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "cannot freeze peer store snapshot: {error}"),
            Self::Overlay(error) => write!(formatter, "cannot build collection snapshot: {error}"),
        }
    }
}

impl<SnapshotError> Error for PeerSnapshotError<SnapshotError>
where
    SnapshotError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Overlay(error) => Some(error.as_ref()),
        }
    }
}

/// A store attached to a collection-scoped network host.
pub struct Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobChildren,
{
    store: Arc<Mutex<S>>,
    sender: NetSender,
    receiver: NetReceiver,
    wake_plane: Option<CollectionWakePlane>,
    qos: ReconcileQos,
    active: ActiveCollections,
    /// Network admissions stay outside the advertised snapshot until their
    /// shared durability barrier succeeds. A failed flush is retried on every
    /// refresh without requiring the remote to redeliver the event first.
    pending_network_flush: bool,
    /// Last local observation used to build the installed immutable inventory.
    /// Equality is a cheap invalidation check supplied by the store; it is not
    /// a portable generation or a semantic version.
    last_store_snapshot: Option<S::Snapshot>,
    /// Last snapshot-bound public provider set sent to the host. Rebuilding it
    /// on every refresh lets proof expiry narrow publication even when the
    /// store prefix itself did not change.
    last_provider_observation: ProviderObservation,
    last_event_at: crate::clock::Mono,
}

impl<S> Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobChildren,
{
    /// Spawn a production host. No team scope or connection proof exists.
    pub fn new(store: S, key: SigningKey, config: PeerConfig) -> Result<Self, PeerOpenError> {
        let qos = config.qos;
        let (sender, receiver, wake_plane) =
            host::spawn(key, config).map_err(PeerOpenError::HostStartup)?;
        Ok(Self::assemble(
            store,
            qos,
            sender,
            receiver,
            Some(wake_plane),
        ))
    }

    /// Attach a store to a caller-owned host, most commonly the deterministic
    /// simulator.
    pub fn with_wiring(
        store: S,
        qos: ReconcileQos,
        sender: NetSender,
        receiver: NetReceiver,
    ) -> Self {
        Self::assemble(store, qos, sender, receiver, None)
    }

    fn assemble(
        store: S,
        qos: ReconcileQos,
        sender: NetSender,
        receiver: NetReceiver,
        wake_plane: Option<CollectionWakePlane>,
    ) -> Self {
        let mut peer = Self {
            store: Arc::new(Mutex::new(store)),
            sender,
            receiver,
            wake_plane,
            qos,
            active: PATCH::new(),
            pending_network_flush: false,
            last_store_snapshot: None,
            last_provider_observation: ProviderObservation::default(),
            last_event_at: crate::clock::mono_now(),
        };
        peer.refresh();
        peer
    }

    pub fn id(&self) -> EndpointId {
        self.sender.id()
    }

    /// Stock gossip wake plane for a production iroh peer.
    ///
    /// Caller-owned wiring has no implicit wake handle and returns `None`.
    /// Collection possession is enough to join a production topic; following a
    /// wake into anti-entropy remains separately authorized.
    pub fn wake_plane(&self) -> Option<CollectionWakePlane> {
        self.wake_plane.clone()
    }

    pub const fn qos(&self) -> ReconcileQos {
        self.qos
    }

    pub fn last_event_at(&self) -> crate::clock::Mono {
        self.last_event_at
    }

    /// Activate one collection for serving, repair, and wake subscription.
    ///
    /// This is ephemeral process state. It writes no OFFER/GOSSIP marker and
    /// creates no global collection registry.
    pub fn activate_collection(&mut self, collection: CollectionHandle) {
        self.active.insert(&PatchEntry::new(&collection.raw));
        self.last_store_snapshot = None;
        self.refresh();
    }

    /// Fetch exact content through its authenticated DHT provider hints.
    /// This primitive neither records a WANT nor mutates the store.
    pub async fn fetch_blob(&self, hash: RawHash) -> Option<Vec<u8>> {
        self.sender
            .fetch_blob(hash, host::INTERACTIVE_FETCH_DEADLINE)
            .await
    }

    pub async fn fetch_blob_with_deadline(
        &self,
        hash: RawHash,
        budget: std::time::Duration,
    ) -> Option<Vec<u8>> {
        self.sender.fetch_blob(hash, budget).await
    }

    /// Look up soft provider hints for an already-known physical artifact.
    pub async fn find_artifact_providers(&self, artifact: ArtifactId) -> Vec<EndpointId> {
        self.sender
            .find_artifact_providers(artifact, host::INTERACTIVE_FETCH_DEADLINE)
            .await
            .into_iter()
            .filter_map(|provider| EndpointId::from_bytes(&provider).ok())
            .collect()
    }

    /// Drain authenticated collection progress, cross one durability barrier,
    /// then replace the immutable active-collection snapshot. Calling this
    /// with no events is still meaningful: file-backed stores reobserve
    /// external appends before periodic repair uses them.
    pub fn refresh(&mut self) {
        if let Err(error) = self.try_refresh() {
            tracing::warn!(%error, "collection serving snapshot unavailable");
        }
    }

    /// Drain pending network evidence and publish one coherent store snapshot.
    pub fn try_refresh(&mut self) -> Result<(), PeerSnapshotError<S::SnapshotError>> {
        let result = self.refresh_checked();
        if result.is_err() {
            self.sender.clear_snapshot();
            self.last_store_snapshot = None;
            self.last_provider_observation = ProviderObservation::default();
        }
        result
    }

    fn refresh_checked(&mut self) -> Result<(), PeerSnapshotError<S::SnapshotError>> {
        let mut incoming = Vec::new();
        for _ in 0..MAX_ADMISSION_BRIDGE_BATCHES {
            let Some(event) = self.receiver.try_recv() else {
                break;
            };
            self.last_event_at = crate::clock::mono_now();
            incoming.push(event);
        }

        let received_batches = incoming.len();
        let received = incoming.iter().map(|batch| batch.len()).sum::<usize>();
        let mut store = self.store.lock().expect("store mutex");
        let mut admitted = false;
        for batch in incoming {
            for event in batch.into_events() {
                match event {
                    NetEvent::CollectionRecord(record) => match store.insert(record) {
                        Ok(()) => admitted = true,
                        Err(error) => {
                            tracing::warn!(?error, "admitting collection repair record failed")
                        }
                    },
                    NetEvent::CapabilityProofBundle(bundle) => {
                        let (proof, claims) = bundle.into_parts();
                        let mut complete = true;
                        for claim in claims {
                            if let Err(error) = store.put::<SimpleArchive, _>(claim) {
                                complete = false;
                                tracing::warn!(
                                    ?error,
                                    "landing collection WRITE claim blob failed"
                                );
                                break;
                            }
                        }
                        if complete {
                            match store.insert_proof(proof) {
                                Ok(()) => admitted = true,
                                Err(error) => tracing::warn!(
                                    ?error,
                                    "admitting collection WRITE proof failed"
                                ),
                            }
                        }
                    }
                }
            }
        }
        self.pending_network_flush |= admitted;
        if self.pending_network_flush {
            match store.flush() {
                Ok(()) => {
                    self.pending_network_flush = false;
                    tracing::debug!(
                        received,
                        received_batches,
                        "collection repair admission durable"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        received,
                        received_batches,
                        "collection repair flush failed; snapshot withheld"
                    );
                }
            }
        }
        if !self.pending_network_flush {
            let snapshot = match store.snapshot() {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        "store snapshot unavailable; keeping prior collection view"
                    );
                    let provider_observation = Self::provider_observation_from_snapshot(
                        self.last_store_snapshot.as_ref(),
                        self.qos.direction.serves(),
                    );
                    Self::observe_provider_observation(
                        &self.sender,
                        &mut self.last_provider_observation,
                        provider_observation,
                    );
                    return Ok(());
                }
            };
            let provider_observation = Self::provider_observation_from_snapshot(
                Some(&snapshot),
                self.qos.direction.serves(),
            );
            let previous_snapshot = self.sender.current_snapshot();
            let changes = if previous_snapshot.is_none() {
                StoreChanges::ALL
            } else {
                self.last_store_snapshot
                    .as_ref()
                    .map_or(StoreChanges::ALL, |previous| {
                        snapshot.changes_since(previous)
                    })
            };
            // Even a semantic no-op installs the fresh read lease. Unchanged
            // activation PATCHes retain their Arc while exact-GET advances to
            // the new immutable store observation.
            let serving = StoreSnapshot::from_store_changes(
                snapshot.clone(),
                &self.active,
                previous_snapshot.as_deref(),
                changes,
            )
            .map_err(PeerSnapshotError::Overlay)?;
            self.sender.update_snapshot(serving);
            self.last_store_snapshot = Some(snapshot);
            Self::observe_provider_observation(
                &self.sender,
                &mut self.last_provider_observation,
                provider_observation,
            );
        } else {
            // A failed admission flush withholds a new snapshot, but the
            // already-installed prefix remains a valid read lease. Recompute
            // only its time-sensitive disclosure boundary.
            let provider_observation = Self::provider_observation_from_snapshot(
                self.last_store_snapshot.as_ref(),
                self.qos.direction.serves(),
            );
            Self::observe_provider_observation(
                &self.sender,
                &mut self.last_provider_observation,
                provider_observation,
            );
        }
        Ok(())
    }

    fn provider_observation_from_snapshot(
        snapshot: Option<&S::Snapshot>,
        serves: bool,
    ) -> ProviderObservation {
        let Some(snapshot) = snapshot else {
            return ProviderObservation::default();
        };
        match DisclosureSnapshot::build_at(snapshot, crate::clock::epoch_now()) {
            Ok(disclosure) => ProviderObservation::from_disclosure(&disclosure, serves),
            Err(error) => {
                tracing::warn!(%error, "collection disclosure snapshot unavailable");
                ProviderObservation::default()
            }
        }
    }

    fn observe_provider_observation(
        sender: &NetSender,
        last: &mut ProviderObservation,
        observation: ProviderObservation,
    ) {
        if *last != observation {
            sender.update_public_providers(observation.clone());
            *last = observation;
        }
    }

    pub fn store(&self) -> MutexGuard<'_, S> {
        self.store.lock().expect("store mutex")
    }

    pub fn into_store(self) -> S {
        let Self { store, .. } = self;
        Arc::try_unwrap(store)
            .unwrap_or_else(|_| {
                panic!("Peer::into_store: an outstanding PeerSnapshot still shares the store")
            })
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn try_local(&mut self, hash: RawHash) -> Option<Bytes> {
        self.snapshot()
            .ok()?
            .get::<Bytes, UnknownBlob>(Inline::new(hash))
            .ok()
    }

    /// Local lookup followed by durable WANT, exact authenticated fetch, and
    /// best-effort landing. A failed fetch leaves the flushed WANT pending.
    pub async fn get_or_fetch_async(
        &mut self,
        hash: RawHash,
    ) -> Result<Option<Bytes>, WantRecordError<S::WantError, <S as StorageFlush>::Error>> {
        if let Some(bytes) = self.try_local(hash) {
            return Ok(Some(bytes));
        }
        {
            let mut store = self.store.lock().expect("store mutex");
            store
                .want(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)))
                .map_err(WantRecordError::Want)?;
            store.flush().map_err(WantRecordError::Flush)?;
        }
        let Some(raw) = self.fetch_blob(hash).await else {
            return Ok(None);
        };
        let bytes = Bytes::from(raw);
        {
            let mut store = self.store.lock().expect("store mutex");
            if let Err(error) = store.put::<UnknownBlob, Bytes>(bytes.clone()) {
                tracing::warn!(?error, "landing demand-fetched blob failed");
            }
        }
        Ok(Some(bytes))
    }
}

impl<S> BlobStorePut for Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobChildren,
{
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> Result<Inline<Handle<Sch>>, Self::PutError>
    where
        Sch: BlobEncoding + 'static,
        T: IntoBlob<Sch>,
        Handle<Sch>: InlineEncoding,
    {
        let mut store = self.store.lock().expect("store mutex");
        store.put(item)
    }
}

impl<S> SnapshotSource for Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobChildren,
{
    type Snapshot = PeerSnapshot<S::Snapshot>;
    type SnapshotError = PeerSnapshotError<S::SnapshotError>;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        self.try_refresh()?;
        let mut store = self.store.lock().expect("store mutex");
        let local = store.snapshot().map_err(PeerSnapshotError::Store)?;
        drop(store);
        let fetch = Some(FetchCap {
            sender: self.sender.clone(),
            sink: Arc::new(SharedStore {
                store: self.store.clone(),
            }),
        });
        Ok(PeerSnapshot { local, fetch })
    }
}

/// A frozen local store observation plus an optional live exact-fetch
/// capability.
///
/// Synchronous reads and listings remain fixed at `local`. Async retrieval may
/// acquire explicitly addressed immutable bytes through the swarm and cache
/// them operationally, but never extends this snapshot's frozen inventory.
pub struct PeerSnapshot<L> {
    local: L,
    fetch: Option<FetchCap>,
}

#[derive(Clone)]
struct FetchCap {
    sender: NetSender,
    sink: Arc<dyn StoreSink>,
}

trait StoreSink: Send + Sync {
    fn record_want(&self, hash: RawHash) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    fn land(&self, bytes: Bytes);
}

struct SharedStore<S> {
    store: Arc<Mutex<S>>,
}

impl<S> StoreSink for SharedStore<S>
where
    S: BlobStorePut + WantStore + StorageFlush + Send + 'static,
{
    fn record_want(&self, hash: RawHash) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut store = self.store.lock().expect("store mutex");
        store
            .want(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)))
            .map_err(|error| {
                Box::new(WantRecordError::<_, <S as StorageFlush>::Error>::Want(
                    error,
                )) as Box<dyn std::error::Error + Send + Sync>
            })?;
        store.flush().map_err(|error| {
            Box::new(WantRecordError::<S::WantError, _>::Flush(error))
                as Box<dyn std::error::Error + Send + Sync>
        })
    }

    fn land(&self, bytes: Bytes) {
        if let Ok(mut store) = self.store.lock() {
            if let Err(error) = store.put::<UnknownBlob, Bytes>(bytes) {
                tracing::warn!(?error, "reader fetch landing failed");
            }
        }
    }
}

impl<L: Clone> Clone for PeerSnapshot<L> {
    fn clone(&self) -> Self {
        Self {
            local: self.local.clone(),
            fetch: self.fetch.clone(),
        }
    }
}

impl<L: PartialEq> PartialEq for PeerSnapshot<L> {
    fn eq(&self, other: &Self) -> bool {
        self.local == other.local
    }
}

impl<L: Eq> Eq for PeerSnapshot<L> {}

impl<L> CoreStoreSnapshot for PeerSnapshot<L>
where
    L: CoreStoreSnapshot,
{
    fn changes_since(&self, previous: &Self) -> StoreChanges {
        self.local.changes_since(&previous.local)
    }
}

#[derive(Debug)]
pub enum PeerSnapshotGetError<E> {
    Conversion(E),
    Unavailable,
    WantRecord(Box<dyn std::error::Error + Send + Sync>),
}

impl<E: std::error::Error> std::fmt::Display for PeerSnapshotGetError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conversion(error) => write!(formatter, "blob conversion failed: {error}"),
            Self::Unavailable => write!(formatter, "blob unavailable"),
            Self::WantRecord(error) => write!(formatter, "blob WANT not durable: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for PeerSnapshotGetError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion(error) => Some(error),
            Self::Unavailable => None,
            Self::WantRecord(error) => Some(error.as_ref()),
        }
    }
}

impl<L> BlobStoreGet for PeerSnapshot<L>
where
    L: BlobStoreGet,
{
    type GetError<E: std::error::Error + Send + Sync + 'static> = L::GetError<E>;

    fn get<T, Sch>(
        &self,
        handle: Inline<Handle<Sch>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<Sch>>::Error>>
    where
        Sch: BlobEncoding + 'static,
        T: TryFromBlob<Sch>,
        Handle<Sch>: InlineEncoding,
    {
        self.local.get::<T, Sch>(handle)
    }
}

impl<L> BlobStoreList for PeerSnapshot<L>
where
    L: BlobStoreList,
{
    type Iter<'a>
        = L::Iter<'a>
    where
        L: 'a;
    type Err = L::Err;

    fn blobs<'a>(&'a self) -> Self::Iter<'a> {
        self.local.blobs()
    }

    fn contains_blob<Sch>(&self, handle: Inline<Handle<Sch>>) -> Result<bool, Self::Err>
    where
        Sch: BlobEncoding + 'static,
        Handle<Sch>: InlineEncoding,
    {
        self.local.contains_blob(handle)
    }
}

impl<L> BlobStoreMeta for PeerSnapshot<L>
where
    L: BlobStoreMeta,
{
    type MetaError = L::MetaError;

    fn metadata<Sch>(
        &self,
        handle: Inline<Handle<Sch>>,
    ) -> Result<Option<triblespace_core::repo::BlobMetadata>, Self::MetaError>
    where
        Sch: BlobEncoding + 'static,
        Handle<Sch>: InlineEncoding,
    {
        self.local.metadata(handle)
    }
}

impl<L> CollectionRead for PeerSnapshot<L>
where
    L: CollectionRead,
{
    type RecordsError = L::RecordsError;
    type RecordIter<'a>
        = L::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.local.records()
    }
}

impl<L> CapabilityProofRead for PeerSnapshot<L>
where
    L: CapabilityProofRead,
{
    type ProofsError = L::ProofsError;
    type ProofIter<'a>
        = L::ProofIter<'a>
    where
        Self: 'a;

    fn proofs<'a>(&'a self) -> Result<Self::ProofIter<'a>, Self::ProofsError> {
        self.local.proofs()
    }
}

impl<L> PeerRead for PeerSnapshot<L>
where
    L: PeerRead,
{
    type PeersError = L::PeersError;
    type PeerIter<'a>
        = L::PeerIter<'a>
    where
        Self: 'a;

    fn peers<'a>(&'a self) -> Result<Self::PeerIter<'a>, Self::PeersError> {
        self.local.peers()
    }
}

impl<L> BlobChildren for PeerSnapshot<L> where L: BlobStoreGet {}

impl<L> triblespace_core::repo::async_store::AsyncBlobStoreGet for PeerSnapshot<L>
where
    L: BlobStoreGet + Clone + Send + 'static,
{
    type GetError<E: std::error::Error + Send + Sync + 'static> = PeerSnapshotGetError<E>;

    fn get<T, Sch>(
        &self,
        handle: Inline<Handle<Sch>>,
    ) -> impl std::future::Future<Output = Result<T, Self::GetError<<T as TryFromBlob<Sch>>::Error>>>
    + Send
    where
        Sch: BlobEncoding + 'static,
        T: TryFromBlob<Sch>,
        Handle<Sch>: InlineEncoding,
    {
        let raw = handle.raw;
        let local = self.local.clone();
        let fetch = self.fetch.clone();
        async move {
            let bytes = if let Ok(bytes) =
                local.get::<Bytes, UnknownBlob>(Inline::<Handle<UnknownBlob>>::new(raw))
            {
                bytes
            } else if let Some(fetch) = fetch {
                fetch
                    .sink
                    .record_want(raw)
                    .map_err(PeerSnapshotGetError::WantRecord)?;
                let Some(bytes) = fetch
                    .sender
                    .fetch_blob(raw, crate::host::INTERACTIVE_FETCH_DEADLINE)
                    .await
                else {
                    return Err(PeerSnapshotGetError::Unavailable);
                };
                let bytes = Bytes::from(bytes);
                fetch.sink.land(bytes.clone());
                bytes
            } else {
                return Err(PeerSnapshotGetError::Unavailable);
            };
            triblespace_core::blob::Blob::<Sch>::new(bytes)
                .try_from_blob()
                .map_err(PeerSnapshotGetError::Conversion)
        }
    }
}
