//! A synchronous store wrapped in authorized team anti-entropy.
//!
//! The host runtime compares immutable four-component inventories. This side
//! owns the only mutable store boundary: authenticated leaves are deduplicated,
//! inserted monotonically, flushed once per drain, and only then exposed in a
//! replacement serving snapshot. Exact blob reads keep their durable-WANT
//! semantics independently of broad inventory mirroring.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use anybytes::Bytes;
use ed25519_dalek::{SigningKey, VerifyingKey};
use iroh_base::EndpointId;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::{BlobEncoding, IntoBlob, TryFromBlob};
use triblespace_core::collection::{CollectionRead, CollectionStore};
use triblespace_core::inline::Inline;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::lazy::WantRecordError;
use triblespace_core::repo::{
    ArtifactOfferSnapshot, ArtifactOfferStore, BlobChildren, BlobStore, BlobStoreGet,
    BlobStoreList, BlobStoreMeta, BlobStorePut, CapabilityProofRead, CapabilityProofStore,
    PeerRead, PeerStore, SnapshotSource, StorageFlush, StoreChanges, StoreRead, StoreScope,
    StoreScopeError, StoreSnapshot as CoreStoreSnapshot, WantRequest, WantStore,
};

use crate::channel::{MAX_ADMISSION_BRIDGE_BATCHES, NetEvent};
use crate::host::{self, NetReceiver, NetSender, StoreSnapshot};
use crate::protocol::RawHash;
use crate::provider::ArtifactId;

pub use crate::host::PeerConfig;
pub use crate::inventory::{BlobReconcileMode, ReconcileDirection, ReconcileQos};

/// Failure while attaching or refreshing a physical store against a
/// team-scoped network host.
#[derive(Debug)]
pub enum PeerOpenError<E> {
    /// The store's scope assertion could not be observed coherently.
    Scope(StoreScopeError<E>),
    /// The store has not yet been explicitly bound to any team.
    Unbound,
    /// The store is bound coherently, but to a different team.
    TeamMismatch {
        /// Team asserted by the physical store.
        bound: VerifyingKey,
        /// Team requested by the peer configuration.
        requested: VerifyingKey,
    },
    /// The production network thread, runtime, or iroh endpoint could not start.
    HostStartup(anyhow::Error),
}

impl<E: fmt::Display> fmt::Display for PeerOpenError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scope(error) => write!(f, "cannot observe network store scope: {error}"),
            Self::Unbound => f.write_str("network store is not explicitly bound to a team"),
            Self::TeamMismatch { bound, requested } => write!(
                f,
                "network store is bound to team {}, not requested team {}",
                hex::encode_upper(bound.as_bytes()),
                hex::encode_upper(requested.as_bytes()),
            ),
            Self::HostStartup(error) => write!(f, "cannot start network host: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for PeerOpenError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Scope(error) => Some(error),
            Self::Unbound | Self::TeamMismatch { .. } => None,
            Self::HostStartup(error) => Some(error.as_ref()),
        }
    }
}

/// Failure while freezing the local observation behind a [`Peer`].
#[derive(Debug)]
pub enum PeerSnapshotError<SnapshotError, ScopeError> {
    /// The backing store could not freeze its coherent observation.
    Store(SnapshotError),
    /// The physical store is no longer valid for this peer's team.
    Scope(PeerOpenError<ScopeError>),
}

impl<SnapshotError, ScopeError> fmt::Display for PeerSnapshotError<SnapshotError, ScopeError>
where
    SnapshotError: fmt::Display,
    ScopeError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "cannot freeze peer store snapshot: {error}"),
            Self::Scope(error) => error.fmt(formatter),
        }
    }
}

impl<SnapshotError, ScopeError> Error for PeerSnapshotError<SnapshotError, ScopeError>
where
    SnapshotError: Error + 'static,
    ScopeError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Scope(error) => Some(error),
        }
    }
}

fn validate_scope<S>(
    store: &mut S,
    requested: VerifyingKey,
) -> Result<(), PeerOpenError<S::ScopeError>>
where
    S: StoreScope,
{
    match store.store_scope().map_err(PeerOpenError::Scope)? {
        None => Err(PeerOpenError::Unbound),
        Some(bound) if bound == requested => Ok(()),
        Some(bound) => Err(PeerOpenError::TeamMismatch { bound, requested }),
    }
}

/// A store attached to one team-scoped network host.
pub struct Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + ArtifactOfferStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobStoreMeta,
{
    store: Arc<Mutex<S>>,
    sender: NetSender,
    receiver: NetReceiver,
    team: VerifyingKey,
    qos: ReconcileQos,
    /// Network admissions stay outside the advertised snapshot until their
    /// shared durability barrier succeeds. A failed flush is retried on every
    /// refresh without requiring the remote to redeliver the event first.
    pending_network_flush: bool,
    /// Last local observation used to build the installed immutable inventory.
    /// Equality is a cheap invalidation check supplied by the store; it is not
    /// a portable generation or a semantic version.
    last_store_snapshot: Option<S::Snapshot>,
    /// Last durable local publication policy sent to the host. Offers are
    /// deliberately observed independently of the four-component inventory
    /// revision because changing them must not rebuild semantic inventory.
    last_artifact_offers: Option<ArtifactOfferSnapshot>,
    last_event_at: crate::clock::Mono,
}

impl<S> Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + ArtifactOfferStore
        + StoreScope
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobStoreMeta,
{
    /// Spawn a production host and attach `store` to exactly `config.team`.
    pub fn new(
        mut store: S,
        key: SigningKey,
        config: PeerConfig,
    ) -> Result<Self, PeerOpenError<S::ScopeError>> {
        let team = config.team;
        let qos = config.qos;
        Self::validate_store_scope(&mut store, team)?;
        let (sender, receiver) = host::spawn(key, config).map_err(PeerOpenError::HostStartup)?;
        Self::assemble(store, team, qos, sender, receiver)
    }

    /// Attach a store to a caller-owned host, most commonly the deterministic
    /// simulator. Team and QoS are explicit because the endpoint transport key
    /// is intentionally not the team trust root.
    pub fn with_wiring(
        store: S,
        team: VerifyingKey,
        qos: ReconcileQos,
        sender: NetSender,
        receiver: NetReceiver,
    ) -> Result<Self, PeerOpenError<S::ScopeError>> {
        let mut store = store;
        Self::validate_store_scope(&mut store, team)?;
        Self::assemble(store, team, qos, sender, receiver)
    }

    fn validate_store_scope(
        store: &mut S,
        requested: VerifyingKey,
    ) -> Result<(), PeerOpenError<S::ScopeError>> {
        validate_scope(store, requested)
    }

    fn assemble(
        mut store: S,
        team: VerifyingKey,
        qos: ReconcileQos,
        sender: NetSender,
        receiver: NetReceiver,
    ) -> Result<Self, PeerOpenError<S::ScopeError>> {
        let endpoint = VerifyingKey::from_bytes(sender.id().as_bytes())
            .expect("an endpoint id is an Ed25519 public key");
        let pending_network_flush = match store.insert_peer(
            triblespace_core::repo::peer::PeerEvidence::new(team, endpoint),
        ) {
            Ok(()) => match store.flush() {
                Ok(()) => false,
                Err(error) => {
                    tracing::warn!(?error, "initial self PEER evidence flush failed");
                    true
                }
            },
            Err(error) => {
                tracing::warn!(?error, "initial self PEER evidence admission failed");
                false
            }
        };
        let mut peer = Self {
            store: Arc::new(Mutex::new(store)),
            sender,
            receiver,
            team,
            qos,
            pending_network_flush,
            last_store_snapshot: None,
            last_artifact_offers: None,
            last_event_at: crate::clock::mono_now(),
        };
        // Reobserve once after assembly so external pile appends that raced
        // construction are included before the first scheduler sweep.
        peer.try_refresh()?;
        Ok(peer)
    }

    pub fn id(&self) -> EndpointId {
        self.sender.id()
    }

    pub const fn team(&self) -> VerifyingKey {
        self.team
    }

    pub const fn qos(&self) -> ReconcileQos {
        self.qos
    }

    pub fn last_event_at(&self) -> crate::clock::Mono {
        self.last_event_at
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

    /// Drain authenticated inventory progress, cross one durability barrier,
    /// then replace the immutable snapshot. Calling this with no events is
    /// still meaningful: file-backed stores reobserve external appends before
    /// manifests and periodic sweeps use them.
    pub fn refresh(&mut self) {
        if let Err(error) = self.try_refresh() {
            tracing::warn!(%error, "network store scope invalid; clearing serving view");
        }
    }

    /// Drain pending network evidence and publish one checked store snapshot.
    ///
    /// Unlike [`Self::refresh`], this reports scope failures to callers that
    /// must not continue operating on a physically conflicted store. Both
    /// surfaces perform the same fail-closed cleanup before returning.
    pub fn try_refresh(&mut self) -> Result<(), PeerOpenError<S::ScopeError>> {
        let result = self.refresh_checked();
        if result.is_err() {
            self.sender.clear_snapshot();
            // A transient scope-observation failure must not strand the peer
            // with no snapshot merely because the sync-visible revision did
            // not change before the next successful refresh.
            self.last_store_snapshot = None;
        }
        result
    }

    fn refresh_checked(&mut self) -> Result<(), PeerOpenError<S::ScopeError>> {
        // Scope is physical-store state, not inventory. Reobserve it before
        // accepting events so a conflicting external append fails closed on
        // the next scheduler/reader refresh instead of extending that store.
        {
            let mut store = self.store.lock().expect("store mutex");
            Self::validate_store_scope(&mut *store, self.team)?;
        }

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
                    NetEvent::Peer(evidence) => match store.insert_peer(evidence) {
                        Ok(()) => admitted = true,
                        Err(error) => {
                            tracing::warn!(?error, "admitting PEER inventory evidence failed")
                        }
                    },
                    NetEvent::CollectionRecord(record) => match store.insert(record) {
                        Ok(()) => admitted = true,
                        Err(error) => tracing::warn!(
                            ?error,
                            "admitting collection-record inventory evidence failed"
                        ),
                    },
                    NetEvent::CapabilityProof(proof) => match store.insert_proof(proof) {
                        Ok(()) => admitted = true,
                        Err(error) => tracing::warn!(
                            ?error,
                            "admitting capability-proof inventory evidence failed"
                        ),
                    },
                    NetEvent::Blob { hash, bytes } => {
                        if blake3::hash(&bytes).as_bytes() != &hash {
                            tracing::warn!(hash = %hex::encode(&hash[..4]), "discarding hash-invalid mirror blob");
                            continue;
                        }
                        match store.put::<UnknownBlob, Bytes>(bytes) {
                            Ok(handle) if handle.raw == hash => admitted = true,
                            Ok(_) => tracing::warn!(
                                "blob store returned a handle different from verified bytes"
                            ),
                            Err(error) => {
                                tracing::warn!(?error, "landing mirror inventory blob failed")
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
                        "inventory admission drain durable"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        received,
                        received_batches,
                        "inventory admission flush failed; snapshot withheld"
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
                        "store snapshot unavailable; keeping prior inventory"
                    );
                    // Offer observation is an independent local policy lane.
                    // Keep the previous semantic inventory, but do not skip a
                    // newly appended offer merely because its unrelated
                    // invalidation token is temporarily unavailable.
                    Self::observe_artifact_offers(
                        &self.sender,
                        self.team,
                        &mut self.last_artifact_offers,
                        &mut *store,
                    )?;
                    return Ok(());
                }
            };
            // `snapshot` may itself reobserve an external append. Scope
            // is intentionally absent from that sync-visible token, so check
            // it again before an equality fast-path can retain a serving view.
            Self::validate_store_scope(&mut *store, self.team)?;
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
            // Even a semantic no-op installs the fresh read lease. The host
            // reuses unchanged PATCH inventories while replacing the Blob
            // reader, so pile remaps and reclaimed generations are not pinned
            // forever merely because their content sets stayed identical.
            if Self::install_validated_snapshot(
                &self.sender,
                &mut *store,
                snapshot.clone(),
                self.team,
                previous_snapshot.as_deref(),
                changes,
            )? {
                self.last_store_snapshot = Some(snapshot);
            }
        }
        Self::observe_artifact_offers(
            &self.sender,
            self.team,
            &mut self.last_artifact_offers,
            &mut *store,
        )?;
        Ok(())
    }

    fn observe_artifact_offers(
        sender: &NetSender,
        team: VerifyingKey,
        last_artifact_offers: &mut Option<ArtifactOfferSnapshot>,
        store: &mut S,
    ) -> Result<(), PeerOpenError<S::ScopeError>> {
        let offers = match store.offers_snapshot() {
            Ok(offers) => offers,
            Err(error) => {
                // The failed observation may still have re-read externally
                // appended records. Recheck scope before retaining the prior
                // host view; a policy read failure must not mask a newly
                // visible cross-team assertion.
                Self::validate_store_scope(store, team)?;
                // Offers are grow-only. Retaining the last coherent snapshot
                // across a transient observation failure is conservative and
                // cannot resurrect retracted intent.
                tracing::warn!(?error, "artifact-offer snapshot unavailable");
                return Ok(());
            }
        };
        // File-backed offer observation may itself reobserve an externally
        // appended scope conflict. Never hand policy to the team-scoped host
        // before checking that boundary again.
        Self::validate_store_scope(store, team)?;
        if last_artifact_offers.as_ref() != Some(&offers) {
            sender.update_artifact_offers(offers.clone());
            *last_artifact_offers = Some(offers);
        }
        Ok(())
    }

    fn install_validated_snapshot(
        sender: &NetSender,
        store: &mut S,
        store_snapshot: S::Snapshot,
        team: VerifyingKey,
        previous: Option<&StoreSnapshot>,
        changes: StoreChanges,
    ) -> Result<bool, PeerOpenError<S::ScopeError>> {
        Self::validate_store_scope(store, team)?;
        let snapshot = match StoreSnapshot::from_store_changes(
            store_snapshot,
            team,
            previous,
            changes,
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tracing::warn!(%error, "store inventory snapshot unavailable; clearing serving view");
                sender.clear_snapshot();
                return Ok(false);
            }
        };
        // Snapshot construction reobserves every externally appendable store
        // component. Reobserve scope once more before publication so a scope
        // record racing that construction cannot authorize a mixed snapshot.
        Self::validate_store_scope(store, team)?;
        sender.update_snapshot(snapshot);
        Ok(true)
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
        + PeerStore
        + ArtifactOfferStore
        + StoreScope
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobStoreMeta,
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
        + PeerStore
        + ArtifactOfferStore
        + StoreScope
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobStoreMeta,
{
    type Snapshot = PeerSnapshot<S::Snapshot>;
    type SnapshotError = PeerSnapshotError<S::SnapshotError, S::ScopeError>;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        // Unlike the scheduler-oriented `refresh`, freezing a caller-visible
        // snapshot must report a scope conflict rather than merely withdrawing
        // the serving inventory and continuing.
        self.try_refresh().map_err(PeerSnapshotError::Scope)?;
        let mut store = self.store.lock().expect("store mutex");
        validate_scope(&mut *store, self.team).map_err(PeerSnapshotError::Scope)?;
        let local = store.snapshot().map_err(PeerSnapshotError::Store)?;
        // A file-backed snapshot may itself reobserve an externally appended
        // scope record. Check again before minting the lazy fetch capability.
        validate_scope(&mut *store, self.team).map_err(PeerSnapshotError::Scope)?;
        drop(store);
        let fetch = Some(FetchCap {
            sender: self.sender.clone(),
            sink: Arc::new(SharedStore {
                store: self.store.clone(),
                team: self.team,
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
    team: VerifyingKey,
}

impl<S> StoreSink for SharedStore<S>
where
    S: BlobStorePut + WantStore + StorageFlush + StoreScope + Send + 'static,
{
    fn record_want(&self, hash: RawHash) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut store = self.store.lock().expect("store mutex");
        validate_scope(&mut *store, self.team)
            .map_err(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync>)?;
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
            if let Err(error) = validate_scope(&mut *store, self.team) {
                tracing::warn!(%error, "refusing to land a fetched blob into an invalidly scoped store");
                return;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use triblespace_core::repo::StoreScope;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::pile::Pile;

    #[test]
    fn simulation_wiring_never_infers_team_from_endpoint_identity() {
        let endpoint = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let team = SigningKey::from_bytes(&[2; 32]).verifying_key();
        assert_ne!(endpoint, team);
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let mut store = MemoryRepo::default();
        store.bind_store_scope(team).unwrap();
        let peer =
            Peer::with_wiring(store, team, ReconcileQos::default(), sender, receiver).unwrap();
        assert_eq!(peer.team(), team);
        assert_eq!(peer.id(), endpoint_id);
    }

    #[test]
    fn semantic_noop_refresh_still_replaces_the_host_read_lease() {
        let endpoint = SigningKey::from_bytes(&[17; 32]).verifying_key();
        let team = SigningKey::from_bytes(&[18; 32]).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let snapshot_probe = sender.clone();
        let mut store = MemoryRepo::default();
        store.bind_store_scope(team).unwrap();
        let mut peer =
            Peer::with_wiring(store, team, ReconcileQos::default(), sender, receiver).unwrap();

        let first = snapshot_probe.current_snapshot().unwrap();
        peer.try_refresh().unwrap();
        let second = snapshot_probe.current_snapshot().unwrap();
        assert!(
            !std::sync::Arc::ptr_eq(&first, &second),
            "a fresh backend observation must replace the host reader even when every semantic PATCH is unchanged",
        );
    }

    #[test]
    fn simulation_wiring_refuses_unbound_store() {
        let endpoint = SigningKey::from_bytes(&[3; 32]).verifying_key();
        let team = SigningKey::from_bytes(&[4; 32]).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let error = match Peer::with_wiring(
            MemoryRepo::default(),
            team,
            ReconcileQos::default(),
            sender,
            receiver,
        ) {
            Ok(_) => panic!("unbound store unexpectedly reached peer assembly"),
            Err(error) => error,
        };
        assert!(matches!(error, PeerOpenError::Unbound));
    }

    #[test]
    fn simulation_wiring_refuses_store_bound_to_another_team() {
        let endpoint = SigningKey::from_bytes(&[5; 32]).verifying_key();
        let bound = SigningKey::from_bytes(&[6; 32]).verifying_key();
        let requested = SigningKey::from_bytes(&[7; 32]).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let mut store = MemoryRepo::default();
        store.bind_store_scope(bound).unwrap();
        let error =
            match Peer::with_wiring(store, requested, ReconcileQos::default(), sender, receiver) {
                Ok(_) => panic!("wrong-team store unexpectedly reached peer assembly"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            PeerOpenError::TeamMismatch {
                bound: actual_bound,
                requested: actual_requested,
            } if actual_bound == bound && actual_requested == requested
        ));
    }

    #[test]
    fn simulation_wiring_refuses_concatenated_conflicting_pile_scopes() {
        let dir = tempfile::tempdir().unwrap();
        let left_path = dir.path().join("left.pile");
        let right_path = dir.path().join("right.pile");
        std::fs::File::create(&left_path).unwrap();
        std::fs::File::create(&right_path).unwrap();
        let left_team = SigningKey::from_bytes(&[8; 32]).verifying_key();
        let right_team = SigningKey::from_bytes(&[9; 32]).verifying_key();

        let mut left = Pile::open(&left_path).unwrap();
        left.bind_store_scope(left_team).unwrap();
        left.close().unwrap();
        let mut right = Pile::open(&right_path).unwrap();
        right.bind_store_scope(right_team).unwrap();
        right.close().unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&left_path)
            .unwrap()
            .write_all(&std::fs::read(&right_path).unwrap())
            .unwrap();

        let endpoint = SigningKey::from_bytes(&[10; 32]).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let store = Pile::open(&left_path).unwrap();
        let error =
            match Peer::with_wiring(store, left_team, ReconcileQos::default(), sender, receiver) {
                Ok(_) => panic!("conflicting pile scopes unexpectedly reached peer assembly"),
                Err(error) => error,
            };
        assert!(matches!(
            error,
            PeerOpenError::Scope(StoreScopeError::Conflict { .. })
        ));
    }

    #[test]
    fn checked_refresh_reports_and_withdraws_external_scope_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let serving_path = dir.path().join("serving.pile");
        let conflicting_path = dir.path().join("conflicting.pile");
        std::fs::File::create(&serving_path).unwrap();
        std::fs::File::create(&conflicting_path).unwrap();
        let serving_team = SigningKey::from_bytes(&[11; 32]).verifying_key();
        let conflicting_team = SigningKey::from_bytes(&[12; 32]).verifying_key();

        let mut serving = Pile::open(&serving_path).unwrap();
        serving.bind_store_scope(serving_team).unwrap();
        let mut conflicting = Pile::open(&conflicting_path).unwrap();
        conflicting.bind_store_scope(conflicting_team).unwrap();
        conflicting.close().unwrap();

        let endpoint = SigningKey::from_bytes(&[13; 32]).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let snapshot_probe = sender.clone();
        let mut peer = Peer::with_wiring(
            serving,
            serving_team,
            ReconcileQos::default(),
            sender,
            receiver,
        )
        .unwrap();
        assert!(snapshot_probe.snapshot_available());

        std::fs::OpenOptions::new()
            .append(true)
            .open(&serving_path)
            .unwrap()
            .write_all(&std::fs::read(&conflicting_path).unwrap())
            .unwrap();
        let error = peer
            .try_refresh()
            .expect_err("checked refresh must report the external scope conflict");

        assert!(
            matches!(
                error,
                PeerOpenError::Scope(StoreScopeError::Conflict { .. })
            ),
            "checked refresh must preserve the exact fail-closed cause",
        );
        assert!(
            !snapshot_probe.snapshot_available(),
            "a peer must stop serving after reobserving conflicting store scopes"
        );
    }

    #[test]
    fn peer_snapshots_fail_closed_after_an_external_scope_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let serving_path = dir.path().join("serving.pile");
        let conflicting_path = dir.path().join("conflicting.pile");
        std::fs::File::create(&serving_path).unwrap();
        std::fs::File::create(&conflicting_path).unwrap();
        let serving_team = SigningKey::from_bytes(&[14; 32]).verifying_key();
        let conflicting_team = SigningKey::from_bytes(&[15; 32]).verifying_key();

        let mut serving = Pile::open(&serving_path).unwrap();
        serving.bind_store_scope(serving_team).unwrap();
        let mut conflicting = Pile::open(&conflicting_path).unwrap();
        conflicting.bind_store_scope(conflicting_team).unwrap();
        conflicting.close().unwrap();

        let endpoint = SigningKey::from_bytes(&[16; 32]).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let mut peer = Peer::with_wiring(
            serving,
            serving_team,
            ReconcileQos::default(),
            sender,
            receiver,
        )
        .unwrap();
        let old_snapshot = peer.snapshot().unwrap();

        std::fs::OpenOptions::new()
            .append(true)
            .open(&serving_path)
            .unwrap()
            .write_all(&std::fs::read(&conflicting_path).unwrap())
            .unwrap();
        let conflicted_len = std::fs::metadata(&serving_path).unwrap().len();

        let fetch = old_snapshot
            .fetch
            .as_ref()
            .expect("peer snapshots carry a lazy fetch capability");
        fetch
            .sink
            .record_want([17; 32])
            .expect_err("an old snapshot must not mutate a newly conflicted store");
        assert_eq!(
            std::fs::metadata(&serving_path).unwrap().len(),
            conflicted_len,
            "scope rejection must happen before the WANT append",
        );

        let error = match peer.snapshot() {
            Ok(_) => panic!("a new snapshot hid the external scope conflict"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            PeerSnapshotError::Scope(PeerOpenError::Scope(StoreScopeError::Conflict { .. }))
        ));
        drop(old_snapshot);
        peer.into_store().close().unwrap();
    }
}
