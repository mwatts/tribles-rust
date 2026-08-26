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
use triblespace_core::collection::CollectionStore;
use triblespace_core::inline::Inline;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::lazy::WantRecordError;
use triblespace_core::repo::{
    BlobChildren, BlobStore, BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut,
    CapabilityProofStore, PeerStore, StorageFlush, StoreRevision, StoreScope, StoreScopeError,
    WantRequest, WantStore,
};

use crate::channel::{MAX_ADMISSION_BRIDGE_BATCHES, NetEvent};
use crate::host::{self, NetReceiver, NetSender, StoreSnapshot};
use crate::protocol::RawHash;

pub use crate::host::PeerConfig;
pub use crate::inventory::{BlobReconcileMode, ReconcileDirection, ReconcileQos};

/// Failure while attaching a physical store to a team-scoped network host.
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

/// A store attached to one team-scoped network host.
pub struct Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + WantStore
        + StorageFlush
        + StoreRevision
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
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
    last_store_revision: Option<S::Revision>,
    last_event_at: crate::clock::Mono,
}

impl<S> Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + StoreScope
        + WantStore
        + StorageFlush
        + StoreRevision
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
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
        match store.store_scope().map_err(PeerOpenError::Scope)? {
            None => Err(PeerOpenError::Unbound),
            Some(bound) if bound == requested => Ok(()),
            Some(bound) => Err(PeerOpenError::TeamMismatch { bound, requested }),
        }
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
            last_store_revision: None,
            last_event_at: crate::clock::mono_now(),
        };
        // Reobserve once after assembly so external pile appends that raced
        // construction are included before the first scheduler sweep.
        peer.refresh_checked()?;
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

    /// Fetch exact content through configured and authenticated PEER routes.
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

    /// Drain authenticated inventory progress, cross one durability barrier,
    /// then replace the immutable snapshot. Calling this with no events is
    /// still meaningful: file-backed stores reobserve external appends before
    /// manifests and periodic sweeps use them.
    pub fn refresh(&mut self) {
        if let Err(error) = self.refresh_checked() {
            tracing::warn!(%error, "network store scope invalid; clearing serving view");
            self.sender.clear_snapshot();
            // A transient scope-observation failure must not strand the peer
            // with no snapshot merely because the sync-visible revision did
            // not change before the next successful refresh.
            self.last_store_revision = None;
        }
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
            let revision = match store.store_revision() {
                Ok(revision) => revision,
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        "store revision unavailable; keeping prior inventory"
                    );
                    return Ok(());
                }
            };
            // `store_revision` may itself reobserve an external append. Scope
            // is intentionally absent from that sync-visible token, so check
            // it again before an equality fast-path can retain a serving view.
            Self::validate_store_scope(&mut *store, self.team)?;
            if self.last_store_revision.as_ref() == Some(&revision) {
                return Ok(());
            }
            if Self::install_validated_snapshot(&self.sender, &mut *store, self.team)? {
                self.last_store_revision = Some(revision);
            }
        }
        Ok(())
    }

    fn install_validated_snapshot(
        sender: &NetSender,
        store: &mut S,
        team: VerifyingKey,
    ) -> Result<bool, PeerOpenError<S::ScopeError>> {
        Self::validate_store_scope(store, team)?;
        let snapshot = match StoreSnapshot::from_store(store, team) {
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
                panic!("Peer::into_store: an outstanding PeerReader still shares the store")
            })
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn try_local(&mut self, hash: RawHash) -> Option<Bytes> {
        self.reader()
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
        + StoreScope
        + WantStore
        + StorageFlush
        + StoreRevision
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
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

impl<S> BlobStore for Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + StoreScope
        + WantStore
        + StorageFlush
        + StoreRevision
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
{
    type Reader = PeerReader<S::Reader>;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.refresh();
        let local = self.store.lock().expect("store mutex").reader()?;
        let fetch = Some(FetchCap {
            sender: self.sender.clone(),
            sink: Arc::new(SharedStore(self.store.clone())),
        });
        Ok(PeerReader { local, fetch })
    }
}

/// A frozen local reader plus an optional exact-fetch capability.
pub struct PeerReader<L> {
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

struct SharedStore<S>(Arc<Mutex<S>>);

impl<S> StoreSink for SharedStore<S>
where
    S: BlobStorePut + WantStore + StorageFlush + Send + 'static,
{
    fn record_want(&self, hash: RawHash) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut store = self.0.lock().expect("store mutex");
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
        if let Ok(mut store) = self.0.lock() {
            if let Err(error) = store.put::<UnknownBlob, Bytes>(bytes) {
                tracing::warn!(?error, "reader fetch landing failed");
            }
        }
    }
}

impl<L: Clone> Clone for PeerReader<L> {
    fn clone(&self) -> Self {
        Self {
            local: self.local.clone(),
            fetch: self.fetch.clone(),
        }
    }
}

impl<L: PartialEq> PartialEq for PeerReader<L> {
    fn eq(&self, other: &Self) -> bool {
        self.local == other.local
    }
}

impl<L: Eq> Eq for PeerReader<L> {}

#[derive(Debug)]
pub enum PeerReaderGetError<E> {
    Conversion(E),
    Unavailable,
    WantRecord(Box<dyn std::error::Error + Send + Sync>),
}

impl<E: std::error::Error> std::fmt::Display for PeerReaderGetError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conversion(error) => write!(formatter, "blob conversion failed: {error}"),
            Self::Unavailable => write!(formatter, "blob unavailable"),
            Self::WantRecord(error) => write!(formatter, "blob WANT not durable: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for PeerReaderGetError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion(error) => Some(error),
            Self::Unavailable => None,
            Self::WantRecord(error) => Some(error.as_ref()),
        }
    }
}

impl<L> BlobStoreGet for PeerReader<L>
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

impl<L> BlobStoreList for PeerReader<L>
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

impl<L> BlobChildren for PeerReader<L> where L: BlobStoreGet {}

impl<L> triblespace_core::repo::async_store::AsyncBlobStoreGet for PeerReader<L>
where
    L: BlobStoreGet + Clone + Send + 'static,
{
    type GetError<E: std::error::Error + Send + Sync + 'static> = PeerReaderGetError<E>;

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
                    .map_err(PeerReaderGetError::WantRecord)?;
                let Some(bytes) = fetch
                    .sender
                    .fetch_blob(raw, crate::host::INTERACTIVE_FETCH_DEADLINE)
                    .await
                else {
                    return Err(PeerReaderGetError::Unavailable);
                };
                let bytes = Bytes::from(bytes);
                fetch.sink.land(bytes.clone());
                bytes
            } else {
                return Err(PeerReaderGetError::Unavailable);
            };
            triblespace_core::blob::Blob::<Sch>::new(bytes)
                .try_from_blob()
                .map_err(PeerReaderGetError::Conversion)
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
    fn refresh_withdraws_snapshot_after_external_scope_conflict() {
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
        peer.refresh();

        assert!(
            !snapshot_probe.snapshot_available(),
            "a peer must stop serving after reobserving conflicting store scopes"
        );
    }
}
