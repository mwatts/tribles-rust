//! A synchronous store wrapped in authorized team anti-entropy.
//!
//! The host runtime compares immutable four-component inventories. This side
//! owns the only mutable store boundary: authenticated leaves are deduplicated,
//! inserted monotonically, flushed once per drain, and only then exposed in a
//! replacement serving snapshot. Exact blob reads keep their durable-WANT
//! semantics independently of broad inventory mirroring.

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
    CapabilityProofStore, PeerStore, StorageFlush, WantRequest, WantStore,
};

use crate::channel::NetEvent;
use crate::host::{self, NetReceiver, NetSender};
use crate::protocol::RawHash;

pub use crate::host::PeerConfig;
pub use crate::inventory::{BlobReconcileMode, ReconcileDirection, ReconcileQos};

/// A store attached to one team-scoped network host.
pub struct Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + WantStore
        + StorageFlush
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
    last_event_at: crate::clock::Mono,
}

impl<S> Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
{
    /// Spawn a production host and attach `store` to exactly `config.team`.
    pub fn new(store: S, key: SigningKey, config: PeerConfig) -> Self {
        let team = config.team;
        let qos = config.qos;
        let (sender, receiver) = host::spawn(key, config);
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
    ) -> Self {
        Self::assemble(store, team, qos, sender, receiver)
    }

    fn assemble(
        mut store: S,
        team: VerifyingKey,
        qos: ReconcileQos,
        sender: NetSender,
        receiver: NetReceiver,
    ) -> Self {
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
            last_event_at: crate::clock::mono_now(),
        };
        // Reobserve once after assembly so external pile appends that raced
        // construction are included before the first scheduler sweep.
        peer.refresh();
        peer
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

    /// Fetch exact content through configured/learned routes and then DHT
    /// providers. This primitive neither records a WANT nor mutates the store.
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
        let mut incoming = Vec::new();
        while let Some(event) = self.receiver.try_recv() {
            self.last_event_at = crate::clock::mono_now();
            incoming.push(event);
        }

        let received = incoming.len();
        let mut store = self.store.lock().expect("store mutex");
        let mut admitted = false;
        for event in incoming {
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
        self.pending_network_flush |= admitted;
        if self.pending_network_flush {
            match store.flush() {
                Ok(()) => {
                    self.pending_network_flush = false;
                    tracing::debug!(received, "inventory admission batch durable");
                }
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        received,
                        "inventory admission flush failed; snapshot withheld"
                    );
                }
            }
        }
        if !self.pending_network_flush {
            self.sender.refresh_store_snapshot(&mut *store, self.team);
        }
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
        + WantStore
        + StorageFlush
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
        let handle = store.put(item)?;
        self.sender.refresh_store_snapshot(&mut *store, self.team);
        Ok(handle)
    }
}

impl<S> BlobStore for Peer<S>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + WantStore
        + StorageFlush
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
    use triblespace_core::repo::memoryrepo::MemoryRepo;

    #[test]
    fn simulation_wiring_never_infers_team_from_endpoint_identity() {
        let endpoint = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let team = SigningKey::from_bytes(&[2; 32]).verifying_key();
        assert_ne!(endpoint, team);
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let peer = Peer::with_wiring(
            MemoryRepo::default(),
            team,
            ReconcileQos::default(),
            sender,
            receiver,
        );
        assert_eq!(peer.team(), team);
        assert_eq!(peer.id(), endpoint_id);
    }
}
