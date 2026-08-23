//! `Peer<S>`: a store wrapped in distributed network sync.
//!
//! Owns the inner store, spawns the iroh network thread on construction,
//! and exposes the standard blob storage traits with network behavior built
//! in:
//!
//! - **Reads** auto-call [`refresh`](Peer::refresh), which drains pending
//!   incoming gossip events into the wrapped store and re-publishes any
//!   deltas from external writers (e.g. another process appended to the
//!   same pile file). Mirrors `Pile::refresh` — the explicit method is
//!   available for tight loops, but normal storage use Just Works.
//! - **Publication** diffs [`CollectionStore`] state and gossips each
//!   strictly verified commit whose collection descriptor says the collection
//!   travels, as inert signed evidence. Receivers may store and relay that
//!   evidence without fetching its referenced blobs; semantic trust belongs to
//!   later collection resolution.
//! - **Blob writes** delegate to the inner store and announce content to the
//!   DHT. CONNECT authority gates the direct-RPC channel; collection
//!   descriptors and WANTs independently govern publication and local demand.
//!
//! There is no separate cache tier: `Peer<S>` takes a **single store**,
//! and any tiering (bounded want retention, generational eviction) lives
//! in `S` — e.g. a [`Yard`](triblespace_core::repo::yard::Yard). Read-miss
//! swarm fetches land in `S` under a **want** ([`WantStore`]),
//! independently of authentication state. The want is
//! recorded durably *before* the fetch — asserted AND
//! flushed ([`StorageFlush`]), so the marker survives an immediate
//! process exit — the demand IS the want-signal (a sync daemon's work
//! queue), then the retention marker for the fetched blob, then the
//! eviction target. A failed fetch leaves the want in place: it
//! remains an outstanding want. The want-on-record invariant holds
//! unconditionally: if the want or its flush FAILS, the read errors out
//! ([`PeerReaderGetError::WantRecord`] /
//! [`Peer::get_or_fetch_async`]'s `Err`) instead of proceeding — the
//! caller never observes a fetch whose demand isn't durably recorded.
//! "Promote to durable" is not an operation; the Peer performs no hidden
//! publication or retention transition.
//!
//! Collection discovery is gossip-driven: immutable signed commit evidence
//! floods the team topic and arrives through `NetEvent`. Referenced content
//! remains independently retrievable by hash, normally through durable wants.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard};

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::{BlobEncoding, IntoBlob, TryFromBlob};
use triblespace_core::collection::{CollectionRecord, CollectionStore};
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::lazy::WantRecordError;
use triblespace_core::repo::{
    BlobChildren, BlobStore, BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut,
    StorageFlush, WantRequest, WantStore,
};

use crate::channel::NetEvent;
use crate::collection_sync::{
    IncomingBatchCounts, IncomingBatchValidationError, prepare_incoming_collection_batch,
};
use crate::collection_wire::relayable_commits;
use crate::host::{self, NetReceiver, NetSender};
use crate::protocol::RawHash;
use triblespace_core::collection::CollectionCommit;

pub use crate::host::{PeerConfig, SyncDirection};

/// Summary of one exact direct collection reconciliation.
pub type CollectionReconcileOutcome = IncomingBatchCounts;

/// Failure while fetching, verifying, authorizing, or durably admitting one
/// direct collection reconciliation.
#[derive(Debug, thiserror::Error)]
pub enum CollectionReconcileError {
    /// Authenticated evidence transfer failed.
    #[error("collection fetch failed: {0}")]
    Fetch(#[source] anyhow::Error),
    /// Cryptographic or collection-semantic validation failed before mutation.
    #[error("incoming collection validation failed: {0}")]
    Validation(#[source] IncomingBatchValidationError),
    /// Caller policy or destination storage rejected admission.
    #[error("incoming collection admission failed: {0}")]
    Admission(#[source] anyhow::Error),
}

/// Materialize every commit this node may pass on: strictly verified, and in
/// a collection whose own descriptor says it travels. A top-level enumeration
/// failure preserves the caller's prior diff baseline; malformed individual
/// rows are inert, and a descriptor this node cannot resolve is a refusal.
fn relayable_collection_evidence<S>(store: &mut S) -> Option<Vec<CollectionCommit>>
where
    S: BlobStore + CollectionStore,
{
    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    let records: Vec<CollectionRecord> = store.records().ok()?.filter_map(Result::ok).collect();
    let reader = store.reader().ok()?;
    Some(relayable_commits(&records, |handle| {
        reader
            .get::<triblespace_core::trible::TribleSet, SimpleArchive>(handle)
            .ok()
    }))
}

/// Canonicalize, strictly verify, and durably admit one gossip drain.
///
/// Duplicate delivery is normal for gossip, so commit ids are reduced before
/// the strict batch boundary. The resulting batch performs exactly one flush.
fn admit_incoming_collection_evidence<S>(
    store: &mut S,
    mut evidence: Vec<CollectionCommit>,
) -> anyhow::Result<IncomingBatchCounts>
where
    S: CollectionStore + StorageFlush,
{
    evidence.sort_by_key(|commit| commit.id());
    evidence.dedup_by_key(|commit| commit.id());
    let prepared = prepare_incoming_collection_batch(evidence)?;
    let authorized = prepared
        .authorize(|_| Ok::<bool, std::convert::Infallible>(true))
        .expect("infallible evidence-storage policy");
    authorized.admit(store).map_err(anyhow::Error::new)
}

/// Whether local direction policy admits this incoming event.
///
fn accepts_incoming_event(direction: SyncDirection, event: &NetEvent) -> bool {
    match event {
        NetEvent::CollectionEvidence(_) => direction != SyncDirection::WriteOnly,
    }
}

/// A store wrapped in distributed network sync.
///
/// See the [module-level docs](self) for the full mental model.
///
/// # Example
///
/// Construction requires the team root and a prebuilt
/// [`AuthorityProof`](triblespace_core::authority::AuthorityProof) whose leaf
/// authorizes this peer's transport key to invoke
/// [`ACTION_CONNECT`](crate::protocol::ACTION_CONNECT). Proof selection is an
/// application concern; the transport sends exactly the proof in
/// [`PeerConfig::connect_proof`].
pub struct Peer<S>
where
    S: BlobStore + BlobStorePut + CollectionStore + WantStore + StorageFlush + Send + 'static,
    S::Reader: BlobStoreMeta,
{
    /// The wrapped store, shared behind a mutex: a `&self` async read on
    /// a [`PeerReader`] must be able to record a want and land a
    /// swarm-fetched blob back into it (the one piece of Peer state the
    /// read snapshot must be able to mutate). All of Peer's own methods
    /// take the same lock.
    store: Arc<Mutex<S>>,

    sender: NetSender,
    receiver: NetReceiver,

    /// Baseline blob snapshot for diff-and-publish on `refresh`. The Reader
    /// is a frozen view (for backends with snapshot semantics like Pile) so
    /// `current.blobs_diff(&last)` returns exactly the blobs added since
    /// the last refresh.
    last_blob_reader: Option<S::Reader>,

    /// Intrinsic commit ids whose matching signed evidence has already
    /// been handed to the host gossip loop. This is only a publication-diff
    /// baseline: durable truth remains in the two grow-only stores.
    last_collection_commits: BTreeSet<Id>,

    /// Direction of swarm participation — controls collection-evidence and
    /// blob publication/reception.
    direction: SyncDirection,

    /// Monotonic time of the most recent NetEvent absorbed in
    /// [`refresh`](Peer::refresh). Drives quiescence-based stopping
    /// in long-running sync drivers. Read through [`crate::clock`] so
    /// simulated runs measure quiescence in virtual time.
    last_event_at: crate::clock::Mono,
}

impl<S> Peer<S>
where
    S: BlobStore + BlobStorePut + CollectionStore + WantStore + StorageFlush + Send + 'static,
    S::Reader: BlobStoreMeta,
{
    /// Wrap a store in a Peer. Spawns the iroh network thread
    /// internally; the thread lives for the Peer's lifetime and shuts
    /// down when the Peer drops.
    pub fn new(store: S, key: SigningKey, config: PeerConfig) -> Self {
        let direction = config.direction;
        let (sender, receiver) = host::spawn(key, config);
        Self::assemble(store, sender, receiver, direction)
    }

    /// Wrap a store in a Peer over caller-provided channel halves — the
    /// host loop runs wherever the caller put it (deterministic
    /// simulation: a local task on a shared paused runtime) instead of
    /// on an internally-spawned thread.
    ///
    /// Pair with [`crate::host::wire`] + [`crate::host::run_host`].
    pub fn with_wiring(
        store: S,
        direction: SyncDirection,
        sender: host::NetSender,
        receiver: host::NetReceiver,
    ) -> Self {
        Self::assemble(store, sender, receiver, direction)
    }

    fn assemble(
        mut store: S,
        sender: host::NetSender,
        receiver: host::NetReceiver,
        direction: SyncDirection,
    ) -> Self {
        // Seed the snapshot served by the network thread so peers
        // requesting via the protocol see our current state immediately.
        sender.refresh_store_snapshot(&mut store);

        // Baseline starts as None. The first `refresh` will diff the
        // store against this and announce every existing blob to the
        // DHT — same outcome as a dedicated startup sweep, but with no
        // race between sweep and baseline capture (a previous design
        // ran both as separate `reader()` calls; an external append
        // landing between them would slip into the baseline without
        // ever being announced).
        let mut peer = Peer {
            store: Arc::new(Mutex::new(store)),
            sender,
            receiver,
            last_blob_reader: None,
            last_collection_commits: BTreeSet::new(),
            direction,
            last_event_at: crate::clock::mono_now(),
        };

        // Drive the first refresh synchronously so the DHT learns
        // about pre-existing blobs before construction returns and the
        // first incoming AUTH can land.
        peer.refresh();

        peer
    }

    /// Monotonic time of the most recent network event absorbed by
    /// [`refresh`](Self::refresh). Useful for quiescence-based stopping:
    /// long-running sync drivers can poll `peer.last_event_at().elapsed()`
    /// and shut down once the swarm goes silent.
    ///
    /// Constructed-at-`Peer::new` initial value, so the first quiescence
    /// window starts at construction rather than at the first event.
    /// Returned as a [`crate::clock::Mono`] — virtual-time-aware under
    /// simulation, `.elapsed()`-compatible either way.
    pub fn last_event_at(&self) -> crate::clock::Mono {
        self.last_event_at
    }

    /// Direction of swarm participation. See [`SyncDirection`].
    pub fn direction(&self) -> SyncDirection {
        self.direction
    }

    /// This peer's network identity (the iroh node id).
    pub fn id(&self) -> EndpointId {
        self.sender.id()
    }

    /// Swarm-addressed on-demand blob fetch — the lazy-replication
    /// read-miss primitive, run **inline** (no command round-trip).
    /// Awaits the verified bytes or `None` (Unavailable); a host that
    /// never came up also resolves to `None`, never a hang. Bounded
    /// end-to-end by [`host::INTERACTIVE_FETCH_DEADLINE`] (the
    /// per-stage dial/op deadlines alone could stack to 40s+ across a
    /// provider list); use
    /// [`fetch_blob_with_deadline`](Self::fetch_blob_with_deadline) to
    /// pass a different budget. Does NOT persist the result and records
    /// no want — that is the caller's policy choice (see
    /// [`get_or_fetch_async`](Self::get_or_fetch_async) for the
    /// want-then-fetch-then-put composition). Used in
    /// deterministic-sim drivers, polled while stepping the sim.
    pub async fn fetch_blob(&self, hash: RawHash) -> Option<Vec<u8>> {
        self.sender
            .fetch_blob(hash, host::INTERACTIVE_FETCH_DEADLINE)
            .await
    }

    /// [`fetch_blob`](Self::fetch_blob) with an explicit end-to-end
    /// budget. Interactive reads keep the tight default; background
    /// work (the want-reconciler's tick) passes a more generous one.
    /// Expiry resolves to `None` — same Unavailable semantics, and any
    /// recorded want stays recorded.
    pub async fn fetch_blob_with_deadline(
        &self,
        hash: RawHash,
        budget: std::time::Duration,
    ) -> Option<Vec<u8>> {
        self.sender.fetch_blob(hash, budget).await
    }

    /// Probe every configured peer for exact merge/derive receipts answering
    /// `request`. The result separates inert deterministic evidence from sweep
    /// completeness, so callers may retain healthy partial answers without
    /// treating a stalled peer as a definitive absence. This method neither
    /// inserts nor flushes records.
    pub async fn fetch_collection_operation_receipts_with_deadline(
        &self,
        request: WantRequest,
        budget: std::time::Duration,
    ) -> crate::host::CollectionOperationProbe {
        self.sender
            .fetch_collection_operation_receipts(request, budget)
            .await
    }

    /// Fetch one exact collection's signed commit evidence from `peer`.
    ///
    /// The transport runs on the jailed host runtime. The returned evidence
    /// is strictly verified but inert: this method does not mutate the local
    /// store or fetch any blob referenced by a commit.
    pub fn fetch_collection_evidence_from(
        &self,
        peer: [u8; 32],
        collection: triblespace_core::collection::CollectionHandle,
    ) -> anyhow::Result<Vec<triblespace_core::collection::CollectionCommit>> {
        self.sender.fetch_collection_evidence(peer, collection)
    }

    /// Reconcile one exact collection from a specific peer.
    ///
    /// Transport work completes before the store lock is taken. Each commit is
    /// independently verified, then caller policy decides the complete batch
    /// before the first mutation. Accepted evidence is inserted and flushed
    /// once; referenced blobs remain independent, lazy resources.
    pub fn reconcile_collection_from<AuthorizationError, Authorize>(
        &mut self,
        peer: [u8; 32],
        collection: triblespace_core::collection::CollectionHandle,
        authorize: Authorize,
    ) -> Result<CollectionReconcileOutcome, CollectionReconcileError>
    where
        AuthorizationError: std::error::Error + Send + Sync + 'static,
        Authorize: FnMut(
            &triblespace_core::collection::CollectionCommit,
        ) -> Result<bool, AuthorizationError>,
    {
        let evidence = self
            .fetch_collection_evidence_from(peer, collection)
            .map_err(CollectionReconcileError::Fetch)?;
        let prepared = prepare_incoming_collection_batch(evidence)
            .map_err(CollectionReconcileError::Validation)?;
        let authorized = prepared
            .authorize(authorize)
            .map_err(|error| CollectionReconcileError::Admission(anyhow::Error::new(error)))?;
        let mut store = self.store.lock().expect("store mutex");
        let outcome = authorized
            .admit(&mut *store)
            .map_err(|error| CollectionReconcileError::Admission(anyhow::Error::new(error)))?;
        if self.sender.refresh_store_snapshot(&mut *store) {
            self.last_blob_reader = store.reader().ok();
        }
        Ok(outcome)
    }

    /// Reconcile this peer with the latest external state.
    ///
    /// Two phases:
    ///
    /// 1. **Drain incoming events** — pulls any pending gossip
    ///    `NetEvent`s from the network thread into the wrapped store.
    /// 2. **Publish external writes** — diffs the wrapped store against
    ///    the last published baseline and gossips/announces any deltas
    ///    that didn't go through the Peer's own write path. Use this to
    ///    catch writes from another process that touched the pile file.
    ///
    /// Auto-called inside the BlobStore read methods, so
    /// callers using the storage normally don't need to invoke it.
    /// Mirrors `Pile::refresh` — the explicit method is available for
    /// "do it now" semantics or tight loops with no read activity.
    pub fn refresh(&mut self) {
        // ── Phase 1: drain incoming events ────────────────────────────
        // WriteOnly suppresses incoming collection convergence. CONNECT is
        // verified in the transport before these events can exist and is not
        // itself replicated state.
        let mut incoming_collection_evidence = Vec::new();
        while let Some(event) = self.receiver.try_recv() {
            self.last_event_at = crate::clock::mono_now();
            if !accepts_incoming_event(self.direction, &event) {
                continue;
            }
            let NetEvent::CollectionEvidence(evidence) = event;
            incoming_collection_evidence.push(evidence);
        }
        // Gossip is a duplicate-delivery medium, while the pure preparation
        // boundary accepts only a canonical, strictly ordered batch. Reduce
        // the entire drain first, then mutate and flush the store once.
        if !incoming_collection_evidence.is_empty() {
            let mut store = self.store.lock().expect("store mutex");
            if let Err(error) =
                admit_incoming_collection_evidence(&mut *store, incoming_collection_evidence)
            {
                tracing::warn!(%error, "peer: invalid collection gossip batch dropped");
            }
        }

        let mut store = self.store.lock().expect("store mutex");

        // ── Phase 2: refresh the snapshot served by the network thread ─
        //
        // MUST happen before any announce/gossip below: peers who hear
        // our announce/gossip will dial us to fetch the closure, and
        // the network thread serves them out of this snapshot. If we
        // gossiped first, a fast-dialing peer would hit `has_blob =
        // false` on the still-stale snapshot and the server would deny
        // OP_CHILDREN/OP_GET_BLOB as "out of scope" — even though we
        // just told them we have it.
        let serving_snapshot_ready = self.sender.refresh_store_snapshot(&mut *store);

        // ── Phase 3: diff-and-publish blob deltas ─────────────────────
        // ReadOnly skips the publish: we still update the baseline
        // reader so we don't accumulate a publish backlog if the
        // direction later changes. On the first refresh the baseline
        // is `None`, so we announce every blob currently in the store —
        // covers the initial pile contents without a separate startup
        // sweep (and without the race that two separate `reader()`
        // calls introduced).
        if serving_snapshot_ready && let Ok(current) = store.reader() {
            if self.direction != SyncDirection::ReadOnly {
                match self.last_blob_reader.as_ref() {
                    Some(baseline) => {
                        for info in current.blobs_diff(baseline).flatten() {
                            self.sender.announce(info.handle.raw);
                        }
                    }
                    None => {
                        use triblespace_core::repo::BlobStoreList;
                        for info in current.blobs().filter_map(Result::ok) {
                            self.sender.announce(info.handle.raw);
                        }
                    }
                }
            }
            self.last_blob_reader = Some(current);
        }

        // ── Phase 4: diff-and-publish collection evidence ────────────
        // A commit is eligible for publication when it verifies and its
        // collection's own descriptor says the collection travels. There is no
        // second store to intersect with: permission is part of the name.
        // Received evidence may be relayed after admission -- its transport
        // carrier is deliberately not treated as its author -- but only if
        // this node can resolve a descriptor that permits it.
        if let Some(evidence) = relayable_collection_evidence(&mut *store) {
            if self.direction != SyncDirection::ReadOnly {
                for item in &evidence {
                    if !self.last_collection_commits.contains(&item.id()) {
                        self.sender.gossip_collection_evidence(*item);
                    }
                }
            }
            self.last_collection_commits = evidence.into_iter().map(|item| item.id()).collect();
        }

        // Pin heads remain local storage state. Collection evidence above is
        // the only semantic state published through gossip.
    }

    /// Lock and borrow the underlying store. Use for store-specific
    /// methods that aren't part of the storage traits (e.g.
    /// `Pile::flush`, `Yard::collect`, `WantStore::wants`).
    ///
    /// Writes through this borrow bypass the Peer's auto-publish and serving
    /// snapshot. Drop the guard and call [`refresh`](Self::refresh) after
    /// store-specific writes. Don't hold the guard across calls back into the
    /// Peer — its own methods take the same lock.
    pub fn store(&self) -> MutexGuard<'_, S> {
        self.store.lock().expect("store mutex")
    }

    /// Consume the Peer and return the underlying store. The network
    /// thread shuts down when the Peer drops.
    ///
    /// # Panics
    ///
    /// Panics if an outstanding [`PeerReader`] still shares the store
    /// (each reader carries a fetch capability that can land blobs into
    /// it) — drop all readers first.
    pub fn into_store(self) -> S {
        let Self { store, .. } = self;
        match Arc::try_unwrap(store) {
            Ok(mutex) => mutex
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Err(_) => panic!(
                "Peer::into_store: an outstanding PeerReader still shares the store; drop readers first"
            ),
        }
    }

    /// Read `hash` from the local store only, without touching the
    /// swarm. `Some(bytes)` on a local hit, `None` on a local miss —
    /// this is the cheap, non-blocking half of the read path, safe to
    /// call speculatively (e.g. the conservative reference scan asking
    /// "do I already hold this?"). Calls [`refresh`](Self::refresh)
    /// first so freshly-gossiped blobs count as local.
    pub fn try_local(&mut self, hash: RawHash) -> Option<Bytes> {
        let reader = self.reader().ok()?;
        reader.get::<Bytes, UnknownBlob>(Inline::new(hash)).ok()
    }

    /// Honest **async** lazy read: return `hash`'s bytes, fetching from
    /// the swarm and landing them wanted into the store on a local
    /// miss.
    ///
    /// 1. **Local** — one lookup in the store
    ///    (via [`try_local`](Self::try_local)). Hit ⇒ return
    ///    immediately, no network, no want.
    /// 2. **Miss** — the demand-born want: `want(hash)` is
    ///    recorded durably FIRST — asserted and **flushed**, so the want
    ///    survives an immediate process exit. The want IS the
    ///    want-signal (a sync daemon's work queue), then — once the
    ///    fetch lands — the retention marker for the fetched blob, then
    ///    the eviction target. Only then is the swarm-addressed fetch
    ///    awaited (DHT-routed, hash-verified) and the verified bytes
    ///    `put` into the store. If the fetch fails, the want stays:
    ///    it remains an outstanding want.
    ///
    /// `Ok(None)` is *Unavailable*: nobody reachable served it before
    /// the budget expired. Existence is semidecidable — there is no
    /// "definitely absent" outcome — and the want stays on record.
    ///
    /// `Err` means the want could NOT be durably recorded (want or flush
    /// failed). No fetch is attempted in that case: proceeding would
    /// hand the caller bytes whose demand isn't on record, silently
    /// breaking the want-on-record invariant every daemon relies on.
    ///
    /// The swarm fetch is *awaited*, never blocking the caller's thread:
    /// the reply rides a tokio oneshot, so this composes inside any async
    /// consumer and drives cleanly on a single-threaded runtime (the
    /// await yields, letting the host produce the reply).
    pub async fn get_or_fetch_async(
        &mut self,
        hash: RawHash,
    ) -> Result<Option<Bytes>, WantRecordError<S::WantError, <S as StorageFlush>::Error>> {
        if let Some(bytes) = self.try_local(hash) {
            return Ok(Some(bytes));
        }
        // Record the want durably BEFORE the fetch — a failed fetch
        // must leave the demand on record, and a failed RECORD must be
        // an error, never a silent proceed. (Guard dropped before the
        // await: never hold the store lock across a suspension.)
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
            if let Err(e) = store.put::<UnknownBlob, Bytes>(bytes.clone()) {
                // Landing failed but the verified bytes are in hand and
                // the want IS on record — a later reconcile pass re-lands
                // it. Loud trace, non-fatal.
                tracing::warn!(
                    hash = %hex::encode(&hash[..4]),
                    error = ?e,
                    "get_or_fetch: landing fetched blob failed"
                );
            }
        }
        Ok(Some(bytes))
    }
}

// ── Trait delegations ───────────────────────────────────────────────
//
// Reads call `refresh()` first so they always see the latest collection
// evidence and any external blob writes get announced. Blob writes delegate
// to the inner store and then publish the new state.

impl<S> BlobStorePut for Peer<S>
where
    S: BlobStore + BlobStorePut + CollectionStore + WantStore + StorageFlush + Send + 'static,
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
        // Snapshot first, then announce — see `refresh` Phase 2 for the
        // ordering rationale. Without this, DHT-receivers of the announce
        // dial us, OP_GET_BLOB hits the stale snapshot, returns missing,
        // and the receiver waits for backoff to retry.
        if self.sender.refresh_store_snapshot(&mut *store) {
            if self.direction != SyncDirection::ReadOnly {
                self.sender.announce(handle.raw);
            }
            // Update the blob baseline so refresh doesn't double-announce.
            self.last_blob_reader = store.reader().ok();
        }
        Ok(handle)
    }
}

impl<S> BlobStore for Peer<S>
where
    S: BlobStore + BlobStorePut + CollectionStore + WantStore + StorageFlush + Send + 'static,
    S::Reader: BlobStoreMeta,
{
    type Reader = PeerReader<S::Reader>;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.refresh();
        let local = self.store.lock().expect("store mutex").reader()?;
        // The fetch capability: a clone of the command sender plus a
        // landing handle into the *shared* store, so a `&self` async
        // read can pull a missing blob from the swarm, record the
        // demand-born want, and land the bytes.
        let fetch = Some(FetchCap {
            sender: self.sender.clone(),
            sink: Arc::new(SharedStore(self.store.clone())),
        });
        Ok(PeerReader { local, fetch })
    }
}

#[cfg(test)]
mod collection_gossip_tests {
    use std::convert::Infallible;

    use triblespace_core::collection::{
        CollectionCommit, CollectionData, CollectionHandle, empty_metadata_handle,
    };

    use super::*;

    #[derive(Default)]
    struct TestStore {
        records: Vec<CollectionRecord>,
        flushes: usize,
    }

    impl CollectionStore for TestStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            Ok(self
                .records
                .iter()
                .copied()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            if !self.records.iter().any(|known| known.id() == record.id()) {
                self.records.push(record);
            }
            Ok(())
        }
    }

    impl StorageFlush for TestStore {
        type Error = Infallible;

        fn flush(&mut self) -> Result<(), Self::Error> {
            self.flushes += 1;
            Ok(())
        }
    }

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    fn commit(author: &SigningKey, collection: CollectionHandle, byte: u8) -> CollectionCommit {
        CollectionCommit::sign(
            author,
            collection,
            CollectionData::new([byte; 32]),
            empty_metadata_handle(),
        )
    }

    /// The peer reads relay permission out of the same store it reads records
    /// from.
    ///
    /// `relayable_commits` is tested on its own in `collection_wire`; what is
    /// checked here is the wiring, because a correct selection function
    /// pointed at the wrong store is exactly as leaky as a wrong one. A real
    /// `Collection::commit` writes its own descriptor as a dependency, so a
    /// publisher's store holds its own permission by construction -- there is
    /// no second act, and nothing to forget.
    #[test]
    fn a_peer_relays_only_what_its_own_store_says_may_travel() {
        use triblespace_core::collection::records::CollectionName;
        use triblespace_core::collection::{Collection, reach};
        use triblespace_core::repo::memoryrepo::MemoryRepo;
        use triblespace_core::trible::{Fragment, TribleSet};

        let author = SigningKey::from_bytes(&[11; 32]);
        let team = author.verifying_key();
        let mut store = MemoryRepo::default();

        let published = Collection::new(
            &mut store,
            &CollectionName::new("published").unwrap(),
            team,
            author.clone(),
            reach::public(),
        )
        .commit(Fragment::from(TribleSet::new()))
        .unwrap();

        let withheld = Collection::new(
            &mut store,
            &CollectionName::new("withheld").unwrap(),
            team,
            author.clone(),
            reach::private(),
        )
        .commit(Fragment::from(TribleSet::new()))
        .unwrap();

        // Two collections, one author, one store. Only the one whose
        // descriptor says so is served.
        assert_ne!(published.collection(), withheld.collection());
        assert_eq!(
            relayable_collection_evidence(&mut store).unwrap(),
            vec![published]
        );

        // And the descriptors really are resident, so the refusal above is
        // about what the private descriptor says rather than about a lookup
        // that failed.
        let reader = store.reader().unwrap();
        for collection in [published.collection(), withheld.collection()] {
            let facts = reader
                .get::<TribleSet, triblespace_core::blob::encodings::simplearchive::SimpleArchive>(
                    collection,
                )
                .expect("commit writes its own descriptor");
            assert!(!facts.is_empty());
        }
        assert!(triblespace_core::collection::reach::travels(
            &reader
                .get::<TribleSet, triblespace_core::blob::encodings::simplearchive::SimpleArchive>(
                    published.collection()
                )
                .unwrap()
        ));
    }

    #[test]
    fn one_gossip_drain_deduplicates_and_flushes_once() {
        let author = SigningKey::from_bytes(&[9; 32]);
        let collection = collection(3);
        let first = commit(&author, collection, 1);
        let second = commit(&author, collection, 2);
        let mut store = TestStore::default();

        let counts =
            admit_incoming_collection_evidence(&mut store, vec![second, first, second]).unwrap();

        assert_eq!(counts.observed, 2);
        assert_eq!(counts.admitted, 2);
        assert_eq!(counts.denied, 0);
        assert_eq!(store.records.len(), 2);
        assert_eq!(store.flushes, 1);
    }

    #[test]
    fn write_only_rejects_incoming_collection_evidence() {
        let author = SigningKey::from_bytes(&[10; 32]);
        let collection = collection(4);
        let evidence = commit(&author, collection, 1);
        let event = NetEvent::CollectionEvidence(evidence);
        assert!(!accepts_incoming_event(SyncDirection::WriteOnly, &event));
        assert!(accepts_incoming_event(SyncDirection::ReadOnly, &event));
        assert!(accepts_incoming_event(SyncDirection::Bidirectional, &event));
    }
}

/// The read view of a [`Peer`]: the store's own reader (`L`) plus a
/// swarm-fetch capability.
///
/// Two read surfaces with deliberately different semantics:
/// - the **sync** [`BlobStoreGet`] is *local only* — one lookup in the
///   store snapshot, never the swarm. This keeps speculative gets (the
///   conservative reference scan, existence checks) cheap and total:
///   enumeration and existence stay local, the decomplecting that lets
///   "the layers above the blob substrate do whatever fancy dance they
///   like" hold.
/// - the **async** [`AsyncBlobStoreGet`] is *transparent* — local
///   lookup, else a demand-born want followed by an awaited swarm
///   fetch that lands the result in the shared store. This is what
///   gives a generic async consumer lazy replication for free, without ever
///   knowing it holds a `Peer`.
///
/// So existence-vs-retrieval is split by *which trait you call*, not by
/// a bespoke method: probe with the sync `get`, retrieve with the async
/// one.
///
/// [`AsyncBlobStoreGet`]: triblespace_core::repo::async_store::AsyncBlobStoreGet
pub struct PeerReader<L> {
    local: L,
    /// Swarm-fetch capability for the async transparent read. The sync
    /// reads never touch it; it carries the command sender plus a
    /// landing handle into the Peer's shared store.
    fetch: Option<FetchCap>,
}

/// The capability a [`PeerReader`] needs to pull a missing blob from the
/// swarm: the host command sender + a want-recording/landing sink into
/// the Peer's shared store.
#[derive(Clone)]
struct FetchCap {
    sender: NetSender,
    sink: Arc<dyn StoreSink>,
}

/// Interior-mutable access to the Peer's shared store for a `&self`
/// async read: record the demand-born want, land the fetched bytes.
/// Erases the concrete store type `S` so `PeerReader` need not carry it
/// — which is also why `record_want`'s error is boxed.
trait StoreSink: Send + Sync {
    /// Durably record the want: want `hash` AND flush it BEFORE the
    /// fetch, so a failed fetch — or an immediate process exit — leaves
    /// the outstanding demand on record. A failed record is an error the
    /// read must surface, never a warn-and-continue.
    fn record_want(&self, hash: RawHash) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    /// Land fetched `bytes` as an `UnknownBlob` into the store.
    fn land(&self, bytes: Bytes);
}

/// `StoreSink` over the Peer's shared store handle.
struct SharedStore<S>(Arc<Mutex<S>>);

impl<S> StoreSink for SharedStore<S>
where
    S: BlobStorePut + WantStore + StorageFlush + Send + 'static,
{
    fn record_want(&self, hash: RawHash) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut store = self.0.lock().expect("store mutex");
        store
            .want(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)))
            .map_err(|e| {
                Box::new(WantRecordError::<_, <S as StorageFlush>::Error>::Want(e))
                    as Box<dyn std::error::Error + Send + Sync>
            })?;
        store.flush().map_err(|e| {
            Box::new(WantRecordError::<S::WantError, _>::Flush(e))
                as Box<dyn std::error::Error + Send + Sync>
        })?;
        Ok(())
    }

    fn land(&self, bytes: Bytes) {
        if let Ok(mut store) = self.0.lock() {
            if let Err(e) = store.put::<UnknownBlob, Bytes>(bytes) {
                tracing::warn!(error = ?e, "reader fetch: landing fetched blob failed");
            }
        }
    }
}

// Identity ignores the fetch capability: two readers are equal iff their
// local store views are — the capability is a handle, not part of the
// snapshot's value. Hand-rolled because `NetSender` / `Arc<dyn
// StoreSink>` are neither `PartialEq` nor (for the sender) `Sync`, so
// the derive can't apply.
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

/// Error from the async transparent read on a [`PeerReader`].
#[derive(Debug)]
pub enum PeerReaderGetError<E> {
    /// The bytes (local or swarm-fetched) didn't convert to the
    /// requested type.
    Conversion(E),
    /// Not held locally and the swarm didn't serve it before the host
    /// resolved the fetch. Existence is semidecidable — this is
    /// "not obtained", never "definitely absent". The demand-born want
    /// recorded before the fetch stays on record.
    Unavailable,
    /// Local miss AND the demand-born want could not be durably
    /// recorded (want or flush failed). No fetch was attempted — the
    /// want-on-record invariant must hold before any bytes move.
    /// Boxed because the reader's store type is erased behind the
    /// fetch capability; the concrete error is a
    /// [`WantRecordError`].
    WantRecord(Box<dyn std::error::Error + Send + Sync>),
}

impl<E: std::error::Error> std::fmt::Display for PeerReaderGetError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conversion(e) => write!(f, "blob conversion failed: {e}"),
            Self::Unavailable => write!(f, "blob unavailable (local miss + swarm did not serve)"),
            Self::WantRecord(e) => {
                write!(f, "blob missing and want not recorded: {e}")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for PeerReaderGetError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conversion(e) => Some(e),
            Self::Unavailable => None,
            Self::WantRecord(e) => Some(e.as_ref()),
        }
    }
}

impl<L> BlobStoreGet for PeerReader<L>
where
    L: BlobStoreGet,
{
    type GetError<E: std::error::Error + Send + Sync + 'static> = L::GetError<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.local.get::<T, S>(handle)
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

    fn contains_blob<S>(&self, handle: Inline<Handle<S>>) -> Result<bool, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.local.contains_blob(handle)
    }
}

// Conservative reference discovery works through the local `get`: the
// default scan checks each 32-byte chunk against the store snapshot,
// which — post-fetch — also holds any wanted lazily-landed blobs.
impl<L> BlobChildren for PeerReader<L> where L: BlobStoreGet {}

/// Transparent async read: local lookup → a demand-born want + an
/// awaited swarm fetch that lands the result in the shared store. This
/// is the surface a *generic* async consumer depends on to get lazy
/// replication for free — it never needs to know it's holding a `Peer`.
impl<L> triblespace_core::repo::async_store::AsyncBlobStoreGet for PeerReader<L>
where
    L: BlobStoreGet + Clone + Send + 'static,
{
    type GetError<E: std::error::Error + Send + Sync + 'static> = PeerReaderGetError<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> impl std::future::Future<Output = Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>> + Send
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        // Clone the owned read handle + fetch capability *before* the
        // async block so the future captures only `Send` values — never
        // `&self` (`NetSender` is `!Sync`). Keeps the future `Send`
        // without forcing `L: Sync`.
        let raw = handle.raw;
        let local = self.local.clone();
        let fetch = self.fetch.clone();
        async move {
            // Universal byte read: the store snapshot locally, else the
            // swarm. Bytes-by-hash everywhere, so deserialization to the
            // requested schema happens once, below.
            let bytes: Bytes = if let Ok(b) = local.get::<Bytes, UnknownBlob>(Inline::new(raw)) {
                b
            } else if let Some(cap) = fetch {
                // The demand-born want: record the want durably
                // FIRST (want + flush), then fetch. A failed fetch
                // leaves the want — it remains an outstanding want. A
                // failed RECORD is an error: never fetch bytes whose
                // demand isn't on record.
                cap.sink
                    .record_want(raw)
                    .map_err(PeerReaderGetError::WantRecord)?;
                // Inline swarm fetch; the host verified
                // blake3(bytes) == raw before returning. Interactive
                // budget: a transparent read is a caller actively
                // waiting.
                match cap
                    .sender
                    .fetch_blob(raw, crate::host::INTERACTIVE_FETCH_DEADLINE)
                    .await
                {
                    Some(v) => {
                        let b = Bytes::from(v);
                        cap.sink.land(b.clone());
                        b
                    }
                    None => return Err(PeerReaderGetError::Unavailable),
                }
            } else {
                return Err(PeerReaderGetError::Unavailable);
            };
            triblespace_core::blob::Blob::<S>::new(bytes)
                .try_from_blob()
                .map_err(PeerReaderGetError::Conversion)
        }
    }
}
