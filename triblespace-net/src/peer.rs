//! `Peer<S>`: a store wrapped in distributed network sync.
//!
//! Owns the inner store, spawns the iroh network thread on construction,
//! and exposes the standard blob storage traits with network behavior built
//! in. Its signer-owned policy collection stays on the
//! inner store and is deliberately not exposed for gossip through `Peer`:
//!
//! - **Reads** auto-call [`refresh`](Peer::refresh), which drains pending
//!   incoming gossip events into the wrapped store and re-publishes any
//!   deltas from external writers (e.g. another process appended to the
//!   same pile file). Mirrors `Pile::refresh` — the explicit method is
//!   available for tight loops, but normal storage use Just Works.
//! - **Publication** diffs [`CollectionStore`] and [`CollectionGossipStore`]
//!   state, then gossips each strictly verified grant-backed commit as inert
//!   signed evidence. Receivers may store and relay that evidence without
//!   fetching its referenced blobs; semantic trust belongs to later
//!   collection resolution.
//! - **Blob writes** delegate to the inner store and announce content to the
//!   DHT. A read-only [`PinSnapshotSource`] supplies the pin-head view still
//!   used by branch-restricted capability checks; pins are not replicated
//!   state and `Peer` exposes no pin mutation API.
//!
//! There is no separate cache tier: `Peer<S>` takes a **single store**,
//! and any tiering (bounded want retention, generational eviction) lives
//! in `S` — e.g. a [`Yard`](triblespace_core::repo::yard::Yard). Read-miss
//! swarm fetches land in `S` under a **want** ([`WantStore`]),
//! independently of named pins and private policy collections. The want is
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
//! "Promote to durable" is not an operation — durability is
//! reachability from strong pins; the Peer performs no promotion.
//!
//! Collection discovery is gossip-driven: immutable grant+commit evidence
//! floods the team topic and arrives through `NetEvent`. Referenced content
//! remains independently retrievable by hash, normally through durable wants.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{BlobEncoding, IntoBlob, TryFromBlob};
use triblespace_core::collection::{
    CollectionGossip, CollectionGossipStore, CollectionRecord, CollectionStore,
};
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::lazy::WantRecordError;
use triblespace_core::repo::{
    BlobChildren, BlobStore, BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut,
    PinSnapshotSource, StorageFlush, WantRequest, WantStore,
};

use crate::channel::{NetEvent, PublisherKey};
use crate::collection_sync::{
    IncomingBatchCounts, IncomingBatchValidationError, prepare_incoming_collection_batch,
};
use crate::collection_wire::{CollectionCommitEvidence, all_grant_backed_commits};
use crate::host::{self, NetReceiver, NetSender};
use crate::protocol::RawHash;

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

/// Materialize the strictly verified intersection of the collection record
/// and publication-grant stores. A top-level enumeration failure preserves
/// the caller's prior diff baseline; malformed individual rows are inert.
fn grant_backed_collection_evidence<S>(store: &mut S) -> Option<Vec<CollectionCommitEvidence>>
where
    S: CollectionGossipStore + CollectionStore,
{
    let grants: Vec<CollectionGossip> = store.gossips().ok()?.filter_map(Result::ok).collect();
    let records: Vec<CollectionRecord> = store.records().ok()?.filter_map(Result::ok).collect();
    Some(all_grant_backed_commits(&records, &grants))
}

/// Canonicalize, strictly verify, and durably admit one gossip drain.
///
/// Duplicate delivery is normal for gossip, so commit ids are reduced before
/// the strict batch boundary. The resulting batch performs exactly one flush.
fn admit_incoming_collection_evidence<S>(
    store: &mut S,
    mut evidence: Vec<CollectionCommitEvidence>,
) -> anyhow::Result<IncomingBatchCounts>
where
    S: CollectionGossipStore + CollectionStore + StorageFlush,
{
    evidence.sort_by_key(|item| item.commit().id());
    evidence.dedup_by_key(|item| item.commit().id());
    let batch = evidence
        .into_iter()
        .map(|item| (item.commit(), item.grant()))
        .collect();
    let prepared = prepare_incoming_collection_batch(batch)?;
    let authorized = prepared
        .authorize(|_, _| Ok::<bool, std::convert::Infallible>(true))
        .expect("infallible evidence-storage policy");
    authorized.admit(store).map_err(anyhow::Error::new)
}

/// Whether local direction policy admits this incoming event.
///
/// Directionality governs replicated data, not the capability control plane:
/// a pure publisher still has to receive join requests, issued capabilities,
/// and delivery acknowledgements in order to administer its access policy.
fn accepts_incoming_event(direction: SyncDirection, event: &NetEvent) -> bool {
    match event {
        NetEvent::Blob(_) | NetEvent::CollectionEvidence(_) => {
            direction != SyncDirection::WriteOnly
        }
        NetEvent::CapRequest { .. }
        | NetEvent::CapDelivered { .. }
        | NetEvent::CapDeliveryConfirmed { .. } => true,
    }
}

/// A store wrapped in distributed network sync.
///
/// See the [module-level docs](self) for the full mental model.
///
/// # Example
///
/// Single-user team-of-one setup against a [`Pile`]: the user is
/// their own team root, and the relay accepts only caps signed by
/// (or chained from) their own key. The `self_cap = [0u8; 32]`
/// sentinel will fail any remote `OP_AUTH` it sends — fine for
/// solo workflows where the peer is purely a server.
///
/// Multi-user setups load `team_root` and `self_cap` from the
/// `TRIBLE_TEAM_ROOT` and `TRIBLE_TEAM_CAP` environment variables;
/// see the [Capability Auth] book chapter for the full team
/// lifecycle.
///
/// [`Pile`]: triblespace_core::repo::pile::Pile
/// [Capability Auth]: https://docs.rs/triblespace/latest/triblespace/book/capability-auth/index.html
///
/// ```rust,no_run
/// use std::path::Path;
/// use ed25519_dalek::SigningKey;
/// use rand::rngs::OsRng;
/// use triblespace_core::repo::pile::Pile;
/// use triblespace_net::peer::{Peer, PeerConfig, SyncDirection};
///
/// let key = SigningKey::generate(&mut OsRng);
/// let pile: Pile = Pile::open(Path::new("./team.pile")).unwrap();
/// let peer = Peer::new(pile, key.clone(), PeerConfig {
///     peers: vec![],                       // bootstrap nodes
///     gossip: true,                        // false = serve/pull-only
///     team_root: key.verifying_key(),      // single-user fallback
///     self_cap: [0u8; 32],
///     direction: SyncDirection::Bidirectional,
/// });
/// // From here `peer` provides network-aware blob storage. Store-specific
/// // local state remains explicitly available through `peer.store()`.
/// drop(peer);
/// ```
pub struct Peer<S>
where
    S: BlobStore
        + BlobStorePut
        + CollectionGossipStore
        + CollectionStore
        + PinSnapshotSource
        + WantStore
        + StorageFlush
        + Send
        + 'static,
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

    /// Intrinsic commit ids whose matching grant-backed evidence has already
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

    /// Team root pubkey, copied from `PeerConfig::team_root` so the
    /// refresh loop can verify incoming `CapDelivered` events against
    /// it without round-tripping through the network thread.
    team_root: ed25519_dalek::VerifyingKey,

    /// Cloned signing key. ed25519's SigningKey is 32 bytes of secret
    /// scalar so cloning is cheap, but we keep it as an explicit
    /// `Clone` instead of `Copy` so the surface area for accidental
    /// duplication stays auditable. Used by `renewal_tick` to sign
    /// fresh caps for heads in the renewal-policy version DAG.
    signing_key: SigningKey,

    /// Per-entry cooldown for undelivered-cap re-dispatch. The
    /// renewal daemon's tick runs every 100 ms; without this gate it
    /// would hammer iroh-connect attempts for any peer that's down.
    /// Recorded against `entry.id`. Cleared (entry-level) when the
    /// delivery confirms; the whole map is in-memory and rebuilds
    /// naturally if the daemon restarts.
    last_dispatch_attempt: HashMap<Id, crate::clock::Mono>,
}

impl<S> Peer<S>
where
    S: BlobStore
        + BlobStorePut
        + CollectionGossipStore
        + CollectionStore
        + PinSnapshotSource
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
{
    /// Wrap a store in a Peer. Spawns the iroh network thread
    /// internally; the thread lives for the Peer's lifetime and shuts
    /// down when the Peer drops.
    pub fn new(store: S, key: SigningKey, config: PeerConfig) -> Self {
        let direction = config.direction;
        let team_root = config.team_root;
        let signing_key = key.clone();
        let (sender, receiver) = host::spawn(key, config);
        Self::assemble(store, sender, receiver, direction, team_root, signing_key)
    }

    /// Wrap a store in a Peer over caller-provided channel halves — the
    /// host loop runs wherever the caller put it (deterministic
    /// simulation: a local task on a shared paused runtime) instead of
    /// on an internally-spawned thread.
    ///
    /// Pair with [`crate::host::wire`] + [`crate::host::run_host`].
    pub fn with_wiring(
        store: S,
        signing_key: SigningKey,
        direction: SyncDirection,
        team_root: ed25519_dalek::VerifyingKey,
        sender: host::NetSender,
        receiver: host::NetReceiver,
    ) -> Self {
        Self::assemble(store, sender, receiver, direction, team_root, signing_key)
    }

    fn assemble(
        mut store: S,
        sender: host::NetSender,
        receiver: host::NetReceiver,
        direction: SyncDirection,
        team_root: ed25519_dalek::VerifyingKey,
        signing_key: SigningKey,
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
            team_root,
            signing_key,
            last_dispatch_attempt: HashMap::new(),
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

    /// Fetch one exact collection's grant-backed commit evidence from `peer`.
    ///
    /// The transport runs on the jailed host runtime. The returned evidence
    /// is strictly verified but inert: this method does not mutate the local
    /// store or fetch any blob referenced by a commit.
    pub fn fetch_collection_evidence_from(
        &self,
        peer: [u8; 32],
        collection: triblespace_core::collection::CollectionHandle,
    ) -> anyhow::Result<Vec<crate::collection_wire::CollectionCommitEvidence>> {
        self.sender.fetch_collection_evidence(peer, collection)
    }

    /// Reconcile one exact collection from a specific peer.
    ///
    /// Transport work completes before the store lock is taken. Each grant /
    /// commit pair is independently verified, then caller policy decides the
    /// complete batch before the first mutation. Accepted evidence is inserted
    /// and flushed once; referenced blobs remain independent, lazy resources.
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
            &triblespace_core::collection::CollectionGossip,
        ) -> Result<bool, AuthorizationError>,
    {
        let evidence = self
            .fetch_collection_evidence_from(peer, collection)
            .map_err(CollectionReconcileError::Fetch)?;
        let evidence = evidence
            .into_iter()
            .map(|item| (item.commit(), item.grant()))
            .collect();
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
        // WriteOnly suppresses incoming *data* convergence while preserving
        // capability-control traffic. Join requests, delivered capabilities,
        // and delivery confirmations are operational messages required to
        // administer even a pure publisher; silently discarding them would
        // make directionality alter the authorization protocol itself.
        let mut incoming_collection_evidence = Vec::new();
        while let Some(event) = self.receiver.try_recv() {
            self.last_event_at = crate::clock::mono_now();
            if !accepts_incoming_event(self.direction, &event) {
                continue;
            }
            match event {
                NetEvent::Blob(data) => {
                    // `data` is already an anybytes::Bytes (refcounted) —
                    // pass it into the store without re-wrapping.
                    let _ = self
                        .store
                        .lock()
                        .expect("store mutex")
                        .put::<UnknownBlob, Bytes>(data);
                }
                NetEvent::CollectionEvidence(evidence) => {
                    incoming_collection_evidence.push(evidence);
                }
                NetEvent::CapRequest {
                    requester,
                    partial_cap_bytes,
                } => {
                    self.absorb_cap_request(requester, partial_cap_bytes);
                }
                NetEvent::CapDelivered {
                    issuer,
                    cap_bytes,
                    sig_bytes,
                } => {
                    // Verify the delivered chain against our configured
                    // team root, then store both blobs locally and append our
                    // team-cap version, whose collection commit retains them
                    // through compaction.
                    self.absorb_cap_delivery(issuer, cap_bytes, sig_bytes);
                }
                NetEvent::CapDeliveryConfirmed {
                    subject,
                    sig_handle,
                } => {
                    // The subject's daemon authenticated against us with
                    // a cap we dispatched. `sig_handle` is the signature
                    // blob handle (what OP_AUTH wires) — match by
                    // subject + latest_sig and mark the entry delivered
                    // so the daemon's next tick skips it from the
                    // re-dispatch set.
                    use triblespace_core::inline::Inline;
                    use triblespace_core::inline::encodings::hash::Handle;
                    let subject_key = match ed25519_dalek::VerifyingKey::from_bytes(&subject) {
                        Ok(k) => k,
                        Err(_) => continue,
                    };
                    let sig_inline: Inline<Handle<SimpleArchive>> = Inline::new(sig_handle);
                    let mut store = self.store.lock().expect("store mutex");
                    match crate::policy::find_policy_entry_by_subject_and_sig(
                        &mut *store,
                        &self.signing_key,
                        subject_key,
                        sig_inline,
                    ) {
                        Ok(Some(entry_id)) => {
                            match crate::policy::mark_policy_delivered(
                                &mut *store,
                                &self.signing_key,
                                entry_id,
                            ) {
                                Ok(()) => tracing::debug!(
                                    subject = %hex::encode(&subject[..4]),
                                    sig = %hex::encode(&sig_handle[..4]),
                                    entry = ?entry_id,
                                    "delivery confirmed; policy acknowledgement recorded"
                                ),
                                Err(error) => tracing::warn!(
                                    entry = ?entry_id,
                                    %error,
                                    "delivery confirmation policy write failed"
                                ),
                            }
                        }
                        Ok(None) => {}
                        Err(error) => tracing::warn!(
                            subject = %hex::encode(&subject[..4]),
                            %error,
                            "delivery confirmation policy lookup failed"
                        ),
                    }
                }
            }
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
        // Commits and publication grants are orthogonal grow-only stores.
        // Only their strictly verified, same-author intersection is eligible
        // for publication. Received evidence may be relayed after admission:
        // its transport carrier is deliberately not treated as its author.
        if let Some(evidence) = grant_backed_collection_evidence(&mut *store) {
            if self.direction != SyncDirection::ReadOnly {
                for item in &evidence {
                    if !self.last_collection_commits.contains(&item.commit().id()) {
                        self.sender.gossip_collection_evidence(*item);
                    }
                }
            }
            self.last_collection_commits = evidence
                .into_iter()
                .map(|item| item.commit().id())
                .collect();
        }

        // Pin heads remain local storage state. Collection evidence above is
        // the only semantic state published through gossip.
    }

    /// Persist an incoming join request: store the partial-cap blob,
    /// then commit a request and observation to the private node-policy
    /// collection. The request id becomes the value `team approve <id>`
    /// consumes; the partial-cap blob is recoverable from the entity's
    /// `request_partial_cap` handle.
    fn absorb_cap_request(&mut self, requester: PublisherKey, partial_cap_bytes: anybytes::Bytes) {
        use triblespace_core::blob::Blob;
        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
        use triblespace_core::inline::TryToInline;

        // Reconstitute the requester pubkey from bytes. If the bytes
        // aren't a valid ed25519 pubkey, drop on the floor — only
        // iroh-verified peers reach this code path, so this is
        // defensive only.
        let Ok(requester_pubkey) = ed25519_dalek::VerifyingKey::from_bytes(&requester) else {
            tracing::warn!(
                requester = %hex::encode(&requester[..4]),
                "CapRequest: bad requester pubkey; dropping"
            );
            return;
        };

        let mut store = self.store.lock().expect("store mutex");

        // Store the partial cap blob so the approver can later read
        // its declared subject/scope/expiry without B re-sending.
        // partial_cap_bytes is already an anybytes::Bytes — wrap it
        // into a typed Blob without re-allocating.
        let blob: Blob<SimpleArchive> = Blob::new(partial_cap_bytes);
        let Ok(partial_cap_handle) = store.put::<SimpleArchive, Blob<SimpleArchive>>(blob) else {
            tracing::warn!("CapRequest: failed to store partial cap blob");
            return;
        };

        // Point-interval at "now" — pending-requests timeline is
        // just "this arrived at T".
        let now = crate::clock::epoch_now();
        let received_at = (now, now).try_to_inline().expect("point interval");

        match crate::policy::record_pending_request(
            &mut *store,
            &self.signing_key,
            requester_pubkey,
            partial_cap_handle,
            received_at,
        ) {
            Ok(req_id) => {
                let req_id_bytes: [u8; 16] = req_id.into();
                tracing::info!(
                    requester = %hex::encode(&requester[..4]),
                    request_id = %hex::encode(req_id_bytes),
                    "CapRequest recorded as pending"
                );
            }
            Err(error) => {
                tracing::warn!(
                    requester = %hex::encode(&requester[..4]),
                    %error,
                    "CapRequest: failed to record in policy collection"
                );
            }
        }
    }

    /// Verify a peer-delivered cap chain against our configured team
    /// root and, on success, store both blobs locally.
    ///
    /// The current pair is recorded as a version in the signer-owned policy
    /// collection, independently of branch authority or gossip permission.
    fn absorb_cap_delivery(
        &mut self,
        issuer: PublisherKey,
        cap_bytes: anybytes::Bytes,
        sig_bytes: anybytes::Bytes,
    ) {
        use triblespace_core::blob::Blob;
        use triblespace_core::repo::BlobStoreGet;

        // Verification + swarm-fetch of any missing chain blobs
        // already happened in the host thread's HandshakeHandler
        // (the OP_DELIVER_CAP path doesn't ack STATUS_OK until the
        // chain verifies under our pubkey). The cap+sig blobs +
        // every fetched parent have already arrived as earlier
        // `NetEvent::Blob` events on this channel, so by the time
        // we get here the store already holds them and we only
        // need to append our team-cap version naming the leaf pair.
        let cap_blob: Blob<SimpleArchive> = Blob::new(cap_bytes);
        let sig_blob: Blob<SimpleArchive> = Blob::new(sig_bytes);
        let cap_handle: Inline<Handle<SimpleArchive>> = (&cap_blob).get_handle();
        let sig_handle: Inline<Handle<SimpleArchive>> = (&sig_blob).get_handle();

        let mut store = self.store.lock().expect("store mutex");

        // Defensive sanity: the cap+sig blobs really are in the
        // store. If not, the host emitted the CapDelivered event
        // without the preceding Blob events somehow — log and bail
        // rather than pin handles that won't resolve.
        let Ok(reader) = store.reader() else {
            tracing::warn!(
                issuer = %hex::encode(&issuer[..4]),
                "CapDelivered: pile reader unavailable; dropping"
            );
            return;
        };
        if reader
            .get::<Blob<SimpleArchive>, SimpleArchive>(cap_handle)
            .is_err()
            || reader
                .get::<Blob<SimpleArchive>, SimpleArchive>(sig_handle)
                .is_err()
        {
            tracing::warn!(
                issuer = %hex::encode(&issuer[..4]),
                "CapDelivered: blobs missing from store (host should have emitted Blob events first)"
            );
            return;
        }

        match crate::policy::set_team_cap(
            &mut *store,
            &self.signing_key,
            self.team_root,
            cap_handle,
            sig_handle,
        ) {
            Ok(()) => {
                tracing::info!(
                    issuer = %hex::encode(&issuer[..4]),
                    sig = %hex::encode(&sig_handle.raw[..4]),
                    "CapDelivered: stored in private policy collection"
                );
            }
            Err(error) => {
                tracing::warn!(
                    issuer = %hex::encode(&issuer[..4]),
                    %error,
                    "CapDelivered: team-cap policy write failed"
                );
            }
        }
    }

    /// Cooldown for re-dispatching undelivered cap blobs. The daemon's
    /// tick cadence is sub-second; without this gate we'd hammer
    /// iroh-connect against a down peer 10× per second.
    const UNDELIVERED_REDISPATCH_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(15);

    /// Re-dispatch the cap+sig pairs for every renewal-policy entry
    /// that's not yet been ack'd by its subject, rate-limited per
    /// entry via `last_dispatch_attempt`. The cap is NOT re-signed —
    /// the same `(latest_cap, latest_sig)` blobs are sent again, so
    /// idempotent on the receiver side (their OP_DELIVER_CAP handler
    /// content-hashes the bytes and dedupes against what's already
    /// pinned).
    ///
    /// Returns the count of entries dispatched this tick.
    fn redispatch_undelivered(&mut self) -> usize {
        use triblespace_core::blob::Blob;
        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
        use triblespace_core::repo::BlobStoreGet;

        let mut store = self.store.lock().expect("store mutex");

        let entries = match crate::policy::undelivered_entries(&mut *store, &self.signing_key) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(%error, "redispatch_undelivered: policy read failed");
                return 0;
            }
        };
        if entries.is_empty() {
            return 0;
        }

        let now = crate::clock::mono_now();
        let Ok(reader) = store.reader() else {
            return 0;
        };

        let mut dispatched = 0usize;
        for entry in entries {
            // Per-entry cooldown.
            if let Some(prev) = self.last_dispatch_attempt.get(&entry.id) {
                if now.duration_since(*prev) < Self::UNDELIVERED_REDISPATCH_COOLDOWN {
                    continue;
                }
            }

            let Ok(cap_blob) = reader.get::<Blob<SimpleArchive>, SimpleArchive>(entry.latest_cap)
            else {
                continue;
            };
            let Ok(sig_blob) = reader.get::<Blob<SimpleArchive>, SimpleArchive>(entry.latest_sig)
            else {
                continue;
            };

            self.sender.deliver_cap(
                entry.subject.to_bytes(),
                cap_blob.bytes.clone(),
                sig_blob.bytes.clone(),
            );
            self.last_dispatch_attempt.insert(entry.id, now);
            dispatched += 1;
            tracing::debug!(
                subject = %hex::encode(entry.subject.to_bytes()),
                entry = ?entry.id,
                "redispatch_undelivered: re-sent OP_DELIVER_CAP"
            );
        }
        dispatched
    }

    /// Run one tick of the auto-renewal scan.
    ///
    /// Performs two pieces of work each tick:
    ///
    /// 1. **Redispatch undelivered entries.** For each renewal-policy
    ///    entry that's not yet been ack'd by its subject, re-send the
    ///    same `(latest_cap, latest_sig)` blobs via
    ///    `crate::channel::NetCommand::DeliverCap`, rate-limited per
    ///    entry by `Self::UNDELIVERED_REDISPATCH_COOLDOWN`. This is
    ///    what catches the case where the initial `team approve`
    ///    delivery failed (subject offline) and the subject comes back
    ///    later.
    ///
    /// 2. **Re-sign near-expiry entries.** For each entry whose current
    ///    cap upper bound falls within `renewal_window` of now, sign a
    ///    fresh cap+sig (using our team-cap as parent) and dispatch.
    ///    The policy entry is updated in lockstep, which also clears
    ///    any `delivered_at` so step (1) on the next tick picks the
    ///    fresh cap up for re-confirmation.
    ///
    /// Returns the total count of dispatches this tick (undelivered
    /// re-sends + fresh renewals). `0` on every tick after the swarm
    /// settles into steady state means the daemon is quiet.
    ///
    /// Designed to be called from `trible pile net sync`'s main loop
    /// alongside `refresh`. The 1-hour default window assumes a tick
    /// cadence well under that; tune both together for production
    /// deployments.
    pub fn renewal_tick(&mut self, renewal_window: hifitime::Duration) -> usize {
        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
        use triblespace_core::blob::{Blob, TryFromBlob};
        use triblespace_core::inline::encodings::hash::Handle;
        use triblespace_core::inline::{Inline, TryToInline};
        use triblespace_core::repo::BlobStoreGet;

        let redispatched = self.redispatch_undelivered();

        let mut store = self.store.lock().expect("store mutex");

        let entries =
            match crate::policy::renewable_within(&mut *store, &self.signing_key, renewal_window) {
                Ok(entries) => entries,
                Err(error) => {
                    tracing::warn!(%error, "renewal_tick: policy read failed");
                    return redispatched;
                }
            };
        if entries.is_empty() {
            return redispatched;
        }

        // Our own current cap is the parent for every renewal. If
        // we don't have one, we can't sign — log and bail.
        let (parent_cap_handle, parent_sig_handle) =
            match crate::policy::current_team_cap(&mut *store, &self.signing_key, self.team_root) {
                Ok(Some(pair)) => pair,
                Ok(None) => {
                    tracing::warn!(
                        renewable = entries.len(),
                        "renewal_tick: no team cap stored; cannot issue successors"
                    );
                    return redispatched;
                }
                Err(error) => {
                    tracing::warn!(%error, "renewal_tick: team-cap policy read failed");
                    return redispatched;
                }
            };

        let Ok(reader) = store.reader() else {
            tracing::warn!("renewal_tick: pile reader unavailable");
            return 0;
        };
        let Ok(parent_cap_blob) =
            reader.get::<Blob<SimpleArchive>, SimpleArchive>(parent_cap_handle)
        else {
            tracing::warn!("renewal_tick: parent cap blob missing");
            return 0;
        };
        let Ok(parent_sig_blob) =
            reader.get::<Blob<SimpleArchive>, SimpleArchive>(parent_sig_handle)
        else {
            tracing::warn!("renewal_tick: parent sig blob missing");
            return 0;
        };

        let mut dispatched = 0usize;
        for entry in entries {
            // Re-derive scope_facts from the previous cap blob —
            // policy entries carry only the scope_root id, not the
            // facts hanging off it.
            let Ok(prev_cap_blob) =
                reader.get::<Blob<SimpleArchive>, SimpleArchive>(entry.latest_cap)
            else {
                tracing::warn!(
                    entry = ?entry.id,
                    "renewal_tick: previous cap blob missing; skipping entry"
                );
                continue;
            };
            let Ok(prev_set): Result<triblespace_core::trible::TribleSet, _> =
                TryFromBlob::try_from_blob(prev_cap_blob)
            else {
                continue;
            };
            // Extract all tribles hanging off the scope_root entity.
            // pattern!() over the cap blob restricted to entities
            // whose entity-id == scope_root gives us the scope sub-graph.
            let scope_facts = extract_scope_subgraph(&prev_set, entry.scope);

            // Fresh expiry interval: [now, now + window * 2]. The
            // factor-of-two is a heuristic — we want the cap to cover
            // at least one more renewal cycle so missed ticks don't
            // immediately break the chain.
            let now = crate::clock::epoch_now();
            let new_upper = now + renewal_window * 2;
            let Ok(new_expiry) = (now, new_upper).try_to_inline() else {
                continue;
            };

            // Sign.
            let (new_cap, new_sig) = match triblespace_core::repo::capability::build_capability(
                &self.signing_key,
                entry.subject,
                Some((parent_cap_blob.clone(), parent_sig_blob.clone())),
                entry.scope,
                scope_facts,
                new_expiry,
            ) {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(
                        entry = ?entry.id,
                        error = ?e,
                        "renewal_tick: build_capability failed; skipping"
                    );
                    continue;
                }
            };

            let new_cap_handle: Inline<Handle<SimpleArchive>> = (&new_cap).get_handle();
            let new_sig_handle: Inline<Handle<SimpleArchive>> = (&new_sig).get_handle();

            // Persist locally — the next tick's policy update points
            // at these handles; the dispatch ships the bytes. Both
            // sites share the same refcounted `anybytes::Bytes`
            // backing the freshly-signed blob (clones are refcount
            // bumps, no byte-copy).
            let cap_bytes = new_cap.bytes.clone();
            let sig_bytes = new_sig.bytes.clone();
            if let Err(error) = store.put::<SimpleArchive, Blob<SimpleArchive>>(new_cap) {
                tracing::warn!(
                    entry = ?entry.id,
                    ?error,
                    "renewal_tick: failed to persist successor cap; not recording or dispatching"
                );
                continue;
            }
            if let Err(error) = store.put::<SimpleArchive, Blob<SimpleArchive>>(new_sig) {
                tracing::warn!(
                    entry = ?entry.id,
                    ?error,
                    "renewal_tick: failed to persist successor signature; not recording or dispatching"
                );
                continue;
            }

            // Publish the successor before dispatch. A crash can leave an
            // undelivered durable version (which the retry loop repairs), but
            // can never deliver a credential that policy forgot to record.
            match crate::policy::update_policy_entry(
                &mut *store,
                &self.signing_key,
                entry.id,
                new_expiry,
                new_cap_handle,
                new_sig_handle,
            ) {
                Ok(new_entry) => {
                    self.sender
                        .deliver_cap(entry.subject.to_bytes(), cap_bytes, sig_bytes);
                    self.last_dispatch_attempt
                        .insert(new_entry, crate::clock::mono_now());
                    dispatched += 1;
                    tracing::info!(
                        subject = %hex::encode(entry.subject.to_bytes()),
                        entry = ?new_entry,
                        predecessor = ?entry.id,
                        "renewal_tick: successor recorded and dispatched"
                    );
                }
                Err(error) => tracing::warn!(
                    entry = ?entry.id,
                    %error,
                    "renewal_tick: successor policy write failed; not dispatching"
                ),
            }
        }
        dispatched + redispatched
    }

    /// Lock and borrow the underlying store. Use for store-specific
    /// methods that aren't part of the storage traits (e.g.
    /// `Pile::flush`, `Yard::collect`, `WantStore::wants`).
    ///
    /// Writes through this borrow bypass the Peer's auto-publish and serving
    /// snapshot. In particular, after changing a pin used by branch-restricted
    /// authorization, drop the guard and call [`refresh`](Self::refresh) before
    /// treating a pin revocation as effective. Don't hold the guard across
    /// calls back into the Peer — its own methods take the same lock.
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
// to the inner store and then publish the new state. Local pin changes made
// explicitly through `Peer::store()` enter the branch-restricted capability
// view on the next `Peer::refresh()` and are never gossiped.

impl<S> BlobStorePut for Peer<S>
where
    S: BlobStore
        + BlobStorePut
        + CollectionGossipStore
        + CollectionStore
        + PinSnapshotSource
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
    S: BlobStore
        + BlobStorePut
        + CollectionGossipStore
        + CollectionStore
        + PinSnapshotSource
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
        grants: Vec<CollectionGossip>,
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

    impl CollectionGossipStore for TestStore {
        type GossipsError = Infallible;
        type GossipError = Infallible;
        type GossipIter<'a> = std::vec::IntoIter<Result<CollectionGossip, Infallible>>;

        fn gossips<'a>(&'a mut self) -> Result<Self::GossipIter<'a>, Self::GossipsError> {
            Ok(self
                .grants
                .iter()
                .copied()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn gossip(&mut self, grant: CollectionGossip) -> Result<(), Self::GossipError> {
            if !self.grants.contains(&grant) {
                self.grants.push(grant);
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

    #[test]
    fn grant_backed_snapshot_filters_then_sorts_and_deduplicates() {
        let author = SigningKey::from_bytes(&[7; 32]);
        let ungranted_author = SigningKey::from_bytes(&[8; 32]);
        let first_collection = collection(1);
        let second_collection = collection(2);
        let first = commit(&author, first_collection, 1);
        let second = commit(&author, first_collection, 2);
        let across_collection = commit(&author, second_collection, 3);
        let ungranted = commit(&ungranted_author, first_collection, 4);

        let mut invalid_grant =
            CollectionGossip::sign(&ungranted_author, first_collection).to_bytes();
        invalid_grant[127] ^= 1;
        let mut store = TestStore {
            records: vec![
                CollectionRecord::Commit(second),
                CollectionRecord::Commit(ungranted),
                CollectionRecord::Commit(first),
                CollectionRecord::Commit(second),
                CollectionRecord::Commit(across_collection),
            ],
            grants: vec![
                CollectionGossip::from_bytes(invalid_grant),
                CollectionGossip::sign(&author, second_collection),
                CollectionGossip::sign(&author, first_collection),
            ],
            flushes: 0,
        };

        let evidence = grant_backed_collection_evidence(&mut store).unwrap();
        let ids: Vec<_> = evidence.iter().map(|item| item.commit().id()).collect();
        let mut expected = vec![first.id(), second.id(), across_collection.id()];
        expected.sort();

        assert_eq!(ids, expected);
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(evidence.iter().all(|item| {
            item.commit().verify_strict().is_ok() && item.grant().verify_strict().is_ok()
        }));
    }

    #[test]
    fn one_gossip_drain_deduplicates_and_flushes_once() {
        let author = SigningKey::from_bytes(&[9; 32]);
        let collection = collection(3);
        let grant = CollectionGossip::sign(&author, collection);
        let first = CollectionCommitEvidence::new(grant, commit(&author, collection, 1)).unwrap();
        let second = CollectionCommitEvidence::new(grant, commit(&author, collection, 2)).unwrap();
        let mut store = TestStore::default();

        let counts =
            admit_incoming_collection_evidence(&mut store, vec![second, first, second]).unwrap();

        assert_eq!(counts.observed, 2);
        assert_eq!(counts.admitted, 2);
        assert_eq!(counts.denied, 0);
        assert_eq!(store.records.len(), 2);
        assert_eq!(store.grants, vec![grant]);
        assert_eq!(store.flushes, 1);
    }

    #[test]
    fn write_only_rejects_replication_data_but_keeps_capability_control() {
        let author = SigningKey::from_bytes(&[10; 32]);
        let collection = collection(4);
        let grant = CollectionGossip::sign(&author, collection);
        let evidence =
            CollectionCommitEvidence::new(grant, commit(&author, collection, 1)).unwrap();
        let bytes = Bytes::from(vec![1, 2, 3]);

        let cases = [
            (NetEvent::Blob(bytes.clone()), false),
            (NetEvent::CollectionEvidence(evidence), false),
            (
                NetEvent::CapRequest {
                    requester: [1; 32],
                    partial_cap_bytes: bytes.clone(),
                },
                true,
            ),
            (
                NetEvent::CapDelivered {
                    issuer: [2; 32],
                    cap_bytes: bytes.clone(),
                    sig_bytes: bytes,
                },
                true,
            ),
            (
                NetEvent::CapDeliveryConfirmed {
                    subject: [3; 32],
                    sig_handle: [4; 32],
                },
                true,
            ),
        ];

        for (event, expected) in cases {
            assert_eq!(
                accepts_incoming_event(SyncDirection::WriteOnly, &event),
                expected,
                "unexpected WriteOnly policy for {event:?}"
            );
        }
    }
}

/// Extract every trible whose entity is `scope_root` from `set`,
/// returning them as a fresh TribleSet. Used by `renewal_tick` to
/// reconstruct the scope-facts argument to `build_capability` from
/// the previous-cap blob — policy entries carry only the
/// `scope_root` id, not the facts hanging off it.
fn extract_scope_subgraph(
    set: &triblespace_core::trible::TribleSet,
    scope_root: triblespace_core::id::Id,
) -> triblespace_core::trible::TribleSet {
    let mut result = triblespace_core::trible::TribleSet::new();
    for trible in set.iter() {
        if *trible.e() == scope_root {
            result.insert(trible);
        }
    }
    result
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
