//! Want-reconcile: service durable **wants**.
//!
//! Blob wants are durable demand/cache interest — "obtain this blob and keep it
//! while policy permits." Merge and derive wants are exact questions about
//! immutable collection receipts. Faculties and other processes append wants
//! to the shared pile out-of-band; a long-running sync daemon
//! (`trible pile net sync`) services that queue.
//!
//! Wants are independent from legacy named pins. The reconciler never changes
//! legacy pin state. Blob demand becomes
//! cache-retention interest after bytes land; operation wants are fulfilled by
//! native collection records and never become blob roots.
//! - **"Absent" is always "not obtained yet", never definitely-absent**
//!   — existence is semidecidable. A want that can't be satisfied stays
//!   pending and is retried with backoff; it is NOT an error and NOT
//!   dropped. The want stays on record until the blob lands or
//!   someone unwants it.
//! - A want whose blob is already present is a retention marker,
//!   not an outstanding want — the presence diff filters it out.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::collection::{CollectionRecord, CollectionStore};
use triblespace_core::repo::{
    BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut, PinSnapshotSource, StorageFlush,
    WantRequest, WantStore,
};

use crate::peer::Peer;

/// Counters from one reconcile pass — the observable a sync daemon
/// surfaces (trace lines, `--quiescent-for` scripts) so lazy progress
/// is legible from the outside.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileStats {
    /// Exact requests seen this pass — the full LWW-resolved want set.
    pub wants: usize,
    /// Requests whose blob or exact operation receipt was absent locally at
    /// the start of the pass.
    pub missing: usize,
    /// Network attempts actually issued this pass (`missing` minus those
    /// backoff-gated), counted per wanted question rather than per peer.
    pub attempted: usize,
    /// Wants satisfied this pass: blob bytes or exact operation receipts
    /// fetched from peers and durably landed in the store.
    pub fulfilled: usize,
    /// Wants still outstanding after the pass. **Normal, not an error**
    /// — they stay on record (the want) and are retried with
    /// backoff on later passes.
    pub pending: usize,
}

/// Per-want retry bookkeeping. A state exists only while the want is
/// outstanding *and* has failed at least once — a want satisfied on its
/// first attempt never allocates one. In-memory only: it rebuilds
/// naturally (first retry immediate) if the daemon restarts, while the
/// wants themselves live durably in the store as wants.
struct WantState {
    /// When the last fetch attempt resolved (Unavailable). Read through
    /// [`crate::clock`] so simulated runs back off in virtual time.
    last_attempt: crate::clock::Mono,
    /// Current retry delay; doubles per failure up to the cap.
    backoff: Duration,
}

/// Drives the want-reconcile loop over a [`Peer`]. Owns only the retry
/// bookkeeping — the wants themselves are durable store state, so
/// dropping/recreating a `Reconciler` loses nothing but the
/// backoff timers.
pub struct Reconciler {
    states: HashMap<WantRequest, WantState>,
    /// Operation questions for which this process has both a durable local
    /// answer and one complete sweep of the configured peers. This is
    /// deliberately in-memory: after restart we sweep again rather than
    /// mistake a partially persisted conflict response for complete evidence.
    completed_operation_sweeps: HashSet<WantRequest>,
    /// Operation questions whose currently visible local receipts crossed the
    /// durability barrier in this process. Kept separate from sweep
    /// completeness so an offline peer causes retries, not repeated fsyncs.
    durable_operation_answers: HashSet<WantRequest>,
    /// Blob wants observed behind this process's durability barrier. A fetched
    /// blob whose flush fails stays outside this set, so later ticks retry the
    /// barrier rather than mistaking process-local visibility for persistence.
    durable_blob_answers: HashSet<WantRequest>,
    initial_backoff: Duration,
    max_backoff: Duration,
    /// End-to-end budget per fetch attempt. Background work, so more
    /// generous than the interactive default
    /// ([`crate::host::INTERACTIVE_FETCH_DEADLINE`]) — nobody is
    /// blocked on a reconcile tick, and a slow multi-provider walk is
    /// worth finishing. Still bounded: an expired budget resolves
    /// Unavailable and the want retries with backoff on a later pass.
    fetch_budget: Duration,
}

/// Default per-fetch budget for background reconcile ticks.
pub const RECONCILE_FETCH_DEADLINE: Duration = Duration::from_secs(30);

impl Default for Reconciler {
    fn default() -> Self {
        Self::new()
    }
}

impl Reconciler {
    /// Default backoff: the crate-shared retry bounds — first retry
    /// `RETRY_BACKOFF_BASE` after a failed attempt, doubling per failure
    /// to the `RETRY_BACKOFF_CAP`. Per-fetch budget defaults to
    /// [`RECONCILE_FETCH_DEADLINE`].
    pub fn new() -> Self {
        Self::with_backoff(crate::RETRY_BACKOFF_BASE, crate::RETRY_BACKOFF_CAP)
    }

    /// Custom backoff bounds — `initial` after the first failure,
    /// doubling to at most `max`.
    pub fn with_backoff(initial: Duration, max: Duration) -> Self {
        Self {
            states: HashMap::new(),
            completed_operation_sweeps: HashSet::new(),
            durable_operation_answers: HashSet::new(),
            durable_blob_answers: HashSet::new(),
            initial_backoff: initial,
            max_backoff: max,
            fetch_budget: RECONCILE_FETCH_DEADLINE,
        }
    }

    /// Override the end-to-end budget each fetch attempt gets.
    pub fn with_fetch_budget(mut self, budget: Duration) -> Self {
        self.fetch_budget = budget;
        self
    }

    /// One reconcile pass.
    ///
    /// 1. Enumerate the wants: `wants()` on the wrapped store. The
    ///    store refreshes itself first (a `Pile` re-scans the file), so
    ///    want records appended by OTHER processes since the last
    ///    tick become visible here.
    /// 2. Diff against presence: take a reader (which also runs
    ///    [`Peer::refresh`] — freshly-gossiped blobs count as present)
    ///    and keep the wants whose blob the local snapshot can't serve.
    /// 3. For each missing want not gated by its backoff timer, drive
    ///    the Peer's swarm fetch and land the verified bytes in the
    ///    store. Failures back off exponentially and are logged once
    ///    per state change (want became pending / pending want
    ///    resolved), not per retry.
    ///
    /// A pass with unsatisfiable wants completes in bounded time (the
    /// fetch resolves Unavailable on the DHT deadline); the wants stay
    /// pending — that is their normal state until a holder is
    /// reachable, never an error.
    pub async fn tick<S>(&mut self, peer: &mut Peer<S>) -> ReconcileStats
    where
        S: BlobStore
            + BlobStorePut
            + CollectionStore
            + PinSnapshotSource
            + WantStore
            + StorageFlush
            + Send
            + 'static,
        S::Reader: BlobStoreMeta,
    {
        let mut stats = ReconcileStats::default();

        // ── Wants: the LWW-resolved want set ──────────────────────
        let requests: Vec<WantRequest> = {
            let mut store = peer.store();
            match store.wants() {
                Ok(iter) => iter.filter_map(Result::ok).collect(),
                Err(e) => {
                    tracing::warn!(error = ?e, "reconcile: wants enumeration failed; skipping pass");
                    return stats;
                }
            }
        };
        stats.wants = requests.len();
        let mut blob_wants = BTreeSet::new();
        let mut operation_wants = BTreeSet::new();
        for request in requests {
            match request {
                WantRequest::Blob { .. } => {
                    blob_wants.insert(request);
                }
                WantRequest::Merge { .. } | WantRequest::Derive { .. } => {
                    operation_wants.insert(request);
                }
            }
        }

        // A visible exact receipt is useful evidence, but does not prove that
        // the configured-peer sweep which found it completed. Flush local
        // answers before use, then require one complete sweep per Reconciler
        // lifetime. After a crash we intentionally probe again: this prevents
        // one surviving record from a partially written conflict response
        // from masquerading as complete evidence.
        self.completed_operation_sweeps
            .retain(|request| operation_wants.contains(request));
        self.durable_operation_answers
            .retain(|request| operation_wants.contains(request));
        operation_wants.retain(|request| !self.completed_operation_sweeps.contains(request));
        let mut durable_local_answers = self.durable_operation_answers.clone();
        if !operation_wants.is_empty() {
            let locally_visible_answers = {
                let mut store = peer.store();
                match store.records() {
                    Ok(records) => records
                        .filter_map(Result::ok)
                        .filter_map(want_request_for_record)
                        .filter(|request| operation_wants.contains(request))
                        .filter(|request| !self.durable_operation_answers.contains(request))
                        .collect::<HashSet<_>>(),
                    Err(error) => {
                        tracing::warn!(
                            error = ?error,
                            "reconcile: collection receipt enumeration failed; operation wants stay pending"
                        );
                        HashSet::new()
                    }
                }
            };
            if !locally_visible_answers.is_empty() {
                let durable = {
                    let mut store = peer.store();
                    match store.flush() {
                        Ok(()) => true,
                        Err(error) => {
                            tracing::warn!(
                                error = ?error,
                                count = locally_visible_answers.len(),
                                "reconcile: local collection receipts are visible but not durable; operation wants stay pending"
                            );
                            false
                        }
                    }
                };
                if durable {
                    self.durable_operation_answers
                        .extend(locally_visible_answers.iter().copied());
                    durable_local_answers.extend(locally_visible_answers);
                }
            }
        }
        self.durable_blob_answers
            .retain(|request| blob_wants.contains(request));

        // ── Presence: which wants the local snapshot already serves ───
        // Peer::reader() runs refresh() (drains gossip, announces
        // external writes) and hands back a frozen local snapshot; the
        // sync get on it is local-only by design.
        let reader = match peer.reader() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = ?e, "reconcile: store reader unavailable; skipping pass");
                stats.missing = operation_wants.len() + blob_wants.len();
                stats.pending = stats.missing;
                return stats;
            }
        };
        let locally_visible_blobs: HashSet<WantRequest> = blob_wants
            .iter()
            .copied()
            .filter(|request| {
                let WantRequest::Blob { handle } = request else {
                    unreachable!("blob_wants contains only Blob requests")
                };
                BlobStoreGet::get::<Bytes, UnknownBlob>(&reader, *handle).is_ok()
            })
            .collect();
        drop(reader);
        self.durable_blob_answers
            .retain(|request| locally_visible_blobs.contains(request));
        let new_local_blobs: HashSet<WantRequest> = locally_visible_blobs
            .difference(&self.durable_blob_answers)
            .copied()
            .collect();
        if !new_local_blobs.is_empty() {
            let durable = {
                let mut store = peer.store();
                match store.flush() {
                    Ok(()) => true,
                    Err(error) => {
                        tracing::warn!(
                            error = ?error,
                            count = new_local_blobs.len(),
                            "reconcile: local blobs are visible but not durable; blob wants stay pending"
                        );
                        false
                    }
                }
            };
            if durable {
                self.durable_blob_answers.extend(new_local_blobs);
            }
        }
        let missing_blobs: Vec<WantRequest> = blob_wants
            .iter()
            .filter(|request| !self.durable_blob_answers.contains(request))
            .copied()
            .collect();
        stats.missing = missing_blobs.len()
            + operation_wants
                .iter()
                .filter(|request| !durable_local_answers.contains(request))
                .count();

        // Drop bookkeeping for wants no longer outstanding — satisfied
        // out-of-band (gossip landed the blob) or retracted.
        let outstanding: HashSet<WantRequest> = missing_blobs
            .iter()
            .copied()
            .chain(operation_wants.iter().copied())
            .collect();
        self.states.retain(|request, _| {
            let keep = outstanding.contains(request);
            if !keep {
                tracing::info!(?request, "reconcile: pending want resolved out-of-band");
            }
            keep
        });

        // ── Discover exact operation receipts (backoff-gated) ────────
        // Probe all configured peers before admitting anything. Distinct
        // outputs are retained as conflicts; all discovered receipts from
        // this pass share one durability flush.
        let mut discovered = Vec::new();
        for request in operation_wants {
            if let Some(st) = self.states.get(&request) {
                if crate::clock::mono_now().duration_since(st.last_attempt) < st.backoff {
                    stats.pending += 1;
                    continue;
                }
            }

            stats.attempted += 1;
            let probe = peer
                .fetch_collection_operation_receipts_with_deadline(request, self.fetch_budget)
                .await;
            if probe.receipts.is_empty()
                && probe.complete
                && durable_local_answers.contains(&request)
            {
                self.states.remove(&request);
                self.completed_operation_sweeps.insert(request);
            } else if probe.receipts.is_empty() {
                stats.pending += 1;
                self.record_unavailable(request);
            } else {
                discovered.push((request, probe.receipts, probe.complete));
            }
        }

        if !discovered.is_empty() {
            let landing = {
                let mut store = peer.store();
                let mut failed = None;
                'requests: for (_, receipts, _) in &discovered {
                    for receipt in receipts {
                        if let Err(error) = store.insert(*receipt) {
                            failed = Some(format!("insert failed: {error:?}"));
                            break 'requests;
                        }
                    }
                }
                if failed.is_none()
                    && let Err(error) = store.flush()
                {
                    failed = Some(format!("flush failed: {error:?}"));
                }
                failed
            };
            match landing {
                None => {
                    // Make the freshly durable records visible to remote
                    // receipt requests before reporting local fulfillment.
                    peer.refresh();
                    for (request, _, complete) in discovered {
                        self.durable_operation_answers.insert(request);
                        if complete {
                            self.states.remove(&request);
                            self.completed_operation_sweeps.insert(request);
                            stats.fulfilled += 1;
                        } else {
                            // Partial evidence is useful and durable, but an
                            // incomplete configured-peer sweep cannot
                            // discharge the question: a recovering peer may
                            // still hold a conflicting result.
                            self.completed_operation_sweeps.remove(&request);
                            stats.pending += 1;
                            self.record_unavailable(request);
                        }
                    }
                }
                Some(error) => {
                    tracing::warn!(%error, "reconcile: landing collection receipts failed");
                    for (request, _, _) in discovered {
                        self.durable_operation_answers.remove(&request);
                        self.completed_operation_sweeps.remove(&request);
                        stats.pending += 1;
                        self.record_unavailable(request);
                    }
                }
            }
        }

        // ── Fetch missing blobs (backoff-gated) ──────────────────────
        for request in missing_blobs {
            let WantRequest::Blob { handle } = request else {
                unreachable!("missing_blobs contains only Blob requests")
            };
            let hash = handle.raw;
            if let Some(st) = self.states.get(&request) {
                if crate::clock::mono_now().duration_since(st.last_attempt) < st.backoff {
                    stats.pending += 1;
                    continue;
                }
            }

            stats.attempted += 1;
            match peer.fetch_blob_with_deadline(hash, self.fetch_budget).await {
                Some(bytes) => {
                    // Land the verified bytes (fetch_blob hash-checked
                    // them) and cross the same durability barrier as
                    // collection receipts before reporting fulfillment. The
                    // demand marker remains on record and retains the blob;
                    // no pin state changes here.
                    let landing = {
                        let mut store = peer.store();
                        match store.put::<UnknownBlob, Bytes>(Bytes::from(bytes)) {
                            Ok(_) => store
                                .flush()
                                .map_err(|error| format!("flush failed: {error:?}")),
                            Err(error) => Err(format!("put failed: {error:?}")),
                        }
                    };
                    if let Err(error) = landing {
                        tracing::warn!(
                            hash = %hex::encode(&hash[..4]),
                            %error,
                            "reconcile: landing fetched blob failed; want stays pending"
                        );
                        stats.pending += 1;
                        self.record_unavailable(request);
                        continue;
                    }
                    self.durable_blob_answers.insert(request);
                    if self.states.remove(&request).is_some() {
                        // State change: a want previously logged as
                        // pending has been satisfied.
                        tracing::info!(
                            hash = %hex::encode(&hash[..4]),
                            "reconcile: pending want fetched"
                        );
                    } else {
                        tracing::debug!(
                            hash = %hex::encode(&hash[..4]),
                            "reconcile: want fetched"
                        );
                    }
                    stats.fulfilled += 1;
                }
                None => {
                    // Unavailable: nobody reachable served it. Normal —
                    // the want stays on record (the want), retried
                    // with backoff. Log once on the state change
                    // (became pending), not per retry.
                    stats.pending += 1;
                    self.record_unavailable(request);
                }
            }
        }

        stats
    }

    fn record_unavailable(&mut self, request: WantRequest) {
        let now = crate::clock::mono_now();
        match self.states.entry(request) {
            Entry::Occupied(mut entry) => {
                let state = entry.get_mut();
                state.last_attempt = now;
                state.backoff = (state.backoff * 2).min(self.max_backoff);
            }
            Entry::Vacant(entry) => {
                tracing::info!(
                    ?request,
                    "reconcile: want unavailable; pending (retried with backoff — not an error)"
                );
                entry.insert(WantState {
                    last_attempt: now,
                    backoff: self.initial_backoff,
                });
            }
        }
    }
}

fn want_request_for_record(record: CollectionRecord) -> Option<WantRequest> {
    match record {
        CollectionRecord::Commit(_) => None,
        CollectionRecord::Merge(merge) => {
            let (low, high) = merge.inputs();
            Some(WantRequest::merge(merge.collection(), low, high))
        }
        CollectionRecord::Derive(derive) => {
            let (input, _) = derive.mapping();
            Some(WantRequest::derive(derive.target(), input))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use triblespace_core::collection::{CollectionDerive, CollectionMerge};
    use triblespace_core::inline::Inline;

    #[test]
    fn exact_receipts_project_to_their_input_only_want_keys() {
        let collection = Inline::new([1; 32]);
        let target = Inline::new([2; 32]);
        let a = Inline::new([3; 32]);
        let b = Inline::new([4; 32]);
        let result = Inline::new([5; 32]);

        assert_eq!(
            want_request_for_record(CollectionRecord::Merge(CollectionMerge::new(
                collection, b, a, result,
            ))),
            Some(WantRequest::merge(collection, a, b))
        );
        assert_eq!(
            want_request_for_record(CollectionRecord::Derive(CollectionDerive::new(
                target, a, result,
            ))),
            Some(WantRequest::derive(target, a))
        );
    }
}
