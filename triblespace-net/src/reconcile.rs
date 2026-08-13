//! Want-reconcile: service durable **wants**.
//!
//! Blob wants are durable demand/cache interest — "obtain this blob and keep it
//! while policy permits." Merge and derive wants are exact questions about
//! immutable collection receipts. Faculties and other processes append wants
//! to the shared pile out-of-band; a long-running sync daemon
//! (`trible pile net sync`) services that queue.
//!
//! Wants are independent from named [`PinStore`](triblespace_core::repo::PinStore)
//! branches. The reconciler never changes named pin state. Blob demand becomes
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
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::collection::{CollectionGossipStore, CollectionRecord, CollectionStore};
use triblespace_core::inline::Inline;
use triblespace_core::repo::{
    BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut, PinStore, StorageFlush, WantRequest,
    WantStore,
};

use crate::peer::Peer;
use crate::protocol::RawHash;

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
    /// Fetches actually issued this pass (`missing` minus the
    /// backoff-gated).
    pub attempted: usize,
    /// Wants satisfied this pass: fetched from the swarm and landed in
    /// the store.
    pub fetched: usize,
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
    states: HashMap<RawHash, WantState>,
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
            + CollectionGossipStore
            + CollectionStore
            + PinStore
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
        let mut blob_wants = Vec::new();
        let mut operation_wants = HashSet::new();
        for request in requests {
            match request {
                WantRequest::Blob { handle } => blob_wants.push(handle.raw),
                WantRequest::Merge { .. } | WantRequest::Derive { .. } => {
                    operation_wants.insert(request);
                }
            }
        }

        // A locally present exact receipt already answers an operation want.
        // Remote receipt discovery lands in the next protocol slice; until
        // then unanswered operation wants remain visibly pending rather than
        // allowing the daemon to report false quiescence.
        if !operation_wants.is_empty() {
            let local_answers = {
                let mut store = peer.store();
                match store.records() {
                    Ok(records) => records
                        .filter_map(Result::ok)
                        .filter_map(want_request_for_record)
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
            operation_wants.retain(|request| !local_answers.contains(request));
        }
        stats.pending = operation_wants.len();

        // ── Presence: which wants the local snapshot already serves ───
        // Peer::reader() runs refresh() (drains gossip, announces
        // external writes) and hands back a frozen local snapshot; the
        // sync get on it is local-only by design.
        let reader = match peer.reader() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = ?e, "reconcile: store reader unavailable; skipping pass");
                return stats;
            }
        };
        let missing: Vec<RawHash> = blob_wants
            .into_iter()
            .filter(|hash| {
                BlobStoreGet::get::<Bytes, UnknownBlob>(&reader, Inline::new(*hash)).is_err()
            })
            .collect();
        stats.missing = missing.len() + operation_wants.len();

        // Drop bookkeeping for wants no longer outstanding — satisfied
        // out-of-band (gossip landed the blob) or retracted.
        let missing_set: HashSet<RawHash> = missing.iter().copied().collect();
        self.states.retain(|hash, _| {
            let keep = missing_set.contains(hash);
            if !keep {
                tracing::info!(
                    hash = %hex::encode(&hash[..4]),
                    "reconcile: pending want resolved out-of-band"
                );
            }
            keep
        });

        // ── Fetch the missing wants (backoff-gated) ───────────────────
        for hash in missing {
            if let Some(st) = self.states.get(&hash) {
                if crate::clock::mono_now().duration_since(st.last_attempt) < st.backoff {
                    // Recently failed; wait out the backoff. Still a
                    // pending want — just not this pass's problem.
                    stats.pending += 1;
                    continue;
                }
            }

            stats.attempted += 1;
            match peer.fetch_blob_with_deadline(hash, self.fetch_budget).await {
                Some(bytes) => {
                    // Land the verified bytes (fetch_blob hash-checked
                    // them). The demand marker is
                    // already on record and now retains the blob — no
                    // pin state changes here.
                    if let Err(e) = peer.store().put::<UnknownBlob, Bytes>(Bytes::from(bytes)) {
                        tracing::warn!(
                            hash = %hex::encode(&hash[..4]),
                            error = ?e,
                            "reconcile: landing fetched blob failed; want stays pending"
                        );
                        stats.pending += 1;
                        continue;
                    }
                    if self.states.remove(&hash).is_some() {
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
                    stats.fetched += 1;
                }
                None => {
                    // Unavailable: nobody reachable served it. Normal —
                    // the want stays on record (the want), retried
                    // with backoff. Log once on the state change
                    // (became pending), not per retry.
                    stats.pending += 1;
                    let now = crate::clock::mono_now();
                    match self.states.entry(hash) {
                        Entry::Occupied(mut e) => {
                            let st = e.get_mut();
                            st.last_attempt = now;
                            st.backoff = (st.backoff * 2).min(self.max_backoff);
                        }
                        Entry::Vacant(e) => {
                            tracing::info!(
                                hash = %hex::encode(&hash[..4]),
                                "reconcile: want unavailable; pending (retried with backoff — not an error)"
                            );
                            e.insert(WantState {
                                last_attempt: now,
                                backoff: self.initial_backoff,
                            });
                        }
                    }
                }
            }
        }

        stats
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
            Some(WantRequest::derive(derive.source(), derive.target(), input))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use triblespace_core::collection::{CollectionDerive, CollectionMerge};

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
                collection, target, a, result,
            ))),
            Some(WantRequest::derive(collection, target, a))
        );
    }
}
