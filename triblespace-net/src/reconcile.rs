//! Service durable exact-content and collection-operation WANTs.
//!
//! Broad team inventory already converges every collection record, including
//! all conflicting MERGE/DERIVE receipts. This reconciler therefore performs
//! no second record RPC and never claims global absence: an operation WANT is
//! satisfied iff at least one matching local receipt is visible; otherwise it
//! remains pending while periodic inventory sweeps continue. Blob WANTs retain
//! their explicit DHT-provider lookup, exact authenticated fetch, and
//! exponential retry backoff.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::collection::{CollectionRecord, CollectionRecordSelector, CollectionStore};
use triblespace_core::repo::{
    ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta, CapabilityProofStore, PeerStore,
    StorageFlush, StoreRevision, StoreScope, WantRequest, WantStore,
};

use crate::peer::Peer;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileStats {
    pub wants: usize,
    pub missing: usize,
    pub attempted: usize,
    pub fulfilled: usize,
    pub pending: usize,
}

struct WantState {
    last_attempt: crate::clock::Mono,
    backoff: Duration,
}

/// Retry state only. Durable demand and all answers remain in the store.
pub struct Reconciler {
    states: HashMap<WantRequest, WantState>,
    durable_blob_answers: HashSet<WantRequest>,
    initial_backoff: Duration,
    max_backoff: Duration,
    fetch_budget: Duration,
}

pub const RECONCILE_FETCH_DEADLINE: Duration = Duration::from_secs(30);

impl Default for Reconciler {
    fn default() -> Self {
        Self::new()
    }
}

impl Reconciler {
    pub fn new() -> Self {
        Self::with_backoff(crate::RETRY_BACKOFF_BASE, crate::RETRY_BACKOFF_CAP)
    }

    pub fn with_backoff(initial: Duration, max: Duration) -> Self {
        Self {
            states: HashMap::new(),
            durable_blob_answers: HashSet::new(),
            initial_backoff: initial,
            max_backoff: max,
            fetch_budget: RECONCILE_FETCH_DEADLINE,
        }
    }

    pub fn with_fetch_budget(mut self, budget: Duration) -> Self {
        self.fetch_budget = budget;
        self
    }

    pub async fn tick<S>(&mut self, peer: &mut Peer<S>) -> ReconcileStats
    where
        S: BlobStore
            + CollectionStore
            + CapabilityProofStore
            + PeerStore
            + ArtifactOfferStore
            + StoreScope
            + WantStore
            + StorageFlush
            + StoreRevision
            + Send
            + 'static,
        S::Reader: BlobStoreMeta,
    {
        let mut stats = ReconcileStats::default();

        // This is also the explicit external-Pile reobservation and inventory
        // admission boundary.
        peer.refresh();
        let requests: Vec<WantRequest> = {
            let mut store = peer.store();
            match store.wants() {
                Ok(wants) => wants.filter_map(Result::ok).collect(),
                Err(error) => {
                    tracing::warn!(?error, "WANT enumeration failed; skipping reconcile pass");
                    return stats;
                }
            }
        };
        stats.wants = requests.len();

        let blob_wants: BTreeSet<_> = requests
            .iter()
            .copied()
            .filter(|request| matches!(request, WantRequest::Blob { .. }))
            .collect();
        let operation_wants: BTreeSet<_> = requests
            .iter()
            .copied()
            .filter(|request| !matches!(request, WantRequest::Blob { .. }))
            .collect();

        // One native indexed union retains every conflicting answer. Empty is
        // only "not obtained yet", never proof that no answer exists.
        let selectors: BTreeSet<_> = operation_wants
            .iter()
            .copied()
            .map(CollectionRecordSelector::Operation)
            .collect();
        let answered_operations: HashSet<_> = {
            let mut store = peer.store();
            match store.select_records(&selectors) {
                Ok(records) => records
                    .into_iter()
                    .filter_map(want_request_for_record)
                    .collect(),
                Err(error) => {
                    tracing::warn!(?error, "operation receipt selection failed");
                    HashSet::new()
                }
            }
        };
        let missing_operations = operation_wants
            .iter()
            .filter(|request| !answered_operations.contains(request))
            .count();

        let reader = match peer.reader() {
            Ok(reader) => reader,
            Err(error) => {
                tracing::warn!(?error, "blob reader unavailable; skipping reconcile pass");
                stats.missing = missing_operations + blob_wants.len();
                stats.pending = stats.missing;
                return stats;
            }
        };
        let visible_blobs: HashSet<_> = blob_wants
            .iter()
            .copied()
            .filter(|request| {
                let WantRequest::Blob { handle } = request else {
                    unreachable!()
                };
                reader.get::<Bytes, UnknownBlob>(*handle).is_ok()
            })
            .collect();
        drop(reader);

        self.durable_blob_answers
            .retain(|request| blob_wants.contains(request) && visible_blobs.contains(request));
        let newly_visible: HashSet<_> = visible_blobs
            .difference(&self.durable_blob_answers)
            .copied()
            .collect();
        if !newly_visible.is_empty() {
            let durable = peer.store().flush();
            match durable {
                Ok(()) => self.durable_blob_answers.extend(newly_visible),
                Err(error) => tracing::warn!(
                    ?error,
                    "visible wanted blobs are not durable; keeping them pending"
                ),
            }
        }

        let missing_blobs: Vec<_> = blob_wants
            .iter()
            .copied()
            .filter(|request| !self.durable_blob_answers.contains(request))
            .collect();
        stats.missing = missing_operations + missing_blobs.len();
        stats.pending = missing_operations;

        let outstanding: HashSet<_> = missing_blobs.iter().copied().collect();
        self.states
            .retain(|request, _| outstanding.contains(request));

        for request in missing_blobs {
            if self.states.get(&request).is_some_and(|state| {
                crate::clock::mono_now().duration_since(state.last_attempt) < state.backoff
            }) {
                stats.pending += 1;
                continue;
            }
            let WantRequest::Blob { handle } = request else {
                unreachable!()
            };
            stats.attempted += 1;
            match peer
                .fetch_blob_with_deadline(handle.raw, self.fetch_budget)
                .await
            {
                Some(bytes) => {
                    let landing = {
                        let mut store = peer.store();
                        match store.put::<UnknownBlob, Bytes>(Bytes::from(bytes)) {
                            Ok(actual) if actual.raw == handle.raw => store
                                .flush()
                                .map_err(|error| format!("flush failed: {error:?}")),
                            Ok(_) => Err("blob store returned a different handle".to_owned()),
                            Err(error) => Err(format!("put failed: {error:?}")),
                        }
                    };
                    if let Err(error) = landing {
                        tracing::warn!(%error, "wanted blob landing failed; WANT remains pending");
                        stats.pending += 1;
                        self.record_unavailable(request);
                        continue;
                    }
                    self.durable_blob_answers.insert(request);
                    self.states.remove(&request);
                    stats.fulfilled += 1;
                    peer.refresh();
                }
                None => {
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
    fn receipts_project_to_exact_input_only_wants() {
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
