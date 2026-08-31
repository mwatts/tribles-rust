//! Service durable exact-content and collection-operation WANTs.
//!
//! Collection repair converges relevant records, including conflicting
//! MERGE/DERIVE receipts. This reconciler never claims global absence: an operation WANT is
//! satisfied iff at least one matching local receipt is visible; otherwise it
//! remains pending while periodic inventory sweeps continue. A routed
//! BlobInCollection(C,H) WANT validates C's resident descriptor policy and uses
//! exact collection-provider discovery without activating C. Bare Blob WANTs
//! are local retention intent and remain pending without a collection route;
//! this reconciler never guesses provenance from configured peers, active
//! collections, or payload bytes.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::collection::{
    CollectionRead, CollectionRecord, CollectionRecordSelector, CollectionStore,
};
use triblespace_core::repo::{
    BlobChildren, BlobStore, BlobStoreGet, CapabilityProofStore, SnapshotSource, StorageFlush,
    StoreRead, WantRequest, WantStore,
};

use crate::collection_activation::load_collection_policy;
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
    durable_blob_answers: HashSet<[u8; 32]>,
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
            + WantStore
            + StorageFlush
            + Send
            + 'static,
        S::Snapshot: StoreRead + BlobChildren,
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
            .filter(|request| request.blob_handle().is_some())
            .collect();
        let operation_wants: BTreeSet<_> = requests
            .iter()
            .copied()
            .filter(|request| {
                matches!(
                    request,
                    WantRequest::Merge { .. } | WantRequest::Derive { .. }
                )
            })
            .collect();

        // One native indexed union retains every conflicting answer. Empty is
        // only "not obtained yet", never proof that no answer exists.
        let selectors: BTreeSet<_> = operation_wants
            .iter()
            .copied()
            .map(CollectionRecordSelector::Operation)
            .collect();
        let snapshot = match peer.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "store snapshot unavailable; skipping reconcile pass"
                );
                stats.missing = operation_wants.len() + blob_wants.len();
                stats.pending = stats.missing;
                return stats;
            }
        };
        let answered_operations: HashSet<_> = {
            match snapshot.select_records(&selectors) {
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

        let wanted_blob_handles: HashSet<_> = blob_wants
            .iter()
            .filter_map(|request| request.blob_handle().map(|handle| handle.raw))
            .collect();
        let visible_blobs: HashSet<_> = wanted_blob_handles
            .iter()
            .copied()
            .filter(|handle| {
                snapshot
                    .get::<Bytes, UnknownBlob>(triblespace_core::inline::Inline::new(*handle))
                    .is_ok()
            })
            .collect();

        self.durable_blob_answers.retain(|handle| {
            wanted_blob_handles.contains(handle) && visible_blobs.contains(handle)
        });
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
            .filter(|request| {
                !self
                    .durable_blob_answers
                    .contains(&request.blob_handle().expect("blob WANT").raw)
            })
            .collect();
        stats.missing = missing_operations + missing_blobs.len();
        stats.pending = missing_operations;

        let outstanding: HashSet<_> = missing_blobs
            .iter()
            .copied()
            .filter(|request| matches!(request, WantRequest::BlobInCollection { .. }))
            .collect();
        self.states
            .retain(|request, _| outstanding.contains(request));

        let mut missing_by_handle = BTreeMap::<[u8; 32], Vec<WantRequest>>::new();
        for request in missing_blobs {
            missing_by_handle
                .entry(request.blob_handle().expect("blob WANT").raw)
                .or_default()
                .push(request);
        }

        for (handle, requests) in missing_by_handle {
            let started = crate::clock::mono_now();
            let mut fulfilled = false;
            for request in requests.iter().copied() {
                let WantRequest::BlobInCollection { collection, .. } = request else {
                    // Bare Blob(H) is a distinct local-only intent. Another
                    // route for this H may still satisfy it below.
                    continue;
                };
                if self.states.get(&request).is_some_and(|state| {
                    crate::clock::mono_now().duration_since(state.last_attempt) < state.backoff
                }) {
                    continue;
                }
                let policy = match load_collection_policy(&snapshot, collection) {
                    Ok(policy) => policy,
                    Err(error) => {
                        tracing::debug!(
                            collection = %hex::encode(collection.raw),
                            ?error,
                            "routed WANT descriptor is absent or invalid; keeping request pending"
                        );
                        continue;
                    }
                };
                let remaining = self
                    .fetch_budget
                    .saturating_sub(crate::clock::mono_now().duration_since(started));
                if remaining.is_zero() {
                    break;
                }
                stats.attempted += 1;
                match peer
                    .fetch_collection_blob_with_policy_and_deadline(
                        collection, policy, handle, remaining,
                    )
                    .await
                {
                    Some(bytes) => {
                        let landing = {
                            let mut store = peer.store();
                            match store.put::<UnknownBlob, Bytes>(Bytes::from(bytes)) {
                                Ok(actual) if actual.raw == handle => store
                                    .flush()
                                    .map_err(|error| format!("flush failed: {error:?}")),
                                Ok(_) => Err("blob store returned a different handle".to_owned()),
                                Err(error) => Err(format!("put failed: {error:?}")),
                            }
                        };
                        if let Err(error) = landing {
                            tracing::warn!(%error, "wanted blob landing failed; WANT remains pending");
                            self.record_unavailable(request);
                            continue;
                        }
                        self.durable_blob_answers.insert(handle);
                        for request in &requests {
                            self.states.remove(request);
                        }
                        stats.fulfilled += requests.len();
                        peer.refresh();
                        fulfilled = true;
                        break;
                    }
                    None => {
                        self.record_unavailable(request);
                    }
                }
            }
            if !fulfilled {
                stats.pending += requests.len();
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
            Some(WantRequest::derive(derive.collection(), derive.input()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use iroh_base::EndpointId;
    use triblespace_core::blob::encodings::UnknownBlob;
    use triblespace_core::collection::{CollectionDerive, CollectionMerge};
    use triblespace_core::inline::Inline;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::repo::memoryrepo::MemoryRepo;

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

    #[tokio::test]
    async fn bare_and_unvalidated_routed_wants_stay_pending_without_network_attempts() {
        let key = SigningKey::from_bytes(&[31; 32]);
        let endpoint = EndpointId::from_bytes(&key.verifying_key().to_bytes()).unwrap();
        let mut store = MemoryRepo::default();
        let wanted = Inline::<Handle<UnknownBlob>>::new([41; 32]);
        let malformed = store
            .put::<UnknownBlob, _>(Bytes::from_source(b"not a descriptor".to_vec()))
            .unwrap();
        let bare = WantRequest::blob(wanted);
        let absent_route = WantRequest::blob_in_collection(Inline::new([42; 32]), wanted);
        let invalid_route = WantRequest::blob_in_collection(
            Inline::new(malformed.raw),
            Inline::<Handle<UnknownBlob>>::new([43; 32]),
        );
        for request in [bare, absent_route, invalid_route] {
            store.want(request).unwrap();
        }
        store.flush().unwrap();

        // Keep the host capability deliberately uninstalled. Any guessed
        // network route would block here until the timeout catches it.
        let (sender, receiver, _wiring) = crate::host::wire(endpoint);
        let mut peer = Peer::with_wiring(
            store,
            crate::inventory::ReconcileQos::default(),
            sender,
            receiver,
        );
        let mut reconciler = Reconciler::new();
        let stats = tokio::time::timeout(Duration::from_millis(50), reconciler.tick(&mut peer))
            .await
            .expect("unroutable WANTs must not wait for a network capability");

        assert_eq!(
            stats,
            ReconcileStats {
                wants: 3,
                missing: 3,
                attempted: 0,
                fulfilled: 0,
                pending: 3,
            }
        );
        assert!(reconciler.states.is_empty());
    }
}
