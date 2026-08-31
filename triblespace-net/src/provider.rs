//! Exact soft-state provider leases for collections and bearer blobs.
//!
//! Collections and resident blob handles map to separate opaque full-width
//! rendezvous keys. Providers renew them at their K closest DHT nodes;
//! directory nodes receive neither raw collection handles nor raw blob handles.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use triblespace_core::collection::CollectionHandle;
use triblespace_core::patch::{
    Entry as PatchEntry, IdentitySchema, PATCH, PATCHIntoOrderedIterator,
};

use crate::bearer::{BearerLocatorIndex, blob_locator};
use crate::clock::Mono;
use crate::transport::PeerId;

/// Opaque rendezvous key for one exact collection or bearer identity.
pub(crate) type ProviderKey = [u8; 32];
pub(crate) type ProviderToken = [u8; 32];
type ProviderIdentity = [u8; 32];
const PROVIDER_TOKEN_CONTEXT: &[u8] = b"triblespace.net/provider-token/v1\0";

/// Receiver-chosen lifetime of one exact provider lease.
pub(crate) const PROVIDER_LEASE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
/// Complete one deterministic renewal traversal within one third of a lease.
/// The first sweep therefore finishes with at least eight hours of latency
/// margin before even its last initial lease can expire.
pub(crate) const PROVIDER_RENEWAL_PERIOD: Duration = Duration::from_secs(8 * 60 * 60);
/// Maximum fan-out retained and returned for one exact rendezvous key.
pub(crate) const MAX_PROVIDERS_PER_KEY: usize = 64;
const _: () = assert!(MAX_PROVIDERS_PER_KEY <= u8::MAX as usize);
/// Receiver-local aggregate soft bound. Exact-key responsibility decides
/// which membership survives when a full shard receives a closer key.
const MAX_PROVIDER_MEMBERSHIPS: usize = 1 << 24;

/// Bound opportunistic expiry reclamation performed by one RPC.
const MAX_EXPIRED_PROVIDER_MEMBERSHIPS_PER_CALL: usize = 64;

/// Derive the opaque provider rendezvous key for one collection session.
pub(crate) fn collection_provider_key(collection: CollectionHandle) -> ProviderKey {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/collection-provider-key/v1");
    hasher.update(&collection.raw);
    *hasher.finalize().as_bytes()
}

/// Endpoint-bound directory proof for one opaque rendezvous key.
///
/// `identity` is H for an exact blob lease and C for a collection-participant
/// hint. Keying by H makes the blob token a proof of bearer-handle knowledge;
/// C need not be secret. Including the already domain-separated rendezvous key
/// keeps both lease roles distinct without storing a second per-H token trie.
pub(crate) fn provider_lease_token(
    identity: [u8; 32],
    key: ProviderKey,
    provider: PeerId,
) -> ProviderToken {
    let mut hasher = blake3::Hasher::new_keyed(&identity);
    hasher.update(PROVIDER_TOKEN_CONTEXT);
    hasher.update(&key);
    hasher.update(&provider);
    *hasher.finalize().as_bytes()
}

pub(crate) fn collection_provider_token(identity: [u8; 32], provider: PeerId) -> ProviderToken {
    provider_lease_token(
        identity,
        collection_provider_key(CollectionHandle::new(identity)),
        provider,
    )
}

pub(crate) fn blob_provider_token(identity: [u8; 32], provider: PeerId) -> ProviderToken {
    provider_lease_token(identity, blob_locator(identity), provider)
}

/// Canonical exact publication set for one serving snapshot. Values retain the
/// hidden identity (H or C); endpoint-bound tokens are derived only when the
/// publisher schedules a key.
type ProviderLeasePatch = PATCH<32, IdentitySchema, ProviderIdentity>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderSet {
    leases: ProviderLeasePatch,
}

/// One snapshot-bound set of exact rendezvous keys and their hidden identity.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderObservation {
    set: ProviderSet,
}

impl ProviderObservation {
    #[cfg(test)]
    pub(crate) fn from_collections(
        collections: impl IntoIterator<Item = CollectionHandle>,
        serves: bool,
    ) -> Self {
        Self::from_locators(collections, serves, &BearerLocatorIndex::new())
    }

    pub(crate) fn into_set(self) -> ProviderSet {
        self.set
    }

    /// Reuse the serving snapshot's L→H PATCH and COW-insert the tiny set of
    /// collection-participant keys. Publication tokens are computed only when
    /// a key is scheduled, so this adds no second per-H trie.
    pub(crate) fn from_locators(
        collections: impl IntoIterator<Item = CollectionHandle>,
        serves: bool,
        locators: &BearerLocatorIndex,
    ) -> Self {
        if !serves {
            return Self::default();
        }
        let mut set = ProviderSet {
            leases: locators.clone(),
        };
        for collection in collections {
            let key = collection_provider_key(collection);
            set.leases
                .replace(&PatchEntry::with_value(&key, collection.raw));
        }
        Self { set }
    }
}

impl ProviderSet {
    #[cfg(test)]
    pub(crate) fn contains(&self, key: &ProviderKey) -> bool {
        self.leases.get(key).is_some()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> u64 {
        self.leases.len()
    }

    #[cfg(test)]
    pub(crate) fn identity(&self, key: &ProviderKey) -> Option<ProviderIdentity> {
        self.leases.get(key).copied()
    }
}

struct PublicationCycle {
    keys: PATCHIntoOrderedIterator<32, IdentitySchema, ProviderIdentity>,
    started: Mono,
    total: u64,
    remaining: u64,
    next_due: Mono,
}

/// Bounded-work publication scheduler for an arbitrarily large resident set.
///
/// Snapshot changes use PATCH difference to queue newly resident keys. One
/// persistent ordered cursor renews the complete set over one third of a lease; no
/// timer tick scans the whole inventory or allocates a flat resident queue.
pub(crate) struct ProviderPublisher {
    resident: ProviderSet,
    initialized: bool,
    additions: ProviderSet,
    retries: ProviderSet,
    retry_at: Mono,
    cycle: Option<PublicationCycle>,
    next_renewal: Mono,
    prefer_cycle: bool,
    prefer_retry: bool,
}

const MAX_PROVIDER_PUBLICATION_RETRIES: u64 = 1 << 10;

impl ProviderPublisher {
    pub(crate) fn new(now: Mono) -> Self {
        Self {
            resident: ProviderSet::default(),
            initialized: false,
            additions: ProviderSet::default(),
            retries: ProviderSet::default(),
            retry_at: now,
            cycle: None,
            next_renewal: now + PROVIDER_RENEWAL_PERIOD,
            prefer_cycle: false,
            prefer_retry: false,
        }
    }

    pub(crate) fn install(&mut self, resident: ProviderSet, now: Mono) {
        if self.initialized {
            let added = resident.leases.difference(&self.resident.leases);
            self.additions.leases = self.additions.leases.intersect(&resident.leases);
            self.additions.leases.union(added);
            self.retries.leases = self.retries.leases.intersect(&resident.leases);
        } else {
            self.initialized = true;
            self.additions = resident.clone();
            self.next_renewal = now + PROVIDER_RENEWAL_PERIOD;
        }
        self.resident = resident;
    }

    /// Queue one failed publication without allowing an outage to duplicate
    /// the whole resident set in retry state. False makes degraded coverage
    /// explicit to the host log rather than silently growing memory.
    pub(crate) fn retry(&mut self, key: ProviderKey, now: Mono) -> bool {
        if self.retries.leases.len() >= MAX_PROVIDER_PUBLICATION_RETRIES
            && self.retries.leases.get(&key).is_none()
        {
            return false;
        }
        let Some(identity) = self.resident.leases.get(&key).copied() else {
            return true;
        };
        let was_empty = self.retries.leases.is_empty();
        self.retries
            .leases
            .replace(&PatchEntry::with_value(&key, identity));
        if was_empty {
            self.retry_at = now + crate::RETRY_BACKOFF_BASE;
        }
        true
    }

    fn start_cycle_if_due(&mut self, now: Mono) {
        if self.cycle.is_some() || !self.initialized || now < self.next_renewal {
            return;
        }
        let total = self.resident.leases.len();
        if total == 0 {
            self.next_renewal = now + PROVIDER_RENEWAL_PERIOD;
            return;
        }
        self.cycle = Some(PublicationCycle {
            keys: self.resident.leases.clone().into_iter_ordered(),
            started: now,
            total,
            remaining: total,
            next_due: now,
        });
    }

    fn pop_pending(
        pending: &mut ProviderSet,
        resident: &ProviderSet,
    ) -> Option<(ProviderKey, ProviderIdentity)> {
        loop {
            let key = *pending.leases.iter_ordered().next()?;
            pending.leases.remove(&key);
            if let Some(identity) = resident.leases.get(&key).copied() {
                return Some((key, identity));
            }
        }
    }

    fn pop_cycle(&mut self, now: Mono) -> Option<(ProviderKey, ProviderIdentity)> {
        loop {
            let (key, completed, started, total) = {
                let cycle = self.cycle.as_mut()?;
                if now < cycle.next_due {
                    return None;
                }
                let key = cycle
                    .keys
                    .next()
                    .expect("a fixed PATCH traversal yields its recorded leaf count");
                cycle.remaining -= 1;
                let interval_nanos = (PROVIDER_RENEWAL_PERIOD.as_nanos() / u128::from(cycle.total))
                    .max(1)
                    .min(u128::from(u64::MAX));
                cycle.next_due = cycle.next_due + Duration::from_nanos(interval_nanos as u64);
                (key, cycle.remaining == 0, cycle.started, cycle.total)
            };
            if completed {
                self.cycle = None;
                self.next_renewal = started + PROVIDER_RENEWAL_PERIOD;
                let elapsed = now.duration_since(started);
                if elapsed > PROVIDER_RENEWAL_PERIOD {
                    tracing::warn!(
                        keys = total,
                        ?elapsed,
                        "provider renewal traversal exceeded its eight-hour budget; discovery coverage may decay"
                    );
                }
            }
            if let Some(identity) = self.resident.leases.get(&key).copied() {
                return Some((key, identity));
            }
            if completed {
                return None;
            }
        }
    }

    pub(crate) fn next(&mut self, now: Mono) -> Option<(ProviderKey, ProviderIdentity)> {
        self.start_cycle_if_due(now);
        if self.prefer_cycle
            && let Some(next) = self.pop_cycle(now)
        {
            self.prefer_cycle = false;
            return Some(next);
        }

        let retry_due = now >= self.retry_at;
        let mut from_retry = false;
        let pending = if self.prefer_retry && retry_due {
            let retry = Self::pop_pending(&mut self.retries, &self.resident);
            from_retry = retry.is_some();
            retry.or_else(|| Self::pop_pending(&mut self.additions, &self.resident))
        } else {
            let addition = Self::pop_pending(&mut self.additions, &self.resident);
            if addition.is_some() {
                addition
            } else if retry_due {
                let retry = Self::pop_pending(&mut self.retries, &self.resident);
                from_retry = retry.is_some();
                retry
            } else {
                None
            }
        };
        if let Some(next) = pending {
            self.prefer_retry = !self.prefer_retry;
            self.prefer_cycle = true;
            if from_retry {
                self.retry_at = now + crate::RETRY_BACKOFF_BASE;
            }
            return Some(next);
        }
        let next = self.pop_cycle(now);
        if next.is_some() {
            self.prefer_cycle = false;
        }
        next
    }
}

/// Receiver-local exact soft directory. The primary map owns lease deadlines;
/// the secondary map locates the at-most-64 providers for one exact key.
pub(crate) struct ProviderDirectory {
    local_id: PeerId,
    memberships: BTreeMap<(ProviderKey, PeerId), (Mono, ProviderToken)>,
    providers_by_key: BTreeMap<ProviderKey, BTreeSet<PeerId>>,
    deadlines: BTreeSet<(Mono, ProviderKey, PeerId)>,
    responsibility: BTreeSet<([u8; 32], ProviderKey, PeerId)>,
    limits: DirectoryLimits,
}

#[derive(Clone, Copy)]
struct DirectoryLimits {
    lease: Duration,
    memberships: usize,
}

impl Default for ProviderDirectory {
    fn default() -> Self {
        Self::new([0; 32])
    }
}

impl ProviderDirectory {
    pub(crate) fn new(local_id: PeerId) -> Self {
        Self {
            local_id,
            memberships: BTreeMap::new(),
            providers_by_key: BTreeMap::new(),
            deadlines: BTreeSet::new(),
            responsibility: BTreeSet::new(),
            limits: DirectoryLimits {
                lease: PROVIDER_LEASE_LIFETIME,
                memberships: MAX_PROVIDER_MEMBERSHIPS,
            },
        }
    }

    /// Install or renew one exact membership. Capacity pressure never prevents
    /// an already-admitted live membership from renewing.
    pub(crate) fn put(
        &mut self,
        key: ProviderKey,
        provider: PeerId,
        token: ProviderToken,
        now: Mono,
    ) -> bool {
        self.prune_expired(now);
        let membership = (key, provider);
        if let Some((previous, _)) = self.memberships.get(&membership).copied() {
            self.deadlines.remove(&(previous, key, provider));
        } else {
            self.prune_expired_key(key, now);
            if self
                .providers_by_key
                .get(&key)
                .is_some_and(|providers| providers.len() >= MAX_PROVIDERS_PER_KEY)
            {
                return false;
            }
            if self.memberships.len() >= self.limits.memberships {
                let candidate = (self.distance(key), key, provider);
                let Some(farthest) = self.responsibility.last().copied() else {
                    return false;
                };
                if candidate >= farthest {
                    return false;
                }
                self.remove_membership(farthest.1, farthest.2);
            }
            self.providers_by_key
                .entry(key)
                .or_default()
                .insert(provider);
            self.responsibility
                .insert((self.distance(key), key, provider));
        }

        let expires_at = now + self.limits.lease;
        self.memberships.insert(membership, (expires_at, token));
        self.deadlines.insert((expires_at, key, provider));
        true
    }

    /// Reclaim the at-most-64 stale memberships that can block this exact key.
    /// Global expiry cleanup remains bounded too, but unrelated older entries
    /// must never make a live insertion spuriously observe a saturated key.
    fn prune_expired_key(&mut self, key: ProviderKey, now: Mono) {
        let expired = self
            .providers_by_key
            .get(&key)
            .into_iter()
            .flatten()
            .copied()
            .filter(|provider| {
                self.memberships
                    .get(&(key, *provider))
                    .is_some_and(|(expires_at, _)| *expires_at <= now)
            })
            .collect::<Vec<_>>();
        for provider in expired {
            self.remove_membership(key, provider);
        }
    }

    /// Return every live provider retained for one exact rendezvous key.
    pub(crate) fn get(&mut self, key: ProviderKey, now: Mono) -> Vec<(PeerId, ProviderToken)> {
        self.prune_expired(now);
        let Some(providers) = self.providers_by_key.get(&key) else {
            return Vec::new();
        };
        let mut result = Vec::with_capacity(MAX_PROVIDERS_PER_KEY.min(providers.len()));
        for provider in providers.iter().copied() {
            if self
                .memberships
                .get(&(key, provider))
                .is_some_and(|(expires_at, _)| *expires_at > now)
            {
                result.push((provider, self.memberships[&(key, provider)].1));
            }
        }
        result
    }

    fn prune_expired(&mut self, now: Mono) {
        for _ in 0..MAX_EXPIRED_PROVIDER_MEMBERSHIPS_PER_CALL {
            let Some((expires_at, key, provider)) = self.deadlines.first().copied() else {
                break;
            };
            if expires_at > now {
                break;
            }
            self.deadlines.remove(&(expires_at, key, provider));
            let membership = (key, provider);
            if self
                .memberships
                .get(&membership)
                .is_none_or(|(deadline, _)| *deadline != expires_at)
            {
                continue;
            }
            self.remove_membership(key, provider);
        }
    }

    fn distance(&self, key: ProviderKey) -> [u8; 32] {
        std::array::from_fn(|index| key[index] ^ self.local_id[index])
    }

    fn remove_membership(&mut self, key: ProviderKey, provider: PeerId) {
        let Some((deadline, _)) = self.memberships.remove(&(key, provider)) else {
            return;
        };
        self.deadlines.remove(&(deadline, key, provider));
        self.responsibility
            .remove(&(self.distance(key), key, provider));
        let remove_key = {
            let providers = self
                .providers_by_key
                .get_mut(&key)
                .expect("stored membership contributes to its exact-key index");
            providers.remove(&provider);
            providers.is_empty()
        };
        if remove_key {
            self.providers_by_key.remove(&key);
        }
    }

    #[cfg(test)]
    fn with_limits(lease: Duration, memberships: usize) -> Self {
        Self {
            limits: DirectoryLimits { lease, memberships },
            ..Self::new([0; 32])
        }
    }
}

#[cfg(test)]
mod tests {
    use anybytes::Bytes;
    use ed25519_dalek::SigningKey;
    use hifitime::Epoch;
    use triblespace_core::blob::encodings::UnknownBlob;
    use triblespace_core::capability::{
        CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
        CapabilityResource, CapabilityValidity,
    };
    use triblespace_core::collection::{
        ACTION_WRITE, AdmissionPolicy, CollectionCommit, CollectionData, CollectionHandle,
        CollectionPolicy, CollectionRecord, CollectionStore, CollectionStoreExt,
    };
    use triblespace_core::inline::Inline;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStorePut, CapabilityProofStore, SnapshotSource};
    use triblespace_core::trible::TribleSet;

    use super::*;

    type BlobHandle = Inline<Handle<UnknownBlob>>;

    fn signing_key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn put_blob(store: &mut MemoryRepo, byte: u8) -> BlobHandle {
        store
            .put::<UnknownBlob, _>(Bytes::from_source(vec![byte; 257]))
            .unwrap()
    }

    fn commit(
        store: &mut MemoryRepo,
        collection: CollectionHandle,
        signer: &SigningKey,
        member: BlobHandle,
    ) -> CollectionCommit {
        let metadata = store
            .put::<triblespace_core::blob::encodings::simplearchive::SimpleArchive, _>(
                TribleSet::new(),
            )
            .unwrap();
        let commit = CollectionCommit::sign(
            signer,
            collection,
            CollectionData::new(member.raw),
            metadata,
        );
        store.insert(CollectionRecord::Commit(commit)).unwrap();
        commit
    }

    fn store_bundle(store: &mut MemoryRepo, bundle: CapabilityProofBundle) {
        use triblespace_core::blob::encodings::simplearchive::SimpleArchive;

        let (proof, claims) = bundle.into_parts();
        for claim in claims {
            store.put::<SimpleArchive, _>(claim).unwrap();
        }
        store.insert_proof(proof).unwrap();
    }

    #[test]
    fn publication_set_contains_only_explicit_collection_participation() {
        let writer = signing_key(11);
        let write_root = signing_key(12);
        let mut store = MemoryRepo::default();
        let open = store
            .collection(
                "open",
                CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open),
            )
            .unwrap();
        let restricted = store
            .collection(
                "restricted",
                CollectionPolicy::new(
                    AdmissionPolicy::direct(signing_key(13).verifying_key()),
                    AdmissionPolicy::Open,
                ),
            )
            .unwrap();
        let unauthorized = store
            .collection(
                "unauthorized",
                CollectionPolicy::new(
                    AdmissionPolicy::Open,
                    AdmissionPolicy::direct(write_root.verifying_key()),
                ),
            )
            .unwrap();
        let public_member = put_blob(&mut store, 21);
        let restricted_member = put_blob(&mut store, 22);
        let unauthorized_member = put_blob(&mut store, 23);
        let _uncommitted_resident = put_blob(&mut store, 24);
        let public_commit = commit(&mut store, open.handle(), &writer, public_member);
        let restricted_commit = commit(&mut store, restricted.handle(), &writer, restricted_member);
        commit(
            &mut store,
            unauthorized.handle(),
            &writer,
            unauthorized_member,
        );

        let set = ProviderObservation::from_collections(
            [open.handle(), restricted.handle(), unauthorized.handle()],
            true,
        )
        .into_set();

        for artifact in [open.handle(), restricted.handle(), unauthorized.handle()] {
            assert!(set.contains(&collection_provider_key(artifact)));
        }
        assert_eq!(set.len(), 3);
        assert_eq!(
            restricted_commit.metadata(),
            public_commit.metadata(),
            "content-addressed empty metadata is shared across both closures"
        );
        assert_eq!(
            ProviderObservation::from_collections([open.handle()], false),
            ProviderObservation::default(),
            "non-serving QoS cannot publish admitted artifacts"
        );
    }

    #[test]
    fn collection_participation_lease_is_independent_of_write_expiry() {
        let root = signing_key(31);
        let writer = signing_key(32);
        let mut store = MemoryRepo::default();
        let collection = store
            .collection(
                "expiring",
                CollectionPolicy::new(
                    AdmissionPolicy::Open,
                    AdmissionPolicy::direct(root.verifying_key()),
                ),
            )
            .unwrap();
        let write = CapabilityAtom::new(
            CapabilityAction::new(ACTION_WRITE),
            CapabilityResource::from(collection.handle()),
        );
        let validity =
            CapabilityValidity::new(Epoch::from_tai_seconds(0.0), Epoch::from_tai_seconds(10.0))
                .unwrap();
        store_bundle(
            &mut store,
            CapabilityProofBundle::issue_root(
                &root,
                CapabilityClaim::root(write, CapabilityMode::Invoke, Some(validity)),
                writer.verifying_key(),
            )
            .unwrap(),
        );
        let member = put_blob(&mut store, 33);
        commit(&mut store, collection.handle(), &writer, member);

        let during_set =
            ProviderObservation::from_collections([collection.handle()], true).into_set();
        let after_set =
            ProviderObservation::from_collections([collection.handle()], true).into_set();
        assert!(during_set.contains(&collection_provider_key(collection.handle())));
        assert!(after_set.contains(&collection_provider_key(collection.handle())));
    }

    #[test]
    fn observation_reuses_resident_locators_and_adds_collection_keys() {
        let mut store = MemoryRepo::default();
        let resident = put_blob(&mut store, 44);
        let snapshot = store.snapshot().unwrap();
        let provider = [45; 32];
        let collection = store
            .collection(
                "locator-projection",
                CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open),
            )
            .unwrap()
            .handle();
        let locators = crate::bearer::locator_index(&snapshot).unwrap();
        let blob_key = blob_locator(resident.raw);
        let collection_key = collection_provider_key(collection);

        let disabled = ProviderObservation::from_locators([collection], false, &locators);
        assert_eq!(disabled, ProviderObservation::default());

        let enabled = ProviderObservation::from_locators([collection], true, &locators).into_set();
        assert_eq!(enabled.identity(&blob_key), Some(resident.raw));
        assert_eq!(enabled.identity(&collection_key), Some(collection.raw));
        assert_eq!(locators.get(&blob_key), Some(&resident.raw));
        assert!(
            locators.get(&collection_key).is_none(),
            "COW insertion must not mutate the snapshot's bearer index"
        );
        assert_eq!(
            provider_lease_token(resident.raw, blob_key, provider),
            blob_provider_token(resident.raw, provider)
        );
        assert_eq!(
            provider_lease_token(collection.raw, collection_key, provider),
            collection_provider_token(collection.raw, provider)
        );
        assert_ne!(
            provider_lease_token(resident.raw, blob_key, provider),
            provider_lease_token(resident.raw, blob_key, [46; 32])
        );
        assert_ne!(
            provider_lease_token(resident.raw, blob_key, provider),
            provider_lease_token([47; 32], blob_key, provider)
        );
        assert_ne!(
            provider_lease_token(resident.raw, blob_key, provider),
            provider_lease_token(resident.raw, [48; 32], provider)
        );
    }

    #[test]
    fn directory_is_addressed_by_exact_provider_key_not_raw_artifact() {
        let artifact = [6; 32];
        let key = collection_provider_key(CollectionHandle::new(artifact));
        let provider = [7; 32];
        let now = crate::clock::mono_now();
        let mut directory = ProviderDirectory::default();
        assert!(directory.put(key, provider, [8; 32], now));

        assert!(directory.get(artifact, now).is_empty());
        assert_eq!(directory.get(key, now), vec![(provider, [8; 32])]);
    }

    #[test]
    fn keys_sharing_the_first_byte_remain_separate_dht_records() {
        let mut left = [0; 32];
        left[0] = 9;
        left[31] = 1;
        let mut right = left;
        right[31] = 2;
        let now = crate::clock::mono_now();
        let mut directory = ProviderDirectory::default();
        assert!(directory.put(left, [1; 32], [11; 32], now));
        assert!(directory.put(right, [2; 32], [12; 32], now));

        assert_eq!(directory.get(left, now), vec![([1; 32], [11; 32])]);
        assert_eq!(directory.get(right, now), vec![([2; 32], [12; 32])]);
    }

    #[test]
    fn renewal_at_capacity_preserves_membership_and_extends_lease() {
        let now = crate::clock::mono_now();
        let key = [1; 32];
        let provider = [2; 32];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(10), 1);
        assert!(directory.put(key, provider, [8; 32], now));
        assert!(!directory.put([3; 32], [4; 32], [9; 32], now));
        assert!(directory.put(key, provider, [8; 32], now + Duration::from_secs(5)));
        assert_eq!(
            directory.get(key, now + Duration::from_secs(14)),
            vec![(provider, [8; 32])]
        );
        assert!(directory.get(key, now + Duration::from_secs(15)).is_empty());
    }

    #[test]
    fn exact_key_rejects_the_sixty_fifth_provider_but_renews_existing() {
        let now = crate::clock::mono_now();
        let key = [3; 32];
        let mut directory = ProviderDirectory::default();
        for byte in 1..=MAX_PROVIDERS_PER_KEY as u8 {
            assert!(directory.put(key, [byte; 32], [byte + 1; 32], now));
        }
        assert!(!directory.put(key, [65; 32], [66; 32], now));
        assert!(directory.put(key, [1; 32], [2; 32], now));
        assert_eq!(directory.get(key, now).len(), MAX_PROVIDERS_PER_KEY);
    }

    #[test]
    fn expired_exact_key_capacity_is_reclaimed_despite_global_backlog() {
        let now = crate::clock::mono_now();
        let expired_at = now + Duration::from_secs(1);
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(1), 1024);
        for index in 0_u16..128 {
            let mut key = [0; 32];
            key[..2].copy_from_slice(&index.to_be_bytes());
            assert!(directory.put(key, [1; 32], [2; 32], now));
        }
        let key = [0xFF; 32];
        for byte in 1..=MAX_PROVIDERS_PER_KEY as u8 {
            assert!(directory.put(key, [byte; 32], [byte; 32], now));
        }

        assert!(directory.put(key, [65; 32], [65; 32], expired_at));
        assert_eq!(directory.get(key, expired_at), vec![([65; 32], [65; 32])]);
    }

    #[test]
    fn one_provider_can_publish_more_than_1024_exact_keys() {
        let now = crate::clock::mono_now();
        let provider = [7; 32];
        let mut directory = ProviderDirectory::default();
        for index in 0_u32..2048 {
            let mut key = [0; 32];
            key[..4].copy_from_slice(&index.to_be_bytes());
            assert!(directory.put(key, provider, [8; 32], now));
        }
        assert_eq!(directory.memberships.len(), 2048);
    }

    #[test]
    fn due_renewal_is_not_starved_by_sustained_additions() {
        let now = crate::clock::mono_now();
        let old_key = [0; 32];
        let old_token = [1; 32];
        let mut resident = ProviderSet::default();
        resident
            .leases
            .replace(&PatchEntry::with_value(&old_key, old_token));
        let mut publisher = ProviderPublisher::new(now);
        publisher.install(resident.clone(), now);
        assert_eq!(publisher.next(now), Some((old_key, old_token)));

        // Make the next due call service a new arrival first. The following
        // call must nevertheless advance the already-due renewal traversal.
        publisher.prefer_cycle = false;
        let due = now + PROVIDER_RENEWAL_PERIOD;
        let first_new = [2; 32];
        resident
            .leases
            .replace(&PatchEntry::with_value(&first_new, [3; 32]));
        publisher.install(resident.clone(), due);
        assert_eq!(publisher.next(due), Some((first_new, [3; 32])));

        let second_new = [4; 32];
        resident
            .leases
            .replace(&PatchEntry::with_value(&second_new, [5; 32]));
        publisher.install(resident, due);
        assert_eq!(publisher.next(due), Some((old_key, old_token)));
    }

    #[test]
    fn failed_publication_waits_for_retry_backoff() {
        let now = crate::clock::mono_now();
        let key = [9; 32];
        let token = [10; 32];
        let mut resident = ProviderSet::default();
        resident
            .leases
            .replace(&PatchEntry::with_value(&key, token));
        let mut publisher = ProviderPublisher::new(now);
        publisher.install(resident, now);
        assert_eq!(publisher.next(now), Some((key, token)));
        assert!(publisher.retry(key, now));

        assert_eq!(publisher.next(now), None);
        assert_eq!(
            publisher.next(now + crate::RETRY_BACKOFF_BASE),
            Some((key, token))
        );
    }

    #[test]
    fn renewal_cycle_uses_an_eight_hour_anchor() {
        let now = crate::clock::mono_now();
        let mut resident = ProviderSet::default();
        for byte in 1..=4 {
            resident
                .leases
                .replace(&PatchEntry::with_value(&[byte; 32], [byte + 10; 32]));
        }
        let mut publisher = ProviderPublisher::new(now);
        publisher.install(resident, now);
        while publisher.next(now).is_some() {}

        let started = now + PROVIDER_RENEWAL_PERIOD;
        let interval =
            Duration::from_nanos((PROVIDER_RENEWAL_PERIOD.as_nanos() / 4).try_into().unwrap());
        for offset in 0..4 {
            assert!(publisher.next(started + interval * offset).is_some());
        }
        assert!(publisher.cycle.is_none());
        assert_eq!(publisher.next_renewal, started + PROVIDER_RENEWAL_PERIOD);
        assert!(
            publisher.next(started + PROVIDER_RENEWAL_PERIOD).is_some(),
            "the next traversal begins at the prior cycle's anchored deadline"
        );
    }

    #[test]
    fn first_sweep_renews_its_last_key_with_a_full_cycle_of_margin() {
        let now = crate::clock::mono_now();
        let mut resident = ProviderSet::default();
        for index in 0_u16..257 {
            let mut key = [0; 32];
            key[..2].copy_from_slice(&index.to_be_bytes());
            resident.leases.replace(&PatchEntry::with_value(&key, key));
        }
        let mut publisher = ProviderPublisher::new(now);
        publisher.install(resident, now);
        while publisher.next(now).is_some() {}

        let mut emitted_at = now + PROVIDER_RENEWAL_PERIOD;
        assert!(publisher.next(emitted_at).is_some());
        while let Some(next_due) = publisher.cycle.as_ref().map(|cycle| cycle.next_due) {
            emitted_at = next_due;
            assert!(publisher.next(emitted_at).is_some());
        }

        assert!(
            emitted_at <= now + (PROVIDER_LEASE_LIFETIME - PROVIDER_RENEWAL_PERIOD),
            "the final initial lease needs one complete renewal-cycle budget before expiry"
        );
    }

    #[test]
    fn expiry_reclamation_is_bounded_and_stale_entries_are_never_returned() {
        let now = crate::clock::mono_now();
        let key = [3; 32];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(1), 100);
        for byte in 1..=64 {
            assert!(directory.put(key, [byte; 32], [byte + 1; 32], now));
        }
        let other = [4; 32];
        assert!(directory.put(other, [65; 32], [66; 32], now));

        assert!(directory.get(key, now + Duration::from_secs(1)).is_empty());
        assert_eq!(directory.memberships.len(), 1);
        assert!(
            directory
                .get(other, now + Duration::from_secs(1))
                .is_empty()
        );
        assert!(directory.memberships.is_empty());
    }
}
