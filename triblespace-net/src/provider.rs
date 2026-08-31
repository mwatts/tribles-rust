//! Exact soft-state provider leases for collection participants.
//!
//! Each serving collection is mapped to one opaque full-width rendezvous key.
//! Providers renew that key at its K closest DHT nodes; directory nodes never
//! receive the collection handle.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use triblespace_core::collection::CollectionHandle;

use crate::clock::Mono;
use crate::transport::PeerId;

/// Opaque rendezvous key for one collection.
pub(crate) type ProviderKey = [u8; 32];
pub(crate) type ProviderToken = [u8; 32];

/// Receiver-chosen lifetime of one exact provider lease.
pub(crate) const PROVIDER_LEASE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
/// Maximum fan-out returned for one collection-participant lookup.
pub(crate) const MAX_PROVIDERS_PER_KEY: usize = 64;
const _: () = assert!(MAX_PROVIDERS_PER_KEY <= u8::MAX as usize);

/// Aggregate receiver-local soft-state bound. Existing memberships can always
/// renew at the bound; new memberships wait for bounded expiry reclamation.
const MAX_PROVIDER_MEMBERSHIPS: usize = 1 << 17;
const MAX_PROVIDER_MEMBERSHIPS_PER_PEER: usize = 1 << 10;
/// Bound work performed by one exact directory lookup. A dense or stale key
/// may return fewer hints, while its cursor makes repeated calls cover all
/// candidates eventually.
const MAX_PROVIDER_MEMBERSHIPS_SCANNED_PER_GET: usize = 256;
/// Bound opportunistic expiry reclamation performed by one RPC.
const MAX_EXPIRED_PROVIDER_MEMBERSHIPS_PER_CALL: usize = 64;

/// Derive the opaque provider rendezvous key for one collection session.
pub(crate) fn collection_provider_key(collection: CollectionHandle) -> ProviderKey {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/collection-provider-key/v1");
    hasher.update(&collection.raw);
    *hasher.finalize().as_bytes()
}

pub(crate) fn collection_provider_token(identity: [u8; 32], provider: PeerId) -> ProviderToken {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/collection-provider-token/v1");
    hasher.update(&identity);
    hasher.update(&provider);
    *hasher.finalize().as_bytes()
}

/// Canonical exact publication set for one serving snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderSet {
    leases: BTreeMap<ProviderKey, ProviderToken>,
}

/// One snapshot-bound set of serving collection-participant leases.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderObservation {
    set: ProviderSet,
}

impl ProviderObservation {
    pub(crate) fn from_collections(
        collections: impl IntoIterator<Item = CollectionHandle>,
        serves: bool,
        provider: PeerId,
    ) -> Self {
        if !serves {
            return Self::default();
        }
        Self::default().with_collections(collections, provider)
    }

    pub(crate) fn into_set(self) -> ProviderSet {
        self.set
    }

    pub(crate) fn with_collections(
        mut self,
        collections: impl IntoIterator<Item = CollectionHandle>,
        provider: PeerId,
    ) -> Self {
        for collection in collections {
            self.set.leases.insert(
                collection_provider_key(collection),
                collection_provider_token(collection.raw, provider),
            );
        }
        self
    }
}

impl ProviderSet {
    pub(crate) fn contains(&self, key: &ProviderKey) -> bool {
        self.leases.contains_key(key)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (ProviderKey, ProviderToken)> + '_ {
        self.leases.iter().map(|(key, token)| (*key, *token))
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.leases.len()
    }
}

/// Receiver-local exact soft directory. The primary map owns lease deadlines;
/// the secondary map supports bounded lookup without scanning unrelated keys.
pub(crate) struct ProviderDirectory {
    memberships: BTreeMap<(ProviderKey, PeerId), (Mono, ProviderToken)>,
    providers_by_key: BTreeMap<ProviderKey, BTreeSet<PeerId>>,
    lookup_cursor_by_key: BTreeMap<ProviderKey, PeerId>,
    deadlines: BTreeSet<(Mono, ProviderKey, PeerId)>,
    memberships_by_provider: BTreeMap<PeerId, usize>,
    limits: DirectoryLimits,
}

#[derive(Clone, Copy)]
struct DirectoryLimits {
    lease: Duration,
    memberships: usize,
}

impl Default for ProviderDirectory {
    fn default() -> Self {
        Self {
            memberships: BTreeMap::new(),
            providers_by_key: BTreeMap::new(),
            lookup_cursor_by_key: BTreeMap::new(),
            deadlines: BTreeSet::new(),
            memberships_by_provider: BTreeMap::new(),
            limits: DirectoryLimits {
                lease: PROVIDER_LEASE_LIFETIME,
                memberships: MAX_PROVIDER_MEMBERSHIPS,
            },
        }
    }
}

impl ProviderDirectory {
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
            if self.memberships.len() >= self.limits.memberships {
                return false;
            }
            if self
                .memberships_by_provider
                .get(&provider)
                .copied()
                .unwrap_or_default()
                >= MAX_PROVIDER_MEMBERSHIPS_PER_PEER
            {
                return false;
            }
            self.providers_by_key
                .entry(key)
                .or_default()
                .insert(provider);
            *self.memberships_by_provider.entry(provider).or_default() += 1;
        }

        let expires_at = now + self.limits.lease;
        self.memberships.insert(membership, (expires_at, token));
        self.deadlines.insert((expires_at, key, provider));
        true
    }

    /// Return bounded live providers for one exact rendezvous key.
    pub(crate) fn get(&mut self, key: ProviderKey, now: Mono) -> Vec<(PeerId, ProviderToken)> {
        self.prune_expired(now);
        let Some(providers) = self.providers_by_key.get(&key) else {
            return Vec::new();
        };
        let cursor = self
            .lookup_cursor_by_key
            .get(&key)
            .copied()
            .or_else(|| providers.last().copied())
            .expect("a retained provider key has at least one provider");

        use std::ops::Bound::{Excluded, Unbounded};
        let mut result = Vec::with_capacity(MAX_PROVIDERS_PER_KEY.min(providers.len()));
        let mut first_scanned = None;
        let mut last_scanned = None;
        let mut scanned = 0;
        for provider in providers
            .range((Excluded(cursor), Unbounded))
            .chain(providers.range(..=cursor))
            .take(MAX_PROVIDER_MEMBERSHIPS_SCANNED_PER_GET)
            .copied()
        {
            first_scanned.get_or_insert(provider);
            last_scanned = Some(provider);
            scanned += 1;
            if self
                .memberships
                .get(&(key, provider))
                .is_some_and(|(expires_at, _)| *expires_at > now)
            {
                result.push((provider, self.memberships[&(key, provider)].1));
                if result.len() == MAX_PROVIDERS_PER_KEY {
                    break;
                }
            }
        }

        let next_cursor = if !result.is_empty() && scanned == providers.len() {
            first_scanned
        } else {
            last_scanned
        };
        if let Some(next) = next_cursor {
            self.lookup_cursor_by_key.insert(key, next);
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
            self.memberships.remove(&membership);
            let count = self
                .memberships_by_provider
                .get_mut(&provider)
                .expect("stored membership contributes to its provider quota");
            *count -= 1;
            if *count == 0 {
                self.memberships_by_provider.remove(&provider);
            }
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
                self.lookup_cursor_by_key.remove(&key);
            }
        }
    }

    #[cfg(test)]
    fn with_limits(lease: Duration, memberships: usize) -> Self {
        Self {
            limits: DirectoryLimits { lease, memberships },
            ..Self::default()
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
    use triblespace_core::repo::{BlobStorePut, CapabilityProofStore};
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

        let provider = [42; 32];
        let set = ProviderObservation::from_collections(
            [open.handle(), restricted.handle(), unauthorized.handle()],
            true,
            provider,
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
            ProviderObservation::from_collections([open.handle()], false, [42; 32]),
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

        let provider = [42; 32];
        let during_set =
            ProviderObservation::from_collections([collection.handle()], true, provider).into_set();
        let after_set =
            ProviderObservation::from_collections([collection.handle()], true, provider).into_set();
        assert!(during_set.contains(&collection_provider_key(collection.handle())));
        assert!(after_set.contains(&collection_provider_key(collection.handle())));
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
    fn exact_lookup_rotates_across_more_than_the_response_cap() {
        let now = crate::clock::mono_now();
        let key = [3; 32];
        let mut directory = ProviderDirectory::default();
        for byte in 1..=70 {
            assert!(directory.put(key, [byte; 32], [byte + 1; 32], now));
        }

        let first = directory.get(key, now);
        let second = directory.get(key, now);
        assert_eq!(first.len(), MAX_PROVIDERS_PER_KEY);
        assert_eq!(second.len(), MAX_PROVIDERS_PER_KEY);
        let seen: BTreeSet<_> = first
            .into_iter()
            .chain(second)
            .map(|(provider, _)| provider)
            .collect();
        assert_eq!(seen.len(), 70, "the response cap is not a storage cap");
    }

    #[test]
    fn expiry_reclamation_is_bounded_and_stale_entries_are_never_returned() {
        let now = crate::clock::mono_now();
        let key = [3; 32];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(1), 100);
        for byte in 1..=65 {
            assert!(directory.put(key, [byte; 32], [byte + 1; 32], now));
        }

        assert!(directory.get(key, now + Duration::from_secs(1)).is_empty());
        assert_eq!(directory.memberships.len(), 1);
        assert!(directory.get(key, now + Duration::from_secs(1)).is_empty());
        assert!(directory.memberships.is_empty());
    }
}
