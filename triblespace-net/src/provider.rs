//! Exact soft-state provider leases for already-known immutable artifacts.
//!
//! Each bearer artifact handle is blinded once into a full-width global
//! rendezvous key. Providers renew that exact key at its K closest DHT nodes;
//! directory nodes never receive the bearer handle and no fixed prefix buckets
//! collapse unrelated artifacts onto shared hotspots.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use hifitime::Epoch;
use triblespace_core::collection::DisclosureSnapshot;

use crate::clock::Mono;
use crate::transport::PeerId;

/// Bare physical identity of one data artifact.
pub type ArtifactId = [u8; 32];

/// Opaque global rendezvous key for one exact artifact.
pub(crate) type ProviderKey = [u8; 32];

/// Receiver-chosen lifetime of one exact provider lease.
pub(crate) const PROVIDER_LEASE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
/// Maximum fan-out returned for one exact artifact lookup.
pub(crate) const MAX_PROVIDERS_PER_KEY: usize = 64;
const _: () = assert!(MAX_PROVIDERS_PER_KEY <= u8::MAX as usize);

/// Aggregate receiver-local soft-state bound. Existing memberships can always
/// renew at the bound; new memberships wait for bounded expiry reclamation.
const MAX_PROVIDER_MEMBERSHIPS: usize = 1 << 24;
/// Bound work performed by one exact directory lookup. A dense or stale key
/// may return fewer hints, while its cursor makes repeated calls cover all
/// candidates eventually.
const MAX_PROVIDER_MEMBERSHIPS_SCANNED_PER_GET: usize = 256;
/// Bound opportunistic expiry reclamation performed by one RPC.
const MAX_EXPIRED_PROVIDER_MEMBERSHIPS_PER_CALL: usize = 64;

/// Derive an exact global lookup key without disclosing the bearer handle.
pub(crate) fn provider_key(artifact: ArtifactId) -> ProviderKey {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/provider-key/v2");
    hasher.update(&artifact);
    *hasher.finalize().as_bytes()
}

/// Canonical exact publication set for one serving snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderSet {
    keys: BTreeSet<ProviderKey>,
}

/// One snapshot-bound authorization to publish an exact provider set.
///
/// `valid_through` is an inclusive, conservative bound on the disclosure
/// observation, not a property of remote provider leases. Once it passes, the
/// host must stop autonomous renewal until the store side sends a fresh
/// observation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProviderObservation {
    set: ProviderSet,
    valid_through: Option<Epoch>,
}

impl ProviderObservation {
    pub(crate) fn from_disclosure(disclosure: &DisclosureSnapshot, serves: bool) -> Self {
        if !serves {
            return Self::default();
        }
        Self {
            set: ProviderSet::from_artifacts(disclosure.public_handles().map(|handle| handle.raw)),
            valid_through: disclosure.public_valid_through(),
        }
    }

    pub(crate) fn into_parts(self) -> (ProviderSet, Option<Epoch>) {
        (self.set, self.valid_through)
    }

    #[cfg(test)]
    pub(crate) fn with_valid_through(set: ProviderSet, valid_through: Option<Epoch>) -> Self {
        Self { set, valid_through }
    }
}

impl ProviderSet {
    pub(crate) fn from_artifacts(artifacts: impl IntoIterator<Item = ArtifactId>) -> Self {
        Self {
            keys: artifacts.into_iter().map(provider_key).collect(),
        }
    }

    pub(crate) fn contains(&self, key: &ProviderKey) -> bool {
        self.keys.contains(key)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = ProviderKey> + '_ {
        self.keys.iter().copied()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Receiver-local exact soft directory. The primary map owns lease deadlines;
/// the secondary map supports bounded lookup without scanning unrelated keys.
pub(crate) struct ProviderDirectory {
    memberships: BTreeMap<(ProviderKey, PeerId), Mono>,
    providers_by_key: BTreeMap<ProviderKey, BTreeSet<PeerId>>,
    lookup_cursor_by_key: BTreeMap<ProviderKey, PeerId>,
    deadlines: BTreeSet<(Mono, ProviderKey, PeerId)>,
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
    pub(crate) fn put(&mut self, key: ProviderKey, provider: PeerId, now: Mono) -> bool {
        self.prune_expired(now);
        let membership = (key, provider);
        if let Some(previous) = self.memberships.get(&membership).copied() {
            self.deadlines.remove(&(previous, key, provider));
        } else {
            if self.memberships.len() >= self.limits.memberships {
                return false;
            }
            self.providers_by_key
                .entry(key)
                .or_default()
                .insert(provider);
        }

        let expires_at = now + self.limits.lease;
        self.memberships.insert(membership, expires_at);
        self.deadlines.insert((expires_at, key, provider));
        true
    }

    /// Return bounded live providers for one exact rendezvous key.
    pub(crate) fn get(&mut self, key: ProviderKey, now: Mono) -> Vec<PeerId> {
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
                .is_some_and(|expires_at| *expires_at > now)
            {
                result.push(provider);
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
            if self.memberships.get(&membership) != Some(&expires_at) {
                continue;
            }
            self.memberships.remove(&membership);
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
        DisclosureSnapshot,
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
    ) {
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
    fn provider_keys_are_deterministic_global_and_opaque() {
        let artifact = [3; 32];
        let key = provider_key(artifact);
        assert_eq!(key, provider_key(artifact));
        assert_ne!(key, artifact);
        assert_ne!(key, provider_key([4; 32]));
    }

    #[test]
    fn publication_set_is_exact_canonical_and_deduplicated() {
        let a = [1; 32];
        let b = [2; 32];
        let set = ProviderSet::from_artifacts([b, a, b]);
        assert_eq!(set.len(), 2);
        assert_eq!(set.iter().collect::<Vec<_>>(), {
            let mut expected = vec![provider_key(a), provider_key(b)];
            expected.sort_unstable();
            expected
        });
    }

    #[test]
    fn publication_set_is_exactly_the_open_admitted_resident_projection() {
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
        let uncommitted_resident = put_blob(&mut store, 24);
        commit(&mut store, open.handle(), &writer, public_member);
        commit(&mut store, restricted.handle(), &writer, restricted_member);
        commit(
            &mut store,
            unauthorized.handle(),
            &writer,
            unauthorized_member,
        );

        let snapshot = store.snapshot().unwrap();
        let disclosure =
            DisclosureSnapshot::build_at(&snapshot, Epoch::from_tai_seconds(0.0)).unwrap();
        let (set, valid_through) =
            ProviderObservation::from_disclosure(&disclosure, true).into_parts();

        assert!(set.contains(&provider_key(public_member.raw)));
        assert_eq!(valid_through, None);
        for hidden in [restricted_member, unauthorized_member, uncommitted_resident] {
            assert!(!set.contains(&provider_key(hidden.raw)));
        }
        assert_eq!(
            ProviderObservation::from_disclosure(&disclosure, false),
            ProviderObservation::default(),
            "non-serving QoS cannot publish even an open disclosure"
        );
    }

    #[test]
    fn expired_write_evidence_is_removed_from_the_future_renewal_set() {
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

        let snapshot = store.snapshot().unwrap();
        let during = DisclosureSnapshot::build_at(&snapshot, Epoch::from_tai_seconds(5.0)).unwrap();
        let after = DisclosureSnapshot::build_at(&snapshot, Epoch::from_tai_seconds(11.0)).unwrap();
        let (during_set, during_valid_through) =
            ProviderObservation::from_disclosure(&during, true).into_parts();
        let (after_set, after_valid_through) =
            ProviderObservation::from_disclosure(&after, true).into_parts();
        assert!(during_set.contains(&provider_key(member.raw)));
        assert_eq!(during_valid_through, Some(Epoch::from_tai_seconds(10.0)));
        assert!(!after_set.contains(&provider_key(member.raw)));
        assert_eq!(after_valid_through, None);
    }

    #[test]
    fn directory_is_addressed_by_exact_provider_key_not_raw_artifact() {
        let artifact = [6; 32];
        let key = provider_key(artifact);
        let provider = [7; 32];
        let now = crate::clock::mono_now();
        let mut directory = ProviderDirectory::default();
        assert!(directory.put(key, provider, now));

        assert!(directory.get(artifact, now).is_empty());
        assert_eq!(directory.get(key, now), vec![provider]);
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
        assert!(directory.put(left, [1; 32], now));
        assert!(directory.put(right, [2; 32], now));

        assert_eq!(directory.get(left, now), vec![[1; 32]]);
        assert_eq!(directory.get(right, now), vec![[2; 32]]);
    }

    #[test]
    fn renewal_at_capacity_preserves_membership_and_extends_lease() {
        let now = crate::clock::mono_now();
        let key = [1; 32];
        let provider = [2; 32];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(10), 1);
        assert!(directory.put(key, provider, now));
        assert!(!directory.put([3; 32], [4; 32], now));
        assert!(directory.put(key, provider, now + Duration::from_secs(5)));
        assert_eq!(
            directory.get(key, now + Duration::from_secs(14)),
            vec![provider]
        );
        assert!(directory.get(key, now + Duration::from_secs(15)).is_empty());
    }

    #[test]
    fn exact_lookup_rotates_across_more_than_the_response_cap() {
        let now = crate::clock::mono_now();
        let key = [3; 32];
        let mut directory = ProviderDirectory::default();
        for byte in 1..=70 {
            assert!(directory.put(key, [byte; 32], now));
        }

        let first = directory.get(key, now);
        let second = directory.get(key, now);
        assert_eq!(first.len(), MAX_PROVIDERS_PER_KEY);
        assert_eq!(second.len(), MAX_PROVIDERS_PER_KEY);
        let seen: BTreeSet<_> = first.into_iter().chain(second).collect();
        assert_eq!(seen.len(), 70, "the response cap is not a storage cap");
    }

    #[test]
    fn expiry_reclamation_is_bounded_and_stale_entries_are_never_returned() {
        let now = crate::clock::mono_now();
        let key = [3; 32];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(1), 100);
        for byte in 1..=65 {
            assert!(directory.put(key, [byte; 32], now));
        }

        assert!(directory.get(key, now + Duration::from_secs(1)).is_empty());
        assert_eq!(directory.memberships.len(), 1);
        assert!(directory.get(key, now + Duration::from_secs(1)).is_empty());
        assert!(directory.memberships.is_empty());
    }
}
