//! Snapshot-bound disclosure closure for collection payloads.
//!
//! This module derives serving policy from one immutable store observation.
//! It does not persist a receipt or change blob lifetime. A handle is globally
//! bearer-publishable only when it lies in the resident conservative closure
//! of a strictly signed, currently WRITE-admitted COMMIT to a READ-open
//! collection. Restricted readers use the collection-scoped closure after
//! separately proving READ admission.
//!
//! Unsigned MERGE and DERIVE equations are deliberately absent from this
//! calculation: computation evidence cannot disclose an otherwise private
//! source. Closure follows the same [`BlobChildren`](crate::repo::BlobChildren)
//! law as conservative garbage collection, where every aligned resident
//! 32-byte blob handle is an edge.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use ed25519_dalek::VerifyingKey;
use hifitime::Epoch;

use crate::blob::encodings::UnknownBlob;
use crate::blob::Blob;
use crate::capability::CapabilityMode;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::patch::{Blake3Merkle, Entry, IdentitySchema, PATCH};
use crate::repo::{reachable, BlobChildren, CapabilityProofRead};

use super::api::{
    admission_evidence_from_bundles, load_collection_descriptor, load_resident_proof_bundles,
};
use super::{
    AdmissionPolicy, CollectionCommit, CollectionHandle, CollectionRead, CollectionRecord,
    ACTION_WRITE,
};

type DisclosureOrder = crate::trible::EAVOrder;
type CollectionDisclosureIndex = PATCH<64, DisclosureOrder, (), Blake3Merkle>;
type PublicDisclosureIndex = PATCH<32, IdentitySchema, (), Blake3Merkle>;

/// Failure to observe either global input set needed for disclosure.
///
/// Malformed descriptors, signatures, proofs, and missing blobs are untrusted
/// candidates rather than build failures. They fail closed within their own
/// collection without suppressing unrelated collections.
#[derive(Debug)]
pub enum DisclosureBuildError<RecordsError, ProofsError> {
    /// The immutable collection-record observation was unavailable.
    Records(RecordsError),
    /// The immutable capability-proof observation was unavailable.
    Proofs(ProofsError),
}

impl<RecordsError, ProofsError> fmt::Display for DisclosureBuildError<RecordsError, ProofsError>
where
    RecordsError: fmt::Display,
    ProofsError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Records(source) => {
                write!(
                    formatter,
                    "failed to enumerate collection records: {source}"
                )
            }
            Self::Proofs(source) => {
                write!(formatter, "failed to enumerate capability proofs: {source}")
            }
        }
    }
}

impl<RecordsError, ProofsError> Error for DisclosureBuildError<RecordsError, ProofsError>
where
    RecordsError: Error + 'static,
    ProofsError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Records(source) => Some(source),
            Self::Proofs(source) => Some(source),
        }
    }
}

/// One immutable, derived publication boundary for a store snapshot.
///
/// `by_collection` is keyed by `collection || blob`. `public` is its READ-open
/// projection. Both are ordinary PATCH values, so cloning a disclosure
/// snapshot is cheap and membership probes are independent of traversal
/// history.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DisclosureSnapshot {
    by_collection: CollectionDisclosureIndex,
    public: PublicDisclosureIndex,
    public_valid_through: Option<Epoch>,
}

impl DisclosureSnapshot {
    /// Derive the current serving/publication boundary at `instant`.
    ///
    /// Expiry is a current-policy decision: an expired WRITE proof can remove
    /// a handle from a later disclosure snapshot. This makes no claim that
    /// bytes learned from an earlier snapshot can become secret again.
    pub fn build_at<S>(
        snapshot: &S,
        instant: Epoch,
    ) -> Result<Self, DisclosureBuildError<S::RecordsError, S::ProofsError>>
    where
        S: BlobChildren + CapabilityProofRead + CollectionRead,
    {
        let mut commits = Vec::new();
        let records = snapshot.records().map_err(DisclosureBuildError::Records)?;
        for record in records {
            let record = record.map_err(DisclosureBuildError::Records)?;
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            if commit.verify_strict().is_ok() {
                commits.push(commit);
            }
        }

        commits.sort_unstable_by_key(|commit| {
            (
                commit.collection().raw,
                commit.public_key().raw,
                commit.id().raw(),
            )
        });

        let proofs = snapshot
            .proofs()
            .map_err(DisclosureBuildError::Proofs)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(DisclosureBuildError::Proofs)?;
        let bundles = load_resident_proof_bundles(snapshot, proofs);

        let mut by_collection = CollectionDisclosureIndex::new();
        let mut public = PublicDisclosureIndex::new();
        let mut public_valid_through = None;
        let mut start = 0;
        while start < commits.len() {
            let collection = commits[start].collection();
            let mut end = start + 1;
            while end < commits.len() && commits[end].collection() == collection {
                end += 1;
            }
            Self::add_collection(
                snapshot,
                instant,
                collection,
                &commits[start..end],
                Arc::clone(&bundles),
                &mut by_collection,
                &mut public,
                &mut public_valid_through,
            );
            start = end;
        }

        Ok(Self {
            by_collection,
            public,
            public_valid_through,
        })
    }

    fn add_collection<S>(
        snapshot: &S,
        instant: Epoch,
        collection: CollectionHandle,
        commits: &[CollectionCommit],
        bundles: Arc<[crate::capability::CapabilityProofBundle]>,
        by_collection: &mut CollectionDisclosureIndex,
        public: &mut PublicDisclosureIndex,
        public_valid_through: &mut Option<Epoch>,
    ) where
        S: BlobChildren,
    {
        let Ok(descriptor) = load_collection_descriptor(snapshot, collection) else {
            return;
        };
        let read_is_open = matches!(descriptor.policy.read(), AdmissionPolicy::Open);
        let write = admission_evidence_from_bundles(
            descriptor.policy.write(),
            ACTION_WRITE,
            CapabilityMode::Invoke,
            collection,
            bundles,
        );

        let mut roots = Vec::new();
        let mut last_subject = None;
        let mut last_admitted = false;
        for commit in commits {
            let subject_raw = commit.public_key().raw;
            if last_subject != Some(subject_raw) {
                last_subject = Some(subject_raw);
                last_admitted = VerifyingKey::from_bytes(&subject_raw)
                    .map(|subject| write.authorizes(subject, instant))
                    .unwrap_or(false);
            }
            if !last_admitted {
                continue;
            }
            for root in [
                Inline::<Handle<UnknownBlob>>::new(collection.raw),
                Inline::<Handle<UnknownBlob>>::new(commit.data().raw),
                commit.metadata().transmute(),
            ] {
                if snapshot.get::<Blob<UnknownBlob>, UnknownBlob>(root).is_ok() {
                    roots.push(root);
                }
            }
        }

        if read_is_open && !roots.is_empty() {
            if let Some(bound) = write.observation_valid_through(instant) {
                *public_valid_through = Some(
                    public_valid_through
                        .map(|current| current.min(bound))
                        .unwrap_or(bound),
                );
            }
        }

        for handle in reachable(snapshot, roots) {
            let mut key = [0; 64];
            key[..32].copy_from_slice(&collection.raw);
            key[32..].copy_from_slice(&handle.raw);
            by_collection.insert(&Entry::new(&key));
            if read_is_open {
                public.insert(&Entry::new(&handle.raw));
            }
        }
    }

    /// Whether `handle` lies in `collection`'s resident admitted closure.
    ///
    /// For a READ-restricted collection this is only the second half of a
    /// serving decision; the requester must separately satisfy READ admission.
    pub fn contains(
        &self,
        collection: CollectionHandle,
        handle: Inline<Handle<UnknownBlob>>,
    ) -> bool {
        let mut key = [0; 64];
        key[..32].copy_from_slice(&collection.raw);
        key[32..].copy_from_slice(&handle.raw);
        self.by_collection.get(&key).is_some()
    }

    /// Whether `handle` is globally bearer-publishable through any READ-open
    /// collection.
    pub fn public_contains(&self, handle: Inline<Handle<UnknownBlob>>) -> bool {
        self.public.get(&handle.raw).is_some()
    }

    /// Visit every disclosed handle in one collection in deterministic order.
    pub fn for_collection(
        &self,
        collection: CollectionHandle,
        mut visit: impl FnMut(Inline<Handle<UnknownBlob>>),
    ) {
        self.by_collection
            .infixes::<32, 32, _>(&collection.raw, |raw| {
                visit(Inline::new(*raw));
            });
    }

    /// Iterate every globally bearer-publishable handle in deterministic order.
    pub fn public_handles(&self) -> impl Iterator<Item = Inline<Handle<UnknownBlob>>> + '_ {
        self.public
            .iter_ordered()
            .map(|raw| Inline::<Handle<UnknownBlob>>::new(*raw))
    }

    /// Conservative inclusive validity bound of this public observation.
    ///
    /// The bound is the earliest expiry among currently usable WRITE proof
    /// paths which contributed a resident READ-open closure. Another path may
    /// keep the same handles publishable afterward, so reaching it means
    /// "reobserve", not "revoked". `None` is unbounded.
    pub fn public_valid_through(&self) -> Option<Epoch> {
        self.public_valid_through
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::capability::{
        CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityProofBundle,
        CapabilityResource, CapabilityValidity,
    };
    use crate::collection::{
        CollectionData, CollectionDerive, CollectionMerge, CollectionPolicy, CollectionStore,
        CollectionStoreExt,
    };
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::{BlobStorePut, CapabilityProofStore, SnapshotSource};
    use crate::trible::TribleSet;

    use super::*;

    type BlobHandle = Inline<Handle<UnknownBlob>>;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn open_policy() -> CollectionPolicy {
        CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open)
    }

    fn put_blob(store: &mut MemoryRepo, bytes: impl Into<crate::blob::Bytes>) -> BlobHandle {
        store
            .put::<UnknownBlob, _>(Blob::<UnknownBlob>::new(bytes.into()))
            .unwrap()
    }

    fn empty_metadata(store: &mut MemoryRepo) -> Inline<Handle<SimpleArchive>> {
        store.put::<SimpleArchive, _>(TribleSet::new()).unwrap()
    }

    fn data(handle: BlobHandle) -> CollectionData {
        Inline::new(handle.raw)
    }

    fn insert_commit(
        store: &mut MemoryRepo,
        collection: CollectionHandle,
        signer: &SigningKey,
        member: BlobHandle,
    ) -> CollectionCommit {
        let metadata = empty_metadata(store);
        let commit = CollectionCommit::sign(signer, collection, data(member), metadata);
        store.insert(CollectionRecord::Commit(commit)).unwrap();
        commit
    }

    fn store_bundle(store: &mut MemoryRepo, bundle: CapabilityProofBundle) {
        let (proof, claims) = bundle.into_parts();
        for claim in claims {
            store.put::<SimpleArchive, _>(claim).unwrap();
        }
        store.insert_proof(proof).unwrap();
    }

    fn build(store: &mut MemoryRepo, instant: f64) -> DisclosureSnapshot {
        DisclosureSnapshot::build_at(&store.snapshot().unwrap(), Epoch::from_tai_seconds(instant))
            .unwrap()
    }

    #[test]
    fn public_commit_discloses_only_aligned_resident_closure() {
        let signer = key(1);
        let mut store = MemoryRepo::default();
        let collection = store.collection("public", open_policy()).unwrap();
        let aligned = put_blob(&mut store, b"aligned child".to_vec());
        let unaligned = put_blob(&mut store, b"unaligned child".to_vec());
        let mut parent_bytes = vec![0; 96];
        parent_bytes[..32].copy_from_slice(&aligned.raw);
        parent_bytes[33..65].copy_from_slice(&unaligned.raw);
        let parent = put_blob(&mut store, parent_bytes);
        let commit = insert_commit(&mut store, collection.handle(), &signer, parent);

        let disclosure = build(&mut store, 0.0);
        let descriptor: BlobHandle = collection.handle().transmute();
        let metadata: BlobHandle = commit.metadata().transmute();

        for handle in [descriptor, parent, metadata, aligned] {
            assert!(disclosure.contains(collection.handle(), handle));
            assert!(disclosure.public_contains(handle));
        }
        assert!(!disclosure.contains(collection.handle(), unaligned));
        assert!(!disclosure.public_contains(unaligned));
    }

    #[test]
    fn unauthorized_and_forged_commits_are_inert() {
        let root = key(2);
        let stranger = key(3);
        let mut store = MemoryRepo::default();
        let collection = store
            .collection(
                "write-restricted",
                CollectionPolicy::new(
                    AdmissionPolicy::Open,
                    AdmissionPolicy::direct(root.verifying_key()),
                ),
            )
            .unwrap();
        let unauthorized = put_blob(&mut store, b"unauthorized".to_vec());
        insert_commit(&mut store, collection.handle(), &stranger, unauthorized);

        let forged_member = put_blob(&mut store, b"forged".to_vec());
        let mut forged = CollectionCommit::sign(
            &root,
            collection.handle(),
            data(forged_member),
            empty_metadata(&mut store),
        )
        .to_bytes();
        forged[128] ^= 1;
        store
            .insert(CollectionRecord::Commit(CollectionCommit::from_bytes(
                forged,
            )))
            .unwrap();

        let disclosure = build(&mut store, 0.0);
        assert!(!disclosure.contains(collection.handle(), unauthorized));
        assert!(!disclosure.public_contains(unauthorized));
        assert!(!disclosure.contains(collection.handle(), forged_member));
        assert!(!disclosure.public_contains(forged_member));
    }

    #[test]
    fn restricted_closure_is_scoped_and_never_bearer_public() {
        let writer = key(4);
        let read_root = key(5);
        let mut store = MemoryRepo::default();
        let restricted = store
            .collection(
                "restricted",
                CollectionPolicy::new(
                    AdmissionPolicy::direct(read_root.verifying_key()),
                    AdmissionPolicy::Open,
                ),
            )
            .unwrap();
        let other = store.collection("other", open_policy()).unwrap();
        let secret = put_blob(&mut store, b"secret".to_vec());
        let other_member = put_blob(&mut store, b"other".to_vec());
        insert_commit(&mut store, restricted.handle(), &writer, secret);
        insert_commit(&mut store, other.handle(), &writer, other_member);

        let disclosure = build(&mut store, 0.0);
        assert!(disclosure.contains(restricted.handle(), secret));
        assert!(!disclosure.public_contains(secret));
        assert!(!disclosure.contains(other.handle(), secret));
        assert!(disclosure.contains(other.handle(), other_member));
        assert!(disclosure.public_contains(other_member));
    }

    #[test]
    fn unsigned_equations_cannot_release_uncommitted_blobs() {
        let writer = key(6);
        let mut store = MemoryRepo::default();
        let collection = store.collection("equations", open_policy()).unwrap();
        let seed = put_blob(&mut store, b"seed".to_vec());
        let private_input = put_blob(&mut store, b"private input".to_vec());
        let private_output = put_blob(&mut store, b"private output".to_vec());
        insert_commit(&mut store, collection.handle(), &writer, seed);
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.handle(),
                data(seed),
                data(private_input),
                data(private_output),
            )))
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.handle(),
                data(private_input),
                data(private_output),
            )))
            .unwrap();

        let disclosure = build(&mut store, 0.0);
        assert!(disclosure.public_contains(seed));
        for handle in [private_input, private_output] {
            assert!(!disclosure.contains(collection.handle(), handle));
            assert!(!disclosure.public_contains(handle));
        }
    }

    #[test]
    fn absent_commit_root_appears_only_in_a_later_store_snapshot() {
        let writer = key(7);
        let mut store = MemoryRepo::default();
        let collection = store.collection("late-root", open_policy()).unwrap();
        let late_blob = Blob::<UnknownBlob>::new(b"arrives later".to_vec().into());
        let late_handle = late_blob.get_handle();
        insert_commit(&mut store, collection.handle(), &writer, late_handle);

        let before = build(&mut store, 0.0);
        assert!(!before.contains(collection.handle(), late_handle));
        assert!(!before.public_contains(late_handle));

        store.put::<UnknownBlob, _>(late_blob).unwrap();
        let after = build(&mut store, 0.0);
        assert!(after.contains(collection.handle(), late_handle));
        assert!(after.public_contains(late_handle));
        assert!(!before.public_contains(late_handle));
    }

    #[test]
    fn malformed_descriptor_does_not_suppress_an_unrelated_collection() {
        let writer = key(8);
        let mut store = MemoryRepo::default();
        let valid = store.collection("valid", open_policy()).unwrap();
        let valid_member = put_blob(&mut store, b"valid member".to_vec());
        insert_commit(&mut store, valid.handle(), &writer, valid_member);

        let malformed_descriptor = store
            .put::<SimpleArchive, _>(Blob::<SimpleArchive>::new(vec![1].into()))
            .unwrap();
        let malformed_member = put_blob(&mut store, b"malformed member".to_vec());
        insert_commit(&mut store, malformed_descriptor, &writer, malformed_member);

        let disclosure = build(&mut store, 0.0);
        assert!(disclosure.public_contains(valid_member));
        assert!(!disclosure.contains(malformed_descriptor, malformed_member));
        assert!(!disclosure.public_contains(malformed_member));
    }

    #[test]
    fn expired_write_proof_changes_future_policy_not_past_snapshots() {
        let root = key(9);
        let writer = key(10);
        let mut store = MemoryRepo::default();
        let collection = store
            .collection(
                "expiring-write",
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
        let member = put_blob(&mut store, b"time bounded".to_vec());
        insert_commit(&mut store, collection.handle(), &writer, member);

        let during = build(&mut store, 5.0);
        let after = build(&mut store, 11.0);
        assert!(during.public_contains(member));
        assert_eq!(
            during.public_valid_through(),
            Some(Epoch::from_tai_seconds(10.0))
        );
        assert!(!after.public_contains(member));
        assert_eq!(after.public_valid_through(), None);
        assert!(during.public_contains(member));
    }

    #[test]
    fn shared_descendants_are_emitted_once() {
        let writer = key(11);
        let mut store = MemoryRepo::default();
        let collection = store.collection("shared-dag", open_policy()).unwrap();
        let child = put_blob(&mut store, b"shared child".to_vec());
        let first = put_blob(&mut store, child.raw.to_vec());
        let mut second_bytes = child.raw.to_vec();
        second_bytes.extend_from_slice(b"different parent");
        let second = put_blob(&mut store, second_bytes);
        insert_commit(&mut store, collection.handle(), &writer, first);
        insert_commit(&mut store, collection.handle(), &writer, second);

        let disclosure = build(&mut store, 0.0);
        let mut seen = 0;
        disclosure.for_collection(collection.handle(), |handle| {
            if handle == child {
                seen += 1;
            }
        });
        assert_eq!(seen, 1);
    }
}
