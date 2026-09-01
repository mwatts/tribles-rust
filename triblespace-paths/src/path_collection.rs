//! Exact-cover materialization of native path-summary collections.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::collection::exact_derived::{
    ExactDerivedCollection, ExactDerivedCollectionError,
};
use triblespace_core::collection::{
    Collection, CollectionPolicy, CollectionRead, CollectionRegistrationError, CollectionStore,
    CollectionStoreExt, Cover,
};
use triblespace_core::repo::{BlobStore, BlobStoreMeta, BlobStorePut};

use crate::path_summary_union::{PathSummaryView, RegularPathMapping};
use crate::{Automaton, PathError, PathIndex, PathSummaryBlob, PathSummaryBlobError};

/// Failure to validate, complete, or materialize one exact path cover.
#[derive(Debug)]
pub enum PathSummaryCollectionError {
    /// Exact-cover resolution, construction, or storage failed.
    Collection(ExactDerivedCollectionError),
    /// A selected summary did not decode under the fixed automaton.
    Summary(PathSummaryBlobError),
    /// Closing the joined summary into the accepted endpoint relation failed.
    Index(PathError),
    /// A store snapshot for the cover could not be frozen.
    Snapshot(String),
}

impl fmt::Display for PathSummaryCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collection(source) => source.fmt(f),
            Self::Summary(source) => source.fmt(f),
            Self::Index(source) => source.fmt(f),
            Self::Snapshot(source) => write!(f, "freeze path-summary cover snapshot: {source}"),
        }
    }
}

impl Error for PathSummaryCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Collection(source) => Some(source),
            Self::Summary(source) => Some(source),
            Self::Index(source) => Some(source),
            Self::Snapshot(_) => None,
        }
    }
}

impl From<ExactDerivedCollectionError> for PathSummaryCollectionError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Collection(source)
    }
}

/// Canonical regular-path projection of one source `SimpleArchive` collection.
#[derive(Clone, Debug)]
pub struct PathSummaryCollection {
    source: Collection<SimpleArchive>,
    target: Collection<PathSummaryBlob>,
    automaton: Automaton,
}

impl PathSummaryCollection {
    /// Bind the facade to store-issued source and target collections.
    pub fn new(
        source: Collection<SimpleArchive>,
        target: Collection<PathSummaryBlob>,
        automaton: Automaton,
    ) -> Self {
        Self {
            source,
            target,
            automaton,
        }
    }

    /// Register a named source and its canonical regular-path projection.
    pub fn create<S>(
        store: &mut S,
        name: &str,
        source_policy: CollectionPolicy,
        automaton: Automaton,
        target_policy: CollectionPolicy,
    ) -> Result<Self, CollectionRegistrationError<<S as BlobStorePut>::PutError>>
    where
        S: BlobStore + CollectionStore,
    {
        let source = store.collection(name, source_policy)?;
        let target = store.derive(
            source,
            RegularPathMapping::new(automaton.clone()),
            target_policy,
        )?;
        Ok(Self::new(source, target, automaton))
    }

    /// Fixed automaton whose fingerprint participates in collection identity.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// Store-issued source collection this projection reads.
    pub fn source_collection(&self) -> Collection<SimpleArchive> {
        self.source
    }

    /// Store-issued path-summary target collection.
    pub fn collection(&self) -> Collection<PathSummaryBlob> {
        self.target
    }

    /// Attach the exact endpoint relation already resident for `source_cover`.
    pub fn attach<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<SimpleArchive>,
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        let cover = self.kernel()?.attach(store, source_cover)?;
        let snapshot = store
            .snapshot()
            .map_err(|source| PathSummaryCollectionError::Snapshot(source.to_string()))?;
        self.index_from_cover(&cover, &snapshot).map(Arc::new)
    }

    /// Ensure and attach the exact endpoint relation for `source_cover`.
    ///
    /// Existing source merges, target merges, and derivations are reused
    /// without algebra replay. Every newly computed target blob precedes its
    /// unsigned record, no flush is implied, and path closure runs once over
    /// the selected resident cover.
    pub fn ensure<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<SimpleArchive>,
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        let cover = store.ensure::<RegularPathMapping>(self.target, source_cover)?;
        let snapshot = store
            .snapshot()
            .map_err(|source| PathSummaryCollectionError::Snapshot(source.to_string()))?;
        self.index_from_cover(&cover, &snapshot).map(Arc::new)
    }

    fn kernel(
        &self,
    ) -> Result<ExactDerivedCollection<RegularPathMapping>, ExactDerivedCollectionError> {
        ExactDerivedCollection::new(self.source, self.target)
    }

    fn index_from_cover<R>(
        &self,
        cover: &Cover<PathSummaryBlob>,
        snapshot: &R,
    ) -> Result<PathIndex, PathSummaryCollectionError>
    where
        R: triblespace_core::repo::BlobStoreGet + BlobStoreMeta + CollectionRead,
    {
        let cover = cover
            .materialize::<PathSummaryView, _>(snapshot)
            .map_err(|source| PathSummaryCollectionError::Snapshot(source.to_string()))?;
        let mut joined = PathSummaryBlob::empty(&self.automaton);
        for segment in cover.into_blobs() {
            joined = PathSummaryBlob::join(&joined, &segment, &self.automaton)
                .map_err(PathSummaryCollectionError::Summary)?;
        }
        let summary = PathSummaryBlob::decode(joined, &self.automaton)
            .map_err(PathSummaryCollectionError::Summary)?;
        PathIndex::from_summary(summary).map_err(PathSummaryCollectionError::Index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ed25519_dalek::{SigningKey, VerifyingKey};
    use triblespace_core::blob::{Blob, BlobEncoding, IntoBlob, TryFromBlob};
    use triblespace_core::capability::{
        CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProof,
        CapabilityProofBundle, CapabilityResource,
    };
    use triblespace_core::collection::simplearchive_union;
    use triblespace_core::collection::{
        CollectionCommit, CollectionDerive, CollectionMerge, CollectionRecord, ACTION_WRITE,
    };
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::inline::{InlineEncoding, RawInline};
    use triblespace_core::metadata;
    use triblespace_core::prelude::entity;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{
        BlobStoreGet, BlobStorePut, CapabilityProofStore, SnapshotSource,
    };
    use triblespace_core::trible::{Fragment, TribleSet};

    use crate::{path_summary_union, Step, Transition};

    #[derive(Default)]
    struct CollectionOnly(MemoryRepo);

    impl BlobStorePut for CollectionOnly {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<E, T>(
            &mut self,
            item: T,
        ) -> Result<triblespace_core::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: triblespace_core::blob::IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.0.put(item)
        }
    }

    impl SnapshotSource for CollectionOnly {
        type Snapshot = <MemoryRepo as SnapshotSource>::Snapshot;
        type SnapshotError = <MemoryRepo as SnapshotSource>::SnapshotError;

        fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
            self.0.snapshot()
        }
    }

    impl CollectionStore for CollectionOnly {
        type InsertError = <MemoryRepo as CollectionStore>::InsertError;

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.0.insert(record)
        }
    }

    impl CapabilityProofStore for CollectionOnly {
        type InsertError = <MemoryRepo as CapabilityProofStore>::InsertError;

        fn insert_proof(&mut self, proof: CapabilityProof) -> Result<(), Self::InsertError> {
            self.0.insert_proof(proof)
        }
    }

    fn id(byte: u8) -> triblespace_core::id::Id {
        triblespace_core::id::Id::new([byte; 16]).unwrap()
    }

    /// The descriptor authority shared by ordinary test collections.
    fn test_authority() -> VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    fn test_name(name: &str) -> String {
        name.to_owned()
    }

    fn policy(authority: VerifyingKey) -> CollectionPolicy {
        CollectionPolicy::new(
            triblespace_core::collection::AdmissionPolicy::direct(authority),
            triblespace_core::collection::AdmissionPolicy::direct(authority),
        )
    }

    fn test_paths(
        store: &mut CollectionOnly,
        name: String,
        automaton: Automaton,
    ) -> PathSummaryCollection {
        PathSummaryCollection::create(
            store,
            &name,
            policy(test_authority()),
            automaton,
            policy(test_authority()),
        )
        .unwrap()
    }

    fn plus() -> Automaton {
        Automaton::new(
            2,
            [0],
            [1],
            [
                Transition::new(0, 1, Step::Forward(metadata::tag.id().into())),
                Transition::new(1, 1, Step::Forward(metadata::tag.id().into())),
            ],
        )
        .unwrap()
    }

    fn edge(source: u8, target: u8) -> TribleSet {
        let source = id(source);
        entity! { ExclusiveId::force_ref(&source) @ metadata::tag: id(target) }.into_facts()
    }

    fn put_data(store: &mut CollectionOnly, facts: &TribleSet) -> Blob<SimpleArchive> {
        let blob = facts.to_blob();
        store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        blob
    }

    fn signed_commit(
        store: &mut CollectionOnly,
        collection: Collection<SimpleArchive>,
        key: u8,
        data: &Blob<SimpleArchive>,
    ) -> CollectionCommit {
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            collection.handle(),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata,
        )
    }

    fn publish(store: &mut CollectionOnly, commit: CollectionCommit) {
        store.insert(CollectionRecord::Commit(commit)).unwrap();
    }

    fn source_cover(
        store: &mut CollectionOnly,
        paths: &PathSummaryCollection,
        commits: impl IntoIterator<Item = CollectionCommit>,
    ) -> Cover<SimpleArchive> {
        let collection = paths.source_collection();
        let authority = SigningKey::from_bytes(&[1; 32]);
        let mut writers: Vec<_> = commits
            .into_iter()
            .map(|commit| VerifyingKey::from_bytes(&commit.public_key().raw).unwrap())
            .filter(|writer| *writer != authority.verifying_key())
            .collect();
        writers.sort_unstable_by_key(VerifyingKey::to_bytes);
        writers.dedup();
        let atom = CapabilityAtom::new(
            CapabilityAction::new(ACTION_WRITE),
            CapabilityResource::from(collection.handle()),
        );
        for writer in writers {
            let bundle = CapabilityProofBundle::issue_root(
                &authority,
                CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
                writer,
            )
            .unwrap();
            let (proof, claims) = bundle.into_parts();
            for claim in claims {
                store.put::<SimpleArchive, _>(claim).unwrap();
            }
            store.insert_proof(proof).unwrap();
        }
        let snapshot = store.snapshot().unwrap();
        collection.admitted(&snapshot).unwrap()
    }

    fn records(store: &mut CollectionOnly) -> Vec<CollectionRecord> {
        store
            .snapshot()
            .unwrap()
            .records()
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    fn descriptor_for<L>(store: &mut CollectionOnly, collection: Collection<L>) -> Fragment
    where
        L: triblespace_core::collection::CollectionEncoding,
    {
        let snapshot = store.snapshot().unwrap();
        let blob: Blob<SimpleArchive> = snapshot.get(collection.handle()).unwrap();
        Fragment::from(TribleSet::try_from_blob(blob).unwrap())
    }

    fn assert_cross_fragment_path(index: &PathIndex) {
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(3))));
    }

    #[test]
    fn source_and_target_policies_are_independent() {
        let mut store = CollectionOnly::default();
        let name = test_name("c9");
        let source_authority = test_authority();
        let target_authority = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let other_source_authority = SigningKey::from_bytes(&[3; 32]).verifying_key();
        let collection = PathSummaryCollection::create(
            &mut store,
            &name,
            policy(source_authority),
            plus(),
            policy(target_authority),
        )
        .unwrap();
        let other_source = PathSummaryCollection::create(
            &mut store,
            &name,
            policy(other_source_authority),
            plus(),
            policy(target_authority),
        )
        .unwrap();

        assert_ne!(
            collection.source_collection(),
            other_source.source_collection()
        );
        assert_ne!(collection.collection(), other_source.collection());
    }

    #[test]
    fn malformed_fixed_representation_capacity_is_fatal() {
        let automaton = Automaton::new(u32::MAX, [0], [0], []).unwrap();
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, test_name("c9"), automaton.clone());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::automaton_fingerprint(&automaton).raw);
        bytes.extend_from_slice(&automaton.state_count().to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&[2; 32]);
        let persisted = Blob::<PathSummaryBlob>::new(bytes.into());
        let descriptor = descriptor_for(&mut store, paths.collection());
        let reader = store.snapshot().unwrap();
        assert!(matches!(
            <PathSummaryBlob as triblespace_core::collection::CollectionEncoding>::validate_member(
                &descriptor,
                &persisted,
                &reader,
            ),
            Err(triblespace_core::collection::CollectionOperationError::Fatal(_))
        ));
    }

    #[test]
    fn empty_cover_is_local_bottom_and_writes_nothing() {
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, test_name("c9"), plus());
        let collection = paths.source_collection();
        let blobs = store.0.blobs.len();
        let record_count = records(&mut store).len();
        let snapshot = store.snapshot().unwrap();
        let cover = collection.admitted(&snapshot).unwrap();
        let index = paths.ensure(&mut store, &cover).unwrap();
        assert_eq!(index.accepted_pair_count(), 0);
        assert_eq!(store.0.blobs.len(), blobs);
        assert_eq!(records(&mut store).len(), record_count);
    }

    #[test]
    fn missing_then_ensure_closes_cross_fragment_path() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, paths.source_collection(), 1, &left);
        let second = signed_commit(&mut store, paths.source_collection(), 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let cover = source_cover(&mut store, &paths, [first, second]);
        assert!(matches!(
            paths.attach(&mut store, &cover),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::IncompleteCover { unsupported_members, .. }
            ))
                if unsupported_members.len() == 2
        ));
        assert_cross_fragment_path(&paths.ensure(&mut store, &cover).unwrap());
        assert_cross_fragment_path(&paths.attach(&mut store, &cover).unwrap());
    }

    #[test]
    fn old_cover_ignores_later_commit_and_its_stored_equation() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, paths.source_collection(), 1, &left);
        let second = signed_commit(&mut store, paths.source_collection(), 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let old_cover = source_cover(&mut store, &paths, [first, second]);
        paths.ensure(&mut store, &old_cover).unwrap();

        let later = put_data(&mut store, &edge(3, 4));
        let third = signed_commit(&mut store, paths.source_collection(), 3, &later);
        publish(&mut store, third);
        let later_summary = path_summary_union::derive_element(&later, paths.automaton()).unwrap();
        store
            .put::<PathSummaryBlob, _>(later_summary.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                paths.collection().handle(),
                third.data(),
                Handle::<PathSummaryBlob>::to_hash(later_summary.get_handle()),
            )))
            .unwrap();

        let old = paths.attach(&mut store, &old_cover).unwrap();
        assert!(!old.contains(&RawInline::from(id(1)), &RawInline::from(id(4))));
    }

    #[test]
    fn duplicate_data_provenance_shares_one_derive() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let data = put_data(&mut store, &edge(1, 2));
        let first = signed_commit(&mut store, paths.source_collection(), 1, &data);
        let second = signed_commit(&mut store, paths.source_collection(), 2, &data);
        publish(&mut store, first);
        publish(&mut store, second);
        let cover = source_cover(&mut store, &paths, [first, first, second]);
        paths.ensure(&mut store, &cover).unwrap();
        let derives = records(&mut store)
            .into_iter()
            .filter(|record| {
                matches!(record, CollectionRecord::Derive(claim)
                if claim.collection() == paths.collection().handle())
            })
            .count();
        assert_eq!(derives, 1);
        paths.attach(&mut store, &cover).unwrap();
    }

    #[test]
    fn derive_before_commit_is_inert_then_becomes_live() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let source = put_data(&mut store, &edge(1, 2));
        let commit = signed_commit(&mut store, paths.source_collection(), 7, &source);
        let output = path_summary_union::derive_element(&source, paths.automaton()).unwrap();
        store.put::<PathSummaryBlob, _>(output.clone()).unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                paths.collection().handle(),
                commit.data(),
                Handle::<PathSummaryBlob>::to_hash(output.get_handle()),
            )))
            .unwrap();
        let empty_cover = source_cover(&mut store, &paths, [commit]);
        assert!(empty_cover.is_empty());
        assert_eq!(
            paths
                .attach(&mut store, &empty_cover)
                .unwrap()
                .accepted_pair_count(),
            0,
        );

        let other_name = test_name("c9-other");
        let other_paths = test_paths(&mut store, other_name, plus());
        let other_commit = signed_commit(&mut store, other_paths.source_collection(), 7, &source);
        publish(&mut store, other_commit);
        let wrong_cover = source_cover(&mut store, &other_paths, [other_commit]);
        assert!(matches!(
            paths.attach(&mut store, &wrong_cover),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::InvalidCover(_)
            ))
        ));

        publish(&mut store, commit);
        let cover = source_cover(&mut store, &paths, [commit]);
        let attached = paths.attach(&mut store, &cover).unwrap();
        assert!(attached.contains(&RawInline::from(id(1)), &RawInline::from(id(2))));
    }

    #[test]
    fn resident_source_merge_is_lowered_once() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, paths.source_collection(), 1, &left);
        let second = signed_commit(&mut store, paths.source_collection(), 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let joined = simplearchive_union::join(&left, &right).unwrap();
        store.put::<SimpleArchive, _>(joined.clone()).unwrap();
        let joined_data = Handle::<SimpleArchive>::to_hash(joined.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.source_collection().handle(),
                first.data(),
                second.data(),
                joined_data,
            )))
            .unwrap();
        let cover = source_cover(&mut store, &paths, [first, second]);
        assert_cross_fragment_path(&paths.ensure(&mut store, &cover).unwrap());
        let inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == paths.collection().handle() =>
                {
                    Some(claim.input())
                }
                _ => None,
            })
            .collect();
        assert_eq!(inputs, vec![joined_data]);
    }

    #[test]
    fn source_cover_can_overlap_an_already_supported_root() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, paths.source_collection(), 1, &left);
        let second = signed_commit(&mut store, paths.source_collection(), 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let joined = simplearchive_union::join(&left, &right).unwrap();
        store.put::<SimpleArchive, _>(joined.clone()).unwrap();
        let joined_data = Handle::<SimpleArchive>::to_hash(joined.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.source_collection().handle(),
                first.data(),
                second.data(),
                joined_data,
            )))
            .unwrap();

        let left_summary = path_summary_union::derive_element(&left, paths.automaton()).unwrap();
        store
            .put::<PathSummaryBlob, _>(left_summary.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                paths.collection().handle(),
                first.data(),
                Handle::<PathSummaryBlob>::to_hash(left_summary.get_handle()),
            )))
            .unwrap();

        let cover = source_cover(&mut store, &paths, [first, second]);
        assert_cross_fragment_path(&paths.ensure(&mut store, &cover).unwrap());
        let mut inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == paths.collection().handle() =>
                {
                    Some(claim.input())
                }
                _ => None,
            })
            .collect();
        inputs.sort_unstable();
        assert_eq!(
            inputs,
            vec![first.data().min(joined_data), first.data().max(joined_data)]
        );
        assert!(!inputs.contains(&second.data()));
    }

    #[test]
    fn existing_target_merge_is_the_single_physical_member() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, paths.source_collection(), 1, &left);
        let second = signed_commit(&mut store, paths.source_collection(), 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let left_summary = path_summary_union::derive_element(&left, paths.automaton()).unwrap();
        let right_summary = path_summary_union::derive_element(&right, paths.automaton()).unwrap();
        for (input, output) in [(&left, &left_summary), (&right, &right_summary)] {
            store.put::<PathSummaryBlob, _>(output.clone()).unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    paths.collection().handle(),
                    Handle::<SimpleArchive>::to_hash(input.get_handle()),
                    Handle::<PathSummaryBlob>::to_hash(output.get_handle()),
                )))
                .unwrap();
        }
        let joined =
            PathSummaryBlob::join(&left_summary, &right_summary, paths.automaton()).unwrap();
        store.put::<PathSummaryBlob, _>(joined.clone()).unwrap();
        let joined_data = Handle::<PathSummaryBlob>::to_hash(joined.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.collection().handle(),
                Handle::<PathSummaryBlob>::to_hash(left_summary.get_handle()),
                Handle::<PathSummaryBlob>::to_hash(right_summary.get_handle()),
                joined_data,
            )))
            .unwrap();
        let source_cover = source_cover(&mut store, &paths, [first, second]);
        let cover = paths
            .kernel()
            .unwrap()
            .attach(&mut store, &source_cover)
            .unwrap();
        assert_eq!(cover.len(), 1);
        assert_eq!(cover.members().next().unwrap(), joined.get_handle());
        assert_cross_fragment_path(&paths.attach(&mut store, &source_cover).unwrap());
    }

    #[test]
    fn absent_source_bytes_report_the_member() {
        let name = test_name("c9");
        let mut store = CollectionOnly::default();
        let paths = test_paths(&mut store, name, plus());
        let absent = edge(1, 2).to_blob();
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[5; 32]),
            paths.source_collection().handle(),
            Handle::<SimpleArchive>::to_hash(absent.get_handle()),
            metadata,
        );
        publish(&mut store, commit);
        let cover = source_cover(&mut store, &paths, [commit]);
        assert!(matches!(
            paths.attach(&mut store, &cover),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::IncompleteCover {
                    unsupported_members,
                    ..
                }
            )) if unsupported_members == vec![commit.data()]
        ));
    }
}
