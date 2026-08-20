//! Exact-ticket materialization of native path-summary collections.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::IntoBlob;
use triblespace_core::collection::exact_derived::{
    ExactAlgebraError, ExactCover, ExactDerivedAlgebra, ExactDerivedCollection,
    ExactDerivedCollectionError,
};
use triblespace_core::collection::records::CollectionName;
use triblespace_core::collection::simplearchive_union;
use triblespace_core::collection::{
    CollectionCommit, CollectionHandle, CollectionStore, VerifyingKey,
};
use triblespace_core::repo::{BlobStore, BlobStoreMeta};
use triblespace_core::trible::Fragment;

use crate::path_summary_union;
use crate::{Automaton, PathError, PathIndex, PathSummaryBlob, PathSummaryBlobError};
use path_summary_union::PathSummaryUnionError;

/// Failure to validate, complete, or materialize one exact path ticket.
#[derive(Debug)]
pub enum PathSummaryCollectionError {
    /// Exact-ticket authority, resolution, construction, or storage failed.
    Collection(ExactDerivedCollectionError),
    /// Canonical path-summary construction failed.
    Algebra(PathSummaryUnionError),
    /// A selected summary did not decode under the fixed automaton.
    Summary(PathSummaryBlobError),
    /// Closing the joined summary into the accepted endpoint relation failed.
    Index(PathError),
}

impl fmt::Display for PathSummaryCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collection(source) => source.fmt(f),
            Self::Algebra(source) => source.fmt(f),
            Self::Summary(source) => source.fmt(f),
            Self::Index(source) => source.fmt(f),
        }
    }
}

impl Error for PathSummaryCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Collection(source) => Some(source),
            Self::Algebra(source) => Some(source),
            Self::Summary(source) => Some(source),
            Self::Index(source) => Some(source),
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
    name: CollectionName,
    team: VerifyingKey,
    automaton: Automaton,
}

impl PathSummaryCollection {
    /// Construct the canonical path projection for one named root and
    /// `automaton`.
    pub fn new(name: CollectionName, team: VerifyingKey, automaton: Automaton) -> Self {
        Self {
            name,
            team,
            automaton,
        }
    }

    /// Name of the root collection this projection is taken over.
    pub fn name(&self) -> &CollectionName {
        &self.name
    }

    /// Team owning the root collection this projection is taken over.
    pub fn team(&self) -> VerifyingKey {
        self.team
    }

    /// Fixed automaton whose fingerprint participates in collection identity.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// Canonical source `SimpleArchive` collection descriptor facts.
    pub fn source_descriptor(&self) -> Fragment {
        simplearchive_union::descriptor(&self.name, self.team)
    }

    /// Identity of the source collection this projection reads.
    pub fn source_collection(&self) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(self.source_descriptor().into_facts()).get_handle()
    }

    /// Canonical target path-summary collection descriptor.
    pub fn descriptor(&self) -> Fragment {
        path_summary_union::descriptor(self.source_collection(), &self.automaton)
    }

    /// Identity of the path summary this projection maintains.
    pub fn collection(&self) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(self.descriptor().into_facts()).get_handle()
    }

    /// Attach the exact endpoint relation already resident for `ticket`.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel().attach_exact(store, ticket, self)?;
        self.index_from_cover(cover).map(Arc::new)
    }

    /// Ensure and attach the exact endpoint relation for `ticket`.
    ///
    /// Existing source merges, target merges, and derivations are reused. New
    /// target blobs precede unsigned records, no flush is implied, and a fresh
    /// pass proves the frozen ticket before path closure runs once.
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel().ensure_exact(store, ticket, self)?;
        self.index_from_cover(cover).map(Arc::new)
    }

    fn kernel(&self) -> ExactDerivedCollection<SimpleArchive, PathSummaryBlob> {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }

    fn index_from_cover(
        &self,
        cover: ExactCover<PathSummaryBlob>,
    ) -> Result<PathIndex, PathSummaryCollectionError> {
        let mut joined = path_summary_union::empty(&self.automaton);
        for segment in cover.into_blobs() {
            joined = path_summary_union::join(&joined, &segment, &self.automaton)
                .map_err(PathSummaryCollectionError::Algebra)?;
        }
        let summary = PathSummaryBlob::decode(joined, &self.automaton)
            .map_err(PathSummaryCollectionError::Summary)?;
        PathIndex::from_summary(summary).map_err(PathSummaryCollectionError::Index)
    }
}

impl ExactDerivedAlgebra<SimpleArchive, PathSummaryBlob> for PathSummaryCollection {
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &triblespace_core::blob::Blob<SimpleArchive>,
    ) -> Result<(), ExactAlgebraError> {
        if *descriptor != self.source_descriptor() {
            return Err(ExactAlgebraError::Fatal(
                "source descriptor does not match this path collection".to_owned(),
            ));
        }
        simplearchive_union::validate_element(source)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn validate_target(
        &self,
        descriptor: &Fragment,
        target: &triblespace_core::blob::Blob<PathSummaryBlob>,
    ) -> Result<(), ExactAlgebraError> {
        if *descriptor != self.descriptor() {
            return Err(ExactAlgebraError::Fatal(
                "target descriptor does not match this path collection".to_owned(),
            ));
        }
        PathSummaryBlob::decode(target.clone(), &self.automaton)
            .map(|_| ())
            // These bytes are persisted evidence. Even a capacity-looking
            // header is malformed input, not a reason to refine the cover.
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn join_source(
        &self,
        low: &triblespace_core::blob::Blob<SimpleArchive>,
        high: &triblespace_core::blob::Blob<SimpleArchive>,
    ) -> Result<triblespace_core::blob::Blob<SimpleArchive>, ExactAlgebraError> {
        simplearchive_union::join(low, high)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn derive(
        &self,
        source: &triblespace_core::blob::Blob<SimpleArchive>,
    ) -> Result<triblespace_core::blob::Blob<PathSummaryBlob>, ExactAlgebraError> {
        path_summary_union::derive_element(source, &self.automaton).map_err(fatal_algebra_error)
    }

    fn join_target(
        &self,
        low: &triblespace_core::blob::Blob<PathSummaryBlob>,
        high: &triblespace_core::blob::Blob<PathSummaryBlob>,
    ) -> Result<triblespace_core::blob::Blob<PathSummaryBlob>, ExactAlgebraError> {
        path_summary_union::join(low, high, &self.automaton).map_err(fatal_algebra_error)
    }
}

fn fatal_algebra_error(error: PathSummaryUnionError) -> ExactAlgebraError {
    // Paths currently rejoins every selected shard before closure, so a finer
    // cover cannot evade fixed summary capacity. Reserve `Capacity` until the
    // public operation supports fragmented closure/materialization.
    ExactAlgebraError::Fatal(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::{Blob, BlobEncoding, IntoBlob};
    use triblespace_core::collection::{CollectionDerive, CollectionMerge, CollectionRecord};
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::inline::{InlineEncoding, RawInline};
    use triblespace_core::metadata;
    use triblespace_core::prelude::entity;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::trible::TribleSet;

    use crate::{Step, Transition};

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

    impl BlobStore for CollectionOnly {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.0.reader()
        }
    }

    impl CollectionStore for CollectionOnly {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = <MemoryRepo as CollectionStore>::InsertError;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.0.records()
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.0.insert(record)
        }
    }

    fn id(byte: u8) -> triblespace_core::id::Id {
        triblespace_core::id::Id::new([byte; 16]).unwrap()
    }

    /// The one team every collection in these tests belongs to.
    fn test_team() -> VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    fn test_name(name: &str) -> CollectionName {
        CollectionName::new(name).unwrap()
    }

    /// These tests only need an identity to file records under; the
    /// descriptor itself is never stored.
    fn collection_of(descriptor: &Fragment) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle()
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
        name: &CollectionName,
        key: u8,
        data: &Blob<SimpleArchive>,
    ) -> CollectionCommit {
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            collection_of(&simplearchive_union::descriptor(name, test_team())),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata,
        )
    }

    fn publish(store: &mut CollectionOnly, commit: CollectionCommit) {
        store.insert(CollectionRecord::Commit(commit)).unwrap();
    }

    fn records(store: &mut CollectionOnly) -> Vec<CollectionRecord> {
        store.records().unwrap().map(Result::unwrap).collect()
    }

    fn assert_cross_fragment_path(index: &PathIndex) {
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(3))));
    }

    #[test]
    fn exact_algebra_treats_fixed_representation_capacity_as_fatal() {
        for error in [
            PathSummaryUnionError::Merge(PathError::TooManyVertices { count: usize::MAX }),
            PathSummaryUnionError::Merge(PathError::ProductCarrierTooLarge {
                vertices: usize::MAX,
                states: u32::MAX,
            }),
            PathSummaryUnionError::Summary(PathSummaryBlobError::CapacityOverflow),
        ] {
            assert!(matches!(
                fatal_algebra_error(error),
                ExactAlgebraError::Fatal(_)
            ));
        }

        let automaton = Automaton::new(u32::MAX, [0], [0], []).unwrap();
        let paths = PathSummaryCollection::new(test_name("c9"), test_team(), automaton.clone());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::automaton_fingerprint(&automaton).raw);
        bytes.extend_from_slice(&automaton.state_count().to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&[2; 32]);
        let persisted = Blob::<PathSummaryBlob>::new(bytes.into());
        assert!(matches!(
            paths.validate_target(&paths.descriptor(), &persisted),
            Err(ExactAlgebraError::Fatal(_))
        ));
    }

    #[test]
    fn empty_ticket_is_local_bottom_and_writes_nothing() {
        let mut store = CollectionOnly::default();
        let paths = PathSummaryCollection::new(test_name("c9"), test_team(), plus());
        let blobs = store.0.blobs.len();
        let record_count = records(&mut store).len();
        let index = paths.ensure_exact(&mut store, &[]).unwrap();
        assert_eq!(index.accepted_pair_count(), 0);
        assert_eq!(store.0.blobs.len(), blobs);
        assert_eq!(records(&mut store).len(), record_count);
    }

    #[test]
    fn missing_then_ensure_closes_cross_fragment_path() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        assert!(matches!(
            paths.attach_exact(&mut store, &[first, second]),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::IncompleteCover { unsupported_commits, .. }
            ))
                if unsupported_commits.len() == 2
        ));
        assert_cross_fragment_path(&paths.ensure_exact(&mut store, &[first, second]).unwrap());
        assert_cross_fragment_path(&paths.attach_exact(&mut store, &[first, second]).unwrap());
    }

    #[test]
    fn old_ticket_ignores_later_commit_and_its_cache_equation() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        paths.ensure_exact(&mut store, &[first, second]).unwrap();

        let later = put_data(&mut store, &edge(3, 4));
        let third = signed_commit(&mut store, &name, 3, &later);
        publish(&mut store, third);
        let later_summary = path_summary_union::derive_element(&later, paths.automaton()).unwrap();
        store
            .put::<PathSummaryBlob, _>(later_summary.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                paths.collection(),
                third.data(),
                Handle::<PathSummaryBlob>::to_hash(later_summary.get_handle()),
            )))
            .unwrap();

        let old = paths.attach_exact(&mut store, &[first, second]).unwrap();
        assert!(!old.contains(&RawInline::from(id(1)), &RawInline::from(id(4))));
    }

    #[test]
    fn duplicate_data_provenance_shares_one_derive() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let data = put_data(&mut store, &edge(1, 2));
        let first = signed_commit(&mut store, &name, 1, &data);
        let second = signed_commit(&mut store, &name, 2, &data);
        publish(&mut store, first);
        publish(&mut store, second);
        paths
            .ensure_exact(&mut store, &[first, first, second])
            .unwrap();
        let derives = records(&mut store)
            .into_iter()
            .filter(|record| {
                matches!(record, CollectionRecord::Derive(claim)
                if claim.target() == paths.collection())
            })
            .count();
        assert_eq!(derives, 1);
        paths.attach_exact(&mut store, &[first, second]).unwrap();
    }

    #[test]
    fn derive_before_commit_is_inert_then_becomes_live() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let source = put_data(&mut store, &edge(1, 2));
        let commit = signed_commit(&mut store, &name, 7, &source);
        let output = path_summary_union::derive_element(&source, paths.automaton()).unwrap();
        store.put::<PathSummaryBlob, _>(output.clone()).unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                paths.collection(),
                commit.data(),
                Handle::<PathSummaryBlob>::to_hash(output.get_handle()),
            )))
            .unwrap();
        assert!(matches!(
            paths.attach_exact(&mut store, &[commit]),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::InvalidTicket(_)
            ))
        ));
        publish(&mut store, commit);
        let attached = paths.attach_exact(&mut store, &[commit]).unwrap();
        assert!(attached.contains(&RawInline::from(id(1)), &RawInline::from(id(2))));
    }

    #[test]
    fn resident_source_merge_is_lowered_once() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let joined = simplearchive_union::join(&left, &right).unwrap();
        store.put::<SimpleArchive, _>(joined.clone()).unwrap();
        let joined_data = Handle::<SimpleArchive>::to_hash(joined.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.source_collection(),
                first.data(),
                second.data(),
                joined_data,
            )))
            .unwrap();
        assert_cross_fragment_path(&paths.ensure_exact(&mut store, &[first, second]).unwrap());
        let inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.target() == paths.collection() =>
                {
                    Some(claim.mapping().0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(inputs, vec![joined_data]);
    }

    #[test]
    fn source_cover_can_overlap_an_already_supported_root() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let joined = simplearchive_union::join(&left, &right).unwrap();
        store.put::<SimpleArchive, _>(joined.clone()).unwrap();
        let joined_data = Handle::<SimpleArchive>::to_hash(joined.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.source_collection(),
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
                paths.collection(),
                first.data(),
                Handle::<PathSummaryBlob>::to_hash(left_summary.get_handle()),
            )))
            .unwrap();

        assert_cross_fragment_path(&paths.ensure_exact(&mut store, &[first, second]).unwrap());
        let mut inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.target() == paths.collection() =>
                {
                    Some(claim.mapping().0)
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
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let left_summary = path_summary_union::derive_element(&left, paths.automaton()).unwrap();
        let right_summary = path_summary_union::derive_element(&right, paths.automaton()).unwrap();
        for (input, output) in [(&left, &left_summary), (&right, &right_summary)] {
            store.put::<PathSummaryBlob, _>(output.clone()).unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    paths.collection(),
                    Handle::<SimpleArchive>::to_hash(input.get_handle()),
                    Handle::<PathSummaryBlob>::to_hash(output.get_handle()),
                )))
                .unwrap();
        }
        let joined =
            path_summary_union::join(&left_summary, &right_summary, paths.automaton()).unwrap();
        store.put::<PathSummaryBlob, _>(joined.clone()).unwrap();
        let joined_data = Handle::<PathSummaryBlob>::to_hash(joined.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.collection(),
                Handle::<PathSummaryBlob>::to_hash(left_summary.get_handle()),
                Handle::<PathSummaryBlob>::to_hash(right_summary.get_handle()),
                joined_data,
            )))
            .unwrap();
        let cover = paths
            .kernel()
            .attach_exact(&mut store, &[first, second], &paths)
            .unwrap();
        assert_eq!(cover.len(), 1);
        assert_eq!(cover.members()[0].0, joined_data);
        assert_cross_fragment_path(&paths.attach_exact(&mut store, &[first, second]).unwrap());
    }

    #[test]
    fn absent_source_bytes_report_the_commit() {
        let name = test_name("c9");
        let paths = PathSummaryCollection::new(name.clone(), test_team(), plus());
        let mut store = CollectionOnly::default();
        let absent = edge(1, 2).to_blob();
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[5; 32]),
            paths.source_collection(),
            Handle::<SimpleArchive>::to_hash(absent.get_handle()),
            metadata,
        );
        publish(&mut store, commit);
        assert!(matches!(
            paths.attach_exact(&mut store, &[commit]),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::IncompleteCommit(found)
            )) if found == commit.id()
        ));
    }
}
