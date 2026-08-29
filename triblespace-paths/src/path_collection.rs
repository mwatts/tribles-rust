//! Exact-cover materialization of native path-summary collections.

// Reach arrives here as a builder argument; only the tests name a
// particular one.
use std::error::Error;
use std::fmt;
use std::sync::Arc;
#[cfg(test)]
use triblespace_core::collection::reach;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::IntoBlob;
use triblespace_core::collection::exact_derived::{
    ExactDerivedCollection, ExactDerivedCollectionError,
};
use triblespace_core::collection::simplearchive_union;
use triblespace_core::collection::{
    CollectionHandle, CollectionStore, Cover, CoverAttachment, TryFromCover, VerifyingKey,
};
use triblespace_core::repo::{ArtifactOfferStore, BlobStore, BlobStoreMeta};
use triblespace_core::trible::Fragment;

use crate::path_summary_union::{self, PathSummaryView, RegularPathMapping};
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
}

impl fmt::Display for PathSummaryCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collection(source) => source.fmt(f),
            Self::Summary(source) => source.fmt(f),
            Self::Index(source) => source.fmt(f),
        }
    }
}

impl Error for PathSummaryCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Collection(source) => Some(source),
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
    name: String,
    source_authority: VerifyingKey,
    automaton: Automaton,
    source_reach: Fragment,
    authority: VerifyingKey,
    reach: Fragment,
}

impl PathSummaryCollection {
    /// Construct the canonical path projection for one named root and
    /// `automaton`.
    /// `source_authority` and `source_reach` complete the root's identity;
    /// `authority` and `reach` belong to this projection. A path summary over private
    /// material can be a perfectly reasonable thing to publish, and a private
    /// summary over published material an equally reasonable thing to keep, so
    /// neither answer is derived from the other.
    pub fn new(
        name: impl Into<String>,
        source_authority: VerifyingKey,
        automaton: Automaton,
        source_reach: Fragment,
        authority: VerifyingKey,
        reach: Fragment,
    ) -> Self {
        Self {
            name: name.into(),
            source_authority,
            automaton,
            source_reach,
            authority,
            reach,
        }
    }

    /// How far the source collection may travel.
    pub fn source_reach(&self) -> &Fragment {
        &self.source_reach
    }

    /// How far this projection may travel.
    pub fn reach(&self) -> &Fragment {
        &self.reach
    }

    /// Name of the root collection this projection is taken over.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Mandatory capability trust root declared by the source collection.
    pub fn source_authority(&self) -> VerifyingKey {
        self.source_authority
    }

    /// Mandatory capability trust root declared by this projection.
    pub fn authority(&self) -> VerifyingKey {
        self.authority
    }

    /// Fixed automaton whose fingerprint participates in collection identity.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// Canonical source `SimpleArchive` collection descriptor facts.
    pub fn source_descriptor(&self) -> Fragment {
        simplearchive_union::descriptor(
            &self.name,
            self.source_authority,
            self.source_reach.clone(),
        )
    }

    /// Identity of the source collection this projection reads.
    pub fn source_collection(&self) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(self.source_descriptor().into_facts()).get_handle()
    }

    /// Canonical target path-summary collection descriptor.
    pub fn descriptor(&self) -> Fragment {
        path_summary_union::descriptor(
            self.source_collection(),
            &self.automaton,
            self.authority,
            self.reach.clone(),
        )
    }

    /// Identity of the path summary this projection maintains.
    pub fn collection(&self) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(self.descriptor().into_facts()).get_handle()
    }

    /// Attach the exact endpoint relation already resident for `source_cover`.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<SimpleArchive>,
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel()?.attach_exact(store, source_cover)?;
        self.index_from_cover(cover).map(Arc::new)
    }

    /// Ensure and attach the exact endpoint relation for `source_cover`.
    ///
    /// Existing source merges, target merges, and derivations are reused. New
    /// target blobs precede unsigned records, no flush is implied, and a fresh
    /// pass proves the frozen cover before path closure runs once.
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<SimpleArchive>,
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel()?.ensure_exact(store, source_cover)?;
        self.index_from_cover(cover).map(Arc::new)
    }

    fn kernel(
        &self,
    ) -> Result<
        ExactDerivedCollection<SimpleArchive, PathSummaryBlob, RegularPathMapping>,
        ExactDerivedCollectionError,
    > {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }

    fn index_from_cover(
        &self,
        cover: CoverAttachment<PathSummaryBlob>,
    ) -> Result<PathIndex, PathSummaryCollectionError> {
        let cover = PathSummaryView::try_from_cover(cover)
            .expect("constructing a lazy path-summary view is infallible");
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

    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::{Blob, BlobEncoding, IntoBlob};
    use triblespace_core::capability::{
        CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
        CapabilityResource,
    };
    use triblespace_core::collection::{
        CapabilityPresentation, CollectionCommit, CollectionDerive, CollectionMerge,
        CollectionRecord, CollectionStoreExt, ACTION_WRITE,
    };
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

    impl triblespace_core::repo::ArtifactOfferStore for CollectionOnly {
        type OfferError = <MemoryRepo as triblespace_core::repo::ArtifactOfferStore>::OfferError;

        fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
        where
            I: IntoIterator<Item = triblespace_core::repo::ArtifactHandle>,
        {
            self.0.offer_all(handles)
        }

        fn offers_snapshot(
            &mut self,
        ) -> Result<triblespace_core::repo::ArtifactOfferSnapshot, Self::OfferError> {
            self.0.offers_snapshot()
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

    fn test_paths(name: String, automaton: Automaton) -> PathSummaryCollection {
        PathSummaryCollection::new(
            name,
            test_authority(),
            automaton,
            reach::private(),
            test_authority(),
            reach::private(),
        )
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
        name: &str,
        key: u8,
        data: &Blob<SimpleArchive>,
    ) -> CollectionCommit {
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            collection_of(&simplearchive_union::descriptor(
                name,
                test_authority(),
                reach::private(),
            )),
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
        let collection = store
            .collection::<SimpleArchive>(paths.source_descriptor())
            .unwrap();
        assert_eq!(collection.handle(), paths.source_collection());
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
        let presentations: Vec<_> = writers
            .into_iter()
            .map(|writer| {
                CapabilityPresentation::new(
                    writer,
                    CapabilityProofBundle::issue_root(
                        &authority,
                        CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
                        writer,
                    )
                    .unwrap(),
                )
            })
            .collect();
        store.cover(collection, &presentations).unwrap()
    }

    fn records(store: &mut CollectionOnly) -> Vec<CollectionRecord> {
        store.records().unwrap().map(Result::unwrap).collect()
    }

    fn assert_cross_fragment_path(index: &PathIndex) {
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(3))));
    }

    #[test]
    fn source_and_target_authorities_are_mandatory_and_independent() {
        let name = test_name("c9");
        let source_authority = test_authority();
        let target_authority = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let other_source_authority = SigningKey::from_bytes(&[3; 32]).verifying_key();
        let collection = PathSummaryCollection::new(
            name.clone(),
            source_authority,
            plus(),
            reach::private(),
            target_authority,
            reach::private(),
        );
        let other_source = PathSummaryCollection::new(
            name,
            other_source_authority,
            plus(),
            reach::private(),
            target_authority,
            reach::private(),
        );

        assert_eq!(collection.source_authority(), source_authority);
        assert_eq!(collection.authority(), target_authority);
        assert_ne!(
            collection.source_collection(),
            other_source.source_collection()
        );
        assert_ne!(collection.collection(), other_source.collection());
    }

    #[test]
    fn malformed_fixed_representation_capacity_is_fatal() {
        let automaton = Automaton::new(u32::MAX, [0], [0], []).unwrap();
        let paths = test_paths(test_name("c9"), automaton.clone());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&crate::automaton_fingerprint(&automaton).raw);
        bytes.extend_from_slice(&automaton.state_count().to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&[2; 32]);
        let persisted = Blob::<PathSummaryBlob>::new(bytes.into());
        assert!(matches!(
            <PathSummaryBlob as triblespace_core::collection::CollectionEncoding>::validate_member(
                &paths.descriptor(),
                &persisted,
            ),
            Err(triblespace_core::collection::CollectionOperationError::Fatal(_))
        ));
    }

    #[test]
    fn empty_cover_is_local_bottom_and_writes_nothing() {
        let mut store = CollectionOnly::default();
        let paths = test_paths(test_name("c9"), plus());
        let collection = store
            .collection::<SimpleArchive>(paths.source_descriptor())
            .unwrap();
        let blobs = store.0.blobs.len();
        let record_count = records(&mut store).len();
        let cover = store.cover(collection, &[]).unwrap();
        let index = paths.ensure_exact(&mut store, &cover).unwrap();
        assert_eq!(index.accepted_pair_count(), 0);
        assert_eq!(store.0.blobs.len(), blobs);
        assert_eq!(records(&mut store).len(), record_count);
    }

    #[test]
    fn missing_then_ensure_closes_cross_fragment_path() {
        let name = test_name("c9");
        let paths = test_paths(name.clone(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let cover = source_cover(&mut store, &paths, [first, second]);
        assert!(matches!(
            paths.attach_exact(&mut store, &cover),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::IncompleteCover { unsupported_members, .. }
            ))
                if unsupported_members.len() == 2
        ));
        assert_cross_fragment_path(&paths.ensure_exact(&mut store, &cover).unwrap());
        assert_cross_fragment_path(&paths.attach_exact(&mut store, &cover).unwrap());
    }

    #[test]
    fn old_cover_ignores_later_commit_and_its_cache_equation() {
        let name = test_name("c9");
        let paths = test_paths(name.clone(), plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let old_cover = source_cover(&mut store, &paths, [first, second]);
        paths.ensure_exact(&mut store, &old_cover).unwrap();

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

        let old = paths.attach_exact(&mut store, &old_cover).unwrap();
        assert!(!old.contains(&RawInline::from(id(1)), &RawInline::from(id(4))));
    }

    #[test]
    fn duplicate_data_provenance_shares_one_derive() {
        let name = test_name("c9");
        let paths = test_paths(name.clone(), plus());
        let mut store = CollectionOnly::default();
        let data = put_data(&mut store, &edge(1, 2));
        let first = signed_commit(&mut store, &name, 1, &data);
        let second = signed_commit(&mut store, &name, 2, &data);
        publish(&mut store, first);
        publish(&mut store, second);
        let cover = source_cover(&mut store, &paths, [first, first, second]);
        paths.ensure_exact(&mut store, &cover).unwrap();
        let derives = records(&mut store)
            .into_iter()
            .filter(|record| {
                matches!(record, CollectionRecord::Derive(claim)
                if claim.collection() == paths.collection())
            })
            .count();
        assert_eq!(derives, 1);
        paths.attach_exact(&mut store, &cover).unwrap();
    }

    #[test]
    fn derive_before_commit_is_inert_then_becomes_live() {
        let name = test_name("c9");
        let paths = test_paths(name.clone(), plus());
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
        let empty_cover = source_cover(&mut store, &paths, [commit]);
        assert!(empty_cover.is_empty());
        assert_eq!(
            paths
                .attach_exact(&mut store, &empty_cover)
                .unwrap()
                .accepted_pair_count(),
            0,
        );

        let other_name = test_name("c9-other");
        let other_paths = test_paths(other_name.clone(), plus());
        let other_commit = signed_commit(&mut store, &other_name, 7, &source);
        publish(&mut store, other_commit);
        let wrong_cover = source_cover(&mut store, &other_paths, [other_commit]);
        assert!(matches!(
            paths.attach_exact(&mut store, &wrong_cover),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::InvalidCover(_)
            ))
        ));

        publish(&mut store, commit);
        let cover = source_cover(&mut store, &paths, [commit]);
        let attached = paths.attach_exact(&mut store, &cover).unwrap();
        assert!(attached.contains(&RawInline::from(id(1)), &RawInline::from(id(2))));
    }

    #[test]
    fn resident_source_merge_is_lowered_once() {
        let name = test_name("c9");
        let paths = test_paths(name.clone(), plus());
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
        let cover = source_cover(&mut store, &paths, [first, second]);
        assert_cross_fragment_path(&paths.ensure_exact(&mut store, &cover).unwrap());
        let inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim) if claim.collection() == paths.collection() => {
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
        let paths = test_paths(name.clone(), plus());
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

        let cover = source_cover(&mut store, &paths, [first, second]);
        assert_cross_fragment_path(&paths.ensure_exact(&mut store, &cover).unwrap());
        let mut inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim) if claim.collection() == paths.collection() => {
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
        let paths = test_paths(name.clone(), plus());
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
            PathSummaryBlob::join(&left_summary, &right_summary, paths.automaton()).unwrap();
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
        let source_cover = source_cover(&mut store, &paths, [first, second]);
        let cover = paths
            .kernel()
            .unwrap()
            .attach_exact(&mut store, &source_cover)
            .unwrap();
        assert_eq!(cover.len(), 1);
        assert_eq!(cover.members()[0].0, joined.get_handle());
        assert_cross_fragment_path(&paths.attach_exact(&mut store, &source_cover).unwrap());
    }

    #[test]
    fn absent_source_bytes_report_the_member() {
        let name = test_name("c9");
        let paths = test_paths(name.clone(), plus());
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
        let cover = source_cover(&mut store, &paths, [commit]);
        assert!(matches!(
            paths.attach_exact(&mut store, &cover),
            Err(PathSummaryCollectionError::Collection(
                ExactDerivedCollectionError::IncompleteMember(found)
            )) if found == commit.data()
        ));
    }
}
