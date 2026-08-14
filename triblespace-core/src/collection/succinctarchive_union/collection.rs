//! Exact-ticket facade for canonical raw SuccinctArchive collections.
//!
//! This remains separate from the legacy `SuccinctRollup` lifecycle. Unsigned
//! equations are reproducible cache evidence rather than authority or durable
//! receipts: attachment reconstructs collected intermediates in use-counted
//! scratch from authenticated source leaves, then freshly validates only the
//! resident artifacts selected by the physical cover. Target compaction is an
//! explicit maintenance call rather than background policy. Attachment rebuilds
//! Rank9 indexes in process memory and inherits the raw format's explicit
//! `u32::MAX` row/domain limit for each derived shard.

use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchive, SuccinctArchiveBlob, SuccinctArchiveError, UnionArchive,
};
use crate::collection::exact_derived::{
    ExactCover, ExactDerivedAlgebra, ExactDerivedCollection, ExactDerivedCollectionError,
};
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::simplearchive_union;
use crate::collection::{CollectionCommit, CollectionDescriptor, CollectionStore};
use crate::id::Id;
use crate::repo::{BlobStore, BlobStoreMeta};

/// Failure to complete or attach one exact Succinct ticket.
#[derive(Debug)]
pub enum SuccinctArchiveCollectionError {
    /// Exact-ticket authority, resolution, construction, or storage failed.
    Exact(ExactDerivedCollectionError),
    /// Explicit target compaction failed.
    Compaction(ExactTargetCompactionError),
    /// A selected canonical raw shard could not become a query runtime.
    Attach(SuccinctArchiveError),
}

impl fmt::Display for SuccinctArchiveCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(f),
            Self::Compaction(source) => source.fmt(f),
            Self::Attach(source) => write!(f, "attach raw SuccinctArchive cover: {source}"),
        }
    }
}

impl Error for SuccinctArchiveCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Compaction(source) => Some(source),
            Self::Attach(source) => Some(source),
        }
    }
}

impl From<ExactDerivedCollectionError> for SuccinctArchiveCollectionError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Exact(source)
    }
}

impl From<ExactTargetCompactionError> for SuccinctArchiveCollectionError {
    fn from(source: ExactTargetCompactionError) -> Self {
        Self::Compaction(source)
    }
}

/// Canonical raw SuccinctArchive projection of one scoped SimpleArchive union.
///
/// Signed source commits remain the only authority. Returned query sources
/// preserve the deterministic resident physical cover as Succinct shards;
/// attachment rebuilds each shard's process-local Rank9 runtime but neither
/// reads nor writes persisted Rank9 accelerator blobs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SuccinctArchiveCollection {
    scope: Id,
}

impl SuccinctArchiveCollection {
    /// Construct the canonical Succinct projection for `scope`.
    pub fn new(scope: Id) -> Self {
        Self { scope }
    }

    /// Dataset scope shared by source and target descriptors.
    pub fn scope(&self) -> Id {
        self.scope
    }

    /// Canonical source SimpleArchive-union descriptor.
    pub fn source_descriptor(&self) -> CollectionDescriptor {
        simplearchive_union::descriptor(self.scope)
    }

    /// Canonical target raw-SuccinctArchive-union descriptor.
    pub fn descriptor(&self) -> CollectionDescriptor {
        super::descriptor(self.scope)
    }

    /// Attach the exact resident Succinct cover for `ticket` without writing.
    ///
    /// An empty ticket returns one authority-free process-local empty shard;
    /// it is not a persisted target member or a provenance assertion.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel().attach_exact(store, ticket, self)?;
        self.attach_cover(cover)
    }

    /// Ensure missing raw derivations and attach the exact sharded cover.
    ///
    /// Completion writes descriptors and canonical raw output blobs before
    /// unsigned `DERIVE` records, performs no flush, and re-admits through a
    /// fresh reader. Existing source and target merges remain reusable.
    /// An empty ticket has the same local-only behavior as [`Self::attach_exact`].
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel().ensure_exact(store, ticket, self)?;
        self.attach_cover(cover)
    }

    /// Explicitly compact and attach the exact raw target cover for `ticket`.
    ///
    /// This first performs ordinary exact completion, then applies the fixed
    /// dyadic byte-size policy to canonical target members. All compacted blobs
    /// precede unsigned `MERGE` records, no flush or signed record is implied,
    /// and the returned cover is freshly re-admitted under the same ticket.
    pub fn compact_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = compact_exact_target(&self.kernel(), store, ticket, self)?;
        self.attach_cover(cover)
    }

    fn kernel(&self) -> ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob> {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }

    fn attach_cover(
        &self,
        cover: ExactCover<SuccinctArchiveBlob>,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError> {
        let mut segments = Vec::with_capacity(cover.len().max(1));
        if cover.is_empty() {
            let bottom: SuccinctArchive<OrderedUniverse> = super::empty()
                .try_from_blob()
                .map_err(SuccinctArchiveCollectionError::Attach)?;
            segments.push(bottom);
        } else {
            for raw in cover.into_blobs() {
                segments.push(
                    raw.try_from_blob()
                        .map_err(SuccinctArchiveCollectionError::Attach)?,
                );
            }
        }
        Ok(UnionArchive::new(segments))
    }
}

impl ExactDerivedAlgebra<SimpleArchive, SuccinctArchiveBlob> for SuccinctArchiveCollection {
    fn validate_source(
        &self,
        descriptor: &CollectionDescriptor,
        source: &crate::blob::Blob<SimpleArchive>,
    ) -> Result<(), String> {
        if *descriptor != self.source_descriptor() {
            return Err("source descriptor does not match this Succinct collection".to_owned());
        }
        simplearchive_union::validate_element(source).map_err(|error| error.to_string())
    }

    fn validate_target(
        &self,
        descriptor: &CollectionDescriptor,
        target: &crate::blob::Blob<SuccinctArchiveBlob>,
    ) -> Result<(), String> {
        if *descriptor != self.descriptor() {
            return Err("target descriptor does not match this Succinct collection".to_owned());
        }
        SuccinctArchiveBlob::merge(std::slice::from_ref(target))
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn join_source(
        &self,
        low: &crate::blob::Blob<SimpleArchive>,
        high: &crate::blob::Blob<SimpleArchive>,
    ) -> Result<crate::blob::Blob<SimpleArchive>, String> {
        simplearchive_union::join(low, high).map_err(|error| error.to_string())
    }

    fn derive(
        &self,
        source: &crate::blob::Blob<SimpleArchive>,
    ) -> Result<crate::blob::Blob<SuccinctArchiveBlob>, String> {
        super::derive_element(source).map_err(|error| error.to_string())
    }

    fn join_target(
        &self,
        low: &crate::blob::Blob<SuccinctArchiveBlob>,
        high: &crate::blob::Blob<SuccinctArchiveBlob>,
    ) -> Result<crate::blob::Blob<SuccinctArchiveBlob>, String> {
        super::join(low, high).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use crate::blob::{Blob, BlobEncoding, IntoBlob};
    use crate::collection::{CollectionData, CollectionDerive, CollectionMerge, CollectionRecord};
    use crate::inline::encodings::hash::Handle;
    use crate::inline::InlineEncoding;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::pile::{Pile, WantRewritePolicy};
    use crate::repo::{BlobStoreList, BlobStorePut, RetentionRoots};
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    /// Compile-time proof that the native API has no PinStore requirement.
    #[derive(Default)]
    struct CollectionOnly(MemoryRepo);

    impl BlobStorePut for CollectionOnly {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<E, T>(&mut self, item: T) -> Result<crate::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: crate::blob::IntoBlob<E>,
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

    struct PanicStore;

    impl BlobStorePut for PanicStore {
        type PutError = Infallible;

        fn put<E, T>(&mut self, _: T) -> Result<crate::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: crate::blob::IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            panic!("empty Succinct ticket attempted a blob write")
        }
    }

    impl BlobStore for PanicStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = Infallible;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            panic!("empty Succinct ticket opened a reader")
        }
    }

    impl CollectionStore for PanicStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            panic!("empty Succinct ticket scanned records")
        }

        fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
            panic!("empty Succinct ticket inserted a record")
        }
    }

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn row(entity: u8, value: u8) -> Trible {
        let mut raw = [value; TRIBLE_LEN];
        raw[..16].fill(entity);
        raw[16..32].fill(9);
        Trible::force_raw(raw).unwrap()
    }

    fn facts(rows: impl IntoIterator<Item = (u8, u8)>) -> TribleSet {
        let mut facts = TribleSet::new();
        for (entity, value) in rows {
            facts.insert(&row(entity, value));
        }
        facts
    }

    fn put_data(store: &mut CollectionOnly, facts: &TribleSet) -> Blob<SimpleArchive> {
        let blob = facts.to_blob();
        store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        blob
    }

    fn signed_commit(
        store: &mut CollectionOnly,
        scope: Id,
        key: u8,
        data: &Blob<SimpleArchive>,
    ) -> CollectionCommit {
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            simplearchive_union::descriptor(scope).handle(),
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

    fn data<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
    where
        Handle<E>: InlineEncoding,
    {
        Handle::<E>::to_hash(blob.get_handle())
    }

    fn attached_facts(archive: &UnionArchive<OrderedUniverse>) -> TribleSet {
        archive.iter().collect()
    }

    fn pile_commit(
        store: &mut Pile,
        collection: &SuccinctArchiveCollection,
        key: u8,
        source: &Blob<SimpleArchive>,
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
    ) -> CollectionCommit {
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            collection.source_descriptor().handle(),
            data(source),
            metadata,
        );
        store.insert(CollectionRecord::Commit(commit)).unwrap();
        commit
    }

    #[test]
    fn empty_ticket_is_one_authority_free_local_shard_and_performs_no_io() {
        let collection = SuccinctArchiveCollection::new(id(7));
        let mut store = PanicStore;
        for attached in [
            collection.attach_exact(&mut store, &[]).unwrap(),
            collection.ensure_exact(&mut store, &[]).unwrap(),
            collection.compact_exact(&mut store, &[]).unwrap(),
        ] {
            assert_eq!(attached.segment_count(), 1);
            assert_eq!(attached.iter().count(), 0);
        }
    }

    #[test]
    fn signed_empty_source_still_publishes_nonempty_ticket_provenance() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let source = put_data(&mut store, &TribleSet::new());
        let commit = signed_commit(&mut store, scope, 1, &source);
        publish(&mut store, commit);

        let attached = collection.ensure_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached.iter().count(), 0);
        let mappings: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == collection.source_descriptor().handle()
                        && claim.target() == collection.descriptor().handle() =>
                {
                    Some(claim.mapping())
                }
                _ => None,
            })
            .collect();
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].0, commit.data());
        collection.attach_exact(&mut store, &[commit]).unwrap();
    }

    #[test]
    fn missing_attach_then_ensure_builds_exact_raw_cover() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        assert!(matches!(
            collection.attach_exact(&mut store, &[first, second]),
            Err(SuccinctArchiveCollectionError::Exact(
                ExactDerivedCollectionError::IncompleteCover { .. }
            ))
        ));
        let attached = collection
            .ensure_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
        assert_eq!(attached.segment_count(), 2);
    }

    #[test]
    fn explicit_compaction_returns_one_exact_real_succinct_shard() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);

        let attached = collection
            .compact_exact(&mut store, &[second, first])
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
        assert!(records(&mut store).into_iter().any(|record| matches!(
            record,
            CollectionRecord::Merge(claim)
                if claim.collection() == collection.descriptor().handle()
        )));
    }

    #[test]
    fn duplicate_provenance_shares_one_raw_derive() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let first = signed_commit(&mut store, scope, 1, &source);
        let second = signed_commit(&mut store, scope, 2, &source);
        publish(&mut store, first);
        publish(&mut store, second);
        let attached = collection
            .ensure_exact(&mut store, &[first, first, second])
            .unwrap();
        assert_eq!(attached_facts(&attached), expected);
        let derives = records(&mut store)
            .into_iter()
            .filter(|record| {
                matches!(record, CollectionRecord::Derive(claim)
                if claim.source() == collection.source_descriptor().handle()
                    && claim.target() == collection.descriptor().handle())
            })
            .count();
        assert_eq!(derives, 1);
        collection
            .attach_exact(&mut store, &[first, second])
            .unwrap();
    }

    #[test]
    fn resident_source_merge_is_reused_as_one_shard() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let source_union = simplearchive_union::join(&left, &right).unwrap();
        store.put::<SimpleArchive, _>(source_union.clone()).unwrap();
        let source_union_data = data(&source_union);
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.source_descriptor().handle(),
                first.data(),
                second.data(),
                source_union_data,
            )))
            .unwrap();
        let attached = collection
            .ensure_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
        let inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == collection.source_descriptor().handle()
                        && claim.target() == collection.descriptor().handle() =>
                {
                    Some(claim.mapping().0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(inputs, vec![source_union_data]);
    }

    #[test]
    fn existing_target_merge_is_selected_as_one_shard() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let left_raw = super::super::derive_element(&left).unwrap();
        let right_raw = super::super::derive_element(&right).unwrap();
        for (input, output) in [(&left, &left_raw), (&right, &right_raw)] {
            store.put::<SuccinctArchiveBlob, _>(output.clone()).unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    collection.source_descriptor().handle(),
                    collection.descriptor().handle(),
                    data(input),
                    data(output),
                )))
                .unwrap();
        }
        let joined = super::super::join(&left_raw, &right_raw).unwrap();
        store.put::<SuccinctArchiveBlob, _>(joined.clone()).unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.descriptor().handle(),
                data(&left_raw),
                data(&right_raw),
                data(&joined),
            )))
            .unwrap();
        let attached = collection
            .attach_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
    }

    #[test]
    fn corrupt_upper_target_artifact_falls_back_to_valid_lower_cover() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);

        let left_raw = super::super::derive_element(&left).unwrap();
        let right_raw = super::super::derive_element(&right).unwrap();
        for (input, output) in [(&left, &left_raw), (&right, &right_raw)] {
            store.put::<SuccinctArchiveBlob, _>(output.clone()).unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    collection.source_descriptor().handle(),
                    collection.descriptor().handle(),
                    data(input),
                    data(output),
                )))
                .unwrap();
        }
        let joined = super::super::join(&left_raw, &right_raw).unwrap();
        let forged =
            Blob::<SuccinctArchiveBlob>::with_handle(left_raw.bytes.clone(), joined.get_handle());
        store.put::<SuccinctArchiveBlob, _>(forged).unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.descriptor().handle(),
                data(&left_raw),
                data(&right_raw),
                data(&joined),
            )))
            .unwrap();

        let attached = collection
            .attach_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(attached.segment_count(), 2);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
    }

    #[test]
    fn old_ticket_stays_stable_after_later_commit_and_cache() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let old_facts = facts([(1, 3)]);
        let old = put_data(&mut store, &old_facts);
        let first = signed_commit(&mut store, scope, 1, &old);
        publish(&mut store, first);
        collection.ensure_exact(&mut store, &[first]).unwrap();

        let later_facts = facts([(2, 4)]);
        let later = put_data(&mut store, &later_facts);
        let second = signed_commit(&mut store, scope, 2, &later);
        publish(&mut store, second);
        let later_raw = super::super::derive_element(&later).unwrap();
        store
            .put::<SuccinctArchiveBlob, _>(later_raw.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.source_descriptor().handle(),
                collection.descriptor().handle(),
                second.data(),
                data(&later_raw),
            )))
            .unwrap();
        assert_eq!(
            attached_facts(&collection.attach_exact(&mut store, &[first]).unwrap()),
            old_facts,
        );
    }

    #[test]
    fn missing_derive_output_is_not_support_and_ensure_rebuilds_it() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let commit = signed_commit(&mut store, scope, 1, &source);
        publish(&mut store, commit);
        let missing = super::super::derive_element(&source).unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.source_descriptor().handle(),
                collection.descriptor().handle(),
                commit.data(),
                data(&missing),
            )))
            .unwrap();
        assert!(matches!(
            collection.attach_exact(&mut store, &[commit]),
            Err(SuccinctArchiveCollectionError::Exact(
                ExactDerivedCollectionError::IncompleteCover { .. }
            ))
        ));
        assert_eq!(
            attached_facts(&collection.ensure_exact(&mut store, &[commit]).unwrap()),
            expected,
        );
    }

    #[test]
    fn ungrounded_source_superset_never_enters_smaller_ticket() {
        let scope = id(7);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let a = put_data(&mut store, &expected);
        let c = put_data(&mut store, &facts([(3, 5)]));
        let commit = signed_commit(&mut store, scope, 1, &a);
        publish(&mut store, commit);
        let superset = simplearchive_union::join(&a, &c).unwrap();
        store.put::<SimpleArchive, _>(superset.clone()).unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.source_descriptor().handle(),
                data(&a),
                data(&c),
                data(&superset),
            )))
            .unwrap();
        let attached = collection.ensure_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        let derive_inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == collection.source_descriptor().handle()
                        && claim.target() == collection.descriptor().handle() =>
                {
                    Some(claim.mapping().0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(derive_inputs, vec![commit.data()]);
    }

    #[test]
    fn retained_rewrite_reconstructs_collected_exact_proof_without_writes() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source.pile");
        let retained_path = directory.path().join("retained.pile");
        std::fs::File::create(&source_path).unwrap();
        std::fs::File::create(&retained_path).unwrap();
        let mut source_store = Pile::open(&source_path).unwrap();
        let mut retained_store = Pile::open(&retained_path).unwrap();

        let collection = SuccinctArchiveCollection::new(id(7));
        let source_descriptor = CollectionDescriptor::to_blob(&collection.source_descriptor());
        let target_descriptor = CollectionDescriptor::to_blob(&collection.descriptor());
        source_store
            .put::<SimpleArchive, _>(source_descriptor.clone())
            .unwrap();
        source_store
            .put::<SimpleArchive, _>(target_descriptor.clone())
            .unwrap();
        let metadata = source_store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();

        let a_facts = facts([(1, 3)]);
        let b_facts = facts([(2, 4)]);
        let c_facts = facts([(3, 5)]);
        let a = (&a_facts).to_blob();
        let b = (&b_facts).to_blob();
        let c = (&c_facts).to_blob();
        for blob in [&a, &b, &c] {
            source_store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        }
        let commits = [
            pile_commit(&mut source_store, &collection, 1, &a, metadata),
            pile_commit(&mut source_store, &collection, 2, &b, metadata),
            pile_commit(&mut source_store, &collection, 3, &c, metadata),
        ];

        let ab = simplearchive_union::join(&a, &b).unwrap();
        let succinct_ab = super::super::derive_element(&ab).unwrap();
        let succinct_c = super::super::derive_element(&c).unwrap();
        let succinct_abc = super::super::join(&succinct_ab, &succinct_c).unwrap();
        source_store.put::<SimpleArchive, _>(ab.clone()).unwrap();
        for blob in [&succinct_ab, &succinct_c, &succinct_abc] {
            source_store
                .put::<SuccinctArchiveBlob, _>(blob.clone())
                .unwrap();
        }
        for record in [
            CollectionRecord::Merge(CollectionMerge::new(
                collection.source_descriptor().handle(),
                data(&a),
                data(&b),
                data(&ab),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.source_descriptor().handle(),
                collection.descriptor().handle(),
                data(&ab),
                data(&succinct_ab),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.source_descriptor().handle(),
                collection.descriptor().handle(),
                data(&c),
                data(&succinct_c),
            )),
            CollectionRecord::Merge(CollectionMerge::new(
                collection.descriptor().handle(),
                data(&succinct_ab),
                data(&succinct_c),
                data(&succinct_abc),
            )),
        ] {
            source_store.insert(record).unwrap();
        }
        source_store.flush().unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_direct(succinct_abc.get_handle());
        source_store
            .rewrite_retained_into(&mut retained_store, &roots, WantRewritePolicy::Drop)
            .unwrap();
        source_store.close().unwrap();
        retained_store.close().unwrap();

        let mut retained_store = Pile::open(&retained_path).unwrap();
        let reader = retained_store.reader().unwrap();
        for handle in [a.get_handle(), b.get_handle(), c.get_handle()] {
            assert!(reader.contains_blob(handle).unwrap());
        }
        assert!(reader
            .contains_blob(source_descriptor.get_handle())
            .unwrap());
        assert!(reader.contains_blob(metadata).unwrap());
        assert!(reader.contains_blob(succinct_abc.get_handle()).unwrap());
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_c.get_handle()).unwrap());
        assert!(!reader
            .contains_blob(target_descriptor.get_handle())
            .unwrap());
        drop(reader);

        let before = std::fs::metadata(&retained_path).unwrap().len();
        let attached = collection
            .attach_exact(&mut retained_store, &commits)
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), a_facts + b_facts + c_facts);
        assert_eq!(std::fs::metadata(&retained_path).unwrap().len(), before);
        let ensured = collection
            .ensure_exact(&mut retained_store, &commits)
            .unwrap();
        assert_eq!(ensured.segment_count(), 1);
        assert_eq!(std::fs::metadata(&retained_path).unwrap().len(), before);

        let reader = retained_store.reader().unwrap();
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_c.get_handle()).unwrap());
        assert!(!reader
            .contains_blob(target_descriptor.get_handle())
            .unwrap());
        drop(reader);
        retained_store.close().unwrap();
    }

    #[test]
    fn ensure_reconstructs_collected_source_ancestry_from_resident_source_seed() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source-seed.pile");
        let retained_path = directory.path().join("source-seed-retained.pile");
        std::fs::File::create(&source_path).unwrap();
        std::fs::File::create(&retained_path).unwrap();
        let mut source_store = Pile::open(&source_path).unwrap();
        let mut retained_store = Pile::open(&retained_path).unwrap();

        let collection = SuccinctArchiveCollection::new(id(8));
        source_store
            .put::<SimpleArchive, _>(CollectionDescriptor::to_blob(
                &collection.source_descriptor(),
            ))
            .unwrap();
        let metadata = source_store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let a = facts([(1, 3)]).to_blob();
        let b = facts([(2, 4)]).to_blob();
        let c = facts([(3, 5)]).to_blob();
        for blob in [&a, &b, &c] {
            source_store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        }
        let commits = [
            pile_commit(&mut source_store, &collection, 4, &a, metadata),
            pile_commit(&mut source_store, &collection, 5, &b, metadata),
            pile_commit(&mut source_store, &collection, 6, &c, metadata),
        ];
        let ab = simplearchive_union::join(&a, &b).unwrap();
        let abc = simplearchive_union::join(&ab, &c).unwrap();
        source_store.put::<SimpleArchive, _>(ab.clone()).unwrap();
        source_store.put::<SimpleArchive, _>(abc.clone()).unwrap();
        for record in [
            CollectionRecord::Merge(CollectionMerge::new(
                collection.source_descriptor().handle(),
                data(&a),
                data(&b),
                data(&ab),
            )),
            CollectionRecord::Merge(CollectionMerge::new(
                collection.source_descriptor().handle(),
                data(&ab),
                data(&c),
                data(&abc),
            )),
        ] {
            source_store.insert(record).unwrap();
        }
        source_store.flush().unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_direct(abc.get_handle());
        source_store
            .rewrite_retained_into(&mut retained_store, &roots, WantRewritePolicy::Drop)
            .unwrap();
        source_store.close().unwrap();
        retained_store.close().unwrap();

        let mut retained_store = Pile::open(&retained_path).unwrap();
        let reader = retained_store.reader().unwrap();
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(reader.contains_blob(abc.get_handle()).unwrap());
        drop(reader);
        let attached = collection
            .ensure_exact(&mut retained_store, &commits)
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        let derive_inputs: Vec<_> = retained_store
            .records()
            .unwrap()
            .map(Result::unwrap)
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == collection.source_descriptor().handle()
                        && claim.target() == collection.descriptor().handle() =>
                {
                    Some(claim.mapping().0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(derive_inputs, vec![data(&abc)]);
        retained_store.close().unwrap();
    }
}
