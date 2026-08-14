//! Exact-ticket facade for canonical raw SuccinctArchive collections.
//!
//! This is a resident-evidence foundation, not a replacement for the legacy
//! `SuccinctRollup` lifecycle. Existing unsigned equations are reusable only
//! while all blobs needed to validate them remain resident; current collection
//! retention does not preserve the whole unsigned proof graph through garbage
//! collection. Missing evidence becomes pending rather than authoritative, and
//! completion can fall back to signed leaves, so this is cache-incomplete but
//! correctness-safe. The facade also has no background target compactor,
//! rebuilds Rank9 indexes in process memory on every attachment, and inherits
//! the raw format's explicit `u32::MAX` row/domain limit for each derived shard.

use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchive, SuccinctArchiveBlob, SuccinctArchiveError, UnionArchive,
};
use crate::collection::exact_derived::{
    DerivedClaim, ExactCover, ExactDerivedCollection, ExactDerivedCollectionError,
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
    /// A selected canonical raw shard could not become a query runtime.
    Attach(SuccinctArchiveError),
}

impl fmt::Display for SuccinctArchiveCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(f),
            Self::Attach(source) => write!(f, "attach raw SuccinctArchive cover: {source}"),
        }
    }
}

impl Error for SuccinctArchiveCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Attach(source) => Some(source),
        }
    }
}

impl From<ExactDerivedCollectionError> for SuccinctArchiveCollectionError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Exact(source)
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
        let cover = self
            .kernel()
            .attach_exact(store, ticket, |claim| self.validate_claim(claim))?;
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
        let cover = self.kernel().ensure_exact(
            store,
            ticket,
            |claim| self.validate_claim(claim),
            super::derive_element,
        )?;
        self.attach_cover(cover)
    }

    fn kernel(&self) -> ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob> {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }

    fn validate_claim(
        &self,
        claim: DerivedClaim<'_, SimpleArchive, SuccinctArchiveBlob>,
    ) -> Result<(), String> {
        let source = self.source_descriptor();
        let target = self.descriptor();
        match claim {
            DerivedClaim::Commit { claim, data } => {
                simplearchive_union::validate_commit(&source, claim, data)
                    .map_err(|error| error.to_string())
            }
            DerivedClaim::SourceMerge {
                claim,
                low,
                high,
                result,
            } => simplearchive_union::validate_merge(&source, claim, low, high, result)
                .map_err(|error| error.to_string()),
            DerivedClaim::TargetMerge {
                claim,
                low,
                high,
                result,
            } => super::validate_merge(&target, claim, low, high, result)
                .map_err(|error| error.to_string()),
            DerivedClaim::Derive {
                claim,
                input,
                output,
            } => super::validate_derive(&source, &target, claim, input, output)
                .map_err(|error| error.to_string()),
        }
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
    use crate::repo::BlobStorePut;
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

    #[test]
    fn empty_ticket_is_one_authority_free_local_shard_and_performs_no_io() {
        let collection = SuccinctArchiveCollection::new(id(7));
        let mut store = PanicStore;
        for attached in [
            collection.attach_exact(&mut store, &[]).unwrap(),
            collection.ensure_exact(&mut store, &[]).unwrap(),
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
}
