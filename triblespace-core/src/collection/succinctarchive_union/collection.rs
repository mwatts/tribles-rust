//! Exact-ticket facade for canonical raw SuccinctArchive collections.
//!
//! This remains separate from the legacy `SuccinctRollup` lifecycle. Unsigned
//! equations are reproducible cache evidence rather than authority or durable
//! receipts: attachment reconstructs collected intermediates in use-counted
//! scratch from authenticated source leaves, then freshly validates only the
//! resident artifacts selected by the physical cover. Target compaction is an
//! explicit maintenance call rather than background policy. The raw exact cover
//! remains authoritative and fixes the returned shard shape; a private second
//! stage attaches an exact persisted Rank9 fiber for each selected raw member or
//! rebuilds that accelerator transiently when optional evidence is absent,
//! invalid, or ambiguous.

use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchive, SuccinctArchiveBlob, SuccinctArchiveRank9IndexBlob,
    UnionArchive,
};
use crate::collection::exact_derived::{
    ExactDerivedAlgebra, ExactDerivedCollection, ExactDerivedCollectionError,
};
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::simplearchive_union;
use crate::collection::{CollectionCommit, CollectionDescriptor, CollectionStore};
use crate::id::Id;
use crate::metadata::MetaDescribe;
use crate::repo::{BlobStore, BlobStoreMeta};

use super::rank9_fiber::Rank9Fiber;

/// Failure to complete or attach one exact Succinct ticket.
#[derive(Debug)]
pub enum SuccinctArchiveCollectionError {
    /// Exact-ticket authority, resolution, construction, or storage failed.
    Exact(ExactDerivedCollectionError),
    /// Explicit target compaction failed.
    Compaction(ExactTargetCompactionError),
    /// Exact Rank9 fiber probing, transient rebuilding, or publication failed.
    Rank9(super::Rank9FiberError),
}

impl fmt::Display for SuccinctArchiveCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(f),
            Self::Compaction(source) => source.fmt(f),
            Self::Rank9(source) => source.fmt(f),
        }
    }
}

impl Error for SuccinctArchiveCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Compaction(source) => Some(source),
            Self::Rank9(source) => Some(source),
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

impl From<super::Rank9FiberError> for SuccinctArchiveCollectionError {
    fn from(source: super::Rank9FiberError) -> Self {
        Self::Rank9(source)
    }
}

/// Canonical raw SuccinctArchive projection of one scoped SimpleArchive union.
///
/// Signed source commits remain the only authority. Returned query sources
/// preserve the deterministic resident physical cover as Succinct shards.
/// Persisted Rank9 `DERIVE` records are optional one-to-one fibers over that
/// cover: they add no authority, retention, target merges, or shard selection.
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

    /// ABI-, format-, and builder-version-qualified lifted Rank9 collection.
    ///
    /// Its representation remains the detached Rank9 blob encoding. The recipe
    /// fixes every canonical-byte determinant for this build, making the map
    /// from one raw member to one Rank9 member functional. Algebraically the
    /// target join is `i(a) join i(b) = i(a join b)`; this facade never produces
    /// target `MERGE` records because constructing that join needs the raw
    /// dependencies named by the sidecars.
    pub fn rank9_descriptor(&self) -> CollectionDescriptor {
        CollectionDescriptor::new(
            self.scope,
            <SuccinctArchiveRank9IndexBlob as MetaDescribe>::id(),
            super::current_rank9_lifted_union_recipe(),
        )
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
        if cover.is_empty() {
            return self.empty_archive();
        }
        Ok(self.rank9_fiber().attach(store, cover)?)
    }

    /// Ensure missing raw derivations and attach the exact sharded cover.
    ///
    /// Completion writes raw outputs first, then ensures one persisted Rank9
    /// sidecar and ordinary `DERIVE` for each member of that fixed admitted raw
    /// cover. Every newly claimed endpoint precedes the first new Rank9
    /// equation, no flush occurs, and the exact expected pairs are strictly
    /// re-read through a fresh reader.
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
        if cover.is_empty() {
            return self.empty_archive();
        }
        Ok(self.rank9_fiber().ensure(store, cover)?)
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
        if cover.is_empty() {
            return self.empty_archive();
        }
        Ok(self.rank9_fiber().ensure(store, cover)?)
    }

    fn kernel(&self) -> ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob> {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }

    fn rank9_fiber(&self) -> Rank9Fiber {
        Rank9Fiber::new(self.descriptor(), self.rank9_descriptor())
    }

    fn empty_archive(
        &self,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError> {
        let raw = super::empty();
        let raw_data = crate::inline::encodings::hash::Handle::<SuccinctArchiveBlob>::to_hash(
            raw.get_handle(),
        );
        let bottom: SuccinctArchive<OrderedUniverse> =
            raw.try_from_blob()
                .map_err(|source| super::Rank9FiberError::Build {
                    raw: raw_data,
                    source,
                })?;
        Ok(UnionArchive::new(vec![bottom]))
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

    use std::any::TypeId;
    use std::collections::BTreeSet;
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::UnknownBlob;
    use crate::blob::{Blob, BlobEncoding, Bytes, IntoBlob, TryFromBlob};
    use crate::collection::{
        collection_physical_cover, discover_collection_records, resolve_collection_semantics,
        CollectionClaimValidation, CollectionData, CollectionDerive, CollectionMerge,
        CollectionRecord,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::pile::{Pile, WantRewritePolicy};
    use crate::repo::{BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut, RetentionRoots};
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    /// Compile-time proof that the native API has no PinStore requirement.
    #[derive(Default)]
    struct CollectionOnly {
        repo: MemoryRepo,
        puts: usize,
        inserts: usize,
    }

    impl CollectionOnly {
        fn reset_writes(&mut self) {
            self.puts = 0;
            self.inserts = 0;
        }

        fn writes(&self) -> (usize, usize) {
            (self.puts, self.inserts)
        }
    }

    impl BlobStorePut for CollectionOnly {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<E, T>(&mut self, item: T) -> Result<crate::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: crate::blob::IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.puts += 1;
            self.repo.put(item)
        }
    }

    impl BlobStore for CollectionOnly {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.repo.reader()
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
            self.repo.records()
        }

        fn select_records(
            &mut self,
            selectors: &BTreeSet<crate::collection::CollectionRecordSelector>,
        ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
            self.repo.select_records(selectors)
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.inserts += 1;
            self.repo.insert(record)
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct InjectedFailure(&'static str);

    impl fmt::Display for InjectedFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for InjectedFailure {}

    struct FaultStore {
        repo: MemoryRepo,
        rank9_target: crate::collection::CollectionId,
        fail_rank9_put: bool,
        drop_rank9_put: bool,
        replace_rank9_on_put: Option<CollectionData>,
        drop_rank9_claim: bool,
        fail_rank9_claim_at: Option<usize>,
        rank9_claim_attempts: usize,
        puts: usize,
        inserts: usize,
    }

    impl FaultStore {
        fn new(repo: MemoryRepo, rank9_target: crate::collection::CollectionId) -> Self {
            Self {
                repo,
                rank9_target,
                fail_rank9_put: false,
                drop_rank9_put: false,
                replace_rank9_on_put: None,
                drop_rank9_claim: false,
                fail_rank9_claim_at: None,
                rank9_claim_attempts: 0,
                puts: 0,
                inserts: 0,
            }
        }

        fn reset_writes(&mut self) {
            self.puts = 0;
            self.inserts = 0;
            self.rank9_claim_attempts = 0;
        }

        fn writes(&self) -> (usize, usize) {
            (self.puts, self.inserts)
        }

        fn is_rank9_claim(&self, record: &CollectionRecord) -> bool {
            matches!(record, CollectionRecord::Derive(claim) if claim.target() == self.rank9_target)
        }
    }

    impl BlobStorePut for FaultStore {
        type PutError = InjectedFailure;

        fn put<E, T>(&mut self, item: T) -> Result<Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.puts += 1;
            let blob = item.to_blob();
            if TypeId::of::<E>() == TypeId::of::<SuccinctArchiveRank9IndexBlob>() {
                if self.fail_rank9_put {
                    return Err(InjectedFailure("injected Rank9 put failure"));
                }
                if self.drop_rank9_put {
                    return Ok(blob.get_handle());
                }
                let output = Handle::<E>::to_hash(blob.get_handle());
                if self.replace_rank9_on_put == Some(output) {
                    let reader = self
                        .repo
                        .blobs
                        .reader()
                        .expect("memory reader is infallible");
                    let retained: Vec<Inline<Handle<UnknownBlob>>> = reader
                        .into_iter()
                        .map(|(resident, _)| resident)
                        .filter(|resident| resident.raw != output.raw)
                        .collect();
                    self.repo.blobs.keep(retained);
                }
            }
            Ok(self.repo.put(blob).expect("memory put is infallible"))
        }
    }

    impl BlobStore for FaultStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.repo.reader()
        }
    }

    impl CollectionStore for FaultStore {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = InjectedFailure;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.repo.records()
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.inserts += 1;
            if self.is_rank9_claim(&record) {
                self.rank9_claim_attempts += 1;
                if self.fail_rank9_claim_at == Some(self.rank9_claim_attempts) {
                    return Err(InjectedFailure("injected Rank9 DERIVE failure"));
                }
                if self.drop_rank9_claim {
                    return Ok(());
                }
            }
            self.repo
                .insert(record)
                .expect("memory record insert is infallible");
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum FiberWriteEvent {
        Put(TypeId, CollectionData),
        Insert(CollectionRecord),
    }

    struct GuardReader {
        inner: <MemoryRepo as BlobStore>::Reader,
        live: Arc<AtomicUsize>,
    }

    impl Clone for GuardReader {
        fn clone(&self) -> Self {
            self.live.fetch_add(1, Ordering::SeqCst);
            Self {
                inner: self.inner.clone(),
                live: Arc::clone(&self.live),
            }
        }
    }

    impl Drop for GuardReader {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl PartialEq for GuardReader {
        fn eq(&self, other: &Self) -> bool {
            self.inner == other.inner && Arc::ptr_eq(&self.live, &other.live)
        }
    }

    impl Eq for GuardReader {}

    impl BlobStoreMeta for GuardReader {
        type MetaError = <<MemoryRepo as BlobStore>::Reader as BlobStoreMeta>::MetaError;

        fn metadata<E>(
            &self,
            handle: Inline<Handle<E>>,
        ) -> Result<Option<crate::repo::BlobMetadata>, Self::MetaError>
        where
            E: BlobEncoding + 'static,
            Handle<E>: InlineEncoding,
        {
            self.inner.metadata(handle)
        }
    }

    impl BlobStoreGet for GuardReader {
        type GetError<E: std::error::Error + Send + Sync + 'static> =
            <<MemoryRepo as BlobStore>::Reader as BlobStoreGet>::GetError<E>;

        fn get<T, E>(
            &self,
            handle: Inline<Handle<E>>,
        ) -> Result<T, Self::GetError<<T as TryFromBlob<E>>::Error>>
        where
            E: BlobEncoding + 'static,
            T: TryFromBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.inner.get(handle)
        }
    }

    impl BlobStoreList for GuardReader {
        type Iter<'a>
            = <<MemoryRepo as BlobStore>::Reader as BlobStoreList>::Iter<'a>
        where
            Self: 'a;
        type Err = <<MemoryRepo as BlobStore>::Reader as BlobStoreList>::Err;

        fn blobs<'a>(&'a self) -> Self::Iter<'a> {
            self.inner.blobs()
        }

        fn contains_blob<E>(&self, handle: Inline<Handle<E>>) -> Result<bool, Self::Err>
        where
            E: BlobEncoding + 'static,
            Handle<E>: InlineEncoding,
        {
            self.inner.contains_blob(handle)
        }
    }

    struct GuardStore {
        repo: MemoryRepo,
        live: Arc<AtomicUsize>,
        events: Vec<FiberWriteEvent>,
    }

    impl GuardStore {
        fn assert_no_reader(&self) {
            assert_eq!(
                self.live.load(Ordering::SeqCst),
                0,
                "Rank9 publication wrote while an old reader was live",
            );
        }
    }

    impl BlobStorePut for GuardStore {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<E, T>(&mut self, item: T) -> Result<Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.assert_no_reader();
            let blob = item.to_blob();
            self.events.push(FiberWriteEvent::Put(
                TypeId::of::<E>(),
                Handle::<E>::to_hash(blob.get_handle()),
            ));
            self.repo.put(blob)
        }
    }

    impl BlobStore for GuardStore {
        type Reader = GuardReader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            let inner = self.repo.reader()?;
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(GuardReader {
                inner,
                live: Arc::clone(&self.live),
            })
        }
    }

    impl CollectionStore for GuardStore {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = <MemoryRepo as CollectionStore>::InsertError;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.repo.records()
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.assert_no_reader();
            self.events.push(FiberWriteEvent::Insert(record));
            self.repo.insert(record)
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

    fn raw_derives<S: CollectionStore>(
        store: &mut S,
        collection: &SuccinctArchiveCollection,
    ) -> Vec<CollectionDerive> {
        let mut claims: Vec<_> = store
            .records()
            .unwrap()
            .map(Result::unwrap)
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == collection.source_descriptor().handle()
                        && claim.target() == collection.descriptor().handle() =>
                {
                    Some(claim)
                }
                _ => None,
            })
            .collect();
        claims.sort_by_key(CollectionDerive::mapping);
        claims
    }

    fn rank9_derives<S: CollectionStore>(
        store: &mut S,
        collection: &SuccinctArchiveCollection,
    ) -> Vec<CollectionDerive> {
        let mut claims: Vec<_> = store
            .records()
            .unwrap()
            .map(Result::unwrap)
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == collection.descriptor().handle()
                        && claim.target() == collection.rank9_descriptor().handle() =>
                {
                    Some(claim)
                }
                _ => None,
            })
            .collect();
        claims.sort_by_key(CollectionDerive::mapping);
        claims
    }

    fn remove_blob<E>(store: &mut CollectionOnly, handle: Inline<Handle<E>>)
    where
        E: BlobEncoding + 'static,
        Handle<E>: InlineEncoding,
    {
        let reader = store.repo.blobs.reader().unwrap();
        let retained: Vec<Inline<Handle<UnknownBlob>>> = reader
            .into_iter()
            .map(|(resident, _)| resident)
            .filter(|resident| resident.raw != handle.raw)
            .collect();
        store.repo.blobs.keep(retained);
    }

    fn one_raw_fixture(
        scope_byte: u8,
    ) -> (
        SuccinctArchiveCollection,
        CollectionOnly,
        CollectionCommit,
        TribleSet,
        Blob<SuccinctArchiveBlob>,
    ) {
        let scope = id(scope_byte);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let commit = signed_commit(&mut store, scope, 1, &source);
        publish(&mut store, commit);
        let cover = collection
            .kernel()
            .ensure_exact(&mut store, &[commit], &collection)
            .unwrap();
        assert_eq!(cover.len(), 1);
        let raw = super::super::derive_element(&source).unwrap();
        assert_eq!(cover.members()[0].0, data(&raw));
        drop(cover);
        store.reset_writes();
        (collection, store, commit, expected, raw)
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
    fn rank9_descriptor_is_abi_profile_separated_and_wrong_recipe_is_inert() {
        let collection = SuccinctArchiveCollection::new(id(7));
        let current = collection.rank9_descriptor();
        let recipes = [
            super::super::RANK9_LIFTED_UNION_RECIPE_V1_32_LE,
            super::super::RANK9_LIFTED_UNION_RECIPE_V1_32_BE,
            super::super::RANK9_LIFTED_UNION_RECIPE_V1_64_LE,
            super::super::RANK9_LIFTED_UNION_RECIPE_V1_64_BE,
        ];
        assert_eq!(
            current.representation(),
            <SuccinctArchiveRank9IndexBlob as MetaDescribe>::id(),
        );
        assert_eq!(
            current.recipe(),
            super::super::current_rank9_lifted_union_recipe(),
        );
        #[cfg(all(target_pointer_width = "32", target_endian = "little"))]
        assert_eq!(current.recipe(), recipes[0]);
        #[cfg(all(target_pointer_width = "32", target_endian = "big"))]
        assert_eq!(current.recipe(), recipes[1]);
        #[cfg(all(target_pointer_width = "64", target_endian = "little"))]
        assert_eq!(current.recipe(), recipes[2]);
        #[cfg(all(target_pointer_width = "64", target_endian = "big"))]
        assert_eq!(current.recipe(), recipes[3]);
        assert_ne!(current.recipe(), collection.descriptor().recipe());
        assert_eq!(recipes.into_iter().collect::<BTreeSet<_>>().len(), 4);
        let descriptors: BTreeSet<_> = recipes
            .into_iter()
            .map(|recipe| {
                CollectionDescriptor::new(
                    collection.scope(),
                    <SuccinctArchiveRank9IndexBlob as MetaDescribe>::id(),
                    recipe,
                )
                .handle()
            })
            .collect();
        assert_eq!(descriptors.len(), 4);

        let (collection, mut store, commit, expected, raw) = one_raw_fixture(8);
        let sidecar = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone()).unwrap();
        store
            .put::<SuccinctArchiveRank9IndexBlob, _>(sidecar.clone())
            .unwrap();
        let wrong_recipe = recipes
            .into_iter()
            .find(|recipe| *recipe != current.recipe())
            .unwrap();
        let wrong_target = CollectionDescriptor::new(
            collection.scope(),
            <SuccinctArchiveRank9IndexBlob as MetaDescribe>::id(),
            wrong_recipe,
        );
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.descriptor().handle(),
                wrong_target.handle(),
                data(&raw),
                data(&sidecar),
            )))
            .unwrap();
        store.reset_writes();

        let attached = collection.attach_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0));
        assert!(rank9_derives(&mut store, &collection).is_empty());
    }

    #[test]
    fn real_lifted_rank9_derives_close_the_commuting_square() {
        let scope = id(9);
        let collection = SuccinctArchiveCollection::new(scope);
        let a = facts([(1, 3)]).to_blob();
        let b = facts([(2, 4)]).to_blob();
        let raw_a = super::super::derive_element(&a).unwrap();
        let raw_b = super::super::derive_element(&b).unwrap();
        let raw_ab = super::super::join(&raw_a, &raw_b).unwrap();
        let rank_a = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw_a.clone()).unwrap();
        let rank_b = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw_b.clone()).unwrap();
        let rank_ab =
            SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw_ab.clone()).unwrap();
        for (raw, rank9) in [
            (raw_a.clone(), rank_a.clone()),
            (raw_b.clone(), rank_b.clone()),
            (raw_ab.clone(), rank_ab.clone()),
        ] {
            SuccinctArchive::<OrderedUniverse>::from_blob_pair(raw, rank9).unwrap();
        }

        let metadata = TribleSet::new().to_blob().get_handle();
        let first = CollectionCommit::sign(
            &SigningKey::from_bytes(&[1; 32]),
            collection.source_descriptor().handle(),
            data(&a),
            metadata,
        );
        let second = CollectionCommit::sign(
            &SigningKey::from_bytes(&[2; 32]),
            collection.source_descriptor().handle(),
            data(&b),
            metadata,
        );
        let mut store = MemoryRepo::default();
        for record in [
            CollectionRecord::Commit(first),
            CollectionRecord::Commit(second),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.source_descriptor().handle(),
                collection.descriptor().handle(),
                data(&a),
                data(&raw_a),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.source_descriptor().handle(),
                collection.descriptor().handle(),
                data(&b),
                data(&raw_b),
            )),
            CollectionRecord::Merge(CollectionMerge::new(
                collection.descriptor().handle(),
                data(&raw_a),
                data(&raw_b),
                data(&raw_ab),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.descriptor().handle(),
                collection.rank9_descriptor().handle(),
                data(&raw_a),
                data(&rank_a),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.descriptor().handle(),
                collection.rank9_descriptor().handle(),
                data(&raw_b),
                data(&rank_b),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.descriptor().handle(),
                collection.rank9_descriptor().handle(),
                data(&raw_ab),
                data(&rank_ab),
            )),
        ] {
            store.insert(record).unwrap();
        }
        assert!(!store.records().unwrap().map(Result::unwrap).any(|record| {
            matches!(record, CollectionRecord::Merge(claim)
                if claim.collection() == collection.rank9_descriptor().handle())
        }));
        let discovered = discover_collection_records(&mut store).unwrap();
        let authorized = BTreeSet::from([first.id(), second.id()]);
        let resolution =
            resolve_collection_semantics::<(), Infallible, _>(&discovered, &authorized, |_| {
                Ok(CollectionClaimValidation::Accepted)
            })
            .unwrap();
        let semantics = resolution.semantics();
        let rank9_collection = collection.rank9_descriptor().handle();
        let expected_frontier = BTreeSet::from([data(&rank_ab)]);
        assert_eq!(
            semantics.frontier(rank9_collection),
            Some(&expected_frontier)
        );
        assert_eq!(
            semantics.supporting_commit_ids(rank9_collection, data(&rank_ab)),
            authorized,
        );
        let resident = BTreeSet::from([data(&rank_a), data(&rank_b), data(&rank_ab)]);
        let physical = collection_physical_cover(semantics, rank9_collection, &resident);
        assert_eq!(physical.cover, expected_frontier);
        assert!(physical.missing.is_empty());
    }

    #[test]
    fn ensured_fibers_are_one_to_one_deterministic_exact_and_zero_write_when_complete() {
        let scope = id(10);
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

        let ensured = collection
            .ensure_exact(&mut store, &[second, first])
            .unwrap();
        assert_eq!(ensured.segment_count(), 2);
        assert_eq!(attached_facts(&ensured), left_facts + right_facts);
        let raw_claims = raw_derives(&mut store, &collection);
        let rank9_claims = rank9_derives(&mut store, &collection);
        let raw_outputs: BTreeSet<_> = raw_claims.iter().map(|claim| claim.mapping().1).collect();
        let rank9_inputs: BTreeSet<_> =
            rank9_claims.iter().map(|claim| claim.mapping().0).collect();
        assert_eq!(raw_outputs, rank9_inputs);
        assert_eq!(rank9_claims.len(), 2);
        let reader = store.reader().unwrap();
        for claim in &rank9_claims {
            let (raw_data, rank9_data) = claim.mapping();
            let raw: Blob<SuccinctArchiveBlob> = reader
                .get(Handle::<SuccinctArchiveBlob>::from_hash(raw_data))
                .unwrap();
            let rank9: Blob<SuccinctArchiveRank9IndexBlob> = reader
                .get(Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(
                    rank9_data,
                ))
                .unwrap();
            assert_eq!(
                SuccinctArchiveRank9IndexBlob::source_handle(&rank9).unwrap(),
                raw.get_handle(),
            );
            SuccinctArchive::<OrderedUniverse>::from_blob_pair(raw, rank9).unwrap();
        }
        drop(reader);

        store.reset_writes();
        let attached = collection
            .attach_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(attached.segment_count(), 2);
        assert_eq!(store.writes(), (0, 0));
        store.reset_writes();
        let repeated = collection
            .ensure_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(repeated.segment_count(), 2);
        assert_eq!(store.writes(), (0, 0));
        assert_eq!(rank9_derives(&mut store, &collection), rank9_claims);
    }

    #[test]
    fn absent_corrupt_and_source_mismatched_rank9_evidence_falls_back_without_writes() {
        let (collection, mut store, commit, expected, raw) = one_raw_fixture(11);
        let attached = collection.attach_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0));

        let malformed = Blob::<SuccinctArchiveRank9IndexBlob>::new(Bytes::from(b"bad".to_vec()));
        store
            .put::<SuccinctArchiveRank9IndexBlob, _>(malformed.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.descriptor().handle(),
                collection.rank9_descriptor().handle(),
                data(&raw),
                data(&malformed),
            )))
            .unwrap();
        store.reset_writes();
        let attached = collection.attach_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0));

        let (other_collection, mut other_store, other_commit, other_expected, other_raw) =
            one_raw_fixture(12);
        let foreign_source = facts([(9, 7)]).to_blob();
        let foreign_raw = super::super::derive_element(&foreign_source).unwrap();
        assert_ne!(data(&other_raw), data(&foreign_raw));
        let foreign = SuccinctArchive::<OrderedUniverse>::build_rank9_index(foreign_raw).unwrap();
        other_store
            .put::<SuccinctArchiveRank9IndexBlob, _>(foreign.clone())
            .unwrap();
        other_store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                other_collection.descriptor().handle(),
                other_collection.rank9_descriptor().handle(),
                data(&other_raw),
                data(&foreign),
            )))
            .unwrap();
        other_store.reset_writes();
        let attached = other_collection
            .attach_exact(&mut other_store, &[other_commit])
            .unwrap();
        assert_eq!(attached_facts(&attached), other_expected);
        assert_eq!(other_store.writes(), (0, 0));
    }

    #[test]
    fn ambiguous_evidence_is_attach_inert_but_canonical_ensure_is_zero_write() {
        let (collection, mut store, commit, expected, raw) = one_raw_fixture(13);
        let canonical = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone()).unwrap();
        let bogus = Blob::<SuccinctArchiveRank9IndexBlob>::new(Bytes::from(b"bogus".to_vec()));
        for sidecar in [&canonical, &bogus] {
            store
                .put::<SuccinctArchiveRank9IndexBlob, _>(sidecar.clone())
                .unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    collection.descriptor().handle(),
                    collection.rank9_descriptor().handle(),
                    data(&raw),
                    data(sidecar),
                )))
                .unwrap();
        }
        store.reset_writes();

        let attached = collection.attach_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0));
        store.reset_writes();
        let ensured = collection.ensure_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&ensured), expected);
        assert_eq!(store.writes(), (0, 0));
        assert_eq!(rank9_derives(&mut store, &collection).len(), 2);
    }

    #[test]
    fn missing_canonical_endpoint_is_repaired_without_duplicate_derive() {
        let (collection, mut store, commit, expected, raw) = one_raw_fixture(14);
        let canonical = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone()).unwrap();
        store
            .put::<SuccinctArchiveRank9IndexBlob, _>(canonical.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.descriptor().handle(),
                collection.rank9_descriptor().handle(),
                data(&raw),
                data(&canonical),
            )))
            .unwrap();
        remove_blob(&mut store, canonical.get_handle());
        store.reset_writes();

        let attached = collection.attach_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0));
        let repaired = collection.ensure_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&repaired), expected);
        assert_eq!(store.writes(), (3, 0));
        assert_eq!(rank9_derives(&mut store, &collection).len(), 1);
        let reader = store.reader().unwrap();
        assert!(reader.contains_blob(canonical.get_handle()).unwrap());
    }

    #[test]
    fn corrupt_canonical_endpoint_is_repaired_without_duplicate_derive() {
        let (collection, mut base, commit, expected, raw) = one_raw_fixture(21);
        let canonical = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone()).unwrap();
        base.put::<SuccinctArchiveRank9IndexBlob, _>(canonical.clone())
            .unwrap();
        base.insert(CollectionRecord::Derive(CollectionDerive::new(
            collection.descriptor().handle(),
            collection.rank9_descriptor().handle(),
            data(&raw),
            data(&canonical),
        )))
        .unwrap();
        remove_blob(&mut base, canonical.get_handle());
        base.repo.blobs.insert(Blob::with_handle(
            Bytes::from(b"corrupt canonical sidecar".to_vec()),
            canonical.get_handle(),
        ));
        let mut store = FaultStore::new(base.repo, collection.rank9_descriptor().handle());
        store.replace_rank9_on_put = Some(data(&canonical));

        let attached = collection.attach_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0));
        let repaired = collection.ensure_exact(&mut store, &[commit]).unwrap();
        assert_eq!(attached_facts(&repaired), expected);
        assert_eq!(store.writes(), (3, 0));
        assert_eq!(rank9_derives(&mut store, &collection).len(), 1);
        let reader = store.reader().unwrap();
        let resident: Blob<SuccinctArchiveRank9IndexBlob> =
            reader.get(canonical.get_handle()).unwrap();
        assert_eq!(data(&resident), data(&canonical));
        SuccinctArchive::<OrderedUniverse>::from_blob_pair(raw, resident).unwrap();
    }

    #[test]
    fn endpoint_failure_precedes_and_prevents_rank9_derive_publication() {
        let (collection, base, commit, _, _) = one_raw_fixture(15);
        let mut store = FaultStore::new(base.repo, collection.rank9_descriptor().handle());
        store.fail_rank9_put = true;
        assert!(matches!(
            collection.ensure_exact(&mut store, &[commit]),
            Err(SuccinctArchiveCollectionError::Rank9(
                super::super::Rank9FiberError::Storage { .. }
            ))
        ));
        assert!(rank9_derives(&mut store, &collection).is_empty());
    }

    #[test]
    fn fresh_verification_rejects_dropped_sidecar_and_dropped_claim() {
        let (collection, base, commit, _, _) = one_raw_fixture(16);
        let mut dropped_sidecar =
            FaultStore::new(base.repo, collection.rank9_descriptor().handle());
        dropped_sidecar.drop_rank9_put = true;
        assert!(matches!(
            collection.ensure_exact(&mut dropped_sidecar, &[commit]),
            Err(SuccinctArchiveCollectionError::Rank9(
                super::super::Rank9FiberError::IncompletePublication { .. }
            ))
        ));
        assert_eq!(rank9_derives(&mut dropped_sidecar, &collection).len(), 1);

        let (collection, base, commit, _, raw) = one_raw_fixture(17);
        let canonical = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw).unwrap();
        let mut dropped_claim = FaultStore::new(base.repo, collection.rank9_descriptor().handle());
        dropped_claim.drop_rank9_claim = true;
        assert!(matches!(
            collection.ensure_exact(&mut dropped_claim, &[commit]),
            Err(SuccinctArchiveCollectionError::Rank9(
                super::super::Rank9FiberError::IncompletePublication { .. }
            ))
        ));
        assert!(rank9_derives(&mut dropped_claim, &collection).is_empty());
        let reader = dropped_claim.reader().unwrap();
        assert!(reader.contains_blob(canonical.get_handle()).unwrap());
    }

    #[test]
    fn partial_claim_publication_retries_to_exactly_one_claim_per_member() {
        let scope = id(18);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut base = CollectionOnly::default();
        let left = put_data(&mut base, &facts([(1, 3)]));
        let right = put_data(&mut base, &facts([(2, 4)]));
        let first = signed_commit(&mut base, scope, 1, &left);
        let second = signed_commit(&mut base, scope, 2, &right);
        publish(&mut base, first);
        publish(&mut base, second);
        let cover = collection
            .kernel()
            .ensure_exact(&mut base, &[first, second], &collection)
            .unwrap();
        assert_eq!(cover.len(), 2);
        drop(cover);
        let mut store = FaultStore::new(base.repo, collection.rank9_descriptor().handle());
        store.fail_rank9_claim_at = Some(2);

        assert!(matches!(
            collection.ensure_exact(&mut store, &[first, second]),
            Err(SuccinctArchiveCollectionError::Rank9(
                super::super::Rank9FiberError::Storage { .. }
            ))
        ));
        assert_eq!(rank9_derives(&mut store, &collection).len(), 1);
        store.fail_rank9_claim_at = None;
        store.reset_writes();
        let retried = collection
            .ensure_exact(&mut store, &[first, second])
            .unwrap();
        assert_eq!(retried.segment_count(), 2);
        assert_eq!(rank9_derives(&mut store, &collection).len(), 2);
        assert_eq!(store.inserts, 1);
        store.reset_writes();
        collection
            .ensure_exact(&mut store, &[second, first])
            .unwrap();
        assert_eq!(store.writes(), (0, 0));
    }

    #[test]
    fn rank9_publication_drops_readers_and_orders_all_endpoints_before_claims() {
        let scope = id(19);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut base = CollectionOnly::default();
        let left = put_data(&mut base, &facts([(1, 3)]));
        let right = put_data(&mut base, &facts([(2, 4)]));
        let first = signed_commit(&mut base, scope, 1, &left);
        let second = signed_commit(&mut base, scope, 2, &right);
        publish(&mut base, first);
        publish(&mut base, second);
        let cover = collection
            .kernel()
            .ensure_exact(&mut base, &[first, second], &collection)
            .unwrap();
        assert_eq!(cover.len(), 2);
        drop(cover);
        let live = Arc::new(AtomicUsize::new(0));
        let mut store = GuardStore {
            repo: base.repo,
            live: Arc::clone(&live),
            events: Vec::new(),
        };

        let attached = collection
            .ensure_exact(&mut store, &[second, first])
            .unwrap();
        assert_eq!(attached.segment_count(), 2);
        assert_eq!(live.load(Ordering::SeqCst), 0);
        let first_claim = store
            .events
            .iter()
            .position(|event| {
                matches!(event, FiberWriteEvent::Insert(CollectionRecord::Derive(claim))
                    if claim.target() == collection.rank9_descriptor().handle())
            })
            .expect("fiber publication emits a Rank9 DERIVE");
        let sidecar_puts: Vec<_> = store
            .events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                FiberWriteEvent::Put(encoding, data)
                    if *encoding == TypeId::of::<SuccinctArchiveRank9IndexBlob>() =>
                {
                    Some((index, *data))
                }
                _ => None,
            })
            .collect();
        assert_eq!(sidecar_puts.len(), 2);
        assert!(sidecar_puts.iter().all(|(index, _)| *index < first_claim));
        let raw_descriptor = Handle::<SimpleArchive>::to_hash(collection.descriptor().handle());
        let rank9_descriptor =
            Handle::<SimpleArchive>::to_hash(collection.rank9_descriptor().handle());
        for descriptor in [raw_descriptor, rank9_descriptor] {
            assert!(store.events[..first_claim].iter().any(
                |event| matches!(event, FiberWriteEvent::Put(_, data) if *data == descriptor)
            ));
        }
        assert!(!store.events[first_claim..]
            .iter()
            .any(|event| matches!(event, FiberWriteEvent::Put(_, _))));
        let claim_inputs: Vec<_> = store.events[first_claim..]
            .iter()
            .filter_map(|event| match event {
                FiberWriteEvent::Insert(CollectionRecord::Derive(claim))
                    if claim.target() == collection.rank9_descriptor().handle() =>
                {
                    Some(claim.mapping().0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(claim_inputs.len(), 2);
        assert!(claim_inputs.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn compaction_builds_only_the_selected_raw_cover_fiber_and_no_rank9_merge() {
        let scope = id(20);
        let collection = SuccinctArchiveCollection::new(scope);
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &facts([(1, 3)]));
        let right = put_data(&mut store, &facts([(2, 4)]));
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let cover = collection
            .kernel()
            .ensure_exact(&mut store, &[first, second], &collection)
            .unwrap();
        assert_eq!(cover.len(), 2);
        drop(cover);
        assert!(rank9_derives(&mut store, &collection).is_empty());

        let compacted = collection
            .compact_exact(&mut store, &[second, first])
            .unwrap();
        assert_eq!(compacted.segment_count(), 1);
        let rank9_claims = rank9_derives(&mut store, &collection);
        assert_eq!(rank9_claims.len(), 1);
        let merged_raw = records(&mut store)
            .into_iter()
            .find_map(|record| match record {
                CollectionRecord::Merge(claim)
                    if claim.collection() == collection.descriptor().handle() =>
                {
                    Some(claim.result())
                }
                _ => None,
            })
            .expect("compaction publishes one selected raw merge");
        assert_eq!(rank9_claims[0].mapping().0, merged_raw);
        assert!(!records(&mut store).into_iter().any(|record| {
            matches!(record, CollectionRecord::Merge(claim)
                if claim.collection() == collection.rank9_descriptor().handle())
        }));
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
    fn retained_rewrite_drops_and_exact_ensure_repairs_unowned_rank9_fiber() {
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
        let ensured = collection
            .ensure_exact(&mut source_store, &commits)
            .unwrap();
        assert_eq!(ensured.segment_count(), 1);
        let rank9_claims = rank9_derives(&mut source_store, &collection);
        assert_eq!(rank9_claims.len(), 1);
        assert_eq!(rank9_claims[0].mapping().0, data(&succinct_abc));
        let rank9_handle =
            Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(rank9_claims[0].mapping().1);
        let rank9_descriptor = CollectionDescriptor::to_blob(&collection.rank9_descriptor());
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
        assert!(!reader.contains_blob(rank9_descriptor.get_handle()).unwrap());
        assert!(!reader.contains_blob(rank9_handle).unwrap());
        drop(reader);
        assert_eq!(
            rank9_derives(&mut retained_store, &collection),
            rank9_claims
        );

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
        let repaired = std::fs::metadata(&retained_path).unwrap().len();
        assert!(repaired > before);
        assert_eq!(
            rank9_derives(&mut retained_store, &collection),
            rank9_claims
        );
        let repeated = collection
            .ensure_exact(&mut retained_store, &commits)
            .unwrap();
        assert_eq!(repeated.segment_count(), 1);
        assert_eq!(std::fs::metadata(&retained_path).unwrap().len(), repaired);

        let reader = retained_store.reader().unwrap();
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_c.get_handle()).unwrap());
        assert!(reader
            .contains_blob(target_descriptor.get_handle())
            .unwrap());
        assert!(reader.contains_blob(rank9_descriptor.get_handle()).unwrap());
        assert!(reader.contains_blob(rank9_handle).unwrap());
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
