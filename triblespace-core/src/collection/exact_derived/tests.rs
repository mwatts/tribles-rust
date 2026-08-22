use crate::collection::reach;
use super::*;

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ed25519_dalek::SigningKey;

use crate::blob::encodings::UnknownBlob;
use crate::blob::{IntoBlob, TryFromBlob};
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::descriptor::{self, identity_for_tests};
use crate::collection::records::CollectionName;
use crate::collection::simplearchive_union;
use crate::inline::encodings::hash::Handle;
use crate::metadata::MetaDescribe;
use crate::repo::memoryrepo::MemoryRepo;
use crate::repo::{BlobStoreList, BlobStorePut};
use crate::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

/// The one team every collection in these tests belongs to.
fn test_team() -> ed25519_dalek::VerifyingKey {
    SigningKey::from_bytes(&[1; 32]).verifying_key()
}

fn test_name(name: &str) -> CollectionName {
    CollectionName::new(name).unwrap()
}

fn source_root() -> Fragment {
    simplearchive_union::descriptor(&test_name("source"), test_team(), reach::private())
}

fn kernel() -> ExactDerivedCollection<SimpleArchive, UnknownBlob> {
    ExactDerivedCollection::new(
        source_root(),
        descriptor::naming(
            &test_name("target"),
            test_team(),
            <UnknownBlob as MetaDescribe>::id(),
            simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        ),
    )
}

fn row(entity: u8, value: u8) -> Trible {
    let mut raw = [value; TRIBLE_LEN];
    raw[..16].fill(entity);
    raw[16..32].fill(9);
    Trible::force_raw(raw).unwrap()
}

fn archive(rows: impl IntoIterator<Item = (u8, u8)>) -> Blob<SimpleArchive> {
    let mut set = TribleSet::new();
    for (entity, value) in rows {
        set.insert(&row(entity, value));
    }
    set.to_blob()
}

fn data<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
where
    Handle<E>: InlineEncoding,
{
    Handle::<E>::to_hash(blob.get_handle())
}

struct TestAlgebra;

impl ExactDerivedAlgebra<SimpleArchive, UnknownBlob> for TestAlgebra {
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &Blob<SimpleArchive>,
    ) -> Result<(), ExactAlgebraError> {
        if descriptor != kernel().source_descriptor() {
            return Err(ExactAlgebraError::Fatal(
                "wrong test source descriptor".to_owned(),
            ));
        }
        simplearchive_union::validate_element(source)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn validate_target(
        &self,
        descriptor: &Fragment,
        target: &Blob<UnknownBlob>,
    ) -> Result<(), ExactAlgebraError> {
        if descriptor != kernel().target_descriptor() {
            return Err(ExactAlgebraError::Fatal(
                "wrong test target descriptor".to_owned(),
            ));
        }
        let Some(source) = target.bytes.as_ref().strip_suffix(&[0xA5]) else {
            return Err(ExactAlgebraError::Fatal(
                "test target lacks its canonical suffix".to_owned(),
            ));
        };
        simplearchive_union::validate_element(&Blob::new(source.to_vec().into()))
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn join_source(
        &self,
        low: &Blob<SimpleArchive>,
        high: &Blob<SimpleArchive>,
    ) -> Result<Blob<SimpleArchive>, ExactAlgebraError> {
        simplearchive_union::join(low, high)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn derive(&self, source: &Blob<SimpleArchive>) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        Ok(derive(source).unwrap())
    }

    fn join_target(
        &self,
        low: &Blob<UnknownBlob>,
        high: &Blob<UnknownBlob>,
    ) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        let descriptor = kernel().target_descriptor().clone();
        self.validate_target(&descriptor, low)?;
        self.validate_target(&descriptor, high)?;
        let low =
            Blob::<SimpleArchive>::new(low.bytes.as_ref()[..low.bytes.len() - 1].to_vec().into());
        let high =
            Blob::<SimpleArchive>::new(high.bytes.as_ref()[..high.bytes.len() - 1].to_vec().into());
        let joined = simplearchive_union::join(&low, &high)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))?;
        Ok(derive(&joined).unwrap())
    }
}

#[derive(Clone, Default)]
struct SelectiveAlgebra {
    capacity_derives: BTreeSet<CollectionData>,
    fatal_derives: BTreeSet<CollectionData>,
    capacity_target_pairs: BTreeSet<(CollectionData, CollectionData)>,
    fatal_target_pairs: BTreeSet<(CollectionData, CollectionData)>,
    derive_attempts: Arc<Mutex<Vec<CollectionData>>>,
    target_attempts: Arc<Mutex<Vec<(CollectionData, CollectionData)>>>,
}

impl SelectiveAlgebra {
    fn pair<T: BlobEncoding>(low: &Blob<T>, high: &Blob<T>) -> (CollectionData, CollectionData)
    where
        Handle<T>: InlineEncoding,
    {
        let mut pair = [data(low), data(high)];
        pair.sort_unstable();
        (pair[0], pair[1])
    }
}

impl ExactDerivedAlgebra<SimpleArchive, UnknownBlob> for SelectiveAlgebra {
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &Blob<SimpleArchive>,
    ) -> Result<(), ExactAlgebraError> {
        TestAlgebra.validate_source(descriptor, source)
    }

    fn validate_target(
        &self,
        descriptor: &Fragment,
        target: &Blob<UnknownBlob>,
    ) -> Result<(), ExactAlgebraError> {
        TestAlgebra.validate_target(descriptor, target)
    }

    fn join_source(
        &self,
        low: &Blob<SimpleArchive>,
        high: &Blob<SimpleArchive>,
    ) -> Result<Blob<SimpleArchive>, ExactAlgebraError> {
        TestAlgebra.join_source(low, high)
    }

    fn derive(&self, source: &Blob<SimpleArchive>) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        let input = data(source);
        self.derive_attempts.lock().unwrap().push(input);
        if self.fatal_derives.contains(&input) {
            return Err(ExactAlgebraError::Fatal("injected fatal derive".to_owned()));
        }
        if self.capacity_derives.contains(&input) {
            return Err(ExactAlgebraError::Capacity(
                "injected derive capacity".to_owned(),
            ));
        }
        TestAlgebra.derive(source)
    }

    fn join_target(
        &self,
        low: &Blob<UnknownBlob>,
        high: &Blob<UnknownBlob>,
    ) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        let pair = Self::pair(low, high);
        self.target_attempts.lock().unwrap().push(pair);
        if self.fatal_target_pairs.contains(&pair) {
            return Err(ExactAlgebraError::Fatal(
                "injected fatal target join".to_owned(),
            ));
        }
        if self.capacity_target_pairs.contains(&pair) {
            return Err(ExactAlgebraError::Capacity(
                "injected target capacity".to_owned(),
            ));
        }
        TestAlgebra.join_target(low, high)
    }
}

fn derive(source: &Blob<SimpleArchive>) -> Result<Blob<UnknownBlob>, Infallible> {
    let mut bytes = source.bytes.as_ref().to_vec();
    bytes.push(0xA5);
    Ok(Blob::new(bytes.into()))
}

fn source_commit(store: &mut MemoryRepo, key: u8, blob: &Blob<SimpleArchive>) -> CollectionCommit {
    store.put::<SimpleArchive, _>(blob.clone()).unwrap();
    let metadata = store
        .put::<SimpleArchive, _>(TribleSet::new().to_blob())
        .unwrap();
    let commit = CollectionCommit::sign(
        &SigningKey::from_bytes(&[key; 32]),
        kernel().source_collection(),
        data(blob),
        metadata,
    );
    store.insert(CollectionRecord::Commit(commit)).unwrap();
    commit
}

fn publish_derive(store: &mut MemoryRepo, input: &Blob<SimpleArchive>) -> Blob<UnknownBlob> {
    let output = derive(input).unwrap();
    store.put::<UnknownBlob, _>(output.clone()).unwrap();
    store
        .insert(CollectionRecord::Derive(CollectionDerive::new(
            kernel().target_collection(),
            data(input),
            data(&output),
        )))
        .unwrap();
    output
}

fn publish_source_merge(
    store: &mut MemoryRepo,
    low: &Blob<SimpleArchive>,
    high: &Blob<SimpleArchive>,
) -> Blob<SimpleArchive> {
    let result = simplearchive_union::join(low, high).unwrap();
    store.put::<SimpleArchive, _>(result.clone()).unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection(),
            data(low),
            data(high),
            data(&result),
        )))
        .unwrap();
    result
}

fn derived_inputs(store: &mut MemoryRepo) -> Vec<CollectionData> {
    store
        .records()
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|record| match record {
            CollectionRecord::Derive(claim)
                if claim.target() == kernel().target_collection() =>
            {
                Some(claim.mapping().0)
            }
            _ => None,
        })
        .collect()
}

struct PanicStore;

impl BlobStorePut for PanicStore {
    type PutError = Infallible;

    fn put<S, T>(&mut self, _: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        panic!("empty ticket attempted a blob write")
    }
}

impl BlobStore for PanicStore {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = Infallible;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        panic!("empty ticket opened a reader")
    }
}

impl CollectionStore for PanicStore {
    type RecordsError = Infallible;
    type InsertError = Infallible;
    type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        panic!("empty ticket scanned records")
    }

    fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
        panic!("empty ticket inserted a record")
    }
}

#[test]
fn zero_ticket_performs_no_store_operation() {
    let mut store = PanicStore;
    assert!(kernel()
        .attach_exact(&mut store, &[], &TestAlgebra)
        .unwrap()
        .is_empty());
    assert!(kernel()
        .ensure_exact(&mut store, &[], &TestAlgebra)
        .unwrap()
        .is_empty());
    assert!(
        compact_exact_target(&kernel(), &mut store, &[], &TestAlgebra)
            .unwrap()
            .is_empty()
    );
}

#[derive(Default)]
struct CountingStore {
    inner: MemoryRepo,
    puts: usize,
    inserts: usize,
}

impl BlobStorePut for CountingStore {
    type PutError = <MemoryRepo as BlobStorePut>::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.puts += 1;
        self.inner.put(item)
    }
}

impl BlobStore for CountingStore {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl CollectionStore for CountingStore {
    type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;
    type RecordIter<'a>
        = <MemoryRepo as CollectionStore>::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.inserts += 1;
        self.inner.insert(record)
    }
}

#[test]
fn complete_probe_ensure_performs_zero_writes() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    publish_derive(&mut inner, &source);
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let cover = kernel()
        .ensure_exact(&mut store, &[commit], &TestAlgebra)
        .unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(store.puts, 0);
    assert_eq!(store.inserts, 0);
}

#[test]
fn stable_exact_cover_compaction_performs_zero_additional_writes() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    publish_derive(&mut inner, &source);
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let cover = compact_exact_target(&kernel(), &mut store, &[commit], &TestAlgebra).unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(store.puts, 0);
    assert_eq!(store.inserts, 0);
}

#[test]
fn capacity_source_upper_replans_to_lower_resident_cover() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let mut inner = MemoryRepo::default();
    let ticket = [
        source_commit(&mut inner, 1, &a),
        source_commit(&mut inner, 2, &b),
    ];
    let upper = publish_source_merge(&mut inner, &a, &b);
    let algebra = SelectiveAlgebra {
        capacity_derives: BTreeSet::from([data(&upper)]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let cover = kernel()
        .ensure_exact(&mut store, &ticket, &algebra)
        .unwrap();
    assert_eq!(cover.len(), 2);
    let mut actual = derived_inputs(&mut store.inner);
    actual.sort_unstable();
    let mut expected = vec![data(&a), data(&b)];
    expected.sort_unstable();
    assert_eq!(actual, expected);
    assert!(algebra
        .derive_attempts
        .lock()
        .unwrap()
        .contains(&data(&upper)));
}

#[test]
fn capacity_source_replan_is_global_across_overlapping_uppers() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let c = archive([(3, 5)]);
    let mut inner = MemoryRepo::default();
    let ticket = [
        source_commit(&mut inner, 1, &a),
        source_commit(&mut inner, 2, &b),
        source_commit(&mut inner, 3, &c),
    ];
    let u = publish_source_merge(&mut inner, &a, &b);
    let v = publish_source_merge(&mut inner, &b, &c);
    let (successful_upper, blocked_upper, final_leaf) = if data(&u) < data(&v) {
        (&u, &v, &c)
    } else {
        (&v, &u, &a)
    };
    let algebra = SelectiveAlgebra {
        capacity_derives: BTreeSet::from([data(blocked_upper)]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    kernel()
        .ensure_exact(&mut store, &ticket, &algebra)
        .unwrap();
    let mut actual = derived_inputs(&mut store.inner);
    actual.sort_unstable();
    let mut expected = vec![data(successful_upper), data(final_leaf)];
    expected.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!(
        algebra
            .derive_attempts
            .lock()
            .unwrap()
            .iter()
            .filter(|input| **input == data(successful_upper))
            .count(),
        2,
        "one planning attempt is reused after replanning, then fresh admission recomputes it",
    );
}

#[test]
fn terminal_source_capacity_is_repeatable_and_zero_write() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    let algebra = SelectiveAlgebra {
        capacity_derives: BTreeSet::from([data(&source)]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    for _ in 0..2 {
        assert!(matches!(
            kernel().ensure_exact(&mut store, &[commit], &algebra),
            Err(ExactDerivedCollectionError::UnrepresentableCover { ref blocked, ref missing })
                if blocked.len() == 1 && missing.len() == 1
        ));
        assert_eq!((store.puts, store.inserts), (0, 0));
    }
}

#[test]
fn mixed_terminal_capacity_publishes_no_prepared_sibling() {
    let first = archive([(1, 3)]);
    let second = archive([(2, 4)]);
    let (successful, blocked) = if data(&first) < data(&second) {
        (&first, &second)
    } else {
        (&second, &first)
    };
    let mut inner = MemoryRepo::default();
    let ticket = [
        source_commit(&mut inner, 1, &first),
        source_commit(&mut inner, 2, &second),
    ];
    let algebra = SelectiveAlgebra {
        capacity_derives: BTreeSet::from([data(blocked)]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    assert!(matches!(
        kernel().ensure_exact(&mut store, &ticket, &algebra),
        Err(ExactDerivedCollectionError::UnrepresentableCover { .. })
    ));
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(derived_inputs(&mut store.inner).is_empty());
    assert!(algebra
        .derive_attempts
        .lock()
        .unwrap()
        .contains(&data(successful)));
}

#[test]
fn fatal_source_construction_is_not_capacity_fallback() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    let algebra = SelectiveAlgebra {
        fatal_derives: BTreeSet::from([data(&source)]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    assert!(matches!(
        kernel().ensure_exact(&mut store, &[commit], &algebra),
        Err(ExactDerivedCollectionError::Derive { input, .. }) if input == data(&source)
    ));
    assert_eq!((store.puts, store.inserts), (0, 0));
}

#[derive(Debug)]
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

    fn metadata<S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<Option<crate::repo::BlobMetadata>, Self::MetaError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.inner.metadata(handle)
    }
}

impl BlobStoreGet for GuardReader {
    type GetError<E: Error + Send + Sync + 'static> =
        <<MemoryRepo as BlobStore>::Reader as BlobStoreGet>::GetError<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
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

    fn contains_blob<S>(&self, handle: Inline<Handle<S>>) -> Result<bool, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.inner.contains_blob(handle)
    }
}

#[derive(Clone, Copy, Debug)]
enum WriteEvent {
    Put(CollectionData),
    Insert(CollectionRecord),
}

struct GuardStore {
    inner: MemoryRepo,
    live: Arc<AtomicUsize>,
    events: Vec<WriteEvent>,
}

impl GuardStore {
    fn assert_no_reader(&self) {
        assert_eq!(
            self.live.load(Ordering::SeqCst),
            0,
            "write while reader is live"
        );
    }
}

impl BlobStorePut for GuardStore {
    type PutError = <MemoryRepo as BlobStorePut>::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.assert_no_reader();
        let blob = item.to_blob();
        self.events
            .push(WriteEvent::Put(Handle::<S>::to_hash(blob.get_handle())));
        self.inner.put(blob)
    }
}

impl BlobStore for GuardStore {
    type Reader = GuardReader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        let inner = self.inner.reader()?;
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
        self.inner.records()
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.assert_no_reader();
        self.events.push(WriteEvent::Insert(record));
        self.inner.insert(record)
    }
}

#[test]
fn reader_is_dropped_before_first_write() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    let live = Arc::new(AtomicUsize::new(0));
    let mut store = GuardStore {
        inner,
        live: Arc::clone(&live),
        events: Vec::new(),
    };
    kernel()
        .ensure_exact(&mut store, &[commit], &TestAlgebra)
        .unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

fn target_merge_records(store: &mut MemoryRepo) -> Vec<CollectionMerge> {
    store
        .records()
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|record| match record {
            CollectionRecord::Merge(claim)
                if claim.collection() == kernel().target_collection() =>
            {
                Some(claim)
            }
            _ => None,
        })
        .collect()
}

fn joined_cover(cover: &ExactCover<UnknownBlob>) -> Blob<UnknownBlob> {
    let mut members = cover.members().iter();
    let mut joined = members
        .next()
        .expect("nonempty ticket has a target member")
        .1
        .clone();
    for (_, member) in members {
        joined = TestAlgebra.join_target(&joined, member).unwrap();
    }
    joined
}

fn cover_ids(cover: &ExactCover<UnknownBlob>) -> Vec<CollectionData> {
    cover.members().iter().map(|(data, _)| *data).collect()
}

#[test]
fn compaction_collapses_same_tier_and_returns_an_exact_tier_stable_cover() {
    let sources = [
        archive([(1, 3)]),
        archive([(2, 4)]),
        archive((10..18).map(|entity| (entity, entity + 20))),
    ];
    let mut store = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut store, index as u8 + 1, source))
        .collect();

    let cover = compact_exact_target(&kernel(), &mut store, &ticket, &TestAlgebra).unwrap();
    let mut tiers = BTreeSet::new();
    for (_, blob) in cover.members() {
        assert!(tiers.insert(blob.bytes.len().max(1).ilog2()));
    }
    assert_eq!(cover.len(), 2);
    assert!(!target_merge_records(&mut store).is_empty());

    let expected_source = sources
        .iter()
        .skip(1)
        .try_fold(sources[0].clone(), |joined, source| {
            simplearchive_union::join(&joined, source)
        })
        .unwrap();
    assert_eq!(
        joined_cover(&cover).bytes,
        derive(&expected_source).unwrap().bytes
    );
}

#[test]
fn target_capacity_retires_only_low() {
    let sources = [archive([(1, 3)]), archive([(2, 4)]), archive([(3, 5)])];
    let mut inner = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let mut targets: Vec<_> = sources
        .iter()
        .map(|source| derive(source).unwrap())
        .collect();
    targets.sort_unstable_by_key(data);
    let first_pair = SelectiveAlgebra::pair(&targets[0], &targets[1]);
    let second_pair = SelectiveAlgebra::pair(&targets[1], &targets[2]);
    let algebra = SelectiveAlgebra {
        capacity_target_pairs: BTreeSet::from([first_pair]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let cover = compact_exact_target(&kernel(), &mut store, &ticket, &algebra).unwrap();
    assert_eq!(cover.len(), 2);
    assert_eq!(
        algebra.target_attempts.lock().unwrap().as_slice(),
        &[first_pair, second_pair, second_pair],
        "the carry attempts two pairs, then fresh admission recomputes the published equation",
    );
    let merges = target_merge_records(&mut store.inner);
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].inputs(), second_pair);
}

#[test]
fn fatal_late_target_join_publishes_no_staged_prefix() {
    let sources = [
        archive([(1, 3)]),
        archive([(2, 4)]),
        archive([(3, 5)]),
        archive([(4, 6)]),
    ];
    let mut inner = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let mut targets: Vec<_> = sources
        .iter()
        .map(|source| derive(source).unwrap())
        .collect();
    targets.sort_unstable_by_key(data);
    let first_pair = SelectiveAlgebra::pair(&targets[0], &targets[1]);
    let fatal_pair = SelectiveAlgebra::pair(&targets[2], &targets[3]);
    let algebra = SelectiveAlgebra {
        fatal_target_pairs: BTreeSet::from([fatal_pair]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    assert!(matches!(
        compact_exact_target(&kernel(), &mut store, &ticket, &algebra),
        Err(ExactTargetCompactionError::Merge { low, high, .. })
            if (low, high) == fatal_pair
    ));
    assert_eq!(
        algebra.target_attempts.lock().unwrap().as_slice(),
        &[first_pair, fatal_pair],
    );
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

#[test]
fn capacity_stable_target_collision_is_repeatable_and_zero_write() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let mut inner = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let targets: Vec<_> = sources
        .iter()
        .map(|source| derive(source).unwrap())
        .collect();
    let capacity_pair = SelectiveAlgebra::pair(&targets[0], &targets[1]);
    let algebra = SelectiveAlgebra {
        capacity_target_pairs: BTreeSet::from([capacity_pair]),
        ..SelectiveAlgebra::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let first = compact_exact_target(&kernel(), &mut store, &ticket, &algebra).unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!((store.puts, store.inserts), (0, 0));
    let second = compact_exact_target(&kernel(), &mut store, &ticket, &algebra).unwrap();
    assert_eq!(cover_ids(&first), cover_ids(&second));
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

#[test]
fn compaction_substitutes_new_resident_uppers_through_an_old_nonresident_proof() {
    let sources = [archive([(1, 3)]), archive([(2, 4)]), archive([(3, 5)])];
    let mut store = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut store, index as u8 + 1, source);
            publish_derive(&mut store, source);
            commit
        })
        .collect();
    let mut targets: Vec<_> = sources
        .iter()
        .map(|source| {
            let target = derive(source).unwrap();
            (data(&target), target)
        })
        .collect();
    targets.sort_unstable_by_key(|(data, _)| *data);
    let old_intermediate = TestAlgebra
        .join_target(&targets[1].1, &targets[2].1)
        .unwrap();
    let old_upper = TestAlgebra
        .join_target(&targets[0].1, &old_intermediate)
        .unwrap();
    for claim in [
        CollectionMerge::new(
            kernel().target_collection(),
            targets[1].0,
            targets[2].0,
            data(&old_intermediate),
        ),
        CollectionMerge::new(
            kernel().target_collection(),
            targets[0].0,
            data(&old_intermediate),
            data(&old_upper),
        ),
    ] {
        store.insert(CollectionRecord::Merge(claim)).unwrap();
    }
    let wrong = derive(&archive([(9, 9)])).unwrap();
    store
        .put::<UnknownBlob, _>(Blob::<UnknownBlob>::with_handle(
            wrong.bytes,
            old_upper.get_handle(),
        ))
        .unwrap();

    let before = kernel()
        .attach_exact(&mut store, &ticket, &TestAlgebra)
        .unwrap();
    assert_eq!(before.len(), 3);
    let after = compact_exact_target(&kernel(), &mut store, &ticket, &TestAlgebra).unwrap();
    let mut tiers = BTreeSet::new();
    assert!(after
        .members()
        .iter()
        .all(|(_, blob)| tiers.insert(blob.bytes.len().max(1).ilog2())));
    assert_eq!(after.len(), 2);
    assert_eq!(joined_cover(&after).bytes, old_upper.bytes);
    assert!(target_merge_records(&mut store)
        .iter()
        .any(|claim| claim.inputs() == (targets[0].0, targets[1].0)));
}

#[test]
fn compaction_is_ticket_order_deterministic_and_repeatedly_idempotent() {
    let sources = [
        archive([(1, 3)]),
        archive([(2, 4)]),
        archive([(3, 5)]),
        archive([(4, 6)]),
    ];
    let mut first_store = MemoryRepo::default();
    let first_ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut first_store, index as u8 + 1, source))
        .collect();
    let mut second_store = MemoryRepo::default();
    let mut second_ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut second_store, index as u8 + 1, source))
        .collect();
    second_ticket.reverse();

    let first =
        compact_exact_target(&kernel(), &mut first_store, &first_ticket, &TestAlgebra).unwrap();
    let second =
        compact_exact_target(&kernel(), &mut second_store, &second_ticket, &TestAlgebra).unwrap();
    assert_eq!(cover_ids(&first), cover_ids(&second));
    assert_eq!(
        target_merge_records(&mut first_store),
        target_merge_records(&mut second_store)
    );

    let records_before: Vec<_> = first_store.records().unwrap().map(Result::unwrap).collect();
    let repeated =
        compact_exact_target(&kernel(), &mut first_store, &first_ticket, &TestAlgebra).unwrap();
    let records_after: Vec<_> = first_store.records().unwrap().map(Result::unwrap).collect();
    assert_eq!(cover_ids(&first), cover_ids(&repeated));
    assert_eq!(records_before, records_after);
}

#[test]
fn compaction_drops_readers_and_puts_all_results_before_the_first_merge() {
    let sources = [
        archive([(1, 3)]),
        archive([(2, 4)]),
        archive([(3, 5)]),
        archive([(4, 6)]),
    ];
    let mut inner = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let live = Arc::new(AtomicUsize::new(0));
    let mut store = GuardStore {
        inner,
        live: Arc::clone(&live),
        events: Vec::new(),
    };

    compact_exact_target(&kernel(), &mut store, &ticket, &TestAlgebra).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let first_merge = store
        .events
        .iter()
        .position(|event| matches!(event, WriteEvent::Insert(CollectionRecord::Merge(_))))
        .expect("colliding cover publishes a MERGE");
    let descriptor_data = Handle::<SimpleArchive>::to_hash(kernel().target_collection());
    assert!(store.events[..first_merge]
        .iter()
        .any(|event| matches!(event, WriteEvent::Put(data) if *data == descriptor_data)));
    let results: Vec<_> = store
        .events
        .iter()
        .filter_map(|event| match event {
            WriteEvent::Insert(CollectionRecord::Merge(claim)) => Some(claim.result()),
            _ => None,
        })
        .collect();
    assert!(!results.is_empty());
    for result in results {
        assert!(store.events[..first_merge]
            .iter()
            .any(|event| matches!(event, WriteEvent::Put(data) if *data == result)));
    }
}

struct FailingJoinAlgebra;

impl ExactDerivedAlgebra<SimpleArchive, UnknownBlob> for FailingJoinAlgebra {
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &Blob<SimpleArchive>,
    ) -> Result<(), ExactAlgebraError> {
        TestAlgebra.validate_source(descriptor, source)
    }

    fn validate_target(
        &self,
        descriptor: &Fragment,
        target: &Blob<UnknownBlob>,
    ) -> Result<(), ExactAlgebraError> {
        TestAlgebra.validate_target(descriptor, target)
    }

    fn join_source(
        &self,
        low: &Blob<SimpleArchive>,
        high: &Blob<SimpleArchive>,
    ) -> Result<Blob<SimpleArchive>, ExactAlgebraError> {
        TestAlgebra.join_source(low, high)
    }

    fn derive(&self, source: &Blob<SimpleArchive>) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        TestAlgebra.derive(source)
    }

    fn join_target(
        &self,
        _: &Blob<UnknownBlob>,
        _: &Blob<UnknownBlob>,
    ) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        Err(ExactAlgebraError::Fatal(
            "injected target join failure".to_owned(),
        ))
    }
}

#[derive(Debug)]
struct RejectedPut;

impl fmt::Display for RejectedPut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected put failure")
    }
}

impl Error for RejectedPut {}

struct RejectPutStore {
    inner: MemoryRepo,
    puts: usize,
}

impl BlobStorePut for RejectPutStore {
    type PutError = RejectedPut;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.puts += 1;
        if self.puts == 2 {
            Err(RejectedPut)
        } else {
            Ok(self
                .inner
                .put(item)
                .expect("MemoryRepo puts are infallible"))
        }
    }
}

struct DropMergeStore {
    inner: MemoryRepo,
}

impl BlobStorePut for DropMergeStore {
    type PutError = <MemoryRepo as BlobStorePut>::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.inner.put(item)
    }
}

impl BlobStore for DropMergeStore {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl CollectionStore for DropMergeStore {
    type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;
    type RecordIter<'a>
        = <MemoryRepo as CollectionStore>::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        if matches!(record, CollectionRecord::Merge(_)) {
            Ok(())
        } else {
            self.inner.insert(record)
        }
    }
}

impl BlobStore for RejectPutStore {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl CollectionStore for RejectPutStore {
    type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;
    type RecordIter<'a>
        = <MemoryRepo as CollectionStore>::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.inner.insert(record)
    }
}

#[test]
fn join_and_put_failures_publish_no_target_merge() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];

    let mut join_store = MemoryRepo::default();
    let join_ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut join_store, index as u8 + 1, source))
        .collect();
    assert!(matches!(
        compact_exact_target(
            &kernel(),
            &mut join_store,
            &join_ticket,
            &FailingJoinAlgebra,
        ),
        Err(ExactTargetCompactionError::Merge { .. })
    ));
    assert!(target_merge_records(&mut join_store).is_empty());

    let mut inner = MemoryRepo::default();
    let put_ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let mut put_store = RejectPutStore { inner, puts: 0 };
    assert!(matches!(
        compact_exact_target(&kernel(), &mut put_store, &put_ticket, &TestAlgebra),
        Err(ExactTargetCompactionError::Storage { .. })
    ));
    assert!(target_merge_records(&mut put_store.inner).is_empty());
}

#[test]
fn discarded_merge_insert_stalls_instead_of_looping() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let mut inner = MemoryRepo::default();
    let ticket: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let mut store = DropMergeStore { inner };

    assert!(matches!(
        compact_exact_target(&kernel(), &mut store, &ticket, &TestAlgebra),
        Err(ExactTargetCompactionError::Stalled { cover }) if cover.len() == 2
    ));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

struct LossyStore {
    inner: MemoryRepo,
    puts: usize,
    discard_put: usize,
}

impl BlobStorePut for LossyStore {
    type PutError = <MemoryRepo as BlobStorePut>::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.puts += 1;
        let blob = item.to_blob();
        if self.puts == self.discard_put {
            Ok(blob.get_handle())
        } else {
            self.inner.put(blob)
        }
    }
}

impl BlobStore for LossyStore {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl CollectionStore for LossyStore {
    type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;
    type RecordIter<'a>
        = <MemoryRepo as CollectionStore>::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.inner.insert(record)
    }
}

#[test]
fn fresh_reprobe_rejects_a_lossy_output_put() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    // Completion writes source descriptor, target descriptor, then output.
    let mut store = LossyStore {
        inner,
        puts: 0,
        discard_put: 3,
    };
    match kernel().ensure_exact(&mut store, &[commit], &TestAlgebra) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected fresh-reprobe error: {error:?}"),
        Ok(_) => panic!("lossy output was incorrectly admitted"),
    }
}

#[test]
fn missing_derive_output_is_pending_and_ensure_rebuilds() {
    let source = archive([(1, 3)]);
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &source);
    let missing = derive(&source).unwrap();
    store
        .insert(CollectionRecord::Derive(CollectionDerive::new(
            kernel().target_collection(),
            data(&source),
            data(&missing),
        )))
        .unwrap();
    match kernel().attach_exact(&mut store, &[commit], &TestAlgebra) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected missing-output error: {error:?}"),
        Ok(_) => panic!("missing output was incorrectly admitted"),
    }
    assert_eq!(
        kernel()
            .ensure_exact(&mut store, &[commit], &TestAlgebra)
            .unwrap()
            .len(),
        1,
    );
}

#[test]
fn corrupt_unsigned_endpoint_is_rejected_as_optional_evidence() {
    let source = archive([(1, 3)]);
    let expected = derive(&source).unwrap();
    let wrong = archive([(9, 9)]);
    let forged = Blob::<UnknownBlob>::with_handle(wrong.bytes.clone(), expected.get_handle());
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &source);
    store.put::<UnknownBlob, _>(forged).unwrap();
    store
        .insert(CollectionRecord::Derive(CollectionDerive::new(
            kernel().target_collection(),
            data(&source),
            data(&expected),
        )))
        .unwrap();

    // Scratch reconstruction proves the canonical DERIVE equation without
    // trusting these bytes. Physical-cover admission then freshly hashes the
    // forged resident artifact, removes it from the optional candidate set,
    // and reports the target incomplete rather than granting cache corruption
    // authority or failing unrelated evidence globally.
    match kernel().attach_exact(&mut store, &[commit], &TestAlgebra) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected corrupt-cache error: {error:?}"),
        Ok(_) => panic!("corrupt unsigned output was incorrectly admitted"),
    }
}

#[test]
fn ungrounded_source_superset_cannot_escape_the_ticket() {
    let a = archive([(1, 3)]);
    let c = archive([(3, 5)]);
    let ac = simplearchive_union::join(&a, &c).unwrap();
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &a);
    store.put::<SimpleArchive, _>(c.clone()).unwrap();
    store.put::<SimpleArchive, _>(ac.clone()).unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection(),
            data(&a),
            data(&c),
            data(&ac),
        )))
        .unwrap();

    let cover = kernel()
        .ensure_exact(&mut store, &[commit], &TestAlgebra)
        .unwrap();
    assert_eq!(cover.members()[0].1.bytes, derive(&a).unwrap().bytes);
    let derives: Vec<_> = store
        .records()
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|record| match record {
            CollectionRecord::Derive(claim) => Some(claim.mapping().0),
            _ => None,
        })
        .collect();
    assert_eq!(derives, vec![data(&a)]);
}

#[test]
fn algebra_rejects_a_lying_source_descriptor() {
    let source = archive([(1, 3)]);
    let lying_source = descriptor::naming(
        &test_name("source"),
        test_team(),
        <UnknownBlob as MetaDescribe>::id(),
        id(99),
        reach::private(),
    );
    let lifecycle = ExactDerivedCollection::<SimpleArchive, UnknownBlob>::new(
        lying_source.clone(),
        kernel().target_descriptor().clone(),
    );
    let mut store = MemoryRepo::default();
    store.put::<SimpleArchive, _>(source.clone()).unwrap();
    let metadata = store
        .put::<SimpleArchive, _>(TribleSet::new().to_blob())
        .unwrap();
    let commit = CollectionCommit::sign(
        &SigningKey::from_bytes(&[7; 32]),
        identity_for_tests(&lying_source),
        data(&source),
        metadata,
    );
    store.insert(CollectionRecord::Commit(commit)).unwrap();

    assert!(matches!(
        lifecycle.attach_exact(&mut store, &[commit], &TestAlgebra),
        Err(ExactDerivedCollectionError::RejectedCommit { commit: found, .. })
            if found == commit.id()
    ));
}

#[test]
#[should_panic(expected = "distinct source and target descriptors")]
fn identity_descriptor_pair_is_rejected() {
    let descriptor = source_root();
    let _ =
        ExactDerivedCollection::<SimpleArchive, SimpleArchive>::new(descriptor.clone(), descriptor);
}
