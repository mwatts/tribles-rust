use super::*;

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ed25519_dalek::SigningKey;

use crate::blob::encodings::UnknownBlob;
use crate::blob::{IntoBlob, TryFromBlob};
use crate::collection::simplearchive_union;
use crate::inline::encodings::hash::Handle;
use crate::metadata::MetaDescribe;
use crate::repo::memoryrepo::MemoryRepo;
use crate::repo::{BlobStoreList, BlobStorePut};
use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

fn kernel() -> ExactDerivedCollection<SimpleArchive, UnknownBlob> {
    ExactDerivedCollection::new(
        simplearchive_union::descriptor(id(1)),
        CollectionDescriptor::new(
            id(2),
            <UnknownBlob as MetaDescribe>::id(),
            simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
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

fn validate(claim: DerivedClaim<'_, SimpleArchive, UnknownBlob>) -> Result<(), String> {
    let source = simplearchive_union::descriptor(id(1));
    let target = kernel().target_descriptor();
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
            claim: _,
            low: _,
            high: _,
            result: _,
        } => Err("test target has no merge equations".to_owned()),
        DerivedClaim::Derive {
            claim,
            input,
            output,
        } => {
            if claim.source() != source.handle() || claim.target() != target.handle() {
                return Err("wrong derived collection endpoints".to_owned());
            }
            if claim.mapping() != (fresh_data_identity(input), fresh_data_identity(output)) {
                return Err("derive endpoint identity mismatch".to_owned());
            }
            simplearchive_union::validate_element(input).map_err(|error| error.to_string())?;
            if derive(input).unwrap().bytes != output.bytes {
                return Err("test derivation output is not canonical".to_owned());
            }
            Ok(())
        }
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
        kernel().source_descriptor().handle(),
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
            kernel().source_descriptor().handle(),
            kernel().target_descriptor().handle(),
            data(input),
            data(&output),
        )))
        .unwrap();
    output
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
        .attach_exact(&mut store, &[], validate)
        .unwrap()
        .is_empty());
    assert!(kernel()
        .ensure_exact(&mut store, &[], validate, derive)
        .unwrap()
        .is_empty());
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
        .ensure_exact(&mut store, &[commit], validate, derive)
        .unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(store.puts, 0);
    assert_eq!(store.inserts, 0);
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

struct GuardStore {
    inner: MemoryRepo,
    live: Arc<AtomicUsize>,
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
        self.inner.put(item)
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
    };
    kernel()
        .ensure_exact(&mut store, &[commit], validate, derive)
        .unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
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
    match kernel().ensure_exact(&mut store, &[commit], validate, derive) {
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
            kernel().source_descriptor().handle(),
            kernel().target_descriptor().handle(),
            data(&source),
            data(&missing),
        )))
        .unwrap();
    match kernel().attach_exact(&mut store, &[commit], validate) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected missing-output error: {error:?}"),
        Ok(_) => panic!("missing output was incorrectly admitted"),
    }
    assert_eq!(
        kernel()
            .ensure_exact(&mut store, &[commit], validate, derive)
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
            kernel().source_descriptor().handle(),
            kernel().target_descriptor().handle(),
            data(&source),
            data(&expected),
        )))
        .unwrap();

    // Candidate reads deliberately do not enforce the cached handle. The
    // concrete validator rejects this unsigned evidence, leaving the target
    // incomplete instead of turning optional cache corruption into a global
    // resolution failure.
    match kernel().attach_exact(&mut store, &[commit], validate) {
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
            kernel().source_descriptor().handle(),
            data(&a),
            data(&c),
            data(&ac),
        )))
        .unwrap();

    let cover = kernel()
        .ensure_exact(&mut store, &[commit], validate, derive)
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
#[should_panic(expected = "distinct source and target descriptors")]
fn identity_descriptor_pair_is_rejected() {
    let descriptor = simplearchive_union::descriptor(id(1));
    let _ = ExactDerivedCollection::<SimpleArchive, SimpleArchive>::new(descriptor, descriptor);
}
