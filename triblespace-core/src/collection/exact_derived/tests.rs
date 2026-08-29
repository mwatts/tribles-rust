use super::*;
use crate::collection::reach;

use std::cell::RefCell;
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ed25519_dalek::SigningKey;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::UnknownBlob;
use crate::blob::{BlobEncoding, IntoBlob, TryFromBlob};
use crate::collection::descriptor;
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::simplearchive_union;
use crate::collection::CollectionCommit;
use crate::id::{ExclusiveId, Id};
use crate::id_hex;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::metadata::MetaDescribe;
use crate::repo::memoryrepo::MemoryRepo;
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreList, BlobStorePut};
use crate::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

macro_rules! inert_test_offers {
    ($($store:ty),+ $(,)?) => {$(
        impl crate::repo::ArtifactOfferStore for $store {
            type OfferError = Infallible;

            fn offer_all<I>(&mut self, _: I) -> Result<(), Self::OfferError>
            where
                I: IntoIterator<Item = crate::repo::ArtifactHandle>,
            {
                Ok(())
            }

            fn offers_snapshot(
                &mut self,
            ) -> Result<crate::repo::ArtifactOfferSnapshot, Self::OfferError> {
                Ok(crate::repo::ArtifactOfferSnapshot::default())
            }
        }
    )+};
}

inert_test_offers!(
    PanicStore,
    CountingStore,
    GuardStore,
    RejectPutStore,
    DropMergeStore,
    LossyStore,
);

/// The one team every collection in these tests belongs to.
fn test_team() -> ed25519_dalek::VerifyingKey {
    SigningKey::from_bytes(&[1; 32]).verifying_key()
}

/// Test-only SimpleArchive-compatible source encoding. It is deliberately
/// distinct from the production encoding so planning tests can instrument its
/// join without changing production semantics.
///
/// Minted with `trible genid` on 2026-08-29.
const TEST_SOURCE_ENCODING_V1: Id = id_hex!("75CA73E3F88BFE4C680115DD992EC807");

struct TestSourceBlob;

impl BlobEncoding for TestSourceBlob {}

impl MetaDescribe for TestSourceBlob {
    fn describe() -> Fragment {
        let id = TEST_SOURCE_ENCODING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-test-source-v1",
            crate::metadata::description: "Test-only canonical SimpleArchive-compatible source encoding with observable joins.",
            crate::metadata::tag: crate::metadata::KIND_BLOB_ENCODING,
        }
    }
}

impl CollectionEncoding for TestSourceBlob {
    fn validate_member(
        _descriptor: &Fragment,
        member: &Blob<Self>,
    ) -> Result<(), CollectionOperationError> {
        simplearchive_union::validate_element(member.as_transmute::<SimpleArchive>())
            .map_err(|error| CollectionOperationError::Fatal(error.to_string()))
    }

    fn join_members(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
    ) -> Result<Blob<Self>, CollectionOperationError> {
        SELECTIVE_POLICY.with(|policy| {
            if let Some(policy) = policy.borrow().as_ref() {
                policy
                    .source_attempts
                    .lock()
                    .unwrap()
                    .push(SelectiveMapping::pair(low, high));
            }
        });
        simplearchive_union::join(
            low.as_transmute::<SimpleArchive>(),
            high.as_transmute::<SimpleArchive>(),
        )
        .map(Blob::transmute::<TestSourceBlob>)
        .map_err(|error| CollectionOperationError::Fatal(error.to_string()))
    }
}

fn source_descriptor(name: &str) -> Fragment {
    crate::macros::entity! {
        crate::metadata::tag: crate::collection::KIND_COLLECTION_DESCRIPTOR,
        crate::collection::collection_name: name.to_owned(),
        crate::collection::collection_authority: test_team(),
        crate::collection::collection_representation*: <TestSourceBlob as MetaDescribe>::describe(),
        crate::collection::collection_reach*: reach::private(),
    }
}

fn source_root() -> Fragment {
    source_descriptor("source")
}

#[derive(Clone, Default)]
struct SelectiveMapping {
    capacity_derives: BTreeSet<CollectionData>,
    fatal_derives: BTreeSet<CollectionData>,
    capacity_target_pairs: BTreeSet<(CollectionData, CollectionData)>,
    fatal_target_pairs: BTreeSet<(CollectionData, CollectionData)>,
    source_attempts: Arc<Mutex<Vec<(CollectionData, CollectionData)>>>,
    derive_attempts: Arc<Mutex<Vec<CollectionData>>>,
    target_attempts: Arc<Mutex<Vec<(CollectionData, CollectionData)>>>,
}

impl SelectiveMapping {
    fn pair<T: BlobEncoding>(low: &Blob<T>, high: &Blob<T>) -> (CollectionData, CollectionData)
    where
        Handle<T>: InlineEncoding,
    {
        let mut pair = [data(low), data(high)];
        pair.sort_unstable();
        (pair[0], pair[1])
    }
}

thread_local! {
    /// Per-test instrumentation for encoding-owned joins. Test worker threads
    /// may run these cases concurrently, so a process-global switch would be
    /// both racy and capable of changing another test's semantics.
    static SELECTIVE_POLICY: RefCell<Option<SelectiveMapping>> = const { RefCell::new(None) };
}

/// Test-only canonical `SimpleArchive || 0xA5` encoding.
/// Minted with `trible genid` on 2026-08-29.
const TEST_TARGET_ENCODING_V1: Id = id_hex!("39B18B6D13B2B1872F2394EF6588F1B5");

struct TestTargetBlob;

impl BlobEncoding for TestTargetBlob {}

impl MetaDescribe for TestTargetBlob {
    fn describe() -> Fragment {
        let id = TEST_TARGET_ENCODING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-test-target-v1",
            crate::metadata::description: "Test-only canonical encoding formed by appending 0xA5 to a test source archive.",
            crate::metadata::tag: crate::metadata::KIND_BLOB_ENCODING,
        }
    }
}

impl CollectionEncoding for TestTargetBlob {
    fn validate_member(
        _descriptor: &Fragment,
        member: &Blob<Self>,
    ) -> Result<(), CollectionOperationError> {
        validate_test_target(member)
    }

    fn join_members(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
    ) -> Result<Blob<Self>, CollectionOperationError> {
        let injected = SELECTIVE_POLICY.with(|policy| {
            let policy = policy.borrow();
            let Some(policy) = policy.as_ref() else {
                return None;
            };
            let pair = SelectiveMapping::pair(low, high);
            policy.target_attempts.lock().unwrap().push(pair);
            if policy.fatal_target_pairs.contains(&pair) {
                Some(CollectionOperationError::Fatal(
                    "injected fatal target join".to_owned(),
                ))
            } else if policy.capacity_target_pairs.contains(&pair) {
                Some(CollectionOperationError::Capacity(
                    "injected target capacity".to_owned(),
                ))
            } else {
                None
            }
        });
        if let Some(error) = injected {
            return Err(error);
        }
        join_test_targets(low, high)
    }
}

/// Test-only canonical `(SimpleArchive || 0xA5) || 0xB6` encoding.
/// Minted with `trible genid` on 2026-08-29.
const SECOND_TEST_TARGET_ENCODING_V1: Id = id_hex!("9318ADD9A6257CB8973AC8BE806D12EC");

struct SecondTestTargetBlob;

impl BlobEncoding for SecondTestTargetBlob {}

impl MetaDescribe for SecondTestTargetBlob {
    fn describe() -> Fragment {
        let id = SECOND_TEST_TARGET_ENCODING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-second-test-target-v1",
            crate::metadata::description: "Test-only canonical encoding formed by appending 0xB6 to an exact-derived test target.",
            crate::metadata::tag: crate::metadata::KIND_BLOB_ENCODING,
        }
    }
}

impl CollectionEncoding for SecondTestTargetBlob {
    fn validate_member(
        _descriptor: &Fragment,
        member: &Blob<Self>,
    ) -> Result<(), CollectionOperationError> {
        validate_second_test_target(member)
    }

    fn join_members(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
    ) -> Result<Blob<Self>, CollectionOperationError> {
        validate_second_test_target(low)?;
        validate_second_test_target(high)?;
        let low =
            Blob::<TestTargetBlob>::new(low.bytes.as_ref()[..low.bytes.len() - 1].to_vec().into());
        let high = Blob::<TestTargetBlob>::new(
            high.bytes.as_ref()[..high.bytes.len() - 1].to_vec().into(),
        );
        let joined = join_test_targets(&low, &high)?;
        let mut bytes = joined.bytes.as_ref().to_vec();
        bytes.push(0xB6);
        Ok(Blob::new(bytes.into()))
    }
}

fn validate_test_target(target: &Blob<TestTargetBlob>) -> Result<(), CollectionOperationError> {
    let Some(source) = target.bytes.as_ref().strip_suffix(&[0xA5]) else {
        return Err(CollectionOperationError::Fatal(
            "test target lacks its canonical suffix".to_owned(),
        ));
    };
    simplearchive_union::validate_element(&Blob::new(source.to_vec().into()))
        .map_err(|error| CollectionOperationError::Fatal(error.to_string()))
}

fn join_test_targets(
    low: &Blob<TestTargetBlob>,
    high: &Blob<TestTargetBlob>,
) -> Result<Blob<TestTargetBlob>, CollectionOperationError> {
    validate_test_target(low)?;
    validate_test_target(high)?;
    let low = Blob::<SimpleArchive>::new(low.bytes.as_ref()[..low.bytes.len() - 1].to_vec().into());
    let high =
        Blob::<SimpleArchive>::new(high.bytes.as_ref()[..high.bytes.len() - 1].to_vec().into());
    let joined = simplearchive_union::join(&low, &high)
        .map_err(|error| CollectionOperationError::Fatal(error.to_string()))?;
    let joined = joined.transmute::<TestSourceBlob>();
    Ok(derive(&joined).unwrap())
}

fn validate_second_test_target(
    target: &Blob<SecondTestTargetBlob>,
) -> Result<(), CollectionOperationError> {
    let Some(source) = target.bytes.as_ref().strip_suffix(&[0xB6]) else {
        return Err(CollectionOperationError::Fatal(
            "second test target lacks its canonical suffix".to_owned(),
        ));
    };
    validate_test_target(&Blob::new(source.to_vec().into()))
}

/// Test-only parameter-free source-to-target mapping algorithm.
/// Minted with `trible genid` on 2026-08-29.
const TEST_SUFFIX_MAPPING_V1: Id = id_hex!("70D406F7483E8A1D384354D0AFD0D717");

struct TestSuffixMappingV1;

impl MetaDescribe for TestSuffixMappingV1 {
    fn describe() -> Fragment {
        let id = TEST_SUFFIX_MAPPING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-test-suffix-mapping-v1",
            crate::metadata::description: "Test-only canonical mapping that appends 0xA5 to a test source archive.",
            crate::metadata::tag: crate::metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
        }
    }
}

fn test_suffix_mapping_fragment() -> Fragment {
    crate::macros::entity! {
        crate::metadata::tag: crate::collection::KIND_COLLECTION_MAPPING,
        crate::collection::mapping_algorithm*: <TestSuffixMappingV1 as MetaDescribe>::describe(),
    }
}

/// Test-only parameter-free target-to-second-target mapping algorithm.
/// Minted with `trible genid` on 2026-08-29.
const SECOND_TEST_SUFFIX_MAPPING_V1: Id = id_hex!("4B671CE9A7CF6F2AEC3AD5F9B2A59FBC");

/// Extrinsic replacement used to prove mapping-entity id substitution is
/// operationally inert. Minted with `trible genid` on 2026-08-29.
const SUBSTITUTED_MAPPING_ENTITY: Id = id_hex!("DE3EB767EC428155B4E3526ABFFFD991");

struct SecondTestSuffixMappingV1;

impl MetaDescribe for SecondTestSuffixMappingV1 {
    fn describe() -> Fragment {
        let id = SECOND_TEST_SUFFIX_MAPPING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-second-test-suffix-mapping-v1",
            crate::metadata::description: "Test-only canonical mapping that appends 0xB6 to an exact-derived test target.",
            crate::metadata::tag: crate::metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
        }
    }
}

fn second_test_suffix_mapping_fragment() -> Fragment {
    crate::macros::entity! {
        crate::metadata::tag: crate::collection::KIND_COLLECTION_MAPPING,
        crate::collection::mapping_algorithm*: <SecondTestSuffixMappingV1 as MetaDescribe>::describe(),
    }
}

fn target_root(source: CollectionHandle) -> Fragment {
    crate::macros::entity! { _ @
        crate::metadata::tag: crate::collection::KIND_COLLECTION_DESCRIPTOR,
        crate::collection::collection_source: source,
        crate::collection::collection_authority: test_team(),
        crate::collection::collection_representation*: <TestTargetBlob as MetaDescribe>::describe(),
        crate::collection::collection_mapping*: test_suffix_mapping_fragment(),
        crate::collection::collection_reach*: reach::private(),
    }
}

fn second_target_root(source: CollectionHandle) -> Fragment {
    crate::macros::entity! { _ @
        crate::metadata::tag: crate::collection::KIND_COLLECTION_DESCRIPTOR,
        crate::collection::collection_source: source,
        crate::collection::collection_authority: test_team(),
        crate::collection::collection_representation*: <SecondTestTargetBlob as MetaDescribe>::describe(),
        crate::collection::collection_mapping*: second_test_suffix_mapping_fragment(),
        crate::collection::collection_reach*: reach::private(),
    }
}

fn substitute_mapping_entity(target: Fragment, replacement: Id) -> Fragment {
    let descriptor_root = target.root().expect("target descriptor root");
    let mapping = descriptor::mapping(target.facts())
        .expect("valid mapping link")
        .expect("derived target has a mapping");
    let (_, facts, metafacts, blobs) = target.into_parts();
    let mut substituted = TribleSet::new();
    for fact in facts.iter() {
        let mut raw = fact.data;
        if fact.e() == &mapping {
            raw[..16].copy_from_slice(&replacement[..]);
        }
        if fact.a() == &crate::collection::collection_mapping.id()
            && raw[32..48] == [0; 16]
            && raw[48..64] == mapping[..]
        {
            raw[48..64].copy_from_slice(&replacement[..]);
        }
        substituted.insert(
            &Trible::force_raw(raw).expect("entity substitution preserves non-nil trible ids"),
        );
    }
    Fragment::rooted_from_parts(descriptor_root, substituted, metafacts, blobs)
}

fn kernel() -> ExactDerivedCollection<TestSourceBlob, TestTargetBlob, TestSuffixMapping> {
    let source = source_root();
    let source_collection = Collection::<TestSourceBlob>::from_descriptor(&source).unwrap();
    ExactDerivedCollection::new(source, target_root(source_collection.handle())).unwrap()
}

#[test]
fn mapping_entity_id_substitution_preserves_binding_semantics() {
    let source = source_root();
    let source_collection = Collection::<TestSourceBlob>::from_descriptor(&source).unwrap();
    let canonical = target_root(source_collection.handle());
    let original_mapping = descriptor::mapping(canonical.facts()).unwrap().unwrap();
    let substituted = substitute_mapping_entity(canonical, SUBSTITUTED_MAPPING_ENTITY);

    assert_ne!(original_mapping, SUBSTITUTED_MAPPING_ENTITY);
    assert_eq!(
        descriptor::mapping(substituted.facts()),
        Ok(Some(SUBSTITUTED_MAPPING_ENTITY))
    );
    assert_eq!(
        descriptor::mapping_algorithm(substituted.facts()),
        Ok(Some(TEST_SUFFIX_MAPPING_V1))
    );
    ExactDerivedCollection::<TestSourceBlob, TestTargetBlob, TestSuffixMapping>::new(
        source,
        substituted,
    )
    .expect("binding depends on mapping facts, not intrinsic-id minting history");
}

fn selective_kernel(
    mapping: SelectiveMapping,
) -> ExactDerivedCollection<TestSourceBlob, TestTargetBlob, SelectiveMapping> {
    let source = source_root();
    let source_collection = Collection::<TestSourceBlob>::from_descriptor(&source).unwrap();
    ExactDerivedCollection::with_mapping(source, target_root(source_collection.handle()), mapping)
        .unwrap()
}

fn second_kernel(
) -> ExactDerivedCollection<TestTargetBlob, SecondTestTargetBlob, SecondTestSuffixMapping> {
    let source = kernel().target_descriptor().clone();
    let source_collection = Collection::<TestTargetBlob>::from_descriptor(&source).unwrap();
    ExactDerivedCollection::new(source, second_target_root(source_collection.handle())).unwrap()
}

struct SelectivePolicyGuard(Option<SelectiveMapping>);

impl Drop for SelectivePolicyGuard {
    fn drop(&mut self) {
        SELECTIVE_POLICY.with(|policy| {
            policy.replace(self.0.take());
        });
    }
}

fn with_selective<R>(
    mapping: &SelectiveMapping,
    operation: impl FnOnce(
        &ExactDerivedCollection<TestSourceBlob, TestTargetBlob, SelectiveMapping>,
    ) -> R,
) -> R {
    let previous = SELECTIVE_POLICY.with(|policy| policy.replace(Some(mapping.clone())));
    let _guard = SelectivePolicyGuard(previous);
    let kernel = selective_kernel(mapping.clone());
    operation(&kernel)
}

fn row(entity: u8, value: u8) -> Trible {
    let mut raw = [value; TRIBLE_LEN];
    raw[..16].fill(entity);
    raw[16..32].fill(9);
    Trible::force_raw(raw).unwrap()
}

fn archive(rows: impl IntoIterator<Item = (u8, u8)>) -> Blob<TestSourceBlob> {
    let mut set = TribleSet::new();
    for (entity, value) in rows {
        set.insert(&row(entity, value));
    }
    IntoBlob::<SimpleArchive>::to_blob(set).transmute()
}

fn data<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
where
    Handle<E>: InlineEncoding,
{
    Handle::<E>::to_hash(blob.get_handle())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TestSuffixMapping;

impl CollectionMapping<TestSourceBlob, TestTargetBlob> for TestSuffixMapping {
    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, TEST_SUFFIX_MAPPING_V1, "test suffix")?;
        Ok(Self)
    }

    fn map(
        &self,
        source: &Blob<TestSourceBlob>,
    ) -> Result<Blob<TestTargetBlob>, CollectionOperationError> {
        Ok(derive(source).unwrap())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SecondTestSuffixMapping;

impl CollectionMapping<TestTargetBlob, SecondTestTargetBlob> for SecondTestSuffixMapping {
    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, SECOND_TEST_SUFFIX_MAPPING_V1, "second test suffix")?;
        Ok(Self)
    }

    fn map(
        &self,
        source: &Blob<TestTargetBlob>,
    ) -> Result<Blob<SecondTestTargetBlob>, CollectionOperationError> {
        let mut bytes = source.bytes.as_ref().to_vec();
        bytes.push(0xB6);
        Ok(Blob::new(bytes.into()))
    }
}

struct IdentityMapping;

impl CollectionMapping<TestSourceBlob, TestSourceBlob> for IdentityMapping {
    fn bind(_source: &Fragment, _target: &Fragment) -> Result<Self, CollectionOperationError> {
        Ok(Self)
    }

    fn map(
        &self,
        source: &Blob<TestSourceBlob>,
    ) -> Result<Blob<TestSourceBlob>, CollectionOperationError> {
        Ok(source.clone())
    }
}

impl CollectionMapping<TestSourceBlob, TestTargetBlob> for SelectiveMapping {
    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, TEST_SUFFIX_MAPPING_V1, "test suffix")?;
        Ok(Self::default())
    }

    fn map(
        &self,
        source: &Blob<TestSourceBlob>,
    ) -> Result<Blob<TestTargetBlob>, CollectionOperationError> {
        let input = data(source);
        self.derive_attempts.lock().unwrap().push(input);
        if self.fatal_derives.contains(&input) {
            return Err(CollectionOperationError::Fatal(
                "injected fatal derive".to_owned(),
            ));
        }
        if self.capacity_derives.contains(&input) {
            return Err(CollectionOperationError::Capacity(
                "injected derive capacity".to_owned(),
            ));
        }
        Ok(derive(source).unwrap())
    }
}

fn require_mapping(
    target: &Fragment,
    expected: Id,
    label: &str,
) -> Result<(), CollectionOperationError> {
    let actual = descriptor::mapping_algorithm(target.facts())
        .map_err(|error| CollectionOperationError::Fatal(error.to_string()))?;
    if actual == Some(expected) {
        return Ok(());
    }
    Err(CollectionOperationError::Fatal(format!(
        "{label} mapping algorithm {:?} does not match {expected:X}",
        actual.map(|id| format!("{id:X}")),
    )))
}

fn derive(source: &Blob<TestSourceBlob>) -> Result<Blob<TestTargetBlob>, Infallible> {
    let mut bytes = source.bytes.as_ref().to_vec();
    bytes.push(0xA5);
    Ok(Blob::new(bytes.into()))
}

fn source_commit(store: &mut MemoryRepo, key: u8, blob: &Blob<TestSourceBlob>) -> CollectionCommit {
    store.put::<TestSourceBlob, _>(blob.clone()).unwrap();
    let metadata = store
        .put::<SimpleArchive, _>(TribleSet::new().to_blob())
        .unwrap();
    let commit = CollectionCommit::sign(
        &SigningKey::from_bytes(&[key; 32]),
        kernel().source_collection().handle(),
        data(blob),
        metadata,
    );
    store.insert(CollectionRecord::Commit(commit)).unwrap();
    commit
}

fn source_cover(commits: &[CollectionCommit]) -> Cover<TestSourceBlob> {
    Cover::from_members(
        kernel().source_collection(),
        commits
            .iter()
            .map(CollectionCommit::data)
            .map(Handle::<TestSourceBlob>::from_hash),
    )
}

fn publish_derive(store: &mut MemoryRepo, input: &Blob<TestSourceBlob>) -> Blob<TestTargetBlob> {
    let output = derive(input).unwrap();
    store.put::<TestTargetBlob, _>(output.clone()).unwrap();
    store
        .insert(CollectionRecord::Derive(CollectionDerive::new(
            kernel().target_collection().handle(),
            data(input),
            data(&output),
        )))
        .unwrap();
    output
}

fn publish_source_merge(
    store: &mut MemoryRepo,
    low: &Blob<TestSourceBlob>,
    high: &Blob<TestSourceBlob>,
) -> Blob<TestSourceBlob> {
    let result = join_test_sources(low, high);
    store.put::<TestSourceBlob, _>(result.clone()).unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            data(low),
            data(high),
            data(&result),
        )))
        .unwrap();
    result
}

fn join_test_sources(
    low: &Blob<TestSourceBlob>,
    high: &Blob<TestSourceBlob>,
) -> Blob<TestSourceBlob> {
    simplearchive_union::join(
        low.as_transmute::<SimpleArchive>(),
        high.as_transmute::<SimpleArchive>(),
    )
    .unwrap()
    .transmute()
}

fn derived_inputs(store: &mut MemoryRepo) -> Vec<CollectionData> {
    store
        .records()
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|record| match record {
            CollectionRecord::Derive(claim)
                if claim.collection() == kernel().target_collection().handle() =>
            {
                Some(claim.input())
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
        panic!("empty cover attempted a blob write")
    }
}

impl BlobStore for PanicStore {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = Infallible;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        panic!("empty cover opened a reader")
    }
}

impl CollectionStore for PanicStore {
    type RecordsError = Infallible;
    type InsertError = Infallible;
    type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        panic!("empty cover scanned records")
    }

    fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
        panic!("empty cover inserted a record")
    }
}

#[test]
fn empty_cover_performs_no_store_operation() {
    let mut store = PanicStore;
    let cover = source_cover(&[]);
    assert!(kernel()
        .attach_exact(&mut store, &cover)
        .unwrap()
        .is_empty());
    assert!(kernel()
        .ensure_exact(&mut store, &cover)
        .unwrap()
        .is_empty());
    assert!(compact_exact_target(&kernel(), &mut store, &cover)
        .unwrap()
        .is_empty());
}

#[test]
fn empty_cover_still_belongs_to_one_exact_collection() {
    let mut store = PanicStore;
    let foreign_descriptor = source_descriptor("foreign");
    let foreign_collection =
        Collection::<TestSourceBlob>::from_descriptor(&foreign_descriptor).unwrap();
    let foreign = Cover::from_members(foreign_collection, []);
    for result in [
        kernel().attach_exact(&mut store, &foreign),
        kernel().ensure_exact(&mut store, &foreign),
    ] {
        assert!(matches!(
            result,
            Err(ExactDerivedCollectionError::InvalidCover(_))
        ));
    }
    assert!(matches!(
        kernel().probe_exact(&mut store, &foreign, &BTreeSet::new()),
        Err(ExactDerivedCollectionError::InvalidCover(_))
    ));
}

#[derive(Default)]
struct CountingStore {
    inner: MemoryRepo,
    puts: usize,
    inserts: usize,
    missing_gets: Arc<AtomicUsize>,
    metadata_failures: Arc<Mutex<BTreeSet<[u8; 32]>>>,
}

#[derive(Debug)]
struct InjectedMetadataError;

impl fmt::Display for InjectedMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("injected metadata failure")
    }
}

impl Error for InjectedMetadataError {}

#[derive(Debug)]
struct DemandGuardReader {
    inner: <MemoryRepo as BlobStore>::Reader,
    missing_gets: Arc<AtomicUsize>,
    metadata_failures: Arc<Mutex<BTreeSet<[u8; 32]>>>,
}

impl Clone for DemandGuardReader {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            missing_gets: Arc::clone(&self.missing_gets),
            metadata_failures: Arc::clone(&self.metadata_failures),
        }
    }
}

impl PartialEq for DemandGuardReader {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
            && Arc::ptr_eq(&self.missing_gets, &other.missing_gets)
            && Arc::ptr_eq(&self.metadata_failures, &other.metadata_failures)
    }
}

impl Eq for DemandGuardReader {}

impl BlobStoreMeta for DemandGuardReader {
    type MetaError = InjectedMetadataError;

    fn metadata<S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<Option<crate::repo::BlobMetadata>, Self::MetaError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        if self.metadata_failures.lock().unwrap().contains(&handle.raw) {
            return Err(InjectedMetadataError);
        }
        self.inner.metadata(handle).map_err(|never| match never {})
    }
}

impl BlobStoreList for DemandGuardReader {
    type Iter<'a>
        = <<MemoryRepo as BlobStore>::Reader as BlobStoreList>::Iter<'a>
    where
        Self: 'a;
    type Err = <<MemoryRepo as BlobStore>::Reader as BlobStoreList>::Err;

    fn blobs<'a>(&'a self) -> Self::Iter<'a> {
        self.inner.blobs()
    }
}

impl BlobStoreGet for DemandGuardReader {
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
        if matches!(self.inner.metadata(handle), Ok(None)) {
            self.missing_gets.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.get(handle)
    }
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
    type Reader = DemandGuardReader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        Ok(DemandGuardReader {
            inner: self.inner.reader()?,
            missing_gets: Arc::clone(&self.missing_gets),
            metadata_failures: Arc::clone(&self.metadata_failures),
        })
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
    let source_cover = source_cover(&[commit]);

    let cover = kernel().ensure_exact(&mut store, &source_cover).unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(store.puts, 0);
    assert_eq!(store.inserts, 0);
}

#[test]
fn ensure_publishes_the_complete_descriptor_attachment_closure() {
    let exact = kernel();
    let source = archive([(1, 3)]);
    let mut store = MemoryRepo::default();
    source_commit(&mut store, 1, &source);
    let source_cover = Cover::from_members(exact.source_collection(), [source.get_handle()]);

    exact.ensure_exact(&mut store, &source_cover).unwrap();

    let reader = store.reader().unwrap();
    for descriptor in [exact.source_descriptor(), exact.target_descriptor()] {
        let stored_descriptor: Blob<SimpleArchive> = reader
            .get(
                crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone())
                    .get_handle(),
            )
            .expect("descriptor archive is resident");
        assert_eq!(
            stored_descriptor.bytes,
            crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone(),).bytes
        );

        let mut embedded = descriptor.blobs().clone();
        let embedded_reader = embedded.reader().unwrap();
        for (handle, expected) in embedded_reader {
            let actual: Blob<UnknownBlob> = reader
                .get(handle)
                .expect("every descriptor attachment is resident");
            assert_eq!(actual.bytes, expected.bytes);
        }
    }
}

#[test]
fn compacted_source_cover_reuses_resident_decomposition_images() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    let c = publish_source_merge(&mut inner, &a, &b);
    let fa = publish_derive(&mut inner, &a);
    let fb = publish_derive(&mut inner, &b);
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target = with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
    .unwrap();

    let mut actual: Vec<_> = target.cover().members().collect();
    actual.sort_unstable();
    let mut expected = vec![fa.get_handle(), fb.get_handle()];
    expected.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(!algebra.derive_attempts.lock().unwrap().contains(&data(&c)));
}

#[test]
fn compacted_source_capacity_falls_back_to_resident_decomposition_inputs() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    let c = publish_source_merge(&mut inner, &a, &b);
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([data(&c)]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target = with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
    .unwrap();

    let mut actual: Vec<_> = target.cover().members().collect();
    actual.sort_unstable();
    let mut expected = vec![
        derive(&a).unwrap().get_handle(),
        derive(&b).unwrap().get_handle(),
    ];
    expected.sort_unstable();
    assert_eq!(actual, expected);
    assert!(store.puts > 0 && store.inserts > 0);
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
    let attempts = algebra.derive_attempts.lock().unwrap();
    assert!(attempts.contains(&data(&c)));
    assert!(attempts.contains(&data(&a)));
    assert!(attempts.contains(&data(&b)));
}

#[test]
fn complete_direct_image_does_not_expand_source_decompositions() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    let c = publish_source_merge(&mut inner, &a, &b);
    let fc = publish_derive(&mut inner, &c);
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target = with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
    .unwrap();

    assert_eq!(
        target.cover().members().collect::<Vec<_>>(),
        vec![fc.get_handle()]
    );
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
}

#[test]
fn unrelated_optional_result_metadata_failure_is_inert() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(source.clone()).unwrap();
    let target = publish_derive(&mut inner, &source);
    let unrelated = Inline::<Hash<Blake3>>::new([0x73; 32]);
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            Inline::new([0x51; 32]),
            Inline::new([0x62; 32]),
            unrelated,
        )))
        .unwrap();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    store
        .metadata_failures
        .lock()
        .unwrap()
        .insert(unrelated.raw);
    let source_cover = Cover::from_members(kernel().source_collection(), [source.get_handle()]);

    let attached = kernel().attach_exact(&mut store, &source_cover).unwrap();

    assert_eq!(
        attached.cover().members().collect::<Vec<_>>(),
        vec![target.get_handle()]
    );
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn missing_optional_decomposition_inputs_fall_back_to_direct_construction() {
    let c = archive([(9, 9)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(c.clone()).unwrap();
    let missing_a = Inline::<Hash<Blake3>>::new([0x31; 32]);
    let missing_b = Inline::<Hash<Blake3>>::new([0x42; 32]);
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            missing_a,
            missing_b,
            data(&c),
        )))
        .unwrap();
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target = with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
    .unwrap();

    assert_eq!(target.len(), 1);
    assert_eq!(target.members()[0].1.bytes, derive(&c).unwrap().bytes);
    assert!(algebra.derive_attempts.lock().unwrap().contains(&data(&c)));
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn forged_reverse_decomposition_cannot_supply_a_cover() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let c = archive([(9, 9)]);
    let mut inner = MemoryRepo::default();
    for source in [&a, &b, &c] {
        inner.put::<TestSourceBlob, _>(source.clone()).unwrap();
    }
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            data(&a),
            data(&b),
            data(&c),
        )))
        .unwrap();
    publish_derive(&mut inner, &a);
    publish_derive(&mut inner, &b);
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    assert!(matches!(
        with_selective(&algebra, |kernel| kernel
            .attach_exact(&mut store, &source_cover)),
        Err(ExactDerivedCollectionError::IncompleteCover { .. })
    ));
    assert_eq!((store.puts, store.inserts), (0, 0));

    let target = with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
    .unwrap();
    assert_eq!(target.len(), 1);
    assert_eq!(target.members()[0].1.bytes, derive(&c).unwrap().bytes);
    assert!(algebra.derive_attempts.lock().unwrap().contains(&data(&c)));
    assert!(store.puts > 0 && store.inserts > 0);
}

#[test]
fn algebra_produced_cover_composes_without_an_intermediate_commit() {
    let source = archive([(1, 3)]);
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &source);
    let source_cover = source_cover(&[commit]);

    let first = kernel().ensure_exact(&mut store, &source_cover).unwrap();
    let first_member = first.cover().members().next().unwrap();
    assert!(
        !store
            .records()
            .unwrap()
            .map(Result::unwrap)
            .any(|record| matches!(
                record,
                CollectionRecord::Commit(commit)
                    if commit.collection() == kernel().target_collection().handle()
                        && commit.data() == Handle::<TestTargetBlob>::to_hash(first_member)
            )),
        "the first algebra result must remain unsigned equation evidence",
    );

    let second = second_kernel()
        .ensure_exact(&mut store, first.cover())
        .unwrap();
    assert_eq!(second.len(), 1);
    let mut expected = derive(&source).unwrap().bytes.as_ref().to_vec();
    expected.push(0xB6);
    assert_eq!(second.members()[0].1.bytes.as_ref(), expected.as_slice());
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
    let source_cover = source_cover(&[commit]);

    let cover = compact_exact_target(&kernel(), &mut store, &source_cover).unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(store.puts, 0);
    assert_eq!(store.inserts, 0);
}

#[test]
fn capacity_source_upper_replans_to_lower_resident_cover() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let mut inner = MemoryRepo::default();
    let commits = [
        source_commit(&mut inner, 1, &a),
        source_commit(&mut inner, 2, &b),
    ];
    let upper = publish_source_merge(&mut inner, &a, &b);
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([data(&upper)]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = source_cover(&commits);

    let cover = with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
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
    let commits = [
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
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([data(blocked_upper)]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = source_cover(&commits);

    with_selective(&algebra, |kernel| {
        kernel.ensure_exact(&mut store, &source_cover)
    })
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
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([data(&source)]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = source_cover(&[commit]);

    for _ in 0..2 {
        assert!(matches!(
            with_selective(&algebra, |kernel| kernel.ensure_exact(&mut store, &source_cover)),
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
    let commits = [
        source_commit(&mut inner, 1, &first),
        source_commit(&mut inner, 2, &second),
    ];
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([data(blocked)]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = source_cover(&commits);

    assert!(matches!(
        with_selective(&algebra, |kernel| kernel
            .ensure_exact(&mut store, &source_cover)),
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
    let algebra = SelectiveMapping {
        fatal_derives: BTreeSet::from([data(&source)]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = source_cover(&[commit]);

    assert!(matches!(
        with_selective(&algebra, |kernel| kernel.ensure_exact(&mut store, &source_cover)),
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
    let source_cover = source_cover(&[commit]);
    kernel().ensure_exact(&mut store, &source_cover).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

fn target_merge_records(store: &mut MemoryRepo) -> Vec<CollectionMerge> {
    store
        .records()
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|record| match record {
            CollectionRecord::Merge(claim)
                if claim.collection() == kernel().target_collection().handle() =>
            {
                Some(claim)
            }
            _ => None,
        })
        .collect()
}

fn joined_cover(cover: &CoverAttachment<TestTargetBlob>) -> Blob<TestTargetBlob> {
    let mut members = cover.members().iter();
    let mut joined = members
        .next()
        .expect("nonempty cover has a target member")
        .1
        .clone();
    for (_, member) in members {
        joined = join_test_targets(&joined, member).unwrap();
    }
    joined
}

fn cover_ids(cover: &CoverAttachment<TestTargetBlob>) -> Vec<CollectionData> {
    cover
        .members()
        .iter()
        .map(|(handle, _)| Handle::<TestTargetBlob>::to_hash(*handle))
        .collect()
}

fn target_handles(
    data: impl IntoIterator<Item = CollectionData>,
) -> BTreeSet<Inline<Handle<TestTargetBlob>>> {
    data.into_iter()
        .map(Handle::<TestTargetBlob>::from_hash)
        .collect()
}

fn target_ids(
    handles: impl IntoIterator<Item = Inline<Handle<TestTargetBlob>>>,
) -> Vec<CollectionData> {
    handles
        .into_iter()
        .map(Handle::<TestTargetBlob>::to_hash)
        .collect()
}

#[test]
fn compaction_collapses_same_tier_and_returns_an_exact_tier_stable_cover() {
    let sources = [
        archive([(1, 3)]),
        archive([(2, 4)]),
        archive((10..18).map(|entity| (entity, entity + 20))),
    ];
    let mut store = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut store, index as u8 + 1, source))
        .collect();
    let source_cover = source_cover(&commits);

    let cover = compact_exact_target(&kernel(), &mut store, &source_cover).unwrap();
    let mut tiers = BTreeSet::new();
    for (_, blob) in cover.members() {
        assert!(tiers.insert(blob.bytes.len().max(1).ilog2()));
    }
    assert_eq!(cover.len(), 2);
    assert!(!target_merge_records(&mut store).is_empty());

    let expected_source = sources
        .iter()
        .skip(1)
        .fold(sources[0].clone(), |joined, source| {
            join_test_sources(&joined, source)
        });
    assert_eq!(
        joined_cover(&cover).bytes,
        derive(&expected_source).unwrap().bytes
    );
}

#[test]
fn target_capacity_retires_only_low() {
    let sources = [archive([(1, 3)]), archive([(2, 4)]), archive([(3, 5)])];
    let mut inner = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let source_cover = source_cover(&commits);
    let mut targets: Vec<_> = sources
        .iter()
        .map(|source| derive(source).unwrap())
        .collect();
    targets.sort_unstable_by_key(data);
    let first_pair = SelectiveMapping::pair(&targets[0], &targets[1]);
    let second_pair = SelectiveMapping::pair(&targets[1], &targets[2]);
    let algebra = SelectiveMapping {
        capacity_target_pairs: BTreeSet::from([first_pair]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let cover = with_selective(&algebra, |kernel| {
        compact_exact_target(kernel, &mut store, &source_cover)
    })
    .unwrap();
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
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let source_cover = source_cover(&commits);
    let mut targets: Vec<_> = sources
        .iter()
        .map(|source| derive(source).unwrap())
        .collect();
    targets.sort_unstable_by_key(data);
    let first_pair = SelectiveMapping::pair(&targets[0], &targets[1]);
    let fatal_pair = SelectiveMapping::pair(&targets[2], &targets[3]);
    let algebra = SelectiveMapping {
        fatal_target_pairs: BTreeSet::from([fatal_pair]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    assert!(matches!(
        with_selective(&algebra, |kernel| {
            compact_exact_target(kernel, &mut store, &source_cover)
        }),
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
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let source_cover = source_cover(&commits);
    let targets: Vec<_> = sources
        .iter()
        .map(|source| derive(source).unwrap())
        .collect();
    let capacity_pair = SelectiveMapping::pair(&targets[0], &targets[1]);
    let algebra = SelectiveMapping {
        capacity_target_pairs: BTreeSet::from([capacity_pair]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    let first = with_selective(&algebra, |kernel| {
        compact_exact_target(kernel, &mut store, &source_cover)
    })
    .unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!((store.puts, store.inserts), (0, 0));
    let second = with_selective(&algebra, |kernel| {
        compact_exact_target(kernel, &mut store, &source_cover)
    })
    .unwrap();
    assert_eq!(cover_ids(&first), cover_ids(&second));
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

#[test]
fn compaction_substitutes_new_resident_uppers_through_an_old_nonresident_proof() {
    let sources = [archive([(1, 3)]), archive([(2, 4)]), archive([(3, 5)])];
    let mut store = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut store, index as u8 + 1, source);
            publish_derive(&mut store, source);
            commit
        })
        .collect();
    let source_cover = source_cover(&commits);
    let mut targets: Vec<_> = sources
        .iter()
        .map(|source| {
            let target = derive(source).unwrap();
            (data(&target), target)
        })
        .collect();
    targets.sort_unstable_by_key(|(data, _)| *data);
    let old_intermediate = join_test_targets(&targets[1].1, &targets[2].1).unwrap();
    let old_upper = join_test_targets(&targets[0].1, &old_intermediate).unwrap();
    for claim in [
        CollectionMerge::new(
            kernel().target_collection().handle(),
            targets[1].0,
            targets[2].0,
            data(&old_intermediate),
        ),
        CollectionMerge::new(
            kernel().target_collection().handle(),
            targets[0].0,
            data(&old_intermediate),
            data(&old_upper),
        ),
    ] {
        store.insert(CollectionRecord::Merge(claim)).unwrap();
    }
    let before = kernel().attach_exact(&mut store, &source_cover).unwrap();
    assert_eq!(before.len(), 3);
    let after = compact_exact_target(&kernel(), &mut store, &source_cover).unwrap();
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
fn compaction_is_cover_order_deterministic_and_repeatedly_idempotent() {
    let sources = [
        archive([(1, 3)]),
        archive([(2, 4)]),
        archive([(3, 5)]),
        archive([(4, 6)]),
    ];
    let mut first_store = MemoryRepo::default();
    let first_commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut first_store, index as u8 + 1, source))
        .collect();
    let mut second_store = MemoryRepo::default();
    let mut second_commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut second_store, index as u8 + 1, source))
        .collect();
    second_commits.reverse();
    let first_cover = source_cover(&first_commits);
    let second_cover = source_cover(&second_commits);

    let first = compact_exact_target(&kernel(), &mut first_store, &first_cover).unwrap();
    let second = compact_exact_target(&kernel(), &mut second_store, &second_cover).unwrap();
    assert_eq!(cover_ids(&first), cover_ids(&second));
    assert_eq!(
        target_merge_records(&mut first_store),
        target_merge_records(&mut second_store)
    );

    let records_before: Vec<_> = first_store.records().unwrap().map(Result::unwrap).collect();
    let repeated = compact_exact_target(&kernel(), &mut first_store, &first_cover).unwrap();
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
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let source_cover = source_cover(&commits);
    let live = Arc::new(AtomicUsize::new(0));
    let mut store = GuardStore {
        inner,
        live: Arc::clone(&live),
        events: Vec::new(),
    };

    compact_exact_target(&kernel(), &mut store, &source_cover).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let first_merge = store
        .events
        .iter()
        .position(|event| matches!(event, WriteEvent::Insert(CollectionRecord::Merge(_))))
        .expect("colliding cover publishes a MERGE");
    let descriptor_data = Handle::<SimpleArchive>::to_hash(kernel().target_collection().handle());
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
    let join_commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut join_store, index as u8 + 1, source))
        .collect();
    let join_cover = source_cover(&join_commits);
    let targets = [derive(&sources[0]).unwrap(), derive(&sources[1]).unwrap()];
    let algebra = SelectiveMapping {
        fatal_target_pairs: BTreeSet::from([SelectiveMapping::pair(&targets[0], &targets[1])]),
        ..SelectiveMapping::default()
    };
    assert!(matches!(
        with_selective(&algebra, |kernel| {
            compact_exact_target(kernel, &mut join_store, &join_cover)
        }),
        Err(ExactTargetCompactionError::Merge { .. })
    ));
    assert!(target_merge_records(&mut join_store).is_empty());

    let mut inner = MemoryRepo::default();
    let put_commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let put_cover = source_cover(&put_commits);
    let mut put_store = RejectPutStore { inner, puts: 0 };
    assert!(matches!(
        compact_exact_target(&kernel(), &mut put_store, &put_cover),
        Err(ExactTargetCompactionError::Storage { .. })
    ));
    assert!(target_merge_records(&mut put_store.inner).is_empty());
}

#[test]
fn discarded_merge_insert_stalls_instead_of_looping() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let mut inner = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let commit = source_commit(&mut inner, index as u8 + 1, source);
            publish_derive(&mut inner, source);
            commit
        })
        .collect();
    let source_cover = source_cover(&commits);
    let mut store = DropMergeStore { inner };

    assert!(matches!(
        compact_exact_target(&kernel(), &mut store, &source_cover),
        Err(ExactTargetCompactionError::Stalled { cover }) if cover.len() == 2
    ));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

struct LossyStore {
    inner: MemoryRepo,
    discard: CollectionData,
}

impl BlobStorePut for LossyStore {
    type PutError = <MemoryRepo as BlobStorePut>::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let blob = item.to_blob();
        if Handle::<S>::to_hash(blob.get_handle()) == self.discard {
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
    let output = derive(&source).unwrap();
    let mut store = LossyStore {
        inner,
        discard: data(&output),
    };
    let source_cover = source_cover(&[commit]);
    match kernel().ensure_exact(&mut store, &source_cover) {
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
            kernel().target_collection().handle(),
            data(&source),
            data(&missing),
        )))
        .unwrap();
    let source_cover = source_cover(&[commit]);
    match kernel().attach_exact(&mut store, &source_cover) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected missing-output error: {error:?}"),
        Ok(_) => panic!("missing output was incorrectly admitted"),
    }
    assert_eq!(
        kernel()
            .ensure_exact(&mut store, &source_cover)
            .unwrap()
            .len(),
        1,
    );
}

#[test]
fn offered_upper_is_selected_as_one_remote_cover_member() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let targets = [derive(&sources[0]).unwrap(), derive(&sources[1]).unwrap()];
    let upper = join_test_targets(&targets[0], &targets[1]).unwrap();
    let mut store = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut store, index as u8 + 1, source))
        .collect();
    let source_cover = source_cover(&commits);
    for (source, target) in sources.iter().zip(&targets) {
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                kernel().target_collection().handle(),
                data(source),
                data(target),
            )))
            .unwrap();
    }
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().target_collection().handle(),
            data(&targets[0]),
            data(&targets[1]),
            data(&upper),
        )))
        .unwrap();

    let offered = target_handles([data(&targets[0]), data(&targets[1]), data(&upper)]);
    match kernel()
        .probe_exact(&mut store, &source_cover, &offered)
        .unwrap()
    {
        ExactAttachPlan::Fetch(fetch) => assert_eq!(fetch, vec![upper.get_handle()]),
        ExactAttachPlan::Ready(_) => panic!("nonresident offered upper was already ready"),
    }

    store.put::<TestTargetBlob, _>(upper.clone()).unwrap();
    match kernel()
        .probe_exact(&mut store, &source_cover, &offered)
        .unwrap()
    {
        ExactAttachPlan::Ready(cover) => {
            assert_eq!(cover_ids(&cover), vec![data(&upper)]);
        }
        ExactAttachPlan::Fetch(fetch) => panic!("landed upper still requested: {fetch:?}"),
    }
}

#[test]
fn unavailable_offered_upper_replans_to_offered_lower_cover() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let targets = [derive(&sources[0]).unwrap(), derive(&sources[1]).unwrap()];
    let upper = join_test_targets(&targets[0], &targets[1]).unwrap();
    let mut store = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut store, index as u8 + 1, source))
        .collect();
    let source_cover = source_cover(&commits);
    for (source, target) in sources.iter().zip(&targets) {
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                kernel().target_collection().handle(),
                data(source),
                data(target),
            )))
            .unwrap();
    }
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().target_collection().handle(),
            data(&targets[0]),
            data(&targets[1]),
            data(&upper),
        )))
        .unwrap();

    let mut offered = target_handles([data(&targets[0]), data(&targets[1]), data(&upper)]);
    match kernel()
        .probe_exact(&mut store, &source_cover, &offered)
        .unwrap()
    {
        ExactAttachPlan::Fetch(fetch) => assert_eq!(fetch, vec![upper.get_handle()]),
        ExactAttachPlan::Ready(_) => panic!("nonresident offered upper was already ready"),
    }

    offered.remove(&upper.get_handle());
    let expected: Vec<_> = offered
        .iter()
        .copied()
        .map(Handle::<TestTargetBlob>::to_hash)
        .collect();
    match kernel()
        .probe_exact(&mut store, &source_cover, &offered)
        .unwrap()
    {
        ExactAttachPlan::Fetch(fetch) => assert_eq!(target_ids(fetch), expected),
        ExactAttachPlan::Ready(_) => panic!("nonresident offered lowers were already ready"),
    }
    for target in targets {
        store.put::<TestTargetBlob, _>(target).unwrap();
    }
    match kernel()
        .probe_exact(&mut store, &source_cover, &offered)
        .unwrap()
    {
        ExactAttachPlan::Ready(cover) => assert_eq!(cover_ids(&cover), expected),
        ExactAttachPlan::Fetch(fetch) => panic!("landed lowers still requested: {fetch:?}"),
    }
}

#[test]
fn offered_upper_does_not_displace_a_complete_resident_lower_cover() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let targets = [derive(&sources[0]).unwrap(), derive(&sources[1]).unwrap()];
    let upper = join_test_targets(&targets[0], &targets[1]).unwrap();
    let mut store = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut store, index as u8 + 1, source))
        .collect();
    let source_cover = source_cover(&commits);
    for (source, target) in sources.iter().zip(&targets) {
        store.put::<TestTargetBlob, _>(target.clone()).unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                kernel().target_collection().handle(),
                data(source),
                data(target),
            )))
            .unwrap();
    }
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().target_collection().handle(),
            data(&targets[0]),
            data(&targets[1]),
            data(&upper),
        )))
        .unwrap();

    match kernel()
        .probe_exact(
            &mut store,
            &source_cover,
            &BTreeSet::from([upper.get_handle()]),
        )
        .unwrap()
    {
        ExactAttachPlan::Ready(cover) => {
            let expected: Vec<_> = targets
                .iter()
                .map(data)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            assert_eq!(cover_ids(&cover), expected);
        }
        ExactAttachPlan::Fetch(fetch) => {
            panic!("remote upper displaced a complete resident cover: {fetch:?}")
        }
    }
}

#[test]
fn unrelated_and_rejected_offers_never_become_fetch_work() {
    let source = archive([(1, 3)]);
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &source);
    let source_cover = source_cover(&[commit]);
    let lying_output = data(&derive(&archive([(9, 9)])).unwrap());
    store
        .insert(CollectionRecord::Derive(CollectionDerive::new(
            kernel().target_collection().handle(),
            data(&source),
            lying_output,
        )))
        .unwrap();
    let unrelated = CollectionData::new([0xEE; 32]);

    match kernel().probe_exact(
        &mut store,
        &source_cover,
        &target_handles([lying_output, unrelated]),
    ) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected rejected-offer error: {error:?}"),
        Ok(ExactAttachPlan::Fetch(fetch)) => {
            panic!("unrelated or rejected offers became fetch work: {fetch:?}")
        }
        Ok(ExactAttachPlan::Ready(_)) => panic!("rejected equation completed the cover"),
    }
}

#[test]
fn ungrounded_source_superset_cannot_escape_the_cover() {
    let a = archive([(1, 3)]);
    let c = archive([(3, 5)]);
    let ac = join_test_sources(&a, &c);
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &a);
    let source_cover = source_cover(&[commit]);
    store.put::<TestSourceBlob, _>(c.clone()).unwrap();
    store.put::<TestSourceBlob, _>(ac.clone()).unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            data(&a),
            data(&c),
            data(&ac),
        )))
        .unwrap();

    let cover = kernel().ensure_exact(&mut store, &source_cover).unwrap();
    assert_eq!(cover.members()[0].1.bytes, derive(&a).unwrap().bytes);
    let derives: Vec<_> = store
        .records()
        .unwrap()
        .map(Result::unwrap)
        .filter_map(|record| match record {
            CollectionRecord::Derive(claim) => Some(claim.input()),
            _ => None,
        })
        .collect();
    assert_eq!(derives, vec![data(&a)]);
}

#[test]
fn typed_lifecycle_rejects_a_lying_source_descriptor() {
    let lying_source = descriptor::naming(
        "source",
        test_team(),
        <TestTargetBlob as MetaDescribe>::id(),
        reach::private(),
    );
    let result = ExactDerivedCollection::<TestSourceBlob, TestTargetBlob, TestSuffixMapping>::new(
        lying_source,
        kernel().target_descriptor().clone(),
    );
    assert!(matches!(
        result,
        Err(ExactDerivedCollectionError::Resolution(_))
    ));
}

#[test]
fn identity_descriptor_pair_is_rejected() {
    let descriptor = source_root();
    let result = ExactDerivedCollection::<TestSourceBlob, TestSourceBlob, IdentityMapping>::new(
        descriptor.clone(),
        descriptor,
    );
    assert!(matches!(
        result,
        Err(ExactDerivedCollectionError::Resolution(reason))
            if reason.contains("distinct source and target descriptors")
    ));
}
