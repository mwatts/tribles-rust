use super::*;

use std::cell::RefCell;
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ed25519_dalek::SigningKey;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::utf8string::UTF8String;
use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, BlobEncoding, IntoBlob, TryFromBlob};
use crate::collection::descriptor;
use crate::collection::simplearchive_union;
use crate::collection::{AdmissionPolicy, CollectionPolicy};
use crate::collection::{CollectionCommit, CollectionRead, CollectionStoreExt};
use crate::id::{ExclusiveId, Id};
use crate::id_hex;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::Inline;
use crate::metadata::MetaDescribe;
use crate::repo::memoryrepo::{MemoryRepo, MemoryRepoSnapshot};
use crate::repo::{
    BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut, SnapshotSource, StoreChanges,
    StoreSnapshot,
};
use crate::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

/// The one team every collection in these tests belongs to.
fn test_team() -> ed25519_dalek::VerifyingKey {
    SigningKey::from_bytes(&[1; 32]).verifying_key()
}

fn test_policy() -> CollectionPolicy {
    CollectionPolicy::new(
        AdmissionPolicy::direct(test_team()),
        AdmissionPolicy::direct(test_team()),
    )
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
    fn validate_member<R>(
        _descriptor: &Fragment,
        member: &Blob<Self>,
        _reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        simplearchive_union::validate_element(member.as_transmute::<SimpleArchive>())
            .map_err(|error| CollectionOperationError::Fatal(error.to_string()))
    }

    fn join_members<R>(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
        _reader: &R,
    ) -> Result<Option<Blob<Self>>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
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
        .map(Some)
        .map_err(|error| CollectionOperationError::Fatal(error.to_string()))
    }
}

fn source_descriptor(name: &str) -> Fragment {
    descriptor::naming::<TestSourceBlob>(name, test_policy())
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

/// Test-only UTF-8 target whose bytes come from a source attachment.
/// Minted with `trible genid` on 2026-08-30.
const ATTACHED_TEXT_ENCODING_V1: Id = id_hex!("9E0CC64DE6D66EC9231B781D7215C2EC");

struct AttachedTextBlob;

impl BlobEncoding for AttachedTextBlob {}

impl MetaDescribe for AttachedTextBlob {
    fn describe() -> Fragment {
        let id = ATTACHED_TEXT_ENCODING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-attached-text-v1",
            crate::metadata::description: "Test-only UTF-8 target resolved from a handle carried by one SimpleArchive source member.",
            crate::metadata::tag: crate::metadata::KIND_BLOB_ENCODING,
        }
    }
}

impl CollectionEncoding for AttachedTextBlob {
    fn validate_member<R>(
        _descriptor: &Fragment,
        member: &Blob<Self>,
        _reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        std::str::from_utf8(member.bytes.as_ref())
            .map(|_| ())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))
    }
}

/// Test-only source-attachment dereference mapping.
/// Minted with `trible genid` on 2026-08-30.
const ATTACHED_TEXT_MAPPING_V1: Id = id_hex!("B0DF19D3C1B35052C31E722F1294D9CB");

struct AttachedTextMappingV1;

impl MetaDescribe for AttachedTextMappingV1 {
    fn describe() -> Fragment {
        let id = ATTACHED_TEXT_MAPPING_V1;
        crate::macros::entity! { ExclusiveId::force_ref(&id) @
            crate::metadata::name: "exact-derived-attached-text-mapping-v1",
            crate::metadata::description: "Test-only mapping that resolves the UTF8String named by metadata::description in a SimpleArchive member.",
            crate::metadata::tag: crate::metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
        }
    }
}

fn attached_text_mapping_fragment() -> Fragment {
    crate::macros::entity! {
        crate::metadata::tag: crate::collection::KIND_COLLECTION_MAPPING,
        crate::collection::mapping_algorithm*: <AttachedTextMappingV1 as MetaDescribe>::describe(),
    }
}

struct AttachedTextMapping;

impl CollectionMapping for AttachedTextMapping {
    type Source = SimpleArchive;
    type Target = AttachedTextBlob;

    fn fragment(&self) -> Fragment {
        attached_text_mapping_fragment()
    }

    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, ATTACHED_TEXT_MAPPING_V1, "attached text")?;
        Ok(Self)
    }

    fn map<R>(
        &self,
        source: &Blob<SimpleArchive>,
        reader: &R,
    ) -> Result<Blob<AttachedTextBlob>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        let description = source
            .bytes
            .as_ref()
            .chunks_exact(TRIBLE_LEN)
            .find(|row| row[16..32] == crate::metadata::description.id()[..])
            .map(|row| {
                Inline::<Handle<UTF8String>>::new(
                    row[32..64]
                        .try_into()
                        .expect("SimpleArchive rows have a 32-byte value"),
                )
            })
            .ok_or_else(|| {
                CollectionOperationError::Fatal(
                    "source member has no metadata::description attachment".to_owned(),
                )
            })?;
        let text: Blob<UTF8String> = reader.get(description).map_err(|source| {
            CollectionOperationError::Fatal(format!(
                "resolve source metadata::description attachment: {source}"
            ))
        })?;
        Ok(Blob::new(text.bytes))
    }
}

impl CollectionEncoding for TestTargetBlob {
    fn validate_member<R>(
        _descriptor: &Fragment,
        member: &Blob<Self>,
        _reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        validate_test_target(member)
    }

    fn join_members<R>(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
        _reader: &R,
    ) -> Result<Option<Blob<Self>>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
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
        join_test_targets(low, high).map(Some)
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
    fn validate_member<R>(
        _descriptor: &Fragment,
        member: &Blob<Self>,
        _reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        validate_second_test_target(member)
    }

    fn join_members<R>(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
        _reader: &R,
    ) -> Result<Option<Blob<Self>>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
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
        Ok(Some(Blob::new(bytes.into())))
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
    descriptor::deriving(source, &TestSuffixMapping, test_policy())
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

fn register_kernel(store: &mut MemoryRepo) -> ExactDerivedCollection<TestSuffixMapping> {
    let source = store
        .register_collection::<TestSourceBlob>(source_root())
        .unwrap();
    let target = store
        .derive(source, TestSuffixMapping, test_policy())
        .unwrap();
    ExactDerivedCollection::new(source, target).unwrap()
}

fn kernel() -> ExactDerivedCollection<TestSuffixMapping> {
    register_kernel(&mut MemoryRepo::default())
}

#[test]
fn mapping_entity_id_substitution_preserves_binding_semantics() {
    let mut store = MemoryRepo::default();
    let source = store
        .register_collection::<TestSourceBlob>(source_root())
        .unwrap();
    let canonical = target_root(source.handle());
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
    let target = store
        .register_collection::<TestTargetBlob>(substituted)
        .unwrap();
    let exact = ExactDerivedCollection::<TestSuffixMapping>::new(source, target).unwrap();
    let input = archive([(1, 2)]);
    store.put::<TestSourceBlob, _>(input.clone()).unwrap();
    let cover = Cover::from_members(source, [input.get_handle()]);
    exact
        .ensure(&mut store, &cover)
        .expect("binding depends on mapping facts, not intrinsic-id minting history");
}

#[test]
fn ensure_resolves_source_member_attachments_through_the_reader() {
    let authority = SigningKey::from_bytes(&[1; 32]);
    let source_descriptor = simplearchive_union::descriptor(
        "reader-aware-source",
        CollectionPolicy::new(
            AdmissionPolicy::direct(authority.verifying_key()),
            AdmissionPolicy::direct(authority.verifying_key()),
        ),
    );
    let expected = "the attachment is part of the source closure";
    let source = crate::macros::entity! {
        crate::metadata::description: expected.to_owned(),
    };

    let mut store = MemoryRepo::default();
    let source_collection = store
        .register_collection::<SimpleArchive>(source_descriptor)
        .unwrap();
    let target_collection = store
        .derive(source_collection, AttachedTextMapping, test_policy())
        .unwrap();
    let exact =
        ExactDerivedCollection::<AttachedTextMapping>::new(source_collection, target_collection)
            .unwrap();
    let commit = store.commit(source_collection, &authority, source).unwrap();
    let cover = Cover::from_data(source_collection, [commit.data()]);

    let attached = exact.ensure(&mut store, &cover).unwrap();
    assert_eq!(attached.len(), 1);
    let snapshot = store.snapshot().unwrap();
    let attached_blob: Blob<AttachedTextBlob> =
        snapshot.get(attached.members().next().unwrap()).unwrap();
    assert_eq!(attached_blob.bytes.as_ref(), expected.as_bytes());

    // A second read-only pass follows the resident DERIVE without remapping the
    // source attachment.
    let reattached = exact.attach(&mut store, &cover).unwrap();
    let snapshot = store.snapshot().unwrap();
    let reattached_blob: Blob<AttachedTextBlob> =
        snapshot.get(reattached.members().next().unwrap()).unwrap();
    assert_eq!(reattached_blob.bytes.as_ref(), expected.as_bytes());
}

fn selective_kernel(mapping: SelectiveMapping) -> ExactDerivedCollection<SelectiveMapping> {
    let mut store = MemoryRepo::default();
    let source = store
        .register_collection::<TestSourceBlob>(source_root())
        .unwrap();
    let target = store
        .derive(source, mapping.clone(), test_policy())
        .unwrap();
    ExactDerivedCollection::with_mapping(source, target, mapping).unwrap()
}

fn register_second_kernel(
    store: &mut MemoryRepo,
) -> ExactDerivedCollection<SecondTestSuffixMapping> {
    let first = register_kernel(store);
    let target = store
        .derive(
            first.target_collection(),
            SecondTestSuffixMapping,
            test_policy(),
        )
        .unwrap();
    ExactDerivedCollection::new(first.target_collection(), target).unwrap()
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
    operation: impl FnOnce(&ExactDerivedCollection<SelectiveMapping>) -> R,
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

impl CollectionMapping for TestSuffixMapping {
    type Source = TestSourceBlob;
    type Target = TestTargetBlob;

    fn fragment(&self) -> Fragment {
        test_suffix_mapping_fragment()
    }

    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, TEST_SUFFIX_MAPPING_V1, "test suffix")?;
        Ok(Self)
    }

    fn map<R>(
        &self,
        source: &Blob<TestSourceBlob>,
        _reader: &R,
    ) -> Result<Blob<TestTargetBlob>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        Ok(derive(source).unwrap())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SecondTestSuffixMapping;

impl CollectionMapping for SecondTestSuffixMapping {
    type Source = TestTargetBlob;
    type Target = SecondTestTargetBlob;

    fn fragment(&self) -> Fragment {
        second_test_suffix_mapping_fragment()
    }

    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, SECOND_TEST_SUFFIX_MAPPING_V1, "second test suffix")?;
        Ok(Self)
    }

    fn map<R>(
        &self,
        source: &Blob<TestTargetBlob>,
        _reader: &R,
    ) -> Result<Blob<SecondTestTargetBlob>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        let mut bytes = source.bytes.as_ref().to_vec();
        bytes.push(0xB6);
        Ok(Blob::new(bytes.into()))
    }
}

struct IdentityMapping;

impl CollectionMapping for IdentityMapping {
    type Source = TestSourceBlob;
    type Target = TestSourceBlob;

    fn fragment(&self) -> Fragment {
        Fragment::empty()
    }

    fn bind(_source: &Fragment, _target: &Fragment) -> Result<Self, CollectionOperationError> {
        Ok(Self)
    }

    fn map<R>(
        &self,
        source: &Blob<TestSourceBlob>,
        _reader: &R,
    ) -> Result<Blob<TestSourceBlob>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        Ok(source.clone())
    }
}

impl CollectionMapping for SelectiveMapping {
    type Source = TestSourceBlob;
    type Target = TestTargetBlob;

    fn fragment(&self) -> Fragment {
        test_suffix_mapping_fragment()
    }

    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        require_mapping(target, TEST_SUFFIX_MAPPING_V1, "test suffix")?;
        Ok(Self::default())
    }

    fn map<R>(
        &self,
        source: &Blob<TestSourceBlob>,
        _reader: &R,
    ) -> Result<Blob<TestTargetBlob>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
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
    let exact = register_kernel(store);
    store.put::<TestSourceBlob, _>(blob.clone()).unwrap();
    let metadata = store
        .put::<SimpleArchive, _>(TribleSet::new().to_blob())
        .unwrap();
    let commit = CollectionCommit::sign(
        &SigningKey::from_bytes(&[key; 32]),
        exact.source_collection().handle(),
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
    let exact = register_kernel(store);
    let output = derive(input).unwrap();
    store.put::<TestTargetBlob, _>(output.clone()).unwrap();
    store
        .insert(CollectionRecord::Derive(CollectionDerive::new(
            exact.target_collection().handle(),
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
    let exact = register_kernel(store);
    let result = join_test_sources(low, high);
    store.put::<TestSourceBlob, _>(result.clone()).unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            exact.source_collection().handle(),
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

fn collection_records(store: &mut MemoryRepo) -> Vec<CollectionRecord> {
    let snapshot = store.snapshot().unwrap();
    snapshot.records().unwrap().map(Result::unwrap).collect()
}

fn derived_inputs(store: &mut MemoryRepo) -> Vec<CollectionData> {
    collection_records(store)
        .into_iter()
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

impl SnapshotSource for PanicStore {
    type Snapshot = MemoryRepoSnapshot;
    type SnapshotError = Infallible;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        panic!("empty cover opened a snapshot")
    }
}

impl CollectionStore for PanicStore {
    type InsertError = Infallible;

    fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
        panic!("empty cover inserted a record")
    }
}

#[test]
fn empty_cover_performs_no_store_operation() {
    let mut store = PanicStore;
    let cover = source_cover(&[]);
    assert!(kernel().attach(&mut store, &cover).unwrap().is_empty());
    assert!(kernel().ensure(&mut store, &cover).unwrap().is_empty());
}

#[test]
fn empty_cover_still_belongs_to_one_exact_collection() {
    let mut store = PanicStore;
    let foreign_collection = MemoryRepo::default()
        .register_collection::<TestSourceBlob>(source_descriptor("foreign"))
        .unwrap();
    let foreign = Cover::from_members(foreign_collection, []);
    for result in [
        kernel().attach(&mut store, &foreign),
        kernel().ensure(&mut store, &foreign),
    ] {
        assert!(matches!(
            result,
            Err(ExactDerivedCollectionError::InvalidCover(_))
        ));
    }
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

struct DemandGuardSnapshot {
    inner: MemoryRepoSnapshot,
    missing_gets: Arc<AtomicUsize>,
    metadata_failures: BTreeSet<[u8; 32]>,
}

impl Clone for DemandGuardSnapshot {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            missing_gets: Arc::clone(&self.missing_gets),
            metadata_failures: self.metadata_failures.clone(),
        }
    }
}

impl StoreSnapshot for DemandGuardSnapshot {
    fn changes_since(&self, previous: &Self) -> StoreChanges {
        let mut changes = self.inner.changes_since(&previous.inner);
        if self.metadata_failures != previous.metadata_failures {
            changes = changes.union(StoreChanges::BLOBS);
        }
        changes
    }
}

impl BlobStoreMeta for DemandGuardSnapshot {
    type MetaError = InjectedMetadataError;

    fn metadata<S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<Option<crate::repo::BlobMetadata>, Self::MetaError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        if self.metadata_failures.contains(&handle.raw) {
            return Err(InjectedMetadataError);
        }
        self.inner.metadata(handle).map_err(|never| match never {})
    }
}

impl BlobStoreList for DemandGuardSnapshot {
    type Iter<'a>
        = <MemoryRepoSnapshot as BlobStoreList>::Iter<'a>
    where
        Self: 'a;
    type Err = <MemoryRepoSnapshot as BlobStoreList>::Err;

    fn blobs<'a>(&'a self) -> Self::Iter<'a> {
        self.inner.blobs()
    }
}

impl BlobStoreGet for DemandGuardSnapshot {
    type GetError<E: Error + Send + Sync + 'static> =
        <MemoryRepoSnapshot as BlobStoreGet>::GetError<E>;

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

impl CollectionRead for DemandGuardSnapshot {
    type RecordsError = <MemoryRepoSnapshot as CollectionRead>::RecordsError;
    type RecordIter<'a>
        = <MemoryRepoSnapshot as CollectionRead>::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
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

impl SnapshotSource for CountingStore {
    type Snapshot = DemandGuardSnapshot;
    type SnapshotError = <MemoryRepo as SnapshotSource>::SnapshotError;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        Ok(DemandGuardSnapshot {
            inner: self.inner.snapshot()?,
            missing_gets: Arc::clone(&self.missing_gets),
            metadata_failures: self.metadata_failures.lock().unwrap().clone(),
        })
    }
}

impl CollectionStore for CountingStore {
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;

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

    let cover = kernel().ensure(&mut store, &source_cover).unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(store.puts, 0);
    assert_eq!(store.inserts, 0);
}

#[test]
fn warm_attach_and_ensure_execute_no_collection_algebra() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let targets = [derive(&sources[0]).unwrap(), derive(&sources[1]).unwrap()];
    let upper = join_test_targets(&targets[0], &targets[1]).unwrap();
    let mut inner = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut inner, index as u8 + 1, source))
        .collect();
    for (source, target) in sources.iter().zip(&targets) {
        inner.put::<TestTargetBlob, _>(target.clone()).unwrap();
        inner
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                kernel().target_collection().handle(),
                data(source),
                data(target),
            )))
            .unwrap();
    }
    inner.put::<TestTargetBlob, _>(upper.clone()).unwrap();
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().target_collection().handle(),
            data(&targets[0]),
            data(&targets[1]),
            data(&upper),
        )))
        .unwrap();

    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = source_cover(&commits);
    with_selective(&algebra, |kernel| {
        assert_eq!(
            cover_ids(&kernel.attach(&mut store, &source_cover).unwrap()),
            vec![data(&upper)]
        );
        assert_eq!(
            cover_ids(&kernel.ensure(&mut store, &source_cover).unwrap()),
            vec![data(&upper)]
        );
    });

    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert!(algebra.derive_attempts.lock().unwrap().is_empty());
    assert!(algebra.target_attempts.lock().unwrap().is_empty());
}

#[test]
fn newly_published_derives_activate_a_resident_target_merge() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let targets = [derive(&sources[0]).unwrap(), derive(&sources[1]).unwrap()];
    let upper = join_test_targets(&targets[0], &targets[1]).unwrap();
    let mut store = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut store, index as u8 + 1, source))
        .collect();

    // The MERGE is useful stored work even while its inputs are absent. Once
    // ensure publishes those inputs and their DERIVEs, fresh resolution must
    // select the existing result rather than return both inputs for a caller
    // to join again.
    store.put::<TestTargetBlob, _>(upper.clone()).unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().target_collection().handle(),
            data(&targets[0]),
            data(&targets[1]),
            data(&upper),
        )))
        .unwrap();

    let ensured = kernel()
        .ensure(&mut store, &source_cover(&commits))
        .unwrap();
    assert_eq!(cover_ids(&ensured), vec![data(&upper)]);
    let mut expected_inputs: Vec<_> = sources.iter().map(data).collect();
    expected_inputs.sort_unstable();
    assert_eq!(derived_inputs(&mut store), expected_inputs);
}

#[test]
fn registration_publishes_the_complete_descriptor_attachment_closure() {
    let mut store = MemoryRepo::default();
    let exact = register_kernel(&mut store);

    let snapshot = store.snapshot().unwrap();
    for collection in [
        exact.source_collection().handle(),
        exact.target_collection().handle(),
    ] {
        let descriptor = crate::collection::api::load_collection_descriptor(&snapshot, collection)
            .unwrap()
            .fragment;
        let stored_descriptor: Blob<SimpleArchive> = snapshot
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
        let embedded_snapshot = embedded.snapshot().unwrap();
        for (handle, expected) in embedded_snapshot {
            let actual: Blob<UnknownBlob> = snapshot
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
        kernel.complete(
            &mut store,
            &source_cover,
            &mut ExactPlannerBlocks::default(),
        )
    })
    .unwrap();

    assert_eq!(
        cover_ids(&target),
        vec![data(&join_test_targets(&fa, &fb).unwrap())]
    );
    assert_eq!((store.puts, store.inserts), (1, 1));
    assert!(!algebra.derive_attempts.lock().unwrap().contains(&data(&c)));
}

#[test]
fn ensure_joins_resident_target_children_before_crossing_the_mapping() {
    let a = archive([(1, 3)]);
    let b = archive((2u8..=9).map(|entity| (entity, entity + 20)));
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    let c = publish_source_merge(&mut inner, &a, &b);
    let fa = publish_derive(&mut inner, &a);
    let fb = publish_derive(&mut inner, &b);
    let expected = join_test_targets(&fa, &fb).unwrap();
    let pair = SelectiveMapping::pair(&fa, &fb);
    assert_ne!(
        fa.bytes.len().max(1).ilog2(),
        fb.bytes.len().max(1).ilog2(),
        "the local target-child rule must not depend on the global LSM tier",
    );
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(cover_ids(&target), vec![data(&expected)]);
    assert!(algebra.derive_attempts.lock().unwrap().is_empty());
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert_eq!(algebra.target_attempts.lock().unwrap().as_slice(), &[pair]);
    assert_eq!((store.puts, store.inserts), (1, 1));
}

#[test]
fn target_child_capacity_falls_back_to_the_corresponding_source_node() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    let c = publish_source_merge(&mut inner, &a, &b);
    let fa = publish_derive(&mut inner, &a);
    let fb = publish_derive(&mut inner, &b);
    let pair = SelectiveMapping::pair(&fa, &fb);
    let algebra = SelectiveMapping {
        capacity_target_pairs: BTreeSet::from([pair]),
        ..SelectiveMapping::default()
    };
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(cover_ids(&target), vec![data(&derive(&c).unwrap())]);
    assert_eq!(algebra.target_attempts.lock().unwrap().as_slice(), &[pair]);
    assert_eq!(
        algebra.derive_attempts.lock().unwrap().as_slice(),
        &[data(&c)]
    );
}

#[test]
fn ensure_uses_a_resident_corresponding_source_node_before_target_children() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let c = join_test_sources(&a, &b);
    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    inner.put::<TestSourceBlob, _>(c.clone()).unwrap();
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            data(&a),
            data(&b),
            data(&c),
        )))
        .unwrap();
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(
        kernel().source_collection(),
        [a.get_handle(), b.get_handle()],
    );

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(cover_ids(&target), vec![data(&derive(&c).unwrap())]);
    assert_eq!(
        algebra.derive_attempts.lock().unwrap().as_slice(),
        &[data(&c)]
    );
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert!(algebra.target_attempts.lock().unwrap().is_empty());
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn ensure_uses_resident_corresponding_source_node_before_target_grandchildren() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let e = archive([(5, 6)]);
    let d = join_test_sources(&a, &b);
    let c = join_test_sources(&d, &e);
    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    inner.put::<TestSourceBlob, _>(c.clone()).unwrap();
    for (low, high, result) in [(&a, &b, &d), (&d, &e, &c)] {
        inner
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                kernel().source_collection().handle(),
                data(low),
                data(high),
                data(result),
            )))
            .unwrap();
    }
    publish_derive(&mut inner, &a);
    publish_derive(&mut inner, &b);
    publish_derive(&mut inner, &e);
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    let derive_attempts = algebra.derive_attempts.lock().unwrap().clone();
    let target_attempts = algebra.target_attempts.lock().unwrap().clone();
    assert_eq!(
        (derive_attempts, target_attempts),
        (vec![data(&c)], Vec::new())
    );
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert_eq!(cover_ids(&target), vec![data(&derive(&c).unwrap())]);
}

#[test]
fn recursive_preferences_follow_the_first_fully_actionable_source_producer() {
    let p1_low_a = archive([(1, 11)]);
    let p1_low_b = archive([(2, 12), (3, 13)]);
    let p1_low = join_test_sources(&p1_low_a, &p1_low_b);
    let p1_high = archive([(4, 14), (5, 15), (6, 16)]);

    let p2_low_a = archive([(1, 11), (4, 14)]);
    let p2_low_b = archive([(5, 15)]);
    let p2_low = join_test_sources(&p2_low_a, &p2_low_b);
    let p2_high = archive([(2, 12), (3, 13), (6, 16)]);

    let c1 = join_test_sources(&p1_low, &p1_high);
    let c2 = join_test_sources(&p2_low, &p2_high);
    assert_eq!(c1.bytes, c2.bytes);

    let mut alternatives = [
        (
            SelectiveMapping::pair(&p1_low, &p1_high),
            (&p1_low, &p1_high, &p1_low_a, &p1_low_b),
        ),
        (
            SelectiveMapping::pair(&p2_low, &p2_high),
            (&p2_low, &p2_high, &p2_low_a, &p2_low_b),
        ),
    ];
    alternatives.sort_unstable_by_key(|(pair, _)| *pair);
    let (_, unavailable) = alternatives[0];
    let (_, available) = alternatives[1];

    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    for (low, high, result) in [
        (
            p1_low_a.get_handle(),
            p1_low_b.get_handle(),
            p1_low.get_handle(),
        ),
        (
            p2_low_a.get_handle(),
            p2_low_b.get_handle(),
            p2_low.get_handle(),
        ),
        (p1_low.get_handle(), p1_high.get_handle(), c1.get_handle()),
        (p2_low.get_handle(), p2_high.get_handle(), c2.get_handle()),
    ] {
        inner
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                kernel().source_collection().handle(),
                Handle::<TestSourceBlob>::to_hash(low),
                Handle::<TestSourceBlob>::to_hash(high),
                Handle::<TestSourceBlob>::to_hash(result),
            )))
            .unwrap();
    }

    let (available_low, available_high, available_low_a, available_low_b) = available;
    for source in [available_low_a, available_low_b, available_high] {
        publish_derive(&mut inner, source);
    }
    let nested_pair = SelectiveMapping::pair(
        &derive(available_low_a).unwrap(),
        &derive(available_low_b).unwrap(),
    );
    let available_low_target = derive(available_low).unwrap();
    let root_pair = SelectiveMapping::pair(&available_low_target, &derive(available_high).unwrap());
    let expected = derive(&c1).unwrap();

    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c1.get_handle()]);

    let target = with_selective(&algebra, |kernel| {
        kernel.complete(
            &mut store,
            &source_cover,
            &mut ExactPlannerBlocks::default(),
        )
    })
    .unwrap();

    assert_eq!(cover_ids(&target), vec![data(&expected)]);
    assert_eq!(
        algebra.target_attempts.lock().unwrap().as_slice(),
        &[nested_pair, root_pair]
    );
    assert!(algebra.derive_attempts.lock().unwrap().is_empty());
    assert_ne!(
        SelectiveMapping::pair(unavailable.0, unavailable.1),
        SelectiveMapping::pair(available_low, available_high),
    );
}

#[test]
fn capacity_source_node_falls_back_to_an_existing_lower_target_cover() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let e = archive([(5, 6)]);
    let d = join_test_sources(&a, &b);
    let c = join_test_sources(&d, &e);
    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    inner.put::<TestSourceBlob, _>(c.clone()).unwrap();
    for (low, high, result) in [(&a, &b, &d), (&d, &e, &c)] {
        inner
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                kernel().source_collection().handle(),
                data(low),
                data(high),
                data(result),
            )))
            .unwrap();
    }
    for source in [&a, &b, &e] {
        publish_derive(&mut inner, source);
    }
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
        kernel.complete(
            &mut store,
            &source_cover,
            &mut ExactPlannerBlocks::default(),
        )
    })
    .unwrap();

    assert_eq!(
        algebra.derive_attempts.lock().unwrap().as_slice(),
        &[data(&c)]
    );
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert!(algebra.target_attempts.lock().unwrap().is_empty());
    assert_eq!(
        joined_cover(&mut store.inner, &target).bytes,
        derive(&c).unwrap().bytes
    );
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn capacity_on_one_corresponding_source_node_does_not_skip_another() {
    let a = archive([(1, 11)]);
    let b = archive([(2, 12)]);
    let e = archive([(3, 13)]);
    let d = join_test_sources(&a, &b);
    let c = join_test_sources(&d, &e);

    let f = archive([(4, 14)]);
    let g = archive([(5, 15)]);
    let h = archive([(6, 16)]);
    let i = join_test_sources(&f, &g);
    let j = join_test_sources(&i, &h);

    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    for root in [&c, &j] {
        inner.put::<TestSourceBlob, _>(root.clone()).unwrap();
    }
    for (low, high, result) in [(&a, &b, &d), (&d, &e, &c), (&f, &g, &i), (&i, &h, &j)] {
        inner
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                kernel().source_collection().handle(),
                data(low),
                data(high),
                data(result),
            )))
            .unwrap();
    }
    for source in [&a, &b, &e, &f, &g, &h] {
        publish_derive(&mut inner, source);
    }

    let mut roots = [c, j];
    roots.sort_unstable_by_key(data);
    let blocked = data(&roots[0]);
    let successful = data(&roots[1]);
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([blocked]),
        ..SelectiveMapping::default()
    };
    let source_cover = Cover::from_members(
        kernel().source_collection(),
        roots.iter().map(Blob::get_handle),
    );
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(
        algebra.derive_attempts.lock().unwrap().as_slice(),
        &[blocked, successful]
    );
    assert!(derived_inputs(&mut store.inner).contains(&successful));
    assert!(!derived_inputs(&mut store.inner).contains(&blocked));
}

#[test]
fn successful_corresponding_source_node_is_visible_before_later_capacity_fallback() {
    let a = archive([(1, 21)]);
    let b = archive([(2, 22)]);
    let e = archive([(3, 23)]);
    let d = join_test_sources(&a, &b);
    let c = join_test_sources(&d, &e);

    let f = archive([(4, 24)]);
    let g = archive([(5, 25)]);
    let h = archive([(6, 26)]);
    let i = join_test_sources(&f, &g);
    let j = join_test_sources(&i, &h);

    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    for root in [&c, &j] {
        inner.put::<TestSourceBlob, _>(root.clone()).unwrap();
    }
    for (low, high, result) in [(&a, &b, &d), (&d, &e, &c), (&f, &g, &i), (&i, &h, &j)] {
        inner
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                kernel().source_collection().handle(),
                data(low),
                data(high),
                data(result),
            )))
            .unwrap();
    }

    let (successful, blocked, blocked_leaves) = if data(&c) < data(&j) {
        (&c, &j, [&f, &g, &h])
    } else {
        (&j, &c, [&a, &b, &e])
    };
    for source in blocked_leaves {
        publish_derive(&mut inner, source);
    }
    let successful = data(successful);
    let blocked = data(blocked);
    let algebra = SelectiveMapping {
        capacity_derives: BTreeSet::from([blocked]),
        ..SelectiveMapping::default()
    };
    let source_cover = Cover::from_data(kernel().source_collection(), [successful, blocked]);
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };

    with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(
        algebra.derive_attempts.lock().unwrap().as_slice(),
        &[successful, blocked]
    );
    let stored = derived_inputs(&mut store.inner);
    assert!(stored.contains(&successful));
    assert!(!stored.contains(&blocked));
}

#[test]
fn ensure_derives_source_children_without_constructing_the_corresponding_source_node() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let c = join_test_sources(&a, &b);
    let fa = derive(&a).unwrap();
    let fb = derive(&b).unwrap();
    let expected = join_test_targets(&fa, &fb).unwrap();
    let pair = SelectiveMapping::pair(&fa, &fb);
    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            data(&a),
            data(&b),
            data(&c),
        )))
        .unwrap();
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(cover_ids(&target), vec![data(&expected)]);
    let mut actual_derives = algebra.derive_attempts.lock().unwrap().clone();
    actual_derives.sort_unstable();
    let mut expected_derives = vec![data(&a), data(&b)];
    expected_derives.sort_unstable();
    assert_eq!(actual_derives, expected_derives);
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert_eq!(algebra.target_attempts.lock().unwrap().as_slice(), &[pair]);
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn ensure_derives_only_the_target_child_not_already_represented() {
    let a = archive([(1, 3)]);
    let b = archive([(2, 4)]);
    let c = join_test_sources(&a, &b);
    let fa = derive(&a).unwrap();
    let fb = derive(&b).unwrap();
    let expected = join_test_targets(&fa, &fb).unwrap();
    let pair = SelectiveMapping::pair(&fa, &fb);
    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
    inner.put::<TestSourceBlob, _>(a.clone()).unwrap();
    inner.put::<TestSourceBlob, _>(b.clone()).unwrap();
    inner
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            kernel().source_collection().handle(),
            data(&a),
            data(&b),
            data(&c),
        )))
        .unwrap();
    publish_derive(&mut inner, &a);
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    let source_cover = Cover::from_members(kernel().source_collection(), [c.get_handle()]);

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(cover_ids(&target), vec![data(&expected)]);
    assert_eq!(
        algebra.derive_attempts.lock().unwrap().as_slice(),
        &[data(&b)]
    );
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert_eq!(algebra.target_attempts.lock().unwrap().as_slice(), &[pair]);
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
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
        kernel.complete(
            &mut store,
            &source_cover,
            &mut ExactPlannerBlocks::default(),
        )
    })
    .unwrap();

    assert_eq!(
        cover_ids(&target),
        vec![data(
            &join_test_targets(&derive(&a).unwrap(), &derive(&b).unwrap()).unwrap()
        )]
    );
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

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(target.members().collect::<Vec<_>>(), vec![fc.get_handle()]);
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

    let attached = kernel().attach(&mut store, &source_cover).unwrap();

    assert_eq!(
        attached.members().collect::<Vec<_>>(),
        vec![target.get_handle()]
    );
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn relevant_warm_metadata_failure_does_not_trigger_remapping() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    inner.put::<TestSourceBlob, _>(source.clone()).unwrap();
    let target = publish_derive(&mut inner, &source);
    let algebra = SelectiveMapping::default();
    let mut store = CountingStore {
        inner,
        ..CountingStore::default()
    };
    store
        .metadata_failures
        .lock()
        .unwrap()
        .insert(target.get_handle().raw);
    let source_cover = Cover::from_members(kernel().source_collection(), [source.get_handle()]);

    assert!(matches!(
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)),
        Err(ExactDerivedCollectionError::Storage { .. })
    ));
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(algebra.derive_attempts.lock().unwrap().is_empty());
    assert!(algebra.target_attempts.lock().unwrap().is_empty());
}

#[test]
fn missing_optional_decomposition_inputs_fall_back_to_direct_construction() {
    let c = archive([(9, 9)]);
    let mut inner = MemoryRepo::default();
    register_kernel(&mut inner);
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

    let target =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();

    assert_eq!(target.len(), 1);
    let actual: Blob<TestTargetBlob> = store
        .snapshot()
        .unwrap()
        .get(target.members().next().unwrap())
        .unwrap();
    assert_eq!(actual.bytes, derive(&c).unwrap().bytes);
    assert!(algebra.derive_attempts.lock().unwrap().contains(&data(&c)));
    assert_eq!(store.missing_gets.load(Ordering::SeqCst), 0);
}

#[test]
fn stored_reverse_decomposition_supplies_a_cover_without_replay() {
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

    let target =
        with_selective(&algebra, |kernel| kernel.attach(&mut store, &source_cover)).unwrap();
    let expected: BTreeSet<_> = [&a, &b]
        .into_iter()
        .map(|source| data(&derive(source).unwrap()))
        .collect();
    assert_eq!(target.data_members().collect::<BTreeSet<_>>(), expected);
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(algebra.source_attempts.lock().unwrap().is_empty());
    assert!(algebra.derive_attempts.lock().unwrap().is_empty());
    assert!(algebra.target_attempts.lock().unwrap().is_empty());
}

#[test]
fn algebra_produced_cover_composes_without_an_intermediate_commit() {
    let source = archive([(1, 3)]);
    let mut store = MemoryRepo::default();
    let commit = source_commit(&mut store, 1, &source);
    let source_cover = source_cover(&[commit]);

    let first = kernel().ensure(&mut store, &source_cover).unwrap();
    let first_member = first.members().next().unwrap();
    assert!(
        !collection_records(&mut store)
            .into_iter()
            .any(|record| matches!(
                record,
                CollectionRecord::Commit(commit)
                    if commit.collection() == kernel().target_collection().handle()
                        && commit.data() == Handle::<TestTargetBlob>::to_hash(first_member)
            )),
        "the first algebra result must remain unsigned equation evidence",
    );

    let second_exact = register_second_kernel(&mut store);
    let second = second_exact.ensure(&mut store, &first).unwrap();
    assert_eq!(second.len(), 1);
    let mut expected = derive(&source).unwrap().bytes.as_ref().to_vec();
    expected.push(0xB6);
    let actual: Blob<SecondTestTargetBlob> = store
        .snapshot()
        .unwrap()
        .get(second.members().next().unwrap())
        .unwrap();
    assert_eq!(actual.bytes.as_ref(), expected.as_slice());
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

    let cover = kernel().ensure(&mut store, &source_cover).unwrap();
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

    let cover =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();
    assert_eq!(cover.len(), 1);
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
    assert_eq!(algebra.target_attempts.lock().unwrap().len(), 1);
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

    with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();
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
        1,
        "each successful mapping is executed exactly once across replanning",
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
            with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)),
            Err(ExactDerivedCollectionError::UnrepresentableCover { ref blocked, ref missing })
                if blocked.len() == 1 && missing.len() == 1
        ));
        assert_eq!((store.puts, store.inserts), (0, 0));
    }
}

#[test]
fn mixed_terminal_capacity_publishes_the_successful_sibling() {
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
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)),
        Err(ExactDerivedCollectionError::UnrepresentableCover { .. })
    ));
    assert_eq!((store.puts, store.inserts), (2, 1));
    assert_eq!(derived_inputs(&mut store.inner), vec![data(successful)]);
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
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)),
        Err(ExactDerivedCollectionError::Derive { input, .. }) if input == data(&source)
    ));
    assert_eq!((store.puts, store.inserts), (0, 0));
}

struct GuardSnapshot {
    inner: MemoryRepoSnapshot,
    live: Arc<AtomicUsize>,
}

impl Clone for GuardSnapshot {
    fn clone(&self) -> Self {
        self.live.fetch_add(1, Ordering::SeqCst);
        Self {
            inner: self.inner.clone(),
            live: Arc::clone(&self.live),
        }
    }
}

impl Drop for GuardSnapshot {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

impl StoreSnapshot for GuardSnapshot {
    fn changes_since(&self, previous: &Self) -> StoreChanges {
        self.inner.changes_since(&previous.inner)
    }
}

impl BlobStoreMeta for GuardSnapshot {
    type MetaError = <MemoryRepoSnapshot as BlobStoreMeta>::MetaError;

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

impl BlobStoreGet for GuardSnapshot {
    type GetError<E: Error + Send + Sync + 'static> =
        <MemoryRepoSnapshot as BlobStoreGet>::GetError<E>;

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

impl BlobStoreList for GuardSnapshot {
    type Iter<'a>
        = <MemoryRepoSnapshot as BlobStoreList>::Iter<'a>
    where
        Self: 'a;
    type Err = <MemoryRepoSnapshot as BlobStoreList>::Err;

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

impl CollectionRead for GuardSnapshot {
    type RecordsError = <MemoryRepoSnapshot as CollectionRead>::RecordsError;
    type RecordIter<'a>
        = <MemoryRepoSnapshot as CollectionRead>::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
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
    fn assert_no_snapshot(&self) {
        assert_eq!(
            self.live.load(Ordering::SeqCst),
            0,
            "write while snapshot is live"
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
        self.assert_no_snapshot();
        let blob = item.to_blob();
        self.events
            .push(WriteEvent::Put(Handle::<S>::to_hash(blob.get_handle())));
        self.inner.put(blob)
    }
}

impl SnapshotSource for GuardStore {
    type Snapshot = GuardSnapshot;
    type SnapshotError = <MemoryRepo as SnapshotSource>::SnapshotError;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        let inner = self.inner.snapshot()?;
        self.live.fetch_add(1, Ordering::SeqCst);
        Ok(GuardSnapshot {
            inner,
            live: Arc::clone(&self.live),
        })
    }
}

impl CollectionStore for GuardStore {
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.assert_no_snapshot();
        self.events.push(WriteEvent::Insert(record));
        self.inner.insert(record)
    }
}

#[test]
fn snapshot_is_dropped_before_first_write() {
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
    kernel().ensure(&mut store, &source_cover).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn ensure_stores_source_before_target_and_derive() {
    let source = archive([(1, 3)]);
    let mut inner = MemoryRepo::default();
    let commit = source_commit(&mut inner, 1, &source);
    let mut store = GuardStore {
        inner,
        live: Arc::new(AtomicUsize::new(0)),
        events: Vec::new(),
    };

    kernel()
        .ensure(&mut store, &source_cover(&[commit]))
        .unwrap();

    let (insert, claim) = store
        .events
        .iter()
        .enumerate()
        .find_map(|(position, event)| match event {
            WriteEvent::Insert(CollectionRecord::Derive(claim)) => Some((position, *claim)),
            _ => None,
        })
        .expect("ensure publishes one DERIVE");
    let source_put = store
        .events
        .iter()
        .position(|event| matches!(event, WriteEvent::Put(data) if *data == claim.input()))
        .expect("ensure stores the selected source");
    let target_put = store
        .events
        .iter()
        .position(|event| matches!(event, WriteEvent::Put(data) if *data == claim.output()))
        .expect("ensure stores the mapped target");

    assert!(source_put < target_put);
    assert!(target_put < insert);
}

#[test]
fn later_derive_put_failure_preserves_the_complete_published_prefix() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let mut inner = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut inner, index as u8 + 1, source))
        .collect();
    let mut store = RejectPutStore {
        inner,
        puts: 0,
        // source + output for the first map, source + rejected output for the
        // second map.
        reject_at: 4,
    };

    assert!(matches!(
        kernel().ensure(&mut store, &source_cover(&commits)),
        Err(ExactDerivedCollectionError::Storage { .. })
    ));
    let derives = derived_inputs(&mut store.inner);
    assert_eq!(derives.len(), 1);
    assert!(sources.iter().map(data).any(|input| input == derives[0]));
}

fn target_merge_records(store: &mut MemoryRepo) -> Vec<CollectionMerge> {
    collection_records(store)
        .into_iter()
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

fn joined_cover(store: &mut MemoryRepo, cover: &Cover<TestTargetBlob>) -> Blob<TestTargetBlob> {
    let snapshot = store.snapshot().unwrap();
    let mut members = cover.members();
    let mut joined: Blob<TestTargetBlob> = members
        .next()
        .map(|handle| snapshot.get(handle).unwrap())
        .expect("nonempty cover has a target member");
    for handle in members {
        let member: Blob<TestTargetBlob> = snapshot.get(handle).unwrap();
        joined = join_test_targets(&joined, &member).unwrap();
    }
    joined
}

fn cover_ids(cover: &Cover<TestTargetBlob>) -> Vec<CollectionData> {
    cover
        .members()
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

    let cover = kernel().ensure(&mut store, &source_cover).unwrap();
    let mut tiers = BTreeSet::new();
    {
        let snapshot = store.snapshot().unwrap();
        for handle in cover.members() {
            let blob: Blob<TestTargetBlob> = snapshot.get(handle).unwrap();
            assert!(tiers.insert(blob.bytes.len().max(1).ilog2()));
        }
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
        joined_cover(&mut store, &cover).bytes,
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

    let cover =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();
    assert_eq!(cover.len(), 2);
    assert_eq!(
        algebra.target_attempts.lock().unwrap().as_slice(),
        &[first_pair, second_pair],
        "each target join is executed exactly once",
    );
    let merges = target_merge_records(&mut store.inner);
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].inputs(), second_pair);
}

#[test]
fn fatal_late_target_join_preserves_the_successful_prefix() {
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
            kernel.ensure(&mut store, &source_cover)
        }),
        Err(ExactDerivedCollectionError::Merge { low, high, .. })
            if (low, high) == fatal_pair
    ));
    assert_eq!(
        algebra.target_attempts.lock().unwrap().as_slice(),
        &[first_pair, fatal_pair],
    );
    assert_eq!((store.puts, store.inserts), (1, 1));
    let merges = target_merge_records(&mut store.inner);
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].inputs(), first_pair);
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

    let first =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!((store.puts, store.inserts), (0, 0));
    let second =
        with_selective(&algebra, |kernel| kernel.ensure(&mut store, &source_cover)).unwrap();
    assert_eq!(cover_ids(&first), cover_ids(&second));
    assert_eq!((store.puts, store.inserts), (0, 0));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

#[test]
fn compaction_substitutes_new_resident_uppers_through_an_old_stored_equation() {
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
    let before = kernel().attach(&mut store, &source_cover).unwrap();
    assert_eq!(before.len(), 3);
    let after = kernel().ensure(&mut store, &source_cover).unwrap();
    let mut tiers = BTreeSet::new();
    let tier_stable = {
        let snapshot = store.snapshot().unwrap();
        after.members().all(|handle| {
            let blob: Blob<TestTargetBlob> = snapshot.get(handle).unwrap();
            tiers.insert(blob.bytes.len().max(1).ilog2())
        })
    };
    assert!(tier_stable);
    assert_eq!(after.len(), 2);
    assert_eq!(joined_cover(&mut store, &after).bytes, old_upper.bytes);
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

    let first = kernel().ensure(&mut first_store, &first_cover).unwrap();
    let second = kernel().ensure(&mut second_store, &second_cover).unwrap();
    assert_eq!(cover_ids(&first), cover_ids(&second));
    assert_eq!(
        target_merge_records(&mut first_store),
        target_merge_records(&mut second_store)
    );

    let records_before = collection_records(&mut first_store);
    let repeated = kernel().ensure(&mut first_store, &first_cover).unwrap();
    let records_after = collection_records(&mut first_store);
    assert_eq!(cover_ids(&first), cover_ids(&repeated));
    assert_eq!(records_before, records_after);
}

#[test]
fn compaction_drops_readers_and_puts_results_without_republishing_the_descriptor() {
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

    kernel().ensure(&mut store, &source_cover).unwrap();
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let descriptor_data = Handle::<SimpleArchive>::to_hash(kernel().target_collection().handle());
    assert!(!store
        .events
        .iter()
        .any(|event| matches!(event, WriteEvent::Put(data) if *data == descriptor_data)));
    let merges: Vec<_> = store
        .events
        .iter()
        .enumerate()
        .filter_map(|(position, event)| match event {
            WriteEvent::Insert(CollectionRecord::Merge(claim)) => Some((position, claim.result())),
            _ => None,
        })
        .collect();
    assert!(!merges.is_empty());
    for (position, result) in merges {
        assert!(store.events[..position]
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
    reject_at: usize,
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
        if self.puts == self.reject_at {
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

impl SnapshotSource for DropMergeStore {
    type Snapshot = MemoryRepoSnapshot;
    type SnapshotError = <MemoryRepo as SnapshotSource>::SnapshotError;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        self.inner.snapshot()
    }
}

impl CollectionStore for DropMergeStore {
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        if matches!(record, CollectionRecord::Merge(_)) {
            Ok(())
        } else {
            self.inner.insert(record)
        }
    }
}

struct DropDeriveStore {
    inner: MemoryRepo,
    dropped_derives: usize,
}

impl BlobStorePut for DropDeriveStore {
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

impl SnapshotSource for DropDeriveStore {
    type Snapshot = MemoryRepoSnapshot;
    type SnapshotError = <MemoryRepo as SnapshotSource>::SnapshotError;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        self.inner.snapshot()
    }
}

impl CollectionStore for DropDeriveStore {
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        if matches!(record, CollectionRecord::Derive(_)) {
            self.dropped_derives += 1;
            Ok(())
        } else {
            self.inner.insert(record)
        }
    }
}

impl SnapshotSource for RejectPutStore {
    type Snapshot = MemoryRepoSnapshot;
    type SnapshotError = <MemoryRepo as SnapshotSource>::SnapshotError;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        self.inner.snapshot()
    }
}

impl CollectionStore for RejectPutStore {
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;

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
            kernel.ensure(&mut join_store, &join_cover)
        }),
        Err(ExactDerivedCollectionError::Merge { .. })
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
    let mut put_store = RejectPutStore {
        inner,
        puts: 0,
        reject_at: 1,
    };
    assert!(matches!(
        kernel().ensure(&mut put_store, &put_cover),
        Err(ExactDerivedCollectionError::Storage { .. })
    ));
    assert!(target_merge_records(&mut put_store.inner).is_empty());
}

#[test]
fn later_target_put_failure_preserves_the_complete_published_prefix() {
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
    let mut store = RejectPutStore {
        inner,
        puts: 0,
        reject_at: 2,
    };

    assert!(matches!(
        kernel().ensure(&mut store, &source_cover(&commits)),
        Err(ExactDerivedCollectionError::Storage { .. })
    ));
    assert_eq!(target_merge_records(&mut store.inner).len(), 1);
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
        kernel().ensure(&mut store, &source_cover),
        Err(ExactDerivedCollectionError::Stalled { cover }) if cover.len() == 2
    ));
    assert!(target_merge_records(&mut store.inner).is_empty());
}

#[test]
fn discarded_derive_insert_stalls_after_the_first_unobserved_publication() {
    let sources = [archive([(1, 3)]), archive([(2, 4)])];
    let mut inner = MemoryRepo::default();
    let commits: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, source)| source_commit(&mut inner, index as u8 + 1, source))
        .collect();
    let source_cover = source_cover(&commits);
    let mut store = DropDeriveStore {
        inner,
        dropped_derives: 0,
    };

    assert!(matches!(
        kernel().ensure(&mut store, &source_cover),
        Err(ExactDerivedCollectionError::Stalled { cover }) if cover.is_empty()
    ));
    assert_eq!(store.dropped_derives, 1);
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
    match kernel().attach(&mut store, &source_cover) {
        Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {}
        Err(error) => panic!("unexpected missing-output error: {error:?}"),
        Ok(_) => panic!("missing output was incorrectly admitted"),
    }
    assert_eq!(kernel().ensure(&mut store, &source_cover).unwrap().len(), 1,);
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

    let cover = kernel().ensure(&mut store, &source_cover).unwrap();
    let actual: Blob<TestTargetBlob> = store
        .snapshot()
        .unwrap()
        .get(cover.members().next().unwrap())
        .unwrap();
    assert_eq!(actual.bytes, derive(&a).unwrap().bytes);
    let derives: Vec<_> = collection_records(&mut store)
        .into_iter()
        .filter_map(|record| match record {
            CollectionRecord::Derive(claim) => Some(claim.input()),
            _ => None,
        })
        .collect();
    assert_eq!(derives, vec![data(&a)]);
}

#[test]
fn typed_lifecycle_rejects_a_lying_source_descriptor() {
    let lying_source = descriptor::naming::<TestTargetBlob>("source", test_policy());
    let result = MemoryRepo::default().register_collection::<TestSourceBlob>(lying_source);
    assert!(matches!(
        result,
        Err(crate::collection::CollectionRegistrationError::WrongType(_))
    ));
}

#[test]
fn identity_descriptor_pair_is_rejected() {
    let mut store = MemoryRepo::default();
    let collection = store
        .register_collection::<TestSourceBlob>(source_root())
        .unwrap();
    let result = ExactDerivedCollection::<IdentityMapping>::new(collection, collection);
    assert!(matches!(
        result,
        Err(ExactDerivedCollectionError::Resolution(reason))
            if reason.contains("distinct source and target descriptors")
    ));
}
