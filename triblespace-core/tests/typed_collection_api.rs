use ed25519_dalek::SigningKey;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::{OrderedUniverse, UnionArchive};
use triblespace_core::blob::{Blob, IntoBlob};
use triblespace_core::collection::exact_derived::ExactDerivedCollection;
use triblespace_core::collection::succinctarchive_union;
use triblespace_core::collection::succinctarchive_union::SimpleToSuccinctMapping;
use triblespace_core::collection::{
    AdmissionPolicy, CollectionPolicy, CollectionStoreExt, TryFromCover,
};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::SnapshotSource;
use triblespace_core::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

fn one_fact(seed: u8) -> TribleSet {
    let mut row = [seed; TRIBLE_LEN];
    row[16..32].fill(seed.wrapping_add(1));
    row[32..].fill(seed.wrapping_add(2));
    let mut facts = TribleSet::new();
    facts.insert(&Trible::force_raw(row).unwrap());
    facts
}

#[test]
fn simplearchive_collection_round_trips_typed_views() {
    let authority = SigningKey::from_bytes(&[41; 32]);
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(authority.verifying_key()),
        AdmissionPolicy::direct(authority.verifying_key()),
    );
    let expected = one_fact(7);
    let expected_member = expected.clone().to_blob().get_handle();
    let mut store = MemoryRepo::default();

    let collection = store.collection("typed-api", policy).unwrap();

    let commit = store
        .commit(collection, &authority, Fragment::from(expected.clone()))
        .unwrap();
    assert_eq!(
        Handle::<SimpleArchive>::from_hash(commit.data()),
        expected_member
    );

    let snapshot = store.snapshot().unwrap();
    let cover = collection.admitted(&snapshot).unwrap();
    assert_eq!(cover.collection(), collection);
    assert_eq!(cover.members().collect::<Vec<_>>(), vec![expected_member]);

    let materialized: TribleSet = collection.read(&snapshot).unwrap();
    assert_eq!(materialized, expected);
}

#[test]
fn succinct_cover_materializes_as_a_typed_union_archive() {
    let authority = SigningKey::from_bytes(&[42; 32]);
    let expected = one_fact(11);
    let source_blob: Blob<SimpleArchive> = expected.clone().to_blob();
    let raw = succinctarchive_union::derive_element(&source_blob).unwrap();
    let raw_handle = raw.get_handle();
    let mut store = MemoryRepo::default();

    let source_policy = CollectionPolicy::new(
        AdmissionPolicy::direct(authority.verifying_key()),
        AdmissionPolicy::direct(authority.verifying_key()),
    );
    let target_policy = source_policy.clone();
    let source = store.collection("typed-api-source", source_policy).unwrap();
    let target = store
        .derive(source, SimpleToSuccinctMapping, target_policy)
        .unwrap();

    store
        .commit(source, &authority, Fragment::from(expected.clone()))
        .unwrap();
    let snapshot = store.snapshot().unwrap();
    let source_cover = source.admitted(&snapshot).unwrap();
    let derived = ExactDerivedCollection::<SimpleToSuccinctMapping>::new(source, target).unwrap();
    let cover = derived.ensure(&mut store, &source_cover).unwrap();
    assert_eq!(cover.collection(), target);
    assert_eq!(cover.members().collect::<Vec<_>>(), vec![raw_handle]);

    let snapshot = store.snapshot().unwrap();
    let materialized = UnionArchive::<OrderedUniverse>::try_from_cover(&cover, &snapshot).unwrap();
    assert_eq!(materialized.segment_count(), 1);
    assert_eq!(materialized.iter().collect::<TribleSet>(), expected);
}
