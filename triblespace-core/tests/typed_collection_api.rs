use ed25519_dalek::SigningKey;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchiveBlob, UnionArchive,
};
use triblespace_core::blob::{Blob, IntoBlob};
use triblespace_core::collection::exact_derived::ExactDerivedCollection;
use triblespace_core::collection::succinctarchive_union::SimpleToSuccinctMapping;
use triblespace_core::collection::{reach, CollectionStoreExt};
use triblespace_core::collection::{simplearchive_union, succinctarchive_union};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::memoryrepo::MemoryRepo;
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
    let descriptor =
        simplearchive_union::descriptor("typed-api", authority.verifying_key(), reach::private());
    let descriptor_handle = descriptor.facts().clone().to_blob().get_handle();
    let expected = one_fact(7);
    let expected_member = expected.clone().to_blob().get_handle();
    let mut store = MemoryRepo::default();

    let collection = store.collection::<SimpleArchive>(descriptor).unwrap();
    assert_eq!(collection.handle(), descriptor_handle);

    let commit = store
        .commit(collection, &authority, Fragment::from(expected.clone()))
        .unwrap();
    assert_eq!(
        Handle::<SimpleArchive>::from_hash(commit.data()),
        expected_member
    );

    let cover = store.cover(collection).unwrap();
    assert_eq!(cover.collection(), collection);
    assert_eq!(cover.members().collect::<Vec<_>>(), vec![expected_member]);

    let snapshot = store.snapshot::<TribleSet, _>(collection).unwrap();
    assert_eq!(snapshot.cover(), &cover);
    assert_eq!(snapshot.facts(), &expected);

    let materialized = store.materialize::<TribleSet, _>(&cover).unwrap();
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

    let source_descriptor = simplearchive_union::descriptor(
        "typed-api-source",
        authority.verifying_key(),
        reach::private(),
    );
    let source = store
        .collection::<SimpleArchive>(source_descriptor.clone())
        .unwrap();
    let target_descriptor = succinctarchive_union::descriptor(
        source.handle(),
        authority.verifying_key(),
        reach::private(),
    );
    let target = store
        .collection::<SuccinctArchiveBlob>(target_descriptor.clone())
        .unwrap();

    store
        .commit(source, &authority, Fragment::from(expected.clone()))
        .unwrap();
    let source_cover = store.cover(source).unwrap();
    let derived = ExactDerivedCollection::<
        SimpleArchive,
        SuccinctArchiveBlob,
        SimpleToSuccinctMapping,
    >::new(source_descriptor, target_descriptor)
    .unwrap();
    let attachment = derived.ensure_exact(&mut store, &source_cover).unwrap();
    let cover = attachment.cover().clone();
    assert_eq!(cover.collection(), target);
    assert_eq!(cover.members().collect::<Vec<_>>(), vec![raw_handle]);

    let materialized = store
        .materialize::<UnionArchive<OrderedUniverse>, _>(&cover)
        .unwrap();
    assert_eq!(materialized.segment_count(), 1);
    assert_eq!(materialized.iter().collect::<TribleSet>(), expected);
}
