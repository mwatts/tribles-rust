//! Publish intrinsic entities to a native collection, then query its exact
//! SuccinctArchive projection without a branch, checkout, hook, or manifest.
//!
//! Run with: `cargo run --example native_succinct_collection`

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::collection::succinctarchive_union::{
    RawToRank9AcceleratedMapping, SimpleToSuccinctMapping, SuccinctArchiveCollection,
};
use triblespace::core::collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt};
use triblespace::core::examples::literature;
use triblespace::prelude::*;

fn main() {
    let tmp = tempfile::tempdir().expect("tmp dir");
    let path = tmp.path().join("native-succinct.pile");
    std::fs::File::create(&path).expect("create pile file");

    let mut pile = Pile::open(&path).expect("open pile");
    pile.refresh().expect("load pile");

    // A root collection is the handle of a self-contained descriptor. Its
    // independent READ and WRITE policies participate in that content identity.
    let name = "literature";
    let signing_key = SigningKey::generate(&mut OsRng);
    let authority = signing_key.verifying_key();
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(authority),
        AdmissionPolicy::direct(authority),
    );
    let collection = pile
        .collection(name, policy.clone())
        .expect("register source collection");

    // Each fragment is one independent signed collection member. Omitting an
    // explicit entity id makes every person intrinsic to their facts.
    for name in ["Ada", "Grace", "Barbara"] {
        pile.commit(
            collection,
            &signing_key,
            entity! { literature::firstname: name },
        )
        .expect("publish person");
    }

    // Freeze one coherent store observation, then discover its exact admitted
    // target frontier without reading the commits' data or metadata blobs.
    let snapshot = pile.snapshot().expect("freeze pile snapshot");
    let cover = collection
        .admitted(&snapshot)
        .expect("discover exact cover");
    assert_eq!(cover.len(), 3);

    // Build any missing canonical raw Succinct shards and their exact Rank9
    // fibers, then query the admitted physical cover directly.
    let raw = pile
        .derive(collection, SimpleToSuccinctMapping, policy.clone())
        .expect("register raw Succinct projection");
    let accelerated = pile
        .derive(raw, RawToRank9AcceleratedMapping, policy)
        .expect("register Rank9-accelerated projection");
    let succinct = SuccinctArchiveCollection::new(collection, raw, accelerated);
    let archive = succinct
        .ensure(&mut pile, &cover)
        .expect("ensure exact Succinct projection");
    let mut names: Vec<String> = find!(
        name: Inline<_>,
        pattern!(archive.view(), [{ _?person @ literature::firstname: ?name }])
    )
    .map(|name| name.try_from_inline::<String>().expect("short string"))
    .collect();
    names.sort();

    println!("queried exact Succinct cover: {names:?}");
    assert_eq!(names, ["Ada", "Barbara", "Grace"]);

    pile.close().expect("close pile");
}
