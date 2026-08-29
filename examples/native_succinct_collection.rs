//! Publish intrinsic entities to a native collection, then query its exact
//! SuccinctArchive projection without a branch, checkout, hook, or manifest.
//!
//! Run with: `cargo run --example native_succinct_collection`

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::collection::succinctarchive_union::SuccinctArchiveCollection;
use triblespace::core::collection::{reach, simplearchive_union, CollectionStoreExt};
use triblespace::core::examples::literature;
use triblespace::prelude::*;

fn main() {
    let tmp = tempfile::tempdir().expect("tmp dir");
    let path = tmp.path().join("native-succinct.pile");
    std::fs::File::create(&path).expect("create pile file");

    let mut pile = Pile::open(&path).expect("open pile");
    pile.refresh().expect("load pile");

    // A root collection is the handle of a self-contained descriptor. Its
    // mandatory authority participates in that content identity.
    let name = "literature";
    let signing_key = SigningKey::generate(&mut OsRng);
    let authority = signing_key.verifying_key();
    let source_reach = reach::private();
    let collection = pile
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            name,
            authority,
            source_reach.clone(),
        ))
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

    // Freeze the exact admitted target frontier. cover() reads collection
    // records, but not these commits' data or metadata blobs.
    let cover = pile.cover(collection).expect("discover exact cover");
    assert_eq!(cover.len(), 3);

    // Build any missing canonical raw Succinct shards and their exact Rank9
    // fibers, then query the admitted physical cover directly.
    let succinct =
        SuccinctArchiveCollection::new(name, authority, source_reach, authority, reach::private());
    let archive = succinct
        .ensure_exact(&mut pile, &cover)
        .expect("ensure exact Succinct projection");
    let mut names: Vec<String> = find!(
        name: Inline<_>,
        pattern!(&archive, [{ _?person @ literature::firstname: ?name }])
    )
    .map(|name| name.try_from_inline::<String>().expect("short string"))
    .collect();
    names.sort();

    println!("queried exact Succinct cover: {names:?}");
    assert_eq!(names, ["Ada", "Barbara", "Grace"]);

    pile.close().expect("close pile");
}
