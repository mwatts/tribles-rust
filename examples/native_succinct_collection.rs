//! Publish intrinsic entities to a native collection, then query its exact
//! SuccinctArchive projection without a branch, checkout, hook, or manifest.
//!
//! Run with: `cargo run --example native_succinct_collection`

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::authority::{self, AuthorityGrant, AuthorityMode, ACTION_WRITE};
use triblespace::core::collection::succinctarchive_union::SuccinctArchiveCollection;
use triblespace::core::examples::literature;
use triblespace::prelude::*;
use triblespace_core::collection::reach;

fn main() {
    let tmp = tempfile::tempdir().expect("tmp dir");
    let path = tmp.path().join("native-succinct.pile");
    std::fs::File::create(&path).expect("create pile file");

    let mut pile = Pile::open(&path).expect("open pile");
    pile.refresh().expect("load pile");

    // A root collection is anchored by a name within a team. This example is
    // a team of one, so the signing key is also the team root: it says so
    // explicitly rather than letting the facade assume it.
    let name = CollectionName::new("literature").expect("legal collection name");
    let signing_key = SigningKey::generate(&mut OsRng);
    let team = signing_key.verifying_key();
    let mut collection = Collection::new(pile, &name, team, signing_key.clone(), reach::private());
    let target = collection.collection();
    authority::publish_grant(
        collection.storage_mut(),
        team,
        &signing_key,
        AuthorityGrant::root(
            signing_key.verifying_key(),
            target,
            ACTION_WRITE,
            AuthorityMode::Invoke,
        ),
    )
    .expect("authorize collection writer");

    // Each fragment is one independent signed collection member. Omitting an
    // explicit entity id makes every person intrinsic to their facts.
    for name in ["Ada", "Grace", "Barbara"] {
        collection
            .commit(entity! { literature::firstname: name })
            .expect("publish person");
    }

    // Freeze the exact authorized target frontier. ticket() reads the
    // authority grant blobs, but not these commits' data or metadata blobs.
    let ticket = collection.ticket().expect("discover exact ticket");
    assert_eq!(ticket.len(), 3);

    // Build any missing canonical raw Succinct shards and their exact Rank9
    // fibers, then query the admitted physical cover directly.
    let succinct =
        SuccinctArchiveCollection::new(name.clone(), team, reach::private(), reach::private());
    let archive = succinct
        .ensure_exact(collection.storage_mut(), &ticket)
        .expect("ensure exact Succinct projection");
    let mut names: Vec<String> = find!(
        name: Inline<_>,
        pattern!(&archive, [{ _?person @ literature::firstname: ?name }])
    )
    .map(|name| name.try_from_inline::<String>().expect("short string"))
    .collect();
    names.sort();

    println!("queried exact Succinct ticket: {names:?}");
    assert_eq!(names, ["Ada", "Barbara", "Grace"]);

    collection.into_storage().close().expect("close pile");
}
