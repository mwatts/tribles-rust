//! Incrementally query a growing collection through a Succinct full view and
//! a SimpleArchive support delta.
//!
//! Run with: `cargo run --example collection_pattern_changes`

use std::error::Error;
use std::io;

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::collection::succinctarchive_union::{
    SuccinctArchiveCollection, SuccinctArchiveView,
};
use triblespace::core::collection::{
    exact_ticket_additions, reach, simplearchive_union, CollectionCommit, CollectionHandle,
    CollectionStoreExt, SimpleArchiveCollection,
};
use triblespace::core::examples::literature;
use triblespace::core::repo::memoryrepo::MemoryRepo;
use triblespace::prelude::*;

// ANCHOR: collection_pattern_changes_observe
fn observe(
    store: &mut MemoryRepo,
    collection: CollectionHandle,
    simple: &SimpleArchiveCollection,
    full_view: &mut SuccinctArchiveView,
    checkpoint: &mut Vec<CollectionCommit>,
    mut consume: impl FnMut(&str) -> Result<(), Box<dyn Error>>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let current = store.ticket(collection, &[])?;
    let added = exact_ticket_additions(collection, checkpoint, current.commits())?;

    // The full view retains its already-admitted immutable Succinct shards and
    // admits only new support. The small SimpleArchive delta stays independent
    // because it drives the change query and advances only after consumption.
    let full = full_view.ensure(store, current.commits())?;
    let changed = simple.attach_exact(store, &added)?;

    let mut titles = Vec::new();
    for title in find!(
        title: String,
        pattern_changes!(&full, &changed, [
            { _?author @ literature::firstname: "Frank" },
            { _?book @
                literature::author: _?author,
                literature::title: ?title
            }
        ])
    ) {
        consume(&title)?;
        titles.push(title);
    }

    // Advance only after the complete fold succeeds. A failed consumer retries
    // the same support delta, so external effects must be transactional or
    // idempotent when exactly-once delivery matters.
    *checkpoint = current.commits().to_vec();
    Ok(titles)
}
// ANCHOR_END: collection_pattern_changes_observe

fn main() -> Result<(), Box<dyn Error>> {
    let signing_key = SigningKey::generate(&mut OsRng);
    let authority = signing_key.verifying_key();
    let name = "incremental-literature";
    let source_reach = reach::private();
    let simple = SimpleArchiveCollection::new(name, authority, source_reach.clone());
    let mut store = MemoryRepo::default();
    let collection = store.collection(simplearchive_union::descriptor(
        name,
        authority,
        source_reach.clone(),
    ))?;

    let author = entity! {
        literature::firstname: "Frank",
        literature::lastname: "Herbert",
    };
    let herbert = author.root().expect("intrinsic author id");
    store.commit(collection, &signing_key, author)?;
    store.commit(
        collection,
        &signing_key,
        entity! {
            literature::title: "Dune",
            literature::author: &herbert,
        },
    )?;

    let succinct =
        SuccinctArchiveCollection::new(name, authority, source_reach, authority, reach::private());
    let mut full_view = succinct.exact_view();
    let mut checkpoint = Vec::new();

    let first = observe(
        &mut store,
        collection,
        &simple,
        &mut full_view,
        &mut checkpoint,
        |_| Ok(()),
    )?;
    assert_eq!(first, ["Dune"]);

    store.commit(
        collection,
        &signing_key,
        entity! {
            literature::title: "Dune Messiah",
            literature::author: &herbert,
        },
    )?;

    let before_failure = checkpoint.clone();
    let failed = observe(
        &mut store,
        collection,
        &simple,
        &mut full_view,
        &mut checkpoint,
        |_| Err(io::Error::other("simulated consumer failure").into()),
    );
    assert!(failed.is_err());
    assert_eq!(checkpoint, before_failure);

    let retry = observe(
        &mut store,
        collection,
        &simple,
        &mut full_view,
        &mut checkpoint,
        |_| Ok(()),
    )?;
    assert_eq!(retry, ["Dune Messiah"]);

    let unchanged = observe(
        &mut store,
        collection,
        &simple,
        &mut full_view,
        &mut checkpoint,
        |_| Ok(()),
    )?;
    assert!(unchanged.is_empty());

    println!("incremental titles: {first:?}, then {retry:?}");
    Ok(())
}
