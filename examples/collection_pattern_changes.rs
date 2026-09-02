//! Incrementally query a growing collection through exact Succinct full and
//! changed snapshots.
//!
//! Run with: `cargo run --example collection_pattern_changes`

use std::error::Error;
use std::io;

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::encodings::succinctarchive::{OrderedUniverse, UnionArchive};
use triblespace::core::collection::succinctarchive_union::{
    RawToRank9AcceleratedMapping, SimpleToSuccinctMapping, SuccinctArchiveCollection,
    SuccinctArchiveSnapshot, SuccinctArchiveSnapshotAdvance,
};
use triblespace::core::collection::{
    AdmissionPolicy, Collection, CollectionPolicy, CollectionStoreExt,
};
use triblespace::core::examples::literature;
use triblespace::core::repo::memoryrepo::MemoryRepo;
use triblespace::prelude::*;

fn rebuild(
    full: &UnionArchive<OrderedUniverse>,
    consume: &mut impl FnMut(&str) -> Result<(), Box<dyn Error>>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut titles = Vec::new();
    for title in find!(
        title: String,
        pattern!(full, [
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
    Ok(titles)
}

fn changes(
    full: &UnionArchive<OrderedUniverse>,
    changed: &UnionArchive<OrderedUniverse>,
    consume: &mut impl FnMut(&str) -> Result<(), Box<dyn Error>>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut titles = Vec::new();
    for title in find!(
        title: String,
        pattern_changes!(full, changed, [
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
    Ok(titles)
}

// ANCHOR: collection_pattern_changes_observe
fn observe(
    store: &mut MemoryRepo,
    collection: Collection<SimpleArchive>,
    succinct: &SuccinctArchiveCollection,
    checkpoint: &mut Option<SuccinctArchiveSnapshot>,
    mut consume: impl FnMut(&str) -> Result<(), Box<dyn Error>>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let snapshot = store.snapshot()?;
    let current = collection.admitted(&snapshot)?;
    let advance = match checkpoint.as_ref() {
        Some(previous) => succinct.advance(store, previous, &current)?,
        None => SuccinctArchiveSnapshotAdvance::Reset {
            next: succinct.ensure(store, &current)?,
        },
    };

    let (next, titles) = match advance {
        SuccinctArchiveSnapshotAdvance::Unchanged => return Ok(Vec::new()),
        SuccinctArchiveSnapshotAdvance::Advanced { next, changed } => {
            let titles = changes(next.view(), changed.view(), &mut consume)?;
            (next, titles)
        }
        SuccinctArchiveSnapshotAdvance::Reset { next } => {
            let titles = rebuild(next.view(), &mut consume)?;
            (next, titles)
        }
    };

    // Adopt only after the complete fold succeeds. A failed consumer retries
    // the same exact Succinct delta, so external effects must be transactional
    // or idempotent when exactly-once delivery matters.
    *checkpoint = Some(next);
    Ok(titles)
}
// ANCHOR_END: collection_pattern_changes_observe

fn main() -> Result<(), Box<dyn Error>> {
    let signing_key = SigningKey::generate(&mut OsRng);
    let authority = signing_key.verifying_key();
    let name = "incremental-literature";
    let policy = CollectionPolicy::new(
        AdmissionPolicy::direct(authority),
        AdmissionPolicy::direct(authority),
    );
    let mut store = MemoryRepo::default();
    let collection = store.collection(name, policy.clone())?;

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

    let raw = store.derive(collection, SimpleToSuccinctMapping, policy.clone())?;
    let accelerated = store.derive(raw, RawToRank9AcceleratedMapping, policy)?;
    let succinct = SuccinctArchiveCollection::new(collection, raw, accelerated);
    let mut checkpoint = None;

    let first = observe(&mut store, collection, &succinct, &mut checkpoint, |_| {
        Ok(())
    })?;
    assert_eq!(first, ["Dune"]);

    store.commit(
        collection,
        &signing_key,
        entity! {
            literature::title: "Dune Messiah",
            literature::author: &herbert,
        },
    )?;

    let before_failure = checkpoint
        .as_ref()
        .map(|snapshot| snapshot.source().clone());
    let failed = observe(&mut store, collection, &succinct, &mut checkpoint, |_| {
        Err(io::Error::other("simulated consumer failure").into())
    });
    assert!(failed.is_err());
    assert_eq!(
        checkpoint
            .as_ref()
            .map(|snapshot| snapshot.source().clone()),
        before_failure,
    );

    let retry = observe(&mut store, collection, &succinct, &mut checkpoint, |_| {
        Ok(())
    })?;
    assert_eq!(retry, ["Dune Messiah"]);

    let unchanged = observe(&mut store, collection, &succinct, &mut checkpoint, |_| {
        Ok(())
    })?;
    assert!(unchanged.is_empty());

    println!("incremental titles: {first:?}, then {retry:?}");
    Ok(())
}
