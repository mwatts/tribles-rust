//! End-to-end test: use our Constraints via the real `find!`
//! macro. Exercises the full propose / confirm / satisfied
//! protocol through the engine's own join machinery, producing
//! typed Rust tuples as output.
//!
//! This is the test that proves "yes, you can drop these
//! constraints into a normal triblespace query and get rows
//! back" — distinct from the unit tests (individual methods) and
//! the IntersectionConstraint tests (manual composition without
//! the macro).

use std::collections::HashSet;

use triblespace::core::find;
use triblespace::core::id::Id;

use triblespace_search::bm25::BM25Builder;
use triblespace_search::tokens::hash_tokens;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

fn sample_index() -> triblespace_search::bm25::BM25Index {
    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("the quick brown fox"));
    b.insert(id(2), hash_tokens("the lazy brown dog"));
    b.insert(id(3), hash_tokens("quick silver fox jumps"));
    b.build()
}

/// Single-variable find!: enumerate every doc that mentions
/// "fox". Expect two rows (docs 1 and 3).
#[test]
fn find_docs_containing_term() {
    let idx = sample_index();
    let fox = hash_tokens("fox")[0];

    let rows: Vec<(Id,)> = find!(
        (doc: Id),
        idx.docs_containing(doc, fox)
    )
    .collect();

    let set: HashSet<Id> = rows.into_iter().map(|(d,)| d).collect();
    assert_eq!(set.len(), 2);
    assert!(set.contains(&id(1)));
    assert!(set.contains(&id(3)));
}

/// Two-variable find!: binds both `doc` and `score`. Uses a
/// corpus where the "fox" postings have *different* BM25 scores
/// (doc 1 is length-1, doc 2 is length-7) so the engine can't
/// collapse the score variable to a single value. Confirms the
/// engine's propose/confirm protocol preserves the (doc, score)
/// correlation — a broken constraint would produce 2×2 = 4
/// Cartesian rows.
#[test]
fn find_docs_and_scores() {
    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("fox"));
    b.insert(id(2), hash_tokens("quick brown fox jumps high today!"));
    b.insert(id(3), hash_tokens("unrelated content"));
    let idx = b.build();
    let fox = hash_tokens("fox")[0];

    // Sanity: the two scores should actually differ.
    let postings: Vec<_> = idx.query_term(&fox).collect();
    assert_eq!(postings.len(), 2);
    let (_, s_a) = postings[0];
    let (_, s_b) = postings[1];
    assert!(
        (s_a - s_b).abs() > f32::EPSILON,
        "test fixture: doc scores should differ"
    );

    let rows: Vec<(Id, f32)> = find!(
        (doc: Id, score: f32),
        idx.docs_and_scores(doc, score, fox)
    )
    .collect();

    // Map each doc to its real posting score for cross-checking.
    let truth: std::collections::HashMap<Id, f32> =
        postings.iter().copied().collect();

    // Every row's (doc, score) must be one of the real postings.
    assert!(!rows.is_empty());
    for (d, s) in &rows {
        let expected = truth.get(d).copied().expect("row doc in postings");
        assert!(
            (expected - s).abs() < 1e-5,
            "row has mismatched score for doc {d:?}: got {s}, expected {expected}"
        );
    }
    // And every posting appears at least once.
    let row_docs: HashSet<Id> = rows.iter().map(|(d, _)| *d).collect();
    for (d, _) in &postings {
        assert!(row_docs.contains(d), "posting doc {d:?} missing from rows");
    }
}

/// `find!` with no variables — pure existence check, matches the
/// `exists!` pattern. Here "quick" appears in two docs, so the
/// query has at least one row.
#[test]
fn find_no_projection_is_existence() {
    let idx = sample_index();
    let quick = hash_tokens("quick")[0];
    let count = find!(
        (doc: Id),
        idx.docs_containing(doc, quick)
    )
    .count();
    assert_eq!(count, 2);
}

/// Two constraints in an `and!`: docs that contain BOTH "fox"
/// AND "quick". Only docs 1 and 3 match in the tiny sample.
/// Verifies that two BM25 constraints sharing a variable
/// intersect correctly through the macro.
#[test]
fn find_intersection_of_two_terms() {
    use triblespace::core::and;

    let idx = sample_index();
    let fox = hash_tokens("fox")[0];
    let quick = hash_tokens("quick")[0];

    let rows: Vec<(Id,)> = find!(
        (doc: Id),
        and!(
            idx.docs_containing(doc, fox),
            idx.docs_containing(doc, quick)
        )
    )
    .collect();

    let set: HashSet<Id> = rows.into_iter().map(|(d,)| d).collect();
    assert_eq!(set.len(), 2);
    assert!(set.contains(&id(1)));
    assert!(set.contains(&id(3)));
}
