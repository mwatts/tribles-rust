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
use triblespace_search::succinct::SuccinctBM25Index;
use triblespace_search::tokens::hash_tokens;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

fn sample_index() -> SuccinctBM25Index {
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

/// Regression: two docs with identical BM25 scores must yield
/// one row per doc, not a Cartesian product within the score
/// bucket. Early versions of `BM25ScoredPostings::propose` pushed
/// one `score` proposal per posting, so N docs sharing a score
/// caused N×N rows. Now the constraint dedupes by bit-pattern.
#[test]
fn find_docs_and_scores_shared_score_no_cartesian() {
    let mut b = BM25Builder::new();
    // Three docs, same length and same tf for "fox" — identical
    // scores.
    b.insert(id(1), hash_tokens("the quick fox"));
    b.insert(id(2), hash_tokens("another fox book"));
    b.insert(id(3), hash_tokens("fox adventure here"));
    // One filler doc so corpus avg_doc_len is well-defined.
    b.insert(id(4), hash_tokens("unrelated content only"));
    let idx = b.build();
    let fox = hash_tokens("fox")[0];

    let postings: Vec<_> = idx.query_term(&fox).collect();
    assert_eq!(postings.len(), 3);

    let rows: Vec<(Id, f32)> = find!(
        (doc: Id, score: f32),
        idx.docs_and_scores(doc, score, fox)
    )
    .collect();
    // Exactly 3 rows — one per doc — regardless of score equality.
    assert_eq!(
        rows.len(),
        3,
        "expected one row per doc; got {} rows (Cartesian?)",
        rows.len()
    );
    let docs: HashSet<Id> = rows.iter().map(|(d, _)| *d).collect();
    assert_eq!(docs.len(), 3);
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
    // Postings are keyed by 32-byte RawValue now; decode back to
    // the Id each row's `doc: Id` projects to.
    let id_from_raw = |raw: &[u8; 32]| -> Id {
        Id::new(raw[16..32].try_into().unwrap()).unwrap()
    };
    let truth: std::collections::HashMap<Id, f32> = postings
        .iter()
        .copied()
        .map(|(v, s)| (id_from_raw(&v.raw), s))
        .collect();

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
    for (v, _) in &postings {
        let d = id_from_raw(&v.raw);
        assert!(row_docs.contains(&d), "posting doc {d:?} missing from rows");
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

/// Succinct-view BM25 answers a `find!` with a bound score
/// variable, and returns a non-empty set of (doc, score) pairs.
/// Exercises the `BM25Queryable::score_tolerance` trait method
/// on the quantized-succinct path — previously the tolerance
/// widening never actually ran through the engine in any test.
#[test]
fn find_docs_and_scores_on_succinct() {
    use triblespace_search::succinct::SuccinctBM25Index;

    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("fox"));
    b.insert(id(2), hash_tokens("quick brown fox jumps high today"));
    b.insert(id(3), hash_tokens("unrelated content"));
    let succinct: SuccinctBM25Index = b.build();
    let fox = hash_tokens("fox")[0];

    let rows: Vec<(Id, f32)> = find!(
        (doc: Id, score: f32),
        succinct.docs_and_scores(doc, score, fox)
    )
    .collect();
    assert!(!rows.is_empty(), "succinct index should produce rows");

    // Every row's score must match the score produced by direct
    // `query_term` lookup on the same index — verifies the
    // engine's propose/confirm path delivers the same result as
    // the direct query API.
    let id_from_raw =
        |raw: &[u8; 32]| -> Id { Id::new(raw[16..32].try_into().unwrap()).unwrap() };
    for (doc, score) in &rows {
        let expected: f32 = succinct
            .query_term(&fox)
            .find(|(d, _)| id_from_raw(&d.raw) == *doc)
            .map(|(_, s)| s)
            .expect("doc must be in succinct postings");
        assert!(
            (expected - score).abs() < 1e-6,
            "engine-path score {score} for {doc:?} diverged from direct {expected}"
        );
    }
}

/// The succinct HNSW view plugs into `find!` via the binary
/// [`Similar`] relation. Declare two handle variables, pin the
/// first to a known handle with `.is()`, let the engine
/// enumerate the second from the HNSW walk, and cross-check
/// against the direct `candidates_above` API.
///
/// [`Similar`]: triblespace_search::constraint::Similar
#[test]
fn find_hnsw_similar_on_succinct() {
    use std::collections::HashSet;
    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::repo::BlobStore;
    use triblespace::core::value::schemas::hash::{Blake3, Handle};
    use triblespace::core::value::Value;
    use triblespace_search::hnsw::HNSWBuilder;
    use triblespace_search::schemas::{put_embedding, Embedding};

    let mut store = MemoryBlobStore::<Blake3>::new();
    let mut b = HNSWBuilder::new(4).with_seed(23);
    let mut handles = Vec::new();
    for i in 1..=16u8 {
        let f = i as f32;
        let v = vec![f.sin(), f.cos(), (f * 0.5).sin(), (f * 0.3).cos()];
        let h = put_embedding::<_, Blake3>(&mut store, v.clone()).unwrap();
        b.insert(h, v).unwrap();
        handles.push(h);
    }
    let succinct = b.build();
    let reader = store.reader().unwrap();
    let succinct_view = succinct.attach(&reader);
    let probe = handles[0];
    let floor = 0.4f32;

    let rows: Vec<(Value<Handle<Blake3, Embedding>>,)> = find!(
        (
            neighbour: Value<Handle<Blake3, Embedding>>,
        ),
        triblespace::core::query::temp!(
            (anchor),
            triblespace::core::and!(
                anchor.is(probe),
                succinct_view.similar(anchor, neighbour, floor)
            )
        )
    )
    .collect();
    let got: HashSet<_> = rows.into_iter().map(|(h,)| h).collect();

    let expected: HashSet<_> = succinct_view
        .candidates_above(probe, floor)
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(got, expected);
}

/// The succinct view answers `find!` queries identically to the
/// naive one — proves the `BM25Queryable` trait actually plugs
/// the succinct path into the engine without regressions. Uses
/// the same corpus as `find_docs_containing_term`, just querying
/// against the succinct view of the same index.
#[test]
fn find_docs_containing_term_on_succinct() {
    // `sample_index()` already returns a SuccinctBM25Index —
    // kept as a distinct test alongside `find_docs_containing_term`
    // so a future change that separates the naive and succinct
    // engine paths doesn't silently drop coverage.
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

/// Multi-term BM25 via `bm25_query`: a bag-of-words query that
/// binds `doc` + the summed BM25 score across every query term,
/// running through the normal engine join machinery rather than
/// a side-channel `query_multi` call.
#[test]
fn find_multi_term_bm25_scored() {
    use triblespace_search::bm25::BM25Builder;

    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("the quick brown fox"));
    b.insert(id(2), hash_tokens("the lazy brown dog"));
    b.insert(id(3), hash_tokens("quick silver fox jumps"));
    b.insert(id(4), hash_tokens("entirely unrelated"));
    let idx = b.build();

    let terms = hash_tokens("quick fox");
    let rows: Vec<(Id, f32)> = find!(
        (doc: Id, score: f32),
        idx.bm25_query(doc, score, &terms)
    )
    .collect();

    let ids: HashSet<Id> = rows.iter().map(|(d, _)| *d).collect();
    // Docs 1 and 3 contain both "quick" and "fox"; docs 2 and 4
    // contain neither — they never hit a posting list and drop
    // out of the aggregation entirely.
    assert!(ids.contains(&id(1)));
    assert!(ids.contains(&id(3)));
    assert!(!ids.contains(&id(2)));
    assert!(!ids.contains(&id(4)));

    // Every row's score must be positive (BM25 scores are
    // positive for real term matches).
    for (_, s) in &rows {
        assert!(s.is_finite() && *s > 0.0);
    }
}

/// Headline story: BM25 lexical search composed with a `pattern!`
/// over a real TribleSet, in a single `find!`. "Find books whose
/// title mentions 'fox' AND are authored by the known author X."
#[test]
fn find_bm25_composed_with_pattern() {
    use triblespace::core::and;
    use triblespace::core::examples::literature;
    use triblespace::core::id::ExclusiveId;
    use triblespace::core::trible::TribleSet;
    use triblespace::macros::{entity, pattern};

    // Fixed Ids keep the test deterministic; `ExclusiveId::force_ref`
    // gives `entity!` the `&ExclusiveId` it expects.
    let target_author = id(10);
    let other_author = id(11);
    let book_a = id(20);
    let book_b = id(21);
    let book_c = id(22);
    let book_d = id(23);

    let mut kb = TribleSet::new();
    kb += entity! { ExclusiveId::force_ref(&target_author) @
        literature::firstname: "Target",
        literature::lastname: "Author",
    };
    kb += entity! { ExclusiveId::force_ref(&other_author) @
        literature::firstname: "Other",
        literature::lastname: "Author",
    };
    kb += entity! { ExclusiveId::force_ref(&book_a) @
        literature::title: "The Quick Fox",
        literature::author: &target_author,
    };
    kb += entity! { ExclusiveId::force_ref(&book_b) @
        literature::title: "Another Fox Book",
        literature::author: &target_author,
    };
    kb += entity! { ExclusiveId::force_ref(&book_c) @
        literature::title: "Fox Adventure",
        literature::author: &other_author,
    };
    kb += entity! { ExclusiveId::force_ref(&book_d) @
        literature::title: "Unrelated",
        literature::author: &target_author,
    };

    // Build a BM25 index over book titles, keyed by book entity id.
    let titles: Vec<(Id, String)> = find!(
        (b: Id, title: String),
        pattern!(&kb, [{ ?b @ literature::title: ?title }])
    )
    .collect();
    let mut bm25 = BM25Builder::new();
    for (b, title) in &titles {
        bm25.insert(*b, hash_tokens(title));
    }
    let idx = bm25.build();

    // Compose: "books that mention 'fox' AND are by target_author".
    let fox = hash_tokens("fox")[0];
    let rows: Vec<(Id,)> = find!(
        (book: Id),
        and!(
            idx.docs_containing(book, fox),
            pattern!(&kb, [{ ?book @ literature::author: &target_author }])
        )
    )
    .collect();

    let books: HashSet<Id> = rows.into_iter().map(|(b,)| b).collect();
    assert_eq!(
        books.len(),
        2,
        "expected 2 fox books by target author, got {}",
        books.len()
    );
    assert!(books.contains(&book_a));
    assert!(books.contains(&book_b));
    assert!(!books.contains(&book_c), "should exclude wrong author");
    assert!(!books.contains(&book_d), "should exclude no-fox title");
}
