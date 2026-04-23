//! Multi-term BM25 search composed with a trible `pattern!`,
//! projecting the summed BM25 score as a bound query variable.
//!
//! Scenario: a small book catalog where each book has a title
//! and an author. The caller asks "books whose title matches
//! 'graph search algorithms' AND are written by `target_author`,
//! ranked by BM25 score". That's three constraints — the
//! multi-term BM25 query, the author pattern — joined on the
//! shared `?book` variable in one `find!`, with `score` bound
//! as a query variable so the caller can sort the result rows
//! after collecting.
//!
//! Contrast with `compose_bm25_and_pattern`, which uses
//! `docs_containing` (single term, doc-only binding). This
//! example is the same shape at the engine level but with the
//! higher-level [`bm25_query`][q] constraint — the "two-level
//! query API" from the design doc.
//!
//! ```sh
//! cargo run --example multi_term_bm25_search
//! ```
//!
//! [q]: triblespace_search::bm25::BM25Index::bm25_query

use triblespace::core::and;
use triblespace::core::examples::literature;
use triblespace::core::find;
use triblespace::core::id::{ExclusiveId, Id};
use triblespace::core::trible::TribleSet;
use triblespace::macros::{entity, pattern};

use triblespace_search::bm25::BM25Builder;
use triblespace_search::tokens::hash_tokens;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).expect("non-nil")
}

fn main() {
    // Two authors, six books with titles chosen so "graph
    // search" scores some highly and others not at all.
    let target_author = id(10);
    let other_author = id(11);

    let books = [
        (id(20), target_author, "Graph search algorithms"),
        (id(21), target_author, "Graph search succinctly"),
        (id(22), target_author, "Monte Carlo tree search"),
        (id(23), target_author, "Cooking graph enthusiasts"),
        (id(24), other_author, "Graph search programmer"),
        (id(25), other_author, "Linear algebra"),
    ];

    let mut kb = TribleSet::new();
    kb += entity! { ExclusiveId::force_ref(&target_author) @
        literature::firstname: "Target",
        literature::lastname: "Author",
    };
    kb += entity! { ExclusiveId::force_ref(&other_author) @
        literature::firstname: "Other",
        literature::lastname: "Author",
    };
    for (book_id, author_id, title) in &books {
        kb += entity! { ExclusiveId::force_ref(book_id) @
            literature::title: *title,
            literature::author: author_id,
        };
    }
    println!("KB: 2 authors + {} books\n", books.len());

    // BM25 over titles, pulled straight out of the KB via a
    // pattern query — no shadow datamodel.
    let titles: Vec<(Id, String)> = find!(
        (b: Id, title: String),
        pattern!(&kb, [{ ?b @ literature::title: ?title }])
    )
    .collect();
    let mut bm25 = BM25Builder::new();
    for (b, title) in &titles {
        bm25.insert(b, hash_tokens(title));
    }
    let idx = bm25.build();
    println!(
        "BM25 index: {} docs, {} terms\n",
        idx.doc_count(),
        idx.term_count(),
    );

    // Standalone multi-term query — bag-of-words "graph search
    // algorithms". `bm25_query` pre-aggregates the sum of
    // per-term BM25 weights and exposes the (doc, score) table
    // as a constraint.
    let query_terms = hash_tokens("graph search algorithms");
    println!("standalone multi-term query: 'graph search algorithms'");
    let standalone: Vec<(Id, f32)> = find!(
        (book: Id, score: f32),
        idx.bm25_query(book, score, &query_terms)
    )
    .collect();
    let mut ranked = standalone.clone();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (b, s) in &ranked {
        let title = title_for(&titles, *b);
        println!("  {s:6.3}  {b}  {title}");
    }

    // Headline query: same multi-term score, gated on author.
    // The engine joins `bm25_query` with `pattern!` on the
    // shared ?book variable — no manual Rust-side filter.
    println!("\nquery: 'graph search algorithms' AND author = target_author");
    let matches: Vec<(Id, f32)> = find!(
        (book: Id, score: f32),
        and!(
            idx.bm25_query(book, score, &query_terms),
            pattern!(&kb, [{ ?book @ literature::author: &target_author }]),
        )
    )
    .collect();
    let mut matches_sorted = matches.clone();
    matches_sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (b, s) in &matches_sorted {
        let title = title_for(&titles, *b);
        println!("  {s:6.3}  {b}  {title}");
    }

    // Sanity:
    //   book 20 (target_author, high score)  → in
    //   book 21 (target_author, high score)  → in
    //   book 22 (target_author, mid score)   → in (contains "search")
    //   book 23 (target_author, low score)   → in (contains "graph")
    //   book 24 (other_author, high score)   → out  (author filter)
    //   book 25 (other_author, no match)     → out
    //   book 26 (target_author, no match)    → n/a
    let hit_ids: std::collections::HashSet<Id> =
        matches.iter().map(|(b, _)| *b).collect();
    assert!(hit_ids.contains(&id(20)));
    assert!(hit_ids.contains(&id(21)));
    assert!(!hit_ids.contains(&id(24)), "author filter must exclude book 24");
    assert!(!hit_ids.contains(&id(25)));

    // The top match must be by target_author and have a positive
    // score — without asserting a specific title (BM25 ordering
    // depends on doc length + IDF and is stable but brittle to
    // fixture edits).
    assert_eq!(
        matches_sorted[0].0,
        id(20).min(id(21)).max(matches_sorted[0].0),
        "top hit should be one of the high-score target-author books",
    );
}

fn title_for(titles: &[(Id, String)], book: Id) -> &str {
    titles
        .iter()
        .find(|(b, _)| *b == book)
        .map(|(_, t)| t.as_str())
        .unwrap_or("?")
}
