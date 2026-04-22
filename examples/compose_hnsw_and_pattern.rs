//! Vector search composed with a `pattern!` over a real
//! `TribleSet`, in a single `find!`.
//!
//! Scenario: a tiny catalog with books that have both an author
//! and an embedding. Build an HNSW index over embeddings (keyed
//! by book id), convert to the succinct view, and ask the query
//! engine for "books whose embedding is close to the query AND
//! are authored by the target author". The HNSW constraint and
//! the pattern! clause join through the shared `?book` variable.
//!
//! ```sh
//! cargo run --example compose_hnsw_and_pattern
//! ```

use triblespace::core::and;
use triblespace::core::blob::MemoryBlobStore;
use triblespace::core::examples::literature;
use triblespace::core::find;
use triblespace::core::id::{ExclusiveId, Id};
use triblespace::core::repo::BlobStore;
use triblespace::core::trible::TribleSet;
use triblespace::core::value::schemas::hash::Blake3;
use triblespace::macros::{entity, pattern};

use triblespace_search::hnsw::HNSWBuilder;
use triblespace_search::schemas::put_embedding;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).expect("non-nil")
}

fn main() {
    // Authors + four books, two per author.
    let target_author = id(10);
    let other_author = id(11);

    let book_a = id(20); // target_author, near-query embedding
    let book_b = id(21); // target_author, somewhat far embedding
    let book_c = id(22); // other_author, near-query embedding
    let book_d = id(23); // other_author, far embedding

    let mut kb = TribleSet::new();
    kb += entity! { ExclusiveId::force_ref(&target_author) @
        literature::firstname: "Target",
        literature::lastname: "Author",
    };
    kb += entity! { ExclusiveId::force_ref(&other_author) @
        literature::firstname: "Other",
        literature::lastname: "Author",
    };
    for (bid, title) in [
        (book_a, "A Near Tale"),
        (book_b, "A Distant Saga"),
        (book_c, "Close Encounters"),
        (book_d, "Unrelated Memoir"),
    ] {
        let author = if bid == book_a || bid == book_b {
            target_author
        } else {
            other_author
        };
        kb += entity! { ExclusiveId::force_ref(&bid) @
            literature::title: title,
            literature::author: &author,
        };
    }

    // Build an HNSW index over 4-D embeddings.
    // Vectors: books A and C sit near the query direction;
    // books B and D sit orthogonal or far.
    let embeddings: Vec<(Id, Vec<f32>)> = vec![
        (book_a, vec![0.9, 0.1, 0.05, 0.02]), // near query
        (book_b, vec![0.0, 0.0, 1.0, 0.0]),   // far
        (book_c, vec![0.85, 0.15, 0.1, 0.0]), // near query
        (book_d, vec![-1.0, 0.0, 0.0, 0.0]),  // opposite direction
    ];

    // Put each embedding into a MemoryBlobStore and keep the
    // handle alongside the (book_id, vec) tuple. HNSWBuilder
    // takes (id, handle, vec) — the vec stays in RAM during
    // build for graph-construction distances; the final index
    // resolves handles via the store at query time.
    let mut store = MemoryBlobStore::<Blake3>::new();
    let mut hb = HNSWBuilder::new(4).with_seed(42);
    for (bid, v) in &embeddings {
        let h = put_embedding::<_, Blake3>(&mut store, v.clone()).unwrap();
        hb.insert(&*bid, h, v.clone()).unwrap();
    }
    let idx = hb.build();
    let reader = store.reader().unwrap();
    println!(
        "HNSW index built: {} docs, dim = {}, max_level = {}",
        idx.doc_count(),
        idx.dim(),
        idx.max_level()
    );

    // Query vector: points in the direction books A and C live in.
    let query = vec![1.0, 0.0, 0.0, 0.0];

    // Standalone similarity — should surface A and C first.
    // `similar_ids` decodes the 32-byte keys back to `Id` under
    // the GenId schema (empty result on non-GenId keys).
    let idx_view = idx.attach(&reader);
    println!("\nsimilarity-only (no author filter):");
    for (d, s) in idx_view.similar_ids(&query, 4, Some(10)).unwrap() {
        println!("  {d}  cos={s:.3}");
    }

    // Headline query: top-k similar AND authored by target_author.
    println!("\nquery: similar to [1,0,0,0] AND author = target_author");
    let matches: Vec<(Id,)> = find!(
        (book: Id),
        and!(
            idx_view.similar_constraint(book, query.clone(), 4, Some(10)).unwrap(),
            pattern!(&kb, [{ ?book @ literature::author: &target_author }])
        )
    )
    .collect();
    println!("  {} rows:", matches.len());
    for (b,) in &matches {
        println!("    {b}");
    }

    // Sanity: book_c is near the query but authored by the
    // wrong author → excluded. book_d is by the target author
    // but similarity is bad and the HNSW top-k truncates it away.
    // Expected survivors: book_a (near + right author) and
    // possibly book_b (right author but low similarity — shows up
    // only because we asked for top-4).
    assert!(matches.iter().any(|(b,)| *b == book_a));
    assert!(!matches.iter().any(|(b,)| *b == book_c));

    // Scored variant: bind cosine score too.
    println!("\nscored: similar to [1,0,0,0] AND author = target_author");
    let scored: Vec<(Id, f32)> = find!(
        (book: Id, score: f32),
        and!(
            idx_view.similar_with_scores(book, score, query.clone(), 4, Some(10)).unwrap(),
            pattern!(&kb, [{ ?book @ literature::author: &target_author }])
        )
    )
    .collect();
    for (b, s) in &scored {
        println!("    {b}  cos={s:.3}");
    }
}
