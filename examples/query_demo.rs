//! Minimal end-to-end demo: build a BM25 index over a handful of
//! doc strings, serialize it to bytes, reload, query.
//!
//! ```sh
//! cargo run --example query_demo
//! ```

use triblespace::core::id::Id;
use triblespace_search::bm25::{BM25Builder, BM25Index};
use triblespace_search::tokens::hash_tokens;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).expect("non-nil")
}

fn main() {
    // A corpus of five fragments.
    let corpus = [
        (id(1), "Typst is a markup-based typesetting system."),
        (
            id(2),
            "Wiki fragments are small typst documents linked by id.",
        ),
        (id(3), "BM25 scores documents by term frequency and IDF."),
        (
            id(4),
            "Each fragment cites other fragments via wiki: links.",
        ),
        (id(5), "Cosine similarity ranks embeddings by direction."),
    ];

    // Build.
    let mut builder = BM25Builder::new();
    for (id, text) in &corpus {
        builder.insert_id(*id, hash_tokens(text));
        println!("indexed {id} ({} tokens)", hash_tokens(text).len());
    }
    let idx = builder.build();
    println!(
        "\nindex: {} docs, {} terms, avg_doc_len = {:.2}",
        idx.doc_count(),
        idx.term_count(),
        idx.avg_doc_len()
    );

    // Serialize round-trip — the same bytes end-to-end.
    let bytes = idx.to_bytes();
    let reloaded = BM25Index::try_from_bytes(&bytes).expect("valid blob");
    println!("\nblob size: {} bytes", bytes.len());
    assert_eq!(reloaded.doc_count(), idx.doc_count());

    // Single-term query.
    println!("\nquery: 'typst'");
    let q = hash_tokens("typst");
    for (doc, score) in reloaded.query_term_ids(&q[0]) {
        println!("  {doc}  score={score:.3}");
    }

    // Multi-term OR-query with sum-of-BM25 ranking.
    println!("\nquery: 'fragment wiki'");
    let q = hash_tokens("fragment wiki");
    for (doc, score) in reloaded.query_multi_ids(&q).into_iter().take(3) {
        println!("  {doc}  score={score:.3}");
    }

    // Value-as-term: use a doc's Id as a "citation term" and
    // index a new micro-corpus where each doc is a list of the
    // fragments it cites. The same BM25 index gives us
    // "documents citing this fragment".
    println!("\ncitation search (term = fragment id):");
    let mut cite_builder = BM25Builder::new();
    cite_builder.insert_id(
        id(10),
        vec![*id(1).as_ref(); 1]
            .iter()
            .map(|r| raw_from_id(r))
            .collect(),
    );
    cite_builder.insert_id(
        id(11),
        vec![id(1), id(3)]
            .iter()
            .map(|i| raw_from_id(i.as_ref()))
            .collect(),
    );
    cite_builder.insert_id(
        id(12),
        vec![id(3)]
            .iter()
            .map(|i| raw_from_id(i.as_ref()))
            .collect(),
    );
    let cite_idx = cite_builder.build();

    let cites_one: Vec<_> = cite_idx.query_term_ids(&raw_from_id(id(1).as_ref())).collect();
    println!("  citations of {}: {} doc(s)", id(1), cites_one.len());
    for (doc, _) in cites_one {
        println!("    cited by {doc}");
    }
}

/// Helper: lift a 16-byte `RawId` into a 32-byte term value by
/// zero-padding. Keeps `Id`s and hashed tokens in the same 32-byte
/// term space.
fn raw_from_id(id: &[u8; 16]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(id);
    out
}
