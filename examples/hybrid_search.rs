//! BM25 + HNSW composed in a single `find!`.
//!
//! Scenario: a tiny catalog of papers where each paper has both
//! a title (suitable for lexical search) and an embedding
//! (suitable for semantic similarity). We build a BM25 index
//! over the titles and an HNSW index over the embeddings, then
//! ask one question that needs both: *"papers whose title
//! mentions 'graph' AND whose embedding is close to the query
//! vector."*
//!
//! Both indexes are keyed by the same entity id, so they meet
//! through the shared `?paper` variable inside `find!`. No
//! manual filtering; the triblespace query engine picks the
//! cheaper constraint to iterate (`estimate()`) and confirms
//! with the other.
//!
//! Everything lives in one `MemoryBlobStore`:
//! - embedding blobs (referenced by the HNSW index via handle)
//! - the `SuccinctHNSWIndex` blob itself
//! - (BM25 stays in-memory for this demo; it's small)
//!
//! ```sh
//! cargo run --example hybrid_search
//! ```
use triblespace::core::and;
use triblespace::core::blob::MemoryBlobStore;
use triblespace::core::find;
use triblespace::core::id::Id;
use triblespace::core::repo::BlobStore;
use triblespace::core::value::schemas::hash::Blake3;

use triblespace_search::bm25::BM25Builder;
use triblespace_search::hnsw::HNSWBuilder;
use triblespace_search::schemas::put_embedding;
use triblespace_search::tokens::hash_tokens;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).expect("non-nil")
}

fn main() {
    // ── Seed a handful of papers with titles + embeddings ─
    //
    // Embeddings are 4-dim toy vectors; in a real setup they'd
    // be caller-supplied (fastembed, Voyage, OpenAI, whatever).
    // Keeping them 4-D here lets the arithmetic stay readable.
    let papers: Vec<(Id, &str, Vec<f32>)> = vec![
        // ─ close to [1,0,0,0] cluster ─
        (
            id(1),
            "Graph neural networks for node classification",
            vec![0.95, 0.1, 0.05, 0.0],
        ),
        (
            id(2),
            "Succinct data structures for graph search",
            vec![0.90, 0.15, 0.10, 0.0],
        ),
        // ─ far from [1,0,0,0] but title has "graph" ─
        (
            id(3),
            "Graph kernels compared to transformer pooling",
            vec![0.0, 0.0, 1.0, 0.0],
        ),
        // ─ close to [1,0,0,0] but no "graph" in title ─
        (
            id(4),
            "Efficient k-NN search with inverted files",
            vec![0.92, 0.08, 0.0, 0.0],
        ),
        // ─ far AND no "graph" ─
        (
            id(5),
            "Monte Carlo tree search for game playing",
            vec![0.0, 0.1, 0.0, 1.0],
        ),
    ];

    println!("Corpus: {} papers\n", papers.len());
    for (pid, title, _) in &papers {
        println!("  {pid}  {title}");
    }

    // ── Build the two indexes ─────────────────────────────
    //
    // One `MemoryBlobStore` holds both the embedding blobs and
    // (below) the persisted HNSW index blob. Content-addressing
    // means two papers with identical embeddings would share a
    // blob automatically — we don't have any duplicates here,
    // but the property is free.
    let mut store = MemoryBlobStore::<Blake3>::new();

    let mut bm25_b = BM25Builder::new();
    let mut hnsw_b = HNSWBuilder::new(4).with_seed(42);
    for (pid, title, vec) in &papers {
        // BM25: tokenize title, key by paper id.
        bm25_b.insert(&*pid, hash_tokens(title));
        // HNSW: put the embedding (normalized) and insert the
        // resulting handle. The builder keeps the vec in memory
        // during graph construction for distance computations;
        // it drops the vec at `build()` so only the handle
        // survives into the final index.
        let h = put_embedding::<_, Blake3>(&mut store, vec.clone()).unwrap();
        hnsw_b.insert(&*pid, h, vec.clone()).unwrap();
    }
    let bm25 = bm25_b.build();
    let hnsw = hnsw_b.build();

    // Attach a reader so the HNSW walk can resolve embedding
    // handles at query time.
    let reader = store.reader().unwrap();
    let hnsw_view = hnsw.attach(&reader);

    // ── The headline query ────────────────────────────────
    //
    // "Papers with 'graph' in the title AND embedding close to
    // [1, 0, 0, 0]." One `find!`, two constraints joined on the
    // shared `?paper` variable.
    let graph_term = hash_tokens("graph")[0];
    let query_vec = vec![1.0, 0.0, 0.0, 0.0];

    println!("\nQuery: title contains 'graph' AND embedding close to [1,0,0,0]");

    // The HNSW side eagerly resolves the top-k against the
    // store at constraint construction; the BM25 side is lazy
    // (iterator over posting list). The engine joins them via
    // `?paper`.
    let hits: Vec<(Id,)> = find!(
        (paper: Id),
        and!(
            bm25.docs_containing(paper, graph_term),
            hnsw_view
                .similar_constraint(paper, query_vec.clone(), 4, Some(10))
                .unwrap(),
        )
    )
    .collect();

    println!("  {} rows:", hits.len());
    for (pid,) in &hits {
        let title = papers
            .iter()
            .find(|(x, _, _)| x == pid)
            .map(|(_, t, _)| *t)
            .unwrap_or("?");
        println!("    {pid}  {title}");
    }

    // ── Expected survivors ─────────────────────────────────
    //
    // Paper 1: "graph" ✓  +  close ✓  → in
    // Paper 2: "graph" ✓  +  close ✓  → in
    // Paper 3: "graph" ✓  +  far  ✗  → out  (semantic filter)
    // Paper 4: "graph" ✗  +  close ✓  → out  (lexical filter)
    // Paper 5: "graph" ✗  +  far  ✗  → out
    let got: std::collections::HashSet<Id> = hits.iter().map(|(p,)| *p).collect();
    assert!(got.contains(&id(1)));
    assert!(got.contains(&id(2)));
    assert!(!got.contains(&id(3)), "paper 3 must be excluded by HNSW");
    assert!(!got.contains(&id(4)), "paper 4 must be excluded by BM25");
    assert!(!got.contains(&id(5)));

    // ── Same idea, now with the cosine score bound ────────
    //
    // The scored variant lets the query project the similarity
    // value itself — useful for downstream ranking or filtering
    // on a score threshold.
    println!("\nScored variant: same filter, project cosine score");
    let scored: Vec<(Id, f32)> = find!(
        (paper: Id, score: f32),
        and!(
            bm25.docs_containing(paper, graph_term),
            hnsw_view
                .similar_with_scores(paper, score, query_vec.clone(), 4, Some(10))
                .unwrap(),
        )
    )
    .collect();
    for (pid, s) in &scored {
        println!("    {pid}  cos={s:.3}");
    }

    println!("\n✓ hybrid AND works — neither constraint alone is sufficient");
}
