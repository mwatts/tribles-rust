//! Sanity-check the naive BM25 + flat k-NN impls at a
//! not-completely-trivial scale (O(thousand) docs) so regressions
//! that only fire on large posting lists or long vector tables
//! surface in CI instead of at JP's first real corpus.
//!
//! These tests are still fast — seconds, not minutes — so they
//! stay in the default test run.

use triblespace::core::id::{Id, RawId};

use triblespace_search::bm25::{BM25Builder, BM25Index};
use triblespace_search::hnsw::{FlatBuilder, FlatIndex, HNSWBuilder, HNSWIndex};
use triblespace_search::succinct::{SuccinctBM25Index, SuccinctHNSWIndex};
use triblespace_search::tokens::hash_tokens;

/// Small pseudo-RNG (SplitMix64) — deterministic, no extra deps.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn id_from_u64(mut n: u64) -> Id {
    // Avoid the nil id by forcing a non-zero byte. Any mapping
    // works; this is just a deterministic "u64 → Id" for tests.
    n = n.max(1);
    let mut raw: RawId = [0; 16];
    raw[..8].copy_from_slice(&n.to_le_bytes());
    raw[8..].copy_from_slice(&n.wrapping_mul(0x9E3779B97F4A7C15).to_le_bytes());
    Id::new(raw).expect("non-nil by construction")
}

/// Produce `n_words` tokens drawn pseudo-randomly from a fixed
/// vocabulary of `vocab_size` English words. The distribution
/// skews Zipfian-ish so some terms are common, some rare.
fn fake_document(rng: &mut SplitMix64, vocab_size: usize, n_words: usize) -> String {
    let mut words: Vec<String> = Vec::with_capacity(n_words);
    for _ in 0..n_words {
        // Zipf-ish skew: square the random u32 to bias low.
        let r = rng.next() as f64 / u64::MAX as f64;
        let biased = (r * r) as f64;
        let idx = (biased * vocab_size as f64) as usize;
        let idx = idx.min(vocab_size - 1);
        words.push(format!("w{idx}"));
    }
    words.join(" ")
}

#[test]
fn bm25_1k_docs_roundtrip_consistency() {
    const N_DOCS: usize = 1_000;
    const VOCAB: usize = 500;
    const DOC_LEN: usize = 20;

    let mut rng = SplitMix64(0xC0FFEE);
    let mut builder = BM25Builder::new();
    for i in 0..N_DOCS {
        let doc = fake_document(&mut rng, VOCAB, DOC_LEN);
        builder.insert(id_from_u64(i as u64 + 1), hash_tokens(&doc));
    }
    let idx = builder.build();
    assert_eq!(idx.doc_count(), N_DOCS);
    assert!(idx.term_count() > 0 && idx.term_count() <= VOCAB);

    // Round-trip: serialize, reload, confirm query parity for a
    // handful of terms.
    let bytes = idx.to_bytes();
    let reloaded = BM25Index::try_from_bytes(&bytes).expect("valid");
    for term_text in ["w0", "w1", "w42", "w250", "w499"] {
        let term = hash_tokens(term_text);
        if term.is_empty() {
            continue;
        }
        let a: Vec<_> = idx.query_term(&term[0]).collect();
        let b: Vec<_> = reloaded.query_term(&term[0]).collect();
        assert_eq!(a.len(), b.len(), "query {term_text} posting lengths");
        for ((id_a, s_a), (id_b, s_b)) in a.iter().zip(b.iter()) {
            assert_eq!(id_a, id_b);
            assert!((s_a - s_b).abs() < 1e-6);
        }
    }
}

#[test]
fn bm25_1k_docs_multi_term_ranks_sanely() {
    let mut rng = SplitMix64(0xDEADBEEF);
    let mut builder = BM25Builder::new();
    for i in 0..1_000 {
        let doc = fake_document(&mut rng, 200, 30);
        builder.insert(id_from_u64(i as u64 + 1), hash_tokens(&doc));
    }
    // Inject a unique "needle" doc.
    let needle_id = id_from_u64(999_999);
    builder.insert(needle_id, hash_tokens("needle needle beacon"));
    let idx = builder.build();

    let q = hash_tokens("needle beacon");
    let ranked = idx.query_multi(&q);
    // Needle doc should be the top hit since both rare tokens
    // land only there.
    assert_eq!(ranked[0].0, needle_id);
}

#[test]
fn hnsw_1k_vectors_recall_against_flat() {
    // Build the same 1k 32-dim vectors into both Flat (ground
    // truth) and HNSW, then check HNSW's top-10 recalls at least
    // 70% of Flat's top-10 across several queries. Plus confirm
    // the HNSW blob round-trips cleanly.
    use std::collections::HashSet;

    const N_DOCS: usize = 1_000;
    const DIM: usize = 32;

    let mut rng = SplitMix64(0xFACE_FEED);
    let mut flat_b = FlatBuilder::new(DIM);
    let mut hnsw_b = HNSWBuilder::new(DIM).with_seed(7);
    for i in 0..N_DOCS {
        let vec: Vec<f32> = (0..DIM)
            .map(|_| (rng.next() as i32 as f32) / (i32::MAX as f32))
            .collect();
        let doc = id_from_u64((i + 1) as u64);
        flat_b.insert(doc, vec.clone()).unwrap();
        hnsw_b.insert(doc, vec).unwrap();
    }
    let flat = flat_b.build();
    let hnsw = hnsw_b.build();

    let mut total_overlap = 0;
    let mut total_k = 0;
    for _ in 0..5 {
        let q: Vec<f32> = (0..DIM)
            .map(|_| (rng.next() as i32 as f32) / (i32::MAX as f32))
            .collect();
        let truth: HashSet<_> =
            flat.similar(&q, 10).into_iter().map(|(d, _)| d).collect();
        let got: HashSet<_> = hnsw
            .similar(&q, 10, Some(100))
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        total_overlap += truth.intersection(&got).count();
        total_k += 10;
    }
    let recall = total_overlap as f32 / total_k as f32;
    assert!(
        recall >= 0.7,
        "HNSW 1k-doc recall {recall:.2} below 0.7 threshold"
    );

    // Round-trip the HNSW blob end-to-end.
    let bytes = hnsw.to_bytes();
    let reloaded = HNSWIndex::try_from_bytes(&bytes).expect("valid blob");
    let q: Vec<f32> = (0..DIM)
        .map(|_| (rng.next() as i32 as f32) / (i32::MAX as f32))
        .collect();
    let orig_ids: HashSet<_> = hnsw
        .similar(&q, 5, Some(50))
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    let loaded_ids: HashSet<_> = reloaded
        .similar(&q, 5, Some(50))
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    assert_eq!(orig_ids, loaded_ids);
}

/// At 1k docs the succinct BM25 must answer identically to the
/// naive one. Catches any encoding/decoding drift that a hand-
/// picked 4-doc test wouldn't stress (long posting lists, many
/// distinct terms, variable doc lengths).
#[test]
fn succinct_bm25_1k_docs_matches_naive() {
    let mut rng = SplitMix64(0xC0FFEE);
    let mut builder = BM25Builder::new();
    for i in 0..1_000 {
        let doc = fake_document(&mut rng, 500, 20);
        builder.insert(id_from_u64(i as u64 + 1), hash_tokens(&doc));
    }
    let naive = builder.build();
    let succinct = SuccinctBM25Index::from_naive(&naive).unwrap();

    assert_eq!(succinct.doc_count(), naive.doc_count());
    assert_eq!(succinct.term_count(), naive.term_count());

    // Sample a handful of terms, including very common and very
    // rare ones from the fake vocab. Every one must produce the
    // same posting list.
    for term_text in ["w0", "w1", "w7", "w50", "w250", "w499"] {
        let term = hash_tokens(term_text);
        if term.is_empty() {
            continue;
        }
        let a: Vec<_> = naive.query_term(&term[0]).collect();
        let b: Vec<_> = succinct.query_term(&term[0]).collect();
        assert_eq!(
            a.len(),
            b.len(),
            "term {term_text}: naive {} vs succinct {} postings",
            a.len(),
            b.len()
        );
        for ((id_a, s_a), (id_b, s_b)) in a.iter().zip(b.iter()) {
            assert_eq!(id_a, id_b);
            assert!(
                (s_a - s_b).abs() < 1e-6,
                "term {term_text}: score mismatch {s_a} vs {s_b}"
            );
        }
    }

    // Blob round-trip at this scale.
    let bytes = succinct.to_bytes();
    let reloaded = SuccinctBM25Index::try_from_bytes(&bytes).expect("valid");
    assert_eq!(reloaded.doc_count(), succinct.doc_count());
    let term = hash_tokens("w7");
    let a: Vec<_> = succinct.query_term(&term[0]).collect();
    let b: Vec<_> = reloaded.query_term(&term[0]).collect();
    assert_eq!(a.len(), b.len());
    for ((id_a, s_a), (id_b, s_b)) in a.iter().zip(b.iter()) {
        assert_eq!(id_a, id_b);
        assert!((s_a - s_b).abs() < 1e-6);
    }
}

/// Succinct HNSW must answer the same top-k as the naive one for
/// a 1k-vector corpus. This is the scale-level sanity check for
/// the succinct graph + vector encoding — any off-by-one in
/// offsets / vector indexing shows up as divergent results.
#[test]
fn succinct_hnsw_1k_docs_matches_naive() {
    const DIM: usize = 16;

    let mut rng = SplitMix64(0xBADF00D);
    let mut builder = HNSWBuilder::new(DIM).with_seed(11);
    for i in 0..1_000 {
        let vec: Vec<f32> = (0..DIM)
            .map(|_| (rng.next() as i32 as f32) / (i32::MAX as f32))
            .collect();
        builder.insert(id_from_u64((i + 1) as u64), vec).unwrap();
    }
    let naive = builder.build();
    let succinct = SuccinctHNSWIndex::from_naive(&naive).unwrap();

    for _ in 0..5 {
        let q: Vec<f32> = (0..DIM)
            .map(|_| (rng.next() as i32 as f32) / (i32::MAX as f32))
            .collect();
        let n = naive.similar(&q, 10, Some(50));
        let s = succinct.similar(&q, 10, Some(50));
        assert_eq!(n.len(), s.len());
        for ((n_id, n_s), (s_id, s_s)) in n.iter().zip(s.iter()) {
            assert_eq!(n_id, s_id, "doc mismatch at 1k scale");
            assert!(
                (n_s - s_s).abs() < 1e-5,
                "score mismatch at 1k scale: {n_s} vs {s_s}"
            );
        }
    }
}

#[test]
fn flat_1k_vectors_top_k_consistent() {
    const N_DOCS: usize = 1_000;
    const DIM: usize = 32;

    let mut rng = SplitMix64(0x1234_5678);
    let mut builder = FlatBuilder::new(DIM);
    let target = id_from_u64(42);
    // One vector we know the answer for — all ones in the first
    // half, zero in the second — and 999 random neighbours.
    let mut target_vec = vec![0.0f32; DIM];
    for v in target_vec.iter_mut().take(DIM / 2) {
        *v = 1.0;
    }
    builder.insert(target, target_vec.clone()).unwrap();
    for i in 0..(N_DOCS - 1) {
        let vec: Vec<f32> = (0..DIM)
            .map(|_| (rng.next() as i32 as f32) / (i32::MAX as f32))
            .collect();
        builder.insert(id_from_u64(i as u64 + 100), vec).unwrap();
    }
    let idx = builder.build();

    // The exact `target_vec` as a query should return `target` at
    // rank 1.
    let hits = idx.similar(&target_vec, 5);
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].0, target);

    // Round-trip through bytes must preserve the top-k.
    let reloaded = FlatIndex::try_from_bytes(&idx.to_bytes()).expect("valid");
    let hits2 = reloaded.similar(&target_vec, 5);
    assert_eq!(hits.len(), hits2.len());
    for (a, b) in hits.iter().zip(hits2.iter()) {
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-6);
    }
}
