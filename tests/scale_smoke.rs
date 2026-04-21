//! Sanity-check the naive BM25 + flat k-NN impls at a
//! not-completely-trivial scale (O(thousand) docs) so regressions
//! that only fire on large posting lists or long vector tables
//! surface in CI instead of at JP's first real corpus.
//!
//! These tests are still fast — seconds, not minutes — so they
//! stay in the default test run.

use triblespace::core::id::{Id, RawId};

use triblespace_search::bm25::{BM25Builder, BM25Index};
use triblespace_search::hnsw::{FlatBuilder, FlatIndex};
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
