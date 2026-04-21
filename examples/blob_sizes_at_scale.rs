//! Prints the naive vs. SB25 blob size for a few fake-corpus
//! sizes, plus build time. Meant for eyeballing compression at a
//! glance — `cargo run --release --example blob_sizes_at_scale`.
//!
//! Not an assertion test; that's what the regression guards in
//! `tests/scale_smoke.rs` are for.

use std::time::Instant;

use triblespace::core::id::{Id, RawId};
use triblespace_search::bm25::BM25Builder;
use triblespace_search::succinct::SuccinctBM25Index;
use triblespace_search::tokens::hash_tokens;

/// Tiny deterministic PRNG (SplitMix64) so runs are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn id_from_u64(n: u64) -> Id {
    let n = n.max(1);
    let mut raw: RawId = [0; 16];
    raw[..8].copy_from_slice(&n.to_le_bytes());
    raw[8..].copy_from_slice(&n.wrapping_mul(0x9E3779B97F4A7C15).to_le_bytes());
    Id::new(raw).unwrap()
}

fn fake_doc(rng: &mut Rng, vocab: usize, n_words: usize) -> String {
    let mut words = Vec::with_capacity(n_words);
    for _ in 0..n_words {
        let r = rng.next() as f64 / u64::MAX as f64;
        // Zipf-ish skew via squaring keeps common terms common.
        let biased = r * r;
        let idx = ((biased * vocab as f64) as usize).min(vocab - 1);
        words.push(format!("w{idx}"));
    }
    words.join(" ")
}

fn bench(n_docs: usize, vocab: usize, doc_len: usize) {
    let mut rng = Rng(0xC0FFEE + n_docs as u64);
    // Materialise the docs once and reuse across serial + parallel
    // build paths so the measured time is build-only, not doc-gen.
    let docs: Vec<(u64, String)> = (0..n_docs)
        .map(|i| (i as u64 + 1, fake_doc(&mut rng, vocab, doc_len)))
        .collect();

    let fresh_builder = || {
        let mut b = BM25Builder::new();
        for (id_u64, doc) in &docs {
            b.insert(id_from_u64(*id_u64), hash_tokens(doc));
        }
        b
    };

    // Single-threaded build.
    let t0 = Instant::now();
    let naive = fresh_builder().build_with_threads(1);
    let build_ms_serial = t0.elapsed().as_secs_f64() * 1000.0;

    // Parallel build. 4 threads is a typical laptop-class sweet
    // spot; push higher and the merge cost starts to eat the win.
    let threads = 4;
    let t_par = Instant::now();
    let parallel_naive = fresh_builder().build_with_threads(threads);
    let build_ms_par = t_par.elapsed().as_secs_f64() * 1000.0;
    // Byte-identical output is the load-bearing invariant.
    debug_assert_eq!(naive.to_bytes(), parallel_naive.to_bytes());

    let t1 = Instant::now();
    let succinct = SuccinctBM25Index::from_naive(&naive).unwrap();
    let encode_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let naive_bytes = naive.to_bytes();
    let succinct_bytes = succinct.to_bytes();

    let ratio = succinct_bytes.len() as f64 / naive_bytes.len() as f64;
    let speedup = build_ms_serial / build_ms_par;

    println!(
        "n={n_docs:>6}  vocab={vocab:>5}  avg_doc_len={doc_len:>3} \
         | build-1 {build_ms_serial:>5.0}ms  build-{threads} {build_ms_par:>5.0}ms \
         ({speedup:>3.1}×)  succinct-encode {encode_ms:>5.0}ms \
         | naive {:>8}  SB25 {:>8}  ratio {:.2}×",
        fmt_bytes(naive_bytes.len()),
        fmt_bytes(succinct_bytes.len()),
        ratio,
    );
}

fn fmt_bytes(n: usize) -> String {
    if n >= 1 << 20 {
        format!("{:.1} MiB", n as f64 / (1 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KiB", n as f64 / (1 << 10) as f64)
    } else {
        format!("{n} B")
    }
}

fn main() {
    println!("BM25 blob size: naive vs SB25 (succinct)");
    println!(
        "-----------------------------------------------------------------\
         ----------------"
    );
    bench(1_000, 400, 24);
    bench(5_000, 1_000, 48);
    bench(10_000, 2_000, 64);
    bench(50_000, 5_000, 96);
}
