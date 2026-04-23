//! Compare `SuccinctGraph` (CSR-style, current) vs
//! `RingGraph` (wavelet-matrix 2-ring, Arroyuelo et al. 2024
//! §4.4) on the same HNSW-shaped graph.
//!
//! We build a synthetic layer-0 graph of the same shape an
//! actual HNSW layer would have at the given scale — each
//! node has up to `M0` undirected edges to pseudo-random
//! neighbours — then encode it both ways and report the
//! blob-section sizes.
//!
//! Size is the headline measure here. Query latency is
//! reported too but it's a microbench without warm-up or
//! outlier filtering — same spirit as `blob_sizes_at_scale`.
//!
//! ```sh
//! cargo run --release --example ring_vs_csr
//! ```

use std::collections::HashSet;
use std::time::Instant;

use anybytes::area::ByteArea;

use triblespace_search::ring::RingGraph;
use triblespace_search::succinct::SuccinctGraph;

/// SplitMix64 PRNG — deterministic, no extra deps.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() as u32) % (hi - lo)
    }
}

/// Generate a synthetic HNSW-layer-shaped adjacency list.
/// Each node picks ~M0 distinct neighbours uniformly at
/// random (symmetric: when v picks u, u also gets v). Returns
/// both the CSR-shaped `Vec<Vec<u32>>` and the canonicalized
/// undirected edge list.
fn synth_graph(n: u32, m0: u32, seed: u64) -> (Vec<Vec<u32>>, Vec<(u32, u32)>) {
    let mut rng = Rng(seed);
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for v in 0..n {
        let target_degree = m0;
        // Pick unique targets until we hit the degree cap or
        // run out of distinct candidates.
        let mut tried = 0u32;
        while (adj[v as usize].len() as u32) < target_degree && tried < target_degree * 4 {
            tried += 1;
            let u = rng.range(0, n);
            if u == v {
                continue;
            }
            // Dedup both directions; canonicalize to (low, high).
            let (lo, hi) = if v < u { (v, u) } else { (u, v) };
            if edges.insert((lo, hi)) {
                adj[v as usize].push(u);
                adj[u as usize].push(v);
            }
        }
    }
    let mut edge_list: Vec<(u32, u32)> = edges.into_iter().collect();
    edge_list.sort_unstable();
    (adj, edge_list)
}

fn encode_csr(adj: &[Vec<u32>], n: u32) -> (usize, SuccinctGraph) {
    // Wrap in the layer-major shape SuccinctGraph::build wants.
    // Single layer, so the outer Vec has length 1.
    let layers: Vec<Vec<Vec<u32>>> = vec![adj.to_vec()];
    let (bytes, meta) = SuccinctGraph::build(&layers, n as usize).expect("build CSR");
    let size = bytes.len();
    let graph = SuccinctGraph::from_bytes(meta, bytes).expect("reload CSR");
    (size, graph)
}

fn encode_ring(edges: &[(u32, u32)], n: u32) -> (usize, RingGraph) {
    let mut area = ByteArea::new().unwrap();
    let mut sections = area.sections();
    let (ring, _meta) = RingGraph::build(edges, n as usize, &mut sections).unwrap();
    let _ = sections;
    let bytes = area.freeze().unwrap();
    (bytes.len(), ring)
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

fn bench(n: u32, m0: u32, seed: u64) {
    let (adj, edges) = synth_graph(n, m0, seed);
    let total_directed_edges: usize = adj.iter().map(|v| v.len()).sum();
    let undirected_edges = edges.len();

    println!(
        "\n── n = {n:>6}  M0 = {m0:>2}  (undirected edges = {und}, directed incidences = {dir}) ──",
        und = undirected_edges,
        dir = total_directed_edges,
    );

    // ─ Encoding size ─
    let (csr_size, csr) = encode_csr(&adj, n);
    let (ring_size, ring) = encode_ring(&edges, n);
    let ratio = ring_size as f64 / csr_size as f64;
    println!(
        "   CSR blob:  {:>10}   ({:.1} bits/directed edge)",
        fmt_bytes(csr_size),
        (csr_size * 8) as f64 / total_directed_edges.max(1) as f64,
    );
    println!(
        "   Ring blob: {:>10}   ({:.1} bits/undirected edge) — {:.2}× CSR",
        fmt_bytes(ring_size),
        (ring_size * 8) as f64 / undirected_edges.max(1) as f64,
        ratio,
    );

    // ─ Neighbour-enumeration latency ─
    // Sample K random vertices, enumerate neighbours, time
    // end-to-end. Same K for both encodings so per-call
    // averages are comparable.
    let k = 500;
    let mut rng_q = Rng(seed ^ 0xC0FFEE);
    let sample: Vec<u32> = (0..k).map(|_| rng_q.range(0, n)).collect();

    // Warm-up pass to let CPU caches settle.
    for &v in &sample {
        let _ = ring.neighbours(v as usize).count();
        let _ = csr.neighbours(v as usize, 0).count();
    }
    let t0 = Instant::now();
    let mut ring_total = 0usize;
    for _ in 0..5 {
        for &v in &sample {
            ring_total += ring.neighbours(v as usize).count();
        }
    }
    let ring_ns = t0.elapsed().as_nanos() / (5 * sample.len() as u128);

    let t1 = Instant::now();
    let mut csr_total = 0usize;
    for _ in 0..5 {
        for &v in &sample {
            csr_total += csr.neighbours(v as usize, 0).count();
        }
    }
    let csr_ns = t1.elapsed().as_nanos() / (5 * sample.len() as u128);
    // Degrees must match — ring stores each edge once but yields
    // both ends per incident vertex, so the total neighbour count
    // for v is the same as the CSR directed-both-ways degree.
    assert_eq!(ring_total, csr_total, "ring vs CSR degree totals disagreed");
    println!(
        "   neighbour enumeration:  ring {} ns/query   SuccinctGraph {} ns/query   ratio {:.1}×",
        ring_ns,
        csr_ns,
        ring_ns as f64 / csr_ns.max(1) as f64,
    );
}

fn main() {
    println!("RingGraph (2-ring, §4.4) vs SuccinctGraph (CSR) on a");
    println!("single HNSW-shaped layer. Blob size is the headline; ");
    println!("neighbour-enumeration latency is a rough microbench.");

    bench(1_000, 32, 0x1234);
    bench(10_000, 32, 0x5678);
    bench(100_000, 32, 0xDEAD);
    bench(100_000, 16, 0xBEEF);
}
