//! Approximate nearest-neighbour search over caller-supplied
//! embeddings.
//!
//! `SuccinctHNSWIndex` is the content-addressed blob;
//! `HNSWIndex` is the zero-copy view produced by `try_from_blob`.
//!
//! The succinct encoding uses one wavelet matrix per HNSW layer
//! for `(source, neighbour)` pairs — same RING approach as
//! `SuccinctArchive`'s trible graph, but unlabeled (no predicate
//! column), so we only pay for one wavelet matrix per layer
//! instead of three.
//!
//! # Current status
//!
//! This module ships a **flat** k-NN stand-in ([`FlatIndex`])
//! first — brute-force cosine over all vectors. Correct, simple,
//! useful for ≤ 100k docs, and exercises the same builder/query
//! API that the proper HNSW graph (next iteration) will
//! implement. That keeps callers' code stable while we swap the
//! inner data structure.
//!
//! # Build and query
//!
//! ```
//! # use triblespace_search::hnsw::FlatBuilder;
//! # use triblespace::core::id::Id;
//! let mut b = FlatBuilder::new(4);
//! b.insert(Id::new([1; 16]).unwrap(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
//! b.insert(Id::new([2; 16]).unwrap(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();
//! b.insert(Id::new([3; 16]).unwrap(), vec![0.9, 0.1, 0.0, 0.0]).unwrap();
//! let idx = b.build();
//!
//! let query = vec![1.0, 0.0, 0.0, 0.0];
//! let hits = idx.similar(&query, 2);
//! assert_eq!(hits.len(), 2);
//! // doc 1 is an exact match, doc 3 nearly so.
//! assert_eq!(hits[0].0, Id::new([1; 16]).unwrap());
//! assert_eq!(hits[1].0, Id::new([3; 16]).unwrap());
//! ```

use triblespace::core::id::Id;

/// Placeholder for the content-addressed succinct HNSW blob.
pub struct SuccinctHNSWIndex;

/// Placeholder for the zero-copy succinct HNSW view.
pub struct HNSWIndex;

/// Caller tried to insert a vector whose length disagrees with
/// the index's configured dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimMismatch {
    pub expected: usize,
    pub got: usize,
}

impl std::fmt::Display for DimMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "embedding dimensionality mismatch: expected {}, got {}",
            self.expected, self.got
        )
    }
}

impl std::error::Error for DimMismatch {}

/// L2-normalize `v` in place. Zero vectors are left untouched.
fn normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    if norm_sq > 0.0 {
        let inv = 1.0 / norm_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Dot product. Assumes both slices have equal length.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Builder for a flat (brute-force) k-NN index.
///
/// All vectors are L2-normalized at insert time so the distance
/// metric at query time is exact cosine similarity (`dot(q, v) =
/// cos(q, v)` for unit vectors). Pre-normalizing moves the
/// division into the build pass and keeps the query hot path a
/// single dot product per doc.
pub struct FlatBuilder {
    dim: usize,
    doc_ids: Vec<Id>,
    /// Row-major; row `i` is `vectors[i * dim .. (i + 1) * dim]`.
    vectors: Vec<f32>,
}

impl FlatBuilder {
    /// Start a fresh builder for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "FlatBuilder: dim must be > 0");
        Self {
            dim,
            doc_ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Insert one vector. Returns `Err(DimMismatch)` if `vec` is
    /// the wrong length. The caller's vector is L2-normalized in
    /// place before storage.
    pub fn insert(&mut self, doc_id: Id, mut vec: Vec<f32>) -> Result<(), DimMismatch> {
        if vec.len() != self.dim {
            return Err(DimMismatch {
                expected: self.dim,
                got: vec.len(),
            });
        }
        normalize(&mut vec);
        self.doc_ids.push(doc_id);
        self.vectors.extend_from_slice(&vec);
        Ok(())
    }

    /// Consume the builder and produce a flat index.
    pub fn build(self) -> FlatIndex {
        FlatIndex {
            dim: self.dim,
            doc_ids: self.doc_ids,
            vectors: self.vectors,
        }
    }

    /// Number of vectors inserted so far.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// `true` if no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Configured embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Brute-force k-NN index. O(n · d) per query; use when the
/// corpus is small enough for that to be fine, or as a baseline
/// against which to validate a later HNSW implementation.
///
/// Scores are cosine similarity in `[-1, 1]`; higher is better.
pub struct FlatIndex {
    dim: usize,
    doc_ids: Vec<Id>,
    vectors: Vec<f32>,
}

impl FlatIndex {
    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Return the top `k` documents by cosine similarity to
    /// `query`. `query` is L2-normalized internally before
    /// scoring. Returns fewer than `k` results if the index has
    /// fewer docs; returns an empty vec on dim mismatch.
    pub fn similar(&self, query: &[f32], k: usize) -> Vec<(Id, f32)> {
        if query.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let mut q = query.to_vec();
        normalize(&mut q);

        // Score every doc; retain top-k via a simple bounded
        // heap. For a flat index the O(n log k) heap is already
        // much cheaper than the O(n · d) scoring pass, so we
        // don't bother with a fancier selection algorithm.
        let mut heap: std::collections::BinaryHeap<MinScored> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for (i, doc_id) in self.doc_ids.iter().enumerate() {
            let row = &self.vectors[i * self.dim..(i + 1) * self.dim];
            let score = dot(&q, row);
            heap.push(MinScored { id: *doc_id, score });
            if heap.len() > k {
                heap.pop();
            }
        }
        let mut out: Vec<(Id, f32)> = heap.into_iter().map(|m| (m.id, m.score)).collect();
        out.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

/// A `(score, id)` wrapper whose `Ord` impl inverts score so
/// pushing into a max-heap of capacity `k` yields a min-heap
/// over scores — the top-k retention trick.
#[derive(Clone, Copy)]
struct MinScored {
    id: Id,
    score: f32,
}

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for MinScored {}

impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Invert so `BinaryHeap` behaves as a min-heap over score.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    #[test]
    fn exact_match_is_top() {
        let mut b = FlatBuilder::new(3);
        b.insert(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        b.insert(id(2), vec![0.0, 1.0, 0.0]).unwrap();
        b.insert(id(3), vec![0.0, 0.0, 1.0]).unwrap();
        let idx = b.build();

        let hits = idx.similar(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id(1));
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ranked_by_similarity() {
        let mut b = FlatBuilder::new(2);
        b.insert(id(1), vec![1.0, 0.0]).unwrap(); // closest
        b.insert(id(2), vec![0.9, 0.1]).unwrap();
        b.insert(id(3), vec![0.0, 1.0]).unwrap();
        let idx = b.build();

        let hits = idx.similar(&[1.0, 0.0], 3);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, id(1));
        assert_eq!(hits[1].0, id(2));
        assert_eq!(hits[2].0, id(3));
        // Scores are monotonically non-increasing.
        assert!(hits[0].1 >= hits[1].1 && hits[1].1 >= hits[2].1);
    }

    #[test]
    fn normalizes_input_vectors() {
        // Two inputs that are parallel but scaled differently
        // must yield identical scores against any query.
        let mut b = FlatBuilder::new(2);
        b.insert(id(1), vec![3.0, 0.0]).unwrap();
        b.insert(id(2), vec![100.0, 0.0]).unwrap();
        let idx = b.build();

        let hits = idx.similar(&[1.0, 0.0], 2);
        assert!((hits[0].1 - hits[1].1).abs() < 1e-5);
    }

    #[test]
    fn dim_mismatch_rejected() {
        let mut b = FlatBuilder::new(3);
        let err = b.insert(id(1), vec![1.0, 0.0]).unwrap_err();
        assert_eq!(err.expected, 3);
        assert_eq!(err.got, 2);
    }

    #[test]
    fn empty_index_is_queryable() {
        let idx = FlatBuilder::new(4).build();
        assert_eq!(idx.similar(&[0.0, 0.0, 0.0, 0.0], 3), vec![]);
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut b = FlatBuilder::new(2);
        b.insert(id(1), vec![1.0, 0.0]).unwrap();
        let idx = b.build();
        assert!(idx.similar(&[1.0, 0.0], 0).is_empty());
    }

    #[test]
    fn wrong_dim_query_returns_empty() {
        let mut b = FlatBuilder::new(3);
        b.insert(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        let idx = b.build();
        assert!(idx.similar(&[1.0, 0.0], 1).is_empty()); // dim 2 vs 3
    }

    #[test]
    fn k_larger_than_corpus_truncates() {
        let mut b = FlatBuilder::new(2);
        b.insert(id(1), vec![1.0, 0.0]).unwrap();
        b.insert(id(2), vec![0.0, 1.0]).unwrap();
        let idx = b.build();
        let hits = idx.similar(&[1.0, 0.0], 10);
        assert_eq!(hits.len(), 2);
    }
}
