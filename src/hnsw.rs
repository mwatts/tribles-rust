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

use triblespace::core::blob::BlobSchema;
use triblespace::core::id::{Id, RawId};
use triblespace::core::id_hex;
use triblespace::core::metadata::ConstId;

use crate::FORMAT_VERSION;

// Byte format for the flat k-NN blob. A separate magic from the
// eventual proper-HNSW blob ("FLAT" vs "HNSW") — they're
// different on-disk shapes (flat = no layers, no graph).
const FLAT_MAGIC: u32 = u32::from_le_bytes(*b"FLAT");
const FLAT_HEADER_LEN: usize = 32;

/// Errors produced by [`FlatIndex::try_from_bytes`].
#[derive(Debug, Clone, PartialEq)]
pub enum FlatLoadError {
    ShortHeader,
    BadMagic,
    VersionMismatch(u16),
    TruncatedSection(&'static str),
    NilDocId,
}

impl std::fmt::Display for FlatLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortHeader => write!(f, "FLAT blob shorter than header"),
            Self::BadMagic => write!(f, "FLAT blob: magic mismatch"),
            Self::VersionMismatch(v) => {
                write!(f, "FLAT blob: version {v} (expected {})", FORMAT_VERSION)
            }
            Self::TruncatedSection(n) => write!(f, "FLAT blob: truncated section `{n}`"),
            Self::NilDocId => write!(f, "FLAT blob: nil doc_id found"),
        }
    }
}

impl std::error::Error for FlatLoadError {}

/// Content-addressed `BlobSchema` marker for the (eventual)
/// succinct HNSW index. The schema id is fixed now — minted via
/// `trible genid` — even though the byte layout will land in a
/// later iteration. Consumers that reference the id in metadata
/// can do so safely today and pick up the real blob bytes when
/// the schema is implemented.
pub enum SuccinctHNSWIndex {}

impl ConstId for SuccinctHNSWIndex {
    const ID: Id = id_hex!("1D235813CE96AC70B8A4D0490810D720");
}

impl BlobSchema for SuccinctHNSWIndex {}

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
#[derive(Debug, Clone)]
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

impl FlatIndex {
    /// Serialize to a self-contained little-endian byte buffer.
    ///
    /// Layout:
    /// ```text
    /// [32 B header]
    ///   magic u32 = "FLAT"
    ///   version u16
    ///   reserved u16
    ///   n_docs u32
    ///   dim u32
    ///   reserved 12 B
    /// [n_docs × 16 B doc_ids]
    /// [n_docs × dim × 4 B f32 vectors, row-major]
    /// ```
    ///
    /// Vectors are stored post-normalization (the same form
    /// `insert` stored), so round-trip preserves query results
    /// exactly — no re-normalization needed on load.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_docs = self.doc_ids.len() as u32;
        let dim = self.dim as u32;

        let mut buf = Vec::with_capacity(
            FLAT_HEADER_LEN
                + self.doc_ids.len() * 16
                + self.vectors.len() * 4,
        );

        buf.extend_from_slice(&FLAT_MAGIC.to_le_bytes()); // 4
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes()); // 2
        buf.extend_from_slice(&0u16.to_le_bytes()); // 2 reserved
        buf.extend_from_slice(&n_docs.to_le_bytes()); // 4
        buf.extend_from_slice(&dim.to_le_bytes()); // 4
        buf.extend_from_slice(&[0u8; 16]); // 16 reserved → total 32
        debug_assert_eq!(buf.len(), FLAT_HEADER_LEN);

        for id in &self.doc_ids {
            let raw: &RawId = id.as_ref();
            buf.extend_from_slice(raw);
        }
        for &v in &self.vectors {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Reload an index previously produced by [`to_bytes`].
    ///
    /// [`to_bytes`]: Self::to_bytes
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, FlatLoadError> {
        use FlatLoadError as E;

        if bytes.len() < FLAT_HEADER_LEN {
            return Err(E::ShortHeader);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != FLAT_MAGIC {
            return Err(E::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(E::VersionMismatch(version));
        }
        let n_docs = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

        let mut pos = FLAT_HEADER_LEN;
        let doc_ids_end = pos + n_docs * 16;
        if bytes.len() < doc_ids_end {
            return Err(E::TruncatedSection("doc_ids"));
        }
        let mut doc_ids = Vec::with_capacity(n_docs);
        for chunk in bytes[pos..doc_ids_end].chunks_exact(16) {
            let raw: RawId = chunk.try_into().unwrap();
            doc_ids.push(Id::new(raw).ok_or(E::NilDocId)?);
        }
        pos = doc_ids_end;

        let vectors_end = pos + n_docs * dim * 4;
        if bytes.len() < vectors_end {
            return Err(E::TruncatedSection("vectors"));
        }
        let vectors: Vec<f32> = bytes[pos..vectors_end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(Self {
            dim,
            doc_ids,
            vectors,
        })
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

    fn sample_flat() -> FlatIndex {
        let mut b = FlatBuilder::new(3);
        b.insert(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        b.insert(id(2), vec![0.0, 1.0, 0.0]).unwrap();
        b.insert(id(3), vec![0.5, 0.5, 0.0]).unwrap();
        b.build()
    }

    #[test]
    fn flat_bytes_round_trip() {
        let original = sample_flat();
        let reloaded =
            FlatIndex::try_from_bytes(&original.to_bytes()).expect("valid blob");
        assert_eq!(reloaded.dim(), original.dim());
        assert_eq!(reloaded.doc_count(), original.doc_count());

        // Query results must match exactly.
        let q = vec![1.0, 0.0, 0.0];
        let a = original.similar(&q, 3);
        let b = reloaded.similar(&q, 3);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.0, y.0);
            assert!((x.1 - y.1).abs() < 1e-6);
        }
    }

    #[test]
    fn flat_blob_is_deterministic() {
        let a = sample_flat().to_bytes();
        let b = sample_flat().to_bytes();
        assert_eq!(a, b);
    }

    #[test]
    fn flat_short_header_rejected() {
        let err = FlatIndex::try_from_bytes(&[0; 8]).unwrap_err();
        assert_eq!(err, FlatLoadError::ShortHeader);
    }

    #[test]
    fn flat_bad_magic_rejected() {
        let mut bytes = sample_flat().to_bytes();
        bytes[0] = b'X';
        let err = FlatIndex::try_from_bytes(&bytes).unwrap_err();
        assert_eq!(err, FlatLoadError::BadMagic);
    }

    #[test]
    fn flat_truncation_rejected() {
        let bytes = sample_flat().to_bytes();
        let err = FlatIndex::try_from_bytes(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(matches!(err, FlatLoadError::TruncatedSection(_)));
    }
}
