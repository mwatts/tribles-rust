//! Content-addressed BM25 + HNSW indexes on top of triblespace
//! piles. See `docs/DESIGN.md` for the full design rationale.
//!
//! Two blob types, loaded zero-copy via [`anybytes`]:
//! - [`bm25::SuccinctBM25Index`] / [`bm25::BM25Index`] —
//!   term → doc retrieval where terms are 32-byte triblespace
//!   `Value`s (text tokens, entity ids, tags, anything).
//! - [`hnsw::SuccinctHNSWIndex`] / [`hnsw::HNSWIndex`] — approximate
//!   k-nearest-neighbour over caller-supplied embeddings.
//!
//! Both indexes are rebuilt-and-replaced (no mutation); the
//! caller persists the resulting handle wherever appropriate
//! (branch metadata, commit metadata, a plain trible, or an
//! in-memory cache). See the design doc for layout specifics.

pub mod bm25;
pub mod constraint;
pub mod hnsw;
pub mod schemas;
pub mod tokens;

/// Current on-disk format version for every index blob.
///
/// Bumped on any byte-layout change; `try_from_blob` refuses
/// older/newer blobs rather than silently misreading them.
pub const FORMAT_VERSION: u16 = 1;
