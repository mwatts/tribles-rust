//! Content-addressed BM25 + HNSW indexes on top of triblespace
//! piles. See `docs/DESIGN.md` for the full design rationale.
//!
//! Two canonical blob types, loaded zero-copy via [`anybytes`]
//! with bit-packed bodies via [`jerky`]:
//! - [`succinct::SuccinctBM25Index`] (schema
//!   [`succinct::SuccinctBM25Blob`]) — term → doc retrieval
//!   where terms are 32-byte triblespace `Value`s (text tokens,
//!   entity ids, tags, anything).
//! - [`succinct::SuccinctHNSWIndex`] (schema
//!   [`succinct::SuccinctHNSWBlob`]) — approximate
//!   k-nearest-neighbour over caller-supplied embeddings.
//!
//! [`bm25::BM25Index`] and [`hnsw::HNSWIndex`] are the naive
//! in-memory builders; convert to the succinct forms via
//! `SuccinctBM25Index::from_naive` /
//! `SuccinctHNSWIndex::from_naive` and persist those.
//!
//! Both indexes are rebuilt-and-replaced (no mutation); the
//! caller persists the resulting handle wherever appropriate
//! (branch metadata, commit metadata, a plain trible, or an
//! in-memory cache).
//!
//! [`jerky`]: https://docs.rs/jerky

pub mod bm25;
pub mod constraint;
pub mod hnsw;
pub mod schemas;
#[cfg(feature = "succinct")]
pub mod succinct;
pub mod tokens;

/// Current on-disk format version for every index blob.
///
/// Bumped on any byte-layout change; `try_from_blob` refuses
/// older/newer blobs rather than silently misreading them.
pub const FORMAT_VERSION: u16 = 1;
