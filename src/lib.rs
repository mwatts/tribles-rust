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
//! # Quickstart
//!
//! ```
//! use triblespace::core::and;
//! use triblespace::core::find;
//! use triblespace::core::id::Id;
//!
//! use triblespace_search::bm25::BM25Builder;
//! use triblespace_search::succinct::SuccinctBM25Index;
//! use triblespace_search::tokens::hash_tokens;
//!
//! // 1. Build an in-memory index.
//! let mut b = BM25Builder::new();
//! b.insert_id(Id::new([1; 16]).unwrap(), hash_tokens("the quick brown fox"));
//! b.insert_id(Id::new([2; 16]).unwrap(), hash_tokens("the lazy brown dog"));
//! b.insert_id(Id::new([3; 16]).unwrap(), hash_tokens("quick silver fox"));
//!
//! // 2. Flip to the succinct form for persistence + smaller bytes.
//! let idx = SuccinctBM25Index::from_naive(&b.build()).unwrap();
//!
//! // 3. Query through the normal engine — the constraint
//! //    plugs into `find!` / `and!` / `pattern!` unchanged.
//! let fox = hash_tokens("fox")[0];
//! let docs: Vec<(Id,)> = find!(
//!     (doc: Id),
//!     idx.docs_containing(doc, fox)
//! ).collect();
//! assert_eq!(docs.len(), 2);
//! ```
//!
//! See the `examples/` directory for TribleSet composition,
//! vector similarity, blob-size benchmarks, and phrase search.
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
///
/// History:
/// - `1`: initial layout. `doc_ids` was a 16-byte `Id` table.
/// - `2`: SB25 only — generalized `doc_ids` → `keys`: 32-byte
///   `RawValue` per entry. Unlocks string / tag / composite-key
///   search through the same index type. See CHANGELOG.
/// - `3`: SB25 keys table → `CompressedUniverse` (DACs-byte
///   fragment dictionary). Compresses correlated keys (entity
///   ids with shared zero-padding, sequential patterns).
///   Postings' `doc_idx` now references the universe code
///   (sorted position), not insertion order. Header grows by
///   40 B for the `CompressedUniverseMeta`.
/// - `4`: HNSW + FLAT blobs: same `doc_ids` → 32-byte `keys`
///   generalization SB25 got in v2. `HNSWBuilder::insert` and
///   `FlatBuilder::insert` now take `RawValue`; `insert_id` /
///   `insert_value<S>` are the typed wrappers. `similar()`
///   returns `(RawValue, f32)`; `similar_ids()` is the GenId
///   decoder. Doubles the keys section (16 B → 32 B), ~2 % of
///   the HNSW blob at typical embedding sizes.
pub const FORMAT_VERSION: u16 = 4;
