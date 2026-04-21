//! Approximate nearest-neighbour search over caller-supplied
//! embeddings.
//!
//! `SuccinctHNSWIndex` is the content-addressed blob; `HNSWIndex`
//! is the zero-copy view produced by `try_from_blob`.
//!
//! The succinct encoding uses one wavelet matrix per HNSW layer
//! for `(source, neighbour)` pairs — same RING approach as
//! `SuccinctArchive`'s trible graph, but unlabeled (no predicate
//! column), so we only pay for one wavelet matrix per layer
//! instead of three.

// Stubbed — implementation lands in subsequent /loop iterations.
// See `docs/DESIGN.md` §`SuccinctHNSWIndex blob layout` for the
// target byte layout.

/// Placeholder for the content-addressed blob type.
pub struct SuccinctHNSWIndex;

/// Zero-copy view produced by `try_from_blob`.
///
/// Not yet implemented.
pub struct HNSWIndex;
