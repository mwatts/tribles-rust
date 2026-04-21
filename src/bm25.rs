//! BM25-style lexical / associative retrieval.
//!
//! `SuccinctBM25Index` is the content-addressed blob; `BM25Index`
//! is the zero-copy view produced by `try_from_blob`.
//!
//! Terms are 32-byte triblespace `Value`s. Callers supply term
//! values however they want — [`crate::tokens::hash_tokens`] is
//! one opt-in helper that Blake3-hashes whitespace-separated
//! tokens, but the index is term-source-agnostic.

// Stubbed — implementation lands in subsequent /loop iterations.
// See `docs/DESIGN.md` §`SuccinctBM25Index blob layout` for the
// target byte layout.

/// Placeholder for the content-addressed blob type. Will carry
/// its own `ValueSchema` / `BlobSchema` id (to be minted via
/// `trible genid` when the blob body stabilizes).
pub struct SuccinctBM25Index;

/// Zero-copy view produced by `try_from_blob`.
///
/// Not yet implemented.
pub struct BM25Index;
