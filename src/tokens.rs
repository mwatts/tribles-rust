//! Opt-in helpers for turning strings into the 32-byte
//! triblespace `Value`s that `bm25::BM25Index` uses as term ids.
//!
//! Nothing in `bm25::` or `hnsw::` depends on this module —
//! callers who have their own tokenizer (language-specific
//! stemming, typst/code-aware splitting, phrase handling) can
//! feed `Value`s directly and skip these helpers entirely.

// Stubbed — a minimal whitespace+lowercase+ASCII-fold tokenizer
// lands in the next iteration, producing `Vec<Value>` via
// Blake3 hashing.

/// Tokenize `text` with a simple whitespace-and-lowercase scheme
/// and return each token as a 32-byte Blake3 hash suitable for
/// use as a `bm25::BM25Index` term value.
///
/// Not yet implemented.
pub fn hash_tokens(_text: &str) -> Vec<[u8; 32]> {
    Vec::new()
}
