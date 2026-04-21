//! Opt-in helpers for turning strings into the 32-byte
//! triblespace `Value`s that `bm25::BM25Index` uses as term ids.
//!
//! Nothing in `bm25::` or `hnsw::` depends on this module —
//! callers who have their own tokenizer (language-specific
//! stemming, typst/code-aware splitting, phrase handling) can
//! feed `Value`s directly and skip these helpers entirely.

use triblespace::core::value::schemas::UnknownValue;
use triblespace::core::value::{RawValue, Value};

/// Tokenize `text` with a simple whitespace-and-lowercase scheme
/// and return each token as a 32-byte Blake3 hash suitable for
/// use as a `bm25::BM25Index` term value.
///
/// Rules:
/// - Split on ASCII whitespace (`char::is_ascii_whitespace`).
/// - Trim leading/trailing ASCII punctuation from each token.
/// - Lowercase ASCII letters; leave non-ASCII bytes as-is.
/// - Drop empty tokens (after trimming).
/// - Duplicates are preserved — the index uses term frequency.
///
/// The hashing is fixed (Blake3) so the same token produces the
/// same 32-byte value across processes and crate versions. That
/// matters because a `bm25::SuccinctBM25Index` stores these
/// hashes directly; callers who want language-aware tokenization
/// should write their own `&str -> Vec<RawValue>` function and
/// skip this helper.
///
/// # Example
///
/// ```
/// # use triblespace_search::tokens::hash_tokens;
/// let vs = hash_tokens("Hello, WORLD — hello.");
/// // "hello" appears twice with the same hash; "world" once.
/// assert_eq!(vs.len(), 3);
/// assert_eq!(vs[0], vs[2]);
/// assert_ne!(vs[0], vs[1]);
/// ```
pub fn hash_tokens(text: &str) -> Vec<RawValue> {
    text.split_ascii_whitespace()
        .filter_map(|raw| {
            let trimmed = raw.trim_matches(|c: char| c.is_ascii_punctuation());
            // Drop tokens with no alphanumeric content at all (em-
            // dashes, pure-symbol clusters). Pure-punctuation
            // tokens would otherwise all hash to the same value
            // and poison the term list.
            if !trimmed.chars().any(|c| c.is_alphanumeric()) {
                return None;
            }
            let mut lower = String::with_capacity(trimmed.len());
            for c in trimmed.chars() {
                lower.push(c.to_ascii_lowercase());
            }
            Some(*blake3::hash(lower.as_bytes()).as_bytes())
        })
        .collect()
}

/// Tagged wrapper: hash the tokens and wrap each one as a
/// `Value<UnknownValue>` for callers that want the triblespace
/// `Value` type rather than the raw byte array.
pub fn hash_tokens_as_values(text: &str) -> Vec<Value<UnknownValue>> {
    hash_tokens(text)
        .into_iter()
        .map(Value::<UnknownValue>::new)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_whitespace() {
        let tokens = hash_tokens("one two three");
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn case_insensitive() {
        let a = hash_tokens("FOO");
        let b = hash_tokens("foo");
        assert_eq!(a, b);
    }

    #[test]
    fn strips_punctuation() {
        let a = hash_tokens("hello,");
        let b = hash_tokens("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn preserves_duplicates() {
        let tokens = hash_tokens("foo bar foo");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], tokens[2]);
    }

    #[test]
    fn drops_empty_tokens() {
        // Pure-punctuation tokens disappear after trimming.
        let tokens = hash_tokens("foo  ---  bar");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn stable_hash() {
        // Regression guard: the Blake3 encoding of "hello" must
        // not drift across crate versions.
        let tokens = hash_tokens("hello");
        let expected = *blake3::hash(b"hello").as_bytes();
        assert_eq!(tokens[0], expected);
    }
}
