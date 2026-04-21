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

/// Character-level n-gram tokenizer. Returns one hashed term per
/// sliding window of `n` characters across the lowercased text.
///
/// Indexing the same document with *both* `hash_tokens` and
/// `ngram_tokens(text, 3)` (concatenating the two `Vec`s before
/// `BM25Builder::insert`) lets the same BM25 index serve:
/// - whole-word queries via `hash_tokens("foo")`, and
/// - prefix / typo queries via `ngram_tokens("fox", 3)` — any
///   shared 3-gram boosts the score even when the surface forms
///   differ.
///
/// Rules:
/// - Non-alphanumeric characters are replaced by a single space
///   (so n-grams never cross punctuation/whitespace boundaries).
/// - Letters are lowercased.
/// - Runs shorter than `n` characters are dropped — no padding.
/// - `n == 0` returns an empty `Vec`.
///
/// Each n-gram is namespaced to `n` before hashing, so a trigram
/// `"fox"` and a bigram `"fo"` + `"ox"` produce distinct term
/// values and can coexist in one index.
///
/// # Example
///
/// ```
/// # use triblespace_search::tokens::ngram_tokens;
/// let tris = ngram_tokens("fox", 3);
/// assert_eq!(tris.len(), 1); // just "fox"
///
/// let tris = ngram_tokens("foxes", 3);
/// // "fox", "oxe", "xes"
/// assert_eq!(tris.len(), 3);
///
/// // "fox" and "foxes" share the "fox" trigram.
/// assert!(ngram_tokens("foxes", 3).contains(&ngram_tokens("fox", 3)[0]));
/// ```
pub fn ngram_tokens(text: &str, n: usize) -> Vec<RawValue> {
    if n == 0 {
        return Vec::new();
    }

    // Normalize: lowercase letters, replace other non-alphanumerics
    // with a single space so runs don't merge across boundaries.
    let mut normalized = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_alphanumeric() {
            for l in c.to_lowercase() {
                normalized.push(l);
            }
        } else {
            normalized.push(' ');
        }
    }

    let mut out = Vec::new();
    for run in normalized.split_ascii_whitespace() {
        let chars: Vec<char> = run.chars().collect();
        if chars.len() < n {
            continue;
        }
        // Namespace by n so {n=2 "fo"} and {n=3 "foo"} don't collide.
        let mut gram = String::with_capacity(n * 4 + 8);
        for window in chars.windows(n) {
            gram.clear();
            gram.push_str(&n.to_string());
            gram.push(':');
            for &c in window {
                gram.push(c);
            }
            out.push(*blake3::hash(gram.as_bytes()).as_bytes());
        }
    }
    out
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

    #[test]
    fn ngram_empty_n_returns_nothing() {
        assert!(ngram_tokens("anything", 0).is_empty());
    }

    #[test]
    fn ngram_skips_short_runs() {
        // "hi" (len 2) drops for n=3.
        assert!(ngram_tokens("hi", 3).is_empty());
    }

    #[test]
    fn ngram_counts() {
        // "foxes" -> fox, oxe, xes = 3 trigrams
        assert_eq!(ngram_tokens("foxes", 3).len(), 3);
        // "foxes" -> fo, ox, xe, es = 4 bigrams
        assert_eq!(ngram_tokens("foxes", 2).len(), 4);
    }

    #[test]
    fn ngram_case_insensitive() {
        let a = ngram_tokens("FOX", 3);
        let b = ngram_tokens("fox", 3);
        assert_eq!(a, b);
    }

    #[test]
    fn ngram_does_not_cross_punctuation() {
        // "foo-bar" splits on '-' so the tri-gram window doesn't
        // span the boundary (no "oo-" or "o-b" grams).
        let dashed = ngram_tokens("foo-bar", 3);
        let spaced = ngram_tokens("foo bar", 3);
        assert_eq!(dashed, spaced);
        assert_eq!(dashed.len(), 2); // "foo" and "bar"
    }

    #[test]
    fn ngram_size_namespaced() {
        // "fo" as a bigram and "fo" as a prefix of a trigram
        // produce distinct hashes — same glyphs, different n.
        let bi = ngram_tokens("fo", 2);
        let tri = ngram_tokens("foo", 3);
        assert_eq!(bi.len(), 1);
        assert_eq!(tri.len(), 1);
        assert_ne!(bi[0], tri[0]);
    }

    #[test]
    fn ngram_shared_prefix_matches_extension() {
        // The key property: "fox" and "foxes" share a trigram, so
        // a BM25 index keyed on ngram_tokens would score them
        // relative to each other — prefix / fuzzy matching for
        // free.
        let short = ngram_tokens("fox", 3);
        let long = ngram_tokens("foxes", 3);
        assert!(long.contains(&short[0]));
    }
}
