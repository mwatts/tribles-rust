//! BM25-style lexical / associative retrieval.
//!
//! Terms are 32-byte triblespace `Value`s (as [`RawValue`], the
//! schema-erased byte array). Callers supply term values however
//! they want — [`crate::tokens::hash_tokens`] is one opt-in
//! helper that Blake3-hashes whitespace-separated tokens, but the
//! index is term-source-agnostic:
//!
//! | Term source | What this gets you |
//! | --- | --- |
//! | `hash(word)` | classic text search |
//! | entity `Id` | "docs mentioning this person" |
//! | tag `Id` | tag-weighted search |
//! | fragment `Id` | "docs citing this fragment" |
//!
//! The same schema handles all four.
//!
//! # Build and query
//!
//! ```
//! # use triblespace_search::bm25::BM25Builder;
//! # use triblespace_search::tokens::hash_tokens;
//! # use triblespace::core::id::Id;
//! let docs = [
//!     (Id::new([1; 16]).unwrap(), "the quick brown fox"),
//!     (Id::new([2; 16]).unwrap(), "the lazy brown dog"),
//!     (Id::new([3; 16]).unwrap(), "quick silver fox"),
//! ];
//! let mut b = BM25Builder::new();
//! for (id, text) in &docs {
//!     b.insert(*id, hash_tokens(text));
//! }
//! let index = b.build();
//!
//! // Query: how many docs mention "fox"?
//! let q = hash_tokens("fox");
//! let hits: Vec<_> = index.query_term(&q[0]).collect();
//! assert_eq!(hits.len(), 2);
//! ```
//!
//! # Current status
//!
//! This is the **naive** (non-succinct) implementation:
//! sorted-term table + flat `Vec<(doc_idx, score)>` postings.
//! Correctness first; the jerky/wavelet-matrix-backed succinct
//! version swaps in later behind the same public API.
//!
//! See `docs/DESIGN.md` for the target blob layout.

use std::collections::HashMap;

use triblespace::core::id::Id;
use triblespace::core::value::RawValue;

/// Classic BM25 tuning. Defaults match Robertson & Zaragoza 2009.
const DEFAULT_K1: f32 = 1.5;
const DEFAULT_B: f32 = 0.75;

/// Stub for the content-addressed blob type. Will serialize the
/// layout documented in `docs/DESIGN.md` once the byte format is
/// frozen; for now the in-memory [`BM25Index`] is the reference
/// implementation.
pub struct SuccinctBM25Index;

/// Accumulator for documents to be indexed. Call [`insert`] once
/// per doc, then [`build`] to produce a [`BM25Index`].
///
/// [`insert`]: Self::insert
/// [`build`]: Self::build
pub struct BM25Builder {
    docs: Vec<(Id, Vec<RawValue>)>,
    k1: f32,
    b: f32,
}

impl Default for BM25Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl BM25Builder {
    /// Create an empty builder with the standard BM25 tuning.
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        }
    }

    /// Override the `k1` term-frequency saturation parameter.
    pub fn k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Override the `b` length-normalization parameter.
    pub fn b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Add a document. `terms` is the caller's tokenization (see
    /// [`crate::tokens::hash_tokens`] for a simple default). The
    /// order of terms is irrelevant; duplicates contribute to
    /// term frequency.
    pub fn insert(&mut self, doc_id: Id, terms: Vec<RawValue>) {
        self.docs.push((doc_id, terms));
    }

    /// Consume the builder and produce an in-memory BM25 index.
    pub fn build(self) -> BM25Index {
        let Self { docs, k1, b } = self;
        let n_docs = docs.len();

        // Per-doc token count; average doc length for normalization.
        let doc_lens: Vec<u32> = docs.iter().map(|(_, t)| t.len() as u32).collect();
        let avg_doc_len = if n_docs == 0 {
            0.0
        } else {
            doc_lens.iter().map(|&n| n as f64).sum::<f64>() as f32 / n_docs as f32
        };

        // doc_id table and term-frequency counts per (doc_idx, term).
        let doc_ids: Vec<Id> = docs.iter().map(|(id, _)| *id).collect();
        // term -> (doc_idx -> tf)
        let mut term_to_tfs: HashMap<RawValue, HashMap<u32, u32>> = HashMap::new();
        for (doc_idx, (_, terms)) in docs.into_iter().enumerate() {
            for term in terms {
                let entry = term_to_tfs
                    .entry(term)
                    .or_default()
                    .entry(doc_idx as u32)
                    .or_insert(0);
                *entry += 1;
            }
        }

        // Sort terms ascending so the term table supports binary
        // search (matches the future succinct layout).
        let mut terms: Vec<RawValue> = term_to_tfs.keys().copied().collect();
        terms.sort_unstable();

        // Per-term postings with pre-baked BM25 scores. IDF follows
        // the Robertson smoothed form: ln(1 + (N - df + 0.5) /
        // (df + 0.5)).
        let mut offsets: Vec<u32> = Vec::with_capacity(terms.len() + 1);
        offsets.push(0);
        let mut postings: Vec<(u32, f32)> = Vec::new();

        let n = n_docs as f32;
        for term in &terms {
            let tfs = &term_to_tfs[term];
            let df = tfs.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            let mut entries: Vec<(u32, f32)> = tfs
                .iter()
                .map(|(&doc_idx, &tf)| {
                    let tf = tf as f32;
                    let dl = doc_lens[doc_idx as usize] as f32;
                    let norm = if avg_doc_len > 0.0 {
                        1.0 - b + b * (dl / avg_doc_len)
                    } else {
                        1.0
                    };
                    let score = idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
                    (doc_idx, score)
                })
                .collect();
            // Postings sorted by doc_idx for future merge-join.
            entries.sort_unstable_by_key(|&(idx, _)| idx);
            postings.extend(entries);
            offsets.push(postings.len() as u32);
        }

        BM25Index {
            doc_ids,
            doc_lens,
            avg_doc_len,
            terms,
            offsets,
            postings,
            k1,
            b,
        }
    }
}

/// In-memory BM25 index. Call [`BM25Builder::build`] to produce one.
///
/// All scores are pre-baked at build time: per-(doc, term) BM25
/// weight with saturating term frequency (`k1`) and
/// length-normalized doc length (`b`).
pub struct BM25Index {
    doc_ids: Vec<Id>,
    doc_lens: Vec<u32>,
    avg_doc_len: f32,
    terms: Vec<RawValue>,
    offsets: Vec<u32>,
    postings: Vec<(u32, f32)>,
    k1: f32,
    b: f32,
}

impl BM25Index {
    /// Number of documents in the index.
    pub fn doc_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Number of distinct terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Average document length in the corpus.
    pub fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    /// Look up a term's posting list.
    ///
    /// Returns `(doc_id, score)` pairs in ascending-doc-id order.
    /// Empty iterator if the term is absent.
    pub fn query_term(&self, term: &RawValue) -> impl Iterator<Item = (Id, f32)> + '_ {
        let lo = self.terms.binary_search(term).ok();
        let range = match lo {
            Some(i) => {
                self.offsets[i] as usize..self.offsets[i + 1] as usize
            }
            None => 0..0,
        };
        self.postings[range]
            .iter()
            .map(|&(doc_idx, score)| (self.doc_ids[doc_idx as usize], score))
    }

    /// Score a multi-term query as the sum of per-term BM25
    /// weights (standard OR-like bag-of-words).
    ///
    /// Returned `(doc_id, score)` pairs are sorted descending by
    /// score. No top-k truncation — caller slices what they need.
    pub fn query_multi(&self, terms: &[RawValue]) -> Vec<(Id, f32)> {
        let mut acc: HashMap<Id, f32> = HashMap::new();
        for term in terms {
            for (doc, score) in self.query_term(term) {
                *acc.entry(doc).or_insert(0.0) += score;
            }
        }
        let mut out: Vec<(Id, f32)> = acc.into_iter().collect();
        out.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// Number of documents containing this term.
    pub fn doc_frequency(&self, term: &RawValue) -> usize {
        self.terms
            .binary_search(term)
            .ok()
            .map(|i| (self.offsets[i + 1] - self.offsets[i]) as usize)
            .unwrap_or(0)
    }

    /// BM25 `k1` used when this index was built.
    pub fn k1(&self) -> f32 {
        self.k1
    }

    /// BM25 `b` used when this index was built.
    pub fn b(&self) -> f32 {
        self.b
    }

    /// Raw doc-length table. `doc_lens()[i]` is the token count
    /// of the document at internal index `i`.
    pub fn doc_lens(&self) -> &[u32] {
        &self.doc_lens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::hash_tokens;

    fn id(byte: u8) -> Id {
        // `Id::new` returns None for the nil [0; 16] id.
        assert!(byte != 0, "test fixture: 0 is the nil Id");
        Id::new([byte; 16]).unwrap()
    }

    #[test]
    fn empty_index_is_queryable() {
        let idx = BM25Builder::new().build();
        assert_eq!(idx.doc_count(), 0);
        assert_eq!(idx.term_count(), 0);
        let term = [0u8; 32];
        assert!(idx.query_term(&term).next().is_none());
    }

    #[test]
    fn three_docs_basic() {
        let mut b = BM25Builder::new();
        b.insert(id(1), hash_tokens("the quick brown fox"));
        b.insert(id(2), hash_tokens("the lazy brown dog"));
        b.insert(id(3), hash_tokens("quick silver fox"));
        let idx = b.build();
        assert_eq!(idx.doc_count(), 3);

        // "fox" appears in docs 1 and 3.
        let fox = hash_tokens("fox");
        let hits: Vec<_> = idx.query_term(&fox[0]).collect();
        assert_eq!(hits.len(), 2);
        let doc_ids: Vec<_> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&id(1)));
        assert!(doc_ids.contains(&id(3)));

        // "the" is in doc 1 and doc 2.
        let the = hash_tokens("the");
        assert_eq!(idx.doc_frequency(&the[0]), 2);

        // Missing term returns nothing.
        let missing = hash_tokens("banana");
        assert!(idx.query_term(&missing[0]).next().is_none());
    }

    #[test]
    fn idf_prefers_rare_terms() {
        let mut b = BM25Builder::new();
        // "rare" appears once, "common" appears in every doc.
        for i in 1..=10 {
            b.insert(id(i), hash_tokens("common common"));
        }
        b.insert(id(100), hash_tokens("common rare"));
        let idx = b.build();

        let rare = hash_tokens("rare");
        let common = hash_tokens("common");
        let rare_score = idx.query_term(&rare[0]).next().unwrap().1;
        let common_score = idx
            .query_term(&common[0])
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .1;
        assert!(
            rare_score > common_score,
            "rare_score={rare_score}, common_score={common_score}"
        );
    }

    #[test]
    fn term_frequency_saturates() {
        // Two docs, one contains "foo" once, one 100 times. With
        // k1 = 1.5 the second's score should be higher but not
        // 100x higher — saturation.
        let mut b = BM25Builder::new();
        b.insert(id(1), hash_tokens("foo bar baz"));
        let many: String = std::iter::repeat("foo ").take(100).collect::<String>();
        b.insert(id(2), hash_tokens(&many));
        let idx = b.build();

        let foo = hash_tokens("foo");
        let scores: HashMap<Id, f32> = idx.query_term(&foo[0]).collect();
        let s1 = scores[&id(1)];
        let s2 = scores[&id(2)];
        assert!(s2 > s1);
        assert!(
            s2 < s1 * 20.0,
            "tf saturation should keep ratio moderate: {s1} -> {s2}"
        );
    }

    #[test]
    fn multi_term_query_sums() {
        let mut b = BM25Builder::new();
        b.insert(id(1), hash_tokens("quick brown fox"));
        b.insert(id(2), hash_tokens("quick red fox"));
        b.insert(id(3), hash_tokens("slow brown dog"));
        let idx = b.build();

        let q = hash_tokens("quick fox");
        let ranked = idx.query_multi(&q);
        // Docs 1 and 2 have both terms; doc 3 has neither.
        assert_eq!(ranked.len(), 2);
        let top_ids: Vec<_> = ranked.iter().map(|(d, _)| *d).collect();
        assert!(top_ids.contains(&id(1)));
        assert!(top_ids.contains(&id(2)));
        // Results are sorted descending by score.
        assert!(ranked[0].1 >= ranked[1].1);
    }

    #[test]
    fn tuning_params_round_trip() {
        let b = BM25Builder::new().k1(1.2).b(0.5);
        let idx = b.build();
        assert!((idx.k1() - 1.2).abs() < 1e-6);
        assert!((idx.b() - 0.5).abs() < 1e-6);
    }
}
