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
//!     b.insert_id(*id, hash_tokens(text));
//! }
//! let index = b.build();
//!
//! // Query: how many docs mention "fox"?
//! let q = hash_tokens("fox");
//! let hits: Vec<_> = index.query_term_ids(&q[0]).collect();
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

use triblespace::core::id::{Id, RawId};
use triblespace::core::value::RawValue;

use crate::FORMAT_VERSION;

/// Classic BM25 tuning. Defaults match Robertson & Zaragoza 2009.
const DEFAULT_K1: f32 = 1.5;
const DEFAULT_B: f32 = 0.75;

// ── Byte layout constants ────────────────────────────────────────────

/// Magic four-byte header tag, little-endian.
const MAGIC: u32 = u32::from_le_bytes(*b"BM25");
/// Header length in bytes. Fixed; indices larger than 4 GiB would
/// overflow the u32 length fields anyway.
const HEADER_LEN: usize = 32;

/// Errors produced by [`BM25Index::try_from_bytes`].
#[derive(Debug, Clone, PartialEq)]
pub enum BM25LoadError {
    /// Blob is shorter than the fixed header.
    ShortHeader,
    /// Magic bytes don't match `"BM25"`.
    BadMagic,
    /// Blob version is newer or older than [`FORMAT_VERSION`].
    VersionMismatch(u16),
    /// Declared section sizes run past the end of the blob.
    TruncatedSection(&'static str),
    /// A posting-list offset is not monotonically non-decreasing.
    NonMonotonicOffsets,
}

impl std::fmt::Display for BM25LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortHeader => write!(f, "BM25 blob shorter than header"),
            Self::BadMagic => write!(f, "BM25 blob: magic mismatch"),
            Self::VersionMismatch(v) => {
                write!(f, "BM25 blob: version {v} (expected {})", FORMAT_VERSION)
            }
            Self::TruncatedSection(name) => {
                write!(f, "BM25 blob: truncated section `{name}`")
            }
            Self::NonMonotonicOffsets => {
                write!(f, "BM25 blob: posting-list offsets are not monotonic")
            }
        }
    }
}

impl std::error::Error for BM25LoadError {}

/// Accumulator for documents to be indexed. Call [`insert`] once
/// per doc, then [`build`] to produce a [`BM25Index`].
///
/// "Doc" here is any 32-byte triblespace [`RawValue`] — same
/// schema-erased byte array the index uses for terms. Callers
/// typically pass entity ids via `Value::<GenId>::new(...).raw`
/// (see [`Self::insert_id`] for a convenience wrapper), but any
/// other schema works: string-valued keys (title search),
/// composite keys, or raw hashes.
///
/// [`insert`]: Self::insert
/// [`build`]: Self::build
pub struct BM25Builder {
    docs: Vec<(RawValue, Vec<RawValue>)>,
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

    /// Add a document. `key` is any 32-byte triblespace value
    /// identifying the doc (entity id via `GenId`, hash, string
    /// value, tag, whatever); `terms` is the caller's
    /// tokenization (see [`crate::tokens::hash_tokens`] for a
    /// simple default). The order of terms is irrelevant;
    /// duplicates contribute to term frequency.
    pub fn insert(&mut self, key: RawValue, terms: Vec<RawValue>) {
        self.docs.push((key, terms));
    }

    /// Convenience: add a document keyed by a triblespace [`Id`]
    /// (16 bytes). Shorthand for `insert(id_to_value(id).raw, ...)`.
    /// Matches the pre-generalization API so migration from the
    /// entity-keyed case is one method rename.
    pub fn insert_id(&mut self, doc_id: Id, terms: Vec<RawValue>) {
        let mut raw = [0u8; 32];
        let id_bytes: &RawId = doc_id.as_ref();
        raw[16..32].copy_from_slice(id_bytes);
        self.docs.push((raw, terms));
    }

    /// Consume the builder and produce an in-memory BM25 index
    /// on a single thread.
    pub fn build(self) -> BM25Index {
        self.build_with_threads(1)
    }

    /// Like [`build`], but shard the tf accumulation across
    /// `threads` worker threads. `threads = 1` is identical to
    /// single-threaded [`build`]; `threads > 1` spawns scoped
    /// workers, each accumulates a local term→tf map over a
    /// contiguous slice of docs, and the maps are merged at the
    /// end.
    ///
    /// Output is byte-identical to [`build`] — doc ids, term
    /// ordering, posting order, and scores all agree. Parallel
    /// speedups depend on the cost ratio of tf-map insertion
    /// vs. merge; tends to be worthwhile past ~5 k docs.
    ///
    /// [`build`]: Self::build
    pub fn build_with_threads(self, threads: usize) -> BM25Index {
        let Self { docs, k1, b } = self;
        let n_docs = docs.len();

        // Per-doc token count; average doc length for normalization.
        let doc_lens: Vec<u32> = docs.iter().map(|(_, t)| t.len() as u32).collect();
        let avg_doc_len = if n_docs == 0 {
            0.0
        } else {
            doc_lens.iter().map(|&n| n as f64).sum::<f64>() as f32 / n_docs as f32
        };

        let keys: Vec<RawValue> = docs.iter().map(|(key, _)| *key).collect();

        let term_to_tfs = if threads <= 1 || n_docs < 2 {
            // Single-threaded tf accumulation — cheap for small
            // corpora; also what we get when threads == 1.
            let mut m: HashMap<RawValue, HashMap<u32, u32>> = HashMap::new();
            for (doc_idx, (_, terms)) in docs.into_iter().enumerate() {
                accumulate_tfs(&mut m, doc_idx as u32, terms);
            }
            m
        } else {
            // Shard docs into `threads` contiguous ranges. Each
            // worker builds a local map over its slice using the
            // *global* doc_idx. Merge at the end.
            let threads = threads.min(n_docs);
            let base_chunk = n_docs / threads;
            let extra = n_docs % threads;

            // Partition `docs` into owned chunks the workers can
            // consume. We keep the start doc_idx of each chunk
            // to preserve global indexing.
            let mut starts = Vec::with_capacity(threads);
            let mut chunks: Vec<Vec<(RawValue, Vec<RawValue>)>> = Vec::with_capacity(threads);
            let mut docs_iter = docs.into_iter();
            let mut idx = 0usize;
            for t in 0..threads {
                let size = base_chunk + if t < extra { 1 } else { 0 };
                let chunk: Vec<_> = (&mut docs_iter).take(size).collect();
                starts.push(idx);
                idx += size;
                chunks.push(chunk);
            }

            // Scoped threads so references to `chunks` stay alive.
            let locals: Vec<HashMap<RawValue, HashMap<u32, u32>>> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(threads);
                for (shard_start, chunk) in starts.iter().zip(chunks.into_iter()) {
                    let start = *shard_start as u32;
                    handles.push(s.spawn(move || {
                        let mut m: HashMap<RawValue, HashMap<u32, u32>> = HashMap::new();
                        for (i, (_, terms)) in chunk.into_iter().enumerate() {
                            accumulate_tfs(&mut m, start + i as u32, terms);
                        }
                        m
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            // Merge local maps into one. Since shards cover
            // disjoint doc_idx ranges, each term's per-shard tf
            // submaps have disjoint keys — `extend` without
            // collision checks is sound and avoids the per-entry
            // hash lookup that `or_insert` costs.
            let mut merged: HashMap<RawValue, HashMap<u32, u32>> = HashMap::new();
            for local in locals {
                for (term, tfs) in local {
                    merged.entry(term).or_default().extend(tfs);
                }
            }
            merged
        };

        // Sort terms ascending so the term table supports binary
        // search (matches the succinct layout).
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
            keys,
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

/// Accumulate token-frequency counts for one doc into `m`.
fn accumulate_tfs(
    m: &mut HashMap<RawValue, HashMap<u32, u32>>,
    doc_idx: u32,
    terms: Vec<RawValue>,
) {
    for term in terms {
        let entry = m.entry(term).or_default().entry(doc_idx).or_insert(0);
        *entry += 1;
    }
}

/// In-memory BM25 index. Call [`BM25Builder::build`] to produce one.
///
/// All scores are pre-baked at build time: per-(doc, term) BM25
/// weight with saturating term frequency (`k1`) and
/// length-normalized doc length (`b`).
#[derive(Debug, Clone)]
pub struct BM25Index {
    /// Per-doc 32-byte keys. Any triblespace `RawValue` — most
    /// commonly an entity id (via `Value<GenId>`), but strings,
    /// tags, or hashes work too.
    keys: Vec<RawValue>,
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
        self.keys.len()
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
    /// Returns `(key, score)` pairs in posting-list order.
    /// Empty iterator if the term is absent.
    pub fn query_term(&self, term: &RawValue) -> impl Iterator<Item = (RawValue, f32)> + '_ {
        let lo = self.terms.binary_search(term).ok();
        let range = match lo {
            Some(i) => self.offsets[i] as usize..self.offsets[i + 1] as usize,
            None => 0..0,
        };
        self.postings[range]
            .iter()
            .map(|&(doc_idx, score)| (self.keys[doc_idx as usize], score))
    }

    /// Convenience: [`query_term`] but decode each key as a
    /// triblespace [`Id`] (assuming the index was keyed via
    /// [`BM25Builder::insert_id`] / `Value<GenId>`). Returns
    /// `None` if the stored key isn't a valid GenId (non-zero
    /// first 16 bytes or nil tail).
    pub fn query_term_ids<'a>(
        &'a self,
        term: &RawValue,
    ) -> impl Iterator<Item = (Id, f32)> + 'a {
        self.query_term(term).filter_map(|(raw, score)| {
            if raw[0..16] != [0u8; 16] {
                return None;
            }
            let id_bytes: RawId = raw[16..32].try_into().ok()?;
            Id::new(id_bytes).map(|id| (id, score))
        })
    }

    /// Convenience: [`query_multi`] decoded as `(Id, score)` pairs.
    ///
    /// [`query_multi`]: Self::query_multi
    pub fn query_multi_ids(&self, terms: &[RawValue]) -> Vec<(Id, f32)> {
        self.query_multi(terms)
            .into_iter()
            .filter_map(|(raw, s)| {
                if raw[0..16] != [0u8; 16] {
                    return None;
                }
                let id_bytes: RawId = raw[16..32].try_into().ok()?;
                Id::new(id_bytes).map(|id| (id, s))
            })
            .collect()
    }

    /// Score a multi-term query as the sum of per-term BM25
    /// weights (standard OR-like bag-of-words).
    ///
    /// Returned `(key, score)` pairs are sorted descending by
    /// score. No top-k truncation — caller slices what they need.
    pub fn query_multi(&self, terms: &[RawValue]) -> Vec<(RawValue, f32)> {
        let mut acc: HashMap<RawValue, f32> = HashMap::new();
        for term in terms {
            for (doc, score) in self.query_term(term) {
                *acc.entry(doc).or_insert(0.0) += score;
            }
        }
        let mut out: Vec<(RawValue, f32)> = acc.into_iter().collect();
        out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

    /// Doc-key table: `keys()[i]` is the external 32-byte
    /// `RawValue` for internal index `i`. Exposed so succinct
    /// re-encoders can snapshot the table without roundtripping
    /// through query_term.
    pub fn keys(&self) -> &[RawValue] {
        &self.keys
    }

    /// Sorted 32-byte term table. Used by succinct re-encoders
    /// and anyone implementing a custom query plan over this
    /// index's internals.
    pub fn terms_slice(&self) -> &[RawValue] {
        &self.terms
    }

    /// Per-term posting list (internal `doc_idx` + score) for the
    /// term at sorted-table position `t`. Returns `&[]` if out of
    /// range. Lower-level than [`query_term`], which joins on
    /// external `Id`s.
    ///
    /// [`query_term`]: Self::query_term
    pub fn postings_for(&self, t: usize) -> &[(u32, f32)] {
        if t >= self.terms.len() {
            return &[];
        }
        let lo = self.offsets[t] as usize;
        let hi = self.offsets[t + 1] as usize;
        &self.postings[lo..hi]
    }

    /// Serialize the index to a self-contained little-endian byte
    /// buffer. The layout is documented in `docs/DESIGN.md`:
    ///
    /// ```text
    /// [32 B header]
    /// [n_docs × 32 B keys]
    /// [n_docs ×  4 B doc_lens]
    /// [n_terms × 32 B terms (sorted)]
    /// [(n_terms + 1) × 4 B postings_offsets]
    /// [total_postings × 8 B (u32 doc_idx, f32 score)]
    /// ```
    ///
    /// Use [`BM25Index::try_from_bytes`] to reload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_docs = self.keys.len() as u32;
        let n_terms = self.terms.len() as u32;
        let total_postings = self.postings.len();

        let mut buf = Vec::with_capacity(
            HEADER_LEN
                + (self.keys.len() * 32)
                + (self.doc_lens.len() * 4)
                + (self.terms.len() * 32)
                + (self.offsets.len() * 4)
                + (total_postings * 8),
        );

        // ── header (32 B) ────────────────────────────────────────
        // Fields occupy 28 bytes; 4 bytes of zero-padding at the
        // end reserve space for a future field without requiring
        // a version bump for callers that only inspect the early
        // portion of the header.
        buf.extend_from_slice(&MAGIC.to_le_bytes()); // 4
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes()); // 2
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved, 2
        buf.extend_from_slice(&n_docs.to_le_bytes()); // 4
        buf.extend_from_slice(&n_terms.to_le_bytes()); // 4
        buf.extend_from_slice(&self.avg_doc_len.to_le_bytes()); // 4
        buf.extend_from_slice(&self.k1.to_le_bytes()); // 4
        buf.extend_from_slice(&self.b.to_le_bytes()); // 4
        buf.extend_from_slice(&[0u8; 4]); // pad to 32
        debug_assert_eq!(buf.len(), HEADER_LEN);

        // ── keys ─────────────────────────────────────────────────
        for key in &self.keys {
            buf.extend_from_slice(key);
        }
        // ── doc_lens ─────────────────────────────────────────────
        for &len in &self.doc_lens {
            buf.extend_from_slice(&len.to_le_bytes());
        }
        // ── terms (sorted) ───────────────────────────────────────
        for term in &self.terms {
            buf.extend_from_slice(term);
        }
        // ── postings_offsets ─────────────────────────────────────
        for &off in &self.offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        // ── postings ─────────────────────────────────────────────
        for &(doc_idx, score) in &self.postings {
            buf.extend_from_slice(&doc_idx.to_le_bytes());
            buf.extend_from_slice(&score.to_le_bytes());
        }
        buf
    }

    /// Reload an index previously produced by [`to_bytes`]. Fails
    /// cleanly on truncation, bad magic, version mismatch, and
    /// malformed offsets — the blob is guaranteed well-formed on
    /// a successful return.
    ///
    /// [`to_bytes`]: Self::to_bytes
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, BM25LoadError> {
        use BM25LoadError as E;

        if bytes.len() < HEADER_LEN {
            return Err(E::ShortHeader);
        }
        // Header.
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(E::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(E::VersionMismatch(version));
        }
        // bytes[6..8] reserved.
        let n_docs = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let n_terms = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let avg_doc_len = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let k1 = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let b = f32::from_le_bytes(bytes[24..28].try_into().unwrap());

        // Section offsets.
        let mut pos = HEADER_LEN;
        let keys_end = pos + n_docs * 32;
        if bytes.len() < keys_end {
            return Err(E::TruncatedSection("keys"));
        }
        let mut keys = Vec::with_capacity(n_docs);
        for chunk in bytes[pos..keys_end].chunks_exact(32) {
            let raw: RawValue = chunk.try_into().unwrap();
            keys.push(raw);
        }
        pos = keys_end;

        let doc_lens_end = pos + n_docs * 4;
        if bytes.len() < doc_lens_end {
            return Err(E::TruncatedSection("doc_lens"));
        }
        let doc_lens: Vec<u32> = bytes[pos..doc_lens_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        pos = doc_lens_end;

        let terms_end = pos + n_terms * 32;
        if bytes.len() < terms_end {
            return Err(E::TruncatedSection("terms"));
        }
        let mut terms: Vec<RawValue> = Vec::with_capacity(n_terms);
        for chunk in bytes[pos..terms_end].chunks_exact(32) {
            let raw: RawValue = chunk.try_into().unwrap();
            terms.push(raw);
        }
        pos = terms_end;

        let offsets_end = pos + (n_terms + 1) * 4;
        if bytes.len() < offsets_end {
            return Err(E::TruncatedSection("offsets"));
        }
        let offsets: Vec<u32> = bytes[pos..offsets_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        pos = offsets_end;

        // Offsets must be monotonically non-decreasing so the
        // posting-list slicing in `query_term` is sound.
        for pair in offsets.windows(2) {
            if pair[0] > pair[1] {
                return Err(E::NonMonotonicOffsets);
            }
        }

        let total_postings = *offsets.last().unwrap_or(&0) as usize;
        let postings_end = pos + total_postings * 8;
        if bytes.len() < postings_end {
            return Err(E::TruncatedSection("postings"));
        }
        let postings: Vec<(u32, f32)> = bytes[pos..postings_end]
            .chunks_exact(8)
            .map(|c| {
                let doc_idx = u32::from_le_bytes(c[0..4].try_into().unwrap());
                let score = f32::from_le_bytes(c[4..8].try_into().unwrap());
                (doc_idx, score)
            })
            .collect();

        Ok(Self {
            keys,
            doc_lens,
            avg_doc_len,
            terms,
            offsets,
            postings,
            k1,
            b,
        })
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

    /// Test helper: the 32-byte `Value<GenId>` representation of
    /// the Id used by `id(byte)`. Matches what `insert_id` stores
    /// internally, so `query_term` results can be compared
    /// against it.
    fn id_key(byte: u8) -> RawValue {
        let mut raw = [0u8; 32];
        let id = id(byte);
        let id_bytes: &RawId = id.as_ref();
        raw[16..32].copy_from_slice(id_bytes);
        raw
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
        b.insert_id(id(1), hash_tokens("the quick brown fox"));
        b.insert_id(id(2), hash_tokens("the lazy brown dog"));
        b.insert_id(id(3), hash_tokens("quick silver fox"));
        let idx = b.build();
        assert_eq!(idx.doc_count(), 3);

        // "fox" appears in docs 1 and 3.
        let fox = hash_tokens("fox");
        let hits: Vec<_> = idx.query_term(&fox[0]).collect();
        assert_eq!(hits.len(), 2);
        let doc_ids: Vec<_> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&id_key(1)));
        assert!(doc_ids.contains(&id_key(3)));

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
            b.insert_id(id(i), hash_tokens("common common"));
        }
        b.insert_id(id(100), hash_tokens("common rare"));
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
        b.insert_id(id(1), hash_tokens("foo bar baz"));
        let many: String = std::iter::repeat("foo ").take(100).collect::<String>();
        b.insert_id(id(2), hash_tokens(&many));
        let idx = b.build();

        let foo = hash_tokens("foo");
        let scores: HashMap<RawValue, f32> = idx.query_term(&foo[0]).collect();
        let s1 = scores[&id_key(1)];
        let s2 = scores[&id_key(2)];
        assert!(s2 > s1);
        assert!(
            s2 < s1 * 20.0,
            "tf saturation should keep ratio moderate: {s1} -> {s2}"
        );
    }

    #[test]
    fn multi_term_query_sums() {
        let mut b = BM25Builder::new();
        b.insert_id(id(1), hash_tokens("quick brown fox"));
        b.insert_id(id(2), hash_tokens("quick red fox"));
        b.insert_id(id(3), hash_tokens("slow brown dog"));
        let idx = b.build();

        let q = hash_tokens("quick fox");
        let ranked = idx.query_multi(&q);
        // Docs 1 and 2 have both terms; doc 3 has neither.
        assert_eq!(ranked.len(), 2);
        let top_ids: Vec<_> = ranked.iter().map(|(d, _)| *d).collect();
        assert!(top_ids.contains(&id_key(1)));
        assert!(top_ids.contains(&id_key(2)));
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

    fn build_sample_index() -> BM25Index {
        let mut b = BM25Builder::new().k1(1.4).b(0.72);
        b.insert_id(id(1), hash_tokens("the quick brown fox"));
        b.insert_id(id(2), hash_tokens("the lazy brown dog"));
        b.insert_id(id(3), hash_tokens("quick silver fox jumps"));
        b.build()
    }

    #[test]
    fn bytes_round_trip_preserves_queries() {
        let original = build_sample_index();
        let bytes = original.to_bytes();
        let reloaded = BM25Index::try_from_bytes(&bytes).expect("valid blob");

        assert_eq!(reloaded.doc_count(), original.doc_count());
        assert_eq!(reloaded.term_count(), original.term_count());
        assert!((reloaded.avg_doc_len() - original.avg_doc_len()).abs() < 1e-6);
        assert_eq!(reloaded.k1(), original.k1());
        assert_eq!(reloaded.b(), original.b());

        // Postings for every stored term must match exactly.
        let fox = hash_tokens("fox");
        let original_hits: Vec<_> = original.query_term(&fox[0]).collect();
        let reloaded_hits: Vec<_> = reloaded.query_term(&fox[0]).collect();
        assert_eq!(reloaded_hits.len(), original_hits.len());
        for (a, b) in original_hits.iter().zip(reloaded_hits.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_index_bytes_round_trip() {
        let idx = BM25Builder::new().build();
        let bytes = idx.to_bytes();
        let reloaded = BM25Index::try_from_bytes(&bytes).expect("valid blob");
        assert_eq!(reloaded.doc_count(), 0);
        assert_eq!(reloaded.term_count(), 0);
    }

    #[test]
    fn short_header_rejected() {
        let err = BM25Index::try_from_bytes(&[0; 10]).unwrap_err();
        assert_eq!(err, BM25LoadError::ShortHeader);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = build_sample_index().to_bytes();
        bytes[0] = b'X';
        let err = BM25Index::try_from_bytes(&bytes).unwrap_err();
        assert_eq!(err, BM25LoadError::BadMagic);
    }

    #[test]
    fn version_mismatch_rejected() {
        let mut bytes = build_sample_index().to_bytes();
        // Bump the version byte to something we don't recognize.
        bytes[4] = 99;
        let err = BM25Index::try_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, BM25LoadError::VersionMismatch(_)));
    }

    #[test]
    fn truncation_rejected() {
        let bytes = build_sample_index().to_bytes();
        let truncated = &bytes[..bytes.len() - 1];
        let err = BM25Index::try_from_bytes(truncated).unwrap_err();
        assert!(matches!(err, BM25LoadError::TruncatedSection(_)));
    }

    #[test]
    fn blob_is_deterministic() {
        // Same corpus → same bytes. Content-addressing depends on
        // this being exactly reproducible across runs.
        let a = build_sample_index().to_bytes();
        let b = build_sample_index().to_bytes();
        assert_eq!(a, b);
    }

    #[test]
    fn parallel_build_matches_single_thread() {
        // Build a richer corpus so shards actually have work to
        // do. Threaded and single-threaded paths must produce
        // byte-identical blobs.
        fn build(threads: usize) -> Vec<u8> {
            let mut b = BM25Builder::new();
            for i in 1..=50u32 {
                let text = format!(
                    "doc {i} text about {} {}",
                    (i % 5) + 1,
                    (i.wrapping_mul(7)) % 13
                );
                let byte = (i as u8).max(1);
                b.insert_id(id(byte), hash_tokens(&text));
            }
            b.build_with_threads(threads).to_bytes()
        }
        let serial = build(1);
        for t in [2usize, 3, 4, 8] {
            assert_eq!(
                build(t),
                serial,
                "threads={t} produced different bytes than serial"
            );
        }
    }

    #[test]
    fn parallel_build_on_empty_corpus() {
        let idx = BM25Builder::new().build_with_threads(4);
        assert_eq!(idx.doc_count(), 0);
        assert_eq!(idx.term_count(), 0);
    }

    #[test]
    fn parallel_build_threads_cap_at_doc_count() {
        // 3 docs × 16 threads — the builder caps threads at n_docs
        // and doesn't spawn idle workers.
        let mut b = BM25Builder::new();
        b.insert_id(id(1), hash_tokens("one two three"));
        b.insert_id(id(2), hash_tokens("two three four"));
        b.insert_id(id(3), hash_tokens("three four five"));
        let idx = b.build_with_threads(16);
        assert_eq!(idx.doc_count(), 3);
        // "three" shows up in all 3 docs.
        let three = hash_tokens("three")[0];
        assert_eq!(idx.doc_frequency(&three), 3);
    }

    #[test]
    fn ngrams_enable_prefix_queries() {
        // The payoff test: index docs with hash_tokens + 3-grams
        // concatenated, and query a prefix as 3-grams to recover
        // the extended form. Surface-exact queries still work via
        // the hash_tokens half.
        use crate::tokens::ngram_tokens;

        fn both(text: &str) -> Vec<RawValue> {
            let mut v = hash_tokens(text);
            v.extend(ngram_tokens(text, 3));
            v
        }

        let mut b = BM25Builder::new();
        b.insert_id(id(1), both("foxes are cunning"));
        b.insert_id(id(2), both("the dog barks"));
        b.insert_id(id(3), both("silver fox at night"));
        let idx = b.build();

        // Query "fox" as trigrams: just one gram, "fox". Both
        // "foxes" (doc 1) and "fox" (doc 3) should score, but
        // "dog" (doc 2) must not.
        let q = ngram_tokens("fox", 3);
        let hits: Vec<_> = idx.query_multi(&q);
        let doc_ids: Vec<_> = hits.iter().map(|(d, _)| *d).collect();
        assert!(doc_ids.contains(&id_key(1)), "prefix should match 'foxes'");
        assert!(doc_ids.contains(&id_key(3)), "prefix should match 'fox'");
        assert!(!doc_ids.contains(&id_key(2)), "must not match 'dog'");
    }
}
