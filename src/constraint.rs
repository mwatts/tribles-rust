//! Triblespace query-engine integration.
//!
//! Exposes [`BM25Index`] as a first-class `Constraint` so
//! callers can plug search into the normal `find!` / `pattern!`
//! / `and!` / `or!` machinery.
//!
//! V1 ships the simplest useful shape:
//!
//! ```rust,ignore
//! let docs_matching: DocsContainingTerm = index.docs_containing(doc_var, term);
//! ```
//!
//! which pins `term` at construction time and constrains the
//! `doc` variable to entity ids whose posting list includes the
//! term. Scores live on [`BM25Index`] and are looked up via
//! [`BM25Index::query_term`] once the caller has the doc — v2
//! will lift `score` to a bound variable.
//!
//! See `docs/QUERY_ENGINE_INTEGRATION.md` for the longer-term
//! design covering bidirectional BM25 and vector similarity.

use std::collections::HashSet;

use triblespace::core::query::{Binding, Constraint, Variable, VariableId, VariableSet};
use triblespace::core::value::schemas::genid::GenId;
use triblespace::core::value::RawValue;

use crate::bm25::BM25Index;
use crate::hnsw::{FlatIndex, HNSWIndex};
use crate::schemas::F32LE;

/// Minimum surface a BM25 index must expose for the
/// [`DocsContainingTerm`] / [`BM25ScoredPostings`] constraints to
/// work against it. Implemented for both the naive
/// [`crate::bm25::BM25Index`] and the succinct
/// [`crate::succinct::SuccinctBM25Index`] so either can plug
/// into `find!` / `pattern!` without changes at the engine layer.
pub trait BM25Queryable {
    /// Iterate `(key, score)` for the posting list of `term`.
    /// Keys are 32-byte triblespace `RawValue`s — the caller's
    /// `Variable<S>` decodes them through whatever `ValueSchema`
    /// is appropriate. Empty iterator if the term is absent.
    fn query_term_boxed<'a>(
        &'a self,
        term: &RawValue,
    ) -> Box<dyn Iterator<Item = (RawValue, f32)> + 'a>;

    /// Number of docs containing `term`. Used by `estimate`.
    fn doc_frequency_for(&self, term: &RawValue) -> usize;

    /// Absolute tolerance the engine should use when comparing a
    /// stored score against a bound `score` variable. Default
    /// `f32::EPSILON` — appropriate for indexes that store
    /// scores losslessly. Quantized / lossy backends should
    /// widen this to their bucket size to still accept
    /// semantically-equal scores that round to different float
    /// representations.
    fn score_tolerance(&self) -> f32 {
        f32::EPSILON
    }
}

impl BM25Queryable for BM25Index {
    fn query_term_boxed<'a>(
        &'a self,
        term: &RawValue,
    ) -> Box<dyn Iterator<Item = (RawValue, f32)> + 'a> {
        Box::new(self.query_term(term))
    }

    fn doc_frequency_for(&self, term: &RawValue) -> usize {
        self.doc_frequency(term)
    }
}

#[cfg(feature = "succinct")]
impl BM25Queryable for crate::succinct::SuccinctBM25Index {
    fn query_term_boxed<'a>(
        &'a self,
        term: &RawValue,
    ) -> Box<dyn Iterator<Item = (RawValue, f32)> + 'a> {
        self.query_term(term)
    }

    fn doc_frequency_for(&self, term: &RawValue) -> usize {
        self.doc_frequency(term)
    }

    fn score_tolerance(&self) -> f32 {
        // Quantization bucket size; widens the equality check to
        // accept scores that round to different f32s after the
        // u16 → f32 dequantization. Call the inherent method
        // explicitly so there's no trait-method recursion.
        crate::succinct::SuccinctBM25Index::score_tolerance(self)
    }
}

/// Minimum surface an HNSW-style index must expose for the
/// [`SimilarToVectorHNSW`] / [`SimilarToVectorHNSWScored`]
/// constraints to work against it. Implemented for both the naive
/// [`crate::hnsw::HNSWIndex`] and the succinct
/// [`crate::succinct::SuccinctHNSWIndex`].
pub trait HNSWQueryable {
    /// Approximate top-`k` nearest neighbours to `query` under
    /// cosine similarity, with optional `ef` search width.
    /// Returns stored 32-byte [`RawValue`] keys — callers with a
    /// `GenId`-schema index decode via the `raw_value_to_id`
    /// helper in `constraint.rs`.
    fn similar_for(&self, query: &[f32], k: usize, ef: Option<usize>) -> Vec<(RawValue, f32)>;

    /// Total indexed docs (for cardinality estimates).
    fn doc_count_for(&self) -> usize;
}

impl HNSWQueryable for HNSWIndex {
    fn similar_for(&self, query: &[f32], k: usize, ef: Option<usize>) -> Vec<(RawValue, f32)> {
        self.similar(query, k, ef)
    }
    fn doc_count_for(&self) -> usize {
        self.doc_count()
    }
}

#[cfg(feature = "succinct")]
impl HNSWQueryable for crate::succinct::SuccinctHNSWIndex {
    fn similar_for(&self, query: &[f32], k: usize, ef: Option<usize>) -> Vec<(RawValue, f32)> {
        self.similar(query, k, ef)
    }
    fn doc_count_for(&self) -> usize {
        self.doc_count()
    }
}

/// Constrains a `Variable<S>` (doc) to the 32-byte
/// [`RawValue`]s in the posting list of the pinned term. `S` is
/// whatever [`ValueSchema`] the caller keys the index with —
/// typically [`GenId`] for entity-keyed indexes, but any schema
/// works (string titles, tags, fragment hashes, etc.).
///
/// Generic over any `I: BM25Queryable`, so it works against
/// [`BM25Index`] or
/// [`crate::succinct::SuccinctBM25Index`] without code duplication.
///
/// Created via [`BM25Index::docs_containing`] (or the succinct
/// equivalent).
pub struct DocsContainingTerm<'a, I: BM25Queryable + ?Sized = BM25Index, S = GenId>
where
    S: triblespace::core::value::ValueSchema,
{
    index: &'a I,
    doc: Variable<S>,
    /// The term value is pinned at constraint-construction time.
    term: [u8; 32],
}

impl<'a, I: BM25Queryable + ?Sized, S> DocsContainingTerm<'a, I, S>
where
    S: triblespace::core::value::ValueSchema,
{
    pub fn new(index: &'a I, doc: Variable<S>, term: [u8; 32]) -> Self {
        Self { index, doc, term }
    }
}

impl BM25Index {
    /// Produce a [`DocsContainingTerm`] constraint for use inside
    /// `pattern!` / `find!`. `S` is the schema the caller
    /// keyed the index with — usually [`GenId`] for entity ids,
    /// but any [`ValueSchema`] works.
    pub fn docs_containing<S>(
        &self,
        doc: Variable<S>,
        term: [u8; 32],
    ) -> DocsContainingTerm<'_, BM25Index, S>
    where
        S: triblespace::core::value::ValueSchema,
    {
        DocsContainingTerm::new(self, doc, term)
    }
}

/// Encode an `f32` as an `F32LE`-schema raw value (bytes [0..4]
/// little-endian, rest zero-padded). Matches
/// `schemas::F32LE::to_value` but avoids the schema-type wrapper
/// for direct `RawValue` writes into the constraint's proposal
/// vector.
fn f32_to_raw_value(v: f32) -> RawValue {
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&v.to_le_bytes());
    out
}

/// Decode an `F32LE`-schema raw value back to `f32`.
fn raw_value_to_f32(raw: &RawValue) -> f32 {
    f32::from_le_bytes(raw[0..4].try_into().unwrap())
}

// ── BM25 constraint with score as a bound variable ───────────────────

/// Two-variable BM25 constraint: binds both `doc` (entity id) and
/// `score` (BM25 weight) for the pinned term. Produced by
/// [`BM25Index::docs_and_scores`]; strict upgrade of
/// [`DocsContainingTerm`] that lets callers project the score
/// into their query results.
///
/// Cardinality (`estimate`) uses `index.doc_frequency(&term)` on
/// the `doc` variable; for `score` we use the same count because
/// each (doc, term) pair has exactly one score — the cardinality
/// is identical from the engine's perspective.
pub struct BM25ScoredPostings<'a, I: BM25Queryable + ?Sized = BM25Index, S = GenId>
where
    S: triblespace::core::value::ValueSchema,
{
    index: &'a I,
    doc: Variable<S>,
    score: Variable<F32LE>,
    term: [u8; 32],
}

impl<'a, I: BM25Queryable + ?Sized, S> BM25ScoredPostings<'a, I, S>
where
    S: triblespace::core::value::ValueSchema,
{
    pub fn new(index: &'a I, doc: Variable<S>, score: Variable<F32LE>, term: [u8; 32]) -> Self {
        Self {
            index,
            doc,
            score,
            term,
        }
    }
}

impl BM25Index {
    /// Constraint that binds `doc` + `score` for each posting of
    /// `term`. Use this when the caller wants to project the
    /// BM25 weight into their result rows (filtering, ordering,
    /// hybrid-ranking combinators above the query).
    pub fn docs_and_scores<S>(
        &self,
        doc: Variable<S>,
        score: Variable<F32LE>,
        term: [u8; 32],
    ) -> BM25ScoredPostings<'_, BM25Index, S>
    where
        S: triblespace::core::value::ValueSchema,
    {
        BM25ScoredPostings::new(self, doc, score, term)
    }
}

impl<'a, I: BM25Queryable + ?Sized + 'a, S> Constraint<'a> for BM25ScoredPostings<'a, I, S>
where
    S: triblespace::core::value::ValueSchema + 'a,
{
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
            .union(VariableSet::new_singleton(self.score.index))
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if variable == self.doc.index || variable == self.score.index {
            Some(self.index.doc_frequency_for(&self.term))
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            let tol = self.index.score_tolerance();
            for (key, score) in self.index.query_term_boxed(&self.term) {
                if let Some(bs) = bound_score {
                    if (score - bs).abs() > tol {
                        continue;
                    }
                }
                proposals.push(key);
            }
        } else if variable == self.score.index {
            let bound_doc = binding.get(self.doc.index).copied();
            // Dedupe score bit-patterns: two docs sharing a BM25
            // score would otherwise produce two identical score
            // proposals, causing the engine to enumerate a
            // Cartesian (doc, score) cross within the bucket.
            let mut seen = HashSet::new();
            for (key, score) in self.index.query_term_boxed(&self.term) {
                if let Some(bd) = bound_doc {
                    if key != bd {
                        continue;
                    }
                }
                if seen.insert(score.to_bits()) {
                    proposals.push(f32_to_raw_value(score));
                }
            }
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            let tol = self.index.score_tolerance();
            let valid: HashSet<RawValue> = self
                .index
                .query_term_boxed(&self.term)
                .filter(|(_, s)| match bound_score {
                    Some(bs) => (s - bs).abs() <= tol,
                    None => true,
                })
                .map(|(k, _)| k)
                .collect();
            proposals.retain(|raw| valid.contains(raw));
        } else if variable == self.score.index {
            let bound_doc = binding.get(self.doc.index).copied();
            let allowed: HashSet<u32> = self
                .index
                .query_term_boxed(&self.term)
                .filter(|(k, _)| match bound_doc {
                    Some(bd) => *k == bd,
                    None => true,
                })
                .map(|(_, s)| s.to_bits())
                .collect();
            proposals.retain(|raw| allowed.contains(&raw_value_to_f32(raw).to_bits()));
        }
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        let bound_doc = binding.get(self.doc.index).copied();
        let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
        let tol = self.index.score_tolerance();
        match (bound_doc, bound_score) {
            (None, None) => true,
            _ => self.index.query_term_boxed(&self.term).any(|(k, s)| {
                bound_doc.map_or(true, |bd| k == bd)
                    && bound_score.map_or(true, |bs| (s - bs).abs() <= tol)
            }),
        }
    }
}

// ── FlatIndex constraint ─────────────────────────────────────────────

/// Constrains a `Variable<GenId>` (doc) to the top-`k` neighbours
/// of a pinned query vector under cosine similarity.
///
/// Created via [`FlatIndex::similar_constraint`]. The `query_vec`
/// and `k` are parameters of the constraint, not bound variables
/// — per `docs/QUERY_ENGINE_INTEGRATION.md`'s design, this
/// matches how similarity is actually used (the query is a
/// concrete embedding, not something the engine solves for).
pub struct SimilarToVector<'a> {
    index: &'a FlatIndex,
    doc: Variable<GenId>,
    query: Vec<f32>,
    k: usize,
}

impl<'a> SimilarToVector<'a> {
    pub fn new(index: &'a FlatIndex, doc: Variable<GenId>, query: Vec<f32>, k: usize) -> Self {
        Self {
            index,
            doc,
            query,
            k,
        }
    }
}

impl FlatIndex {
    /// Build a [`SimilarToVector`] constraint for use inside
    /// `pattern!` / `find!`.
    pub fn similar_constraint(
        &self,
        doc: Variable<GenId>,
        query: Vec<f32>,
        k: usize,
    ) -> SimilarToVector<'_> {
        SimilarToVector::new(self, doc, query, k)
    }
}

impl<'a> Constraint<'a> for SimilarToVector<'a> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            // We'll emit at most `k` hits; `k` is a hard upper
            // bound on the proposal count, so it's also our
            // cardinality estimate.
            Some(self.k.min(self.index.doc_count()))
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        for (key, _score) in self.index.similar(&self.query, self.k) {
            proposals.push(key);
        }
    }

    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        // Collect the top-k result keys and keep only proposals
        // whose doc appears in that set.
        let top: HashSet<RawValue> = self
            .index
            .similar(&self.query, self.k)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        proposals.retain(|raw| top.contains(raw));
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.doc.index) {
            Some(raw) => self
                .index
                .similar(&self.query, self.k)
                .into_iter()
                .any(|(k, _)| k == *raw),
            None => true,
        }
    }
}

// ── FlatIndex constraint with score ─────────────────────────────────

/// Scored variant of [`SimilarToVector`]: binds both `doc` and
/// `score` per top-`k` hit. Produced by
/// [`FlatIndex::similar_with_scores`]. See [`BM25ScoredPostings`]
/// for the sibling BM25 version.
pub struct SimilarToVectorScored<'a> {
    index: &'a FlatIndex,
    doc: Variable<GenId>,
    score: Variable<F32LE>,
    query: Vec<f32>,
    k: usize,
}

impl<'a> SimilarToVectorScored<'a> {
    pub fn new(
        index: &'a FlatIndex,
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        query: Vec<f32>,
        k: usize,
    ) -> Self {
        Self {
            index,
            doc,
            score,
            query,
            k,
        }
    }
}

impl FlatIndex {
    /// Two-variable similarity constraint — binds both `doc` and
    /// the cosine-similarity `score` for each top-k hit.
    pub fn similar_with_scores(
        &self,
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        query: Vec<f32>,
        k: usize,
    ) -> SimilarToVectorScored<'_> {
        SimilarToVectorScored::new(self, doc, score, query, k)
    }
}

impl<'a> Constraint<'a> for SimilarToVectorScored<'a> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
            .union(VariableSet::new_singleton(self.score.index))
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if variable == self.doc.index || variable == self.score.index {
            Some(self.k.min(self.index.doc_count()))
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            for (key, score) in self.index.similar(&self.query, self.k) {
                if let Some(bs) = bound_score {
                    if (score - bs).abs() > f32::EPSILON {
                        continue;
                    }
                }
                proposals.push(key);
            }
        } else if variable == self.score.index {
            let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
            // Dedupe by bit-pattern to avoid Cartesian blow-up
            // when two neighbours share a similarity value.
            let mut seen = HashSet::new();
            for (key, score) in self.index.similar(&self.query, self.k) {
                if let Some(bd) = bound_doc {
                    if key != bd {
                        continue;
                    }
                }
                if seen.insert(score.to_bits()) {
                    proposals.push(f32_to_raw_value(score));
                }
            }
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        let top: Vec<(RawValue, f32)> = self.index.similar(&self.query, self.k);
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            let allowed: HashSet<RawValue> = top
                .iter()
                .filter(|(_, s)| match bound_score {
                    Some(bs) => (s - bs).abs() <= f32::EPSILON,
                    None => true,
                })
                .map(|(k, _)| *k)
                .collect();
            proposals.retain(|raw| allowed.contains(raw));
        } else if variable == self.score.index {
            let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
            let allowed: HashSet<u32> = top
                .iter()
                .filter(|(k, _)| match bound_doc {
                    Some(bd) => *k == bd,
                    None => true,
                })
                .map(|(_, s)| s.to_bits())
                .collect();
            proposals.retain(|raw| allowed.contains(&raw_value_to_f32(raw).to_bits()));
        }
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
        let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
        match (bound_doc, bound_score) {
            (None, None) => true,
            _ => self
                .index
                .similar(&self.query, self.k)
                .iter()
                .any(|&(k, s)| {
                    bound_doc.map_or(true, |bd| k == bd)
                        && bound_score.map_or(true, |bs| (s - bs).abs() <= f32::EPSILON)
                }),
        }
    }
}

// ── HNSWIndex constraint ─────────────────────────────────────────────

/// HNSW version of [`SimilarToVector`]. Same shape; different
/// backing index. Approximate rather than exact top-k.
///
/// Generic over `I: HNSWQueryable`, so it works against
/// [`HNSWIndex`] or
/// [`crate::succinct::SuccinctHNSWIndex`] without duplication.
pub struct SimilarToVectorHNSW<'a, I: HNSWQueryable + ?Sized = HNSWIndex> {
    index: &'a I,
    doc: Variable<GenId>,
    query: Vec<f32>,
    k: usize,
    /// HNSW's `ef_search` parameter. Larger = better recall at
    /// higher query cost. Defaults to `k` when `None`.
    ef: Option<usize>,
}

impl<'a, I: HNSWQueryable + ?Sized> SimilarToVectorHNSW<'a, I> {
    pub fn new(
        index: &'a I,
        doc: Variable<GenId>,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> Self {
        Self {
            index,
            doc,
            query,
            k,
            ef,
        }
    }
}

impl HNSWIndex {
    /// Build a [`SimilarToVectorHNSW`] constraint for use inside
    /// `pattern!` / `find!`. Pass `ef = Some(n)` to widen search
    /// (higher recall, slower); `None` defaults to `k`.
    pub fn similar_constraint(
        &self,
        doc: Variable<GenId>,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> SimilarToVectorHNSW<'_, HNSWIndex> {
        SimilarToVectorHNSW::new(self, doc, query, k, ef)
    }
}

impl<'a, I: HNSWQueryable + ?Sized + 'a> Constraint<'a> for SimilarToVectorHNSW<'a, I> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            Some(self.k.min(self.index.doc_count_for()))
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        for (key, _score) in self.index.similar_for(&self.query, self.k, self.ef) {
            proposals.push(key);
        }
    }

    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        let top: HashSet<RawValue> = self
            .index
            .similar_for(&self.query, self.k, self.ef)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        proposals.retain(|raw| top.contains(raw));
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.doc.index) {
            Some(raw) => self
                .index
                .similar_for(&self.query, self.k, self.ef)
                .into_iter()
                .any(|(k, _)| k == *raw),
            None => true,
        }
    }
}

// ── HNSWIndex constraint with score ─────────────────────────────────

/// Scored variant of [`SimilarToVectorHNSW`]: binds both `doc`
/// and the cosine `score` per top-`k` hit.
///
/// Generic over `I: HNSWQueryable`.
pub struct SimilarToVectorHNSWScored<'a, I: HNSWQueryable + ?Sized = HNSWIndex> {
    index: &'a I,
    doc: Variable<GenId>,
    score: Variable<F32LE>,
    query: Vec<f32>,
    k: usize,
    ef: Option<usize>,
}

impl<'a, I: HNSWQueryable + ?Sized> SimilarToVectorHNSWScored<'a, I> {
    pub fn new(
        index: &'a I,
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> Self {
        Self {
            index,
            doc,
            score,
            query,
            k,
            ef,
        }
    }
}

impl HNSWIndex {
    /// Two-variable similarity constraint for HNSW — binds both
    /// `doc` and cosine `score` for each top-k hit.
    pub fn similar_with_scores(
        &self,
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> SimilarToVectorHNSWScored<'_, HNSWIndex> {
        SimilarToVectorHNSWScored::new(self, doc, score, query, k, ef)
    }
}

impl<'a, I: HNSWQueryable + ?Sized + 'a> Constraint<'a> for SimilarToVectorHNSWScored<'a, I> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
            .union(VariableSet::new_singleton(self.score.index))
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if variable == self.doc.index || variable == self.score.index {
            Some(self.k.min(self.index.doc_count_for()))
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            for (key, score) in self.index.similar_for(&self.query, self.k, self.ef) {
                if let Some(bs) = bound_score {
                    if (score - bs).abs() > f32::EPSILON {
                        continue;
                    }
                }
                proposals.push(key);
            }
        } else if variable == self.score.index {
            let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
            // Dedupe by bit-pattern to avoid Cartesian blow-up
            // when two neighbours share a similarity value.
            let mut seen = HashSet::new();
            for (key, score) in self.index.similar_for(&self.query, self.k, self.ef) {
                if let Some(bd) = bound_doc {
                    if key != bd {
                        continue;
                    }
                }
                if seen.insert(score.to_bits()) {
                    proposals.push(f32_to_raw_value(score));
                }
            }
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        let top = self.index.similar_for(&self.query, self.k, self.ef);
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            let allowed: HashSet<RawValue> = top
                .iter()
                .filter(|(_, s)| match bound_score {
                    Some(bs) => (s - bs).abs() <= f32::EPSILON,
                    None => true,
                })
                .map(|(k, _)| *k)
                .collect();
            proposals.retain(|raw| allowed.contains(raw));
        } else if variable == self.score.index {
            let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
            let allowed: HashSet<u32> = top
                .iter()
                .filter(|(k, _)| match bound_doc {
                    Some(bd) => *k == bd,
                    None => true,
                })
                .map(|(_, s)| s.to_bits())
                .collect();
            proposals.retain(|raw| allowed.contains(&raw_value_to_f32(raw).to_bits()));
        }
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
        let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
        match (bound_doc, bound_score) {
            (None, None) => true,
            _ => self
                .index
                .similar_for(&self.query, self.k, self.ef)
                .into_iter()
                .any(|(k, s)| {
                    bound_doc.map_or(true, |bd| k == bd)
                        && bound_score.map_or(true, |bs| (s - bs).abs() <= f32::EPSILON)
                }),
        }
    }
}

impl<'a, I: BM25Queryable + ?Sized + 'a, S> Constraint<'a> for DocsContainingTerm<'a, I, S>
where
    S: triblespace::core::value::ValueSchema + 'a,
{
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            Some(self.index.doc_frequency_for(&self.term))
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        for (key, _score) in self.index.query_term_boxed(&self.term) {
            proposals.push(key);
        }
    }

    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        // Collect the term's posting list into a set for O(1)
        // membership checks. For very long posting lists a sorted
        // binary-search would be lighter on memory; for v1 this is
        // fine up to tens of thousands of postings.
        let docs: HashSet<RawValue> =
            self.index.query_term_boxed(&self.term).map(|(k, _)| k).collect();
        proposals.retain(|raw| docs.contains(raw));
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        // If `doc` is already bound, the constraint is satisfied
        // iff that doc appears in the term's posting list.
        match binding.get(self.doc.index).copied() {
            Some(bound_key) => self
                .index
                .query_term_boxed(&self.term)
                .any(|(k, _)| k == bound_key),
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::BM25Builder;
    use crate::tokens::hash_tokens;
    use triblespace::core::id::{Id, RawId};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    /// 32-byte `Value<GenId>` form of `id(byte)` — matches what
    /// `BM25Builder::insert_id` stores and what the index's
    /// `query_term` returns.
    fn id_key(byte: u8) -> RawValue {
        let mut raw = [0u8; 32];
        let id = id(byte);
        let id_bytes: &RawId = id.as_ref();
        raw[16..32].copy_from_slice(id_bytes);
        raw
    }

    /// `GenId`-schema RawValue → `Id` test helper.
    fn raw_value_to_id(raw: &RawValue) -> Option<Id> {
        if raw[0..16] != [0u8; 16] {
            return None;
        }
        let raw_id: RawId = raw[16..32].try_into().ok()?;
        Id::new(raw_id)
    }

    /// `Id` → `GenId`-schema RawValue test helper.
    fn id_to_raw_value(id: Id) -> RawValue {
        let mut out = [0u8; 32];
        let raw: &RawId = id.as_ref();
        out[16..32].copy_from_slice(raw);
        out
    }

    fn sample_index() -> BM25Index {
        let mut b = BM25Builder::new();
        b.insert_id(id(1), hash_tokens("the quick brown fox"));
        b.insert_id(id(2), hash_tokens("the lazy brown dog"));
        b.insert_id(id(3), hash_tokens("quick silver fox jumps"));
        b.build()
    }

    #[test]
    fn constraint_variables_is_singleton_of_doc() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_containing(doc, term);

        let vars = c.variables();
        assert!(vars.is_set(doc.index));
        // And no other variable is touched.
        let mut found = 0;
        for i in 0..32 {
            if vars.is_set(i) {
                found += 1;
            }
        }
        assert_eq!(found, 1);
    }

    #[test]
    fn constraint_estimate_is_doc_frequency() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_containing(doc, term);

        let binding = Binding::default();
        assert_eq!(c.estimate(doc.index, &binding), Some(2));
        // Unknown variable → None.
        let unk_binding = Binding::default();
        // Pick an unused variable id (one past what VariableContext allocated).
        assert_eq!(c.estimate(255, &unk_binding), None);
    }

    #[test]
    fn constraint_proposes_posting_list_as_genid_values() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_containing(doc, term);

        let binding = Binding::default();
        let mut props: Vec<RawValue> = Vec::new();
        c.propose(doc.index, &binding, &mut props);
        assert_eq!(props.len(), 2);

        // All proposals decode cleanly into Ids, and those Ids
        // are exactly the docs that had "fox" in the posting
        // list.
        let ids: HashSet<Id> = props
            .iter()
            .map(|r| raw_value_to_id(r).expect("valid GenId value"))
            .collect();
        assert!(ids.contains(&id(1)));
        assert!(ids.contains(&id(3)));
    }

    #[test]
    fn constraint_confirm_filters_non_matching_docs() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_containing(doc, term);

        let binding = Binding::default();
        let mut props: Vec<RawValue> = vec![
            id_to_raw_value(id(1)),
            id_to_raw_value(id(2)),
            id_to_raw_value(id(3)),
        ];
        c.confirm(doc.index, &binding, &mut props);
        // Doc 2 doesn't contain "fox" — should be filtered out.
        // Docs 1 and 3 do.
        let ids: HashSet<Id> = props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id(1)));
        assert!(!ids.contains(&id(2)));
        assert!(ids.contains(&id(3)));
    }

    #[test]
    fn constraint_satisfied_checks_bound_doc() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_containing(doc, term);

        // Unbound → trivially satisfied.
        let empty = Binding::default();
        assert!(c.satisfied(&empty));

        // Bound to a matching doc → satisfied.
        let mut bound = Binding::default();
        bound.set(doc.index, &id_to_raw_value(id(1)));
        assert!(c.satisfied(&bound));

        // Bound to a non-matching doc → unsatisfied.
        let mut unmatching = Binding::default();
        unmatching.set(doc.index, &id_to_raw_value(id(2)));
        assert!(!c.satisfied(&unmatching));
    }

    fn sample_flat() -> FlatIndex {
        use crate::hnsw::FlatBuilder;
        let mut b = FlatBuilder::new(3);
        b.insert_id(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        b.insert_id(id(2), vec![0.0, 1.0, 0.0]).unwrap();
        b.insert_id(id(3), vec![0.9, 0.1, 0.0]).unwrap();
        b.insert_id(id(4), vec![0.0, 0.0, 1.0]).unwrap();
        b.build()
    }

    #[test]
    fn similar_constraint_estimate_is_k() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 2);

        let binding = Binding::default();
        // k=2 and corpus has 4 docs → estimate is 2.
        assert_eq!(c.estimate(doc.index, &binding), Some(2));
    }

    #[test]
    fn similar_constraint_estimate_clamps_to_corpus() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        // Larger k than corpus — estimate must clamp to doc_count.
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 100);
        let binding = Binding::default();
        assert_eq!(c.estimate(doc.index, &binding), Some(4));
    }

    #[test]
    fn similar_constraint_proposes_top_k() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 2);

        let binding = Binding::default();
        let mut props = Vec::new();
        c.propose(doc.index, &binding, &mut props);
        let ids: HashSet<Id> = props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
        // top-2 for [1,0,0] query should be id(1) (exact) and id(3) (close).
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id(1)));
        assert!(ids.contains(&id(3)));
    }

    #[test]
    fn similar_constraint_confirm_filters_non_top_k() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 2);

        let binding = Binding::default();
        let mut props: Vec<RawValue> = vec![
            id_to_raw_value(id(1)),
            id_to_raw_value(id(2)), // not in top-2
            id_to_raw_value(id(3)),
            id_to_raw_value(id(4)), // not in top-2
        ];
        c.confirm(doc.index, &binding, &mut props);
        let ids: HashSet<Id> = props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
        assert!(ids.contains(&id(1)));
        assert!(!ids.contains(&id(2)));
        assert!(ids.contains(&id(3)));
        assert!(!ids.contains(&id(4)));
    }

    #[test]
    fn similar_constraint_satisfied_checks_bound_doc() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 2);

        // Unbound → trivially satisfied.
        assert!(c.satisfied(&Binding::default()));
        // Bound to an in-top-k doc → satisfied.
        let mut bound = Binding::default();
        bound.set(doc.index, &id_to_raw_value(id(1)));
        assert!(c.satisfied(&bound));
        // Bound to a not-in-top-k doc → unsatisfied.
        let mut unmatching = Binding::default();
        unmatching.set(doc.index, &id_to_raw_value(id(4)));
        assert!(!c.satisfied(&unmatching));
    }

    // ── BM25ScoredPostings (doc + score bound) ────────────────

    #[test]
    fn scored_constraint_proposes_both_variables() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_and_scores(doc, score, term);

        // Variables: both doc and score.
        let vars = c.variables();
        assert!(vars.is_set(doc.index));
        assert!(vars.is_set(score.index));

        // Cardinality: same doc_frequency for both variables.
        assert_eq!(c.estimate(doc.index, &Binding::default()), Some(2));
        assert_eq!(c.estimate(score.index, &Binding::default()), Some(2));

        // Propose doc with no binding: 2 entries, each decodes
        // to a real id.
        let mut doc_props = Vec::new();
        c.propose(doc.index, &Binding::default(), &mut doc_props);
        assert_eq!(doc_props.len(), 2);
        for p in &doc_props {
            raw_value_to_id(p).expect("genid round-trip");
        }

        // Propose score with no binding: the two "fox" docs in
        // sample_index have identical BM25 scores (same length,
        // same tf), so dedupe collapses them to one proposal.
        // Every entry must decode to a positive finite f32.
        let mut score_props = Vec::new();
        c.propose(score.index, &Binding::default(), &mut score_props);
        assert_eq!(score_props.len(), 1);
        for p in &score_props {
            let v = raw_value_to_f32(p);
            assert!(v.is_finite() && v > 0.0);
        }
    }

    #[test]
    fn scored_constraint_binds_doc_given_score() {
        // Purpose-built corpus where "fox" appears in two docs of
        // *different* lengths — so the two postings have
        // different BM25 scores and the score-filter
        // meaningfully distinguishes them.
        let mut b = BM25Builder::new();
        b.insert_id(id(1), hash_tokens("fox"));
        b.insert_id(id(2), hash_tokens("quick brown fox jumps high today"));
        b.insert_id(id(3), hash_tokens("lazy dog"));
        let idx = b.build();

        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_and_scores(doc, score, term);

        let doc1_score = idx
            .query_term(&term)
            .find(|(d, _)| *d == id_key(1))
            .map(|(_, s)| s)
            .unwrap();
        let doc2_score = idx
            .query_term(&term)
            .find(|(d, _)| *d == id_key(2))
            .map(|(_, s)| s)
            .unwrap();
        assert!(
            (doc1_score - doc2_score).abs() > f32::EPSILON,
            "test fixture: doc1 / doc2 scores should differ"
        );

        let mut binding = Binding::default();
        binding.set(score.index, &f32_to_raw_value(doc1_score));
        let mut props = Vec::new();
        c.propose(doc.index, &binding, &mut props);
        assert_eq!(props.len(), 1);
        assert_eq!(raw_value_to_id(&props[0]).unwrap(), id(1));
    }

    #[test]
    fn scored_constraint_binds_score_given_doc() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_and_scores(doc, score, term);

        let mut binding = Binding::default();
        binding.set(doc.index, &id_to_raw_value(id(1)));

        let mut props = Vec::new();
        c.propose(score.index, &binding, &mut props);
        // Exactly one score for (doc = id(1), term = fox).
        assert_eq!(props.len(), 1);
        assert!(raw_value_to_f32(&props[0]) > 0.0);
    }

    #[test]
    fn scored_constraint_confirm_filters_bad_scores() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_and_scores(doc, score, term);

        // Offer a "score" proposal list mixing one real score
        // from the index with some made-up ones. Confirm must
        // keep only the real ones.
        let real = idx.query_term(&term).map(|(_, s)| s).collect::<Vec<f32>>();
        let mut props = vec![
            f32_to_raw_value(real[0]),
            f32_to_raw_value(999.0),
            f32_to_raw_value(0.001),
        ];
        c.confirm(score.index, &Binding::default(), &mut props);
        assert_eq!(props.len(), 1);
        assert!((raw_value_to_f32(&props[0]) - real[0]).abs() < 1e-6);
    }

    fn sample_hnsw() -> crate::hnsw::HNSWIndex {
        use crate::hnsw::HNSWBuilder;
        let mut b = HNSWBuilder::new(3).with_seed(42);
        b.insert_id(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        b.insert_id(id(2), vec![0.0, 1.0, 0.0]).unwrap();
        b.insert_id(id(3), vec![0.9, 0.1, 0.0]).unwrap();
        b.insert_id(id(4), vec![0.0, 0.0, 1.0]).unwrap();
        b.build()
    }

    #[test]
    fn hnsw_constraint_proposes_top_k() {
        let idx = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 2, Some(10));

        let binding = Binding::default();
        let mut props = Vec::new();
        c.propose(doc.index, &binding, &mut props);
        let ids: HashSet<Id> = props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
        // Top-2 neighbours of [1,0,0] should include docs 1 and 3
        // (exact and near-exact matches respectively). HNSW is
        // approximate; allow either to be present and just check
        // neither of 3/4 dominates.
        assert!(ids.len() <= 2);
        assert!(ids.contains(&id(1)) || ids.contains(&id(3)));
    }

    #[test]
    fn hnsw_constraint_estimate_clamps_to_corpus() {
        let idx = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 100, None);
        assert_eq!(c.estimate(doc.index, &Binding::default()), Some(4));
    }

    #[test]
    fn hnsw_constraint_satisfied_respects_binding() {
        let idx = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.similar_constraint(doc, vec![1.0, 0.0, 0.0], 2, Some(10));

        // Unbound → trivially satisfied.
        assert!(c.satisfied(&Binding::default()));
    }

    // ── SimilarToVectorScored (flat + score) ──────────────────

    #[test]
    fn flat_scored_constraint_proposes_both() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let c = idx.similar_with_scores(doc, score, vec![1.0, 0.0, 0.0], 3);

        let vars = c.variables();
        assert!(vars.is_set(doc.index));
        assert!(vars.is_set(score.index));

        let mut doc_props = Vec::new();
        c.propose(doc.index, &Binding::default(), &mut doc_props);
        assert_eq!(doc_props.len(), 3);
        for p in &doc_props {
            raw_value_to_id(p).expect("genid round-trip");
        }

        let mut score_props = Vec::new();
        c.propose(score.index, &Binding::default(), &mut score_props);
        assert_eq!(score_props.len(), 3);
        for p in &score_props {
            let s = raw_value_to_f32(p);
            assert!(s.is_finite() && s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn flat_scored_binds_doc_given_score() {
        let idx = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let c = idx.similar_with_scores(doc, score, vec![1.0, 0.0, 0.0], 4);

        // Prime `score` to the exact-match value (1.0). Only
        // id(1) — the [1,0,0] doc — should come back.
        let mut binding = Binding::default();
        binding.set(score.index, &f32_to_raw_value(1.0));

        let mut props = Vec::new();
        c.propose(doc.index, &binding, &mut props);
        assert_eq!(props.len(), 1);
        assert_eq!(raw_value_to_id(&props[0]).unwrap(), id(1));
    }

    // ── SimilarToVectorHNSWScored (hnsw + score) ──────────────

    #[test]
    fn hnsw_scored_constraint_proposes_both() {
        let idx = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let c = idx.similar_with_scores(doc, score, vec![1.0, 0.0, 0.0], 2, Some(10));

        let vars = c.variables();
        assert!(vars.is_set(doc.index));
        assert!(vars.is_set(score.index));

        let mut doc_props = Vec::new();
        c.propose(doc.index, &Binding::default(), &mut doc_props);
        assert!(!doc_props.is_empty());
        for p in &doc_props {
            raw_value_to_id(p).expect("genid round-trip");
        }

        let mut score_props = Vec::new();
        c.propose(score.index, &Binding::default(), &mut score_props);
        assert_eq!(score_props.len(), doc_props.len());
    }

    /// Sanity-check that `propose` yields values every consumer
    /// will be able to decode back into `Id`s.
    #[test]
    fn proposed_values_decode_back_to_ids() {
        let idx = sample_index();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_containing(doc, term);

        let binding = Binding::default();
        let mut proposals = Vec::new();
        c.propose(doc.index, &binding, &mut proposals);
        for raw in &proposals {
            raw_value_to_id(raw).expect("genid roundtrip");
        }
    }
}
