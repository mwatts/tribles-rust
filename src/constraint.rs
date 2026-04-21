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
/// The HNSW constraints now store eagerly-computed top-k
/// (same shape as the Flat constraints), so there's no trait
/// indirection at query time — the trait previously here
/// (`HNSWQueryable`) was dropped in the HNSW handle port.

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
pub struct SimilarToVector {
    doc: Variable<GenId>,
    /// Top-`k` `(key, score)` pairs computed eagerly at
    /// construction. Handle-fetching against the blob store
    /// happens once, not once per `propose`/`confirm`/
    /// `satisfied`.
    top: Vec<(RawValue, f32)>,
}

impl SimilarToVector {
    /// Build directly from the already-computed top-k. Used by
    /// [`FlatIndex::similar_constraint`] internally.
    pub fn from_top(doc: Variable<GenId>, top: Vec<(RawValue, f32)>) -> Self {
        Self { doc, top }
    }
}

impl<'a, B> crate::hnsw::AttachedFlatIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<
            triblespace::core::value::schemas::hash::Blake3,
        >,
{
    /// Build a [`SimilarToVector`] constraint for use inside
    /// `pattern!` / `find!`. Eagerly resolves the top-`k`
    /// against the attached store up front so subsequent
    /// engine calls don't re-scan.
    pub fn similar_constraint(
        &self,
        doc: Variable<GenId>,
        query: Vec<f32>,
        k: usize,
    ) -> Result<SimilarToVector, B::GetError<anybytes::view::ViewError>> {
        let top = self.similar(&query, k)?;
        Ok(SimilarToVector::from_top(doc, top))
    }
}

impl<'a> Constraint<'a> for SimilarToVector {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            Some(self.top.len())
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        for (key, _score) in &self.top {
            proposals.push(*key);
        }
    }

    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        let top: HashSet<RawValue> = self.top.iter().map(|(k, _)| *k).collect();
        proposals.retain(|raw| top.contains(raw));
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.doc.index) {
            Some(raw) => self.top.iter().any(|(k, _)| k == raw),
            None => true,
        }
    }
}

// ── FlatIndex constraint with score ─────────────────────────────────

/// Scored variant of [`SimilarToVector`]: binds both `doc` and
/// `score` per top-`k` hit. Produced by
/// [`FlatIndex::similar_with_scores`]. See [`BM25ScoredPostings`]
/// for the sibling BM25 version.
pub struct SimilarToVectorScored {
    doc: Variable<GenId>,
    score: Variable<F32LE>,
    /// Eagerly computed top-`k` `(key, score)` pairs.
    top: Vec<(RawValue, f32)>,
}

impl SimilarToVectorScored {
    pub fn from_top(
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        top: Vec<(RawValue, f32)>,
    ) -> Self {
        Self { doc, score, top }
    }
}

impl<'a, B> crate::hnsw::AttachedFlatIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<
            triblespace::core::value::schemas::hash::Blake3,
        >,
{
    /// Two-variable similarity constraint — binds both `doc`
    /// and the cosine-similarity `score` for each top-k hit.
    /// Eagerly resolves the top-`k` against the attached store.
    pub fn similar_with_scores(
        &self,
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        query: Vec<f32>,
        k: usize,
    ) -> Result<SimilarToVectorScored, B::GetError<anybytes::view::ViewError>> {
        let top = self.similar(&query, k)?;
        Ok(SimilarToVectorScored::from_top(doc, score, top))
    }
}

impl<'a> Constraint<'a> for SimilarToVectorScored {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
            .union(VariableSet::new_singleton(self.score.index))
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if variable == self.doc.index || variable == self.score.index {
            Some(self.top.len())
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            for (key, score) in &self.top {
                if let Some(bs) = bound_score {
                    if (score - bs).abs() > f32::EPSILON {
                        continue;
                    }
                }
                proposals.push(*key);
            }
        } else if variable == self.score.index {
            let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
            // Dedupe by bit-pattern to avoid Cartesian blow-up
            // when two neighbours share a similarity value.
            let mut seen = HashSet::new();
            for (key, score) in &self.top {
                if let Some(bd) = bound_doc {
                    if *key != bd {
                        continue;
                    }
                }
                if seen.insert(score.to_bits()) {
                    proposals.push(f32_to_raw_value(*score));
                }
            }
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            let allowed: HashSet<RawValue> = self
                .top
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
            let allowed: HashSet<u32> = self
                .top
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
            _ => self.top.iter().any(|&(k, s)| {
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
/// Eagerly resolves the top-`k` against the caller-supplied
/// blob store at construction time and caches the result.
/// `propose` / `confirm` / `satisfied` iterate the cached
/// list — no re-walking the HNSW graph per method call.
pub struct SimilarToVectorHNSW {
    doc: Variable<GenId>,
    top: Vec<(RawValue, f32)>,
}

impl SimilarToVectorHNSW {
    pub fn from_top(doc: Variable<GenId>, top: Vec<(RawValue, f32)>) -> Self {
        Self { doc, top }
    }
}

impl<'a, B> crate::hnsw::AttachedHNSWIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<
            triblespace::core::value::schemas::hash::Blake3,
        >,
{
    /// Build a [`SimilarToVectorHNSW`] constraint for use
    /// inside `pattern!` / `find!`. Pass `ef = Some(n)` to
    /// widen search (higher recall, slower); `None` defaults
    /// to `k`. The HNSW walk happens once at construction
    /// against the attached store.
    pub fn similar_constraint(
        &self,
        doc: Variable<GenId>,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> Result<SimilarToVectorHNSW, B::GetError<anybytes::view::ViewError>> {
        let top = self.similar(&query, k, ef)?;
        Ok(SimilarToVectorHNSW::from_top(doc, top))
    }
}

impl<'a> Constraint<'a> for SimilarToVectorHNSW {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            Some(self.top.len())
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        for (key, _score) in &self.top {
            proposals.push(*key);
        }
    }

    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.doc.index != variable {
            return;
        }
        let top: HashSet<RawValue> = self.top.iter().map(|(k, _)| *k).collect();
        proposals.retain(|raw| top.contains(raw));
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.doc.index) {
            Some(raw) => self.top.iter().any(|(k, _)| k == raw),
            None => true,
        }
    }
}

// ── HNSWIndex constraint with score ─────────────────────────────────

/// Scored variant of [`SimilarToVectorHNSW`]: binds both `doc`
/// and the cosine `score` per top-`k` hit. Eagerly resolves
/// top-`k` at construction.
pub struct SimilarToVectorHNSWScored {
    doc: Variable<GenId>,
    score: Variable<F32LE>,
    top: Vec<(RawValue, f32)>,
}

impl SimilarToVectorHNSWScored {
    pub fn from_top(
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        top: Vec<(RawValue, f32)>,
    ) -> Self {
        Self { doc, score, top }
    }
}

impl<'a, B> crate::hnsw::AttachedHNSWIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<
            triblespace::core::value::schemas::hash::Blake3,
        >,
{
    /// Two-variable similarity constraint for HNSW — binds
    /// both `doc` and cosine `score` for each top-k hit.
    /// Eagerly resolves against the attached store.
    pub fn similar_with_scores(
        &self,
        doc: Variable<GenId>,
        score: Variable<F32LE>,
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> Result<SimilarToVectorHNSWScored, B::GetError<anybytes::view::ViewError>> {
        let top = self.similar(&query, k, ef)?;
        Ok(SimilarToVectorHNSWScored::from_top(doc, score, top))
    }
}

impl<'a> Constraint<'a> for SimilarToVectorHNSWScored {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
            .union(VariableSet::new_singleton(self.score.index))
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if variable == self.doc.index || variable == self.score.index {
            Some(self.top.len())
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            for (key, score) in &self.top {
                if let Some(bs) = bound_score {
                    if (score - bs).abs() > f32::EPSILON {
                        continue;
                    }
                }
                proposals.push(*key);
            }
        } else if variable == self.score.index {
            let bound_doc: Option<RawValue> = binding.get(self.doc.index).copied();
            // Dedupe by bit-pattern to avoid Cartesian blow-up
            // when two neighbours share a similarity value.
            let mut seen = HashSet::new();
            for (key, score) in &self.top {
                if let Some(bd) = bound_doc {
                    if *key != bd {
                        continue;
                    }
                }
                if seen.insert(score.to_bits()) {
                    proposals.push(f32_to_raw_value(*score));
                }
            }
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.doc.index {
            let bound_score = binding.get(self.score.index).map(raw_value_to_f32);
            let allowed: HashSet<RawValue> = self
                .top
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
            let allowed: HashSet<u32> = self
                .top
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
            _ => self.top.iter().any(|&(k, s)| {
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
    use triblespace::core::repo::BlobStore;

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

    fn sample_flat() -> (
        crate::hnsw::FlatIndex,
        triblespace::core::blob::MemoryBlobStore<
            triblespace::core::value::schemas::hash::Blake3,
        >,
    ) {
        use crate::hnsw::FlatBuilder;
        use triblespace::core::blob::MemoryBlobStore;
        use triblespace::core::value::schemas::hash::Blake3;
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = FlatBuilder::new(3);
        for (i, v) in [
            (1u8, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.9, 0.1, 0.0]),
            (4, vec![0.0, 0.0, 1.0]),
        ] {
            let h = crate::schemas::put_embedding::<_, Blake3>(&mut store, v).unwrap();
            b.insert_id(id(i), h);
        }
        (b.build(), store)
    }

    #[test]
    fn similar_constraint_estimate_is_k() {
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 2)
            .unwrap();

        let binding = Binding::default();
        // k=2 and corpus has 4 docs → estimate is 2.
        assert_eq!(c.estimate(doc.index, &binding), Some(2));
    }

    #[test]
    fn similar_constraint_estimate_clamps_to_corpus() {
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        // Larger k than corpus — estimate must clamp to doc_count.
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 100)
            .unwrap();
        let binding = Binding::default();
        assert_eq!(c.estimate(doc.index, &binding), Some(4));
    }

    #[test]
    fn similar_constraint_proposes_top_k() {
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 2)
            .unwrap();

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
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 2)
            .unwrap();

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
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 2)
            .unwrap();

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

    fn sample_hnsw() -> (
        crate::hnsw::HNSWIndex,
        triblespace::core::blob::MemoryBlobStore<
            triblespace::core::value::schemas::hash::Blake3,
        >,
    ) {
        use crate::hnsw::HNSWBuilder;
        use triblespace::core::blob::MemoryBlobStore;
        use triblespace::core::value::schemas::hash::Blake3;
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3).with_seed(42);
        for (i, v) in [
            (1u8, vec![1.0f32, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.9, 0.1, 0.0]),
            (4, vec![0.0, 0.0, 1.0]),
        ] {
            let h = crate::schemas::put_embedding::<_, Blake3>(&mut store, v.clone()).unwrap();
            b.insert_id(id(i), h, v).unwrap();
        }
        (b.build(), store)
    }

    #[test]
    fn hnsw_constraint_proposes_top_k() {
        let (idx, mut store) = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 2, Some(10))
            .unwrap();

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
        let (idx, mut store) = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 100, None)
            .unwrap();
        assert_eq!(c.estimate(doc.index, &Binding::default()), Some(4));
    }

    #[test]
    fn hnsw_constraint_satisfied_respects_binding() {
        let (idx, mut store) = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_constraint(doc, vec![1.0, 0.0, 0.0], 2, Some(10))
            .unwrap();

        // Unbound → trivially satisfied.
        assert!(c.satisfied(&Binding::default()));
    }

    // ── SimilarToVectorScored (flat + score) ──────────────────

    #[test]
    fn flat_scored_constraint_proposes_both() {
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_with_scores(doc, score, vec![1.0, 0.0, 0.0], 3)
            .unwrap();

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
        let (idx, mut store) = sample_flat();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let c = idx.attach(&store.reader().unwrap()).similar_with_scores(doc, score, vec![1.0, 0.0, 0.0], 4)
            .unwrap();

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
        let (idx, mut store) = sample_hnsw();
        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let c = idx
            .attach(&store.reader().unwrap())
            .similar_with_scores(doc, score, vec![1.0, 0.0, 0.0], 2, Some(10))
            .unwrap();

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
