//! Triblespace query-engine integration.
//!
//! Two constraint shapes currently ship:
//!
//! * [`DocsContainingTerm`] / [`BM25ScoredPostings`] — BM25
//!   posting-list constraints produced by
//!   [`BM25Index::docs_containing`] /
//!   [`BM25Index::docs_and_scores`].
//! * [`Similar`] — a binary relation
//!   `similar(a: Variable<Handle<Blake3, Embedding>>, b:
//!   Variable<Handle<Blake3, Embedding>>, score_floor: f32)`
//!   produced by the `similar()` method on an
//!   [`AttachedHNSWIndex`] / [`AttachedFlatIndex`] /
//!   [`AttachedSuccinctHNSWIndex`]. The relation is symmetric
//!   (cosine similarity), `a` and `b` are both embedding
//!   handles, and `score_floor` is a fixed cosine threshold —
//!   *not* a bound variable. Callers who need the exact score
//!   fetch both embeddings and compute it directly (no
//!   quantization).
//!
//! See `docs/QUERY_ENGINE_INTEGRATION.md` for the long-form
//! design.

use std::collections::HashSet;

use triblespace::core::query::{Binding, Constraint, Variable, VariableId, VariableSet};
use triblespace::core::value::schemas::genid::GenId;
use triblespace::core::value::schemas::hash::{Blake3, Handle};
use triblespace::core::value::{RawValue, Value};

use crate::bm25::BM25Index;
use crate::schemas::{Embedding, F32LE};

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

impl<D: triblespace::core::value::ValueSchema, T: triblespace::core::value::ValueSchema>
    BM25Queryable for BM25Index<D, T>
{
    fn query_term_boxed<'a>(
        &'a self,
        term: &RawValue,
    ) -> Box<dyn Iterator<Item = (RawValue, f32)> + 'a> {
        // Wrap the raw bytes in `Value<T>` at the trait boundary
        // — the typed API inside the index expects `&Value<T>`.
        let term_val = Value::<T>::new(*term);
        Box::new(self.query_term(&term_val).map(|(v, s)| (v.raw, s)))
    }

    fn doc_frequency_for(&self, term: &RawValue) -> usize {
        let term_val = Value::<T>::new(*term);
        self.doc_frequency(&term_val)
    }
}

#[cfg(feature = "succinct")]
impl<D: triblespace::core::value::ValueSchema, T: triblespace::core::value::ValueSchema>
    BM25Queryable for crate::succinct::SuccinctBM25Index<D, T>
{
    fn query_term_boxed<'a>(
        &'a self,
        term: &RawValue,
    ) -> Box<dyn Iterator<Item = (RawValue, f32)> + 'a> {
        let term_val = Value::<T>::new(*term);
        Box::new(self.query_term(&term_val).map(|(v, s)| (v.raw, s)))
    }

    fn doc_frequency_for(&self, term: &RawValue) -> usize {
        let term_val = Value::<T>::new(*term);
        self.doc_frequency(&term_val)
    }

    fn score_tolerance(&self) -> f32 {
        // Quantization bucket size; widens the equality check to
        // accept scores that round to different f32s after the
        // u16 → f32 dequantization. Call the inherent method
        // explicitly so there's no trait-method recursion.
        crate::succinct::SuccinctBM25Index::<D, T>::score_tolerance(self)
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
pub struct DocsContainingTerm<'a, I: BM25Queryable + ?Sized, S = GenId>
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

impl<D: triblespace::core::value::ValueSchema, T: triblespace::core::value::ValueSchema>
    BM25Index<D, T>
{
    /// Produce a [`DocsContainingTerm`] constraint for use inside
    /// `pattern!` / `find!`. The doc `Variable<D>` is tied to the
    /// index's doc schema; `term` is a typed `Value<T>` from the
    /// index's term schema.
    pub fn docs_containing(
        &self,
        doc: Variable<D>,
        term: Value<T>,
    ) -> DocsContainingTerm<'_, BM25Index<D, T>, D> {
        DocsContainingTerm::new(self, doc, term.raw)
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
pub struct BM25ScoredPostings<'a, I: BM25Queryable + ?Sized, S = GenId>
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

impl<D: triblespace::core::value::ValueSchema, T: triblespace::core::value::ValueSchema>
    BM25Index<D, T>
{
    /// Constraint that binds `doc` + `score` for each posting of
    /// `term`. Use this when the caller wants to project the
    /// BM25 weight into their result rows (filtering, ordering,
    /// hybrid-ranking combinators above the query).
    pub fn docs_and_scores(
        &self,
        doc: Variable<D>,
        score: Variable<F32LE>,
        term: Value<T>,
    ) -> BM25ScoredPostings<'_, BM25Index<D, T>, D> {
        BM25ScoredPostings::new(self, doc, score, term.raw)
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

// ── Similarity constraint ───────────────────────────────────────────

/// Backing surface a similarity index must expose for the
/// [`Similar`] binary-relation constraint. Implemented for the
/// three attached views:
/// [`crate::hnsw::AttachedHNSWIndex`],
/// [`crate::hnsw::AttachedFlatIndex`], and
/// [`crate::succinct::AttachedSuccinctHNSWIndex`].
///
/// Both methods are infallible at the trait boundary —
/// implementations map storage / fetch failures to "no results"
/// (empty [`Vec`] or [`None`]). The engine's `propose` / `confirm`
/// / `satisfied` hooks have no error channel, so failing open
/// with "no match" is the only engine-safe choice; debug-time
/// diagnostics belong in the concrete attached view's inherent
/// methods, not here.
pub trait SimilaritySearch {
    /// Return every handle `b` in the index such that the cosine
    /// similarity `cos(*from, *b) ≥ score_floor`. `from` may or
    /// may not be in the index (e.g. it could be a query vector
    /// put into the pile for this one call).
    fn neighbours_above(
        &self,
        from: Value<Handle<Blake3, Embedding>>,
        score_floor: f32,
    ) -> Vec<Value<Handle<Blake3, Embedding>>>;

    /// Exact cosine similarity between the two handles, or
    /// [`None`] if either blob can't be fetched / parsed.
    fn cosine_between(
        &self,
        a: Value<Handle<Blake3, Embedding>>,
        b: Value<Handle<Blake3, Embedding>>,
    ) -> Option<f32>;
}

/// Binary similarity-relation constraint:
/// `similar(a, b, score_floor)` holds iff `a` and `b` are both
/// embedding handles with `cosine(*a, *b) ≥ score_floor`.
///
/// Semantics are symmetric (cosine is symmetric). Operationally,
/// at least one of `a` / `b` must be bound so the engine can walk
/// the index from that side; when both are bound, the constraint
/// fetches both embeddings and checks the threshold directly.
///
/// `score_floor` is fixed at constraint-construction — it's a
/// query parameter, not a bound variable. Callers who need the
/// exact score can fetch both handles after the query and
/// compute it without the approximation / quantisation that a
/// score-variable would bring in.
///
/// Produced by the `similar` method on an
/// [`crate::hnsw::AttachedHNSWIndex`] /
/// [`crate::hnsw::AttachedFlatIndex`] /
/// [`crate::succinct::AttachedSuccinctHNSWIndex`].
pub struct Similar<'a, I: SimilaritySearch + ?Sized> {
    index: &'a I,
    a: Variable<Handle<Blake3, Embedding>>,
    b: Variable<Handle<Blake3, Embedding>>,
    score_floor: f32,
}

impl<'a, I: SimilaritySearch + ?Sized> Similar<'a, I> {
    /// Build a constraint. Usually invoked through the `similar`
    /// method on an attached index rather than directly.
    pub fn new(
        index: &'a I,
        a: Variable<Handle<Blake3, Embedding>>,
        b: Variable<Handle<Blake3, Embedding>>,
        score_floor: f32,
    ) -> Self {
        Self {
            index,
            a,
            b,
            score_floor,
        }
    }
}

impl<'a, I: SimilaritySearch + ?Sized + 'a> Constraint<'a> for Similar<'a, I> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.a.index).union(VariableSet::new_singleton(self.b.index))
    }

    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        if variable != self.a.index && variable != self.b.index {
            return None;
        }
        let other = if variable == self.a.index {
            self.b.index
        } else {
            self.a.index
        };
        match binding.get(other).copied() {
            // Other side bound: count the candidates from the
            // walk and report an exact cardinality.
            Some(from) => Some(
                self.index
                    .neighbours_above(Value::new(from), self.score_floor)
                    .len(),
            ),
            // Other side unbound: the engine is still ordering
            // the join — signal "expensive" so it picks a
            // cheaper constraint first, rather than `None` which
            // would flag the variable as unconstrained.
            None => Some(usize::MAX),
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable != self.a.index && variable != self.b.index {
            return;
        }
        let other = if variable == self.a.index {
            self.b.index
        } else {
            self.a.index
        };
        let Some(from) = binding.get(other).copied() else {
            // Can't propose without a pivot; engine should pick
            // another constraint first.
            return;
        };
        for h in self
            .index
            .neighbours_above(Value::new(from), self.score_floor)
        {
            proposals.push(h.raw);
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable != self.a.index && variable != self.b.index {
            return;
        }
        let other = if variable == self.a.index {
            self.b.index
        } else {
            self.a.index
        };
        let Some(from) = binding.get(other).copied() else {
            // With no pivot, we can only keep proposals that pair
            // with *something* in the index above the floor. Keep
            // them all — the engine will revisit once the other
            // side is bound.
            return;
        };
        let allowed: HashSet<RawValue> = self
            .index
            .neighbours_above(Value::new(from), self.score_floor)
            .into_iter()
            .map(|h| h.raw)
            .collect();
        proposals.retain(|raw| allowed.contains(raw));
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match (binding.get(self.a.index), binding.get(self.b.index)) {
            (Some(a), Some(b)) => {
                // Both bound: compute cosine directly. No engine
                // reason to prefer the walk here — exact beats
                // approximate once we've paid the two blob fetches.
                match self.index.cosine_between(Value::new(*a), Value::new(*b)) {
                    Some(sim) => sim >= self.score_floor,
                    None => false,
                }
            }
            // Only one side bound: treated as trivially satisfied
            // — the engine will exercise propose/confirm on the
            // free side before binding it.
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::BM25Builder;
    use crate::tokens::hash_tokens;
    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::id::{Id, RawId};
    use triblespace::core::repo::BlobStore;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    /// 32-byte `Value<GenId>` form of `id(byte)` — matches what
    /// `BM25Builder::insert` stores and what the index's
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
        b.insert(&id(1), hash_tokens("the quick brown fox"));
        b.insert(&id(2), hash_tokens("the lazy brown dog"));
        b.insert(&id(3), hash_tokens("quick silver fox jumps"));
        b.build_naive()
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

        let empty = Binding::default();
        assert!(c.satisfied(&empty));

        let mut bound = Binding::default();
        bound.set(doc.index, &id_to_raw_value(id(1)));
        assert!(c.satisfied(&bound));

        let mut unmatching = Binding::default();
        unmatching.set(doc.index, &id_to_raw_value(id(2)));
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

        let vars = c.variables();
        assert!(vars.is_set(doc.index));
        assert!(vars.is_set(score.index));

        assert_eq!(c.estimate(doc.index, &Binding::default()), Some(2));
        assert_eq!(c.estimate(score.index, &Binding::default()), Some(2));

        let mut doc_props = Vec::new();
        c.propose(doc.index, &Binding::default(), &mut doc_props);
        assert_eq!(doc_props.len(), 2);
        for p in &doc_props {
            raw_value_to_id(p).expect("genid round-trip");
        }

        // sample_index's two "fox" docs share identical BM25
        // scores — dedupe collapses them to one proposal.
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
        // different BM25 scores.
        let mut b = BM25Builder::new();
        b.insert(&id(1), hash_tokens("fox"));
        b.insert(&id(2), hash_tokens("quick brown fox jumps high today"));
        b.insert(&id(3), hash_tokens("lazy dog"));
        let idx = b.build();

        let mut ctx = triblespace::core::query::VariableContext::new();
        let doc: Variable<GenId> = ctx.next_variable();
        let score: Variable<F32LE> = ctx.next_variable();
        let term = hash_tokens("fox")[0];
        let c = idx.docs_and_scores(doc, score, term);

        let doc1_score = idx
            .query_term(&term)
            .find(|(d, _)| d.raw == id_key(1))
            .map(|(_, s)| s)
            .unwrap();
        let doc2_score = idx
            .query_term(&term)
            .find(|(d, _)| d.raw == id_key(2))
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

    // ── Similar (binary-relation similarity) ──────────────────

    /// Build a 3-doc corpus where doc 1 = [1,0,0], doc 2 = [0,1,0],
    /// doc 3 ≈ doc 1. Returns (flat_index, hnsw_index, store,
    /// handles) — handles is parallel-indexed `[h1, h2, h3]`.
    fn sample_sim() -> (
        crate::hnsw::FlatIndex,
        crate::hnsw::HNSWIndex,
        MemoryBlobStore<Blake3>,
        [Value<Handle<Blake3, Embedding>>; 3],
    ) {
        use crate::hnsw::{FlatBuilder, HNSWBuilder};
        let mut store = MemoryBlobStore::<Blake3>::new();
        let vecs = [
            vec![1.0f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.9, 0.1, 0.0],
        ];
        let mut handles: [Value<Handle<Blake3, Embedding>>; 3] =
            [Value::new([0u8; 32]); 3];
        for (i, v) in vecs.iter().enumerate() {
            handles[i] =
                crate::schemas::put_embedding::<_, Blake3>(&mut store, v.clone()).unwrap();
        }
        let mut flat = FlatBuilder::new(3);
        for h in handles.iter() {
            flat.insert(*h);
        }
        let mut hnsw = HNSWBuilder::new(3).with_seed(42);
        for (i, v) in vecs.iter().enumerate() {
            hnsw.insert(handles[i], v.clone()).unwrap();
        }
        (flat.build(), hnsw.build_naive(), store, handles)
    }

    #[test]
    fn flat_similar_proposes_candidates_above_floor() {
        let (flat, _hnsw, mut store, handles) = sample_sim();
        let reader = store.reader().unwrap();
        let view = flat.attach(&reader);

        let mut ctx = triblespace::core::query::VariableContext::new();
        let a: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let b: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let c = view.similar(a, b, 0.8);

        // Bind `a` to doc 1's handle; propose `b`. Expect doc 1
        // (cos=1.0) and doc 3 (cos≈0.994) above 0.8; doc 2
        // (cos=0) below.
        let mut binding = Binding::default();
        binding.set(a.index, &handles[0].raw);

        let mut props = Vec::new();
        c.propose(b.index, &binding, &mut props);
        let got: HashSet<RawValue> = props.iter().copied().collect();
        assert!(got.contains(&handles[0].raw));
        assert!(got.contains(&handles[2].raw));
        assert!(!got.contains(&handles[1].raw));
    }

    #[test]
    fn flat_similar_symmetric_bind_on_b() {
        let (flat, _hnsw, mut store, handles) = sample_sim();
        let reader = store.reader().unwrap();
        let view = flat.attach(&reader);

        let mut ctx = triblespace::core::query::VariableContext::new();
        let a: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let b: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let c = view.similar(a, b, 0.8);

        let mut binding = Binding::default();
        binding.set(b.index, &handles[2].raw);

        let mut props = Vec::new();
        c.propose(a.index, &binding, &mut props);
        let got: HashSet<RawValue> = props.iter().copied().collect();
        assert!(got.contains(&handles[0].raw));
        assert!(got.contains(&handles[2].raw));
    }

    #[test]
    fn flat_similar_satisfied_both_bound() {
        let (flat, _hnsw, mut store, handles) = sample_sim();
        let reader = store.reader().unwrap();
        let view = flat.attach(&reader);

        let mut ctx = triblespace::core::query::VariableContext::new();
        let a: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let b: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let c = view.similar(a, b, 0.8);

        // doc 1 vs doc 3: cos ≈ 0.994, above 0.8 → satisfied.
        let mut good = Binding::default();
        good.set(a.index, &handles[0].raw);
        good.set(b.index, &handles[2].raw);
        assert!(c.satisfied(&good));

        // doc 1 vs doc 2: cos = 0.0, below 0.8 → not satisfied.
        let mut bad = Binding::default();
        bad.set(a.index, &handles[0].raw);
        bad.set(b.index, &handles[1].raw);
        assert!(!c.satisfied(&bad));
    }

    #[test]
    fn hnsw_similar_proposes_candidates_above_floor() {
        let (_flat, hnsw, mut store, handles) = sample_sim();
        let reader = store.reader().unwrap();
        let view = hnsw.attach(&reader);

        let mut ctx = triblespace::core::query::VariableContext::new();
        let a: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let b: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let c = view.similar(a, b, 0.8);

        let mut binding = Binding::default();
        binding.set(a.index, &handles[0].raw);

        let mut props = Vec::new();
        c.propose(b.index, &binding, &mut props);
        let got: HashSet<RawValue> = props.iter().copied().collect();
        assert!(got.contains(&handles[0].raw));
        assert!(got.contains(&handles[2].raw));
        assert!(!got.contains(&handles[1].raw));
    }

    #[test]
    fn similar_estimate_saturates_when_other_unbound() {
        // The engine's "unconstrained variable" check rejects
        // `None` at construction time — so `similar` reports
        // `usize::MAX` (infinitely expensive) until the other
        // side is bound, and `None` only for unrelated vars.
        let (flat, _hnsw, mut store, _handles) = sample_sim();
        let reader = store.reader().unwrap();
        let view = flat.attach(&reader);

        let mut ctx = triblespace::core::query::VariableContext::new();
        let a: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let b: Variable<Handle<Blake3, Embedding>> = ctx.next_variable();
        let unrelated: Variable<GenId> = ctx.next_variable();
        let c = view.similar(a, b, 0.8);

        assert_eq!(c.estimate(a.index, &Binding::default()), Some(usize::MAX));
        assert_eq!(c.estimate(b.index, &Binding::default()), Some(usize::MAX));
        assert_eq!(c.estimate(unrelated.index, &Binding::default()), None);
    }
}
