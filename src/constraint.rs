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

use triblespace::core::id::{Id, RawId};
use triblespace::core::query::{Binding, Constraint, Variable, VariableId, VariableSet};
use triblespace::core::value::schemas::genid::GenId;
use triblespace::core::value::RawValue;

use crate::bm25::BM25Index;
use crate::hnsw::{FlatIndex, HNSWIndex};

/// Constrains a `Variable<GenId>` (doc) to entity ids in the
/// posting list of the pinned term.
///
/// Created via [`BM25Index::docs_containing`].
pub struct DocsContainingTerm<'a> {
    index: &'a BM25Index,
    doc: Variable<GenId>,
    /// The term value is pinned at constraint-construction time.
    term: [u8; 32],
}

impl<'a> DocsContainingTerm<'a> {
    pub fn new(index: &'a BM25Index, doc: Variable<GenId>, term: [u8; 32]) -> Self {
        Self { index, doc, term }
    }
}

impl BM25Index {
    /// Produce a [`DocsContainingTerm`] constraint for use inside
    /// `pattern!` / `find!`.
    pub fn docs_containing(
        &self,
        doc: Variable<GenId>,
        term: [u8; 32],
    ) -> DocsContainingTerm<'_> {
        DocsContainingTerm::new(self, doc, term)
    }
}

/// Convert a `GenId`-schema raw value (zero-padded in bytes
/// [0..16], Id in [16..32]) back to an `Id`. Returns `None` for
/// malformed / nil values.
fn raw_value_to_id(raw: &RawValue) -> Option<Id> {
    if raw[0..16] != [0u8; 16] {
        return None;
    }
    let raw_id: RawId = raw[16..32].try_into().ok()?;
    Id::new(raw_id)
}

/// Encode an `Id` as a `GenId`-schema raw value.
fn id_to_raw_value(id: Id) -> RawValue {
    let mut out = [0u8; 32];
    let raw: &RawId = id.as_ref();
    out[16..32].copy_from_slice(raw);
    out
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

    fn propose(
        &self,
        variable: VariableId,
        _binding: &Binding,
        proposals: &mut Vec<RawValue>,
    ) {
        if self.doc.index != variable {
            return;
        }
        for (doc_id, _score) in self.index.similar(&self.query, self.k) {
            proposals.push(id_to_raw_value(doc_id));
        }
    }

    fn confirm(
        &self,
        variable: VariableId,
        _binding: &Binding,
        proposals: &mut Vec<RawValue>,
    ) {
        if self.doc.index != variable {
            return;
        }
        // Collect the top-k result ids and keep only proposals
        // whose doc appears in that set.
        let top: HashSet<Id> = self
            .index
            .similar(&self.query, self.k)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        proposals.retain(|raw| {
            raw_value_to_id(raw)
                .map(|id| top.contains(&id))
                .unwrap_or(false)
        });
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.doc.index) {
            Some(raw) => {
                let Some(doc_id) = raw_value_to_id(raw) else {
                    return false;
                };
                self.index
                    .similar(&self.query, self.k)
                    .into_iter()
                    .any(|(d, _)| d == doc_id)
            }
            None => true,
        }
    }
}

// ── HNSWIndex constraint ─────────────────────────────────────────────

/// HNSW version of [`SimilarToVector`]. Same shape; different
/// backing index. Approximate rather than exact top-k.
pub struct SimilarToVectorHNSW<'a> {
    index: &'a HNSWIndex,
    doc: Variable<GenId>,
    query: Vec<f32>,
    k: usize,
    /// HNSW's `ef_search` parameter. Larger = better recall at
    /// higher query cost. Defaults to `k` when `None`.
    ef: Option<usize>,
}

impl<'a> SimilarToVectorHNSW<'a> {
    pub fn new(
        index: &'a HNSWIndex,
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
    ) -> SimilarToVectorHNSW<'_> {
        SimilarToVectorHNSW::new(self, doc, query, k, ef)
    }
}

impl<'a> Constraint<'a> for SimilarToVectorHNSW<'a> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            Some(self.k.min(self.index.doc_count()))
        } else {
            None
        }
    }

    fn propose(
        &self,
        variable: VariableId,
        _binding: &Binding,
        proposals: &mut Vec<RawValue>,
    ) {
        if self.doc.index != variable {
            return;
        }
        for (doc_id, _score) in self.index.similar(&self.query, self.k, self.ef) {
            proposals.push(id_to_raw_value(doc_id));
        }
    }

    fn confirm(
        &self,
        variable: VariableId,
        _binding: &Binding,
        proposals: &mut Vec<RawValue>,
    ) {
        if self.doc.index != variable {
            return;
        }
        let top: HashSet<Id> = self
            .index
            .similar(&self.query, self.k, self.ef)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        proposals.retain(|raw| {
            raw_value_to_id(raw)
                .map(|id| top.contains(&id))
                .unwrap_or(false)
        });
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.doc.index) {
            Some(raw) => {
                let Some(doc_id) = raw_value_to_id(raw) else {
                    return false;
                };
                self.index
                    .similar(&self.query, self.k, self.ef)
                    .into_iter()
                    .any(|(d, _)| d == doc_id)
            }
            None => true,
        }
    }
}

impl<'a> Constraint<'a> for DocsContainingTerm<'a> {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.doc.index)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.doc.index == variable {
            Some(self.index.doc_frequency(&self.term))
        } else {
            None
        }
    }

    fn propose(
        &self,
        variable: VariableId,
        _binding: &Binding,
        proposals: &mut Vec<RawValue>,
    ) {
        if self.doc.index != variable {
            return;
        }
        for (doc_id, _score) in self.index.query_term(&self.term) {
            proposals.push(id_to_raw_value(doc_id));
        }
    }

    fn confirm(
        &self,
        variable: VariableId,
        _binding: &Binding,
        proposals: &mut Vec<RawValue>,
    ) {
        if self.doc.index != variable {
            return;
        }
        // Collect the term's posting list into a set for O(1)
        // membership checks. For very long posting lists a sorted
        // binary-search would be lighter on memory; for v1 this is
        // fine up to tens of thousands of postings.
        let docs: HashSet<Id> =
            self.index.query_term(&self.term).map(|(d, _)| d).collect();
        proposals.retain(|raw| {
            raw_value_to_id(raw)
                .map(|id| docs.contains(&id))
                .unwrap_or(false)
        });
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        // If `doc` is already bound, the constraint is satisfied
        // iff that doc appears in the term's posting list.
        match binding.get(self.doc.index) {
            Some(raw) => {
                let Some(doc_id) = raw_value_to_id(raw) else {
                    return false;
                };
                self.index.query_term(&self.term).any(|(d, _)| d == doc_id)
            }
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::BM25Builder;
    use crate::tokens::hash_tokens;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn sample_index() -> BM25Index {
        let mut b = BM25Builder::new();
        b.insert(id(1), hash_tokens("the quick brown fox"));
        b.insert(id(2), hash_tokens("the lazy brown dog"));
        b.insert(id(3), hash_tokens("quick silver fox jumps"));
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
        let mut props: Vec<RawValue> =
            vec![id_to_raw_value(id(1)), id_to_raw_value(id(2)), id_to_raw_value(id(3))];
        c.confirm(doc.index, &binding, &mut props);
        // Doc 2 doesn't contain "fox" — should be filtered out.
        // Docs 1 and 3 do.
        let ids: HashSet<Id> =
            props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
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
        b.insert(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        b.insert(id(2), vec![0.0, 1.0, 0.0]).unwrap();
        b.insert(id(3), vec![0.9, 0.1, 0.0]).unwrap();
        b.insert(id(4), vec![0.0, 0.0, 1.0]).unwrap();
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
        let ids: HashSet<Id> =
            props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
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
        let ids: HashSet<Id> =
            props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
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

    fn sample_hnsw() -> crate::hnsw::HNSWIndex {
        use crate::hnsw::HNSWBuilder;
        let mut b = HNSWBuilder::new(3).with_seed(42);
        b.insert(id(1), vec![1.0, 0.0, 0.0]).unwrap();
        b.insert(id(2), vec![0.0, 1.0, 0.0]).unwrap();
        b.insert(id(3), vec![0.9, 0.1, 0.0]).unwrap();
        b.insert(id(4), vec![0.0, 0.0, 1.0]).unwrap();
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
        let ids: HashSet<Id> =
            props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
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
