use std::sync::Arc;

use crate::inline::encodings::genid::GenId;
use crate::inline::InlineEncoding;
use crate::query::unionconstraint::UnionConstraint;
use crate::query::{
    Binding, Candidates, Constraint, Frontier, ProposalBuffer, Term, TriblePattern, VariableId,
    VariableSet,
};

use super::{SuccinctArchive, SuccinctArchiveConstraint, Universe};

/// A [`TriblePattern`] view that unions several Succinct archive shards.
///
/// Owns its shard list (`Arc<[SuccinctArchive]>` — the archives underneath
/// are `Bytes`/`Arc`-backed views, so cloning shards in is a handful of
/// refcount bumps, never a data copy). Ownership makes the union `'static`
/// wherever its universe is, so it can flow into type-erased consumers
/// without borrowed-slice gymnastics.
#[derive(Clone)]
pub struct UnionArchive<U> {
    segments: Arc<[SuccinctArchive<U>]>,
}

impl<U> UnionArchive<U> {
    /// Wrap attached physical shards.
    ///
    /// # Panics
    ///
    /// Panics when `segments` is empty. A physical union requires at least
    /// one shard; use a different constraint to represent an empty relation.
    pub fn new(segments: impl Into<Arc<[SuccinctArchive<U>]>>) -> Self {
        let segments = segments.into();
        assert!(
            !segments.is_empty(),
            "UnionArchive requires at least one physical shard"
        );
        Self { segments }
    }

    /// Number of physical Succinct shards behind this logical union.
    ///
    /// This is storage provenance, not a logical cardinality: compaction may
    /// change it without changing the relation exposed by [`TriblePattern`].
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// Atomic normalized union over one finite set of Succinct archive shards.
///
/// A thin wrapper over [`UnionConstraint`]: every shard constraint carries
/// the pattern's [`Term`]s natively (constant positions included), so all
/// shards declare the same variable set by construction and the union's
/// equal-variable-set requirement holds trivially. The wrapper exists so
/// the shard union stays structurally opaque — one logical source, not a
/// user-visible `or!` that formula rewrites could split back into
/// independently materialized arms.
pub struct UnionArchiveConstraint<'a, U>
where
    U: Universe,
{
    union: UnionConstraint<SuccinctArchiveConstraint<'a, U>>,
}

impl<'a, U> UnionArchiveConstraint<'a, U>
where
    U: Universe,
{
    fn new(constraints: Vec<SuccinctArchiveConstraint<'a, U>>) -> Self {
        Self {
            union: UnionConstraint::new(constraints),
        }
    }
}

impl<'a, U> Constraint<'a> for UnionArchiveConstraint<'a, U>
where
    U: Universe,
{
    fn variables(&self) -> VariableSet {
        self.union.variables()
    }

    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        self.union.estimate(variable, binding)
    }

    fn propose(
        &self,
        variable: VariableId,
        frontier: &Frontier<'_>,
        proposals: &mut ProposalBuffer,
    ) {
        self.union.propose(variable, frontier, proposals)
    }

    fn confirm(&self, variable: VariableId, frontier: &Frontier<'_>, cands: &mut Candidates<'_>) {
        self.union.confirm(variable, frontier, cands)
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        self.union.satisfied(binding)
    }

    fn influence(&self, variable: VariableId) -> VariableSet {
        self.union.influence(variable)
    }
}

impl<U> TriblePattern for UnionArchive<U>
where
    U: Universe + Send + Sync,
{
    type PatternConstraint<'p>
        = UnionArchiveConstraint<'p, U>
    where
        Self: 'p;

    fn pattern<'p, V: InlineEncoding>(
        &'p self,
        e: impl Into<Term<GenId>>,
        a: impl Into<Term<GenId>>,
        v: impl Into<Term<V>>,
    ) -> Self::PatternConstraint<'p> {
        let e: Term<GenId> = e.into();
        let a: Term<GenId> = a.into();
        let v: Term<V> = v.into();
        UnionArchiveConstraint::new(
            self.segments
                .iter()
                .map(|segment| segment.pattern(e, a, v))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::examples::literature;
    use crate::prelude::*;

    use super::*;
    use crate::blob::encodings::succinctarchive::OrderedUniverse;

    #[test]
    fn query_union_deduplicates_overlap_across_shards() {
        let ada = ufoid();
        let grace = ufoid();

        let mut left = TribleSet::new();
        left += entity! { &ada @ literature::firstname: "Ada" };

        let mut right = TribleSet::new();
        right += entity! { &ada @ literature::firstname: "Ada" };
        right += entity! { &grace @ literature::firstname: "Grace" };

        let segments: Vec<SuccinctArchive<OrderedUniverse>> =
            [&left, &right].into_iter().map(Into::into).collect();
        let union = UnionArchive::new(segments);

        let mut names: Vec<String> = find!(
            name: String,
            pattern!(&union, [{ _?person @ literature::firstname: ?name }])
        )
        .collect();
        names.sort();

        assert_eq!(union.segment_count(), 2);
        assert_eq!(names, ["Ada", "Grace"]);
    }

    #[test]
    #[should_panic(expected = "UnionArchive requires at least one physical shard")]
    fn rejects_empty_physical_union() {
        UnionArchive::<OrderedUniverse>::new(Vec::<SuccinctArchive<OrderedUniverse>>::new());
    }
}
