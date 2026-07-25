use super::*;
use ahash::AHashSet;
use itertools::Itertools;

/// Logical disjunction of constraints (OR).
///
/// A value is accepted if *any* variant accepts it. Built by the
/// [`or!`](crate::or) macro, by [`pattern_changes!`](crate::macros::pattern_changes),
/// or directly via [`new`](Self::new).
///
/// All variants must declare the same [`VariableSet`]; this is asserted at
/// construction time. Branch-local variables are unsupported because the
/// engine's result schema is flat — every row binds the same variable set
/// exactly once, so a variable that exists only in some alternatives has
/// no representation. (This is a result-model restriction, not a semantic
/// one: the union itself is monotonic.) Since `pattern!` folds attribute
/// constants and literal values into constant [`Term`](crate::query::Term)s
/// (they never become variables), the requirement is about the *query
/// variables the caller wrote*: every arm must mention the same ones.
/// Estimates are summed across variants. Proposals concatenate each live
/// arm's occurrence stream in arm order, and confirmation retains every
/// original occurrence supported by at least one live arm. Logical
/// idempotence belongs to the engine's SET-admission boundary rather than this
/// physical composite.
///
/// Before proposing or confirming, the union checks each variant's
/// [`satisfied`](Constraint::satisfied) status and skips variants that are
/// provably dead. This prevents a value confirmed by a dead variant from
/// leaking into the result set — the fix for spurious results in
/// multi-entity [`pattern_changes!`](crate::macros::pattern_changes) joins.
pub struct UnionConstraint<C> {
    constraints: Vec<C>,
}

impl<'a, C> UnionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    /// Creates a union over the given constraints.
    ///
    /// # Panics
    ///
    /// Panics if `constraints` is empty (a zero-arm union has no
    /// well-defined variable set), or if the variants do not all
    /// declare the same variable set.
    pub fn new(constraints: Vec<C>) -> Self {
        assert!(
            !constraints.is_empty(),
            "UnionConstraint requires at least one variant; \
             use a different constraint type for the empty case"
        );
        if let Some((i, (a, b))) = constraints
            .iter()
            .map(|c| c.variables())
            .tuple_windows()
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            panic!(
                "all union (or!) variants must mention the same query \
                 variables: variant {} declares {:?} but variant {} \
                 declares {:?}",
                i,
                a,
                i + 1,
                b
            );
        }
        UnionConstraint { constraints }
    }

    fn source_estimate(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        out: &mut EstimateSink<'_>,
    ) -> bool {
        if !self.variables().is_set(variable) {
            return false;
        }

        match out {
            EstimateSink::Scalar(slot) => {
                let mut total = 0usize;
                for constraint in &self.constraints {
                    if !constraint.satisfied(view) {
                        continue;
                    }
                    let mut estimate = 0usize;
                    if !constraint.estimate(
                        variable,
                        view,
                        &mut EstimateSink::Scalar(&mut estimate),
                    ) {
                        total = usize::MAX;
                        break;
                    }
                    total = total.saturating_add(estimate);
                }
                **slot = total;
            }
            EstimateSink::Column(out) => {
                let mut totals = vec![0usize; view.len()];
                let mut scratch = Vec::new();
                for constraint in &self.constraints {
                    scratch.clear();
                    let quoted = constraint.estimate(
                        variable,
                        view,
                        &mut EstimateSink::Column(&mut scratch),
                    );
                    if quoted {
                        debug_assert_eq!(scratch.len(), view.len());
                    }
                    for (row, total) in totals.iter_mut().enumerate() {
                        if !constraint.satisfied(&view.row_view(row)) {
                            continue;
                        }
                        if quoted {
                            *total = total.saturating_add(scratch[row]);
                        } else {
                            *total = usize::MAX;
                        }
                    }
                }
                out.extend(totals);
            }
        }
        true
    }

    fn propose_union(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        debug_assert!(
            candidates.is_empty(),
            "propose expects an empty sink (see the Constraint::propose protocol law)"
        );
        let mut variant_values: Vec<RawInline> = Vec::new();
        for (i, row) in view.iter().enumerate() {
            let row_view = RowsView::new(view.vars, row);
            self.constraints
                .iter()
                .filter(|constraint| constraint.satisfied(&row_view))
                .for_each(|constraint| {
                    debug_assert!(
                        variant_values.is_empty(),
                        "each union arm must receive an independent empty proposal sink"
                    );
                    constraint.propose(
                        variable,
                        &row_view,
                        &mut CandidateSink::Values(&mut variant_values),
                    );
                    candidates.extend_row(i as u32, variant_values.drain(..));
                });
        }
    }

    fn confirm_union(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        confirm_per_row(view, candidates, |row, values| {
            let row_view = RowsView::new(view.vars, row);
            let mut supported = AHashSet::new();
            for constraint in self
                .constraints
                .iter()
                .filter(|constraint| constraint.satisfied(&row_view))
            {
                // Every arm confirms the complete original occurrence stream.
                // The support set records only the relational union; retaining
                // below preserves the caller's physical order and multiplicity.
                let mut survivors = values.clone();
                constraint.confirm(
                    variable,
                    &row_view,
                    &mut CandidateSink::Values(&mut survivors),
                );
                supported.extend(survivors);
            }
            values.retain(|value| supported.contains(value));
        });
    }
}

impl<'a, C> Constraint<'a> for UnionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    /// Returns the variable set of the first variant (all variants share
    /// the same set, enforced at construction).
    fn variables(&self) -> VariableSet {
        self.constraints[0].variables()
    }

    /// Every potentially live arm must cover the target. The union receipt is
    /// therefore the meet of its arm receipts: one `None` arm removes source
    /// eligibility, all-`Exact` remains exact, and every other complete mix is
    /// covering.
    fn proposal_coverage(&self, variable: VariableId, bound: VariableSet) -> ProposalCoverage {
        if bound.is_set(variable) || !self.variables().is_set(variable) {
            return ProposalCoverage::None;
        }
        self.constraints
            .iter()
            .map(|constraint| constraint.proposal_coverage(variable, bound))
            .min()
            .unwrap_or(ProposalCoverage::None)
    }

    /// Appends the elementwise **sum** of estimates across all potentially
    /// live variants. Any such arm without a quote makes that row's cost
    /// unknown ([`usize::MAX`]); an arm may be omitted only when
    /// [`satisfied`](Constraint::satisfied) proves it dead.
    fn estimate(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        out: &mut EstimateSink<'_>,
    ) -> bool {
        self.source_estimate(variable, view, out)
    }

    /// Per row: concatenates proposals from every *satisfied* variant (via a
    /// single-row borrowed view), preserving arm order and every physical
    /// occurrence. Dead variants (where [`satisfied`](Constraint::satisfied)
    /// returns `false` for the row) are skipped so their stale bindings cannot
    /// inject values that no live variant would produce.
    ///
    /// Each variant proposes into its **own empty buffer** and the union
    /// concatenates the independent per-variant outputs. This upholds the
    /// empty-sink law of [`propose`](Constraint::propose): a composite
    /// variant (e.g. an intersection) filters the sink it is handed via its
    /// children's `confirm`, so sharing one buffer across variants would let
    /// a later variant delete candidates an earlier variant produced —
    /// making the result depend on variant order and, worse, letting a
    /// monotonic growth of the underlying data *remove* results (a CALM
    /// violation observed in [`pattern_changes!`](crate::macros::pattern_changes)
    /// joins).
    fn propose(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        self.propose_union(variable, view, candidates)
    }

    /// Confirms each row's complete original candidate group against every
    /// *satisfied* variant independently. The union of survivor values is
    /// treated as relational support, then applied to the original stream so
    /// order and multiplicity are retained. A value passes if *any* live
    /// variant confirms it.
    fn confirm(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        self.confirm_union(variable, view, candidates)
    }

    /// Returns `true` when **at least one** variant is satisfied for
    /// every row.
    fn satisfied(&self, view: &RowsView<'_>) -> bool {
        view.iter().all(|row| {
            let row_view = RowsView::new(view.vars, row);
            self.constraints.iter().any(|c| c.satisfied(&row_view))
        })
    }

    /// Returns the union of all variants' influence sets for `variable`.
    fn influence(&self, variable: VariableId) -> VariableSet {
        self.constraints
            .iter()
            .fold(VariableSet::new_empty(), |acc, c| {
                acc.union(c.influence(variable))
            })
    }

    fn residual_union_children(&self) -> Option<&dyn ConstraintChildren<'a>> {
        Some(self)
    }
}

impl<'a, C> ConstraintChildren<'a> for UnionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    fn len(&self) -> usize {
        self.constraints.len()
    }

    fn child(&self, index: usize) -> &dyn Constraint<'a> {
        &self.constraints[index]
    }
}

/// Combines constraints into a [`UnionConstraint`] (logical OR).
///
/// A result is produced when *any* of the given constraints is satisfied.
/// All constraints must declare the same variable set.
///
/// ```rust,ignore
/// or!(pattern!(&set_a, [...]), pattern!(&set_b, [...]))
/// ```
#[macro_export]
macro_rules! or {
    ($($c:expr),+ $(,)?) => (
        ::std::sync::Arc::new(
            $crate::query::unionconstraint::UnionConstraint::new(vec![
                $(Box::new($c)
                    as Box<dyn $crate::query::Constraint + Send + Sync>),+
            ])
        )
    )
}

/// Re-export of the [`or!`] macro.
pub use or;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inline::encodings::UnknownInline;
    use crate::query::constantconstraint::ConstantConstraint;

    const X: VariableId = 0;
    const Y: VariableId = 1;
    const DEAD: RawInline = [0x21; 32];
    const LIVE: RawInline = [0x42; 32];

    #[derive(Clone, Copy)]
    enum Liveness {
        Always,
        Never,
        WhenY(RawInline),
    }

    #[derive(Clone, Copy)]
    struct EstimateArm {
        quote: Option<usize>,
        liveness: Liveness,
    }

    impl EstimateArm {
        fn row_is_live(&self, view: &RowsView<'_>, row: &[RawInline]) -> bool {
            match self.liveness {
                Liveness::Always => true,
                Liveness::Never => false,
                Liveness::WhenY(value) => view.col(Y).is_none_or(|column| row[column] == value),
            }
        }
    }

    impl Constraint<'static> for EstimateArm {
        fn variables(&self) -> VariableSet {
            VariableSet::new_singleton(X).union(VariableSet::new_singleton(Y))
        }

        fn estimate(
            &self,
            variable: VariableId,
            view: &RowsView<'_>,
            out: &mut EstimateSink<'_>,
        ) -> bool {
            let Some(quote) = self.quote.filter(|_| variable == X) else {
                return false;
            };
            out.fill(quote, view.len());
            true
        }

        fn propose(
            &self,
            _variable: VariableId,
            _view: &RowsView<'_>,
            _candidates: &mut CandidateSink<'_>,
        ) {
        }

        fn confirm(
            &self,
            _variable: VariableId,
            _view: &RowsView<'_>,
            _candidates: &mut CandidateSink<'_>,
        ) {
        }

        fn satisfied(&self, view: &RowsView<'_>) -> bool {
            view.iter().all(|row| self.row_is_live(view, row))
        }
    }

    #[test]
    fn union_marks_only_rows_with_a_live_unquoted_arm_unknown() {
        let constraint = UnionConstraint::new(vec![
            EstimateArm {
                quote: Some(3),
                liveness: Liveness::Always,
            },
            EstimateArm {
                quote: None,
                liveness: Liveness::WhenY(LIVE),
            },
        ]);
        let rows = [DEAD, LIVE];
        let view = RowsView::new(&[Y], &rows);
        let mut estimates = Vec::new();

        assert!(constraint.estimate(X, &view, &mut EstimateSink::Column(&mut estimates),));
        assert_eq!(estimates, vec![3, usize::MAX]);
    }

    #[test]
    fn union_ignores_an_unquoted_arm_proven_dead() {
        let constraint = UnionConstraint::new(vec![
            EstimateArm {
                quote: Some(3),
                liveness: Liveness::Always,
            },
            EstimateArm {
                quote: None,
                liveness: Liveness::Never,
            },
        ]);
        let mut estimate = 0;

        assert!(constraint.estimate(
            X,
            &RowsView::EMPTY,
            &mut EstimateSink::Scalar(&mut estimate),
        ));
        assert_eq!(estimate, 3);
    }

    #[test]
    #[should_panic(expected = "UnionConstraint requires at least one variant")]
    fn empty_union_panics_at_construction() {
        // Without this assert, `variables()` would later panic on
        // `self.constraints[0]` with an unhelpful index-out-of-bounds.
        let _: UnionConstraint<ConstantConstraint> = UnionConstraint::new(vec![]);
    }

    #[test]
    fn proposal_concatenates_arm_occurrences_in_order() {
        let x = Variable::<UnknownInline>::new(X);
        let constraint = UnionConstraint::new(vec![
            ConstantConstraint::new(x, Inline::new(DEAD)),
            ConstantConstraint::new(x, Inline::new(LIVE)),
            ConstantConstraint::new(x, Inline::new(DEAD)),
        ]);
        let mut values = Vec::new();

        constraint.propose(X, &RowsView::EMPTY, &mut CandidateSink::Values(&mut values));

        assert_eq!(values, [DEAD, LIVE, DEAD]);
    }

    #[test]
    fn confirmation_preserves_supported_occurrence_order_and_multiplicity() {
        let x = Variable::<UnknownInline>::new(X);
        let constraint = UnionConstraint::new(vec![
            ConstantConstraint::new(x, Inline::new(DEAD)),
            ConstantConstraint::new(x, Inline::new(LIVE)),
        ]);
        let unsupported = [0x63; 32];
        let mut values = vec![LIVE, DEAD, DEAD, unsupported, LIVE];

        constraint.confirm(X, &RowsView::EMPTY, &mut CandidateSink::Values(&mut values));

        assert_eq!(values, [LIVE, DEAD, DEAD, LIVE]);
    }
}
