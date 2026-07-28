use super::*;
use smallvec::SmallVec;

/// Logical conjunction of constraints (AND).
///
/// All children must agree on every variable binding. Built by the
/// [`and!`](crate::and) macro or directly via [`new`](Self::new).
///
/// The intersection delegates to its children using cardinality-aware
/// ordering: the child with the lowest [`estimate`](Constraint::estimate)
/// proposes candidates, and the remaining children
/// [`confirm`](Constraint::confirm) them in order of increasing estimate.
/// This strategy keeps the candidate set small from the start and avoids
/// materialising cross products.
///
/// Variables from all children are exposed as a single union, so the
/// engine sees one flat set of variables regardless of how many
/// sub-constraints contribute.
pub struct IntersectionConstraint<C> {
    constraints: Vec<C>,
}

impl<'a, C> IntersectionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    /// Creates an intersection over the given constraints.
    pub fn new(constraints: Vec<C>) -> Self {
        IntersectionConstraint { constraints }
    }
}

impl<'a, C> Constraint<'a> for IntersectionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    /// Returns the union of all children's variable sets.
    fn variables(&self) -> VariableSet {
        self.constraints
            .iter()
            .fold(VariableSet::new_empty(), |vs, c| vs.union(c.variables()))
    }

    /// Returns the **minimum** estimate across children that constrain
    /// `variable`. The tightest child bounds the search, reflecting the
    /// intersection semantics: every child must agree, so the smallest
    /// candidate set dominates.
    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        self.constraints
            .iter()
            .filter_map(|c| c.estimate(variable, binding))
            .min()
    }

    /// Lets each row's tightest child propose, then confirms through the
    /// rest — kills land in the region's liveness words; nothing is
    /// compacted. Children that return `None` for this variable are
    /// skipped.
    ///
    /// The tightest child is a *per-row* decision, exactly as the variable
    /// choice is: an estimate is a function of the row's binding, and one
    /// row's binding can make the archive the selective source while
    /// another's makes a hash set selective. Rows that agree — the common
    /// case — travel as one batch with no row copied; genuine disagreement
    /// pays for one [`Frontier::with_select`] sub-batch per source, which
    /// is what keeps each `propose`/`confirm` pass homogeneous enough for a
    /// batched executor to dispatch it as a unit.
    ///
    /// Only the tail region this call appended (from the incoming buffer
    /// length onward) is confirmed, so proposals appended by sibling
    /// constraints in an enclosing composite are never filtered through
    /// this intersection's children.
    fn propose(
        &self,
        variable: VariableId,
        frontier: &Frontier<'_>,
        proposals: &mut ProposalBuffer,
    ) {
        let rows = frontier.len();
        if rows == 0 {
            return;
        }

        // Per-row: the tightest child, and which children have an opinion
        // about `variable` at all.
        let mut choice: Vec<usize> = Vec::with_capacity(rows);
        let mut relevant: SmallVec<[bool; 8]> = SmallVec::from_elem(false, self.constraints.len());
        for row in 0..rows {
            let binding = frontier.row(row);
            let mut best = usize::MAX;
            let mut best_child = usize::MAX;
            for (i, c) in self.constraints.iter().enumerate() {
                if let Some(estimate) = c.estimate(variable, &binding) {
                    relevant[i] = true;
                    if best_child == usize::MAX || estimate < best {
                        best = estimate;
                        best_child = i;
                    }
                }
            }
            choice.push(best_child);
        }

        // No child constrains this variable anywhere in the batch.
        if choice.iter().all(|&c| c == usize::MAX) {
            return;
        }

        let confirmers: SmallVec<[usize; 8]> = relevant
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| r.then_some(i))
            .collect();

        let single = choice.iter().all(|&c| c == choice[0]);
        if single {
            let base = proposals.len();
            self.constraints[choice[0]].propose(variable, frontier, proposals);
            let mut region = proposals.region(base);
            for &i in confirmers.iter().filter(|&&i| i != choice[0]) {
                self.constraints[i].confirm(variable, frontier, &mut region);
            }
            return;
        }

        let mut select: Vec<u32> = Vec::new();
        let mut positions: Vec<u32> = Vec::new();
        for proposer in 0..self.constraints.len() {
            positions.clear();
            positions.extend(
                choice
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c == proposer)
                    .map(|(row, _)| row as u32),
            );
            if positions.is_empty() {
                continue;
            }
            select.clear();
            frontier.compose(positions.iter().copied(), &mut select);
            let sub = frontier.with_select(&select);

            let base = proposals.len();
            self.constraints[proposer].propose(variable, &sub, proposals);
            let mut region = proposals.region(base);
            for &i in confirmers.iter().filter(|&&i| i != proposer) {
                self.constraints[i].confirm(variable, &sub, &mut region);
            }
            // The child tagged its candidates with sub-batch row numbers;
            // lift them back into this frontier's coordinates before the
            // region reaches our caller.
            proposals.remap_region(base, &positions);
        }
    }

    /// Confirms proposals through all children that constrain `variable`,
    /// all killing into the shared mask.
    ///
    /// Ordering the children by estimate is a pure cost heuristic — kills
    /// conjoin regardless — so the batch's first row supplies the ordering
    /// for the whole pass rather than each row re-deciding it.
    fn confirm(&self, variable: VariableId, frontier: &Frontier<'_>, cands: &mut Candidates<'_>) {
        if frontier.is_empty() {
            return;
        }
        let binding = frontier.row(0);
        let mut relevant_constraints: SmallVec<[(usize, &C); 8]> = self
            .constraints
            .iter()
            .filter_map(|c| Some((c.estimate(variable, &binding)?, c)))
            .collect();
        relevant_constraints.sort_unstable_by_key(|(estimate, _)| *estimate);

        for (_, c) in relevant_constraints.iter() {
            c.confirm(variable, frontier, cands);
        }
    }

    /// Returns `true` only when **every** child is satisfied.
    fn satisfied(&self, binding: &Binding) -> bool {
        self.constraints.iter().all(|c| c.satisfied(binding))
    }

    /// Returns the union of all children's influence sets for `variable`.
    fn influence(&self, variable: VariableId) -> VariableSet {
        self.constraints
            .iter()
            .fold(VariableSet::new_empty(), |acc, c| {
                acc.union(c.influence(variable))
            })
    }
}

/// Combines constraints into an [`IntersectionConstraint`] (logical AND).
///
/// All constraints must agree on every variable binding for a result to
/// be produced. Accepts one or more constraint expressions.
///
/// ```rust,ignore
/// and!(set.pattern(e, a, v), allowed.has(v))
/// ```
#[macro_export]
macro_rules! and {
    // Emits `Arc<IntersectionConstraint<Box<dyn Constraint + Send + Sync>>>`.
    // The outer `Arc` makes the whole tree cheap to `Clone` (single
    // refcount bump) — required by the `parallel` feature's `Query::clone`
    // during rayon split. `Send + Sync` on the trait object lets the tree
    // cross rayon thread boundaries. Every in-tree constraint built via
    // this macro already satisfies Send + Sync; non-thread-safe constraint
    // types (e.g. `Rc`-backed ContainsConstraint variants) can still be
    // used via direct `IntersectionConstraint::new` construction.
    ($($c:expr),+ $(,)?) => (
        ::std::sync::Arc::new(
            $crate::query::intersectionconstraint::IntersectionConstraint::new(vec![
                $(Box::new($c)
                    as Box<dyn $crate::query::Constraint + Send + Sync>),+
            ])
        )
    )
}

/// Re-export of the [`and!`] macro.
pub use and;
