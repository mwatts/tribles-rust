use super::*;

/// Constrains two variables to have the same value.
///
/// Used to express variable equality when two positions in a triple
/// share the same logical variable but need distinct [`VariableId`]s
/// for the `TribleSetConstraint`
/// (which assumes its three positions have distinct ids).
///
/// The macro layer emits this automatically when a `_?var` appears in
/// both the entity and value positions of the same triple.
pub struct EqualityConstraint {
    a: VariableId,
    b: VariableId,
}

impl EqualityConstraint {
    /// Creates a constraint requiring `a` and `b` to be bound to the
    /// same raw value.
    pub fn new(a: VariableId, b: VariableId) -> Self {
        EqualityConstraint { a, b }
    }

    /// Column of the peer of `variable` in `view`, when `variable` is
    /// one of the constrained pair and the peer is bound.
    fn peer_col(&self, variable: VariableId, view: &RowsView<'_>) -> Option<usize> {
        let peer = if variable == self.a {
            self.b
        } else if variable == self.b {
            self.a
        } else {
            return None;
        };
        view.col(peer)
    }

    fn peer(&self, variable: VariableId) -> Option<VariableId> {
        if variable == self.a {
            Some(self.b)
        } else if variable == self.b {
            Some(self.a)
        } else {
            None
        }
    }
}

impl<'c> Constraint<'c> for EqualityConstraint {
    fn variables(&self) -> VariableSet {
        let mut vs = VariableSet::new_empty();
        vs.set(self.a);
        vs.set(self.b);
        vs
    }

    /// Equality becomes an exact finite source only after the peer variable is
    /// bound. With both variables free it remains a validator rather than
    /// pretending to own the universe of raw values.
    fn proposal_coverage(&self, variable: VariableId, bound: VariableSet) -> ProposalCoverage {
        if bound.is_set(variable) {
            return ProposalCoverage::None;
        }
        match self.peer(variable) {
            Some(peer) if peer != variable && bound.is_set(peer) => ProposalCoverage::Exact,
            _ => ProposalCoverage::None,
        }
    }

    /// Estimates exactly one candidate per row when the peer variable is
    /// already bound. Returns `false` when the peer is unbound — the
    /// constraint has no independent opinion about the variable's
    /// cardinality and defers to other constraints in the intersection.
    /// This is safe as long as each variable also appears in at least
    /// one other constraint (which the macro desugaring guarantees).
    fn estimate(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        out: &mut EstimateSink<'_>,
    ) -> bool {
        if self.peer_col(variable, view).is_none() {
            return false;
        }
        out.fill(1, view.len());
        true
    }

    /// Proposes each row's peer value.
    fn propose(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        let Some(col) = self.peer_col(variable, view) else {
            return;
        };
        for (i, row) in view.iter().enumerate() {
            candidates.push(i as u32, row[col]);
        }
    }

    /// Retains only candidates matching their row's peer value.
    fn confirm(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        let Some(col) = self.peer_col(variable, view) else {
            return;
        };
        candidates.retain(|row, v| *v == view.row(row as usize)[col]);
    }

    /// Returns `false` when any row binds the pair to different values.
    fn satisfied(&self, view: &RowsView<'_>) -> bool {
        match (view.col(self.a), view.col(self.b)) {
            (Some(ca), Some(cb)) => view.iter().all(|row| row[ca] == row[cb]),
            _ => true,
        }
    }
}
