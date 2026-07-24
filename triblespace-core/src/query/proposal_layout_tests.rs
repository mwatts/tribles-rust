use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::debug::query::{DebugConstraint, EstimateOverrideConstraint};
use crate::inline::encodings::UnknownInline;
use crate::query::constantconstraint::ConstantConstraint;
use crate::query::equalityconstraint::EqualityConstraint;
use crate::query::intersectionconstraint::IntersectionConstraint;
use crate::query::unionconstraint::UnionConstraint;

use super::*;

const TARGET: VariableId = 0;
const SELECTOR: VariableId = 1;
const A: RawInline = [0x17; 32];
const B: RawInline = [0x83; 32];
const SELECT_LEFT: RawInline = [0x31; 32];
const SELECT_RIGHT: RawInline = [0x72; 32];

#[derive(Clone)]
struct BasicSource {
    values: &'static [RawInline],
    layout_is_set: bool,
    coverage: ProposalCoverage,
    accepted: &'static [RawInline],
}

impl Constraint<'static> for BasicSource {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(TARGET)
    }

    fn proposal_coverage(&self, variable: VariableId, bound: VariableSet) -> ProposalCoverage {
        if variable == TARGET && !bound.is_set(TARGET) {
            self.coverage
        } else {
            ProposalCoverage::None
        }
    }

    fn estimate(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        out: &mut EstimateSink<'_>,
    ) -> bool {
        if variable != TARGET {
            return false;
        }
        out.fill(self.values.len(), view.len());
        true
    }

    fn propose(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        if variable == TARGET {
            for row in 0..view.len() as u32 {
                candidates.extend_row(row, self.values.iter().copied());
            }
        }
    }

    fn propose_with_layout(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) -> ProposalLayout {
        self.propose(variable, view, candidates);
        if self.layout_is_set {
            ProposalLayout::grouped_set()
        } else {
            ProposalLayout::default()
        }
    }

    fn confirm(
        &self,
        variable: VariableId,
        _view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        if variable != TARGET {
            return;
        }
        candidates.retain(|_, value| self.accepted.contains(value));
    }

    fn satisfied(&self, view: &RowsView<'_>) -> bool {
        view.col(TARGET)
            .is_none_or(|column| view.iter().all(|row| self.accepted.contains(&row[column])))
    }
}

#[derive(Clone)]
struct CountingValidator {
    accepted: &'static [RawInline],
    calls: Arc<AtomicUsize>,
}

impl Constraint<'static> for CountingValidator {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(TARGET)
    }

    fn estimate(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        out: &mut EstimateSink<'_>,
    ) -> bool {
        if variable != TARGET {
            return false;
        }
        out.fill(self.accepted.len(), view.len());
        true
    }

    fn propose(
        &self,
        variable: VariableId,
        _view: &RowsView<'_>,
        _candidates: &mut CandidateSink<'_>,
    ) {
        assert_ne!(variable, TARGET, "confirmation-only child became a source");
    }

    fn confirm(
        &self,
        variable: VariableId,
        _view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        if variable == TARGET {
            self.calls.fetch_add(1, Ordering::Relaxed);
            candidates.retain(|_, value| self.accepted.contains(value));
        }
    }

    fn satisfied(&self, view: &RowsView<'_>) -> bool {
        view.col(TARGET)
            .is_none_or(|column| view.iter().all(|row| self.accepted.contains(&row[column])))
    }
}

#[derive(Clone)]
struct AdaptiveSource {
    preferred_selector: RawInline,
    layout_is_set: bool,
}

impl Constraint<'static> for AdaptiveSource {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(SELECTOR).union(VariableSet::new_singleton(TARGET))
    }

    fn proposal_coverage(&self, variable: VariableId, bound: VariableSet) -> ProposalCoverage {
        if variable == TARGET && bound.is_set(SELECTOR) && !bound.is_set(TARGET) {
            ProposalCoverage::Exact
        } else {
            ProposalCoverage::None
        }
    }

    fn estimate(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        out: &mut EstimateSink<'_>,
    ) -> bool {
        if variable != TARGET {
            return false;
        }
        let Some(selector) = view.col(SELECTOR) else {
            return false;
        };
        out.extend(view.iter().map(|row| {
            if row[selector] == self.preferred_selector {
                1
            } else {
                8
            }
        }));
        true
    }

    fn propose(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        if variable != TARGET {
            return;
        }
        let copies = if self.layout_is_set { 1 } else { 2 };
        for row in 0..view.len() as u32 {
            candidates.extend_row(row, std::iter::repeat_n(A, copies));
        }
    }

    fn propose_with_layout(
        &self,
        variable: VariableId,
        view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) -> ProposalLayout {
        self.propose(variable, view, candidates);
        if self.layout_is_set {
            ProposalLayout::grouped_set()
        } else {
            ProposalLayout::default()
        }
    }

    fn confirm(
        &self,
        variable: VariableId,
        _view: &RowsView<'_>,
        candidates: &mut CandidateSink<'_>,
    ) {
        if variable == TARGET {
            candidates.retain(|_, value| *value == A);
        }
    }

    fn satisfied(&self, view: &RowsView<'_>) -> bool {
        view.col(TARGET)
            .is_none_or(|column| view.iter().all(|row| row[column] == A))
    }
}

#[test]
fn covering_grouped_set_preserves_results_through_validation() {
    let validator_calls = Arc::new(AtomicUsize::new(0));
    let root: Arc<IntersectionConstraint<Box<dyn Constraint<'static> + Send + Sync>>> =
        Arc::new(IntersectionConstraint::new(vec![
            Box::new(BasicSource {
                values: &[B, A],
                layout_is_set: true,
                coverage: ProposalCoverage::Exact,
                accepted: &[A, B],
            }),
            Box::new(CountingValidator {
                accepted: &[A],
                calls: Arc::clone(&validator_calls),
            }),
        ]));
    assert_eq!(
        root.proposal_coverage(TARGET, VariableSet::new_empty()),
        ProposalCoverage::Covering
    );
    let values: Vec<_> = Query::new(root, |binding| binding.get(TARGET).copied()).collect();

    assert_eq!(values, [A]);
    assert!(
        validator_calls.load(Ordering::Relaxed) > 0,
        "the grouped-set receipt must not skip the confirmation-only child"
    );
}

fn nonuniform_nested_layout(second_is_set: bool) -> (ProposalLayout, Candidates) {
    let nested = IntersectionConstraint::new(vec![AdaptiveSource {
        preferred_selector: SELECT_RIGHT,
        layout_is_set: second_is_set,
    }]);
    let root: IntersectionConstraint<Box<dyn Constraint<'static>>> =
        IntersectionConstraint::new(vec![
            Box::new(AdaptiveSource {
                preferred_selector: SELECT_LEFT,
                layout_is_set: true,
            }),
            Box::new(nested),
        ]);
    let rows = [SELECT_LEFT, SELECT_RIGHT];
    let view = RowsView::new(&[SELECTOR], &rows);
    let mut candidates = Vec::new();
    let layout =
        root.propose_with_layout(TARGET, &view, &mut CandidateSink::Tagged(&mut candidates));
    (layout, candidates)
}

#[test]
fn nested_nonuniform_intersection_downgrades_if_any_selected_row_is_a_bag() {
    let (all_set, set_candidates) = nonuniform_nested_layout(true);
    assert!(all_set.is_grouped_set());
    assert_eq!(set_candidates, [(0, A), (1, A)]);

    let (mixed, bag_candidates) = nonuniform_nested_layout(false);
    assert!(!mixed.is_grouped_set());
    assert_eq!(bag_candidates, [(0, A), (1, A), (1, A)]);
}

#[test]
fn union_constant_and_equality_issue_construction_proven_sets() {
    let union: UnionConstraint<Box<dyn Constraint<'static>>> = UnionConstraint::new(vec![
        Box::new(BasicSource {
            values: &[B, A, A],
            layout_is_set: false,
            coverage: ProposalCoverage::Exact,
            accepted: &[A, B],
        }),
        Box::new(BasicSource {
            values: &[B, B],
            layout_is_set: false,
            coverage: ProposalCoverage::Exact,
            accepted: &[B],
        }),
    ]);
    let mut union_values = Vec::new();
    let union_layout = union.propose_with_layout(
        TARGET,
        &RowsView::EMPTY,
        &mut CandidateSink::Values(&mut union_values),
    );
    assert!(union_layout.is_grouped_set());
    assert_eq!(union_values, [A, B]);

    let variable = Variable::<UnknownInline>::new(TARGET);
    let constant = ConstantConstraint::new(variable, Inline::new(B));
    let mut constant_values = Vec::new();
    let constant_layout = constant.propose_with_layout(
        TARGET,
        &RowsView::EMPTY,
        &mut CandidateSink::Values(&mut constant_values),
    );
    assert!(constant_layout.is_grouped_set());
    assert_eq!(constant_values, [B]);

    let equality = EqualityConstraint::new(TARGET, SELECTOR);
    let mut equality_values = Vec::new();
    let equality_layout = equality.propose_with_layout(
        TARGET,
        &RowsView::new(&[SELECTOR], &[A]),
        &mut CandidateSink::Values(&mut equality_values),
    );
    assert!(equality_layout.is_grouped_set());
    assert_eq!(equality_values, [A]);
}

#[test]
fn diagnostic_wrappers_forward_the_opaque_receipt() {
    let inner = BasicSource {
        values: &[B, A],
        layout_is_set: true,
        coverage: ProposalCoverage::Exact,
        accepted: &[A, B],
    };
    let override_constraint = EstimateOverrideConstraint::new(inner);
    let record = Rc::new(RefCell::new(Vec::new()));
    let debug = DebugConstraint::new(override_constraint, Rc::clone(&record));
    let mut values = Vec::new();
    let layout = debug.propose_with_layout(
        TARGET,
        &RowsView::EMPTY,
        &mut CandidateSink::Values(&mut values),
    );

    assert_eq!(&*record.borrow(), &[TARGET]);
    assert!(layout.is_grouped_set());
    assert_eq!(values, [B, A]);
}
