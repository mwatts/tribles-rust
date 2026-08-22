//! The proposal-region high-water mark is independent of both frontier width
//! and the cumulative proposal count.

use triblespace_core::inline::RawInline;
use triblespace_core::query::intersectionconstraint::IntersectionConstraint;
use triblespace_core::query::{
    Binding, Candidates, Constraint, Frontier, ProposalBuffer, Query, VariableId, VariableSet,
};

fn value(i: u32) -> RawInline {
    let mut value = [0u8; 32];
    value[28..].copy_from_slice(&i.to_be_bytes());
    value
}

/// Proposes the same finite domain for every row of a frontier.
struct FanOut {
    variable: VariableId,
    count: u32,
}

impl<'a> Constraint<'a> for FanOut {
    fn variables(&self) -> VariableSet {
        let mut variables = VariableSet::new_empty();
        variables.set(self.variable);
        variables
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        (variable == self.variable).then_some(self.count as usize)
    }

    fn propose(
        &self,
        variable: VariableId,
        frontier: &Frontier<'_>,
        proposals: &mut ProposalBuffer,
    ) {
        if variable != self.variable {
            return;
        }

        for row in 0..frontier.len() {
            proposals.open(row as u32);
            for candidate in 0..self.count {
                proposals.push(value(candidate));
            }
        }
    }

    fn confirm(
        &self,
        variable: VariableId,
        _frontier: &Frontier<'_>,
        candidates: &mut Candidates<'_>,
    ) {
        if variable == self.variable {
            candidates.retain(|candidate| {
                let suffix: [u8; 4] = candidate[28..].try_into().expect("four-byte suffix");
                u32::from_be_bytes(suffix) < self.count
            });
        }
    }
}

#[test]
fn narrow_frontier_can_materialise_a_large_proposal_region() {
    const FAN_OUT: u32 = 50_000;
    let query = Query::new(
        FanOut {
            variable: 0,
            count: FAN_OUT,
        },
        |binding: &Binding| binding.get(0).copied(),
    );
    let stats = query.stats();

    assert_eq!(query.count(), FAN_OUT as usize);
    assert_eq!(stats.widest(), 1, "the frontier stayed one row wide");
    assert_eq!(
        stats.peak_region(),
        FAN_OUT as u64,
        "one refill materialised the whole fan-out"
    );
}

#[test]
fn peak_region_is_a_maximum_not_a_cumulative_total() {
    const FAN_OUT: u32 = 4_096;
    let constraint = IntersectionConstraint::new(vec![
        Box::new(FanOut {
            variable: 0,
            count: 1,
        }) as Box<dyn Constraint + Send + Sync>,
        Box::new(FanOut {
            variable: 1,
            count: FAN_OUT,
        }),
    ]);
    let query = Query::new(constraint, |binding: &Binding| {
        Some((*binding.get(0)?, *binding.get(1)?))
    });
    let stats = query.stats();

    assert_eq!(query.count(), FAN_OUT as usize);
    assert_eq!(stats.peak_region(), FAN_OUT as u64);
    assert_eq!(
        stats.proposals(),
        FAN_OUT as u64 + 1,
        "the cumulative counter includes both materialised levels"
    );
}
