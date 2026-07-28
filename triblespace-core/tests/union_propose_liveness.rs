//! `UnionConstraint::propose` must not resurrect entries that a nested
//! constraint already killed inside its own propose.
//!
//! The union needs set semantics, so it sort-dedups its freshly appended
//! region and republishes it with `rewrite_region`. Reading that region
//! through the `Deref` yields values without liveness, and `rewrite_region`
//! marks everything it is handed as live — so a value an inner `and!` arm
//! killed on the way past comes back from the dead. The buffer is kill-only;
//! this is the one primitive that could violate it.

use triblespace_core::inline::RawInline;
use triblespace_core::query::intersectionconstraint::IntersectionConstraint;
use triblespace_core::query::unionconstraint::UnionConstraint;
use triblespace_core::query::{
    Binding, Candidates, Constraint, ProposalBuffer, Query, VariableId, VariableSet,
};

fn v(i: u8) -> RawInline {
    let mut x = [0u8; 32];
    x[31] = i;
    x
}

/// Proposes `values`, and confirms membership in `values`.
struct Src {
    variable: VariableId,
    values: Vec<RawInline>,
    estimate: usize,
}

impl<'a> Constraint<'a> for Src {
    fn variables(&self) -> VariableSet {
        let mut s = VariableSet::new_empty();
        s.set(self.variable);
        s
    }

    fn estimate(&self, var: VariableId, _b: &Binding) -> Option<usize> {
        (var == self.variable).then_some(self.estimate)
    }

    fn propose(&self, var: VariableId, _b: &Binding, p: &mut ProposalBuffer) {
        if var == self.variable {
            p.extend_from_slice(&self.values);
        }
    }

    fn confirm(&self, var: VariableId, _b: &Binding, c: &mut Candidates<'_>) {
        if var == self.variable {
            c.retain(|x| self.values.contains(x));
        }
    }

    fn satisfied(&self, b: &Binding) -> bool {
        match b.get(self.variable) {
            Some(x) => self.values.contains(x),
            None => true,
        }
    }
}

type Dyn = Box<dyn Constraint<'static> + Send + Sync>;

/// `or!( and!(wide, narrow), and!(other) )` over one variable.
///
/// `wide` proposes {1,2,3} and `narrow` admits only {1}, so the first arm
/// contributes {1} alone — `narrow` kills 2 and 3 while the intersection is
/// still inside its own propose. The second arm contributes {5}. The union
/// is therefore {1, 5}.
fn rows() -> Vec<RawInline> {
    let wide = Src {
        variable: 0,
        values: vec![v(1), v(2), v(3)],
        estimate: 3,
    };
    let narrow = Src {
        variable: 0,
        values: vec![v(1)],
        // Deliberately the larger estimate, so the intersection picks `wide`
        // as proposer and leaves `narrow` to confirm — which is what makes
        // the kills happen inside the arm's own propose.
        estimate: 100,
    };
    let other = Src {
        variable: 0,
        values: vec![v(5)],
        estimate: 1,
    };

    let arm_a = IntersectionConstraint::new(vec![Box::new(wide) as Dyn, Box::new(narrow)]);
    let arm_b = IntersectionConstraint::new(vec![Box::new(other) as Dyn]);
    let union = UnionConstraint::new(vec![Box::new(arm_a) as Dyn, Box::new(arm_b)]);

    Query::new(union, |b: &Binding| b.get(0).copied()).collect()
}

#[test]
fn union_propose_must_not_resurrect_a_nested_arms_kills() {
    let mut got = rows();
    got.sort_unstable();
    assert_eq!(
        got,
        vec![v(1), v(5)],
        "union republished values the inner intersection had already killed"
    );
}
