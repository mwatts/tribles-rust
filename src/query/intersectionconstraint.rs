use super::*;

pub struct IntersectionConstraint<C> {
    constraints: Vec<C>,
}

impl<'a, C> IntersectionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    pub fn new(constraints: Vec<C>) -> Self {
        IntersectionConstraint { constraints }
    }
}

impl<'a, C> Constraint<'a> for IntersectionConstraint<C>
where
    C: Constraint<'a> + 'a,
{
    fn variables(&self) -> VariableSet {
        self.constraints
            .iter()
            .fold(VariableSet::new_empty(), |vs, c| vs.union(c.variables()))
    }

    fn variable(&self, variable: VariableId) -> bool {
        self.constraints.iter().any(|c| c.variable(variable))
    }

    fn estimate(&self, variable: VariableId, binding: &Binding) -> usize {
        self.constraints
            .iter()
            .filter(|c| c.variable(variable))
            .map(|c| c.estimate(variable, binding))
            .min()
            .unwrap()
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        let mut relevant_constraints: Vec<_> = self
            .constraints
            .iter()
            .filter(|c| c.variable(variable))
            .collect();
        relevant_constraints.sort_by_cached_key(|c| c.estimate(variable, binding));

        relevant_constraints[0].propose(variable, binding, proposals);

        relevant_constraints[1..]
            .iter()
            .for_each(|c| c.confirm(variable, binding, proposals));
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        let mut relevant_constraints: Vec<_> = self
            .constraints
            .iter()
            .filter(|c| c.variable(variable))
            .collect();
        relevant_constraints.sort_by_cached_key(|c| c.estimate(variable, binding));

        relevant_constraints
            .iter()
            .for_each(|c| c.confirm(variable, binding, proposals));
    }
}

#[macro_export]
macro_rules! and {
    ($($c:expr),+ $(,)?) => (
        $crate::query::intersectionconstraint::IntersectionConstraint::new(vec![
            $(Box::new($c) as Box<dyn $crate::query::Constraint>),+
        ])
    )
}

pub use and;
