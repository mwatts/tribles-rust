use std::collections::HashSet;
use std::collections::VecDeque;

use crate::id::id_from_value;
use crate::id::id_into_value;
use crate::id::RawId;
use crate::id::ID_LEN;
use crate::patch::PATCH;
use crate::query::Binding;
use crate::query::Constraint;
use crate::query::Variable;
use crate::query::VariableId;
use crate::query::VariableSet;
use crate::trible::EAVOrder;
use crate::trible::TribleSet;
use crate::trible::TRIBLE_LEN;
use crate::value::schemas::genid::GenId;
use crate::value::RawValue;

// ── Path expression types ────────────────────────────────────────────────

/// Postfix-encoded path operations (used by the `path!` macro).
#[derive(Clone)]
pub enum PathOp {
    Attr(RawId),
    Concat,
    Union,
    Star,
    Plus,
}

/// Tree-structured path expression for recursive evaluation.
#[derive(Clone)]
enum PathExpr {
    Attr(RawId),
    Concat(Box<PathExpr>, Box<PathExpr>),
    Union(Box<PathExpr>, Box<PathExpr>),
    Star(Box<PathExpr>),
    Plus(Box<PathExpr>),
}

impl PathExpr {
    fn from_postfix(ops: &[PathOp]) -> Self {
        let mut stack: Vec<PathExpr> = Vec::new();
        for op in ops {
            match op {
                PathOp::Attr(id) => stack.push(PathExpr::Attr(*id)),
                PathOp::Concat => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(PathExpr::Concat(Box::new(a), Box::new(b)));
                }
                PathOp::Union => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(PathExpr::Union(Box::new(a), Box::new(b)));
                }
                PathOp::Star => {
                    let a = stack.pop().unwrap();
                    stack.push(PathExpr::Star(Box::new(a)));
                }
                PathOp::Plus => {
                    let a = stack.pop().unwrap();
                    stack.push(PathExpr::Plus(Box::new(a)));
                }
            }
        }
        stack.pop().unwrap()
    }

    /// Collect first-hop attribute IDs (for shallow estimation).
    fn root_attrs(&self) -> Vec<RawId> {
        match self {
            PathExpr::Attr(a) => vec![*a],
            PathExpr::Concat(lhs, _) => lhs.root_attrs(),
            PathExpr::Union(lhs, rhs) => {
                let mut attrs = lhs.root_attrs();
                for a in rhs.root_attrs() {
                    if !attrs.contains(&a) {
                        attrs.push(a);
                    }
                }
                attrs
            }
            PathExpr::Star(inner) | PathExpr::Plus(inner) => inner.root_attrs(),
        }
    }
}

// ── Recursive path evaluator ─────────────────────────────────────────────

fn eval_from(set: &TribleSet, expr: &PathExpr, start: &RawId) -> HashSet<RawId> {
    match expr {
        PathExpr::Attr(attr) => {
            let mut results = HashSet::new();
            let mut prefix = [0u8; ID_LEN * 2];
            prefix[..ID_LEN].copy_from_slice(start);
            prefix[ID_LEN..].copy_from_slice(attr);
            set.eav
                .infixes::<{ ID_LEN * 2 }, 32, _>(&prefix, |value: &[u8; 32]| {
                    if value[..ID_LEN] == [0; ID_LEN] {
                        let dest: RawId = value[ID_LEN..].try_into().unwrap();
                        results.insert(dest);
                    }
                });
            results
        }
        PathExpr::Concat(lhs, rhs) => {
            let intermediates = eval_from(set, lhs, start);
            let mut results = HashSet::new();
            for mid in &intermediates {
                results.extend(eval_from(set, rhs, mid));
            }
            results
        }
        PathExpr::Union(lhs, rhs) => {
            let mut results = eval_from(set, lhs, start);
            results.extend(eval_from(set, rhs, start));
            results
        }
        PathExpr::Plus(body) => {
            let mut visited: HashSet<RawId> = HashSet::new();
            let mut results: HashSet<RawId> = HashSet::new();
            let mut frontier: VecDeque<RawId> = VecDeque::new();
            frontier.push_back(*start);
            visited.insert(*start);

            while let Some(node) = frontier.pop_front() {
                let reached = eval_from(set, body, &node);
                for dest in reached {
                    results.insert(dest);
                    if visited.insert(dest) {
                        frontier.push_back(dest);
                    }
                }
            }
            results
        }
        PathExpr::Star(body) => {
            let mut results = eval_from(set, &PathExpr::Plus(body.clone()), start);
            results.insert(*start);
            results
        }
    }
}

fn has_path(set: &TribleSet, expr: &PathExpr, from: &RawId, to: &RawId) -> bool {
    match expr {
        PathExpr::Attr(attr) => {
            let mut found = false;
            let mut prefix = [0u8; ID_LEN * 2];
            prefix[..ID_LEN].copy_from_slice(from);
            prefix[ID_LEN..].copy_from_slice(attr);
            set.eav
                .infixes::<{ ID_LEN * 2 }, 32, _>(&prefix, |value: &[u8; 32]| {
                    if !found && value[..ID_LEN] == [0; ID_LEN] {
                        let dest: RawId = value[ID_LEN..].try_into().unwrap();
                        if dest == *to {
                            found = true;
                        }
                    }
                });
            found
        }
        PathExpr::Concat(lhs, rhs) => {
            let intermediates = eval_from(set, lhs, from);
            intermediates
                .iter()
                .any(|mid| has_path(set, rhs, mid, to))
        }
        PathExpr::Union(lhs, rhs) => {
            has_path(set, lhs, from, to) || has_path(set, rhs, from, to)
        }
        PathExpr::Plus(body) => {
            let mut visited: HashSet<RawId> = HashSet::new();
            let mut frontier: VecDeque<RawId> = VecDeque::new();
            frontier.push_back(*from);
            visited.insert(*from);

            while let Some(node) = frontier.pop_front() {
                let reached = eval_from(set, body, &node);
                for dest in reached {
                    if dest == *to {
                        return true;
                    }
                    if visited.insert(dest) {
                        frontier.push_back(dest);
                    }
                }
            }
            false
        }
        PathExpr::Star(body) => {
            if from == to {
                return true;
            }
            has_path(set, &PathExpr::Plus(body.clone()), from, to)
        }
    }
}

fn count_matching_edges(
    eav: &PATCH<TRIBLE_LEN, EAVOrder, ()>,
    entity: &RawId,
    labels: &[RawId],
) -> usize {
    let mut count = 0;
    for label in labels {
        let mut prefix = [0u8; ID_LEN * 2];
        prefix[..ID_LEN].copy_from_slice(entity);
        prefix[ID_LEN..].copy_from_slice(label);
        eav.infixes::<{ ID_LEN * 2 }, 32, _>(&prefix, |value: &[u8; 32]| {
            if value[..ID_LEN] == [0; ID_LEN] {
                count += 1;
            }
        });
    }
    count
}

// ── Constraint ───────────────────────────────────────────────────────────

pub struct RegularPathConstraint {
    start: VariableId,
    end: VariableId,
    expr: PathExpr,
    set: TribleSet,
    nodes: Vec<RawValue>,
}

impl RegularPathConstraint {
    pub fn new(
        set: TribleSet,
        start: Variable<GenId>,
        end: Variable<GenId>,
        ops: &[PathOp],
    ) -> Self {
        let expr = PathExpr::from_postfix(ops);
        let mut node_set: HashSet<RawValue> = HashSet::new();
        for t in set.iter() {
            let v = &t.data[32..64];
            if v[..ID_LEN] == [0; ID_LEN] {
                let dest: RawId = v[ID_LEN..].try_into().unwrap();
                node_set.insert(id_into_value(&dest));
                let e: RawId = t.data[..ID_LEN].try_into().unwrap();
                node_set.insert(id_into_value(&e));
            }
        }
        RegularPathConstraint {
            start: start.index,
            end: end.index,
            expr,
            set,
            nodes: node_set.into_iter().collect(),
        }
    }
}

impl<'a> Constraint<'a> for RegularPathConstraint {
    fn variables(&self) -> VariableSet {
        let mut vars = VariableSet::new_empty();
        vars.set(self.start);
        vars.set(self.end);
        vars
    }

    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        if variable == self.end {
            if let Some(start_val) = binding.get(self.start) {
                if let Some(start_id) = id_from_value(start_val) {
                    let labels = self.expr.root_attrs();
                    return Some(count_matching_edges(&self.set.eav, &start_id, &labels).max(1));
                }
                return Some(0);
            }
            Some(self.nodes.len())
        } else if variable == self.start {
            Some(self.nodes.len())
        } else {
            None
        }
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.end {
            if let Some(start_val) = binding.get(self.start) {
                if let Some(start_id) = id_from_value(start_val) {
                    let reachable = eval_from(&self.set, &self.expr, &start_id);
                    proposals.extend(reachable.iter().map(id_into_value));
                }
                return;
            }
        }
        if variable == self.start || variable == self.end {
            proposals.extend(self.nodes.iter().cloned());
        }
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.start {
            if let Some(end_val) = binding.get(self.end) {
                if let Some(end_id) = id_from_value(end_val) {
                    proposals.retain(|v| {
                        if let Some(start_id) = id_from_value(v) {
                            has_path(&self.set, &self.expr, &start_id, &end_id)
                        } else {
                            false
                        }
                    });
                } else {
                    proposals.clear();
                }
            }
        } else if variable == self.end {
            if let Some(start_val) = binding.get(self.start) {
                if let Some(start_id) = id_from_value(start_val) {
                    proposals.retain(|v| {
                        if let Some(end_id) = id_from_value(v) {
                            has_path(&self.set, &self.expr, &start_id, &end_id)
                        } else {
                            false
                        }
                    });
                } else {
                    proposals.clear();
                }
            }
        }
    }
}
