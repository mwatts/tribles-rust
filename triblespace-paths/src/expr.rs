use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::{Automaton, StateId, Step, Transition};

/// A canonical regular expression over graph-property [`Step`]s.
///
/// Construction normalizes exclusion lists, flattens associative sequences,
/// and flattens, sorts, and deduplicates alternatives. This is structural
/// canonicalization, not full language minimization: distributively equivalent
/// expressions may still compile to different (but language-equivalent)
/// automata. The durable alternative order is atom, sequence, alternative,
/// star, plus, optional; atom kinds are forward, reverse, forward-except,
/// reverse-except, with payloads ordered lexicographically.
///
/// ```
/// use triblespace_paths::{PathExpr, Step};
///
/// let parent = [1; 16];
/// let sibling = [2; 16];
/// let expression = PathExpr::from(Step::Forward(parent))
///     .plus()
///     .or(PathExpr::from(Step::Forward(sibling)).inverse());
/// let automaton = expression.compile();
///
/// assert_eq!(automaton.initial_states().collect::<Vec<_>>(), vec![0]);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathExpr {
    node: Node,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Node {
    Atom(Step),
    Sequence(Vec<PathExpr>),
    Alternative(Vec<PathExpr>),
    Star(Box<PathExpr>),
    Plus(Box<PathExpr>),
    Optional(Box<PathExpr>),
}

// This order is part of PathExpr's canonical form. Keep the explicit tags
// stable: changing source enum order must not perturb alternative order or the
// automaton fingerprint produced by `compile`.
const TAG_ATOM: u8 = 0;
const TAG_SEQUENCE: u8 = 1;
const TAG_ALTERNATIVE: u8 = 2;
const TAG_STAR: u8 = 3;
const TAG_PLUS: u8 = 4;
const TAG_OPTIONAL: u8 = 5;

fn node_tag(node: &Node) -> u8 {
    match node {
        Node::Atom(_) => TAG_ATOM,
        Node::Sequence(_) => TAG_SEQUENCE,
        Node::Alternative(_) => TAG_ALTERNATIVE,
        Node::Star(_) => TAG_STAR,
        Node::Plus(_) => TAG_PLUS,
        Node::Optional(_) => TAG_OPTIONAL,
    }
}

fn step_tag(step: &Step) -> u8 {
    match step {
        Step::Forward(_) => 0,
        Step::Reverse(_) => 1,
        Step::ForwardExcept(_) => 2,
        Step::ReverseExcept(_) => 3,
    }
}

fn cmp_steps(left: &Step, right: &Step) -> Ordering {
    step_tag(left)
        .cmp(&step_tag(right))
        .then_with(|| match (left, right) {
            (Step::Forward(left), Step::Forward(right))
            | (Step::Reverse(left), Step::Reverse(right)) => left.cmp(right),
            (Step::ForwardExcept(left), Step::ForwardExcept(right))
            | (Step::ReverseExcept(left), Step::ReverseExcept(right)) => left.cmp(right),
            _ => Ordering::Equal,
        })
}

impl Ord for PathExpr {
    fn cmp(&self, other: &Self) -> Ordering {
        node_tag(&self.node)
            .cmp(&node_tag(&other.node))
            .then_with(|| match (&self.node, &other.node) {
                (Node::Atom(left), Node::Atom(right)) => cmp_steps(left, right),
                (Node::Sequence(left), Node::Sequence(right))
                | (Node::Alternative(left), Node::Alternative(right)) => left.cmp(right),
                (Node::Star(left), Node::Star(right))
                | (Node::Plus(left), Node::Plus(right))
                | (Node::Optional(left), Node::Optional(right)) => left.cmp(right),
                _ => Ordering::Equal,
            })
    }
}

impl PartialOrd for PathExpr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<Step> for PathExpr {
    fn from(step: Step) -> Self {
        Self {
            node: Node::Atom(canonical_step(step)),
        }
    }
}

impl PathExpr {
    /// Concatenates this expression with `next`.
    ///
    /// Nested sequences are flattened without reordering their operands.
    #[must_use]
    pub fn then(self, next: impl Into<Self>) -> Self {
        Self::sequence([self, next.into()])
    }

    /// Forms the set union of this expression and `alternative`.
    ///
    /// Nested alternatives are flattened, sorted by an explicit stable node
    /// order, and deduplicated.
    #[must_use]
    pub fn or(self, alternative: impl Into<Self>) -> Self {
        Self::alternative([self, alternative.into()])
    }

    /// Matches zero or more repetitions of this expression.
    #[must_use]
    pub fn star(self) -> Self {
        Self {
            node: Node::Star(Box::new(self)),
        }
    }

    /// Matches one or more repetitions of this expression.
    #[must_use]
    pub fn plus(self) -> Self {
        Self {
            node: Node::Plus(Box::new(self)),
        }
    }

    /// Matches this expression zero or one time.
    #[must_use]
    pub fn optional(self) -> Self {
        Self {
            node: Node::Optional(Box::new(self)),
        }
    }

    /// Reverses path direction.
    ///
    /// Inversion flips every atomic step, reverses sequence order, distributes
    /// through alternatives, and preserves repetition operators.
    #[must_use]
    pub fn inverse(self) -> Self {
        match self.node {
            Node::Atom(step) => canonical_step(inverse_step(step)).into(),
            Node::Sequence(parts) => Self::sequence(parts.into_iter().rev().map(Self::inverse)),
            Node::Alternative(arms) => Self::alternative(arms.into_iter().map(Self::inverse)),
            Node::Star(body) => body.inverse().star(),
            Node::Plus(body) => body.inverse().plus(),
            Node::Optional(body) => body.inverse().optional(),
        }
    }

    /// Whether this expression accepts the empty path.
    pub fn is_nullable(&self) -> bool {
        match &self.node {
            Node::Atom(_) => false,
            Node::Sequence(parts) => parts.iter().all(Self::is_nullable),
            Node::Alternative(arms) => arms.iter().any(Self::is_nullable),
            Node::Star(_) | Node::Optional(_) => true,
            Node::Plus(body) => body.is_nullable(),
        }
    }

    /// Compiles to a fixed epsilon-free position automaton.
    ///
    /// Atomic occurrences are numbered deterministically in canonical-tree
    /// order. State zero is the sole initial state; every other state denotes
    /// the most recently consumed atomic occurrence. The construction is
    /// output-sensitive in the expression and generated `follow` relation;
    /// ordered-set maintenance may add logarithmic factors. It introduces no
    /// epsilon transitions or determinization step.
    ///
    /// # Panics
    ///
    /// Panics if the expression contains `u32::MAX` atomic occurrences. Such
    /// an expression cannot be represented by [`StateId`].
    pub fn compile(&self) -> Automaton {
        let mut compiler = PositionCompiler::default();
        let fragment = compiler.analyze(self);
        let position_count = compiler.steps.len();
        let state_count = position_count
            .checked_add(1)
            .and_then(|count| StateId::try_from(count).ok())
            .expect("path expression exceeds the u32 automaton state space");

        let mut transitions = Vec::new();
        for position in &fragment.first {
            transitions.push(Transition::new(
                0,
                *position,
                compiler.step(*position).clone(),
            ));
        }
        for (from, targets) in compiler.follow.iter().enumerate() {
            let from = StateId::try_from(from + 1).expect("position checked above");
            for to in targets {
                transitions.push(Transition::new(from, *to, compiler.step(*to).clone()));
            }
        }

        let mut accepting = fragment.last;
        if fragment.nullable {
            accepting.insert(0);
        }
        Automaton::new(state_count, [0], accepting, transitions)
            .expect("the position construction emits a valid fixed automaton")
    }

    fn sequence(parts: impl IntoIterator<Item = Self>) -> Self {
        let mut flattened = Vec::new();
        for part in parts {
            match part.node {
                Node::Sequence(nested) => flattened.extend(nested),
                node => flattened.push(Self { node }),
            }
        }
        debug_assert!(!flattened.is_empty());
        if flattened.len() == 1 {
            flattened.pop().expect("checked length")
        } else {
            Self {
                node: Node::Sequence(flattened),
            }
        }
    }

    fn alternative(arms: impl IntoIterator<Item = Self>) -> Self {
        let mut flattened = Vec::new();
        for arm in arms {
            match arm.node {
                Node::Alternative(nested) => flattened.extend(nested),
                node => flattened.push(Self { node }),
            }
        }
        flattened.sort_unstable();
        flattened.dedup();
        debug_assert!(!flattened.is_empty());
        if flattened.len() == 1 {
            flattened.pop().expect("checked length")
        } else {
            Self {
                node: Node::Alternative(flattened),
            }
        }
    }
}

fn canonical_step(step: Step) -> Step {
    match step {
        Step::ForwardExcept(mut excluded) => {
            excluded.sort_unstable();
            excluded.dedup();
            Step::ForwardExcept(excluded)
        }
        Step::ReverseExcept(mut excluded) => {
            excluded.sort_unstable();
            excluded.dedup();
            Step::ReverseExcept(excluded)
        }
        step @ (Step::Forward(_) | Step::Reverse(_)) => step,
    }
}

fn inverse_step(step: Step) -> Step {
    match step {
        Step::Forward(attribute) => Step::Reverse(attribute),
        Step::Reverse(attribute) => Step::Forward(attribute),
        Step::ForwardExcept(excluded) => Step::ReverseExcept(excluded),
        Step::ReverseExcept(excluded) => Step::ForwardExcept(excluded),
    }
}

#[derive(Default)]
struct PositionCompiler {
    steps: Vec<Step>,
    follow: Vec<BTreeSet<StateId>>,
}

struct Fragment {
    nullable: bool,
    first: BTreeSet<StateId>,
    last: BTreeSet<StateId>,
}

impl PositionCompiler {
    fn analyze(&mut self, expression: &PathExpr) -> Fragment {
        match &expression.node {
            Node::Atom(step) => {
                let position = StateId::try_from(self.steps.len() + 1)
                    .expect("path expression exceeds the u32 automaton state space");
                self.steps.push(step.clone());
                self.follow.push(BTreeSet::new());
                Fragment {
                    nullable: false,
                    first: BTreeSet::from([position]),
                    last: BTreeSet::from([position]),
                }
            }
            Node::Sequence(parts) => {
                let mut parts = parts.iter();
                let mut result =
                    self.analyze(parts.next().expect("canonical sequence is nonempty"));
                for part in parts {
                    let right = self.analyze(part);
                    self.link(&result.last, &right.first);

                    let mut first = result.first;
                    if result.nullable {
                        first.extend(right.first.iter().copied());
                    }
                    let mut last = right.last;
                    if right.nullable {
                        last.extend(result.last.iter().copied());
                    }
                    result = Fragment {
                        nullable: result.nullable && right.nullable,
                        first,
                        last,
                    };
                }
                result
            }
            Node::Alternative(arms) => {
                let mut nullable = false;
                let mut first = BTreeSet::new();
                let mut last = BTreeSet::new();
                for arm in arms {
                    let fragment = self.analyze(arm);
                    nullable |= fragment.nullable;
                    first.extend(fragment.first);
                    last.extend(fragment.last);
                }
                Fragment {
                    nullable,
                    first,
                    last,
                }
            }
            Node::Star(body) => {
                let fragment = self.analyze(body);
                self.link(&fragment.last, &fragment.first);
                Fragment {
                    nullable: true,
                    ..fragment
                }
            }
            Node::Plus(body) => {
                let fragment = self.analyze(body);
                self.link(&fragment.last, &fragment.first);
                fragment
            }
            Node::Optional(body) => {
                let fragment = self.analyze(body);
                Fragment {
                    nullable: true,
                    ..fragment
                }
            }
        }
    }

    fn link(&mut self, from: &BTreeSet<StateId>, to: &BTreeSet<StateId>) {
        for &source in from {
            self.follow[source as usize - 1].extend(to.iter().copied());
        }
    }

    fn step(&self, position: StateId) -> &Step {
        &self.steps[position as usize - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Token = (bool, [u8; 16]);

    fn label(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn atom(byte: u8) -> PathExpr {
        Step::Forward(label(byte)).into()
    }

    fn step_matches(step: &Step, token: Token) -> bool {
        step.is_reverse() == token.0 && step.matches(&token.1)
    }

    fn direct_ends(expression: &PathExpr, word: &[Token], start: usize) -> BTreeSet<usize> {
        match &expression.node {
            Node::Atom(step) => {
                if word
                    .get(start)
                    .is_some_and(|&token| step_matches(step, token))
                {
                    BTreeSet::from([start + 1])
                } else {
                    BTreeSet::new()
                }
            }
            Node::Sequence(parts) => {
                let mut positions = BTreeSet::from([start]);
                for part in parts {
                    positions = positions
                        .into_iter()
                        .flat_map(|position| direct_ends(part, word, position))
                        .collect();
                }
                positions
            }
            Node::Alternative(arms) => arms
                .iter()
                .flat_map(|arm| direct_ends(arm, word, start))
                .collect(),
            Node::Star(body) => repeat_ends(body, word, start, true),
            Node::Plus(body) => repeat_ends(body, word, start, false),
            Node::Optional(body) => {
                let mut ends = direct_ends(body, word, start);
                ends.insert(start);
                ends
            }
        }
    }

    fn repeat_ends(
        body: &PathExpr,
        word: &[Token],
        start: usize,
        include_empty: bool,
    ) -> BTreeSet<usize> {
        let mut result = if include_empty {
            BTreeSet::from([start])
        } else {
            BTreeSet::new()
        };
        let mut frontier = direct_ends(body, word, start);
        while let Some(position) = frontier.pop_first() {
            if result.insert(position) {
                frontier.extend(direct_ends(body, word, position));
            }
        }
        result
    }

    fn automaton_accepts(automaton: &Automaton, word: &[Token]) -> bool {
        let mut active = automaton.initial_states().collect::<BTreeSet<_>>();
        for &token in word {
            active = automaton
                .transitions()
                .iter()
                .filter(|transition| {
                    active.contains(&transition.from) && step_matches(&transition.step, token)
                })
                .map(|transition| transition.to)
                .collect();
        }
        active
            .into_iter()
            .any(|state| automaton.is_accepting(state))
    }

    fn words(alphabet: &[Token], max_len: usize) -> Vec<Vec<Token>> {
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            frontier = frontier
                .iter()
                .flat_map(|prefix| {
                    alphabet.iter().map(move |&token| {
                        let mut word = prefix.clone();
                        word.push(token);
                        word
                    })
                })
                .collect();
            words.extend(frontier.iter().cloned());
        }
        words
    }

    #[test]
    fn exhaustive_small_expressions_match_the_direct_word_oracle() {
        let base = vec![
            atom(1),
            atom(2),
            PathExpr::from(Step::Reverse(label(1))),
            PathExpr::from(Step::ForwardExcept(vec![label(2)])),
            PathExpr::from(Step::ReverseExcept(vec![label(1)])),
            PathExpr::from(Step::forward_any()),
            PathExpr::from(Step::reverse_any()),
        ];
        let mut expressions = base.iter().cloned().collect::<BTreeSet<_>>();
        for body in &base {
            expressions.extend([
                body.clone().star(),
                body.clone().plus(),
                body.clone().optional(),
                body.clone().inverse(),
            ]);
        }
        for left in &base {
            for right in &base {
                expressions.insert(left.clone().then(right.clone()));
                expressions.insert(left.clone().or(right.clone()));
            }
        }
        let first_layer = expressions.iter().cloned().collect::<Vec<_>>();
        for expression in &first_layer {
            expressions.extend([
                expression.clone().star(),
                expression.clone().plus(),
                expression.clone().optional(),
                expression.clone().inverse(),
            ]);
            for right in &base {
                expressions.insert(expression.clone().then(right.clone()));
                expressions.insert(right.clone().then(expression.clone()));
                expressions.insert(expression.clone().or(right.clone()));
            }
        }

        let alphabet = [
            (false, label(1)),
            (false, label(2)),
            (true, label(1)),
            (true, label(2)),
        ];
        let words = words(&alphabet, 3);
        for (expression_index, expression) in expressions.iter().enumerate() {
            let automaton = expression.compile();
            assert_eq!(expression.is_nullable(), automaton.is_accepting(0));
            for word in &words {
                assert_eq!(
                    automaton_accepts(&automaton, word),
                    direct_ends(expression, word, 0).contains(&word.len()),
                    "expression {expression_index}: {expression:?}, word {word:?}"
                );
            }
        }
    }

    #[test]
    fn repeated_sequence_completes_a_second_cycle() {
        let expression = atom(1).then(atom(2)).star();
        let word = [
            (false, label(1)),
            (false, label(2)),
            (false, label(1)),
            (false, label(2)),
        ];

        assert!(direct_ends(&expression, &word, 0).contains(&word.len()));
        assert!(automaton_accepts(&expression.compile(), &word));
    }

    #[test]
    fn canonical_builders_stabilize_structure_and_automata() {
        let a = atom(1);
        let b = atom(2);
        let c = atom(3);

        assert_eq!(
            a.clone().then(b.clone()).then(c.clone()),
            a.clone().then(b.clone().then(c.clone()))
        );
        assert_eq!(
            a.clone().or(b.clone()).or(a.clone()).or(c.clone()),
            c.clone().or(b.clone().or(a.clone()))
        );
        assert_eq!(
            PathExpr::from(Step::ForwardExcept(vec![label(2), label(1), label(1)])),
            PathExpr::from(Step::ForwardExcept(vec![label(1), label(2)]))
        );

        let left = a.clone().or(b.clone()).or(a.clone()).compile();
        let right = b.or(a).compile();
        assert_eq!(left, right);
        assert_eq!(
            crate::automaton_fingerprint(&left),
            crate::automaton_fingerprint(&right)
        );
    }

    #[test]
    fn inverse_is_an_involution_and_reverses_sequences() {
        let expression = atom(1)
            .then(atom(2).optional())
            .or(PathExpr::from(Step::ForwardExcept(vec![label(3)])).plus());
        assert_eq!(expression.clone().inverse().inverse(), expression);

        let forward = atom(1).then(atom(2));
        let expected =
            PathExpr::from(Step::Reverse(label(2))).then(PathExpr::from(Step::Reverse(label(1))));
        assert_eq!(forward.inverse(), expected);
    }
}
