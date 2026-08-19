//! Registers — a set of states, an order over them, and the reader's choice.
//!
//! A great many designs model *the same thing, changing over time* as a set
//! of immutable states. What differs between them is never the states; it is
//! **how a reader decides which of them is current**. This module makes that
//! decision a parameter.
//!
//! # A register is an order, and a policy is a choice of order
//!
//! The obvious design has two knobs — an *order* over states and a *policy*
//! for choosing among the maximal ones (multi-value, last-write-wins,
//! first-write-wins, ...). Writing them out, the second knob collapses into
//! the first:
//!
//! | policy | is |
//! |---|---|
//! | multi-value (MV) | the maximal set under a **partial** order |
//! | last-write-wins (LWW) | the maximal set under a **total** order |
//! | first-write-wins (FWW) | the maximal set under that order **reversed** |
//! | named by the reader | the maximal set under the **empty** order |
//!
//! There is exactly one operation — *take the maximal elements* — and the
//! policies are orders to take it under. LWW is not a different computation
//! from MV; it is MV over an order fine enough to leave a singleton, which
//! is why [`sole`] is a *check on the result* rather than a tie-break rule
//! that silently invents one. Manufacturing an order between genuinely
//! concurrent states is the one thing this substrate refuses to do on the
//! reader's behalf.
//!
//! # The third axis: who is allowed to dominate
//!
//! Order and end are the two obvious knobs. The third is the one that
//! decides whether real code can use this at all: **which states count as
//! dominators**.
//!
//! [`latest`](crate::query::frontier::latest) admits the whole frame — if
//! anything at all observes a state, that state has been moved past. It is
//! the widest reading, and for a supersedes DAG it is right. It is also, in
//! practice, the reading almost nobody hand-rolls. Head resolution in the
//! wild narrows the *observer* set far more often than the candidate set: a
//! snapshot track admits only same-track snapshots, an event log only events
//! of the same kind. Narrowing the candidates cannot express that, because
//! domination is asked of the frame, not of the candidate set — a state
//! outside the candidates can still dominate one inside it, which is exactly
//! the property that makes the unscoped form correct in the first place.
//!
//! So [`within`](ObservationOrder::within) admits only states sharing the
//! candidate's group, and [`among`](StatedOrder::among) only states
//! asserting a given fact. A stated key needs this more urgently than an
//! edge does: a grouping attribute is rarely as selective as it looks, and
//! comparing a status event against a note that happens to hang off the same
//! subject silently answers the wrong question.
//!
//! # There is no global "current"
//!
//! Only the current state **for a given set of commits**. That set is
//! whatever [`TriblePattern`] source the order was built over. Two readers
//! holding different sets legitimately disagree, and both are right; that is
//! frame-relativity, not a consistency bug. `latest`/`resolve` are therefore
//! always asked *in a frame*, never of the data as such.
//!
//! # Monotonicity
//!
//! Resolution is a join homomorphism between two lattices. The domain is the
//! commit-set lattice ordered by inclusion, joined by union. The codomain is
//! the **antichain lattice** ordered by domination, joined by taking the
//! maximal elements of the union:
//!
//! ```text
//! resolve(C1 union C2) = resolve(C1) join resolve(C2)
//! ```
//!
//! Head resolution looks non-monotone only when it is evaluated in the
//! *inclusion* lattice, where adding a successor shrinks the answer. In the
//! domination lattice a taller element absorbing a shorter one **is** the
//! join, and the map is monotone. This holds for every order here, including
//! the reversed ones: reversing exchanges the lattice for its dual, and a
//! join homomorphism into a dual lattice is still a join homomorphism — the
//! meet of the original order. `min` is as lawful a join as `max`; it is
//! simply the join of the opposite order.
//!
//! # Two exposures
//!
//! * [`resolve`] materialises the answer as a set. Sorted into a slice it
//!   becomes a proposing constraint with an **exact** cardinality via
//!   [`SortedSlice`](crate::query::sortedsliceconstraint::SortedSlice), so
//!   the join planner can order around it like any other relation.
//! * [`maximal`] is a streaming filter-only [`Constraint`] in the shape of
//!   [`InlineRange`](crate::query::rangeconstraint::InlineRange): it never
//!   proposes, estimates `usize::MAX` so the planner always sorts it last,
//!   and kills dominated candidates that a `pattern!` proposed. This is the
//!   form that removes the caller's obligation to materialise candidates.
//!
//! # Example
//!
//! ```
//! use triblespace_core::macros::entity;
//! use triblespace_core::metadata;
//! use triblespace_core::prelude::*;
//! use triblespace_core::query::register::{resolve, ObservationOrder};
//!
//! let first = ufoid();
//! let second = ufoid();
//! let mut facts = TribleSet::new();
//! facts += entity! { &second @ metadata::supersedes: &first };
//!
//! let order = ObservationOrder::new(&facts, metadata::supersedes.id());
//! assert_eq!(resolve(&order, [*first, *second]), [*second].into_iter().collect());
//! ```

use std::collections::BTreeSet;
use std::marker::PhantomData;

use crate::id::Id;
use crate::inline::encodings::genid::GenId;
use crate::inline::{Inline, InlineEncoding, IntoInline, TryFromInline};
use crate::query::intersectionconstraint::and;
use crate::query::rangeconstraint::value_range;
use crate::query::{
    exists, find, temp, Binding, Candidates, Constraint, Frontier, ProposalBuffer, TriblePattern,
    Variable, VariableId, VariableSet,
};

/// Which end of the order the reader is asking for.
///
/// The substrate always computes *maximal* elements; `First` obtains the
/// minimal ones by resolving in the opposite order. LWW is `Last`, FWW is
/// `First`, and both are the same operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum End {
    /// The greatest states — nothing in the frame is above them.
    Last,
    /// The least states — nothing in the frame is below them.
    First,
}

impl End {
    /// The opposite end.
    pub fn flip(self) -> Self {
        match self {
            Self::Last => Self::First,
            Self::First => Self::Last,
        }
    }
}

/// An order over the states of a register.
///
/// The whole substrate rests on one predicate. Both concrete orders below
/// answer it with a single short-circuited index probe, which is what keeps
/// resolution linear in the candidate count rather than quadratic or
/// transitive.
pub trait RegisterOrder {
    /// Whether some **other** state in this order's frame dominates `state`.
    ///
    /// A state no other state dominates is maximal, and therefore current.
    /// A state the order cannot compare at all (no key, no edge) is
    /// dominated by nothing and survives — an incomparable state is a
    /// genuine concurrent value, not an error.
    fn dominated(&self, state: Id) -> bool;
}

impl<O: RegisterOrder + ?Sized> RegisterOrder for &O {
    fn dominated(&self, state: Id) -> bool {
        (**self).dominated(state)
    }
}

/// The maximal states of `candidates` under `order`.
///
/// Candidates are the reader's business: they say which states are even in
/// scope. States *outside* that scope may still dominate ones inside it,
/// because domination is asked of the whole frame.
pub fn resolve<O>(order: &O, candidates: impl IntoIterator<Item = Id>) -> BTreeSet<Id>
where
    O: RegisterOrder + ?Sized,
{
    candidates
        .into_iter()
        .filter(|candidate| !order.dominated(*candidate))
        .collect()
}

/// The single maximal state, or `None` when the order left a fork.
///
/// This is the last-write-wins read, and its `None` is load-bearing: under a
/// total order (a timestamp with an id tie-break, say) a fork is impossible,
/// so `None` means the order was not as total as the reader believed. The
/// alternative — picking one — would invent an order the data does not have
/// and hide exactly the divergence a register exists to expose.
pub fn sole<O>(order: &O, candidates: impl IntoIterator<Item = Id>) -> Option<Id>
where
    O: RegisterOrder + ?Sized,
{
    let mut resolved = resolve(order, candidates).into_iter();
    match (resolved.next(), resolved.next()) {
        (Some(only), None) => Some(only),
        _ => None,
    }
}

/// The empty order: nothing dominates anything, so every state is current.
///
/// This is the taxonomy's *named by the reader* rule, and it is the cheapest
/// and most principled of the four — the reader declares the frame it wants
/// instead of asking the data to have a "now" it cannot have.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Unordered;

impl RegisterOrder for Unordered {
    fn dominated(&self, _state: Id) -> bool {
        false
    }
}

/// Order by an observation DAG — each state names the states it observed.
///
/// The book's "Direction and consistency" rule says the arrow runs from the
/// entity making the claim to the entity being described, and that the
/// observer owns the identifier it writes under. "I observed that" is
/// therefore a claim a writer is entitled to make about its own new state;
/// "I replace that" would be a claim about somebody else's entity. So the
/// DAG is stored successor-to-predecessor and the question is asked against
/// the *reverse* index.
///
/// This is a genuinely **partial** order, which is the point: two writers
/// who both observed the same predecessor are visibly concurrent, and
/// [`resolve`] reports both. A purely temporal coordinate would destroy that
/// property by handing them distinct timestamps.
///
/// # The predicate is local
///
/// ```text
/// s is maximal in C  <=>  no state in C observes s
/// ```
///
/// Note "no state in `C`", not "no state in the frontier". If anything in
/// `C` observes `s` then that thing — or something above it — dominates `s`
/// regardless. **Immediate edges suffice; no transitive closure is needed**,
/// which is why this is one reverse-index probe per candidate and not a path
/// query.
#[derive(Clone, Copy, Debug)]
pub struct ObservationOrder<'a, P> {
    facts: &'a P,
    observes: Inline<GenId>,
    end: End,
    within: Option<Inline<GenId>>,
    observer_states: Option<(Inline<GenId>, Inline<GenId>)>,
}

impl<'a, P> ObservationOrder<'a, P>
where
    P: TriblePattern,
{
    /// Order the states of `facts` by the DAG named by `observes`.
    ///
    /// `observes` is a parameter rather than a constant because the edge is
    /// the same edge whichever verb names it —
    /// [`metadata::supersedes`](crate::metadata::supersedes) is the
    /// published one, but a design is free to bring its own.
    pub fn new(facts: &'a P, observes: Id) -> Self {
        Self {
            facts,
            observes: observes.to_inline(),
            end: End::Last,
            within: None,
            observer_states: None,
        }
    }

    /// Only states sharing this candidate's value for `group` may dominate
    /// it.
    ///
    /// The default observer set is the **whole frame**: if anything at all
    /// observes a state, that state has been moved past, whatever kind of
    /// thing did the observing. That is the widest reading, and it is the
    /// one `latest` has always taken.
    ///
    /// It is not the only defensible one, and in practice it is not even
    /// the common one. Hand-rolled head resolution in the wild narrows the
    /// observer set far more often than it narrows the candidate set: a
    /// snapshot track admits only same-track snapshots as observers, so a
    /// stray edge from outside the track cannot silently retire a head.
    /// Scoping candidates cannot express that — an observer need not be a
    /// candidate — which is why it is a separate knob.
    ///
    /// A state that states no value for `group` belongs to no register
    /// under this scope, so nothing can dominate it.
    pub fn within(mut self, group: Id) -> Self {
        self.within = Some(group.to_inline());
        self
    }

    /// Only states asserting `attribute: value` may dominate.
    ///
    /// The fixed-fact form of [`within`](Self::within), for the common case
    /// where observers are identified by a kind tag rather than by sharing
    /// the candidate's subject.
    pub fn among(mut self, attribute: Id, value: Inline<GenId>) -> Self {
        self.observer_states = Some((attribute.to_inline(), value));
        self
    }

    /// Resolve to the DAG's **roots** instead of its frontier.
    ///
    /// First-write-wins over an observation DAG: the states that observed
    /// nothing, i.e. the originals nothing was derived from.
    pub fn first(mut self) -> Self {
        self.end = End::First;
        self
    }

    /// Which end this order resolves to.
    pub fn end(&self) -> End {
        self.end
    }
}

impl<P> RegisterOrder for ObservationOrder<'_, P>
where
    P: TriblePattern,
{
    fn dominated(&self, state: Id) -> bool {
        let inline_state: Inline<GenId> = state.to_inline();

        // The unscoped hot path stays exactly what it was: one
        // short-circuited reverse-index probe, no observer materialised.
        if self.within.is_none() && self.observer_states.is_none() {
            return match self.end {
                // Something observed me, so it is above me.
                End::Last => exists!(temp!(
                    (observer),
                    self.facts.pattern(observer, self.observes, inline_state)
                )),
                // I observed something, so it is below me.
                End::First => exists!(temp!(
                    (observed),
                    self.facts.pattern::<GenId>(inline_state, self.observes, observed)
                )),
            };
        }

        // Under `within`, a state with no group value is in no register,
        // so no other state shares one with it.
        let group = match self.within {
            Some(group) => match self.group_of(inline_state) {
                Some(value) => Some((group, value)),
                None => return false,
            },
            None => None,
        };

        // Scoped: enumerate the neighbours across the edge and check each
        // against the scope. Both checks are point probes and `any`
        // short-circuits on the first witness.
        let neighbours: Box<dyn Iterator<Item = Id>> = match self.end {
            End::Last => Box::new(find!(
                observer: Id,
                self.facts.pattern(observer, self.observes, inline_state)
            )),
            End::First => Box::new(find!(
                observed: Id,
                self.facts
                    .pattern::<GenId>(inline_state, self.observes, observed)
            )),
        };
        neighbours.into_iter().any(|neighbour| {
            let neighbour: Inline<GenId> = neighbour.to_inline();
            let shares_group = match group {
                Some((attribute, value)) => {
                    exists!(self.facts.pattern(neighbour, attribute, value))
                }
                None => true,
            };
            let states_fact = match self.observer_states {
                Some((attribute, value)) => {
                    exists!(self.facts.pattern(neighbour, attribute, value))
                }
                None => true,
            };
            shares_group && states_fact
        })
    }
}

impl<P> ObservationOrder<'_, P>
where
    P: TriblePattern,
{
    /// This state's value for the `within` group attribute, if it has one.
    fn group_of(&self, state: Inline<GenId>) -> Option<Inline<GenId>> {
        let group = self.within?;
        find!(
            value: Id,
            self.facts.pattern(state, group, value)
        )
        .next()
        .map(|value| value.to_inline())
    }
}

/// Order by a **stated** key, compared by value within a group.
///
/// Where [`ObservationOrder`] reads an edge the writer asserted, this reads
/// a value the writer stated — a wall-clock timestamp, a counter, a version
/// number — and compares states that name the same subject.
///
/// Two attributes parameterise it:
///
/// * `group` points from a state to the subject it is a state *of*. Unlike
///   the observation DAG, where the edge itself joins the states of one
///   register, a stated key is meaningless across registers: comparing one
///   goal's timestamp with another's answers nothing. The grouping the
///   observation order leaves implicit must therefore be named here.
/// * `key` carries the comparable value.
///
/// # Byte order is value order
///
/// Keys are compared as raw inline bytes, which equals the value order
/// exactly when the encoding is order-preserving (big-endian numerics,
/// interval timestamps). This is the same contract
/// [`value_range`](crate::query::rangeconstraint::value_range) already
/// carries, and the `>=` half of every comparison is pushed into the engine
/// through it.
///
/// # Ties, and what a tie-break buys
///
/// Without [`tiebreak_by_id`](Self::tiebreak_by_id) the order is partial:
/// equal keys are incomparable, so two states written in the same clock tick
/// both survive and the reader sees the tie. With it the order is total —
/// `(key, id)` compared lexicographically — so [`sole`] always answers, and
/// that is precisely a last-write-wins register over a stated clock.
///
/// The tie-break is opt-in because it is not free: a total order reports one
/// winner where there were two writers, and whether that is a resolution or
/// a concealment is the reader's call, not the substrate's.
#[derive(Clone, Copy, Debug)]
pub struct StatedOrder<'a, P, K: InlineEncoding> {
    facts: &'a P,
    group: Inline<GenId>,
    key: Inline<GenId>,
    end: End,
    tiebreak: bool,
    among: Option<(Inline<GenId>, Inline<GenId>)>,
    _key: PhantomData<K>,
}

impl<'a, P, K> StatedOrder<'a, P, K>
where
    P: TriblePattern,
    K: InlineEncoding,
{
    /// Order states of `facts` by `key`, among states sharing a `group`.
    pub fn new(facts: &'a P, group: Id, key: Id) -> Self {
        Self {
            facts,
            group: group.to_inline(),
            key: key.to_inline(),
            end: End::Last,
            tiebreak: false,
            among: None,
            _key: PhantomData,
        }
    }

    /// Only states asserting `attribute: value` participate in this
    /// register.
    ///
    /// A grouping attribute is rarely as selective as it looks. Compass
    /// hangs status events, notes, and priority events off a goal through
    /// the *same* `board::task` edge, and every one of them carries a
    /// timestamp — so "the greatest key among states naming this goal"
    /// silently compares a status event against a note, and a goal whose
    /// last activity was a note has no current status at all. That is not a
    /// hypothetical: it is what the live pile does, on 778 of 2939 goals.
    ///
    /// The register is the *status events* of a goal, not everything
    /// attached to it, and a kind tag is how that is said. Restricting the
    /// candidate set is not enough, because domination asks about the whole
    /// frame — a state outside the candidate set can still dominate one
    /// inside it, which is exactly the property that makes the unscoped
    /// form correct for the observation DAG and wrong here.
    pub fn among(mut self, attribute: Id, value: Inline<GenId>) -> Self {
        self.among = Some((attribute.to_inline(), value));
        self
    }

    /// Resolve to the **least** key instead of the greatest.
    ///
    /// First-write-wins. The order is the dual of the original, and `min` is
    /// its join, so this is as lawful a derivation as `max`.
    pub fn first(mut self) -> Self {
        self.end = End::First;
        self
    }

    /// Break key ties by state id, making the order total.
    ///
    /// This is the composite order — timestamp first, id second — that a
    /// last-write-wins register wants, and it is what lets [`sole`] be
    /// total rather than partial.
    pub fn tiebreak_by_id(mut self) -> Self {
        self.tiebreak = true;
        self
    }

    /// Which end this order resolves to.
    pub fn end(&self) -> End {
        self.end
    }

    /// This state's group and key, if it states both.
    ///
    /// A state asserting several groups or several keys has no single
    /// coordinate, and this takes the first pair the index yields. That is
    /// deterministic but arbitrary, and it is the one place this order
    /// assumes something about the data rather than reading it: a stated
    /// key is only a coordinate if there is exactly one of it. Designs that
    /// permit multiplicity want the observation order, where a state with
    /// two predecessors is a merge rather than an ambiguity.
    fn coordinate(&self, state: Inline<GenId>) -> Option<(Id, Inline<K>)> {
        find!(
            (group: Id, key: Inline<K>),
            and!(
                self.facts.pattern(state, self.group, group),
                self.facts.pattern(state, self.key, key),
            )
        )
        .next()
    }
}

impl<P, K> RegisterOrder for StatedOrder<'_, P, K>
where
    P: TriblePattern,
    K: InlineEncoding,
{
    fn dominated(&self, state: Id) -> bool {
        let inline_state: Inline<GenId> = state.to_inline();
        // A state that states no coordinate is incomparable to everything,
        // so nothing dominates it and it survives as a concurrent value.
        let Some((group, key)) = self.coordinate(inline_state) else {
            return false;
        };
        let group: Inline<GenId> = group.to_inline();

        // Push the wide half of the comparison into the engine: only states
        // on the dominating side of `key` can possibly dominate. The strict
        // and tie-break halves are then decided per row, and `any`
        // short-circuits on the first witness.
        let (low, high) = match self.end {
            End::Last => (key, Inline::<K>::new([0xff; 32])),
            End::First => (Inline::<K>::new([0x00; 32]), key),
        };
        find!(
            (other: Id, candidate: Inline<K>),
            and!(
                self.facts.pattern(other, self.group, group),
                self.facts.pattern(other, self.key, candidate),
                value_range(candidate, low, high),
            )
        )
        .any(|(other, candidate)| {
            if !self.beats(candidate.raw, other, key.raw, state) {
                return false;
            }
            match self.among {
                Some((attribute, value)) => {
                    let other: Inline<GenId> = other.to_inline();
                    exists!(self.facts.pattern(other, attribute, value))
                }
                None => true,
            }
        })
    }
}

impl<P, K> StatedOrder<'_, P, K>
where
    P: TriblePattern,
    K: InlineEncoding,
{
    /// Whether `(challenger_key, challenger)` is strictly above
    /// `(key, state)` at this order's end.
    fn beats(
        &self,
        challenger_key: [u8; 32],
        challenger: Id,
        key: [u8; 32],
        state: Id,
    ) -> bool {
        // Equal keys are incomparable unless the reader asked for a total
        // order, and a state never dominates itself under either.
        if challenger_key == key {
            return self.tiebreak
                && match self.end {
                    End::Last => challenger > state,
                    End::First => challenger < state,
                };
        }
        match self.end {
            End::Last => challenger_key > key,
            End::First => challenger_key < key,
        }
    }
}

/// Restricts a variable to the states that are maximal under an order.
///
/// A filter-only [`Constraint`] in the shape of
/// [`InlineRange`](crate::query::rangeconstraint::InlineRange): it never
/// proposes, and its estimate of `usize::MAX` makes the intersection sort it
/// last, so a `pattern!` proposes the states in scope and this kills the
/// dominated ones.
///
/// That ordering is exactly right. Resolution has no independent selectivity
/// to offer — it can only shrink a set somebody else enumerated — so being
/// planned last is the planner using it correctly, not a limitation.
///
/// ```rust,ignore
/// find!((entry: Id),
///     and!(
///         pattern!(&facts, [{ ?entry @ kind: WIKI_ENTRY }]),
///         maximal(entry, &order),
///     )
/// )
/// ```
///
/// When an exact cardinality *is* wanted — so the planner can order around
/// resolution rather than after it — materialise with [`resolve`], sort, and
/// use [`SortedSlice`](crate::query::sortedsliceconstraint::SortedSlice),
/// whose estimate is the resolved count itself.
pub struct Maximal<'a, O> {
    variable: VariableId,
    order: &'a O,
}

impl<'a, O> Maximal<'a, O>
where
    O: RegisterOrder,
{
    /// Constrain `variable` to states nothing dominates under `order`.
    pub fn new(variable: Variable<GenId>, order: &'a O) -> Self {
        Self {
            variable: variable.index,
            order,
        }
    }

    /// Whether the state encoded in `raw` survives this order.
    fn survives(&self, raw: &[u8; 32]) -> bool {
        match Id::try_from_inline(Inline::<GenId>::as_transmute_raw(raw)) {
            Ok(state) => !self.order.dominated(state),
            // A value that is not a well-formed id is not a state of any
            // register, so it cannot be one of the current ones.
            Err(_) => false,
        }
    }
}

/// Convenience constructor for [`Maximal`].
pub fn maximal<'a, O>(variable: Variable<GenId>, order: &'a O) -> Maximal<'a, O>
where
    O: RegisterOrder,
{
    Maximal::new(variable, order)
}

impl<'a, O> Constraint<'a> for Maximal<'a, O>
where
    O: RegisterOrder,
{
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.variable)
    }

    /// Returns `usize::MAX` so the intersection never chooses this
    /// constraint as the proposer — it only confirms.
    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.variable == variable {
            Some(usize::MAX)
        } else {
            None
        }
    }

    /// Does not propose — the paired pattern constraint enumerates scope.
    fn propose(
        &self,
        _variable: VariableId,
        _frontier: &Frontier<'_>,
        _proposals: &mut ProposalBuffer,
    ) {
        // Intentionally empty: this constraint only confirms.
    }

    /// Kills every proposal that something else in the frame dominates.
    ///
    /// The verdict does not depend on the parent binding — domination is a
    /// property of the state and the frame — so the parent tags are ignored.
    fn confirm(&self, variable: VariableId, _frontier: &Frontier<'_>, cands: &mut Candidates<'_>) {
        if self.variable != variable {
            return;
        }
        for i in 0..cands.len() {
            if cands.is_live(i) && !self.survives(&cands.values()[i]) {
                cands.kill(i);
            }
        }
    }

    /// Returns `false` when the bound state is dominated.
    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.variable) {
            Some(raw) => self.survives(raw),
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ExclusiveId;
    use crate::inline::encodings::time::{i128_to_ordered_be, NsTAIInterval};
    use crate::macros::entity;
    use crate::metadata;
    use crate::prelude::*;

    attributes! {
        /// The subject a note is a note *of* — the register's grouping.
        "46D95EBAB8D5D0E9103148B35731065C" as note_of: crate::inline::encodings::genid::GenId;
        /// When the note was written — the stated key, order-preserving.
        "3B7AD155842AE30B164A311858D4D9A6" as note_at: NsTAIInterval;
    }

    /// A point interval, which is the shape compass writes and validates.
    ///
    /// `NsTAIInterval` is order-preserving big-endian `(lower, upper)`, so
    /// byte-lexicographic comparison of a point equals comparison of its
    /// instant — which is exactly compass's `interval_key`.
    fn at(nanos: i128) -> Inline<NsTAIInterval> {
        let bound = i128_to_ordered_be(nanos);
        let mut raw = [0u8; 32];
        raw[0..16].copy_from_slice(&bound);
        raw[16..32].copy_from_slice(&bound);
        Inline::new(raw)
    }

    fn edge(successor: &ExclusiveId, predecessor: &ExclusiveId) -> TribleSet {
        entity! { successor @ metadata::supersedes: predecessor }.into()
    }

    fn observation<P: TriblePattern>(facts: &P) -> ObservationOrder<'_, P> {
        ObservationOrder::new(facts, metadata::supersedes.id())
    }

    fn note(subject: &ExclusiveId, state: &ExclusiveId, nanos: i128) -> TribleSet {
        let when = at(nanos);
        entity! { state @ note_of: subject, note_at: when }.into()
    }

    fn stated<P: TriblePattern>(facts: &P) -> StatedOrder<'_, P, NsTAIInterval> {
        StatedOrder::new(facts, note_of.id(), note_at.id())
    }

    // ---------------------------------------------------------------
    // The observation order — the behaviour `latest` already had.
    // ---------------------------------------------------------------

    #[test]
    fn a_lone_state_is_its_own_frontier() {
        let only = ufoid();
        let facts = TribleSet::new();
        assert_eq!(
            resolve(&observation(&facts), [*only]),
            [*only].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn concurrent_states_both_survive() {
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();
        let mut facts = TribleSet::new();
        facts += edge(&left, &base);
        facts += edge(&right, &base);
        assert_eq!(
            resolve(&observation(&facts), [*base, *left, *right]),
            [*left, *right].into_iter().collect::<BTreeSet<_>>()
        );
        // A fork is exactly where last-write-wins has no answer, and
        // reporting one would be inventing the order.
        assert_eq!(sole(&observation(&facts), [*base, *left, *right]), None);
    }

    #[test]
    fn first_resolves_to_the_roots() {
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();
        let mut facts = TribleSet::new();
        facts += edge(&left, &base);
        facts += edge(&right, &base);
        let candidates = [*base, *left, *right];
        // The DAG has one root, so first-write-wins is total here even
        // though last-write-wins is not.
        assert_eq!(
            resolve(&observation(&facts).first(), candidates),
            [*base].into_iter().collect::<BTreeSet<_>>()
        );
        assert_eq!(sole(&observation(&facts).first(), candidates), Some(*base));
    }

    // ---------------------------------------------------------------
    // Observer scope — the axis every hand-rolled holdout differs on.
    // ---------------------------------------------------------------

    #[test]
    fn an_observer_outside_the_track_demotes_only_when_unscoped() {
        let track = ufoid();
        let other_track = ufoid();
        let head = ufoid();
        let interloper = ufoid();

        let mut facts = TribleSet::new();
        facts += entity! { &head @ note_of: &track };
        // Something on a *different* track observes our head.
        facts += entity! { &interloper @ note_of: &other_track };
        facts += edge(&interloper, &head);

        // Unscoped — any observer at all demotes. This is what `latest`
        // does today, and what the ERP corporate-group port widened to.
        assert_eq!(
            resolve(&observation(&facts), [*head]),
            BTreeSet::new(),
            "an unscoped frontier lets any entity in the frame demote a head"
        );

        // Scoped to the register — only states sharing my subject may
        // dominate me. This is the rule `track_head` hand-rolls.
        assert_eq!(
            resolve(&observation(&facts).within(note_of.id()), [*head]),
            [*head].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn observers_can_be_restricted_by_a_stated_fact() {
        let head = ufoid();
        let snapshot = ufoid();
        let bystander = ufoid();
        let kind = ufoid();

        let mut facts = TribleSet::new();
        facts += entity! { &bystander @ metadata::name: "not a snapshot" };
        facts += edge(&bystander, &head);

        // Only snapshots count as observers — the rule revolver's
        // `Heads::build` hand-rolls. The bystander cannot demote.
        let scoped = observation(&facts).among(metadata::tag.id(), (*kind).to_inline());
        assert_eq!(
            resolve(&scoped, [*head]),
            [*head].into_iter().collect::<BTreeSet<_>>()
        );

        // Tag a real snapshot as an observer and the head falls.
        let mut with_snapshot = facts.clone();
        with_snapshot += entity! { &snapshot @ metadata::tag: &kind };
        with_snapshot += edge(&snapshot, &head);
        let scoped =
            observation(&with_snapshot).among(metadata::tag.id(), (*kind).to_inline());
        assert_eq!(resolve(&scoped, [*head]), BTreeSet::new());
    }

    // ---------------------------------------------------------------
    // The stated order — the case the generalisation exists for.
    // ---------------------------------------------------------------

    #[test]
    fn stated_keys_resolve_within_their_own_group() {
        let goal = ufoid();
        let other_goal = ufoid();
        let early = ufoid();
        let late = ufoid();
        let elsewhere = ufoid();

        let mut facts = TribleSet::new();
        facts += note(&goal, &early, 10);
        facts += note(&goal, &late, 20);
        // A larger key in a *different* register must not dominate.
        facts += note(&other_goal, &elsewhere, 99);

        let candidates = [*early, *late, *elsewhere];
        assert_eq!(
            resolve(&stated(&facts), candidates),
            [*late, *elsewhere].into_iter().collect::<BTreeSet<_>>()
        );
        // First-write-wins is the same operation at the other end.
        assert_eq!(
            resolve(&stated(&facts).first(), candidates),
            [*early, *elsewhere].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn equal_keys_are_a_fork_until_the_reader_asks_for_a_total_order() {
        let goal = ufoid();
        let one = ufoid();
        let two = ufoid();
        let mut facts = TribleSet::new();
        facts += note(&goal, &one, 42);
        facts += note(&goal, &two, 42);
        let candidates = [*one, *two];

        // Partial: the tie is visible, and LWW correctly has no answer.
        assert_eq!(
            resolve(&stated(&facts), candidates),
            [*one, *two].into_iter().collect::<BTreeSet<_>>()
        );
        assert_eq!(sole(&stated(&facts), candidates), None);

        // Total: the id breaks the tie, so LWW always answers.
        assert_eq!(
            sole(&stated(&facts).tiebreak_by_id(), candidates),
            Some((*one).max(*two))
        );
        assert_eq!(
            sole(&stated(&facts).tiebreak_by_id().first(), candidates),
            Some((*one).min(*two))
        );
    }

    /// The grouping attribute is shared by more kinds of state than the
    /// register contains, so the dominator side needs its own scope. This
    /// is compass's shape exactly: a note and a status event both hang off
    /// the goal and both carry a timestamp.
    #[test]
    fn a_foreign_kind_on_the_same_group_dominates_unless_scoped() {
        let goal = ufoid();
        let status = ufoid();
        let commentary = ufoid();
        let status_kind = ufoid();

        let mut facts = TribleSet::new();
        facts += note(&goal, &status, 10);
        facts += entity! { &status @ metadata::tag: &status_kind };
        // A later note on the same goal, which is not a status event.
        facts += note(&goal, &commentary, 20);

        // Unscoped, the note dominates the status event and the goal has
        // no current status at all.
        assert_eq!(resolve(&stated(&facts), [*status]), BTreeSet::new());
        assert_eq!(sole(&stated(&facts).tiebreak_by_id(), [*status]), None);

        // Scoped to the register's own kind, the status event stands.
        let order = stated(&facts)
            .tiebreak_by_id()
            .among(metadata::tag.id(), (*status_kind).to_inline());
        assert_eq!(sole(&order, [*status]), Some(*status));
    }

    #[test]
    fn a_state_without_a_key_is_incomparable_rather_than_lost() {
        let goal = ufoid();
        let keyed = ufoid();
        let bare = ufoid();
        let mut facts = TribleSet::new();
        facts += note(&goal, &keyed, 7);
        facts += entity! { &bare @ metadata::name: "no coordinate" };
        assert_eq!(
            resolve(&stated(&facts), [*keyed, *bare]),
            [*keyed, *bare].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn the_empty_order_leaves_every_state_current() {
        let a = ufoid();
        let b = ufoid();
        let mut facts = TribleSet::new();
        facts += edge(&b, &a);
        // Named by the reader: the DAG is there, and this reader declines
        // to resolve by it.
        assert_eq!(
            resolve(&Unordered, [*a, *b]),
            [*a, *b].into_iter().collect::<BTreeSet<_>>()
        );
        let _ = facts;
    }

    // ---------------------------------------------------------------
    // The laws. Tested per order, never assumed from one of them.
    // ---------------------------------------------------------------

    /// Order-independence: the predicate reads a finished set, never a
    /// running one, so insertion order cannot matter.
    #[test]
    fn arrival_order_does_not_change_the_result() {
        let goal = ufoid();
        let early = ufoid();
        let late = ufoid();
        let candidates = [*early, *late];
        let expected: BTreeSet<Id> = [*late].into_iter().collect();

        let mut forwards = TribleSet::new();
        forwards += note(&goal, &early, 1);
        forwards += note(&goal, &late, 2);
        let mut backwards = TribleSet::new();
        backwards += note(&goal, &late, 2);
        backwards += note(&goal, &early, 1);
        assert_eq!(resolve(&stated(&forwards), candidates), expected);
        assert_eq!(resolve(&stated(&backwards), candidates), expected);

        // ... and the same for the observation order.
        let mut edge_forwards = TribleSet::new();
        edge_forwards += entity! { &early @ metadata::name: "e" };
        edge_forwards += edge(&late, &early);
        let mut edge_backwards = TribleSet::new();
        edge_backwards += edge(&late, &early);
        edge_backwards += entity! { &early @ metadata::name: "e" };
        assert_eq!(resolve(&observation(&edge_forwards), candidates), expected);
        assert_eq!(resolve(&observation(&edge_backwards), candidates), expected);
    }

    /// Frame-relativity: two readers holding different commit sets
    /// legitimately disagree about what is current.
    #[test]
    fn frames_disagree_and_both_are_right() {
        let goal = ufoid();
        let first = ufoid();
        let second = ufoid();
        let candidates = [*first, *second];

        let mut early = TribleSet::new();
        early += note(&goal, &first, 1);
        let mut late = early.clone();
        late += note(&goal, &second, 2);

        // The reader who has not seen the correction still says `first` —
        // and `second`, which its frame knows nothing about, states no
        // coordinate there, so it is incomparable rather than absent.
        assert_eq!(
            resolve(&stated(&early), candidates),
            [*first, *second].into_iter().collect::<BTreeSet<_>>()
        );
        assert_eq!(
            resolve(&stated(&late), candidates),
            [*second].into_iter().collect::<BTreeSet<_>>()
        );
    }

    /// The join-homomorphism identity
    /// `resolve(C1 ∪ C2) = resolve(C1) ⊔ resolve(C2)`, where `⊔` is the
    /// antichain join — the maximal elements of the two answers taken
    /// together, computed in the union frame.
    ///
    /// A macro rather than a function because the order borrows the frame
    /// it is built over, which a `Fn(&TribleSet) -> O` cannot express.
    macro_rules! assert_join_homomorphism {
        ($build:expr, $c1:expr, $c2:expr, $candidates:expr $(,)?) => {{
            let c1 = $c1;
            let c2 = $c2;
            let candidates = $candidates;
            let build = $build;

            let mut union = c1.clone();
            union += c2.clone();

            let l1 = resolve(&build(&c1), candidates);
            let l2 = resolve(&build(&c2), candidates);
            let joined = resolve(&build(&union), l1.iter().chain(l2.iter()).copied());
            assert_eq!(resolve(&build(&union), candidates), joined);
            joined
        }};
    }

    /// The join in the codomain is the **antichain** join, not set union,
    /// and this pins the difference — without it the homomorphism tests
    /// above could pass under a wrong-but-coincidental join.
    ///
    /// In the inclusion lattice the same map is antitone: a state maximal
    /// in `C1` alone can be demoted by a successor that only `C2` knows
    /// about, so `resolve(C1) union resolve(C2)` strictly overshoots. The
    /// operation was never non-monotone; it was being read in the wrong
    /// lattice.
    #[test]
    fn plain_union_is_not_the_join_in_the_codomain() {
        let base = ufoid();
        let left = ufoid();
        let candidates = [*base, *left];

        // One frame holds only the base; the other holds the successor.
        let c1 = TribleSet::new();
        let mut c2 = TribleSet::new();
        c2 += edge(&left, &base);
        let mut union = c1.clone();
        union += c2.clone();

        let l1 = resolve(&observation(&c1), candidates);
        let l2 = resolve(&observation(&c2), candidates);
        let truth = resolve(&observation(&union), candidates);

        // Naive set union keeps `base`, which the union frame has moved
        // past ...
        let naive: BTreeSet<Id> = l1.union(&l2).copied().collect();
        assert_eq!(naive, [*base, *left].into_iter().collect::<BTreeSet<_>>());
        assert_ne!(naive, truth);

        // ... while re-resolving in the union frame — the antichain join —
        // agrees exactly.
        let joined = resolve(&observation(&union), l1.iter().chain(l2.iter()).copied());
        assert_eq!(joined, truth);
        assert_eq!(truth, [*left].into_iter().collect::<BTreeSet<_>>());
    }

    #[test]
    fn observation_order_is_a_join_homomorphism() {
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();
        let merge = ufoid();

        let mut c1 = TribleSet::new();
        c1 += edge(&left, &base);
        let mut c2 = TribleSet::new();
        c2 += edge(&right, &base);
        c2 += edge(&merge, &right);

        let joined = assert_join_homomorphism!(
            |facts| observation(facts),
            c1,
            c2,
            [*base, *left, *right, *merge],
        );
        // Guard against the law passing vacuously on an empty answer.
        assert_eq!(joined, [*left, *merge].into_iter().collect::<BTreeSet<_>>());
    }

    #[test]
    fn last_write_wins_is_a_join_homomorphism() {
        let goal = ufoid();
        let a = ufoid();
        let b = ufoid();
        let c = ufoid();
        let d = ufoid();

        let mut c1 = TribleSet::new();
        c1 += note(&goal, &a, 1);
        c1 += note(&goal, &c, 3);
        let mut c2 = TribleSet::new();
        c2 += note(&goal, &b, 2);
        c2 += note(&goal, &d, 4);

        let joined = assert_join_homomorphism!(
            |facts| stated(facts).tiebreak_by_id(),
            c1,
            c2,
            [*a, *b, *c, *d],
        );
        assert_eq!(joined, [*d].into_iter().collect::<BTreeSet<_>>());
    }

    /// `min` is the join of the opposite order, so first-write-wins is as
    /// lawful a derivation as last-write-wins. Checked, not assumed.
    #[test]
    fn first_write_wins_is_a_join_homomorphism() {
        let goal = ufoid();
        let a = ufoid();
        let b = ufoid();
        let c = ufoid();
        let d = ufoid();

        let mut c1 = TribleSet::new();
        c1 += note(&goal, &a, 10);
        c1 += note(&goal, &c, 30);
        let mut c2 = TribleSet::new();
        c2 += note(&goal, &b, 20);
        c2 += note(&goal, &d, 40);

        let joined = assert_join_homomorphism!(
            |facts| stated(facts).tiebreak_by_id().first(),
            c1,
            c2,
            [*a, *b, *c, *d],
        );
        assert_eq!(joined, [*a].into_iter().collect::<BTreeSet<_>>());
    }

    /// The partial stated order — no tie-break — is a join homomorphism
    /// too. This is the multi-value register over a stated clock, and it
    /// is the one whose answer is genuinely a set.
    #[test]
    fn multi_value_stated_order_is_a_join_homomorphism() {
        let goal = ufoid();
        let a = ufoid();
        let b = ufoid();
        let c = ufoid();
        let d = ufoid();

        let mut c1 = TribleSet::new();
        c1 += note(&goal, &a, 5);
        c1 += note(&goal, &c, 5);
        let mut c2 = TribleSet::new();
        c2 += note(&goal, &b, 5);
        c2 += note(&goal, &d, 2);

        let joined = assert_join_homomorphism!(|facts| stated(facts), c1, c2, [*a, *b, *c, *d]);
        assert_eq!(joined, [*a, *b, *c].into_iter().collect::<BTreeSet<_>>());
    }

    /// The scoped observation order is a join homomorphism as well —
    /// narrowing the observer set does not disturb the lattice argument,
    /// because it only changes which edges count as domination.
    #[test]
    fn scoped_observation_order_is_a_join_homomorphism() {
        let track = ufoid();
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();

        let mut c1 = TribleSet::new();
        for state in [&base, &left, &right] {
            c1 += entity! { state @ note_of: &track };
        }
        c1 += edge(&left, &base);
        let mut c2 = TribleSet::new();
        c2 += edge(&right, &base);

        let joined = assert_join_homomorphism!(
            |facts| observation(facts).within(note_of.id()),
            c1,
            c2,
            [*base, *left, *right, *track],
        );
        assert_eq!(
            joined,
            [*left, *right, *track].into_iter().collect::<BTreeSet<_>>()
        );
    }

    // ---------------------------------------------------------------
    // The constraint exposure.
    // ---------------------------------------------------------------

    #[test]
    fn maximal_composes_inside_a_query() {
        let goal = ufoid();
        let early = ufoid();
        let late = ufoid();
        let mut facts = TribleSet::new();
        facts += note(&goal, &early, 1);
        facts += note(&goal, &late, 2);

        // Without the register, the pattern proposes both states ...
        let all: BTreeSet<Id> = find!(
            state: Id,
            pattern!(&facts, [{ ?state @ note_of: &goal }])
        )
        .collect();
        assert_eq!(all, [*early, *late].into_iter().collect::<BTreeSet<_>>());

        // ... and with it, the planner proposes from the pattern and the
        // register kills the dominated one. No candidate materialisation
        // on the caller's side.
        let order = stated(&facts).tiebreak_by_id();
        let current: BTreeSet<Id> = find!(
            state: Id,
            and!(
                pattern!(&facts, [{ ?state @ note_of: &goal }]),
                maximal(state, &order),
            )
        )
        .collect();
        assert_eq!(current, [*late].into_iter().collect::<BTreeSet<_>>());
        assert_eq!(current, resolve(&order, all));
    }

    #[test]
    fn maximal_agrees_with_resolve_over_the_observation_order() {
        let track = ufoid();
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();
        let mut facts = TribleSet::new();
        facts += edge(&left, &base);
        facts += edge(&right, &base);
        for state in [&base, &left, &right] {
            facts += entity! { state @ note_of: &track };
        }

        let order = observation(&facts);
        let queried: BTreeSet<Id> = find!(
            state: Id,
            and!(
                pattern!(&facts, [{ ?state @ note_of: &track }]),
                maximal(state, &order),
            )
        )
        .collect();
        assert_eq!(queried, resolve(&order, [*base, *left, *right]));
        assert_eq!(queried, [*left, *right].into_iter().collect::<BTreeSet<_>>());
    }
}
