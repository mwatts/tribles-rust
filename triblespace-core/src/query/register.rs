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
//! # A register needs two fields, and a timestamp is only one of them
//!
//! A register is a set of states that are *versions of the same thing*,
//! ordered. That is two facts, and an order carrying only one of them
//! cannot be resolved at all. [`ObservationOrder`] gets both from a single
//! edge: "I observed that" says *these are versions of the same thing* and
//! *this one is later* in the same breath, which is why an observation DAG
//! takes no second attribute and needs no scope.
//!
//! A stated key carries no identity whatever. A timestamp says *when*;
//! nothing in it says *of what*. [`StatedOrder`] therefore names the
//! identity as its own attribute, and the two together — an identity and
//! an order — are the whole register.
//!
//! Inferring the identity from a weaker relation over-includes, and must.
//! Compass hangs status events, notes, and priority events off a goal
//! through the *same* `board::task` edge, and every one of them carries a
//! timestamp. `board::task` means *belongs to this goal*, which is not an
//! identity: a note and a status event both belong to the goal and are not
//! versions of one another. Read as identity it lets a note written at
//! t=20 dominate a status event written at t=10, so the goal reports no
//! current status — on 778 of 2939 goals in the live pile.
//!
//! An earlier shape patched that with a scoping knob — admit as dominators
//! only states that also assert a kind tag — reconstructing the composite
//! identity `(goal, status-kind)` out of a grouping plus a type filter.
//! That was never a third axis; it was the missing half of the first,
//! spelled out at every call site. The cure is to give the data the
//! identity it lacks, an attribute meaning *the status of goal G*, after
//! which no scope is wanted: a note is simply not in that register.
//!
//! # Identity lives on the collection, not at the call site
//!
//! Which attribute carries identity and which carries order is a property
//! of the *register*, never of the question a reader asks. Both are carried
//! as tribles on the collection descriptor, alongside a recipe id that names
//! the law -- [`STATED_ORDER_RECIPE_V1`] -- and nothing else. Two registers
//! over the same dataset but different attributes are therefore distinct
//! collections, because the descriptor's content differs and so does its
//! handle; they cannot be confused for one another's cache.
//!
//! An earlier design hashed the two attributes *into* the recipe id. That
//! made the digest the only carrier of the pair: nothing stored them, so no
//! reader could recover which attributes a register was over. The same
//! correction applies to
//! [`observed_union`](crate::collection::observed_union)'s observed edge and
//! to a path collection's automaton fingerprint, both of which are now
//! descriptor arguments too.
//!
//! What is left at the call site is the frame — which commits the reader
//! holds — and nothing else.
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

/// What an order left standing: nothing, exactly one state, or a fork.
///
/// The three cases are kept apart deliberately. An earlier shape returned
/// `Option<Id>` and collapsed "no states at all" and "several tied states"
/// into the same `None`, so a reader could not tell an empty register from an
/// undecided one — and had to call [`resolve`] again to learn WHICH states
/// forked, without even knowing whether that was worth doing. `resolve` had
/// already computed the fork; the old shape simply discarded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// No candidate survived, because there were none to begin with.
    Empty,
    /// Exactly one state is maximal. This is the last-write-wins read.
    Sole(Id),
    /// Several states are mutually incomparable, and here they are.
    ///
    /// Under a total order (a timestamp with an id tie-break, say) a fork is
    /// impossible, so this variant means the order was not as total as the
    /// reader believed. Picking one would invent an order the data does not
    /// have and hide exactly the divergence a register exists to expose, so
    /// the fork is handed back instead of resolved.
    Fork(BTreeSet<Id>),
}

impl Resolution {
    /// The single maximal state, if the order left exactly one.
    pub fn sole(&self) -> Option<Id> {
        match self {
            Resolution::Sole(id) => Some(*id),
            _ => None,
        }
    }

    /// Every maximal state, whatever the shape.
    pub fn states(&self) -> BTreeSet<Id> {
        match self {
            Resolution::Empty => BTreeSet::new(),
            Resolution::Sole(id) => BTreeSet::from([*id]),
            Resolution::Fork(set) => set.clone(),
        }
    }
}

/// Resolve to exactly one state, an empty register, or the fork itself.
pub fn sole<O>(order: &O, candidates: impl IntoIterator<Item = Id>) -> Resolution
where
    O: RegisterOrder + ?Sized,
{
    let resolved = resolve(order, candidates);
    match resolved.len() {
        0 => Resolution::Empty,
        1 => Resolution::Sole(*resolved.iter().next().expect("length checked")),
        _ => Resolution::Fork(resolved),
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
///
/// # It needs no identity attribute, because the edge is one
///
/// A supersedes edge asserts, in one fact, both halves a register wants:
/// that the two states are versions of the same thing, and which of them
/// is later. So there is nothing here for a scope to narrow. An edge
/// arriving from what a reader considers a different track is not a state
/// outside the register — it is a claim to be *in* the register, and if
/// that claim is wrong the edge set is wrong. Referential integrity is a
/// validation pass over the edges; resolution must not quietly disagree
/// with the data it is reading.
#[derive(Clone, Copy, Debug)]
pub struct ObservationOrder<'a, P> {
    facts: &'a P,
    observes: Inline<GenId>,
    end: End,
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
        }
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
        match self.end {
            // Something observed me, so it is above me.
            End::Last => exists!(temp!(
                (observer),
                self.facts.pattern(observer, self.observes, inline_state)
            )),
            // I observed something, so it is below me.
            End::First => exists!(temp!(
                (observed),
                self.facts
                    .pattern::<GenId>(inline_state, self.observes, observed)
            )),
        }
    }
}

/// Order by a stated **identity** and a stated **order**.
///
/// Where [`ObservationOrder`] reads an edge whose single assertion carries
/// both halves of a register, this reads two values the writer stated
/// separately:
///
/// * `identity` points from a state to the register it is a state *of*.
///   Not to a subject it merely hangs off: `board::task` says a status
///   event *belongs to* a goal, and so do the goal's notes, which is why
///   that edge is not an identity and reading it as one lets a note retire
///   a status. The attribute wanted is the one that means *the status of
///   goal G* and nothing else.
/// * `order` carries the comparable value — a wall-clock timestamp, a
///   counter, a version number.
///
/// Neither is a scope, and there is no third knob. A state whose identity
/// is some other register is not filtered out; it was never in this one.
///
/// # Byte order is value order
///
/// Order values are compared as raw inline bytes, which equals the value
/// order exactly when the encoding is order-preserving (big-endian
/// numerics, interval timestamps). This is the same contract
/// [`value_range`](crate::query::rangeconstraint::value_range) already
/// carries, and the `>=` half of every comparison is pushed into the
/// engine through it.
///
/// # Ties, and what a tie-break buys
///
/// Without [`tiebreak_by_id`](Self::tiebreak_by_id) the order is partial:
/// equal values are incomparable, so two states written in the same clock
/// tick both survive and the reader sees the tie. With it the order is
/// total — `(order, id)` compared lexicographically — so [`sole`] always
/// answers, and that is precisely a last-write-wins register over a stated
/// clock.
///
/// The tie-break is opt-in because it is not free: a total order reports
/// one winner where there were two writers, and whether that is a
/// resolution or a concealment is the reader's call, not the substrate's.
#[derive(Clone, Copy, Debug)]
pub struct StatedOrder<'a, P, K: InlineEncoding> {
    facts: &'a P,
    identity: Inline<GenId>,
    order: Inline<GenId>,
    end: End,
    tiebreak: bool,
    _order: PhantomData<K>,
}

impl<'a, P, K> StatedOrder<'a, P, K>
where
    P: TriblePattern,
    K: InlineEncoding,
{
    /// The register whose states share an `identity`, ordered by `order`.
    pub fn new(facts: &'a P, identity: Id, order: Id) -> Self {
        Self {
            facts,
            identity: identity.to_inline(),
            order: order.to_inline(),
            end: End::Last,
            tiebreak: false,
            _order: PhantomData,
        }
    }

    /// Resolve to the **least** value instead of the greatest.
    ///
    /// First-write-wins. The order is the dual of the original, and `min`
    /// is its join, so this is as lawful a derivation as `max`.
    pub fn first(mut self) -> Self {
        self.end = End::First;
        self
    }

    /// Break ties on the order value by state id, making the order total.
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

    /// The identity of this measure of domination.
    ///
    /// Both attributes fold in, because both are the register: swap either
    /// one and the answer to "which states are current" is a different
    /// answer about a different thing. This is the same construction
    /// [`observed_union`](crate::collection::observed_union) uses for its
    /// observation edge and a path collection uses for its automaton
    /// fingerprint, and it is what lets several `current` lattices over
    /// one dataset coexist without sharing a cache.
    ///
    /// [`End`] and the tie-break are deliberately *not* folded in. They
    /// choose which end of this coordinate system a reader wants and how
    /// finely it is cut; the coordinate system is the same one either way.
    /// A maintained artifact that is specific to one end — as
    /// [`observed_union`](crate::collection::observed_union)'s dominated
    /// set is to [`End::Last`] — has to say so in its own descriptor.
    pub fn recipe() -> Id {
        STATED_ORDER_RECIPE_V1
    }
}

crate::macros::attributes! {
    /// The identity attribute a stated-order register reads.
    ///
    /// Minted with `trible genid` on 2026-08-19.
    "AAC4B8AD847759A6910662DB2F0321BA" as pub register_identity: GenId;
    /// The order attribute a stated-order register reads.
    ///
    /// Minted with `trible genid` on 2026-08-19.
    "435F580DA18908BAEB4EB675557E0BFD" as pub register_orders: GenId;
}

/// The stated-order law: an identity attribute plus a total order attribute.
///
/// This names the law only. *Which* attributes carry identity and order are
/// parameters on the collection descriptor, not folded into this id. Hashing
/// them in would leave the digest as the sole carrier of arguments that are
/// stored nowhere, so a reader could never recover what the register means
/// from the pile alone.
///
/// Minted with `trible genid` on 2026-08-19.
pub const STATED_ORDER_RECIPE_V1: Id = crate::id_hex!("6DD9E3F484DDDFF83BAC505ED33C8394");

/// The stated-order law, as a describable type.
pub struct StatedOrderV1;

impl crate::metadata::MetaDescribe for StatedOrderV1 {
    fn describe() -> crate::trible::Fragment {
        let id: Id = STATED_ORDER_RECIPE_V1;
        crate::macros::entity! {
            crate::id::ExclusiveId::force_ref(&id) @
                crate::metadata::name: "stated-order-v1",
                crate::metadata::description: "A register whose states are ordered by a stated value rather than by observation. One state dominates another when they share an identity and its order value is greater, so the resolution is the set of states nothing beats: a single winner when the order is total, a tie when it is not. Takes two arguments, carried as tribles on the collection descriptor: `register_identity`, the attribute saying two states are about the same thing, and `register_orders`, the attribute carrying the order. Identity and order are separate because an order value alone carries no identity.",
                crate::metadata::tag: crate::metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

impl<P, K> StatedOrder<'_, P, K>
where
    P: TriblePattern,
    K: InlineEncoding,
{
    /// This state's register and order value, if it states both.
    ///
    /// A state asserting several identities or several order values has no
    /// single coordinate, and this takes the first pair the index yields.
    /// That is deterministic but arbitrary, and it is the one place this
    /// order assumes something about the data rather than reading it: a
    /// stated coordinate is only a coordinate if there is exactly one of
    /// it. Designs that permit multiplicity want the observation order,
    /// where a state with two predecessors is a merge rather than an
    /// ambiguity.
    fn coordinate(&self, state: Inline<GenId>) -> Option<(Id, Inline<K>)> {
        find!(
            (register: Id, order: Inline<K>),
            and!(
                self.facts.pattern(state, self.identity, register),
                self.facts.pattern(state, self.order, order),
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
        // A state that states no coordinate is in no register under this
        // order, so nothing dominates it and it survives as a concurrent
        // value.
        let Some((register, key)) = self.coordinate(inline_state) else {
            return false;
        };
        let register: Inline<GenId> = register.to_inline();

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
                self.facts.pattern(other, self.identity, register),
                self.facts.pattern(other, self.order, candidate),
                value_range(candidate, low, high),
            )
        )
        .any(|(other, candidate)| self.beats(candidate.raw, other, key.raw, state))
    }
}

impl<P, K> StatedOrder<'_, P, K>
where
    P: TriblePattern,
    K: InlineEncoding,
{
    /// Whether `(challenger_key, challenger)` is strictly above
    /// `(key, state)` at this order's end.
    fn beats(&self, challenger_key: [u8; 32], challenger: Id, key: [u8; 32], state: Id) -> bool {
        // Equal values are incomparable unless the reader asked for a total
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
        /// The register a state is a state *of* — the stated identity.
        "46D95EBAB8D5D0E9103148B35731065C" as note_of: crate::inline::encodings::genid::GenId;
        /// When the state was written — the stated order, order-preserving.
        "3B7AD155842AE30B164A311858D4D9A6" as note_at: NsTAIInterval;
        /// The register that is *the status of* a subject — an identity, and
        /// a different one from `note_of` even on the same subject.
        "3DE10799440A0C3D4B40261E7089DCAD" as status_of: crate::inline::encodings::genid::GenId;
        /// A second order attribute, so the recipe can be shown to depend
        /// on both halves and not just the identity.
        "641C46439161BDA3F7EE8ED412AD5DEF" as other_clock: NsTAIInterval;
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
        assert_eq!(
            sole(&observation(&facts), [*base, *left, *right]).sole(),
            None
        );
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
        assert_eq!(
            sole(&observation(&facts).first(), candidates),
            Resolution::Sole(*base)
        );
    }

    /// The edge is the identity claim, so an observer a reader thinks
    /// belongs to a different track still dominates. There is deliberately
    /// no knob to suppress that: a supersedes edge across tracks is a
    /// wrong edge, and resolution reading the edge set faithfully is what
    /// lets a validation pass find it. Pinned so the axis cannot creep
    /// back in as a "fix".
    #[test]
    fn an_edge_dominates_whatever_the_reader_thinks_of_its_source() {
        let track = ufoid();
        let other_track = ufoid();
        let head = ufoid();
        let interloper = ufoid();

        let mut facts = TribleSet::new();
        facts += entity! { &head @ note_of: &track };
        facts += entity! { &interloper @ note_of: &other_track };
        facts += edge(&interloper, &head);

        assert_eq!(
            resolve(&observation(&facts), [*head]),
            BTreeSet::new(),
            "an observation edge is a claim of same-register membership, \
             and resolution must not silently disbelieve it"
        );
    }

    // ---------------------------------------------------------------
    // The stated order — the case the generalisation exists for.
    // ---------------------------------------------------------------

    #[test]
    fn states_resolve_only_within_their_own_register() {
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
        assert_eq!(sole(&stated(&facts), candidates).sole(), None);

        // Total: the id breaks the tie, so LWW always answers.
        assert_eq!(
            sole(&stated(&facts).tiebreak_by_id(), candidates),
            Resolution::Sole((*one).max(*two))
        );
        assert_eq!(
            sole(&stated(&facts).tiebreak_by_id().first(), candidates),
            Resolution::Sole((*one).min(*two))
        );
    }

    /// An empty register and a forked one are different answers, and the
    /// old `Option<Id>` shape could not tell them apart. A reader that sees
    /// "no single answer" must be able to ask whether that is because there
    /// is nothing here or because the order failed to decide — and if it
    /// failed, WHICH states it failed between.
    #[test]
    fn empty_and_forked_are_distinguishable_and_the_fork_is_returned() {
        let goal = ufoid();
        let one = ufoid();
        let two = ufoid();

        let mut facts = TribleSet::new();
        facts += note(&goal, &one, 10);
        facts += note(&goal, &two, 10); // equal keys: a genuine fork

        // Nothing to resolve at all.
        assert_eq!(sole(&stated(&facts), []), Resolution::Empty);

        // A fork, and it hands back the states that forked.
        let forked = sole(&stated(&facts), [*one, *two]);
        assert_eq!(
            forked,
            Resolution::Fork([*one, *two].into_iter().collect::<BTreeSet<_>>())
        );
        assert_eq!(forked.states().len(), 2, "the fork must not be empty");

        // Both report "no single answer", but they are not the same answer.
        assert_eq!(forked.sole(), None);
        assert_eq!(Resolution::Empty.sole(), None);
        assert_ne!(forked, Resolution::Empty);
    }

    /// The failure that produced the removed scoping axis, and its actual
    /// cure. `note_of` is *belongs to this subject*, shared by more kinds
    /// of state than any one register holds; read as an identity it lets a
    /// later note retire a status event, which is what the live pile did
    /// on 778 of 2939 goals. The register wanted is *the status of this
    /// subject*, and the fix is to say so in the data — after which the
    /// order needs nothing added to it, because the note was never in the
    /// register to be filtered out.
    #[test]
    fn a_weak_grouping_read_as_identity_lets_a_foreign_kind_dominate() {
        let goal = ufoid();
        let status = ufoid();
        let commentary = ufoid();

        let mut facts = TribleSet::new();
        facts += note(&goal, &status, 10);
        // A later note on the same subject, which is not a status event.
        facts += note(&goal, &commentary, 20);
        // The identity the register actually wants: the status *of* the
        // goal, which the note does not claim to be a version of.
        facts += entity! { &status @ status_of: &goal };

        // Grouping-as-identity: the note dominates, and the goal has no
        // current status at all.
        assert_eq!(resolve(&stated(&facts), [*status]), BTreeSet::new());
        assert_eq!(
            sole(&stated(&facts).tiebreak_by_id(), [*status]).sole(),
            None
        );

        // The identity attribute, with nothing else changed about the
        // order: the status event stands, and the note is simply not here.
        let order = StatedOrder::<_, NsTAIInterval>::new(&facts, status_of.id(), note_at.id())
            .tiebreak_by_id();
        assert_eq!(sole(&order, [*status]), Resolution::Sole(*status));
    }

    /// Two registers over one dataset share the *law* and are told apart by
    /// its arguments. The arguments live on the collection descriptor as
    /// ordinary tribles, so the descriptor's handle separates them while the
    /// recipe id stays a readable name for what kind of register this is.
    ///
    /// This replaces an earlier design in which the pair of attributes was
    /// hashed into the recipe id itself. That made the digest the only carrier
    /// of the pair: nothing stored them, so no reader could recover which
    /// attributes a register was over.
    #[test]
    fn registers_share_a_law_and_differ_by_their_parameters() {
        use crate::collection::descriptor;
        use crate::collection::records::{
            collection_name, collection_recipe, collection_representation, collection_team,
            CollectionName, KIND_COLLECTION_DESCRIPTOR,
        };
        // The law is one minted name, identical for every stated-order register.
        assert_eq!(
            StatedOrder::<TribleSet, NsTAIInterval>::recipe(),
            STATED_ORDER_RECIPE_V1
        );

        let team = ed25519_dalek::SigningKey::from_bytes(&[19; 32]).verifying_key();
        let representation = *ufoid();
        fn describe(
            team: ed25519_dalek::VerifyingKey,
            representation: Id,
            identity: Id,
            order: Id,
        ) -> Fragment {
            let name = CollectionName::new("register").unwrap();
            let representation: Inline<GenId> = representation.to_inline();
            let identity: Inline<GenId> = identity.to_inline();
            let order: Inline<GenId> = order.to_inline();
            crate::macros::entity! { _ @
                crate::metadata::tag: KIND_COLLECTION_DESCRIPTOR,
                collection_name: name.as_str(),
                collection_team: team,
                collection_representation: representation,
                collection_recipe*: <StatedOrderV1 as crate::metadata::MetaDescribe>::describe(),
                register_identity: identity,
                register_orders: order,
            }
        }

        let notes = describe(team, representation, note_of.id(), note_at.id());
        let statuses = describe(team, representation, status_of.id(), note_at.id());
        assert_ne!(
            descriptor::identity_for_tests(&notes),
            descriptor::identity_for_tests(&statuses),
            "two registers over the same order attribute must be two collections"
        );
        // Swapping the order attribute is as much a different register as
        // swapping the identity.
        assert_ne!(
            descriptor::identity_for_tests(&notes),
            descriptor::identity_for_tests(&describe(
                team,
                representation,
                note_of.id(),
                other_clock.id()
            ))
        );
        // And it is a function of the pair, not of the call.
        assert_eq!(
            descriptor::identity_for_tests(&notes),
            descriptor::identity_for_tests(&describe(
                team,
                representation,
                note_of.id(),
                note_at.id()
            ))
        );
        // The law itself is shared, not per-register.
        assert_eq!(
            descriptor::recipe(notes.facts()).unwrap(),
            descriptor::recipe(statuses.facts()).unwrap()
        );

        // The point of the change: the arguments are recoverable from the
        // descriptor. A reader holding only the pile can say what this
        // register is over.
        assert_eq!(
            descriptor::argument(notes.facts(), register_identity.id()),
            Some(<Id as crate::inline::IntoInline<GenId>>::to_inline(note_of.id()).raw),
        );
        assert_eq!(
            descriptor::argument(notes.facts(), register_orders.id()),
            Some(<Id as crate::inline::IntoInline<GenId>>::to_inline(note_at.id()).raw),
        );
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
        assert_eq!(
            queried,
            [*left, *right].into_iter().collect::<BTreeSet<_>>()
        );
    }
}
