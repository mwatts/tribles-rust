//! How far a collection may travel, stated as facts rather than as a type.
//!
//! Reach is a *fragment* the caller hands to a descriptor builder, not a Rust
//! enum the builder interprets. The builder spreads it into
//! [`collection_reach`](crate::collection::records::collection_reach), so what
//! the fragment exports is what the descriptor declares and what the fragment
//! carries alongside travels with it into the same blob.
//!
//! That is the whole reason it is data. One law is implemented and it is the
//! coarse one; the design has always said a narrower law would be *a different
//! id carrying its audience with it*. An enum cannot hold that case -- it would
//! need a new variant, a new signature, and every caller recompiled -- while a
//! fragment already can: [`public`] exports one id and carries nothing, and a
//! future team-subset law exports its own id and carries the audience facts
//! that say whose it is, with no change to any signature between here and the
//! descriptor.
//!
//! [`private`] is [`Fragment::empty`]: it exports nothing, so nothing is
//! written, so a private descriptor stays byte-identical to one built before
//! reach existed. That absence is load-bearing -- see the identity test in
//! [`descriptor`](crate::collection::descriptor).
//!
//! Two things a future law's author should know. A fragment exporting more
//! than one id writes more than one reach row, and [`declared`] reads that as
//! *no* reach rather than picking one, so a malformed reach fails closed the
//! same way a law this binary cannot read does. And a reach fragment's
//! carried facts join the descriptor blob without entering the descriptor
//! entity's content-derived id -- only the exported id does -- so a law with
//! arguments must root itself in those arguments, which an `entity!` does by
//! construction and a hand-rolled [`Fragment::rooted`] does not.

use crate::id::{id_hex, Id};
use crate::prelude::{find, pattern};
use crate::trible::{Fragment, TribleSet};

use super::records::collection_reach;

/// The reach law naming "any holder may relay this collection's strictly
/// verified commits to any peer that asks".
///
/// One law is implemented, and it is the coarse one. What it forecloses is
/// per-recipient scoping: this says *whether* a collection travels, not *to
/// whom*, so it cannot express "these two teammates but not the third". That
/// is deliberate for now rather than permanent -- a narrower law is a
/// different id carrying its audience with it, and needs no change to
/// [`collection_reach`] to exist. What is permanent is that reach is not
/// per-*author*: a collection travels or it does not, and an author who wants
/// different answers for different material writes it into different
/// collections. That is the same mechanism at a finer grain rather than a
/// second one.
///
/// Minted with `trible genid` on 2026-08-21.
pub const PUBLIC: Id = id_hex!("A7ACA286FE5599D92DB87E8A84A7767E");

/// The collection does not travel.
///
/// An empty fragment declares nothing, so the descriptor it is spread into
/// carries no reach attribute at all and keeps the identity it had before
/// reach existed. This is a value a caller must pass deliberately: there is no
/// default, because the old silent failure was exactly a reach nobody
/// remembered to state.
pub fn private() -> Fragment {
    Fragment::empty()
}

/// The collection travels under [`PUBLIC`].
///
/// The fragment exports the law id and carries no further facts, because this
/// law has no arguments -- it says *whether*, not *to whom*.
pub fn public() -> Fragment {
    Fragment::rooted(PUBLIC, TribleSet::new())
}

/// The reach law this descriptor declares, if it declares one readably.
///
/// Answers the raw law id rather than a verdict, because a reader that
/// implements a law this binary does not should be able to see which one it
/// met. A descriptor declaring nothing, declaring something unreadable, or
/// declaring twice all answer `None`: each is a descriptor that has not
/// stated a single reach, and [`travels`] treats them alike.
pub fn declared(facts: &TribleSet) -> Option<Id> {
    let descriptor = super::descriptor::entity(facts).ok()?;
    let mut rows = find!(
        (v: Id?),
        pattern!(facts, [{ descriptor @ collection_reach: ?v }])
    )
    .map(|(v,)| v);
    let first = rows.next()?.ok()?;
    if rows.next().is_some() {
        return None;
    }
    Some(first)
}

/// Whether this collection may be relayed to a peer.
///
/// **Read the collection's own descriptor and nothing else.** Authority is
/// likewise a local mandatory descriptor field; neither property walks
/// [`collection_source`](crate::collection::records::collection_source). A
/// derived collection declares its own reach or has none. Inheriting would be wrong
/// in both directions. A derivation can expose what its source did not -- an
/// index over private material still leaks the material's shape -- so
/// publishing a source must not publish everything computed from it. And an
/// aggregate deliberately published over private inputs is an ordinary thing
/// to want, which inheritance would forbid. Locality also keeps the answer
/// decidable from one blob, which is what lets a relay refuse without
/// resolving a chain it may not hold.
///
/// Absence is a `false`, not a missing answer. Every descriptor written
/// before this attribute existed says nothing, and every one of them stays
/// put.
pub fn travels(facts: &TribleSet) -> bool {
    declared(facts) == Some(PUBLIC)
}
