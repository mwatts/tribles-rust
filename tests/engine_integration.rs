//! Drive the triblespace query engine with our constraints —
//! validates that `Constraint::propose` + `confirm` actually
//! cooperate correctly when composed via
//! `IntersectionConstraint`. Unit tests exercise each method in
//! isolation; this test is the belt-and-braces check that the
//! full protocol holds in a real join.

use triblespace::core::id::{Id, RawId};
use triblespace::core::query::intersectionconstraint::IntersectionConstraint;
use triblespace::core::query::{Binding, Constraint, Variable, VariableContext};
use triblespace::core::value::schemas::genid::GenId;
use triblespace::core::value::RawValue;

use triblespace_search::bm25::BM25Builder;
use triblespace_search::succinct::SuccinctBM25Index;
use triblespace_search::tokens::hash_tokens;

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

fn id_as_raw_value(id: Id) -> RawValue {
    let mut out = [0u8; 32];
    let raw: &RawId = id.as_ref();
    out[16..32].copy_from_slice(raw);
    out
}

fn raw_value_to_id(raw: &RawValue) -> Option<Id> {
    if raw[0..16] != [0u8; 16] {
        return None;
    }
    let raw16: RawId = raw[16..32].try_into().ok()?;
    Id::new(raw16)
}

/// Build a tiny index and run two constraints through an
/// IntersectionConstraint. The two terms overlap on one doc;
/// the intersection should expose exactly that doc via
/// `propose` because the engine picks the smaller posting list
/// as the proposer and the other as the confirmer.
#[test]
fn intersection_of_two_bm25_constraints_yields_overlap() {
    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("the quick brown fox"));
    b.insert(id(2), hash_tokens("the lazy brown dog"));
    b.insert(id(3), hash_tokens("quick silver fox jumps"));
    let idx: SuccinctBM25Index = b.build();

    let mut ctx = VariableContext::new();
    let doc: Variable<GenId> = ctx.next_variable();

    let fox_term = hash_tokens("fox")[0];
    let quick_term = hash_tokens("quick")[0];
    // Both constraints touch `doc`. Box them so they share a
    // type for IntersectionConstraint<Vec<_>>.
    let c_fox: Box<dyn Constraint> = Box::new(idx.docs_containing(doc, fox_term));
    let c_quick: Box<dyn Constraint> = Box::new(idx.docs_containing(doc, quick_term));
    let intersection = IntersectionConstraint::new(vec![c_fox, c_quick]);

    // Sanity-check the composed variable set / estimate.
    assert!(intersection.variables().is_set(doc.index));

    let binding = Binding::default();
    // The intersection's estimate is the minimum of the two
    // children's estimates — both are 2, so 2.
    assert_eq!(intersection.estimate(doc.index, &binding), Some(2));

    // `propose` should yield the intersection of the two posting
    // lists. "fox" is in docs {1,3}; "quick" is in docs {1,3};
    // both sets happen to be identical → proposes both.
    let mut props: Vec<RawValue> = Vec::new();
    intersection.propose(doc.index, &binding, &mut props);
    let ids: std::collections::HashSet<Id> =
        props.iter().map(|r| raw_value_to_id(r).unwrap()).collect();
    assert!(ids.contains(&id(1)));
    assert!(ids.contains(&id(3)));
    assert!(!ids.contains(&id(2))); // "lazy brown dog" has neither term
}

/// Intersection with a disjoint term-pair — "banana" is absent
/// from the corpus entirely, so the intersection should propose
/// no docs at all.
#[test]
fn intersection_with_absent_term_proposes_nothing() {
    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("the quick brown fox"));
    b.insert(id(2), hash_tokens("the lazy brown dog"));
    let idx: SuccinctBM25Index = b.build();

    let mut ctx = VariableContext::new();
    let doc: Variable<GenId> = ctx.next_variable();

    let brown_term = hash_tokens("brown")[0];
    let banana_term = hash_tokens("banana")[0];
    let c_brown: Box<dyn Constraint> = Box::new(idx.docs_containing(doc, brown_term));
    let c_banana: Box<dyn Constraint> = Box::new(idx.docs_containing(doc, banana_term));
    let intersection = IntersectionConstraint::new(vec![c_brown, c_banana]);

    let binding = Binding::default();
    // The "banana" constraint's estimate is 0, so the
    // intersection's minimum-estimate is 0.
    assert_eq!(intersection.estimate(doc.index, &binding), Some(0));

    let mut props = Vec::new();
    intersection.propose(doc.index, &binding, &mut props);
    assert!(
        props.is_empty(),
        "no proposals for absent-term intersection"
    );
}

/// Pre-binding `doc` should let `satisfied` succeed only when
/// the bound id is in BOTH posting lists.
#[test]
fn satisfied_respects_both_clauses() {
    let mut b = BM25Builder::new();
    b.insert(id(1), hash_tokens("quick fox"));
    b.insert(id(2), hash_tokens("quick dog"));
    let idx: SuccinctBM25Index = b.build();

    let mut ctx = VariableContext::new();
    let doc: Variable<GenId> = ctx.next_variable();

    let quick_term = hash_tokens("quick")[0];
    let fox_term = hash_tokens("fox")[0];
    let c_quick: Box<dyn Constraint> = Box::new(idx.docs_containing(doc, quick_term));
    let c_fox: Box<dyn Constraint> = Box::new(idx.docs_containing(doc, fox_term));
    let intersection = IntersectionConstraint::new(vec![c_quick, c_fox]);

    // doc = 1: has both "quick" and "fox" → satisfied.
    let mut bind1 = Binding::default();
    bind1.set(doc.index, &id_as_raw_value(id(1)));
    assert!(intersection.satisfied(&bind1));

    // doc = 2: has "quick" but not "fox" → unsatisfied.
    let mut bind2 = Binding::default();
    bind2.set(doc.index, &id_as_raw_value(id(2)));
    assert!(!intersection.satisfied(&bind2));
}
