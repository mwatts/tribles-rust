//! Backend parity for `path!`: the same regular path query over a
//! `TribleSet` and over a `SuccinctArchive` view of the same graph must
//! produce identical result sets. The TribleSet side runs the direct
//! PATCH fast paths (and the typed residual program); the archive side
//! runs every hop through its one-pattern constraint surface.

use std::collections::HashSet;

use triblespace::core::blob::encodings::succinctarchive::OrderedUniverse;
use triblespace::core::blob::encodings::succinctarchive::SuccinctArchive;
use triblespace::prelude::*;

pub mod social {
    use triblespace::prelude::*;

    attributes! {
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" as follows: inlineencodings::GenId;
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" as likes: inlineencodings::GenId;
    }
}

struct Fixture {
    set: TribleSet,
    archive: SuccinctArchive<OrderedUniverse>,
    a: Inline<inlineencodings::GenId>,
    c: Inline<inlineencodings::GenId>,
}

/// A small graph with a follows-cycle, a branch, and likes edges:
/// follows: a→b, b→c, c→a, a→d; likes: b→d, d→e.
fn fixture() -> Fixture {
    let ids: Vec<_> = (0..5).map(|_| fucid()).collect();
    let (a, b, c, d, e) = (&ids[0], &ids[1], &ids[2], &ids[3], &ids[4]);
    let mut set = TribleSet::new();
    set += entity! { a @ social::follows: b };
    set += entity! { b @ social::follows: c };
    set += entity! { c @ social::follows: a };
    set += entity! { a @ social::follows: d };
    set += entity! { b @ social::likes: d };
    set += entity! { d @ social::likes: e };
    let archive: SuccinctArchive<OrderedUniverse> = (&set).into();
    Fixture {
        set,
        archive,
        a: a.to_inline(),
        c: c.to_inline(),
    }
}

/// Runs the same query body over both backends and asserts identical
/// result sets. The body is duplicated per backend because the two
/// sources have different types; `$src` is an owned clone of each.
macro_rules! assert_parity {
    ($fixture:expr, |$src:ident| $q:expr) => {{
        let set_results: HashSet<_> = {
            let $src = $fixture.set.clone();
            $q
        };
        let archive_results: HashSet<_> = {
            let $src = $fixture.archive.clone();
            $q
        };
        assert_eq!(
            set_results, archive_results,
            "TribleSet and SuccinctArchive disagreed"
        );
        assert!(
            !set_results.is_empty(),
            "parity case matched nothing; fixture no longer exercises it"
        );
        set_results
    }};
}

#[test]
fn parity_single_attr() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s social::follows e)
    )
    .collect());
}

#[test]
fn parity_plus() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s social::follows+ e)
    )
    .collect());
}

#[test]
fn parity_star() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s social::follows* e)
    )
    .collect());
}

#[test]
fn parity_inverse() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s ^social::follows e)
    )
    .collect());
}

#[test]
fn parity_inverse_plus() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s (^social::follows)+ e)
    )
    .collect());
}

#[test]
fn parity_alternation() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s (social::follows | social::likes) e)
    )
    .collect());
}

#[test]
fn parity_alternation_plus() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s (social::follows | social::likes)+ e)
    )
    .collect());
}

#[test]
fn parity_concat() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s social::follows social::likes e)
    )
    .collect());
}

#[test]
fn parity_concat_with_closure() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s social::follows+ social::likes e)
    )
    .collect());
}

#[test]
fn parity_optional() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s social::likes? e)
    )
    .collect());
}

#[test]
fn parity_negated_attribute() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s !social::follows e)
    )
    .collect());
}

#[test]
fn parity_negated_attribute_plus() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        path!(src.clone(), s (!social::likes)+ e)
    )
    .collect());
}

#[test]
fn parity_same_variable_selfloop() {
    let f = fixture();
    assert_parity!(f, |src| find!(
        (x: Inline<inlineencodings::GenId>),
        path!(src.clone(), x social::follows+ x)
    )
    .collect());
}

#[test]
fn parity_bound_start() {
    let f = fixture();
    let a = f.a;
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        and!(s.is(a), path!(src.clone(), s social::follows+ e))
    )
    .collect());
}

#[test]
fn parity_bound_end() {
    let f = fixture();
    let c = f.c;
    assert_parity!(f, |src| find!(
        (s: Inline<inlineencodings::GenId>, e: Inline<inlineencodings::GenId>),
        and!(e.is(c), path!(src.clone(), s social::follows+ e))
    )
    .collect());
}
