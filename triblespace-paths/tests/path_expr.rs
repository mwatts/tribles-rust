use std::collections::BTreeSet;

use triblespace_core::inline::RawInline;
use triblespace_paths::{automaton_fingerprint, GraphEdge, PathExpr, PathIndex, Step};

fn vertex(byte: u8) -> RawInline {
    [byte; 32]
}

fn attribute(byte: u8) -> [u8; 16] {
    [byte; 16]
}

fn edge(source: u8, label: u8, target: u8) -> GraphEdge {
    GraphEdge {
        source: vertex(source),
        attribute: attribute(label),
        target: vertex(target),
    }
}

#[test]
fn public_expression_api_materializes_compound_paths() {
    let expression = PathExpr::from(Step::Forward(attribute(1)))
        .then(PathExpr::from(Step::Forward(attribute(2))).optional())
        .or(PathExpr::from(Step::Forward(attribute(3))).inverse().plus());
    let index = PathIndex::from_edges(
        expression.compile(),
        [edge(1, 1, 2), edge(2, 2, 3), edge(4, 3, 3), edge(5, 3, 4)],
    )
    .unwrap();

    assert_eq!(
        index.accepted_pairs().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            (vertex(1), vertex(2)),
            (vertex(1), vertex(3)),
            (vertex(3), vertex(4)),
            (vertex(3), vertex(5)),
            (vertex(4), vertex(5)),
        ])
    );
}

#[test]
fn canonical_expression_construction_stabilizes_automaton_fingerprints() {
    let first: PathExpr = Step::Forward(attribute(1)).into();
    let second: PathExpr = Step::ForwardExcept(vec![attribute(3), attribute(2)]).into();
    let left = first.clone().or(second.clone()).or(first).compile();
    let right = PathExpr::from(Step::ForwardExcept(vec![
        attribute(2),
        attribute(3),
        attribute(2),
    ]))
    .or(PathExpr::from(Step::Forward(attribute(1))))
    .compile();

    assert_eq!(left, right);
    assert_eq!(automaton_fingerprint(&left), automaton_fingerprint(&right));
}
