use triblespace_core::id::RawId;
use triblespace_core::inline::RawInline;

use crate::{Automaton, GraphEdge, PathIndex, PathSummary, Step, Transition};

const VERTEX_COUNT: usize = 2;
const STATE_COUNT: usize = 2;
const PRODUCT_COUNT: usize = VERTEX_COUNT * STATE_COUNT;

fn vertex(index: usize) -> RawInline {
    [(index + 1) as u8; 32]
}

fn attribute(index: u8) -> RawId {
    [index; 16]
}

fn product(vertex: usize, state: usize) -> usize {
    vertex * STATE_COUNT + state
}

/// Decode every directed edge cell independently. Two bits select absent,
/// forward-A, reverse-B, or an unmatched X edge, so one byte enumerates all
/// 4^4 labeled graphs in this bounded family.
fn graph_from_choices(choices: u8) -> Vec<GraphEdge> {
    let mut edges = Vec::with_capacity(VERTEX_COUNT * VERTEX_COUNT);
    for source in 0..VERTEX_COUNT {
        for target in 0..VERTEX_COUNT {
            let cell = source * VERTEX_COUNT + target;
            let choice = (choices >> (cell * 2)) & 0b11;
            if choice == 0 {
                continue;
            }
            edges.push(GraphEdge {
                source: vertex(source),
                attribute: attribute(choice),
                target: vertex(target),
            });
        }
    }
    edges
}

/// A smaller symbolic family for CBMC: all subgraphs of these five edges.
/// The candidates retain both graph directions, both matched labels, an
/// unmatched label, cycles, and cross-endpoint accepted paths.
#[cfg(kani)]
fn graph_from_presence(mask: u8) -> Vec<GraphEdge> {
    let candidates = [
        GraphEdge {
            source: vertex(0),
            attribute: attribute(1),
            target: vertex(1),
        },
        GraphEdge {
            source: vertex(1),
            attribute: attribute(2),
            target: vertex(1),
        },
        GraphEdge {
            source: vertex(1),
            attribute: attribute(1),
            target: vertex(0),
        },
        GraphEdge {
            source: vertex(0),
            attribute: attribute(2),
            target: vertex(0),
        },
        GraphEdge {
            source: vertex(0),
            attribute: attribute(3),
            target: vertex(0),
        },
    ];
    let mut edges = Vec::with_capacity(candidates.len());
    for (bit, edge) in candidates.into_iter().enumerate() {
        if mask & (1 << bit) != 0 {
            edges.push(edge);
        }
    }
    edges
}

/// Independent oracle for the fixed automaton below. It deliberately does not
/// inspect `Automaton::transitions` or reuse path-summary lowering.
fn product_oracle(edges: &[GraphEdge]) -> ([[bool; VERTEX_COUNT]; VERTEX_COUNT], [bool; 2], usize) {
    let mut domain = [false; VERTEX_COUNT];
    let mut reach = [[false; PRODUCT_COUNT]; PRODUCT_COUNT];
    let mut direct_arc_count = 0;

    for (node, row) in reach.iter_mut().enumerate() {
        row[node] = true;
    }

    for edge in edges {
        let source = if edge.source == vertex(0) { 0 } else { 1 };
        let target = if edge.target == vertex(0) { 0 } else { 1 };
        domain[source] = true;
        domain[target] = true;

        if edge.attribute == attribute(1) {
            // State 0 consumes A in the graph's forward direction.
            reach[product(source, 0)][product(target, 1)] = true;
            direct_arc_count += 1;
        } else if edge.attribute == attribute(2) {
            // State 1 consumes B in the graph's reverse direction.
            reach[product(target, 1)][product(source, 0)] = true;
            direct_arc_count += 1;
        }
    }

    for intermediate in 0..PRODUCT_COUNT {
        for source in 0..PRODUCT_COUNT {
            for target in 0..PRODUCT_COUNT {
                reach[source][target] |= reach[source][intermediate] && reach[intermediate][target];
            }
        }
    }

    let mut accepted = [[false; VERTEX_COUNT]; VERTEX_COUNT];
    for source in 0..VERTEX_COUNT {
        for target in 0..VERTEX_COUNT {
            // State 0 is both initial and accepting. Restricting the reflexive
            // product closure to the supplied endpoint domain gives exactly
            // nullable identity plus positive accepted paths.
            accepted[source][target] =
                domain[source] && domain[target] && reach[product(source, 0)][product(target, 0)];
        }
    }
    (accepted, domain, direct_arc_count)
}

fn check_edges(edges: Vec<GraphEdge>) {
    let automaton = Automaton::new(
        STATE_COUNT as u32,
        [0],
        [0],
        [
            Transition::new(0, 1, Step::Forward(attribute(1))),
            Transition::new(1, 0, Step::Reverse(attribute(2))),
        ],
    )
    .expect("the fixed automaton is valid");
    let (expected, domain, expected_direct_arcs) = product_oracle(&edges);

    let summary = PathSummary::from_edges(automaton, edges);
    assert_eq!(summary.direct_arc_count(), expected_direct_arcs);
    let expected_vertex_count = domain.iter().filter(|&&present| present).count();
    assert_eq!(summary.vertices().len(), expected_vertex_count);

    let index = PathIndex::from_summary(summary).expect("the bounded carrier fits");
    let mut expected_pairs = Vec::new();
    for (source, row) in expected.iter().enumerate() {
        for (target, &is_accepted) in row.iter().enumerate() {
            assert_eq!(
                index.contains(&vertex(source), &vertex(target)),
                is_accepted
            );
            if is_accepted {
                expected_pairs.push((vertex(source), vertex(target)));
            }
        }
    }

    assert_eq!(index.vertex_count(), expected_vertex_count);
    assert_eq!(index.accepted_pair_count(), expected_pairs.len());
    assert_eq!(index.accepted_pairs().collect::<Vec<_>>(), expected_pairs);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(8)]
fn path_index_matches_two_vertex_product_oracle() {
    let mask: u8 = kani::any();
    kani::assume(mask < 32);
    check_edges(graph_from_presence(mask));
}

#[cfg(test)]
#[test]
fn all_two_vertex_graphs_match_product_oracle() {
    for choices in u8::MIN..=u8::MAX {
        check_edges(graph_from_choices(choices));
    }
}
