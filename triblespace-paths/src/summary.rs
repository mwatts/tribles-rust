use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use triblespace_core::id::RawId;
use triblespace_core::inline::encodings::UnknownInline;
use triblespace_core::inline::RawInline;
use triblespace_core::trible::Trible;

use crate::{Automaton, StateId};

/// One directed, attribute-labeled graph edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GraphEdge {
    /// Edge source.
    pub source: RawInline,
    /// Edge label.
    pub attribute: RawId,
    /// Edge target. It may be an ID or any other inline value.
    pub target: RawInline,
}

impl From<&Trible> for GraphEdge {
    fn from(trible: &Trible) -> Self {
        Self {
            source: RawInline::from(*trible.e()),
            attribute: RawId::from(*trible.a()),
            target: trible.v::<UnknownInline>().raw,
        }
    }
}

impl From<Trible> for GraphEdge {
    fn from(trible: Trible) -> Self {
        Self::from(&trible)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ProductNode {
    vertex: RawInline,
    state: StateId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ProductArc {
    source: ProductNode,
    target: ProductNode,
}

/// Canonical, unionable constructional summary for one fixed automaton.
///
/// Nullable automata retain both endpoints of every supplied graph edge so
/// their zero-hop identity has the correct graph domain. Non-nullable
/// automata retain only endpoints of edges matching at least one transition:
/// every positive-length path lies entirely in that support. The retained
/// arcs are direct product arcs, never a transitive closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PathSummary {
    pub(crate) automaton: Automaton,
    pub(crate) vertices: Vec<RawInline>,
    arcs: Vec<ProductArc>,
}

/// Failure to compose or materialize path data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PathError {
    /// An operation needing a defining automaton received no summaries.
    EmptyInput,
    /// Every composed summary must use the same canonical automaton.
    DifferentAutomata,
    /// Vertex ordinals no longer fit in the production `u32` representation.
    TooManyVertices {
        /// Number of distinct graph terms.
        count: usize,
    },
    /// Product ordinals no longer fit in the production `u32` representation.
    ProductCarrierTooLarge {
        /// Number of distinct graph terms.
        vertices: usize,
        /// Number of automaton states.
        states: StateId,
    },
    /// A checked allocation dimension overflowed `usize`.
    CapacityOverflow,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "cannot combine an empty path-summary list"),
            Self::DifferentAutomata => write!(f, "path summaries use different automata"),
            Self::TooManyVertices { count } => {
                write!(f, "{count} path vertices do not fit in u32 ordinals")
            }
            Self::ProductCarrierTooLarge { vertices, states } => write!(
                f,
                "the {vertices} by {states} product carrier does not fit in u32 ordinals"
            ),
            Self::CapacityOverflow => write!(f, "path-index allocation dimensions overflow usize"),
        }
    }
}

impl Error for PathError {}

impl PathSummary {
    pub(crate) fn from_canonical_ordinals(
        automaton: Automaton,
        vertices: Vec<RawInline>,
        arcs: Vec<(u32, u32)>,
    ) -> Result<Self, PathError> {
        let state_count = automaton.state_count();
        let product_count = vertices
            .len()
            .checked_mul(state_count as usize)
            .ok_or(PathError::CapacityOverflow)?;
        let mut product_arcs = Vec::with_capacity(arcs.len());
        for (source, target) in arcs {
            if source as usize >= product_count || target as usize >= product_count {
                return Err(PathError::ProductCarrierTooLarge {
                    vertices: vertices.len(),
                    states: state_count,
                });
            }
            product_arcs.push(ProductArc {
                source: ProductNode {
                    vertex: vertices[source as usize / state_count as usize],
                    state: source % state_count,
                },
                target: ProductNode {
                    vertex: vertices[target as usize / state_count as usize],
                    state: target % state_count,
                },
            });
        }
        Ok(Self {
            automaton,
            vertices,
            arcs: product_arcs,
        })
    }

    /// Lowers a SET of graph edges into canonical direct product arcs.
    pub fn from_edges(automaton: Automaton, edges: impl IntoIterator<Item = GraphEdge>) -> Self {
        let edges = edges.into_iter().collect::<BTreeSet<_>>();
        let mut vertices = if automaton.accepts_empty() {
            edges
                .iter()
                .flat_map(|edge| [edge.source, edge.target])
                .collect::<BTreeSet<_>>()
        } else {
            BTreeSet::new()
        };
        let mut arcs = BTreeSet::new();
        for edge in edges {
            for transition in automaton.transitions() {
                if !transition.step.matches(&edge.attribute) {
                    continue;
                }
                let (source, target) = if transition.step.is_reverse() {
                    (edge.target, edge.source)
                } else {
                    (edge.source, edge.target)
                };
                vertices.insert(source);
                vertices.insert(target);
                arcs.insert(ProductArc {
                    source: ProductNode {
                        vertex: source,
                        state: transition.from,
                    },
                    target: ProductNode {
                        vertex: target,
                        state: transition.to,
                    },
                });
            }
        }
        Self {
            automaton,
            vertices: vertices.into_iter().collect(),
            arcs: arcs.into_iter().collect(),
        }
    }

    /// Lowers tribles directly into a summary.
    pub fn from_tribles<'a>(
        automaton: Automaton,
        tribles: impl IntoIterator<Item = &'a Trible>,
    ) -> Self {
        Self::from_edges(automaton, tribles.into_iter().map(GraphEdge::from))
    }

    /// Fixed automaton defining this summary.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// Sorted graph-term domain required by this fixed automaton.
    pub fn vertices(&self) -> &[RawInline] {
        &self.vertices
    }

    /// Number of canonical direct product arcs.
    pub fn direct_arc_count(&self) -> usize {
        self.arcs.len()
    }

    /// Canonical set union of two summaries.
    pub fn merge(&self, other: &Self) -> Result<Self, PathError> {
        Self::merge_all([self, other])
    }

    /// Canonical set union of any nonempty collection of summaries.
    pub fn merge_all<'a>(summaries: impl IntoIterator<Item = &'a Self>) -> Result<Self, PathError> {
        let summaries = summaries.into_iter().collect::<Vec<_>>();
        let Some(first) = summaries.first() else {
            return Err(PathError::EmptyInput);
        };
        if summaries
            .iter()
            .any(|summary| summary.automaton != first.automaton)
        {
            return Err(PathError::DifferentAutomata);
        }
        let vertices = summaries
            .iter()
            .flat_map(|summary| summary.vertices.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let arcs = summaries
            .iter()
            .flat_map(|summary| summary.arcs.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            automaton: first.automaton.clone(),
            vertices,
            arcs,
        })
    }

    pub(crate) fn ordinal_arcs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.ordinal_arcs_in(&self.vertices)
    }

    pub(crate) fn active_vertices(&self) -> Vec<RawInline> {
        self.arcs
            .iter()
            .flat_map(|arc| [arc.source.vertex, arc.target.vertex])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub(crate) fn has_canonical_domain(&self) -> bool {
        self.automaton.accepts_empty() || self.vertices == self.active_vertices()
    }

    pub(crate) fn ordinal_arcs_in<'a>(
        &'a self,
        vertices: &'a [RawInline],
    ) -> impl Iterator<Item = (u32, u32)> + 'a {
        let states = self.automaton.state_count();
        self.arcs.iter().map(move |arc| {
            let source_vertex = vertices
                .binary_search(&arc.source.vertex)
                .expect("summary arc source belongs to the requested product carrier")
                as u32;
            let target_vertex = vertices
                .binary_search(&arc.target.vertex)
                .expect("summary arc target belongs to the requested product carrier")
                as u32;
            (
                source_vertex * states + arc.source.state,
                target_vertex * states + arc.target.state,
            )
        })
    }
}
