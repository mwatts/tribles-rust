//! Exact regular-path relations kept outside the core query solver.
//!
//! A [`PathSummary`] is the unionable, constructional form: a canonical fixed
//! automaton, its complete graph-term universe, and the direct arcs of their
//! product. A [`PathIndex`] materializes one snapshot with a single algorithm:
//! SCC condensation followed by reverse-topological bitset propagation. No
//! product transitive closure survives materialization: the index retains the
//! constructional summary and the accepted endpoint relation.

mod automaton;
mod constraint;
mod index;
mod persistence;
mod summary;

pub use automaton::{Automaton, AutomatonError, StateId, Step, Transition};
pub use constraint::PathConstraint;
pub use index::PathIndex;
pub use persistence::{
    automaton_fingerprint, path_automaton_fingerprint, seg_path_summary, PathRollup,
    PathSummaryBlob, PathSummaryBlobError,
};
pub use summary::{GraphEdge, PathError, PathSummary};
