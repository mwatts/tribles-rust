//! Canonical records, discovery, and semantic resolution for typed collections.
//!
//! Wire decoding and strict self-signature checks remain structural. Production
//! [resolution](crate::collection::resolution) admits only caller-authorized,
//! representation-validated claims. The larger generic oracle remains
//! test-only: it exercises algebraic laws rather than serving as another
//! runtime implementation.

/// Narrow write facade for a scoped fact collection.
pub mod api;
pub mod discovery;
pub mod records;
/// Stateless semantic admission, closure, provenance, and physical-cover view.
pub mod resolution;
/// Strong retention planning for authorized collection commits.
pub mod retention;
/// Canonical `SimpleArchive` set-union collection kind.
pub mod simplearchive_union;
/// Native grow-only storage for collection-calculus records.
pub mod store;
/// Canonical raw `SuccinctArchiveBlob` set-union collection kind.
pub mod succinctarchive_union;

pub use api::*;
pub use discovery::*;
pub use records::*;
pub use resolution::*;
pub use retention::*;
pub use store::*;

#[cfg(test)]
mod oracle;
