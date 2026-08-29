//! Canonical records, discovery, and semantic resolution for typed collections.
//!
//! Wire decoding and strict self-signature checks remain structural. Production
//! [resolution](crate::collection::resolution) admits only caller-authorized,
//! representation-validated claims. The larger generic oracle remains
//! test-only: it exercises algebraic laws rather than serving as another
//! runtime implementation.

use crate::id::{id_hex, Id};

/// The exact action required to contribute a signed commit to a collection.
///
/// Minted with `trible genid` on 2026-08-22. Capability policies pair this
/// stable action with one exact collection descriptor handle.
pub const ACTION_WRITE: Id = id_hex!("66B660A5481E04E552A1FA96AA9ECC48");

/// Narrow write facade for a scoped fact collection.
pub mod api;
/// Reading one collection descriptor's facts.
pub mod descriptor;
pub mod discovery;
/// Canonical collection encodings and join-preserving mappings.
pub mod encoding;
/// Shared exact-cover lifecycle for canonical derived collections.
pub mod exact_derived;
/// Explicit size-tiered maintenance for exact canonical target covers.
pub mod exact_target_compaction;
/// Maintained stated last-write-wins registers over exact source covers.
pub mod lww_register;
/// Unsigned reusable evidence for exact mapping computations.
pub mod mapping_evidence;
/// Maintained observed-set projection — the monotone half of register
/// resolution, derived and joined by the store.
pub mod observed_union;
/// How far a collection may travel, as a fragment rather than a flag.
pub mod reach;
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
/// Logical values reconstructed from typed physical covers.
pub mod view;

/// Ed25519 public key, re-exported.
///
/// Root descriptors use public keys for their identity namespace and may use
/// another as an external capability trust root. Downstream crates should not
/// have to take a direct `ed25519-dalek` dependency to name a type this API
/// demands of them.
pub use ed25519_dalek::VerifyingKey;

pub use api::*;
pub use discovery::*;
pub use encoding::*;
pub use mapping_evidence::*;
pub use records::*;
pub use resolution::*;
pub use retention::*;
pub use simplearchive_union::SimpleArchiveCollection;
pub use store::*;
pub use view::*;

#[cfg(test)]
mod oracle;
