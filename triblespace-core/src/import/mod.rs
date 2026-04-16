//! Data import and conversion helpers bridging external formats into Trible Space.
//!
//! This module hosts adapters that translate common interchange formats into
//! [`TribleSet`](crate::trible::TribleSet) changes ready to merge into a
//! repository or workspace.

mod import_attribute;
pub mod json;
pub mod json_tree;
pub mod ntriples;

pub(crate) use import_attribute::ImportAttribute;

use triblespace_core_macros::attributes;

use crate::blob::schemas::longstring::LongString;
use crate::value::schemas::hash::{Blake3, Handle};

attributes! {
    /// The canonical RDF URI for an entity. Use this when importing data
    /// from an external vocabulary where the entity's identity is a URI —
    /// the same URI always deterministically maps to the same triblespace
    /// Id by round-tripping through an `rdf_uri` fragment.
    "AA68DE115445A63D62A63FF3284D030C" as pub rdf_uri: Handle<Blake3, LongString>;
}
