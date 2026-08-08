//! Typed facts shared by the range-native search index recipes.
//!
//! Every stable attribute identifier in this module was minted with
//! `trible genid`. Rotation dates are recorded at the declarations that have
//! changed. Artifact attributes intentionally carry their exact blob encodings:
//! search manifests never erase these handles to `UnknownBlob`.

use triblespace_core::inline::encodings::genid::GenId;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::encodings::iu256::U256BE;
use triblespace_core::prelude::attributes;

use crate::portable_bm25::PortableBM25Blob;
use crate::succinct::SuccinctHNSWBlob;

attributes! {
    /// One portable canonical exact-TF BM25 artifact. Rotated on 2026-08-08
    /// when range persistence moved off the native succinct accelerator.
    "570272A9F9C994D2152EFB10712F5275" unsafe as pub seg_bm25: Handle<PortableBM25Blob>;
    /// One physical succinct HNSW artifact.
    "54B0D283B85698E875A8A270E2570CF7" unsafe as pub seg_hnsw: Handle<SuccinctHNSWBlob>;
    /// Source attribute projected by a search recipe.
    "38FA73632BEF15C5D125AA4A8E168D84" unsafe as pub index_source_attribute: GenId;
    /// Vector dimension of an HNSW recipe.
    "45818F54828F1EAEC1FB8E34C8E290EB" unsafe as pub index_dimension: U256BE;
    /// Deterministic graph-construction seed of an HNSW recipe.
    "7E03D090721DF88BD001AA3ACCCA7256" unsafe as pub index_seed: U256BE;
}
