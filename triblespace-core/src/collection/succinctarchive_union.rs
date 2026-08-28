//! Canonical raw SuccinctArchive set union and its derivation from
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive).
//!
//! The SimpleArchive and raw SuccinctArchive collections use the same
//! [`TRIBLE_SET_UNION_RECIPE_V1`](super::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1):
//! the recipe names the set-union law, while the collection descriptor's
//! representation field distinguishes their bytes. The canonical conversion
//! is therefore a join homomorphism:
//!
//! ```text
//! succinct(a ∪ b) = succinct(a) ∪ succinct(b)
//! ```
//!
//! This module validates exactly those `DERIVE` and `MERGE` equations. It does
//! not authorize commits, select semantic roots, retain artifacts, or assign
//! authority to construction records. `DERIVE` and `MERGE` remain unsigned,
//! reproducible evidence.

use super::simplearchive_union::TribleSetUnionV1;
use crate::id::ExclusiveId;
// Reach arrives here as a builder argument; only the tests name a
// particular one.
use super::descriptor as descriptor_facts;
use super::records::{
    collection_authority, collection_reach, collection_recipe, collection_representation,
    collection_source, RecordDecodeError, KIND_COLLECTION_DESCRIPTOR,
};
#[cfg(test)]
use crate::collection::reach;
use crate::metadata;
use crate::prelude::entity;
use crate::trible::Fragment;
use ed25519_dalek::VerifyingKey;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    SuccinctArchiveBlob, SuccinctArchiveRawBuildError, SuccinctArchiveRawMergeError,
};
use crate::blob::{Blob, BlobEncoding};
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::hash::{Blake3, Hash};
use crate::inline::Inline;
use crate::metadata::MetaDescribe;

use super::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1;
use super::{CollectionData, CollectionDerive, CollectionHandle, CollectionMerge};

mod collection;
mod rank9_fiber;
pub use collection::*;
pub use rank9_fiber::Rank9FiberError;

/// Lifted Rank9-union recipe for 32-bit little-endian targets.
///
/// Minted with `trible genid` on 2026-08-14. The profile pins the current portable
/// SuccinctArchive source schema, detached Rank9 format
/// marker/version/flags, the canonical Rank9 builder and Jerky serialization
/// epoch, pointer width, and byte order. Any change that can alter canonical
/// sidecar bytes requires a newly minted recipe id.
pub const RANK9_LIFTED_UNION_RECIPE_V1_32_LE: Id = id_hex!("0685616E15F332468977EB59BDA4EB9D");

/// Lifted Rank9-union recipe for 32-bit big-endian targets.
///
/// Minted with `trible genid` on 2026-08-14; see
/// [`RANK9_LIFTED_UNION_RECIPE_V1_32_LE`] for the versioning contract.
pub const RANK9_LIFTED_UNION_RECIPE_V1_32_BE: Id = id_hex!("154A792188583355B1CDAA9910E60748");

/// Lifted Rank9-union recipe for 64-bit little-endian targets.
///
/// Minted with `trible genid` on 2026-08-14; see
/// [`RANK9_LIFTED_UNION_RECIPE_V1_32_LE`] for the versioning contract.
pub const RANK9_LIFTED_UNION_RECIPE_V1_64_LE: Id = id_hex!("E4A77808BBF9E373244789F007E81261");

/// Lifted Rank9-union recipe for 64-bit big-endian targets.
///
/// Minted with `trible genid` on 2026-08-14; see
/// [`RANK9_LIFTED_UNION_RECIPE_V1_32_LE`] for the versioning contract.
pub const RANK9_LIFTED_UNION_RECIPE_V1_64_BE: Id = id_hex!("A470EAEB76777091CE795D9B108C79D0");

/// The lifted Rank9-union law for this ABI, as a describable type.
pub struct Rank9LiftedUnionV1_32Le;

impl MetaDescribe for Rank9LiftedUnionV1_32Le {
    fn describe() -> Fragment {
        let id: Id = RANK9_LIFTED_UNION_RECIPE_V1_32_LE;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "rank9-lifted-union-v1-32-le",
                metadata::description: "Set union of a SuccinctArchive collection lifted through its detached Rank9 sidecar. Same associative, commutative, idempotent law as trible-set-union, applied to the archive's canonical bytes rather than to loose tribles, so merging two indexed states yields the index of their union. The recipe pins everything that can change canonical sidecar bytes: the portable SuccinctArchive source schema, the Rank9 format marker, version and flags, the Rank9 builder and Jerky serialization epoch, and the target's 32-bit pointer width and little-endian byte order. A different width or byte order is a different law, because it yields different canonical bytes.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

/// The lifted Rank9-union law for this ABI, as a describable type.
pub struct Rank9LiftedUnionV1_32Be;

impl MetaDescribe for Rank9LiftedUnionV1_32Be {
    fn describe() -> Fragment {
        let id: Id = RANK9_LIFTED_UNION_RECIPE_V1_32_BE;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "rank9-lifted-union-v1-32-be",
                metadata::description: "Set union of a SuccinctArchive collection lifted through its detached Rank9 sidecar. Same associative, commutative, idempotent law as trible-set-union, applied to the archive's canonical bytes rather than to loose tribles, so merging two indexed states yields the index of their union. The recipe pins everything that can change canonical sidecar bytes: the portable SuccinctArchive source schema, the Rank9 format marker, version and flags, the Rank9 builder and Jerky serialization epoch, and the target's 32-bit pointer width and big-endian byte order. A different width or byte order is a different law, because it yields different canonical bytes.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

/// The lifted Rank9-union law for this ABI, as a describable type.
pub struct Rank9LiftedUnionV1_64Le;

impl MetaDescribe for Rank9LiftedUnionV1_64Le {
    fn describe() -> Fragment {
        let id: Id = RANK9_LIFTED_UNION_RECIPE_V1_64_LE;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "rank9-lifted-union-v1-64-le",
                metadata::description: "Set union of a SuccinctArchive collection lifted through its detached Rank9 sidecar. Same associative, commutative, idempotent law as trible-set-union, applied to the archive's canonical bytes rather than to loose tribles, so merging two indexed states yields the index of their union. The recipe pins everything that can change canonical sidecar bytes: the portable SuccinctArchive source schema, the Rank9 format marker, version and flags, the Rank9 builder and Jerky serialization epoch, and the target's 64-bit pointer width and little-endian byte order. A different width or byte order is a different law, because it yields different canonical bytes.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

/// The lifted Rank9-union law for this ABI, as a describable type.
pub struct Rank9LiftedUnionV1_64Be;

impl MetaDescribe for Rank9LiftedUnionV1_64Be {
    fn describe() -> Fragment {
        let id: Id = RANK9_LIFTED_UNION_RECIPE_V1_64_BE;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "rank9-lifted-union-v1-64-be",
                metadata::description: "Set union of a SuccinctArchive collection lifted through its detached Rank9 sidecar. Same associative, commutative, idempotent law as trible-set-union, applied to the archive's canonical bytes rather than to loose tribles, so merging two indexed states yields the index of their union. The recipe pins everything that can change canonical sidecar bytes: the portable SuccinctArchive source schema, the Rank9 format marker, version and flags, the Rank9 builder and Jerky serialization epoch, and the target's 64-bit pointer width and big-endian byte order. A different width or byte order is a different law, because it yields different canonical bytes.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

#[cfg(all(target_pointer_width = "32", target_endian = "little"))]
const CURRENT_RANK9_LIFTED_UNION_RECIPE: Id = RANK9_LIFTED_UNION_RECIPE_V1_32_LE;
#[cfg(all(target_pointer_width = "32", target_endian = "big"))]
const CURRENT_RANK9_LIFTED_UNION_RECIPE: Id = RANK9_LIFTED_UNION_RECIPE_V1_32_BE;
#[cfg(all(target_pointer_width = "64", target_endian = "little"))]
const CURRENT_RANK9_LIFTED_UNION_RECIPE: Id = RANK9_LIFTED_UNION_RECIPE_V1_64_LE;
#[cfg(all(target_pointer_width = "64", target_endian = "big"))]
const CURRENT_RANK9_LIFTED_UNION_RECIPE: Id = RANK9_LIFTED_UNION_RECIPE_V1_64_BE;

/// Recipe id for the exact Rank9 ABI supported by this build.
pub const fn current_rank9_lifted_union_recipe() -> Id {
    CURRENT_RANK9_LIFTED_UNION_RECIPE
}

/// A collection descriptor participating in a validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DescriptorRole {
    /// Canonical SimpleArchive source of a derivation.
    Source,
    /// Canonical raw SuccinctArchive target or merge collection.
    Target,
}

impl fmt::Display for DescriptorRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source => formatter.write_str("source"),
            Self::Target => formatter.write_str("target"),
        }
    }
}

/// A collection element participating in a validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementRole {
    /// SimpleArchive input of a derivation.
    DeriveInput,
    /// Raw SuccinctArchive output of a derivation.
    DeriveOutput,
    /// Canonically lower raw SuccinctArchive merge input.
    MergeLow,
    /// Canonically higher raw SuccinctArchive merge input.
    MergeHigh,
    /// Claimed raw SuccinctArchive merge output.
    MergeResult,
}

impl fmt::Display for ElementRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeriveInput => formatter.write_str("derive input"),
            Self::DeriveOutput => formatter.write_str("derive output"),
            Self::MergeLow => formatter.write_str("merge low input"),
            Self::MergeHigh => formatter.write_str("merge high input"),
            Self::MergeResult => formatter.write_str("merge result"),
        }
    }
}

/// Failure to validate the canonical raw SuccinctArchive collection law.
#[derive(Debug)]
pub enum SuccinctArchiveUnionValidationError {
    /// The target does not derive from the source it was checked against.
    WrongSource {
        /// The source descriptor's handle.
        expected: CollectionHandle,
        /// What the target actually names, if anything.
        actual: Option<CollectionHandle>,
    },
    /// The descriptor does not carry a field this check needs.
    Malformed(RecordDecodeError),
    /// A descriptor names another blob representation.
    WrongRepresentation {
        /// Descriptor being checked.
        role: DescriptorRole,
        /// Required representation descriptor.
        expected: Id,
        /// Representation found in the descriptor.
        actual: Id,
    },
    /// A descriptor names another semantic recipe.
    WrongRecipe {
        /// Descriptor being checked.
        role: DescriptorRole,
        /// Required union recipe.
        expected: Id,
        /// Recipe found in the descriptor.
        actual: Id,
    },
    /// A record names another collection descriptor.
    WrongCollection {
        /// Record endpoint being checked.
        role: DescriptorRole,
        /// Descriptor required at this endpoint.
        expected: CollectionHandle,
        /// Descriptor named by the record.
        actual: CollectionHandle,
    },
    /// Supplied bytes do not have the content identity named by the record.
    EndpointMismatch {
        /// Endpoint being checked.
        role: ElementRole,
        /// Identity named by the record.
        expected: CollectionData,
        /// Fresh identity computed from the supplied bytes.
        actual: CollectionData,
    },
    /// The SimpleArchive source could not be canonically converted.
    SourceBuild(SuccinctArchiveRawBuildError),
    /// Raw merge-input validation or canonical union construction failed.
    RawMerge(SuccinctArchiveRawMergeError),
    /// The claimed derivation is not the canonical conversion of its source.
    WrongDeriveOutput,
    /// The claimed merge result is not the canonical union of its inputs.
    WrongMergeResult,
}

impl fmt::Display for SuccinctArchiveUnionValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongSource { expected, actual } => match actual {
                Some(actual) => write!(
                    formatter,
                    "collection derives from {actual:?}, not {expected:?}"
                ),
                None => write!(
                    formatter,
                    "collection is a root, expected a derivation of {expected:?}"
                ),
            },
            Self::Malformed(error) => {
                write!(formatter, "malformed collection descriptor: {error}")
            }
            Self::WrongRepresentation {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} collection representation {actual:X} does not match {expected:X}"
            ),
            Self::WrongRecipe {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} collection recipe {actual:X} does not match TribleSet union {expected:X}"
            ),
            Self::WrongCollection {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "record {role} collection {} does not match descriptor {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::EndpointMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} handle {} does not match claimed {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::SourceBuild(source) => {
                write!(formatter, "cannot derive raw SuccinctArchive: {source}")
            }
            Self::RawMerge(source) => {
                write!(
                    formatter,
                    "cannot validate and merge raw SuccinctArchives: {source}"
                )
            }
            Self::WrongDeriveOutput => formatter
                .write_str("derive output is not the canonical raw SuccinctArchive of its input"),
            Self::WrongMergeResult => formatter.write_str(
                "merge result is not the exact canonical union of its raw SuccinctArchive inputs",
            ),
        }
    }
}

impl Error for SuccinctArchiveUnionValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SourceBuild(source) => Some(source),
            Self::RawMerge(source) => Some(source),
            _ => None,
        }
    }
}

/// Describe the raw SuccinctArchive collection derived from one source.
///
/// A derivation is anchored by the collection it is computed from, so this
/// takes that source's handle and carries no name of its own. Authority is
/// mandatory and local: the target names its own trust root rather than
/// inheriting one from the source.
///
/// This intentionally reuses the SimpleArchive collection's set-union recipe.
/// Representation, not recipe proliferation, distinguishes the two lattices.
pub fn descriptor(source: CollectionHandle, authority: VerifyingKey, reach: Fragment) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_source: source,
        collection_authority: authority,
        collection_representation*: <SuccinctArchiveBlob as MetaDescribe>::describe(),
        collection_recipe*: <TribleSetUnionV1 as MetaDescribe>::describe(),
        collection_reach*: reach,
    }
}

/// Return the canonical empty raw SuccinctArchive artifact.
pub fn empty() -> Blob<SuccinctArchiveBlob> {
    SuccinctArchiveBlob::merge(&[])
        .expect("the fixed empty raw SuccinctArchive construction cannot fail")
}

/// Canonically derive one raw SuccinctArchive element from a SimpleArchive.
pub fn derive_element(
    source: &Blob<SimpleArchive>,
) -> Result<Blob<SuccinctArchiveBlob>, SuccinctArchiveRawBuildError> {
    SuccinctArchiveBlob::build_from_simple_archive(source)
}

/// Compute the canonical union of two raw SuccinctArchive elements.
pub fn join(
    left: &Blob<SuccinctArchiveBlob>,
    right: &Blob<SuccinctArchiveBlob>,
) -> Result<Blob<SuccinctArchiveBlob>, SuccinctArchiveRawMergeError> {
    SuccinctArchiveBlob::merge(&[left.clone(), right.clone()])
}

/// Validate an exact canonical `SimpleArchive -> SuccinctArchiveBlob` mapping.
///
/// This checks both descriptors, requires the target to name this exact source
/// by handle, binds the record and supplied endpoint bytes in both directions,
/// validates the target's portable format, and compares it byte-for-byte with a
/// fresh direct construction from the source.
pub fn validate_derive(
    source_descriptor: &Fragment,
    target_descriptor: &Fragment,
    claim: &CollectionDerive,
    input: &Blob<SimpleArchive>,
    output: &Blob<SuccinctArchiveBlob>,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_source_descriptor(source_descriptor)?;
    validate_descriptor(target_descriptor)?;
    let source_collection: CollectionHandle =
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(source_descriptor.facts().clone())
            .get_handle();
    let target_collection: CollectionHandle =
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(target_descriptor.facts().clone())
            .get_handle();
    // The target names its source by handle, so this checks the lineage
    // itself rather than a label both sides could independently claim.
    let actual_source = descriptor_facts::source(target_descriptor.facts())?;
    match actual_source {
        Some(source) if source == source_collection => {}
        _ => {
            return Err(SuccinctArchiveUnionValidationError::WrongSource {
                expected: source_collection,
                actual: actual_source,
            })
        }
    }
    validate_collection(DescriptorRole::Target, target_collection, claim.target())?;

    let (expected_input, expected_output) = claim.mapping();
    validate_endpoint(ElementRole::DeriveInput, expected_input, input)?;
    validate_endpoint(ElementRole::DeriveOutput, expected_output, output)?;
    let expected =
        derive_element(input).map_err(SuccinctArchiveUnionValidationError::SourceBuild)?;
    if output.bytes != expected.bytes {
        return Err(SuccinctArchiveUnionValidationError::WrongDeriveOutput);
    }
    Ok(())
}

/// Validate an exact canonical raw SuccinctArchive union equation.
///
/// All three endpoint identities are recomputed from bytes. The raw merge
/// exact-validates both inputs while constructing their canonical union;
/// byte-for-byte equality with that union proves the claimed result canonical.
pub fn validate_merge(
    descriptor: &Fragment,
    claim: &CollectionMerge,
    low: &Blob<SuccinctArchiveBlob>,
    high: &Blob<SuccinctArchiveBlob>,
    result: &Blob<SuccinctArchiveBlob>,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_descriptor(descriptor)?;
    let collection: CollectionHandle =
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle();
    validate_collection(DescriptorRole::Target, collection, claim.collection())?;

    let (expected_low, expected_high) = claim.inputs();
    validate_endpoint(ElementRole::MergeLow, expected_low, low)?;
    validate_endpoint(ElementRole::MergeHigh, expected_high, high)?;
    validate_endpoint(ElementRole::MergeResult, claim.result(), result)?;

    let expected = join(low, high).map_err(SuccinctArchiveUnionValidationError::RawMerge)?;
    if result.bytes != expected.bytes {
        return Err(SuccinctArchiveUnionValidationError::WrongMergeResult);
    }
    Ok(())
}

fn validate_source_descriptor(
    descriptor: &Fragment,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_descriptor_parts(
        DescriptorRole::Source,
        descriptor,
        <SimpleArchive as MetaDescribe>::id(),
    )
}

fn validate_descriptor(descriptor: &Fragment) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_descriptor_parts(
        DescriptorRole::Target,
        descriptor,
        <SuccinctArchiveBlob as MetaDescribe>::id(),
    )
}

fn validate_descriptor_parts(
    role: DescriptorRole,
    descriptor: &Fragment,
    expected_representation: Id,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    descriptor_facts::authority(descriptor.facts())?;
    let representation = descriptor_facts::representation(descriptor.facts())?;
    if representation != expected_representation {
        return Err(SuccinctArchiveUnionValidationError::WrongRepresentation {
            role,
            expected: expected_representation,
            actual: representation,
        });
    }
    let recipe = descriptor_facts::recipe(descriptor.facts())?;
    if recipe != TRIBLE_SET_UNION_RECIPE_V1 {
        return Err(SuccinctArchiveUnionValidationError::WrongRecipe {
            role,
            expected: TRIBLE_SET_UNION_RECIPE_V1,
            actual: recipe,
        });
    }
    Ok(())
}

fn validate_collection(
    role: DescriptorRole,
    expected: CollectionHandle,
    actual: CollectionHandle,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    if actual != expected {
        return Err(SuccinctArchiveUnionValidationError::WrongCollection {
            role,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_endpoint<S: BlobEncoding>(
    role: ElementRole,
    expected: CollectionData,
    blob: &Blob<S>,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    let actual = data_identity(blob);
    if actual != expected {
        return Err(SuccinctArchiveUnionValidationError::EndpointMismatch {
            role,
            expected,
            actual,
        });
    }
    Ok(())
}

fn data_identity<S: BlobEncoding>(blob: &Blob<S>) -> CollectionData {
    Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    use anybytes::Bytes;

    use ed25519_dalek::SigningKey;

    use crate::blob::IntoBlob;
    use crate::collection::descriptor::identity_for_tests;
    use crate::collection::simplearchive_union;
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    fn authority() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    /// The named `SimpleArchive` root these tests derive from.
    fn raw_root(name: &str) -> Fragment {
        simplearchive_union::descriptor(name, authority(), reach::private())
    }

    fn row(entity: u8, attribute: u8, value: u8) -> [u8; TRIBLE_LEN] {
        let mut row = [value; TRIBLE_LEN];
        row[..16].fill(entity);
        row[16..32].fill(attribute);
        row
    }

    fn archive(rows: impl IntoIterator<Item = [u8; TRIBLE_LEN]>) -> Blob<SimpleArchive> {
        let mut facts = TribleSet::new();
        for row in rows {
            facts.insert(&Trible::force_raw(row).unwrap());
        }
        facts.to_blob()
    }

    fn ordered<'a, S: BlobEncoding>(
        left: &'a Blob<S>,
        right: &'a Blob<S>,
    ) -> (&'a Blob<S>, &'a Blob<S>) {
        if data_identity(left) <= data_identity(right) {
            (left, right)
        } else {
            (right, left)
        }
    }

    /// The indexed collection is the raw one's derivation: same law, different
    /// representation, and it says so by naming its source rather than by
    /// sharing a label with it.
    #[test]
    fn the_index_derives_from_the_raw_collection_under_the_same_law() {
        let source = raw_root("first");
        let target = descriptor(identity_for_tests(&source), authority(), reach::private());

        // The target points at exactly this source, and carries no anchor of
        // its own: what it derives from is what anchors it.
        assert_eq!(
            crate::collection::descriptor::source(target.facts()),
            Ok(Some(identity_for_tests(&source)))
        );
        assert!(
            crate::collection::descriptor::name(target.facts())
                .unwrap()
                .is_none(),
            "a derivation needs no anchor"
        );
        assert!(
            crate::collection::descriptor::source(source.facts())
                .unwrap()
                .is_none(),
            "the raw collection is a root"
        );
        // A derivation of the same shape over different data is a different
        // collection, because its source is.
        assert_ne!(
            identity_for_tests(&target),
            identity_for_tests(&descriptor(
                identity_for_tests(&raw_root("second")),
                authority(),
                reach::private()
            ))
        );

        assert_eq!(
            crate::collection::descriptor::recipe(source.facts()),
            crate::collection::descriptor::recipe(target.facts())
        );
        assert_eq!(
            crate::collection::descriptor::recipe(target.facts()).unwrap(),
            TRIBLE_SET_UNION_RECIPE_V1
        );
        assert_eq!(
            crate::collection::descriptor::representation(source.facts()).unwrap(),
            <SimpleArchive as MetaDescribe>::id()
        );
        assert_eq!(
            crate::collection::descriptor::representation(target.facts()).unwrap(),
            <SuccinctArchiveBlob as MetaDescribe>::id()
        );
        assert_ne!(identity_for_tests(&source), identity_for_tests(&target));
    }

    #[test]
    fn canonical_empty_is_the_derived_bottom_and_merge_identity() {
        let source_descriptor = raw_root("first");
        let target_descriptor = descriptor(
            identity_for_tests(&raw_root("first")),
            authority(),
            reach::private(),
        );
        let source_empty: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let derived_empty = derive_element(&source_empty).unwrap();
        let canonical_empty = empty();

        assert_eq!(derived_empty.bytes, canonical_empty.bytes);
        assert_eq!(derived_empty.get_handle(), canonical_empty.get_handle());

        let derive = CollectionDerive::new(
            identity_for_tests(&target_descriptor),
            data_identity(&source_empty),
            data_identity(&canonical_empty),
        );
        validate_derive(
            &source_descriptor,
            &target_descriptor,
            &derive,
            &source_empty,
            &canonical_empty,
        )
        .unwrap();

        let element_source = archive([row(1, 9, 3)]);
        let element = derive_element(&element_source).unwrap();
        let joined = join(&canonical_empty, &element).unwrap();
        assert_eq!(joined.bytes, element.bytes);
        assert_eq!(joined.get_handle(), element.get_handle());

        let (low, high) = ordered(&canonical_empty, &element);
        let merge = CollectionMerge::new(
            identity_for_tests(&target_descriptor),
            data_identity(low),
            data_identity(high),
            data_identity(&joined),
        );
        validate_merge(&target_descriptor, &merge, low, high, &joined).unwrap();
    }

    #[test]
    fn derive_and_merge_commute_to_identical_canonical_bytes() {
        let source_descriptor = raw_root("first");
        let target_descriptor = descriptor(
            identity_for_tests(&raw_root("first")),
            authority(),
            reach::private(),
        );
        let shared = row(3, 10, 40);
        let left = archive([row(2, 10, 60), shared]);
        let right = archive([row(1, 10, 20), shared]);

        let source_union = simplearchive_union::join(&left, &right).unwrap();
        let derive_after_merge = derive_element(&source_union).unwrap();
        let derived_left = derive_element(&left).unwrap();
        let derived_right = derive_element(&right).unwrap();
        let merge_after_derive = join(&derived_left, &derived_right).unwrap();

        assert_eq!(derive_after_merge.bytes, merge_after_derive.bytes);
        assert_eq!(
            derive_after_merge.get_handle(),
            merge_after_derive.get_handle()
        );

        for (input, output) in [
            (&left, &derived_left),
            (&right, &derived_right),
            (&source_union, &derive_after_merge),
        ] {
            let claim = CollectionDerive::new(
                identity_for_tests(&target_descriptor),
                data_identity(input),
                data_identity(output),
            );
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &claim,
                input,
                output,
            )
            .unwrap();
        }

        let (low, high) = ordered(&derived_left, &derived_right);
        let merge = CollectionMerge::new(
            identity_for_tests(&target_descriptor),
            data_identity(low),
            data_identity(high),
            data_identity(&merge_after_derive),
        );
        validate_merge(&target_descriptor, &merge, low, high, &merge_after_derive).unwrap();
    }

    #[test]
    fn validators_reject_valid_but_wrong_canonical_outputs() {
        let source_descriptor = raw_root("first");
        let target_descriptor = descriptor(
            identity_for_tests(&raw_root("first")),
            authority(),
            reach::private(),
        );
        let input = archive([row(1, 9, 3)]);
        let wrong_source = archive([row(2, 9, 4)]);
        let wrong_output = derive_element(&wrong_source).unwrap();
        let claim = CollectionDerive::new(
            identity_for_tests(&target_descriptor),
            data_identity(&input),
            data_identity(&wrong_output),
        );

        assert!(matches!(
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &claim,
                &input,
                &wrong_output,
            ),
            Err(SuccinctArchiveUnionValidationError::WrongDeriveOutput)
        ));

        let left = derive_element(&input).unwrap();
        let right = derive_element(&wrong_source).unwrap();
        let correct = join(&left, &right).unwrap();
        let wrong = empty();
        let (low, high) = ordered(&left, &right);
        let merge = CollectionMerge::new(
            identity_for_tests(&target_descriptor),
            data_identity(low),
            data_identity(high),
            data_identity(&wrong),
        );
        assert_ne!(correct.bytes, wrong.bytes);
        assert!(matches!(
            validate_merge(&target_descriptor, &merge, low, high, &wrong),
            Err(SuccinctArchiveUnionValidationError::WrongMergeResult)
        ));
    }

    #[test]
    fn malformed_target_is_rejected_before_equation_admission() {
        let source_descriptor = raw_root("first");
        let target_descriptor = descriptor(
            identity_for_tests(&raw_root("first")),
            authority(),
            reach::private(),
        );
        let input = archive([row(1, 9, 3)]);
        let malformed = Blob::<SuccinctArchiveBlob>::new(Bytes::from(vec![0xAA; 17]));
        let claim = CollectionDerive::new(
            identity_for_tests(&target_descriptor),
            data_identity(&input),
            data_identity(&malformed),
        );

        assert!(matches!(
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &claim,
                &input,
                &malformed,
            ),
            Err(SuccinctArchiveUnionValidationError::WrongDeriveOutput)
        ));
    }
}

impl From<RecordDecodeError> for SuccinctArchiveUnionValidationError {
    fn from(error: RecordDecodeError) -> Self {
        Self::Malformed(error)
    }
}
