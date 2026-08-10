//! Canonical raw SuccinctArchive set union and its derivation from
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive).
//!
//! The SimpleArchive and raw SuccinctArchive collections use the same
//! [`TRIBLE_SET_UNION_RECIPE_V1`](super::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1):
//! the recipe names the set-union law, while the collection definition's
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

use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    SuccinctArchiveBlob, SuccinctArchiveError, SuccinctArchiveRawBuildError,
};
use crate::blob::{Blob, BlobEncoding};
use crate::id::Id;
use crate::inline::encodings::hash::{Blake3, Hash};
use crate::inline::Inline;
use crate::metadata::MetaDescribe;

use super::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1;
use super::{CollectionData, CollectionDefinition, CollectionDerive, CollectionMerge};

/// A collection definition participating in a validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DefinitionRole {
    /// Canonical SimpleArchive source of a derivation.
    Source,
    /// Canonical raw SuccinctArchive target or merge collection.
    Target,
}

impl fmt::Display for DefinitionRole {
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
    /// A definition names another blob representation.
    WrongRepresentation {
        /// Definition being checked.
        role: DefinitionRole,
        /// Required representation descriptor.
        expected: Id,
        /// Representation found in the definition.
        actual: Id,
    },
    /// A definition names another semantic recipe.
    WrongRecipe {
        /// Definition being checked.
        role: DefinitionRole,
        /// Required union recipe.
        expected: Id,
        /// Recipe found in the definition.
        actual: Id,
    },
    /// A derivation crosses dataset scopes.
    ScopeMismatch {
        /// Source dataset scope.
        source: Id,
        /// Target dataset scope.
        target: Id,
    },
    /// A record names another collection definition.
    WrongCollection {
        /// Record endpoint being checked.
        role: DefinitionRole,
        /// Definition required at this endpoint.
        expected: Id,
        /// Definition named by the record.
        actual: Id,
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
    RawMerge(SuccinctArchiveError),
    /// The claimed derivation is not the canonical conversion of its source.
    WrongDeriveOutput,
    /// The claimed merge result is not the canonical union of its inputs.
    WrongMergeResult,
}

impl fmt::Display for SuccinctArchiveUnionValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::ScopeMismatch { source, target } => write!(
                formatter,
                "derive source scope {source:X} does not match target scope {target:X}"
            ),
            Self::WrongCollection {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "record {role} collection {actual:X} does not match definition {expected:X}"
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

/// Construct the raw SuccinctArchive collection for an extrinsic dataset scope.
///
/// This intentionally reuses the SimpleArchive collection's set-union recipe.
/// Representation, not recipe proliferation, distinguishes the two lattices.
pub fn definition(scope: Id) -> CollectionDefinition {
    CollectionDefinition::new(
        scope,
        <SuccinctArchiveBlob as MetaDescribe>::id(),
        TRIBLE_SET_UNION_RECIPE_V1,
    )
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
) -> Result<Blob<SuccinctArchiveBlob>, SuccinctArchiveError> {
    SuccinctArchiveBlob::merge(&[left.clone(), right.clone()])
}

/// Validate an exact canonical `SimpleArchive -> SuccinctArchiveBlob` mapping.
///
/// This checks both definitions, requires their dataset scopes to agree, binds
/// the record and supplied endpoint bytes in both directions, validates the
/// target's portable format, and compares it byte-for-byte with a fresh direct
/// construction from the source.
pub fn validate_derive(
    source_definition: &CollectionDefinition,
    target_definition: &CollectionDefinition,
    claim: &CollectionDerive,
    input: &Blob<SimpleArchive>,
    output: &Blob<SuccinctArchiveBlob>,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_source_definition(source_definition)?;
    validate_definition(target_definition)?;
    if source_definition.scope() != target_definition.scope() {
        return Err(SuccinctArchiveUnionValidationError::ScopeMismatch {
            source: source_definition.scope(),
            target: target_definition.scope(),
        });
    }
    validate_collection(
        DefinitionRole::Source,
        source_definition.id(),
        claim.source(),
    )?;
    validate_collection(
        DefinitionRole::Target,
        target_definition.id(),
        claim.target(),
    )?;

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
    definition: &CollectionDefinition,
    claim: &CollectionMerge,
    low: &Blob<SuccinctArchiveBlob>,
    high: &Blob<SuccinctArchiveBlob>,
    result: &Blob<SuccinctArchiveBlob>,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_definition(definition)?;
    validate_collection(DefinitionRole::Target, definition.id(), claim.collection())?;

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

fn validate_source_definition(
    definition: &CollectionDefinition,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_definition_parts(
        DefinitionRole::Source,
        definition,
        <SimpleArchive as MetaDescribe>::id(),
    )
}

fn validate_definition(
    definition: &CollectionDefinition,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_definition_parts(
        DefinitionRole::Target,
        definition,
        <SuccinctArchiveBlob as MetaDescribe>::id(),
    )
}

fn validate_definition_parts(
    role: DefinitionRole,
    definition: &CollectionDefinition,
    expected_representation: Id,
) -> Result<(), SuccinctArchiveUnionValidationError> {
    if definition.representation() != expected_representation {
        return Err(SuccinctArchiveUnionValidationError::WrongRepresentation {
            role,
            expected: expected_representation,
            actual: definition.representation(),
        });
    }
    if definition.recipe() != TRIBLE_SET_UNION_RECIPE_V1 {
        return Err(SuccinctArchiveUnionValidationError::WrongRecipe {
            role,
            expected: TRIBLE_SET_UNION_RECIPE_V1,
            actual: definition.recipe(),
        });
    }
    Ok(())
}

fn validate_collection(
    role: DefinitionRole,
    expected: Id,
    actual: Id,
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

    use crate::blob::IntoBlob;
    use crate::collection::simplearchive_union;
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
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

    #[test]
    fn definitions_share_scope_and_union_law_but_not_representation() {
        let source = simplearchive_union::definition(id(1));
        let target = definition(id(1));

        assert_eq!(source.scope(), target.scope());
        assert_eq!(source.recipe(), target.recipe());
        assert_eq!(target.recipe(), TRIBLE_SET_UNION_RECIPE_V1);
        assert_eq!(
            source.representation(),
            <SimpleArchive as MetaDescribe>::id()
        );
        assert_eq!(
            target.representation(),
            <SuccinctArchiveBlob as MetaDescribe>::id()
        );
        assert_ne!(source.id(), target.id());
    }

    #[test]
    fn canonical_empty_is_the_derived_bottom_and_merge_identity() {
        let source_definition = simplearchive_union::definition(id(1));
        let target_definition = definition(id(1));
        let source_empty: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let derived_empty = derive_element(&source_empty).unwrap();
        let canonical_empty = empty();

        assert_eq!(derived_empty.bytes, canonical_empty.bytes);
        assert_eq!(derived_empty.get_handle(), canonical_empty.get_handle());

        let derive = CollectionDerive::new(
            source_definition.id(),
            target_definition.id(),
            data_identity(&source_empty),
            data_identity(&canonical_empty),
        );
        validate_derive(
            &source_definition,
            &target_definition,
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
            target_definition.id(),
            data_identity(low),
            data_identity(high),
            data_identity(&joined),
        );
        validate_merge(&target_definition, &merge, low, high, &joined).unwrap();
    }

    #[test]
    fn derive_and_merge_commute_to_identical_canonical_bytes() {
        let source_definition = simplearchive_union::definition(id(1));
        let target_definition = definition(id(1));
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
                source_definition.id(),
                target_definition.id(),
                data_identity(input),
                data_identity(output),
            );
            validate_derive(
                &source_definition,
                &target_definition,
                &claim,
                input,
                output,
            )
            .unwrap();
        }

        let (low, high) = ordered(&derived_left, &derived_right);
        let merge = CollectionMerge::new(
            target_definition.id(),
            data_identity(low),
            data_identity(high),
            data_identity(&merge_after_derive),
        );
        validate_merge(&target_definition, &merge, low, high, &merge_after_derive).unwrap();
    }

    #[test]
    fn validators_reject_valid_but_wrong_canonical_outputs() {
        let source_definition = simplearchive_union::definition(id(1));
        let target_definition = definition(id(1));
        let input = archive([row(1, 9, 3)]);
        let wrong_source = archive([row(2, 9, 4)]);
        let wrong_output = derive_element(&wrong_source).unwrap();
        let claim = CollectionDerive::new(
            source_definition.id(),
            target_definition.id(),
            data_identity(&input),
            data_identity(&wrong_output),
        );

        assert!(matches!(
            validate_derive(
                &source_definition,
                &target_definition,
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
            target_definition.id(),
            data_identity(low),
            data_identity(high),
            data_identity(&wrong),
        );
        assert_ne!(correct.bytes, wrong.bytes);
        assert!(matches!(
            validate_merge(&target_definition, &merge, low, high, &wrong),
            Err(SuccinctArchiveUnionValidationError::WrongMergeResult)
        ));
    }

    #[test]
    fn malformed_target_is_rejected_before_equation_admission() {
        let source_definition = simplearchive_union::definition(id(1));
        let target_definition = definition(id(1));
        let input = archive([row(1, 9, 3)]);
        let malformed = Blob::<SuccinctArchiveBlob>::new(Bytes::from(vec![0xAA; 17]));
        let claim = CollectionDerive::new(
            source_definition.id(),
            target_definition.id(),
            data_identity(&input),
            data_identity(&malformed),
        );

        assert!(matches!(
            validate_derive(
                &source_definition,
                &target_definition,
                &claim,
                &input,
                &malformed,
            ),
            Err(SuccinctArchiveUnionValidationError::WrongDeriveOutput)
        ));
    }
}
