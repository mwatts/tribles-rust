//! Canonical path-summary union and its derivation from `SimpleArchive`.
//!
//! One collection is fixed by an extrinsic dataset scope and one canonical
//! path automaton. Its elements retain only the graph domain and direct
//! product arcs required by that automaton; their join is exact set union.
//! Lowering graph facts into those direct arcs is therefore a join
//! homomorphism:
//!
//! ```text
//! paths(a ∪ b) = paths(a) ⊔ paths(b)
//! ```
//!
//! Transitive closure is deliberately absent from the collection law. It is
//! performed once when a [`PathIndex`](crate::PathIndex) is materialized, so
//! paths whose edges live in different source fragments remain discoverable.

// Reach arrives here as a builder argument; only the tests name a
// particular one.
use std::error::Error;
use std::fmt;
#[cfg(test)]
use triblespace_core::collection::reach;
use triblespace_core::prelude::entity;

use triblespace_core::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use triblespace_core::blob::{Blob, BlobEncoding, IntoBlob};
use triblespace_core::collection::descriptor;
use triblespace_core::collection::simplearchive_union::{self, TRIBLE_SET_UNION_RECIPE_V1};
use triblespace_core::collection::{
    CollectionData, CollectionDerive, CollectionHandle, CollectionMerge, VerifyingKey,
};
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::hash::{Blake3, Hash};
use triblespace_core::inline::Inline;
use triblespace_core::metadata::MetaDescribe;
use triblespace_core::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

use crate::persistence::{
    automaton_fingerprint, path_automaton_fingerprint, PathSummaryV1, PATH_SUMMARY_RECIPE_V1,
};
use crate::{Automaton, GraphEdge, PathError, PathSummary, PathSummaryBlob, PathSummaryBlobError};

/// A collection descriptor participating in a validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DescriptorRole {
    /// Canonical `SimpleArchive` source of a derivation.
    Source,
    /// Canonical path-summary target or merge collection.
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
    /// `SimpleArchive` input of a derivation.
    DeriveInput,
    /// Path-summary output of a derivation.
    DeriveOutput,
    /// Canonically lower path-summary merge input.
    MergeLow,
    /// Canonically higher path-summary merge input.
    MergeHigh,
    /// Claimed path-summary merge output.
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

/// Failure to construct one canonical path-summary collection element.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PathSummaryUnionError {
    /// The source is not a canonical `SimpleArchive`.
    Source(UnarchiveError),
    /// Path-summary encoding or decoding failed.
    Summary(PathSummaryBlobError),
    /// Two decoded summaries could not be joined.
    Merge(PathError),
}

impl fmt::Display for PathSummaryUnionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(source) => write!(formatter, "invalid SimpleArchive source: {source}"),
            Self::Summary(source) => write!(formatter, "invalid path-summary element: {source}"),
            Self::Merge(source) => write!(formatter, "cannot join path summaries: {source}"),
        }
    }
}

impl Error for PathSummaryUnionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(source) => Some(source),
            Self::Summary(source) => Some(source),
            Self::Merge(source) => Some(source),
        }
    }
}

/// Failure to validate the canonical path-summary collection law.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PathSummaryUnionValidationError {
    /// The summary does not derive from the source it was checked against.
    WrongSource,
    /// The descriptor summarises a different automaton than the one supplied.
    WrongAutomaton,
    /// The descriptor does not carry a field this check needs.
    Malformed(triblespace_core::collection::records::RecordDecodeError),
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
        /// Required recipe.
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
    /// Fresh canonical derivation failed.
    Derive(PathSummaryUnionError),
    /// Fresh canonical merge failed.
    Merge(PathSummaryUnionError),
    /// The claimed derivation is not the canonical lowering of its source.
    WrongDeriveOutput,
    /// The claimed merge result is not the canonical union of its inputs.
    WrongMergeResult,
}

impl fmt::Display for PathSummaryUnionValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongSource => {
                write!(formatter, "summary derives from a different collection")
            }
            Self::WrongAutomaton => {
                write!(formatter, "collection summarises a different automaton")
            }
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
                "{role} collection recipe {actual:X} does not match {expected:X}"
            ),
            Self::WrongCollection {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "record {role} collection {actual:?} does not match descriptor {expected:?}"
            ),
            Self::EndpointMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} identity {actual:?} does not match claimed {expected:?}"
            ),
            Self::Derive(source) => write!(formatter, "cannot derive path summary: {source}"),
            Self::Merge(source) => write!(formatter, "cannot merge path summaries: {source}"),
            Self::WrongDeriveOutput => {
                formatter.write_str("derive output is not the canonical path summary of its input")
            }
            Self::WrongMergeResult => {
                formatter.write_str("merge result is not the exact canonical union of its inputs")
            }
        }
    }
}

impl Error for PathSummaryUnionValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Derive(source) | Self::Merge(source) => Some(source),
            _ => None,
        }
    }
}

/// Construct the path-summary collection for one dataset scope and automaton.
///
/// The automaton's canonical fingerprint participates in the recipe identity,
/// so two path expressions over the same source scope form distinct target
/// collections.
pub fn descriptor(
    source: triblespace_core::collection::records::CollectionHandle,
    automaton: &Automaton,
    authority: VerifyingKey,
    reach: Fragment,
) -> triblespace_core::trible::Fragment {
    let fingerprint = automaton_fingerprint(automaton);
    entity! { _ @
        triblespace_core::metadata::tag: triblespace_core::collection::records::KIND_COLLECTION_DESCRIPTOR,
        triblespace_core::collection::records::collection_source: source,
        triblespace_core::collection::records::collection_authority: authority,
        triblespace_core::collection::records::collection_representation*:
            <PathSummaryBlob as MetaDescribe>::describe(),
        triblespace_core::collection::records::collection_recipe*:
            <PathSummaryV1 as MetaDescribe>::describe(),
        path_automaton_fingerprint: fingerprint,
        triblespace_core::collection::records::collection_reach*: reach,
    }
}

/// Return the canonical empty path summary for one fixed automaton.
pub fn empty(automaton: &Automaton) -> Blob<PathSummaryBlob> {
    let summary = PathSummary::from_edges(automaton.clone(), std::iter::empty::<GraphEdge>());
    PathSummaryBlob::encode(&summary)
        .expect("the fixed empty path-summary construction cannot fail")
}

/// Canonically derive one path-summary element from a `SimpleArchive`.
pub fn derive_element(
    source: &Blob<SimpleArchive>,
    automaton: &Automaton,
) -> Result<Blob<PathSummaryBlob>, PathSummaryUnionError> {
    simplearchive_union::validate_element(source).map_err(PathSummaryUnionError::Source)?;
    let edges = source.bytes.as_ref().chunks_exact(TRIBLE_LEN).map(|chunk| {
        let raw: &[u8; TRIBLE_LEN] = chunk
            .try_into()
            .expect("validated SimpleArchive chunks have fixed trible length");
        let trible = Trible::as_transmute_force_raw(raw)
            .expect("validated SimpleArchive contains valid tribles");
        GraphEdge::from(trible)
    });
    let summary = PathSummary::from_edges(automaton.clone(), edges);
    PathSummaryBlob::encode(&summary).map_err(PathSummaryUnionError::Summary)
}

/// Compute the canonical union of two path-summary elements.
pub fn join(
    left: &Blob<PathSummaryBlob>,
    right: &Blob<PathSummaryBlob>,
    automaton: &Automaton,
) -> Result<Blob<PathSummaryBlob>, PathSummaryUnionError> {
    let left =
        PathSummaryBlob::decode(left.clone(), automaton).map_err(PathSummaryUnionError::Summary)?;
    let right = PathSummaryBlob::decode(right.clone(), automaton)
        .map_err(PathSummaryUnionError::Summary)?;
    let joined = left.merge(&right).map_err(PathSummaryUnionError::Merge)?;
    PathSummaryBlob::encode(&joined).map_err(PathSummaryUnionError::Summary)
}

/// Validate an exact canonical `SimpleArchive -> PathSummaryBlob` mapping.
///
/// Both descriptor handles, both endpoint identities, dataset scope, source
/// canonicality, and byte-exact output are checked. Authorization of source
/// commits remains an independent collection-resolution concern.
pub fn validate_derive(
    source_descriptor: &TribleSet,
    target_descriptor: &TribleSet,
    claim: &CollectionDerive,
    input: &Blob<SimpleArchive>,
    output: &Blob<PathSummaryBlob>,
    automaton: &Automaton,
) -> Result<(), PathSummaryUnionValidationError> {
    validate_source_descriptor(source_descriptor)?;
    validate_target_descriptor(target_descriptor, automaton)?;
    let source_collection: CollectionHandle =
        IntoBlob::<SimpleArchive>::to_blob(source_descriptor.clone()).get_handle();
    let target_collection: CollectionHandle =
        IntoBlob::<SimpleArchive>::to_blob(target_descriptor.clone()).get_handle();
    match descriptor::source(target_descriptor)? {
        Some(source) if source == source_collection => {}
        _ => return Err(PathSummaryUnionValidationError::WrongSource),
    }
    validate_collection(DescriptorRole::Target, target_collection, claim.target())?;

    let (expected_input, expected_output) = claim.mapping();
    validate_endpoint(ElementRole::DeriveInput, expected_input, input)?;
    validate_endpoint(ElementRole::DeriveOutput, expected_output, output)?;
    let expected =
        derive_element(input, automaton).map_err(PathSummaryUnionValidationError::Derive)?;
    if output.bytes != expected.bytes {
        return Err(PathSummaryUnionValidationError::WrongDeriveOutput);
    }
    Ok(())
}

/// Validate an exact canonical path-summary union equation.
///
/// All endpoint identities are recomputed from supplied bytes. Both inputs
/// are exact-decoded against the collection automaton before their freshly
/// constructed union is compared byte-for-byte with the claimed result.
pub fn validate_merge(
    collection_descriptor: &TribleSet,
    claim: &CollectionMerge,
    low: &Blob<PathSummaryBlob>,
    high: &Blob<PathSummaryBlob>,
    result: &Blob<PathSummaryBlob>,
    automaton: &Automaton,
) -> Result<(), PathSummaryUnionValidationError> {
    validate_target_descriptor(collection_descriptor, automaton)?;
    let collection: CollectionHandle =
        IntoBlob::<SimpleArchive>::to_blob(collection_descriptor.clone()).get_handle();
    validate_collection(DescriptorRole::Target, collection, claim.collection())?;

    let (expected_low, expected_high) = claim.inputs();
    validate_endpoint(ElementRole::MergeLow, expected_low, low)?;
    validate_endpoint(ElementRole::MergeHigh, expected_high, high)?;
    validate_endpoint(ElementRole::MergeResult, claim.result(), result)?;

    let expected = join(low, high, automaton).map_err(PathSummaryUnionValidationError::Merge)?;
    if result.bytes != expected.bytes {
        return Err(PathSummaryUnionValidationError::WrongMergeResult);
    }
    Ok(())
}

fn validate_source_descriptor(
    collection_descriptor: &TribleSet,
) -> Result<(), PathSummaryUnionValidationError> {
    validate_descriptor_parts(
        DescriptorRole::Source,
        collection_descriptor,
        <SimpleArchive as MetaDescribe>::id(),
        TRIBLE_SET_UNION_RECIPE_V1,
    )
}

fn validate_target_descriptor(
    collection_descriptor: &TribleSet,
    automaton: &Automaton,
) -> Result<(), PathSummaryUnionValidationError> {
    validate_descriptor_parts(
        DescriptorRole::Target,
        collection_descriptor,
        <PathSummaryBlob as MetaDescribe>::id(),
        PATH_SUMMARY_RECIPE_V1,
    )?;
    let expected = automaton_fingerprint(automaton);
    match descriptor::argument(collection_descriptor, path_automaton_fingerprint.id())? {
        Some(actual) if actual == expected.raw => Ok(()),
        _ => Err(PathSummaryUnionValidationError::WrongAutomaton),
    }
}

fn validate_descriptor_parts(
    role: DescriptorRole,
    collection_descriptor: &TribleSet,
    expected_representation: Id,
    expected_recipe: Id,
) -> Result<(), PathSummaryUnionValidationError> {
    descriptor::authority(collection_descriptor)?;
    let representation = descriptor::representation(collection_descriptor)?;
    if representation != expected_representation {
        return Err(PathSummaryUnionValidationError::WrongRepresentation {
            role,
            expected: expected_representation,
            actual: representation,
        });
    }
    let recipe = descriptor::recipe(collection_descriptor)?;
    if recipe != expected_recipe {
        return Err(PathSummaryUnionValidationError::WrongRecipe {
            role,
            expected: expected_recipe,
            actual: recipe,
        });
    }
    Ok(())
}

fn validate_collection(
    role: DescriptorRole,
    expected: CollectionHandle,
    actual: CollectionHandle,
) -> Result<(), PathSummaryUnionValidationError> {
    if actual != expected {
        return Err(PathSummaryUnionValidationError::WrongCollection {
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
) -> Result<(), PathSummaryUnionValidationError> {
    let actual = data_identity(blob);
    if actual != expected {
        return Err(PathSummaryUnionValidationError::EndpointMismatch {
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

    use ed25519_dalek::SigningKey;
    use ed25519_dalek::VerifyingKey;
    use triblespace_core::blob::IntoBlob;
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::inline::RawInline;
    use triblespace_core::metadata;
    use triblespace_core::prelude::entity;
    use triblespace_core::trible::Fragment;
    use triblespace_core::trible::TribleSet;

    use crate::{PathIndex, Step, Transition};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    /// Authority shared by these test collections.
    fn authority() -> VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    /// The source collection these tests summarise.
    fn source_collection() -> Fragment {
        simplearchive_union::descriptor("edges", authority(), reach::private())
    }

    /// These tests only need identities to bind claims to; nothing stores the
    /// descriptors they come from.
    fn collection_of(descriptor: &Fragment) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle()
    }

    fn label(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn plus(attribute: [u8; 16]) -> Automaton {
        Automaton::new(
            2,
            [0],
            [1],
            [
                Transition::new(0, 1, Step::Forward(attribute)),
                Transition::new(1, 1, Step::Forward(attribute)),
            ],
        )
        .unwrap()
    }

    fn edge_facts(source_byte: u8, target_byte: u8) -> TribleSet {
        let source = id(source_byte);
        let target = id(target_byte);
        entity! { ExclusiveId::force_ref(&source) @ metadata::tag: target }.into_facts()
    }

    fn archive(facts: &TribleSet) -> Blob<SimpleArchive> {
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
    fn descriptor_identity_includes_source_representation_and_automaton() {
        let first_automaton = plus(label(7));
        let second_automaton = plus(label(8));
        let source = source_collection();
        let first = descriptor(
            collection_of(&source),
            &first_automaton,
            authority(),
            reach::private(),
        );
        let repeated = descriptor(
            collection_of(&source),
            &first_automaton,
            authority(),
            reach::private(),
        );
        let second = descriptor(
            collection_of(&source),
            &second_automaton,
            authority(),
            reach::private(),
        );

        assert_eq!(first, repeated);
        // A summary names the collection it summarises, and carries no anchor
        // of its own.
        assert_eq!(
            descriptor::source(first.facts()),
            Ok(Some(collection_of(&source)))
        );
        assert!(
            descriptor::name(first.facts()).unwrap().is_none(),
            "a derivation needs no anchor"
        );
        assert_eq!(descriptor::authority(first.facts()), Ok(authority()));
        // The same automaton over a different source is a different summary.
        assert_ne!(
            collection_of(&first),
            collection_of(&descriptor(
                collection_of(&simplearchive_union::descriptor(
                    "other-edges",
                    authority(),
                    reach::private()
                )),
                &first_automaton,
                authority(),
                reach::private(),
            ))
        );
        assert_eq!(
            descriptor::representation(first.facts()).unwrap(),
            <PathSummaryBlob as MetaDescribe>::id()
        );
        assert_ne!(
            descriptor::representation(first.facts()),
            descriptor::representation(source.facts())
        );
        assert_ne!(
            descriptor::recipe(first.facts()),
            descriptor::recipe(source.facts())
        );
        // Two summaries over different automata share the law and are told
        // apart by its argument, so the recipe matches while the collections
        // differ. The automaton is readable from the descriptor rather than
        // hidden inside a derived recipe id.
        assert_eq!(
            descriptor::recipe(first.facts()),
            descriptor::recipe(second.facts())
        );
        assert_ne!(
            descriptor::argument(first.facts(), path_automaton_fingerprint.id()),
            descriptor::argument(second.facts(), path_automaton_fingerprint.id())
        );
        assert_ne!(collection_of(&first), collection_of(&second));
    }

    #[test]
    fn canonical_empty_is_total_derived_bottom_and_join_identity() {
        let automaton = plus(label(7));
        let source_descriptor = source_collection();
        let target_descriptor = descriptor(
            collection_of(&source_collection()),
            &automaton,
            authority(),
            reach::private(),
        );
        let source_empty = archive(&TribleSet::new());
        let canonical_empty = empty(&automaton);

        assert_eq!(canonical_empty.bytes.len(), 48);
        let mut expected_empty = Vec::with_capacity(48);
        expected_empty.extend_from_slice(&crate::automaton_fingerprint(&automaton).raw);
        expected_empty.extend_from_slice(&automaton.state_count().to_le_bytes());
        expected_empty.extend_from_slice(&0u32.to_le_bytes());
        expected_empty.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(canonical_empty.bytes.as_ref(), expected_empty);
        assert_eq!(canonical_empty.get_handle(), empty(&automaton).get_handle());
        let decoded = PathSummaryBlob::decode(canonical_empty.clone(), &automaton).unwrap();
        assert!(decoded.vertices().is_empty());
        assert_eq!(decoded.direct_arc_count(), 0);

        let derived_empty = derive_element(&source_empty, &automaton).unwrap();
        assert_eq!(derived_empty.bytes, canonical_empty.bytes);

        // `metadata::tag` does not match the fixed label(7) transition.
        let unmatched_source = archive(&edge_facts(2, 3));
        let derived_unmatched = derive_element(&unmatched_source, &automaton).unwrap();
        assert_eq!(derived_unmatched.bytes, canonical_empty.bytes);

        for (input, output) in [
            (&source_empty, &derived_empty),
            (&unmatched_source, &derived_unmatched),
        ] {
            let claim = CollectionDerive::new(
                collection_of(&target_descriptor),
                data_identity(input),
                data_identity(output),
            );
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &claim,
                input,
                output,
                &automaton,
            )
            .unwrap();
        }

        let matching_automaton = plus(metadata::tag.id().into());
        let matching = derive_element(&archive(&edge_facts(4, 5)), &matching_automaton).unwrap();
        let matching_empty = empty(&matching_automaton);
        let left_identity = join(&matching_empty, &matching, &matching_automaton).unwrap();
        let right_identity = join(&matching, &matching_empty, &matching_automaton).unwrap();
        let idempotent = join(&matching, &matching, &matching_automaton).unwrap();
        for joined in [left_identity, right_identity, idempotent] {
            assert_eq!(joined.bytes, matching.bytes);
            assert_eq!(joined.get_handle(), matching.get_handle());
        }
    }

    #[test]
    fn derive_and_merge_commute_and_close_cross_fragment_paths() {
        let automaton = plus(metadata::tag.id().into());
        let source_descriptor = source_collection();
        let target_descriptor = descriptor(
            collection_of(&source_collection()),
            &automaton,
            authority(),
            reach::private(),
        );
        let left = archive(&edge_facts(1, 2));
        let right = archive(&edge_facts(2, 3));

        let source_union = simplearchive_union::join(&left, &right).unwrap();
        let derive_after_source_join = derive_element(&source_union, &automaton).unwrap();
        let derived_left = derive_element(&left, &automaton).unwrap();
        let derived_right = derive_element(&right, &automaton).unwrap();
        let join_after_derive = join(&derived_left, &derived_right, &automaton).unwrap();

        assert_eq!(derive_after_source_join.bytes, join_after_derive.bytes);
        assert_eq!(
            derive_after_source_join.get_handle(),
            join_after_derive.get_handle()
        );

        for (input, output) in [
            (&left, &derived_left),
            (&right, &derived_right),
            (&source_union, &derive_after_source_join),
        ] {
            let claim = CollectionDerive::new(
                collection_of(&target_descriptor),
                data_identity(input),
                data_identity(output),
            );
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &claim,
                input,
                output,
                &automaton,
            )
            .unwrap();
        }

        let (low, high) = ordered(&derived_left, &derived_right);
        let merge = CollectionMerge::new(
            collection_of(&target_descriptor),
            data_identity(low),
            data_identity(high),
            data_identity(&join_after_derive),
        );
        validate_merge(
            &target_descriptor,
            &merge,
            low,
            high,
            &join_after_derive,
            &automaton,
        )
        .unwrap();

        let summary = PathSummaryBlob::decode(join_after_derive, &automaton).unwrap();
        let index = PathIndex::from_summary(summary).unwrap();
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(3)),));
    }

    #[test]
    fn nullable_unmatched_domain_obeys_the_same_homomorphism() {
        let automaton = Automaton::new(1, [0], [0], []).unwrap();
        let source_descriptor = source_collection();
        let target_descriptor = descriptor(
            collection_of(&source_collection()),
            &automaton,
            authority(),
            reach::private(),
        );
        let left = archive(&edge_facts(1, 2));
        let right = archive(&edge_facts(2, 3));
        let source_union = simplearchive_union::join(&left, &right).unwrap();

        let derive_after_source_join = derive_element(&source_union, &automaton).unwrap();
        let derived_left = derive_element(&left, &automaton).unwrap();
        let derived_right = derive_element(&right, &automaton).unwrap();
        let join_after_derive = join(&derived_left, &derived_right, &automaton).unwrap();

        assert_eq!(derive_after_source_join.bytes, join_after_derive.bytes);
        let derive = CollectionDerive::new(
            collection_of(&target_descriptor),
            data_identity(&source_union),
            data_identity(&derive_after_source_join),
        );
        validate_derive(
            &source_descriptor,
            &target_descriptor,
            &derive,
            &source_union,
            &derive_after_source_join,
            &automaton,
        )
        .unwrap();
        let (low, high) = ordered(&derived_left, &derived_right);
        let merge = CollectionMerge::new(
            collection_of(&target_descriptor),
            data_identity(low),
            data_identity(high),
            data_identity(&join_after_derive),
        );
        validate_merge(
            &target_descriptor,
            &merge,
            low,
            high,
            &join_after_derive,
            &automaton,
        )
        .unwrap();

        let summary = PathSummaryBlob::decode(join_after_derive, &automaton).unwrap();
        assert_eq!(summary.vertices().len(), 3);
        assert_eq!(summary.direct_arc_count(), 0);
        let index = PathIndex::from_summary(summary).unwrap();
        assert_eq!(index.accepted_pair_count(), 3);

        let joined_with_empty = join(&empty(&automaton), &derived_left, &automaton).unwrap();
        assert_eq!(joined_with_empty.bytes, derived_left.bytes);
    }

    #[test]
    fn validators_reject_wrong_descriptors_endpoints_and_equations() {
        let automaton = plus(metadata::tag.id().into());
        let source_descriptor = source_collection();
        let target_descriptor = descriptor(
            collection_of(&source_collection()),
            &automaton,
            authority(),
            reach::private(),
        );
        let input = archive(&edge_facts(1, 2));
        let other_input = archive(&edge_facts(3, 4));
        let output = derive_element(&input, &automaton).unwrap();
        let other_output = derive_element(&other_input, &automaton).unwrap();

        let wrong_equation = CollectionDerive::new(
            collection_of(&target_descriptor),
            data_identity(&input),
            data_identity(&other_output),
        );
        assert!(matches!(
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &wrong_equation,
                &input,
                &other_output,
                &automaton,
            ),
            Err(PathSummaryUnionValidationError::WrongDeriveOutput)
        ));

        let wrong_endpoint = CollectionDerive::new(
            collection_of(&target_descriptor),
            data_identity(&other_input),
            data_identity(&output),
        );
        assert!(matches!(
            validate_derive(
                &source_descriptor,
                &target_descriptor,
                &wrong_endpoint,
                &input,
                &output,
                &automaton,
            ),
            Err(PathSummaryUnionValidationError::EndpointMismatch {
                role: ElementRole::DeriveInput,
                ..
            })
        ));

        let foreign_automaton = plus(label(9));
        let foreign_target = descriptor(
            collection_of(&source_collection()),
            &foreign_automaton,
            authority(),
            reach::private(),
        );
        let foreign_claim = CollectionDerive::new(
            collection_of(&foreign_target),
            data_identity(&input),
            data_identity(&output),
        );
        assert!(matches!(
            validate_derive(
                &source_descriptor,
                &foreign_target,
                &foreign_claim,
                &input,
                &output,
                &automaton,
            ),
            Err(PathSummaryUnionValidationError::WrongAutomaton)
        ));

        let (low, high) = ordered(&output, &other_output);
        let wrong_result = empty(&automaton);
        let merge = CollectionMerge::new(
            collection_of(&target_descriptor),
            data_identity(low),
            data_identity(high),
            data_identity(&wrong_result),
        );
        assert!(matches!(
            validate_merge(
                &target_descriptor,
                &merge,
                low,
                high,
                &wrong_result,
                &automaton,
            ),
            Err(PathSummaryUnionValidationError::WrongMergeResult)
        ));
    }
}

impl From<triblespace_core::collection::records::RecordDecodeError>
    for PathSummaryUnionValidationError
{
    fn from(error: triblespace_core::collection::records::RecordDecodeError) -> Self {
        Self::Malformed(error)
    }
}
