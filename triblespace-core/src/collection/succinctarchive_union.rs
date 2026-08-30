//! Canonical raw SuccinctArchive set union and its derivation from
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive).
//!
//! SimpleArchive and raw SuccinctArchive each own canonical set union for
//! their bytes. The concrete mapping between them witnesses that the
//! canonical conversion preserves those joins:
//!
//! ```text
//! succinct(a ∪ b) = succinct(a) ∪ succinct(b)
//! ```
//!
//! This module validates exactly those `DERIVE` and `MERGE` equations. It does
//! not authorize commits, select semantic roots, retain artifacts, or assign
//! authority to construction records. `DERIVE` and `MERGE` remain unsigned,
//! reproducible evidence.

use crate::id::ExclusiveId;
// Reach arrives here as a builder argument; only the tests name a
// particular one.
use super::descriptor as descriptor_facts;
use super::records::{
    collection_authority, collection_mapping, collection_reach, collection_representation,
    collection_source, mapping_algorithm, RecordDecodeError, KIND_COLLECTION_DESCRIPTOR,
    KIND_COLLECTION_MAPPING,
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
    OrderedUniverse, Rank9AcceleratedSuccinctArchiveBlob, SuccinctArchive, SuccinctArchiveBlob,
    SuccinctArchiveRawBuildError, SuccinctArchiveRawMergeError,
};
use crate::blob::{Blob, BlobEncoding};
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::hash::Handle;
use crate::metadata::MetaDescribe;
use crate::repo::{BlobStoreGet, BlobStoreMeta};

use super::{
    CollectionData, CollectionDerive, CollectionEncoding, CollectionHandle, CollectionMapping,
    CollectionMerge, CollectionOperationError,
};

mod collection;
pub use collection::*;

impl CollectionEncoding for SuccinctArchiveBlob {
    fn validate_member<R>(
        _descriptor: &Fragment,
        member: &Blob<Self>,
        _reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        SuccinctArchiveBlob::merge(std::slice::from_ref(member))
            .map(|_| ())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))
    }

    fn join_members(
        _descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
    ) -> Result<Option<Blob<Self>>, CollectionOperationError> {
        join(low, high).map(Some).map_err(|source| match source {
            SuccinctArchiveRawMergeError::DomainTooWide
            | SuccinctArchiveRawMergeError::TooManyRows => {
                CollectionOperationError::Capacity(source.to_string())
            }
            SuccinctArchiveRawMergeError::InvalidInput { .. }
            | SuccinctArchiveRawMergeError::Construction(_) => {
                CollectionOperationError::Fatal(source.to_string())
            }
        })
    }
}

impl CollectionEncoding for Rank9AcceleratedSuccinctArchiveBlob {
    fn validate_member<R>(
        _descriptor: &Fragment,
        member: &Blob<Self>,
        reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        let source = Self::source_handle(member)
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?;
        let raw = reader
            .get::<Blob<SuccinctArchiveBlob>, SuccinctArchiveBlob>(source)
            .map_err(|source| {
                CollectionOperationError::Fatal(format!(
                    "accelerated SuccinctArchive raw child is not resident: {source}"
                ))
            })?;
        SuccinctArchive::<OrderedUniverse>::from_accelerated_parts(raw, member.clone())
            .map(|_| ())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))
    }
}

/// Mapping algorithm for canonical SimpleArchive-to-SuccinctArchive conversion.
///
/// Minted with `trible genid` on 2026-08-29.
pub const SIMPLE_TO_SUCCINCT_MAPPING_V1: Id = id_hex!("9C8CFEB097B0A336E09D506E8DD361C2");

/// Self-description of the parameter-free canonical conversion.
pub struct SimpleToSuccinctMappingV1;

impl MetaDescribe for SimpleToSuccinctMappingV1 {
    fn describe() -> Fragment {
        let id = SIMPLE_TO_SUCCINCT_MAPPING_V1;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "simple-to-succinct-v1",
                metadata::description: "Canonical conversion from a SimpleArchive trible set to its raw SuccinctArchive encoding. The mapping preserves set union and has no parameters.",
                metadata::tag: metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
        }
    }
}

fn mapping_fragment() -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_MAPPING,
        mapping_algorithm*: <SimpleToSuccinctMappingV1 as MetaDescribe>::describe(),
    }
}

/// Bound canonical SimpleArchive-to-SuccinctArchive mapping.
pub struct SimpleToSuccinctMapping;

impl CollectionMapping<SimpleArchive, SuccinctArchiveBlob> for SimpleToSuccinctMapping {
    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        let actual = descriptor_facts::mapping_algorithm(target.facts())
            .map_err(|error| CollectionOperationError::Fatal(error.to_string()))?;
        let expected = Some(SIMPLE_TO_SUCCINCT_MAPPING_V1);
        if actual != expected {
            return Err(CollectionOperationError::Fatal(format!(
                "succinct collection mapping algorithm {:?} does not match simple-to-succinct algorithm {}",
                actual.map(|id| format!("{id:X}")),
                format!("{:X}", expected.expect("mapping algorithm")),
            )));
        }
        Ok(Self)
    }

    fn map(
        &self,
        source: &Blob<SimpleArchive>,
    ) -> Result<Blob<SuccinctArchiveBlob>, CollectionOperationError> {
        derive_element(source).map_err(|source| match source {
            SuccinctArchiveRawBuildError::TooManyRows(_)
            | SuccinctArchiveRawBuildError::DomainTooWide(_) => {
                CollectionOperationError::Capacity(source.to_string())
            }
            SuccinctArchiveRawBuildError::Source(_)
            | SuccinctArchiveRawBuildError::Construction(_) => {
                CollectionOperationError::Fatal(source.to_string())
            }
        })
    }
}

/// Raw-to-accelerated mapping algorithm for 32-bit little-endian targets.
///
/// Minted with `trible genid` on 2026-08-29. The identity pins the portable
/// source schema, accelerated-root format, canonical Rank9 builder, Jerky
/// serialization epoch, pointer width, and byte order. A change that can alter
/// canonical root bytes requires a new mapping identity.
pub const RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_LE: Id =
    id_hex!("57756614C20DA2C9B1D33679136176A7");

/// Raw-to-accelerated mapping algorithm for 32-bit big-endian targets.
pub const RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_BE: Id =
    id_hex!("D225E3B88CAE65ECAA2DADA8E861D4B5");

/// Raw-to-accelerated mapping algorithm for 64-bit little-endian targets.
pub const RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_LE: Id =
    id_hex!("30B1F8ABE7BA8348551D8A94209A841C");

/// Raw-to-accelerated mapping algorithm for 64-bit big-endian targets.
pub const RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_BE: Id =
    id_hex!("7B628024E742E031F4E58789D2847D70");

/// ABI-qualified raw-to-accelerated mapping description.
pub struct RawToRank9AcceleratedMappingV1;

impl MetaDescribe for RawToRank9AcceleratedMappingV1 {
    fn describe() -> Fragment {
        let id = current_rank9_accelerated_mapping_algorithm();
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "raw-to-rank9-accelerated-succinctarchive-v1",
                metadata::description: "Canonical mapping from one portable SuccinctArchive member to its ABI-qualified Rank9-accelerated encoding. Compaction joins portable inputs first and then derives the exact source-bound accelerator, so no independent accelerated-side join is required.",
                metadata::tag: metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
        }
    }
}

#[cfg(all(target_pointer_width = "32", target_endian = "little"))]
const CURRENT_RANK9_ACCELERATED_MAPPING: Id = RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_LE;
#[cfg(all(target_pointer_width = "32", target_endian = "big"))]
const CURRENT_RANK9_ACCELERATED_MAPPING: Id = RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_BE;
#[cfg(all(target_pointer_width = "64", target_endian = "little"))]
const CURRENT_RANK9_ACCELERATED_MAPPING: Id = RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_LE;
#[cfg(all(target_pointer_width = "64", target_endian = "big"))]
const CURRENT_RANK9_ACCELERATED_MAPPING: Id = RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_BE;

/// Mapping algorithm id for the exact accelerated ABI supported by this build.
pub const fn current_rank9_accelerated_mapping_algorithm() -> Id {
    CURRENT_RANK9_ACCELERATED_MAPPING
}

fn rank9_accelerated_mapping_fragment() -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_MAPPING,
        mapping_algorithm*: <RawToRank9AcceleratedMappingV1 as MetaDescribe>::describe(),
    }
}

/// Bound canonical raw-to-accelerated SuccinctArchive mapping.
pub struct RawToRank9AcceleratedMapping;

impl CollectionMapping<SuccinctArchiveBlob, Rank9AcceleratedSuccinctArchiveBlob>
    for RawToRank9AcceleratedMapping
{
    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        let actual = descriptor_facts::mapping_algorithm(target.facts())
            .map_err(|error| CollectionOperationError::Fatal(error.to_string()))?;
        let expected = Some(current_rank9_accelerated_mapping_algorithm());
        if actual != expected {
            return Err(CollectionOperationError::Fatal(format!(
                "accelerated SuccinctArchive mapping algorithm {:?} does not match {}",
                actual.map(|id| format!("{id:X}")),
                format!("{:X}", expected.expect("mapping algorithm")),
            )));
        }
        Ok(Self)
    }

    fn map(
        &self,
        source: &Blob<SuccinctArchiveBlob>,
    ) -> Result<Blob<Rank9AcceleratedSuccinctArchiveBlob>, CollectionOperationError> {
        SuccinctArchive::<OrderedUniverse>::build_accelerated_root(source.clone())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))
    }
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
    /// A derived descriptor names another mapping algorithm.
    WrongMapping {
        /// Descriptor being checked.
        role: DescriptorRole,
        /// Required mapping algorithm.
        expected: Id,
        /// Mapping algorithm found in the descriptor.
        actual: Option<Id>,
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
    /// The supplied blob's trusted cached identity differs from the record.
    EndpointMismatch {
        /// Endpoint being checked.
        role: ElementRole,
        /// Identity named by the record.
        expected: CollectionData,
        /// Cached identity carried by the supplied blob.
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
            Self::WrongMapping {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} collection mapping algorithm {:?} does not match {}",
                actual.map(|id| format!("{id:X}")),
                format!("{expected:X}"),
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
pub fn descriptor(source: CollectionHandle, authority: VerifyingKey, reach: Fragment) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_source: source,
        collection_authority: authority,
        collection_representation*: <SuccinctArchiveBlob as MetaDescribe>::describe(),
        collection_mapping*: mapping_fragment(),
        collection_reach*: reach,
    }
}

/// Describe the Rank9-accelerated collection derived from one exact raw
/// SuccinctArchive collection.
pub fn accelerated_descriptor(
    source: CollectionHandle,
    authority: VerifyingKey,
    reach: Fragment,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_source: source,
        collection_authority: authority,
        collection_representation*: <Rank9AcceleratedSuccinctArchiveBlob as MetaDescribe>::describe(),
        collection_mapping*: rank9_accelerated_mapping_fragment(),
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
    validate_collection(
        DescriptorRole::Target,
        target_collection,
        claim.collection(),
    )?;

    let (expected_input, expected_output) = (claim.input(), claim.output());
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
/// All three endpoint identities are checked through their trusted cached
/// handles. The raw merge structurally validates both inputs while constructing
/// their canonical union;
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
        None,
    )
}

fn validate_descriptor(descriptor: &Fragment) -> Result<(), SuccinctArchiveUnionValidationError> {
    validate_descriptor_parts(
        DescriptorRole::Target,
        descriptor,
        <SuccinctArchiveBlob as MetaDescribe>::id(),
        Some(SIMPLE_TO_SUCCINCT_MAPPING_V1),
    )
}

fn validate_descriptor_parts(
    role: DescriptorRole,
    descriptor: &Fragment,
    expected_representation: Id,
    expected_mapping: Option<Id>,
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
    let actual_mapping = descriptor_facts::mapping_algorithm(descriptor.facts())?;
    if actual_mapping != expected_mapping {
        return Err(SuccinctArchiveUnionValidationError::WrongMapping {
            role,
            expected: expected_mapping.unwrap_or(SIMPLE_TO_SUCCINCT_MAPPING_V1),
            actual: actual_mapping,
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
    let actual = Handle::<S>::to_hash(blob.get_handle());
    if actual != expected {
        return Err(SuccinctArchiveUnionValidationError::EndpointMismatch {
            role,
            expected,
            actual,
        });
    }
    Ok(())
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

    fn data_identity<S: BlobEncoding>(blob: &Blob<S>) -> CollectionData {
        Handle::<S>::to_hash(blob.get_handle())
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

    /// The indexed collection is the raw one's derivation: a different
    /// encoding connected by one explicit, concrete mapping.
    #[test]
    fn the_index_derives_from_the_raw_collection_through_its_mapping() {
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
            crate::collection::descriptor::mapping(source.facts()),
            Ok(None)
        );
        assert_eq!(
            crate::collection::descriptor::mapping_algorithm(target.facts()),
            Ok(Some(SIMPLE_TO_SUCCINCT_MAPPING_V1))
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
