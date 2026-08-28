//! Store-centric publication and snapshots for canonical collections.
//!
//! A collection is its descriptor handle. Registering a descriptor through
//! [`CollectionStoreExt::collection`] stores and offers its complete
//! attachment closure, while later operations take only that handle. The
//! descriptor's mandatory authority is therefore part of the collection's
//! identity rather than a caller-supplied policy which could disagree with it.
//!
//! Local publication is deliberately unconditional: a store may record any
//! structurally valid, strictly signed commit. Authority is enforced when a
//! ticket or snapshot is constructed. The descriptor authority's own commits
//! are admitted directly; delegated writers need an explicit proof for
//! [`ACTION_WRITE`] on that exact descriptor handle.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::encodings::utf8string::UTF8String;
use crate::blob::encodings::UnknownBlob;
use crate::blob::Blob;
use crate::capability::{
    CapabilityAction, CapabilityAtom, CapabilityMode, CapabilityProofBundle, CapabilityProofError,
    CapabilityRequest, CapabilityResource,
};
use crate::clock;
use crate::id::Id;
use crate::inline::encodings::ed25519::ED25519PublicKey;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::repo::{
    ArtifactHandle, ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut,
    OfferCapture, OfferCaptureInsertError,
};
// Reach arrives here as a builder argument; only the tests name a
// particular one.
use crate::trible::{Fragment, TribleSet};

use super::discovery::{discover_collection_records_for_collection_ticket, validate_exact_ticket};
use super::simplearchive_union::{
    self, MaterializationError, PublicationError, SimpleArchiveUnionValidationError,
};
use super::{
    collection_physical_cover, descriptor, discover_collection_records_authorized,
    resolve_collection_semantics, CollectionClaimValidation, CollectionCommit, CollectionData,
    CollectionDiscoveryError, CollectionFunctionalConflict, CollectionHandle,
    CollectionResolutionError, CollectionStore, CollectionValidationRequest,
    DiscoveredCollectionRecords, ExactTicketError, RecordDecodeError, ACTION_WRITE,
};

/// One owned proof together with the exact leaf subject it is expected to
/// authorize.
///
/// Keeping the expectation beside the untrusted proof prevents callers from
/// accidentally treating whatever subject a proof happens to contain as an
/// admission decision. Verification still binds the proof to the exact
/// collection action/resource atom at operation time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityPresentation {
    expected_leaf: VerifyingKey,
    bundle: CapabilityProofBundle,
}

impl CapabilityPresentation {
    /// Pair an expected leaf subject with one untrusted owned proof bundle.
    pub fn new(expected_leaf: VerifyingKey, bundle: CapabilityProofBundle) -> Self {
        Self {
            expected_leaf,
            bundle,
        }
    }

    /// Exact leaf subject this presentation is expected to authorize.
    pub fn expected_leaf(&self) -> VerifyingKey {
        self.expected_leaf
    }

    /// Candidate proof and its claim closure, verified afresh for each operation.
    pub fn bundle(&self) -> &CapabilityProofBundle {
        &self.bundle
    }

    /// Consume the presentation into its expected leaf and proof bundle.
    pub fn into_parts(self) -> (VerifyingKey, CapabilityProofBundle) {
        (self.expected_leaf, self.bundle)
    }
}

/// One explicitly supplied capability presentation failed verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionAdmissionError {
    presentation: usize,
    expected_leaf: VerifyingKey,
    source: CapabilityProofError,
}

impl CollectionAdmissionError {
    /// Zero-based index of the invalid presentation.
    pub const fn presentation(&self) -> usize {
        self.presentation
    }

    /// Expected leaf subject paired with the invalid proof.
    pub fn expected_leaf(&self) -> VerifyingKey {
        self.expected_leaf
    }

    /// Exact proof-verification failure.
    pub const fn proof_error(&self) -> &CapabilityProofError {
        &self.source
    }
}

impl fmt::Display for CollectionAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "capability presentation {} for expected leaf {} is invalid: {}",
            self.presentation,
            hex::encode_upper(self.expected_leaf.to_bytes()),
            self.source,
        )
    }
}

impl Error for CollectionAdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Failure while reading and exact-validating a collection descriptor by its
/// content identity.
#[derive(Debug)]
pub enum CollectionDescriptorError<ReaderError, GetError> {
    /// The blob-reader snapshot could not be created.
    Reader(ReaderError),
    /// The descriptor blob could not be fetched.
    Get {
        /// Requested collection identity.
        collection: CollectionHandle,
        /// Backend fetch failure.
        source: GetError,
    },
    /// Returned bytes did not hash to the requested collection identity.
    Identity {
        /// Requested collection identity.
        expected: CollectionHandle,
        /// Identity recomputed from the returned bytes.
        actual: CollectionHandle,
    },
    /// The bytes were not a canonical, generically well-formed descriptor.
    Invalid {
        /// Requested collection identity.
        collection: CollectionHandle,
        /// Exact structural failure.
        source: RecordDecodeError,
    },
}

impl<ReaderError, GetError> fmt::Display for CollectionDescriptorError<ReaderError, GetError>
where
    ReaderError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reader(source) => {
                write!(formatter, "failed to open descriptor blob view: {source}")
            }
            Self::Get { collection, source } => write!(
                formatter,
                "failed to fetch collection descriptor {}: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::Identity { expected, actual } => write!(
                formatter,
                "collection descriptor bytes hash to {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::Invalid { collection, source } => write!(
                formatter,
                "collection descriptor {} is invalid: {source}",
                hex::encode_upper(collection.raw),
            ),
        }
    }
}

impl<ReaderError, GetError> Error for CollectionDescriptorError<ReaderError, GetError>
where
    ReaderError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Reader(source) => Some(source),
            Self::Get { source, .. } => Some(source),
            Self::Identity { .. } => None,
            Self::Invalid { source, .. } => Some(source),
        }
    }
}

/// Failure to register and advertise a self-contained collection descriptor.
#[derive(Debug)]
pub enum CollectionRegistrationError<PutError, OfferError> {
    /// Descriptor facts did not have the mandatory generic shape.
    InvalidDescriptor(RecordDecodeError),
    /// A mandatory descriptor attachment was absent or did not match the
    /// handle carried by the descriptor facts.
    InvalidAttachment {
        /// Semantic role of the mandatory attachment.
        role: &'static str,
        /// Content identity named by the descriptor.
        artifact: ArtifactHandle,
    },
    /// One attachment or the canonical descriptor archive could not be stored.
    DependencyPut(PutError),
    /// The complete stored closure could not be advertised.
    Offer {
        /// Backend offer failure.
        source: OfferError,
        /// Canonical retry-all batch. Some members may already be offered.
        artifacts: Vec<ArtifactHandle>,
    },
}

impl<PutError, OfferError> fmt::Display for CollectionRegistrationError<PutError, OfferError>
where
    PutError: fmt::Display,
    OfferError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(source) => {
                write!(formatter, "invalid collection descriptor: {source}")
            }
            Self::InvalidAttachment { role, artifact } => write!(
                formatter,
                "collection descriptor's {role} attachment {} is missing or invalid",
                hex::encode_upper(artifact.raw),
            ),
            Self::DependencyPut(source) => {
                write!(
                    formatter,
                    "failed to store collection descriptor closure: {source}"
                )
            }
            Self::Offer { source, artifacts } => write!(
                formatter,
                "failed to offer {} collection descriptor artifact(s): {source}",
                artifacts.len(),
            ),
        }
    }
}

impl<PutError, OfferError> Error for CollectionRegistrationError<PutError, OfferError>
where
    PutError: Error + 'static,
    OfferError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidDescriptor(source) => Some(source),
            Self::InvalidAttachment { .. } => None,
            Self::DependencyPut(source) => Some(source),
            Self::Offer { source, .. } => Some(source),
        }
    }
}

/// One exact admitted collection frontier.
///
/// The commits are duplicate-free and ordered by intrinsic record id. Keeping
/// the collection identity beside them prevents an exact ticket for one
/// descriptor from being accidentally attached to another.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionTicket {
    collection: CollectionHandle,
    commits: Vec<CollectionCommit>,
}

impl CollectionTicket {
    pub(crate) fn from_canonical(
        collection: CollectionHandle,
        commits: Vec<CollectionCommit>,
    ) -> Self {
        debug_assert!(commits.windows(2).all(|pair| pair[0].id() < pair[1].id()));
        debug_assert!(commits
            .iter()
            .all(|commit| commit.collection() == collection));
        Self {
            collection,
            commits,
        }
    }

    /// Exact descriptor this ticket observes.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Canonical admitted commit set.
    pub fn commits(&self) -> &[CollectionCommit] {
        &self.commits
    }

    /// Number of admitted commits.
    pub fn len(&self) -> usize {
        self.commits.len()
    }

    /// Whether no commit was admitted.
    pub fn is_empty(&self) -> bool {
        self.commits.is_empty()
    }
}

/// One coherent known-prefix view of a scoped collection.
///
/// [`commits`](Self::commits) is the exact set of commits from the single
/// collection-record discovery pass that admitted [`facts`](Self::facts).
/// [`reader`](Self::reader) is the blob-reader snapshot used to validate and
/// materialize those facts. The reader may contain physically available blobs
/// published after record discovery, but those blobs acquire no semantic role
/// unless their commits are present in this snapshot.
pub struct CollectionSnapshot<R> {
    facts: TribleSet,
    ticket: CollectionTicket,
    reader: R,
}

impl<R> CollectionSnapshot<R> {
    /// Materialized union admitted by this snapshot's exact commit set.
    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    /// Exact admitted commits, ordered by intrinsic record id.
    pub fn commits(&self) -> &[CollectionCommit] {
        self.ticket.commits()
    }

    /// Exact collection frontier from which the facts were materialized.
    pub fn ticket(&self) -> &CollectionTicket {
        &self.ticket
    }

    /// Blob-reader snapshot used to validate and materialize the facts.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Consume the snapshot and return its materialized facts.
    pub fn into_facts(self) -> TribleSet {
        self.facts
    }

    /// Consume the snapshot into materialized facts, exact commits, and reader.
    pub fn into_parts(self) -> (TribleSet, CollectionTicket, R) {
        (self.facts, self.ticket, self.reader)
    }
}

/// Failure to discover one exact admitted commit ticket.
#[derive(Debug)]
pub enum CollectionTicketError<RecordsError, ReaderError, GetError> {
    /// The collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<ReaderError, GetError>),
    /// One explicitly supplied capability proof was invalid at this operation's
    /// single clock observation.
    Admission(CollectionAdmissionError),
    /// Target collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
}

impl<RecordsError, ReaderError, GetError> fmt::Display
    for CollectionTicketError<RecordsError, ReaderError, GetError>
where
    RecordsError: fmt::Display,
    ReaderError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(source) => source.fmt(formatter),
            Self::Admission(source) => source.fmt(formatter),
            Self::Discovery(source) => source.fmt(formatter),
        }
    }
}

impl<RecordsError, ReaderError, GetError> Error
    for CollectionTicketError<RecordsError, ReaderError, GetError>
where
    RecordsError: Error + 'static,
    ReaderError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(source) => Some(source),
            Self::Admission(source) => Some(source),
            Self::Discovery(source) => Some(source),
        }
    }
}

/// Failure to publish one collection element through explicit admission.
#[derive(Debug)]
pub enum CollectionCommitError<ReaderError, GetError, PutError, InsertError> {
    /// The named collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<ReaderError, GetError>),
    /// Canonical fragment publication failed.
    Publication(PublicationError<PutError, InsertError>),
}

impl<ReaderError, GetError, PutError, InsertError> fmt::Display
    for CollectionCommitError<ReaderError, GetError, PutError, InsertError>
where
    ReaderError: fmt::Display,
    GetError: fmt::Display,
    PutError: fmt::Display,
    InsertError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(source) => source.fmt(formatter),
            Self::Publication(source) => source.fmt(formatter),
        }
    }
}

impl<ReaderError, GetError, PutError, InsertError> Error
    for CollectionCommitError<ReaderError, GetError, PutError, InsertError>
where
    ReaderError: Error + 'static,
    GetError: Error + 'static,
    PutError: Error + 'static,
    InsertError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(source) => Some(source),
            Self::Publication(source) => Some(source),
        }
    }
}

/// Failure to materialize the complete admitted value of a collection.
///
/// Every admitted strictly verified commit is ground truth, so its
/// descriptor, data, and metadata fail loud. Unsigned equations are only
/// replaceable cache evidence: missing or invalid equations are omitted from
/// the resolved semantics and cannot hide a valid committed leaf.
#[derive(Debug)]
pub enum CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError> {
    /// One explicitly supplied capability proof was invalid.
    Admission(CollectionAdmissionError),
    /// Native collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
    /// A supplied exact ticket was not an exact resident commit set.
    ExactTicket(ExactTicketError),
    /// An admitted commit's canonical descriptor blob could not be fetched.
    DescriptorGet {
        /// Canonical collection-descriptor handle.
        collection: CollectionHandle,
        /// Backend fetch failure.
        source: GetError,
    },
    /// The fetched descriptor bytes were not the exact canonical descriptor
    /// archive named by the collection handle.
    InvalidDescriptor {
        /// Canonical collection-descriptor handle.
        collection: CollectionHandle,
        /// Structural descriptor decoding failure.
        source: RecordDecodeError,
    },
    /// The fetched descriptor bytes did not hash to the handle named by the
    /// commits.
    DescriptorIdentity {
        /// Descriptor handle named by the commits.
        expected: CollectionHandle,
        /// Handle recomputed from the fetched bytes.
        actual: CollectionHandle,
    },
    /// The fetched canonical descriptor did not equal the descriptor expected
    /// by this facade.
    DescriptorMismatch {
        /// Descriptor handle named by this facade and its commits.
        collection: CollectionHandle,
    },
    /// The blob reader could not be created after record discovery.
    Reader(ReaderError),
    /// An admitted commit's data blob could not be fetched.
    CommitDataGet {
        /// Intrinsic commit record id.
        commit: Id,
        /// Claimed data identity.
        data: CollectionData,
        /// Backend fetch failure.
        source: GetError,
    },
    /// An admitted commit's data failed exact `SimpleArchive` collection
    /// validation.
    InvalidCommitData {
        /// Intrinsic commit record id.
        commit: Id,
        /// Exact representation or identity diagnostic.
        source: SimpleArchiveUnionValidationError,
    },
    /// An admitted commit's mandatory metadata archive could not be fetched.
    CommitMetadataGet {
        /// Intrinsic commit record id.
        commit: Id,
        /// Mandatory metadata archive handle.
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Backend fetch failure.
        source: GetError,
    },
    /// An admitted commit's mandatory metadata was not a canonical
    /// `SimpleArchive`.
    InvalidCommitMetadata {
        /// Intrinsic commit record id.
        commit: Id,
        /// Mandatory metadata archive handle.
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Canonical archive failure.
        source: UnarchiveError,
    },
    /// An admitted commit's canonical metadata bytes did not have the exact
    /// identity signed by the commit.
    InvalidCommitMetadataIdentity {
        /// Intrinsic commit record id.
        commit: Id,
        /// Signed metadata archive handle.
        expected: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Blake3 handle recomputed from the returned bytes.
        actual: crate::inline::Inline<Handle<SimpleArchive>>,
    },
    /// Positively validated equations contradicted operation functionality.
    ResolutionConflict(Box<CollectionFunctionalConflict>),
    /// The resolved semantic frontier could not be physically materialized.
    Materialize(MaterializationError<MetaError, GetError>),
}

impl<RecordsError, ReaderError, MetaError, GetError>
    From<CollectionTicketError<RecordsError, ReaderError, GetError>>
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
{
    fn from(source: CollectionTicketError<RecordsError, ReaderError, GetError>) -> Self {
        match source {
            CollectionTicketError::Descriptor(CollectionDescriptorError::Reader(source)) => {
                Self::Reader(source)
            }
            CollectionTicketError::Descriptor(CollectionDescriptorError::Get {
                collection,
                source,
            }) => Self::DescriptorGet { collection, source },
            CollectionTicketError::Descriptor(CollectionDescriptorError::Identity {
                expected,
                actual,
            }) => Self::DescriptorIdentity { expected, actual },
            CollectionTicketError::Descriptor(CollectionDescriptorError::Invalid {
                collection,
                source,
            }) => Self::InvalidDescriptor { collection, source },
            CollectionTicketError::Admission(source) => Self::Admission(source),
            CollectionTicketError::Discovery(source) => Self::Discovery(source),
        }
    }
}

impl<RecordsError, ReaderError, MetaError, GetError>
    From<CollectionDescriptorError<ReaderError, GetError>>
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
{
    fn from(source: CollectionDescriptorError<ReaderError, GetError>) -> Self {
        match source {
            CollectionDescriptorError::Reader(source) => Self::Reader(source),
            CollectionDescriptorError::Get { collection, source } => {
                Self::DescriptorGet { collection, source }
            }
            CollectionDescriptorError::Identity { expected, actual } => {
                Self::DescriptorIdentity { expected, actual }
            }
            CollectionDescriptorError::Invalid { collection, source } => {
                Self::InvalidDescriptor { collection, source }
            }
        }
    }
}

impl<RecordsError, ReaderError, MetaError, GetError> fmt::Display
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
where
    RecordsError: fmt::Display,
    ReaderError: fmt::Display,
    MetaError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(source) => source.fmt(f),
            Self::Discovery(source) => source.fmt(f),
            Self::ExactTicket(source) => write!(f, "invalid exact ticket: {source}"),
            Self::DescriptorGet { collection, source } => write!(
                f,
                "failed to fetch admitted collection descriptor {}: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::InvalidDescriptor { collection, source } => write!(
                f,
                "admitted collection descriptor {} is invalid: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::DescriptorIdentity { expected, actual } => write!(
                f,
                "admitted collection descriptor bytes hash to {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::DescriptorMismatch { collection } => write!(
                f,
                "admitted collection descriptor {} does not match the facade descriptor",
                hex::encode_upper(collection.raw),
            ),
            Self::Reader(source) => write!(f, "failed to open collection blob view: {source}"),
            Self::CommitDataGet {
                commit,
                data,
                source,
            } => write!(
                f,
                "failed to fetch data {} for admitted commit {commit:X}: {source}",
                hex::encode_upper(data.raw),
            ),
            Self::InvalidCommitData { commit, source } => {
                write!(f, "admitted commit {commit:X} has invalid data: {source}")
            }
            Self::CommitMetadataGet {
                commit,
                metadata,
                source,
            } => write!(
                f,
                "failed to fetch metadata {} for admitted commit {commit:X}: {source}",
                hex::encode_upper(metadata.raw),
            ),
            Self::InvalidCommitMetadata {
                commit,
                metadata,
                source,
            } => write!(
                f,
                "admitted commit {commit:X} has invalid metadata {}: {source}",
                hex::encode_upper(metadata.raw),
            ),
            Self::InvalidCommitMetadataIdentity {
                commit,
                expected,
                actual,
            } => write!(
                f,
                "admitted commit {commit:X} metadata bytes hash to {} instead of signed {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::ResolutionConflict(source) => source.fmt(f),
            Self::Materialize(source) => source.fmt(f),
        }
    }
}

impl<RecordsError, ReaderError, MetaError, GetError> Error
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
where
    RecordsError: Error + 'static,
    ReaderError: Error + 'static,
    MetaError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::Discovery(source) => Some(source),
            Self::ExactTicket(source) => Some(source),
            Self::DescriptorGet { source, .. } => Some(source),
            Self::InvalidDescriptor { source, .. } => Some(source),
            Self::DescriptorIdentity { .. } | Self::DescriptorMismatch { .. } => None,
            Self::Reader(source) => Some(source),
            Self::CommitDataGet { source, .. } => Some(source),
            Self::InvalidCommitData { source, .. } => Some(source),
            Self::CommitMetadataGet { source, .. } => Some(source),
            Self::InvalidCommitMetadata { source, .. } => Some(source),
            Self::InvalidCommitMetadataIdentity { .. } => None,
            Self::ResolutionConflict(source) => Some(source),
            Self::Materialize(source) => Some(source),
        }
    }
}

#[cfg(any(test, not(feature = "parallel")))]
fn validate_unique_commit_dependencies<E, ValidateData, ValidateMetadata>(
    commits: &[CollectionCommit],
    mut validate_data: ValidateData,
    mut validate_metadata: ValidateMetadata,
) -> Result<BTreeMap<CollectionData, Blob<SimpleArchive>>, E>
where
    ValidateData: FnMut(&CollectionCommit) -> Result<Blob<SimpleArchive>, E>,
    ValidateMetadata: FnMut(&CollectionCommit) -> Result<(), E>,
{
    let mut known = BTreeMap::new();
    let mut validated_metadata = BTreeSet::new();
    for commit in commits {
        let data = commit.data();
        if let std::collections::btree_map::Entry::Vacant(entry) = known.entry(data) {
            entry.insert(validate_data(commit)?);
        }
        if validated_metadata.insert(commit.metadata()) {
            validate_metadata(commit)?;
        }
    }
    Ok(known)
}

#[cfg(feature = "parallel")]
struct FetchedCommitData {
    commit: CollectionCommit,
    data: CollectionData,
    blob: Blob<SimpleArchive>,
}

/// Fetch dependencies through the possibly non-`Sync` reader on the caller
/// thread in serial data-then-metadata order, then parallelize concrete data
/// checks. Error replay by intrinsic commit keeps data before metadata and
/// prevents a later prefetched failure from outranking an earlier failure.
#[cfg(feature = "parallel")]
fn validate_unique_commit_dependencies_parallel<
    E,
    FetchData,
    FetchMetadata,
    ValidateMetadata,
    MapDataError,
>(
    commits: &[CollectionCommit],
    descriptor: &Fragment,
    mut fetch_data: FetchData,
    mut fetch_metadata: FetchMetadata,
    mut validate_metadata: ValidateMetadata,
    mut map_data_error: MapDataError,
) -> Result<BTreeMap<CollectionData, Blob<SimpleArchive>>, E>
where
    FetchData: FnMut(&CollectionCommit) -> Result<Blob<SimpleArchive>, E>,
    FetchMetadata: FnMut(&CollectionCommit) -> Result<Blob<SimpleArchive>, E>,
    ValidateMetadata: FnMut(&CollectionCommit, &Blob<SimpleArchive>) -> Result<(), E>,
    MapDataError: FnMut(&CollectionCommit, SimpleArchiveUnionValidationError) -> E,
{
    use rayon::prelude::*;

    if let [commit] = commits {
        let data = commit.data();
        let data_blob = fetch_data(commit)?;
        simplearchive_union::validate_commit(descriptor, commit, &data_blob)
            .map_err(|error| map_data_error(commit, error))?;
        let metadata_blob = fetch_metadata(commit)?;
        validate_metadata(commit, &metadata_blob)?;
        return Ok(BTreeMap::from([(data, data_blob)]));
    }

    let mut seen_data = BTreeSet::new();
    let mut seen_metadata = BTreeSet::new();
    let mut fetched = Vec::with_capacity(commits.len());
    let mut dependency_error = None;
    for commit in commits {
        if seen_data.insert(commit.data()) {
            match fetch_data(commit) {
                Ok(blob) => fetched.push(FetchedCommitData {
                    commit: *commit,
                    data: commit.data(),
                    blob,
                }),
                Err(error) => {
                    dependency_error = Some((commit.id(), error));
                    break;
                }
            }
        }
        if seen_metadata.insert(commit.metadata()) {
            match fetch_metadata(commit) {
                Ok(blob) => {
                    if let Err(error) = validate_metadata(commit, &blob) {
                        dependency_error = Some((commit.id(), error));
                        break;
                    }
                }
                Err(error) => {
                    dependency_error = Some((commit.id(), error));
                    break;
                }
            }
        }
    }

    let mut fetched = fetched
        .into_par_iter()
        .map(|fetched| {
            let validation =
                simplearchive_union::validate_commit(descriptor, &fetched.commit, &fetched.blob);
            (fetched, validation)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .peekable();

    let mut known = BTreeMap::new();
    for commit in commits {
        if fetched
            .peek()
            .is_some_and(|(fetched, _)| fetched.commit.id() == commit.id())
        {
            let (fetched, validation) = fetched.next().expect("peeked commit data");
            validation.map_err(|error| map_data_error(commit, error))?;
            known.insert(fetched.data, fetched.blob);
        }
        if dependency_error
            .as_ref()
            .is_some_and(|(failed_commit, _)| *failed_commit == commit.id())
        {
            return Err(dependency_error.take().expect("matched dependency error").1);
        }
    }
    Ok(known)
}

fn validate_generic_descriptor(facts: &TribleSet) -> Result<VerifyingKey, RecordDecodeError> {
    descriptor::entity(facts)?;
    descriptor::representation(facts)?;
    descriptor::recipe(facts)?;
    match (descriptor::name(facts)?, descriptor::source(facts)?) {
        (Some(_), None) | (None, Some(_)) => {}
        (None, None) => {
            return Err(RecordDecodeError::MissingField(
                "collection_name or collection_source",
            ));
        }
        (Some(_), Some(_)) => {
            return Err(RecordDecodeError::RepeatedField(
                "collection anchor (name/source)",
            ));
        }
    }
    descriptor::authority(facts)
}

fn register_collection<S>(
    store: &mut S,
    descriptor: Fragment,
) -> Result<CollectionHandle, CollectionRegistrationError<S::PutError, S::OfferError>>
where
    S: BlobStorePut + ArtifactOfferStore,
{
    validate_generic_descriptor(descriptor.facts())
        .map_err(CollectionRegistrationError::InvalidDescriptor)?;
    let name = descriptor::name(descriptor.facts())
        .map_err(CollectionRegistrationError::InvalidDescriptor)?;
    let (_, facts, _, mut blobs) = descriptor.into_parts();
    if let Some(name) = name {
        let reader = blobs
            .reader()
            .expect("MemoryBlobStore::reader is infallible");
        let valid = reader
            .get::<Blob<UTF8String>, UTF8String>(name)
            .ok()
            .filter(|blob| Blob::<UTF8String>::new(blob.bytes.clone()).get_handle() == name)
            .and_then(|blob| blob.try_from_blob::<anybytes::View<str>>().ok())
            .is_some();
        if !valid {
            return Err(CollectionRegistrationError::InvalidAttachment {
                role: "name",
                artifact: name.transmute(),
            });
        }
    }
    let mut embedded: Vec<Blob<UnknownBlob>> = blobs
        .reader()
        .expect("MemoryBlobStore::reader is infallible")
        .into_iter()
        .map(|(_, blob)| blob)
        .collect();
    embedded.sort_unstable_by_key(|blob| blob.get_handle().raw);

    // Registration is the only operation which still owns descriptor
    // attachment bytes. Capture and advertise the complete closure now;
    // later handle-only commits can re-offer the descriptor archive but cannot
    // reconstruct its UTF-8 name or description blobs from bare facts.
    let mut capture = OfferCapture::new(store);
    for blob in embedded {
        capture
            .put::<UnknownBlob, _>(blob)
            .map_err(CollectionRegistrationError::DependencyPut)?;
    }
    let collection = capture
        .put::<SimpleArchive, _>(facts)
        .map_err(CollectionRegistrationError::DependencyPut)?;
    if let Err(source) = capture.offer_pending() {
        return Err(CollectionRegistrationError::Offer {
            source,
            artifacts: capture.pending().collect(),
        });
    }
    Ok(collection)
}

struct LoadedCollectionDescriptor {
    fragment: Fragment,
    authority: VerifyingKey,
}

fn load_collection_descriptor<S>(
    store: &mut S,
    collection: CollectionHandle,
) -> Result<
    LoadedCollectionDescriptor,
    CollectionDescriptorError<S::ReaderError, <S::Reader as BlobStoreGet>::GetError<Infallible>>,
>
where
    S: BlobStore,
{
    let reader = store.reader().map_err(CollectionDescriptorError::Reader)?;
    let descriptor_blob: Blob<SimpleArchive> = reader
        .get(collection)
        .map_err(|source| CollectionDescriptorError::Get { collection, source })?;
    let descriptor_blob = Blob::<SimpleArchive>::new(descriptor_blob.bytes.clone());
    let actual = descriptor_blob.get_handle();
    if actual != collection {
        return Err(CollectionDescriptorError::Identity {
            expected: collection,
            actual,
        });
    }
    let facts =
        <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(descriptor_blob)
            .map_err(|source| CollectionDescriptorError::Invalid {
                collection,
                source: RecordDecodeError::from(source),
            })?;
    let authority = validate_generic_descriptor(&facts)
        .map_err(|source| CollectionDescriptorError::Invalid { collection, source })?;
    Ok(LoadedCollectionDescriptor {
        fragment: Fragment::from(facts),
        authority,
    })
}

fn admitted_subjects_at(
    authority: VerifyingKey,
    collection: CollectionHandle,
    instant: hifitime::Epoch,
    presentations: &[CapabilityPresentation],
) -> Result<BTreeSet<Inline<ED25519PublicKey>>, CollectionAdmissionError> {
    let atom = CapabilityAtom::new(
        CapabilityAction::new(ACTION_WRITE),
        CapabilityResource::from(collection),
    );
    let mut admitted = BTreeSet::from([Inline::new(authority.to_bytes())]);
    for (presentation_index, presentation) in presentations.iter().enumerate() {
        presentation
            .bundle()
            .verify(
                authority,
                instant,
                presentation.expected_leaf(),
                CapabilityRequest::new(atom, CapabilityMode::Invoke),
            )
            .map_err(|source| CollectionAdmissionError {
                presentation: presentation_index,
                expected_leaf: presentation.expected_leaf(),
                source,
            })?;
        admitted.insert(Inline::new(presentation.expected_leaf().to_bytes()));
    }
    Ok(admitted)
}

fn discover_admitted_commits_at<S>(
    store: &mut S,
    collection: CollectionHandle,
    presentations: &[CapabilityPresentation],
    instant: hifitime::Epoch,
) -> Result<
    (Fragment, DiscoveredCollectionRecords, CollectionTicket),
    CollectionTicketError<
        S::RecordsError,
        S::ReaderError,
        <S::Reader as BlobStoreGet>::GetError<Infallible>,
    >,
>
where
    S: BlobStore + CollectionStore,
{
    let loaded =
        load_collection_descriptor(store, collection).map_err(CollectionTicketError::Descriptor)?;
    let admitted = admitted_subjects_at(loaded.authority, collection, instant, presentations)
        .map_err(CollectionTicketError::Admission)?;
    let discovered = discover_collection_records_authorized(store, collection, |subject| {
        admitted.contains(subject)
    })
    .map_err(CollectionTicketError::Discovery)?;
    let ticket = CollectionTicket::from_canonical(collection, discovered.commits().to_vec());
    Ok((loaded.fragment, discovered, ticket))
}

fn widen_preparation_error<PutError, InsertError>(
    error: simplearchive_union::PreparationError,
) -> PublicationError<PutError, InsertError> {
    match error {
        PublicationError::Validation(source) => PublicationError::Validation(source),
        PublicationError::InvalidMetadata(source) => PublicationError::InvalidMetadata(source),
        PublicationError::DependencyPut(never) => match never {},
        PublicationError::RecordInsert(never) => match never {},
        PublicationError::MergeInputAbsent { role, data } => {
            PublicationError::MergeInputAbsent { role, data }
        }
    }
}

/// Ergonomic collection operations implemented directly by the backing store.
///
/// The trait carries no state and is blanket-implemented. Its purpose is only
/// method syntax: collections remain plain descriptor handles and stores
/// retain their native ownership, flushing, and closing APIs.
pub trait CollectionStoreExt: BlobStore + CollectionStore + ArtifactOfferStore + Sized {
    /// Register and advertise one self-contained descriptor, returning the
    /// handle produced by the store for its canonical facts.
    fn collection(
        &mut self,
        descriptor: Fragment,
    ) -> Result<
        CollectionHandle,
        CollectionRegistrationError<
            <Self as BlobStorePut>::PutError,
            <Self as ArtifactOfferStore>::OfferError,
        >,
    > {
        register_collection(self, descriptor)
    }

    /// Publish one signed fragment into an already registered collection.
    ///
    /// This performs no capability check. Local storage is a grow-only claim
    /// ledger; authority is applied only by [`ticket`](Self::ticket) and
    /// [`snapshot`](Self::snapshot). The descriptor is fetched and
    /// exact-validated before dependencies are staged, and the signed record is
    /// inserted last without an implicit durability flush.
    fn commit(
        &mut self,
        collection: CollectionHandle,
        signing_key: &SigningKey,
        fragment: Fragment,
    ) -> Result<
        CollectionCommit,
        CollectionCommitError<
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
            <Self as BlobStorePut>::PutError,
            OfferCaptureInsertError<
                <Self as ArtifactOfferStore>::OfferError,
                <Self as CollectionStore>::InsertError,
            >,
        >,
    > {
        let loaded = load_collection_descriptor(self, collection)
            .map_err(CollectionCommitError::Descriptor)?;
        let prepared = simplearchive_union::prepare_fragment_commit(&loaded.fragment, fragment)
            .map_err(|source| {
                CollectionCommitError::Publication(widen_preparation_error(source))
            })?;
        prepared
            .stage_for(self, collection, signing_key)
            .map_err(CollectionCommitError::Publication)?
            .finalize()
            .map_err(CollectionCommitError::Publication)
    }

    /// Discover one canonical admitted commit frontier.
    ///
    /// The descriptor authority is always admitted directly. Every delegated
    /// writer must be named by an explicitly supplied proof for exact
    /// [`ACTION_WRITE`] on `collection`; an invalid supplied proof fails the
    /// whole operation rather than silently changing its meaning.
    fn ticket(
        &mut self,
        collection: CollectionHandle,
        presentations: &[CapabilityPresentation],
    ) -> Result<
        CollectionTicket,
        CollectionTicketError<
            <Self as CollectionStore>::RecordsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    > {
        let (_, _, ticket) =
            discover_admitted_commits_at(self, collection, presentations, clock::epoch_now())?;
        Ok(ticket)
    }

    /// Capture one coherent known-prefix fact, ticket, and reader snapshot.
    fn snapshot(
        &mut self,
        collection: CollectionHandle,
        presentations: &[CapabilityPresentation],
    ) -> Result<
        CollectionSnapshot<<Self as BlobStore>::Reader>,
        CollectionMaterializationError<
            <Self as CollectionStore>::RecordsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreMeta>::MetaError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        <Self as BlobStore>::Reader: BlobStoreMeta,
    {
        let (descriptor, discovered, ticket) =
            discover_admitted_commits_at(self, collection, presentations, clock::epoch_now())
                .map_err(CollectionMaterializationError::from)?;
        snapshot_from_observation(self, &descriptor, discovered, ticket)
    }

    /// Replay one already-admitted exact ticket against this store.
    ///
    /// Unlike [`snapshot`](Self::snapshot), this performs no capability
    /// discovery. Every complete commit in `ticket` must byte-match one
    /// resident, strictly verified record for the ticket's descriptor. Other
    /// commits are inert; same-descriptor merge records may still accelerate
    /// the physical union.
    fn materialize(
        &mut self,
        ticket: &CollectionTicket,
    ) -> Result<
        TribleSet,
        CollectionMaterializationError<
            <Self as CollectionStore>::RecordsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreMeta>::MetaError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        <Self as BlobStore>::Reader: BlobStoreMeta,
    {
        let collection = ticket.collection();
        let descriptor = load_collection_descriptor(self, collection)
            .map_err(CollectionMaterializationError::from)?
            .fragment;
        let discovered = if ticket.is_empty() {
            DiscoveredCollectionRecords::default()
        } else {
            let requested = ticket
                .commits()
                .iter()
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>();
            let discovered =
                discover_collection_records_for_collection_ticket(self, &requested, collection)
                    .map_err(CollectionMaterializationError::Discovery)?;
            validate_exact_ticket(&discovered, ticket.commits())
                .map_err(CollectionMaterializationError::ExactTicket)?;
            discovered
        };
        snapshot_from_observation(self, &descriptor, discovered, ticket.clone())
            .map(CollectionSnapshot::into_facts)
    }
}

impl<S> CollectionStoreExt for S where S: BlobStore + CollectionStore + ArtifactOfferStore {}

/// Materialize one already-discovered exact commit frontier.
///
/// Ordinary admitted snapshots and exact-ticket collection kinds use this
/// single validator so descriptor, mandatory dependency, merge-cover, and
/// reader-snapshot semantics cannot drift apart.
pub(crate) fn snapshot_from_observation<S>(
    storage: &mut S,
    descriptor: &Fragment,
    discovered: DiscoveredCollectionRecords,
    ticket: CollectionTicket,
) -> Result<
    CollectionSnapshot<S::Reader>,
    CollectionMaterializationError<
        S::RecordsError,
        S::ReaderError,
        <S::Reader as BlobStoreMeta>::MetaError,
        <S::Reader as BlobStoreGet>::GetError<Infallible>,
    >,
>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    let collection = ticket.collection();
    let commits = ticket.commits();
    let admitted: BTreeSet<_> = commits.iter().map(CollectionCommit::id).collect();

    let reader = storage
        .reader()
        .map_err(CollectionMaterializationError::Reader)?;

    if commits.is_empty() {
        return Ok(CollectionSnapshot {
            facts: TribleSet::new(),
            ticket,
            reader,
        });
    }

    // The descriptor handle is the collection identity. Once an admitted
    // commit makes this collection nonempty, its descriptor is mandatory
    // ground truth just like the signed data and metadata below. Fetch by
    // the exact handle, recompute the identity rather than trusting a
    // cached handle, decode the canonical archive, and bind it back to the
    // facade's expected semantics before interpreting any element.
    let descriptor_blob: Blob<SimpleArchive> = reader
        .get(collection)
        .map_err(|source| CollectionMaterializationError::DescriptorGet { collection, source })?;
    let descriptor_blob = Blob::<SimpleArchive>::new(descriptor_blob.bytes.clone());
    let actual_descriptor = descriptor_blob.get_handle();
    if actual_descriptor != collection {
        return Err(CollectionMaterializationError::DescriptorIdentity {
            expected: collection,
            actual: actual_descriptor,
        });
    }
    let decoded_descriptor =
        <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(descriptor_blob)
            .map_err(|source| CollectionMaterializationError::InvalidDescriptor {
                collection,
                source: RecordDecodeError::from(source),
            })?;
    if decoded_descriptor != *descriptor.facts() {
        return Err(CollectionMaterializationError::DescriptorMismatch { collection });
    }

    // Authenticate and exact-validate every mandatory leaf first. Commit
    // signatures were verified individually during discovery. Blob
    // validation is instead keyed by content identity: several distinct
    // signed commits may name the same data or metadata, and one reader
    // snapshot cannot give that handle different bytes. Fetch and
    // canonical-check each distinct handle once while retaining every
    // commit as provenance and every data handle as a semantic root.
    // Authenticated data remains available for fallback; derived scratch
    // values below have a shorter, use-counted lifetime.
    let fetch_data = |claim: &CollectionCommit| {
        let data = claim.data();
        reader
            .get(Handle::<SimpleArchive>::from_hash(data))
            .map_err(|source| CollectionMaterializationError::CommitDataGet {
                commit: claim.id(),
                data,
                source,
            })
    };
    let fetch_metadata = |claim: &CollectionCommit| {
        let metadata = claim.metadata();
        reader
            .get(metadata)
            .map_err(|source| CollectionMaterializationError::CommitMetadataGet {
                commit: claim.id(),
                metadata,
                source,
            })
    };
    let validate_metadata = |claim: &CollectionCommit, metadata_blob: &Blob<SimpleArchive>| {
        let metadata = claim.metadata();
        simplearchive_union::validate_element(metadata_blob).map_err(|source| {
            CollectionMaterializationError::InvalidCommitMetadata {
                commit: claim.id(),
                metadata,
                source,
            }
        })?;
        let actual_metadata = Blob::<SimpleArchive>::new(metadata_blob.bytes.clone()).get_handle();
        if actual_metadata != metadata {
            return Err(
                CollectionMaterializationError::InvalidCommitMetadataIdentity {
                    commit: claim.id(),
                    expected: metadata,
                    actual: actual_metadata,
                },
            );
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    let mut known = validate_unique_commit_dependencies_parallel(
        commits,
        descriptor,
        fetch_data,
        fetch_metadata,
        validate_metadata,
        |claim, source| CollectionMaterializationError::InvalidCommitData {
            commit: claim.id(),
            source,
        },
    )?;
    #[cfg(not(feature = "parallel"))]
    let mut known = validate_unique_commit_dependencies(
        commits,
        |claim| {
            let data_blob = fetch_data(claim)?;
            simplearchive_union::validate_commit(descriptor, claim, &data_blob).map_err(
                |source| CollectionMaterializationError::InvalidCommitData {
                    commit: claim.id(),
                    source,
                },
            )?;
            Ok(data_blob)
        },
        |claim| {
            let metadata_blob = fetch_metadata(claim)?;
            validate_metadata(claim, &metadata_blob)
        },
    )?;
    let roots: BTreeSet<_> = known.keys().copied().collect();

    // Unsigned merges are useful only when they can contribute to a
    // resident physical cover. Walk backwards from resident result hashes
    // first, then validate that finite subgraph forwards from authenticated
    // leaves. This retains the resolver's nonresident-intermediate model:
    // an intermediate need not be stored when its computed bytes feed a
    // later resident result.
    let merges: Vec<_> = discovered
        .merges()
        .iter()
        .filter(|claim| claim.collection() == collection)
        .copied()
        .collect();
    let mut producers = BTreeMap::<CollectionData, Vec<usize>>::new();
    for (index, claim) in merges.iter().enumerate() {
        producers.entry(claim.result()).or_default().push(index);
    }

    let mut resident_results = BTreeSet::new();
    let mut reverse_seen = BTreeSet::new();
    let mut reverse_queue = VecDeque::new();
    for result in producers.keys().copied() {
        let resident = known.contains_key(&result)
            || matches!(
                reader.metadata(Handle::<SimpleArchive>::from_hash(result)),
                Ok(Some(_))
            );
        if resident {
            resident_results.insert(result);
            if reverse_seen.insert(result) {
                reverse_queue.push_back(result);
            }
        }
    }

    let mut candidates = BTreeSet::new();
    while let Some(result) = reverse_queue.pop_front() {
        let Some(indices) = producers.get(&result) else {
            continue;
        };
        for &index in indices {
            candidates.insert(index);
            let (low, high) = merges[index].inputs();
            for input in [low, high] {
                if reverse_seen.insert(input) {
                    reverse_queue.push_back(input);
                }
            }
        }
    }

    // Index each candidate by its missing inputs. Newly admitted results
    // wake only their direct dependants, avoiding repeated global scans as
    // a deep LSM cover becomes grounded.
    let mut missing = vec![u8::MAX; merges.len()];
    let mut waiters = BTreeMap::<CollectionData, Vec<usize>>::new();
    let mut remaining_uses = BTreeMap::<CollectionData, usize>::new();
    let mut ready = BTreeSet::new();
    for &index in &candidates {
        let claim = &merges[index];
        let (low, high) = claim.inputs();
        *remaining_uses.entry(low).or_default() += 1;
        if high != low {
            *remaining_uses.entry(high).or_default() += 1;
        }
        let mut count = 0u8;
        if !known.contains_key(&low) {
            waiters.entry(low).or_default().push(index);
            count += 1;
        }
        if high != low && !known.contains_key(&high) {
            waiters.entry(high).or_default().push(index);
            count += 1;
        }
        missing[index] = count;
        if count == 0 {
            ready.insert((claim.id(), index));
        }
    }

    let mut accepted_merges = BTreeSet::new();
    let mut expected_hashes = BTreeMap::<(CollectionData, CollectionData), CollectionData>::new();
    while let Some((_, index)) = ready.pop_first() {
        let claim = &merges[index];
        let (low, high) = claim.inputs();
        let pair = (low, high);

        let mut joined = None;
        let expected_data = if let Some(expected) = expected_hashes.get(&pair).copied() {
            Some(expected)
        } else {
            match (known.get(&low), known.get(&high)) {
                (Some(low_blob), Some(high_blob)) => {
                    match simplearchive_union::join(low_blob, high_blob) {
                        Ok(value) => {
                            let expected = Handle::<SimpleArchive>::to_hash(value.get_handle());
                            expected_hashes.insert(pair, expected);
                            joined = Some(value);
                            Some(expected)
                        }
                        // `known` contains only exact-validated canonical
                        // elements, so this is a defensive invariant guard.
                        Err(_) => None,
                    }
                }
                _ => None,
            }
        };

        if let Some(expected_data) = expected_data.filter(|data| *data == claim.result()) {
            let retain_result = remaining_uses
                .get(&expected_data)
                .copied()
                .unwrap_or_default()
                > 0;
            let inserted = if known.contains_key(&expected_data) || !retain_result {
                false
            } else {
                if joined.is_none() {
                    joined = match (known.get(&low), known.get(&high)) {
                        (Some(low_blob), Some(high_blob)) => {
                            simplearchive_union::join(low_blob, high_blob).ok()
                        }
                        _ => None,
                    };
                }
                if let Some(value) = joined {
                    debug_assert_eq!(
                        Handle::<SimpleArchive>::to_hash(value.get_handle()),
                        expected_data
                    );
                    known.insert(expected_data, value);
                    true
                } else {
                    false
                }
            };

            // A terminal result needs no retained bytes: its canonical
            // hash already validates the equation, and a selected
            // physical artifact is exact-checked later. A nonterminal
            // result is accepted only when its bytes remain available to
            // validate its dependants.
            if known.contains_key(&expected_data) || !retain_result {
                accepted_merges.insert(claim.id());
                if inserted {
                    for dependent in waiters.remove(&expected_data).unwrap_or_default() {
                        debug_assert!(missing[dependent] > 0 && missing[dependent] <= 2);
                        missing[dependent] -= 1;
                        if missing[dependent] == 0 {
                            ready.insert((merges[dependent].id(), dependent));
                        }
                    }
                };
            }
        }

        // Computed intermediate bytes live only until their final
        // candidate consumer has run. This keeps a balanced LSM to one
        // live derived frontier instead of retaining the dataset once per
        // level. Authenticated leaves stay cached for mandatory fallback.
        for input in [Some(low), (high != low).then_some(high)]
            .into_iter()
            .flatten()
        {
            let uses = remaining_uses
                .get_mut(&input)
                .expect("candidate inputs have reference counts");
            debug_assert!(*uses > 0);
            *uses -= 1;
            if *uses == 0 && !roots.contains(&input) {
                known.remove(&input);
            }
        }
    }

    // A plain collection facade owns one root collection and derives
    // nothing, so it declares no lineage.
    let resolution =
        resolve_collection_semantics(&discovered, &BTreeMap::new(), &admitted, |request| {
            Ok::<CollectionClaimValidation<()>, Infallible>(match request {
                CollectionValidationRequest::Commit { .. } => CollectionClaimValidation::Accepted,
                CollectionValidationRequest::Merge { claim, .. }
                    if accepted_merges.contains(&claim.id()) =>
                {
                    CollectionClaimValidation::Accepted
                }
                CollectionValidationRequest::Merge { .. }
                | CollectionValidationRequest::Derive { .. } => CollectionClaimValidation::Pending,
            })
        });

    let resolution = match resolution {
        Ok(resolution) => resolution,
        Err(CollectionResolutionError::Validation { source, .. }) => match source {},
        Err(CollectionResolutionError::Conflict(source)) => {
            return Err(CollectionMaterializationError::ResolutionConflict(source));
        }
    };

    // Optional physical results are accelerators, not failure authority.
    // Offer only metadata-resident, semantically accepted results to the
    // cover algorithm, then exact-check just the members it selects. A bad
    // candidate is removed and the cover is recomputed, so corrupt or
    // stale artifacts fall back to another cover without forcing eager
    // reads of the full historical LSM. Mandatory root bytes already
    // failed loud above.
    let semantics = resolution.semantics();
    let mut resident = roots.clone();
    for data in semantics.members(collection).into_iter().flatten().copied() {
        if resident_results.contains(&data) {
            resident.insert(data);
        }
    }

    let mut selected = BTreeMap::<CollectionData, Blob<SimpleArchive>>::new();
    let cover = loop {
        let cover = collection_physical_cover(semantics, collection, &resident);
        if !cover.missing.is_empty() {
            return Err(CollectionMaterializationError::Materialize(
                MaterializationError::Missing {
                    obligations: cover.missing,
                },
            ));
        }

        selected.retain(|data, _| cover.cover.contains(data));
        let mut rejected = Vec::new();
        for data in cover.cover.iter().copied() {
            if roots.contains(&data) || selected.contains_key(&data) {
                continue;
            }
            let handle = Handle::<SimpleArchive>::from_hash(data);
            let actual: Result<Blob<SimpleArchive>, _> = reader.get(handle);
            match actual {
                Ok(actual) => {
                    let actual = Blob::<SimpleArchive>::new(actual.bytes.clone());
                    let actual_data = Handle::<SimpleArchive>::to_hash(actual.get_handle());
                    if actual_data == data && simplearchive_union::validate_element(&actual).is_ok()
                    {
                        selected.insert(data, actual);
                    } else {
                        rejected.push(data);
                    }
                }
                Err(_) => rejected.push(data),
            }
        }

        if rejected.is_empty() {
            break cover.cover;
        }
        for data in rejected {
            resident.remove(&data);
        }
    };

    let mut members = Vec::with_capacity(cover.len());
    for data in cover {
        let blob = if roots.contains(&data) {
            known
                .get(&data)
                .expect("authenticated root bytes stay cached")
        } else {
            selected
                .get(&data)
                .expect("optional cover members were exact-validated")
        };
        members.push((data, blob));
    }
    let facts = match members.as_slice() {
        [(data, blob)] => (*blob).clone().try_from_blob().map_err(|source| {
            CollectionMaterializationError::Materialize(MaterializationError::InvalidElement {
                data: *data,
                source,
            })
        })?,
        _ => {
            // The union's handle is never asked for here — it is decoded
            // and dropped — so it is computed without one. Hashing 1.7 GB
            // to name a value this expression consumes is a fifth of the
            // merge that produced it.
            let union = simplearchive_union::join_many_bytes(members.iter().map(|(_, blob)| *blob))
                .map_err(|(index, source)| {
                    CollectionMaterializationError::Materialize(
                        MaterializationError::InvalidElement {
                            data: members[index].0,
                            source,
                        },
                    )
                })?;
            crate::blob::encodings::simplearchive::try_from_archive_bytes(union)
                .expect("join_many emits one canonical SimpleArchive")
        }
    };
    Ok(CollectionSnapshot {
        facts,
        ticket,
        reader,
    })
}
