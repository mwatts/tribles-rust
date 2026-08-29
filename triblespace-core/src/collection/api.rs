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
//! cover or snapshot is constructed. The descriptor authority's own commits
//! are admitted directly; delegated writers need an explicit proof for
//! [`ACTION_WRITE`] on that exact descriptor handle.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::utf8string::UTF8String;
use crate::blob::encodings::UnknownBlob;
use crate::blob::Blob;
use crate::capability::{
    CapabilityAction, CapabilityAtom, CapabilityMode, CapabilityProofBundle, CapabilityProofError,
    CapabilityRequest, CapabilityResource,
};
use crate::clock;
use crate::inline::encodings::ed25519::ED25519PublicKey;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::patch::{Blake3Merkle, IdentitySchema, PATCH};
use crate::repo::{
    ArtifactHandle, ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut,
    OfferCapture, OfferCaptureInsertError,
};
// Reach arrives here as a builder argument; only the tests name a
// particular one.
use crate::trible::{Fragment, TribleSet};

use super::discovery::{
    discover_collection_claims_for_cover, discover_collection_equations_for_cover, ExactCoverError,
};
use super::simplearchive_union::{
    self, MaterializationError, PublicationError, SimpleArchiveUnionValidationError,
};
use super::{
    collection_physical_cover, descriptor, discover_collection_records_authorized,
    resolve_collection_semantics_from_roots, CollectionClaimValidation, CollectionCommit,
    CollectionData, CollectionDiscoveryError, CollectionFunctionalConflict, CollectionHandle,
    CollectionResolutionError, CollectionStore, CollectionValidationRequest,
    DiscoveredCollectionRecords, RecordDecodeError, ACTION_WRITE,
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

/// One exact point in a collection lattice.
///
/// Members are content identities, not signatures. Several commits may attest
/// the same member with different authors or metadata without changing this
/// value. The private constructor makes a `Cover` an opaque result of
/// admission or validated collection algebra rather than a caller-forged set
/// of hashes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cover {
    collection: CollectionHandle,
    members: PATCH<32, IdentitySchema, (), Blake3Merkle>,
}

impl Cover {
    pub(crate) fn from_members(
        collection: CollectionHandle,
        members: impl IntoIterator<Item = CollectionData>,
    ) -> Self {
        Self {
            collection,
            members: PATCH::from_keys(members.into_iter().map(|member| member.raw)),
        }
    }

    pub(crate) fn from_patch(
        collection: CollectionHandle,
        members: PATCH<32, IdentitySchema, (), Blake3Merkle>,
    ) -> Self {
        Self {
            collection,
            members,
        }
    }

    /// Exact descriptor whose lattice contains these members.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Canonical member identities in ascending byte order.
    pub fn members(&self) -> impl ExactSizeIterator<Item = CollectionData> + '_ {
        self.members
            .iter_ordered()
            .map(|member| Inline::new(*member))
    }

    /// Whether this cover contains one exact member identity.
    pub fn contains(&self, member: CollectionData) -> bool {
        self.members.get(&member.raw).is_some()
    }

    /// Number of distinct collection members.
    pub fn len(&self) -> usize {
        self.members.len().min(usize::MAX as u64) as usize
    }

    /// Whether this is the lattice bottom.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Return the members added since an earlier observation.
    ///
    /// This is PATCH set difference over payload identities. A new signature
    /// or metadata archive for an existing member is provenance, not a data
    /// delta. Shrinking observations fail because additions-only maintenance
    /// would no longer be sound.
    pub fn additions_since(&self, previous: &Self) -> Result<Self, CoverAdvanceError> {
        if self.collection != previous.collection {
            return Err(CoverAdvanceError::DifferentCollection {
                previous: previous.collection,
                current: self.collection,
            });
        }
        let missing = previous.members.difference(&self.members);
        if let Some(member) = missing.iter_ordered().next() {
            return Err(CoverAdvanceError::ResetRequired {
                missing: Inline::new(*member),
            });
        }
        Ok(Self::from_patch(
            self.collection,
            self.members.difference(&previous.members),
        ))
    }
}

/// Failure to treat two covers as one additions-only continuation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoverAdvanceError {
    /// The two covers belong to different collection lattices.
    DifferentCollection {
        /// Earlier collection descriptor.
        previous: CollectionHandle,
        /// Current collection descriptor.
        current: CollectionHandle,
    },
    /// A member of the earlier cover is absent from the current cover.
    ResetRequired {
        /// First missing member in canonical content order.
        missing: CollectionData,
    },
}

impl fmt::Display for CoverAdvanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentCollection { previous, current } => write!(
                formatter,
                "cover collection {} differs from {}",
                hex::encode_upper(current.raw),
                hex::encode_upper(previous.raw),
            ),
            Self::ResetRequired { missing } => write!(
                formatter,
                "previous cover member {} is absent from the current observation; additions-only processing requires a reset",
                hex::encode_upper(missing.raw),
            ),
        }
    }
}

impl Error for CoverAdvanceError {}

/// One coherent known-prefix view of a scoped collection.
///
/// [`cover`](Self::cover) is the exact payload set materialized as
/// [`facts`](Self::facts).
/// [`reader`](Self::reader) is the blob-reader snapshot used to validate and
/// materialize those facts. The reader may contain physically available blobs
/// published after record discovery, but those blobs acquire no semantic role
/// unless their payloads are present in this snapshot's cover.
pub struct CollectionSnapshot<R> {
    facts: TribleSet,
    cover: Cover,
    reader: R,
}

impl<R> CollectionSnapshot<R> {
    /// Materialized union named by this snapshot's exact payload cover.
    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    /// Exact collection cover from which the facts were materialized.
    pub fn cover(&self) -> &Cover {
        &self.cover
    }

    /// Blob-reader snapshot used to validate and materialize the facts.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Consume the snapshot and return its materialized facts.
    pub fn into_facts(self) -> TribleSet {
        self.facts
    }

    /// Consume the snapshot into materialized facts, exact cover, and reader.
    pub fn into_parts(self) -> (TribleSet, Cover, R) {
        (self.facts, self.cover, self.reader)
    }
}

/// Failure to discover one exact admitted payload cover.
#[derive(Debug)]
pub enum CollectionCoverError<RecordsError, ReaderError, GetError> {
    /// The collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<ReaderError, GetError>),
    /// One explicitly supplied capability proof was invalid at this operation's
    /// single clock observation.
    Admission(CollectionAdmissionError),
    /// Target collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
}

impl<RecordsError, ReaderError, GetError> fmt::Display
    for CollectionCoverError<RecordsError, ReaderError, GetError>
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
    for CollectionCoverError<RecordsError, ReaderError, GetError>
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

/// Failure to publish one collection element into local storage.
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

/// Failure to materialize the complete value named by an opaque cover.
///
/// Every cover member is explicit ground truth, so its descriptor and data
/// fail loud. Signatures and metadata remain queryable provenance rather
/// than becoming coordinates of the payload lattice. Unsigned equations are
/// replaceable cache evidence: missing or invalid equations are omitted from
/// the resolved semantics and cannot hide an explicit cover member.
#[derive(Debug)]
pub enum CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError> {
    /// One explicitly supplied capability proof was invalid.
    Admission(CollectionAdmissionError),
    /// Native collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
    /// A supplied exact cover names the wrong collection descriptor.
    ExactCover(ExactCoverError),
    /// The cover's canonical descriptor blob could not be fetched.
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
    /// cover.
    DescriptorIdentity {
        /// Descriptor handle named by the cover.
        expected: CollectionHandle,
        /// Handle recomputed from the fetched bytes.
        actual: CollectionHandle,
    },
    /// The fetched canonical descriptor did not equal the descriptor expected
    /// by this facade.
    DescriptorMismatch {
        /// Descriptor handle named by this facade and cover.
        collection: CollectionHandle,
    },
    /// The blob reader could not be created after record discovery.
    Reader(ReaderError),
    /// A cover member's data blob could not be fetched.
    MemberGet {
        /// Exact payload identity.
        member: CollectionData,
        /// Backend fetch failure.
        source: GetError,
    },
    /// A cover member failed exact `SimpleArchive` collection
    /// validation.
    InvalidMember {
        /// Exact payload identity.
        member: CollectionData,
        /// Exact representation or identity diagnostic.
        source: SimpleArchiveUnionValidationError,
    },
    /// Positively validated equations contradicted operation functionality.
    ResolutionConflict(Box<CollectionFunctionalConflict>),
    /// The resolved semantic frontier could not be physically materialized.
    Materialize(MaterializationError<MetaError, GetError>),
}

impl<RecordsError, ReaderError, MetaError, GetError>
    From<CollectionCoverError<RecordsError, ReaderError, GetError>>
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
{
    fn from(source: CollectionCoverError<RecordsError, ReaderError, GetError>) -> Self {
        match source {
            CollectionCoverError::Descriptor(CollectionDescriptorError::Reader(source)) => {
                Self::Reader(source)
            }
            CollectionCoverError::Descriptor(CollectionDescriptorError::Get {
                collection,
                source,
            }) => Self::DescriptorGet { collection, source },
            CollectionCoverError::Descriptor(CollectionDescriptorError::Identity {
                expected,
                actual,
            }) => Self::DescriptorIdentity { expected, actual },
            CollectionCoverError::Descriptor(CollectionDescriptorError::Invalid {
                collection,
                source,
            }) => Self::InvalidDescriptor { collection, source },
            CollectionCoverError::Admission(source) => Self::Admission(source),
            CollectionCoverError::Discovery(source) => Self::Discovery(source),
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
            Self::ExactCover(source) => write!(f, "invalid exact cover: {source}"),
            Self::DescriptorGet { collection, source } => write!(
                f,
                "failed to fetch collection descriptor {}: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::InvalidDescriptor { collection, source } => write!(
                f,
                "collection descriptor {} is invalid: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::DescriptorIdentity { expected, actual } => write!(
                f,
                "collection descriptor bytes hash to {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::DescriptorMismatch { collection } => write!(
                f,
                "collection descriptor {} does not match the facade descriptor",
                hex::encode_upper(collection.raw),
            ),
            Self::Reader(source) => write!(f, "failed to open collection blob view: {source}"),
            Self::MemberGet { member, source } => write!(
                f,
                "failed to fetch cover member {}: {source}",
                hex::encode_upper(member.raw),
            ),
            Self::InvalidMember { member, source } => write!(
                f,
                "cover member {} is invalid: {source}",
                hex::encode_upper(member.raw),
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
            Self::ExactCover(source) => Some(source),
            Self::DescriptorGet { source, .. } => Some(source),
            Self::InvalidDescriptor { source, .. } => Some(source),
            Self::DescriptorIdentity { .. } | Self::DescriptorMismatch { .. } => None,
            Self::Reader(source) => Some(source),
            Self::MemberGet { source, .. } => Some(source),
            Self::InvalidMember { source, .. } => Some(source),
            Self::ResolutionConflict(source) => Some(source),
            Self::Materialize(source) => Some(source),
        }
    }
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

fn discover_admitted_cover_at<S>(
    store: &mut S,
    collection: CollectionHandle,
    presentations: &[CapabilityPresentation],
    instant: hifitime::Epoch,
) -> Result<
    (Fragment, DiscoveredCollectionRecords, Cover),
    CollectionCoverError<
        S::RecordsError,
        S::ReaderError,
        <S::Reader as BlobStoreGet>::GetError<Infallible>,
    >,
>
where
    S: BlobStore + CollectionStore,
{
    let loaded =
        load_collection_descriptor(store, collection).map_err(CollectionCoverError::Descriptor)?;
    let admitted = admitted_subjects_at(loaded.authority, collection, instant, presentations)
        .map_err(CollectionCoverError::Admission)?;
    let discovered = discover_collection_records_authorized(store, collection, |subject| {
        admitted.contains(subject)
    })
    .map_err(CollectionCoverError::Discovery)?;
    let cover = Cover::from_members(
        collection,
        discovered.commits().iter().map(CollectionCommit::data),
    );
    Ok((loaded.fragment, discovered, cover))
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
    /// ledger; authority is applied only by [`cover`](Self::cover) and
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

    /// Discover one canonical admitted payload cover.
    ///
    /// The descriptor authority is always admitted directly. Every delegated
    /// writer must be named by an explicitly supplied proof for exact
    /// [`ACTION_WRITE`] on `collection`; an invalid supplied proof fails the
    /// whole operation rather than silently changing its meaning.
    fn cover(
        &mut self,
        collection: CollectionHandle,
        presentations: &[CapabilityPresentation],
    ) -> Result<
        Cover,
        CollectionCoverError<
            <Self as CollectionStore>::RecordsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    > {
        let (_, _, cover) =
            discover_admitted_cover_at(self, collection, presentations, clock::epoch_now())?;
        Ok(cover)
    }

    /// Capture one coherent known-prefix fact, cover, and reader snapshot.
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
        let (descriptor, discovered, cover) =
            discover_admitted_cover_at(self, collection, presentations, clock::epoch_now())
                .map_err(CollectionMaterializationError::from)?;
        snapshot_from_observation(self, &descriptor, discovered, cover)
    }

    /// Replay one opaque exact cover against this store.
    ///
    /// Unlike [`snapshot`](Self::snapshot), this performs no capability or
    /// provenance discovery. Only the descriptor and payload bytes named by
    /// `cover` are mandatory; resident commits and metadata are unnecessary.
    /// Same-descriptor merge records may still accelerate the physical union.
    fn materialize(
        &mut self,
        cover: &Cover,
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
        let collection = cover.collection();
        let descriptor = load_collection_descriptor(self, collection)
            .map_err(CollectionMaterializationError::from)?
            .fragment;
        let discovered = if cover.is_empty() {
            DiscoveredCollectionRecords::default()
        } else {
            discover_collection_equations_for_cover(self, cover)
                .map_err(CollectionMaterializationError::Discovery)?
        };
        snapshot_from_observation(self, &descriptor, discovered, cover.clone())
            .map(CollectionSnapshot::into_facts)
    }

    /// Return every strictly verified provenance claim currently known for the
    /// members of an opaque cover.
    ///
    /// This query is intentionally broader than the admission event which
    /// minted the cover: later authorship or metadata claims over the same
    /// payloads are visible without changing the cover itself.
    fn claims(
        &mut self,
        cover: &Cover,
    ) -> Result<Vec<CollectionCommit>, CollectionDiscoveryError<Self::RecordsError>> {
        let discovered = discover_collection_claims_for_cover(self, cover)?;
        Ok(discovered.commits().to_vec())
    }
}

impl<S> CollectionStoreExt for S where S: BlobStore + CollectionStore + ArtifactOfferStore {}

/// Materialize one already-discovered exact payload cover.
///
/// Ordinary admitted snapshots, opaque replay, and exact-derived collection kinds use this
/// single validator so descriptor, mandatory member, merge-cover, and
/// reader-snapshot semantics cannot drift apart. Signed claims may have
/// established a cover originally, but replay does not require them: metadata
/// and authors remain optional provenance over the same payload member.
pub(crate) fn snapshot_from_observation<S>(
    storage: &mut S,
    descriptor: &Fragment,
    discovered: DiscoveredCollectionRecords,
    cover: Cover,
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
    let collection = cover.collection();
    let reader = storage
        .reader()
        .map_err(CollectionMaterializationError::Reader)?;

    if cover.is_empty() {
        return Ok(CollectionSnapshot {
            facts: TribleSet::new(),
            cover,
            reader,
        });
    }

    // The descriptor handle is the collection identity. A nonempty cover makes
    // its descriptor and named payloads mandatory ground truth. Fetch by
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

    // Fetch and validate each cover payload exactly once. Claims, authorship,
    // and metadata are intentionally absent: replay remains valid even when no
    // provenance record or metadata blob is resident.
    let mut known = BTreeMap::new();
    for member in cover.members() {
        let blob = reader
            .get(Handle::<SimpleArchive>::from_hash(member))
            .map_err(|source| CollectionMaterializationError::MemberGet { member, source })?;
        simplearchive_union::validate_member(descriptor, collection, member, &blob)
            .map_err(|source| CollectionMaterializationError::InvalidMember { member, source })?;
        known.insert(member, blob);
    }
    let roots: BTreeSet<_> = cover.members().collect();
    let explicit_roots: BTreeSet<_> = roots
        .iter()
        .copied()
        .map(|data| (collection, data))
        .collect();

    // Unsigned merges are useful only when they can contribute to a
    // resident physical cover. Walk backwards from resident result hashes
    // first, then validate that finite subgraph forwards from the explicit
    // cover roots. This retains the resolver's nonresident-intermediate model:
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

    // Index each candidate by its missing inputs. Newly validated results
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
    let resolution = resolve_collection_semantics_from_roots(
        &discovered,
        &BTreeMap::new(),
        &explicit_roots,
        |request| {
            Ok::<CollectionClaimValidation<()>, Infallible>(match request {
                CollectionValidationRequest::Merge { claim, .. }
                    if accepted_merges.contains(&claim.id()) =>
                {
                    CollectionClaimValidation::Accepted
                }
                CollectionValidationRequest::Commit { .. }
                | CollectionValidationRequest::Merge { .. }
                | CollectionValidationRequest::Derive { .. } => CollectionClaimValidation::Pending,
            })
        },
    );

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
    let physical_cover = loop {
        let candidate = collection_physical_cover(semantics, collection, &resident);
        if !candidate.missing.is_empty() {
            return Err(CollectionMaterializationError::Materialize(
                MaterializationError::Missing {
                    obligations: candidate.missing,
                },
            ));
        }

        selected.retain(|data, _| candidate.cover.contains(data));
        let mut rejected = Vec::new();
        for data in candidate.cover.iter().copied() {
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
            break candidate.cover;
        }
        for data in rejected {
            resident.remove(&data);
        }
    };

    let mut members = Vec::with_capacity(physical_cover.len());
    for data in physical_cover {
        let blob = if roots.contains(&data) {
            known
                .get(&data)
                .expect("explicit cover-root bytes stay cached")
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
        cover,
        reader,
    })
}
