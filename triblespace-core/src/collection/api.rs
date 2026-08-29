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
//! are admitted directly; delegated writers are admitted when the store holds
//! a valid proof for [`ACTION_WRITE`] on that exact descriptor handle.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::utf8string::UTF8String;
use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, IntoBlob};
use crate::capability::{
    CapabilityAction, CapabilityAtom, CapabilityMode, CapabilityProof, CapabilityProofBundle,
    CapabilityRequest, CapabilityResource,
};
use crate::clock;
use crate::inline::encodings::ed25519::ED25519PublicKey;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::metadata::Describe;
use crate::patch::{Blake3Merkle, IdentitySchema, PATCH};
use crate::repo::{
    ArtifactHandle, ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut,
    CapabilityProofStore, OfferCapture, OfferCaptureInsertError,
};
// Reach arrives here as a builder argument; only the tests name a
// particular one.
use crate::trible::{Fragment, TribleSet};

use super::discovery::{
    discover_collection_claims_for_cover, discover_collection_equations_for_cover, ExactCoverError,
};
use super::simplearchive_union::FactViewError;
use super::{
    collection_physical_cover, descriptor, discover_collection_records_authorized,
    resolve_collection_semantics_from_roots, Collection, CollectionArtifact,
    CollectionClaimValidation, CollectionCommit, CollectionData, CollectionDiscoveryError,
    CollectionEncoding, CollectionFunctionalConflict, CollectionHandle, CollectionOperationError,
    CollectionResolutionError, CollectionStore, CollectionTypeError, CollectionValidationRequest,
    CoverAttachment, DiscoveredCollectionRecords, RecordDecodeError, TryFromCover, ACTION_WRITE,
};

/// Failure to discover the resident capability evidence used for collection
/// admission.
///
/// Individual candidate proofs are untrusted evidence: a missing claim,
/// malformed claim, invalid signature, wrong request, or expired validity
/// interval merely grants nothing. This error is reserved for failure of the
/// proof-store observation itself, where silently returning a smaller cover
/// would confuse unavailable evidence with negative authorization.
#[derive(Debug)]
pub enum CollectionEvidenceDiscoveryError<ProofsError, ReaderError> {
    /// The store could not enumerate its resident proof snapshot.
    Proofs(ProofsError),
    /// The blob-reader snapshot used for all candidate claim closures could
    /// not be created.
    Reader(ReaderError),
}

impl<ProofsError, ReaderError> fmt::Display
    for CollectionEvidenceDiscoveryError<ProofsError, ReaderError>
where
    ProofsError: fmt::Display,
    ReaderError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proofs(source) => {
                write!(
                    formatter,
                    "failed to enumerate resident capability proofs: {source}"
                )
            }
            Self::Reader(source) => {
                write!(
                    formatter,
                    "failed to open capability-claim blob view: {source}"
                )
            }
        }
    }
}

impl<ProofsError, ReaderError> Error for CollectionEvidenceDiscoveryError<ProofsError, ReaderError>
where
    ProofsError: Error + 'static,
    ReaderError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Proofs(source) => Some(source),
            Self::Reader(source) => Some(source),
        }
    }
}

/// Failure while reading and structurally validating a collection descriptor
/// by its content identity.
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
            Self::Invalid { source, .. } => Some(source),
        }
    }
}

/// Failure to register and advertise a self-contained collection descriptor.
#[derive(Debug)]
pub enum CollectionRegistrationError<PutError, OfferError> {
    /// The descriptor names another encoding or invalid encoding context.
    WrongType(CollectionTypeError),
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
            Self::WrongType(source) => write!(formatter, "wrong collection type: {source}"),
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
            Self::WrongType(source) => Some(source),
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
/// admission or validated collection operations rather than a caller-forged
/// set of hashes.
pub struct Cover<L: CollectionEncoding> {
    collection: Collection<L>,
    members: PATCH<32, IdentitySchema, (), Blake3Merkle>,
}

impl<L: CollectionEncoding> Cover<L> {
    pub(crate) fn from_members(
        collection: Collection<L>,
        members: impl IntoIterator<Item = Inline<Handle<L>>>,
    ) -> Self {
        Self {
            collection,
            members: PATCH::from_keys(members.into_iter().map(|member| member.raw)),
        }
    }

    pub(crate) fn from_data(
        collection: Collection<L>,
        members: impl IntoIterator<Item = CollectionData>,
    ) -> Self {
        Self {
            collection,
            members: PATCH::from_keys(members.into_iter().map(|member| member.raw)),
        }
    }

    pub(crate) fn from_patch(
        collection: Collection<L>,
        members: PATCH<32, IdentitySchema, (), Blake3Merkle>,
    ) -> Self {
        Self {
            collection,
            members,
        }
    }

    /// Exact descriptor whose lattice contains these members.
    pub const fn collection(&self) -> Collection<L> {
        self.collection
    }

    /// Canonical member identities in ascending byte order.
    pub fn members(&self) -> impl ExactSizeIterator<Item = Inline<Handle<L>>> + '_ {
        self.members
            .iter_ordered()
            .map(|member| Inline::new(*member))
    }

    /// Whether this cover contains one exact member identity.
    pub fn contains(&self, member: Inline<Handle<L>>) -> bool {
        self.members.get(&member.raw).is_some()
    }

    pub(crate) fn data_members(&self) -> impl ExactSizeIterator<Item = CollectionData> + '_ {
        self.members
            .iter_ordered()
            .map(|member| Inline::new(*member))
    }

    pub(crate) fn contains_data(&self, member: CollectionData) -> bool {
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
                previous: previous.collection.handle(),
                current: self.collection.handle(),
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

impl<L: CollectionEncoding> Clone for Cover<L> {
    fn clone(&self) -> Self {
        Self {
            collection: self.collection,
            members: self.members.clone(),
        }
    }
}

impl<L: CollectionEncoding> fmt::Debug for Cover<L> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Cover")
            .field("collection", &self.collection)
            .field("members", &self.data_members().collect::<Vec<_>>())
            .finish()
    }
}

impl<L: CollectionEncoding> PartialEq for Cover<L> {
    fn eq(&self, other: &Self) -> bool {
        self.collection == other.collection && self.members == other.members
    }
}

impl<L: CollectionEncoding> Eq for Cover<L> {}

/// Typed exact cover of the canonical fact collection.
pub type FactCover = Cover<SimpleArchive>;

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
/// [`cover`](Self::cover) is the exact payload set reconstructed as
/// [`value`](Self::value).
/// [`reader`](Self::reader) is the blob-reader snapshot used to validate and
/// materialize those facts. The reader may contain physically available blobs
/// published after record discovery, but those blobs acquire no semantic role
/// unless their payloads are present in this snapshot's cover.
pub struct Snapshot<L: CollectionEncoding, V, R> {
    value: V,
    cover: Cover<L>,
    reader: R,
}

impl<L: CollectionEncoding, V, R> Snapshot<L, V, R> {
    /// Materialized logical value named by this snapshot's exact cover.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Exact collection cover from which the value was materialized.
    pub fn cover(&self) -> &Cover<L> {
        &self.cover
    }

    /// Blob-reader snapshot used to validate and reconstruct the logical value.
    pub fn reader(&self) -> &R {
        &self.reader
    }

    /// Consume the snapshot and return its materialized value.
    pub fn into_value(self) -> V {
        self.value
    }

    /// Consume the snapshot into materialized value, exact cover, and reader.
    pub fn into_parts(self) -> (V, Cover<L>, R) {
        (self.value, self.cover, self.reader)
    }
}

impl<R> Snapshot<SimpleArchive, TribleSet, R> {
    /// Materialized fact union named by this snapshot's exact payload cover.
    pub fn facts(&self) -> &TribleSet {
        self.value()
    }

    /// Consume the snapshot and return its materialized fact union.
    pub fn into_facts(self) -> TribleSet {
        self.into_value()
    }
}

/// Snapshot of the canonical SimpleArchive fact-union lattice.
pub type FactSnapshot<R> = Snapshot<SimpleArchive, TribleSet, R>;

/// Failure to discover one exact admitted payload cover.
#[derive(Debug)]
pub enum CollectionCoverError<RecordsError, ProofsError, ReaderError, GetError> {
    /// The collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<ReaderError, GetError>),
    /// The resident capability-proof observation could not be completed.
    Evidence(CollectionEvidenceDiscoveryError<ProofsError, ReaderError>),
    /// Target collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
}

impl<RecordsError, ProofsError, ReaderError, GetError> fmt::Display
    for CollectionCoverError<RecordsError, ProofsError, ReaderError, GetError>
where
    RecordsError: fmt::Display,
    ProofsError: fmt::Display,
    ReaderError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(source) => source.fmt(formatter),
            Self::Evidence(source) => source.fmt(formatter),
            Self::Discovery(source) => source.fmt(formatter),
        }
    }
}

impl<RecordsError, ProofsError, ReaderError, GetError> Error
    for CollectionCoverError<RecordsError, ProofsError, ReaderError, GetError>
where
    RecordsError: Error + 'static,
    ProofsError: Error + 'static,
    ReaderError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(source) => Some(source),
            Self::Evidence(source) => Some(source),
            Self::Discovery(source) => Some(source),
        }
    }
}

/// Failure to publish one collection element into local storage.
#[derive(Debug)]
pub enum CollectionCommitError<ReaderError, GetError, PutError, InsertError> {
    /// The named collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<ReaderError, GetError>),
    /// The encoded member is not a canonical element of the collection lattice.
    InvalidMember(CollectionOperationError),
    /// A described attachment, member, or metadata archive could not be stored.
    DependencyPut(PutError),
    /// The signed visibility record could not be inserted after its dependencies.
    RecordInsert(InsertError),
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
            Self::InvalidMember(source) => write!(formatter, "invalid collection member: {source}"),
            Self::DependencyPut(source) => {
                write!(formatter, "failed to store collection dependency: {source}")
            }
            Self::RecordInsert(source) => {
                write!(formatter, "failed to insert collection commit: {source}")
            }
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
            Self::InvalidMember(source) => Some(source),
            Self::DependencyPut(source) => Some(source),
            Self::RecordInsert(source) => Some(source),
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
pub enum CollectionMaterializationError<
    RecordsError,
    ReaderError,
    GetError,
    ViewError,
    EvidenceError = Infallible,
> {
    /// Resident admission evidence could not be observed.
    Evidence(EvidenceError),
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
    /// The blob reader could not be created after record discovery.
    Reader(ReaderError),
    /// A cover member's data blob could not be fetched.
    MemberGet {
        /// Exact payload identity.
        member: CollectionData,
        /// Backend fetch failure.
        source: GetError,
    },
    /// A cover member failed exact encoding validation.
    InvalidMember {
        /// Exact payload identity.
        member: CollectionData,
        /// Exact representation or identity diagnostic.
        source: CollectionOperationError,
    },
    /// Positively validated equations contradicted operation functionality.
    ResolutionConflict(Box<CollectionFunctionalConflict>),
    /// No resident physical cover spans every semantic obligation.
    Missing {
        /// Uncovered members of the collection's semantic frontier.
        obligations: BTreeSet<CollectionData>,
    },
    /// The selected exact attachment could not form the requested logical view.
    View(ViewError),
}

/// Materialization failure for the canonical SimpleArchive fact view.
pub type FactMaterializationError<RecordsError, ReaderError, GetError> =
    CollectionMaterializationError<RecordsError, ReaderError, GetError, FactViewError>;

/// Snapshot failure including typed discovery of resident authorization
/// evidence.
pub type CollectionSnapshotError<RecordsError, ProofsError, ReaderError, GetError, ViewError> =
    CollectionMaterializationError<
        RecordsError,
        ReaderError,
        GetError,
        ViewError,
        CollectionEvidenceDiscoveryError<ProofsError, ReaderError>,
    >;

impl<RecordsError, ProofsError, ReaderError, GetError, ViewError>
    From<CollectionCoverError<RecordsError, ProofsError, ReaderError, GetError>>
    for CollectionSnapshotError<RecordsError, ProofsError, ReaderError, GetError, ViewError>
{
    fn from(
        source: CollectionCoverError<RecordsError, ProofsError, ReaderError, GetError>,
    ) -> Self {
        match source {
            CollectionCoverError::Descriptor(CollectionDescriptorError::Reader(source)) => {
                Self::Reader(source)
            }
            CollectionCoverError::Descriptor(CollectionDescriptorError::Get {
                collection,
                source,
            }) => Self::DescriptorGet { collection, source },
            CollectionCoverError::Descriptor(CollectionDescriptorError::Invalid {
                collection,
                source,
            }) => Self::InvalidDescriptor { collection, source },
            CollectionCoverError::Evidence(source) => Self::Evidence(source),
            CollectionCoverError::Discovery(source) => Self::Discovery(source),
        }
    }
}

impl<RecordsError, ReaderError, GetError, ViewError, EvidenceError>
    From<CollectionDescriptorError<ReaderError, GetError>>
    for CollectionMaterializationError<
        RecordsError,
        ReaderError,
        GetError,
        ViewError,
        EvidenceError,
    >
{
    fn from(source: CollectionDescriptorError<ReaderError, GetError>) -> Self {
        match source {
            CollectionDescriptorError::Reader(source) => Self::Reader(source),
            CollectionDescriptorError::Get { collection, source } => {
                Self::DescriptorGet { collection, source }
            }
            CollectionDescriptorError::Invalid { collection, source } => {
                Self::InvalidDescriptor { collection, source }
            }
        }
    }
}

impl<RecordsError, ReaderError, GetError, ViewError, EvidenceError> fmt::Display
    for CollectionMaterializationError<
        RecordsError,
        ReaderError,
        GetError,
        ViewError,
        EvidenceError,
    >
where
    RecordsError: fmt::Display,
    ReaderError: fmt::Display,
    GetError: fmt::Display,
    ViewError: fmt::Display,
    EvidenceError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evidence(source) => source.fmt(f),
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
            Self::Missing { obligations } => write!(
                f,
                "{} semantic frontier obligation(s) have no resident physical cover",
                obligations.len(),
            ),
            Self::View(source) => source.fmt(f),
        }
    }
}

impl<RecordsError, ReaderError, GetError, ViewError, EvidenceError> Error
    for CollectionMaterializationError<
        RecordsError,
        ReaderError,
        GetError,
        ViewError,
        EvidenceError,
    >
where
    RecordsError: Error + 'static,
    ReaderError: Error + 'static,
    GetError: Error + 'static,
    ViewError: Error + 'static,
    EvidenceError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Evidence(source) => Some(source),
            Self::Discovery(source) => Some(source),
            Self::ExactCover(source) => Some(source),
            Self::DescriptorGet { source, .. } => Some(source),
            Self::InvalidDescriptor { source, .. } => Some(source),
            Self::Reader(source) => Some(source),
            Self::MemberGet { source, .. } => Some(source),
            Self::InvalidMember { source, .. } => Some(source),
            Self::ResolutionConflict(source) => Some(source),
            Self::Missing { .. } => None,
            Self::View(source) => Some(source),
        }
    }
}

fn validate_generic_descriptor(facts: &TribleSet) -> Result<VerifyingKey, RecordDecodeError> {
    descriptor::validate(facts)
}

fn register_collection<S>(
    store: &mut S,
    descriptor: Fragment,
) -> Result<(), CollectionRegistrationError<S::PutError, S::OfferError>>
where
    S: BlobStorePut + ArtifactOfferStore,
{
    let name = descriptor::name(descriptor.facts())
        .expect("typed collection descriptor was structurally validated");
    let (_, facts, _, mut blobs) = descriptor.into_parts();
    if let Some(name) = name {
        let reader = blobs
            .reader()
            .expect("MemoryBlobStore::reader is infallible");
        let valid = reader
            .get::<Blob<UTF8String>, UTF8String>(name)
            .ok()
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
    capture
        .put::<SimpleArchive, _>(facts)
        .map_err(CollectionRegistrationError::DependencyPut)?;
    if let Err(source) = capture.offer_pending() {
        return Err(CollectionRegistrationError::Offer {
            source,
            artifacts: capture.pending().collect(),
        });
    }
    Ok(())
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

fn admitted_subjects_at<R>(
    reader: &R,
    authority: VerifyingKey,
    collection: CollectionHandle,
    instant: hifitime::Epoch,
    proofs: impl IntoIterator<Item = CapabilityProof>,
) -> BTreeSet<Inline<ED25519PublicKey>>
where
    R: BlobStoreGet,
{
    let atom = CapabilityAtom::new(
        CapabilityAction::new(ACTION_WRITE),
        CapabilityResource::from(collection),
    );
    let request = CapabilityRequest::new(atom, CapabilityMode::Invoke);
    let mut admitted = BTreeSet::from([Inline::new(authority.to_bytes())]);
    for proof in proofs {
        if proof.root_key() != authority {
            continue;
        }

        let claims: Option<Vec<Blob<SimpleArchive>>> = proof
            .claim_handles()
            .map(|claim| reader.get::<Blob<SimpleArchive>, SimpleArchive>(claim).ok())
            .collect();
        let Some(claims) = claims else {
            continue;
        };
        let leaf = proof.leaf_key();
        let bundle = CapabilityProofBundle::new(proof, claims);
        if bundle.verify(authority, instant, leaf, request).is_ok() {
            admitted.insert(Inline::new(leaf.to_bytes()));
        }
    }
    admitted
}

fn discover_admitted_cover_at<S, L>(
    store: &mut S,
    collection: Collection<L>,
    instant: hifitime::Epoch,
) -> Result<
    (Fragment, DiscoveredCollectionRecords, Cover<L>),
    CollectionCoverError<
        S::RecordsError,
        S::ProofsError,
        S::ReaderError,
        <S::Reader as BlobStoreGet>::GetError<Infallible>,
    >,
>
where
    S: BlobStore + CapabilityProofStore + CollectionStore,
    L: CollectionEncoding,
{
    let loaded = load_collection_descriptor(store, collection.handle())
        .map_err(CollectionCoverError::Descriptor)?;
    // Proof publication stores every claim blob before its proof record.
    // Freeze proof membership first, then open one reader: every observed
    // proof's prior claim closure is consequently visible in that reader.
    let proofs = store
        .proofs()
        .map_err(CollectionEvidenceDiscoveryError::Proofs)
        .map_err(CollectionCoverError::Evidence)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(CollectionEvidenceDiscoveryError::Proofs)
        .map_err(CollectionCoverError::Evidence)?;
    let reader = store
        .reader()
        .map_err(CollectionEvidenceDiscoveryError::Reader)
        .map_err(CollectionCoverError::Evidence)?;
    let admitted = admitted_subjects_at(
        &reader,
        loaded.authority,
        collection.handle(),
        instant,
        proofs,
    );
    let discovered =
        discover_collection_records_authorized(store, collection.handle(), |subject| {
            admitted.contains(subject)
        })
        .map_err(CollectionCoverError::Discovery)?;
    let cover = Cover::from_data(
        collection,
        discovered.commits().iter().map(CollectionCommit::data),
    );
    Ok((loaded.fragment, discovered, cover))
}

/// Ergonomic collection operations implemented directly by the backing store.
///
/// The trait carries no state and is blanket-implemented. Its purpose is only
/// method syntax: collections remain plain descriptor handles and stores
/// retain their native ownership, flushing, and closing APIs.
pub trait CollectionStoreExt: BlobStore + CollectionStore + ArtifactOfferStore + Sized {
    /// Register and advertise one self-contained descriptor, returning the
    /// handle produced by the store for its canonical facts.
    fn collection<L>(
        &mut self,
        descriptor: Fragment,
    ) -> Result<
        Collection<L>,
        CollectionRegistrationError<
            <Self as BlobStorePut>::PutError,
            <Self as ArtifactOfferStore>::OfferError,
        >,
    >
    where
        L: CollectionEncoding,
    {
        let collection =
            Collection::<L>::from_descriptor(&descriptor).map_err(|source| match source {
                CollectionTypeError::Malformed(source) => {
                    CollectionRegistrationError::InvalidDescriptor(source)
                }
                source => CollectionRegistrationError::WrongType(source),
            })?;
        register_collection(self, descriptor)?;
        Ok(collection)
    }

    /// Publish one signed fragment into an already registered collection.
    ///
    /// This performs no capability check. Local storage is a grow-only claim
    /// ledger; authority is applied only by [`cover`](Self::cover) and
    /// [`snapshot`](Self::snapshot). The descriptor is fetched and structurally
    /// decoded before dependencies are staged, and the signed record is
    /// inserted last without an implicit durability flush.
    fn commit<L, T>(
        &mut self,
        collection: Collection<L>,
        signing_key: &SigningKey,
        value: T,
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
    >
    where
        L: CollectionEncoding,
        T: Describe + IntoBlob<L>,
        Self::Reader: BlobStoreMeta,
    {
        let loaded = load_collection_descriptor(self, collection.handle())
            .map_err(CollectionCommitError::Descriptor)?;

        // Description happens before the value is consumed by its encoding.
        // A Fragment description deliberately retains its shared blob store,
        // so data and metadata attachments travel through this one generic
        // path without an encoding-specific side channel.
        let description = value.describe();
        let data_root = value.to_blob();
        let (_, metadata_facts, _, mut described_blobs) = description.into_parts();
        let metadata: Blob<SimpleArchive> = metadata_facts.to_blob();
        let mut attachments: Vec<Blob<UnknownBlob>> = described_blobs
            .reader()
            .expect("MemoryBlobStore::reader is infallible")
            .into_iter()
            .map(|(_, blob)| blob)
            .collect();
        attachments.sort_unstable_by_key(|blob| blob.get_handle().raw);

        let mut store = OfferCapture::new(self);
        for blob in attachments {
            store
                .put::<UnknownBlob, _>(blob)
                .map_err(CollectionCommitError::DependencyPut)?;
        }
        // Described attachments are now resident, so a Merkle member can
        // attach its complete closure without an encoding-specific commit
        // path. Publication is still repeated through the artifact to enforce
        // dependency-before-root ordering at the semantic record boundary.
        let reader = store.reader().map_err(|source| {
            CollectionCommitError::Descriptor(CollectionDescriptorError::Reader(source))
        })?;
        let data = L::attach_member(&loaded.fragment, data_root, &reader)
            .map_err(CollectionCommitError::InvalidMember)?;
        drop(reader);

        data.publish(&mut store)
            .map_err(CollectionCommitError::DependencyPut)?;
        let data_handle = data.root().get_handle();
        let metadata = store
            .put::<SimpleArchive, _>(metadata)
            .map_err(CollectionCommitError::DependencyPut)?;
        let commit = CollectionCommit::sign(
            signing_key,
            collection.handle(),
            Handle::<L>::to_hash(data_handle),
            metadata,
        );
        store
            .insert(super::CollectionRecord::Commit(commit))
            .map_err(CollectionCommitError::RecordInsert)?;
        Ok(commit)
    }

    /// Discover one canonical admitted payload cover.
    ///
    /// The descriptor authority is always admitted directly. Every resident
    /// proof rooted at that authority is considered for exact [`ACTION_WRITE`]
    /// on `collection`. Invalid, irrelevant, expired, or incomplete candidate
    /// evidence grants nothing; failure to observe the proof store remains an
    /// error.
    fn cover<L>(
        &mut self,
        collection: Collection<L>,
    ) -> Result<
        Cover<L>,
        CollectionCoverError<
            <Self as CollectionStore>::RecordsError,
            <Self as CapabilityProofStore>::ProofsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        Self: CapabilityProofStore,
        L: CollectionEncoding,
    {
        let (_, _, cover) = discover_admitted_cover_at(self, collection, clock::epoch_now())?;
        Ok(cover)
    }

    /// Capture one coherent known-prefix logical value, cover, and reader snapshot.
    fn snapshot<V, L>(
        &mut self,
        collection: Collection<L>,
    ) -> Result<
        Snapshot<L, V, <Self as BlobStore>::Reader>,
        CollectionSnapshotError<
            <Self as CollectionStore>::RecordsError,
            <Self as CapabilityProofStore>::ProofsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
            V::Error,
        >,
    >
    where
        Self: CapabilityProofStore,
        <Self as BlobStore>::Reader: BlobStoreMeta,
        L: CollectionEncoding,
        V: TryFromCover<L>,
    {
        let (descriptor, discovered, cover) =
            discover_admitted_cover_at(self, collection, clock::epoch_now())
                .map_err(CollectionMaterializationError::from)?;
        snapshot_from_observation(self, &descriptor, discovered, cover)
    }

    /// Replay one opaque exact cover against this store.
    ///
    /// Unlike [`snapshot`](Self::snapshot), this performs no capability or
    /// provenance discovery. Only the descriptor and payload bytes named by
    /// `cover` are mandatory; resident commits and metadata are unnecessary.
    /// Same-descriptor merge records may still accelerate the physical union.
    fn materialize<V, L>(
        &mut self,
        cover: &Cover<L>,
    ) -> Result<
        V,
        CollectionMaterializationError<
            <Self as CollectionStore>::RecordsError,
            <Self as BlobStore>::ReaderError,
            <<Self as BlobStore>::Reader as BlobStoreGet>::GetError<Infallible>,
            V::Error,
        >,
    >
    where
        <Self as BlobStore>::Reader: BlobStoreMeta,
        L: CollectionEncoding,
        V: TryFromCover<L>,
    {
        let collection = cover.collection();
        let descriptor = load_collection_descriptor(self, collection.handle())
            .map_err(CollectionMaterializationError::from)?
            .fragment;
        let discovered = if cover.is_empty() {
            DiscoveredCollectionRecords::default()
        } else {
            discover_collection_equations_for_cover(self, cover)
                .map_err(CollectionMaterializationError::Discovery)?
        };
        snapshot_from_observation(self, &descriptor, discovered, cover.clone())
            .map(Snapshot::into_value)
    }

    /// Return every strictly verified provenance claim currently known for the
    /// members of an opaque cover.
    ///
    /// This query is intentionally broader than the admission event which
    /// minted the cover: later authorship or metadata claims over the same
    /// payloads are visible without changing the cover itself.
    fn claims<L>(
        &mut self,
        cover: &Cover<L>,
    ) -> Result<Vec<CollectionCommit>, CollectionDiscoveryError<Self::RecordsError>>
    where
        L: CollectionEncoding,
    {
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
pub(crate) fn snapshot_from_observation<S, L, V, EvidenceError>(
    storage: &mut S,
    descriptor: &Fragment,
    discovered: DiscoveredCollectionRecords,
    cover: Cover<L>,
) -> Result<
    Snapshot<L, V, S::Reader>,
    CollectionMaterializationError<
        S::RecordsError,
        S::ReaderError,
        <S::Reader as BlobStoreGet>::GetError<Infallible>,
        V::Error,
        EvidenceError,
    >,
>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
    L: CollectionEncoding,
    V: TryFromCover<L>,
{
    let collection = cover.collection().handle();
    let reader = storage
        .reader()
        .map_err(CollectionMaterializationError::Reader)?;

    if cover.is_empty() {
        let snapshot_cover = cover.clone();
        let value = V::try_from_cover(CoverAttachment::from_parts(cover, Vec::new()))
            .map_err(CollectionMaterializationError::View)?;
        return Ok(Snapshot {
            value,
            cover: snapshot_cover,
            reader,
        });
    }

    // The descriptor was bound to this typed collection before this coherent
    // reader was opened. Its immutable content address already binds these
    // facts; another fetch cannot strengthen that proof.

    // Fetch and validate each cover payload exactly once. Claims, authorship,
    // and metadata are intentionally absent: replay remains valid even when no
    // provenance record or metadata blob is resident.
    let mut known = BTreeMap::new();
    for member_handle in cover.members() {
        let member = Handle::<L>::to_hash(member_handle);
        let blob = reader
            .get(member_handle)
            .map_err(|source| CollectionMaterializationError::MemberGet { member, source })?;
        let artifact = L::attach_member(descriptor, blob, &reader)
            .map_err(|source| CollectionMaterializationError::InvalidMember { member, source })?;
        known.insert(member, artifact);
    }
    let roots: BTreeSet<_> = cover.data_members().collect();
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
            || matches!(reader.metadata(Handle::<L>::from_hash(result)), Ok(Some(_)));
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
                (Some(low_artifact), Some(high_artifact)) => {
                    match L::join_members(descriptor, low_artifact, high_artifact) {
                        Ok(value) => {
                            let expected = Handle::<L>::to_hash(value.root().get_handle());
                            expected_hashes.insert(pair, expected);
                            joined = Some(value);
                            Some(expected)
                        }
                        // `known` contains only structurally validated members,
                        // so this is a defensive invariant guard around the
                        // encoding's join implementation.
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
                        (Some(low_artifact), Some(high_artifact)) => {
                            L::join_members(descriptor, low_artifact, high_artifact).ok()
                        }
                        _ => None,
                    };
                }
                if let Some(value) = joined {
                    debug_assert_eq!(
                        Handle::<L>::to_hash(value.root().get_handle()),
                        expected_data,
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

    let mut selected = BTreeMap::<CollectionData, L::Artifact>::new();
    let physical_cover = loop {
        let candidate = collection_physical_cover(semantics, collection, &resident);
        if !candidate.missing.is_empty() {
            return Err(CollectionMaterializationError::Missing {
                obligations: candidate.missing,
            });
        }

        selected.retain(|data, _| candidate.cover.contains(data));
        let mut rejected = Vec::new();
        for data in candidate.cover.iter().copied() {
            if roots.contains(&data) || selected.contains_key(&data) {
                continue;
            }
            let handle = Handle::<L>::from_hash(data);
            let root: Result<Blob<L>, _> = reader.get(handle);
            match root {
                Ok(root) => match L::attach_member(descriptor, root, &reader) {
                    Ok(actual) => {
                        selected.insert(data, actual);
                    }
                    Err(_) => rejected.push(data),
                },
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
                .expect("selected optional cover members stay cached")
        };
        members.push((Handle::<L>::from_hash(data), (*blob).clone()));
    }
    let snapshot_cover = cover.clone();
    let value = V::try_from_cover(CoverAttachment::from_parts(cover, members))
        .map_err(CollectionMaterializationError::View)?;
    Ok(Snapshot {
        value,
        cover: snapshot_cover,
        reader,
    })
}
