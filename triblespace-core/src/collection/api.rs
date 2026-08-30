//! Store-centric publication and immutable reads for canonical collections.
//!
//! A collection is its descriptor handle. Registering a descriptor through
//! [`CollectionStoreExt::register_collection`] stores its complete attachment
//! closure, while later operations take only that handle. The
//! descriptor's mandatory policy is therefore part of the collection's
//! identity rather than a caller-supplied policy which could disagree with it.
//!
//! Local publication is deliberately unconditional: a store may record any
//! structurally valid, strictly signed commit. Authority is enforced when a
//! an admitted cover or logical value is constructed. The descriptor policy
//! independently governs READ and WRITE admission.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, IntoBlob};
use crate::capability::{
    capability_quorum_authorizes, capability_quorum_observation_valid_through, CapabilityAction,
    CapabilityAtom, CapabilityMode, CapabilityProof, CapabilityProofBundle, CapabilityRequest,
    CapabilityResource,
};
use crate::clock;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::patch::{Blake3Merkle, IdentitySchema, PATCH};
use crate::repo::{BlobStoreGet, BlobStoreMeta, BlobStorePut, CapabilityProofRead, SnapshotSource};
use crate::trible::{Fragment, TribleSet};

use super::discovery::{
    discover_collection_claims_for_cover, discover_collection_equations_for_cover, ExactCoverError,
};
use super::simplearchive_union::FactViewError;
use super::{
    collection_physical_cover, descriptor, discover_collection_records_authorized,
    resolve_collection_semantics_from_roots, Collection, CollectionClaimValidation,
    CollectionCommit, CollectionData, CollectionDiscoveryError, CollectionEncoding,
    CollectionFunctionalConflict, CollectionHandle, CollectionOperationError, CollectionRead,
    CollectionResolutionError, CollectionStore, CollectionTypeError, CollectionValidationRequest,
    DiscoveredCollectionRecords, RecordDecodeError, TryFromCover, TryFromCoverError, ACTION_READ,
    ACTION_WRITE,
};
use super::{AdmissionPolicy, CollectionMapping, CollectionPolicy};

/// Failure to discover the resident capability evidence used for collection
/// admission.
///
/// Individual candidate proofs are untrusted evidence: a missing claim,
/// malformed claim, invalid signature, wrong request, or expired validity
/// interval merely grants nothing. This error is reserved for failure of the
/// proof-store observation itself, where silently returning a smaller cover
/// would confuse unavailable evidence with negative authorization.
#[derive(Debug)]
pub enum CollectionEvidenceDiscoveryError<ProofsError> {
    /// The store could not enumerate its resident proof snapshot.
    Proofs(ProofsError),
}

impl<ProofsError> fmt::Display for CollectionEvidenceDiscoveryError<ProofsError>
where
    ProofsError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proofs(source) => {
                write!(
                    formatter,
                    "failed to enumerate resident capability proofs: {source}"
                )
            }
        }
    }
}

impl<ProofsError> Error for CollectionEvidenceDiscoveryError<ProofsError>
where
    ProofsError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Proofs(source) => Some(source),
        }
    }
}

/// Failure while reading and structurally validating a collection descriptor
/// by its content identity.
#[derive(Debug)]
pub enum CollectionDescriptorError<GetError> {
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

impl<GetError> fmt::Display for CollectionDescriptorError<GetError>
where
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

impl<GetError> Error for CollectionDescriptorError<GetError>
where
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Get { source, .. } => Some(source),
            Self::Invalid { source, .. } => Some(source),
        }
    }
}

/// Failure to decide whether one subject is currently admitted to a collection.
///
/// Descriptor failures mean that the collection's admission policy could not
/// be established. Evidence failures mean that the resident capability-proof
/// observation itself was unavailable; invalid or irrelevant individual
/// proofs merely fail to admit their subjects and are not errors.
#[derive(Debug)]
pub enum CollectionAdmissionError<ProofsError, GetError> {
    /// The collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<GetError>),
    /// The resident capability-proof observation could not be completed.
    Evidence(CollectionEvidenceDiscoveryError<ProofsError>),
}

impl<ProofsError, GetError> fmt::Display for CollectionAdmissionError<ProofsError, GetError>
where
    ProofsError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(source) => source.fmt(formatter),
            Self::Evidence(source) => source.fmt(formatter),
        }
    }
}

impl<ProofsError, GetError> Error for CollectionAdmissionError<ProofsError, GetError>
where
    ProofsError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(source) => Some(source),
            Self::Evidence(source) => Some(source),
        }
    }
}

/// Failure to register a self-contained collection descriptor.
#[derive(Debug)]
pub enum CollectionRegistrationError<PutError> {
    /// The descriptor names another encoding or invalid encoding context.
    WrongType(CollectionTypeError),
    /// Descriptor facts did not have the mandatory generic shape.
    InvalidDescriptor(RecordDecodeError),
    /// One attachment or the canonical descriptor archive could not be stored.
    DependencyPut(PutError),
}

impl<PutError> fmt::Display for CollectionRegistrationError<PutError>
where
    PutError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongType(source) => write!(formatter, "wrong collection type: {source}"),
            Self::InvalidDescriptor(source) => {
                write!(formatter, "invalid collection descriptor: {source}")
            }
            Self::DependencyPut(source) => {
                write!(
                    formatter,
                    "failed to store collection descriptor closure: {source}"
                )
            }
        }
    }
}

impl<PutError> Error for CollectionRegistrationError<PutError>
where
    PutError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::WrongType(source) => Some(source),
            Self::InvalidDescriptor(source) => Some(source),
            Self::DependencyPut(source) => Some(source),
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

/// Failure to discover one exact admitted payload cover.
#[derive(Debug)]
pub enum CollectionCoverError<RecordsError, ProofsError, GetError> {
    /// The collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<GetError>),
    /// The resident capability-proof observation could not be completed.
    Evidence(CollectionEvidenceDiscoveryError<ProofsError>),
    /// Target collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
}

impl<RecordsError, ProofsError, GetError> fmt::Display
    for CollectionCoverError<RecordsError, ProofsError, GetError>
where
    RecordsError: fmt::Display,
    ProofsError: fmt::Display,
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

impl<RecordsError, ProofsError, GetError> Error
    for CollectionCoverError<RecordsError, ProofsError, GetError>
where
    RecordsError: Error + 'static,
    ProofsError: Error + 'static,
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
pub enum CollectionCommitError<PutError, InsertError> {
    /// A described attachment, member, or metadata archive could not be stored.
    DependencyPut(PutError),
    /// The signed visibility record could not be inserted after its dependencies.
    RecordInsert(InsertError),
}

impl<PutError, InsertError> fmt::Display for CollectionCommitError<PutError, InsertError>
where
    PutError: fmt::Display,
    InsertError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DependencyPut(source) => {
                write!(formatter, "failed to store collection dependency: {source}")
            }
            Self::RecordInsert(source) => {
                write!(formatter, "failed to insert collection commit: {source}")
            }
        }
    }
}

impl<PutError, InsertError> Error for CollectionCommitError<PutError, InsertError>
where
    PutError: Error + 'static,
    InsertError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
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
    /// The selected physical cover could not form the requested logical view.
    View(ViewError),
}

/// Materialization failure for the canonical SimpleArchive fact view.
pub type FactMaterializationError<RecordsError, GetError> =
    CollectionMaterializationError<RecordsError, GetError, FactViewError>;

/// Read failure including typed discovery of resident authorization evidence.
pub type CollectionReadError<RecordsError, ProofsError, GetError, ViewError> =
    CollectionMaterializationError<
        RecordsError,
        GetError,
        ViewError,
        CollectionEvidenceDiscoveryError<ProofsError>,
    >;

impl<RecordsError, ProofsError, GetError, ViewError>
    From<CollectionCoverError<RecordsError, ProofsError, GetError>>
    for CollectionReadError<RecordsError, ProofsError, GetError, ViewError>
{
    fn from(source: CollectionCoverError<RecordsError, ProofsError, GetError>) -> Self {
        match source {
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

impl<RecordsError, GetError, ViewError, EvidenceError> From<CollectionDescriptorError<GetError>>
    for CollectionMaterializationError<RecordsError, GetError, ViewError, EvidenceError>
{
    fn from(source: CollectionDescriptorError<GetError>) -> Self {
        match source {
            CollectionDescriptorError::Get { collection, source } => {
                Self::DescriptorGet { collection, source }
            }
            CollectionDescriptorError::Invalid { collection, source } => {
                Self::InvalidDescriptor { collection, source }
            }
        }
    }
}

impl<RecordsError, GetError, ViewError, EvidenceError> From<TryFromCoverError<GetError, ViewError>>
    for CollectionMaterializationError<RecordsError, GetError, ViewError, EvidenceError>
{
    fn from(source: TryFromCoverError<GetError, ViewError>) -> Self {
        match source {
            TryFromCoverError::MemberGet { member, source } => Self::MemberGet { member, source },
            TryFromCoverError::View(source) => Self::View(source),
        }
    }
}

impl<RecordsError, GetError, ViewError, EvidenceError> fmt::Display
    for CollectionMaterializationError<RecordsError, GetError, ViewError, EvidenceError>
where
    RecordsError: fmt::Display,
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

impl<RecordsError, GetError, ViewError, EvidenceError> Error
    for CollectionMaterializationError<RecordsError, GetError, ViewError, EvidenceError>
where
    RecordsError: Error + 'static,
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
            Self::MemberGet { source, .. } => Some(source),
            Self::InvalidMember { source, .. } => Some(source),
            Self::ResolutionConflict(source) => Some(source),
            Self::Missing { .. } => None,
            Self::View(source) => Some(source),
        }
    }
}

fn validate_generic_descriptor(facts: &TribleSet) -> Result<CollectionPolicy, RecordDecodeError> {
    descriptor::validate(facts)
}

pub(crate) struct LoadedCollectionDescriptor {
    pub(crate) fragment: Fragment,
    pub(crate) policy: CollectionPolicy,
}

pub(crate) fn load_collection_descriptor<R>(
    snapshot: &R,
    collection: CollectionHandle,
) -> Result<LoadedCollectionDescriptor, CollectionDescriptorError<R::GetError<Infallible>>>
where
    R: BlobStoreGet,
{
    let descriptor_blob: Blob<SimpleArchive> = snapshot
        .get(collection)
        .map_err(|source| CollectionDescriptorError::Get { collection, source })?;
    let facts =
        <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(descriptor_blob)
            .map_err(|source| CollectionDescriptorError::Invalid {
                collection,
                source: RecordDecodeError::from(source),
            })?;
    let policy = validate_generic_descriptor(&facts)
        .map_err(|source| CollectionDescriptorError::Invalid { collection, source })?;
    Ok(LoadedCollectionDescriptor {
        fragment: Fragment::from(facts),
        policy,
    })
}

pub(crate) enum AdmissionEvidence {
    Open,
    Quorum {
        roots: Vec<VerifyingKey>,
        invoke_threshold: NonZeroUsize,
        delegate_threshold: Option<NonZeroUsize>,
        request: CapabilityRequest,
        bundles: Arc<[CapabilityProofBundle]>,
    },
}

impl AdmissionEvidence {
    pub(crate) fn authorizes(&self, subject: VerifyingKey, instant: hifitime::Epoch) -> bool {
        match self {
            Self::Open => true,
            Self::Quorum {
                roots,
                invoke_threshold,
                delegate_threshold,
                request,
                bundles,
            } => capability_quorum_authorizes(
                bundles.iter(),
                roots.iter().copied(),
                instant,
                subject,
                *request,
                *invoke_threshold,
                *delegate_threshold,
            ),
        }
    }

    /// Conservative inclusive bound on this immutable authority observation.
    ///
    /// `None` means no currently usable proof path is time-bounded. A returned
    /// bound can precede the actual loss of authorization when another proof
    /// path is redundant; callers should reobserve rather than infer
    /// revocation.
    pub(crate) fn observation_valid_through(
        &self,
        instant: hifitime::Epoch,
    ) -> Option<hifitime::Epoch> {
        match self {
            Self::Open => None,
            Self::Quorum {
                roots,
                request,
                bundles,
                ..
            } => capability_quorum_observation_valid_through(
                bundles.iter(),
                roots.iter().copied(),
                instant,
                request.atom(),
            ),
        }
    }
}

pub(crate) fn load_resident_proof_bundles<R>(
    reader: &R,
    proofs: impl IntoIterator<Item = CapabilityProof>,
) -> Arc<[CapabilityProofBundle]>
where
    R: BlobStoreGet,
{
    proofs
        .into_iter()
        .filter_map(|proof| {
            let claims = proof
                .claim_handles()
                .map(|claim| reader.get::<Blob<SimpleArchive>, SimpleArchive>(claim).ok())
                .collect::<Option<Vec<_>>>()?;
            Some(CapabilityProofBundle::new(proof, claims))
        })
        .collect::<Vec<_>>()
        .into()
}

pub(crate) fn admission_evidence_from_bundles(
    policy: &AdmissionPolicy,
    action: crate::id::Id,
    required: CapabilityMode,
    collection: CollectionHandle,
    bundles: Arc<[CapabilityProofBundle]>,
) -> AdmissionEvidence {
    let AdmissionPolicy::Quorum(quorum) = policy else {
        return AdmissionEvidence::Open;
    };
    let atom = CapabilityAtom::new(
        CapabilityAction::new(action),
        CapabilityResource::from(collection),
    );
    let request = CapabilityRequest::new(atom, required);
    AdmissionEvidence::Quorum {
        roots: quorum.roots().to_vec(),
        invoke_threshold: NonZeroUsize::new(quorum.invoke_threshold() as usize)
            .expect("validated collection policy has a nonzero invoke threshold"),
        delegate_threshold: quorum.delegate_threshold().map(|threshold| {
            NonZeroUsize::new(threshold as usize)
                .expect("validated collection policy has a nonzero delegate threshold")
        }),
        request,
        bundles,
    }
}

fn admission_evidence_at<R>(
    reader: &R,
    policy: &AdmissionPolicy,
    action: crate::id::Id,
    required: CapabilityMode,
    collection: CollectionHandle,
    proofs: impl IntoIterator<Item = CapabilityProof>,
) -> AdmissionEvidence
where
    R: BlobStoreGet,
{
    admission_evidence_from_bundles(
        policy,
        action,
        required,
        collection,
        load_resident_proof_bundles(reader, proofs),
    )
}

fn discover_admission_evidence_at<S>(
    snapshot: &S,
    policy: &AdmissionPolicy,
    action: crate::id::Id,
    required: CapabilityMode,
    collection: CollectionHandle,
) -> Result<AdmissionEvidence, CollectionEvidenceDiscoveryError<S::ProofsError>>
where
    S: BlobStoreGet + CapabilityProofRead,
{
    let needs_proofs = matches!(policy, AdmissionPolicy::Quorum(_));
    if !needs_proofs {
        return Ok(admission_evidence_at(
            snapshot,
            policy,
            action,
            required,
            collection,
            [],
        ));
    }
    let proofs = snapshot
        .proofs()
        .map_err(CollectionEvidenceDiscoveryError::Proofs)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(CollectionEvidenceDiscoveryError::Proofs)?;
    Ok(admission_evidence_at(
        snapshot, policy, action, required, collection, proofs,
    ))
}

fn discover_admitted_cover_at<S, L>(
    snapshot: &S,
    collection: Collection<L>,
    instant: hifitime::Epoch,
) -> Result<
    (Fragment, DiscoveredCollectionRecords, Cover<L>),
    CollectionCoverError<S::RecordsError, S::ProofsError, S::GetError<Infallible>>,
>
where
    S: BlobStoreGet + CapabilityProofRead + CollectionRead,
    L: CollectionEncoding,
{
    let loaded = load_collection_descriptor(snapshot, collection.handle())
        .map_err(CollectionCoverError::Descriptor)?;
    let evidence = discover_admission_evidence_at(
        snapshot,
        loaded.policy.write(),
        ACTION_WRITE,
        CapabilityMode::Invoke,
        collection.handle(),
    )
    .map_err(CollectionCoverError::Evidence)?;
    let mut authorized = BTreeMap::<[u8; 32], bool>::new();
    let discovered =
        discover_collection_records_authorized(snapshot, collection.handle(), |subject| {
            *authorized.entry(subject.raw).or_insert_with(|| {
                VerifyingKey::from_bytes(&subject.raw)
                    .map(|subject| evidence.authorizes(subject, instant))
                    .unwrap_or(false)
            })
        })
        .map_err(CollectionCoverError::Discovery)?;
    let cover = Cover::from_data(
        collection,
        discovered.commits().iter().map(CollectionCommit::data),
    );
    Ok((loaded.fragment, discovered, cover))
}

/// Decide READ admission for an untyped collection handle from explicitly
/// supplied portable proof bundles.
///
/// Network discovery starts from the descriptor handle carried on the wire,
/// before its member encoding is known. This boundary therefore validates the
/// generic descriptor and its READ policy without manufacturing a typed
/// [`Collection`]. It neither enumerates nor persists ambient proof state.
pub fn collection_reader_is_admitted_by_at<S>(
    snapshot: &S,
    collection: CollectionHandle,
    subject: VerifyingKey,
    bundles: &[CapabilityProofBundle],
    instant: hifitime::Epoch,
) -> Result<bool, CollectionDescriptorError<S::GetError<Infallible>>>
where
    S: BlobStoreGet,
{
    let loaded = load_collection_descriptor(snapshot, collection)?;
    Ok(collection_reader_is_admitted_by_policy_at(
        collection,
        &loaded.policy,
        subject,
        bundles,
        instant,
    ))
}

/// Decide READ admission against one already validated collection policy.
///
/// This is the pure seam for a network host which pinned the descriptor policy
/// while constructing its immutable per-collection activation overlay. It
/// performs no store access, proof discovery, persistence, or clock sampling.
pub fn collection_reader_is_admitted_by_policy_at(
    collection: CollectionHandle,
    policy: &CollectionPolicy,
    subject: VerifyingKey,
    bundles: &[CapabilityProofBundle],
    instant: hifitime::Epoch,
) -> bool {
    let evidence = admission_evidence_from_bundles(
        policy.read(),
        ACTION_READ,
        CapabilityMode::Invoke,
        collection,
        bundles.to_vec().into(),
    );
    evidence.authorizes(subject, instant)
}

impl<L: CollectionEncoding> Collection<L> {
    /// Decide whether `subject` is admitted as a writer at `instant`.
    ///
    /// Open policy admits every subject. A quorum admits the subject only when
    /// the exact-action proof forest carries support from enough distinct
    /// configured roots.
    pub fn writer_is_admitted_at<S>(
        self,
        snapshot: &S,
        subject: VerifyingKey,
        instant: hifitime::Epoch,
    ) -> Result<bool, CollectionAdmissionError<S::ProofsError, S::GetError<Infallible>>>
    where
        S: BlobStoreGet + CapabilityProofRead,
    {
        let loaded = load_collection_descriptor(snapshot, self.handle())
            .map_err(CollectionAdmissionError::Descriptor)?;
        let evidence = discover_admission_evidence_at(
            snapshot,
            loaded.policy.write(),
            ACTION_WRITE,
            CapabilityMode::Invoke,
            self.handle(),
        )
        .map_err(CollectionAdmissionError::Evidence)?;
        Ok(evidence.authorizes(subject, instant))
    }

    /// Decide whether `subject` is admitted at the current clock instant.
    pub fn writer_is_admitted<S>(
        self,
        snapshot: &S,
        subject: VerifyingKey,
    ) -> Result<bool, CollectionAdmissionError<S::ProofsError, S::GetError<Infallible>>>
    where
        S: BlobStoreGet + CapabilityProofRead,
    {
        self.writer_is_admitted_at(snapshot, subject, clock::epoch_now())
    }

    /// Decide whether `subject` is admitted as a reader at `instant`.
    ///
    /// This is the disclosure boundary corresponding to the descriptor's READ
    /// ceiling. Local materialization remains caller-controlled because a
    /// local store cannot infer which external principal will receive bytes.
    pub fn reader_is_admitted_at<S>(
        self,
        snapshot: &S,
        subject: VerifyingKey,
        instant: hifitime::Epoch,
    ) -> Result<bool, CollectionAdmissionError<S::ProofsError, S::GetError<Infallible>>>
    where
        S: BlobStoreGet + CapabilityProofRead,
    {
        let loaded = load_collection_descriptor(snapshot, self.handle())
            .map_err(CollectionAdmissionError::Descriptor)?;
        let evidence = discover_admission_evidence_at(
            snapshot,
            loaded.policy.read(),
            ACTION_READ,
            CapabilityMode::Invoke,
            self.handle(),
        )
        .map_err(CollectionAdmissionError::Evidence)?;
        Ok(evidence.authorizes(subject, instant))
    }

    /// Decide READ admission from one explicitly supplied portable proof set.
    ///
    /// This is the network-facing counterpart of [`Self::reader_is_admitted_at`].
    /// It loads only the immutable collection descriptor from `snapshot`; it
    /// neither enumerates nor persists ambient proof-store state. An open READ
    /// policy therefore succeeds with an empty bundle slice, while a quorum is
    /// evaluated over every supplied bundle as one fixed-point proof forest.
    pub fn reader_is_admitted_by_at<S>(
        self,
        snapshot: &S,
        subject: VerifyingKey,
        bundles: &[CapabilityProofBundle],
        instant: hifitime::Epoch,
    ) -> Result<bool, CollectionDescriptorError<S::GetError<Infallible>>>
    where
        S: BlobStoreGet,
    {
        collection_reader_is_admitted_by_at(snapshot, self.handle(), subject, bundles, instant)
    }

    /// Decide whether `subject` is admitted as a reader now.
    pub fn reader_is_admitted<S>(
        self,
        snapshot: &S,
        subject: VerifyingKey,
    ) -> Result<bool, CollectionAdmissionError<S::ProofsError, S::GetError<Infallible>>>
    where
        S: BlobStoreGet + CapabilityProofRead,
    {
        self.reader_is_admitted_at(snapshot, subject, clock::epoch_now())
    }

    /// Discover the exact payload cover admitted at `instant`.
    ///
    /// The result is the semantic COMMIT frontier. It deliberately does not
    /// substitute resident MERGE results; call [`Cover::resolve`] when a
    /// physical cover is needed.
    pub fn admitted_at<S>(
        self,
        snapshot: &S,
        instant: hifitime::Epoch,
    ) -> Result<
        Cover<L>,
        CollectionCoverError<S::RecordsError, S::ProofsError, S::GetError<Infallible>>,
    >
    where
        S: BlobStoreGet + CapabilityProofRead + CollectionRead,
    {
        discover_admitted_cover_at(snapshot, self, instant).map(|(_, _, cover)| cover)
    }

    /// Discover the exact payload cover admitted at the current clock instant.
    pub fn admitted<S>(
        self,
        snapshot: &S,
    ) -> Result<
        Cover<L>,
        CollectionCoverError<S::RecordsError, S::ProofsError, S::GetError<Infallible>>,
    >
    where
        S: BlobStoreGet + CapabilityProofRead + CollectionRead,
    {
        self.admitted_at(snapshot, clock::epoch_now())
    }

    /// Discover an admitted cover and the exact COMMIT roots selected by the
    /// same authorization decision.
    pub fn admitted_with_commits_at<S>(
        self,
        snapshot: &S,
        instant: hifitime::Epoch,
    ) -> Result<
        (Cover<L>, Vec<CollectionCommit>),
        CollectionCoverError<S::RecordsError, S::ProofsError, S::GetError<Infallible>>,
    >
    where
        S: BlobStoreGet + CapabilityProofRead + CollectionRead,
    {
        let (_, discovered, cover) = discover_admitted_cover_at(snapshot, self, instant)?;
        Ok((cover, discovered.commits().to_vec()))
    }

    /// Discover an admitted cover and its exact roots at the current instant.
    pub fn admitted_with_commits<S>(
        self,
        snapshot: &S,
    ) -> Result<
        (Cover<L>, Vec<CollectionCommit>),
        CollectionCoverError<S::RecordsError, S::ProofsError, S::GetError<Infallible>>,
    >
    where
        S: BlobStoreGet + CapabilityProofRead + CollectionRead,
    {
        self.admitted_with_commits_at(snapshot, clock::epoch_now())
    }

    /// Read one logical value admitted at `instant` through one immutable
    /// store observation.
    pub fn read_at<V, S>(
        self,
        snapshot: &S,
        instant: hifitime::Epoch,
    ) -> Result<
        V,
        CollectionReadError<S::RecordsError, S::ProofsError, S::GetError<Infallible>, V::Error>,
    >
    where
        S: BlobStoreGet + BlobStoreMeta + CapabilityProofRead + CollectionRead,
        V: TryFromCover<L>,
    {
        let (descriptor, discovered, admitted) =
            discover_admitted_cover_at(snapshot, self, instant)
                .map_err(CollectionMaterializationError::from)?;
        let resolved = resolve_cover_from_observation::<S, L, V::Error, _>(
            snapshot,
            &descriptor,
            discovered,
            admitted,
        )?;
        V::try_from_cover(&resolved, snapshot).map_err(CollectionMaterializationError::from)
    }

    /// Read one logical value, sampling the clock exactly once.
    pub fn read<V, S>(
        self,
        snapshot: &S,
    ) -> Result<
        V,
        CollectionReadError<S::RecordsError, S::ProofsError, S::GetError<Infallible>, V::Error>,
    >
    where
        S: BlobStoreGet + BlobStoreMeta + CapabilityProofRead + CollectionRead,
        V: TryFromCover<L>,
    {
        self.read_at(snapshot, clock::epoch_now())
    }
}

impl<L: CollectionEncoding> Cover<L> {
    /// Resolve this exact semantic cover to one resident physical cover.
    ///
    /// The returned `Cover` names the members actually selected from the same
    /// snapshot. Support-equivalent compaction may therefore make it differ
    /// from `self`; additions-only deltas belong on admitted covers, not on
    /// these replaceable physical decompositions.
    pub fn resolve<S>(
        &self,
        snapshot: &S,
    ) -> Result<
        Cover<L>,
        CollectionMaterializationError<S::RecordsError, S::GetError<Infallible>, Infallible>,
    >
    where
        S: BlobStoreGet + BlobStoreMeta + CollectionRead,
    {
        let descriptor = load_collection_descriptor(snapshot, self.collection().handle())
            .map_err(CollectionMaterializationError::from)?
            .fragment;
        let discovered = if self.is_empty() {
            DiscoveredCollectionRecords::default()
        } else {
            discover_collection_equations_for_cover(snapshot, self)
                .map_err(CollectionMaterializationError::Discovery)?
        };
        resolve_cover_from_observation::<S, L, Infallible, Infallible>(
            snapshot,
            &descriptor,
            discovered,
            self.clone(),
        )
    }

    /// Return every strictly verified provenance COMMIT currently present for
    /// these payload members in `snapshot`.
    pub fn commits<S>(
        &self,
        snapshot: &S,
    ) -> Result<Vec<CollectionCommit>, CollectionDiscoveryError<S::RecordsError>>
    where
        S: CollectionRead,
    {
        let discovered = discover_collection_claims_for_cover(snapshot, self)?;
        Ok(discovered.commits().to_vec())
    }
}

/// Ergonomic collection operations implemented directly by the backing store.
///
/// The trait carries no state and is blanket-implemented. Its purpose is only
/// method syntax: collections remain plain descriptor handles and stores
/// retain their native ownership, flushing, and closing APIs.
pub trait CollectionStoreExt: BlobStorePut + CollectionStore + Sized {
    /// Register a complete custom descriptor at the raw typed boundary.
    ///
    /// Normal root and derived construction should use [`collection`](Self::collection)
    /// and [`derive`](Self::derive), which make canonical descriptors by
    /// construction.
    fn register_collection<L>(
        &mut self,
        descriptor: Fragment,
    ) -> Result<Collection<L>, CollectionRegistrationError<<Self as BlobStorePut>::PutError>>
    where
        L: CollectionEncoding,
    {
        super::encoding::validate_descriptor_type::<L>(&descriptor).map_err(
            |source| match source {
                CollectionTypeError::Malformed(source) => {
                    CollectionRegistrationError::InvalidDescriptor(source)
                }
                source => CollectionRegistrationError::WrongType(source),
            },
        )?;
        let handle = descriptor::put_closure(self, &descriptor)
            .map_err(CollectionRegistrationError::DependencyPut)?;
        Ok(Collection::from_handle(handle))
    }

    /// Create and register one named root fact collection.
    fn collection(
        &mut self,
        name: &str,
        policy: CollectionPolicy,
    ) -> Result<
        Collection<SimpleArchive>,
        CollectionRegistrationError<<Self as BlobStorePut>::PutError>,
    > {
        self.register_collection::<SimpleArchive>(descriptor::naming::<SimpleArchive>(name, policy))
    }

    /// Create and register one canonical derived collection.
    ///
    /// The mapping value owns its concrete parameters and description; its
    /// associated target encoding owns the target representation.
    fn derive<M>(
        &mut self,
        source: Collection<M::Source>,
        mapping: M,
        policy: CollectionPolicy,
    ) -> Result<Collection<M::Target>, CollectionRegistrationError<<Self as BlobStorePut>::PutError>>
    where
        M: CollectionMapping,
    {
        self.register_collection::<M::Target>(descriptor::deriving(
            source.handle(),
            &mapping,
            policy,
        ))
    }

    /// Publish one signed fragment into an already registered collection.
    ///
    /// This performs no capability or descriptor check. Local storage is a
    /// grow-only claim ledger; authority and descriptor validity are applied
    /// at untrusted read and synchronization boundaries. Fragment attachments,
    /// data, and metadata are stored before the signed record is inserted,
    /// without an implicit durability flush.
    fn commit(
        &mut self,
        collection: Collection<SimpleArchive>,
        signing_key: &SigningKey,
        fragment: Fragment,
    ) -> Result<
        CollectionCommit,
        CollectionCommitError<
            <Self as BlobStorePut>::PutError,
            <Self as CollectionStore>::InsertError,
        >,
    > {
        // A signed commit introduces authored graph facts. Other collection
        // encodings are reproducible representations introduced through
        // DERIVE and MERGE records, not alternative signed leaf formats.
        let (_, facts, metadata_facts, mut fragment_blobs) = fragment.into_parts();
        let data_root: Blob<SimpleArchive> = facts.to_blob();
        let metadata: Blob<SimpleArchive> = metadata_facts.to_blob();
        let mut attachments: Vec<Blob<UnknownBlob>> = fragment_blobs
            .snapshot()
            .expect("MemoryBlobStore::snapshot is infallible")
            .into_iter()
            .map(|(_, blob)| blob)
            .collect();
        attachments.sort_unstable_by_key(|blob| blob.get_handle().raw);

        for blob in attachments {
            self.put::<UnknownBlob, _>(blob)
                .map_err(CollectionCommitError::DependencyPut)?;
        }
        let data_handle = self
            .put::<SimpleArchive, _>(data_root)
            .map_err(CollectionCommitError::DependencyPut)?;
        let metadata = self
            .put::<SimpleArchive, _>(metadata)
            .map_err(CollectionCommitError::DependencyPut)?;
        let commit = CollectionCommit::sign(
            signing_key,
            collection.handle(),
            Handle::<SimpleArchive>::to_hash(data_handle),
            metadata,
        );
        self.insert(super::CollectionRecord::Commit(commit))
            .map_err(CollectionCommitError::RecordInsert)?;
        Ok(commit)
    }
}

impl<S> CollectionStoreExt for S where S: BlobStorePut + CollectionStore {}

/// Resolve one already-discovered exact payload cover.
///
/// Admission and opaque replay use this single validator so descriptor,
/// mandatory member, and merge-cover semantics cannot drift apart. The result
/// is the actual resident physical cover selected by the resolver.
pub(crate) fn resolve_cover_from_observation<S, L, ViewError, EvidenceError>(
    snapshot: &S,
    descriptor: &Fragment,
    discovered: DiscoveredCollectionRecords,
    cover: Cover<L>,
) -> Result<
    Cover<L>,
    CollectionMaterializationError<
        S::RecordsError,
        S::GetError<Infallible>,
        ViewError,
        EvidenceError,
    >,
>
where
    S: BlobStoreGet + BlobStoreMeta + CollectionRead,
    L: CollectionEncoding,
{
    let collection = cover.collection().handle();
    let reader = snapshot;

    if cover.is_empty() {
        return Ok(cover);
    }

    // The descriptor was fetched and bound to this typed collection through
    // this same immutable snapshot. Its content address already binds these
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
        L::validate_member(descriptor, &blob, reader)
            .map_err(|source| CollectionMaterializationError::InvalidMember { member, source })?;
        known.insert(member, blob);
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
                (Some(low_blob), Some(high_blob)) => {
                    match L::join_members(descriptor, low_blob, high_blob, reader) {
                        Ok(Some(value)) => {
                            let expected = Handle::<L>::to_hash(value.get_handle());
                            expected_hashes.insert(pair, expected);
                            joined = Some(value);
                            Some(expected)
                        }
                        // `Ok(None)` means this representation deliberately
                        // performs compaction in another collection lattice.
                        // Invalid optional cache evidence is equally inert.
                        Ok(None) | Err(_) => None,
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
                            L::join_members(descriptor, low_blob, high_blob, reader)
                                .ok()
                                .flatten()
                        }
                        _ => None,
                    };
                }
                if let Some(value) = joined {
                    debug_assert_eq!(Handle::<L>::to_hash(value.get_handle()), expected_data,);
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

    let mut selected = BTreeMap::<CollectionData, Blob<L>>::new();
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
                Ok(root) => match L::validate_member(descriptor, &root, reader) {
                    Ok(()) => {
                        selected.insert(data, root);
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

    Ok(Cover::from_data(cover.collection(), physical_cover))
}
