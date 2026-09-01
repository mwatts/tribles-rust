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

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::Blob;
use crate::capability::{
    capability_quorum_authorizes, CapabilityAction, CapabilityAtom, CapabilityMode,
    CapabilityProof, CapabilityProofBundle, CapabilityRequest, CapabilityResource,
};
use crate::clock;
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding};
use crate::patch::{Blake3Merkle, IdentitySchema, PATCH};
use crate::repo::{
    BlobStore, BlobStoreGet, BlobStoreList, BlobStoreMeta, BlobStorePut, CapabilityProofRead,
};
use crate::repo::{CapabilityProofStore, SnapshotSource};
use crate::trible::{Fragment, TribleSet};

use super::discovery::{
    discover_collection_claims_for_cover, discover_collection_equations_for_cover,
};
use super::encoding::{
    collection_member_availability, collection_member_structural_availability,
    CollectionMemberAvailability,
};
use super::exact_derived::{ExactDerivedCollection, ExactDerivedCollectionError};
use super::simplearchive_union::{FactViewError, PreparedCollectionCommit};
use super::{
    collection_complete_physical_cover, descriptor, discover_collection_records_authorized,
    resolve_collection_semantics_from_roots, Collection, CollectionClaimValidation,
    CollectionCommit, CollectionData, CollectionDiscoveryError, CollectionEncoding,
    CollectionFunctionalConflict, CollectionHandle, CollectionOperationError, CollectionRead,
    CollectionResolutionError, CollectionSemantics, CollectionStore, CollectionTypeError,
    CollectionValidationRequest, DiscoveredCollectionRecords, RecordDecodeError, TryFromCover,
    TryFromCoverError, ACTION_READ, ACTION_WRITE,
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

/// Failure to open an existing collection as one concrete member encoding.
///
/// Opening is a read-only validation boundary. It neither registers the
/// descriptor nor writes any of its attachment closure into the store.
#[derive(Debug)]
pub enum CollectionOpenError<GetError> {
    /// The descriptor could not be fetched or decoded generically.
    Descriptor(CollectionDescriptorError<GetError>),
    /// The descriptor does not denote the requested member encoding.
    WrongType(CollectionTypeError),
}

impl<GetError> fmt::Display for CollectionOpenError<GetError>
where
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(source) => source.fmt(formatter),
            Self::WrongType(source) => write!(formatter, "wrong collection type: {source}"),
        }
    }
}

impl<GetError> Error for CollectionOpenError<GetError>
where
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(source) => Some(source),
            Self::WrongType(source) => Some(source),
        }
    }
}

/// Failure to persist one root-issued, unbounded collection grant.
///
/// The operation validates the descriptor and the exact action roots before
/// writing anything. Claim blobs are then stored before the native proof
/// record, so a failed or interrupted publication can only leave inert content
/// behind.
#[derive(Debug)]
pub enum CollectionGrantError<SnapshotError, GetError, PutError, InsertError> {
    /// The store could not freeze the read boundary used to validate policy.
    Snapshot(SnapshotError),
    /// The collection descriptor was unavailable or malformed.
    Descriptor(CollectionDescriptorError<GetError>),
    /// The collection action is already open and therefore needs no proof.
    OpenPolicy {
        /// Exact action whose policy is open.
        action: Id,
        /// Exact collection whose action policy is open.
        collection: CollectionHandle,
    },
    /// The supplied signing key is not one of this collection action's roots.
    RootNotAuthorized {
        /// Exact action whose policy rejected the signer.
        action: Id,
        /// Exact collection whose policy rejected the signer.
        collection: CollectionHandle,
        /// Public half of the supplied root signing key.
        root: VerifyingKey,
    },
    /// A canonical claim blob could not be stored.
    ClaimPut(PutError),
    /// The native proof record could not be inserted after its claim closure.
    ProofInsert(InsertError),
}

impl<SnapshotError, GetError, PutError, InsertError> fmt::Display
    for CollectionGrantError<SnapshotError, GetError, PutError, InsertError>
where
    SnapshotError: fmt::Display,
    GetError: fmt::Display,
    PutError: fmt::Display,
    InsertError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Snapshot(source) => {
                write!(formatter, "failed to freeze store snapshot: {source}")
            }
            Self::Descriptor(source) => source.fmt(formatter),
            Self::OpenPolicy { action, collection } => write!(
                formatter,
                "collection {} has an open {} policy; no proof is required",
                hex::encode_upper(collection.raw),
                collection_action_label(*action),
            ),
            Self::RootNotAuthorized {
                action,
                collection,
                root,
            } => write!(
                formatter,
                "key {} is not a {} root of collection {}",
                hex::encode_upper(root.to_bytes()),
                collection_action_label(*action),
                hex::encode_upper(collection.raw),
            ),
            Self::ClaimPut(source) => {
                write!(
                    formatter,
                    "failed to store capability claim before proof: {source}"
                )
            }
            Self::ProofInsert(source) => write!(
                formatter,
                "failed to insert capability proof after its claims: {source}",
            ),
        }
    }
}

impl<SnapshotError, GetError, PutError, InsertError> Error
    for CollectionGrantError<SnapshotError, GetError, PutError, InsertError>
where
    SnapshotError: Error + 'static,
    GetError: Error + 'static,
    PutError: Error + 'static,
    InsertError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Snapshot(source) => Some(source),
            Self::Descriptor(source) => Some(source),
            Self::OpenPolicy { .. } | Self::RootNotAuthorized { .. } => None,
            Self::ClaimPut(source) => Some(source),
            Self::ProofInsert(source) => Some(source),
        }
    }
}

/// Error returned by [`grant_collection_read`].
pub type CollectionReadGrantError<SnapshotError, GetError, PutError, InsertError> =
    CollectionGrantError<SnapshotError, GetError, PutError, InsertError>;

/// Error returned by [`grant_collection_write`].
pub type CollectionWriteGrantError<SnapshotError, GetError, PutError, InsertError> =
    CollectionGrantError<SnapshotError, GetError, PutError, InsertError>;

fn collection_action_label(action: Id) -> &'static str {
    match action {
        ACTION_READ => "READ",
        ACTION_WRITE => "WRITE",
        _ => "unknown collection action",
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
/// admission or stored collection operations rather than a caller-forged set
/// of hashes.
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

    fn ensure_same_collection(&self, other: &Self) -> Result<(), CoverAlgebraError> {
        if self.collection == other.collection {
            Ok(())
        } else {
            Err(CoverAlgebraError::DifferentCollection {
                left: self.collection.handle(),
                right: other.collection.handle(),
            })
        }
    }

    /// Return the set union of two points in the same collection lattice.
    pub fn union(&self, other: &Self) -> Result<Self, CoverAlgebraError> {
        self.ensure_same_collection(other)?;
        let mut members = self.members.clone();
        members.union(other.members.clone());
        Ok(Self::from_patch(self.collection, members))
    }

    /// Return the set intersection of two points in the same collection lattice.
    pub fn intersection(&self, other: &Self) -> Result<Self, CoverAlgebraError> {
        self.ensure_same_collection(other)?;
        Ok(Self::from_patch(
            self.collection,
            self.members.intersect(&other.members),
        ))
    }

    /// Return the members of `self` absent from `other`.
    pub fn difference(&self, other: &Self) -> Result<Self, CoverAlgebraError> {
        self.ensure_same_collection(other)?;
        Ok(Self::from_patch(
            self.collection,
            self.members.difference(&other.members),
        ))
    }

    /// Whether every member of `self` is present in `other`.
    pub fn is_subset(&self, other: &Self) -> Result<bool, CoverAlgebraError> {
        self.ensure_same_collection(other)?;
        Ok(self.members.difference(&other.members).is_empty())
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

/// Failure to combine covers from distinct collection lattices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoverAlgebraError {
    /// The operands carry different canonical collection descriptors.
    DifferentCollection {
        /// Descriptor carried by the left operand.
        left: CollectionHandle,
        /// Descriptor carried by the right operand.
        right: CollectionHandle,
    },
}

impl fmt::Display for CoverAlgebraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentCollection { left, right } => write!(
                formatter,
                "cover collection {} differs from {}",
                hex::encode_upper(left.raw),
                hex::encode_upper(right.raw),
            ),
        }
    }
}

impl Error for CoverAlgebraError {}

/// Failure to determine the greatest resident subset of a semantic cover.
///
/// An absent or representation-incomplete blob is ordinary unavailability,
/// not an error. These variants mean the immutable snapshot itself could not
/// be observed coherently or its stored equations were contradictory.
#[derive(Debug)]
pub enum CoverAvailabilityError<RecordsError, ResidencyError> {
    /// Native collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
    /// Blob residency could not be observed for one semantic member.
    Residency {
        /// Exact member whose residency was being inspected.
        member: CollectionData,
        /// Backend residency-observation failure.
        source: ResidencyError,
    },
    /// Stored equations contradicted operation functionality.
    ResolutionConflict(Box<CollectionFunctionalConflict>),
}

impl<RecordsError, ResidencyError> fmt::Display
    for CoverAvailabilityError<RecordsError, ResidencyError>
where
    RecordsError: fmt::Display,
    ResidencyError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(source) => source.fmt(formatter),
            Self::Residency { member, source } => write!(
                formatter,
                "failed to inspect residency of collection member {}: {source}",
                hex::encode_upper(member.raw),
            ),
            Self::ResolutionConflict(source) => source.fmt(formatter),
        }
    }
}

impl<RecordsError, ResidencyError> Error for CoverAvailabilityError<RecordsError, ResidencyError>
where
    RecordsError: Error + 'static,
    ResidencyError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Discovery(source) => Some(source),
            Self::Residency { source, .. } => Some(source),
            Self::ResolutionConflict(source) => Some(source),
        }
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
/// Signatures and metadata remain queryable provenance rather than becoming
/// coordinates of the payload lattice. Stored equations are materialized LSM
/// work: resolution follows them without replaying their algebra. Missing
/// bytes remain an ordinary physical-cover miss, and the eventual typed view
/// owns decoding of the selected members.
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
    /// One resident optional materialization could not expose its immutable
    /// representation closure and no valid alternate cover was available.
    InvalidMember {
        /// Exact payload identity of the unusable materialization.
        member: CollectionData,
        /// Encoding-specific availability failure.
        source: CollectionOperationError,
    },
    /// Stored equations contradicted operation functionality.
    ResolutionConflict(Box<CollectionFunctionalConflict>),
    /// No resident physical cover spans every semantic obligation.
    Missing {
        /// Requested semantic members lacking a complete resident realization.
        obligations: BTreeSet<CollectionData>,
        /// Named immutable representation dependencies which would make an
        /// otherwise useful resident member complete.
        dependencies: BTreeSet<CollectionData>,
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
                "resident collection member {} is unusable: {source}",
                hex::encode_upper(member.raw),
            ),
            Self::ResolutionConflict(source) => source.fmt(f),
            Self::Missing {
                obligations,
                dependencies,
            } => {
                write!(
                    f,
                    "{} requested semantic member(s) have no complete resident realization",
                    obligations.len(),
                )?;
                if !dependencies.is_empty() {
                    write!(
                        f,
                        " ({} representation dependency blob(s) missing)",
                        dependencies.len(),
                    )?;
                }
                Ok(())
            }
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

/// Persist one deterministic root-issued READ/Invoke proof for `recipient`.
///
/// This is deliberately representation-neutral: READ authority concerns the
/// exact descriptor handle, not the collection member encoding. The
/// descriptor is first loaded and structurally validated through one coherent
/// store snapshot. `root` must be named by its READ quorum; an open READ policy
/// needs no grant and is rejected as a redundant operation.
///
/// The claim closure is written before the proof record. Repeating the same
/// call with the same collection, root, and recipient therefore reproduces the
/// same content identities and is idempotent on a conforming store.
pub fn grant_collection_read<S>(
    store: &mut S,
    collection: CollectionHandle,
    root: &SigningKey,
    recipient: VerifyingKey,
) -> Result<
    CapabilityProofBundle,
    CollectionReadGrantError<
        <S as SnapshotSource>::SnapshotError,
        <<S as SnapshotSource>::Snapshot as BlobStoreGet>::GetError<Infallible>,
        <S as BlobStorePut>::PutError,
        <S as CapabilityProofStore>::InsertError,
    >,
>
where
    S: SnapshotSource + BlobStorePut + CapabilityProofStore,
    <S as SnapshotSource>::Snapshot: BlobStoreGet,
{
    grant_collection_action(store, collection, root, recipient, ACTION_READ)
}

/// Persist one deterministic root-issued WRITE/Invoke proof for `recipient`.
///
/// This is the WRITE counterpart of [`grant_collection_read`]. The descriptor
/// is validated from one coherent snapshot, `root` must be named by its WRITE
/// quorum, and the canonical claim closure is stored before the proof record.
/// The proof activates matching COMMITs by `recipient` when a later collection
/// snapshot performs WRITE admission; local publication itself remains
/// unconditional.
pub fn grant_collection_write<S>(
    store: &mut S,
    collection: CollectionHandle,
    root: &SigningKey,
    recipient: VerifyingKey,
) -> Result<
    CapabilityProofBundle,
    CollectionWriteGrantError<
        <S as SnapshotSource>::SnapshotError,
        <<S as SnapshotSource>::Snapshot as BlobStoreGet>::GetError<Infallible>,
        <S as BlobStorePut>::PutError,
        <S as CapabilityProofStore>::InsertError,
    >,
>
where
    S: SnapshotSource + BlobStorePut + CapabilityProofStore,
    <S as SnapshotSource>::Snapshot: BlobStoreGet,
{
    grant_collection_action(store, collection, root, recipient, ACTION_WRITE)
}

fn grant_collection_action<S>(
    store: &mut S,
    collection: CollectionHandle,
    root: &SigningKey,
    recipient: VerifyingKey,
    action: Id,
) -> Result<
    CapabilityProofBundle,
    CollectionGrantError<
        <S as SnapshotSource>::SnapshotError,
        <<S as SnapshotSource>::Snapshot as BlobStoreGet>::GetError<Infallible>,
        <S as BlobStorePut>::PutError,
        <S as CapabilityProofStore>::InsertError,
    >,
>
where
    S: SnapshotSource + BlobStorePut + CapabilityProofStore,
    <S as SnapshotSource>::Snapshot: BlobStoreGet,
{
    let snapshot = store.snapshot().map_err(CollectionGrantError::Snapshot)?;
    let descriptor = load_collection_descriptor(&snapshot, collection)
        .map_err(CollectionGrantError::Descriptor)?;
    let policy = match action {
        ACTION_READ => descriptor.policy.read(),
        ACTION_WRITE => descriptor.policy.write(),
        _ => unreachable!("grant wrappers only supply collection actions"),
    };
    let roots = policy
        .roots()
        .ok_or(CollectionGrantError::OpenPolicy { action, collection })?;
    let root_key = root.verifying_key();
    if !roots.contains(&root_key) {
        return Err(CollectionGrantError::RootNotAuthorized {
            action,
            collection,
            root: root_key,
        });
    }
    drop(snapshot);

    let atom = CapabilityAtom::new(
        CapabilityAction::new(action),
        CapabilityResource::from(collection),
    );
    let claim = crate::capability::CapabilityClaim::root(atom, CapabilityMode::Invoke, None);
    let bundle = CapabilityProofBundle::issue_root(root, claim, recipient)
        .expect("the root READ claim constructed here has no parent");

    for claim in bundle.claims().iter().cloned() {
        store
            .put::<SimpleArchive, _>(claim)
            .map_err(CollectionGrantError::ClaimPut)?;
    }
    store
        .insert_proof(bundle.proof().clone())
        .map_err(CollectionGrantError::ProofInsert)?;
    Ok(bundle)
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

/// Decide WRITE admission against one already validated collection policy and
/// an explicitly supplied portable proof forest.
///
/// This pure seam lets a network repair client discard inert records before
/// they cross the local admission boundary. It performs no store access,
/// persistence, or clock sampling.
pub fn collection_writer_is_admitted_by_policy_at(
    collection: CollectionHandle,
    policy: &CollectionPolicy,
    subject: VerifyingKey,
    bundles: &[CapabilityProofBundle],
    instant: hifitime::Epoch,
) -> bool {
    let evidence = admission_evidence_from_bundles(
        policy.write(),
        ACTION_WRITE,
        CapabilityMode::Invoke,
        collection,
        bundles.to_vec().into(),
    );
    evidence.authorizes(subject, instant)
}

impl<L: CollectionEncoding> Collection<L> {
    /// Open an existing descriptor as a typed collection without mutating the
    /// backing store.
    ///
    /// `snapshot` should be the caller's coherent immutable read boundary. The
    /// descriptor is fetched by `handle`, validated generically, and then
    /// checked against `L` before the cheap typed handle is returned.
    pub fn open<S>(
        snapshot: &S,
        handle: CollectionHandle,
    ) -> Result<Self, CollectionOpenError<S::GetError<Infallible>>>
    where
        S: BlobStoreGet,
    {
        let descriptor = load_collection_descriptor(snapshot, handle)
            .map_err(CollectionOpenError::Descriptor)?;
        super::encoding::validate_descriptor_type::<L>(&descriptor.fragment)
            .map_err(CollectionOpenError::WrongType)?;
        Ok(Self::from_handle(handle))
    }

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
    /// expose a replaceable physical decomposition. Use [`Cover::available`]
    /// to inspect resident semantic support or [`Cover::materialize`] to
    /// construct a logical value through this same snapshot.
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
        let (_descriptor, discovered, admitted) =
            discover_admitted_cover_at(snapshot, self, instant)
                .map_err(CollectionMaterializationError::from)?;
        materialize_cover_from_observation::<S, L, V, _>(snapshot, discovered, admitted)
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
    /// Return the greatest subset with a structurally resident realization.
    ///
    /// The result remains in this cover's semantic coordinates. A compacted
    /// resident member may therefore make several requested members available
    /// without appearing in the returned value itself. Consequently
    /// `cover.available(snapshot)? == cover` means a complete representation
    /// closure is structurally resident in that same immutable snapshot, while
    /// `cover.difference(&available)?` names its missing semantic support. This
    /// is not a payload-validation claim: [`Self::materialize`] remains the
    /// boundary which validates content addresses and decodes the selected
    /// representation.
    ///
    /// This method does not acquire missing content or record demand. An
    /// encoding may read a structurally resident root to discover its named
    /// representation dependencies. Network acquisition belongs to a
    /// store-level workflow which produces a later snapshot.
    pub fn available<S>(
        &self,
        snapshot: &S,
    ) -> Result<Cover<L>, CoverAvailabilityError<S::RecordsError, S::Err>>
    where
        S: BlobStoreGet + BlobStoreList + BlobStoreMeta + CollectionRead,
    {
        let discovered = if self.is_empty() {
            DiscoveredCollectionRecords::default()
        } else {
            discover_collection_equations_for_cover(snapshot, self)
                .map_err(CoverAvailabilityError::Discovery)?
        };
        let semantics = resolve_cover_semantics(&discovered, self)
            .map_err(CoverAvailabilityError::ResolutionConflict)?;
        available_cover_from_semantics(snapshot, &semantics, self)
    }

    /// Materialize this semantic cover through one immutable snapshot.
    ///
    /// Physical LSM decomposition is deliberately private and is recomputed
    /// from the supplied snapshot for every call. Passing a later or otherwise
    /// different snapshot which lacks the necessary realization therefore
    /// reports [`CollectionMaterializationError::Missing`] instead of pairing
    /// semantic coordinates with stale physical assumptions.
    pub fn materialize<V, S>(
        &self,
        snapshot: &S,
    ) -> Result<V, CollectionMaterializationError<S::RecordsError, S::GetError<Infallible>, V::Error>>
    where
        S: BlobStoreGet + BlobStoreMeta + CollectionRead,
        V: TryFromCover<L>,
    {
        let discovered = if self.is_empty() {
            DiscoveredCollectionRecords::default()
        } else {
            discover_collection_equations_for_cover(snapshot, self)
                .map_err(CollectionMaterializationError::Discovery)?
        };
        materialize_cover_from_observation::<S, L, V, Infallible>(
            snapshot,
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

    /// Ensure one derived lattice point through its canonical mapping.
    ///
    /// The source collection is carried by `source`; the target descriptor
    /// carries the mapping parameters and policy. Storage owns the singular
    /// deterministic merge/derive schedule, while the encoding and mapping
    /// traits own only their algebra.
    fn ensure<M>(
        &mut self,
        target: Collection<M::Target>,
        source: &Cover<M::Source>,
    ) -> Result<Cover<M::Target>, ExactDerivedCollectionError>
    where
        M: CollectionMapping,
        Self: BlobStore,
        Self::Snapshot: BlobStoreMeta + CollectionRead,
        Handle<M::Source>: InlineEncoding,
        Handle<M::Target>: InlineEncoding,
    {
        ExactDerivedCollection::<M>::new(source.collection(), target)?.ensure(self, source)
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
        PreparedCollectionCommit::from_fragment(fragment)
            .stage_for(self, collection, signing_key)?
            .finalize()
    }
}

impl<S> CollectionStoreExt for S where S: BlobStorePut + CollectionStore {}

/// Resolve the semantic closure of one already-discovered exact payload cover.
fn resolve_cover_semantics<L>(
    discovered: &DiscoveredCollectionRecords,
    cover: &Cover<L>,
) -> Result<CollectionSemantics, Box<CollectionFunctionalConflict>>
where
    L: CollectionEncoding,
{
    let collection = cover.collection().handle();
    let explicit_roots: BTreeSet<_> = cover
        .data_members()
        .map(|data| (collection, data))
        .collect();

    // MERGE records are materialized LSM equations. They are operational
    // evidence, not algebra which needs to be replayed during a read.
    let resolution = resolve_collection_semantics_from_roots(
        discovered,
        &BTreeMap::new(),
        &explicit_roots,
        |request| {
            Ok::<CollectionClaimValidation<()>, Infallible>(match request {
                CollectionValidationRequest::Merge { claim }
                    if claim.collection() == collection =>
                {
                    CollectionClaimValidation::Accepted
                }
                CollectionValidationRequest::Commit { .. }
                | CollectionValidationRequest::Merge { .. }
                | CollectionValidationRequest::Derive { .. } => CollectionClaimValidation::Pending,
            })
        },
    );

    match resolution {
        Ok(resolution) => Ok(resolution.into_semantics()),
        Err(CollectionResolutionError::Validation { source, .. }) => match source {},
        Err(CollectionResolutionError::Conflict(source)) => Err(source),
    }
}

/// Project complete resident realizations back into requested coordinates.
fn available_cover_from_semantics<S, L>(
    snapshot: &S,
    semantics: &CollectionSemantics,
    cover: &Cover<L>,
) -> Result<Cover<L>, CoverAvailabilityError<S::RecordsError, S::Err>>
where
    S: BlobStoreGet + BlobStoreList + BlobStoreMeta + CollectionRead,
    L: CollectionEncoding,
{
    let collection = cover.collection().handle();
    let mut complete = Vec::new();
    for member in semantics.members(collection).into_iter().flatten().copied() {
        match collection_member_structural_availability::<L, _>(member, snapshot) {
            Ok(CollectionMemberAvailability::Complete) => complete.push((collection, member)),
            Ok(CollectionMemberAvailability::Absent)
            | Ok(CollectionMemberAvailability::Incomplete)
            | Ok(CollectionMemberAvailability::Unusable) => {}
            Err(source) => {
                return Err(CoverAvailabilityError::Residency { member, source });
            }
        }
    }

    let supporting = semantics.supporting_data_for(complete);
    Ok(Cover::from_data(
        cover.collection(),
        cover
            .data_members()
            .filter(|member| supporting.contains(member)),
    ))
}

/// Resolve one already-discovered exact payload cover to a private physical
/// decomposition.
///
/// Stored equations describe support; blob metadata describes residency. This
/// lookup performs no collection algebra and never reads a payload merely to
/// prove work which was already materialized. The eventual [`TryFromCover`]
/// implementation interprets exactly the physical members selected here;
/// eager views may decode them, while lazy views may retain their shards.
fn resolve_physical_cover_from_observation<S, L, ViewError, EvidenceError>(
    snapshot: &S,
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

    let semantics = resolve_cover_semantics(&discovered, &cover)
        .map_err(CollectionMaterializationError::ResolutionConflict)?;

    // Equation semantics and blob residency are orthogonal. Select from every
    // currently complete resident semantic member. Absent roots and incomplete
    // Merkle closures remain uncovered obligations, allowing the physical
    // cover algorithm to fall back to finer support-equivalent members. Exact
    // semantic validation still belongs to the eventual view.
    let mut resident_roots = BTreeSet::new();
    for data in semantics.members(collection).into_iter().flatten().copied() {
        if matches!(reader.metadata(Handle::<L>::from_hash(data)), Ok(Some(_))) {
            resident_roots.insert(data);
        }
    }

    let selected =
        collection_complete_physical_cover::<L, _>(&semantics, collection, &resident_roots, reader);
    if selected.physical.missing.is_empty() {
        return Ok(Cover::from_data(
            cover.collection(),
            selected.physical.cover,
        ));
    }
    if let Some((member, source)) = selected.unusable {
        return Err(CollectionMaterializationError::InvalidMember { member, source });
    }

    // Report missing support in the caller's semantic coordinates, never in
    // the private physical frontier selected while searching. Metadata errors
    // retain the historical materialization behavior of counting as absent;
    // callers that need the distinction use `available`, which propagates it.
    let complete = semantics
        .members(collection)
        .into_iter()
        .flatten()
        .copied()
        .filter(|member| {
            matches!(
                collection_member_availability::<L, _>(*member, reader),
                Ok(CollectionMemberAvailability::Complete)
            )
        })
        .map(|member| (collection, member));
    let supporting = semantics.supporting_data_for(complete);
    let obligations = cover
        .data_members()
        .filter(|member| !supporting.contains(member))
        .collect();

    Err(CollectionMaterializationError::Missing {
        obligations,
        dependencies: selected.dependencies,
    })
}

/// Materialize through the private physical decomposition selected from the
/// same immutable snapshot.
fn materialize_cover_from_observation<S, L, V, EvidenceError>(
    snapshot: &S,
    discovered: DiscoveredCollectionRecords,
    cover: Cover<L>,
) -> Result<
    V,
    CollectionMaterializationError<
        S::RecordsError,
        S::GetError<Infallible>,
        V::Error,
        EvidenceError,
    >,
>
where
    S: BlobStoreGet + BlobStoreMeta + CollectionRead,
    L: CollectionEncoding,
    V: TryFromCover<L>,
{
    let physical = resolve_physical_cover_from_observation::<S, L, V::Error, EvidenceError>(
        snapshot, discovered, cover,
    )?;
    V::try_from_cover(&physical, snapshot).map_err(CollectionMaterializationError::from)
}
