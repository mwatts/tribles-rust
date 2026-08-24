//! Narrow facade for one scoped collection.
//!
//! [`Collection`] owns the storage, canonical `SimpleArchive`-union
//! descriptor, and signing key needed to publish [`Fragment`] values and read
//! the complete known union admitted by an explicit capability presentation or
//! by an explicitly open policy. It is not a repository abstraction: it has no
//! head, branch, CAS, retry, ambient authority lookup, or planning policy.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::Blob;
use crate::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProof,
    CapabilityProofError, CapabilityResource,
};
use crate::clock;
use crate::id::Id;
use crate::inline::encodings::ed25519::ED25519PublicKey;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::repo::{
    BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut, StorageClose, StorageFlush,
};
// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use crate::collection::reach;
use crate::collection::records::CollectionName;
use crate::trible::{Fragment, TribleSet};

use super::simplearchive_union::{
    self, MaterializationError, PublicationError, SimpleArchiveUnionValidationError,
};
use super::{
    collection_physical_cover, discover_collection_records_authorized,
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
    subject: VerifyingKey,
    proof: CapabilityProof,
}

impl CapabilityPresentation {
    /// Pair an expected leaf subject with one untrusted owned proof.
    pub fn new(subject: VerifyingKey, proof: CapabilityProof) -> Self {
        Self { subject, proof }
    }

    /// Exact leaf subject this presentation is expected to authorize.
    pub fn subject(&self) -> VerifyingKey {
        self.subject
    }

    /// Candidate root-to-leaf proof, verified afresh for each operation.
    pub fn proof(&self) -> &CapabilityProof {
        &self.proof
    }

    /// Consume the presentation into its expected subject and proof.
    pub fn into_parts(self) -> (VerifyingKey, CapabilityProof) {
        (self.subject, self.proof)
    }
}

/// Explicit signer-admission policy for one collection facade.
///
/// `Open` admits every strictly self-signed commit for the exact descriptor.
/// `Capability` admits exactly the subjects whose supplied proofs verify
/// against `trust_root`; it never enumerates a store looking for more grants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionAdmission {
    /// Admit every strictly verified commit signer.
    Open,
    /// Admit exactly the subjects established by these presentations.
    Capability {
        /// External trust root against which every proof is verified.
        trust_root: VerifyingKey,
        /// Explicit candidate proofs. Every entry must verify or the operation
        /// fails rather than silently omitting it.
        presentations: Vec<CapabilityPresentation>,
    },
}

impl CollectionAdmission {
    /// Construct explicitly open admission.
    pub const fn open() -> Self {
        Self::Open
    }

    /// Construct explicit capability admission.
    pub fn capability(
        trust_root: VerifyingKey,
        presentations: Vec<CapabilityPresentation>,
    ) -> Self {
        Self::Capability {
            trust_root,
            presentations,
        }
    }

    /// Descriptor authority declared by this policy.
    pub fn trust_root(&self) -> Option<VerifyingKey> {
        match self {
            Self::Open => None,
            Self::Capability { trust_root, .. } => Some(*trust_root),
        }
    }

    /// Explicit presentations, empty for open admission.
    pub fn presentations(&self) -> &[CapabilityPresentation] {
        match self {
            Self::Open => &[],
            Self::Capability { presentations, .. } => presentations,
        }
    }
}

/// One explicitly supplied capability presentation failed verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionAdmissionError {
    presentation: usize,
    subject: VerifyingKey,
    source: CapabilityProofError,
}

impl CollectionAdmissionError {
    /// Zero-based index of the invalid presentation.
    pub const fn presentation(&self) -> usize {
        self.presentation
    }

    /// Expected leaf subject paired with the invalid proof.
    pub fn subject(&self) -> VerifyingKey {
        self.subject
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
            "capability presentation {} for subject {} is invalid: {}",
            self.presentation,
            hex::encode_upper(self.subject.to_bytes()),
            self.source,
        )
    }
}

impl Error for CollectionAdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// A scoped `SimpleArchive`-union collection and one prospective writer.
///
/// Construction is pure with respect to `storage`: the canonical descriptor
/// is derived in memory and is not inserted until a [`commit`](Self::commit)
/// publication begins. The signing key is not ambient authority: ordinary
/// reads admit signers according to `admission`, and publication proves the
/// local signing key is in that same admitted set before writing anything.
pub struct Collection<S> {
    storage: S,
    descriptor: Fragment,
    signing_key: SigningKey,
    admission: CollectionAdmission,
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
    commits: Vec<CollectionCommit>,
    reader: R,
}

impl<R> CollectionSnapshot<R> {
    /// Materialized union admitted by this snapshot's exact commit set.
    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    /// Exact admitted commits, ordered by intrinsic record id.
    pub fn commits(&self) -> &[CollectionCommit] {
        &self.commits
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
    pub fn into_parts(self) -> (TribleSet, Vec<CollectionCommit>, R) {
        (self.facts, self.commits, self.reader)
    }
}

/// Failure to discover one exact admitted commit ticket.
#[derive(Debug)]
pub enum CollectionTicketError<RecordsError> {
    /// One explicitly supplied capability proof was invalid at this operation's
    /// single clock observation.
    Admission(CollectionAdmissionError),
    /// Target collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
}

impl<RecordsError> fmt::Display for CollectionTicketError<RecordsError>
where
    RecordsError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(source) => source.fmt(formatter),
            Self::Discovery(source) => source.fmt(formatter),
        }
    }
}

impl<RecordsError> Error for CollectionTicketError<RecordsError>
where
    RecordsError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::Discovery(source) => Some(source),
        }
    }
}

/// Failure to publish one collection element through explicit admission.
#[derive(Debug)]
pub enum CollectionCommitError<PutError, InsertError> {
    /// One explicitly supplied capability proof was invalid.
    Admission(CollectionAdmissionError),
    /// The facade's signing key is absent from the admitted signer set.
    WriteDenied {
        /// Prospective commit signer.
        writer: Inline<ED25519PublicKey>,
        /// Exact collection descriptor for which the writer was denied.
        collection: CollectionHandle,
    },
    /// Admission succeeded, but canonical fragment publication failed.
    Publication(PublicationError<PutError, InsertError>),
}

impl<PutError, InsertError> fmt::Display for CollectionCommitError<PutError, InsertError>
where
    PutError: fmt::Display,
    InsertError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(source) => source.fmt(formatter),
            Self::WriteDenied { writer, collection } => write!(
                formatter,
                "writer {} is not admitted to WRITE collection {}",
                hex::encode_upper(writer.raw),
                hex::encode_upper(collection.raw),
            ),
            Self::Publication(source) => source.fmt(formatter),
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
            Self::Admission(source) => Some(source),
            Self::WriteDenied { .. } => None,
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

impl<RecordsError, ReaderError, MetaError, GetError> From<CollectionTicketError<RecordsError>>
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
{
    fn from(source: CollectionTicketError<RecordsError>) -> Self {
        match source {
            CollectionTicketError::Admission(source) => Self::Admission(source),
            CollectionTicketError::Discovery(source) => Self::Discovery(source),
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

impl<S> Collection<S> {
    /// Construct a collection facade without reading from or writing to
    /// `storage`.
    ///
    /// `namespace` distinguishes this root's name. It does not grant
    /// authority. `admission` independently determines both runtime signer
    /// admission and the descriptor's optional authority fact: open admission
    /// writes none, while capability admission writes exactly its trust root.
    /// There is therefore no separately supplied authority value which could
    /// disagree with the policy actually enforced.
    ///
    /// `reach` states whether this collection travels, and there is no
    /// default because the answer is part of the collection's name: a private
    /// and a public collection of the same content are different collections
    /// with different handles. Saying it here is also what removed the old
    /// silent failure -- reach used to be a separate signed grant that nothing
    /// in production ever minted, so publishing replicated nothing and
    /// reported no error. There is no second act left to forget.
    pub fn new(
        storage: S,
        name: &CollectionName,
        namespace: VerifyingKey,
        signing_key: SigningKey,
        reach: Fragment,
        admission: CollectionAdmission,
    ) -> Self {
        let authority = admission.trust_root();
        Self {
            storage,
            descriptor: simplearchive_union::descriptor(name, namespace, authority, reach),
            signing_key,
            admission,
        }
    }

    /// Canonical collection descriptor facts built from the constructor name
    /// and namespace, with authority fixed by admission.
    pub fn descriptor(&self) -> &Fragment {
        &self.descriptor
    }

    /// Content identity of this facade's collection.
    ///
    /// The facade holds the descriptor it will publish, so naming the
    /// collection before any write means hashing those facts here. Once a
    /// commit is published the stored blob's own handle is the same value.
    pub fn collection(&self) -> CollectionHandle {
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.descriptor.facts().clone())
            .get_handle()
    }

    /// Explicit admission policy enforced by this facade.
    pub fn admission(&self) -> &CollectionAdmission {
        &self.admission
    }

    /// Borrow the underlying storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Mutably borrow the underlying storage.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Consume the facade and recover its underlying storage.
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Verify every explicit presentation at one shared operation instant.
    ///
    /// `None` denotes open admission. `Some(empty)` is intentionally distinct:
    /// it is a valid closed capability policy which admits nobody.
    fn admitted_subjects_at(
        &self,
        instant: hifitime::Epoch,
    ) -> Result<Option<BTreeSet<Inline<ED25519PublicKey>>>, CollectionAdmissionError> {
        let CollectionAdmission::Capability {
            trust_root,
            presentations,
        } = &self.admission
        else {
            return Ok(None);
        };

        let atom = CapabilityAtom::new(
            CapabilityAction::new(ACTION_WRITE),
            CapabilityResource::from(self.collection()),
        );
        let mut admitted = BTreeSet::new();
        for (presentation_index, presentation) in presentations.iter().enumerate() {
            presentation
                .proof()
                .verify_claim(
                    *trust_root,
                    instant,
                    CapabilityClaim::new(presentation.subject(), atom, CapabilityMode::Invoke),
                )
                .map_err(|source| CollectionAdmissionError {
                    presentation: presentation_index,
                    subject: presentation.subject(),
                    source,
                })?;
            admitted.insert(Inline::new(presentation.subject().to_bytes()));
        }
        Ok(Some(admitted))
    }
}

impl<S> Collection<S>
where
    S: CollectionStore,
{
    /// Discover the exact strictly verified admitted commits.
    ///
    /// The operation observes the clock once, verifies every explicit proof
    /// against [`ACTION_WRITE`] on this exact descriptor, then discovers target
    /// records. Open admission accepts every strict signer; capability
    /// admission accepts exactly the presented subjects. Unadmitted and invalid
    /// commits remain inert. The returned ticket is ordered by intrinsic id.
    ///
    /// This is a known-prefix observation rather than a global-latest
    /// transaction. A commit inserted after target discovery appears in a
    /// later ticket. Newly acquired evidence participates only after the
    /// caller supplies it in a later facade's explicit admission policy.
    pub fn ticket(
        &mut self,
    ) -> Result<Vec<CollectionCommit>, CollectionTicketError<S::RecordsError>> {
        let instant = clock::epoch_now();
        let (_, commits) = self.discover_admitted_commits_at(instant)?;
        Ok(commits)
    }

    fn discover_admitted_commits_at(
        &mut self,
        instant: hifitime::Epoch,
    ) -> Result<
        (DiscoveredCollectionRecords, Vec<CollectionCommit>),
        CollectionTicketError<S::RecordsError>,
    > {
        let collection = self.collection();
        let admitted = self
            .admitted_subjects_at(instant)
            .map_err(CollectionTicketError::Admission)?;
        let discovered =
            discover_collection_records_authorized(&mut self.storage, collection, |subject| {
                admitted
                    .as_ref()
                    .map_or(true, |subjects| subjects.contains(subject))
            })
            .map_err(CollectionTicketError::Discovery)?;
        let commits = discovered.commits().to_vec();

        Ok((discovered, commits))
    }
}

impl<S> Collection<S>
where
    S: BlobStorePut + CollectionStore,
{
    /// Publish one self-contained fragment under explicit admission.
    ///
    /// The operation observes the clock once and verifies every explicit
    /// capability proof before any descriptor, attachment, data, metadata, or
    /// record is written. The local signing key must be in the resulting
    /// admitted set; open admission includes it automatically. Facts are the
    /// collection element, metafacts are commit metadata, and fragment
    /// attachments are then staged through
    /// the same ordered, content-addressed path as
    /// [`simplearchive_union::publish_fragment_commit`]. Repeating identical
    /// input is idempotent; distinct commits coexist without selecting a head.
    /// The parameter is deliberately `Fragment`, rather than `Into<Fragment>`,
    /// so a bare fact set cannot accidentally publish without its metafacts.
    pub fn commit(
        &mut self,
        fragment: Fragment,
    ) -> Result<CollectionCommit, CollectionCommitError<S::PutError, S::InsertError>> {
        let instant = clock::epoch_now();
        let collection = self.collection();
        let writer = Inline::new(self.signing_key.verifying_key().to_bytes());
        let admitted = self
            .admitted_subjects_at(instant)
            .map_err(CollectionCommitError::Admission)?;
        if admitted
            .as_ref()
            .is_some_and(|subjects| !subjects.contains(&writer))
        {
            return Err(CollectionCommitError::WriteDenied { writer, collection });
        }
        simplearchive_union::publish_fragment_commit(
            &mut self.storage,
            &self.descriptor,
            fragment,
            &self.signing_key,
        )
        .map_err(CollectionCommitError::Publication)
    }
}

impl<S> Collection<S>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    /// Capture one coherent known-prefix snapshot of this admitted collection.
    ///
    /// The call observes the clock once, verifies every explicit capability
    /// presentation, discovers commits from exactly the resulting signer set,
    /// and then opens one target blob-reader snapshot. Open admission instead
    /// accepts every strictly verified signer. The returned facts are
    /// materialized solely from that commit set; unadmitted commits remain
    /// inert even when every byte they name is locally resident.
    ///
    /// This is not a global-latest transaction: a later call may observe more
    /// commits. It is a coherent admission boundary for consumers that need
    /// to carry facts, source commits, and the validating blob view together.
    pub fn snapshot(
        &mut self,
    ) -> Result<
        CollectionSnapshot<S::Reader>,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    > {
        let instant = clock::epoch_now();
        let (discovered, commits) = self
            .discover_admitted_commits_at(instant)
            .map_err(CollectionMaterializationError::from)?;
        Self::snapshot_from_observation(&mut self.storage, &self.descriptor, discovered, commits)
    }

    /// Materialize the complete known `TribleSet` admitted by this facade.
    ///
    /// One call first discovers a deterministic observed view of native
    /// collection records, then opens a blob-reader snapshot. Every strictly
    /// signed commit in that record view whose signer is admitted for exact
    /// [`ACTION_WRITE`] on this collection is mandatory membership. All
    /// admitted commit dependencies are exact-validated and
    /// fail loud. Merge validation is restricted to the subgraph between
    /// authenticated leaves and resident result artifacts, while allowing
    /// nonresident intermediate equations. Exact resident results may replace
    /// redundant leaves in the physical cover; corrupt artifacts and missing,
    /// invalid, or irrelevant unsigned equations are cache misses and the
    /// committed leaves remain authoritative.
    /// Derivations are not admitted by this `SimpleArchive`-only facade.
    /// This boundary assumes records have already passed deployment admission:
    /// it prevents unsigned records from acquiring semantic authority, but does
    /// not bound CPU spent validating arbitrarily many admitted equations.
    ///
    /// This is a **known-prefix** read, not a global-latest transaction:
    /// [`CollectionStore`] does not promise a coherent snapshot under
    /// concurrent insertion. A commit first observed after this discovery pass
    /// appears on a later call. The returned set is nevertheless complete for
    /// all admitted commits observed by this pass, or the call returns an
    /// error instead of a partial set. If no admitted commit is observed,
    /// the result is empty without fetching the target descriptor.
    pub fn materialize(
        &mut self,
    ) -> Result<
        TribleSet,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    > {
        let instant = clock::epoch_now();
        let (discovered, commits) = self
            .discover_admitted_commits_at(instant)
            .map_err(CollectionMaterializationError::from)?;
        if commits.is_empty() {
            return Ok(TribleSet::new());
        }
        Self::snapshot_from_observation(&mut self.storage, &self.descriptor, discovered, commits)
            .map(CollectionSnapshot::into_facts)
    }

    /// Materialize one already-discovered exact commit frontier.
    ///
    /// Both ordinary and exact-ticket facades use this single validator
    /// so descriptor, mandatory dependency, merge-cover, and reader-snapshot
    /// semantics cannot drift apart.
    pub(crate) fn snapshot_from_observation(
        storage: &mut S,
        descriptor: &Fragment,
        discovered: DiscoveredCollectionRecords,
        commits: Vec<CollectionCommit>,
    ) -> Result<
        CollectionSnapshot<S::Reader>,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    > {
        let collection =
            crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone())
                .get_handle();
        let admitted: BTreeSet<_> = commits.iter().map(CollectionCommit::id).collect();

        let reader = storage
            .reader()
            .map_err(CollectionMaterializationError::Reader)?;

        if commits.is_empty() {
            return Ok(CollectionSnapshot {
                facts: TribleSet::new(),
                commits,
                reader,
            });
        }

        // The descriptor handle is the collection identity. Once an admitted
        // commit makes this collection nonempty, its descriptor is mandatory
        // ground truth just like the signed data and metadata below. Fetch by
        // the exact handle, recompute the identity rather than trusting a
        // cached handle, decode the canonical archive, and bind it back to the
        // facade's expected semantics before interpreting any element.
        let descriptor_blob: Blob<SimpleArchive> = reader.get(collection).map_err(|source| {
            CollectionMaterializationError::DescriptorGet { collection, source }
        })?;
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
            reader.get(metadata).map_err(|source| {
                CollectionMaterializationError::CommitMetadataGet {
                    commit: claim.id(),
                    metadata,
                    source,
                }
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
            let actual_metadata =
                Blob::<SimpleArchive>::new(metadata_blob.bytes.clone()).get_handle();
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
            &commits,
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
            &commits,
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
        let mut expected_hashes =
            BTreeMap::<(CollectionData, CollectionData), CollectionData>::new();
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
                    CollectionValidationRequest::Commit { .. } => {
                        CollectionClaimValidation::Accepted
                    }
                    CollectionValidationRequest::Merge { claim, .. }
                        if accepted_merges.contains(&claim.id()) =>
                    {
                        CollectionClaimValidation::Accepted
                    }
                    CollectionValidationRequest::Merge { .. }
                    | CollectionValidationRequest::Derive { .. } => {
                        CollectionClaimValidation::Pending
                    }
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
                        if actual_data == data
                            && simplearchive_union::validate_element(&actual).is_ok()
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
                let union =
                    simplearchive_union::join_many_bytes(members.iter().map(|(_, blob)| *blob))
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
            commits,
            reader,
        })
    }
}

impl<S> Collection<S>
where
    S: StorageClose,
{
    /// Consume the facade and close its underlying storage.
    pub fn close(self) -> Result<(), S::Error> {
        self.storage.close()
    }
}

impl<S> Collection<S>
where
    S: StorageFlush,
{
    /// Explicitly make pending storage writes crash-durable.
    ///
    /// Ordinary collection publication deliberately does not call this. The
    /// caller chooses the durability cadence and may batch any number of
    /// commits or equations behind one barrier.
    pub fn flush(&mut self) -> Result<(), S::Error> {
        self.storage.flush()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use std::cell::Cell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::convert::Infallible;

    use super::*;

    use crate::blob::encodings::{utf8string::UTF8String, UnknownBlob};
    use crate::blob::{BlobEncoding, Bytes, IntoBlob};
    use crate::capability::{
        CapabilityGrant, CapabilityProofStep, CapabilityValidity, KIND_CAPABILITY_CLAIM,
    };
    use crate::collection::descriptor::identity_for_tests;
    use crate::collection::{discover_collection_records, CollectionMerge, CollectionRecord};
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    fn test_trust_key() -> SigningKey {
        SigningKey::from_bytes(&[1; 32])
    }

    fn test_namespace() -> VerifyingKey {
        test_trust_key().verifying_key()
    }

    fn test_name() -> CollectionName {
        CollectionName::new("owned").unwrap()
    }

    fn other_name() -> CollectionName {
        CollectionName::new("other").unwrap()
    }

    struct AppendAfterFirstDiscovery {
        inner: MemoryRepo,
        initially_visible: BTreeSet<Id>,
        records_calls: usize,
    }

    impl CollectionStore for AppendAfterFirstDiscovery {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            let mut records = self
                .inner
                .records()?
                .collect::<Result<Vec<_>, Infallible>>()?;
            self.records_calls += 1;
            // Admission is supplied explicitly, so `snapshot` performs one
            // target discovery. Admit only the earlier frontier on that pass.
            if self.records_calls == 1 {
                records.retain(|record| self.initially_visible.contains(&record.id()));
            }
            Ok(records.into_iter().map(Ok).collect::<Vec<_>>().into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.inner.insert(record)
        }
    }

    impl BlobStorePut for AppendAfterFirstDiscovery {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
        where
            S: BlobEncoding + 'static,
            T: IntoBlob<S>,
            Handle<S>: InlineEncoding,
        {
            self.inner.put(item)
        }
    }

    impl BlobStore for AppendAfterFirstDiscovery {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.inner.reader()
        }
    }

    fn fragment(entity: u8, attachment: bool) -> Fragment {
        let mut row = [entity; TRIBLE_LEN];
        row[16..32].fill(1);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        let mut fragment = Fragment::from(facts);
        if attachment {
            let _: Inline<Handle<UTF8String>> = fragment.put("one attachment".to_owned());
        }
        fragment
    }

    fn archive(entity: u8) -> Blob<SimpleArchive> {
        fragment(entity, false).facts().clone().to_blob()
    }

    fn fragment_with_metadata(entity: u8, metadata_entity: u8) -> Fragment {
        let mut built = fragment(entity, false);
        *built.metafacts_mut() += fragment(metadata_entity, false).into_facts();
        built
    }

    fn write_atom(collection: CollectionHandle) -> CapabilityAtom {
        CapabilityAtom::new(
            CapabilityAction::new(ACTION_WRITE),
            CapabilityResource::from(collection),
        )
    }

    fn root_presentation(
        root: &SigningKey,
        subject: VerifyingKey,
        collection: CollectionHandle,
        mode: CapabilityMode,
    ) -> CapabilityPresentation {
        CapabilityPresentation::new(
            subject,
            CapabilityProof::new(vec![CapabilityProofStep::issue(
                root,
                CapabilityGrant::root(subject, write_atom(collection), mode, None),
            )]),
        )
    }

    fn capability_descriptor(name: &CollectionName, trust_root: VerifyingKey) -> Fragment {
        simplearchive_union::descriptor(name, test_namespace(), Some(trust_root), reach::private())
    }

    fn capability_admission(
        root: &SigningKey,
        subjects: impl IntoIterator<Item = VerifyingKey>,
        collection: CollectionHandle,
    ) -> CollectionAdmission {
        CollectionAdmission::capability(
            root.verifying_key(),
            subjects
                .into_iter()
                .map(|subject| root_presentation(root, subject, collection, CapabilityMode::Invoke))
                .collect(),
        )
    }

    fn open_collection(name: &CollectionName, signing_key: SigningKey) -> Collection<MemoryRepo> {
        Collection::new(
            MemoryRepo::default(),
            name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        )
    }

    fn keep_only<I>(store: &mut MemoryRepo, handles: I)
    where
        I: IntoIterator<Item = Inline<Handle<UnknownBlob>>>,
    {
        store.blobs.keep(handles);
    }

    fn invalid_signature(commit: CollectionCommit) -> CollectionCommit {
        let (r, mut s) = commit.signature();
        s.raw[0] ^= 1;
        CollectionCommit::from_parts(
            commit.collection(),
            commit.data(),
            commit.metadata(),
            commit.public_key(),
            r,
            s,
        )
    }

    #[test]
    fn open_ticket_discovers_every_exact_strictly_verified_commit() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let collection_id = identity_for_tests(&simplearchive_union::descriptor(
            &name,
            test_namespace(),
            None,
            reach::private(),
        ));
        let other_collection = identity_for_tests(&simplearchive_union::descriptor(
            &other_name(),
            test_namespace(),
            None,
            reach::private(),
        ));
        let metadata = super::super::empty_metadata_handle();
        let first =
            CollectionCommit::sign(&signing_key, collection_id, Inline::new([10; 32]), metadata);
        let second =
            CollectionCommit::sign(&signing_key, collection_id, Inline::new([11; 32]), metadata);
        let foreign_signer =
            CollectionCommit::sign(&foreign_key, collection_id, Inline::new([12; 32]), metadata);
        let foreign_collection = CollectionCommit::sign(
            &signing_key,
            other_collection,
            Inline::new([13; 32]),
            metadata,
        );
        let invalid = invalid_signature(CollectionCommit::sign(
            &signing_key,
            collection_id,
            Inline::new([14; 32]),
            metadata,
        ));

        let mut store = MemoryRepo::default();
        for record in [
            CollectionRecord::Commit(second),
            CollectionRecord::Commit(foreign_signer),
            CollectionRecord::Commit(invalid),
            CollectionRecord::Commit(first),
            CollectionRecord::Commit(foreign_collection),
        ] {
            store.insert(record).unwrap();
        }
        let mut collection = Collection::new(
            store,
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );

        let ticket = collection.ticket().unwrap();
        let mut expected = vec![first, second, foreign_signer];
        expected.sort_unstable_by_key(CollectionCommit::id);

        assert_eq!(ticket, expected);
    }

    #[test]
    fn empty_capability_admission_yields_an_empty_ticket() {
        let root = test_trust_key();
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &test_name(),
            test_namespace(),
            SigningKey::from_bytes(&[7; 32]),
            reach::private(),
            CollectionAdmission::capability(root.verifying_key(), Vec::new()),
        );

        assert!(collection.ticket().unwrap().is_empty());
    }

    #[test]
    fn construction_is_pure_and_derives_the_scoped_descriptor() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );

        assert_eq!(
            collection.descriptor(),
            &simplearchive_union::descriptor(&name, test_namespace(), None, reach::private(),)
        );
        assert_eq!(collection.admission(), &CollectionAdmission::Open);
        assert!(collection.storage().blobs.is_empty());
        assert!(collection.storage_mut().records().unwrap().next().is_none());

        collection.close().unwrap();
    }

    #[test]
    fn borrowed_facade_reuses_one_open_storage() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut storage = MemoryRepo::default();

        {
            let mut collection = Collection::new(
                &mut storage,
                &name,
                test_namespace(),
                signing_key.clone(),
                reach::private(),
                CollectionAdmission::Open,
            );
            collection.commit(expected.clone()).unwrap();
            assert_eq!(collection.materialize().unwrap(), expected.facts().clone());
        }

        let target = identity_for_tests(&simplearchive_union::descriptor(
            &name,
            test_namespace(),
            None,
            reach::private(),
        ));
        let discovered = discover_collection_records(&mut storage).unwrap();
        assert_eq!(
            discovered
                .commits()
                .iter()
                .filter(|commit| commit.collection() == target)
                .count(),
            1
        );

        let mut collection = Collection::new(
            &mut storage,
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );
        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn open_ticket_attaches_through_a_matching_exact_facade() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut collection = open_collection(&name, signing_key);
        let commit = collection.commit(expected.clone()).unwrap();
        let ticket = collection.ticket().unwrap();
        let exact = simplearchive_union::SimpleArchiveCollection::new(
            name.clone(),
            test_namespace(),
            collection.admission().trust_root(),
            reach::private(),
        );
        let succinct = crate::collection::succinctarchive_union::SuccinctArchiveCollection::new(
            name,
            test_namespace(),
            collection.admission().trust_root(),
            reach::private(),
            None,
            reach::private(),
        );

        assert_eq!(ticket, vec![commit]);
        assert_eq!(
            exact
                .attach_exact(collection.storage_mut(), &ticket)
                .unwrap(),
            expected.into_facts()
        );
        succinct
            .ensure_exact(collection.storage_mut(), &ticket)
            .unwrap();
    }

    #[test]
    fn capability_admission_gates_exact_presented_subjects() {
        let name = test_name();
        let root = test_trust_key();
        let local_key = SigningKey::from_bytes(&[7; 32]);
        let remote_key = SigningKey::from_bytes(&[9; 32]);
        let stranger_key = SigningKey::from_bytes(&[11; 32]);
        let descriptor = capability_descriptor(&name, root.verifying_key());
        let collection_handle = identity_for_tests(&descriptor);
        let mut storage = MemoryRepo::default();
        let local_commit = simplearchive_union::publish_fragment_commit(
            &mut storage,
            &descriptor,
            fragment(1, false),
            &local_key,
        )
        .unwrap();
        let remote_commit = simplearchive_union::publish_fragment_commit(
            &mut storage,
            &descriptor,
            fragment(2, false),
            &remote_key,
        )
        .unwrap();
        let stranger_commit = simplearchive_union::publish_fragment_commit(
            &mut storage,
            &descriptor,
            fragment(3, false),
            &stranger_key,
        )
        .unwrap();

        let admission = capability_admission(
            &root,
            [local_key.verifying_key(), remote_key.verifying_key()],
            collection_handle,
        );
        let mut local = Collection::new(
            storage,
            &name,
            test_namespace(),
            local_key,
            reach::private(),
            admission,
        );

        assert_eq!(
            discover_collection_records(local.storage_mut())
                .unwrap()
                .commits()
                .iter()
                .filter(|commit| commit.collection() == identity_for_tests(&descriptor))
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([local_commit.id(), remote_commit.id(), stranger_commit.id()])
        );

        assert_eq!(
            local
                .ticket()
                .unwrap()
                .iter()
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([local_commit.id(), remote_commit.id()])
        );

        let snapshot = local.snapshot().unwrap();
        let mut expected = fragment(1, false).facts().clone();
        expected += fragment(2, false).facts().clone();
        assert_eq!(snapshot.facts(), &expected);
        assert!(!snapshot
            .facts()
            .iter()
            .any(|trible| fragment(3, false).facts().contains(trible)));
    }

    #[test]
    fn newly_presented_proof_activates_an_already_resident_commit() {
        let name = test_name();
        let root = test_trust_key();
        let writer = SigningKey::from_bytes(&[9; 32]);
        let observer = SigningKey::from_bytes(&[7; 32]);
        let descriptor = capability_descriptor(&name, root.verifying_key());
        let collection_handle = identity_for_tests(&descriptor);
        let expected = fragment(2, false);
        let mut storage = MemoryRepo::default();
        let commit = simplearchive_union::publish_fragment_commit(
            &mut storage,
            &descriptor,
            expected.clone(),
            &writer,
        )
        .unwrap();
        let mut collection = Collection::new(
            storage,
            &name,
            test_namespace(),
            observer.clone(),
            reach::private(),
            CollectionAdmission::capability(root.verifying_key(), Vec::new()),
        );

        assert!(collection.ticket().unwrap().is_empty());
        assert_eq!(collection.materialize().unwrap(), TribleSet::new());

        let storage = collection.into_storage();
        let admission = capability_admission(&root, [writer.verifying_key()], collection_handle);
        let mut collection = Collection::new(
            storage,
            &name,
            test_namespace(),
            observer,
            reach::private(),
            admission,
        );

        assert_eq!(collection.ticket().unwrap(), vec![commit]);
        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn empty_capability_admission_rejects_commit_before_collection_publication() {
        let name = test_name();
        let root = test_trust_key();
        let writer = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &name,
            test_namespace(),
            writer.clone(),
            reach::private(),
            CollectionAdmission::capability(root.verifying_key(), Vec::new()),
        );
        let before_blobs = collection.storage().blobs.len();
        let before_records = collection
            .storage_mut()
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let error = collection.commit(fragment(1, true)).unwrap_err();

        assert!(matches!(
            error,
            CollectionCommitError::WriteDenied { writer: denied, collection: target }
                if denied.raw == writer.verifying_key().to_bytes()
                    && target == collection.collection()
        ));
        assert_eq!(collection.storage().blobs.len(), before_blobs);
        assert_eq!(
            collection
                .storage_mut()
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            before_records
        );
    }

    #[test]
    fn delegated_invoke_capability_enables_commit_and_read() {
        let name = test_name();
        let root = test_trust_key();
        let delegate = SigningKey::from_bytes(&[6; 32]);
        let writer = SigningKey::from_bytes(&[7; 32]);
        let descriptor = capability_descriptor(&name, root.verifying_key());
        let target = identity_for_tests(&descriptor);
        let parent = CapabilityProofStep::issue(
            &root,
            CapabilityGrant::root(
                delegate.verifying_key(),
                write_atom(target),
                CapabilityMode::InvokeAndDelegate,
                None,
            ),
        );
        let child = CapabilityProofStep::issue(
            &delegate,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                writer.verifying_key(),
                write_atom(target),
                CapabilityMode::Invoke,
                None,
            ),
        );
        let admission = CollectionAdmission::capability(
            root.verifying_key(),
            vec![CapabilityPresentation::new(
                writer.verifying_key(),
                CapabilityProof::new(vec![parent, child]),
            )],
        );

        let expected = fragment(4, false);
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &name,
            test_namespace(),
            writer,
            reach::private(),
            admission,
        );
        collection.commit(expected.clone()).unwrap();

        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn bounded_invoke_and_delegate_capability_satisfies_minimum_invoke() {
        let name = test_name();
        let root = test_trust_key();
        let writer = SigningKey::from_bytes(&[7; 32]);
        let target = identity_for_tests(&capability_descriptor(&name, root.verifying_key()));
        let now_ns = clock::epoch_now().to_tai_duration().total_nanoseconds();
        let validity = CapabilityValidity::new(
            hifitime::Epoch::from_tai_duration(hifitime::Duration::from_total_nanoseconds(
                now_ns - 60_000_000_000,
            )),
            hifitime::Epoch::from_tai_duration(hifitime::Duration::from_total_nanoseconds(
                now_ns + 60_000_000_000,
            )),
        )
        .unwrap();
        let proof = CapabilityProof::new(vec![CapabilityProofStep::issue(
            &root,
            CapabilityGrant::root(
                writer.verifying_key(),
                write_atom(target),
                CapabilityMode::InvokeAndDelegate,
                Some(validity),
            ),
        )]);
        let admission = CollectionAdmission::capability(
            root.verifying_key(),
            vec![CapabilityPresentation::new(writer.verifying_key(), proof)],
        );
        let expected = fragment(5, false);
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &name,
            test_namespace(),
            writer,
            reach::private(),
            admission,
        );

        collection.commit(expected.clone()).unwrap();
        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn wrong_capability_atom_mode_or_subject_fails_loud_before_publication() {
        let name = test_name();
        let root = test_trust_key();
        let writer = SigningKey::from_bytes(&[7; 32]);
        let other_subject = SigningKey::from_bytes(&[8; 32]);
        let target = identity_for_tests(&capability_descriptor(&name, root.verifying_key()));
        let other = identity_for_tests(&capability_descriptor(&other_name(), root.verifying_key()));

        let cases = [
            CapabilityGrant::root(
                writer.verifying_key(),
                write_atom(target),
                CapabilityMode::Delegate,
                None,
            ),
            CapabilityGrant::root(
                writer.verifying_key(),
                CapabilityAtom::new(
                    CapabilityAction::new(KIND_CAPABILITY_CLAIM),
                    CapabilityResource::from(target),
                ),
                CapabilityMode::Invoke,
                None,
            ),
            CapabilityGrant::root(
                writer.verifying_key(),
                write_atom(other),
                CapabilityMode::Invoke,
                None,
            ),
            CapabilityGrant::root(
                other_subject.verifying_key(),
                write_atom(target),
                CapabilityMode::Invoke,
                None,
            ),
        ];

        for grant in cases {
            let admission = CollectionAdmission::capability(
                root.verifying_key(),
                vec![CapabilityPresentation::new(
                    writer.verifying_key(),
                    CapabilityProof::new(vec![CapabilityProofStep::issue(&root, grant)]),
                )],
            );
            let mut collection = Collection::new(
                MemoryRepo::default(),
                &name,
                test_namespace(),
                writer.clone(),
                reach::private(),
                admission,
            );
            let before_blobs = collection.storage().blobs.len();
            let before_records = collection
                .storage_mut()
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

            assert!(matches!(
                collection.commit(fragment(1, true)),
                Err(CollectionCommitError::Admission(CollectionAdmissionError {
                    source: CapabilityProofError::ClaimMismatch { .. },
                    ..
                }))
            ));
            assert_eq!(collection.storage().blobs.len(), before_blobs);
            assert_eq!(
                collection
                    .storage_mut()
                    .records()
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
                before_records
            );
        }
    }

    #[test]
    fn one_invalid_explicit_presentation_rejects_the_whole_admission_set() {
        let name = test_name();
        let root = test_trust_key();
        let writer = SigningKey::from_bytes(&[7; 32]);
        let target = identity_for_tests(&capability_descriptor(&name, root.verifying_key()));
        let valid = root_presentation(
            &root,
            writer.verifying_key(),
            target,
            CapabilityMode::Invoke,
        );
        let invalid = CapabilityPresentation::new(
            writer.verifying_key(),
            CapabilityProof::new(vec![CapabilityProofStep::issue(
                &root,
                CapabilityGrant::root(
                    writer.verifying_key(),
                    write_atom(target),
                    CapabilityMode::Delegate,
                    None,
                ),
            )]),
        );
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &name,
            test_namespace(),
            writer,
            reach::private(),
            CollectionAdmission::capability(root.verifying_key(), vec![valid, invalid]),
        );

        assert!(matches!(
            collection.ticket(),
            Err(CollectionTicketError::Admission(error)) if error.presentation() == 1
        ));
    }

    #[test]
    fn distinct_commits_coexist_and_repeated_commits_are_idempotent() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = open_collection(&name, signing_key);
        let first_fragment = fragment(1, true);
        let second_fragment = fragment(2, false);

        let first = collection.commit(first_fragment.clone()).unwrap();
        let after_first = collection.storage().blobs.len();
        let second = collection.commit(second_fragment).unwrap();
        let after_second = collection.storage().blobs.len();
        let repeated = collection.commit(first_fragment).unwrap();

        assert_eq!(repeated, first);
        assert_ne!(second, first);
        assert!(after_second > after_first);
        assert_eq!(collection.storage().blobs.len(), after_second);

        let descriptor = collection.descriptor().clone();
        let descriptor_handle = identity_for_tests(&descriptor);
        let reader = collection.storage_mut().reader().unwrap();
        let descriptor_blob: Blob<SimpleArchive> = reader.get(descriptor_handle).unwrap();
        assert_eq!(
            Blob::<SimpleArchive>::new(descriptor_blob.bytes.clone()).get_handle(),
            descriptor_handle
        );
        assert_eq!(
            <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(descriptor_blob)
                .unwrap(),
            *descriptor.facts()
        );

        let discovered = discover_collection_records(collection.storage_mut()).unwrap();
        assert_eq!(
            discovered
                .commits()
                .iter()
                .filter(|commit| commit.collection() == descriptor_handle)
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([first.id(), second.id()])
        );
        assert_eq!(
            collection
                .storage_mut()
                .records()
                .unwrap()
                .collect::<Result<Vec<CollectionRecord>, _>>()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn empty_snapshot_still_carries_a_reader() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            &test_name(),
            test_namespace(),
            SigningKey::from_bytes(&[7; 32]),
            reach::private(),
            CollectionAdmission::Open,
        );

        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), &TribleSet::new());
        assert!(snapshot.commits().is_empty());
        assert!(snapshot.reader().is_empty());
    }

    #[test]
    fn snapshot_keeps_one_record_frontier_when_a_commit_appears_before_reader_open() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let first = fragment(1, false);
        let second = fragment(2, false);
        let mut expected_all = first.facts().clone();
        expected_all += second.facts().clone();

        let mut seeded = Collection::new(
            MemoryRepo::default(),
            &name,
            test_namespace(),
            signing_key.clone(),
            reach::private(),
            CollectionAdmission::Open,
        );
        let first_commit = seeded.commit(first.clone()).unwrap();
        let second_commit = seeded.commit(second).unwrap();
        let storage = AppendAfterFirstDiscovery {
            inner: seeded.into_storage(),
            initially_visible: BTreeSet::from([first_commit.id()]),
            records_calls: 0,
        };
        let mut collection = Collection::new(
            storage,
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );

        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), first.facts());
        assert_eq!(snapshot.commits(), &[first_commit]);
        assert_eq!(collection.storage().records_calls, 1);

        // The later commit's bytes are already in the captured blob reader.
        // They remain semantically inert because its signed record was not in
        // the exact record frontier returned with this snapshot.
        let _: Blob<SimpleArchive> = snapshot
            .reader()
            .get(Handle::<SimpleArchive>::from_hash(second_commit.data()))
            .unwrap();

        let next = collection.snapshot().unwrap();
        assert_eq!(next.facts(), &expected_all);
        assert_eq!(
            next.commits()
                .iter()
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([first_commit.id(), second_commit.id()])
        );
        assert_eq!(collection.storage().records_calls, 2);
    }

    #[test]
    fn admitted_commits_materialize_completely_and_repeat_deterministically() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let first = fragment(1, false);
        let second = fragment(2, false);
        let mut expected = first.facts().clone();
        expected += second.facts().clone();

        collection.commit(first).unwrap();
        collection.commit(second).unwrap();

        let observed = collection.materialize().unwrap();
        assert_eq!(observed, expected);
        assert_eq!(collection.materialize().unwrap(), observed);
    }

    #[test]
    fn shared_commit_handles_are_validated_once_without_losing_provenance() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let descriptor =
            simplearchive_union::descriptor(&name, test_namespace(), None, reach::private());
        let first_data = archive(1);
        let second_data = archive(2);
        let first_metadata = archive(8);
        let second_metadata = archive(9);
        let mut storage = MemoryRepo::default();
        let first_commit = simplearchive_union::publish_commit(
            &mut storage,
            &descriptor,
            &first_data,
            &first_metadata,
            &signing_key,
        )
        .unwrap();
        let second_commit = simplearchive_union::publish_commit(
            &mut storage,
            &descriptor,
            &first_data,
            &second_metadata,
            &signing_key,
        )
        .unwrap();
        let third_commit = simplearchive_union::publish_commit(
            &mut storage,
            &descriptor,
            &second_data,
            &first_metadata,
            &signing_key,
        )
        .unwrap();
        let commits = [first_commit, second_commit, third_commit];

        let data_by_handle = BTreeMap::from([
            (
                Handle::<SimpleArchive>::to_hash(first_data.get_handle()),
                first_data.clone(),
            ),
            (
                Handle::<SimpleArchive>::to_hash(second_data.get_handle()),
                second_data.clone(),
            ),
        ]);
        let mut data_validations = BTreeMap::new();
        let mut metadata_validations = BTreeMap::new();
        let known = validate_unique_commit_dependencies(
            &commits,
            |commit| {
                *data_validations.entry(commit.data()).or_insert(0) += 1;
                Ok::<_, Infallible>(data_by_handle[&commit.data()].clone())
            },
            |commit| {
                *metadata_validations.entry(commit.metadata()).or_insert(0) += 1;
                Ok::<_, Infallible>(())
            },
        )
        .unwrap();
        assert_eq!(known.len(), 2);
        assert_eq!(
            known.keys().copied().collect::<BTreeSet<_>>(),
            data_by_handle.keys().copied().collect(),
        );
        assert!(data_validations.values().all(|count| *count == 1));
        assert_eq!(metadata_validations.len(), 2);
        assert!(metadata_validations.values().all(|count| *count == 1));

        let mut expected = first_data.clone().try_from_blob::<TribleSet>().unwrap();
        expected += second_data.clone().try_from_blob::<TribleSet>().unwrap();

        let mut collection = Collection::new(
            storage,
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );
        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(
            snapshot
                .commits()
                .iter()
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            commits.iter().map(CollectionCommit::id).collect(),
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_dependency_fetches_keep_serial_stage_order() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let mut commits = [
            collection.commit(fragment(1, false)).unwrap(),
            collection.commit(fragment(2, false)).unwrap(),
        ];
        commits.sort_unstable_by_key(CollectionCommit::id);
        let lower = commits[0];
        let descriptor = collection.descriptor().clone();
        let reader = collection.storage_mut().reader().unwrap();

        let calls = Cell::new(1usize);
        for corrupt_data in [false, true] {
            // The descriptor consumed get #1. The one-shot failure on get #3
            // must stay on lower metadata; when lower data is also corrupt,
            // its validation failure must still take precedence. Cell keeps
            // both reader callbacks deliberately non-Sync.
            calls.set(1);
            let observed = validate_unique_commit_dependencies_parallel(
                &commits,
                &descriptor,
                |commit| {
                    calls.set(calls.get() + 1);
                    if corrupt_data && commit.id() == lower.id() {
                        Ok(Blob::with_handle(
                            Bytes::from(vec![0; 63]),
                            Handle::<SimpleArchive>::from_hash(commit.data()),
                        ))
                    } else if calls.get() == 3 {
                        Err(("data", commit.id()))
                    } else {
                        Ok(reader
                            .get(Handle::<SimpleArchive>::from_hash(commit.data()))
                            .unwrap())
                    }
                },
                |commit| {
                    calls.set(calls.get() + 1);
                    if calls.get() == 3 {
                        Err(("metadata", commit.id()))
                    } else {
                        Ok(reader.get(commit.metadata()).unwrap())
                    }
                },
                |_commit, _blob| Ok(()),
                |commit, _source| ("invalid-data", commit.id()),
            )
            .expect_err("the lower dependency must fail");

            let stage = if corrupt_data {
                "invalid-data"
            } else {
                "metadata"
            };
            assert_eq!(observed, (stage, lower.id()));
            assert_eq!(calls.get(), 3);
        }
    }

    #[test]
    fn invalid_foreign_commit_is_inert_in_open_mode() {
        let admitted_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let mut collection = open_collection(&test_name(), admitted_key);
        let descriptor = collection.descriptor().clone();
        let expected = fragment(2, false);
        let admitted_commit = collection.commit(expected.clone()).unwrap();
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();

        let foreign = CollectionCommit::sign(
            &foreign_key,
            identity_for_tests(&descriptor),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata.get_handle(),
        );
        collection.storage_mut().blobs.insert(data);
        collection.storage_mut().blobs.insert(metadata);
        collection
            .storage_mut()
            .insert(CollectionRecord::Commit(invalid_signature(foreign)))
            .unwrap();

        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), expected.facts());
        assert_eq!(snapshot.commits(), &[admitted_commit]);
    }

    #[test]
    fn admitted_commit_without_its_descriptor_blob_fails_loud() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let descriptor =
            simplearchive_union::descriptor(&name, test_namespace(), None, reach::private());
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &signing_key,
            identity_for_tests(&descriptor),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata.get_handle(),
        );
        let mut storage = MemoryRepo::default();
        storage.blobs.insert(data);
        storage.blobs.insert(metadata);
        storage.insert(CollectionRecord::Commit(commit)).unwrap();
        let mut collection = Collection::new(
            storage,
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::DescriptorGet { collection, .. })
                if collection == identity_for_tests(&descriptor)
        ));
    }

    #[test]
    fn admitted_descriptor_bytes_must_match_the_collection_handle() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let descriptor =
            simplearchive_union::descriptor(&name, test_namespace(), None, reach::private());
        let descriptor_handle = identity_for_tests(&descriptor);
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &signing_key,
            descriptor_handle,
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata.get_handle(),
        );
        let wrong_descriptor = IntoBlob::<SimpleArchive>::to_blob(
            simplearchive_union::descriptor(
                &other_name(),
                test_namespace(),
                None,
                reach::private(),
            )
            .into_facts(),
        );
        let actual = wrong_descriptor.get_handle();
        let mut storage = MemoryRepo::default();
        storage.blobs.insert(data);
        storage.blobs.insert(metadata);
        storage
            .blobs
            .insert(Blob::with_handle(wrong_descriptor.bytes, descriptor_handle));
        storage.insert(CollectionRecord::Commit(commit)).unwrap();
        let mut collection = Collection::new(
            storage,
            &name,
            test_namespace(),
            signing_key,
            reach::private(),
            CollectionAdmission::Open,
        );

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::DescriptorIdentity {
                expected,
                actual: observed,
            }) if expected == descriptor_handle && observed == actual
        ));
    }

    #[test]
    fn invalid_signature_claiming_an_admitted_key_is_inert() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut collection = open_collection(&name, signing_key);
        let valid = collection.commit(expected.clone()).unwrap();
        let invalid = invalid_signature(valid);
        collection
            .storage_mut()
            .insert(CollectionRecord::Commit(invalid))
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn missing_or_corrupt_admitted_data_fails_loud() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut missing = open_collection(&name, signing_key.clone());
        let missing_commit = missing.commit(fragment_with_metadata(1, 8)).unwrap();
        let missing_descriptor = missing.collection();
        keep_only(
            missing.storage_mut(),
            [
                missing_descriptor.transmute::<Handle<UnknownBlob>>(),
                missing_commit.metadata().transmute::<Handle<UnknownBlob>>(),
            ],
        );
        assert!(matches!(
            missing.materialize(),
            Err(CollectionMaterializationError::CommitDataGet { commit, .. })
                if commit == missing_commit.id()
        ));

        let mut corrupt = open_collection(&name, signing_key);
        let corrupt_commit = corrupt.commit(fragment_with_metadata(1, 8)).unwrap();
        let corrupt_descriptor = corrupt.collection();
        keep_only(
            corrupt.storage_mut(),
            [
                corrupt_descriptor.transmute::<Handle<UnknownBlob>>(),
                corrupt_commit.metadata().transmute::<Handle<UnknownBlob>>(),
            ],
        );
        let claimed = Handle::<SimpleArchive>::from_hash(corrupt_commit.data());
        corrupt
            .storage_mut()
            .blobs
            .insert(Blob::with_handle(Bytes::from(vec![0; 63]), claimed));
        assert!(matches!(
            corrupt.materialize(),
            Err(CollectionMaterializationError::InvalidCommitData { commit, .. })
                if commit == corrupt_commit.id()
        ));
    }

    #[test]
    fn missing_or_corrupt_admitted_metadata_fails_loud() {
        let name = test_name();
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut missing = open_collection(&name, signing_key.clone());
        let missing_commit = missing.commit(fragment_with_metadata(1, 8)).unwrap();
        let missing_descriptor = missing.collection();
        keep_only(
            missing.storage_mut(),
            [
                missing_descriptor.transmute::<Handle<UnknownBlob>>(),
                Handle::<SimpleArchive>::from_hash(missing_commit.data())
                    .transmute::<Handle<UnknownBlob>>(),
            ],
        );
        assert!(matches!(
            missing.materialize(),
            Err(CollectionMaterializationError::CommitMetadataGet { commit, .. })
                if commit == missing_commit.id()
        ));

        let mut corrupt = open_collection(&name, signing_key);
        let corrupt_commit = corrupt.commit(fragment_with_metadata(1, 8)).unwrap();
        let corrupt_descriptor = corrupt.collection();
        keep_only(
            corrupt.storage_mut(),
            [
                corrupt_descriptor.transmute::<Handle<UnknownBlob>>(),
                Handle::<SimpleArchive>::from_hash(corrupt_commit.data())
                    .transmute::<Handle<UnknownBlob>>(),
            ],
        );
        corrupt.storage_mut().blobs.insert(Blob::with_handle(
            Bytes::from(vec![0; 63]),
            corrupt_commit.metadata(),
        ));
        assert!(matches!(
            corrupt.materialize(),
            Err(CollectionMaterializationError::InvalidCommitMetadata { commit, .. })
                if commit == corrupt_commit.id()
        ));
    }

    #[test]
    fn canonical_admitted_metadata_must_match_the_signed_handle() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let commit = collection.commit(fragment_with_metadata(1, 8)).unwrap();
        let wrong_metadata = archive(9);
        let descriptor = collection.collection();
        keep_only(
            collection.storage_mut(),
            [
                descriptor.transmute::<Handle<UnknownBlob>>(),
                Handle::<SimpleArchive>::from_hash(commit.data())
                    .transmute::<Handle<UnknownBlob>>(),
            ],
        );
        collection
            .storage_mut()
            .blobs
            .insert(Blob::with_handle(wrong_metadata.bytes, commit.metadata()));

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::InvalidCommitMetadataIdentity {
                commit: observed,
                expected,
                ..
            }) if observed == commit.id() && expected == commit.metadata()
        ));
    }

    #[test]
    fn valid_merge_cover_materializes_the_committed_union() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let left = left_fragment.facts().clone().to_blob();
        let right = right_fragment.facts().clone().to_blob();
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        collection.commit(left_fragment).unwrap();
        collection.commit(right_fragment).unwrap();
        let descriptor = collection.descriptor().clone();

        simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(left.get_handle()),
            Handle::<SimpleArchive>::to_hash(right.get_handle()),
        )
        .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn resident_top_cover_can_use_a_nonresident_intermediate() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let first = fragment(1, false);
        let second = fragment(2, false);
        let third = fragment(3, false);
        let first_blob = first.facts().clone().to_blob();
        let second_blob = second.facts().clone().to_blob();
        let third_blob = third.facts().clone().to_blob();
        let mut expected = first.facts().clone();
        expected += second.facts().clone();
        expected += third.facts().clone();
        let first_commit = collection.commit(first).unwrap();
        let second_commit = collection.commit(second).unwrap();
        let third_commit = collection.commit(third).unwrap();
        let descriptor = collection.descriptor().clone();

        let (_, first_two) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(first_blob.get_handle()),
            Handle::<SimpleArchive>::to_hash(second_blob.get_handle()),
        )
        .unwrap();
        let (_, top) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(first_two.get_handle()),
            Handle::<SimpleArchive>::to_hash(third_blob.get_handle()),
        )
        .unwrap();
        keep_only(
            collection.storage_mut(),
            [
                identity_for_tests(&descriptor).transmute(),
                Handle::<SimpleArchive>::from_hash(first_commit.data()).transmute(),
                first_commit.metadata().transmute(),
                Handle::<SimpleArchive>::from_hash(second_commit.data()).transmute(),
                second_commit.metadata().transmute(),
                Handle::<SimpleArchive>::from_hash(third_commit.data()).transmute(),
                third_commit.metadata().transmute(),
                top.get_handle().transmute(),
            ],
        );

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn shared_nonresident_intermediate_lives_through_all_consumers() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let fragments = [
            fragment(1, false),
            fragment(2, false),
            fragment(3, false),
            fragment(4, false),
        ];
        let blobs: Vec<_> = fragments
            .iter()
            .map(|fragment| fragment.facts().clone().to_blob())
            .collect();
        let mut expected = TribleSet::new();
        let mut commits = Vec::new();
        for fragment in fragments {
            expected += fragment.facts().clone();
            commits.push(collection.commit(fragment).unwrap());
        }
        let descriptor = collection.descriptor().clone();

        // X is shared by two children. Reference-counted scratch must retain
        // it until both Y and Z have consumed it, even though none of X/Y/Z is
        // physically resident when the final top artifact is materialized.
        let (_, x) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(blobs[0].get_handle()),
            Handle::<SimpleArchive>::to_hash(blobs[1].get_handle()),
        )
        .unwrap();
        let (_, y) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(x.get_handle()),
            Handle::<SimpleArchive>::to_hash(blobs[2].get_handle()),
        )
        .unwrap();
        let (_, z) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(x.get_handle()),
            Handle::<SimpleArchive>::to_hash(blobs[3].get_handle()),
        )
        .unwrap();
        let (_, top) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(y.get_handle()),
            Handle::<SimpleArchive>::to_hash(z.get_handle()),
        )
        .unwrap();

        let mut keep = Vec::new();
        keep.push(identity_for_tests(&descriptor).transmute());
        for commit in commits {
            keep.push(Handle::<SimpleArchive>::from_hash(commit.data()).transmute());
            keep.push(commit.metadata().transmute());
        }
        keep.push(top.get_handle().transmute());
        keep_only(collection.storage_mut(), keep);

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn corrupt_optional_merge_result_falls_back_to_committed_leaves() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let left_blob = left_fragment.facts().clone().to_blob();
        let right_blob = right_fragment.facts().clone().to_blob();
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        let left = collection.commit(left_fragment).unwrap();
        let right = collection.commit(right_fragment).unwrap();
        let descriptor = collection.descriptor().clone();
        let (_, merged) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            Handle::<SimpleArchive>::to_hash(left_blob.get_handle()),
            Handle::<SimpleArchive>::to_hash(right_blob.get_handle()),
        )
        .unwrap();
        let merged_handle = merged.get_handle();

        keep_only(
            collection.storage_mut(),
            [
                identity_for_tests(&descriptor).transmute(),
                Handle::<SimpleArchive>::from_hash(left.data()).transmute(),
                left.metadata().transmute(),
                Handle::<SimpleArchive>::from_hash(right.data()).transmute(),
                right.metadata().transmute(),
            ],
        );
        let wrong = archive(9);
        collection
            .storage_mut()
            .blobs
            .insert(Blob::with_handle(wrong.bytes, merged_handle));

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn broken_unsigned_merge_falls_back_to_committed_leaves() {
        let mut collection = open_collection(&test_name(), SigningKey::from_bytes(&[7; 32]));
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        let left = collection.commit(left_fragment).unwrap();
        let right = collection.commit(right_fragment).unwrap();
        let broken = CollectionMerge::new(
            collection.collection(),
            left.data(),
            right.data(),
            left.data(),
        );
        collection
            .storage_mut()
            .insert(CollectionRecord::Merge(broken))
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected);
    }
}
