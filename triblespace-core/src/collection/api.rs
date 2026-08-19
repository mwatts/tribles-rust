//! Narrow owned facade for one scoped collection.
//!
//! [`Collection`] owns the storage, canonical `SimpleArchive`-union
//! descriptor, and signing key needed to publish [`Fragment`] values and read
//! the complete known union authorized by that same key. It is not a
//! repository abstraction: it has no head, branch, CAS, retry, read-admission,
//! or planning policy.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::SigningKey;

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::Blob;
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::repo::{
    BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut, StorageClose, StorageFlush,
};
use crate::trible::{Fragment, TribleSet};

use super::simplearchive_union::{
    self, MaterializationError, PublicationError, SimpleArchiveUnionValidationError,
};
use super::{
    collection_physical_cover, discover_collection_records_scoped, resolve_collection_semantics,
    CollectionClaimValidation, CollectionCommit, CollectionData, CollectionDescriptor,
    CollectionDiscoveryError, CollectionFunctionalConflict, CollectionHandle,
    CollectionResolutionError, CollectionStore, CollectionValidationRequest,
    DiscoveredCollectionRecords, ExactTicketError, RecordDecodeError,
};

/// A scoped `SimpleArchive`-union collection and its signing authority.
///
/// Construction is pure with respect to `storage`: the canonical descriptor
/// is derived in memory and is not inserted until a [`commit`](Self::commit)
/// publication begins.
pub struct Collection<S> {
    storage: S,
    descriptor: CollectionDescriptor,
    signing_key: SigningKey,
}

/// One coherent known-prefix view of an owned collection.
///
/// [`commits`](Self::commits) is the exact set of commits from the single
/// collection-record discovery pass that authorized [`facts`](Self::facts).
/// [`reader`](Self::reader) is the blob-reader snapshot used to validate and
/// materialize those facts. The reader may contain physically available blobs
/// published after record discovery, but those blobs acquire no authority
/// unless their commits are present in this snapshot.
pub struct CollectionSnapshot<R> {
    facts: TribleSet,
    commits: Vec<CollectionCommit>,
    reader: R,
}

impl<R> CollectionSnapshot<R> {
    /// Materialized union authorized by this snapshot's exact commit set.
    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    /// Exact authorized commits, ordered by intrinsic record id.
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

/// Failure to materialize the complete authorized value of a collection.
///
/// Every authorized strictly verified commit is ground truth, so its
/// descriptor, data, and metadata fail loud. Unsigned equations are only
/// replaceable cache evidence: missing or invalid equations are omitted from
/// the resolved semantics and cannot hide a valid committed leaf.
#[derive(Debug)]
pub enum CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError> {
    /// Native collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
    /// A supplied exact ticket was not an exact resident authority set.
    ExactTicket(ExactTicketError),
    /// An authorized commit's canonical descriptor blob could not be fetched.
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
    /// by this owned facade.
    DescriptorMismatch {
        /// Descriptor handle named by this facade and its commits.
        collection: CollectionHandle,
    },
    /// The blob reader could not be created after record discovery.
    Reader(ReaderError),
    /// An authorized commit's data blob could not be fetched.
    CommitDataGet {
        /// Intrinsic commit record id.
        commit: Id,
        /// Claimed data identity.
        data: CollectionData,
        /// Backend fetch failure.
        source: GetError,
    },
    /// An authorized commit's data failed exact `SimpleArchive` collection
    /// validation.
    InvalidCommitData {
        /// Intrinsic commit record id.
        commit: Id,
        /// Exact representation or identity diagnostic.
        source: SimpleArchiveUnionValidationError,
    },
    /// An authorized commit's mandatory metadata archive could not be fetched.
    CommitMetadataGet {
        /// Intrinsic commit record id.
        commit: Id,
        /// Mandatory metadata archive handle.
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Backend fetch failure.
        source: GetError,
    },
    /// An authorized commit's mandatory metadata was not a canonical
    /// `SimpleArchive`.
    InvalidCommitMetadata {
        /// Intrinsic commit record id.
        commit: Id,
        /// Mandatory metadata archive handle.
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Canonical archive failure.
        source: UnarchiveError,
    },
    /// An authorized commit's canonical metadata bytes did not have the exact
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
            Self::Discovery(source) => source.fmt(f),
            Self::ExactTicket(source) => write!(f, "invalid exact ticket: {source}"),
            Self::DescriptorGet { collection, source } => write!(
                f,
                "failed to fetch authorized collection descriptor {}: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::InvalidDescriptor { collection, source } => write!(
                f,
                "authorized collection descriptor {} is invalid: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::DescriptorIdentity { expected, actual } => write!(
                f,
                "authorized collection descriptor bytes hash to {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::DescriptorMismatch { collection } => write!(
                f,
                "authorized collection descriptor {} does not match the facade descriptor",
                hex::encode_upper(collection.raw),
            ),
            Self::Reader(source) => write!(f, "failed to open collection blob view: {source}"),
            Self::CommitDataGet {
                commit,
                data,
                source,
            } => write!(
                f,
                "failed to fetch data {} for authorized commit {commit:X}: {source}",
                hex::encode_upper(data.raw),
            ),
            Self::InvalidCommitData { commit, source } => {
                write!(f, "authorized commit {commit:X} has invalid data: {source}")
            }
            Self::CommitMetadataGet {
                commit,
                metadata,
                source,
            } => write!(
                f,
                "failed to fetch metadata {} for authorized commit {commit:X}: {source}",
                hex::encode_upper(metadata.raw),
            ),
            Self::InvalidCommitMetadata {
                commit,
                metadata,
                source,
            } => write!(
                f,
                "authorized commit {commit:X} has invalid metadata {}: {source}",
                hex::encode_upper(metadata.raw),
            ),
            Self::InvalidCommitMetadataIdentity {
                commit,
                expected,
                actual,
            } => write!(
                f,
                "authorized commit {commit:X} metadata bytes hash to {} instead of signed {}",
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
    descriptor: &CollectionDescriptor,
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
    /// Construct a write facade without reading from or writing to `storage`.
    pub fn new(storage: S, scope: Id, signing_key: SigningKey) -> Self {
        Self {
            storage,
            descriptor: simplearchive_union::descriptor(scope),
            signing_key,
        }
    }

    /// Canonical collection descriptor derived from the constructor scope.
    pub fn descriptor(&self) -> &CollectionDescriptor {
        &self.descriptor
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
}

impl<S> Collection<S>
where
    S: CollectionStore,
{
    /// Discover the exact strictly verified commits owned by this facade.
    ///
    /// The returned ticket comes from one deterministic native-record view and
    /// is ordered by intrinsic record id. Only commits naming this facade's
    /// exact descriptor and signing key are included; foreign and invalid
    /// commits are excluded. This operation reads no blobs and does not open a
    /// blob-reader snapshot, so callers can freeze source authority before
    /// deciding which representation to materialize.
    ///
    /// Like every [`CollectionStore`] enumeration, this is a known-prefix
    /// observation rather than a global-latest transaction. A concurrent
    /// commit first observed after this call appears in a later ticket.
    pub fn ticket(
        &mut self,
    ) -> Result<Vec<CollectionCommit>, CollectionDiscoveryError<S::RecordsError>> {
        let (_, commits) = self.discover_owned_commits()?;
        Ok(commits)
    }

    fn discover_owned_commits(
        &mut self,
    ) -> Result<
        (DiscoveredCollectionRecords, Vec<CollectionCommit>),
        CollectionDiscoveryError<S::RecordsError>,
    > {
        let collection = self.descriptor.handle();
        let public_key = crate::inline::Inline::new(self.signing_key.verifying_key().to_bytes());
        let discovered =
            discover_collection_records_scoped(&mut self.storage, collection, public_key)?;
        let commits = discovered.commits().to_vec();

        Ok((discovered, commits))
    }
}

impl<S> Collection<S>
where
    S: BlobStorePut + CollectionStore,
{
    /// Publish one self-contained fragment as an independent signed commit.
    ///
    /// Facts are the collection element, metafacts are commit metadata, and
    /// fragment attachments are staged through the same ordered,
    /// content-addressed path as
    /// [`simplearchive_union::publish_fragment_commit`]. Repeating identical
    /// input is idempotent; distinct commits coexist without selecting a head.
    /// The parameter is deliberately `Fragment`, rather than `Into<Fragment>`,
    /// so a bare fact set cannot accidentally publish without its metafacts.
    pub fn commit(
        &mut self,
        fragment: Fragment,
    ) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>> {
        simplearchive_union::publish_fragment_commit(
            &mut self.storage,
            &self.descriptor,
            fragment,
            &self.signing_key,
        )
    }
}

impl<S> Collection<S>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    /// Capture one coherent known-prefix snapshot of this owned collection.
    ///
    /// The call enumerates collection records exactly once, selects the exact
    /// commits signed by this facade's key for this descriptor, and then opens
    /// one blob-reader snapshot. The returned facts are materialized solely
    /// from that commit set. A concurrently published commit first observed
    /// after record discovery therefore cannot influence the facts even when
    /// its content-addressed blobs are already visible to the reader.
    ///
    /// This is not a global-latest transaction: a later call may observe more
    /// commits. It is a coherent authority boundary for consumers that need
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
        let (discovered, commits) = self
            .discover_owned_commits()
            .map_err(CollectionMaterializationError::Discovery)?;
        Self::snapshot_from_observation(&mut self.storage, &self.descriptor, discovered, commits)
    }

    /// Materialize the complete known `TribleSet` authorized by this facade's
    /// signing identity.
    ///
    /// One call first discovers a deterministic observed view of native
    /// collection records, then opens a blob-reader snapshot. Every strictly
    /// signed commit in that record view which names this exact collection and
    /// this facade's public key is mandatory membership; commits from foreign
    /// keys are ignored. All own commit dependencies are exact-validated and
    /// fail loud. Merge validation is restricted to the subgraph between
    /// authenticated leaves and resident result artifacts, while allowing
    /// nonresident intermediate equations. Exact resident results may replace
    /// redundant leaves in the physical cover; corrupt artifacts and missing,
    /// invalid, or irrelevant unsigned equations are cache misses and the
    /// committed leaves remain authoritative.
    /// Derivations are not admitted by this `SimpleArchive`-only facade.
    /// This boundary assumes records have already passed deployment admission:
    /// it prevents unsigned records from acquiring blob authority, but does
    /// not bound CPU spent validating arbitrarily many admitted equations.
    ///
    /// This is a **known-prefix** read, not a global-latest transaction:
    /// [`CollectionStore`] does not promise a coherent snapshot under
    /// concurrent insertion. A commit first observed after this discovery pass
    /// appears on a later call. The returned set is nevertheless complete for
    /// all own commits observed by this pass, or the call returns an error
    /// instead of a partial set. If no own commit is observed, the result is
    /// empty without opening a blob view or fetching the collection descriptor.
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
        let (discovered, commits) = self
            .discover_owned_commits()
            .map_err(CollectionMaterializationError::Discovery)?;
        if commits.is_empty() {
            return Ok(TribleSet::new());
        }
        Self::snapshot_from_observation(&mut self.storage, &self.descriptor, discovered, commits)
            .map(CollectionSnapshot::into_facts)
    }

    /// Materialize one already-discovered exact authority frontier.
    ///
    /// Both signer-owned and caller-ticketed facades use this single validator
    /// so descriptor, mandatory dependency, merge-cover, and reader-snapshot
    /// semantics cannot drift apart.
    pub(crate) fn snapshot_from_observation(
        storage: &mut S,
        descriptor: &CollectionDescriptor,
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
        let collection = descriptor.handle();
        let authorized: BTreeSet<_> = commits.iter().map(CollectionCommit::id).collect();

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

        // The descriptor handle is the collection identity. Once an authorized
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
            CollectionDescriptor::decode(&descriptor_blob).map_err(|source| {
                CollectionMaterializationError::InvalidDescriptor { collection, source }
            })?;
        if decoded_descriptor != *descriptor {
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

        let resolution = resolve_collection_semantics(&discovered, &authorized, |request| {
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
                let union = simplearchive_union::join_many(members.iter().map(|(_, blob)| *blob))
                    .map_err(|(index, source)| {
                    CollectionMaterializationError::Materialize(
                        MaterializationError::InvalidElement {
                            data: members[index].0,
                            source,
                        },
                    )
                })?;
                union
                    .try_from_blob()
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

    use crate::blob::encodings::{longstring::LongString, UnknownBlob};
    use crate::blob::{BlobEncoding, Bytes, IntoBlob};
    use crate::collection::{discover_collection_records, CollectionMerge, CollectionRecord};
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    struct EmptyWithoutReader;

    impl CollectionStore for EmptyWithoutReader {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::iter::Empty<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            Ok(std::iter::empty())
        }

        fn insert(&mut self, _record: CollectionRecord) -> Result<(), Self::InsertError> {
            Ok(())
        }
    }

    impl BlobStorePut for EmptyWithoutReader {
        type PutError = Infallible;

        fn put<S, T>(&mut self, _item: T) -> Result<Inline<Handle<S>>, Self::PutError>
        where
            S: BlobEncoding + 'static,
            T: IntoBlob<S>,
            Handle<S>: InlineEncoding,
        {
            panic!("empty collection must not put a blob")
        }
    }

    impl BlobStore for EmptyWithoutReader {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = Infallible;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            panic!("empty collection must not open a blob reader")
        }
    }

    #[derive(Default)]
    struct TicketStore {
        records: Vec<CollectionRecord>,
        records_calls: usize,
    }

    impl CollectionStore for TicketStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.records_calls += 1;
            Ok(self
                .records
                .iter()
                .copied()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.records.push(record);
            self.records.sort_unstable_by_key(CollectionRecord::id);
            self.records.dedup_by_key(|record| record.id());
            Ok(())
        }
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

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn fragment(entity: u8, attachment: bool) -> Fragment {
        let mut row = [entity; TRIBLE_LEN];
        row[16..32].fill(1);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        let mut fragment = Fragment::from(facts);
        if attachment {
            let _: Inline<Handle<LongString>> = fragment.put("one attachment".to_owned());
        }
        fragment
    }

    fn archive(entity: u8) -> Blob<SimpleArchive> {
        fragment(entity, false).facts().clone().to_blob()
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
    fn ticket_discovers_only_exact_strictly_verified_owned_commits_without_blob_access() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let collection_id = simplearchive_union::descriptor(scope).handle();
        let other_collection = simplearchive_union::descriptor(id(2)).handle();
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

        // TicketStore deliberately implements only CollectionStore. This test
        // cannot compile if ticket discovery acquires a BlobStore dependency.
        let mut store = TicketStore::default();
        for record in [
            CollectionRecord::Commit(second),
            CollectionRecord::Commit(foreign_signer),
            CollectionRecord::Commit(invalid),
            CollectionRecord::Commit(first),
            CollectionRecord::Commit(foreign_collection),
        ] {
            store.insert(record).unwrap();
        }
        let mut collection = Collection::new(store, scope, signing_key);

        let ticket = collection.ticket().unwrap();
        let mut expected = vec![first, second];
        expected.sort_unstable_by_key(CollectionCommit::id);

        assert_eq!(ticket, expected);
        assert_eq!(collection.storage().records_calls, 1);
    }

    #[test]
    fn ticket_for_an_empty_collection_is_empty_without_blob_access() {
        let mut collection = Collection::new(
            TicketStore::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );

        assert!(collection.ticket().unwrap().is_empty());
        assert_eq!(collection.storage().records_calls, 1);
    }

    #[test]
    fn construction_is_pure_and_derives_the_scoped_descriptor() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);

        assert_eq!(
            collection.descriptor(),
            &simplearchive_union::descriptor(scope)
        );
        assert!(collection.storage().blobs.is_empty());
        assert!(collection.storage_mut().records().unwrap().next().is_none());

        collection.close().unwrap();
    }

    #[test]
    fn borrowed_facade_reuses_one_open_storage() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut storage = MemoryRepo::default();

        {
            let mut collection = Collection::new(&mut storage, scope, signing_key.clone());
            collection.commit(expected.clone()).unwrap();
            assert_eq!(collection.materialize().unwrap(), expected.facts().clone());
        }

        let discovered = discover_collection_records(&mut storage).unwrap();
        assert_eq!(discovered.commits().len(), 1);

        let mut collection = Collection::new(&mut storage, scope, signing_key);
        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn distinct_commits_coexist_and_repeated_commits_are_idempotent() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);
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
        let descriptor_handle = descriptor.handle();
        let reader = collection.storage_mut().reader().unwrap();
        let descriptor_blob: Blob<SimpleArchive> = reader.get(descriptor_handle).unwrap();
        assert_eq!(
            Blob::<SimpleArchive>::new(descriptor_blob.bytes.clone()).get_handle(),
            descriptor_handle
        );
        assert_eq!(
            CollectionDescriptor::decode(&descriptor_blob).unwrap(),
            descriptor
        );

        let discovered = discover_collection_records(collection.storage_mut()).unwrap();
        assert_eq!(
            discovered
                .commits()
                .iter()
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
    fn empty_owned_collection_materializes_without_opening_a_blob_reader() {
        let mut collection =
            Collection::new(EmptyWithoutReader, id(1), SigningKey::from_bytes(&[7; 32]));

        assert_eq!(collection.materialize().unwrap(), TribleSet::new());
    }

    #[test]
    fn empty_snapshot_still_carries_a_reader() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );

        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), &TribleSet::new());
        assert!(snapshot.commits().is_empty());
        assert!(snapshot.reader().is_empty());
    }

    #[test]
    fn snapshot_keeps_one_authority_frontier_when_a_commit_appears_before_reader_open() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let first = fragment(1, false);
        let second = fragment(2, false);
        let mut expected_all = first.facts().clone();
        expected_all += second.facts().clone();

        let mut seeded = Collection::new(MemoryRepo::default(), scope, signing_key.clone());
        let first_commit = seeded.commit(first.clone()).unwrap();
        let second_commit = seeded.commit(second).unwrap();
        let storage = AppendAfterFirstDiscovery {
            inner: seeded.into_storage(),
            initially_visible: BTreeSet::from([first_commit.id()]),
            records_calls: 0,
        };
        let mut collection = Collection::new(storage, scope, signing_key);

        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), first.facts());
        assert_eq!(snapshot.commits(), &[first_commit]);
        assert_eq!(collection.storage().records_calls, 1);

        // The later commit's bytes are already in the captured blob reader.
        // They remain semantically inert because its signed record was not in
        // the exact authority frontier returned with this snapshot.
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
    fn own_commits_materialize_completely_and_repeat_deterministically() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
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
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let descriptor = simplearchive_union::descriptor(scope);
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

        let mut collection = Collection::new(storage, scope, signing_key);
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
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
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
    fn owned_snapshot_remains_signer_scoped_after_shared_materializer_refactor() {
        let own_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), id(1), own_key);
        let descriptor = collection.descriptor().clone();
        let expected = fragment(2, false);
        let own_commit = collection.commit(expected.clone()).unwrap();
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();

        let foreign = simplearchive_union::publish_commit(
            collection.storage_mut(),
            &descriptor,
            &data,
            &metadata,
            &foreign_key,
        )
        .unwrap();
        collection
            .storage_mut()
            .insert(CollectionRecord::Commit(invalid_signature(foreign)))
            .unwrap();

        let snapshot = collection.snapshot().unwrap();
        assert_eq!(snapshot.facts(), expected.facts());
        assert_eq!(snapshot.commits(), &[own_commit]);
    }

    #[test]
    fn own_commit_without_its_descriptor_blob_fails_loud() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let descriptor = simplearchive_union::descriptor(scope);
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &signing_key,
            descriptor.handle(),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata.get_handle(),
        );
        let mut storage = MemoryRepo::default();
        storage.blobs.insert(data);
        storage.blobs.insert(metadata);
        storage.insert(CollectionRecord::Commit(commit)).unwrap();
        let mut collection = Collection::new(storage, scope, signing_key);

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::DescriptorGet { collection, .. })
                if collection == descriptor.handle()
        ));
    }

    #[test]
    fn own_descriptor_bytes_must_match_the_collection_handle() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let descriptor = simplearchive_union::descriptor(scope);
        let descriptor_handle = descriptor.handle();
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &signing_key,
            descriptor_handle,
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata.get_handle(),
        );
        let wrong_descriptor =
            CollectionDescriptor::to_blob(&simplearchive_union::descriptor(id(2)));
        let actual = wrong_descriptor.get_handle();
        let mut storage = MemoryRepo::default();
        storage.blobs.insert(data);
        storage.blobs.insert(metadata);
        storage
            .blobs
            .insert(Blob::with_handle(wrong_descriptor.bytes, descriptor_handle));
        storage.insert(CollectionRecord::Commit(commit)).unwrap();
        let mut collection = Collection::new(storage, scope, signing_key);

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::DescriptorIdentity {
                expected,
                actual: observed,
            }) if expected == descriptor_handle && observed == actual
        ));
    }

    #[test]
    fn invalid_signature_claiming_the_owned_key_is_inert() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);
        let valid = collection.commit(expected.clone()).unwrap();
        let invalid = invalid_signature(valid);
        collection
            .storage_mut()
            .insert(CollectionRecord::Commit(invalid))
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn missing_or_corrupt_owned_data_fails_loud() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut missing = Collection::new(MemoryRepo::default(), scope, signing_key.clone());
        let missing_commit = missing.commit(fragment(1, false)).unwrap();
        let missing_descriptor = missing.descriptor().handle();
        missing.storage_mut().blobs.keep([
            missing_descriptor.transmute::<Handle<UnknownBlob>>(),
            missing_commit.metadata().transmute::<Handle<UnknownBlob>>(),
        ]);
        assert!(matches!(
            missing.materialize(),
            Err(CollectionMaterializationError::CommitDataGet { commit, .. })
                if commit == missing_commit.id()
        ));

        let mut corrupt = Collection::new(MemoryRepo::default(), scope, signing_key);
        let corrupt_commit = corrupt.commit(fragment(1, false)).unwrap();
        let corrupt_descriptor = corrupt.descriptor().handle();
        corrupt.storage_mut().blobs.keep([
            corrupt_descriptor.transmute::<Handle<UnknownBlob>>(),
            corrupt_commit.metadata().transmute::<Handle<UnknownBlob>>(),
        ]);
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
    fn missing_or_corrupt_owned_metadata_fails_loud() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut missing = Collection::new(MemoryRepo::default(), scope, signing_key.clone());
        let missing_commit = missing.commit(fragment(1, false)).unwrap();
        let missing_descriptor = missing.descriptor().handle();
        missing.storage_mut().blobs.keep([
            missing_descriptor.transmute::<Handle<UnknownBlob>>(),
            Handle::<SimpleArchive>::from_hash(missing_commit.data())
                .transmute::<Handle<UnknownBlob>>(),
        ]);
        assert!(matches!(
            missing.materialize(),
            Err(CollectionMaterializationError::CommitMetadataGet { commit, .. })
                if commit == missing_commit.id()
        ));

        let mut corrupt = Collection::new(MemoryRepo::default(), scope, signing_key);
        let corrupt_commit = corrupt.commit(fragment(1, false)).unwrap();
        let corrupt_descriptor = corrupt.descriptor().handle();
        corrupt.storage_mut().blobs.keep([
            corrupt_descriptor.transmute::<Handle<UnknownBlob>>(),
            Handle::<SimpleArchive>::from_hash(corrupt_commit.data())
                .transmute::<Handle<UnknownBlob>>(),
        ]);
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
    fn canonical_owned_metadata_must_match_the_signed_handle() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let commit = collection.commit(fragment(1, false)).unwrap();
        let wrong_metadata = archive(9);
        let descriptor = collection.descriptor().handle();
        collection.storage_mut().blobs.keep([
            descriptor.transmute::<Handle<UnknownBlob>>(),
            Handle::<SimpleArchive>::from_hash(commit.data()).transmute::<Handle<UnknownBlob>>(),
        ]);
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
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let left = left_fragment.facts().clone().to_blob();
        let right = right_fragment.facts().clone().to_blob();
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        collection.commit(left_fragment).unwrap();
        collection.commit(right_fragment).unwrap();
        let descriptor = collection.descriptor().clone();

        simplearchive_union::publish_merge(collection.storage_mut(), &descriptor, &left, &right)
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn resident_top_cover_can_use_a_nonresident_intermediate() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
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
            &first_blob,
            &second_blob,
        )
        .unwrap();
        let (_, top) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            &first_two,
            &third_blob,
        )
        .unwrap();
        collection.storage_mut().blobs.keep([
            descriptor.handle().transmute(),
            Handle::<SimpleArchive>::from_hash(first_commit.data()).transmute(),
            first_commit.metadata().transmute(),
            Handle::<SimpleArchive>::from_hash(second_commit.data()).transmute(),
            second_commit.metadata().transmute(),
            Handle::<SimpleArchive>::from_hash(third_commit.data()).transmute(),
            third_commit.metadata().transmute(),
            top.get_handle().transmute(),
        ]);

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn shared_nonresident_intermediate_lives_through_all_consumers() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
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
            &blobs[0],
            &blobs[1],
        )
        .unwrap();
        let (_, y) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            &x,
            &blobs[2],
        )
        .unwrap();
        let (_, z) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &descriptor,
            &x,
            &blobs[3],
        )
        .unwrap();
        let (_, top) =
            simplearchive_union::publish_merge(collection.storage_mut(), &descriptor, &y, &z)
                .unwrap();

        let mut keep = Vec::new();
        keep.push(descriptor.handle().transmute());
        for commit in commits {
            keep.push(Handle::<SimpleArchive>::from_hash(commit.data()).transmute());
            keep.push(commit.metadata().transmute());
        }
        keep.push(top.get_handle().transmute());
        collection.storage_mut().blobs.keep(keep);

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn corrupt_optional_merge_result_falls_back_to_committed_leaves() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
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
            &left_blob,
            &right_blob,
        )
        .unwrap();
        let merged_handle = merged.get_handle();

        collection.storage_mut().blobs.keep([
            descriptor.handle().transmute(),
            Handle::<SimpleArchive>::from_hash(left.data()).transmute(),
            left.metadata().transmute(),
            Handle::<SimpleArchive>::from_hash(right.data()).transmute(),
            right.metadata().transmute(),
        ]);
        let wrong = archive(9);
        collection
            .storage_mut()
            .blobs
            .insert(Blob::with_handle(wrong.bytes, merged_handle));

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn broken_unsigned_merge_falls_back_to_committed_leaves() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        let left = collection.commit(left_fragment).unwrap();
        let right = collection.commit(right_fragment).unwrap();
        let broken = CollectionMerge::new(
            collection.descriptor().handle(),
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
