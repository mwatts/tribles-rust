//! Destination admission for collection-native replication.
//!
//! The wire layer supplies one signed commit, its matching signed gossip
//! grant, and a deterministic content-addressed bundle. This module verifies
//! that evidence without touching the destination, asks caller-owned policy,
//! and only then publishes it with the commit last:
//!
//! 1. write every supplied blob and the gossip grant;
//! 2. flush those dependencies;
//! 3. insert the `COMMIT` record;
//! 4. flush the record.
//!
//! The bundle must contain the commit's exact descriptor, data, and metadata.
//! The wire walker is responsible for adding a conservative attachment
//! closure. Generic collection admission cannot prove that application-level
//! handles embedded in archives are complete; it only verifies the identity
//! of every blob it is given.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;

use anybytes::Bytes;
use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use triblespace_core::collection::simplearchive_union::{self, SimpleArchiveUnionValidationError};
use triblespace_core::collection::{
    CollectionCommit, CollectionData, CollectionDescriptor, CollectionGossip,
    CollectionGossipStore, CollectionRecord, CollectionStore, CommitVerificationError,
    GossipVerificationError, RecordDecodeError,
};
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::{Blake3, Hash};
use triblespace_core::metadata::MetaDescribe;
use triblespace_core::repo::{BlobStorePut, StorageFlush};

/// A deterministic content-addressed bundle fetched by the wire walker.
///
/// Keys are raw Blake3 identities, deliberately type-erased across descriptor,
/// data, metadata, and application attachments.
pub type IncomingBlobBundle = BTreeMap<CollectionData, Bytes>;

/// Purely verify an incoming SimpleArchive-union commit and fetched bundle.
///
/// Both signatures, the grant/commit author and descriptor correspondence,
/// every supplied content hash, the canonical collection descriptor and data,
/// and canonical metadata are checked. No destination store is touched.
pub fn prepare_incoming_simplearchive_union_commit(
    commit: CollectionCommit,
    grant: CollectionGossip,
    blobs: IncomingBlobBundle,
) -> Result<PreparedIncomingCollectionCommit, IncomingValidationError> {
    let normalized = normalize_blobs(blobs)?;
    let descriptor = validate_simplearchive_union_evidence(
        &commit,
        &grant,
        &normalized,
        &mut ValidationCache::default(),
    )?;

    Ok(PreparedIncomingCollectionCommit {
        blobs: normalized.into_values().collect(),
        descriptor,
        commit,
        grant,
    })
}

fn normalize_blobs(
    blobs: IncomingBlobBundle,
) -> Result<BTreeMap<CollectionData, Blob<UnknownBlob>>, IncomingValidationError> {
    let mut normalized = BTreeMap::new();
    for (expected, bytes) in blobs {
        let blob = Blob::<UnknownBlob>::new(bytes);
        let actual = Inline::<Hash<Blake3>>::new(blob.get_handle().raw);
        if actual != expected {
            return Err(IncomingValidationError::BlobIdentityMismatch { expected, actual });
        }
        normalized.insert(expected, blob);
    }
    Ok(normalized)
}

fn validate_simplearchive_union_evidence(
    commit: &CollectionCommit,
    grant: &CollectionGossip,
    blobs: &BTreeMap<CollectionData, Blob<UnknownBlob>>,
    cache: &mut ValidationCache,
) -> Result<CollectionDescriptor, IncomingValidationError> {
    commit
        .verify_strict()
        .map_err(IncomingValidationError::InvalidCommitSignature)?;
    grant
        .verify_strict()
        .map_err(IncomingValidationError::InvalidGossipSignature)?;

    if grant.collection() != commit.collection() {
        return Err(IncomingValidationError::GossipCollectionMismatch);
    }
    if grant.public_key() != commit.public_key() {
        return Err(IncomingValidationError::GossipAuthorMismatch);
    }

    let descriptor_handle = Inline::<Hash<Blake3>>::new(commit.collection().raw);
    let descriptor = if let Some(descriptor) = cache.descriptors.get(&descriptor_handle) {
        *descriptor
    } else {
        let descriptor_blob =
            required_blob(blobs, commit.collection().raw, RequiredBlob::Descriptor)?
                .as_transmute::<SimpleArchive>();
        let descriptor = CollectionDescriptor::decode(descriptor_blob)
            .map_err(IncomingValidationError::InvalidDescriptor)?;
        validate_simplearchive_union_descriptor(&descriptor)?;
        cache.descriptors.insert(descriptor_handle, descriptor);
        descriptor
    };

    let data = required_blob(blobs, commit.data().raw, RequiredBlob::Data)?;
    validate_canonical_element_once(&mut cache.elements, commit.data(), data, RequiredBlob::Data)?;
    let metadata_handle = Inline::<Hash<Blake3>>::new(commit.metadata().raw);
    let metadata = required_blob(blobs, commit.metadata().raw, RequiredBlob::Metadata)?;
    validate_canonical_element_once(
        &mut cache.elements,
        metadata_handle,
        metadata,
        RequiredBlob::Metadata,
    )?;

    Ok(descriptor)
}

#[derive(Default)]
struct ValidationCache {
    descriptors: BTreeMap<CollectionData, CollectionDescriptor>,
    elements: BTreeSet<CollectionData>,
}

fn validate_simplearchive_union_descriptor(
    descriptor: &CollectionDescriptor,
) -> Result<(), IncomingValidationError> {
    let expected_representation = <SimpleArchive as MetaDescribe>::id();
    if descriptor.representation() != expected_representation {
        return Err(IncomingValidationError::InvalidCommitData(
            SimpleArchiveUnionValidationError::WrongRepresentation {
                expected: expected_representation,
                actual: descriptor.representation(),
            },
        ));
    }
    if descriptor.recipe() != simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1 {
        return Err(IncomingValidationError::InvalidCommitData(
            SimpleArchiveUnionValidationError::WrongRecipe {
                expected: simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
                actual: descriptor.recipe(),
            },
        ));
    }
    Ok(())
}

fn validate_canonical_element_once(
    validated: &mut BTreeSet<CollectionData>,
    handle: CollectionData,
    blob: &Blob<UnknownBlob>,
    role: RequiredBlob,
) -> Result<(), IncomingValidationError> {
    if validated.contains(&handle) {
        return Ok(());
    }
    let result = simplearchive_union::validate_element(blob.as_transmute::<SimpleArchive>());
    match role {
        RequiredBlob::Data => result.map_err(|source| {
            IncomingValidationError::InvalidCommitData(
                SimpleArchiveUnionValidationError::InvalidElement {
                    role: simplearchive_union::ElementRole::CommitData,
                    source,
                },
            )
        })?,
        RequiredBlob::Metadata => result.map_err(IncomingValidationError::InvalidMetadata)?,
        RequiredBlob::Descriptor => unreachable!("descriptors have their own canonical decoder"),
    }
    validated.insert(handle);
    Ok(())
}

fn required_blob(
    blobs: &BTreeMap<CollectionData, Blob<UnknownBlob>>,
    raw: [u8; 32],
    role: RequiredBlob,
) -> Result<&Blob<UnknownBlob>, IncomingValidationError> {
    let handle = Inline::<Hash<Blake3>>::new(raw);
    blobs
        .get(&handle)
        .ok_or(IncomingValidationError::MissingRequiredBlob { role, handle })
}

/// A required dependency named directly by the signed commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequiredBlob {
    /// Canonical self-describing collection descriptor.
    Descriptor,
    /// Canonical collection element asserted by the commit.
    Data,
    /// Canonical commit metadata archive.
    Metadata,
}

/// A fully verified incoming commit and deterministic blob bundle.
///
/// Preparation is pure. [`admit`](Self::admit) consults caller policy before
/// its first destination mutation.
#[derive(Clone, Debug)]
#[must_use = "a prepared incoming commit has no effect until admitted"]
pub struct PreparedIncomingCollectionCommit {
    blobs: Vec<Blob<UnknownBlob>>,
    descriptor: CollectionDescriptor,
    commit: CollectionCommit,
    grant: CollectionGossip,
}

impl PreparedIncomingCollectionCommit {
    /// Exact strictly verified commit awaiting admission.
    pub fn commit(&self) -> &CollectionCommit {
        &self.commit
    }

    /// Canonical descriptor decoded from the fetched descriptor bytes.
    pub fn descriptor(&self) -> &CollectionDescriptor {
        &self.descriptor
    }

    /// Matching strictly verified redistribution grant.
    pub fn grant(&self) -> &CollectionGossip {
        &self.grant
    }

    /// Ask caller policy, then durably publish dependencies and commit.
    ///
    /// A denied decision or authorization error leaves the destination
    /// completely untouched. On acceptance, blobs are inserted in ascending
    /// handle order, followed by the grant and a dependency flush. The commit
    /// record is inserted only after that barrier and receives its own flush.
    /// Replaying the same prepared value is idempotent under the grow-only
    /// storage trait contracts.
    pub fn admit<S, AuthorizationError, Authorize>(
        self,
        store: &mut S,
        authorize: Authorize,
    ) -> IncomingAdmissionResult<S, AuthorizationError>
    where
        S: BlobStorePut + CollectionGossipStore + CollectionStore + StorageFlush,
        AuthorizationError: Error + Send + Sync + 'static,
        Authorize: FnOnce(
            &CollectionDescriptor,
            &CollectionCommit,
            &CollectionGossip,
        ) -> Result<bool, AuthorizationError>,
    {
        let Self {
            blobs,
            descriptor,
            commit,
            grant,
        } = self;

        if !authorize(&descriptor, &commit, &grant)
            .map_err(IncomingAdmissionError::Authorization)?
        {
            return Ok(IncomingAdmissionOutcome::Unauthorized {
                record: commit.id(),
            });
        }

        for blob in blobs {
            store
                .put::<UnknownBlob, _>(blob)
                .map_err(IncomingAdmissionError::BlobPut)?;
        }
        store
            .gossip(grant)
            .map_err(IncomingAdmissionError::GossipInsert)?;
        store
            .flush()
            .map_err(IncomingAdmissionError::DependencyFlush)?;
        store
            .insert(CollectionRecord::Commit(commit))
            .map_err(IncomingAdmissionError::CommitInsert)?;
        store.flush().map_err(IncomingAdmissionError::CommitFlush)?;

        Ok(IncomingAdmissionOutcome::Admitted {
            record: commit.id(),
        })
    }
}

/// Purely verify a canonical batch of incoming SimpleArchive-union evidence.
///
/// `evidence` must be in strictly increasing intrinsic commit-id order, which
/// is the ordering emitted by the collection wire protocol. The shared blob
/// bundle is hash-normalized once, then reused to validate every commit's
/// exact descriptor, data, and metadata.
pub fn prepare_incoming_simplearchive_union_batch(
    evidence: Vec<(CollectionCommit, CollectionGossip)>,
    blobs: IncomingBlobBundle,
) -> Result<PreparedIncomingCollectionBatch, IncomingBatchValidationError> {
    for pair in evidence.windows(2) {
        let previous = pair[0].0.id();
        let current = pair[1].0.id();
        if previous >= current {
            return Err(IncomingBatchValidationError::NonCanonicalEvidenceOrder {
                previous,
                current,
            });
        }
    }

    let blobs = normalize_blobs(blobs).map_err(IncomingBatchValidationError::BlobBundle)?;
    let mut entries = Vec::with_capacity(evidence.len());
    let mut cache = ValidationCache::default();
    for (index, (commit, grant)) in evidence.into_iter().enumerate() {
        let descriptor = validate_simplearchive_union_evidence(&commit, &grant, &blobs, &mut cache)
            .map_err(|source| IncomingBatchValidationError::Evidence { index, source })?;
        entries.push(PreparedIncomingCollectionEntry {
            descriptor,
            commit,
            grant,
        });
    }

    Ok(PreparedIncomingCollectionBatch { blobs, entries })
}

/// A fully verified batch whose authorization policy has not run yet.
///
/// This type intentionally has no store-taking method. Call
/// [`authorize`](Self::authorize) first; only its successful result can mutate
/// a destination.
#[derive(Clone, Debug)]
#[must_use = "a prepared incoming batch has no effect until authorized and admitted"]
pub struct PreparedIncomingCollectionBatch {
    blobs: BTreeMap<CollectionData, Blob<UnknownBlob>>,
    entries: Vec<PreparedIncomingCollectionEntry>,
}

#[derive(Clone, Debug)]
struct PreparedIncomingCollectionEntry {
    descriptor: CollectionDescriptor,
    commit: CollectionCommit,
    grant: CollectionGossip,
}

impl PreparedIncomingCollectionBatch {
    /// Number of strictly verified evidence pairs observed in this batch.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the batch contains no evidence pairs.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Decide every evidence pair without exposing a destination store.
    ///
    /// If policy returns an error, no authorized batch is produced and no
    /// storage operation could have occurred. Accepted roots are then reduced
    /// to the union of their conservative content-addressed closures, so
    /// blobs reachable only from denied commits are discarded before the
    /// mutation phase.
    pub fn authorize<AuthorizationError, Authorize>(
        self,
        mut authorize: Authorize,
    ) -> Result<AuthorizedIncomingCollectionBatch, AuthorizationError>
    where
        Authorize: FnMut(
            &CollectionDescriptor,
            &CollectionCommit,
            &CollectionGossip,
        ) -> Result<bool, AuthorizationError>,
    {
        let observed = self.entries.len();
        let mut accepted = Vec::new();
        let mut roots = BTreeSet::new();

        for entry in self.entries {
            if authorize(&entry.descriptor, &entry.commit, &entry.grant)? {
                roots.extend([
                    Inline::<Hash<Blake3>>::new(entry.commit.collection().raw),
                    entry.commit.data(),
                    Inline::<Hash<Blake3>>::new(entry.commit.metadata().raw),
                ]);
                accepted.push((entry.commit, entry.grant));
            }
        }

        let closure = conservative_blob_closure(&self.blobs, roots);
        let blobs = self
            .blobs
            .into_iter()
            .filter_map(|(handle, blob)| closure.contains(&handle).then_some(blob))
            .collect();
        let admitted = accepted.len();

        Ok(AuthorizedIncomingCollectionBatch {
            blobs,
            accepted,
            counts: IncomingBatchCounts {
                observed,
                admitted,
                denied: observed - admitted,
            },
        })
    }
}

fn conservative_blob_closure(
    blobs: &BTreeMap<CollectionData, Blob<UnknownBlob>>,
    mut pending: BTreeSet<CollectionData>,
) -> BTreeSet<CollectionData> {
    let mut closure = BTreeSet::new();
    while let Some(parent) = pending.pop_first() {
        if !closure.insert(parent) {
            continue;
        }
        let Some(blob) = blobs.get(&parent) else {
            // Every root was validated above. Descendants enter `pending`
            // only after a positive map lookup, so this is unreachable for a
            // well-formed prepared batch.
            continue;
        };
        for chunk in blob.bytes.chunks(32) {
            if chunk.len() != 32 {
                continue;
            }
            let mut raw = [0u8; 32];
            raw.copy_from_slice(chunk);
            let child = Inline::<Hash<Blake3>>::new(raw);
            if blobs.contains_key(&child) && !closure.contains(&child) {
                pending.insert(child);
            }
        }
    }
    closure
}

/// A verified batch whose policy decisions are complete and immutable.
#[derive(Clone, Debug)]
#[must_use = "an authorized incoming batch has no effect until admitted"]
pub struct AuthorizedIncomingCollectionBatch {
    blobs: Vec<Blob<UnknownBlob>>,
    accepted: Vec<(CollectionCommit, CollectionGossip)>,
    counts: IncomingBatchCounts,
}

impl AuthorizedIncomingCollectionBatch {
    /// Counts fixed by the completed authorization phase.
    pub fn counts(&self) -> IncomingBatchCounts {
        self.counts
    }

    /// Publish the accepted union with exactly two durability barriers.
    ///
    /// Every selected blob and distinct grant is written before the first
    /// flush. Every distinct commit is inserted after that barrier, followed
    /// by the second flush. An all-denied or empty batch is a no-op and needs
    /// no durability barrier.
    pub fn admit<S>(self, store: &mut S) -> IncomingBatchAdmissionResult<S>
    where
        S: BlobStorePut + CollectionGossipStore + CollectionStore + StorageFlush,
    {
        if self.accepted.is_empty() {
            return Ok(self.counts);
        }

        for blob in self.blobs {
            store
                .put::<UnknownBlob, _>(blob)
                .map_err(IncomingBatchAdmissionError::BlobPut)?;
        }

        let grants: BTreeSet<_> = self.accepted.iter().map(|(_, grant)| *grant).collect();
        for grant in grants {
            store
                .gossip(grant)
                .map_err(IncomingBatchAdmissionError::GossipInsert)?;
        }
        store
            .flush()
            .map_err(IncomingBatchAdmissionError::DependencyFlush)?;

        let commits: BTreeMap<_, _> = self
            .accepted
            .into_iter()
            .map(|(commit, _)| (commit.id(), commit))
            .collect();
        for commit in commits.into_values() {
            store
                .insert(CollectionRecord::Commit(commit))
                .map_err(IncomingBatchAdmissionError::CommitInsert)?;
        }
        store
            .flush()
            .map_err(IncomingBatchAdmissionError::CommitFlush)?;

        Ok(self.counts)
    }
}

/// Aggregate result of one batch reconciliation attempt.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IncomingBatchCounts {
    /// Strictly verified evidence pairs presented to policy.
    pub observed: usize,
    /// Evidence pairs accepted and durably inserted.
    pub admitted: usize,
    /// Evidence pairs declined by policy.
    pub denied: usize,
}

/// Pure batch validation failure before authorization or storage.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum IncomingBatchValidationError {
    /// Evidence was not strictly sorted by unique intrinsic commit id.
    #[error(
        "incoming evidence is not in strictly increasing commit-id order ({previous:?} then {current:?})"
    )]
    NonCanonicalEvidenceOrder { previous: Id, current: Id },
    /// Shared bundle hash normalization failed.
    #[error("invalid incoming blob bundle: {0}")]
    BlobBundle(#[source] IncomingValidationError),
    /// One evidence pair failed strict validation.
    #[error("incoming evidence item {index} is invalid: {source}")]
    Evidence {
        /// Zero-based index in canonical evidence order.
        index: usize,
        /// Exact validation failure.
        #[source]
        source: IncomingValidationError,
    },
}

/// Destination-specific result type for
/// [`AuthorizedIncomingCollectionBatch::admit`].
pub type IncomingBatchAdmissionResult<S> = Result<
    IncomingBatchCounts,
    IncomingBatchAdmissionError<
        <S as BlobStorePut>::PutError,
        <S as CollectionGossipStore>::GossipError,
        <S as CollectionStore>::InsertError,
        <S as StorageFlush>::Error,
    >,
>;

/// Failure while durably publishing an already-authorized batch.
#[derive(Debug, thiserror::Error)]
pub enum IncomingBatchAdmissionError<PutError, GossipError, InsertError, FlushError> {
    /// A selected blob could not be inserted.
    #[error("failed to stage incoming batch blob: {0}")]
    BlobPut(#[source] PutError),
    /// A matching gossip grant could not be inserted.
    #[error("failed to stage incoming batch gossip grant: {0}")]
    GossipInsert(#[source] GossipError),
    /// Dependencies or grants could not be made durable.
    #[error("failed to flush incoming batch dependencies: {0}")]
    DependencyFlush(#[source] FlushError),
    /// A commit record could not be inserted.
    #[error("failed to insert incoming batch commit: {0}")]
    CommitInsert(#[source] InsertError),
    /// Commit records could not be made durable.
    #[error("failed to flush incoming batch commits: {0}")]
    CommitFlush(#[source] FlushError),
}

/// Pure validation failure before caller policy or storage is consulted.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum IncomingValidationError {
    /// The commit signature is invalid.
    #[error("invalid commit signature: {0}")]
    InvalidCommitSignature(CommitVerificationError),
    /// The gossip-grant signature is invalid.
    #[error("invalid gossip signature: {0}")]
    InvalidGossipSignature(GossipVerificationError),
    /// Grant and commit name different collection descriptors.
    #[error("gossip grant and commit name different collections")]
    GossipCollectionMismatch,
    /// Grant and commit were signed by different authors.
    #[error("gossip grant and commit have different authors")]
    GossipAuthorMismatch,
    /// One fetched blob was keyed by a digest other than its bytes.
    #[error("fetched blob was not keyed by its Blake3 identity")]
    BlobIdentityMismatch {
        /// Claimed bundle key.
        expected: CollectionData,
        /// Blake3 identity recomputed from the bytes.
        actual: CollectionData,
    },
    /// The bundle omitted a dependency named directly by the commit.
    #[error("incoming bundle is missing its {role:?} blob")]
    MissingRequiredBlob {
        /// Dependency role.
        role: RequiredBlob,
        /// Missing content identity.
        handle: CollectionData,
    },
    /// Descriptor bytes were not one exact canonical descriptor archive.
    #[error("invalid collection descriptor: {0}")]
    InvalidDescriptor(RecordDecodeError),
    /// Descriptor or data does not implement the asserted collection commit.
    #[error("invalid collection commit data: {0}")]
    InvalidCommitData(SimpleArchiveUnionValidationError),
    /// Commit metadata is not a canonical `SimpleArchive`.
    #[error("invalid commit metadata: {0}")]
    InvalidMetadata(UnarchiveError),
}

/// Result of caller policy and durable destination admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncomingAdmissionOutcome {
    /// Dependencies, grant, and commit are durable.
    Admitted { record: Id },
    /// Policy declined before any destination mutation.
    Unauthorized { record: Id },
}

/// Destination-specific result type for [`PreparedIncomingCollectionCommit::admit`].
pub type IncomingAdmissionResult<S, AuthorizationError> = Result<
    IncomingAdmissionOutcome,
    IncomingAdmissionError<
        AuthorizationError,
        <S as BlobStorePut>::PutError,
        <S as CollectionGossipStore>::GossipError,
        <S as CollectionStore>::InsertError,
        <S as StorageFlush>::Error,
    >,
>;

/// Failure while authorizing or durably publishing verified input.
#[derive(Debug, thiserror::Error)]
pub enum IncomingAdmissionError<AuthorizationError, PutError, GossipError, InsertError, FlushError>
{
    /// Caller-owned policy could not decide.
    #[error("incoming commit authorization failed: {0}")]
    Authorization(#[source] AuthorizationError),
    /// A fetched blob could not be inserted.
    #[error("failed to stage incoming blob: {0}")]
    BlobPut(#[source] PutError),
    /// The matching gossip grant could not be inserted.
    #[error("failed to stage gossip grant: {0}")]
    GossipInsert(#[source] GossipError),
    /// Dependencies or grant could not be made durable.
    #[error("failed to flush incoming dependencies: {0}")]
    DependencyFlush(#[source] FlushError),
    /// The commit record could not be inserted.
    #[error("failed to insert incoming commit: {0}")]
    CommitInsert(#[source] InsertError),
    /// The commit record could not be made durable.
    #[error("failed to flush incoming commit: {0}")]
    CommitFlush(#[source] FlushError),
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::convert::Infallible;
    use std::fmt;

    use triblespace_core::blob::{BlobEncoding, IntoBlob};
    use triblespace_core::collection::simplearchive_union;
    use triblespace_core::inline::InlineEncoding;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::repo::pile::Pile;
    use triblespace_core::repo::{BlobStore, BlobStoreGet};
    use triblespace_core::trible::{TRIBLE_LEN, Trible, TribleSet};

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Event {
        Put(CollectionData),
        Gossip(CollectionGossip),
        Flush,
        Insert(Id),
    }

    #[derive(Default)]
    struct ProbeStore {
        events: Vec<Event>,
        pending_blobs: BTreeMap<CollectionData, Bytes>,
        durable_blobs: BTreeMap<CollectionData, Bytes>,
        pending_gossips: BTreeSet<CollectionGossip>,
        durable_gossips: BTreeSet<CollectionGossip>,
        pending_records: BTreeMap<Id, CollectionRecord>,
        durable_records: BTreeMap<Id, CollectionRecord>,
        insert_saw_durable_dependencies: bool,
    }

    impl BlobStorePut for ProbeStore {
        type PutError = Infallible;

        fn put<S, T>(
            &mut self,
            item: T,
        ) -> Result<triblespace_core::inline::Inline<Handle<S>>, Self::PutError>
        where
            S: BlobEncoding + 'static,
            T: IntoBlob<S>,
            Handle<S>: InlineEncoding,
        {
            let blob: Blob<S> = item.to_blob();
            let handle = blob.get_handle();
            let erased = Inline::<Hash<Blake3>>::new(handle.raw);
            self.events.push(Event::Put(erased));
            self.pending_blobs.entry(erased).or_insert(blob.bytes);
            Ok(handle)
        }
    }

    impl CollectionGossipStore for ProbeStore {
        type GossipsError = Infallible;
        type GossipError = Infallible;
        type GossipIter<'a> = std::vec::IntoIter<Result<CollectionGossip, Infallible>>;

        fn gossips<'a>(&'a mut self) -> Result<Self::GossipIter<'a>, Self::GossipsError> {
            Ok(self
                .durable_gossips
                .iter()
                .copied()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn gossip(&mut self, grant: CollectionGossip) -> Result<(), Self::GossipError> {
            self.events.push(Event::Gossip(grant));
            self.pending_gossips.insert(grant);
            Ok(())
        }
    }

    impl CollectionStore for ProbeStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            Ok(self
                .durable_records
                .values()
                .copied()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            let CollectionRecord::Commit(commit) = record else {
                unreachable!("admission inserts only commits")
            };
            self.insert_saw_durable_dependencies = self.pending_blobs.is_empty()
                && self.pending_gossips.is_empty()
                && self
                    .durable_blobs
                    .contains_key(&Inline::new(commit.collection().raw))
                && self.durable_blobs.contains_key(&commit.data())
                && self
                    .durable_blobs
                    .contains_key(&Inline::new(commit.metadata().raw))
                && self.durable_gossips.iter().any(|grant| {
                    grant.collection() == commit.collection()
                        && grant.public_key() == commit.public_key()
                });
            self.events.push(Event::Insert(commit.id()));
            self.pending_records.entry(commit.id()).or_insert(record);
            Ok(())
        }
    }

    impl StorageFlush for ProbeStore {
        type Error = Infallible;

        fn flush(&mut self) -> Result<(), Self::Error> {
            self.events.push(Event::Flush);
            self.durable_blobs.append(&mut self.pending_blobs);
            self.durable_gossips.append(&mut self.pending_gossips);
            self.durable_records.append(&mut self.pending_records);
            Ok(())
        }
    }

    #[derive(Clone)]
    struct Fixture {
        descriptor: CollectionDescriptor,
        commit: CollectionCommit,
        grant: CollectionGossip,
        blobs: IncomingBlobBundle,
    }

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn archive(byte: u8) -> Blob<SimpleArchive> {
        let mut row = [byte; TRIBLE_LEN];
        row[16..32].fill(byte.wrapping_add(1));
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        facts.to_blob()
    }

    fn erased<S>(blob: &Blob<S>) -> CollectionData
    where
        S: BlobEncoding,
        Handle<S>: InlineEncoding,
    {
        Inline::new(blob.get_handle().raw)
    }

    fn fixture() -> Fixture {
        let author = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
        let descriptor = simplearchive_union::descriptor(id(1));
        let descriptor_blob = CollectionDescriptor::to_blob(&descriptor);
        let data = archive(2);
        let metadata = archive(3);
        let attachment = Blob::<UnknownBlob>::new(Bytes::from_source(vec![4, 5, 6, 7]));
        let commit = CollectionCommit::sign(
            &author,
            descriptor.handle(),
            erased(&data),
            metadata.get_handle(),
        );
        let grant = CollectionGossip::sign(&author, descriptor.handle());
        let blobs = [
            (erased(&descriptor_blob), descriptor_blob.bytes),
            (erased(&data), data.bytes),
            (erased(&metadata), metadata.bytes),
            (erased(&attachment), attachment.bytes),
        ]
        .into_iter()
        .collect();
        Fixture {
            descriptor,
            commit,
            grant,
            blobs,
        }
    }

    fn archive_referencing(byte: u8, child: CollectionData) -> Blob<SimpleArchive> {
        let mut row = [0; TRIBLE_LEN];
        row[..16].fill(byte);
        row[16..32].fill(byte.wrapping_add(1));
        row[32..].copy_from_slice(&child.raw);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        facts.to_blob()
    }

    struct BatchFixture {
        evidence: Vec<(CollectionCommit, CollectionGossip)>,
        blobs: IncomingBlobBundle,
        first_data: CollectionData,
        first_attachment: CollectionData,
        second_data: CollectionData,
        second_attachment: CollectionData,
        descriptor: CollectionData,
        metadata: CollectionData,
    }

    fn batch_fixture() -> BatchFixture {
        let author = ed25519_dalek::SigningKey::from_bytes(&[23; 32]);
        let descriptor_value = simplearchive_union::descriptor(id(31));
        let descriptor_blob = CollectionDescriptor::to_blob(&descriptor_value);
        let metadata_blob = archive(32);
        let first_attachment_blob = Blob::<UnknownBlob>::new(Bytes::from_source(vec![33, 34, 35]));
        let second_attachment_blob = Blob::<UnknownBlob>::new(Bytes::from_source(vec![36, 37, 38]));
        let first_attachment = erased(&first_attachment_blob);
        let second_attachment = erased(&second_attachment_blob);
        let first_data_blob = archive_referencing(39, first_attachment);
        let second_data_blob = archive_referencing(40, second_attachment);
        let first_data = erased(&first_data_blob);
        let second_data = erased(&second_data_blob);
        let first_commit = CollectionCommit::sign(
            &author,
            descriptor_value.handle(),
            first_data,
            metadata_blob.get_handle(),
        );
        let second_commit = CollectionCommit::sign(
            &author,
            descriptor_value.handle(),
            second_data,
            metadata_blob.get_handle(),
        );
        let grant = CollectionGossip::sign(&author, descriptor_value.handle());
        let mut evidence = vec![(first_commit, grant), (second_commit, grant)];
        evidence.sort_by_key(|(commit, _)| commit.id());

        let descriptor = erased(&descriptor_blob);
        let metadata = erased(&metadata_blob);
        let blobs = [
            (descriptor, descriptor_blob.bytes),
            (metadata, metadata_blob.bytes),
            (first_attachment, first_attachment_blob.bytes),
            (second_attachment, second_attachment_blob.bytes),
            (first_data, first_data_blob.bytes),
            (second_data, second_data_blob.bytes),
        ]
        .into_iter()
        .collect();

        BatchFixture {
            evidence,
            blobs,
            first_data,
            first_attachment,
            second_data,
            second_attachment,
            descriptor,
            metadata,
        }
    }

    #[test]
    fn admission_flushes_all_blobs_and_grant_before_commit_visibility() {
        let fixture = fixture();
        let expected_handles = fixture.blobs.keys().copied().collect::<Vec<_>>();
        let prepared = prepare_incoming_simplearchive_union_commit(
            fixture.commit,
            fixture.grant,
            fixture.blobs,
        )
        .unwrap();
        assert_eq!(prepared.descriptor(), &fixture.descriptor);
        assert_eq!(prepared.commit(), &fixture.commit);
        assert_eq!(prepared.grant(), &fixture.grant);

        let mut store = ProbeStore::default();
        let outcome = prepared
            .admit(&mut store, |descriptor, commit, grant| {
                assert_eq!(descriptor, &fixture.descriptor);
                assert_eq!(commit, &fixture.commit);
                assert_eq!(grant, &fixture.grant);
                Ok::<_, Infallible>(true)
            })
            .unwrap();

        assert_eq!(
            outcome,
            IncomingAdmissionOutcome::Admitted {
                record: fixture.commit.id()
            }
        );
        assert_eq!(
            store.events,
            expected_handles
                .iter()
                .copied()
                .map(Event::Put)
                .chain([
                    Event::Gossip(fixture.grant),
                    Event::Flush,
                    Event::Insert(fixture.commit.id()),
                    Event::Flush,
                ])
                .collect::<Vec<_>>()
        );
        assert!(store.insert_saw_durable_dependencies);
        assert_eq!(store.durable_blobs.len(), expected_handles.len());
        assert_eq!(store.durable_gossips.len(), 1);
        assert_eq!(store.durable_records.len(), 1);
    }

    #[test]
    fn denial_and_authorization_error_do_not_mutate_destination() {
        let fixture = fixture();
        let prepared = prepare_incoming_simplearchive_union_commit(
            fixture.commit,
            fixture.grant,
            fixture.blobs.clone(),
        )
        .unwrap();
        let mut denied = ProbeStore::default();
        assert_eq!(
            prepared
                .clone()
                .admit(&mut denied, |_, _, _| Ok::<_, Infallible>(false))
                .unwrap(),
            IncomingAdmissionOutcome::Unauthorized {
                record: fixture.commit.id()
            }
        );
        assert!(denied.events.is_empty());

        #[derive(Debug)]
        struct PolicyError;
        impl fmt::Display for PolicyError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "policy failed")
            }
        }
        impl Error for PolicyError {}

        let mut failed = ProbeStore::default();
        assert!(matches!(
            prepared.admit(&mut failed, |_, _, _| Err::<bool, _>(PolicyError)),
            Err(IncomingAdmissionError::Authorization(PolicyError))
        ));
        assert!(failed.events.is_empty());
    }

    #[test]
    fn forged_bundle_hash_is_rejected() {
        let mut fixture = fixture();
        let (actual, bytes) = fixture.blobs.pop_first().unwrap();
        let forged = Inline::new([0xFF; 32]);
        fixture.blobs.insert(forged, bytes);
        assert!(matches!(
            prepare_incoming_simplearchive_union_commit(
                fixture.commit,
                fixture.grant,
                fixture.blobs,
            ),
            Err(IncomingValidationError::BlobIdentityMismatch {
                expected,
                actual: found,
            }) if expected == forged && found == actual
        ));
    }

    #[test]
    fn grant_must_match_both_collection_and_author() {
        let fixture = fixture();
        let author = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
        let other_descriptor = simplearchive_union::descriptor(id(9));
        let wrong_collection = CollectionGossip::sign(&author, other_descriptor.handle());
        assert_eq!(
            prepare_incoming_simplearchive_union_commit(
                fixture.commit,
                wrong_collection,
                fixture.blobs.clone(),
            )
            .unwrap_err(),
            IncomingValidationError::GossipCollectionMismatch
        );

        let other_author = ed25519_dalek::SigningKey::from_bytes(&[19; 32]);
        let wrong_author = CollectionGossip::sign(&other_author, fixture.commit.collection());
        assert_eq!(
            prepare_incoming_simplearchive_union_commit(
                fixture.commit,
                wrong_author,
                fixture.blobs,
            )
            .unwrap_err(),
            IncomingValidationError::GossipAuthorMismatch
        );
    }

    #[test]
    fn required_blobs_and_canonical_metadata_are_enforced() {
        let mut missing = fixture();
        missing
            .blobs
            .remove(&Inline::new(missing.commit.data().raw));
        assert!(matches!(
            prepare_incoming_simplearchive_union_commit(
                missing.commit,
                missing.grant,
                missing.blobs,
            ),
            Err(IncomingValidationError::MissingRequiredBlob {
                role: RequiredBlob::Data,
                ..
            })
        ));

        let author = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
        let mut malformed = fixture();
        let old_metadata = Inline::new(malformed.commit.metadata().raw);
        malformed.blobs.remove(&old_metadata);
        let bad_metadata = Blob::<SimpleArchive>::new(Bytes::from_source(vec![1, 2, 3]));
        malformed.commit = CollectionCommit::sign(
            &author,
            malformed.descriptor.handle(),
            malformed.commit.data(),
            bad_metadata.get_handle(),
        );
        malformed.grant = CollectionGossip::sign(&author, malformed.descriptor.handle());
        malformed
            .blobs
            .insert(erased(&bad_metadata), bad_metadata.bytes);
        assert_eq!(
            prepare_incoming_simplearchive_union_commit(
                malformed.commit,
                malformed.grant,
                malformed.blobs,
            )
            .unwrap_err(),
            IncomingValidationError::InvalidMetadata(UnarchiveError::BadArchive)
        );
    }

    #[test]
    fn retry_is_logically_idempotent() {
        let fixture = fixture();
        let mut store = ProbeStore::default();
        for _ in 0..2 {
            let outcome = prepare_incoming_simplearchive_union_commit(
                fixture.commit,
                fixture.grant,
                fixture.blobs.clone(),
            )
            .unwrap()
            .admit(&mut store, |_, _, _| Ok::<_, Infallible>(true))
            .unwrap();
            assert!(matches!(outcome, IncomingAdmissionOutcome::Admitted { .. }));
        }
        assert_eq!(store.durable_blobs.len(), fixture.blobs.len());
        assert_eq!(store.durable_gossips, BTreeSet::from([fixture.grant]));
        assert_eq!(store.durable_records.len(), 1);
    }

    #[test]
    fn pile_admission_is_additive_and_preserves_existing_blob_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("replica.pile");
        std::fs::File::create(&path).unwrap();
        let legacy = Blob::<UnknownBlob>::new(Bytes::from_source(vec![9, 8, 7, 6, 5]));
        let legacy_handle = legacy.get_handle();

        let mut pile = Pile::open(&path).unwrap();
        pile.put::<UnknownBlob, _>(legacy.clone()).unwrap();
        pile.flush().unwrap();

        let fixture = fixture();
        prepare_incoming_simplearchive_union_commit(fixture.commit, fixture.grant, fixture.blobs)
            .unwrap()
            .admit(&mut pile, |_, _, _| Ok::<_, Infallible>(true))
            .unwrap();
        pile.close().unwrap();

        let mut reopened = Pile::open(&path).unwrap();
        let reader = reopened.reader().unwrap();
        let reread: Blob<UnknownBlob> = reader.get(legacy_handle).unwrap();
        assert_eq!(reread.bytes, legacy.bytes);
        assert!(reopened
            .records()
            .unwrap()
            .any(|record| matches!(record, Ok(CollectionRecord::Commit(commit)) if commit == fixture.commit)));
    }

    #[test]
    fn batch_admission_stores_only_the_accepted_union_with_two_flushes() {
        let fixture = batch_fixture();
        let accepted_record = fixture
            .evidence
            .iter()
            .find(|(commit, _)| commit.data() == fixture.first_data)
            .unwrap()
            .0
            .id();
        let accepted_grant = fixture.evidence[0].1;
        let prepared =
            prepare_incoming_simplearchive_union_batch(fixture.evidence, fixture.blobs).unwrap();
        assert_eq!(prepared.len(), 2);

        let authorized = prepared
            .authorize(|_, commit, _| Ok::<_, Infallible>(commit.data() == fixture.first_data))
            .unwrap();
        assert_eq!(
            authorized.counts(),
            IncomingBatchCounts {
                observed: 2,
                admitted: 1,
                denied: 1,
            }
        );

        let mut store = ProbeStore::default();
        let counts = authorized.admit(&mut store).unwrap();
        assert_eq!(counts.observed, 2);
        assert_eq!(counts.admitted, 1);
        assert_eq!(counts.denied, 1);
        assert_eq!(
            store
                .events
                .iter()
                .filter(|event| **event == Event::Flush)
                .count(),
            2
        );
        assert_eq!(store.durable_gossips, BTreeSet::from([accepted_grant]));
        assert_eq!(store.durable_records.len(), 1);
        assert!(store.durable_records.contains_key(&accepted_record));

        let accepted_handles = BTreeSet::from([
            fixture.descriptor,
            fixture.metadata,
            fixture.first_data,
            fixture.first_attachment,
        ]);
        assert_eq!(
            store.durable_blobs.keys().copied().collect::<BTreeSet<_>>(),
            accepted_handles
        );
        assert!(!store.durable_blobs.contains_key(&fixture.second_data));
        assert!(!store.durable_blobs.contains_key(&fixture.second_attachment));
        assert!(store.insert_saw_durable_dependencies);
    }

    #[test]
    fn batch_deduplicates_shared_dependencies_and_grants() {
        let fixture = batch_fixture();
        let expected_blob_count = fixture.blobs.len();
        let prepared =
            prepare_incoming_simplearchive_union_batch(fixture.evidence, fixture.blobs).unwrap();
        let authorized = prepared
            .authorize(|_, _, _| Ok::<_, Infallible>(true))
            .unwrap();
        let mut store = ProbeStore::default();
        let counts = authorized.admit(&mut store).unwrap();

        assert_eq!(
            counts,
            IncomingBatchCounts {
                observed: 2,
                admitted: 2,
                denied: 0,
            }
        );
        assert_eq!(store.durable_blobs.len(), expected_blob_count);
        assert_eq!(store.durable_gossips.len(), 1);
        assert_eq!(store.durable_records.len(), 2);
        assert_eq!(
            store
                .events
                .iter()
                .filter(|event| **event == Event::Flush)
                .count(),
            2
        );
    }

    #[test]
    fn all_denied_batch_is_a_storage_noop() {
        let fixture = batch_fixture();
        let authorized =
            prepare_incoming_simplearchive_union_batch(fixture.evidence, fixture.blobs)
                .unwrap()
                .authorize(|_, _, _| Ok::<_, Infallible>(false))
                .unwrap();
        let mut store = ProbeStore::default();
        assert_eq!(
            authorized.admit(&mut store).unwrap(),
            IncomingBatchCounts {
                observed: 2,
                admitted: 0,
                denied: 2,
            }
        );
        assert!(store.events.is_empty());
    }

    #[test]
    fn batch_requires_canonical_unique_evidence_order() {
        let mut fixture = batch_fixture();
        fixture.evidence.reverse();
        assert!(matches!(
            prepare_incoming_simplearchive_union_batch(fixture.evidence, fixture.blobs),
            Err(IncomingBatchValidationError::NonCanonicalEvidenceOrder { .. })
        ));
    }
}
