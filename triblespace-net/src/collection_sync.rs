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

use std::collections::BTreeMap;
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

    let mut normalized = Vec::with_capacity(blobs.len());
    for (expected, bytes) in blobs {
        let blob = Blob::<UnknownBlob>::new(bytes);
        let actual = Inline::<Hash<Blake3>>::new(blob.get_handle().raw);
        if actual != expected {
            return Err(IncomingValidationError::BlobIdentityMismatch { expected, actual });
        }
        normalized.push((expected, blob));
    }

    let descriptor_bytes = required_blob(
        &normalized,
        commit.collection().raw,
        RequiredBlob::Descriptor,
    )?;
    let data_bytes = required_blob(&normalized, commit.data().raw, RequiredBlob::Data)?;
    let metadata_bytes = required_blob(&normalized, commit.metadata().raw, RequiredBlob::Metadata)?;

    let descriptor_blob = Blob::<SimpleArchive>::new(descriptor_bytes.clone());
    let descriptor = CollectionDescriptor::decode(&descriptor_blob)
        .map_err(IncomingValidationError::InvalidDescriptor)?;
    let data = Blob::<SimpleArchive>::new(data_bytes.clone());
    simplearchive_union::validate_commit(&descriptor, &commit, &data)
        .map_err(IncomingValidationError::InvalidCommitData)?;
    let metadata = Blob::<SimpleArchive>::new(metadata_bytes.clone());
    simplearchive_union::validate_element(&metadata)
        .map_err(IncomingValidationError::InvalidMetadata)?;

    Ok(PreparedIncomingCollectionCommit {
        blobs: normalized.into_iter().map(|(_, blob)| blob).collect(),
        descriptor,
        commit,
        grant,
    })
}

fn required_blob(
    blobs: &[(CollectionData, Blob<UnknownBlob>)],
    raw: [u8; 32],
    role: RequiredBlob,
) -> Result<&Bytes, IncomingValidationError> {
    let handle = Inline::<Hash<Blake3>>::new(raw);
    blobs
        .binary_search_by_key(&handle, |(key, _)| *key)
        .ok()
        .map(|index| &blobs[index].1.bytes)
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
}
