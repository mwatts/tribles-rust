//! Sparse destination admission for collection-native replication.
//!
//! Replication evidence is deliberately only a signed [`CollectionCommit`]
//! and its matching signed [`CollectionGossip`] grant. The content-addressed
//! blobs named by the commit are not an admission dependency: a receiver may
//! learn the commit first and resolve any descriptor, data, metadata, or
//! attachment blobs later through its ordinary want policy.
//!
//! Admission has three explicit phases:
//!
//! 1. [`prepare_incoming_collection_batch`] purely verifies evidence and its
//!    canonical ordering;
//! 2. [`PreparedIncomingCollectionBatch::authorize`] runs caller policy while
//!    no destination store is available;
//! 3. [`AuthorizedIncomingCollectionBatch::admit`] inserts the accepted
//!    grow-only evidence and performs one durability flush.

use std::collections::{BTreeMap, BTreeSet};

use triblespace_core::collection::{
    CollectionCommit, CollectionGossip, CollectionGossipStore, CollectionRecord, CollectionStore,
    CommitVerificationError, GossipVerificationError,
};
use triblespace_core::id::Id;
use triblespace_core::repo::StorageFlush;

/// Purely verify one canonically ordered batch of sparse replication evidence.
///
/// Evidence must be in strictly increasing intrinsic commit-id order. Every
/// commit and grant is strictly signature-verified, and each pair must name
/// the same collection and author. No blob lookup or destination mutation is
/// performed.
pub fn prepare_incoming_collection_batch(
    evidence: Vec<(CollectionCommit, CollectionGossip)>,
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

    for (index, (commit, grant)) in evidence.iter().enumerate() {
        validate_evidence(commit, grant)
            .map_err(|source| IncomingBatchValidationError::Evidence { index, source })?;
    }

    Ok(PreparedIncomingCollectionBatch { evidence })
}

fn validate_evidence(
    commit: &CollectionCommit,
    grant: &CollectionGossip,
) -> Result<(), IncomingValidationError> {
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
    Ok(())
}

/// A strictly verified batch whose authorization policy has not run yet.
#[derive(Clone, Debug)]
#[must_use = "a prepared incoming batch has no effect until authorized and admitted"]
pub struct PreparedIncomingCollectionBatch {
    evidence: Vec<(CollectionCommit, CollectionGossip)>,
}

impl PreparedIncomingCollectionBatch {
    /// Number of strictly verified evidence pairs observed in this batch.
    pub fn len(&self) -> usize {
        self.evidence.len()
    }

    /// Whether the batch contains no evidence pairs.
    pub fn is_empty(&self) -> bool {
        self.evidence.is_empty()
    }

    /// Decide every verified evidence pair without exposing a destination.
    ///
    /// Policy failure returns directly and cannot leave partial storage
    /// effects because this phase has no store parameter. Accepted grants and
    /// commits are reduced to their distinct grow-only set members before the
    /// mutation phase.
    pub fn authorize<AuthorizationError, Authorize>(
        self,
        mut authorize: Authorize,
    ) -> Result<AuthorizedIncomingCollectionBatch, AuthorizationError>
    where
        Authorize: FnMut(&CollectionCommit, &CollectionGossip) -> Result<bool, AuthorizationError>,
    {
        let observed = self.evidence.len();
        let mut admitted = 0;
        let mut grants = BTreeSet::new();
        let mut commits = BTreeMap::new();

        for (commit, grant) in self.evidence {
            if authorize(&commit, &grant)? {
                admitted += 1;
                grants.insert(grant);
                commits.insert(commit.id(), commit);
            }
        }

        Ok(AuthorizedIncomingCollectionBatch {
            grants,
            commits,
            counts: IncomingBatchCounts {
                observed,
                admitted,
                denied: observed - admitted,
            },
        })
    }
}

/// A verified batch whose policy decisions are complete and immutable.
#[derive(Clone, Debug)]
#[must_use = "an authorized incoming batch has no effect until admitted"]
pub struct AuthorizedIncomingCollectionBatch {
    grants: BTreeSet<CollectionGossip>,
    commits: BTreeMap<Id, CollectionCommit>,
    counts: IncomingBatchCounts,
}

impl AuthorizedIncomingCollectionBatch {
    /// Counts fixed by the completed authorization phase.
    pub fn counts(&self) -> IncomingBatchCounts {
        self.counts
    }

    /// Insert distinct accepted grants and commits, then flush exactly once.
    ///
    /// Empty and all-denied batches perform no mutation and no flush.
    /// Replaying an accepted batch is logically idempotent under the grow-only
    /// storage trait contracts.
    pub fn admit<S>(self, store: &mut S) -> IncomingBatchAdmissionResult<S>
    where
        S: CollectionGossipStore + CollectionStore + StorageFlush,
    {
        if self.commits.is_empty() {
            return Ok(self.counts);
        }

        for grant in self.grants {
            store
                .gossip(grant)
                .map_err(IncomingBatchAdmissionError::GossipInsert)?;
        }
        for commit in self.commits.into_values() {
            store
                .insert(CollectionRecord::Commit(commit))
                .map_err(IncomingBatchAdmissionError::CommitInsert)?;
        }
        store.flush().map_err(IncomingBatchAdmissionError::Flush)?;

        Ok(self.counts)
    }
}

/// Aggregate result of one batch admission attempt.
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

/// Pure validation failure for one commit/grant pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
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
}

/// Destination-specific result type for
/// [`AuthorizedIncomingCollectionBatch::admit`].
pub type IncomingBatchAdmissionResult<S> = Result<
    IncomingBatchCounts,
    IncomingBatchAdmissionError<
        <S as CollectionGossipStore>::GossipError,
        <S as CollectionStore>::InsertError,
        <S as StorageFlush>::Error,
    >,
>;

/// Failure while durably publishing an already-authorized batch.
#[derive(Debug, thiserror::Error)]
pub enum IncomingBatchAdmissionError<GossipError, InsertError, FlushError> {
    /// A matching gossip grant could not be inserted.
    #[error("failed to insert incoming batch gossip grant: {0}")]
    GossipInsert(#[source] GossipError),
    /// A commit record could not be inserted.
    #[error("failed to insert incoming batch commit: {0}")]
    CommitInsert(#[source] InsertError),
    /// Accepted evidence could not be made durable.
    #[error("failed to flush incoming batch evidence: {0}")]
    Flush(#[source] FlushError),
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::convert::Infallible;
    use std::fmt;

    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace_core::collection::{CollectionData, CollectionHandle};
    use triblespace_core::inline::Inline;
    use triblespace_core::inline::encodings::hash::Handle;

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Event {
        Gossip(CollectionGossip),
        Insert(Id),
        Flush,
    }

    #[derive(Default)]
    struct ProbeStore {
        events: Vec<Event>,
        pending_gossips: BTreeSet<CollectionGossip>,
        durable_gossips: BTreeSet<CollectionGossip>,
        pending_records: BTreeMap<Id, CollectionRecord>,
        durable_records: BTreeMap<Id, CollectionRecord>,
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
            self.events.push(Event::Insert(record.id()));
            self.pending_records.entry(record.id()).or_insert(record);
            Ok(())
        }
    }

    impl StorageFlush for ProbeStore {
        type Error = Infallible;

        fn flush(&mut self) -> Result<(), Self::Error> {
            self.events.push(Event::Flush);
            self.durable_gossips.append(&mut self.pending_gossips);
            self.durable_records.append(&mut self.pending_records);
            Ok(())
        }
    }

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn metadata(byte: u8) -> Inline<Handle<SimpleArchive>> {
        Inline::new([byte; 32])
    }

    fn pair(
        author: &ed25519_dalek::SigningKey,
        collection: CollectionHandle,
        byte: u8,
    ) -> (CollectionCommit, CollectionGossip) {
        (
            CollectionCommit::sign(
                author,
                collection,
                data(byte),
                metadata(byte.wrapping_add(1)),
            ),
            CollectionGossip::sign(author, collection),
        )
    }

    fn fixture() -> Vec<(CollectionCommit, CollectionGossip)> {
        let author = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
        let collection = collection(1);
        let mut evidence = vec![pair(&author, collection, 2), pair(&author, collection, 4)];
        evidence.sort_by_key(|(commit, _)| commit.id());
        evidence
    }

    #[test]
    fn valid_sparse_evidence_is_admitted_without_referenced_blobs() {
        let evidence = fixture();
        let expected_commits = evidence
            .iter()
            .map(|(commit, _)| commit.id())
            .collect::<BTreeSet<_>>();
        let expected_grant = evidence[0].1;

        let prepared = prepare_incoming_collection_batch(evidence).unwrap();
        assert_eq!(prepared.len(), 2);
        assert!(!prepared.is_empty());
        let authorized = prepared
            .authorize(|_, _| Ok::<_, Infallible>(true))
            .unwrap();
        assert_eq!(
            authorized.counts(),
            IncomingBatchCounts {
                observed: 2,
                admitted: 2,
                denied: 0,
            }
        );

        // ProbeStore has no blob-storage implementation at all. Admission is
        // therefore proof that descriptor/data/metadata residency is not a
        // prerequisite.
        let mut store = ProbeStore::default();
        assert_eq!(authorized.admit(&mut store).unwrap().admitted, 2);
        assert_eq!(store.durable_gossips, BTreeSet::from([expected_grant]));
        assert_eq!(
            store
                .durable_records
                .keys()
                .copied()
                .collect::<BTreeSet<_>>(),
            expected_commits
        );
        assert_eq!(
            store
                .events
                .iter()
                .filter(|event| **event == Event::Flush)
                .count(),
            1
        );
        assert!(matches!(store.events.last(), Some(Event::Flush)));
    }

    #[test]
    fn denial_and_authorization_error_cannot_mutate_destination() {
        let prepared = prepare_incoming_collection_batch(fixture()).unwrap();
        let denied = prepared
            .clone()
            .authorize(|_, _| Ok::<_, Infallible>(false))
            .unwrap();
        let mut store = ProbeStore::default();
        assert_eq!(
            denied.admit(&mut store).unwrap(),
            IncomingBatchCounts {
                observed: 2,
                admitted: 0,
                denied: 2,
            }
        );
        assert!(store.events.is_empty());

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct PolicyError;
        impl fmt::Display for PolicyError {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "policy failed")
            }
        }

        let result = prepared.authorize(|_, _| Err::<bool, _>(PolicyError));
        assert!(matches!(result, Err(PolicyError)));
        assert!(store.events.is_empty());
    }

    #[test]
    fn empty_batch_is_a_storage_noop() {
        let prepared = prepare_incoming_collection_batch(Vec::new()).unwrap();
        assert!(prepared.is_empty());
        let authorized = prepared
            .authorize(|_, _| Ok::<_, Infallible>(true))
            .unwrap();
        let mut store = ProbeStore::default();
        assert_eq!(
            authorized.admit(&mut store).unwrap(),
            IncomingBatchCounts::default()
        );
        assert!(store.events.is_empty());
    }

    #[test]
    fn retry_is_logically_idempotent() {
        let evidence = fixture();
        let mut store = ProbeStore::default();
        for _ in 0..2 {
            prepare_incoming_collection_batch(evidence.clone())
                .unwrap()
                .authorize(|_, _| Ok::<_, Infallible>(true))
                .unwrap()
                .admit(&mut store)
                .unwrap();
        }
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
    fn batch_requires_canonical_unique_commit_order() {
        let mut reversed = fixture();
        reversed.reverse();
        assert!(matches!(
            prepare_incoming_collection_batch(reversed),
            Err(IncomingBatchValidationError::NonCanonicalEvidenceOrder { .. })
        ));

        let pair = fixture()[0];
        assert!(matches!(
            prepare_incoming_collection_batch(vec![pair, pair]),
            Err(IncomingBatchValidationError::NonCanonicalEvidenceOrder {
                previous,
                current,
            }) if previous == current
        ));
    }

    #[test]
    fn commit_and_grant_signatures_are_strictly_verified() {
        let (commit, grant) = fixture()[0];
        let mut commit_bytes = commit.to_bytes();
        commit_bytes[191] ^= 1;
        let invalid_commit = CollectionCommit::from_bytes(commit_bytes);
        assert!(matches!(
            prepare_incoming_collection_batch(vec![(invalid_commit, grant)]),
            Err(IncomingBatchValidationError::Evidence {
                index: 0,
                source: IncomingValidationError::InvalidCommitSignature(_),
            })
        ));

        let mut grant_bytes = grant.to_bytes();
        grant_bytes[127] ^= 1;
        let invalid_grant = CollectionGossip::from_bytes(grant_bytes);
        assert!(matches!(
            prepare_incoming_collection_batch(vec![(commit, invalid_grant)]),
            Err(IncomingBatchValidationError::Evidence {
                index: 0,
                source: IncomingValidationError::InvalidGossipSignature(_),
            })
        ));
    }

    #[test]
    fn grant_must_match_both_collection_and_author() {
        let author = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
        let other_author = ed25519_dalek::SigningKey::from_bytes(&[19; 32]);
        let (commit, _) = pair(&author, collection(1), 2);

        let wrong_collection = CollectionGossip::sign(&author, collection(9));
        assert_eq!(
            prepare_incoming_collection_batch(vec![(commit, wrong_collection)]).unwrap_err(),
            IncomingBatchValidationError::Evidence {
                index: 0,
                source: IncomingValidationError::GossipCollectionMismatch,
            }
        );

        let wrong_author = CollectionGossip::sign(&other_author, commit.collection());
        assert_eq!(
            prepare_incoming_collection_batch(vec![(commit, wrong_author)]).unwrap_err(),
            IncomingBatchValidationError::Evidence {
                index: 0,
                source: IncomingValidationError::GossipAuthorMismatch,
            }
        );
    }
}
