//! Discovery over native collection-record storage.
//!
//! A [`CollectionStore`](super::CollectionStore) is responsible for decoding
//! its physical representation into structurally canonical typed records.
//! Discovery therefore performs no blob-store scan and has no malformed-record
//! recovery path: a structural storage failure is fatal. It verifies signed
//! commits, classifies records, and canonicalizes the resulting semantic view.

use std::error::Error;
use std::fmt;

use crate::id::Id;
use crate::inline::encodings::ed25519::ED25519PublicKey;
use crate::inline::Inline;

use super::{
    CollectionCommit, CollectionDerive, CollectionId, CollectionMerge, CollectionRecord,
    CollectionStore, CommitVerificationError,
};

/// One collection record with a discovery-time validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionRecordDiagnostic {
    /// Intrinsic id of the record carrying this diagnostic.
    pub id: Id,
    /// Cryptographic validation failure.
    pub error: CollectionRecordDiagnosticError,
}

/// Observable semantic validation failure for a structurally valid record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionRecordDiagnosticError {
    /// A structurally canonical commit failed strict Ed25519 verification.
    InvalidCommit(CommitVerificationError),
}

impl fmt::Display for CollectionRecordDiagnosticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommit(error) => write!(f, "invalid collection commit: {error}"),
        }
    }
}

impl Error for CollectionRecordDiagnosticError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidCommit(error) => Some(error),
        }
    }
}

/// Structurally canonical records and diagnostics from one store enumeration.
///
/// Every collection is sorted by intrinsic record id, as are diagnostics. The
/// result therefore does not expose backend enumeration or append order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiscoveredCollectionRecords {
    commits: Vec<CollectionCommit>,
    merges: Vec<CollectionMerge>,
    derives: Vec<CollectionDerive>,
    diagnostics: Vec<CollectionRecordDiagnostic>,
}

impl DiscoveredCollectionRecords {
    /// Commits with valid strict self-signatures, ordered by intrinsic id.
    ///
    /// Signature validity does not authorize the signing key. Callers apply
    /// local authorization policy before treating a commit as a membership
    /// root.
    pub fn commits(&self) -> &[CollectionCommit] {
        &self.commits
    }

    /// Structurally canonical merge claims, ordered by intrinsic id.
    pub fn merges(&self) -> &[CollectionMerge] {
        &self.merges
    }

    /// Structurally canonical derive claims, ordered by intrinsic id.
    pub fn derives(&self) -> &[CollectionDerive] {
        &self.derives
    }

    /// Structurally valid records whose semantic verification failed.
    pub fn diagnostics(&self) -> &[CollectionRecordDiagnostic] {
        &self.diagnostics
    }

    fn canonicalize(&mut self) {
        self.commits.sort_unstable_by_key(CollectionCommit::id);
        self.commits.dedup_by_key(|record| record.id());
        self.merges.sort_unstable_by_key(CollectionMerge::id);
        self.merges.dedup_by_key(|record| record.id());
        self.derives.sort_unstable_by_key(CollectionDerive::id);
        self.derives.dedup_by_key(|record| record.id());
        self.diagnostics.sort_unstable_by_key(|entry| entry.id);
        self.diagnostics.dedup_by_key(|entry| entry.id);
    }
}

/// A native-store failure that prevents one complete record enumeration.
#[derive(Debug)]
pub enum CollectionDiscoveryError<RecordsError> {
    /// The store could not start or complete record enumeration.
    Records(RecordsError),
}

impl<RecordsError> fmt::Display for CollectionDiscoveryError<RecordsError>
where
    RecordsError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Records(error) => write!(f, "failed to enumerate collection records: {error}"),
        }
    }
}

impl<RecordsError> Error for CollectionDiscoveryError<RecordsError>
where
    RecordsError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Records(error) => Some(error),
        }
    }
}

/// Discover and cryptographically classify native collection records.
///
/// The store owns physical decoding and returns only structurally canonical
/// [`CollectionRecord`] values. Any failure to begin or continue enumeration
/// aborts discovery rather than turning storage corruption into a partial
/// semantic view. Commits enter the accepted result only after strict
/// self-signature verification. This establishes authorship, not
/// authorization; callers still choose which signing keys may introduce
/// membership roots. Representation-specific `MERGE` and `DERIVE` validation
/// likewise remains the resolver callback's responsibility.
pub fn discover_collection_records<S>(
    store: &mut S,
) -> Result<DiscoveredCollectionRecords, CollectionDiscoveryError<S::RecordsError>>
where
    S: CollectionStore,
{
    discover_collection_records_matching(store, |_| true)
}

/// Discover records for one exact collection and authorized signer.
///
/// A commit's descriptor and public-key fields are structurally available
/// before its signature is verified. Commits outside this exact scope are
/// therefore discarded without paying for Ed25519 verification. Matching
/// commits still undergo the same strict verification and produce the same
/// diagnostics as [`discover_collection_records`].
///
/// `MERGE` and `DERIVE` records are retained in full. They are unsigned
/// equations whose relevance can span collection boundaries, so narrowing
/// them here would change the semantic graph seen by downstream resolution.
/// Physical decoding remains the store's responsibility: a malformed record
/// or iteration failure is fatal even when it appears outside the requested
/// commit scope.
pub fn discover_collection_records_scoped<S>(
    store: &mut S,
    collection: CollectionId,
    signer: Inline<ED25519PublicKey>,
) -> Result<DiscoveredCollectionRecords, CollectionDiscoveryError<S::RecordsError>>
where
    S: CollectionStore,
{
    discover_collection_records_matching(store, |commit| {
        commit.collection() == collection && commit.public_key() == signer
    })
}

fn discover_collection_records_matching<S, F>(
    store: &mut S,
    mut include_commit: F,
) -> Result<DiscoveredCollectionRecords, CollectionDiscoveryError<S::RecordsError>>
where
    S: CollectionStore,
    F: FnMut(&CollectionCommit) -> bool,
{
    let mut discovered = DiscoveredCollectionRecords::default();
    let records = store.records().map_err(CollectionDiscoveryError::Records)?;

    for record in records {
        let record = record.map_err(CollectionDiscoveryError::Records)?;
        match record {
            CollectionRecord::Commit(record) if include_commit(&record) => {
                match record.verify_strict() {
                    Ok(()) => discovered.commits.push(record),
                    Err(error) => discovered.diagnostics.push(CollectionRecordDiagnostic {
                        id: record.id(),
                        error: CollectionRecordDiagnosticError::InvalidCommit(error),
                    }),
                }
            }
            CollectionRecord::Commit(_) => {}
            CollectionRecord::Merge(record) => discovered.merges.push(record),
            CollectionRecord::Derive(record) => discovered.derives.push(record),
        }
    }

    discovered.canonicalize();
    Ok(discovered)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use crate::collection::{empty_metadata_handle, CollectionData, CollectionId};
    use crate::inline::Inline;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ProbeRecordsError(&'static str);

    impl fmt::Display for ProbeRecordsError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for ProbeRecordsError {}

    #[derive(Default)]
    struct ProbeStore {
        start_error: Option<ProbeRecordsError>,
        records: Vec<Result<CollectionRecord, ProbeRecordsError>>,
    }

    impl CollectionStore for ProbeStore {
        type RecordsError = ProbeRecordsError;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Self::RecordsError>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            if let Some(error) = self.start_error {
                return Err(error);
            }
            Ok(self.records.clone().into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.records.push(Ok(record));
            Ok(())
        }
    }

    fn hash(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn collection(byte: u8) -> CollectionId {
        Inline::new([byte; 32])
    }

    fn signer(key: &SigningKey) -> Inline<ED25519PublicKey> {
        Inline::new(key.verifying_key().to_bytes())
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

    fn fixture_records() -> (Vec<CollectionRecord>, CollectionCommit) {
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            collection(1),
            hash(4),
            empty_metadata_handle(),
        );
        let merge = CollectionMerge::new(collection(1), hash(4), hash(5), hash(6));
        let derive = CollectionDerive::new(collection(1), collection(7), hash(4), hash(8));

        let invalid_commit = invalid_signature(commit);

        (
            vec![
                CollectionRecord::Commit(commit),
                CollectionRecord::Merge(merge),
                CollectionRecord::Derive(derive),
                CollectionRecord::Commit(invalid_commit),
            ],
            invalid_commit,
        )
    }

    #[test]
    fn native_records_are_verified_classified_and_order_independent() {
        let (records, invalid_commit) = fixture_records();
        let mut forward = ProbeStore {
            records: records.iter().copied().map(Ok).collect(),
            ..ProbeStore::default()
        };
        let mut reverse = ProbeStore {
            records: records.iter().rev().copied().map(Ok).collect(),
            ..ProbeStore::default()
        };

        let forward_records = discover_collection_records(&mut forward).unwrap();
        let reverse_records = discover_collection_records(&mut reverse).unwrap();

        assert_eq!(forward_records, reverse_records);
        assert_eq!(forward_records.commits().len(), 1);
        assert_eq!(forward_records.merges().len(), 1);
        assert_eq!(forward_records.derives().len(), 1);
        assert_eq!(
            forward_records.diagnostics(),
            &[CollectionRecordDiagnostic {
                id: invalid_commit.id(),
                error: CollectionRecordDiagnosticError::InvalidCommit(
                    CommitVerificationError::InvalidSignature,
                ),
            }]
        );
    }

    #[test]
    fn scoped_discovery_ignores_unrelated_invalid_signatures_and_rejects_relevant_ones() {
        let target = collection(1);
        let other = collection(2);
        let authorized_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let valid =
            CollectionCommit::sign(&authorized_key, target, hash(1), empty_metadata_handle());
        let relevant_invalid = invalid_signature(CollectionCommit::sign(
            &authorized_key,
            target,
            hash(2),
            empty_metadata_handle(),
        ));
        let wrong_collection = invalid_signature(CollectionCommit::sign(
            &authorized_key,
            other,
            hash(3),
            empty_metadata_handle(),
        ));
        let wrong_signer = invalid_signature(CollectionCommit::sign(
            &foreign_key,
            target,
            hash(4),
            empty_metadata_handle(),
        ));
        let target_merge = CollectionMerge::new(target, hash(1), hash(2), hash(5));
        let other_merge = CollectionMerge::new(other, hash(3), hash(4), hash(6));
        let crossing_derive = CollectionDerive::new(target, other, hash(5), hash(6));

        let mut store = ProbeStore {
            records: [
                CollectionRecord::Commit(wrong_signer),
                CollectionRecord::Merge(other_merge),
                CollectionRecord::Commit(relevant_invalid),
                CollectionRecord::Derive(crossing_derive),
                CollectionRecord::Commit(valid),
                CollectionRecord::Commit(wrong_collection),
                CollectionRecord::Merge(target_merge),
            ]
            .into_iter()
            .map(Ok)
            .collect(),
            ..ProbeStore::default()
        };

        let discovered =
            discover_collection_records_scoped(&mut store, target, signer(&authorized_key))
                .unwrap();

        assert_eq!(discovered.commits(), &[valid]);
        assert_eq!(
            discovered.diagnostics(),
            &[CollectionRecordDiagnostic {
                id: relevant_invalid.id(),
                error: CollectionRecordDiagnosticError::InvalidCommit(
                    CommitVerificationError::InvalidSignature,
                ),
            }]
        );
        assert_eq!(discovered.merges().len(), 2);
        assert!(discovered.merges().contains(&target_merge));
        assert!(discovered.merges().contains(&other_merge));
        assert_eq!(discovered.derives(), &[crossing_derive]);
    }

    #[test]
    fn scoped_discovery_equals_full_discovery_projected_to_valid_matching_commits() {
        let target = collection(1);
        let other = collection(2);
        let authorized_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let matching = [
            CollectionCommit::sign(&authorized_key, target, hash(1), empty_metadata_handle()),
            CollectionCommit::sign(&authorized_key, target, hash(2), empty_metadata_handle()),
        ];
        let records = vec![
            CollectionRecord::Commit(CollectionCommit::sign(
                &foreign_key,
                target,
                hash(3),
                empty_metadata_handle(),
            )),
            CollectionRecord::Merge(CollectionMerge::new(other, hash(3), hash(4), hash(5))),
            CollectionRecord::Commit(matching[1]),
            CollectionRecord::Derive(CollectionDerive::new(other, target, hash(5), hash(2))),
            CollectionRecord::Commit(CollectionCommit::sign(
                &authorized_key,
                other,
                hash(4),
                empty_metadata_handle(),
            )),
            CollectionRecord::Commit(matching[0]),
            CollectionRecord::Merge(CollectionMerge::new(target, hash(1), hash(2), hash(6))),
        ];
        let mut full_store = ProbeStore {
            records: records.iter().copied().map(Ok).collect(),
            ..ProbeStore::default()
        };
        let mut scoped_store = ProbeStore {
            records: records.into_iter().map(Ok).collect(),
            ..ProbeStore::default()
        };

        let full = discover_collection_records(&mut full_store).unwrap();
        let scoped =
            discover_collection_records_scoped(&mut scoped_store, target, signer(&authorized_key))
                .unwrap();
        let expected_commits: Vec<_> = full
            .commits()
            .iter()
            .copied()
            .filter(|commit| {
                commit.collection() == target && commit.public_key() == signer(&authorized_key)
            })
            .collect();

        assert_eq!(scoped.commits(), expected_commits);
        assert_eq!(scoped.merges(), full.merges());
        assert_eq!(scoped.derives(), full.derives());
        assert_eq!(scoped.diagnostics(), full.diagnostics());
        assert_eq!(expected_commits.len(), matching.len());
    }

    #[test]
    fn start_and_iteration_failures_are_fatal() {
        let mut start_failure = ProbeStore {
            start_error: Some(ProbeRecordsError("start")),
            ..ProbeStore::default()
        };
        assert!(matches!(
            discover_collection_records(&mut start_failure),
            Err(CollectionDiscoveryError::Records(ProbeRecordsError(
                "start"
            )))
        ));

        let (records, _) = fixture_records();
        let mut iteration_failure = ProbeStore {
            records: vec![Ok(records[0]), Err(ProbeRecordsError("iteration"))],
            ..ProbeStore::default()
        };
        assert!(matches!(
            discover_collection_records(&mut iteration_failure),
            Err(CollectionDiscoveryError::Records(ProbeRecordsError(
                "iteration"
            )))
        ));

        let mut scoped_failure = ProbeStore {
            records: vec![Ok(records[0]), Err(ProbeRecordsError("scoped iteration"))],
            ..ProbeStore::default()
        };
        assert!(matches!(
            discover_collection_records_scoped(
                &mut scoped_failure,
                collection(99),
                signer(&SigningKey::from_bytes(&[9; 32])),
            ),
            Err(CollectionDiscoveryError::Records(ProbeRecordsError(
                "scoped iteration"
            )))
        ));
    }
}
