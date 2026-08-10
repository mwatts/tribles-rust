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

use super::{
    CollectionCommit, CollectionDefinition, CollectionDerive, CollectionMerge, CollectionRecord,
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
    definitions: Vec<CollectionDefinition>,
    commits: Vec<CollectionCommit>,
    merges: Vec<CollectionMerge>,
    derives: Vec<CollectionDerive>,
    diagnostics: Vec<CollectionRecordDiagnostic>,
}

impl DiscoveredCollectionRecords {
    /// Canonical collection definitions, ordered by intrinsic id.
    pub fn definitions(&self) -> &[CollectionDefinition] {
        &self.definitions
    }

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
        self.definitions
            .sort_unstable_by_key(CollectionDefinition::id);
        self.definitions.dedup_by_key(|record| record.id());
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
    let mut discovered = DiscoveredCollectionRecords::default();
    let records = store.records().map_err(CollectionDiscoveryError::Records)?;

    for record in records {
        let record = record.map_err(CollectionDiscoveryError::Records)?;
        match record {
            CollectionRecord::Definition(record) => discovered.definitions.push(record),
            CollectionRecord::Commit(record) => match record.verify_strict() {
                Ok(()) => discovered.commits.push(record),
                Err(error) => discovered.diagnostics.push(CollectionRecordDiagnostic {
                    id: record.id(),
                    error: CollectionRecordDiagnosticError::InvalidCommit(error),
                }),
            },
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

    use crate::collection::{empty_metadata_handle, CollectionData};
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

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn hash(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn fixture_records() -> (Vec<CollectionRecord>, CollectionCommit) {
        let definition = CollectionDefinition::new(id(1), id(2), id(3));
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            definition.id(),
            hash(4),
            empty_metadata_handle(),
        );
        let merge = CollectionMerge::new(definition.id(), hash(4), hash(5), hash(6));
        let derive = CollectionDerive::new(definition.id(), id(7), hash(4), hash(8));

        let (r, mut s) = commit.signature();
        s.raw[0] ^= 1;
        let invalid_commit = CollectionCommit::from_parts(
            commit.collection(),
            commit.data(),
            commit.metadata(),
            commit.public_key(),
            r,
            s,
        );

        (
            vec![
                CollectionRecord::Definition(definition),
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
        assert_eq!(forward_records.definitions().len(), 1);
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
    }
}
