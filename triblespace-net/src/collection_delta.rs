//! Policy-independent collection-record delta mechanics.
//!
//! This module owns only the immutable evidence boundary: strict framing,
//! intrinsic collection matching, COMMIT signature verification, canonical
//! id ordering, and bounded `current - previous` selection. It deliberately
//! does not resolve referenced blobs or decide READ/WRITE policy. A future
//! authorized overlay can therefore store sparse MERGE/DERIVE equations as
//! inert evidence and apply semantic validation only when a resolver uses one.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use triblespace_core::collection::{
    CollectionHandle, CollectionRead, CollectionRecord, CollectionRecordSelector,
    CommitVerificationError, RecordDecodeError,
};
use triblespace_core::id::Id;
use triblespace_core::patch::{Blake3Merkle, Entry as PatchEntry, IdentitySchema, PATCH};

use crate::patch_repair::PatchSummary;

/// Canonical valued PATCH of the records naming one exact collection.
#[derive(Clone, Debug)]
pub struct CollectionRecordPatch {
    collection: CollectionHandle,
    records: PATCH<16, IdentitySchema, CollectionRecord, Blake3Merkle>,
}

impl CollectionRecordPatch {
    /// Exact collection named by every record in this PATCH.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Root and count of this immutable per-collection PATCH.
    pub fn summary(&self) -> PatchSummary {
        PatchSummary::from_patch(&self.records)
    }

    /// Number of canonical records in this collection overlay.
    pub fn len(&self) -> u64 {
        self.records.len()
    }

    /// Whether this collection overlay has no known records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Look up one record by intrinsic id.
    pub fn get(&self, id: Id) -> Option<CollectionRecord> {
        self.records.get(&id.raw()).copied()
    }

    /// Enumerate canonical records in intrinsic-id order.
    pub fn records(&self) -> impl Iterator<Item = CollectionRecord> + '_ {
        self.records.iter_ordered().map(|id| {
            *self
                .records
                .get(id)
                .expect("an ordered per-collection PATCH key retains its record value")
        })
    }
}

/// Failure while constructing an exact per-collection record PATCH.
#[derive(Debug)]
pub enum CollectionRecordPatchError<E> {
    Store(E),
    Evidence(CollectionDeltaError),
}

impl<E: fmt::Display> fmt::Display for CollectionRecordPatchError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(f, "select collection records: {error}"),
            Self::Evidence(error) => error.fmt(f),
        }
    }
}

impl<E> Error for CollectionRecordPatchError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Evidence(error) => Some(error),
        }
    }
}

/// Result of comparing two coherent, monotone collection-record snapshots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionDeltaSelection {
    /// Complete bounded push delta, in intrinsic record-id order.
    Push(Vec<CollectionRecord>),
    /// The gap is too large for push. Send only the collection root/count wake
    /// hint and let the per-collection PATCH repair protocol reconcile it.
    Repair { missing: usize },
}

/// Failure at the sparse immutable-evidence boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionDeltaError {
    Decode(RecordDecodeError),
    InvalidCommit(CommitVerificationError),
    WrongCollection,
    MismatchedCollections,
    IntrinsicIdCollision(Id),
    NonMonotoneSnapshot,
}

impl fmt::Display for CollectionDeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "decode collection record: {error}"),
            Self::InvalidCommit(error) => write!(f, "verify collection COMMIT: {error}"),
            Self::WrongCollection => write!(f, "record names another collection"),
            Self::MismatchedCollections => {
                write!(f, "collection-record snapshots name different collections")
            }
            Self::IntrinsicIdCollision(id) => {
                write!(f, "distinct records share intrinsic id {id}")
            }
            Self::NonMonotoneSnapshot => {
                write!(f, "current collection-record snapshot is not a superset")
            }
        }
    }
}

impl Error for CollectionDeltaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::InvalidCommit(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RecordDecodeError> for CollectionDeltaError {
    fn from(error: RecordDecodeError) -> Self {
        Self::Decode(error)
    }
}

/// Build the exact valued PATCH for one collection through its semantic
/// selector.
///
/// This is the sole collection-overlay construction path: it never asks the
/// caller for the store's global record stream. Backends remain free to answer
/// the selector from a secondary index or another sparse representation.
pub fn collection_record_patch<R>(
    snapshot: &R,
    collection: CollectionHandle,
) -> Result<CollectionRecordPatch, CollectionRecordPatchError<R::RecordsError>>
where
    R: CollectionRead,
{
    let selectors = BTreeSet::from([CollectionRecordSelector::Collection(collection)]);
    let records = snapshot
        .select_records(&selectors)
        .map_err(CollectionRecordPatchError::Store)?;
    canonical_records(collection, records).map_err(CollectionRecordPatchError::Evidence)
}

/// Encode one sparse record after checking its intrinsic collection and the
/// embedded COMMIT signature. WRITE authorization is intentionally absent: it
/// governs activation, not whether canonical inert evidence may exist.
pub fn encode_record(
    expected: CollectionHandle,
    record: CollectionRecord,
) -> Result<Vec<u8>, CollectionDeltaError> {
    validate_record(expected, record)?;
    Ok(record.to_bytes())
}

/// Strictly decode one complete self-tagged record for an implicit collection
/// overlay. Trailing bytes, noncanonical MERGE inputs, unknown tags, wrong
/// collections, and invalid COMMIT signatures fail before insertion.
pub fn decode_record(
    expected: CollectionHandle,
    bytes: &[u8],
) -> Result<CollectionRecord, CollectionDeltaError> {
    let record = CollectionRecord::from_bytes(bytes)?;
    validate_record(expected, record)?;
    Ok(record)
}

/// Select one bounded complete live delta from coherent old/new snapshots.
///
/// This function requires an actual prior snapshot; it is not a durable
/// announced cursor. A caller with no prior in-memory snapshot, a nonmonotone
/// observation, or `Repair` must send only `(collection, root, count)` as a
/// wake hint and use PATCH repair rather than flooding the current set.
pub fn select_bounded_delta(
    previous: &CollectionRecordPatch,
    current: &CollectionRecordPatch,
    max_push_records: usize,
) -> Result<CollectionDeltaSelection, CollectionDeltaError> {
    if previous.collection != current.collection {
        return Err(CollectionDeltaError::MismatchedCollections);
    }
    if previous
        .records
        .iter_ordered()
        .any(|id| current.records.get(id) != previous.records.get(id))
    {
        return Err(CollectionDeltaError::NonMonotoneSnapshot);
    }

    let delta = current.records.difference(&previous.records);
    let missing = usize::try_from(delta.len())
        .expect("a PATCH built from one process-local iterator fits usize");
    if missing > max_push_records {
        return Ok(CollectionDeltaSelection::Repair { missing });
    }
    Ok(CollectionDeltaSelection::Push(
        delta
            .iter_ordered()
            .map(|id| {
                *delta
                    .get(id)
                    .expect("an ordered PATCH key retains its record value")
            })
            .collect(),
    ))
}

fn validate_record(
    expected: CollectionHandle,
    record: CollectionRecord,
) -> Result<(), CollectionDeltaError> {
    if record_collection(record) != expected {
        return Err(CollectionDeltaError::WrongCollection);
    }
    if let CollectionRecord::Commit(commit) = record {
        commit
            .verify_strict()
            .map_err(CollectionDeltaError::InvalidCommit)?;
    }
    Ok(())
}

fn record_collection(record: CollectionRecord) -> CollectionHandle {
    match record {
        CollectionRecord::Commit(record) => record.collection(),
        CollectionRecord::Merge(record) => record.collection(),
        CollectionRecord::Derive(record) => record.collection(),
    }
}

fn canonical_records(
    expected: CollectionHandle,
    records: impl IntoIterator<Item = CollectionRecord>,
) -> Result<CollectionRecordPatch, CollectionDeltaError> {
    let mut canonical = PATCH::new();
    for record in records {
        validate_record(expected, record)?;
        let id = record.id();
        let key = id.raw();
        if let Some(existing) = canonical.get(&key) {
            if existing != &record {
                return Err(CollectionDeltaError::IntrinsicIdCollision(id));
            }
            continue;
        }
        canonical.insert(&PatchEntry::with_value(&key, record));
    }
    Ok(CollectionRecordPatch {
        collection: expected,
        records: canonical,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;
    use triblespace_core::collection::{
        COLLECTION_RECORD_KIND_MERGE_V1, CollectionCommit, CollectionData, CollectionDerive,
        CollectionMerge, empty_metadata_handle,
    };
    use triblespace_core::inline::Inline;

    use super::*;

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn records(expected: CollectionHandle) -> [CollectionRecord; 3] {
        [
            CollectionRecord::Commit(CollectionCommit::sign(
                &SigningKey::from_bytes(&[7; 32]),
                expected,
                data(1),
                empty_metadata_handle(),
            )),
            CollectionRecord::Merge(CollectionMerge::new(expected, data(2), data(3), data(4))),
            CollectionRecord::Derive(CollectionDerive::new(expected, data(4), data(5))),
        ]
    }

    fn patch(
        expected: CollectionHandle,
        records: impl IntoIterator<Item = CollectionRecord>,
    ) -> CollectionRecordPatch {
        canonical_records(expected, records).unwrap()
    }

    #[test]
    fn all_sparse_record_variants_roundtrip_for_the_implicit_collection() {
        let expected = collection(1);
        for record in records(expected) {
            let bytes = encode_record(expected, record).unwrap();
            assert_eq!(decode_record(expected, &bytes).unwrap(), record);
        }
    }

    #[test]
    fn framing_collection_and_commit_signature_fail_before_admission() {
        let expected = collection(1);
        let commit = records(expected)[0];
        let bytes = encode_record(expected, commit).unwrap();

        assert_eq!(
            decode_record(collection(2), &bytes),
            Err(CollectionDeltaError::WrongCollection)
        );
        let mut tampered = bytes.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert!(matches!(
            decode_record(expected, &tampered),
            Err(CollectionDeltaError::InvalidCommit(_))
        ));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            decode_record(expected, &trailing),
            Err(CollectionDeltaError::Decode(_))
        ));
        assert!(decode_record(expected, &bytes[..bytes.len() - 1]).is_err());
        assert!(decode_record(expected, &[99]).is_err());
    }

    #[test]
    fn noncanonical_merge_inputs_fail_before_admission() {
        let expected = collection(1);
        let merge = CollectionMerge::new(expected, data(2), data(3), data(4));
        let mut bytes = merge.to_bytes();
        bytes[32..64].fill(9);
        bytes[64..96].fill(1);
        let mut tagged = Vec::with_capacity(1 + bytes.len());
        tagged.push(COLLECTION_RECORD_KIND_MERGE_V1);
        tagged.extend_from_slice(&bytes);
        assert!(matches!(
            decode_record(expected, &tagged),
            Err(CollectionDeltaError::Decode(
                RecordDecodeError::NonCanonicalMergeInputs
            ))
        ));
    }

    #[test]
    fn selection_is_actual_sorted_deduplicated_set_difference() {
        let expected = collection(1);
        let [commit, merge, derive] = records(expected);
        let previous = patch(expected, [commit]);
        let current = patch(expected, [derive, commit, merge, merge]);
        let selected = select_bounded_delta(&previous, &current, 2).unwrap();
        let CollectionDeltaSelection::Push(selected) = selected else {
            panic!("two records fit the push bound")
        };
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&merge));
        assert!(selected.contains(&derive));
        assert!(selected.windows(2).all(|pair| pair[0].id() < pair[1].id()));
    }

    #[test]
    fn large_gap_returns_only_a_repair_decision() {
        let expected = collection(1);
        let previous = patch(expected, std::iter::empty());
        let current = patch(expected, records(expected));
        assert_eq!(
            select_bounded_delta(&previous, &current, 2).unwrap(),
            CollectionDeltaSelection::Repair { missing: 3 }
        );
    }

    #[test]
    fn old_snapshot_must_be_a_subset_of_current_truth() {
        let expected = collection(1);
        let [commit, merge, _] = records(expected);
        let previous = patch(expected, [commit, merge]);
        let current = patch(expected, [commit]);
        assert_eq!(
            select_bounded_delta(&previous, &current, 8),
            Err(CollectionDeltaError::NonMonotoneSnapshot)
        );
    }

    #[test]
    fn sparse_unsigned_equations_need_no_referenced_blobs_or_write_principal() {
        let expected = collection(1);
        let [_, merge, derive] = records(expected);
        let previous = patch(expected, std::iter::empty());
        let current = patch(expected, [derive, merge]);
        let selected = select_bounded_delta(&previous, &current, 2).unwrap();
        assert!(matches!(
            selected,
            CollectionDeltaSelection::Push(records) if records.len() == 2
        ));
    }

    #[test]
    fn relay_roundtrip_preserves_the_embedded_commit_author() {
        let expected = collection(1);
        let commit = records(expected)[0];
        let CollectionRecord::Commit(before) = commit else {
            unreachable!()
        };
        let after = decode_record(expected, &encode_record(expected, commit).unwrap()).unwrap();
        let CollectionRecord::Commit(after) = after else {
            unreachable!()
        };
        assert_eq!(after.public_key(), before.public_key());
        assert_eq!(after.id(), before.id());
    }

    struct ExactSelectorStore {
        expected: CollectionHandle,
        selected: Vec<CollectionRecord>,
        global_enumerations: Cell<usize>,
        selections: Cell<usize>,
    }

    impl CollectionRead for ExactSelectorStore {
        type RecordsError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.global_enumerations
                .set(self.global_enumerations.get() + 1);
            Ok(Vec::new().into_iter())
        }

        fn select_records(
            &self,
            selectors: &BTreeSet<CollectionRecordSelector>,
        ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
            assert_eq!(
                selectors,
                &BTreeSet::from([CollectionRecordSelector::Collection(self.expected)])
            );
            self.selections.set(self.selections.get() + 1);
            Ok(self.selected.clone())
        }
    }

    #[test]
    fn overlay_patch_uses_only_the_exact_collection_selector() {
        let expected = collection(1);
        let selected = records(expected).to_vec();
        let store = ExactSelectorStore {
            expected,
            selected: selected.clone(),
            global_enumerations: Cell::new(0),
            selections: Cell::new(0),
        };

        let overlay = collection_record_patch(&store, expected).unwrap();

        assert_eq!(overlay.len(), selected.len() as u64);
        assert!(
            selected
                .iter()
                .all(|record| { overlay.get(record.id()) == Some(*record) })
        );
        assert_eq!(store.selections.get(), 1);
        assert_eq!(store.global_enumerations.get(), 0);
    }

    #[test]
    fn delta_rejects_patches_for_different_collections() {
        let left_collection = collection(1);
        let right_collection = collection(2);
        let left = patch(left_collection, records(left_collection));
        let right = patch(right_collection, records(right_collection));

        assert_eq!(
            select_bounded_delta(&left, &right, usize::MAX),
            Err(CollectionDeltaError::MismatchedCollections)
        );
    }
}
