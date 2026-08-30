//! Policy-independent collection-record delta mechanics.
//!
//! This module owns only the immutable evidence boundary: strict framing,
//! intrinsic collection matching, COMMIT signature verification, canonical
//! id ordering, and bounded `current - previous` selection. It deliberately
//! does not resolve referenced blobs or decide READ/WRITE policy. A future
//! authorized overlay can therefore store sparse MERGE/DERIVE equations as
//! inert evidence and apply semantic validation only when a resolver uses one.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use triblespace_core::collection::{
    CollectionHandle, CollectionRecord, CommitVerificationError, RecordDecodeError,
};
use triblespace_core::id::Id;

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
    IntrinsicIdCollision(Id),
    NonMonotoneSnapshot,
}

impl fmt::Display for CollectionDeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(error) => write!(f, "decode collection record: {error}"),
            Self::InvalidCommit(error) => write!(f, "verify collection COMMIT: {error}"),
            Self::WrongCollection => write!(f, "record names another collection"),
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
    expected: CollectionHandle,
    previous: impl IntoIterator<Item = CollectionRecord>,
    current: impl IntoIterator<Item = CollectionRecord>,
    max_push_records: usize,
) -> Result<CollectionDeltaSelection, CollectionDeltaError> {
    let previous = canonical_records(expected, previous)?;
    let current = canonical_records(expected, current)?;
    if previous
        .iter()
        .any(|(id, record)| current.get(id) != Some(record))
    {
        return Err(CollectionDeltaError::NonMonotoneSnapshot);
    }

    let missing = current.len() - previous.len();
    if missing > max_push_records {
        return Ok(CollectionDeltaSelection::Repair { missing });
    }
    Ok(CollectionDeltaSelection::Push(
        current
            .into_iter()
            .filter_map(|(id, record)| (!previous.contains_key(&id)).then_some(record))
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
) -> Result<BTreeMap<Id, CollectionRecord>, CollectionDeltaError> {
    let mut canonical = BTreeMap::new();
    for record in records {
        validate_record(expected, record)?;
        let id = record.id();
        if let Some(existing) = canonical.insert(id, record)
            && existing != record
        {
            return Err(CollectionDeltaError::IntrinsicIdCollision(id));
        }
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
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
        let selected =
            select_bounded_delta(expected, [commit], [derive, commit, merge, merge], 2).unwrap();
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
        assert_eq!(
            select_bounded_delta(expected, [], records(expected), 2).unwrap(),
            CollectionDeltaSelection::Repair { missing: 3 }
        );
    }

    #[test]
    fn old_snapshot_must_be_a_subset_of_current_truth() {
        let expected = collection(1);
        let [commit, merge, _] = records(expected);
        assert_eq!(
            select_bounded_delta(expected, [commit, merge], [commit], 8),
            Err(CollectionDeltaError::NonMonotoneSnapshot)
        );
    }

    #[test]
    fn sparse_unsigned_equations_need_no_referenced_blobs_or_write_principal() {
        let expected = collection(1);
        let [_, merge, derive] = records(expected);
        let selected = select_bounded_delta(expected, [], [derive, merge], 2).unwrap();
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
}
