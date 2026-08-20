//! Native storage for collection-calculus records.
//!
//! A collection store is a grow-only set keyed by each record's intrinsic id.
//! It deliberately exposes no mutable head, deletion, compare-and-swap, or
//! point-in-time snapshot contract. Backends may discover additional records
//! between calls; each individual enumeration is required only to be
//! deterministic for the records it returns.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Debug;

use crate::id::Id;
use crate::repo::WantRequest;

use super::{CollectionHandle, CollectionRecord};

/// One semantic route into the grow-only collection-record set.
///
/// A batch of selectors is interpreted as set union. Exact operation lookup
/// deliberately names only the inputs: every distinct asserted result remains
/// visible to callers as conflicting evidence.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CollectionRecordSelector {
    /// Select one record by its intrinsic content-derived id.
    Id(Id),
    /// Select every `MERGE` asserted for one collection descriptor.
    MergeCollection(CollectionHandle),
    /// Select every `DERIVE` into one exact target descriptor.
    ///
    /// The source is not part of the selector: a target has one source, named
    /// by its descriptor, so selecting the target selects the mapping.
    DeriveTarget(CollectionHandle),
    /// Select every receipt answering one exact merge or derive request.
    ///
    /// `WantRequest::Blob` has no collection-record answer and selects
    /// nothing.
    Operation(WantRequest),
}

fn collection_record_operation(record: CollectionRecord) -> Option<WantRequest> {
    match record {
        CollectionRecord::Commit(_) => None,
        CollectionRecord::Merge(record) => {
            let (low, high) = record.inputs();
            Some(WantRequest::merge(record.collection(), low, high))
        }
        CollectionRecord::Derive(record) => {
            let (input, _) = record.mapping();
            Some(WantRequest::derive(record.target(), input))
        }
    }
}

pub(crate) fn selectors_match_record(
    selectors: &BTreeSet<CollectionRecordSelector>,
    record: CollectionRecord,
) -> bool {
    if selectors.contains(&CollectionRecordSelector::Id(record.id())) {
        return true;
    }
    match record {
        CollectionRecord::Commit(_) => false,
        CollectionRecord::Merge(merge) => {
            selectors.contains(&CollectionRecordSelector::MergeCollection(
                merge.collection(),
            )) || selectors.contains(&CollectionRecordSelector::Operation(
                collection_record_operation(record).expect("MERGE has an operation key"),
            ))
        }
        CollectionRecord::Derive(derive) => {
            selectors.contains(&CollectionRecordSelector::DeriveTarget(derive.target())) || selectors.contains(&CollectionRecordSelector::Operation(
                collection_record_operation(record).expect("DERIVE has an operation key"),
            ))
        }
    }
}

/// Storage surface for canonical collection-calculus records.
///
/// Inserting the same intrinsic record id more than once is an idempotent
/// success. Records are never replaced through this interface. Implementations
/// enumerate their currently known records in deterministic intrinsic-id
/// order, without promising that the enumeration is a globally coherent
/// snapshot of a concurrently changing or distributed backend.
pub trait CollectionStore {
    /// Failure while enumerating stored records.
    type RecordsError: Error + Debug + Send + Sync + 'static;
    /// Failure while admitting one canonical record.
    type InsertError: Error + Debug + Send + Sync + 'static;

    /// Borrowing iterator over one deterministic view of known records.
    type RecordIter<'a>: Iterator<Item = Result<CollectionRecord, Self::RecordsError>>
    where
        Self: 'a;

    /// Enumerate currently known records in deterministic intrinsic-id order.
    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError>;

    /// Select one deterministic union of semantic record routes.
    ///
    /// The default implementation performs exactly one ordinary enumeration
    /// and filters it. Backends with primary or secondary indexes may override
    /// this method without changing the grow-only set contract. Returned
    /// records remain deduplicated and sorted by intrinsic id. An empty union
    /// returns immediately without asking the backend for a view.
    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        if selectors.is_empty() {
            return Ok(Vec::new());
        }
        let records = self.records()?;
        let mut selected = Vec::new();
        for record in records {
            let record = record?;
            if selectors_match_record(selectors, record) {
                selected.push(record);
            }
        }
        Ok(selected)
    }

    /// Insert one canonical record.
    ///
    /// Re-inserting a record with the same intrinsic id is success and does not
    /// add another logical set member.
    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError>;
}

impl<S> CollectionStore for &mut S
where
    S: CollectionStore + ?Sized,
{
    type RecordsError = S::RecordsError;
    type InsertError = S::InsertError;
    type RecordIter<'a>
        = S::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        (**self).records()
    }

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        (**self).select_records(selectors)
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        (**self).insert(record)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::collection::{
        empty_metadata_handle, CollectionCommit, CollectionData, CollectionDerive, CollectionMerge,
    };
    use crate::inline::Inline;

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn fixture() -> Vec<CollectionRecord> {
        let source = collection(1);
        let target = collection(2);
        let other = collection(3);
        let input = data(10);
        let mut records = vec![
            CollectionRecord::Commit(CollectionCommit::sign(
                &SigningKey::from_bytes(&[7; 32]),
                source,
                data(4),
                empty_metadata_handle(),
            )),
            CollectionRecord::Merge(CollectionMerge::new(source, data(4), data(5), data(6))),
            CollectionRecord::Merge(CollectionMerge::new(other, data(4), data(5), data(7))),
            CollectionRecord::Derive(CollectionDerive::new(target, input, data(11))),
            CollectionRecord::Derive(CollectionDerive::new(target, input, data(12))),
            CollectionRecord::Derive(CollectionDerive::new(target, data(13), data(14))),
            CollectionRecord::Derive(CollectionDerive::new(other, input, data(15))),
        ];
        records.sort_unstable_by_key(CollectionRecord::id);
        records
    }

    #[derive(Default)]
    struct FallbackStore {
        records: Vec<CollectionRecord>,
        enumerations: usize,
    }

    impl CollectionStore for FallbackStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.enumerations += 1;
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

    #[test]
    fn default_selection_scans_once_unions_routes_and_retains_operation_conflicts() {
        let records = fixture();
        let commit = records
            .iter()
            .find(|record| matches!(record, CollectionRecord::Commit(_)))
            .copied()
            .unwrap();
        let source = collection(1);
        let target = collection(2);
        let exact_derive = WantRequest::derive(target, data(10));
        let overlapping_id = records
            .iter()
            .find(|record| match record {
                CollectionRecord::Derive(derive) => {
                    derive.target() == target && derive.mapping().0 == data(10)
                }
                _ => false,
            })
            .unwrap()
            .id();
        let selectors = [
            CollectionRecordSelector::Id(commit.id()),
            CollectionRecordSelector::Id(overlapping_id),
            CollectionRecordSelector::MergeCollection(source),
            CollectionRecordSelector::Operation(exact_derive),
        ]
        .into_iter()
        .collect();
        let mut store = FallbackStore {
            records: records.clone(),
            ..FallbackStore::default()
        };

        let selected = store.select_records(&selectors).unwrap();

        let mut expected: Vec<_> = records
            .into_iter()
            .filter(|record| match record {
                CollectionRecord::Commit(_) => record.id() == commit.id(),
                CollectionRecord::Merge(merge) => merge.collection() == source,
                CollectionRecord::Derive(derive) => {
                    derive.target() == target && derive.mapping().0 == data(10)
                }
            })
            .collect();
        expected.sort_unstable_by_key(CollectionRecord::id);
        assert_eq!(selected, expected);
        assert_eq!(store.enumerations, 1);
        assert_eq!(
            selected
                .iter()
                .filter(|record| matches!(record, CollectionRecord::Derive(_)))
                .count(),
            2,
            "different outputs for one exact DERIVE remain visible"
        );
    }

    #[test]
    fn default_pair_selection_includes_all_inputs_and_excludes_other_pairs() {
        let records = fixture();
        let target = collection(2);
        let pair = [CollectionRecordSelector::DeriveTarget(target)]
            .into_iter()
            .collect();
        let mut store = FallbackStore {
            records,
            ..FallbackStore::default()
        };

        let selected = store.select_records(&pair).unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|record| match record {
            CollectionRecord::Derive(derive) => derive.target() == target,
            _ => false,
        }));
    }

    #[test]
    fn empty_selection_returns_without_enumeration() {
        let mut store = FallbackStore {
            records: fixture(),
            ..FallbackStore::default()
        };

        assert!(store.select_records(&BTreeSet::new()).unwrap().is_empty());
        assert_eq!(store.enumerations, 0);
    }

    #[derive(Default)]
    struct OverrideStore {
        records_calls: usize,
        selection_calls: usize,
    }

    impl CollectionStore for OverrideStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.records_calls += 1;
            Ok(Vec::new().into_iter())
        }

        fn select_records(
            &mut self,
            _selectors: &BTreeSet<CollectionRecordSelector>,
        ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
            self.selection_calls += 1;
            Ok(Vec::new())
        }

        fn insert(&mut self, _record: CollectionRecord) -> Result<(), Self::InsertError> {
            Ok(())
        }
    }

    #[test]
    fn mutable_reference_forwards_selection_override() {
        let mut store = OverrideStore::default();
        let mut borrowed = &mut store;
        let selectors = [CollectionRecordSelector::Id(Id::new([1; 16]).unwrap())]
            .into_iter()
            .collect();
        CollectionStore::select_records(&mut borrowed, &selectors).unwrap();
        assert_eq!(store.selection_calls, 1);
        assert_eq!(store.records_calls, 0);
    }
}
