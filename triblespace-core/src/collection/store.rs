//! Native storage for collection-calculus records.
//!
//! A collection store is a grow-only set keyed by each record's intrinsic id.
//! It deliberately exposes no mutable head, deletion, compare-and-swap, or
//! point-in-time snapshot contract. Backends may discover additional records
//! between calls; each individual enumeration is required only to be
//! deterministic for the records it returns.

use std::error::Error;
use std::fmt::Debug;

use super::CollectionRecord;

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

    /// Insert one canonical record.
    ///
    /// Re-inserting a record with the same intrinsic id is success and does not
    /// add another logical set member.
    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError>;
}
