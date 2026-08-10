//! Local named operational values.
//!
//! Cells are intentionally independent of repository branches, collection
//! authority, and blob wants. They are a small storage primitive for local
//! replaceable policy state, not a distributed coordination protocol.

use std::error::Error;
use std::fmt::Debug;
use std::future::Future;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;

/// Storage backend for local named replaceable values.
///
/// A cell is a store-local policy slot, not a branch, history head, or
/// distributed coordination primitive. Writes replace the current value with
/// last-writer-wins semantics and carry no compare-and-swap guard. Cells are
/// never gossipable through this interface.
///
/// Only the replacement itself is atomic. A read-modify-write sequence over a
/// value containing several logical records is not: concurrent writers may
/// overwrite one another according to the backend's local write order. Route
/// such mutations through one writer per cell, split independent state across
/// cells, or deliberately accept whole-value last-writer-wins behavior.
///
/// Cell values are `SimpleArchive` handles so higher layers can keep typed
/// local state in ordinary queryable tribles. Storage backends with garbage
/// collection must treat every current cell value as a recursive local root;
/// that ownership is operational retention only and grants no collection
/// authority.
pub trait LocalCellStore {
    /// Failure while reading or replacing a cell.
    type CellError: Error + Debug + Send + Sync + 'static;

    /// Read the current value of `id`, or `None` when the cell is absent.
    fn cell(&mut self, id: Id) -> Result<Option<Inline<Handle<SimpleArchive>>>, Self::CellError>;

    /// Replace the current value of `id`.
    ///
    /// Passing `None` clears the cell. Repeating the current state is an
    /// idempotent success. There is intentionally no expected-old parameter:
    /// callers already hold `&mut self`, and cells are local policy rather than
    /// a distributed linearization surface.
    fn set_cell(
        &mut self,
        id: Id,
        value: Option<Inline<Handle<SimpleArchive>>>,
    ) -> Result<(), Self::CellError>;
}

/// Async counterpart of [`LocalCellStore`].
pub trait AsyncLocalCellStore {
    /// Failure while reading or replacing a cell.
    type CellError: Error + Debug + Send + Sync + 'static;

    /// Read one local cell.
    fn cell(
        &mut self,
        id: Id,
    ) -> impl Future<Output = Result<Option<Inline<Handle<SimpleArchive>>>, Self::CellError>> + Send;

    /// Replace or clear one local cell without distributed CAS semantics.
    fn set_cell(
        &mut self,
        id: Id,
        value: Option<Inline<Handle<SimpleArchive>>>,
    ) -> impl Future<Output = Result<(), Self::CellError>> + Send;
}
