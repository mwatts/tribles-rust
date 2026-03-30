//! Re-exports of blob schema types for convenient glob imports.

/// Re-export of [`Array`](crate::blob::schemas::array::Array) and [`ArrayElement`](crate::blob::schemas::array::ArrayElement).
pub use crate::blob::schemas::array::{Array, ArrayElement};
/// Re-export of built-in element types (`F32`, `U8`, etc.).
pub use crate::blob::schemas::array::elements;
/// Re-export of [`FileBytes`](crate::blob::schemas::filebytes::FileBytes).
pub use crate::blob::schemas::filebytes::FileBytes;
/// Re-export of [`LongString`](crate::blob::schemas::longstring::LongString).
pub use crate::blob::schemas::longstring::LongString;
/// Re-export of [`SimpleArchive`](crate::blob::schemas::simplearchive::SimpleArchive).
pub use crate::blob::schemas::simplearchive::SimpleArchive;

/// Re-export of [`SuccinctArchive`](crate::blob::schemas::succinctarchive::SuccinctArchive) and [`SuccinctArchiveBlob`](crate::blob::schemas::succinctarchive::SuccinctArchiveBlob).
pub use crate::blob::schemas::succinctarchive::{SuccinctArchive, SuccinctArchiveBlob};
/// Re-export of [`UnknownBlob`](crate::blob::schemas::UnknownBlob).
pub use crate::blob::schemas::UnknownBlob;
/// Re-export of [`WasmCode`](crate::blob::schemas::wasmcode::WasmCode).
pub use crate::blob::schemas::wasmcode::WasmCode;
