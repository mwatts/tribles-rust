//! Flat typed array blob schema.
//!
//! `Array<T>` is a structural blob schema: it says "this blob is a flat array
//! of T values in native byte order." The semantics (weight tensor, audio
//! samples, embeddings) come from the TribleSpace attributes that reference
//! the blob, not the schema itself — same as `LongString` being structural
//! rather than semantic.

use core::marker::PhantomData;

use anybytes::view::ViewError;
use anybytes::{Bytes, View};
use zerocopy::{Immutable, IntoBytes, KnownLayout, TryFromBytes};

use crate::blob::{Blob, BlobSchema, ToBlob, TryFromBlob};
use crate::id::Id;
use crate::macros::entity;
use crate::metadata;
use crate::metadata::{ConstDescribe, ConstId};
use crate::repo::BlobStore;
use crate::trible::Fragment;
use crate::value::schemas::hash::Blake3;

/// Maps a schema element marker to its native Rust type.
///
/// Implement this for zero-sized marker types (e.g. `F32`, `BF16`, `I8`)
/// that identify an element format. The `ConstId` provides the schema
/// identity; `Native` provides the actual data type for zerocopy access.
pub trait ArrayElement: ConstId + 'static {
    /// The native Rust type for this element.
    type Native: IntoBytes + Immutable + TryFromBytes + KnownLayout + Sync + Send + 'static;
}

/// A flat array of `T` values in native byte order.
///
/// The blob schema ID is derived at compile time from T's ConstId via
/// const_blake3, following the same pattern as `Handle<H, S>`.
///
/// Shape metadata lives in TribleSpace triples, not in the blob.
/// Use `View<[T::Native]>` for zero-copy access via `TryFromBlob`.
pub struct Array<T: ArrayElement>(PhantomData<T>);

impl<T: ArrayElement> BlobSchema for Array<T> {}

impl<T: ArrayElement> ConstId for Array<T> {
    const ID: Id = {
        let mut hasher = const_blake3::Hasher::new();
        hasher.update(b"array:");
        hasher.update(&T::ID.raw());
        let mut digest = [0u8; 32];
        hasher.finalize(&mut digest);
        let mut raw = [0u8; 16];
        let mut i = 0;
        while i < raw.len() {
            raw[i] = digest[16 + i];
            i += 1;
        }
        match Id::new(raw) {
            Some(id) => id,
            None => panic!("derived array schema id must be non-nil"),
        }
    };
}

impl<T: ArrayElement> ConstDescribe for Array<T> {
    fn describe<B>(blobs: &mut B) -> Result<Fragment, B::PutError>
    where
        B: BlobStore<Blake3>,
    {
        let id = Self::ID;
        Ok(entity! {
            crate::id::ExclusiveId::force_ref(&id) @
                metadata::name: blobs.put("array")?,
                metadata::description: blobs.put(
                    "Flat array of typed values in native byte order. \
                     Shape is stored externally in TribleSpace triples.",
                )?,
                metadata::tag: metadata::KIND_BLOB_SCHEMA,
        })
    }
}

/// Store a `Vec<T::Native>` as an `Array<T>` blob (zero-copy via ByteSource).
impl<T: ArrayElement> ToBlob<Array<T>> for Vec<T::Native> {
    fn to_blob(self) -> Blob<Array<T>> {
        Blob::new(Bytes::from_source(self))
    }
}

/// Retrieve raw bytes from an Array blob.
impl<T: ArrayElement> TryFromBlob<Array<T>> for Bytes {
    type Error = core::convert::Infallible;
    fn try_from_blob(blob: Blob<Array<T>>) -> Result<Self, Self::Error> {
        Ok(blob.bytes)
    }
}

/// Built-in element types for common native Rust types.
///
/// Access as `blobschemas::array::F32`, `blobschemas::array::U8`, etc.
pub mod elements {
    use super::ArrayElement;
    use crate::id::Id;
    use crate::metadata::{ConstDescribe, ConstId};

    macro_rules! impl_array_element {
        ($marker:ident, $native:ty, $id:expr, $doc:expr) => {
            #[doc = $doc]
            pub struct $marker;

            impl ConstId for $marker {
                const ID: Id = crate::id_hex!($id);
            }

            impl ConstDescribe for $marker {}

            impl ArrayElement for $marker {
                type Native = $native;
            }
        };
    }

    impl_array_element!(
        F32,
        f32,
        "92F4DB8D84519C8D6E212CB810FF40D4",
        "32-bit IEEE-754 float."
    );
    impl_array_element!(
        F64,
        f64,
        "FA3AD8DEC844D5F409AB728269B7A3FE",
        "64-bit IEEE-754 float."
    );
    impl_array_element!(
        U8,
        u8,
        "D16AC7C02F25E4799F4D47EB1E51EF6E",
        "Unsigned 8-bit integer."
    );
    impl_array_element!(
        U16,
        u16,
        "C14453D98F283B96A1010A9F24D53B17",
        "Unsigned 16-bit integer."
    );
    impl_array_element!(
        U32,
        u32,
        "1B9DD214A02C58D9141EF802273120F8",
        "Unsigned 32-bit integer."
    );
    impl_array_element!(
        U64,
        u64,
        "323C0143534D3AD4898D69EA5597414A",
        "Unsigned 64-bit integer."
    );
    impl_array_element!(
        I8,
        i8,
        "E68060AF27227583CB1AEDF89E17E278",
        "Signed 8-bit integer."
    );
    impl_array_element!(
        I16,
        i16,
        "E72199687209A576562B5BD7196FD755",
        "Signed 16-bit integer."
    );
    impl_array_element!(
        I32,
        i32,
        "AB831A6CCDAF7F49BA5BEADEA32CA04E",
        "Signed 32-bit integer."
    );
    impl_array_element!(
        I64,
        i64,
        "53426475A3C695420B23C329285DCA57",
        "Signed 64-bit integer."
    );
}

/// Zero-copy typed view directly from an Array blob.
///
/// ```ignore
/// let floats: View<[f32]> = blobs.get(handle)?;
/// ```
impl<T: ArrayElement> TryFromBlob<Array<T>> for View<[T::Native]> {
    type Error = ViewError;
    fn try_from_blob(blob: Blob<Array<T>>) -> Result<Self, Self::Error> {
        blob.bytes.view()
    }
}
