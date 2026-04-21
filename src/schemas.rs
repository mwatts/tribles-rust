//! Value and blob schemas minted for triblespace-search.
//!
//! - [`F32LE`] (value schema): packs an f32 into a 32-byte
//!   triblespace value, used by score-as-bound-variable
//!   constraints.
//! - [`Embedding`] (blob schema): an arbitrary-length `[f32]`
//!   (little-endian) stored as a blob. HNSW indexes no longer
//!   inline vectors — they store `Handle<Embedding>` instead,
//!   so embeddings are content-addressed and dedupe across
//!   indexes.
//!
//! Other blob schemas (`SuccinctBM25Blob`, `SuccinctHNSWBlob`)
//! live next to their index types.

use std::convert::Infallible;

use anybytes::View;
use triblespace::core::blob::{Blob, BlobSchema, ToBlob, TryFromBlob};
use triblespace::core::id::Id;
use triblespace::core::id_hex;
use triblespace::core::metadata::ConstId;
use triblespace::core::value::{ToValue, TryFromValue, Value, ValueSchema};

/// 32-bit IEEE-754 little-endian float packed into a 32-byte
/// triblespace `Value`. Bytes `[0..4]` hold the raw f32 bytes;
/// bytes `[4..32]` are zero-padded.
///
/// Schema id was minted via `trible genid` and is fixed:
/// `816B4751EA8C12644CCB572F36188EBA`.
///
/// Every bit pattern decodes to some f32 (including NaN +
/// signed zero), so validation is infallible. Callers that want
/// stricter invariants (non-NaN, within a specific range)
/// should wrap `Value<F32LE>` with their own newtype + checked
/// conversion.
pub enum F32LE {}

impl ConstId for F32LE {
    const ID: Id = id_hex!("816B4751EA8C12644CCB572F36188EBA");
}

impl ValueSchema for F32LE {
    type ValidationError = Infallible;
}

impl ToValue<F32LE> for f32 {
    fn to_value(self) -> Value<F32LE> {
        let mut raw = [0u8; 32];
        raw[0..4].copy_from_slice(&self.to_le_bytes());
        Value::new(raw)
    }
}

impl ToValue<F32LE> for &f32 {
    fn to_value(self) -> Value<F32LE> {
        (*self).to_value()
    }
}

impl TryFromValue<'_, F32LE> for f32 {
    type Error = Infallible;

    fn try_from_value(value: &Value<F32LE>) -> Result<Self, Self::Error> {
        Ok(f32::from_le_bytes(value.raw[0..4].try_into().unwrap()))
    }
}

/// An arbitrary-length `[f32]` (little-endian) stored as a blob.
///
/// HNSW indexes reference embeddings by
/// [`Handle<Blake3, Embedding>`][h] so two indexes that embed
/// the same entity share one on-disk blob. A blob is just the
/// raw f32 LE bytes, length = `dim × 4`. The dim isn't
/// recorded in the blob header — the HNSW index that owns the
/// handle carries it (one `dim` per index).
///
/// Schema id minted via `trible genid`:
/// `EEC5DFDEA2FFCED70850DF83B03CB62B`.
///
/// [h]: triblespace::core::value::schemas::hash::Handle
pub struct Embedding {}

impl ConstId for Embedding {
    const ID: Id = id_hex!("EEC5DFDEA2FFCED70850DF83B03CB62B");
}

impl BlobSchema for Embedding {}

/// Decode a blob back into a zero-copy `View<[f32]>`. Fails
/// iff the blob's byte length isn't a multiple of 4 (malformed)
/// or the backing buffer can't be aligned to `f32`.
impl TryFromBlob<Embedding> for View<[f32]> {
    type Error = anybytes::view::ViewError;

    fn try_from_blob(b: Blob<Embedding>) -> Result<Self, Self::Error> {
        b.bytes.view()
    }
}

impl ToBlob<Embedding> for View<[f32]> {
    fn to_blob(self) -> Blob<Embedding> {
        Blob::new(self.bytes())
    }
}

impl ToBlob<Embedding> for Vec<f32> {
    fn to_blob(self) -> Blob<Embedding> {
        // f32 is `IntoBytes` (zerocopy) so this is a straight
        // byte-copy of the `Vec`'s backing storage.
        let mut bytes = Vec::with_capacity(self.len() * 4);
        for v in &self {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Blob::new(bytes.into())
    }
}

impl ToBlob<Embedding> for &[f32] {
    fn to_blob(self) -> Blob<Embedding> {
        let mut bytes = Vec::with_capacity(self.len() * 4);
        for v in self {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Blob::new(bytes.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_positive() {
        let original: f32 = 0.123;
        let v: Value<F32LE> = original.to_value();
        let back: f32 = f32::try_from_value(&v).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn round_trip_negative() {
        let original: f32 = -42.75;
        let v: Value<F32LE> = original.to_value();
        let back: f32 = f32::try_from_value(&v).unwrap();
        assert_eq!(original, back);
    }

    #[test]
    fn round_trip_zero() {
        let original: f32 = 0.0;
        let v: Value<F32LE> = original.to_value();
        let back: f32 = f32::try_from_value(&v).unwrap();
        assert_eq!(original.to_bits(), back.to_bits());
    }

    #[test]
    fn round_trip_nan() {
        let original: f32 = f32::NAN;
        let v: Value<F32LE> = original.to_value();
        let back: f32 = f32::try_from_value(&v).unwrap();
        assert!(back.is_nan());
    }

    #[test]
    fn padding_is_zero() {
        let v: Value<F32LE> = 3.14f32.to_value();
        assert_eq!(&v.raw[4..32], &[0u8; 28]);
    }

    #[test]
    fn deterministic_same_input_same_value() {
        let a: Value<F32LE> = 1.5f32.to_value();
        let b: Value<F32LE> = 1.5f32.to_value();
        assert_eq!(a.raw, b.raw);
    }

    #[test]
    fn embedding_blob_round_trip() {
        let original: Vec<f32> = vec![0.1, -0.5, 3.25, f32::consts::PI];
        let blob: Blob<Embedding> = original.clone().to_blob();
        let view: View<[f32]> = TryFromBlob::try_from_blob(blob).unwrap();
        assert_eq!(view.as_ref(), original.as_slice());
    }

    #[test]
    fn embedding_handle_is_content_addressed() {
        use triblespace::core::value::schemas::hash::{Blake3, Handle};

        let v1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v2: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v3: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let h1: Value<Handle<Blake3, Embedding>> = v1.to_blob().get_handle();
        let h2: Value<Handle<Blake3, Embedding>> = v2.to_blob().get_handle();
        let h3: Value<Handle<Blake3, Embedding>> = v3.to_blob().get_handle();

        assert_eq!(h1, h2, "identical vectors must dedup by handle");
        assert_ne!(h1, h3, "different vectors must have different handles");
    }

    use std::f32;
}
