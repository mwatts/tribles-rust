//! Value schemas minted for triblespace-search.
//!
//! For now the only schema is [`F32LE`], used by the eventual
//! score-as-bound-variable BM25/HNSW constraints. Blob schemas
//! live next to their types (`bm25::SuccinctBM25Index`,
//! `hnsw::SuccinctHNSWIndex`).

use std::convert::Infallible;

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
}
