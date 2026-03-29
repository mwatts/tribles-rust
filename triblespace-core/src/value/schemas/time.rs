use crate::id::ExclusiveId;
use crate::id::Id;
use crate::id_hex;
use crate::macros::entity;
use crate::metadata;
use crate::metadata::{ConstDescribe, ConstId};
use crate::repo::BlobStore;
use crate::trible::Fragment;
use crate::value::schemas::hash::Blake3;
use crate::value::TryFromValue;
use crate::value::TryToValue;
use crate::value::Value;
use crate::value::ValueSchema;
use std::convert::Infallible;

use std::convert::TryInto;

use hifitime::prelude::*;

/// A value schema for a TAI interval (little-endian, legacy).
///
/// A TAI interval is a pair of TAI epochs stored as two 128-bit signed
/// integers (lower, upper) in **little-endian** byte order. Both bounds
/// are inclusive.
///
/// Little-endian encoding means byte-order ≠ time-order, so this schema
/// is not suitable for range scans on the trie. For ordered queries, use
/// [`OrderedNsTAIInterval`] instead.
pub struct NsTAIInterval;

/// A value schema for a TAI interval (big-endian, sortable).
///
/// Same semantics as [`NsTAIInterval`] but stored in **order-preserving
/// big-endian**: each i128 is XOR'd with the sign bit (mapping i128::MIN→0,
/// 0→2^127, i128::MAX→u128::MAX) then written big-endian. Byte-lexicographic
/// order matches numeric order across the full i128 range, enabling efficient
/// range scans on the trie.
pub struct OrderedNsTAIInterval;

impl ConstId for NsTAIInterval {
    const ID: Id = id_hex!("675A2E885B12FCBC0EEC01E6AEDD8AA8");
}

impl ConstDescribe for NsTAIInterval {
    fn describe<B>(blobs: &mut B) -> Result<Fragment, B::PutError>
    where
        B: BlobStore<Blake3>,
    {
        let id = Self::ID;
        let description = blobs.put(
            "Inclusive TAI interval encoded as two little-endian i128 nanosecond bounds. TAI is monotonic and does not include leap seconds, making it ideal for precise ordering.\n\nUse for time windows, scheduling, or event ranges where monotonic time matters. If you need civil time, time zones, or calendar semantics, store a separate representation alongside this interval.\n\nIntervals are inclusive on both ends. If you need half-open intervals or offsets, consider RangeU128 with your own epoch mapping.",
        )?;
        let tribles = entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: blobs.put("nstai_interval")?,
                metadata::description: description,
                metadata::tag: metadata::KIND_VALUE_SCHEMA,
        };

        #[cfg(feature = "wasm")]
        let tribles = {
            let mut tribles = tribles;
            tribles += entity! { ExclusiveId::force_ref(&id) @
                metadata::value_formatter: blobs.put(wasm_formatter::NSTAI_INTERVAL_WASM)?,
            };
            tribles
        };
        Ok(tribles)
    }
}

#[cfg(feature = "wasm")]
mod wasm_formatter {
    use core::fmt::Write;

    use triblespace_core_macros::value_formatter;

    #[value_formatter]
    pub(crate) fn nstai_interval(raw: &[u8; 32], out: &mut impl Write) -> Result<(), u32> {
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&raw[0..16]);
        let lower = i128::from_le_bytes(buf);
        buf.copy_from_slice(&raw[16..32]);
        let upper = i128::from_le_bytes(buf);

        write!(out, "{lower}..={upper}").map_err(|_| 1u32)?;
        Ok(())
    }
}

impl ConstId for OrderedNsTAIInterval {
    const ID: Id = id_hex!("2170014368272A2B1B18B86B1F1F1CB5");
}

impl ConstDescribe for OrderedNsTAIInterval {
    fn describe<B>(blobs: &mut B) -> Result<Fragment, B::PutError>
    where
        B: BlobStore<Blake3>,
    {
        let id = Self::ID;
        let description = blobs.put(
            "Inclusive TAI interval encoded as two offset-big-endian i128 nanosecond bounds. Each i128 is XOR'd with i128::MIN then stored big-endian, so byte-lexicographic order matches numeric order. This enables efficient range scans on ordered indexes.\n\nSemantically identical to NsTAIInterval — same inclusive bounds, same TAI monotonic time. Prefer this schema when range queries or sorted iteration matter.",
        )?;
        Ok(entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: blobs.put("nstai_interval_be")?,
                metadata::description: description,
                metadata::tag: metadata::KIND_VALUE_SCHEMA,
        })
    }
}

const SIGN_BIT: u128 = 1u128 << 127;

/// Encode i128 as order-preserving big-endian: flip sign bit, then BE.
/// Maps i128::MIN→0, 0→2^127, i128::MAX→u128::MAX.
fn i128_to_ordered_be(v: i128) -> [u8; 16] {
    ((v as u128) ^ SIGN_BIT).to_be_bytes()
}

/// Decode order-preserving big-endian back to i128.
fn i128_from_ordered_be(bytes: [u8; 16]) -> i128 {
    (u128::from_be_bytes(bytes) ^ SIGN_BIT) as i128
}

impl ValueSchema for OrderedNsTAIInterval {
    type ValidationError = InvertedIntervalError;

    fn validate(value: Value<Self>) -> Result<Value<Self>, Self::ValidationError> {
        let lower = i128_from_ordered_be(value.raw[0..16].try_into().unwrap());
        let upper = i128_from_ordered_be(value.raw[16..32].try_into().unwrap());
        if lower > upper {
            Err(InvertedIntervalError { lower, upper })
        } else {
            Ok(value)
        }
    }
}

impl TryToValue<OrderedNsTAIInterval> for (Epoch, Epoch) {
    type Error = InvertedIntervalError;
    fn try_to_value(self) -> Result<Value<OrderedNsTAIInterval>, InvertedIntervalError> {
        let lower = self.0.to_tai_duration().total_nanoseconds();
        let upper = self.1.to_tai_duration().total_nanoseconds();
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        let mut value = [0; 32];
        value[0..16].copy_from_slice(&i128_to_ordered_be(lower));
        value[16..32].copy_from_slice(&i128_to_ordered_be(upper));
        Ok(Value::new(value))
    }
}

impl TryFromValue<'_, OrderedNsTAIInterval> for (Epoch, Epoch) {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<OrderedNsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128_from_ordered_be(v.raw[0..16].try_into().unwrap());
        let upper = i128_from_ordered_be(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok((
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower)),
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper)),
        ))
    }
}

impl TryFromValue<'_, OrderedNsTAIInterval> for (i128, i128) {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<OrderedNsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128_from_ordered_be(v.raw[0..16].try_into().unwrap());
        let upper = i128_from_ordered_be(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok((lower, upper))
    }
}

impl TryFromValue<'_, OrderedNsTAIInterval> for Lower {
    type Error = Infallible;
    fn try_from_value(v: &Value<OrderedNsTAIInterval>) -> Result<Self, Infallible> {
        Ok(Lower(i128_from_ordered_be(v.raw[0..16].try_into().unwrap())))
    }
}

impl TryFromValue<'_, OrderedNsTAIInterval> for Upper {
    type Error = Infallible;
    fn try_from_value(v: &Value<OrderedNsTAIInterval>) -> Result<Self, Infallible> {
        Ok(Upper(i128_from_ordered_be(v.raw[16..32].try_into().unwrap())))
    }
}

impl TryFromValue<'_, OrderedNsTAIInterval> for Midpoint {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<OrderedNsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128_from_ordered_be(v.raw[0..16].try_into().unwrap());
        let upper = i128_from_ordered_be(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok(Midpoint(lower + (upper - lower) / 2))
    }
}

impl TryFromValue<'_, OrderedNsTAIInterval> for Width {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<OrderedNsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128_from_ordered_be(v.raw[0..16].try_into().unwrap());
        let upper = i128_from_ordered_be(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok(Width(upper - lower))
    }
}

/// The lower bound exceeds the upper bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvertedIntervalError {
    /// The lower bound that was greater than `upper`.
    pub lower: i128,
    /// The upper bound that was less than `lower`.
    pub upper: i128,
}

impl std::fmt::Display for InvertedIntervalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inverted interval: lower {} > upper {}", self.lower, self.upper)
    }
}

impl ValueSchema for NsTAIInterval {
    type ValidationError = InvertedIntervalError;

    fn validate(value: Value<Self>) -> Result<Value<Self>, Self::ValidationError> {
        let lower = i128::from_le_bytes(value.raw[0..16].try_into().unwrap());
        let upper = i128::from_le_bytes(value.raw[16..32].try_into().unwrap());
        if lower > upper {
            Err(InvertedIntervalError { lower, upper })
        } else {
            Ok(value)
        }
    }
}

impl TryToValue<NsTAIInterval> for (Epoch, Epoch) {
    type Error = InvertedIntervalError;
    fn try_to_value(self) -> Result<Value<NsTAIInterval>, InvertedIntervalError> {
        let lower = self.0.to_tai_duration().total_nanoseconds();
        let upper = self.1.to_tai_duration().total_nanoseconds();
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        let mut value = [0; 32];
        value[0..16].copy_from_slice(&lower.to_le_bytes());
        value[16..32].copy_from_slice(&upper.to_le_bytes());
        Ok(Value::new(value))
    }
}

impl TryFromValue<'_, NsTAIInterval> for (Epoch, Epoch) {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<NsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128::from_le_bytes(v.raw[0..16].try_into().unwrap());
        let upper = i128::from_le_bytes(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok((
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower)),
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper)),
        ))
    }
}

impl TryFromValue<'_, NsTAIInterval> for (i128, i128) {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<NsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128::from_le_bytes(v.raw[0..16].try_into().unwrap());
        let upper = i128::from_le_bytes(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok((lower, upper))
    }
}

/// The lower bound of a TAI interval in nanoseconds.
/// Use this when you want to sort or compare by interval start time.
///
/// ```rust,ignore
/// find!(t: Lower, pattern!(&space, [{ entity @ attr: ?t }]))
///     .max_by_key(|t| *t)  // latest start time
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lower(pub i128);

/// The upper bound of a TAI interval in nanoseconds.
/// Use this when you want to sort or compare by interval end time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Upper(pub i128);

/// The midpoint of a TAI interval in nanoseconds.
/// Use this when you want to sort or compare by interval center.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Midpoint(pub i128);

/// The width of a TAI interval in nanoseconds.
/// Use this when you want to sort or compare by interval duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Width(pub i128);

impl TryFromValue<'_, NsTAIInterval> for Lower {
    type Error = Infallible;
    fn try_from_value(v: &Value<NsTAIInterval>) -> Result<Self, Infallible> {
        let lower = i128::from_le_bytes(v.raw[0..16].try_into().unwrap());
        Ok(Lower(lower))
    }
}

impl TryFromValue<'_, NsTAIInterval> for Upper {
    type Error = Infallible;
    fn try_from_value(v: &Value<NsTAIInterval>) -> Result<Self, Infallible> {
        let upper = i128::from_le_bytes(v.raw[16..32].try_into().unwrap());
        Ok(Upper(upper))
    }
}

impl TryFromValue<'_, NsTAIInterval> for Midpoint {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<NsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128::from_le_bytes(v.raw[0..16].try_into().unwrap());
        let upper = i128::from_le_bytes(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok(Midpoint(lower + (upper - lower) / 2))
    }
}

impl TryFromValue<'_, NsTAIInterval> for Width {
    type Error = InvertedIntervalError;
    fn try_from_value(v: &Value<NsTAIInterval>) -> Result<Self, InvertedIntervalError> {
        let lower = i128::from_le_bytes(v.raw[0..16].try_into().unwrap());
        let upper = i128::from_le_bytes(v.raw[16..32].try_into().unwrap());
        if lower > upper {
            return Err(InvertedIntervalError { lower, upper });
        }
        Ok(Width(upper - lower))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hifitime_conversion() {
        let epoch = Epoch::from_tai_duration(Duration::from_total_nanoseconds(0));
        let time_in: (Epoch, Epoch) = (epoch, epoch);
        let interval: Value<NsTAIInterval> = time_in.try_to_value().unwrap();
        let time_out: (Epoch, Epoch) = interval.try_from_value().unwrap();

        assert_eq!(time_in, time_out);
    }

    #[test]
    fn projection_types() {
        let lower_ns: i128 = 1_000_000_000;
        let upper_ns: i128 = 3_000_000_000;
        let lower = Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower_ns));
        let upper = Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper_ns));
        let interval: Value<NsTAIInterval> = (lower, upper).try_to_value().unwrap();

        let l: Lower = interval.from_value();
        let u: Upper = interval.from_value();
        let m: Midpoint = interval.try_from_value().unwrap();
        let w: Width = interval.try_from_value().unwrap();

        assert_eq!(l.0, lower_ns);
        assert_eq!(u.0, upper_ns);
        assert_eq!(m.0, 2_000_000_000); // midpoint
        assert_eq!(w.0, 2_000_000_000); // width
        assert!(l < Lower(upper_ns)); // Ord works
    }

    #[test]
    fn try_to_value_rejects_inverted() {
        let lower = Epoch::from_tai_duration(Duration::from_total_nanoseconds(2_000_000_000));
        let upper = Epoch::from_tai_duration(Duration::from_total_nanoseconds(1_000_000_000));
        let result: Result<Value<NsTAIInterval>, _> = (lower, upper).try_to_value();
        assert!(result.is_err());
    }

    #[test]
    fn validate_accepts_equal() {
        let t = Epoch::from_tai_duration(Duration::from_total_nanoseconds(1_000_000_000));
        let interval: Value<NsTAIInterval> = (t, t).try_to_value().unwrap();
        assert!(NsTAIInterval::validate(interval).is_ok());
    }

    #[test]
    fn nanosecond_conversion() {
        let lower_ns: i128 = 1_000_000_000;
        let upper_ns: i128 = 2_000_000_000;
        let lower = Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower_ns));
        let upper = Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper_ns));
        let interval: Value<NsTAIInterval> = (lower, upper).try_to_value().unwrap();

        let (out_lower, out_upper): (i128, i128) = interval.try_from_value().unwrap();
        assert_eq!(out_lower, lower_ns);
        assert_eq!(out_upper, upper_ns);
    }

    #[test]
    fn be_roundtrip() {
        let lower = Epoch::from_tai_duration(Duration::from_total_nanoseconds(1_000_000_000));
        let upper = Epoch::from_tai_duration(Duration::from_total_nanoseconds(2_000_000_000));
        let interval: Value<OrderedNsTAIInterval> = (lower, upper).try_to_value().unwrap();
        let (out_lower, out_upper): (Epoch, Epoch) = interval.try_from_value().unwrap();
        assert_eq!((lower, upper), (out_lower, out_upper));

        let (l, u): (i128, i128) = interval.try_from_value().unwrap();
        assert_eq!(l, 1_000_000_000);
        assert_eq!(u, 2_000_000_000);
    }

    #[test]
    fn be_projection_types() {
        let lower_ns: i128 = 1_000_000_000;
        let upper_ns: i128 = 3_000_000_000;
        let lower = Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower_ns));
        let upper = Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper_ns));
        let interval: Value<OrderedNsTAIInterval> = (lower, upper).try_to_value().unwrap();

        let l: Lower = interval.from_value();
        let u: Upper = interval.from_value();
        assert_eq!(l.0, lower_ns);
        assert_eq!(u.0, upper_ns);
    }

    #[test]
    fn be_byte_order_matches_numeric_order() {
        // Order-preserving BE: byte order = i128 numeric order.
        let times = [i128::MIN, -1_000_000_000, -1, 0, 1, 1_000_000_000, i128::MAX];
        for pair in times.windows(2) {
            let a = i128_to_ordered_be(pair[0]);
            let b = i128_to_ordered_be(pair[1]);
            assert!(a < b, "{} should sort before {} in bytes", pair[0], pair[1]);
        }
    }

    #[test]
    fn be_roundtrip_edge_cases() {
        for v in [i128::MIN, -1, 0, 1, i128::MAX] {
            assert_eq!(i128_from_ordered_be(i128_to_ordered_be(v)), v);
        }
    }
}
