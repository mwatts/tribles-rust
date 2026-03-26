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

/// A value schema for a TAI interval.
/// A TAI interval is a pair of TAI epochs.
/// The interval is stored as two 128-bit signed integers, the lower and upper bounds.
/// The lower bound is stored in the first 16 bytes and the upper bound is stored in the last 16 bytes.
/// Both the lower and upper bounds are stored in little-endian byte order.
/// Both the lower and upper bounds are inclusive. That is, the interval contains all TAI epochs between the lower and upper bounds.
pub struct NsTAIInterval;

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
}
