//! Order-preserving exact rational encoding.
//!
//! [`R256`](super::r256::R256) stores a rational as two `i128` limbs. That is
//! exact and canonical, but the *bytes* sort by numerator first, so `1/1` sorts
//! below `2/3` even though `1 > 2/3`. Index ranges over an `R256` column are
//! therefore meaningless as numeric ranges.
//!
//! `ROrd256` fixes that: the 32 stored bytes sort — under plain unsigned
//! bytewise comparison, which is exactly [`Inline`]'s [`Ord`] and the order the
//! indexes use — in *numeric* order, while still being exact and canonical.
//!
//! # Construction
//!
//! The Stern–Brocot tree is a binary search tree over the rationals, so its
//! in-order traversal is numerically sorted, and a root path (`L`/`R`) with a
//! terminator that sorts between `L` and `R` compares lexicographically in
//! numeric order. A raw path is not width-bounded though — `1/1000000` is
//! 999999 left branches deep.
//!
//! The continued fraction `[a0; a1, a2, …]` is the run-length encoding of that
//! same path (the terms are the run lengths, with the final run one shorter),
//! so it carries the identical order at O(log min(p,q)) terms instead of
//! O(p + q) branches. Alternating run directions become the classic alternating
//! comparison: `a0` ascending, `a1` descending, `a2` ascending, … A CF that has
//! ended behaves like a `+∞` term at the next position.
//!
//! So the layout is: each term as a self-delimiting, order-preserving bit code,
//! bit-complemented at odd positions (turning "descending" back into plain
//! ascending), followed by a terminator that is above every code — also
//! complemented at odd positions, which correctly turns `+∞` into `-∞` there.
//! Because the term code is prefix-free, the concatenation of codes compares
//! term-by-term, which is precisely the alternating rule.
//!
//! # Term code
//!
//! For `m >= 1`: `floor(log2 m)` one-bits, a zero, then the low
//! `floor(log2 m)` bits of `m`. This is Elias gamma with the unary length
//! prefix written as ones rather than zeros, which is what makes it
//! *increasing* rather than merely self-delimiting.
//!
//! ```text
//! 1 -> 0        2 -> 100     3 -> 101     4 -> 11000   …   8 -> 1110000
//! ```
//!
//! It is prefix-free (the unary prefix delimits it) and strictly increasing
//! under lexicographic comparison, and it costs a single bit for `m == 1` —
//! which matters, because worst-case-length continued fractions are exactly the
//! all-ones ones.
//!
//! `a0` may be any integer, so it is written as a sign bit (`1` for
//! non-negative) followed by the term code for `a0 + 1` resp. `-a0`, the latter
//! bit-complemented so that negatives sort descending among themselves. Terms
//! `a1..` are all `>= 1` and need no sign.
//!
//! The terminator is "every remaining bit is the pad value" — all ones at even
//! positions, all zeros at odd ones. No term code can be all ones (each
//! contains the separator zero), so the terminator is unambiguous and sorts
//! above every code, as required.
//!
//! # Canonicality
//!
//! Continued fractions admit two forms, `[…, n]` and `[…, n-1, 1]`. The
//! Euclidean algorithm only ever emits the first: after the initial term the
//! running value is always `> 1`, so a terminating final term is necessarily
//! `>= 2`. [`ROrd256::validate`] rejects stored bytes whose final term is `1`,
//! along with any other non-canonical pattern, so exactly one byte string
//! exists per value and intrinsic ids stay stable.
//!
//! # Representable subset
//!
//! Encoding fails cleanly with [`OrderedRatioError::OutOfDomain`] rather than
//! rounding. Cost is roughly `2 * log2(max(|p|, q))` bits, so as rules of
//! thumb:
//!
//! * **Guaranteed**: every `p/q` with `max(|p|, q) <= 2^104` fits. (The
//!   smallest value that does *not* fit needs `max(|p|, q) > 2^104.7`; the
//!   worst case is a long continued fraction of small terms.)
//! * **Typical**: random 128-bit `p` and `q` fit about 16% of the time; below
//!   96 bits essentially everything fits.
//! * **Always**: every `i128` integer, and every `1/n` for `n: i128`.
//!
//! [`R256`](super::r256::R256) covers the full `i128 × i128` box and encodes in
//! a few nanoseconds; prefer it when you do not need numeric byte order.

use crate::id::ExclusiveId;
use crate::id::Id;
use crate::id_hex;
use crate::inline::Encodes;
use crate::inline::Inline;
use crate::inline::InlineEncoding;
use crate::inline::TryFromInline;
use crate::inline::TryToInline;
use crate::macros::entity;
use crate::metadata;
use crate::metadata::MetaDescribe;
use crate::trible::Fragment;

use core::fmt;

use num_rational::Ratio;

/// Number of payload bits in an inline value.
const INLINE_BITS: usize = 256;

/// A 256-bit exact rational whose bytes sort in numeric order.
///
/// The value is the canonical continued fraction of the rational, written with
/// an order-preserving prefix-free code per term and bit-complemented at the
/// positions where continued-fraction comparison runs descending. See the
/// [module documentation](self) for the construction, the canonical form, and
/// the representable subset.
///
/// ```
/// use triblespace_core::prelude::*;
/// use triblespace_core::inline::TryToInline;
/// use triblespace_core::inline::encodings::rord256::ROrd256;
/// use num_rational::Ratio;
///
/// let two_thirds: Inline<ROrd256> = Ratio::new(2i128, 3).try_to_inline().unwrap();
/// let one: Inline<ROrd256> = Ratio::new(1i128, 1).try_to_inline().unwrap();
/// assert!(two_thirds < one); // bytewise, and 2/3 < 1
///
/// let back: Ratio<i128> = two_thirds.try_from_inline().unwrap();
/// assert_eq!(back, Ratio::new(2, 3));
/// ```
pub struct ROrd256;

/// Errors produced when converting to or from [`ROrd256`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedRatioError {
    /// The rational is exact and well-formed but its canonical continued
    /// fraction needs more than 256 bits. See the [module
    /// documentation](self#representable-subset) for what fits.
    OutOfDomain,
    /// The denominator is zero, which is not a rational.
    ZeroDenominator,
    /// The stored bytes are not the canonical encoding of any rational: a
    /// truncated term code, or a final continued-fraction term of `1` (which
    /// denotes a value the canonical form spells differently).
    NonCanonical,
    /// The stored bytes decode to a rational whose numerator or denominator
    /// does not fit in `i128`.
    Overflow,
}

impl fmt::Display for OrderedRatioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfDomain => f.write_str("rational too wide for a 256-bit ordered encoding"),
            Self::ZeroDenominator => f.write_str("denominator is zero"),
            Self::NonCanonical => f.write_str("bytes are not a canonical ordered rational"),
            Self::Overflow => f.write_str("decoded rational exceeds i128"),
        }
    }
}

impl std::error::Error for OrderedRatioError {}

// ---------------------------------------------------------------------------
// bit plumbing
// ---------------------------------------------------------------------------

/// MSB-first bit writer over the 32-byte payload.
///
/// MSB-first is what makes bytewise comparison agree with bitwise
/// lexicographic comparison.
struct BitWriter {
    buf: [u8; 32],
    pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: [0; 32],
            pos: 0,
        }
    }

    fn push(&mut self, bit: bool) -> Result<(), OrderedRatioError> {
        if self.pos >= INLINE_BITS {
            return Err(OrderedRatioError::OutOfDomain);
        }
        if bit {
            self.buf[self.pos / 8] |= 0x80 >> (self.pos % 8);
        }
        self.pos += 1;
        Ok(())
    }

    /// Write the order-preserving prefix-free code for `m >= 1`, optionally
    /// bit-complemented.
    fn push_term(&mut self, m: u128, invert: bool) -> Result<(), OrderedRatioError> {
        debug_assert!(m >= 1);
        let k = (127 - m.leading_zeros()) as usize;
        for _ in 0..k {
            self.push(!invert)?;
        }
        self.push(invert)?;
        for i in (0..k).rev() {
            self.push(((m >> i) & 1 == 1) ^ invert)?;
        }
        Ok(())
    }

    /// Fill the remainder with the terminator and yield the payload.
    fn finish(mut self, pad: bool) -> [u8; 32] {
        if pad {
            while self.pos < INLINE_BITS {
                // `push` cannot fail below INLINE_BITS.
                let _ = self.push(true);
            }
        }
        self.buf
    }
}

/// MSB-first bit reader over the 32-byte payload.
struct BitReader<'a> {
    buf: &'a [u8; 32],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8; 32]) -> Self {
        Self { buf, pos: 0 }
    }

    fn next_bit(&mut self) -> Result<bool, OrderedRatioError> {
        if self.pos >= INLINE_BITS {
            return Err(OrderedRatioError::NonCanonical);
        }
        let bit = self.buf[self.pos / 8] & (0x80 >> (self.pos % 8)) != 0;
        self.pos += 1;
        Ok(bit)
    }

    /// Is every remaining bit equal to `v`? Vacuously true once exhausted,
    /// which is how an encoding that exactly fills 256 bits terminates.
    fn rest_is(&self, v: bool) -> bool {
        let fill = if v { 0xFFu8 } else { 0x00u8 };
        let mut p = self.pos;
        while p < INLINE_BITS && p % 8 != 0 {
            if (self.buf[p / 8] & (0x80 >> (p % 8)) != 0) != v {
                return false;
            }
            p += 1;
        }
        while p < INLINE_BITS {
            if self.buf[p / 8] != fill {
                return false;
            }
            p += 8;
        }
        true
    }

    /// Read one term code, optionally bit-complemented. Returns `m >= 1`.
    fn next_term(&mut self, invert: bool) -> Result<u128, OrderedRatioError> {
        let mut k = 0usize;
        loop {
            if self.next_bit()? ^ invert {
                k += 1;
                // A `u128` term has at most 127 leading ones; more can never be
                // a term, only a truncated or corrupt pattern.
                if k > 127 {
                    return Err(OrderedRatioError::NonCanonical);
                }
            } else {
                break;
            }
        }
        let mut m: u128 = 1;
        for _ in 0..k {
            m = (m << 1) | u128::from(self.next_bit()? ^ invert);
        }
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// encode / decode
// ---------------------------------------------------------------------------

/// Encode `numer / denom` (denominator must be positive) into the payload.
fn encode_parts(numer: i128, denom: i128) -> Result<[u8; 32], OrderedRatioError> {
    if denom == 0 {
        return Err(OrderedRatioError::ZeroDenominator);
    }
    // `Ratio` normalizes to a positive denominator; be defensive about
    // `new_raw`-built values. `i128::MIN` cannot be negated, but as a
    // denominator it is out of domain anyway.
    let (mut num, mut den) = if denom < 0 {
        match (numer.checked_neg(), denom.checked_neg()) {
            (Some(n), Some(d)) => (n, d),
            _ => return Err(OrderedRatioError::OutOfDomain),
        }
    } else {
        (numer, denom)
    };

    let mut w = BitWriter::new();

    // First term only: `num` may be negative, so this needs a true floor.
    // `den > 0` means neither `div_euclid` nor `rem_euclid` can overflow, but
    // reconstructing one from the other could (`floor(i128::MIN/3) * 3` is
    // below `i128::MIN`), so this one step pays for two divisions.
    let a0 = num.div_euclid(den);
    let mut rem = num.rem_euclid(den);
    if a0 >= 0 {
        w.push(true)?;
        w.push_term(a0 as u128 + 1, false)?;
    } else {
        w.push(false)?;
        w.push_term(a0.unsigned_abs(), true)?;
    }
    let mut position = 1usize;

    // From here `num > den > 0`, so `a * den <= num` holds and one division per
    // term is enough. That halves the divisions, which dominate encoding.
    while rem != 0 {
        num = den;
        den = rem;
        if let Ok(mut n) = u64::try_from(num) {
            // The remainders shrink fast, so most terms of a long continued
            // fraction are found here — and 64-bit division is several times
            // cheaper than the compiler's 128-bit division helper.
            let mut d = den as u64;
            loop {
                let a = n / d;
                w.push_term(u128::from(a), position % 2 == 1)?;
                position += 1;
                let r = n - a * d;
                if r == 0 {
                    return Ok(w.finish(position % 2 == 0));
                }
                n = d;
                d = r;
            }
        }
        let a = num / den;
        rem = num - a * den;
        debug_assert!(a >= 1);
        w.push_term(a as u128, position % 2 == 1)?;
        position += 1;
    }

    // The terminator sits at `position`, and is all ones at even positions.
    Ok(w.finish(position % 2 == 0))
}

/// Decode the payload into a reduced `(numer, denom)` pair.
fn decode_parts(raw: &[u8; 32]) -> Result<(i128, i128), OrderedRatioError> {
    const LIMIT: u128 = i128::MAX as u128 + 1;

    let mut r = BitReader::new(raw);

    let a0 = if r.next_bit()? {
        let m = r.next_term(false)?;
        if m > LIMIT {
            return Err(OrderedRatioError::Overflow);
        }
        (m - 1) as i128
    } else {
        let m = r.next_term(true)?;
        if m > LIMIT {
            return Err(OrderedRatioError::Overflow);
        }
        if m == LIMIT {
            i128::MIN
        } else {
            -(m as i128)
        }
    };

    // Continuants: p_i = a_i * p_{i-1} + p_{i-2}, seeded with p_{-1} = 1,
    // p_{-2} = 0 and q_{-1} = 0, q_{-2} = 1.
    let (mut p1, mut p2) = (1i128, 0i128);
    let (mut q1, mut q2) = (0i128, 1i128);
    let step = |a: i128, x1: i128, x2: i128| -> Result<i128, OrderedRatioError> {
        a.checked_mul(x1)
            .and_then(|v| v.checked_add(x2))
            .ok_or(OrderedRatioError::Overflow)
    };
    (p1, p2) = (step(a0, p1, p2)?, p1);
    (q1, q2) = (step(a0, q1, q2)?, q1);

    let mut position = 1usize;
    let mut last = None;
    loop {
        let invert = position % 2 == 1;
        // The terminator is the pad value repeated to the end: ones at even
        // positions, zeros at odd ones.
        if r.rest_is(!invert) {
            break;
        }
        let m = r.next_term(invert)?;
        let a = i128::try_from(m).map_err(|_| OrderedRatioError::Overflow)?;
        (p1, p2) = (step(a, p1, p2)?, p1);
        (q1, q2) = (step(a, q1, q2)?, q1);
        last = Some(a);
        position += 1;
    }

    // Canonicality: the Euclidean algorithm never emits a trailing 1 after the
    // first term, so bytes that claim one are a second spelling of a value that
    // already has an encoding.
    if last == Some(1) {
        return Err(OrderedRatioError::NonCanonical);
    }

    debug_assert!(q1 > 0);
    Ok((p1, q1))
}

// ---------------------------------------------------------------------------
// schema
// ---------------------------------------------------------------------------

impl MetaDescribe for ROrd256 {
    fn describe() -> Fragment {
        let id: Id = id_hex!("59CEE98837ACCBD6A8528A4F7293F2EF");
        #[allow(unused_mut)]
        let mut tribles = entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "rord256",
                metadata::description: "Exact rational whose 32 bytes sort in numeric order under plain bytewise comparison. The value is stored as its canonical continued fraction, each term in an order-preserving prefix-free code, bit-complemented at the positions where continued-fraction comparison runs descending.\n\nUse when you need exact rational arithmetic AND numeric range queries on the same column, so an index range is an exact answer rather than a float-keyed approximation. Prefer R256 when you only need exactness (it is simpler, faster, and covers the full i128 x i128 box) and F64/F256 when approximation is fine.\n\nEvery p/q with max(|p|,q) up to 2^104 is representable, as is every i128 integer; wider values are rejected rather than rounded. Exactly one byte string exists per value, so intrinsic ids are stable.",
                metadata::tag: metadata::KIND_INLINE_ENCODING,
        };

        #[cfg(feature = "wasm")]
        {
            tribles += entity! { ExclusiveId::force_ref(&id) @
                metadata::value_formatter: wasm_formatter::RORD256_WASM,
            };
        }
        tribles
    }
}

impl InlineEncoding for ROrd256 {
    type ValidationError = OrderedRatioError;
    type Encoding = Self;

    fn validate(value: Inline<Self>) -> Result<Inline<Self>, Self::ValidationError> {
        // Decoding enforces canonicality: every bit is consumed either as a
        // term code or as the terminator pad, term codes must be complete, and
        // a trailing `1` term is rejected. Re-encoding therefore reproduces the
        // input exactly, which is what content addressing depends on.
        decode_parts(&value.raw)?;
        Ok(value)
    }
}

#[cfg(feature = "wasm")]
pub(crate) mod wasm_formatter {
    use core::fmt::Write;

    use triblespace_core_macros::value_formatter;

    #[value_formatter]
    pub(crate) fn rord256(raw: &[u8; 32], out: &mut impl Write) -> Result<(), u32> {
        let mut pos = 0usize;
        let bit = |p: usize| raw[p / 8] & (0x80 >> (p % 8)) != 0;

        let term = |pos: &mut usize, invert: bool| -> Result<u128, u32> {
            let mut k = 0usize;
            loop {
                if *pos >= 256 {
                    return Err(2);
                }
                let b = bit(*pos) ^ invert;
                *pos += 1;
                if !b {
                    break;
                }
                k += 1;
                if k > 127 {
                    return Err(2);
                }
            }
            let mut m: u128 = 1;
            for _ in 0..k {
                if *pos >= 256 {
                    return Err(2);
                }
                m = (m << 1) | u128::from(bit(*pos) ^ invert);
                *pos += 1;
            }
            Ok(m)
        };

        if pos >= 256 {
            return Err(2);
        }
        let sign = bit(pos);
        pos += 1;
        let m = term(&mut pos, !sign)?;
        if m > i128::MAX as u128 + 1 {
            return Err(2);
        }
        let a0 = if sign {
            (m - 1) as i128
        } else if m == i128::MAX as u128 + 1 {
            i128::MIN
        } else {
            -(m as i128)
        };

        let (mut p1, mut p2) = (1i128, 0i128);
        let (mut q1, mut q2) = (0i128, 1i128);
        macro_rules! step {
            ($a:expr) => {{
                let np = $a
                    .checked_mul(p1)
                    .and_then(|v| v.checked_add(p2))
                    .ok_or(2u32)?;
                let nq = $a
                    .checked_mul(q1)
                    .and_then(|v| v.checked_add(q2))
                    .ok_or(2u32)?;
                p2 = p1;
                p1 = np;
                q2 = q1;
                q1 = nq;
            }};
        }
        step!(a0);

        let mut position = 1usize;
        loop {
            let invert = position % 2 == 1;
            let pad = !invert;
            let mut all_pad = true;
            let mut p = pos;
            while p < 256 {
                if bit(p) != pad {
                    all_pad = false;
                    break;
                }
                p += 1;
            }
            if all_pad {
                break;
            }
            let m = term(&mut pos, invert)?;
            if m > i128::MAX as u128 {
                return Err(2);
            }
            let a = m as i128;
            step!(a);
            position += 1;
        }

        if q1 == 1 {
            write!(out, "{p1}").map_err(|_| 1u32)?;
        } else {
            write!(out, "{p1}/{q1}").map_err(|_| 1u32)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// conversions
// ---------------------------------------------------------------------------

impl TryToInline<ROrd256> for Ratio<i128> {
    type Error = OrderedRatioError;

    fn try_to_inline(self) -> Result<Inline<ROrd256>, Self::Error> {
        Ok(Inline::new(encode_parts(*self.numer(), *self.denom())?))
    }
}

impl TryFromInline<'_, ROrd256> for Ratio<i128> {
    type Error = OrderedRatioError;

    fn try_from_inline(v: &Inline<ROrd256>) -> Result<Self, Self::Error> {
        let (n, d) = decode_parts(&v.raw)?;
        // Continuants of a continued fraction are always coprime, so the pair
        // is already reduced.
        Ok(Ratio::new_raw(n, d))
    }
}

/// Every `i128` is representable: the widest term code is 255 bits, and with
/// the sign bit that exactly fills the payload.
impl Encodes<i128> for ROrd256 {
    type Output = Inline<ROrd256>;
    fn encode(source: i128) -> Inline<ROrd256> {
        Inline::new(
            encode_parts(source, 1).expect("every i128 integer fits in an ordered rational"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inline::TryFromInline;
    use proptest::prelude::*;

    fn enc(n: i128, d: i128) -> Inline<ROrd256> {
        Ratio::new(n, d).try_to_inline().expect("in domain")
    }

    fn dec(v: &Inline<ROrd256>) -> Ratio<i128> {
        Ratio::<i128>::try_from_inline(v).expect("valid")
    }

    // -- the term code, tested in isolation --------------------------------

    #[test]
    fn term_code_is_prefix_free_and_increasing() {
        // Render the code for m as a bit string.
        fn code(m: u128) -> Vec<bool> {
            let mut w = BitWriter::new();
            w.push_term(m, false).unwrap();
            (0..w.pos)
                .map(|p| w.buf[p / 8] & (0x80 >> (p % 8)) != 0)
                .collect()
        }
        let codes: Vec<Vec<bool>> = (1u128..2000).map(code).collect();
        // strictly increasing lexicographically
        for w in codes.windows(2) {
            assert!(w[0] < w[1], "term code not increasing");
        }
        // prefix-free
        for (i, a) in codes.iter().enumerate() {
            for b in codes.iter().skip(i + 1) {
                assert!(!b.starts_with(a), "term code not prefix-free");
            }
        }
        // known values from the module docs
        assert_eq!(code(1), vec![false]);
        assert_eq!(code(2), vec![true, false, false]);
        assert_eq!(code(3), vec![true, false, true]);
        assert_eq!(code(4), vec![true, true, false, false, false]);
        // no code is all ones, so the terminator is unambiguous
        for c in &codes {
            assert!(c.iter().any(|b| !b));
        }
    }

    #[test]
    fn term_code_roundtrips_over_the_whole_u128_range() {
        for m in (0..128).map(|k| 1u128 << k).chain(1u128..1000) {
            for invert in [false, true] {
                let mut w = BitWriter::new();
                if w.push_term(m, invert).is_err() {
                    continue;
                }
                let buf = w.finish(false);
                let mut r = BitReader::new(&buf);
                assert_eq!(r.next_term(invert).unwrap(), m, "m={m} invert={invert}");
            }
        }
    }

    // -- ordering ----------------------------------------------------------

    #[test]
    fn dense_grid_sorts_numerically() {
        let mut vals: Vec<Ratio<i128>> = Vec::new();
        for n in -60i128..=60 {
            for d in 1i128..=60 {
                vals.push(Ratio::new(n, d));
            }
        }
        vals.sort();
        vals.dedup();

        let mut by_bytes: Vec<(Inline<ROrd256>, Ratio<i128>)> = vals
            .iter()
            .map(|r| (enc(*r.numer(), *r.denom()), *r))
            .collect();
        by_bytes.sort_by(|a, b| a.0.raw.cmp(&b.0.raw));

        let sorted: Vec<Ratio<i128>> = by_bytes.iter().map(|(_, r)| *r).collect();
        assert_eq!(sorted, vals, "bytewise order != numeric order");
    }

    #[test]
    fn zero_integers_and_negatives_sit_where_they_should() {
        let cases = [
            (-3i128, 1i128),
            (-5, 2),
            (-2, 1),
            (-1, 1),
            (-1, 2),
            (-1, 1000000),
            (0, 1),
            (1, 1000000),
            (1, 3),
            (1, 2),
            (2, 3),
            (1, 1),
            (3, 2),
            (2, 1),
            (1000000, 1),
        ];
        let mut prev: Option<Inline<ROrd256>> = None;
        for (n, d) in cases {
            let v = enc(n, d);
            assert_eq!(dec(&v), Ratio::new(n, d));
            if let Some(p) = prev {
                assert!(p.raw < v.raw, "{n}/{d} did not sort above its predecessor");
            }
            prev = Some(v);
        }
    }

    // -- adversarial inputs ------------------------------------------------

    #[test]
    fn fibonacci_ratios_have_the_longest_cfs_and_still_sort() {
        // All-ones continued fractions are the *longest* CF for a given
        // magnitude — and, because a term of 1 costs a single bit, also the
        // *cheapest*. Every Fibonacci ratio that fits in `i128` at all
        // (F(184)/F(183) is the last) encodes in well under 256 bits, which is
        // why the binding worst case is medium-sized terms instead.
        let mut fib: Vec<i128> = vec![1, 1];
        while let Some(next) = fib[fib.len() - 1].checked_add(fib[fib.len() - 2]) {
            fib.push(next);
        }
        let mut encoded: Vec<(Ratio<i128>, Inline<ROrd256>)> = Vec::new();
        for w in fib.windows(2) {
            let r = Ratio::new(w[1], w[0]);
            let v = r.try_to_inline().expect("every i128 Fibonacci ratio fits");
            assert_eq!(dec(&v), r);
            encoded.push((r, v));
        }
        assert!(encoded.len() > 180, "expected ~184 Fibonacci ratios");

        let mut a = encoded.clone();
        a.sort_by(|x, y| x.0.cmp(&y.0));
        let mut b = encoded;
        b.sort_by(|x, y| x.1.raw.cmp(&y.1.raw));
        assert_eq!(
            a.iter().map(|x| x.0).collect::<Vec<_>>(),
            b.iter().map(|x| x.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn one_over_n_and_n_over_one_span_the_domain() {
        for k in 0..127u32 {
            let n = 1i128 << k;
            // n/1 always fits.
            let v: Inline<ROrd256> = ROrd256::inline_from(n);
            assert_eq!(dec(&v), Ratio::new(n, 1));
            // 1/n always fits too.
            let v = enc(1, n);
            assert_eq!(dec(&v), Ratio::new(1, n));
        }
        let v: Inline<ROrd256> = ROrd256::inline_from(i128::MIN);
        assert_eq!(dec(&v), Ratio::new(i128::MIN, 1));
        let v: Inline<ROrd256> = ROrd256::inline_from(i128::MAX);
        assert_eq!(dec(&v), Ratio::new(i128::MAX, 1));
    }

    /// Build the rational whose continued fraction is `terms`, or `None` if its
    /// continuants leave `i128`.
    fn from_cf(terms: &[i128]) -> Option<Ratio<i128>> {
        let (mut p1, mut p2) = (1i128, 0i128);
        let (mut q1, mut q2) = (0i128, 1i128);
        for &a in terms {
            let np = a.checked_mul(p1)?.checked_add(p2)?;
            let nq = a.checked_mul(q1)?.checked_add(q2)?;
            p2 = p1;
            p1 = np;
            q2 = q1;
            q1 = nq;
        }
        Some(Ratio::new_raw(p1, q1))
    }

    #[test]
    fn values_that_exactly_fill_the_payload_still_order_correctly() {
        // `i128::MIN` and `i128::MAX` need all 256 bits (sign bit + a 255-bit
        // term code), so their terminator is an implicit end-of-buffer rather
        // than a run of pad bits. That is the subtlest path in the decoder, and
        // these two must still bracket everything else.
        let min: Inline<ROrd256> = ROrd256::inline_from(i128::MIN);
        let max: Inline<ROrd256> = ROrd256::inline_from(i128::MAX);
        assert_eq!(used_bits(&min), 256);
        assert_eq!(used_bits(&max), 256);

        let mut others: Vec<(Ratio<i128>, Inline<ROrd256>)> = Vec::new();
        for (n, d) in [
            (i128::MIN + 1, 1i128),
            (i128::MAX - 1, 1),
            (-(1i128 << 100), 3),
            (1i128 << 100, 3),
            (0, 1),
            (1, 1),
            (-1, 1),
            (1, i128::MAX),
            (-1, i128::MAX),
        ] {
            others.push((Ratio::new(n, d), enc(n, d)));
        }
        for (r, v) in &others {
            assert!(min.raw < v.raw, "i128::MIN did not sort below {r}");
            assert!(max.raw > v.raw, "i128::MAX did not sort above {r}");
        }

        // And the whole set, edge values included, sorts numerically.
        let mut all = others;
        all.push((Ratio::new(i128::MIN, 1), min));
        all.push((Ratio::new(i128::MAX, 1), max));
        let mut by_value = all.clone();
        by_value.sort_by_key(|(r, _)| *r);
        let mut by_bytes = all;
        by_bytes.sort_by(|a, b| a.1.raw.cmp(&b.1.raw));
        assert_eq!(
            by_value.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
            by_bytes.iter().map(|(r, _)| *r).collect::<Vec<_>>()
        );
    }

    /// Bits of payload the terms occupy, before the terminator.
    ///
    /// Deliberately computed from the continued fraction rather than by
    /// scanning back over the stored pad: a value's own final bits may equal
    /// the pad value (`i128::MIN` genuinely ends in 128 one-bits), so scanning
    /// under-reports.
    fn used_bits(v: &Inline<ROrd256>) -> usize {
        let r = dec(v);
        let (mut n, mut d) = (*r.numer(), *r.denom());
        let code = |m: u128| 2 * (127 - m.leading_zeros() as usize) + 1;
        let mut bits = 1; // sign bit
        let mut first = true;
        loop {
            let a = n.div_euclid(d);
            let rem = n.rem_euclid(d);
            bits += if first {
                code(if a >= 0 {
                    a as u128 + 1
                } else {
                    a.unsigned_abs()
                })
            } else {
                code(a as u128)
            };
            first = false;
            if rem == 0 {
                break;
            }
            n = d;
            d = rem;
        }
        bits
    }

    #[test]
    fn guaranteed_domain_holds_and_beyond_it_fails_cleanly() {
        // The costliest term per bit of magnitude is a small-but-not-one term,
        // so the adversarial continued fractions are constant runs of 2, 4, and
        // 8 (and mixtures of them). Sweep those right up to the edge.
        for period in [
            vec![2i128],
            vec![4],
            vec![8],
            vec![1, 8],
            vec![2, 4],
            vec![1, 2],
            vec![1, 4],
            vec![3],
        ] {
            let mut terms = vec![1i128];
            let mut checked = 0;
            loop {
                terms.push(period[(terms.len() - 1) % period.len()]);
                let Some(r) = from_cf(&terms) else { break };
                if *r.numer() > (1i128 << 104) {
                    break;
                }
                assert!(
                    r.try_to_inline().is_ok(),
                    "{r} has max(|p|,q) <= 2^104 and must encode (cf {terms:?})"
                );
                checked += 1;
            }
            assert!(
                checked > 10,
                "period {period:?} barely exercised the domain"
            );
        }

        // Out-of-domain is a typed error, never a silent approximation.
        for r in [
            Ratio::new(i128::MAX, i128::MAX - 1),
            from_cf(&vec![2i128; 90]).expect("stays inside i128"),
            from_cf(&vec![4i128; 60]).expect("stays inside i128"),
        ] {
            assert_eq!(
                r.try_to_inline().err(),
                Some(OrderedRatioError::OutOfDomain),
                "{r} should be rejected"
            );
        }
    }

    #[test]
    fn zero_denominator_is_rejected() {
        assert_eq!(
            Ratio::new_raw(1i128, 0).try_to_inline().err(),
            Some(OrderedRatioError::ZeroDenominator)
        );
    }

    #[test]
    fn non_canonical_trailing_one_is_rejected() {
        // Hand-build [1; 1], which is the non-canonical spelling of 2 = [2].
        let mut w = BitWriter::new();
        w.push(true).unwrap();
        w.push_term(2, false).unwrap(); // a0 = 1
        w.push_term(1, true).unwrap(); // a1 = 1  (odd position -> inverted)
                                       // The terminator sits at position 2, which is even, so the pad is ones.
        let raw = w.finish(true);

        let v = Inline::<ROrd256>::new(raw);
        assert_eq!(
            Ratio::<i128>::try_from_inline(&v).err(),
            Some(OrderedRatioError::NonCanonical)
        );
        assert!(ROrd256::validate(Inline::<ROrd256>::new(raw)).is_err());

        // ...while the canonical spelling of the same value, [2], is accepted.
        assert_eq!(dec(&enc(2, 1)), Ratio::new(2, 1));
    }

    #[test]
    fn bit_flips_of_valid_encodings_never_yield_a_second_spelling() {
        // Random 32-byte patterns are almost never valid, so they barely probe
        // canonicality. Mutating *valid* encodings does: every accepted mutant
        // must re-encode to itself, or two byte strings would name one value
        // and intrinsic ids would fork.
        let seeds = [
            (0i128, 1i128),
            (1, 1),
            (-1, 3),
            (2, 3),
            (355, 113),
            (-1, 1 << 40),
            (i128::MAX, 1),
            (1, i128::MAX),
        ];
        let mut accepted = 0;
        for (n, d) in seeds {
            let base = enc(n, d);
            for bit in 0..256 {
                let mut raw = base.raw;
                raw[bit / 8] ^= 0x80 >> (bit % 8);
                let v = Inline::<ROrd256>::new(raw);
                if let Ok(r) = Ratio::<i128>::try_from_inline(&v) {
                    accepted += 1;
                    let re = r.try_to_inline().expect("decoded values re-encode");
                    assert_eq!(raw, re.raw, "bit {bit} of {n}/{d} decoded non-canonically");
                }
            }
        }
        assert!(
            accepted > 200,
            "mutation test accepted too few mutants to be meaningful (got {accepted})"
        );
    }

    /// The wasm formatter is a hand-rolled second decoder compiled to a
    /// separate target, so run it through the real VM rather than trusting that
    /// it merely compiled.
    #[cfg(feature = "wasm")]
    #[test]
    fn wasm_formatter_renders_values() {
        use crate::value_formatter::WasmValueFormatter;

        let f = WasmValueFormatter::new(wasm_formatter::RORD256_WASM).expect("formatter loads");
        for (n, d, want) in [
            (0i128, 1i128, "0"),
            (1, 1, "1"),
            (-7, 1, "-7"),
            (2, 3, "2/3"),
            (-355, 113, "-355/113"),
            (1, 1_000_000, "1/1000000"),
        ] {
            let v = enc(n, d);
            assert_eq!(f.format_value(&v.raw).expect("formats"), want);
        }
    }

    // -- property tests ----------------------------------------------------

    fn arb_ratio() -> impl Strategy<Value = Ratio<i128>> {
        // Spread across magnitudes so small, medium, and near-edge values all
        // get exercised.
        (1u32..=104u32, any::<u128>(), any::<u128>(), any::<bool>()).prop_map(
            |(bits, n, d, neg)| {
                let mask = if bits >= 127 {
                    u128::MAX >> 1
                } else {
                    (1u128 << bits) - 1
                };
                let num = ((n & mask) as i128).max(0);
                let den = ((d & mask) as i128).max(1);
                Ratio::new(if neg { -num } else { num }, den)
            },
        )
    }

    proptest! {
        #[test]
        fn roundtrip(r in arb_ratio()) {
            let v = r.try_to_inline().expect("within the guaranteed domain");
            prop_assert_eq!(dec(&v), r);
        }

        #[test]
        fn bytes_order_matches_numeric_order(a in arb_ratio(), b in arb_ratio()) {
            let va = a.try_to_inline().expect("in domain");
            let vb = b.try_to_inline().expect("in domain");
            prop_assert_eq!(a.cmp(&b), va.raw.cmp(&vb.raw));
        }

        #[test]
        fn one_encoding_per_value(a in arb_ratio()) {
            // Same value spelled with a common factor must encode identically.
            let scaled = Ratio::new_raw(*a.numer(), *a.denom());
            let v1 = a.try_to_inline().expect("in domain");
            let v2 = scaled.try_to_inline().expect("in domain");
            prop_assert_eq!(v1.raw, v2.raw);
            // And decode-then-encode is the identity on valid bytes.
            let back = dec(&v1).try_to_inline().expect("in domain");
            prop_assert_eq!(v1.raw, back.raw);
        }

        #[test]
        fn arbitrary_bytes_either_fail_or_are_canonical(raw in any::<[u8; 32]>()) {
            let v = Inline::<ROrd256>::new(raw);
            if let Ok(r) = Ratio::<i128>::try_from_inline(&v) {
                let re = r.try_to_inline().expect("decoded values re-encode");
                prop_assert_eq!(raw, re.raw, "accepted a non-canonical pattern");
            }
        }

        #[test]
        fn integers_always_fit(n: i128) {
            let v: Inline<ROrd256> = ROrd256::inline_from(n);
            prop_assert_eq!(dec(&v), Ratio::new(n, 1));
        }

        #[test]
        fn reciprocals_always_fit(n in any::<i128>().prop_filter("non-zero", |n| *n != 0)) {
            let r = Ratio::new(1, n);
            let v = r.try_to_inline().expect("1/n always fits");
            prop_assert_eq!(dec(&v), r);
        }

        #[test]
        fn validate_agrees_with_decode(raw in any::<[u8; 32]>()) {
            let v = Inline::<ROrd256>::new(raw);
            prop_assert_eq!(
                ROrd256::validate(Inline::<ROrd256>::new(raw)).is_ok(),
                Ratio::<i128>::try_from_inline(&v).is_ok()
            );
        }
    }
}
