//! What does numeric byte order cost, relative to `R256`'s two-`i128` store?
//!
//! `R256` writes two `to_be_bytes` and reads two `from_be_bytes`; `ROrd256` has
//! to run a Euclidean continued-fraction expansion on encode and a continuant
//! recurrence on decode, both O(number of CF terms). This measures that gap
//! across the shapes that actually differ — short CFs (small fractions,
//! integers), typical 64-bit fractions, and the long-CF worst cases — and also
//! reports how much of the 256-bit budget each shape consumes, which is what
//! bounds the representable subset.
//!
//! Run: cargo bench -p triblespace-core --bench ordered_rational

use std::hint::black_box;
use std::time::Instant;

use num_rational::Ratio;
use triblespace_core::inline::encodings::r256::{R256BE, R256LE};
use triblespace_core::inline::encodings::rord256::ROrd256;
use triblespace_core::inline::{Inline, InlineEncoding, TryFromInline, TryToInline};

/// Small deterministic PRNG so the numbers are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, bits: u32) -> i128 {
        let hi = (self.next() as u128) << 64 | self.next() as u128;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
        ((hi & mask) as i128).max(1)
    }
}

fn continuant(terms: &[i128]) -> Option<Ratio<i128>> {
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

/// How many of the 256 bits do this value's terms occupy, before the
/// terminator? Computed from the continued fraction rather than by scanning
/// back over the pad, because a value's own final bits may equal the pad value
/// (`i128::MIN` genuinely ends in 128 one-bits) and scanning under-reports.
fn used_bits(r: Ratio<i128>) -> usize {
    let (mut n, mut d) = (*r.numer(), *r.denom());
    let code = |m: u128| 2 * (127 - m.leading_zeros() as usize) + 1;
    let mut bits = 1; // sign bit
    let mut first = true;
    loop {
        let a = n.div_euclid(d);
        let rem = n.rem_euclid(d);
        bits += if first {
            code(if a >= 0 { a as u128 + 1 } else { a.unsigned_abs() })
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

fn bench(name: &str, samples: &[Ratio<i128>]) {
    assert!(!samples.is_empty());
    let reps = (2_000_000 / samples.len()).max(4);

    // --- baseline: loop + black_box overhead only, so the rest is readable ---
    let t = Instant::now();
    for _ in 0..reps {
        for r in samples {
            black_box(black_box(*r));
        }
    }
    let base = t.elapsed().as_nanos() as f64 / (reps * samples.len()) as f64;

    // --- encode ---
    let t = Instant::now();
    for _ in 0..reps {
        for r in samples {
            black_box(R256BE::inline_from(black_box(*r)));
        }
    }
    let r256_enc = t.elapsed().as_nanos() as f64 / (reps * samples.len()) as f64;

    let t = Instant::now();
    for _ in 0..reps {
        for r in samples {
            black_box(TryToInline::<ROrd256>::try_to_inline(black_box(*r)).unwrap());
        }
    }
    let rord_enc = t.elapsed().as_nanos() as f64 / (reps * samples.len()) as f64;

    // R256LE's encode is the genuinely trivial path: two `to_le_bytes`, no gcd.
    // (R256BE's runs `reduced()` first — a pre-existing asymmetry between the
    // two endian variants.) This is the honest floor to compare against.
    let t = Instant::now();
    for _ in 0..reps {
        for r in samples {
            black_box(R256LE::inline_from(black_box(*r)));
        }
    }
    let r256le_enc = t.elapsed().as_nanos() as f64 / (reps * samples.len()) as f64;

    // --- decode ---
    let a: Vec<Inline<R256BE>> = samples.iter().map(|r| R256BE::inline_from(*r)).collect();
    let b: Vec<Inline<ROrd256>> = samples
        .iter()
        .map(|r| TryToInline::<ROrd256>::try_to_inline(*r).unwrap())
        .collect();

    let t = Instant::now();
    for _ in 0..reps {
        for v in &a {
            black_box(Ratio::<i128>::try_from_inline(black_box(v)).unwrap());
        }
    }
    let r256_dec = t.elapsed().as_nanos() as f64 / (reps * a.len()) as f64;

    let t = Instant::now();
    for _ in 0..reps {
        for v in &b {
            black_box(Ratio::<i128>::try_from_inline(black_box(v)).unwrap());
        }
    }
    let rord_dec = t.elapsed().as_nanos() as f64 / (reps * b.len()) as f64;

    let widths: Vec<usize> = samples.iter().map(|r| used_bits(*r)).collect();
    let mean_w = widths.iter().sum::<usize>() as f64 / widths.len() as f64;
    let max_w = widths.iter().copied().max().unwrap();

    println!(
        "{name:<26} enc {:>6.1} / {:>5.1} / {:>5.1} ns ({:>4.1}x/{:>5.1}x)  dec {:>6.1} / {:>5.1} ns ({:>4.1}x)  [base {base:>4.1}]  bits mean {mean_w:>5.1} max {max_w:>3}",
        rord_enc,
        r256_enc,
        r256le_enc,
        rord_enc / r256_enc,
        rord_enc / r256le_enc,
        rord_dec,
        r256_dec,
        rord_dec / r256_dec,
    );
}

fn main() {
    println!("ROrd256 vs R256BE vs R256LE   (enc: ROrd256 / R256BE(+gcd) / R256LE(raw))\n");

    let mut rng = Rng(0x9E3779B97F4A7C15);

    bench(
        "integers",
        &(1..=64)
            .map(|k| Ratio::new(rng.below(k.min(100)), 1))
            .collect::<Vec<_>>(),
    );
    bench(
        "small fractions (<=16 bit)",
        &(0..64)
            .map(|_| Ratio::new(rng.below(16), rng.below(16)))
            .collect::<Vec<_>>(),
    );
    bench(
        "64-bit p/q",
        &(0..64)
            .map(|_| Ratio::new(rng.below(64), rng.below(64)))
            .collect::<Vec<_>>(),
    );
    bench(
        "100-bit p/q (near edge)",
        &(0..64)
            .map(|_| Ratio::new(rng.below(100), rng.below(100)))
            .collect::<Vec<_>>(),
    );

    // Worst case for ROrd256: the longest continued fractions that still fit.
    let mut fib: Vec<i128> = vec![1, 1];
    while let Some(n) = fib[fib.len() - 1].checked_add(fib[fib.len() - 2]) {
        fib.push(n);
    }
    bench(
        "Fibonacci (longest CF)",
        &fib.windows(2)
            .skip(120)
            .map(|w| Ratio::new(w[1], w[0]))
            .collect::<Vec<_>>(),
    );

    // Costliest case per bit of magnitude: constant runs of small-but-not-one
    // terms. These are what pin the guaranteed domain.
    let adversarial: Vec<Ratio<i128>> = [2i128, 3, 4, 8]
        .iter()
        .flat_map(|&t| {
            (4..90).filter_map(move |n| {
                let r = continuant(&vec![t; n])?;
                TryToInline::<ROrd256>::try_to_inline(r).ok().map(|_| r)
            })
        })
        .collect();
    bench("adversarial CFs (2,3,4,8)", &adversarial);

    // --- domain coverage: what fraction of random p/q fits at each width? ---
    println!("\ndomain coverage (10k random p/q per row)");
    for bits in [32u32, 64, 96, 104, 110, 120, 127] {
        let mut fit = 0;
        let mut widths = Vec::new();
        for _ in 0..10_000 {
            let r = Ratio::new(rng.below(bits), rng.below(bits));
            if TryToInline::<ROrd256>::try_to_inline(r).is_ok() {
                fit += 1;
                widths.push(used_bits(r));
            }
        }
        widths.sort_unstable();
        let med = widths.get(widths.len() / 2).copied().unwrap_or(0);
        let p99 = widths
            .get(widths.len() * 99 / 100)
            .copied()
            .unwrap_or(0);
        println!(
            "  p,q < 2^{bits:<4} fits {:>6.2}%   used bits: median {med:>3}  p99 {p99:>3}",
            fit as f64 / 100.0
        );
    }
}
