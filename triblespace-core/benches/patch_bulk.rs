//! Owned key-only PATCH construction: repeated persistent edits versus the
//! canonical bottom-up builder.
//!
//! Run: `cargo bench -p triblespace-core --bench patch_bulk`

use std::hint::black_box;
use std::time::{Duration, Instant};

use triblespace_core::patch::{Blake3Merkle, Entry, IdentitySchema, PATCH};

fn keys<const KEY_LEN: usize>(len: usize) -> Vec<[u8; KEY_LEN]> {
    assert!(KEY_LEN >= 8);
    let mut keys: Vec<_> = (0..len)
        .map(|index| {
            let mut key = [0u8; KEY_LEN];
            key[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let mut state = (index as u64) ^ 0x243f_6a88_85a3_08d3;
            for byte in &mut key[8..] {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                *byte = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8;
            }
            key
        })
        .collect();

    let mut state = 0x1319_8a2e_0370_7344u64;
    for upper in (1..keys.len()).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let at = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as usize % (upper + 1);
        keys.swap(upper, at);
    }
    keys
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn one<const KEY_LEN: usize>(len: usize) {
    type Hash = Blake3Merkle;
    let keys = keys::<KEY_LEN>(len);
    let repetitions = if len < 100_000 { 5 } else { 3 };

    let mut insert_samples = Vec::with_capacity(repetitions);
    let mut bulk_samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        let mut inserted = PATCH::<KEY_LEN, IdentitySchema, (), Hash>::new();
        for key in &keys {
            inserted.insert(&Entry::new(key));
        }
        black_box(inserted.merkle_root());
        insert_samples.push(start.elapsed());
        drop(inserted);

        // Cloning the caller's source buffer is setup, not constructor work.
        let input = keys.clone();
        let start = Instant::now();
        let bulk = PATCH::<KEY_LEN, IdentitySchema, (), Hash>::from_keys(input);
        black_box(bulk.merkle_root());
        bulk_samples.push(start.elapsed());
        drop(bulk);
    }

    let insert = median(insert_samples);
    let bulk = median(bulk_samples);
    let throughput = |duration: Duration| len as f64 / duration.as_secs_f64() / 1_000_000.0;
    println!(
        "key={KEY_LEN:>2} n={len:>6}  insert={:>9.3} ms ({:>6.2} Mkey/s)  bulk={:>9.3} ms ({:>6.2} Mkey/s)  speedup={:>5.2}x",
        insert.as_secs_f64() * 1_000.0,
        throughput(insert),
        bulk.as_secs_f64() * 1_000.0,
        throughput(bulk),
        insert.as_secs_f64() / bulk.as_secs_f64(),
    );
}

fn main() {
    println!("PATCH Blake3Merkle owned-key construction (median, source clone excluded)");
    one::<16>(10_000);
    one::<16>(100_000);
    one::<32>(10_000);
    one::<32>(100_000);
    one::<64>(10_000);
    one::<64>(100_000);
}
