use std::sync::{Arc, Barrier};
use std::thread;

use triblespace_core::patch::{Entry, IdentitySchema, PATCH};

type TestPatch = PATCH<16, IdentitySchema, ()>;

#[test]
fn concurrent_entry_first_use_observes_one_immutable_key() {
    const WORKERS: usize = 16;
    let barrier = Arc::new(Barrier::new(WORKERS));
    let key = [0xa5; 16];

    let threads: Vec<_> = (0..WORKERS)
        .map(|_| {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait();

                // Every worker deliberately constructs its Entry before its
                // first PATCH, racing the process-global initialization seam.
                let entry = Entry::new(&key);
                let mut patch = TestPatch::new();
                patch.insert(&entry);
                patch
            })
        })
        .collect();

    let patches: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().expect("hash-initialization worker panicked"))
        .collect();

    let reference_entry = Entry::new(&key);
    let mut reference = TestPatch::new();
    reference.insert(&reference_entry);
    for patch in patches {
        assert_eq!(patch, reference);
    }
}
