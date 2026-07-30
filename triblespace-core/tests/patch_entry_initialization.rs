use triblespace_core::patch::{Entry, IdentitySchema, PATCH};

type TestPatch = PATCH<16, IdentitySchema, ()>;

#[test]
fn entry_created_before_first_patch_uses_the_process_key() {
    let key = [0x42; 16];

    // This order is the regression: Entry is a safe public constructor and
    // must initialize the hash key even when no PATCH exists yet.
    let early_entry = Entry::new(&key);
    let mut early_patch = TestPatch::new();
    early_patch.insert(&early_entry);

    let late_entry = Entry::new(&key);
    let mut late_patch = TestPatch::new();
    late_patch.insert(&late_entry);

    assert_eq!(early_patch, late_patch);
}
