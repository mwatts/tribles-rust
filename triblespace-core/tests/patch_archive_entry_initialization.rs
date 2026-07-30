use std::ptr::NonNull;
use std::sync::Arc;

use triblespace_core::patch::{ArchiveEntry, ArchiveOwner, Entry, IdentitySchema, PATCH};

type TestPatch = PATCH<16, IdentitySchema, ()>;

#[repr(C, align(16))]
struct AlignedKey([u8; 16]);

#[test]
fn archive_entry_created_before_first_patch_uses_the_process_key() {
    let storage = Arc::new(AlignedKey([0x24; 16]));
    let owner: Arc<dyn ArchiveOwner> = storage.clone();

    // SAFETY: AlignedKey gives the immutable key the required 16-byte
    // alignment, and `owner` retains the allocation through insertion.
    let archive_entry = unsafe { ArchiveEntry::new(NonNull::from(&storage.0), &owner) };
    let mut archive_patch = TestPatch::new();
    archive_patch.insert_archive(&archive_entry);

    let heap_entry = Entry::new(&storage.0);
    let mut heap_patch = TestPatch::new();
    heap_patch.insert(&heap_entry);

    assert_eq!(archive_patch, heap_patch);
}
