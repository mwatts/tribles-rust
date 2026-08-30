use anybytes::Bytes;
use std::collections::HashSet;

use triblespace::core::blob::encodings::UnknownBlob;
use triblespace::core::blob::{Blob, MemoryBlobStore};
use triblespace::core::inline::INLINE_LEN;
use triblespace::core::repo::{reachable, transfer, BlobStoreGet, SnapshotSource};

#[test]
fn reachable_keep_and_transfer() {
    let mut source = MemoryBlobStore::new();

    // Insert a child blob that will be referenced by the root.
    let child_blob = Blob::<UnknownBlob>::new(Bytes::from(vec![1u8; INLINE_LEN * 2]));
    let child_handle = source.insert(child_blob);

    // Insert an orphan blob that should be dropped by keep.
    let orphan_blob = Blob::<UnknownBlob>::new(Bytes::from(vec![2u8; INLINE_LEN * 2]));
    let orphan_handle = source.insert(orphan_blob);

    // Root blob references the child handle in its first 32-byte slot.
    let mut root_bytes = Vec::with_capacity(INLINE_LEN * 2);
    root_bytes.extend_from_slice(&child_handle.raw);
    root_bytes.extend_from_slice(&[0u8; INLINE_LEN]);
    let root_blob = Blob::<UnknownBlob>::new(Bytes::from(root_bytes));
    let root_handle = source.insert(root_blob);

    // Retain only blobs reachable from the root handle.
    let snapshot = source.snapshot().expect("snapshot");
    source.keep(reachable(&snapshot, [root_handle]));

    let refreshed = source.snapshot().expect("refreshed snapshot");
    assert!(refreshed
        .get::<Blob<UnknownBlob>, UnknownBlob>(root_handle)
        .is_ok());
    assert!(refreshed
        .get::<Blob<UnknownBlob>, UnknownBlob>(child_handle)
        .is_ok());
    assert!(refreshed
        .get::<Blob<UnknownBlob>, UnknownBlob>(orphan_handle)
        .is_err());

    // Copy only the handles reported by the reachable walker into a fresh store.
    let snapshot = source.snapshot().expect("post-keep snapshot");
    let mut target = MemoryBlobStore::new();
    let copied = transfer(&snapshot, &mut target, reachable(&snapshot, [root_handle]))
        .collect::<Result<Vec<_>, _>>()
        .expect("transfer handles");

    let copied_handles: HashSet<_> = copied.iter().map(|(old, _)| *old).collect();
    assert_eq!(copied_handles.len(), 2);
    assert!(copied_handles.contains(&root_handle));
    assert!(copied_handles.contains(&child_handle));

    let target_snapshot = target.snapshot().expect("target snapshot");
    assert_eq!(target_snapshot.len(), 2);
    assert!(target_snapshot
        .get::<Blob<UnknownBlob>, UnknownBlob>(root_handle)
        .is_ok());
    assert!(target_snapshot
        .get::<Blob<UnknownBlob>, UnknownBlob>(child_handle)
        .is_ok());
}
