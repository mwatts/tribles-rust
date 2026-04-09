//! Tracking branch management.
//!
//! A tracking branch is a local reification of a remote branch. It uses
//! `remote_name` instead of `name` in its metadata, making it invisible
//! to normal operations (ensure_branch, resolve_branch_name, faculties).
//!
//! The tracking branch has its own local ID. Repository can pull/merge
//! it like any other branch.

use triblespace_core::blob::schemas::longstring::LongString;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::{Id, genid};
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut, BranchStore, PushResult};
use triblespace_core::trible::TribleSet;
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};
use triblespace_core::prelude::valueschemas::GenId;
use triblespace_core::prelude::attributes;
use triblespace_core::macros::{find, pattern, entity};

use crate::protocol::{RawHash, RawBranchId};

// Minted attribute IDs for tracking branches.
attributes! {
    "FD45B98C108B3F9F2D18C0B5373BC9FB" as pub remote_name: Handle<Blake3, LongString>;
    "ACEBAE99F0B5B1E12DAE3FDC1E2BC575" as pub tracking_remote_branch: GenId;
}

/// Find a tracking branch for the given remote branch ID.
/// Returns the local tracking branch ID if found.
pub fn find_tracking_branch<S>(
    store: &mut S,
    remote_branch_id: Id,
) -> Option<Id>
where
    S: BlobStore<Blake3> + BranchStore<Blake3>,
{
    let branch_ids: Vec<Id> = store.branches().ok()?.filter_map(|r| r.ok()).collect();
    for bid in branch_ids {
        let head_handle = store.head(bid).ok()??;
        let reader = store.reader().ok()?;
        let meta: TribleSet = reader.get(head_handle).ok()?;
        let tracks: Option<Id> = find!(
            v: Id,
            pattern!(&meta, [{ _?e @ tracking_remote_branch: ?v }])
        ).next();
        if tracks == Some(remote_branch_id) {
            return Some(bid);
        }
    }
    None
}

/// Create a new tracking branch. Returns the local tracking branch ID.
pub fn create_tracking_branch<S>(
    store: &mut S,
    remote_branch_id: Id,
    remote_head_hash: &RawHash,
    remote_name_str: &str,
) -> Option<Id>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    let tracking_eid = genid();
    let tracking_id: Id = tracking_eid.id;

    // Store the remote name as a blob.
    let name_string = remote_name_str.to_string();
    let name_handle: Value<Handle<Blake3, LongString>> = store.put::<LongString, String>(name_string).ok()?;

    // Build tracking branch metadata with content-derived entity ID.
    let head_handle = Value::<Handle<Blake3, SimpleArchive>>::new(*remote_head_hash);

    let meta = entity! { &tracking_eid @
        triblespace_core::repo::branch: tracking_id,
        triblespace_core::repo::head: head_handle,
        remote_name: name_handle,
        tracking_remote_branch: remote_branch_id,
    };

    let meta_set: TribleSet = meta.into();
    let meta_handle: Value<Handle<Blake3, SimpleArchive>> = store.put(meta_set).ok()?;

    // Create the branch.
    match store.update(tracking_id, None, Some(meta_handle)).ok()? {
        PushResult::Success() => Some(tracking_id),
        PushResult::Conflict(_) => None,
    }
}

/// Update a tracking branch's head.
pub fn update_tracking_branch<S>(
    store: &mut S,
    tracking_branch_id: Id,
    remote_branch_id: Id,
    new_head_hash: &RawHash,
    remote_name_str: &str,
) -> Option<()>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    let old_meta = store.head(tracking_branch_id).ok()??;

    let name_string = remote_name_str.to_string();
    let name_handle: Value<Handle<Blake3, LongString>> = store.put::<LongString, String>(name_string).ok()?;
    let head_handle = Value::<Handle<Blake3, SimpleArchive>>::new(*new_head_hash);

    let eid = genid();
    let meta = entity! { &eid @
        triblespace_core::repo::branch: tracking_branch_id,
        triblespace_core::repo::head: head_handle,
        remote_name: name_handle,
        tracking_remote_branch: remote_branch_id,
    };
    let meta_set: TribleSet = meta.into();

    let meta_handle: Value<Handle<Blake3, SimpleArchive>> = store.put(meta_set).ok()?;

    match store.update(tracking_branch_id, Some(old_meta), Some(meta_handle)).ok()? {
        PushResult::Success() => Some(()),
        PushResult::Conflict(_) => None,
    }
}

/// Find or create a tracking branch. Returns the local tracking branch ID.
pub fn ensure_tracking_branch<S>(
    store: &mut S,
    remote_branch_id: Id,
    remote_head_hash: &RawHash,
    remote_name_str: &str,
) -> Option<Id>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    if let Some(tracking_id) = find_tracking_branch(store, remote_branch_id) {
        update_tracking_branch(store, tracking_id, remote_branch_id, remote_head_hash, remote_name_str);
        Some(tracking_id)
    } else {
        create_tracking_branch(store, remote_branch_id, remote_head_hash, remote_name_str)
    }
}
