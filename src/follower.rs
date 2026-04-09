//! `Follower<S>`: store wrapper that receives incoming sync events.
//!
//! Fully sync. `poll()` drains events from the Host, puts blobs,
//! updates tracking refs. No async, no Arc, no background threads.

use std::collections::HashMap;

use anybytes::Bytes;
use triblespace_core::blob::{BlobSchema, ToBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStorePut, BranchStore, PushResult};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::channel::NetEvent;
use crate::host::HostReceiver;
use crate::protocol::{RawHash, RawBranchId};

/// Store wrapper that receives incoming sync events from the Host.
pub struct Follower<S> {
    store: S,
    host: HostReceiver,
    remote_heads: HashMap<RawBranchId, RawHash>,
}

impl<S> Follower<S> {
    pub fn new(store: S, host: HostReceiver) -> Self {
        Self { store, host, remote_heads: HashMap::new() }
    }

    pub fn store(&self) -> &S { &self.store }
    pub fn store_mut(&mut self) -> &mut S { &mut self.store }
    pub fn into_store(self) -> S { self.store }
    pub fn host(&self) -> &HostReceiver { &self.host }

    /// Latest known remote HEAD for a branch (by raw branch ID).
    pub fn remote_head_raw(&self, branch: &RawBranchId) -> Option<RawHash> {
        self.remote_heads.get(branch).copied()
    }

    /// Latest known remote HEAD for a branch (by Id).
    pub fn remote_head(&self, branch_id: Id) -> Option<RawHash> {
        let key: [u8; 16] = branch_id.into();
        self.remote_heads.get(&key).copied()
    }

    /// All known remote branch heads.
    pub fn remote_heads(&self) -> &HashMap<RawBranchId, RawHash> {
        &self.remote_heads
    }
}

impl<S: BlobStorePut<Blake3> + BlobStore<Blake3> + BranchStore<Blake3>> Follower<S> {
    /// Drain pending events: store blobs, create/update tracking branches.
    /// Returns the number of events processed.
    pub fn poll(&mut self) -> usize {
        let mut count = 0;
        while let Some(event) = self.host.try_recv() {
            match event {
                NetEvent::Blob(data) => {
                    let bytes: Bytes = data.into();
                    let _ = self.store.put::<UnknownBlob, Bytes>(bytes);
                }
                NetEvent::Head { branch, head } => {
                    self.remote_heads.insert(branch, head);
                    // Read the branch name from the metadata blob and
                    // create/update a tracking branch.
                    if let Some(remote_id) = triblespace_core::id::Id::new(branch) {
                        if let Some(name) = read_remote_name(&mut self.store, &head) {
                            crate::tracking::ensure_tracking_branch(
                                &mut self.store, remote_id, &head, &name,
                            );
                        }
                    }
                }
            }
            count += 1;
        }
        count
    }
}

/// Read the branch name from a branch metadata blob.
fn read_remote_name<S: BlobStore<Blake3>>(store: &mut S, head_hash: &RawHash) -> Option<String> {
    use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
    use triblespace_core::blob::schemas::longstring::LongString;
    use triblespace_core::repo::BlobStoreGet;
    use triblespace_core::macros::{find, pattern};

    let reader = store.reader().ok()?;
    let meta_handle = Value::<Handle<Blake3, SimpleArchive>>::new(*head_hash);
    let meta: triblespace_core::trible::TribleSet = reader.get(meta_handle).ok()?;

    // Try `name` first (normal branch), then `remote_name` (tracking branch).
    let name_handle: Value<Handle<Blake3, LongString>> = find!(
        h: Value<Handle<Blake3, LongString>>,
        pattern!(&meta, [{ _?e @ triblespace_core::metadata::name: ?h }])
    ).next().or_else(|| find!(
        h: Value<Handle<Blake3, LongString>>,
        pattern!(&meta, [{ _?e @ crate::tracking::remote_name: ?h }])
    ).next())?;

    let name_view: anybytes::View<str> = reader.get(name_handle).ok()?;
    Some(name_view.as_ref().to_string())
}

// ── Trait delegations ────────────────────────────────────────────────

impl<S: BlobStorePut<Blake3>> BlobStorePut<Blake3> for Follower<S> {
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        self.store.put(item)
    }
}

impl<S: BlobStore<Blake3>> BlobStore<Blake3> for Follower<S> {
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.store.reader()
    }
}

impl<S: BranchStore<Blake3>> BranchStore<Blake3> for Follower<S> {
    type BranchesError = S::BranchesError;
    type HeadError = S::HeadError;
    type UpdateError = S::UpdateError;
    type ListIter<'a> = S::ListIter<'a> where S: 'a;

    fn branches<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::BranchesError> {
        self.store.branches()
    }

    fn head(&mut self, id: Id) -> Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        self.store.head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> Result<PushResult<Blake3>, Self::UpdateError> {
        self.store.update(id, old, new)
    }
}
