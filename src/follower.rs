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

impl<S: BlobStorePut<Blake3>> Follower<S> {
    /// Drain pending events: store blobs, update tracking refs.
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
                }
            }
            count += 1;
        }
        count
    }
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
