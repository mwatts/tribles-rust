//! `Leader<S>`: store wrapper that sends outgoing network effects.
//!
//! Fully sync. `put()` stores + announces + refreshes reader.
//! `update()` CAS + gossips. All via the Host handle.

use triblespace_core::blob::{BlobSchema, ToBlob};
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStorePut, BranchStore, PushResult};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::host::{HostSender, StoreSnapshot};

/// Store wrapper that sends outgoing network effects via a Host handle.
pub struct Leader<S> {
    inner: S,
    host: HostSender,
}

impl<S> Leader<S> {
    pub fn new(store: S, host: HostSender) -> Self {
        Self { inner: store, host }
    }

    pub fn store(&self) -> &S { &self.inner }
    pub fn store_mut(&mut self) -> &mut S { &mut self.inner }
    pub fn into_store(self) -> S { self.inner }
    pub fn host(&self) -> &HostSender { &self.host }
}

impl<S: BlobStorePut<Blake3> + BlobStore<Blake3> + BranchStore<Blake3>> BlobStorePut<Blake3> for Leader<S> {
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        let handle = self.inner.put(item)?;
        self.host.announce(handle.raw);
        if let Some(snap) = StoreSnapshot::from_store(&mut self.inner) {
            self.host.update_snapshot(snap);
        }
        Ok(handle)
    }
}

impl<S: BlobStore<Blake3> + BranchStore<Blake3> + BlobStorePut<Blake3>> BlobStore<Blake3> for Leader<S> {
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl<S: BranchStore<Blake3> + BlobStore<Blake3> + BlobStorePut<Blake3>> BranchStore<Blake3> for Leader<S> {
    type BranchesError = S::BranchesError;
    type HeadError = S::HeadError;
    type UpdateError = S::UpdateError;
    type ListIter<'a> = S::ListIter<'a> where S: 'a;

    fn branches<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::BranchesError> {
        self.inner.branches()
    }

    fn head(&mut self, id: Id) -> Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        self.inner.head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> Result<PushResult<Blake3>, Self::UpdateError> {
        let result = self.inner.update(id, old, new.clone())?;
        if let PushResult::Success() = &result {
            if let Some(head) = new {
                // Tracking branches are local mirror state and must NOT be
                // re-gossiped — otherwise the publisher would receive its own
                // tracking branch back and create a tracking-of-the-tracking,
                // ad infinitum.
                if !crate::tracking::is_tracking_branch(&mut self.inner, id) {
                    let branch: [u8; 16] = id.into();
                    self.host.gossip(branch, head.raw);
                }
                if let Some(snap) = StoreSnapshot::from_store(&mut self.inner) {
                    self.host.update_snapshot(snap);
                }
            }
        }
        Ok(result)
    }
}
