//! `Leader<S>`: store wrapper that sends outgoing network effects.
//!
//! Fully sync. `put()` stores locally + sends Announce command.
//! `update()` CAS locally + sends Gossip command. Channel sends
//! are non-blocking — the Host thread processes them asynchronously.

use std::sync::mpsc;

use triblespace_core::blob::{BlobSchema, ToBlob};
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStorePut, BranchStore, PushResult};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::channel::NetCommand;

/// Store wrapper that sends outgoing network effects.
pub struct Leader<S> {
    inner: S,
    commands: mpsc::Sender<NetCommand>,
}

impl<S> Leader<S> {
    pub fn new(store: S, commands: mpsc::Sender<NetCommand>) -> Self {
        Self { inner: store, commands }
    }

    /// Access the inner store.
    pub fn store(&self) -> &S { &self.inner }

    /// Access the inner store mutably.
    pub fn store_mut(&mut self) -> &mut S { &mut self.inner }

    /// Consume the leader, return the inner store.
    pub fn into_store(self) -> S { self.inner }
}

// ── Trait delegations with side effects ──────────────────────────────

impl<S: BlobStorePut<Blake3>> BlobStorePut<Blake3> for Leader<S> {
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        let handle = self.inner.put(item)?;
        let _ = self.commands.send(NetCommand::Announce(handle.raw));
        Ok(handle)
    }
}

impl<S: BlobStore<Blake3>> BlobStore<Blake3> for Leader<S> {
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl<S: BranchStore<Blake3>> BranchStore<Blake3> for Leader<S> {
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
                let branch: [u8; 16] = id.into();
                let _ = self.commands.send(NetCommand::Gossip { branch, head: head.raw });
            }
        }
        Ok(result)
    }
}
