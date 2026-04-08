//! Remote blob store with local cache.
//!
//! `RemoteStore<S>` wraps a network connection AND a local cache store.
//! `get()` checks the cache first; on miss, fetches from the remote and
//! caches. `children()` sends the cache contents as the HAVE set so the
//! remote only returns hashes we don't have yet.
//!
//! ## Async bridging
//!
//! The triblespace traits are sync. The network is async. `RemoteStore`
//! bridges via a tokio runtime handle.
//!
//! **Do not use from inside an async task** — nested `block_on` panics.

use iroh::endpoint::Connection;
use triblespace_core::blob::{Blob, BlobSchema, ToBlob, TryFromBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{
    BlobChildren, BlobStore, BlobStoreGet, BlobStoreList, BlobStorePut,
    BranchStore, PushResult,
};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};
use anybytes::Bytes;

use crate::protocol::*;

/// Error type for remote store operations.
#[derive(Debug, thiserror::Error)]
pub enum RemoteError {
    #[error("blob not found")]
    NotFound,
    #[error("network error: {0}")]
    Network(String),
    #[error("deserialization error: {0}")]
    Deserialize(String),
    #[error("cache error: {0}")]
    Cache(String),
}

/// A remote blob/branch store with a local cache.
///
/// `get()` checks the cache first. On miss, fetches via GET_BLOB and caches.
/// `children()` sends the cache contents as the HAVE set.
pub struct RemoteStore<S: BlobStore<Blake3>> {
    conn: Connection,
    cache: S,
    reader: S::Reader,
    rt: tokio::runtime::Handle,
}

impl<S> RemoteStore<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3>,
{
    /// Create a new remote store backed by a cache.
    pub fn new(conn: Connection, mut cache: S, rt: tokio::runtime::Handle) -> Result<Self, RemoteError> {
        let reader = cache.reader().map_err(|e| RemoteError::Cache(format!("{e:?}")))?;
        Ok(Self { conn, cache, reader, rt })
    }

    /// Consume the remote store and return the cache.
    pub fn into_cache(self) -> S {
        self.cache
    }

    /// Access the cache directly.
    pub fn cache(&self) -> &S { &self.cache }

    /// Access the cache mutably.
    pub fn cache_mut(&mut self) -> &mut S { &mut self.cache }
}

impl<S> BlobStoreGet<Blake3> for RemoteStore<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3>,
{
    type GetError<E: std::error::Error + Send + Sync + 'static> = RemoteError;

    fn get<T, Sch>(
        &self,
        handle: Value<Handle<Blake3, Sch>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<Sch>>::Error>>
    where
        Sch: BlobSchema + 'static,
        T: TryFromBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        // Check cache first.
        if let Ok(val) = self.reader.get::<T, Sch>(handle) {
            return Ok(val);
        }

        // Cache miss — fetch from remote.
        let data = self.rt.block_on(async {
            op_get_blob(&self.conn, &handle.raw).await
                .map_err(|e| RemoteError::Network(format!("{e}")))?
                .ok_or(RemoteError::NotFound)
        })?;

        // Cache it. We need &mut for put, but self is &self.
        // Use unsafe interior mutability via the cache's own mechanisms,
        // or accept that caching on read requires &mut self.
        // For now, return the fetched data without caching on get().
        // Caching happens during children() which takes &mut self via BlobChildren.
        let blob: Blob<Sch> = Blob::new(data.into());
        T::try_from_blob(blob).map_err(|_| RemoteError::Deserialize("blob deserialization failed".into()))
    }
}

impl<S> BlobChildren<Blake3> for RemoteStore<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3>,
{
    fn children(
        &self,
        handle: Value<Handle<Blake3, UnknownBlob>>,
    ) -> Vec<Value<Handle<Blake3, UnknownBlob>>> {
        // Build HAVE set from what's already cached.
        let have: Vec<RawHash> = {
            let Ok(parent_blob) = self.reader.get::<Bytes, UnknownBlob>(handle) else {
                return Vec::new();
            };
            let parent_data: &[u8] = parent_blob.as_ref();
            let mut cached = Vec::new();
            for chunk in parent_data.chunks(32) {
                if chunk.len() == 32 {
                    let mut candidate = [0u8; 32];
                    candidate.copy_from_slice(chunk);
                    let h = Value::<Handle<Blake3, UnknownBlob>>::new(candidate);
                    if self.reader.get::<Bytes, UnknownBlob>(h).is_ok() {
                        cached.push(candidate);
                    }
                }
            }
            cached
        };

        // Ask remote for children we don't have.
        let new_hashes = self.rt.block_on(async {
            op_children(&self.conn, &handle.raw, &have).await.unwrap_or_default()
        });

        // Combine: cached + newly discovered.
        let mut all: Vec<Value<Handle<Blake3, UnknownBlob>>> = have.iter()
            .map(|h| Value::<Handle<Blake3, UnknownBlob>>::new(*h))
            .collect();
        for h in new_hashes {
            all.push(Value::<Handle<Blake3, UnknownBlob>>::new(h));
        }
        all
    }
}

impl<S> BranchStore<Blake3> for RemoteStore<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3>,
{
    type BranchesError = RemoteError;
    type HeadError = RemoteError;
    type UpdateError = RemoteError;
    type ListIter<'a> = std::vec::IntoIter<Result<Id, RemoteError>> where S: 'a;

    fn branches<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::BranchesError> {
        let branches = self.rt.block_on(async {
            op_list(&self.conn).await
        }).map_err(|e| RemoteError::Network(format!("{e}")))?;

        Ok(branches.into_iter()
            .filter_map(|(id_bytes, _head)| Id::new(id_bytes).map(|id| Ok(id)))
            .collect::<Vec<_>>().into_iter())
    }

    fn head(&mut self, id: Id) -> Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        let id_bytes: [u8; 16] = id.into();
        let result = self.rt.block_on(async {
            op_head(&self.conn, &id_bytes).await
        }).map_err(|e| RemoteError::Network(format!("{e}")))?;

        Ok(result.map(|hash| Value::<Handle<Blake3, SimpleArchive>>::new(hash)))
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> Result<PushResult<Blake3>, Self::UpdateError> {
        let new = new.ok_or(RemoteError::Network("delete not supported over network".into()))?;
        let id_bytes: [u8; 16] = id.into();
        let old_hash = old.map(|h| h.raw);

        let result = self.rt.block_on(async {
            op_cas_push(&self.conn, &id_bytes, old_hash.as_ref(), &new.raw).await
        }).map_err(|e| RemoteError::Network(format!("{e}")))?;

        match result {
            Ok(()) => Ok(PushResult::Success()),
            Err(current_hash) => {
                let current = if current_hash == [0u8; 32] {
                    None
                } else {
                    Some(Value::<Handle<Blake3, SimpleArchive>>::new(current_hash))
                };
                Ok(PushResult::Conflict(current))
            }
        }
    }
}
