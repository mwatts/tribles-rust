//! Remote blob store backed by a network connection.
//!
//! `RemoteStore` implements `BlobStoreGet + BranchStore` over the sync
//! protocol, making a remote node look like a local storage backend.
//!
//! ## Async bridging
//!
//! The triblespace traits are sync. The network is async. `RemoteStore`
//! bridges this by holding a tokio runtime handle and blocking internally.
//!
//! **Do not use from inside an async task** — nested `block_on` panics.
//! This is designed for the sync-over-async pattern where the outermost
//! layer is async (CLI's `Runtime::block_on`) but the library internals
//! are trait-based and sync.

use iroh::endpoint::Connection;
use triblespace_core::blob::{Blob, BlobSchema, TryFromBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobChildren, BlobStoreGet, BlobStoreList, BranchStore, PushResult};
use triblespace_core::value::Value;
use triblespace_core::value::schemas::hash::{Blake3, Handle};
use triblespace_core::value::ValueSchema;

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
}

/// A remote blob/branch store backed by a network connection.
///
/// Implements `BlobStoreGet<Blake3>` and `BranchStore<Blake3>`, so it
/// can be used anywhere a local store is expected.
#[derive(Clone)]
pub struct RemoteStore {
    conn: Connection,
    rt: tokio::runtime::Handle,
}

impl RemoteStore {
    /// Create a new remote store from an existing connection.
    ///
    /// The connection should be established with `PILE_SYNC_ALPN`.
    /// The `rt` handle is used to bridge async I/O with sync traits.
    pub fn new(conn: Connection, rt: tokio::runtime::Handle) -> Self {
        Self { conn, rt }
    }

    /// Fetch a raw blob by hash (one QUIC stream per call).
    fn fetch_blob(&self, hash: &RawHash) -> Result<Vec<u8>, RemoteError> {
        self.rt.block_on(async {
            op_get_blob(&self.conn, hash).await
                .map_err(|e| RemoteError::Network(format!("{e}")))?
                .ok_or(RemoteError::NotFound)
        })
    }
}

impl BlobStoreGet<Blake3> for RemoteStore {
    type GetError<E: std::error::Error + Send + Sync + 'static> = RemoteError;

    fn get<T, S>(
        &self,
        handle: Value<Handle<Blake3, S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobSchema + 'static,
        T: TryFromBlob<S>,
        Handle<Blake3, S>: ValueSchema,
    {
        let data = self.fetch_blob(&handle.raw)?;
        let blob: Blob<S> = Blob::new(data.into());
        T::try_from_blob(blob).map_err(|_| RemoteError::Deserialize("blob deserialization failed".into()))
    }
}

/// A clonable reader snapshot of a remote store.
///
/// Since the remote store is stateless (each request is independent),
/// the reader is just a clone of the store.
#[derive(Clone, PartialEq, Eq)]
pub struct RemoteReader {
    store: RemoteStore,
}

impl PartialEq for RemoteStore {
    fn eq(&self, other: &Self) -> bool {
        self.conn.stable_id() == other.conn.stable_id()
    }
}

impl Eq for RemoteStore {}

impl BlobStoreGet<Blake3> for RemoteReader {
    type GetError<E: std::error::Error + Send + Sync + 'static> = RemoteError;

    fn get<T, S>(
        &self,
        handle: Value<Handle<Blake3, S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobSchema + 'static,
        T: TryFromBlob<S>,
        Handle<Blake3, S>: ValueSchema,
    {
        self.store.get(handle)
    }
}

impl BlobStoreList<Blake3> for RemoteReader {
    type Iter<'a> = std::iter::Empty<Result<Value<Handle<Blake3, UnknownBlob>>, Self::Err>>;
    type Err = RemoteError;

    fn blobs<'a>(&'a self) -> Self::Iter<'a> {
        // Remote listing not supported via this interface.
        // Use list_branches() for branch enumeration.
        std::iter::empty()
    }
}

/// Optimized children enumeration using the SYNC protocol.
///
/// Instead of fetching each candidate individually (N round-trips),
/// sends one SYNC request with an empty HAVE set and receives all
/// children in a single round-trip.
impl BlobChildren<Blake3> for RemoteStore {
    fn children(
        &self,
        handle: Value<Handle<Blake3, UnknownBlob>>,
    ) -> Vec<Value<Handle<Blake3, UnknownBlob>>> {
        self.rt.block_on(async {
            // CHILDREN with empty HAVE set = "give me all children"
            match op_children(&self.conn, &handle.raw, &[]).await {
                Ok(blobs) => blobs.into_iter()
                    .map(|(hash, _data)| Value::<Handle<Blake3, UnknownBlob>>::new(hash))
                    .collect(),
                Err(_) => Vec::new(),
            }
        })
    }
}

impl BlobChildren<Blake3> for RemoteReader {
    fn children(
        &self,
        handle: Value<Handle<Blake3, UnknownBlob>>,
    ) -> Vec<Value<Handle<Blake3, UnknownBlob>>> {
        self.store.children(handle)
    }
}

impl BranchStore<Blake3> for RemoteStore {
    type BranchesError = RemoteError;
    type HeadError = RemoteError;
    type UpdateError = RemoteError;

    type ListIter<'a> = std::vec::IntoIter<Result<Id, RemoteError>>;

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
        _id: Id,
        _old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        _new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> Result<PushResult<Blake3>, Self::UpdateError> {
        // CAS_PUSH not implemented yet.
        Err(RemoteError::Network("remote CAS_PUSH not yet supported".into()))
    }
}
