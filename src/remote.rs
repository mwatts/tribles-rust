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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anybytes::Bytes;
use iroh::endpoint::Connection;
use triblespace_core::blob::{Blob, BlobSchema, TryFromBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStoreGet, BlobStoreList, BranchStore, PushResult};
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

    /// Fetch a raw blob by hash over the network.
    fn fetch_blob(&self, hash: &RawHash) -> Result<Vec<u8>, RemoteError> {
        self.rt.block_on(async {
            let (mut send, mut recv) = self.conn.open_bi().await
                .map_err(|e| RemoteError::Network(format!("open_bi: {e}")))?;

            send_u8(&mut send, REQ_GET_BLOB).await
                .map_err(|e| RemoteError::Network(format!("{e}")))?;
            send_hash(&mut send, hash).await
                .map_err(|e| RemoteError::Network(format!("{e}")))?;

            let rsp = recv_u8(&mut recv).await
                .map_err(|e| RemoteError::Network(format!("{e}")))?;

            match rsp {
                RSP_BLOB => {
                    let (_h, data) = recv_blob_data(&mut recv).await
                        .map_err(|e| RemoteError::Network(format!("{e}")))?;
                    send_u8(&mut send, REQ_DONE).await.ok();
                    send.finish().ok();
                    Ok(data)
                }
                RSP_MISSING => {
                    send_u8(&mut send, REQ_DONE).await.ok();
                    send.finish().ok();
                    Err(RemoteError::NotFound)
                }
                _ => Err(RemoteError::Network(format!("unexpected response: {rsp}"))),
            }
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

impl BranchStore<Blake3> for RemoteStore {
    type BranchesError = RemoteError;
    type HeadError = RemoteError;
    type UpdateError = RemoteError;

    type ListIter<'a> = std::vec::IntoIter<Result<Id, RemoteError>>;

    fn branches<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::BranchesError> {
        let branches = self.rt.block_on(async {
            crate::sync::list_branches(&self.conn).await
        }).map_err(|e| RemoteError::Network(format!("{e}")))?;

        Ok(branches.into_iter().map(|(id, _head)| Ok(id)).collect::<Vec<_>>().into_iter())
    }

    fn head(&mut self, id: Id) -> Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        let result = self.rt.block_on(async {
            crate::sync::remote_head(&self.conn, id).await
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
