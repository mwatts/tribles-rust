use std::array::TryFromSliceError;
use std::convert::Infallible;
use std::convert::TryInto;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use anybytes::Bytes;
use futures::StreamExt;

use object_store::parse_url;
use object_store::path::Path;
use object_store::ObjectStore;
use object_store::PutMode;
use object_store::UpdateVersion;
use object_store::{self};
use url::Url;

use hex::FromHex;

use crate::blob::Blob;
use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::blob::TryFromBlob;
use crate::collection::{CollectionRecord, RecordDecodeError};
use crate::id::Id;
use crate::id::RawId;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::Inline;
use crate::inline::InlineEncoding;
use crate::inline::RawInline;
use crate::local_cell::AsyncLocalCellStore;
use crate::prelude::blobencodings::SimpleArchive;

use super::async_store::{
    AsyncBlobStore, AsyncBlobStoreForget, AsyncBlobStoreGet, AsyncBlobStoreList,
    AsyncBlobStoreMeta, AsyncBlobStorePut, AsyncCollectionStore, AsyncPinStore,
};
use super::PushResult;
use super::{BlobInfo, BlobMetadata};

const BRANCH_INFIX: &str = "branches";
const BLOB_INFIX: &str = "blobs";
const COLLECTION_RECORD_INFIX: &str = "collection-records";
const LOCAL_CELL_INFIX: &str = "local-cells";

/// Repository backed by an [`object_store`] compatible storage backend.
///
/// All data is stored in an external service (e.g. S3, local filesystem)
/// via the `object_store` crate, which is async at its core — so this
/// type is **async-native**: it implements the
/// [`AsyncBlobStore`] family
/// directly, awaiting each operation, with no owned runtime.
///
/// Synchronous callers wrap it in
/// [`Blocking`](super::async_store::Blocking), which carries the single
/// `block_on` boundary:
///
/// ```no_run
/// # use url::Url;
/// # use triblespace_core::repo::objectstore::ObjectStoreRemote;
/// # use triblespace_core::repo::async_store::Blocking;
/// # fn f(url: &Url) -> Result<(), Box<dyn std::error::Error>> {
/// let remote = ObjectStoreRemote::with_url(url)?;
/// let mut store = Blocking::new(remote)?; // now a plain sync BlobStore
/// # let _ = &mut store;
/// # Ok(())
/// # }
/// ```
pub struct ObjectStoreRemote {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
}

impl fmt::Debug for ObjectStoreRemote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectStoreRemote")
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl fmt::Debug for ObjectStoreReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectStoreReader")
            .field("prefix", &self.prefix)
            .finish()
    }
}

/// Read-only handle into an [`ObjectStoreRemote`] that can be cloned and
/// shared.
#[derive(Clone)]
pub struct ObjectStoreReader {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
}

impl PartialEq for ObjectStoreReader {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.store, &other.store) && self.prefix == other.prefix
    }
}

impl Eq for ObjectStoreReader {}

impl ObjectStoreRemote {
    /// Creates a repository pointing at the object store described by
    /// `url`. The returned value is async-native — wrap it in
    /// [`Blocking`](super::async_store::Blocking) for synchronous use.
    pub fn with_url(url: &Url) -> Result<ObjectStoreRemote, object_store::Error> {
        let (store, path) = parse_url(url)?;
        Ok(ObjectStoreRemote {
            store: Arc::from(store),
            prefix: path,
        })
    }
}

impl AsyncBlobStorePut for ObjectStoreRemote {
    type PutError = object_store::Error;

    fn put<S, T>(
        &mut self,
        item: T,
    ) -> impl Future<Output = Result<Inline<Handle<S>>, Self::PutError>> + Send
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        // Serialise + capture only Send primitives before the await (the
        // phantom-typed handle is `!Send` when the schema is).
        let blob = item.to_blob();
        let raw = blob.get_handle().raw;
        let bytes: bytes::Bytes = blob.bytes.into();
        async move {
            let path = self.prefix.child(BLOB_INFIX).child(hex::encode(raw));
            let result = self
                .store
                .put_opts(&path, bytes.into(), PutMode::Create.into())
                .await;
            match result {
                Ok(_) | Err(object_store::Error::AlreadyExists { .. }) => Ok(Inline::new(raw)),
                Err(e) => Err(e),
            }
        }
    }
}

impl AsyncBlobStore for ObjectStoreRemote {
    type Reader = ObjectStoreReader;
    type ReaderError = Infallible;

    fn reader(&mut self) -> impl Future<Output = Result<Self::Reader, Self::ReaderError>> + Send {
        let reader = ObjectStoreReader {
            store: self.store.clone(),
            prefix: self.prefix.clone(),
        };
        async move { Ok(reader) }
    }
}

impl AsyncCollectionStore for ObjectStoreRemote {
    type RecordsError = ListCollectionRecordsErr;
    type InsertError = InsertCollectionRecordErr;

    fn records(
        &mut self,
    ) -> impl Future<
        Output = Result<Vec<Result<CollectionRecord, Self::RecordsError>>, Self::RecordsError>,
    > + Send {
        async move {
            let prefix = self.prefix.child(COLLECTION_RECORD_INFIX);
            let mut observed = Vec::new();

            // Object-store LIST is an observed monotone view, not a coherent
            // snapshot. A concurrent immutable insertion may be visible on
            // this call or the next one; every object this call does observe
            // is nevertheless validated before it is returned.
            let listed = self.store.list(Some(&prefix)).collect::<Vec<_>>().await;
            for item in listed {
                let (sort_id, sort_path, result) = match item {
                    Err(error) => {
                        let sort_path = error.to_string();
                        (None, sort_path, Err(ListCollectionRecordsErr::List(error)))
                    }
                    Ok(meta) => {
                        let sort_path = meta.location.to_string();
                        let path_id = collection_record_id_from_path(&prefix, &meta.location).ok();
                        let result =
                            read_collection_record(&*self.store, &prefix, meta.location).await;
                        let sort_id = result.as_ref().map(CollectionRecord::id).ok().or(path_id);
                        (sort_id, sort_path, result)
                    }
                };
                observed.push((sort_id, sort_path, result));
            }

            // Remote LIST order is backend-specific. Normalize successful
            // records by intrinsic id, and malformed entries by path, so the
            // observed view is deterministic independent of provider order.
            observed.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
            Ok(observed.into_iter().map(|(_, _, result)| result).collect())
        }
    }

    fn insert(
        &mut self,
        record: CollectionRecord,
    ) -> impl Future<Output = Result<(), Self::InsertError>> + Send {
        let id = record.id();
        let path = self
            .prefix
            .child(COLLECTION_RECORD_INFIX)
            .child(hex::encode(id));
        let expected: bytes::Bytes = CollectionRecord::to_blob(&record).bytes.into();

        async move {
            match self
                .store
                .put_opts(&path, expected.clone().into(), PutMode::Create.into())
                .await
            {
                Ok(_) => Ok(()),
                Err(object_store::Error::AlreadyExists { .. }) => {
                    // The namespace is immutable. A replay is success only
                    // when the already-present canonical bytes are identical;
                    // never turn insertion into a mutable CAS update.
                    let object = self
                        .store
                        .get(&path)
                        .await
                        .map_err(InsertCollectionRecordErr::ReadExisting)?;
                    let actual = object
                        .bytes()
                        .await
                        .map_err(InsertCollectionRecordErr::ReadExisting)?;
                    if actual == expected {
                        Ok(())
                    } else {
                        Err(InsertCollectionRecordErr::ExistingMismatch { id })
                    }
                }
                Err(error) => Err(InsertCollectionRecordErr::Store(error)),
            }
        }
    }
}

impl AsyncLocalCellStore for ObjectStoreRemote {
    type CellError = ObjectCellError;

    fn cell(
        &mut self,
        id: Id,
    ) -> impl Future<Output = Result<Option<Inline<Handle<SimpleArchive>>>, Self::CellError>> + Send
    {
        async move {
            let path = self.prefix.child(LOCAL_CELL_INFIX).child(hex::encode(id));
            match self.store.get(&path).await {
                Ok(object) => {
                    let bytes = object.bytes().await.map_err(ObjectCellError::Store)?;
                    if bytes.is_empty() {
                        return Ok(None);
                    }
                    let raw: RawInline = (&bytes[..])
                        .try_into()
                        .map_err(|_| ObjectCellError::InvalidLength(bytes.len()))?;
                    Ok(Some(Inline::new(raw)))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(error) => Err(ObjectCellError::Store(error)),
            }
        }
    }

    fn set_cell(
        &mut self,
        id: Id,
        value: Option<Inline<Handle<SimpleArchive>>>,
    ) -> impl Future<Output = Result<(), Self::CellError>> + Send {
        let bytes = value
            .map(|value| bytes::Bytes::copy_from_slice(&value.raw))
            .unwrap_or_default();
        async move {
            let path = self.prefix.child(LOCAL_CELL_INFIX).child(hex::encode(id));
            self.store
                .put_opts(&path, bytes.into(), PutMode::Overwrite.into())
                .await
                .map_err(ObjectCellError::Store)?;
            Ok(())
        }
    }
}

impl AsyncPinStore for ObjectStoreRemote {
    type PinsError = ListBranchesErr;
    type HeadError = PullBranchErr;
    type UpdateError = PushBranchErr;

    fn pins(
        &mut self,
    ) -> impl Future<Output = Result<Vec<Result<Id, Self::PinsError>>, Self::PinsError>> + Send
    {
        async move {
            let prefix = self.prefix.child(BRANCH_INFIX);
            let stream = self.store.list(Some(&prefix)).filter_map(|r| async move {
                match r {
                    Ok(meta) if meta.size == 0 => None, // tombstoned branch (0-byte object)
                    Ok(meta) => {
                        let name = match meta.location.filename() {
                            Some(name) => name,
                            None => return Some(Err(ListBranchesErr::NotAFile("no filename"))),
                        };
                        let digest = match RawId::from_hex(name) {
                            Ok(digest) => digest,
                            Err(e) => return Some(Err(ListBranchesErr::BadNameHex(e))),
                        };
                        let Some(id) = Id::new(digest) else {
                            return Some(Err(ListBranchesErr::BadId));
                        };
                        Some(Ok(id))
                    }
                    Err(e) => Some(Err(ListBranchesErr::List(e))),
                }
            });
            Ok(stream.collect().await)
        }
    }

    fn head(
        &mut self,
        id: Id,
    ) -> impl Future<Output = Result<Option<Inline<Handle<SimpleArchive>>>, Self::HeadError>> + Send
    {
        async move {
            let path = self.prefix.child(BRANCH_INFIX).child(hex::encode(id));
            match self.store.get(&path).await {
                Ok(object) => {
                    let bytes = object.bytes().await?;
                    if bytes.is_empty() {
                        return Ok(None);
                    }
                    let value = (&bytes[..]).try_into()?;
                    Ok(Some(Inline::new(value)))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(PullBranchErr::StoreErr(e)),
            }
        }
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Inline<Handle<SimpleArchive>>>,
        new: Option<Inline<Handle<SimpleArchive>>>,
    ) -> impl Future<Output = Result<PushResult, Self::UpdateError>> + Send {
        async move {
            let path = self.prefix.child(BRANCH_INFIX).child(hex::encode(id));
            // We encode "deleted branch" as an empty object. This lets us
            // preserve CAS semantics for delete via conditional PUT
            // (PutMode::Update), since `object_store` does not currently
            // expose conditional delete.
            //
            // TODO: Once `object_store` supports conditional delete,
            // migrate away from 0-byte tombstones and treat empty objects
            // as corruption.
            let new_bytes = match new {
                Some(new) => bytes::Bytes::copy_from_slice(&new.raw),
                None => bytes::Bytes::new(),
            };

            let parse_branch = |bytes: &bytes::Bytes| -> Result<
                Option<Inline<Handle<SimpleArchive>>>,
                TryFromSliceError,
            > {
                if bytes.is_empty() {
                    return Ok(None);
                }
                let value = (&bytes[..]).try_into()?;
                Ok(Some(Inline::new(value)))
            };

            if let Some(old_hash) = old {
                let mut result = self.store.get(&path).await;
                loop {
                    match result {
                        Ok(obj) => {
                            let version = UpdateVersion {
                                e_tag: obj.meta.e_tag.clone(),
                                version: obj.meta.version.clone(),
                            };
                            let stored_bytes = obj.bytes().await?;
                            let stored_hash = parse_branch(&stored_bytes)?;
                            if stored_hash != Some(old_hash) {
                                return Ok(PushResult::Conflict(stored_hash));
                            }
                            match self
                                .store
                                .put_opts(
                                    &path,
                                    new_bytes.clone().into(),
                                    PutMode::Update(version).into(),
                                )
                                .await
                            {
                                Ok(_) => return Ok(PushResult::Success()),
                                Err(object_store::Error::Precondition { .. }) => {
                                    result = self.store.get(&path).await;
                                    continue;
                                }
                                Err(e) => return Err(PushBranchErr::StoreErr(e)),
                            }
                        }
                        Err(object_store::Error::NotFound { .. }) => {
                            return Ok(PushResult::Conflict(None))
                        }
                        Err(e) => return Err(PushBranchErr::StoreErr(e)),
                    }
                }
            } else {
                loop {
                    match self
                        .store
                        .put_opts(&path, new_bytes.clone().into(), PutMode::Create.into())
                        .await
                    {
                        Ok(_) => return Ok(PushResult::Success()),
                        Err(object_store::Error::AlreadyExists { .. }) => {
                            let mut result = self.store.get(&path).await;
                            loop {
                                match result {
                                    Ok(obj) => {
                                        let version = UpdateVersion {
                                            e_tag: obj.meta.e_tag.clone(),
                                            version: obj.meta.version.clone(),
                                        };
                                        let stored_bytes = obj.bytes().await?;
                                        let stored_hash = parse_branch(&stored_bytes)?;
                                        if stored_hash.is_some() {
                                            return Ok(PushResult::Conflict(stored_hash));
                                        }
                                        match self
                                            .store
                                            .put_opts(
                                                &path,
                                                new_bytes.clone().into(),
                                                PutMode::Update(version).into(),
                                            )
                                            .await
                                        {
                                            Ok(_) => return Ok(PushResult::Success()),
                                            Err(object_store::Error::Precondition { .. }) => {
                                                result = self.store.get(&path).await;
                                                continue;
                                            }
                                            Err(e) => return Err(PushBranchErr::StoreErr(e)),
                                        }
                                    }
                                    // raced with delete; retry create
                                    Err(object_store::Error::NotFound { .. }) => break,
                                    Err(e) => return Err(PushBranchErr::StoreErr(e)),
                                }
                            }
                            continue;
                        }
                        Err(e) => return Err(PushBranchErr::StoreErr(e)),
                    }
                }
            }
        }
    }
}

impl AsyncBlobStoreForget for ObjectStoreRemote {
    type ForgetError = object_store::Error;

    fn forget<S>(
        &mut self,
        handle: Inline<Handle<S>>,
    ) -> impl Future<Output = Result<(), Self::ForgetError>> + Send
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let raw = handle.raw;
        async move {
            let path = self.prefix.child(BLOB_INFIX).child(hex::encode(raw));
            match self.store.delete(&path).await {
                Ok(_) => Ok(()),
                Err(object_store::Error::NotFound { .. }) => Ok(()),
                Err(e) => Err(e),
            }
        }
    }
}

impl crate::repo::StorageClose for ObjectStoreRemote {
    type Error = Infallible;

    fn close(self) -> Result<(), Self::Error> {
        // No explicit close necessary for the remote object store adapter.
        Ok(())
    }
}

impl ObjectStoreReader {
    fn blob_path(&self, handle_hex: String) -> Path {
        self.prefix.child(BLOB_INFIX).child(handle_hex)
    }
}

impl AsyncBlobStoreGet for ObjectStoreReader {
    type GetError<E: Error + Send + Sync + 'static> = GetBlobErr<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> impl Future<Output = Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>> + Send
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let raw = handle.raw;
        async move {
            let path = self.blob_path(hex::encode(raw));
            let object = self.store.get(&path).await?;
            let bytes = object.bytes().await?;
            let bytes: Bytes = bytes.into();
            let blob: Blob<S> = Blob::new(bytes);
            let expected = Inline::<Hash<Blake3>>::new(raw);
            let actual = blob.get_handle().into();
            if actual != expected {
                return Err(GetBlobErr::HashMismatch { expected, actual });
            }
            blob.try_from_blob().map_err(GetBlobErr::Conversion)
        }
    }
}

impl AsyncBlobStoreList for ObjectStoreReader {
    type Err = ListBlobsErr;

    fn blobs(&self) -> impl Future<Output = Vec<Result<BlobInfo, Self::Err>>> + Send {
        async move {
            let prefix = self.prefix.child(BLOB_INFIX);
            let stream = self.store.list(Some(&prefix)).map(|r| match r {
                Ok(meta) => {
                    let blob_name = meta
                        .location
                        .filename()
                        .ok_or(ListBlobsErr::NotAFile("no filename"))?;
                    let digest =
                        RawInline::from_hex(blob_name).map_err(ListBlobsErr::BadNameHex)?;
                    Ok(BlobInfo {
                        handle: Inline::new(digest),
                        length: meta.size,
                    })
                }
                Err(e) => Err(ListBlobsErr::List(e)),
            });
            stream.collect().await
        }
    }
}

impl AsyncBlobStoreMeta for ObjectStoreReader {
    type MetaError = object_store::Error;

    fn metadata<S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> impl Future<Output = Result<Option<BlobMetadata>, Self::MetaError>> + Send
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let raw = handle.raw;
        async move {
            let path = self.prefix.child(BLOB_INFIX).child(hex::encode(raw));
            match self.store.head(&path).await {
                Ok(meta) => {
                    let ts = meta.last_modified.timestamp_millis() as u64;
                    let len = meta.size;
                    Ok(Some(BlobMetadata {
                        timestamp: ts,
                        length: len,
                    }))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => Err(e),
            }
        }
    }
}

fn collection_record_id_from_path(
    prefix: &Path,
    location: &Path,
) -> Result<Id, ListCollectionRecordsErr> {
    let name = location
        .filename()
        .ok_or(ListCollectionRecordsErr::NotAFile("no filename"))?;
    if location != &prefix.child(name) {
        return Err(ListCollectionRecordsErr::NotDirectChild(
            location.to_string(),
        ));
    }
    let raw = RawId::from_hex(name).map_err(ListCollectionRecordsErr::BadNameHex)?;
    Id::new(raw).ok_or(ListCollectionRecordsErr::BadId)
}

async fn read_collection_record(
    store: &dyn ObjectStore,
    prefix: &Path,
    location: Path,
) -> Result<CollectionRecord, ListCollectionRecordsErr> {
    let path_id = collection_record_id_from_path(prefix, &location)?;
    let object = store
        .get(&location)
        .await
        .map_err(ListCollectionRecordsErr::Get)?;
    let bytes = object
        .bytes()
        .await
        .map_err(ListCollectionRecordsErr::Get)?;
    let bytes: Bytes = bytes.into();
    let blob: Blob<SimpleArchive> = Blob::new(bytes);
    let record = CollectionRecord::decode(&blob)
        .map_err(ListCollectionRecordsErr::Decode)?
        .ok_or(ListCollectionRecordsErr::UnknownKind)?;
    let record_id = record.id();
    if record_id != path_id {
        return Err(ListCollectionRecordsErr::IdMismatch {
            path: path_id,
            record: record_id,
        });
    }
    Ok(record)
}

/// Error returned while enumerating native collection records.
#[derive(Debug)]
pub enum ListCollectionRecordsErr {
    /// The object-store LIST operation failed.
    List(object_store::Error),
    /// A listed object had no filename component.
    NotAFile(&'static str),
    /// A listed object was nested below the one-id-per-object namespace.
    NotDirectChild(String),
    /// A listed filename was not a hexadecimal intrinsic id.
    BadNameHex(<RawId as FromHex>::Error),
    /// The decoded filename represented the nil id.
    BadId,
    /// A listed record object could not be fetched.
    Get(object_store::Error),
    /// The stored bytes were not a canonical collection record.
    Decode(RecordDecodeError),
    /// The stored archive carried no recognized collection-record kind.
    UnknownKind,
    /// The record's intrinsic id did not match its object path.
    IdMismatch { path: Id, record: Id },
}

impl fmt::Display for ListCollectionRecordsErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::List(error) => write!(f, "collection-record list failed: {error}"),
            Self::NotAFile(error) => write!(f, "collection-record list failed: {error}"),
            Self::NotDirectChild(path) => {
                write!(f, "collection-record object is not a direct child: {path}")
            }
            Self::BadNameHex(error) => {
                write!(f, "collection-record filename is not hexadecimal: {error}")
            }
            Self::BadId => write!(f, "collection-record filename is the nil id"),
            Self::Get(error) => write!(f, "collection-record fetch failed: {error}"),
            Self::Decode(error) => write!(f, "collection-record decode failed: {error}"),
            Self::UnknownKind => write!(f, "object is not a collection record"),
            Self::IdMismatch { path, record } => write!(
                f,
                "collection-record path id {path:X} does not match decoded id {record:X}"
            ),
        }
    }
}

impl Error for ListCollectionRecordsErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::List(error) | Self::Get(error) => Some(error),
            Self::BadNameHex(error) => Some(error),
            Self::Decode(error) => Some(error),
            Self::NotAFile(_)
            | Self::NotDirectChild(_)
            | Self::BadId
            | Self::UnknownKind
            | Self::IdMismatch { .. } => None,
        }
    }
}

/// Error returned while inserting one immutable collection record.
#[derive(Debug)]
pub enum InsertCollectionRecordErr {
    /// Creating the immutable record object failed.
    Store(object_store::Error),
    /// An existing object could not be fetched for idempotency validation.
    ReadExisting(object_store::Error),
    /// The intrinsic-id path already contained different bytes.
    ExistingMismatch { id: Id },
}

impl fmt::Display for InsertCollectionRecordErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(f, "collection-record insert failed: {error}"),
            Self::ReadExisting(error) => {
                write!(f, "failed to validate existing collection record: {error}")
            }
            Self::ExistingMismatch { id } => write!(
                f,
                "collection-record id {id:X} already contains different bytes"
            ),
        }
    }
}

impl Error for InsertCollectionRecordErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) | Self::ReadExisting(error) => Some(error),
            Self::ExistingMismatch { .. } => None,
        }
    }
}

/// Error returned when retrieving a blob from the object store.
#[derive(Debug)]
pub enum GetBlobErr<E: Error> {
    /// The underlying object store operation failed.
    Store(object_store::Error),
    /// The fetched object's bytes did not hash to the requested content address.
    HashMismatch {
        /// Digest encoded by the requested object path.
        expected: Inline<Hash<Blake3>>,
        /// Digest computed from the fetched bytes.
        actual: Inline<Hash<Blake3>>,
    },
    /// The blob bytes could not be converted to the requested type.
    Conversion(E),
}

impl<E: Error> fmt::Display for GetBlobErr<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(e) => write!(f, "object store error: {e}"),
            Self::HashMismatch { expected, actual } => write!(
                f,
                "object content hash mismatch: expected {}, got {}",
                Hash::<Blake3>::to_hex(expected),
                Hash::<Blake3>::to_hex(actual)
            ),
            Self::Conversion(e) => write!(f, "conversion error: {e}"),
        }
    }
}

impl<E: Error> Error for GetBlobErr<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            Self::HashMismatch { .. } | Self::Conversion(_) => None,
        }
    }
}

impl<E: Error> From<object_store::Error> for GetBlobErr<E> {
    fn from(e: object_store::Error) -> Self {
        Self::Store(e)
    }
}

/// Error returned when listing blobs from the object store.
#[derive(Debug)]
pub enum ListBlobsErr {
    /// The underlying list operation failed.
    List(object_store::Error),
    /// A listed object had no filename component.
    NotAFile(&'static str),
    /// A listed object's filename was not valid hexadecimal.
    BadNameHex(<RawInline as FromHex>::Error),
}

impl fmt::Display for ListBlobsErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::List(e) => write!(f, "list failed: {e}"),
            Self::NotAFile(e) => write!(f, "list failed: {e}"),
            Self::BadNameHex(e) => write!(f, "list failed: {e}"),
        }
    }
}
impl Error for ListBlobsErr {}

/// Error returned while reading or replacing a local policy cell.
#[derive(Debug)]
pub enum ObjectCellError {
    /// The underlying object-store operation failed.
    Store(object_store::Error),
    /// A non-tombstone cell object was not exactly one handle wide.
    InvalidLength(usize),
}

impl fmt::Display for ObjectCellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(f, "local-cell operation failed: {error}"),
            Self::InvalidLength(length) => {
                write!(f, "local-cell value has invalid byte length {length}")
            }
        }
    }
}

impl Error for ObjectCellError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::InvalidLength(_) => None,
        }
    }
}

/// Error returned when listing branches from the object store.
#[derive(Debug)]
pub enum ListBranchesErr {
    /// The underlying list operation failed.
    List(object_store::Error),
    /// A listed object had no filename component.
    NotAFile(&'static str),
    /// A listed object's filename was not valid hexadecimal.
    BadNameHex(<RawId as FromHex>::Error),
    /// The decoded bytes represent the nil identifier.
    BadId,
}

impl fmt::Display for ListBranchesErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::List(e) => write!(f, "list failed: {e}"),
            Self::NotAFile(e) => write!(f, "list failed: {e}"),
            Self::BadNameHex(e) => write!(f, "list failed: {e}"),
            Self::BadId => write!(f, "list failed: bad id"),
        }
    }
}
impl Error for ListBranchesErr {}

/// Error returned when reading a branch head from the object store.
#[derive(Debug)]
pub enum PullBranchErr {
    /// The stored bytes could not be parsed as a valid handle.
    ValidationErr(TryFromSliceError),
    /// The underlying object store operation failed.
    StoreErr(object_store::Error),
}

impl fmt::Display for PullBranchErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::StoreErr(e) => write!(f, "pull failed: {e}"),
            Self::ValidationErr(e) => write!(f, "pull failed: {e}"),
        }
    }
}

impl Error for PullBranchErr {}

impl From<object_store::Error> for PullBranchErr {
    fn from(err: object_store::Error) -> Self {
        Self::StoreErr(err)
    }
}

impl From<TryFromSliceError> for PullBranchErr {
    fn from(err: TryFromSliceError) -> Self {
        Self::ValidationErr(err)
    }
}

/// Error returned when updating a branch head in the object store.
#[derive(Debug)]
pub enum PushBranchErr {
    /// The stored bytes could not be parsed as a valid handle during a
    /// compare-and-swap.
    ValidationErr(TryFromSliceError),
    /// The underlying object store operation failed.
    StoreErr(object_store::Error),
}

impl fmt::Display for PushBranchErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ValidationErr(e) => write!(f, "commit failed: {e}"),
            Self::StoreErr(e) => write!(f, "commit failed: {e}"),
        }
    }
}

impl Error for PushBranchErr {}

impl From<object_store::Error> for PushBranchErr {
    fn from(err: object_store::Error) -> Self {
        Self::StoreErr(err)
    }
}

impl From<TryFromSliceError> for PushBranchErr {
    fn from(err: TryFromSliceError) -> Self {
        Self::ValidationErr(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::executor::block_on;
    use object_store::memory::InMemory;

    use crate::collection::{CollectionDescriptor, CollectionMerge, CollectionStore};
    use crate::repo::async_store::Blocking;
    use crate::repo::StorageFlush;

    fn remote() -> ObjectStoreRemote {
        ObjectStoreRemote {
            store: Arc::new(InMemory::new()),
            prefix: Path::from("test-repository"),
        }
    }

    fn record(tag: u8) -> CollectionRecord {
        let descriptor = CollectionDescriptor::new(
            Id::new([tag; 16]).unwrap(),
            Id::new([tag.wrapping_add(1).max(1); 16]).unwrap(),
            Id::new([tag.wrapping_add(2).max(1); 16]).unwrap(),
        );
        CollectionRecord::Merge(CollectionMerge::new(
            descriptor.handle(),
            Inline::new([tag.wrapping_add(3); 32]),
            Inline::new([tag.wrapping_add(4); 32]),
            Inline::new([tag.wrapping_add(5); 32]),
        ))
    }

    #[test]
    fn native_collection_records_are_sorted_and_idempotent() {
        block_on(async {
            let mut store = remote();
            let first = record(1);
            let second = record(9);

            AsyncCollectionStore::insert(&mut store, second)
                .await
                .unwrap();
            AsyncCollectionStore::insert(&mut store, first)
                .await
                .unwrap();
            AsyncCollectionStore::insert(&mut store, second)
                .await
                .unwrap();

            let actual = AsyncCollectionStore::records(&mut store)
                .await
                .unwrap()
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let mut expected = vec![first, second];
            expected.sort_unstable_by_key(CollectionRecord::id);
            assert_eq!(actual, expected);
        });
    }

    #[test]
    fn collection_record_path_must_match_decoded_intrinsic_id() {
        block_on(async {
            let mut store = remote();
            let path_record = record(1);
            let stored_record = record(2);
            let path = store
                .prefix
                .child(COLLECTION_RECORD_INFIX)
                .child(hex::encode(path_record.id()));
            let bytes: bytes::Bytes = CollectionRecord::to_blob(&stored_record).bytes.into();
            store.store.put(&path, bytes.into()).await.unwrap();

            assert!(matches!(
                AsyncCollectionStore::insert(&mut store, path_record).await,
                Err(InsertCollectionRecordErr::ExistingMismatch { id }) if id == path_record.id()
            ));

            let records = AsyncCollectionStore::records(&mut store).await.unwrap();
            assert_eq!(records.len(), 1);
            assert!(matches!(
                &records[0],
                Err(ListCollectionRecordsErr::IdMismatch { path, record })
                    if *path == path_record.id() && *record == stored_record.id()
            ));
        });
    }

    #[test]
    fn blocking_object_store_supports_collection_publication_flush() {
        let mut store = Blocking::new(remote()).unwrap();
        let record = record(17);

        CollectionStore::insert(&mut store, record).unwrap();
        StorageFlush::flush(&mut store).unwrap();
        let actual = CollectionStore::records(&mut store)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, vec![record]);
    }

    #[test]
    fn local_cells_are_lww_and_disjoint_from_remote_branches() {
        block_on(async {
            let mut store = remote();
            let id = Id::new([31; 16]).unwrap();
            let first = Inline::<Handle<SimpleArchive>>::new([41; 32]);
            let second = Inline::<Handle<SimpleArchive>>::new([42; 32]);

            AsyncLocalCellStore::set_cell(&mut store, id, Some(first))
                .await
                .unwrap();
            assert_eq!(
                AsyncLocalCellStore::cell(&mut store, id).await.unwrap(),
                Some(first)
            );
            AsyncLocalCellStore::set_cell(&mut store, id, Some(second))
                .await
                .unwrap();
            assert_eq!(
                AsyncLocalCellStore::cell(&mut store, id).await.unwrap(),
                Some(second)
            );
            assert!(AsyncPinStore::pins(&mut store).await.unwrap().is_empty());

            AsyncLocalCellStore::set_cell(&mut store, id, None)
                .await
                .unwrap();
            assert_eq!(
                AsyncLocalCellStore::cell(&mut store, id).await.unwrap(),
                None
            );
        });
    }
}
