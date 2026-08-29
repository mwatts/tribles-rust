use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::collection::{CollectionRecord, CollectionRecordSelector, CollectionStore};
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::inline::InlineEncoding;
use crate::repo::BlobStore;
use crate::repo::BlobStorePut;
use crate::repo::StorageFlush;
use crate::repo::{ArtifactHandle, ArtifactOfferSnapshot, ArtifactOfferStore};
use crate::repo::{WantRequest, WantStore};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Store that delegates blob/want/offer and collection-record operations to two
/// independent stores.
///
/// This allows mixing different storage implementations, for example an
/// on-disk blob store with an in-memory collection-record store.
#[derive(Debug)]
pub struct HybridStore<B, R> {
    /// Storage for content-addressed blobs, durable typed wants, and offers.
    pub blobs: B,
    /// Storage for native collection records.
    pub records: R,
}

impl<B, R> ArtifactOfferStore for HybridStore<B, R>
where
    B: ArtifactOfferStore,
{
    type OfferError = B::OfferError;

    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>,
    {
        self.blobs.offer_all(handles)
    }

    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError> {
        self.blobs.offers_snapshot()
    }
}

impl<B, R> HybridStore<B, R> {
    /// Creates a new [`HybridStore`] from the given blob and record stores.
    pub fn new(blobs: B, records: R) -> Self {
        Self { blobs, records }
    }
}

/// Failure while crash-ordering writes across a [`HybridStore`].
#[derive(Debug)]
pub enum HybridFlushError<BlobError, RecordError> {
    /// The content-addressed blob store could not make staged data durable.
    Blobs(BlobError),
    /// The record store could not make collection evidence durable.
    Records(RecordError),
}

impl<BlobError, RecordError> fmt::Display for HybridFlushError<BlobError, RecordError>
where
    BlobError: fmt::Display,
    RecordError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blobs(error) => write!(formatter, "failed to flush hybrid blob store: {error}"),
            Self::Records(error) => {
                write!(formatter, "failed to flush hybrid record store: {error}")
            }
        }
    }
}

impl<BlobError, RecordError> Error for HybridFlushError<BlobError, RecordError>
where
    BlobError: Error + 'static,
    RecordError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Blobs(error) => Some(error),
            Self::Records(error) => Some(error),
        }
    }
}

impl<B, R> StorageFlush for HybridStore<B, R>
where
    B: StorageFlush,
    R: StorageFlush,
{
    type Error = HybridFlushError<B::Error, R::Error>;

    fn flush(&mut self) -> Result<(), Self::Error> {
        // Never let authoritative records become durable ahead of the blobs
        // they name. If the first barrier fails, leave record durability
        // untouched; if the second fails, only harmless orphan blobs remain.
        self.blobs.flush().map_err(HybridFlushError::Blobs)?;
        self.records.flush().map_err(HybridFlushError::Records)
    }
}

impl<B, R> BlobStorePut for HybridStore<B, R>
where
    B: BlobStorePut,
{
    type PutError = B::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.blobs.put(item)
    }
}

impl<B, R> BlobStore for HybridStore<B, R>
where
    B: BlobStore,
{
    type Reader = B::Reader;
    type ReaderError = B::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.blobs.reader()
    }
}

impl<B, R> CollectionStore for HybridStore<B, R>
where
    R: CollectionStore,
{
    type RecordsError = R::RecordsError;
    type InsertError = R::InsertError;

    type RecordIter<'a>
        = R::RecordIter<'a>
    where
        B: 'a,
        R: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.records.records()
    }

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        self.records.select_records(selectors)
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.records.insert(record)
    }
}

impl<B, R> WantStore for HybridStore<B, R>
where
    B: WantStore,
{
    type WantError = B::WantError;

    type WantIter<'a>
        = B::WantIter<'a>
    where
        B: 'a,
        R: 'a;

    fn want(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.blobs.want(request)
    }

    fn unwant(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.blobs.unwant(request)
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        self.blobs.wants()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::reach;
    use crate::id::Id;

    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::blob::IntoBlob;
    use crate::collection::descriptor;
    use crate::collection::{
        simplearchive_union, CollectionHandle, CollectionMerge, CollectionStoreExt,
    };
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::trible::Fragment;
    use ed25519_dalek::SigningKey;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    #[test]
    fn collection_records_delegate_only_to_the_record_side() {
        let team = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let facts = descriptor::naming("hybrid", team, id(2), id(3), reach::private()).into_facts();
        // Only the identity matters here; nothing resolves this descriptor.
        let collection: CollectionHandle = IntoBlob::<SimpleArchive>::to_blob(facts).get_handle();
        let record = CollectionRecord::Merge(CollectionMerge::new(
            collection,
            Inline::new([4; 32]),
            Inline::new([5; 32]),
            Inline::new([6; 32]),
        ));
        let mut hybrid = HybridStore::new(MemoryRepo::default(), MemoryRepo::default());

        CollectionStore::insert(&mut hybrid, record).unwrap();
        assert_eq!(
            CollectionStore::records(&mut hybrid)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![record]
        );
        let selectors = [CollectionRecordSelector::MergeCollection(collection)]
            .into_iter()
            .collect();
        assert_eq!(
            CollectionStore::select_records(&mut hybrid, &selectors).unwrap(),
            vec![record]
        );
        assert_eq!(
            CollectionStore::records(&mut hybrid.records)
                .unwrap()
                .count(),
            1
        );
        assert_eq!(
            CollectionStore::records(&mut hybrid.blobs).unwrap().count(),
            0
        );
    }

    #[test]
    fn artifact_offers_delegate_only_to_the_blob_side() {
        let offered = ArtifactHandle::new([31; 32]);
        let mut hybrid = HybridStore::new(MemoryRepo::default(), MemoryRepo::default());

        hybrid.offer(offered).unwrap();
        assert!(hybrid.offers_snapshot().unwrap().contains(offered));
        assert!(hybrid.blobs.offers_snapshot().unwrap().contains(offered));
        assert!(hybrid.records.offers_snapshot().unwrap().is_empty());
    }

    #[test]
    fn collection_publication_and_read_work_across_both_sides() {
        let mut hybrid = HybridStore::new(MemoryRepo::default(), MemoryRepo::default());
        let signing_key = SigningKey::from_bytes(&[8; 32]);
        let name = "hybrid";
        let team = signing_key.verifying_key();
        let target = hybrid
            .collection(simplearchive_union::descriptor(
                name,
                team,
                reach::private(),
            ))
            .unwrap();

        let commit = hybrid
            .commit(target, &signing_key, Fragment::empty())
            .unwrap();
        assert_eq!(hybrid.snapshot(target, &[]).unwrap().facts().len(), 0);
        assert_eq!(commit.collection(), target.handle());
        assert!(hybrid.blobs.blobs.len() >= 2);
        assert_eq!(
            hybrid
                .records
                .records()
                .unwrap()
                .filter_map(Result::ok)
                .filter(|record| {
                    matches!(record, CollectionRecord::Commit(commit) if commit.collection() == target.handle())
                })
                .count(),
            1
        );
        assert_eq!(hybrid.blobs.records().unwrap().count(), 0);
    }

    #[test]
    fn wants_delegate_only_to_the_blob_side() {
        use crate::blob::encodings::UnknownBlob;

        let handle = Inline::<Handle<UnknownBlob>>::new([9; 32]);
        let request = WantRequest::blob(handle);
        let mut hybrid = HybridStore::new(MemoryRepo::default(), MemoryRepo::default());

        hybrid.want(request).unwrap();
        assert_eq!(
            hybrid
                .wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![request]
        );
        assert_eq!(hybrid.blobs.wants().unwrap().count(), 1);
        assert_eq!(hybrid.records.wants().unwrap().count(), 0);
    }
}
