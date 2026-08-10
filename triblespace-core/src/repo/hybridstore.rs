use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::collection::{CollectionRecord, CollectionStore};
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::inline::InlineEncoding;
use crate::prelude::blobencodings::SimpleArchive;
use crate::repo::BlobStore;
use crate::repo::BlobStorePut;
use crate::repo::PinStore;
use crate::repo::PushResult;

/// Store that delegates blob and branch operations to two independent stores.
///
/// This allows mixing different storage implementations in one repository,
/// e.g. an on-disk blob store with an in-memory branch store.
#[derive(Debug)]
pub struct HybridStore<B, R> {
    /// Storage for commit, content and metadata blobs.
    pub blobs: B,
    /// Storage for branch heads and native collection records.
    ///
    /// The field retains its historical name while both record families
    /// coexist; changing the public layout is a separate migration.
    pub branches: R,
}

impl<B, R> HybridStore<B, R> {
    /// Creates a new [`HybridStore`] from the given blob and branch stores.
    pub fn new(blobs: B, branches: R) -> Self {
        Self { blobs, branches }
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

impl<B, R> PinStore for HybridStore<B, R>
where
    R: PinStore,
{
    type PinsError = R::PinsError;
    type HeadError = R::HeadError;
    type UpdateError = R::UpdateError;

    type ListIter<'a>
        = R::ListIter<'a>
    where
        R: 'a,
        B: 'a;

    fn pins<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::PinsError> {
        self.branches.pins()
    }

    fn head(&mut self, id: Id) -> Result<Option<Inline<Handle<SimpleArchive>>>, Self::HeadError> {
        self.branches.head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Inline<Handle<SimpleArchive>>>,
        new: Option<Inline<Handle<SimpleArchive>>>,
    ) -> Result<PushResult, Self::UpdateError> {
        self.branches.update(id, old, new)
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
        self.branches.records()
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.branches.insert(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collection::CollectionDefinition;
    use crate::repo::memoryrepo::MemoryRepo;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    #[test]
    fn collection_records_delegate_only_to_the_record_side() {
        let record = CollectionRecord::Definition(CollectionDefinition::new(id(1), id(2), id(3)));
        let mut hybrid = HybridStore::new(MemoryRepo::default(), MemoryRepo::default());

        CollectionStore::insert(&mut hybrid, record).unwrap();
        assert_eq!(
            CollectionStore::records(&mut hybrid)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![record]
        );
        assert_eq!(
            CollectionStore::records(&mut hybrid.branches)
                .unwrap()
                .count(),
            1
        );
        assert_eq!(
            CollectionStore::records(&mut hybrid.blobs).unwrap().count(),
            0
        );
    }
}
