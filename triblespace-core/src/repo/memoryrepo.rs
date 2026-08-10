use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::Infallible;

use crate::blob::encodings::UnknownBlob;
use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::blob::MemoryBlobStore;
use crate::collection::{CollectionRecord, CollectionStore};
use crate::prelude::blobencodings::SimpleArchive;
use crate::prelude::*;
use crate::repo::PinStore;
use crate::repo::PushResult;
use crate::repo::WantStore;

use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;

/// Simple in-memory implementation of [`BlobStore`] and [`PinStore`].
///
/// Useful for unit tests or ephemeral repositories where persistence is not
/// required.
#[derive(Debug, Default)]
pub struct MemoryRepo {
    /// In-memory blob store for all repository blobs.
    pub blobs: MemoryBlobStore,
    /// Map from pin id to the handle of its current head (a commit for content branches; arbitrary SimpleArchive blob for other pin roles).
    pub branches: HashMap<Id, Inline<Handle<SimpleArchive>>>,
    /// LWW-resolved wanted handles (see [`WantStore`]). In memory the
    /// last-writer-wins resolution is just insert/remove. Wants here are
    /// exactly as ephemeral as the blobs themselves — the trait is a
    /// capability, durability is the store's own property.
    pub wants: HashSet<Inline<Handle<UnknownBlob>>>,
    /// Canonical collection records keyed by intrinsic record id.
    collection_records: BTreeMap<Id, CollectionRecord>,
}

impl CollectionStore for MemoryRepo {
    type RecordsError = Infallible;
    type InsertError = Infallible;

    type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Self::RecordsError>>;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        Ok(self
            .collection_records
            .values()
            .copied()
            .map(Ok)
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.collection_records.entry(record.id()).or_insert(record);
        Ok(())
    }
}

impl crate::repo::BlobStorePut for MemoryRepo {
    type PutError = <MemoryBlobStore as crate::repo::BlobStorePut>::PutError;
    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.blobs.put(item)
    }
}

impl crate::repo::BlobStore for MemoryRepo {
    type Reader = <MemoryBlobStore as crate::repo::BlobStore>::Reader;
    type ReaderError = <MemoryBlobStore as crate::repo::BlobStore>::ReaderError;
    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.blobs.reader()
    }
}

impl crate::repo::BlobStoreKeep for MemoryRepo {
    fn keep<I>(&mut self, handles: I)
    where
        I: IntoIterator<Item = Inline<Handle<UnknownBlob>>>,
    {
        self.blobs.keep(handles);
    }
}

impl PinStore for MemoryRepo {
    type PinsError = Infallible;
    type HeadError = Infallible;
    type UpdateError = Infallible;

    type ListIter<'a> = std::vec::IntoIter<Result<Id, Self::PinsError>>;

    fn pins<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::PinsError> {
        // Sorted (not HashMap order): pin iteration order feeds
        // gossip-publish order and snapshot construction; HashMap's
        // per-instance seed would make every run reorder them, which
        // breaks deterministic simulation replay. Pile's PATCH-backed
        // pins() is already byte-ordered for the same reason.
        let mut ids: Vec<Id> = self.branches.keys().cloned().collect();
        ids.sort();
        Ok(ids.into_iter().map(Ok).collect::<Vec<_>>().into_iter())
    }

    fn head(&mut self, id: Id) -> Result<Option<Inline<Handle<SimpleArchive>>>, Self::HeadError> {
        Ok(self.branches.get(&id).cloned())
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Inline<Handle<SimpleArchive>>>,
        new: Option<Inline<Handle<SimpleArchive>>>,
    ) -> Result<PushResult, Self::UpdateError> {
        let current = self.branches.get(&id);
        if current != old.as_ref() {
            return Ok(PushResult::Conflict(current.cloned()));
        }
        match new {
            Some(new) => {
                self.branches.insert(id, new);
            }
            None => {
                self.branches.remove(&id);
            }
        }
        Ok(PushResult::Success())
    }
}

impl WantStore for MemoryRepo {
    type WantError = Infallible;

    type WantIter<'a> = std::vec::IntoIter<Result<Inline<Handle<UnknownBlob>>, Self::WantError>>;

    fn want<S>(&mut self, handle: Inline<Handle<S>>) -> Result<(), Self::WantError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.wants.insert(handle.transmute());
        Ok(())
    }

    fn unwant<S>(&mut self, handle: Inline<Handle<S>>) -> Result<(), Self::WantError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.wants.remove(&handle.transmute());
        Ok(())
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        // Sorted for the same reason as `pins()`: want enumeration
        // feeds sync-daemon fetch order, and HashSet's per-instance seed
        // would break deterministic simulation replay.
        let mut handles: Vec<Inline<Handle<UnknownBlob>>> = self.wants.iter().copied().collect();
        handles.sort();
        Ok(handles.into_iter().map(Ok).collect::<Vec<_>>().into_iter())
    }
}

impl crate::repo::StorageFlush for MemoryRepo {
    type Error = Infallible;

    fn flush(&mut self) -> Result<(), Self::Error> {
        // In-memory state has no sync point; durability is exactly the
        // process lifetime, same as the blobs themselves.
        Ok(())
    }
}

impl crate::repo::StorageClose for MemoryRepo {
    type Error = Infallible;

    fn close(self) -> Result<(), Self::Error> {
        // Nothing to do for the in-memory backend.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::collection::{CollectionDefinition, CollectionMerge};

    fn handle(byte: u8) -> Inline<Handle<UnknownBlob>> {
        Inline::new([byte; 32])
    }

    /// Wants resolve last-writer-wins: want → listed, unwant →
    /// gone, re-want → listed again. Enumeration is sorted (stable
    /// across runs despite HashSet backing).
    #[test]
    fn wants_lww_roundtrip() {
        let mut repo = MemoryRepo::default();
        assert_eq!(repo.wants().unwrap().count(), 0);

        repo.want(handle(2)).unwrap();
        repo.want(handle(1)).unwrap();
        // Reasserting an existing want is idempotent.
        repo.want(handle(1)).unwrap();
        let wants: Vec<_> = repo.wants().unwrap().map(Result::unwrap).collect();
        assert_eq!(wants, vec![handle(1), handle(2)], "sorted enumeration");

        repo.unwant(handle(1)).unwrap();
        let wants: Vec<_> = repo.wants().unwrap().map(Result::unwrap).collect();
        assert_eq!(wants, vec![handle(2)]);

        // A later want wins over the earlier retraction.
        repo.want(handle(1)).unwrap();
        assert_eq!(repo.wants().unwrap().count(), 2);
    }

    #[test]
    fn collection_records_are_idempotent_and_intrinsically_ordered() {
        let definition = CollectionRecord::Definition(CollectionDefinition::new(
            Id::new([1; 16]).unwrap(),
            Id::new([2; 16]).unwrap(),
            Id::new([3; 16]).unwrap(),
        ));
        let merge = CollectionRecord::Merge(CollectionMerge::new(
            definition.id(),
            Inline::new([4; 32]),
            Inline::new([5; 32]),
            Inline::new([6; 32]),
        ));
        let mut expected = vec![definition, merge];
        expected.sort_unstable_by_key(CollectionRecord::id);

        let mut repo = MemoryRepo::default();
        CollectionStore::insert(&mut repo, merge).unwrap();
        CollectionStore::insert(&mut repo, definition).unwrap();
        CollectionStore::insert(&mut repo, merge).unwrap();

        let actual = repo
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, expected);
    }
}
