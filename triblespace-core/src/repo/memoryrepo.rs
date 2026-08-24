use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::convert::Infallible;

use crate::blob::encodings::UnknownBlob;
use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::blob::MemoryBlobStore;
use crate::collection::store::selectors_match_record;
use crate::collection::{CollectionRecord, CollectionRecordSelector, CollectionStore};
use crate::prelude::*;
use crate::repo::{WantRequest, WantStore};

use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;

/// Simple in-memory implementation of the repository storage traits.
///
/// Useful for unit tests or ephemeral repositories where persistence is not
/// required.
#[derive(Clone, Debug, Default)]
pub struct MemoryRepo {
    /// In-memory blob store for all repository blobs.
    pub blobs: MemoryBlobStore,
    /// LWW-resolved typed requests (see [`WantStore`]). In memory the
    /// last-writer-wins resolution is just insert/remove. Wants here are
    /// exactly as ephemeral as the blobs themselves — the trait is a
    /// capability, durability is the store's own property.
    pub wants: HashSet<WantRequest>,
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

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        if selectors.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .collection_records
            .values()
            .copied()
            .filter(|record| selectors_match_record(selectors, *record))
            .collect())
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
        let reader = self.blobs.reader().expect("memory reader is infallible");
        let mut roots = crate::repo::RetentionRoots::new();
        for record in self.collection_records.values() {
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            if commit.verify_strict().is_err() {
                continue;
            }
            for root in [
                Inline::<Handle<UnknownBlob>>::new(commit.collection().raw),
                Inline::<Handle<UnknownBlob>>::new(commit.data().raw),
                commit.metadata().transmute(),
            ] {
                if crate::repo::BlobStoreList::contains_blob(&reader, root).unwrap_or(false) {
                    roots.retain_recursive(root);
                }
            }
        }
        self.blobs
            .keep(handles.into_iter().chain(roots.expanded(&reader)));
    }
}

impl WantStore for MemoryRepo {
    type WantError = Infallible;

    type WantIter<'a> = std::vec::IntoIter<Result<WantRequest, Self::WantError>>;

    fn want(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.wants.insert(request);
        Ok(())
    }

    fn unwant(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.wants.remove(&request);
        Ok(())
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        // Want enumeration feeds sync-daemon fetch order, and HashSet's
        // per-instance seed would break deterministic simulation replay.
        let mut requests: Vec<WantRequest> = self.wants.iter().copied().collect();
        requests.sort();
        Ok(requests.into_iter().map(Ok).collect::<Vec<_>>().into_iter())
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
    use crate::collection::reach;

    use crate::collection::descriptor::{identity_for_tests, named_for_tests};
    use crate::collection::{CollectionDerive, CollectionMerge};

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

        let first = WantRequest::blob(handle(1));
        let second = WantRequest::blob(handle(2));
        repo.want(second).unwrap();
        repo.want(first).unwrap();
        // Reasserting an existing want is idempotent.
        repo.want(first).unwrap();
        let wants: Vec<_> = repo.wants().unwrap().map(Result::unwrap).collect();
        assert_eq!(wants, vec![first, second], "sorted enumeration");

        repo.unwant(first).unwrap();
        let wants: Vec<_> = repo.wants().unwrap().map(Result::unwrap).collect();
        assert_eq!(wants, vec![second]);

        // A later want wins over the earlier retraction.
        repo.want(first).unwrap();
        assert_eq!(repo.wants().unwrap().count(), 2);
    }

    #[test]
    fn collection_records_are_idempotent_and_intrinsically_ordered() {
        let descriptor = named_for_tests(
            "merged",
            Id::new([2; 16]).unwrap(),
            Id::new([3; 16]).unwrap(),
        );
        let target = named_for_tests(
            "derived",
            Id::new([8; 16]).unwrap(),
            Id::new([9; 16]).unwrap(),
        );
        let merge = CollectionRecord::Merge(CollectionMerge::new(
            identity_for_tests(&descriptor),
            Inline::new([4; 32]),
            Inline::new([5; 32]),
            Inline::new([6; 32]),
        ));
        let derive = CollectionRecord::Derive(CollectionDerive::new(
            identity_for_tests(&target),
            Inline::new([10; 32]),
            Inline::new([11; 32]),
        ));
        let mut expected = vec![derive, merge];
        expected.sort_unstable_by_key(CollectionRecord::id);

        let mut repo = MemoryRepo::default();
        CollectionStore::insert(&mut repo, merge).unwrap();
        CollectionStore::insert(&mut repo, derive).unwrap();
        CollectionStore::insert(&mut repo, merge).unwrap();

        let actual = repo
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn collection_primary_selection_answers_group_and_exact_conflicting_operations() {
        let source = identity_for_tests(&named_for_tests(
            "source",
            Id::new([22; 16]).unwrap(),
            Id::new([23; 16]).unwrap(),
        ));
        let target = identity_for_tests(&named_for_tests(
            "target",
            Id::new([25; 16]).unwrap(),
            Id::new([26; 16]).unwrap(),
        ));
        let other = identity_for_tests(&named_for_tests(
            "other",
            Id::new([28; 16]).unwrap(),
            Id::new([29; 16]).unwrap(),
        ));
        let input = Inline::new([30; 32]);
        let merge = CollectionRecord::Merge(CollectionMerge::new(
            source,
            Inline::new([31; 32]),
            Inline::new([32; 32]),
            Inline::new([33; 32]),
        ));
        let first =
            CollectionRecord::Derive(CollectionDerive::new(target, input, Inline::new([34; 32])));
        let conflicting =
            CollectionRecord::Derive(CollectionDerive::new(target, input, Inline::new([35; 32])));
        let sibling = CollectionRecord::Derive(CollectionDerive::new(
            target,
            Inline::new([36; 32]),
            Inline::new([37; 32]),
        ));
        let unrelated =
            CollectionRecord::Derive(CollectionDerive::new(other, input, Inline::new([38; 32])));
        let mut repo = MemoryRepo::default();
        for record in [unrelated, conflicting, merge, first, sibling, first] {
            repo.insert(record).unwrap();
        }

        let exact = [CollectionRecordSelector::Operation(WantRequest::derive(
            target, input,
        ))]
        .into_iter()
        .collect();
        let mut expected = vec![first, conflicting];
        expected.sort_unstable_by_key(CollectionRecord::id);
        assert_eq!(repo.select_records(&exact).unwrap(), expected);

        let grouped = [
            CollectionRecordSelector::MergeCollection(source),
            CollectionRecordSelector::DeriveTarget(target),
        ]
        .into_iter()
        .collect();
        let mut expected = vec![merge, first, conflicting, sibling];
        expected.sort_unstable_by_key(CollectionRecord::id);
        assert_eq!(repo.select_records(&grouped).unwrap(), expected);
        assert!(!repo.select_records(&grouped).unwrap().contains(&unrelated));
    }

    #[test]
    fn valid_collection_commits_and_owned_closure_survive_memory_keep() {
        use ed25519_dalek::SigningKey;

        use crate::authority::{self, AuthorityGrant, AuthorityMode, ACTION_WRITE};
        use crate::blob::encodings::utf8string::UTF8String;
        use crate::collection::{simplearchive_union, Collection};
        use crate::repo::{BlobStoreGet, BlobStoreKeep};

        let mut repo = MemoryRepo::default();
        let child = repo.put::<UTF8String, _>("owned child".to_owned()).unwrap();
        let fragment = entity! { crate::metadata::name: child };
        let name = crate::collection::records::CollectionName::new("owned").unwrap();
        let key = SigningKey::from_bytes(&[23; 32]);
        let team = key.verifying_key();
        let descriptor = simplearchive_union::descriptor(&name, team, Some(team), reach::private());
        let collection = identity_for_tests(&descriptor);
        authority::publish_grant(
            &mut repo,
            team,
            &key,
            AuthorityGrant::root(
                key.verifying_key(),
                collection,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();
        let commit = Collection::new(&mut repo, &name, team, key, reach::private())
            .commit(fragment)
            .unwrap();
        let orphan = repo.put::<UTF8String, _>("orphan".to_owned()).unwrap();

        repo.keep(std::iter::empty::<Inline<Handle<UnknownBlob>>>());

        let reader = repo.reader().unwrap();
        for retained in [
            collection.transmute(),
            Inline::<Handle<UnknownBlob>>::new(commit.data().raw),
            commit.metadata().transmute(),
            child.transmute(),
        ] {
            assert!(reader.get::<Blob<UnknownBlob>, _>(retained).is_ok());
        }
        assert!(reader
            .get::<Blob<UnknownBlob>, _>(orphan.transmute())
            .is_err());
    }
}
