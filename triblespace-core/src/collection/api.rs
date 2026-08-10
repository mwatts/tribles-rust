//! Narrow content-publication facade for one scoped collection.
//!
//! [`Collection`] owns the storage, canonical `SimpleArchive`-union
//! definition, and signing key needed to publish [`Fragment`] values. It is a
//! write path, not a repository abstraction: it has no head, branch, CAS,
//! retry, read-admission, or planning policy.

use ed25519_dalek::SigningKey;

use crate::id::Id;
use crate::repo::{BlobStorePut, StorageClose, StorageFlush};
use crate::trible::Fragment;

use super::simplearchive_union::{self, PublicationError};
use super::{CollectionCommit, CollectionDefinition, CollectionStore};

/// A scoped `SimpleArchive`-union collection and its publication authority.
///
/// Construction is pure with respect to `storage`: the canonical definition
/// is derived in memory and is not inserted until a [`commit`](Self::commit)
/// publication begins.
pub struct Collection<S> {
    storage: S,
    definition: CollectionDefinition,
    signing_key: SigningKey,
}

impl<S> Collection<S> {
    /// Construct a write facade without reading from or writing to `storage`.
    pub fn new(storage: S, scope: Id, signing_key: SigningKey) -> Self {
        Self {
            storage,
            definition: simplearchive_union::definition(scope),
            signing_key,
        }
    }

    /// Canonical collection definition derived from the constructor scope.
    pub fn definition(&self) -> &CollectionDefinition {
        &self.definition
    }

    /// Borrow the underlying storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Mutably borrow the underlying storage.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Consume the facade and recover its underlying storage.
    pub fn into_storage(self) -> S {
        self.storage
    }
}

impl<S> Collection<S>
where
    S: BlobStorePut + CollectionStore + StorageFlush,
{
    /// Publish one self-contained fragment as an independent signed commit.
    ///
    /// Facts are the collection element, metafacts are commit metadata, and
    /// fragment attachments are staged through the same crash-ordered,
    /// content-addressed path as
    /// [`simplearchive_union::publish_fragment_commit`]. Repeating identical
    /// input is idempotent; distinct commits coexist without selecting a head.
    /// The parameter is deliberately `Fragment`, rather than `Into<Fragment>`,
    /// so a bare fact set cannot accidentally publish without its metafacts.
    pub fn commit(
        &mut self,
        fragment: Fragment,
    ) -> Result<
        CollectionCommit,
        PublicationError<S::PutError, S::InsertError, <S as StorageFlush>::Error>,
    > {
        simplearchive_union::publish_fragment_commit(
            &mut self.storage,
            &self.definition,
            fragment,
            &self.signing_key,
        )
    }
}

impl<S> Collection<S>
where
    S: StorageClose,
{
    /// Consume the facade and close its underlying storage.
    pub fn close(self) -> Result<(), S::Error> {
        self.storage.close()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    use crate::blob::encodings::longstring::LongString;
    use crate::collection::{discover_collection_records, CollectionRecord};
    use crate::inline::encodings::hash::Handle;
    use crate::inline::Inline;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn fragment(entity: u8, attachment: bool) -> Fragment {
        let mut row = [entity; TRIBLE_LEN];
        row[16..32].fill(1);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        let mut fragment = Fragment::from(facts);
        if attachment {
            let _: Inline<Handle<LongString>> = fragment.put("one attachment".to_owned());
        }
        fragment
    }

    #[test]
    fn construction_is_pure_and_derives_the_scoped_definition() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);

        assert_eq!(
            collection.definition(),
            &simplearchive_union::definition(scope)
        );
        assert!(collection.storage().blobs.is_empty());
        assert!(collection.storage_mut().records().unwrap().next().is_none());

        collection.close().unwrap();
    }

    #[test]
    fn distinct_commits_coexist_and_repeated_commits_are_idempotent() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);
        let first_fragment = fragment(1, true);
        let second_fragment = fragment(2, false);

        let first = collection.commit(first_fragment.clone()).unwrap();
        let after_first = collection.storage().blobs.len();
        let second = collection.commit(second_fragment).unwrap();
        let after_second = collection.storage().blobs.len();
        let repeated = collection.commit(first_fragment).unwrap();

        assert_eq!(repeated, first);
        assert_ne!(second, first);
        assert!(after_second > after_first);
        assert_eq!(collection.storage().blobs.len(), after_second);

        let definition = *collection.definition();
        let discovered = discover_collection_records(collection.storage_mut()).unwrap();
        assert_eq!(discovered.definitions(), &[definition]);
        assert_eq!(
            discovered
                .commits()
                .iter()
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([first.id(), second.id()])
        );
        assert_eq!(
            collection
                .storage_mut()
                .records()
                .unwrap()
                .collect::<Result<Vec<CollectionRecord>, _>>()
                .unwrap()
                .len(),
            3
        );
    }
}
