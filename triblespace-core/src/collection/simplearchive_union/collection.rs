//! Read-only exact-ticket facade for one scoped `SimpleArchive` collection.
//!
//! [`SimpleArchiveCollection`] carries only the fixed canonical descriptor.
//! Callers supply a mathematical set of complete signed commits on every read;
//! no signing key, mutable head, signer-wide discovery rule, or write surface is
//! present. Byte-identical ticket repeats collapse, while the stored records,
//! signatures, descriptor, data, and mandatory metadata are all checked before
//! facts are returned.

use ed25519_dalek::VerifyingKey;

// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use crate::collection::reach;
use crate::collection::records::CollectionName;

use std::collections::BTreeSet;
use std::convert::Infallible;

use crate::collection::api::{Collection, CollectionSnapshot};
use crate::collection::discovery::{
    canonicalize_exact_ticket, discover_collection_records_for_collection_ticket,
    validate_exact_ticket,
};
use crate::collection::{
    CollectionCommit, CollectionHandle, CollectionMaterializationError, CollectionStore,
    DiscoveredCollectionRecords,
};
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use crate::trible::{Fragment, TribleSet};

/// Read-only exact-ticket view of one canonical `SimpleArchive` union.
///
/// The scope fixes the descriptor, while each call supplies the complete
/// commit authority set. The facade borrows storage only for the duration of a
/// read and has no API capable of inserting blobs or collection records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimpleArchiveCollection {
    name: CollectionName,
    namespace: VerifyingKey,
    authority: Option<VerifyingKey>,
    reach: Fragment,
}

impl SimpleArchiveCollection {
    /// Construct a read-only exact-ticket facade for one named root.
    ///
    /// `reach` is not decoration on a read facade: it is part of the
    /// descriptor this facade hashes, so a facade that names the wrong reach
    /// names a different collection and matches no ticket.
    pub fn new(
        name: CollectionName,
        namespace: VerifyingKey,
        authority: Option<VerifyingKey>,
        reach: Fragment,
    ) -> Self {
        Self {
            name,
            namespace,
            authority,
            reach,
        }
    }

    /// How far this collection may travel.
    pub fn reach(&self) -> &Fragment {
        &self.reach
    }

    /// Name this collection is known by within its namespace.
    pub fn name(&self) -> &CollectionName {
        &self.name
    }

    /// Public-key namespace which scopes this root's name.
    pub fn namespace(&self) -> VerifyingKey {
        self.namespace
    }

    /// Optional external capability trust root in this descriptor.
    pub fn authority(&self) -> Option<VerifyingKey> {
        self.authority
    }

    /// Canonical `SimpleArchive` set-union descriptor facts.
    pub fn descriptor(&self) -> Fragment {
        super::descriptor(
            &self.name,
            self.namespace,
            self.authority,
            self.reach.clone(),
        )
    }

    /// Content identity of this collection's descriptor.
    ///
    /// This is the read side: the facade is not storing anything, it is
    /// naming the collection a ticket must match. A write path takes its
    /// handle from what `put` returns instead.
    pub fn collection(&self) -> CollectionHandle {
        use crate::blob::encodings::simplearchive::SimpleArchive;
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.descriptor().into_facts()).get_handle()
    }

    /// Attach the exact ticket as a materialized fact set without writing.
    ///
    /// The ticket is a set of complete [`CollectionCommit`] records, not a
    /// signer filter. Every member may therefore have a different author, but
    /// it must name this exact descriptor, byte-match a strictly verified
    /// stored record, and have resident valid descriptor, data, and metadata
    /// dependencies. Commits present in storage but absent from `ticket` are
    /// inert. Same-descriptor `MERGE` records may provide an exact physical
    /// cover, but never add authority. Stores should bound unsigned merge
    /// admission because every admitted same-descriptor equation is candidate
    /// cache evidence for this read.
    ///
    /// An empty ticket returns the local empty set without touching storage.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<
        TribleSet,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let commits = canonicalize_exact_ticket(ticket, self.collection())
            .map_err(CollectionMaterializationError::ExactTicket)?;
        if commits.is_empty() {
            return Ok(TribleSet::new());
        }
        self.snapshot_canonical(store, commits)
            .map(CollectionSnapshot::into_facts)
    }

    /// Capture one coherent exact-ticket fact, commit, and reader snapshot.
    ///
    /// Record selection happens once before one blob reader is opened. The
    /// returned commits are canonical intrinsic-id ordered and duplicate-free,
    /// and only those commits can authorize the returned facts even if the
    /// reader physically contains later or otherwise unselected blobs.
    /// An empty ticket still opens and returns a reader, matching
    /// [`crate::collection::Collection::snapshot`].
    pub fn snapshot_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<
        CollectionSnapshot<S::Reader>,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let commits = canonicalize_exact_ticket(ticket, self.collection())
            .map_err(CollectionMaterializationError::ExactTicket)?;
        self.snapshot_canonical(store, commits)
    }

    fn snapshot_canonical<S>(
        &self,
        store: &mut S,
        commits: Vec<CollectionCommit>,
    ) -> Result<
        CollectionSnapshot<S::Reader>,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let descriptor = self.descriptor();
        let collection = self.collection();
        if commits.is_empty() {
            return Collection::<S>::snapshot_from_observation(
                store,
                &descriptor,
                DiscoveredCollectionRecords::default(),
                commits,
            );
        }

        let requested: BTreeSet<_> = commits.iter().map(CollectionCommit::id).collect();
        let discovered =
            discover_collection_records_for_collection_ticket(store, &requested, collection)
                .map_err(CollectionMaterializationError::Discovery)?;
        validate_exact_ticket(&discovered, &commits)
            .map_err(CollectionMaterializationError::ExactTicket)?;
        Collection::<S>::snapshot_from_observation(store, &descriptor, discovered, commits)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::encodings::{simplearchive::SimpleArchive, UnknownBlob};
    use crate::blob::{Blob, BlobEncoding, Bytes, IntoBlob};
    use crate::collection::descriptor::identity_for_tests;
    use crate::collection::records::CollectionName;
    use crate::collection::{CollectionRecord, CollectionRecordSelector, ExactTicketError};
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::BlobStorePut;
    use crate::trible::{Trible, TRIBLE_LEN};

    fn test_facade(name: &str) -> SimpleArchiveCollection {
        SimpleArchiveCollection::new(
            CollectionName::new(name).unwrap(),
            SigningKey::from_bytes(&[1; 32]).verifying_key(),
            Some(SigningKey::from_bytes(&[1; 32]).verifying_key()),
            reach::private(),
        )
    }

    fn facts(entity: u8) -> TribleSet {
        let mut row = [entity; TRIBLE_LEN];
        row[16..32].fill(1);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        facts
    }

    fn archive(entity: u8) -> Blob<SimpleArchive> {
        facts(entity).to_blob()
    }

    fn publish(
        store: &mut MemoryRepo,
        descriptor: &Fragment,
        key: u8,
        entity: u8,
    ) -> (CollectionCommit, Blob<SimpleArchive>) {
        let data = archive(entity);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let commit = super::super::publish_commit(
            store,
            descriptor,
            &data,
            &metadata,
            &SigningKey::from_bytes(&[key; 32]),
        )
        .unwrap();
        (commit, data)
    }

    fn invalid_signature(commit: CollectionCommit) -> CollectionCommit {
        let (r, mut s) = commit.signature();
        s.raw[0] ^= 1;
        CollectionCommit::from_parts(
            commit.collection(),
            commit.data(),
            commit.metadata(),
            commit.public_key(),
            r,
            s,
        )
    }

    #[derive(Default)]
    struct ReadOnlyCountingStore {
        inner: MemoryRepo,
        selections: usize,
        readers: usize,
        last_selectors: Option<BTreeSet<CollectionRecordSelector>>,
    }

    impl BlobStorePut for ReadOnlyCountingStore {
        type PutError = Infallible;

        fn put<E, T>(&mut self, _item: T) -> Result<Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            panic!("read-only exact facade attempted to write a blob")
        }
    }

    impl BlobStore for ReadOnlyCountingStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.readers += 1;
            self.inner.reader()
        }
    }

    impl CollectionStore for ReadOnlyCountingStore {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = Infallible;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            panic!("exact facade must use the selector boundary")
        }

        fn select_records(
            &mut self,
            selectors: &BTreeSet<CollectionRecordSelector>,
        ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
            self.selections += 1;
            self.last_selectors = Some(selectors.clone());
            self.inner.select_records(selectors)
        }

        fn insert(&mut self, _record: CollectionRecord) -> Result<(), Self::InsertError> {
            panic!("read-only exact facade attempted to insert a record")
        }
    }

    #[test]
    fn mixed_authors_form_one_exact_sorted_set_and_unselected_commits_are_inert() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut store = MemoryRepo::default();
        let (first, _) = publish(&mut store, &descriptor, 7, 1);
        let (second, _) = publish(&mut store, &descriptor, 8, 2);
        let (_unselected, _) = publish(&mut store, &descriptor, 9, 3);

        let snapshot = facade
            .snapshot_exact(&mut store, &[second, first, first])
            .unwrap();
        let mut expected = facts(1);
        expected += facts(2);
        let mut expected_commits = vec![first, second];
        expected_commits.sort_unstable_by_key(CollectionCommit::id);

        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(snapshot.commits(), expected_commits);
        assert_ne!(first.public_key(), second.public_key());
    }

    #[test]
    fn exact_reads_select_records_once_and_open_one_coherent_reader() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut inner = MemoryRepo::default();
        let (commit, _) = publish(&mut inner, &descriptor, 7, 1);
        let mut store = ReadOnlyCountingStore {
            inner,
            ..ReadOnlyCountingStore::default()
        };

        let snapshot = facade.snapshot_exact(&mut store, &[commit]).unwrap();

        assert_eq!(snapshot.facts(), &facts(1));
        assert_eq!(snapshot.commits(), &[commit]);
        assert_eq!(store.selections, 1);
        assert_eq!(store.readers, 1);
        assert_eq!(
            store.last_selectors,
            Some(BTreeSet::from([
                CollectionRecordSelector::Id(commit.id()),
                CollectionRecordSelector::MergeCollection(identity_for_tests(&descriptor)),
            ])),
        );
    }

    #[test]
    fn conflicting_ticket_bytes_and_a_mismatched_stored_record_fail_closed() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let metadata = crate::collection::empty_metadata_handle();
        let first = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            identity_for_tests(&descriptor),
            Inline::new([3; 32]),
            metadata,
        );
        let conflicting = CollectionCommit::sign(
            &SigningKey::from_bytes(&[8; 32]),
            identity_for_tests(&descriptor),
            Inline::new([4; 32]),
            metadata,
        )
        .with_test_id(first.id());
        let mut untouched = ReadOnlyCountingStore::default();
        assert!(matches!(
            facade.attach_exact(&mut untouched, &[first, conflicting]),
            Err(CollectionMaterializationError::ExactTicket(
                ExactTicketError::ConflictingCommit { commit }
            )) if commit == first.id()
        ));
        assert_eq!((untouched.selections, untouched.readers), (0, 0));

        let mut inner = MemoryRepo::default();
        let (stored, _) = publish(&mut inner, &descriptor, 8, 2);
        let requested = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            stored.collection(),
            stored.data(),
            stored.metadata(),
        );
        inner
            .insert(CollectionRecord::Commit(
                stored.with_test_id(requested.id()),
            ))
            .unwrap();
        let mut mismatched = ReadOnlyCountingStore {
            inner,
            ..ReadOnlyCountingStore::default()
        };
        assert!(matches!(
            facade.attach_exact(&mut mismatched, &[requested]),
            Err(CollectionMaterializationError::ExactTicket(
                ExactTicketError::StoredCommitMismatch { commit }
            )) if commit == requested.id()
        ));
        assert_eq!((mismatched.selections, mismatched.readers), (1, 0));
    }

    #[test]
    fn absent_mismatched_and_invalid_records_do_not_satisfy_a_ticket() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();

        let mut source = MemoryRepo::default();
        let (absent, _) = publish(&mut source, &descriptor, 7, 1);
        let mut empty = MemoryRepo::default();
        assert!(matches!(
            facade.attach_exact(&mut empty, &[absent]),
            Err(CollectionMaterializationError::ExactTicket(
                ExactTicketError::MissingOrInvalidCommit { commit }
            )) if commit == absent.id()
        ));

        // The same descriptor, data, and metadata under another author's
        // signature is semantically similar but not the same full record.
        let mut mismatched = MemoryRepo::default();
        let (stored, _) = publish(&mut mismatched, &descriptor, 8, 2);
        let requested = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            stored.collection(),
            stored.data(),
            stored.metadata(),
        );
        assert_ne!(requested.to_bytes(), stored.to_bytes());
        assert!(matches!(
            facade.attach_exact(&mut mismatched, &[requested]),
            Err(CollectionMaterializationError::ExactTicket(
                ExactTicketError::MissingOrInvalidCommit { commit }
            )) if commit == requested.id()
        ));

        let mut invalid_store = source;
        let invalid = invalid_signature(absent);
        invalid_store
            .insert(CollectionRecord::Commit(invalid))
            .unwrap();
        assert!(matches!(
            facade.attach_exact(&mut invalid_store, &[invalid]),
            Err(CollectionMaterializationError::ExactTicket(
                ExactTicketError::MissingOrInvalidCommit { commit }
            )) if commit == invalid.id()
        ));
    }

    #[test]
    fn a_ticket_for_another_descriptor_is_rejected_before_storage_access() {
        let facade = test_facade("first");
        let other = test_facade("second");
        let mut source = MemoryRepo::default();
        let (commit, _) = publish(&mut source, &other.descriptor(), 7, 1);
        let mut store = ReadOnlyCountingStore {
            inner: source,
            ..ReadOnlyCountingStore::default()
        };

        assert!(matches!(
            facade.attach_exact(&mut store, &[commit]),
            Err(CollectionMaterializationError::ExactTicket(
                ExactTicketError::WrongCollection {
                    commit: found,
                    expected,
                    actual,
                }
            )) if found == commit.id()
                && expected == facade.collection()
                && actual == other.collection()
        ));
        assert_eq!(store.selections, 0);
        assert_eq!(store.readers, 0);
    }

    #[test]
    fn every_signed_descriptor_data_and_metadata_dependency_is_mandatory() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut base = MemoryRepo::default();
        let (commit, _) = publish(&mut base, &descriptor, 7, 1);
        let descriptor_handle = identity_for_tests(&descriptor);
        let data_handle = Handle::<SimpleArchive>::from_hash(commit.data());
        let metadata_handle = commit.metadata();

        let mut missing_descriptor = base.clone();
        missing_descriptor.blobs.keep([
            data_handle.transmute::<Handle<UnknownBlob>>(),
            metadata_handle.transmute::<Handle<UnknownBlob>>(),
        ]);
        assert!(matches!(
            facade.attach_exact(&mut missing_descriptor, &[commit]),
            Err(CollectionMaterializationError::DescriptorGet { collection, .. })
                if collection == descriptor_handle
        ));

        let mut corrupt_descriptor = base.clone();
        corrupt_descriptor.blobs.keep([
            data_handle.transmute::<Handle<UnknownBlob>>(),
            metadata_handle.transmute::<Handle<UnknownBlob>>(),
        ]);
        let wrong_descriptor = crate::blob::IntoBlob::<SimpleArchive>::to_blob(
            test_facade("ninth").descriptor().into_facts(),
        );
        corrupt_descriptor
            .blobs
            .insert(Blob::with_handle(wrong_descriptor.bytes, descriptor_handle));
        assert!(matches!(
            facade.attach_exact(&mut corrupt_descriptor, &[commit]),
            Err(CollectionMaterializationError::DescriptorIdentity { expected, .. })
                if expected == descriptor_handle
        ));

        let mut missing_data = base.clone();
        missing_data.blobs.keep([
            descriptor_handle.transmute::<Handle<UnknownBlob>>(),
            metadata_handle.transmute::<Handle<UnknownBlob>>(),
        ]);
        assert!(matches!(
            facade.attach_exact(&mut missing_data, &[commit]),
            Err(CollectionMaterializationError::CommitDataGet { commit: found, .. })
                if found == commit.id()
        ));

        let mut corrupt_data = missing_data;
        corrupt_data
            .blobs
            .insert(Blob::with_handle(Bytes::from(vec![0; 63]), data_handle));
        assert!(matches!(
            facade.attach_exact(&mut corrupt_data, &[commit]),
            Err(CollectionMaterializationError::InvalidCommitData { commit: found, .. })
                if found == commit.id()
        ));

        let mut missing_metadata = base.clone();
        missing_metadata.blobs.keep([
            descriptor_handle.transmute::<Handle<UnknownBlob>>(),
            data_handle.transmute::<Handle<UnknownBlob>>(),
        ]);
        assert!(matches!(
            facade.attach_exact(&mut missing_metadata, &[commit]),
            Err(CollectionMaterializationError::CommitMetadataGet { commit: found, .. })
                if found == commit.id()
        ));

        let mut corrupt_metadata = missing_metadata;
        corrupt_metadata
            .blobs
            .insert(Blob::with_handle(Bytes::from(vec![0; 63]), metadata_handle));
        assert!(matches!(
            facade.attach_exact(&mut corrupt_metadata, &[commit]),
            Err(CollectionMaterializationError::InvalidCommitMetadata { commit: found, .. })
                if found == commit.id()
        ));
    }

    #[test]
    fn exact_merge_cover_cannot_import_an_unselected_commit() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut store = MemoryRepo::default();
        let (first, first_blob) = publish(&mut store, &descriptor, 7, 1);
        let (second, second_blob) = publish(&mut store, &descriptor, 8, 2);
        let (_extra, extra_blob) = publish(&mut store, &descriptor, 9, 3);
        let (_, selected_cover) = super::super::publish_merge(
            &mut store,
            &descriptor,
            Handle::<SimpleArchive>::to_hash(first_blob.get_handle()),
            Handle::<SimpleArchive>::to_hash(second_blob.get_handle()),
        )
        .unwrap();
        super::super::publish_merge(
            &mut store,
            &descriptor,
            Handle::<SimpleArchive>::to_hash(selected_cover.get_handle()),
            Handle::<SimpleArchive>::to_hash(extra_blob.get_handle()),
        )
        .unwrap();

        let attached = facade.attach_exact(&mut store, &[second, first]).unwrap();
        let mut expected = facts(1);
        expected += facts(2);
        assert_eq!(attached, expected);
    }

    #[test]
    fn empty_attach_is_store_free_while_empty_snapshot_returns_one_reader() {
        let facade = test_facade("first");
        let mut store = ReadOnlyCountingStore::default();

        assert_eq!(
            facade.attach_exact(&mut store, &[]).unwrap(),
            TribleSet::new()
        );
        assert_eq!((store.selections, store.readers), (0, 0));

        let snapshot = facade.snapshot_exact(&mut store, &[]).unwrap();
        assert_eq!(snapshot.facts(), &TribleSet::new());
        assert!(snapshot.commits().is_empty());
        assert_eq!((store.selections, store.readers), (0, 1));
    }
}
