//! Read-only exact-cover facade for one scoped `SimpleArchive` collection.
//!
//! [`SimpleArchiveCollection`] carries only the fixed canonical descriptor.
//! Callers supply one opaque payload cover on every read; no signing key,
//! mutable head, signer-wide discovery rule, or write surface is present.
//! Signatures and metadata remain queryable provenance over those payloads and
//! are not coordinates of the materialized value.

// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use crate::collection::reach;
use ed25519_dalek::VerifyingKey;

use std::convert::Infallible;

use crate::collection::api::{
    snapshot_from_observation, FactCover, FactMaterializationError, FactSnapshot,
};
use crate::collection::discovery::discover_collection_equations_for_cover;
use crate::collection::{Collection, CollectionStore, DiscoveredCollectionRecords};
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use crate::trible::{Fragment, TribleSet};

/// Read-only exact-cover view of one canonical `SimpleArchive` union.
///
/// The scope fixes the descriptor, while each call supplies the complete
/// explicit payload set. The facade borrows storage only for the duration of a
/// read and has no API capable of inserting blobs or collection records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SimpleArchiveCollection {
    name: String,
    authority: VerifyingKey,
    reach: Fragment,
}

impl SimpleArchiveCollection {
    /// Construct a read-only exact-cover facade for one named root.
    ///
    /// `reach` is not decoration on a read facade: it is part of the
    /// descriptor this facade hashes, so a facade that names the wrong reach
    /// names a different collection and matches no cover.
    pub fn new(name: impl Into<String>, authority: VerifyingKey, reach: Fragment) -> Self {
        Self {
            name: name.into(),
            authority,
            reach,
        }
    }

    /// How far this collection may travel.
    pub fn reach(&self) -> &Fragment {
        &self.reach
    }

    /// Human-readable name of this root collection.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Mandatory capability trust root in this descriptor.
    pub fn authority(&self) -> VerifyingKey {
        self.authority
    }

    /// Canonical `SimpleArchive` set-union descriptor facts.
    pub fn descriptor(&self) -> Fragment {
        super::descriptor(&self.name, self.authority, self.reach.clone())
    }

    /// Content identity of this collection's descriptor.
    ///
    /// This is the read side: the facade is not storing anything, it is
    /// naming the collection a cover must match. A write path takes its
    /// handle from what `put` returns instead.
    pub fn collection(&self) -> Collection<super::SimpleArchive> {
        Collection::from_descriptor(&self.descriptor())
            .expect("SimpleArchiveCollection constructs its own typed descriptor")
    }

    /// Attach the exact opaque cover as a materialized fact set without writing.
    ///
    /// Every member must belong to this exact descriptor and have resident,
    /// canonical data. Same-descriptor `MERGE` records may provide an exact
    /// physical realization, but never add payload members. Other commits
    /// and provenance claims in storage are inert.
    ///
    /// An empty cover returns the local empty set without touching storage.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        cover: &FactCover,
    ) -> Result<
        TribleSet,
        FactMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        if cover.collection() != self.collection() {
            return Err(FactMaterializationError::ExactCover(
                crate::collection::ExactCoverError::WrongCollection {
                    expected: self.collection().handle(),
                    actual: cover.collection().handle(),
                },
            ));
        }
        if cover.is_empty() {
            return Ok(TribleSet::new());
        }
        self.snapshot_canonical(store, cover.clone())
            .map(FactSnapshot::into_facts)
    }

    /// Capture one coherent exact-cover fact, cover, and reader snapshot.
    ///
    /// Record selection happens once before one blob reader is opened. The
    /// returned payload members are canonical and duplicate-free, and only
    /// those members can contribute facts even if the reader physically
    /// contains later or otherwise unselected blobs. An empty cover still
    /// opens and returns a reader, matching
    /// [`crate::collection::CollectionStoreExt::snapshot`].
    pub fn snapshot_exact<S>(
        &self,
        store: &mut S,
        cover: &FactCover,
    ) -> Result<
        FactSnapshot<S::Reader>,
        FactMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        if cover.collection() != self.collection() {
            return Err(FactMaterializationError::ExactCover(
                crate::collection::ExactCoverError::WrongCollection {
                    expected: self.collection().handle(),
                    actual: cover.collection().handle(),
                },
            ));
        }
        self.snapshot_canonical(store, cover.clone())
    }

    fn snapshot_canonical<S>(
        &self,
        store: &mut S,
        cover: FactCover,
    ) -> Result<
        FactSnapshot<S::Reader>,
        FactMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    >
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let descriptor = self.descriptor();
        if cover.is_empty() {
            return snapshot_from_observation(
                store,
                &descriptor,
                DiscoveredCollectionRecords::default(),
                cover,
            );
        }

        let discovered = discover_collection_equations_for_cover(store, &cover)
            .map_err(FactMaterializationError::Discovery)?;
        snapshot_from_observation(store, &descriptor, discovered, cover)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::encodings::{simplearchive::SimpleArchive, UnknownBlob};
    use crate::blob::{Blob, BlobEncoding, Bytes, IntoBlob};
    use crate::collection::descriptor::identity_for_tests;
    use crate::collection::{
        CollectionCommit, CollectionRecord, CollectionRecordSelector, CollectionStoreExt,
        ExactCoverError,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::BlobStorePut;
    use crate::trible::{Trible, TRIBLE_LEN};

    fn test_facade(name: &str) -> SimpleArchiveCollection {
        SimpleArchiveCollection::new(
            name.to_owned(),
            SigningKey::from_bytes(&[1; 32]).verifying_key(),
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

    fn cover(
        facade: &SimpleArchiveCollection,
        commits: impl IntoIterator<Item = CollectionCommit>,
    ) -> FactCover {
        FactCover::from_data(
            facade.collection(),
            commits.into_iter().map(|commit| commit.data()),
        )
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
    fn mixed_authors_form_one_exact_sorted_set_and_unselected_claims_are_inert() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut store = MemoryRepo::default();
        let (first, _) = publish(&mut store, &descriptor, 7, 1);
        let (second, _) = publish(&mut store, &descriptor, 8, 2);
        let (_unselected, _) = publish(&mut store, &descriptor, 9, 3);

        let requested = cover(&facade, [second, first, first]);
        let snapshot = facade.snapshot_exact(&mut store, &requested).unwrap();
        let mut expected = facts(1);
        expected += facts(2);
        let expected_cover = cover(&facade, [first, second]);

        assert_eq!(snapshot.facts(), &expected);
        assert_eq!(snapshot.cover(), &expected_cover);
        assert_eq!(snapshot.cover().len(), 2);
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

        let requested = cover(&facade, [commit]);
        let snapshot = facade.snapshot_exact(&mut store, &requested).unwrap();

        assert_eq!(snapshot.facts(), &facts(1));
        assert_eq!(snapshot.cover(), &requested);
        assert_eq!(store.selections, 1);
        assert_eq!(store.readers, 1);
        assert_eq!(
            store.last_selectors,
            Some(BTreeSet::from([CollectionRecordSelector::MergeCollection(
                identity_for_tests(&descriptor),
            )])),
        );
    }

    #[test]
    fn duplicate_same_data_claims_and_their_metadata_do_not_change_the_cover() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut store = MemoryRepo::default();
        let (first, data) = publish(&mut store, &descriptor, 7, 1);
        let missing_metadata = Inline::new([42; 32]);
        let missing_metadata_claim = CollectionCommit::sign(
            &SigningKey::from_bytes(&[8; 32]),
            first.collection(),
            first.data(),
            missing_metadata,
        );
        let corrupt_metadata = Inline::new([43; 32]);
        let corrupt_metadata_claim = CollectionCommit::sign(
            &SigningKey::from_bytes(&[9; 32]),
            first.collection(),
            first.data(),
            corrupt_metadata,
        );
        store
            .insert(CollectionRecord::Commit(missing_metadata_claim))
            .unwrap();
        store
            .insert(CollectionRecord::Commit(corrupt_metadata_claim))
            .unwrap();

        let descriptor_handle = identity_for_tests(&descriptor);
        let data_handle = data.get_handle();
        store.blobs.keep([
            descriptor_handle.transmute::<Handle<UnknownBlob>>(),
            data_handle.transmute::<Handle<UnknownBlob>>(),
        ]);
        store.blobs.insert(Blob::with_handle(
            Bytes::from(vec![0; 63]),
            corrupt_metadata,
        ));

        let requested = cover(
            &facade,
            [first, missing_metadata_claim, corrupt_metadata_claim],
        );
        assert_eq!(requested.len(), 1);
        assert_eq!(
            facade.attach_exact(&mut store, &requested).unwrap(),
            facts(1)
        );

        let claims = store.claims(&requested).unwrap();
        assert_eq!(claims.len(), 3);
        assert!(claims.contains(&first));
        assert!(claims.contains(&missing_metadata_claim));
        assert!(claims.contains(&corrupt_metadata_claim));
    }

    #[test]
    fn an_opaque_cover_replays_payload_without_rechecking_its_claim() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();

        let mut source = MemoryRepo::default();
        let (absent, _) = publish(&mut source, &descriptor, 7, 1);
        let requested = cover(&facade, [absent]);
        let mut claimless = MemoryRepo::default();
        claimless.blobs = source.blobs.clone();
        assert_eq!(
            facade.attach_exact(&mut claimless, &requested).unwrap(),
            facts(1)
        );

        // A different author's valid signature is inert provenance over the
        // same payload. Replay neither requires nor rechecks it.
        let alternate_author = CollectionCommit::sign(
            &SigningKey::from_bytes(&[8; 32]),
            absent.collection(),
            absent.data(),
            absent.metadata(),
        );
        let mut alternate_store = MemoryRepo::default();
        alternate_store.blobs = source.blobs.clone();
        alternate_store
            .insert(CollectionRecord::Commit(alternate_author))
            .unwrap();
        assert_eq!(
            facade
                .attach_exact(&mut alternate_store, &requested)
                .unwrap(),
            facts(1)
        );

        let invalid = invalid_signature(absent);
        let mut invalid_store = MemoryRepo::default();
        invalid_store.blobs = source.blobs;
        invalid_store
            .insert(CollectionRecord::Commit(invalid))
            .unwrap();
        assert_eq!(
            facade.attach_exact(&mut invalid_store, &requested).unwrap(),
            facts(1)
        );
    }

    #[test]
    fn a_cover_for_another_descriptor_is_rejected_before_storage_access() {
        let facade = test_facade("first");
        let other = test_facade("second");
        let mut source = MemoryRepo::default();
        let (commit, _) = publish(&mut source, &other.descriptor(), 7, 1);
        let requested = cover(&other, [commit]);
        let mut store = ReadOnlyCountingStore {
            inner: source,
            ..ReadOnlyCountingStore::default()
        };

        assert!(matches!(
            facade.attach_exact(&mut store, &requested),
            Err(FactMaterializationError::ExactCover(
                ExactCoverError::WrongCollection {
                    expected,
                    actual,
                }
            )) if expected == facade.collection().handle()
                && actual == other.collection().handle()
        ));
        assert_eq!(store.selections, 0);
        assert_eq!(store.readers, 0);
    }

    #[test]
    fn descriptor_and_member_data_are_mandatory() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut base = MemoryRepo::default();
        let (commit, _) = publish(&mut base, &descriptor, 7, 1);
        let requested = cover(&facade, [commit]);
        let descriptor_handle = identity_for_tests(&descriptor);
        let data_handle = Handle::<SimpleArchive>::from_hash(commit.data());

        let mut missing_descriptor = base.clone();
        missing_descriptor
            .blobs
            .keep([data_handle.transmute::<Handle<UnknownBlob>>()]);
        assert!(matches!(
            facade.attach_exact(&mut missing_descriptor, &requested),
            Err(FactMaterializationError::DescriptorGet { collection, .. })
                if collection == descriptor_handle
        ));

        let mut corrupt_descriptor = base.clone();
        corrupt_descriptor
            .blobs
            .keep([data_handle.transmute::<Handle<UnknownBlob>>()]);
        let wrong_descriptor = crate::blob::IntoBlob::<SimpleArchive>::to_blob(
            test_facade("ninth").descriptor().into_facts(),
        );
        corrupt_descriptor
            .blobs
            .insert(Blob::with_handle(wrong_descriptor.bytes, descriptor_handle));
        assert!(matches!(
            facade.attach_exact(&mut corrupt_descriptor, &requested),
            Err(FactMaterializationError::DescriptorIdentity { expected, .. })
                if expected == descriptor_handle
        ));

        let mut missing_data = base.clone();
        missing_data
            .blobs
            .keep([descriptor_handle.transmute::<Handle<UnknownBlob>>()]);
        assert!(matches!(
            facade.attach_exact(&mut missing_data, &requested),
            Err(FactMaterializationError::MemberGet { member, .. })
                if member == commit.data()
        ));

        let mut corrupt_data = missing_data;
        corrupt_data
            .blobs
            .insert(Blob::with_handle(Bytes::from(vec![0; 63]), data_handle));
        assert!(matches!(
            facade.attach_exact(&mut corrupt_data, &requested),
            Err(FactMaterializationError::InvalidMember { member, .. })
                if member == commit.data()
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

        let requested = cover(&facade, [second, first]);
        let attached = facade.attach_exact(&mut store, &requested).unwrap();
        let mut expected = facts(1);
        expected += facts(2);
        assert_eq!(attached, expected);
    }

    #[test]
    fn empty_attach_is_store_free_while_empty_snapshot_returns_one_reader() {
        let facade = test_facade("first");
        let mut store = ReadOnlyCountingStore::default();
        let empty = FactCover::from_data(facade.collection(), std::iter::empty());

        assert_eq!(
            facade.attach_exact(&mut store, &empty).unwrap(),
            TribleSet::new()
        );
        assert_eq!((store.selections, store.readers), (0, 0));

        let snapshot = facade.snapshot_exact(&mut store, &empty).unwrap();
        assert_eq!(snapshot.facts(), &TribleSet::new());
        assert!(snapshot.cover().is_empty());
        assert_eq!(snapshot.cover(), &empty);
        assert_eq!((store.selections, store.readers), (0, 1));
    }
}
