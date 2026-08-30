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

use crate::collection::api::{resolve_cover_from_observation, FactCover, FactMaterializationError};
use crate::collection::discovery::discover_collection_equations_for_cover;
use crate::collection::{Collection, CollectionRead, TryFromCover};
use crate::repo::{BlobStoreGet, BlobStoreMeta};
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
        snapshot: &S,
        cover: &FactCover,
    ) -> Result<TribleSet, FactMaterializationError<S::RecordsError, S::GetError<Infallible>>>
    where
        S: BlobStoreGet + BlobStoreMeta + CollectionRead,
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
        let resolved = self.resolve_canonical(snapshot, cover.clone())?;
        TribleSet::try_from_cover(&resolved, snapshot).map_err(FactMaterializationError::from)
    }

    fn resolve_canonical<S>(
        &self,
        snapshot: &S,
        cover: FactCover,
    ) -> Result<FactCover, FactMaterializationError<S::RecordsError, S::GetError<Infallible>>>
    where
        S: BlobStoreGet + BlobStoreMeta + CollectionRead,
    {
        let descriptor = self.descriptor();
        if cover.is_empty() {
            return Ok(cover);
        }

        let discovered = discover_collection_equations_for_cover(snapshot, &cover)
            .map_err(FactMaterializationError::Discovery)?;
        resolve_cover_from_observation::<S, super::SimpleArchive, super::FactViewError, Infallible>(
            snapshot,
            &descriptor,
            discovered,
            cover,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::Infallible;
    use std::error::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::encodings::{simplearchive::SimpleArchive, UnknownBlob};
    use crate::blob::{Blob, BlobEncoding, Bytes, IntoBlob, TryFromBlob};
    use crate::collection::descriptor::identity_for_tests;
    use crate::collection::{
        CollectionCommit, CollectionRead, CollectionRecord, CollectionRecordSelector,
        CollectionStore, ExactCoverError,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::repo::memoryrepo::{MemoryRepo, MemoryRepoSnapshot};
    use crate::repo::{BlobMetadata, BlobStoreGet, BlobStoreMeta, SnapshotSource, StoreSnapshot};
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
        selections: Arc<AtomicUsize>,
        last_selectors: Arc<Mutex<Option<BTreeSet<CollectionRecordSelector>>>>,
    }

    #[derive(Clone)]
    struct ReadOnlyCountingSnapshot {
        inner: MemoryRepoSnapshot,
        selections: Arc<AtomicUsize>,
        last_selectors: Arc<Mutex<Option<BTreeSet<CollectionRecordSelector>>>>,
    }

    impl StoreSnapshot for ReadOnlyCountingSnapshot {}

    impl BlobStoreGet for ReadOnlyCountingSnapshot {
        type GetError<E: Error + Send + Sync + 'static> =
            <MemoryRepoSnapshot as BlobStoreGet>::GetError<E>;

        fn get<T, E>(
            &self,
            handle: Inline<Handle<E>>,
        ) -> Result<T, Self::GetError<<T as TryFromBlob<E>>::Error>>
        where
            E: BlobEncoding,
            T: TryFromBlob<E>,
        {
            self.inner.get(handle)
        }
    }

    impl BlobStoreMeta for ReadOnlyCountingSnapshot {
        type MetaError = <MemoryRepoSnapshot as BlobStoreMeta>::MetaError;

        fn metadata<E>(
            &self,
            handle: Inline<Handle<E>>,
        ) -> Result<Option<BlobMetadata>, Self::MetaError>
        where
            E: BlobEncoding + 'static,
            Handle<E>: InlineEncoding,
        {
            self.inner.metadata(handle)
        }
    }

    impl CollectionRead for ReadOnlyCountingSnapshot {
        type RecordsError = <MemoryRepoSnapshot as CollectionRead>::RecordsError;
        type RecordIter<'a> = <MemoryRepoSnapshot as CollectionRead>::RecordIter<'a>;

        fn records<'a>(&'a self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            panic!("exact facade must use the selector boundary")
        }

        fn select_records(
            &self,
            selectors: &BTreeSet<CollectionRecordSelector>,
        ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
            self.selections.fetch_add(1, Ordering::SeqCst);
            *self.last_selectors.lock().expect("selector mutex") = Some(selectors.clone());
            self.inner.select_records(selectors)
        }
    }

    impl SnapshotSource for ReadOnlyCountingStore {
        type Snapshot = ReadOnlyCountingSnapshot;
        type SnapshotError = Infallible;

        fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
            Ok(ReadOnlyCountingSnapshot {
                inner: self.inner.snapshot()?,
                selections: self.selections.clone(),
                last_selectors: self.last_selectors.clone(),
            })
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
        let snapshot = store.snapshot().unwrap();
        let actual = facade.attach_exact(&snapshot, &requested).unwrap();
        let mut expected = facts(1);
        expected += facts(2);
        let expected_cover = cover(&facade, [first, second]);

        assert_eq!(actual, expected);
        assert_eq!(requested, expected_cover);
        assert_eq!(requested.len(), 2);
        assert_ne!(first.public_key(), second.public_key());
    }

    #[test]
    fn exact_reads_select_records_once_from_one_coherent_snapshot() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut inner = MemoryRepo::default();
        let (commit, _) = publish(&mut inner, &descriptor, 7, 1);
        let mut store = ReadOnlyCountingStore {
            inner,
            ..ReadOnlyCountingStore::default()
        };

        let requested = cover(&facade, [commit]);
        let snapshot = store.snapshot().unwrap();
        let actual = facade.attach_exact(&snapshot, &requested).unwrap();

        assert_eq!(actual, facts(1));
        assert_eq!(store.selections.load(Ordering::SeqCst), 1);
        assert_eq!(
            store.last_selectors.lock().expect("selector mutex").clone(),
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
        let corrupt_metadata_blob = Blob::new(Bytes::from(vec![0; 63]));
        let corrupt_metadata = corrupt_metadata_blob.get_handle();
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
        store.blobs.insert(corrupt_metadata_blob);

        let requested = cover(
            &facade,
            [first, missing_metadata_claim, corrupt_metadata_claim],
        );
        assert_eq!(requested.len(), 1);
        let snapshot = store.snapshot().unwrap();
        assert_eq!(
            facade.attach_exact(&snapshot, &requested).unwrap(),
            facts(1)
        );

        let claims = requested.claims(&snapshot).unwrap();
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
        let snapshot = claimless.snapshot().unwrap();
        assert_eq!(
            facade.attach_exact(&snapshot, &requested).unwrap(),
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
        let snapshot = alternate_store.snapshot().unwrap();
        assert_eq!(
            facade.attach_exact(&snapshot, &requested).unwrap(),
            facts(1)
        );

        let invalid = invalid_signature(absent);
        let mut invalid_store = MemoryRepo::default();
        invalid_store.blobs = source.blobs;
        invalid_store
            .insert(CollectionRecord::Commit(invalid))
            .unwrap();
        let snapshot = invalid_store.snapshot().unwrap();
        assert_eq!(
            facade.attach_exact(&snapshot, &requested).unwrap(),
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
        let snapshot = store.snapshot().unwrap();

        assert!(matches!(
            facade.attach_exact(&snapshot, &requested),
            Err(FactMaterializationError::ExactCover(
                ExactCoverError::WrongCollection {
                    expected,
                    actual,
                }
            )) if expected == facade.collection().handle()
                && actual == other.collection().handle()
        ));
        assert_eq!(store.selections.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn locally_bound_descriptor_need_not_be_resident_but_member_data_must() {
        let facade = test_facade("first");
        let descriptor = facade.descriptor();
        let mut base = MemoryRepo::default();
        let (commit, _) = publish(&mut base, &descriptor, 7, 1);
        let requested = cover(&facade, [commit]);
        let data_handle = Handle::<SimpleArchive>::from_hash(commit.data());

        let mut missing_descriptor = base.clone();
        missing_descriptor
            .blobs
            .keep([data_handle.transmute::<Handle<UnknownBlob>>()]);
        let snapshot = missing_descriptor.snapshot().unwrap();
        assert_eq!(
            facade.attach_exact(&snapshot, &requested).unwrap(),
            facts(1),
        );

        let mut missing_data = base.clone();
        missing_data.blobs.keep([]);
        let snapshot = missing_data.snapshot().unwrap();
        assert!(matches!(
            facade.attach_exact(&snapshot, &requested),
            Err(FactMaterializationError::MemberGet { member, .. })
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
        let snapshot = store.snapshot().unwrap();
        let attached = facade.attach_exact(&snapshot, &requested).unwrap();
        let mut expected = facts(1);
        expected += facts(2);
        assert_eq!(attached, expected);
    }

    #[test]
    fn empty_attach_does_not_query_the_snapshot() {
        let facade = test_facade("first");
        let mut store = ReadOnlyCountingStore::default();
        let empty = FactCover::from_data(facade.collection(), std::iter::empty());
        let snapshot = store.snapshot().unwrap();

        assert_eq!(
            facade.attach_exact(&snapshot, &empty).unwrap(),
            TribleSet::new()
        );
        assert_eq!(store.selections.load(Ordering::SeqCst), 0);
    }
}
