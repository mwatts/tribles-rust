//! Durable positive local willingness to serve content-addressed artifacts.
//!
//! An [`ArtifactOfferStore`] records only local service intent. An offer grants
//! no authority, proves neither reachability nor residency, creates no demand,
//! and is not a garbage-collection root. The algebra is a grow-only set: there
//! is intentionally no retraction operation.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Debug};

use crate::blob::encodings::UnknownBlob;
use crate::blob::{BlobEncoding, IntoBlob};
use crate::collection::{CollectionRecord, CollectionRecordSelector, CollectionStore};
use crate::id::{id_hex, Id};
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding, INLINE_LEN};
use crate::patch::{Entry, IdentitySchema, XorSip128, PATCH};
use crate::repo::{BlobStore, BlobStorePut};

/// Stable semantic kind of a positive local artifact offer.
///
/// Minted with `trible genid` on 2026-08-27. The pile record description is
/// rooted at this same anchor.
pub const KIND_ARTIFACT_OFFER: Id = id_hex!("6EE89EEA7E6ECB2463FA5EE9C955B378");

/// Content-addressed artifact named by an offer.
pub type ArtifactHandle = Inline<Handle<UnknownBlob>>;

type ArtifactOfferIndex = PATCH<INLINE_LEN, IdentitySchema, (), XorSip128>;

/// Cheap immutable deterministic view of the offers known to one store.
///
/// Cloning this snapshot is O(1). Iteration is in canonical handle-byte order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArtifactOfferSnapshot {
    offers: ArtifactOfferIndex,
}

impl ArtifactOfferSnapshot {
    /// Number of distinct offered artifacts.
    pub fn len(&self) -> u64 {
        self.offers.len()
    }

    /// Whether this snapshot contains no offers.
    pub fn is_empty(&self) -> bool {
        self.offers.is_empty()
    }

    /// Whether this snapshot contains `handle`.
    pub fn contains(&self, handle: ArtifactHandle) -> bool {
        self.offers.get(&handle.raw).is_some()
    }

    /// Iterate over offered handles in canonical byte order.
    pub fn iter(&self) -> impl Iterator<Item = ArtifactHandle> + '_ {
        self.offers
            .iter_ordered()
            .map(|raw| ArtifactHandle::new(*raw))
    }

    pub(crate) fn insert(&mut self, handle: ArtifactHandle) {
        self.offers.insert(&Entry::new(&handle.raw));
    }

    pub(crate) fn union(&mut self, other: Self) {
        self.offers.union(other.offers);
    }
}

/// Grow-only storage for positive local willingness to serve artifacts.
///
/// The bulk operation is primary so durable backends can refresh, lock, and
/// append once per batch. Implementations should deduplicate both the input
/// batch and offers they already know. If an error follows one or more durable
/// insertions, those insertions lawfully remain as partial grow-only success.
pub trait ArtifactOfferStore {
    /// Failure while observing or extending the offer set.
    type OfferError: Error + Debug + Send + Sync + 'static;

    /// Add every distinct handle in `handles` to the local offer set.
    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>;

    /// Add one handle to the local offer set.
    fn offer(&mut self, handle: ArtifactHandle) -> Result<(), Self::OfferError> {
        self.offer_all([handle])
    }

    /// Take a cheap immutable deterministic snapshot of the current offer set.
    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError>;
}

impl<S> ArtifactOfferStore for &mut S
where
    S: ArtifactOfferStore + ?Sized,
{
    type OfferError = S::OfferError;

    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>,
    {
        (**self).offer_all(handles)
    }

    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError> {
        (**self).offers_snapshot()
    }
}

impl<S> ArtifactOfferStore for OfferCapture<S>
where
    S: ArtifactOfferStore,
{
    type OfferError = S::OfferError;

    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>,
    {
        self.inner.offer_all(handles)
    }

    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError> {
        self.inner.offers_snapshot()
    }
}

/// A publication-scoped adapter that turns successful blob writes into OFFER
/// intent immediately before the semantic record that names them.
///
/// This is deliberately not a permanent storage facade. It collects only puts
/// made through this value, offers their returned handles as one canonical
/// batch at the next successful [`CollectionStore::insert`] boundary, and
/// then delegates that insert. A failed operation can therefore abandon the
/// adapter without an unrelated later record accidentally publishing its
/// orphan writes.
///
/// The ordering is:
///
/// ```text
/// dependency/result puts -> OFFER markers -> COMMIT/MERGE/DERIVE record
/// ```
///
/// OFFER is neither authority nor retention. If the final record insert
/// fails, the preceding positive offers merely name resident content and are
/// safe to retain. No durability flush is implied.
pub struct OfferCapture<S> {
    inner: S,
    pending: BTreeSet<ArtifactHandle>,
}

impl<S> OfferCapture<S> {
    /// Begin one scoped publication session over `inner`.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            pending: BTreeSet::new(),
        }
    }

    /// Borrow the wrapped store without bypassing captured writes.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Handles awaiting the next semantic record, in canonical byte order.
    pub fn pending(&self) -> impl ExactSizeIterator<Item = ArtifactHandle> + '_ {
        self.pending.iter().copied()
    }

    /// Consume the scope and recover its store.
    ///
    /// Any pending handles are deliberately abandoned. This is the expected
    /// cleanup path after a semantic operation fails before its record insert.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S> OfferCapture<S>
where
    S: ArtifactOfferStore,
{
    /// Retry the exact pending OFFER batch without inserting a semantic record.
    ///
    /// Partial success is lawful because offers are grow-only. On failure the
    /// complete request remains pending, so retrying all handles is idempotent.
    pub fn retry_offers(&mut self) -> Result<(), S::OfferError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.inner.offer_all(self.pending.iter().copied())?;
        self.pending.clear();
        Ok(())
    }
}

/// Failure of the OFFER gate or of the semantic record insert behind it.
#[derive(Debug)]
pub enum OfferCaptureInsertError<OfferError, InsertError> {
    /// OFFER intent could not be completely recorded, so the semantic record
    /// was not attempted. `artifacts` is the full canonical retry-all batch,
    /// including handles that may already have succeeded partially.
    Offer {
        /// Backend offer failure.
        source: OfferError,
        /// Exact deterministic batch safe to replay in full.
        artifacts: Vec<ArtifactHandle>,
    },
    /// OFFER succeeded, but the semantic record insert failed.
    Insert(InsertError),
}

impl<OfferError, InsertError> fmt::Display for OfferCaptureInsertError<OfferError, InsertError>
where
    OfferError: fmt::Display,
    InsertError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offer { source, artifacts } => write!(
                f,
                "failed to offer {} publication artifact(s) before semantic record insertion: {source}",
                artifacts.len(),
            ),
            Self::Insert(source) => write!(f, "failed to insert semantic record: {source}"),
        }
    }
}

impl<OfferError, InsertError> Error for OfferCaptureInsertError<OfferError, InsertError>
where
    OfferError: Error + 'static,
    InsertError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Offer { source, .. } => Some(source),
            Self::Insert(source) => Some(source),
        }
    }
}

impl<S> BlobStorePut for OfferCapture<S>
where
    S: BlobStorePut,
{
    type PutError = S::PutError;

    fn put<E, T>(&mut self, item: T) -> Result<Inline<Handle<E>>, Self::PutError>
    where
        E: BlobEncoding + 'static,
        T: IntoBlob<E>,
        Handle<E>: InlineEncoding,
    {
        let handle = self.inner.put(item)?;
        self.pending.insert(handle.transmute());
        Ok(handle)
    }
}

impl<S> BlobStore for OfferCapture<S>
where
    S: BlobStore,
{
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl<S> CollectionStore for OfferCapture<S>
where
    S: CollectionStore + ArtifactOfferStore,
{
    type RecordsError = S::RecordsError;
    type InsertError = OfferCaptureInsertError<S::OfferError, S::InsertError>;
    type RecordIter<'a>
        = S::RecordIter<'a>
    where
        Self: 'a;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
    }

    fn record(&mut self, id: Id) -> Result<Option<CollectionRecord>, Self::RecordsError> {
        self.inner.record(id)
    }

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        self.inner.select_records(selectors)
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        if let Err(source) = self.retry_offers() {
            return Err(OfferCaptureInsertError::Offer {
                source,
                artifacts: self.pending.iter().copied().collect(),
            });
        }
        self.inner
            .insert(record)
            .map_err(OfferCaptureInsertError::Insert)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::{Blob, Bytes};
    use crate::collection::{CollectionData, CollectionHandle, CollectionMerge};
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::StoreRevision;
    use std::convert::Infallible;

    fn handle(byte: u8) -> ArtifactHandle {
        ArtifactHandle::new([byte; INLINE_LEN])
    }

    #[test]
    fn memory_offers_are_grow_only_deduplicated_and_snapshotted() {
        let mut repo = MemoryRepo::default();
        repo.offer_all([handle(2), handle(1), handle(2)]).unwrap();
        let before = repo.offers_snapshot().unwrap();

        assert_eq!(
            before.iter().collect::<Vec<_>>(),
            vec![handle(1), handle(2)]
        );
        assert!(before.contains(handle(1)));
        assert_eq!(before.len(), 2);

        repo.offer(handle(3)).unwrap();
        assert_eq!(
            before.len(),
            2,
            "snapshots do not change under later writes"
        );
        assert_eq!(repo.offers_snapshot().unwrap().len(), 3);
    }

    #[test]
    fn local_offers_do_not_change_sync_visible_revision() {
        let mut repo = MemoryRepo::default();
        let before = repo.store_revision().unwrap();
        repo.offer(handle(9)).unwrap();
        let after = repo.store_revision().unwrap();
        assert!(before == after);
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum ProbeEvent {
        Put(ArtifactHandle),
        Offer(Vec<ArtifactHandle>),
        Insert(Id),
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ProbeError(&'static str);

    impl fmt::Display for ProbeError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for ProbeError {}

    #[derive(Default)]
    struct ProbeStore {
        events: Vec<ProbeEvent>,
        offers: ArtifactOfferSnapshot,
        records: Vec<CollectionRecord>,
        fail_offer: bool,
        fail_insert: bool,
    }

    impl BlobStorePut for ProbeStore {
        type PutError = Infallible;

        fn put<E, T>(&mut self, item: T) -> Result<Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            let handle = item.to_blob().get_handle();
            self.events.push(ProbeEvent::Put(handle.transmute()));
            Ok(handle)
        }
    }

    impl ArtifactOfferStore for ProbeStore {
        type OfferError = ProbeError;

        fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
        where
            I: IntoIterator<Item = ArtifactHandle>,
        {
            let mut handles: Vec<_> = handles.into_iter().collect();
            handles.sort_unstable();
            handles.dedup();
            self.events.push(ProbeEvent::Offer(handles.clone()));
            if self.fail_offer {
                if let Some(first) = handles.first().copied() {
                    self.offers.insert(first);
                }
                return Err(ProbeError("offer"));
            }
            for handle in handles {
                self.offers.insert(handle);
            }
            Ok(())
        }

        fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError> {
            Ok(self.offers.clone())
        }
    }

    impl CollectionStore for ProbeStore {
        type RecordsError = Infallible;
        type InsertError = ProbeError;
        type RecordIter<'a>
            = std::vec::IntoIter<Result<CollectionRecord, Infallible>>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            Ok(self
                .records
                .clone()
                .into_iter()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.events.push(ProbeEvent::Insert(record.id()));
            if self.fail_insert {
                return Err(ProbeError("insert"));
            }
            self.records.push(record);
            Ok(())
        }
    }

    fn merge_record(byte: u8) -> CollectionRecord {
        let collection = CollectionHandle::new([byte; INLINE_LEN]);
        let data = |offset| CollectionData::new([byte.wrapping_add(offset); INLINE_LEN]);
        CollectionRecord::Merge(CollectionMerge::new(collection, data(1), data(2), data(3)))
    }

    fn blob(bytes: &'static [u8]) -> Blob<UnknownBlob> {
        Blob::new(Bytes::from_source(bytes))
    }

    #[test]
    fn capture_orders_puts_then_one_canonical_offer_then_record() {
        let mut capture = OfferCapture::new(ProbeStore::default());
        let first = capture.put::<UnknownBlob, _>(blob(b"first")).unwrap();
        let second = capture.put::<UnknownBlob, _>(blob(b"second")).unwrap();
        let record = merge_record(7);
        capture.insert(record).unwrap();

        let mut offered = vec![first, second];
        offered.sort_unstable();
        assert_eq!(
            capture.inner().events,
            vec![
                ProbeEvent::Put(first),
                ProbeEvent::Put(second),
                ProbeEvent::Offer(offered),
                ProbeEvent::Insert(record.id()),
            ]
        );
        assert!(capture.pending().next().is_none());
    }

    #[test]
    fn offer_failure_withholds_record_and_retains_full_retry_batch() {
        let mut store = ProbeStore {
            fail_offer: true,
            ..ProbeStore::default()
        };
        let mut capture = OfferCapture::new(&mut store);
        let first = capture.put::<UnknownBlob, _>(blob(b"a")).unwrap();
        let second = capture.put::<UnknownBlob, _>(blob(b"b")).unwrap();
        let record = merge_record(9);
        let error = capture.insert(record).unwrap_err();
        let OfferCaptureInsertError::Offer { artifacts, .. } = error else {
            panic!("offer failure must precede record insertion")
        };
        let mut expected = vec![first, second];
        expected.sort_unstable();
        assert_eq!(artifacts, expected);
        assert_eq!(capture.pending().collect::<Vec<_>>(), expected);
        assert!(capture.inner().records.is_empty());
        assert!(
            capture.inner().offers.len() == 1,
            "partial success is retained"
        );

        capture.inner.fail_offer = false;
        capture.insert(record).unwrap();
        assert!(capture.pending().next().is_none());
        assert_eq!(capture.inner().records, vec![record]);
        assert_eq!(capture.inner().offers.len(), 2);
    }

    #[test]
    fn record_failure_leaves_offers_and_replay_remains_idempotent() {
        let mut capture = OfferCapture::new(ProbeStore {
            fail_insert: true,
            ..ProbeStore::default()
        });
        let artifact = capture.put::<UnknownBlob, _>(blob(b"resident")).unwrap();
        let record = merge_record(11);
        assert!(matches!(
            capture.insert(record),
            Err(OfferCaptureInsertError::Insert(ProbeError("insert")))
        ));
        assert!(capture.inner().offers.contains(artifact));
        assert!(capture.inner().records.is_empty());

        capture.inner.fail_insert = false;
        capture.insert(record).unwrap();
        assert_eq!(capture.inner().records, vec![record]);
        assert_eq!(capture.inner().offers.len(), 1);
    }

    #[test]
    fn abandoned_scope_cannot_flush_into_a_later_publication() {
        let mut store = ProbeStore::default();
        let orphan = {
            let mut abandoned = OfferCapture::new(&mut store);
            abandoned.put::<UnknownBlob, _>(blob(b"orphan")).unwrap()
        };
        let record = merge_record(13);
        OfferCapture::new(&mut store).insert(record).unwrap();

        assert!(!store.offers.contains(orphan));
        assert_eq!(
            store.events,
            vec![ProbeEvent::Put(orphan), ProbeEvent::Insert(record.id())]
        );
    }
}
