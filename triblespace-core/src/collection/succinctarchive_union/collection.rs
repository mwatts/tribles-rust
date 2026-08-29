//! Exact-cover facade for canonical raw SuccinctArchive collections.
//!
//! Unsigned equations are reproducible cache evidence rather than authority or
//! durable receipts: attachment reconstructs collected intermediates in
//! use-counted scratch from explicit source-cover leaves, then freshly validates
//! only the resident artifacts selected by the physical cover. Target
//! compaction is an explicit maintenance call rather than background policy.
//! The raw exact cover remains authoritative and fixes the returned shard shape;
//! a private second stage attaches an exact persisted Rank9 fiber for each
//! selected raw member or rebuilds that accelerator transiently when optional
//! evidence is absent, invalid, or ambiguous.

use ed25519_dalek::VerifyingKey;

use std::cell::Cell;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
#[cfg(test)]
use crate::blob::encodings::succinctarchive::SuccinctArchiveRank9IndexBlob;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchive, SuccinctArchiveBlob, UnionArchive,
};
use crate::collection::exact_derived::{ExactDerivedCollection, ExactDerivedCollectionError};
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::simplearchive_union;
use crate::collection::CoverAdvanceError;
use crate::trible::Fragment;
// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use crate::collection::reach;
#[cfg(test)]
use crate::collection::CollectionEncoding;
use crate::collection::{
    CollectionHandle, CollectionMapping, CollectionOperationError, CollectionStore,
    CoverAttachment, FactCover, MappingEvidenceStore, TryFromCover,
};
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreMeta};

use super::rank9_fiber::Rank9Fiber;

impl TryFromCover<super::SuccinctArchiveBlob> for UnionArchive<OrderedUniverse> {
    type Error = super::Rank9FiberError;

    fn try_from_cover(
        attachment: CoverAttachment<super::SuccinctArchiveBlob>,
    ) -> Result<Self, Self::Error> {
        let mut segments = Vec::with_capacity(attachment.len().max(1));
        for (handle, raw) in attachment.into_members() {
            let raw_data =
                crate::inline::encodings::hash::Handle::<SuccinctArchiveBlob>::to_hash(handle);
            segments.push(
                raw.try_from_blob()
                    .map_err(|source| super::Rank9FiberError::Build {
                        raw: raw_data,
                        source,
                    })?,
            );
        }
        if segments.is_empty() {
            let raw = super::empty();
            let raw_data = crate::inline::encodings::hash::Handle::<SuccinctArchiveBlob>::to_hash(
                raw.get_handle(),
            );
            segments.push(
                raw.try_from_blob()
                    .map_err(|source| super::Rank9FiberError::Build {
                        raw: raw_data,
                        source,
                    })?,
            );
        }
        Ok(UnionArchive::new(segments))
    }
}

/// Failure to complete or attach one exact Succinct cover.
#[derive(Debug)]
pub enum SuccinctArchiveCollectionError {
    /// Exact-cover resolution, construction, or storage failed.
    Exact(ExactDerivedCollectionError),
    /// Explicit target compaction failed.
    Compaction(ExactTargetCompactionError),
    /// Exact Rank9 fiber probing, transient rebuilding, or publication failed.
    Rank9(super::Rank9FiberError),
}

impl fmt::Display for SuccinctArchiveCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(f),
            Self::Compaction(source) => source.fmt(f),
            Self::Rank9(source) => source.fmt(f),
        }
    }
}

impl Error for SuccinctArchiveCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Compaction(source) => Some(source),
            Self::Rank9(source) => Some(source),
        }
    }
}

impl From<ExactDerivedCollectionError> for SuccinctArchiveCollectionError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Exact(source)
    }
}

impl From<ExactTargetCompactionError> for SuccinctArchiveCollectionError {
    fn from(source: ExactTargetCompactionError) -> Self {
        Self::Compaction(source)
    }
}

impl From<super::Rank9FiberError> for SuccinctArchiveCollectionError {
    fn from(source: super::Rank9FiberError) -> Self {
        Self::Rank9(source)
    }
}

/// Canonical raw SuccinctArchive projection of one scoped SimpleArchive union.
///
/// The opaque source cover is the operation's value boundary. Returned query
/// sources preserve the deterministic resident physical cover as Succinct shards.
/// Persisted Rank9 mapping evidence is an optional one-to-one accelerator over
/// that cover: it adds no authority, retention, target lattice, or shard
/// selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuccinctArchiveCollection {
    name: String,
    source_authority: VerifyingKey,
    source_reach: Fragment,
    authority: VerifyingKey,
    reach: Fragment,
}

/// Exact work performed by one successful [`SuccinctArchiveView::ensure`].
///
/// The cover fields make continuation reuse explicit. The derivation counters
/// report actual source-to-target mapping operations made while materializing
/// this observation; validation and join belong to the typed encodings
/// themselves.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SuccinctArchiveViewWork {
    /// Distinct payload members represented after the call.
    pub cover_members: usize,
    /// Payload members newly processed by this call.
    pub processed_members: usize,
    /// Previously materialized payload members reused without replaying data work.
    pub reused_members: usize,
    /// Canonical source-to-target derivations.
    pub derive: u64,
    /// Cumulative bytes supplied to source-to-target derivations.
    pub input_bytes: u64,
}

impl SuccinctArchiveViewWork {
    fn with_support(cover_members: usize, processed_members: usize, reused_members: usize) -> Self {
        Self {
            cover_members,
            processed_members,
            reused_members,
            ..Self::default()
        }
    }
}

struct MeasuredSuccinctHomomorphism {
    inner: super::SimpleToSuccinctMapping,
    work: Cell<SuccinctArchiveViewWork>,
}

impl MeasuredSuccinctHomomorphism {
    fn new(inner: super::SimpleToSuccinctMapping, work: SuccinctArchiveViewWork) -> Self {
        Self {
            inner,
            work: Cell::new(work),
        }
    }

    fn bump(&self, update: impl FnOnce(&mut SuccinctArchiveViewWork)) {
        let mut work = self.work.get();
        update(&mut work);
        self.work.set(work);
    }
}

impl CollectionMapping<SimpleArchive, super::SuccinctArchiveBlob> for MeasuredSuccinctHomomorphism {
    fn bind(source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        Ok(Self::new(
            super::SimpleToSuccinctMapping::bind(source, target)?,
            SuccinctArchiveViewWork::default(),
        ))
    }

    fn map(
        &self,
        source: &crate::blob::Blob<SimpleArchive>,
    ) -> Result<crate::blob::Blob<SuccinctArchiveBlob>, CollectionOperationError> {
        self.bump(|work| {
            work.derive += 1;
            work.input_bytes += source.bytes.len() as u64;
        });
        self.inner.map(source)
    }
}

/// One in-process Succinct view maintained across exact cover observations.
///
/// Every retained shard was materialized by an earlier ordinary
/// [`SuccinctArchiveCollection::ensure_exact`] call. When the next cover is a
/// monotone extension, only its new payload support is processed and the two
/// immutable archives are unioned. An unchanged cover performs no storage I/O;
/// a shrinking cover rebuilds from the new exact observation.
///
/// This is continuation state, not durable authority or a cache receipt. It
/// deliberately retains the physical shards already returned to the caller,
/// exactly as any long-lived query source may do.
#[derive(Clone)]
pub struct SuccinctArchiveView {
    collection: SuccinctArchiveCollection,
    cover: Option<FactCover>,
    archive: Option<UnionArchive<OrderedUniverse>>,
    last_work: Option<SuccinctArchiveViewWork>,
}

impl SuccinctArchiveView {
    fn new(collection: SuccinctArchiveCollection) -> Self {
        Self {
            collection,
            cover: None,
            archive: None,
            last_work: None,
        }
    }

    /// Exact payload support represented by the current archive.
    pub fn cover(&self) -> Option<&FactCover> {
        self.cover.as_ref()
    }

    /// Current queryable archive, if the first observation has succeeded.
    pub fn archive(&self) -> Option<&UnionArchive<OrderedUniverse>> {
        self.archive.as_ref()
    }

    /// Work performed by the last successful observation.
    ///
    /// A failed call leaves both the retained view and this report unchanged.
    pub fn last_work(&self) -> Option<SuccinctArchiveViewWork> {
        self.last_work
    }

    /// Ensure and retain the exact view for the current cover.
    ///
    /// State advances only after every derivation, Rank9 attachment, and
    /// logical union succeeds. Retrying after an error therefore observes the
    /// same previous checkpoint.
    pub fn ensure<S>(
        &mut self,
        store: &mut S,
        current: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + MappingEvidenceStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        if self.cover.as_ref() == Some(current) {
            if let Some(previous) = &self.archive {
                let previous = previous.clone();
                self.last_work = Some(SuccinctArchiveViewWork::with_support(
                    current.len(),
                    0,
                    current.len(),
                ));
                return Ok(previous);
            }
        }
        let (next, work) = match self.archive.as_ref() {
            None => {
                let work = SuccinctArchiveViewWork::with_support(current.len(), current.len(), 0);
                self.ensure_measured(store, current, work)?
            }
            Some(previous) => match current.additions_since(
                self.cover
                    .as_ref()
                    .expect("an existing archive has a cover checkpoint"),
            ) {
                Ok(additions) if additions.is_empty() => (
                    previous.clone(),
                    SuccinctArchiveViewWork::with_support(current.len(), 0, current.len()),
                ),
                Ok(additions) => {
                    let work = SuccinctArchiveViewWork::with_support(
                        current.len(),
                        additions.len(),
                        self.cover.as_ref().map_or(0, FactCover::len),
                    );
                    let (delta, work) = self.ensure_measured(store, &additions, work)?;
                    (previous.union(&delta), work)
                }
                Err(CoverAdvanceError::ResetRequired { .. }) => {
                    let work =
                        SuccinctArchiveViewWork::with_support(current.len(), current.len(), 0);
                    self.ensure_measured(store, current, work)?
                }
                Err(error) => {
                    return Err(SuccinctArchiveCollectionError::Exact(
                        ExactDerivedCollectionError::InvalidCover(error.to_string()),
                    ));
                }
            },
        };

        self.cover = Some(current.clone());
        self.archive = Some(next.clone());
        self.last_work = Some(work);
        Ok(next)
    }

    fn ensure_measured<S>(
        &self,
        store: &mut S,
        cover: &FactCover,
        work: SuccinctArchiveViewWork,
    ) -> Result<
        (UnionArchive<OrderedUniverse>, SuccinctArchiveViewWork),
        SuccinctArchiveCollectionError,
    >
    where
        S: BlobStore + CollectionStore + MappingEvidenceStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let source = self.collection.source_descriptor();
        let target = self.collection.descriptor();
        let inner = super::SimpleToSuccinctMapping::bind(&source, &target)
            .map_err(|error| ExactDerivedCollectionError::Resolution(error.to_string()))?;
        let measured = MeasuredSuccinctHomomorphism::new(inner, work);
        let kernel = ExactDerivedCollection::with_mapping(source, target, measured)?;
        let target_cover = kernel.ensure_exact(store, cover)?;
        let work = kernel.mapping().work.get();
        let archive = if target_cover.is_empty() {
            self.collection.empty_archive()?
        } else {
            self.collection.rank9_fiber().ensure(store, target_cover)?
        };
        Ok((archive, work))
    }
}

impl SuccinctArchiveCollection {
    /// Create an empty in-process continuation for this exact projection.
    pub fn exact_view(&self) -> SuccinctArchiveView {
        SuccinctArchiveView::new(self.clone())
    }

    /// Construct the canonical Succinct projection for one named root.
    ///
    /// Two reaches, because a derivation and its source are two collections
    /// and neither inherits the other's answer. `source_reach` completes the
    /// root's identity so this facade hashes the same descriptor the root
    /// does; `reach` is this projection's own. A public index over a private
    /// source and a private index over a public one are both ordinary things
    /// to want, and an index can expose what its source did not, so the two
    /// are stated separately rather than derived from one another.
    /// `source_authority` and `authority` are independent mandatory descriptor
    /// facts: the former must exactly match the root cover, while the latter
    /// governs the raw Succinct collection. Rank9 cache evidence is unsigned
    /// and carries no authority of its own.
    pub fn new(
        name: impl Into<String>,
        source_authority: VerifyingKey,
        source_reach: Fragment,
        authority: VerifyingKey,
        reach: Fragment,
    ) -> Self {
        Self {
            name: name.into(),
            source_authority,
            source_reach,
            authority,
            reach,
        }
    }

    /// How far the source collection may travel.
    pub fn source_reach(&self) -> &Fragment {
        &self.source_reach
    }

    /// How far this projection may travel.
    pub fn reach(&self) -> &Fragment {
        &self.reach
    }

    /// Name of the root collection this projection is taken over.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Mandatory capability trust root declared by the source descriptor.
    pub fn source_authority(&self) -> VerifyingKey {
        self.source_authority
    }

    /// Mandatory capability trust root declared by this derived family.
    pub fn authority(&self) -> VerifyingKey {
        self.authority
    }

    /// Canonical source SimpleArchive-union descriptor facts.
    pub fn source_descriptor(&self) -> Fragment {
        simplearchive_union::descriptor(
            &self.name,
            self.source_authority,
            self.source_reach.clone(),
        )
    }

    /// Identity of the source collection this projection reads.
    pub fn source_collection(&self) -> CollectionHandle {
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.source_descriptor().into_facts())
            .get_handle()
    }

    /// Canonical target raw-SuccinctArchive-union descriptor.
    pub fn descriptor(&self) -> Fragment {
        super::descriptor(self.source_collection(), self.authority, self.reach.clone())
    }

    /// Identity of the raw Succinct cover this projection maintains.
    pub fn collection(&self) -> CollectionHandle {
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.descriptor().into_facts()).get_handle()
    }

    /// Attach the exact resident Succinct cover for `source_cover` without writing.
    ///
    /// An empty cover returns one authority-free process-local empty shard;
    /// it is not a persisted target member or a provenance assertion.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + MappingEvidenceStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel()?.attach_exact(store, source_cover)?;
        if cover.is_empty() {
            return self.empty_archive();
        }
        Ok(self.rank9_fiber().attach(store, cover)?)
    }

    /// Ensure missing raw derivations and attach the exact sharded cover.
    ///
    /// Completion writes raw outputs first, then ensures one persisted Rank9
    /// sidecar and native mapping-evidence equation for each member of that
    /// fixed selected raw cover. Rank9 remains an accelerator rather than a
    /// collection. Freshly built, validated runtimes are retained across the
    /// successful publication instead of being discarded and re-read.
    /// An empty cover has the same local-only behavior as [`Self::attach_exact`].
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + MappingEvidenceStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel()?.ensure_exact(store, source_cover)?;
        if cover.is_empty() {
            return self.empty_archive();
        }
        Ok(self.rank9_fiber().ensure(store, cover)?)
    }

    /// Explicitly compact and attach the exact raw target cover for `source_cover`.
    ///
    /// This first performs ordinary exact completion, then applies the fixed
    /// dyadic byte-size policy to canonical target members. All compacted blobs
    /// precede unsigned `MERGE` records, no flush or signed record is implied,
    /// and the returned logical union retains the attached Rank9 runtimes built
    /// or reused for that exact compacted cover.
    pub fn compact_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + MappingEvidenceStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let kernel = self.kernel()?;
        let cover = compact_exact_target(&kernel, store, source_cover)?;
        if cover.is_empty() {
            return self.empty_archive();
        }
        Ok(self.rank9_fiber().ensure(store, cover)?)
    }

    fn kernel(
        &self,
    ) -> Result<
        ExactDerivedCollection<
            SimpleArchive,
            super::SuccinctArchiveBlob,
            super::SimpleToSuccinctMapping,
        >,
        ExactDerivedCollectionError,
    > {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }

    fn rank9_fiber(&self) -> Rank9Fiber {
        Rank9Fiber::new(self.descriptor())
    }

    fn empty_archive(
        &self,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError> {
        let raw = super::empty();
        let raw_data = crate::inline::encodings::hash::Handle::<SuccinctArchiveBlob>::to_hash(
            raw.get_handle(),
        );
        let bottom: SuccinctArchive<OrderedUniverse> =
            raw.try_from_blob()
                .map_err(|source| super::Rank9FiberError::Build {
                    raw: raw_data,
                    source,
                })?;
        Ok(UnionArchive::new(vec![bottom]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::any::TypeId;
    use std::collections::BTreeSet;
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::UnknownBlob;
    use crate::blob::{Blob, BlobEncoding, Bytes, IntoBlob, TryFromBlob};
    use crate::collection::descriptor::{self, identity_for_tests};
    use crate::collection::{
        Collection, CollectionCommit, CollectionData, CollectionDerive, CollectionMerge,
        CollectionRecord, MappingEvidence, MappingEvidenceSelector,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::inline::{Inline, InlineEncoding};
    use crate::metadata::MetaDescribe;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::pile::{Pile, WantRewritePolicy};
    use crate::repo::{BlobStoreGet, BlobStoreList, BlobStorePut, RetentionRoots};
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    /// The one team every collection in these tests belongs to.
    fn test_team() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    fn test_collection(name: &str) -> SuccinctArchiveCollection {
        SuccinctArchiveCollection::new(
            name.to_owned(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        )
    }

    #[test]
    fn source_and_derived_descriptors_carry_independent_mandatory_authorities() {
        let source_authority = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let target_authority = SigningKey::from_bytes(&[3; 32]).verifying_key();
        let name = "source".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            source_authority,
            reach::private(),
            target_authority,
            reach::private(),
        );

        assert_eq!(
            collection.source_descriptor(),
            simplearchive_union::descriptor(&name, source_authority, reach::private())
        );
        assert_eq!(
            descriptor::authority(collection.source_descriptor().facts()),
            Ok(source_authority)
        );
        assert_eq!(
            descriptor::authority(collection.descriptor().facts()),
            Ok(target_authority)
        );
    }

    #[test]
    fn exact_succinct_member_validation_is_typed_and_contextual() {
        let collection = test_collection("first");
        let malformed = Blob::<SuccinctArchiveBlob>::new(Bytes::from(vec![0u8; 1]));
        assert!(matches!(
            super::super::SuccinctArchiveBlob::validate_member(
                &collection.descriptor(),
                &malformed,
            ),
            Err(CollectionOperationError::Fatal(_))
        ));
    }

    /// Compile-time proof that the native API has no legacy pin requirement.
    #[derive(Default)]
    struct CollectionOnly {
        repo: MemoryRepo,
        puts: usize,
        inserts: usize,
    }

    impl CollectionOnly {
        fn reset_writes(&mut self) {
            self.puts = 0;
            self.inserts = 0;
        }

        fn writes(&self) -> (usize, usize) {
            (self.puts, self.inserts)
        }
    }

    impl crate::repo::ArtifactOfferStore for CollectionOnly {
        type OfferError = <MemoryRepo as crate::repo::ArtifactOfferStore>::OfferError;

        fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
        where
            I: IntoIterator<Item = crate::repo::ArtifactHandle>,
        {
            self.repo.offer_all(handles)
        }

        fn offers_snapshot(
            &mut self,
        ) -> Result<crate::repo::ArtifactOfferSnapshot, Self::OfferError> {
            self.repo.offers_snapshot()
        }
    }

    impl BlobStorePut for CollectionOnly {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<E, T>(&mut self, item: T) -> Result<crate::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: crate::blob::IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.puts += 1;
            self.repo.put(item)
        }
    }

    impl BlobStore for CollectionOnly {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.repo.reader()
        }
    }

    impl CollectionStore for CollectionOnly {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = <MemoryRepo as CollectionStore>::InsertError;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.repo.records()
        }

        fn select_records(
            &mut self,
            selectors: &BTreeSet<crate::collection::CollectionRecordSelector>,
        ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
            self.repo.select_records(selectors)
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.inserts += 1;
            self.repo.insert(record)
        }
    }

    impl MappingEvidenceStore for CollectionOnly {
        type EvidenceError = <MemoryRepo as MappingEvidenceStore>::EvidenceError;
        type InsertError = <MemoryRepo as MappingEvidenceStore>::InsertError;
        type EvidenceIter<'a>
            = <MemoryRepo as MappingEvidenceStore>::EvidenceIter<'a>
        where
            Self: 'a;

        fn evidence<'a>(&'a mut self) -> Result<Self::EvidenceIter<'a>, Self::EvidenceError> {
            self.repo.evidence()
        }

        fn select_evidence(
            &mut self,
            selectors: &BTreeSet<MappingEvidenceSelector>,
        ) -> Result<Vec<MappingEvidence>, Self::EvidenceError> {
            self.repo.select_evidence(selectors)
        }

        fn insert_evidence(&mut self, evidence: MappingEvidence) -> Result<(), Self::InsertError> {
            self.inserts += 1;
            self.repo.insert_evidence(evidence)
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct InjectedFailure(&'static str);

    impl fmt::Display for InjectedFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for InjectedFailure {}

    struct FaultStore {
        repo: MemoryRepo,
        rank9_mapping: crate::collection::MappingHandle,
        fail_rank9_put: bool,
        drop_rank9_put: bool,
        replace_rank9_on_put: Option<CollectionData>,
        drop_rank9_evidence: bool,
        fail_rank9_evidence_at: Option<usize>,
        rank9_evidence_attempts: usize,
        puts: usize,
        inserts: usize,
    }

    impl FaultStore {
        fn new(repo: MemoryRepo, rank9_mapping: crate::collection::MappingHandle) -> Self {
            Self {
                repo,
                rank9_mapping,
                fail_rank9_put: false,
                drop_rank9_put: false,
                replace_rank9_on_put: None,
                drop_rank9_evidence: false,
                fail_rank9_evidence_at: None,
                rank9_evidence_attempts: 0,
                puts: 0,
                inserts: 0,
            }
        }

        fn is_rank9_evidence(&self, evidence: MappingEvidence) -> bool {
            evidence.mapping() == self.rank9_mapping
        }
    }

    impl BlobStorePut for FaultStore {
        type PutError = InjectedFailure;

        fn put<E, T>(&mut self, item: T) -> Result<Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.puts += 1;
            let blob = item.to_blob();
            if TypeId::of::<E>() == TypeId::of::<SuccinctArchiveRank9IndexBlob>() {
                if self.fail_rank9_put {
                    return Err(InjectedFailure("injected Rank9 put failure"));
                }
                if self.drop_rank9_put {
                    return Ok(blob.get_handle());
                }
                let output = Handle::<E>::to_hash(blob.get_handle());
                if self.replace_rank9_on_put == Some(output) {
                    let reader = self
                        .repo
                        .blobs
                        .reader()
                        .expect("memory reader is infallible");
                    let retained: Vec<Inline<Handle<UnknownBlob>>> = reader
                        .into_iter()
                        .map(|(resident, _)| resident)
                        .filter(|resident| resident.raw != output.raw)
                        .collect();
                    self.repo.blobs.keep(retained);
                }
            }
            Ok(self.repo.put(blob).expect("memory put is infallible"))
        }
    }

    impl BlobStore for FaultStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.repo.reader()
        }
    }

    impl CollectionStore for FaultStore {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = InjectedFailure;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.repo.records()
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.inserts += 1;
            self.repo
                .insert(record)
                .expect("memory record insert is infallible");
            Ok(())
        }
    }

    impl MappingEvidenceStore for FaultStore {
        type EvidenceError = <MemoryRepo as MappingEvidenceStore>::EvidenceError;
        type InsertError = InjectedFailure;
        type EvidenceIter<'a>
            = <MemoryRepo as MappingEvidenceStore>::EvidenceIter<'a>
        where
            Self: 'a;

        fn evidence<'a>(&'a mut self) -> Result<Self::EvidenceIter<'a>, Self::EvidenceError> {
            self.repo.evidence()
        }

        fn select_evidence(
            &mut self,
            selectors: &BTreeSet<MappingEvidenceSelector>,
        ) -> Result<Vec<MappingEvidence>, Self::EvidenceError> {
            self.repo.select_evidence(selectors)
        }

        fn insert_evidence(&mut self, evidence: MappingEvidence) -> Result<(), Self::InsertError> {
            self.inserts += 1;
            if self.is_rank9_evidence(evidence) {
                self.rank9_evidence_attempts += 1;
                if self.fail_rank9_evidence_at == Some(self.rank9_evidence_attempts) {
                    return Err(InjectedFailure("injected Rank9 evidence failure"));
                }
                if self.drop_rank9_evidence {
                    return Ok(());
                }
            }
            self.repo
                .insert_evidence(evidence)
                .expect("memory evidence insert is infallible");
            Ok(())
        }
    }

    impl crate::repo::ArtifactOfferStore for FaultStore {
        type OfferError = <MemoryRepo as crate::repo::ArtifactOfferStore>::OfferError;

        fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
        where
            I: IntoIterator<Item = crate::repo::ArtifactHandle>,
        {
            self.repo.offer_all(handles)
        }

        fn offers_snapshot(
            &mut self,
        ) -> Result<crate::repo::ArtifactOfferSnapshot, Self::OfferError> {
            self.repo.offers_snapshot()
        }
    }

    struct PanicStore;

    impl BlobStorePut for PanicStore {
        type PutError = Infallible;

        fn put<E, T>(&mut self, _: T) -> Result<crate::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: crate::blob::IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            panic!("empty Succinct cover attempted a blob write")
        }
    }

    impl BlobStore for PanicStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = Infallible;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            panic!("empty Succinct cover opened a reader")
        }
    }

    impl CollectionStore for PanicStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            panic!("empty Succinct cover scanned records")
        }

        fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
            panic!("empty Succinct cover inserted a record")
        }
    }

    impl MappingEvidenceStore for PanicStore {
        type EvidenceError = Infallible;
        type InsertError = Infallible;
        type EvidenceIter<'a> = std::vec::IntoIter<Result<MappingEvidence, Infallible>>;

        fn evidence<'a>(&'a mut self) -> Result<Self::EvidenceIter<'a>, Self::EvidenceError> {
            panic!("empty Succinct cover scanned mapping evidence")
        }

        fn insert_evidence(&mut self, _: MappingEvidence) -> Result<(), Self::InsertError> {
            panic!("empty Succinct cover inserted mapping evidence")
        }
    }

    impl crate::repo::ArtifactOfferStore for PanicStore {
        type OfferError = Infallible;

        fn offer_all<I>(&mut self, _: I) -> Result<(), Self::OfferError>
        where
            I: IntoIterator<Item = crate::repo::ArtifactHandle>,
        {
            panic!("empty Succinct cover attempted an OFFER write")
        }

        fn offers_snapshot(
            &mut self,
        ) -> Result<crate::repo::ArtifactOfferSnapshot, Self::OfferError> {
            Ok(crate::repo::ArtifactOfferSnapshot::default())
        }
    }

    fn row(entity: u8, value: u8) -> Trible {
        let mut raw = [value; TRIBLE_LEN];
        raw[..16].fill(entity);
        raw[16..32].fill(9);
        Trible::force_raw(raw).unwrap()
    }

    fn facts(rows: impl IntoIterator<Item = (u8, u8)>) -> TribleSet {
        let mut facts = TribleSet::new();
        for (entity, value) in rows {
            facts.insert(&row(entity, value));
        }
        facts
    }

    fn put_data(store: &mut CollectionOnly, facts: &TribleSet) -> Blob<SimpleArchive> {
        let blob = facts.to_blob();
        store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        blob
    }

    fn signed_commit(
        store: &mut CollectionOnly,
        name: &str,
        key: u8,
        data: &Blob<SimpleArchive>,
    ) -> CollectionCommit {
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            identity_for_tests(&simplearchive_union::descriptor(
                name,
                test_team(),
                reach::private(),
            )),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata,
        )
    }

    fn publish(store: &mut CollectionOnly, commit: CollectionCommit) {
        store.insert(CollectionRecord::Commit(commit)).unwrap();
    }

    fn records(store: &mut CollectionOnly) -> Vec<CollectionRecord> {
        store.records().unwrap().map(Result::unwrap).collect()
    }

    fn data<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
    where
        Handle<E>: InlineEncoding,
    {
        Handle::<E>::to_hash(blob.get_handle())
    }

    fn source_cover(
        collection: &SuccinctArchiveCollection,
        commits: &[CollectionCommit],
    ) -> FactCover {
        FactCover::from_members(
            Collection::<SimpleArchive>::from_handle(collection.source_collection()),
            commits
                .iter()
                .map(|commit| Handle::<SimpleArchive>::from_hash(commit.data())),
        )
    }

    fn empty_source_cover(collection: &SuccinctArchiveCollection) -> FactCover {
        FactCover::from_members(
            Collection::<SimpleArchive>::from_handle(collection.source_collection()),
            [],
        )
    }

    fn raw_derives<S: CollectionStore>(
        store: &mut S,
        collection: &SuccinctArchiveCollection,
    ) -> Vec<CollectionDerive> {
        let mut claims: Vec<_> = store
            .records()
            .unwrap()
            .map(Result::unwrap)
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == collection.collection() =>
                {
                    Some(claim)
                }
                _ => None,
            })
            .collect();
        claims.sort_by_key(|claim| (claim.input(), claim.output()));
        claims
    }

    fn rank9_mapping_handle() -> crate::collection::MappingHandle {
        super::super::rank9_mapping_fragment()
            .facts()
            .clone()
            .to_blob()
            .get_handle()
    }

    fn rank9_evidence<S>(store: &mut S) -> Vec<MappingEvidence>
    where
        S: MappingEvidenceStore,
        S::EvidenceError: fmt::Debug,
    {
        let selectors = BTreeSet::from([MappingEvidenceSelector::Mapping(rank9_mapping_handle())]);
        let mut evidence = store.select_evidence(&selectors).unwrap();
        evidence.sort_by_key(|evidence| (evidence.input(), evidence.output()));
        evidence
    }

    fn remove_blob<E>(store: &mut CollectionOnly, handle: Inline<Handle<E>>)
    where
        E: BlobEncoding + 'static,
        Handle<E>: InlineEncoding,
    {
        let reader = store.repo.blobs.reader().unwrap();
        let retained: Vec<Inline<Handle<UnknownBlob>>> = reader
            .into_iter()
            .map(|(resident, _)| resident)
            .filter(|resident| resident.raw != handle.raw)
            .collect();
        store.repo.blobs.keep(retained);
    }

    fn one_raw_fixture(
        scope_byte: u8,
    ) -> (
        SuccinctArchiveCollection,
        CollectionOnly,
        CollectionCommit,
        TribleSet,
        Blob<SuccinctArchiveBlob>,
    ) {
        let name = format!("c{scope_byte}");
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let commit = signed_commit(&mut store, &name, 1, &source);
        publish(&mut store, commit);
        let cover = collection
            .kernel()
            .unwrap()
            .ensure_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
        assert_eq!(cover.len(), 1);
        let raw = super::super::derive_element(&source).unwrap();
        assert_eq!(cover.members()[0].0, raw.get_handle());
        drop(cover);
        store.reset_writes();
        (collection, store, commit, expected, raw)
    }

    fn attached_facts(archive: &UnionArchive<OrderedUniverse>) -> TribleSet {
        archive.iter().collect()
    }

    fn pile_commit(
        store: &mut Pile,
        collection: &SuccinctArchiveCollection,
        key: u8,
        source: &Blob<SimpleArchive>,
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
    ) -> CollectionCommit {
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            collection.source_collection(),
            data(source),
            metadata,
        );
        store.insert(CollectionRecord::Commit(commit)).unwrap();
        commit
    }

    #[test]
    fn rank9_mapping_is_abi_profile_separated() {
        let algorithms = [
            super::super::RANK9_SIDECAR_MAPPING_V1_32_LE,
            super::super::RANK9_SIDECAR_MAPPING_V1_32_BE,
            super::super::RANK9_SIDECAR_MAPPING_V1_64_LE,
            super::super::RANK9_SIDECAR_MAPPING_V1_64_BE,
        ];
        let mapping_for = |algorithm| {
            let description = if algorithm == super::super::RANK9_SIDECAR_MAPPING_V1_32_LE {
                super::super::Rank9SidecarMappingV1_32Le::describe()
            } else if algorithm == super::super::RANK9_SIDECAR_MAPPING_V1_32_BE {
                super::super::Rank9SidecarMappingV1_32Be::describe()
            } else if algorithm == super::super::RANK9_SIDECAR_MAPPING_V1_64_LE {
                super::super::Rank9SidecarMappingV1_64Le::describe()
            } else {
                super::super::Rank9SidecarMappingV1_64Be::describe()
            };
            crate::prelude::entity! {
                crate::metadata::tag: crate::collection::records::KIND_COLLECTION_MAPPING,
                crate::collection::records::mapping_algorithm*: description,
            }
        };
        let current_algorithm = super::super::current_rank9_mapping_algorithm();
        let current = super::super::rank9_mapping_fragment();
        assert_eq!(current.facts(), mapping_for(current_algorithm).facts());
        assert_eq!(algorithms.into_iter().collect::<BTreeSet<_>>().len(), 4);
        assert_eq!(
            algorithms
                .into_iter()
                .map(|algorithm| IntoBlob::<SimpleArchive>::to_blob(
                    mapping_for(algorithm).into_facts(),
                )
                .get_handle())
                .collect::<BTreeSet<_>>()
                .len(),
            4,
        );
    }

    #[test]
    fn ensured_rank9_evidence_is_exact_complete_and_zero_write_when_resident() {
        let name = "rank9-evidence".to_owned();
        let collection = test_collection(&name);
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);

        let ensured = collection
            .ensure_exact(&mut store, &source_cover(&collection, &[second, first]))
            .unwrap();
        assert_eq!(
            ensured.segment_count(),
            2,
            "the raw cover shape is preserved"
        );
        assert_eq!(attached_facts(&ensured), left_facts + right_facts);
        let evidence = rank9_evidence(&mut store);
        assert_eq!(evidence.len(), 2);
        assert_eq!(
            raw_derives(&mut store, &collection)
                .iter()
                .map(CollectionDerive::output)
                .collect::<BTreeSet<_>>(),
            evidence
                .iter()
                .map(|evidence| evidence.input())
                .collect::<BTreeSet<_>>(),
        );

        let reader = store.reader().unwrap();
        for evidence in &evidence {
            assert_eq!(evidence.mapping(), rank9_mapping_handle());
            let raw: Blob<SuccinctArchiveBlob> = reader
                .get(Handle::<SuccinctArchiveBlob>::from_hash(evidence.input()))
                .unwrap();
            let rank9: Blob<SuccinctArchiveRank9IndexBlob> = reader
                .get(Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(
                    evidence.output(),
                ))
                .unwrap();
            assert_eq!(
                SuccinctArchiveRank9IndexBlob::source_handle(&rank9).unwrap(),
                raw.get_handle(),
            );
            SuccinctArchive::<OrderedUniverse>::from_blob_pair(raw, rank9).unwrap();
        }
        let mapping = super::super::rank9_mapping_fragment();
        let mapping_blob: Blob<SimpleArchive> = reader.get(rank9_mapping_handle()).unwrap();
        assert_eq!(
            TribleSet::try_from_blob(mapping_blob).unwrap(),
            mapping.facts().clone(),
        );
        let mut attachments = mapping.blobs().clone();
        for (handle, expected) in attachments.reader().unwrap() {
            let actual: Blob<UnknownBlob> = reader.get(handle).unwrap();
            assert_eq!(actual.bytes, expected.bytes);
        }
        drop(reader);

        store.reset_writes();
        let attached = collection
            .attach_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
        assert_eq!(attached.segment_count(), 2);
        assert_eq!(store.writes(), (0, 0));
        collection
            .ensure_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
        assert_eq!(store.writes(), (0, 0));
        assert_eq!(rank9_evidence(&mut store), evidence);
    }

    #[test]
    fn bad_or_ambiguous_evidence_is_a_cache_miss_and_ensure_repairs() {
        let (collection, mut store, commit, expected, raw) = one_raw_fixture(13);
        crate::collection::descriptor::put_closure(
            &mut store,
            &super::super::rank9_mapping_fragment(),
        )
        .unwrap();
        let canonical = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone()).unwrap();
        let bogus = Blob::<SuccinctArchiveRank9IndexBlob>::new(Bytes::from(b"bogus".to_vec()));
        for sidecar in [&canonical, &bogus] {
            store
                .put::<SuccinctArchiveRank9IndexBlob, _>(sidecar.clone())
                .unwrap();
            store
                .insert_evidence(MappingEvidence::new(
                    rank9_mapping_handle(),
                    data(&raw),
                    data(sidecar),
                ))
                .unwrap();
        }
        store.reset_writes();

        let attached = collection
            .attach_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
        assert_eq!(attached_facts(&attached), expected);
        assert_eq!(store.writes(), (0, 0), "attach never repairs cache state");

        let ensured = collection
            .ensure_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
        assert_eq!(attached_facts(&ensured), expected);
        assert!(store.writes().0 > 0 && store.writes().1 > 0);
        assert_eq!(rank9_evidence(&mut store).len(), 2);

        let sidecar = canonical.get_handle();
        remove_blob(&mut store, sidecar);
        store.reset_writes();
        collection
            .ensure_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
        assert!(store.reader().unwrap().contains_blob(sidecar).unwrap());
    }

    #[test]
    fn failed_sidecar_put_never_publishes_mapping_evidence() {
        let (collection, base, commit, _, _) = one_raw_fixture(15);
        let mut store = FaultStore::new(base.repo, rank9_mapping_handle());
        store.fail_rank9_put = true;
        assert!(matches!(
            collection.ensure_exact(&mut store, &source_cover(&collection, &[commit])),
            Err(SuccinctArchiveCollectionError::Rank9(
                super::super::Rank9FiberError::Storage { .. }
            ))
        ));
        assert!(rank9_evidence(&mut store).is_empty());
    }

    #[test]
    fn compaction_builds_evidence_only_for_the_selected_raw_cover() {
        let name = "rank9-compaction".to_owned();
        let collection = test_collection(&name);
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &facts([(1, 3)]));
        let right = put_data(&mut store, &facts([(2, 4)]));
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let raw_cover = collection
            .kernel()
            .unwrap()
            .ensure_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
        assert_eq!(raw_cover.len(), 2);
        assert!(rank9_evidence(&mut store).is_empty());

        let compacted = collection
            .compact_exact(&mut store, &source_cover(&collection, &[second, first]))
            .unwrap();
        assert_eq!(compacted.segment_count(), 1);
        let evidence = rank9_evidence(&mut store);
        assert_eq!(evidence.len(), 1);
        let merged_raw = records(&mut store)
            .into_iter()
            .find_map(|record| match record {
                CollectionRecord::Merge(claim) if claim.collection() == collection.collection() => {
                    Some(claim.result())
                }
                _ => None,
            })
            .expect("compaction publishes one selected raw merge");
        assert_eq!(evidence[0].input(), merged_raw);
    }

    #[test]
    fn empty_cover_is_one_authority_free_local_shard_and_performs_no_io() {
        let collection = test_collection("c7");
        let mut store = PanicStore;
        let empty = empty_source_cover(&collection);
        for attached in [
            collection.attach_exact(&mut store, &empty).unwrap(),
            collection.ensure_exact(&mut store, &empty).unwrap(),
            collection.compact_exact(&mut store, &empty).unwrap(),
        ] {
            assert_eq!(attached.segment_count(), 1);
            assert_eq!(attached.iter().count(), 0);
        }
    }

    #[test]
    fn exact_view_reuses_unchanged_support_without_storage_io() {
        let name = "maintained".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let commit = signed_commit(&mut store, &name, 1, &source);
        publish(&mut store, commit);

        let mut maintained = collection.exact_view();
        let cover = source_cover(&collection, &[commit]);
        let first = maintained.ensure(&mut store, &cover).unwrap();
        assert_eq!(attached_facts(&first), expected);
        assert_eq!(maintained.cover(), Some(&cover));
        let first_work = maintained.last_work().expect("first observation work");
        assert_eq!(first_work.cover_members, 1);
        assert_eq!(first_work.processed_members, 1);
        assert_eq!(first_work.reused_members, 0);
        assert!(first_work.derive > 0);

        let repeated = maintained.ensure(&mut PanicStore, &cover).unwrap();
        assert_eq!(attached_facts(&repeated), expected);
        assert_eq!(maintained.cover(), Some(&cover));
        assert_eq!(
            maintained.last_work(),
            Some(SuccinctArchiveViewWork::with_support(1, 0, 1)),
            "an identical cover performs no raw proof or derivation work",
        );
    }

    #[test]
    fn exact_view_preserves_set_semantics_for_duplicate_support() {
        let name = "maintained-overlap".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let first = signed_commit(&mut store, &name, 1, &source);
        let second = signed_commit(&mut store, &name, 2, &source);
        publish(&mut store, first);
        publish(&mut store, second);

        let mut maintained = collection.exact_view();
        let first_cover = source_cover(&collection, &[first]);
        let duplicate_provenance_cover = source_cover(&collection, &[first, second]);
        assert_eq!(duplicate_provenance_cover, first_cover);
        maintained.ensure(&mut store, &first_cover).unwrap();
        let grown = maintained
            .ensure(&mut PanicStore, &duplicate_provenance_cover)
            .unwrap();

        assert_eq!(attached_facts(&grown), expected);
        assert_eq!(maintained.cover(), Some(&first_cover));
        assert_eq!(maintained.cover().unwrap().len(), 1);
        assert_eq!(
            maintained.last_work(),
            Some(SuccinctArchiveViewWork::with_support(1, 0, 1)),
            "another signature over the same payload is provenance-only",
        );
    }

    #[test]
    fn exact_view_unions_additions_and_rebuilds_after_shrink() {
        let name = "maintained-growth".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);

        let mut maintained = collection.exact_view();
        let first_cover = source_cover(&collection, &[first]);
        let full_cover = source_cover(&collection, &[second, first]);
        let second_cover = source_cover(&collection, &[second]);
        maintained.ensure(&mut store, &first_cover).unwrap();
        let first_work = maintained.last_work().expect("first observation work");
        remove_blob(&mut store, left.get_handle());
        drop(left);
        let grown = maintained.ensure(&mut store, &full_cover).unwrap();
        assert_eq!(
            attached_facts(&grown),
            left_facts.clone() + right_facts.clone()
        );
        let grown_work = maintained.last_work().expect("extension work");
        assert_eq!(grown_work.cover_members, 2);
        assert_eq!(grown_work.processed_members, 1);
        assert_eq!(grown_work.reused_members, 1);
        assert_eq!(
            grown_work.derive, first_work.derive,
            "one-commit extension admits only its delta",
        );
        assert_eq!(maintained.cover(), Some(&full_cover));

        let shrunk = maintained.ensure(&mut store, &second_cover).unwrap();
        assert_eq!(attached_facts(&shrunk), right_facts);
        assert_eq!(maintained.cover(), Some(&second_cover));
    }

    #[test]
    fn exact_view_does_not_advance_on_invalid_cover_shape() {
        let name = "maintained-errors".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let commit = signed_commit(&mut store, &name, 1, &source);
        publish(&mut store, commit);

        let mut maintained = collection.exact_view();
        let cover = source_cover(&collection, &[commit]);
        maintained.ensure(&mut store, &cover).unwrap();
        let successful_work = maintained.last_work();
        let foreign_name = "foreign".to_owned();
        let foreign = signed_commit(&mut store, &foreign_name, 2, &source);
        let foreign_collection = test_collection(&foreign_name);
        let foreign_cover = source_cover(&foreign_collection, &[foreign]);

        assert!(matches!(
            maintained.ensure(&mut store, &foreign_cover),
            Err(SuccinctArchiveCollectionError::Exact(
                ExactDerivedCollectionError::InvalidCover(_)
            ))
        ));
        assert_eq!(maintained.cover(), Some(&cover));
        assert_eq!(maintained.last_work(), successful_work);
        assert_eq!(
            attached_facts(maintained.archive().expect("previous archive remains")),
            expected
        );
    }

    #[test]
    fn exact_view_does_not_advance_when_delta_attachment_fails() {
        let name = "maintained-attachment-failure".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut base = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut base, &left_facts);
        let right = put_data(&mut base, &right_facts);
        let first = signed_commit(&mut base, &name, 1, &left);
        let second = signed_commit(&mut base, &name, 2, &right);
        publish(&mut base, first);
        publish(&mut base, second);

        let mut store = FaultStore::new(base.repo, rank9_mapping_handle());
        let mut maintained = collection.exact_view();
        let first_cover = source_cover(&collection, &[first]);
        let full_cover = source_cover(&collection, &[first, second]);
        maintained.ensure(&mut store, &first_cover).unwrap();
        let successful_work = maintained.last_work();

        store.fail_rank9_put = true;
        assert!(matches!(
            maintained.ensure(&mut store, &full_cover),
            Err(SuccinctArchiveCollectionError::Rank9(
                super::super::Rank9FiberError::Storage { .. }
            ))
        ));
        assert_eq!(maintained.cover(), Some(&first_cover));
        assert_eq!(maintained.last_work(), successful_work);
        assert_eq!(
            attached_facts(maintained.archive().expect("previous archive remains")),
            left_facts
        );

        store.fail_rank9_put = false;
        let retried = maintained.ensure(&mut store, &full_cover).unwrap();
        assert_eq!(attached_facts(&retried), left_facts + right_facts);
    }

    #[test]
    fn signed_empty_source_still_publishes_nonempty_cover_provenance() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let source = put_data(&mut store, &TribleSet::new());
        let commit = signed_commit(&mut store, &name, 1, &source);
        publish(&mut store, commit);

        let attached = collection
            .ensure_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached.iter().count(), 0);
        let mappings: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == collection.collection() =>
                {
                    Some((claim.input(), claim.output()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].0, commit.data());
        collection
            .attach_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
    }

    #[test]
    fn missing_attach_then_ensure_builds_exact_raw_cover() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        assert!(matches!(
            collection.attach_exact(&mut store, &source_cover(&collection, &[first, second]),),
            Err(SuccinctArchiveCollectionError::Exact(
                ExactDerivedCollectionError::IncompleteCover { .. }
            ))
        ));
        let attached = collection
            .ensure_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
        assert_eq!(attached.segment_count(), 2);
        let mut derived_outputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == collection.collection() =>
                {
                    Some(claim.output().transmute())
                }
                _ => None,
            })
            .collect();
        derived_outputs.extend(
            rank9_evidence(&mut store)
                .into_iter()
                .map(|evidence| evidence.output().transmute()),
        );
        let offers = store.offers_snapshot().unwrap();
        assert!(!derived_outputs.is_empty());
        assert!(derived_outputs
            .into_iter()
            .all(|output| offers.contains(output)));
    }

    #[test]
    fn explicit_compaction_returns_one_exact_real_succinct_shard() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);

        let attached = collection
            .compact_exact(&mut store, &source_cover(&collection, &[second, first]))
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
        let merged = records(&mut store)
            .into_iter()
            .find_map(|record| match record {
                CollectionRecord::Merge(claim) if claim.collection() == collection.collection() => {
                    Some(claim.result().transmute())
                }
                _ => None,
            })
            .expect("compaction published a raw MERGE");
        assert!(store.offers_snapshot().unwrap().contains(merged));
    }

    #[test]
    fn duplicate_provenance_shares_one_raw_derive() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let first = signed_commit(&mut store, &name, 1, &source);
        let second = signed_commit(&mut store, &name, 2, &source);
        publish(&mut store, first);
        publish(&mut store, second);
        let attached = collection
            .ensure_exact(
                &mut store,
                &source_cover(&collection, &[first, first, second]),
            )
            .unwrap();
        assert_eq!(attached_facts(&attached), expected);
        let derives = records(&mut store)
            .into_iter()
            .filter(|record| {
                matches!(record, CollectionRecord::Derive(claim)
                if claim.collection() == collection.collection())
            })
            .count();
        assert_eq!(derives, 1);
        collection
            .attach_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
    }

    #[test]
    fn resident_source_merge_is_reused_as_one_shard() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let source_union = simplearchive_union::join(&left, &right).unwrap();
        store.put::<SimpleArchive, _>(source_union.clone()).unwrap();
        let source_union_data = data(&source_union);
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.source_collection(),
                first.data(),
                second.data(),
                source_union_data,
            )))
            .unwrap();
        let attached = collection
            .ensure_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
        let inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == collection.collection() =>
                {
                    Some(claim.input())
                }
                _ => None,
            })
            .collect();
        assert_eq!(inputs, vec![source_union_data]);
    }

    #[test]
    fn existing_target_merge_is_selected_as_one_shard() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let left_facts = facts([(1, 3)]);
        let right_facts = facts([(2, 4)]);
        let left = put_data(&mut store, &left_facts);
        let right = put_data(&mut store, &right_facts);
        let first = signed_commit(&mut store, &name, 1, &left);
        let second = signed_commit(&mut store, &name, 2, &right);
        publish(&mut store, first);
        publish(&mut store, second);
        let left_raw = super::super::derive_element(&left).unwrap();
        let right_raw = super::super::derive_element(&right).unwrap();
        for (input, output) in [(&left, &left_raw), (&right, &right_raw)] {
            store.put::<SuccinctArchiveBlob, _>(output.clone()).unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    collection.collection(),
                    data(input),
                    data(output),
                )))
                .unwrap();
        }
        let joined = super::super::join(&left_raw, &right_raw).unwrap();
        store.put::<SuccinctArchiveBlob, _>(joined.clone()).unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.collection(),
                data(&left_raw),
                data(&right_raw),
                data(&joined),
            )))
            .unwrap();
        let attached = collection
            .attach_exact(&mut store, &source_cover(&collection, &[first, second]))
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), left_facts + right_facts);
    }

    #[test]
    fn old_cover_stays_stable_after_later_commit_and_cache() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let old_facts = facts([(1, 3)]);
        let old = put_data(&mut store, &old_facts);
        let first = signed_commit(&mut store, &name, 1, &old);
        publish(&mut store, first);
        let old_cover = source_cover(&collection, &[first]);
        collection.ensure_exact(&mut store, &old_cover).unwrap();

        let later_facts = facts([(2, 4)]);
        let later = put_data(&mut store, &later_facts);
        let second = signed_commit(&mut store, &name, 2, &later);
        publish(&mut store, second);
        let later_raw = super::super::derive_element(&later).unwrap();
        store
            .put::<SuccinctArchiveBlob, _>(later_raw.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.collection(),
                second.data(),
                data(&later_raw),
            )))
            .unwrap();
        assert_eq!(
            attached_facts(&collection.attach_exact(&mut store, &old_cover).unwrap()),
            old_facts,
        );
    }

    #[test]
    fn missing_derive_output_is_not_support_and_ensure_rebuilds_it() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let source = put_data(&mut store, &expected);
        let commit = signed_commit(&mut store, &name, 1, &source);
        publish(&mut store, commit);
        let missing = super::super::derive_element(&source).unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                collection.collection(),
                commit.data(),
                data(&missing),
            )))
            .unwrap();
        assert!(matches!(
            collection.attach_exact(&mut store, &source_cover(&collection, &[commit])),
            Err(SuccinctArchiveCollectionError::Exact(
                ExactDerivedCollectionError::IncompleteCover { .. }
            ))
        ));
        assert_eq!(
            attached_facts(
                &collection
                    .ensure_exact(&mut store, &source_cover(&collection, &[commit]))
                    .unwrap()
            ),
            expected,
        );
    }

    #[test]
    fn ungrounded_source_superset_never_enters_smaller_cover() {
        let name = "c7".to_owned();
        let collection = SuccinctArchiveCollection::new(
            name.clone(),
            test_team(),
            reach::private(),
            test_team(),
            reach::private(),
        );
        let mut store = CollectionOnly::default();
        let expected = facts([(1, 3)]);
        let a = put_data(&mut store, &expected);
        let c = put_data(&mut store, &facts([(3, 5)]));
        let commit = signed_commit(&mut store, &name, 1, &a);
        publish(&mut store, commit);
        let superset = simplearchive_union::join(&a, &c).unwrap();
        store.put::<SimpleArchive, _>(superset.clone()).unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                collection.source_collection(),
                data(&a),
                data(&c),
                data(&superset),
            )))
            .unwrap();
        let attached = collection
            .ensure_exact(&mut store, &source_cover(&collection, &[commit]))
            .unwrap();
        assert_eq!(attached_facts(&attached), expected);
        let derive_inputs: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == collection.collection() =>
                {
                    Some(claim.input())
                }
                _ => None,
            })
            .collect();
        assert_eq!(derive_inputs, vec![commit.data()]);
    }

    #[test]
    fn retained_rewrite_drops_and_exact_ensure_repairs_unowned_rank9_fiber() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source.pile");
        let retained_path = directory.path().join("retained.pile");
        std::fs::File::create(&source_path).unwrap();
        std::fs::File::create(&retained_path).unwrap();
        let mut source_store = Pile::open(&source_path).unwrap();
        let mut retained_store = Pile::open(&retained_path).unwrap();

        let collection = test_collection("c7");
        let source_descriptor =
            IntoBlob::<SimpleArchive>::to_blob(collection.source_descriptor().into_facts());
        let target_descriptor =
            IntoBlob::<SimpleArchive>::to_blob(collection.descriptor().into_facts());
        source_store
            .put::<SimpleArchive, _>(source_descriptor.clone())
            .unwrap();
        source_store
            .put::<SimpleArchive, _>(target_descriptor.clone())
            .unwrap();
        let metadata = source_store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();

        let a_facts = facts([(1, 3)]);
        let b_facts = facts([(2, 4)]);
        let c_facts = facts([(3, 5)]);
        let a = (&a_facts).to_blob();
        let b = (&b_facts).to_blob();
        let c = (&c_facts).to_blob();
        for blob in [&a, &b, &c] {
            source_store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        }
        let commits = [
            pile_commit(&mut source_store, &collection, 1, &a, metadata),
            pile_commit(&mut source_store, &collection, 2, &b, metadata),
            pile_commit(&mut source_store, &collection, 3, &c, metadata),
        ];

        let ab = simplearchive_union::join(&a, &b).unwrap();
        let succinct_ab = super::super::derive_element(&ab).unwrap();
        let succinct_c = super::super::derive_element(&c).unwrap();
        let succinct_abc = super::super::join(&succinct_ab, &succinct_c).unwrap();
        source_store.put::<SimpleArchive, _>(ab.clone()).unwrap();
        for blob in [&succinct_ab, &succinct_c, &succinct_abc] {
            source_store
                .put::<SuccinctArchiveBlob, _>(blob.clone())
                .unwrap();
        }
        for record in [
            CollectionRecord::Merge(CollectionMerge::new(
                collection.source_collection(),
                data(&a),
                data(&b),
                data(&ab),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.collection(),
                data(&ab),
                data(&succinct_ab),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                collection.collection(),
                data(&c),
                data(&succinct_c),
            )),
            CollectionRecord::Merge(CollectionMerge::new(
                collection.collection(),
                data(&succinct_ab),
                data(&succinct_c),
                data(&succinct_abc),
            )),
        ] {
            source_store.insert(record).unwrap();
        }
        let ensured = collection
            .ensure_exact(&mut source_store, &source_cover(&collection, &commits))
            .unwrap();
        assert_eq!(ensured.segment_count(), 1);
        let expected_evidence = rank9_evidence(&mut source_store);
        assert_eq!(expected_evidence.len(), 1);
        assert_eq!(expected_evidence[0].input(), data(&succinct_abc));
        let rank9_handle =
            Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(expected_evidence[0].output());
        let rank9_mapping =
            IntoBlob::<SimpleArchive>::to_blob(super::super::rank9_mapping_fragment().into_facts());
        source_store.flush().unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_direct(succinct_abc.get_handle());
        source_store
            .rewrite_retained_into(&mut retained_store, &roots, WantRewritePolicy::Drop)
            .unwrap();
        source_store.close().unwrap();
        retained_store.close().unwrap();

        let mut retained_store = Pile::open(&retained_path).unwrap();
        let reader = retained_store.reader().unwrap();
        for handle in [a.get_handle(), b.get_handle(), c.get_handle()] {
            assert!(reader.contains_blob(handle).unwrap());
        }
        assert!(reader
            .contains_blob(source_descriptor.get_handle())
            .unwrap());
        assert!(reader.contains_blob(metadata).unwrap());
        assert!(reader.contains_blob(succinct_abc.get_handle()).unwrap());
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_c.get_handle()).unwrap());
        assert!(!reader
            .contains_blob(target_descriptor.get_handle())
            .unwrap());
        assert!(!reader.contains_blob(rank9_mapping.get_handle()).unwrap());
        assert!(!reader.contains_blob(rank9_handle).unwrap());
        drop(reader);
        assert_eq!(rank9_evidence(&mut retained_store), expected_evidence);

        let before = std::fs::metadata(&retained_path).unwrap().len();
        let attached = collection
            .attach_exact(&mut retained_store, &source_cover(&collection, &commits))
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        assert_eq!(attached_facts(&attached), a_facts + b_facts + c_facts);
        assert_eq!(std::fs::metadata(&retained_path).unwrap().len(), before);
        let ensured = collection
            .ensure_exact(&mut retained_store, &source_cover(&collection, &commits))
            .unwrap();
        assert_eq!(ensured.segment_count(), 1);
        let repaired = std::fs::metadata(&retained_path).unwrap().len();
        assert!(repaired > before);
        assert_eq!(rank9_evidence(&mut retained_store), expected_evidence);
        let repeated = collection
            .ensure_exact(&mut retained_store, &source_cover(&collection, &commits))
            .unwrap();
        assert_eq!(repeated.segment_count(), 1);
        assert_eq!(std::fs::metadata(&retained_path).unwrap().len(), repaired);

        let reader = retained_store.reader().unwrap();
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_ab.get_handle()).unwrap());
        assert!(!reader.contains_blob(succinct_c.get_handle()).unwrap());
        assert!(!reader
            .contains_blob(target_descriptor.get_handle())
            .unwrap());
        assert!(reader.contains_blob(rank9_mapping.get_handle()).unwrap());
        assert!(reader.contains_blob(rank9_handle).unwrap());
        drop(reader);
        retained_store.close().unwrap();
    }

    #[test]
    fn ensure_reconstructs_collected_source_ancestry_from_resident_source_seed() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source-seed.pile");
        let retained_path = directory.path().join("source-seed-retained.pile");
        std::fs::File::create(&source_path).unwrap();
        std::fs::File::create(&retained_path).unwrap();
        let mut source_store = Pile::open(&source_path).unwrap();
        let mut retained_store = Pile::open(&retained_path).unwrap();

        let collection = test_collection("c8");
        source_store
            .put::<SimpleArchive, _>(IntoBlob::<SimpleArchive>::to_blob(
                collection.source_descriptor().into_facts(),
            ))
            .unwrap();
        let metadata = source_store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let a = facts([(1, 3)]).to_blob();
        let b = facts([(2, 4)]).to_blob();
        let c = facts([(3, 5)]).to_blob();
        for blob in [&a, &b, &c] {
            source_store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        }
        let commits = [
            pile_commit(&mut source_store, &collection, 4, &a, metadata),
            pile_commit(&mut source_store, &collection, 5, &b, metadata),
            pile_commit(&mut source_store, &collection, 6, &c, metadata),
        ];
        let ab = simplearchive_union::join(&a, &b).unwrap();
        let abc = simplearchive_union::join(&ab, &c).unwrap();
        source_store.put::<SimpleArchive, _>(ab.clone()).unwrap();
        source_store.put::<SimpleArchive, _>(abc.clone()).unwrap();
        for record in [
            CollectionRecord::Merge(CollectionMerge::new(
                collection.source_collection(),
                data(&a),
                data(&b),
                data(&ab),
            )),
            CollectionRecord::Merge(CollectionMerge::new(
                collection.source_collection(),
                data(&ab),
                data(&c),
                data(&abc),
            )),
        ] {
            source_store.insert(record).unwrap();
        }
        source_store.flush().unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_direct(abc.get_handle());
        source_store
            .rewrite_retained_into(&mut retained_store, &roots, WantRewritePolicy::Drop)
            .unwrap();
        source_store.close().unwrap();
        retained_store.close().unwrap();

        let mut retained_store = Pile::open(&retained_path).unwrap();
        let reader = retained_store.reader().unwrap();
        assert!(!reader.contains_blob(ab.get_handle()).unwrap());
        assert!(reader.contains_blob(abc.get_handle()).unwrap());
        drop(reader);
        let attached = collection
            .ensure_exact(&mut retained_store, &source_cover(&collection, &commits))
            .unwrap();
        assert_eq!(attached.segment_count(), 1);
        let derive_inputs: Vec<_> = retained_store
            .records()
            .unwrap()
            .map(Result::unwrap)
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.collection() == collection.collection() =>
                {
                    Some(claim.input())
                }
                _ => None,
            })
            .collect();
        assert_eq!(derive_inputs, vec![data(&abc)]);
        retained_store.close().unwrap();
    }
}
