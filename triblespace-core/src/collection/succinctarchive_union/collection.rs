//! Exact-cover facade for canonical Rank9-accelerated SuccinctArchive collections.
//!
//! The public view is an ordinary two-stage collection derivation:
//!
//! ```text
//! SimpleArchive --DERIVE--> SuccinctArchiveBlob
//!                --DERIVE--> Rank9AcceleratedSuccinctArchiveBlob
//! ```
//!
//! Both stages use the same exact-cover resolver and ordinary `DERIVE`
//! records. Raw SuccinctArchive members own the directly materialized `MERGE`;
//! every accelerated member is instead the exact image of one raw member. The
//! accelerated encoding is an ordinary blob whose header names its portable
//! raw source. Attaching it resolves that source and constructs the query
//! runtime without a sidecar record family or wrapper artifact.

use std::cell::Cell;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, Rank9AcceleratedSuccinctArchiveBlob, SuccinctArchive, SuccinctArchiveBlob,
    SuccinctArchiveError, UnionArchive,
};
use crate::blob::{Blob, TryFromBlob};
use crate::collection::exact_derived::{ExactDerivedCollection, ExactDerivedCollectionError};
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::{
    Collection, CollectionData, CollectionMapping, CollectionOperationError, CollectionStore,
    Cover, CoverAdvanceError, FactCover, TryFromCover, TryFromCoverError,
};
use crate::inline::encodings::hash::Handle;
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta};
use crate::trible::Fragment;

use super::{RawToRank9AcceleratedMapping, SimpleToSuccinctMapping};

impl TryFromCover<SuccinctArchiveBlob> for UnionArchive<OrderedUniverse> {
    type Error = SuccinctArchiveError;

    fn try_from_cover<R>(
        cover: &Cover<SuccinctArchiveBlob>,
        snapshot: &R,
    ) -> Result<Self, TryFromCoverError<R::GetError<Infallible>, Self::Error>>
    where
        R: BlobStoreGet,
    {
        let mut segments = Vec::with_capacity(cover.len());
        for handle in cover.members() {
            let member = Handle::<SuccinctArchiveBlob>::to_hash(handle);
            let root = snapshot
                .get::<Blob<SuccinctArchiveBlob>, SuccinctArchiveBlob>(handle)
                .map_err(|source| TryFromCoverError::MemberGet { member, source })?;
            segments.push(SuccinctArchive::try_from_blob(root).map_err(TryFromCoverError::View)?);
        }
        if segments.is_empty() {
            segments.push(
                super::empty()
                    .try_from_blob()
                    .map_err(TryFromCoverError::View)?,
            );
        }
        Ok(UnionArchive::new(segments))
    }
}

/// Failure to attach one Rank9 root to the raw archive named in its header.
#[derive(Debug)]
pub enum Rank9AcceleratedViewError {
    /// The accelerated root header or exact raw/index pair is invalid.
    Invalid(SuccinctArchiveError),
    /// The exact raw child named by an accelerated root is not resident.
    MissingRaw {
        /// Accelerated root whose child could not be loaded.
        member: CollectionData,
        /// Backend diagnostic.
        reason: String,
    },
}

impl fmt::Display for Rank9AcceleratedViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(source) => source.fmt(formatter),
            Self::MissingRaw { member, reason } => write!(
                formatter,
                "accelerated SuccinctArchive member {} is missing its raw child: {reason}",
                hex::encode_upper(member.raw),
            ),
        }
    }
}

impl Error for Rank9AcceleratedViewError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalid(source) => Some(source),
            Self::MissingRaw { .. } => None,
        }
    }
}

impl TryFromCover<Rank9AcceleratedSuccinctArchiveBlob> for UnionArchive<OrderedUniverse> {
    type Error = Rank9AcceleratedViewError;

    fn try_from_cover<R>(
        cover: &Cover<Rank9AcceleratedSuccinctArchiveBlob>,
        snapshot: &R,
    ) -> Result<Self, TryFromCoverError<R::GetError<Infallible>, Self::Error>>
    where
        R: BlobStoreGet,
    {
        let mut segments = Vec::with_capacity(cover.len());
        for handle in cover.members() {
            let member = Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(handle);
            let root = snapshot
                .get::<Blob<Rank9AcceleratedSuccinctArchiveBlob>, _>(handle)
                .map_err(|source| TryFromCoverError::MemberGet { member, source })?;
            let source = Rank9AcceleratedSuccinctArchiveBlob::source_handle(&root)
                .map_err(Rank9AcceleratedViewError::Invalid)
                .map_err(TryFromCoverError::View)?;
            let raw = snapshot
                .get::<crate::blob::Blob<SuccinctArchiveBlob>, SuccinctArchiveBlob>(source)
                .map_err(|source| {
                    TryFromCoverError::View(Rank9AcceleratedViewError::MissingRaw {
                        member,
                        reason: source.to_string(),
                    })
                })?;
            segments.push(
                SuccinctArchive::from_accelerated_parts(raw, root)
                    .map_err(Rank9AcceleratedViewError::Invalid)
                    .map_err(TryFromCoverError::View)?,
            );
        }
        if segments.is_empty() {
            segments.push(
                super::empty()
                    .try_from_blob()
                    .map_err(Rank9AcceleratedViewError::Invalid)
                    .map_err(TryFromCoverError::View)?,
            );
        }
        Ok(UnionArchive::new(segments))
    }
}

/// Failure to complete or attach one exact accelerated Succinct cover.
#[derive(Debug)]
pub enum SuccinctArchiveCollectionError {
    /// Exact-cover resolution, construction, or storage failed.
    Exact(ExactDerivedCollectionError),
    /// Explicit raw target compaction failed.
    Compaction(ExactTargetCompactionError),
    /// A freshly attached accelerated cover could not become a query view.
    View(Rank9AcceleratedViewError),
    /// A selected accelerated member could not be fetched.
    MemberGet {
        /// Selected physical member.
        member: CollectionData,
        /// Backend diagnostic.
        reason: String,
    },
    /// A fresh immutable store observation could not be frozen after writes.
    Snapshot(String),
}

impl fmt::Display for SuccinctArchiveCollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(formatter),
            Self::Compaction(source) => source.fmt(formatter),
            Self::View(source) => source.fmt(formatter),
            Self::MemberGet { member, reason } => write!(
                formatter,
                "accelerated SuccinctArchive member {} could not be fetched: {reason}",
                hex::encode_upper(member.raw),
            ),
            Self::Snapshot(source) => {
                write!(formatter, "freeze accelerated-cover snapshot: {source}")
            }
        }
    }
}

impl Error for SuccinctArchiveCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Compaction(source) => Some(source),
            Self::View(source) => Some(source),
            Self::MemberGet { .. } => None,
            Self::Snapshot(_) => None,
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

impl From<Rank9AcceleratedViewError> for SuccinctArchiveCollectionError {
    fn from(source: Rank9AcceleratedViewError) -> Self {
        Self::View(source)
    }
}

fn accelerated_view<R>(
    snapshot: &R,
    cover: &Cover<Rank9AcceleratedSuccinctArchiveBlob>,
) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
where
    R: BlobStoreGet,
{
    match UnionArchive::try_from_cover(cover, snapshot) {
        Ok(archive) => Ok(archive),
        Err(TryFromCoverError::MemberGet { member, source }) => {
            Err(SuccinctArchiveCollectionError::MemberGet {
                member,
                reason: source.to_string(),
            })
        }
        Err(TryFromCoverError::View(source)) => Err(source.into()),
    }
}

/// Canonical accelerated SuccinctArchive projection of one SimpleArchive union.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuccinctArchiveCollection {
    source: Collection<SimpleArchive>,
    raw: Collection<SuccinctArchiveBlob>,
    accelerated: Collection<Rank9AcceleratedSuccinctArchiveBlob>,
}

/// Exact work performed by one successful [`SuccinctArchiveView::ensure`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SuccinctArchiveViewWork {
    /// Distinct payload members represented after the call.
    pub cover_members: usize,
    /// Payload members newly processed by this call.
    pub processed_members: usize,
    /// Previously materialized payload members reused without replaying data work.
    pub reused_members: usize,
    /// Canonical SimpleArchive-to-raw derivations.
    pub derive: u64,
    /// Cumulative bytes supplied to SimpleArchive-to-raw derivations.
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
    inner: SimpleToSuccinctMapping,
    work: Cell<SuccinctArchiveViewWork>,
}

impl MeasuredSuccinctHomomorphism {
    fn new(inner: SimpleToSuccinctMapping, work: SuccinctArchiveViewWork) -> Self {
        Self {
            inner,
            work: Cell::new(work),
        }
    }
}

impl CollectionMapping for MeasuredSuccinctHomomorphism {
    type Source = SimpleArchive;
    type Target = SuccinctArchiveBlob;

    fn fragment(&self) -> Fragment {
        self.inner.fragment()
    }

    fn bind(source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        Ok(Self::new(
            SimpleToSuccinctMapping::bind(source, target)?,
            SuccinctArchiveViewWork::default(),
        ))
    }

    fn map<R>(
        &self,
        source: &crate::blob::Blob<SimpleArchive>,
        reader: &R,
    ) -> Result<crate::blob::Blob<SuccinctArchiveBlob>, CollectionOperationError>
    where
        R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
    {
        let mut work = self.work.get();
        work.derive += 1;
        work.input_bytes += source.bytes.len() as u64;
        self.work.set(work);
        self.inner.map(source, reader)
    }
}

/// One in-process accelerated Succinct view maintained across observations.
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
    pub fn last_work(&self) -> Option<SuccinctArchiveViewWork> {
        self.last_work
    }

    /// Ensure and retain the exact view for the current source cover.
    pub fn ensure<S>(
        &mut self,
        store: &mut S,
        current: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Snapshot: BlobStoreMeta + crate::collection::CollectionRead,
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
            None => self.ensure_measured(
                store,
                current,
                SuccinctArchiveViewWork::with_support(current.len(), current.len(), 0),
            )?,
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
                Err(CoverAdvanceError::ResetRequired { .. }) => self.ensure_measured(
                    store,
                    current,
                    SuccinctArchiveViewWork::with_support(current.len(), current.len(), 0),
                )?,
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
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Snapshot: BlobStoreMeta + crate::collection::CollectionRead,
    {
        let measured = MeasuredSuccinctHomomorphism::new(SimpleToSuccinctMapping, work);
        let raw_kernel = ExactDerivedCollection::with_mapping(
            self.collection.source,
            self.collection.raw,
            measured,
        )?;
        let raw_cover = raw_kernel.ensure_exact(store, cover)?;
        let work = raw_kernel
            .mapping_override()
            .expect("measured kernel retains its explicit mapping")
            .work
            .get();
        let accelerated = self
            .collection
            .rank9_derivation()?
            .ensure_member_images(store, &raw_cover)?;
        let snapshot = store
            .snapshot()
            .map_err(|source| SuccinctArchiveCollectionError::Snapshot(source.to_string()))?;
        Ok((accelerated_view(&snapshot, &accelerated)?, work))
    }
}

impl SuccinctArchiveCollection {
    /// Bind the facade to its three store-created lifecycle values.
    pub fn new(
        source: Collection<SimpleArchive>,
        raw: Collection<SuccinctArchiveBlob>,
        accelerated: Collection<Rank9AcceleratedSuccinctArchiveBlob>,
    ) -> Self {
        Self {
            source,
            raw,
            accelerated,
        }
    }

    /// Create an empty in-process continuation for this projection.
    pub fn exact_view(&self) -> SuccinctArchiveView {
        SuccinctArchiveView::new(self.clone())
    }

    /// Store-issued source collection.
    pub fn source_collection(&self) -> Collection<SimpleArchive> {
        self.source
    }

    /// Store-issued raw intermediate collection.
    pub fn raw_collection(&self) -> Collection<SuccinctArchiveBlob> {
        self.raw
    }

    /// Store-issued accelerated collection.
    pub fn collection(&self) -> Collection<Rank9AcceleratedSuccinctArchiveBlob> {
        self.accelerated
    }

    /// Attach the exact accelerated cover without writing.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + crate::collection::CollectionRead,
    {
        let raw_cover = self.raw_kernel()?.attach_exact(store, source_cover)?;
        let accelerated = self
            .rank9_derivation()?
            .attach_member_images(store, &raw_cover)?;
        let snapshot = store
            .snapshot()
            .map_err(|source| SuccinctArchiveCollectionError::Snapshot(source.to_string()))?;
        accelerated_view(&snapshot, &accelerated)
    }

    /// Ensure both ordinary derivation stages and attach the exact view.
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Snapshot: BlobStoreMeta + crate::collection::CollectionRead,
    {
        let raw_cover = self.raw_kernel()?.ensure_exact(store, source_cover)?;
        let accelerated = self
            .rank9_derivation()?
            .ensure_member_images(store, &raw_cover)?;
        let snapshot = store
            .snapshot()
            .map_err(|source| SuccinctArchiveCollectionError::Snapshot(source.to_string()))?;
        accelerated_view(&snapshot, &accelerated)
    }

    /// Compact the raw target, then ensure the matching accelerated cover.
    pub fn compact_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Snapshot: BlobStoreMeta + crate::collection::CollectionRead,
    {
        let raw = compact_exact_target(&self.raw_kernel()?, store, source_cover)?;
        let accelerated = self.rank9_derivation()?.ensure_member_images(store, &raw)?;
        let snapshot = store
            .snapshot()
            .map_err(|source| SuccinctArchiveCollectionError::Snapshot(source.to_string()))?;
        accelerated_view(&snapshot, &accelerated)
    }

    fn raw_kernel(
        &self,
    ) -> Result<ExactDerivedCollection<SimpleToSuccinctMapping>, ExactDerivedCollectionError> {
        ExactDerivedCollection::new(self.source, self.raw)
    }

    fn rank9_derivation(
        &self,
    ) -> Result<ExactDerivedCollection<RawToRank9AcceleratedMapping>, ExactDerivedCollectionError>
    {
        ExactDerivedCollection::new(self.raw, self.accelerated)
    }
}

#[cfg(test)]
mod tests {
    use anybytes::Bytes;
    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::blob::encodings::succinctarchive::{
        Rank9AcceleratedSuccinctArchiveBlob, SuccinctArchiveBlob,
    };
    use crate::blob::{Blob, IntoBlob};
    use crate::collection::descriptor;
    use crate::collection::{
        CollectionDerive, CollectionEncoding, CollectionMapping, CollectionMerge, CollectionPolicy,
        CollectionRead, CollectionRecord, CollectionStore, CollectionStoreExt, Cover, FactCover,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::metadata::MetaDescribe;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::{BlobStorePut, SnapshotSource};
    use crate::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

    use super::super::RawToRank9AcceleratedMapping;
    use super::*;

    fn authority() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[7; 32]).verifying_key()
    }

    fn direct_policy() -> CollectionPolicy {
        CollectionPolicy::new(
            crate::collection::AdmissionPolicy::direct(authority()),
            crate::collection::AdmissionPolicy::direct(authority()),
        )
    }

    fn facade(store: &mut MemoryRepo) -> SuccinctArchiveCollection {
        let source = store.collection("facts", direct_policy()).unwrap();
        let raw = store
            .derive(source, SimpleToSuccinctMapping, direct_policy())
            .unwrap();
        let accelerated = store
            .derive(raw, RawToRank9AcceleratedMapping, direct_policy())
            .unwrap();
        SuccinctArchiveCollection::new(source, raw, accelerated)
    }

    fn row(entity: u8, attribute: u8, value: u8) -> Trible {
        let mut row = [value; TRIBLE_LEN];
        row[..16].fill(entity);
        row[16..32].fill(attribute);
        Trible::force_raw(row).unwrap()
    }

    fn raw(rows: impl IntoIterator<Item = Trible>) -> Blob<SuccinctArchiveBlob> {
        let mut set = TribleSet::new();
        for row in rows {
            set.insert(&row);
        }
        let simple: Blob<SimpleArchive> = set.to_blob();
        super::super::derive_element(&simple).unwrap()
    }

    fn accelerated(raw: &Blob<SuccinctArchiveBlob>) -> Blob<Rank9AcceleratedSuccinctArchiveBlob> {
        let mut store = MemoryRepo::default();
        let snapshot = store.snapshot().unwrap();
        RawToRank9AcceleratedMapping.map(raw, &snapshot).unwrap()
    }

    #[test]
    fn facade_lifecycle_is_a_two_stage_ordinary_derivation() {
        let mut store = MemoryRepo::default();
        let facade = facade(&mut store);
        let snapshot = store.snapshot().unwrap();
        let raw_descriptor = crate::collection::api::load_collection_descriptor(
            &snapshot,
            facade.raw_collection().handle(),
        )
        .unwrap()
        .fragment;
        let accelerated_descriptor = crate::collection::api::load_collection_descriptor(
            &snapshot,
            facade.collection().handle(),
        )
        .unwrap()
        .fragment;
        assert_eq!(
            descriptor::source(raw_descriptor.facts()).unwrap(),
            Some(facade.source_collection().handle())
        );
        assert_eq!(
            descriptor::source(accelerated_descriptor.facts()).unwrap(),
            Some(facade.raw_collection().handle())
        );
        assert_eq!(
            descriptor::representation(raw_descriptor.facts()).unwrap(),
            SuccinctArchiveBlob::id()
        );
        assert_eq!(
            descriptor::representation(accelerated_descriptor.facts()).unwrap(),
            Rank9AcceleratedSuccinctArchiveBlob::id()
        );
    }

    #[test]
    fn ensure_uses_ordinary_derives_for_both_stages() {
        let mut store = MemoryRepo::default();
        let facade = facade(&mut store);
        let source: Blob<SimpleArchive> = [row(1, 2, 3), row(4, 5, 6)]
            .into_iter()
            .collect::<TribleSet>()
            .to_blob();
        store.put::<SimpleArchive, _>(source.clone()).unwrap();
        let cover = FactCover::from_data(
            facade.source_collection(),
            [Handle::<SimpleArchive>::to_hash(source.get_handle())],
        );

        let archive = facade.ensure_exact(&mut store, &cover).unwrap();
        assert_eq!(archive.iter().count(), 2);

        let snapshot = store.snapshot().unwrap();
        let records = snapshot
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let raw = records
            .iter()
            .find_map(|record| match record {
                CollectionRecord::Derive(derive)
                    if derive.collection() == facade.raw_collection().handle() =>
                {
                    Some(derive.output())
                }
                _ => None,
            })
            .expect("raw DERIVE was published");
        let accelerated = records
            .iter()
            .find_map(|record| match record {
                CollectionRecord::Derive(derive)
                    if derive.collection() == facade.collection().handle() =>
                {
                    assert_eq!(derive.input(), raw);
                    Some(derive.output())
                }
                _ => None,
            })
            .expect("accelerated DERIVE was published");
        assert_ne!(raw, accelerated);
    }

    #[test]
    fn compacting_raw_constructs_the_exact_accelerated_image() {
        let mut store = MemoryRepo::default();
        let facade = facade(&mut store);
        let a: Blob<SimpleArchive> = [row(1, 2, 3)].into_iter().collect::<TribleSet>().to_blob();
        let b: Blob<SimpleArchive> = [row(4, 5, 6)].into_iter().collect::<TribleSet>().to_blob();
        store.put::<SimpleArchive, _>(a.clone()).unwrap();
        store.put::<SimpleArchive, _>(b.clone()).unwrap();
        let cover = FactCover::from_data(
            facade.source_collection(),
            [
                Handle::<SimpleArchive>::to_hash(a.get_handle()),
                Handle::<SimpleArchive>::to_hash(b.get_handle()),
            ],
        );

        let raw = facade
            .raw_kernel()
            .unwrap()
            .ensure_exact(&mut store, &cover)
            .unwrap();
        facade
            .rank9_derivation()
            .unwrap()
            .ensure_member_images(&mut store, &raw)
            .unwrap();

        let archive = facade.compact_exact(&mut store, &cover).unwrap();
        assert_eq!(archive.iter().count(), 2);

        let snapshot = store.snapshot().unwrap();
        let records = snapshot
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let compacted_raw = records
            .iter()
            .find_map(|record| match record {
                CollectionRecord::Merge(merge)
                    if merge.collection() == facade.raw_collection().handle() =>
                {
                    Some(merge.result())
                }
                _ => None,
            })
            .expect("raw compaction published a MERGE");
        assert!(records.iter().any(|record| matches!(
            record,
            CollectionRecord::Derive(derive)
                if derive.collection() == facade.collection().handle()
                    && derive.input() == compacted_raw
        )));
        assert!(!records.iter().any(|record| matches!(
            record,
            CollectionRecord::Merge(merge)
                if merge.collection() == facade.collection().handle()
        )));
    }

    #[test]
    fn exact_acceleration_does_not_replace_named_raw_members_with_their_union() {
        let mut store = MemoryRepo::default();
        let facade = facade(&mut store);
        let a = raw([row(1, 2, 3)]);
        let b = raw([row(4, 5, 6)]);
        let c = super::super::join(&a, &b).unwrap();
        let a_data = Handle::<SuccinctArchiveBlob>::to_hash(a.get_handle());
        let b_data = Handle::<SuccinctArchiveBlob>::to_hash(b.get_handle());
        let c_data = Handle::<SuccinctArchiveBlob>::to_hash(c.get_handle());
        let fc = accelerated(&c);
        let fc_data = Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(fc.get_handle());

        for member in [a.clone(), b.clone(), c] {
            store.put::<SuccinctArchiveBlob, _>(member).unwrap();
        }
        store
            .put::<Rank9AcceleratedSuccinctArchiveBlob, _>(fc)
            .unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                facade.raw_collection().handle(),
                a_data,
                b_data,
                c_data,
            )))
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                facade.collection().handle(),
                c_data,
                fc_data,
            )))
            .unwrap();

        let raw_cover = Cover::from_data(facade.raw_collection(), [a_data, b_data]);
        let attached = facade
            .rank9_derivation()
            .unwrap()
            .ensure_member_images(&mut store, &raw_cover)
            .unwrap();

        let mut actual = attached.data_members().collect::<Vec<_>>();
        let mut expected = [a, b]
            .iter()
            .map(|raw| {
                let root = accelerated(raw);
                Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(root.get_handle())
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
        assert!(!actual.contains(&fc_data));

        let snapshot = store.snapshot().unwrap();
        let records = snapshot
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        for input in [a_data, b_data] {
            assert!(records.iter().any(|record| matches!(
                record,
                CollectionRecord::Derive(derive)
                    if derive.collection() == facade.collection().handle()
                        && derive.input() == input
            )));
        }
    }

    #[test]
    fn accelerated_root_without_raw_child_is_not_resident() {
        let raw = raw([row(1, 2, 3)]);
        let root = accelerated(&raw);
        let mut store = MemoryRepo::default();
        store
            .put::<Rank9AcceleratedSuccinctArchiveBlob, _>(root.clone())
            .unwrap();
        let snapshot = store.snapshot().unwrap();

        assert!(matches!(
            Rank9AcceleratedSuccinctArchiveBlob::validate_member(
                &Fragment::empty(), &root, &snapshot,
            ),
            Err(CollectionOperationError::Fatal(reason))
                if reason.contains("raw child is not resident")
        ));
    }

    #[test]
    fn accelerated_member_validation_rejects_a_corrupt_index() {
        let raw = raw([row(1, 2, 3)]);
        let root = accelerated(&raw);
        let mut bytes = root.bytes.as_ref().to_vec();
        let last = bytes.last_mut().expect("Rank9 root is not empty");
        *last ^= 1;
        let corrupted = Blob::<Rank9AcceleratedSuccinctArchiveBlob>::new(Bytes::from_source(bytes));
        let mut store = MemoryRepo::default();
        store.put::<SuccinctArchiveBlob, _>(raw).unwrap();
        let snapshot = store.snapshot().unwrap();

        assert!(matches!(
            Rank9AcceleratedSuccinctArchiveBlob::validate_member(
                &Fragment::empty(),
                &corrupted,
                &snapshot,
            ),
            Err(CollectionOperationError::Fatal(_))
        ));
    }
}
