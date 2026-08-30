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
use std::error::Error;
use std::fmt;

use ed25519_dalek::VerifyingKey;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, Rank9AcceleratedSuccinctArchiveBlob, SuccinctArchive, SuccinctArchiveBlob,
    SuccinctArchiveError, UnionArchive,
};
use crate::blob::TryFromBlob;
use crate::collection::exact_derived::{ExactDerivedCollection, ExactDerivedCollectionError};
use crate::collection::exact_target_compaction::{
    compact_exact_target, ExactTargetCompactionError,
};
use crate::collection::simplearchive_union;
use crate::collection::{
    CollectionData, CollectionHandle, CollectionMapping, CollectionOperationError, CollectionStore,
    CoverAdvanceError, CoverAttachment, FactCover, TryFromCover,
};
use crate::inline::encodings::hash::Handle;
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta};
use crate::trible::Fragment;

use super::{RawToRank9AcceleratedMapping, SimpleToSuccinctMapping};

impl TryFromCover<SuccinctArchiveBlob> for UnionArchive<OrderedUniverse> {
    type Error = SuccinctArchiveError;

    fn try_from_cover<R>(
        attachment: CoverAttachment<SuccinctArchiveBlob>,
        _reader: &R,
    ) -> Result<Self, Self::Error>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        let mut segments = attachment
            .into_blobs()
            .map(SuccinctArchive::try_from_blob)
            .collect::<Result<Vec<_>, _>>()?;
        if segments.is_empty() {
            segments.push(super::empty().try_from_blob()?);
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
        attachment: CoverAttachment<Rank9AcceleratedSuccinctArchiveBlob>,
        reader: &R,
    ) -> Result<Self, Self::Error>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        let mut segments = Vec::with_capacity(attachment.len());
        for root in attachment.into_blobs() {
            let member = Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(root.get_handle());
            let source = Rank9AcceleratedSuccinctArchiveBlob::source_handle(&root)
                .map_err(Rank9AcceleratedViewError::Invalid)?;
            let raw = reader
                .get::<crate::blob::Blob<SuccinctArchiveBlob>, SuccinctArchiveBlob>(source)
                .map_err(|source| Rank9AcceleratedViewError::MissingRaw {
                    member,
                    reason: source.to_string(),
                })?;
            segments.push(
                SuccinctArchive::from_accelerated_parts(raw, root)
                    .map_err(Rank9AcceleratedViewError::Invalid)?,
            );
        }
        if segments.is_empty() {
            segments.push(
                super::empty()
                    .try_from_blob()
                    .map_err(Rank9AcceleratedViewError::Invalid)?,
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
    /// A fresh reader for the just-attached cover could not be opened.
    Reader(String),
}

impl fmt::Display for SuccinctArchiveCollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(formatter),
            Self::Compaction(source) => source.fmt(formatter),
            Self::View(source) => source.fmt(formatter),
            Self::Reader(source) => write!(formatter, "open accelerated cover reader: {source}"),
        }
    }
}

impl Error for SuccinctArchiveCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Compaction(source) => Some(source),
            Self::View(source) => Some(source),
            Self::Reader(_) => None,
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

fn accelerated_view<S>(
    store: &mut S,
    attachment: CoverAttachment<Rank9AcceleratedSuccinctArchiveBlob>,
) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
where
    S: BlobStore,
    S::Reader: BlobStoreMeta,
{
    let reader = store
        .reader()
        .map_err(|source| SuccinctArchiveCollectionError::Reader(source.to_string()))?;
    UnionArchive::try_from_cover(attachment, &reader).map_err(Into::into)
}

/// Canonical accelerated SuccinctArchive projection of one SimpleArchive union.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuccinctArchiveCollection {
    name: String,
    source_authority: VerifyingKey,
    source_reach: Fragment,
    authority: VerifyingKey,
    reach: Fragment,
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

impl CollectionMapping<SimpleArchive, SuccinctArchiveBlob> for MeasuredSuccinctHomomorphism {
    fn bind(source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        Ok(Self::new(
            SimpleToSuccinctMapping::bind(source, target)?,
            SuccinctArchiveViewWork::default(),
        ))
    }

    fn map(
        &self,
        source: &crate::blob::Blob<SimpleArchive>,
    ) -> Result<crate::blob::Blob<SuccinctArchiveBlob>, CollectionOperationError> {
        let mut work = self.work.get();
        work.derive += 1;
        work.input_bytes += source.bytes.len() as u64;
        self.work.set(work);
        self.inner.map(source)
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
        S::Reader: BlobStoreMeta,
    {
        let source = self.collection.source_descriptor();
        let raw = self.collection.raw_descriptor();
        let inner = SimpleToSuccinctMapping::bind(&source, &raw)
            .map_err(|error| ExactDerivedCollectionError::Resolution(error.to_string()))?;
        let measured = MeasuredSuccinctHomomorphism::new(inner, work);
        let raw_kernel = ExactDerivedCollection::with_mapping(source, raw, measured)?;
        let raw_attachment = raw_kernel.ensure_exact(store, cover)?;
        let work = raw_kernel.mapping().work.get();
        let raw_cover = raw_attachment.cover().clone();
        let accelerated = self
            .collection
            .rank9_derivation()?
            .ensure_member_images(store, &raw_cover)?;
        Ok((accelerated_view(store, accelerated)?, work))
    }
}

impl SuccinctArchiveCollection {
    /// Construct the canonical accelerated projection for one named root.
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

    /// Create an empty in-process continuation for this projection.
    pub fn exact_view(&self) -> SuccinctArchiveView {
        SuccinctArchiveView::new(self.clone())
    }

    /// How far the source collection may travel.
    pub fn source_reach(&self) -> &Fragment {
        &self.source_reach
    }

    /// How far both derived Succinct collections may travel.
    pub fn reach(&self) -> &Fragment {
        &self.reach
    }

    /// Name of the source collection this projection is taken over.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Mandatory capability trust root declared by the source descriptor.
    pub fn source_authority(&self) -> VerifyingKey {
        self.source_authority
    }

    /// Mandatory capability trust root shared by the two derived collections.
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

    /// Canonical intermediate raw-SuccinctArchive descriptor.
    pub fn raw_descriptor(&self) -> Fragment {
        super::descriptor(self.source_collection(), self.authority, self.reach.clone())
    }

    /// Identity of the intermediate raw SuccinctArchive collection.
    pub fn raw_collection(&self) -> CollectionHandle {
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.raw_descriptor().into_facts())
            .get_handle()
    }

    /// Canonical public Rank9-accelerated SuccinctArchive descriptor.
    pub fn descriptor(&self) -> Fragment {
        super::accelerated_descriptor(self.raw_collection(), self.authority, self.reach.clone())
    }

    /// Identity of the accelerated collection this facade maintains.
    pub fn collection(&self) -> CollectionHandle {
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.descriptor().into_facts()).get_handle()
    }

    /// Attach the exact accelerated cover without writing.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let raw_attachment = self.raw_kernel()?.attach_exact(store, source_cover)?;
        let raw_cover = raw_attachment.cover().clone();
        let accelerated = self
            .rank9_derivation()?
            .attach_member_images(store, &raw_cover)?;
        accelerated_view(store, accelerated)
    }

    /// Ensure both ordinary derivation stages and attach the exact view.
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let raw_attachment = self.raw_kernel()?.ensure_exact(store, source_cover)?;
        let raw_cover = raw_attachment.cover().clone();
        let accelerated = self
            .rank9_derivation()?
            .ensure_member_images(store, &raw_cover)?;
        accelerated_view(store, accelerated)
    }

    /// Compact the raw target, then ensure the matching accelerated cover.
    pub fn compact_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<UnionArchive<OrderedUniverse>, SuccinctArchiveCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let raw = compact_exact_target(&self.raw_kernel()?, store, source_cover)?;
        let raw_cover = raw.cover().clone();
        let accelerated = self
            .rank9_derivation()?
            .ensure_member_images(store, &raw_cover)?;
        accelerated_view(store, accelerated)
    }

    fn raw_kernel(
        &self,
    ) -> Result<
        ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob, SimpleToSuccinctMapping>,
        ExactDerivedCollectionError,
    > {
        ExactDerivedCollection::new(self.source_descriptor(), self.raw_descriptor())
    }

    fn rank9_derivation(
        &self,
    ) -> Result<
        ExactDerivedCollection<
            SuccinctArchiveBlob,
            Rank9AcceleratedSuccinctArchiveBlob,
            RawToRank9AcceleratedMapping,
        >,
        ExactDerivedCollectionError,
    > {
        ExactDerivedCollection::new(self.raw_descriptor(), self.descriptor())
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
    use crate::collection::reach;
    use crate::collection::{
        Collection, CollectionDerive, CollectionEncoding, CollectionMapping, CollectionMerge,
        CollectionRecord, CollectionStore, Cover, FactCover,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::metadata::MetaDescribe;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::{BlobStore, BlobStorePut};
    use crate::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

    use super::super::RawToRank9AcceleratedMapping;
    use super::*;

    fn authority() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[7; 32]).verifying_key()
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

    #[test]
    fn facade_descriptor_is_a_two_stage_ordinary_derivation() {
        let facade = SuccinctArchiveCollection::new(
            "facts",
            authority(),
            reach::private(),
            authority(),
            reach::private(),
        );
        assert_eq!(
            descriptor::source(facade.raw_descriptor().facts()).unwrap(),
            Some(facade.source_collection())
        );
        assert_eq!(
            descriptor::source(facade.descriptor().facts()).unwrap(),
            Some(facade.raw_collection())
        );
        assert_eq!(
            descriptor::representation(facade.raw_descriptor().facts()).unwrap(),
            SuccinctArchiveBlob::id()
        );
        assert_eq!(
            descriptor::representation(facade.descriptor().facts()).unwrap(),
            Rank9AcceleratedSuccinctArchiveBlob::id()
        );
    }

    #[test]
    fn ensure_uses_ordinary_derives_for_both_stages() {
        let facade = SuccinctArchiveCollection::new(
            "facts",
            authority(),
            reach::private(),
            authority(),
            reach::private(),
        );
        let source: Blob<SimpleArchive> = [row(1, 2, 3), row(4, 5, 6)]
            .into_iter()
            .collect::<TribleSet>()
            .to_blob();
        let mut store = MemoryRepo::default();
        store.put::<SimpleArchive, _>(source.clone()).unwrap();
        let source_collection = crate::collection::Collection::<SimpleArchive>::from_descriptor(
            &facade.source_descriptor(),
        )
        .unwrap();
        let cover = FactCover::from_data(
            source_collection,
            [Handle::<SimpleArchive>::to_hash(source.get_handle())],
        );

        let archive = facade.ensure_exact(&mut store, &cover).unwrap();
        assert_eq!(archive.iter().count(), 2);

        let records = store
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let raw = records
            .iter()
            .find_map(|record| match record {
                CollectionRecord::Derive(derive)
                    if derive.collection() == facade.raw_collection() =>
                {
                    Some(derive.output())
                }
                _ => None,
            })
            .expect("raw DERIVE was published");
        let accelerated = records
            .iter()
            .find_map(|record| match record {
                CollectionRecord::Derive(derive) if derive.collection() == facade.collection() => {
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
        let facade = SuccinctArchiveCollection::new(
            "facts",
            authority(),
            reach::private(),
            authority(),
            reach::private(),
        );
        let a: Blob<SimpleArchive> = [row(1, 2, 3)].into_iter().collect::<TribleSet>().to_blob();
        let b: Blob<SimpleArchive> = [row(4, 5, 6)].into_iter().collect::<TribleSet>().to_blob();
        let mut store = MemoryRepo::default();
        store.put::<SimpleArchive, _>(a.clone()).unwrap();
        store.put::<SimpleArchive, _>(b.clone()).unwrap();
        let source_collection = crate::collection::Collection::<SimpleArchive>::from_descriptor(
            &facade.source_descriptor(),
        )
        .unwrap();
        let cover = FactCover::from_data(
            source_collection,
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
            .ensure_member_images(&mut store, raw.cover())
            .unwrap();

        let archive = facade.compact_exact(&mut store, &cover).unwrap();
        assert_eq!(archive.iter().count(), 2);

        let records = store
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let compacted_raw = records
            .iter()
            .find_map(|record| match record {
                CollectionRecord::Merge(merge) if merge.collection() == facade.raw_collection() => {
                    Some(merge.result())
                }
                _ => None,
            })
            .expect("raw compaction published a MERGE");
        assert!(records.iter().any(|record| matches!(
            record,
            CollectionRecord::Derive(derive)
                if derive.collection() == facade.collection()
                    && derive.input() == compacted_raw
        )));
        assert!(!records.iter().any(|record| matches!(
            record,
            CollectionRecord::Merge(merge) if merge.collection() == facade.collection()
        )));
    }

    #[test]
    fn exact_acceleration_does_not_replace_named_raw_members_with_their_union() {
        let facade = SuccinctArchiveCollection::new(
            "facts",
            authority(),
            reach::private(),
            authority(),
            reach::private(),
        );
        let a = raw([row(1, 2, 3)]);
        let b = raw([row(4, 5, 6)]);
        let c = super::super::join(&a, &b).unwrap();
        let a_data = Handle::<SuccinctArchiveBlob>::to_hash(a.get_handle());
        let b_data = Handle::<SuccinctArchiveBlob>::to_hash(b.get_handle());
        let c_data = Handle::<SuccinctArchiveBlob>::to_hash(c.get_handle());
        let fc = RawToRank9AcceleratedMapping.map(&c).unwrap();
        let fc_data = Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(fc.get_handle());

        let mut store = MemoryRepo::default();
        for member in [a.clone(), b.clone(), c] {
            store.put::<SuccinctArchiveBlob, _>(member).unwrap();
        }
        store
            .put::<Rank9AcceleratedSuccinctArchiveBlob, _>(fc)
            .unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                facade.raw_collection(),
                a_data,
                b_data,
                c_data,
            )))
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                facade.collection(),
                c_data,
                fc_data,
            )))
            .unwrap();

        let raw_collection =
            Collection::<SuccinctArchiveBlob>::from_descriptor(&facade.raw_descriptor()).unwrap();
        let raw_cover = Cover::from_data(raw_collection, [a_data, b_data]);
        let attached = facade
            .rank9_derivation()
            .unwrap()
            .ensure_member_images(&mut store, &raw_cover)
            .unwrap();

        let mut actual = attached.cover().data_members().collect::<Vec<_>>();
        let mut expected = [a, b]
            .iter()
            .map(|raw| {
                let root = RawToRank9AcceleratedMapping.map(raw).unwrap();
                Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(root.get_handle())
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
        assert!(!actual.contains(&fc_data));

        let records = store
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        for input in [a_data, b_data] {
            assert!(records.iter().any(|record| matches!(
                record,
                CollectionRecord::Derive(derive)
                    if derive.collection() == facade.collection() && derive.input() == input
            )));
        }
    }

    #[test]
    fn accelerated_root_without_raw_child_is_not_resident() {
        let raw = raw([row(1, 2, 3)]);
        let root = RawToRank9AcceleratedMapping.map(&raw).unwrap();
        let mut store = MemoryRepo::default();
        store
            .put::<Rank9AcceleratedSuccinctArchiveBlob, _>(root.clone())
            .unwrap();
        let reader = store.reader().unwrap();

        assert!(matches!(
            Rank9AcceleratedSuccinctArchiveBlob::validate_member(
                &Fragment::empty(), &root, &reader,
            ),
            Err(CollectionOperationError::Fatal(reason))
                if reason.contains("raw child is not resident")
        ));
    }

    #[test]
    fn accelerated_member_validation_rejects_a_corrupt_index() {
        let raw = raw([row(1, 2, 3)]);
        let root = RawToRank9AcceleratedMapping.map(&raw).unwrap();
        let mut bytes = root.bytes.as_ref().to_vec();
        let last = bytes.last_mut().expect("Rank9 root is not empty");
        *last ^= 1;
        let corrupted = Blob::<Rank9AcceleratedSuccinctArchiveBlob>::new(Bytes::from_source(bytes));
        let mut store = MemoryRepo::default();
        store.put::<SuccinctArchiveBlob, _>(raw).unwrap();
        let reader = store.reader().unwrap();

        assert!(matches!(
            Rank9AcceleratedSuccinctArchiveBlob::validate_member(
                &Fragment::empty(),
                &corrupted,
                &reader,
            ),
            Err(CollectionOperationError::Fatal(_))
        ));
    }
}
