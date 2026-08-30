//! Explicit size-tiered maintenance for exact derived target collections.
//!
//! Compaction remains separate from cover construction. It starts from an
//! opaque exact cover, publishes only canonical target blobs and unsigned
//! `MERGE` equations, then proves the result again through a fresh attachment.
//! The source cover remains the value boundary, and published equations add
//! neither provenance, retention roots, nor durable validation receipts.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreMeta, OfferCapture};
use crate::trible::Fragment;

use super::exact_derived::{data_identity, ExactDerivedCollection, ExactDerivedCollectionError};
use super::{
    CollectionData, CollectionEncoding, CollectionHandle, CollectionMapping, CollectionMerge,
    CollectionOperationError, CollectionRecord, CollectionStore, Cover, CoverAttachment,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while explicitly compacting one exact target cover.
#[derive(Debug)]
pub enum ExactTargetCompactionError {
    /// Exact-cover completion or fresh attachment failed.
    Exact(ExactDerivedCollectionError),
    /// The target encoding could not join one deterministic pair.
    Merge {
        /// Canonically lower input content identity.
        low: CollectionData,
        /// Canonically higher input content identity.
        high: CollectionData,
        /// Concrete construction failure.
        reason: String,
    },
    /// A storage operation failed.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Backend failure.
        source: BoxError,
    },
    /// Fresh attachment repeated an unstable physical cover.
    Stalled {
        /// Repeated cover in canonical content-handle order.
        cover: Vec<CollectionData>,
    },
}

impl ExactTargetCompactionError {
    fn storage(operation: &'static str, source: impl Error + Send + Sync + 'static) -> Self {
        Self::Storage {
            operation,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for ExactTargetCompactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(f),
            Self::Merge { low, high, reason } => write!(
                f,
                "merge exact target elements {} and {}: {reason}",
                hex::encode_upper(low.raw),
                hex::encode_upper(high.raw),
            ),
            Self::Storage { operation, source } => write!(f, "{operation}: {source}"),
            Self::Stalled { cover } => write!(
                f,
                "exact target compaction repeated an unstable {}-member cover",
                cover.len(),
            ),
        }
    }
}

impl Error for ExactTargetCompactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Storage { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<ExactDerivedCollectionError> for ExactTargetCompactionError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Exact(source)
    }
}

/// Ensure and explicitly compact one exact canonical target cover.
///
/// The fixed policy assigns each canonical target blob to
/// `floor(log2(max(1, serialized_len)))`, then repeatedly joins the two lowest
/// content handles in the lowest colliding tier. A stable cover therefore has
/// at most one physical member per dyadic byte-size tier, except when a fixed
/// representation cannot encode the join or deliberately has no directly
/// materialized join. A capacity-stable cover may retain colliding members:
/// the lower member is retired for that planning round while the higher member
/// remains eligible for the next pair. `Ok(None)` leaves the original cover
/// unchanged. Every capacity-limited attempt shrinks the active set, so a
/// round attempts at most `n - 1` pairs. No policy knobs, manifest, receipt,
/// signed record, retention record, or implicit flush are involved.
///
/// Each maintenance round first stages its complete deterministic carry in
/// memory. Fatal construction errors and rounds with no successful join write
/// nothing. Otherwise, after the attachment reader has been dropped, the target
/// descriptor and every staged result are stored before the topologically
/// ordered `MERGE` records. A fresh read pass then validates the result. Freshly
/// selected covers are compacted again when concurrent or older evidence
/// exposes another collision. Repetition of any non-stable canonical cover
/// returns [`ExactTargetCompactionError::Stalled`].
pub fn compact_exact_target<S, Source, Target, H>(
    exact: &ExactDerivedCollection<Source, Target, H>,
    store: &mut S,
    source_cover: &Cover<Source>,
) -> Result<CoverAttachment<Target>, ExactTargetCompactionError>
where
    S: BlobStore + CollectionStore + ArtifactOfferStore,
    S::Reader: BlobStoreMeta,
    Source: CollectionEncoding,
    Target: CollectionEncoding,
    Handle<Source>: InlineEncoding,
    Handle<Target>: InlineEncoding,
    H: CollectionMapping<Source, Target>,
{
    let mut capture = OfferCapture::new(store);
    compact_exact_target_unoffered(exact, &mut capture, source_cover)
}

pub(crate) fn compact_exact_target_unoffered<S, Source, Target, H>(
    exact: &ExactDerivedCollection<Source, Target, H>,
    store: &mut S,
    source_cover: &Cover<Source>,
) -> Result<CoverAttachment<Target>, ExactTargetCompactionError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
    Source: CollectionEncoding,
    Target: CollectionEncoding,
    Handle<Source>: InlineEncoding,
    Handle<Target>: InlineEncoding,
    H: CollectionMapping<Source, Target>,
{
    let mut cover = exact.ensure_exact_unoffered(store, source_cover)?;
    let mut seen = BTreeSet::new();
    seen.insert(cover_identity(&cover));

    loop {
        if !has_tier_collision(&cover) {
            return Ok(cover);
        }

        match publish_round::<S, Target>(
            exact.target_descriptor().clone(),
            exact.target_collection().handle(),
            store,
            cover,
        )? {
            RoundOutcome::Published => {}
            RoundOutcome::Stable(cover) => return Ok(cover),
        }
        cover = exact.attach_exact(store, source_cover)?;
        let identity = cover_identity(&cover);
        if !seen.insert(identity.clone()) {
            return Err(ExactTargetCompactionError::Stalled { cover: identity });
        }
    }
}

fn target_tier<Target: CollectionEncoding>(blob: &Blob<Target>) -> u32 {
    blob.bytes.len().max(1).ilog2()
}

fn cover_identity<Target: CollectionEncoding>(
    cover: &CoverAttachment<Target>,
) -> Vec<CollectionData> {
    cover
        .members()
        .iter()
        .map(|(data, _)| Handle::<Target>::to_hash(*data))
        .collect()
}

fn has_tier_collision<Target: CollectionEncoding>(cover: &CoverAttachment<Target>) -> bool {
    let mut tiers = BTreeSet::new();
    cover
        .members()
        .iter()
        .any(|(_, blob)| !tiers.insert(target_tier::<Target>(blob)))
}

enum RoundOutcome<Target: CollectionEncoding> {
    Published,
    Stable(CoverAttachment<Target>),
}

fn publish_round<S, Target>(
    descriptor: Fragment,
    collection: CollectionHandle,
    store: &mut S,
    cover: CoverAttachment<Target>,
) -> Result<RoundOutcome<Target>, ExactTargetCompactionError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
    Target: CollectionEncoding,
    Handle<Target>: InlineEncoding,
{
    let reader = store.reader().map_err(|source| {
        ExactTargetCompactionError::storage("open target-compaction reader", source)
    })?;
    let mut tiers = BTreeMap::<u32, BTreeMap<CollectionData, Blob<Target>>>::new();
    let mut locations = BTreeMap::<CollectionData, u32>::new();
    for (data, blob) in cover.members().iter().cloned() {
        let data = Handle::<Target>::to_hash(data);
        let tier = target_tier::<Target>(&blob);
        locations.insert(data, tier);
        tiers.entry(tier).or_default().insert(data, blob);
    }

    let mut outputs = BTreeMap::<CollectionData, Blob<Target>>::new();
    let mut claims = Vec::<CollectionMerge>::new();
    loop {
        let Some(tier) = tiers
            .iter()
            .find_map(|(tier, bin)| (bin.len() >= 2).then_some(*tier))
        else {
            break;
        };
        let mut bin = tiers.remove(&tier).expect("selected tier exists");
        let (low_data, low) = bin.pop_first().expect("colliding tier has a low input");
        let (high_data, high) = bin.pop_first().expect("colliding tier has a high input");
        if !bin.is_empty() {
            tiers.insert(tier, bin);
        }
        locations.remove(&low_data);
        locations.remove(&high_data);

        let constructed = match Target::join_members(&descriptor, &low, &high, &reader) {
            Ok(Some(constructed)) => constructed,
            Ok(None) => return Ok(RoundOutcome::Stable(cover)),
            Err(CollectionOperationError::Fatal(reason)) => {
                return Err(ExactTargetCompactionError::Merge {
                    low: low_data,
                    high: high_data,
                    reason,
                });
            }
            Err(CollectionOperationError::Capacity(_)) => {
                locations.insert(high_data, tier);
                tiers.entry(tier).or_default().insert(high_data, high);
                continue;
            }
        };
        let result_data = data_identity::<Target>(&constructed);
        let result = constructed;
        let claim = CollectionMerge::new(collection, low_data, high_data, result_data);

        if let Some(existing_tier) = locations.remove(&result_data) {
            let existing_bin = tiers
                .get_mut(&existing_tier)
                .expect("located result tier exists");
            existing_bin.remove(&result_data);
            if existing_bin.is_empty() {
                tiers.remove(&existing_tier);
            }
        }
        let result_tier = target_tier::<Target>(&result);
        locations.insert(result_data, result_tier);
        tiers
            .entry(result_tier)
            .or_default()
            .insert(result_data, result.clone());
        outputs.insert(result_data, result);
        claims.push(claim);
    }

    if claims.is_empty() {
        return Ok(RoundOutcome::Stable(cover));
    }

    // Closure-dependent joins observe one immutable reader boundary. Do not
    // retain it across publication.
    drop(reader);

    super::descriptor::put_closure(store, &descriptor)
        .map_err(|error| ExactTargetCompactionError::storage("store target descriptor", error))?;
    for result in outputs.into_values() {
        store.put::<Target, _>(result).map_err(|error| {
            ExactTargetCompactionError::storage("store compacted target", error)
        })?;
    }
    for claim in claims {
        store
            .insert(CollectionRecord::Merge(claim))
            .map_err(|error| ExactTargetCompactionError::storage("publish target MERGE", error))?;
    }
    Ok(RoundOutcome::Published)
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::{BlobEncoding, IntoBlob};
    use crate::collection::{reach, Collection, KIND_COLLECTION_DESCRIPTOR};
    use crate::id::{ExclusiveId, Id};
    use crate::id_hex;
    use crate::inline::Inline;
    use crate::metadata::MetaDescribe;
    use crate::repo::memoryrepo::MemoryRepo;

    /// Test-only encoding with no directly materialized join.
    /// Minted with `trible genid` on 2026-08-30.
    const NO_JOIN_ENCODING_V1: Id = id_hex!("0C6D098C0E9E283EEAD323885B81E784");

    struct NoJoinEncoding;

    impl BlobEncoding for NoJoinEncoding {}

    impl MetaDescribe for NoJoinEncoding {
        fn describe() -> Fragment {
            let id = NO_JOIN_ENCODING_V1;
            crate::macros::entity! { ExclusiveId::force_ref(&id) @
                crate::metadata::name: "exact-target-no-join-test-v1",
                crate::metadata::description: "Test-only collection encoding without a directly materialized join.",
                crate::metadata::tag: crate::metadata::KIND_BLOB_ENCODING,
            }
        }
    }

    impl CollectionEncoding for NoJoinEncoding {
        fn validate_member<R>(
            _descriptor: &Fragment,
            _member: &Blob<Self>,
            _reader: &R,
        ) -> Result<(), CollectionOperationError>
        where
            R: crate::repo::BlobStoreGet + crate::repo::BlobStoreMeta,
        {
            Ok(())
        }
    }

    #[derive(Default)]
    struct NoWriteStore {
        blobs: MemoryRepo,
    }

    impl crate::repo::BlobStorePut for NoWriteStore {
        type PutError = Infallible;

        fn put<S, T>(&mut self, _: T) -> Result<Inline<Handle<S>>, Self::PutError>
        where
            S: BlobEncoding + 'static,
            T: IntoBlob<S>,
            Handle<S>: InlineEncoding,
        {
            panic!("stable no-join compaction attempted a blob write")
        }
    }

    impl BlobStore for NoWriteStore {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = Infallible;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.blobs.reader()
        }
    }

    impl CollectionStore for NoWriteStore {
        type RecordsError = Infallible;
        type InsertError = Infallible;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            Ok(Vec::new().into_iter())
        }

        fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
            panic!("stable no-join compaction attempted to publish a MERGE")
        }
    }

    fn descriptor() -> Fragment {
        crate::macros::entity! {
            crate::metadata::tag: KIND_COLLECTION_DESCRIPTOR,
            crate::collection::collection_name: "exact-target-no-join-test",
            crate::collection::collection_authority: SigningKey::from_bytes(&[7; 32]).verifying_key(),
            crate::collection::collection_representation*: <NoJoinEncoding as MetaDescribe>::describe(),
            crate::collection::collection_reach*: reach::private(),
        }
    }

    #[test]
    fn no_direct_join_leaves_a_colliding_cover_unchanged_without_writes() {
        let descriptor = descriptor();
        let collection = Collection::<NoJoinEncoding>::from_descriptor(&descriptor).unwrap();
        let mut members = vec![
            Blob::<NoJoinEncoding>::new(vec![1; 8].into()),
            Blob::<NoJoinEncoding>::new(vec![2; 8].into()),
        ];
        members.sort_unstable_by_key(|member| member.get_handle().raw);
        let cover =
            Cover::from_members(collection, members.iter().map(|member| member.get_handle()));
        let original = cover_identity(&CoverAttachment::from_parts(
            cover.clone(),
            members
                .iter()
                .cloned()
                .map(|member| (member.get_handle(), member))
                .collect(),
        ));
        let attachment = CoverAttachment::from_parts(
            cover,
            members
                .into_iter()
                .map(|member| (member.get_handle(), member))
                .collect(),
        );

        let result = publish_round(
            descriptor,
            collection.handle(),
            &mut NoWriteStore::default(),
            attachment,
        )
        .unwrap();
        let RoundOutcome::Stable(result) = result else {
            panic!("a no-join encoding published a compaction round");
        };
        assert_eq!(cover_identity(&result), original);
    }
}
