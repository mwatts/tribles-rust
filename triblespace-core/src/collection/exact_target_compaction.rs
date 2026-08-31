//! Deterministic size-tiered maintenance for derived target collections.
//!
//! This is the horizontal half of [`ExactDerivedCollection::ensure`]. It starts
//! from a completed cover and publishes one canonical target carry before
//! re-entering the per-point planner. A later failed or unsupported join does
//! not erase earlier useful work. The source cover remains the value boundary,
//! and yard/GC policy alone decides the lifetime of materialized nodes.

use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use std::collections::{BTreeMap, BTreeSet};

use super::exact_derived::{
    data_identity, ExactDerivedCollection, ExactDerivedCollectionError, ExactPlannerBlocks,
    TargetMergeBlock, TargetMergePair,
};
use super::{
    Collection, CollectionData, CollectionEncoding, CollectionMapping, CollectionMerge,
    CollectionOperationError, CollectionRead, CollectionRecord, CollectionStore, Cover,
};

/// Complete and maintain one canonical target cover.
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
/// Each maintenance round finds at most one successful deterministic carry
/// under one immutable reader. After that reader has been dropped, the result
/// is stored before its `MERGE` record. The per-point planner then observes the
/// fresh equation before another arbitrary dyadic pair can be chosen. Freshly
/// selected covers are carried again when concurrent or older evidence exposes
/// another collision. Repetition of any non-stable canonical cover returns
/// [`ExactDerivedCollectionError::Stalled`].
pub(super) fn ensure_target<S, H>(
    exact: &ExactDerivedCollection<H>,
    store: &mut S,
    source_cover: &Cover<H::Source>,
) -> Result<Cover<H::Target>, ExactDerivedCollectionError>
where
    S: BlobStore + CollectionStore,
    S::Snapshot: BlobStoreMeta + CollectionRead,
    H: CollectionMapping,
    Handle<H::Source>: InlineEncoding,
    Handle<H::Target>: InlineEncoding,
{
    let mut blocks = ExactPlannerBlocks::default();
    let mut cover = exact.complete(store, source_cover, &mut blocks)?;
    let mut seen = BTreeSet::new();
    seen.insert(cover_identity(&cover));

    loop {
        match publish_round::<S, H::Target>(
            exact.target_collection(),
            store,
            cover,
            &blocks.target_merges,
        )? {
            RoundOutcome::Published => {}
            RoundOutcome::Stable(cover) => return Ok(cover),
        }
        // Every successful carry can create an exact child image for a known
        // source-lattice point. Re-enter vertical planning before choosing the
        // next global size-tiered pair.
        cover = exact.complete(store, source_cover, &mut blocks)?;
        let identity = cover_identity(&cover);
        if !seen.insert(identity.clone()) {
            return Err(ExactDerivedCollectionError::Stalled { cover: identity });
        }
    }
}

fn target_tier<Target: CollectionEncoding>(blob: &Blob<Target>) -> u32 {
    blob.bytes.len().max(1).ilog2()
}

fn cover_identity<Target: CollectionEncoding>(cover: &Cover<Target>) -> Vec<CollectionData> {
    cover.members().map(Handle::<Target>::to_hash).collect()
}

enum RoundOutcome<Target: CollectionEncoding> {
    Published,
    Stable(Cover<Target>),
}

fn publish_round<S, Target>(
    collection: Collection<Target>,
    store: &mut S,
    cover: Cover<Target>,
    blocked_target_merges: &BTreeMap<TargetMergePair, TargetMergeBlock>,
) -> Result<RoundOutcome<Target>, ExactDerivedCollectionError>
where
    S: BlobStore + CollectionStore,
    S::Snapshot: BlobStoreMeta + CollectionRead,
    Target: CollectionEncoding,
    Handle<Target>: InlineEncoding,
{
    if cover.len() < 2 {
        return Ok(RoundOutcome::Stable(cover));
    }

    let reader = store.snapshot().map_err(|source| {
        ExactDerivedCollectionError::storage("open target-maintenance snapshot", source)
    })?;
    let descriptor = super::api::load_collection_descriptor(&reader, collection.handle())
        .map_err(|error| {
            ExactDerivedCollectionError::Resolution(format!("load target descriptor: {error}"))
        })?
        .fragment;
    super::encoding::validate_descriptor_type::<Target>(&descriptor).map_err(|error| {
        ExactDerivedCollectionError::Resolution(format!("invalid target descriptor: {error}"))
    })?;
    let mut tiers = BTreeMap::<u32, BTreeMap<CollectionData, Blob<Target>>>::new();
    for handle in cover.members() {
        let data = Handle::<Target>::to_hash(handle);
        let blob = reader.get(handle).map_err(|source| {
            ExactDerivedCollectionError::storage("load target-maintenance member", source)
        })?;
        let tier = target_tier::<Target>(&blob);
        tiers.entry(tier).or_default().insert(data, blob);
    }

    loop {
        let Some(tier) = tiers
            .iter()
            .find_map(|(tier, bin)| (bin.len() >= 2).then_some(*tier))
        else {
            return Ok(RoundOutcome::Stable(cover));
        };
        let mut bin = tiers.remove(&tier).expect("selected tier exists");
        let (low_data, low) = bin.pop_first().expect("colliding tier has a low input");
        let (high_data, high) = bin.pop_first().expect("colliding tier has a high input");
        if !bin.is_empty() {
            tiers.insert(tier, bin);
        }
        match blocked_target_merges.get(&(low_data, high_data)) {
            Some(TargetMergeBlock::Unsupported) => {
                return Ok(RoundOutcome::Stable(cover));
            }
            Some(TargetMergeBlock::Capacity) => {
                tiers.entry(tier).or_default().insert(high_data, high);
                continue;
            }
            None => {}
        }

        let constructed = match Target::join_members(&descriptor, &low, &high, &reader) {
            Ok(Some(constructed)) => constructed,
            Ok(None) => return Ok(RoundOutcome::Stable(cover)),
            Err(CollectionOperationError::Fatal(reason)) => {
                return Err(ExactDerivedCollectionError::Merge {
                    low: low_data,
                    high: high_data,
                    reason,
                });
            }
            Err(CollectionOperationError::Capacity(_)) => {
                tiers.entry(tier).or_default().insert(high_data, high);
                continue;
            }
        };
        let result_data = data_identity::<Target>(&constructed);
        let claim = CollectionMerge::new(collection.handle(), low_data, high_data, result_data);

        // Closure-dependent joins observe one immutable reader boundary. Do
        // not retain it across publication. Publish exactly this carry, then
        // let the caller re-enter per-point planning against a fresh snapshot.
        drop(reader);
        store
            .put::<Target, _>(constructed)
            .map_err(|error| ExactDerivedCollectionError::storage("store merged target", error))?;
        store
            .insert(CollectionRecord::Merge(claim))
            .map_err(|error| ExactDerivedCollectionError::storage("publish target MERGE", error))?;
        return Ok(RoundOutcome::Published);
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::{BlobEncoding, IntoBlob};
    use crate::collection::{CollectionPolicy, CollectionStoreExt};
    use crate::id::{ExclusiveId, Id};
    use crate::id_hex;
    use crate::inline::Inline;
    use crate::metadata::MetaDescribe;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::BlobStorePut;
    use crate::trible::Fragment;

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

    impl crate::repo::SnapshotSource for NoWriteStore {
        type Snapshot = <MemoryRepo as crate::repo::SnapshotSource>::Snapshot;
        type SnapshotError = Infallible;

        fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
            crate::repo::SnapshotSource::snapshot(&mut self.blobs)
        }
    }

    impl CollectionStore for NoWriteStore {
        type InsertError = Infallible;

        fn insert(&mut self, _: CollectionRecord) -> Result<(), Self::InsertError> {
            panic!("stable no-join compaction attempted to publish a MERGE")
        }
    }

    fn descriptor() -> Fragment {
        crate::collection::descriptor::naming::<NoJoinEncoding>(
            "exact-target-no-join-test",
            CollectionPolicy::new(
                crate::collection::AdmissionPolicy::direct(
                    SigningKey::from_bytes(&[7; 32]).verifying_key(),
                ),
                crate::collection::AdmissionPolicy::direct(
                    SigningKey::from_bytes(&[7; 32]).verifying_key(),
                ),
            ),
        )
    }

    #[test]
    fn no_direct_join_leaves_a_colliding_cover_unchanged_without_writes() {
        let descriptor = descriptor();
        let mut blobs = MemoryRepo::default();
        let collection = blobs
            .register_collection::<NoJoinEncoding>(descriptor)
            .unwrap();
        let mut members = vec![
            Blob::<NoJoinEncoding>::new(vec![1; 8].into()),
            Blob::<NoJoinEncoding>::new(vec![2; 8].into()),
        ];
        members.sort_unstable_by_key(|member| member.get_handle().raw);
        let cover =
            Cover::from_members(collection, members.iter().map(|member| member.get_handle()));
        let original = cover_identity(&cover);
        for member in members {
            blobs.put::<NoJoinEncoding, _>(member).unwrap();
        }

        let result = publish_round(
            collection,
            &mut NoWriteStore { blobs },
            cover,
            &BTreeMap::new(),
        )
        .unwrap();
        let RoundOutcome::Stable(result) = result else {
            panic!("a no-join encoding published a compaction round");
        };
        assert_eq!(cover_identity(&result), original);
    }
}
