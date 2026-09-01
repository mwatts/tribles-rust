//! Deterministic size-tiered maintenance for derived target collections.
//!
//! This is the horizontal half of [`ExactDerivedCollection::ensure`]. It starts
//! from a completed cover and publishes one canonical target carry before
//! re-entering the per-point planner. A later failed or capacity-limited join does
//! not erase earlier useful work. The source cover remains the value boundary,
//! and yard/GC policy alone decides the lifetime of materialized nodes.

use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use std::collections::{BTreeMap, BTreeSet};

use super::exact_derived::{
    data_identity, ExactDerivedCollection, ExactDerivedCollectionError, ExactPlannerBlocks,
    TargetMergePair,
};
use super::{
    CollectionData, CollectionEncoding, CollectionMapping, CollectionMerge,
    CollectionOperationError, CollectionRead, CollectionRecord, CollectionStore, Cover,
};

/// Complete and maintain one canonical target cover.
///
/// The fixed policy assigns each canonical target blob to
/// `floor(log2(max(1, serialized_len)))`, then repeatedly joins the two lowest
/// content handles in the lowest colliding tier. A stable cover therefore has
/// at most one physical member per dyadic byte-size tier, except when a fixed
/// representation cannot encode the join. A capacity-stable cover may retain colliding members:
/// the lower member is retired for that planning round while the higher member
/// remains eligible for the next pair. Every capacity-limited attempt shrinks the active set, so a
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
        match publish_round::<S, H>(exact, store, source_cover, cover, &mut blocks.target_merges)? {
            RoundOutcome::TargetPublished => {}
            RoundOutcome::Retry(current) => {
                cover = current;
                continue;
            }
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
    TargetPublished,
    Retry(Cover<Target>),
    Stable(Cover<Target>),
}

fn publish_round<S, Mapping>(
    exact: &ExactDerivedCollection<Mapping>,
    store: &mut S,
    source_cover: &Cover<Mapping::Source>,
    cover: Cover<Mapping::Target>,
    blocked_target_merges: &mut BTreeSet<TargetMergePair>,
) -> Result<RoundOutcome<Mapping::Target>, ExactDerivedCollectionError>
where
    S: BlobStore + CollectionStore,
    S::Snapshot: BlobStoreMeta + CollectionRead,
    Mapping: CollectionMapping,
    Handle<Mapping::Source>: InlineEncoding,
    Handle<Mapping::Target>: InlineEncoding,
{
    if cover.len() < 2 {
        return Ok(RoundOutcome::Stable(cover));
    }

    let collection = exact.target_collection();
    let reader = store.snapshot().map_err(|source| {
        ExactDerivedCollectionError::storage("open target-maintenance snapshot", source)
    })?;
    let descriptor = super::api::load_collection_descriptor(&reader, collection.handle())
        .map_err(|error| {
            ExactDerivedCollectionError::Resolution(format!("load target descriptor: {error}"))
        })?
        .fragment;
    super::encoding::validate_descriptor_type::<Mapping::Target>(&descriptor).map_err(|error| {
        ExactDerivedCollectionError::Resolution(format!("invalid target descriptor: {error}"))
    })?;
    let mut tiers = BTreeMap::<u32, BTreeMap<CollectionData, Blob<Mapping::Target>>>::new();
    for handle in cover.members() {
        let data = Handle::<Mapping::Target>::to_hash(handle);
        let blob = reader.get(handle).map_err(|source| {
            ExactDerivedCollectionError::storage("load target-maintenance member", source)
        })?;
        let tier = target_tier::<Mapping::Target>(&blob);
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
        if blocked_target_merges.contains(&(low_data, high_data)) {
            tiers.entry(tier).or_default().insert(high_data, high);
            continue;
        }

        let constructed = match <Mapping::Target as CollectionEncoding>::join_members(
            &descriptor,
            &low,
            &high,
            &reader,
        ) {
            Ok(constructed) => constructed,
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
            Err(CollectionOperationError::MissingDependency(member)) => {
                drop(reader);
                if exact.materialize_target_join_dependency(
                    store,
                    source_cover,
                    low_data,
                    high_data,
                    member,
                )? {
                    return Ok(RoundOutcome::Retry(cover));
                }
                return Err(ExactDerivedCollectionError::MissingDependency { member });
            }
        };
        let result_data = data_identity::<Mapping::Target>(&constructed);
        let claim = CollectionMerge::new(collection.handle(), low_data, high_data, result_data);

        // Closure-dependent joins observe one immutable reader boundary. Do
        // not retain it across publication. Publish exactly this carry, then
        // let the caller re-enter per-point planning against a fresh snapshot.
        drop(reader);
        store
            .put::<Mapping::Target, _>(constructed)
            .map_err(|error| ExactDerivedCollectionError::storage("store merged target", error))?;
        store
            .insert(CollectionRecord::Merge(claim))
            .map_err(|error| ExactDerivedCollectionError::storage("publish target MERGE", error))?;
        return Ok(RoundOutcome::TargetPublished);
    }
}
