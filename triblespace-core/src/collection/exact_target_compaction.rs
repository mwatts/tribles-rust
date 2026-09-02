//! Deterministic horizontal LSM maintenance in one collection lattice.
//!
//! Vertical realization is complete before this module runs.  Carries only
//! join target members and publish target `MERGE` equations.  A capacity limit
//! or a missing optional join dependency keeps the finer cover; it never
//! triggers construction in an upstream lattice.

use std::collections::{BTreeMap, BTreeSet};

use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::repo::{BlobStoreGet, Store};

use super::exact_derived::{attach_collection, data_identity, CollectionRealizationError};
use super::{
    Collection, CollectionData, CollectionEncoding, CollectionMerge, CollectionOperationError,
    CollectionRecord, Cover, Support,
};

/// Carry one exact target realization to its deterministic dyadic LSM fixed
/// point.
pub(super) fn maintain_target<S, E>(
    store: &mut S,
    target: Collection<E>,
    support: &Support,
) -> Result<(), CollectionRealizationError>
where
    S: Store,
    E: CollectionEncoding,
{
    let mut blocked = BTreeSet::new();
    let mut seen = BTreeSet::new();

    loop {
        let snapshot = store.snapshot().map_err(|error| {
            CollectionRealizationError::storage("open target-maintenance snapshot", error)
        })?;
        let (_, cover) = attach_collection(&snapshot, target, Some(support))?;
        let identity = cover_identity(&cover);
        if !seen.insert(identity.clone()) {
            return Err(CollectionRealizationError::Stalled { cover: identity });
        }
        let prepared = prepare_next_carry(&snapshot, target, &cover, &mut blocked)?;
        drop(snapshot);

        let Some((low, high, output)) = prepared else {
            return Ok(());
        };
        let result = data_identity::<E>(&output);
        store.put::<E, _>(output).map_err(|error| {
            CollectionRealizationError::storage("store merged target member", error)
        })?;
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                target.handle(),
                low,
                high,
                result,
            )))
            .map_err(|error| CollectionRealizationError::storage("publish target MERGE", error))?;
    }
}

fn tier<E: CollectionEncoding>(blob: &Blob<E>) -> u32 {
    blob.bytes.len().max(1).ilog2()
}

fn cover_identity<E: CollectionEncoding>(cover: &Cover<E>) -> Vec<CollectionData> {
    cover.data_members().collect()
}

fn prepare_next_carry<R, E>(
    snapshot: &R,
    target: Collection<E>,
    cover: &Cover<E>,
    blocked: &mut BTreeSet<(CollectionData, CollectionData)>,
) -> Result<Option<(CollectionData, CollectionData, Blob<E>)>, CollectionRealizationError>
where
    R: BlobStoreGet + crate::repo::BlobStoreMeta,
    E: CollectionEncoding,
{
    if cover.len() < 2 {
        return Ok(None);
    }
    let descriptor = super::api::load_collection_descriptor(snapshot, target.handle())
        .map_err(|error| {
            CollectionRealizationError::Resolution(format!(
                "load target descriptor for maintenance: {error}"
            ))
        })?
        .fragment;
    super::encoding::validate_descriptor_type::<E>(&descriptor).map_err(|error| {
        CollectionRealizationError::Resolution(format!(
            "invalid target descriptor for maintenance: {error}"
        ))
    })?;

    let mut tiers = BTreeMap::<u32, BTreeMap<CollectionData, Blob<E>>>::new();
    for handle in cover.members() {
        let data = Handle::<E>::to_hash(handle);
        let blob = snapshot.get(handle).map_err(|error| {
            CollectionRealizationError::storage("load target-maintenance member", error)
        })?;
        tiers.entry(tier(&blob)).or_default().insert(data, blob);
    }

    for (_, mut members) in tiers {
        while members.len() >= 2 {
            let (low_data, low) = members
                .pop_first()
                .expect("colliding target tier contains a lower member");
            let (high_data, high) = members
                .pop_first()
                .expect("colliding target tier contains a higher member");
            if blocked.contains(&(low_data, high_data)) {
                members.insert(high_data, high);
                continue;
            }
            match E::join_members(&descriptor, &low, &high, snapshot) {
                Ok(output) => return Ok(Some((low_data, high_data, output))),
                Err(CollectionOperationError::Fatal(reason)) => {
                    return Err(CollectionRealizationError::Merge {
                        low: low_data,
                        high: high_data,
                        reason,
                    });
                }
                Err(CollectionOperationError::Capacity(_))
                | Err(CollectionOperationError::MissingDependency(_)) => {
                    // Retire the lower input for this planning pass and leave
                    // the higher one eligible for the next deterministic pair.
                    // The exact finer cover remains the valid result.
                    blocked.insert((low_data, high_data));
                    members.insert(high_data, high);
                }
            }
        }
    }
    Ok(None)
}
