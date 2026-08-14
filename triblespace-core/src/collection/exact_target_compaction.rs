//! Explicit size-tiered maintenance for exact derived target collections.
//!
//! Compaction remains separate from exact-ticket admission. It starts from an
//! admitted exact cover, publishes only canonical target blobs and unsigned
//! `MERGE` equations, then proves the result again through a fresh attachment.
//! Signed source commits remain the sole authority, and published equations add
//! neither retention roots nor durable validation receipts.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::{Blob, BlobEncoding};
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{BlobStore, BlobStoreMeta, BlobStorePut};

use super::exact_derived::{
    fresh_data_identity, ExactCover, ExactDerivedAlgebra, ExactDerivedCollection,
    ExactDerivedCollectionError,
};
use super::{
    CollectionCommit, CollectionData, CollectionDescriptor, CollectionId, CollectionMerge,
    CollectionRecord, CollectionStore,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while explicitly compacting one exact target cover.
#[derive(Debug)]
pub enum ExactTargetCompactionError {
    /// Exact-ticket completion or fresh admission failed.
    Exact(ExactDerivedCollectionError),
    /// The target algebra could not join one deterministic pair.
    Merge {
        /// Canonically lower input content identity.
        low: CollectionData,
        /// Canonically higher input content identity.
        high: CollectionData,
        /// Concrete construction failure.
        reason: String,
    },
    /// A freshly constructed target did not validate under the fixed descriptor.
    InvalidResult {
        /// Canonically lower input content identity.
        low: CollectionData,
        /// Canonically higher input content identity.
        high: CollectionData,
        /// Fresh result content identity.
        result: CollectionData,
        /// Concrete validation failure.
        reason: String,
    },
    /// A storage operation failed.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Backend failure.
        source: BoxError,
    },
    /// The blob store returned another handle for the canonical target descriptor.
    NonCanonicalDescriptorPut {
        /// Descriptor handle computed from canonical bytes.
        expected: CollectionId,
        /// Handle returned by the blob store.
        actual: CollectionId,
    },
    /// The blob store returned another handle for a freshly hashed target result.
    NonCanonicalTargetPut {
        /// Target identity computed from canonical bytes.
        expected: CollectionData,
        /// Identity returned by the blob store.
        actual: CollectionData,
    },
    /// Fresh admission repeated an unstable physical cover.
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
            Self::InvalidResult {
                low,
                high,
                result,
                reason,
            } => write!(
                f,
                "merge exact target elements {} and {} produced invalid result {}: {reason}",
                hex::encode_upper(low.raw),
                hex::encode_upper(high.raw),
                hex::encode_upper(result.raw),
            ),
            Self::Storage { operation, source } => write!(f, "{operation}: {source}"),
            Self::NonCanonicalDescriptorPut { expected, actual } => write!(
                f,
                "blob store returned descriptor handle {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::NonCanonicalTargetPut { expected, actual } => write!(
                f,
                "blob store returned target handle {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
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
/// at most one physical member per dyadic byte-size tier. No policy knobs,
/// manifest, receipt, signed record, retention record, or implicit flush are
/// involved.
///
/// Each maintenance round drops the admission reader, stores the target
/// descriptor, and computes its deterministic carry while storing each result.
/// Only after the full carry succeeds are its topologically ordered `MERGE`
/// records inserted, so every result blob precedes the first equation. A fresh
/// read pass then admits the result. Freshly selected covers are compacted again
/// when concurrent or older evidence exposes another collision. Repetition of
/// any unstable canonical cover returns [`ExactTargetCompactionError::Stalled`].
pub fn compact_exact_target<S, Source, Target, A>(
    exact: &ExactDerivedCollection<Source, Target>,
    store: &mut S,
    ticket: &[CollectionCommit],
    algebra: &A,
) -> Result<ExactCover<Target>, ExactTargetCompactionError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
    Source: BlobEncoding + 'static,
    Target: BlobEncoding + 'static,
    Handle<Source>: InlineEncoding,
    Handle<Target>: InlineEncoding,
    A: ExactDerivedAlgebra<Source, Target> + ?Sized,
{
    let mut cover = exact.ensure_exact(store, ticket, algebra)?;
    let mut seen = BTreeSet::new();
    seen.insert(cover_identity(&cover));

    loop {
        if !has_tier_collision(&cover) {
            return Ok(cover);
        }

        publish_round(exact.target_descriptor(), store, cover, algebra)?;
        cover = exact.attach_exact(store, ticket, algebra)?;
        let identity = cover_identity(&cover);
        if !seen.insert(identity.clone()) {
            return Err(ExactTargetCompactionError::Stalled { cover: identity });
        }
    }
}

fn target_tier<Target: BlobEncoding>(blob: &Blob<Target>) -> u32 {
    blob.bytes.len().max(1).ilog2()
}

fn cover_identity<Target: BlobEncoding>(cover: &ExactCover<Target>) -> Vec<CollectionData> {
    cover.members().iter().map(|(data, _)| *data).collect()
}

fn has_tier_collision<Target: BlobEncoding>(cover: &ExactCover<Target>) -> bool {
    let mut tiers = BTreeSet::new();
    cover
        .members()
        .iter()
        .any(|(_, blob)| !tiers.insert(target_tier(blob)))
}

fn publish_round<S, Source, Target, A>(
    descriptor: CollectionDescriptor,
    store: &mut S,
    cover: ExactCover<Target>,
    algebra: &A,
) -> Result<(), ExactTargetCompactionError>
where
    S: BlobStorePut + CollectionStore,
    Source: BlobEncoding,
    Target: BlobEncoding + 'static,
    Handle<Target>: InlineEncoding,
    A: ExactDerivedAlgebra<Source, Target> + ?Sized,
{
    let expected_descriptor = descriptor.handle();
    let actual_descriptor = store
        .put::<SimpleArchive, _>(CollectionDescriptor::to_blob(&descriptor))
        .map_err(|error| ExactTargetCompactionError::storage("store target descriptor", error))?;
    if actual_descriptor != expected_descriptor {
        return Err(ExactTargetCompactionError::NonCanonicalDescriptorPut {
            expected: expected_descriptor,
            actual: actual_descriptor,
        });
    }

    let mut tiers = BTreeMap::<u32, BTreeMap<CollectionData, Blob<Target>>>::new();
    let mut locations = BTreeMap::<CollectionData, u32>::new();
    for (data, blob) in cover.into_members() {
        let tier = target_tier(&blob);
        locations.insert(data, tier);
        tiers.entry(tier).or_default().insert(data, blob);
    }

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

        let constructed = algebra.join_target(&low, &high).map_err(|reason| {
            ExactTargetCompactionError::Merge {
                low: low_data,
                high: high_data,
                reason,
            }
        })?;
        let result_data = fresh_data_identity(&constructed);
        let result = Blob::with_handle(constructed.bytes, Handle::<Target>::from_hash(result_data));
        algebra
            .validate_target(&descriptor, &result)
            .map_err(|reason| ExactTargetCompactionError::InvalidResult {
                low: low_data,
                high: high_data,
                result: result_data,
                reason,
            })?;
        let claim = CollectionMerge::new(descriptor.handle(), low_data, high_data, result_data);

        let actual = store.put::<Target, _>(result.clone()).map_err(|error| {
            ExactTargetCompactionError::storage("store compacted target", error)
        })?;
        let actual = Handle::<Target>::to_hash(actual);
        if actual != result_data {
            return Err(ExactTargetCompactionError::NonCanonicalTargetPut {
                expected: result_data,
                actual,
            });
        }

        if let Some(existing_tier) = locations.remove(&result_data) {
            let existing_bin = tiers
                .get_mut(&existing_tier)
                .expect("located result tier exists");
            existing_bin.remove(&result_data);
            if existing_bin.is_empty() {
                tiers.remove(&existing_tier);
            }
        }
        let result_tier = target_tier(&result);
        locations.insert(result_data, result_tier);
        tiers
            .entry(result_tier)
            .or_default()
            .insert(result_data, result);
        claims.push(claim);
    }

    for claim in claims {
        store
            .insert(CollectionRecord::Merge(claim))
            .map_err(|error| ExactTargetCompactionError::storage("publish target MERGE", error))?;
    }
    Ok(())
}
