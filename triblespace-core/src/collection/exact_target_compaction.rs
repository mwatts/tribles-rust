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

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreMeta, BlobStorePut, OfferCapture};
use crate::trible::Fragment;

use super::exact_derived::{
    fresh_data_identity, ExactDerivedCollection, ExactDerivedCollectionError,
};
use super::{
    CollectionData, CollectionHandle, CollectionHomomorphism, CollectionLattice,
    CollectionLatticeError, CollectionMerge, CollectionRecord, CollectionStore, Cover,
    CoverAttachment,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while explicitly compacting one exact target cover.
#[derive(Debug)]
pub enum ExactTargetCompactionError {
    /// Exact-cover completion or fresh attachment failed.
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
        expected: CollectionHandle,
        /// Handle returned by the blob store.
        actual: CollectionHandle,
    },
    /// The blob store returned another handle for a freshly hashed target result.
    NonCanonicalTargetPut {
        /// Target identity computed from canonical bytes.
        expected: CollectionData,
        /// Identity returned by the blob store.
        actual: CollectionData,
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
/// at most one physical member per dyadic byte-size tier, except when a fixed
/// representation cannot encode the join. Such a capacity-stable cover may
/// retain colliding members: the lower member is retired for that planning
/// round while the higher member remains eligible for the next pair. Every
/// attempt shrinks the active set, so a round attempts at most `n - 1` pairs.
/// No policy knobs, manifest, receipt, signed record, retention record, or
/// implicit flush are involved.
///
/// Each maintenance round first stages its complete deterministic carry in
/// memory. Fatal construction errors and rounds with no successful join write
/// nothing. Otherwise, after the attachment reader has been dropped, the target
/// descriptor and every staged result are stored before the topologically
/// ordered `MERGE` records. A fresh read pass then validates the result. Freshly
/// selected covers are compacted again when concurrent or older evidence
/// exposes another collision. Repetition of any non-capacity-stable canonical
/// cover returns [`ExactTargetCompactionError::Stalled`].
pub fn compact_exact_target<S, Source, Target, H>(
    exact: &ExactDerivedCollection<Source, Target, H>,
    store: &mut S,
    source_cover: &Cover<Source>,
) -> Result<CoverAttachment<Target>, ExactTargetCompactionError>
where
    S: BlobStore + CollectionStore + ArtifactOfferStore,
    S::Reader: BlobStoreMeta,
    Source: CollectionLattice,
    Target: CollectionLattice,
    Handle<Source::Encoding>: InlineEncoding,
    Handle<Target::Encoding>: InlineEncoding,
    H: CollectionHomomorphism<Source, Target>,
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
    Source: CollectionLattice,
    Target: CollectionLattice,
    Handle<Source::Encoding>: InlineEncoding,
    Handle<Target::Encoding>: InlineEncoding,
    H: CollectionHomomorphism<Source, Target>,
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
            RoundOutcome::CapacityStable(cover) => return Ok(cover),
        }
        cover = exact.attach_exact(store, source_cover)?;
        let identity = cover_identity(&cover);
        if !seen.insert(identity.clone()) {
            return Err(ExactTargetCompactionError::Stalled { cover: identity });
        }
    }
}

fn target_tier<Target: CollectionLattice>(blob: &Blob<Target::Encoding>) -> u32 {
    blob.bytes.len().max(1).ilog2()
}

fn cover_identity<Target: CollectionLattice>(
    cover: &CoverAttachment<Target>,
) -> Vec<CollectionData> {
    cover
        .members()
        .iter()
        .map(|(data, _)| Handle::<Target::Encoding>::to_hash(*data))
        .collect()
}

fn has_tier_collision<Target: CollectionLattice>(cover: &CoverAttachment<Target>) -> bool {
    let mut tiers = BTreeSet::new();
    cover
        .members()
        .iter()
        .any(|(_, blob)| !tiers.insert(target_tier::<Target>(blob)))
}

enum RoundOutcome<Target: CollectionLattice> {
    Published,
    CapacityStable(CoverAttachment<Target>),
}

fn publish_round<S, Target>(
    descriptor: Fragment,
    collection: CollectionHandle,
    store: &mut S,
    cover: CoverAttachment<Target>,
) -> Result<RoundOutcome<Target>, ExactTargetCompactionError>
where
    S: BlobStorePut + CollectionStore,
    Target: CollectionLattice,
    Handle<Target::Encoding>: InlineEncoding,
{
    let mut tiers = BTreeMap::<u32, BTreeMap<CollectionData, Blob<Target::Encoding>>>::new();
    let mut locations = BTreeMap::<CollectionData, u32>::new();
    for (data, blob) in cover.members().iter().cloned() {
        let data = Handle::<Target::Encoding>::to_hash(data);
        let tier = target_tier::<Target>(&blob);
        locations.insert(data, tier);
        tiers.entry(tier).or_default().insert(data, blob);
    }

    let mut outputs = BTreeMap::<CollectionData, Blob<Target::Encoding>>::new();
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

        let constructed = match Target::merge_members(&descriptor, &low, &high) {
            Ok(constructed) => constructed,
            Err(CollectionLatticeError::Fatal(reason)) => {
                return Err(ExactTargetCompactionError::Merge {
                    low: low_data,
                    high: high_data,
                    reason,
                });
            }
            Err(CollectionLatticeError::Capacity(_)) => {
                locations.insert(high_data, tier);
                tiers.entry(tier).or_default().insert(high_data, high);
                continue;
            }
        };
        let result_data = fresh_data_identity(&constructed);
        let result = Blob::with_handle(
            constructed.bytes,
            Handle::<Target::Encoding>::from_hash(result_data),
        );
        match Target::validate_member(&descriptor, &result) {
            Ok(()) => {}
            Err(CollectionLatticeError::Fatal(reason)) => {
                return Err(ExactTargetCompactionError::InvalidResult {
                    low: low_data,
                    high: high_data,
                    result: result_data,
                    reason,
                });
            }
            Err(CollectionLatticeError::Capacity(_)) => {
                locations.insert(high_data, tier);
                tiers.entry(tier).or_default().insert(high_data, high);
                continue;
            }
        }
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
        return Ok(RoundOutcome::CapacityStable(cover));
    }

    let actual_descriptor = store
        .put::<SimpleArchive, _>(crate::blob::IntoBlob::<SimpleArchive>::to_blob(
            descriptor.into_facts(),
        ))
        .map_err(|error| ExactTargetCompactionError::storage("store target descriptor", error))?;
    if actual_descriptor != collection {
        return Err(ExactTargetCompactionError::NonCanonicalDescriptorPut {
            expected: collection,
            actual: actual_descriptor,
        });
    }
    for (expected, result) in outputs {
        let actual = store.put::<Target::Encoding, _>(result).map_err(|error| {
            ExactTargetCompactionError::storage("store compacted target", error)
        })?;
        let actual = Handle::<Target::Encoding>::to_hash(actual);
        if actual != expected {
            return Err(ExactTargetCompactionError::NonCanonicalTargetPut { expected, actual });
        }
    }
    for claim in claims {
        store
            .insert(CollectionRecord::Merge(claim))
            .map_err(|error| ExactTargetCompactionError::storage("publish target MERGE", error))?;
    }
    Ok(RoundOutcome::Published)
}
