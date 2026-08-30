//! Lattice-aware exact collection attachment over bearer artifact lookup.
//!
//! The network layer never plans collection semantics. It freezes the target
//! artifact identities named by the already-converged record inventory, lets
//! the core exact resolver choose a canonical physical cover, and fetches only
//! those selected members. Stale offers are removed monotonically and the
//! resolver replans. If no advertised physical cover is available, ordinary
//! local exact construction remains the final oracle.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use anybytes::Bytes;
use triblespace_core::blob::Blob;
use triblespace_core::collection::exact_derived::{
    ExactAttachPlan, ExactDerivedCollection, ExactDerivedCollectionError,
};
use triblespace_core::collection::{
    CollectionData, CollectionMapping, CollectionRead, CollectionRecord, CollectionRecordSelector,
    CollectionStore, Cover,
};
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::{
    BlobChildren, BlobStore, CapabilityProofStore, SnapshotSource, StorageFlush, StoreRead,
    WantStore,
};

use crate::peer::Peer;

type BoxError = Box<dyn Error + Send + Sync + 'static>;

fn remaining_fetch_budget(deadline: tokio::time::Instant) -> Option<std::time::Duration> {
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    (!remaining.is_zero()).then_some(remaining)
}

/// Failure while obtaining one exact derived cover from local or remote state.
#[derive(Debug)]
pub enum ExactDerivedSyncError {
    /// Exact-cover resolution, validation, or construction failed.
    Exact(ExactDerivedCollectionError),
    /// Reading or landing physical evidence failed.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Backend failure.
        source: BoxError,
    },
    /// A store returned a content identity different from the verified bytes.
    LandingIdentity {
        /// Exact target identity requested from the team.
        expected: CollectionData,
        /// Identity reported by the local store.
        actual: CollectionData,
    },
}

impl ExactDerivedSyncError {
    fn storage(operation: &'static str, source: impl Error + Send + Sync + 'static) -> Self {
        Self::Storage {
            operation,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for ExactDerivedSyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(source) => source.fmt(f),
            Self::Storage { operation, source } => write!(f, "{operation}: {source}"),
            Self::LandingIdentity { expected, actual } => write!(
                f,
                "landing fetched target returned {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
        }
    }
}

impl Error for ExactDerivedSyncError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Exact(source) => Some(source),
            Self::Storage { source, .. } => Some(source.as_ref()),
            Self::LandingIdentity { .. } => None,
        }
    }
}

impl From<ExactDerivedCollectionError> for ExactDerivedSyncError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Exact(source)
    }
}

/// Ensure one exact derived cover, opportunistically reusing a remote peer's
/// physical cover before reconstructing missing target artifacts locally.
///
/// The source cover and all source dependencies must already be resident; a
/// remote artifact cannot substitute for an explicit source-cover member. Target
/// offers are frozen from one local record view, are never treated as proof of
/// residency, and shrink after every attempted fetch. The operation therefore
/// terminates even when every offer is stale. Exact fetches neither create
/// durable [`WantStore`] entries nor change collection authority.
pub async fn ensure_exact_derived<S, Mapping>(
    peer: &mut Peer<S>,
    lifecycle: &ExactDerivedCollection<Mapping>,
    source_cover: &Cover<Mapping::Source>,
) -> Result<Cover<Mapping::Target>, ExactDerivedSyncError>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + WantStore
        + StorageFlush
        + Send
        + 'static,
    S::Snapshot: StoreRead + BlobChildren,
    Mapping: CollectionMapping,
    Handle<Mapping::Source>: InlineEncoding,
    Handle<Mapping::Target>: InlineEncoding,
{
    if source_cover.is_empty() {
        return lifecycle
            .attach_exact(&mut *peer.store(), source_cover)
            .map_err(Into::into);
    }

    // Admit any already-arrived record inventory exactly once, then freeze the
    // speculative offer set. Moving inventory must not create a retry loop.
    peer.try_refresh().map_err(|error| {
        ExactDerivedSyncError::storage("refresh exact-derived network store", error)
    })?;
    let mut offered = {
        let snapshot = peer.snapshot().map_err(|error| {
            ExactDerivedSyncError::storage("freeze exact-derived store snapshot", error)
        })?;
        let selectors = BTreeSet::from([
            CollectionRecordSelector::MergeCollection(lifecycle.target_collection().handle()),
            CollectionRecordSelector::DeriveTarget(lifecycle.target_collection().handle()),
        ]);
        snapshot
            .select_records(&selectors)
            .map_err(|error| {
                ExactDerivedSyncError::storage("enumerate target artifact offers", error)
            })?
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Commit(_) => None,
                CollectionRecord::Merge(merge) => Some(merge.result()),
                CollectionRecord::Derive(derive) => Some(derive.output()),
            })
            .map(Handle::<Mapping::Target>::from_hash)
            .collect::<BTreeSet<_>>()
    };

    // Speculative reuse is one interactive operation, not one independent
    // operation per selected cover member. A malicious or merely stale cover
    // must not multiply the end-to-end network deadline by its width.
    let fetch_deadline = tokio::time::Instant::now() + crate::host::INTERACTIVE_FETCH_DEADLINE;

    'replan: loop {
        let plan = {
            let mut store = peer.store();
            lifecycle.probe_exact(&mut *store, source_cover, &offered)
        };
        match plan {
            Ok(ExactAttachPlan::Ready(cover)) => return Ok(cover),
            Ok(ExactAttachPlan::Fetch(handles)) => {
                debug_assert!(!handles.is_empty(), "a fetch plan contains work");
                for expected in handles {
                    let Some(remaining) = remaining_fetch_budget(fetch_deadline) else {
                        // Re-probe without speculative members so already
                        // landed bytes remain reusable before local fallback.
                        offered.clear();
                        continue 'replan;
                    };
                    let Some(raw) = peer.fetch_blob_with_deadline(expected.raw, remaining).await
                    else {
                        offered.remove(&expected);
                        continue 'replan;
                    };
                    let blob = Blob::<Mapping::Target>::new(Bytes::from(raw));
                    let actual = blob.get_handle();
                    if actual != expected {
                        // `fetch_blob` already checks this. Keep the boundary
                        // explicit so a future transport cannot weaken it.
                        offered.remove(&expected);
                        continue 'replan;
                    }
                    let landed = peer
                        .store()
                        .put::<Mapping::Target, _>(blob)
                        .map_err(|error| {
                            ExactDerivedSyncError::storage("land fetched target artifact", error)
                        })?;
                    if landed != expected {
                        return Err(ExactDerivedSyncError::LandingIdentity {
                            expected: Handle::<Mapping::Target>::to_hash(expected),
                            actual: Handle::<Mapping::Target>::to_hash(landed),
                        });
                    }
                    // Local residency, freshly re-read by the next probe, now
                    // carries this member. Removing every attempted hint makes
                    // progress explicit even for an unusual lossy store.
                    offered.remove(&expected);
                }
            }
            Err(ExactDerivedCollectionError::IncompleteCover { .. }) => {
                let mut store = peer.store();
                return lifecycle
                    .ensure_exact(&mut *store, source_cover)
                    .map_err(Into::into);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn speculative_cover_members_share_one_absolute_fetch_budget() {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        assert_eq!(
            remaining_fetch_budget(deadline),
            Some(std::time::Duration::from_secs(10)),
        );

        tokio::time::advance(std::time::Duration::from_secs(7)).await;
        assert_eq!(
            remaining_fetch_budget(deadline),
            Some(std::time::Duration::from_secs(3)),
            "a later cover member receives only the first member's remainder",
        );

        tokio::time::advance(std::time::Duration::from_secs(3)).await;
        assert_eq!(
            remaining_fetch_budget(deadline),
            None,
            "cover width cannot renew the operation-wide deadline",
        );
    }
}
