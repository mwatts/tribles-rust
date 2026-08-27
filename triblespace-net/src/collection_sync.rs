//! Lattice-aware exact collection attachment over the generic team transport.
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
use triblespace_core::blob::{Blob, BlobEncoding};
use triblespace_core::collection::exact_derived::{
    ExactAttachPlan, ExactCover, ExactDerivedAlgebra, ExactDerivedCollection,
    ExactDerivedCollectionError,
};
use triblespace_core::collection::{
    CollectionCommit, CollectionData, CollectionRecord, CollectionRecordSelector, CollectionStore,
};
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::{
    ArtifactOfferStore, BlobStore, BlobStoreMeta, CapabilityProofStore, OfferCapture, PeerStore,
    StorageFlush, StoreRevision, StoreScope, WantStore,
};

use crate::peer::Peer;

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while obtaining one exact derived cover from local or team state.
#[derive(Debug)]
pub enum ExactDerivedSyncError {
    /// Exact-ticket authority, resolution, validation, or construction failed.
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

/// Ensure one exact derived ticket, opportunistically reusing a team member's
/// physical cover before reconstructing missing target artifacts locally.
///
/// The source ticket and all source dependencies must already be resident; a
/// remote artifact cannot substitute for signed source authority. Target
/// offers are frozen from one local record view, are never treated as proof of
/// residency, and shrink after every attempted fetch. The operation therefore
/// terminates even when every offer is stale. Exact fetches neither create
/// durable [`WantStore`] entries nor change collection authority.
pub async fn ensure_exact_derived<S, Source, Target, A>(
    peer: &mut Peer<S>,
    lifecycle: &ExactDerivedCollection<Source, Target>,
    ticket: &[CollectionCommit],
    algebra: &A,
) -> Result<ExactCover<Target>, ExactDerivedSyncError>
where
    S: BlobStore
        + CollectionStore
        + CapabilityProofStore
        + PeerStore
        + ArtifactOfferStore
        + StoreScope
        + WantStore
        + StorageFlush
        + StoreRevision
        + Send
        + 'static,
    S::Reader: BlobStoreMeta,
    Source: BlobEncoding + 'static,
    Target: BlobEncoding + 'static,
    Handle<Source>: InlineEncoding,
    Handle<Target>: InlineEncoding,
    A: ExactDerivedAlgebra<Source, Target> + ?Sized,
{
    if ticket.is_empty() {
        return Ok(ExactCover::default());
    }

    // Admit any already-arrived record inventory exactly once, then freeze the
    // speculative offer set. Moving inventory must not create a retry loop.
    peer.refresh();
    let mut offered = {
        let mut store = peer.store();
        let selectors = BTreeSet::from([
            CollectionRecordSelector::MergeCollection(lifecycle.target_collection()),
            CollectionRecordSelector::DeriveTarget(lifecycle.target_collection()),
        ]);
        store
            .select_records(&selectors)
            .map_err(|error| {
                ExactDerivedSyncError::storage("enumerate target artifact offers", error)
            })?
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Commit(_) => None,
                CollectionRecord::Merge(merge) => Some(merge.result()),
                CollectionRecord::Derive(derive) => Some(derive.mapping().1),
            })
            .collect::<BTreeSet<_>>()
    };

    'replan: loop {
        let plan = {
            let mut store = peer.store();
            lifecycle.probe_exact(&mut *store, ticket, algebra, &offered)
        };
        match plan {
            Ok(ExactAttachPlan::Ready(cover)) => return Ok(cover),
            Ok(ExactAttachPlan::Fetch(handles)) => {
                debug_assert!(!handles.is_empty(), "a fetch plan contains work");
                for expected in handles {
                    let Some(raw) = peer.fetch_blob(expected.raw).await else {
                        offered.remove(&expected);
                        continue 'replan;
                    };
                    let blob = Blob::<Target>::new(Bytes::from(raw));
                    let actual = Handle::<Target>::to_hash(blob.get_handle());
                    if actual != expected {
                        // `fetch_blob` already checks this. Keep the boundary
                        // explicit so a future transport cannot weaken it.
                        offered.remove(&expected);
                        continue 'replan;
                    }
                    let landed = peer.store().put::<Target, _>(blob).map_err(|error| {
                        ExactDerivedSyncError::storage("land fetched target artifact", error)
                    })?;
                    let landed = Handle::<Target>::to_hash(landed);
                    if landed != expected {
                        return Err(ExactDerivedSyncError::LandingIdentity {
                            expected,
                            actual: landed,
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
                let mut capture = OfferCapture::new(&mut *store);
                return lifecycle
                    .ensure_exact(&mut capture, ticket, algebra)
                    .map_err(Into::into);
            }
            Err(error) => return Err(error.into()),
        }
    }
}
