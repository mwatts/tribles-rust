//! Logical values over exact typed collection covers.
//!
//! A [`Cover`](super::Cover) names semantic members of one exact lattice
//! point. [`Cover::materialize`](super::Cover::materialize) privately selects a
//! support-equivalent resident decomposition and invokes [`TryFromCover`] to
//! reconstruct a logical value through that same immutable store snapshot.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use crate::repo::BlobStoreGet;

use super::{CollectionData, CollectionEncoding, Cover};

/// One immutable logical view paired with the exact source cover it represents.
///
/// The source cover is the continuation value. The view may retain a distinct,
/// support-equivalent physical decomposition, such as mmap-backed Succinct
/// shards, without exposing that storage choice to callers.
pub struct CollectionSnapshot<S: CollectionEncoding, V> {
    source: Cover<S>,
    view: V,
}

impl<S: CollectionEncoding, V: Clone> Clone for CollectionSnapshot<S, V> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            view: self.view.clone(),
        }
    }
}

impl<S: CollectionEncoding, V> CollectionSnapshot<S, V> {
    /// Pair one exact source cover with its logical view.
    pub(crate) fn new(source: Cover<S>, view: V) -> Self {
        Self { source, view }
    }

    /// Exact source cover represented by this view.
    pub fn source(&self) -> &Cover<S> {
        &self.source
    }

    /// Logical value reconstructed from the source cover.
    pub fn view(&self) -> &V {
        &self.view
    }

    /// Consume this snapshot into its source cover and logical view.
    pub fn into_parts(self) -> (Cover<S>, V) {
        (self.source, self.view)
    }
}

/// Functional transition between two immutable collection snapshots.
///
/// The caller retains its previous snapshot and adopts `next` only after
/// downstream consumption succeeds. A reset carries a complete replacement;
/// strict growth additionally carries the exact newly added source support.
pub enum CollectionSnapshotAdvance<S: CollectionEncoding, V> {
    /// The source cover is unchanged, so the caller keeps its previous value.
    Unchanged,
    /// The source cover grew monotonically.
    Advanced {
        /// Candidate complete snapshot for the new source cover.
        next: CollectionSnapshot<S, V>,
        /// Exact logical view of only the newly added source members.
        changed: CollectionSnapshot<S, V>,
    },
    /// Additions-only processing is unsound; replace state with this snapshot.
    Reset {
        /// Candidate complete snapshot for the current source cover.
        next: CollectionSnapshot<S, V>,
    },
}

/// Failure at the generic boundary between a physical cover and its logical
/// interpretation.
#[derive(Debug)]
pub enum TryFromCoverError<GetError, ViewError> {
    /// One exact physical member could not be fetched from the snapshot.
    MemberGet {
        /// Selected physical member.
        member: CollectionData,
        /// Backend fetch failure.
        source: GetError,
    },
    /// Resident member bytes could not form the requested logical value.
    View(ViewError),
}

impl<GetError, ViewError> fmt::Display for TryFromCoverError<GetError, ViewError>
where
    GetError: fmt::Display,
    ViewError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemberGet { member, source } => write!(
                formatter,
                "failed to fetch cover member {}: {source}",
                hex::encode_upper(member.raw),
            ),
            Self::View(source) => source.fmt(formatter),
        }
    }
}

impl<GetError, ViewError> Error for TryFromCoverError<GetError, ViewError>
where
    GetError: Error + 'static,
    ViewError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MemberGet { source, .. } => Some(source),
            Self::View(source) => Some(source),
        }
    }
}

/// Low-level reconstruction hook for one selected typed physical cover.
///
/// This is deliberately cover-aware rather than a blanket blob conversion:
/// some values eagerly join members, while others retain mmap-backed shards
/// and answer queries over the union without constructing one monolith.
pub trait TryFromCover<L: CollectionEncoding>: Sized {
    /// Failure to construct the logical view from the resolved cover.
    type Error: Error + Send + Sync + 'static;

    /// Reconstruct the logical value named by `cover` through `snapshot`.
    ///
    /// Normal callers use [`Cover::materialize`](super::Cover::materialize).
    /// Its private resolver passes the actual physical decomposition selected
    /// from `snapshot`, never semantic coordinates paired with bytes from a
    /// different support-equivalent realization.
    fn try_from_cover<R>(
        cover: &Cover<L>,
        snapshot: &R,
    ) -> Result<Self, TryFromCoverError<R::GetError<Infallible>, Self::Error>>
    where
        R: BlobStoreGet;
}
