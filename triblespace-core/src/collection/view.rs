//! Immutable collection observations and logical views over their realized
//! covers.
//!
//! A [`CollectionSnapshot`] owns the store observation against which its
//! frozen support and realized target cover are valid. Logical values remain
//! caller-chosen projections reconstructed through [`TryFromCover`].

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use crate::repo::{BlobStoreGet, StoreSnapshot};

use super::{CollectionData, CollectionEncoding, Cover};

/// One immutable collection observation and its exact realized target cover.
///
/// `support` is the snapshot-valid source support represented by `cover`. It
/// is admitted support for the ordinary collection path and caller-requested
/// support for the explicit path. Keeping both covers with the store
/// observation prevents a caller from accidentally interpreting a physical
/// realization through a different snapshot. Logical views are reconstructed
/// on demand with [`Self::view`].
pub struct CollectionSnapshot<R, S, T>
where
    R: StoreSnapshot,
    S: CollectionEncoding,
    T: CollectionEncoding,
{
    snapshot: R,
    support: Cover<S>,
    cover: Cover<T>,
}

impl<R, S, T> Clone for CollectionSnapshot<R, S, T>
where
    R: StoreSnapshot,
    S: CollectionEncoding,
    T: CollectionEncoding,
{
    fn clone(&self) -> Self {
        Self {
            snapshot: self.snapshot.clone(),
            support: self.support.clone(),
            cover: self.cover.clone(),
        }
    }
}

impl<R, S, T> CollectionSnapshot<R, S, T>
where
    R: StoreSnapshot,
    S: CollectionEncoding,
    T: CollectionEncoding,
{
    /// Pair one store observation with frozen support and its realization.
    pub(crate) fn new(snapshot: R, support: Cover<S>, cover: Cover<T>) -> Self {
        Self {
            snapshot,
            support,
            cover,
        }
    }

    /// Immutable store observation against which both covers are valid.
    pub fn snapshot(&self) -> &R {
        &self.snapshot
    }

    /// Snapshot-valid source support represented by this snapshot.
    pub fn support(&self) -> &Cover<S> {
        &self.support
    }

    /// Exact target cover realized for [`Self::support`].
    pub fn cover(&self) -> &Cover<T> {
        &self.cover
    }

    /// Reconstruct one caller-chosen logical value from the realized cover.
    pub fn view<V>(&self) -> Result<V, TryFromCoverError<R::GetError<Infallible>, V::Error>>
    where
        R: BlobStoreGet,
        V: TryFromCover<T>,
    {
        V::try_from_cover(&self.cover, &self.snapshot)
    }

    /// Consume this snapshot into its store observation and exact covers.
    pub fn into_parts(self) -> (R, Cover<S>, Cover<T>) {
        (self.snapshot, self.support, self.cover)
    }
}

/// Functional transition between two immutable collection snapshots.
///
/// The caller retains its previous snapshot and adopts `next` only after
/// downstream consumption succeeds. A reset carries a complete replacement;
/// strict growth additionally carries the exact newly added support.
pub enum CollectionSnapshotAdvance<R, S, T>
where
    R: StoreSnapshot,
    S: CollectionEncoding,
    T: CollectionEncoding,
{
    /// The support is unchanged, so the caller keeps its previous value.
    Unchanged,
    /// The support grew monotonically.
    Advanced {
        /// Candidate complete snapshot for the new support.
        next: CollectionSnapshot<R, S, T>,
        /// Snapshot realizing exactly the newly added support.
        changed: CollectionSnapshot<R, S, T>,
    },
    /// Additions-only processing is unsound; replace state with this snapshot.
    Reset {
        /// Candidate complete snapshot for the current support.
        next: CollectionSnapshot<R, S, T>,
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
    /// Normal callers use [`CollectionSnapshot::view`], which passes the exact
    /// realized cover together with the immutable store observation that
    /// validated it.
    fn try_from_cover<R>(
        cover: &Cover<L>,
        snapshot: &R,
    ) -> Result<Self, TryFromCoverError<R::GetError<Infallible>, Self::Error>>
    where
        R: BlobStoreGet;
}
