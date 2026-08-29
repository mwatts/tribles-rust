//! Logical values over exact typed collection covers.
//!
//! A [`Cover`](super::Cover) names the physical members of one exact lattice
//! point.  [`CoverAttachment`] pairs those typed identities with freshly
//! validated bytes, and [`TryFromCover`] decides whether a consumer eagerly
//! joins them or retains the shards as one lazy logical value.

use std::error::Error;

use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;

use super::{Collection, CollectionEncoding, Cover};

/// Freshly validated bytes attached to one exact semantic cover.
///
/// This is transient materialization state, not another durable record or a
/// second kind of cover.  The cover remains the identity passed between
/// collection stages; attached blobs merely avoid rereading selected members.
pub struct CoverAttachment<L: CollectionEncoding> {
    cover: Cover<L>,
    members: Vec<(Inline<Handle<L>>, L::Artifact)>,
}

impl<L: CollectionEncoding> CoverAttachment<L> {
    pub(crate) fn empty(collection: Collection<L>) -> Self {
        Self {
            cover: Cover::from_members(collection, []),
            members: Vec::new(),
        }
    }

    pub(crate) fn from_parts(
        cover: Cover<L>,
        members: Vec<(Inline<Handle<L>>, L::Artifact)>,
    ) -> Self {
        Self { cover, members }
    }

    /// Exact semantic cover whose bytes are attached.
    pub fn cover(&self) -> &Cover<L> {
        &self.cover
    }

    /// Number of selected physical members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the attachment is the store-free empty-cover bottom.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Borrow the ordered physical members.
    pub fn members(&self) -> &[(Inline<Handle<L>>, L::Artifact)] {
        &self.members
    }

    /// Consume the ordered physical members.
    pub fn into_members(self) -> Vec<(Inline<Handle<L>>, L::Artifact)> {
        self.members
    }

    /// Consume just the ordered attached artifacts.
    pub fn into_artifacts(self) -> impl ExactSizeIterator<Item = L::Artifact> {
        self.members.into_iter().map(|(_, artifact)| artifact)
    }
}

/// A logical value reconstructed from one exact typed physical cover.
///
/// This is deliberately cover-aware rather than a blanket blob conversion:
/// some values eagerly join members, while others retain mmap-backed shards
/// and answer queries over the union without constructing one monolith.
pub trait TryFromCover<L: CollectionEncoding>: Sized {
    /// Failure to construct the logical view from already validated members.
    type Error: Error + Send + Sync + 'static;

    /// Consume the exact attachment into its logical value.
    fn try_from_cover(attachment: CoverAttachment<L>) -> Result<Self, Self::Error>;
}
