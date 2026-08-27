//! Durable positive local willingness to serve content-addressed artifacts.
//!
//! An [`ArtifactOfferStore`] records only local service intent. An offer grants
//! no authority, proves neither reachability nor residency, creates no demand,
//! and is not a garbage-collection root. The algebra is a grow-only set: there
//! is intentionally no retraction operation.

use std::error::Error;
use std::fmt::Debug;

use crate::blob::encodings::UnknownBlob;
use crate::id::{id_hex, Id};
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, INLINE_LEN};
use crate::patch::{Entry, IdentitySchema, XorSip128, PATCH};

/// Stable semantic kind of a positive local artifact offer.
///
/// Minted with `trible genid` on 2026-08-27. The pile record description is
/// rooted at this same anchor.
pub const KIND_ARTIFACT_OFFER: Id = id_hex!("6EE89EEA7E6ECB2463FA5EE9C955B378");

/// Content-addressed artifact named by an offer.
pub type ArtifactHandle = Inline<Handle<UnknownBlob>>;

type ArtifactOfferIndex = PATCH<INLINE_LEN, IdentitySchema, (), XorSip128>;

/// Cheap immutable deterministic view of the offers known to one store.
///
/// Cloning this snapshot is O(1). Iteration is in canonical handle-byte order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArtifactOfferSnapshot {
    offers: ArtifactOfferIndex,
}

impl ArtifactOfferSnapshot {
    /// Number of distinct offered artifacts.
    pub fn len(&self) -> u64 {
        self.offers.len()
    }

    /// Whether this snapshot contains no offers.
    pub fn is_empty(&self) -> bool {
        self.offers.is_empty()
    }

    /// Whether this snapshot contains `handle`.
    pub fn contains(&self, handle: ArtifactHandle) -> bool {
        self.offers.get(&handle.raw).is_some()
    }

    /// Iterate over offered handles in canonical byte order.
    pub fn iter(&self) -> impl Iterator<Item = ArtifactHandle> + '_ {
        self.offers
            .iter_ordered()
            .map(|raw| ArtifactHandle::new(*raw))
    }

    pub(crate) fn insert(&mut self, handle: ArtifactHandle) {
        self.offers.insert(&Entry::new(&handle.raw));
    }

    pub(crate) fn union(&mut self, other: Self) {
        self.offers.union(other.offers);
    }
}

/// Grow-only storage for positive local willingness to serve artifacts.
///
/// The bulk operation is primary so durable backends can refresh, lock, and
/// append once per batch. Implementations should deduplicate both the input
/// batch and offers they already know. If an error follows one or more durable
/// insertions, those insertions lawfully remain as partial grow-only success.
pub trait ArtifactOfferStore {
    /// Failure while observing or extending the offer set.
    type OfferError: Error + Debug + Send + Sync + 'static;

    /// Add every distinct handle in `handles` to the local offer set.
    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>;

    /// Add one handle to the local offer set.
    fn offer(&mut self, handle: ArtifactHandle) -> Result<(), Self::OfferError> {
        self.offer_all([handle])
    }

    /// Take a cheap immutable deterministic snapshot of the current offer set.
    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError>;
}

impl<S> ArtifactOfferStore for &mut S
where
    S: ArtifactOfferStore + ?Sized,
{
    type OfferError = S::OfferError;

    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>,
    {
        (**self).offer_all(handles)
    }

    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError> {
        (**self).offers_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::StoreRevision;

    fn handle(byte: u8) -> ArtifactHandle {
        ArtifactHandle::new([byte; INLINE_LEN])
    }

    #[test]
    fn memory_offers_are_grow_only_deduplicated_and_snapshotted() {
        let mut repo = MemoryRepo::default();
        repo.offer_all([handle(2), handle(1), handle(2)]).unwrap();
        let before = repo.offers_snapshot().unwrap();

        assert_eq!(
            before.iter().collect::<Vec<_>>(),
            vec![handle(1), handle(2)]
        );
        assert!(before.contains(handle(1)));
        assert_eq!(before.len(), 2);

        repo.offer(handle(3)).unwrap();
        assert_eq!(
            before.len(),
            2,
            "snapshots do not change under later writes"
        );
        assert_eq!(repo.offers_snapshot().unwrap().len(), 3);
    }

    #[test]
    fn local_offers_do_not_change_sync_visible_revision() {
        let mut repo = MemoryRepo::default();
        let before = repo.store_revision().unwrap();
        repo.offer(handle(9)).unwrap();
        let after = repo.store_revision().unwrap();
        assert!(before == after);
    }
}
