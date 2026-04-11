//! `Peer<S>`: a store wrapped in distributed network sync.
//!
//! Owns the inner store, spawns the iroh network thread on construction,
//! and exposes the standard storage traits (`BlobStore + BlobStorePut +
//! BranchStore`) with two layers of network behavior built in:
//!
//! - **Reads** auto-call [`refresh`](Peer::refresh), which drains pending
//!   incoming gossip events into the wrapped store and re-publishes any
//!   deltas from external writers (e.g. another process appended to the
//!   same pile file). Mirrors `Pile::refresh` — the explicit method is
//!   available for tight loops, but normal storage use Just Works.
//! - **Writes** delegate to the inner store and then announce blobs to
//!   the DHT and gossip branch updates over the topic mesh, all via the
//!   network thread.
//!
//! Use [`fetch`](Peer::fetch) for one-shot pulls of a remote branch from
//! a specific peer (the `pile net pull` workflow). Set
//! `gossip_topic: None` in [`PeerConfig`] for pull-only mode where the
//! peer doesn't subscribe to a flood mesh.

use std::collections::HashMap;

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::blob::{BlobSchema, ToBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{
    BlobStore, BlobStoreList, BlobStorePut, BranchStore, PushResult,
};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::channel::NetEvent;
use crate::host::{self, HostReceiver, HostSender, StoreSnapshot};
use crate::protocol::{RawBranchId, RawHash};

/// Configuration for a [`Peer`]. Re-export of `HostConfig` since the
/// network thread is now a private implementation detail of `Peer`.
pub use crate::host::HostConfig as PeerConfig;

/// A store wrapped in distributed network sync.
///
/// See the [module-level docs](self) for the full mental model.
pub struct Peer<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    store: S,
    sender: HostSender,
    receiver: HostReceiver,

    /// Baseline blob snapshot for diff-and-publish on `refresh`. The Reader
    /// is a frozen view (for backends with snapshot semantics like Pile) so
    /// `current.blobs_diff(&last)` returns exactly the blobs added since
    /// the last refresh.
    last_blob_reader: Option<S::Reader>,

    /// Baseline branch heads for diff-and-publish on `refresh`. Updated on
    /// every Peer-driven write so we don't double-gossip our own changes.
    last_branches: HashMap<Id, RawHash>,
}

impl<S> Peer<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    /// Wrap a store in a Peer. Spawns the iroh network thread internally.
    ///
    /// The thread lives for the Peer's lifetime and shuts down when the
    /// Peer drops.
    pub fn new(mut store: S, key: SigningKey, config: PeerConfig) -> Self {
        let (sender, receiver) = host::spawn(key, config);

        // Seed the snapshot served by the network thread so peers
        // requesting via the protocol see our current state immediately.
        if let Some(snap) = StoreSnapshot::from_store(&mut store) {
            sender.update_snapshot(snap);
        }

        // Take an initial blob baseline so refresh()'s first call doesn't
        // re-announce every blob we already had on disk.
        let last_blob_reader = store.reader().ok();

        Peer {
            store,
            sender,
            receiver,
            last_blob_reader,
            last_branches: HashMap::new(),
        }
    }

    /// This peer's network identity (the iroh node id).
    pub fn id(&self) -> EndpointId {
        self.sender.id()
    }

    /// One-shot fetch of a branch from a specific peer.
    ///
    /// Used by `pile net pull` and other "go get this from over there"
    /// workflows. Does not require `gossip_topic` to be set — works in
    /// pull-only mode too. The fetched data lands in the wrapped store
    /// via the same auto-drain path that `refresh` uses.
    pub fn fetch(&self, peer: EndpointId, branch: RawBranchId) {
        self.sender.fetch(peer, branch);
    }

    /// Reconcile this peer with the latest external state.
    ///
    /// Two phases:
    ///
    /// 1. **Drain incoming events** — pulls any pending gossip
    ///    `NetEvent`s from the network thread into the wrapped store
    ///    (creating tracking branches as needed).
    /// 2. **Publish external writes** — diffs the wrapped store against
    ///    the last published baseline and gossips/announces any deltas
    ///    that didn't go through the Peer's own write path. Use this to
    ///    catch writes from another process that touched the pile file.
    ///
    /// Auto-called inside the BlobStore/BranchStore read methods, so
    /// callers using the storage normally don't need to invoke it.
    /// Mirrors `Pile::refresh` — the explicit method is available for
    /// "do it now" semantics or tight loops with no read activity.
    pub fn refresh(&mut self) {
        // ── Phase 1: drain incoming events ────────────────────────────
        while let Some(event) = self.receiver.try_recv() {
            match event {
                NetEvent::Blob(data) => {
                    let bytes: Bytes = data.into();
                    let _ = self.store.put::<UnknownBlob, Bytes>(bytes);
                }
                NetEvent::Head { branch, head, publisher } => {
                    if let Some(remote_id) = Id::new(branch) {
                        if let Some(name) = read_remote_name(&mut self.store, &head) {
                            crate::tracking::ensure_tracking_branch(
                                &mut self.store,
                                remote_id,
                                &head,
                                &name,
                                &publisher,
                            );
                        }
                    }
                }
            }
        }

        // ── Phase 2: diff-and-publish blob deltas ─────────────────────
        if let Ok(current) = self.store.reader() {
            if let Some(baseline) = self.last_blob_reader.as_ref() {
                for handle in current.blobs_diff(baseline).flatten() {
                    self.sender.announce(handle.raw);
                }
            }
            self.last_blob_reader = Some(current);
        }

        // ── Phase 3: diff-and-publish branch deltas ───────────────────
        let bids: Vec<Id> = match self.store.branches() {
            Ok(it) => it.filter_map(|r| r.ok()).collect(),
            Err(_) => return,
        };
        for bid in bids {
            if crate::tracking::is_tracking_branch(&mut self.store, bid) {
                continue;
            }
            let head = match self.store.head(bid) {
                Ok(Some(h)) => h,
                _ => continue,
            };
            if self.last_branches.get(&bid) != Some(&head.raw) {
                let bid_bytes: [u8; 16] = bid.into();
                self.sender.gossip(bid_bytes, head.raw);
                self.last_branches.insert(bid, head.raw);
            }
        }

        // ── Phase 4: refresh the snapshot served by the network thread ─
        if let Some(snap) = StoreSnapshot::from_store(&mut self.store) {
            self.sender.update_snapshot(snap);
        }
    }

    /// Force-republish all current non-tracking branches to the gossip
    /// topic, regardless of whether they appear changed since the last
    /// publish.
    ///
    /// Use this for periodic "I'm still here, here's my state"
    /// announcements that help newly-joined gossip neighbors learn about
    /// us. Long-running sync daemons typically call this every few seconds.
    /// Cheap to call repeatedly — iroh-gossip dedupes identical messages
    /// on the wire.
    ///
    /// Distinct from [`refresh`](Self::refresh): refresh publishes only
    /// the deltas it detects against its diff baselines. This method
    /// republishes everything unconditionally.
    pub fn republish_branches(&mut self) {
        let bids: Vec<Id> = match self.store.branches() {
            Ok(it) => it.filter_map(|r| r.ok()).collect(),
            Err(_) => return,
        };
        for bid in bids {
            if crate::tracking::is_tracking_branch(&mut self.store, bid) {
                continue;
            }
            if let Ok(Some(head)) = self.store.head(bid) {
                let bid_bytes: [u8; 16] = bid.into();
                self.sender.gossip(bid_bytes, head.raw);
                self.last_branches.insert(bid, head.raw);
            }
        }
        // Refresh the snapshot served by the network thread.
        if let Some(snap) = StoreSnapshot::from_store(&mut self.store) {
            self.sender.update_snapshot(snap);
        }
    }

    /// Borrow the underlying store. Use for store-specific methods that
    /// aren't part of the BlobStore/BranchStore traits (e.g. `Pile::flush`).
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Mutably borrow the underlying store. Writes through this borrow
    /// bypass the Peer's auto-publish and become invisible to the network
    /// until the next [`refresh`](Self::refresh) (which is auto-called on
    /// the next read).
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Consume the Peer and return the underlying store. The network
    /// thread shuts down when the Peer drops.
    pub fn into_store(self) -> S {
        self.store
    }
}

// ── Trait delegations ───────────────────────────────────────────────
//
// Reads (`reader`, `head`, `branches`) call `refresh()` first so they
// always see the latest gossiped state AND any external writes that
// landed since the last refresh get announced. Writes (`put`, `update`)
// delegate to the inner store and then push the new state out via the
// network thread, updating the diff baselines so refresh doesn't
// double-announce.

impl<S> BlobStorePut<Blake3> for Peer<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        let handle = self.store.put(item)?;
        self.sender.announce(handle.raw);
        // Update the blob baseline so refresh doesn't double-announce.
        self.last_blob_reader = self.store.reader().ok();
        Ok(handle)
    }
}

impl<S> BlobStore<Blake3> for Peer<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.refresh();
        self.store.reader()
    }
}

impl<S> BranchStore<Blake3> for Peer<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + BranchStore<Blake3>,
{
    type BranchesError = S::BranchesError;
    type HeadError = S::HeadError;
    type UpdateError = S::UpdateError;
    type ListIter<'a> = S::ListIter<'a> where S: 'a;

    fn branches<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::BranchesError> {
        self.refresh();
        self.store.branches()
    }

    fn head(
        &mut self,
        id: Id,
    ) -> Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        self.refresh();
        self.store.head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> Result<PushResult<Blake3>, Self::UpdateError> {
        let result = self.store.update(id, old, new.clone())?;
        if let PushResult::Success() = &result {
            if let Some(head) = new {
                // Tracking branches are local mirror state and must NOT be
                // re-gossiped — otherwise the publisher would receive its
                // own tracking branch back and create a tracking-of-the-
                // tracking, ad infinitum.
                if !crate::tracking::is_tracking_branch(&mut self.store, id) {
                    let bid_bytes: [u8; 16] = id.into();
                    self.sender.gossip(bid_bytes, head.raw);
                    self.last_branches.insert(id, head.raw);
                }
                // Refresh the snapshot served by the network thread.
                if let Some(snap) = StoreSnapshot::from_store(&mut self.store) {
                    self.sender.update_snapshot(snap);
                }
            }
        }
        Ok(result)
    }
}

/// Read the branch name from a branch metadata blob. Tries `metadata::name`
/// first (normal branches) and falls back to `remote_name` (tracking
/// branches mirrored from a remote peer).
fn read_remote_name<S: BlobStore<Blake3>>(store: &mut S, head_hash: &RawHash) -> Option<String> {
    use triblespace_core::blob::schemas::longstring::LongString;
    use triblespace_core::repo::BlobStoreGet;
    use triblespace_core::macros::{find, pattern};

    let reader = store.reader().ok()?;
    let meta_handle = Value::<Handle<Blake3, SimpleArchive>>::new(*head_hash);
    let meta: triblespace_core::trible::TribleSet = reader.get(meta_handle).ok()?;

    let name_handle: Value<Handle<Blake3, LongString>> = find!(
        h: Value<Handle<Blake3, LongString>>,
        pattern!(&meta, [{ _?e @ triblespace_core::metadata::name: ?h }])
    )
    .next()
    .or_else(|| {
        find!(
            h: Value<Handle<Blake3, LongString>>,
            pattern!(&meta, [{ _?e @ crate::tracking::remote_name: ?h }])
        )
        .next()
    })?;

    let name_view: anybytes::View<str> = reader.get(name_handle).ok()?;
    Some(name_view.as_ref().to_string())
}
