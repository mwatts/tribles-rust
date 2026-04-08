//! `Follower<S>`: keeps a local store in sync with a remote peer.
//!
//! All networking is internal — background tasks handle gossip and blob
//! pulling. The public API is fully synchronous. `Repository<Follower<S>>`
//! just works — Repository has no idea there's networking happening.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use anybytes::Bytes;
use iroh::endpoint::Connection;
use iroh_gossip::api::{Event as GossipEvent, GossipReceiver};
use futures::TryStreamExt;
use triblespace_core::blob::{BlobSchema, ToBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut, BranchStore, PushResult};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::protocol::*;

/// Keeps a local store in sync with a remote peer.
///
/// All networking is hidden — background tasks listen to gossip and
/// pull blobs. The public API is completely synchronous.
pub struct Follower<S> {
    store: Arc<Mutex<S>>,
    conn: Connection,
    remote_heads: Arc<Mutex<HashMap<Id, RawHash>>>,
    _tasks: Vec<tokio::task::JoinHandle<()>>,
}

pub struct FollowerBuilder<S> {
    store: S,
    conn: Connection,
    gossip: Option<GossipReceiver>,
}

impl<S> FollowerBuilder<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + Send + 'static,
{
    /// Provide a gossip receiver for automatic sync on HEAD changes.
    pub fn gossip(mut self, receiver: GossipReceiver) -> Self {
        self.gossip = Some(receiver);
        self
    }

    /// Build the Follower, spawning background tasks.
    pub fn build(self, rt: &tokio::runtime::Handle) -> Follower<S> {
        let store = Arc::new(Mutex::new(self.store));
        let remote_heads: Arc<Mutex<HashMap<Id, RawHash>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let mut tasks = Vec::new();

        // Gossip listener task.
        if let Some(mut receiver) = self.gossip {
            let store = store.clone();
            let heads = remote_heads.clone();
            let conn = self.conn.clone();
            tasks.push(rt.spawn(async move {
                while let Ok(Some(event)) = receiver.try_next().await {
                    if let GossipEvent::Received(msg) = event {
                        // HEAD message: 0x01 + branch_id(16) + hash(32) = 49 bytes
                        if msg.content.len() == 49 && msg.content[0] == 0x01 {
                            let mut branch_id_bytes = [0u8; 16];
                            branch_id_bytes.copy_from_slice(&msg.content[1..17]);
                            let mut head = [0u8; 32];
                            head.copy_from_slice(&msg.content[17..49]);

                            let Some(branch_id) = Id::new(branch_id_bytes) else { continue; };

                            // Pull blobs reachable from the new head.
                            if let Err(e) = pull_into(&store, &conn, &head).await {
                                eprintln!("follower pull error: {e}");
                                continue;
                            }

                            // Update tracking ref.
                            heads.lock().unwrap().insert(branch_id, head);
                        }
                    }
                }
            }));
        }

        Follower { store, conn: self.conn, remote_heads, _tasks: tasks }
    }
}

impl<S> Follower<S>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3> + Send + 'static,
{
    pub fn builder(store: S, conn: Connection) -> FollowerBuilder<S> {
        FollowerBuilder { store, conn, gossip: None }
    }

    /// Manually trigger a sync for a branch. Queries the remote HEAD,
    /// pulls all reachable blobs, updates the tracking ref.
    /// Use this for one-shot pulls without gossip.
    pub async fn sync(&self, branch_id: Id) -> Result<()> {
        let id_bytes: [u8; 16] = branch_id.into();
        let Some(head) = op_head(&self.conn, &id_bytes).await? else {
            return Ok(());
        };
        pull_into(&self.store, &self.conn, &head).await?;
        self.remote_heads.lock().unwrap().insert(branch_id, head);
        Ok(())
    }

    /// What's the remote's latest known head for this branch?
    /// Returns None if no gossip/sync has been received for this branch yet.
    pub fn remote_head(&self, branch_id: Id) -> Option<RawHash> {
        self.remote_heads.lock().unwrap().get(&branch_id).copied()
    }

    /// All known remote branch heads.
    pub fn remote_heads(&self) -> HashMap<Id, RawHash> {
        self.remote_heads.lock().unwrap().clone()
    }
}

/// Pull all blobs reachable from a root hash into a locked store.
async fn pull_into<S>(
    store: &Arc<Mutex<S>>,
    conn: &Connection,
    root: &RawHash,
) -> Result<()>
where
    S: BlobStore<Blake3> + BlobStorePut<Blake3>,
{
    // Check if root is already local.
    if has_blob_locked(store, root) {
        return Ok(());
    }

    // Fetch root.
    if let Some(data) = op_get_blob(conn, root).await? {
        put_blob_locked(store, data)?;
    }

    // BFS.
    let mut seen = std::collections::HashSet::new();
    seen.insert(*root);
    let mut current_level = vec![*root];

    while !current_level.is_empty() {
        let mut next_level = Vec::new();
        for parent in &current_level {
            let children = op_children(conn, parent).await?;
            for hash in children {
                if !seen.insert(hash) { continue; }
                if has_blob_locked(store, &hash) { continue; }
                if let Some(data) = op_get_blob(conn, &hash).await? {
                    put_blob_locked(store, data)?;
                    next_level.push(hash);
                }
            }
        }
        current_level = next_level;
    }
    Ok(())
}

fn has_blob_locked<S: BlobStore<Blake3>>(store: &Mutex<S>, hash: &RawHash) -> bool {
    let mut guard = store.lock().unwrap();
    let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
    let Ok(reader) = guard.reader() else { return false; };
    reader.get::<Bytes, UnknownBlob>(handle).is_ok()
}

fn put_blob_locked<S: BlobStorePut<Blake3>>(store: &Mutex<S>, data: Vec<u8>) -> Result<()> {
    let bytes: Bytes = data.into();
    let _: Value<Handle<Blake3, UnknownBlob>> = store.lock().unwrap()
        .put::<UnknownBlob, Bytes>(bytes)
        .map_err(|e| anyhow!("put: {e:?}"))?;
    Ok(())
}

// ── Trait delegations: Follower IS a store ───────────────────────────

impl<S> BlobStorePut<Blake3> for Follower<S>
where
    S: BlobStorePut<Blake3>,
{
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> std::result::Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        self.store.lock().unwrap().put(item)
    }
}

impl<S> BlobStore<Blake3> for Follower<S>
where
    S: BlobStore<Blake3>,
{
    type Reader = S::Reader;
    type ReaderError = S::ReaderError;

    fn reader(&mut self) -> std::result::Result<Self::Reader, Self::ReaderError> {
        self.store.lock().unwrap().reader()
    }
}

impl<S> BranchStore<Blake3> for Follower<S>
where
    S: BranchStore<Blake3>,
{
    type BranchesError = S::BranchesError;
    type HeadError = S::HeadError;
    type UpdateError = S::UpdateError;
    type ListIter<'a> = std::vec::IntoIter<std::result::Result<Id, S::BranchesError>> where S: 'a;

    fn branches<'a>(&'a mut self) -> std::result::Result<Self::ListIter<'a>, Self::BranchesError> {
        let items: Vec<_> = self.store.lock().unwrap().branches()?.collect();
        Ok(items.into_iter())
    }

    fn head(&mut self, id: Id) -> std::result::Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        self.store.lock().unwrap().head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> std::result::Result<PushResult<Blake3>, Self::UpdateError> {
        self.store.lock().unwrap().update(id, old, new)
    }
}
