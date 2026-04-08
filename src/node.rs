//! `NetworkStore`: middleware that wraps any `BlobStore + BranchStore`
//! and makes it a full network participant.
//!
//! - `put()` → store locally + announce blob to DHT
//! - `update()` → local CAS + gossip HEAD change
//! - Serves incoming connections via iroh Router
//! - Reads delegate to the inner store (no network overhead)

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use iroh::endpoint::{Connection, presets};
use iroh::protocol::{AcceptError, ProtocolHandler, Router};
use iroh::Endpoint;
use iroh_base::EndpointId;
use iroh_gossip::Gossip;
use iroh_gossip::api::GossipSender;
use triblespace_core::blob::{BlobSchema, ToBlob, TryFromBlob};
use triblespace_core::blob::schemas::UnknownBlob;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::repo::{
    BlobStore, BlobStoreGet, BlobStorePut, BranchStore, PushResult,
};
use triblespace_core::value::Value;
use triblespace_core::value::ValueSchema;
use triblespace_core::value::schemas::hash::{Blake3, Handle};

use crate::identity::iroh_secret;
use crate::protocol::PILE_SYNC_ALPN;

/// A network-aware wrapper around any `BlobStore + BranchStore`.
pub struct NetworkStore<S> {
    inner: Arc<Mutex<S>>,
    endpoint: Endpoint,
    router: Router,
    dht_api: Option<iroh_dht::api::ApiClient>,
    gossip_sender: Option<GossipSender>,
    my_id: EndpointId,
    rt: tokio::runtime::Handle,
}

/// Builder for constructing a `NetworkStore`.
pub struct NetworkStoreBuilder<S> {
    inner: S,
    signing_key: ed25519_dalek::SigningKey,
    dht_bootstrap: Vec<EndpointId>,
    gossip_topic: Option<String>,
    gossip_peers: Vec<EndpointId>,
}

impl<S> NetworkStoreBuilder<S>
where
    S: BlobStore<Blake3> + BranchStore<Blake3> + Send + 'static,
{
    /// Enable DHT with the given bootstrap peers.
    pub fn dht(mut self, bootstrap: Vec<EndpointId>) -> Self {
        self.dht_bootstrap = bootstrap;
        self
    }

    /// Enable gossip on the given topic with initial peers.
    pub fn gossip(mut self, topic: impl Into<String>, peers: Vec<EndpointId>) -> Self {
        self.gossip_topic = Some(topic.into());
        self.gossip_peers = peers;
        self
    }

    /// Build and start the network store.
    pub async fn build(self) -> Result<NetworkStore<S>> {
        let secret = iroh_secret(&self.signing_key);
        let public = secret.public();
        let my_id: EndpointId = public.into();

        let ep = Endpoint::builder(presets::N0).secret_key(secret).bind().await
            .map_err(|e| anyhow!("bind: {e}"))?;
        ep.online().await;

        let inner = Arc::new(Mutex::new(self.inner));
        let rt = tokio::runtime::Handle::current();

        let handler = StoreHandler { inner: inner.clone() };
        let mut router_builder = Router::builder(ep.clone())
            .accept(PILE_SYNC_ALPN, handler);

        // DHT.
        let dht_api = if !self.dht_bootstrap.is_empty() {
            let dht_alpn = iroh_dht::rpc::ALPN;
            let pool = iroh_blobs::util::connection_pool::ConnectionPool::new(
                ep.clone(), dht_alpn,
                iroh_blobs::util::connection_pool::Options {
                    max_connections: 64,
                    idle_timeout: std::time::Duration::from_secs(30),
                    connect_timeout: std::time::Duration::from_secs(10),
                    on_connected: None,
                },
            );
            let iroh_pool = iroh_dht::pool::IrohPool::new(ep.clone(), pool);
            let (rpc, api) = iroh_dht::create_node(
                my_id, iroh_pool.clone(), self.dht_bootstrap, Default::default(),
            );
            iroh_pool.set_self_client(Some(rpc.downgrade()));
            let dht_sender = rpc.inner().as_local().expect("local sender");
            router_builder = router_builder
                .accept(dht_alpn, irpc_iroh::IrohProtocol::with_sender(dht_sender));
            Some(api)
        } else {
            None
        };

        // Gossip.
        let gossip_sender = if let Some(topic_name) = self.gossip_topic {
            let gossip = Gossip::builder().spawn(ep.clone());
            router_builder = router_builder
                .accept(iroh_gossip::ALPN, gossip.clone());

            let topic_id = iroh_gossip::TopicId::from_bytes(
                *blake3::hash(topic_name.as_bytes()).as_bytes()
            );
            let topic = if self.gossip_peers.is_empty() {
                gossip.subscribe(topic_id, self.gossip_peers).await
            } else {
                gossip.subscribe_and_join(topic_id, self.gossip_peers).await
            }.map_err(|e| anyhow!("gossip subscribe: {e}"))?;
            let (sender, _receiver) = topic.split();
            Some(sender)
        } else {
            None
        };

        let router = router_builder.spawn();

        Ok(NetworkStore {
            inner, endpoint: ep, router, dht_api, gossip_sender, my_id, rt,
        })
    }
}

impl<S> NetworkStore<S>
where
    S: BlobStore<Blake3> + BranchStore<Blake3> + Send + 'static,
{
    /// Start building a network store.
    pub fn builder(inner: S, signing_key: ed25519_dalek::SigningKey) -> NetworkStoreBuilder<S> {
        NetworkStoreBuilder {
            inner, signing_key,
            dht_bootstrap: Vec::new(),
            gossip_topic: None,
            gossip_peers: Vec::new(),
        }
    }

    /// The node's endpoint ID (public key).
    pub fn id(&self) -> EndpointId { self.my_id }

    /// The iroh endpoint.
    pub fn endpoint(&self) -> &Endpoint { &self.endpoint }

    /// Wait until interrupted.
    pub async fn run_until_ctrl_c(self) -> Result<()> {
        tokio::signal::ctrl_c().await?;
        self.router.shutdown().await.map_err(|e| anyhow!("shutdown: {e}"))
    }

    /// Access the inner store directly (locked).
    pub fn with_inner<R>(&self, f: impl FnOnce(&mut S) -> R) -> R {
        f(&mut *self.inner.lock().unwrap())
    }
}

// ── Trait implementations ────────────────────────────────────────────

impl<S> BlobStorePut<Blake3> for NetworkStore<S>
where
    S: BlobStorePut<Blake3> + Send + 'static,
{
    type PutError = S::PutError;

    fn put<Sch, T>(&mut self, item: T) -> Result<Value<Handle<Blake3, Sch>>, Self::PutError>
    where
        Sch: BlobSchema + 'static,
        T: ToBlob<Sch>,
        Handle<Blake3, Sch>: ValueSchema,
    {
        let handle = self.inner.lock().unwrap().put(item)?;

        // Fire-and-forget DHT announce.
        if let Some(ref api) = self.dht_api {
            let hash = blake3::Hash::from_bytes(handle.raw);
            let my_id = self.my_id;
            let api = api.clone();
            self.rt.spawn(async move {
                let _ = api.announce_provider(hash, my_id).await;
            });
        }

        Ok(handle)
    }
}

impl<S> BranchStore<Blake3> for NetworkStore<S>
where
    S: BranchStore<Blake3> + Send + 'static,
{
    type BranchesError = S::BranchesError;
    type HeadError = S::HeadError;
    type UpdateError = S::UpdateError;
    type ListIter<'a> = std::vec::IntoIter<Result<Id, S::BranchesError>> where S: 'a;

    fn branches<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::BranchesError> {
        // Collect to avoid holding the lock across iteration.
        let items: Vec<_> = self.inner.lock().unwrap().branches()?.collect();
        Ok(items.into_iter())
    }

    fn head(&mut self, id: Id) -> Result<Option<Value<Handle<Blake3, SimpleArchive>>>, Self::HeadError> {
        self.inner.lock().unwrap().head(id)
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Value<Handle<Blake3, SimpleArchive>>>,
        new: Option<Value<Handle<Blake3, SimpleArchive>>>,
    ) -> Result<PushResult<Blake3>, Self::UpdateError> {
        let result = self.inner.lock().unwrap().update(id, old, new.clone())?;

        // Fire-and-forget gossip on success.
        if let PushResult::Success() = &result {
            if let (Some(sender), Some(new_head)) = (&self.gossip_sender, new) {
                let id_bytes: [u8; 16] = id.into();
                let mut msg = Vec::with_capacity(1 + 16 + 32);
                msg.push(0x01); // HEAD message marker
                msg.extend_from_slice(&id_bytes);
                msg.extend_from_slice(&new_head.raw);
                let sender = sender.clone();
                self.rt.spawn(async move {
                    let _ = sender.broadcast(msg.into()).await;
                });
            }
        }

        Ok(result)
    }
}

// ── Protocol handler ─────────────────────────────────────────────────

/// Serves incoming streams from the inner store.
#[derive(Clone)]
struct StoreHandler<S> {
    inner: Arc<Mutex<S>>,
}

/// Serve one stream: read request (async), process (sync under lock), write response (async).
/// The MutexGuard never crosses an await point.
async fn serve_one_stream<S>(
    store: &Arc<Mutex<S>>,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<()>
where
    S: BlobStore<Blake3> + BranchStore<Blake3>,
{
    use crate::protocol::*;
    use triblespace_core::blob::schemas::UnknownBlob;
    use triblespace_core::repo::BlobStoreGet;
    use anybytes::Bytes;

    // Phase 1: read request (async, no lock).
    let op = recv_u8(recv).await?;

    enum Response {
        List(Vec<([u8; 16], [u8; 32])>),
        Head(Option<[u8; 32]>),
        Blob(Option<Vec<u8>>),
        Children(Vec<([u8; 32], Vec<u8>)>),
        Unknown,
    }

    // Phase 1b: read remaining request data (async, no lock).
    enum Request {
        List,
        Head([u8; 16]),
        GetBlob([u8; 32]),
        Children { parent: [u8; 32], have: Vec<[u8; 32]> },
        Unknown,
    }

    let request = match op {
        OP_LIST => Request::List,
        OP_HEAD => Request::Head(recv_branch_id(recv).await?),
        OP_GET_BLOB => Request::GetBlob(recv_hash(recv).await?),
        OP_CHILDREN => {
            let parent = recv_hash(recv).await?;
            let have_count = recv_u32_be(recv).await? as usize;
            let mut have = Vec::with_capacity(have_count);
            for _ in 0..have_count {
                have.push(recv_hash(recv).await?);
            }
            Request::Children { parent, have }
        }
        _ => Request::Unknown,
    };

    // Phase 2: process under lock (sync, no await — MutexGuard never crosses await).
    let response = {
        let mut guard = store.lock().unwrap();
        match request {
            Request::List => {
                let ids: Vec<Id> = guard.branches()
                    .map_err(|e| anyhow!("branches: {e:?}"))?
                    .filter_map(|r| r.ok())
                    .collect();
                let mut entries = Vec::new();
                for id in ids {
                    if let Ok(Some(head)) = guard.head(id) {
                        entries.push((id.into(), head.raw));
                    }
                }
                Response::List(entries)
            }
            Request::Head(id_bytes) => {
                let hash = Id::new(id_bytes).and_then(|bid| {
                    guard.head(bid).ok().flatten().map(|h| h.raw)
                });
                Response::Head(hash)
            }
            Request::GetBlob(hash) => {
                let handle = Value::<Handle<Blake3, UnknownBlob>>::new(hash);
                let data = guard.reader().ok()
                    .and_then(|r| r.get::<Bytes, UnknownBlob>(handle).ok())
                    .map(|b: Bytes| b.to_vec());
                Response::Blob(data)
            }
            Request::Children { parent, have } => {
                let have_set: std::collections::HashSet<[u8; 32]> = have.into_iter().collect();
                let parent_handle = Value::<Handle<Blake3, UnknownBlob>>::new(parent);
                let parent_blob = guard.reader().ok()
                    .and_then(|r| r.get::<Bytes, UnknownBlob>(parent_handle).ok());
                match parent_blob {
                    Some(blob) => {
                        let parent_data: Vec<u8> = blob.to_vec();
                        let mut result = Vec::new();
                        for chunk in parent_data.chunks(32) {
                            if chunk.len() == 32 {
                                let mut candidate = [0u8; 32];
                                candidate.copy_from_slice(chunk);
                                if !have_set.contains(&candidate) {
                                    let h = Value::<Handle<Blake3, UnknownBlob>>::new(candidate);
                                    if let Some(data) = guard.reader().ok()
                                        .and_then(|r| r.get::<Bytes, UnknownBlob>(h).ok())
                                        .map(|b: Bytes| b.to_vec()) {
                                        result.push((candidate, data));
                                    }
                                }
                            }
                        }
                        Response::Children(result)
                    }
                    None => Response::Children(Vec::new()),
                }
            }
            Request::Unknown => Response::Unknown,
        }
    }; // guard dropped here, before any await

    // Phase 3: write response (async, no lock).
    match response {
        Response::List(entries) => {
            for (id_bytes, head) in &entries {
                send_u8(send, RSP_BLOB).await?;
                send_branch_id(send, id_bytes).await?;
                send_hash(send, head).await?;
            }
            send_u8(send, RSP_END).await?;
        }
        Response::Head(Some(hash)) => {
            send_u8(send, RSP_HEAD_OK).await?;
            send_hash(send, &hash).await?;
        }
        Response::Head(None) => {
            send_u8(send, RSP_NONE).await?;
        }
        Response::Blob(Some(data)) => {
            send_u8(send, RSP_BLOB).await?;
            send_u32_be(send, data.len() as u32).await?;
            send.write_all(&data).await.map_err(|e| anyhow!("send: {e}"))?;
        }
        Response::Blob(None) => {
            send_u8(send, RSP_MISSING).await?;
        }
        Response::Children(children) => {
            for (hash, data) in &children {
                send_u8(send, RSP_BLOB).await?;
                send_hash(send, hash).await?;
                send_u32_be(send, data.len() as u32).await?;
                send.write_all(data).await.map_err(|e| anyhow!("send: {e}"))?;
            }
            send_u8(send, RSP_END).await?;
        }
        Response::Unknown => {}
    }
    Ok(())
}

impl<S> std::fmt::Debug for StoreHandler<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreHandler").finish()
    }
}

impl<S> ProtocolHandler for StoreHandler<S>
where
    S: BlobStore<Blake3> + BranchStore<Blake3> + Send + 'static,
{
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let inner = self.inner.clone();
        loop {
            let (mut send, mut recv) = match connection.accept_bi().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            // Process on the accept task — no spawn needed since one-stream-per-op
            // means each stream is short-lived. The lock is only held during
            // the synchronous data collection, not across network I/O.
            if let Err(e) = serve_one_stream(&inner, &mut send, &mut recv).await {
                eprintln!("handler error: {e}");
            }
            let _ = send.finish();
        }
        Ok(())
    }
}
