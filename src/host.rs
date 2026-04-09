//! `Host`: the network thread.
//!
//! Owns the iroh endpoint, gossip, DHT, and protocol server.
//! Runs in its own thread with its own tokio runtime.
//! Cloneable handle (`Arc` inside) — Leader and Follower hold clones.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use iroh_base::EndpointId;
use ed25519_dalek::SigningKey;

use crate::channel::{NetCommand, NetEvent};
use crate::identity::iroh_secret;
use crate::protocol::*;

/// Configuration for the Host.
pub struct HostConfig {
    pub gossip_topic: Option<String>,
    pub gossip_peers: Vec<EndpointId>,
    pub dht_bootstrap: Vec<EndpointId>,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            gossip_topic: None,
            gossip_peers: Vec::new(),
            dht_bootstrap: Vec::new(),
        }
    }
}

/// Snapshot of store state for serving protocol requests.
/// Captures both blob reader and branch heads at a point in time.
pub struct StoreSnapshot<R> {
    pub reader: R,
    pub branches: Vec<(RawBranchId, RawHash)>,
}

impl StoreSnapshot<()> {
    /// Create a snapshot from a store.
    pub fn from_store<S>(store: &mut S) -> Option<StoreSnapshot<S::Reader>>
    where
        S: triblespace_core::repo::BlobStore<triblespace_core::value::schemas::hash::Blake3>
            + triblespace_core::repo::BranchStore<triblespace_core::value::schemas::hash::Blake3>,
    {
        let ids: Vec<triblespace_core::id::Id> = store.branches().ok()?
            .filter_map(|r| r.ok())
            .collect();
        let mut branches = Vec::new();
        for id in ids {
            if let Ok(Some(head)) = store.head(id) {
                let id_bytes: [u8; 16] = id.into();
                branches.push((id_bytes, head.raw));
            }
        }
        let reader = store.reader().ok()?;
        Some(StoreSnapshot { reader, branches })
    }
}

/// Trait object for serving — type-erased snapshot.
pub trait AnySnapshot: Send + 'static {
    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>>;
    fn has_blob(&self, hash: &RawHash) -> bool;
    fn list_branches(&self) -> &[(RawBranchId, RawHash)];
    fn head(&self, branch: &RawBranchId) -> Option<RawHash>;
}

impl<R> AnySnapshot for StoreSnapshot<R>
where
    R: triblespace_core::repo::BlobStoreGet<triblespace_core::value::schemas::hash::Blake3>
        + Send + 'static,
{
    fn get_blob(&self, hash: &RawHash) -> Option<Vec<u8>> {
        use triblespace_core::blob::schemas::UnknownBlob;
        use triblespace_core::value::Value;
        use triblespace_core::value::schemas::hash::{Blake3, Handle};
        let handle = Value::<Handle<Blake3, UnknownBlob>>::new(*hash);
        self.reader.get::<anybytes::Bytes, UnknownBlob>(handle).ok().map(|b| b.to_vec())
    }

    fn has_blob(&self, hash: &RawHash) -> bool {
        self.get_blob(hash).is_some()
    }

    fn list_branches(&self) -> &[(RawBranchId, RawHash)] {
        &self.branches
    }

    fn head(&self, branch: &RawBranchId) -> Option<RawHash> {
        self.branches.iter().find(|(b, _)| b == branch).map(|(_, h)| *h)
    }
}

struct HostInner {
    reader: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    cmd_tx: mpsc::Sender<NetCommand>,
    evt_rx: Mutex<mpsc::Receiver<NetEvent>>,
    id: EndpointId,
    _thread: thread::JoinHandle<()>,
}

/// The network presence. Cloneable handle.
#[derive(Clone)]
pub struct Host(Arc<HostInner>);

impl Host {
    pub fn new(key: SigningKey, config: HostConfig) -> Self {
        let secret = iroh_secret(&key);
        let id: EndpointId = secret.public().into();

        let (cmd_tx, cmd_rx) = mpsc::channel::<NetCommand>();
        let (evt_tx, evt_rx) = mpsc::channel::<NetEvent>();

        // Shared reader — Host thread reads, Leader updates.
        let reader = Arc::new(Mutex::new(None::<Box<dyn AnySnapshot>>));
        let thread_reader = reader.clone();

        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(host_loop(secret, config, cmd_rx, evt_tx, thread_reader));
        });

        Host(Arc::new(HostInner {
            reader,
            cmd_tx,
            evt_rx: Mutex::new(evt_rx),
            id,
            _thread: thread,
        }))
    }

    /// This node's endpoint ID (public key).
    pub fn id(&self) -> EndpointId { self.0.id }

    // ── Leader interface ─────────────────────────────────────────────

    /// Send a command to the network thread (non-blocking).
    pub fn send_command(&self, cmd: NetCommand) {
        let _ = self.0.cmd_tx.send(cmd);
    }

    /// Announce a blob hash to the DHT.
    pub fn announce(&self, hash: RawHash) {
        self.send_command(NetCommand::Announce(hash));
    }

    /// Gossip a HEAD change.
    pub fn gossip(&self, branch: RawBranchId, head: RawHash) {
        self.send_command(NetCommand::Gossip { branch, head });
    }

    /// Update the reader snapshot used for serving protocol requests.
    pub fn update_snapshot(&self, reader: impl AnySnapshot) {
        *self.0.reader.lock().unwrap() = Some(Box::new(reader));
    }

    // ── Follower interface ───────────────────────────────────────────

    /// Try to receive an event from the network thread (non-blocking).
    pub fn try_recv(&self) -> Option<NetEvent> {
        self.0.evt_rx.lock().unwrap().try_recv().ok()
    }

    // ── Control ──────────────────────────────────────────────────────

    /// Request a fetch from a remote peer.
    pub fn fetch(&self, peer: EndpointId, branch: RawBranchId) {
        self.send_command(NetCommand::Fetch { peer, branch });
    }
}

/// The async event loop running inside the Host thread.
async fn host_loop(
    secret: iroh_base::SecretKey,
    config: HostConfig,
    commands: mpsc::Receiver<NetCommand>,
    events: mpsc::Sender<NetEvent>,
    reader: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
) {
    use iroh::endpoint::presets;
    use iroh::protocol::{AcceptError, ProtocolHandler, Router};
    use iroh::Endpoint;
    use iroh_gossip::Gossip;
    use iroh_gossip::api::GossipSender;
    use futures::TryStreamExt;

    // Bind endpoint.
    let ep = match Endpoint::builder(presets::N0).secret_key(secret).bind().await {
        Ok(ep) => ep,
        Err(e) => { eprintln!("host: bind failed: {e}"); return; }
    };
    ep.online().await;

    let my_id = ep.id();
    let mut router_builder = Router::builder(ep.clone());

    // Protocol handler: serves from the shared reader.
    let handler = ReaderHandler { reader: reader.clone() };
    router_builder = router_builder.accept(PILE_SYNC_ALPN, handler);

    // Gossip.
    let mut gossip_sender: Option<GossipSender> = None;
    if let Some(ref topic_name) = config.gossip_topic {
        let gossip = Gossip::builder().spawn(ep.clone());
        router_builder = router_builder.accept(iroh_gossip::ALPN, gossip.clone());

        let topic_id = iroh_gossip::TopicId::from_bytes(
            *blake3::hash(topic_name.as_bytes()).as_bytes()
        );
        let topic = if config.gossip_peers.is_empty() {
            gossip.subscribe(topic_id, config.gossip_peers.clone()).await
        } else {
            gossip.subscribe_and_join(topic_id, config.gossip_peers.clone()).await
        };
        if let Ok(topic) = topic {
            let (sender, receiver) = topic.split();
            gossip_sender = Some(sender);

            let events_tx = events.clone();
            tokio::spawn(async move {
                let mut receiver = receiver;
                while let Ok(Some(event)) = receiver.try_next().await {
                    if let iroh_gossip::api::Event::Received(msg) = event {
                        if msg.content.len() == 49 && msg.content[0] == 0x01 {
                            let mut branch = [0u8; 16];
                            branch.copy_from_slice(&msg.content[1..17]);
                            let mut head = [0u8; 32];
                            head.copy_from_slice(&msg.content[17..49]);
                            let _ = events_tx.send(NetEvent::Head { branch, head });
                        }
                    }
                }
            });
        }
    }

    // DHT.
    if !config.dht_bootstrap.is_empty() {
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
        let (rpc, _api) = iroh_dht::create_node(
            my_id, iroh_pool.clone(), config.dht_bootstrap, Default::default(),
        );
        iroh_pool.set_self_client(Some(rpc.downgrade()));
        let dht_sender = rpc.inner().as_local().expect("local sender");
        router_builder = router_builder
            .accept(dht_alpn, irpc_iroh::IrohProtocol::with_sender(dht_sender));
    }

    let _router = router_builder.spawn();

    // Command loop.
    loop {
        while let Ok(cmd) = commands.try_recv() {
            match cmd {
                NetCommand::Announce(_hash) => {
                    // TODO: DHT announce via _api
                }
                NetCommand::Gossip { branch, head } => {
                    if let Some(ref sender) = gossip_sender {
                        let mut msg = Vec::with_capacity(49);
                        msg.push(0x01);
                        msg.extend_from_slice(&branch);
                        msg.extend_from_slice(&head);
                        let sender = sender.clone();
                        tokio::spawn(async move {
                            let _ = sender.broadcast(msg.into()).await;
                        });
                    }
                }
                NetCommand::Fetch { peer, branch } => {
                    let ep = ep.clone();
                    let events_tx = events.clone();
                    tokio::spawn(async move {
                        if let Err(e) = fetch_from_peer(&ep, peer, &branch, &events_tx).await {
                            eprintln!("host: fetch error: {e}");
                        }
                    });
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Fetch blobs reachable from a remote branch HEAD.
async fn fetch_from_peer(
    ep: &iroh::Endpoint,
    peer: EndpointId,
    branch: &RawBranchId,
    events: &mpsc::Sender<NetEvent>,
) -> anyhow::Result<()> {
    let conn = ep.connect(peer, PILE_SYNC_ALPN).await
        .map_err(|e| anyhow::anyhow!("connect: {e}"))?;

    let Some(head) = op_head(&conn, branch).await? else {
        return Ok(());
    };

    let mut seen: HashSet<RawHash> = HashSet::new();
    seen.insert(head);

    if let Some(data) = op_get_blob(&conn, &head).await? {
        let _ = events.send(NetEvent::Blob(data));
    }

    let mut current_level = vec![head];
    while !current_level.is_empty() {
        let mut next_level = Vec::new();
        for parent in &current_level {
            let children = op_children(&conn, parent).await?;
            for hash in children {
                if !seen.insert(hash) { continue; }
                if let Some(data) = op_get_blob(&conn, &hash).await? {
                    let _ = events.send(NetEvent::Blob(data));
                    next_level.push(hash);
                }
            }
        }
        current_level = next_level;
    }

    let _ = events.send(NetEvent::Head { branch: *branch, head });
    conn.close(0u32.into(), b"done");
    Ok(())
}

// ── Protocol handler serving from shared reader ─────────────────────

#[derive(Clone)]
struct ReaderHandler {
    reader: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
}

impl std::fmt::Debug for ReaderHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderHandler").finish()
    }
}

impl iroh::protocol::ProtocolHandler for ReaderHandler {
    async fn accept(&self, connection: iroh::endpoint::Connection) -> Result<(), iroh::protocol::AcceptError> {
        let reader_arc = self.reader.clone();
        loop {
            let (mut send, mut recv) = match connection.accept_bi().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            let reader_arc = reader_arc.clone();
            tokio::spawn(async move {
                if let Err(e) = serve_from_reader(&reader_arc, &mut send, &mut recv).await {
                    eprintln!("handler error: {e}");
                }
                let _ = send.finish();
            });
        }
        Ok(())
    }
}

async fn serve_from_reader(
    reader_arc: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> anyhow::Result<()> {
    let op = recv_u8(recv).await?;

    match op {
        OP_LIST => {
            let branches = reader_arc.lock().unwrap().as_ref()
                .map(|r| r.list_branches().to_vec())
                .unwrap_or_default();
            for (id, head) in &branches {
                send_branch_id(send, id).await?;
                send_hash(send, head).await?;
            }
            send_branch_id(send, &NIL_BRANCH_ID).await?;
        }

        OP_HEAD => {
            let id_bytes = recv_branch_id(recv).await?;
            let hash = reader_arc.lock().unwrap().as_ref()
                .and_then(|r| r.head(&id_bytes))
                .unwrap_or(NIL_HASH);
            send_hash(send, &hash).await?;
        }

        OP_GET_BLOB => {
            let hash = recv_hash(recv).await?;
            // Lock, read, unlock BEFORE any await.
            let data = reader_arc.lock().unwrap().as_ref()
                .and_then(|r| r.get_blob(&hash));
            match data {
                Some(data) => {
                    send_u64_be(send, data.len() as u64).await?;
                    send.write_all(&data).await.map_err(|e| anyhow::anyhow!("send: {e}"))?;
                }
                None => send_u64_be(send, u64::MAX).await?,
            }
        }

        OP_CHILDREN => {
            let parent_hash = recv_hash(recv).await?;
            // Lock, collect children, unlock BEFORE any await.
            let children: Vec<RawHash> = {
                let guard = reader_arc.lock().unwrap();
                match guard.as_ref() {
                    None => Vec::new(),
                    Some(reader) => {
                        match reader.get_blob(&parent_hash) {
                            None => Vec::new(),
                            Some(parent_data) => {
                                let mut result = Vec::new();
                                for chunk in parent_data.chunks(32) {
                                    if chunk.len() == 32 {
                                        let mut candidate = [0u8; 32];
                                        candidate.copy_from_slice(chunk);
                                        if reader.has_blob(&candidate) {
                                            result.push(candidate);
                                        }
                                    }
                                }
                                result
                            }
                        }
                    }
                }
            };
            for hash in &children {
                send_hash(send, hash).await?;
            }
            send_hash(send, &NIL_HASH).await?;
        }

        _ => {}
    }
    Ok(())
}
