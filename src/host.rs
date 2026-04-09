//! Network thread: spawns iroh endpoint, gossip, DHT, protocol server.
//!
//! `spawn()` returns two halves:
//! - `HostSender`: for Leader (commands + snapshot updates)
//! - `HostReceiver`: for Follower (events)
//!
//! Async is jailed inside the spawned thread.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use iroh_base::EndpointId;
use ed25519_dalek::SigningKey;

use crate::channel::{NetCommand, NetEvent};
use crate::identity::iroh_secret;
use crate::protocol::*;

/// Configuration for the host thread.
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
pub struct StoreSnapshot<R> {
    pub reader: R,
    pub branches: Vec<(RawBranchId, RawHash)>,
}

impl StoreSnapshot<()> {
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

/// Type-erased snapshot for the host thread.
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

// ── Leader's half ────────────────────────────────────────────────────

/// Send commands to the host thread + update the serving snapshot.
#[derive(Clone)]
pub struct HostSender {
    cmd_tx: mpsc::Sender<NetCommand>,
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    id: EndpointId,
}

impl HostSender {
    pub fn id(&self) -> EndpointId { self.id }

    pub fn announce(&self, hash: RawHash) {
        let _ = self.cmd_tx.send(NetCommand::Announce(hash));
    }

    pub fn gossip(&self, branch: RawBranchId, head: RawHash) {
        let _ = self.cmd_tx.send(NetCommand::Gossip { branch, head });
    }

    pub fn fetch(&self, peer: EndpointId, branch: RawBranchId) {
        let _ = self.cmd_tx.send(NetCommand::Fetch { peer, branch });
    }

    pub fn update_snapshot(&self, snapshot: impl AnySnapshot) {
        *self.snapshot.lock().unwrap() = Some(Box::new(snapshot));
    }
}

// ── Follower's half ──────────────────────────────────────────────────

/// Receive events from the host thread.
pub struct HostReceiver {
    evt_rx: mpsc::Receiver<NetEvent>,
    id: EndpointId,
}

impl HostReceiver {
    pub fn id(&self) -> EndpointId { self.id }

    pub fn try_recv(&self) -> Option<NetEvent> {
        self.evt_rx.try_recv().ok()
    }
}

// ── Spawn ────────────────────────────────────────────────────────────

/// Spawn the host thread. Returns sender (for Leader) and receiver (for Follower).
pub fn spawn(key: SigningKey, config: HostConfig) -> (HostSender, HostReceiver) {
    let secret = iroh_secret(&key);
    let id: EndpointId = secret.public().into();

    let (cmd_tx, cmd_rx) = mpsc::channel::<NetCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<NetEvent>();

    let snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
        Arc::new(Mutex::new(None));
    let thread_snapshot = snapshot.clone();

    let _thread = thread::spawn(move || {
        eprintln!("[host-thread] starting");
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        eprintln!("[host-thread] runtime created, entering host_loop");
        rt.block_on(host_loop(secret, config, cmd_rx, evt_tx, thread_snapshot));
        eprintln!("[host-thread] host_loop returned");
    });

    let sender = HostSender { cmd_tx, snapshot, id };
    let receiver = HostReceiver { evt_rx, id };
    (sender, receiver)
}

// ── Host thread event loop ───────────────────────────────────────────

async fn host_loop(
    secret: iroh_base::SecretKey,
    config: HostConfig,
    commands: mpsc::Receiver<NetCommand>,
    events: mpsc::Sender<NetEvent>,
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
) {
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
    use iroh::Endpoint;
    use iroh_gossip::Gossip;
    use iroh_gossip::api::GossipSender;
    use futures::TryStreamExt;

    eprintln!("[host] binding endpoint...");
    let ep = match Endpoint::builder(presets::N0).secret_key(secret).bind().await {
        Ok(ep) => ep,
        Err(e) => { eprintln!("host: bind failed: {e}"); return; }
    };
    eprintln!("[host] bound, waiting for online...");
    ep.online().await;
    eprintln!("[host] online!");

    let my_id = ep.id();
    let mut router_builder = Router::builder(ep.clone());

    // Protocol handler.
    let handler = SnapshotHandler { snapshot: snapshot.clone() };
    router_builder = router_builder.accept(PILE_SYNC_ALPN, handler);

    // DHT (before gossip — gossip tasks need DHT API).
    let mut dht_api: Option<iroh_dht::api::ApiClient> = None;
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
        let (rpc, api) = iroh_dht::create_node(
            my_id, iroh_pool.clone(), config.dht_bootstrap, Default::default(),
        );
        iroh_pool.set_self_client(Some(rpc.downgrade()));
        let dht_sender = rpc.inner().as_local().expect("local sender");
        router_builder = router_builder
            .accept(dht_alpn, irpc_iroh::IrohProtocol::with_sender(dht_sender));
        dht_api = Some(api);
    }

    // Gossip.
    let mut gossip_sender: Option<GossipSender> = None;
    if let Some(ref topic_name) = config.gossip_topic {
        let gossip = Gossip::builder().spawn(ep.clone());
        router_builder = router_builder.accept(iroh_gossip::ALPN, gossip.clone());

        let topic_id = iroh_gossip::TopicId::from_bytes(
            *blake3::hash(topic_name.as_bytes()).as_bytes()
        );
        // Always use subscribe (non-blocking). The join happens in the background
        // as peers come online. subscribe_and_join blocks until at least one peer
        // is reachable, which causes hangs if peers start at different times.
        let topic = gossip.subscribe(topic_id, config.gossip_peers.clone()).await;
        eprintln!("[host] gossip subscribe result: {}", topic.is_ok());
        if let Ok(topic) = topic {
            let (sender, receiver) = topic.split();
            gossip_sender = Some(sender);
            let events_tx = events.clone();
            let ep2 = ep.clone();
            let dht_api2 = dht_api.clone();
            tokio::spawn(async move {
                let mut receiver = receiver;
                eprintln!("[host] gossip receiver started");
                while let Ok(Some(event)) = receiver.try_next().await {
                    match &event {
                        iroh_gossip::api::Event::Received(msg) => {
                            eprintln!("[host] gossip received: {} bytes from {}", msg.content.len(), msg.delivered_from.fmt_short());
                            if msg.content.len() == 49 && msg.content[0] == 0x01 {
                                let mut branch = [0u8; 16];
                                branch.copy_from_slice(&msg.content[1..17]);
                                let mut head = [0u8; 32];
                                head.copy_from_slice(&msg.content[17..49]);
                                // Fetch blobs THEN send HEAD event.
                                let ep2 = ep2.clone();
                                let events_tx2 = events_tx.clone();
                                let dht2 = dht_api2.clone();
                                let from: iroh_base::EndpointId = msg.delivered_from.into();
                                tokio::spawn(async move {
                                    eprintln!("[host] fetching blobs for HEAD {} from {}", hex::encode(&head[..4]), from.fmt_short());
                                    if let Err(e) = fetch_reachable(&ep2, from, &head, &dht2, &events_tx2).await {
                                        eprintln!("[host] fetch error: {e}");
                                    } else {
                                        let _ = events_tx2.send(NetEvent::Head { branch, head });
                                    }
                                });
                            }
                        }
                        iroh_gossip::api::Event::NeighborUp(peer) => {
                            eprintln!("[host] gossip neighbor up: {}", peer.fmt_short());
                        }
                        iroh_gossip::api::Event::NeighborDown(peer) => {
                            eprintln!("[host] gossip neighbor down: {}", peer.fmt_short());
                        }
                        _ => {}
                    }
                }
                eprintln!("[host] gossip receiver ended");
            });
        }
    }

    let _router = router_builder.spawn();

    eprintln!("[host] entering command loop");
    // Command loop.
    loop {
        while let Ok(cmd) = commands.try_recv() {
            match cmd {
                NetCommand::Announce(hash) => {
                    if let Some(ref api) = dht_api {
                        let api = api.clone();
                        tokio::spawn(async move {
                            let blake3_hash = blake3::Hash::from_bytes(hash);
                            let _ = api.announce_provider(blake3_hash, my_id).await;
                        });
                    }
                }
                NetCommand::Gossip { branch, head } => {
                    if let Some(ref sender) = gossip_sender {
                        let mut msg = Vec::with_capacity(49);
                        msg.push(0x01);
                        msg.extend_from_slice(&branch);
                        msg.extend_from_slice(&head);
                        eprintln!("[host] broadcasting gossip HEAD {}", hex::encode(&head[..4]));
                        let sender = sender.clone();
                        tokio::spawn(async move {
                            match sender.broadcast(msg.into()).await {
                                Ok(()) => eprintln!("[host] gossip broadcast ok"),
                                Err(e) => eprintln!("[host] gossip broadcast error: {e}"),
                            }
                        });
                    } else {
                        eprintln!("[host] gossip command but no sender configured");
                    }
                }
                NetCommand::Fetch { peer, branch } => {
                    let ep = ep.clone();
                    let events_tx = events.clone();
                    let dht = dht_api.clone();
                    tokio::spawn(async move {
                        // Get remote HEAD first.
                        let conn = match ep.connect(peer, PILE_SYNC_ALPN).await {
                            Ok(c) => c,
                            Err(e) => { eprintln!("host: connect: {e}"); return; }
                        };
                        let head = match op_head(&conn, &branch).await {
                            Ok(Some(h)) => h,
                            Ok(None) => { eprintln!("host: no head"); return; }
                            Err(e) => { eprintln!("host: head: {e}"); return; }
                        };
                        conn.close(0u32.into(), b"ok");
                        // Fetch blobs (DHT first, peer fallback).
                        if let Err(e) = fetch_reachable(&ep, peer, &head, &dht, &events_tx).await {
                            eprintln!("host: fetch error: {e}");
                        } else {
                            let _ = events_tx.send(NetEvent::Head { branch, head });
                        }
                    });
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Fetch a single blob by hash: try DHT providers first, fall back to a direct peer.
async fn fetch_blob(
    ep: &iroh::Endpoint,
    hash: &RawHash,
    dht: &Option<iroh_dht::api::ApiClient>,
    fallback_peer: EndpointId,
) -> anyhow::Result<Option<Vec<u8>>> {
    // Try DHT first.
    if let Some(ref api) = dht {
        let blake3_hash = blake3::Hash::from_bytes(*hash);
        if let Ok(providers) = api.find_providers(blake3_hash).await {
            for provider in providers {
                if let Ok(conn) = ep.connect(provider, PILE_SYNC_ALPN).await {
                    if let Ok(Some(data)) = op_get_blob(&conn, hash).await {
                        conn.close(0u32.into(), b"ok");
                        return Ok(Some(data));
                    }
                }
            }
        }
    }

    // Fallback: fetch from the gossip sender directly.
    let conn = ep.connect(fallback_peer, PILE_SYNC_ALPN).await
        .map_err(|e| anyhow::anyhow!("connect fallback: {e}"))?;
    let result = op_get_blob(&conn, hash).await?;
    conn.close(0u32.into(), b"ok");
    Ok(result)
}

/// Fetch all blobs reachable from a remote HEAD.
/// Uses DHT for blob discovery when available, falls back to direct peer.
async fn fetch_reachable(
    ep: &iroh::Endpoint,
    peer: EndpointId,
    head: &RawHash,
    dht: &Option<iroh_dht::api::ApiClient>,
    events: &mpsc::Sender<NetEvent>,
) -> anyhow::Result<()> {
    let mut seen: HashSet<RawHash> = HashSet::new();
    seen.insert(*head);

    // Fetch head blob.
    if let Some(data) = fetch_blob(ep, head, dht, peer).await? {
        let _ = events.send(NetEvent::Blob(data));
    }

    // BFS: use CHILDREN from the peer for structure, DHT for blob data.
    let mut current_level = vec![*head];
    while !current_level.is_empty() {
        let mut next_level = Vec::new();
        for parent in &current_level {
            // CHILDREN from the gossip sender (they know the structure).
            let conn = ep.connect(peer, PILE_SYNC_ALPN).await
                .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
            let children = op_children(&conn, parent).await?;
            conn.close(0u32.into(), b"ok");

            for hash in children {
                if !seen.insert(hash) { continue; }
                if let Some(data) = fetch_blob(ep, &hash, dht, peer).await? {
                    let _ = events.send(NetEvent::Blob(data));
                    next_level.push(hash);
                }
            }
        }
        current_level = next_level;
    }

    Ok(())
}

// ── Protocol handler ─────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
}

impl std::fmt::Debug for SnapshotHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotHandler").finish()
    }
}

impl iroh::protocol::ProtocolHandler for SnapshotHandler {
    async fn accept(&self, connection: iroh::endpoint::Connection) -> Result<(), iroh::protocol::AcceptError> {
        let snap = self.snapshot.clone();
        loop {
            let (mut send, mut recv) = match connection.accept_bi().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            let snap = snap.clone();
            tokio::spawn(async move {
                if let Err(e) = serve_from_snapshot(&snap, &mut send, &mut recv).await {
                    eprintln!("handler error: {e}");
                }
                let _ = send.finish();
            });
        }
        Ok(())
    }
}

async fn serve_from_snapshot(
    snap_arc: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> anyhow::Result<()> {
    let op = recv_u8(recv).await?;

    match op {
        OP_LIST => {
            let branches = snap_arc.lock().unwrap().as_ref()
                .map(|s| s.list_branches().to_vec())
                .unwrap_or_default();
            for (id, head) in &branches {
                send_branch_id(send, id).await?;
                send_hash(send, head).await?;
            }
            send_branch_id(send, &NIL_BRANCH_ID).await?;
        }

        OP_HEAD => {
            let id_bytes = recv_branch_id(recv).await?;
            let hash = snap_arc.lock().unwrap().as_ref()
                .and_then(|s| s.head(&id_bytes))
                .unwrap_or(NIL_HASH);
            send_hash(send, &hash).await?;
        }

        OP_GET_BLOB => {
            let hash = recv_hash(recv).await?;
            let data = snap_arc.lock().unwrap().as_ref()
                .and_then(|s| s.get_blob(&hash));
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
            let children: Vec<RawHash> = {
                let guard = snap_arc.lock().unwrap();
                match guard.as_ref() {
                    None => Vec::new(),
                    Some(snap) => {
                        match snap.get_blob(&parent_hash) {
                            None => Vec::new(),
                            Some(parent_data) => {
                                let mut result = Vec::new();
                                for chunk in parent_data.chunks(32) {
                                    if chunk.len() == 32 {
                                        let mut candidate = [0u8; 32];
                                        candidate.copy_from_slice(chunk);
                                        if snap.has_blob(&candidate) {
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
