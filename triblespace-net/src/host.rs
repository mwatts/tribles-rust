//! Network thread: spawns iroh endpoint, gossip, DHT, protocol server.
//!
//! Private implementation detail of [`crate::peer::Peer`] — `spawn()`
//! returns the [`NetSender`] / [`NetReceiver`] pair the Peer uses to
//! communicate with the async world (commands + snapshot updates one
//! way, events the other).
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
pub struct PeerConfig {
    /// Peers to connect to (used for both gossip and DHT bootstrap).
    pub peers: Vec<EndpointId>,
    /// Gossip topic name (None = no gossip, serve-only).
    pub gossip_topic: Option<String>,
    /// The team root public key — verifies all incoming capability
    /// chains. Every connection's first stream must present a cap that
    /// chains back to this key. See `triblespace_core::repo::capability`.
    pub team_root: ed25519_dalek::VerifyingKey,
    /// Pubkeys whose capabilities are revoked. Cascades transitively
    /// through the chain.
    pub revoked: std::collections::HashSet<ed25519_dalek::VerifyingKey>,
    /// This node's own capability sig handle. Presented to remote peers
    /// as the first stream on every outgoing connection so they can
    /// authorise us. Required — protocol v4 has mandatory auth on both
    /// directions of a connection.
    pub self_cap: RawHash,
}

// No `Default` impl: every PeerConfig must specify a team root because
// auth is mandatory in protocol v4. For a single-user OSS deployment
// the convention is `team_root = signing_key.verifying_key()` (the user
// is the team root and the founder of a team-of-one).

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

// ── Outgoing half ────────────────────────────────────────────────────

/// Send commands to the host thread + update the serving snapshot.
#[derive(Clone)]
pub struct NetSender {
    cmd_tx: mpsc::Sender<NetCommand>,
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    id: EndpointId,
}

impl NetSender {
    pub fn id(&self) -> EndpointId { self.id }

    pub fn announce(&self, hash: RawHash) {
        let _ = self.cmd_tx.send(NetCommand::Announce(hash));
    }

    pub fn gossip(&self, branch: RawBranchId, head: RawHash) {
        let _ = self.cmd_tx.send(NetCommand::Gossip { branch, head });
    }

    pub fn track(&self, peer: EndpointId, branch: RawBranchId) {
        let _ = self.cmd_tx.send(NetCommand::Track { peer, branch });
    }

    /// RPC: list a remote peer's branches. Blocks the calling thread until
    /// the network thread completes one protocol round trip.
    pub fn list_remote_branches(
        &self,
        peer: EndpointId,
    ) -> anyhow::Result<Vec<(triblespace_core::id::Id, RawHash)>> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(NetCommand::ListBranches { peer, reply: tx })
            .map_err(|_| anyhow::anyhow!("network thread dropped"))?;
        rx.recv().map_err(|_| anyhow::anyhow!("network thread dropped"))?
    }

    /// RPC: query a remote peer for its current head of one branch.
    pub fn head_of_remote(
        &self,
        peer: EndpointId,
        branch: RawBranchId,
    ) -> anyhow::Result<Option<RawHash>> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(NetCommand::HeadOfRemote { peer, branch, reply: tx })
            .map_err(|_| anyhow::anyhow!("network thread dropped"))?;
        rx.recv().map_err(|_| anyhow::anyhow!("network thread dropped"))?
    }

    /// RPC: fetch a single blob's bytes from a remote peer. Returns the
    /// raw bytes (or `None` if the remote doesn't have the blob); the
    /// caller is responsible for putting them into a local store.
    pub fn fetch(
        &self,
        peer: EndpointId,
        hash: RawHash,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(NetCommand::Fetch { peer, hash, reply: tx })
            .map_err(|_| anyhow::anyhow!("network thread dropped"))?;
        rx.recv().map_err(|_| anyhow::anyhow!("network thread dropped"))?
    }

    pub fn update_snapshot(&self, snapshot: impl AnySnapshot) {
        *self.snapshot.lock().unwrap() = Some(Box::new(snapshot));
    }
}

// ── Incoming half ────────────────────────────────────────────────────

/// Receive events from the network thread.
pub struct NetReceiver {
    evt_rx: mpsc::Receiver<NetEvent>,
}

impl NetReceiver {
    pub fn try_recv(&self) -> Option<NetEvent> {
        self.evt_rx.try_recv().ok()
    }
}

// ── Spawn ────────────────────────────────────────────────────────────

/// Spawn the network thread. Returns the outgoing/incoming channel halves
/// — used internally by [`Peer::new`](crate::peer::Peer::new).
pub fn spawn(key: SigningKey, config: PeerConfig) -> (NetSender, NetReceiver) {
    let secret = iroh_secret(&key);
    let id: EndpointId = secret.public().into();

    let (cmd_tx, cmd_rx) = mpsc::channel::<NetCommand>();
    let (evt_tx, evt_rx) = mpsc::channel::<NetEvent>();

    let snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
        Arc::new(Mutex::new(None));
    let thread_snapshot = snapshot.clone();

    let _thread = thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(host_loop(secret, config, cmd_rx, evt_tx, thread_snapshot));
    });

    let sender = NetSender { cmd_tx, snapshot, id };
    let receiver = NetReceiver { evt_rx };
    (sender, receiver)
}

// ── Network thread event loop ────────────────────────────────────────

/// Connect to a peer over the pile-sync ALPN and immediately present
/// our capability so subsequent ops are authorised. Protocol v4 makes
/// this mandatory — the server rejects any op until the connection
/// completes auth.
async fn connect_authed(
    ep: &iroh::Endpoint,
    peer: EndpointId,
    self_cap: &RawHash,
) -> anyhow::Result<iroh::endpoint::Connection> {
    let conn = ep.connect(peer, PILE_SYNC_ALPN).await
        .map_err(|e| anyhow::anyhow!("connect: {e}"))?;
    op_auth(&conn, self_cap).await
        .map_err(|e| anyhow::anyhow!("auth: {e}"))?;
    Ok(conn)
}

async fn host_loop(
    secret: iroh_base::SecretKey,
    config: PeerConfig,
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

    let ep = match Endpoint::builder(presets::N0).secret_key(secret).bind().await {
        Ok(ep) => ep,
        Err(e) => { eprintln!("[net] bind failed: {e}"); return; }
    };
    ep.online().await;

    let my_id = ep.id();
    let self_cap: RawHash = config.self_cap;
    let mut router_builder = Router::builder(ep.clone());

    // Protocol handler.
    let revoked_lock = Arc::new(tokio::sync::RwLock::new(config.revoked.clone()));
    let handler = SnapshotHandler {
        snapshot: snapshot.clone(),
        team_root: config.team_root,
        revoked: revoked_lock,
    };
    router_builder = router_builder.accept(PILE_SYNC_ALPN, handler);

    // DHT — always on. Peers bootstrap the routing table.
    let dht_alpn = crate::dht::rpc::ALPN;
    let pool = iroh_blobs::util::connection_pool::ConnectionPool::new(
        ep.clone(), dht_alpn,
        iroh_blobs::util::connection_pool::Options {
            max_connections: 64,
            idle_timeout: std::time::Duration::from_secs(30),
            connect_timeout: std::time::Duration::from_secs(10),
            on_connected: None,
        },
    );
    let iroh_pool = crate::dht::pool::IrohPool::new(ep.clone(), pool);
    let (rpc, dht_api) = crate::dht::create_node(
        my_id, iroh_pool.clone(), config.peers.clone(), Default::default(),
    );
    iroh_pool.set_self_client(Some(rpc.downgrade()));
    let dht_sender = rpc.inner().as_local().expect("local sender");
    router_builder = router_builder
        .accept(dht_alpn, irpc_iroh::IrohProtocol::with_sender(dht_sender));
    let dht_api = Some(dht_api);

    // Gossip.
    let mut gossip_sender: Option<GossipSender> = None;
    if let Some(topic_name) = config.gossip_topic {
        let gossip = Gossip::builder().spawn(ep.clone());
        router_builder = router_builder.accept(iroh_gossip::ALPN, gossip.clone());

        let topic_id = iroh_gossip::TopicId::from_bytes(
            *blake3::hash(topic_name.as_bytes()).as_bytes()
        );
        // Always use subscribe (non-blocking). The join happens in the background
        // as peers come online. subscribe_and_join blocks until at least one peer
        // is reachable, which causes hangs if peers start at different times.
        let topic = gossip.subscribe(topic_id, config.peers.clone()).await;
        if let Ok(topic) = topic {
            let (sender, receiver) = topic.split();
            gossip_sender = Some(sender);
            let events_tx = events.clone();
            let ep2 = ep.clone();
            let dht_api2 = dht_api.clone();
            tokio::spawn(async move {
                let mut receiver = receiver;
                while let Ok(Some(event)) = receiver.try_next().await {
                    match &event {
                        iroh_gossip::api::Event::Received(msg) => {
                            // Gossip HEAD message: 0x01 + branch(16) + head(32) + publisher(32) = 81 bytes
                            if msg.content.len() == 81 && msg.content[0] == 0x01 {
                                let mut branch = [0u8; 16];
                                branch.copy_from_slice(&msg.content[1..17]);
                                let mut head = [0u8; 32];
                                head.copy_from_slice(&msg.content[17..49]);
                                let mut publisher = [0u8; 32];
                                publisher.copy_from_slice(&msg.content[49..81]);

                                let ep2 = ep2.clone();
                                let events_tx2 = events_tx.clone();
                                let dht2 = dht_api2.clone();
                                let self_cap2 = self_cap;
                                // Use publisher key to connect for fetch (they're the source).
                                let fetch_peer = if let Ok(pk) = iroh_base::PublicKey::from_bytes(&publisher) {
                                    pk.into()
                                } else {
                                    msg.delivered_from.into()
                                };
                                tokio::spawn(async move {
                                    eprintln!("[net] fetching HEAD {} from publisher {}", hex::encode(&head[..4]), hex::encode(&publisher[..4]));
                                    track_known_head(&ep2, fetch_peer, branch, head, publisher, &dht2, &events_tx2, &self_cap2).await;
                                });
                            }
                        }
                        iroh_gossip::api::Event::NeighborUp(peer) => {
                            eprintln!("[net] gossip neighbor up: {}", peer.fmt_short());
                        }
                        iroh_gossip::api::Event::NeighborDown(peer) => {
                            eprintln!("[net] gossip neighbor down: {}", peer.fmt_short());
                        }
                        _ => {}
                    }
                }
            });
        }
    }

    let _router = router_builder.spawn();

    // Command loop.
    loop {
        while let Ok(cmd) = commands.try_recv() {
            match cmd {
                NetCommand::Announce(hash) => {
                    if let Some(api) = &dht_api {
                        let api = api.clone();
                        tokio::spawn(async move {
                            let blake3_hash = blake3::Hash::from_bytes(hash);
                            let _ = api.announce_provider(blake3_hash, my_id).await;
                        });
                    }
                }
                NetCommand::Gossip { branch, head } => {
                    if let Some(sender) = &gossip_sender {
                        let mut msg = Vec::with_capacity(81);
                        msg.push(0x01);
                        msg.extend_from_slice(&branch);
                        msg.extend_from_slice(&head);
                        msg.extend_from_slice(my_id.as_bytes());
                        let sender = sender.clone();
                        tokio::spawn(async move {
                            let _ = sender.broadcast(msg.into()).await;
                        });
                    }
                }
                NetCommand::Track { peer, branch } => {
                    let ep = ep.clone();
                    let events_tx = events.clone();
                    let dht = dht_api.clone();
                    let self_cap = self_cap;
                    tokio::spawn(async move {
                        // Discover the remote HEAD (gossip would have it for
                        // free; explicit track has to ask).
                        let conn = match connect_authed(&ep, peer, &self_cap).await {
                            Ok(c) => c,
                            Err(e) => { eprintln!("[net] connect: {e}"); return; }
                        };
                        let head = match op_head(&conn, &branch).await {
                            Ok(Some(h)) => h,
                            Ok(None) => { eprintln!("[net] no head"); return; }
                            Err(e) => { eprintln!("[net] head: {e}"); return; }
                        };
                        conn.close(0u32.into(), b"ok");
                        // For explicit track, the publisher is the peer
                        // we asked (they vouched for this head).
                        let mut publisher = [0u8; 32];
                        publisher.copy_from_slice(peer.as_bytes());
                        track_known_head(&ep, peer, branch, head, publisher, &dht, &events_tx, &self_cap).await;
                    });
                }
                NetCommand::ListBranches { peer, reply } => {
                    let ep = ep.clone();
                    let self_cap = self_cap;
                    tokio::spawn(async move {
                        let result = async {
                            let conn = connect_authed(&ep, peer, &self_cap).await?;
                            let pairs = op_list(&conn).await?;
                            conn.close(0u32.into(), b"ok");
                            let out: Vec<(triblespace_core::id::Id, RawHash)> = pairs
                                .into_iter()
                                .filter_map(|(bid, head)| {
                                    triblespace_core::id::Id::new(bid).map(|id| (id, head))
                                })
                                .collect();
                            Ok(out)
                        }.await;
                        let _ = reply.send(result);
                    });
                }
                NetCommand::HeadOfRemote { peer, branch, reply } => {
                    let ep = ep.clone();
                    let self_cap = self_cap;
                    tokio::spawn(async move {
                        let result = async {
                            let conn = connect_authed(&ep, peer, &self_cap).await?;
                            let head = op_head(&conn, &branch).await?;
                            conn.close(0u32.into(), b"ok");
                            Ok(head)
                        }.await;
                        let _ = reply.send(result);
                    });
                }
                NetCommand::Fetch { peer, hash, reply } => {
                    let ep = ep.clone();
                    let dht = dht_api.clone();
                    let self_cap = self_cap;
                    tokio::spawn(async move {
                        let result = fetch_blob(&ep, &hash, &dht, peer, &self_cap).await;
                        let _ = reply.send(result);
                    });
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Fetch a single blob by hash from any available source.
/// Tries DHT providers, then the hint peer. Verifies blake3 hash before returning.
async fn fetch_blob(
    ep: &iroh::Endpoint,
    hash: &RawHash,
    dht: &Option<crate::dht::api::ApiClient>,
    hint_peer: EndpointId,
    self_cap: &RawHash,
) -> anyhow::Result<Option<Vec<u8>>> {
    let verify = |data: &[u8]| -> bool {
        let computed = blake3::hash(data);
        computed.as_bytes() == hash
    };

    // DHT: ask the network who has this blob.
    if let Some(api) = dht {
        let blake3_hash = blake3::Hash::from_bytes(*hash);
        if let Ok(providers) = api.find_providers(blake3_hash).await {
            for provider in providers {
                if let Ok(conn) = connect_authed(ep, provider, self_cap).await {
                    if let Ok(Some(data)) = op_get_blob(&conn, hash).await {
                        conn.close(0u32.into(), b"ok");
                        if verify(&data) {
                            return Ok(Some(data));
                        }
                        eprintln!("[net] hash mismatch from DHT provider {}", provider.fmt_short());
                    }
                }
            }
        }
    }

    // Hint peer: the gossip sender likely has it.
    if let Ok(conn) = connect_authed(ep, hint_peer, self_cap).await {
        if let Ok(Some(data)) = op_get_blob(&conn, hash).await {
            conn.close(0u32.into(), b"ok");
            if verify(&data) {
                return Ok(Some(data));
            }
            eprintln!("[net] hash mismatch from hint peer {}", hint_peer.fmt_short());
        }
    }

    Ok(None)
}

/// Fetch all blobs reachable from a remote HEAD.
/// Uses DHT for blob discovery when available, falls back to direct peer.
async fn fetch_reachable(
    ep: &iroh::Endpoint,
    peer: EndpointId,
    head: &RawHash,
    dht: &Option<crate::dht::api::ApiClient>,
    events: &mpsc::Sender<NetEvent>,
    self_cap: &RawHash,
) -> anyhow::Result<()> {
    let mut seen: HashSet<RawHash> = HashSet::new();
    seen.insert(*head);

    // Fetch head blob.
    if let Some(data) = fetch_blob(ep, head, dht, peer, self_cap).await? {
        let _ = events.send(NetEvent::Blob(data));
    }

    // BFS: use CHILDREN from the peer for structure, DHT for blob data.
    let mut current_level = vec![*head];
    while !current_level.is_empty() {
        let mut next_level = Vec::new();
        for parent in &current_level {
            // CHILDREN from the gossip sender (they know the structure).
            let conn = connect_authed(ep, peer, self_cap).await?;
            let children = op_children(&conn, parent).await?;
            conn.close(0u32.into(), b"ok");

            for hash in children {
                if !seen.insert(hash) { continue; }
                if let Some(data) = fetch_blob(ep, &hash, dht, peer, self_cap).await? {
                    let _ = events.send(NetEvent::Blob(data));
                    next_level.push(hash);
                }
            }
        }
        current_level = next_level;
    }

    Ok(())
}

/// Fetch the reachable closure from `head` on `fetch_peer` and, on
/// success, emit a [`NetEvent::Head`] so the Peer materializes a
/// tracking branch.
///
/// Shared tail of the gossip-arrival handler and the `Track` command:
/// both know (fetch_peer, branch, head, publisher) by the time they
/// get here. Gossip gets the head directly from the broadcast message;
/// `Track` asks the peer via `op_head` first.
async fn track_known_head(
    ep: &iroh::Endpoint,
    fetch_peer: EndpointId,
    branch: RawBranchId,
    head: RawHash,
    publisher: crate::channel::PublisherKey,
    dht: &Option<crate::dht::api::ApiClient>,
    events: &mpsc::Sender<NetEvent>,
    self_cap: &RawHash,
) {
    if let Err(e) = fetch_reachable(ep, fetch_peer, &head, dht, events, self_cap).await {
        eprintln!("[net] fetch error: {e}");
    } else {
        let _ = events.send(NetEvent::Head { branch, head, publisher });
    }
}

// ── Protocol handler ─────────────────────────────────────────────────

#[derive(Clone)]
struct SnapshotHandler {
    snapshot: Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    /// Verifies all incoming capability chains. Required — protocol v4
    /// has mandatory auth.
    team_root: ed25519_dalek::VerifyingKey,
    /// Pubkeys whose capabilities are revoked. Cascades transitively.
    /// `tokio::sync::RwLock` so revocations can be added at runtime
    /// (e.g. via gossip) without restarting the handler.
    revoked: Arc<tokio::sync::RwLock<std::collections::HashSet<ed25519_dalek::VerifyingKey>>>,
}

impl std::fmt::Debug for SnapshotHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotHandler").finish()
    }
}

impl iroh::protocol::ProtocolHandler for SnapshotHandler {
    async fn accept(&self, connection: iroh::endpoint::Connection) -> Result<(), iroh::protocol::AcceptError> {
        let snap = self.snapshot.clone();
        let team_root = self.team_root;
        let revoked = self.revoked.clone();

        // Extract the connecting peer's verified ed25519 identity from
        // iroh's TLS handshake.
        let peer_endpoint = connection.remote_id();
        let peer_pubkey = match ed25519_dalek::VerifyingKey::from_bytes(
            peer_endpoint.as_bytes(),
        ) {
            Ok(k) => k,
            Err(_) => return Ok(()),
        };

        // Per-connection auth state. Set by the first `OP_AUTH` stream;
        // read by every subsequent stream to gate access.
        let auth_state: Arc<tokio::sync::RwLock<
            Option<triblespace_core::repo::capability::VerifiedCapability>,
        >> = Arc::new(tokio::sync::RwLock::new(None));

        loop {
            let (mut send, mut recv) = match connection.accept_bi().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            let snap = snap.clone();
            let auth_state = auth_state.clone();
            let revoked = revoked.clone();
            tokio::spawn(async move {
                if let Err(e) = serve_stream(
                    &snap,
                    team_root,
                    peer_pubkey,
                    auth_state,
                    revoked,
                    &mut send,
                    &mut recv,
                ).await {
                    eprintln!("handler error: {e}");
                }
                let _ = send.finish();
            });
        }
        Ok(())
    }
}

async fn serve_stream(
    snap_arc: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    team_root: ed25519_dalek::VerifyingKey,
    peer_pubkey: ed25519_dalek::VerifyingKey,
    auth_state: Arc<tokio::sync::RwLock<
        Option<triblespace_core::repo::capability::VerifiedCapability>,
    >>,
    revoked: Arc<tokio::sync::RwLock<std::collections::HashSet<ed25519_dalek::VerifyingKey>>>,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) -> anyhow::Result<()> {
    use triblespace_core::blob::Blob;
    use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
    use triblespace_core::value::schemas::hash::{Blake3, Handle};
    use triblespace_core::value::Value;

    let op = recv_u8(recv).await?;

    if op == OP_AUTH {
        let cap_handle_raw = recv_hash(recv).await?;
        let cap_handle: Value<Handle<Blake3, SimpleArchive>> =
            Value::new(cap_handle_raw);

        let revoked_snapshot = revoked.read().await.clone();
        let snap_for_fetch = snap_arc.clone();
        let result = triblespace_core::repo::capability::verify_chain(
            team_root,
            cap_handle,
            peer_pubkey,
            &revoked_snapshot,
            move |h: Value<Handle<Blake3, SimpleArchive>>| -> Option<Blob<SimpleArchive>> {
                let bytes = snap_for_fetch
                    .lock()
                    .unwrap()
                    .as_ref()?
                    .get_blob(&h.raw)?;
                Some(Blob::new(anybytes::Bytes::from_source(bytes)))
            },
        );

        match result {
            Ok(verified) => {
                *auth_state.write().await = Some(verified);
                send_u8(send, AUTH_OK).await?;
            }
            Err(_) => {
                send_u8(send, AUTH_REJECTED).await?;
            }
        }
        return Ok(());
    }

    // All other ops require a verified cap on the connection. Snapshot
    // the auth state once so the scope gate sees a stable view of the
    // verified cap for the rest of this stream's lifetime.
    let verified = match auth_state.read().await.clone() {
        Some(v) => v,
        None => {
            // Not authenticated. Close the stream silently — the client
            // should have presented OP_AUTH first.
            return Ok(());
        }
    };
    // Two-tier scope gate:
    //
    //  - branch level: `OP_LIST` and `OP_HEAD` are filtered by
    //    `verified.grants_read_on(branch)`.
    //  - blob level: `OP_GET_BLOB` and `OP_CHILDREN` are filtered by
    //    blob-graph reachability from the allowed heads. A peer with a
    //    cap restricted to branch X cannot fetch blobs that only branch
    //    Y reaches, even if they probe by raw hash. Unrestricted caps
    //    (`granted_branches() == None`) skip the reachability filter.
    //
    // Reachability is recomputed per OP_GET_BLOB / OP_CHILDREN call for
    // simplicity; for chain-walk-heavy workloads, a per-stream cache
    // would be the obvious next optimisation.

    match op {
        OP_LIST => {
            let branches = snap_arc.lock().unwrap().as_ref()
                .map(|s| s.list_branches().to_vec())
                .unwrap_or_default();
            for (id_bytes, head) in &branches {
                let Some(id) = triblespace_core::id::Id::new(*id_bytes) else {
                    // Skip malformed branch ids (the all-zeros sentinel
                    // value `NIL_BRANCH_ID` round-trips through this path
                    // when the snapshot accidentally yields it).
                    continue;
                };
                if !verified.grants_read_on(&id) {
                    continue;
                }
                send_branch_id(send, id_bytes).await?;
                send_hash(send, head).await?;
            }
            send_branch_id(send, &NIL_BRANCH_ID).await?;
        }

        OP_HEAD => {
            let id_bytes = recv_branch_id(recv).await?;
            let allowed = triblespace_core::id::Id::new(id_bytes)
                .is_some_and(|id| verified.grants_read_on(&id));
            let hash = if allowed {
                snap_arc.lock().unwrap().as_ref()
                    .and_then(|s| s.head(&id_bytes))
                    .unwrap_or(NIL_HASH)
            } else {
                NIL_HASH
            };
            send_hash(send, &hash).await?;
        }

        OP_GET_BLOB => {
            let hash = recv_hash(recv).await?;
            let data = {
                let guard = snap_arc.lock().unwrap();
                guard.as_ref().and_then(|snap| {
                    if !blob_in_scope(snap.as_ref(), &verified, &hash) {
                        return None;
                    }
                    snap.get_blob(&hash)
                })
            };
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
                        if !blob_in_scope(snap.as_ref(), &verified, &parent_hash) {
                            Vec::new()
                        } else {
                            match snap.get_blob(&parent_hash) {
                                None => Vec::new(),
                                Some(parent_data) => {
                                    let mut result = Vec::new();
                                    for chunk in parent_data.chunks(32) {
                                        if chunk.len() == 32 {
                                            let mut candidate = [0u8; 32];
                                            candidate.copy_from_slice(chunk);
                                            // Both presence AND scope —
                                            // a parent blob that's in
                                            // scope can still reference
                                            // children that aren't (it
                                            // can't, given how piles are
                                            // shaped, but the explicit
                                            // check is cheap and keeps
                                            // the gate honest).
                                            if snap.has_blob(&candidate)
                                                && blob_in_scope(snap.as_ref(), &verified, &candidate)
                                            {
                                                result.push(candidate);
                                            }
                                        }
                                    }
                                    result
                                }
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

/// Returns `true` if `hash` is reachable (transitively, via 32-byte-chunk
/// children references) from at least one branch head the `verified` cap
/// grants read access on. Unrestricted caps short-circuit to `true` for
/// every hash present in the snapshot.
///
/// O(snapshot blob count) worst case — runs once per `OP_GET_BLOB` /
/// `OP_CHILDREN` call. A per-stream cache would amortise chain walks if
/// this becomes a bottleneck.
fn blob_in_scope(
    snap: &dyn AnySnapshot,
    verified: &triblespace_core::repo::capability::VerifiedCapability,
    hash: &RawHash,
) -> bool {
    if !snap.has_blob(hash) {
        return false;
    }
    if verified.granted_branches().is_none() {
        // Unrestricted cap: every blob present in the snapshot is in
        // scope. (The cap may still lack read permission entirely; in
        // that case `grants_read()` is false and the branch-level gate
        // would have filtered every head — but the granted_branches
        // shape only tells us "no restriction", so cross-check here.)
        return verified.grants_read();
    }

    // Walk reachable from allowed heads.
    let mut frontier: Vec<RawHash> = snap
        .list_branches()
        .iter()
        .filter_map(|(bid, head)| {
            triblespace_core::id::Id::new(*bid)
                .filter(|id| verified.grants_read_on(id))
                .map(|_| *head)
        })
        .collect();
    let mut seen: HashSet<RawHash> = HashSet::new();
    while let Some(h) = frontier.pop() {
        if !seen.insert(h) {
            continue;
        }
        if h == *hash {
            return true;
        }
        if let Some(data) = snap.get_blob(&h) {
            for chunk in data.chunks(32) {
                if chunk.len() == 32 {
                    let mut child = [0u8; 32];
                    child.copy_from_slice(chunk);
                    if snap.has_blob(&child) && !seen.contains(&child) {
                        frontier.push(child);
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    //! Glue tests for the snapshot → verify_chain wiring.
    //!
    //! These cover the auth-side bridge: cap+sig blobs put into a
    //! `MemoryRepo`, snapshotted via [`StoreSnapshot`], boxed as
    //! [`AnySnapshot`], and used as the `fetch_blob` callback that
    //! [`triblespace_core::repo::capability::verify_chain`] needs. That
    //! callback is the *only* new wiring on top of what the capability
    //! lib tests already cover; testing it in isolation pins down the
    //! contract without dragging in iroh's QUIC / DNS / relay stack
    //! (which is its own integration concern).
    //!
    //! End-to-end tests over a real iroh transport are deferred to a
    //! separate harness — they need a relay or address-lookup service
    //! configured for two endpoints to discover each other in-process,
    //! and the capability-verification logic this module wires up
    //! does not depend on the transport choice.
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use triblespace_core::blob::Blob;
    use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
    use triblespace_core::id::{ExclusiveId, ufoid};
    use triblespace_core::macros::entity;
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::repo::capability::{
        VerifyError, build_capability, verify_chain, PERM_READ,
    };
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::trible::TribleSet;
    use triblespace_core::value::TryToValue;
    use triblespace_core::value::Value;
    use triblespace_core::value::schemas::hash::{Blake3, Handle};
    use triblespace_core::value::schemas::time::NsTAIInterval;
    use hifitime::Epoch;

    fn now_plus_24h() -> Value<NsTAIInterval> {
        let now = Epoch::now().expect("system time");
        let later = now + hifitime::Duration::from_seconds(24.0 * 3600.0);
        (now, later).try_to_value().expect("valid interval")
    }

    fn empty_scope() -> (triblespace_core::id::Id, TribleSet) {
        let scope_root = ufoid();
        let facts = entity! { ExclusiveId::force_ref(&scope_root) @
            triblespace_core::metadata::tag: PERM_READ,
        };
        (*scope_root, TribleSet::from(facts))
    }

    /// Build a `Box<dyn AnySnapshot>` containing the given blobs — the
    /// same shape `serve_stream` reaches into when verifying an OP_AUTH
    /// capability handle.
    fn snapshot_with_blobs(
        blobs: &[Blob<SimpleArchive>],
    ) -> Box<dyn AnySnapshot> {
        let mut store = MemoryRepo::default();
        for blob in blobs {
            store
                .put::<SimpleArchive, _>(blob.clone())
                .expect("put blob");
        }
        Box::new(StoreSnapshot::from_store(&mut store).expect("snapshot"))
    }

    /// Wrap a snapshot in the `fetch_blob` callback shape that
    /// [`verify_chain`] consumes. Mirrors the closure built inside
    /// [`serve_stream`]: `&h.raw → snap.get_blob → Blob<SimpleArchive>`.
    fn fetch_via_snapshot(
        snap: &Arc<Mutex<Option<Box<dyn AnySnapshot>>>>,
    ) -> impl FnMut(Value<Handle<Blake3, SimpleArchive>>) -> Option<Blob<SimpleArchive>>
    {
        let snap = snap.clone();
        move |h: Value<Handle<Blake3, SimpleArchive>>| -> Option<Blob<SimpleArchive>> {
            let bytes = snap.lock().unwrap().as_ref()?.get_blob(&h.raw)?;
            Some(Blob::new(anybytes::Bytes::from_source(bytes)))
        }
    }

    #[test]
    fn snapshot_lookup_serves_a_valid_cap_chain_to_verify_chain() {
        let team_root = SigningKey::generate(&mut OsRng);
        let founder = SigningKey::generate(&mut OsRng);
        let (scope_root, scope_facts) = empty_scope();
        let (cap_blob, sig_blob) = build_capability(
            &team_root,
            founder.verifying_key(),
            None,
            scope_root,
            scope_facts,
            now_plus_24h(),
        )
        .expect("cap builds");
        let sig_handle: Value<Handle<Blake3, SimpleArchive>> =
            (&sig_blob).get_handle();

        let snap_box = snapshot_with_blobs(&[cap_blob, sig_blob]);
        let snap_arc: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
            Arc::new(Mutex::new(Some(snap_box)));

        let revoked = HashSet::new();
        let result = verify_chain(
            team_root.verifying_key(),
            sig_handle,
            founder.verifying_key(),
            &revoked,
            fetch_via_snapshot(&snap_arc),
        );

        let verified = result.expect("snapshot served chain to verifier; chain valid");
        assert_eq!(verified.subject, founder.verifying_key());
        assert_eq!(verified.scope_root, scope_root);
    }

    #[test]
    fn snapshot_lookup_rejects_unknown_handle_as_chain_break() {
        let team_root = SigningKey::generate(&mut OsRng);
        let founder = SigningKey::generate(&mut OsRng);
        let snap_arc: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
            Arc::new(Mutex::new(Some(snapshot_with_blobs(&[]))));

        // Empty snapshot: no blob keyed by the all-zeros handle exists,
        // so `verify_chain` cannot fetch the leaf signature blob.
        let zero_handle: Value<Handle<Blake3, SimpleArchive>> =
            Value::new([0u8; 32]);
        let revoked = HashSet::new();
        let result = verify_chain(
            team_root.verifying_key(),
            zero_handle,
            founder.verifying_key(),
            &revoked,
            fetch_via_snapshot(&snap_arc),
        );
        // The exact variant is `Fetch` (the verifier's `fetch_blob`
        // callback returned None); what matters here is that an absent
        // handle cleanly fails verification rather than panicking or
        // hanging.
        assert!(
            matches!(result, Err(VerifyError::Fetch)),
            "unknown handle must surface as Fetch; got {:?}",
            result,
        );
    }

    /// Construct a `VerifiedCapability` with a hand-crafted scope facts
    /// set, bypassing chain verification. Used to exercise scope-gating
    /// helpers that depend only on the cap_set shape.
    fn manual_verified_cap(
        scope_root: triblespace_core::id::Id,
        permissions: &[triblespace_core::id::Id],
        branches: &[triblespace_core::id::Id],
    ) -> triblespace_core::repo::capability::VerifiedCapability {
        let mut cap_set = TribleSet::new();
        for perm in permissions {
            cap_set += TribleSet::from(entity! {
                ExclusiveId::force_ref(&scope_root) @
                triblespace_core::metadata::tag: *perm,
            });
        }
        for b in branches {
            cap_set += TribleSet::from(entity! {
                ExclusiveId::force_ref(&scope_root) @
                triblespace_core::repo::capability::scope_branch: *b,
            });
        }
        let dummy_subject = SigningKey::generate(&mut OsRng).verifying_key();
        triblespace_core::repo::capability::VerifiedCapability {
            subject: dummy_subject,
            scope_root,
            cap_set,
        }
    }

    /// Build a snapshot containing two disjoint branch subgraphs:
    /// branch_a → head_a → leaf_a; branch_b → head_b → leaf_b.
    /// Returns `(snap, branch_a, branch_b, head_a, leaf_a, head_b, leaf_b)`.
    fn two_branch_snapshot() -> (
        Box<dyn AnySnapshot>,
        triblespace_core::id::Id,
        triblespace_core::id::Id,
        RawHash,
        RawHash,
        RawHash,
        RawHash,
    ) {
        use triblespace_core::blob::schemas::UnknownBlob;
        use triblespace_core::repo::BranchStore;
        let mut store = MemoryRepo::default();

        // Distinct content per leaf so blake3 hashes diverge.
        let leaf_a_bytes = anybytes::Bytes::from_source(b"leaf_a".to_vec());
        let leaf_a = store.put::<UnknownBlob, _>(leaf_a_bytes).unwrap();

        let leaf_b_bytes = anybytes::Bytes::from_source(b"leaf_b".to_vec());
        let leaf_b = store.put::<UnknownBlob, _>(leaf_b_bytes).unwrap();

        // Each "head" blob is a 32-byte chunk pointing at its leaf — the
        // same shape OP_CHILDREN walks. (Real branch metadata is richer,
        // but the reachability gate only cares about the chunk pattern.)
        let head_a_bytes = anybytes::Bytes::from_source(leaf_a.raw.to_vec());
        let head_a = store.put::<UnknownBlob, _>(head_a_bytes).unwrap();

        let head_b_bytes = anybytes::Bytes::from_source(leaf_b.raw.to_vec());
        let head_b = store.put::<UnknownBlob, _>(head_b_bytes).unwrap();

        let branch_a = ufoid();
        let branch_b = ufoid();
        let head_a_simple: Value<Handle<Blake3, SimpleArchive>> =
            Value::new(head_a.raw);
        let head_b_simple: Value<Handle<Blake3, SimpleArchive>> =
            Value::new(head_b.raw);
        store.update(*branch_a, None, Some(head_a_simple)).unwrap();
        store.update(*branch_b, None, Some(head_b_simple)).unwrap();

        let snap: Box<dyn AnySnapshot> =
            Box::new(StoreSnapshot::from_store(&mut store).expect("snapshot"));
        (snap, *branch_a, *branch_b, head_a.raw, leaf_a.raw, head_b.raw, leaf_b.raw)
    }

    #[test]
    fn blob_in_scope_filters_by_branch_reachability() {
        let (snap, branch_a, _branch_b, head_a, leaf_a, head_b, leaf_b) =
            two_branch_snapshot();
        let scope_root = *ufoid();
        // Cap allows reading branch_a only.
        let verified =
            manual_verified_cap(scope_root, &[PERM_READ], &[branch_a]);

        assert!(
            blob_in_scope(snap.as_ref(), &verified, &head_a),
            "head reachable from allowed branch is in scope",
        );
        assert!(
            blob_in_scope(snap.as_ref(), &verified, &leaf_a),
            "leaf reachable from allowed branch is in scope",
        );
        assert!(
            !blob_in_scope(snap.as_ref(), &verified, &head_b),
            "head of disallowed branch is out of scope",
        );
        assert!(
            !blob_in_scope(snap.as_ref(), &verified, &leaf_b),
            "leaf reachable only from disallowed branch is out of scope",
        );
    }

    #[test]
    fn blob_in_scope_unrestricted_admits_any_present_blob() {
        let (snap, _branch_a, _branch_b, head_a, _leaf_a, head_b, _leaf_b) =
            two_branch_snapshot();
        let scope_root = *ufoid();
        // Unrestricted: PERM_READ, no scope_branch tribles.
        let verified = manual_verified_cap(scope_root, &[PERM_READ], &[]);

        assert!(blob_in_scope(snap.as_ref(), &verified, &head_a));
        assert!(
            blob_in_scope(snap.as_ref(), &verified, &head_b),
            "unrestricted cap admits all branches' heads",
        );
        let absent = [0xFFu8; 32];
        assert!(
            !blob_in_scope(snap.as_ref(), &verified, &absent),
            "blobs absent from the snapshot are never in scope",
        );
    }

    #[test]
    fn blob_in_scope_with_no_read_permission_admits_nothing() {
        let (snap, branch_a, _branch_b, head_a, _leaf_a, _head_b, _leaf_b) =
            two_branch_snapshot();
        let scope_root = *ufoid();
        // Cap with branch restriction but no read permission tag.
        let verified = manual_verified_cap(scope_root, &[], &[branch_a]);

        assert!(
            !blob_in_scope(snap.as_ref(), &verified, &head_a),
            "cap without read permission cannot reach any blob, even of \
             a notionally-allowed branch",
        );
    }

    #[test]
    fn snapshot_lookup_rejects_chain_signed_by_a_foreign_root() {
        let real_team_root = SigningKey::generate(&mut OsRng);
        let fake_team_root = SigningKey::generate(&mut OsRng);
        let founder = SigningKey::generate(&mut OsRng);
        let (scope_root, scope_facts) = empty_scope();
        // Cap is structurally well-formed and chained one link deep —
        // but the signing key is not the configured team root.
        let (cap_blob, sig_blob) = build_capability(
            &fake_team_root,
            founder.verifying_key(),
            None,
            scope_root,
            scope_facts,
            now_plus_24h(),
        )
        .expect("cap builds");
        let sig_handle: Value<Handle<Blake3, SimpleArchive>> =
            (&sig_blob).get_handle();

        let snap_box = snapshot_with_blobs(&[cap_blob, sig_blob]);
        let snap_arc: Arc<Mutex<Option<Box<dyn AnySnapshot>>>> =
            Arc::new(Mutex::new(Some(snap_box)));

        let revoked = HashSet::new();
        let result = verify_chain(
            real_team_root.verifying_key(),
            sig_handle,
            founder.verifying_key(),
            &revoked,
            fetch_via_snapshot(&snap_arc),
        );
        assert!(
            result.is_err(),
            "chain signed by a foreign root must fail verification; got {:?}",
            result,
        );
    }
}
