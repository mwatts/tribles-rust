//! `Host`: the network thread.
//!
//! Owns the iroh endpoint, gossip, DHT, and protocol server.
//! Runs in its own thread with its own tokio runtime. Communicates
//! with the sync world via channels only.

use std::collections::HashSet;
use std::sync::mpsc;
use std::thread;

use iroh_base::EndpointId;
use ed25519_dalek::SigningKey;

use crate::channel::{NetCommand, NetEvent};
use crate::identity::iroh_secret;
use crate::protocol::*;

/// Configuration for the Host.
pub struct HostConfig {
    /// Gossip topic name (None = no gossip).
    pub gossip_topic: Option<String>,
    /// Initial gossip peers.
    pub gossip_peers: Vec<EndpointId>,
    /// DHT bootstrap peers (empty = no DHT).
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

/// The network presence. Runs async in its own thread.
pub struct Host {
    /// Send commands to the network thread.
    commands: mpsc::Sender<NetCommand>,
    /// Receive events from the network thread.
    events: mpsc::Receiver<NetEvent>,
    /// This node's public identity.
    id: EndpointId,
    /// The network thread handle.
    _thread: thread::JoinHandle<()>,
}

impl Host {
        /// Spawn a new Host with the given signing key and configuration.
    ///
    /// `reader` is a store reader snapshot for serving protocol requests.
    /// It's Send + Clone + 'static, so it can live in the Host thread.
    /// Call `update_reader()` to refresh it when the store changes.
    pub fn new<R>(key: SigningKey, config: HostConfig, reader: R) -> Self
    where
        R: triblespace_core::repo::BlobStoreGet<Blake3>
            + triblespace_core::repo::BlobStoreList<Blake3>
            + Clone + Send + Sync + 'static,
    {
        let secret = iroh_secret(&key);
        let id: EndpointId = secret.public().into();

        let (cmd_tx, cmd_rx) = mpsc::channel::<NetCommand>();
        let (evt_tx, evt_rx) = mpsc::channel::<NetEvent>();

        let thread = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(host_loop(secret, config, cmd_rx, evt_tx));
        });

        Self {
            commands: cmd_tx,
            events: evt_rx,
            id,
            _thread: thread,
        }
    }

    /// This node's endpoint ID (public key).
    pub fn id(&self) -> EndpointId { self.id }

    /// Get a sender for commands (clone for Leader).
    pub fn commands(&self) -> mpsc::Sender<NetCommand> {
        self.commands.clone()
    }

    /// Take the event receiver (move into Follower).
    /// Can only be called once — the receiver is moved.
    pub fn take_events(&mut self) -> Option<mpsc::Receiver<NetEvent>> {
        // Swap with a dead channel receiver.
        let (_, dead_rx) = mpsc::channel();
        let rx = std::mem::replace(&mut self.events, dead_rx);
        Some(rx)
    }
}

/// The async event loop running inside the Host thread.
async fn host_loop(
    secret: iroh_base::SecretKey,
    config: HostConfig,
    commands: mpsc::Receiver<NetCommand>,
    events: mpsc::Sender<NetEvent>,
) {
    use iroh::endpoint::presets;
    use iroh::protocol::Router;
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

    // Protocol handler: serves from a shared event sender.
    // The handler forwards GET_BLOB/CHILDREN requests by reading from
    // connected peers. For serving LOCAL blobs, the Leader/Follower
    // side handles it — the Host thread only serves as a relay.
    // TODO: integrate with a shared blob store for serving local blobs.

    // Gossip setup.
    let mut gossip_sender: Option<GossipSender> = None;
    let gossip_handle;
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
        match topic {
            Ok(topic) => {
                let (sender, receiver) = topic.split();
                gossip_sender = Some(sender);

                // Spawn gossip receiver task: forward HEAD events.
                let events_tx = events.clone();
                gossip_handle = Some(tokio::spawn(async move {
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
                }));
            }
            Err(e) => {
                eprintln!("host: gossip subscribe failed: {e}");
                gossip_handle = None;
            }
        }
    } else {
        gossip_handle = None;
    }

    // DHT setup.
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

    // Command processing loop.
    loop {
        // Process all pending commands (non-blocking).
        while let Ok(cmd) = commands.try_recv() {
            match cmd {
                NetCommand::Announce(hash) => {
                    // TODO: DHT announce
                    let _ = hash;
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
                    // Connect to peer, fetch HEAD, BFS pull, send events.
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

        // Yield to let other tasks run, then check commands again.
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

    // Get remote head.
    let Some(head) = op_head(&conn, branch).await? else {
        return Ok(());
    };

    // BFS pull.
    let mut seen: HashSet<RawHash> = HashSet::new();
    let mut current_level = vec![head];
    seen.insert(head);

    // Fetch root.
    if let Some(data) = op_get_blob(&conn, &head).await? {
        let _ = events.send(NetEvent::Blob(data));
    }

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

    // Send the HEAD event after all blobs are sent.
    let _ = events.send(NetEvent::Head { branch: *branch, head });

    conn.close(0u32.into(), b"done");
    Ok(())
}
