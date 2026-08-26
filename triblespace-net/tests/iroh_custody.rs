//! Custody replication over real iroh QUIC with deterministic packet routing.
//!
//! `TestNetwork` replaces only the packet transport and address lookup. The
//! endpoint, QUIC, pile-sync ALPN router, CONNECT exchange, REPLICATE proof,
//! and custody framing are the production implementations.
#![cfg(feature = "sim")]

use std::path::Path;
use std::time::Duration;

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh::test_utils::{run_relay_server, test_transport::TestNetwork};
use iroh_base::{EndpointAddr, EndpointId, SecretKey};
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::capability::{CapabilityClaim, CapabilityMode, CapabilityProofBundle};
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut};
use triblespace_net::host::{self, PeerConfig, SyncDirection};
use triblespace_net::replica::{
    CustodyReconcileOutcome, CustodyReplica, CustodyReplicaConfig, ReplicaServerConfig,
    ReplicaSetId, replicate_capability_atom,
};

fn key(n: u8) -> SigningKey {
    SigningKey::from_bytes(&[n; 32])
}

fn connect_proof(root: &SigningKey, subject: &SigningKey) -> CapabilityProofBundle {
    CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(
            triblespace_net::protocol::connect_capability_atom(root.verifying_key()),
            CapabilityMode::Invoke,
            None,
        ),
        subject.verifying_key(),
    )
    .unwrap()
}

fn replica_proof(
    root: &SigningKey,
    subject: &SigningKey,
    replica_set: ReplicaSetId,
) -> CapabilityProofBundle {
    CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(
            replicate_capability_atom(replica_set),
            CapabilityMode::Invoke,
            None,
        ),
        subject.verifying_key(),
    )
    .unwrap()
}

async fn test_endpoint(network: &TestNetwork, secret: SecretKey) -> Endpoint {
    let transport = network
        .create_transport(secret.public())
        .expect("create test transport");
    Endpoint::builder(presets::N0)
        .secret_key(secret)
        .relay_mode(iroh::RelayMode::Disabled)
        .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
        .add_custom_transport(transport)
        .clear_ip_transports()
        .clear_address_lookup()
        .address_lookup(network.address_lookup())
        .bind()
        .await
        .expect("bind test endpoint")
}

/// Bind an endpoint whose only possible data path is the supplied relay.
/// Address lookup and direct IP transports are both absent by construction.
async fn relay_only_endpoint(relay_map: iroh::RelayMap, secret: SecretKey) -> Endpoint {
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .relay_mode(iroh::RelayMode::Custom(relay_map))
        .ca_tls_config(iroh::tls::CaTlsConfig::insecure_skip_verify())
        .clear_ip_transports()
        .clear_address_lookup()
        .bind()
        .await
        .expect("bind relay-only endpoint");
    tokio::time::timeout(Duration::from_secs(10), endpoint.online())
        .await
        .expect("relay-only endpoint did not become reachable");
    endpoint
}

#[allow(clippy::too_many_arguments)]
async fn bring_up(
    network: &TestNetwork,
    signing_key: &SigningKey,
    store: MemoryRepo,
    connect_root: &SigningKey,
    replica_root: &SigningKey,
    replica_set: ReplicaSetId,
    bootstrap: Vec<EndpointAddr>,
    receive_dir: &Path,
) -> CustodyReplica<MemoryRepo> {
    let secret = triblespace_net::identity::iroh_secret(signing_key);
    let endpoint = test_endpoint(network, secret).await;
    bring_up_with_endpoint(
        endpoint,
        signing_key,
        store,
        connect_root,
        replica_root,
        replica_set,
        bootstrap,
        receive_dir,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn bring_up_with_endpoint(
    endpoint: Endpoint,
    signing_key: &SigningKey,
    store: MemoryRepo,
    connect_root: &SigningKey,
    replica_root: &SigningKey,
    replica_set: ReplicaSetId,
    bootstrap: Vec<EndpointAddr>,
    receive_dir: &Path,
) -> CustodyReplica<MemoryRepo> {
    let id: EndpointId = endpoint.id();
    let connect = connect_proof(connect_root, signing_key);
    let replicate = replica_proof(replica_root, signing_key, replica_set);
    let peer_config = PeerConfig {
        peers: bootstrap.clone(),
        gossip_topic: None,
        connect_root: connect_root.verifying_key(),
        connect_proof: connect.clone(),
        direction: SyncDirection::ReadOnly,
    };
    let harness =
        triblespace_net::transport::iroh::bind_custody_with_endpoint(endpoint, &peer_config).await;
    let (sender, receiver, wiring) = host::wire(id);
    tokio::spawn(host::run_custody_host(
        harness,
        peer_config,
        wiring,
        ReplicaServerConfig {
            trust_root: replica_root.verifying_key(),
            replica_set,
        },
    ));
    CustodyReplica::with_wiring(
        store,
        signing_key.verifying_key(),
        sender,
        receiver,
        CustodyReplicaConfig {
            peers: bootstrap,
            connect_root: connect_root.verifying_key(),
            connect_proof: connect,
            replica_root: replica_root.verifying_key(),
            replica_set,
            replica_proof: replicate,
            receive_temp_dir: receive_dir.to_owned(),
        },
    )
    .unwrap()
}

async fn reconcile(replica: &mut CustodyReplica<MemoryRepo>) -> CustodyReconcileOutcome {
    tokio::time::timeout(Duration::from_secs(15), replica.reconcile_once())
        .await
        .expect("custody reconciliation timed out")
        .expect("custody reconciliation failed")
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_inbound_summary_enables_reverse_pull_over_iroh() {
    init_tracing();
    let network = TestNetwork::new();
    let connect_root = key(0xF1);
    let replica_root = key(0xF2);
    let ka = key(0xA2);
    let kb = key(0xB2);
    let replica_set = ReplicaSetId::new([0xC2; 32]);
    let receive = tempfile::tempdir().expect("custody receive directory");
    let payload = Bytes::from(vec![0x5A; 1024 * 1024 + 17]);
    let mut store_a = MemoryRepo::default();
    let handle = store_a
        .put::<UnknownBlob, _>(payload.clone())
        .expect("seed A payload");

    // B has no bootstrap routes. A knows only B, and its first authenticated
    // summary request is therefore the sole possible introduction of A to B.
    let mut b = bring_up(
        &network,
        &kb,
        MemoryRepo::default(),
        &connect_root,
        &replica_root,
        replica_set,
        Vec::new(),
        receive.path(),
    )
    .await;
    let mut a = bring_up(
        &network,
        &ka,
        store_a,
        &connect_root,
        &replica_root,
        replica_set,
        vec![b.id().into()],
        receive.path(),
    )
    .await;

    assert_eq!(reconcile(&mut b).await.peers_attempted, 0);
    let introduction = reconcile(&mut a).await;
    assert_eq!(introduction.peers_attempted, 1);
    assert_eq!(introduction.peers_completed, 1);

    let reverse_pull = reconcile(&mut b).await;
    assert_eq!(
        reverse_pull.peers_attempted, 1,
        "B can only know A from A's authorized inbound summary"
    );
    assert_eq!(reverse_pull.peers_completed, 1);
    assert_eq!(reverse_pull.blobs_added, 1);
    assert_eq!(reverse_pull.blob_bytes_added, payload.len() as u64);

    let reader = b.store_mut().reader().expect("B reader");
    let received: Bytes = reader
        .get::<Bytes, UnknownBlob>(handle)
        .expect("B received A's payload through the learned reverse route");
    assert_eq!(received, payload);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_reverse_pull_uses_the_local_relay_only() {
    init_tracing();
    let (relay_map, relay_url, _relay_server) =
        run_relay_server().await.expect("start local relay server");
    let connect_root = key(0xF3);
    let replica_root = key(0xF4);
    let ka = key(0xA3);
    let kb = key(0xB3);
    let replica_set = ReplicaSetId::new([0xC3; 32]);
    let receive = tempfile::tempdir().expect("custody receive directory");
    let payload = Bytes::from(vec![0x6B; 128 * 1024 + 17]);
    let mut store_a = MemoryRepo::default();
    let handle = store_a
        .put::<UnknownBlob, _>(payload.clone())
        .expect("seed relay payload");

    let b_endpoint = relay_only_endpoint(
        relay_map.clone(),
        triblespace_net::identity::iroh_secret(&kb),
    )
    .await;
    let mut b = bring_up_with_endpoint(
        b_endpoint,
        &kb,
        MemoryRepo::default(),
        &connect_root,
        &replica_root,
        replica_set,
        Vec::new(),
        receive.path(),
    )
    .await;

    let a_endpoint =
        relay_only_endpoint(relay_map, triblespace_net::identity::iroh_secret(&ka)).await;
    let b_relay_addr = EndpointAddr::new(b.id()).with_relay_url(relay_url);
    let mut a = bring_up_with_endpoint(
        a_endpoint,
        &ka,
        store_a,
        &connect_root,
        &replica_root,
        replica_set,
        vec![b_relay_addr],
        receive.path(),
    )
    .await;

    assert_eq!(reconcile(&mut b).await.peers_attempted, 0);
    let introduction = reconcile(&mut a).await;
    assert_eq!(introduction.peers_attempted, 1);
    assert_eq!(introduction.peers_completed, 1);

    // B has no configured peers or address lookup. Its learned A route came
    // from the valid inbound summary; with both endpoints' IP transports
    // disabled, this pull and payload transfer can only traverse the relay.
    let reverse_pull = reconcile(&mut b).await;
    assert_eq!(reverse_pull.peers_attempted, 1);
    assert_eq!(reverse_pull.peers_completed, 1);
    assert_eq!(reverse_pull.blobs_added, 1);
    assert_eq!(reverse_pull.blob_bytes_added, payload.len() as u64);

    let reader = b.store_mut().reader().expect("relay recipient reader");
    let received: Bytes = reader
        .get::<Bytes, UnknownBlob>(handle)
        .expect("B received A's payload over the local relay");
    assert_eq!(received, payload);
}
