//! Real-Iroh collection wake and repair coverage.

use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh::test_utils::test_transport::TestNetwork;
use iroh_base::{EndpointAddr, SecretKey};
use triblespace_core::collection::{
    AdmissionPolicy, CollectionCommit, CollectionData, CollectionPolicy, CollectionRead,
    CollectionRecord, CollectionStore, CollectionStoreExt, empty_metadata_handle,
};
use triblespace_core::repo::SnapshotSource;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_net::host::{self, PeerConfig};
use triblespace_net::inventory::ReconcileQos;
use triblespace_net::peer::Peer;

fn key(byte: u8) -> SigningKey {
    SigningKey::from_bytes(&[byte; 32])
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

async fn bring_up(
    endpoint: Endpoint,
    store: MemoryRepo,
    peers: Vec<EndpointAddr>,
) -> Peer<MemoryRepo> {
    let id = endpoint.id();
    let config = PeerConfig {
        peers,
        qos: ReconcileQos::default(),
    };
    let harness = triblespace_net::transport::iroh::bind_with_endpoint(endpoint, &config).await;
    let (sender, receiver, wiring) = host::wire(id);
    tokio::spawn(host::run_host(harness, config, wiring));
    Peer::with_wiring(store, ReconcileQos::default(), sender, receiver)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn signed_collection_wake_repairs_before_periodic_fallback() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();
    let network = TestNetwork::new();
    let server_key = key(0xA1);
    let reader_key = key(0xB1);
    let server_endpoint = test_endpoint(
        &network,
        triblespace_net::identity::iroh_secret(&server_key),
    )
    .await;
    let reader_endpoint = test_endpoint(
        &network,
        triblespace_net::identity::iroh_secret(&reader_key),
    )
    .await;
    let server_addr = server_endpoint.addr();

    let policy = CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open);
    let mut server_store = MemoryRepo::default();
    let collection = server_store
        .collection("real-iroh-collection-wake", policy.clone())
        .unwrap();
    let mut reader_store = MemoryRepo::default();
    let reader_collection = reader_store
        .collection("real-iroh-collection-wake", policy)
        .unwrap();
    assert_eq!(reader_collection.handle(), collection.handle());

    let mut server = bring_up(server_endpoint, server_store, Vec::new()).await;
    let mut reader = bring_up(reader_endpoint, reader_store, vec![server_addr]).await;
    server.activate_collection(collection.handle());
    reader.activate_collection(collection.handle());

    // Let initial empty repair and the exact-handle gossip subscriptions settle.
    tokio::time::sleep(Duration::from_secs(2)).await;
    server.refresh();
    reader.refresh();

    server
        .store()
        .insert(CollectionRecord::Commit(CollectionCommit::sign(
            &server_key,
            collection.handle(),
            CollectionData::new([0xC7; 32]),
            empty_metadata_handle(),
        )))
        .unwrap();
    server.refresh();

    let started = Instant::now();
    let repaired = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            reader.refresh();
            if reader
                .snapshot()
                .unwrap()
                .records()
                .unwrap()
                .next()
                .is_some()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    assert!(
        repaired.is_ok(),
        "signed wake did not trigger collection repair"
    );
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "repair must precede the 30-second periodic fallback"
    );

    drop((server.into_store(), reader.into_store()));
}
