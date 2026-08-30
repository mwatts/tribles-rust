//! Stock `iroh-gossip` collection wakes on the production endpoint/router.

use std::time::Duration;

use ed25519_dalek::SigningKey;
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh::test_utils::test_transport::TestNetwork;
use iroh_base::{EndpointAddr, SecretKey};
use triblespace_core::capability::{CapabilityClaim, CapabilityMode, CapabilityProofBundle};
use triblespace_core::collection::CollectionHandle;
use triblespace_net::inventory::{ReconcileQos, sync_team_capability_atom};
use triblespace_net::peer::PeerConfig;
use triblespace_net::protocol::connect_capability_atom;
use triblespace_net::transport::Transport;
use triblespace_net::transport::iroh::bind_with_endpoint;
use triblespace_net::wake::{CollectionWakeEvent, CollectionWakeRoot, ReceivedCollectionWake};

fn key(byte: u8) -> SigningKey {
    SigningKey::from_bytes(&[byte; 32])
}

fn proof(
    root: &SigningKey,
    subject: &SigningKey,
    atom: triblespace_core::capability::CapabilityAtom,
) -> CapabilityProofBundle {
    CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
        subject.verifying_key(),
    )
    .unwrap()
}

fn config(root: &SigningKey, subject: &SigningKey, peers: Vec<EndpointAddr>) -> PeerConfig {
    let team = root.verifying_key();
    PeerConfig {
        peers,
        team,
        connect_proof: proof(root, subject, connect_capability_atom(team)),
        sync_proof: proof(root, subject, sync_team_capability_atom(team)),
        qos: ReconcileQos::default(),
    }
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

async fn next_wake(
    topic: &mut triblespace_net::wake::CollectionWakeTopic,
) -> ReceivedCollectionWake {
    loop {
        match topic.next_event().await.expect("wake topic remains open") {
            Some(CollectionWakeEvent::Received(wake)) => return wake,
            Some(CollectionWakeEvent::Rejected { error, .. }) => {
                panic!("typed peer emitted an invalid wake: {error}")
            }
            Some(_) => {}
            None => panic!("wake topic closed"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stock_gossip_shares_router_and_delivers_distinct_opaque_roots() {
    let network = TestNetwork::new();
    let root = key(0xF0);
    let key_a = key(0xA1);
    let key_b = key(0xB1);
    let secret_a = triblespace_net::identity::iroh_secret(&key_a);
    let secret_b = triblespace_net::identity::iroh_secret(&key_b);
    let endpoint_a = test_endpoint(&network, secret_a).await;
    let endpoint_b = test_endpoint(&network, secret_b).await;
    let id_a = endpoint_a.id();
    let addr_a = endpoint_a.addr();

    let harness_a = bind_with_endpoint(endpoint_a, &config(&root, &key_a, Vec::new())).await;
    let harness_b = bind_with_endpoint(endpoint_b, &config(&root, &key_b, vec![addr_a])).await;
    let plane_a = harness_a.transport.wake_plane();
    let plane_b = harness_b.transport.wake_plane();
    let collection = CollectionHandle::new([0xC1; 32]);
    let mut topic_a = plane_a
        .subscribe(collection, Vec::new())
        .await
        .expect("subscribe first peer");
    let mut topic_b = plane_b
        .subscribe(collection, vec![id_a])
        .await
        .expect("subscribe bootstrap peer");

    tokio::time::timeout(Duration::from_secs(10), async {
        tokio::try_join!(topic_a.joined(), topic_b.joined())
    })
    .await
    .expect("stock gossip peers join")
    .expect("gossip join succeeds");

    let first = CollectionWakeRoot::new([1; 32]);
    let second = CollectionWakeRoot::new([2; 32]);
    topic_a
        .broadcast(first)
        .await
        .expect("broadcast first root");
    topic_a
        .broadcast(second)
        .await
        .expect("broadcast second root");

    let (received_first, received_second) = tokio::time::timeout(Duration::from_secs(10), async {
        (next_wake(&mut topic_b).await, next_wake(&mut topic_b).await)
    })
    .await
    .expect("both distinct root wakes arrive");
    assert_eq!(received_first.wake.root(), first);
    assert_eq!(received_second.wake.root(), second);
    for received in [received_first, received_second] {
        assert_eq!(received.wake.origin(), id_a);
        assert_eq!(received.delivered_from, id_a);
        assert!(received.scope.is_direct());
    }

    drop((topic_a, topic_b, plane_a, plane_b));
    harness_a.transport.shutdown().await;
    harness_b.transport.shutdown().await;
}
