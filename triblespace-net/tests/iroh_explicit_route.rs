//! A configured [`EndpointAddr`] must be used as the dial target, not reduced
//! to its endpoint id.  This matters for deployments with a dedicated fabric:
//! discovery is allowed to choose a route only when the caller supplied none.

use ed25519_dalek::SigningKey;
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh::test_utils::test_transport::{TestNetwork, to_custom_addr};
use iroh_base::{EndpointAddr, EndpointId, SecretKey, TransportAddr};
use rand::rngs::OsRng;
use triblespace_core::authority::{
    AuthorityGrant, AuthorityMode, AuthorityProof, AuthorityProofStep,
};
use triblespace_core::blob::IntoBlob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::collection::{CollectionCommit, empty_metadata_handle};

use triblespace_net::host::{PeerConfig, SyncDirection};
use triblespace_net::protocol::{ACTION_CONNECT, PILE_SYNC_ALPN};
use triblespace_net::transport::{Transport, iroh::bind_with_endpoint};

async fn undiscoverable_endpoint(network: &TestNetwork, secret: SecretKey) -> Endpoint {
    let transport = network
        .create_transport(secret.public())
        .expect("create test transport");
    Endpoint::builder(presets::N0)
        .secret_key(secret)
        .relay_mode(iroh::RelayMode::Disabled)
        .clear_address_lookup()
        .clear_ip_transports()
        .add_custom_transport(transport)
        .bind()
        .await
        .expect("bind endpoint")
}

fn direct_addr(id: EndpointId) -> EndpointAddr {
    EndpointAddr::from_parts(
        id,
        std::iter::once(TransportAddr::Custom(to_custom_addr(id))),
    )
}

fn connect_proof(key: &SigningKey) -> AuthorityProof {
    let collection = triblespace_core::authority::collection(key.verifying_key());
    let grant = AuthorityGrant::root(
        key.verifying_key(),
        collection,
        ACTION_CONNECT,
        AuthorityMode::Invoke,
    );
    let data = grant.fragment().into_facts().to_blob();
    let commit = CollectionCommit::sign(
        key,
        collection,
        triblespace_core::inline::encodings::hash::Handle::<SimpleArchive>::to_hash(
            data.get_handle(),
        ),
        empty_metadata_handle(),
    );
    AuthorityProof::new(vec![AuthorityProofStep::new(commit, data)])
}

fn config(key: &SigningKey, peers: Vec<EndpointAddr>) -> PeerConfig {
    PeerConfig {
        peers,
        gossip: false,
        team_root: key.verifying_key(),
        connect_proof: connect_proof(key),
        direction: SyncDirection::Bidirectional,
    }
}

#[tokio::test]
async fn dial_uses_configured_route_when_discovery_has_none() {
    let network = TestNetwork::new();
    let server_key = SigningKey::generate(&mut OsRng);
    let client_key = SigningKey::generate(&mut OsRng);
    let server_secret = SecretKey::from_bytes(&server_key.to_bytes());
    let client_secret = SecretKey::from_bytes(&client_key.to_bytes());
    let server_id = EndpointId::from(server_secret.public());

    let server_endpoint = undiscoverable_endpoint(&network, server_secret).await;
    let client_endpoint = undiscoverable_endpoint(&network, client_secret).await;

    let mut server = bind_with_endpoint(server_endpoint, &config(&server_key, Vec::new())).await;
    let client = bind_with_endpoint(
        client_endpoint,
        &config(&client_key, vec![direct_addr(server_id)]),
    )
    .await;

    let connect = client.transport.dial(*server_id.as_bytes(), PILE_SYNC_ALPN);
    let accept = server.incoming.recv();
    let (connection, incoming) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(connect, accept)
    })
    .await
    .expect("direct dial timed out");

    connection.expect("configured direct route should dial");
    let incoming = incoming.expect("server should receive connection");
    assert_eq!(incoming.alpn, PILE_SYNC_ALPN);
}
