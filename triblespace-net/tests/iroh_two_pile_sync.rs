//! Two-pile sync over the REAL iroh transport stack — the v0.47.0
//! release-gate integration test.
//!
//! The deterministic-simulation suite (`sim_lazy`,
//! `sim_collection_gossip`) proves the protocol *logic*; this test proves the
//! *transport*: two
//! `Peer<Pile>`s — real pile files on disk — run the full production
//! stack (`transport::iroh::bind_with_endpoint`: embedded DHT node,
//! protocol router, iroh-gossip topic mesh, OP_AUTH with cap-chain
//! verification) over real iroh QUIC endpoints wired through
//! `iroh::test_utils` `TestNetwork` (an in-memory packet transport —
//! no relays, no DNS, no OS sockets — everything above the packet
//! layer is the production code path).
//!
//! A content blob lives only in pile A. B durably records a want for its hash;
//! a `Reconciler::tick` services the want over B's explicitly configured route
//! to A and lands the verified bytes in pile B under the still-recorded want.
//! This preserves real-transport, pile persistence, capability authentication,
//! content verification, and durable-want coverage without depending on the
//! retired mutable-HEAD/tracking protocol.
//!
//! Piles are created under `std::env::temp_dir()` — set `TMPDIR` to
//! redirect.
//!
//! Run with:
//! `cargo test -p triblespace-net --test iroh_two_pile_sync`

use std::time::Duration;

use ed25519_dalek::SigningKey;
use iroh::Endpoint;
use iroh::endpoint::presets;
use iroh::test_utils::test_transport::TestNetwork;
use iroh_base::{EndpointAddr, EndpointId, SecretKey};
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, IntoBlob};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::encodings::time::NsTAIInterval;
use triblespace_core::inline::{Inline, TryToInline};
use triblespace_core::prelude::BlobStore;
use triblespace_core::repo::capability::{self, PERM_ADMIN};
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::{BlobStoreGet, BlobStorePut, WantRequest, WantStore};
use triblespace_core::trible::TribleSet;
use triblespace_net::clock;
use triblespace_net::host;
use triblespace_net::peer::{Peer, PeerConfig, SyncDirection};
use triblespace_net::reconcile::Reconciler;

fn key(n: u8) -> SigningKey {
    SigningKey::from_bytes(&[n; 32])
}

/// Sign a PERM_ADMIN root cap for `subject` — the out-of-band
/// provisioning a team admin would do. Same as the sim suite's helper.
fn admin_cap(
    root: &SigningKey,
    subject: &SigningKey,
) -> (Blob<SimpleArchive>, Blob<SimpleArchive>) {
    use triblespace_core::id::{ExclusiveId, ufoid};
    use triblespace_core::macros::entity;

    let scope_root = *ufoid();
    let scope_facts = TribleSet::from(entity! {
        ExclusiveId::force_ref(&scope_root) @
        triblespace_core::metadata::tag: PERM_ADMIN,
    });
    let now = clock::epoch_now();
    let expiry: Inline<NsTAIInterval> = (now, now + hifitime::Duration::from_days(30.0))
        .try_to_inline()
        .expect("interval");
    capability::build_capability(
        root,
        subject.verifying_key(),
        None,
        scope_root,
        scope_facts,
        expiry,
    )
    .expect("build cap")
}

/// A fresh pile file in a temp dir, seeded with the given cap+sig
/// blobs so OP_AUTH verifies locally on both ends.
fn fresh_pile(
    dir: &std::path::Path,
    name: &str,
    caps: &[(Blob<SimpleArchive>, Blob<SimpleArchive>)],
) -> Pile {
    let path = dir.join(name);
    std::fs::File::create(&path).expect("create pile file");
    let mut pile = Pile::open(&path).expect("open pile");
    for (cap, sig) in caps {
        pile.put::<SimpleArchive, _>(cap.clone()).expect("seed cap");
        pile.put::<SimpleArchive, _>(sig.clone()).expect("seed sig");
    }
    pile.flush().expect("flush seeded pile");
    pile
}

/// Bind a real iroh endpoint whose only packet path is the shared
/// `TestNetwork` (mirrors `auth_handshake_e2e::test_endpoint`), with
/// the network's address-lookup service replacing the N0 discovery
/// stack so bare-`EndpointId` dials resolve without DNS/pkarr.
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
        .expect("bind endpoint")
}

/// Bring one node up over the TestNetwork: bind the endpoint, wire the
/// full production transport stack (`bind_with_endpoint`: DHT node,
/// protocol router, gossip topic), spawn the host loop as a tokio
/// task, and wrap the pile in a `Peer`.
async fn bring_up(
    network: &TestNetwork,
    signing_key: &SigningKey,
    store: Pile,
    team_root: ed25519_dalek::VerifyingKey,
    self_cap: [u8; 32],
    bootstrap: Vec<EndpointAddr>,
) -> Peer<Pile> {
    let secret = triblespace_net::identity::iroh_secret(signing_key);
    let id: EndpointId = secret.public().into();
    let ep = test_endpoint(network, secret).await;
    let config = PeerConfig {
        peers: bootstrap,
        gossip: true,
        team_root,
        self_cap,
        direction: SyncDirection::Bidirectional,
    };
    let harness = triblespace_net::transport::iroh::bind_with_endpoint(ep, &config).await;
    let (sender, receiver, wiring) = host::wire(id);
    tokio::spawn(host::run_host(harness, config, wiring));
    Peer::with_wiring(
        store,
        signing_key.clone(),
        SyncDirection::Bidirectional,
        team_root,
        sender,
        receiver,
    )
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

/// The shared two-node bring-up: team root + two admin caps, two piles
/// (both seeded with both chains), A up first with no bootstrap, B up
/// second with an explicit route to A.
struct TwoNodes {
    peer_a: Peer<Pile>,
    peer_b: Peer<Pile>,
    _dir: tempfile::TempDir,
}

async fn two_nodes(
    network: &TestNetwork,
    ka: &SigningKey,
    kb: &SigningKey,
    seed_a: impl FnOnce(&mut Pile),
) -> TwoNodes {
    let root = key(0xF0);
    let team_root = root.verifying_key();
    let cap_a = admin_cap(&root, ka);
    let cap_b = admin_cap(&root, kb);
    let self_cap_a = (&cap_a.1).get_handle().raw;
    let self_cap_b = (&cap_b.1).get_handle().raw;
    let caps = [cap_a, cap_b];

    let dir = tempfile::tempdir().expect("temp dir for piles");
    let mut pile_a = fresh_pile(dir.path(), "a.pile", &caps);
    seed_a(&mut pile_a);
    let pile_b = fresh_pile(dir.path(), "b.pile", &caps);

    let peer_a = bring_up(network, ka, pile_a, team_root, self_cap_a, Vec::new()).await;
    let a_id: EndpointAddr = peer_a.id().into();
    let peer_b = bring_up(network, kb, pile_b, team_root, self_cap_b, vec![a_id]).await;

    TwoNodes {
        peer_a,
        peer_b,
        _dir: dir,
    }
}

/// A content blob lives only in pile A. B records a durable want; the
/// Reconciler services it through the explicitly configured route and lands
/// the bytes in pile B.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn want_fetches_from_holder_over_iroh() {
    init_tracing();
    let network = TestNetwork::new();
    let ka = key(0xA1);
    let kb = key(0xB1);

    // The lazy payload: an otherwise unreferenced blob in pile A.
    let payload: TribleSet = {
        use triblespace_core::id::{ExclusiveId, ufoid};
        use triblespace_core::macros::entity;
        let e = *ufoid();
        let tag = *ufoid();
        TribleSet::from(entity! {
            ExclusiveId::force_ref(&e) @
            triblespace_core::metadata::tag: tag,
        })
    };
    let blob: Blob<SimpleArchive> = payload.to_blob();
    let hash = blob.get_handle().raw;

    let TwoNodes {
        mut peer_a,
        mut peer_b,
        _dir,
    } = two_nodes(&network, &ka, &kb, |pile| {
        pile.put::<SimpleArchive, _>(blob.clone())
            .expect("seed payload");
        pile.flush().expect("flush payload");
    })
    .await;

    // Precondition: B does not hold the payload.
    {
        let reader = peer_b.reader().expect("b reader");
        let held: Result<anybytes::Bytes, _> =
            BlobStoreGet::get::<anybytes::Bytes, UnknownBlob>(&reader, Inline::new(hash));
        assert!(
            held.is_err(),
            "precondition: B must NOT hold the never-committed payload"
        );
    }

    // The durable want: want the hash in pile B and flush — the
    // marker survives a process exit; the Reconciler is the daemon
    // that services the queue.
    {
        let mut store = peer_b.store();
        store
            .want(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)))
            .expect("record want");
        store.flush().expect("flush want");
    }

    // Service the want. Each tick diffs wants against presence and
    // drives the swarm fetch for the missing ones.
    let mut reconciler =
        Reconciler::with_backoff(Duration::from_millis(200), Duration::from_secs(2))
            .with_fetch_budget(Duration::from_secs(10));
    let mut fetched = false;
    for _ in 0..60u32 {
        peer_a.refresh(); // keep A serving a fresh snapshot
        let stats = reconciler.tick(&mut peer_b).await;
        if stats.fulfilled >= 1 {
            fetched = true;
            break;
        }
        // wants=1 expected throughout; missing goes 1 → 0 on success.
        assert!(stats.wants >= 1, "the recorded want must stay on record");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        fetched,
        "Reconciler must fetch the want from A over the iroh transport"
    );

    // The payload landed in pile B…
    {
        let reader = peer_b.reader().expect("b reader");
        let got: anybytes::Bytes =
            BlobStoreGet::get::<anybytes::Bytes, UnknownBlob>(&reader, Inline::new(hash))
                .expect("B holds the payload after reconcile");
        assert_eq!(
            blake3::hash(&got).as_bytes(),
            &hash,
            "landed bytes verify against the requested hash"
        );
    }
    // …and the demand marker is still on record —
    // it is now the retention marker for the fetched blob.
    {
        let mut store = peer_b.store();
        let still_wanted = store
            .wants()
            .expect("wants")
            .filter_map(Result::ok)
            .any(|request| request == WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)));
        assert!(
            still_wanted,
            "the want stays on record as the retention marker"
        );
    }
}
