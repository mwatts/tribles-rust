//! Shared scaffolding for the deterministic-simulation integration
//! tests. Included via `mod common;` in each `sim_*.rs` test binary.
//! (Cargo treats `tests/common/mod.rs` as a submodule, not its own
//! test binary.)
#![allow(dead_code)]
#![cfg(feature = "sim")]

use std::sync::{Arc, OnceLock};

use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::authority::{
    AuthorityGrant, AuthorityMode, AuthorityProof, AuthorityProofStep,
};
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, IntoBlob};
use triblespace_core::clock::{self, VirtualClock};
use triblespace_core::collection::{CollectionCommit, empty_metadata_handle};
use triblespace_core::id::rngid::seed_ids;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_net::host;
use triblespace_net::peer::{Peer, PeerConfig, SyncDirection};
use triblespace_net::protocol::ACTION_CONNECT;
use triblespace_net::transport::sim::SimNet;

/// One virtual clock per test process (install_virtual is
/// once-per-process; each test binary is its own process).
pub fn vclock() -> Arc<VirtualClock> {
    static CLOCK: OnceLock<Arc<VirtualClock>> = OnceLock::new();
    CLOCK
        .get_or_init(|| {
            let base = hifitime::Epoch::from_gregorian_utc_at_midnight(2026, 1, 1);
            let vc = VirtualClock::new(base);
            clock::install_virtual(vc.clone()).expect("first clock install");
            vc
        })
        .clone()
}

/// Sim tests share the process-global virtual clock — serialize them.
pub fn sim_guard() -> std::sync::MutexGuard<'static, ()> {
    static SIM_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match SIM_SERIAL.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

pub fn key(n: u8) -> SigningKey {
    SigningKey::from_bytes(&[n; 32])
}

pub fn pk(k: &SigningKey) -> [u8; 32] {
    k.verifying_key().to_bytes()
}

/// Sign the one-step positive proof authorizing `subject` to CONNECT to the
/// team rooted at `root`.
pub fn connect_proof(root: &SigningKey, subject: &SigningKey) -> AuthorityProof {
    let collection = triblespace_core::authority::collection(root.verifying_key());
    let grant = AuthorityGrant::root(
        subject.verifying_key(),
        collection,
        ACTION_CONNECT,
        AuthorityMode::Invoke,
    );
    let data: Blob<SimpleArchive> = grant.fragment().into_facts().to_blob();
    let commit = CollectionCommit::sign(
        root,
        collection,
        triblespace_core::inline::encodings::hash::Handle::<SimpleArchive>::to_hash(
            data.get_handle(),
        ),
        empty_metadata_handle(),
    );
    AuthorityProof::new(vec![AuthorityProofStep::new(commit, data)])
}

/// A paused, single-thread tokio runtime + LocalSet runner — the
/// deterministic-sim execution context. `body` is the async test.
pub fn run_paused<F, T>(seed: u64, body: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_test_writer()
        .try_init();
    let vc = vclock();
    vc.reset();
    seed_ids(seed);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .expect("paused current-thread runtime");
    let local = tokio::task::LocalSet::new();
    rt.block_on(local.run_until(body))
}

/// Bring one node up on `net`: join the sim mesh, wire the host loop
/// as a local task, return the `Peer<MemoryRepo>`. `store` is the
/// node's pre-seeded local store. The CONNECT proof is sent inline and need
/// not be resident in that store. `gossip` controls team-topic participation.
pub fn bring_up(
    net: &SimNet,
    signing_key: &SigningKey,
    store: MemoryRepo,
    team_root: ed25519_dalek::VerifyingKey,
    connect_proof: AuthorityProof,
    gossip: bool,
) -> Peer<MemoryRepo> {
    bring_up_with_peers(
        net,
        signing_key,
        store,
        team_root,
        connect_proof,
        gossip,
        Vec::new(),
    )
}

/// [`bring_up`] with an explicit configured-peer discovery boundary.
///
/// The simulator routes by peer identity, so address-less `EndpointAddr`
/// values are sufficient. Keeping the conversion here lets scenarios state
/// the topology as the same `[u8; 32]` identities used by `SimNet`.
pub fn bring_up_with_peers(
    net: &SimNet,
    signing_key: &SigningKey,
    store: MemoryRepo,
    team_root: ed25519_dalek::VerifyingKey,
    connect_proof: AuthorityProof,
    gossip: bool,
    peers: Vec<[u8; 32]>,
) -> Peer<MemoryRepo> {
    let id = pk(signing_key);
    let harness = net.join(id, gossip);
    let (sender, receiver, wiring) = host::wire(EndpointId::from_bytes(&id).expect("endpoint id"));
    tokio::task::spawn_local(host::run_host(
        harness,
        PeerConfig {
            peers: peers
                .into_iter()
                .map(|peer| {
                    iroh_base::EndpointAddr::from(
                        EndpointId::from_bytes(&peer).expect("configured peer endpoint id"),
                    )
                })
                .collect(),
            gossip,
            team_root,
            connect_proof,
            direction: SyncDirection::Bidirectional,
        },
        wiring,
    ));
    Peer::with_wiring(store, SyncDirection::Bidirectional, sender, receiver)
}

/// An empty store intentionally independent of CONNECT proof residency.
pub fn empty_store() -> MemoryRepo {
    MemoryRepo::default()
}
