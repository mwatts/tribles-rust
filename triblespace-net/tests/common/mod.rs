//! Shared scaffolding for the deterministic-simulation integration
//! tests. Included via `mod common;` in each `sim_*.rs` test binary.
//! (Cargo treats `tests/common/mod.rs` as a submodule, not its own
//! test binary.)
#![allow(dead_code)]
#![cfg(feature = "sim")]

use std::sync::{Arc, OnceLock};

use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::capability::{
    CapabilityClaim, CapabilityMode, CapabilityProofBundle, CapabilityValidity,
};
use triblespace_core::clock::{self, VirtualClock};
use triblespace_core::id::rngid::seed_ids;
use triblespace_core::repo::StoreScope;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_net::host;
use triblespace_net::inventory::{ReconcileQos, sync_team_capability_atom};
use triblespace_net::peer::{Peer, PeerConfig};
use triblespace_net::protocol::connect_capability_atom;
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

#[derive(Clone)]
pub struct TeamProofs {
    pub connect: CapabilityProofBundle,
    pub sync: CapabilityProofBundle,
}

/// Sign the independent one-step CONNECT and SYNC_TEAM proofs used by a node.
pub fn team_proofs(root: &SigningKey, leaf: &SigningKey) -> TeamProofs {
    team_proofs_with_validity(root, leaf, None)
}

pub fn team_proofs_with_validity(
    root: &SigningKey,
    leaf: &SigningKey,
    validity: Option<CapabilityValidity>,
) -> TeamProofs {
    let connect = CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(
            connect_capability_atom(root.verifying_key()),
            CapabilityMode::Invoke,
            validity,
        ),
        leaf.verifying_key(),
    )
    .unwrap();
    let sync = CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(
            sync_team_capability_atom(root.verifying_key()),
            CapabilityMode::Invoke,
            validity,
        ),
        leaf.verifying_key(),
    )
    .unwrap();
    TeamProofs { connect, sync }
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
/// node's pre-seeded local store. Both proofs are sent inline and need not be
/// resident in that store. `gossip` controls team-topic participation.
pub fn bring_up(
    net: &SimNet,
    signing_key: &SigningKey,
    store: MemoryRepo,
    connect_root: ed25519_dalek::VerifyingKey,
    proofs: TeamProofs,
    gossip: bool,
) -> Peer<MemoryRepo> {
    bring_up_with_peers(
        net,
        signing_key,
        store,
        connect_root,
        proofs,
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
    connect_root: ed25519_dalek::VerifyingKey,
    proofs: TeamProofs,
    gossip: bool,
    peers: Vec<[u8; 32]>,
) -> Peer<MemoryRepo> {
    bring_up_with_qos(
        net,
        signing_key,
        store,
        connect_root,
        proofs,
        gossip,
        peers,
        ReconcileQos::default(),
    )
}

pub fn bring_up_with_qos(
    net: &SimNet,
    signing_key: &SigningKey,
    mut store: MemoryRepo,
    connect_root: ed25519_dalek::VerifyingKey,
    proofs: TeamProofs,
    gossip: bool,
    peers: Vec<[u8; 32]>,
    qos: ReconcileQos,
) -> Peer<MemoryRepo> {
    store
        .bind_store_scope(connect_root)
        .expect("simulation store accepts its explicit team scope");
    let id = pk(signing_key);
    let gossip_topic = gossip.then_some(connect_root.to_bytes());
    let harness = net.join(id, gossip_topic);
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
            team: connect_root,
            connect_proof: proofs.connect,
            sync_proof: proofs.sync,
            qos,
        },
        wiring,
    ));
    Peer::with_wiring(store, connect_root, qos, sender, receiver)
        .expect("simulation store was explicitly bound to this team")
}

/// An empty store intentionally independent of CONNECT proof-bundle residency.
pub fn empty_store() -> MemoryRepo {
    MemoryRepo::default()
}
