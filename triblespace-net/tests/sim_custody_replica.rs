//! Deterministic end-to-end tests for private semantic custody replication.
#![cfg(feature = "sim")]

mod common;

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::capability::{
    CapabilityClaim, CapabilityMode, CapabilityProof, CapabilityProofBundle,
};
use triblespace_core::collection::{
    CollectionCommit, CollectionData, CollectionMerge, CollectionRecord, CollectionStore,
    empty_metadata_handle,
};
use triblespace_core::inline::Inline;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{
    BlobStore, BlobStoreList, BlobStorePut, CapabilityProofStore, WantRequest, WantStore,
};
use triblespace_net::host::{self, PeerConfig, SyncDirection};
use triblespace_net::replica::{
    CustodyReconcileOutcome, CustodyReplica, CustodyReplicaConfig, ReplicaServerConfig,
    ReplicaSetId, replicate_capability_atom,
};
use triblespace_net::transport::sim::{DhtMode, SimConfig, SimNet};

use common::{connect_proof, key, pk, run_paused, vclock};

fn replica_proof(
    root: &SigningKey,
    leaf: &SigningKey,
    replica_set: ReplicaSetId,
) -> CapabilityProofBundle {
    CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(
            replicate_capability_atom(replica_set),
            CapabilityMode::Invoke,
            None,
        ),
        leaf.verifying_key(),
    )
    .unwrap()
}

fn peer_addr(id: [u8; 32]) -> iroh_base::EndpointAddr {
    iroh_base::EndpointAddr::from(EndpointId::from_bytes(&id).unwrap())
}

#[allow(clippy::too_many_arguments)]
fn bring_up(
    net: &SimNet,
    signing_key: &SigningKey,
    store: MemoryRepo,
    connect_root: &SigningKey,
    replica_root: &SigningKey,
    replica_set: ReplicaSetId,
    presented_replica_proof: CapabilityProofBundle,
    peers: Vec<[u8; 32]>,
    receive_dir: &Path,
) -> CustodyReplica<MemoryRepo> {
    let id = pk(signing_key);
    let harness = net.join(id, None);
    let (sender, receiver, wiring) = host::wire(EndpointId::from_bytes(&id).expect("endpoint id"));
    let configured_peers: Vec<_> = peers.into_iter().map(peer_addr).collect();
    let connect = connect_proof(connect_root, signing_key);
    tokio::task::spawn_local(host::run_custody_host(
        harness,
        PeerConfig {
            peers: configured_peers.clone(),
            gossip_topic: None,
            connect_root: connect_root.verifying_key(),
            connect_proof: connect.clone(),
            direction: SyncDirection::ReadOnly,
        },
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
            peers: configured_peers,
            connect_root: connect_root.verifying_key(),
            connect_proof: connect,
            replica_root: replica_root.verifying_key(),
            replica_set,
            replica_proof: presented_replica_proof,
            // The simulator ignores production sockets; with_wiring validates
            // capability semantics but deliberately does not apply the iroh
            // fixed-route preflight.
            bind_addr: "127.0.0.1:49152".parse().unwrap(),
            receive_temp_dir: receive_dir.to_owned(),
        },
    )
    .unwrap()
}

fn seeded_store(label: u8, proof: CapabilityProofBundle, large: bool) -> MemoryRepo {
    let mut store = MemoryRepo::default();
    let len = if large { 1024 * 1024 + 17 } else { 97 };
    store
        .put::<UnknownBlob, _>(Bytes::from(vec![label; len]))
        .unwrap();
    store
        .insert(CollectionRecord::Merge(CollectionMerge::new(
            Inline::new([label; 32]),
            CollectionData::new([label.wrapping_add(1); 32]),
            CollectionData::new([label.wrapping_add(2); 32]),
            CollectionData::new([label.wrapping_add(3); 32]),
        )))
        .unwrap();
    let signing_key = key(label);
    let signed = CollectionCommit::sign(
        &signing_key,
        Inline::new([label.wrapping_add(4); 32]),
        CollectionData::new([label.wrapping_add(5); 32]),
        empty_metadata_handle(),
    );
    let mut invalid_commit_bytes = signed.to_bytes();
    invalid_commit_bytes[128] ^= 0x80;
    let invalid_commit = CollectionCommit::from_bytes(invalid_commit_bytes);
    assert!(invalid_commit.verify_strict().is_err());
    store
        .insert(CollectionRecord::Commit(invalid_commit))
        .unwrap();
    store.insert_proof(proof.proof().clone()).unwrap();
    // Structural custody preserves evidence rather than assigning authority.
    // A canonical K(SCK)+ carrier with a bad signature must therefore survive
    // the union just like a conflicting collection receipt.
    let mut invalid_bytes = proof.proof().as_bytes().to_vec();
    invalid_bytes[32] ^= 0x80;
    let invalid = CapabilityProof::from_bytes(&invalid_bytes).unwrap();
    assert!(invalid.verify_signatures().is_err());
    store.insert_proof(invalid).unwrap();
    store
        .want(WantRequest::blob::<UnknownBlob>(Inline::new(
            [label.wrapping_add(0x40); 32],
        )))
        .unwrap();
    store
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Fingerprint {
    blobs: BTreeSet<[u8; 32]>,
    records: BTreeSet<[u8; 16]>,
    proofs: BTreeSet<[u8; 32]>,
}

impl Fingerprint {
    fn union(mut self, other: Self) -> Self {
        self.blobs.extend(other.blobs);
        self.records.extend(other.records);
        self.proofs.extend(other.proofs);
        self
    }
}

fn fingerprint(store: &mut MemoryRepo) -> Fingerprint {
    let reader = store.reader().unwrap();
    Fingerprint {
        blobs: reader
            .blobs()
            .map(|info| info.unwrap().handle.raw)
            .collect(),
        records: store
            .records()
            .unwrap()
            .map(|record| record.unwrap().id().raw())
            .collect(),
        proofs: store
            .proofs()
            .unwrap()
            .map(|proof| proof.unwrap().id().raw)
            .collect(),
    }
}

fn wants(store: &mut MemoryRepo) -> BTreeSet<WantRequest> {
    store.wants().unwrap().map(|want| want.unwrap()).collect()
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct InvalidEvidence {
    commits: BTreeSet<Vec<u8>>,
    proofs: BTreeSet<Vec<u8>>,
}

impl InvalidEvidence {
    fn union(mut self, other: Self) -> Self {
        self.commits.extend(other.commits);
        self.proofs.extend(other.proofs);
        self
    }
}

fn invalid_evidence(store: &mut MemoryRepo) -> InvalidEvidence {
    InvalidEvidence {
        commits: store
            .records()
            .unwrap()
            .map(|record| record.unwrap())
            .filter_map(|record| match record {
                CollectionRecord::Commit(commit) if commit.verify_strict().is_err() => {
                    Some(CollectionRecord::Commit(commit).to_bytes())
                }
                _ => None,
            })
            .collect(),
        proofs: store
            .proofs()
            .unwrap()
            .map(|proof| proof.unwrap())
            .filter(|proof| proof.verify_signatures().is_err())
            .map(|proof| proof.as_bytes().to_vec())
            .collect(),
    }
}

async fn reconcile_result(
    replica: &mut CustodyReplica<MemoryRepo>,
) -> anyhow::Result<CustodyReconcileOutcome> {
    // The protocol future carries fixed 3×256 inventory summaries. Keep that
    // state on the heap so default-size Rust test threads remain representative
    // rather than requiring RUST_MIN_STACK tuning.
    let mut future = Box::pin(replica.reconcile_once());
    for _ in 0..2_000 {
        if let std::task::Poll::Ready(result) = futures::poll!(future.as_mut()) {
            return result;
        }
        SimNet::step(&vclock(), Duration::from_millis(10)).await;
    }
    panic!("custody reconciliation exceeded deterministic step budget")
}

async fn reconcile(replica: &mut CustodyReplica<MemoryRepo>) -> CustodyReconcileOutcome {
    reconcile_result(replica).await.unwrap()
}

async fn fetch_public_blob(
    peer: &triblespace_net::peer::Peer<MemoryRepo>,
    hash: [u8; 32],
) -> Option<Vec<u8>> {
    let mut future = Box::pin(peer.fetch_blob_with_deadline(hash, Duration::from_millis(500)));
    for _ in 0..100 {
        if let std::task::Poll::Ready(result) = futures::poll!(future.as_mut()) {
            return result;
        }
        SimNet::step(&vclock(), Duration::from_millis(10)).await;
    }
    panic!("public blob probe exceeded deterministic step budget")
}

#[test]
fn healthy_pair_progresses_during_partition_then_restart_and_heal_converge() {
    let _guard = common::sim_guard();
    run_paused(0xC057_0D1A, async {
        let connect_root = key(0xE0);
        let replica_root = key(0xE1);
        let ka = key(0xA0);
        let kb = key(0xB0);
        let kc = key(0xC0);
        let replica_set = ReplicaSetId::new([0x55; 32]);
        let proof_a = replica_proof(&replica_root, &ka, replica_set);
        let proof_b = replica_proof(&replica_root, &kb, replica_set);
        let proof_c = replica_proof(&replica_root, &kc, replica_set);
        let mut store_a = seeded_store(0x11, proof_a.clone(), true);
        let mut store_b = seeded_store(0x22, proof_b.clone(), false);
        let mut store_c = seeded_store(0x33, proof_c.clone(), false);
        let wants_a = wants(&mut store_a);
        let wants_b = wants(&mut store_b);
        let wants_c = wants(&mut store_c);
        let invalid_ac = invalid_evidence(&mut store_a).union(invalid_evidence(&mut store_c));
        let invalid_all = invalid_ac.clone().union(invalid_evidence(&mut store_b));
        let expected_ac = fingerprint(&mut store_a).union(fingerprint(&mut store_c));
        let expected_all = expected_ac.clone().union(fingerprint(&mut store_b));

        let receive = tempfile::tempdir().unwrap();
        let net = SimNet::new(
            0xC057_0D1A,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let mut a = bring_up(
            &net,
            &ka,
            store_a,
            &connect_root,
            &replica_root,
            replica_set,
            proof_a,
            vec![pk(&kb), pk(&kc)],
            receive.path(),
        );
        let b = bring_up(
            &net,
            &kb,
            store_b,
            &connect_root,
            &replica_root,
            replica_set,
            proof_b.clone(),
            vec![pk(&ka), pk(&kc)],
            receive.path(),
        );
        let mut c = bring_up(
            &net,
            &kc,
            store_c,
            &connect_root,
            &replica_root,
            replica_set,
            proof_c,
            vec![pk(&ka), pk(&kb)],
            receive.path(),
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        net.partition(pk(&ka), pk(&kb));
        net.partition(pk(&kb), pk(&kc));
        let a_outcome = reconcile(&mut a).await;
        assert_eq!(a_outcome.peers_completed, 1);
        assert_eq!(a_outcome.peer_errors.len(), 1);
        assert_eq!(fingerprint(a.store_mut()), expected_ac);
        assert_eq!(invalid_evidence(a.store_mut()), invalid_ac);
        assert_eq!(wants(a.store_mut()), wants_a, "WANTs crossed custody union");
        let c_outcome = reconcile(&mut c).await;
        assert_eq!(c_outcome.peers_completed, 1);
        assert_eq!(fingerprint(c.store_mut()), expected_ac);
        assert_eq!(wants(c.store_mut()), wants_c, "WANTs crossed custody union");
        assert_eq!(
            a_outcome.generation, c_outcome.generation,
            "equal semantic inventories produced different generations"
        );

        // Preserve B's store across a process-style network restart.
        net.crash(pk(&kb));
        let store_b = b.shutdown().unwrap();
        SimNet::step(&vclock(), Duration::from_millis(100)).await;
        net.heal(pk(&ka), pk(&kb));
        net.heal(pk(&kb), pk(&kc));
        let mut b = bring_up(
            &net,
            &kb,
            store_b,
            &connect_root,
            &replica_root,
            replica_set,
            proof_b,
            vec![pk(&ka), pk(&kc)],
            receive.path(),
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let b_outcome = reconcile(&mut b).await;
        assert_eq!(b_outcome.peers_completed, 2);
        let a_outcome = reconcile(&mut a).await;
        let c_outcome = reconcile(&mut c).await;
        assert_eq!(fingerprint(a.store_mut()), expected_all);
        assert_eq!(fingerprint(b.store_mut()), expected_all);
        assert_eq!(fingerprint(c.store_mut()), expected_all);
        assert_eq!(invalid_evidence(a.store_mut()), invalid_all);
        assert_eq!(invalid_evidence(b.store_mut()), invalid_all);
        assert_eq!(invalid_evidence(c.store_mut()), invalid_all);
        assert_eq!(wants(a.store_mut()), wants_a);
        assert_eq!(wants(b.store_mut()), wants_b);
        assert_eq!(wants(c.store_mut()), wants_c);
        assert_eq!(a_outcome.generation, b_outcome.generation);
        assert_eq!(a_outcome.generation, c_outcome.generation);

        let idle = reconcile(&mut a).await;
        assert_eq!(idle.blobs_added, 0);
        assert_eq!(idle.collection_records_added, 0);
        assert_eq!(idle.capability_proofs_added, 0);
        assert_eq!(idle.pages_read, 0, "equal inventories still paged");
        assert_eq!(idle.generation, a_outcome.generation);
    });
}

#[test]
fn an_idle_expired_pooled_connection_redials_within_the_same_sweep() {
    let _guard = common::sim_guard();
    run_paused(0xC057_0D1C, async {
        let connect_root = key(0xE8);
        let replica_root = key(0xE9);
        let ka = key(0xA8);
        let kb = key(0xB8);
        let replica_set = ReplicaSetId::new([0x88; 32]);
        let proof_a = replica_proof(&replica_root, &ka, replica_set);
        let proof_b = replica_proof(&replica_root, &kb, replica_set);
        let receive = tempfile::tempdir().unwrap();
        let net = SimNet::new(
            0xC057_0D1C,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let mut a = bring_up(
            &net,
            &ka,
            MemoryRepo::default(),
            &connect_root,
            &replica_root,
            replica_set,
            proof_a,
            vec![pk(&kb)],
            receive.path(),
        );
        let _b = bring_up(
            &net,
            &kb,
            seeded_store(0x58, proof_b.clone(), false),
            &connect_root,
            &replica_root,
            replica_set,
            proof_b,
            vec![pk(&ka)],
            receive.path(),
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        assert_eq!(reconcile(&mut a).await.peers_completed, 1);
        SimNet::step(&vclock(), Duration::from_secs(121)).await;
        let after_idle = reconcile(&mut a).await;
        assert_eq!(after_idle.peers_completed, 1);
        assert!(after_idle.peer_errors.is_empty());
    });
}

#[test]
fn receive_storage_failure_is_local_and_aborts_the_sweep() {
    let _guard = common::sim_guard();
    run_paused(0xC057_0D1D, async {
        let connect_root = key(0xEA);
        let replica_root = key(0xEB);
        let ka = key(0xAA);
        let kb = key(0xBA);
        let replica_set = ReplicaSetId::new([0x99; 32]);
        let proof_a = replica_proof(&replica_root, &ka, replica_set);
        let proof_b = replica_proof(&replica_root, &kb, replica_set);
        let receive_a = tempfile::tempdir().unwrap();
        let receive_a_path = receive_a.path().to_owned();
        let receive_b = tempfile::tempdir().unwrap();
        let net = SimNet::new(
            0xC057_0D1D,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let mut a = bring_up(
            &net,
            &ka,
            MemoryRepo::default(),
            &connect_root,
            &replica_root,
            replica_set,
            proof_a,
            vec![pk(&kb)],
            &receive_a_path,
        );
        let _b = bring_up(
            &net,
            &kb,
            seeded_store(0x59, proof_b.clone(), false),
            &connect_root,
            &replica_root,
            replica_set,
            proof_b,
            vec![pk(&ka)],
            receive_b.path(),
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;
        receive_a.close().unwrap();

        let error = reconcile_result(&mut a).await.unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("admit or flush custody blob component"));
        assert!(message.contains("create replica receive file"));
    });
}

#[test]
fn connect_only_wrong_replica_authority_leaks_no_inventory() {
    let _guard = common::sim_guard();
    run_paused(0xC057_0D1B, async {
        let connect_root = key(0xE4);
        let replica_root = key(0xE5);
        let wrong_replica_root = key(0xE6);
        let server_key = key(0xA4);
        let client_key = key(0xB4);
        let replica_set = ReplicaSetId::new([0x77; 32]);
        let server_proof = replica_proof(&replica_root, &server_key, replica_set);
        let wrong_proof = replica_proof(&wrong_replica_root, &client_key, replica_set);
        let mut server_store = seeded_store(0x44, server_proof.clone(), false);
        let secret_hash = *fingerprint(&mut server_store).blobs.first().unwrap();
        let receive = tempfile::tempdir().unwrap();
        let net = SimNet::new(
            0xC057_0D1B,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let _server = bring_up(
            &net,
            &server_key,
            server_store,
            &connect_root,
            &replica_root,
            replica_set,
            server_proof,
            vec![pk(&client_key)],
            receive.path(),
        );
        let mut client = bring_up(
            &net,
            &client_key,
            MemoryRepo::default(),
            &connect_root,
            &wrong_replica_root,
            replica_set,
            wrong_proof,
            vec![pk(&server_key)],
            receive.path(),
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let outcome = reconcile(&mut client).await;
        assert_eq!(outcome.peers_completed, 0);
        assert_eq!(outcome.peer_errors.len(), 1);
        assert_eq!(outcome.pages_read, 0, "summary authorization leaked a page");
        assert_eq!(outcome.blobs_added, 0);
        assert_eq!(outcome.collection_records_added, 0);
        assert_eq!(outcome.capability_proofs_added, 0);
        assert_eq!(fingerprint(client.store_mut()), Fingerprint::default());

        // A custody endpoint rejects an unconfigured peer before CONNECT and
        // independently rejects ordinary known-hash RPCs after CONNECT. Even
        // with the exact secret hash, no resident custody bytes are disclosed.
        let probe_key = key(0xC4);
        let probe = common::bring_up_with_peers(
            &net,
            &probe_key,
            MemoryRepo::default(),
            connect_root.verifying_key(),
            connect_proof(&connect_root, &probe_key),
            false,
            vec![pk(&server_key)],
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;
        assert_eq!(fetch_public_blob(&probe, secret_hash).await, None);
    });
}
