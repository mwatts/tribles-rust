//! Unified authorized-inventory protocol — deterministic simulation.
//!
//! These scenarios exercise the production host loop over `SimNet`. Gossip is
//! only a lossy wake hint; all semantic evidence and blob bytes cross the same
//! CONNECT + SYNC_TEAM authenticated inventory protocol.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::capability::{
    CapabilityClaim, CapabilityMode, CapabilityProofBundle, CapabilityValidity,
};
use triblespace_core::collection::{CollectionMerge, CollectionRecord, CollectionStore};
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::peer::PeerEvidence;
use triblespace_core::repo::{
    BlobStorePut, CapabilityProofStore, PeerStore, StorageFlush, WantRequest, WantStore,
};
use triblespace_net::inventory::{
    BlobReconcileMode, ReconcileDirection, ReconcileQos, sync_team_capability_atom,
};
use triblespace_net::peer::Peer;
use triblespace_net::protocol::connect_capability_atom;
use triblespace_net::reconcile::Reconciler;
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::*;

type TestPeer = Peer<MemoryRepo>;

fn qos(direction: ReconcileDirection, blobs: BlobReconcileMode) -> ReconcileQos {
    ReconcileQos { direction, blobs }
}

fn record(byte: u8) -> CollectionRecord {
    CollectionRecord::Merge(CollectionMerge::new(
        Inline::new([byte; 32]),
        Inline::new([byte.wrapping_add(1); 32]),
        Inline::new([byte.wrapping_add(2); 32]),
        Inline::new([byte.wrapping_add(3); 32]),
    ))
}

fn put_blob(store: &mut MemoryRepo, byte: u8) -> [u8; 32] {
    store
        .put::<UnknownBlob, _>(Bytes::from_source(vec![byte; 257]))
        .unwrap()
        .raw
}

fn has_blob(peer: &mut TestPeer, hash: [u8; 32]) -> bool {
    peer.try_local(hash).is_some()
}

fn has_record(peer: &TestPeer, expected: CollectionRecord) -> bool {
    peer.store().record(expected.id()).unwrap() == Some(expected)
}

fn has_proof(peer: &TestPeer, expected: &triblespace_core::capability::CapabilityProof) -> bool {
    peer.store().proof(expected.id()).unwrap().as_ref() == Some(expected)
}

fn has_peer(peer: &TestPeer, expected: PeerEvidence) -> bool {
    peer.store()
        .peers()
        .unwrap()
        .any(|candidate| candidate.unwrap() == expected)
}

fn wants(peer: &TestPeer) -> Vec<WantRequest> {
    peer.store()
        .wants()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

async fn step(net: &SimNet, peers: &mut [&mut TestPeer], count: usize) {
    let _ = net;
    for _ in 0..count {
        SimNet::step(&vclock(), Duration::from_millis(20)).await;
        for peer in peers.iter_mut() {
            peer.refresh();
        }
    }
}

fn proof_with_mode(
    root: &ed25519_dalek::SigningKey,
    leaf: &ed25519_dalek::SigningKey,
    atom: triblespace_core::capability::CapabilityAtom,
    mode: CapabilityMode,
    validity: Option<CapabilityValidity>,
) -> CapabilityProofBundle {
    CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(atom, mode, validity),
        leaf.verifying_key(),
    )
    .unwrap()
}

#[test]
fn component_and_blob_qos_share_one_authenticated_protocol() {
    let _guard = sim_guard();
    run_paused(0x1A11_0001, async {
        let net = SimNet::new(0x1A11_0001, SimConfig::default());
        let root = key(0xF1);
        let publisher_key = key(0xA1);
        let demand_key = key(0xB1);
        let mirror_key = key(0xC1);
        let team = root.verifying_key();

        let publisher_proofs = team_proofs(&root, &publisher_key);
        let stored_proof = publisher_proofs.sync.proof().clone();
        let expected_record = record(0x11);
        let mut publisher_store = empty_store();
        publisher_store.insert(expected_record).unwrap();
        publisher_store.insert_proof(stored_proof.clone()).unwrap();
        let blob = put_blob(&mut publisher_store, 0x42);

        let mut publisher = bring_up_with_qos(
            &net,
            &publisher_key,
            publisher_store,
            team,
            publisher_proofs,
            true,
            Vec::new(),
            qos(ReconcileDirection::WriteOnly, BlobReconcileMode::Demand),
        );
        let mut demand = bring_up_with_qos(
            &net,
            &demand_key,
            empty_store(),
            team,
            team_proofs(&root, &demand_key),
            true,
            vec![pk(&publisher_key)],
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Demand),
        );
        let mut mirror = bring_up_with_qos(
            &net,
            &mirror_key,
            empty_store(),
            team,
            team_proofs(&root, &mirror_key),
            true,
            vec![pk(&publisher_key)],
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Mirror),
        );

        step(&net, &mut [&mut publisher, &mut demand, &mut mirror], 300).await;

        let publisher_evidence = PeerEvidence::new(team, publisher_key.verifying_key());
        for peer in [&demand, &mirror] {
            assert!(has_peer(peer, publisher_evidence));
            assert!(has_record(peer, expected_record));
            assert!(has_proof(peer, &stored_proof));
        }
        assert!(
            !has_blob(&mut demand, blob),
            "Demand skips un-WANTed blob inventory"
        );
        assert!(has_blob(&mut mirror, blob), "Mirror copies resident blobs");
        assert!(wants(&mirror).is_empty(), "Mirror is not a durable WANT");
    });
}

#[test]
fn durable_exact_want_fetches_over_the_authenticated_demand_path() {
    let _guard = sim_guard();
    run_paused(0x1A11_0002, async {
        let net = SimNet::new(0x1A11_0002, SimConfig::default());
        let root = key(0xF2);
        let publisher_key = key(0xA2);
        let consumer_key = key(0xB2);
        let team = root.verifying_key();
        let mut publisher_store = empty_store();
        let blob = put_blob(&mut publisher_store, 0x52);

        let mut publisher = bring_up_with_qos(
            &net,
            &publisher_key,
            publisher_store,
            team,
            team_proofs(&root, &publisher_key),
            false,
            Vec::new(),
            qos(ReconcileDirection::WriteOnly, BlobReconcileMode::Demand),
        );
        let mut consumer = bring_up_with_qos(
            &net,
            &consumer_key,
            empty_store(),
            team,
            team_proofs(&root, &consumer_key),
            false,
            vec![pk(&publisher_key)],
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Demand),
        );
        step(&net, &mut [&mut publisher, &mut consumer], 80).await;
        assert!(!has_blob(&mut consumer, blob));

        let request = WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(blob));
        {
            let mut store = consumer.store();
            store.want(request).unwrap();
            store.flush().unwrap();
        }
        let mut reconciler = Reconciler::new().with_fetch_budget(Duration::from_secs(2));
        let mut future = Box::pin(reconciler.tick(&mut consumer));
        let stats = loop {
            if let std::task::Poll::Ready(stats) = futures::poll!(future.as_mut()) {
                break stats;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            publisher.refresh();
        };
        drop(future);

        assert_eq!(stats.fulfilled, 1);
        assert_eq!(stats.pending, 0);
        assert!(has_blob(&mut consumer, blob));
        assert_eq!(wants(&consumer), vec![request]);
    });
}

#[test]
fn read_only_peer_discloses_neither_inventory_nor_blob_bytes() {
    let _guard = sim_guard();
    run_paused(0x1A11_0003, async {
        let net = SimNet::new(0x1A11_0003, SimConfig::default());
        let root = key(0xF3);
        let source_key = key(0xA3);
        let reader_key = key(0xB3);
        let team = root.verifying_key();
        let expected_record = record(0x23);
        let mut source_store = empty_store();
        source_store.insert(expected_record).unwrap();
        let blob = put_blob(&mut source_store, 0x63);

        let mut source = bring_up_with_qos(
            &net,
            &source_key,
            source_store,
            team,
            team_proofs(&root, &source_key),
            false,
            Vec::new(),
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Mirror),
        );
        let mut reader = bring_up_with_qos(
            &net,
            &reader_key,
            empty_store(),
            team,
            team_proofs(&root, &reader_key),
            false,
            vec![pk(&source_key)],
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Mirror),
        );
        step(&net, &mut [&mut source, &mut reader], 180).await;

        assert!(!has_peer(
            &reader,
            PeerEvidence::new(team, source_key.verifying_key())
        ));
        assert!(!has_record(&reader, expected_record));
        assert!(!has_blob(&mut reader, blob));

        let mut future =
            Box::pin(reader.fetch_blob_with_deadline(blob, Duration::from_millis(500)));
        let fetched = loop {
            if let std::task::Poll::Ready(fetched) = futures::poll!(future.as_mut()) {
                break fetched;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            source.refresh();
        };
        assert!(fetched.is_none(), "ReadOnly rejects exact data serving too");
    });
}

#[test]
fn sync_authority_checks_root_leaf_time_and_invoke_mode() {
    let _guard = sim_guard();
    run_paused(0x1A11_0004, async {
        let net = SimNet::new(0x1A11_0004, SimConfig::default());
        let root = key(0xF4);
        let publisher_key = key(0xA4);
        let founder_key = root.clone();
        let delegate_key = key(0xB4);
        let wrong_root_key = key(0xC4);
        let wrong_leaf_key = key(0xD4);
        let expired_key = key(0xE4);
        let unrelated_leaf = key(0x94);
        let alien_root = key(0x84);
        let team = root.verifying_key();
        let expected_record = record(0x34);
        let publisher_evidence = PeerEvidence::new(team, publisher_key.verifying_key());
        let mut publisher_store = empty_store();
        publisher_store.insert(expected_record).unwrap();
        let blob = put_blob(&mut publisher_store, 0x74);

        let mut publisher = bring_up_with_qos(
            &net,
            &publisher_key,
            publisher_store,
            team,
            team_proofs(&root, &publisher_key),
            false,
            Vec::new(),
            qos(ReconcileDirection::WriteOnly, BlobReconcileMode::Demand),
        );

        let mirror = qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Mirror);
        let connect = |leaf: &ed25519_dalek::SigningKey| {
            proof_with_mode(
                &root,
                leaf,
                connect_capability_atom(team),
                CapabilityMode::Invoke,
                None,
            )
        };
        let sync = |issuer: &ed25519_dalek::SigningKey,
                    leaf: &ed25519_dalek::SigningKey,
                    mode,
                    validity| {
            proof_with_mode(
                issuer,
                leaf,
                sync_team_capability_atom(team),
                mode,
                validity,
            )
        };

        let mut founder = bring_up_with_qos(
            &net,
            &founder_key,
            empty_store(),
            team,
            TeamProofs {
                connect: proof_with_mode(
                    &root,
                    &founder_key,
                    connect_capability_atom(team),
                    CapabilityMode::InvokeAndDelegate,
                    None,
                ),
                sync: sync(&root, &founder_key, CapabilityMode::InvokeAndDelegate, None),
            },
            false,
            vec![pk(&publisher_key)],
            mirror,
        );
        let mut delegate_only = bring_up_with_qos(
            &net,
            &delegate_key,
            empty_store(),
            team,
            TeamProofs {
                connect: connect(&delegate_key),
                sync: sync(&root, &delegate_key, CapabilityMode::Delegate, None),
            },
            false,
            vec![pk(&publisher_key)],
            mirror,
        );
        let mut wrong_root = bring_up_with_qos(
            &net,
            &wrong_root_key,
            empty_store(),
            team,
            TeamProofs {
                connect: connect(&wrong_root_key),
                sync: sync(&alien_root, &wrong_root_key, CapabilityMode::Invoke, None),
            },
            false,
            vec![pk(&publisher_key)],
            mirror,
        );
        let mut wrong_leaf = bring_up_with_qos(
            &net,
            &wrong_leaf_key,
            empty_store(),
            team,
            TeamProofs {
                connect: connect(&wrong_leaf_key),
                sync: sync(&root, &unrelated_leaf, CapabilityMode::Invoke, None),
            },
            false,
            vec![pk(&publisher_key)],
            mirror,
        );
        let now_ns = triblespace_core::clock::epoch_now()
            .to_tai_duration()
            .total_nanoseconds();
        let expired = CapabilityValidity::new(
            hifitime::Epoch::from_tai_duration(hifitime::Duration::from_total_nanoseconds(
                now_ns - 2_000_000_000,
            )),
            hifitime::Epoch::from_tai_duration(hifitime::Duration::from_total_nanoseconds(
                now_ns - 1_000_000_000,
            )),
        )
        .unwrap();
        let mut expired_peer = bring_up_with_qos(
            &net,
            &expired_key,
            empty_store(),
            team,
            TeamProofs {
                connect: connect(&expired_key),
                sync: sync(&root, &expired_key, CapabilityMode::Invoke, Some(expired)),
            },
            false,
            vec![pk(&publisher_key)],
            mirror,
        );

        step(
            &net,
            &mut [
                &mut publisher,
                &mut founder,
                &mut delegate_only,
                &mut wrong_root,
                &mut wrong_leaf,
                &mut expired_peer,
            ],
            320,
        )
        .await;

        assert!(has_peer(&founder, publisher_evidence));
        assert!(has_record(&founder, expected_record));
        assert!(has_blob(&mut founder, blob));
        for rejected in [
            &mut delegate_only,
            &mut wrong_root,
            &mut wrong_leaf,
            &mut expired_peer,
        ] {
            assert!(
                !has_peer(rejected, publisher_evidence),
                "rejected SYNC authority must not admit even routing evidence"
            );
            assert!(!has_record(rejected, expected_record));
            assert!(
                !has_blob(rejected, blob),
                "rejected SYNC authority must leak no mirrored content"
            );
        }
    });
}

#[test]
fn periodic_reconciliation_recovers_total_wake_loss() {
    let _guard = sim_guard();
    run_paused(0x1A11_0005, async {
        let net = SimNet::new(
            0x1A11_0005,
            SimConfig {
                gossip_drop_prob: 1.0,
                ..SimConfig::default()
            },
        );
        let root = key(0xF5);
        let publisher_key = key(0xA5);
        let consumer_key = key(0xB5);
        let team = root.verifying_key();
        let mut publisher = bring_up_with_qos(
            &net,
            &publisher_key,
            empty_store(),
            team,
            team_proofs(&root, &publisher_key),
            true,
            Vec::new(),
            qos(ReconcileDirection::WriteOnly, BlobReconcileMode::Demand),
        );
        let mut consumer = bring_up_with_qos(
            &net,
            &consumer_key,
            empty_store(),
            team,
            team_proofs(&root, &consumer_key),
            true,
            vec![pk(&publisher_key)],
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Demand),
        );
        step(&net, &mut [&mut publisher, &mut consumer], 100).await;

        let late = record(0x45);
        publisher.store().insert(late).unwrap();
        publisher.refresh();
        step(&net, &mut [&mut publisher, &mut consumer], 20).await;
        assert!(
            !has_record(&consumer, late),
            "with every gossip wake dropped, the update waits for anti-entropy"
        );

        SimNet::step(&vclock(), Duration::from_secs(31)).await;
        step(&net, &mut [&mut publisher, &mut consumer], 100).await;
        assert!(
            has_record(&consumer, late),
            "periodic authenticated pull is the correctness path"
        );
    });
}

#[test]
fn authenticated_peer_inventory_expands_the_route_set() {
    let _guard = sim_guard();
    run_paused(0x1A11_0006, async {
        let net = SimNet::new(0x1A11_0006, SimConfig::default());
        let root = key(0xF6);
        let bootstrap_key = key(0xA6);
        let discovered_key = key(0xB6);
        let consumer_key = key(0xC6);
        let team = root.verifying_key();
        let discovered_evidence = PeerEvidence::new(team, discovered_key.verifying_key());
        let discovered_record = record(0x56);

        let mut bootstrap_store = empty_store();
        bootstrap_store.insert_peer(discovered_evidence).unwrap();
        let mut discovered_store = empty_store();
        discovered_store.insert(discovered_record).unwrap();

        let mut bootstrap = bring_up_with_qos(
            &net,
            &bootstrap_key,
            bootstrap_store,
            team,
            team_proofs(&root, &bootstrap_key),
            false,
            Vec::new(),
            qos(ReconcileDirection::WriteOnly, BlobReconcileMode::Demand),
        );
        let mut discovered = bring_up_with_qos(
            &net,
            &discovered_key,
            discovered_store,
            team,
            team_proofs(&root, &discovered_key),
            false,
            Vec::new(),
            qos(ReconcileDirection::WriteOnly, BlobReconcileMode::Demand),
        );
        let mut consumer = bring_up_with_qos(
            &net,
            &consumer_key,
            empty_store(),
            team,
            team_proofs(&root, &consumer_key),
            false,
            vec![pk(&bootstrap_key)],
            qos(ReconcileDirection::ReadOnly, BlobReconcileMode::Demand),
        );

        step(
            &net,
            &mut [&mut bootstrap, &mut discovered, &mut consumer],
            400,
        )
        .await;

        assert!(has_peer(&consumer, discovered_evidence));
        assert!(
            has_record(&consumer, discovered_record),
            "the only route to B came from A's authenticated PEER inventory"
        );
    });
}
