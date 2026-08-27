//! Artifact-provider directory behavior over the real authenticated host stack.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::repo::BlobStorePut;
use triblespace_net::inventory::{BlobReconcileMode, ReconcileDirection, ReconcileQos};
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::*;

async fn settle(
    peers: &mut [&mut triblespace_net::peer::Peer<
        triblespace_core::repo::memoryrepo::MemoryRepo,
    >],
) {
    for _ in 0..20 {
        SimNet::step(&vclock(), Duration::from_millis(20)).await;
        for peer in peers.iter_mut() {
            peer.refresh();
        }
    }
}

#[test]
fn provider_hints_bind_to_the_announcer_and_repair_after_partition() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C701, async {
        let net = SimNet::new(0xD1AE_C701, SimConfig::default());
        let root = key(0xF1);
        let key_a = key(0xA1);
        let key_b = key(0xB1);
        let team = root.verifying_key();
        let mut a = bring_up_with_peers(
            &net,
            &key_a,
            empty_store(),
            team,
            team_proofs(&root, &key_a),
            vec![pk(&key_b)],
        );
        let mut b = bring_up_with_peers(
            &net,
            &key_b,
            empty_store(),
            team,
            team_proofs(&root, &key_b),
            vec![pk(&key_a)],
        );
        settle(&mut [&mut a, &mut b]).await;

        let artifact = [0x73; 32];
        net.partition(pk(&key_a), pk(&key_b));
        assert_eq!(
            a.announce_artifact(artifact).await,
            1,
            "self lease survives the cut"
        );
        assert!(
            b.find_artifact_providers(artifact).await.is_empty(),
            "a missed lease is a temporary unknown, never a fabricated answer"
        );

        net.heal(pk(&key_a), pk(&key_b));
        assert_eq!(a.announce_artifact(artifact).await, 2);
        let providers = b.find_artifact_providers(artifact).await;
        assert_eq!(providers, vec![a.id()]);
        assert_ne!(providers, vec![b.id()]);
    });
}

#[test]
fn the_same_artifact_id_does_not_cross_team_authorization() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C702, async {
        let net = SimNet::new(0xD1AE_C702, SimConfig::default());
        let root_a = key(0xE1);
        let root_b = key(0xE2);
        let key_a = key(0xA2);
        let key_b = key(0xB2);
        let mut a = bring_up_with_peers(
            &net,
            &key_a,
            empty_store(),
            root_a.verifying_key(),
            team_proofs(&root_a, &key_a),
            vec![pk(&key_b)],
        );
        let mut b = bring_up_with_peers(
            &net,
            &key_b,
            empty_store(),
            root_b.verifying_key(),
            team_proofs(&root_b, &key_b),
            vec![pk(&key_a)],
        );
        settle(&mut [&mut a, &mut b]).await;

        let artifact = [0x74; 32];
        assert_eq!(a.announce_artifact(artifact).await, 1);
        assert!(b.find_artifact_providers(artifact).await.is_empty());
    });
}

#[test]
fn unannounced_holder_is_not_discovered_from_known_peer_evidence() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C704, async {
        let net = SimNet::new(0xD1AE_C704, SimConfig::default());
        let root = key(0xF4);
        let key_a = key(0xA4);
        let key_b = key(0xB4);
        let team = root.verifying_key();
        let bytes = vec![0x84; 257];
        let mut store_a = empty_store();
        let artifact = store_a
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .expect("memory store accepts artifact")
            .raw;
        let mut a = bring_up_with_peers(
            &net,
            &key_a,
            store_a,
            team,
            team_proofs(&root, &key_a),
            vec![pk(&key_b)],
        );
        let mut b = bring_up_with_peers(
            &net,
            &key_b,
            empty_store(),
            team,
            team_proofs(&root, &key_b),
            vec![pk(&key_a)],
        );
        settle(&mut [&mut a, &mut b]).await;

        assert!(b.find_artifact_providers(artifact).await.is_empty());
        assert_eq!(b.fetch_blob(artifact).await, None);
        assert!(b.find_artifact_providers(artifact).await.is_empty());
    });
}

#[test]
fn sparse_line_discovers_and_fetches_an_artifact_across_multiple_hops() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C703, async {
        let net = SimNet::new(0xD1AE_C703, SimConfig::default());
        let root = key(0xF3);
        let keys = [key(0xA3), key(0xB3), key(0xC3), key(0xD3)];
        let ids = keys.each_ref().map(pk);
        let team = root.verifying_key();

        let bytes = vec![0x83; 257];
        let mut store_a = empty_store();
        let artifact = store_a
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .expect("memory store accepts artifact")
            .raw;

        let mut a = bring_up_with_peers(
            &net,
            &keys[0],
            store_a,
            team,
            team_proofs(&root, &keys[0]),
            vec![ids[1]],
        );
        let mut b = bring_up_with_peers(
            &net,
            &keys[1],
            empty_store(),
            team,
            team_proofs(&root, &keys[1]),
            vec![ids[0], ids[2]],
        );
        let mut c = bring_up_with_peers(
            &net,
            &keys[2],
            empty_store(),
            team,
            team_proofs(&root, &keys[2]),
            vec![ids[1], ids[3]],
        );
        let mut d = bring_up_with_peers(
            &net,
            &keys[3],
            empty_store(),
            team,
            team_proofs(&root, &keys[3]),
            vec![ids[2]],
        );
        settle(&mut [&mut a, &mut b, &mut c, &mut d]).await;

        assert!(
            a.announce_artifact(artifact).await >= 3,
            "iterative lookup must leave the source's immediate neighborhood"
        );
        let providers = d.find_artifact_providers(artifact).await;
        assert_eq!(providers, vec![a.id()]);
        assert_eq!(d.fetch_blob(artifact).await, Some(bytes));
    });
}

#[test]
fn stalled_seed_does_not_block_a_healthy_referral() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C705, async {
        let net = SimNet::new(0xD1AE_C705, SimConfig::default());
        let root = key(0xF5);
        let stalled_key = key(0x95);
        let healthy_key = key(0xA5);
        let referred_key = key(0xB5);
        let source_key = key(0xC5);
        let team = root.verifying_key();
        let mut stalled = bring_up(
            &net,
            &stalled_key,
            empty_store(),
            team,
            team_proofs(&root, &stalled_key),
        );
        let mut referred = bring_up_with_peers(
            &net,
            &referred_key,
            empty_store(),
            team,
            team_proofs(&root, &referred_key),
            vec![pk(&healthy_key)],
        );
        let mut healthy = bring_up_with_peers(
            &net,
            &healthy_key,
            empty_store(),
            team,
            team_proofs(&root, &healthy_key),
            vec![pk(&referred_key)],
        );
        let mut source = bring_up_with_qos(
            &net,
            &source_key,
            empty_store(),
            team,
            team_proofs(&root, &source_key),
            vec![pk(&stalled_key), pk(&healthy_key)],
            ReconcileQos {
                direction: ReconcileDirection::WriteOnly,
                blobs: BlobReconcileMode::Demand,
            },
        );
        settle(&mut [&mut stalled, &mut healthy, &mut referred, &mut source]).await;
        net.stall_dials(pk(&stalled_key));

        let artifact = [0x85; 32];
        let mut announcement = Box::pin(source.announce_artifact(artifact));
        let stored = loop {
            if let std::task::Poll::Ready(stored) = futures::poll!(announcement.as_mut()) {
                break stored;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            stalled.refresh();
            healthy.refresh();
            referred.refresh();
        };
        assert!(
            stored >= 3,
            "healthy seed must reveal and query its closer referral before the global deadline"
        );
    });
}

#[test]
fn alpha_black_holed_providers_do_not_starve_a_healthy_exact_fetch() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C706, async {
        let net = SimNet::new(0xD1AE_C706, SimConfig::default());
        let root = key(0xF6);
        let directory_key = key(0x86);
        let bad_key = key(0x91);
        let bad_key_2 = key(0x92);
        let bad_key_3 = key(0x93);
        let good_key = key(0xA6);
        let requester_key = key(0xB6);
        let team = root.verifying_key();
        let bytes = vec![0x86; 257];
        let mut bad_store = empty_store();
        let artifact = bad_store
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .unwrap()
            .raw;
        let mut good_store = empty_store();
        good_store
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .unwrap();

        let mut directory = bring_up_with_qos(
            &net,
            &directory_key,
            empty_store(),
            team,
            team_proofs(&root, &directory_key),
            vec![pk(&bad_key), pk(&bad_key_2), pk(&bad_key_3), pk(&good_key)],
            ReconcileQos {
                direction: ReconcileDirection::WriteOnly,
                blobs: BlobReconcileMode::Demand,
            },
        );
        let mut bad = bring_up_with_peers(
            &net,
            &bad_key,
            bad_store,
            team,
            team_proofs(&root, &bad_key),
            vec![pk(&directory_key)],
        );
        let mut good = bring_up_with_peers(
            &net,
            &good_key,
            good_store,
            team,
            team_proofs(&root, &good_key),
            vec![pk(&directory_key)],
        );
        let mut bad_2 = bring_up_with_peers(
            &net,
            &bad_key_2,
            empty_store(),
            team,
            team_proofs(&root, &bad_key_2),
            vec![pk(&directory_key)],
        );
        let mut bad_3 = bring_up_with_peers(
            &net,
            &bad_key_3,
            empty_store(),
            team,
            team_proofs(&root, &bad_key_3),
            vec![pk(&directory_key)],
        );
        settle(&mut [&mut directory, &mut bad, &mut bad_2, &mut bad_3, &mut good]).await;
        assert!(bad.announce_artifact(artifact).await >= 2);
        assert!(bad_2.announce_artifact(artifact).await >= 2);
        assert!(bad_3.announce_artifact(artifact).await >= 2);
        assert!(good.announce_artifact(artifact).await >= 2);

        net.stall_dials(pk(&bad_key));
        net.stall_dials(pk(&bad_key_2));
        net.stall_dials(pk(&bad_key_3));
        let mut requester = bring_up_with_peers(
            &net,
            &requester_key,
            empty_store(),
            team,
            team_proofs(&root, &requester_key),
            vec![pk(&directory_key)],
        );
        settle(&mut [&mut directory, &mut good, &mut requester]).await;

        let mut fetch = Box::pin(requester.fetch_blob(artifact));
        let fetched = loop {
            if let std::task::Poll::Ready(fetched) = futures::poll!(fetch.as_mut()) {
                break fetched;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            directory.refresh();
            good.refresh();
        };
        assert_eq!(fetched, Some(bytes));
    });
}
