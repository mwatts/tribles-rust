//! Artifact-provider directory behavior over the real authenticated host stack.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use anybytes::Bytes;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::repo::{ArtifactHandle, ArtifactOfferStore, BlobStoreKeep, BlobStorePut};
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
fn automatic_offer_retries_remote_replication_after_partition() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C701, async {
        let net = SimNet::new(0xD1AE_C701, SimConfig::default());
        let root = key(0xF1);
        let key_a = key(0xA1);
        let key_b = key(0xB1);
        let team = root.verifying_key();
        let bytes = vec![0x73; 257];
        let mut store_a = empty_store();
        let artifact = store_a
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .unwrap()
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

        net.partition(pk(&key_a), pk(&key_b));
        offer_resident(&mut a, artifact).await;
        assert!(
            b.find_artifact_providers(artifact).await.is_empty(),
            "a missed lease is a temporary unknown, never a fabricated answer"
        );

        net.heal(pk(&key_a), pk(&key_b));
        SimNet::step(&vclock(), Duration::from_secs(2)).await;
        settle(&mut [&mut a, &mut b]).await;
        net.partition(pk(&key_a), pk(&key_b));
        let providers = b.find_artifact_providers(artifact).await;
        assert_eq!(
            providers,
            vec![a.id()],
            "the retry populated B's remote directory; A is unreachable during this lookup"
        );
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
        let bytes = vec![0x74; 257];
        let mut store_a = empty_store();
        let artifact = store_a
            .put::<UnknownBlob, _>(Bytes::from_source(bytes))
            .unwrap()
            .raw;
        store_a.offer(ArtifactHandle::new(artifact)).unwrap();
        let mut a = bring_up_with_peers(
            &net,
            &key_a,
            store_a,
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
fn offer_only_change_publishes_an_already_resident_blob() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C707, async {
        let net = SimNet::new(0xD1AE_C707, SimConfig::default());
        let root = key(0xF7);
        let source_key = key(0xA7);
        let reader_key = key(0xB7);
        let team = root.verifying_key();
        let bytes = vec![0x87; 257];
        let mut source_store = empty_store();
        let artifact = source_store
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .unwrap()
            .raw;
        let mut source = bring_up_with_peers(
            &net,
            &source_key,
            source_store,
            team,
            team_proofs(&root, &source_key),
            vec![pk(&reader_key)],
        );
        let mut reader = bring_up_with_peers(
            &net,
            &reader_key,
            empty_store(),
            team,
            team_proofs(&root, &reader_key),
            vec![pk(&source_key)],
        );
        settle(&mut [&mut source, &mut reader]).await;
        assert!(reader.find_artifact_providers(artifact).await.is_empty());

        offer_resident(&mut source, artifact).await;
        assert_eq!(
            reader.find_artifact_providers(artifact).await,
            vec![source.id()]
        );
        assert_eq!(reader.fetch_blob(artifact).await, Some(bytes));
    });
}

#[test]
fn offered_absent_blob_stays_dormant_until_resident() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C708, async {
        let net = SimNet::new(0xD1AE_C708, SimConfig::default());
        let root = key(0xF8);
        let source_key = key(0xA8);
        let reader_key = key(0xB8);
        let team = root.verifying_key();
        let bytes = vec![0x88; 257];
        let artifact = *blake3::hash(&bytes).as_bytes();
        let mut source = bring_up_with_peers(
            &net,
            &source_key,
            empty_store(),
            team,
            team_proofs(&root, &source_key),
            vec![pk(&reader_key)],
        );
        let mut reader = bring_up_with_peers(
            &net,
            &reader_key,
            empty_store(),
            team,
            team_proofs(&root, &reader_key),
            vec![pk(&source_key)],
        );
        settle(&mut [&mut source, &mut reader]).await;

        source.store().offer(ArtifactHandle::new(artifact)).unwrap();
        source.refresh();
        settle(&mut [&mut source, &mut reader]).await;
        assert!(reader.find_artifact_providers(artifact).await.is_empty());

        let landed = source
            .store()
            .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
            .unwrap();
        assert_eq!(landed.raw, artifact);
        source.refresh();
        settle(&mut [&mut source, &mut reader]).await;
        assert_eq!(
            reader.find_artifact_providers(artifact).await,
            vec![source.id()]
        );
        assert_eq!(reader.fetch_blob(artifact).await, Some(bytes));
    });
}

#[test]
fn readonly_offer_is_dormant() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C709, async {
        let net = SimNet::new(0xD1AE_C709, SimConfig::default());
        let root = key(0xF9);
        let source_key = key(0xA9);
        let reader_key = key(0xB9);
        let team = root.verifying_key();
        let mut source_store = empty_store();
        let artifact = source_store
            .put::<UnknownBlob, _>(Bytes::from_source(vec![0x89; 257]))
            .unwrap()
            .raw;
        source_store.offer(ArtifactHandle::new(artifact)).unwrap();
        let mut source = bring_up_with_qos(
            &net,
            &source_key,
            source_store,
            team,
            team_proofs(&root, &source_key),
            vec![pk(&reader_key)],
            ReconcileQos {
                direction: ReconcileDirection::ReadOnly,
                blobs: BlobReconcileMode::Demand,
            },
        );
        let mut reader = bring_up_with_peers(
            &net,
            &reader_key,
            empty_store(),
            team,
            team_proofs(&root, &reader_key),
            vec![pk(&source_key)],
        );
        settle(&mut [&mut source, &mut reader]).await;
        assert!(reader.find_artifact_providers(artifact).await.is_empty());
        assert_eq!(reader.fetch_blob(artifact).await, None);
    });
}

#[test]
fn automatic_publication_renews_before_provider_lease_expiry() {
    let _guard = sim_guard();
    run_paused(0xD1AE_C70A, async {
        let net = SimNet::new(0xD1AE_C70A, SimConfig::default());
        let root = key(0xFA);
        let source_key = key(0xAA);
        let reader_key = key(0xBA);
        let team = root.verifying_key();
        let mut source_store = empty_store();
        let artifact = source_store
            .put::<UnknownBlob, _>(Bytes::from_source(vec![0x8A; 257]))
            .unwrap()
            .raw;
        source_store.offer(ArtifactHandle::new(artifact)).unwrap();
        let mut source = bring_up_with_peers(
            &net,
            &source_key,
            source_store,
            team,
            team_proofs(&root, &source_key),
            vec![pk(&reader_key)],
        );
        let mut reader = bring_up_with_peers(
            &net,
            &reader_key,
            empty_store(),
            team,
            team_proofs(&root, &reader_key),
            vec![pk(&source_key)],
        );
        settle(&mut [&mut source, &mut reader]).await;
        assert_eq!(
            reader.find_artifact_providers(artifact).await,
            vec![source.id()]
        );

        // Lease policy reads the injectable protocol clock. Jump it directly
        // rather than making the paused Tokio runtime visit 4.6 million
        // unrelated 10 ms host-poll timers on the way to the half-life.
        vclock().advance(Duration::from_secs(13 * 60 * 60));
        settle(&mut [&mut source, &mut reader]).await;
        vclock().advance(Duration::from_secs(12 * 60 * 60 + 1));
        settle(&mut [&mut source, &mut reader]).await;

        assert_eq!(
            reader.find_artifact_providers(artifact).await,
            vec![source.id()],
            "the original 24-hour lease has elapsed, so this is a renewal"
        );
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

        offer_resident(&mut a, artifact).await;
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
        let bytes = vec![0x85; 257];
        let mut source_store = empty_store();
        let artifact = source_store
            .put::<UnknownBlob, _>(Bytes::from_source(bytes))
            .unwrap()
            .raw;
        let mut source = bring_up_with_qos(
            &net,
            &source_key,
            source_store,
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

        offer_resident(&mut source, artifact).await;
        SimNet::step(&vclock(), Duration::from_secs(4)).await;
        settle(&mut [&mut stalled, &mut healthy, &mut referred, &mut source]).await;

        net.partition(pk(&referred_key), pk(&source_key));
        net.partition(pk(&referred_key), pk(&healthy_key));
        net.partition(pk(&referred_key), pk(&stalled_key));
        assert_eq!(
            referred.find_artifact_providers(artifact).await,
            vec![source.id()],
            "the healthy seed revealed a closer remote directory despite the stalled seed"
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
        let make_offered_store = || {
            let mut store = empty_store();
            let artifact = store
                .put::<UnknownBlob, _>(Bytes::from_source(bytes.clone()))
                .unwrap()
                .raw;
            store.offer(ArtifactHandle::new(artifact)).unwrap();
            (store, artifact)
        };
        let (bad_store, artifact) = make_offered_store();
        let (bad_store_2, artifact_2) = make_offered_store();
        let (bad_store_3, artifact_3) = make_offered_store();
        let (good_store, good_artifact) = make_offered_store();
        assert_eq!([artifact_2, artifact_3, good_artifact], [artifact; 3]);

        let mut directory = bring_up_with_peers(
            &net,
            &directory_key,
            empty_store(),
            team,
            team_proofs(&root, &directory_key),
            vec![pk(&bad_key), pk(&bad_key_2), pk(&bad_key_3), pk(&good_key)],
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
            bad_store_2,
            team,
            team_proofs(&root, &bad_key_2),
            vec![pk(&directory_key)],
        );
        let mut bad_3 = bring_up_with_peers(
            &net,
            &bad_key_3,
            bad_store_3,
            team,
            team_proofs(&root, &bad_key_3),
            vec![pk(&directory_key)],
        );
        settle(&mut [&mut directory, &mut bad, &mut bad_2, &mut bad_3, &mut good]).await;
        SimNet::step(&vclock(), Duration::from_secs(4)).await;
        settle(&mut [&mut directory, &mut bad, &mut bad_2, &mut bad_3, &mut good]).await;
        let published = directory.find_artifact_providers(artifact).await;
        for provider in [bad.id(), bad_2.id(), bad_3.id(), good.id()] {
            assert!(
                published.contains(&provider),
                "all truthful offers must reach the shared directory before eviction; missing {provider} from {published:?}"
            );
        }

        // A stale lease is production-realistic: each bad provider really
        // offered resident bytes, then evicted them without a non-monotone
        // unpublish operation. The directory hint remains soft until expiry.
        for peer in [&mut bad, &mut bad_2, &mut bad_3] {
            peer.store().keep(Vec::<ArtifactHandle>::new());
            peer.refresh();
            assert!(peer.try_local(artifact).is_none());
        }

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
