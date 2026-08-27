//! Element-provider directory behavior over the real authenticated host stack.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

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
            false,
            vec![pk(&key_b)],
        );
        let mut b = bring_up_with_peers(
            &net,
            &key_b,
            empty_store(),
            team,
            team_proofs(&root, &key_b),
            false,
            vec![pk(&key_a)],
        );
        settle(&mut [&mut a, &mut b]).await;

        let element = [0x73; 32];
        net.partition(pk(&key_a), pk(&key_b));
        assert_eq!(
            a.announce_element(element).await,
            1,
            "self lease survives the cut"
        );
        assert!(
            b.find_element_providers(element).await.is_empty(),
            "a missed lease is a temporary unknown, never a fabricated answer"
        );

        net.heal(pk(&key_a), pk(&key_b));
        assert_eq!(a.announce_element(element).await, 2);
        let providers = b.find_element_providers(element).await;
        assert_eq!(providers, vec![a.id()]);
        assert_ne!(providers, vec![b.id()]);
    });
}

#[test]
fn the_same_element_id_does_not_cross_team_authorization() {
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
            false,
            vec![pk(&key_b)],
        );
        let mut b = bring_up_with_peers(
            &net,
            &key_b,
            empty_store(),
            root_b.verifying_key(),
            team_proofs(&root_b, &key_b),
            false,
            vec![pk(&key_a)],
        );
        settle(&mut [&mut a, &mut b]).await;

        let element = [0x74; 32];
        assert_eq!(a.announce_element(element).await, 1);
        assert!(b.find_element_providers(element).await.is_empty());
    });
}
