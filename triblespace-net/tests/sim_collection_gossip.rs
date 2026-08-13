//! Deterministic end-to-end collection evidence gossip.
//!
//! Exercises the complete simulated host/gossip/Peer path and verifies that
//! collection discovery transfers only the signed sparse evidence. Referenced
//! content remains independently lazy and gossip replay remains idempotent.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{IntoBlob, TryFromBlob};
use triblespace_core::collection::{
    Collection, CollectionGossip, CollectionGossipStore, CollectionRecord, CollectionStore,
    simplearchive_union,
};
use triblespace_core::id::Id;
use triblespace_core::repo::{BlobStore, BlobStoreGet, PinStore, WantStore};
use triblespace_core::trible::{Fragment, TRIBLE_LEN, Trible, TribleSet};
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::{admin_cap, bring_up, key, run_paused, self_cap_of, store_with_caps, vclock};

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

fn archive(byte: u8) -> Blob<SimpleArchive> {
    let mut row = [byte; TRIBLE_LEN];
    row[16..32].fill(byte.wrapping_add(1));
    let mut facts = TribleSet::new();
    facts.insert(&Trible::force_raw(row).unwrap());
    facts.to_blob()
}

#[test]
fn live_gossip_admits_only_sparse_collection_evidence_idempotently() {
    let _guard = common::sim_guard();
    run_paused(0xC011EC_6015, async {
        let root = key(0xF0);
        let author = key(0xA0);
        let receiver = key(0xB0);
        let author_cap = admin_cap(&root, &author);
        let receiver_cap = admin_cap(&root, &receiver);
        let caps = [author_cap.clone(), receiver_cap.clone()];
        let mut author_store = store_with_caps(&caps);
        let receiver_store = store_with_caps(&caps);

        let descriptor = simplearchive_union::descriptor(id(0x31));
        let data = archive(0x41);
        let metadata = archive(0x51);
        let mut fragment = Fragment::from(TribleSet::try_from_blob(data.clone()).unwrap());
        *fragment.metafacts_mut() = TribleSet::try_from_blob(metadata.clone()).unwrap();
        let commit = Collection::new(&mut author_store, id(0x31), author.clone())
            .commit(fragment)
            .unwrap();
        assert_eq!(commit.collection(), descriptor.handle());
        assert_eq!(commit.data(), data.get_handle().into());
        assert_eq!(commit.metadata(), metadata.get_handle());

        let grant = CollectionGossip::sign(&author, descriptor.handle());
        author_store.gossip(grant).unwrap();

        let net = SimNet::new(0xC011EC_6015, SimConfig::default());
        // Join the receiver first so the author's construction-time refresh
        // is itself the publication event under test.
        let mut receiver_peer = bring_up(
            &net,
            &receiver,
            receiver_store,
            root.verifying_key(),
            self_cap_of(&receiver_cap.1),
            true,
        );
        let mut author_peer = bring_up(
            &net,
            &author,
            author_store,
            root.verifying_key(),
            self_cap_of(&author_cap.1),
            true,
        );

        for _ in 0..100 {
            SimNet::step(&vclock(), Duration::from_millis(1)).await;
            author_peer.refresh();
            receiver_peer.refresh();

            let observed = {
                let mut store = receiver_peer.store();
                store
                    .records()
                    .unwrap()
                    .filter_map(Result::ok)
                    .any(|record| record == CollectionRecord::Commit(commit))
            };
            if observed {
                break;
            }
        }

        {
            let mut store = receiver_peer.store();
            let records: Vec<_> = store.records().unwrap().collect::<Result<_, _>>().unwrap();
            let grants: Vec<_> = store.gossips().unwrap().collect::<Result<_, _>>().unwrap();
            assert_eq!(records, vec![CollectionRecord::Commit(commit)]);
            assert_eq!(grants, vec![grant]);

            let reader = store.reader().unwrap();
            assert!(
                reader
                    .get::<TribleSet, SimpleArchive>(descriptor.handle())
                    .is_err()
            );
            assert!(
                reader
                    .get::<TribleSet, SimpleArchive>(data.get_handle())
                    .is_err()
            );
            assert!(
                reader
                    .get::<TribleSet, SimpleArchive>(metadata.get_handle())
                    .is_err()
            );
            assert!(store.wants().unwrap().next().is_none());
            assert!(store.pins().unwrap().next().is_none());
        }

        // Hosts periodically replay their live collection snapshot with a
        // fresh nonce for late joiners. Advance through that real replay path;
        // the grow-only stores must still retain one logical evidence pair.
        SimNet::step(&vclock(), Duration::from_secs(31)).await;
        author_peer.refresh();
        receiver_peer.refresh();

        let mut store = receiver_peer.store();
        assert_eq!(store.records().unwrap().count(), 1);
        assert_eq!(store.gossips().unwrap().count(), 1);
        assert!(store.wants().unwrap().next().is_none());
    });
}

#[test]
fn periodic_replay_reaches_a_late_joiner_without_fetching_content() {
    let _guard = common::sim_guard();
    run_paused(0xC011EC_1A7E, async {
        let root = key(0xE0);
        let author = key(0xA1);
        let receiver = key(0xB1);
        let author_cap = admin_cap(&root, &author);
        let receiver_cap = admin_cap(&root, &receiver);
        let caps = [author_cap.clone(), receiver_cap.clone()];
        let mut author_store = store_with_caps(&caps);
        let receiver_store = store_with_caps(&caps);

        let descriptor = simplearchive_union::descriptor(id(0x32));
        let data = archive(0x42);
        let metadata = archive(0x52);
        let mut fragment = Fragment::from(TribleSet::try_from_blob(data.clone()).unwrap());
        *fragment.metafacts_mut() = TribleSet::try_from_blob(metadata.clone()).unwrap();
        let commit = Collection::new(&mut author_store, id(0x32), author.clone())
            .commit(fragment)
            .unwrap();
        let grant = CollectionGossip::sign(&author, descriptor.handle());
        author_store.gossip(grant).unwrap();

        let net = SimNet::new(0xC011EC_1A7E, SimConfig::default());
        let mut author_peer = bring_up(
            &net,
            &author,
            author_store,
            root.verifying_key(),
            self_cap_of(&author_cap.1),
            true,
        );

        // Let the construction-time publication run while there is nobody on
        // the mesh to receive it. The later receiver can therefore learn this
        // evidence only from the host's periodic replay.
        SimNet::step(&vclock(), Duration::from_millis(100)).await;
        author_peer.refresh();

        let mut receiver_peer = bring_up(
            &net,
            &receiver,
            receiver_store,
            root.verifying_key(),
            self_cap_of(&receiver_cap.1),
            true,
        );
        SimNet::step(&vclock(), Duration::from_millis(100)).await;
        author_peer.refresh();
        receiver_peer.refresh();
        {
            let mut store = receiver_peer.store();
            assert!(store.records().unwrap().next().is_none());
            assert!(store.gossips().unwrap().next().is_none());
        }

        // The simulation clock becomes visible to protocol code after a
        // discrete step. A follow-up host tick observes the elapsed replay
        // interval and broadcasts a freshly nonced frame to the late joiner.
        SimNet::step(&vclock(), Duration::from_secs(31)).await;
        SimNet::step(&vclock(), Duration::from_millis(100)).await;
        for _ in 0..100 {
            author_peer.refresh();
            receiver_peer.refresh();
            let observed = {
                let mut store = receiver_peer.store();
                store
                    .records()
                    .unwrap()
                    .filter_map(Result::ok)
                    .any(|record| record == CollectionRecord::Commit(commit))
            };
            if observed {
                break;
            }
            SimNet::step(&vclock(), Duration::from_millis(1)).await;
        }

        let mut store = receiver_peer.store();
        let records: Vec<_> = store.records().unwrap().collect::<Result<_, _>>().unwrap();
        let grants: Vec<_> = store.gossips().unwrap().collect::<Result<_, _>>().unwrap();
        assert_eq!(records, vec![CollectionRecord::Commit(commit)]);
        assert_eq!(grants, vec![grant]);

        let reader = store.reader().unwrap();
        assert!(
            reader
                .get::<TribleSet, SimpleArchive>(descriptor.handle())
                .is_err()
        );
        assert!(
            reader
                .get::<TribleSet, SimpleArchive>(data.get_handle())
                .is_err()
        );
        assert!(
            reader
                .get::<TribleSet, SimpleArchive>(metadata.get_handle())
                .is_err()
        );
        assert!(store.wants().unwrap().next().is_none());
        assert!(store.pins().unwrap().next().is_none());
    });
}
