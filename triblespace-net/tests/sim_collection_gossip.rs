//! Deterministic end-to-end collection evidence gossip.
//!
//! Exercises the complete simulated host/gossip/Peer path and verifies that
//! collection discovery transfers only the signed sparse evidence. Referenced
//! content remains independently lazy and gossip replay remains idempotent.
//!
//! These collections declare `reach::public()`, and that declaration is the
//! whole reason their sparse evidence replicates. Relay reach and semantic
//! `WRITE` authority are orthogonal: the first transport fixtures deliberately
//! publish low-level commits without grants, while the final fixture proves
//! that ordinary materialization admits only the separately authorized writer.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;
use triblespace_core::authority::{self, ACTION_WRITE, AuthorityGrant, AuthorityMode};
use triblespace_core::collection::reach;

use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{IntoBlob, TryFromBlob};
use triblespace_core::collection::records::CollectionName;
use triblespace_core::collection::{
    Collection, CollectionHandle, CollectionRecord, CollectionStore, VerifyingKey,
    simplearchive_union,
};
use triblespace_core::repo::{BlobStore, BlobStoreGet, WantStore};
use triblespace_core::trible::Fragment as DescriptorFragment;
use triblespace_core::trible::{Fragment, TRIBLE_LEN, Trible, TribleSet};
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::{bring_up, connect_proof, empty_store, key, run_paused, vclock};

fn collection_name(name: &str) -> CollectionName {
    CollectionName::new(name).unwrap()
}

/// The one team every collection in these simulations belongs to.
fn test_team() -> VerifyingKey {
    key(0xF0).verifying_key()
}

fn named_root(name: &str) -> DescriptorFragment {
    simplearchive_union::descriptor(&collection_name(name), test_team(), reach::public())
}

/// The identity of a descriptor these simulations only address, never store.
fn collection_of(descriptor: &DescriptorFragment) -> CollectionHandle {
    descriptor.facts().clone().to_blob().get_handle()
}

fn archive(byte: u8) -> Blob<SimpleArchive> {
    let mut row = [byte; TRIBLE_LEN];
    row[16..32].fill(byte.wrapping_add(1));
    let mut facts = TribleSet::new();
    facts.insert(&Trible::force_raw(row).unwrap());
    facts.to_blob()
}

/// Drive one inline swarm read while advancing the deterministic network.
async fn drive_future<T, Fut, F>(fut: Fut, mut on_step: F, steps: u32) -> Option<T>
where
    Fut: std::future::Future<Output = T>,
    F: FnMut(),
{
    let mut fut = Box::pin(fut);
    for _ in 0..steps {
        if let std::task::Poll::Ready(value) = futures::poll!(fut.as_mut()) {
            return Some(value);
        }
        SimNet::step(&vclock(), Duration::from_millis(20)).await;
        on_step();
    }
    None
}

#[test]
fn live_gossip_admits_only_sparse_collection_evidence_idempotently() {
    let _guard = common::sim_guard();
    run_paused(0xC011EC_6015, async {
        let root = key(0xF0);
        let author = key(0xA0);
        let receiver = key(0xB0);
        let author_proof = connect_proof(&root, &author);
        let receiver_proof = connect_proof(&root, &receiver);
        let mut author_store = empty_store();
        let receiver_store = empty_store();

        let descriptor = named_root("c31");
        let data = archive(0x41);
        let metadata = archive(0x51);
        let mut fragment = Fragment::from(TribleSet::try_from_blob(data.clone()).unwrap());
        *fragment.metafacts_mut() = TribleSet::try_from_blob(metadata.clone()).unwrap();
        let commit = simplearchive_union::publish_fragment_commit(
            &mut author_store,
            &descriptor,
            fragment,
            &author,
        )
        .unwrap();
        assert_eq!(commit.collection(), collection_of(&descriptor));
        assert_eq!(commit.data(), data.get_handle().into());
        assert_eq!(commit.metadata(), metadata.get_handle());

        let net = SimNet::new(0xC011EC_6015, SimConfig::default());
        // Join the receiver first so the author's construction-time refresh
        // is itself the publication event under test.
        let mut receiver_peer = bring_up(
            &net,
            &receiver,
            receiver_store,
            root.verifying_key(),
            receiver_proof.clone(),
            true,
        );
        let mut author_peer = bring_up(
            &net,
            &author,
            author_store,
            root.verifying_key(),
            author_proof.clone(),
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
            assert_eq!(records, vec![CollectionRecord::Commit(commit)]);

            let reader = store.reader().unwrap();
            assert!(
                reader
                    .get::<TribleSet, SimpleArchive>(collection_of(&descriptor))
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
        }

        // Hosts periodically replay their live collection snapshot with a
        // fresh nonce for late joiners. Advance through that real replay path;
        // the grow-only stores must still retain one logical evidence pair.
        SimNet::step(&vclock(), Duration::from_secs(31)).await;
        author_peer.refresh();
        receiver_peer.refresh();

        let mut store = receiver_peer.store();
        assert_eq!(store.records().unwrap().count(), 1);
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
        let author_proof = connect_proof(&root, &author);
        let receiver_proof = connect_proof(&root, &receiver);
        let mut author_store = empty_store();
        let receiver_store = empty_store();

        let descriptor = named_root("c32");
        let data = archive(0x42);
        let metadata = archive(0x52);
        let mut fragment = Fragment::from(TribleSet::try_from_blob(data.clone()).unwrap());
        *fragment.metafacts_mut() = TribleSet::try_from_blob(metadata.clone()).unwrap();
        let commit = simplearchive_union::publish_fragment_commit(
            &mut author_store,
            &descriptor,
            fragment,
            &author,
        )
        .unwrap();

        let net = SimNet::new(0xC011EC_1A7E, SimConfig::default());
        let mut author_peer = bring_up(
            &net,
            &author,
            author_store,
            root.verifying_key(),
            author_proof.clone(),
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
            receiver_proof.clone(),
            true,
        );
        SimNet::step(&vclock(), Duration::from_millis(100)).await;
        author_peer.refresh();
        receiver_peer.refresh();
        {
            let mut store = receiver_peer.store();
            assert!(store.records().unwrap().next().is_none());
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
        assert_eq!(records, vec![CollectionRecord::Commit(commit)]);

        let reader = store.reader().unwrap();
        assert!(
            reader
                .get::<TribleSet, SimpleArchive>(collection_of(&descriptor))
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
    });
}

#[test]
fn sparse_gossip_retains_all_commits_but_authority_admits_only_the_writer() {
    let _guard = common::sim_guard();
    run_paused(0xA071_0A17, async {
        let root = key(0xF0);
        let writer = key(0xA2);
        let stranger = key(0xC2);
        let receiver = key(0xB2);
        // CONNECT proofs authenticate only the simulated transport endpoints.
        // Target-data admission below comes solely from the positive authority
        // ledger.
        let writer_proof = connect_proof(&root, &writer);
        let receiver_proof = connect_proof(&root, &receiver);
        let mut source_store = empty_store();
        let receiver_store = empty_store();

        let name = collection_name("authority-boundary");
        let target_descriptor = named_root(name.as_str());
        let target = collection_of(&target_descriptor);
        let writer_facts = TribleSet::try_from_blob(archive(0x61)).unwrap();
        let stranger_facts = TribleSet::try_from_blob(archive(0x71)).unwrap();

        let grant = authority::publish_grant(
            &mut source_store,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();
        let writer_commit = Collection::new(
            &mut source_store,
            &name,
            root.verifying_key(),
            writer.clone(),
            reach::public(),
        )
        .commit(Fragment::from(writer_facts.clone()))
        .unwrap();
        let stranger_commit = simplearchive_union::publish_fragment_commit(
            &mut source_store,
            &target_descriptor,
            Fragment::from(stranger_facts.clone()),
            &stranger,
        )
        .unwrap();

        let net = SimNet::new(0xA071_0A17, SimConfig::default());
        let mut receiver_peer = bring_up(
            &net,
            &receiver,
            receiver_store,
            root.verifying_key(),
            receiver_proof.clone(),
            true,
        );
        let mut source_peer = bring_up(
            &net,
            &writer,
            source_store,
            root.verifying_key(),
            writer_proof.clone(),
            true,
        );

        let expected_records = std::collections::BTreeSet::from([
            grant.id(),
            writer_commit.id(),
            stranger_commit.id(),
        ]);
        for _ in 0..100 {
            SimNet::step(&vclock(), Duration::from_millis(1)).await;
            source_peer.refresh();
            receiver_peer.refresh();
            let observed = receiver_peer
                .store()
                .records()
                .unwrap()
                .filter_map(Result::ok)
                .filter_map(|record| match record {
                    CollectionRecord::Commit(commit) => Some(commit.id()),
                    _ => None,
                })
                .collect::<std::collections::BTreeSet<_>>();
            if observed == expected_records {
                break;
            }
        }

        let stored = receiver_peer
            .store()
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            stored
                .iter()
                .filter_map(|record| match record {
                    CollectionRecord::Commit(commit) => Some(commit.id()),
                    _ => None,
                })
                .collect::<std::collections::BTreeSet<_>>(),
            expected_records,
            "gossip storage retains valid evidence without deciding authority"
        );
        assert!(receiver_peer.try_local(grant.data().raw).is_none());
        assert!(receiver_peer.try_local(writer_commit.data().raw).is_none());
        assert!(
            receiver_peer
                .try_local(stranger_commit.data().raw)
                .is_none()
        );

        // Sparse gossip carried only the records. Fetch every referenced blob,
        // including the stranger's valid data, so physical absence cannot be
        // what excludes that commit from the semantic snapshot below.
        let mut closure = vec![
            authority::collection(root.verifying_key()).raw,
            target.raw,
            grant.data().raw,
            grant.metadata().raw,
            writer_commit.data().raw,
            writer_commit.metadata().raw,
            stranger_commit.data().raw,
            stranger_commit.metadata().raw,
        ];
        closure.sort_unstable();
        closure.dedup();
        for hash in closure {
            let fetched = drive_future(
                receiver_peer.get_or_fetch_async(hash),
                || source_peer.refresh(),
                200,
            )
            .await
            .expect("swarm fetch completes")
            .expect("MemoryRepo records the want")
            .expect("source serves every referenced blob");
            assert_eq!(blake3::hash(&fetched).as_bytes(), &hash);
        }
        assert!(
            receiver_peer
                .try_local(stranger_commit.data().raw)
                .is_some()
        );

        let authority = {
            let mut store = receiver_peer.store();
            authority::resolve_authority(&mut *store, root.verifying_key()).unwrap()
        };
        assert!(authority.allows(&writer_commit.public_key(), ACTION_WRITE, target));
        assert!(!authority.allows(&stranger_commit.public_key(), ACTION_WRITE, target));

        let snapshot = {
            let mut store = receiver_peer.store();
            let mut collection = Collection::new(
                &mut *store,
                &name,
                root.verifying_key(),
                receiver.clone(),
                reach::public(),
            );
            collection.snapshot().unwrap()
        };
        assert_eq!(snapshot.commits(), &[writer_commit]);
        assert_eq!(snapshot.facts(), &writer_facts);
        assert_ne!(snapshot.facts(), &stranger_facts);
    });
}
