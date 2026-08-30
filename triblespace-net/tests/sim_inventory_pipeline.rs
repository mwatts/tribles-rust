//! Deterministic first-sync coverage for the bounded inventory pipeline.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use triblespace_core::collection::{
    CollectionMerge, CollectionRead, CollectionRecord, CollectionStore,
};
use triblespace_core::inline::Inline;
use triblespace_core::repo::SnapshotSource;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_net::inventory::{BlobReconcileMode, ReconcileDirection, ReconcileQos};
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::*;

const RECORD_COUNT: u16 = 768;

fn record(index: u16) -> CollectionRecord {
    let field = |tag: u8| {
        let mut bytes = [0; 32];
        bytes[0] = tag;
        bytes[1..3].copy_from_slice(&index.to_be_bytes());
        bytes
    };
    CollectionRecord::Merge(CollectionMerge::new(
        Inline::new(field(1)),
        Inline::new(field(2)),
        Inline::new(field(3)),
        Inline::new(field(4)),
    ))
}

#[test]
fn one_store_drain_admits_more_than_the_old_per_leaf_bridge_capacity() {
    let _guard = sim_guard();
    run_paused(0x1A11_BA7C, async {
        let root = key(0xF7);
        let publisher_key = key(0xA7);
        let consumer_key = key(0xB7);
        let team = root.verifying_key();
        let net = SimNet::new(0x1A11_BA7C, SimConfig::default());

        let mut publisher_store = MemoryRepo::default();
        for index in 0..RECORD_COUNT {
            publisher_store.insert(record(index)).unwrap();
        }
        let mut publisher = bring_up_with_qos(
            &net,
            &publisher_key,
            publisher_store,
            team,
            team_proofs(&root, &publisher_key),
            Vec::new(),
            ReconcileQos {
                direction: ReconcileDirection::WriteOnly,
                blobs: BlobReconcileMode::Demand,
            },
        );
        let mut consumer = bring_up_with_qos(
            &net,
            &consumer_key,
            MemoryRepo::default(),
            team,
            team_proofs(&root, &consumer_key),
            vec![pk(&publisher_key)],
            ReconcileQos {
                direction: ReconcileDirection::ReadOnly,
                blobs: BlobReconcileMode::Demand,
            },
        );

        // Let the complete authenticated walk fill the bounded batch bridge,
        // but deliberately do not refresh the synchronous consumer yet. The
        // former one-item/64-slot bridge stalled after 64 leaves here.
        for _ in 0..2_000 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
        }
        {
            let mut store = consumer.store();
            let snapshot = store.snapshot().unwrap();
            assert_eq!(snapshot.records().unwrap().count(), 0);
        }

        // A single drain admits all queued batches and crosses one durability
        // barrier before publishing the replacement local snapshot.
        consumer.refresh();
        let received = {
            let mut store = consumer.store();
            let snapshot = store.snapshot().unwrap();
            snapshot
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(received.len(), usize::from(RECORD_COUNT));
        for index in 0..RECORD_COUNT {
            assert!(received.contains(&record(index)));
        }

        // Keep the publisher live through the assertion; dropping it earlier
        // would intentionally shut down its host command loop.
        publisher.refresh();
    });
}
