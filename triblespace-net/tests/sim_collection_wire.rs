//! End-to-end sparse collection evidence transfer over the complete
//! authenticated host/protocol/simulator stack.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use ed25519_dalek::SigningKey;
use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{IntoBlob, TryFromBlob};
use triblespace_core::collection::records::CollectionName;
use triblespace_core::collection::{
    Collection, CollectionData, CollectionDerive, CollectionGossip, CollectionGossipStore,
    CollectionHandle, CollectionMerge, CollectionRecord, CollectionStore, VerifyingKey,
    simplearchive_union,
};
use triblespace_core::trible::Fragment as DescriptorFragment;
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::time::NsTAIInterval;
use triblespace_core::inline::{Inline, TryToInline};
use triblespace_core::repo::WantRequest;
use triblespace_core::repo::capability::{self, PERM_READ, scope_branch};
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut, PinStore, WantStore};
use triblespace_core::trible::{Fragment, TRIBLE_LEN, Trible, TribleSet};
use triblespace_net::peer::Peer;
use triblespace_net::transport::sim::{DhtMode, SimConfig, SimNet};

use common::{
    admin_cap, bring_up, bring_up_with_peers, key, pk, run_paused, self_cap_of, store_with_caps,
    vclock,
};

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

fn collection_name(name: &str) -> CollectionName {
    CollectionName::new(name).unwrap()
}

/// The one team every collection in these simulations belongs to.
fn test_team() -> VerifyingKey {
    key(0xF3).verifying_key()
}

/// A named root of the canonical `SimpleArchive` union kind.
fn named_root(name: &str) -> DescriptorFragment {
    simplearchive_union::descriptor(&collection_name(name), test_team())
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

fn data(byte: u8) -> CollectionData {
    Inline::new([byte; 32])
}

fn branch_restricted_read_cap(
    root: &SigningKey,
    subject: &SigningKey,
) -> (Blob<SimpleArchive>, Blob<SimpleArchive>) {
    use triblespace_core::id::{ExclusiveId, ufoid};
    use triblespace_core::macros::entity;

    let scope_root = *ufoid();
    let branch = *ufoid();
    let mut scope_facts = TribleSet::from(entity! {
        ExclusiveId::force_ref(&scope_root) @
        triblespace_core::metadata::tag: PERM_READ,
    });
    scope_facts += TribleSet::from(entity! {
        ExclusiveId::force_ref(&scope_root) @
        scope_branch: branch,
    });
    let now = triblespace_core::clock::epoch_now();
    let expiry: Inline<NsTAIInterval> = (now, now + hifitime::Duration::from_days(30.0))
        .try_to_inline()
        .unwrap();
    capability::build_capability(
        root,
        subject.verifying_key(),
        None,
        scope_root,
        scope_facts,
        expiry,
    )
    .unwrap()
}

#[test]
fn direct_collection_evidence_fetch_is_verified_and_does_not_fetch_or_admit_blobs() {
    let _guard = common::sim_guard();
    run_paused(0xC011EC7, async {
        let root = key(0xF0);
        let server_key = key(0xA0);
        let client_key = key(0xB0);
        let (server_cap, server_sig) = admin_cap(&root, &server_key);
        let (client_cap, client_sig) = admin_cap(&root, &client_key);

        let mut server_store = store_with_caps(&[
            (server_cap.clone(), server_sig.clone()),
            (client_cap.clone(), client_sig.clone()),
        ]);
        let client_store = store_with_caps(&[
            (server_cap, server_sig.clone()),
            (client_cap, client_sig.clone()),
        ]);

        let descriptor = named_root("c1");
        let data = archive(2);
        let metadata = archive(3);
        server_store
            .put::<SimpleArchive, _>(descriptor.clone().into_facts().to_blob())
            .unwrap();
        server_store.put::<SimpleArchive, _>(data.clone()).unwrap();
        server_store
            .put::<SimpleArchive, _>(metadata.clone())
            .unwrap();
        let mut fragment = Fragment::from(TribleSet::try_from_blob(data.clone()).unwrap());
        *fragment.metafacts_mut() = TribleSet::try_from_blob(metadata.clone()).unwrap();
        let mut collection = Collection::new(
            &mut server_store,
            &collection_name("c1"),
            test_team(),
            server_key.clone(),
        );
        let commit = collection.commit(fragment).unwrap();
        server_store
            .gossip(CollectionGossip::sign(&server_key, collection_of(&descriptor)))
            .unwrap();

        let net = SimNet::new(0xC011EC7, SimConfig::default());
        let server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            self_cap_of(&server_sig),
            false,
        );
        let client = bring_up(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_sig),
            false,
        );

        // Let both host tasks publish their capabilities before the direct op.
        SimNet::step(&vclock(), Duration::from_millis(1)).await;
        let (client, fetched) =
            fetch_evidence_while_stepping(client, pk(&server_key), collection_of(&descriptor)).await;

        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].commit(), commit);
        assert_eq!(fetched[0].grant().collection(), collection_of(&descriptor));

        // Fetch is inert sparse evidence: neither the commit/grant nor any
        // blob it names is admitted into the client store.
        assert!(client.store().records().unwrap().next().is_none());
        assert!(client.store().gossips().unwrap().next().is_none());
        let reader = client.store().reader().unwrap();
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

        // Evidence enumeration does not disturb the serving peer's blob store.
        let server_reader = server.store().reader().unwrap();
        assert_eq!(
            server_reader
                .get::<TribleSet, SimpleArchive>(data.get_handle())
                .unwrap(),
            TribleSet::try_from_blob(data).unwrap(),
        );
    });
}

#[test]
fn direct_collection_evidence_fetch_omits_commits_without_author_grants() {
    let _guard = common::sim_guard();
    run_paused(0xC011EC8, async {
        let root = key(0xF1);
        let server_key = SigningKey::from_bytes(&[0xA1; 32]);
        let client_key = SigningKey::from_bytes(&[0xB1; 32]);
        let (server_cap, server_sig) = admin_cap(&root, &server_key);
        let (client_cap, client_sig) = admin_cap(&root, &client_key);
        let mut server_store = store_with_caps(&[
            (server_cap.clone(), server_sig.clone()),
            (client_cap.clone(), client_sig.clone()),
        ]);
        let client_store = store_with_caps(&[
            (server_cap, server_sig.clone()),
            (client_cap, client_sig.clone()),
        ]);

        let descriptor = named_root("c4");
        let data: Blob<SimpleArchive> = TribleSet::new().to_blob();
        server_store
            .put::<SimpleArchive, _>(descriptor.clone().into_facts().to_blob())
            .unwrap();
        server_store.put::<SimpleArchive, _>(data.clone()).unwrap();
        Collection::new(
            &mut server_store,
            &collection_name("c4"),
            test_team(),
            server_key.clone(),
        )
            .commit(Fragment::empty())
            .unwrap();

        let net = SimNet::new(0xC011EC8, SimConfig::default());
        let _server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            self_cap_of(&server_sig),
            false,
        );
        let client = bring_up(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_sig),
            false,
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let (_client, fetched) =
            fetch_evidence_while_stepping(client, pk(&server_key), collection_of(&descriptor)).await;
        assert!(fetched.is_empty());
    });
}

#[test]
fn branch_restricted_capability_cannot_enumerate_collections() {
    let _guard = common::sim_guard();
    run_paused(0xC011EC9, async {
        let root = key(0xF2);
        let server_key = key(0xA2);
        let client_key = key(0xB2);
        let (server_cap, server_sig) = admin_cap(&root, &server_key);
        let (client_cap, client_sig) = branch_restricted_read_cap(&root, &client_key);
        let server_store = store_with_caps(&[
            (server_cap.clone(), server_sig.clone()),
            (client_cap.clone(), client_sig.clone()),
        ]);
        let client_store = store_with_caps(&[
            (server_cap, server_sig.clone()),
            (client_cap, client_sig.clone()),
        ]);

        let net = SimNet::new(0xC011EC9, SimConfig::default());
        let _server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            self_cap_of(&server_sig),
            false,
        );
        let client = bring_up(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_sig),
            false,
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let collection = collection_of(&named_root("c5"));
        let (_client, result) =
            fetch_evidence_result_while_stepping(client, pk(&server_key), collection).await;
        let error = result.unwrap_err();
        assert!(error.to_string().contains("unrestricted read"));
    });
}

#[test]
fn direct_collection_reconcile_admits_sparse_evidence_without_blobs_pins_or_wants() {
    let _guard = common::sim_guard();
    run_paused(0xC011ECA, async {
        let root = key(0xF3);
        let server_key = key(0xA3);
        let client_key = key(0xB3);
        let (server_cap, server_sig) = admin_cap(&root, &server_key);
        let (client_cap, client_sig) = admin_cap(&root, &client_key);
        let mut server_store = store_with_caps(&[
            (server_cap.clone(), server_sig.clone()),
            (client_cap.clone(), client_sig.clone()),
        ]);
        let client_store = store_with_caps(&[
            (server_cap, server_sig.clone()),
            (client_cap, client_sig.clone()),
        ]);

        let descriptor = named_root("c6");
        let data = archive(7);
        let facts = TribleSet::try_from_blob(data.clone()).unwrap();
        let mut collection = Collection::new(
            &mut server_store,
            &collection_name("c6"),
            test_team(),
            server_key.clone(),
        );
        let commit = collection.commit(Fragment::from(facts)).unwrap();
        let grant = CollectionGossip::sign(&server_key, collection_of(&descriptor));
        server_store.gossip(grant).unwrap();

        let net = SimNet::new(0xC011ECA, SimConfig::default());
        let _server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            self_cap_of(&server_sig),
            false,
        );
        let client = bring_up(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_sig),
            false,
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let worker = std::thread::spawn(move || {
            let mut client = client;
            let outcome = client
                .reconcile_collection_from(pk(&server_key), collection_of(&descriptor), |_, _| {
                    Ok::<_, std::convert::Infallible>(true)
                })
                .unwrap();
            (client, outcome)
        });
        while !worker.is_finished() {
            SimNet::step(&vclock(), Duration::from_millis(1)).await;
        }
        let (client, outcome) = worker.join().unwrap();
        assert_eq!(outcome.observed, 1);
        assert_eq!(outcome.admitted, 1);
        assert_eq!(outcome.denied, 0);

        let mut store = client.store();
        assert!(store.records().unwrap().any(|record| {
            matches!(record, Ok(triblespace_core::collection::CollectionRecord::Commit(found)) if found == commit)
        }));
        assert!(store.gossips().unwrap().any(|found| found == Ok(grant)));
        let reader = store.reader().unwrap();
        assert!(
            reader
                .get::<TribleSet, SimpleArchive>(data.get_handle())
                .is_err()
        );
        assert!(store.pins().unwrap().next().is_none());
        assert!(store.wants().unwrap().next().is_none());
    });
}

#[test]
fn configured_peer_probe_roundtrips_exact_operation_receipts_without_dht_or_gossip_grants() {
    let _guard = common::sim_guard();
    run_paused(0xC011ECB, async {
        let root = key(0xF4);
        let server_key = key(0xA4);
        let configured_key = key(0xB4);
        let unconfigured_key = key(0xC4);
        let server_cap = admin_cap(&root, &server_key);
        let configured_cap = admin_cap(&root, &configured_key);
        let unconfigured_cap = admin_cap(&root, &unconfigured_key);

        let all_caps = [
            server_cap.clone(),
            configured_cap.clone(),
            unconfigured_cap.clone(),
        ];
        let mut server_store = store_with_caps(&all_caps);
        let configured_store = store_with_caps(&all_caps);
        let unconfigured_store = store_with_caps(&all_caps);

        let source = named_root("c20");
        let target = named_root("c21");
        let other = named_root("c22");
        let merge_request = WantRequest::merge(collection_of(&source), data(1), data(2));
        let merge_first = CollectionMerge::new(collection_of(&source), data(1), data(2), data(3));
        let merge_conflict = CollectionMerge::new(collection_of(&source), data(1), data(2), data(4));
        let merge_unrelated = CollectionMerge::new(collection_of(&source), data(1), data(9), data(5));
        let merge_wrong_collection =
            CollectionMerge::new(collection_of(&other), data(1), data(2), data(6));

        // A derive names only its target: the target's descriptor already
        // says what it derives from.
        let derive_request = WantRequest::derive(collection_of(&target), data(7));
        let derive_first = CollectionDerive::new(collection_of(&target), data(7), data(8));
        let derive_conflict = CollectionDerive::new(collection_of(&target), data(7), data(9));
        let derive_unrelated = CollectionDerive::new(collection_of(&target), data(10), data(11));

        for record in [
            CollectionRecord::Derive(derive_unrelated),
            CollectionRecord::Merge(merge_conflict),
            CollectionRecord::Merge(merge_unrelated),
            CollectionRecord::Derive(derive_first),
            CollectionRecord::Merge(merge_first),
            CollectionRecord::Derive(derive_conflict),
            CollectionRecord::Merge(merge_wrong_collection),
        ] {
            server_store.insert(record).unwrap();
        }
        assert!(
            server_store.gossips().unwrap().next().is_none(),
            "unsigned operation receipts need no collection gossip grant"
        );

        // A black-hole DHT makes the discovery choice observable: configured
        // operation probes still dial their named peers directly, while an
        // unconfigured client has no discovery fallback to wait on.
        let net = SimNet::new(
            0xC011ECB,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let _server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            self_cap_of(&server_cap.1),
            false,
        );
        let configured = bring_up_with_peers(
            &net,
            &configured_key,
            configured_store,
            root.verifying_key(),
            self_cap_of(&configured_cap.1),
            false,
            vec![pk(&server_key)],
        );
        let unconfigured = bring_up(
            &net,
            &unconfigured_key,
            unconfigured_store,
            root.verifying_key(),
            self_cap_of(&unconfigured_cap.1),
            false,
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let merge_receipts = probe_operation_while_stepping(&configured, merge_request)
            .await
            .receipts;
        let mut expected_merges = vec![
            CollectionRecord::Merge(merge_first),
            CollectionRecord::Merge(merge_conflict),
        ];
        expected_merges.sort_by_key(CollectionRecord::id);
        assert_eq!(merge_receipts, expected_merges);

        let derive_receipts = probe_operation_while_stepping(&configured, derive_request)
            .await
            .receipts;
        let mut expected_derives = vec![
            CollectionRecord::Derive(derive_first),
            CollectionRecord::Derive(derive_conflict),
        ];
        expected_derives.sort_by_key(CollectionRecord::id);
        assert_eq!(derive_receipts, expected_derives);

        assert!(
            probe_operation_while_stepping(&unconfigured, merge_request)
                .await
                .receipts
                .is_empty(),
            "operation discovery is configured-peer probing, not a DHT lookup"
        );
    });
}

#[test]
fn configured_peer_probe_unions_conflicting_receipts_split_across_peers() {
    let _guard = common::sim_guard();
    run_paused(0xC011ECD, async {
        let root = key(0xF6);
        let first_server_key = key(0xA6);
        let second_server_key = key(0xA7);
        let client_key = key(0xB6);
        let first_server_cap = admin_cap(&root, &first_server_key);
        let second_server_cap = admin_cap(&root, &second_server_key);
        let client_cap = admin_cap(&root, &client_key);
        let all_caps = [
            first_server_cap.clone(),
            second_server_cap.clone(),
            client_cap.clone(),
        ];
        let mut first_store = store_with_caps(&all_caps);
        let mut second_store = store_with_caps(&all_caps);
        let client_store = store_with_caps(&all_caps);

        let descriptor = named_root("c24");
        let request = WantRequest::merge(collection_of(&descriptor), data(1), data(2));
        let first = CollectionRecord::Merge(CollectionMerge::new(
            collection_of(&descriptor),
            data(1),
            data(2),
            data(3),
        ));
        let conflicting = CollectionRecord::Merge(CollectionMerge::new(
            collection_of(&descriptor),
            data(1),
            data(2),
            data(4),
        ));
        first_store.insert(first).unwrap();
        second_store.insert(conflicting).unwrap();

        let net = SimNet::new(
            0xC011ECD,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let _first_server = bring_up(
            &net,
            &first_server_key,
            first_store,
            root.verifying_key(),
            self_cap_of(&first_server_cap.1),
            false,
        );
        let _second_server = bring_up(
            &net,
            &second_server_key,
            second_store,
            root.verifying_key(),
            self_cap_of(&second_server_cap.1),
            false,
        );
        let client = bring_up_with_peers(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_cap.1),
            false,
            vec![pk(&second_server_key), pk(&first_server_key)],
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let mut expected = vec![first, conflicting];
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(
            probe_operation_while_stepping(&client, request)
                .await
                .receipts,
            expected,
            "configured-peer probing must union conflicting exact evidence from every peer"
        );
    });
}

#[test]
fn configured_peer_probe_keeps_partial_evidence_and_recovers_conflict_after_stall() {
    let _guard = common::sim_guard();
    run_paused(0xC011ECE, async {
        let root = key(0xF7);
        let healthy_key = key(0xA8);
        let stalled_key = key(0xA9);
        let client_key = key(0xB7);
        let healthy_cap = admin_cap(&root, &healthy_key);
        let stalled_cap = admin_cap(&root, &stalled_key);
        let client_cap = admin_cap(&root, &client_key);
        let all_caps = [healthy_cap.clone(), stalled_cap.clone(), client_cap.clone()];
        let mut healthy_store = store_with_caps(&all_caps);
        let mut stalled_store = store_with_caps(&all_caps);
        let client_store = store_with_caps(&all_caps);

        let descriptor = named_root("c25");
        let request = WantRequest::merge(collection_of(&descriptor), data(1), data(2));
        let receipt = CollectionRecord::Merge(CollectionMerge::new(
            collection_of(&descriptor),
            data(1),
            data(2),
            data(3),
        ));
        let conflict = CollectionRecord::Merge(CollectionMerge::new(
            collection_of(&descriptor),
            data(1),
            data(2),
            data(4),
        ));
        healthy_store.insert(receipt).unwrap();
        stalled_store.insert(conflict).unwrap();

        let net = SimNet::new(
            0xC011ECE,
            SimConfig {
                dht: DhtMode::Blackhole,
                ..SimConfig::default()
            },
        );
        let _healthy = bring_up(
            &net,
            &healthy_key,
            healthy_store,
            root.verifying_key(),
            self_cap_of(&healthy_cap.1),
            false,
        );
        let _stalled = bring_up(
            &net,
            &stalled_key,
            stalled_store,
            root.verifying_key(),
            self_cap_of(&stalled_cap.1),
            false,
        );
        let client = bring_up_with_peers(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_cap.1),
            false,
            vec![pk(&stalled_key), pk(&healthy_key)],
        );
        net.stall_dials(pk(&stalled_key));
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        let future = client
            .fetch_collection_operation_receipts_with_deadline(request, Duration::from_secs(1));
        tokio::pin!(future);
        let mut answer = None;
        for _ in 0..100 {
            if let std::task::Poll::Ready(receipts) = futures::poll!(future.as_mut()) {
                answer = Some(receipts);
                break;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
        }
        let answer = answer.expect("outer receipt budget completes despite stalled dial");
        assert_eq!(
            answer.receipts,
            vec![receipt],
            "the deadline must preserve answers completed by healthy configured peers"
        );
        assert!(
            !answer.complete,
            "the caller must retain that one configured peer did not answer"
        );

        net.unstall_dials(pk(&stalled_key));
        let recovered = probe_operation_while_stepping(&client, request).await;
        let mut expected = vec![receipt, conflict];
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(recovered.receipts, expected);
        assert!(
            recovered.complete,
            "after recovery every configured peer completed and the conflict is visible"
        );
    });
}

#[test]
fn branch_restricted_capability_receives_no_operation_receipts() {
    let _guard = common::sim_guard();
    run_paused(0xC011ECC, async {
        let root = key(0xF5);
        let server_key = key(0xA5);
        let client_key = key(0xB5);
        let control_key = key(0xC5);
        let server_cap = admin_cap(&root, &server_key);
        let client_cap = branch_restricted_read_cap(&root, &client_key);
        let control_cap = admin_cap(&root, &control_key);
        let all_caps = [server_cap.clone(), client_cap.clone(), control_cap.clone()];
        let mut server_store = store_with_caps(&all_caps);
        let client_store = store_with_caps(&all_caps);
        let control_store = store_with_caps(&all_caps);

        let descriptor = named_root("c23");
        let request = WantRequest::merge(collection_of(&descriptor), data(1), data(2));
        let receipt = CollectionRecord::Merge(CollectionMerge::new(
            collection_of(&descriptor),
            data(1),
            data(2),
            data(3),
        ));
        server_store.insert(receipt).unwrap();

        let net = SimNet::new(0xC011ECC, SimConfig::default());
        let _server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            self_cap_of(&server_cap.1),
            false,
        );
        let client = bring_up_with_peers(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            self_cap_of(&client_cap.1),
            false,
            vec![pk(&server_key)],
        );
        let control = bring_up_with_peers(
            &net,
            &control_key,
            control_store,
            root.verifying_key(),
            self_cap_of(&control_cap.1),
            false,
            vec![pk(&server_key)],
        );
        SimNet::step(&vclock(), Duration::from_millis(1)).await;

        assert_eq!(
            probe_operation_while_stepping(&control, request)
                .await
                .receipts,
            vec![receipt],
            "the same route serves an unrestricted reader"
        );
        assert!(
            probe_operation_while_stepping(&client, request)
                .await
                .receipts
                .is_empty(),
            "the server's rejection is intentionally exposed as no receipts by the Peer probe"
        );
    });
}

async fn probe_operation_while_stepping(
    peer: &Peer<MemoryRepo>,
    request: WantRequest,
) -> triblespace_net::host::CollectionOperationProbe {
    let future =
        peer.fetch_collection_operation_receipts_with_deadline(request, Duration::from_secs(1));
    tokio::pin!(future);
    for _ in 0..100 {
        if let std::task::Poll::Ready(receipts) = futures::poll!(future.as_mut()) {
            return receipts;
        }
        SimNet::step(&vclock(), Duration::from_millis(20)).await;
    }
    panic!("collection operation probe exceeded deterministic step budget")
}

async fn fetch_evidence_result_while_stepping(
    client: triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    peer: [u8; 32],
    collection: triblespace_core::collection::CollectionHandle,
) -> (
    triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    anyhow::Result<Vec<triblespace_net::collection_wire::CollectionCommitEvidence>>,
) {
    let worker = std::thread::spawn(move || {
        let result = client.fetch_collection_evidence_from(peer, collection);
        (client, result)
    });
    while !worker.is_finished() {
        SimNet::step(&vclock(), Duration::from_millis(1)).await;
    }
    worker.join().expect("collection fetch worker panicked")
}

async fn fetch_evidence_while_stepping(
    client: triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    peer: [u8; 32],
    collection: triblespace_core::collection::CollectionHandle,
) -> (
    triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    Vec<triblespace_net::collection_wire::CollectionCommitEvidence>,
) {
    let (client, result) = fetch_evidence_result_while_stepping(client, peer, collection).await;
    (client, result.unwrap())
}
