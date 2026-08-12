//! End-to-end collection-native evidence + closure fetch over the complete
//! authenticated host/protocol/simulator stack.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;

use ed25519_dalek::SigningKey;
use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{IntoBlob, TryFromBlob};
use triblespace_core::collection::{
    Collection, CollectionGossip, CollectionGossipStore, CollectionStore, simplearchive_union,
};
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::time::NsTAIInterval;
use triblespace_core::inline::{Inline, TryToInline};
use triblespace_core::repo::capability::{self, PERM_READ, scope_branch};
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut};
use triblespace_core::trible::{Fragment, TRIBLE_LEN, Trible, TribleSet};
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::{admin_cap, bring_up, key, pk, run_paused, self_cap_of, store_with_caps, vclock};

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
fn direct_collection_fetch_returns_verified_evidence_and_blob_closure_without_admission() {
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

        let descriptor = simplearchive_union::descriptor(id(1));
        let data = archive(2);
        let metadata = archive(3);
        server_store
            .put::<SimpleArchive, _>(triblespace_core::collection::CollectionDescriptor::to_blob(
                &descriptor,
            ))
            .unwrap();
        server_store.put::<SimpleArchive, _>(data.clone()).unwrap();
        server_store
            .put::<SimpleArchive, _>(metadata.clone())
            .unwrap();
        let mut fragment = Fragment::from(TribleSet::try_from_blob(data.clone()).unwrap());
        *fragment.metafacts_mut() = TribleSet::try_from_blob(metadata.clone()).unwrap();
        let mut collection = Collection::new(&mut server_store, id(1), server_key.clone());
        let commit = collection.commit(fragment).unwrap();
        server_store
            .gossip(CollectionGossip::sign(&server_key, descriptor.handle()))
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
            fetch_while_stepping(client, pk(&server_key), descriptor.handle()).await;

        assert_eq!(fetched.collection(), descriptor.handle());
        assert_eq!(fetched.evidence().len(), 1);
        assert_eq!(fetched.evidence()[0].commit(), commit);
        assert!(fetched.blobs().contains_key(&descriptor.handle().raw));
        assert!(fetched.blobs().contains_key(&data.get_handle().raw));
        assert!(fetched.blobs().contains_key(&metadata.get_handle().raw));

        // Fetch is a capability, not admission: client storage stays free of
        // both the native commit and its data blob.
        assert!(client.store().records().unwrap().next().is_none());
        let reader = client.store().reader().unwrap();
        assert!(
            reader
                .get::<TribleSet, SimpleArchive>(data.get_handle())
                .is_err()
        );

        // Existing legacy blob reads remain unchanged on the serving peer.
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
fn direct_collection_fetch_omits_commits_without_author_grants() {
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

        let descriptor = simplearchive_union::descriptor(id(4));
        let data: Blob<SimpleArchive> = TribleSet::new().to_blob();
        server_store
            .put::<SimpleArchive, _>(triblespace_core::collection::CollectionDescriptor::to_blob(
                &descriptor,
            ))
            .unwrap();
        server_store.put::<SimpleArchive, _>(data.clone()).unwrap();
        Collection::new(&mut server_store, id(4), server_key.clone())
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
            fetch_while_stepping(client, pk(&server_key), descriptor.handle()).await;
        assert!(fetched.evidence().is_empty());
        assert!(fetched.roots().is_empty());
        assert!(fetched.blobs().is_empty());
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

        let collection = simplearchive_union::descriptor(id(5)).handle();
        let (_client, result) =
            fetch_result_while_stepping(client, pk(&server_key), collection).await;
        let error = result.unwrap_err();
        assert!(error.to_string().contains("unrestricted read"));
    });
}

async fn fetch_result_while_stepping(
    client: triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    peer: [u8; 32],
    collection: triblespace_core::collection::CollectionId,
) -> (
    triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    anyhow::Result<triblespace_net::collection_wire::CollectionFetch>,
) {
    let worker = std::thread::spawn(move || {
        let result = client.fetch_collection_from(peer, collection);
        (client, result)
    });
    while !worker.is_finished() {
        SimNet::step(&vclock(), Duration::from_millis(1)).await;
    }
    worker.join().expect("collection fetch worker panicked")
}

async fn fetch_while_stepping(
    client: triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    peer: [u8; 32],
    collection: triblespace_core::collection::CollectionId,
) -> (
    triblespace_net::peer::Peer<triblespace_core::repo::memoryrepo::MemoryRepo>,
    triblespace_net::collection_wire::CollectionFetch,
) {
    let (client, result) = fetch_result_while_stepping(client, peer, collection).await;
    (client, result.unwrap())
}
