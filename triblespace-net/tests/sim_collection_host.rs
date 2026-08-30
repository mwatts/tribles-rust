//! End-to-end collection-host cutover coverage over deterministic transport.
#![cfg(feature = "sim")]

use std::sync::{Arc, Mutex, OnceLock};

use anybytes::Bytes;
use ed25519_dalek::SigningKey;
use iroh_base::EndpointId;
use triblespace_core::blob::MemoryBlobStore;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    CapabilityResource,
};
use triblespace_core::clock::{self, VirtualClock};
use triblespace_core::collection::{
    ACTION_READ, ACTION_WRITE, AdmissionPolicy, Collection, CollectionCommit, CollectionData,
    CollectionHandle, CollectionPolicy, CollectionRead, CollectionRecord, CollectionStore,
    CollectionStoreExt, empty_metadata_handle,
};
use triblespace_core::id::{ExclusiveId, Id};
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{
    BlobStorePut, CapabilityProofRead, CapabilityProofStore, SnapshotSource,
};
use triblespace_core::trible::{Fragment, Trible, TribleSet};
use triblespace_net::host::{self, PeerConfig};
use triblespace_net::inventory::{ReconcileDirection, ReconcileQos};
use triblespace_net::peer::Peer;
use triblespace_net::transport::sim::{SimConfig, SimNet};

fn key(byte: u8) -> SigningKey {
    SigningKey::from_bytes(&[byte; 32])
}

fn virtual_clock() -> Arc<VirtualClock> {
    static CLOCK: OnceLock<Arc<VirtualClock>> = OnceLock::new();
    CLOCK
        .get_or_init(|| {
            let clock =
                VirtualClock::new(hifitime::Epoch::from_gregorian_utc_at_midnight(2026, 1, 1));
            clock::install_virtual(clock.clone()).expect("first virtual-clock install");
            clock
        })
        .clone()
}

fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn bundle(
    root: &SigningKey,
    leaf: &SigningKey,
    action: triblespace_core::capability::CapabilityAction,
    collection: CollectionHandle,
) -> CapabilityProofBundle {
    CapabilityProofBundle::issue_root(
        root,
        CapabilityClaim::root(
            CapabilityAtom::new(action, CapabilityResource::from(collection)),
            CapabilityMode::Invoke,
            None,
        ),
        leaf.verifying_key(),
    )
    .unwrap()
}

fn store_bundle(store: &mut MemoryRepo, bundle: CapabilityProofBundle) {
    let (proof, claims) = bundle.into_parts();
    for claim in claims {
        store.put::<SimpleArchive, _>(claim).unwrap();
    }
    store.insert_proof(proof).unwrap();
}

fn register(store: &mut MemoryRepo, policy: CollectionPolicy) -> Collection<SimpleArchive> {
    store.collection("collection-host-e2e", policy).unwrap()
}

fn bring_up(
    net: &SimNet,
    endpoint: &SigningKey,
    store: MemoryRepo,
    peers: Vec<[u8; 32]>,
    direction: ReconcileDirection,
) -> Peer<MemoryRepo> {
    let id = endpoint.verifying_key().to_bytes();
    let harness = net.join(id);
    let (sender, receiver, wiring) =
        host::wire(EndpointId::from_bytes(&id).expect("valid endpoint id"));
    let qos = ReconcileQos { direction };
    tokio::task::spawn_local(host::run_host(
        harness,
        PeerConfig {
            peers: peers
                .into_iter()
                .map(|peer| {
                    iroh_base::EndpointAddr::from(
                        EndpointId::from_bytes(&peer).expect("valid configured peer"),
                    )
                })
                .collect(),
            qos,
        },
        wiring,
    ));
    Peer::with_wiring(store, qos, sender, receiver)
}

async fn advance(clock: &Arc<VirtualClock>, peers: &mut [&mut Peer<MemoryRepo>], seconds: u64) {
    for _ in 0..seconds * 10 {
        SimNet::step(clock, std::time::Duration::from_millis(100)).await;
        for peer in peers.iter_mut() {
            peer.refresh();
        }
    }
}

#[test]
fn write_proof_later_activates_an_already_repaired_commit() {
    let _guard = test_guard();
    let clock = virtual_clock();
    clock.reset();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(async {
        let net = SimNet::new(0xC011_EC71, SimConfig::default());
        let server_key = key(1);
        let reader_key = key(2);
        let read_root = key(3);
        let write_root = key(4);
        let writer = key(5);
        let policy = CollectionPolicy::new(
            AdmissionPolicy::direct(read_root.verifying_key()),
            AdmissionPolicy::direct(write_root.verifying_key()),
        );

        let mut server_store = MemoryRepo::default();
        let collection = register(&mut server_store, policy.clone());
        server_store
            .insert(CollectionRecord::Commit(CollectionCommit::sign(
                &writer,
                collection.handle(),
                CollectionData::new([9; 32]),
                empty_metadata_handle(),
            )))
            .unwrap();
        let write = bundle(
            &write_root,
            &writer,
            CapabilityAction::new(ACTION_WRITE),
            collection.handle(),
        );

        let mut reader_store = MemoryRepo::default();
        let reader_collection = register(&mut reader_store, policy);
        assert_eq!(reader_collection.handle(), collection.handle());
        store_bundle(
            &mut reader_store,
            bundle(
                &read_root,
                &reader_key,
                CapabilityAction::new(ACTION_READ),
                collection.handle(),
            ),
        );

        let server_id = server_key.verifying_key().to_bytes();
        let mut server = bring_up(
            &net,
            &server_key,
            server_store,
            Vec::new(),
            ReconcileDirection::WriteOnly,
        );
        let mut reader = bring_up(
            &net,
            &reader_key,
            reader_store,
            vec![server_id],
            ReconcileDirection::ReadOnly,
        );
        server.activate_collection(collection.handle());
        reader.activate_collection(collection.handle());

        advance(&clock, &mut [&mut server, &mut reader], 3).await;
        let before = reader.snapshot().unwrap();
        assert_eq!(before.records().unwrap().count(), 1);
        assert!(reader_collection.admitted(&before).unwrap().is_empty());

        store_bundle(&mut server.store(), write);
        server.refresh();
        // Sim transport has no gossip plane; periodic anti-entropy is the
        // repair mechanism and must observe a proof-only root change.
        advance(&clock, &mut [&mut server, &mut reader], 32).await;
        let after = reader.snapshot().unwrap();
        assert_eq!(after.records().unwrap().count(), 1);
        assert_eq!(after.proofs().unwrap().count(), 2);
        assert_eq!(reader_collection.admitted(&after).unwrap().len(), 1);
    }));
}

#[test]
fn request_supplied_read_proof_admits_reader_and_rejects_writer_only_peer() {
    let _guard = test_guard();
    let clock = virtual_clock();
    clock.reset();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(async {
        let net = SimNet::new(0xC011_EC72, SimConfig::default());
        let server_key = key(11);
        let reader_key = key(12);
        let writer_key = key(13);
        let read_root = key(14);
        let write_root = key(15);
        let policy = CollectionPolicy::new(
            AdmissionPolicy::direct(read_root.verifying_key()),
            AdmissionPolicy::direct(write_root.verifying_key()),
        );

        let mut server_store = MemoryRepo::default();
        let collection = register(&mut server_store, policy.clone());
        let write = bundle(
            &write_root,
            &writer_key,
            CapabilityAction::new(ACTION_WRITE),
            collection.handle(),
        );
        store_bundle(&mut server_store, write.clone());
        server_store
            .insert(CollectionRecord::Commit(CollectionCommit::sign(
                &writer_key,
                collection.handle(),
                CollectionData::new([17; 32]),
                empty_metadata_handle(),
            )))
            .unwrap();

        let mut reader_store = MemoryRepo::default();
        let reader_collection = register(&mut reader_store, policy.clone());
        store_bundle(
            &mut reader_store,
            bundle(
                &read_root,
                &reader_key,
                CapabilityAction::new(ACTION_READ),
                collection.handle(),
            ),
        );
        let mut writer_store = MemoryRepo::default();
        register(&mut writer_store, policy);
        store_bundle(&mut writer_store, write);

        let server_id = server_key.verifying_key().to_bytes();
        let mut server = bring_up(
            &net,
            &server_key,
            server_store,
            Vec::new(),
            ReconcileDirection::WriteOnly,
        );
        let mut reader = bring_up(
            &net,
            &reader_key,
            reader_store,
            vec![server_id],
            ReconcileDirection::ReadOnly,
        );
        let mut writer_only = bring_up(
            &net,
            &writer_key,
            writer_store,
            vec![server_id],
            ReconcileDirection::ReadOnly,
        );
        for peer in [&mut server, &mut reader, &mut writer_only] {
            peer.activate_collection(collection.handle());
        }

        advance(&clock, &mut [&mut server, &mut reader, &mut writer_only], 4).await;
        let reader_snapshot = reader.snapshot().unwrap();
        assert_eq!(
            reader_collection.admitted(&reader_snapshot).unwrap().len(),
            1
        );
        let writer_snapshot = writer_only.snapshot().unwrap();
        assert_eq!(
            writer_snapshot.records().unwrap().count(),
            0,
            "WRITE(C) without READ(C) must not learn even the collection manifest"
        );
    }));
}

#[test]
fn public_provider_lookup_and_exact_get_remain_bearer_mechanisms() {
    let _guard = test_guard();
    let clock = virtual_clock();
    clock.reset();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(async {
        let net = SimNet::new(0xC011_EC73, SimConfig::default());
        let server_key = key(21);
        let client_key = key(22);
        let mut server_store = MemoryRepo::default();
        let payload = Bytes::from_source(b"public bearer payload".to_vec());
        let payload_handle = server_store.put::<UnknownBlob, _>(payload.clone()).unwrap();
        let collection = server_store
            .collection(
                "public-bearer-provider",
                CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open),
            )
            .unwrap();
        let entity = Id::new([1; 16]).unwrap();
        let attribute = Id::new([2; 16]).unwrap();
        let value = Inline::<Handle<UnknownBlob>>::new(payload_handle.raw);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::new(
            ExclusiveId::force_ref(&entity),
            &attribute,
            &value,
        ));
        server_store
            .commit(
                collection,
                &server_key,
                Fragment::from_parts(facts, TribleSet::new(), MemoryBlobStore::new()),
            )
            .unwrap();

        let server_id = server_key.verifying_key().to_bytes();
        let mut server = bring_up(
            &net,
            &server_key,
            server_store,
            Vec::new(),
            ReconcileDirection::WriteOnly,
        );
        let client = bring_up(
            &net,
            &client_key,
            MemoryRepo::default(),
            vec![server_id],
            ReconcileDirection::ReadOnly,
        );
        advance(&clock, &mut [&mut server], 3).await;

        let mut fetch = Box::pin(client.fetch_blob(payload_handle.raw));
        let got = loop {
            tokio::select! {
                result = &mut fetch => break result,
                () = SimNet::step(&clock, std::time::Duration::from_millis(100)) => {
                    server.refresh();
                }
            }
        };
        assert_eq!(got, Some(payload.to_vec()));
    }));
}
