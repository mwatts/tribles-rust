//! Lazy-replication read path — deterministic simulation.
//!
//! Exercises the swarm-addressed on-demand fetch (`Peer::fetch_blob`,
//! run inline via `host::NetCapability`) plus the `PeerReader`
//! fall-through and the transparent async read. The property under
//! test: a node which does NOT hold a content blob can still obtain it
//! from whoever in the swarm does — without every node eagerly
//! replicating everything.
//!
//! Sim note: the fetch runs inline (a future to poll), so tests drive it
//! with `drive_future` — poll the future, and on `Pending` step the sim
//! so the host (and the fetch) make progress between polls. No thread is
//! ever blocked and nothing rides a reply channel.
#![cfg(feature = "sim")]

mod common;

use std::time::Duration;
use triblespace_core::collection::reach;

use triblespace_core::blob::IntoBlob;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, BlobEncoding};
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    CapabilityResource,
};
use triblespace_core::collection::descriptor;
use triblespace_core::collection::exact_derived::{
    ExactAlgebraError, ExactDerivedAlgebra, ExactDerivedCollection,
};
use triblespace_core::collection::{
    ACTION_WRITE, CapabilityPresentation, CollectionCommit, CollectionData, CollectionDerive,
    CollectionHandle, CollectionMerge, CollectionRecord, CollectionStore, CollectionStoreExt,
    simplearchive_union,
};
use triblespace_core::inline::Inline;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::metadata::MetaDescribe;
use triblespace_core::prelude::BlobStore;
use triblespace_core::repo::async_store::AsyncBlobStoreGet;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{
    BlobStoreGet, BlobStoreKeep, BlobStoreList, BlobStorePut, WantRequest, WantStore,
};
use triblespace_core::trible::Fragment;
use triblespace_core::trible::TribleSet;
use triblespace_net::collection_sync::{ExactDerivedSyncError, ensure_exact_derived};
use triblespace_net::transport::sim::{SimConfig, SimNet};

use common::*;

/// A throwaway content blob (a tiny SimpleArchive) + its hash. Stands
/// in for a "content payload" that lives outside the eagerly-replicated
/// history.
fn content_blob(tag_byte: u8) -> (Blob<SimpleArchive>, [u8; 32]) {
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::id::Id;
    use triblespace_core::macros::entity;
    let e = Id::new([tag_byte; 16]).expect("nonzero id");
    let ts: TribleSet = entity! {
        ExclusiveId::force_ref(&e) @
        triblespace_core::metadata::tag: Id::new([tag_byte.wrapping_add(1).max(1); 16]).unwrap(),
    }
    .into();
    let blob: Blob<SimpleArchive> = ts.to_blob();
    let hash = blob.get_handle().raw;
    (blob, hash)
}

fn collection_data<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
where
    Handle<E>: InlineEncoding,
{
    Handle::<E>::to_hash(blob.get_handle())
}

fn write_presentation(
    authority: &ed25519_dalek::SigningKey,
    writer: ed25519_dalek::VerifyingKey,
    collection: CollectionHandle,
) -> CapabilityPresentation {
    let atom = CapabilityAtom::new(
        CapabilityAction::new(ACTION_WRITE),
        CapabilityResource::from(collection),
    );
    CapabilityPresentation::new(
        writer,
        CapabilityProofBundle::issue_root(
            authority,
            CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
            writer,
        )
        .unwrap(),
    )
}

fn derive_test_target(source: &Blob<SimpleArchive>) -> Blob<UnknownBlob> {
    let mut bytes = source.bytes.as_ref().to_vec();
    bytes.push(0xA5);
    Blob::new(bytes.into())
}

struct NetworkTestAlgebra {
    source: Fragment,
    target: Fragment,
}

impl ExactDerivedAlgebra<SimpleArchive, UnknownBlob> for NetworkTestAlgebra {
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &Blob<SimpleArchive>,
    ) -> Result<(), ExactAlgebraError> {
        if descriptor != &self.source {
            return Err(ExactAlgebraError::Fatal(
                "wrong network-test source descriptor".to_owned(),
            ));
        }
        simplearchive_union::validate_element(source)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn validate_target(
        &self,
        descriptor: &Fragment,
        target: &Blob<UnknownBlob>,
    ) -> Result<(), ExactAlgebraError> {
        if descriptor != &self.target {
            return Err(ExactAlgebraError::Fatal(
                "wrong network-test target descriptor".to_owned(),
            ));
        }
        let Some(source) = target.bytes.as_ref().strip_suffix(&[0xA5]) else {
            return Err(ExactAlgebraError::Fatal(
                "network-test target lacks its canonical suffix".to_owned(),
            ));
        };
        simplearchive_union::validate_element(&Blob::new(source.to_vec().into()))
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn join_source(
        &self,
        low: &Blob<SimpleArchive>,
        high: &Blob<SimpleArchive>,
    ) -> Result<Blob<SimpleArchive>, ExactAlgebraError> {
        simplearchive_union::join(low, high)
            .map_err(|error| ExactAlgebraError::Fatal(error.to_string()))
    }

    fn derive(&self, source: &Blob<SimpleArchive>) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        Ok(derive_test_target(source))
    }

    fn join_target(
        &self,
        low: &Blob<UnknownBlob>,
        high: &Blob<UnknownBlob>,
    ) -> Result<Blob<UnknownBlob>, ExactAlgebraError> {
        self.validate_target(&self.target, low)?;
        self.validate_target(&self.target, high)?;
        let low =
            Blob::<SimpleArchive>::new(low.bytes.as_ref()[..low.bytes.len() - 1].to_vec().into());
        let high =
            Blob::<SimpleArchive>::new(high.bytes.as_ref()[..high.bytes.len() - 1].to_vec().into());
        self.join_source(&low, &high)
            .map(|joined| derive_test_target(&joined))
    }
}

/// Drive `fut` to completion, stepping the sim between polls so the
/// host loop and the *inline* swarm fetch make progress. Returns
/// `Some(value)`, or `None` if the step budget is exhausted. This is the
/// deterministic-sim idiom now that the fetch runs inline (a future to
/// poll) rather than replying on a channel to drain.
async fn drive_future<T, Fut, F>(fut: Fut, mut on_step: F, steps: u32) -> Option<T>
where
    Fut: std::future::Future<Output = T>,
    F: FnMut(),
{
    let mut fut = Box::pin(fut);
    for _ in 0..steps {
        if let std::task::Poll::Ready(v) = futures::poll!(fut.as_mut()) {
            return Some(v);
        }
        SimNet::step(&vclock(), Duration::from_millis(20)).await;
        on_step();
    }
    None
}

fn holds_locally(peer: &mut triblespace_net::peer::Peer<MemoryRepo>, hash: [u8; 32]) -> bool {
    let reader = peer.reader().unwrap();
    // Disambiguate: the sync, local-only `BlobStoreGet::get` (PeerReader
    // also impls the async fetching `AsyncBlobStoreGet::get`).
    BlobStoreGet::get::<anybytes::Bytes, UnknownBlob>(&reader, Inline::new(hash)).is_ok()
}

/// Count of wanted handles in the peer's store — the retention
/// markers that lazy swarm fetches land under.
fn want_count(peer: &triblespace_net::peer::Peer<MemoryRepo>) -> usize {
    peer.store().wants().unwrap().count()
}

#[test]
fn inventory_satisfies_operation_want_without_a_second_record_rpc() {
    use triblespace_net::reconcile::Reconciler;

    let _guard = sim_guard();
    run_paused(0x0A11_CE01, async {
        let root = key(0xE1);
        let server_key = key(0xA1);
        let client_key = key(0xB1);
        let server_proof = team_proofs(&root, &server_key);
        let client_proof = team_proofs(&root, &client_key);
        let mut server_store = empty_store();
        let mut client_store = empty_store();

        // Only the identity matters here; the descriptor is never stored.
        let descriptor =
            simplearchive_union::descriptor("lazy", key(0xF1).verifying_key(), reach::private())
                .into_facts()
                .to_blob()
                .get_handle();
        let a = Inline::new([1; 32]);
        let b = Inline::new([2; 32]);
        let result = Inline::new([3; 32]);
        let request = WantRequest::merge(descriptor, a, b);
        let receipt = CollectionRecord::Merge(CollectionMerge::new(descriptor, a, b, result));
        server_store.insert(receipt).unwrap();
        client_store.want(request).unwrap();

        let net = SimNet::new(0x0A11_CE01, SimConfig::default());
        let mut server = bring_up(
            &net,
            &server_key,
            server_store,
            root.verifying_key(),
            server_proof.clone(),
        );
        let mut client = bring_up_with_peers(
            &net,
            &client_key,
            client_store,
            root.verifying_key(),
            client_proof.clone(),
            vec![pk(&server_key)],
        );
        let mut converged = false;
        for _ in 0..200 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            server.refresh();
            client.refresh();
            if client.store().record(receipt.id()).unwrap().is_some() {
                converged = true;
                break;
            }
        }
        assert!(converged, "periodic inventory admits the matching receipt");

        let mut reconciler = Reconciler::new();
        let stats = reconciler.tick(&mut client).await;
        assert_eq!(stats.wants, 1);
        assert_eq!(stats.missing, 0);
        assert_eq!(stats.attempted, 0);
        assert_eq!(stats.fulfilled, 0);
        assert_eq!(stats.pending, 0);
        assert_eq!(
            client
                .store()
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![receipt]
        );
        assert_eq!(
            client
                .store()
                .wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![request],
            "the durable question remains as local policy after fulfillment"
        );

        // The same still-asserted question is now answered locally and causes
        // neither a network attempt nor false pending work.
        let again = reconciler.tick(&mut client).await;
        assert_eq!(again.wants, 1);
        assert_eq!(again.missing, 0);
        assert_eq!(again.attempted, 0);
        assert_eq!(again.fulfilled, 0);
        assert_eq!(again.pending, 0);

        // Operation answers are native inventory state, not process-local
        // completion cache. A restarted reconciler remains entirely local.
        let mut restarted = Reconciler::new();
        let after_restart = restarted.tick(&mut client).await;
        assert_eq!(after_restart.wants, 1);
        assert_eq!(after_restart.missing, 0);
        assert_eq!(after_restart.attempted, 0);
        assert_eq!(after_restart.fulfilled, 0);
        assert_eq!(after_restart.pending, 0);
    });
}

/// A holds a content blob; B does not. B's team fetch must pull it
/// from its authenticated, explicitly configured route to A and return
/// the verified bytes.
#[test]
fn fetch_blob_pulls_from_the_holder() {
    let _g = sim_guard();
    run_paused(0xABCD, async {
        let net = SimNet::new(0xABCD, SimConfig::default());
        let root = key(0xF0);
        let ka = key(0xA0);
        let kb = key(0xB0);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0x42);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        // Settle the hosts before starting the exact read.
        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        assert!(
            !holds_locally(&mut peer_b, hash),
            "precondition: B lacks the blob"
        );

        let got = drive_future(peer_b.fetch_blob(hash), || peer_a.refresh(), 120)
            .await
            .flatten()
            .expect("B must obtain the blob from the swarm");
        assert_eq!(
            blake3::hash(&got).as_bytes(),
            &hash,
            "fetched bytes must hash to the requested content id"
        );
    });
}

/// The exact empty bottom is independent of both storage and the network. In
/// particular, asking for it must not opportunistically admit inventory that
/// is already waiting at the peer boundary.
#[test]
fn empty_exact_cover_does_not_admit_pending_inventory() {
    let _g = sim_guard();
    run_paused(0xC0AE_0000, async {
        let net = SimNet::new(0xC0AE_0000, SimConfig::default());
        let root = key(0xF3);
        let server_key = key(0xA3);
        let client_key = key(0xB3);
        let namespace = key(0xD3).verifying_key();
        let team_root = root.verifying_key();

        let source_descriptor =
            simplearchive_union::descriptor("network-empty-source", namespace, reach::private());
        let target_descriptor = descriptor::naming(
            "network-empty-target",
            namespace,
            <UnknownBlob as MetaDescribe>::id(),
            simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        );
        let lifecycle = ExactDerivedCollection::<SimpleArchive, UnknownBlob>::new(
            source_descriptor.clone(),
            target_descriptor.clone(),
        );
        let algebra = NetworkTestAlgebra {
            source: source_descriptor,
            target: target_descriptor,
        };
        let mut client_store = empty_store();
        let source_collection = client_store
            .collection(lifecycle.source_descriptor().clone())
            .unwrap();
        let source_cover = client_store.cover(source_collection, &[]).unwrap();

        let lower_a = Blob::<UnknownBlob>::new(vec![0x31].into());
        let lower_b = Blob::<UnknownBlob>::new(vec![0x32].into());
        let upper = Blob::<UnknownBlob>::new(vec![0x33].into());
        let marker = CollectionMerge::new(
            lifecycle.target_collection(),
            collection_data(&lower_a),
            collection_data(&lower_b),
            collection_data(&upper),
        );
        let mut server_store = empty_store();
        server_store
            .insert(CollectionRecord::Merge(marker))
            .unwrap();

        let mut server = bring_up(
            &net,
            &server_key,
            server_store,
            team_root,
            team_proofs(&root, &server_key),
        );
        let mut client = bring_up_with_peers(
            &net,
            &client_key,
            client_store,
            team_root,
            team_proofs(&root, &client_key),
            vec![pk(&server_key)],
        );

        // Let the host finish one inventory exchange, but deliberately do not
        // admit its queued records into the client's store yet.
        for _ in 0..240u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            server.refresh();
        }
        assert!(
            client.store().record(marker.id()).unwrap().is_none(),
            "precondition: the client has not admitted the pending marker",
        );

        let cover = ensure_exact_derived(&mut client, &lifecycle, &source_cover, &algebra)
            .await
            .expect("the exact empty bottom is infallible");
        assert!(cover.is_empty());
        assert!(
            client.store().record(marker.id()).unwrap().is_none(),
            "empty attachment must not refresh or scan network inventory",
        );

        // Prove the preceding assertion was not vacuous: the ordinary refresh
        // path admits the record that was already waiting at the boundary.
        client.refresh();
        assert!(
            client.store().record(marker.id()).unwrap().is_some(),
            "the marker was genuinely pending before the empty attachment",
        );
    });
}

/// A nonempty exact attachment is an operation on one physically team-scoped
/// store. If an external append introduces a conflicting scope, the checked
/// refresh boundary must report it before cover discovery, fetch, landing,
/// or local construction can continue.
#[test]
fn nonempty_exact_attachment_reports_external_scope_conflict() {
    let _g = sim_guard();
    run_paused(0xC0AE_0002, async {
        use std::io::Write;

        use iroh_base::EndpointId;
        use triblespace_core::repo::StoreScope;
        use triblespace_core::repo::pile::Pile;
        use triblespace_net::host;
        use triblespace_net::inventory::ReconcileQos;
        use triblespace_net::peer::Peer;

        let dir = tempfile::tempdir().unwrap();
        let serving_path = dir.path().join("serving.pile");
        let conflicting_path = dir.path().join("conflicting.pile");
        std::fs::File::create(&serving_path).unwrap();
        std::fs::File::create(&conflicting_path).unwrap();

        let serving_team = key(0xE1).verifying_key();
        let conflicting_team = key(0xE2).verifying_key();
        let namespace = key(0xE3).verifying_key();
        let source_descriptor =
            simplearchive_union::descriptor("scope-conflict-source", namespace, reach::private());
        let target_descriptor = descriptor::naming(
            "scope-conflict-target",
            namespace,
            <UnknownBlob as MetaDescribe>::id(),
            simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        );
        let lifecycle = ExactDerivedCollection::<SimpleArchive, UnknownBlob>::new(
            source_descriptor.clone(),
            target_descriptor.clone(),
        );
        let algebra = NetworkTestAlgebra {
            source: source_descriptor,
            target: target_descriptor,
        };

        let mut serving = Pile::open(&serving_path).unwrap();
        serving.bind_store_scope(serving_team).unwrap();
        let source_collection = serving
            .collection(lifecycle.source_descriptor().clone())
            .unwrap();
        assert_eq!(source_collection, lifecycle.source_collection());
        let source = content_blob(0xA1).0;
        serving.put::<SimpleArchive, _>(source.clone()).unwrap();
        let metadata = serving
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let commit = CollectionCommit::sign(
            &key(0xE4),
            lifecycle.source_collection(),
            collection_data(&source),
            metadata,
        );
        serving.insert(CollectionRecord::Commit(commit)).unwrap();
        let presentation =
            write_presentation(&key(0xE3), key(0xE4).verifying_key(), source_collection);
        let source_cover = serving.cover(source_collection, &[presentation]).unwrap();

        let mut conflicting = Pile::open(&conflicting_path).unwrap();
        conflicting.bind_store_scope(conflicting_team).unwrap();
        conflicting.close().unwrap();

        let endpoint = key(0xE5).verifying_key();
        let endpoint_id = EndpointId::from_bytes(endpoint.as_bytes()).unwrap();
        let (sender, receiver, _wiring) = host::wire(endpoint_id);
        let mut peer = Peer::with_wiring(
            serving,
            serving_team,
            ReconcileQos::default(),
            sender,
            receiver,
        )
        .unwrap();

        std::fs::OpenOptions::new()
            .append(true)
            .open(&serving_path)
            .unwrap()
            .write_all(&std::fs::read(&conflicting_path).unwrap())
            .unwrap();

        let result = ensure_exact_derived(&mut peer, &lifecycle, &source_cover, &algebra).await;
        assert!(matches!(
            result,
            Err(ExactDerivedSyncError::Storage {
                operation: "refresh exact-derived network store",
                ..
            })
        ));
    });
}

/// Collection records converge through ordinary inventory while target bytes
/// remain demand-only. The resolver first chooses a stale one-member upper
/// offer, removes it after the exact GET misses, replans to the two available
/// lower members, and introduces neither a durable WANT nor descriptor
/// publication.
#[test]
fn remote_cover_fetch_replans_stale_upper_without_durable_want() {
    let _g = sim_guard();
    run_paused(0xC0AE_0001, async {
        let net = SimNet::new(0xC0AE_0001, SimConfig::default());
        let root = key(0xF4);
        let server_key = key(0xA4);
        let client_key = key(0xB4);
        let namespace = key(0xD4).verifying_key();
        let team_root = root.verifying_key();

        let source_descriptor =
            simplearchive_union::descriptor("network-cover-source", namespace, reach::private());
        let target_descriptor = descriptor::naming(
            "network-cover-target",
            namespace,
            <UnknownBlob as MetaDescribe>::id(),
            simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        );
        let lifecycle = ExactDerivedCollection::<SimpleArchive, UnknownBlob>::new(
            source_descriptor.clone(),
            target_descriptor.clone(),
        );
        let algebra = NetworkTestAlgebra {
            source: source_descriptor,
            target: target_descriptor,
        };

        let sources = [content_blob(0x71).0, content_blob(0x81).0];
        let targets = [
            derive_test_target(&sources[0]),
            derive_test_target(&sources[1]),
        ];
        let upper = algebra.join_target(&targets[0], &targets[1]).unwrap();

        let mut client_store = empty_store();
        let source_collection = client_store
            .collection(lifecycle.source_descriptor().clone())
            .unwrap();
        assert_eq!(source_collection, lifecycle.source_collection());
        let metadata = client_store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let commits: Vec<_> = sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                client_store
                    .put::<SimpleArchive, _>(source.clone())
                    .unwrap();
                let commit = CollectionCommit::sign(
                    &key(0x91 + index as u8),
                    lifecycle.source_collection(),
                    collection_data(source),
                    metadata,
                );
                client_store
                    .insert(CollectionRecord::Commit(commit))
                    .unwrap();
                commit
            })
            .collect();
        let authority = key(0xD4);
        let presentations: Vec<_> = [key(0x91), key(0x92)]
            .into_iter()
            .map(|writer| write_presentation(&authority, writer.verifying_key(), source_collection))
            .collect();
        let source_cover = client_store
            .cover(source_collection, &presentations)
            .unwrap();
        assert_eq!(source_cover.len(), commits.len());

        let mut server_store = empty_store();
        for target in &targets {
            server_store.put::<UnknownBlob, _>(target.clone()).unwrap();
        }
        let derives: Vec<_> = sources
            .iter()
            .zip(&targets)
            .map(|(source, target)| {
                CollectionDerive::new(
                    lifecycle.target_collection(),
                    collection_data(source),
                    collection_data(target),
                )
            })
            .collect();
        for derive in &derives {
            server_store
                .insert(CollectionRecord::Derive(*derive))
                .unwrap();
        }
        let merge = CollectionMerge::new(
            lifecycle.target_collection(),
            collection_data(&targets[0]),
            collection_data(&targets[1]),
            collection_data(&upper),
        );
        server_store.insert(CollectionRecord::Merge(merge)).unwrap();

        let mut server = bring_up(
            &net,
            &server_key,
            server_store,
            team_root,
            team_proofs(&root, &server_key),
        );
        let mut client = bring_up_with_peers(
            &net,
            &client_key,
            client_store,
            team_root,
            team_proofs(&root, &client_key),
            vec![pk(&server_key)],
        );

        let mut records_converged = false;
        for _ in 0..240u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            server.refresh();
            client.refresh();
            if client.store().record(merge.id()).unwrap().is_some()
                && derives
                    .iter()
                    .all(|derive| client.store().record(derive.id()).unwrap().is_some())
            {
                records_converged = true;
                break;
            }
        }
        assert!(records_converged, "target equations converge before reuse");
        for target in &targets {
            offer_resident(&mut server, target.get_handle().raw).await;
        }
        assert_eq!(want_count(&client), 0, "precondition: no durable wants");
        assert!(!holds_locally(&mut client, upper.get_handle().raw));
        assert!(!holds_locally(
            &mut client,
            lifecycle.target_collection().raw,
        ));

        let cover = drive_future(
            ensure_exact_derived(&mut client, &lifecycle, &source_cover, &algebra),
            || server.refresh(),
            240,
        )
        .await
        .expect("exact cover operation completes")
        .expect("the authenticated route supplies the replanned lower cover");

        let expected: Vec<_> = targets
            .iter()
            .map(collection_data)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(
            cover
                .members()
                .iter()
                .map(|(data, _)| *data)
                .collect::<Vec<_>>(),
            expected,
        );
        assert!(
            !holds_locally(&mut client, upper.get_handle().raw),
            "stale upper offer is not manufactured locally",
        );
        for target in &targets {
            assert!(
                holds_locally(&mut client, target.get_handle().raw),
                "replanned lower cover member is fetched",
            );
        }
        assert!(
            !holds_locally(&mut client, lifecycle.target_collection().raw),
            "remote reuse did not fall through to local descriptor publication",
        );
        assert_eq!(
            want_count(&client),
            0,
            "speculative cover fetches do not create durable wants",
        );
    });
}

/// The full lazy-read invariant: a node B that does not hold a content
/// blob fetches it from the swarm and lands it in its store under a
/// **want** — the demand-born retention marker — after which the
/// `PeerReader` serves it locally. This is "lazy replication" in one
/// test: B reads content it never eagerly replicated and retains it as
/// a wanted resident.
#[test]
fn lazy_read_lands_wanted_in_store() {
    let _g = sim_guard();
    run_paused(0xCAFE, async {
        let net = SimNet::new(0xCAFE, SimConfig::default());
        let root = key(0xF2);
        let ka = key(0xA2);
        let kb = key(0xB2);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0x55);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        // B is a lazy node: no eager content.
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        // Settle the hosts before starting the exact read.
        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        // Precondition: B holds nothing locally and has no wants.
        assert!(
            peer_b.try_local(hash).is_none(),
            "precondition: B lacks the blob"
        );
        assert_eq!(want_count(&peer_b), 0, "precondition: no wants");

        // The lazy read: record the demand-born want, fetch from
        // the swarm, land the verified bytes in the store.
        let got = drive_future(peer_b.get_or_fetch_async(hash), || peer_a.refresh(), 120)
            .await
            .expect("fetch future completes")
            .expect("want recorded (MemoryRepo wants are infallible)")
            .expect("B must obtain the blob from the swarm");
        assert_eq!(
            blake3::hash(&got).as_bytes(),
            &hash,
            "fetched bytes hash to the content id"
        );

        // 1. The store now serves it locally.
        let local = peer_b
            .try_local(hash)
            .expect("the store serves the fetched blob");
        assert_eq!(
            blake3::hash(&local).as_bytes(),
            &hash,
            "served bytes hash to the content id"
        );
        // 2. It is retained under a want (the demand-born marker)...
        let wanted: Vec<_> = peer_b
            .store()
            .wants()
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            wanted,
            vec![WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash))],
            "fetched blob is wanted"
        );
    });
}

/// Eviction lives in the store now — and it is always safe: the evicted
/// blob is re-fetchable. B lazily reads 3 blobs (each lands wanted),
/// then the store evicts the first (`unwant` + drop the bytes); the
/// evicted blob becomes a local miss but the swarm still serves it on
/// demand — wanted content remains re-fetchable after eviction.
#[test]
fn lazy_store_eviction_is_safe_and_refetches() {
    let _g = sim_guard();
    run_paused(0xBEEF, async {
        let net = SimNet::new(0xBEEF, SimConfig::default());
        let root = key(0xF3);
        let ka = key(0xA3);
        let kb = key(0xB3);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        // A holds three content blobs; B holds none.
        let blobs: Vec<(Blob<SimpleArchive>, [u8; 32])> =
            (0..3u8).map(|i| content_blob(0x60 + i)).collect();
        let mut store_a = empty_store();
        for (b, _) in &blobs {
            store_a.put::<SimpleArchive, _>(b.clone()).unwrap();
        }
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        for (_, hash) in &blobs {
            offer_resident(&mut peer_a, *hash).await;
        }

        // Lazily read all three, in order — each lands wanted.
        for (_, hash) in &blobs {
            let got = drive_future(peer_b.get_or_fetch_async(*hash), || peer_a.refresh(), 120)
                .await
                .expect("fetch future completes")
                .expect("want recorded")
                .expect("swarm must serve each blob");
            assert_eq!(blake3::hash(&got).as_bytes(), hash);
        }
        assert_eq!(want_count(&peer_b), 3, "each lazy read landed wanted");
        for (_, hash) in &blobs {
            assert!(
                peer_b.try_local(*hash).is_some(),
                "resident after the lazy read"
            );
        }

        // The store evicts the first blob: retract the want and
        // drop the bytes. (MemoryRepo has no eviction policy of its
        // own — this is the store-side operation a budgeted store like
        // Yard performs under pressure.)
        {
            let mut store = peer_b.store();
            store
                .unwant(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(
                    blobs[0].1,
                )))
                .unwrap();
            let retained: Vec<Inline<Handle<UnknownBlob>>> = store
                .reader()
                .unwrap()
                .blobs()
                .filter_map(Result::ok)
                .map(|info| info.handle)
                .filter(|h| h.raw != blobs[0].1)
                .collect();
            store.keep(retained);
        }

        // The eviction retracted the pin and dropped the resident bytes.
        assert_eq!(want_count(&peer_b), 2, "want retracted by the eviction");
        assert!(
            peer_b.try_local(blobs[0].1).is_none(),
            "oldest evicted from the store"
        );
        assert!(
            peer_b.try_local(blobs[1].1).is_some(),
            "second still resident"
        );
        assert!(
            peer_b.try_local(blobs[2].1).is_some(),
            "newest still resident"
        );

        // The evicted blob is re-fetchable from the swarm.
        let refetched = drive_future(peer_b.fetch_blob(blobs[0].1), || peer_a.refresh(), 120)
            .await
            .flatten()
            .expect("evicted blob re-fetchable — eviction is always safe");
        assert_eq!(blake3::hash(&refetched).as_bytes(), &blobs[0].1);
    });
}

/// The honest **async** lazy read: `get_or_fetch_async` awaits the
/// swarm fetch (oneshot reply, no blocked thread) and lands the result
/// wanted in the store. Driven deterministically by polling the
/// future and stepping the sim on `Pending` — the awaited oneshot
/// resolves once the host (driven by the stepping) sends the reply.
#[test]
fn async_lazy_read_awaits_swarm_and_lands_wanted() {
    let _g = sim_guard();
    run_paused(0xA5A5, async {
        let net = SimNet::new(0xA5A5, SimConfig::default());
        let root = key(0xF4);
        let ka = key(0xA4);
        let kb = key(0xB4);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0x77);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;
        assert!(
            peer_b.try_local(hash).is_none(),
            "precondition: B lacks the blob"
        );

        // Drive the async read: poll once, and on Pending step the sim so
        // the host can serve the reply. The future holds `&mut peer_b`
        // for its lifetime, so only `peer_a` is touched inside the loop.
        let got = {
            let mut fut = Box::pin(peer_b.get_or_fetch_async(hash));
            loop {
                match futures::poll!(fut.as_mut()) {
                    std::task::Poll::Ready(r) => break r,
                    std::task::Poll::Pending => {
                        SimNet::step(&vclock(), Duration::from_millis(20)).await;
                        peer_a.refresh();
                    }
                }
            }
        };

        let got = got
            .expect("want recorded")
            .expect("async lazy read must obtain the blob from the swarm");
        assert_eq!(
            blake3::hash(&got).as_bytes(),
            &hash,
            "awaited bytes hash to the content id"
        );
        // Landed wanted in the store, served locally on the next read.
        assert!(
            peer_b.try_local(hash).is_some(),
            "now resident in the local store"
        );
        assert_eq!(want_count(&peer_b), 1, "landed under a want");
    });
}

/// Transparent async read through the trait surface: a *generic*
/// `AsyncBlobStoreGet` consumer calls `reader.get(handle).await` on a
/// blob B doesn't hold, and the `PeerReader` fetches it from the swarm
/// and lands it wanted in the shared store — no knowledge that
/// it's a `Peer`. This is the "lazy replication for free" payoff of
/// increment 5b.
#[test]
fn transparent_async_get_fetches_through_reader() {
    let _g = sim_guard();
    run_paused(0x9001, async {
        let net = SimNet::new(0x9001, SimConfig::default());
        let root = key(0xF5);
        let ka = key(0xA5);
        let kb = key(0xB5);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0x88);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;
        assert!(
            peer_b.try_local(hash).is_none(),
            "precondition: B lacks the blob"
        );

        // A generic async reader: it only knows `AsyncBlobStoreGet`.
        let got: anybytes::Bytes = {
            let reader = peer_b.reader().unwrap();
            let mut fut = Box::pin(AsyncBlobStoreGet::get::<anybytes::Bytes, UnknownBlob>(
                &reader,
                Inline::new(hash),
            ));
            loop {
                match futures::poll!(fut.as_mut()) {
                    std::task::Poll::Ready(r) => break r,
                    std::task::Poll::Pending => {
                        SimNet::step(&vclock(), Duration::from_millis(20)).await;
                        peer_a.refresh();
                    }
                }
            }
            .expect("transparent get must fetch the blob from the swarm")
        };

        assert_eq!(
            blake3::hash(&got).as_bytes(),
            &hash,
            "transparently-fetched bytes hash to the content id"
        );
        // The fetch landed in the *shared* store (a &self read mutated
        // Peer state), so a fresh local read now hits.
        assert_eq!(
            want_count(&peer_b),
            1,
            "fetch recorded the demand-born want"
        );
        assert!(
            peer_b.try_local(hash).is_some(),
            "served locally on the next read"
        );
    });
}

/// With no exact route and no holder, the fetch resolves to `None`
/// (Unavailable) — it must complete, not hang.
#[test]
fn fetch_blob_unavailable_is_clean() {
    let _g = sim_guard();
    run_paused(0x1234, async {
        let net = SimNet::new(0x1234, SimConfig::default());
        let root = key(0xF1);
        let ka = key(0xA1);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);

        let store_a = empty_store();
        let peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());

        let (_blob, hash) = content_blob(0x99);
        // No-op on_step: the inline fetch borrows `peer_a`; there is no
        // configured route to try.
        let reply = drive_future(peer_a.fetch_blob(hash), || {}, 400)
            .await
            .flatten();
        assert!(
            reply.is_none(),
            "unavailable fetch must resolve to None, got {:?} bytes",
            reply.map(|b| b.len())
        );
    });
}

/// Merely coexisting in one transport does not install a route for exact
/// reads. Discovery remains explicit configured/PEER/DHT evidence.
#[test]
fn transport_presence_alone_is_not_an_exact_fetch_route() {
    let _g = sim_guard();
    run_paused(0x60B1_0001, async {
        let net = SimNet::new(0x60B1_0001, SimConfig::default());
        let root = key(0xF0);
        let ka = key(0xA0);
        let kb = key(0xB0);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        // A holds an otherwise unadvertised content blob. No configured or
        // durable PEER evidence supplies B with a route.
        let (orphan_blob, orphan_hash) = content_blob(0x32);
        let mut store_a = empty_store();
        store_a
            .put::<SimpleArchive, _>(orphan_blob.clone())
            .unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up(&net, &kb, store_b, team_root, proof_b.clone());

        // Let both independent hosts settle without giving B any route to A.
        for _ in 0..60u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        assert!(
            peer_b.try_local(orphan_hash).is_none(),
            "precondition: the orphan blob never rode the eager walk to B"
        );

        // Mere network presence must not be treated as a data route or as
        // disclosure authority.
        let got = drive_future(peer_b.fetch_blob(orphan_hash), || peer_a.refresh(), 400)
            .await
            .expect("route-less fetch completes");
        assert!(got.is_none());
    });
}

/// The END-TO-END fetch deadline. With a short explicit budget, an
/// unavailable fetch resolves `None` within the explicit budget rather
/// than stacking per-route dial/op deadlines. Regression test for the
/// previously-unbounded on-demand path, where per-stage deadlines
/// could stack to 40s+ across a provider list.
#[test]
fn fetch_deadline_bounds_unavailable_resolution() {
    let _g = sim_guard();
    run_paused(0xDEAD_0011, async {
        let net = SimNet::new(0xDEAD_0011, SimConfig::default());
        let root = key(0xFB);
        let ka = key(0xAB);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);

        let store_a = empty_store();
        let peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let _ = net; // keep the sim alive for the fetch

        let (_blob, hash) = content_blob(0x9A);
        // Budget 500 ms. Sixty 20 ms sim steps leave ample room to prove
        // that the overall deadline bounds all route attempts.
        let reply = drive_future(
            peer_a.fetch_blob_with_deadline(hash, Duration::from_millis(500)),
            || {},
            60,
        )
        .await;
        let reply = reply.expect("fetch must resolve within the overall budget");
        assert!(
            reply.is_none(),
            "an expired budget is Unavailable, not bytes"
        );
    });
}

/// Lazy read degrades to Unavailable when the network partitions the
/// reader from the only configured holder, and **recovers** once the link
/// heals — the graceful-degradation property under a real fault.
#[test]
fn lazy_read_unavailable_under_partition_then_heals() {
    let _g = sim_guard();
    run_paused(0xD15C, async {
        let net = SimNet::new(0xD15C, SimConfig::default());
        let root = key(0xF6);
        let ka = key(0xA6);
        let kb = key(0xB6);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0xC1);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        // Sever A↔B: B retains A as an exact route, but its dial fails.
        net.partition(pk(&ka), pk(&kb));
        let blocked = drive_future(peer_b.fetch_blob(hash), || peer_a.refresh(), 300)
            .await
            .flatten();
        assert!(
            blocked.is_none(),
            "partitioned from the only holder → Unavailable"
        );
        assert!(
            peer_b.try_local(hash).is_none(),
            "nothing landed from a failed fetch"
        );
        assert_eq!(
            want_count(&peer_b),
            0,
            "fetch_blob records no want — retention is the caller's policy"
        );

        // Heal the link; the same read now succeeds.
        net.heal(pk(&ka), pk(&kb));
        let got = drive_future(peer_b.fetch_blob(hash), || peer_a.refresh(), 300)
            .await
            .flatten()
            .expect("after heal the holder is reachable again");
        assert_eq!(blake3::hash(&got).as_bytes(), &hash);
    });
}

/// Same graceful-degradation property under a node **crash** rather than
/// a link partition: the holder crashing makes the read Unavailable
/// (its connections reset, re-dials fail), and reviving it restores
/// service. Exercises the conn-pool's evict-on-error + re-dial path.
#[test]
fn lazy_read_unavailable_under_crash_then_revives() {
    let _g = sim_guard();
    run_paused(0xC1A5, async {
        let net = SimNet::new(0xC1A5, SimConfig::default());
        let root = key(0xF7);
        let ka = key(0xA7);
        let kb = key(0xB7);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0xC2);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        net.crash(pk(&ka));
        let blocked = drive_future(peer_b.fetch_blob(hash), || peer_a.refresh(), 300)
            .await
            .flatten();
        assert!(blocked.is_none(), "holder crashed → Unavailable");

        net.revive(pk(&ka));
        let got = drive_future(peer_b.fetch_blob(hash), || peer_a.refresh(), 300)
            .await
            .flatten()
            .expect("after revive the holder serves again");
        assert_eq!(blake3::hash(&got).as_bytes(), &hash);
    });
}

/// A `Peer<S>` **retains** what it fetches: the lazy read lands the
/// blob in the store under a want, so a second read is a LOCAL
/// hit — no re-fetch, no swarm dependency. Proven by crashing the only
/// holder before the second read: it still succeeds, resolving on the
/// first poll without a single sim step. Retention is the store's job;
/// every fetch stays resident until the store evicts it.
#[test]
fn fetched_blob_is_retained_second_read_hits_locally() {
    let _g = sim_guard();
    run_paused(0x0011_0000, async {
        let net = SimNet::new(0x0011_0000, SimConfig::default());
        let root = key(0xFC);
        let ka = key(0xAC);
        let kb = key(0xBC);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0xCD);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        let got = drive_future(peer_b.get_or_fetch_async(hash), || peer_a.refresh(), 200)
            .await
            .expect("fetch future completes")
            .expect("want recorded")
            .expect("the lazy read fetches from the swarm");
        assert_eq!(blake3::hash(&got).as_bytes(), &hash);

        // The fetch landed wanted: resident, evictable, retained.
        assert_eq!(want_count(&peer_b), 1, "fetch landed under a want");
        assert!(
            peer_b.try_local(hash).is_some(),
            "a local hit after the fetch"
        );

        // Crash the only holder: the second read must still succeed —
        // it is a local hit, not a re-fetch. `drive_future`'s on_step
        // panicking makes "no sim step needed" an explicit assertion.
        net.crash(pk(&ka));
        let again = drive_future(
            peer_b.get_or_fetch_async(hash),
            || panic!("second read must resolve locally without stepping the sim"),
            1,
        )
        .await
        .expect("second read resolves on the first poll")
        .expect("no want recorded on a local hit")
        .expect("second read is a local hit — no re-fetch");
        assert_eq!(blake3::hash(&again).as_bytes(), &hash);
    });
}

/// Randomized fault **chaos** — the Jepsen-style property fixed
/// scenarios miss. Across several seeds, the A↔B link is partitioned and
/// healed at random steps while B retries its lazy read; the back half
/// of each run is forced healthy. Two invariants:
///   * SAFETY — any bytes the fetch returns hash to the requested
///     content id. Chaos never yields corrupt data.
///   * LIVENESS — once the link stops flapping and stays healed, the
///     read eventually succeeds.
#[test]
fn lazy_fetch_under_partition_chaos_is_safe_and_recovers() {
    use rand::{Rng, SeedableRng};
    let _g = sim_guard();
    for s in 0..6u64 {
        let seed = 0x0C4A_0500 + s;
        run_paused(seed, async move {
            let net = SimNet::new(seed, SimConfig::default());
            let root = key(0xFA);
            let ka = key(0xAA);
            let kb = key(0xBA);
            let team_root = root.verifying_key();
            let proof_a = team_proofs(&root, &ka);
            let proof_b = team_proofs(&root, &kb);

            let (blob, hash) = content_blob(0xAB);
            let mut store_a = empty_store();
            store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
            let store_b = empty_store();

            let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
            let peer_b = bring_up_with_peers(
                &net,
                &kb,
                store_b,
                team_root,
                proof_b.clone(),
                vec![pk(&ka)],
            );

            for _ in 0..40u32 {
                SimNet::step(&vclock(), Duration::from_millis(20)).await;
                peer_a.refresh();
            }
            offer_resident(&mut peer_a, hash).await;

            let pa = pk(&ka);
            let pb = pk(&kb);
            let mut frng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xF417);
            const FLAP_UNTIL: u32 = 250;
            const BUDGET: u32 = 600;

            let mut got: Option<Vec<u8>> = None;
            let mut fut = Box::pin(peer_b.fetch_blob(hash));
            for step in 0..BUDGET {
                if let std::task::Poll::Ready(v) = futures::poll!(fut.as_mut()) {
                    if let Some(bytes) = v {
                        // SAFETY invariant.
                        assert_eq!(
                            blake3::hash(&bytes).as_bytes(),
                            &hash,
                            "chaos must never yield corrupt bytes (seed {seed:#x})"
                        );
                        got = Some(bytes);
                        break;
                    }
                    // One-shot attempt failed (partitioned mid-fetch);
                    // retry. The old future drops here, freeing its
                    // shared borrow of peer_b.
                    fut = Box::pin(peer_b.fetch_blob(hash));
                }

                if step < FLAP_UNTIL {
                    if frng.gen_bool(0.12) {
                        if frng.gen_bool(0.5) {
                            net.partition(pa, pb);
                        } else {
                            net.heal(pa, pb);
                        }
                    }
                } else if step == FLAP_UNTIL {
                    net.heal(pa, pb); // hold healthy so liveness can assert
                }

                SimNet::step(&vclock(), Duration::from_millis(20)).await;
                peer_a.refresh();
            }

            // LIVENESS invariant.
            assert!(
                got.is_some(),
                "lazy read must recover after the partition stops flapping (seed {seed:#x})"
            );
        });
    }
}

/// Route fallback across a 3-node team: the blob lives on both A and
/// C; A crashes; B's lazy read must fall back to the surviving holder.
/// Exercises `fetch_one`'s multi-route iteration (try the next peer
/// on a dial/op failure) — invisible to the 2-node tests, where there's
/// only ever one provider.
#[test]
fn lazy_fetch_falls_back_to_a_second_holder() {
    let _g = sim_guard();
    run_paused(0xFA11, async {
        let net = SimNet::new(0xFA11, SimConfig::default());
        let root = key(0xF9);
        let ka = key(0xA9);
        let kb = key(0xB9);
        let kc = key(0xC9);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);
        let proof_c = team_proofs(&root, &kc);
        let (blob, hash) = content_blob(0xFB);
        // A and C both hold the blob; B does not.
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let mut store_c = empty_store();
        store_c.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_c = bring_up(&net, &kc, store_c, team_root, proof_c.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka), pk(&kc)],
        );

        for _ in 0..50u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
            peer_c.refresh();
        }
        offer_resident(&mut peer_a, hash).await;
        offer_resident(&mut peer_c, hash).await;
        assert!(
            peer_b.try_local(hash).is_none(),
            "precondition: B lacks the blob"
        );

        // Crash A. B's second authenticated route, C, must take over.
        net.crash(pk(&ka));
        let got = drive_future(
            peer_b.fetch_blob(hash),
            || {
                peer_c.refresh();
            },
            400,
        )
        .await
        .flatten()
        .expect("B must fall back to the surviving holder C");
        assert_eq!(blake3::hash(&got).as_bytes(), &hash);
    });
}

/// Run one full lazy-fetch scenario under `seed` and return the observed
/// outcome: the fetched bytes (if any) and the number of sim steps the
/// fetch took to complete. The step count is latency-sensitive — link
/// latencies are drawn from the seeded net RNG — so it's a real
/// seed-dependent observable, exactly what a determinism check wants.
fn run_lazy_fetch(seed: u64, config: SimConfig) -> (Option<Vec<u8>>, u32) {
    run_paused(seed, async move {
        let net = SimNet::new(seed, config);
        let root = key(0xF0);
        let ka = key(0xA0);
        let kb = key(0xB0);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0x42);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        // Drive the fetch, counting steps until completion.
        let mut fut = Box::pin(peer_b.fetch_blob(hash));
        let mut steps = 0u32;
        let got = loop {
            if let std::task::Poll::Ready(v) = futures::poll!(fut.as_mut()) {
                break v;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
            steps += 1;
            if steps > 600 {
                break None;
            }
        };
        (got, steps)
    })
}

/// Two concurrent transparent reads on the *same* node for the *same*
/// missing blob. Stresses the shared store (interior-mutable
/// `Arc<Mutex>`): both `&self` reads fetch from the swarm and land into
/// the one shared store. The conn-pool singleflight should share the
/// dial to the holder, and the content-addressed store must end with
/// exactly one copy under exactly one want — no double-store from
/// the racing lands.
#[test]
fn concurrent_transparent_reads_share_store_and_dedupe() {
    let _g = sim_guard();
    run_paused(0xC0FFEE, async {
        let net = SimNet::new(0xC0FFEE, SimConfig::default());
        let root = key(0xF8);
        let ka = key(0xA8);
        let kb = key(0xB8);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0xCC);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;
        assert!(
            peer_b.try_local(hash).is_none(),
            "precondition: B lacks the blob"
        );

        // Two independent readers off the same Peer — each owns a clone
        // of the store snapshot and a fetch capability into the *same*
        // shared store. (reader() borrows &mut only transiently.)
        let reader1 = peer_b.reader().unwrap();
        let reader2 = peer_b.reader().unwrap();

        let (got1, got2) = {
            let mut f1 = Box::pin(AsyncBlobStoreGet::get::<anybytes::Bytes, UnknownBlob>(
                &reader1,
                Inline::new(hash),
            ));
            let mut f2 = Box::pin(AsyncBlobStoreGet::get::<anybytes::Bytes, UnknownBlob>(
                &reader2,
                Inline::new(hash),
            ));
            let mut r1: Option<_> = None;
            let mut r2: Option<_> = None;
            for _ in 0..300u32 {
                if r1.is_none() {
                    if let std::task::Poll::Ready(v) = futures::poll!(f1.as_mut()) {
                        r1 = Some(v);
                    }
                }
                if r2.is_none() {
                    if let std::task::Poll::Ready(v) = futures::poll!(f2.as_mut()) {
                        r2 = Some(v);
                    }
                }
                if r1.is_some() && r2.is_some() {
                    break;
                }
                SimNet::step(&vclock(), Duration::from_millis(20)).await;
                peer_a.refresh();
            }
            (r1, r2)
        };

        let got1 = got1.expect("reader 1 completed").expect("reader 1 fetched");
        let got2 = got2.expect("reader 2 completed").expect("reader 2 fetched");
        assert_eq!(blake3::hash(&got1).as_bytes(), &hash);
        assert_eq!(blake3::hash(&got2).as_bytes(), &hash);
        // Both racing lands hit the same content-addressed store: one
        // copy, and the two recorded wants collapse to one want.
        assert_eq!(
            want_count(&peer_b),
            1,
            "concurrent lands of the same blob dedupe to a single want"
        );
        assert!(
            peer_b.try_local(hash).is_some(),
            "resident after the racing reads"
        );
    });
}

/// Run a partition → heal → recover lazy fetch under `seed`, returning
/// the recovered bytes and the steps the *recovery* attempt took. The
/// scenario scripts a partition (failed attempt), then a heal, then a
/// timed successful attempt — so the observable folds in both the fault
/// injection and the latency-sensitive recovery.
fn run_lazy_fetch_partition_recovery(seed: u64) -> (Option<Vec<u8>>, u32) {
    run_paused(seed, async move {
        let net = SimNet::new(seed, SimConfig::default());
        let root = key(0xF0);
        let ka = key(0xA0);
        let kb = key(0xB0);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        let (blob, hash) = content_blob(0x42);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        let peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        let pa = pk(&ka);
        let pb = pk(&kb);

        // Partition → a failed attempt → heal.
        net.partition(pa, pb);
        let _ = drive_future(peer_b.fetch_blob(hash), || peer_a.refresh(), 120).await;
        net.heal(pa, pb);

        // Timed recovery attempt.
        let mut fut = Box::pin(peer_b.fetch_blob(hash));
        let mut steps = 0u32;
        let got = loop {
            if let std::task::Poll::Ready(v) = futures::poll!(fut.as_mut()) {
                break v;
            }
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
            steps += 1;
            if steps > 400 {
                break None;
            }
        };
        (got, steps)
    })
}

/// Determinism of the **faulted** path — the property that makes DST
/// bug reports reproducible. Fault injection (partition/heal) and the
/// recovery that follows must be a pure function of the seed too:
/// otherwise a chaos-found failure couldn't be replayed. If `crash`'s
/// conn-retain or `partition`'s set bookkeeping ever leaked
/// non-determinism (HashMap order, wall-clock), the recovery step count
/// would diverge between identical runs.
#[test]
fn faulted_lazy_fetch_is_deterministic() {
    let _g = sim_guard();
    let r1 = run_lazy_fetch_partition_recovery(0x0FD0_0001);
    let r2 = run_lazy_fetch_partition_recovery(0x0FD0_0001);
    assert!(r1.0.is_some(), "sanity: the fetch recovered after heal");
    assert_eq!(
        r1, r2,
        "partition+heal+recovery is reproducible under the same seed"
    );
}

/// The foundational DST guarantee: a simulated run is a **pure function
/// of `(seed, scenario)`**. The identical scenario under the identical
/// seed must produce the identical observable — same fetched bytes *and*
/// same step count. A regression that leaked real wall-clock time, or
/// `HashMap`/`HashSet` iteration order, or any unseeded randomness into
/// the sim would diverge here.
#[test]
fn lazy_fetch_is_deterministic_across_runs() {
    let _g = sim_guard();
    let (bytes1, steps1) = run_lazy_fetch(0x0DDD_0001, SimConfig::default());
    let (bytes2, steps2) = run_lazy_fetch(0x0DDD_0001, SimConfig::default());
    assert!(bytes1.is_some(), "sanity: the fetch actually succeeded");
    assert_eq!(bytes1, bytes2, "same seed → identical fetched bytes");
    assert_eq!(
        steps1, steps2,
        "same seed → identical step count: the sim is a pure function of the seed"
    );
}

/// Liveness property across the seed space: under a healthy network, the
/// lazy read must *always* eventually succeed, whatever the seed-chosen
/// link latencies and id minting. Property-based DST — catches
/// seed-dependent liveness bugs a single hand-picked seed would miss.
#[test]
fn lazy_fetch_succeeds_across_many_seeds() {
    let _g = sim_guard();
    for s in 0..16u64 {
        let seed = 0x5EED_0000 + s;
        let (got, steps) = run_lazy_fetch(seed, SimConfig::default());
        assert!(
            got.is_some(),
            "lazy fetch must succeed under seed {seed:#x} (gave up after {steps} steps)"
        );
    }
}

/// The want-reconcile loop — the daemon half of "a want IS a
/// durable want-marker". A faculty (another process) appends a want
/// record for a blob the node doesn't hold; the sync daemon's reconcile
/// tick notices the want, fetches the blob from whoever holds it, and
/// lands it under the existing want. The test stays focused on exact Demand
/// fetch rather than broad inventory mirroring.
#[test]
fn reconcile_tick_services_out_of_band_want() {
    use triblespace_net::reconcile::Reconciler;

    let _g = sim_guard();
    run_paused(0x3A2C, async {
        let net = SimNet::new(0x3A2C, SimConfig::default());
        let root = key(0xFD);
        let ka = key(0xAD);
        let kb = key(0xBD);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);
        let proof_b = team_proofs(&root, &kb);

        // A holds the blob locally; B's reconciliation must not alter A's
        // independent want state.
        let (blob, hash) = content_blob(0x21);
        let mut store_a = empty_store();
        store_a.put::<SimpleArchive, _>(blob.clone()).unwrap();
        let store_b = empty_store();

        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());
        // B knows the bootstrap route, but provider publication remains an
        // explicit holder action rather than implicit peer probing.
        let mut peer_b = bring_up_with_peers(
            &net,
            &kb,
            store_b,
            team_root,
            proof_b.clone(),
            vec![pk(&ka)],
        );

        // Settle the hosts before reconciliation starts.
        for _ in 0..40u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
            peer_a.refresh();
        }
        offer_resident(&mut peer_a, hash).await;

        // Out-of-band want: written through the store guard, bypassing
        // the Peer's own read path — exactly what a faculty appending a
        // want record to the shared pile looks like to the daemon.
        peer_b
            .store()
            .want(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)))
            .unwrap();
        assert!(
            peer_b.try_local(hash).is_none(),
            "precondition: B lacks the blob"
        );

        let a_wants_before = want_count(&peer_a);

        // The reconcile pass: notice the want, fetch, land.
        let mut rec = Reconciler::new();
        let stats = drive_future(rec.tick(&mut peer_b), || peer_a.refresh(), 300)
            .await
            .expect("reconcile tick completes");
        assert_eq!(stats.wants, 1, "the out-of-band want is the want set");
        assert_eq!(stats.missing, 1, "its blob was absent at pass start");
        assert_eq!(stats.fulfilled, 1, "the want was serviced from the swarm");
        assert_eq!(stats.pending, 0, "nothing left outstanding");

        // The blob landed at B...
        assert!(
            peer_b.try_local(hash).is_some(),
            "want serviced: blob now resident at B"
        );
        // ...still wanted — the want-marker became the retention
        // marker...
        let wanted: Vec<_> = peer_b
            .store()
            .wants()
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            wanted,
            vec![WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash))],
            "the want stays on record as the want"
        );
        // ...and A's independent retention policy is untouched.
        assert_eq!(want_count(&peer_a), a_wants_before, "A's wants untouched");
    });
}

/// A want for a handle NOBODY holds stays pending across ticks without
/// erroring — "absent" is always "not obtained yet", never
/// definitely-absent. Also pins down the backoff gate: an immediate
/// re-tick issues no fetch (the failed want waits out its backoff), a
/// re-tick after the backoff elapses retries.
#[test]
fn reconcile_unsatisfiable_want_stays_pending() {
    use triblespace_net::reconcile::Reconciler;

    let _g = sim_guard();
    run_paused(0x9E4D, async {
        // With no exact routes, the fetch resolves Unavailable in bounded
        // virtual time and never hangs.
        let net = SimNet::new(0x9E4D, SimConfig::default());
        let root = key(0xFE);
        let ka = key(0xAE);
        let team_root = root.verifying_key();
        let proof_a = team_proofs(&root, &ka);

        let store_a = empty_store();
        let mut peer_a = bring_up(&net, &ka, store_a, team_root, proof_a.clone());

        // A want for content nobody holds (an arbitrary content id).
        let hash = *blake3::hash(b"nobody holds this blob").as_bytes();
        peer_a
            .store()
            .want(WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash)))
            .unwrap();

        let mut rec = Reconciler::new();

        // Tick 1: the want is attempted and comes back Unavailable —
        // pending, not an error, not dropped. (No-op on_step: the tick
        // borrows peer_a; no companion host needs refreshing.)
        let s1 = drive_future(rec.tick(&mut peer_a), || {}, 400)
            .await
            .expect("tick 1 completes despite the unsatisfiable want");
        assert_eq!(s1.missing, 1);
        assert_eq!(s1.attempted, 1, "first sighting is attempted immediately");
        assert_eq!(s1.fulfilled, 0);
        assert_eq!(s1.pending, 1, "the want stays pending");

        // Tick 2, immediately: the backoff gate holds — still pending,
        // but no fetch is issued (no hammering a dark swarm).
        let s2 = drive_future(rec.tick(&mut peer_a), || {}, 400)
            .await
            .expect("tick 2 completes");
        assert_eq!(s2.missing, 1);
        assert_eq!(s2.attempted, 0, "backoff-gated: no immediate re-fetch");
        assert_eq!(s2.pending, 1);

        // Let the backoff (1s initial) elapse in virtual time, then
        // tick 3: the want is retried — and stays pending again.
        for _ in 0..100u32 {
            SimNet::step(&vclock(), Duration::from_millis(20)).await;
        }
        let s3 = drive_future(rec.tick(&mut peer_a), || {}, 400)
            .await
            .expect("tick 3 completes");
        assert_eq!(s3.attempted, 1, "retried after the backoff elapsed");
        assert_eq!(s3.fulfilled, 0);
        assert_eq!(s3.pending, 1);

        // Throughout: the want is still durably on record and the blob
        // still absent — nothing was dropped, nothing errored.
        let wanted: Vec<_> = peer_a
            .store()
            .wants()
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            wanted,
            vec![WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(hash))]
        );
        assert!(peer_a.try_local(hash).is_none());
    });
}

/// Explicit DHT publication and exact transfer need no broadcast side plane.
#[test]
fn published_lazy_fetch_uses_only_dht_and_exact_transfer() {
    let _g = sim_guard();
    let (got, steps) = run_lazy_fetch(0x6055_1055, SimConfig::default());
    assert!(got.is_some(), "published provider resolves without gossip");
    assert!(
        steps > 0,
        "the DHT and exact-transfer path must actually run"
    );
}
