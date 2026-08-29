use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;

use ed25519_dalek::{SigningKey, VerifyingKey};

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::{
    OrderedUniverse, Rank9AcceleratedSuccinctArchiveBlob, SuccinctArchiveBlob, UnionArchive,
};
use triblespace_core::blob::{Blob, BlobEncoding, IntoBlob};
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProof,
    CapabilityProofBundle, CapabilityProofId, CapabilityResource, CapabilityValidity,
};
use triblespace_core::collection::records::{
    collection_authority, collection_name, collection_representation, KIND_COLLECTION_DESCRIPTOR,
};
use triblespace_core::collection::simplearchive_union;
use triblespace_core::collection::succinctarchive_union::{
    self, Rank9AcceleratedSuccinctArchiveArtifact, SuccinctArchiveCollection,
};
use triblespace_core::collection::{
    reach, Collection, CollectionArtifact, CollectionRecord, CollectionRecordSelector,
    CollectionStore, CollectionStoreExt, ACTION_WRITE,
};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::{Inline, InlineEncoding};
use triblespace_core::metadata::{self, MetaDescribe};
use triblespace_core::prelude::entity;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::{
    ArtifactHandle, ArtifactOfferSnapshot, ArtifactOfferStore, BlobStore, BlobStoreGet,
    BlobStorePut, CapabilityProofStore,
};
use triblespace_core::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

#[derive(Clone, Debug, Eq, PartialEq)]
enum StoreEvent {
    Put([u8; 32]),
    Offer(Vec<[u8; 32]>),
    Insert(triblespace_core::id::Id),
}

#[derive(Default)]
struct CountingRepo {
    inner: MemoryRepo,
    puts: BTreeMap<[u8; 32], usize>,
    events: Vec<StoreEvent>,
}

impl CountingRepo {
    fn puts_for<S>(&self, handle: Inline<Handle<S>>) -> usize
    where
        S: BlobEncoding,
        Handle<S>: InlineEncoding,
    {
        self.puts.get(&handle.raw).copied().unwrap_or_default()
    }
}

impl BlobStorePut for CountingRepo {
    type PutError = <MemoryRepo as BlobStorePut>::PutError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let handle = self.inner.put(item)?;
        *self.puts.entry(handle.raw).or_default() += 1;
        self.events.push(StoreEvent::Put(handle.raw));
        Ok(handle)
    }
}

impl BlobStore for CountingRepo {
    type Reader = <MemoryRepo as BlobStore>::Reader;
    type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.inner.reader()
    }
}

impl CollectionStore for CountingRepo {
    type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;
    type RecordIter<'a> = <MemoryRepo as CollectionStore>::RecordIter<'a>;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.inner.records()
    }

    fn record(
        &mut self,
        id: triblespace_core::id::Id,
    ) -> Result<Option<CollectionRecord>, Self::RecordsError> {
        self.inner.record(id)
    }

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        self.inner.select_records(selectors)
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.events.push(StoreEvent::Insert(record.id()));
        self.inner.insert(record)
    }
}

impl CapabilityProofStore for CountingRepo {
    type ProofsError = <MemoryRepo as CapabilityProofStore>::ProofsError;
    type InsertError = <MemoryRepo as CapabilityProofStore>::InsertError;
    type ProofIter<'a> = <MemoryRepo as CapabilityProofStore>::ProofIter<'a>;

    fn proofs<'a>(&'a mut self) -> Result<Self::ProofIter<'a>, Self::ProofsError> {
        self.inner.proofs()
    }

    fn proof(
        &mut self,
        id: CapabilityProofId,
    ) -> Result<Option<CapabilityProof>, Self::ProofsError> {
        self.inner.proof(id)
    }

    fn insert_proof(&mut self, proof: CapabilityProof) -> Result<(), Self::InsertError> {
        self.inner.insert_proof(proof)
    }
}

impl ArtifactOfferStore for CountingRepo {
    type OfferError = <MemoryRepo as ArtifactOfferStore>::OfferError;

    fn offer_all<I>(&mut self, handles: I) -> Result<(), Self::OfferError>
    where
        I: IntoIterator<Item = ArtifactHandle>,
    {
        let handles = handles.into_iter().collect::<Vec<_>>();
        self.events.push(StoreEvent::Offer(
            handles.iter().map(|handle| handle.raw).collect(),
        ));
        self.inner.offer_all(handles)
    }

    fn offers_snapshot(&mut self) -> Result<ArtifactOfferSnapshot, Self::OfferError> {
        self.inner.offers_snapshot()
    }
}

fn fragment(entity: u8) -> Fragment {
    let mut row = [entity; TRIBLE_LEN];
    row[16..32].fill(1);
    let mut facts = TribleSet::new();
    facts.insert(&Trible::force_raw(row).unwrap());
    Fragment::from(facts)
}

fn store_write_proof<S>(
    store: &mut S,
    authority: &SigningKey,
    writer: VerifyingKey,
    collection: Collection<SimpleArchive>,
) where
    S: BlobStorePut + CapabilityProofStore,
{
    let atom = CapabilityAtom::new(
        CapabilityAction::new(ACTION_WRITE),
        CapabilityResource::from(collection.handle()),
    );
    let bundle = CapabilityProofBundle::issue_root(
        authority,
        CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
        writer,
    )
    .unwrap();
    store_proof_bundle(store, bundle);
}

fn store_proof_bundle<S>(store: &mut S, bundle: CapabilityProofBundle)
where
    S: BlobStorePut + CapabilityProofStore,
{
    let (proof, claims) = bundle.into_parts();
    for claim in claims {
        store.put::<SimpleArchive, _>(claim).unwrap();
    }
    store.insert_proof(proof).unwrap();
}

#[test]
fn registration_offers_the_complete_descriptor_closure_once() {
    let authority = SigningKey::from_bytes(&[1; 32]);
    let descriptor =
        simplearchive_union::descriptor("closure", authority.verifying_key(), reach::private());
    let mut attachments = descriptor.blobs().clone();
    let attachment_handles: Vec<_> = attachments
        .reader()
        .unwrap()
        .into_iter()
        .map(|(handle, _)| handle)
        .collect();
    assert!(!attachment_handles.is_empty());

    let mut store = CountingRepo::default();
    let collection = store.collection::<SimpleArchive>(descriptor).unwrap();
    let offers = store.offers_snapshot().unwrap();

    assert!(offers.contains(collection.handle().transmute()));
    for attachment in attachment_handles {
        assert!(offers.contains(attachment));
    }
    assert_eq!(store.puts_for(collection.handle()), 1);
}

#[test]
fn direct_stage_retains_descriptor_attachments_and_publishes_commit_last() {
    use triblespace_core::blob::encodings::utf8string::UTF8String;
    use triblespace_core::collection::descriptor;

    let signer = SigningKey::from_bytes(&[11; 32]);
    let collection_descriptor = simplearchive_union::descriptor(
        "direct-stage-name",
        signer.verifying_key(),
        reach::private(),
    );
    let name = descriptor::name(collection_descriptor.facts())
        .unwrap()
        .expect("root descriptor name");
    let collection: Inline<Handle<SimpleArchive>> =
        collection_descriptor.facts().clone().to_blob().get_handle();
    let empty: Blob<SimpleArchive> = TribleSet::new().to_blob();
    let prepared = simplearchive_union::prepare_commit(&collection_descriptor, &empty, &empty)
        .expect("prepare direct commit");
    let mut store = CountingRepo::default();

    let mut staged = prepared.stage(&mut store, &signer).unwrap();
    let commit = *staged.commit();
    assert!(staged
        .store_mut()
        .records()
        .unwrap()
        .collect::<Result<Vec<_>, Infallible>>()
        .unwrap()
        .is_empty());
    staged.finalize().unwrap();

    let name_put = store
        .events
        .iter()
        .position(|event| matches!(event, StoreEvent::Put(raw) if *raw == name.raw))
        .expect("name attachment put");
    let descriptor_put = store
        .events
        .iter()
        .position(|event| matches!(event, StoreEvent::Put(raw) if *raw == collection.raw))
        .expect("descriptor put");
    let insert = store
        .events
        .iter()
        .position(|event| matches!(event, StoreEvent::Insert(id) if *id == commit.id()))
        .expect("commit insert");
    assert!(name_put < descriptor_put && descriptor_put < insert);

    let reader = store.reader().unwrap();
    let stored_name: Blob<UTF8String> = reader.get(name).unwrap();
    assert_eq!(
        std::str::from_utf8(&stored_name.bytes).unwrap(),
        "direct-stage-name"
    );
    let offers = store.offers_snapshot().unwrap();
    assert!(offers.contains(name.transmute()));
    assert!(offers.contains(collection.transmute()));
}

#[test]
fn commit_does_not_rewrite_the_registered_descriptor() {
    let authority = SigningKey::from_bytes(&[2; 32]);
    let mut store = CountingRepo::default();
    let collection = store
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            "no-reput",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let descriptor_puts = store.puts_for(collection.handle());

    store.commit(collection, &authority, fragment(1)).unwrap();

    assert_eq!(store.puts_for(collection.handle()), descriptor_puts);
}

#[test]
fn commit_offers_every_dependency_before_one_idempotent_record() {
    let authority = SigningKey::from_bytes(&[9; 32]);
    let mut store = CountingRepo::default();
    let collection = store
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            "publication-order",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    store.events.clear();

    let mut committed = fragment(1);
    *committed.metafacts_mut() += fragment(2).into_facts();
    let attachment = committed
        .put::<triblespace_core::blob::encodings::utf8string::UTF8String, _>("attached".to_owned());
    let data: Inline<Handle<SimpleArchive>> = committed.facts().clone().to_blob().get_handle();
    let metadata: Inline<Handle<SimpleArchive>> =
        committed.metafacts().clone().to_blob().get_handle();

    let first = store
        .commit(collection, &authority, committed.clone())
        .unwrap();
    let insert = store
        .events
        .iter()
        .position(|event| matches!(event, StoreEvent::Insert(id) if *id == first.id()))
        .unwrap();
    let offered = store.events[..insert]
        .iter()
        .find_map(|event| match event {
            StoreEvent::Offer(handles) => Some(handles),
            _ => None,
        })
        .unwrap();
    let expected_offers: [ArtifactHandle; 3] = [
        attachment.transmute(),
        data.transmute(),
        metadata.transmute(),
    ];
    for handle in expected_offers {
        assert!(offered.contains(&handle.raw));
    }
    assert!(!store.events[..insert]
        .iter()
        .any(|event| matches!(event, StoreEvent::Put(raw) if *raw == collection.handle().raw)));

    let repeated = store.commit(collection, &authority, committed).unwrap();
    assert_eq!(repeated, first);
    assert_eq!(
        store
            .inner
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, Infallible>>()
            .unwrap()
            .len(),
        1,
    );
}

#[test]
fn authority_is_descriptor_local_and_delegation_activates_resident_commits() {
    let authority = SigningKey::from_bytes(&[3; 32]);
    let delegate = SigningKey::from_bytes(&[4; 32]);
    let mut store = CountingRepo::default();
    let collection = store
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            "authority",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();

    let delegated = store.commit(collection, &delegate, fragment(2)).unwrap();
    assert!(store.cover(collection).unwrap().is_empty());

    let root = store.commit(collection, &authority, fragment(1)).unwrap();
    let authority_cover = store.cover(collection).unwrap();
    assert_eq!(authority_cover.collection(), collection);
    assert_eq!(
        authority_cover.members().collect::<Vec<_>>(),
        vec![Handle::<SimpleArchive>::from_hash(root.data())]
    );

    store_write_proof(&mut store, &authority, delegate.verifying_key(), collection);
    let cover_before_duplicate = store.cover(collection).unwrap();
    assert_eq!(cover_before_duplicate.len(), 2);

    // A second authorized signer may attest the same payload with a distinct
    // signed record. Provenance grows, but the payload lattice point does not.
    let duplicate = store.commit(collection, &delegate, fragment(1)).unwrap();
    assert_ne!(duplicate.id(), root.id());
    assert_eq!(duplicate.data(), root.data());

    let cover = store.cover(collection).unwrap();
    assert_eq!(cover, cover_before_duplicate);
    assert_eq!(cover.collection(), collection);
    assert_eq!(cover.len(), 2);
    assert_eq!(
        cover.members().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            Handle::<SimpleArchive>::from_hash(root.data()),
            Handle::<SimpleArchive>::from_hash(delegated.data()),
        ]),
    );
    assert_eq!(
        store
            .claims(&cover)
            .unwrap()
            .into_iter()
            .map(|claim| claim.id())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([root.id(), delegated.id(), duplicate.id()]),
    );

    let snapshot = store.snapshot(collection).unwrap();
    let mut expected = fragment(1).into_facts();
    expected += fragment(2).into_facts();
    assert_eq!(snapshot.facts(), &expected);
    assert_eq!(snapshot.cover(), &cover);
    let materialized: TribleSet = store.materialize(&cover).unwrap();
    assert_eq!(materialized, expected);
}

#[test]
fn invalid_resident_proof_grants_nothing_without_poisoning_valid_evidence() {
    let authority = SigningKey::from_bytes(&[5; 32]);
    let wrong_resource = SigningKey::from_bytes(&[6; 32]);
    let valid_delegate = SigningKey::from_bytes(&[16; 32]);
    let missing_claim = SigningKey::from_bytes(&[19; 32]);
    let expired = SigningKey::from_bytes(&[20; 32]);
    let invalid_signature = SigningKey::from_bytes(&[21; 32]);
    let mut store = MemoryRepo::default();
    let collection = store
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            "target",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let other = store
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            "other",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let valid_commit = store
        .commit(collection, &valid_delegate, fragment(1))
        .unwrap();
    store
        .commit(collection, &wrong_resource, fragment(2))
        .unwrap();
    store
        .commit(collection, &missing_claim, fragment(3))
        .unwrap();
    store.commit(collection, &expired, fragment(4)).unwrap();
    store
        .commit(collection, &invalid_signature, fragment(5))
        .unwrap();

    store_write_proof(
        &mut store,
        &authority,
        wrong_resource.verifying_key(),
        other,
    );
    store_write_proof(
        &mut store,
        &authority,
        valid_delegate.verifying_key(),
        collection,
    );

    let atom = CapabilityAtom::new(
        CapabilityAction::new(ACTION_WRITE),
        CapabilityResource::from(collection.handle()),
    );
    let missing_bundle = CapabilityProofBundle::issue_root(
        &authority,
        CapabilityClaim::root(
            atom,
            CapabilityMode::Invoke,
            Some(
                CapabilityValidity::new(
                    hifitime::Epoch::from_tai_seconds(0.0),
                    hifitime::Epoch::from_tai_seconds(1_000_000_000_000.0),
                )
                .unwrap(),
            ),
        ),
        missing_claim.verifying_key(),
    )
    .unwrap();
    store.insert_proof(missing_bundle.into_parts().0).unwrap();

    let expired_validity = CapabilityValidity::new(
        hifitime::Epoch::from_tai_seconds(0.0),
        hifitime::Epoch::from_tai_seconds(1.0),
    )
    .unwrap();
    let expired_bundle = CapabilityProofBundle::issue_root(
        &authority,
        CapabilityClaim::root(atom, CapabilityMode::Invoke, Some(expired_validity)),
        expired.verifying_key(),
    )
    .unwrap();
    store_proof_bundle(&mut store, expired_bundle);

    let tampered_bundle = CapabilityProofBundle::issue_root(
        &authority,
        CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
        invalid_signature.verifying_key(),
    )
    .unwrap();
    let (proof, claims) = tampered_bundle.into_parts();
    let mut proof_bytes = proof.into_bytes();
    proof_bytes[32] ^= 0x80;
    let tampered = CapabilityProof::from_bytes(&proof_bytes).unwrap();
    store_proof_bundle(&mut store, CapabilityProofBundle::new(tampered, claims));

    let cover = store.cover(collection).unwrap();
    assert_eq!(
        cover.members().collect::<Vec<_>>(),
        vec![Handle::<SimpleArchive>::from_hash(valid_commit.data())]
    );
}

#[test]
fn pile_reopen_discovers_resident_delegation_proof_and_claims() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("automatic-collection-auth.pile");
    std::fs::File::create(&path).unwrap();

    let authority = SigningKey::from_bytes(&[17; 32]);
    let delegate = SigningKey::from_bytes(&[18; 32]);
    let mut pile = Pile::open(&path).unwrap();
    let collection = pile
        .collection::<SimpleArchive>(simplearchive_union::descriptor(
            "reopen-auth",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let committed = pile.commit(collection, &delegate, fragment(3)).unwrap();
    let atom = CapabilityAtom::new(
        CapabilityAction::new(ACTION_WRITE),
        CapabilityResource::from(collection.handle()),
    );
    let bundle = CapabilityProofBundle::issue_root(
        &authority,
        CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
        delegate.verifying_key(),
    )
    .unwrap();
    let (proof, claims) = bundle.into_parts();
    for claim in claims {
        pile.put::<SimpleArchive, _>(claim).unwrap();
    }
    pile.insert_proof(proof).unwrap();
    pile.close().unwrap();

    let mut reopened = Pile::open(&path).unwrap();
    reopened.refresh().unwrap();
    let cover = reopened.cover(collection).unwrap();
    assert_eq!(
        cover.members().collect::<Vec<_>>(),
        vec![Handle::<SimpleArchive>::from_hash(committed.data())]
    );
    assert_eq!(
        reopened.snapshot(collection).unwrap().facts(),
        &fragment(3).into_facts()
    );
    reopened.close().unwrap();
}

#[test]
fn pile_commit_reopens_complete_rank9_accelerated_artifact() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("accelerated-artifact-commit.pile");
    std::fs::File::create(&path).unwrap();

    let authority = SigningKey::from_bytes(&[19; 32]);
    let facade = SuccinctArchiveCollection::new(
        "direct-accelerated",
        authority.verifying_key(),
        reach::private(),
        authority.verifying_key(),
        reach::private(),
    );
    let raw = succinctarchive_union::derive_element(&fragment(4).into_facts().to_blob()).unwrap();
    let artifact = Rank9AcceleratedSuccinctArchiveArtifact::from_raw(raw.clone()).unwrap();
    let root = artifact.root().clone();

    let mut pile = Pile::open(&path).unwrap();
    let collection = pile
        .collection::<Rank9AcceleratedSuccinctArchiveBlob>(facade.descriptor())
        .unwrap();
    let committed = pile.commit(collection, &authority, artifact).unwrap();
    assert_eq!(
        committed.data(),
        Handle::<Rank9AcceleratedSuccinctArchiveBlob>::to_hash(root.get_handle())
    );
    pile.close().unwrap();

    let mut reopened = Pile::open(&path).unwrap();
    let reader = reopened.reader().unwrap();
    reader
        .get::<Blob<SuccinctArchiveBlob>, _>(raw.get_handle())
        .unwrap();
    reader
        .get::<Blob<Rank9AcceleratedSuccinctArchiveBlob>, _>(root.get_handle())
        .unwrap();
    drop(reader);
    let snapshot: triblespace_core::collection::Snapshot<
        Rank9AcceleratedSuccinctArchiveBlob,
        UnionArchive<OrderedUniverse>,
        _,
    > = reopened.snapshot(collection).unwrap();
    assert_eq!(snapshot.value().iter().count(), 1);
    reopened.close().unwrap();
}

#[test]
fn anchorless_descriptor_is_rejected_before_storage() {
    let authority = SigningKey::from_bytes(&[7; 32]).verifying_key();
    let anchorless = entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_authority: authority,
        collection_representation: <SimpleArchive as MetaDescribe>::id(),
    };
    let mut store = CountingRepo::default();

    assert!(matches!(
        store.collection::<SimpleArchive>(anchorless),
        Err(
            triblespace_core::collection::CollectionRegistrationError::InvalidDescriptor(
                triblespace_core::collection::RecordDecodeError::MissingField(_)
            )
        )
    ));
    assert!(store.puts.is_empty());
}

#[test]
fn root_descriptor_without_its_name_blob_is_rejected_before_storage() {
    let authority = SigningKey::from_bytes(&[8; 32]);
    let descriptor = simplearchive_union::descriptor(
        "stripped-name",
        authority.verifying_key(),
        reach::private(),
    );
    let (_, facts, metafacts, _) = descriptor.into_parts();
    let stripped = Fragment::from_parts(facts, metafacts, Default::default());
    let mut store = CountingRepo::default();

    assert!(matches!(
        store.collection::<SimpleArchive>(stripped),
        Err(
            triblespace_core::collection::CollectionRegistrationError::InvalidAttachment {
                role: "name",
                ..
            }
        )
    ));
    assert!(store.puts.is_empty());
}

#[test]
fn descriptor_without_authority_is_rejected_before_storage() {
    let missing_authority = entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: "missing-authority".to_owned(),
        collection_representation: <SimpleArchive as MetaDescribe>::id(),
    };
    let mut store = CountingRepo::default();

    assert!(matches!(
        store.collection::<SimpleArchive>(missing_authority),
        Err(
            triblespace_core::collection::CollectionRegistrationError::InvalidDescriptor(
                triblespace_core::collection::RecordDecodeError::MissingField(
                    "collection_authority"
                )
            )
        )
    ));
    assert!(store.puts.is_empty());
}

// Keep the projected memory-repository error in this test crate's type graph;
// it catches accidental changes to the blanket extension's concrete bounds.
const _: Option<Infallible> = None;
