use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;

use ed25519_dalek::{SigningKey, VerifyingKey};

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{BlobEncoding, IntoBlob};
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    CapabilityResource,
};
use triblespace_core::collection::records::{
    collection_authority, collection_name, collection_recipe, collection_representation,
    KIND_COLLECTION_DESCRIPTOR,
};
use triblespace_core::collection::simplearchive_union::{self, TRIBLE_SET_UNION_RECIPE_V1};
use triblespace_core::collection::{
    reach, CapabilityPresentation, CollectionRecord, CollectionRecordSelector, CollectionStore,
    CollectionStoreExt, ACTION_WRITE,
};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::{Inline, InlineEncoding};
use triblespace_core::metadata::{self, MetaDescribe};
use triblespace_core::prelude::entity;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{
    ArtifactHandle, ArtifactOfferSnapshot, ArtifactOfferStore, BlobStore, BlobStorePut,
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

fn write_presentation(
    authority: &SigningKey,
    writer: VerifyingKey,
    collection: triblespace_core::collection::CollectionHandle,
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
    let collection = store.collection(descriptor).unwrap();
    let offers = store.offers_snapshot().unwrap();

    assert!(offers.contains(collection.transmute()));
    for attachment in attachment_handles {
        assert!(offers.contains(attachment));
    }
    assert_eq!(store.puts_for(collection), 1);
}

#[test]
fn commit_does_not_rewrite_the_registered_descriptor() {
    let authority = SigningKey::from_bytes(&[2; 32]);
    let mut store = CountingRepo::default();
    let collection = store
        .collection(simplearchive_union::descriptor(
            "no-reput",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let descriptor_puts = store.puts_for(collection);

    store.commit(collection, &authority, fragment(1)).unwrap();

    assert_eq!(store.puts_for(collection), descriptor_puts);
}

#[test]
fn commit_offers_every_dependency_before_one_idempotent_record() {
    let authority = SigningKey::from_bytes(&[9; 32]);
    let mut store = CountingRepo::default();
    let collection = store
        .collection(simplearchive_union::descriptor(
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
        .any(|event| matches!(event, StoreEvent::Put(raw) if *raw == collection.raw)));

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
        .collection(simplearchive_union::descriptor(
            "authority",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();

    let delegated = store.commit(collection, &delegate, fragment(2)).unwrap();
    assert!(store.ticket(collection, &[]).unwrap().is_empty());

    let root = store.commit(collection, &authority, fragment(1)).unwrap();
    assert_eq!(store.ticket(collection, &[]).unwrap().commits(), &[root]);

    let proof = write_presentation(&authority, delegate.verifying_key(), collection);
    let ticket = store.ticket(collection, &[proof.clone()]).unwrap();
    assert_eq!(ticket.collection(), collection);
    assert_eq!(
        ticket
            .commits()
            .iter()
            .map(|commit| commit.id())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([root.id(), delegated.id()]),
    );

    let snapshot = store.snapshot(collection, &[proof]).unwrap();
    let mut expected = fragment(1).into_facts();
    expected += fragment(2).into_facts();
    assert_eq!(snapshot.facts(), &expected);
    assert_eq!(snapshot.ticket(), &ticket);
    assert_eq!(store.materialize(&ticket).unwrap(), expected);
}

#[test]
fn invalid_explicit_presentation_fails_loud() {
    let authority = SigningKey::from_bytes(&[5; 32]);
    let delegate = SigningKey::from_bytes(&[6; 32]);
    let other_delegate = SigningKey::from_bytes(&[16; 32]);
    let mut store = CountingRepo::default();
    let collection = store
        .collection(simplearchive_union::descriptor(
            "target",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let other = store
        .collection(simplearchive_union::descriptor(
            "other",
            authority.verifying_key(),
            reach::private(),
        ))
        .unwrap();
    let wrong = write_presentation(&authority, delegate.verifying_key(), other);
    let valid = write_presentation(&authority, other_delegate.verifying_key(), collection);

    assert!(matches!(
        store.ticket(collection, &[valid, wrong]),
        Err(triblespace_core::collection::CollectionTicketError::Admission(error))
            if error.presentation() == 1
    ));
}

#[test]
fn anchorless_descriptor_is_rejected_before_storage() {
    let authority = SigningKey::from_bytes(&[7; 32]).verifying_key();
    let anchorless = entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_authority: authority,
        collection_representation: <SimpleArchive as MetaDescribe>::id(),
        collection_recipe: TRIBLE_SET_UNION_RECIPE_V1,
    };
    let mut store = CountingRepo::default();

    assert!(matches!(
        store.collection(anchorless),
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
        store.collection(stripped),
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
        collection_recipe: TRIBLE_SET_UNION_RECIPE_V1,
    };
    let mut store = CountingRepo::default();

    assert!(matches!(
        store.collection(missing_authority),
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
