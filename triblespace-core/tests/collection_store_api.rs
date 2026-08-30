use std::collections::BTreeMap;

use ed25519_dalek::{SigningKey, VerifyingKey};
use hifitime::Epoch;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::utf8string::UTF8String;
use triblespace_core::blob::{BlobEncoding, IntoBlob};
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    CapabilityRequest, CapabilityResource,
};
use triblespace_core::collection::descriptor;
use triblespace_core::collection::{
    AdmissionPolicy, Collection, CollectionPolicy, CollectionRecord, CollectionStore,
    CollectionStoreExt, ACTION_READ, ACTION_WRITE,
};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::{Inline, InlineEncoding};
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{BlobStoreGet, BlobStorePut, CapabilityProofStore, SnapshotSource};
use triblespace_core::trible::{Fragment, Trible, TribleSet, TRIBLE_LEN};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StoreEvent {
    Put([u8; 32]),
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
        T: triblespace_core::blob::IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let handle = self.inner.put(item)?;
        *self.puts.entry(handle.raw).or_default() += 1;
        self.events.push(StoreEvent::Put(handle.raw));
        Ok(handle)
    }
}

impl CollectionStore for CountingRepo {
    type InsertError = <MemoryRepo as CollectionStore>::InsertError;

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.events.push(StoreEvent::Insert(record.id()));
        self.inner.insert(record)
    }
}

fn key(byte: u8) -> SigningKey {
    SigningKey::from_bytes(&[byte; 32])
}

fn policy(root: VerifyingKey) -> CollectionPolicy {
    CollectionPolicy::new(
        AdmissionPolicy::delegable(root),
        AdmissionPolicy::direct(root),
    )
}

fn fragment(entity: u8) -> Fragment {
    let mut row = [entity; TRIBLE_LEN];
    row[16..32].fill(entity.wrapping_add(1));
    row[32..].fill(entity.wrapping_add(2));
    let mut facts = TribleSet::new();
    facts.insert(&Trible::force_raw(row).unwrap());
    Fragment::from(facts)
}

fn atom(action: triblespace_core::id::Id, collection: Collection<SimpleArchive>) -> CapabilityAtom {
    CapabilityAtom::new(
        CapabilityAction::new(action),
        CapabilityResource::from(collection.handle()),
    )
}

fn store_bundle(store: &mut MemoryRepo, bundle: CapabilityProofBundle) {
    let (proof, claims) = bundle.into_parts();
    for claim in claims {
        store.put::<SimpleArchive, _>(claim).unwrap();
    }
    store.insert_proof(proof).unwrap();
}

#[test]
fn root_creation_registers_a_self_contained_descriptor() {
    let root = key(1);
    let expected_policy = policy(root.verifying_key());
    let mut store = MemoryRepo::default();

    let collection = store
        .collection("collection-store-api", expected_policy.clone())
        .unwrap();
    let snapshot = store.snapshot().unwrap();
    let descriptor_blob = snapshot
        .get::<TribleSet, SimpleArchive>(collection.handle())
        .unwrap();

    assert_eq!(descriptor::policy(&descriptor_blob), Ok(expected_policy));
    let name = descriptor::name(&descriptor_blob).unwrap().unwrap();
    let name: anybytes::View<str> = snapshot.get::<_, UTF8String>(name).unwrap();
    assert_eq!(&*name, "collection-store-api");
}

#[test]
fn commit_is_local_and_correct_by_construction() {
    let root = key(2);
    let mut store = CountingRepo::default();
    let collection = store
        .collection("registered", policy(root.verifying_key()))
        .unwrap();
    let descriptor_puts = store.puts_for(collection.handle());
    store.events.clear();

    let expected_data = fragment(7).facts().clone().to_blob().get_handle();
    let commit = store.commit(collection, &root, fragment(7)).unwrap();

    assert_eq!(
        Handle::<SimpleArchive>::from_hash(commit.data()),
        expected_data
    );
    assert_eq!(descriptor_puts, 1);
    assert_eq!(store.puts_for(collection.handle()), descriptor_puts);
    assert_eq!(store.events.last(), Some(&StoreEvent::Insert(commit.id())));
}

#[test]
fn read_and_write_policies_are_independent() {
    let root = key(3);
    let stranger = key(4);
    let mut store = MemoryRepo::default();
    let collection = store
        .collection(
            "independent-actions",
            CollectionPolicy::new(
                AdmissionPolicy::Open,
                AdmissionPolicy::direct(root.verifying_key()),
            ),
        )
        .unwrap();
    store.commit(collection, &root, fragment(1)).unwrap();
    store.commit(collection, &stranger, fragment(2)).unwrap();

    let snapshot = store.snapshot().unwrap();
    assert!(collection
        .reader_is_admitted_at(
            &snapshot,
            stranger.verifying_key(),
            Epoch::from_tai_seconds(0.0)
        )
        .unwrap());
    assert!(!collection
        .writer_is_admitted_at(
            &snapshot,
            stranger.verifying_key(),
            Epoch::from_tai_seconds(0.0)
        )
        .unwrap());
    let (cover, commits) = collection
        .admitted_with_commits_at(&snapshot, Epoch::from_tai_seconds(0.0))
        .unwrap();
    assert_eq!(cover.len(), 1);
    assert_eq!(commits.len(), 1);
    assert_eq!(commits[0].public_key().raw, root.verifying_key().to_bytes());
}

#[test]
fn direct_policy_accepts_root_grants_but_blocks_redelegation() {
    let root = key(5);
    let intermediary = key(6);
    let leaf = key(7);
    let mut store = MemoryRepo::default();
    let collection = store
        .collection(
            "direct-only",
            CollectionPolicy::new(
                AdmissionPolicy::direct(root.verifying_key()),
                AdmissionPolicy::direct(root.verifying_key()),
            ),
        )
        .unwrap();
    let write_atom = atom(ACTION_WRITE, collection);
    let parent_bundle = CapabilityProofBundle::issue_root(
        &root,
        CapabilityClaim::root(write_atom, CapabilityMode::InvokeAndDelegate, None),
        intermediary.verifying_key(),
    )
    .unwrap();
    let parent = parent_bundle
        .verify(
            root.verifying_key(),
            Epoch::from_tai_seconds(0.0),
            intermediary.verifying_key(),
            CapabilityRequest::new(write_atom, CapabilityMode::InvokeAndDelegate),
        )
        .unwrap();
    let child_bundle = parent
        .delegate(
            &intermediary,
            CapabilityClaim::delegated(
                parent.claim_handle(),
                write_atom,
                CapabilityMode::Invoke,
                None,
            ),
            leaf.verifying_key(),
        )
        .unwrap();
    store_bundle(&mut store, parent_bundle);
    store_bundle(&mut store, child_bundle);

    let snapshot = store.snapshot().unwrap();
    let instant = Epoch::from_tai_seconds(0.0);
    assert!(collection
        .writer_is_admitted_at(&snapshot, intermediary.verifying_key(), instant)
        .unwrap());
    assert!(!collection
        .writer_is_admitted_at(&snapshot, leaf.verifying_key(), instant)
        .unwrap());
}

#[test]
fn read_grants_use_the_distinct_read_action() {
    let root = key(8);
    let reader = key(9);
    let mut store = MemoryRepo::default();
    let collection = store
        .collection(
            "read-action",
            CollectionPolicy::new(
                AdmissionPolicy::direct(root.verifying_key()),
                AdmissionPolicy::direct(root.verifying_key()),
            ),
        )
        .unwrap();
    let read_atom = atom(ACTION_READ, collection);
    store_bundle(
        &mut store,
        CapabilityProofBundle::issue_root(
            &root,
            CapabilityClaim::root(read_atom, CapabilityMode::Invoke, None),
            reader.verifying_key(),
        )
        .unwrap(),
    );

    let snapshot = store.snapshot().unwrap();
    let instant = Epoch::from_tai_seconds(0.0);
    assert!(collection
        .reader_is_admitted_at(&snapshot, reader.verifying_key(), instant)
        .unwrap());
    assert!(!collection
        .writer_is_admitted_at(&snapshot, reader.verifying_key(), instant)
        .unwrap());
}

#[test]
fn collection_quorum_needs_support_from_distinct_roots() {
    let first_root = key(10);
    let second_root = key(11);
    let writer = key(12);
    let mut store = MemoryRepo::default();
    let collection = store
        .collection(
            "two-root-write-quorum",
            CollectionPolicy::new(
                AdmissionPolicy::Open,
                AdmissionPolicy::quorum(
                    [first_root.verifying_key(), second_root.verifying_key()],
                    2,
                    None,
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let write_atom = atom(ACTION_WRITE, collection);
    let instant = Epoch::from_tai_seconds(0.0);

    store_bundle(
        &mut store,
        CapabilityProofBundle::issue_root(
            &first_root,
            CapabilityClaim::root(write_atom, CapabilityMode::Invoke, None),
            writer.verifying_key(),
        )
        .unwrap(),
    );
    assert!(!collection
        .writer_is_admitted_at(&store.snapshot().unwrap(), writer.verifying_key(), instant)
        .unwrap());

    store_bundle(
        &mut store,
        CapabilityProofBundle::issue_root(
            &second_root,
            CapabilityClaim::root(write_atom, CapabilityMode::Invoke, None),
            writer.verifying_key(),
        )
        .unwrap(),
    );
    assert!(collection
        .writer_is_admitted_at(&store.snapshot().unwrap(), writer.verifying_key(), instant)
        .unwrap());
}
