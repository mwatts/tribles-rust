use std::collections::BTreeSet;
use std::collections::HashSet;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::VerifyingKey;

use crate::blob::encodings::UnknownBlob;
use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::blob::{MemoryBlobStore, MemoryBlobStoreSnapshot, TryFromBlob};
use crate::capability::{CapabilityProof, CapabilityProofId};
use crate::collection::store::selectors_match_record;
use crate::collection::{
    CollectionRead, CollectionRecord, CollectionRecordSelector, CollectionStore,
};
use crate::id::ID_LEN;
use crate::inline::INLINE_LEN;
use crate::patch::{Entry, IdentitySchema, XorSip128, PATCH};
use crate::prelude::*;
use crate::repo::peer::{PeerEvidence, PeerRead, PeerStore, PEER_EVIDENCE_BYTES_LEN};
use crate::repo::proof::{CapabilityProofRead, CapabilityProofStore};
use crate::repo::{
    BlobInfo, BlobMetadata, BlobStoreGet, BlobStoreList, BlobStoreMeta, SnapshotSource,
    StoreChanges, StoreScope, StoreScopeError, StoreSnapshot, WantRequest, WantStore,
};

use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;

type CollectionRecordIndex = PATCH<ID_LEN, IdentitySchema, CollectionRecord, XorSip128>;
type CapabilityProofIndex = PATCH<INLINE_LEN, IdentitySchema, CapabilityProof, XorSip128>;
type PeerEvidenceIndex = PATCH<PEER_EVIDENCE_BYTES_LEN, IdentitySchema, (), XorSip128>;

/// Simple in-memory implementation of the repository storage traits.
///
/// Useful for unit tests or ephemeral repositories where persistence is not
/// required.
#[derive(Clone, Debug, Default)]
pub struct MemoryRepo {
    /// In-memory blob store for all repository blobs.
    pub blobs: MemoryBlobStore,
    /// LWW-resolved typed requests (see [`WantStore`]). In memory the
    /// last-writer-wins resolution is just insert/remove of the complete
    /// request identity, including a collection route. Wants here are exactly
    /// as ephemeral as the blobs themselves — the trait is a capability,
    /// durability is the store's own property.
    pub wants: HashSet<WantRequest>,
    /// Canonical collection records keyed by intrinsic record id.
    collection_records: CollectionRecordIndex,
    /// Canonical complete capability proofs keyed by exact-body content id.
    capability_proofs: CapabilityProofIndex,
    /// Positive peer-routing evidence keyed by its complete canonical body.
    peer_evidence: PeerEvidenceIndex,
    /// Monotone local safety assertion binding this store to one team.
    store_scope: Option<VerifyingKey>,
}

impl StoreScope for MemoryRepo {
    type ScopeError = Infallible;

    fn store_scope(&mut self) -> Result<Option<VerifyingKey>, StoreScopeError<Self::ScopeError>> {
        Ok(self.store_scope)
    }

    fn bind_store_scope(
        &mut self,
        team: VerifyingKey,
    ) -> Result<(), StoreScopeError<Self::ScopeError>> {
        match self.store_scope {
            None => {
                self.store_scope = Some(team);
                Ok(())
            }
            Some(bound) if bound == team => Ok(()),
            Some(bound) => Err(StoreScopeError::conflict(bound, team)),
        }
    }
}

/// One O(1)-clone immutable observation of a [`MemoryRepo`].
///
/// The blob snapshot and all semantic indexes are frozen together, so
/// collection admission, capability verification, peer discovery, and payload
/// decoding cannot observe different prefixes. Local wants and store scope are
/// operational state and intentionally excluded.
#[derive(Clone, PartialEq, Eq)]
pub struct MemoryRepoSnapshot {
    blobs: MemoryBlobStoreSnapshot,
    collection_records: CollectionRecordIndex,
    capability_proofs: CapabilityProofIndex,
    peer_evidence: PeerEvidenceIndex,
}

impl StoreSnapshot for MemoryRepoSnapshot {
    fn changes_since(&self, previous: &Self) -> StoreChanges {
        let mut changes = StoreChanges::NONE;
        if previous.blobs != self.blobs {
            changes = changes.union(StoreChanges::BLOBS);
        }
        if previous.collection_records != self.collection_records {
            changes = changes.union(StoreChanges::COLLECTION_RECORDS);
        }
        if previous.capability_proofs != self.capability_proofs {
            changes = changes.union(StoreChanges::CAPABILITY_PROOFS);
        }
        if previous.peer_evidence != self.peer_evidence {
            changes = changes.union(StoreChanges::PEERS);
        }
        changes
    }
}

impl SnapshotSource for MemoryRepo {
    type Snapshot = MemoryRepoSnapshot;
    type SnapshotError = Infallible;

    fn snapshot(&mut self) -> Result<Self::Snapshot, Self::SnapshotError> {
        Ok(MemoryRepoSnapshot {
            blobs: self.blobs.snapshot()?,
            collection_records: self.collection_records.clone(),
            capability_proofs: self.capability_proofs.clone(),
            peer_evidence: self.peer_evidence.clone(),
        })
    }
}

/// Deterministic persistent snapshot of in-memory peer evidence.
pub struct MemoryPeerIter {
    inner: crate::patch::PATCHIntoOrderedIterator<
        PEER_EVIDENCE_BYTES_LEN,
        IdentitySchema,
        (),
        XorSip128,
    >,
}

impl Iterator for MemoryPeerIter {
    type Item = Result<PeerEvidence, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|bytes| {
            Ok(PeerEvidence::from_bytes(bytes)
                .expect("MemoryRepo only indexes validated peer evidence"))
        })
    }
}

impl PeerRead for MemoryRepoSnapshot {
    type PeersError = Infallible;
    type PeerIter<'a> = MemoryPeerIter;

    fn peers<'a>(&'a self) -> Result<Self::PeerIter<'a>, Self::PeersError> {
        Ok(MemoryPeerIter {
            inner: self.peer_evidence.clone().into_iter_ordered(),
        })
    }
}

impl PeerStore for MemoryRepo {
    type InsertError = Infallible;

    fn insert_peer(&mut self, evidence: PeerEvidence) -> Result<(), Self::InsertError> {
        self.peer_evidence.insert(&Entry::new(evidence.as_bytes()));
        Ok(())
    }
}

/// Deterministic persistent snapshot of in-memory collection records.
pub struct MemoryCollectionRecordIter {
    keys:
        crate::patch::PATCHIntoOrderedIterator<ID_LEN, IdentitySchema, CollectionRecord, XorSip128>,
    lookup: CollectionRecordIndex,
}

impl Iterator for MemoryCollectionRecordIter {
    type Item = Result<CollectionRecord, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.keys.next()?;
        let record = *self
            .lookup
            .get(&key)
            .expect("collection key from PATCH snapshot must retain its value");
        debug_assert_eq!(record.id().raw(), key);
        Some(Ok(record))
    }
}

/// Deterministic persistent snapshot of in-memory capability proofs.
pub struct MemoryCapabilityProofIter {
    keys: crate::patch::PATCHIntoOrderedIterator<
        INLINE_LEN,
        IdentitySchema,
        CapabilityProof,
        XorSip128,
    >,
    lookup: CapabilityProofIndex,
}

impl Iterator for MemoryCapabilityProofIter {
    type Item = Result<CapabilityProof, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.keys.next()?;
        let proof = self
            .lookup
            .get(&key)
            .expect("proof key from PATCH snapshot must retain its value");
        debug_assert_eq!(proof.id().raw, key);
        Some(Ok(proof.clone()))
    }
}

/// Failure while admitting a proof to [`MemoryRepo`].
#[derive(Debug)]
pub enum MemoryProofInsertError {
    /// An infeasible BLAKE3 collision named different canonical proof bytes.
    IdCollision { id: CapabilityProofId },
}

impl fmt::Display for MemoryProofInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdCollision { id } => {
                write!(f, "capability proof id {id:?} names different bytes")
            }
        }
    }
}

impl Error for MemoryProofInsertError {}

/// Failure while inserting a collection record into [`MemoryRepo`].
#[derive(Debug)]
pub enum MemoryCollectionInsertError {
    /// An infeasible intrinsic-id collision named different canonical bytes.
    IdCollision { id: Id },
}

impl fmt::Display for MemoryCollectionInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdCollision { id } => {
                write!(f, "collection record id {id} names different bytes")
            }
        }
    }
}

impl Error for MemoryCollectionInsertError {}

impl CapabilityProofRead for MemoryRepoSnapshot {
    type ProofsError = Infallible;
    type ProofIter<'a> = MemoryCapabilityProofIter;

    fn proofs<'a>(&'a self) -> Result<Self::ProofIter<'a>, Self::ProofsError> {
        let keys = self.capability_proofs.clone().into_iter_ordered();
        Ok(MemoryCapabilityProofIter {
            keys,
            lookup: self.capability_proofs.clone(),
        })
    }

    fn proof(&self, id: CapabilityProofId) -> Result<Option<CapabilityProof>, Self::ProofsError> {
        Ok(self.capability_proofs.get(&id.raw).cloned())
    }
}

impl CapabilityProofStore for MemoryRepo {
    type InsertError = MemoryProofInsertError;

    fn insert_proof(&mut self, proof: CapabilityProof) -> Result<(), Self::InsertError> {
        let id = proof.id();
        if let Some(existing) = self.capability_proofs.get(&id.raw) {
            return if existing.as_bytes() == proof.as_bytes() {
                Ok(())
            } else {
                Err(MemoryProofInsertError::IdCollision { id })
            };
        }
        self.capability_proofs
            .insert(&Entry::with_value(&id.raw, proof));
        Ok(())
    }
}

impl CollectionRead for MemoryRepoSnapshot {
    type RecordsError = Infallible;
    type RecordIter<'a> = MemoryCollectionRecordIter;

    fn records<'a>(&'a self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        let keys = self.collection_records.clone().into_iter_ordered();
        Ok(MemoryCollectionRecordIter {
            keys,
            lookup: self.collection_records.clone(),
        })
    }

    fn record(&self, id: Id) -> Result<Option<CollectionRecord>, Self::RecordsError> {
        Ok(self.collection_records.get(&id.raw()).copied())
    }

    fn select_records(
        &self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        if selectors.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .collection_records
            .iter_ordered()
            .map(|key| {
                *self
                    .collection_records
                    .get(key)
                    .expect("collection key from PATCH must retain its value")
            })
            .filter(|record| selectors_match_record(selectors, *record))
            .collect())
    }
}

impl CollectionStore for MemoryRepo {
    type InsertError = MemoryCollectionInsertError;

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        let id = record.id();
        if let Some(existing) = self.collection_records.get(&id.raw()) {
            return if existing == &record {
                Ok(())
            } else {
                Err(MemoryCollectionInsertError::IdCollision { id })
            };
        }
        self.collection_records
            .insert(&Entry::with_value(&id.raw(), record));
        Ok(())
    }
}

impl crate::repo::BlobStorePut for MemoryRepo {
    type PutError = <MemoryBlobStore as crate::repo::BlobStorePut>::PutError;
    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.blobs.put(item)
    }
}

impl BlobStoreList for MemoryRepoSnapshot {
    type Iter<'a>
        = <MemoryBlobStoreSnapshot as BlobStoreList>::Iter<'a>
    where
        Self: 'a;
    type Err = <MemoryBlobStoreSnapshot as BlobStoreList>::Err;

    fn blobs<'a>(&'a self) -> Self::Iter<'a> {
        self.blobs.blobs()
    }

    fn contains_blob<S>(&self, handle: Inline<Handle<S>>) -> Result<bool, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.blobs.contains_blob(handle)
    }

    fn blob_info<S>(&self, handle: Inline<Handle<S>>) -> Result<Option<BlobInfo>, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.blobs.blob_info(handle)
    }

    fn blobs_diff<'a>(&'a self, old: &Self) -> Self::Iter<'a> {
        self.blobs.blobs_diff(&old.blobs)
    }
}

impl BlobStoreMeta for MemoryRepoSnapshot {
    type MetaError = <MemoryBlobStoreSnapshot as BlobStoreMeta>::MetaError;

    fn metadata<S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<Option<BlobMetadata>, Self::MetaError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        self.blobs.metadata(handle)
    }
}

impl BlobStoreGet for MemoryRepoSnapshot {
    type GetError<E: Error + Send + Sync + 'static> =
        <MemoryBlobStoreSnapshot as BlobStoreGet>::GetError<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.blobs.get(handle)
    }
}

impl crate::repo::BlobChildren for MemoryRepoSnapshot {}

impl crate::repo::BlobStoreKeep for MemoryRepo {
    fn keep<I>(&mut self, handles: I)
    where
        I: IntoIterator<Item = Inline<Handle<UnknownBlob>>>,
    {
        let reader = self
            .blobs
            .snapshot()
            .expect("memory snapshot is infallible");
        let mut roots = crate::repo::RetentionRoots::new();
        for key in self.collection_records.iter() {
            let record = self
                .collection_records
                .get(key)
                .expect("collection key from PATCH must retain its value");
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            if commit.verify_strict().is_err() {
                continue;
            }
            for root in [
                Inline::<Handle<UnknownBlob>>::new(commit.collection().raw),
                Inline::<Handle<UnknownBlob>>::new(commit.data().raw),
                commit.metadata().transmute(),
            ] {
                if crate::repo::BlobStoreList::contains_blob(&reader, root).unwrap_or(false) {
                    roots.retain_recursive(root);
                }
            }
        }
        for key in self.capability_proofs.iter() {
            let proof = self
                .capability_proofs
                .get(key)
                .expect("proof key from PATCH must retain its value");
            if proof.verify_signatures().is_err() {
                continue;
            }
            for claim in proof.claim_handles() {
                let claim: Inline<Handle<UnknownBlob>> = claim.transmute();
                if crate::repo::BlobStoreList::contains_blob(&reader, claim).unwrap_or(false) {
                    roots.retain_direct(claim);
                }
            }
        }
        self.blobs
            .keep(handles.into_iter().chain(roots.expanded(&reader)));
    }
}

impl WantStore for MemoryRepo {
    type WantError = Infallible;

    type WantIter<'a> = std::vec::IntoIter<Result<WantRequest, Self::WantError>>;

    fn want(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.wants.insert(request);
        Ok(())
    }

    fn unwant(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.wants.remove(&request);
        Ok(())
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        // Want enumeration feeds sync-daemon fetch order, and HashSet's
        // per-instance seed would break deterministic simulation replay.
        let mut requests: Vec<WantRequest> = self.wants.iter().copied().collect();
        requests.sort();
        Ok(requests.into_iter().map(Ok).collect::<Vec<_>>().into_iter())
    }
}

impl crate::repo::StorageFlush for MemoryRepo {
    type Error = Infallible;

    fn flush(&mut self) -> Result<(), Self::Error> {
        // In-memory state has no sync point; durability is exactly the
        // process lifetime, same as the blobs themselves.
        Ok(())
    }
}

impl crate::repo::StorageClose for MemoryRepo {
    type Error = Infallible;

    fn close(self) -> Result<(), Self::Error> {
        // Nothing to do for the in-memory backend.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anybytes::Bytes;
    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::capability::{
        CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
        CapabilityResource,
    };
    use crate::collection::descriptor::{identity_for_tests, named_for_tests};
    use crate::collection::{CollectionDerive, CollectionMerge, CollectionPolicy};

    fn handle(byte: u8) -> Inline<Handle<UnknownBlob>> {
        Inline::new([byte; 32])
    }

    #[test]
    fn capability_proofs_are_an_idempotent_set_and_root_verified_claims() {
        use crate::repo::{BlobStoreGet, BlobStoreKeep};

        let mut repo = MemoryRepo::default();
        let root = SigningKey::from_bytes(&[61; 32]);
        let leaf = SigningKey::from_bytes(&[62; 32]);
        let action = CapabilityAction::new(Id::new([63; 16]).unwrap());
        let claim = CapabilityClaim::root(
            CapabilityAtom::new(action, CapabilityResource::new([64; 32])),
            CapabilityMode::Invoke,
            None,
        );
        let bundle = CapabilityProofBundle::issue_root(&root, claim, leaf.verifying_key()).unwrap();
        let proof = bundle.proof().clone();
        let claim_handle = repo
            .put::<crate::blob::encodings::simplearchive::SimpleArchive, _>(
                bundle.claims()[0].clone(),
            )
            .unwrap();

        repo.insert_proof(proof.clone()).unwrap();
        repo.insert_proof(proof.clone()).unwrap();
        let snapshot = repo.snapshot().unwrap();
        assert_eq!(snapshot.proof(proof.id()).unwrap(), Some(proof.clone()));
        assert_eq!(snapshot.proof(Inline::new([0; 32])).unwrap(), None);
        assert_eq!(
            snapshot
                .proofs()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![proof]
        );

        repo.keep(std::iter::empty());
        let snapshot = repo.snapshot().unwrap();
        assert!(snapshot
            .get::<Blob<crate::blob::encodings::simplearchive::SimpleArchive>, _>(claim_handle)
            .is_ok());
    }

    #[test]
    fn capability_proof_claim_roots_do_not_follow_coincident_resource_handles() {
        use crate::repo::{BlobStoreGet, BlobStoreKeep};

        let mut repo = MemoryRepo::default();
        let coincident_resource = repo
            .put::<UnknownBlob, _>(Bytes::from_source(b"opaque resource".to_vec()))
            .unwrap();
        let root = SigningKey::from_bytes(&[65; 32]);
        let leaf = SigningKey::from_bytes(&[66; 32]);
        let claim = CapabilityClaim::root(
            CapabilityAtom::new(
                CapabilityAction::new(Id::new([67; 16]).unwrap()),
                CapabilityResource::new(coincident_resource.raw),
            ),
            CapabilityMode::Invoke,
            None,
        );
        let bundle = CapabilityProofBundle::issue_root(&root, claim, leaf.verifying_key()).unwrap();
        let claim_handle = repo
            .put::<crate::blob::encodings::simplearchive::SimpleArchive, _>(
                bundle.claims()[0].clone(),
            )
            .unwrap();
        repo.insert_proof(bundle.proof().clone()).unwrap();

        repo.keep(std::iter::empty());
        let snapshot = repo.snapshot().unwrap();
        assert!(snapshot
            .get::<Blob<crate::blob::encodings::simplearchive::SimpleArchive>, _>(claim_handle)
            .is_ok());
        assert!(snapshot
            .get::<Blob<UnknownBlob>, _>(coincident_resource)
            .is_err());
    }

    /// Wants resolve last-writer-wins: want → listed, unwant →
    /// gone, re-want → listed again. Enumeration is sorted (stable
    /// across runs despite HashSet backing).
    #[test]
    fn wants_lww_roundtrip() {
        let mut repo = MemoryRepo::default();
        assert_eq!(repo.wants().unwrap().count(), 0);

        let first = WantRequest::blob(handle(1));
        let second = WantRequest::blob(handle(2));
        repo.want(second).unwrap();
        repo.want(first).unwrap();
        // Reasserting an existing want is idempotent.
        repo.want(first).unwrap();
        let wants: Vec<_> = repo.wants().unwrap().map(Result::unwrap).collect();
        assert_eq!(wants, vec![first, second], "sorted enumeration");

        repo.unwant(first).unwrap();
        let wants: Vec<_> = repo.wants().unwrap().map(Result::unwrap).collect();
        assert_eq!(wants, vec![second]);

        // A later want wins over the earlier retraction.
        repo.want(first).unwrap();
        assert_eq!(repo.wants().unwrap().count(), 2);
    }

    #[test]
    fn routed_blob_wants_keep_exact_identity_while_local_presence_satisfies_each() {
        let blob = Blob::<UnknownBlob>::new(Bytes::from_source(b"shared bytes".to_vec()));
        let handle = blob.get_handle();
        let first_route = WantRequest::blob_in_collection(Inline::new([3; 32]), handle);
        let second_route = WantRequest::blob_in_collection(Inline::new([4; 32]), handle);
        let local = WantRequest::blob(handle);

        let mut repo = MemoryRepo::default();
        repo.put::<UnknownBlob, _>(blob).unwrap();
        for request in [second_route, local, first_route] {
            repo.want(request).unwrap();
        }

        assert_eq!(
            repo.wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![local, first_route, second_route]
        );
        let snapshot = repo.snapshot().unwrap();
        for request in [local, first_route, second_route] {
            let requested = request
                .blob_handle()
                .expect("all fixtures are blob requests");
            assert!(snapshot.get::<Blob<UnknownBlob>, _>(requested).is_ok());
        }
        drop(snapshot);

        repo.unwant(first_route).unwrap();
        assert_eq!(
            repo.wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![local, second_route],
            "retracting one route must not erase another route or bare local intent"
        );
    }

    #[test]
    fn collection_records_are_idempotent_and_intrinsically_ordered() {
        let descriptor = named_for_tests("merged", Id::new([2; 16]).unwrap());
        let target = named_for_tests("derived", Id::new([8; 16]).unwrap());
        let merge = CollectionRecord::Merge(CollectionMerge::new(
            identity_for_tests(&descriptor),
            Inline::new([4; 32]),
            Inline::new([5; 32]),
            Inline::new([6; 32]),
        ));
        let derive = CollectionRecord::Derive(CollectionDerive::new(
            identity_for_tests(&target),
            Inline::new([10; 32]),
            Inline::new([11; 32]),
        ));
        let mut expected = vec![derive, merge];
        expected.sort_unstable_by_key(CollectionRecord::id);

        let mut repo = MemoryRepo::default();
        CollectionStore::insert(&mut repo, merge).unwrap();
        CollectionStore::insert(&mut repo, derive).unwrap();
        CollectionStore::insert(&mut repo, merge).unwrap();

        let snapshot = repo.snapshot().unwrap();
        let actual = snapshot
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(snapshot.record(merge.id()).unwrap(), Some(merge));
        assert_eq!(snapshot.record(Id::new([0xff; 16]).unwrap()).unwrap(), None);
    }

    #[test]
    fn collection_index_rejects_a_different_body_under_an_existing_key() {
        let target = identity_for_tests(&named_for_tests("target", Id::new([12; 16]).unwrap()));
        let expected = CollectionRecord::Derive(CollectionDerive::new(
            target,
            Inline::new([14; 32]),
            Inline::new([15; 32]),
        ));
        let mismatched = CollectionRecord::Derive(CollectionDerive::new(
            target,
            Inline::new([16; 32]),
            Inline::new([17; 32]),
        ));
        let id = expected.id();

        let mut repo = MemoryRepo::default();
        repo.collection_records
            .insert(&Entry::with_value(&id.raw(), mismatched));

        assert!(matches!(
            repo.insert(expected),
            Err(MemoryCollectionInsertError::IdCollision { id: found }) if found == id
        ));
        assert_eq!(repo.collection_records.get(&id.raw()), Some(&mismatched));
    }

    #[test]
    fn collection_primary_selection_answers_group_and_exact_conflicting_operations() {
        let source = identity_for_tests(&named_for_tests("source", Id::new([22; 16]).unwrap()));
        let target = identity_for_tests(&named_for_tests("target", Id::new([25; 16]).unwrap()));
        let other = identity_for_tests(&named_for_tests("other", Id::new([28; 16]).unwrap()));
        let input = Inline::new([30; 32]);
        let merge = CollectionRecord::Merge(CollectionMerge::new(
            source,
            Inline::new([31; 32]),
            Inline::new([32; 32]),
            Inline::new([33; 32]),
        ));
        let first =
            CollectionRecord::Derive(CollectionDerive::new(target, input, Inline::new([34; 32])));
        let conflicting =
            CollectionRecord::Derive(CollectionDerive::new(target, input, Inline::new([35; 32])));
        let sibling = CollectionRecord::Derive(CollectionDerive::new(
            target,
            Inline::new([36; 32]),
            Inline::new([37; 32]),
        ));
        let unrelated =
            CollectionRecord::Derive(CollectionDerive::new(other, input, Inline::new([38; 32])));
        let mut repo = MemoryRepo::default();
        for record in [unrelated, conflicting, merge, first, sibling, first] {
            repo.insert(record).unwrap();
        }

        let exact = [CollectionRecordSelector::Operation(WantRequest::derive(
            target, input,
        ))]
        .into_iter()
        .collect();
        let mut expected = vec![first, conflicting];
        expected.sort_unstable_by_key(CollectionRecord::id);
        let snapshot = repo.snapshot().unwrap();
        assert_eq!(snapshot.select_records(&exact).unwrap(), expected);

        let grouped = [
            CollectionRecordSelector::MergeCollection(source),
            CollectionRecordSelector::DeriveTarget(target),
        ]
        .into_iter()
        .collect();
        let mut expected = vec![merge, first, conflicting, sibling];
        expected.sort_unstable_by_key(CollectionRecord::id);
        assert_eq!(snapshot.select_records(&grouped).unwrap(), expected);
        assert!(!snapshot
            .select_records(&grouped)
            .unwrap()
            .contains(&unrelated));
    }

    #[test]
    fn valid_collection_commits_and_owned_closure_survive_memory_keep() {
        use ed25519_dalek::SigningKey;

        use crate::blob::encodings::utf8string::UTF8String;
        use crate::collection::{simplearchive_union, CollectionStoreExt};
        use crate::repo::{BlobStoreGet, BlobStoreKeep};

        let mut repo = MemoryRepo::default();
        let child = repo.put::<UTF8String, _>("owned child".to_owned()).unwrap();
        let fragment = entity! { crate::metadata::name: child };
        let name = "owned";
        let key = SigningKey::from_bytes(&[23; 32]);
        let policy = CollectionPolicy::new(
            crate::collection::AdmissionPolicy::direct(key.verifying_key()),
            crate::collection::AdmissionPolicy::direct(key.verifying_key()),
        );
        let descriptor = simplearchive_union::descriptor(name, policy.clone());
        let expected_collection = identity_for_tests(&descriptor);
        let collection: crate::collection::Collection<SimpleArchive> =
            repo.collection(name, policy).unwrap();
        assert_eq!(collection.handle(), expected_collection);
        let commit = repo.commit(collection, &key, fragment).unwrap();
        let orphan = repo.put::<UTF8String, _>("orphan".to_owned()).unwrap();

        repo.keep(std::iter::empty::<Inline<Handle<UnknownBlob>>>());

        let reader = repo.snapshot().unwrap();
        for retained in [
            collection.handle().transmute(),
            Inline::<Handle<UnknownBlob>>::new(commit.data().raw),
            commit.metadata().transmute(),
            child.transmute(),
        ] {
            assert!(reader.get::<Blob<UnknownBlob>, _>(retained).is_ok());
        }
        assert!(reader
            .get::<Blob<UnknownBlob>, _>(orphan.transmute())
            .is_err());
    }

    #[test]
    fn snapshot_changes_track_exactly_the_semantic_sets() {
        use crate::blob::encodings::utf8string::UTF8String;

        let mut repo = MemoryRepo::default();
        let empty = repo.snapshot().unwrap();

        // WANT is local operational policy, not advertised inventory.
        repo.want(WantRequest::blob(handle(1))).unwrap();
        let after_want = repo.snapshot().unwrap();
        assert!(empty == after_want);
        assert_eq!(after_want.changes_since(&empty), StoreChanges::NONE,);

        repo.put::<UTF8String, _>("revision fixture".to_owned())
            .unwrap();
        let after_blob = repo.snapshot().unwrap();
        assert!(after_want != after_blob);
        assert_eq!(after_blob.changes_since(&after_want), StoreChanges::BLOBS,);

        let target = identity_for_tests(&named_for_tests(
            "revision-target",
            Id::new([71; 16]).unwrap(),
        ));
        repo.insert(CollectionRecord::Derive(CollectionDerive::new(
            target,
            handle(73).into(),
            handle(74).into(),
        )))
        .unwrap();
        let after_record = repo.snapshot().unwrap();
        assert!(after_blob != after_record);
        assert_eq!(
            after_record.changes_since(&after_blob),
            StoreChanges::COLLECTION_RECORDS,
        );

        let root = SigningKey::from_bytes(&[75; 32]);
        let leaf = SigningKey::from_bytes(&[76; 32]);
        let claim = CapabilityClaim::root(
            CapabilityAtom::new(
                CapabilityAction::new(Id::new([77; 16]).unwrap()),
                CapabilityResource::new([78; 32]),
            ),
            CapabilityMode::Invoke,
            None,
        );
        let proof = CapabilityProofBundle::issue_root(&root, claim, leaf.verifying_key())
            .unwrap()
            .proof()
            .clone();
        repo.insert_proof(proof).unwrap();
        let after_proof = repo.snapshot().unwrap();
        assert!(after_record != after_proof);
        assert_eq!(
            after_proof.changes_since(&after_record),
            StoreChanges::CAPABILITY_PROOFS,
        );

        repo.insert_peer(PeerEvidence::new(
            root.verifying_key(),
            leaf.verifying_key(),
        ))
        .unwrap();
        let after_peer = repo.snapshot().unwrap();
        assert!(after_proof != after_peer);
        assert_eq!(after_peer.changes_since(&after_proof), StoreChanges::PEERS,);
    }

    #[test]
    fn snapshot_freezes_blob_collection_and_peer_reads_together() {
        use crate::blob::encodings::utf8string::UTF8String;

        let mut repo = MemoryRepo::default();
        let before = repo.snapshot().unwrap();
        let blob = repo
            .put::<UTF8String, _>("after snapshot".to_owned())
            .unwrap();
        let target = identity_for_tests(&named_for_tests(
            "snapshot-target",
            Id::new([81; 16]).unwrap(),
        ));
        let record = CollectionRecord::Derive(CollectionDerive::new(
            target,
            handle(82).into(),
            handle(83).into(),
        ));
        repo.insert(record).unwrap();
        let evidence = PeerEvidence::new(
            SigningKey::from_bytes(&[84; 32]).verifying_key(),
            SigningKey::from_bytes(&[85; 32]).verifying_key(),
        );
        repo.insert_peer(evidence).unwrap();
        let after = repo.snapshot().unwrap();

        assert!(!before.contains_blob(blob).unwrap());
        assert_eq!(before.record(record.id()).unwrap(), None);
        assert!(before.peers().unwrap().next().is_none());

        assert!(after.contains_blob(blob).unwrap());
        assert_eq!(after.record(record.id()).unwrap(), Some(record));
        assert_eq!(
            after
                .peers()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![evidence],
        );
    }
}
