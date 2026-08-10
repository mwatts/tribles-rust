//! Narrow owned facade for one scoped collection.
//!
//! [`Collection`] owns the storage, canonical `SimpleArchive`-union
//! definition, and signing key needed to publish [`Fragment`] values and read
//! the complete known union authorized by that same key. It is not a
//! repository abstraction: it has no head, branch, CAS, retry, read-admission,
//! or planning policy.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::SigningKey;

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::Blob;
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::repo::{
    BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut, StorageClose, StorageFlush,
};
use crate::trible::{Fragment, TribleSet};

use super::simplearchive_union::{
    self, MaterializationError, PublicationError, SimpleArchiveUnionValidationError,
};
use super::{
    discover_collection_records, resolve_collection_semantics, CollectionClaimValidation,
    CollectionCommit, CollectionData, CollectionDefinition, CollectionDiscoveryError,
    CollectionFunctionalConflict, CollectionResolutionError, CollectionStore,
    CollectionValidationRequest,
};

/// A scoped `SimpleArchive`-union collection and its signing authority.
///
/// Construction is pure with respect to `storage`: the canonical definition
/// is derived in memory and is not inserted until a [`commit`](Self::commit)
/// publication begins.
pub struct Collection<S> {
    storage: S,
    definition: CollectionDefinition,
    signing_key: SigningKey,
}

/// Failure to materialize the complete known value of an owned collection.
///
/// Strictly verified commits by this collection's own public key are ground
/// truth, so their definition, data, and metadata fail loud. A record with an
/// invalid signature authenticates none of its fields and is ignored as an
/// inert discovery diagnostic. Unsigned equations are only replaceable cache
/// evidence: missing or invalid equations are omitted from the resolved
/// semantics and cannot hide a valid committed leaf.
#[derive(Debug)]
pub enum CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError> {
    /// Native collection-record discovery did not complete.
    Discovery(CollectionDiscoveryError<RecordsError>),
    /// At least one own commit was observed, but its canonical collection
    /// definition was absent from the same record view.
    MissingDefinition {
        /// Intrinsic collection definition id.
        collection: Id,
    },
    /// The blob reader could not be created after record discovery.
    Reader(ReaderError),
    /// An own commit's data blob could not be fetched.
    CommitDataGet {
        /// Intrinsic commit record id.
        commit: Id,
        /// Claimed data identity.
        data: CollectionData,
        /// Backend fetch failure.
        source: GetError,
    },
    /// An own commit's data failed exact `SimpleArchive` collection
    /// validation.
    InvalidCommitData {
        /// Intrinsic commit record id.
        commit: Id,
        /// Exact representation or identity diagnostic.
        source: SimpleArchiveUnionValidationError,
    },
    /// An own commit's mandatory metadata archive could not be fetched.
    CommitMetadataGet {
        /// Intrinsic commit record id.
        commit: Id,
        /// Mandatory metadata archive handle.
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Backend fetch failure.
        source: GetError,
    },
    /// An own commit's mandatory metadata was not a canonical
    /// `SimpleArchive`.
    InvalidCommitMetadata {
        /// Intrinsic commit record id.
        commit: Id,
        /// Mandatory metadata archive handle.
        metadata: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Canonical archive failure.
        source: UnarchiveError,
    },
    /// An own commit's canonical metadata bytes did not have the exact
    /// identity signed by the commit.
    InvalidCommitMetadataIdentity {
        /// Intrinsic commit record id.
        commit: Id,
        /// Signed metadata archive handle.
        expected: crate::inline::Inline<Handle<SimpleArchive>>,
        /// Blake3 handle recomputed from the returned bytes.
        actual: crate::inline::Inline<Handle<SimpleArchive>>,
    },
    /// Positively validated equations contradicted operation functionality.
    ResolutionConflict(Box<CollectionFunctionalConflict>),
    /// The resolved semantic frontier could not be physically materialized.
    Materialize(MaterializationError<MetaError, GetError>),
}

impl<RecordsError, ReaderError, MetaError, GetError> fmt::Display
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
where
    RecordsError: fmt::Display,
    ReaderError: fmt::Display,
    MetaError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(source) => source.fmt(f),
            Self::MissingDefinition { collection } => write!(
                f,
                "owned collection {collection:X} has commits but no canonical definition"
            ),
            Self::Reader(source) => write!(f, "failed to open collection blob view: {source}"),
            Self::CommitDataGet {
                commit,
                data,
                source,
            } => write!(
                f,
                "failed to fetch data {} for owned commit {commit:X}: {source}",
                hex::encode_upper(data.raw),
            ),
            Self::InvalidCommitData { commit, source } => {
                write!(f, "owned commit {commit:X} has invalid data: {source}")
            }
            Self::CommitMetadataGet {
                commit,
                metadata,
                source,
            } => write!(
                f,
                "failed to fetch metadata {} for owned commit {commit:X}: {source}",
                hex::encode_upper(metadata.raw),
            ),
            Self::InvalidCommitMetadata {
                commit,
                metadata,
                source,
            } => write!(
                f,
                "owned commit {commit:X} has invalid metadata {}: {source}",
                hex::encode_upper(metadata.raw),
            ),
            Self::InvalidCommitMetadataIdentity {
                commit,
                expected,
                actual,
            } => write!(
                f,
                "owned commit {commit:X} metadata bytes hash to {} instead of signed {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::ResolutionConflict(source) => source.fmt(f),
            Self::Materialize(source) => source.fmt(f),
        }
    }
}

impl<RecordsError, ReaderError, MetaError, GetError> Error
    for CollectionMaterializationError<RecordsError, ReaderError, MetaError, GetError>
where
    RecordsError: Error + 'static,
    ReaderError: Error + 'static,
    MetaError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Discovery(source) => Some(source),
            Self::MissingDefinition { .. } => None,
            Self::Reader(source) => Some(source),
            Self::CommitDataGet { source, .. } => Some(source),
            Self::InvalidCommitData { source, .. } => Some(source),
            Self::CommitMetadataGet { source, .. } => Some(source),
            Self::InvalidCommitMetadata { source, .. } => Some(source),
            Self::InvalidCommitMetadataIdentity { .. } => None,
            Self::ResolutionConflict(source) => Some(source),
            Self::Materialize(source) => Some(source),
        }
    }
}

impl<S> Collection<S> {
    /// Construct a write facade without reading from or writing to `storage`.
    pub fn new(storage: S, scope: Id, signing_key: SigningKey) -> Self {
        Self {
            storage,
            definition: simplearchive_union::definition(scope),
            signing_key,
        }
    }

    /// Canonical collection definition derived from the constructor scope.
    pub fn definition(&self) -> &CollectionDefinition {
        &self.definition
    }

    /// Borrow the underlying storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Mutably borrow the underlying storage.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Consume the facade and recover its underlying storage.
    pub fn into_storage(self) -> S {
        self.storage
    }
}

impl<S> Collection<S>
where
    S: BlobStorePut + CollectionStore + StorageFlush,
{
    /// Publish one self-contained fragment as an independent signed commit.
    ///
    /// Facts are the collection element, metafacts are commit metadata, and
    /// fragment attachments are staged through the same crash-ordered,
    /// content-addressed path as
    /// [`simplearchive_union::publish_fragment_commit`]. Repeating identical
    /// input is idempotent; distinct commits coexist without selecting a head.
    /// The parameter is deliberately `Fragment`, rather than `Into<Fragment>`,
    /// so a bare fact set cannot accidentally publish without its metafacts.
    pub fn commit(
        &mut self,
        fragment: Fragment,
    ) -> Result<
        CollectionCommit,
        PublicationError<S::PutError, S::InsertError, <S as StorageFlush>::Error>,
    > {
        simplearchive_union::publish_fragment_commit(
            &mut self.storage,
            &self.definition,
            fragment,
            &self.signing_key,
        )
    }
}

impl<S> Collection<S>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    /// Materialize the complete known `TribleSet` authorized by this facade's
    /// signing identity.
    ///
    /// One call first discovers a deterministic observed view of native
    /// collection records, then opens a blob-reader snapshot. Every strictly
    /// signed commit in that record view which names this exact collection and
    /// this facade's public key is mandatory membership; commits from foreign
    /// keys are ignored. All own commit dependencies are exact-validated and
    /// fail loud. Exact resident merge equations may replace redundant leaves
    /// in the physical cover, while missing or invalid unsigned equations are
    /// treated as cache misses and the committed leaves remain authoritative.
    /// Derivations are not admitted by this `SimpleArchive`-only facade.
    ///
    /// This is a **known-prefix** read, not a global-latest transaction:
    /// [`CollectionStore`] does not promise a coherent snapshot under
    /// concurrent insertion. A commit first observed after this discovery pass
    /// appears on a later call. The returned set is nevertheless complete for
    /// all own commits observed by this pass, or the call returns an error
    /// instead of a partial set. If no own commit is observed, the result is
    /// empty even when the collection definition is absent.
    pub fn materialize(
        &mut self,
    ) -> Result<
        TribleSet,
        CollectionMaterializationError<
            S::RecordsError,
            S::ReaderError,
            <S::Reader as BlobStoreMeta>::MetaError,
            <S::Reader as BlobStoreGet>::GetError<Infallible>,
        >,
    > {
        let discovered = discover_collection_records(&mut self.storage)
            .map_err(CollectionMaterializationError::Discovery)?;
        let collection = self.definition.id();
        let public_key = self.signing_key.verifying_key().to_bytes();

        let authorized: BTreeSet<_> = discovered
            .commits()
            .iter()
            .filter(|commit| {
                commit.collection() == collection && commit.public_key().raw == public_key
            })
            .map(CollectionCommit::id)
            .collect();

        if authorized.is_empty() {
            return Ok(TribleSet::new());
        }
        if !discovered.definitions().contains(&self.definition) {
            return Err(CollectionMaterializationError::MissingDefinition { collection });
        }

        let reader = self
            .storage
            .reader()
            .map_err(CollectionMaterializationError::Reader)?;

        // Authenticate and exact-validate every mandatory leaf first. Besides
        // enforcing the signed boundary, this cache means an unsigned merge
        // never causes the same large endpoint to be fetched and scanned over
        // and over in one materialization.
        let mut known = BTreeMap::<CollectionData, Blob<SimpleArchive>>::new();
        for claim in discovered
            .commits()
            .iter()
            .filter(|claim| authorized.contains(&claim.id()))
        {
            let data = claim.data();
            let data_blob: Blob<SimpleArchive> = reader
                .get(Handle::<SimpleArchive>::from_hash(data))
                .map_err(|source| CollectionMaterializationError::CommitDataGet {
                    commit: claim.id(),
                    data,
                    source,
                })?;
            simplearchive_union::validate_commit(&self.definition, claim, &data_blob).map_err(
                |source| CollectionMaterializationError::InvalidCommitData {
                    commit: claim.id(),
                    source,
                },
            )?;

            let metadata = claim.metadata();
            let metadata_blob: Blob<SimpleArchive> = reader.get(metadata).map_err(|source| {
                CollectionMaterializationError::CommitMetadataGet {
                    commit: claim.id(),
                    metadata,
                    source,
                }
            })?;
            simplearchive_union::validate_element(&metadata_blob).map_err(|source| {
                CollectionMaterializationError::InvalidCommitMetadata {
                    commit: claim.id(),
                    metadata,
                    source,
                }
            })?;
            let actual_metadata =
                Blob::<SimpleArchive>::new(metadata_blob.bytes.clone()).get_handle();
            if actual_metadata != metadata {
                return Err(
                    CollectionMaterializationError::InvalidCommitMetadataIdentity {
                        commit: claim.id(),
                        expected: metadata,
                        actual: actual_metadata,
                    },
                );
            }
            known.entry(data).or_insert(data_blob);
        }

        // Grow only through merge equations grounded in authenticated leaves.
        // The exact join for one input pair is cached and compared before any
        // speculative result fetch, so arbitrary result hashes neither mint
        // Yard wants nor trigger repeated large-input scans. A result is
        // admitted only when it is already resident and byte-for-byte equal to
        // that canonical join; all other unsigned evidence remains inert.
        let mut accepted_merges = BTreeSet::new();
        let mut examined_merges = BTreeSet::new();
        let mut joins = BTreeMap::<(CollectionData, CollectionData), Blob<SimpleArchive>>::new();
        loop {
            let mut changed = false;
            for claim in discovered
                .merges()
                .iter()
                .filter(|claim| claim.collection() == collection)
            {
                if examined_merges.contains(&claim.id()) {
                    continue;
                }
                let (low, high) = claim.inputs();
                let (Some(low_blob), Some(high_blob)) = (known.get(&low), known.get(&high)) else {
                    continue;
                };

                let expected = match joins.entry((low, high)) {
                    std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        let Ok(joined) = simplearchive_union::join(low_blob, high_blob) else {
                            // `known` contains only exact-validated canonical
                            // elements, so this is a defensive invariant guard.
                            examined_merges.insert(claim.id());
                            continue;
                        };
                        entry.insert(joined)
                    }
                };
                let expected_data = Handle::<SimpleArchive>::to_hash(expected.get_handle());
                if claim.result() != expected_data {
                    examined_merges.insert(claim.id());
                    continue;
                }

                if !known.contains_key(&expected_data) {
                    let result_handle = Handle::<SimpleArchive>::from_hash(expected_data);
                    if !matches!(reader.metadata(result_handle), Ok(Some(_))) {
                        examined_merges.insert(claim.id());
                        continue;
                    }
                    let Ok(result): Result<Blob<SimpleArchive>, _> = reader.get(result_handle)
                    else {
                        examined_merges.insert(claim.id());
                        continue;
                    };
                    if result.bytes != expected.bytes {
                        examined_merges.insert(claim.id());
                        continue;
                    }
                    known.insert(
                        expected_data,
                        Blob::with_handle(expected.bytes.clone(), expected.get_handle()),
                    );
                }

                accepted_merges.insert(claim.id());
                examined_merges.insert(claim.id());
                changed = true;
            }
            if !changed {
                break;
            }
        }

        let resolution = resolve_collection_semantics(&discovered, &authorized, |request| {
            Ok::<CollectionClaimValidation<()>, Infallible>(match request {
                CollectionValidationRequest::Commit { .. } => CollectionClaimValidation::Accepted,
                CollectionValidationRequest::Merge { claim, .. }
                    if accepted_merges.contains(&claim.id()) =>
                {
                    CollectionClaimValidation::Accepted
                }
                CollectionValidationRequest::Merge { .. }
                | CollectionValidationRequest::Derive { .. } => CollectionClaimValidation::Pending,
            })
        });

        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(CollectionResolutionError::Validation { source, .. }) => match source {},
            Err(CollectionResolutionError::Conflict(source)) => {
                return Err(CollectionMaterializationError::ResolutionConflict(source));
            }
        };

        simplearchive_union::materialize(resolution.semantics(), &self.definition, &reader)
            .map_err(CollectionMaterializationError::Materialize)
    }
}

impl<S> Collection<S>
where
    S: StorageClose,
{
    /// Consume the facade and close its underlying storage.
    pub fn close(self) -> Result<(), S::Error> {
        self.storage.close()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    use crate::blob::encodings::{longstring::LongString, UnknownBlob};
    use crate::blob::{Bytes, IntoBlob};
    use crate::collection::{discover_collection_records, CollectionMerge, CollectionRecord};
    use crate::inline::encodings::hash::Handle;
    use crate::inline::Inline;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn fragment(entity: u8, attachment: bool) -> Fragment {
        let mut row = [entity; TRIBLE_LEN];
        row[16..32].fill(1);
        let mut facts = TribleSet::new();
        facts.insert(&Trible::force_raw(row).unwrap());
        let mut fragment = Fragment::from(facts);
        if attachment {
            let _: Inline<Handle<LongString>> = fragment.put("one attachment".to_owned());
        }
        fragment
    }

    fn archive(entity: u8) -> Blob<SimpleArchive> {
        fragment(entity, false).facts().clone().to_blob()
    }

    fn invalid_signature(commit: CollectionCommit) -> CollectionCommit {
        let (r, mut s) = commit.signature();
        s.raw[0] ^= 1;
        CollectionCommit::from_parts(
            commit.collection(),
            commit.data(),
            commit.metadata(),
            commit.public_key(),
            r,
            s,
        )
    }

    #[test]
    fn construction_is_pure_and_derives_the_scoped_definition() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);

        assert_eq!(
            collection.definition(),
            &simplearchive_union::definition(scope)
        );
        assert!(collection.storage().blobs.is_empty());
        assert!(collection.storage_mut().records().unwrap().next().is_none());

        collection.close().unwrap();
    }

    #[test]
    fn borrowed_facade_reuses_one_open_storage() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut storage = MemoryRepo::default();

        {
            let mut collection = Collection::new(&mut storage, scope, signing_key.clone());
            collection.commit(expected.clone()).unwrap();
            assert_eq!(collection.materialize().unwrap(), expected.facts().clone());
        }

        let discovered = discover_collection_records(&mut storage).unwrap();
        assert_eq!(discovered.commits().len(), 1);

        let mut collection = Collection::new(&mut storage, scope, signing_key);
        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn distinct_commits_coexist_and_repeated_commits_are_idempotent() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);
        let first_fragment = fragment(1, true);
        let second_fragment = fragment(2, false);

        let first = collection.commit(first_fragment.clone()).unwrap();
        let after_first = collection.storage().blobs.len();
        let second = collection.commit(second_fragment).unwrap();
        let after_second = collection.storage().blobs.len();
        let repeated = collection.commit(first_fragment).unwrap();

        assert_eq!(repeated, first);
        assert_ne!(second, first);
        assert!(after_second > after_first);
        assert_eq!(collection.storage().blobs.len(), after_second);

        let definition = *collection.definition();
        let discovered = discover_collection_records(collection.storage_mut()).unwrap();
        assert_eq!(discovered.definitions(), &[definition]);
        assert_eq!(
            discovered
                .commits()
                .iter()
                .map(CollectionCommit::id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([first.id(), second.id()])
        );
        assert_eq!(
            collection
                .storage_mut()
                .records()
                .unwrap()
                .collect::<Result<Vec<CollectionRecord>, _>>()
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn empty_owned_collection_materializes_without_a_definition() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );

        assert_eq!(collection.materialize().unwrap(), TribleSet::new());
    }

    #[test]
    fn own_commits_materialize_completely_and_repeat_deterministically() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let first = fragment(1, false);
        let second = fragment(2, false);
        let mut expected = first.facts().clone();
        expected += second.facts().clone();

        collection.commit(first).unwrap();
        collection.commit(second).unwrap();

        let observed = collection.materialize().unwrap();
        assert_eq!(observed, expected);
        assert_eq!(collection.materialize().unwrap(), observed);
    }

    #[test]
    fn commits_from_a_foreign_signer_are_ignored() {
        let own_key = SigningKey::from_bytes(&[7; 32]);
        let foreign_key = SigningKey::from_bytes(&[8; 32]);
        let mut collection = Collection::new(MemoryRepo::default(), id(1), own_key);
        let definition = *collection.definition();
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();

        let foreign = simplearchive_union::publish_commit(
            collection.storage_mut(),
            &definition,
            &data,
            &metadata,
            &foreign_key,
        )
        .unwrap();
        collection
            .storage_mut()
            .insert(CollectionRecord::Commit(invalid_signature(foreign)))
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), TribleSet::new());
    }

    #[test]
    fn own_commit_without_its_definition_fails_loud() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let definition = simplearchive_union::definition(scope);
        let data = archive(1);
        let metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &signing_key,
            definition.id(),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata.get_handle(),
        );
        let mut storage = MemoryRepo::default();
        storage.blobs.insert(data);
        storage.blobs.insert(metadata);
        storage.insert(CollectionRecord::Commit(commit)).unwrap();
        let mut collection = Collection::new(storage, scope, signing_key);

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::MissingDefinition { collection })
                if collection == definition.id()
        ));
    }

    #[test]
    fn invalid_signature_claiming_the_owned_key_is_inert() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let expected = fragment(1, false);
        let mut collection = Collection::new(MemoryRepo::default(), scope, signing_key);
        let valid = collection.commit(expected.clone()).unwrap();
        let invalid = invalid_signature(valid);
        collection
            .storage_mut()
            .insert(CollectionRecord::Commit(invalid))
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected.into_facts());
    }

    #[test]
    fn missing_or_corrupt_owned_data_fails_loud() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut missing = Collection::new(MemoryRepo::default(), scope, signing_key.clone());
        let missing_commit = missing.commit(fragment(1, false)).unwrap();
        missing
            .storage_mut()
            .blobs
            .keep([missing_commit.metadata().transmute::<Handle<UnknownBlob>>()]);
        assert!(matches!(
            missing.materialize(),
            Err(CollectionMaterializationError::CommitDataGet { commit, .. })
                if commit == missing_commit.id()
        ));

        let mut corrupt = Collection::new(MemoryRepo::default(), scope, signing_key);
        let corrupt_commit = corrupt.commit(fragment(1, false)).unwrap();
        corrupt
            .storage_mut()
            .blobs
            .keep([corrupt_commit.metadata().transmute::<Handle<UnknownBlob>>()]);
        let claimed = Handle::<SimpleArchive>::from_hash(corrupt_commit.data());
        corrupt
            .storage_mut()
            .blobs
            .insert(Blob::with_handle(Bytes::from(vec![0; 63]), claimed));
        assert!(matches!(
            corrupt.materialize(),
            Err(CollectionMaterializationError::InvalidCommitData { commit, .. })
                if commit == corrupt_commit.id()
        ));
    }

    #[test]
    fn missing_or_corrupt_owned_metadata_fails_loud() {
        let scope = id(1);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let mut missing = Collection::new(MemoryRepo::default(), scope, signing_key.clone());
        let missing_commit = missing.commit(fragment(1, false)).unwrap();
        missing
            .storage_mut()
            .blobs
            .keep([Handle::<SimpleArchive>::from_hash(missing_commit.data())
                .transmute::<Handle<UnknownBlob>>()]);
        assert!(matches!(
            missing.materialize(),
            Err(CollectionMaterializationError::CommitMetadataGet { commit, .. })
                if commit == missing_commit.id()
        ));

        let mut corrupt = Collection::new(MemoryRepo::default(), scope, signing_key);
        let corrupt_commit = corrupt.commit(fragment(1, false)).unwrap();
        corrupt
            .storage_mut()
            .blobs
            .keep([Handle::<SimpleArchive>::from_hash(corrupt_commit.data())
                .transmute::<Handle<UnknownBlob>>()]);
        corrupt.storage_mut().blobs.insert(Blob::with_handle(
            Bytes::from(vec![0; 63]),
            corrupt_commit.metadata(),
        ));
        assert!(matches!(
            corrupt.materialize(),
            Err(CollectionMaterializationError::InvalidCommitMetadata { commit, .. })
                if commit == corrupt_commit.id()
        ));
    }

    #[test]
    fn canonical_owned_metadata_must_match_the_signed_handle() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let commit = collection.commit(fragment(1, false)).unwrap();
        let wrong_metadata = archive(9);
        collection
            .storage_mut()
            .blobs
            .keep([Handle::<SimpleArchive>::from_hash(commit.data())
                .transmute::<Handle<UnknownBlob>>()]);
        collection
            .storage_mut()
            .blobs
            .insert(Blob::with_handle(wrong_metadata.bytes, commit.metadata()));

        assert!(matches!(
            collection.materialize(),
            Err(CollectionMaterializationError::InvalidCommitMetadataIdentity {
                commit: observed,
                expected,
                ..
            }) if observed == commit.id() && expected == commit.metadata()
        ));
    }

    #[test]
    fn valid_merge_cover_materializes_the_committed_union() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let left = left_fragment.facts().clone().to_blob();
        let right = right_fragment.facts().clone().to_blob();
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        collection.commit(left_fragment).unwrap();
        collection.commit(right_fragment).unwrap();
        let definition = *collection.definition();

        simplearchive_union::publish_merge(collection.storage_mut(), &definition, &left, &right)
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn grounded_merge_chain_materializes_the_committed_union() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let first = fragment(1, false);
        let second = fragment(2, false);
        let third = fragment(3, false);
        let first_blob = first.facts().clone().to_blob();
        let second_blob = second.facts().clone().to_blob();
        let third_blob = third.facts().clone().to_blob();
        let mut expected = first.facts().clone();
        expected += second.facts().clone();
        expected += third.facts().clone();
        collection.commit(first).unwrap();
        collection.commit(second).unwrap();
        collection.commit(third).unwrap();
        let definition = *collection.definition();

        let (_, first_two) = simplearchive_union::publish_merge(
            collection.storage_mut(),
            &definition,
            &first_blob,
            &second_blob,
        )
        .unwrap();
        simplearchive_union::publish_merge(
            collection.storage_mut(),
            &definition,
            &first_two,
            &third_blob,
        )
        .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected);
    }

    #[test]
    fn broken_unsigned_merge_falls_back_to_committed_leaves() {
        let mut collection = Collection::new(
            MemoryRepo::default(),
            id(1),
            SigningKey::from_bytes(&[7; 32]),
        );
        let left_fragment = fragment(1, false);
        let right_fragment = fragment(2, false);
        let mut expected = left_fragment.facts().clone();
        expected += right_fragment.facts().clone();
        let left = collection.commit(left_fragment).unwrap();
        let right = collection.commit(right_fragment).unwrap();
        let broken = CollectionMerge::new(
            collection.definition().id(),
            left.data(),
            right.data(),
            left.data(),
        );
        collection
            .storage_mut()
            .insert(CollectionRecord::Merge(broken))
            .unwrap();

        assert_eq!(collection.materialize().unwrap(), expected);
    }
}
