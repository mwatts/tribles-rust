//! Canonical TribleSet set union over
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive)
//! elements.
//!
//! This is the first concrete production collection kind. A collection pairs
//! an extrinsic scope with the existing `SimpleArchive` representation and the
//! [`TRIBLE_SET_UNION_RECIPE_V1`](crate::collection::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1)
//! semantic recipe. Every element is an exact, canonical EAV-ordered stream of
//! 64-byte tribles. Its join is ordinary set union, so canonical output bytes
//! and their Blake3 identity are associative, commutative, and idempotent.
//!
//! Validation, joins, and publication operate directly on the canonical byte
//! streams. They deliberately do not construct [`crate::trible::TribleSet`] or
//! PATCH indexes; query-time decoding keeps its independently optimized path.
//! Missing endpoint blobs are likewise outside this module: callers defer an
//! equation until its three blobs are resident, then call
//! [`validate_merge`](crate::collection::simplearchive_union::validate_merge).

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use anybytes::{Bytes, View};
use ed25519_dalek::SigningKey;

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, MemoryBlobStore};
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::Inline;
use crate::metadata::MetaDescribe;
use crate::repo::{BlobStore, BlobStorePut};
use crate::trible::{Fragment, Trible, TRIBLE_LEN};

use super::{
    CollectionCommit, CollectionData, CollectionDescriptor, CollectionId, CollectionMerge,
    CollectionRecord, CollectionStore,
};

mod materialize;
pub use materialize::*;

/// Canonical TribleSet set-union recipe, version 1.
///
/// This identifies the semantic law independently of its direct-stream
/// implementation and of the collection's blob representation. Minted with
/// `trible genid` on 2026-08-07.
pub const TRIBLE_SET_UNION_RECIPE_V1: Id = id_hex!("6D64C5F4B9E9B73F57C5F8702AB7FE45");

/// The collection endpoint involved in a validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementRole {
    /// Data introduced by a signed commit.
    CommitData,
    /// Canonically lower merge input.
    MergeLow,
    /// Canonically higher merge input.
    MergeHigh,
    /// Claimed merge output.
    MergeResult,
}

impl fmt::Display for ElementRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommitData => write!(f, "commit data"),
            Self::MergeLow => write!(f, "merge low input"),
            Self::MergeHigh => write!(f, "merge high input"),
            Self::MergeResult => write!(f, "merge result"),
        }
    }
}

/// Failure to validate a commit or merge against this concrete collection kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SimpleArchiveUnionValidationError {
    /// The descriptor names another blob representation.
    WrongRepresentation { expected: Id, actual: Id },
    /// The descriptor names another semantic recipe.
    WrongRecipe { expected: Id, actual: Id },
    /// The record belongs to another collection descriptor.
    WrongCollection {
        expected: CollectionId,
        actual: CollectionId,
    },
    /// Supplied bytes do not have the content identity named by the record.
    EndpointMismatch {
        role: ElementRole,
        expected: CollectionData,
        actual: CollectionData,
    },
    /// An endpoint is not a canonical `SimpleArchive` element.
    InvalidElement {
        role: ElementRole,
        source: UnarchiveError,
    },
    /// The claimed result is not the exact canonical union of the two inputs.
    WrongMergeResult,
}

impl fmt::Display for SimpleArchiveUnionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRepresentation { expected, actual } => write!(
                f,
                "collection representation {actual:X} does not match SimpleArchive {expected:X}"
            ),
            Self::WrongRecipe { expected, actual } => write!(
                f,
                "collection recipe {actual:X} does not match TribleSet union {expected:X}"
            ),
            Self::WrongCollection { expected, actual } => write!(
                f,
                "record collection {} does not match descriptor {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::EndpointMismatch {
                role,
                expected,
                actual,
            } => write!(
                f,
                "{role} handle {} does not match claimed {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::InvalidElement { role, source } => {
                write!(f, "{role} is not a canonical SimpleArchive: {source}")
            }
            Self::WrongMergeResult => {
                write!(f, "merge result is not the exact canonical input union")
            }
        }
    }
}

impl Error for SimpleArchiveUnionValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidElement { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Failure to publish an ordered collection record.
///
/// Dependencies are written before the native record, but publication does
/// not imply crash durability. Callers that need a durability boundary choose
/// when to invoke their store's explicit flush operation. Failed I/O may still
/// require backend-specific recovery before retrying; once the store is usable
/// again, replaying the same logical publication is content-addressed and
/// deterministic.
#[derive(Debug)]
pub enum PublicationError<PutError, InsertError> {
    /// The descriptor or collection data is invalid for this concrete kind.
    Validation(SimpleArchiveUnionValidationError),
    /// Commit metadata is not a canonical `SimpleArchive`.
    InvalidMetadata(UnarchiveError),
    /// An embedded fragment blob is not stored under its actual byte identity.
    InvalidEmbeddedBlob {
        /// Type-erased identity used as the `MemoryBlobStore` key.
        store_key: CollectionData,
        /// Type-erased identity cached inside the stored `Blob`.
        cached_handle: CollectionData,
        /// Fresh Blake3 identity recomputed from the blob bytes.
        actual: CollectionData,
    },
    /// An element, result, metadata, or embedded-attachment write failed.
    DependencyPut(PutError),
    /// The final commit or merge record could not be admitted to the native record store.
    RecordInsert(InsertError),
}

/// Validation failure while preparing a canonical collection commit in memory.
///
/// The uninhabited storage error parameters make it impossible for this phase
/// to report an I/O failure: preparation does not touch the destination store.
pub type PreparationError = PublicationError<Infallible, Infallible>;

/// A canonical signed collection commit whose bytes have not been published.
///
/// Preparation validates and normalizes the descriptor, data, metadata, and
/// embedded fragment blobs entirely in memory. Call [`Self::stage`] to write
/// every dependency while retaining the commit record itself. Dropping a
/// prepared value has no storage effect.
#[derive(Clone, Debug)]
#[must_use = "a prepared collection commit has no effect until it is staged and finalized"]
pub struct PreparedCollectionCommit {
    embedded: Vec<Blob<UnknownBlob>>,
    descriptor: CollectionDescriptor,
    data: Blob<SimpleArchive>,
    metadata: Blob<SimpleArchive>,
    commit: CollectionCommit,
}

impl PreparedCollectionCommit {
    /// Inspect the exact canonical commit that finalization will publish.
    pub fn commit(&self) -> &CollectionCommit {
        &self.commit
    }

    /// Stage every dependency without publishing the commit record.
    ///
    /// The exact store-call order is the collection descriptor blob,
    /// embedded blobs (in handle order), data, and metadata.
    /// On success the returned value retains the same mutable store borrow, so
    /// a caller may append unsigned `MERGE` or `DERIVE` artifacts through
    /// [`StagedCollectionCommit::store_mut`] before consuming the value with
    /// [`StagedCollectionCommit::finalize`].
    pub fn stage<'store, S>(
        self,
        store: &'store mut S,
    ) -> Result<StagedCollectionCommit<'store, S>, PublicationError<S::PutError, S::InsertError>>
    where
        S: BlobStorePut + CollectionStore,
    {
        let Self {
            embedded,
            descriptor,
            data,
            metadata,
            commit,
        } = self;

        store
            .put::<SimpleArchive, _>(descriptor.to_blob())
            .map_err(PublicationError::DependencyPut)?;
        for blob in embedded {
            store
                .put::<UnknownBlob, _>(blob)
                .map_err(PublicationError::DependencyPut)?;
        }
        store
            .put::<SimpleArchive, _>(data)
            .map_err(PublicationError::DependencyPut)?;
        store
            .put::<SimpleArchive, _>(metadata)
            .map_err(PublicationError::DependencyPut)?;
        Ok(StagedCollectionCommit { store, commit })
    }
}

/// A canonical commit whose complete dependency set has been written first.
///
/// This type holds the exact store borrow used for staging. Its
/// [`store_mut`](Self::store_mut) escape hatch exists so reproducible unsigned
/// equations and their artifacts can be appended before the source membership
/// root becomes visible. Only consuming [`finalize`](Self::finalize) appends
/// the signed `COMMIT` record. Drop is deliberately inert and never
/// auto-finalizes.
#[must_use = "dropping a staged collection commit leaves its dependencies inert; call finalize to publish it"]
pub struct StagedCollectionCommit<'store, S>
where
    S: BlobStorePut + CollectionStore,
{
    store: &'store mut S,
    commit: CollectionCommit,
}

impl<'store, S> StagedCollectionCommit<'store, S>
where
    S: BlobStorePut + CollectionStore,
{
    /// Inspect the exact commit that remains withheld from the store.
    pub fn commit(&self) -> &CollectionCommit {
        &self.commit
    }

    /// Borrow the staged publication's destination for intervening artifacts.
    ///
    /// Writes performed here occur after the dependency writes and before the
    /// final commit append. The caller remains responsible for the validity and
    /// dependency ordering of any unsigned records it writes.
    pub fn store_mut(&mut self) -> &mut S {
        self.store
    }

    /// Append the canonical signed commit last.
    ///
    /// This is the sole visibility boundary. If the insert fails,
    /// backend-specific recovery may be required before deterministic replay.
    /// Durability remains an explicit caller-selected store operation.
    pub fn finalize(
        self,
    ) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>> {
        let Self { store, commit } = self;
        store
            .insert(CollectionRecord::Commit(commit))
            .map_err(PublicationError::RecordInsert)?;
        Ok(commit)
    }
}

impl<PutError, InsertError> fmt::Display for PublicationError<PutError, InsertError>
where
    PutError: fmt::Display,
    InsertError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => write!(f, "invalid collection publication: {error}"),
            Self::InvalidMetadata(error) => {
                write!(
                    f,
                    "commit metadata is not a canonical SimpleArchive: {error}"
                )
            }
            Self::InvalidEmbeddedBlob {
                store_key,
                cached_handle,
                actual,
            } => write!(
                f,
                "embedded blob store key {} and cached handle {} do not both match byte identity {}",
                hex::encode_upper(store_key.raw),
                hex::encode_upper(cached_handle.raw),
                hex::encode_upper(actual.raw),
            ),
            Self::DependencyPut(error) => {
                write!(f, "failed to write a collection dependency: {error}")
            }
            Self::RecordInsert(error) => write!(f, "failed to insert collection record: {error}"),
        }
    }
}

impl<PutError, InsertError> Error for PublicationError<PutError, InsertError>
where
    PutError: Error + 'static,
    InsertError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            Self::InvalidMetadata(error) => Some(error),
            Self::InvalidEmbeddedBlob { .. } => None,
            Self::DependencyPut(error) => Some(error),
            Self::RecordInsert(error) => Some(error),
        }
    }
}

/// Construct this collection kind for an extrinsic dataset scope.
pub fn descriptor(scope: Id) -> CollectionDescriptor {
    CollectionDescriptor::new(
        scope,
        <SimpleArchive as MetaDescribe>::id(),
        TRIBLE_SET_UNION_RECIPE_V1,
    )
}

/// Validate one canonical `SimpleArchive` collection element without decoding
/// it into query indexes.
pub fn validate_element(blob: &Blob<SimpleArchive>) -> Result<(), UnarchiveError> {
    canonical_rows(blob).map(|_| ())
}

/// Compute the exact canonical union of two `SimpleArchive` elements.
///
/// Both inputs are validated before an identity fast path or output allocation
/// is taken. Equal and empty inputs reuse their immutable bytes but recompute
/// the returned handle; every other case performs one lexicographic two-pointer
/// merge and emits shared rows once.
pub fn join(
    left: &Blob<SimpleArchive>,
    right: &Blob<SimpleArchive>,
) -> Result<Blob<SimpleArchive>, UnarchiveError> {
    let left_rows = canonical_rows(left)?;
    let right_rows = canonical_rows(right)?;
    Ok(join_canonical_rows(left, right, &left_rows, &right_rows))
}

/// Validate a discovered commit as one canonical root of this collection.
///
/// This binds the concrete descriptor, record collection, endpoint identity,
/// and element bytes in one check. The record's strict self-signature and the
/// caller's authorization policy remain separate admission prerequisites.
pub fn validate_commit(
    descriptor: &CollectionDescriptor,
    commit: &CollectionCommit,
    data_blob: &Blob<SimpleArchive>,
) -> Result<(), SimpleArchiveUnionValidationError> {
    validate_descriptor(descriptor)?;
    validate_collection(descriptor, commit.collection())?;
    validate_endpoint(ElementRole::CommitData, commit.data(), data_blob)?;
    Ok(())
}

/// Validate a claimed exact union without materializing another result blob.
///
/// All endpoints are first bound to their record hashes and validated as
/// canonical archives. The expected two-way union is then compared row-for-row
/// with `result`, using constant auxiliary space.
pub fn validate_merge(
    descriptor: &CollectionDescriptor,
    claim: &CollectionMerge,
    low: &Blob<SimpleArchive>,
    high: &Blob<SimpleArchive>,
    result: &Blob<SimpleArchive>,
) -> Result<(), SimpleArchiveUnionValidationError> {
    validate_descriptor(descriptor)?;
    validate_collection(descriptor, claim.collection())?;

    let (expected_low, expected_high) = claim.inputs();
    validate_handle(ElementRole::MergeLow, expected_low, low)?;
    validate_handle(ElementRole::MergeHigh, expected_high, high)?;
    validate_handle(ElementRole::MergeResult, claim.result(), result)?;

    let low_rows = canonical_rows(low).map_err(|source| {
        SimpleArchiveUnionValidationError::InvalidElement {
            role: ElementRole::MergeLow,
            source,
        }
    })?;
    let high_rows = canonical_rows(high).map_err(|source| {
        SimpleArchiveUnionValidationError::InvalidElement {
            role: ElementRole::MergeHigh,
            source,
        }
    })?;
    let result_rows = canonical_rows(result).map_err(|source| {
        SimpleArchiveUnionValidationError::InvalidElement {
            role: ElementRole::MergeResult,
            source,
        }
    })?;

    if !UnionRows::new(&low_rows, &high_rows).eq(result_rows.iter()) {
        return Err(SimpleArchiveUnionValidationError::WrongMergeResult);
    }
    Ok(())
}

/// Prepare a canonical signed membership root entirely in memory.
///
/// Supplied data and metadata are normalized from their bytes before either is
/// validated or included in the signed transcript, so a forged
/// [`Blob::with_handle`] cache cannot enter storage or determine the commit
/// identity. No store is touched. The returned value can be inspected, staged,
/// abandoned inertly, or finalized later.
pub fn prepare_commit(
    descriptor: &CollectionDescriptor,
    data: &Blob<SimpleArchive>,
    metadata: &Blob<SimpleArchive>,
    signing_key: &SigningKey,
) -> Result<PreparedCollectionCommit, PreparationError> {
    prepare_commit_with_embedded(descriptor, data, metadata, signing_key, Vec::new())
}

/// Prepare a self-contained fact fragment as a canonical signed membership root.
///
/// The fragment's facts become collection data and its metafacts become commit
/// metadata. Its one shared blob store may back handles in either set. This
/// boundary recomputes every embedded blob's identity and requires both its
/// [`MemoryBlobStore`] key and cached [`Blob`] handle to match the bytes. A
/// mismatch is rejected rather than normalized because either fact set may
/// name the forged identity. Fragment exports are not serialized. No
/// destination store is touched.
pub fn prepare_fragment_commit(
    descriptor: &CollectionDescriptor,
    fragment: Fragment,
    signing_key: &SigningKey,
) -> Result<PreparedCollectionCommit, PreparationError> {
    let (_, facts, metafacts, blobs) = fragment.into_parts();

    let mut embedded =
        checked_embedded_blobs(blobs).map_err(|(store_key, cached_handle, actual)| {
            PublicationError::InvalidEmbeddedBlob {
                store_key,
                cached_handle,
                actual,
            }
        })?;
    embedded.sort_unstable_by_key(|blob| blob.get_handle().raw);

    let data: Blob<SimpleArchive> = crate::blob::IntoBlob::to_blob(facts);
    let metadata: Blob<SimpleArchive> = crate::blob::IntoBlob::to_blob(metafacts);
    prepare_commit_with_embedded(descriptor, &data, &metadata, signing_key, embedded)
}

fn prepare_commit_with_embedded(
    descriptor: &CollectionDescriptor,
    data: &Blob<SimpleArchive>,
    metadata: &Blob<SimpleArchive>,
    signing_key: &SigningKey,
    embedded: Vec<Blob<UnknownBlob>>,
) -> Result<PreparedCollectionCommit, PreparationError> {
    validate_descriptor(descriptor).map_err(PublicationError::Validation)?;

    let data = normalize_blob(data);
    validate_element(&data).map_err(|source| {
        PublicationError::Validation(SimpleArchiveUnionValidationError::InvalidElement {
            role: ElementRole::CommitData,
            source,
        })
    })?;

    let metadata = normalize_blob(metadata);
    validate_element(&metadata).map_err(PublicationError::InvalidMetadata)?;

    let commit = CollectionCommit::sign(
        signing_key,
        descriptor.handle(),
        normalized_data_identity(&data),
        metadata.get_handle(),
    );

    Ok(PreparedCollectionCommit {
        embedded,
        descriptor: *descriptor,
        data,
        metadata,
        commit,
    })
}

fn widen_preparation_error<PutError, InsertError>(
    error: PreparationError,
) -> PublicationError<PutError, InsertError> {
    match error {
        PublicationError::Validation(error) => PublicationError::Validation(error),
        PublicationError::InvalidMetadata(error) => PublicationError::InvalidMetadata(error),
        PublicationError::InvalidEmbeddedBlob {
            store_key,
            cached_handle,
            actual,
        } => PublicationError::InvalidEmbeddedBlob {
            store_key,
            cached_handle,
            actual,
        },
        PublicationError::DependencyPut(never) => match never {},
        PublicationError::RecordInsert(never) => match never {},
    }
}

/// Publish a signed membership root after writing its dependencies.
///
/// Supplied data and metadata are normalized from their bytes before either is
/// validated or stored, so a forged [`Blob::with_handle`] cache cannot enter
/// storage or the signed transcript. The exact write order is:
///
/// 1. collection-descriptor blob, data blob, metadata blob;
/// 2. signed commit record.
///
/// A completed prefix before the record write leaves only content-addressed
/// dependencies. This function deliberately performs no durability flush;
/// callers may group any number of publications behind one explicit barrier or
/// rely on store close. Failed backend I/O may require recovery according to
/// that backend's contract; after recovery, replay with the same arguments is
/// deterministic and idempotent. Signature authorization remains a reader-side
/// policy decision.
pub fn publish_commit<S>(
    store: &mut S,
    descriptor: &CollectionDescriptor,
    data: &Blob<SimpleArchive>,
    metadata: &Blob<SimpleArchive>,
    signing_key: &SigningKey,
) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStorePut + CollectionStore,
{
    let prepared =
        prepare_commit(descriptor, data, metadata, signing_key).map_err(widen_preparation_error)?;
    prepared.stage(store)?.finalize()
}

/// Publish a self-contained fact fragment as a signed membership root.
///
/// The fragment's facts and metafacts become the signed data and metadata
/// elements, and its shared blob store backs handles in either set. Before
/// touching `store`, this boundary recomputes every embedded blob's identity
/// and requires both its [`MemoryBlobStore`] key and cached [`Blob`] handle to
/// match the bytes. A mismatch is rejected rather than normalized because a
/// fact may name the forged identity.
///
/// Fragment exports are not serialized. The two fact sets become canonical
/// `SimpleArchive` data and metadata elements. The shared prepared-publication
/// path writes the descriptor blob, embedded blobs, and both
/// archives before inserting the signed record last.
/// The same backend-recovery boundary documented by [`PublicationError`]
/// applies.
pub fn publish_fragment_commit<S>(
    store: &mut S,
    descriptor: &CollectionDescriptor,
    fragment: Fragment,
    signing_key: &SigningKey,
) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStorePut + CollectionStore,
{
    let prepared = prepare_fragment_commit(descriptor, fragment, signing_key)
        .map_err(widen_preparation_error)?;
    prepared.stage(store)?.finalize()
}

/// Publish an exact merge after writing its descriptor, inputs, and result.
///
/// Input blobs are normalized from their bytes, ordered by their freshly
/// computed Blake3 identities, validated, and joined directly. The exact write
/// order is:
///
/// 1. collection-descriptor blob, canonical low input, canonical high input,
///    result;
/// 2. merge record.
///
/// The returned pair is `(canonical record, canonical result blob)`. A merge
/// record is never attempted before successful dependency writes. No
/// durability flush is implied. Failed backend I/O may require recovery
/// according to that backend's contract; after recovery, replay with the same
/// arguments is deterministic and idempotent.
pub fn publish_merge<S>(
    store: &mut S,
    descriptor: &CollectionDescriptor,
    low: &Blob<SimpleArchive>,
    high: &Blob<SimpleArchive>,
) -> Result<(CollectionMerge, Blob<SimpleArchive>), PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStorePut + CollectionStore,
{
    validate_descriptor(descriptor).map_err(PublicationError::Validation)?;

    let mut low = normalize_blob(low);
    let mut high = normalize_blob(high);
    let mut low_data = normalized_data_identity(&low);
    let mut high_data = normalized_data_identity(&high);
    if high_data < low_data {
        std::mem::swap(&mut low, &mut high);
        std::mem::swap(&mut low_data, &mut high_data);
    }

    let low_rows = canonical_rows(&low).map_err(|source| {
        PublicationError::Validation(SimpleArchiveUnionValidationError::InvalidElement {
            role: ElementRole::MergeLow,
            source,
        })
    })?;
    let high_rows = canonical_rows(&high).map_err(|source| {
        PublicationError::Validation(SimpleArchiveUnionValidationError::InvalidElement {
            role: ElementRole::MergeHigh,
            source,
        })
    })?;
    let result = join_canonical_rows(&low, &high, &low_rows, &high_rows);
    let merge = CollectionMerge::new(
        descriptor.handle(),
        low_data,
        high_data,
        normalized_data_identity(&result),
    );

    store
        .put::<SimpleArchive, _>(descriptor.to_blob())
        .map_err(PublicationError::DependencyPut)?;
    store
        .put::<SimpleArchive, _>(low)
        .map_err(PublicationError::DependencyPut)?;
    store
        .put::<SimpleArchive, _>(high)
        .map_err(PublicationError::DependencyPut)?;
    store
        .put::<SimpleArchive, _>(result.clone())
        .map_err(PublicationError::DependencyPut)?;
    store
        .insert(CollectionRecord::Merge(merge))
        .map_err(PublicationError::RecordInsert)?;

    Ok((merge, result))
}

fn validate_descriptor(
    descriptor: &CollectionDescriptor,
) -> Result<(), SimpleArchiveUnionValidationError> {
    let expected_representation = <SimpleArchive as MetaDescribe>::id();
    if descriptor.representation() != expected_representation {
        return Err(SimpleArchiveUnionValidationError::WrongRepresentation {
            expected: expected_representation,
            actual: descriptor.representation(),
        });
    }
    if descriptor.recipe() != TRIBLE_SET_UNION_RECIPE_V1 {
        return Err(SimpleArchiveUnionValidationError::WrongRecipe {
            expected: TRIBLE_SET_UNION_RECIPE_V1,
            actual: descriptor.recipe(),
        });
    }
    Ok(())
}

fn validate_collection(
    descriptor: &CollectionDescriptor,
    actual: CollectionId,
) -> Result<(), SimpleArchiveUnionValidationError> {
    if actual != descriptor.handle() {
        return Err(SimpleArchiveUnionValidationError::WrongCollection {
            expected: descriptor.handle(),
            actual,
        });
    }
    Ok(())
}

fn validate_endpoint(
    role: ElementRole,
    expected: CollectionData,
    blob: &Blob<SimpleArchive>,
) -> Result<(), SimpleArchiveUnionValidationError> {
    validate_handle(role, expected, blob)?;
    validate_element(blob)
        .map_err(|source| SimpleArchiveUnionValidationError::InvalidElement { role, source })
}

fn validate_handle(
    role: ElementRole,
    expected: CollectionData,
    blob: &Blob<SimpleArchive>,
) -> Result<(), SimpleArchiveUnionValidationError> {
    // `Blob::with_handle` is an explicitly trusted read-path constructor, so
    // an admission boundary must not rely on its cached handle. Recompute the
    // content identity from the supplied bytes before accepting the endpoint.
    let actual = Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes));
    if actual != expected {
        return Err(SimpleArchiveUnionValidationError::EndpointMismatch {
            role,
            expected,
            actual,
        });
    }
    Ok(())
}

fn normalize_blob(blob: &Blob<SimpleArchive>) -> Blob<SimpleArchive> {
    Blob::new(blob.bytes.clone())
}

fn checked_embedded_blobs(
    mut blobs: MemoryBlobStore,
) -> Result<Vec<Blob<UnknownBlob>>, (CollectionData, CollectionData, CollectionData)> {
    let reader = blobs
        .reader()
        .expect("MemoryBlobStore::reader is infallible");
    let mut checked = Vec::with_capacity(reader.len());
    for (store_key, blob) in reader {
        let cached_handle = blob.get_handle();
        let actual = Blob::<UnknownBlob>::new(blob.bytes.clone()).get_handle();
        if store_key != actual || cached_handle != actual {
            return Err((
                Handle::<UnknownBlob>::to_hash(store_key),
                Handle::<UnknownBlob>::to_hash(cached_handle),
                Handle::<UnknownBlob>::to_hash(actual),
            ));
        }
        checked.push(blob);
    }
    Ok(checked)
}

fn normalized_data_identity(blob: &Blob<SimpleArchive>) -> CollectionData {
    Handle::<SimpleArchive>::to_hash(blob.get_handle())
}

fn join_canonical_rows(
    left: &Blob<SimpleArchive>,
    right: &Blob<SimpleArchive>,
    left_rows: &[[u8; TRIBLE_LEN]],
    right_rows: &[[u8; TRIBLE_LEN]],
) -> Blob<SimpleArchive> {
    if left.bytes == right.bytes || right_rows.is_empty() {
        return Blob::new(left.bytes.clone());
    }
    if left_rows.is_empty() {
        return Blob::new(right.bytes.clone());
    }

    let mut rows = Vec::with_capacity(left_rows.len() + right_rows.len());
    rows.extend(UnionRows::new(left_rows, right_rows).copied());
    Blob::new(Bytes::from(rows))
}

fn canonical_rows(blob: &Blob<SimpleArchive>) -> Result<View<[[u8; TRIBLE_LEN]]>, UnarchiveError> {
    let rows: View<[[u8; TRIBLE_LEN]]> = blob
        .bytes
        .clone()
        .view()
        .map_err(|_| UnarchiveError::BadArchive)?;
    let mut previous: Option<&[u8; TRIBLE_LEN]> = None;
    for row in rows.iter() {
        if Trible::as_transmute_force_raw(row).is_none() {
            return Err(UnarchiveError::BadTrible);
        }
        if let Some(previous) = previous {
            if previous == row {
                return Err(UnarchiveError::BadCanonicalizationRedundancy);
            }
            if previous > row {
                return Err(UnarchiveError::BadCanonicalizationOrdering);
            }
        }
        previous = Some(row);
    }
    Ok(rows)
}

struct UnionRows<'a> {
    left: &'a [[u8; TRIBLE_LEN]],
    right: &'a [[u8; TRIBLE_LEN]],
    left_index: usize,
    right_index: usize,
}

impl<'a> UnionRows<'a> {
    fn new(left: &'a [[u8; TRIBLE_LEN]], right: &'a [[u8; TRIBLE_LEN]]) -> Self {
        Self {
            left,
            right,
            left_index: 0,
            right_index: 0,
        }
    }
}

impl<'a> Iterator for UnionRows<'a> {
    type Item = &'a [u8; TRIBLE_LEN];

    fn next(&mut self) -> Option<Self::Item> {
        match (
            self.left.get(self.left_index),
            self.right.get(self.right_index),
        ) {
            (Some(left), Some(right)) => match left.cmp(right) {
                std::cmp::Ordering::Less => {
                    self.left_index += 1;
                    Some(left)
                }
                std::cmp::Ordering::Equal => {
                    self.left_index += 1;
                    self.right_index += 1;
                    Some(left)
                }
                std::cmp::Ordering::Greater => {
                    self.right_index += 1;
                    Some(right)
                }
            },
            (Some(left), None) => {
                self.left_index += 1;
                Some(left)
            }
            (None, Some(right)) => {
                self.right_index += 1;
                Some(right)
            }
            (None, None) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.len() - self.left_index;
        let right = self.right.len() - self.right_index;
        (left.max(right), left.checked_add(right))
    }
}

impl std::iter::FusedIterator for UnionRows<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::{BTreeMap, BTreeSet};

    use ed25519_dalek::SigningKey;
    use hex_literal::hex;

    use crate::blob::encodings::longstring::LongString;
    use crate::blob::encodings::rawbytes::RawBytes;
    use crate::blob::{BlobEncoding, IntoBlob};
    use crate::collection::{
        discover_collection_records, empty_metadata_handle, plan_collection_retention,
        resolve_collection_semantics, CollectionClaimValidation, CollectionDerive,
    };
    use crate::inline::InlineEncoding;
    use crate::macros::entity;
    use crate::repo::pile::Pile;
    use crate::repo::{BlobStore, BlobStoreGet};
    use crate::trible::TribleSet;

    mod fragment_ns {
        use crate::prelude::*;

        attributes! {
            // Test-only sentinel attributes; these are not protocol ids.
            "DD00000000000000DD00000000000031" unsafe as pub text: inlineencodings::Handle<blobencodings::LongString>;
            "DD00000000000000DD00000000000032" unsafe as pub payload: inlineencodings::Handle<blobencodings::RawBytes>;
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ProbeFailure(usize);

    impl fmt::Display for ProbeFailure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "injected failure at operation {}", self.0)
        }
    }

    impl Error for ProbeFailure {}

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ProbeEvent {
        Put([u8; 32]),
        Insert(Id),
    }

    #[derive(Default)]
    struct ProbeStore {
        events: Vec<ProbeEvent>,
        known: BTreeSet<[u8; 32]>,
        records: BTreeMap<Id, CollectionRecord>,
        fail_at: Option<usize>,
    }

    impl ProbeStore {
        // This probe fails before an operation takes effect, so it exercises
        // publication ordering at trait-operation boundaries. BlobStorePut
        // does not promise that a real backend cannot leave torn physical I/O.
        fn failing_before_effect_at(operation: usize) -> Self {
            Self {
                fail_at: Some(operation),
                ..Self::default()
            }
        }

        fn attempt(&mut self, event: ProbeEvent) -> Result<(), ProbeFailure> {
            self.events.push(event);
            let operation = self.events.len();
            if self.fail_at == Some(operation) {
                return Err(ProbeFailure(operation));
            }
            Ok(())
        }

        fn recover(&mut self) {
            self.fail_at = None;
        }
    }

    impl BlobStorePut for ProbeStore {
        type PutError = ProbeFailure;

        fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
        where
            S: BlobEncoding + 'static,
            T: IntoBlob<S>,
            Handle<S>: InlineEncoding,
        {
            let blob: Blob<S> = item.to_blob();
            let handle = blob.get_handle();
            self.attempt(ProbeEvent::Put(handle.raw))?;
            self.known.insert(handle.raw);
            Ok(handle)
        }
    }

    impl CollectionStore for ProbeStore {
        type RecordsError = Infallible;
        type InsertError = ProbeFailure;
        type RecordIter<'a> = std::vec::IntoIter<Result<CollectionRecord, Infallible>>;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            Ok(self
                .records
                .values()
                .copied()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter())
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.attempt(ProbeEvent::Insert(record.id()))?;
            self.records.entry(record.id()).or_insert(record);
            Ok(())
        }
    }

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn row(entity: u8, attribute: u8, value: u8) -> [u8; TRIBLE_LEN] {
        let mut row = [value; TRIBLE_LEN];
        row[..16].fill(entity);
        row[16..32].fill(attribute);
        row
    }

    fn archive(rows: impl IntoIterator<Item = [u8; TRIBLE_LEN]>) -> Blob<SimpleArchive> {
        let mut facts = TribleSet::new();
        for row in rows {
            facts.insert(&Trible::force_raw(row).unwrap());
        }
        facts.to_blob()
    }

    fn raw_archive(rows: Vec<[u8; TRIBLE_LEN]>) -> Blob<SimpleArchive> {
        Blob::new(Bytes::from(rows))
    }

    fn data(blob: &Blob<SimpleArchive>) -> CollectionData {
        Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
    }

    fn ordered_inputs<'a>(
        left: &'a Blob<SimpleArchive>,
        right: &'a Blob<SimpleArchive>,
    ) -> (&'a Blob<SimpleArchive>, &'a Blob<SimpleArchive>) {
        if data(left) <= data(right) {
            (left, right)
        } else {
            (right, left)
        }
    }

    fn put_event<S>(blob: &Blob<S>) -> ProbeEvent
    where
        S: BlobEncoding,
        Handle<S>: InlineEncoding,
    {
        ProbeEvent::Put(blob.get_handle().raw)
    }

    fn insert_event(record: CollectionRecord) -> ProbeEvent {
        ProbeEvent::Insert(record.id())
    }

    fn fragment_fixture() -> (
        Fragment,
        Inline<Handle<LongString>>,
        Inline<Handle<RawBytes>>,
    ) {
        let text: Blob<LongString> = String::from("a self-contained content blob").to_blob();
        let text_handle = text.get_handle();
        let mut content = entity! { fragment_ns::text: text };

        let payload: Blob<RawBytes> = vec![0, 1, 2, 3, 0xFE, 0xFF].to_blob();
        let payload_handle = payload.get_handle();
        let metadata = entity! { fragment_ns::payload: payload };
        content.describe_with(metadata);

        (content, text_handle, payload_handle)
    }

    fn embedded_put_events(fragment: &Fragment) -> Vec<ProbeEvent> {
        let mut blobs = fragment.blobs().clone();
        let mut handles: Vec<_> = blobs
            .reader()
            .expect("memory store reader is infallible")
            .iter()
            .map(|(handle, _)| handle.raw)
            .collect();
        handles.sort_unstable();
        handles.into_iter().map(ProbeEvent::Put).collect()
    }

    fn commit_fixture() -> (
        CollectionDescriptor,
        Blob<SimpleArchive>,
        Blob<SimpleArchive>,
        SigningKey,
        CollectionCommit,
    ) {
        let descriptor = descriptor(id(1));
        let data_blob = archive([row(1, 1, 1), row(3, 1, 3)]);
        let metadata = archive([row(9, 1, 9)]);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let commit = CollectionCommit::sign(
            &signing_key,
            descriptor.handle(),
            data(&data_blob),
            metadata.get_handle(),
        );
        (descriptor, data_blob, metadata, signing_key, commit)
    }

    #[test]
    fn prepared_fragment_is_canonical_idempotent_and_commits_after_caller_artifacts() {
        let source_descriptor = descriptor(id(1));
        let target = descriptor(id(2));
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, _text_handle, _payload_handle) = fragment_fixture();
        let embedded = embedded_put_events(&fragment);
        let content_archive: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let metadata_archive: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();
        let expected = CollectionCommit::sign(
            &signing_key,
            source_descriptor.handle(),
            data(&content_archive),
            metadata_archive.get_handle(),
        );

        let prepared =
            prepare_fragment_commit(&source_descriptor, fragment.clone(), &signing_key).unwrap();
        let repeated = prepare_fragment_commit(&source_descriptor, fragment, &signing_key).unwrap();
        assert_eq!(prepared.commit(), &expected);
        assert_eq!(repeated.commit(), &expected);
        assert_eq!(prepared.commit().id(), repeated.commit().id());
        assert_eq!(prepared.commit().to_bytes(), repeated.commit().to_bytes());

        let derive = CollectionDerive::new(
            source_descriptor.handle(),
            target.handle(),
            expected.data(),
            Inline::new([0x42; 32]),
        );
        let derive_record = CollectionRecord::Derive(derive);
        let commit_record = CollectionRecord::Commit(expected);
        let sequence = [
            vec![put_event(&CollectionDescriptor::to_blob(
                &source_descriptor,
            ))],
            embedded,
            vec![
                put_event(&content_archive),
                put_event(&metadata_archive),
                insert_event(derive_record),
                insert_event(commit_record),
            ],
        ]
        .concat();

        let mut store = ProbeStore::default();
        for prepared in [prepared, repeated] {
            let mut staged = prepared.stage(&mut store).unwrap();
            assert_eq!(staged.commit(), &expected);
            staged.store_mut().insert(derive_record).unwrap();
            assert_eq!(staged.finalize().unwrap(), expected);
        }

        let mut expected_events = sequence.clone();
        expected_events.extend(sequence);
        assert_eq!(store.events, expected_events);
        assert!(store.records.contains_key(&derive.id()));
        assert!(store.records.contains_key(&expected.id()));
        validate_commit(&source_descriptor, &expected, &content_archive).unwrap();
    }

    #[test]
    fn staged_fragment_is_not_a_discoverable_commit_and_drop_is_inert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("staged-only.pile");
        std::fs::File::create(&path).unwrap();

        let descriptor = descriptor(id(1));
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, text_handle, payload_handle) = fragment_fixture();
        let expected_content: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let expected_metadata: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();
        let prepared = prepare_fragment_commit(&descriptor, fragment, &signing_key).unwrap();
        let withheld = prepared.commit().clone();

        let mut pile = Pile::open(&path).unwrap();
        let mut staged = prepared.stage(&mut pile).unwrap();
        {
            let discovered = discover_collection_records(staged.store_mut()).unwrap();
            let reader = staged.store_mut().reader().unwrap();
            assert!(discovered.commits().is_empty());
            assert!(discovered.merges().is_empty());
            assert!(discovered.derives().is_empty());
            let descriptor_blob: Blob<SimpleArchive> = reader.get(descriptor.handle()).unwrap();
            assert_eq!(
                CollectionDescriptor::decode(&descriptor_blob).unwrap(),
                descriptor
            );

            let resolution = resolve_collection_semantics(&discovered, &BTreeSet::new(), |_| {
                Ok::<_, Infallible>(CollectionClaimValidation::<()>::Pending)
            })
            .unwrap();
            assert!(resolution.admitted_claims().is_empty());
            assert!(resolution
                .semantics()
                .members(descriptor.handle())
                .is_none());
            let roots = plan_collection_retention(&discovered, &resolution, &reader).unwrap();
            assert!(roots.is_empty());
            assert!(roots.expanded(&reader).is_empty());

            let content: Blob<SimpleArchive> = reader
                .get::<Blob<SimpleArchive>, SimpleArchive>(withheld.data().transmute())
                .unwrap();
            let metadata: Blob<SimpleArchive> = reader.get(withheld.metadata()).unwrap();
            let text: View<str> = reader.get::<View<str>, LongString>(text_handle).unwrap();
            let payload: Bytes = reader.get::<Bytes, RawBytes>(payload_handle).unwrap();
            assert_eq!(content, expected_content);
            assert_eq!(metadata, expected_metadata);
            assert_eq!(&*text, "a self-contained content blob");
            assert_eq!(&*payload, &[0, 1, 2, 3, 0xFE, 0xFF]);
        }

        // Drop deliberately does not cross the visibility boundary. Explicit
        // close still succeeds and preserves only the staged dependencies.
        drop(staged);
        pile.close().unwrap();

        let mut reopened = Pile::open(&path).unwrap();
        let discovered = discover_collection_records(&mut reopened).unwrap();
        let reader = reopened.reader().unwrap();
        assert!(discovered.commits().is_empty());
        assert!(!discovered
            .commits()
            .iter()
            .any(|commit| commit.id() == withheld.id()));
        let descriptor_blob: Blob<SimpleArchive> = reader.get(descriptor.handle()).unwrap();
        assert_eq!(
            CollectionDescriptor::decode(&descriptor_blob).unwrap(),
            descriptor
        );
        drop(reader);
        reopened.close().unwrap();
    }

    #[test]
    fn fragment_without_metafacts_still_stages_the_canonical_empty_metadata_archive() {
        let descriptor = descriptor(id(1));
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let empty_archive: Blob<SimpleArchive> = TribleSet::new().to_blob();

        let prepared =
            prepare_fragment_commit(&descriptor, Fragment::empty(), &signing_key).unwrap();

        assert_eq!(prepared.commit().metadata(), empty_metadata_handle());
        assert_eq!(prepared.metadata.get_handle(), empty_metadata_handle());
        assert_eq!(prepared.metadata, empty_archive);

        let mut store = ProbeStore::default();
        let staged = prepared.stage(&mut store).unwrap();
        drop(staged);
        assert_eq!(
            store
                .events
                .iter()
                .filter(|event| **event == ProbeEvent::Put(empty_metadata_handle().raw))
                .count(),
            2,
            "empty data and empty metadata are both staged explicitly"
        );
    }

    #[test]
    fn commit_publication_normalizes_orders_and_replays_idempotently() {
        let (descriptor, data_blob, metadata, signing_key, expected) = commit_fixture();
        let bogus = archive([row(14, 1, 14)]);
        let forged_data = Blob::with_handle(data_blob.bytes.clone(), bogus.get_handle());
        let forged_metadata = Blob::with_handle(metadata.bytes.clone(), bogus.get_handle());
        let sequence = vec![
            put_event(&CollectionDescriptor::to_blob(&descriptor)),
            put_event(&data_blob),
            put_event(&metadata),
            insert_event(CollectionRecord::Commit(expected)),
        ];

        let mut store = ProbeStore::default();
        let first = publish_commit(
            &mut store,
            &descriptor,
            &forged_data,
            &forged_metadata,
            &signing_key,
        )
        .unwrap();
        let second = publish_commit(
            &mut store,
            &descriptor,
            &forged_data,
            &forged_metadata,
            &signing_key,
        )
        .unwrap();

        assert_eq!(first, expected);
        assert_eq!(second, expected);
        assert_eq!(first.data(), data(&data_blob));
        assert_eq!(first.metadata(), metadata.get_handle());
        first.verify_strict().unwrap();
        validate_commit(&descriptor, &first, &data_blob).unwrap();

        let mut expected_events = sequence.clone();
        expected_events.extend(sequence);
        assert_eq!(store.events, expected_events);
        let expected_handles = BTreeSet::from([
            descriptor.handle().raw,
            data_blob.get_handle().raw,
            metadata.get_handle().raw,
        ]);
        assert_eq!(store.known, expected_handles);
        assert_eq!(
            store.records.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([expected.id()])
        );
        assert!(!store.known.contains(&bogus.get_handle().raw));
    }

    #[test]
    fn fragment_commit_puts_embedded_dependencies_before_record_and_replays_idempotently() {
        let descriptor = descriptor(id(1));
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, _text_handle, _payload_handle) = fragment_fixture();
        let embedded = embedded_put_events(&fragment);
        let content_archive: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let metadata_archive: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();
        let expected = CollectionCommit::sign(
            &signing_key,
            descriptor.handle(),
            data(&content_archive),
            metadata_archive.get_handle(),
        );
        let sequence = [
            vec![put_event(&CollectionDescriptor::to_blob(&descriptor))],
            embedded.clone(),
            vec![
                put_event(&content_archive),
                put_event(&metadata_archive),
                insert_event(CollectionRecord::Commit(expected)),
            ],
        ]
        .concat();

        let mut store = ProbeStore::default();
        let first =
            publish_fragment_commit(&mut store, &descriptor, fragment.clone(), &signing_key)
                .unwrap();
        let second =
            publish_fragment_commit(&mut store, &descriptor, fragment, &signing_key).unwrap();

        assert_eq!(first, expected);
        assert_eq!(second, expected);
        let mut expected_events = sequence.clone();
        expected_events.extend(sequence);
        assert_eq!(store.events, expected_events);

        let mut expected_handles: BTreeSet<_> = embedded
            .into_iter()
            .map(|event| match event {
                ProbeEvent::Put(handle) => handle,
                ProbeEvent::Insert(_) => {
                    unreachable!("embedded events are puts")
                }
            })
            .collect();
        expected_handles.extend([
            descriptor.handle().raw,
            content_archive.get_handle().raw,
            metadata_archive.get_handle().raw,
        ]);
        assert_eq!(store.known, expected_handles);
        assert_eq!(
            store.records.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([expected.id()])
        );
    }

    #[test]
    fn fragment_commit_rejects_forged_embedded_identities_before_writing() {
        let descriptor = descriptor(id(1));
        let signing_key = SigningKey::from_bytes(&[7; 32]);

        // This is the exact forged-cache shape that `entity!` deliberately
        // preserves: both the fact and MemoryBlobStore key name the bogus
        // cached handle even though the bytes hash elsewhere.
        let bogus_text_handle = Inline::<Handle<LongString>>::new([0xAA; 32]);
        let forged_text = Blob::<LongString>::with_handle(
            Bytes::from(b"bytes behind a forged cached handle".to_vec()),
            bogus_text_handle,
        );
        let actual_text_handle = Blob::<LongString>::new(forged_text.bytes.clone()).get_handle();
        let forged_content = entity! { fragment_ns::text: forged_text };
        let mut store = ProbeStore::default();
        let error = publish_fragment_commit(&mut store, &descriptor, forged_content, &signing_key)
            .unwrap_err();
        assert!(matches!(
            error,
            PublicationError::InvalidEmbeddedBlob {
                store_key,
                cached_handle,
                actual,
            } if store_key == Handle::<LongString>::to_hash(bogus_text_handle)
                && cached_handle == Handle::<LongString>::to_hash(bogus_text_handle)
                && actual == Handle::<LongString>::to_hash(actual_text_handle)
        ));
        assert!(store.events.is_empty());

        // `MemoryBlobStore::from_iter` can independently forge its PATCH key
        // even when the Blob's own cache is correct. Reject that shape too.
        let payload: Blob<RawBytes> = vec![9, 8, 7].to_blob();
        let actual_payload_handle: Inline<Handle<UnknownBlob>> = payload.get_handle().transmute();
        let bogus_store_key = Inline::<Handle<UnknownBlob>>::new([0xBB; 32]);
        let embedded: MemoryBlobStore =
            std::iter::once((bogus_store_key, payload.transmute())).collect();
        let forged_content = Fragment::from_facts_and_blobs(TribleSet::new(), embedded);
        let mut store = ProbeStore::default();
        let error = publish_fragment_commit(&mut store, &descriptor, forged_content, &signing_key)
            .unwrap_err();
        assert!(matches!(
            error,
            PublicationError::InvalidEmbeddedBlob {
                store_key,
                cached_handle,
                actual,
            } if store_key == Handle::<UnknownBlob>::to_hash(bogus_store_key)
                && cached_handle == Handle::<UnknownBlob>::to_hash(actual_payload_handle)
                && actual == Handle::<UnknownBlob>::to_hash(actual_payload_handle)
        ));
        assert!(store.events.is_empty());
    }

    #[test]
    fn merge_publication_normalizes_canonicalizes_and_replays_idempotently() {
        let descriptor = descriptor(id(1));
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let bogus = archive([row(14, 1, 14)]);
        let forged_left = Blob::with_handle(left.bytes.clone(), bogus.get_handle());
        let forged_right = Blob::with_handle(right.bytes.clone(), bogus.get_handle());
        let (low, high) = ordered_inputs(&left, &right);
        let expected_result = join(low, high).unwrap();
        let expected_merge = CollectionMerge::new(
            descriptor.handle(),
            data(low),
            data(high),
            data(&expected_result),
        );
        let sequence = vec![
            put_event(&CollectionDescriptor::to_blob(&descriptor)),
            put_event(low),
            put_event(high),
            put_event(&expected_result),
            insert_event(CollectionRecord::Merge(expected_merge)),
        ];

        let mut store = ProbeStore::default();
        let first = publish_merge(&mut store, &descriptor, &forged_right, &forged_left).unwrap();
        let second = publish_merge(&mut store, &descriptor, &forged_left, &forged_right).unwrap();

        assert_eq!(first, (expected_merge.clone(), expected_result.clone()));
        assert_eq!(second, (expected_merge.clone(), expected_result.clone()));
        validate_merge(&descriptor, &first.0, low, high, &first.1).unwrap();

        let mut expected_events = sequence.clone();
        expected_events.extend(sequence);
        assert_eq!(store.events, expected_events);
        let expected_handles = BTreeSet::from([
            descriptor.handle().raw,
            low.get_handle().raw,
            high.get_handle().raw,
            expected_result.get_handle().raw,
        ]);
        assert_eq!(store.known, expected_handles);
        assert_eq!(
            store.records.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([expected_merge.id()])
        );
        assert!(!store.known.contains(&bogus.get_handle().raw));
    }

    #[test]
    fn commit_publication_orders_completed_prefixes_and_replays_after_recovery() {
        let (descriptor, data_blob, metadata, signing_key, expected) = commit_fixture();
        for fail_at in 1..=4 {
            let mut store = ProbeStore::failing_before_effect_at(fail_at);
            let error =
                publish_commit(&mut store, &descriptor, &data_blob, &metadata, &signing_key)
                    .unwrap_err();
            match (fail_at, error) {
                (1..=3, PublicationError::DependencyPut(ProbeFailure(at)))
                | (4, PublicationError::RecordInsert(ProbeFailure(at))) => {
                    assert_eq!(at, fail_at)
                }
                (_, error) => panic!("unexpected publication error: {error}"),
            }

            assert!(!store.records.contains_key(&expected.id()));
            if fail_at <= 3 {
                assert!(!store.events.contains(&ProbeEvent::Insert(expected.id())));
            }

            store.recover();
            let retried =
                publish_commit(&mut store, &descriptor, &data_blob, &metadata, &signing_key)
                    .unwrap();
            assert_eq!(retried, expected);
            assert!(store.records.contains_key(&expected.id()));
        }
    }

    #[test]
    fn merge_publication_orders_completed_prefixes_and_replays_after_recovery() {
        let descriptor = descriptor(id(1));
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let (low, high) = ordered_inputs(&left, &right);
        let result = join(low, high).unwrap();
        let expected =
            CollectionMerge::new(descriptor.handle(), data(low), data(high), data(&result));
        for fail_at in 1..=5 {
            let mut store = ProbeStore::failing_before_effect_at(fail_at);
            let error = publish_merge(&mut store, &descriptor, &left, &right).unwrap_err();
            match (fail_at, error) {
                (1..=4, PublicationError::DependencyPut(ProbeFailure(at)))
                | (5, PublicationError::RecordInsert(ProbeFailure(at))) => {
                    assert_eq!(at, fail_at)
                }
                (_, error) => panic!("unexpected publication error: {error}"),
            }

            assert!(!store.records.contains_key(&expected.id()));
            if fail_at <= 4 {
                assert!(!store.events.contains(&ProbeEvent::Insert(expected.id())));
            }

            store.recover();
            let retried = publish_merge(&mut store, &descriptor, &left, &right).unwrap();
            assert_eq!(retried, (expected.clone(), result.clone()));
            assert!(store.records.contains_key(&expected.id()));
        }
    }

    #[test]
    fn publication_rejects_every_invalid_input_before_writing() {
        let (descriptor, data_blob, metadata, signing_key, _) = commit_fixture();
        let mut store = ProbeStore::default();
        let wrong_descriptor =
            CollectionDescriptor::new(descriptor.scope(), id(8), TRIBLE_SET_UNION_RECIPE_V1);
        assert!(matches!(
            publish_commit(
                &mut store,
                &wrong_descriptor,
                &data_blob,
                &metadata,
                &signing_key,
            ),
            Err(PublicationError::Validation(
                SimpleArchiveUnionValidationError::WrongRepresentation { .. }
            ))
        ));
        assert!(store.events.is_empty());

        let invalid_data = raw_archive(vec![row(2, 1, 2), row(1, 1, 1)]);
        assert!(matches!(
            publish_commit(
                &mut store,
                &descriptor,
                &invalid_data,
                &metadata,
                &signing_key,
            ),
            Err(PublicationError::Validation(
                SimpleArchiveUnionValidationError::InvalidElement { .. }
            ))
        ));
        assert!(store.events.is_empty());

        let invalid_metadata = raw_archive(vec![row(4, 1, 4), row(3, 1, 3)]);
        assert!(matches!(
            publish_commit(
                &mut store,
                &descriptor,
                &data_blob,
                &invalid_metadata,
                &signing_key,
            ),
            Err(PublicationError::InvalidMetadata(
                UnarchiveError::BadCanonicalizationOrdering
            ))
        ));
        assert!(store.events.is_empty());

        assert!(matches!(
            publish_merge(&mut store, &descriptor, &invalid_data, &data_blob,),
            Err(PublicationError::Validation(
                SimpleArchiveUnionValidationError::InvalidElement { .. }
            ))
        ));
        assert!(store.events.is_empty());
    }

    #[test]
    fn pile_publication_roundtrips_through_discovery_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("collections.pile");
        std::fs::File::create(&path).unwrap();

        let descriptor = descriptor(id(1));
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let metadata = archive([row(9, 1, 9)]);
        let signing_key = SigningKey::from_bytes(&[7; 32]);

        let (commit, merge, result) = {
            let mut pile = Pile::open(&path).unwrap();
            let commit =
                publish_commit(&mut pile, &descriptor, &left, &metadata, &signing_key).unwrap();
            let (merge, result) = publish_merge(&mut pile, &descriptor, &right, &left).unwrap();
            pile.close().unwrap();
            (commit, merge, result)
        };

        let mut reopened = Pile::open(&path).unwrap();
        let discovered = discover_collection_records(&mut reopened).unwrap();
        let reader = reopened.reader().unwrap();
        assert_eq!(discovered.commits(), &[commit.clone()]);
        assert_eq!(discovered.merges(), &[merge.clone()]);
        assert!(discovered.derives().is_empty());
        assert!(discovered.diagnostics().is_empty());

        let fetched_descriptor: Blob<SimpleArchive> = reader.get(descriptor.handle()).unwrap();
        let fetched_left: Blob<SimpleArchive> = reader.get(left.get_handle()).unwrap();
        let fetched_right: Blob<SimpleArchive> = reader.get(right.get_handle()).unwrap();
        let fetched_metadata: Blob<SimpleArchive> = reader.get(metadata.get_handle()).unwrap();
        let fetched_result: Blob<SimpleArchive> = reader.get(result.get_handle()).unwrap();
        assert_eq!(
            CollectionDescriptor::decode(&fetched_descriptor).unwrap(),
            descriptor
        );
        assert_eq!(fetched_left, left);
        assert_eq!(fetched_right, right);
        assert_eq!(fetched_metadata, metadata);
        assert_eq!(fetched_result, result);
        validate_commit(&descriptor, &commit, &fetched_left).unwrap();
        let (low, high) = ordered_inputs(&fetched_left, &fetched_right);
        validate_merge(&descriptor, &merge, low, high, &fetched_result).unwrap();

        drop(reader);
        reopened.close().unwrap();
    }

    #[test]
    fn fragment_commit_roundtrips_embedded_blobs_through_a_reopened_pile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fragment-commit.pile");
        std::fs::File::create(&path).unwrap();

        let descriptor = descriptor(id(1));
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, text_handle, payload_handle) = fragment_fixture();
        let expected_content: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let expected_metadata: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();

        let commit = {
            let mut pile = Pile::open(&path).unwrap();
            let commit =
                publish_fragment_commit(&mut pile, &descriptor, fragment, &signing_key).unwrap();
            pile.close().unwrap();
            commit
        };

        let mut reopened = Pile::open(&path).unwrap();
        let discovered = discover_collection_records(&mut reopened).unwrap();
        let reader = reopened.reader().unwrap();
        assert_eq!(discovered.commits(), &[commit.clone()]);
        assert!(discovered.merges().is_empty());
        assert!(discovered.derives().is_empty());
        assert!(discovered.diagnostics().is_empty());

        let fetched_descriptor: Blob<SimpleArchive> = reader.get(descriptor.handle()).unwrap();
        let content_handle: Inline<Handle<SimpleArchive>> = commit.data().transmute();
        let fetched_content: Blob<SimpleArchive> = reader.get(content_handle).unwrap();
        let fetched_metadata: Blob<SimpleArchive> = reader.get(commit.metadata()).unwrap();
        assert_eq!(
            CollectionDescriptor::decode(&fetched_descriptor).unwrap(),
            descriptor
        );
        assert_eq!(fetched_content, expected_content);
        assert_eq!(fetched_metadata, expected_metadata);
        validate_commit(&descriptor, &commit, &fetched_content).unwrap();

        let fetched_text: View<str> = reader.get::<View<str>, LongString>(text_handle).unwrap();
        let fetched_payload: Bytes = reader.get::<Bytes, RawBytes>(payload_handle).unwrap();
        assert_eq!(&*fetched_text, "a self-contained content blob");
        assert_eq!(&*fetched_payload, &[0, 1, 2, 3, 0xFE, 0xFF]);

        drop(reader);
        reopened.close().unwrap();
    }

    #[test]
    fn descriptor_and_empty_element_are_golden() {
        let descriptor = descriptor(id(1));
        assert_eq!(
            <SimpleArchive as MetaDescribe>::id(),
            id_hex!("8F4A27C8581DADCBA1ADA8BA228069B6")
        );
        assert_eq!(
            TRIBLE_SET_UNION_RECIPE_V1,
            id_hex!("6D64C5F4B9E9B73F57C5F8702AB7FE45")
        );
        assert_eq!(descriptor.scope(), id(1));
        assert_eq!(
            descriptor.entity_id(),
            id_hex!("4B6F24A289B950F2CF20896EAB7A1658")
        );
        assert_eq!(
            descriptor.handle().raw,
            hex!("A639BFB1D8F4DD5E9AF4667512A23673812866F2CBF01D3F11DEF89850FA65B9")
        );
        assert_eq!(
            CollectionDescriptor::to_blob(&descriptor).get_handle(),
            descriptor.handle()
        );

        let empty: Blob<SimpleArchive> = TribleSet::new().to_blob();
        validate_element(&empty).unwrap();
        assert!(empty.bytes.is_empty());
        assert_eq!(
            empty.get_handle().raw,
            hex!("AF1349B9F5F9A1A6A0404DEA36DCC9499BCB25C9ADC112B7CC9A93CAE41F3262")
        );
    }

    #[test]
    fn element_validation_matches_simplearchive_canonical_rules() {
        let first = row(1, 1, 1);
        let second = row(2, 1, 2);
        validate_element(&raw_archive(vec![first, second])).unwrap();
        assert_eq!(
            validate_element(&Blob::new(vec![0_u8; TRIBLE_LEN - 1].into())),
            Err(UnarchiveError::BadArchive)
        );

        let mut nil_entity = first;
        nil_entity[..16].fill(0);
        assert_eq!(
            validate_element(&raw_archive(vec![nil_entity])),
            Err(UnarchiveError::BadTrible)
        );
        assert_eq!(
            validate_element(&raw_archive(vec![first, first])),
            Err(UnarchiveError::BadCanonicalizationRedundancy)
        );
        assert_eq!(
            validate_element(&raw_archive(vec![second, first])),
            Err(UnarchiveError::BadCanonicalizationOrdering)
        );
    }

    #[test]
    fn join_obeys_empty_idempotent_commutative_and_associative_laws() {
        let empty = archive([]);
        let a = archive([row(1, 1, 1), row(3, 1, 3)]);
        let b = archive([row(2, 1, 2), row(3, 1, 3)]);
        let c = archive([row(1, 2, 4), row(4, 1, 5)]);

        assert_eq!(join(&empty, &a).unwrap(), a);
        assert_eq!(join(&a, &empty).unwrap(), a);
        assert_eq!(join(&a, &a).unwrap(), a);
        assert_eq!(join(&a, &b).unwrap(), join(&b, &a).unwrap());

        let forged = Blob::with_handle(a.bytes.clone(), empty.get_handle());
        assert_ne!(forged.get_handle().raw, data(&forged).raw);
        let normalized = join(&forged, &empty).unwrap();
        assert_eq!(normalized.bytes, a.bytes);
        assert_eq!(normalized.get_handle().raw, data(&normalized).raw);

        let left_associated = join(&join(&a, &b).unwrap(), &c).unwrap();
        let right_associated = join(&a, &join(&b, &c).unwrap()).unwrap();
        assert_eq!(left_associated, right_associated);
        assert_eq!(left_associated.bytes.len(), 5 * TRIBLE_LEN);
    }

    #[test]
    fn commit_validation_binds_descriptor_collection_handle_and_bytes() {
        let descriptor = descriptor(id(1));
        let blob = archive([row(1, 1, 1)]);
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            descriptor.handle(),
            data(&blob),
            empty_metadata_handle(),
        );
        validate_commit(&descriptor, &commit, &blob).unwrap();

        let wrong_representation =
            CollectionDescriptor::new(descriptor.scope(), id(9), TRIBLE_SET_UNION_RECIPE_V1);
        assert!(matches!(
            validate_commit(&wrong_representation, &commit, &blob),
            Err(SimpleArchiveUnionValidationError::WrongRepresentation { .. })
        ));

        let wrong_recipe = CollectionDescriptor::new(
            descriptor.scope(),
            <SimpleArchive as MetaDescribe>::id(),
            id(9),
        );
        assert!(matches!(
            validate_commit(&wrong_recipe, &commit, &blob),
            Err(SimpleArchiveUnionValidationError::WrongRecipe { .. })
        ));

        let other_descriptor = super::descriptor(id(2));
        assert_eq!(
            validate_commit(&other_descriptor, &commit, &blob),
            Err(SimpleArchiveUnionValidationError::WrongCollection {
                expected: other_descriptor.handle(),
                actual: descriptor.handle(),
            })
        );

        let other_blob = archive([row(2, 1, 2)]);
        assert!(matches!(
            validate_commit(&descriptor, &commit, &other_blob),
            Err(SimpleArchiveUnionValidationError::EndpointMismatch {
                role: ElementRole::CommitData,
                ..
            })
        ));

        let forged = Blob::with_handle(other_blob.bytes.clone(), blob.get_handle());
        assert_eq!(
            validate_commit(&descriptor, &commit, &forged),
            Err(SimpleArchiveUnionValidationError::EndpointMismatch {
                role: ElementRole::CommitData,
                expected: data(&blob),
                actual: data(&other_blob),
            })
        );

        let invalid = raw_archive(vec![row(2, 1, 2), row(1, 1, 1)]);
        let invalid_commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            descriptor.handle(),
            data(&invalid),
            empty_metadata_handle(),
        );
        assert_eq!(
            validate_commit(&descriptor, &invalid_commit, &invalid),
            Err(SimpleArchiveUnionValidationError::InvalidElement {
                role: ElementRole::CommitData,
                source: UnarchiveError::BadCanonicalizationOrdering,
            })
        );
    }

    #[test]
    fn merge_validation_is_exact_and_binds_every_endpoint() {
        let descriptor = descriptor(id(1));
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let result = join(&left, &right).unwrap();
        let claim = CollectionMerge::new(
            descriptor.handle(),
            data(&left),
            data(&right),
            data(&result),
        );
        let (low, high) = ordered_inputs(&left, &right);
        validate_merge(&descriptor, &claim, low, high, &result).unwrap();

        let wrong_collection = CollectionMerge::new(
            super::descriptor(id(9)).handle(),
            data(low),
            data(high),
            data(&result),
        );
        assert!(matches!(
            validate_merge(&descriptor, &wrong_collection, low, high, &result),
            Err(SimpleArchiveUnionValidationError::WrongCollection { .. })
        ));

        assert!(matches!(
            validate_merge(&descriptor, &claim, high, low, &result),
            Err(SimpleArchiveUnionValidationError::EndpointMismatch {
                role: ElementRole::MergeLow,
                ..
            })
        ));

        let forged_high = Blob::with_handle(low.bytes.clone(), high.get_handle());
        assert_eq!(
            validate_merge(&descriptor, &claim, low, &forged_high, &result),
            Err(SimpleArchiveUnionValidationError::EndpointMismatch {
                role: ElementRole::MergeHigh,
                expected: data(high),
                actual: data(low),
            })
        );

        let other_result = archive([row(4, 1, 4)]);
        assert!(matches!(
            validate_merge(&descriptor, &claim, low, high, &other_result),
            Err(SimpleArchiveUnionValidationError::EndpointMismatch {
                role: ElementRole::MergeResult,
                ..
            })
        ));

        let wrong_result = archive([row(1, 1, 1), row(2, 1, 2)]);
        let wrong_claim = CollectionMerge::new(
            descriptor.handle(),
            data(low),
            data(high),
            data(&wrong_result),
        );
        assert_eq!(
            validate_merge(&descriptor, &wrong_claim, low, high, &wrong_result),
            Err(SimpleArchiveUnionValidationError::WrongMergeResult)
        );

        let invalid_result = raw_archive(vec![row(2, 1, 2), row(1, 1, 1)]);
        let invalid_claim = CollectionMerge::new(
            descriptor.handle(),
            data(low),
            data(high),
            data(&invalid_result),
        );
        assert_eq!(
            validate_merge(&descriptor, &invalid_claim, low, high, &invalid_result),
            Err(SimpleArchiveUnionValidationError::InvalidElement {
                role: ElementRole::MergeResult,
                source: UnarchiveError::BadCanonicalizationOrdering,
            })
        );
    }

    #[cfg(feature = "proptest")]
    mod property_tests {
        use super::*;

        use proptest::collection::vec;
        use proptest::prelude::*;

        fn arb_trible() -> impl Strategy<Value = Trible> {
            (
                prop::array::uniform16(1_u8..=255),
                prop::array::uniform16(1_u8..=255),
                prop::array::uniform32(any::<u8>()),
            )
                .prop_map(|(entity, attribute, value)| {
                    let mut raw = [0; TRIBLE_LEN];
                    raw[..16].copy_from_slice(&entity);
                    raw[16..32].copy_from_slice(&attribute);
                    raw[32..].copy_from_slice(&value);
                    Trible::force_raw(raw).unwrap()
                })
        }

        fn arb_set(max: usize) -> impl Strategy<Value = TribleSet> {
            vec(arb_trible(), 0..max).prop_map(|tribles| {
                let mut set = TribleSet::new();
                for trible in &tribles {
                    set.insert(trible);
                }
                set
            })
        }

        proptest! {
            #[test]
            fn direct_union_matches_the_patch_oracle(
                left in arb_set(64),
                right in arb_set(64),
            ) {
                let expected: Blob<SimpleArchive> = (left.clone() + right.clone()).to_blob();
                let left: Blob<SimpleArchive> = left.to_blob();
                let right: Blob<SimpleArchive> = right.to_blob();
                let actual = join(&left, &right).unwrap();

                prop_assert_eq!(&actual, &expected);
                let collection = descriptor(id(1));
                let claim = CollectionMerge::new(
                    collection.handle(),
                    data(&left),
                    data(&right),
                    data(&actual),
                );
                let (low, high) = ordered_inputs(&left, &right);
                prop_assert!(validate_merge(&collection, &claim, low, high, &actual).is_ok());
                prop_assert_eq!(actual, join(&right, &left).unwrap());
            }

            #[test]
            fn direct_union_obeys_identity_and_aci(
                a in arb_set(32),
                b in arb_set(32),
                c in arb_set(32),
            ) {
                let empty: Blob<SimpleArchive> = TribleSet::new().to_blob();
                let a: Blob<SimpleArchive> = a.to_blob();
                let b: Blob<SimpleArchive> = b.to_blob();
                let c: Blob<SimpleArchive> = c.to_blob();

                prop_assert_eq!(join(&empty, &a).unwrap(), a.clone());
                prop_assert_eq!(join(&a, &empty).unwrap(), a.clone());
                prop_assert_eq!(join(&a, &a).unwrap(), a.clone());
                prop_assert_eq!(join(&a, &b).unwrap(), join(&b, &a).unwrap());

                let left_associated = join(&join(&a, &b).unwrap(), &c).unwrap();
                let right_associated = join(&a, &join(&b, &c).unwrap()).unwrap();
                prop_assert_eq!(left_associated, right_associated);
            }
        }
    }
}
