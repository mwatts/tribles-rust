//! Canonical TribleSet set union over
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive)
//! elements.
//!
//! This is the first concrete production collection kind. A collection pairs
//! a name within a public-key namespace with the existing `SimpleArchive`
//! representation and the
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

use crate::id::ExclusiveId;
// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use crate::collection::reach;
use crate::metadata;
use crate::prelude::entity;
use ed25519_dalek::VerifyingKey;

use super::records::{
    collection_authority, collection_name, collection_namespace, collection_reach,
    collection_recipe, collection_representation, CollectionName, RecordDecodeError,
    KIND_COLLECTION_DESCRIPTOR,
};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use anybytes::{Bytes, View};
use ed25519_dalek::SigningKey;

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::encodings::UnknownBlob;
use crate::blob::Blob;
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::Inline;
use crate::metadata::MetaDescribe;
use crate::repo::{BlobStore, BlobStoreGet, BlobStorePut};
use crate::trible::{Fragment, Trible, TRIBLE_LEN};

use super::descriptor as descriptor_facts;
use super::{
    CollectionCommit, CollectionData, CollectionHandle, CollectionMerge, CollectionRecord,
    CollectionStore,
};

mod collection;
mod materialize;
pub use collection::*;
pub use materialize::*;

/// Canonical TribleSet set-union recipe, version 1.
///
/// This identifies the semantic law independently of its direct-stream
/// implementation and of the collection's blob representation. Minted with
/// `trible genid` on 2026-08-07.
pub const TRIBLE_SET_UNION_RECIPE_V1: Id = id_hex!("6D64C5F4B9E9B73F57C5F8702AB7FE45");

/// The TribleSet set-union law, as a describable type.
///
/// A descriptor embeds this description rather than only the id above, so a
/// reader holding the pile can learn what the law is without the code that
/// minted it.
pub struct TribleSetUnionV1;

impl MetaDescribe for TribleSetUnionV1 {
    fn describe() -> Fragment {
        let id: Id = TRIBLE_SET_UNION_RECIPE_V1;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "trible-set-union-v1",
                metadata::description: "Set union of the tribles carried by a collection's elements. Associative, commutative and idempotent, so any two states have a least upper bound and merging is order-independent: a collection's value is the union over every element committed to it, and two replicas that have seen the same elements agree regardless of the order they arrived in. Takes no arguments.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

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
    /// The descriptor does not carry a field this check needs.
    Malformed(RecordDecodeError),
    /// The descriptor names another blob representation.
    WrongRepresentation { expected: Id, actual: Id },
    /// The descriptor names another semantic recipe.
    WrongRecipe { expected: Id, actual: Id },
    /// The record belongs to another collection descriptor.
    WrongCollection {
        expected: CollectionHandle,
        actual: CollectionHandle,
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
            Self::Malformed(error) => write!(f, "malformed collection descriptor: {error}"),
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
    /// An element, result, metadata, or embedded-attachment write failed.
    DependencyPut(PutError),
    /// The final commit or merge record could not be admitted to the native record store.
    RecordInsert(InsertError),
    /// A merge names an input this store does not hold.
    ///
    /// A merge is an equation between two states *of a collection*, and a
    /// state is resident because the commit that made it one put it there. An
    /// absent input means the equation is about something that was never
    /// committed, so the record must not be published.
    MergeInputAbsent {
        /// Which side of the merge was missing.
        role: ElementRole,
        /// The identity that could not be read.
        data: CollectionData,
    },
}

/// Validation failure while preparing a canonical collection commit in memory.
///
/// The uninhabited storage error parameters make it impossible for this phase
/// to report an I/O failure: preparation does not touch the destination store.
pub type PreparationError = PublicationError<Infallible, Infallible>;

/// A canonical collection commit whose bytes have not been published.
///
/// Preparation validates and normalizes the descriptor, data, metadata, and
/// embedded fragment blobs entirely in memory, without touching any store and
/// without needing a signing key. Call [`Self::stage`] to write every
/// dependency and sign the resulting commit over the handles the store itself
/// returned. Dropping a prepared value has no storage effect.
#[derive(Clone, Debug)]
#[must_use = "a prepared collection commit has no effect until it is staged and finalized"]
pub struct PreparedCollectionCommit {
    embedded: Vec<Blob<UnknownBlob>>,
    descriptor: Fragment,
    data: Blob<SimpleArchive>,
    metadata: Blob<SimpleArchive>,
}

impl PreparedCollectionCommit {
    /// Stage every dependency and sign the commit over the stored handles.
    ///
    /// The exact store-call order is the collection descriptor blob,
    /// embedded blobs (in handle order), data, and metadata. Every handle the
    /// signed commit names is a handle one of those writes handed back, so the
    /// commit's whole dependency closure is present by construction rather
    /// than by two independent hash computations happening to agree.
    ///
    /// On success the returned value retains the same mutable store borrow, so
    /// a caller may append unsigned `MERGE` or `DERIVE` artifacts through
    /// [`StagedCollectionCommit::store_mut`] before consuming the value with
    /// [`StagedCollectionCommit::finalize`].
    pub fn stage<'store, S>(
        self,
        store: &'store mut S,
        signing_key: &SigningKey,
    ) -> Result<StagedCollectionCommit<'store, S>, PublicationError<S::PutError, S::InsertError>>
    where
        S: BlobStorePut + CollectionStore,
    {
        let Self {
            embedded,
            descriptor,
            data,
            metadata,
        } = self;

        let collection: CollectionHandle = store
            .put::<SimpleArchive, _>(crate::blob::IntoBlob::<SimpleArchive>::to_blob(
                descriptor.into_facts(),
            ))
            .map_err(PublicationError::DependencyPut)?;
        for blob in embedded {
            store
                .put::<UnknownBlob, _>(blob)
                .map_err(PublicationError::DependencyPut)?;
        }
        let data_handle = store
            .put::<SimpleArchive, _>(data)
            .map_err(PublicationError::DependencyPut)?;
        let metadata_handle = store
            .put::<SimpleArchive, _>(metadata)
            .map_err(PublicationError::DependencyPut)?;

        let commit = CollectionCommit::sign(
            signing_key,
            collection,
            Handle::<SimpleArchive>::to_hash(data_handle),
            metadata_handle,
        );
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
            Self::DependencyPut(error) => {
                write!(f, "failed to write a collection dependency: {error}")
            }
            Self::RecordInsert(error) => write!(f, "failed to insert collection record: {error}"),
            Self::MergeInputAbsent { role, data } => write!(
                f,
                "the {role} is not in this store, so the merge is over a state \
                 that was never committed: {data:?}"
            ),
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
            Self::DependencyPut(error) => Some(error),
            Self::RecordInsert(error) => Some(error),
            Self::MergeInputAbsent { .. } => None,
        }
    }
}

/// Describe this collection kind as a root named within a namespace.
///
/// This is the one home for what a `SimpleArchive` set-union collection *is*:
/// that representation, that recipe. Everything else about a particular
/// collection -- which one it is -- is the name and namespace passed in.
/// `authority` is an optional capability trust root with a distinct semantic
/// role; when present, it is nevertheless an identity-bearing descriptor fact.
///
/// It returns the facts, not a handle. Getting a handle means putting the
/// blob, and `put` gives you the handle back, so a stored descriptor is a
/// side effect of naming one rather than a second thing to remember. Hashing
/// a descriptor you never stored would leave a phantom collection: records
/// that reference it, and nothing that can decode what they reference.
pub fn descriptor(
    name: &CollectionName,
    namespace: VerifyingKey,
    authority: Option<VerifyingKey>,
    reach: Fragment,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.as_str(),
        collection_namespace: namespace,
        collection_authority?: authority,
        collection_representation*: <SimpleArchive as MetaDescribe>::describe(),
        collection_recipe*: <TribleSetUnionV1 as MetaDescribe>::describe(),
        collection_reach*: reach,
    }
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

/// Compute one canonical union over many `SimpleArchive` elements.
///
/// Each input is validated before output construction. The error carries the
/// zero-based input position so callers can retain the identity of a malformed
/// member. A heap merge keeps one current row per input and writes the result
/// once, avoiding the intermediate archives produced by repeated two-way
/// joins.
pub(crate) fn join_many<'a>(
    elements: impl IntoIterator<Item = &'a Blob<SimpleArchive>>,
) -> Result<Blob<SimpleArchive>, (usize, UnarchiveError)> {
    Ok(Blob::new(join_many_bytes(elements)?))
}

/// Compute the canonical union's bytes without naming them.
///
/// Same output as [`join_many`], minus the Blake3 pass that turns bytes into a
/// handle. Naming a blob is a separate act from computing one, and a caller
/// that decodes the union and drops it — `snapshot_from_observation` does —
/// pays a fifth of the merge for a handle it never reads.
pub(crate) fn join_many_bytes<'a>(
    elements: impl IntoIterator<Item = &'a Blob<SimpleArchive>>,
) -> Result<Bytes, (usize, UnarchiveError)> {
    let elements: Vec<_> = elements.into_iter().collect();
    let mut element_rows = Vec::with_capacity(elements.len());
    for (index, element) in elements.iter().enumerate() {
        element_rows.push(canonical_rows(element).map_err(|source| (index, source))?);
    }

    match elements.as_slice() {
        [] => return Ok(Bytes::from(Vec::<[u8; TRIBLE_LEN]>::new())),
        // Canonical bytes are already the union of themselves; `join_many`
        // renames them, which is what `normalize_blob` did here before.
        [element] => return Ok(element.bytes.clone()),
        _ => {}
    }

    let slices: Vec<&[[u8; TRIBLE_LEN]]> = element_rows.iter().map(|rows| &rows[..]).collect();

    #[cfg(feature = "parallel")]
    {
        if let Some(union) = parallel_merge_canonical(&slices) {
            return Ok(Bytes::from(union));
        }
    }

    Ok(Bytes::from(merge_canonical_range(&slices, None, None)))
}

/// Heap-merge the rows of every input that fall in `[low, high)`.
///
/// `None` bounds are open. Inputs are canonical — sorted and distinct — so one
/// pass with one live row per input emits the range's union in order, and the
/// only duplicates it can see are equal rows from different inputs.
fn merge_canonical_range(
    slices: &[&[[u8; TRIBLE_LEN]]],
    low: Option<&[u8; TRIBLE_LEN]>,
    high: Option<&[u8; TRIBLE_LEN]>,
) -> Vec<[u8; TRIBLE_LEN]> {
    let mut cursors = Vec::with_capacity(slices.len());
    let mut capacity = 0usize;
    for rows in slices {
        let start = match low {
            Some(low) => rows.partition_point(|row| row < low),
            None => 0,
        };
        let end = match high {
            Some(high) => rows.partition_point(|row| row < high),
            None => rows.len(),
        };
        capacity = capacity.saturating_add(end.saturating_sub(start));
        cursors.push((start, end));
    }

    let mut union = Vec::with_capacity(capacity);
    let mut heap = BinaryHeap::with_capacity(slices.len());
    for (element, (start, end)) in cursors.iter().copied().enumerate() {
        if start < end {
            heap.push(Reverse((slices[element][start], element, start)));
        }
    }

    let mut previous = None;
    while let Some(Reverse((row, element, index))) = heap.pop() {
        if previous != Some(row) {
            union.push(row);
            previous = Some(row);
        }
        let next = index + 1;
        if next < cursors[element].1 {
            heap.push(Reverse((slices[element][next], element, next)));
        }
    }
    union
}

/// Rows below which a partitioned merge is not worth its splitter search.
#[cfg(feature = "parallel")]
const PARALLEL_MERGE_THRESHOLD: usize = 1 << 16;

/// Merge canonical inputs by disjoint key range, one range per worker.
///
/// The serial heap merge is one thread deciding 26 M times which of 404
/// streams is next, and that decision is exactly what a key range makes
/// independent: partitions cover disjoint key intervals, so each worker's
/// output is a complete, deduplicated, sorted run and concatenating the runs in
/// range order reproduces the serial result byte for byte. Splitters are chosen
/// by regular sampling, so they affect balance only — never the output.
///
/// Returns `None` when the input is too small to be worth partitioning, leaving
/// the caller on the serial path.
#[cfg(feature = "parallel")]
fn parallel_merge_canonical(slices: &[&[[u8; TRIBLE_LEN]]]) -> Option<Vec<[u8; TRIBLE_LEN]>> {
    use rayon::prelude::*;

    let total: usize = slices.iter().map(|rows| rows.len()).sum();
    let workers = rayon::current_num_threads();
    if total < PARALLEL_MERGE_THRESHOLD || workers < 2 {
        return None;
    }

    // Regular sampling: every input contributes candidates at even offsets, so
    // one huge member cannot alone decide the cut points and a skewed member
    // cannot hide a dense key range from the sample.
    let per_input = workers.min(64);
    let mut samples = Vec::with_capacity(slices.len().saturating_mul(per_input));
    for rows in slices {
        if rows.is_empty() {
            continue;
        }
        for step in 1..per_input {
            let index = rows.len().saturating_mul(step) / per_input;
            if index < rows.len() {
                samples.push(rows[index]);
            }
        }
    }
    samples.sort_unstable();
    samples.dedup();
    if samples.is_empty() {
        return None;
    }

    let cuts = workers.min(samples.len() + 1);
    let mut splitters = Vec::with_capacity(cuts.saturating_sub(1));
    for step in 1..cuts {
        let index = samples.len().saturating_mul(step) / cuts;
        let candidate = samples[index.min(samples.len() - 1)];
        if splitters.last() != Some(&candidate) {
            splitters.push(candidate);
        }
    }
    if splitters.is_empty() {
        return None;
    }

    let mut bounds: Vec<(Option<[u8; TRIBLE_LEN]>, Option<[u8; TRIBLE_LEN]>)> =
        Vec::with_capacity(splitters.len() + 1);
    let mut low = None;
    for splitter in splitters {
        bounds.push((low, Some(splitter)));
        low = Some(splitter);
    }
    bounds.push((low, None));

    let runs: Vec<Vec<[u8; TRIBLE_LEN]>> = bounds
        .par_iter()
        .map(|(low, high)| merge_canonical_range(slices, low.as_ref(), high.as_ref()))
        .collect();

    let mut union = Vec::with_capacity(runs.iter().map(Vec::len).sum());
    for run in runs {
        union.extend_from_slice(&run);
    }
    Some(union)
}

/// Validate a discovered commit as one canonical root of this collection.
///
/// This binds the concrete descriptor, record collection, endpoint identity,
/// and element bytes in one check. The record's strict self-signature and the
/// caller's authorization policy remain separate admission prerequisites.
pub fn validate_commit(
    descriptor: &Fragment,
    commit: &CollectionCommit,
    data_blob: &Blob<SimpleArchive>,
) -> Result<(), SimpleArchiveUnionValidationError> {
    validate_descriptor(descriptor)?;
    let collection: CollectionHandle =
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle();
    validate_collection(collection, commit.collection())?;
    validate_endpoint(ElementRole::CommitData, commit.data(), data_blob)?;
    Ok(())
}

/// Validate a claimed exact union without materializing another result blob.
///
/// All endpoints are first bound to their record hashes and validated as
/// canonical archives. The expected two-way union is then compared row-for-row
/// with `result`, using constant auxiliary space.
pub fn validate_merge(
    descriptor: &Fragment,
    claim: &CollectionMerge,
    low: &Blob<SimpleArchive>,
    high: &Blob<SimpleArchive>,
    result: &Blob<SimpleArchive>,
) -> Result<(), SimpleArchiveUnionValidationError> {
    validate_descriptor(descriptor)?;
    let collection: CollectionHandle =
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle();
    validate_collection(collection, claim.collection())?;

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

/// Prepare a canonical membership root entirely in memory.
///
/// Supplied data and metadata are normalized from their bytes before either is
/// validated, so a forged [`Blob::with_handle`] cache cannot enter storage or
/// determine the commit identity. No store is touched and no key is needed:
/// the commit is signed by [`PreparedCollectionCommit::stage`] over the
/// handles the store returns. The returned value can be staged, abandoned
/// inertly, or finalized later.
pub fn prepare_commit(
    descriptor: &Fragment,
    data: &Blob<SimpleArchive>,
    metadata: &Blob<SimpleArchive>,
) -> Result<PreparedCollectionCommit, PreparationError> {
    prepare_commit_with_embedded(descriptor, data, metadata, Vec::new())
}

/// Prepare a self-contained fact fragment as a canonical membership root.
///
/// The fragment's facts become collection data and its metafacts become commit
/// metadata. Its one shared blob store may back handles in either set, and its
/// blobs are staged under the identities they already carry: bytes are hashed
/// where they enter a trust boundary, and an in-memory fragment we just built
/// is not such a boundary. Fragment exports are not serialized. No destination
/// store is touched.
pub fn prepare_fragment_commit(
    descriptor: &Fragment,
    fragment: Fragment,
) -> Result<PreparedCollectionCommit, PreparationError> {
    let (_, facts, metafacts, mut blobs) = fragment.into_parts();

    // The sort key is the identity `put` will file each blob under, so the
    // staged order is the documented handle order.
    let mut embedded: Vec<Blob<UnknownBlob>> = blobs
        .reader()
        .expect("MemoryBlobStore::reader is infallible")
        .into_iter()
        .map(|(_, blob)| blob)
        .collect();
    embedded.sort_unstable_by_key(|blob| blob.get_handle().raw);

    let data: Blob<SimpleArchive> = crate::blob::IntoBlob::to_blob(facts);
    let metadata: Blob<SimpleArchive> = crate::blob::IntoBlob::to_blob(metafacts);
    prepare_commit_with_embedded(descriptor, &data, &metadata, embedded)
}

fn prepare_commit_with_embedded(
    descriptor: &Fragment,
    data: &Blob<SimpleArchive>,
    metadata: &Blob<SimpleArchive>,
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

    Ok(PreparedCollectionCommit {
        embedded,
        descriptor: descriptor.clone(),
        data,
        metadata,
    })
}

fn widen_preparation_error<PutError, InsertError>(
    error: PreparationError,
) -> PublicationError<PutError, InsertError> {
    match error {
        PublicationError::Validation(error) => PublicationError::Validation(error),
        PublicationError::InvalidMetadata(error) => PublicationError::InvalidMetadata(error),
        PublicationError::DependencyPut(never) => match never {},
        PublicationError::RecordInsert(never) => match never {},
        // Preparation touches no store, so it cannot find an input missing
        // from one. The variant is unreachable here rather than merely unused.
        PublicationError::MergeInputAbsent { role, data } => {
            PublicationError::MergeInputAbsent { role, data }
        }
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
    descriptor: &Fragment,
    data: &Blob<SimpleArchive>,
    metadata: &Blob<SimpleArchive>,
    signing_key: &SigningKey,
) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStorePut + CollectionStore,
{
    let prepared = prepare_commit(descriptor, data, metadata).map_err(widen_preparation_error)?;
    prepared.stage(store, signing_key)?.finalize()
}

/// Publish a self-contained fact fragment as a signed membership root.
///
/// The fragment's facts and metafacts become the signed data and metadata
/// elements, and its shared blob store backs handles in either set. Embedded
/// blobs are staged under the identities they already carry.
///
/// Fragment exports are not serialized. The two fact sets become canonical
/// `SimpleArchive` data and metadata elements. The shared prepared-publication
/// path writes the descriptor blob, embedded blobs, and both
/// archives before inserting the signed record last.
/// The same backend-recovery boundary documented by [`PublicationError`]
/// applies.
pub fn publish_fragment_commit<S>(
    store: &mut S,
    descriptor: &Fragment,
    fragment: Fragment,
    signing_key: &SigningKey,
) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStorePut + CollectionStore,
{
    let prepared =
        prepare_fragment_commit(descriptor, fragment).map_err(widen_preparation_error)?;
    prepared.stage(store, signing_key)?.finalize()
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
/// Read one merge input out of the store.
///
/// A merge input is a state of the collection, so it is resident by
/// construction -- and if it is not, the merge is over something that was
/// never committed and must fail here rather than be published.
fn fetch_merge_input<S>(
    reader: &S::Reader,
    role: ElementRole,
    data: CollectionData,
) -> Result<Blob<SimpleArchive>, PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStore + CollectionStore,
{
    reader
        .get::<Blob<SimpleArchive>, _>(Handle::<SimpleArchive>::from_hash(data))
        .map_err(|_| PublicationError::MergeInputAbsent { role, data })
}

pub fn publish_merge<S>(
    store: &mut S,
    descriptor: &Fragment,
    low: CollectionData,
    high: CollectionData,
) -> Result<(CollectionMerge, Blob<SimpleArchive>), PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStore + CollectionStore,
{
    validate_descriptor(descriptor).map_err(PublicationError::Validation)?;

    // A merge names two states OF a collection, so both are already in the
    // store -- the commits that made them states put them there. Taking them
    // by handle rather than by value says so: an input that is not resident
    // cannot be named, because nothing can be computed without reading it.
    //
    // Taking them by value could not say that. It let a caller publish a MERGE
    // over two blobs nobody had ever committed -- a record asserting an
    // equation between states the collection does not contain. It also wrote
    // them a second time, and re-normalized bytes that were normalized on the
    // way in.
    let reader = store
        .reader()
        .map_err(|_| PublicationError::MergeInputAbsent {
            role: ElementRole::MergeLow,
            data: low,
        })?;
    let (low_data, low) = (
        low,
        fetch_merge_input::<S>(&reader, ElementRole::MergeLow, low)?,
    );
    let (high_data, high) = (
        high,
        fetch_merge_input::<S>(&reader, ElementRole::MergeHigh, high)?,
    );
    drop(reader);

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

    // Write first, then name what was written. The descriptor used to be
    // serialized and hashed twice -- once to name the collection, once to
    // store it -- and the record was built from the first while the store
    // received the second. They cannot disagree, but only because nothing
    // arranged for them to; taking the handle from `put` makes "this record
    // names blobs this store holds" true by construction.
    //
    // Nothing here needs an identity before its write. This function used to
    // hash both inputs first and swap them into digest order, which looked
    // load-bearing and was not: `CollectionMerge::new` sorts its own inputs by
    // digest, and `join_canonical_rows` is a union over sorted rows whose
    // early-outs are symmetric, so neither the record nor the result can tell
    // which argument arrived first.
    let collection: CollectionHandle = store
        .put::<SimpleArchive, _>(descriptor.facts().clone())
        .map_err(PublicationError::DependencyPut)?;
    let result_data = store
        .put::<SimpleArchive, _>(result.clone())
        .map_err(PublicationError::DependencyPut)?;

    let merge = CollectionMerge::new(
        collection,
        low_data,
        high_data,
        Handle::<SimpleArchive>::to_hash(result_data),
    );
    store
        .insert(CollectionRecord::Merge(merge))
        .map_err(PublicationError::RecordInsert)?;

    Ok((merge, result))
}

fn validate_descriptor(descriptor: &Fragment) -> Result<(), SimpleArchiveUnionValidationError> {
    let expected_representation = <SimpleArchive as MetaDescribe>::id();
    let representation = descriptor_facts::representation(descriptor.facts())?;
    if representation != expected_representation {
        return Err(SimpleArchiveUnionValidationError::WrongRepresentation {
            expected: expected_representation,
            actual: representation,
        });
    }
    let recipe = descriptor_facts::recipe(descriptor.facts())?;
    if recipe != TRIBLE_SET_UNION_RECIPE_V1 {
        return Err(SimpleArchiveUnionValidationError::WrongRecipe {
            expected: TRIBLE_SET_UNION_RECIPE_V1,
            actual: recipe,
        });
    }
    Ok(())
}

fn validate_collection(
    expected: CollectionHandle,
    actual: CollectionHandle,
) -> Result<(), SimpleArchiveUnionValidationError> {
    if actual != expected {
        return Err(SimpleArchiveUnionValidationError::WrongCollection { expected, actual });
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

    use crate::blob::encodings::rawbytes::RawBytes;
    use crate::blob::encodings::utf8string::UTF8String;
    use crate::blob::{BlobEncoding, IntoBlob};
    use crate::collection::descriptor::identity_for_tests;
    use crate::collection::records::CollectionName;
    use crate::collection::{
        discover_collection_records, empty_metadata_handle, plan_collection_retention,
        resolve_collection_semantics, CollectionClaimValidation, CollectionDerive,
    };
    use crate::inline::InlineEncoding;
    use crate::macros::entity;
    use crate::repo::pile::Pile;
    use crate::repo::{BlobStore, BlobStoreGet};
    use crate::trible::TribleSet;

    /// The one team every collection in these tests belongs to.
    fn test_team() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    /// One named root of this collection kind.
    fn root(name: &str) -> Fragment {
        super::descriptor(
            &CollectionName::new(name).unwrap(),
            test_team(),
            Some(test_team()),
            reach::private(),
        )
    }

    /// The same anchor as `root("first")`, but naming a different
    /// representation or recipe: a different collection of a shape this kind
    /// does not accept.
    fn test_naming(representation: Id, recipe: Id) -> Fragment {
        crate::collection::descriptor::naming(
            &CollectionName::new("first").unwrap(),
            test_team(),
            Some(test_team()),
            representation,
            recipe,
            reach::private(),
        )
    }

    mod fragment_ns {
        use crate::prelude::*;

        attributes! {
            // Test-only sentinel attributes; these are not protocol ids.
            "DD00000000000000DD00000000000031" unsafe as pub text: inlineencodings::Handle<blobencodings::UTF8String>;
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
        // The probe records the *sequence* of store operations; the bytes are
        // delegated so it can also be read from. A merge takes its inputs by
        // handle now, so a store that cannot be read cannot be merged into --
        // which is the point of that change, and this is the probe catching up
        // with it rather than a convenience.
        blobs: crate::blob::MemoryBlobStore,
    }

    impl ProbeStore {
        // This probe fails before an operation takes effect, so it exercises
        // publication ordering at trait-operation boundaries. BlobStorePut
        // does not promise that a real backend cannot leave torn physical I/O.
        /// Make a blob resident without recording an operation.
        ///
        /// A merge's inputs are states the collection already holds, put there
        /// by the commits that made them states. Those puts are not part of
        /// the merge, so seeding them here keeps the recorded sequence about
        /// what the merge itself writes.
        fn seed(&mut self, blob: &Blob<SimpleArchive>) {
            self.known.insert(blob.get_handle().raw);
            self.blobs.insert(blob.clone());
        }

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

    impl crate::repo::BlobStore for ProbeStore {
        type Reader = <crate::blob::MemoryBlobStore as crate::repo::BlobStore>::Reader;
        type ReaderError = <crate::blob::MemoryBlobStore as crate::repo::BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.blobs.reader()
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
            self.blobs.insert(blob.clone());
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
        Inline<Handle<UTF8String>>,
        Inline<Handle<RawBytes>>,
    ) {
        let text: Blob<UTF8String> = String::from("a self-contained content blob").to_blob();
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
        Fragment,
        Blob<SimpleArchive>,
        Blob<SimpleArchive>,
        SigningKey,
        CollectionCommit,
    ) {
        let descriptor = root("first");
        let data_blob = archive([row(1, 1, 1), row(3, 1, 3)]);
        let metadata = archive([row(9, 1, 9)]);
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let commit = CollectionCommit::sign(
            &signing_key,
            identity_for_tests(&descriptor),
            data(&data_blob),
            metadata.get_handle(),
        );
        (descriptor, data_blob, metadata, signing_key, commit)
    }

    #[test]
    fn prepared_fragment_is_canonical_idempotent_and_commits_after_caller_artifacts() {
        let source_descriptor = root("first");
        let target = root("second");
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, _text_handle, _payload_handle) = fragment_fixture();
        let embedded = embedded_put_events(&fragment);
        let content_archive: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let metadata_archive: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();
        let expected = CollectionCommit::sign(
            &signing_key,
            identity_for_tests(&source_descriptor),
            data(&content_archive),
            metadata_archive.get_handle(),
        );

        let prepared = prepare_fragment_commit(&source_descriptor, fragment.clone()).unwrap();
        let repeated = prepare_fragment_commit(&source_descriptor, fragment).unwrap();

        let derive = CollectionDerive::new(
            identity_for_tests(&target),
            expected.data(),
            Inline::new([0x42; 32]),
        );
        let derive_record = CollectionRecord::Derive(derive);
        let commit_record = CollectionRecord::Commit(expected);
        let sequence = [
            vec![put_event(&IntoBlob::<SimpleArchive>::to_blob(
                source_descriptor.facts().clone(),
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
        let mut signed = Vec::new();
        for prepared in [prepared, repeated] {
            let mut staged = prepared.stage(&mut store, &signing_key).unwrap();
            assert_eq!(staged.commit(), &expected);
            signed.push(*staged.commit());
            staged.store_mut().insert(derive_record).unwrap();
            assert_eq!(staged.finalize().unwrap(), expected);
        }
        assert_eq!(signed[0].id(), signed[1].id());
        assert_eq!(signed[0].to_bytes(), signed[1].to_bytes());

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

        let descriptor = root("first");
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, text_handle, payload_handle) = fragment_fixture();
        let expected_content: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let expected_metadata: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();
        let prepared = prepare_fragment_commit(&descriptor, fragment).unwrap();

        let mut pile = Pile::open(&path).unwrap();
        let mut staged = prepared.stage(&mut pile, &signing_key).unwrap();
        let withheld = *staged.commit();
        {
            let discovered = discover_collection_records(staged.store_mut()).unwrap();
            let reader = staged.store_mut().reader().unwrap();
            assert!(discovered.commits().is_empty());
            assert!(discovered.merges().is_empty());
            assert!(discovered.derives().is_empty());
            let descriptor_blob: Blob<SimpleArchive> =
                reader.get(identity_for_tests(&descriptor)).unwrap();
            assert_eq!(
                <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(
                    descriptor_blob
                )
                .unwrap(),
                *descriptor.facts()
            );

            let resolution = resolve_collection_semantics(
                &discovered,
                &std::collections::BTreeMap::new(),
                &BTreeSet::new(),
                |_| Ok::<_, Infallible>(CollectionClaimValidation::<()>::Pending),
            )
            .unwrap();
            assert!(resolution.admitted_claims().is_empty());
            assert!(resolution
                .semantics()
                .members(identity_for_tests(&descriptor))
                .is_none());
            let roots = plan_collection_retention(&discovered, &resolution, &reader).unwrap();
            assert!(roots.is_empty());
            assert!(roots.expanded(&reader).is_empty());

            let content: Blob<SimpleArchive> = reader
                .get::<Blob<SimpleArchive>, SimpleArchive>(withheld.data().transmute())
                .unwrap();
            let metadata: Blob<SimpleArchive> = reader.get(withheld.metadata()).unwrap();
            let text: View<str> = reader.get::<View<str>, UTF8String>(text_handle).unwrap();
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
        let descriptor_blob: Blob<SimpleArchive> =
            reader.get(identity_for_tests(&descriptor)).unwrap();
        assert_eq!(
            <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(descriptor_blob)
                .unwrap(),
            *descriptor.facts()
        );
        drop(reader);
        reopened.close().unwrap();
    }

    #[test]
    fn fragment_without_metafacts_still_stages_the_canonical_empty_metadata_archive() {
        let descriptor = root("first");
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let empty_archive: Blob<SimpleArchive> = TribleSet::new().to_blob();

        let prepared = prepare_fragment_commit(&descriptor, Fragment::empty()).unwrap();

        assert_eq!(prepared.metadata.get_handle(), empty_metadata_handle());
        assert_eq!(prepared.metadata, empty_archive);

        let mut store = ProbeStore::default();
        let staged = prepared.stage(&mut store, &signing_key).unwrap();
        assert_eq!(staged.commit().metadata(), empty_metadata_handle());
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
            put_event(&IntoBlob::<SimpleArchive>::to_blob(
                descriptor.facts().clone(),
            )),
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
            identity_for_tests(&descriptor).raw,
            data_blob.get_handle().raw,
            metadata.get_handle().raw,
        ]);
        assert_eq!(store.known, expected_handles);
        assert_eq!(
            store.records.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([expected.id()])
        );
    }

    #[test]
    fn fragment_commit_puts_embedded_dependencies_before_record_and_replays_idempotently() {
        let descriptor = root("first");
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let (fragment, _text_handle, _payload_handle) = fragment_fixture();
        let embedded = embedded_put_events(&fragment);
        let content_archive: Blob<SimpleArchive> = fragment.facts().clone().to_blob();
        let metadata_archive: Blob<SimpleArchive> = fragment.metafacts().clone().to_blob();
        let expected = CollectionCommit::sign(
            &signing_key,
            identity_for_tests(&descriptor),
            data(&content_archive),
            metadata_archive.get_handle(),
        );
        let sequence = [
            vec![put_event(&IntoBlob::<SimpleArchive>::to_blob(
                descriptor.facts().clone(),
            ))],
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
            identity_for_tests(&descriptor).raw,
            content_archive.get_handle().raw,
            metadata_archive.get_handle().raw,
        ]);
        assert_eq!(store.known, expected_handles);
        assert_eq!(
            store.records.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([expected.id()])
        );
    }

    /// A merge writes its result and nothing else, and says the same thing
    /// twice.
    ///
    /// It used to normalize the two inputs, because it received them as bytes
    /// and a caller could hand it anything. It receives them as the identities
    /// of states the collection already holds, so there is nothing left to
    /// normalize: the bytes come from the store, and a state was canonicalized
    /// by the commit that made it one. The forged-input case this test used to
    /// carry moved with that responsibility -- an input that is not resident is
    /// now refused by identity, which `a_merge_refuses_an_input_the_store_does_not_hold`
    /// pins.
    #[test]
    fn merge_publication_writes_only_its_result_and_replays_idempotently() {
        let descriptor = root("first");
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let (low, high) = ordered_inputs(&left, &right);
        let expected_result = join(low, high).unwrap();
        let expected_merge = CollectionMerge::new(
            identity_for_tests(&descriptor),
            data(low),
            data(high),
            data(&expected_result),
        );
        // Three operations, not five. The inputs are already states of the
        // collection, so the merge writes only what it computed.
        let sequence = vec![
            put_event(&IntoBlob::<SimpleArchive>::to_blob(
                descriptor.facts().clone(),
            )),
            put_event(&expected_result),
            insert_event(CollectionRecord::Merge(expected_merge)),
        ];

        let mut store = ProbeStore::default();
        store.seed(&left);
        store.seed(&right);
        let first = publish_merge(&mut store, &descriptor, data(&right), data(&left)).unwrap();
        let second = publish_merge(&mut store, &descriptor, data(&left), data(&right)).unwrap();

        assert_eq!(first, (expected_merge.clone(), expected_result.clone()));
        assert_eq!(second, (expected_merge.clone(), expected_result.clone()));
        validate_merge(&descriptor, &first.0, low, high, &first.1).unwrap();

        let mut expected_events = sequence.clone();
        expected_events.extend(sequence);
        assert_eq!(store.events, expected_events);
        let expected_handles = BTreeSet::from([
            identity_for_tests(&descriptor).raw,
            low.get_handle().raw,
            high.get_handle().raw,
            expected_result.get_handle().raw,
        ]);
        assert_eq!(store.known, expected_handles);
        assert_eq!(
            store.records.keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([expected_merge.id()])
        );
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
        let descriptor = root("first");
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let (low, high) = ordered_inputs(&left, &right);
        let result = join(low, high).unwrap();
        let expected = CollectionMerge::new(
            identity_for_tests(&descriptor),
            data(low),
            data(high),
            data(&result),
        );
        // Three operations now, not five: descriptor put, result put, record
        // insert. The inputs are read, not written.
        for fail_at in 1..=3 {
            let mut store = ProbeStore::failing_before_effect_at(fail_at);
            store.seed(&left);
            store.seed(&right);
            let error =
                publish_merge(&mut store, &descriptor, data(&left), data(&right)).unwrap_err();
            match (fail_at, error) {
                (1..=2, PublicationError::DependencyPut(ProbeFailure(at)))
                | (3, PublicationError::RecordInsert(ProbeFailure(at))) => {
                    assert_eq!(at, fail_at)
                }
                (_, error) => panic!("unexpected publication error: {error}"),
            }

            assert!(!store.records.contains_key(&expected.id()));
            if fail_at <= 2 {
                assert!(!store.events.contains(&ProbeEvent::Insert(expected.id())));
            }

            store.recover();
            store.seed(&left);
            store.seed(&right);
            let retried =
                publish_merge(&mut store, &descriptor, data(&left), data(&right)).unwrap();
            assert_eq!(retried, (expected.clone(), result.clone()));
            assert!(store.records.contains_key(&expected.id()));
        }
    }

    #[test]
    fn publication_rejects_every_invalid_input_before_writing() {
        let (descriptor, data_blob, metadata, signing_key, _) = commit_fixture();
        let mut store = ProbeStore::default();
        let wrong_descriptor = test_naming(id(8), TRIBLE_SET_UNION_RECIPE_V1);
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

        // Resident but not canonical: the merge must still refuse, and for the
        // element's own reason rather than for absence.
        store.seed(&invalid_data);
        store.seed(&data_blob);
        assert!(matches!(
            publish_merge(
                &mut store,
                &descriptor,
                Handle::<SimpleArchive>::to_hash(invalid_data.get_handle()),
                Handle::<SimpleArchive>::to_hash(data_blob.get_handle()),
            ),
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

        let descriptor = root("first");
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let metadata = archive([row(9, 1, 9)]);
        let signing_key = SigningKey::from_bytes(&[7; 32]);

        let (commit, right_commit, merge, result) = {
            let mut pile = Pile::open(&path).unwrap();
            let commit =
                publish_commit(&mut pile, &descriptor, &left, &metadata, &signing_key).unwrap();
            // Both sides must be states before they can be merged -- the merge
            // reads them rather than writing them, so a side nobody committed
            // is refused by identity.
            let right_commit =
                publish_commit(&mut pile, &descriptor, &right, &metadata, &signing_key).unwrap();
            let (merge, result) =
                publish_merge(&mut pile, &descriptor, data(&right), data(&left)).unwrap();
            pile.close().unwrap();
            (commit, right_commit, merge, result)
        };

        let mut reopened = Pile::open(&path).unwrap();
        let discovered = discover_collection_records(&mut reopened).unwrap();
        let reader = reopened.reader().unwrap();
        // Both sides are commits now: a merge is an equation between states,
        // and each side became a state by being committed.
        let mut expected_commits = vec![commit.clone(), right_commit.clone()];
        expected_commits.sort_by_key(CollectionCommit::id);
        assert_eq!(discovered.commits(), expected_commits.as_slice());
        assert_eq!(discovered.merges(), &[merge.clone()]);
        assert!(discovered.derives().is_empty());
        assert!(discovered.diagnostics().is_empty());

        let fetched_descriptor: Blob<SimpleArchive> =
            reader.get(identity_for_tests(&descriptor)).unwrap();
        let fetched_left: Blob<SimpleArchive> = reader.get(left.get_handle()).unwrap();
        let fetched_right: Blob<SimpleArchive> = reader.get(right.get_handle()).unwrap();
        let fetched_metadata: Blob<SimpleArchive> = reader.get(metadata.get_handle()).unwrap();
        let fetched_result: Blob<SimpleArchive> = reader.get(result.get_handle()).unwrap();
        assert_eq!(
            <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(
                fetched_descriptor
            )
            .unwrap(),
            *descriptor.facts()
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

        let descriptor = root("first");
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

        let fetched_descriptor: Blob<SimpleArchive> =
            reader.get(identity_for_tests(&descriptor)).unwrap();
        let content_handle: Inline<Handle<SimpleArchive>> = commit.data().transmute();
        let fetched_content: Blob<SimpleArchive> = reader.get(content_handle).unwrap();
        let fetched_metadata: Blob<SimpleArchive> = reader.get(commit.metadata()).unwrap();
        assert_eq!(
            <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(
                fetched_descriptor
            )
            .unwrap(),
            *descriptor.facts()
        );
        assert_eq!(fetched_content, expected_content);
        assert_eq!(fetched_metadata, expected_metadata);
        validate_commit(&descriptor, &commit, &fetched_content).unwrap();

        let fetched_text: View<str> = reader.get::<View<str>, UTF8String>(text_handle).unwrap();
        let fetched_payload: Bytes = reader.get::<Bytes, RawBytes>(payload_handle).unwrap();
        assert_eq!(&*fetched_text, "a self-contained content blob");
        assert_eq!(&*fetched_payload, &[0, 1, 2, 3, 0xFE, 0xFF]);

        drop(reader);
        reopened.close().unwrap();
    }

    #[test]
    fn descriptor_and_empty_element_are_golden() {
        let descriptor = root("first");
        assert_eq!(
            <SimpleArchive as MetaDescribe>::id(),
            id_hex!("8F4A27C8581DADCBA1ADA8BA228069B6")
        );
        assert_eq!(
            TRIBLE_SET_UNION_RECIPE_V1,
            id_hex!("6D64C5F4B9E9B73F57C5F8702AB7FE45")
        );
        assert_eq!(
            crate::collection::descriptor::name(descriptor.facts())
                .unwrap()
                .unwrap()
                .as_str(),
            "first"
        );
        assert_eq!(
            crate::collection::descriptor::namespace(descriptor.facts())
                .unwrap()
                .unwrap(),
            test_team()
        );
        // The descriptor entity is intrinsic in its own attributes, so the
        // root moves only when those attributes move.
        assert_eq!(
            descriptor.root().unwrap(),
            id_hex!("C4E24340A80C70267458E2B6DD8EFDE4")
        );
        // The handle pins the whole current descriptor: namespace, optional
        // authority, reach, and the travelling schema and law descriptions.
        assert_eq!(
            identity_for_tests(&descriptor).raw,
            hex!("F5C0F55C167849EBE735E6C29A332BAD43298F3A55E9DB20A4944375CF8ADE79")
        );
        assert_eq!(
            IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle(),
            identity_for_tests(&descriptor)
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
    fn join_many_unions_overlaps_in_one_canonical_stream() {
        let empty = archive([]);
        let a = archive([row(1, 1, 1), row(3, 1, 3)]);
        let b = archive([row(2, 1, 2), row(3, 1, 3)]);
        let c = archive([row(1, 2, 4), row(4, 1, 5)]);

        assert_eq!(join_many(std::iter::empty()).unwrap(), empty);
        assert_eq!(join_many([&a]).unwrap(), a);

        let expected = join(&join(&a, &b).unwrap(), &c).unwrap();
        assert_eq!(join_many([&c, &empty, &a, &b, &a]).unwrap(), expected);
    }

    /// One canonical archive of `count` rows drawn from a deterministic
    /// sequence, so overlapping members really do share rows.
    fn strided_archive(offset: u64, stride: u64, count: usize) -> Blob<SimpleArchive> {
        let mut rows: Vec<[u8; TRIBLE_LEN]> = Vec::with_capacity(count);
        for step in 0..count as u64 {
            let key = offset + step * stride;
            let mut row = [0u8; TRIBLE_LEN];
            // A nonzero entity and attribute are what `Trible` demands; the
            // value half carries the scrambled key so the rows are spread over
            // the whole ordering rather than clustered under one prefix.
            row[8..16].copy_from_slice(&(key % 977 + 1).to_be_bytes());
            row[24..32].copy_from_slice(&(key % 31 + 1).to_be_bytes());
            let mut mixed = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            mixed ^= mixed >> 29;
            row[32..40].copy_from_slice(&mixed.to_be_bytes());
            row[40..48].copy_from_slice(&key.to_be_bytes());
            rows.push(row);
        }
        rows.sort_unstable();
        rows.dedup();
        raw_archive(rows)
    }

    /// The union any correct implementation must produce: every input row,
    /// sorted and deduplicated by the standard library.
    fn sort_dedup_oracle(elements: &[&Blob<SimpleArchive>]) -> Vec<[u8; TRIBLE_LEN]> {
        let mut rows: Vec<[u8; TRIBLE_LEN]> = Vec::new();
        for element in elements {
            let view: View<[[u8; TRIBLE_LEN]]> = element.bytes.clone().view().unwrap();
            rows.extend_from_slice(&view);
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// The partitioned merge is a performance decision, so it owes byte
    /// identity to the answer it replaced — not merely the same set.
    ///
    /// Sizes are deliberately unequal and the strides deliberately overlap:
    /// regular sampling has to survive one member large enough to dominate the
    /// sample and duplicates dense enough to straddle a partition boundary. The
    /// row count clears `PARALLEL_MERGE_THRESHOLD` so the parallel path is the
    /// one under test.
    #[test]
    fn join_many_partitioned_merge_matches_the_sorted_oracle_byte_for_byte() {
        let big = strided_archive(0, 1, 90_000);
        let overlapping = strided_archive(0, 2, 45_000);
        let shifted = strided_archive(1, 3, 30_000);
        let disjoint = strided_archive(1_000_000, 1, 12_000);
        let tiny = strided_archive(7, 500, 3);
        let empty = archive([]);
        let elements = [&big, &overlapping, &shifted, &disjoint, &tiny, &empty];

        let expected = sort_dedup_oracle(&elements);
        assert!(
            expected.len() > PARALLEL_MERGE_THRESHOLD,
            "the fixture must clear the partitioning threshold",
        );

        let union = join_many(elements).unwrap();
        let rows: View<[[u8; TRIBLE_LEN]]> = union.bytes.clone().view().unwrap();
        assert_eq!(&rows[..], &expected[..]);

        // The serial range merge is the same function the partitions call, so
        // agreeing with it pins the partition seams specifically.
        let views: Vec<View<[[u8; TRIBLE_LEN]]>> = elements
            .iter()
            .map(|element| element.bytes.clone().view().unwrap())
            .collect();
        let slices: Vec<&[[u8; TRIBLE_LEN]]> = views.iter().map(|view| &view[..]).collect();
        assert_eq!(merge_canonical_range(&slices, None, None), expected);

        // Order of arrival cannot matter: the union is commutative.
        let shuffled = join_many([&tiny, &disjoint, &empty, &shifted, &overlapping, &big]).unwrap();
        assert_eq!(shuffled, union);
    }

    #[test]
    fn join_many_reports_the_malformed_input_position() {
        let valid = archive([row(1, 1, 1)]);
        let invalid = raw_archive(vec![row(3, 1, 3), row(2, 1, 2)]);

        assert_eq!(
            join_many([&valid, &invalid, &valid]),
            Err((1, UnarchiveError::BadCanonicalizationOrdering)),
        );
    }

    #[test]
    fn commit_validation_binds_descriptor_collection_handle_and_bytes() {
        let descriptor = root("first");
        let blob = archive([row(1, 1, 1)]);
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            identity_for_tests(&descriptor),
            data(&blob),
            empty_metadata_handle(),
        );
        validate_commit(&descriptor, &commit, &blob).unwrap();

        let wrong_representation = test_naming(id(9), TRIBLE_SET_UNION_RECIPE_V1);
        assert!(matches!(
            validate_commit(&wrong_representation, &commit, &blob),
            Err(SimpleArchiveUnionValidationError::WrongRepresentation { .. })
        ));

        let wrong_recipe = test_naming(<SimpleArchive as MetaDescribe>::id(), id(9));
        assert!(matches!(
            validate_commit(&wrong_recipe, &commit, &blob),
            Err(SimpleArchiveUnionValidationError::WrongRecipe { .. })
        ));

        let other_descriptor = root("second");
        assert_eq!(
            validate_commit(&other_descriptor, &commit, &blob),
            Err(SimpleArchiveUnionValidationError::WrongCollection {
                expected: identity_for_tests(&other_descriptor),
                actual: identity_for_tests(&descriptor),
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
            identity_for_tests(&descriptor),
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
        let descriptor = root("first");
        let left = archive([row(1, 1, 1), row(3, 1, 3)]);
        let right = archive([row(2, 1, 2), row(3, 1, 3)]);
        let result = join(&left, &right).unwrap();
        let claim = CollectionMerge::new(
            identity_for_tests(&descriptor),
            data(&left),
            data(&right),
            data(&result),
        );
        let (low, high) = ordered_inputs(&left, &right);
        validate_merge(&descriptor, &claim, low, high, &result).unwrap();

        let wrong_collection = CollectionMerge::new(
            identity_for_tests(&root("ninth")),
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
            identity_for_tests(&descriptor),
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
            identity_for_tests(&descriptor),
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
                let collection = root("first");
                let claim = CollectionMerge::new(
                    identity_for_tests(&collection),
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

impl From<RecordDecodeError> for SimpleArchiveUnionValidationError {
    fn from(error: RecordDecodeError) -> Self {
        Self::Malformed(error)
    }
}
