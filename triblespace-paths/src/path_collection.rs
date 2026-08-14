//! Exact-ticket materialization of native path-summary collections.
//!
//! The signed source commits are the authority. `MERGE` and `DERIVE` records
//! are reproducible cache evidence: valid resident equations may replace leaf
//! work, but they never add roots to the caller's frozen ticket.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, BlobEncoding};
use triblespace_core::collection::simplearchive_union;
use triblespace_core::collection::{
    collection_physical_cover, discover_collection_records, resolve_collection_semantics,
    CollectionClaimValidation, CollectionCommit, CollectionData, CollectionDerive,
    CollectionDescriptor, CollectionMerge, CollectionRecord, CollectionStore,
    CollectionValidationRequest, DiscoveredCollectionRecords,
};
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut};

use crate::path_summary_union;
use crate::{Automaton, PathError, PathIndex, PathSummaryBlob, PathSummaryBlobError};
use path_summary_union::PathSummaryUnionError;

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure to validate, complete, or materialize one exact path ticket.
#[derive(Debug)]
pub enum PathSummaryCollectionError {
    /// A storage operation failed.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Backend error.
        source: BoxError,
    },
    /// The supplied ticket is not an exact set of discovered source commits.
    InvalidTicket(String),
    /// One ticket commit could not yet be validated from resident dependencies.
    IncompleteCommit(Id),
    /// One ticket commit was concretely rejected.
    RejectedCommit {
        /// Intrinsic commit id.
        commit: Id,
        /// Deterministic validation diagnostic.
        reason: String,
    },
    /// Collection resolution found contradictory accepted equations.
    Resolution(String),
    /// No resident target cover proves the entire frozen ticket.
    IncompleteCover {
        /// Target-frontier elements with no resident proof.
        missing: Vec<CollectionData>,
        /// Signed ticket roots not supported by the resident target cover.
        unsupported_commits: Vec<Id>,
    },
    /// Canonical path-summary construction failed.
    Algebra(PathSummaryUnionError),
    /// A selected summary did not decode under the fixed automaton.
    Summary(PathSummaryBlobError),
    /// Closing the joined summary into the accepted endpoint relation failed.
    Index(PathError),
}

impl PathSummaryCollectionError {
    fn storage<E>(operation: &'static str, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::Storage {
            operation,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for PathSummaryCollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage { operation, source } => write!(formatter, "{operation}: {source}"),
            Self::InvalidTicket(reason) => write!(formatter, "invalid exact path ticket: {reason}"),
            Self::IncompleteCommit(commit) => {
                write!(formatter, "source commit {commit:X} is incomplete")
            }
            Self::RejectedCommit { commit, reason } => {
                write!(formatter, "source commit {commit:X} was rejected: {reason}")
            }
            Self::Resolution(reason) => write!(formatter, "resolve path collection: {reason}"),
            Self::IncompleteCover {
                missing,
                unsupported_commits,
            } => write!(
                formatter,
                "path collection is incomplete ({} missing target element(s), {} unsupported source commit(s))",
                missing.len(),
                unsupported_commits.len(),
            ),
            Self::Algebra(source) => source.fmt(formatter),
            Self::Summary(source) => source.fmt(formatter),
            Self::Index(source) => source.fmt(formatter),
        }
    }
}

impl Error for PathSummaryCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Storage { source, .. } => Some(source.as_ref()),
            Self::Algebra(source) => Some(source),
            Self::Summary(source) => Some(source),
            Self::Index(source) => Some(source),
            Self::InvalidTicket(_)
            | Self::IncompleteCommit(_)
            | Self::RejectedCommit { .. }
            | Self::Resolution(_)
            | Self::IncompleteCover { .. } => None,
        }
    }
}

/// Canonical regular-path projection of one source `SimpleArchive` collection.
///
/// Collection identity is determined by the source scope and the canonical
/// fingerprint of the fixed automaton. Signed source commits remain the only
/// authority; path `DERIVE` and `MERGE` records are reproducible cache
/// evidence used to materialize an exact caller-supplied ticket.
#[derive(Clone, Debug)]
pub struct PathSummaryCollection {
    scope: Id,
    automaton: Automaton,
}

impl PathSummaryCollection {
    /// Construct the canonical path projection for `scope` and `automaton`.
    pub fn new(scope: Id, automaton: Automaton) -> Self {
        Self { scope, automaton }
    }

    /// Dataset scope shared with the source `SimpleArchive` collection.
    pub fn scope(&self) -> Id {
        self.scope
    }

    /// Fixed automaton whose fingerprint participates in collection identity.
    pub fn automaton(&self) -> &Automaton {
        &self.automaton
    }

    /// Canonical source `SimpleArchive` collection descriptor.
    pub fn source_descriptor(&self) -> CollectionDescriptor {
        simplearchive_union::descriptor(self.scope)
    }

    /// Canonical target path-summary collection descriptor.
    pub fn descriptor(&self) -> CollectionDescriptor {
        path_summary_union::descriptor(self.scope, &self.automaton)
    }

    /// Attach the exact endpoint relation already resident for `ticket`.
    ///
    /// This is the read-only half of the API. The target collection's logical
    /// frontier must be supported by exactly the supplied signed roots, and
    /// its physical cover must be complete. Commit metadata is deliberately
    /// outside this path-semantic boundary: strict signatures and canonical
    /// source data are validated here; callers that require metadata closure
    /// validate it when constructing the ticket.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        if ticket.is_empty() {
            return self
                .index_from_blob(path_summary_union::empty(&self.automaton))
                .map(Arc::new);
        }

        let source = self.source_descriptor();
        let target = self.descriptor();
        let probe = self.probe(store, ticket, &source, &target)?;
        if !probe.is_complete() {
            return Err(probe.incomplete_error());
        }
        let mut joined = path_summary_union::empty(&self.automaton);
        for data in probe.cover {
            let segment = require_resident::<_, PathSummaryBlob>(
                &probe.reader,
                data,
                "read selected path-summary cover element",
            )?;
            joined = path_summary_union::join(&joined, &segment, &self.automaton)
                .map_err(PathSummaryCollectionError::Algebra)?;
        }
        self.index_from_blob(joined).map(Arc::new)
    }

    /// Ensure and attach the exact endpoint relation for `ticket`.
    ///
    /// Existing valid source merges, target merges, and source-to-target
    /// derives are reused. Only distinct source elements whose signed roots
    /// remain unsupported are lowered. Output and descriptor blobs are
    /// published before their unsigned `DERIVE` records; no durability flush
    /// is implied. A fresh discovery pass must then prove that the resident
    /// target cover supports exactly the supplied ticket before closure runs
    /// once over the joined summary.
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
    ) -> Result<Arc<PathIndex>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        if ticket.is_empty() {
            return self.attach_exact(store, ticket);
        }

        let source = self.source_descriptor();
        let target = self.descriptor();
        let probe = self.probe(store, ticket, &source, &target)?;
        let prepared = if probe.is_complete() {
            Vec::new()
        } else {
            let residual = probe.source_residual_cover.clone();
            if residual.is_empty() {
                return Err(probe.incomplete_error());
            }

            let mut prepared = Vec::with_capacity(residual.len());
            for input_data in residual {
                let input = require_resident::<_, SimpleArchive>(
                    &probe.reader,
                    input_data,
                    "read path source element",
                )?;
                let output = path_summary_union::derive_element(&input, &self.automaton)
                    .map_err(PathSummaryCollectionError::Algebra)?;
                let output_data = Handle::<PathSummaryBlob>::to_hash(output.get_handle());
                let claim = CollectionDerive::new(
                    source.handle(),
                    target.handle(),
                    input_data,
                    output_data,
                );
                path_summary_union::validate_derive(
                    &source,
                    &target,
                    &claim,
                    &input,
                    &output,
                    &self.automaton,
                )
                .map_err(|error| PathSummaryCollectionError::Resolution(error.to_string()))?;
                prepared.push((output, claim));
            }
            prepared
        };

        // Never retain a snapshot reader across writes. Every dependency is
        // prepared first; publication then happens blobs-before-records.
        drop(probe);
        self.publish_descriptors(store, &source, &target)?;
        for (output, _) in &prepared {
            let expected = output.get_handle();
            let actual = store
                .put::<PathSummaryBlob, _>(output.clone())
                .map_err(|error| {
                    PathSummaryCollectionError::storage("store derived path-summary element", error)
                })?;
            if actual != expected {
                return Err(PathSummaryCollectionError::Resolution(
                    "blob store returned a noncanonical path-summary handle".to_owned(),
                ));
            }
        }
        for (_, claim) in prepared {
            store
                .insert(CollectionRecord::Derive(claim))
                .map_err(|error| {
                    PathSummaryCollectionError::storage("publish path DERIVE", error)
                })?;
        }

        // Construction is not admission. Re-discover and attach the same
        // byte-identical frozen ticket through a wholly fresh reader.
        self.attach_exact(store, ticket)
    }

    fn publish_descriptors<S>(
        &self,
        store: &mut S,
        source: &CollectionDescriptor,
        target: &CollectionDescriptor,
    ) -> Result<(), PathSummaryCollectionError>
    where
        S: BlobStorePut,
    {
        for descriptor in [source, target] {
            let actual = store
                .put::<SimpleArchive, _>(CollectionDescriptor::to_blob(descriptor))
                .map_err(|error| {
                    PathSummaryCollectionError::storage("store path collection descriptor", error)
                })?;
            if actual != descriptor.handle() {
                return Err(PathSummaryCollectionError::Resolution(
                    "blob store returned a noncanonical collection descriptor handle".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn probe<S>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        source: &CollectionDescriptor,
        target: &CollectionDescriptor,
    ) -> Result<ExactPathProbe<S::Reader>, PathSummaryCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let discovered = discover_collection_records(&mut *store).map_err(|error| {
            PathSummaryCollectionError::storage("discover path collection records", error)
        })?;
        let authorized = exact_ticket_ids(&discovered, ticket, source)?;
        let reader = store.reader().map_err(|error| {
            PathSummaryCollectionError::storage("open path collection reader", error)
        })?;
        let resolution = resolve_collection_semantics(&discovered, &authorized, |request| {
            self.validate_request(&reader, source, target, request)
        })
        .map_err(|error| PathSummaryCollectionError::Resolution(error.to_string()))?;

        for commit in ticket {
            if resolution.validation_pending().contains(&commit.id()) {
                return Err(PathSummaryCollectionError::IncompleteCommit(commit.id()));
            }
            if let Some(reason) = resolution.rejected().get(&commit.id()) {
                return Err(PathSummaryCollectionError::RejectedCommit {
                    commit: commit.id(),
                    reason: reason.clone(),
                });
            }
            if !resolution.admitted_claims().contains(&commit.id()) {
                return Err(PathSummaryCollectionError::InvalidTicket(format!(
                    "commit {:X} was not admitted",
                    commit.id(),
                )));
            }
        }

        let logically_supported: BTreeSet<_> = resolution
            .semantics()
            .frontier(target.handle())
            .into_iter()
            .flatten()
            .flat_map(|data| {
                resolution
                    .semantics()
                    .supporting_commit_ids(target.handle(), *data)
            })
            .collect();
        if let Some(commit) = logically_supported.difference(&authorized).next() {
            return Err(PathSummaryCollectionError::InvalidTicket(format!(
                "target frontier escaped the ticket through commit {commit:X}",
            )));
        }
        let unsupported_commits: BTreeSet<_> = authorized
            .difference(&logically_supported)
            .copied()
            .collect();

        // Lower a deterministic resident cover of the source lattice, not
        // necessarily its signed leaves. This is what makes an existing
        // source C = A ⊔ B reusable before any path DERIVE for C exists.
        // A selected source member is still useful when it also covers an
        // already-supported root: adding its image remains inside the exact
        // authorized ticket and can discharge another unsupported root.
        let mut source_resident = BTreeSet::new();
        for data in resolution
            .semantics()
            .members(source.handle())
            .into_iter()
            .flatten()
        {
            if contains::<_, SimpleArchive>(&reader, *data, "inspect path source residency")? {
                source_resident.insert(*data);
            }
        }
        let source_physical =
            collection_physical_cover(resolution.semantics(), source.handle(), &source_resident);
        if !source_physical.missing.is_empty() {
            return Err(PathSummaryCollectionError::Resolution(format!(
                "validated source collection lacks a resident cover for {} frontier element(s)",
                source_physical.missing.len(),
            )));
        }
        let source_residual_cover = source_physical
            .cover
            .into_iter()
            .filter(|data| {
                !resolution
                    .semantics()
                    .supporting_commit_ids(source.handle(), *data)
                    .is_disjoint(&unsupported_commits)
            })
            .collect();

        let mut resident = BTreeSet::new();
        for data in resolution
            .semantics()
            .members(target.handle())
            .into_iter()
            .flatten()
        {
            if contains::<_, PathSummaryBlob>(&reader, *data, "inspect path-summary residency")? {
                resident.insert(*data);
            }
        }
        let physical =
            collection_physical_cover(resolution.semantics(), target.handle(), &resident);

        Ok(ExactPathProbe {
            reader,
            cover: physical.cover,
            missing: physical.missing,
            unsupported_commits,
            source_residual_cover,
        })
    }

    fn validate_request<R>(
        &self,
        reader: &R,
        source: &CollectionDescriptor,
        target: &CollectionDescriptor,
        request: CollectionValidationRequest<'_>,
    ) -> Result<CollectionClaimValidation<String>, PathSummaryCollectionError>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        match request {
            CollectionValidationRequest::Commit { claim } => {
                let Some(data) = load_resident::<_, SimpleArchive>(
                    reader,
                    claim.data(),
                    "read path source commit data",
                )?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                Ok(
                    match simplearchive_union::validate_commit(source, claim, &data) {
                        Ok(()) => CollectionClaimValidation::Accepted,
                        Err(error) => CollectionClaimValidation::Rejected(error.to_string()),
                    },
                )
            }
            CollectionValidationRequest::Merge { claim }
                if claim.collection() == source.handle() =>
            {
                let Some((low, high, result)) = load_merge::<_, SimpleArchive>(
                    reader,
                    claim,
                    "read path source MERGE endpoint",
                )?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                Ok(
                    match simplearchive_union::validate_merge(source, claim, &low, &high, &result) {
                        Ok(()) => CollectionClaimValidation::Accepted,
                        Err(error) => CollectionClaimValidation::Rejected(error.to_string()),
                    },
                )
            }
            CollectionValidationRequest::Merge { claim }
                if claim.collection() == target.handle() =>
            {
                let Some((low, high, result)) = load_merge::<_, PathSummaryBlob>(
                    reader,
                    claim,
                    "read path target MERGE endpoint",
                )?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                Ok(
                    match path_summary_union::validate_merge(
                        target,
                        claim,
                        &low,
                        &high,
                        &result,
                        &self.automaton,
                    ) {
                        Ok(()) => CollectionClaimValidation::Accepted,
                        Err(error) => CollectionClaimValidation::Rejected(error.to_string()),
                    },
                )
            }
            CollectionValidationRequest::Derive { claim }
                if claim.source() == source.handle() && claim.target() == target.handle() =>
            {
                let (input_data, output_data) = claim.mapping();
                let Some(input) = load_resident::<_, SimpleArchive>(
                    reader,
                    input_data,
                    "read path DERIVE input",
                )?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                let Some(output) = load_resident::<_, PathSummaryBlob>(
                    reader,
                    output_data,
                    "read path DERIVE output",
                )?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                Ok(
                    match path_summary_union::validate_derive(
                        source,
                        target,
                        claim,
                        &input,
                        &output,
                        &self.automaton,
                    ) {
                        Ok(()) => CollectionClaimValidation::Accepted,
                        Err(error) => CollectionClaimValidation::Rejected(error.to_string()),
                    },
                )
            }
            CollectionValidationRequest::Merge { .. }
            | CollectionValidationRequest::Derive { .. } => Ok(CollectionClaimValidation::Pending),
        }
    }

    fn index_from_blob(
        &self,
        blob: Blob<PathSummaryBlob>,
    ) -> Result<PathIndex, PathSummaryCollectionError> {
        let summary = PathSummaryBlob::decode(blob, &self.automaton)
            .map_err(PathSummaryCollectionError::Summary)?;
        PathIndex::from_summary(summary).map_err(PathSummaryCollectionError::Index)
    }
}

struct ExactPathProbe<R> {
    reader: R,
    cover: BTreeSet<CollectionData>,
    missing: BTreeSet<CollectionData>,
    unsupported_commits: BTreeSet<Id>,
    source_residual_cover: BTreeSet<CollectionData>,
}

impl<R> ExactPathProbe<R> {
    fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unsupported_commits.is_empty()
    }

    fn incomplete_error(&self) -> PathSummaryCollectionError {
        PathSummaryCollectionError::IncompleteCover {
            missing: self.missing.iter().copied().collect(),
            unsupported_commits: self.unsupported_commits.iter().copied().collect(),
        }
    }
}

fn exact_ticket_ids(
    discovered: &DiscoveredCollectionRecords,
    ticket: &[CollectionCommit],
    source: &CollectionDescriptor,
) -> Result<BTreeSet<Id>, PathSummaryCollectionError> {
    let mut ids = BTreeSet::new();
    for commit in ticket {
        if commit.collection() != source.handle() {
            return Err(PathSummaryCollectionError::InvalidTicket(format!(
                "commit {:X} names another source collection",
                commit.id(),
            )));
        }
        match discovered
            .commits()
            .binary_search_by_key(&commit.id(), CollectionCommit::id)
        {
            Ok(index) if discovered.commits()[index] == *commit => {}
            Ok(_) => {
                return Err(PathSummaryCollectionError::InvalidTicket(format!(
                    "commit {:X} does not byte-match the discovered record",
                    commit.id(),
                )));
            }
            Err(_) => {
                return Err(PathSummaryCollectionError::InvalidTicket(format!(
                    "commit {:X} is absent from the store",
                    commit.id(),
                )));
            }
        }
        ids.insert(commit.id());
    }
    Ok(ids)
}

fn contains<R, E>(
    reader: &R,
    data: CollectionData,
    operation: &'static str,
) -> Result<bool, PathSummaryCollectionError>
where
    R: BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    reader
        .metadata(Handle::<E>::from_hash(data))
        .map(|metadata| metadata.is_some())
        .map_err(|error| PathSummaryCollectionError::storage(operation, error))
}

fn load_resident<R, E>(
    reader: &R,
    data: CollectionData,
    operation: &'static str,
) -> Result<Option<Blob<E>>, PathSummaryCollectionError>
where
    R: BlobStoreGet + BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    if !contains::<R, E>(reader, data, operation)? {
        return Ok(None);
    }
    reader
        .get(Handle::<E>::from_hash(data))
        .map(Some)
        .map_err(|error| PathSummaryCollectionError::storage(operation, error))
}

fn require_resident<R, E>(
    reader: &R,
    data: CollectionData,
    operation: &'static str,
) -> Result<Blob<E>, PathSummaryCollectionError>
where
    R: BlobStoreGet + BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    load_resident(reader, data, operation)?.ok_or_else(|| {
        PathSummaryCollectionError::IncompleteCover {
            missing: vec![data],
            unsupported_commits: Vec::new(),
        }
    })
}

fn load_merge<R, E>(
    reader: &R,
    claim: &CollectionMerge,
    operation: &'static str,
) -> Result<Option<(Blob<E>, Blob<E>, Blob<E>)>, PathSummaryCollectionError>
where
    R: BlobStoreGet + BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    let (low_data, high_data) = claim.inputs();
    let Some(low) = load_resident(reader, low_data, operation)? else {
        return Ok(None);
    };
    let Some(high) = load_resident(reader, high_data, operation)? else {
        return Ok(None);
    };
    let Some(result) = load_resident(reader, claim.result(), operation)? else {
        return Ok(None);
    };
    Ok(Some((low, high, result)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::IntoBlob;
    use triblespace_core::collection::CollectionRecord;
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::inline::RawInline;
    use triblespace_core::metadata;
    use triblespace_core::prelude::entity;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::trible::TribleSet;

    use crate::{Step, Transition};

    /// A deliberate compile-time proof that native paths require no PinStore.
    #[derive(Default)]
    struct CollectionOnly(MemoryRepo);

    impl BlobStorePut for CollectionOnly {
        type PutError = <MemoryRepo as BlobStorePut>::PutError;

        fn put<E, T>(
            &mut self,
            item: T,
        ) -> Result<triblespace_core::inline::Inline<Handle<E>>, Self::PutError>
        where
            E: BlobEncoding + 'static,
            T: triblespace_core::blob::IntoBlob<E>,
            Handle<E>: InlineEncoding,
        {
            self.0.put(item)
        }
    }

    impl BlobStore for CollectionOnly {
        type Reader = <MemoryRepo as BlobStore>::Reader;
        type ReaderError = <MemoryRepo as BlobStore>::ReaderError;

        fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
            self.0.reader()
        }
    }

    impl CollectionStore for CollectionOnly {
        type RecordsError = <MemoryRepo as CollectionStore>::RecordsError;
        type InsertError = <MemoryRepo as CollectionStore>::InsertError;
        type RecordIter<'a>
            = <MemoryRepo as CollectionStore>::RecordIter<'a>
        where
            Self: 'a;

        fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
            self.0.records()
        }

        fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
            self.0.insert(record)
        }
    }

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn plus() -> Automaton {
        Automaton::new(
            2,
            [0],
            [1],
            [
                Transition::new(0, 1, Step::Forward(metadata::tag.id().into())),
                Transition::new(1, 1, Step::Forward(metadata::tag.id().into())),
            ],
        )
        .unwrap()
    }

    fn edge(source: u8, target: u8) -> TribleSet {
        let source = id(source);
        let target = id(target);
        entity! { ExclusiveId::force_ref(&source) @ metadata::tag: target }.into_facts()
    }

    fn put_data(store: &mut CollectionOnly, facts: &TribleSet) -> Blob<SimpleArchive> {
        let blob = facts.to_blob();
        store.put::<SimpleArchive, _>(blob.clone()).unwrap();
        blob
    }

    fn signed_commit(
        store: &mut CollectionOnly,
        scope: Id,
        key: u8,
        data: &Blob<SimpleArchive>,
    ) -> CollectionCommit {
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        CollectionCommit::sign(
            &SigningKey::from_bytes(&[key; 32]),
            simplearchive_union::descriptor(scope).handle(),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            metadata,
        )
    }

    fn publish_commit(store: &mut CollectionOnly, commit: CollectionCommit) {
        store.insert(CollectionRecord::Commit(commit)).unwrap();
    }

    fn records(store: &mut CollectionOnly) -> Vec<CollectionRecord> {
        store.records().unwrap().map(Result::unwrap).collect()
    }

    fn assert_cross_fragment_path(index: &PathIndex) {
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(3))));
    }

    #[test]
    fn empty_ticket_is_local_bottom_and_writes_nothing() {
        let mut store = CollectionOnly::default();
        let paths = PathSummaryCollection::new(id(9), plus());
        let before_blobs = store.0.blobs.len();
        let before_records = records(&mut store).len();

        let index = paths.ensure_exact(&mut store, &[]).unwrap();

        assert_eq!(index.vertex_count(), 0);
        assert_eq!(index.accepted_pair_count(), 0);
        assert_eq!(store.0.blobs.len(), before_blobs);
        assert_eq!(records(&mut store).len(), before_records);
    }

    #[test]
    fn incomplete_ticket_is_diagnostic_then_ensure_closes_cross_fragment_path() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let left_commit = signed_commit(&mut store, scope, 1, &left);
        let right_commit = signed_commit(&mut store, scope, 2, &right);
        publish_commit(&mut store, left_commit);
        publish_commit(&mut store, right_commit);
        let ticket = [left_commit, right_commit];

        assert!(matches!(
            paths.attach_exact(&mut store, &ticket),
            Err(PathSummaryCollectionError::IncompleteCover {
                missing,
                unsupported_commits,
            }) if missing.is_empty() && unsupported_commits.len() == 2
        ));

        let index = paths.ensure_exact(&mut store, &ticket).unwrap();
        assert_cross_fragment_path(&index);
        assert_cross_fragment_path(&paths.attach_exact(&mut store, &ticket).unwrap());
    }

    #[test]
    fn old_exact_ticket_stays_stable_after_a_later_commit() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let left_commit = signed_commit(&mut store, scope, 1, &left);
        let right_commit = signed_commit(&mut store, scope, 2, &right);
        publish_commit(&mut store, left_commit);
        publish_commit(&mut store, right_commit);
        let ticket = [left_commit, right_commit];
        let first = paths.ensure_exact(&mut store, &ticket).unwrap();

        let later = put_data(&mut store, &edge(3, 4));
        let later_commit = signed_commit(&mut store, scope, 3, &later);
        publish_commit(&mut store, later_commit);
        let repeated = paths.attach_exact(&mut store, &ticket).unwrap();

        assert_eq!(
            first.accepted_pairs().collect::<Vec<_>>(),
            repeated.accepted_pairs().collect::<Vec<_>>()
        );
        assert!(!repeated.contains(&RawInline::from(id(1)), &RawInline::from(id(4))));
    }

    #[test]
    fn derive_before_commit_is_inert_then_becomes_live() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let source = put_data(&mut store, &edge(1, 2));
        let commit = signed_commit(&mut store, scope, 7, &source);
        let output = path_summary_union::derive_element(&source, paths.automaton()).unwrap();
        store.put::<PathSummaryBlob, _>(output.clone()).unwrap();
        let derive = CollectionDerive::new(
            paths.source_descriptor().handle(),
            paths.descriptor().handle(),
            commit.data(),
            Handle::<PathSummaryBlob>::to_hash(output.get_handle()),
        );
        store.insert(CollectionRecord::Derive(derive)).unwrap();

        assert!(matches!(
            paths.attach_exact(&mut store, &[commit]),
            Err(PathSummaryCollectionError::InvalidTicket(_))
        ));
        publish_commit(&mut store, commit);
        let index = paths.attach_exact(&mut store, &[commit]).unwrap();
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(2))));
    }

    #[test]
    fn identical_data_commits_share_one_derive_without_losing_provenance() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let data = put_data(&mut store, &edge(1, 2));
        let first = signed_commit(&mut store, scope, 1, &data);
        let second = signed_commit(&mut store, scope, 2, &data);
        publish_commit(&mut store, first);
        publish_commit(&mut store, second);

        // Repeating the exact same commit is set-idempotent; the distinct
        // signer remains a separate provenance root over shared data.
        let index = paths
            .ensure_exact(&mut store, &[first, first, second])
            .unwrap();
        assert!(index.contains(&RawInline::from(id(1)), &RawInline::from(id(2))));
        let derives = records(&mut store)
            .into_iter()
            .filter(|record| {
                matches!(record, CollectionRecord::Derive(claim)
                if claim.source() == paths.source_descriptor().handle()
                    && claim.target() == paths.descriptor().handle()
                    && claim.mapping().0 == first.data())
            })
            .count();
        assert_eq!(derives, 1);
        paths.attach_exact(&mut store, &[first, second]).unwrap();
    }

    #[test]
    fn existing_source_merge_then_single_derive_covers_both_roots() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish_commit(&mut store, first);
        publish_commit(&mut store, second);

        let joined_source = simplearchive_union::join(&left, &right).unwrap();
        store
            .put::<SimpleArchive, _>(joined_source.clone())
            .unwrap();
        let joined_data = Handle::<SimpleArchive>::to_hash(joined_source.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.source_descriptor().handle(),
                first.data(),
                second.data(),
                joined_data,
            )))
            .unwrap();
        let index = paths.ensure_exact(&mut store, &[first, second]).unwrap();
        assert_cross_fragment_path(&index);
        let derives: Vec<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == paths.source_descriptor().handle()
                        && claim.target() == paths.descriptor().handle() =>
                {
                    Some(claim)
                }
                _ => None,
            })
            .collect();
        assert_eq!(derives.len(), 1);
        assert_eq!(derives[0].mapping().0, joined_data);
    }

    #[test]
    fn source_cover_can_overlap_an_already_supported_root() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish_commit(&mut store, first);
        publish_commit(&mut store, second);

        let joined_source = simplearchive_union::join(&left, &right).unwrap();
        store
            .put::<SimpleArchive, _>(joined_source.clone())
            .unwrap();
        let joined_data = Handle::<SimpleArchive>::to_hash(joined_source.get_handle());
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.source_descriptor().handle(),
                first.data(),
                second.data(),
                joined_data,
            )))
            .unwrap();

        // The left root is already supported in the target. The resident
        // source cover nevertheless selects the merged upper element because
        // it also discharges the unsupported right root.
        let left_summary = path_summary_union::derive_element(&left, paths.automaton()).unwrap();
        store
            .put::<PathSummaryBlob, _>(left_summary.clone())
            .unwrap();
        store
            .insert(CollectionRecord::Derive(CollectionDerive::new(
                paths.source_descriptor().handle(),
                paths.descriptor().handle(),
                first.data(),
                Handle::<PathSummaryBlob>::to_hash(left_summary.get_handle()),
            )))
            .unwrap();

        let index = paths.ensure_exact(&mut store, &[first, second]).unwrap();
        assert_cross_fragment_path(&index);
        let inputs: BTreeSet<_> = records(&mut store)
            .into_iter()
            .filter_map(|record| match record {
                CollectionRecord::Derive(claim)
                    if claim.source() == paths.source_descriptor().handle()
                        && claim.target() == paths.descriptor().handle() =>
                {
                    Some(claim.mapping().0)
                }
                _ => None,
            })
            .collect();
        assert_eq!(inputs, [first.data(), joined_data].into_iter().collect());
        assert!(!inputs.contains(&second.data()));
    }

    #[test]
    fn existing_target_merge_is_selected_as_the_physical_cover() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let left = put_data(&mut store, &edge(1, 2));
        let right = put_data(&mut store, &edge(2, 3));
        let first = signed_commit(&mut store, scope, 1, &left);
        let second = signed_commit(&mut store, scope, 2, &right);
        publish_commit(&mut store, first);
        publish_commit(&mut store, second);

        let left_summary = path_summary_union::derive_element(&left, paths.automaton()).unwrap();
        let right_summary = path_summary_union::derive_element(&right, paths.automaton()).unwrap();
        for (source, summary) in [(&left, &left_summary), (&right, &right_summary)] {
            store.put::<PathSummaryBlob, _>(summary.clone()).unwrap();
            store
                .insert(CollectionRecord::Derive(CollectionDerive::new(
                    paths.source_descriptor().handle(),
                    paths.descriptor().handle(),
                    Handle::<SimpleArchive>::to_hash(source.get_handle()),
                    Handle::<PathSummaryBlob>::to_hash(summary.get_handle()),
                )))
                .unwrap();
        }
        let joined =
            path_summary_union::join(&left_summary, &right_summary, paths.automaton()).unwrap();
        store.put::<PathSummaryBlob, _>(joined.clone()).unwrap();
        store
            .insert(CollectionRecord::Merge(CollectionMerge::new(
                paths.descriptor().handle(),
                Handle::<PathSummaryBlob>::to_hash(left_summary.get_handle()),
                Handle::<PathSummaryBlob>::to_hash(right_summary.get_handle()),
                Handle::<PathSummaryBlob>::to_hash(joined.get_handle()),
            )))
            .unwrap();

        let source = paths.source_descriptor();
        let target = paths.descriptor();
        let probe = paths
            .probe(&mut store, &[first, second], &source, &target)
            .unwrap();
        assert!(probe.is_complete());
        assert_eq!(
            probe.cover,
            [Handle::<PathSummaryBlob>::to_hash(joined.get_handle())]
                .into_iter()
                .collect()
        );

        let index = paths.attach_exact(&mut store, &[first, second]).unwrap();
        assert_cross_fragment_path(&index);
    }

    #[test]
    fn absent_source_bytes_report_the_exact_commit() {
        let scope = id(9);
        let paths = PathSummaryCollection::new(scope, plus());
        let mut store = CollectionOnly::default();
        let absent = edge(1, 2).to_blob();
        let metadata = store
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[5; 32]),
            paths.source_descriptor().handle(),
            Handle::<SimpleArchive>::to_hash(absent.get_handle()),
            metadata,
        );
        publish_commit(&mut store, commit);

        assert!(matches!(
            paths.attach_exact(&mut store, &[commit]),
            Err(PathSummaryCollectionError::IncompleteCommit(found)) if found == commit.id()
        ));
    }
}
