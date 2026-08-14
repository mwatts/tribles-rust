//! Exact-ticket attachment shared by canonical derived collections.
//!
//! Concrete facades supply descriptors, algebra validation, derivation, and
//! final query materialization. This kernel owns the common authority and I/O
//! lifecycle around signed source roots and reproducible unsigned evidence.
//!
//! Unsigned equations are resident cache evidence, not durable validation
//! receipts: every endpoint needed to validate a `MERGE` or `DERIVE` must be
//! present on each admission pass. This kernel neither retains that proof
//! graph across garbage collection nor supplies a compaction policy. A facade
//! that promises durable compacted attachment must add deterministic recursive
//! reconstruction (or a heavier proof-retention scheme). Missing evidence is
//! safely pending, so signed leaves remain available for fallback completion.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::{Blob, BlobEncoding};
use crate::id::Id;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::{Inline, InlineEncoding};
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut};

use super::discovery::discover_collection_records_for_ticket;
use super::{
    collection_physical_cover, resolve_collection_semantics, CollectionClaimValidation,
    CollectionCommit, CollectionData, CollectionDerive, CollectionDescriptor, CollectionMerge,
    CollectionRecord, CollectionStore, CollectionValidationRequest, DiscoveredCollectionRecords,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// A resident typed claim handed to a concrete algebra validator.
///
/// The validator is the representation-specific security boundary: it must
/// bind the record descriptors and endpoint handles to freshly computed byte
/// identities, then prove the concrete algebra equation. Returning `Err`
/// rejects only this unsigned cache claim (or rejects a ticket root when the
/// variant is [`Commit`](Self::Commit)); it does not abort unrelated evidence.
pub enum DerivedClaim<'a, Source: BlobEncoding, Target: BlobEncoding> {
    /// An authorized source root.
    Commit {
        /// Signed claim.
        claim: &'a CollectionCommit,
        /// Claimed source bytes.
        data: &'a Blob<Source>,
    },
    /// A source-lattice merge.
    SourceMerge {
        /// Unsigned equation.
        claim: &'a CollectionMerge,
        /// Lower input.
        low: &'a Blob<Source>,
        /// Higher input.
        high: &'a Blob<Source>,
        /// Claimed result.
        result: &'a Blob<Source>,
    },
    /// A target-lattice merge.
    TargetMerge {
        /// Unsigned equation.
        claim: &'a CollectionMerge,
        /// Lower input.
        low: &'a Blob<Target>,
        /// Higher input.
        high: &'a Blob<Target>,
        /// Claimed result.
        result: &'a Blob<Target>,
    },
    /// A source-to-target derivation.
    Derive {
        /// Unsigned equation.
        claim: &'a CollectionDerive,
        /// Source input.
        input: &'a Blob<Source>,
        /// Target output.
        output: &'a Blob<Target>,
    },
}

/// A resident target cover in ascending content-handle order.
pub struct ExactCover<Target: BlobEncoding> {
    members: Vec<(CollectionData, Blob<Target>)>,
}

impl<Target: BlobEncoding> ExactCover<Target> {
    fn empty() -> Self {
        Self {
            members: Vec::new(),
        }
    }

    /// Number of selected physical members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the cover is the store-free empty-ticket bottom.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Borrow the ordered physical members.
    pub fn members(&self) -> &[(CollectionData, Blob<Target>)] {
        &self.members
    }

    /// Consume the ordered physical members.
    pub fn into_members(self) -> Vec<(CollectionData, Blob<Target>)> {
        self.members
    }

    /// Consume just the ordered blobs.
    pub fn into_blobs(self) -> impl ExactSizeIterator<Item = Blob<Target>> {
        self.members.into_iter().map(|(_, blob)| blob)
    }
}

/// Failure to attach or complete one exact derived ticket.
#[derive(Debug)]
pub enum ExactDerivedCollectionError {
    /// A storage operation failed.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Backend failure.
        source: BoxError,
    },
    /// The supplied records are not an exact resident source ticket.
    InvalidTicket(String),
    /// One signed root lacks resident dependencies.
    IncompleteCommit(Id),
    /// One signed root failed concrete validation.
    RejectedCommit {
        /// Intrinsic root ID.
        commit: Id,
        /// Concrete diagnostic.
        reason: String,
    },
    /// Resolution, identity, or freshly constructed evidence was invalid.
    Resolution(String),
    /// The target does not yet physically and logically cover the ticket.
    IncompleteCover {
        /// Missing target-frontier bytes.
        missing: Vec<CollectionData>,
        /// Ticket roots absent from logical target support.
        unsupported_commits: Vec<Id>,
    },
    /// Canonical source-to-target construction failed.
    Derive {
        /// Source member being lowered.
        input: CollectionData,
        /// Concrete construction failure.
        source: BoxError,
    },
}

impl ExactDerivedCollectionError {
    fn storage(operation: &'static str, source: impl Error + Send + Sync + 'static) -> Self {
        Self::Storage {
            operation,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for ExactDerivedCollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage { operation, source } => write!(f, "{operation}: {source}"),
            Self::InvalidTicket(reason) => write!(f, "invalid exact ticket: {reason}"),
            Self::IncompleteCommit(commit) => write!(f, "source commit {commit:X} is incomplete"),
            Self::RejectedCommit { commit, reason } => {
                write!(f, "source commit {commit:X} was rejected: {reason}")
            }
            Self::Resolution(reason) => write!(f, "resolve derived collection: {reason}"),
            Self::IncompleteCover {
                missing,
                unsupported_commits,
            } => write!(
                f,
                "derived collection is incomplete ({} missing target element(s), {} unsupported source commit(s))",
                missing.len(),
                unsupported_commits.len(),
            ),
            Self::Derive { input, source } => write!(
                f,
                "derive source element {}: {source}",
                hex::encode_upper(input.raw),
            ),
        }
    }
}

impl Error for ExactDerivedCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Storage { source, .. } | Self::Derive { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// Exact-ticket lifecycle for one fixed source-to-target homomorphism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactDerivedCollection<Source: BlobEncoding, Target: BlobEncoding> {
    source: CollectionDescriptor,
    target: CollectionDescriptor,
    encodings: PhantomData<fn() -> (Source, Target)>,
}

impl<Source, Target> ExactDerivedCollection<Source, Target>
where
    Source: BlobEncoding + 'static,
    Target: BlobEncoding + 'static,
    Handle<Source>: InlineEncoding,
    Handle<Target>: InlineEncoding,
{
    /// Bind the lifecycle to two distinct canonical descriptors.
    ///
    /// # Panics
    ///
    /// Panics when source and target are the same collection. An identity
    /// mapping is not a derived-collection lifecycle and would make claim
    /// dispatch ambiguous.
    pub fn new(source: CollectionDescriptor, target: CollectionDescriptor) -> Self {
        assert_ne!(
            source.handle(),
            target.handle(),
            "exact derived collection requires distinct source and target descriptors",
        );
        Self {
            source,
            target,
            encodings: PhantomData,
        }
    }

    /// Source descriptor.
    pub fn source_descriptor(&self) -> CollectionDescriptor {
        self.source
    }

    /// Target descriptor.
    pub fn target_descriptor(&self) -> CollectionDescriptor {
        self.target
    }

    /// Attach an already complete exact cover without writing.
    pub fn attach_exact<S, V>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        validate: V,
    ) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        V: for<'a> Fn(DerivedClaim<'a, Source, Target>) -> Result<(), String>,
    {
        self.attach_with(store, ticket, &validate)
    }

    /// Complete missing derivations, then attach through a fresh read pass.
    ///
    /// Empty tickets perform no I/O. A complete first probe returns without
    /// writes. Otherwise the reader is dropped before descriptors and all
    /// output blobs are written ahead of unsigned `DERIVE` records. No flush
    /// or signed record is emitted.
    pub fn ensure_exact<S, V, F, E>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        validate: V,
        derive: F,
    ) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        V: for<'a> Fn(DerivedClaim<'a, Source, Target>) -> Result<(), String>,
        F: Fn(&Blob<Source>) -> Result<Blob<Target>, E>,
        E: Error + Send + Sync + 'static,
    {
        if ticket.is_empty() {
            return Ok(ExactCover::empty());
        }

        let probe = self.probe(store, ticket, &validate, true)?;
        if probe.is_complete() {
            return probe.load_target_cover();
        }
        if probe.source_residual_cover.is_empty() {
            return Err(probe.incomplete_error());
        }

        let mut prepared = Vec::with_capacity(probe.source_residual_cover.len());
        for input_data in &probe.source_residual_cover {
            let input = require_exact::<_, Source>(
                &probe.reader,
                *input_data,
                "read selected source residual",
            )?;
            let output = derive(&input).map_err(|source| ExactDerivedCollectionError::Derive {
                input: *input_data,
                source: Box::new(source),
            })?;
            let output_data = fresh_data_identity(&output);
            let claim = CollectionDerive::new(
                self.source.handle(),
                self.target.handle(),
                *input_data,
                output_data,
            );
            validate(DerivedClaim::Derive {
                claim: &claim,
                input: &input,
                output: &output,
            })
            .map_err(|error| {
                ExactDerivedCollectionError::Resolution(format!(
                    "fresh DERIVE for {} failed validation: {error}",
                    hex::encode_upper(input_data.raw),
                ))
            })?;
            prepared.push((output_data, output, claim));
        }

        // Never retain an observed reader snapshot across publication.
        drop(probe);
        self.publish_descriptors(store)?;
        for (expected, output, _) in &prepared {
            let actual = store.put::<Target, _>(output.clone()).map_err(|error| {
                ExactDerivedCollectionError::storage("store derived target", error)
            })?;
            if Handle::<Target>::to_hash(actual) != *expected {
                return Err(ExactDerivedCollectionError::Resolution(
                    "blob store returned a noncanonical target handle".to_owned(),
                ));
            }
        }
        for (_, _, claim) in prepared {
            store
                .insert(CollectionRecord::Derive(claim))
                .map_err(|error| ExactDerivedCollectionError::storage("publish DERIVE", error))?;
        }

        // Construction is not admission.
        self.attach_with(store, ticket, &validate)
    }

    fn attach_with<S, V>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        validate: &V,
    ) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        V: for<'a> Fn(DerivedClaim<'a, Source, Target>) -> Result<(), String>,
    {
        if ticket.is_empty() {
            return Ok(ExactCover::empty());
        }
        let probe = self.probe(store, ticket, validate, false)?;
        if !probe.is_complete() {
            return Err(probe.incomplete_error());
        }
        probe.load_target_cover()
    }

    fn publish_descriptors<S: BlobStorePut>(
        &self,
        store: &mut S,
    ) -> Result<(), ExactDerivedCollectionError> {
        for descriptor in [self.source, self.target] {
            let blob = descriptor.to_blob();
            let actual = store
                .put::<SimpleArchive, _>(blob)
                .map_err(|error| ExactDerivedCollectionError::storage("store descriptor", error))?;
            if actual != descriptor.handle() {
                return Err(ExactDerivedCollectionError::Resolution(
                    "blob store returned a noncanonical descriptor handle".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn probe<S, V>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        validate: &V,
        plan_source_residual: bool,
    ) -> Result<ExactProbe<S::Reader>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        V: for<'a> Fn(DerivedClaim<'a, Source, Target>) -> Result<(), String>,
    {
        let requested: BTreeSet<_> = ticket.iter().map(CollectionCommit::id).collect();
        let discovered =
            discover_collection_records_for_ticket(store, &requested).map_err(|error| {
                ExactDerivedCollectionError::storage("discover exact ticket", error)
            })?;
        let authorized = exact_ticket_ids(&discovered, ticket, &self.source)?;
        let reader = store.reader().map_err(|error| {
            ExactDerivedCollectionError::storage("open exact-ticket reader", error)
        })?;
        let resolution = resolve_collection_semantics(&discovered, &authorized, |request| {
            self.validate_request(&reader, validate, request)
        })
        .map_err(|error| match error {
            super::CollectionResolutionError::Validation { source, .. } => source,
            super::CollectionResolutionError::Conflict(conflict) => {
                ExactDerivedCollectionError::Resolution(conflict.to_string())
            }
        })?;

        for commit in ticket {
            if resolution.validation_pending().contains(&commit.id()) {
                return Err(ExactDerivedCollectionError::IncompleteCommit(commit.id()));
            }
            if let Some(reason) = resolution.rejected().get(&commit.id()) {
                return Err(ExactDerivedCollectionError::RejectedCommit {
                    commit: commit.id(),
                    reason: reason.clone(),
                });
            }
            if !resolution.admitted_claims().contains(&commit.id()) {
                return Err(ExactDerivedCollectionError::InvalidTicket(format!(
                    "commit {:X} was not admitted",
                    commit.id(),
                )));
            }
        }

        let target = self.target.handle();
        let logically_supported: BTreeSet<_> = resolution
            .semantics()
            .frontier(target)
            .into_iter()
            .flatten()
            .flat_map(|data| resolution.semantics().supporting_commit_ids(target, *data))
            .collect();
        if let Some(commit) = logically_supported.difference(&authorized).next() {
            return Err(ExactDerivedCollectionError::InvalidTicket(format!(
                "target frontier escaped the ticket through commit {commit:X}",
            )));
        }
        let unsupported_commits: BTreeSet<_> = authorized
            .difference(&logically_supported)
            .copied()
            .collect();

        let target_resident = resident_members::<_, Target>(
            &reader,
            resolution.semantics().members(target),
            "inspect target residency",
        )?;
        let target_physical =
            collection_physical_cover(resolution.semantics(), target, &target_resident);

        let complete = target_physical.missing.is_empty() && unsupported_commits.is_empty();
        let source_residual_cover = if !plan_source_residual || complete {
            BTreeSet::new()
        } else {
            let source = self.source.handle();
            let source_resident = resident_members::<_, Source>(
                &reader,
                resolution.semantics().members(source),
                "inspect source residency",
            )?;
            let source_physical =
                collection_physical_cover(resolution.semantics(), source, &source_resident);
            if !source_physical.missing.is_empty() {
                return Err(ExactDerivedCollectionError::Resolution(format!(
                    "validated source lacks a resident cover for {} frontier element(s)",
                    source_physical.missing.len(),
                )));
            }

            // A logically supported target may have lost its bytes. Its
            // provenance is required work just like a root missing logically.
            let mut required = unsupported_commits.clone();
            for data in &target_physical.missing {
                required.extend(resolution.semantics().supporting_commit_ids(target, *data));
            }
            source_physical
                .cover
                .into_iter()
                .filter(|data| {
                    !resolution
                        .semantics()
                        .supporting_commit_ids(source, *data)
                        .is_disjoint(&required)
                })
                .collect()
        };

        Ok(ExactProbe {
            reader,
            cover: target_physical.cover,
            missing: target_physical.missing,
            unsupported_commits,
            source_residual_cover,
        })
    }

    fn validate_request<R, V>(
        &self,
        reader: &R,
        validate: &V,
        request: CollectionValidationRequest<'_>,
    ) -> Result<CollectionClaimValidation<String>, ExactDerivedCollectionError>
    where
        R: BlobStoreGet + BlobStoreMeta,
        V: for<'a> Fn(DerivedClaim<'a, Source, Target>) -> Result<(), String>,
    {
        let verdict = match request {
            CollectionValidationRequest::Commit { claim } => {
                let Some(data) = load_candidate::<_, Source>(reader, claim.data(), "read COMMIT")?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                validate(DerivedClaim::Commit { claim, data: &data })
            }
            CollectionValidationRequest::Merge { claim }
                if claim.collection() == self.source.handle() =>
            {
                let Some((low, high, result)) =
                    load_merge::<_, Source>(reader, claim, "read source MERGE")?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                validate(DerivedClaim::SourceMerge {
                    claim,
                    low: &low,
                    high: &high,
                    result: &result,
                })
            }
            CollectionValidationRequest::Merge { claim }
                if claim.collection() == self.target.handle() =>
            {
                let Some((low, high, result)) =
                    load_merge::<_, Target>(reader, claim, "read target MERGE")?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                validate(DerivedClaim::TargetMerge {
                    claim,
                    low: &low,
                    high: &high,
                    result: &result,
                })
            }
            CollectionValidationRequest::Derive { claim }
                if claim.source() == self.source.handle()
                    && claim.target() == self.target.handle() =>
            {
                let (input_data, output_data) = claim.mapping();
                let Some(input) =
                    load_candidate::<_, Source>(reader, input_data, "read DERIVE input")?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                let Some(output) =
                    load_candidate::<_, Target>(reader, output_data, "read DERIVE output")?
                else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                validate(DerivedClaim::Derive {
                    claim,
                    input: &input,
                    output: &output,
                })
            }
            CollectionValidationRequest::Merge { .. }
            | CollectionValidationRequest::Derive { .. } => {
                return Ok(CollectionClaimValidation::Pending);
            }
        };
        Ok(match verdict {
            Ok(()) => CollectionClaimValidation::Accepted,
            Err(error) => CollectionClaimValidation::Rejected(error.to_string()),
        })
    }
}

struct ExactProbe<R> {
    reader: R,
    cover: BTreeSet<CollectionData>,
    missing: BTreeSet<CollectionData>,
    unsupported_commits: BTreeSet<Id>,
    source_residual_cover: BTreeSet<CollectionData>,
}

impl<R: BlobStoreGet + BlobStoreMeta> ExactProbe<R> {
    fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unsupported_commits.is_empty()
    }

    fn incomplete_error(&self) -> ExactDerivedCollectionError {
        ExactDerivedCollectionError::IncompleteCover {
            missing: self.missing.iter().copied().collect(),
            unsupported_commits: self.unsupported_commits.iter().copied().collect(),
        }
    }

    fn load_target_cover<Target>(self) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        Target: BlobEncoding + 'static,
        Handle<Target>: InlineEncoding,
    {
        let mut members = Vec::with_capacity(self.cover.len());
        for data in self.cover {
            members.push((
                data,
                require_exact::<_, Target>(&self.reader, data, "read selected target cover")?,
            ));
        }
        Ok(ExactCover { members })
    }
}

fn exact_ticket_ids(
    discovered: &DiscoveredCollectionRecords,
    ticket: &[CollectionCommit],
    source: &CollectionDescriptor,
) -> Result<BTreeSet<Id>, ExactDerivedCollectionError> {
    let mut ids = BTreeSet::new();
    for commit in ticket {
        if commit.collection() != source.handle() {
            return Err(ExactDerivedCollectionError::InvalidTicket(format!(
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
                return Err(ExactDerivedCollectionError::InvalidTicket(format!(
                    "commit {:X} does not byte-match the stored record",
                    commit.id(),
                )));
            }
            Err(_) => {
                return Err(ExactDerivedCollectionError::InvalidTicket(format!(
                    "commit {:X} is absent or fails strict signature verification",
                    commit.id(),
                )));
            }
        }
        ids.insert(commit.id());
    }
    Ok(ids)
}

fn resident_members<R, E>(
    reader: &R,
    members: Option<&BTreeSet<CollectionData>>,
    operation: &'static str,
) -> Result<BTreeSet<CollectionData>, ExactDerivedCollectionError>
where
    R: BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    let mut resident = BTreeSet::new();
    for data in members.into_iter().flatten() {
        if contains::<_, E>(reader, *data, operation)? {
            resident.insert(*data);
        }
    }
    Ok(resident)
}

fn contains<R, E>(
    reader: &R,
    data: CollectionData,
    operation: &'static str,
) -> Result<bool, ExactDerivedCollectionError>
where
    R: BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    reader
        .metadata(Handle::<E>::from_hash(data))
        .map(|metadata| metadata.is_some())
        .map_err(|error| ExactDerivedCollectionError::storage(operation, error))
}

fn load_candidate<R, E>(
    reader: &R,
    data: CollectionData,
    operation: &'static str,
) -> Result<Option<Blob<E>>, ExactDerivedCollectionError>
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
        .map_err(|error| ExactDerivedCollectionError::storage(operation, error))
}

fn require_exact<R, E>(
    reader: &R,
    data: CollectionData,
    operation: &'static str,
) -> Result<Blob<E>, ExactDerivedCollectionError>
where
    R: BlobStoreGet + BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    let blob = load_candidate(reader, data, operation)?.ok_or_else(|| {
        ExactDerivedCollectionError::IncompleteCover {
            missing: vec![data],
            unsupported_commits: Vec::new(),
        }
    })?;
    let actual = fresh_data_identity(&blob);
    if actual != data {
        return Err(ExactDerivedCollectionError::Resolution(format!(
            "selected resident bytes hash to {} instead of {}",
            hex::encode_upper(actual.raw),
            hex::encode_upper(data.raw),
        )));
    }
    Ok(blob)
}

fn load_merge<R, E>(
    reader: &R,
    claim: &CollectionMerge,
    operation: &'static str,
) -> Result<Option<(Blob<E>, Blob<E>, Blob<E>)>, ExactDerivedCollectionError>
where
    R: BlobStoreGet + BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
{
    let (low_data, high_data) = claim.inputs();
    let Some(low) = load_candidate(reader, low_data, operation)? else {
        return Ok(None);
    };
    let Some(high) = load_candidate(reader, high_data, operation)? else {
        return Ok(None);
    };
    let Some(result) = load_candidate(reader, claim.result(), operation)? else {
        return Ok(None);
    };
    Ok(Some((low, high, result)))
}

fn fresh_data_identity<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData {
    Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
}

#[cfg(test)]
mod tests;
