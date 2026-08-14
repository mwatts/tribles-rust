//! Exact-ticket attachment shared by canonical derived collections.
//!
//! Concrete facades supply descriptors, one canonical five-operation algebra,
//! and final query materialization. This kernel owns the common authority and
//! I/O lifecycle around signed source roots and reproducible unsigned evidence.
//!
//! Unsigned equations are cache evidence, not durable validation receipts.
//! Admission walks backwards from resident source and target results, then
//! recomputes that finite proof graph forwards from authenticated source leaves.
//! Canonical intermediates live only in use-counted scratch, so garbage
//! collection may discard them without invalidating a resident upper result.
//! Selected optional artifacts are still freshly hashed and representation-
//! validated; bad cache bytes are removed from consideration and the physical
//! cover falls back without acquiring authority.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::{Blob, BlobEncoding};
use crate::id::Id;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::{Inline, InlineEncoding};
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut};

use super::discovery::{
    canonicalize_exact_ticket, discover_collection_records_for_ticket, validate_exact_ticket,
};
use super::{
    collection_physical_cover, resolve_collection_semantics, CollectionClaimValidation,
    CollectionCommit, CollectionData, CollectionDerive, CollectionDescriptor, CollectionId,
    CollectionMerge, CollectionRecord, CollectionSemantics, CollectionStore,
    CollectionValidationRequest, DiscoveredCollectionRecords,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure of one fixed canonical collection operation.
///
/// `Capacity` is reserved for deterministic geometry limits of the chosen
/// representation (for example, a `u32` ordinal space). It must never stand
/// for transient allocation, I/O, or accelerator failure. Persisted malformed
/// or noncanonical bytes are always `Fatal`, even when their decoder happens
/// to encounter a capacity-looking field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExactAlgebraError {
    /// The operation or supplied bytes are invalid for reasons that cannot be
    /// repaired by selecting a finer physical cover.
    Fatal(String),
    /// The canonical result does not fit this fixed representation, but the
    /// operation may still proceed through finer physical cover members.
    Capacity(String),
}

impl fmt::Display for ExactAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(reason) | Self::Capacity(reason) => f.write_str(reason),
        }
    }
}

impl Error for ExactAlgebraError {}

/// Canonical operations needed to reconstruct one exact derived collection.
///
/// The kernel binds descriptors, records, and freshly computed content
/// identities. This trait is the representation boundary: validators prove
/// that resident terminal bytes belong to their lattice, while constructors
/// compute the unique canonical result of each algebraic operation. Errors on
/// unsigned equations reject only that optional cache evidence; errors on an
/// authenticated source root reject the ticket.
pub trait ExactDerivedAlgebra<Source: BlobEncoding, Target: BlobEncoding> {
    /// Validate the exact source descriptor and one canonical source element.
    fn validate_source(
        &self,
        descriptor: &CollectionDescriptor,
        source: &Blob<Source>,
    ) -> Result<(), ExactAlgebraError>;

    /// Validate the exact target descriptor and one canonical target element.
    fn validate_target(
        &self,
        descriptor: &CollectionDescriptor,
        target: &Blob<Target>,
    ) -> Result<(), ExactAlgebraError>;

    /// Compute the canonical source join.
    fn join_source(
        &self,
        low: &Blob<Source>,
        high: &Blob<Source>,
    ) -> Result<Blob<Source>, ExactAlgebraError>;

    /// Compute the canonical source-to-target homomorphism.
    fn derive(&self, source: &Blob<Source>) -> Result<Blob<Target>, ExactAlgebraError>;

    /// Compute the canonical target join.
    fn join_target(
        &self,
        low: &Blob<Target>,
        high: &Blob<Target>,
    ) -> Result<Blob<Target>, ExactAlgebraError>;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TypedData {
    Source(CollectionData),
    Target(CollectionData),
}

impl TypedData {
    fn data(self) -> CollectionData {
        match self {
            Self::Source(data) | Self::Target(data) => data,
        }
    }
}

enum ScratchValue<Source: BlobEncoding, Target: BlobEncoding> {
    Source(Blob<Source>),
    Target(Blob<Target>),
}

#[derive(Clone, Copy, Debug)]
enum Candidate {
    SourceMerge(CollectionMerge),
    Derive(CollectionDerive),
    TargetMerge(CollectionMerge),
}

impl Candidate {
    fn id(self) -> Id {
        match self {
            Self::SourceMerge(claim) | Self::TargetMerge(claim) => claim.id(),
            Self::Derive(claim) => claim.id(),
        }
    }

    fn inputs(self) -> (TypedData, Option<TypedData>) {
        match self {
            Self::SourceMerge(claim) => {
                let (low, high) = claim.inputs();
                (
                    TypedData::Source(low),
                    (high != low).then_some(TypedData::Source(high)),
                )
            }
            Self::Derive(claim) => (TypedData::Source(claim.mapping().0), None),
            Self::TargetMerge(claim) => {
                let (low, high) = claim.inputs();
                (
                    TypedData::Target(low),
                    (high != low).then_some(TypedData::Target(high)),
                )
            }
        }
    }

    fn result(self) -> TypedData {
        match self {
            Self::SourceMerge(claim) => TypedData::Source(claim.result()),
            Self::Derive(claim) => TypedData::Target(claim.mapping().1),
            Self::TargetMerge(claim) => TypedData::Target(claim.result()),
        }
    }

    fn kind_order(self) -> u8 {
        match self {
            Self::SourceMerge(_) => 0,
            Self::Derive(_) => 1,
            Self::TargetMerge(_) => 2,
        }
    }
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
        reason: String,
    },
    /// No physical source cover can represent every signed root after
    /// capacity-terminal source members are excluded.
    UnrepresentableCover {
        /// Source members whose canonical target images exceeded the fixed
        /// representation geometry.
        blocked: Vec<(CollectionData, String)>,
        /// Source frontier obligations no longer covered by any usable member.
        missing: Vec<CollectionData>,
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
            Self::Derive { input, reason } => write!(
                f,
                "derive source element {}: {reason}",
                hex::encode_upper(input.raw),
            ),
            Self::UnrepresentableCover { blocked, missing } => write!(
                f,
                "exact source cover is unrepresentable ({} capacity-terminal member(s), {} uncovered source obligation(s))",
                blocked.len(),
                missing.len(),
            ),
        }
    }
}

impl Error for ExactDerivedCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Storage { source, .. } => Some(source.as_ref()),
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
    ///
    /// Missing unsigned intermediates are reconstructed in use-counted scratch
    /// from authenticated source roots. Scratch validation never publishes a
    /// blob or equation.
    pub fn attach_exact<S, A>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        algebra: &A,
    ) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        self.attach_with(store, ticket, algebra)
    }

    /// Complete missing derivations, then attach through a fresh read pass.
    ///
    /// Empty tickets perform no I/O. A complete first probe returns without
    /// writes. Deterministic capacity excludes the selected source member and
    /// globally replans under the same snapshot; terminal unrepresentability
    /// returns before any write. For a final feasible plan, the reader is
    /// dropped before descriptors and all output blobs are written ahead of
    /// unsigned `DERIVE` records. No flush or signed record is emitted.
    pub fn ensure_exact<S, A>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        algebra: &A,
    ) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        if ticket.is_empty() {
            return Ok(ExactCover::empty());
        }

        let probe = self.probe(store, ticket, algebra, true)?;
        if probe.is_complete() {
            return Ok(probe.into_target_cover());
        }
        // Capacity belongs to a selected physical source member, not to its
        // logical support. Excluding that member can expose a completely
        // different overlap-aware cover, so every capacity result restarts
        // global cover selection against this same reader/resolution snapshot.
        // Successful images are cached by source identity, but only members of
        // the final feasible plan are ever published.
        let mut blocked = BTreeMap::<CollectionData, String>::new();
        let mut cached = BTreeMap::<CollectionData, PreparedDerive<Target>>::new();
        let prepared = loop {
            let source_cover = probe.source_residual_cover(algebra, &blocked)?;
            if source_cover.is_empty() {
                return Err(probe.incomplete_error());
            }

            let mut selected = Vec::with_capacity(source_cover.len());
            let mut replan = None;
            for (input_data, input) in source_cover {
                if !cached.contains_key(&input_data) {
                    let output = match algebra.derive(&input) {
                        Ok(output) => output,
                        Err(ExactAlgebraError::Fatal(reason)) => {
                            return Err(ExactDerivedCollectionError::Derive {
                                input: input_data,
                                reason,
                            });
                        }
                        Err(ExactAlgebraError::Capacity(reason)) => {
                            replan = Some((input_data, reason));
                            break;
                        }
                    };
                    match algebra.validate_target(&self.target, &output) {
                        Ok(()) => {}
                        Err(ExactAlgebraError::Fatal(reason)) => {
                            return Err(ExactDerivedCollectionError::Resolution(format!(
                                "fresh DERIVE for {} constructed an invalid target: {reason}",
                                hex::encode_upper(input_data.raw),
                            )));
                        }
                        Err(ExactAlgebraError::Capacity(reason)) => {
                            replan = Some((input_data, reason));
                            break;
                        }
                    }
                    let output_data = fresh_data_identity(&output);
                    let claim = CollectionDerive::new(
                        self.source.handle(),
                        self.target.handle(),
                        input_data,
                        output_data,
                    );
                    cached.insert(
                        input_data,
                        PreparedDerive {
                            output_data,
                            output,
                            claim,
                        },
                    );
                }
                selected.push(input_data);
            }

            if let Some((input, reason)) = replan {
                blocked.insert(input, reason);
                continue;
            }
            break selected
                .into_iter()
                .map(|input| {
                    cached
                        .remove(&input)
                        .expect("every feasible source member has a cached image")
                })
                .collect::<Vec<_>>();
        };

        // Never retain an observed reader snapshot across publication.
        drop(probe);
        self.publish_descriptors(store)?;
        for prepared in &prepared {
            let actual = store
                .put::<Target, _>(prepared.output.clone())
                .map_err(|error| {
                    ExactDerivedCollectionError::storage("store derived target", error)
                })?;
            if Handle::<Target>::to_hash(actual) != prepared.output_data {
                return Err(ExactDerivedCollectionError::Resolution(
                    "blob store returned a noncanonical target handle".to_owned(),
                ));
            }
        }
        for prepared in prepared {
            store
                .insert(CollectionRecord::Derive(prepared.claim))
                .map_err(|error| ExactDerivedCollectionError::storage("publish DERIVE", error))?;
        }

        // Construction is not admission.
        self.attach_with(store, ticket, algebra)
    }

    fn attach_with<S, A>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        algebra: &A,
    ) -> Result<ExactCover<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        if ticket.is_empty() {
            return Ok(ExactCover::empty());
        }
        let probe = self.probe(store, ticket, algebra, false)?;
        if !probe.is_complete() {
            return Err(probe.incomplete_error());
        }
        Ok(probe.into_target_cover())
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

    fn probe<S, A>(
        &self,
        store: &mut S,
        ticket: &[CollectionCommit],
        algebra: &A,
        plan_source_residual: bool,
    ) -> Result<ExactProbe<S::Reader, Source, Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        let ticket = canonicalize_exact_ticket(ticket, self.source.handle())
            .map_err(|error| ExactDerivedCollectionError::InvalidTicket(error.to_string()))?;
        let requested: BTreeSet<_> = ticket.iter().map(CollectionCommit::id).collect();
        let discovered = discover_collection_records_for_ticket(
            store,
            &requested,
            self.source.handle(),
            self.target.handle(),
        )
        .map_err(|error| ExactDerivedCollectionError::storage("discover exact ticket", error))?;
        let authorized = validate_exact_ticket(&discovered, &ticket)
            .map_err(|error| ExactDerivedCollectionError::InvalidTicket(error.to_string()))?;
        let reader = store.reader().map_err(|error| {
            ExactDerivedCollectionError::storage("open exact-ticket reader", error)
        })?;

        let mut known = BTreeMap::<TypedData, ScratchValue<Source, Target>>::new();
        let mut roots = BTreeSet::new();
        for commit in discovered
            .commits()
            .iter()
            .filter(|commit| authorized.contains(&commit.id()))
        {
            let node = TypedData::Source(commit.data());
            if !known.contains_key(&node) {
                let Some(blob) =
                    load_candidate::<_, Source>(&reader, commit.data(), "read source COMMIT")?
                else {
                    return Err(ExactDerivedCollectionError::IncompleteCommit(commit.id()));
                };
                let actual = fresh_data_identity(&blob);
                if actual != commit.data() {
                    return Err(ExactDerivedCollectionError::RejectedCommit {
                        commit: commit.id(),
                        reason: format!(
                            "source bytes hash to {} instead of {}",
                            hex::encode_upper(actual.raw),
                            hex::encode_upper(commit.data().raw),
                        ),
                    });
                }
                algebra
                    .validate_source(&self.source, &blob)
                    .map_err(|error| ExactDerivedCollectionError::RejectedCommit {
                        commit: commit.id(),
                        reason: match error {
                            ExactAlgebraError::Fatal(reason) => reason,
                            ExactAlgebraError::Capacity(reason) => format!(
                                "persisted source exceeds representation capacity: {reason}"
                            ),
                        },
                    })?;
                known.insert(node, ScratchValue::Source(blob));
            }
            roots.insert(node);
        }

        let candidates = self.candidates(&discovered);
        let mut producers = BTreeMap::<TypedData, Vec<usize>>::new();
        for (index, candidate) in candidates.iter().copied().enumerate() {
            producers.entry(candidate.result()).or_default().push(index);
        }

        // Source compaction results are seeds too: ensure may reuse a resident
        // source upper bound even when no target artifact exists yet.
        let mut resident_results = BTreeSet::new();
        let mut reverse_seen = BTreeSet::new();
        let mut reverse_queue = VecDeque::new();
        for result in producers.keys().copied() {
            if known.contains_key(&result)
                || self.contains_typed(&reader, result, "inspect reconstructed result")?
            {
                resident_results.insert(result);
                if reverse_seen.insert(result) {
                    reverse_queue.push_back(result);
                }
            }
        }

        // Include every producer path, including producers of authenticated
        // roots: another ticket commit may be reachable only through that
        // merge history, so first-proof traversal would lose provenance.
        let mut candidate_indices = BTreeSet::new();
        while let Some(result) = reverse_queue.pop_front() {
            let Some(indices) = producers.get(&result) else {
                continue;
            };
            for &index in indices {
                candidate_indices.insert(index);
                let (first, second) = candidates[index].inputs();
                for input in [Some(first), second].into_iter().flatten() {
                    if reverse_seen.insert(input) {
                        reverse_queue.push_back(input);
                    }
                }
            }
        }

        let (accepted, rejected) = evaluate_candidates(
            &candidates,
            &candidate_indices,
            &roots,
            &mut known,
            &self.source,
            &self.target,
            algebra,
        );
        let resolution = resolve_collection_semantics(&discovered, &authorized, |request| {
            let claim = request.claim_id();
            Ok::<CollectionClaimValidation<String>, std::convert::Infallible>(
                if matches!(request, CollectionValidationRequest::Commit { .. }) {
                    CollectionClaimValidation::Accepted
                } else if accepted.contains(&claim) {
                    CollectionClaimValidation::Accepted
                } else if let Some(reason) = rejected.get(&claim) {
                    CollectionClaimValidation::Rejected(reason.clone())
                } else {
                    CollectionClaimValidation::Pending
                },
            )
        });
        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(super::CollectionResolutionError::Validation { source, .. }) => match source {},
            Err(super::CollectionResolutionError::Conflict(conflict)) => {
                return Err(ExactDerivedCollectionError::Resolution(
                    conflict.to_string(),
                ));
            }
        };

        for commit in &ticket {
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

        let target_resident = resolution
            .semantics()
            .members(target)
            .into_iter()
            .flatten()
            .copied()
            .filter(|data| resident_results.contains(&TypedData::Target(*data)))
            .collect();
        let target_physical = validated_physical_cover(
            &reader,
            resolution.semantics(),
            target,
            target_resident,
            &BTreeMap::new(),
            |blob| algebra.validate_target(&self.target, blob),
        );

        let complete = target_physical.missing.is_empty() && unsupported_commits.is_empty();
        let source_plan_parts = if !plan_source_residual || complete {
            None
        } else {
            let source = self.source.handle();
            let source_roots: BTreeMap<_, _> = roots
                .iter()
                .filter_map(|node| match (node, known.get(node)) {
                    (TypedData::Source(data), Some(ScratchValue::Source(blob))) => {
                        Some((*data, blob.clone()))
                    }
                    _ => None,
                })
                .collect();
            let source_resident = resolution
                .semantics()
                .members(source)
                .into_iter()
                .flatten()
                .copied()
                .filter(|data| {
                    source_roots.contains_key(data)
                        || resident_results.contains(&TypedData::Source(*data))
                })
                .collect();
            // A logically supported target may have lost its bytes. Its
            // provenance is required work just like a root missing logically.
            let mut required = unsupported_commits.clone();
            for data in &target_physical.missing {
                required.extend(resolution.semantics().supporting_commit_ids(target, *data));
            }

            Some((source, source_resident, source_roots, required))
        };
        let source_plan =
            source_plan_parts.map(|(collection, resident, mandatory, required_commits)| {
                SourcePlan {
                    descriptor: self.source,
                    semantics: resolution.into_semantics(),
                    collection,
                    resident,
                    mandatory,
                    required_commits,
                }
            });

        Ok(ExactProbe {
            reader,
            target_cover: ExactCover {
                members: target_physical
                    .cover
                    .iter()
                    .map(|data| {
                        (
                            *data,
                            target_physical
                                .blobs
                                .get(data)
                                .expect("validated target cover retains selected bytes")
                                .clone(),
                        )
                    })
                    .collect(),
            },
            missing: target_physical.missing,
            unsupported_commits,
            source_plan,
        })
    }

    fn candidates(&self, discovered: &DiscoveredCollectionRecords) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        candidates.extend(
            discovered
                .merges()
                .iter()
                .filter(|claim| claim.collection() == self.source.handle())
                .copied()
                .map(Candidate::SourceMerge),
        );
        candidates.extend(
            discovered
                .derives()
                .iter()
                .filter(|claim| {
                    claim.source() == self.source.handle() && claim.target() == self.target.handle()
                })
                .copied()
                .map(Candidate::Derive),
        );
        candidates.extend(
            discovered
                .merges()
                .iter()
                .filter(|claim| claim.collection() == self.target.handle())
                .copied()
                .map(Candidate::TargetMerge),
        );
        candidates.sort_unstable_by_key(|candidate| (candidate.id(), candidate.kind_order()));
        candidates
    }

    fn contains_typed<R: BlobStoreMeta>(
        &self,
        reader: &R,
        data: TypedData,
        operation: &'static str,
    ) -> Result<bool, ExactDerivedCollectionError> {
        match data {
            TypedData::Source(data) => contains::<_, Source>(reader, data, operation),
            TypedData::Target(data) => contains::<_, Target>(reader, data, operation),
        }
    }
}

struct PreparedDerive<Target: BlobEncoding> {
    output_data: CollectionData,
    output: Blob<Target>,
    claim: CollectionDerive,
}

struct SourcePlan<Source: BlobEncoding> {
    descriptor: CollectionDescriptor,
    semantics: CollectionSemantics,
    collection: CollectionId,
    resident: BTreeSet<CollectionData>,
    mandatory: BTreeMap<CollectionData, Blob<Source>>,
    required_commits: BTreeSet<Id>,
}

struct ExactProbe<R, Source: BlobEncoding, Target: BlobEncoding> {
    reader: R,
    target_cover: ExactCover<Target>,
    missing: BTreeSet<CollectionData>,
    unsupported_commits: BTreeSet<Id>,
    source_plan: Option<SourcePlan<Source>>,
}

impl<R, Source: BlobEncoding, Target: BlobEncoding> ExactProbe<R, Source, Target> {
    fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unsupported_commits.is_empty()
    }

    fn incomplete_error(&self) -> ExactDerivedCollectionError {
        ExactDerivedCollectionError::IncompleteCover {
            missing: self.missing.iter().copied().collect(),
            unsupported_commits: self.unsupported_commits.iter().copied().collect(),
        }
    }

    fn into_target_cover(self) -> ExactCover<Target> {
        self.target_cover
    }
}

impl<R, Source, Target> ExactProbe<R, Source, Target>
where
    R: BlobStoreGet + BlobStoreMeta,
    Source: BlobEncoding + 'static,
    Target: BlobEncoding,
    Handle<Source>: InlineEncoding,
{
    fn source_residual_cover<A>(
        &self,
        algebra: &A,
        blocked: &BTreeMap<CollectionData, String>,
    ) -> Result<Vec<(CollectionData, Blob<Source>)>, ExactDerivedCollectionError>
    where
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        let Some(plan) = &self.source_plan else {
            return Ok(Vec::new());
        };
        let blocked_set: BTreeSet<_> = blocked.keys().copied().collect();
        let resident = plan.resident.difference(&blocked_set).copied().collect();
        let physical = validated_physical_cover(
            &self.reader,
            &plan.semantics,
            plan.collection,
            resident,
            &plan.mandatory,
            |blob| algebra.validate_source(&plan.descriptor, blob),
        );
        if !physical.missing.is_empty() {
            if blocked.is_empty() {
                return Err(ExactDerivedCollectionError::Resolution(format!(
                    "validated source lacks a resident cover for {} frontier element(s)",
                    physical.missing.len(),
                )));
            }
            return Err(ExactDerivedCollectionError::UnrepresentableCover {
                blocked: blocked
                    .iter()
                    .map(|(data, reason)| (*data, reason.clone()))
                    .collect(),
                missing: physical.missing.into_iter().collect(),
            });
        }
        Ok(physical
            .cover
            .into_iter()
            .filter(|data| {
                !plan
                    .semantics
                    .supporting_commit_ids(plan.collection, *data)
                    .is_disjoint(&plan.required_commits)
            })
            .map(|data| {
                let blob = physical
                    .blobs
                    .get(&data)
                    .expect("validated source cover retains selected bytes")
                    .clone();
                (data, blob)
            })
            .collect())
    }
}

struct ValidatedPhysicalCover<E: BlobEncoding> {
    cover: BTreeSet<CollectionData>,
    missing: BTreeSet<CollectionData>,
    blobs: BTreeMap<CollectionData, Blob<E>>,
}

fn validated_physical_cover<R, E, V>(
    reader: &R,
    semantics: &CollectionSemantics,
    collection: CollectionId,
    mut resident: BTreeSet<CollectionData>,
    mandatory: &BTreeMap<CollectionData, Blob<E>>,
    validate: V,
) -> ValidatedPhysicalCover<E>
where
    R: BlobStoreGet + BlobStoreMeta,
    E: BlobEncoding + 'static,
    Handle<E>: InlineEncoding,
    V: Fn(&Blob<E>) -> Result<(), ExactAlgebraError>,
{
    let mut selected = BTreeMap::new();
    loop {
        let physical = collection_physical_cover(semantics, collection, &resident);
        selected.retain(|data, _| physical.cover.contains(data));
        let mut rejected = Vec::new();
        for data in physical.cover.iter().copied() {
            if mandatory.contains_key(&data) || selected.contains_key(&data) {
                continue;
            }
            let handle = Handle::<E>::from_hash(data);
            let actual: Result<Blob<E>, _> = reader.get(handle);
            match actual {
                Ok(actual) if fresh_data_identity(&actual) == data && validate(&actual).is_ok() => {
                    selected.insert(data, actual);
                }
                Ok(_) | Err(_) => rejected.push(data),
            }
        }
        if rejected.is_empty() {
            let mut blobs = selected;
            for data in &physical.cover {
                if let Some(blob) = mandatory.get(data) {
                    blobs.insert(*data, blob.clone());
                }
            }
            return ValidatedPhysicalCover {
                cover: physical.cover,
                missing: physical.missing,
                blobs,
            };
        }
        for data in rejected {
            resident.remove(&data);
        }
    }
}

fn evaluate_candidates<Source, Target, A>(
    candidates: &[Candidate],
    candidate_indices: &BTreeSet<usize>,
    roots: &BTreeSet<TypedData>,
    known: &mut BTreeMap<TypedData, ScratchValue<Source, Target>>,
    source_descriptor: &CollectionDescriptor,
    target_descriptor: &CollectionDescriptor,
    algebra: &A,
) -> (BTreeSet<Id>, BTreeMap<Id, String>)
where
    Source: BlobEncoding,
    Target: BlobEncoding,
    A: ExactDerivedAlgebra<Source, Target> + ?Sized,
{
    let mut missing = vec![u8::MAX; candidates.len()];
    let mut waiters = BTreeMap::<TypedData, Vec<usize>>::new();
    let mut remaining_uses = BTreeMap::<TypedData, usize>::new();
    let mut ready = BTreeSet::new();

    for &index in candidate_indices {
        let candidate = candidates[index];
        let (first, second) = candidate.inputs();
        let mut count = 0u8;
        for input in [Some(first), second].into_iter().flatten() {
            *remaining_uses.entry(input).or_default() += 1;
            if !known.contains_key(&input) {
                waiters.entry(input).or_default().push(index);
                count += 1;
            }
        }
        missing[index] = count;
        if count == 0 {
            ready.insert((candidate.id(), candidate.kind_order(), index));
        }
    }

    let mut accepted = BTreeSet::new();
    let mut rejected = BTreeMap::new();
    while let Some((_, _, index)) = ready.pop_first() {
        let candidate = candidates[index];
        let result = candidate.result();
        match evaluate_candidate(candidate, known, algebra) {
            Ok(value) => {
                let actual = match &value {
                    ScratchValue::Source(blob) => fresh_data_identity(blob),
                    ScratchValue::Target(blob) => fresh_data_identity(blob),
                };
                let representation = match &value {
                    ScratchValue::Source(blob) => algebra.validate_source(source_descriptor, blob),
                    ScratchValue::Target(blob) => algebra.validate_target(target_descriptor, blob),
                };
                if actual != result.data() {
                    rejected.insert(
                        candidate.id(),
                        format!(
                            "canonical result hashes to {} instead of {}",
                            hex::encode_upper(actual.raw),
                            hex::encode_upper(result.data().raw),
                        ),
                    );
                } else {
                    match representation {
                        Err(ExactAlgebraError::Fatal(reason)) => {
                            rejected.insert(candidate.id(), reason);
                        }
                        Err(ExactAlgebraError::Capacity(reason)) => {
                            rejected.insert(
                                candidate.id(),
                                format!("canonical operation exceeded representation capacity: {reason}"),
                            );
                        }
                        Ok(()) => {
                            let retain_result =
                                remaining_uses.get(&result).copied().unwrap_or_default() > 0;
                            let inserted = !known.contains_key(&result) && retain_result;
                            if inserted {
                                known.insert(result, value);
                            }
                            accepted.insert(candidate.id());
                            if inserted {
                                for dependent_index in waiters.remove(&result).unwrap_or_default() {
                                    debug_assert!(
                                        missing[dependent_index] > 0
                                            && missing[dependent_index] <= 2
                                    );
                                    missing[dependent_index] -= 1;
                                    if missing[dependent_index] == 0 {
                                        let dependent = candidates[dependent_index];
                                        ready.insert((
                                            dependent.id(),
                                            dependent.kind_order(),
                                            dependent_index,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(ExactAlgebraError::Fatal(reason)) => {
                rejected.insert(candidate.id(), reason);
            }
            Err(ExactAlgebraError::Capacity(reason)) => {
                rejected.insert(
                    candidate.id(),
                    format!("canonical operation exceeded representation capacity: {reason}"),
                );
            }
        }

        let (first, second) = candidate.inputs();
        for input in [Some(first), second].into_iter().flatten() {
            let uses = remaining_uses
                .get_mut(&input)
                .expect("candidate inputs have reference counts");
            debug_assert!(*uses > 0);
            *uses -= 1;
            if *uses == 0 && !roots.contains(&input) {
                known.remove(&input);
            }
        }
    }

    (accepted, rejected)
}

fn evaluate_candidate<Source, Target, A>(
    candidate: Candidate,
    known: &BTreeMap<TypedData, ScratchValue<Source, Target>>,
    algebra: &A,
) -> Result<ScratchValue<Source, Target>, ExactAlgebraError>
where
    Source: BlobEncoding,
    Target: BlobEncoding,
    A: ExactDerivedAlgebra<Source, Target> + ?Sized,
{
    match candidate {
        Candidate::SourceMerge(claim) => {
            let (low, high) = claim.inputs();
            let Some(ScratchValue::Source(low)) = known.get(&TypedData::Source(low)) else {
                return Err(ExactAlgebraError::Fatal(
                    "source merge became ready without its low input".to_owned(),
                ));
            };
            let Some(ScratchValue::Source(high)) = known.get(&TypedData::Source(high)) else {
                return Err(ExactAlgebraError::Fatal(
                    "source merge became ready without its high input".to_owned(),
                ));
            };
            algebra.join_source(low, high).map(ScratchValue::Source)
        }
        Candidate::Derive(claim) => {
            let input = claim.mapping().0;
            let Some(ScratchValue::Source(input)) = known.get(&TypedData::Source(input)) else {
                return Err(ExactAlgebraError::Fatal(
                    "derive became ready without its source input".to_owned(),
                ));
            };
            algebra.derive(input).map(ScratchValue::Target)
        }
        Candidate::TargetMerge(claim) => {
            let (low, high) = claim.inputs();
            let Some(ScratchValue::Target(low)) = known.get(&TypedData::Target(low)) else {
                return Err(ExactAlgebraError::Fatal(
                    "target merge became ready without its low input".to_owned(),
                ));
            };
            let Some(ScratchValue::Target(high)) = known.get(&TypedData::Target(high)) else {
                return Err(ExactAlgebraError::Fatal(
                    "target merge became ready without its high input".to_owned(),
                ));
            };
            algebra.join_target(low, high).map(ScratchValue::Target)
        }
    }
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

pub(super) fn fresh_data_identity<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData {
    Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
}

#[cfg(test)]
mod tests;
