//! Exact-cover attachment shared by canonical derived collections.
//!
//! Concrete facades bind one [`CollectionMapping`] to typed source and target
//! descriptors, then choose a final logical view. Stored `MERGE` and `DERIVE`
//! equations are materialized LSM work: resolution consumes them without
//! replaying their algebra. When completion executes a map, it persists the
//! source, target, and equation before returning. Yard/GC policy alone decides
//! when reusable artifacts leave local storage.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use crate::trible::Fragment;

use super::discovery::discover_collection_records_for_derived_cover;
use super::{
    collection_physical_cover, collection_physical_cover_for,
    resolve_collection_semantics_from_roots, Collection, CollectionClaimValidation, CollectionData,
    CollectionDerive, CollectionEncoding, CollectionHandle, CollectionMapping, CollectionMerge,
    CollectionOperationError, CollectionRead, CollectionRecord, CollectionSemantics,
    CollectionStore, CollectionValidationRequest, Cover,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;
type MappingSource<M> = <M as CollectionMapping>::Source;
type MappingTarget<M> = <M as CollectionMapping>::Target;

#[derive(Clone, Copy)]
enum SourceRoute {
    SupportEquivalent,
    ExactMembers,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ProbeScope {
    Direct,
    SupportEquivalent,
    ExactMembers,
}

/// Failure to attach or complete one exact derived cover.
#[derive(Debug)]
pub enum ExactDerivedCollectionError {
    /// A storage operation failed.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Backend failure.
        source: BoxError,
    },
    /// The supplied value is not an exact resident source cover.
    InvalidCover(String),
    /// Resolution, identity, or freshly constructed evidence was invalid.
    Resolution(String),
    /// The target does not yet physically and logically cover the source.
    IncompleteCover {
        /// Missing target-frontier bytes.
        missing: Vec<CollectionData>,
        /// Source-cover members absent from logical target support.
        unsupported_members: Vec<CollectionData>,
    },
    /// Canonical source-to-target construction failed.
    Derive {
        /// Source member being lowered.
        input: CollectionData,
        /// Concrete construction failure.
        reason: String,
    },
    /// No physical source cover can represent every required member after
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
            Self::InvalidCover(reason) => write!(f, "invalid exact cover: {reason}"),
            Self::Resolution(reason) => write!(f, "resolve derived collection: {reason}"),
            Self::IncompleteCover {
                missing,
                unsupported_members,
            } => write!(
                f,
                "derived collection is incomplete ({} missing target element(s), {} unsupported source member(s))",
                missing.len(),
                unsupported_members.len(),
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

/// Exact-cover lifecycle for one fixed source-to-target mapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactDerivedCollection<Mapping: CollectionMapping> {
    source_collection: Collection<MappingSource<Mapping>>,
    target_collection: Collection<MappingTarget<Mapping>>,
    mapping_override: Option<Mapping>,
}

impl<Mapping: CollectionMapping> ExactDerivedCollection<Mapping> {
    /// Bind exact maintenance to two collections returned by store lifecycle APIs.
    ///
    /// Descriptor bytes and mapping parameters are loaded and validated from
    /// the operation's immutable store snapshot. This constructor never hashes
    /// or reconstructs a descriptor from caller-supplied facts.
    pub fn new(
        source_collection: Collection<MappingSource<Mapping>>,
        target_collection: Collection<MappingTarget<Mapping>>,
    ) -> Result<Self, ExactDerivedCollectionError> {
        Self::construct(source_collection, target_collection, None)
    }

    /// Bind an already constructed observational mapping implementation.
    /// Kept crate-private so public callers cannot swap semantic behavior
    /// between operations on one typed lifecycle.
    pub(crate) fn with_mapping(
        source_collection: Collection<MappingSource<Mapping>>,
        target_collection: Collection<MappingTarget<Mapping>>,
        mapping: Mapping,
    ) -> Result<Self, ExactDerivedCollectionError> {
        Self::construct(source_collection, target_collection, Some(mapping))
    }

    fn construct(
        source_collection: Collection<MappingSource<Mapping>>,
        target_collection: Collection<MappingTarget<Mapping>>,
        mapping_override: Option<Mapping>,
    ) -> Result<Self, ExactDerivedCollectionError> {
        if source_collection.handle() == target_collection.handle() {
            return Err(ExactDerivedCollectionError::Resolution(
                "exact derived collection requires distinct source and target descriptors"
                    .to_owned(),
            ));
        }
        Ok(Self {
            source_collection,
            target_collection,
            mapping_override,
        })
    }

    fn load_descriptors<R>(
        &self,
        reader: &R,
    ) -> Result<(Fragment, Fragment), ExactDerivedCollectionError>
    where
        R: BlobStoreGet,
    {
        let source =
            super::api::load_collection_descriptor(reader, self.source_collection.handle())
                .map_err(|error| {
                    ExactDerivedCollectionError::Resolution(format!(
                        "load exact source descriptor: {error}"
                    ))
                })?
                .fragment;
        super::encoding::validate_descriptor_type::<MappingSource<Mapping>>(&source).map_err(
            |error| {
                ExactDerivedCollectionError::Resolution(format!(
                    "invalid exact source descriptor: {error}"
                ))
            },
        )?;

        let target =
            super::api::load_collection_descriptor(reader, self.target_collection.handle())
                .map_err(|error| {
                    ExactDerivedCollectionError::Resolution(format!(
                        "load exact target descriptor: {error}"
                    ))
                })?
                .fragment;
        super::encoding::validate_descriptor_type::<MappingTarget<Mapping>>(&target).map_err(
            |error| {
                ExactDerivedCollectionError::Resolution(format!(
                    "invalid exact target descriptor: {error}"
                ))
            },
        )?;

        let declared_source = super::descriptor::source(target.facts()).map_err(|error| {
            ExactDerivedCollectionError::Resolution(format!(
                "invalid target collection_source: {error}"
            ))
        })?;
        if declared_source != Some(self.source_collection.handle()) {
            return Err(ExactDerivedCollectionError::Resolution(format!(
                "target collection_source {:?} does not match source descriptor {}",
                declared_source.map(|handle| hex::encode_upper(handle.raw)),
                hex::encode_upper(self.source_collection.handle().raw),
            )));
        }
        Ok((source, target))
    }

    /// Identity of the source collection.
    pub fn source_collection(&self) -> Collection<MappingSource<Mapping>> {
        self.source_collection
    }

    /// Identity of the target collection.
    pub fn target_collection(&self) -> Collection<MappingTarget<Mapping>> {
        self.target_collection
    }

    pub(crate) fn mapping_override(&self) -> Option<&Mapping> {
        self.mapping_override.as_ref()
    }

    fn require_source_cover(
        &self,
        source_cover: &Cover<MappingSource<Mapping>>,
    ) -> Result<(), ExactDerivedCollectionError> {
        if source_cover.collection() == self.source_collection {
            return Ok(());
        }
        Err(ExactDerivedCollectionError::InvalidCover(format!(
            "source descriptor {} differs from {}",
            hex::encode_upper(source_cover.collection().handle().raw),
            hex::encode_upper(self.source_collection.handle().raw),
        )))
    }

    /// Attach an already complete exact cover without writing.
    ///
    /// Stored equations are followed as materialized LSM evidence. Attachment
    /// performs no joins, mappings, or payload validation; it only selects
    /// resident members. The eventual typed view owns payload decoding.
    pub fn attach<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.attach_with_route(store, source_cover, SourceRoute::SupportEquivalent)
    }

    /// Attach only images of the exact physical source members.
    ///
    /// This is crate-private because it is a mapping law, not a caller tuning
    /// knob. Source-bound encodings such as Rank9 acceleration use it when an
    /// equal-support decomposition is not the same physical input cover.
    pub(crate) fn attach_member_images<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.attach_with_route(store, source_cover, SourceRoute::ExactMembers)
    }

    /// Complete missing derivations, then attach through a fresh read pass.
    ///
    /// Empty covers perform no I/O. A complete first probe returns without
    /// writes. Deterministic capacity excludes the selected source member and
    /// globally replans under the same snapshot. Every successful mapping is
    /// persisted even when a later capacity or fatal result changes the final
    /// plan. The reader is dropped before source and output blobs are written
    /// ahead of their `DERIVE` records. No flush or signed record is emitted.
    pub fn ensure<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.ensure_with_route(store, source_cover, SourceRoute::SupportEquivalent)
    }

    /// Ensure one target image for every exact physical source member.
    pub(crate) fn ensure_member_images<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.ensure_with_route(store, source_cover, SourceRoute::ExactMembers)
    }

    fn ensure_with_route<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        route: SourceRoute,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(Cover::from_members(self.target_collection, []));
        }

        let probe = match route {
            SourceRoute::SupportEquivalent => self.probe(store, source_cover, true)?,
            SourceRoute::ExactMembers => {
                self.probe_once(store, source_cover, true, ProbeScope::ExactMembers)?
            }
        };
        if probe.is_complete() {
            return Ok(probe.into_target_cover());
        }
        let bound_mapping = Mapping::bind(&probe.source_descriptor, &probe.target_descriptor)
            .map_err(|error| {
                ExactDerivedCollectionError::Resolution(format!(
                    "invalid exact collection mapping: {error}"
                ))
            })?;
        let mapping = self.mapping_override.as_ref().unwrap_or(&bound_mapping);
        // Capacity belongs to a selected physical source member, not to its
        // logical support. Excluding that member can expose a completely
        // different overlap-aware cover, so every capacity result restarts
        // global cover selection against this same reader/resolution snapshot.
        // Successful images are retained by source identity. Planning may
        // later choose another overlapping cover, but executed algebra is
        // still useful LSM work and is always published.
        let mut blocked = BTreeMap::<CollectionData, String>::new();
        let mut cached = BTreeMap::<
            CollectionData,
            PreparedDerive<MappingSource<Mapping>, MappingTarget<Mapping>>,
        >::new();
        let plan = 'planning: loop {
            let source_cover = match probe.source_residual_cover(&blocked) {
                Ok(source_cover) => source_cover,
                Err(error) => break Err(error),
            };
            if source_cover.is_empty() {
                break Err(probe.incomplete_error());
            }

            let mut replan = None;
            for (input_data, input) in source_cover {
                if !cached.contains_key(&input_data) {
                    let output = match mapping.map(&input, &probe.reader) {
                        Ok(output) => output,
                        Err(CollectionOperationError::Fatal(reason)) => {
                            break 'planning Err(ExactDerivedCollectionError::Derive {
                                input: input_data,
                                reason,
                            });
                        }
                        Err(CollectionOperationError::Capacity(reason)) => {
                            replan = Some((input_data, reason));
                            break;
                        }
                    };
                    let output_data = data_identity::<MappingTarget<Mapping>>(&output);
                    let claim = CollectionDerive::new(
                        self.target_collection.handle(),
                        input_data,
                        output_data,
                    );
                    cached.insert(
                        input_data,
                        PreparedDerive {
                            input: input.clone(),
                            output,
                            claim,
                        },
                    );
                }
            }

            if let Some((input, reason)) = replan {
                blocked.insert(input, reason);
                continue;
            }
            break Ok(());
        };

        // Never retain an observed store snapshot across publication.
        drop(probe);
        for prepared in cached.values() {
            store
                .put::<MappingSource<Mapping>, _>(prepared.input.clone())
                .map_err(|error| {
                    ExactDerivedCollectionError::storage("store derived source", error)
                })?;
            store
                .put::<MappingTarget<Mapping>, _>(prepared.output.clone())
                .map_err(|error| {
                    ExactDerivedCollectionError::storage("store derived target", error)
                })?;
            store
                .insert(CollectionRecord::Derive(prepared.claim))
                .map_err(|error| ExactDerivedCollectionError::storage("publish DERIVE", error))?;
        }

        plan?;

        // Publication may activate resident target equations which the
        // pre-write snapshot could not yet observe. Resolve once more from a
        // fresh snapshot so an already materialized MERGE is selected instead
        // of handing its inputs to a compactor which would recompute it. This
        // pass is metadata-only: stored equations are trusted and no algebra
        // is replayed.
        self.attach_with_route(store, source_cover, route)
    }

    fn attach_with_route<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        route: SourceRoute,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(Cover::from_members(self.target_collection, []));
        }
        let probe = match route {
            SourceRoute::SupportEquivalent => self.probe(store, source_cover, false)?,
            SourceRoute::ExactMembers => {
                self.probe_once(store, source_cover, false, ProbeScope::ExactMembers)?
            }
        };
        if !probe.is_complete() {
            return Err(probe.incomplete_error());
        }
        Ok(probe.into_target_cover())
    }

    fn probe<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        plan_source_residual: bool,
    ) -> Result<ExactProbe<S::Snapshot, MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        // Preserve the low-latency direct path. Reverse decomposition may fan
        // out over many unsigned MERGE observations, so consult it only when
        // explicit Cover members and their ordinary resident images cannot
        // already answer the request.
        let direct = self.probe_once(
            store,
            source_cover,
            plan_source_residual,
            ProbeScope::Direct,
        )?;
        if direct.is_complete() {
            return Ok(direct);
        }
        drop(direct);
        self.probe_once(
            store,
            source_cover,
            plan_source_residual,
            ProbeScope::SupportEquivalent,
        )
    }

    fn probe_once<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        plan_source_residual: bool,
        scope: ProbeScope,
    ) -> Result<ExactProbe<S::Snapshot, MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.require_source_cover(source_cover)?;
        let reader = store.snapshot().map_err(|error| {
            ExactDerivedCollectionError::storage("open exact-cover snapshot", error)
        })?;
        let (source_descriptor, target_descriptor) = self.load_descriptors(&reader)?;
        Mapping::bind(&source_descriptor, &target_descriptor).map_err(|error| {
            ExactDerivedCollectionError::Resolution(format!(
                "invalid exact collection mapping: {error}"
            ))
        })?;
        let discovered = discover_collection_records_for_derived_cover(
            &reader,
            source_cover,
            self.target_collection.handle(),
        )
        .map_err(|error| ExactDerivedCollectionError::storage("discover exact cover", error))?;

        // Cover identities are sufficient for lookup. Warm attachment never
        // reads source payloads; an incomplete ensure loads only the physical
        // residual members it actually maps.
        let roots: BTreeSet<_> = source_cover.data_members().collect();

        // A Cover may name a compacted source member `c` while stored LSM
        // evidence says `a join b = c`. Support-equivalent derivation may walk
        // that exact reverse lineage. No bytes are reconstructed here: every
        // equation is already a materialized store record.
        let mut semantic_roots = roots.clone();
        if scope == ProbeScope::SupportEquivalent {
            let mut producers = BTreeMap::<CollectionData, Vec<CollectionMerge>>::new();
            for claim in discovered
                .merges()
                .iter()
                .filter(|claim| claim.collection() == self.source_collection.handle())
                .copied()
            {
                producers.entry(claim.result()).or_default().push(claim);
            }
            let mut decomposition_queue: VecDeque<_> = roots.iter().copied().collect();
            while let Some(result) = decomposition_queue.pop_front() {
                for claim in producers.get(&result).into_iter().flatten() {
                    let (low, high) = claim.inputs();
                    for input in [low, high] {
                        if semantic_roots.insert(input) {
                            decomposition_queue.push_back(input);
                        }
                    }
                }
            }
        }

        // This kernel holds both descriptors, so it can state the lineage the
        // derive records observe.
        let lineage = BTreeMap::from([(
            self.target_collection.handle(),
            self.source_collection.handle(),
        )]);
        let explicit_roots: BTreeSet<_> = semantic_roots
            .iter()
            .map(|member| (self.source_collection.handle(), *member))
            .collect();
        let resolution = resolve_collection_semantics_from_roots(
            &discovered,
            &lineage,
            &explicit_roots,
            |request| {
                let verdict = match request {
                    CollectionValidationRequest::Merge { claim }
                        if scope != ProbeScope::ExactMembers
                            && (claim.collection() == self.source_collection.handle()
                                || claim.collection() == self.target_collection.handle()) =>
                    {
                        CollectionClaimValidation::Accepted
                    }
                    CollectionValidationRequest::Derive { claim }
                        if claim.collection() == self.target_collection.handle()
                            && (scope != ProbeScope::ExactMembers
                                || roots.contains(&claim.input())) =>
                    {
                        CollectionClaimValidation::Accepted
                    }
                    CollectionValidationRequest::Commit { .. }
                    | CollectionValidationRequest::Merge { .. }
                    | CollectionValidationRequest::Derive { .. } => {
                        CollectionClaimValidation::Pending
                    }
                };
                Ok::<CollectionClaimValidation<()>, std::convert::Infallible>(verdict)
            },
        );
        let resolution = match resolution {
            Ok(resolution) => resolution,
            Err(super::CollectionResolutionError::Validation { source, .. }) => match source {},
            Err(super::CollectionResolutionError::Conflict(conflict)) => {
                return Err(ExactDerivedCollectionError::Resolution(
                    conflict.to_string(),
                ));
            }
        };

        let target = self.target_collection.handle();
        let logically_supported: BTreeSet<_> = resolution
            .semantics()
            .frontier(target)
            .into_iter()
            .flatten()
            .flat_map(|data| resolution.semantics().supporting_data(target, *data))
            .collect();
        let source_members: BTreeSet<_> = source_cover.data_members().collect();
        let source = self.source_collection.handle();

        // Compare supports in the source lattice, not as raw handle sets. A
        // stored `a join b = c` makes Covers `{a, b}` and `{c}` distinct
        // physical representations of the same support. Both directions are
        // required: target support must not escape the supplied Cover, and it
        // must jointly discharge every supplied Cover member.
        let escaped = collection_physical_cover_for(
            resolution.semantics(),
            source,
            &logically_supported,
            &source_members,
        );
        if let Some(member) = escaped.missing.first() {
            return Err(ExactDerivedCollectionError::InvalidCover(format!(
                "target frontier escaped the source cover through member {}",
                hex::encode_upper(member.raw),
            )));
        }
        let unsupported_members = collection_physical_cover_for(
            resolution.semantics(),
            source,
            &source_members,
            &logically_supported,
        )
        .missing;

        let mut target_resident = BTreeSet::new();
        for data in resolution
            .semantics()
            .members(target)
            .into_iter()
            .flatten()
            .copied()
        {
            if reader
                .metadata(Handle::<MappingTarget<Mapping>>::from_hash(data))
                .map_err(|error| {
                    ExactDerivedCollectionError::storage("inspect exact target residency", error)
                })?
                .is_some()
            {
                target_resident.insert(data);
            }
        }
        let target_physical =
            collection_physical_cover(resolution.semantics(), target, &target_resident);

        let complete = target_physical.missing.is_empty() && unsupported_members.is_empty();
        let source_plan_parts = if !plan_source_residual || complete {
            None
        } else {
            let source = self.source_collection.handle();
            let mut source_resident = BTreeSet::new();
            for data in resolution
                .semantics()
                .members(source)
                .into_iter()
                .flatten()
                .copied()
            {
                if reader
                    .metadata(Handle::<MappingSource<Mapping>>::from_hash(data))
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage(
                            "inspect exact source residency",
                            error,
                        )
                    })?
                    .is_some()
                {
                    source_resident.insert(data);
                }
            }
            // A logically supported target may have lost its bytes. Its
            // support is required work just like a root missing logically.
            let mut required = unsupported_members.clone();
            for data in &target_physical.missing {
                required.extend(resolution.semantics().supporting_data(target, *data));
            }

            Some((source, source_resident, required))
        };
        let source_plan =
            source_plan_parts.map(|(collection, resident, required_members)| SourcePlan {
                semantics: resolution.into_semantics(),
                collection,
                resident,
                required_members,
            });

        Ok(ExactProbe {
            reader,
            source_descriptor,
            target_descriptor,
            target_cover: Cover::from_data(
                self.target_collection,
                target_physical.cover.iter().copied(),
            ),
            missing: target_physical.missing,
            unsupported_members,
            source_plan,
        })
    }
}

struct PreparedDerive<Source: CollectionEncoding, Target: CollectionEncoding> {
    input: Blob<Source>,
    output: Blob<Target>,
    claim: CollectionDerive,
}

struct SourcePlan {
    semantics: CollectionSemantics,
    collection: CollectionHandle,
    resident: BTreeSet<CollectionData>,
    required_members: BTreeSet<CollectionData>,
}

struct ExactProbe<R, Target: CollectionEncoding> {
    reader: R,
    source_descriptor: Fragment,
    target_descriptor: Fragment,
    target_cover: Cover<Target>,
    missing: BTreeSet<CollectionData>,
    unsupported_members: BTreeSet<CollectionData>,
    source_plan: Option<SourcePlan>,
}

impl<R, Target: CollectionEncoding> ExactProbe<R, Target> {
    fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unsupported_members.is_empty()
    }

    fn incomplete_error(&self) -> ExactDerivedCollectionError {
        ExactDerivedCollectionError::IncompleteCover {
            missing: self.missing.iter().copied().collect(),
            unsupported_members: self.unsupported_members.iter().copied().collect(),
        }
    }

    fn into_target_cover(self) -> Cover<Target> {
        self.target_cover
    }
}

impl<R, Target> ExactProbe<R, Target>
where
    R: BlobStoreGet + BlobStoreMeta,
    Target: CollectionEncoding,
{
    fn source_residual_cover<Source: CollectionEncoding>(
        &self,
        blocked: &BTreeMap<CollectionData, String>,
    ) -> Result<Vec<(CollectionData, Blob<Source>)>, ExactDerivedCollectionError> {
        let Some(plan) = &self.source_plan else {
            return Ok(Vec::new());
        };
        let blocked_set: BTreeSet<_> = blocked.keys().copied().collect();
        let resident = plan.resident.difference(&blocked_set).copied().collect();
        let physical: LoadedPhysicalCover<Source> =
            loaded_physical_cover(&self.reader, &plan.semantics, plan.collection, resident)?;
        if !physical.missing.is_empty() {
            if blocked.is_empty() {
                return Err(ExactDerivedCollectionError::Resolution(format!(
                    "source lacks a resident cover for {} frontier element(s)",
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
        let required_physical = collection_physical_cover_for(
            &plan.semantics,
            plan.collection,
            &plan.required_members,
            &physical.cover,
        );
        if !required_physical.missing.is_empty() {
            return Err(ExactDerivedCollectionError::Resolution(format!(
                "selected source cover does not discharge {} required support element(s)",
                required_physical.missing.len(),
            )));
        }
        Ok(physical
            .cover
            .into_iter()
            .filter(|data| required_physical.cover.contains(data))
            .map(|data| {
                let blob = physical
                    .blobs
                    .get(&data)
                    .expect("loaded source cover retains selected bytes")
                    .clone();
                (data, blob)
            })
            .collect())
    }
}

struct LoadedPhysicalCover<E: CollectionEncoding> {
    cover: BTreeSet<CollectionData>,
    missing: BTreeSet<CollectionData>,
    blobs: BTreeMap<CollectionData, Blob<E>>,
}

fn loaded_physical_cover<R, E>(
    reader: &R,
    semantics: &CollectionSemantics,
    collection: CollectionHandle,
    resident: BTreeSet<CollectionData>,
) -> Result<LoadedPhysicalCover<E>, ExactDerivedCollectionError>
where
    R: BlobStoreGet,
    E: CollectionEncoding,
    Handle<E>: InlineEncoding,
{
    let physical = collection_physical_cover(semantics, collection, &resident);
    let mut blobs = BTreeMap::new();
    for data in physical.cover.iter().copied() {
        let root = reader.get(Handle::<E>::from_hash(data)).map_err(|error| {
            ExactDerivedCollectionError::storage("load exact source member", error)
        })?;
        blobs.insert(data, root);
    }
    Ok(LoadedPhysicalCover {
        cover: physical.cover,
        missing: physical.missing,
        blobs,
    })
}

pub(super) fn data_identity<E: CollectionEncoding>(blob: &Blob<E>) -> CollectionData {
    Handle::<E>::to_hash(blob.get_handle())
}

#[cfg(test)]
mod tests;
