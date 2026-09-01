//! Generic exact-cover maintenance across two collection lattices.
//!
//! A [`CollectionMapping`] supplies only the join-homomorphic conversion;
//! storage owns the singular schedule that combines source joins, target joins,
//! and mappings. Domain facades merely bind typed descriptors and choose a
//! final logical view. Stored `MERGE` and `DERIVE` equations are materialized
//! LSM work: resolution consumes them without replaying their algebra. When
//! completion performs new work, it persists the member and equation before
//! returning. Yard/GC policy alone decides when reusable artifacts leave local
//! storage.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use crate::blob::Blob;
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use crate::trible::Fragment;

use super::discovery::discover_collection_records_for_derived_cover;
use super::encoding::{collection_member_availability, CollectionMemberAvailability};
use super::{
    collection_complete_physical_cover, collection_physical_cover, collection_physical_cover_for,
    resolve_collection_semantics_from_roots, Collection, CollectionClaimValidation, CollectionData,
    CollectionDerive, CollectionEncoding, CollectionHandle, CollectionMapping, CollectionMerge,
    CollectionOperationError, CollectionRead, CollectionRecord, CollectionSemantics,
    CollectionStore, CollectionValidationRequest, Cover,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;
type MappingSource<M> = <M as CollectionMapping>::Source;
type MappingTarget<M> = <M as CollectionMapping>::Target;

#[derive(Clone, Copy, Eq, PartialEq)]
enum ProbeScope {
    Direct,
    SupportEquivalent,
}

pub(super) type TargetMergePair = (CollectionData, CollectionData);

#[derive(Default)]
pub(super) struct ExactPlannerBlocks {
    pub(super) sources: BTreeMap<CollectionData, String>,
    pub(super) target_merges: BTreeSet<TargetMergePair>,
}

/// Failure to attach or maintain one exact derived cover.
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
    /// The target encoding could not join one deterministic LSM pair.
    Merge {
        /// Canonically lower input content identity.
        low: CollectionData,
        /// Canonically higher input content identity.
        high: CollectionData,
        /// Concrete construction failure.
        reason: String,
    },
    /// A canonical operation needs one immutable blob which is absent from the
    /// current store snapshot.
    MissingDependency {
        /// Exact missing content identity.
        member: CollectionData,
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
    /// Fresh attachment repeated an unstable physical target cover.
    Stalled {
        /// Repeated cover in canonical content-handle order.
        cover: Vec<CollectionData>,
    },
}

impl ExactDerivedCollectionError {
    pub(super) fn storage(
        operation: &'static str,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
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
            Self::Merge { low, high, reason } => write!(
                f,
                "merge target elements {} and {}: {reason}",
                hex::encode_upper(low.raw),
                hex::encode_upper(high.raw),
            ),
            Self::MissingDependency { member } => write!(
                f,
                "derived collection requires resident blob {}",
                hex::encode_upper(member.raw),
            ),
            Self::UnrepresentableCover { blocked, missing } => write!(
                f,
                "exact source cover is unrepresentable ({} capacity-terminal member(s), {} uncovered source obligation(s))",
                blocked.len(),
                missing.len(),
            ),
            Self::Stalled { cover } => write!(
                f,
                "target maintenance repeated an unstable {}-member cover",
                cover.len(),
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
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(Cover::from_members(self.target_collection, []));
        }
        let blocked_target_merges = BTreeSet::new();
        let blocked_sources = BTreeMap::new();
        let probe = self.probe(
            store,
            source_cover,
            false,
            &blocked_sources,
            &blocked_target_merges,
        )?;
        if !probe.is_complete() {
            return Err(probe.incomplete_error());
        }
        Ok(probe.into_target_cover())
    }

    /// Ensure the deterministic maintained target cover.
    ///
    /// At each source-lattice point, an exact resident target image is reused;
    /// otherwise two exact resident target children take precedence over the
    /// corresponding resident source node, and only then does planning
    /// descend to source children. Source work is never synthesized merely for
    /// target compaction policy; it is materialized only when the selected
    /// target join explicitly names its exact source-lattice result as a
    /// missing representation dependency. Every join or mapping actually
    /// executed is persisted before this method returns.
    pub fn ensure<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        super::exact_target_compaction::ensure_target(self, store, source_cover)
    }

    /// Complete the per-point target choices before global size-tiered maintenance.
    ///
    /// This internal phase executes exact target-child carries and source
    /// mappings, re-probing after every preferred operation. Global LSM carries
    /// alternate with it through the singular public [`Self::ensure`] path.
    pub(super) fn complete<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        blocks: &mut ExactPlannerBlocks,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.ensure_with_blocks(store, source_cover, blocks)
    }

    /// Materialize the exact source join requested by one target join.
    ///
    /// This is the only inverse-looking step in collection maintenance, and it
    /// is still ordinary forward lattice work. The resolved mapping relation
    /// supplies every canonical preimage of the two target inputs. A candidate
    /// source join is published only when its content identity is exactly the
    /// dependency named by the target encoding; non-injective mappings cannot
    /// make the executor guess.
    pub(super) fn materialize_target_join_dependency<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        target_low: CollectionData,
        target_high: CollectionData,
        dependency: CollectionData,
    ) -> Result<bool, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        let blocked_sources = BTreeMap::new();
        let blocked_target_merges = BTreeSet::new();
        let probe = self.probe(
            store,
            source_cover,
            false,
            &blocked_sources,
            &blocked_target_merges,
        )?;
        let source = self.source_collection.handle();
        let target = self.target_collection.handle();
        let low_preimages = probe.semantics.derive_preimages(source, target, target_low);
        let high_preimages = probe
            .semantics
            .derive_preimages(source, target, target_high);
        let mut candidates = BTreeSet::new();
        for low in low_preimages {
            for high in high_preimages.iter().copied() {
                let pair = if low <= high {
                    (low, high)
                } else {
                    (high, low)
                };
                candidates.insert(pair);
            }
        }

        let mut prepared = None;
        let mut nested_dependencies = BTreeSet::new();
        for (low_data, high_data) in candidates {
            let low_handle = Handle::<MappingSource<Mapping>>::from_hash(low_data);
            let high_handle = Handle::<MappingSource<Mapping>>::from_hash(high_data);
            let low_resident = probe.reader.metadata(low_handle).map_err(|error| {
                ExactDerivedCollectionError::storage("inspect source dependency-merge input", error)
            })?;
            let high_resident = probe.reader.metadata(high_handle).map_err(|error| {
                ExactDerivedCollectionError::storage("inspect source dependency-merge input", error)
            })?;
            if low_resident.is_none() || high_resident.is_none() {
                continue;
            }
            let low = probe.reader.get(low_handle).map_err(|error| {
                ExactDerivedCollectionError::storage("load source dependency-merge input", error)
            })?;
            let high = probe.reader.get(high_handle).map_err(|error| {
                ExactDerivedCollectionError::storage("load source dependency-merge input", error)
            })?;
            let output = match MappingSource::<Mapping>::join_members(
                &probe.source_descriptor,
                &low,
                &high,
                &probe.reader,
            ) {
                Ok(output) => output,
                Err(CollectionOperationError::Capacity(_)) => continue,
                Err(CollectionOperationError::MissingDependency(member)) => {
                    nested_dependencies.insert(member);
                    continue;
                }
                Err(CollectionOperationError::Fatal(reason)) => {
                    return Err(ExactDerivedCollectionError::Merge {
                        low: low_data,
                        high: high_data,
                        reason,
                    });
                }
            };
            if data_identity::<MappingSource<Mapping>>(&output) != dependency {
                continue;
            }
            prepared = Some((
                output,
                CollectionMerge::new(source, low_data, high_data, dependency),
            ));
            break;
        }

        drop(probe);
        let Some((output, claim)) = prepared else {
            if let Some(member) = nested_dependencies.into_iter().next() {
                return Err(ExactDerivedCollectionError::MissingDependency { member });
            }
            return Ok(false);
        };
        store
            .put::<MappingSource<Mapping>, _>(output)
            .map_err(|error| {
                ExactDerivedCollectionError::storage("store source dependency merge", error)
            })?;
        store
            .insert(CollectionRecord::Merge(claim))
            .map_err(|error| {
                ExactDerivedCollectionError::storage("publish source dependency MERGE", error)
            })?;
        Ok(true)
    }

    fn ensure_with_blocks<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        blocks: &mut ExactPlannerBlocks,
    ) -> Result<Cover<MappingTarget<Mapping>>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Snapshot: BlobStoreMeta + CollectionRead,
    {
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(Cover::from_members(self.target_collection, []));
        }

        let mut published_target_merges = BTreeSet::<(CollectionData, CollectionData)>::new();
        let mut published_source_derives = BTreeSet::<CollectionData>::new();

        loop {
            let probe = self.probe(
                store,
                source_cover,
                true,
                &blocks.sources,
                &blocks.target_merges,
            )?;
            if probe.is_complete() {
                return Ok(probe.into_target_cover());
            }

            // Exact resident target children are the first constructive route
            // at their source-lattice point. Preserve those precise pairs;
            // global size-tiered maintenance must not substitute an unrelated
            // pair or skip a cross-tier carry.
            if !probe.preferred_target_merges.is_empty() {
                let stalled_cover = probe
                    .target_cover
                    .members()
                    .map(Handle::<MappingTarget<Mapping>>::to_hash)
                    .collect();
                let (low_data, high_data) = *probe
                    .preferred_target_merges
                    .first()
                    .expect("non-empty preferred target-merge set has a first pair");
                if published_target_merges.contains(&(low_data, high_data)) {
                    return Err(ExactDerivedCollectionError::Stalled {
                        cover: stalled_cover,
                    });
                }
                let low = probe
                    .reader
                    .get(Handle::<MappingTarget<Mapping>>::from_hash(low_data))
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage(
                            "load preferred target-merge input",
                            error,
                        )
                    })?;
                let high = probe
                    .reader
                    .get(Handle::<MappingTarget<Mapping>>::from_hash(high_data))
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage(
                            "load preferred target-merge input",
                            error,
                        )
                    })?;
                let output = match MappingTarget::<Mapping>::join_members(
                    &probe.target_descriptor,
                    &low,
                    &high,
                    &probe.reader,
                ) {
                    Ok(output) => output,
                    Err(CollectionOperationError::Capacity(_)) => {
                        blocks.target_merges.insert((low_data, high_data));
                        continue;
                    }
                    Err(CollectionOperationError::Fatal(reason)) => {
                        return Err(ExactDerivedCollectionError::Merge {
                            low: low_data,
                            high: high_data,
                            reason,
                        });
                    }
                    Err(CollectionOperationError::MissingDependency(member)) => {
                        drop(probe);
                        if self.materialize_target_join_dependency(
                            store,
                            source_cover,
                            low_data,
                            high_data,
                            member,
                        )? {
                            continue;
                        }
                        return Err(ExactDerivedCollectionError::MissingDependency { member });
                    }
                };
                let output_data = data_identity::<MappingTarget<Mapping>>(&output);
                let claim = CollectionMerge::new(
                    self.target_collection.handle(),
                    low_data,
                    high_data,
                    output_data,
                );

                // Publish one carry, then re-enter the per-point planner. The
                // new result may itself be an exact child which changes the
                // next preferred route.
                drop(probe);
                store
                    .put::<MappingTarget<Mapping>, _>(output)
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage("store preferred target merge", error)
                    })?;
                store
                    .insert(CollectionRecord::Merge(claim))
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage(
                            "publish preferred target MERGE",
                            error,
                        )
                    })?;
                published_target_merges.insert(claim.inputs());
                continue;
            }

            let bound_mapping = Mapping::bind(&probe.source_descriptor, &probe.target_descriptor)
                .map_err(|error| {
                ExactDerivedCollectionError::Resolution(format!(
                    "invalid exact collection mapping: {error}"
                ))
            })?;
            let mapping = self.mapping_override.as_ref().unwrap_or(&bound_mapping);

            // Capacity belongs to a selected physical source member, not to
            // its logical support. Excluding that member can expose another
            // overlap-aware cover. Successful images are still persisted even
            // when a later capacity result requires a fresh semantic probe.
            let mut cached = BTreeMap::<
                CollectionData,
                PreparedDerive<MappingSource<Mapping>, MappingTarget<Mapping>>,
            >::new();
            let plan = 'planning: loop {
                let source_cover = match probe.source_residual_cover(&blocks.sources) {
                    Ok(source_cover) => source_cover,
                    Err(error) => break Err(error),
                };
                if source_cover.is_empty() {
                    break if probe.semantically_complete() {
                        Ok(())
                    } else {
                        Err(probe.incomplete_error())
                    };
                }

                let mut replan = None;
                for (input_data, input) in source_cover {
                    if published_source_derives.contains(&input_data) {
                        break 'planning Err(ExactDerivedCollectionError::Stalled {
                            cover: probe
                                .target_cover
                                .members()
                                .map(Handle::<MappingTarget<Mapping>>::to_hash)
                                .collect(),
                        });
                    }
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
                            Err(CollectionOperationError::MissingDependency(member)) => {
                                break 'planning Err(
                                    ExactDerivedCollectionError::MissingDependency { member },
                                );
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
                    blocks.sources.insert(input, reason);
                    continue;
                }
                break Ok(());
            };

            // Never retain an observed store snapshot across publication.
            drop(probe);
            let published = !cached.is_empty();
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
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage("publish DERIVE", error)
                    })?;
                published_source_derives.insert(prepared.claim.input());
            }

            match plan {
                Ok(()) if published => continue,
                Ok(()) => return self.attach(store, source_cover),
                Err(
                    ExactDerivedCollectionError::UnrepresentableCover { .. }
                    | ExactDerivedCollectionError::IncompleteCover { .. },
                ) if published => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn probe<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        plan_source_residual: bool,
        blocked_sources: &BTreeMap<CollectionData, String>,
        blocked_target_merges: &BTreeSet<TargetMergePair>,
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
            blocked_sources,
            blocked_target_merges,
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
            blocked_sources,
            blocked_target_merges,
        )
    }

    fn probe_once<S>(
        &self,
        store: &mut S,
        source_cover: &Cover<MappingSource<Mapping>>,
        plan_source_residual: bool,
        scope: ProbeScope,
        blocked_sources: &BTreeMap<CollectionData, String>,
        blocked_target_merges: &BTreeSet<TargetMergePair>,
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
                        if claim.collection() == self.source_collection.handle()
                            || claim.collection() == self.target_collection.handle() =>
                    {
                        CollectionClaimValidation::Accepted
                    }
                    CollectionValidationRequest::Derive { claim }
                        if claim.collection() == self.target_collection.handle() =>
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

        let semantics = resolution.into_semantics();
        let target = self.target_collection.handle();
        let logically_supported: BTreeSet<_> = semantics
            .frontier(target)
            .into_iter()
            .flatten()
            .flat_map(|data| semantics.supporting_data(target, *data))
            .collect();
        let source_members: BTreeSet<_> = source_cover.data_members().collect();
        let source = self.source_collection.handle();

        // Compare supports in the source lattice, not as raw handle sets. A
        // stored `a join b = c` makes Covers `{a, b}` and `{c}` distinct
        // physical representations of the same support. Both directions are
        // required: target support must not escape the supplied Cover, and it
        // must jointly discharge every supplied Cover member.
        let escaped = collection_physical_cover_for(
            &semantics,
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
            &semantics,
            source,
            &source_members,
            &logically_supported,
        )
        .missing;

        // Root metadata is the cheap common path. First ask the ordinary
        // closure-aware resolver whether the frontier already has one complete
        // physical representative. A warm compacted target therefore reads
        // only the selected root's representation metadata, rather than every
        // historical target member. Planning alternatives needs the full
        // complete resident set only when the selected cover is incomplete or
        // still contains more than one member.
        let mut target_roots = BTreeSet::new();
        for data in semantics.members(target).into_iter().flatten().copied() {
            if reader
                .metadata(Handle::<MappingTarget<Mapping>>::from_hash(data))
                .map_err(|error| {
                    ExactDerivedCollectionError::storage("inspect exact target residency", error)
                })?
                .is_some()
            {
                target_roots.insert(data);
            }
        }
        let selected_target = collection_complete_physical_cover::<MappingTarget<Mapping>, _>(
            &semantics,
            target,
            &target_roots,
            &reader,
        );
        let mut target_resident = selected_target.physical.cover.clone();
        let mut target_physical = selected_target.physical;
        if !target_physical.missing.is_empty() || target_physical.cover.len() > 1 {
            target_resident.clear();
            for data in target_roots {
                match collection_member_availability::<MappingTarget<Mapping>, _>(data, &reader)
                    .map_err(|error| {
                        ExactDerivedCollectionError::storage(
                            "inspect exact target representation closure",
                            error,
                        )
                    })? {
                    CollectionMemberAvailability::Complete => {
                        target_resident.insert(data);
                    }
                    CollectionMemberAvailability::Absent
                    | CollectionMemberAvailability::Incomplete
                    | CollectionMemberAvailability::Unusable => {}
                }
            }
            target_physical = collection_physical_cover(&semantics, target, &target_resident);
        }
        let represented_source: BTreeSet<_> = target_physical
            .cover
            .iter()
            .flat_map(|data| semantics.supporting_data(target, *data))
            .collect();

        let semantically_complete =
            target_physical.missing.is_empty() && unsupported_members.is_empty();
        let plan_preferences =
            plan_source_residual && !(semantically_complete && target_physical.cover.len() <= 1);
        let source_resident =
            if plan_source_residual && (!semantically_complete || plan_preferences) {
                let mut source_resident = BTreeSet::new();
                for data in semantics.members(source).into_iter().flatten().copied() {
                    match collection_member_availability::<MappingSource<Mapping>, _>(data, &reader)
                        .map_err(|error| {
                            ExactDerivedCollectionError::storage(
                                "inspect exact source residency",
                                error,
                            )
                        })? {
                        CollectionMemberAvailability::Complete => {
                            source_resident.insert(data);
                        }
                        CollectionMemberAvailability::Absent
                        | CollectionMemberAvailability::Incomplete
                        | CollectionMemberAvailability::Unusable => {}
                    }
                }
                source_resident
            } else {
                BTreeSet::new()
            };
        let preferred = if plan_preferences {
            preferred_routes(
                &semantics,
                source,
                target,
                &source_resident,
                &target_resident,
                &represented_source,
                blocked_sources,
                blocked_target_merges,
            )
        } else {
            PreferredRoutes::default()
        };

        let complete = semantically_complete
            && preferred.source_members.is_empty()
            && preferred.target_merges.is_empty();
        let source_plan_parts = if !plan_source_residual || complete {
            None
        } else {
            // A logically supported target may have lost its bytes. Its
            // support is required work just like a root missing logically.
            let mut required = unsupported_members.clone();
            for data in &target_physical.missing {
                required.extend(semantics.supporting_data(target, *data));
            }
            required.extend(preferred.source_members.iter().copied());

            Some((source, source_resident, required))
        };
        let source_plan =
            source_plan_parts.map(|(collection, resident, required_members)| SourcePlan {
                collection,
                resident,
                represented_source,
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
            preferred_source_members: preferred.source_members,
            preferred_target_merges: preferred.target_merges,
            semantics,
            source_plan,
        })
    }
}

#[derive(Clone, Default)]
struct PreferredRoutes {
    source_members: BTreeSet<CollectionData>,
    target_merges: BTreeSet<(CollectionData, CollectionData)>,
}

/// Select exact target carries and corresponding resident source nodes before recursively
/// constructing lower target members.
///
/// For each maximal source point the choice is local and deterministic:
/// reuse its exact resident target image; otherwise prefer a target join when
/// one canonical source decomposition already has both exact target images;
/// otherwise cross the mapping if the source point itself is resident; only
/// then descend through the first fully actionable canonical source
/// decomposition. Route selection never requests a source join. Execution may
/// materialize one later only when the selected target join names its exact
/// result as an immutable representation dependency.
fn preferred_routes(
    semantics: &CollectionSemantics,
    source: CollectionHandle,
    target: CollectionHandle,
    source_resident: &BTreeSet<CollectionData>,
    target_resident: &BTreeSet<CollectionData>,
    represented_source: &BTreeSet<CollectionData>,
    blocked_sources: &BTreeMap<CollectionData, String>,
    blocked_target_merges: &BTreeSet<TargetMergePair>,
) -> PreferredRoutes {
    fn target_image_is_resident(
        semantics: &CollectionSemantics,
        source: CollectionHandle,
        target: CollectionHandle,
        input: CollectionData,
        target_resident: &BTreeSet<CollectionData>,
    ) -> bool {
        semantics
            .derive_output(source, target, input)
            .is_some_and(|output| target_resident.contains(&output))
    }

    fn target_pair(
        semantics: &CollectionSemantics,
        source: CollectionHandle,
        target: CollectionHandle,
        low: CollectionData,
        high: CollectionData,
        target_resident: &BTreeSet<CollectionData>,
    ) -> Option<(CollectionData, CollectionData)> {
        let mut low = semantics.derive_output(source, target, low)?;
        let mut high = semantics.derive_output(source, target, high)?;
        if !target_resident.contains(&low) || !target_resident.contains(&high) {
            return None;
        }
        if high < low {
            std::mem::swap(&mut low, &mut high);
        }
        Some((low, high))
    }

    #[allow(clippy::too_many_arguments)]
    fn visit(
        semantics: &CollectionSemantics,
        source: CollectionHandle,
        target: CollectionHandle,
        member: CollectionData,
        source_resident: &BTreeSet<CollectionData>,
        target_resident: &BTreeSet<CollectionData>,
        represented_source: &BTreeSet<CollectionData>,
        blocked_sources: &BTreeMap<CollectionData, String>,
        blocked_target_merges: &BTreeSet<TargetMergePair>,
        allow_equivalent_cover: bool,
        visiting: &mut BTreeSet<CollectionData>,
        selected: &mut PreferredRoutes,
    ) -> bool {
        if !visiting.insert(member) {
            return false;
        }
        if target_image_is_resident(semantics, source, target, member, target_resident) {
            visiting.remove(&member);
            return true;
        }

        let producers = semantics.merge_producers(source, member);
        if let Some(pair) = producers
            .iter()
            .filter_map(|(low, high)| {
                target_pair(semantics, source, target, *low, *high, target_resident)
            })
            .find(|pair| !blocked_target_merges.contains(pair))
        {
            selected.target_merges.insert(pair);
            visiting.remove(&member);
            return true;
        }

        let allow_equivalent_cover =
            allow_equivalent_cover || blocked_sources.contains_key(&member);
        if allow_equivalent_cover {
            let obligation = BTreeSet::from([member]);
            if collection_physical_cover_for(semantics, source, &obligation, represented_source)
                .missing
                .is_empty()
            {
                visiting.remove(&member);
                return true;
            }
        }

        if source_resident.contains(&member) && !blocked_sources.contains_key(&member) {
            selected.source_members.insert(member);
            visiting.remove(&member);
            return true;
        }

        // Follow the first canonical producer whose entire subtree is
        // physically actionable. This is the same backtracking shape as
        // physical-cover selection: an unavailable earlier decomposition must
        // not hide exact preferences in a later one.
        for (low, high) in producers {
            let mut candidate = selected.clone();
            let low_is_actionable = visit(
                semantics,
                source,
                target,
                low,
                source_resident,
                target_resident,
                represented_source,
                blocked_sources,
                blocked_target_merges,
                allow_equivalent_cover,
                visiting,
                &mut candidate,
            );
            let high_is_actionable = low_is_actionable
                && visit(
                    semantics,
                    source,
                    target,
                    high,
                    source_resident,
                    target_resident,
                    represented_source,
                    blocked_sources,
                    blocked_target_merges,
                    allow_equivalent_cover,
                    visiting,
                    &mut candidate,
                );
            if high_is_actionable {
                *selected = candidate;
                visiting.remove(&member);
                return true;
            }
        }
        visiting.remove(&member);
        false
    }

    let mut selected = PreferredRoutes::default();
    let mut visiting = BTreeSet::new();
    for member in semantics.frontier(source).into_iter().flatten().copied() {
        let _ = visit(
            semantics,
            source,
            target,
            member,
            source_resident,
            target_resident,
            represented_source,
            blocked_sources,
            blocked_target_merges,
            false,
            &mut visiting,
            &mut selected,
        );
    }
    selected
}

struct PreparedDerive<Source: CollectionEncoding, Target: CollectionEncoding> {
    input: Blob<Source>,
    output: Blob<Target>,
    claim: CollectionDerive,
}

struct SourcePlan {
    collection: CollectionHandle,
    resident: BTreeSet<CollectionData>,
    represented_source: BTreeSet<CollectionData>,
    required_members: BTreeSet<CollectionData>,
}

struct ExactProbe<R, Target: CollectionEncoding> {
    reader: R,
    source_descriptor: Fragment,
    target_descriptor: Fragment,
    target_cover: Cover<Target>,
    missing: BTreeSet<CollectionData>,
    unsupported_members: BTreeSet<CollectionData>,
    preferred_source_members: BTreeSet<CollectionData>,
    preferred_target_merges: BTreeSet<(CollectionData, CollectionData)>,
    semantics: CollectionSemantics,
    source_plan: Option<SourcePlan>,
}

impl<R, Target: CollectionEncoding> ExactProbe<R, Target> {
    fn is_complete(&self) -> bool {
        self.semantically_complete()
            && self.preferred_source_members.is_empty()
            && self.preferred_target_merges.is_empty()
    }

    fn semantically_complete(&self) -> bool {
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

        // Preferred corresponding source nodes are route optimizations over an already
        // complete lower target cover. Treat capacity independently at each
        // point: skip only blocked nodes and keep trying the others. Their
        // bytes are known resident in this immutable snapshot, so no source
        // reconstruction is necessary for the fallback path.
        if self.semantically_complete() {
            return self
                .preferred_source_members
                .difference(&blocked_set)
                .copied()
                .map(|data| {
                    self.reader
                        .get(Handle::<Source>::from_hash(data))
                        .map(|blob| (data, blob))
                        .map_err(|error| {
                            ExactDerivedCollectionError::storage(
                                "load preferred corresponding source node",
                                error,
                            )
                        })
                })
                .collect();
        }

        let resident = plan.resident.difference(&blocked_set).copied().collect();
        let physical: LoadedPhysicalCover<Source> =
            loaded_physical_cover(&self.reader, &self.semantics, plan.collection, resident)?;
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
            &self.semantics,
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
            .filter(|data| {
                if self.preferred_source_members.contains(data) {
                    return true;
                }
                let obligation = BTreeSet::from([*data]);
                !collection_physical_cover_for(
                    &self.semantics,
                    plan.collection,
                    &obligation,
                    &plan.represented_source,
                )
                .missing
                .is_empty()
            })
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
