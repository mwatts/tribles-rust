//! Exact-cover attachment shared by canonical derived collections.
//!
//! Concrete facades supply descriptors, one canonical five-operation algebra,
//! and final query materialization. This kernel owns the common I/O lifecycle
//! around opaque source-cover roots and reproducible unsigned evidence.
//!
//! Unsigned equations are cache evidence, not durable validation receipts.
//! Resolution walks backwards from resident source and target results, then
//! recomputes that finite proof graph forwards from explicit source-cover
//! members and any validated source decompositions rooted beneath them.
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
use crate::repo::{
    ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStoreMeta, BlobStorePut, OfferCapture,
};
use crate::trible::Fragment;

use super::discovery::discover_collection_records_for_derived_cover;
use super::{
    collection_physical_cover, collection_physical_cover_for,
    resolve_collection_semantics_from_roots, CollectionClaimValidation, CollectionData,
    CollectionDerive, CollectionHandle, CollectionMerge, CollectionRecord, CollectionSemantics,
    CollectionStore, Cover, DiscoveredCollectionRecords,
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
/// explicit source-cover member reject the cover.
pub trait ExactDerivedAlgebra<Source: BlobEncoding, Target: BlobEncoding> {
    /// Validate the exact source descriptor and one canonical source element.
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &Blob<Source>,
    ) -> Result<(), ExactAlgebraError>;

    /// Validate the exact target descriptor and one canonical target element.
    fn validate_target(
        &self,
        descriptor: &Fragment,
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

/// Freshly validated bytes attached to one exact semantic [`Cover`].
///
/// This is transient materialization state, not a second kind of cover. The
/// contained `Cover` remains the identity passed between collection stages;
/// the blobs merely avoid rereading members while one operation consumes them.
pub struct CoverAttachment<Target: BlobEncoding> {
    cover: Cover,
    members: Vec<(CollectionData, Blob<Target>)>,
}

impl<Target: BlobEncoding> CoverAttachment<Target> {
    fn empty(collection: CollectionHandle) -> Self {
        Self {
            cover: Cover::from_members(collection, []),
            members: Vec::new(),
        }
    }

    /// Exact semantic cover whose bytes are attached.
    pub fn cover(&self) -> &Cover {
        &self.cover
    }

    /// Number of selected physical members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the attachment is the store-free empty-cover bottom.
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

/// Read-only outcome of probing an exact derived collection with speculative
/// target-artifact availability.
///
/// Offers are operational hints, never semantic evidence. The kernel first
/// validates the source cover's named bytes and recomputes the relevant
/// equations from those explicit roots. Only then may an offered target member appear in
/// [`Self::Fetch`]. A caller that cannot obtain one of those exact handles
/// removes it from the offered set and probes again; ordinary physical-cover
/// selection then chooses another valid cover or reports incompleteness.
pub enum ExactAttachPlan<Target: BlobEncoding> {
    /// Every selected physical member is resident and freshly validated.
    Ready(CoverAttachment<Target>),
    /// Exact target members selected from the offered set but not resident.
    ///
    /// Handles are returned in ascending content-identity order. Fetching them
    /// is deliberately outside this synchronous storage kernel.
    Fetch(Vec<CollectionData>),
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
    /// One source-cover member lacks resident bytes.
    IncompleteMember(CollectionData),
    /// One source-cover member failed concrete validation.
    RejectedMember {
        /// Exact payload identity.
        member: CollectionData,
        /// Concrete diagnostic.
        reason: String,
    },
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
            Self::IncompleteMember(member) => write!(
                f,
                "source cover member {} is incomplete",
                hex::encode_upper(member.raw),
            ),
            Self::RejectedMember { member, reason } => {
                write!(
                    f,
                    "source cover member {} was rejected: {reason}",
                    hex::encode_upper(member.raw),
                )
            }
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

/// Exact-cover lifecycle for one fixed source-to-target homomorphism.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactDerivedCollection<Source: BlobEncoding, Target: BlobEncoding> {
    source: Fragment,
    source_collection: CollectionHandle,
    target: Fragment,
    target_collection: CollectionHandle,
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
    pub fn new(source: Fragment, target: Fragment) -> Self {
        let source_collection: CollectionHandle =
            crate::blob::IntoBlob::<SimpleArchive>::to_blob(source.facts().clone()).get_handle();
        let target_collection: CollectionHandle =
            crate::blob::IntoBlob::<SimpleArchive>::to_blob(target.facts().clone()).get_handle();
        assert_ne!(
            source_collection, target_collection,
            "exact derived collection requires distinct source and target descriptors",
        );
        Self {
            source,
            source_collection,
            target,
            target_collection,
            encodings: PhantomData,
        }
    }

    /// Source descriptor.
    pub fn source_descriptor(&self) -> &Fragment {
        &self.source
    }

    /// Target descriptor.
    pub fn target_descriptor(&self) -> &Fragment {
        &self.target
    }

    /// Identity of the source collection.
    pub fn source_collection(&self) -> CollectionHandle {
        self.source_collection
    }

    /// Identity of the target collection.
    pub fn target_collection(&self) -> CollectionHandle {
        self.target_collection
    }

    fn require_source_cover(
        &self,
        source_cover: &Cover,
    ) -> Result<(), ExactDerivedCollectionError> {
        if source_cover.collection() == self.source_collection {
            return Ok(());
        }
        Err(ExactDerivedCollectionError::InvalidCover(format!(
            "source descriptor {} differs from {}",
            hex::encode_upper(source_cover.collection().raw),
            hex::encode_upper(self.source_collection.raw),
        )))
    }

    /// Attach an already complete exact cover without writing.
    ///
    /// Missing unsigned intermediates are reconstructed in use-counted scratch
    /// from explicit source-cover roots. Scratch validation never publishes a
    /// blob or equation.
    pub fn attach_exact<S, A>(
        &self,
        store: &mut S,
        source_cover: &Cover,
        algebra: &A,
    ) -> Result<CoverAttachment<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        self.attach_with(store, source_cover, algebra)
    }

    /// Probe an exact cover while treating target handles as speculative
    /// remote availability hints.
    ///
    /// This method performs no writes and no network I/O. Unknown, unrelated,
    /// or algebraically invalid offers are ignored. A [`ExactAttachPlan::Fetch`]
    /// result contains only members of the exact physical cover selected by
    /// the ordinary collection resolver. Once valid bytes have been landed
    /// and remain visible under the same immutable record evidence, calling
    /// this method again returns [`ExactAttachPlan::Ready`]. If a fetch fails,
    /// remove that handle from `offered_target` and re-probe.
    ///
    /// Every source Cover member remains mandatory resident evidence. Remote
    /// offers cannot replace those bytes or establish a source decomposition.
    pub fn probe_exact<S, A>(
        &self,
        store: &mut S,
        source_cover: &Cover,
        algebra: &A,
        offered_target: &BTreeSet<CollectionData>,
    ) -> Result<ExactAttachPlan<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(ExactAttachPlan::Ready(CoverAttachment::empty(
                self.target_collection,
            )));
        }
        let probe = self.probe(store, source_cover, algebra, false, offered_target)?;
        if probe.is_complete() {
            return Ok(ExactAttachPlan::Ready(probe.into_target_cover()));
        }
        if probe.missing.is_empty()
            && probe.unsupported_members.is_empty()
            && !probe.target_fetch.is_empty()
        {
            return Ok(ExactAttachPlan::Fetch(
                probe.target_fetch.iter().copied().collect(),
            ));
        }
        Err(probe.incomplete_error())
    }

    /// Complete missing derivations, then attach through a fresh read pass.
    ///
    /// Empty covers perform no I/O. A complete first probe returns without
    /// writes. Deterministic capacity excludes the selected source member and
    /// globally replans under the same snapshot; terminal unrepresentability
    /// returns before any write. For a final feasible plan, the reader is
    /// dropped before descriptors and all output blobs are written ahead of
    /// unsigned `DERIVE` records. No flush or signed record is emitted.
    pub fn ensure_exact<S, A>(
        &self,
        store: &mut S,
        source_cover: &Cover,
        algebra: &A,
    ) -> Result<CoverAttachment<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        let mut capture = OfferCapture::new(store);
        self.ensure_exact_unoffered(&mut capture, source_cover, algebra)
    }

    pub(crate) fn ensure_exact_unoffered<S, A>(
        &self,
        store: &mut S,
        source_cover: &Cover,
        algebra: &A,
    ) -> Result<CoverAttachment<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(CoverAttachment::empty(self.target_collection));
        }

        let probe = self.probe(store, source_cover, algebra, true, &BTreeSet::new())?;
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
                    let claim =
                        CollectionDerive::new(self.target_collection, input_data, output_data);
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

        // Construction does not change the opaque source cover; a fresh
        // attachment validates the outputs just published.
        self.attach_with(store, source_cover, algebra)
    }

    fn attach_with<S, A>(
        &self,
        store: &mut S,
        source_cover: &Cover,
        algebra: &A,
    ) -> Result<CoverAttachment<Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        self.require_source_cover(source_cover)?;
        if source_cover.is_empty() {
            return Ok(CoverAttachment::empty(self.target_collection));
        }
        let probe = self.probe(store, source_cover, algebra, false, &BTreeSet::new())?;
        if !probe.is_complete() {
            return Err(probe.incomplete_error());
        }
        Ok(probe.into_target_cover())
    }

    fn publish_descriptors<S: BlobStorePut>(
        &self,
        store: &mut S,
    ) -> Result<(), ExactDerivedCollectionError> {
        for (descriptor, expected) in [
            (&self.source, self.source_collection),
            (&self.target, self.target_collection),
        ] {
            let blob = crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone());
            let actual = store
                .put::<SimpleArchive, _>(blob)
                .map_err(|error| ExactDerivedCollectionError::storage("store descriptor", error))?;
            if actual != expected {
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
        source_cover: &Cover,
        algebra: &A,
        plan_source_residual: bool,
        offered_target: &BTreeSet<CollectionData>,
    ) -> Result<ExactProbe<S::Reader, Source, Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        // Preserve the low-latency direct path. Reverse decomposition may fan
        // out over many unsigned MERGE observations, so consult it only when
        // explicit Cover members and their ordinary resident images cannot
        // already answer the request.
        let direct = self.probe_once(
            store,
            source_cover,
            algebra,
            plan_source_residual,
            offered_target,
            false,
        )?;
        if direct.is_complete() {
            return Ok(direct);
        }
        drop(direct);
        self.probe_once(
            store,
            source_cover,
            algebra,
            plan_source_residual,
            offered_target,
            true,
        )
    }

    fn probe_once<S, A>(
        &self,
        store: &mut S,
        source_cover: &Cover,
        algebra: &A,
        plan_source_residual: bool,
        offered_target: &BTreeSet<CollectionData>,
        allow_source_decomposition: bool,
    ) -> Result<ExactProbe<S::Reader, Source, Target>, ExactDerivedCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
        A: ExactDerivedAlgebra<Source, Target> + ?Sized,
    {
        self.require_source_cover(source_cover)?;
        let discovered = discover_collection_records_for_derived_cover(
            store,
            source_cover,
            self.target_collection,
        )
        .map_err(|error| ExactDerivedCollectionError::storage("discover exact cover", error))?;
        let reader = store.reader().map_err(|error| {
            ExactDerivedCollectionError::storage("open exact-cover reader", error)
        })?;

        let mut known = BTreeMap::<TypedData, ScratchValue<Source, Target>>::new();
        let mut roots = BTreeSet::new();
        for member in source_cover.members() {
            let node = TypedData::Source(member);
            if !known.contains_key(&node) {
                let Some(blob) =
                    load_candidate::<_, Source>(&reader, member, "read source cover member")?
                else {
                    return Err(ExactDerivedCollectionError::IncompleteMember(member));
                };
                let actual = fresh_data_identity(&blob);
                if actual != member {
                    return Err(ExactDerivedCollectionError::RejectedMember {
                        member,
                        reason: format!(
                            "source bytes hash to {} instead of {}",
                            hex::encode_upper(actual.raw),
                            hex::encode_upper(member.raw),
                        ),
                    });
                }
                algebra
                    .validate_source(&self.source, &blob)
                    .map_err(|error| ExactDerivedCollectionError::RejectedMember {
                        member,
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

        let mut local_results = roots.clone();
        let mut resident_results = roots.clone();

        // A Cover may name a compacted member `c` while resident canonical
        // evidence proves `a join b = c`. Walk only source-MERGE producers of
        // the supplied roots (and their recursive inputs), then load the
        // optional input bytes into scratch. The equations are still accepted
        // only after `evaluate_candidates` recomputes every join forwards.
        // Restricting this walk to producers of Cover roots prevents unrelated
        // resident source data from becoming semantic support by proximity.
        let mut decomposition_seen = roots.clone();
        if allow_source_decomposition {
            let mut decomposition_queue: VecDeque<_> = roots.iter().copied().collect();
            while let Some(result) = decomposition_queue.pop_front() {
                for &index in producers.get(&result).into_iter().flatten() {
                    if !matches!(candidates[index], Candidate::SourceMerge(_)) {
                        continue;
                    }
                    let (first, second) = candidates[index].inputs();
                    for input in [Some(first), second].into_iter().flatten() {
                        if decomposition_seen.insert(input) {
                            decomposition_queue.push_back(input);
                        }
                    }
                }
            }
        }
        for node in decomposition_seen.iter().copied() {
            let TypedData::Source(member) = node else {
                continue;
            };
            if known.contains_key(&node) {
                continue;
            }
            // Decomposition inputs are optional local cache evidence. Never
            // route an absent input through a reader whose miss semantics may
            // record a durable WANT (for example `LazyReader`).
            let Ok(Some(_)) = reader.metadata(Handle::<Source>::from_hash(member)) else {
                continue;
            };
            let Ok(blob) = reader.get(Handle::<Source>::from_hash(member)) else {
                continue;
            };
            if fresh_data_identity(&blob) != member
                || algebra.validate_source(&self.source, &blob).is_err()
            {
                continue;
            }
            known.insert(node, ScratchValue::Source(blob));
            local_results.insert(node);
            resident_results.insert(node);
        }

        // Source compaction results are seeds too: ensure may reuse a resident
        // source upper bound even when no target artifact exists yet.
        let mut reverse_seen = BTreeSet::new();
        let mut reverse_queue = VecDeque::new();
        for result in producers.keys().copied() {
            let offered = match result {
                TypedData::Source(_) => false,
                TypedData::Target(data) => offered_target.contains(&data),
            };
            let local = known.contains_key(&result) || self.contains_typed(&reader, result);
            if local {
                local_results.insert(result);
            }
            if offered || local {
                resident_results.insert(result);
                if reverse_seen.insert(result) {
                    reverse_queue.push_back(result);
                }
            }
        }

        // Include every producer path, including producers of explicit roots:
        // another payload member may be reachable only through that merge
        // history, so first-proof traversal would lose support.
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

        // Only successfully recomputed decompositions become semantic seeds.
        // They are alternative physical leaves of the supplied Cover, not
        // ambient resident data discovered elsewhere in the collection.
        let mut semantic_roots = roots.clone();
        let mut semantic_queue: VecDeque<_> = roots.iter().copied().collect();
        while let Some(result) = semantic_queue.pop_front() {
            for &index in producers.get(&result).into_iter().flatten() {
                let candidate = candidates[index];
                if !matches!(candidate, Candidate::SourceMerge(_))
                    || !accepted.contains(&candidate.id())
                {
                    continue;
                }
                let (first, second) = candidate.inputs();
                for input in [Some(first), second].into_iter().flatten() {
                    if semantic_roots.insert(input) {
                        semantic_queue.push_back(input);
                    }
                }
            }
        }

        // This kernel holds both descriptors, so it can state the lineage the
        // derive records observe.
        let lineage = BTreeMap::from([(self.target_collection, self.source_collection)]);
        let explicit_roots: BTreeSet<_> = semantic_roots
            .iter()
            .filter_map(|node| match node {
                TypedData::Source(member) => Some((self.source_collection, *member)),
                TypedData::Target(_) => None,
            })
            .collect();
        let resolution = resolve_collection_semantics_from_roots(
            &discovered,
            &lineage,
            &explicit_roots,
            |request| {
                let claim = request.claim_id();
                Ok::<CollectionClaimValidation<String>, std::convert::Infallible>(
                    if accepted.contains(&claim) {
                        CollectionClaimValidation::Accepted
                    } else if let Some(reason) = rejected.get(&claim) {
                        CollectionClaimValidation::Rejected(reason.clone())
                    } else {
                        CollectionClaimValidation::Pending
                    },
                )
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

        let target = self.target_collection;
        let logically_supported: BTreeSet<_> = resolution
            .semantics()
            .frontier(target)
            .into_iter()
            .flatten()
            .flat_map(|data| resolution.semantics().supporting_data(target, *data))
            .collect();
        let source_members: BTreeSet<_> = source_cover.members().collect();
        let source = self.source_collection;

        // Compare supports in the source lattice, not as raw handle sets. A
        // validated `a join b = c` makes Covers `{a, b}` and `{c}` distinct
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

        let target_local = resolution
            .semantics()
            .members(target)
            .into_iter()
            .flatten()
            .copied()
            .filter(|data| local_results.contains(&TypedData::Target(*data)))
            .collect();
        let local_physical = validated_physical_cover(
            &reader,
            resolution.semantics(),
            target,
            target_local,
            &BTreeMap::new(),
            &BTreeSet::new(),
            |blob| algebra.validate_target(&self.target, blob),
        );
        // A speculative offer must never displace a complete resident cover.
        // Only widen physical selection to offered members when local bytes do
        // not already answer the exact cover without network I/O.
        let target_physical = if local_physical.missing.is_empty() && unsupported_members.is_empty()
        {
            local_physical
        } else {
            let target_resident = resolution
                .semantics()
                .members(target)
                .into_iter()
                .flatten()
                .copied()
                .filter(|data| resident_results.contains(&TypedData::Target(*data)))
                .collect();
            validated_physical_cover(
                &reader,
                resolution.semantics(),
                target,
                target_resident,
                &BTreeMap::new(),
                offered_target,
                |blob| algebra.validate_target(&self.target, blob),
            )
        };

        let complete = target_physical.missing.is_empty()
            && target_physical.fetch.is_empty()
            && unsupported_members.is_empty();
        let source_plan_parts = if !plan_source_residual || complete {
            None
        } else {
            let source = self.source_collection;
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
            // support is required work just like a root missing logically.
            let mut required = unsupported_members.clone();
            for data in &target_physical.missing {
                required.extend(resolution.semantics().supporting_data(target, *data));
            }

            Some((source, source_resident, source_roots, required))
        };
        let source_plan =
            source_plan_parts.map(|(collection, resident, mandatory, required_members)| {
                SourcePlan {
                    descriptor: self.source.clone(),
                    semantics: resolution.into_semantics(),
                    collection,
                    resident,
                    mandatory,
                    required_members,
                }
            });

        Ok(ExactProbe {
            reader,
            target_cover: target_physical.fetch.is_empty().then(|| CoverAttachment {
                cover: Cover::from_members(
                    self.target_collection,
                    target_physical.cover.iter().copied(),
                ),
                members: target_physical
                    .cover
                    .iter()
                    .map(|data| {
                        (
                            *data,
                            target_physical
                                .blobs
                                .get(data)
                                .expect("resident target cover retains selected bytes")
                                .clone(),
                        )
                    })
                    .collect(),
            }),
            target_fetch: target_physical.fetch,
            missing: target_physical.missing,
            unsupported_members,
            source_plan,
        })
    }

    fn candidates(&self, discovered: &DiscoveredCollectionRecords) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        candidates.extend(
            discovered
                .merges()
                .iter()
                .filter(|claim| claim.collection() == self.source_collection)
                .copied()
                .map(Candidate::SourceMerge),
        );
        candidates.extend(
            discovered
                .derives()
                .iter()
                .filter(|claim| claim.target() == self.target_collection)
                .copied()
                .map(Candidate::Derive),
        );
        candidates.extend(
            discovered
                .merges()
                .iter()
                .filter(|claim| claim.collection() == self.target_collection)
                .copied()
                .map(Candidate::TargetMerge),
        );
        candidates.sort_unstable_by_key(|candidate| (candidate.id(), candidate.kind_order()));
        candidates
    }

    fn contains_typed<R: BlobStoreMeta>(&self, reader: &R, data: TypedData) -> bool {
        match data {
            TypedData::Source(data) => reader
                .metadata(Handle::<Source>::from_hash(data))
                .ok()
                .flatten()
                .is_some(),
            TypedData::Target(data) => reader
                .metadata(Handle::<Target>::from_hash(data))
                .ok()
                .flatten()
                .is_some(),
        }
    }
}

struct PreparedDerive<Target: BlobEncoding> {
    output_data: CollectionData,
    output: Blob<Target>,
    claim: CollectionDerive,
}

struct SourcePlan<Source: BlobEncoding> {
    descriptor: Fragment,
    semantics: CollectionSemantics,
    collection: CollectionHandle,
    resident: BTreeSet<CollectionData>,
    mandatory: BTreeMap<CollectionData, Blob<Source>>,
    required_members: BTreeSet<CollectionData>,
}

struct ExactProbe<R, Source: BlobEncoding, Target: BlobEncoding> {
    reader: R,
    target_cover: Option<CoverAttachment<Target>>,
    target_fetch: BTreeSet<CollectionData>,
    missing: BTreeSet<CollectionData>,
    unsupported_members: BTreeSet<CollectionData>,
    source_plan: Option<SourcePlan<Source>>,
}

impl<R, Source: BlobEncoding, Target: BlobEncoding> ExactProbe<R, Source, Target> {
    fn is_complete(&self) -> bool {
        self.target_cover.is_some()
            && self.target_fetch.is_empty()
            && self.missing.is_empty()
            && self.unsupported_members.is_empty()
    }

    fn incomplete_error(&self) -> ExactDerivedCollectionError {
        ExactDerivedCollectionError::IncompleteCover {
            missing: self.missing.iter().copied().collect(),
            unsupported_members: self.unsupported_members.iter().copied().collect(),
        }
    }

    fn into_target_cover(self) -> CoverAttachment<Target> {
        self.target_cover
            .expect("complete exact probe has a resident target cover")
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
            &BTreeSet::new(),
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
    fetch: BTreeSet<CollectionData>,
}

fn validated_physical_cover<R, E, V>(
    reader: &R,
    semantics: &CollectionSemantics,
    collection: CollectionHandle,
    mut resident: BTreeSet<CollectionData>,
    mandatory: &BTreeMap<CollectionData, Blob<E>>,
    offered: &BTreeSet<CollectionData>,
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
        let mut fetch = BTreeSet::new();
        for data in physical.cover.iter().copied() {
            if mandatory.contains_key(&data) || selected.contains_key(&data) {
                continue;
            }
            let handle = Handle::<E>::from_hash(data);
            // `resident` may include a speculative remote offer. Presence
            // metadata is the non-demanding probe; calling `get` for an
            // absent handle can itself publish a durable WANT on lazy stores.
            match reader.metadata(handle) {
                Ok(Some(_)) => {}
                Ok(None) if offered.contains(&data) => {
                    fetch.insert(data);
                    continue;
                }
                Ok(None) | Err(_) => {
                    rejected.push(data);
                    continue;
                }
            }
            let actual: Result<Blob<E>, _> = reader.get(handle);
            match actual {
                Ok(actual) if fresh_data_identity(&actual) == data && validate(&actual).is_ok() => {
                    selected.insert(data, actual);
                }
                Ok(_) => rejected.push(data),
                Err(_) => rejected.push(data),
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
                fetch,
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
    source_descriptor: &Fragment,
    target_descriptor: &Fragment,
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
