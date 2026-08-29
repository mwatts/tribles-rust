//! A maintained index of the states something has observed.
//!
//! [`resolve`](crate::query::register::resolve) answers domination with one
//! reverse-index probe per candidate. That is cheap, but it is paid on every
//! read, and at ERP row counts the per-candidate form is the difference
//! between seconds and hours — which is why the holdouts in the wild scan
//! the whole collection once and subtract instead.
//!
//! This is that scan, as a derived collection the store maintains.
//!
//! # What is maintained, and why it is the *dominated* half
//!
//! The obvious thing to materialise is the frontier itself. It is the wrong
//! thing, for the reason the taxonomy gives: the head set is **antitone** in
//! the inclusion lattice the store runs on, so a newly arriving commit can
//! *remove* a member, and a derive whose output can shrink is not lawful.
//!
//! The dominated set is the monotone half of the same computation. A commit
//! can only ever add to it, so:
//!
//! ```text
//! observed(C1 union C2) = observed(C1) union observed(C2)
//! ```
//!
//! is a join homomorphism into a plain union lattice — the simplest lattice
//! there is — and the derive is exact, incremental, and order-independent.
//! The reader recovers the frontier by subtraction, which is where the
//! antitone step lives: outside the store, in the reader's frame, exactly
//! where the light-cone argument says currency belongs.
//!
//! So the split is: **the store maintains what accumulates, the reader
//! performs what negates.** Materialising the frontier would have pushed a
//! non-monotone operation into a monotone engine; materialising its
//! complement does not.
//!
//! # What it costs the reader
//!
//! [`ObservedIndex`](crate::collection::observed_union::ObservedIndex)
//! implements [`RegisterOrder`](crate::query::register::RegisterOrder), so it
//! is a drop-in for
//! [`ObservationOrder`](crate::query::register::ObservationOrder) in
//! [`resolve`](crate::query::register::resolve),
//! [`sole`](crate::query::register::sole) and
//! [`maximal`](crate::query::register::maximal). Domination becomes a binary
//! search over a sorted `Vec` rather than a query into the fact source, and
//! nothing else about the call changes.
//!
//! # Identity
//!
//! The observed attribute is a canonical descriptor argument, the way a path
//! collection carries its automaton fingerprint: two registers over the same
//! dataset but different edges are distinct collections, and cannot be
//! confused for one another's maintained artifacts.

use ed25519_dalek::VerifyingKey;

use anybytes::Bytes;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::{Blob, BlobEncoding};
use crate::id::{ExclusiveId, Id};
use crate::id_hex;
use crate::inline::encodings::genid::GenId;
use crate::inline::Inline;
use crate::macros::entity;
// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use crate::collection::reach;
use crate::metadata;
use crate::metadata::MetaDescribe;
use crate::query::register::RegisterOrder;
use crate::trible::{Fragment, A_START, TRIBLE_LEN, V_START};

use super::exact_derived::{ExactDerivedCollection, ExactDerivedCollectionError};
use super::records::{
    collection_authority, collection_reach, collection_recipe, collection_representation,
    collection_source, CollectionHandle, KIND_COLLECTION_DESCRIPTOR,
};
use super::{
    simplearchive_union::{self, SimpleArchiveUnion},
    CollectionHomomorphism, CollectionLattice, CollectionLatticeError, CollectionStore,
    CoverAttachment, FactCover, TryFromCover,
};
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreMeta};

/// Width of one stored id.
const ID_LEN: usize = 16;

crate::macros::attributes! {
    /// The observation attribute a derived observed-set collection reads.
    ///
    /// Minted with `trible genid` on 2026-08-19.
    "E61092974C734142217EC718CC184673" as pub register_observes: GenId;
}

/// Minted with `trible genid` on 2026-08-19.

/// Canonical sorted set of observed state ids.
///
/// The bytes are a strictly increasing sequence of 16-byte ids and nothing
/// else, so the canonical form of a set is unique and the empty set is zero
/// bytes. Strictly increasing rather than merely sorted: a duplicate would
/// give one set two encodings, and the exact-derive kernel compares target
/// bytes for equality.
pub struct ObservedSetBlob;

impl BlobEncoding for ObservedSetBlob {}

impl MetaDescribe for ObservedSetBlob {
    fn describe() -> Fragment {
        // Minted with `trible genid` on 2026-08-19.
        let id: Id = id_hex!("3C98E1A6F691E8EE888F3F49D10B8CF2");
        entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "observed-set-v1",
            metadata::description: "Strictly increasing sequence of 16-byte state ids that some entity observes over one fixed attribute. The monotone half of register resolution: readers subtract this set from their candidates to obtain the frontier. Empty is zero bytes.",
            metadata::tag: metadata::KIND_BLOB_ENCODING,
        }
    }
}

/// Canonical observed-set validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservedSetError {
    /// The payload is not a whole number of ids.
    BadLength(usize),
    /// The ids are not strictly increasing.
    NotStrictlyIncreasing,
}

impl fmt::Display for ObservedSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadLength(len) => {
                write!(
                    formatter,
                    "observed set of {len} bytes is not a whole number of {ID_LEN}-byte ids"
                )
            }
            Self::NotStrictlyIncreasing => {
                formatter.write_str("observed set ids are not strictly increasing")
            }
        }
    }
}

impl Error for ObservedSetError {}

/// Validate one canonical observed-set element.
pub fn validate_element(blob: &Blob<ObservedSetBlob>) -> Result<(), ObservedSetError> {
    let bytes = blob.bytes.as_ref();
    if bytes.len() % ID_LEN != 0 {
        return Err(ObservedSetError::BadLength(bytes.len()));
    }
    if bytes
        .chunks_exact(ID_LEN)
        .zip(bytes.chunks_exact(ID_LEN).skip(1))
        .any(|(low, high)| low >= high)
    {
        return Err(ObservedSetError::NotStrictlyIncreasing);
    }
    Ok(())
}

/// The canonical empty observed set.
pub fn empty() -> Blob<ObservedSetBlob> {
    Blob::new(Bytes::from_source(Vec::<u8>::new()))
}

/// Canonically derive one observed set from a `SimpleArchive`.
///
/// Every trible written under `observes` contributes its **value** — the
/// state that was observed, and therefore the state that has been moved
/// past. Values that are not well-formed ids are skipped rather than
/// rejected: an unrelated encoding stored under the same attribute is not
/// evidence about any register, and this derivation must never fail on
/// facts it simply has no opinion about.
pub fn derive_element(
    source: &Blob<SimpleArchive>,
    observes: Id,
) -> Result<Blob<ObservedSetBlob>, ObservedSetError> {
    let bytes = source.bytes.as_ref();
    if bytes.len() % TRIBLE_LEN != 0 {
        return Err(ObservedSetError::BadLength(bytes.len()));
    }
    let mut observed: Vec<[u8; ID_LEN]> = Vec::new();
    for trible in bytes.chunks_exact(TRIBLE_LEN) {
        if trible[A_START..A_START + ID_LEN] != observes[..] {
            continue;
        }
        let value = &trible[V_START..V_START + 32];
        // A GenId keeps the id in the low 16 bytes and zeroes the high 16.
        if value[0..16].iter().any(|&byte| byte != 0) {
            continue;
        }
        let low: [u8; ID_LEN] = value[16..32].try_into().expect("16-byte tail");
        if low.iter().all(|&byte| byte == 0) {
            continue;
        }
        observed.push(low);
    }
    observed.sort_unstable();
    observed.dedup();
    Ok(Blob::new(Bytes::from_source(observed.concat())))
}

/// The canonical union of two observed sets.
pub fn join(
    low: &Blob<ObservedSetBlob>,
    high: &Blob<ObservedSetBlob>,
) -> Result<Blob<ObservedSetBlob>, ObservedSetError> {
    validate_element(low)?;
    validate_element(high)?;
    let left = low.bytes.as_ref();
    let right = high.bytes.as_ref();
    let mut merged: Vec<u8> = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < left.len() && j < right.len() {
        let a = &left[i..i + ID_LEN];
        let b = &right[j..j + ID_LEN];
        match a.cmp(b) {
            std::cmp::Ordering::Less => {
                merged.extend_from_slice(a);
                i += ID_LEN;
            }
            std::cmp::Ordering::Greater => {
                merged.extend_from_slice(b);
                j += ID_LEN;
            }
            std::cmp::Ordering::Equal => {
                merged.extend_from_slice(a);
                i += ID_LEN;
                j += ID_LEN;
            }
        }
    }
    merged.extend_from_slice(&left[i..]);
    merged.extend_from_slice(&right[j..]);
    Ok(Blob::new(Bytes::from_source(merged)))
}

/// Construct the observed-set collection for one source and edge.
///
/// The target's mandatory authority is explicit and independent of its source.
pub fn descriptor(
    source: CollectionHandle,
    observes: Id,
    authority: VerifyingKey,
    reach: Fragment,
) -> Fragment {
    let observes: Inline<GenId> = crate::inline::IntoInline::to_inline(observes);
    let fragment = entity! { _ @
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_source: source,
        collection_authority: authority,
        collection_representation*: <ObservedSetBlob as MetaDescribe>::describe(),
        collection_recipe*: <ObservedUnionV1 as MetaDescribe>::describe(),
        register_observes: observes,
        collection_reach*: reach,
    };
    fragment
}

/// The observed-union law.
///
/// This names the law only. Which attribute is observed is a parameter on the
/// descriptor entity, not folded into this id: a digest of an unstored
/// argument would make the collection's meaning unrecoverable from the pile.
pub const OBSERVED_UNION_RECIPE_V1: Id = id_hex!("A808ECA30730EF0F1C7FD96F3FC7CB03");

/// The observed-union law, as a describable type.
pub struct ObservedUnionV1;

impl MetaDescribe for ObservedUnionV1 {
    fn describe() -> Fragment {
        let id: Id = OBSERVED_UNION_RECIPE_V1;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "observed-union-v1",
                metadata::description: "Union of the state ids observed over one attribute: the monotone half of register resolution. Readers subtract this set from their candidates to obtain the frontier, which is why the set itself only ever grows and merges by union. Takes one argument, carried as a trible on the collection descriptor: `register_observes`, the attribute whose observations are accumulated.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

/// The canonical observed-set representation under the observed-union law.
pub struct ObservedSetUnion;

fn observed_attribute(descriptor: &Fragment) -> Result<Id, CollectionLatticeError> {
    let raw = crate::collection::descriptor::argument(descriptor.facts(), register_observes.id())
        .map_err(|source| CollectionLatticeError::Fatal(source.to_string()))?
        .ok_or_else(|| {
            CollectionLatticeError::Fatal(
                "observed-set descriptor is missing register_observes".to_owned(),
            )
        })?;
    Inline::<GenId>::new(raw)
        .try_from_inline::<Id>()
        .map_err(|source| {
            CollectionLatticeError::Fatal(format!(
                "observed-set descriptor has an invalid register_observes: {source:?}"
            ))
        })
}

impl CollectionLattice for ObservedSetUnion {
    type Encoding = ObservedSetBlob;
    type Recipe = ObservedUnionV1;

    fn validate_arguments(descriptor: &Fragment) -> Result<(), CollectionLatticeError> {
        let source = crate::collection::descriptor::source(descriptor.facts())
            .map_err(|source| CollectionLatticeError::Fatal(source.to_string()))?;
        if source.is_none() {
            return Err(CollectionLatticeError::Fatal(
                "observed-set descriptor is missing its source collection".to_owned(),
            ));
        }

        observed_attribute(descriptor).map(|_| ())
    }

    fn validate_member(
        _descriptor: &Fragment,
        member: &Blob<Self::Encoding>,
    ) -> Result<(), CollectionLatticeError> {
        validate_element(member).map_err(|source| CollectionLatticeError::Fatal(source.to_string()))
    }

    fn merge_members(
        _descriptor: &Fragment,
        low: &Blob<Self::Encoding>,
        high: &Blob<Self::Encoding>,
    ) -> Result<Blob<Self::Encoding>, CollectionLatticeError> {
        join(low, high).map_err(|source| CollectionLatticeError::Fatal(source.to_string()))
    }
}

/// Bound projection from one fact-set member to its observed-state set.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ObserveStates {
    observes: Id,
}

impl CollectionHomomorphism<SimpleArchiveUnion, ObservedSetUnion> for ObserveStates {
    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionLatticeError> {
        Ok(Self {
            observes: observed_attribute(target)?,
        })
    }

    fn map(
        &self,
        source: &Blob<SimpleArchive>,
    ) -> Result<Blob<ObservedSetBlob>, CollectionLatticeError> {
        derive_element(source, self.observes)
            .map_err(|source| CollectionLatticeError::Fatal(source.to_string()))
    }
}

/// A resolved observed set, ready to answer domination.
///
/// Implements [`RegisterOrder`], so it substitutes for a live
/// [`ObservationOrder`](crate::query::register::ObservationOrder) anywhere
/// the substrate takes an order — the difference is a binary search instead
/// of an index probe, and nothing else.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObservedIndex {
    observed: Vec<[u8; ID_LEN]>,
}

impl ObservedIndex {
    /// Decode a validated observed set.
    pub fn decode(blob: &Blob<ObservedSetBlob>) -> Result<Self, ObservedSetError> {
        validate_element(blob)?;
        Ok(Self {
            observed: blob
                .bytes
                .as_ref()
                .chunks_exact(ID_LEN)
                .map(|chunk| chunk.try_into().expect("16-byte chunk"))
                .collect(),
        })
    }

    /// How many distinct states have been observed.
    ///
    /// An exact count, so a caller that wants the planner to order around
    /// resolution has a real cardinality to hand it.
    pub fn len(&self) -> usize {
        self.observed.len()
    }

    /// Whether nothing has been observed yet.
    pub fn is_empty(&self) -> bool {
        self.observed.is_empty()
    }
}

impl RegisterOrder for ObservedIndex {
    fn dominated(&self, state: Id) -> bool {
        let raw: [u8; ID_LEN] = state[..].try_into().expect("id is 16 bytes");
        self.observed.binary_search(&raw).is_ok()
    }
}

impl TryFromCover<ObservedSetUnion> for ObservedIndex {
    type Error = ObservedSetError;

    fn try_from_cover(attachment: CoverAttachment<ObservedSetUnion>) -> Result<Self, Self::Error> {
        let mut joined = empty();
        for segment in attachment.into_blobs() {
            joined = join(&joined, &segment)?;
        }
        Self::decode(&joined)
    }
}

/// Canonical observed-set projection of one source `SimpleArchive`
/// collection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedSetCollection {
    name: String,
    source_authority: VerifyingKey,
    observes: Id,
    source_reach: Fragment,
    authority: VerifyingKey,
    reach: Fragment,
}

impl ObservedSetCollection {
    /// Construct the observed-set projection over one named root.
    ///
    /// `source_reach` completes the root's identity; `reach` is this
    /// projection's own. A derivation never inherits its source's reach --
    /// see [`reach::travels`](crate::collection::reach::travels).
    /// `source_authority` must match the root descriptor exactly; `authority`
    /// independently declares the derived collection's mandatory trust root.
    pub fn new(
        name: impl Into<String>,
        source_authority: VerifyingKey,
        observes: Id,
        source_reach: Fragment,
        authority: VerifyingKey,
        reach: Fragment,
    ) -> Self {
        Self {
            name: name.into(),
            source_authority,
            observes,
            source_reach,
            authority,
            reach,
        }
    }

    /// How far the source collection may travel.
    pub fn source_reach(&self) -> &Fragment {
        &self.source_reach
    }

    /// How far this projection may travel.
    pub fn reach(&self) -> &Fragment {
        &self.reach
    }

    /// Name of the root collection this projection is taken over.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Mandatory capability trust root declared by the source descriptor.
    pub fn source_authority(&self) -> VerifyingKey {
        self.source_authority
    }

    /// Mandatory capability trust root declared by this derived collection.
    pub fn authority(&self) -> VerifyingKey {
        self.authority
    }

    /// The observation attribute this collection reads.
    pub fn observes(&self) -> Id {
        self.observes
    }

    /// Canonical source `SimpleArchive` collection descriptor facts.
    pub fn source_descriptor(&self) -> Fragment {
        simplearchive_union::descriptor(
            &self.name,
            self.source_authority,
            self.source_reach.clone(),
        )
    }

    /// Identity of the source collection this projection reads.
    pub fn source_collection(&self) -> CollectionHandle {
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(self.source_descriptor().into_facts())
            .get_handle()
    }

    /// Canonical target observed-set collection descriptor.
    pub fn descriptor(&self) -> Fragment {
        descriptor(
            self.source_collection(),
            self.observes,
            self.authority,
            self.reach.clone(),
        )
    }

    /// Attach the observed set already resident for `source_cover`.
    pub fn attach_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<ObservedIndex, ObservedSetCollectionError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel()?.attach_exact(store, source_cover)?;
        ObservedIndex::try_from_cover(cover).map_err(ObservedSetCollectionError::Algebra)
    }

    /// Ensure and attach the observed set for `source_cover`.
    pub fn ensure_exact<S>(
        &self,
        store: &mut S,
        source_cover: &FactCover,
    ) -> Result<ObservedIndex, ObservedSetCollectionError>
    where
        S: BlobStore + CollectionStore + ArtifactOfferStore,
        S::Reader: BlobStoreMeta,
    {
        let cover = self.kernel()?.ensure_exact(store, source_cover)?;
        ObservedIndex::try_from_cover(cover).map_err(ObservedSetCollectionError::Algebra)
    }

    fn kernel(
        &self,
    ) -> Result<
        ExactDerivedCollection<SimpleArchiveUnion, ObservedSetUnion, ObserveStates>,
        ExactDerivedCollectionError,
    > {
        ExactDerivedCollection::new(self.source_descriptor(), self.descriptor())
    }
}

/// Failure to validate, complete, or materialize one observed-set cover.
#[derive(Debug)]
pub enum ObservedSetCollectionError {
    /// Exact-cover resolution, construction, or storage failed.
    Collection(ExactDerivedCollectionError),
    /// Canonical observed-set construction failed.
    Algebra(ObservedSetError),
}

impl fmt::Display for ObservedSetCollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collection(source) => source.fmt(formatter),
            Self::Algebra(source) => source.fmt(formatter),
        }
    }
}

impl Error for ObservedSetCollectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Collection(source) => Some(source),
            Self::Algebra(source) => Some(source),
        }
    }
}

impl From<ExactDerivedCollectionError> for ObservedSetCollectionError {
    fn from(source: ExactDerivedCollectionError) -> Self {
        Self::Collection(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::query::register::{resolve, ObservationOrder};
    use crate::trible::TribleSet;
    use std::collections::BTreeSet;

    fn archive(facts: &TribleSet) -> Blob<SimpleArchive> {
        facts.clone().to_blob()
    }

    fn edge(successor: &crate::id::ExclusiveId, predecessor: &crate::id::ExclusiveId) -> TribleSet {
        entity! { successor @ metadata::supersedes: predecessor }.into()
    }

    fn observed_of(facts: &TribleSet) -> Blob<ObservedSetBlob> {
        derive_element(&archive(facts), metadata::supersedes.id()).expect("derives")
    }

    #[test]
    fn the_derived_index_agrees_with_the_live_order() {
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();
        let mut facts = TribleSet::new();
        facts += edge(&left, &base);
        facts += edge(&right, &base);
        let candidates = [*base, *left, *right];

        let index = ObservedIndex::decode(&observed_of(&facts)).expect("decodes");
        assert_eq!(index.len(), 1);
        assert_eq!(
            resolve(&index, candidates),
            resolve(
                &ObservationOrder::new(&facts, metadata::supersedes.id()),
                candidates
            )
        );
        assert_eq!(
            resolve(&index, candidates),
            [*left, *right].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn derive_is_a_join_homomorphism_into_the_union_lattice() {
        let base = ufoid();
        let left = ufoid();
        let right = ufoid();
        let merge = ufoid();

        let mut c1 = TribleSet::new();
        c1 += edge(&left, &base);
        let mut c2 = TribleSet::new();
        c2 += edge(&right, &base);
        c2 += edge(&merge, &right);
        let mut union = c1.clone();
        union += c2.clone();

        // derive(C1 union C2) == join(derive(C1), derive(C2)), byte-exact.
        // This is the equation the exact-derive kernel checks when it
        // reuses a cached shard, so byte equality is the operative form.
        let direct = observed_of(&union);
        let incremental = join(&observed_of(&c1), &observed_of(&c2)).expect("joins");
        assert_eq!(direct.bytes.as_ref(), incremental.bytes.as_ref());

        // ... and the frontier read off it matches the live order.
        let candidates = [*base, *left, *right, *merge];
        let index = ObservedIndex::decode(&incremental).expect("decodes");
        assert_eq!(index.len(), 2);
        assert_eq!(
            resolve(&index, candidates),
            resolve(
                &ObservationOrder::new(&union, metadata::supersedes.id()),
                candidates
            )
        );
        assert_eq!(
            resolve(&index, candidates),
            [*left, *merge].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn the_join_is_idempotent_commutative_and_has_empty_as_its_unit() {
        let a = ufoid();
        let b = ufoid();
        let c = ufoid();
        let mut left = TribleSet::new();
        left += edge(&b, &a);
        let mut right = TribleSet::new();
        right += edge(&c, &b);

        let l = observed_of(&left);
        let r = observed_of(&right);

        let lr = join(&l, &r).expect("joins");
        let rl = join(&r, &l).expect("joins");
        assert_eq!(lr.bytes.as_ref(), rl.bytes.as_ref(), "commutative");
        assert_eq!(
            join(&lr, &lr).expect("joins").bytes.as_ref(),
            lr.bytes.as_ref(),
            "idempotent"
        );
        assert_eq!(
            join(&lr, &empty()).expect("joins").bytes.as_ref(),
            lr.bytes.as_ref(),
            "empty is the unit"
        );
        validate_element(&lr).expect("the join is canonical");
    }

    #[test]
    fn the_observed_attribute_participates_in_collection_identity() {
        let authority = ed25519_dalek::SigningKey::from_bytes(&[1; 32]).verifying_key();
        let root = |name: &str| {
            crate::blob::IntoBlob::<SimpleArchive>::to_blob(
                simplearchive_union::descriptor(name, authority, reach::private()).into_facts(),
            )
            .get_handle()
        };
        let source = root("source");
        assert_ne!(
            descriptor(
                source,
                metadata::supersedes.id(),
                authority,
                reach::private()
            ),
            descriptor(source, metadata::tag.id(), authority, reach::private()),
            "two registers over different edges are different collections"
        );
        // A derived collection carries no anchor of its own; two derivations
        // of the same shape differ exactly when their sources differ.
        let other = root("other-source");
        assert_ne!(
            descriptor(source, metadata::tag.id(), authority, reach::private()),
            descriptor(other, metadata::tag.id(), authority, reach::private()),
            "the same derivation over different sources is a different collection"
        );
        // ... and the derivation genuinely reads the attribute it is told to.
        let a = ufoid();
        let b = ufoid();
        let mut facts = TribleSet::new();
        facts += edge(&b, &a);
        let other = derive_element(&archive(&facts), metadata::tag.id()).expect("derives");
        assert!(other.bytes.as_ref().is_empty());
    }

    #[test]
    fn source_and_derived_descriptors_carry_independent_mandatory_authorities() {
        use crate::collection::descriptor as descriptor_facts;

        let source_authority = ed25519_dalek::SigningKey::from_bytes(&[9; 32]).verifying_key();
        let target_authority = ed25519_dalek::SigningKey::from_bytes(&[10; 32]).verifying_key();
        let name = "observed-source".to_owned();
        let collection = ObservedSetCollection::new(
            name.clone(),
            source_authority,
            metadata::supersedes.id(),
            reach::private(),
            target_authority,
            reach::private(),
        );

        assert_eq!(
            collection.source_descriptor(),
            simplearchive_union::descriptor(&name, source_authority, reach::private())
        );
        assert_eq!(
            descriptor_facts::authority(collection.source_descriptor().facts()),
            Ok(source_authority)
        );
        assert_eq!(
            descriptor_facts::authority(collection.descriptor().facts()),
            Ok(target_authority)
        );
    }

    #[test]
    fn a_non_canonical_element_is_rejected() {
        let mut bytes = vec![0u8; ID_LEN * 2];
        bytes[ID_LEN - 1] = 9;
        // Second id sorts below the first, so the sequence is not
        // strictly increasing.
        let blob: Blob<ObservedSetBlob> = Blob::new(Bytes::from_source(bytes));
        assert_eq!(
            validate_element(&blob),
            Err(ObservedSetError::NotStrictlyIncreasing)
        );
        let ragged: Blob<ObservedSetBlob> = Blob::new(Bytes::from_source(vec![0u8; ID_LEN + 1]));
        assert_eq!(
            validate_element(&ragged),
            Err(ObservedSetError::BadLength(ID_LEN + 1))
        );
    }
}
