//! Private one-to-one Rank9 fibers over an already admitted raw exact cover.
//!
//! For one fixed ABI recipe, the target lattice is the image `i(a)` of the raw
//! SuccinctArchive lattice. Each target blob embeds the exact raw handle and
//! its join is defined by `i(a) join i(b) = i(a join b)`, so an ordinary
//! `DERIVE` is truthful. This helper intentionally consumes the raw
//! [`ExactCover`] selected by the authoritative lifecycle instead of running a
//! second [`ExactDerivedCollection`](crate::collection::exact_derived::ExactDerivedCollection):
//! raw collection members are derived cache artifacts and have no signed
//! commits of their own.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchive, SuccinctArchiveBlob, SuccinctArchiveError,
    SuccinctArchiveRank9IndexBlob, UnionArchive,
};
use crate::blob::{Blob, BlobEncoding};
use crate::collection::exact_derived::ExactCover;
use crate::collection::{
    CollectionData, CollectionDerive, CollectionDescriptor, CollectionRecord,
    CollectionRecordSelector, CollectionStore,
};
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::{Inline, InlineEncoding};
use crate::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while attaching or publishing exact Rank9 fibers.
#[derive(Debug)]
pub enum Rank9FiberError {
    /// A repository operation failed before optional evidence could be judged.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Concrete backend failure.
        source: BoxError,
    },
    /// The admitted raw member did not have the content identity carried by its cover.
    InvalidRawCover {
        /// Identity selected by exact raw admission.
        expected: CollectionData,
        /// Fresh identity of the supplied raw bytes.
        actual: CollectionData,
    },
    /// A transient or persisted Rank9 runtime could not be constructed.
    Build {
        /// Raw member whose Rank9 runtime failed.
        raw: CollectionData,
        /// Exact raw/Rank9 validation failure.
        source: SuccinctArchiveError,
    },
    /// A descriptor put acknowledged a different content identity.
    NonCanonicalDescriptorPut {
        /// Which descriptor was being published.
        role: &'static str,
        /// Canonical descriptor identity.
        expected: crate::collection::CollectionId,
        /// Identity returned by the backend.
        actual: crate::collection::CollectionId,
    },
    /// A Rank9 put acknowledged a different content identity.
    NonCanonicalRank9Put {
        /// Raw source member.
        raw: CollectionData,
        /// Canonical Rank9 content identity.
        expected: CollectionData,
        /// Identity returned by the backend.
        actual: CollectionData,
    },
    /// Fresh post-publication admission could not prove an expected exact pair.
    IncompletePublication {
        /// Raw source member.
        raw: CollectionData,
        /// Expected Rank9 member, when construction reached that point.
        rank9: Option<CollectionData>,
        /// Concrete missing, corrupt, or mismatched dependency.
        reason: String,
    },
}

impl Rank9FiberError {
    fn storage(operation: &'static str, source: impl Error + Send + Sync + 'static) -> Self {
        Self::Storage {
            operation,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for Rank9FiberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage { operation, source } => write!(f, "{operation}: {source}"),
            Self::InvalidRawCover { expected, actual } => write!(
                f,
                "exact raw cover member {} hashes to {}",
                hex::encode_upper(expected.raw),
                hex::encode_upper(actual.raw),
            ),
            Self::Build { raw, source } => write!(
                f,
                "build Rank9 fiber for raw member {}: {source}",
                hex::encode_upper(raw.raw),
            ),
            Self::NonCanonicalDescriptorPut {
                role,
                expected,
                actual,
            } => write!(
                f,
                "blob store returned {role} descriptor {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::NonCanonicalRank9Put {
                raw,
                expected,
                actual,
            } => write!(
                f,
                "blob store returned Rank9 handle {} instead of {} for raw member {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
                hex::encode_upper(raw.raw),
            ),
            Self::IncompletePublication { raw, rank9, reason } => write!(
                f,
                "fresh Rank9 admission for raw member {} and sidecar {} failed: {reason}",
                hex::encode_upper(raw.raw),
                rank9
                    .map(|data| hex::encode_upper(data.raw))
                    .unwrap_or_else(|| "<unbuilt>".to_owned()),
            ),
        }
    }
}

impl Error for Rank9FiberError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Storage { source, .. } => Some(source.as_ref()),
            Self::Build { source, .. } => Some(source),
            _ => None,
        }
    }
}

struct ProbedMember {
    raw_data: CollectionData,
    raw: Blob<SuccinctArchiveBlob>,
    output: Option<CollectionData>,
    claim_present: bool,
    prepared: Option<Blob<SuccinctArchiveRank9IndexBlob>>,
    runtime: Option<SuccinctArchive<OrderedUniverse>>,
}

struct CanonicalCandidate {
    output: CollectionData,
    claim_present: bool,
    prepared: Blob<SuccinctArchiveRank9IndexBlob>,
}

struct CandidateOutputs {
    first: CollectionData,
    ambiguous: bool,
    canonical: Option<CanonicalCandidate>,
}

struct FiberProbe {
    members: Vec<ProbedMember>,
}

impl FiberProbe {
    fn is_complete(&self) -> bool {
        self.members.iter().all(|member| member.runtime.is_some())
    }

    fn into_archive(self) -> UnionArchive<OrderedUniverse> {
        UnionArchive::new(
            self.members
                .into_iter()
                .map(|member| member.runtime.expect("complete probe has every runtime"))
                .collect::<Vec<_>>(),
        )
    }
}

#[derive(Clone, Copy)]
pub(super) struct Rank9Fiber {
    source: CollectionDescriptor,
    target: CollectionDescriptor,
}

impl Rank9Fiber {
    pub(super) fn new(source: CollectionDescriptor, target: CollectionDescriptor) -> Self {
        Self { source, target }
    }

    /// Attach persisted fibers when unambiguous and exact; otherwise rebuild
    /// that raw member's Rank9 runtime transiently without writing.
    pub(super) fn attach<S>(
        &self,
        store: &mut S,
        cover: ExactCover<SuccinctArchiveBlob>,
    ) -> Result<UnionArchive<OrderedUniverse>, Rank9FiberError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let probe = self.probe(store, cover, false)?;
        let mut segments = Vec::with_capacity(probe.members.len());
        for member in probe.members {
            match member.runtime {
                Some(runtime) => segments.push(runtime),
                None => segments.push(member.raw.try_from_blob().map_err(|source| {
                    Rank9FiberError::Build {
                        raw: member.raw_data,
                        source,
                    }
                })?),
            }
        }
        Ok(UnionArchive::new(segments))
    }

    /// Ensure one persisted exact Rank9 fiber for every member of this fixed
    /// admitted raw cover, then strictly re-read those same expected pairs.
    pub(super) fn ensure<S>(
        &self,
        store: &mut S,
        cover: ExactCover<SuccinctArchiveBlob>,
    ) -> Result<UnionArchive<OrderedUniverse>, Rank9FiberError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let probe = self.probe(store, cover, true)?;
        if probe.is_complete() {
            return Ok(probe.into_archive());
        }

        let mut members = probe.members;
        for member in &mut members {
            member.runtime = None;
        }

        self.publish_descriptor(store, "raw", self.source)?;
        self.publish_descriptor(store, "Rank9", self.target)?;

        let mut claims = Vec::new();
        for member in &mut members {
            let Some(rank9) = member.prepared.take() else {
                continue;
            };
            let output = fresh_data_identity(&rank9);
            let actual = store
                .put::<SuccinctArchiveRank9IndexBlob, _>(rank9)
                .map_err(|error| Rank9FiberError::storage("store Rank9 sidecar", error))?;
            let actual = Handle::<SuccinctArchiveRank9IndexBlob>::to_hash(actual);
            if actual != output {
                return Err(Rank9FiberError::NonCanonicalRank9Put {
                    raw: member.raw_data,
                    expected: output,
                    actual,
                });
            }
            if let Some(expected) = member.output {
                debug_assert_eq!(output, expected, "fixed Rank9 recipe is functional");
            }
            member.output = Some(output);
            if !member.claim_present {
                claims.push(CollectionDerive::new(
                    self.source.handle(),
                    self.target.handle(),
                    member.raw_data,
                    output,
                ));
                member.claim_present = true;
            }
        }

        // Every newly claimed endpoint precedes the first new equation. A
        // failure above can leave only harmless orphan blobs, never a new
        // claim naming bytes this attempt failed to store.
        for claim in claims {
            store
                .insert(CollectionRecord::Derive(claim))
                .map_err(|error| Rank9FiberError::storage("publish Rank9 DERIVE", error))?;
        }

        self.verify_published(store, members)
    }

    fn probe<S>(
        &self,
        store: &mut S,
        cover: ExactCover<SuccinctArchiveBlob>,
        ensure: bool,
    ) -> Result<FiberProbe, Rank9FiberError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let members = cover.into_members();
        let raw_by_input: BTreeMap<_, _> = members.iter().map(|(data, raw)| (*data, raw)).collect();
        let mut candidates = self.candidates(store, &raw_by_input, ensure)?;
        let reader = store
            .reader()
            .map_err(|error| Rank9FiberError::storage("open Rank9 fiber reader", error))?;

        let mut probed = Vec::with_capacity(members.len());
        for (raw_data, raw) in members {
            let actual = fresh_data_identity(&raw);
            if actual != raw_data {
                return Err(Rank9FiberError::InvalidRawCover {
                    expected: raw_data,
                    actual,
                });
            }
            let mut evidence = candidates.remove(&raw_data);
            let unique = evidence
                .as_ref()
                .and_then(|outputs| (!outputs.ambiguous).then_some(outputs.first));
            let mut full_validation_attempted = false;
            let mut output = None;
            let mut claim_present = false;
            let mut prepared = None;
            let mut runtime = None;

            if let Some(candidate) = unique {
                full_validation_attempted = true;
                let claim = CollectionDerive::new(
                    self.source.handle(),
                    self.target.handle(),
                    raw_data,
                    candidate,
                );
                if let Some(attached) = self.try_attach(&reader, raw_data, &raw, claim)? {
                    output = Some(candidate);
                    claim_present = true;
                    runtime = Some(attached);
                }
            }

            if ensure && runtime.is_none() {
                // Ambiguous evidence is never searched. Build once to learn the
                // one canonical output for this fixed recipe, then validate at
                // most that exact claimed endpoint. Retaining these bytes lets
                // publication proceed after this reader is dropped without a
                // second canonical build.
                let canonical_candidate = evidence
                    .as_mut()
                    .and_then(|outputs| outputs.canonical.take());
                let (canonical_blob, canonical, canonical_was_claimed) = if let Some(candidate) =
                    canonical_candidate
                {
                    (
                        candidate.prepared,
                        candidate.output,
                        candidate.claim_present,
                    )
                } else {
                    let blob = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone())
                        .map_err(|source| Rank9FiberError::Build {
                        raw: raw_data,
                        source,
                    })?;
                    let output = fresh_data_identity(&blob);
                    let claimed = evidence.as_ref().is_some_and(|outputs| {
                        debug_assert!(!outputs.ambiguous);
                        outputs.first == output
                    });
                    (blob, output, claimed)
                };
                claim_present = canonical_was_claimed;
                output = Some(canonical);
                if claim_present && !full_validation_attempted {
                    let claim = CollectionDerive::new(
                        self.source.handle(),
                        self.target.handle(),
                        raw_data,
                        canonical,
                    );
                    runtime = self.try_attach(&reader, raw_data, &raw, claim)?;
                }
                if runtime.is_none() {
                    prepared = Some(canonical_blob);
                }
            }
            probed.push(ProbedMember {
                raw_data,
                raw,
                output,
                claim_present,
                prepared,
                runtime,
            });
        }
        Ok(FiberProbe { members: probed })
    }

    /// Enumerate records once and retain bounded evidence per selected input.
    /// One first output plus an ambiguity bit suffices for attachment. Ensure
    /// builds and retains the canonical sidecar as soon as a second distinct
    /// output appears, so later records need only note whether that exact
    /// output was claimed. This avoids both O(cover * records) work and memory
    /// proportional to attacker-supplied distinct outputs.
    fn candidates<S: CollectionStore>(
        &self,
        store: &mut S,
        raw_by_input: &BTreeMap<CollectionData, &Blob<SuccinctArchiveBlob>>,
        ensure: bool,
    ) -> Result<BTreeMap<CollectionData, CandidateOutputs>, Rank9FiberError> {
        let mut candidates = BTreeMap::<CollectionData, CandidateOutputs>::new();
        let selectors = [CollectionRecordSelector::DerivePair {
            source: self.source.handle(),
            target: self.target.handle(),
        }]
        .into_iter()
        .collect();
        let records = store
            .select_records(&selectors)
            .map_err(|error| Rank9FiberError::storage("select Rank9 DERIVEs", error))?;
        for record in records {
            let CollectionRecord::Derive(claim) = record else {
                continue;
            };
            let (input, output) = claim.mapping();
            let Some(raw) = raw_by_input.get(&input) else {
                continue;
            };
            if claim.source() != self.source.handle() || claim.target() != self.target.handle() {
                continue;
            }
            let Some(outputs) = candidates.get_mut(&input) else {
                candidates.insert(
                    input,
                    CandidateOutputs {
                        first: output,
                        ambiguous: false,
                        canonical: None,
                    },
                );
                continue;
            };
            if output == outputs.first {
                continue;
            }
            if !outputs.ambiguous {
                outputs.ambiguous = true;
                if ensure {
                    let prepared =
                        SuccinctArchive::<OrderedUniverse>::build_rank9_index((*raw).clone())
                            .map_err(|source| Rank9FiberError::Build { raw: input, source })?;
                    let canonical = fresh_data_identity(&prepared);
                    outputs.canonical = Some(CanonicalCandidate {
                        output: canonical,
                        claim_present: outputs.first == canonical || output == canonical,
                        prepared,
                    });
                }
                continue;
            }
            if let Some(canonical) = &mut outputs.canonical {
                canonical.claim_present |= output == canonical.output;
            }
        }
        Ok(candidates)
    }

    fn try_attach<R>(
        &self,
        reader: &R,
        raw_data: CollectionData,
        raw: &Blob<SuccinctArchiveBlob>,
        claim: CollectionDerive,
    ) -> Result<Option<SuccinctArchive<OrderedUniverse>>, Rank9FiberError>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        if claim.source() != self.source.handle() || claim.target() != self.target.handle() {
            return Ok(None);
        }
        let (input, output) = claim.mapping();
        // `probe` freshly hashed `raw` against `raw_data` immediately before
        // this call; bind the claim to that proof without hashing the same raw
        // bytes a second time.
        if input != raw_data {
            return Ok(None);
        }

        let handle = Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(output);
        let Some(_) = reader
            .metadata(handle)
            .map_err(|error| Rank9FiberError::storage("inspect Rank9 sidecar", error))?
        else {
            return Ok(None);
        };
        let Ok(rank9): Result<Blob<SuccinctArchiveRank9IndexBlob>, _> = reader.get(handle) else {
            return Ok(None);
        };
        if fresh_data_identity(&rank9) != output {
            return Ok(None);
        }
        let Ok(source) = SuccinctArchiveRank9IndexBlob::source_handle(&rank9) else {
            return Ok(None);
        };
        if Handle::<SuccinctArchiveBlob>::to_hash(source) != raw_data {
            return Ok(None);
        }
        Ok(SuccinctArchive::from_blob_pair(raw.clone(), rank9).ok())
    }

    fn publish_descriptor<S: BlobStore>(
        &self,
        store: &mut S,
        role: &'static str,
        descriptor: CollectionDescriptor,
    ) -> Result<(), Rank9FiberError> {
        let expected = descriptor.handle();
        let actual = store
            .put::<SimpleArchive, _>(descriptor.to_blob())
            .map_err(|error| Rank9FiberError::storage("store Rank9 descriptor", error))?;
        if actual != expected {
            return Err(Rank9FiberError::NonCanonicalDescriptorPut {
                role,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn verify_published<S>(
        &self,
        store: &mut S,
        members: Vec<ProbedMember>,
    ) -> Result<UnionArchive<OrderedUniverse>, Rank9FiberError>
    where
        S: BlobStore + CollectionStore,
        S::Reader: BlobStoreMeta,
    {
        let expected_claims: BTreeMap<_, _> = members
            .iter()
            .map(|member| {
                let output = member
                    .output
                    .expect("every incomplete probe member is assigned before publication");
                let claim = CollectionDerive::new(
                    self.source.handle(),
                    self.target.handle(),
                    member.raw_data,
                    output,
                );
                (claim.id(), claim)
            })
            .collect();
        let selectors = expected_claims
            .keys()
            .copied()
            .map(CollectionRecordSelector::Id)
            .collect();
        let mut found = BTreeSet::new();
        let records = store.select_records(&selectors).map_err(|error| {
            Rank9FiberError::storage("re-select published Rank9 DERIVEs", error)
        })?;
        for record in records {
            if let CollectionRecord::Derive(claim) = record {
                if expected_claims.get(&claim.id()) == Some(&claim) {
                    found.insert(claim.id());
                }
            }
        }
        for (id, claim) in &expected_claims {
            if !found.contains(id) {
                let (raw, rank9) = claim.mapping();
                return Err(Rank9FiberError::IncompletePublication {
                    raw,
                    rank9: Some(rank9),
                    reason: format!("expected DERIVE {id:X} is not resident"),
                });
            }
        }

        let reader = store.reader().map_err(|error| {
            Rank9FiberError::storage("open fresh Rank9 verification reader", error)
        })?;
        let raw_context = members
            .first()
            .expect("nonempty fixed cover reaches publication")
            .raw_data;
        self.verify_descriptor(&reader, "raw", self.source, raw_context)?;
        self.verify_descriptor(&reader, "Rank9", self.target, raw_context)?;

        let mut segments = Vec::with_capacity(members.len());
        for member in members {
            let output = member.output.expect("published member has an output");
            let raw_handle = Handle::<SuccinctArchiveBlob>::from_hash(member.raw_data);
            let raw: Blob<SuccinctArchiveBlob> =
                reader
                    .get(raw_handle)
                    .map_err(|error| Rank9FiberError::IncompletePublication {
                        raw: member.raw_data,
                        rank9: Some(output),
                        reason: format!("expected raw endpoint is not readable: {error}"),
                    })?;
            if fresh_data_identity(&raw) != member.raw_data {
                return Err(Rank9FiberError::IncompletePublication {
                    raw: member.raw_data,
                    rank9: Some(output),
                    reason: "fresh raw endpoint hash does not match its DERIVE".to_owned(),
                });
            }

            let rank9_handle = Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(output);
            let rank9: Blob<SuccinctArchiveRank9IndexBlob> =
                reader.get(rank9_handle).map_err(|error| {
                    Rank9FiberError::IncompletePublication {
                        raw: member.raw_data,
                        rank9: Some(output),
                        reason: format!("expected Rank9 endpoint is not readable: {error}"),
                    }
                })?;
            if fresh_data_identity(&rank9) != output {
                return Err(Rank9FiberError::IncompletePublication {
                    raw: member.raw_data,
                    rank9: Some(output),
                    reason: "fresh Rank9 endpoint hash does not match its DERIVE".to_owned(),
                });
            }
            let source = SuccinctArchiveRank9IndexBlob::source_handle(&rank9).map_err(|error| {
                Rank9FiberError::IncompletePublication {
                    raw: member.raw_data,
                    rank9: Some(output),
                    reason: format!("Rank9 source header is invalid: {error}"),
                }
            })?;
            if Handle::<SuccinctArchiveBlob>::to_hash(source) != member.raw_data {
                return Err(Rank9FiberError::IncompletePublication {
                    raw: member.raw_data,
                    rank9: Some(output),
                    reason: "Rank9 source header names another raw member".to_owned(),
                });
            }
            segments.push(
                SuccinctArchive::from_blob_pair(raw, rank9).map_err(|error| {
                    Rank9FiberError::IncompletePublication {
                        raw: member.raw_data,
                        rank9: Some(output),
                        reason: format!("exact raw/Rank9 validation failed: {error}"),
                    }
                })?,
            );
        }
        Ok(UnionArchive::new(segments))
    }

    fn verify_descriptor<R: BlobStoreGet>(
        &self,
        reader: &R,
        role: &'static str,
        expected: CollectionDescriptor,
        raw: CollectionData,
    ) -> Result<(), Rank9FiberError> {
        let blob: Blob<SimpleArchive> = reader.get(expected.handle()).map_err(|error| {
            Rank9FiberError::IncompletePublication {
                raw,
                rank9: None,
                reason: format!("expected {role} descriptor is not readable: {error}"),
            }
        })?;
        if Blob::<SimpleArchive>::new(blob.bytes.clone()).get_handle() != expected.handle()
            || CollectionDescriptor::decode(&blob).ok() != Some(expected)
        {
            return Err(Rank9FiberError::IncompletePublication {
                raw,
                rank9: None,
                reason: format!("fresh {role} descriptor is not canonical"),
            });
        }
        Ok(())
    }
}

fn fresh_data_identity<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
where
    Handle<E>: InlineEncoding,
{
    Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
}
