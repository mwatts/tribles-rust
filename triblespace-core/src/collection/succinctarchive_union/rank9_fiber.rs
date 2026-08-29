//! Private one-to-one Rank9 accelerators over an already selected raw cover.
//!
//! Rank9 is not a collection. It is a deterministic implementation detail of
//! the raw [`SuccinctArchiveBlob`] view: one ABI-qualified mapping turns one raw
//! member into one detached sidecar. [`MappingEvidence`] records cacheable
//! observations of that mapping without granting the sidecars collection
//! membership or inventing a second frontier.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchive, SuccinctArchiveBlob, SuccinctArchiveError,
    SuccinctArchiveRank9IndexBlob, UnionArchive,
};
use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, BlobEncoding, IntoBlob};
use crate::collection::{
    CollectionData, CollectionHandle, CoverAttachment, MappingEvidence, MappingEvidenceSelector,
    MappingEvidenceStore, MappingHandle,
};
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::{Inline, InlineEncoding};
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStorePut, OfferCapture};
use crate::trible::{Fragment, TribleSet};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

/// Failure while attaching or publishing exact Rank9 accelerators.
#[derive(Debug)]
pub enum Rank9FiberError {
    /// A repository operation failed before optional evidence could be judged.
    Storage {
        /// Operation that failed.
        operation: &'static str,
        /// Concrete backend failure.
        source: BoxError,
    },
    /// The supplied raw cover belongs to another collection descriptor.
    WrongRawCollection {
        /// Raw collection this accelerator indexes.
        expected: CollectionHandle,
        /// Collection carried by the supplied cover.
        actual: CollectionHandle,
    },
    /// The selected raw member did not have the content identity carried by its cover.
    InvalidRawCover {
        /// Identity selected by the exact raw cover.
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
    /// A mapping-fragment put acknowledged a different content identity.
    NonCanonicalMappingPut {
        /// Canonical mapping-fragment identity.
        expected: MappingHandle,
        /// Identity returned by the backend.
        actual: MappingHandle,
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
    /// Fresh post-publication verification could not prove an expected exact pair.
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
            Self::WrongRawCollection { expected, actual } => write!(
                f,
                "raw cover belongs to collection {} instead of {}",
                hex::encode_upper(actual.raw),
                hex::encode_upper(expected.raw),
            ),
            Self::InvalidRawCover { expected, actual } => write!(
                f,
                "exact raw cover member {} hashes to {}",
                hex::encode_upper(expected.raw),
                hex::encode_upper(actual.raw),
            ),
            Self::Build { raw, source } => write!(
                f,
                "build Rank9 accelerator for raw member {}: {source}",
                hex::encode_upper(raw.raw),
            ),
            Self::NonCanonicalMappingPut { expected, actual } => write!(
                f,
                "blob store returned Rank9 mapping {} instead of {}",
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
                "fresh Rank9 verification for raw member {} and sidecar {} failed: {reason}",
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
    prepared: Option<Blob<SuccinctArchiveRank9IndexBlob>>,
    runtime: Option<SuccinctArchive<OrderedUniverse>>,
}

struct CandidateOutputs {
    first: CollectionData,
    ambiguous: bool,
}

struct FiberProbe {
    members: Vec<ProbedMember>,
    mapping_complete: bool,
}

impl FiberProbe {
    fn is_complete(&self) -> bool {
        self.mapping_complete && self.members.iter().all(|member| member.runtime.is_some())
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

#[derive(Clone)]
pub(super) struct Rank9Fiber {
    source_collection: CollectionHandle,
    mapping: Fragment,
    mapping_handle: MappingHandle,
}

impl Rank9Fiber {
    pub(super) fn new(source: Fragment) -> Self {
        let source_collection = source.facts().clone().to_blob().get_handle();
        let mapping = super::rank9_mapping_fragment();
        let mapping_handle = mapping.facts().clone().to_blob().get_handle();
        Self {
            source_collection,
            mapping,
            mapping_handle,
        }
    }

    /// Attach unambiguous, valid persisted accelerators. Every cache miss is
    /// rebuilt transiently and this method writes nothing.
    pub(super) fn attach<S>(
        &self,
        store: &mut S,
        cover: CoverAttachment<SuccinctArchiveBlob>,
    ) -> Result<UnionArchive<OrderedUniverse>, Rank9FiberError>
    where
        S: BlobStore + MappingEvidenceStore,
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

    /// Ensure one persisted exact Rank9 accelerator for every member of this
    /// fixed raw cover, then strictly re-read those same expected pairs.
    pub(super) fn ensure<S>(
        &self,
        store: &mut S,
        cover: CoverAttachment<SuccinctArchiveBlob>,
    ) -> Result<UnionArchive<OrderedUniverse>, Rank9FiberError>
    where
        S: BlobStore + MappingEvidenceStore + ArtifactOfferStore,
    {
        let probe = self.probe(store, cover, true)?;
        if probe.is_complete() {
            return Ok(probe.into_archive());
        }

        let mut members = probe.members;
        // Verification below deliberately reopens every endpoint, including
        // members whose accelerator was already valid before this repair.
        for member in &mut members {
            member.runtime = None;
        }

        let mut capture = OfferCapture::new(store);
        let store = &mut capture;
        self.publish_mapping_closure(store)?;

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
            member.output = Some(output);
        }

        // OfferCapture advertises the complete mapping closure and every new
        // sidecar before admitting the first equation. Re-inserting equations
        // for already-valid pairs is idempotent and also flushes a repaired
        // mapping closure when no sidecar itself needed repair.
        for member in &members {
            let output = member
                .output
                .expect("every ensured Rank9 member has one exact output");
            store
                .insert_evidence(MappingEvidence::new(
                    self.mapping_handle,
                    member.raw_data,
                    output,
                ))
                .map_err(|error| {
                    Rank9FiberError::storage("publish Rank9 mapping evidence", error)
                })?;
        }

        self.verify_published(store, members)
    }

    fn probe<S>(
        &self,
        store: &mut S,
        cover: CoverAttachment<SuccinctArchiveBlob>,
        ensure: bool,
    ) -> Result<FiberProbe, Rank9FiberError>
    where
        S: BlobStore + MappingEvidenceStore,
    {
        if cover.cover().collection().handle() != self.source_collection {
            return Err(Rank9FiberError::WrongRawCollection {
                expected: self.source_collection,
                actual: cover.cover().collection().handle(),
            });
        }
        let members: Vec<_> = cover
            .into_members()
            .into_iter()
            .map(|(data, raw)| (Handle::<SuccinctArchiveBlob>::to_hash(data), raw))
            .collect();
        let selectors = members
            .iter()
            .map(|(input, _)| MappingEvidenceSelector::MappingInput(self.mapping_handle, *input))
            .collect();
        let mut candidates = self.candidates(store, &selectors)?;
        let reader = store
            .reader()
            .map_err(|error| Rank9FiberError::storage("open Rank9 accelerator reader", error))?;
        let mapping_complete = self.mapping_closure_is_complete(&reader);

        let mut probed = Vec::with_capacity(members.len());
        for (raw_data, raw) in members {
            let actual = fresh_data_identity(&raw);
            if actual != raw_data {
                return Err(Rank9FiberError::InvalidRawCover {
                    expected: raw_data,
                    actual,
                });
            }
            let unique = candidates
                .remove(&raw_data)
                .and_then(|outputs| (!outputs.ambiguous).then_some(outputs.first));
            let mut output = None;
            let mut prepared = None;
            let mut runtime = None;

            if mapping_complete {
                if let Some(candidate) = unique {
                    if let Some(attached) = self.try_attach(&reader, raw_data, &raw, candidate) {
                        output = Some(candidate);
                        runtime = Some(attached);
                    }
                }
            }

            if ensure && runtime.is_none() {
                // Missing, corrupt, source-mismatched, or ambiguous evidence
                // is a cache miss. Rebuild the one canonical sidecar instead
                // of choosing among claims supplied by the store.
                let rank9 = SuccinctArchive::<OrderedUniverse>::build_rank9_index(raw.clone())
                    .map_err(|source| Rank9FiberError::Build {
                        raw: raw_data,
                        source,
                    })?;
                output = Some(fresh_data_identity(&rank9));
                prepared = Some(rank9);
            }
            probed.push(ProbedMember {
                raw_data,
                raw,
                output,
                prepared,
                runtime,
            });
        }
        Ok(FiberProbe {
            members: probed,
            mapping_complete,
        })
    }

    /// Retain only one first output plus an ambiguity bit per selected input.
    /// This bounds memory even if a hostile store contains many equations for
    /// the same input.
    fn candidates<S: MappingEvidenceStore>(
        &self,
        store: &mut S,
        selectors: &BTreeSet<MappingEvidenceSelector>,
    ) -> Result<BTreeMap<CollectionData, CandidateOutputs>, Rank9FiberError> {
        let evidence = store
            .select_evidence(selectors)
            .map_err(|error| Rank9FiberError::storage("select Rank9 mapping evidence", error))?;
        let mut candidates = BTreeMap::<CollectionData, CandidateOutputs>::new();
        for evidence in evidence {
            if evidence.mapping() != self.mapping_handle {
                continue;
            }
            let input = evidence.input();
            let output = evidence.output();
            match candidates.get_mut(&input) {
                None => {
                    candidates.insert(
                        input,
                        CandidateOutputs {
                            first: output,
                            ambiguous: false,
                        },
                    );
                }
                Some(outputs) if outputs.first != output => outputs.ambiguous = true,
                Some(_) => {}
            }
        }
        Ok(candidates)
    }

    fn try_attach<R: BlobStoreGet>(
        &self,
        reader: &R,
        raw_data: CollectionData,
        raw: &Blob<SuccinctArchiveBlob>,
        output: CollectionData,
    ) -> Option<SuccinctArchive<OrderedUniverse>> {
        let handle = Handle::<SuccinctArchiveRank9IndexBlob>::from_hash(output);
        let rank9: Blob<SuccinctArchiveRank9IndexBlob> = reader.get(handle).ok()?;
        if fresh_data_identity(&rank9) != output {
            return None;
        }
        let source = SuccinctArchiveRank9IndexBlob::source_handle(&rank9).ok()?;
        if Handle::<SuccinctArchiveBlob>::to_hash(source) != raw_data {
            return None;
        }
        SuccinctArchive::from_blob_pair(raw.clone(), rank9).ok()
    }

    fn publish_mapping_closure<S: BlobStorePut>(
        &self,
        store: &mut S,
    ) -> Result<(), Rank9FiberError> {
        let actual = crate::collection::descriptor::put_closure(store, &self.mapping)
            .map_err(|error| Rank9FiberError::storage("store Rank9 mapping closure", error))?;
        if actual != self.mapping_handle {
            return Err(Rank9FiberError::NonCanonicalMappingPut {
                expected: self.mapping_handle,
                actual,
            });
        }
        Ok(())
    }

    fn mapping_closure_is_complete<R: BlobStoreGet>(&self, reader: &R) -> bool {
        let Ok(blob): Result<Blob<SimpleArchive>, _> = reader.get(self.mapping_handle) else {
            return false;
        };
        if blob.get_handle() != self.mapping_handle {
            return false;
        }
        let Ok(facts) = <TribleSet as crate::blob::TryFromBlob<SimpleArchive>>::try_from_blob(blob)
        else {
            return false;
        };
        if &facts != self.mapping.facts() {
            return false;
        }
        let mut blobs = self.mapping.blobs().clone();
        for (handle, expected) in blobs
            .reader()
            .expect("MemoryBlobStore::reader is infallible")
        {
            let Ok(actual): Result<Blob<UnknownBlob>, _> = reader.get(handle) else {
                return false;
            };
            if actual.get_handle() != handle || actual.bytes != expected.bytes {
                return false;
            }
        }
        true
    }

    fn verify_published<S>(
        &self,
        store: &mut S,
        members: Vec<ProbedMember>,
    ) -> Result<UnionArchive<OrderedUniverse>, Rank9FiberError>
    where
        S: BlobStore + MappingEvidenceStore,
    {
        let expected: Vec<_> = members
            .iter()
            .map(|member| {
                MappingEvidence::new(
                    self.mapping_handle,
                    member.raw_data,
                    member
                        .output
                        .expect("published Rank9 member has an exact output"),
                )
            })
            .collect();
        let selectors = expected
            .iter()
            .map(|evidence| MappingEvidenceSelector::Id(evidence.id()))
            .collect();
        let found: BTreeSet<_> = store
            .select_evidence(&selectors)
            .map_err(|error| {
                Rank9FiberError::storage("re-select published Rank9 mapping evidence", error)
            })?
            .into_iter()
            .collect();
        for evidence in &expected {
            if !found.contains(evidence) {
                return Err(Rank9FiberError::IncompletePublication {
                    raw: evidence.input(),
                    rank9: Some(evidence.output()),
                    reason: format!(
                        "expected mapping evidence {} is not resident",
                        evidence.id()
                    ),
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
        if !self.mapping_closure_is_complete(&reader) {
            return Err(Rank9FiberError::IncompletePublication {
                raw: raw_context,
                rank9: None,
                reason: "Rank9 mapping fragment or one of its attachments is not resident"
                    .to_owned(),
            });
        }

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
                    reason: "fresh raw endpoint hash does not match its cover".to_owned(),
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
                    reason: "fresh Rank9 endpoint hash does not match its evidence".to_owned(),
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
}

fn fresh_data_identity<E: BlobEncoding>(blob: &Blob<E>) -> CollectionData
where
    Handle<E>: InlineEncoding,
{
    Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
}
