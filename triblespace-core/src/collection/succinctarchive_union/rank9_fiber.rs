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
use crate::blob::{Blob, IntoBlob};
use crate::collection::{
    CollectionData, CollectionHandle, CoverAttachment, MappingEvidence, MappingEvidenceSelector,
    MappingEvidenceStore, MappingHandle,
};
use crate::inline::encodings::hash::Handle;
use crate::repo::{ArtifactOfferStore, BlobStore, BlobStoreGet, BlobStorePut, OfferCapture};
use crate::trible::Fragment;

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
    /// A transient or persisted Rank9 runtime could not be constructed.
    Build {
        /// Raw member whose Rank9 runtime failed.
        raw: CollectionData,
        /// Exact raw/Rank9 validation failure.
        source: SuccinctArchiveError,
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
            Self::Build { raw, source } => write!(
                f,
                "build Rank9 accelerator for raw member {}: {source}",
                hex::encode_upper(raw.raw),
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
        self.mapping_complete
            && self
                .members
                .iter()
                .all(|member| member.runtime.is_some() && member.prepared.is_none())
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
    /// fixed raw cover, retaining freshly built runtimes across publication.
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

        let mut capture = OfferCapture::new(store);
        let store = &mut capture;
        self.publish_mapping_closure(store)?;

        for member in &mut members {
            let Some(rank9) = member.prepared.take() else {
                continue;
            };
            store
                .put::<SuccinctArchiveRank9IndexBlob, _>(rank9)
                .map_err(|error| Rank9FiberError::storage("store Rank9 sidecar", error))?;
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

        Ok(UnionArchive::new(
            members
                .into_iter()
                .map(|member| {
                    member
                        .runtime
                        .expect("every ensured Rank9 member retains its runtime")
                })
                .collect::<Vec<_>>(),
        ))
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
            let unique = candidates
                .remove(&raw_data)
                .and_then(|outputs| (!outputs.ambiguous).then_some(outputs.first));
            let mut output = None;
            let mut prepared = None;
            let mut runtime = None;

            if let Some(candidate) = unique {
                if let Some(attached) = self.try_attach(&reader, raw_data, &raw, candidate) {
                    output = Some(candidate);
                    runtime = Some(attached);
                }
            }

            if ensure && runtime.is_none() {
                // Missing, corrupt, source-mismatched, or ambiguous evidence
                // is a cache miss. Rebuild the one canonical sidecar instead
                // of choosing among claims supplied by the store.
                let built: SuccinctArchive<OrderedUniverse> =
                    raw.clone()
                        .try_from_blob()
                        .map_err(|source| Rank9FiberError::Build {
                            raw: raw_data,
                            source,
                        })?;
                let rank9 = built.rank9_blob();
                output = Some(Handle::<SuccinctArchiveRank9IndexBlob>::to_hash(
                    rank9.get_handle(),
                ));
                prepared = Some(rank9);
                runtime = Some(built);
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
        crate::collection::descriptor::put_closure(store, &self.mapping)
            .map_err(|error| Rank9FiberError::storage("store Rank9 mapping closure", error))?;
        Ok(())
    }

    fn mapping_closure_is_complete<R: BlobStoreGet>(&self, reader: &R) -> bool {
        let Ok(_): Result<Blob<SimpleArchive>, _> = reader.get(self.mapping_handle) else {
            return false;
        };
        let mut blobs = self.mapping.blobs().clone();
        for (handle, _) in blobs
            .reader()
            .expect("MemoryBlobStore::reader is infallible")
        {
            let Ok(_): Result<Blob<UnknownBlob>, _> = reader.get(handle) else {
                return false;
            };
        }
        true
    }
}
