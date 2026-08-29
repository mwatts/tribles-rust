//! Native, unsigned evidence for reusable collection mappings.
//!
//! A [`MappingEvidence`] value records one observed equation
//! `mapping(input) = output`. It is cache metadata rather than collection
//! membership: it carries no signature, grants no authority, and does not
//! choose among competing outputs. Stores retain every distinct equation in a
//! grow-only set, so independently produced evidence converges by union and
//! pile concatenation remains a merge.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::Debug;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::id::{id_hex, Id};
use crate::inline::encodings::hash::{Blake3, Handle};
use crate::inline::Inline;

use super::CollectionData;

/// Stable semantic kind of `MAPPING_EVIDENCE(mapping, input, output)`.
///
/// Minted with `trible genid` on 2026-08-29. The pile record description is
/// rooted at this same anchor.
pub const KIND_MAPPING_EVIDENCE: Id = id_hex!("8CDA7348DEC34BEBC11A32D550BAB7F6");

/// Exact byte length of one canonical mapping-evidence body.
pub const MAPPING_EVIDENCE_BYTES_LEN: usize = 3 * 32;

/// Version of mapping-evidence intrinsic identity derivation.
pub const MAPPING_EVIDENCE_ID_VERSION: u32 = 1;

/// Domain prefix of mapping-evidence intrinsic identity derivation.
pub const MAPPING_EVIDENCE_ID_DOMAIN: &[u8] = b"triblespace.mapping.evidence.id";

/// Content address of one canonical `SimpleArchive` mapping fragment.
///
/// A mapping fragment is ordinary queryable data. Archiving its facts gives
/// it this content identity without introducing a second opaque mapping
/// namespace.
pub type MappingHandle = Inline<Handle<SimpleArchive>>;

/// One exact observed equation `mapping(input) = output`.
///
/// The canonical dense body is exactly the mapping-fragment handle followed
/// by the input and output collection-member digests. Its intrinsic id is
/// domain-separated from every other native record kind. Multiple outputs for
/// one `(mapping, input)` are deliberately distinct visible set members.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MappingEvidence([u8; MAPPING_EVIDENCE_BYTES_LEN]);

impl MappingEvidence {
    /// Construct one canonical mapping equation.
    pub fn new(mapping: MappingHandle, input: CollectionData, output: CollectionData) -> Self {
        let mut bytes = [0u8; MAPPING_EVIDENCE_BYTES_LEN];
        bytes[..32].copy_from_slice(&mapping.raw);
        bytes[32..64].copy_from_slice(&input.raw);
        bytes[64..].copy_from_slice(&output.raw);
        Self(bytes)
    }

    /// Decode one exact canonical dense body.
    ///
    /// Every 32-byte BLAKE3 digest is a valid content address, so fixed length
    /// is the complete structural validity condition.
    pub const fn from_bytes(bytes: [u8; MAPPING_EVIDENCE_BYTES_LEN]) -> Self {
        Self(bytes)
    }

    /// Return the exact canonical dense body.
    pub const fn to_bytes(self) -> [u8; MAPPING_EVIDENCE_BYTES_LEN] {
        self.0
    }

    /// Borrow the exact canonical dense body.
    pub const fn as_bytes(&self) -> &[u8; MAPPING_EVIDENCE_BYTES_LEN] {
        &self.0
    }

    /// Content address of the mapping fragment whose computation was observed.
    pub fn mapping(self) -> MappingHandle {
        Inline::new(field(&self.0, 0))
    }

    /// Source member supplied to the mapping.
    pub fn input(self) -> CollectionData {
        Inline::new(field(&self.0, 1))
    }

    /// Target member produced by the mapping.
    pub fn output(self) -> CollectionData {
        Inline::new(field(&self.0, 2))
    }

    /// Domain-separated intrinsic id of this exact equation.
    pub fn id(self) -> Id {
        let mut hasher = Blake3::new();
        hasher.update(MAPPING_EVIDENCE_ID_DOMAIN);
        hasher.update(&MAPPING_EVIDENCE_ID_VERSION.to_be_bytes());
        hasher.update(&KIND_MAPPING_EVIDENCE.raw());
        hasher.update(&self.0);
        let digest = hasher.finalize();
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&digest[digest.len() - 16..]);
        Id::new(raw).expect("BLAKE3-derived mapping evidence ids must be non-nil")
    }
}

/// One semantic route into a grow-only mapping-evidence set.
///
/// A selector batch is interpreted as set union. In particular,
/// [`MappingInput`](Self::MappingInput) returns every output observed for one
/// input rather than projecting an arbitrary winner.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MappingEvidenceSelector {
    /// Select one exact equation by intrinsic id.
    Id(Id),
    /// Select every observed output for one exact mapping and input.
    MappingInput(MappingHandle, CollectionData),
    /// Select every equation observed for one mapping.
    Mapping(MappingHandle),
}

pub(crate) fn selectors_match_mapping_evidence(
    selectors: &BTreeSet<MappingEvidenceSelector>,
    evidence: MappingEvidence,
) -> bool {
    selectors.contains(&MappingEvidenceSelector::Id(evidence.id()))
        || selectors.contains(&MappingEvidenceSelector::MappingInput(
            evidence.mapping(),
            evidence.input(),
        ))
        || selectors.contains(&MappingEvidenceSelector::Mapping(evidence.mapping()))
}

/// Grow-only storage for unsigned mapping cache evidence.
///
/// Re-inserting the same intrinsic equation is an idempotent success. There is
/// no deletion, replacement, or unique-output operation: choosing whether an
/// output is valid belongs to the mapping implementation and its caller, not
/// to this convergence substrate.
pub trait MappingEvidenceStore {
    /// Failure while enumerating stored evidence.
    type EvidenceError: Error + Debug + Send + Sync + 'static;
    /// Failure while admitting one canonical equation.
    type InsertError: Error + Debug + Send + Sync + 'static;

    /// Borrowing iterator over one deterministic view of known evidence.
    type EvidenceIter<'a>: Iterator<Item = Result<MappingEvidence, Self::EvidenceError>>
    where
        Self: 'a;

    /// Enumerate currently known evidence in intrinsic-id order.
    fn evidence<'a>(&'a mut self) -> Result<Self::EvidenceIter<'a>, Self::EvidenceError>;

    /// Look up one equation by intrinsic id.
    fn evidence_by_id(&mut self, id: Id) -> Result<Option<MappingEvidence>, Self::EvidenceError> {
        for evidence in self.evidence()? {
            let evidence = evidence?;
            match evidence.id().cmp(&id) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Equal => return Ok(Some(evidence)),
                std::cmp::Ordering::Greater => break,
            }
        }
        Ok(None)
    }

    /// Select one deterministic union of semantic evidence routes.
    fn select_evidence(
        &mut self,
        selectors: &BTreeSet<MappingEvidenceSelector>,
    ) -> Result<Vec<MappingEvidence>, Self::EvidenceError> {
        if selectors.is_empty() {
            return Ok(Vec::new());
        }
        let mut selected = Vec::new();
        for evidence in self.evidence()? {
            let evidence = evidence?;
            if selectors_match_mapping_evidence(selectors, evidence) {
                selected.push(evidence);
            }
        }
        Ok(selected)
    }

    /// Insert one canonical equation into the grow-only set.
    fn insert_evidence(&mut self, evidence: MappingEvidence) -> Result<(), Self::InsertError>;
}

impl<S> MappingEvidenceStore for &mut S
where
    S: MappingEvidenceStore + ?Sized,
{
    type EvidenceError = S::EvidenceError;
    type InsertError = S::InsertError;
    type EvidenceIter<'a>
        = S::EvidenceIter<'a>
    where
        Self: 'a;

    fn evidence<'a>(&'a mut self) -> Result<Self::EvidenceIter<'a>, Self::EvidenceError> {
        (**self).evidence()
    }

    fn evidence_by_id(&mut self, id: Id) -> Result<Option<MappingEvidence>, Self::EvidenceError> {
        (**self).evidence_by_id(id)
    }

    fn select_evidence(
        &mut self,
        selectors: &BTreeSet<MappingEvidenceSelector>,
    ) -> Result<Vec<MappingEvidence>, Self::EvidenceError> {
        (**self).select_evidence(selectors)
    }

    fn insert_evidence(&mut self, evidence: MappingEvidence) -> Result<(), Self::InsertError> {
        (**self).insert_evidence(evidence)
    }
}

fn field(bytes: &[u8; MAPPING_EVIDENCE_BYTES_LEN], index: usize) -> [u8; 32] {
    bytes[index * 32..(index + 1) * 32]
        .try_into()
        .expect("mapping evidence field bounds are static")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo::memoryrepo::MemoryRepo;

    fn mapping(byte: u8) -> MappingHandle {
        Inline::new([byte; 32])
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    #[test]
    fn dense_codec_has_exact_mapping_input_output_layout() {
        let evidence = MappingEvidence::new(mapping(1), data(2), data(3));
        let mut expected = [0u8; MAPPING_EVIDENCE_BYTES_LEN];
        expected[..32].fill(1);
        expected[32..64].fill(2);
        expected[64..].fill(3);

        assert_eq!(evidence.to_bytes(), expected);
        assert_eq!(MappingEvidence::from_bytes(expected), evidence);
        assert_eq!(evidence.mapping(), mapping(1));
        assert_eq!(evidence.input(), data(2));
        assert_eq!(evidence.output(), data(3));
    }

    #[test]
    fn intrinsic_id_is_stable_and_field_sensitive() {
        let evidence = MappingEvidence::new(mapping(1), data(2), data(3));
        assert_eq!(evidence.id(), id_hex!("FB3E925CEE02CFD5254D67D348024C7D"));
        assert_eq!(
            evidence.id(),
            MappingEvidence::new(mapping(1), data(2), data(3)).id()
        );
        assert_ne!(
            evidence.id(),
            MappingEvidence::new(mapping(4), data(2), data(3)).id()
        );
        assert_ne!(
            evidence.id(),
            MappingEvidence::new(mapping(1), data(4), data(3)).id()
        );
        assert_ne!(
            evidence.id(),
            MappingEvidence::new(mapping(1), data(2), data(4)).id()
        );
    }

    #[test]
    fn memory_store_preserves_every_output_for_one_mapping_input() {
        let low = MappingEvidence::new(mapping(1), data(2), data(3));
        let high = MappingEvidence::new(mapping(1), data(2), data(4));
        let unrelated = MappingEvidence::new(mapping(5), data(2), data(6));
        let mut repo = MemoryRepo::default();
        repo.insert_evidence(high).unwrap();
        repo.insert_evidence(low).unwrap();
        repo.insert_evidence(unrelated).unwrap();
        repo.insert_evidence(low).unwrap();

        let selectors =
            BTreeSet::from([MappingEvidenceSelector::MappingInput(mapping(1), data(2))]);
        let selected = repo.select_evidence(&selectors).unwrap();
        let mut expected = vec![low, high];
        expected.sort_unstable_by_key(|evidence| evidence.id());
        assert_eq!(selected, expected);
    }
}
