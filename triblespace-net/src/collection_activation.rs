//! Exact per-collection state which can change commit activation.
//!
//! Collection records and WRITE capability evidence are independent grow-only
//! sets. A newly arrived proof may activate an old COMMIT without changing the
//! record PATCH, so a collection wake must commit to both components. This
//! module constructs that product without applying READ policy or a clock.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519_dalek::VerifyingKey;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, TryFromBlob};
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityProofBundle, CapabilityProofBundleError,
    CapabilityProofError, CapabilityProofId, CapabilityResource,
};
use triblespace_core::collection::{
    ACTION_READ, ACTION_WRITE, AdmissionPolicy, CollectionDescriptorError, CollectionHandle,
    CollectionPolicy, CollectionRead, CollectionRecord, RecordDecodeError,
    collection_writer_is_admitted_by_policy_at, descriptor,
};
use triblespace_core::patch::{Blake3Merkle, Entry as PatchEntry, IdentitySchema, PATCH};
use triblespace_core::repo::{BlobStoreGet, CapabilityProofRead};
use triblespace_core::trible::TribleSet;

use crate::collection_delta::{
    CollectionRecordPatch, CollectionRecordPatchError, collection_record_patch,
};
use crate::patch_repair::PatchSummary;

const COLLECTION_ACTIVATION_ROOT_DOMAIN: &[u8] = b"triblespace.collection.activation-overlay\0";
const COLLECTION_ACTIVATION_ROOT_VERSION: u32 = 1;

type WriteEvidencePatch = PATCH<32, IdentitySchema, CapabilityProofBundle, Blake3Merkle>;

/// Canonical set of complete WRITE proof bundles relevant to one collection.
///
/// Keys are exact proof ids. The corresponding full portable bundle is kept as
/// the leaf value so later PATCH repair can transfer one self-contained proof
/// without a claim-fetch round trip. The Merkle root commits to keys only; each
/// proof key already commits to every claim handle, and construction verifies
/// that the carried claim bytes have those exact handles.
#[derive(Clone, Debug)]
pub struct CollectionWriteEvidencePatch {
    collection: CollectionHandle,
    bundles: WriteEvidencePatch,
}

impl CollectionWriteEvidencePatch {
    /// Exact collection whose WRITE atom shaped this evidence set.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Root and count of the immutable proof-bundle PATCH.
    pub fn summary(&self) -> PatchSummary {
        PatchSummary::from_patch(&self.bundles)
    }

    /// Number of distinct complete proof bundles.
    pub fn len(&self) -> u64 {
        self.bundles.len()
    }

    /// Whether no proof evidence is needed or currently complete.
    pub fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }

    /// Look up one exact portable bundle by proof identity.
    pub fn get(&self, id: CapabilityProofId) -> Option<&CapabilityProofBundle> {
        self.bundles.get(&id.raw)
    }

    /// Enumerate every retained bundle in proof-id order.
    pub fn bundles(&self) -> impl Iterator<Item = &CapabilityProofBundle> {
        self.bundles.iter_ordered().map(|id| {
            self.bundles
                .get(id)
                .expect("an ordered WRITE-evidence key retains its bundle value")
        })
    }

    pub(crate) const fn patch(&self) -> &WriteEvidencePatch {
        &self.bundles
    }
}

/// The two immutable components which determine one collection's activation.
#[derive(Clone, Debug)]
pub struct CollectionActivationOverlay {
    collection: CollectionHandle,
    policy: CollectionPolicy,
    records: CollectionRecordPatch,
    write_evidence: CollectionWriteEvidencePatch,
}

impl CollectionActivationOverlay {
    /// Exact collection represented by both component PATCHes.
    pub const fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Validated immutable descriptor policy which shaped this overlay.
    ///
    /// A host can reuse this value for request-supplied READ admission without
    /// retaining a generic store snapshot or decoding the descriptor twice.
    pub const fn policy(&self) -> &CollectionPolicy {
        &self.policy
    }

    /// Currently WRITE-admitted signed COMMITs for this collection.
    pub const fn records(&self) -> &CollectionRecordPatch {
        &self.records
    }

    /// Complete, structurally valid WRITE proof bundles relevant to policy.
    pub const fn write_evidence(&self) -> &CollectionWriteEvidencePatch {
        &self.write_evidence
    }

    /// Opaque digest suitable for the collection gossip wake root.
    ///
    /// Counts participate alongside roots so the digest commits to the same
    /// authenticated component summaries used by PATCH repair. Neither a
    /// proof, record, count, nor component root is disclosed by this value.
    pub fn wake_root(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(COLLECTION_ACTIVATION_ROOT_DOMAIN);
        hasher.update(&COLLECTION_ACTIVATION_ROOT_VERSION.to_be_bytes());
        hasher.update(&self.collection.raw);
        update_summary(&mut hasher, self.records.summary());
        update_summary(&mut hasher, self.write_evidence.summary());
        *hasher.finalize().as_bytes()
    }
}

fn update_summary(hasher: &mut blake3::Hasher, summary: PatchSummary) {
    match summary.root() {
        Some(root) => {
            hasher.update(&[1]);
            hasher.update(&root);
        }
        None => {
            hasher.update(&[0]);
            hasher.update(&[0; 32]);
        }
    }
    hasher.update(&summary.leaf_count().to_be_bytes());
}

/// A portable bundle is not exact WRITE evidence for the stated policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionWriteEvidenceError {
    /// Open WRITE policy needs no proof evidence.
    OpenPolicy,
    /// The proof starts outside the descriptor's canonical root set.
    WrongRoot,
    /// Signature, claim closure, parentage, atom, mode, or validity geometry is invalid.
    Invalid(CapabilityProofError),
    /// The bounded portable frame is malformed.
    Codec(CapabilityProofBundleError),
    /// Cryptographically distinct bundle values claimed one proof identity.
    ProofIdCollision(CapabilityProofId),
}

impl fmt::Display for CollectionWriteEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenPolicy => formatter.write_str("open WRITE policy needs no proof evidence"),
            Self::WrongRoot => {
                formatter.write_str("capability proof starts outside the WRITE policy roots")
            }
            Self::Invalid(source) => write!(formatter, "invalid WRITE proof bundle: {source}"),
            Self::Codec(source) => write!(formatter, "decode WRITE proof bundle: {source}"),
            Self::ProofIdCollision(id) => write!(
                formatter,
                "distinct WRITE proof bundles share id {}",
                hex::encode_upper(id.raw),
            ),
        }
    }
}

impl Error for CollectionWriteEvidenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalid(source) => Some(source),
            Self::Codec(source) => Some(source),
            _ => None,
        }
    }
}

/// Failure while selecting the bounded portable READ proof forest for `C`.
#[derive(Debug)]
pub enum CollectionReadEvidenceError<ProofsError, GetError> {
    /// The descriptor is absent or structurally invalid.
    Descriptor(CollectionDescriptorError<GetError>),
    /// The coherent proof-store observation failed.
    Proofs(ProofsError),
    /// More relevant bundles exist than the caller's transport bound permits.
    TooMany {
        /// Exact number of canonical relevant bundles.
        count: usize,
        /// Caller-supplied maximum.
        limit: usize,
    },
    /// Cryptographically distinct bundle values claimed one proof identity.
    ProofIdCollision(CapabilityProofId),
}

impl<ProofsError, GetError> fmt::Display for CollectionReadEvidenceError<ProofsError, GetError>
where
    ProofsError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(source) => source.fmt(formatter),
            Self::Proofs(source) => write!(formatter, "enumerate capability proofs: {source}"),
            Self::TooMany { count, limit } => write!(
                formatter,
                "collection READ evidence has {count} bundles; limit is {limit}",
            ),
            Self::ProofIdCollision(id) => write!(
                formatter,
                "distinct READ proof bundles share id {}",
                hex::encode_upper(id.raw),
            ),
        }
    }
}

impl<ProofsError, GetError> Error for CollectionReadEvidenceError<ProofsError, GetError>
where
    ProofsError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(source) => Some(source),
            Self::Proofs(source) => Some(source),
            Self::TooMany { .. } | Self::ProofIdCollision(_) => None,
        }
    }
}

/// Failure while freezing the activation-relevant overlay of one collection.
#[derive(Debug)]
pub enum CollectionActivationOverlayError<RecordsError, ProofsError, GetError> {
    /// Exact collection-record selection failed.
    Records(CollectionRecordPatchError<RecordsError>),
    /// The descriptor is absent or structurally invalid.
    Descriptor(CollectionDescriptorError<GetError>),
    /// The coherent proof-store observation failed.
    Proofs(ProofsError),
    /// Canonical evidence construction found a proof-id collision.
    Evidence(CollectionWriteEvidenceError),
}

impl<RecordsError, ProofsError, GetError> fmt::Display
    for CollectionActivationOverlayError<RecordsError, ProofsError, GetError>
where
    RecordsError: fmt::Display,
    ProofsError: fmt::Display,
    GetError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Records(source) => source.fmt(formatter),
            Self::Descriptor(source) => source.fmt(formatter),
            Self::Proofs(source) => write!(formatter, "enumerate capability proofs: {source}"),
            Self::Evidence(source) => source.fmt(formatter),
        }
    }
}

impl<RecordsError, ProofsError, GetError> Error
    for CollectionActivationOverlayError<RecordsError, ProofsError, GetError>
where
    RecordsError: Error + 'static,
    ProofsError: Error + 'static,
    GetError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Records(source) => Some(source),
            Self::Descriptor(source) => Some(source),
            Self::Proofs(source) => Some(source),
            Self::Evidence(source) => Some(source),
        }
    }
}

/// Freeze the exact currently WRITE-accountable COMMIT and structural
/// WRITE-evidence PATCHes for `C`.
///
/// Missing or malformed descriptors fail closed. Invalid, incomplete, or
/// irrelevant ambient proofs are inert, matching ordinary collection
/// admission; failure to enumerate the coherent proof snapshot is an error.
pub fn collection_activation_overlay<R>(
    snapshot: &R,
    collection: CollectionHandle,
) -> Result<
    CollectionActivationOverlay,
    CollectionActivationOverlayError<R::RecordsError, R::ProofsError, R::GetError<Infallible>>,
>
where
    R: BlobStoreGet + CapabilityProofRead + CollectionRead,
{
    collection_activation_overlay_at(snapshot, collection, crate::clock::epoch_now())
}

pub(crate) fn collection_activation_overlay_at<R>(
    snapshot: &R,
    collection: CollectionHandle,
    instant: hifitime::Epoch,
) -> Result<
    CollectionActivationOverlay,
    CollectionActivationOverlayError<R::RecordsError, R::ProofsError, R::GetError<Infallible>>,
>
where
    R: BlobStoreGet + CapabilityProofRead + CollectionRead,
{
    let policy = load_collection_policy(snapshot, collection)
        .map_err(CollectionActivationOverlayError::Descriptor)?;
    let write_evidence = collection_write_evidence_patch(snapshot, collection, policy.write())?;
    let records = collection_record_patch(snapshot, collection)
        .map_err(CollectionActivationOverlayError::Records)?;
    let bundles = write_evidence.bundles().cloned().collect::<Vec<_>>();
    let mut admitted = BTreeMap::new();
    let records = records.filter(|record| {
        let CollectionRecord::Commit(commit) = record else {
            return false;
        };
        *admitted.entry(commit.public_key().raw).or_insert_with(|| {
            VerifyingKey::from_bytes(&commit.public_key().raw).is_ok_and(|writer| {
                collection_writer_is_admitted_by_policy_at(
                    collection, &policy, writer, &bundles, instant,
                )
            })
        })
    });
    Ok(CollectionActivationOverlay {
        collection,
        policy,
        records,
        write_evidence,
    })
}

pub(crate) fn load_collection_policy<R>(
    snapshot: &R,
    collection: CollectionHandle,
) -> Result<CollectionPolicy, CollectionDescriptorError<R::GetError<Infallible>>>
where
    R: BlobStoreGet,
{
    let descriptor_blob: Blob<SimpleArchive> = snapshot
        .get(collection)
        .map_err(|source| CollectionDescriptorError::Get { collection, source })?;
    let facts = TribleSet::try_from_blob(descriptor_blob).map_err(|source| {
        CollectionDescriptorError::Invalid {
            collection,
            source: RecordDecodeError::from(source),
        }
    })?;
    descriptor::validate(&facts)
        .map_err(|source| CollectionDescriptorError::Invalid { collection, source })
}

/// Select the deterministic bounded portable proof forest for exact READ(C).
///
/// The descriptor's canonical READ roots shape the result. Each returned
/// bundle has a valid signature/claim chain and exact READ atom. Selection and
/// deletion minimization sample one current instant; the receiver independently
/// applies its own current instant during admission. Invalid, incomplete,
/// irrelevant, and duplicate ambient proofs are inert. The caller chooses `max_bundles`; a
/// larger forest fails rather than silently dropping paths required by quorum
/// or fixed-point delegation.
pub fn collection_read_evidence_bundles<R>(
    snapshot: &R,
    collection: CollectionHandle,
    subject: VerifyingKey,
    max_bundles: usize,
) -> Result<
    Vec<CapabilityProofBundle>,
    CollectionReadEvidenceError<R::ProofsError, R::GetError<Infallible>>,
>
where
    R: BlobStoreGet + CapabilityProofRead,
{
    collection_read_evidence_bundles_at(
        snapshot,
        collection,
        subject,
        max_bundles,
        crate::clock::epoch_now(),
    )
}

pub(crate) fn collection_read_evidence_bundles_at<R>(
    snapshot: &R,
    collection: CollectionHandle,
    subject: VerifyingKey,
    max_bundles: usize,
    instant: hifitime::Epoch,
) -> Result<
    Vec<CapabilityProofBundle>,
    CollectionReadEvidenceError<R::ProofsError, R::GetError<Infallible>>,
>
where
    R: BlobStoreGet + CapabilityProofRead,
{
    let policy = load_collection_policy(snapshot, collection)
        .map_err(CollectionReadEvidenceError::Descriptor)?;
    let read = policy.read();
    if matches!(read, AdmissionPolicy::Open) {
        return Ok(Vec::new());
    }

    let proofs = snapshot
        .proofs()
        .map_err(CollectionReadEvidenceError::Proofs)?;
    let atom = collection_atom(ACTION_READ, collection);
    let mut canonical = BTreeMap::<[u8; 32], CapabilityProofBundle>::new();
    for proof in proofs {
        let proof = proof.map_err(CollectionReadEvidenceError::Proofs)?;
        if !root_is_relevant(read, proof.root_key()) {
            continue;
        }
        let Some(claims) = proof
            .claim_handles()
            .map(|claim| {
                snapshot
                    .get::<Blob<SimpleArchive>, SimpleArchive>(claim)
                    .ok()
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let bundle = CapabilityProofBundle::new(proof, claims);
        if bundle.validate_structure_for_atom(atom).is_err() {
            continue;
        }
        let id = bundle.proof().id();
        if let Some(existing) = canonical.get(&id.raw) {
            if existing != &bundle {
                return Err(CollectionReadEvidenceError::ProofIdCollision(id));
            }
            continue;
        }
        canonical.insert(id.raw, bundle);
    }
    let mut selected = canonical.into_values().collect::<Vec<_>>();
    if !triblespace_core::collection::collection_reader_is_admitted_by_policy_at(
        collection, &policy, subject, &selected, instant,
    ) {
        return Ok(Vec::new());
    }
    // Delete every bundle that is not required by the actual fixed-point
    // witness. This preserves intermediate multi-root delegation support while
    // withholding unrelated ambient grants from the remote endpoint.
    let mut index = selected.len();
    while index > 0 {
        index -= 1;
        let removed = selected.remove(index);
        if !triblespace_core::collection::collection_reader_is_admitted_by_policy_at(
            collection, &policy, subject, &selected, instant,
        ) {
            selected.insert(index, removed);
        }
    }
    if selected.len() > max_bundles {
        return Err(CollectionReadEvidenceError::TooMany {
            count: selected.len(),
            limit: max_bundles,
        });
    }
    Ok(selected)
}

fn collection_write_evidence_patch<R>(
    snapshot: &R,
    collection: CollectionHandle,
    policy: &AdmissionPolicy,
) -> Result<
    CollectionWriteEvidencePatch,
    CollectionActivationOverlayError<R::RecordsError, R::ProofsError, R::GetError<Infallible>>,
>
where
    R: BlobStoreGet + CapabilityProofRead + CollectionRead,
{
    if matches!(policy, AdmissionPolicy::Open) {
        return Ok(CollectionWriteEvidencePatch {
            collection,
            bundles: PATCH::new(),
        });
    }

    let proofs = snapshot
        .proofs()
        .map_err(CollectionActivationOverlayError::Proofs)?;
    let mut bundles = Vec::new();
    for proof in proofs {
        let proof = proof.map_err(CollectionActivationOverlayError::Proofs)?;
        if !root_is_relevant(policy, proof.root_key()) {
            continue;
        }
        let Some(claims) = proof
            .claim_handles()
            .map(|claim| {
                snapshot
                    .get::<Blob<SimpleArchive>, SimpleArchive>(claim)
                    .ok()
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        bundles.push(CapabilityProofBundle::new(proof, claims));
    }
    canonical_write_evidence(collection, policy, bundles)
        .map_err(CollectionActivationOverlayError::Evidence)
}

fn canonical_write_evidence(
    collection: CollectionHandle,
    policy: &AdmissionPolicy,
    bundles: impl IntoIterator<Item = CapabilityProofBundle>,
) -> Result<CollectionWriteEvidencePatch, CollectionWriteEvidenceError> {
    let mut canonical = WriteEvidencePatch::new();
    for bundle in bundles {
        if validate_write_evidence_bundle(collection, policy, &bundle).is_err() {
            continue;
        }
        let id = bundle.proof().id();
        if let Some(existing) = canonical.get(&id.raw) {
            if existing != &bundle {
                return Err(CollectionWriteEvidenceError::ProofIdCollision(id));
            }
            continue;
        }
        canonical.insert(&PatchEntry::with_value(&id.raw, bundle));
    }
    Ok(CollectionWriteEvidencePatch {
        collection,
        bundles: canonical,
    })
}

fn root_is_relevant(policy: &AdmissionPolicy, root: ed25519_dalek::VerifyingKey) -> bool {
    policy.roots().is_some_and(|roots| {
        roots
            .binary_search_by_key(&root.to_bytes(), ed25519_dalek::VerifyingKey::to_bytes)
            .is_ok()
    })
}

fn collection_atom(
    action: triblespace_core::id::Id,
    collection: CollectionHandle,
) -> CapabilityAtom {
    CapabilityAtom::new(
        CapabilityAction::new(action),
        CapabilityResource::from(collection),
    )
}

fn write_atom(collection: CollectionHandle) -> CapabilityAtom {
    collection_atom(ACTION_WRITE, collection)
}

/// Strictly validate one complete portable bundle for a descriptor's WRITE law.
pub fn validate_write_evidence_bundle(
    collection: CollectionHandle,
    policy: &AdmissionPolicy,
    bundle: &CapabilityProofBundle,
) -> Result<(), CollectionWriteEvidenceError> {
    if matches!(policy, AdmissionPolicy::Open) {
        return Err(CollectionWriteEvidenceError::OpenPolicy);
    }
    if !root_is_relevant(policy, bundle.proof().root_key()) {
        return Err(CollectionWriteEvidenceError::WrongRoot);
    }
    bundle
        .validate_structure_for_atom(write_atom(collection))
        .map_err(CollectionWriteEvidenceError::Invalid)
}

/// Encode one exact, bounded WRITE-evidence leaf after strict validation.
pub fn encode_write_evidence_bundle(
    collection: CollectionHandle,
    policy: &AdmissionPolicy,
    bundle: &CapabilityProofBundle,
) -> Result<Vec<u8>, CollectionWriteEvidenceError> {
    validate_write_evidence_bundle(collection, policy, bundle)?;
    bundle
        .to_bytes()
        .map_err(CollectionWriteEvidenceError::Codec)
}

/// Decode one complete bounded WRITE-evidence leaf and reject irrelevant data.
pub fn decode_write_evidence_bundle(
    collection: CollectionHandle,
    policy: &AdmissionPolicy,
    bytes: &[u8],
) -> Result<CapabilityProofBundle, CollectionWriteEvidenceError> {
    let bundle =
        CapabilityProofBundle::from_bytes(bytes).map_err(CollectionWriteEvidenceError::Codec)?;
    validate_write_evidence_bundle(collection, policy, &bundle)?;
    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use ed25519_dalek::SigningKey;
    use hifitime::Epoch;
    use triblespace_core::capability::{
        CapabilityClaim, CapabilityMode, CapabilityRequest, CapabilityValidity,
        capability_quorum_authorizes,
    };
    use triblespace_core::collection::{
        CollectionCommit, CollectionData, CollectionPolicy, CollectionRecord, CollectionStore,
        CollectionStoreExt, empty_metadata_handle,
    };
    use triblespace_core::inline::Inline;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStorePut, CapabilityProofStore, SnapshotSource};

    use super::*;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn policy(roots: &[SigningKey], invoke: u32, delegate: Option<u32>) -> CollectionPolicy {
        CollectionPolicy::new(
            AdmissionPolicy::Open,
            AdmissionPolicy::quorum(
                roots.iter().map(SigningKey::verifying_key),
                invoke,
                delegate,
            )
            .unwrap(),
        )
    }

    fn root_bundle(
        root: &SigningKey,
        subject: &SigningKey,
        atom: CapabilityAtom,
        mode: CapabilityMode,
        validity: Option<CapabilityValidity>,
    ) -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            root,
            CapabilityClaim::root(atom, mode, validity),
            subject.verifying_key(),
        )
        .unwrap()
    }

    fn store_bundle(store: &mut MemoryRepo, bundle: CapabilityProofBundle) {
        let (proof, claims) = bundle.into_parts();
        for claim in claims {
            store.put::<SimpleArchive, _>(claim).unwrap();
        }
        store.insert_proof(proof).unwrap();
    }

    #[test]
    fn later_proof_changes_wake_root_without_changing_records() {
        let root = key(1);
        let writer = key(2);
        let mut store = MemoryRepo::default();
        let collection = store
            .collection("activation", policy(&[root.clone()], 1, None))
            .unwrap();
        store
            .insert(CollectionRecord::Commit(CollectionCommit::sign(
                &writer,
                collection.handle(),
                data(3),
                empty_metadata_handle(),
            )))
            .unwrap();

        let before_snapshot = store.snapshot().unwrap();
        let before = collection_activation_overlay(&before_snapshot, collection.handle()).unwrap();
        assert!(
            !collection
                .writer_is_admitted_at(
                    &before_snapshot,
                    writer.verifying_key(),
                    Epoch::from_tai_seconds(0.0),
                )
                .unwrap()
        );
        let atom = write_atom(collection.handle());
        store_bundle(
            &mut store,
            root_bundle(&root, &writer, atom, CapabilityMode::Invoke, None),
        );
        let after_snapshot = store.snapshot().unwrap();
        let after = collection_activation_overlay(&after_snapshot, collection.handle()).unwrap();
        assert!(
            collection
                .writer_is_admitted_at(
                    &after_snapshot,
                    writer.verifying_key(),
                    Epoch::from_tai_seconds(0.0),
                )
                .unwrap()
        );

        assert!(before.records().summary().root().is_none());
        assert_eq!(after.records().summary().leaf_count(), 1);
        assert_ne!(
            before.write_evidence().summary(),
            after.write_evidence().summary()
        );
        assert_ne!(before.wake_root(), after.wake_root());
    }

    #[test]
    fn evidence_shape_is_independent_of_the_clock() {
        let root = key(4);
        let writer = key(5);
        let collection = Inline::new([6; 32]);
        let atom = write_atom(collection);
        let validity =
            CapabilityValidity::new(Epoch::from_tai_seconds(10.0), Epoch::from_tai_seconds(20.0))
                .unwrap();
        let bundle = root_bundle(&root, &writer, atom, CapabilityMode::Invoke, Some(validity));
        let write_policy = AdmissionPolicy::direct(root.verifying_key());
        let evidence =
            canonical_write_evidence(collection, &write_policy, [bundle.clone()]).unwrap();

        assert_eq!(evidence.len(), 1);
        assert!(bundle.validate_structure_for_atom(atom).is_ok());
        let request = CapabilityRequest::new(atom, CapabilityMode::Invoke);
        assert!(
            bundle
                .verify(
                    root.verifying_key(),
                    Epoch::from_tai_seconds(0.0),
                    writer.verifying_key(),
                    request,
                )
                .is_err()
        );
        assert!(
            bundle
                .verify(
                    root.verifying_key(),
                    Epoch::from_tai_seconds(30.0),
                    writer.verifying_key(),
                    request,
                )
                .is_err()
        );
    }

    #[test]
    fn portable_codec_rejects_wrong_scope_and_tampering() {
        let root = key(7);
        let other_root = key(8);
        let writer = key(9);
        let collection = Inline::new([10; 32]);
        let write_policy = AdmissionPolicy::direct(root.verifying_key());
        let bundle = root_bundle(
            &root,
            &writer,
            write_atom(collection),
            CapabilityMode::Invoke,
            None,
        );
        let bytes = encode_write_evidence_bundle(collection, &write_policy, &bundle).unwrap();
        assert_eq!(
            decode_write_evidence_bundle(collection, &write_policy, &bytes).unwrap(),
            bundle
        );

        assert!(matches!(
            decode_write_evidence_bundle(Inline::new([11; 32]), &write_policy, &bytes),
            Err(CollectionWriteEvidenceError::Invalid(
                CapabilityProofError::WrongAtom { .. }
            ))
        ));
        let wrong_action = root_bundle(
            &root,
            &writer,
            CapabilityAtom::new(
                CapabilityAction::new(triblespace_core::id::Id::new([12; 16]).unwrap()),
                CapabilityResource::from(collection),
            ),
            CapabilityMode::Invoke,
            None,
        );
        assert!(matches!(
            encode_write_evidence_bundle(collection, &write_policy, &wrong_action),
            Err(CollectionWriteEvidenceError::Invalid(
                CapabilityProofError::WrongAtom { .. }
            ))
        ));
        let wrong_root = root_bundle(
            &other_root,
            &writer,
            write_atom(collection),
            CapabilityMode::Invoke,
            None,
        );
        assert!(matches!(
            encode_write_evidence_bundle(collection, &write_policy, &wrong_root),
            Err(CollectionWriteEvidenceError::WrongRoot)
        ));

        let mut bad_signature = bytes.clone();
        bad_signature[2 + 32] ^= 1;
        assert!(matches!(
            decode_write_evidence_bundle(collection, &write_policy, &bad_signature),
            Err(CollectionWriteEvidenceError::Invalid(
                CapabilityProofError::InvalidSignature { .. }
            ))
        ));
        let proof_len = bundle.proof().as_bytes().len();
        let claim_start = 2 + proof_len + 2;
        let mut bad_claim = bytes;
        bad_claim[claim_start + 63] ^= 1;
        assert!(matches!(
            decode_write_evidence_bundle(collection, &write_policy, &bad_claim),
            Err(CollectionWriteEvidenceError::Invalid(
                CapabilityProofError::ClaimHandleMismatch { .. }
            ))
        ));
    }

    #[test]
    fn canonical_patch_ignores_arrival_order_duplicates_and_irrelevant_proofs() {
        let root = key(13);
        let other_root = key(14);
        let a = key(15);
        let b = key(16);
        let collection = Inline::new([17; 32]);
        let write_policy = AdmissionPolicy::direct(root.verifying_key());
        let first = root_bundle(
            &root,
            &a,
            write_atom(collection),
            CapabilityMode::Invoke,
            None,
        );
        let second = root_bundle(
            &root,
            &b,
            write_atom(collection),
            CapabilityMode::Invoke,
            None,
        );
        let irrelevant = root_bundle(
            &other_root,
            &b,
            write_atom(collection),
            CapabilityMode::Invoke,
            None,
        );

        let left = canonical_write_evidence(
            collection,
            &write_policy,
            [first.clone(), second.clone(), first.clone(), irrelevant],
        )
        .unwrap();
        let right = canonical_write_evidence(collection, &write_policy, [second, first]).unwrap();
        assert_eq!(left.len(), 2);
        assert_eq!(left.summary(), right.summary());
    }

    #[test]
    fn every_bundle_needed_by_fixed_point_quorum_is_preserved() {
        let root_a = key(18);
        let root_b = key(19);
        let bridge = key(20);
        let writer = key(21);
        let collection = Inline::new([22; 32]);
        let atom = write_atom(collection);
        let write_policy =
            AdmissionPolicy::quorum([root_a.verifying_key(), root_b.verifying_key()], 2, Some(2))
                .unwrap();

        let delegated = |root: &SigningKey| {
            let parent = root_bundle(root, &bridge, atom, CapabilityMode::InvokeAndDelegate, None);
            let verified = parent
                .verify(
                    root.verifying_key(),
                    Epoch::from_tai_seconds(0.0),
                    bridge.verifying_key(),
                    CapabilityRequest::new(atom, CapabilityMode::InvokeAndDelegate),
                )
                .unwrap();
            verified
                .delegate(
                    &bridge,
                    CapabilityClaim::delegated(
                        verified.claim_handle(),
                        atom,
                        CapabilityMode::Invoke,
                        None,
                    ),
                    writer.verifying_key(),
                )
                .unwrap()
        };
        let evidence = canonical_write_evidence(
            collection,
            &write_policy,
            [delegated(&root_a), delegated(&root_b)],
        )
        .unwrap();

        assert_eq!(evidence.len(), 2);
        assert!(capability_quorum_authorizes(
            evidence.bundles(),
            [root_a.verifying_key(), root_b.verifying_key()],
            Epoch::from_tai_seconds(0.0),
            writer.verifying_key(),
            CapabilityRequest::new(atom, CapabilityMode::Invoke),
            NonZeroUsize::new(2).unwrap(),
            Some(NonZeroUsize::new(2).unwrap()),
        ));
    }

    #[test]
    fn missing_descriptor_fails_closed_before_overlay_exists() {
        let mut store = MemoryRepo::default();
        let snapshot = store.snapshot().unwrap();
        let result = collection_activation_overlay(&snapshot, Inline::new([23; 32]));
        assert!(matches!(
            result,
            Err(CollectionActivationOverlayError::Descriptor(_))
        ));
    }

    #[test]
    fn read_evidence_is_exact_deterministic_and_transport_bounded() {
        let root = key(24);
        let other_root = key(25);
        let reader = key(26);
        let mut store = MemoryRepo::default();
        let collection = store
            .collection(
                "read-evidence",
                CollectionPolicy::new(
                    AdmissionPolicy::direct(root.verifying_key()),
                    AdmissionPolicy::Open,
                ),
            )
            .unwrap();
        let relevant = root_bundle(
            &root,
            &reader,
            collection_atom(ACTION_READ, collection.handle()),
            CapabilityMode::Invoke,
            None,
        );
        let wrong_action = root_bundle(
            &root,
            &reader,
            write_atom(collection.handle()),
            CapabilityMode::Invoke,
            None,
        );
        let wrong_root = root_bundle(
            &other_root,
            &reader,
            collection_atom(ACTION_READ, collection.handle()),
            CapabilityMode::Invoke,
            None,
        );
        let unrelated_reader = root_bundle(
            &root,
            &key(28),
            collection_atom(ACTION_READ, collection.handle()),
            CapabilityMode::Invoke,
            None,
        );
        store_bundle(&mut store, wrong_root);
        store_bundle(&mut store, unrelated_reader);
        store_bundle(&mut store, relevant.clone());
        store_bundle(&mut store, wrong_action);

        let snapshot = store.snapshot().unwrap();
        let selected = collection_read_evidence_bundles(
            &snapshot,
            collection.handle(),
            reader.verifying_key(),
            1,
        )
        .unwrap();
        assert_eq!(selected, [relevant]);
        let overlay = collection_activation_overlay(&snapshot, collection.handle()).unwrap();
        assert!(
            triblespace_core::collection::collection_reader_is_admitted_by_policy_at(
                collection.handle(),
                overlay.policy(),
                reader.verifying_key(),
                &selected,
                Epoch::from_tai_seconds(0.0),
            )
        );
        assert!(matches!(
            collection_read_evidence_bundles(
                &snapshot,
                collection.handle(),
                reader.verifying_key(),
                0
            ),
            Err(CollectionReadEvidenceError::TooMany { count: 1, limit: 0 })
        ));
    }

    #[test]
    fn open_read_policy_needs_no_portable_evidence() {
        let mut store = MemoryRepo::default();
        let collection = store
            .collection(
                "open-read",
                CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open),
            )
            .unwrap();
        let snapshot = store.snapshot().unwrap();
        let selected = collection_read_evidence_bundles(
            &snapshot,
            collection.handle(),
            key(27).verifying_key(),
            0,
        )
        .unwrap();
        assert!(selected.is_empty());
        assert!(
            triblespace_core::collection::collection_reader_is_admitted_by_policy_at(
                collection.handle(),
                collection_activation_overlay(&snapshot, collection.handle())
                    .unwrap()
                    .policy(),
                key(27).verifying_key(),
                &[],
                Epoch::from_tai_seconds(0.0),
            )
        );
    }
}
