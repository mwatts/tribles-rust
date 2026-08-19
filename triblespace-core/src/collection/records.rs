//! Dense typed records for the top-level collection calculus.
//!
//! [`CollectionCommit`], [`CollectionMerge`], and [`CollectionDerive`] are
//! native algebra records, not graph data. Their canonical representations are
//! the fixed-width byte layouts exposed by their `to_bytes`/`from_bytes`
//! methods. Only [`CollectionDescriptor`] remains a self-describing
//! [`SimpleArchive`]: its blob handle is the collection identity.
//!
//! Structural decoding and semantic verification are deliberately separate.
//! Every fixed-width commit or derive payload has a structural representation;
//! a merge additionally rejects noncanonical input order. A decoded commit can
//! still carry an invalid public key or signature; [`CollectionCommit::verify_strict`]
//! performs that cryptographic check over a fixed, domain-separated transcript.

use std::error::Error;
use std::fmt;

use ed25519::signature::Signer;
use ed25519::Signature;
use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::attribute::Attribute;
use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::{Blob, TryFromBlob};
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::ed25519::{ED25519PublicKey, ED25519RComponent, ED25519SComponent};
use crate::inline::encodings::genid::{GenId, IdParseError};
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::{Inline, InlineEncoding};
use crate::metadata;
use crate::prelude::{attributes, entity};
use crate::id::RawId;
use crate::inline::RawInline;
use crate::trible::{Fragment, TribleSet, TRIBLE_LEN};

/// Tag identifying a canonical collection descriptor.
///
/// Minted with `trible genid` on 2026-08-07.
pub const KIND_COLLECTION_DESCRIPTOR: Id = id_hex!("C5E238729BB95FA4A55E3939B11B3C29");
/// Stable semantic kind of a signed `COMMIT(descriptor, data, metadata)` assertion.
///
/// Minted with `trible genid` on 2026-08-11.
pub const KIND_COLLECTION_COMMIT: Id = id_hex!("B34817308188C4515A3C51967A91A603");
/// Stable semantic kind of an unsigned commutative `MERGE` equation.
///
/// Minted with `trible genid` on 2026-08-11.
pub const KIND_COLLECTION_MERGE: Id = id_hex!("5F20FFC64313969B7E046A7677874D39");
/// Stable semantic kind of an unsigned `DERIVE` equation.
///
/// Minted with `trible genid` on 2026-08-11.
pub const KIND_COLLECTION_DERIVE: Id = id_hex!("46C621338B6DD5B71C8E1E6DD74B087C");

/// The three-field derive's predecessor, which also named its source.
///
/// A derive's source is what the target's descriptor says it is, so naming it
/// again in the record only created a way for the two to disagree. Records
/// under this kind are not read: a derivation is a computation with a
/// checkable artifact, so the cheapest correct thing to do with a stale one is
/// recompute it. Kept here so the id is not minted twice.
///
/// Minted with `trible genid` on 2026-08-07, retired 2026-08-20.
pub const KIND_COLLECTION_DERIVE_V1: Id = id_hex!("6DB0214CB4F3BD8259F0117CDC127331");

/// Byte length of a canonical collection-descriptor `SimpleArchive`.
pub const COLLECTION_DESCRIPTOR_ARCHIVE_LEN: u64 = (4 * TRIBLE_LEN) as u64;
/// Byte length of a dense signed commit.
pub const COLLECTION_COMMIT_BYTES_LEN: usize = 6 * 32;
/// Byte length of a dense merge equation.
pub const COLLECTION_MERGE_BYTES_LEN: usize = 4 * 32;
/// Byte length of a dense derive equation.
pub const COLLECTION_DERIVE_BYTES_LEN: usize = 3 * 32;

/// Version of collection-record identity derivation.
pub const COLLECTION_RECORD_ID_VERSION: u32 = 1;

/// Domain prefix of collection-record identity derivation.
pub const COLLECTION_RECORD_ID_DOMAIN: &[u8] = b"triblespace.collection.record.id";

attributes! {
    /// Stable extrinsic anchor of a *root* dataset.
    ///
    /// Only a collection that is not derived from another carries one. A
    /// derived collection needs no anchor of its own: its source already
    /// anchors it, and its identity follows from that source together with its
    /// representation, recipe and arguments. Minting an anchor for a derived
    /// collection would assert a second, weaker claim about the same lineage.
    ///
    /// Minted with `trible genid` on 2026-08-07.
    "D3418873C70392E3ADAA05C00E11A583" unsafe as pub collection_scope: GenId;
    /// The collection this one derives from, by descriptor handle.
    ///
    /// This says *what* a derived collection is computed from; which state of
    /// that source a given commit reflects belongs on the commit, not here.
    /// A handle rather than a shared label means a descriptor cannot claim a
    /// lineage it does not have: it names one exact source descriptor.
    ///
    /// Minted with `trible genid` on 2026-08-19.
    "8D93B2A626CD32182C0A026BC8D5A014" unsafe as pub collection_source: Handle<SimpleArchive>;
    /// Blob representation carried by the elements of this collection.
    /// Minted with `trible genid` on 2026-08-07.
    "620FA4F2B456357DCD1882E583B85CC3" unsafe as pub collection_representation: GenId;
    /// Canonical recipe governing construction and merge for this collection.
    /// Minted with `trible genid` on 2026-08-07.
    "5D338C58D897B969BE1AE0956CCFE301" unsafe as pub collection_recipe: GenId;
}

/// Type-erased content identity of one collection element.
///
/// The concrete blob encoding is named by the collection's
/// [`collection_representation`] field. Keeping the element itself as a bare
/// Blake3 digest avoids falsely claiming that it has the `UnknownBlob`
/// encoding; after validating the collection descriptor, callers can transmute
/// this digest into the representation's typed [`Handle`].
pub type CollectionData = Inline<Hash<Blake3>>;

/// Content identity of one canonical collection descriptor.
///
/// The descriptor is an ordinary [`SimpleArchive`] blob. Claims carry this
/// handle directly so their collection semantics can be recovered through
/// ordinary blob resolution without a separate definition-record namespace.
pub type CollectionHandle = Inline<Handle<SimpleArchive>>;

/// Version of the signed collection-commit transcript.
pub const COMMIT_TRANSCRIPT_VERSION: u32 = 2;

/// Domain prefix of the signed collection-commit transcript.
pub const COMMIT_TRANSCRIPT_DOMAIN: &[u8] = b"triblespace.collection.commit.transcript";

/// Number of bytes in a version-2 commit transcript.
pub const COMMIT_TRANSCRIPT_LEN: usize = COMMIT_TRANSCRIPT_DOMAIN.len()
    + 16 // kind id
    + 4 // version
    + 32 // public key
    + 32 // collection descriptor handle
    + 32 // data hash
    + 32; // metadata handle

/// Return the canonical handle of an empty metadata archive.
///
/// Metadata is mandatory in a [`CollectionCommit`]. Callers with no metadata
/// use this handle rather than omitting the field, so record arity and signed
/// transcript shape never vary.
pub fn empty_metadata_handle() -> Inline<Handle<SimpleArchive>> {
    encode_archive(TribleSet::new()).get_handle()
}

/// Structural decoding failure for a collection record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordDecodeError {
    /// The bytes were not a canonical `SimpleArchive`.
    Archive(UnarchiveError),
    /// The archive contained no facts.
    Empty,
    /// More than one entity occurred in a record archive.
    MultipleEntities,
    /// A required field was absent.
    MissingField(&'static str),
    /// A single-valued field occurred more than once.
    RepeatedField(&'static str),
    /// A `GenId` field had a noncanonical or nil inline representation.
    InvalidId(&'static str),
    /// The record's marker names another record kind.
    WrongKind { expected: Id, actual: Id },
    /// The stored subject was not the intrinsic root of the canonical fields.
    NonCanonicalRoot { stored: Id, expected: Id },
    /// The archive contained a fact outside the exact canonical record shape.
    NonCanonicalFacts,
    /// A dense record had no kind byte or the wrong payload length.
    InvalidLength { expected: usize, actual: usize },
    /// A tagged dense record used an unknown variant byte.
    UnknownKind(u8),
    /// A merge payload did not carry its inputs in ascending digest order.
    NonCanonicalMergeInputs,
}

impl fmt::Display for RecordDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Archive(error) => write!(f, "invalid SimpleArchive record: {error}"),
            Self::Empty => write!(f, "collection record is empty"),
            Self::MultipleEntities => {
                write!(f, "collection record must contain exactly one entity")
            }
            Self::MissingField(field) => write!(f, "collection record is missing {field}"),
            Self::RepeatedField(field) => {
                write!(f, "collection record contains repeated {field}")
            }
            Self::InvalidId(field) => write!(f, "collection record contains invalid {field}"),
            Self::WrongKind { expected, actual } => write!(
                f,
                "collection record kind {actual:X} does not match expected {expected:X}"
            ),
            Self::NonCanonicalRoot { stored, expected } => write!(
                f,
                "collection record root {stored:X} does not match canonical root {expected:X}"
            ),
            Self::NonCanonicalFacts => {
                write!(f, "collection record contains noncanonical or extra facts")
            }
            Self::InvalidLength { expected, actual } => write!(
                f,
                "collection record has {actual} bytes; expected exactly {expected}"
            ),
            Self::UnknownKind(kind) => {
                write!(f, "collection record has unknown dense kind {kind}")
            }
            Self::NonCanonicalMergeInputs => {
                write!(f, "collection merge inputs are not canonically ordered")
            }
        }
    }
}

impl Error for RecordDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Archive(error) => Some(error),
            _ => None,
        }
    }
}

impl From<UnarchiveError> for RecordDecodeError {
    fn from(error: UnarchiveError) -> Self {
        Self::Archive(error)
    }
}

/// Semantic verification failure for a signed collection commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitVerificationError {
    /// The public-key bytes do not encode an Ed25519 verifying key.
    InvalidPublicKey,
    /// Strict Ed25519 verification rejected the transcript/signature pair.
    InvalidSignature,
}

impl fmt::Display for CommitVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKey => write!(f, "collection commit has an invalid public key"),
            Self::InvalidSignature => write!(f, "collection commit signature is invalid"),
        }
    }
}

impl Error for CommitVerificationError {}

/// Canonical self-describing blob payload for one concrete typed collection.
///
/// `scope` is an extrinsic dataset anchor shared by related
/// representations. The descriptor entity has an intrinsic root, but the
/// collection identity carried by claims is the descriptor blob's 32-byte
/// [`CollectionHandle`]. Constructing a descriptor never manufactures an
/// [`crate::id::ExclusiveId`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionDescriptor {
    facts: TribleSet,
}

impl CollectionDescriptor {
    /// Construct a descriptor that carries its representation's and recipe's
    /// own descriptions.
    ///
    /// The descriptor entity still names them by id, so this does not change
    /// which schema or which law a collection uses. What changes is that their
    /// descriptions travel inside the descriptor blob, so a reader holding
    /// that one blob can say what the collection is without resolving anything
    /// else. That matters for a peer receiving a collection it has never seen:
    /// a bare id is only recognisable to someone who already holds the code
    /// that minted it.
    pub fn new(scope: Id, representation: Fragment, recipe: Fragment) -> Self {
        let fragment = entity! {
            metadata::tag: KIND_COLLECTION_DESCRIPTOR,
            collection_scope: scope,
            collection_representation*: representation,
            collection_recipe*: recipe,
        };
        Self::from_fragment(&fragment)
    }

    /// Construct a descriptor for a collection derived from another.
    ///
    /// A derived collection carries no dataset anchor of its own. Its source
    /// is what anchors it, so two derivations of the same shape over different
    /// data are different collections because their sources differ. Naming the
    /// source by handle rather than by a shared label means the lineage is
    /// exact and cannot be joined by assertion.
    pub fn derived(
        source: CollectionHandle,
        representation: Fragment,
        recipe: Fragment,
    ) -> Self {
        let fragment = entity! {
            metadata::tag: KIND_COLLECTION_DESCRIPTOR,
            collection_source: source,
            collection_representation*: representation,
            collection_recipe*: recipe,
        };
        Self::from_fragment(&fragment)
    }

    /// The collection this one derives from, if it derives from one.
    ///
    /// A root collection has no source and answers `None`; that is not a
    /// failure, it is what being a root means.
    pub fn source(&self) -> Option<CollectionHandle> {
        let attribute = collection_source.id();
        self.facts
            .iter()
            .find(|fact| *fact.a() == attribute)
            .map(|fact| *fact.v::<Handle<SimpleArchive>>())
    }

    /// Construct a descriptor that names its representation and recipe without
    /// describing them.
    ///
    /// Prefer [`new`](Self::new), which embeds their descriptions. This exists
    /// for callers holding only ids: the resulting collection is perfectly
    /// usable, it just cannot tell a stranger what it means.
    pub fn naming(scope: Id, representation: Id, recipe: Id) -> Self {
        Self::new(
            scope,
            Fragment::rooted(representation, TribleSet::new()),
            Fragment::rooted(recipe, TribleSet::new()),
        )
    }

    /// Adopt a descriptor authored by [`entity!`](crate::macros::entity).
    ///
    /// This is how a parameterised recipe builds its collection: write the
    /// descriptor as an ordinary intrinsic entity, with the recipe's own
    /// arguments as further attributes on the same entity, and hand the
    /// fragment here. Those arguments are covered by the descriptor blob's
    /// hash and readable by anyone holding the pile, so two collections differ
    /// exactly when their scope, representation, recipe, or arguments differ.
    pub fn from_fragment(fragment: &Fragment) -> Self {
        Self::from_tribles(fragment.facts())
    }

    /// Decode an exact collection-descriptor archive without external lookups.
    pub fn decode(blob: &Blob<SimpleArchive>) -> Result<Self, RecordDecodeError> {
        Ok(Self::from_tribles(&decode_archive(blob)?))
    }

    /// Decode an exact collection-descriptor entity from an already parsed set.
    ///
    /// Attributes beyond the structural four are recipe arguments. They are
    /// kept verbatim rather than remodelled, so a descriptor for a recipe this
    /// binary has never heard of still decodes, still answers questions about
    /// its scope and law, and still re-emits byte-for-byte on the way out.
    pub fn from_tribles(facts: &TribleSet) -> Self {
        Self {
            facts: facts.clone(),
        }
    }

    /// Intrinsic entity root inside the descriptor archive.
    ///
    /// This is not the collection identity carried by claims; use
    /// [`handle`](Self::handle) for that.
    /// Entity the descriptor's own attributes hang off.
    ///
    /// The archive holds more than one entity: the descriptor, plus the
    /// embedded descriptions of its representation and its recipe. The
    /// descriptor is the one tagged [`KIND_COLLECTION_DESCRIPTOR`].
    ///
    /// This is not the collection identity carried by claims; use
    /// [`handle`](Self::handle) for that.
    pub fn entity_id(&self) -> Result<Id, RecordDecodeError> {
        let tag = metadata::tag.id();
        let expected: Inline<GenId> =
            crate::inline::IntoInline::to_inline(KIND_COLLECTION_DESCRIPTOR);
        let mut found = None;
        for fact in self.facts.iter() {
            if *fact.a() == tag && fact.v::<GenId>().raw == expected.raw {
                if found.is_some() {
                    return Err(RecordDecodeError::MultipleEntities);
                }
                found = Some(*fact.e());
            }
        }
        found.ok_or(RecordDecodeError::WrongKind {
            expected: KIND_COLLECTION_DESCRIPTOR,
            actual: KIND_COLLECTION_DESCRIPTOR,
        })
    }

    /// Canonical content identity of this collection descriptor.
    pub fn handle(&self) -> CollectionHandle {
        self.to_blob().get_handle()
    }

    /// Extrinsic dataset scope shared by related collections.
    pub fn scope(&self) -> Result<Id, RecordDecodeError> {
        one_id(&self.facts, &collection_scope, "collection_scope")
    }

    /// Blob-representation descriptor id.
    pub fn representation(&self) -> Result<Id, RecordDecodeError> {
        one_id(
            &self.facts,
            &collection_representation,
            "collection_representation",
        )
    }

    /// Canonical construction/merge recipe id.
    ///
    /// This names the *law*. Its arguments, if any, are the remaining
    /// attributes on the descriptor entity; see [`argument`](Self::argument).
    pub fn recipe(&self) -> Result<Id, RecordDecodeError> {
        one_id(&self.facts, &collection_recipe, "collection_recipe")
    }

    /// Look up one recipe argument by attribute.
    pub fn argument(&self, attribute: Id) -> Option<RawInline> {
        self.arguments()
            .find(|(a, _)| *a == attribute)
            .map(|(_, v)| v)
    }

    /// Every recipe argument carried by this descriptor.
    pub fn arguments(&self) -> impl Iterator<Item = (Id, RawInline)> + '_ {
        self.facts.iter().filter_map(|fact| {
            let attribute = *fact.a();
            (!structural_attributes().contains(&attribute))
                .then(|| (attribute, fact.v::<GenId>().raw))
        })
    }


    /// Reconstruct the exact one-root trible record.
    ///
    /// This is the archive the descriptor was decoded from, unchanged. There
    /// is no second model of it to drift.
    pub fn to_tribles(&self) -> TribleSet {
        self.facts.clone()
    }

    /// Canonical `SimpleArchive` payload of this descriptor.
    pub fn to_blob(&self) -> Blob<SimpleArchive> {
        encode_archive(self.to_tribles())
    }
}

/// Signed exogenous membership assertion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionCommit {
    id: Id,
    collection: CollectionHandle,
    data: CollectionData,
    metadata: Inline<Handle<SimpleArchive>>,
    public_key: Inline<ED25519PublicKey>,
    signature_r: Inline<ED25519RComponent>,
    signature_s: Inline<ED25519SComponent>,
}

impl CollectionCommit {
    /// Sign a canonical `COMMIT(descriptor, data, metadata)` statement.
    pub fn sign(
        signing_key: &SigningKey,
        collection: CollectionHandle,
        data_hash: CollectionData,
        metadata: Inline<Handle<SimpleArchive>>,
    ) -> Self {
        let public_key = Inline::new(signing_key.verifying_key().to_bytes());
        let transcript = commit_transcript(public_key, collection, data_hash, metadata);
        let signature: Signature = signing_key.sign(&transcript);
        Self::from_parts(
            collection,
            data_hash,
            metadata,
            public_key,
            Inline::new(*signature.r_bytes()),
            Inline::new(*signature.s_bytes()),
        )
    }

    pub(crate) fn from_parts(
        collection: CollectionHandle,
        data_hash: CollectionData,
        metadata: Inline<Handle<SimpleArchive>>,
        public_key: Inline<ED25519PublicKey>,
        r_component: Inline<ED25519RComponent>,
        s_component: Inline<ED25519SComponent>,
    ) -> Self {
        let bytes = commit_bytes(
            collection,
            data_hash,
            metadata,
            public_key,
            r_component,
            s_component,
        );
        let id = collection_record_id(KIND_COLLECTION_COMMIT, &bytes);
        Self {
            id,
            collection,
            data: data_hash,
            metadata,
            public_key,
            signature_r: r_component,
            signature_s: s_component,
        }
    }

    /// Decode one exact dense payload without trusting its signature.
    ///
    /// Every byte string of this fixed size has a structural representation;
    /// invalid public keys and signatures are rejected by [`verify_strict`](Self::verify_strict).
    pub fn from_bytes(bytes: [u8; COLLECTION_COMMIT_BYTES_LEN]) -> Self {
        Self::from_parts(
            Inline::new(field(&bytes, 0)),
            Inline::new(field(&bytes, 1)),
            Inline::new(field(&bytes, 2)),
            Inline::new(field(&bytes, 3)),
            Inline::new(field(&bytes, 4)),
            Inline::new(field(&bytes, 5)),
        )
    }

    /// Strictly verify the Ed25519 signature over the canonical transcript.
    ///
    /// This proves only that the embedded public key signed the record. Key
    /// authorization is a separate caller policy.
    pub fn verify_strict(&self) -> Result<(), CommitVerificationError> {
        let public_key = VerifyingKey::from_bytes(&self.public_key.raw)
            .map_err(|_| CommitVerificationError::InvalidPublicKey)?;
        self.verify_signature_strict(&public_key)
    }

    /// Verify with an already parsed key when it matches this record's key.
    ///
    /// Scoped collection discovery compares the raw key field before calling
    /// this helper. The equality check here keeps that optimization local and
    /// fail-safe if another caller ever violates the precondition.
    pub(crate) fn verify_strict_with_key(
        &self,
        public_key: &VerifyingKey,
    ) -> Result<(), CommitVerificationError> {
        if public_key.to_bytes() != self.public_key.raw {
            return self.verify_strict();
        }
        self.verify_signature_strict(public_key)
    }

    fn verify_signature_strict(
        &self,
        public_key: &VerifyingKey,
    ) -> Result<(), CommitVerificationError> {
        let signature = Signature::from_components(self.signature_r.raw, self.signature_s.raw);
        let transcript =
            commit_transcript(self.public_key, self.collection, self.data, self.metadata);
        public_key
            .verify_strict(&transcript, &signature)
            .map_err(|_| CommitVerificationError::InvalidSignature)
    }

    /// Exact bytes attested by this commit's signature.
    pub fn signing_transcript(&self) -> Vec<u8> {
        commit_transcript(self.public_key, self.collection, self.data, self.metadata).to_vec()
    }

    /// Intrinsic record id.
    pub fn id(&self) -> Id {
        self.id
    }

    #[cfg(test)]
    pub(crate) fn with_test_id(mut self, id: Id) -> Self {
        self.id = id;
        self
    }

    /// Collection receiving the asserted member.
    pub fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Asserted member's content hash.
    pub fn data(&self) -> CollectionData {
        self.data
    }

    /// Mandatory metadata archive handle.
    pub fn metadata(&self) -> Inline<Handle<SimpleArchive>> {
        self.metadata
    }

    /// Raw public-key field. It becomes trusted only after strict verification.
    pub fn public_key(&self) -> Inline<ED25519PublicKey> {
        self.public_key
    }

    /// Raw signature components.
    pub fn signature(&self) -> (Inline<ED25519RComponent>, Inline<ED25519SComponent>) {
        (self.signature_r, self.signature_s)
    }

    /// Encode this record into its exact dense 192-byte layout.
    pub fn to_bytes(&self) -> [u8; COLLECTION_COMMIT_BYTES_LEN] {
        commit_bytes(
            self.collection,
            self.data,
            self.metadata,
            self.public_key,
            self.signature_r,
            self.signature_s,
        )
    }
}

/// Unsigned exact join equation inside one collection lattice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionMerge {
    id: Id,
    collection: CollectionHandle,
    low: CollectionData,
    high: CollectionData,
    result: CollectionData,
}

impl CollectionMerge {
    /// Construct a commutative merge record, sorting its two inputs by digest.
    pub fn new(
        collection: CollectionHandle,
        mut left: CollectionData,
        mut right: CollectionData,
        result: CollectionData,
    ) -> Self {
        if right < left {
            std::mem::swap(&mut left, &mut right);
        }
        Self::from_ordered(collection, left, right, result)
    }

    fn from_ordered(
        collection: CollectionHandle,
        low: CollectionData,
        high: CollectionData,
        result: CollectionData,
    ) -> Self {
        let bytes = merge_bytes(collection, low, high, result);
        let id = collection_record_id(KIND_COLLECTION_MERGE, &bytes);
        Self {
            id,
            collection,
            low,
            high,
            result,
        }
    }

    /// Decode one exact, canonically ordered dense merge payload.
    pub fn from_bytes(bytes: [u8; COLLECTION_MERGE_BYTES_LEN]) -> Result<Self, RecordDecodeError> {
        let collection = Inline::new(field(&bytes, 0));
        let low = Inline::new(field(&bytes, 1));
        let high = Inline::new(field(&bytes, 2));
        if high < low {
            return Err(RecordDecodeError::NonCanonicalMergeInputs);
        }
        Ok(Self::from_ordered(
            collection,
            low,
            high,
            Inline::new(field(&bytes, 3)),
        ))
    }

    /// Intrinsic record id.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Collection whose join law is asserted.
    pub fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Canonically ordered merge inputs.
    pub fn inputs(&self) -> (CollectionData, CollectionData) {
        (self.low, self.high)
    }

    /// Asserted exact join result.
    pub fn result(&self) -> CollectionData {
        self.result
    }

    /// Encode this equation into its exact dense 128-byte layout.
    pub fn to_bytes(&self) -> [u8; COLLECTION_MERGE_BYTES_LEN] {
        merge_bytes(self.collection, self.low, self.high, self.result)
    }
}

/// One unsigned exact observation of the canonical join homomorphism between
/// two collection lattices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionDerive {
    id: Id,
    target: CollectionHandle,
    input: CollectionData,
    output: CollectionData,
}

impl CollectionDerive {
    /// Construct a canonical `DERIVE(target, input, output)` record.
    ///
    /// The target is named by descriptor handle, exactly as a commit names its
    /// collection, and that descriptor already says which collection is the
    /// source and by what recipe. A derive therefore says *which instance* of
    /// a mapping was computed, never *which mapping*.
    pub fn new(
        target: CollectionHandle,
        input: CollectionData,
        output: CollectionData,
    ) -> Self {
        let bytes = derive_bytes(target, input, output);
        let id = collection_record_id(KIND_COLLECTION_DERIVE, &bytes);
        Self {
            id,
            target,
            input,
            output,
        }
    }

    /// Decode one exact dense derive payload.
    pub fn from_bytes(bytes: [u8; COLLECTION_DERIVE_BYTES_LEN]) -> Self {
        Self::new(
            Inline::new(field(&bytes, 0)),
            Inline::new(field(&bytes, 1)),
            Inline::new(field(&bytes, 2)),
        )
    }

    /// Intrinsic record id.
    pub fn id(&self) -> Id {
        self.id
    }

    /// Target collection.
    pub fn target(&self) -> CollectionHandle {
        self.target
    }

    /// Source and target elements.
    pub fn mapping(&self) -> (CollectionData, CollectionData) {
        (self.input, self.output)
    }

    /// Encode this equation into its exact dense 96-byte layout.
    pub fn to_bytes(&self) -> [u8; COLLECTION_DERIVE_BYTES_LEN] {
        derive_bytes(self.target, self.input, self.output)
    }
}

/// A structurally canonical native collection record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectionRecord {
    /// Signed membership assertion whose embedded signature can be verified.
    Commit(CollectionCommit),
    /// Unsigned exact join equation.
    Merge(CollectionMerge),
    /// Unsigned exact mapping equation.
    Derive(CollectionDerive),
}

impl CollectionRecord {
    /// Decode the self-tagged dense form used by generic record stores.
    ///
    /// The first byte identifies the variant; the remainder is that variant's
    /// exact untagged payload. Typed protocols should use the concrete record
    /// codecs directly and avoid this extra byte.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RecordDecodeError> {
        let Some((&kind, payload)) = bytes.split_first() else {
            return Err(RecordDecodeError::InvalidLength {
                expected: 1,
                actual: 0,
            });
        };
        match kind {
            COLLECTION_RECORD_KIND_COMMIT_V1 => {
                let bytes = exact_array::<COLLECTION_COMMIT_BYTES_LEN>(payload)?;
                Ok(Self::Commit(CollectionCommit::from_bytes(bytes)))
            }
            COLLECTION_RECORD_KIND_MERGE_V1 => {
                let bytes = exact_array::<COLLECTION_MERGE_BYTES_LEN>(payload)?;
                Ok(Self::Merge(CollectionMerge::from_bytes(bytes)?))
            }
            COLLECTION_RECORD_KIND_DERIVE_V1 => {
                let bytes = exact_array::<COLLECTION_DERIVE_BYTES_LEN>(payload)?;
                Ok(Self::Derive(CollectionDerive::from_bytes(bytes)))
            }
            unknown => Err(RecordDecodeError::UnknownKind(unknown)),
        }
    }

    /// Intrinsic id of the decoded record entity.
    pub fn id(&self) -> Id {
        match self {
            Self::Commit(record) => record.id(),
            Self::Merge(record) => record.id(),
            Self::Derive(record) => record.id(),
        }
    }

    /// Encode the self-tagged dense form used by generic record stores.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Commit(record) => {
                tagged_bytes(COLLECTION_RECORD_KIND_COMMIT_V1, &record.to_bytes())
            }
            Self::Merge(record) => {
                tagged_bytes(COLLECTION_RECORD_KIND_MERGE_V1, &record.to_bytes())
            }
            Self::Derive(record) => {
                tagged_bytes(COLLECTION_RECORD_KIND_DERIVE_V1, &record.to_bytes())
            }
        }
    }
}

/// Dense generic-store tag for the version-1 [`CollectionRecord::Commit`] layout.
///
/// A future payload layout allocates a new tag rather than reinterpreting this
/// one, so stored bytes remain self-versioning without a second prefix byte.
pub const COLLECTION_RECORD_KIND_COMMIT_V1: u8 = 1;
/// Dense generic-store tag for the version-1 [`CollectionRecord::Merge`] layout.
pub const COLLECTION_RECORD_KIND_MERGE_V1: u8 = 2;
/// Dense generic-store tag for the version-1 [`CollectionRecord::Derive`] layout.
pub const COLLECTION_RECORD_KIND_DERIVE_V1: u8 = 3;


/// The four attributes every descriptor carries. Anything else on the entity
/// is an argument to the recipe.
fn structural_attributes() -> [Id; 4] {
    [
        metadata::tag.id(),
        collection_scope.id(),
        collection_representation.id(),
        collection_recipe.id(),
    ]
}


fn commit_bytes(
    collection: CollectionHandle,
    data_hash: CollectionData,
    metadata_handle: Inline<Handle<SimpleArchive>>,
    public_key: Inline<ED25519PublicKey>,
    r: Inline<ED25519RComponent>,
    s: Inline<ED25519SComponent>,
) -> [u8; COLLECTION_COMMIT_BYTES_LEN] {
    concat_fields([
        collection.raw,
        data_hash.raw,
        metadata_handle.raw,
        public_key.raw,
        r.raw,
        s.raw,
    ])
}

fn merge_bytes(
    collection: CollectionHandle,
    low: CollectionData,
    high: CollectionData,
    result: CollectionData,
) -> [u8; COLLECTION_MERGE_BYTES_LEN] {
    concat_fields([collection.raw, low.raw, high.raw, result.raw])
}

fn derive_bytes(
    target: CollectionHandle,
    input: CollectionData,
    output: CollectionData,
) -> [u8; COLLECTION_DERIVE_BYTES_LEN] {
    concat_fields([target.raw, input.raw, output.raw])
}

fn collection_record_id(kind: Id, payload: &[u8]) -> Id {
    let mut hasher = Blake3::new();
    hasher.update(COLLECTION_RECORD_ID_DOMAIN);
    hasher.update(&COLLECTION_RECORD_ID_VERSION.to_be_bytes());
    hasher.update(&kind.raw());
    hasher.update(payload);
    let digest = hasher.finalize();
    let mut raw = [0u8; 16];
    raw.copy_from_slice(&digest[digest.len() - 16..]);
    Id::new(raw).expect("BLAKE3-derived collection record ids must be non-nil")
}

fn concat_fields<const N: usize, const OUT: usize>(fields: [[u8; 32]; N]) -> [u8; OUT] {
    debug_assert_eq!(OUT, N * 32);
    let mut bytes = [0u8; OUT];
    for (index, value) in fields.into_iter().enumerate() {
        bytes[index * 32..(index + 1) * 32].copy_from_slice(&value);
    }
    bytes
}

fn field<const N: usize>(bytes: &[u8; N], index: usize) -> [u8; 32] {
    bytes[index * 32..(index + 1) * 32]
        .try_into()
        .expect("fixed dense record field")
}

fn exact_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], RecordDecodeError> {
    bytes
        .try_into()
        .map_err(|_| RecordDecodeError::InvalidLength {
            expected: N,
            actual: bytes.len(),
        })
}

fn tagged_bytes(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1 + payload.len());
    bytes.push(kind);
    bytes.extend_from_slice(payload);
    bytes
}

fn commit_transcript(
    public_key: Inline<ED25519PublicKey>,
    collection: CollectionHandle,
    data_hash: CollectionData,
    metadata: Inline<Handle<SimpleArchive>>,
) -> [u8; COMMIT_TRANSCRIPT_LEN] {
    let mut transcript = [0; COMMIT_TRANSCRIPT_LEN];
    let mut offset = 0;
    let mut append = |bytes: &[u8]| {
        let end = offset + bytes.len();
        transcript[offset..end].copy_from_slice(bytes);
        offset = end;
    };
    append(COMMIT_TRANSCRIPT_DOMAIN);
    append(&KIND_COLLECTION_COMMIT.raw());
    append(&COMMIT_TRANSCRIPT_VERSION.to_be_bytes());
    append(&public_key.raw);
    append(&collection.raw);
    append(&data_hash.raw);
    append(&metadata.raw);
    debug_assert_eq!(offset, COMMIT_TRANSCRIPT_LEN);
    transcript
}

fn encode_archive(facts: TribleSet) -> Blob<SimpleArchive> {
    <TribleSet as crate::blob::IntoBlob<SimpleArchive>>::to_blob(facts)
}

fn decode_archive(blob: &Blob<SimpleArchive>) -> Result<TribleSet, RecordDecodeError> {
    Ok(<TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(
        blob.clone(),
    )?)
}

fn record_root_and_kind(facts: &TribleSet, expected: Id) -> Result<Id, RecordDecodeError> {
    let mut iter = facts.iter();
    let Some(first) = iter.next() else {
        return Err(RecordDecodeError::Empty);
    };
    let root = *first.e();
    if iter.any(|fact| fact.e() != &root) {
        return Err(RecordDecodeError::MultipleEntities);
    }
    let actual = one_id(facts, &metadata::tag, "metadata::tag")?;
    if actual != expected {
        return Err(RecordDecodeError::WrongKind { expected, actual });
    }
    Ok(root)
}

#[cfg(test)]
pub(crate) fn one_id_for_test(facts: &TribleSet, attribute: &Attribute<GenId>) -> Id {
    one_id(facts, attribute, "test").expect("present")
}

fn one_id(
    facts: &TribleSet,
    attribute: &Attribute<GenId>,
    field: &'static str,
) -> Result<Id, RecordDecodeError> {
    let value: Inline<GenId> = one_inline(facts, attribute, field)?;
    value
        .try_from_inline::<Id>()
        .map_err(|_: IdParseError| RecordDecodeError::InvalidId(field))
}

fn one_inline<S: InlineEncoding>(
    facts: &TribleSet,
    attribute: &Attribute<S>,
    field: &'static str,
) -> Result<Inline<S>, RecordDecodeError> {
    let mut values = facts
        .iter()
        .filter(|fact| fact.a() == &attribute.id())
        .map(|fact| *fact.v::<S>());
    let Some(value) = values.next() else {
        return Err(RecordDecodeError::MissingField(field));
    };
    if values.next().is_some() {
        return Err(RecordDecodeError::RepeatedField(field));
    }
    Ok(value)
}

fn ensure_canonical(
    stored_facts: &TribleSet,
    stored_root: Id,
    expected_root: Id,
    expected_facts: TribleSet,
) -> Result<(), RecordDecodeError> {
    if stored_root != expected_root {
        return Err(RecordDecodeError::NonCanonicalRoot {
            stored: stored_root,
            expected: expected_root,
        });
    }
    let exact = stored_facts.len() == expected_facts.len()
        && stored_facts
            .eav
            .iter_ordered()
            .eq(expected_facts.eav.iter_ordered());
    if !exact {
        return Err(RecordDecodeError::NonCanonicalFacts);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use hex_literal::hex;

    use crate::id::Id;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn hash(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    fn collection(byte: u8) -> CollectionHandle {
        Inline::new([byte; 32])
    }

    fn fixture_key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    #[test]
    fn collection_descriptor_is_scope_specific_and_roundtrips() {
        let a = CollectionDescriptor::naming(id(1), id(2), id(3));
        let b = CollectionDescriptor::naming(id(4), id(2), id(3));
        let c = CollectionDescriptor::naming(id(1), id(4), id(3));
        let d = CollectionDescriptor::naming(id(1), id(2), id(4));
        assert_ne!(a.handle(), b.handle());
        assert_ne!(a.handle(), c.handle());
        assert_ne!(a.handle(), d.handle());
        assert_eq!(CollectionDescriptor::decode(&a.to_blob()).unwrap(), a);
        let root = a.entity_id().unwrap();
        assert!(a.to_tribles().iter().all(|fact| fact.e() == &root));
    }

    #[test]
    fn malformed_archive_is_a_structural_error() {
        let malformed: Blob<SimpleArchive> = Blob::new(vec![0].into());
        assert_eq!(
            CollectionDescriptor::decode(&malformed),
            Err(RecordDecodeError::Archive(UnarchiveError::BadArchive))
        );
    }

    #[test]
    fn empty_metadata_is_the_canonical_empty_archive() {
        let empty = encode_archive(TribleSet::new());
        assert_eq!(empty_metadata_handle(), empty.get_handle());
        assert!(empty.bytes.is_empty());
    }

    #[test]
    fn signed_commit_decodes_before_it_verifies_and_retries_identically() {
        let key = fixture_key();
        let first = CollectionCommit::sign(&key, collection(1), hash(2), empty_metadata_handle());
        let retry = CollectionCommit::sign(&key, collection(1), hash(2), empty_metadata_handle());
        assert_eq!(first, retry);
        assert_eq!(first.to_bytes(), retry.to_bytes());
        assert_eq!(CollectionCommit::from_bytes(first.to_bytes()), first);
        first.verify_strict().unwrap();

        let mut bad_s = first.signature_s;
        bad_s.raw[0] ^= 1;
        let bad = CollectionCommit::from_parts(
            first.collection,
            first.data,
            first.metadata,
            first.public_key,
            first.signature_r,
            bad_s,
        );
        let decoded = CollectionCommit::from_bytes(bad.to_bytes());
        assert_eq!(
            decoded.verify_strict(),
            Err(CommitVerificationError::InvalidSignature)
        );

        let mut bad_r = first.signature_r;
        bad_r.raw[0] ^= 1;
        let bad = CollectionCommit::from_parts(
            first.collection,
            first.data,
            first.metadata,
            first.public_key,
            bad_r,
            first.signature_s,
        );
        assert_eq!(
            bad.verify_strict(),
            Err(CommitVerificationError::InvalidSignature)
        );

        let mut invalid_key = [0; 32];
        invalid_key[0] = 2;
        let invalid_key = CollectionCommit::from_parts(
            first.collection,
            first.data,
            first.metadata,
            Inline::new(invalid_key),
            first.signature_r,
            first.signature_s,
        );
        let decoded = CollectionCommit::from_bytes(invalid_key.to_bytes());
        assert_eq!(
            decoded.verify_strict(),
            Err(CommitVerificationError::InvalidPublicKey)
        );
    }

    #[test]
    fn every_signed_field_is_bound_by_the_transcript() {
        let valid =
            CollectionCommit::sign(&fixture_key(), collection(1), hash(2), Inline::new([3; 32]));
        valid.verify_strict().unwrap();

        let mut alterations = Vec::new();
        alterations.push(CollectionCommit::from_parts(
            collection(9),
            valid.data,
            valid.metadata,
            valid.public_key,
            valid.signature_r,
            valid.signature_s,
        ));
        alterations.push(CollectionCommit::from_parts(
            valid.collection,
            hash(9),
            valid.metadata,
            valid.public_key,
            valid.signature_r,
            valid.signature_s,
        ));
        alterations.push(CollectionCommit::from_parts(
            valid.collection,
            valid.data,
            Inline::new([9; 32]),
            valid.public_key,
            valid.signature_r,
            valid.signature_s,
        ));
        let mut public_key = valid.public_key;
        public_key.raw[0] ^= 1;
        alterations.push(CollectionCommit::from_parts(
            valid.collection,
            valid.data,
            valid.metadata,
            public_key,
            valid.signature_r,
            valid.signature_s,
        ));

        assert!(alterations
            .iter()
            .all(|altered| altered.verify_strict().is_err()));
    }

    #[test]
    fn merge_is_commutative_in_dense_encoding() {
        let forward = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));
        let reverse = CollectionMerge::new(collection(1), hash(3), hash(2), hash(4));
        assert_eq!(forward, reverse);
        assert_eq!(forward.to_bytes(), reverse.to_bytes());
        assert_eq!(
            CollectionMerge::from_bytes(forward.to_bytes()).unwrap(),
            forward
        );
    }

    #[test]
    fn derive_roundtrips() {
        let record = CollectionDerive::new(collection(2), hash(3), hash(4));
        assert_eq!(CollectionDerive::from_bytes(record.to_bytes()), record);
    }

    #[test]
    fn generic_codec_tags_each_variant() {
        let commit = CollectionCommit::sign(
            &fixture_key(),
            collection(1),
            hash(2),
            empty_metadata_handle(),
        );
        let merge = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));
        let derive = CollectionDerive::new(collection(2), hash(3), hash(4));
        for record in [
            CollectionRecord::Commit(commit),
            CollectionRecord::Merge(merge),
            CollectionRecord::Derive(derive),
        ] {
            assert_eq!(
                CollectionRecord::from_bytes(&record.to_bytes()).unwrap(),
                record
            );
        }
        assert_eq!(
            CollectionRecord::from_bytes(&[99]),
            Err(RecordDecodeError::UnknownKind(99))
        );
    }

    #[test]
    fn generic_codec_rejects_wrong_lengths() {
        assert_eq!(
            CollectionRecord::from_bytes(&[COLLECTION_RECORD_KIND_COMMIT_V1]),
            Err(RecordDecodeError::InvalidLength {
                expected: COLLECTION_COMMIT_BYTES_LEN,
                actual: 0,
            })
        );
    }

    #[test]
    fn merge_decoder_rejects_noncanonical_input_order() {
        let record = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));
        let mut bytes = record.to_bytes();
        bytes[32..64].fill(9);
        bytes[64..96].fill(1);
        assert_eq!(
            CollectionMerge::from_bytes(bytes),
            Err(RecordDecodeError::NonCanonicalMergeInputs)
        );
    }

    #[test]
    fn transcript_and_record_roots_are_golden() {
        let descriptor = CollectionDescriptor::naming(id(1), id(2), id(3));
        let commit =
            CollectionCommit::sign(&fixture_key(), collection(1), hash(2), Inline::new([3; 32]));
        let merge = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));
        let derive = CollectionDerive::new(collection(2), hash(3), hash(4));

        // Descriptor wire bytes are unchanged by the identity cutover.
        assert_eq!(
            descriptor.entity_id().unwrap(),
            id_hex!("D28DF8A2FAAABEDCD2943FD73920EECD")
        );
        assert_eq!(
            descriptor.handle().raw,
            hex!("51F16FDE006E9A38C68B939A20B4255BC049C1795597BF05D499634AF3CCAA9F")
        );
        assert_eq!(
            descriptor.to_blob().bytes.len() as u64,
            COLLECTION_DESCRIPTOR_ARCHIVE_LEN
        );
        assert_eq!(commit.to_bytes().len(), COLLECTION_COMMIT_BYTES_LEN);
        assert_eq!(merge.to_bytes().len(), COLLECTION_MERGE_BYTES_LEN);
        assert_eq!(derive.to_bytes().len(), COLLECTION_DERIVE_BYTES_LEN);

        assert_eq!(commit.signing_transcript().len(), COMMIT_TRANSCRIPT_LEN);
        assert_eq!(commit.id(), id_hex!("21FE95F313A7AADD236286EE83B5AA39"));
        assert_eq!(
            commit.signature_r.raw,
            hex!("F89FCF5C72BC7EC3E376C6AB6BDEFC6ECEA3ADBBCA7A36DBF1729413A7820564")
        );
        assert_eq!(
            commit.signature_s.raw,
            hex!("F684108AF3E8E3898904D20EA458DCAE68F0F97F4E5C06DAFA0FAE0691F68D0B")
        );
        assert_eq!(merge.id(), id_hex!("032390A36A86A2F5A44604B78EF6FA8C"));
        // The derive id moved once, when the record stopped naming its source
        // and gained a new kind: the id is a digest over the kind and payload,
        // and both changed. Commit and merge ids are untouched.
        assert_eq!(derive.id(), id_hex!("5F7EF9C1C56832B2F098486612592ACD"));
        assert_eq!(
            commit.signing_transcript(),
            hex!(
                "747269626C6573706163652E636F6C6C656374696F6E2E636F6D6D69742E7472616E736372697074
                 B34817308188C4515A3C51967A91A603
                 00000002
                 EA4A6C63E29C520ABEF5507B132EC5F9954776AEBEBE7B92421EEA691446D22C
                 0101010101010101010101010101010101010101010101010101010101010101
                 0202020202020202020202020202020202020202020202020202020202020202
                 0303030303030303030303030303030303030303030303030303030303030303"
            )
            .to_vec()
        );
        commit.verify_strict().unwrap();
    }
}

#[cfg(test)]
mod recipe_description_tests {
    use crate::collection::observed_union::ObservedUnionV1;
    use crate::collection::simplearchive_union::TribleSetUnionV1;
    use crate::collection::succinctarchive_union::{
        Rank9LiftedUnionV1_32Be, Rank9LiftedUnionV1_32Le, Rank9LiftedUnionV1_64Be,
        Rank9LiftedUnionV1_64Le,
    };
    use crate::metadata::{self, MetaDescribe};
    use crate::query::register::StatedOrderV1;

    /// Every law describes itself, and the description is rooted at the id the
    /// law was already minted under. A descriptor can therefore embed the
    /// description without changing which law it names.
    #[test]
    fn every_recipe_describes_itself_under_its_own_id() {
        fn check<L: MetaDescribe>(expected: crate::id::Id, name: &str) {
            let fragment = <L as MetaDescribe>::describe();
            assert_eq!(
                <L as MetaDescribe>::id(),
                expected,
                "{name} describes itself under a different id than it was minted with"
            );
            let facts = fragment.facts();
            let kind = crate::collection::records::one_id_for_test(&facts, &metadata::tag);
            assert_eq!(
                kind,
                metadata::KIND_COLLECTION_RECIPE,
                "{name} is not tagged as a collection recipe"
            );
        }
        check::<TribleSetUnionV1>(
            crate::collection::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1,
            "trible-set-union-v1",
        );
        check::<ObservedUnionV1>(
            crate::collection::observed_union::OBSERVED_UNION_RECIPE_V1,
            "observed-union-v1",
        );
        check::<StatedOrderV1>(crate::query::register::STATED_ORDER_RECIPE_V1, "stated-order-v1");
        check::<Rank9LiftedUnionV1_32Le>(
            crate::collection::succinctarchive_union::RANK9_LIFTED_UNION_RECIPE_V1_32_LE,
            "rank9-32-le",
        );
        check::<Rank9LiftedUnionV1_32Be>(
            crate::collection::succinctarchive_union::RANK9_LIFTED_UNION_RECIPE_V1_32_BE,
            "rank9-32-be",
        );
        check::<Rank9LiftedUnionV1_64Le>(
            crate::collection::succinctarchive_union::RANK9_LIFTED_UNION_RECIPE_V1_64_LE,
            "rank9-64-le",
        );
        check::<Rank9LiftedUnionV1_64Be>(
            crate::collection::succinctarchive_union::RANK9_LIFTED_UNION_RECIPE_V1_64_BE,
            "rank9-64-be",
        );
    }
}
