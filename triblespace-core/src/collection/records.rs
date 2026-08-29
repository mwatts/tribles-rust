//! Dense typed records for the top-level collection calculus.
//!
//! [`CollectionCommit`], [`CollectionMerge`], and [`CollectionDerive`] are
//! native algebra records, not graph data. Their canonical representations are
//! the fixed-width byte layouts exposed by their `to_bytes`/`from_bytes`
//! methods. A collection *descriptor* is not a record at all: it is an
//! ordinary [`TribleSet`] stored as a self-describing [`SimpleArchive`], and
//! that blob's handle is the collection identity. See
//! [`descriptor`](crate::collection::descriptor) for reading one.
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

#[cfg(test)]
use crate::attribute::Attribute;
use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::encodings::utf8string::UTF8String;
use crate::blob::Blob;
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::ed25519::{ED25519PublicKey, ED25519RComponent, ED25519SComponent};
use crate::inline::encodings::genid::GenId;
#[cfg(test)]
use crate::inline::encodings::genid::IdParseError;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::Inline;
#[cfg(test)]
use crate::inline::InlineEncoding;
use crate::prelude::attributes;
use crate::trible::{TribleSet, TRIBLE_LEN};

/// Tag identifying a canonical collection descriptor.
///
/// Minted with `trible genid` on 2026-08-07.
pub const KIND_COLLECTION_DESCRIPTOR: Id = id_hex!("C5E238729BB95FA4A55E3939B11B3C29");
/// Tag identifying a concrete collection mapping instance.
///
/// Minted with `trible genid` on 2026-08-29.
pub const KIND_COLLECTION_MAPPING: Id = id_hex!("8725AF624FE719B70289A67B21174FEA");
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

/// Retired semantic kind of a signed collection-gossip grant.
///
/// A grant was an author-signed, irrevocable permission to redistribute that
/// author's commits in one collection. It is gone because reach moved into the
/// descriptor: committing into a collection whose identity says it travels
/// *is* the consent, and cannot be given by accident, since a collection that
/// stays put is a different collection with a different handle. The grant only
/// ever restated what the descriptor now declares.
///
/// No pile has ever held one -- 21.2 GB across six piles were scanned for the
/// record marker before the kind was removed, and the same scan found commit
/// records, so the absence was the grant's and not the scan's. Nothing in
/// production minted them: `sign` appeared only in tests, which is exactly the
/// silent failure the move to the descriptor removes.
///
/// Minted with `trible genid` on 2026-08-12, retired 2026-08-21. Kept here so
/// the id is not minted twice.
pub const KIND_COLLECTION_GOSSIP_V1: Id = id_hex!("9BB5B1F4D6FD8FB850B494C2CF51B5CA");

/// Byte length of a canonical bare root collection-descriptor `SimpleArchive`.
///
/// Four facts: the kind tag, name, mandatory authority, and encoding. A
/// descriptor that embeds its encoding's description or declares reach is
/// longer. A derived descriptor additionally embeds one mapping instance. The
/// name's UTF-8 bytes are a separate attachment and do not contribute rows to
/// the archive.
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
    /// The human-readable name of a root collection.
    ///
    /// The UTF-8 bytes live in an attachment so the identity has no arbitrary
    /// 32-byte ceiling. The collection's mandatory authority provides the
    /// globally unique scope; the name is descriptive identity within it, not
    /// a second authorization mechanism.
    ///
    /// Anchor minted with `trible genid` on 2026-08-28:
    /// `A2EEF06D4E1AA4B17B745AA2E8C37867`.
    "A2EEF06D4E1AA4B17B745AA2E8C37867" as pub collection_name: Handle<UTF8String>;
    /// External capability trust root for this exact collection.
    ///
    /// Every descriptor names exactly one authority directly. It is not
    /// inherited through [`collection_source`], and the authority key itself
    /// is the sovereign signer while delegates prove exact collection rights.
    ///
    /// Anchor minted with `trible genid` on 2026-08-24:
    /// `7C31D328E9C369CCB6049D05CC8E8C77`.
    "7C31D328E9C369CCB6049D05CC8E8C77" as pub collection_authority: ED25519PublicKey;
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
    /// Concrete mapping instance carried by a derived collection.
    ///
    /// The linked entity names one mapping algorithm and carries its concrete
    /// parameters. Root collections omit this field; derived collections carry
    /// it together with [`collection_source`].
    /// Anchor minted with `trible genid` on 2026-08-29:
    /// `D63BCA62411CA97CCF5DB2132EE57CDF`.
    "D63BCA62411CA97CCF5DB2132EE57CDF" as pub collection_mapping: GenId;
    /// Algorithm used by one concrete collection mapping instance.
    ///
    /// The algorithm entity is embedded through Fragment spread, keeping the
    /// descriptor self-describing. Canonical builders derive the mapping id
    /// from its concrete parameters, while readers validate these facts
    /// directly so equivalent extrinsic ids remain valid.
    /// Anchor minted with `trible genid` on 2026-08-29:
    /// `AF43D02D710977411CA7DA164BDA5CD9`.
    "AF43D02D710977411CA7DA164BDA5CD9" as pub mapping_algorithm: GenId;
    /// How far this collection may travel, by the id of a *reach law*.
    ///
    /// This is the descriptor's answer to "may these bytes be relayed?", and
    /// putting it here is what makes the answer unforgeable. Reach used to be
    /// a separately signed record any keyholder could mint later, so "private"
    /// meant only "nobody has signed one yet". Named in the descriptor, reach
    /// is part of the collection's identity: a private collection and a public
    /// one are *different collections* with different handles, and publishing
    /// something after the fact is not forbidden but meaningless, because it
    /// would have to name a handle whose own descriptor refuses to travel.
    ///
    /// This names a *law* rather than a value. A
    /// law admits an "I do not implement that" answer, which a boolean cannot:
    /// a reader meeting a mode it has never heard of denies rather than
    /// guesses, so the fail-closed rule extends to modes invented after this
    /// binary was built. Any arguments a richer mode needs -- an audience, a
    /// subset of a team -- are further attributes on this same entity, read
    /// through [`descriptor::argument`](crate::collection::descriptor::argument).
    ///
    /// The attribute is optional, and absence means *no reach*: a descriptor
    /// that predates this field, or declines to declare, does not travel. That
    /// direction is not a default chosen for convenience -- the opposite
    /// default would silently publish every collection written before reach
    /// existed, which is the exact hazard this field removes.
    ///
    /// Anchor minted with `trible genid` on 2026-08-21. Declared with the
    /// anchored `as` form rather than the pinned `unsafe as` form used by its
    /// neighbours: those preserve byte identities already written into piles,
    /// whereas no row has ever carried this attribute, so its encoding is free
    /// to participate in its identity.
    "7CCF99CCE4657117EE8CDD1B8E11FDA3" as pub collection_reach: GenId;
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
    /// A required field was absent.
    MissingField(&'static str),
    /// A single-valued field occurred more than once.
    RepeatedField(&'static str),
    /// A descriptor field occurred on an entity other than the tagged
    /// descriptor entity.
    FieldOnWrongEntity(&'static str),
    /// A typed descriptor field had a noncanonical inline representation.
    InvalidId(&'static str),
    /// A retired descriptor field would be ambiguous under the current model.
    ObsoleteField(&'static str),
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
            Self::MissingField(field) => write!(f, "collection record is missing {field}"),
            Self::RepeatedField(field) => {
                write!(f, "collection record contains repeated {field}")
            }
            Self::FieldOnWrongEntity(field) => {
                write!(f, "collection record contains {field} on another entity")
            }
            Self::InvalidId(field) => write!(f, "collection record contains invalid {field}"),
            Self::ObsoleteField(field) => {
                write!(f, "collection record contains obsolete {field}")
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

/// One unsigned exact observation of the canonical mapping from a source
/// collection member to a target collection member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionDerive {
    id: Id,
    collection: CollectionHandle,
    input: CollectionData,
    output: CollectionData,
}

impl CollectionDerive {
    /// Construct a canonical `DERIVE(collection, input, output)` record.
    ///
    /// The output collection is named by descriptor handle, exactly as a commit names its
    /// collection, and that descriptor already says which collection is the
    /// source and embeds the exact mapping facts. A derive therefore
    /// says *which input/output equation* was computed, never restates the
    /// mapping definition.
    pub fn new(
        collection: CollectionHandle,
        input: CollectionData,
        output: CollectionData,
    ) -> Self {
        let bytes = derive_bytes(collection, input, output);
        let id = collection_record_id(KIND_COLLECTION_DERIVE, &bytes);
        Self {
            id,
            collection,
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

    /// Collection containing the output member.
    pub fn collection(&self) -> CollectionHandle {
        self.collection
    }

    /// Source member mapped by this equation.
    pub fn input(&self) -> CollectionData {
        self.input
    }

    /// Target member produced by this equation.
    pub fn output(&self) -> CollectionData {
        self.output
    }

    /// Encode this equation into its exact dense 96-byte layout.
    pub fn to_bytes(&self) -> [u8; COLLECTION_DERIVE_BYTES_LEN] {
        derive_bytes(self.collection, self.input, self.output)
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

#[cfg(test)]
pub(crate) fn one_id_for_test(facts: &TribleSet, attribute: &Attribute<GenId>) -> Id {
    one_id(facts, attribute, "test").expect("present")
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::reach;

    use hex_literal::hex;

    use crate::blob::TryFromBlob;
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

    /// A root is named under one mandatory authority. Name, authority,
    /// representation, and reach all participate in descriptor identity.
    #[test]
    fn collection_descriptor_is_anchor_specific_and_roundtrips() {
        use crate::collection::descriptor;

        let authority = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let other_authority = SigningKey::from_bytes(&[2; 32]).verifying_key();

        let a_fragment = descriptor::naming("first", authority, id(2), reach::private());
        let expected_name = descriptor::name(a_fragment.facts()).unwrap().unwrap();
        let a = a_fragment.into_facts();
        let renamed = descriptor::naming("second", authority, id(2), reach::private()).into_facts();
        let reauthorized =
            descriptor::naming("first", other_authority, id(2), reach::private()).into_facts();
        let other_representation =
            descriptor::naming("first", authority, id(4), reach::private()).into_facts();
        let other_reach =
            descriptor::naming("first", authority, id(2), reach::public()).into_facts();

        let handle = |facts: &TribleSet| {
            <TribleSet as crate::blob::IntoBlob<SimpleArchive>>::to_blob(facts.clone()).get_handle()
        };
        assert_ne!(handle(&a), handle(&renamed));
        assert_ne!(handle(&a), handle(&reauthorized));
        assert_ne!(handle(&a), handle(&other_representation));
        assert_ne!(handle(&a), handle(&other_reach));

        // The descriptor is its own archive: encoding and decoding is the
        // identity, because there is no second model of it to drift.
        let blob = <TribleSet as crate::blob::IntoBlob<SimpleArchive>>::to_blob(a.clone());
        assert_eq!(
            <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob).unwrap(),
            a
        );
        let root = descriptor::entity(&a).unwrap();
        assert!(a.iter().all(|fact| fact.e() == &root));

        assert_eq!(
            descriptor::name(&a).unwrap().unwrap(),
            expected_name,
            "the anchor reads back as what was written"
        );
        assert_eq!(descriptor::authority(&a), Ok(authority));
    }

    #[test]
    fn malformed_archive_is_a_structural_error() {
        let malformed: Blob<SimpleArchive> = Blob::new(vec![0].into());
        assert_eq!(
            <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(malformed)
                .map_err(RecordDecodeError::from),
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
        let collection_name_anchor = id_hex!("A2EEF06D4E1AA4B17B745AA2E8C37867");
        assert_eq!(
            collection_name.raw(),
            Attribute::<Handle<UTF8String>>::anchored(collection_name_anchor).raw(),
            "the UTF-8 name attribute is derived from the minted anchor and encoding"
        );
        assert_ne!(
            collection_name.id(),
            collection_name_anchor,
            "the anchored form must not pin the anchor as the attribute id"
        );
        let descriptor = crate::collection::descriptor::naming(
            "first",
            SigningKey::from_bytes(&[1; 32]).verifying_key(),
            id(2),
            reach::private(),
        )
        .into_facts();
        let descriptor_blob =
            <TribleSet as crate::blob::IntoBlob<SimpleArchive>>::to_blob(descriptor.clone());
        let commit =
            CollectionCommit::sign(&fixture_key(), collection(1), hash(2), Inline::new([3; 32]));
        let merge = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));
        let derive = CollectionDerive::new(collection(2), hash(3), hash(4));

        // Pin the bare-root row count. The descriptor/entity/blob identities
        // deliberately moved in this epoch: namespace was removed, authority
        // became mandatory, and the name became a UTF-8 blob handle.
        assert_eq!(
            descriptor_blob.bytes.len() as u64,
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
mod mapping_algorithm_description_tests {
    use crate::blob::encodings::utf8string::UTF8String;
    use crate::blob::IntoBlob;
    use crate::collection::lww_register::RegisterCoordinatesMappingV1;
    use crate::collection::observed_union::ObserveStatesMappingV1;
    use crate::collection::succinctarchive_union::{
        RawToRank9AcceleratedMappingV1, SimpleToSuccinctMappingV1,
    };
    use crate::inline::encodings::hash::Handle;
    use crate::inline::Inline;
    use crate::metadata::{self, MetaDescribe};

    /// Every mapping algorithm describes itself under its minted identity.
    /// Concrete, parameterized mapping entities embed these descriptions
    /// without conflating the algorithm with one particular mapping.
    #[test]
    fn every_mapping_algorithm_describes_itself_under_its_own_id() {
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
                metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
                "{name} is not tagged as a collection mapping algorithm"
            );
            let actual_name = super::one_inline(facts, &metadata::name, "name")
                .expect("mapping algorithm has one name");
            let expected_name: Inline<Handle<UTF8String>> = name.to_owned().to_blob().get_handle();
            assert_eq!(
                actual_name, expected_name,
                "mapping algorithm description retained a stale name"
            );
        }
        check::<SimpleToSuccinctMappingV1>(
            crate::collection::succinctarchive_union::SIMPLE_TO_SUCCINCT_MAPPING_V1,
            "simple-to-succinct-v1",
        );
        check::<ObserveStatesMappingV1>(
            crate::collection::observed_union::OBSERVE_STATES_MAPPING_V1,
            "observe-states-v1",
        );
        check::<RegisterCoordinatesMappingV1>(
            crate::collection::lww_register::REGISTER_COORDINATES_MAPPING_V1,
            "register-coordinates-v1",
        );
        check::<RawToRank9AcceleratedMappingV1>(
            crate::collection::succinctarchive_union::current_rank9_accelerated_mapping_algorithm(),
            "raw-to-rank9-accelerated-succinctarchive-v1",
        );
    }
}
