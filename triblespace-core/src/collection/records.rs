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
use crate::blob::Blob;
use crate::id::Id;
use crate::id_hex;
use crate::inline::encodings::ed25519::{ED25519PublicKey, ED25519RComponent, ED25519SComponent};
use crate::inline::encodings::genid::GenId;
#[cfg(test)]
use crate::inline::encodings::genid::IdParseError;
use crate::inline::encodings::hash::{Blake3, Handle, Hash};
use crate::inline::encodings::shortstring::ShortString;
use crate::inline::Inline;
#[cfg(test)]
use crate::inline::InlineEncoding;
use crate::prelude::attributes;
use crate::trible::{TribleSet, TRIBLE_LEN};

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

/// Stable semantic kind of a signed commutative `MERGE` equation.
///
/// The equation is the same one [`KIND_COLLECTION_MERGE`] states. The
/// signature is evidence about who asserted it, carried so a reader that
/// already trusts that asserter may admit the equation without recomputing
/// it; it never makes the equation true, and a reader that recomputes needs
/// none of it.
///
/// Minted with `trible genid` on 2026-08-21.
pub const KIND_COLLECTION_SIGNED_MERGE: Id = id_hex!("84E166592582DAF1DC2B966A8FD9B71A");

/// Stable semantic kind of a signed `DERIVE` equation.
///
/// The signed counterpart of [`KIND_COLLECTION_DERIVE`], on the same terms as
/// [`KIND_COLLECTION_SIGNED_MERGE`].
///
/// Minted with `trible genid` on 2026-08-21.
pub const KIND_COLLECTION_SIGNED_DERIVE: Id = id_hex!("B2518513A56D87F90E15BD0B7508C9ED");

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

/// Byte length of a canonical bare root collection-descriptor `SimpleArchive`.
///
/// Five facts: the kind tag, the name and team that anchor the root, and the
/// representation and recipe it names. A descriptor that embeds its
/// representation's and recipe's own descriptions, or that carries recipe
/// arguments, is longer.
pub const COLLECTION_DESCRIPTOR_ARCHIVE_LEN: u64 = (5 * TRIBLE_LEN) as u64;
/// Byte length of a dense signed commit.
pub const COLLECTION_COMMIT_BYTES_LEN: usize = 6 * 32;
/// Byte length of a dense merge equation.
pub const COLLECTION_MERGE_BYTES_LEN: usize = 4 * 32;
/// Byte length of a dense derive equation.
pub const COLLECTION_DERIVE_BYTES_LEN: usize = 3 * 32;
/// Byte length of a dense signed merge equation.
///
/// The four equation fields plus a public key and the two signature
/// components: 224 bytes, which is 32 more than the 192 a 256-byte pile block
/// leaves after its frame, so a stored signed merge spans two blocks where
/// every other collection record spans one.
///
/// That overrun is worth paying, and the asymmetry with
/// [`COLLECTION_SIGNED_DERIVE_BYTES_LEN`] is not an oversight. It would be
/// tempting to sign derives only, on the theory that a merge is a cheap
/// re-check -- and for a `SimpleArchive` collection it is, a linear pass over
/// two sorted archives. But
/// [`succinctarchive_union::join`](crate::collection::succinctarchive_union::join)
/// is `SuccinctArchiveBlob::merge`, a full rebuild of the wavelet and rank
/// structures, so in the one collection kind whose recomputation cost started
/// this design a merge is at least as expensive as a derive. Signing derives
/// only would have excluded exactly the LSM cover that makes a cold read fast.
///
/// The padding is also not a real cost. A signature is only ever worth adding
/// where recomputing costs more than verifying one, which is never true of a
/// cheap merge, so the second block lands only on records that stand for
/// something large. Scanning the live piles on 2026-08-21 for the record-kind
/// markers found no merge record at all, in any generation of the format.
pub const COLLECTION_SIGNED_MERGE_BYTES_LEN: usize = COLLECTION_MERGE_BYTES_LEN + 3 * 32;
/// Byte length of a dense signed derive equation.
///
/// Three equation fields plus a signature is `6 * 32`, exactly the shape of a
/// [`CollectionCommit`].
pub const COLLECTION_SIGNED_DERIVE_BYTES_LEN: usize = COLLECTION_DERIVE_BYTES_LEN + 3 * 32;

/// Version of collection-record identity derivation.
pub const COLLECTION_RECORD_ID_VERSION: u32 = 1;

/// Domain prefix of collection-record identity derivation.
pub const COLLECTION_RECORD_ID_DOMAIN: &[u8] = b"triblespace.collection.record.id";

attributes! {
    /// The name a *root* collection is known by within its team.
    ///
    /// Half of a root's anchor; see [`collection_team`] for the other half.
    /// Together they replaced an opaque minted scope id, which discriminated
    /// roots correctly but told a reader nothing: every faculty carried its
    /// scope as a hex constant in its own source, so "which collection is
    /// this?" was answerable only by someone holding the code.
    ///
    /// The name is part of the identity, so it does not change. A rename is a
    /// new collection, reached by deriving from the old one. Mutable labels
    /// are ordinary facts published *about* a collection and are free to
    /// disagree; this one is the address.
    ///
    /// Minted with `trible genid` on 2026-08-20.
    "436A04C372CBBFBD9C619CF50F59C4A1" unsafe as pub collection_name: ShortString;
    /// Root public key of the team a *root* collection belongs to.
    ///
    /// The other half of a root's anchor. A team has one immutable root
    /// keypair, generated at team creation and thereafter archived offline, so
    /// it is a genesis fact rather than an operational one and can be part of
    /// an identity without going stale. Membership evolves as capabilities
    /// delegated beneath it.
    ///
    /// Naming it here is what lets a collection authorize itself: a reader
    /// holding this descriptor and the pile can walk cap chains from this root
    /// and decide which commits may count, instead of being handed an
    /// authorized set out of band. A pile with no team uses its own node key,
    /// which is a team of one.
    ///
    /// Minted with `trible genid` on 2026-08-20.
    "6C1ED6495491E32FEBB9FDD4EE5E8907" unsafe as pub collection_team: ED25519PublicKey;
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

/// Version of the signed collection-equation transcript.
pub const EQUATION_TRANSCRIPT_VERSION: u32 = 1;

/// Domain prefix of the signed collection-equation transcript.
pub const EQUATION_TRANSCRIPT_DOMAIN: &[u8] = b"triblespace.collection.equation.transcript";

/// Number of bytes in a version-1 signed `MERGE` transcript.
pub const MERGE_TRANSCRIPT_LEN: usize = EQUATION_TRANSCRIPT_DOMAIN.len()
    + 16 // kind id
    + 4 // version
    + 32 // public key
    + 32 // collection descriptor handle
    + 32 // low input digest
    + 32 // high input digest
    + 32; // result digest

/// Number of bytes in a version-1 signed `DERIVE` transcript.
pub const DERIVE_TRANSCRIPT_LEN: usize = EQUATION_TRANSCRIPT_DOMAIN.len()
    + 16 // kind id
    + 4 // version
    + 32 // public key
    + 32 // target descriptor handle
    + 32 // input digest
    + 32; // output digest

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
    /// A `GenId` field had a noncanonical or nil inline representation.
    InvalidId(&'static str),
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
            Self::InvalidId(field) => write!(f, "collection record contains invalid {field}"),
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

/// Semantic verification failure for a signed collection record.
///
/// Shared by the three record kinds that can carry a signature: a commit,
/// which is always signed, and a merge or derive, which may be.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignatureVerificationError {
    /// The public-key bytes do not encode an Ed25519 verifying key.
    InvalidPublicKey,
    /// Strict Ed25519 verification rejected the transcript/signature pair.
    InvalidSignature,
}

impl fmt::Display for SignatureVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPublicKey => write!(f, "collection record has an invalid public key"),
            Self::InvalidSignature => write!(f, "collection record signature is invalid"),
        }
    }
}

impl Error for SignatureVerificationError {}

/// The Ed25519 evidence a signed collection record carries.
///
/// A signature is never part of the statement a record makes. A commit's
/// statement is *authority* -- "this key admits this element" -- and so it is
/// meaningless unsigned. A merge or derive states an *equation*, which is true
/// or false on its own, and its signature is evidence about who asserted it:
/// grounds for a reader that trusts the signer to skip recomputing the
/// equation, never grounds for the equation to hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionSignature {
    public_key: Inline<ED25519PublicKey>,
    signature_r: Inline<ED25519RComponent>,
    signature_s: Inline<ED25519SComponent>,
}

impl CollectionSignature {
    /// Assemble the three raw fields without verifying them.
    pub fn from_parts(
        public_key: Inline<ED25519PublicKey>,
        signature_r: Inline<ED25519RComponent>,
        signature_s: Inline<ED25519SComponent>,
    ) -> Self {
        Self {
            public_key,
            signature_r,
            signature_s,
        }
    }

    /// Raw public-key field. It becomes trusted only after strict verification.
    pub fn public_key(&self) -> Inline<ED25519PublicKey> {
        self.public_key
    }

    /// Raw signature components.
    pub fn components(&self) -> (Inline<ED25519RComponent>, Inline<ED25519SComponent>) {
        (self.signature_r, self.signature_s)
    }

    /// Strictly verify this signature over an already-built transcript.
    ///
    /// Returns the parsed verifying key so a caller can hand it straight to a
    /// team-membership check without re-parsing the raw bytes.
    fn verify_strict(&self, transcript: &[u8]) -> Result<VerifyingKey, SignatureVerificationError> {
        let public_key = VerifyingKey::from_bytes(&self.public_key.raw)
            .map_err(|_| SignatureVerificationError::InvalidPublicKey)?;
        let signature = Signature::from_components(self.signature_r.raw, self.signature_s.raw);
        public_key
            .verify_strict(transcript, &signature)
            .map_err(|_| SignatureVerificationError::InvalidSignature)?;
        Ok(public_key)
    }

    fn sign(signing_key: &SigningKey, transcript: &[u8]) -> Self {
        let signature: Signature = signing_key.sign(transcript);
        Self {
            public_key: Inline::new(signing_key.verifying_key().to_bytes()),
            signature_r: Inline::new(*signature.r_bytes()),
            signature_s: Inline::new(*signature.s_bytes()),
        }
    }

    fn raw_fields(&self) -> [[u8; 32]; 3] {
        [
            self.public_key.raw,
            self.signature_r.raw,
            self.signature_s.raw,
        ]
    }
}


/// A collection name that is legal as part of an identity.
///
/// Names are compared byte for byte, because that is what hashing a
/// descriptor does. So `compass`, `Compass` and `compass ` would be three
/// different collections that a person reads as one. The charset exists to
/// make that class of accident unrepresentable rather than merely unlikely:
/// lowercase ASCII letters, digits and `-`, starting with a letter, ending
/// with a letter or digit, at most 32 bytes.
///
/// It rejects rather than normalises. Silently lowercasing what a caller
/// wrote would mean the stored identity is not the one they typed, and the
/// whole reason a name replaced an opaque scope id was so that what is stored
/// can be read back and recognised.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CollectionName(String);

/// Why a string cannot be a [`CollectionName`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidCollectionName {
    /// The name was empty.
    Empty,
    /// The name exceeded the 32 bytes a `ShortString` holds inline.
    TooLong {
        /// Length of the offending name, in bytes.
        len: usize,
    },
    /// The name did not begin with a lowercase ASCII letter.
    BadStart,
    /// The name did not end with a lowercase ASCII letter or digit.
    BadEnd,
    /// The name contained a byte outside `[a-z0-9-]`.
    BadByte {
        /// The offending byte.
        byte: u8,
    },
}

impl fmt::Display for InvalidCollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "collection name is empty"),
            Self::TooLong { len } => {
                write!(f, "collection name is {len} bytes, the maximum is 32")
            }
            Self::BadStart => write!(
                f,
                "collection name must start with a lowercase ASCII letter"
            ),
            Self::BadEnd => write!(
                f,
                "collection name must end with a lowercase ASCII letter or digit"
            ),
            Self::BadByte { byte } => write!(
                f,
                "collection name may only contain [a-z0-9-]; found byte {byte:#04X}"
            ),
        }
    }
}

impl Error for InvalidCollectionName {}

impl CollectionName {
    /// Accept a string as a collection name, or say exactly why it is not one.
    pub fn new(text: &str) -> Result<Self, InvalidCollectionName> {
        let bytes = text.as_bytes();
        let Some(&first) = bytes.first() else {
            return Err(InvalidCollectionName::Empty);
        };
        if bytes.len() > 32 {
            return Err(InvalidCollectionName::TooLong { len: bytes.len() });
        }
        if !first.is_ascii_lowercase() {
            return Err(InvalidCollectionName::BadStart);
        }
        let last = bytes[bytes.len() - 1];
        if !(last.is_ascii_lowercase() || last.is_ascii_digit()) {
            return Err(InvalidCollectionName::BadEnd);
        }
        for &byte in bytes {
            if !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-') {
                return Err(InvalidCollectionName::BadByte { byte });
            }
        }
        Ok(Self(text.to_owned()))
    }

    /// The name as written, which is also the name as stored.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CollectionName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
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
    pub fn verify_strict(&self) -> Result<(), SignatureVerificationError> {
        let public_key = VerifyingKey::from_bytes(&self.public_key.raw)
            .map_err(|_| SignatureVerificationError::InvalidPublicKey)?;
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
    ) -> Result<(), SignatureVerificationError> {
        if public_key.to_bytes() != self.public_key.raw {
            return self.verify_strict();
        }
        self.verify_signature_strict(public_key)
    }

    fn verify_signature_strict(
        &self,
        public_key: &VerifyingKey,
    ) -> Result<(), SignatureVerificationError> {
        let signature = Signature::from_components(self.signature_r.raw, self.signature_s.raw);
        let transcript =
            commit_transcript(self.public_key, self.collection, self.data, self.metadata);
        public_key
            .verify_strict(&transcript, &signature)
            .map_err(|_| SignatureVerificationError::InvalidSignature)
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

/// Exact join equation inside one collection lattice.
///
/// The equation is the whole claim: `low join high = result` under the
/// collection's recipe, true or false regardless of who wrote the record. A
/// merge may additionally carry a [`CollectionSignature`], which asserts
/// nothing further about the equation and only says who stands behind it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionMerge {
    id: Id,
    collection: CollectionHandle,
    low: CollectionData,
    high: CollectionData,
    result: CollectionData,
    signature: Option<CollectionSignature>,
}

impl CollectionMerge {
    /// Construct an unsigned commutative merge record, sorting its two inputs
    /// by digest.
    pub fn new(
        collection: CollectionHandle,
        left: CollectionData,
        right: CollectionData,
        result: CollectionData,
    ) -> Self {
        Self::from_parts(collection, left, right, result, None)
    }

    /// Sign a canonical `MERGE(collection, low, high) = result` equation.
    ///
    /// Signing is an offer, not a promotion: the same equation asserted
    /// without a signature remains admissible by recomputation, which is the
    /// stronger guarantee because it needs no trust at all.
    pub fn sign(
        signing_key: &SigningKey,
        collection: CollectionHandle,
        left: CollectionData,
        right: CollectionData,
        result: CollectionData,
    ) -> Self {
        let (low, high) = ordered_inputs(left, right);
        let public_key = Inline::new(signing_key.verifying_key().to_bytes());
        let transcript = merge_transcript(public_key, collection, low, high, result);
        let signature = CollectionSignature::sign(signing_key, &transcript);
        Self::from_ordered(collection, low, high, result, Some(signature))
    }

    /// Assemble a merge from raw fields, sorting its two inputs by digest.
    pub fn from_parts(
        collection: CollectionHandle,
        left: CollectionData,
        right: CollectionData,
        result: CollectionData,
        signature: Option<CollectionSignature>,
    ) -> Self {
        let (low, high) = ordered_inputs(left, right);
        Self::from_ordered(collection, low, high, result, signature)
    }

    fn from_ordered(
        collection: CollectionHandle,
        low: CollectionData,
        high: CollectionData,
        result: CollectionData,
        signature: Option<CollectionSignature>,
    ) -> Self {
        let id = match &signature {
            None => collection_record_id(
                KIND_COLLECTION_MERGE,
                &merge_bytes(collection, low, high, result),
            ),
            Some(signature) => collection_record_id(
                KIND_COLLECTION_SIGNED_MERGE,
                &signed_merge_bytes(collection, low, high, result, signature),
            ),
        };
        Self {
            id,
            collection,
            low,
            high,
            result,
            signature,
        }
    }

    /// Decode one exact, canonically ordered dense merge payload.
    ///
    /// The two layouts are told apart by length, which is what makes the
    /// untagged form self-describing: [`COLLECTION_MERGE_BYTES_LEN`] is the
    /// bare equation and [`COLLECTION_SIGNED_MERGE_BYTES_LEN`] is that
    /// equation followed by a signature.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RecordDecodeError> {
        let signature = match bytes.len() {
            COLLECTION_MERGE_BYTES_LEN => None,
            COLLECTION_SIGNED_MERGE_BYTES_LEN => Some(signature_from_fields(bytes, 4)),
            actual => {
                return Err(RecordDecodeError::InvalidLength {
                    expected: COLLECTION_MERGE_BYTES_LEN,
                    actual,
                })
            }
        };
        let collection = Inline::new(field_at(bytes, 0));
        let low: CollectionData = Inline::new(field_at(bytes, 1));
        let high: CollectionData = Inline::new(field_at(bytes, 2));
        if high < low {
            return Err(RecordDecodeError::NonCanonicalMergeInputs);
        }
        Ok(Self::from_ordered(
            collection,
            low,
            high,
            Inline::new(field_at(bytes, 3)),
            signature,
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

    /// Evidence about who asserted this equation, if anyone signed it.
    pub fn signature(&self) -> Option<CollectionSignature> {
        self.signature
    }

    /// This record's equation, stripped of any signature.
    ///
    /// The equation is the primary object, and this is its canonical name:
    /// `merge.unsigned().id()` is the same id for every signer of one
    /// equation, and the id a resolver synthesises for that equation as an
    /// implied theorem.
    pub fn unsigned(&self) -> Self {
        Self::from_ordered(self.collection, self.low, self.high, self.result, None)
    }

    /// Strictly verify this record's signature, if it has one.
    ///
    /// `Ok(None)` means the record is unsigned, which is not a failure: an
    /// unsigned equation is admitted by recomputing it. `Ok(Some(key))` means
    /// `key` really did sign this exact equation; whether `key` is authorized
    /// for the collection's team is a separate question.
    pub fn verify_strict(&self) -> Result<Option<VerifyingKey>, SignatureVerificationError> {
        let Some(signature) = &self.signature else {
            return Ok(None);
        };
        let transcript = merge_transcript(
            signature.public_key,
            self.collection,
            self.low,
            self.high,
            self.result,
        );
        signature.verify_strict(&transcript).map(Some)
    }

    /// Exact bytes a signature over this equation attests to.
    pub fn signing_transcript(
        &self,
        public_key: Inline<ED25519PublicKey>,
    ) -> [u8; MERGE_TRANSCRIPT_LEN] {
        merge_transcript(public_key, self.collection, self.low, self.high, self.result)
    }

    /// Encode this equation into its exact dense layout.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.signature {
            None => merge_bytes(self.collection, self.low, self.high, self.result).to_vec(),
            Some(signature) => {
                signed_merge_bytes(self.collection, self.low, self.high, self.result, signature)
                    .to_vec()
            }
        }
    }
}

/// One exact observation of the canonical join homomorphism between two
/// collection lattices.
///
/// Like a [`CollectionMerge`], the equation is the whole claim and an optional
/// [`CollectionSignature`] is evidence about its asserter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionDerive {
    id: Id,
    target: CollectionHandle,
    input: CollectionData,
    output: CollectionData,
    signature: Option<CollectionSignature>,
}

impl CollectionDerive {
    /// Construct an unsigned canonical `DERIVE(target, input, output)` record.
    ///
    /// The target is named by descriptor handle, exactly as a commit names its
    /// collection, and that descriptor already says which collection is the
    /// source and by what recipe. A derive therefore says *which instance* of
    /// a mapping was computed, never *which mapping*.
    pub fn new(target: CollectionHandle, input: CollectionData, output: CollectionData) -> Self {
        Self::from_parts(target, input, output, None)
    }

    /// Sign a canonical `DERIVE(target, input) = output` equation.
    ///
    /// This is the equation whose recomputation is expensive enough to be
    /// worth an assertion: re-deriving a `SuccinctArchive` from its source
    /// rebuilds every prefix, mask and rotation. Signing offers a reader who
    /// trusts the signer a way out of that work; it does not withdraw the
    /// unsigned, recomputable form.
    pub fn sign(
        signing_key: &SigningKey,
        target: CollectionHandle,
        input: CollectionData,
        output: CollectionData,
    ) -> Self {
        let public_key = Inline::new(signing_key.verifying_key().to_bytes());
        let transcript = derive_transcript(public_key, target, input, output);
        let signature = CollectionSignature::sign(signing_key, &transcript);
        Self::from_parts(target, input, output, Some(signature))
    }

    /// Assemble a derive from raw fields.
    pub fn from_parts(
        target: CollectionHandle,
        input: CollectionData,
        output: CollectionData,
        signature: Option<CollectionSignature>,
    ) -> Self {
        let id = match &signature {
            None => {
                collection_record_id(KIND_COLLECTION_DERIVE, &derive_bytes(target, input, output))
            }
            Some(signature) => collection_record_id(
                KIND_COLLECTION_SIGNED_DERIVE,
                &signed_derive_bytes(target, input, output, signature),
            ),
        };
        Self {
            id,
            target,
            input,
            output,
            signature,
        }
    }

    /// Decode one exact dense derive payload.
    ///
    /// The two layouts are told apart by length, exactly as
    /// [`CollectionMerge::from_bytes`] does.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RecordDecodeError> {
        let signature = match bytes.len() {
            COLLECTION_DERIVE_BYTES_LEN => None,
            COLLECTION_SIGNED_DERIVE_BYTES_LEN => Some(signature_from_fields(bytes, 3)),
            actual => {
                return Err(RecordDecodeError::InvalidLength {
                    expected: COLLECTION_DERIVE_BYTES_LEN,
                    actual,
                })
            }
        };
        Ok(Self::from_parts(
            Inline::new(field_at(bytes, 0)),
            Inline::new(field_at(bytes, 1)),
            Inline::new(field_at(bytes, 2)),
            signature,
        ))
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

    /// Evidence about who asserted this equation, if anyone signed it.
    pub fn signature(&self) -> Option<CollectionSignature> {
        self.signature
    }

    /// This record's equation, stripped of any signature.
    ///
    /// See [`CollectionMerge::unsigned`]: this is the equation's canonical
    /// name, independent of who asserted it.
    pub fn unsigned(&self) -> Self {
        Self::from_parts(self.target, self.input, self.output, None)
    }

    /// Strictly verify this record's signature, if it has one.
    ///
    /// `Ok(None)` means the record is unsigned. See
    /// [`CollectionMerge::verify_strict`].
    pub fn verify_strict(&self) -> Result<Option<VerifyingKey>, SignatureVerificationError> {
        let Some(signature) = &self.signature else {
            return Ok(None);
        };
        let transcript =
            derive_transcript(signature.public_key, self.target, self.input, self.output);
        signature.verify_strict(&transcript).map(Some)
    }

    /// Exact bytes a signature over this equation attests to.
    pub fn signing_transcript(
        &self,
        public_key: Inline<ED25519PublicKey>,
    ) -> [u8; DERIVE_TRANSCRIPT_LEN] {
        derive_transcript(public_key, self.target, self.input, self.output)
    }

    /// Encode this equation into its exact dense layout.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.signature {
            None => derive_bytes(self.target, self.input, self.output).to_vec(),
            Some(signature) => {
                signed_derive_bytes(self.target, self.input, self.output, signature).to_vec()
            }
        }
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
                exact_len(payload, COLLECTION_MERGE_BYTES_LEN)?;
                Ok(Self::Merge(CollectionMerge::from_bytes(payload)?))
            }
            COLLECTION_RECORD_KIND_DERIVE_V1 => {
                exact_len(payload, COLLECTION_DERIVE_BYTES_LEN)?;
                Ok(Self::Derive(CollectionDerive::from_bytes(payload)?))
            }
            COLLECTION_RECORD_KIND_SIGNED_MERGE_V1 => {
                exact_len(payload, COLLECTION_SIGNED_MERGE_BYTES_LEN)?;
                Ok(Self::Merge(CollectionMerge::from_bytes(payload)?))
            }
            COLLECTION_RECORD_KIND_SIGNED_DERIVE_V1 => {
                exact_len(payload, COLLECTION_SIGNED_DERIVE_BYTES_LEN)?;
                Ok(Self::Derive(CollectionDerive::from_bytes(payload)?))
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
                let kind = match record.signature() {
                    None => COLLECTION_RECORD_KIND_MERGE_V1,
                    Some(_) => COLLECTION_RECORD_KIND_SIGNED_MERGE_V1,
                };
                tagged_bytes(kind, &record.to_bytes())
            }
            Self::Derive(record) => {
                let kind = match record.signature() {
                    None => COLLECTION_RECORD_KIND_DERIVE_V1,
                    Some(_) => COLLECTION_RECORD_KIND_SIGNED_DERIVE_V1,
                };
                tagged_bytes(kind, &record.to_bytes())
            }
        }
    }

    /// This record with any equation signature stripped.
    ///
    /// A commit is unchanged: its signature *is* its statement. A merge or
    /// derive loses only the evidence about who asserted it, keeping the
    /// equation, which is what a reader who recomputes needs.
    pub fn unsigned_equation(&self) -> Self {
        match self {
            Self::Commit(record) => Self::Commit(*record),
            Self::Merge(record) => Self::Merge(record.unsigned()),
            Self::Derive(record) => Self::Derive(record.unsigned()),
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
/// Dense generic-store tag for the signed [`CollectionRecord::Merge`] layout.
///
/// A separate tag rather than a length sniff at the outer level, so a reader
/// that does not know about signed equations rejects the record by kind
/// instead of misreading its length.
pub const COLLECTION_RECORD_KIND_SIGNED_MERGE_V1: u8 = 4;
/// Dense generic-store tag for the signed [`CollectionRecord::Derive`] layout.
pub const COLLECTION_RECORD_KIND_SIGNED_DERIVE_V1: u8 = 5;


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

fn signed_merge_bytes(
    collection: CollectionHandle,
    low: CollectionData,
    high: CollectionData,
    result: CollectionData,
    signature: &CollectionSignature,
) -> [u8; COLLECTION_SIGNED_MERGE_BYTES_LEN] {
    let [public_key, r, s] = signature.raw_fields();
    concat_fields([collection.raw, low.raw, high.raw, result.raw, public_key, r, s])
}

fn signed_derive_bytes(
    target: CollectionHandle,
    input: CollectionData,
    output: CollectionData,
    signature: &CollectionSignature,
) -> [u8; COLLECTION_SIGNED_DERIVE_BYTES_LEN] {
    let [public_key, r, s] = signature.raw_fields();
    concat_fields([target.raw, input.raw, output.raw, public_key, r, s])
}

fn ordered_inputs(
    mut left: CollectionData,
    mut right: CollectionData,
) -> (CollectionData, CollectionData) {
    if right < left {
        std::mem::swap(&mut left, &mut right);
    }
    (left, right)
}

fn signature_from_fields(bytes: &[u8], first: usize) -> CollectionSignature {
    CollectionSignature::from_parts(
        Inline::new(field_at(bytes, first)),
        Inline::new(field_at(bytes, first + 1)),
        Inline::new(field_at(bytes, first + 2)),
    )
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

fn field_at(bytes: &[u8], index: usize) -> [u8; 32] {
    bytes[index * 32..(index + 1) * 32]
        .try_into()
        .expect("fixed dense record field")
}

fn exact_len(bytes: &[u8], expected: usize) -> Result<(), RecordDecodeError> {
    if bytes.len() == expected {
        Ok(())
    } else {
        Err(RecordDecodeError::InvalidLength {
            expected,
            actual: bytes.len(),
        })
    }
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

fn merge_transcript(
    public_key: Inline<ED25519PublicKey>,
    collection: CollectionHandle,
    low: CollectionData,
    high: CollectionData,
    result: CollectionData,
) -> [u8; MERGE_TRANSCRIPT_LEN] {
    let mut transcript = [0; MERGE_TRANSCRIPT_LEN];
    let written = write_equation_transcript(
        &mut transcript,
        KIND_COLLECTION_SIGNED_MERGE,
        public_key,
        &[collection.raw, low.raw, high.raw, result.raw],
    );
    debug_assert_eq!(written, MERGE_TRANSCRIPT_LEN);
    transcript
}

fn derive_transcript(
    public_key: Inline<ED25519PublicKey>,
    target: CollectionHandle,
    input: CollectionData,
    output: CollectionData,
) -> [u8; DERIVE_TRANSCRIPT_LEN] {
    let mut transcript = [0; DERIVE_TRANSCRIPT_LEN];
    let written = write_equation_transcript(
        &mut transcript,
        KIND_COLLECTION_SIGNED_DERIVE,
        public_key,
        &[target.raw, input.raw, output.raw],
    );
    debug_assert_eq!(written, DERIVE_TRANSCRIPT_LEN);
    transcript
}

/// Lay out one domain-separated equation transcript.
///
/// The kind id separates a signed merge from a signed derive, so a signature
/// gathered over one equation shape can never be replayed as the other.
fn write_equation_transcript(
    transcript: &mut [u8],
    kind: Id,
    public_key: Inline<ED25519PublicKey>,
    fields: &[[u8; 32]],
) -> usize {
    let mut offset = 0;
    let mut append = |bytes: &[u8]| {
        let end = offset + bytes.len();
        transcript[offset..end].copy_from_slice(bytes);
        offset = end;
    };
    append(EQUATION_TRANSCRIPT_DOMAIN);
    append(&kind.raw());
    append(&EQUATION_TRANSCRIPT_VERSION.to_be_bytes());
    append(&public_key.raw);
    for field in fields {
        append(field);
    }
    offset
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

    /// A root is anchored by its name *and* its team, and its identity is a
    /// function of that anchor together with the representation and recipe it
    /// names. Change any one of the four and it is a different collection.
    #[test]
    fn collection_descriptor_is_anchor_specific_and_roundtrips() {
        use crate::collection::descriptor;

        let team = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let other_team = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let name = CollectionName::new("first").unwrap();
        let other_name = CollectionName::new("second").unwrap();

        let a = descriptor::naming(&name, team, id(2), id(3)).into_facts();
        let renamed = descriptor::naming(&other_name, team, id(2), id(3)).into_facts();
        let reteamed = descriptor::naming(&name, other_team, id(2), id(3)).into_facts();
        let other_representation = descriptor::naming(&name, team, id(4), id(3)).into_facts();
        let other_recipe = descriptor::naming(&name, team, id(2), id(4)).into_facts();

        let handle = |facts: &TribleSet| {
            <TribleSet as crate::blob::IntoBlob<SimpleArchive>>::to_blob(facts.clone()).get_handle()
        };
        assert_ne!(handle(&a), handle(&renamed));
        assert_ne!(handle(&a), handle(&reteamed));
        assert_ne!(handle(&a), handle(&other_representation));
        assert_ne!(handle(&a), handle(&other_recipe));

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
            name,
            "the anchor reads back as what was written"
        );
        assert_eq!(descriptor::team(&a).unwrap().unwrap(), team);
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
            Err(SignatureVerificationError::InvalidSignature)
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
            Err(SignatureVerificationError::InvalidSignature)
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
            Err(SignatureVerificationError::InvalidPublicKey)
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
            CollectionMerge::from_bytes(&forward.to_bytes()).unwrap(),
            forward
        );
    }

    #[test]
    fn a_signature_rides_beside_the_equation_it_never_changes() {
        let key = SigningKey::from_bytes(&[13; 32]);
        let signed = CollectionMerge::sign(&key, collection(1), hash(2), hash(3), hash(4));
        let plain = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));

        // The equation is identical; only the evidence differs, and stripping
        // it recovers exactly the record nobody signed.
        assert_eq!(signed.collection(), plain.collection());
        assert_eq!(signed.inputs(), plain.inputs());
        assert_eq!(signed.result(), plain.result());
        assert_eq!(signed.unsigned(), plain);
        assert_eq!(plain.unsigned(), plain);

        // Distinct records, because an intrinsic id is a function of bytes and
        // the store is keyed by it. `unsigned().id()` is the equation's own
        // name, shared by every signer of it.
        assert_ne!(signed.id(), plain.id());
        assert_eq!(signed.unsigned().id(), plain.id());

        let other = CollectionMerge::sign(
            &SigningKey::from_bytes(&[14; 32]),
            collection(1),
            hash(2),
            hash(3),
            hash(4),
        );
        assert_ne!(other.id(), signed.id());
        assert_eq!(other.unsigned().id(), signed.unsigned().id());
    }

    #[test]
    fn signed_equations_roundtrip_and_verify() {
        let key = SigningKey::from_bytes(&[15; 32]);
        let merge = CollectionMerge::sign(&key, collection(1), hash(3), hash(2), hash(4));
        let derive = CollectionDerive::sign(&key, collection(2), hash(5), hash(6));

        assert_eq!(merge.to_bytes().len(), COLLECTION_SIGNED_MERGE_BYTES_LEN);
        assert_eq!(derive.to_bytes().len(), COLLECTION_SIGNED_DERIVE_BYTES_LEN);
        assert_eq!(CollectionMerge::from_bytes(&merge.to_bytes()), Ok(merge));
        assert_eq!(CollectionDerive::from_bytes(&derive.to_bytes()), Ok(derive));

        assert_eq!(merge.verify_strict(), Ok(Some(key.verifying_key())));
        assert_eq!(derive.verify_strict(), Ok(Some(key.verifying_key())));

        // Signing is commutative in the inputs, exactly as the equation is.
        assert_eq!(
            merge,
            CollectionMerge::sign(&key, collection(1), hash(2), hash(3), hash(4))
        );
    }

    #[test]
    fn an_unsigned_equation_is_not_a_verification_failure() {
        assert_eq!(
            CollectionMerge::new(collection(1), hash(2), hash(3), hash(4)).verify_strict(),
            Ok(None)
        );
        assert_eq!(
            CollectionDerive::new(collection(2), hash(3), hash(4)).verify_strict(),
            Ok(None)
        );
    }

    #[test]
    fn a_signature_does_not_travel_to_another_equation() {
        let key = SigningKey::from_bytes(&[16; 32]);
        let signed = CollectionMerge::sign(&key, collection(1), hash(2), hash(3), hash(4));
        let signature = signed.signature().expect("signed");

        // Same signature, different result: the transcript covers the whole
        // equation, so the evidence no longer applies.
        let moved =
            CollectionMerge::from_parts(collection(1), hash(2), hash(3), hash(9), Some(signature));
        assert_eq!(
            moved.verify_strict(),
            Err(SignatureVerificationError::InvalidSignature)
        );

        // Nor to the other equation shape: the transcript is domain-separated
        // by record kind, so a merge signature cannot be replayed as a derive.
        let crossed = CollectionDerive::from_parts(collection(1), hash(2), hash(3), Some(signature));
        assert_eq!(
            crossed.verify_strict(),
            Err(SignatureVerificationError::InvalidSignature)
        );
    }

    #[test]
    fn signed_and_unsigned_layouts_are_told_apart_by_length() {
        let key = SigningKey::from_bytes(&[17; 32]);
        let signed = CollectionDerive::sign(&key, collection(2), hash(5), hash(6));
        let plain = signed.unsigned();

        assert_eq!(
            CollectionDerive::from_bytes(&plain.to_bytes()),
            Ok(plain),
            "the bare equation still decodes as itself"
        );
        assert_eq!(
            CollectionDerive::from_bytes(&signed.to_bytes()[..COLLECTION_DERIVE_BYTES_LEN]),
            Ok(plain),
            "the signed layout begins with its own equation"
        );
        assert_eq!(
            CollectionDerive::from_bytes(&[0u8; 7]),
            Err(RecordDecodeError::InvalidLength {
                expected: COLLECTION_DERIVE_BYTES_LEN,
                actual: 7,
            })
        );
    }

    #[test]
    fn the_generic_codec_tags_signed_equations_separately() {
        let key = SigningKey::from_bytes(&[18; 32]);
        for record in [
            CollectionRecord::Merge(CollectionMerge::sign(
                &key,
                collection(1),
                hash(2),
                hash(3),
                hash(4),
            )),
            CollectionRecord::Derive(CollectionDerive::sign(&key, collection(2), hash(5), hash(6))),
        ] {
            let bytes = record.to_bytes();
            assert!(matches!(
                bytes[0],
                COLLECTION_RECORD_KIND_SIGNED_MERGE_V1 | COLLECTION_RECORD_KIND_SIGNED_DERIVE_V1
            ));
            assert_eq!(CollectionRecord::from_bytes(&bytes), Ok(record));

            // A reader that only knows the unsigned tags refuses the record by
            // kind rather than misreading its length.
            let mut mislabelled = bytes.clone();
            mislabelled[0] = match bytes[0] {
                COLLECTION_RECORD_KIND_SIGNED_MERGE_V1 => COLLECTION_RECORD_KIND_MERGE_V1,
                _ => COLLECTION_RECORD_KIND_DERIVE_V1,
            };
            assert!(CollectionRecord::from_bytes(&mislabelled).is_err());

            // Stripping the signature yields the plain equation and its tag.
            let unsigned = record.unsigned_equation().to_bytes();
            assert!(matches!(
                unsigned[0],
                COLLECTION_RECORD_KIND_MERGE_V1 | COLLECTION_RECORD_KIND_DERIVE_V1
            ));
        }
    }

    #[test]
    fn derive_roundtrips() {
        let record = CollectionDerive::new(collection(2), hash(3), hash(4));
        assert_eq!(CollectionDerive::from_bytes(&record.to_bytes()), Ok(record));
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
            CollectionMerge::from_bytes(&bytes),
            Err(RecordDecodeError::NonCanonicalMergeInputs)
        );
    }

    #[test]
    fn transcript_and_record_roots_are_golden() {
        let descriptor = crate::collection::descriptor::naming(
            &CollectionName::new("first").unwrap(),
            SigningKey::from_bytes(&[1; 32]).verifying_key(),
            id(2),
            id(3),
        )
        .into_facts();
        let descriptor_blob =
            <TribleSet as crate::blob::IntoBlob<SimpleArchive>>::to_blob(descriptor.clone());
        let commit =
            CollectionCommit::sign(&fixture_key(), collection(1), hash(2), Inline::new([3; 32]));
        let merge = CollectionMerge::new(collection(1), hash(2), hash(3), hash(4));
        let derive = CollectionDerive::new(collection(2), hash(3), hash(4));

        // Descriptor wire bytes are unchanged by the identity cutover.
        assert_eq!(
            crate::collection::descriptor::entity(&descriptor).unwrap(),
            id_hex!("D3942D72389636880F528243079C24DF")
        );
        assert_eq!(
            descriptor_blob.get_handle().raw,
            hex!("27BDE8E0150DCEC4F5330DF88D12EAEE0E1B174AA59AB6F2E10A3F9B20B8B8D7")
        );
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
