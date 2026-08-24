//! Store-independent, blob-native capability verification.
//!
//! A capability occurrence is two ordinary canonical
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive)
//! blobs:
//!
//! - a claim naming one subject, one exact action/resource atom, an
//!   invocation/delegation mode, an optional inclusive validity interval, and
//!   an optional exact parent signature blob; and
//! - a signature record naming that exact claim blob and carrying the
//!   issuer's Ed25519 signature over the claim's canonical bytes.
//!
//! Proofs carry those pairs in root-to-leaf order. Verification needs only the
//! proof, an externally supplied trust root, an explicit instant, and the
//! exact leaf claim the caller expects. It does not enumerate a store, resolve
//! a collection, consult a log, or infer ambient authority.

use std::collections::HashSet;
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use ed25519::signature::Signer;
use ed25519::Signature;
use ed25519_dalek::{SigningKey, VerifyingKey};
use hifitime::{Duration, Epoch};

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::{Blob, IntoBlob, TryFromBlob};
use crate::id::{id_hex, ExclusiveId, Id};
use crate::inline::encodings::ed25519::{ED25519PublicKey, ED25519RComponent, ED25519SComponent};
use crate::inline::encodings::genid::GenId;
use crate::inline::encodings::hash::{Blake3, Handle};
use crate::inline::encodings::time::NsTAIInterval;
use crate::inline::{Encodes, Inline, InlineEncoding, TryFromInline, TryToInline};
use crate::metadata::{self, MetaDescribe};
use crate::prelude::{attributes, entity, find, pattern};
use crate::trible::{Fragment, TribleSet, TRIBLE_LEN};

/// Stable kind of a canonical capability claim blob.
///
/// Minted with `trible genid` on 2026-08-24.
pub const KIND_CAPABILITY_CLAIM: Id = id_hex!("A5A0B81E2FABC64DDE9E81C4F4772768");

/// Stable kind of a canonical capability signature blob.
///
/// Minted with `trible genid` on 2026-08-24.
pub const KIND_CAPABILITY_SIGNATURE: Id = id_hex!("B59FB06BE8FB5201B3E8341C1DD844DC");

/// Stable stored value for [`CapabilityMode::Invoke`].
///
/// Minted with `trible genid` on 2026-08-24.
const MODE_INVOKE: Id = id_hex!("917C8891DA2350793577BD10AB88008E");

/// Stable stored value for [`CapabilityMode::Delegate`].
///
/// Minted with `trible genid` on 2026-08-24.
const MODE_DELEGATE: Id = id_hex!("1A9F33A5DC8CEAE7C2ACDF77945CE2EF");

/// Stable stored value for [`CapabilityMode::InvokeAndDelegate`].
///
/// Minted with `trible genid` on 2026-08-24.
const MODE_INVOKE_AND_DELEGATE: Id = id_hex!("3838CF88E3EB1596DBAD87666801ADF3");

/// Inline encoding for an action-specific, type-erased resource identity.
///
/// The kernel compares these 32 bytes exactly. An action-specific adapter is
/// responsible for converting its concrete Rust resource type to and from
/// [`CapabilityResource`]; the kernel deliberately has no resource registry.
pub struct CapabilityResourceEncoding;

impl MetaDescribe for CapabilityResourceEncoding {
    fn describe() -> Fragment {
        // Minted with `trible genid` on 2026-08-24.
        let id = id_hex!("52297CA2A448E6163158E9498F10559C");
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "capability_resource",
                metadata::description: "Opaque 32-byte resource identity interpreted by the exact capability action. The capability kernel compares these bytes without a registry or ambient resource hierarchy.",
                metadata::tag: metadata::KIND_INLINE_ENCODING,
        }
    }
}

impl InlineEncoding for CapabilityResourceEncoding {
    type ValidationError = Infallible;
    type Encoding = Self;
}

/// Exact, opaque 32-byte identity of a resource governed by an action.
///
/// Use [`From<Inline<S>>`](Self::from) to erase a typed inline resource into
/// the kernel. Decoding it again belongs to the adapter for the associated
/// [`CapabilityAction`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CapabilityResource([u8; 32]);

impl CapabilityResource {
    /// Construct an opaque resource from its exact portable bytes.
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the exact portable resource bytes.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Borrow the exact portable resource bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl<S: InlineEncoding> From<Inline<S>> for CapabilityResource {
    fn from(resource: Inline<S>) -> Self {
        Self(resource.raw)
    }
}

impl Encodes<CapabilityResource> for CapabilityResourceEncoding {
    type Output = Inline<CapabilityResourceEncoding>;

    fn encode(source: CapabilityResource) -> Self::Output {
        Inline::new(source.0)
    }
}

impl Encodes<&CapabilityResource> for CapabilityResourceEncoding {
    type Output = Inline<CapabilityResourceEncoding>;

    fn encode(source: &CapabilityResource) -> Self::Output {
        Inline::new(source.0)
    }
}

impl TryFromInline<'_, CapabilityResourceEncoding> for CapabilityResource {
    type Error = Infallible;

    fn try_from_inline(value: &Inline<CapabilityResourceEncoding>) -> Result<Self, Self::Error> {
        Ok(Self(value.raw))
    }
}

/// Exact, uninterpreted 128-bit action identity.
///
/// Actions do not imply one another. Invocation and delegation modes apply
/// only to the same byte-exact action/resource atom.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CapabilityAction(Id);

impl CapabilityAction {
    /// Wrap one exact action identifier.
    pub const fn new(id: Id) -> Self {
        Self(id)
    }

    /// Return the exact action identifier.
    pub const fn id(self) -> Id {
        self.0
    }
}

impl From<Id> for CapabilityAction {
    fn from(id: Id) -> Self {
        Self(id)
    }
}

/// One exact action/resource authorization atom.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CapabilityAtom {
    action: CapabilityAction,
    resource: CapabilityResource,
}

impl CapabilityAtom {
    /// Pair one exact action with one exact opaque resource identity.
    pub const fn new(action: CapabilityAction, resource: CapabilityResource) -> Self {
        Self { action, resource }
    }

    /// Exact action governed by this atom.
    pub const fn action(self) -> CapabilityAction {
        self.action
    }

    /// Exact resource governed by this atom.
    pub const fn resource(self) -> CapabilityResource {
        self.resource
    }
}

/// The three nonempty invocation/delegation modes.
///
/// Invocation and delegation are independent uses. A mode satisfies a
/// requirement when it contains every use the requirement names.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CapabilityMode {
    /// Invoke this exact action on this exact resource.
    Invoke,
    /// Delegate this exact action/resource atom without invoking it.
    Delegate,
    /// Both invoke and delegate this exact action/resource atom.
    InvokeAndDelegate,
}

impl CapabilityMode {
    /// Whether this mode satisfies a caller's minimum required mode.
    pub const fn satisfies(self, required: Self) -> bool {
        self.bits() & required.bits() == required.bits()
    }

    /// Whether this mode may issue a child capability.
    pub const fn delegates(self) -> bool {
        self.bits() & Self::Delegate.bits() != 0
    }

    const fn id(self) -> Id {
        match self {
            Self::Invoke => MODE_INVOKE,
            Self::Delegate => MODE_DELEGATE,
            Self::InvokeAndDelegate => MODE_INVOKE_AND_DELEGATE,
        }
    }

    fn from_id(id: Id) -> Option<Self> {
        match id {
            MODE_INVOKE => Some(Self::Invoke),
            MODE_DELEGATE => Some(Self::Delegate),
            MODE_INVOKE_AND_DELEGATE => Some(Self::InvokeAndDelegate),
            _ => None,
        }
    }

    const fn bits(self) -> u8 {
        match self {
            Self::Invoke => 0b01,
            Self::Delegate => 0b10,
            Self::InvokeAndDelegate => 0b11,
        }
    }
}

/// A validated inclusive validity interval for a capability claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityValidity(Inline<NsTAIInterval>);

impl CapabilityValidity {
    /// Construct an inclusive validity interval.
    pub fn new(lower: Epoch, upper: Epoch) -> Result<Self, CapabilityValidityError> {
        let lower_ns = lower.to_tai_duration().total_nanoseconds();
        let upper_ns = upper.to_tai_duration().total_nanoseconds();
        let inline = (lower, upper)
            .try_to_inline()
            .map_err(|_| CapabilityValidityError { lower_ns, upper_ns })?;
        Ok(Self(inline))
    }

    /// Inclusive lower and upper bounds.
    pub fn bounds(self) -> (Epoch, Epoch) {
        let (lower, upper) = self.bounds_ns();
        (
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower)),
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper)),
        )
    }

    /// Whether `instant` lies inside both inclusive bounds.
    pub fn contains(self, instant: Epoch) -> bool {
        let instant = instant.to_tai_duration().total_nanoseconds();
        let (lower, upper) = self.bounds_ns();
        lower <= instant && instant <= upper
    }

    fn from_inline(inline: Inline<NsTAIInterval>) -> Result<Self, CapabilityValidityError> {
        let (lower_ns, upper_ns) =
            inline
                .try_from_inline::<(i128, i128)>()
                .map_err(|error| CapabilityValidityError {
                    lower_ns: error.lower,
                    upper_ns: error.upper,
                })?;
        debug_assert!(lower_ns <= upper_ns);
        Ok(Self(inline))
    }

    fn inline(self) -> Inline<NsTAIInterval> {
        self.0
    }

    fn from_bounds_ns(lower: i128, upper: i128) -> Self {
        Self::new(
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(lower)),
            Epoch::from_tai_duration(Duration::from_total_nanoseconds(upper)),
        )
        .expect("the intersection of valid intervals is valid")
    }

    fn bounds_ns(self) -> (i128, i128) {
        self.0
            .try_from_inline::<(i128, i128)>()
            .expect("CapabilityValidity is validated at construction")
    }
}

/// An attempted validity interval had its lower bound after its upper bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityValidityError {
    lower_ns: i128,
    upper_ns: i128,
}

impl CapabilityValidityError {
    /// Rejected inclusive lower bound, in TAI nanoseconds.
    pub const fn lower_ns(self) -> i128 {
        self.lower_ns
    }

    /// Rejected inclusive upper bound, in TAI nanoseconds.
    pub const fn upper_ns(self) -> i128 {
        self.upper_ns
    }
}

impl fmt::Display for CapabilityValidityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "capability validity interval is inverted: {} > {}",
            self.lower_ns, self.upper_ns
        )
    }
}

impl Error for CapabilityValidityError {}

/// Content identity of a capability claim or signature `SimpleArchive` blob.
pub type CapabilityBlobHandle = Inline<Handle<SimpleArchive>>;

attributes! {
    /// Direct Ed25519 subject receiving the capability.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "FDE6F0937778AE0E1DB227EF1287EFCE" as capability_subject: ED25519PublicKey;
    /// Exact opaque resource identity interpreted by the action.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "39739A88E72B2B219E2E4CFEF204F5E4" as capability_resource: CapabilityResourceEncoding;
    /// Exact uninterpreted action identifier.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "E68BACD3068B30DA051D3A4A2B8795FC" as capability_action: GenId;
    /// Exact nonempty invocation/delegation mode.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "BFA79BC8429F869C461039CFBC303F37" as capability_mode: GenId;
    /// Exact parent signature blob for a delegated occurrence.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "DC08211F1A8F0E9A7C3074A32EC0C515" as capability_parent: Handle<SimpleArchive>;
    /// Optional inclusive interval during which this claim participates.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "3641AFF8C318A1B8F42E3DD6B624C64F" as capability_validity: NsTAIInterval;
    /// Exact claim blob attested to by a capability signature blob.
    /// Anchor minted with `trible genid` on 2026-08-24.
    "0C8B33A0D75D5D39194D55EC96F7038C" as capability_signed_claim: Handle<SimpleArchive>;
}

const CLAIM_REQUIRED_TRIBLES: usize = 5;
const CLAIM_MAX_TRIBLES: usize = 7;
const SIGNATURE_TRIBLES: usize = 5;

/// One canonical capability claim before its issuer signs it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityGrant {
    parent: Option<CapabilityBlobHandle>,
    subject: Inline<ED25519PublicKey>,
    atom: CapabilityAtom,
    mode: CapabilityMode,
    validity: Option<CapabilityValidity>,
}

impl CapabilityGrant {
    /// Construct a trust-root-issued claim.
    pub fn root(
        subject: VerifyingKey,
        atom: CapabilityAtom,
        mode: CapabilityMode,
        validity: Option<CapabilityValidity>,
    ) -> Self {
        Self::new(None, subject, atom, mode, validity)
    }

    /// Construct a delegated claim naming one exact parent signature blob.
    pub fn delegated(
        parent_signature: CapabilityBlobHandle,
        subject: VerifyingKey,
        atom: CapabilityAtom,
        mode: CapabilityMode,
        validity: Option<CapabilityValidity>,
    ) -> Self {
        Self::new(Some(parent_signature), subject, atom, mode, validity)
    }

    fn new(
        parent: Option<CapabilityBlobHandle>,
        subject: VerifyingKey,
        atom: CapabilityAtom,
        mode: CapabilityMode,
        validity: Option<CapabilityValidity>,
    ) -> Self {
        Self {
            parent,
            subject: Inline::new(subject.to_bytes()),
            atom,
            mode,
            validity,
        }
    }

    /// Exact parent signature blob, absent only on a root claim.
    pub fn parent(self) -> Option<CapabilityBlobHandle> {
        self.parent
    }

    /// Direct Ed25519 subject receiving this claim.
    pub fn subject(self) -> VerifyingKey {
        VerifyingKey::from_bytes(&self.subject.raw)
            .expect("CapabilityGrant validates subject bytes when decoded or constructed")
    }

    /// Exact action/resource atom governed by this claim.
    pub const fn atom(self) -> CapabilityAtom {
        self.atom
    }

    /// Invocation/delegation mode carried by this claim.
    pub const fn mode(self) -> CapabilityMode {
        self.mode
    }

    /// Optional inclusive validity interval; `None` is unbounded.
    pub const fn validity(self) -> Option<CapabilityValidity> {
        self.validity
    }

    /// Encode this claim as its canonical content-addressed archive blob.
    pub fn to_blob(self) -> Blob<SimpleArchive> {
        entity! {
            metadata::tag: KIND_CAPABILITY_CLAIM,
            capability_subject: self.subject,
            capability_resource: self.atom.resource,
            capability_action: self.atom.action.id(),
            capability_mode: self.mode.id(),
            capability_parent?: self.parent,
            capability_validity?: self.validity.map(CapabilityValidity::inline),
        }
        .into_facts()
        .to_blob()
    }

    /// Parse one closed canonical capability-claim shape.
    pub fn from_blob(blob: Blob<SimpleArchive>) -> Result<Self, CapabilityGrantDecodeError> {
        decode_grant(&blob)
    }
}

impl TryFromBlob<SimpleArchive> for CapabilityGrant {
    type Error = CapabilityGrantDecodeError;

    fn try_from_blob(blob: Blob<SimpleArchive>) -> Result<Self, Self::Error> {
        Self::from_blob(blob)
    }
}

/// Why a capability claim blob was not one closed canonical grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityGrantDecodeError {
    /// The archive cannot have the closed claim shape at this byte length.
    InvalidLength {
        /// Shortest canonical claim, in bytes.
        min: usize,
        /// Longest canonical claim, in bytes.
        max: usize,
        /// Actual blob length.
        actual: usize,
    },
    /// The bytes were not a canonical `SimpleArchive`.
    Archive(UnarchiveError),
    /// A required field is absent.
    MissingField(&'static str),
    /// A single-valued field occurs more than once.
    RepeatedField(&'static str),
    /// The subject bytes are not an Ed25519 public key.
    InvalidSubject,
    /// An identifier field is nil or not in canonical `GenId` form.
    InvalidId(&'static str),
    /// The stored mode is not one of the three protocol modes.
    InvalidMode,
    /// The optional validity interval is inverted.
    InvalidValidity(CapabilityValidityError),
    /// Extra fields, entities, or a non-intrinsic entity id were present.
    NonCanonicalShape,
}

impl fmt::Display for CapabilityGrantDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { min, max, actual } => write!(
                formatter,
                "capability claim has {actual} bytes; expected {min}..={max}"
            ),
            Self::Archive(error) => write!(formatter, "invalid claim archive: {error}"),
            Self::MissingField(field) => write!(formatter, "capability claim is missing {field}"),
            Self::RepeatedField(field) => write!(formatter, "capability claim repeats {field}"),
            Self::InvalidSubject => {
                formatter.write_str("capability claim subject is not an Ed25519 key")
            }
            Self::InvalidId(field) => write!(formatter, "capability claim has invalid {field}"),
            Self::InvalidMode => formatter.write_str("capability claim has an unknown mode"),
            Self::InvalidValidity(error) => write!(formatter, "{error}"),
            Self::NonCanonicalShape => {
                formatter.write_str("capability claim is not one closed canonical claim entity")
            }
        }
    }
}

impl Error for CapabilityGrantDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Archive(error) => Some(error),
            Self::InvalidValidity(error) => Some(error),
            _ => None,
        }
    }
}

impl From<UnarchiveError> for CapabilityGrantDecodeError {
    fn from(error: UnarchiveError) -> Self {
        Self::Archive(error)
    }
}

fn decode_grant(blob: &Blob<SimpleArchive>) -> Result<CapabilityGrant, CapabilityGrantDecodeError> {
    let min = CLAIM_REQUIRED_TRIBLES * TRIBLE_LEN;
    let max = CLAIM_MAX_TRIBLES * TRIBLE_LEN;
    if !(min..=max).contains(&blob.bytes.len()) {
        return Err(CapabilityGrantDecodeError::InvalidLength {
            min,
            max,
            actual: blob.bytes.len(),
        });
    }
    let facts: TribleSet = TryFromBlob::try_from_blob(blob.clone())?;
    let entity = exactly_one::<_, CapabilityGrantDecodeError>(
        find!(
            (entity: Id),
            pattern!(&facts, [{ ?entity @ metadata::tag: KIND_CAPABILITY_CLAIM }])
        )
        .map(|(entity,)| entity),
        "metadata::tag",
    )?;

    let subject = exactly_one::<_, CapabilityGrantDecodeError>(
        find!(
            (value: Inline<ED25519PublicKey>),
            pattern!(&facts, [{ entity @ capability_subject: ?value }])
        )
        .map(|(value,)| value),
        "capability_subject",
    )?;
    VerifyingKey::from_bytes(&subject.raw)
        .map_err(|_| CapabilityGrantDecodeError::InvalidSubject)?;

    let resource = exactly_one::<_, CapabilityGrantDecodeError>(
        find!(
            (value: Inline<CapabilityResourceEncoding>),
            pattern!(&facts, [{ entity @ capability_resource: ?value }])
        )
        .map(|(value,)| CapabilityResource(value.raw)),
        "capability_resource",
    )?;

    let action = exactly_one::<_, CapabilityGrantDecodeError>(
        find!(
            (value: Inline<GenId>),
            pattern!(&facts, [{ entity @ capability_action: ?value }])
        )
        .map(|(value,)| value),
        "capability_action",
    )?
    .try_from_inline::<Id>()
    .map_err(|_| CapabilityGrantDecodeError::InvalidId("capability_action"))?;

    let mode = exactly_one::<_, CapabilityGrantDecodeError>(
        find!(
            (value: Inline<GenId>),
            pattern!(&facts, [{ entity @ capability_mode: ?value }])
        )
        .map(|(value,)| value),
        "capability_mode",
    )?
    .try_from_inline::<Id>()
    .map_err(|_| CapabilityGrantDecodeError::InvalidId("capability_mode"))?;
    let mode = CapabilityMode::from_id(mode).ok_or(CapabilityGrantDecodeError::InvalidMode)?;

    let parent = at_most_one::<_, CapabilityGrantDecodeError>(
        find!(
            (value: CapabilityBlobHandle),
            pattern!(&facts, [{ entity @ capability_parent: ?value }])
        )
        .map(|(value,)| value),
        "capability_parent",
    )?;

    let validity = at_most_one::<_, CapabilityGrantDecodeError>(
        find!(
            (value: Inline<NsTAIInterval>),
            pattern!(&facts, [{ entity @ capability_validity: ?value }])
        )
        .map(|(value,)| value),
        "capability_validity",
    )?
    .map(CapabilityValidity::from_inline)
    .transpose()
    .map_err(CapabilityGrantDecodeError::InvalidValidity)?;

    let grant = CapabilityGrant {
        parent,
        subject,
        atom: CapabilityAtom::new(CapabilityAction(action), resource),
        mode,
        validity,
    };
    if grant.to_blob().bytes != blob.bytes {
        return Err(CapabilityGrantDecodeError::NonCanonicalShape);
    }
    Ok(grant)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CapabilitySignature {
    claim: CapabilityBlobHandle,
    signer: Inline<ED25519PublicKey>,
    r: Inline<ED25519RComponent>,
    s: Inline<ED25519SComponent>,
}

impl CapabilitySignature {
    fn new(claim: CapabilityBlobHandle, signer: VerifyingKey, signature: Signature) -> Self {
        Self {
            claim,
            signer: Inline::new(signer.to_bytes()),
            r: ED25519RComponent::from_signature(signature),
            s: ED25519SComponent::from_signature(signature),
        }
    }

    fn to_blob(self) -> Blob<SimpleArchive> {
        entity! {
            metadata::tag: KIND_CAPABILITY_SIGNATURE,
            capability_signed_claim: self.claim,
            crate::attestation::signed_by: self.signer,
            crate::attestation::signature_r: self.r,
            crate::attestation::signature_s: self.s,
        }
        .into_facts()
        .to_blob()
    }

    fn signer(self) -> VerifyingKey {
        VerifyingKey::from_bytes(&self.signer.raw)
            .expect("CapabilitySignature validates signer bytes when decoded or constructed")
    }

    fn signature(self) -> Signature {
        Signature::from_components(self.r.raw, self.s.raw)
    }
}

/// Why a capability signature blob was not one closed canonical signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilitySignatureDecodeError {
    /// The archive was not exactly the five-field signature shape in length.
    InvalidLength {
        /// Required byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// The bytes were not a canonical `SimpleArchive`.
    Archive(UnarchiveError),
    /// A required field is absent.
    MissingField(&'static str),
    /// A single-valued field occurs more than once.
    RepeatedField(&'static str),
    /// The signer bytes are not an Ed25519 public key.
    InvalidSigner,
    /// Extra fields, entities, or a non-intrinsic entity id were present.
    NonCanonicalShape,
}

impl fmt::Display for CapabilitySignatureDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => write!(
                formatter,
                "capability signature has {actual} bytes; expected exactly {expected}"
            ),
            Self::Archive(error) => write!(formatter, "invalid signature archive: {error}"),
            Self::MissingField(field) => {
                write!(formatter, "capability signature is missing {field}")
            }
            Self::RepeatedField(field) => {
                write!(formatter, "capability signature repeats {field}")
            }
            Self::InvalidSigner => {
                formatter.write_str("capability signature signer is not an Ed25519 key")
            }
            Self::NonCanonicalShape => formatter
                .write_str("capability signature is not one closed canonical signature entity"),
        }
    }
}

impl Error for CapabilitySignatureDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Archive(error) => Some(error),
            _ => None,
        }
    }
}

impl From<UnarchiveError> for CapabilitySignatureDecodeError {
    fn from(error: UnarchiveError) -> Self {
        Self::Archive(error)
    }
}

fn decode_signature(
    blob: &Blob<SimpleArchive>,
) -> Result<CapabilitySignature, CapabilitySignatureDecodeError> {
    let expected = SIGNATURE_TRIBLES * TRIBLE_LEN;
    if blob.bytes.len() != expected {
        return Err(CapabilitySignatureDecodeError::InvalidLength {
            expected,
            actual: blob.bytes.len(),
        });
    }
    let facts: TribleSet = TryFromBlob::try_from_blob(blob.clone())?;
    let entity = exactly_one::<_, CapabilitySignatureDecodeError>(
        find!(
            (entity: Id),
            pattern!(&facts, [{ ?entity @ metadata::tag: KIND_CAPABILITY_SIGNATURE }])
        )
        .map(|(entity,)| entity),
        "metadata::tag",
    )?;
    let claim = exactly_one::<_, CapabilitySignatureDecodeError>(
        find!(
            (value: CapabilityBlobHandle),
            pattern!(&facts, [{ entity @ capability_signed_claim: ?value }])
        )
        .map(|(value,)| value),
        "capability_signed_claim",
    )?;
    let signer = exactly_one::<_, CapabilitySignatureDecodeError>(
        find!(
            (value: Inline<ED25519PublicKey>),
            pattern!(&facts, [{ entity @ crate::attestation::signed_by: ?value }])
        )
        .map(|(value,)| value),
        "attestation::signed_by",
    )?;
    VerifyingKey::from_bytes(&signer.raw)
        .map_err(|_| CapabilitySignatureDecodeError::InvalidSigner)?;
    let r = exactly_one::<_, CapabilitySignatureDecodeError>(
        find!(
            (value: Inline<ED25519RComponent>),
            pattern!(&facts, [{ entity @ crate::attestation::signature_r: ?value }])
        )
        .map(|(value,)| value),
        "attestation::signature_r",
    )?;
    let s = exactly_one::<_, CapabilitySignatureDecodeError>(
        find!(
            (value: Inline<ED25519SComponent>),
            pattern!(&facts, [{ entity @ crate::attestation::signature_s: ?value }])
        )
        .map(|(value,)| value),
        "attestation::signature_s",
    )?;
    let signature = CapabilitySignature {
        claim,
        signer,
        r,
        s,
    };
    if signature.to_blob().bytes != blob.bytes {
        return Err(CapabilitySignatureDecodeError::NonCanonicalShape);
    }
    Ok(signature)
}

/// One received claim/signature pair in a portable proof.
///
/// Construction is permissive so untrusted bytes can be represented before
/// verification. [`CapabilityProof::verify_claim`] is the admission boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityProofStep {
    claim: Blob<SimpleArchive>,
    signature: Blob<SimpleArchive>,
}

impl CapabilityProofStep {
    /// Pair received claim and signature blobs without trusting them.
    pub fn new(claim: Blob<SimpleArchive>, signature: Blob<SimpleArchive>) -> Self {
        Self { claim, signature }
    }

    /// Canonically encode and sign one grant.
    pub fn issue(issuer: &SigningKey, grant: CapabilityGrant) -> Self {
        let claim = grant.to_blob();
        let claim_handle = content_handle(&claim);
        let signature = issuer.sign(&claim.bytes);
        let signature =
            CapabilitySignature::new(claim_handle, issuer.verifying_key(), signature).to_blob();
        Self { claim, signature }
    }

    /// Candidate claim blob.
    pub fn claim(&self) -> &Blob<SimpleArchive> {
        &self.claim
    }

    /// Candidate signature blob.
    pub fn signature(&self) -> &Blob<SimpleArchive> {
        &self.signature
    }

    /// Recomputed content identity of this step's signature blob.
    pub fn signature_handle(&self) -> CapabilityBlobHandle {
        content_handle(&self.signature)
    }
}

/// Exact leaf authority a caller expects a proof to establish.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityClaim {
    subject: Inline<ED25519PublicKey>,
    atom: CapabilityAtom,
    required: CapabilityMode,
}

impl CapabilityClaim {
    /// Construct an exact subject/action/resource claim with a minimum mode.
    pub fn new(subject: VerifyingKey, atom: CapabilityAtom, required: CapabilityMode) -> Self {
        Self {
            subject: Inline::new(subject.to_bytes()),
            atom,
            required,
        }
    }

    /// Expected leaf subject.
    pub fn subject(self) -> VerifyingKey {
        VerifyingKey::from_bytes(&self.subject.raw)
            .expect("CapabilityClaim is constructed from a valid key")
    }

    /// Expected exact action/resource atom.
    pub const fn atom(self) -> CapabilityAtom {
        self.atom
    }

    /// Minimum mode the leaf must carry.
    pub const fn required(self) -> CapabilityMode {
        self.required
    }

    fn is_satisfied_by(self, grant: CapabilityGrant) -> bool {
        self.subject == grant.subject
            && self.atom == grant.atom
            && grant.mode.satisfies(self.required)
    }
}

/// A root-to-leaf sequence of exact claim/signature blob pairs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityProof {
    steps: Vec<CapabilityProofStep>,
}

impl CapabilityProof {
    /// Construct received evidence in claimed root-to-leaf order.
    pub fn new(steps: Vec<CapabilityProofStep>) -> Self {
        Self { steps }
    }

    /// Claimed root-to-leaf steps.
    pub fn steps(&self) -> &[CapabilityProofStep] {
        &self.steps
    }

    /// Recomputed content identity of the leaf signature blob.
    pub fn credential(&self) -> Option<CapabilityBlobHandle> {
        self.steps.last().map(CapabilityProofStep::signature_handle)
    }

    /// Load a root-to-leaf proof from one leaf signature handle.
    ///
    /// `get_blob` is the only storage adapter: it performs exact-handle lookup
    /// in whatever blob store the caller already uses. The walk follows each
    /// canonical signature to its claim and each claim to its exact parent
    /// signature, stopping only at a root claim and reversing the gathered
    /// steps once. It neither enumerates storage nor imposes a depth limit.
    /// Returned blobs are rehashed rather than trusting cached handles.
    ///
    /// This reconstructs evidence but does not authorize it. Pass the result
    /// to [`Self::verify_claim`] with an external trust root, explicit instant,
    /// and expected leaf claim.
    pub fn load<E>(
        credential: CapabilityBlobHandle,
        mut get_blob: impl FnMut(CapabilityBlobHandle) -> Result<Option<Blob<SimpleArchive>>, E>,
    ) -> Result<Self, CapabilityProofLoadError<E>> {
        let mut next = credential;
        let mut seen = HashSet::new();
        let mut reverse_steps = Vec::new();

        loop {
            if !seen.insert(next) {
                return Err(CapabilityProofLoadError::RepeatedSignature { handle: next });
            }

            let signature_blob = load_exact_blob(next, &mut get_blob)?;
            let signature = decode_signature(&signature_blob).map_err(|source| {
                CapabilityProofLoadError::InvalidSignatureBlob {
                    handle: next,
                    source,
                }
            })?;

            let claim_blob = load_exact_blob(signature.claim, &mut get_blob)?;
            let grant = decode_grant(&claim_blob).map_err(|source| {
                CapabilityProofLoadError::InvalidClaim {
                    handle: signature.claim,
                    source,
                }
            })?;
            reverse_steps.push(CapabilityProofStep::new(claim_blob, signature_blob));

            match grant.parent {
                Some(parent) => next = parent,
                None => break,
            }
        }

        reverse_steps.reverse();
        Ok(Self::new(reverse_steps))
    }

    /// Verify this exact chain against an external trust root and leaf claim.
    ///
    /// Every blob is parsed as a closed canonical shape and named from its
    /// bytes rather than its cached handle. Every signature is verified
    /// strictly over the adjacent claim bytes. The first signature must be by
    /// `trust_root`; each later claim must name the immediately preceding
    /// signature blob and be signed by the preceding subject. The parent's
    /// mode must grant delegation and contain the child's mode. Action and
    /// resource remain byte-exact throughout.
    /// Optional validity intervals are inclusive, and both their lower and
    /// upper bounds are enforced at `instant` for every step.
    pub fn verify_claim(
        &self,
        trust_root: VerifyingKey,
        instant: Epoch,
        expected: CapabilityClaim,
    ) -> Result<VerifiedCapability, CapabilityProofError> {
        if self.steps.is_empty() {
            return Err(CapabilityProofError::Empty);
        }

        let instant_ns = instant.to_tai_duration().total_nanoseconds();
        let trust_root = Inline::<ED25519PublicKey>::new(trust_root.to_bytes());
        let mut previous: Option<(CapabilityGrant, CapabilityBlobHandle)> = None;
        let mut leaf: Option<(CapabilityGrant, CapabilityBlobHandle, CapabilityBlobHandle)> = None;
        let mut effective_validity: Option<(i128, i128)> = None;

        for (step, proof_step) in self.steps.iter().enumerate() {
            let claim_handle = content_handle(&proof_step.claim);
            let signature_handle = content_handle(&proof_step.signature);
            let grant = decode_grant(&proof_step.claim)
                .map_err(|source| CapabilityProofError::InvalidClaim { step, source })?;
            let signature = decode_signature(&proof_step.signature)
                .map_err(|source| CapabilityProofError::InvalidSignatureBlob { step, source })?;

            if signature.claim != claim_handle {
                return Err(CapabilityProofError::SignatureNamesWrongClaim {
                    step,
                    expected: claim_handle,
                    actual: signature.claim,
                });
            }
            signature
                .signer()
                .verify_strict(&proof_step.claim.bytes, &signature.signature())
                .map_err(|_| CapabilityProofError::InvalidSignature { step })?;

            if let Some(validity) = grant.validity {
                let (lower, upper) = validity.bounds_ns();
                if instant_ns < lower {
                    return Err(CapabilityProofError::NotYetValid { step, lower });
                }
                if instant_ns > upper {
                    return Err(CapabilityProofError::Expired { step, upper });
                }
                effective_validity = Some(match effective_validity {
                    Some((effective_lower, effective_upper)) => {
                        (effective_lower.max(lower), effective_upper.min(upper))
                    }
                    None => (lower, upper),
                });
            }

            match previous {
                None => {
                    if grant.parent.is_some() {
                        return Err(CapabilityProofError::WrongParent {
                            step,
                            expected: None,
                            actual: grant.parent,
                        });
                    }
                    if signature.signer != trust_root {
                        return Err(CapabilityProofError::WrongRootSigner { step });
                    }
                }
                Some((parent, parent_signature)) => {
                    if grant.parent != Some(parent_signature) {
                        return Err(CapabilityProofError::WrongParent {
                            step,
                            expected: Some(parent_signature),
                            actual: grant.parent,
                        });
                    }
                    if signature.signer != parent.subject {
                        return Err(CapabilityProofError::IssuerIsNotParentSubject { step });
                    }
                    if !parent.mode.delegates() {
                        return Err(CapabilityProofError::ParentCannotDelegate { step });
                    }
                    if !parent.mode.satisfies(grant.mode) {
                        return Err(CapabilityProofError::ModeEscalation {
                            step,
                            parent: parent.mode,
                            child: grant.mode,
                        });
                    }
                    if grant.atom != parent.atom {
                        return Err(CapabilityProofError::AtomChanged {
                            step,
                            parent: parent.atom,
                            child: grant.atom,
                        });
                    }
                }
            }

            previous = Some((grant, signature_handle));
            leaf = Some((grant, claim_handle, signature_handle));
        }

        let (grant, claim_handle, credential) =
            leaf.expect("a nonempty proof always assigns one leaf");
        if !expected.is_satisfied_by(grant) {
            return Err(CapabilityProofError::ClaimMismatch {
                expected,
                actual: grant,
            });
        }
        Ok(VerifiedCapability {
            grant,
            claim_handle,
            credential,
            effective_validity: effective_validity
                .map(|(lower, upper)| CapabilityValidity::from_bounds_ns(lower, upper)),
        })
    }
}

fn load_exact_blob<E>(
    handle: CapabilityBlobHandle,
    get_blob: &mut impl FnMut(CapabilityBlobHandle) -> Result<Option<Blob<SimpleArchive>>, E>,
) -> Result<Blob<SimpleArchive>, CapabilityProofLoadError<E>> {
    let blob = get_blob(handle)
        .map_err(|source| CapabilityProofLoadError::Get { handle, source })?
        .ok_or(CapabilityProofLoadError::Missing { handle })?;
    let actual = content_handle(&blob);
    if actual != handle {
        return Err(CapabilityProofLoadError::HandleMismatch {
            requested: handle,
            actual,
        });
    }
    Ok(blob)
}

/// Why exact-handle proof reconstruction failed before authorization.
#[derive(Debug)]
pub enum CapabilityProofLoadError<E> {
    /// The caller's blob adapter failed while looking up one exact handle.
    Get {
        /// Requested content handle.
        handle: CapabilityBlobHandle,
        /// Adapter-specific retrieval error.
        source: E,
    },
    /// The caller's blob adapter had no blob for an exact required handle.
    Missing {
        /// Missing content handle.
        handle: CapabilityBlobHandle,
    },
    /// The adapter returned bytes whose recomputed identity was not requested.
    HandleMismatch {
        /// Requested content handle.
        requested: CapabilityBlobHandle,
        /// Identity recomputed from the returned bytes.
        actual: CapabilityBlobHandle,
    },
    /// A required signature blob was not one closed canonical signature.
    InvalidSignatureBlob {
        /// Exact signature handle being followed.
        handle: CapabilityBlobHandle,
        /// Signature parsing failure.
        source: CapabilitySignatureDecodeError,
    },
    /// A required claim blob was not one closed canonical claim.
    InvalidClaim {
        /// Exact claim handle named by its signature.
        handle: CapabilityBlobHandle,
        /// Claim parsing failure.
        source: CapabilityGrantDecodeError,
    },
    /// Following parent edges encountered the same signature handle twice.
    RepeatedSignature {
        /// Repeated signature handle.
        handle: CapabilityBlobHandle,
    },
}

impl<E: fmt::Display> fmt::Display for CapabilityProofLoadError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get { source, .. } => {
                write!(
                    formatter,
                    "capability proof blob retrieval failed: {source}"
                )
            }
            Self::Missing { .. } => formatter.write_str("capability proof requires a missing blob"),
            Self::HandleMismatch { .. } => {
                formatter.write_str("capability proof loader returned bytes under the wrong handle")
            }
            Self::InvalidSignatureBlob { source, .. } => {
                write!(
                    formatter,
                    "capability proof has an invalid signature blob: {source}"
                )
            }
            Self::InvalidClaim { source, .. } => {
                write!(
                    formatter,
                    "capability proof has an invalid claim blob: {source}"
                )
            }
            Self::RepeatedSignature { .. } => {
                formatter.write_str("capability proof parent walk repeated a signature handle")
            }
        }
    }
}

impl<E: Error + 'static> Error for CapabilityProofLoadError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Get { source, .. } => Some(source),
            Self::InvalidSignatureBlob { source, .. } => Some(source),
            Self::InvalidClaim { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The exact verified leaf and its content identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifiedCapability {
    grant: CapabilityGrant,
    claim_handle: CapabilityBlobHandle,
    credential: CapabilityBlobHandle,
    effective_validity: Option<CapabilityValidity>,
}

impl VerifiedCapability {
    /// Verified canonical leaf grant.
    pub const fn grant(self) -> CapabilityGrant {
        self.grant
    }

    /// Recomputed content identity of the verified leaf claim blob.
    pub const fn claim_handle(self) -> CapabilityBlobHandle {
        self.claim_handle
    }

    /// Recomputed content identity of the verified leaf signature blob.
    pub const fn credential(self) -> CapabilityBlobHandle {
        self.credential
    }

    /// Intersection of every bounded interval in the verified chain.
    ///
    /// `None` means every step was unbounded. The explicit verification
    /// instant was inside both inclusive bounds of the returned interval.
    pub const fn effective_validity(self) -> Option<CapabilityValidity> {
        self.effective_validity
    }
}

/// Why standalone, claim-directed capability verification failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapabilityProofError {
    /// A proof must contain a root-issued occurrence.
    Empty,
    /// One claim was not a closed canonical capability grant.
    InvalidClaim {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Claim parsing failure.
        source: CapabilityGrantDecodeError,
    },
    /// One signature blob was not the exact closed canonical shape.
    InvalidSignatureBlob {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Signature parsing failure.
        source: CapabilitySignatureDecodeError,
    },
    /// A signature blob named a different claim handle than its adjacent blob.
    SignatureNamesWrongClaim {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Recomputed adjacent claim identity.
        expected: CapabilityBlobHandle,
        /// Claim identity encoded in the signature blob.
        actual: CapabilityBlobHandle,
    },
    /// Strict Ed25519 verification rejected one signature.
    InvalidSignature {
        /// Zero-based root-to-leaf step index.
        step: usize,
    },
    /// The explicit instant precedes one claim's inclusive lower bound.
    NotYetValid {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Inclusive lower bound in TAI nanoseconds.
        lower: i128,
    },
    /// The explicit instant follows one claim's inclusive upper bound.
    Expired {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Inclusive upper bound in TAI nanoseconds.
        upper: i128,
    },
    /// The first claim unexpectedly named a parent, or a child did not name
    /// the immediately preceding signature blob.
    WrongParent {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Required exact parent signature handle.
        expected: Option<CapabilityBlobHandle>,
        /// Parent handle encoded by the claim.
        actual: Option<CapabilityBlobHandle>,
    },
    /// The first valid signature was not made by the supplied trust root.
    WrongRootSigner {
        /// Zero-based root-to-leaf step index.
        step: usize,
    },
    /// A child was not signed by its exact parent's subject.
    IssuerIsNotParentSubject {
        /// Zero-based root-to-leaf step index.
        step: usize,
    },
    /// A child followed a parent that carried invocation only.
    ParentCannotDelegate {
        /// Zero-based root-to-leaf step index.
        step: usize,
    },
    /// A child requested an invocation/delegation use absent from its parent.
    ModeEscalation {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Mode carried by the exact parent occurrence.
        parent: CapabilityMode,
        /// Escalating mode carried by the child.
        child: CapabilityMode,
    },
    /// A child changed its parent's exact action/resource atom.
    AtomChanged {
        /// Zero-based root-to-leaf step index.
        step: usize,
        /// Parent's exact atom.
        parent: CapabilityAtom,
        /// Child's exact atom.
        child: CapabilityAtom,
    },
    /// The valid chain's leaf did not match the caller's exact subject, atom,
    /// and minimum mode.
    ClaimMismatch {
        /// Exact authority requested by the caller.
        expected: CapabilityClaim,
        /// Canonical but unsuitable leaf grant.
        actual: CapabilityGrant,
    },
}

impl fmt::Display for CapabilityProofError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("capability proof is empty"),
            Self::InvalidClaim { step, source } => {
                write!(
                    formatter,
                    "capability proof step {step} has an invalid claim: {source}"
                )
            }
            Self::InvalidSignatureBlob { step, source } => write!(
                formatter,
                "capability proof step {step} has an invalid signature blob: {source}"
            ),
            Self::SignatureNamesWrongClaim { step, .. } => write!(
                formatter,
                "capability proof step {step} signature names a different claim"
            ),
            Self::InvalidSignature { step } => {
                write!(
                    formatter,
                    "capability proof step {step} has an invalid signature"
                )
            }
            Self::NotYetValid { step, lower } => write!(
                formatter,
                "capability proof step {step} is not valid before TAI nanosecond {lower}"
            ),
            Self::Expired { step, upper } => write!(
                formatter,
                "capability proof step {step} expired after TAI nanosecond {upper}"
            ),
            Self::WrongParent { step, .. } => write!(
                formatter,
                "capability proof step {step} does not name its exact predecessor"
            ),
            Self::WrongRootSigner { step } => write!(
                formatter,
                "capability proof step {step} was not signed by the supplied trust root"
            ),
            Self::IssuerIsNotParentSubject { step } => write!(
                formatter,
                "capability proof step {step} issuer is not its parent subject"
            ),
            Self::ParentCannotDelegate { step } => write!(
                formatter,
                "capability proof step {step} follows a non-delegating parent"
            ),
            Self::ModeEscalation { step, .. } => write!(
                formatter,
                "capability proof step {step} requests authority absent from its parent"
            ),
            Self::AtomChanged { step, .. } => write!(
                formatter,
                "capability proof step {step} changed its parent's exact atom"
            ),
            Self::ClaimMismatch { .. } => formatter.write_str(
                "capability proof leaf does not satisfy the caller's exact subject, atom, and mode",
            ),
        }
    }
}

impl Error for CapabilityProofError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidClaim { source, .. } => Some(source),
            Self::InvalidSignatureBlob { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn content_handle(blob: &Blob<SimpleArchive>) -> CapabilityBlobHandle {
    Inline::new(Blake3::digest(&blob.bytes))
}

fn exactly_one<T, E>(mut rows: impl Iterator<Item = T>, field: &'static str) -> Result<T, E>
where
    E: FieldDecodeError,
{
    let first = rows.next().ok_or_else(|| E::missing(field))?;
    if rows.next().is_some() {
        return Err(E::repeated(field));
    }
    Ok(first)
}

fn at_most_one<T, E>(mut rows: impl Iterator<Item = T>, field: &'static str) -> Result<Option<T>, E>
where
    E: FieldDecodeError,
{
    let first = rows.next();
    if rows.next().is_some() {
        return Err(E::repeated(field));
    }
    Ok(first)
}

trait FieldDecodeError {
    fn missing(field: &'static str) -> Self;
    fn repeated(field: &'static str) -> Self;
}

impl FieldDecodeError for CapabilityGrantDecodeError {
    fn missing(field: &'static str) -> Self {
        Self::MissingField(field)
    }

    fn repeated(field: &'static str) -> Self {
        Self::RepeatedField(field)
    }
}

impl FieldDecodeError for CapabilitySignatureDecodeError {
    fn missing(field: &'static str) -> Self {
        Self::MissingField(field)
    }

    fn repeated(field: &'static str) -> Self {
        Self::RepeatedField(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inline::encodings::boolean::Boolean;
    use anybytes::Bytes;
    use std::collections::HashMap;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn action(byte: u8) -> CapabilityAction {
        CapabilityAction::new(Id::new([byte; 16]).expect("nonzero test action"))
    }

    fn atom(action_byte: u8, resource_byte: u8) -> CapabilityAtom {
        CapabilityAtom::new(
            action(action_byte),
            CapabilityResource::new([resource_byte; 32]),
        )
    }

    fn epoch(seconds: f64) -> Epoch {
        Epoch::from_tai_seconds(seconds)
    }

    fn validity(lower: f64, upper: f64) -> CapabilityValidity {
        CapabilityValidity::new(epoch(lower), epoch(upper)).expect("ordered test interval")
    }

    fn root_step(
        root: &SigningKey,
        subject: VerifyingKey,
        atom: CapabilityAtom,
        mode: CapabilityMode,
        validity: Option<CapabilityValidity>,
    ) -> CapabilityProofStep {
        CapabilityProofStep::issue(root, CapabilityGrant::root(subject, atom, mode, validity))
    }

    #[test]
    fn resource_wrapper_is_exactly_32_bytes_and_erases_inline_type() {
        let typed = Inline::<Boolean>::new([0xA5; 32]);
        let resource = CapabilityResource::from(typed);
        assert_eq!(std::mem::size_of::<CapabilityResource>(), 32);
        assert_eq!(resource.into_bytes(), [0xA5; 32]);
    }

    #[test]
    fn root_claim_and_signature_round_trip() {
        let root = key(1);
        let subject = key(2);
        let atom = atom(3, 4);
        let step = root_step(
            &root,
            subject.verifying_key(),
            atom,
            CapabilityMode::Invoke,
            None,
        );
        let proof = CapabilityProof::new(vec![step.clone()]);
        let verified = proof
            .verify_claim(
                root.verifying_key(),
                epoch(100.0),
                CapabilityClaim::new(subject.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect("root-issued proof verifies");

        assert_eq!(verified.grant().subject(), subject.verifying_key());
        assert_eq!(verified.claim_handle(), content_handle(step.claim()));
        assert_eq!(verified.credential(), step.signature_handle());
        assert_eq!(proof.credential(), Some(verified.credential()));
        assert_eq!(verified.effective_validity(), None);
        assert_eq!(
            CapabilityGrant::from_blob(step.claim().clone()).expect("decode claim"),
            verified.grant()
        );
        decode_signature(step.signature()).expect("decode signature");
    }

    #[test]
    fn loader_walks_exact_blobs_from_leaf_handle_and_preserves_failures() {
        let root = key(5);
        let issuer = key(6);
        let leaf = key(7);
        let atom = atom(8, 9);
        let parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );
        let child = CapabilityProofStep::issue(
            &issuer,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
        );
        let proof = CapabilityProof::new(vec![parent, child]);
        let credential = proof.credential().expect("nonempty proof");
        let mut blobs = HashMap::new();
        for step in proof.steps() {
            blobs.insert(content_handle(step.claim()), step.claim().clone());
            blobs.insert(step.signature_handle(), step.signature().clone());
        }

        let loaded = CapabilityProof::load(credential, |handle| {
            Ok::<_, &'static str>(blobs.get(&handle).cloned())
        })
        .expect("exact-handle walk reconstructs root-to-leaf evidence");
        assert_eq!(loaded, proof);
        loaded
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect("loaded evidence still requires and passes verification");

        assert!(matches!(
            CapabilityProof::load::<Infallible>(credential, |_| Ok(None)),
            Err(CapabilityProofLoadError::Missing { handle }) if handle == credential
        ));
        assert!(matches!(
            CapabilityProof::load(credential, |_| {
                Err::<Option<Blob<SimpleArchive>>, _>("offline")
            }),
            Err(CapabilityProofLoadError::Get {
                handle,
                source: "offline",
            }) if handle == credential
        ));

        let wrong_blob = proof.steps()[0].claim().clone();
        assert!(matches!(
            CapabilityProof::load::<Infallible>(credential, |_| Ok(Some(wrong_blob.clone()))),
            Err(CapabilityProofLoadError::HandleMismatch { requested, .. })
                if requested == credential
        ));
    }

    #[test]
    fn delegated_chain_is_exact_and_combined_mode_satisfies_each_use() {
        let root = key(10);
        let delegate = key(11);
        let leaf = key(12);
        let atom = atom(13, 14);
        let parent = root_step(
            &root,
            delegate.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );
        let child = CapabilityProofStep::issue(
            &delegate,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::InvokeAndDelegate,
                None,
            ),
        );
        let proof = CapabilityProof::new(vec![parent, child]);

        for required in [CapabilityMode::Invoke, CapabilityMode::Delegate] {
            proof
                .verify_claim(
                    root.verifying_key(),
                    epoch(0.0),
                    CapabilityClaim::new(leaf.verifying_key(), atom, required),
                )
                .expect("the combined mode contains each individual use");
        }
        let verified = proof
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(
                    leaf.verifying_key(),
                    atom,
                    CapabilityMode::InvokeAndDelegate,
                ),
            )
            .expect("the combined mode satisfies the combined requirement");
        assert_eq!(verified.grant().mode(), CapabilityMode::InvokeAndDelegate);
        assert!(!CapabilityMode::Delegate.satisfies(CapabilityMode::Invoke));
        assert!(!CapabilityMode::Invoke.satisfies(CapabilityMode::Delegate));
        assert!(CapabilityMode::InvokeAndDelegate.satisfies(CapabilityMode::InvokeAndDelegate));
    }

    #[test]
    fn invoke_only_parent_cannot_delegate() {
        let root = key(20);
        let issuer = key(21);
        let leaf = key(22);
        let atom = atom(23, 24);
        let parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::Invoke,
            None,
        );
        let child = CapabilityProofStep::issue(
            &issuer,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
        );
        let error = CapabilityProof::new(vec![parent, child])
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect_err("invoke-only parent cannot issue a child");
        assert!(matches!(
            error,
            CapabilityProofError::ParentCannotDelegate { step: 1 }
        ));
    }

    #[test]
    fn child_mode_must_attenuate_parent_mode() {
        let root = key(25);
        let issuer = key(26);
        let leaf = key(27);
        let atom = atom(28, 29);

        let delegate_only_parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::Delegate,
            None,
        );
        for child_mode in [CapabilityMode::Invoke, CapabilityMode::InvokeAndDelegate] {
            let child = CapabilityProofStep::issue(
                &issuer,
                CapabilityGrant::delegated(
                    delegate_only_parent.signature_handle(),
                    leaf.verifying_key(),
                    atom,
                    child_mode,
                    None,
                ),
            );
            assert!(matches!(
                CapabilityProof::new(vec![delegate_only_parent.clone(), child]).verify_claim(
                    root.verifying_key(),
                    epoch(0.0),
                    CapabilityClaim::new(leaf.verifying_key(), atom, child_mode),
                ),
                Err(CapabilityProofError::ModeEscalation {
                    step: 1,
                    parent: CapabilityMode::Delegate,
                    child,
                }) if child == child_mode
            ));
        }

        let full_parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );
        for child_mode in [
            CapabilityMode::Invoke,
            CapabilityMode::Delegate,
            CapabilityMode::InvokeAndDelegate,
        ] {
            let child = CapabilityProofStep::issue(
                &issuer,
                CapabilityGrant::delegated(
                    full_parent.signature_handle(),
                    leaf.verifying_key(),
                    atom,
                    child_mode,
                    None,
                ),
            );
            CapabilityProof::new(vec![full_parent.clone(), child])
                .verify_claim(
                    root.verifying_key(),
                    epoch(0.0),
                    CapabilityClaim::new(leaf.verifying_key(), atom, child_mode),
                )
                .expect("the combined mode may attenuate to any nonempty subset");
        }
    }

    #[test]
    fn child_cannot_change_action_or_resource() {
        let root = key(30);
        let issuer = key(31);
        let leaf = key(32);
        let parent_atom = atom(33, 34);
        let parent = root_step(
            &root,
            issuer.verifying_key(),
            parent_atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );

        for child_atom in [atom(35, 34), atom(33, 36)] {
            let child = CapabilityProofStep::issue(
                &issuer,
                CapabilityGrant::delegated(
                    parent.signature_handle(),
                    leaf.verifying_key(),
                    child_atom,
                    CapabilityMode::Invoke,
                    None,
                ),
            );
            let error = CapabilityProof::new(vec![parent.clone(), child])
                .verify_claim(
                    root.verifying_key(),
                    epoch(0.0),
                    CapabilityClaim::new(leaf.verifying_key(), child_atom, CapabilityMode::Invoke),
                )
                .expect_err("atom changes fail closed");
            assert!(matches!(
                error,
                CapabilityProofError::AtomChanged { step: 1, .. }
            ));
        }
    }

    #[test]
    fn child_names_the_exact_parent_signature_blob() {
        let root = key(40);
        let issuer = key(41);
        let leaf = key(42);
        let atom = atom(43, 44);
        let actual_parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );
        let other_parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            Some(validity(0.0, 200.0)),
        );
        let child = CapabilityProofStep::issue(
            &issuer,
            CapabilityGrant::delegated(
                other_parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
        );
        let error = CapabilityProof::new(vec![actual_parent, child])
            .verify_claim(
                root.verifying_key(),
                epoch(100.0),
                CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect_err("a different valid parent occurrence is not interchangeable");
        assert!(matches!(
            error,
            CapabilityProofError::WrongParent { step: 1, .. }
        ));
    }

    #[test]
    fn external_trust_root_and_parent_subject_are_enforced() {
        let root = key(50);
        let issuer = key(51);
        let attacker = key(52);
        let leaf = key(53);
        let atom = atom(54, 55);
        let parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );

        let wrong_root_error = CapabilityProof::new(vec![parent.clone()])
            .verify_claim(
                attacker.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(issuer.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect_err("trust root is supplied by the verifier");
        assert!(matches!(
            wrong_root_error,
            CapabilityProofError::WrongRootSigner { step: 0 }
        ));

        let child = CapabilityProofStep::issue(
            &attacker,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
        );
        let error = CapabilityProof::new(vec![parent, child])
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect_err("public proof possession is not the parent's private key");
        assert!(matches!(
            error,
            CapabilityProofError::IssuerIsNotParentSubject { step: 1 }
        ));
    }

    #[test]
    fn both_inclusive_validity_bounds_are_enforced_on_every_step() {
        let root = key(60);
        let issuer = key(61);
        let leaf = key(62);
        let atom = atom(63, 64);
        let parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            Some(validity(10.0, 20.0)),
        );
        let child = CapabilityProofStep::issue(
            &issuer,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                Some(validity(5.0, 25.0)),
            ),
        );
        let proof = CapabilityProof::new(vec![parent, child]);
        let claim = CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke);

        for instant in [10.0, 20.0] {
            let verified = proof
                .verify_claim(root.verifying_key(), epoch(instant), claim)
                .expect("both endpoints are inclusive");
            let (lower, upper) = verified
                .effective_validity()
                .expect("bounded chain has an effective interval")
                .bounds();
            assert_eq!(
                lower.to_tai_duration().total_nanoseconds(),
                epoch(10.0).to_tai_duration().total_nanoseconds()
            );
            assert_eq!(
                upper.to_tai_duration().total_nanoseconds(),
                epoch(20.0).to_tai_duration().total_nanoseconds()
            );
        }
        assert!(matches!(
            proof.verify_claim(root.verifying_key(), epoch(9.0), claim),
            Err(CapabilityProofError::NotYetValid { step: 0, .. })
        ));
        assert!(matches!(
            proof.verify_claim(root.verifying_key(), epoch(21.0), claim),
            Err(CapabilityProofError::Expired { step: 0, .. })
        ));
    }

    #[test]
    fn explicit_leaf_claim_prevents_truncated_prefix_substitution() {
        let root = key(70);
        let issuer = key(71);
        let leaf = key(72);
        let atom = atom(73, 74);
        let parent = root_step(
            &root,
            issuer.verifying_key(),
            atom,
            CapabilityMode::InvokeAndDelegate,
            None,
        );
        let child = CapabilityProofStep::issue(
            &issuer,
            CapabilityGrant::delegated(
                parent.signature_handle(),
                leaf.verifying_key(),
                atom,
                CapabilityMode::Invoke,
                None,
            ),
        );
        let full = CapabilityProof::new(vec![parent.clone(), child]);
        full.verify_claim(
            root.verifying_key(),
            epoch(0.0),
            CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke),
        )
        .expect("full proof verifies");

        let truncated = CapabilityProof::new(vec![parent]);
        let error = truncated
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(leaf.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect_err("a valid prefix proves only its own subject");
        assert!(matches!(error, CapabilityProofError::ClaimMismatch { .. }));
    }

    #[test]
    fn signature_is_bound_to_exact_claim_bytes_and_strictly_verified() {
        let root = key(80);
        let subject = key(81);
        let attacker = key(82);
        let expected_atom = atom(83, 84);
        let step = root_step(
            &root,
            subject.verifying_key(),
            expected_atom,
            CapabilityMode::Invoke,
            None,
        );

        let wrong_claim = CapabilityGrant::root(
            subject.verifying_key(),
            atom(83, 85),
            CapabilityMode::Invoke,
            None,
        )
        .to_blob();
        let wrong_claim_step = CapabilityProofStep::new(wrong_claim, step.signature.clone());
        assert!(matches!(
            CapabilityProof::new(vec![wrong_claim_step]).verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(
                    subject.verifying_key(),
                    expected_atom,
                    CapabilityMode::Invoke,
                ),
            ),
            Err(CapabilityProofError::SignatureNamesWrongClaim { step: 0, .. })
        ));

        let bad_signature = attacker.sign(&step.claim.bytes);
        let parsed = decode_signature(&step.signature).expect("canonical signature");
        let bad_signature_blob =
            CapabilitySignature::new(parsed.claim, root.verifying_key(), bad_signature).to_blob();
        assert!(matches!(
            CapabilityProof::new(vec![CapabilityProofStep::new(
                step.claim,
                bad_signature_blob,
            )])
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(
                    subject.verifying_key(),
                    expected_atom,
                    CapabilityMode::Invoke,
                ),
            ),
            Err(CapabilityProofError::InvalidSignature { step: 0 })
        ));
    }

    #[test]
    fn cached_blob_handles_are_never_trusted() {
        let root = key(90);
        let subject = key(91);
        let atom = atom(92, 93);
        let step = root_step(
            &root,
            subject.verifying_key(),
            atom,
            CapabilityMode::Invoke,
            None,
        );
        let bogus = Inline::new([0xFF; 32]);
        let claim = Blob::with_handle(step.claim.bytes.clone(), bogus);
        let signature = Blob::with_handle(step.signature.bytes.clone(), bogus);
        let proof = CapabilityProof::new(vec![CapabilityProofStep::new(claim, signature)]);

        let verified = proof
            .verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(subject.verifying_key(), atom, CapabilityMode::Invoke),
            )
            .expect("raw bytes, not cached handles, define identity");
        assert_ne!(verified.claim_handle(), bogus);
        assert_ne!(verified.credential(), bogus);
    }

    #[test]
    fn claim_and_signature_parsers_reject_open_shapes() {
        let root = key(100);
        let subject = key(101);
        let atom = atom(102, 103);
        let step = root_step(
            &root,
            subject.verifying_key(),
            atom,
            CapabilityMode::Invoke,
            None,
        );

        let mut claim_facts: TribleSet =
            TryFromBlob::try_from_blob(step.claim.clone()).expect("canonical claim");
        let claim_entity = exactly_one::<_, CapabilityGrantDecodeError>(
            find!(
                (entity: Id),
                pattern!(&claim_facts, [{ ?entity @ metadata::tag: KIND_CAPABILITY_CLAIM }])
            )
            .map(|(entity,)| entity),
            "metadata::tag",
        )
        .unwrap();
        claim_facts += entity! {
            ExclusiveId::force_ref(&claim_entity) @
            metadata::tag: metadata::KIND_MULTI,
        };
        assert!(matches!(
            CapabilityGrant::from_blob(claim_facts.to_blob()),
            Err(CapabilityGrantDecodeError::NonCanonicalShape)
                | Err(CapabilityGrantDecodeError::InvalidLength { .. })
        ));

        let mut signature_facts: TribleSet =
            TryFromBlob::try_from_blob(step.signature).expect("canonical signature");
        let signature_entity = exactly_one::<_, CapabilitySignatureDecodeError>(
            find!(
                (entity: Id),
                pattern!(&signature_facts, [{ ?entity @ metadata::tag: KIND_CAPABILITY_SIGNATURE }])
            )
            .map(|(entity,)| entity),
            "metadata::tag",
        )
        .unwrap();
        signature_facts += entity! {
            ExclusiveId::force_ref(&signature_entity) @
            metadata::tag: metadata::KIND_MULTI,
        };
        assert!(matches!(
            decode_signature(&signature_facts.to_blob()),
            Err(CapabilitySignatureDecodeError::InvalidLength { .. })
                | Err(CapabilitySignatureDecodeError::NonCanonicalShape)
        ));
    }

    #[test]
    fn parser_rejects_noncanonical_archive_ordering_before_shape() {
        let root = key(110);
        let subject = key(111);
        let step = root_step(
            &root,
            subject.verifying_key(),
            atom(112, 113),
            CapabilityMode::InvokeAndDelegate,
            Some(validity(0.0, 1.0)),
        );
        let mut rows: Vec<[u8; TRIBLE_LEN]> = step
            .claim
            .bytes
            .chunks_exact(TRIBLE_LEN)
            .map(|row| row.try_into().unwrap())
            .collect();
        rows.reverse();
        let malformed = Blob::<SimpleArchive>::new(Bytes::from(rows));
        assert!(matches!(
            CapabilityGrant::from_blob(malformed),
            Err(CapabilityGrantDecodeError::Archive(
                UnarchiveError::BadCanonicalizationOrdering
            ))
        ));
    }

    #[test]
    fn validity_constructor_rejects_inversion() {
        let error = CapabilityValidity::new(epoch(2.0), epoch(1.0))
            .expect_err("inverted intervals are not constructible");
        assert!(error.lower_ns() > error.upper_ns());
    }

    #[test]
    fn empty_proof_is_rejected() {
        let root = key(120);
        let subject = key(121);
        let atom = atom(122, 123);
        assert!(matches!(
            CapabilityProof::new(Vec::new()).verify_claim(
                root.verifying_key(),
                epoch(0.0),
                CapabilityClaim::new(subject.verifying_key(), atom, CapabilityMode::Invoke),
            ),
            Err(CapabilityProofError::Empty)
        ));
    }
}
