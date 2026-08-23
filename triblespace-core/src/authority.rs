//! Positive, enumerable authority over collection commits.
//!
//! Authority is a grow-only set of signed grant occurrences. Each occurrence
//! is the data of exactly one [`CollectionCommit`](crate::collection::CollectionCommit):
//! the outer commit signer is the issuer and the commit's intrinsic id is the
//! grant occurrence id. A
//! grant names one subject key, one action, one exact collection resource, and
//! at most one parent occurrence. Independent signed occurrences provide OR;
//! there is no permission hierarchy, negative fact, ambient clock, or mutable
//! membership head in this kernel.
//!
//! The resolver deliberately inspects candidate commits one by one. Passing an
//! attacker-controlled candidate set to an all-or-nothing collection snapshot
//! would let one validly signed record naming a missing blob suppress every
//! otherwise usable grant. Here such a candidate remains inert and diagnostic,
//! while independently complete candidates continue through the positive
//! fixed point.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::{Blob, TryFromBlob};
use crate::collection::simplearchive_union::{
    self, PublicationError, SimpleArchiveUnionValidationError,
};
use crate::collection::{
    discover_collection_records_authorized, empty_metadata_handle, CollectionCommit,
    CollectionData, CollectionDiscoveryError, CollectionHandle, CollectionName,
    CollectionRecordDiagnostic, CollectionStore,
};
use crate::id::{id_hex, Id};
use crate::inline::encodings::boolean::Boolean;
use crate::inline::encodings::ed25519::ED25519PublicKey;
use crate::inline::encodings::genid::GenId;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::metadata;
use crate::prelude::{attributes, entity, find, pattern};
use crate::repo::{BlobStore, BlobStoreGet, BlobStorePut};
use crate::trible::{Fragment, TribleSet};

/// Stable kind of one atomic positive authority grant.
///
/// Minted with `trible genid` on 2026-08-22.
pub const KIND_AUTHORITY_GRANT: Id = id_hex!("411A564F0ED4EA6B577C9F9E2B492600");

/// The action required to contribute a signed commit to a collection.
///
/// Minted with `trible genid` on 2026-08-22.
pub const ACTION_WRITE: Id = id_hex!("66B660A5481E04E552A1FA96AA9ECC48");

/// Stable name of the public authority collection rooted in each team.
pub const AUTHORITY_COLLECTION_NAME: &str = "authority";

attributes! {
    /// Direct public-key principal receiving this grant.
    /// Minted with `trible genid` on 2026-08-22.
    "194BCB6BD8F229EBF43028F7E6818144" as pub authority_subject: ED25519PublicKey;
    /// Exact collection descriptor this grant governs.
    /// Minted with `trible genid` on 2026-08-22.
    "40E42C8164A930E19231AE8E3B647FB3" as pub authority_resource: Handle<SimpleArchive>;
    /// One uninterpreted action id; no action implicitly contains another.
    /// Minted with `trible genid` on 2026-08-22.
    "06FFCE24DE393E3F03160341C9EBE9FC" as pub authority_action: GenId;
    /// Exact parent grant occurrence, identified by its collection commit id.
    /// Minted with `trible genid` on 2026-08-22.
    "CA3AF10504A5DB286A8E5276B1451CE7" as pub authority_parent: GenId;
    /// Whether the subject may invoke the named action on the resource.
    /// Minted with `trible genid` on 2026-08-22.
    "BFD270009322755EDE43BBD9E2DAA400" as pub authority_invoke: Boolean;
    /// Whether the subject may issue attenuated child grants for this atom.
    /// Minted with `trible genid` on 2026-08-22.
    "032A475D8C019F1548995D899D9425B1" as pub authority_delegate: Boolean;
}

/// One atomic positive grant carried by one signed authority commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorityGrant {
    parent: Option<Id>,
    subject: Inline<ED25519PublicKey>,
    resource: CollectionHandle,
    action: Id,
    mode: AuthorityMode,
}

/// Nonempty authority carried by one grant occurrence.
///
/// Invocation and delegation are independent. This enum keeps the invalid
/// `(false, false)` pair out of the construction API while retaining the two
/// explicit canonical booleans in the persisted grant entity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorityMode {
    /// The subject may invoke the action but not issue child grants.
    Invoke,
    /// The subject may issue child grants but not invoke the action.
    Delegate,
    /// The subject may both invoke the action and issue child grants.
    InvokeAndDelegate,
}

impl AuthorityMode {
    fn from_flags(invoke: bool, delegate: bool) -> Result<Self, AuthorityGrantDecodeError> {
        match (invoke, delegate) {
            (true, false) => Ok(Self::Invoke),
            (false, true) => Ok(Self::Delegate),
            (true, true) => Ok(Self::InvokeAndDelegate),
            (false, false) => Err(AuthorityGrantDecodeError::EmptyGrant),
        }
    }

    /// Whether this mode grants invocation.
    pub fn invokes(self) -> bool {
        matches!(self, Self::Invoke | Self::InvokeAndDelegate)
    }

    /// Whether this mode grants delegation.
    pub fn delegates(self) -> bool {
        matches!(self, Self::Delegate | Self::InvokeAndDelegate)
    }
}

impl AuthorityGrant {
    /// Construct a grant issued directly by the team's root key.
    pub fn root(
        subject: VerifyingKey,
        resource: CollectionHandle,
        action: Id,
        mode: AuthorityMode,
    ) -> Self {
        Self::new(None, subject, resource, action, mode)
    }

    /// Construct a grant delegated from one exact accepted grant occurrence.
    pub fn delegated(
        parent: Id,
        subject: VerifyingKey,
        resource: CollectionHandle,
        action: Id,
        mode: AuthorityMode,
    ) -> Self {
        Self::new(Some(parent), subject, resource, action, mode)
    }

    fn new(
        parent: Option<Id>,
        subject: VerifyingKey,
        resource: CollectionHandle,
        action: Id,
        mode: AuthorityMode,
    ) -> Self {
        Self {
            parent,
            subject: Inline::new(subject.to_bytes()),
            resource,
            action,
            mode,
        }
    }

    /// Exact parent occurrence, or `None` for a root-issued grant.
    pub fn parent(&self) -> Option<Id> {
        self.parent
    }

    /// Direct public-key principal receiving this grant.
    pub fn subject(&self) -> Inline<ED25519PublicKey> {
        self.subject
    }

    /// Exact collection descriptor this grant governs.
    pub fn resource(&self) -> CollectionHandle {
        self.resource
    }

    /// Uninterpreted action id governed by this grant.
    pub fn action(&self) -> Id {
        self.action
    }

    /// Whether the subject may invoke the action.
    pub fn invoke(&self) -> bool {
        self.mode.invokes()
    }

    /// Whether the subject may issue child grants for this same atom.
    pub fn delegate(&self) -> bool {
        self.mode.delegates()
    }

    /// Nonempty invocation/delegation mode carried by this grant.
    pub fn mode(&self) -> AuthorityMode {
        self.mode
    }

    /// Canonical one-entity fragment signed as this grant's collection data.
    ///
    /// Protocol grant commits always use empty metadata. The attribute
    /// descriptions produced while constructing the entity are intentionally
    /// not signed as metadata: they do not affect authority semantics, and
    /// allowing metadata variation would create distinct occurrences of the
    /// same semantic grant.
    pub fn fragment(&self) -> Fragment {
        let built = entity! {
            metadata::tag: KIND_AUTHORITY_GRANT,
            authority_subject: self.subject,
            authority_resource: self.resource,
            authority_action: self.action,
            authority_parent?: self.parent,
            authority_invoke: self.mode.invokes(),
            authority_delegate: self.mode.delegates(),
        };
        let root = built
            .root()
            .expect("one intrinsic authority entity exports one root");
        Fragment::rooted(root, built.into_facts())
    }

    fn decode(facts: &TribleSet) -> Result<Self, AuthorityGrantDecodeError> {
        let entity = exactly_one(
            find!(
                (entity: Id),
                pattern!(facts, [{ ?entity @ metadata::tag: KIND_AUTHORITY_GRANT }])
            )
            .map(|(entity,)| entity),
            "metadata::tag",
        )?;

        let subject = exactly_one(
            find!(
                (value: Inline<ED25519PublicKey>),
                pattern!(facts, [{ entity @ authority_subject: ?value }])
            )
            .map(|(value,)| value),
            "authority_subject",
        )?;
        VerifyingKey::from_bytes(&subject.raw)
            .map_err(|_| AuthorityGrantDecodeError::InvalidSubject)?;

        let resource = exactly_one(
            find!(
                (value: CollectionHandle),
                pattern!(facts, [{ entity @ authority_resource: ?value }])
            )
            .map(|(value,)| value),
            "authority_resource",
        )?;

        let action = exactly_one(
            find!(
                (value: Inline<GenId>),
                pattern!(facts, [{ entity @ authority_action: ?value }])
            )
            .map(|(value,)| value),
            "authority_action",
        )?
        .try_from_inline::<Id>()
        .map_err(|_| AuthorityGrantDecodeError::InvalidId("authority_action"))?;

        let parent = at_most_one(
            find!(
                (value: Inline<GenId>),
                pattern!(facts, [{ entity @ authority_parent: ?value }])
            )
            .map(|(value,)| value),
            "authority_parent",
        )?
        .map(|value| {
            value
                .try_from_inline::<Id>()
                .map_err(|_| AuthorityGrantDecodeError::InvalidId("authority_parent"))
        })
        .transpose()?;

        let invoke = exactly_one(
            find!(
                (value: Inline<Boolean>),
                pattern!(facts, [{ entity @ authority_invoke: ?value }])
            )
            .map(|(value,)| value),
            "authority_invoke",
        )?
        .try_from_inline::<bool>()
        .map_err(|_| AuthorityGrantDecodeError::InvalidBoolean("authority_invoke"))?;

        let delegate = exactly_one(
            find!(
                (value: Inline<Boolean>),
                pattern!(facts, [{ entity @ authority_delegate: ?value }])
            )
            .map(|(value,)| value),
            "authority_delegate",
        )?
        .try_from_inline::<bool>()
        .map_err(|_| AuthorityGrantDecodeError::InvalidBoolean("authority_delegate"))?;

        let mode = AuthorityMode::from_flags(invoke, delegate)?;

        let grant = Self {
            parent,
            subject,
            resource,
            action,
            mode,
        };
        if grant.fragment().facts() != facts {
            return Err(AuthorityGrantDecodeError::NonCanonicalShape);
        }
        Ok(grant)
    }
}

/// Why canonical grant facts could not be decoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorityGrantDecodeError {
    /// A required field is absent.
    MissingField(&'static str),
    /// A single-valued field occurs more than once.
    RepeatedField(&'static str),
    /// An id field is nil or does not use the canonical `GenId` layout.
    InvalidId(&'static str),
    /// A boolean field is not the canonical all-zero/all-one representation.
    InvalidBoolean(&'static str),
    /// The subject bytes are not an Ed25519 verifying key.
    InvalidSubject,
    /// Neither invocation nor delegation is granted.
    EmptyGrant,
    /// Facts beyond the one exact intrinsic grant entity are present.
    NonCanonicalShape,
}

impl fmt::Display for AuthorityGrantDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField(field) => write!(formatter, "grant is missing {field}"),
            Self::RepeatedField(field) => write!(formatter, "grant repeats {field}"),
            Self::InvalidId(field) => write!(formatter, "grant has an invalid {field}"),
            Self::InvalidBoolean(field) => {
                write!(formatter, "grant has an invalid boolean in {field}")
            }
            Self::InvalidSubject => formatter.write_str("grant subject is not an Ed25519 key"),
            Self::EmptyGrant => formatter.write_str("grant neither invokes nor delegates"),
            Self::NonCanonicalShape => {
                formatter.write_str("grant data is not exactly one canonical grant entity")
            }
        }
    }
}

impl Error for AuthorityGrantDecodeError {}

/// One accepted signed grant occurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedAuthorityGrant {
    commit: CollectionCommit,
    grant: AuthorityGrant,
}

impl AcceptedAuthorityGrant {
    /// Signed collection commit whose intrinsic id names this occurrence.
    pub fn commit(&self) -> CollectionCommit {
        self.commit
    }

    /// Atomic authority statement carried by the commit data.
    pub fn grant(&self) -> AuthorityGrant {
        self.grant
    }
}

/// Per-candidate evidence that remained inert during authority resolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthorityDiagnostic {
    /// A structurally valid authority commit failed strict signature checking.
    InvalidCommit(CollectionRecordDiagnostic),
    /// Protocol authority commits must carry the canonical empty metadata archive.
    NonCanonicalMetadata {
        /// Candidate commit id.
        commit: Id,
    },
    /// The candidate's signed data handle is not currently readable.
    DataUnavailable {
        /// Candidate commit id.
        commit: Id,
        /// Missing or unreadable signed data identity.
        data: CollectionData,
    },
    /// The candidate data is not the exact canonical element it claims to be.
    InvalidData {
        /// Candidate commit id.
        commit: Id,
        /// Exact collection-element validation failure.
        source: SimpleArchiveUnionValidationError,
    },
    /// The canonical archive does not encode one exact grant atom.
    InvalidGrant {
        /// Candidate commit id.
        commit: Id,
        /// Grant-shape failure.
        source: AuthorityGrantDecodeError,
    },
    /// A no-parent grant was not signed by the team root.
    InvalidRootIssuer {
        /// Candidate commit id.
        commit: Id,
    },
    /// The child signer is not the parent grant's subject.
    ParentSubjectMismatch {
        /// Candidate commit id.
        commit: Id,
        /// Named parent occurrence.
        parent: Id,
    },
    /// The child changed the parent's exact resource.
    ResourceEscalation {
        /// Candidate commit id.
        commit: Id,
        /// Named parent occurrence.
        parent: Id,
    },
    /// The child changed the parent's exact action.
    ActionEscalation {
        /// Candidate commit id.
        commit: Id,
        /// Named parent occurrence.
        parent: Id,
    },
    /// The accepted parent grants no delegation authority.
    ParentCannotDelegate {
        /// Candidate commit id.
        commit: Id,
        /// Named parent occurrence.
        parent: Id,
    },
    /// The named parent is absent or not grounded in the positive fixed point.
    UnresolvedParent {
        /// Candidate commit id.
        commit: Id,
        /// Named parent occurrence.
        parent: Id,
    },
}

impl AuthorityDiagnostic {
    /// Candidate commit carrying this diagnostic.
    pub fn commit(&self) -> Id {
        match self {
            Self::InvalidCommit(diagnostic) => diagnostic.id,
            Self::NonCanonicalMetadata { commit }
            | Self::DataUnavailable { commit, .. }
            | Self::InvalidData { commit, .. }
            | Self::InvalidGrant { commit, .. }
            | Self::InvalidRootIssuer { commit }
            | Self::ParentSubjectMismatch { commit, .. }
            | Self::ResourceEscalation { commit, .. }
            | Self::ActionEscalation { commit, .. }
            | Self::ParentCannotDelegate { commit, .. }
            | Self::UnresolvedParent { commit, .. } => *commit,
        }
    }
}

/// Accepted positive authority and inert-candidate diagnostics for one prefix.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AuthorityResolution {
    accepted: BTreeMap<Id, AcceptedAuthorityGrant>,
    diagnostics: Vec<AuthorityDiagnostic>,
}

impl AuthorityResolution {
    /// Accepted grant occurrences in intrinsic commit-id order.
    pub fn grants(&self) -> impl Iterator<Item = &AcceptedAuthorityGrant> {
        self.accepted.values()
    }

    /// Look up one accepted occurrence by its collection commit id.
    pub fn grant(&self, commit: Id) -> Option<&AcceptedAuthorityGrant> {
        self.accepted.get(&commit)
    }

    /// Candidates excluded from this known-prefix fixed point.
    pub fn diagnostics(&self) -> &[AuthorityDiagnostic] {
        &self.diagnostics
    }

    /// Whether `subject` may invoke one exact `(action, resource)` atom.
    pub fn allows(
        &self,
        subject: &Inline<ED25519PublicKey>,
        action: Id,
        resource: CollectionHandle,
    ) -> bool {
        self.accepted.values().any(|accepted| {
            let grant = accepted.grant;
            grant.invoke()
                && grant.subject == *subject
                && grant.action == action
                && grant.resource == resource
        })
    }

    /// Whether `subject` may delegate one exact `(action, resource)` atom.
    pub fn allows_delegation(
        &self,
        subject: &Inline<ED25519PublicKey>,
        action: Id,
        resource: CollectionHandle,
    ) -> bool {
        self.accepted.values().any(|accepted| {
            let grant = accepted.grant;
            grant.delegate()
                && grant.subject == *subject
                && grant.action == action
                && grant.resource == resource
        })
    }
}

/// Fatal failure to obtain one complete known-prefix authority observation.
#[derive(Debug)]
pub enum AuthorityResolutionError<RecordsError, ReaderError> {
    /// Native collection-record enumeration failed.
    Discovery(CollectionDiscoveryError<RecordsError>),
    /// A coherent blob-reader view could not be opened after discovery.
    Reader(ReaderError),
}

impl<RecordsError, ReaderError> fmt::Display for AuthorityResolutionError<RecordsError, ReaderError>
where
    RecordsError: fmt::Display,
    ReaderError: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(source) => source.fmt(formatter),
            Self::Reader(source) => {
                write!(formatter, "failed to open authority blob view: {source}")
            }
        }
    }
}

impl<RecordsError, ReaderError> Error for AuthorityResolutionError<RecordsError, ReaderError>
where
    RecordsError: Error + 'static,
    ReaderError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Discovery(source) => Some(source),
            Self::Reader(source) => Some(source),
        }
    }
}

/// Canonical public authority collection descriptor for one team root.
pub fn descriptor(team_root: VerifyingKey) -> Fragment {
    simplearchive_union::descriptor(
        &CollectionName::new(AUTHORITY_COLLECTION_NAME)
            .expect("the static authority collection name is valid"),
        team_root,
        crate::collection::reach::public(),
    )
}

/// Content identity of one team's canonical authority collection.
pub fn collection(team_root: VerifyingKey) -> CollectionHandle {
    crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor(team_root).into_facts()).get_handle()
}

/// Publish one canonical authority grant under its issuer's commit signature.
pub fn publish_grant<S>(
    store: &mut S,
    team_root: VerifyingKey,
    issuer: &SigningKey,
    grant: AuthorityGrant,
) -> Result<CollectionCommit, PublicationError<S::PutError, S::InsertError>>
where
    S: BlobStorePut + CollectionStore,
{
    simplearchive_union::publish_fragment_commit(
        store,
        &descriptor(team_root),
        grant.fragment(),
        issuer,
    )
}

/// Resolve the positive least fixed point of one team's known authority grants.
///
/// Record discovery and reader creation are global observation boundaries and
/// fail the call. Every candidate dependency and grant shape is independent:
/// unavailable or malformed candidates remain inert and diagnostic instead of
/// poisoning otherwise grounded authority.
pub fn resolve_authority<S>(
    store: &mut S,
    team_root: VerifyingKey,
) -> Result<AuthorityResolution, AuthorityResolutionError<S::RecordsError, S::ReaderError>>
where
    S: BlobStore + CollectionStore,
{
    let expected_descriptor = descriptor(team_root);
    let authority_collection =
        crate::blob::IntoBlob::<SimpleArchive>::to_blob(expected_descriptor.facts().clone())
            .get_handle();
    let discovered = discover_collection_records_authorized(store, authority_collection, |_| true)
        .map_err(AuthorityResolutionError::Discovery)?;
    let reader = store.reader().map_err(AuthorityResolutionError::Reader)?;

    let mut diagnostics: Vec<_> = discovered
        .diagnostics()
        .iter()
        .cloned()
        .map(AuthorityDiagnostic::InvalidCommit)
        .collect();
    let mut candidates = BTreeMap::<Id, Candidate>::new();

    for commit in discovered.commits().iter().copied() {
        if commit.metadata() != empty_metadata_handle() {
            diagnostics.push(AuthorityDiagnostic::NonCanonicalMetadata {
                commit: commit.id(),
            });
            continue;
        }

        let handle = Handle::<SimpleArchive>::from_hash(commit.data());
        let data_blob: Blob<SimpleArchive> = match reader.get(handle) {
            Ok(blob) => blob,
            Err(_) => {
                diagnostics.push(AuthorityDiagnostic::DataUnavailable {
                    commit: commit.id(),
                    data: commit.data(),
                });
                continue;
            }
        };
        if let Err(source) =
            simplearchive_union::validate_commit(&expected_descriptor, &commit, &data_blob)
        {
            diagnostics.push(AuthorityDiagnostic::InvalidData {
                commit: commit.id(),
                source,
            });
            continue;
        }
        let facts = TribleSet::try_from_blob(data_blob)
            .expect("validate_commit established canonical SimpleArchive data");
        let grant = match AuthorityGrant::decode(&facts) {
            Ok(grant) => grant,
            Err(source) => {
                diagnostics.push(AuthorityDiagnostic::InvalidGrant {
                    commit: commit.id(),
                    source,
                });
                continue;
            }
        };
        candidates.insert(commit.id(), Candidate { commit, grant });
    }

    let root = Inline::<ED25519PublicKey>::new(team_root.to_bytes());
    let mut accepted = BTreeMap::<Id, AcceptedAuthorityGrant>::new();
    let mut children = BTreeMap::<Id, Vec<Id>>::new();
    let mut evaluated = BTreeSet::<Id>::new();
    let mut queue = VecDeque::<Id>::new();

    for (id, candidate) in &candidates {
        match candidate.grant.parent {
            None if candidate.commit.public_key() == root => {
                accepted.insert(
                    *id,
                    AcceptedAuthorityGrant {
                        commit: candidate.commit,
                        grant: candidate.grant,
                    },
                );
                evaluated.insert(*id);
                queue.push_back(*id);
            }
            None => {
                diagnostics.push(AuthorityDiagnostic::InvalidRootIssuer { commit: *id });
                evaluated.insert(*id);
            }
            Some(parent) => children.entry(parent).or_default().push(*id),
        }
    }

    while let Some(parent_id) = queue.pop_front() {
        let Some(parent) = accepted.get(&parent_id).copied() else {
            continue;
        };
        let Some(child_ids) = children.get(&parent_id) else {
            continue;
        };
        for child_id in child_ids.iter().copied() {
            if evaluated.contains(&child_id) {
                continue;
            }
            let child = candidates
                .get(&child_id)
                .expect("child index contains only parsed candidates");
            let rejection = if child.commit.public_key() != parent.grant.subject {
                Some(AuthorityDiagnostic::ParentSubjectMismatch {
                    commit: child_id,
                    parent: parent_id,
                })
            } else if child.grant.resource != parent.grant.resource {
                Some(AuthorityDiagnostic::ResourceEscalation {
                    commit: child_id,
                    parent: parent_id,
                })
            } else if child.grant.action != parent.grant.action {
                Some(AuthorityDiagnostic::ActionEscalation {
                    commit: child_id,
                    parent: parent_id,
                })
            } else if !parent.grant.delegate() {
                Some(AuthorityDiagnostic::ParentCannotDelegate {
                    commit: child_id,
                    parent: parent_id,
                })
            } else {
                None
            };

            evaluated.insert(child_id);
            if let Some(diagnostic) = rejection {
                diagnostics.push(diagnostic);
                continue;
            }
            accepted.insert(
                child_id,
                AcceptedAuthorityGrant {
                    commit: child.commit,
                    grant: child.grant,
                },
            );
            queue.push_back(child_id);
        }
    }

    for (id, candidate) in &candidates {
        if evaluated.contains(id) {
            continue;
        }
        let parent = candidate
            .grant
            .parent
            .expect("every unevaluated root candidate was classified above");
        diagnostics.push(AuthorityDiagnostic::UnresolvedParent {
            commit: *id,
            parent,
        });
    }
    diagnostics.sort_by_key(AuthorityDiagnostic::commit);

    Ok(AuthorityResolution {
        accepted,
        diagnostics,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Candidate {
    commit: CollectionCommit,
    grant: AuthorityGrant,
}

fn exactly_one<T>(
    mut values: impl Iterator<Item = T>,
    field: &'static str,
) -> Result<T, AuthorityGrantDecodeError> {
    let Some(value) = values.next() else {
        return Err(AuthorityGrantDecodeError::MissingField(field));
    };
    if values.next().is_some() {
        return Err(AuthorityGrantDecodeError::RepeatedField(field));
    }
    Ok(value)
}

fn at_most_one<T>(
    mut values: impl Iterator<Item = T>,
    field: &'static str,
) -> Result<Option<T>, AuthorityGrantDecodeError> {
    let value = values.next();
    if values.next().is_some() {
        return Err(AuthorityGrantDecodeError::RepeatedField(field));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use anybytes::Bytes;
    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::IntoBlob;
    use crate::collection::{Collection, CollectionRecord};
    use crate::inline::encodings::hash::Handle;
    use crate::repo::memoryrepo::MemoryRepo;
    use crate::repo::BlobStorePut;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn target(team_root: VerifyingKey, name: &str) -> (Fragment, CollectionHandle) {
        let descriptor = simplearchive_union::descriptor(
            &CollectionName::new(name).unwrap(),
            team_root,
            crate::collection::reach::private(),
        );
        let handle = crate::blob::IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone())
            .get_handle();
        (descriptor, handle)
    }

    #[test]
    fn root_grant_filters_a_multi_author_collection() {
        let root = key(1);
        let writer = key(2);
        let stranger = key(3);
        let (target_descriptor, target) = target(root.verifying_key(), "documents");
        let mut repo = MemoryRepo::default();

        publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let writer_facts = entity! { metadata::tag: metadata::KIND_MULTI };
        let stranger_facts = entity! { metadata::tag: metadata::KIND_INLINE_ENCODING };
        simplearchive_union::publish_fragment_commit(
            &mut repo,
            &target_descriptor,
            writer_facts.clone(),
            &writer,
        )
        .unwrap();
        simplearchive_union::publish_fragment_commit(
            &mut repo,
            &target_descriptor,
            stranger_facts,
            &stranger,
        )
        .unwrap();

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let mut facade = Collection::new(
            repo,
            &CollectionName::new("documents").unwrap(),
            root.verifying_key(),
            key(4),
            crate::collection::reach::private(),
        );
        let facts = facade
            .snapshot_authorized(|subject| authority.allows(subject, ACTION_WRITE, target))
            .unwrap();

        assert_eq!(facts.facts(), writer_facts.facts());
        assert_eq!(authority.grants().count(), 1);
        assert!(authority.diagnostics().is_empty());
    }

    #[test]
    fn one_parent_delegation_reaches_a_positive_fixed_point() {
        let root = key(10);
        let delegate = key(11);
        let writer = key(12);
        let (_, target) = target(root.verifying_key(), "delegated");
        let mut repo = MemoryRepo::default();

        let parent = publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                delegate.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Delegate,
            ),
        )
        .unwrap();
        let child = publish_grant(
            &mut repo,
            root.verifying_key(),
            &delegate,
            AuthorityGrant::delegated(
                parent.id(),
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let delegate_subject = Inline::new(delegate.verifying_key().to_bytes());
        let writer = Inline::new(writer.verifying_key().to_bytes());
        assert!(!authority.allows(&delegate_subject, ACTION_WRITE, target));
        assert!(authority.allows_delegation(&delegate_subject, ACTION_WRITE, target));
        assert!(authority.allows(&writer, ACTION_WRITE, target));
        assert!(!authority.allows_delegation(&writer, ACTION_WRITE, target));
        assert!(authority.grant(parent.id()).is_some());
        assert!(authority.grant(child.id()).is_some());
        assert!(authority.diagnostics().is_empty());
    }

    #[test]
    fn delegation_cannot_change_subject_chain_resource_or_action() {
        let root = key(20);
        let delegate = key(21);
        let impostor = key(22);
        let writer = key(23);
        let (_, target) = target(root.verifying_key(), "target");
        let (_, other) = self::target(root.verifying_key(), "other");
        let other_action = id_hex!("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
        let mut repo = MemoryRepo::default();

        let parent = publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                delegate.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Delegate,
            ),
        )
        .unwrap();
        for (issuer, resource, action) in [
            (&impostor, target, ACTION_WRITE),
            (&delegate, other, ACTION_WRITE),
            (&delegate, target, other_action),
        ] {
            publish_grant(
                &mut repo,
                root.verifying_key(),
                issuer,
                AuthorityGrant::delegated(
                    parent.id(),
                    writer.verifying_key(),
                    resource,
                    action,
                    AuthorityMode::Invoke,
                ),
            )
            .unwrap();
        }

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let writer = Inline::new(writer.verifying_key().to_bytes());
        assert!(!authority.allows(&writer, ACTION_WRITE, target));
        assert_eq!(authority.grants().count(), 1);
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::ParentSubjectMismatch { .. }
        )));
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::ResourceEscalation { .. }
        )));
        assert!(authority
            .diagnostics()
            .iter()
            .any(|diagnostic| matches!(diagnostic, AuthorityDiagnostic::ActionEscalation { .. })));
    }

    #[test]
    fn unresolved_and_malformed_candidates_do_not_poison_a_valid_grant() {
        let root = key(30);
        let writer = key(31);
        let attacker = key(32);
        let (_, target) = target(root.verifying_key(), "resilient");
        let mut repo = MemoryRepo::default();

        publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let authority_collection = collection(root.verifying_key());
        let missing = CollectionCommit::sign(
            &attacker,
            authority_collection,
            Inline::new([0x71; 32]),
            empty_metadata_handle(),
        );
        CollectionStore::insert(&mut repo, CollectionRecord::Commit(missing)).unwrap();

        let malformed = Blob::<SimpleArchive>::new(Bytes::from_source(vec![1, 2, 3]));
        let malformed_handle: Inline<Handle<SimpleArchive>> = repo.put(malformed).unwrap();
        let malformed_commit = CollectionCommit::sign(
            &attacker,
            authority_collection,
            Handle::<SimpleArchive>::to_hash(malformed_handle),
            empty_metadata_handle(),
        );
        CollectionStore::insert(&mut repo, CollectionRecord::Commit(malformed_commit)).unwrap();

        let orphan = publish_grant(
            &mut repo,
            root.verifying_key(),
            &attacker,
            AuthorityGrant::delegated(
                id_hex!("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"),
                attacker.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let writer = Inline::new(writer.verifying_key().to_bytes());
        assert!(authority.allows(&writer, ACTION_WRITE, target));
        assert_eq!(authority.grants().count(), 1);
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::DataUnavailable { commit, .. } if *commit == missing.id()
        )));
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::InvalidData { commit, .. }
                if *commit == malformed_commit.id()
        )));
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::UnresolvedParent { commit, .. } if *commit == orphan.id()
        )));
    }

    #[test]
    fn parent_without_delegation_cannot_issue_a_child() {
        let root = key(40);
        let member = key(41);
        let writer = key(42);
        let (_, target) = target(root.verifying_key(), "no-delegation");
        let mut repo = MemoryRepo::default();

        let parent = publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                member.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();
        let child = publish_grant(
            &mut repo,
            root.verifying_key(),
            &member,
            AuthorityGrant::delegated(
                parent.id(),
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let member_subject = Inline::new(member.verifying_key().to_bytes());
        assert!(authority.allows(&member_subject, ACTION_WRITE, target));
        assert!(!authority.allows_delegation(&member_subject, ACTION_WRITE, target));
        assert!(authority.grant(child.id()).is_none());
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::ParentCannotDelegate { commit, .. } if *commit == child.id()
        )));
    }

    #[test]
    fn child_before_parent_becomes_valid_without_losing_prior_authority() {
        let root = key(50);
        let delegate = key(51);
        let writer = key(52);
        let independent_writer = key(53);
        let (_, target) = target(root.verifying_key(), "prefix-growth");

        let mut staged = MemoryRepo::default();
        let parent = publish_grant(
            &mut staged,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                delegate.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Delegate,
            ),
        )
        .unwrap();

        let mut repo = MemoryRepo::default();
        repo.blobs.union(staged.blobs);
        let independent = publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                independent_writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();
        let child = publish_grant(
            &mut repo,
            root.verifying_key(),
            &delegate,
            AuthorityGrant::delegated(
                parent.id(),
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let before = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let before_ids: BTreeSet<_> = before
            .grants()
            .map(|accepted| accepted.commit().id())
            .collect();
        assert_eq!(before_ids, BTreeSet::from([independent.id()]));
        assert!(before.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::UnresolvedParent { commit, parent: named }
                if *commit == child.id() && *named == parent.id()
        )));

        CollectionStore::insert(&mut repo, CollectionRecord::Commit(parent)).unwrap();
        let after = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let after_ids: BTreeSet<_> = after
            .grants()
            .map(|accepted| accepted.commit().id())
            .collect();
        assert!(before_ids.is_subset(&after_ids));
        assert_eq!(
            after_ids,
            BTreeSet::from([independent.id(), parent.id(), child.id()])
        );
        assert!(after.diagnostics().is_empty());
    }

    #[test]
    fn no_parent_grant_requires_the_team_root_signature() {
        let root = key(60);
        let outsider = key(61);
        let writer = key(62);
        let (_, target) = target(root.verifying_key(), "root-anchor");
        let mut repo = MemoryRepo::default();

        let candidate = publish_grant(
            &mut repo,
            root.verifying_key(),
            &outsider,
            AuthorityGrant::root(
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::InvokeAndDelegate,
            ),
        )
        .unwrap();

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        assert!(authority.grants().next().is_none());
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::InvalidRootIssuer { commit } if *commit == candidate.id()
        )));
    }

    #[test]
    fn noncanonical_candidates_cannot_poison_valid_authority() {
        let root = key(70);
        let writer = key(71);
        let noisy_writer = key(72);
        let (_, target) = target(root.verifying_key(), "canonical-grants");
        let authority_descriptor = descriptor(root.verifying_key());
        let mut repo = MemoryRepo::default();

        let valid = publish_grant(
            &mut repo,
            root.verifying_key(),
            &root,
            AuthorityGrant::root(
                writer.verifying_key(),
                target,
                ACTION_WRITE,
                AuthorityMode::Invoke,
            ),
        )
        .unwrap();

        let noisy_grant = AuthorityGrant::root(
            noisy_writer.verifying_key(),
            target,
            ACTION_WRITE,
            AuthorityMode::Invoke,
        );
        let noisy_data: Blob<SimpleArchive> = noisy_grant.fragment().into_facts().to_blob();
        let noisy_metadata: Blob<SimpleArchive> = entity! { metadata::tag: metadata::KIND_MULTI }
            .into_facts()
            .to_blob();
        let nonempty_metadata = simplearchive_union::publish_commit(
            &mut repo,
            &authority_descriptor,
            &noisy_data,
            &noisy_metadata,
            &root,
        )
        .unwrap();

        let extra_grant = AuthorityGrant::root(
            noisy_writer.verifying_key(),
            target,
            ACTION_WRITE,
            AuthorityMode::Delegate,
        );
        let mut extra_facts = extra_grant.fragment().into_facts();
        extra_facts += entity! { metadata::tag: metadata::KIND_INLINE_ENCODING };
        let extra_data: Blob<SimpleArchive> = extra_facts.to_blob();
        let empty_metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        let extra_shape = simplearchive_union::publish_commit(
            &mut repo,
            &authority_descriptor,
            &extra_data,
            &empty_metadata,
            &root,
        )
        .unwrap();

        let authority = resolve_authority(&mut repo, root.verifying_key()).unwrap();
        let writer = Inline::new(writer.verifying_key().to_bytes());
        assert!(authority.allows(&writer, ACTION_WRITE, target));
        assert_eq!(authority.grants().count(), 1);
        assert!(authority.grant(valid.id()).is_some());
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::NonCanonicalMetadata { commit }
                if *commit == nonempty_metadata.id()
        )));
        assert!(authority.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            AuthorityDiagnostic::InvalidGrant {
                commit,
                source: AuthorityGrantDecodeError::NonCanonicalShape,
            } if *commit == extra_shape.id()
        )));
    }
}
