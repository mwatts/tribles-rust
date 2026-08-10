//! Trust boundary for admitting collection claims learned from a network.
//!
//! The current pile-sync wire protocol is deliberately read-only: it can land
//! content-addressed blobs, but it has no operation that inserts native
//! [`CollectionRecord`] values into a [`CollectionStore`]. This module defines
//! the boundary any future gossip or reconciliation carrier must cross. It does
//! not add a write opcode, a mutable head, or a branch compatibility layer.
//!
//! Fetched blobs may be cached before admission because they are inert content
//! addresses. Native collection records are different: a stored signed commit
//! participates in retention, and every stored record occupies permanent
//! grow-only evidence space. [`admit_collection_claim`] therefore inserts no
//! definition or claim until all of these checks have completed:
//!
//! 1. the request names the exact intrinsic collection definition(s);
//! 2. a commit's embedded signature verifies strictly;
//! 3. the *commit author*, not the transport carrier, has a currently verified
//!    capability granting write access to the definition's generic scope;
//! 4. the caller's exact validator confirms the complete data/metadata
//!    dependency closure is resident and content-valid and proves the named
//!    representation/recipe equation.
//!
//! `MERGE` and `DERIVE` are permissionless equations, so they skip author
//! authorization but still require the same exact validator verdict. Pending
//! or rejected work remains transient. A capability is consulted only during
//! this call; expiry never retracts a record admitted earlier.

use std::error::Error;
use std::fmt;

use ed25519_dalek::VerifyingKey;

use triblespace_core::collection::{
    CollectionClaimValidation, CollectionDefinition, CollectionRecord, CollectionStore,
    CollectionValidationRequest, CommitVerificationError,
};
use triblespace_core::id::Id;
use triblespace_core::repo::StorageFlush;
use triblespace_core::repo::capability::VerifiedCapability;

/// Result of a semantically complete network admission attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionAdmissionOutcome<D> {
    /// The exact intrinsic record is now durable (or was already present).
    Admitted { record: Id },
    /// Required bytes or other validation evidence are not available yet.
    Pending { record: Id },
    /// The exact recipe validator deterministically rejected the claim.
    Rejected { record: Id, diagnostic: D },
    /// The signed author has no currently valid write capability for the
    /// collection's generic scope.
    Unauthorized {
        record: Id,
        author: [u8; 32],
        scope: Id,
    },
}

/// Operational or structural failure at the native-record admission boundary.
#[derive(Debug)]
pub enum CollectionAdmissionError<AuthorizationError, ValidationError, InsertError, FlushError> {
    /// A commit's embedded Ed25519 signature did not authenticate its fields.
    InvalidCommitSignature {
        record: Id,
        source: CommitVerificationError,
    },
    /// A commit or merge names a different collection than the supplied exact
    /// definition.
    DefinitionMismatch {
        record: Id,
        role: &'static str,
        expected: Id,
        actual: Id,
    },
    /// Looking up or verifying the author's current capability failed
    /// operationally.
    Authorization {
        record: Id,
        source: AuthorizationError,
    },
    /// Exact dependency/recipe validation failed operationally. Deterministic
    /// invalidity is returned as [`CollectionAdmissionOutcome::Rejected`].
    Validation { record: Id, source: ValidationError },
    /// Inserting one exact definition failed before the claim became visible.
    DefinitionInsert { definition: Id, source: InsertError },
    /// The dependency-and-definition durability barrier failed.
    DependencyFlush { source: FlushError },
    /// Inserting the admitted claim failed.
    RecordInsert { record: Id, source: InsertError },
    /// The final admitted-record durability barrier failed.
    RecordFlush { record: Id, source: FlushError },
}

impl<AuthorizationError, ValidationError, InsertError, FlushError> fmt::Display
    for CollectionAdmissionError<AuthorizationError, ValidationError, InsertError, FlushError>
where
    AuthorizationError: fmt::Display,
    ValidationError: fmt::Display,
    InsertError: fmt::Display,
    FlushError: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCommitSignature { record, source } => {
                write!(
                    f,
                    "collection commit {record:X} failed strict verification: {source}"
                )
            }
            Self::DefinitionMismatch {
                record,
                role,
                expected,
                actual,
            } => write!(
                f,
                "collection claim {record:X} names {role} {actual:X}, not exact definition {expected:X}"
            ),
            Self::Authorization { record, source } => write!(
                f,
                "author capability lookup for collection claim {record:X} failed: {source}"
            ),
            Self::Validation { record, source } => write!(
                f,
                "exact validation for collection claim {record:X} failed: {source}"
            ),
            Self::DefinitionInsert { definition, source } => write!(
                f,
                "failed to insert collection definition {definition:X}: {source}"
            ),
            Self::DependencyFlush { source } => {
                write!(
                    f,
                    "failed to flush admitted collection dependencies: {source}"
                )
            }
            Self::RecordInsert { record, source } => {
                write!(
                    f,
                    "failed to insert admitted collection claim {record:X}: {source}"
                )
            }
            Self::RecordFlush { record, source } => {
                write!(
                    f,
                    "failed to flush admitted collection claim {record:X}: {source}"
                )
            }
        }
    }
}

impl<AuthorizationError, ValidationError, InsertError, FlushError> Error
    for CollectionAdmissionError<AuthorizationError, ValidationError, InsertError, FlushError>
where
    AuthorizationError: Error + 'static,
    ValidationError: Error + 'static,
    InsertError: Error + 'static,
    FlushError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidCommitSignature { source, .. } => Some(source),
            Self::Authorization { source, .. } => Some(source),
            Self::Validation { source, .. } => Some(source),
            Self::DefinitionInsert { source, .. } | Self::RecordInsert { source, .. } => {
                Some(source)
            }
            Self::DependencyFlush { source } | Self::RecordFlush { source, .. } => Some(source),
            Self::DefinitionMismatch { .. } => None,
        }
    }
}

/// Result type for admission into a concrete durable collection store.
pub type CollectionAdmissionResult<S, D, AuthorizationError, ValidationError> = Result<
    CollectionAdmissionOutcome<D>,
    CollectionAdmissionError<
        AuthorizationError,
        ValidationError,
        <S as CollectionStore>::InsertError,
        <S as StorageFlush>::Error,
    >,
>;

/// Admit one exact, already-decoded collection claim into durable storage.
///
/// `author_capability` is invoked only for `COMMIT`. It receives the public key
/// authenticated by that commit's own signature and the exact extrinsic
/// collection scope. It must verify a capability chain *now* and return the
/// matching [`VerifiedCapability`]. The transport peer identity is
/// intentionally absent from this API, so a relay cannot lend its authority to
/// another author's commit.
///
/// `validate` is the representation/recipe trust boundary. `Accepted` means
/// more than endpoint syntax: every data and metadata root, its complete
/// conservative child closure, and every equation endpoint must already be
/// resident under freshly checked content identities. Any absence returns
/// `Pending`; deterministic wrong bytes or equations return `Rejected`.
///
/// Only after both callbacks succeed are the exact definitions inserted and a
/// durability barrier taken. The claim is inserted last and flushed again.
/// Nothing in this function revisits capabilities for records already stored.
pub fn admit_collection_claim<S, D, AuthorizationError, ValidationError, Authorize, Validate>(
    store: &mut S,
    request: CollectionValidationRequest<'_>,
    mut author_capability: Authorize,
    mut validate: Validate,
) -> CollectionAdmissionResult<S, D, AuthorizationError, ValidationError>
where
    S: CollectionStore + StorageFlush,
    Authorize: FnMut(VerifyingKey, Id) -> Result<Option<VerifiedCapability>, AuthorizationError>,
    Validate: for<'a> FnMut(
        CollectionValidationRequest<'a>,
    ) -> Result<CollectionClaimValidation<D>, ValidationError>,
{
    let record = request.claim_id();
    let definitions = exact_definitions(request).map_err(|mismatch| {
        CollectionAdmissionError::DefinitionMismatch {
            record,
            role: mismatch.role,
            expected: mismatch.expected,
            actual: mismatch.actual,
        }
    })?;

    if let CollectionValidationRequest::Commit { definition, claim } = request {
        claim.verify_strict().map_err(|source| {
            CollectionAdmissionError::InvalidCommitSignature { record, source }
        })?;

        // Parsing succeeds because strict verification already parsed the same
        // bytes as an Ed25519 key.
        let author = VerifyingKey::from_bytes(&claim.public_key().raw)
            .expect("strictly verified commit carries a valid public key");
        let scope = definition.scope();
        let capability = author_capability(author, scope)
            .map_err(|source| CollectionAdmissionError::Authorization { record, source })?;
        let authorized = capability.as_ref().is_some_and(|capability| {
            capability.subject == author && capability.grants_write_on_resource(&scope)
        });
        if !authorized {
            return Ok(CollectionAdmissionOutcome::Unauthorized {
                record,
                author: author.to_bytes(),
                scope,
            });
        }
    }

    match validate(request)
        .map_err(|source| CollectionAdmissionError::Validation { record, source })?
    {
        CollectionClaimValidation::Pending => {
            return Ok(CollectionAdmissionOutcome::Pending { record });
        }
        CollectionClaimValidation::Rejected(diagnostic) => {
            return Ok(CollectionAdmissionOutcome::Rejected { record, diagnostic });
        }
        CollectionClaimValidation::Accepted => {}
    }

    for definition in definitions {
        store
            .insert(CollectionRecord::Definition(definition))
            .map_err(|source| CollectionAdmissionError::DefinitionInsert {
                definition: definition.id(),
                source,
            })?;
    }
    store
        .flush()
        .map_err(|source| CollectionAdmissionError::DependencyFlush { source })?;

    let admitted = match request {
        CollectionValidationRequest::Commit { claim, .. } => CollectionRecord::Commit(*claim),
        CollectionValidationRequest::Merge { claim, .. } => CollectionRecord::Merge(*claim),
        CollectionValidationRequest::Derive { claim, .. } => CollectionRecord::Derive(*claim),
    };
    store
        .insert(admitted)
        .map_err(|source| CollectionAdmissionError::RecordInsert { record, source })?;
    store
        .flush()
        .map_err(|source| CollectionAdmissionError::RecordFlush { record, source })?;

    Ok(CollectionAdmissionOutcome::Admitted { record })
}

#[derive(Clone, Copy)]
struct DefinitionMismatch {
    role: &'static str,
    expected: Id,
    actual: Id,
}

fn exact_definitions(
    request: CollectionValidationRequest<'_>,
) -> Result<Vec<CollectionDefinition>, DefinitionMismatch> {
    match request {
        CollectionValidationRequest::Commit { definition, claim } => {
            require_definition("collection", definition, claim.collection())?;
            Ok(vec![*definition])
        }
        CollectionValidationRequest::Merge { definition, claim } => {
            require_definition("collection", definition, claim.collection())?;
            Ok(vec![*definition])
        }
        CollectionValidationRequest::Derive {
            source_definition,
            target_definition,
            claim,
        } => {
            require_definition("source collection", source_definition, claim.source())?;
            require_definition("target collection", target_definition, claim.target())?;
            if source_definition.id() == target_definition.id() {
                Ok(vec![*source_definition])
            } else {
                Ok(vec![*source_definition, *target_definition])
            }
        }
    }
}

fn require_definition(
    role: &'static str,
    definition: &CollectionDefinition,
    actual: Id,
) -> Result<(), DefinitionMismatch> {
    let expected = definition.id();
    if actual == expected {
        Ok(())
    } else {
        Err(DefinitionMismatch {
            role,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeSet;
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    use super::*;
    use triblespace_core::attestation::{signature_r, signature_s, signed_by};
    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace_core::blob::{Blob, IntoBlob};
    use triblespace_core::collection::{
        CollectionCommit, CollectionDerive, CollectionMerge, KIND_COLLECTION_COMMIT, collection,
        data as collection_data, simplearchive_union,
    };
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::macros::entity;
    use triblespace_core::metadata::{self, archive as commit_metadata};
    use triblespace_core::repo::capability::{PERM_READ, PERM_WRITE, scope_resource};
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::trible::{TRIBLE_LEN, Trible, TribleSet};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn archive(byte: u8) -> Blob<SimpleArchive> {
        let mut row = [byte; TRIBLE_LEN];
        row[16..32].fill(1);
        let mut set = TribleSet::new();
        set.insert(&Trible::force_raw(row).unwrap());
        set.to_blob()
    }

    fn data_id(blob: &Blob<SimpleArchive>) -> triblespace_core::collection::CollectionData {
        Handle::<SimpleArchive>::to_hash(blob.get_handle())
    }

    fn capability(key: &SigningKey, permission: Id, resource: Id) -> VerifiedCapability {
        let scope_root = triblespace_core::id::ufoid();
        let mut cap_set = TribleSet::from(entity! {
            ExclusiveId::force_ref(&scope_root) @
            triblespace_core::metadata::tag: permission,
        });
        cap_set += TribleSet::from(entity! {
            ExclusiveId::force_ref(&scope_root) @
            scope_resource: resource,
        });
        VerifiedCapability {
            subject: key.verifying_key(),
            scope_root: *scope_root,
            cap_set,
        }
    }

    fn records(store: &mut MemoryRepo) -> BTreeSet<Id> {
        store
            .records()
            .unwrap()
            .map(|record| record.unwrap().id())
            .collect()
    }

    #[test]
    fn scoped_author_not_carrier_admits_commit_idempotently() {
        let author = SigningKey::generate(&mut OsRng);
        let carrier = SigningKey::generate(&mut OsRng);
        let scope = id(1);
        let definition = simplearchive_union::definition(scope);
        let data = archive(2);
        let metadata = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &author,
            definition.id(),
            data_id(&data),
            metadata.get_handle(),
        );
        let author_capability = capability(&author, PERM_WRITE, scope);
        let carrier_capability = capability(&carrier, PERM_WRITE, scope);
        let request = CollectionValidationRequest::Commit {
            definition: &definition,
            claim: &commit,
        };
        let mut store = MemoryRepo::default();

        let outcome = admit_collection_claim(
            &mut store,
            request,
            |requested_author, requested_scope| {
                assert_eq!(requested_author, author.verifying_key());
                assert_ne!(requested_author, carrier_capability.subject);
                assert_eq!(requested_scope, scope);
                Ok::<_, Infallible>(Some(author_capability.clone()))
            },
            |request| {
                let CollectionValidationRequest::Commit { definition, claim } = request else {
                    unreachable!()
                };
                simplearchive_union::validate_commit(definition, claim, &data).unwrap();
                simplearchive_union::validate_element(&metadata).unwrap();
                Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted)
            },
        )
        .unwrap();

        assert_eq!(
            outcome,
            CollectionAdmissionOutcome::Admitted {
                record: commit.id()
            }
        );
        assert_eq!(
            records(&mut store),
            BTreeSet::from([definition.id(), commit.id()])
        );

        let replay = admit_collection_claim(
            &mut store,
            request,
            |requested_author, requested_scope| {
                assert_eq!(requested_author, author.verifying_key());
                assert_eq!(requested_scope, scope);
                Ok::<_, Infallible>(Some(author_capability.clone()))
            },
            |_| Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted),
        )
        .unwrap();
        assert_eq!(
            replay,
            CollectionAdmissionOutcome::Admitted {
                record: commit.id()
            }
        );
        assert_eq!(
            records(&mut store),
            BTreeSet::from([definition.id(), commit.id()])
        );
    }

    #[test]
    fn read_only_and_wrong_scope_caps_cannot_admit() {
        let author = SigningKey::generate(&mut OsRng);
        let scope = id(3);
        let definition = simplearchive_union::definition(scope);
        let data = archive(4);
        let metadata = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &author,
            definition.id(),
            data_id(&data),
            metadata.get_handle(),
        );
        let request = CollectionValidationRequest::Commit {
            definition: &definition,
            claim: &commit,
        };

        for denied in [
            capability(&author, PERM_READ, scope),
            capability(&author, PERM_WRITE, id(99)),
        ] {
            let mut store = MemoryRepo::default();
            let outcome = admit_collection_claim(
                &mut store,
                request,
                |_, _| Ok::<_, Infallible>(Some(denied.clone())),
                |_| Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted),
            )
            .unwrap();
            assert!(matches!(
                outcome,
                CollectionAdmissionOutcome::Unauthorized { .. }
            ));
            assert!(records(&mut store).is_empty());
        }
    }

    #[test]
    fn invalid_commit_signature_rejects_before_callbacks() {
        let author = SigningKey::generate(&mut OsRng);
        let scope = id(15);
        let definition = simplearchive_union::definition(scope);
        let metadata = TribleSet::new().to_blob();
        let valid = CollectionCommit::sign(
            &author,
            definition.id(),
            data_id(&archive(16)),
            metadata.get_handle(),
        );
        let (r, mut s) = valid.signature();
        s.raw[0] ^= 1;
        let invalid = CollectionCommit::from_tribles(
            &entity! {
                metadata::tag: KIND_COLLECTION_COMMIT,
                collection: valid.collection(),
                collection_data: valid.data(),
                commit_metadata: valid.metadata(),
                signed_by: valid.public_key(),
                signature_r: r,
                signature_s: s,
            }
            .into_facts(),
        )
        .unwrap();
        let request = CollectionValidationRequest::Commit {
            definition: &definition,
            claim: &invalid,
        };
        let mut store = MemoryRepo::default();
        let called = Cell::new(false);

        let error = admit_collection_claim::<_, (), Infallible, Infallible, _, _>(
            &mut store,
            request,
            |_, _| {
                called.set(true);
                Ok(None)
            },
            |_| {
                called.set(true);
                Ok(CollectionClaimValidation::Accepted)
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CollectionAdmissionError::InvalidCommitSignature {
                source: CommitVerificationError::InvalidSignature,
                ..
            }
        ));
        assert!(!called.get());
        assert!(records(&mut store).is_empty());
    }

    #[test]
    fn capability_expiry_affects_new_admission_not_stored_evidence() {
        let author = SigningKey::generate(&mut OsRng);
        let scope = id(5);
        let definition = simplearchive_union::definition(scope);
        let data = archive(6);
        let metadata = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &author,
            definition.id(),
            data_id(&data),
            metadata.get_handle(),
        );
        let request = CollectionValidationRequest::Commit {
            definition: &definition,
            claim: &commit,
        };
        let cap = capability(&author, PERM_WRITE, scope);
        let mut store = MemoryRepo::default();

        admit_collection_claim(
            &mut store,
            request,
            |_, _| Ok::<_, Infallible>(Some(cap.clone())),
            |_| Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted),
        )
        .unwrap();
        let before = records(&mut store);

        let outcome = admit_collection_claim(
            &mut store,
            request,
            |_, _| Ok::<_, Infallible>(None),
            |_| Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted),
        )
        .unwrap();
        assert!(matches!(
            outcome,
            CollectionAdmissionOutcome::Unauthorized { .. }
        ));
        assert_eq!(records(&mut store), before);
    }

    #[test]
    fn pending_and_rejected_claims_leave_no_native_records() {
        let author = SigningKey::generate(&mut OsRng);
        let scope = id(7);
        let definition = simplearchive_union::definition(scope);
        let data = archive(8);
        let metadata = TribleSet::new().to_blob();
        let commit = CollectionCommit::sign(
            &author,
            definition.id(),
            data_id(&data),
            metadata.get_handle(),
        );
        let request = CollectionValidationRequest::Commit {
            definition: &definition,
            claim: &commit,
        };
        let cap = capability(&author, PERM_WRITE, scope);

        let mut pending_store = MemoryRepo::default();
        let pending = admit_collection_claim(
            &mut pending_store,
            request,
            |_, _| Ok::<_, Infallible>(Some(cap.clone())),
            |_| Ok::<_, Infallible>(CollectionClaimValidation::<&str>::Pending),
        )
        .unwrap();
        assert_eq!(
            pending,
            CollectionAdmissionOutcome::Pending {
                record: commit.id()
            }
        );
        assert!(records(&mut pending_store).is_empty());

        let mut rejected_store = MemoryRepo::default();
        let rejected = admit_collection_claim(
            &mut rejected_store,
            request,
            |_, _| Ok::<_, Infallible>(Some(cap.clone())),
            |_| Ok::<_, Infallible>(CollectionClaimValidation::Rejected("wrong recipe bytes")),
        )
        .unwrap();
        assert_eq!(
            rejected,
            CollectionAdmissionOutcome::Rejected {
                record: commit.id(),
                diagnostic: "wrong recipe bytes"
            }
        );
        assert!(records(&mut rejected_store).is_empty());
    }

    #[test]
    fn equations_are_permissionless_but_exactly_validated() {
        let scope = id(9);
        let definition = simplearchive_union::definition(scope);
        let mut low = archive(10);
        let mut high = archive(11);
        if data_id(&high) < data_id(&low) {
            std::mem::swap(&mut low, &mut high);
        }
        let result = simplearchive_union::join(&low, &high).unwrap();
        let merge = CollectionMerge::new(
            definition.id(),
            data_id(&low),
            data_id(&high),
            data_id(&result),
        );
        let request = CollectionValidationRequest::Merge {
            definition: &definition,
            claim: &merge,
        };
        let mut store = MemoryRepo::default();
        let mut authorization_called = false;

        let outcome = admit_collection_claim(
            &mut store,
            request,
            |_, _| {
                authorization_called = true;
                Ok::<_, Infallible>(None)
            },
            |request| {
                let CollectionValidationRequest::Merge { definition, claim } = request else {
                    unreachable!()
                };
                simplearchive_union::validate_merge(definition, claim, &low, &high, &result)
                    .unwrap();
                Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted)
            },
        )
        .unwrap();
        assert!(!authorization_called);
        assert_eq!(
            outcome,
            CollectionAdmissionOutcome::Admitted { record: merge.id() }
        );
        assert_eq!(
            records(&mut store),
            BTreeSet::from([definition.id(), merge.id()])
        );

        // A same-definition derive proves the two explicit definition roles
        // are handled without inventing any scope relation.
        let derive = CollectionDerive::new(
            definition.id(),
            definition.id(),
            data_id(&low),
            data_id(&low),
        );
        let derive_request = CollectionValidationRequest::Derive {
            source_definition: &definition,
            target_definition: &definition,
            claim: &derive,
        };
        let outcome = admit_collection_claim(
            &mut store,
            derive_request,
            |_, _| Ok::<_, Infallible>(None),
            |_| Ok::<_, Infallible>(CollectionClaimValidation::<()>::Accepted),
        )
        .unwrap();
        assert_eq!(
            outcome,
            CollectionAdmissionOutcome::Admitted {
                record: derive.id()
            }
        );
    }

    #[test]
    fn exact_definition_mismatch_rejects_before_callbacks() {
        let author = SigningKey::generate(&mut OsRng);
        let actual_definition = simplearchive_union::definition(id(12));
        let supplied_definition = simplearchive_union::definition(id(13));
        let commit = CollectionCommit::sign(
            &author,
            actual_definition.id(),
            data_id(&archive(14)),
            TribleSet::new().to_blob().get_handle(),
        );
        let request = CollectionValidationRequest::Commit {
            definition: &supplied_definition,
            claim: &commit,
        };
        let mut store = MemoryRepo::default();
        let called = Cell::new(false);

        let error = admit_collection_claim::<_, (), Infallible, Infallible, _, _>(
            &mut store,
            request,
            |_, _| {
                called.set(true);
                Ok(None)
            },
            |_| {
                called.set(true);
                Ok(CollectionClaimValidation::Accepted)
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CollectionAdmissionError::DefinitionMismatch { .. }
        ));
        assert!(!called.get());
        assert!(records(&mut store).is_empty());
    }
}
