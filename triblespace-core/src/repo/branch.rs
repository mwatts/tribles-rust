use crate::macros::pattern;
use ed25519::Signature;
use ed25519_dalek::SignatureError;
use ed25519_dalek::Verifier;
use ed25519_dalek::VerifyingKey;
use itertools::Itertools;

use crate::blob::Blob;
use crate::find;
use crate::id::Id;
use crate::inline::Inline;
use crate::prelude::blobencodings::SimpleArchive;
use crate::trible::TribleSet;

/// Why a branch metadata subject could not be identified uniquely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchEntityError {
    /// No entity in the metadata identifies itself as the expected branch.
    Missing,
    /// More than one entity identifies itself as the expected branch.
    Ambiguous,
}

/// Resolve the unique entity describing `branch_id` in one metadata blob.
///
/// Branch metadata may carry arbitrary facts on unrelated annotation entities.
/// Consumers must first establish this subject and then read branch fields from
/// that subject only; scanning the whole [`TribleSet`] lets an annotation's
/// `metadata::name`, `repo::head`, or timestamp impersonate branch state.
pub fn branch_entity(meta: &TribleSet, branch_id: Id) -> Result<Id, BranchEntityError> {
    let mut entities = find!(
        entity: Id,
        pattern!(meta, [{ ?entity @ super::branch: branch_id }])
    );
    let Some(entity) = entities.next() else {
        return Err(BranchEntityError::Missing);
    };
    if entities.next().is_some() {
        return Err(BranchEntityError::Ambiguous);
    }
    Ok(entity)
}

/// Error returned when branch signature verification fails.
pub enum ValidationError {
    /// The metadata has no unique entity for the expected branch id.
    InvalidBranchMetadata,
    /// The metadata contains multiple signature entities for the same commit.
    AmbiguousSignature,
    /// No signature information was found in the metadata.
    MissingSignature,
    /// The signature did not match the commit bytes or the public key was invalid.
    FailedValidation,
}

impl From<SignatureError> for ValidationError {
    /// Converts an Ed25519 signature error into a [`ValidationError::FailedValidation`].
    fn from(_: SignatureError) -> Self {
        ValidationError::FailedValidation
    }
}

/// Checks that the metadata signature matches the provided commit blob.
///
/// The function extracts the public key and signature from `metadata` and
/// verifies that it signs the `commit_head` blob. If the metadata is missing a
/// signature or contains multiple signature entities the appropriate
/// `ValidationError` variant is returned.
pub fn verify(
    branch_id: Id,
    commit_head: Blob<SimpleArchive>,
    metadata: TribleSet,
) -> Result<(), ValidationError> {
    let handle = commit_head.get_handle();
    let branch_entity =
        branch_entity(&metadata, branch_id).map_err(|_| ValidationError::InvalidBranchMetadata)?;
    let (pubkey, r, s) = match find!(
    (pubkey: Inline<_>, r, s),
    pattern!(&metadata, [
    {
        branch_entity @ super::head: handle,
        crate::attestation::signed_by: ?pubkey,
        crate::attestation::signature_r: ?r,
        crate::attestation::signature_s: ?s,
    }]))
    .at_most_one()
    {
        Ok(Some(result)) => result,
        Ok(None) => return Err(ValidationError::MissingSignature),
        Err(_) => return Err(ValidationError::AmbiguousSignature),
    };

    let Ok(pubkey): Result<VerifyingKey, _> = pubkey.try_from_inline() else {
        return Err(ValidationError::FailedValidation);
    };
    let signature = Signature::from_components(r, s);
    pubkey.verify(&commit_head.bytes, &signature)?;

    Ok(())
}
