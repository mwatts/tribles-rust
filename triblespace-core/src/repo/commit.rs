use crate::macros::pattern;
use ed25519::Signature;
use ed25519_dalek::SignatureError;
use ed25519_dalek::Verifier;
use ed25519_dalek::VerifyingKey;
use itertools::Itertools;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::Blob;
use crate::inline::Inline;
use crate::query::find;
use crate::trible::TribleSet;

/// Error returned when commit signature verification fails.
pub enum ValidationError {
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

/// Validates that the `metadata` blob genuinely signs the supplied commit
/// `content`.
///
/// Returns an error if the signature information is missing, malformed or does
/// not match the commit bytes.
pub fn verify(content: Blob<SimpleArchive>, metadata: TribleSet) -> Result<(), ValidationError> {
    let handle = content.get_handle();
    let (pubkey, r, s) = match find!(
    (pubkey: Inline<_>, r, s),
    pattern!(&metadata, [
    {
        super::content: handle,
        crate::attestation::signed_by: ?pubkey,
        crate::attestation::signature_r: ?r,
        crate::attestation::signature_s: ?s
    }]))
    .at_most_one()
    {
        Ok(Some(result)) => result,
        Ok(None) => return Err(ValidationError::MissingSignature),
        Err(_) => return Err(ValidationError::AmbiguousSignature),
    };

    let pubkey: VerifyingKey = pubkey.try_from_inline()?;
    let signature = Signature::from_components(r, s);
    pubkey.verify(&content.bytes, &signature)?;
    Ok(())
}
