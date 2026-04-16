use crate::macros::entity;
use crate::macros::pattern;
use ed25519::signature::Signer;
use ed25519::Signature;
use ed25519_dalek::SignatureError;
use ed25519_dalek::SigningKey;
use ed25519_dalek::Verifier;
use ed25519_dalek::VerifyingKey;
use hifitime::prelude::*;
use itertools::Itertools;

use crate::blob::schemas::longstring::LongString;
use crate::blob::Blob;
use crate::find;
use crate::id::Id;
use crate::metadata;
use crate::prelude::blobschemas::SimpleArchive;
use crate::trible::TribleSet;
use crate::value::schemas::hash::{Blake3, Handle};
use crate::value::schemas::time::NsTAIInterval;
use crate::value::TryToValue;
use crate::value::Value;

/// Current TAI time as a collapsed `NsTAIInterval`. Used as
/// `metadata::updated_at` on every branch metadata blob so that peers can
/// order concurrent HEAD gossips without walking ancestor chains.
///
/// TAI is strictly monotone (no leap-second jumps). Wall-clock regressions
/// still mean subsequent publishes land "in the past" from the publisher's
/// view; receivers simply hold out until the publisher's clock catches up
/// and a fresher timestamp arrives.
fn now_updated_at() -> Value<NsTAIInterval> {
    let now = Epoch::now().unwrap_or_else(|_| Epoch::from_gregorian_utc(1970, 1, 1, 0, 0, 0, 0));
    (now, now).try_to_value().expect("same epoch is a valid point interval")
}

/// Compute a content-derived entity id for branch metadata so that
/// regenerating the same (branch_id, head, signer, name) produces the
/// same metadata blob hash.
///
/// This makes branch metadata blobs converge under gossip — pushing the
/// same state twice produces the same blob, so peers don't see spurious
/// "new HEAD" events.
fn derive_metadata_entity(
    branch_id: Id,
    head: &[u8; 32],
    signer_pub: &[u8; 32],
    name_handle: &[u8; 32],
) -> crate::id::ExclusiveId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&branch_id.raw());
    hasher.update(head);
    hasher.update(signer_pub);
    hasher.update(name_handle);
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let mut raw = [0u8; 16];
    raw.copy_from_slice(&bytes[16..32]);
    let id = Id::new(raw).expect("derived metadata entity id must be non-nil");
    crate::id::ExclusiveId::force(id)
}

/// Builds a metadata [`TribleSet`] describing a branch and signs it.
///
/// The metadata records the branch `name` handle, its unique `branch_id` and
/// optionally the handle of the initial commit. The commit handle is signed with
/// `signing_key` allowing the repository to verify its authenticity.
///
/// The metadata entity id is derived from the content (branch_id, head,
/// signer, name) so that identical state always produces an identical
/// metadata blob — important for gossip convergence in distributed sync.
pub fn branch_metadata(
    signing_key: &SigningKey,
    branch_id: Id,
    name: Value<Handle<Blake3, LongString>>,
    commit_head: Option<Blob<SimpleArchive>>,
) -> TribleSet {
    let mut metadata: TribleSet = Default::default();

    let head_bytes: [u8; 32] = match &commit_head {
        Some(blob) => blob.get_handle::<Blake3>().raw,
        None => [0u8; 32],
    };
    let signer_pub: [u8; 32] = signing_key.verifying_key().to_bytes();
    let metadata_entity = derive_metadata_entity(branch_id, &head_bytes, &signer_pub, &name.raw);

    metadata += entity! { &metadata_entity @  super::branch: branch_id  };
    if let Some(commit_head) = commit_head {
        let handle = commit_head.get_handle();
        let signature = signing_key.sign(&commit_head.bytes);

        metadata += entity! { &metadata_entity @
           super::head: handle,
           super::signed_by: signing_key.verifying_key(),
           super::signature_r: signature,
           super::signature_s: signature,
        };
    }
    metadata += entity! { &metadata_entity @  metadata::name: name  };
    metadata += entity! { &metadata_entity @ metadata::updated_at: now_updated_at() };

    metadata
}

/// Unsigned variant of [`branch`] used when authenticity is not required.
///
/// The resulting set omits any signature information and can therefore be
/// created without access to a private key. Like the signed variant, the
/// metadata entity id is content-derived for gossip convergence.
pub fn branch_unsigned(
    branch_id: Id,
    name: Value<Handle<Blake3, LongString>>,
    commit_head: Option<Blob<SimpleArchive>>,
) -> TribleSet {
    let head_bytes: [u8; 32] = match &commit_head {
        Some(blob) => blob.get_handle::<Blake3>().raw,
        None => [0u8; 32],
    };
    // Unsigned: use a zero "signer" key — still deterministic per (branch, head, name).
    let metadata_entity = derive_metadata_entity(branch_id, &head_bytes, &[0u8; 32], &name.raw);

    let mut metadata: TribleSet = Default::default();

    metadata += entity! { &metadata_entity @  super::branch: branch_id  };

    if let Some(commit_head) = commit_head {
        let handle = commit_head.get_handle();
        metadata += entity! { &metadata_entity @  super::head: handle  };
    }

    metadata += entity! { &metadata_entity @  metadata::name: name  };
    metadata += entity! { &metadata_entity @ metadata::updated_at: now_updated_at() };

    metadata
}

/// Error returned when branch signature verification fails.
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

/// Checks that the metadata signature matches the provided commit blob.
///
/// The function extracts the public key and signature from `metadata` and
/// verifies that it signs the `commit_head` blob. If the metadata is missing a
/// signature or contains multiple signature entities the appropriate
/// `ValidationError` variant is returned.
pub fn verify(
    commit_head: Blob<SimpleArchive>,
    metadata: TribleSet,
) -> Result<(), ValidationError> {
    let handle = commit_head.get_handle();
    let (pubkey, r, s) = match find!(
    (pubkey: Value<_>, r, s),
    pattern!(&metadata, [
    {
        super::head: handle,
        super::signed_by: ?pubkey,
        super::signature_r: ?r,
        super::signature_s: ?s,
    }]))
    .at_most_one()
    {
        Ok(Some(result)) => result,
        Ok(None) => return Err(ValidationError::MissingSignature),
        Err(_) => return Err(ValidationError::AmbiguousSignature),
    };

    let Ok(pubkey): Result<VerifyingKey, _> = pubkey.try_from_value() else {
        return Err(ValidationError::FailedValidation);
    };
    let signature = Signature::from_components(r, s);
    pubkey.verify(&commit_head.bytes, &signature)?;

    Ok(())
}
