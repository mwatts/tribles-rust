//! Capability-based authorization for triblespace networks.
//!
//! Implements a chain-of-trust capability system where:
//!
//! - A team has a single immutable root keypair (the "team root"), generated
//!   once at team creation and used to sign exactly one capability — the
//!   founder's. The team root never operates online; it's the constitutional
//!   document for the team's identity.
//! - All other capabilities chain off the founder's via delegation. Any holder
//!   of a capability can sign a sub-capability for someone else, as long as
//!   the sub-cap's scope is a subset of their own. Verification walks the
//!   chain back to the team root.
//! - Each capability link is two blobs: a `cap` blob (the claim) and a `sig`
//!   blob (the issuer's signature over the cap blob's bytes). For chains of
//!   length > 1, each non-root cap embeds its parent's signature inline as a
//!   sub-entity, which halves the cold-cache verification fetch count by
//!   eliminating a separate round-trip per intermediate signature.
//! - Signatures attest to the cap blob's canonical bytes (SimpleArchive's
//!   serialization is already canonical), not to a hash of those bytes —
//!   matching the existing commit-signing convention. This keeps signatures
//!   hash-agnostic across any future Blake3 migration.
//!
//! Scope is encoded as tribles inside the cap blob, anchored at
//! [`cap_scope_root`]. Permissions are tagged via [`metadata::tag`] linking
//! to constants like [`PERM_READ`], [`PERM_WRITE`], [`PERM_ADMIN`]. Optional
//! per-resource restrictions like [`scope_branch`] narrow a permission to a
//! specific branch.
//!
//! See `docs/sync_relay_auth_design.md` (or the `shared.pile` wiki fragment
//! titled "Sync Relay Auth Design") for the full design rationale.

use crate::id::Id;
use crate::id_hex;

/// Tag indicating a scope grants read access on the resources in scope.
pub const PERM_READ: Id = id_hex!("A75EED8224A553DD8002576E2E8A6823");
/// Tag indicating a scope grants write access on the resources in scope.
pub const PERM_WRITE: Id = id_hex!("C56AAF4191DD4FBB9F197B79435B881D");
/// Tag indicating a scope grants admin (delegation + revocation) authority.
pub const PERM_ADMIN: Id = id_hex!("EC68A0CBF9EF421F59A0A69ED80FD79F");

use crate::value::schemas::ed25519 as ed;
use crate::value::schemas::hash::Blake3;
use crate::blob::schemas::simplearchive::SimpleArchive;
use crate::value::schemas::genid::GenId;
use crate::value::schemas::hash::Handle;

triblespace_core_macros::attributes! {
    // ── Cap blob ──────────────────────────────────────────────────────
    /// The pubkey this capability authorizes. Must match the verified
    /// peer identity at connection time (i.e. the connecting peer's
    /// iroh `EndpointId`).
    "1A8A6A9D8CA1DA67FACAB373DE21233B" as pub cap_subject: ed::ED25519PublicKey;
    /// The pubkey of the entity that signed this capability. Must match
    /// the `signed_by` field of the accompanying signature blob.
    /// Recorded in the cap so verification can detect a sig-blob/cap
    /// issuer mismatch without an extra fetch.
    "2E9CD97ED0698FAF18EAEB74B5893685" as pub cap_issuer: ed::ED25519PublicKey;
    /// Entity id within the cap blob anchoring the scope tribles. The
    /// scope sub-graph hanging off this id encodes which permissions
    /// (and optionally which resources) the capability grants.
    "1A7DD2026BEFBE55A354CE10839CFDD6" as pub cap_scope_root: GenId;
    /// Handle of the parent cap blob in the chain. Absent on the
    /// founder's cap (the chain terminator), which is signed directly
    /// by the team root.
    "E825B3A8D387B4DAE1720B0EDCBFAA9E" as pub cap_parent: Handle<Blake3, SimpleArchive>;
    /// Entity id within the cap blob holding the parent's signature
    /// inline (the "targeted merge" optimisation — see module docs).
    /// The sub-entity carries `signed_by`, `signature_r`, `signature_s`
    /// reusing the existing commit-signature attribute conventions.
    /// Absent on the founder's cap.
    "008F7784A309CA9DEF007E4F63F87121" as pub cap_embedded_parent_sig: GenId;

    // ── Scope ─────────────────────────────────────────────────────────
    /// Optional restriction of a permission to a specific branch.
    /// Repeated when a permission applies to multiple branches; absent
    /// when the permission is unrestricted (applies to every branch
    /// the holder is otherwise authorised on).
    "46246789D627C1B0F81B21418E179DFD" as pub scope_branch: GenId;

    // ── Sig blob ──────────────────────────────────────────────────────
    /// Handle of the cap blob this signature attests to. The signature
    /// itself is over the cap blob's canonical bytes (i.e.
    /// `cap_blob.bytes`), not over the handle. SimpleArchive is already
    /// canonical, so the bytes the signer signs are exactly what the
    /// hasher hashes.
    "230E175A083E29155C860B38BD44F2F3" as pub sig_signs: Handle<Blake3, SimpleArchive>;
    // Note: sig_signer + sig_value (r/s) reuse the existing
    // `repo::signed_by`, `repo::signature_r`, `repo::signature_s`
    // attributes — same convention as commit signatures, plus
    // structural reuse (a sig blob has the same shape inside as the
    // signature portion of a commit's metadata blob).

    // ── Revocation blob ──────────────────────────────────────────────
    /// The pubkey being revoked. When this pubkey appears in any cap
    /// chain as either issuer or subject, that cap is invalidated.
    /// Revocation cascades transitively (revoking issuer K invalidates
    /// every cap K signed and every cap derived from those).
    "E146824999D1DA7F1F0E54025F52EE13" as pub rev_target: ed::ED25519PublicKey;
    // Note: revocation timestamp reuses `metadata::created_at`;
    // revocation signer + signature reuse `repo::signed_by` +
    // `repo::signature_r/s`.
}

/// Tag identifying a blob as a capability claim.
#[allow(dead_code)]
pub const KIND_CAPABILITY: Id = id_hex!("B8D76786ACD20F344A4E5CBFC0F75772");
/// Tag identifying a blob as a capability signature.
#[allow(dead_code)]
pub const KIND_CAPABILITY_SIG: Id = id_hex!("E6BB52CE6E02D51C3676ECE1EEA9094F");
/// Tag identifying a blob as a capability revocation.
#[allow(dead_code)]
pub const KIND_REVOCATION: Id = id_hex!("1EEAF2CF25A776547A26080E755D111C");

// Suppress dead-code warnings on the auto-generated `value_schema_*`
// helpers until the builder/verifier in subsequent commits consume them.
#[allow(dead_code)]
const _: () = ();
