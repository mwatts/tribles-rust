//! `trible team` -- direct, exact team capability proofs.
//!
//! A team is identified by one Ed25519 trust-root public key. CONNECT uses
//! those exact 32 bytes as its capability resource. A portable proof is the
//! canonical `K (S C K)+` body together with the exact ordered claim blobs it
//! names. Piles retain accepted proofs natively; there is no authority
//! collection, credential wallet, membership scan, or ambient registry.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Datelike, Timelike, Utc};
use clap::Parser;
use ed25519_dalek::{SigningKey, VerifyingKey};
use hifitime::Epoch;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::Blob;
use triblespace_core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode, CapabilityProofBundle,
    CapabilityProofId, CapabilityRequest, CapabilityValidity, MAX_CAPABILITY_PROOF_BUNDLE_BYTES,
};
use triblespace_core::repo::pile::{GetBlobError, Pile};
use triblespace_core::repo::proof::CapabilityProofStore;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut};

use triblespace_net::protocol::{connect_capability_atom, ACTION_CONNECT};

const MAX_INVITE_BYTES: usize = MAX_CAPABILITY_PROOF_BUNDLE_BYTES;

#[derive(Parser)]
pub enum Command {
    /// Create a team and issue the founder a direct CONNECT proof.
    Create {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Founder's signing key (generated at the conventional path if absent).
        #[arg(long)]
        key: Option<PathBuf>,
        /// Durable offline team-root key file. Defaults beside the pile.
        #[arg(long)]
        root_key: Option<PathBuf>,
        /// Inclusive RFC 3339 lower validity bound (requires --valid-until).
        #[arg(long, value_parser = parse_epoch, requires = "valid_until")]
        valid_from: Option<Epoch>,
        /// Inclusive RFC 3339 upper validity bound (requires --valid-from).
        #[arg(long, value_parser = parse_epoch, requires = "valid_from")]
        valid_until: Option<Epoch>,
    },
    /// Issue one portable CONNECT invite from an exact parent proof.
    Invite {
        /// Path to the issuer's pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team trust-root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact parent proof id (BLAKE3 of canonical proof bytes).
        #[arg(long)]
        parent_proof: String,
        /// Issuer's existing signing key.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Invitee's Ed25519 public key (32-byte hex).
        #[arg(long)]
        invitee: String,
        /// Let the invitee issue child CONNECT proofs too.
        #[arg(long)]
        delegate: bool,
        /// Inclusive RFC 3339 lower validity bound (requires --valid-until).
        #[arg(long, value_parser = parse_epoch, requires = "valid_until")]
        valid_from: Option<Epoch>,
        /// Inclusive RFC 3339 upper validity bound (requires --valid-from).
        #[arg(long, value_parser = parse_epoch, requires = "valid_from")]
        valid_until: Option<Epoch>,
        /// Portable public invite bundle to write.
        #[arg(long)]
        out: PathBuf,
    },
    /// Validate and import one portable invite into a local pile.
    Join {
        /// Path to the receiving pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Expected team trust-root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Invitee's existing signing key.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Portable invite bundle produced by `team invite`.
        #[arg(long)]
        invite: PathBuf,
    },
    /// Verify and show one exact proof ancestry.
    Show {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team trust-root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact proof id (BLAKE3 of canonical proof bytes).
        #[arg(long)]
        proof: String,
    },
}

pub fn run(command: Command) -> Result<()> {
    match command {
        Command::Create {
            pile,
            key,
            root_key,
            valid_from,
            valid_until,
        } => run_create(pile, key, root_key, valid_from, valid_until),
        Command::Invite {
            pile,
            team_root,
            parent_proof,
            key,
            invitee,
            delegate,
            valid_from,
            valid_until,
            out,
        } => run_invite(
            pile,
            team_root,
            parent_proof,
            key,
            invitee,
            delegate,
            valid_from,
            valid_until,
            out,
        ),
        Command::Join {
            pile,
            team_root,
            key,
            invite,
        } => run_join(pile, team_root, key, invite),
        Command::Show {
            pile,
            team_root,
            proof,
        } => run_show(pile, team_root, proof),
    }
}

fn open_pile(path: &Path) -> Result<Pile> {
    crate::cli::pile::open_refreshed(path)
}

fn with_pile<T>(path: &Path, operation: impl FnOnce(&mut Pile) -> Result<T>) -> Result<T> {
    let mut pile = open_pile(path)?;
    let result = operation(&mut pile);
    let close_error = pile.close().err();
    match (result, close_error) {
        (Ok(value), None) => Ok(value),
        (Ok(_), Some(error)) => Err(anyhow!("pile close: {error:?}")),
        (Err(error), None) => Err(error),
        (Err(error), Some(close_error)) => Err(anyhow!(
            "{error:#}; additionally pile close failed: {close_error:?}"
        )),
    }
}

fn load_or_generate_signing_key(path: Option<PathBuf>, pile: &Path) -> Result<SigningKey> {
    let path = triblespace_core::signing_key_file::resolve_path(path.as_deref(), pile);
    triblespace_core::signing_key_file::init(&path).map_err(Into::into)
}

fn load_existing_signing_key(path: Option<PathBuf>, pile: &Path) -> Result<SigningKey> {
    let path = triblespace_core::signing_key_file::resolve_path(path.as_deref(), pile);
    triblespace_core::signing_key_file::load_existing(&path).map_err(Into::into)
}

fn load_or_generate_root_key(path: Option<PathBuf>, pile: &Path) -> Result<(PathBuf, SigningKey)> {
    let path = path.unwrap_or_else(|| pile.with_extension("team-root.key"));
    let key = triblespace_core::signing_key_file::init(&path)?;
    Ok((path, key))
}

pub(crate) fn parse_team_root(text: &str) -> Result<VerifyingKey> {
    parse_public_key(text, "team root")
}

pub(crate) fn parse_public_key(text: &str, label: &str) -> Result<VerifyingKey> {
    let bytes = hex::decode(text).map_err(|error| anyhow!("decode {label} hex: {error}"))?;
    let raw: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("{label} must be 32 bytes"))?;
    VerifyingKey::from_bytes(&raw).map_err(|error| anyhow!("invalid {label}: {error}"))
}

pub(crate) fn parse_proof_id(text: &str) -> Result<CapabilityProofId> {
    let bytes = hex::decode(text).map_err(|error| anyhow!("decode proof id hex: {error}"))?;
    let raw: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("proof id must be a 32-byte BLAKE3 digest"))?;
    Ok(triblespace_core::inline::Inline::new(raw))
}

fn parse_epoch(text: &str) -> std::result::Result<Epoch, String> {
    let parsed = DateTime::parse_from_rfc3339(text)
        .map_err(|error| format!("invalid RFC 3339 instant: {error}"))?;
    let utc = parsed.with_timezone(&Utc);
    Epoch::maybe_from_gregorian_utc(
        utc.year(),
        utc.month() as u8,
        utc.day() as u8,
        utc.hour() as u8,
        utc.minute() as u8,
        utc.second() as u8,
        utc.nanosecond(),
    )
    .map_err(|error| format!("instant is outside the supported UTC range: {error}"))
}

fn validity(
    valid_from: Option<Epoch>,
    valid_until: Option<Epoch>,
) -> Result<Option<CapabilityValidity>> {
    match (valid_from, valid_until) {
        (None, None) => Ok(None),
        (Some(lower), Some(upper)) => CapabilityValidity::new(lower, upper)
            .map(Some)
            .map_err(|error| anyhow!(error)),
        _ => bail!("--valid-from and --valid-until must be supplied together"),
    }
}

fn connect_atom(team_root: VerifyingKey) -> CapabilityAtom {
    connect_capability_atom(team_root)
}

fn format_proof_id(id: CapabilityProofId) -> String {
    hex::encode(id.raw)
}

/// Load one explicitly named proof and only the claim blobs it names.
pub(crate) fn load_capability_bundle(
    pile: &mut Pile,
    proof_id: CapabilityProofId,
) -> Result<CapabilityProofBundle> {
    let proof = pile
        .proof(proof_id)
        .map_err(|error| anyhow!("read capability proof store: {error}"))?
        .with_context(|| {
            format!(
                "capability proof {} is not present",
                format_proof_id(proof_id)
            )
        })?;
    let reader = pile.reader().context("open capability claim snapshot")?;
    let mut claims = Vec::with_capacity(proof.step_count());
    for (step, handle) in proof.claim_handles().enumerate() {
        let claim: Blob<SimpleArchive> = match reader.get(handle) {
            Ok(claim) => claim,
            Err(GetBlobError::BlobNotFound) => {
                bail!(
                    "capability proof {} is missing claim {step} ({})",
                    format_proof_id(proof_id),
                    hex::encode(handle.raw)
                )
            }
            Err(error) => return Err(anyhow!("read capability claim {step}: {error}")),
        };
        claims.push(claim);
    }
    Ok(CapabilityProofBundle::new(proof, claims))
}

/// Persist an accepted claim closure first, then publish its native proof root.
pub(crate) fn store_capability_bundle(
    pile: &mut Pile,
    bundle: &CapabilityProofBundle,
) -> Result<()> {
    if bundle.claims().len() != bundle.proof().step_count() {
        bail!(
            "capability bundle has {} claims for {} proof steps",
            bundle.claims().len(),
            bundle.proof().step_count()
        );
    }
    for (step, (claim, expected)) in bundle
        .claims()
        .iter()
        .zip(bundle.proof().claim_handles())
        .enumerate()
    {
        // Rebuild from bytes so no untrusted cached handle crosses into Pile.
        let claim = Blob::<SimpleArchive>::new(claim.bytes.clone());
        let actual = pile
            .put::<SimpleArchive, _>(claim)
            .map_err(|error| anyhow!("store capability claim {step}: {error:?}"))?;
        if actual != expected {
            bail!(
                "capability claim {step} hashes to {} instead of signed {}",
                hex::encode(actual.raw),
                hex::encode(expected.raw)
            );
        }
    }
    pile.insert_proof(bundle.proof().clone())
        .map_err(|error| anyhow!("store native capability proof: {error}"))
}

/// Load and verify the exact CONNECT proof used by `pile net`.
pub(crate) fn resolve_connect_bundle(
    pile: &mut Pile,
    team_root: VerifyingKey,
    proof_id: CapabilityProofId,
    expected_subject: VerifyingKey,
) -> Result<CapabilityProofBundle> {
    let bundle = load_capability_bundle(pile, proof_id)?;
    bundle
        .verify(
            team_root,
            triblespace_core::clock::epoch_now(),
            expected_subject,
            CapabilityRequest::new(connect_atom(team_root), CapabilityMode::Invoke),
        )
        .map_err(|error| anyhow!("CONNECT proof rejected: {error}"))?;
    bundle
        .to_bytes()
        .map_err(|error| anyhow!("CONNECT proof is not transport-portable: {error}"))?;
    Ok(bundle)
}

fn write_invite(path: &Path, bundle: &CapabilityProofBundle) -> Result<()> {
    let encoded = bundle
        .to_bytes()
        .map_err(|error| anyhow!("encode capability proof bundle: {error}"))?;
    fs::write(path, encoded).map_err(|error| anyhow!("write invite {}: {error}", path.display()))
}

fn read_invite(path: &Path) -> Result<CapabilityProofBundle> {
    let file =
        fs::File::open(path).map_err(|error| anyhow!("open invite {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(MAX_INVITE_BYTES + 1);
    file.take((MAX_INVITE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| anyhow!("read invite {}: {error}", path.display()))?;
    if bytes.len() > MAX_INVITE_BYTES {
        bail!("invite bundle exceeds the {MAX_INVITE_BYTES}-byte limit");
    }
    let bundle = CapabilityProofBundle::from_bytes(&bytes)
        .map_err(|error| anyhow!("decode capability proof bundle: {error}"))?;
    Ok(bundle)
}

fn print_root_key_warning() {
    eprintln!("TEAM ROOT KEY -- STORE OFFLINE");
    eprintln!("Anyone holding it can issue independent root proofs for this team.");
}

fn run_create(
    pile_path: PathBuf,
    key_path: Option<PathBuf>,
    root_key_path: Option<PathBuf>,
    valid_from: Option<Epoch>,
    valid_until: Option<Epoch>,
) -> Result<()> {
    let founder = load_or_generate_signing_key(key_path, &pile_path)?;
    let (root_key_path, team_root_key) = load_or_generate_root_key(root_key_path, &pile_path)?;
    let team_root = team_root_key.verifying_key();
    let claim = CapabilityClaim::root(
        connect_atom(team_root),
        CapabilityMode::InvokeAndDelegate,
        validity(valid_from, valid_until)?,
    );
    let bundle = CapabilityProofBundle::issue_root(&team_root_key, claim, founder.verifying_key())
        .map_err(|error| anyhow!("issue founder proof: {error}"))?;
    with_pile(&pile_path, |pile| store_capability_bundle(pile, &bundle))?;

    println!("team root pubkey: {}", hex::encode(team_root.to_bytes()));
    print_root_key_warning();
    println!("team root key:    {}", root_key_path.display());
    println!("founder proof id: {}", format_proof_id(bundle.proof().id()));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_invite(
    pile_path: PathBuf,
    team_root_text: String,
    parent_text: String,
    key_path: Option<PathBuf>,
    invitee_text: String,
    delegate: bool,
    valid_from: Option<Epoch>,
    valid_until: Option<Epoch>,
    out: PathBuf,
) -> Result<()> {
    let team_root = parse_team_root(&team_root_text)?;
    let parent_id = parse_proof_id(&parent_text)?;
    let issuer_key = load_existing_signing_key(key_path, &pile_path)?;
    let issuer = issuer_key.verifying_key();
    let invitee = parse_public_key(&invitee_text, "invitee public key")?;
    let child_validity = validity(valid_from, valid_until)?;
    let instant = triblespace_core::clock::epoch_now();

    let bundle = with_pile(&pile_path, |pile| {
        let parent_bundle = load_capability_bundle(pile, parent_id)?;
        let parent = parent_bundle
            .verify(
                team_root,
                instant,
                issuer,
                CapabilityRequest::new(connect_atom(team_root), CapabilityMode::Delegate),
            )
            .map_err(|error| anyhow!("parent proof rejected: {error}"))?;

        let child_mode = if delegate {
            CapabilityMode::InvokeAndDelegate
        } else {
            CapabilityMode::Invoke
        };
        let child = CapabilityClaim::delegated(
            parent.claim_handle(),
            connect_atom(team_root),
            child_mode,
            child_validity,
        );
        let bundle = parent
            .delegate(&issuer_key, child, invitee)
            .map_err(|error| anyhow!("issue child proof: {error}"))?;
        store_capability_bundle(pile, &bundle)?;
        Ok(bundle)
    })?;

    write_invite(&out, &bundle)?;
    println!("issued proof id: {}", format_proof_id(bundle.proof().id()));
    println!("invite bundle:   {}", out.display());
    println!("proof steps:     {}", bundle.proof().step_count());
    Ok(())
}

fn run_join(
    pile_path: PathBuf,
    team_root_text: String,
    key_path: Option<PathBuf>,
    invite_path: PathBuf,
) -> Result<()> {
    let local_key = load_existing_signing_key(key_path, &pile_path)?;
    let team_root = parse_team_root(&team_root_text)?;
    let bundle = read_invite(&invite_path)?;
    bundle
        .verify(
            team_root,
            triblespace_core::clock::epoch_now(),
            local_key.verifying_key(),
            CapabilityRequest::new(connect_atom(team_root), CapabilityMode::Invoke),
        )
        .map_err(|error| anyhow!("invite proof rejected: {error}"))?;
    with_pile(&pile_path, |pile| store_capability_bundle(pile, &bundle))?;

    println!("team root:        {}", hex::encode(team_root.to_bytes()));
    println!("accepted proof:   {}", format_proof_id(bundle.proof().id()));
    println!("proof steps:      {}", bundle.proof().step_count());
    Ok(())
}

fn action_label(action: CapabilityAction) -> String {
    if action.id() == ACTION_CONNECT {
        "CONNECT".to_owned()
    } else {
        format!("{:X}", action.id())
    }
}

fn mode_label(mode: CapabilityMode) -> &'static str {
    match mode {
        CapabilityMode::Invoke => "invoke",
        CapabilityMode::Delegate => "delegate",
        CapabilityMode::InvokeAndDelegate => "invoke+delegate",
    }
}

fn print_claim(
    claim: CapabilityClaim,
    handle: triblespace_core::capability::CapabilityClaimHandle,
) {
    println!("  claim:      {}", hex::encode(handle.raw));
    match claim.parent() {
        Some(parent) => println!("  parent:     {}", hex::encode(parent.raw)),
        None => println!("  parent:     root"),
    }
    println!("  action:     {}", action_label(claim.atom().action()));
    println!(
        "  resource:   {}",
        hex::encode(claim.atom().resource().into_bytes())
    );
    println!("  mode:       {}", mode_label(claim.mode()));
    match claim.validity() {
        Some(validity) => {
            let (lower, upper) = validity.bounds();
            println!(
                "  validity:   {}..={} TAI ns",
                lower.to_tai_duration().total_nanoseconds(),
                upper.to_tai_duration().total_nanoseconds()
            );
        }
        None => println!("  validity:   unbounded"),
    }
}

fn run_show(pile_path: PathBuf, team_root_text: String, proof_text: String) -> Result<()> {
    let team_root = parse_team_root(&team_root_text)?;
    let proof_id = parse_proof_id(&proof_text)?;
    with_pile(&pile_path, |pile| {
        let bundle = load_capability_bundle(pile, proof_id)?;
        let mut claims = bundle
            .claims()
            .iter()
            .cloned()
            .enumerate()
            .map(|(step, blob)| {
                CapabilityClaim::from_blob(blob)
                    .map_err(|error| anyhow!("decode capability claim {step}: {error}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let first = *claims.first().context("capability proof has no claims")?;
        let mut effective_mode = first.mode();
        for claim in &claims[1..] {
            effective_mode = effective_mode
                .meet(claim.mode())
                .context("capability proof has an empty mode meet")?;
        }
        let required = if effective_mode.satisfies(CapabilityMode::Invoke) {
            CapabilityMode::Invoke
        } else {
            CapabilityMode::Delegate
        };
        let verified = bundle
            .verify(
                team_root,
                triblespace_core::clock::epoch_now(),
                bundle.proof().leaf_key(),
                CapabilityRequest::new(connect_atom(team_root), required),
            )
            .map_err(|error| anyhow!("capability proof rejected: {error}"))?;

        println!("team root:      {}", hex::encode(team_root.to_bytes()));
        println!("proof id:       {}", format_proof_id(proof_id));
        println!(
            "leaf principal:  {}",
            hex::encode(verified.subject().to_bytes())
        );
        println!("effective mode: {}", mode_label(verified.effective_mode()));
        println!("ancestry:       {} step(s), root to leaf", claims.len());
        for (level, (claim, handle)) in claims
            .drain(..)
            .zip(bundle.proof().claim_handles())
            .enumerate()
        {
            println!();
            println!("level {level}:");
            print_claim(claim, handle);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use triblespace_core::repo::pile::WantRewritePolicy;
    use triblespace_core::repo::RetentionRoots;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    #[test]
    fn native_proof_retains_complete_claim_closure_through_pile_rewrite() {
        let root = key(1);
        let issuer = key(2);
        let member = key(3);
        let root_claim = CapabilityClaim::root(
            connect_atom(root.verifying_key()),
            CapabilityMode::InvokeAndDelegate,
            None,
        );
        let root_bundle =
            CapabilityProofBundle::issue_root(&root, root_claim, issuer.verifying_key()).unwrap();
        let verified = root_bundle
            .verify(
                root.verifying_key(),
                triblespace_core::clock::epoch_now(),
                issuer.verifying_key(),
                CapabilityRequest::new(
                    connect_atom(root.verifying_key()),
                    CapabilityMode::Delegate,
                ),
            )
            .unwrap();
        let leaf_claim = CapabilityClaim::delegated(
            verified.claim_handle(),
            connect_atom(root.verifying_key()),
            CapabilityMode::Invoke,
            None,
        );
        let bundle = verified
            .delegate(&issuer, leaf_claim, member.verifying_key())
            .unwrap();
        let proof_id = bundle.proof().id();

        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source.pile");
        let destination_path = directory.path().join("destination.pile");
        std::fs::File::create(&source_path).unwrap();
        std::fs::File::create(&destination_path).unwrap();
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();

        store_capability_bundle(&mut source, &bundle).unwrap();
        source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();

        let retained = load_capability_bundle(&mut destination, proof_id).unwrap();
        assert_eq!(retained, bundle);
        retained
            .verify(
                root.verifying_key(),
                triblespace_core::clock::epoch_now(),
                member.verifying_key(),
                CapabilityRequest::new(connect_atom(root.verifying_key()), CapabilityMode::Invoke),
            )
            .unwrap();

        destination.close().unwrap();
        source.close().unwrap();
    }
}
