//! `trible team` -- exact blob-native team capabilities.
//!
//! A team is identified by one Ed25519 trust-root public key. CONNECT uses
//! those exact 32 bytes as its capability resource. Claims and signatures are
//! ordinary content-addressed blobs: commands load one explicitly named leaf
//! credential and follow only its parent handles. There is no authority
//! collection, membership scan, or ambient credential registry.

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
    CapabilityAction, CapabilityAtom, CapabilityBlobHandle, CapabilityClaim, CapabilityGrant,
    CapabilityMode, CapabilityProof, CapabilityProofStep, CapabilityValidity,
};
use triblespace_core::repo::pile::{GetBlobError, Pile};
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStorePut};

use triblespace_net::protocol::{
    connect_capability_atom, decode_capability_proof, encode_capability_proof, ACTION_CONNECT,
    MAX_CAPABILITY_PROOF_BYTES,
};

const TEAM_ROOT_BYTES: usize = 32;
const MAX_INVITE_BYTES: usize = TEAM_ROOT_BYTES + MAX_CAPABILITY_PROOF_BYTES;

#[derive(Parser)]
pub enum Command {
    /// Create a team and issue the founder an exact CONNECT credential.
    Create {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Founder's signing key (generated at the conventional path if absent).
        #[arg(long)]
        key: Option<PathBuf>,
        /// Inclusive RFC 3339 lower validity bound (requires --valid-until).
        #[arg(long, value_parser = parse_epoch, requires = "valid_until")]
        valid_from: Option<Epoch>,
        /// Inclusive RFC 3339 upper validity bound (requires --valid-from).
        #[arg(long, value_parser = parse_epoch, requires = "valid_from")]
        valid_until: Option<Epoch>,
    },
    /// Issue one portable CONNECT invite from an exact parent credential.
    Invite {
        /// Path to the issuer's pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team trust-root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact parent credential (32-byte leaf signature-blob handle).
        #[arg(long)]
        parent: String,
        /// Issuer's existing signing key.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Invitee's Ed25519 public key (32-byte hex).
        #[arg(long)]
        invitee: String,
        /// Let the invitee issue child CONNECT credentials too.
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
        /// Invitee's existing signing key.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Portable invite bundle produced by `team invite`.
        #[arg(long)]
        invite: PathBuf,
    },
    /// Verify and show one exact credential ancestry.
    Show {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team trust-root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact credential (32-byte leaf signature-blob handle).
        #[arg(long)]
        credential: String,
    },
}

pub fn run(command: Command) -> Result<()> {
    match command {
        Command::Create {
            pile,
            key,
            valid_from,
            valid_until,
        } => run_create(pile, key, valid_from, valid_until),
        Command::Invite {
            pile,
            team_root,
            parent,
            key,
            invitee,
            delegate,
            valid_from,
            valid_until,
            out,
        } => run_invite(
            pile,
            team_root,
            parent,
            key,
            invitee,
            delegate,
            valid_from,
            valid_until,
            out,
        ),
        Command::Join { pile, key, invite } => run_join(pile, key, invite),
        Command::Show {
            pile,
            team_root,
            credential,
        } => run_show(pile, team_root, credential),
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

fn fresh_signing_key() -> Result<SigningKey> {
    let mut seed = [0; 32];
    getrandom::fill(&mut seed).map_err(|error| anyhow!("generate key: {error}"))?;
    Ok(SigningKey::from_bytes(&seed))
}

pub(crate) fn parse_team_root(text: &str) -> Result<VerifyingKey> {
    parse_public_key(text, "team root")
}

fn parse_public_key(text: &str, label: &str) -> Result<VerifyingKey> {
    let bytes = hex::decode(text).map_err(|error| anyhow!("decode {label} hex: {error}"))?;
    let raw: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("{label} must be 32 bytes"))?;
    VerifyingKey::from_bytes(&raw).map_err(|error| anyhow!("invalid {label}: {error}"))
}

pub(crate) fn parse_credential(text: &str) -> Result<CapabilityBlobHandle> {
    let bytes = hex::decode(text).map_err(|error| anyhow!("decode credential hex: {error}"))?;
    let raw: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("credential must be a 32-byte signature-blob handle"))?;
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

fn format_credential(credential: CapabilityBlobHandle) -> String {
    hex::encode(credential.raw)
}

fn load_proof(pile: &mut Pile, credential: CapabilityBlobHandle) -> Result<CapabilityProof> {
    let reader = pile.reader().context("open capability blob snapshot")?;
    CapabilityProof::load(credential, |handle| {
        let result: std::result::Result<Blob<SimpleArchive>, _> = reader.get(handle);
        match result {
            Ok(blob) => Ok(Some(blob)),
            Err(GetBlobError::BlobNotFound) => Ok(None),
            Err(error) => Err(anyhow!("read capability blob: {error}")),
        }
    })
    .map_err(|error| {
        anyhow!(
            "load credential {} by exact blob handles: {error}",
            format_credential(credential)
        )
    })
}

fn store_proof(pile: &mut Pile, proof: &CapabilityProof) -> Result<()> {
    for (index, step) in proof.steps().iter().enumerate() {
        // Rebuild from bytes before insertion so Pile never receives a blob
        // carrying an unverified cached handle. Proof verification likewise
        // hashes the bytes rather than trusting that cache.
        let claim = Blob::<SimpleArchive>::new(step.claim().bytes.clone());
        let expected_claim = claim.get_handle();
        let actual_claim = pile
            .put::<SimpleArchive, _>(claim)
            .map_err(|error| anyhow!("store capability claim {index}: {error:?}"))?;
        if actual_claim != expected_claim {
            bail!("stored capability claim {index} under a different content handle");
        }

        let signature = Blob::<SimpleArchive>::new(step.signature().bytes.clone());
        let expected_signature = signature.get_handle();
        let actual_signature = pile
            .put::<SimpleArchive, _>(signature)
            .map_err(|error| anyhow!("store capability signature {index}: {error:?}"))?;
        if actual_signature != expected_signature {
            bail!("stored capability signature {index} under a different content handle");
        }
    }
    Ok(())
}

/// Load and verify the exact CONNECT proof used by `pile net`.
pub(crate) fn resolve_connect_proof(
    pile: &mut Pile,
    team_root: VerifyingKey,
    credential: CapabilityBlobHandle,
    expected_subject: VerifyingKey,
) -> Result<CapabilityProof> {
    let proof = load_proof(pile, credential)?;
    proof
        .verify_claim(
            team_root,
            triblespace_core::clock::epoch_now(),
            CapabilityClaim::new(
                expected_subject,
                connect_atom(team_root),
                CapabilityMode::Invoke,
            ),
        )
        .map_err(|error| anyhow!("CONNECT credential rejected: {error}"))?;
    encode_capability_proof(&proof)
        .map_err(|error| anyhow!("CONNECT proof is not transport-portable: {error}"))?;
    Ok(proof)
}

fn write_invite(path: &Path, team_root: VerifyingKey, proof: &CapabilityProof) -> Result<()> {
    let encoded = encode_capability_proof(proof)
        .map_err(|error| anyhow!("encode capability proof: {error}"))?;
    let mut bundle = Vec::with_capacity(TEAM_ROOT_BYTES + encoded.len());
    bundle.extend_from_slice(&team_root.to_bytes());
    bundle.extend_from_slice(&encoded);
    fs::write(path, bundle).map_err(|error| anyhow!("write invite {}: {error}", path.display()))
}

fn read_invite(path: &Path) -> Result<(VerifyingKey, CapabilityProof)> {
    let file =
        fs::File::open(path).map_err(|error| anyhow!("open invite {}: {error}", path.display()))?;
    let mut bundle = Vec::with_capacity(MAX_INVITE_BYTES + 1);
    file.take((MAX_INVITE_BYTES + 1) as u64)
        .read_to_end(&mut bundle)
        .map_err(|error| anyhow!("read invite {}: {error}", path.display()))?;
    if bundle.len() > MAX_INVITE_BYTES {
        bail!("invite bundle exceeds the {MAX_INVITE_BYTES}-byte limit");
    }
    if bundle.len() <= TEAM_ROOT_BYTES {
        bail!("invite bundle is truncated");
    }
    let root_bytes: [u8; 32] = bundle[..TEAM_ROOT_BYTES]
        .try_into()
        .expect("a 32-byte prefix has a 32-byte array shape");
    let team_root = VerifyingKey::from_bytes(&root_bytes)
        .map_err(|error| anyhow!("invite has an invalid team root: {error}"))?;
    let proof = decode_capability_proof(&bundle[TEAM_ROOT_BYTES..])
        .map_err(|error| anyhow!("decode capability proof: {error}"))?;
    Ok((team_root, proof))
}

fn print_root_secret_warning() {
    eprintln!("TEAM ROOT SECRET -- STORE OFFLINE");
    eprintln!("Anyone holding it can issue independent root credentials for this team.");
}

fn run_create(
    pile_path: PathBuf,
    key_path: Option<PathBuf>,
    valid_from: Option<Epoch>,
    valid_until: Option<Epoch>,
) -> Result<()> {
    let founder = load_or_generate_signing_key(key_path, &pile_path)?;
    let team_root_key = fresh_signing_key()?;
    let team_root = team_root_key.verifying_key();
    let founder_step = CapabilityProofStep::issue(
        &team_root_key,
        CapabilityGrant::root(
            founder.verifying_key(),
            connect_atom(team_root),
            CapabilityMode::InvokeAndDelegate,
            validity(valid_from, valid_until)?,
        ),
    );
    let proof = CapabilityProof::new(vec![founder_step]);
    encode_capability_proof(&proof)
        .map_err(|error| anyhow!("founder proof is not portable: {error}"))?;
    with_pile(&pile_path, |pile| store_proof(pile, &proof))?;
    let credential = proof
        .credential()
        .expect("a one-step founder proof has one leaf credential");

    println!(
        "team root pubkey:     {}",
        hex::encode(team_root.to_bytes())
    );
    print_root_secret_warning();
    println!(
        "team root SECRET:     {}",
        hex::encode(team_root_key.to_bytes())
    );
    println!("founder credential:  {}", format_credential(credential));
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
    let parent_credential = parse_credential(&parent_text)?;
    let issuer_key = load_existing_signing_key(key_path, &pile_path)?;
    let issuer = issuer_key.verifying_key();
    let invitee = parse_public_key(&invitee_text, "invitee public key")?;
    let child_validity = validity(valid_from, valid_until)?;
    let instant = triblespace_core::clock::epoch_now();

    let proof = with_pile(&pile_path, |pile| {
        let parent_proof = load_proof(pile, parent_credential)?;
        let parent = parent_proof
            .verify_claim(
                team_root,
                instant,
                CapabilityClaim::new(issuer, connect_atom(team_root), CapabilityMode::Delegate),
            )
            .map_err(|error| anyhow!("parent credential rejected: {error}"))?;
        if parent.credential() != parent_credential {
            bail!("loaded proof does not end at the designated parent credential");
        }

        let child_mode = if delegate {
            CapabilityMode::InvokeAndDelegate
        } else {
            CapabilityMode::Invoke
        };
        if !parent.grant().mode().satisfies(child_mode) {
            bail!("parent credential does not contain the requested child mode");
        }

        let child = CapabilityProofStep::issue(
            &issuer_key,
            CapabilityGrant::delegated(
                parent_credential,
                invitee,
                connect_atom(team_root),
                child_mode,
                child_validity,
            ),
        );
        let mut steps = parent_proof.steps().to_vec();
        steps.push(child);
        let proof = CapabilityProof::new(steps);
        encode_capability_proof(&proof)
            .map_err(|error| anyhow!("child proof is not portable: {error}"))?;
        store_proof(pile, &proof)?;
        Ok(proof)
    })?;

    write_invite(&out, team_root, &proof)?;
    let credential = proof
        .credential()
        .expect("an issued child proof has one leaf credential");
    println!("issued credential:  {}", format_credential(credential));
    println!("invite bundle:      {}", out.display());
    println!("proof steps:        {}", proof.steps().len());
    Ok(())
}

fn run_join(pile_path: PathBuf, key_path: Option<PathBuf>, invite_path: PathBuf) -> Result<()> {
    let local_key = load_existing_signing_key(key_path, &pile_path)?;
    let (team_root, proof) = read_invite(&invite_path)?;
    let verified = proof
        .verify_claim(
            team_root,
            triblespace_core::clock::epoch_now(),
            CapabilityClaim::new(
                local_key.verifying_key(),
                connect_atom(team_root),
                CapabilityMode::Invoke,
            ),
        )
        .map_err(|error| anyhow!("invite proof rejected: {error}"))?;

    with_pile(&pile_path, |pile| store_proof(pile, &proof))?;

    println!("team root:           {}", hex::encode(team_root.to_bytes()));
    println!(
        "accepted credential:  {}",
        format_credential(verified.credential())
    );
    println!("proof steps:         {}", proof.steps().len());
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

fn print_grant(
    grant: CapabilityGrant,
    credential: CapabilityBlobHandle,
    issuer: VerifyingKey,
    indent: &str,
) {
    println!("{indent}credential: {}", format_credential(credential));
    println!("{indent}issuer:     {}", hex::encode(issuer.to_bytes()));
    println!(
        "{indent}subject:    {}",
        hex::encode(grant.subject().to_bytes())
    );
    println!(
        "{indent}action:     {}",
        action_label(grant.atom().action())
    );
    println!(
        "{indent}resource:   {}",
        hex::encode(grant.atom().resource().into_bytes())
    );
    match grant.parent() {
        Some(parent) => println!("{indent}parent:     {}", format_credential(parent)),
        None => println!("{indent}parent:     root"),
    }
    println!("{indent}mode:       {}", mode_label(grant.mode()));
    match grant.validity() {
        Some(validity) => {
            let (lower, upper) = validity.bounds();
            println!(
                "{indent}validity:   {}..={} TAI ns",
                lower.to_tai_duration().total_nanoseconds(),
                upper.to_tai_duration().total_nanoseconds()
            );
        }
        None => println!("{indent}validity:   unbounded"),
    }
}

fn run_show(pile_path: PathBuf, team_root_text: String, credential_text: String) -> Result<()> {
    let team_root = parse_team_root(&team_root_text)?;
    let credential = parse_credential(&credential_text)?;
    with_pile(&pile_path, |pile| {
        let proof = load_proof(pile, credential)?;
        let leaf = proof
            .steps()
            .last()
            .context("loaded capability proof is empty")?;
        let leaf_grant = CapabilityGrant::from_blob(leaf.claim().clone())
            .map_err(|error| anyhow!("decode leaf capability claim: {error}"))?;
        let verified = proof
            .verify_claim(
                team_root,
                triblespace_core::clock::epoch_now(),
                CapabilityClaim::new(
                    leaf_grant.subject(),
                    connect_atom(team_root),
                    leaf_grant.mode(),
                ),
            )
            .map_err(|error| anyhow!("credential proof rejected: {error}"))?;
        if verified.credential() != credential {
            bail!("loaded proof does not end at the designated credential");
        }

        println!("team root:  {}", hex::encode(team_root.to_bytes()));
        println!("credential: {}", format_credential(credential));
        println!("ancestry:   {} step(s), root to leaf", proof.steps().len());
        let mut issuer = team_root;
        for (level, step) in proof.steps().iter().enumerate() {
            let grant = CapabilityGrant::from_blob(step.claim().clone())
                .map_err(|error| anyhow!("decode capability claim at level {level}: {error}"))?;
            println!();
            println!("level {level}:");
            print_grant(grant, step.signature_handle(), issuer, "  ");
            issuer = grant.subject();
        }
        Ok(())
    })
}
