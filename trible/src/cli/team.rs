//! `trible team` -- positive team authority management.
//!
//! A team is rooted in one Ed25519 key and has one public, grow-only
//! authority collection. Each grant is a signed collection commit naming one
//! direct subject, one exact resource, one action, and optionally one exact
//! delegating parent occurrence. Invites carry a bounded, self-contained
//! root-to-leaf proof; joining validates it before importing its public
//! evidence into the local pile.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use ed25519_dalek::{SigningKey, VerifyingKey};

use triblespace_core::authority::{
    self, AcceptedAuthorityGrant, AuthorityClaim, AuthorityGrant, AuthorityMode, AuthorityProof,
};
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, IntoBlob};
use triblespace_core::collection::{CollectionRecord, CollectionStore};
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::ed25519::ED25519PublicKey;
use triblespace_core::inline::Inline;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::BlobStorePut;
use triblespace_core::trible::TribleSet;

use triblespace_net::protocol::{
    decode_authority_proof, encode_authority_proof, ACTION_CONNECT, MAX_AUTHORITY_PROOF_BYTES,
    MAX_AUTHORITY_PROOF_STEPS,
};

const TEAM_ROOT_BYTES: usize = 32;
const MAX_INVITE_BYTES: usize = TEAM_ROOT_BYTES + MAX_AUTHORITY_PROOF_BYTES;

#[derive(Parser)]
pub enum Command {
    /// Create a team and grant the founder CONNECT authority with delegation.
    Create {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Founder's signing key (generated at the conventional path if absent).
        #[arg(long)]
        key: Option<PathBuf>,
    },
    /// Issue one portable CONNECT invite from an exact delegating grant.
    Invite {
        /// Path to the issuer's pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact parent grant occurrence (16-byte hex commit id).
        #[arg(long)]
        parent: String,
        /// Issuer's existing signing key.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Invitee's Ed25519 public key (32-byte hex).
        #[arg(long)]
        invitee: String,
        /// Let the invitee issue child CONNECT grants too.
        #[arg(long)]
        delegate: bool,
        /// Portable public invite bundle to write.
        #[arg(long)]
        out: PathBuf,
    },
    /// Validate and import a portable invite into a local pile.
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
    /// List accepted grants and inert candidate diagnostics for one team.
    List {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
    },
    /// Show the accepted root-to-leaf ancestry of one exact grant occurrence.
    Show {
        /// Path to the local pile file.
        #[arg(long)]
        pile: PathBuf,
        /// Team root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact grant occurrence (16-byte hex commit id).
        #[arg(long)]
        grant: String,
    },
}

pub fn run(command: Command) -> Result<()> {
    match command {
        Command::Create { pile, key } => run_create(pile, key),
        Command::Invite {
            pile,
            team_root,
            parent,
            key,
            invitee,
            delegate,
            out,
        } => run_invite(pile, team_root, parent, key, invitee, delegate, out),
        Command::Join { pile, key, invite } => run_join(pile, key, invite),
        Command::List { pile, team_root } => run_list(pile, team_root),
        Command::Show {
            pile,
            team_root,
            grant,
        } => run_show(pile, team_root, grant),
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
    let bytes = hex::decode(text).map_err(|error| anyhow!("decode team root hex: {error}"))?;
    let raw: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("team root must be 32 bytes"))?;
    VerifyingKey::from_bytes(&raw).map_err(|error| anyhow!("invalid team root: {error}"))
}

fn parse_public_key(text: &str, label: &str) -> Result<VerifyingKey> {
    let bytes = hex::decode(text).map_err(|error| anyhow!("decode {label} hex: {error}"))?;
    let raw: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("{label} must be 32 bytes"))?;
    VerifyingKey::from_bytes(&raw).map_err(|error| anyhow!("invalid {label}: {error}"))
}

pub(crate) fn parse_grant_id(text: &str) -> Result<Id> {
    Id::from_hex(text).ok_or_else(|| anyhow!("grant must be a nonzero 16-byte hex id"))
}

fn subject_inline(key: VerifyingKey) -> Inline<ED25519PublicKey> {
    Inline::new(key.to_bytes())
}

fn connect_grant_is_exact(grant: AuthorityGrant, team_root: VerifyingKey) -> bool {
    grant.action() == ACTION_CONNECT && grant.resource() == authority::collection(team_root)
}

/// Resolve and build the exact CONNECT proof used by `pile net`.
pub(crate) fn resolve_connect_proof(
    pile: &mut Pile,
    team_root: VerifyingKey,
    grant_id: Id,
    expected_subject: VerifyingKey,
) -> Result<AuthorityProof> {
    let resolution = authority::resolve_authority(pile, team_root)
        .map_err(|error| anyhow!("resolve team authority: {error}"))?;
    let accepted = resolution.grant(grant_id).ok_or_else(|| {
        let diagnostic = resolution
            .diagnostics()
            .iter()
            .find(|diagnostic| diagnostic.commit() == grant_id);
        match diagnostic {
            Some(diagnostic) => anyhow!("grant {grant_id:X} is inert: {diagnostic:?}"),
            None => anyhow!("grant {grant_id:X} is not present in this team authority collection"),
        }
    })?;
    let grant = accepted.grant();
    if grant.subject() != subject_inline(expected_subject) {
        bail!("grant {grant_id:X} belongs to a different subject key");
    }
    if !connect_grant_is_exact(grant, team_root) || !grant.invoke() {
        bail!("grant {grant_id:X} does not invoke CONNECT on this team's authority collection");
    }
    let proof = resolution.proof(grant_id).ok_or_else(|| {
        anyhow!("accepted grant {grant_id:X} has no reconstructible ancestry proof")
    })?;
    encode_authority_proof(&proof)
        .map_err(|error| anyhow!("CONNECT proof is not transport-portable: {error}"))?;
    Ok(proof)
}

fn write_invite(path: &Path, team_root: VerifyingKey, proof: &AuthorityProof) -> Result<()> {
    let encoded = encode_authority_proof(proof)
        .map_err(|error| anyhow!("encode authority proof: {error}"))?;
    let mut bundle = Vec::with_capacity(TEAM_ROOT_BYTES + encoded.len());
    bundle.extend_from_slice(&team_root.to_bytes());
    bundle.extend_from_slice(&encoded);
    fs::write(path, bundle).map_err(|error| anyhow!("write invite {}: {error}", path.display()))
}

fn read_invite(path: &Path) -> Result<(VerifyingKey, AuthorityProof)> {
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
    let proof = decode_authority_proof(&bundle[TEAM_ROOT_BYTES..])
        .map_err(|error| anyhow!("decode authority proof: {error}"))?;
    Ok((team_root, proof))
}

fn print_root_secret_warning() {
    eprintln!("TEAM ROOT SECRET -- STORE OFFLINE");
    eprintln!("Anyone holding it can create independent root grants for this team.");
}

fn run_create(pile_path: PathBuf, key_path: Option<PathBuf>) -> Result<()> {
    let founder = load_or_generate_signing_key(key_path, &pile_path)?;
    let team_root_key = fresh_signing_key()?;
    let team_root = team_root_key.verifying_key();
    let founder_grant = AuthorityGrant::root(
        founder.verifying_key(),
        authority::collection(team_root),
        ACTION_CONNECT,
        AuthorityMode::InvokeAndDelegate,
    );
    let commit = with_pile(&pile_path, |pile| {
        authority::publish_grant(pile, team_root, &team_root_key, founder_grant)
            .map_err(|error| anyhow!("publish founder grant: {error:?}"))
    })?;

    println!("team root pubkey:  {}", hex::encode(team_root.to_bytes()));
    print_root_secret_warning();
    println!(
        "team root SECRET:  {}",
        hex::encode(team_root_key.to_bytes())
    );
    println!("founder grant:     {:X}", commit.id());
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
    out: PathBuf,
) -> Result<()> {
    let team_root = parse_team_root(&team_root_text)?;
    let parent_id = parse_grant_id(&parent_text)?;
    let issuer_key = load_existing_signing_key(key_path, &pile_path)?;
    let issuer = issuer_key.verifying_key();
    let invitee = parse_public_key(&invitee_text, "invitee public key")?;

    let (commit, proof) = with_pile(&pile_path, |pile| {
        let resolution = authority::resolve_authority(pile, team_root)
            .map_err(|error| anyhow!("resolve team authority: {error}"))?;
        let parent = resolution
            .grant(parent_id)
            .ok_or_else(|| anyhow!("parent grant {parent_id:X} is not accepted"))?;
        let parent_grant = parent.grant();
        if parent_grant.subject() != subject_inline(issuer) {
            bail!("parent grant {parent_id:X} belongs to a different issuer key");
        }
        if !connect_grant_is_exact(parent_grant, team_root) {
            bail!("parent grant {parent_id:X} does not govern CONNECT for this team");
        }
        if !parent_grant.delegate() {
            bail!("parent grant {parent_id:X} does not permit delegation");
        }
        let parent_proof = resolution
            .proof(parent_id)
            .ok_or_else(|| anyhow!("accepted parent grant has no reconstructible proof"))?;
        encode_authority_proof(&parent_proof)
            .map_err(|error| anyhow!("parent authority proof is not portable: {error}"))?;
        if parent_proof.steps().len() >= MAX_AUTHORITY_PROOF_STEPS {
            bail!(
                "parent proof already has the transport maximum of {MAX_AUTHORITY_PROOF_STEPS} steps"
            );
        }

        let mode = if delegate {
            AuthorityMode::InvokeAndDelegate
        } else {
            AuthorityMode::Invoke
        };
        let child = AuthorityGrant::delegated(
            parent_id,
            invitee,
            authority::collection(team_root),
            ACTION_CONNECT,
            mode,
        );
        let commit = authority::publish_grant(pile, team_root, &issuer_key, child)
            .map_err(|error| anyhow!("publish invite grant: {error:?}"))?;

        let updated = authority::resolve_authority(pile, team_root)
            .map_err(|error| anyhow!("resolve published invite: {error}"))?;
        let proof = updated
            .proof(commit.id())
            .ok_or_else(|| anyhow!("published invite grant was not accepted"))?;
        Ok((commit, proof))
    })?;

    write_invite(&out, team_root, &proof)?;
    println!("issued grant:      {:X}", commit.id());
    println!("invite bundle:     {}", out.display());
    println!("proof steps:       {}", proof.steps().len());
    Ok(())
}

fn run_join(pile_path: PathBuf, key_path: Option<PathBuf>, invite_path: PathBuf) -> Result<()> {
    let local_key = load_existing_signing_key(key_path, &pile_path)?;
    let (team_root, proof) = read_invite(&invite_path)?;
    let claim = AuthorityClaim::new(
        local_key.verifying_key(),
        ACTION_CONNECT,
        authority::collection(team_root),
        AuthorityMode::Invoke,
    );
    let leaf = proof
        .verify_claim(team_root, claim)
        .map_err(|error| anyhow!("invite proof rejected: {error}"))?;
    let grant = leaf.grant();
    if grant.subject() != subject_inline(local_key.verifying_key()) {
        bail!("invite belongs to a different local signing key");
    }
    if !connect_grant_is_exact(grant, team_root) || !grant.invoke() {
        bail!("invite does not invoke CONNECT on this team's authority collection");
    }

    with_pile(&pile_path, |pile| {
        let descriptor: Blob<SimpleArchive> =
            authority::descriptor(team_root).into_facts().to_blob();
        pile.put::<SimpleArchive, _>(descriptor)
            .map_err(|error| anyhow!("store authority descriptor: {error:?}"))?;
        let empty_metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
        pile.put::<SimpleArchive, _>(empty_metadata)
            .map_err(|error| anyhow!("store empty metadata: {error:?}"))?;
        for step in proof.steps() {
            pile.put::<SimpleArchive, _>(step.data().clone())
                .map_err(|error| anyhow!("store grant data: {error:?}"))?;
            CollectionStore::insert(pile, CollectionRecord::Commit(step.commit()))
                .map_err(|error| anyhow!("store grant commit: {error:?}"))?;
        }
        Ok(())
    })?;

    println!("team root:         {}", hex::encode(team_root.to_bytes()));
    println!("accepted grant:    {:X}", leaf.commit().id());
    println!("proof steps:       {}", proof.steps().len());
    Ok(())
}

fn action_label(action: Id) -> String {
    if action == ACTION_CONNECT {
        "CONNECT".to_owned()
    } else if action == authority::ACTION_WRITE {
        "WRITE".to_owned()
    } else {
        format!("{action:X}")
    }
}

fn print_grant(accepted: &AcceptedAuthorityGrant, indent: &str) {
    let commit = accepted.commit();
    let grant = accepted.grant();
    println!("{indent}grant:    {:X}", commit.id());
    println!("{indent}issuer:   {}", hex::encode(commit.public_key().raw));
    println!("{indent}subject:  {}", hex::encode(grant.subject().raw));
    println!("{indent}action:   {}", action_label(grant.action()));
    println!("{indent}resource: {}", hex::encode(grant.resource().raw));
    match grant.parent() {
        Some(parent) => println!("{indent}parent:   {parent:X}"),
        None => println!("{indent}parent:   root"),
    }
    println!("{indent}invoke:   {}", grant.invoke());
    println!("{indent}delegate: {}", grant.delegate());
}

fn run_list(pile_path: PathBuf, team_root_text: String) -> Result<()> {
    let team_root = parse_team_root(&team_root_text)?;
    with_pile(&pile_path, |pile| {
        let resolution = authority::resolve_authority(pile, team_root)
            .map_err(|error| anyhow!("resolve team authority: {error}"))?;
        println!("team root:       {}", hex::encode(team_root.to_bytes()));
        println!("accepted grants: {}", resolution.grants().count());
        for accepted in resolution.grants() {
            println!();
            print_grant(accepted, "  ");
        }
        println!();
        println!("diagnostics:     {}", resolution.diagnostics().len());
        for diagnostic in resolution.diagnostics() {
            println!("  {:X}: {diagnostic:?}", diagnostic.commit());
        }
        Ok(())
    })
}

fn run_show(pile_path: PathBuf, team_root_text: String, grant_text: String) -> Result<()> {
    let team_root = parse_team_root(&team_root_text)?;
    let grant_id = parse_grant_id(&grant_text)?;
    with_pile(&pile_path, |pile| {
        let resolution = authority::resolve_authority(pile, team_root)
            .map_err(|error| anyhow!("resolve team authority: {error}"))?;
        if resolution.grant(grant_id).is_none() {
            if let Some(diagnostic) = resolution
                .diagnostics()
                .iter()
                .find(|diagnostic| diagnostic.commit() == grant_id)
            {
                bail!("grant {grant_id:X} is inert: {diagnostic:?}");
            }
            bail!("grant {grant_id:X} is not present in this team authority collection");
        }
        let proof = resolution
            .proof(grant_id)
            .ok_or_else(|| anyhow!("accepted grant {grant_id:X} has no ancestry proof"))?;
        println!("team root: {}", hex::encode(team_root.to_bytes()));
        println!("ancestry: {} step(s), root to leaf", proof.steps().len());
        for (level, step) in proof.steps().iter().enumerate() {
            let accepted = resolution
                .grant(step.commit().id())
                .expect("proof steps came from this resolution");
            println!();
            println!("level {level}:");
            print_grant(accepted, "  ");
        }
        Ok(())
    })
}
