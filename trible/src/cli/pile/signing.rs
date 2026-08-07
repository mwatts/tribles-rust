use anyhow::Result;
use clap::Parser;
use ed25519_dalek::SigningKey;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use triblespace_core::signing_key_file;

#[derive(Parser)]
pub enum Command {
    /// Explicitly initialize the durable signing identity used beside a pile.
    Init {
        /// Pile whose lexical parent supplies the default self.key path.
        pile: PathBuf,
        /// Explicit key path instead of TRIBLESPACE_KEY or self.key beside the pile.
        #[arg(long)]
        key: Option<PathBuf>,
    },
}

pub fn run(command: Command) -> Result<()> {
    match command {
        Command::Init { pile, key } => {
            let path = signing_key_file::resolve_path(key.as_deref(), &pile);
            let signer = signing_key_file::init(&path)?;
            println!("path: {}", path.display());
            println!(
                "public-key: {}",
                hex::encode_upper(signer.verifying_key().to_bytes())
            );
            Ok(())
        }
    }
}

/// Load a signing key from an explicit path, the TRIBLES_SIGNING_KEY env var,
/// or generate an ephemeral key.  Used by commands that don't have a pile
/// (e.g. genid) or where persistence doesn't matter.
pub(super) fn load_signing_key(path_opt: &Option<PathBuf>) -> Result<SigningKey, anyhow::Error> {
    let key_path_opt: Option<PathBuf> = if let Some(p) = path_opt {
        Some(p.clone())
    } else if let Ok(s) = env::var("TRIBLES_SIGNING_KEY") {
        Some(PathBuf::from(s))
    } else {
        None
    };

    if let Some(p) = key_path_opt {
        return load_key_from_file(&p);
    }

    generate_ephemeral_key()
}

fn load_key_from_file(p: &Path) -> Result<SigningKey, anyhow::Error> {
    let content = fs::read_to_string(p)
        .map_err(|e| anyhow::anyhow!("failed to read signing key {}: {e}", p.display()))?;
    let hexstr = content.trim();
    if hexstr.len() != 64 || !hexstr.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!(
            "signing key file {} does not contain valid 64-char hex",
            p.display()
        );
    }
    let bytes =
        hex::decode(hexstr).map_err(|e| anyhow::anyhow!("invalid hex in signing key file: {e}"))?;
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(SigningKey::from_bytes(&arr))
}

fn generate_ephemeral_key() -> Result<SigningKey, anyhow::Error> {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed)
        .map_err(|e| anyhow::anyhow!("failed to generate signing key: {e}"))?;
    Ok(SigningKey::from_bytes(&seed))
}
