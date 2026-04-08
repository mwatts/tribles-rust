//! Node identity management.
//!
//! Each pile gets a persistent ed25519 identity stored in a companion
//! `.key` file. The same key signs commits and identifies the node
//! on the network.

use std::path::Path;
use std::fs;
use anyhow::{Result, anyhow};
use ed25519_dalek::SigningKey;
use iroh_base::SecretKey;

/// Load or create a persistent signing key for a pile.
///
/// Resolution: explicit path → TRIBLES_SIGNING_KEY env → `<pile>.key` auto-create.
pub fn load_or_create_pile_key(
    explicit_path: &Option<std::path::PathBuf>,
    pile_path: &Path,
) -> Result<SigningKey> {
    if let Some(p) = explicit_path {
        return load_key_from_file(p);
    }
    if let Ok(s) = std::env::var("TRIBLES_SIGNING_KEY") {
        return load_key_from_file(Path::new(&s));
    }
    let key_path = pile_path.with_extension("pile.key");
    if key_path.exists() {
        return load_key_from_file(&key_path);
    }
    let key = generate_key()?;
    let hex_str = hex::encode(key.to_bytes());
    fs::write(&key_path, &hex_str)
        .map_err(|e| anyhow!("write key to {}: {e}", key_path.display()))?;
    eprintln!("generated new node key: {}", key_path.display());
    Ok(key)
}

/// Convert an ed25519 signing key to an iroh secret key.
pub fn iroh_secret(key: &SigningKey) -> SecretKey {
    SecretKey::from(key.to_bytes())
}

fn load_key_from_file(p: &Path) -> Result<SigningKey> {
    let content = fs::read_to_string(p)
        .map_err(|e| anyhow!("read key {}: {e}", p.display()))?;
    let hexstr = content.trim();
    if hexstr.len() != 64 || !hexstr.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!("key file {} is not valid 64-char hex", p.display());
    }
    let bytes = hex::decode(hexstr)?;
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(SigningKey::from_bytes(&arr))
}

fn generate_key() -> Result<SigningKey> {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed)
        .map_err(|e| anyhow!("generate key: {e}"))?;
    Ok(SigningKey::from_bytes(&seed))
}
