//! Shared structural attributes for Ed25519 attestations.
//!
//! These fields identify a signer and carry the two components of an Ed25519
//! signature. They do not define what bytes are signed, establish that the
//! signature is valid, or authorize the signer. Each record format owns those
//! transcript, verification, and admission rules.

use crate::inline::encodings::ed25519;
use crate::prelude::attributes;

attributes! {
    /// The author of an attestation, identified by their Ed25519 public key.
    "ADB4FFAD247C886848161297EFF5A05B" unsafe as pub signed_by: ed25519::ED25519PublicKey;
    /// The `r` component of an Ed25519 signature.
    "9DF34F84959928F93A3C40AEB6E9E499" unsafe as pub signature_r: ed25519::ED25519RComponent;
    /// The `s` component of an Ed25519 signature.
    "1ACE03BF70242B289FDF00E4327C3BC6" unsafe as pub signature_s: ed25519::ED25519SComponent;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_attribute_ids_are_preserved() {
        assert_eq!(
            signed_by.id(),
            crate::id_hex!("ADB4FFAD247C886848161297EFF5A05B")
        );
        assert_eq!(
            signature_r.id(),
            crate::id_hex!("9DF34F84959928F93A3C40AEB6E9E499")
        );
        assert_eq!(
            signature_s.id(),
            crate::id_hex!("1ACE03BF70242B289FDF00E4327C3BC6")
        );
    }
}
