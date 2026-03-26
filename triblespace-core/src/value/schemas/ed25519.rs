use ed25519::ComponentBytes;
use ed25519::Signature;
use ed25519_dalek::SignatureError;
/// Re-export of the Ed25519 verifying (public) key type from [`ed25519_dalek`].
pub use ed25519_dalek::VerifyingKey;

use crate::id::ExclusiveId;
use crate::id::Id;
use crate::id_hex;
use crate::macros::entity;
use crate::metadata;
use crate::metadata::{ConstDescribe, ConstId};
use crate::repo::BlobStore;
use crate::trible::Fragment;
use crate::value::schemas::hash::Blake3;
use crate::value::ToValue;
use crate::value::TryFromValue;
use crate::value::Value;
use crate::value::ValueSchema;
use std::convert::Infallible;

/// A value schema for the R component of an Ed25519 signature.
pub struct ED25519RComponent;

impl ConstId for ED25519RComponent {
    const ID: Id = id_hex!("995A86FFC83DB95ECEAA17E226208897");
}

/// A value schema for the S component of an Ed25519 signature.
pub struct ED25519SComponent;

impl ConstId for ED25519SComponent {
    const ID: Id = id_hex!("10D35B0B628E9E409C549D8EC1FB3598");
}

/// A value schema for an Ed25519 public key.
pub struct ED25519PublicKey;

impl ConstId for ED25519PublicKey {
    const ID: Id = id_hex!("69A872254E01B4C1ED36E08E40445E93");
}

impl ConstDescribe for ED25519RComponent {
    fn describe<B>(blobs: &mut B) -> Result<Fragment, B::PutError>
    where
        B: BlobStore<Blake3>,
    {
        let id = Self::ID;
        let description = blobs.put(
            "Ed25519 signature R component stored as a 32-byte field. This is one half of the standard 64-byte Ed25519 signature.\n\nUse when you store signatures as structured values or need to index the components separately. Pair with the S component to reconstruct or verify the full signature.\n\nIf you prefer storing the signature as a single binary blob, use a blob schema (for example LongString with base64 or a custom blob schema).",
        )?;
        let tribles = entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: blobs.put("ed25519:r")?,
                metadata::description: description,
                metadata::tag: metadata::KIND_VALUE_SCHEMA,
        };

        #[cfg(feature = "wasm")]
        let tribles = {
            let mut tribles = tribles;
            tribles += entity! { ExclusiveId::force_ref(&id) @
                metadata::value_formatter: blobs.put(wasm_formatter::ED25519_R_WASM)?,
            };
            tribles
        };
        Ok(tribles)
    }
}
impl ValueSchema for ED25519RComponent {
    type ValidationError = Infallible;
}
impl ConstDescribe for ED25519SComponent {
    fn describe<B>(blobs: &mut B) -> Result<Fragment, B::PutError>
    where
        B: BlobStore<Blake3>,
    {
        let id = Self::ID;
        let description = blobs.put(
            "Ed25519 signature S component stored as a 32-byte field. This is the second half of the standard Ed25519 signature.\n\nUse when storing or querying signatures in a structured form. Pair with the R component to reconstruct or verify the full signature.\n\nAs with the R component, treat this as public data; private signing keys should be stored separately and securely.",
        )?;
        let tribles = entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: blobs.put("ed25519:s")?,
                metadata::description: description,
                metadata::tag: metadata::KIND_VALUE_SCHEMA,
        };

        #[cfg(feature = "wasm")]
        let tribles = {
            let mut tribles = tribles;
            tribles += entity! { ExclusiveId::force_ref(&id) @
                metadata::value_formatter: blobs.put(wasm_formatter::ED25519_S_WASM)?,
            };
            tribles
        };
        Ok(tribles)
    }
}
impl ValueSchema for ED25519SComponent {
    type ValidationError = Infallible;
}
impl ConstDescribe for ED25519PublicKey {
    fn describe<B>(blobs: &mut B) -> Result<Fragment, B::PutError>
    where
        B: BlobStore<Blake3>,
    {
        let id = Self::ID;
        let description = blobs.put(
            "Ed25519 public key stored as a 32-byte field. Public keys verify signatures and identify signing identities.\n\nUse for signer registries, verification records, or key references associated with signatures. Private keys are not represented by a built-in schema and should be handled separately.\n\nEd25519 is widely supported and deterministic; if you need another scheme, define a custom schema with its own metadata.",
        )?;
        let tribles = entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: blobs.put("ed25519:pubkey")?,
                metadata::description: description,
                metadata::tag: metadata::KIND_VALUE_SCHEMA,
        };

        #[cfg(feature = "wasm")]
        let tribles = {
            let mut tribles = tribles;
            tribles += entity! { ExclusiveId::force_ref(&id) @
                metadata::value_formatter: blobs.put(wasm_formatter::ED25519_PUBKEY_WASM)?,
            };
            tribles
        };
        Ok(tribles)
    }
}
impl ValueSchema for ED25519PublicKey {
    type ValidationError = Infallible;
}

#[cfg(feature = "wasm")]
mod wasm_formatter {
    use core::fmt::Write;

    use triblespace_core_macros::value_formatter;

    #[value_formatter(const_wasm = ED25519_R_WASM)]
    pub(crate) fn ed25519_r(raw: &[u8; 32], out: &mut impl Write) -> Result<(), u32> {
        out.write_str("ed25519:r:").map_err(|_| 1u32)?;
        const TABLE: &[u8; 16] = b"0123456789ABCDEF";
        for &byte in raw {
            let hi = (byte >> 4) as usize;
            let lo = (byte & 0x0F) as usize;
            out.write_char(TABLE[hi] as char).map_err(|_| 1u32)?;
            out.write_char(TABLE[lo] as char).map_err(|_| 1u32)?;
        }
        Ok(())
    }

    #[value_formatter(const_wasm = ED25519_S_WASM)]
    pub(crate) fn ed25519_s(raw: &[u8; 32], out: &mut impl Write) -> Result<(), u32> {
        out.write_str("ed25519:s:").map_err(|_| 1u32)?;
        const TABLE: &[u8; 16] = b"0123456789ABCDEF";
        for &byte in raw {
            let hi = (byte >> 4) as usize;
            let lo = (byte & 0x0F) as usize;
            out.write_char(TABLE[hi] as char).map_err(|_| 1u32)?;
            out.write_char(TABLE[lo] as char).map_err(|_| 1u32)?;
        }
        Ok(())
    }

    #[value_formatter(const_wasm = ED25519_PUBKEY_WASM)]
    pub(crate) fn ed25519_pubkey(raw: &[u8; 32], out: &mut impl Write) -> Result<(), u32> {
        out.write_str("ed25519:pubkey:").map_err(|_| 1u32)?;
        const TABLE: &[u8; 16] = b"0123456789ABCDEF";
        for &byte in raw {
            let hi = (byte >> 4) as usize;
            let lo = (byte & 0x0F) as usize;
            out.write_char(TABLE[hi] as char).map_err(|_| 1u32)?;
            out.write_char(TABLE[lo] as char).map_err(|_| 1u32)?;
        }
        Ok(())
    }
}

impl ED25519RComponent {
    /// Extracts the R component from a full Ed25519 signature.
    pub fn from_signature(s: Signature) -> Value<ED25519RComponent> {
        Value::new(*s.r_bytes())
    }
}

impl ED25519SComponent {
    /// Extracts the S component from a full Ed25519 signature.
    pub fn from_signature(s: Signature) -> Value<ED25519SComponent> {
        Value::new(*s.s_bytes())
    }
}

impl ToValue<ED25519RComponent> for Signature {
    fn to_value(self) -> Value<ED25519RComponent> {
        ED25519RComponent::from_signature(self)
    }
}

impl ToValue<ED25519SComponent> for Signature {
    fn to_value(self) -> Value<ED25519SComponent> {
        ED25519SComponent::from_signature(self)
    }
}

impl ToValue<ED25519RComponent> for ComponentBytes {
    fn to_value(self) -> Value<ED25519RComponent> {
        Value::new(self)
    }
}

impl TryFromValue<'_, ED25519RComponent> for ComponentBytes {
    type Error = Infallible;
    fn try_from_value(v: &Value<ED25519RComponent>) -> Result<Self, Infallible> {
        Ok(v.raw)
    }
}

impl ToValue<ED25519SComponent> for ComponentBytes {
    fn to_value(self) -> Value<ED25519SComponent> {
        Value::new(self)
    }
}

impl TryFromValue<'_, ED25519SComponent> for ComponentBytes {
    type Error = Infallible;
    fn try_from_value(v: &Value<ED25519SComponent>) -> Result<Self, Infallible> {
        Ok(v.raw)
    }
}

impl ToValue<ED25519PublicKey> for VerifyingKey {
    fn to_value(self) -> Value<ED25519PublicKey> {
        Value::new(self.to_bytes())
    }
}

impl TryFromValue<'_, ED25519PublicKey> for VerifyingKey {
    type Error = SignatureError;

    fn try_from_value(v: &Value<ED25519PublicKey>) -> Result<Self, Self::Error> {
        VerifyingKey::from_bytes(&v.raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{ToValue, TryFromValue};
    use ed25519::Signature;
    use ed25519_dalek::SigningKey;

    fn test_signature() -> Signature {
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        use ed25519_dalek::Signer;
        let sig = signing_key.sign(b"test message");
        Signature::from_bytes(&sig.to_bytes())
    }

    fn test_verifying_key() -> VerifyingKey {
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        signing_key.verifying_key()
    }

    #[test]
    fn r_component_from_signature_roundtrip() {
        let sig = test_signature();
        let value: Value<ED25519RComponent> = sig.to_value();
        let bytes: ComponentBytes = value.from_value();
        assert_eq!(&bytes, sig.r_bytes());
    }

    #[test]
    fn s_component_from_signature_roundtrip() {
        let sig = test_signature();
        let value: Value<ED25519SComponent> = sig.to_value();
        let bytes: ComponentBytes = value.from_value();
        assert_eq!(&bytes, sig.s_bytes());
    }

    #[test]
    fn r_component_bytes_roundtrip() {
        let input: ComponentBytes = [0xAB; 32];
        let value: Value<ED25519RComponent> = input.to_value();
        let output: ComponentBytes = value.from_value();
        assert_eq!(input, output);
    }

    #[test]
    fn s_component_bytes_roundtrip() {
        let input: ComponentBytes = [0xCD; 32];
        let value: Value<ED25519SComponent> = input.to_value();
        let output: ComponentBytes = value.from_value();
        assert_eq!(input, output);
    }

    #[test]
    fn r_component_zero_bytes() {
        let input: ComponentBytes = [0u8; 32];
        let value: Value<ED25519RComponent> = input.to_value();
        let output: ComponentBytes = value.from_value();
        assert_eq!(input, output);
    }

    #[test]
    fn s_component_max_bytes() {
        let input: ComponentBytes = [0xFF; 32];
        let value: Value<ED25519SComponent> = input.to_value();
        let output: ComponentBytes = value.from_value();
        assert_eq!(input, output);
    }

    #[test]
    fn verifying_key_roundtrip() {
        let key = test_verifying_key();
        let value: Value<ED25519PublicKey> = key.to_value();
        let recovered = VerifyingKey::try_from_value(&value).expect("valid key");
        assert_eq!(key, recovered);
    }

    #[test]
    fn verifying_key_invalid_bytes() {
        // A y-coordinate of 2 is not on the Ed25519 curve, so decompression
        // should fail.
        let mut raw = [0u8; 32];
        raw[0] = 2;
        let value: Value<ED25519PublicKey> = Value::new(raw);
        assert!(VerifyingKey::try_from_value(&value).is_err());
    }

    #[test]
    fn signature_r_and_s_reconstruct() {
        let sig = test_signature();
        let r_val: Value<ED25519RComponent> = sig.to_value();
        let s_val: Value<ED25519SComponent> = sig.to_value();
        let r_bytes: ComponentBytes = r_val.from_value();
        let s_bytes: ComponentBytes = s_val.from_value();
        let mut combined = [0u8; 64];
        combined[..32].copy_from_slice(&r_bytes);
        combined[32..].copy_from_slice(&s_bytes);
        let reconstructed = Signature::from_bytes(&combined);
        assert_eq!(sig, reconstructed);
    }
}
