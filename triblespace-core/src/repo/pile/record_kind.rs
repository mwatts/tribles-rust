//! Self-describing pile record kinds.
//!
//! A record's kind used to be a bare 16-byte minted id. An id alone is only
//! *recognisable*: a reader either already holds the code that minted it or
//! learns nothing at all. The V2 envelope widens the field to 32 bytes and
//! fills it with a blob handle naming a description of the record's own
//! layout, so a reader meeting an unfamiliar record can **resolve** what it is.
//! This is the same move the collection layer already made at the descriptor
//! level: stop naming things by opaque ids that resolve to nothing.
//!
//! Each description is an ordinary [`SimpleArchive`] blob holding one entity,
//! rooted at the 16-byte marker the kind was already minted under, carrying a
//! name, a prose statement of the exact byte layout, and the
//! [`KIND_PILE_RECORD`] tag. Rooting at the historical id means the widening
//! does not renumber anything: the same kind is still named by the same id,
//! the description is simply reachable now.
//!
//! The handles below are pinned so the on-disk format is stated in the source
//! rather than implied by a doc string. `record_kind_handles_match_their_descriptions`
//! recomputes every one of them, so editing a description fails the test with
//! the new value instead of silently reframing the pile.

use std::collections::BTreeSet;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, IntoBlob};
use crate::id::{ExclusiveId, Id};
use crate::id_hex;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, RawInline};
use crate::metadata::{self, MetaDescribe};
use crate::prelude::entity;
use crate::repo::BlobStore;
use crate::trible::{Fragment, TribleSet};

/// Tag identifying an entity that describes one pile record kind.
///
/// Minted with `trible genid` on 2026-08-20.
pub const KIND_PILE_RECORD: Id = id_hex!("29D9F7F6B5062623F65D63DBF4F633B3");

/// Canonical 32-byte name of one record kind: the handle of its description.
pub type RecordKind = Inline<Handle<SimpleArchive>>;

/// Archive one description fragment and take its content identity.
///
/// Only the fragment's facts are archived, exactly as a collection descriptor
/// does. The name and layout strings live in the fragment's own blob store and
/// are referenced from those facts by handle; [`description_blobs`] hands both
/// halves out together so a pile can make the whole description resolvable.
pub fn describe_blob(fragment: &Fragment) -> Blob<SimpleArchive> {
    <TribleSet as IntoBlob<SimpleArchive>>::to_blob(fragment.facts().clone())
}

macro_rules! record_kinds {
    ($(
        $(#[$meta:meta])*
        $ty:ident = $id:ident $id_hex:literal, $handle:ident $handle_hex:literal,
        $name:literal, $layout:literal;
    )*) => {
        $(
            $(#[$meta])*
            ///
            /// The 16-byte id this description is rooted at.
            pub const $id: Id = id_hex!($id_hex);

            $(#[$meta])*
            ///
            /// The 32-byte record kind written into bytes `32..64` of the
            /// envelope: the handle of this kind's description archive.
            pub const $handle: RawInline = hex_literal::hex!($handle_hex);

            $(#[$meta])*
            pub struct $ty;

            impl MetaDescribe for $ty {
                fn describe() -> Fragment {
                    let id: Id = $id;
                    entity! {
                        ExclusiveId::force_ref(&id) @
                            metadata::name: $name,
                            metadata::description: $layout,
                            metadata::tag: KIND_PILE_RECORD,
                    }
                }
            }
        )*

        /// Every record kind this reader knows, as `(handle, description)`.
        ///
        /// The order is the declaration order above and is stable.
        pub fn described_kinds() -> Vec<(RawInline, Fragment)> {
            vec![$(($handle, <$ty as MetaDescribe>::describe())),*]
        }
    };
}

record_kinds! {
    /// A blob record: fixed header followed by the payload.
    BlobRecordV1 = KIND_ID_BLOB "9C33EEB525065A62EAEC4BE43DCC355A",
        KIND_BLOB "01148F301FE56E346D16596A8480532E8B4420C4EFD00C8DFF437D0DF9810ED0",
        "pile-blob-v1",
        "A content-addressed blob. Envelope bytes 64..72 hold the insertion timestamp in Unix milliseconds as an unsigned little-endian 64-bit integer, 72..80 the exact unpadded payload length in bytes in the same encoding, 80..96 zeros, 96..128 the BLAKE3 digest of the payload, and 128..256 zeros. The payload begins at record_start + 256 and is post-padded with zeros to the declared block span. Padding is not covered by the digest.";

    /// A pin (branch) head assignment.
    PinHeadRecordV1 = KIND_ID_PIN_HEAD "AC363D04AFE1AF17B39581B1E23021D7",
        KIND_PIN_HEAD "2BC0B9FE0EFDB0BC53654E17BB9D06E01259F36AF93EEE54AD5D557B12DF706D",
        "pile-pin-head-v1",
        "A last-writer-wins assignment of one pin (branch) identifier to the handle of its metadata blob. Envelope bytes 64..80 hold the 16-byte pin identifier, 80..96 zeros, 96..128 the BLAKE3 handle of the head SimpleArchive, and 128..256 zeros. The record spans exactly one 256-byte block and has no payload. The pile does not require the referenced blob to be resident.";

    /// A pin (branch) tombstone.
    PinTombstoneRecordV1 = KIND_ID_PIN_TOMBSTONE "D0CBA0C8EAAB4C0C73121C3205671E4F",
        KIND_PIN_TOMBSTONE "8D9F27E76D3620EEC29B781F841E9EF77F2607B40DC702FE3DAED007E9228CA5",
        "pile-pin-tombstone-v1",
        "Retraction of a pin (branch) head assignment, resolved last-writer-wins against pile-pin-head-v1 records for the same identifier. Envelope bytes 64..80 hold the 16-byte pin identifier and 80..256 are zeros. The record spans exactly one 256-byte block and has no payload.";

    /// A blob want assertion, in the historical weak-pin encoding.
    BlobWantAssertRecordV1 = KIND_ID_BLOB_WANT_ASSERT "8F3EEFEDECD491F63F6EAAA5FD6F3D5E",
        KIND_BLOB_WANT_ASSERT "EC1C024C04AF08243DB3AE318C93FA500355C74395C0F553CFFC0AF0A4BA0346",
        "pile-blob-want-assert-v1",
        "Durable local demand for one blob, keyed by its handle and resolved last-writer-wins against pile-blob-want-retract-v1. Envelope bytes 64..96 hold the wanted BLAKE3 blob handle and 96..256 are zeros. Blob wants deliberately keep this kind rather than moving to pile-want-assert-v2, so a reader that skips the typed operation wants still observes the complete blob demand history.";

    /// A blob want retraction, in the historical weak-unpin encoding.
    BlobWantRetractRecordV1 = KIND_ID_BLOB_WANT_RETRACT "2D76662DFF0187EC36A8C90B12BB8B0D",
        KIND_BLOB_WANT_RETRACT "ACCB531FC7489357C40FCEF0DDE8BD9088F2AC1924A652EA211ADD5C30B95B46",
        "pile-blob-want-retract-v1",
        "Retraction of durable local demand for one blob. Envelope bytes 64..96 hold the BLAKE3 blob handle and 96..256 are zeros.";

    /// A typed operation-want assertion.
    WantAssertRecordV2 = KIND_ID_WANT_ASSERT "9A06797600FA90B8A8259B0ED029EC21",
        KIND_WANT_ASSERT "65EE9E4279FFE01D263E75A8E2DF6289B6DE403CB4468098A0EAB925F81C28ED",
        "pile-want-assert-v2",
        "Durable local demand for one reproducible collection operation, keyed by a canonical 97-byte request. Envelope byte 64 holds the versioned request tag, 65..96 are zeros, 96..128 hold field A, 128..160 field B, 160..192 field C, and 192..256 are zeros. Tag 2 is a merge request (A the collection descriptor handle, B and C the two input digests in lexicographic order); tag 3 is a derive request (A the source descriptor handle, B the target descriptor handle, C the input digest). Tag 1, a blob request, is not valid here: blob wants use pile-blob-want-assert-v1.";

    /// A typed operation-want retraction.
    WantRetractRecordV2 = KIND_ID_WANT_RETRACT "2D957A780A52E474F58A06D44D6FE46C",
        KIND_WANT_RETRACT "A57C866A83A90635090A947D92464B19D9F898C0C961AB7A91C79A979F9F1483",
        "pile-want-retract-v2",
        "Retraction of durable local demand for one reproducible collection operation. The request layout is identical to pile-want-assert-v2, and the two resolve last-writer-wins per exact request key.";

    /// A signed collection commit.
    CollectionCommitRecordV4 = KIND_ID_COLLECTION_COMMIT "CBF2CF97D52A3486E16C12D70D397C66",
        KIND_COLLECTION_COMMIT "A1322BB3F5214287C314D42AFCC1A97CB264FACD9A22B4938838BE78DB31AA59",
        "pile-collection-commit-v4",
        "A signed COMMIT(collection, data, metadata) assertion. Envelope bytes 64..96 hold the collection descriptor handle, 96..128 the data digest, 128..160 the metadata archive handle, 160..192 the author Ed25519 public key, 192..224 the signature R component, and 224..256 the signature S component. This is the tightest record the pile writes: it fills the block exactly and reserves nothing. The signature covers a domain-separated transcript, not these bytes, so a commit survives reframing unchanged.";

    /// An unsigned merge equation.
    CollectionMergeRecordV4 = KIND_ID_COLLECTION_MERGE "9F5D028D4C423620D6957A5F726FA727",
        KIND_COLLECTION_MERGE "0CEE320DE0BDA40A6A6F52221C5E4E4D2CE3B165B69C858673FD13D98F655379",
        "pile-collection-merge-v4",
        "An unsigned MERGE equation asserting that two element digests join to a third under the collection's recipe. Envelope bytes 64..96 hold the collection descriptor handle, 96..128 the lexicographically lower input digest, 128..160 the higher input digest, 160..192 the result digest, and 192..256 are zeros. Storing the inputs in order means operand order cannot produce a second representation of the same commutative equation.";

    /// An unsigned derive equation.
    CollectionDeriveRecordV5 = KIND_ID_COLLECTION_DERIVE "ED6B46F7286D4556B076C17B79FD8315",
        KIND_COLLECTION_DERIVE "7ACE1ED10F3EBC632627058CC461DC1CC171CD2E56C52E5DCE60EA4C8DC23C36",
        "pile-collection-derive-v5",
        "An unsigned DERIVE equation asserting that an input state of a derived collection's source maps to an output state of that collection. Envelope bytes 64..96 hold the target collection's descriptor handle, 96..128 the input digest, 128..160 the output digest, and 160..256 are zeros. The source is not named here because the target's descriptor already names it, and naming it twice only creates a way for the two to disagree.";

}

/// Every blob needed to resolve every known record kind.
///
/// This is the description archives themselves plus the name and layout
/// strings they reference by handle. Publishing all of them into a pile makes
/// the pile answer "what is this record?" without any external lookup.
pub fn description_blobs() -> Vec<Blob<UnknownBlob>> {
    // Deduplicated by handle: the descriptions share the metafacts of the
    // attributes they use, so the raw concatenation repeats most of itself.
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let mut push = |blob: Blob<UnknownBlob>| {
        if seen.insert(blob.get_handle().raw) {
            out.push(blob);
        }
    };
    for (_, fragment) in described_kinds() {
        push(describe_blob(&fragment).transmute());
        let reader = fragment
            .blobs()
            .clone()
            .reader()
            .expect("MemoryBlobStore reader is infallible");
        for (_, blob) in reader.iter() {
            push(blob);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pinned record kind is exactly the handle of its own description.
    ///
    /// This is what makes the 32-byte kind resolvable rather than merely
    /// recognisable, and it is why editing a description is a format change:
    /// the failure message below carries the new value to pin.
    #[test]
    fn record_kind_handles_match_their_descriptions() {
        for (pinned, fragment) in described_kinds() {
            let computed = describe_blob(&fragment).get_handle().raw;
            assert_eq!(
                pinned,
                computed,
                "record kind description changed; pin {}",
                computed
                    .iter()
                    .map(|b| format!("{b:02X}"))
                    .collect::<String>()
            );
        }
    }
}
