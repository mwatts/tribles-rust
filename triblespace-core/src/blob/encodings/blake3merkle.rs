//! Blob schema for canonical addressable BLAKE3 PATCH Merkle nodes.

use crate::blob::BlobEncoding;
use crate::id::{id_hex, ExclusiveId, Id};
use crate::macros::entity;
use crate::metadata::{self, MetaDescribe};
use crate::trible::Fragment;

/// Blob encoding for every canonical version-3 BLAKE3 PATCH Merkle node.
///
/// The encoded header carries the key width, so this single schema names leaf
/// and branch nodes for every `PATCH<KEY_LEN, _, _, Blake3Merkle>`. Consumers
/// select and validate the expected width with
/// [`Blake3MerkleNodeBlob`](crate::patch::Blake3MerkleNodeBlob).
pub struct Blake3MerkleNode;

impl BlobEncoding for Blake3MerkleNode {}

impl MetaDescribe for Blake3MerkleNode {
    fn describe() -> Fragment {
        // Minted with `trible genid` on 2026-08-27.
        let id: Id = id_hex!("4FEB14E2C9BE28B792CBA42C8F936BEA");
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "blake3-merkle-node-v3",
                metadata::description: "Canonical 32-byte-aligned version-3 BLAKE3 PATCH Merkle node. The blob's ordinary BLAKE3 content address is the node digest; aligned child digest fields form its conservative storage DAG.",
                metadata::tag: metadata::KIND_BLOB_ENCODING,
        }
    }
}
