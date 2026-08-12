//! Collection-native, read-only reconciliation over an authenticated peer.
//!
//! The wire surface deliberately carries evidence rather than admission
//! decisions. A server enumerates only strictly verified
//! [`CollectionGossip`] / [`CollectionCommit`] pairs for one exact
//! descriptor handle. A client verifies the pair again, then may fetch the
//! complete content-addressed closure rooted at the descriptor, data, and
//! metadata handles. Nothing in this module mutates a destination store.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use anybytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use triblespace_core::blob::Blob;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::collection::{
    COLLECTION_COMMIT_ARCHIVE_LEN, COLLECTION_GOSSIP_BYTES_LEN, CollectionCommit, CollectionGossip,
    CollectionId, CollectionRecord, CommitVerificationError, GossipVerificationError,
    RecordDecodeError,
};

use crate::protocol::{
    COLLECTION_EVIDENCE_REJECTED, OP_COLLECTION_EVIDENCE, RawHash, op_children, op_get_blob,
    recv_u32_be, send_hash, send_u8,
};
use crate::transport::Conn;

/// Exact byte length of one grant-backed commit evidence item.
///
/// The layout is the canonical 128-byte gossip witness followed by the
/// canonical 448-byte [`CollectionCommit`] `SimpleArchive`.
pub const COLLECTION_COMMIT_EVIDENCE_LEN: usize =
    COLLECTION_GOSSIP_BYTES_LEN + COLLECTION_COMMIT_ARCHIVE_LEN as usize;

/// Defensive upper bound on the number of evidence items accepted from one
/// response. The wire count remains a `u32`; this bound prevents a malicious
/// server from forcing an unbounded allocation before any evidence is read.
pub const MAX_COLLECTION_EVIDENCE_ITEMS: u32 = 1 << 20;

/// One author-signed collection commit accompanied by that same author's
/// permanent redistribution grant for the exact same collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollectionCommitEvidence {
    grant: CollectionGossip,
    commit: CollectionCommit,
}

impl CollectionCommitEvidence {
    /// Construct evidence only after strict signature and correspondence
    /// validation.
    pub fn new(
        grant: CollectionGossip,
        commit: CollectionCommit,
    ) -> Result<Self, CollectionCommitEvidenceError> {
        grant
            .verify_strict()
            .map_err(CollectionCommitEvidenceError::InvalidGrantSignature)?;
        commit
            .verify_strict()
            .map_err(CollectionCommitEvidenceError::InvalidCommitSignature)?;
        validate_correspondence(&grant, &commit)?;
        Ok(Self { grant, commit })
    }

    /// Decode one exact fixed-width evidence item and strictly verify both
    /// signatures plus author/collection correspondence.
    pub fn decode(bytes: &[u8]) -> Result<Self, CollectionCommitEvidenceError> {
        if bytes.len() != COLLECTION_COMMIT_EVIDENCE_LEN {
            return Err(CollectionCommitEvidenceError::WrongLength {
                expected: COLLECTION_COMMIT_EVIDENCE_LEN,
                actual: bytes.len(),
            });
        }

        let grant = CollectionGossip::from_bytes(
            bytes[..COLLECTION_GOSSIP_BYTES_LEN]
                .try_into()
                .expect("checked evidence length"),
        );
        let commit = CollectionCommit::decode(&Blob::<SimpleArchive>::new(Bytes::from_source(
            bytes[COLLECTION_GOSSIP_BYTES_LEN..].to_vec(),
        )))
        .map_err(CollectionCommitEvidenceError::InvalidCommitRecord)?;
        Self::new(grant, commit)
    }

    /// Encode the already-verified pair into its fixed canonical layout.
    pub fn encode(&self) -> [u8; COLLECTION_COMMIT_EVIDENCE_LEN] {
        let mut bytes = [0u8; COLLECTION_COMMIT_EVIDENCE_LEN];
        bytes[..COLLECTION_GOSSIP_BYTES_LEN].copy_from_slice(&self.grant.to_bytes());
        let commit = self.commit.to_blob();
        debug_assert_eq!(commit.bytes.len(), COLLECTION_COMMIT_ARCHIVE_LEN as usize);
        bytes[COLLECTION_GOSSIP_BYTES_LEN..].copy_from_slice(&commit.bytes);
        bytes
    }

    /// Exact author-signed redistribution witness.
    pub fn grant(&self) -> CollectionGossip {
        self.grant
    }

    /// Exact canonical signed membership assertion.
    pub fn commit(&self) -> CollectionCommit {
        self.commit
    }

    /// Exact blob roots named by this commit: descriptor, data, and metadata.
    pub fn roots(&self) -> [RawHash; 3] {
        [
            self.commit.collection().raw,
            self.commit.data().raw,
            self.commit.metadata().raw,
        ]
    }
}

fn validate_correspondence(
    grant: &CollectionGossip,
    commit: &CollectionCommit,
) -> Result<(), CollectionCommitEvidenceError> {
    if grant.public_key() != commit.public_key() {
        return Err(CollectionCommitEvidenceError::AuthorMismatch {
            grant: grant.public_key().raw,
            commit: commit.public_key().raw,
        });
    }
    if grant.collection() != commit.collection() {
        return Err(CollectionCommitEvidenceError::CollectionMismatch {
            grant: grant.collection().raw,
            commit: commit.collection().raw,
        });
    }
    Ok(())
}

/// Strict evidence decoding, signature, or correspondence failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionCommitEvidenceError {
    /// The byte slice was not exactly one fixed-width evidence item.
    WrongLength { expected: usize, actual: usize },
    /// The grant failed strict Ed25519 verification.
    InvalidGrantSignature(GossipVerificationError),
    /// The commit archive was not structurally canonical.
    InvalidCommitRecord(RecordDecodeError),
    /// The commit failed strict Ed25519 verification.
    InvalidCommitSignature(CommitVerificationError),
    /// Grant and commit name different authors.
    AuthorMismatch { grant: [u8; 32], commit: [u8; 32] },
    /// Grant and commit name different collections.
    CollectionMismatch { grant: [u8; 32], commit: [u8; 32] },
}

impl fmt::Display for CollectionCommitEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { expected, actual } => {
                write!(
                    formatter,
                    "collection evidence is {actual} bytes, expected {expected}"
                )
            }
            Self::InvalidGrantSignature(error) => {
                write!(
                    formatter,
                    "collection gossip grant failed verification: {error}"
                )
            }
            Self::InvalidCommitRecord(error) => {
                write!(formatter, "collection commit is not canonical: {error}")
            }
            Self::InvalidCommitSignature(error) => {
                write!(formatter, "collection commit failed verification: {error}")
            }
            Self::AuthorMismatch { .. } => {
                write!(formatter, "collection grant and commit authors differ")
            }
            Self::CollectionMismatch { .. } => {
                write!(formatter, "collection grant and commit collections differ")
            }
        }
    }
}

impl Error for CollectionCommitEvidenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidGrantSignature(error) => Some(error),
            Self::InvalidCommitRecord(error) => Some(error),
            Self::InvalidCommitSignature(error) => Some(error),
            Self::WrongLength { .. }
            | Self::AuthorMismatch { .. }
            | Self::CollectionMismatch { .. } => None,
        }
    }
}

/// Deterministically pair valid commits with valid same-author grants for one
/// exact collection. Malformed or unrelated structural evidence is inert and
/// omitted; clients independently verify every returned item.
pub(crate) fn grant_backed_commits(
    records: &[CollectionRecord],
    grants: &[CollectionGossip],
    collection: CollectionId,
) -> Vec<CollectionCommitEvidence> {
    let valid_grants: BTreeMap<[u8; 32], CollectionGossip> = grants
        .iter()
        .copied()
        .filter(|grant| grant.collection() == collection && grant.verify_strict().is_ok())
        .map(|grant| (grant.public_key().raw, grant))
        .collect();

    let mut evidence: Vec<_> = records
        .iter()
        .filter_map(|record| match record {
            CollectionRecord::Commit(commit) if commit.collection() == collection => {
                let grant = valid_grants.get(&commit.public_key().raw)?;
                CollectionCommitEvidence::new(*grant, *commit).ok()
            }
            CollectionRecord::Commit(_)
            | CollectionRecord::Merge(_)
            | CollectionRecord::Derive(_) => None,
        })
        .collect();
    evidence.sort_by_key(|item| item.commit.id());
    evidence.dedup_by_key(|item| item.commit.id());
    evidence
}

/// Complete read-only result of fetching one peer's grant-backed commits and
/// the content-addressed closure rooted by those commits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionFetch {
    collection: CollectionId,
    evidence: Vec<CollectionCommitEvidence>,
    roots: BTreeSet<RawHash>,
    blobs: BTreeMap<RawHash, Vec<u8>>,
}

impl CollectionFetch {
    /// Exact collection requested from the peer.
    pub fn collection(&self) -> CollectionId {
        self.collection
    }

    /// Strictly verified evidence, sorted by intrinsic commit id.
    pub fn evidence(&self) -> &[CollectionCommitEvidence] {
        &self.evidence
    }

    /// Descriptor/data/metadata roots named by the evidence.
    pub fn roots(&self) -> &BTreeSet<RawHash> {
        &self.roots
    }

    /// Verified bytes for every root and conservatively discovered descendant,
    /// keyed by exact Blake3 identity.
    pub fn blobs(&self) -> &BTreeMap<RawHash, Vec<u8>> {
        &self.blobs
    }

    /// Consume the transfer into its verified evidence and content-addressed
    /// blob bundle.
    ///
    /// This is the admission path: ownership moves into the store-facing
    /// validator without cloning every fetched blob.
    pub fn into_parts(self) -> (Vec<CollectionCommitEvidence>, BTreeMap<RawHash, Vec<u8>>) {
        (self.evidence, self.blobs)
    }
}

/// Enumerate strictly framed evidence for one exact collection over an
/// already-authenticated connection.
pub async fn op_collection_evidence<C: Conn>(
    conn: &C,
    collection: CollectionId,
) -> anyhow::Result<Vec<CollectionCommitEvidence>> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|error| anyhow::anyhow!("open collection evidence stream: {error}"))?;
    send_u8(&mut send, OP_COLLECTION_EVIDENCE).await?;
    send_hash(&mut send, &collection.raw).await?;
    send.shutdown()
        .await
        .map_err(|error| anyhow::anyhow!("finish collection evidence request: {error}"))?;

    let count = recv_u32_be(&mut recv).await?;
    if count == COLLECTION_EVIDENCE_REJECTED {
        return Err(anyhow::anyhow!(
            "server rejected collection evidence request: unrestricted read capability required"
        ));
    }
    if count > MAX_COLLECTION_EVIDENCE_ITEMS {
        return Err(anyhow::anyhow!(
            "collection evidence count {count} exceeds limit {MAX_COLLECTION_EVIDENCE_ITEMS}"
        ));
    }

    let mut evidence = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut bytes = [0u8; COLLECTION_COMMIT_EVIDENCE_LEN];
        recv.read_exact(&mut bytes)
            .await
            .map_err(|error| anyhow::anyhow!("read collection evidence: {error}"))?;
        let item = CollectionCommitEvidence::decode(&bytes)?;
        if item.commit().collection() != collection {
            return Err(anyhow::anyhow!(
                "server returned evidence for collection {} while {} was requested",
                hex::encode_upper(item.commit().collection().raw),
                hex::encode_upper(collection.raw),
            ));
        }
        evidence.push(item);
    }

    let mut trailing = [0u8; 1];
    if recv
        .read(&mut trailing)
        .await
        .map_err(|error| anyhow::anyhow!("finish collection evidence response: {error}"))?
        != 0
    {
        return Err(anyhow::anyhow!(
            "collection evidence response contains trailing bytes"
        ));
    }
    if !evidence
        .windows(2)
        .all(|pair| pair[0].commit().id() < pair[1].commit().id())
    {
        return Err(anyhow::anyhow!(
            "collection evidence response is not in canonical commit-id order"
        ));
    }
    Ok(evidence)
}

/// Fetch a peer's grant-backed evidence and every blob in the closure rooted
/// at the exact descriptor/data/metadata handles. The returned bytes are
/// hash-verified and remain inert; callers choose whether and how to admit
/// them.
pub async fn fetch_collection<C: Conn>(
    conn: &C,
    collection: CollectionId,
) -> anyhow::Result<CollectionFetch> {
    let evidence = op_collection_evidence(conn, collection).await?;
    let roots: BTreeSet<RawHash> = evidence
        .iter()
        .flat_map(CollectionCommitEvidence::roots)
        .collect();

    let mut closure = BTreeSet::new();
    let mut pending = roots.clone();
    while let Some(parent) = pending.pop_first() {
        if !closure.insert(parent) {
            continue;
        }
        for child in op_children(conn, &parent).await? {
            if !closure.contains(&child) {
                pending.insert(child);
            }
        }
    }

    let mut blobs = BTreeMap::new();
    for hash in closure {
        let bytes = op_get_blob(conn, &hash)
            .await?
            .ok_or_else(|| anyhow::anyhow!("peer omitted collection blob {}", hex::encode(hash)))?;
        let actual = *blake3::hash(&bytes).as_bytes();
        if actual != hash {
            return Err(anyhow::anyhow!(
                "collection blob hash mismatch: expected {}, received {}",
                hex::encode(hash),
                hex::encode(actual),
            ));
        }
        blobs.insert(hash, bytes);
    }

    Ok(CollectionFetch {
        collection,
        evidence,
        roots,
        blobs,
    })
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::IntoBlob;
    use triblespace_core::collection::{
        CollectionDescriptor, empty_metadata_handle, simplearchive_union,
    };
    use triblespace_core::id::Id;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::trible::TribleSet;

    use super::*;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn commit(author: &SigningKey, descriptor: &CollectionDescriptor) -> CollectionCommit {
        let data: Blob<SimpleArchive> = TribleSet::new().to_blob();
        CollectionCommit::sign(
            author,
            descriptor.handle(),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            empty_metadata_handle(),
        )
    }

    #[test]
    fn evidence_codec_roundtrips_and_checks_correspondence() {
        let author = SigningKey::from_bytes(&[7; 32]);
        let descriptor = simplearchive_union::descriptor(id(1));
        let commit = commit(&author, &descriptor);
        let grant = CollectionGossip::sign(&author, descriptor.handle());
        let evidence = CollectionCommitEvidence::new(grant, commit).unwrap();

        assert_eq!(
            CollectionCommitEvidence::decode(&evidence.encode()).unwrap(),
            evidence
        );
        assert_eq!(
            evidence.roots(),
            [
                descriptor.handle().raw,
                commit.data().raw,
                commit.metadata().raw,
            ]
        );

        let other = SigningKey::from_bytes(&[8; 32]);
        assert!(matches!(
            CollectionCommitEvidence::new(
                CollectionGossip::sign(&other, descriptor.handle()),
                commit,
            ),
            Err(CollectionCommitEvidenceError::AuthorMismatch { .. })
        ));
    }

    #[test]
    fn server_selection_is_exact_verified_and_deterministic() {
        let first_author = SigningKey::from_bytes(&[9; 32]);
        let second_author = SigningKey::from_bytes(&[10; 32]);
        let ungranted_author = SigningKey::from_bytes(&[11; 32]);
        let descriptor = simplearchive_union::descriptor(id(2));
        let other_descriptor = simplearchive_union::descriptor(id(3));
        let first = commit(&first_author, &descriptor);
        let second = commit(&second_author, &descriptor);
        let ungranted = commit(&ungranted_author, &descriptor);
        let unrelated = commit(&first_author, &other_descriptor);

        let records = vec![
            CollectionRecord::Commit(second),
            CollectionRecord::Commit(unrelated),
            CollectionRecord::Commit(ungranted),
            CollectionRecord::Commit(first),
        ];
        let grants = vec![
            CollectionGossip::sign(&second_author, descriptor.handle()),
            CollectionGossip::sign(&first_author, other_descriptor.handle()),
            CollectionGossip::sign(&first_author, descriptor.handle()),
        ];

        let selected = grant_backed_commits(&records, &grants, descriptor.handle());
        let mut expected = vec![first, second];
        expected.sort_by_key(CollectionCommit::id);
        assert_eq!(
            selected
                .iter()
                .map(CollectionCommitEvidence::commit)
                .collect::<Vec<_>>(),
            expected
        );
    }
}
