//! Collection-native, read-only reconciliation over an authenticated peer.
//!
//! The wire surface deliberately carries evidence rather than admission
//! decisions. A server enumerates only strictly verified
//! [`CollectionGossip`] / [`CollectionCommit`] pairs for one exact
//! descriptor handle. A client verifies the pair again and may admit that
//! sparse evidence without fetching any referenced blob. Nothing in this
//! module mutates a destination store.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use triblespace_core::collection::{
    COLLECTION_COMMIT_BYTES_LEN, COLLECTION_GOSSIP_BYTES_LEN, COLLECTION_MERGE_BYTES_LEN,
    CollectionCommit, CollectionDerive, CollectionGossip, CollectionId, CollectionMerge,
    CollectionRecord, CommitVerificationError, GossipVerificationError, RecordDecodeError,
};
use triblespace_core::id::Id;
use triblespace_core::repo::{WANT_REQUEST_BYTES_LEN, WantRequest, WantRequestDecodeError};

use crate::protocol::{
    COLLECTION_EVIDENCE_REJECTED, COLLECTION_OPERATION_RECEIPTS_REJECTED, OP_COLLECTION_EVIDENCE,
    OP_COLLECTION_OPERATION_RECEIPTS, recv_u32_be, send_hash, send_u8,
};
use crate::transport::Conn;

/// Exact byte length of one grant-backed commit evidence item.
///
/// The layout is the canonical 128-byte gossip witness followed by the
/// canonical 192-byte dense [`CollectionCommit`].
pub const COLLECTION_COMMIT_EVIDENCE_LEN: usize =
    COLLECTION_GOSSIP_BYTES_LEN + COLLECTION_COMMIT_BYTES_LEN;

/// Defensive upper bound on the number of evidence items accepted from one
/// response. The wire count remains a `u32`; this bound prevents a malicious
/// server from forcing an unbounded allocation before any evidence is read.
pub const MAX_COLLECTION_EVIDENCE_ITEMS: u32 = 1 << 20;

/// Exact byte length of one untagged collection-operation receipt.
///
/// Merge and derive records deliberately share this width. The request kind
/// supplies the type information, so the response does not repeat a tag for
/// every item.
pub const COLLECTION_OPERATION_RECEIPT_BYTES_LEN: usize = COLLECTION_MERGE_BYTES_LEN;

/// Rejection sentinel in the collection-operation receipt response count.
/// Defensive upper bound on operation receipts accepted in one response.
pub const MAX_COLLECTION_OPERATION_RECEIPTS: u32 = 1 << 20;

/// A decoded collection-operation response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionOperationReceiptResponse {
    /// The peer declined the request, normally because authorization failed.
    Rejected,
    /// Canonically ordered, distinct exact receipts answering the request.
    Receipts(Vec<CollectionRecord>),
}

/// Encode one exact collection-operation request.
///
/// Blob wants use the blob protocol and are rejected at this typed boundary.
pub fn encode_collection_operation_request(
    request: WantRequest,
) -> Result<[u8; WANT_REQUEST_BYTES_LEN], CollectionOperationWireError> {
    ensure_operation_request(request)?;
    let bytes = request.to_bytes();
    // `WantRequest` variants are public so callers may bypass the canonical
    // constructors (notably Merge input ordering). Refuse to put a request on
    // the wire unless its own strict decoder accepts the representation.
    let canonical =
        WantRequest::from_bytes(bytes).map_err(CollectionOperationWireError::InvalidRequest)?;
    debug_assert_eq!(canonical, request);
    Ok(bytes)
}

/// Decode one exact canonical collection-operation request.
pub fn decode_collection_operation_request(
    bytes: [u8; WANT_REQUEST_BYTES_LEN],
) -> Result<WantRequest, CollectionOperationWireError> {
    let request =
        WantRequest::from_bytes(bytes).map_err(CollectionOperationWireError::InvalidRequest)?;
    ensure_operation_request(request)?;
    Ok(request)
}

/// Select exact receipts for a request in deterministic intrinsic-ID order.
///
/// Distinct results for the same inputs are retained as explicit conflicting
/// evidence. Byte-identical duplicate records collapse by intrinsic id.
pub fn collection_operation_receipts(
    request: WantRequest,
    records: impl IntoIterator<Item = CollectionRecord>,
) -> Result<Vec<CollectionRecord>, CollectionOperationWireError> {
    ensure_operation_request(request)?;
    let receipts: BTreeMap<Id, CollectionRecord> = records
        .into_iter()
        .filter(|record| record_answers_request(*record, request))
        .map(|record| (record.id(), record))
        .collect();
    Ok(receipts.values().copied().collect())
}

/// Encode a canonical response from arbitrary locally known records.
///
/// The four-byte big-endian count is followed by full untagged 128-byte merge
/// or derive payloads. Selection, sorting, and deduplication happen here so a
/// store's iteration order cannot leak into the wire representation.
pub fn encode_collection_operation_receipts(
    request: WantRequest,
    records: impl IntoIterator<Item = CollectionRecord>,
) -> Result<Vec<u8>, CollectionOperationWireError> {
    let receipts = collection_operation_receipts(request, records)?;
    if receipts.len() > MAX_COLLECTION_OPERATION_RECEIPTS as usize {
        return Err(CollectionOperationWireError::TooManyReceipts {
            count: receipts.len(),
            limit: MAX_COLLECTION_OPERATION_RECEIPTS,
        });
    }

    let mut bytes = Vec::with_capacity(4 + receipts.len() * COLLECTION_OPERATION_RECEIPT_BYTES_LEN);
    bytes.extend_from_slice(&(receipts.len() as u32).to_be_bytes());
    for receipt in receipts {
        match receipt {
            CollectionRecord::Merge(record) => bytes.extend_from_slice(&record.to_bytes()),
            CollectionRecord::Derive(record) => bytes.extend_from_slice(&record.to_bytes()),
            CollectionRecord::Commit(_) => {
                unreachable!("receipt selection excludes collection commits")
            }
        }
    }
    Ok(bytes)
}

/// Encode the reserved rejection response.
pub const fn encode_collection_operation_rejection() -> [u8; 4] {
    COLLECTION_OPERATION_RECEIPTS_REJECTED.to_be_bytes()
}

/// Ask one already-authenticated peer for every exact receipt answering an
/// operation request. The RPC remains read-only: callers decide whether and
/// where to admit the returned evidence.
pub async fn op_collection_operation_receipts<C: Conn>(
    conn: &C,
    request: WantRequest,
) -> anyhow::Result<CollectionOperationReceiptResponse> {
    let request_bytes = encode_collection_operation_request(request)?;
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|error| anyhow::anyhow!("open collection receipt stream: {error}"))?;
    send_u8(&mut send, OP_COLLECTION_OPERATION_RECEIPTS).await?;
    send.write_all(&request_bytes)
        .await
        .map_err(|error| anyhow::anyhow!("send collection receipt request: {error}"))?;
    send.shutdown()
        .await
        .map_err(|error| anyhow::anyhow!("finish collection receipt request: {error}"))?;

    let count = recv_u32_be(&mut recv).await?;
    if count == COLLECTION_OPERATION_RECEIPTS_REJECTED {
        let mut trailing = [0u8; 1];
        if recv.read(&mut trailing).await.map_err(|error| {
            anyhow::anyhow!("finish rejected collection receipt response: {error}")
        })? != 0
        {
            return Err(anyhow::anyhow!(
                "rejected collection receipt response contains trailing bytes"
            ));
        }
        return Ok(CollectionOperationReceiptResponse::Rejected);
    }
    if count > MAX_COLLECTION_OPERATION_RECEIPTS {
        return Err(CollectionOperationWireError::ReceiptCountExceedsLimit {
            count,
            limit: MAX_COLLECTION_OPERATION_RECEIPTS,
        }
        .into());
    }

    let payload_len = (count as usize)
        .checked_mul(COLLECTION_OPERATION_RECEIPT_BYTES_LEN)
        .expect("bounded receipt count cannot overflow usize");
    let mut response = Vec::with_capacity(4 + payload_len);
    response.extend_from_slice(&count.to_be_bytes());
    response.resize(4 + payload_len, 0);
    recv.read_exact(&mut response[4..])
        .await
        .map_err(|error| anyhow::anyhow!("read collection operation receipts: {error}"))?;
    let mut trailing = [0u8; 1];
    if recv
        .read(&mut trailing)
        .await
        .map_err(|error| anyhow::anyhow!("finish collection receipt response: {error}"))?
        != 0
    {
        return Err(anyhow::anyhow!(
            "collection receipt response contains trailing bytes"
        ));
    }
    Ok(decode_collection_operation_receipts(request, &response)?)
}

/// Decode one complete canonical collection-operation response.
///
/// The caller supplies the request because response items intentionally omit
/// a per-record tag. Successful responses must use strictly increasing record
/// IDs and contain only exact answers to the request.
pub fn decode_collection_operation_receipts(
    request: WantRequest,
    bytes: &[u8],
) -> Result<CollectionOperationReceiptResponse, CollectionOperationWireError> {
    ensure_operation_request(request)?;
    if bytes.len() < 4 {
        return Err(CollectionOperationWireError::WrongResponseLength {
            expected: 4,
            actual: bytes.len(),
        });
    }

    let count = u32::from_be_bytes(bytes[..4].try_into().expect("checked count width"));
    if count == COLLECTION_OPERATION_RECEIPTS_REJECTED {
        if bytes.len() != 4 {
            return Err(CollectionOperationWireError::WrongResponseLength {
                expected: 4,
                actual: bytes.len(),
            });
        }
        return Ok(CollectionOperationReceiptResponse::Rejected);
    }
    if count > MAX_COLLECTION_OPERATION_RECEIPTS {
        return Err(CollectionOperationWireError::ReceiptCountExceedsLimit {
            count,
            limit: MAX_COLLECTION_OPERATION_RECEIPTS,
        });
    }

    let expected = 4usize
        .checked_add(
            (count as usize)
                .checked_mul(COLLECTION_OPERATION_RECEIPT_BYTES_LEN)
                .expect("bounded receipt count cannot overflow usize"),
        )
        .expect("bounded receipt response cannot overflow usize");
    if bytes.len() != expected {
        return Err(CollectionOperationWireError::WrongResponseLength {
            expected,
            actual: bytes.len(),
        });
    }

    let mut receipts = Vec::with_capacity(count as usize);
    let mut previous = None;
    for payload in bytes[4..].chunks_exact(COLLECTION_OPERATION_RECEIPT_BYTES_LEN) {
        let payload: [u8; COLLECTION_OPERATION_RECEIPT_BYTES_LEN] =
            payload.try_into().expect("checked response framing");
        let receipt = decode_operation_receipt(request, payload)?;
        if !record_answers_request(receipt, request) {
            return Err(CollectionOperationWireError::ReceiptDoesNotAnswerRequest {
                receipt: receipt.id(),
            });
        }
        if let Some(previous) = previous
            && previous >= receipt.id()
        {
            return Err(CollectionOperationWireError::NonCanonicalReceiptOrder {
                previous,
                current: receipt.id(),
            });
        }
        previous = Some(receipt.id());
        receipts.push(receipt);
    }
    Ok(CollectionOperationReceiptResponse::Receipts(receipts))
}

fn ensure_operation_request(request: WantRequest) -> Result<(), CollectionOperationWireError> {
    match request {
        WantRequest::Merge { .. } | WantRequest::Derive { .. } => Ok(()),
        WantRequest::Blob { .. } => Err(CollectionOperationWireError::BlobRequest),
    }
}

fn record_answers_request(record: CollectionRecord, request: WantRequest) -> bool {
    match (record, request) {
        (
            CollectionRecord::Merge(record),
            WantRequest::Merge {
                collection,
                low,
                high,
            },
        ) => record.collection() == collection && record.inputs() == (low, high),
        (
            CollectionRecord::Derive(record),
            WantRequest::Derive {
                source,
                target,
                input,
            },
        ) => {
            let (record_input, _) = record.mapping();
            record.source() == source && record.target() == target && record_input == input
        }
        (CollectionRecord::Commit(_), _)
        | (CollectionRecord::Merge(_), _)
        | (CollectionRecord::Derive(_), _) => false,
    }
}

fn decode_operation_receipt(
    request: WantRequest,
    bytes: [u8; COLLECTION_OPERATION_RECEIPT_BYTES_LEN],
) -> Result<CollectionRecord, CollectionOperationWireError> {
    match request {
        WantRequest::Merge { .. } => CollectionMerge::from_bytes(bytes)
            .map(CollectionRecord::Merge)
            .map_err(CollectionOperationWireError::InvalidReceipt),
        WantRequest::Derive { .. } => Ok(CollectionRecord::Derive(CollectionDerive::from_bytes(
            bytes,
        ))),
        WantRequest::Blob { .. } => Err(CollectionOperationWireError::BlobRequest),
    }
}

/// Structural or canonicality failure in the collection-operation wire codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionOperationWireError {
    /// The 97-byte request was not structurally canonical.
    InvalidRequest(WantRequestDecodeError),
    /// Blob wants belong to the blob-fetch protocol.
    BlobRequest,
    /// A successful response was too short, truncated, or had trailing bytes.
    WrongResponseLength { expected: usize, actual: usize },
    /// A local response could not be represented without colliding with the
    /// reserved rejection sentinel.
    TooManyReceipts { count: usize, limit: u32 },
    /// The peer claimed more receipts than this implementation will allocate.
    ReceiptCountExceedsLimit { count: u32, limit: u32 },
    /// One dense merge payload was structurally noncanonical.
    InvalidReceipt(RecordDecodeError),
    /// A decoded receipt named different operation inputs.
    ReceiptDoesNotAnswerRequest { receipt: Id },
    /// Response records were not strictly increasing by intrinsic id.
    NonCanonicalReceiptOrder { previous: Id, current: Id },
}

impl fmt::Display for CollectionOperationWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(error) => write!(formatter, "invalid operation request: {error}"),
            Self::BlobRequest => formatter.write_str("blob want is not a collection operation"),
            Self::WrongResponseLength { expected, actual } => write!(
                formatter,
                "collection operation response is {actual} bytes, expected {expected}"
            ),
            Self::TooManyReceipts { count, limit } => write!(
                formatter,
                "collection operation response has {count} receipts; limit is {limit}"
            ),
            Self::ReceiptCountExceedsLimit { count, limit } => write!(
                formatter,
                "collection operation receipt count {count} exceeds limit {limit}"
            ),
            Self::InvalidReceipt(error) => {
                write!(formatter, "invalid collection operation receipt: {error}")
            }
            Self::ReceiptDoesNotAnswerRequest { receipt } => write!(
                formatter,
                "collection operation receipt {receipt:X} does not answer the request"
            ),
            Self::NonCanonicalReceiptOrder { previous, current } => write!(
                formatter,
                "collection operation receipts are not in canonical order ({previous:X} then {current:X})"
            ),
        }
    }
}

impl Error for CollectionOperationWireError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidRequest(error) => Some(error),
            Self::InvalidReceipt(error) => Some(error),
            Self::BlobRequest
            | Self::WrongResponseLength { .. }
            | Self::TooManyReceipts { .. }
            | Self::ReceiptCountExceedsLimit { .. }
            | Self::ReceiptDoesNotAnswerRequest { .. }
            | Self::NonCanonicalReceiptOrder { .. } => None,
        }
    }
}

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
        let commit = CollectionCommit::from_bytes(
            bytes[COLLECTION_GOSSIP_BYTES_LEN..]
                .try_into()
                .expect("checked evidence length"),
        );
        Self::new(grant, commit)
    }

    /// Encode the already-verified pair into its fixed canonical layout.
    pub fn encode(&self) -> [u8; COLLECTION_COMMIT_EVIDENCE_LEN] {
        let mut bytes = [0u8; COLLECTION_COMMIT_EVIDENCE_LEN];
        bytes[..COLLECTION_GOSSIP_BYTES_LEN].copy_from_slice(&self.grant.to_bytes());
        bytes[COLLECTION_GOSSIP_BYTES_LEN..].copy_from_slice(&self.commit.to_bytes());
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

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::encodings::UnknownBlob;
    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace_core::blob::{Blob, IntoBlob};
    use triblespace_core::collection::{
        CollectionData, CollectionDescriptor, empty_metadata_handle, simplearchive_union,
    };
    use triblespace_core::id::Id;
    use triblespace_core::inline::Inline;
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

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    #[test]
    fn operation_request_codec_reuses_want_bytes_and_rejects_blobs() {
        let source = simplearchive_union::descriptor(id(41));
        let target = simplearchive_union::descriptor(id(42));
        let merge = WantRequest::merge(source.handle(), data(9), data(2));
        let derive = WantRequest::derive(source.handle(), target.handle(), data(3));

        for request in [merge, derive] {
            let bytes = encode_collection_operation_request(request).unwrap();
            assert_eq!(bytes, request.to_bytes());
            assert_eq!(decode_collection_operation_request(bytes), Ok(request));
        }

        let blob = WantRequest::blob(Inline::<Handle<UnknownBlob>>::new([7; 32]));
        assert_eq!(
            encode_collection_operation_request(blob),
            Err(CollectionOperationWireError::BlobRequest)
        );
        assert_eq!(
            decode_collection_operation_request(blob.to_bytes()),
            Err(CollectionOperationWireError::BlobRequest)
        );

        let mut noncanonical = merge.to_bytes();
        noncanonical[1 + 32..1 + 64].fill(0xFF);
        noncanonical[1 + 64..1 + 96].fill(0x01);
        assert_eq!(
            decode_collection_operation_request(noncanonical),
            Err(CollectionOperationWireError::InvalidRequest(
                WantRequestDecodeError::NonCanonicalMergeInputs,
            ))
        );

        let WantRequest::Merge {
            collection,
            low,
            high,
        } = merge
        else {
            unreachable!()
        };
        let directly_constructed = WantRequest::Merge {
            collection,
            low: high,
            high: low,
        };
        assert_eq!(
            encode_collection_operation_request(directly_constructed),
            Err(CollectionOperationWireError::InvalidRequest(
                WantRequestDecodeError::NonCanonicalMergeInputs,
            ))
        );
    }

    #[test]
    fn merge_receipt_response_is_untagged_exact_and_deterministic() {
        let descriptor = simplearchive_union::descriptor(id(43));
        let other_descriptor = simplearchive_union::descriptor(id(44));
        let request = WantRequest::merge(descriptor.handle(), data(1), data(2));
        let first = CollectionMerge::new(descriptor.handle(), data(1), data(2), data(3));
        let conflicting = CollectionMerge::new(descriptor.handle(), data(1), data(2), data(4));
        let unrelated = CollectionMerge::new(descriptor.handle(), data(1), data(9), data(5));
        let wrong_collection =
            CollectionMerge::new(other_descriptor.handle(), data(1), data(2), data(6));
        let derive = CollectionDerive::new(
            descriptor.handle(),
            other_descriptor.handle(),
            data(1),
            data(7),
        );
        let signed = commit(&SigningKey::from_bytes(&[45; 32]), &descriptor);

        let encoded = encode_collection_operation_receipts(
            request,
            [
                CollectionRecord::Derive(derive),
                CollectionRecord::Merge(conflicting),
                CollectionRecord::Merge(unrelated),
                CollectionRecord::Commit(signed),
                CollectionRecord::Merge(first),
                CollectionRecord::Merge(wrong_collection),
                CollectionRecord::Merge(conflicting),
            ],
        )
        .unwrap();

        let mut expected = vec![
            CollectionRecord::Merge(first),
            CollectionRecord::Merge(conflicting),
        ];
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(&encoded[..4], &2u32.to_be_bytes());
        assert_eq!(encoded.len(), 4 + 2 * COLLECTION_MERGE_BYTES_LEN);
        for (payload, record) in encoded[4..]
            .chunks_exact(COLLECTION_OPERATION_RECEIPT_BYTES_LEN)
            .zip(&expected)
        {
            let CollectionRecord::Merge(record) = record else {
                unreachable!()
            };
            assert_eq!(payload, record.to_bytes());
        }
        assert_eq!(
            decode_collection_operation_receipts(request, &encoded),
            Ok(CollectionOperationReceiptResponse::Receipts(expected))
        );
    }

    #[test]
    fn derive_receipts_preserve_conflicts_and_require_exact_inputs() {
        let source = simplearchive_union::descriptor(id(46));
        let target = simplearchive_union::descriptor(id(47));
        let request = WantRequest::derive(source.handle(), target.handle(), data(1));
        let first = CollectionDerive::new(source.handle(), target.handle(), data(1), data(2));
        let conflicting = CollectionDerive::new(source.handle(), target.handle(), data(1), data(3));
        let unrelated = CollectionDerive::new(source.handle(), target.handle(), data(9), data(2));

        let encoded = encode_collection_operation_receipts(
            request,
            [
                CollectionRecord::Derive(conflicting),
                CollectionRecord::Derive(unrelated),
                CollectionRecord::Derive(first),
            ],
        )
        .unwrap();
        let CollectionOperationReceiptResponse::Receipts(receipts) =
            decode_collection_operation_receipts(request, &encoded).unwrap()
        else {
            panic!("successful response decoded as rejection")
        };
        assert_eq!(receipts.len(), 2);
        assert!(receipts.contains(&CollectionRecord::Derive(first)));
        assert!(receipts.contains(&CollectionRecord::Derive(conflicting)));

        let mut mismatched = Vec::from(1u32.to_be_bytes());
        mismatched.extend_from_slice(&unrelated.to_bytes());
        assert!(matches!(
            decode_collection_operation_receipts(request, &mismatched),
            Err(CollectionOperationWireError::ReceiptDoesNotAnswerRequest { .. })
        ));
    }

    #[test]
    fn receipt_response_decoder_enforces_count_framing_order_and_rejection() {
        let descriptor = simplearchive_union::descriptor(id(48));
        let request = WantRequest::merge(descriptor.handle(), data(1), data(2));
        let first = CollectionMerge::new(descriptor.handle(), data(1), data(2), data(3));
        let second = CollectionMerge::new(descriptor.handle(), data(1), data(2), data(4));
        let mut ordered = [first, second];
        ordered.sort_by_key(CollectionMerge::id);

        assert_eq!(
            decode_collection_operation_receipts(request, &encode_collection_operation_rejection()),
            Ok(CollectionOperationReceiptResponse::Rejected)
        );

        let mut rejected_with_trailing = Vec::from(encode_collection_operation_rejection());
        rejected_with_trailing.push(0);
        assert_eq!(
            decode_collection_operation_receipts(request, &rejected_with_trailing),
            Err(CollectionOperationWireError::WrongResponseLength {
                expected: 4,
                actual: 5,
            })
        );

        let mut reversed = Vec::from(2u32.to_be_bytes());
        reversed.extend_from_slice(&ordered[1].to_bytes());
        reversed.extend_from_slice(&ordered[0].to_bytes());
        assert!(matches!(
            decode_collection_operation_receipts(request, &reversed),
            Err(CollectionOperationWireError::NonCanonicalReceiptOrder { .. })
        ));

        let mut truncated = Vec::from(2u32.to_be_bytes());
        truncated.extend_from_slice(&ordered[0].to_bytes());
        assert_eq!(
            decode_collection_operation_receipts(request, &truncated),
            Err(CollectionOperationWireError::WrongResponseLength {
                expected: 4 + 2 * COLLECTION_OPERATION_RECEIPT_BYTES_LEN,
                actual: 4 + COLLECTION_OPERATION_RECEIPT_BYTES_LEN,
            })
        );

        let over_limit = (MAX_COLLECTION_OPERATION_RECEIPTS + 1).to_be_bytes();
        assert_eq!(
            decode_collection_operation_receipts(request, &over_limit),
            Err(CollectionOperationWireError::ReceiptCountExceedsLimit {
                count: MAX_COLLECTION_OPERATION_RECEIPTS + 1,
                limit: MAX_COLLECTION_OPERATION_RECEIPTS,
            })
        );
    }

    #[test]
    fn evidence_codec_roundtrips_and_checks_correspondence() {
        let author = SigningKey::from_bytes(&[7; 32]);
        let descriptor = simplearchive_union::descriptor(id(1));
        let commit = commit(&author, &descriptor);
        let grant = CollectionGossip::sign(&author, descriptor.handle());
        let evidence = CollectionCommitEvidence::new(grant, commit).unwrap();

        assert_eq!(COLLECTION_COMMIT_EVIDENCE_LEN, 320);

        assert_eq!(
            CollectionCommitEvidence::decode(&evidence.encode()).unwrap(),
            evidence
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
