//! Collection-native, read-only reconciliation over an authenticated peer.
//!
//! The wire surface deliberately carries evidence rather than admission
//! decisions. A server enumerates only strictly verified
//! [`CollectionCommit`]s whose collection's own descriptor says the
//! collection travels. A client verifies each commit again and may admit it
//! without fetching any referenced blob. Nothing in this module mutates a
//! destination store.
//!
//! There used to be a second half to every item: a signed publication grant,
//! paired with the commit and checked alongside it. It is gone. Committing
//! into a collection whose identity declares it public *is* the consent, and
//! it cannot be given by accident, because a collection that stays put is a
//! different collection with a different handle.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use triblespace_core::collection::reach as collection_reach;
use triblespace_core::collection::{
    COLLECTION_COMMIT_BYTES_LEN, COLLECTION_DERIVE_BYTES_LEN, COLLECTION_MERGE_BYTES_LEN,
    CollectionCommit, CollectionDerive, CollectionHandle, CollectionMerge, CollectionRecord,
    CommitVerificationError, RecordDecodeError,
};
use triblespace_core::trible::TribleSet;
use triblespace_core::id::Id;
use triblespace_core::repo::{WANT_REQUEST_BYTES_LEN, WantRequest, WantRequestDecodeError};

use crate::protocol::{
    COLLECTION_EVIDENCE_REJECTED, COLLECTION_OPERATION_RECEIPTS_REJECTED, OP_COLLECTION_EVIDENCE,
    OP_COLLECTION_OPERATION_RECEIPTS, recv_u32_be, send_hash, send_u8,
};
use crate::transport::Conn;

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
            CollectionRecord::Derive(record) => {
                // A derive is shorter than a merge; the slot is sized to the
                // larger and the tail is zero, which the decoder requires.
                let mut slot = [0u8; COLLECTION_OPERATION_RECEIPT_BYTES_LEN];
                slot[..COLLECTION_DERIVE_BYTES_LEN].copy_from_slice(&record.to_bytes());
                bytes.extend_from_slice(&slot);
            }
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
        (CollectionRecord::Derive(record), WantRequest::Derive { target, input }) => {
            let (record_input, _) = record.mapping();
            record.target() == target && record_input == input
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
        WantRequest::Derive { .. } => {
            // A derive receipt is shorter than a merge one, and the slot is
            // sized to the larger. The tail must be zero so a receipt has one
            // canonical encoding.
            if bytes[COLLECTION_DERIVE_BYTES_LEN..].iter().any(|b| *b != 0) {
                return Err(CollectionOperationWireError::BlobRequest);
            }
            let mut exact = [0u8; COLLECTION_DERIVE_BYTES_LEN];
            exact.copy_from_slice(&bytes[..COLLECTION_DERIVE_BYTES_LEN]);
            Ok(CollectionRecord::Derive(CollectionDerive::from_bytes(exact)))
        }
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

/// Deterministically select every commit a relay may pass on.
///
/// The question this answers used to be "is there a matching grant?" -- a
/// separately signed record that anyone holding the key could mint later, for
/// a collection that already existed. It is now "does this collection's own
/// descriptor say it travels?", which is a question about the collection's
/// *name*. A collection that stays put and one that travels hash differently,
/// so the answer cannot be changed after the fact without changing which
/// collection is under discussion.
///
/// `descriptor` resolves a collection handle to its descriptor facts, and
/// `None` -- not resident -- is a refusal. That is the direction that fails
/// safe: a relay that cannot see permission does not have it.
///
/// The receiving side deliberately does *not* run this check. Admitting what
/// a peer handed you leaks nothing, and requiring a descriptor there would
/// make admission depend on blob residency the protocol is careful not to
/// require. The invariant still holds transitively, because a receiver that
/// later relays is a relay, and answers this question then.
pub fn relayable_commits(
    records: &[CollectionRecord],
    mut descriptor: impl FnMut(CollectionHandle) -> Option<TribleSet>,
) -> Vec<CollectionCommit> {
    let mut travels: BTreeMap<[u8; 32], bool> = BTreeMap::new();
    let mut selected: Vec<CollectionCommit> = records
        .iter()
        .filter_map(|record| match record {
            CollectionRecord::Commit(commit) => Some(commit),
            CollectionRecord::Merge(_) | CollectionRecord::Derive(_) => None,
        })
        .filter(|commit| commit.verify_strict().is_ok())
        .filter(|commit| {
            let collection = commit.collection();
            *travels.entry(collection.raw).or_insert_with(|| {
                descriptor(collection)
                    .as_ref()
                    .map(collection_reach::travels)
                    .unwrap_or(false)
            })
        })
        .copied()
        .collect();
    selected.sort_by_key(|commit| commit.id());
    selected.dedup_by_key(|commit| commit.id());
    selected
}

/// [`relayable_commits`], narrowed to one exact collection.
pub fn relayable_commits_for(
    records: &[CollectionRecord],
    descriptor: impl FnMut(CollectionHandle) -> Option<TribleSet>,
    collection: CollectionHandle,
) -> Vec<CollectionCommit> {
    relayable_commits(records, descriptor)
        .into_iter()
        .filter(|commit| commit.collection() == collection)
        .collect()
}

/// Enumerate strictly framed evidence for one exact collection over an
/// already-authenticated connection.
pub async fn op_collection_evidence<C: Conn>(
    conn: &C,
    collection: CollectionHandle,
) -> anyhow::Result<Vec<CollectionCommit>> {
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
        let mut bytes = [0u8; COLLECTION_COMMIT_BYTES_LEN];
        recv.read_exact(&mut bytes)
            .await
            .map_err(|error| anyhow::anyhow!("read collection evidence: {error}"))?;
        let item = CollectionCommit::from_bytes(bytes);
        item.verify_strict()
            .map_err(|error| anyhow::anyhow!("collection evidence failed verification: {error}"))?;
        if item.collection() != collection {
            return Err(anyhow::anyhow!(
                "server returned evidence for collection {} while {} was requested",
                hex::encode_upper(item.collection().raw),
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
        .all(|pair| pair[0].id() < pair[1].id())
    {
        return Err(anyhow::anyhow!(
            "collection evidence response is not in canonical commit-id order"
        ));
    }
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use triblespace_core::collection::reach;
    use ed25519_dalek::SigningKey;
    use triblespace_core::blob::encodings::UnknownBlob;
    use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace_core::blob::{Blob, IntoBlob};
    use triblespace_core::collection::records::CollectionName;
    use triblespace_core::collection::{
        CollectionData, CollectionHandle, empty_metadata_handle, simplearchive_union,
    };
    use triblespace_core::trible::Fragment;
    use triblespace_core::inline::Inline;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::trible::TribleSet;

    use super::*;

    /// The one team every collection in these tests belongs to.
    fn test_team() -> triblespace_core::collection::VerifyingKey {
        SigningKey::from_bytes(&[1; 32]).verifying_key()
    }

    /// A named root of the canonical `SimpleArchive` union kind that stays
    /// put: it declares no reach, and so travels nowhere.
    fn root(name: &str) -> Fragment {
        simplearchive_union::descriptor(
            &CollectionName::new(name).unwrap(),
            test_team(),
            reach::private(),
        )
    }

    /// The same root, named to travel. It is a *different collection*: same
    /// name, same team, different handle.
    fn public_root(name: &str) -> Fragment {
        simplearchive_union::descriptor(
            &CollectionName::new(name).unwrap(),
            test_team(),
            reach::public(),
        )
    }

    /// A descriptor lookup holding exactly the descriptors listed.
    fn resident(descriptors: &[&Fragment]) -> std::collections::BTreeMap<[u8; 32], TribleSet> {
        descriptors
            .iter()
            .map(|fragment| (collection_of(fragment).raw, fragment.facts().clone()))
            .collect()
    }

    /// These wire tests only need identities to address records by; nothing
    /// stores the descriptors they come from.
    fn collection_of(descriptor: &Fragment) -> CollectionHandle {
        IntoBlob::<SimpleArchive>::to_blob(descriptor.facts().clone()).get_handle()
    }

    fn commit(author: &SigningKey, descriptor: &Fragment) -> CollectionCommit {
        let data: Blob<SimpleArchive> = TribleSet::new().to_blob();
        CollectionCommit::sign(
            author,
            collection_of(descriptor),
            Handle::<SimpleArchive>::to_hash(data.get_handle()),
            empty_metadata_handle(),
        )
    }

    fn data(byte: u8) -> CollectionData {
        Inline::new([byte; 32])
    }

    #[test]
    fn operation_request_codec_reuses_want_bytes_and_rejects_blobs() {
        let source = root("c41");
        let target = root("c42");
        let merge = WantRequest::merge(collection_of(&source), data(9), data(2));
        let derive = WantRequest::derive(collection_of(&target), data(3));

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
        let descriptor = root("c43");
        let other_descriptor = root("c44");
        let request = WantRequest::merge(collection_of(&descriptor), data(1), data(2));
        let first = CollectionMerge::new(collection_of(&descriptor), data(1), data(2), data(3));
        let conflicting = CollectionMerge::new(collection_of(&descriptor), data(1), data(2), data(4));
        let unrelated = CollectionMerge::new(collection_of(&descriptor), data(1), data(9), data(5));
        let wrong_collection =
            CollectionMerge::new(collection_of(&other_descriptor), data(1), data(2), data(6));
        let derive = CollectionDerive::new(collection_of(&other_descriptor), data(1), data(7));
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
        let target = root("c47");
        let request = WantRequest::derive(collection_of(&target), data(1));
        let first = CollectionDerive::new(collection_of(&target), data(1), data(2));
        let conflicting = CollectionDerive::new(collection_of(&target), data(1), data(3));
        let unrelated = CollectionDerive::new(collection_of(&target), data(9), data(2));

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
        // Derive receipts occupy a merge-sized slot, zero-padded.
        let mut slot = [0u8; COLLECTION_OPERATION_RECEIPT_BYTES_LEN];
        slot[..COLLECTION_DERIVE_BYTES_LEN].copy_from_slice(&unrelated.to_bytes());
        mismatched.extend_from_slice(&slot);
        assert!(matches!(
            decode_collection_operation_receipts(request, &mismatched),
            Err(CollectionOperationWireError::ReceiptDoesNotAnswerRequest { .. })
        ));
    }

    #[test]
    fn receipt_response_decoder_enforces_count_framing_order_and_rejection() {
        let descriptor = root("c48");
        let request = WantRequest::merge(collection_of(&descriptor), data(1), data(2));
        let first = CollectionMerge::new(collection_of(&descriptor), data(1), data(2), data(3));
        let second = CollectionMerge::new(collection_of(&descriptor), data(1), data(2), data(4));
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
    fn a_relay_serves_what_the_descriptor_permits_and_nothing_else() {
        let first_author = SigningKey::from_bytes(&[9; 32]);
        let second_author = SigningKey::from_bytes(&[10; 32]);
        let public = public_root("c2");
        let private = root("c3");

        let first = commit(&first_author, &public);
        let second = commit(&second_author, &public);
        // The same author, writing into a collection that stays put.
        let secret = commit(&first_author, &private);

        let records = vec![
            CollectionRecord::Commit(second),
            CollectionRecord::Commit(secret),
            CollectionRecord::Commit(first),
        ];
        let store = resident(&[&public, &private]);
        let lookup = |handle: CollectionHandle| store.get(&handle.raw).cloned();

        // Both authors' commits travel, because the *collection* travels.
        // Reach is not per-author, and neither author signed anything beyond
        // the commit itself.
        let mut expected = vec![first, second];
        expected.sort_by_key(|commit| commit.id());
        assert_eq!(relayable_commits(&records, lookup), expected);

        // The private collection's commit does not travel, though the author
        // of one of the published commits wrote it.
        assert!(relayable_commits_for(&records, lookup, collection_of(&private)).is_empty());
        assert_eq!(
            relayable_commits_for(&records, lookup, collection_of(&public)),
            expected
        );
    }

    #[test]
    fn a_descriptor_a_relay_cannot_see_is_a_refusal() {
        let author = SigningKey::from_bytes(&[12; 32]);
        let public = public_root("c4");
        let records = vec![CollectionRecord::Commit(commit(&author, &public))];

        // Permission it cannot read is permission it does not have. Serving
        // on the strength of an unresolvable descriptor would publish exactly
        // the material whose descriptor might have refused.
        let empty = resident(&[]);
        assert!(
            relayable_commits(&records, |handle: CollectionHandle| empty
                .get(&handle.raw)
                .cloned())
            .is_empty()
        );

        let store = resident(&[&public]);
        assert_eq!(
            relayable_commits(&records, |handle: CollectionHandle| store
                .get(&handle.raw)
                .cloned())
            .len(),
            1
        );
    }
}
