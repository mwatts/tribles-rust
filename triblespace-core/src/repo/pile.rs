//! A Pile is an append-only collection of blobs and branches stored in a single
//! file. It is designed as a durable local repository storage that can be safely
//! shared between threads.
//!
//! The pile operates as a **WAL-as-a-DB**: the write-ahead log _is_ the database.
//! All indices and metadata are reconstructed from the log on startup and no
//! additional state is persisted elsewhere.
//!
//! The pile treats its file as an immutable append-only log. Once a record lies
//! below `applied_length` and its bytes have been returned by
//! `get` or `apply_next`, those bytes are
//! assumed permanent. Modifying any part of the pile other than appending new
//! records is undefined behaviour. The un-applied tail may hide a partial
//! append after a crash, so validation and repair only operate on offsets
//! beyond `applied_length`. Each record's [`ValidationState`](crate::repo::pile::ValidationState) is cached for the
//! lifetime of the process under this immutability assumption.
//!
//! For layout and recovery details see the [Pile
//! Format](../../book/src/pile-format.md) chapter of the Tribles Book.

use anybytes::Bytes;
use hex_literal::hex;
use memmap2::MmapOptions;
use memmap2::MmapRaw;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::convert::Infallible;
use std::error::Error;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::IoSlice;
use std::io::Write;
use std::path::Path;
use std::ptr::slice_from_raw_parts;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use zerocopy::Immutable;
use zerocopy::IntoBytes;
use zerocopy::KnownLayout;
use zerocopy::TryFromBytes;

use crate::blob::encodings::UnknownBlob;
use crate::blob::Blob;
use crate::blob::BlobEncoding;
use crate::blob::IntoBlob;
use crate::blob::TryFromBlob;
use crate::collection::store::selectors_match_record;
use crate::collection::{
    CollectionCommit, CollectionDerive, CollectionGossip, CollectionGossipStore, CollectionMerge,
    CollectionRecord, CollectionRecordSelector, CollectionStore, KIND_COLLECTION_GOSSIP,
};
use crate::id::Id;
use crate::id::RawId;
use crate::inline::encodings::ed25519::{ED25519PublicKey, ED25519RComponent, ED25519SComponent};
use crate::inline::encodings::hash::Blake3;
use crate::inline::encodings::hash::Hash;
use crate::inline::Inline;
use crate::inline::InlineEncoding;
use crate::inline::RawInline;
use crate::patch::Entry;
use crate::patch::IdentitySchema;
use crate::patch::PATCH;
use crate::prelude::blobencodings::SimpleArchive;
use crate::prelude::inlineencodings::Handle;
use crate::repo::{WantRequest, WANT_REQUEST_BYTES_LEN};

const MAGIC_MARKER_BLOB: RawId = hex!("1E08B022FF2F47B6EBACF1D68EB35D96");
const MAGIC_MARKER_BRANCH: RawId = hex!("2BC991A7F5D5D2A3A468C53B0AA03504");
const MAGIC_MARKER_BRANCH_TOMBSTONE: RawId = hex!("E888CC787202D2AE4C654BFE9699C430");
/// Generic forward-compatible pile-record envelope marker.
///
/// Minted on 2026-08-11 with `trible genid`:
/// `E5A95E5D8A0BBA8782E46B9C9E73B313`.
///
/// New records place this marker first, their existing V3/V4 marker in the
/// next 16 bytes as a semantic record-kind id, and their total span as a
/// count of 256-byte blocks in bytes 32..36. Readers can therefore cross an
/// unknown record kind without assigning semantics to it.
const MAGIC_MARKER_ENVELOPE: RawId = hex!("E5A95E5D8A0BBA8782E46B9C9E73B313");
/// V3 record markers, minted 2026-06-29 via `trible genid`. Legacy V3 records
/// place these first; current records reuse them as envelope kind IDs. Both
/// layouts have a fixed 256-byte header and 256-byte record granularity. Consequences:
///   * blob data starts at a constant `record_start + 256` — reads are
///     position-INDEPENDENT (no offset-derived pad), so a record survives
///     relocation/`cat` and is found correctly regardless of its offset;
///   * because every record is a 256-multiple, a current pile stays 256-aligned
///     throughout under ATOMIC lock-free append (no exclusive lock needed), so
///     `cat a >> b` of two current piles is a valid merge AND the data stays
///     256-aligned for zero-copy GPU aliasing (CUDA/Metal `min_storage_buffer_offset_alignment`).
/// The reader still accepts the original V1 and unenveloped V3 records so
/// existing piles read byte-identical.
const MAGIC_MARKER_BLOB_V3: RawId = hex!("9C33EEB525065A62EAEC4BE43DCC355A");
const MAGIC_MARKER_BRANCH_V3: RawId = hex!("AC363D04AFE1AF17B39581B1E23021D7");
const MAGIC_MARKER_BRANCH_TOMBSTONE_V3: RawId = hex!("D0CBA0C8EAAB4C0C73121C3205671E4F");
/// Legacy physical marker pair used to encode [`WantStore`] state (minted
/// 2026-07-01 via `trible genid`). The historical names remain part of the
/// on-disk format, but their semantic payload is simply a per-handle LWW want:
/// the first marker asserts durable demand/cache interest and the second
/// retracts it. Reopening a pile reconstructs the current wanted set.
const MAGIC_MARKER_WEAK_PIN_V3: RawId = hex!("8F3EEFEDECD491F63F6EAAA5FD6F3D5E");
const MAGIC_MARKER_WEAK_UNPIN_V3: RawId = hex!("2D76662DFF0187EC36A8C90B12BB8B0D");
/// Typed want assertion and retraction markers, minted on 2026-08-13 with
/// `trible genid`.
///
/// Unlike the legacy weak-pin pair, these carry the complete canonical
/// [`WantRequest`] key and can therefore name blob fetches as well as exact
/// merge and derive receipt lookups.
const MAGIC_MARKER_WANT_ASSERT_V2: RawId = hex!("9A06797600FA90B8A8259B0ED029EC21");
const MAGIC_MARKER_WANT_RETRACT_V2: RawId = hex!("2D957A780A52E474F58A06D44D6FE46C");
/// Legacy V3 collection-record markers, minted on 2026-08-10 with
/// `trible genid`.
///
/// These physical records predate descriptor-handle collection identities.
/// They remain recognizable for safe replay and conservative rewriting, but
/// are inert: they are never reconstructed as current [`CollectionRecord`]s
/// and never enter [`CollectionStore`]. New writes use the V4 markers below.
const MAGIC_MARKER_COLLECTION_DEFINITION_V3: RawId = hex!("3BE108504E4F5242FB24AA72D6D94CE1");
const MAGIC_MARKER_COLLECTION_COMMIT_V3: RawId = hex!("BB758AA6F79FBFC4D1958592A8956777");
const MAGIC_MARKER_COLLECTION_MERGE_V3: RawId = hex!("CC0108AC1DF4F335AFA856A529C42BE9");
const MAGIC_MARKER_COLLECTION_DERIVE_V3: RawId = hex!("07ECF056F6F015D94389FFF21F851480");
/// Current collection-record markers, minted on 2026-08-11 with
/// `trible genid`.
///
/// V4 collection records carry 32-byte canonical descriptor handles directly.
/// There is deliberately no V4 definition record: descriptors are ordinary
/// `SimpleArchive` blobs named by those handles.
const MAGIC_MARKER_COLLECTION_COMMIT_V4: RawId = hex!("CBF2CF97D52A3486E16C12D70D397C66");
const MAGIC_MARKER_COLLECTION_MERGE_V4: RawId = hex!("9F5D028D4C423620D6957A5F726FA727");
const MAGIC_MARKER_COLLECTION_DERIVE_V4: RawId = hex!("ECFB2EE90ED8042244F7BAC704454BB9");
/// Grow-only signed collection-publication grant.
///
/// Unlike a collection-calculus record this is orthogonal low-level store
/// metadata and does not retain the named descriptor or any collection data.
/// The semantic kind was minted with `trible genid` on 2026-08-12.
const MAGIC_MARKER_COLLECTION_GOSSIP_V1: RawId = KIND_COLLECTION_GOSSIP.raw();
/// Retired local-cell record markers, minted on 2026-08-10 with `trible genid`.
///
/// These values remain private solely so old piles can be crossed at their
/// known 256-byte boundaries. They decode as opaque migration evidence and
/// never reconstruct operational state.
const MAGIC_MARKER_LOCAL_CELL_V3: RawId = hex!("24264FA9EE46A1ACC0E024AE69774B09");
const MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3: RawId = hex!("4FE372AE868D22A44DED7A60D579B651");

const BLOB_HEADER_LEN: usize = std::mem::size_of::<BlobHeader>();
const BLOB_ALIGNMENT: usize = BLOB_HEADER_LEN;
/// GPU storage-buffer binding-offset requirement (CUDA / Metal
/// `min_storage_buffer_offset_alignment`); a current blob record's data start lands here.
const GPU_DATA_ALIGNMENT: usize = 256;
/// Fixed header length and record alignment inherited from V3
/// (== GPU_DATA_ALIGNMENT). Current envelope headers retain this width; blob
/// data follows at `record_start + ENVELOPE_HEADER_LEN`.
const V3_HEADER_LEN: usize = 256;
const ENVELOPE_HEADER_LEN: usize = 256;
const ENVELOPE_BLOCK_LEN: usize = GPU_DATA_ALIGNMENT;
const ENVELOPE_HEADER_BLOCKS: u32 = 1;
/// Post-data padding that rounds a fixed-header record up to a 256-byte block.
fn block_post_pad(data_len: usize) -> usize {
    (ENVELOPE_BLOCK_LEN - (data_len % ENVELOPE_BLOCK_LEN)) % ENVELOPE_BLOCK_LEN
}

/// Largest single blob record we'll write with the concurrent `write_vectored`
/// fast path. Linux caps a single `writev` at `MAX_RW_COUNT` (`INT_MAX &
/// ~(PAGE_SIZE - 1)`, ~2 GiB) and macOS caps it at `INT_MAX`. Below this
/// threshold we rely on kernel atomicity and let concurrent writers hold a
/// shared lock. Above it we switch to an exclusive-lock fallback that
/// issues plain `write_all` calls — still append-only, still recoverable
/// via [`Pile::amputate`], just serialized with other writers for the
/// duration of the large append. The margin keeps us comfortably below
/// any platform's single-call ceiling.
const ATOMIC_WRITE_LIMIT: usize = 1 << 30;

/// Payloads at least this large may use BLAKE3's Rayon join strategy when the
/// current pool has more than one worker. Smaller payloads stay on the serial
/// one-shot path to avoid paying scheduling overhead for short validations.
#[cfg(any(feature = "parallel", test))]
const PARALLEL_BLAKE3_THRESHOLD: usize = 1 << 20;

/// Lazily-computed validation status of a blob record in the pile.
#[derive(Debug, Clone, Copy)]
pub enum ValidationState {
    /// The blob's hash matches its stored digest.
    Validated,
    /// The blob's hash does not match — the record is corrupt.
    Invalid,
}

#[cfg(feature = "parallel")]
fn should_parallelize_validation(len: usize) -> bool {
    len >= PARALLEL_BLAKE3_THRESHOLD && rayon::current_num_threads() > 1
}

#[derive(Debug, Clone, Copy)]
enum ValidationStrategy {
    /// Hash on the calling thread.
    Serial,
    /// For a sufficiently large first miss, use BLAKE3's Rayon join strategy.
    ParallelIfLarge,
}

fn classify_validation(
    computed: Inline<Hash<Blake3>>,
    expected: &Inline<Hash<Blake3>>,
) -> ValidationState {
    if computed == *expected {
        ValidationState::Validated
    } else {
        ValidationState::Invalid
    }
}

/// Computes the validation state of one immutable pile payload.
fn compute_validation_state(
    bytes: &Bytes,
    expected: &Inline<Hash<Blake3>>,
    strategy: ValidationStrategy,
) -> ValidationState {
    #[cfg(not(feature = "parallel"))]
    let _ = strategy;

    #[cfg(feature = "parallel")]
    if matches!(strategy, ValidationStrategy::ParallelIfLarge)
        && should_parallelize_validation(bytes.len())
    {
        let mut hasher = blake3::Hasher::new();
        hasher.update_rayon(bytes);
        let computed = Inline::new(*hasher.finalize().as_bytes());
        return classify_validation(computed, expected);
    }

    classify_validation(Hash::<Blake3>::digest(bytes), expected)
}

/// Sparse validation state shared by a pile and all reader snapshots derived
/// from it. Replay itself leaves this empty; entries appear only when a blob is
/// read or an on-disk duplicate challenges an earlier candidate.
///
/// Hashing happens outside the mutex. Concurrent first misses may duplicate
/// deterministic work, then converge through `or_insert` without blocking the
/// cache while Rayon executes.
#[derive(Debug, Clone, Default)]
struct ValidationCache {
    states: Arc<Mutex<HashMap<usize, ValidationState>>>,
}

impl ValidationCache {
    fn state(
        &self,
        record_offset: usize,
        bytes: &Bytes,
        expected: &Inline<Hash<Blake3>>,
        strategy: ValidationStrategy,
    ) -> ValidationState {
        if let Some(cached) = self
            .states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&record_offset)
            .copied()
        {
            return cached;
        }

        let computed = compute_validation_state(bytes, expected, strategy);
        *self
            .states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(record_offset)
            .or_insert(computed)
    }
}

#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    record_offset: usize,
}

impl IndexEntry {
    fn new(record_offset: usize) -> Self {
        Self { record_offset }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BranchHeader {
    magic_marker: RawId,
    branch_id: RawId,
    hash: RawInline,
}

// `BranchHeader` / `BranchTombstoneHeader` have no constructors; these structs
// exist only so the reader can decode legacy V1 records.

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BranchTombstoneHeader {
    magic_marker: RawId,
    branch_id: RawId,
    /// Reserved bytes to preserve 64 byte record alignment.
    reserved: RawInline,
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BlobHeader {
    magic_marker: RawId,
    timestamp: u64,
    length: u64,
    hash: RawInline,
}

impl BlobHeader {
    /// V1 blob constructor — retained only for the legacy-format backward-compat
    /// test (V1 blob records are otherwise read, never written).
    #[cfg(test)]
    fn new(timestamp: u64, length: u64, hash: Inline<Hash<Blake3>>) -> Self {
        Self {
            magic_marker: MAGIC_MARKER_BLOB,
            timestamp,
            length,
            hash: hash.raw,
        }
    }
}

/// V3 blob header — fixed 256 bytes. Same load-bearing fields as V1; the data
/// follows at `record_start + V3_HEADER_LEN` with no offset-derived pre-pad.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BlobHeaderV3 {
    magic_marker: RawId,
    timestamp: u64,
    length: u64,
    hash: RawInline,
    /// Pads the header to V3_HEADER_LEN (256), zeroed. NOT part of the content
    /// hash, so it never affects blob identity or dedup. Deliberately empty:
    /// genuinely useful per-record metadata belongs in tribles (keyed by the
    /// referencing attribute), and the encoding/schema must NOT live here — else
    /// identical bytes would fork into distinct blobs. Fill only when a concrete,
    /// content-independent need names itself.
    reserved: [u8; 192],
}

impl BlobHeaderV3 {
    #[cfg(test)]
    fn new(timestamp: u64, length: u64, hash: Inline<Hash<Blake3>>) -> Self {
        Self {
            magic_marker: MAGIC_MARKER_BLOB_V3,
            timestamp,
            length,
            hash: hash.raw,
            reserved: [0u8; 192],
        }
    }
}

/// V3 branch head — fixed 256 bytes (mirrors `BranchHeader` + reserved pad).
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BranchHeaderV3 {
    magic_marker: RawId,
    branch_id: RawId,
    hash: RawInline,
    reserved: [u8; 192],
}

/// V3 branch tombstone — fixed 256 bytes.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BranchTombstoneHeaderV3 {
    magic_marker: RawId,
    branch_id: RawId,
    reserved: [u8; 224],
}

/// V3 want marker using the legacy weak-pin encoding — fixed 256 bytes and
/// keyed by blob handle (no branch id).
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct WeakPinHeaderV3 {
    magic_marker: RawId,
    handle: RawInline,
    reserved: [u8; 208],
}

/// V3 want-retraction marker using the legacy weak-unpin encoding.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct WeakUnpinHeaderV3 {
    magic_marker: RawId,
    handle: RawInline,
    reserved: [u8; 208],
}

/// Legacy V3 collection definition: `(scope, representation, recipe)`.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionDefinitionHeaderV3 {
    magic_marker: RawId,
    scope: RawId,
    representation: RawId,
    recipe: RawId,
    reserved: [u8; 192],
}

/// Legacy V3 signed collection commit using a 16-byte intrinsic definition id.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionCommitHeaderV3 {
    magic_marker: RawId,
    collection: RawId,
    data: RawInline,
    metadata: RawInline,
    public_key: RawInline,
    signature_r: RawInline,
    signature_s: RawInline,
    reserved: [u8; 64],
}

/// Legacy V3 exact join equation using a 16-byte intrinsic definition id.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionMergeHeaderV3 {
    magic_marker: RawId,
    collection: RawId,
    low: RawInline,
    high: RawInline,
    result: RawInline,
    reserved: [u8; 128],
}

/// Legacy V3 mapping equation using 16-byte intrinsic definition ids.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionDeriveHeaderV3 {
    magic_marker: RawId,
    source: RawId,
    target: RawId,
    input: RawInline,
    output: RawInline,
    reserved: [u8; 144],
}

/// V4 signed collection commit. The complete 32-byte descriptor handle bound
/// by the V2 signature transcript is stored directly in the fixed header.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionCommitHeaderV4 {
    magic_marker: RawId,
    collection: RawInline,
    data: RawInline,
    metadata: RawInline,
    public_key: RawInline,
    signature_r: RawInline,
    signature_s: RawInline,
    reserved: [u8; 48],
}

/// V4 exact join equation. Inputs are stored in canonical digest order.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionMergeHeaderV4 {
    magic_marker: RawId,
    collection: RawInline,
    low: RawInline,
    high: RawInline,
    result: RawInline,
    reserved: [u8; 112],
}

/// V4 exact mapping equation between two descriptor-identified collections.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionDeriveHeaderV4 {
    magic_marker: RawId,
    source: RawInline,
    target: RawInline,
    input: RawInline,
    output: RawInline,
    reserved: [u8; 112],
}

/// Common prefix of every newly written pile record. `span_blocks` is a
/// canonical little-endian `u32` at bytes 32..36, includes the 256-byte header
/// itself, and is never zero.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct EnvelopePrefix {
    magic_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BlobHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    timestamp: [u8; 8],
    length: [u8; 8],
    hash: RawInline,
    reserved: [u8; 172],
}

impl BlobHeaderEnvelope {
    fn new(span_blocks: u32, timestamp: u64, length: u64, hash: Inline<Hash<Blake3>>) -> Self {
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_BLOB_V3,
            span_blocks: span_blocks.to_le_bytes(),
            timestamp: timestamp.to_le_bytes(),
            length: length.to_le_bytes(),
            hash: hash.raw,
            reserved: [0u8; 172],
        }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BranchHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    branch_id: RawId,
    hash: RawInline,
    reserved: [u8; 172],
}

impl BranchHeaderEnvelope {
    fn new(branch_id: Id, hash: Inline<Handle<SimpleArchive>>) -> Self {
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_BRANCH_V3,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            branch_id: *branch_id,
            hash: hash.raw,
            reserved: [0u8; 172],
        }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct BranchTombstoneHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    branch_id: RawId,
    reserved: [u8; 204],
}

impl BranchTombstoneHeaderEnvelope {
    fn new(branch_id: Id) -> Self {
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_BRANCH_TOMBSTONE_V3,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            branch_id: *branch_id,
            reserved: [0u8; 204],
        }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct WantHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    handle: RawInline,
    reserved: [u8; 188],
}

impl WantHeaderEnvelope {
    fn new(handle: Inline<Handle<UnknownBlob>>, asserted: bool) -> Self {
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: if asserted {
                MAGIC_MARKER_WEAK_PIN_V3
            } else {
                MAGIC_MARKER_WEAK_UNPIN_V3
            },
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            handle: handle.raw,
            reserved: [0u8; 188],
        }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct TypedWantHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    request_kind: u8,
    field_a: RawInline,
    field_b: RawInline,
    field_c: RawInline,
    reserved: [u8; 123],
}

impl TypedWantHeaderEnvelope {
    /// Construct the physical envelope used only by collection-operation
    /// wants. Blob wants deliberately retain the legacy weak-pin envelope so
    /// an older reader sees the same forgetful projection as a current one.
    fn new_operation(request: WantRequest, asserted: bool) -> Option<Self> {
        if matches!(request, WantRequest::Blob { .. }) {
            return None;
        }
        let bytes = request.to_bytes();
        let mut field_a = [0u8; 32];
        let mut field_b = [0u8; 32];
        let mut field_c = [0u8; 32];
        field_a.copy_from_slice(&bytes[1..33]);
        field_b.copy_from_slice(&bytes[33..65]);
        field_c.copy_from_slice(&bytes[65..97]);
        Some(Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: if asserted {
                MAGIC_MARKER_WANT_ASSERT_V2
            } else {
                MAGIC_MARKER_WANT_RETRACT_V2
            },
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            request_kind: bytes[0],
            field_a,
            field_b,
            field_c,
            reserved: [0u8; 123],
        })
    }

    fn request(&self) -> Result<WantRequest, crate::repo::WantRequestDecodeError> {
        let mut bytes = [0u8; WANT_REQUEST_BYTES_LEN];
        bytes[0] = self.request_kind;
        bytes[1..33].copy_from_slice(&self.field_a);
        bytes[33..65].copy_from_slice(&self.field_b);
        bytes[65..97].copy_from_slice(&self.field_c);
        WantRequest::from_bytes(bytes)
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionCommitHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    collection: RawInline,
    data: RawInline,
    metadata: RawInline,
    public_key: RawInline,
    signature_r: RawInline,
    signature_s: RawInline,
    reserved: [u8; 28],
}

impl CollectionCommitHeaderEnvelope {
    fn new(record: &CollectionCommit) -> Self {
        let (signature_r, signature_s) = record.signature();
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_COLLECTION_COMMIT_V4,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            collection: record.collection().raw,
            data: record.data().raw,
            metadata: record.metadata().raw,
            public_key: record.public_key().raw,
            signature_r: signature_r.raw,
            signature_s: signature_s.raw,
            reserved: [0u8; 28],
        }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionMergeHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    collection: RawInline,
    low: RawInline,
    high: RawInline,
    result: RawInline,
    reserved: [u8; 92],
}

impl CollectionMergeHeaderEnvelope {
    fn new(record: &CollectionMerge) -> Self {
        let (low, high) = record.inputs();
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_COLLECTION_MERGE_V4,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            collection: record.collection().raw,
            low: low.raw,
            high: high.raw,
            result: record.result().raw,
            reserved: [0u8; 92],
        }
    }
}

#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionDeriveHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    source: RawInline,
    target: RawInline,
    input: RawInline,
    output: RawInline,
    reserved: [u8; 92],
}

/// Signed grow-only publication grant for one author's commits in a
/// descriptor-identified collection.
#[derive(TryFromBytes, IntoBytes, Immutable, KnownLayout, Copy, Clone)]
#[repr(C)]
struct CollectionGossipHeaderEnvelope {
    envelope_marker: RawId,
    record_kind: RawId,
    span_blocks: [u8; 4],
    collection: RawInline,
    public_key: RawInline,
    signature_r: RawInline,
    signature_s: RawInline,
    reserved: [u8; 92],
}

impl CollectionGossipHeaderEnvelope {
    fn new(grant: &CollectionGossip) -> Self {
        let (signature_r, signature_s) = grant.signature();
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_COLLECTION_GOSSIP_V1,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            collection: grant.collection().raw,
            public_key: grant.public_key().raw,
            signature_r: signature_r.raw,
            signature_s: signature_s.raw,
            reserved: [0u8; 92],
        }
    }
}

impl CollectionDeriveHeaderEnvelope {
    fn new(record: &CollectionDerive) -> Self {
        let (input, output) = record.mapping();
        Self {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_COLLECTION_DERIVE_V4,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            source: record.source().raw,
            target: record.target().raw,
            input: input.raw,
            output: output.raw,
            reserved: [0u8; 92],
        }
    }
}

fn envelope_blocks_for_payload(data_len: usize) -> Option<u32> {
    let payload_blocks = data_len.checked_add(ENVELOPE_BLOCK_LEN - 1)? / ENVELOPE_BLOCK_LEN;
    u32::try_from(payload_blocks)
        .ok()?
        .checked_add(ENVELOPE_HEADER_BLOCKS)
}

fn collection_record_header(record: &CollectionRecord) -> [u8; ENVELOPE_HEADER_LEN] {
    let mut bytes = [0u8; ENVELOPE_HEADER_LEN];
    match record {
        CollectionRecord::Commit(record) => {
            bytes.copy_from_slice(CollectionCommitHeaderEnvelope::new(record).as_bytes())
        }
        CollectionRecord::Merge(record) => {
            bytes.copy_from_slice(CollectionMergeHeaderEnvelope::new(record).as_bytes())
        }
        CollectionRecord::Derive(record) => {
            bytes.copy_from_slice(CollectionDeriveHeaderEnvelope::new(record).as_bytes())
        }
    }
    bytes
}

// Compile-time guarantee that every legacy and current fixed header is exactly
// 256 bytes.
const _: () = {
    assert!(std::mem::size_of::<BlobHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<BranchHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<BranchTombstoneHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<WeakPinHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<WeakUnpinHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionDefinitionHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionCommitHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionMergeHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionDeriveHeaderV3>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionCommitHeaderV4>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionMergeHeaderV4>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionDeriveHeaderV4>() == V3_HEADER_LEN);
    assert!(std::mem::size_of::<EnvelopePrefix>() == 36);
    assert!(std::mem::size_of::<BlobHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<BranchHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<BranchTombstoneHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<WantHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<TypedWantHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionCommitHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionMergeHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionDeriveHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
    assert!(std::mem::size_of::<CollectionGossipHeaderEnvelope>() == ENVELOPE_HEADER_LEN);
};

/// A single record decoded from a pile file.
///
/// Yielded by [`PileRecords`], the raw record-level view of a pile. The
/// record's header starts at `offset` and the whole record (header + payload +
/// padding) spans `len` bytes, so `offset + len` is the offset of the next
/// record. This is the same decoder the [`Pile`] itself replays on open, so it
/// understands every record format ever written (V1, unenveloped V3/V4, and
/// the generic envelope alike).
#[derive(Debug, Clone, Copy)]
pub struct PileRecord {
    /// Byte offset of the record header within the pile file.
    pub offset: usize,
    /// Total on-disk length of the record (header + payload + padding).
    pub len: usize,
    /// The decoded record content.
    pub content: PileRecordContent,
}

/// Kind of one recognized but semantically inert legacy V3 collection header.
///
/// The raw header bytes remain available through [`PileRecords::bytes`] using
/// the enclosing [`PileRecord`]'s `offset` and `len`. This enum deliberately
/// exposes only the physical kind: V3's 16-byte definition identities and V1
/// commit transcripts must not be mistaken for current descriptor-handle
/// collection authority.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum LegacyCollectionRecordKindV3 {
    /// Legacy standalone collection definition.
    Definition,
    /// Legacy signed commit over a 16-byte definition id.
    Commit,
    /// Legacy merge equation over a 16-byte definition id.
    Merge,
    /// Legacy derive equation between 16-byte definition ids.
    Derive,
}

/// Decoded content of a [`PileRecord`], independent of on-disk format version.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum PileRecordContent {
    /// A blob record. The payload bytes live at
    /// `data_offset..data_offset + data_len` in the pile file; trailing
    /// alignment padding after the payload is not content and is not covered
    /// by `hash`.
    Blob {
        /// Insertion timestamp in milliseconds since the Unix epoch.
        timestamp: u64,
        /// Blake3 digest of the payload as recorded in the header.
        hash: Inline<Hash<Blake3>>,
        /// Byte offset of the payload within the pile file.
        data_offset: usize,
        /// Payload length in bytes (excluding padding).
        data_len: usize,
    },
    /// A branch head update.
    Branch {
        /// The branch being updated.
        branch_id: Id,
        /// The new head (a branch-metadata blob handle).
        head: Inline<Handle<SimpleArchive>>,
    },
    /// A branch tombstone (deletion marker).
    BranchTombstone {
        /// The branch being tombstoned.
        branch_id: Id,
    },
    /// A want assertion in the legacy weak-pin physical encoding.
    WeakPin {
        /// The wanted blob handle.
        handle: Inline<Handle<UnknownBlob>>,
    },
    /// A want retraction in the legacy weak-unpin physical encoding.
    WeakUnpin {
        /// The no-longer-wanted blob handle.
        handle: Inline<Handle<UnknownBlob>>,
    },
    /// A typed local request assertion.
    WantAssert {
        /// The exact request key being asserted.
        request: WantRequest,
    },
    /// A typed local request retraction.
    WantRetract {
        /// The exact request key being retracted.
        request: WantRequest,
    },
    /// One immutable current collection-algebra record. Three distinct V4
    /// magic markers share this typed raw-inspection surface.
    Collection {
        /// Canonically reconstructed semantic record.
        record: CollectionRecord,
    },
    /// One signed grow-only collection-publication grant.
    CollectionGossip {
        /// Structural evidence; consumers verify its signature before use.
        grant: CollectionGossip,
    },
    /// One recognized legacy V3 collection header.
    ///
    /// Replay treats this as inert physical evidence. It is excluded from
    /// [`CollectionStore`] but retained byte-for-byte by ordinary pile rewrite.
    LegacyCollectionV3 {
        /// The historical physical record kind.
        kind: LegacyCollectionRecordKindV3,
    },
    /// A record whose semantic kind is not active in this reader. This covers
    /// structurally valid unknown generic envelopes and the two retired,
    /// fixed-width unenveloped local-cell markers. Replay deliberately
    /// projects it away, while [`PileRecords`] exposes its exact offset and
    /// length so raw migration tooling can preserve the bytes.
    Opaque {
        /// Inert semantic record-kind id.
        kind: RawId,
    },
}

fn decode_enveloped_record(bytes: &[u8], offset: usize) -> Result<PileRecord, ReadError> {
    let corrupt = || ReadError::CorruptPile {
        valid_length: offset,
    };
    let (prefix, _) = EnvelopePrefix::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
    let declared_blocks = u32::from_le_bytes(prefix.span_blocks);
    if prefix.magic_marker != MAGIC_MARKER_ENVELOPE || declared_blocks == 0 {
        return Err(corrupt());
    }
    let span_blocks = usize::try_from(declared_blocks).map_err(|_| corrupt())?;
    let len = span_blocks
        .checked_mul(ENVELOPE_BLOCK_LEN)
        .ok_or_else(corrupt)?;
    if len < ENVELOPE_HEADER_LEN || bytes.len() < len {
        return Err(corrupt());
    }

    let fixed_header = || {
        if declared_blocks == ENVELOPE_HEADER_BLOCKS {
            Ok(())
        } else {
            Err(corrupt())
        }
    };
    match prefix.record_kind {
        MAGIC_MARKER_BLOB_V3 => {
            let (header, _) =
                BlobHeaderEnvelope::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            let data_len =
                usize::try_from(u64::from_le_bytes(header.length)).map_err(|_| corrupt())?;
            let expected_blocks = envelope_blocks_for_payload(data_len).ok_or_else(corrupt)?;
            if declared_blocks != expected_blocks {
                return Err(corrupt());
            }
            let data_offset = offset
                .checked_add(ENVELOPE_HEADER_LEN)
                .ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Blob {
                    timestamp: u64::from_le_bytes(header.timestamp),
                    hash: Inline::new(header.hash),
                    data_offset,
                    data_len,
                },
            })
        }
        MAGIC_MARKER_BRANCH_V3 => {
            fixed_header()?;
            let (header, _) =
                BranchHeaderEnvelope::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            let branch_id = Id::new(header.branch_id).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Branch {
                    branch_id,
                    head: Inline::<Hash<Blake3>>::new(header.hash).into(),
                },
            })
        }
        MAGIC_MARKER_BRANCH_TOMBSTONE_V3 => {
            fixed_header()?;
            let (header, _) = BranchTombstoneHeaderEnvelope::try_read_from_prefix(bytes)
                .map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            let branch_id = Id::new(header.branch_id).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::BranchTombstone { branch_id },
            })
        }
        MAGIC_MARKER_WEAK_PIN_V3 | MAGIC_MARKER_WEAK_UNPIN_V3 => {
            fixed_header()?;
            let (header, _) =
                WantHeaderEnvelope::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            let handle = Inline::new(header.handle);
            let content = if prefix.record_kind == MAGIC_MARKER_WEAK_PIN_V3 {
                PileRecordContent::WeakPin { handle }
            } else {
                PileRecordContent::WeakUnpin { handle }
            };
            Ok(PileRecord {
                offset,
                len,
                content,
            })
        }
        MAGIC_MARKER_WANT_ASSERT_V2 | MAGIC_MARKER_WANT_RETRACT_V2 => {
            fixed_header()?;
            let (header, _) =
                TypedWantHeaderEnvelope::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            let request = header.request().map_err(|_| corrupt())?;
            // Blob wants must use the legacy weak-pin physical marker pair.
            // Accepting the typed representation here would let a newer
            // writer construct a history whose Blob LWW projection differs
            // between current readers and older readers that skip this
            // unknown record kind.
            if matches!(request, WantRequest::Blob { .. }) {
                return Err(corrupt());
            }
            let content = if prefix.record_kind == MAGIC_MARKER_WANT_ASSERT_V2 {
                PileRecordContent::WantAssert { request }
            } else {
                PileRecordContent::WantRetract { request }
            };
            Ok(PileRecord {
                offset,
                len,
                content,
            })
        }
        MAGIC_MARKER_COLLECTION_COMMIT_V4 => {
            fixed_header()?;
            let (header, _) = CollectionCommitHeaderEnvelope::try_read_from_prefix(bytes)
                .map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Collection {
                    record: CollectionRecord::Commit(CollectionCommit::from_parts(
                        Inline::new(header.collection),
                        Inline::new(header.data),
                        Inline::new(header.metadata),
                        Inline::<ED25519PublicKey>::new(header.public_key),
                        Inline::<ED25519RComponent>::new(header.signature_r),
                        Inline::<ED25519SComponent>::new(header.signature_s),
                    )),
                },
            })
        }
        MAGIC_MARKER_COLLECTION_MERGE_V4 => {
            fixed_header()?;
            let (header, _) = CollectionMergeHeaderEnvelope::try_read_from_prefix(bytes)
                .map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) || header.high < header.low {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Collection {
                    record: CollectionRecord::Merge(CollectionMerge::new(
                        Inline::new(header.collection),
                        Inline::new(header.low),
                        Inline::new(header.high),
                        Inline::new(header.result),
                    )),
                },
            })
        }
        MAGIC_MARKER_COLLECTION_DERIVE_V4 => {
            fixed_header()?;
            let (header, _) = CollectionDeriveHeaderEnvelope::try_read_from_prefix(bytes)
                .map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Collection {
                    record: CollectionRecord::Derive(CollectionDerive::new(
                        Inline::new(header.source),
                        Inline::new(header.target),
                        Inline::new(header.input),
                        Inline::new(header.output),
                    )),
                },
            })
        }
        MAGIC_MARKER_COLLECTION_GOSSIP_V1 => {
            fixed_header()?;
            let (header, _) = CollectionGossipHeaderEnvelope::try_read_from_prefix(bytes)
                .map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::CollectionGossip {
                    grant: CollectionGossip::from_parts(
                        Inline::new(header.collection),
                        Inline::<ED25519PublicKey>::new(header.public_key),
                        Inline::<ED25519RComponent>::new(header.signature_r),
                        Inline::<ED25519SComponent>::new(header.signature_s),
                    ),
                },
            })
        }
        kind @ (MAGIC_MARKER_LOCAL_CELL_V3 | MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3) => {
            fixed_header()?;
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Opaque { kind },
            })
        }
        kind => Ok(PileRecord {
            offset,
            len,
            content: PileRecordContent::Opaque { kind },
        }),
    }
}

/// Decodes the record starting at the beginning of `bytes`, which is the pile
/// file's content from `offset` onward. This is the single source of truth for
/// record parsing: [`Pile::refresh`]/[`Pile::amputate`] replay records through
/// it, and [`PileRecords`] exposes it for raw inspection. An unknown legacy
/// marker yields [`ReadError::UnsupportedRecord`] because this reader cannot
/// know its length. The two retired fixed-width local-cell markers are
/// recognized as opaque migration evidence. An unknown kind inside the generic
/// envelope has an exact span and yields [`PileRecordContent::Opaque`]. A
/// truncated record yields
/// [`ReadError::CorruptPile`] pointing at `offset`.
fn decode_record(bytes: &[u8], offset: usize) -> Result<PileRecord, ReadError> {
    let corrupt = || ReadError::CorruptPile {
        valid_length: offset,
    };
    if bytes.len() < 16 {
        return Err(corrupt());
    }
    let magic: RawId = bytes[0..16].try_into().unwrap();
    if magic == MAGIC_MARKER_ENVELOPE {
        return decode_enveloped_record(bytes, offset);
    }
    match magic {
        MAGIC_MARKER_BLOB => {
            let (header, _) = BlobHeader::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            let data_len = header.length as usize;
            let pad = padding_for_blob(data_len);
            let len = BLOB_HEADER_LEN
                .checked_add(data_len)
                .and_then(|l| l.checked_add(pad))
                .ok_or_else(corrupt)?;
            if bytes.len() < len {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Blob {
                    timestamp: header.timestamp,
                    hash: Inline::new(header.hash),
                    data_offset: offset + BLOB_HEADER_LEN,
                    data_len,
                },
            })
        }
        MAGIC_MARKER_BRANCH => {
            let (header, _) = BranchHeader::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            let branch_id = Id::new(header.branch_id).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: std::mem::size_of::<BranchHeader>(),
                content: PileRecordContent::Branch {
                    branch_id,
                    head: Inline::<Hash<Blake3>>::new(header.hash).into(),
                },
            })
        }
        MAGIC_MARKER_BRANCH_TOMBSTONE => {
            let (header, _) =
                BranchTombstoneHeader::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            let branch_id = Id::new(header.branch_id).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: std::mem::size_of::<BranchTombstoneHeader>(),
                content: PileRecordContent::BranchTombstone { branch_id },
            })
        }
        MAGIC_MARKER_BLOB_V3 => {
            // Fixed 256-byte header; data at a constant `record_start +
            // V3_HEADER_LEN` (no offset-derived pad — position-independent),
            // record padded to a 256-byte multiple.
            let (header, _) = BlobHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            let data_len = header.length as usize;
            let post_pad = block_post_pad(data_len);
            let len = V3_HEADER_LEN
                .checked_add(data_len)
                .and_then(|l| l.checked_add(post_pad))
                .ok_or_else(corrupt)?;
            if bytes.len() < len {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len,
                content: PileRecordContent::Blob {
                    timestamp: header.timestamp,
                    hash: Inline::new(header.hash),
                    data_offset: offset + V3_HEADER_LEN,
                    data_len,
                },
            })
        }
        MAGIC_MARKER_BRANCH_V3 => {
            let (header, _) = BranchHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            let branch_id = Id::new(header.branch_id).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::Branch {
                    branch_id,
                    head: Inline::<Hash<Blake3>>::new(header.hash).into(),
                },
            })
        }
        MAGIC_MARKER_BRANCH_TOMBSTONE_V3 => {
            let (header, _) =
                BranchTombstoneHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            let branch_id = Id::new(header.branch_id).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::BranchTombstone { branch_id },
            })
        }
        MAGIC_MARKER_LOCAL_CELL_V3 | MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3 => {
            if bytes.len() < V3_HEADER_LEN {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::Opaque { kind: magic },
            })
        }
        MAGIC_MARKER_WEAK_PIN_V3 => {
            let (header, _) =
                WeakPinHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::WeakPin {
                    handle: Inline::new(header.handle),
                },
            })
        }
        MAGIC_MARKER_WEAK_UNPIN_V3 => {
            let (header, _) =
                WeakUnpinHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::WeakUnpin {
                    handle: Inline::new(header.handle),
                },
            })
        }
        MAGIC_MARKER_COLLECTION_DEFINITION_V3 => {
            let (header, _) =
                CollectionDefinitionHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Id::new(header.scope).ok_or_else(corrupt)?;
            Id::new(header.representation).ok_or_else(corrupt)?;
            Id::new(header.recipe).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::LegacyCollectionV3 {
                    kind: LegacyCollectionRecordKindV3::Definition,
                },
            })
        }
        MAGIC_MARKER_COLLECTION_COMMIT_V3 => {
            let (header, _) =
                CollectionCommitHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Id::new(header.collection).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::LegacyCollectionV3 {
                    kind: LegacyCollectionRecordKindV3::Commit,
                },
            })
        }
        MAGIC_MARKER_COLLECTION_MERGE_V3 => {
            let (header, _) =
                CollectionMergeHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) || header.high < header.low {
                return Err(corrupt());
            }
            Id::new(header.collection).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::LegacyCollectionV3 {
                    kind: LegacyCollectionRecordKindV3::Merge,
                },
            })
        }
        MAGIC_MARKER_COLLECTION_DERIVE_V3 => {
            let (header, _) =
                CollectionDeriveHeaderV3::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Id::new(header.source).ok_or_else(corrupt)?;
            Id::new(header.target).ok_or_else(corrupt)?;
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::LegacyCollectionV3 {
                    kind: LegacyCollectionRecordKindV3::Derive,
                },
            })
        }
        MAGIC_MARKER_COLLECTION_COMMIT_V4 => {
            let (header, _) =
                CollectionCommitHeaderV4::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::Collection {
                    record: CollectionRecord::Commit(CollectionCommit::from_parts(
                        Inline::new(header.collection),
                        Inline::new(header.data),
                        Inline::new(header.metadata),
                        Inline::<ED25519PublicKey>::new(header.public_key),
                        Inline::<ED25519RComponent>::new(header.signature_r),
                        Inline::<ED25519SComponent>::new(header.signature_s),
                    )),
                },
            })
        }
        MAGIC_MARKER_COLLECTION_MERGE_V4 => {
            let (header, _) =
                CollectionMergeHeaderV4::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) || header.high < header.low {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::Collection {
                    record: CollectionRecord::Merge(CollectionMerge::new(
                        Inline::new(header.collection),
                        Inline::new(header.low),
                        Inline::new(header.high),
                        Inline::new(header.result),
                    )),
                },
            })
        }
        MAGIC_MARKER_COLLECTION_DERIVE_V4 => {
            let (header, _) =
                CollectionDeriveHeaderV4::try_read_from_prefix(bytes).map_err(|_| corrupt())?;
            if header.reserved.iter().any(|byte| *byte != 0) {
                return Err(corrupt());
            }
            Ok(PileRecord {
                offset,
                len: V3_HEADER_LEN,
                content: PileRecordContent::Collection {
                    record: CollectionRecord::Derive(CollectionDerive::new(
                        Inline::new(header.source),
                        Inline::new(header.target),
                        Inline::new(header.input),
                        Inline::new(header.output),
                    )),
                },
            })
        }
        _ => Err(ReadError::UnsupportedRecord {
            offset,
            marker: magic,
        }),
    }
}

/// Header metadata recovered from the immutable record named by one in-memory
/// blob index entry.
struct IndexedBlobHeader {
    timestamp: u64,
    data_offset: usize,
    data_len: usize,
}

/// Resolves an offset-only index entry through the canonical record decoder
/// without touching or hashing its payload.
fn indexed_blob_header(
    mmap: &Arc<MmapRaw>,
    covered_len: usize,
    entry: IndexEntry,
    expected: &Inline<Hash<Blake3>>,
) -> IndexedBlobHeader {
    assert!(
        entry.record_offset < covered_len,
        "blob index offset lies outside its accepted pile prefix"
    );
    assert!(
        covered_len <= mmap.len(),
        "accepted pile prefix lies outside its mapping"
    );
    let record_bytes = unsafe {
        slice_from_raw_parts(
            mmap.as_ptr().add(entry.record_offset),
            covered_len - entry.record_offset,
        )
        .as_ref()
        .unwrap()
    };
    let record = decode_record(record_bytes, entry.record_offset)
        .expect("indexed blob record changed below the accepted pile prefix");
    let PileRecordContent::Blob {
        timestamp,
        hash,
        data_offset,
        data_len,
    } = record.content
    else {
        panic!("blob index offset no longer names a blob record");
    };
    assert_eq!(
        hash, *expected,
        "blob index key no longer matches its record header"
    );
    IndexedBlobHeader {
        timestamp,
        data_offset,
        data_len,
    }
}

/// Payload and metadata recovered from the immutable record named by one
/// in-memory blob index entry.
struct IndexedBlobRecord {
    bytes: Bytes,
    #[cfg(test)]
    payload_offset: usize,
    timestamp: u64,
}

/// Resolves an offset-only index entry through the canonical record decoder.
///
/// Entries are created only after `decode_record` accepted the complete record,
/// and `covered_len` is the exact accepted prefix captured with this mapping.
/// A failure here therefore means bytes below an applied boundary changed,
/// which violates Pile's append-only safety contract.
fn indexed_blob_record(
    mmap: &Arc<MmapRaw>,
    covered_len: usize,
    entry: IndexEntry,
    expected: &Inline<Hash<Blake3>>,
) -> IndexedBlobRecord {
    let header = indexed_blob_header(mmap, covered_len, entry, expected);
    let bytes = unsafe {
        let slice = slice_from_raw_parts(mmap.as_ptr().add(header.data_offset), header.data_len)
            .as_ref()
            .unwrap();
        Bytes::from_raw_parts(slice, mmap.clone())
    };
    IndexedBlobRecord {
        bytes,
        #[cfg(test)]
        payload_offset: header.data_offset,
        timestamp: header.timestamp,
    }
}

/// Iterator over the raw records of a pile file, in log order.
///
/// This is the record-level view of the append-only log: every blob, branch
/// update, branch tombstone, and legacy-encoded want marker ever appended,
/// including records that later ones supersede (superseded branch heads,
/// tombstoned branches, retracted wants). It shares its decoder with the [`Pile`]
/// replay path, so V1, unenveloped V3/V4, and generic-envelope records are
/// understood; tools that need
/// history or forensics (reflogs, consolidation, corruption reports) should
/// consume this instead of hand-rolling a parser.
///
/// Unknown envelope kinds are yielded as [`PileRecordContent::Opaque`] with a
/// known boundary. The iterator yields an error and ends for an unknown
/// unenveloped marker or a truncated record: the former surfaces as
/// [`ReadError::UnsupportedRecord`] and the latter as
/// [`ReadError::CorruptPile`].
#[derive(Debug)]
pub struct PileRecords {
    bytes: Bytes,
    offset: usize,
    failed: bool,
}

impl PileRecords {
    /// Opens the pile file at `path` read-only and returns an iterator over
    /// its records. No index is built and nothing is validated eagerly; blob
    /// payloads are not hashed.
    pub fn open(path: &Path) -> Result<Self, ReadError> {
        let file = File::open(path)?;
        let length = file.metadata()?.len();
        let bytes = if length == 0 {
            // Mapping a zero-length file is an error on most platforms; an
            // empty pile simply has no records.
            Bytes::empty()
        } else {
            // SAFETY: the pile file is append-only by contract; existing
            // bytes are never mutated, so the mapping stays valid.
            unsafe { Bytes::map_file(&file)? }
        };
        Ok(Self {
            bytes,
            offset: 0,
            failed: false,
        })
    }

    /// The raw bytes of the pile file, e.g. to inspect a blob payload at the
    /// `data_offset`/`data_len` reported by [`PileRecordContent::Blob`].
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }
}

impl Iterator for PileRecords {
    type Item = Result<PileRecord, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.offset >= self.bytes.len() {
            return None;
        }
        match decode_record(&self.bytes[self.offset..], self.offset) {
            Ok(record) => {
                self.offset += record.len;
                Some(Ok(record))
            }
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}

#[derive(Debug)]
enum Applied {
    Blob { hash: Inline<Hash<Blake3>> },
    Branch { id: Id, hash: Inline<Hash<Blake3>> },
    BranchTombstone { id: Id },
    WantAssert { request: WantRequest },
    WantRetract { request: WantRequest },
    Collection { id: Id },
    CollectionGossip { grant: CollectionGossip },
    LegacyCollectionV3,
    Opaque,
}

#[derive(Debug)]
/// A grow-only collection of blobs and pin heads backed by a single file on disk.
///
/// Branch updates do not verify that referenced blobs exist in the pile, allowing the
/// pile to operate as a head-only store when blob data lives elsewhere.
///
/// [`Pile::refresh`] aborts immediately if the underlying file shrinks below
/// data that has already been applied, preventing undefined behavior from
/// dangling [`Bytes`] handles.
pub struct Pile {
    file: File,
    mmap: Arc<MmapRaw>,
    /// Whether this handle has appended or truncated bytes since its last
    /// successful durability barrier. Refreshing bytes written by another
    /// handle does not make this handle responsible for flushing them.
    dirty: bool,
    blobs: PATCH<32, IdentitySchema, IndexEntry>,
    validations: ValidationCache,
    branches: PATCH<16, IdentitySchema, Inline<Handle<SimpleArchive>>>,
    /// Immutable collection records keyed by their intrinsic entity id.
    /// `BTreeMap` makes enumeration independent of append/cat order.
    collection_records: BTreeMap<Id, CollectionRecord>,
    /// Immutable signed collection-publication grants. This is a grow-only
    /// set and contributes no blob-retention roots.
    collection_gossips: BTreeSet<CollectionGossip>,
    /// Exact byte-distinct legacy V3 collection headers accepted during replay.
    /// They remain inert but are conservatively carried through retained
    /// rewrites so an explicit future migration still has its source evidence.
    legacy_collection_headers: BTreeSet<[u8; V3_HEADER_LEN]>,
    /// Number of structurally valid records projected as opaque. This includes
    /// unknown generic-envelope kinds and retired local-cell encodings.
    /// Destructive physical rewrites refuse while this is nonzero.
    opaque_records: usize,
    /// LWW-resolved typed request set. Legacy weak-pin records project to blob
    /// requests; current records carry the complete 97-byte canonical key.
    /// Log-order application makes the last record for one exact key win.
    wants: PATCH<WANT_REQUEST_BYTES_LEN, IdentitySchema>,
    /// Length of the file that has been validated and applied.
    ///
    /// Offsets below this value are guaranteed valid; corruption detection
    /// only operates on the un-applied tail beyond this boundary.
    applied_length: usize,
}

fn padding_for_blob(blob_size: usize) -> usize {
    (BLOB_ALIGNMENT - ((BLOB_HEADER_LEN + blob_size) % BLOB_ALIGNMENT)) % BLOB_ALIGNMENT
}

#[derive(Debug, Clone)]
/// Read-only handle referencing a [`Pile`].
///
/// Multiple `PileReader` instances can coexist and provide concurrent access to
/// the same underlying pile data.
pub struct PileReader {
    mmap: Arc<MmapRaw>,
    covered_len: usize,
    blobs: PATCH<32, IdentitySchema, IndexEntry>,
    validations: ValidationCache,
}

impl PartialEq for PileReader {
    fn eq(&self, other: &Self) -> bool {
        self.blobs == other.blobs
    }
}

impl Eq for PileReader {}

impl PileReader {
    fn new(
        mmap: Arc<MmapRaw>,
        covered_len: usize,
        blobs: PATCH<32, IdentitySchema, IndexEntry>,
        validations: ValidationCache,
    ) -> Self {
        Self {
            mmap,
            covered_len,
            blobs,
            validations,
        }
    }

    /// Returns an iterator over all blobs currently stored in the pile.
    ///
    /// This creates an owned snapshot of the current keys/indices so the
    /// returned iterator does not borrow from the underlying PATCH.
    pub fn iter(&self) -> PileBlobStoreIter {
        // PATCH is persistent, so these clones capture one immutable point in
        // time while later pile refreshes continue independently.
        let for_iter = self.blobs.clone();
        let lookup = for_iter.clone();
        let inner = for_iter.into_iter();
        PileBlobStoreIter {
            mmap: self.mmap.clone(),
            covered_len: self.covered_len,
            inner,
            lookup,
            validations: self.validations.clone(),
        }
    }

    /// Returns unvalidated listing metadata for a resident blob.
    ///
    /// This reads only the already-accepted pile record header. Callers that
    /// consume the payload must still use [`BlobStoreGet::get`].
    pub(crate) fn blob_info(&self, handle: Inline<Handle<UnknownBlob>>) -> Option<super::BlobInfo> {
        let hash: &Inline<Hash<Blake3>> = handle.as_transmute();
        let entry = *self.blobs.get(&hash.raw)?;
        let header = indexed_blob_header(&self.mmap, self.covered_len, entry, hash);
        Some(super::BlobInfo {
            handle,
            length: header.data_len as u64,
        })
    }

    // metadata moved into BlobStoreMeta impl below
}

impl BlobStoreGet for PileReader {
    type GetError<E: Error + Send + Sync + 'static> = GetBlobError<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let hash: &Inline<Hash<Blake3>> = handle.as_transmute();
        let Some(entry) = self.blobs.get(&hash.raw) else {
            return Err(GetBlobError::BlobNotFound);
        };
        let entry = *entry;
        let record = indexed_blob_record(&self.mmap, self.covered_len, entry, hash);
        let state = self.validations.state(
            entry.record_offset,
            &record.bytes,
            hash,
            ValidationStrategy::ParallelIfLarge,
        );
        match state {
            ValidationState::Validated => {
                // The handle is what we just validated against — reuse
                // it to skip Blake3 recomputation in Blob::new.
                let blob: Blob<S> = Blob::with_handle(record.bytes.clone(), handle);
                match blob.try_from_blob() {
                    Ok(value) => Ok(value),
                    Err(e) => Err(GetBlobError::ConversionError(e)),
                }
            }
            ValidationState::Invalid => Err(GetBlobError::ValidationError(record.bytes)),
        }
    }
}

impl super::BlobChildren for PileReader {}

impl BlobStore for Pile {
    type Reader = PileReader;
    type ReaderError = ReadError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        self.refresh()?;
        Ok(PileReader::new(
            self.mmap.clone(),
            self.applied_length,
            self.blobs.clone(),
            self.validations.clone(),
        ))
    }
}

/// Error returned when opening or refreshing a [`Pile`].
#[derive(Debug)]
pub enum ReadError {
    /// Underlying I/O failure.
    IoError(std::io::Error),
    /// The pile contains corrupted data starting at `valid_length`.
    CorruptPile {
        /// Byte offset where the first malformed or truncated known record was
        /// found.
        valid_length: usize,
    },
    /// The pile contains a complete unenveloped magic marker this reader does
    /// not know.
    ///
    /// The marker may name a record introduced by a newer binary. Its length
    /// is unknowable to this reader, so it is unsafe to skip or amputate it.
    UnsupportedRecord {
        /// Byte offset where the unsupported record begins.
        offset: usize,
        /// Unrecognized 16-byte record marker.
        marker: RawId,
    },
    /// The pile file exceeds the addressable range.
    FileTooLarge {
        /// Actual file length.
        length: usize,
    },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::IoError(err) => write!(f, "IO error: {err}"),
            ReadError::CorruptPile { valid_length } => {
                write!(f, "Corrupt pile at byte {valid_length}")
            }
            ReadError::UnsupportedRecord { offset, marker } => write!(
                f,
                "Unsupported pile record marker {} at byte {offset}; a newer reader may be required",
                hex::encode_upper(marker)
            ),
            ReadError::FileTooLarge { length } => {
                write!(f, "Pile of length {length} exceeds supported size")
            }
        }
    }
}
impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(err) => Some(err),
            Self::CorruptPile { .. }
            | Self::UnsupportedRecord { .. }
            | Self::FileTooLarge { .. } => None,
        }
    }
}

impl From<std::io::Error> for ReadError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<ReadError> for std::io::Error {
    fn from(err: ReadError) -> Self {
        match err {
            ReadError::IoError(e) => e,
            ReadError::CorruptPile { valid_length } => {
                std::io::Error::other(format!("corrupt pile at byte {valid_length}"))
            }
            ReadError::UnsupportedRecord { offset, marker } => std::io::Error::other(format!(
                "unsupported pile record marker {} at byte {offset}; a newer reader may be required",
                hex::encode_upper(marker)
            )),
            ReadError::FileTooLarge { length } => {
                std::io::Error::other(format!("pile length {length} exceeds supported size"))
            }
        }
    }
}

/// Error returned when appending a blob to a [`Pile`].
#[derive(Debug)]
pub enum InsertError {
    /// Underlying I/O failure.
    IoError(std::io::Error),
    /// System clock error when timestamping the record.
    TimeError(std::time::SystemTimeError),
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InsertError::IoError(err) => write!(f, "IO error: {err}"),
            InsertError::TimeError(err) => write!(f, "system time error: {err}"),
        }
    }
}
impl std::error::Error for InsertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(err) => Some(err),
            Self::TimeError(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for InsertError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<std::time::SystemTimeError> for InsertError {
    fn from(err: std::time::SystemTimeError) -> Self {
        Self::TimeError(err)
    }
}

impl From<ReadError> for InsertError {
    fn from(err: ReadError) -> Self {
        Self::IoError(err.into())
    }
}

/// Error returned when appending a pin-head update or want marker to a
/// [`Pile`].
pub enum PileWriteError {
    /// Underlying I/O failure.
    IoError(std::io::Error),
}

impl std::error::Error for PileWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(err) => Some(err),
        }
    }
}

impl std::fmt::Debug for PileWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PileWriteError::IoError(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl std::fmt::Display for PileWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PileWriteError::IoError(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl From<std::io::Error> for PileWriteError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<ReadError> for PileWriteError {
    fn from(err: ReadError) -> Self {
        Self::IoError(err.into())
    }
}

/// Failure while appending an immutable native collection record.
#[derive(Debug)]
pub enum CollectionInsertError {
    /// Existing pile state could not be refreshed or decoded.
    Read(ReadError),
    /// The fixed record could not be appended or the file lock released.
    Io(std::io::Error),
    /// The intrinsic id already names different canonical fields.
    IdCollision { id: Id },
    /// Readback observed a record other than the exclusively appended one.
    UnexpectedReadback,
}

impl std::fmt::Display for CollectionInsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => write!(f, "failed to refresh collection records: {error}"),
            Self::Io(error) => write!(f, "failed to append collection record: {error}"),
            Self::IdCollision { id } => {
                write!(f, "collection record id {id:X} names different fields")
            }
            Self::UnexpectedReadback => {
                f.write_str("collection append read back an unexpected pile record")
            }
        }
    }
}

impl Error for CollectionInsertError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::IdCollision { .. } | Self::UnexpectedReadback => None,
        }
    }
}

impl From<ReadError> for CollectionInsertError {
    fn from(error: ReadError) -> Self {
        Self::Read(error)
    }
}

impl From<std::io::Error> for CollectionInsertError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Error returned when retrieving a blob from a [`Pile`].
#[derive(Debug)]
pub enum GetBlobError<E: Error> {
    /// No blob with the given handle exists in the pile.
    BlobNotFound,
    /// The blob's hash does not match its stored digest.
    ValidationError(Bytes),
    /// The blob was found and valid but deserialization failed.
    ConversionError(E),
}

impl<E: Error> std::fmt::Display for GetBlobError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GetBlobError::BlobNotFound => write!(f, "Blob not found"),
            GetBlobError::ConversionError(err) => write!(f, "Conversion error: {err}"),
            GetBlobError::ValidationError(_) => write!(f, "Validation error"),
        }
    }
}

impl<E: Error> std::error::Error for GetBlobError<E> {}

/// Error returned by [`Pile::flush`] and [`Pile::close`].
#[derive(Debug)]
pub enum FlushError {
    /// Underlying I/O failure.
    IoError(std::io::Error),
}

impl From<std::io::Error> for FlushError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl std::fmt::Display for FlushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlushError::IoError(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl std::error::Error for FlushError {}

impl Pile {
    /// Opens an existing pile file. Returns an error if the file does not
    /// exist — create the file first with [`std::fs::File::create`] or
    /// equivalent if you need a fresh pile.
    ///
    /// The returned pile has no in-memory index; callers should invoke
    /// [`Self::refresh`] to load existing data. After a crash left a torn
    /// tail, [`Self::amputate`] loads and **truncates the file at the first
    /// malformed record** — a destructive last resort, not an open path.
    /// Complete opaque envelopes are crossed; unknown unenveloped markers are
    /// refused without truncation.
    pub fn open(path: &Path) -> Result<Self, ReadError> {
        let file = OpenOptions::new().read(true).append(true).open(path)?;
        let length_u64 = file.metadata()?.len();
        let length = usize::try_from(length_u64)
            .map_err(|_| ReadError::FileTooLarge { length: usize::MAX })?;
        let page_size = page_size::get();
        let base_size = page_size * 1024;
        let mapped_size = base_size.max(
            length
                .checked_next_power_of_two()
                .ok_or(ReadError::FileTooLarge { length })?,
        );

        let mmap = MmapOptions::new()
            .len(mapped_size)
            .map_raw_read_only(&file)?;
        let mmap = Arc::new(mmap);

        Ok(Self {
            file,
            mmap,
            dirty: false,
            blobs: PATCH::<32, IdentitySchema, IndexEntry>::new(),
            validations: ValidationCache::default(),
            branches: PATCH::<16, IdentitySchema, Inline<Handle<SimpleArchive>>>::new(),
            collection_records: BTreeMap::new(),
            collection_gossips: BTreeSet::new(),
            legacy_collection_headers: BTreeSet::new(),
            opaque_records: 0,
            wants: PATCH::<WANT_REQUEST_BYTES_LEN, IdentitySchema>::new(),
            applied_length: 0,
        })
    }

    fn ensure_mapped(&mut self, file_len: usize) -> Result<(), ReadError> {
        if file_len <= self.mmap.len() {
            return Ok(());
        }
        let mapped_size = file_len
            .checked_next_power_of_two()
            .ok_or(ReadError::FileTooLarge { length: file_len })?;
        self.mmap = Arc::new(
            MmapOptions::new()
                .len(mapped_size)
                .map_raw_read_only(&self.file)?,
        );
        Ok(())
    }

    /// Refreshes in-memory state from newly appended records.
    ///
    /// Aborts immediately if the underlying pile file has shrunk below the
    /// portion already applied since the last refresh. Truncating validated data
    /// would invalidate existing `Bytes` handles and continuing would result in
    /// undefined behavior.
    ///
    /// This acquires a shared file lock to avoid racing with [`Self::amputate`],
    /// which takes an exclusive lock before truncating.
    pub fn refresh(&mut self) -> Result<(), ReadError> {
        self.file.lock_shared()?;
        let res = self.refresh_locked();
        let unlock_res = self.file.unlock();
        res?;
        unlock_res?;
        Ok(())
    }

    /// Applies the next record from disk to in-memory indices.
    ///
    /// Aborts if the pile file is observed to shrink below the portion already
    /// applied, which would otherwise leave existing `Bytes` handles dangling
    /// and lead to undefined behavior.
    fn apply_next(&mut self) -> Result<Option<Applied>, ReadError> {
        let file_len = self.observed_file_len()?;
        self.ensure_mapped(file_len)?;
        self.apply_next_bounded(file_len)
    }

    fn observed_file_len(&self) -> Result<usize, ReadError> {
        usize::try_from(self.file.metadata()?.len())
            .map_err(|_| ReadError::FileTooLarge { length: usize::MAX })
    }

    /// Applies one record from a file-length snapshot already covered by the
    /// current mapping. Keeping the bound stable lets `refresh_locked` replay
    /// a complete observed prefix with one metadata lookup instead of one
    /// syscall per record. Appends after the snapshot are picked up by the
    /// next refresh, while post-write readback continues to use `apply_next`.
    fn apply_next_bounded(&mut self, file_len: usize) -> Result<Option<Applied>, ReadError> {
        if file_len < self.applied_length {
            // Truncation below `applied_length` invalidates previously issued
            // `Bytes` handles, so there is no safe recovery path.
            std::process::abort();
        }
        if file_len == self.applied_length {
            return Ok(None);
        }
        debug_assert!(file_len <= self.mmap.len());
        let start_offset = self.applied_length;
        let slice = unsafe {
            slice_from_raw_parts(
                self.mmap.as_ptr().add(start_offset),
                file_len - start_offset,
            )
            .as_ref()
            .unwrap()
        };
        // Single decoder shared with [`PileRecords`] — understands every
        // record format ever written (V1, unenveloped V3/V4, and envelope).
        let record = decode_record(slice, start_offset)?;
        let next_applied_length = start_offset + record.len;
        let legacy_collection_header =
            matches!(record.content, PileRecordContent::LegacyCollectionV3 { .. }).then(|| {
                let mut header = [0u8; V3_HEADER_LEN];
                header.copy_from_slice(&slice[..V3_HEADER_LEN]);
                header
            });
        let applied = match record.content {
            PileRecordContent::Blob { hash, .. } => {
                let candidate = IndexEntry::new(start_offset);
                match self.blobs.get(&hash.raw).copied() {
                    None => {
                        self.blobs.insert(&Entry::with_value(&hash.raw, candidate));
                    }
                    Some(existing) => {
                        // Duplicate hash: keep the existing record unless it
                        // turns out to be corrupt, in which case the fresh
                        // copy replaces it.
                        let record =
                            indexed_blob_record(&self.mmap, self.applied_length, existing, &hash);
                        let state = self.validations.state(
                            existing.record_offset,
                            &record.bytes,
                            &hash,
                            ValidationStrategy::Serial,
                        );
                        if let ValidationState::Invalid = state {
                            self.blobs.replace(&Entry::with_value(&hash.raw, candidate));
                        }
                    }
                }
                Applied::Blob { hash }
            }
            PileRecordContent::Branch { branch_id, head } => {
                let entry = Entry::with_value(&branch_id.into(), head);
                // Replace existing mapping (if any) with the new head.
                self.branches.replace(&entry);
                Applied::Branch {
                    id: branch_id,
                    hash: head.into(),
                }
            }
            PileRecordContent::BranchTombstone { branch_id } => {
                self.branches.remove(&branch_id.into());
                Applied::BranchTombstone { id: branch_id }
            }
            PileRecordContent::WeakPin { handle } => {
                let request = WantRequest::blob(handle);
                self.wants.insert(&Entry::new(&request.to_bytes()));
                Applied::WantAssert { request }
            }
            PileRecordContent::WeakUnpin { handle } => {
                let request = WantRequest::blob(handle);
                self.wants.remove(&request.to_bytes());
                Applied::WantRetract { request }
            }
            PileRecordContent::WantAssert { request } => {
                self.wants.insert(&Entry::new(&request.to_bytes()));
                Applied::WantAssert { request }
            }
            PileRecordContent::WantRetract { request } => {
                self.wants.remove(&request.to_bytes());
                Applied::WantRetract { request }
            }
            PileRecordContent::Collection { record } => {
                let id = record.id();
                if let Some(existing) = self.collection_records.get(&id) {
                    if existing != &record {
                        return Err(ReadError::CorruptPile {
                            valid_length: start_offset,
                        });
                    }
                } else {
                    self.collection_records.insert(id, record);
                }
                Applied::Collection { id }
            }
            PileRecordContent::CollectionGossip { grant } => {
                self.collection_gossips.insert(grant);
                Applied::CollectionGossip { grant }
            }
            PileRecordContent::LegacyCollectionV3 { .. } => {
                self.legacy_collection_headers.insert(
                    legacy_collection_header
                        .expect("legacy collection record must retain its physical header"),
                );
                Applied::LegacyCollectionV3
            }
            PileRecordContent::Opaque { .. } => {
                self.opaque_records = self
                    .opaque_records
                    .checked_add(1)
                    .expect("opaque pile-record count overflow");
                Applied::Opaque
            }
        };
        self.applied_length = next_applied_length;
        Ok(Some(applied))
    }

    fn refresh_locked(&mut self) -> Result<(), ReadError> {
        // The observed length is the refresh linearization point. Small atomic
        // writers share this lock and may append afterwards; those records are
        // intentionally left for the next refresh. Exclusive writers and
        // amputation remain excluded for the complete bounded replay.
        let file_len = self.observed_file_len()?;
        if file_len < self.applied_length {
            std::process::abort();
        }
        self.ensure_mapped(file_len)?;
        while self.apply_next_bounded(file_len)?.is_some() {}
        Ok(())
    }

    /// Amputates the pile's tail: **TRUNCATES the file at the first malformed
    /// or truncated record, destroying everything after it.**
    ///
    /// This is a last-resort surgical recovery for a torn tail left by a
    /// crashed or interrupted append — never a routine open path. Everything
    /// past the malformed record is *gone from disk*. Complete opaque envelopes
    /// are crossed, and a torn one has a known starting boundary. An unknown
    /// unenveloped marker is instead reported as
    /// [`ReadError::UnsupportedRecord`] and is never truncated because its
    /// length is unknowable. If you are not certain the tail is a torn write, take a copy of
    /// the file first and prefer the non-mutating [`Self::refresh`], which
    /// fails loud without touching the file.
    ///
    /// The method first attempts a regular [`Self::refresh`]. If corruption is
    /// detected, it acquires an exclusive lock, re-attempts the refresh and,
    /// upon confirming the corruption, truncates the pile to the last known
    /// good offset. The exclusive lock blocks other readers so truncation
    /// cannot race with [`Self::refresh`].
    pub fn amputate(&mut self) -> Result<(), ReadError> {
        match self.refresh() {
            Ok(()) => Ok(()),
            Err(ReadError::CorruptPile { .. }) => {
                self.amputate_exclusive(None).map(|_truncated| ())
            }
            Err(e) => Err(e),
        }
    }

    /// Amputates only when the malformed record begins at
    /// `expected_valid_length`.
    ///
    /// Unlike checking [`Self::refresh`] before calling [`Self::amputate`],
    /// this compares the observed boundary and truncates under one exclusive
    /// file lock. If the boundary differs, it returns
    /// [`ReadError::CorruptPile`] with the current boundary and leaves the file
    /// unchanged. The returned boolean says whether a tail was truncated; it
    /// is false when the pile was already valid.
    pub fn amputate_at(&mut self, expected_valid_length: usize) -> Result<bool, ReadError> {
        self.amputate_exclusive(Some(expected_valid_length))
    }

    fn amputate_exclusive(
        &mut self,
        expected_valid_length: Option<usize>,
    ) -> Result<bool, ReadError> {
        self.file.lock()?;
        let result = (|| match self.refresh_locked() {
            Ok(()) => Ok(false),
            Err(ReadError::CorruptPile { valid_length })
                if expected_valid_length.is_some_and(|expected| expected != valid_length) =>
            {
                Err(ReadError::CorruptPile { valid_length })
            }
            Err(ReadError::CorruptPile { valid_length }) => {
                self.file.set_len(valid_length as u64)?;
                self.dirty = true;
                self.flush().map_err(|err| match err {
                    FlushError::IoError(err) => ReadError::IoError(err),
                })?;
                self.applied_length = valid_length;
                Ok(true)
            }
            Err(error) => Err(error),
        })();
        let unlock = self.file.unlock().map_err(ReadError::from);
        match (result, unlock) {
            (Ok(truncated), Ok(())) => Ok(truncated),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    /// Persists all writes and metadata to the underlying pile file.
    pub fn flush(&mut self) -> Result<(), FlushError> {
        self.file.sync_all()?;
        self.dirty = false;
        Ok(())
    }

    fn flush_if_dirty(&mut self) -> Result<(), FlushError> {
        if self.dirty {
            self.flush()?;
        }
        Ok(())
    }

    /// Flushes pending mutations made through this handle and consumes the
    /// pile, returning an error if the flush fails.
    pub fn close(mut self) -> Result<(), FlushError> {
        let res = self.flush_if_dirty();

        let mut this = std::mem::ManuallyDrop::new(self);
        unsafe {
            std::ptr::drop_in_place(&mut this.mmap);
            std::ptr::drop_in_place(&mut this.file);
            std::ptr::drop_in_place(&mut this.blobs);
            std::ptr::drop_in_place(&mut this.validations);
            std::ptr::drop_in_place(&mut this.branches);
            std::ptr::drop_in_place(&mut this.collection_records);
            std::ptr::drop_in_place(&mut this.collection_gossips);
            std::ptr::drop_in_place(&mut this.legacy_collection_headers);
            std::ptr::drop_in_place(&mut this.wants);
        }

        res
    }
}

impl Drop for Pile {
    fn drop(&mut self) {
        eprintln!("warning: Pile dropped without calling close(); data may not be persisted");
    }
}

// Implement the repository storage close trait so callers can call
// `repo.close()` when the repository was created with a `Pile` storage.
impl crate::repo::StorageClose for Pile {
    type Error = FlushError;

    fn close(self) -> Result<(), Self::Error> {
        Pile::close(self)
    }
}

// Generic durability hook: appended records (blobs, branch updates,
// collection records, want markers) are not crash-durable until flushed — see the
// inherent [`Pile::flush`].
impl crate::repo::StorageFlush for Pile {
    type Error = FlushError;

    fn flush(&mut self) -> Result<(), Self::Error> {
        Pile::flush(self)
    }
}

use super::BlobInfo;
use super::BlobStore;
use super::BlobStoreGet;
use super::BlobStoreList;
use super::BlobStorePut;
use super::PinStore;
use super::PushResult;
use super::WantStore;

/// Iterator returned by [`PileReader::iter`].
///
/// Iterates over all `(Handle, Blob)` pairs currently stored in the pile.
/// Owned iterator over all blobs currently stored in the pile. This collects
/// a snapshot of keys/indices at iterator creation so the iterator does not
/// borrow the underlying [`PATCH`] and can live independently of the [`Pile`].
pub struct PileBlobStoreIter {
    mmap: Arc<MmapRaw>,
    covered_len: usize,
    inner: crate::patch::PATCHIntoIterator<32, IdentitySchema, IndexEntry>,
    lookup: PATCH<32, IdentitySchema, IndexEntry>,
    validations: ValidationCache,
}

impl Iterator for PileBlobStoreIter {
    type Item = Result<(Inline<Handle<UnknownBlob>>, Blob<UnknownBlob>), GetBlobError<Infallible>>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.inner.next()?;
        let hash = Inline::<Hash<Blake3>>::new(key);
        let Some(entry) = self.lookup.get(&key) else {
            return Some(Err(GetBlobError::BlobNotFound));
        };
        let entry = *entry;
        let record = indexed_blob_record(&self.mmap, self.covered_len, entry, &hash);
        match self.validations.state(
            entry.record_offset,
            &record.bytes,
            &hash,
            ValidationStrategy::ParallelIfLarge,
        ) {
            ValidationState::Validated => {
                let handle: Inline<Handle<UnknownBlob>> = hash.into();
                let blob = Blob::with_handle(record.bytes, handle);
                Some(Ok((handle, blob)))
            }
            ValidationState::Invalid => Some(Err(GetBlobError::ValidationError(record.bytes))),
        }
    }
}

/// Adapter that yields blob information from an owned PATCH snapshot.
pub struct PileBlobStoreListIter {
    reader: PileReader,
    inner: crate::patch::PATCHIntoIterator<32, IdentitySchema, IndexEntry>,
}

impl Iterator for PileBlobStoreListIter {
    type Item = Result<BlobInfo, GetBlobError<Infallible>>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.inner.next()?;
        let hash = Inline::<Hash<Blake3>>::new(key);
        let handle = hash.into();
        Some(Ok(self.reader.blob_info(handle).expect(
            "key from PATCH iterator must resolve in the same snapshot",
        )))
    }
}

impl BlobStoreList for PileReader {
    type Err = GetBlobError<Infallible>;
    type Iter<'a> = PileBlobStoreListIter;

    fn blobs(&self) -> Self::Iter<'_> {
        PileBlobStoreListIter {
            reader: self.clone(),
            inner: self.blobs.clone().into_iter(),
        }
    }

    fn contains_blob<S>(&self, handle: Inline<Handle<S>>) -> Result<bool, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        Ok(self.blobs.get(&handle.raw).is_some())
    }

    /// Cheap PATCH-level set difference between two immutable reader snapshots.
    fn blobs_diff(&self, old: &Self) -> Self::Iter<'_> {
        PileBlobStoreListIter {
            reader: self.clone(),
            inner: self.blobs.difference(&old.blobs).into_iter(),
        }
    }
}

/// Iterator over pin ids stored in the pile's PATCH, using the PATCH's
/// built-in key iterator to avoid allocating a full Vec of ids.
pub struct PileBranchStoreIter {
    inner:
        crate::patch::PATCHIntoOrderedIterator<16, IdentitySchema, Inline<Handle<SimpleArchive>>>,
}

impl Iterator for PileBranchStoreIter {
    type Item = Result<Id, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        // The owned ordered iterator yields key arrays ([u8; 16]) by value.
        // The `apply_next` path guarantees that a nil (all-zero) pin id
        // is never inserted into the PATCH; therefore we can safely `expect`
        // a valid `Id` here and treat a nil id as an invariant violation.
        let key = self.inner.next()?;
        let id = Id::new(key).expect("nil pin id inserted into patch");
        Some(Ok(id))
    }
}

/// Deterministic owned snapshot of the pile's native collection records.
pub struct PileCollectionRecordIter {
    inner: std::collections::btree_map::IntoValues<Id, CollectionRecord>,
}

/// Deterministic owned snapshot of the pile's grow-only publication grants.
pub struct PileCollectionGossipIter {
    inner: std::collections::btree_set::IntoIter<CollectionGossip>,
}

impl Iterator for PileCollectionGossipIter {
    type Item = Result<CollectionGossip, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Ok)
    }
}

impl Iterator for PileCollectionRecordIter {
    type Item = Result<CollectionRecord, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Ok)
    }
}

impl Pile {
    /// Return the number of unknown generic-envelope records in the coherent
    /// applied prefix. Physical rewriting must refuse while this is nonzero,
    /// because an older binary cannot compute an unknown kind's preservation
    /// or retention semantics.
    pub fn opaque_record_count(&mut self) -> Result<usize, ReadError> {
        self.refresh()?;
        Ok(self.opaque_records)
    }

    /// Copy every byte-distinct inert legacy V3 collection header into
    /// `destination`.
    ///
    /// This is an internal physical-rewrite primitive. Legacy headers do not
    /// participate in the current collection algebra, but reclaim must retain
    /// their exact source evidence for a later explicit migration. The source
    /// is refreshed before the snapshot and destination insertion is
    /// idempotent by exact header bytes.
    pub(crate) fn preserve_legacy_collection_headers_into(
        &mut self,
        destination: &mut Pile,
    ) -> Result<(), CollectionInsertError> {
        self.refresh()?;
        let headers = self.legacy_collection_headers.clone();
        for header in headers {
            destination.preserve_legacy_collection_header(header)?;
        }
        Ok(())
    }

    /// Append one already-validated legacy collection header if this pile does
    /// not already contain the same physical evidence.
    fn preserve_legacy_collection_header(
        &mut self,
        header: [u8; V3_HEADER_LEN],
    ) -> Result<(), CollectionInsertError> {
        debug_assert!(matches!(
            decode_record(&header, 0),
            Ok(PileRecord {
                content: PileRecordContent::LegacyCollectionV3 { .. },
                ..
            })
        ));

        self.file.lock()?;
        let result = (|| {
            self.refresh_locked()?;

            if self.legacy_collection_headers.contains(&header) {
                return Ok(());
            }

            self.dirty = true;
            let written = self.file.write(&header)?;
            if written != V3_HEADER_LEN {
                return Err(CollectionInsertError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write complete legacy collection record",
                )));
            }

            match self.apply_next()? {
                Some(Applied::LegacyCollectionV3) => Ok(()),
                Some(_) | None => Err(CollectionInsertError::UnexpectedReadback),
            }
        })();
        let unlock = self.file.unlock();
        result?;
        unlock?;
        Ok(())
    }
}

impl CollectionStore for Pile {
    type RecordsError = ReadError;
    type InsertError = CollectionInsertError;
    type RecordIter<'a> = PileCollectionRecordIter;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        self.refresh()?;
        Ok(PileCollectionRecordIter {
            inner: self.collection_records.clone().into_values(),
        })
    }

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        if selectors.is_empty() {
            return Ok(Vec::new());
        }
        self.refresh()?;
        Ok(self
            .collection_records
            .values()
            .copied()
            .filter(|record| selectors_match_record(selectors, *record))
            .collect())
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        let id = record.id();
        let header = collection_record_header(&record);

        self.file.lock()?;
        let result = (|| {
            self.refresh_locked()?;

            if let Some(existing) = self.collection_records.get(&id) {
                return if existing == &record {
                    Ok(())
                } else {
                    Err(CollectionInsertError::IdCollision { id })
                };
            }

            self.dirty = true;
            let written = self.file.write(&header)?;
            if written != ENVELOPE_HEADER_LEN {
                return Err(CollectionInsertError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write complete collection record",
                )));
            }

            match self.apply_next()? {
                Some(Applied::Collection { id: applied }) if applied == id => Ok(()),
                Some(_) | None => Err(CollectionInsertError::UnexpectedReadback),
            }
        })();
        let unlock = self.file.unlock();
        result?;
        unlock?;
        Ok(())
    }
}

impl CollectionGossipStore for Pile {
    type GossipsError = ReadError;
    type GossipError = CollectionInsertError;
    type GossipIter<'a> = PileCollectionGossipIter;

    fn gossips<'a>(&'a mut self) -> Result<Self::GossipIter<'a>, Self::GossipsError> {
        self.refresh()?;
        Ok(PileCollectionGossipIter {
            inner: self.collection_gossips.clone().into_iter(),
        })
    }

    fn gossip(&mut self, grant: CollectionGossip) -> Result<(), Self::GossipError> {
        let header = CollectionGossipHeaderEnvelope::new(&grant);

        self.file.lock()?;
        let result = (|| {
            self.refresh_locked()?;

            if self.collection_gossips.contains(&grant) {
                return Ok(());
            }

            self.dirty = true;
            let written = self.file.write(header.as_bytes())?;
            if written != ENVELOPE_HEADER_LEN {
                return Err(CollectionInsertError::Io(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write complete collection-gossip grant",
                )));
            }

            match self.apply_next()? {
                Some(Applied::CollectionGossip { grant: applied }) if applied == grant => Ok(()),
                Some(_) | None => Err(CollectionInsertError::UnexpectedReadback),
            }
        })();
        let unlock = self.file.unlock();
        result?;
        unlock?;
        Ok(())
    }
}

impl BlobStorePut for Pile {
    type PutError = InsertError;

    /// Inserts a blob into the pile and returns its handle.
    ///
    /// For records up to `ATOMIC_WRITE_LIMIT` the append relies on the
    /// kernel's atomic `write_vectored` guarantee, so multiple writers can
    /// hold a shared file lock and proceed concurrently. Larger records
    /// take an exclusive lock and append via plain `write_all`, trading
    /// concurrency for reach — the recovery path
    /// ([`Pile::amputate`]) truncates any partial tail left by a crash,
    /// so a multi-`write` record is still crash-safe. Multiple writers
    /// are safe only on filesystems guaranteeing atomic `write`/`vwrite`
    /// appends; other filesystems may corrupt the pile.
    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        self.put_impl(item)
    }
}

impl Pile {
    /// Shared blob-append. Writes an enveloped record: a fixed 256-byte header, the blob
    /// data at `record_start + ENVELOPE_HEADER_LEN`, and post-padding to a 256-byte
    /// multiple. Because the envelope has no offset-derived pad, the append uses the atomic
    /// shared-lock fast path for records up to `ATOMIC_WRITE_LIMIT` (no exclusive lock needed —
    /// a fixed header has no start offset to stabilize). The data is
    /// absolutely 256-aligned (zero-copy GPU-aliasable) in a current pile, which
    /// stays 256-aligned because every record span is a count of 256-byte blocks.
    fn put_impl<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, InsertError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let blob = IntoBlob::to_blob(item);
        let blob_size = blob.bytes.len();
        let padding = block_post_pad(blob_size);
        let span_blocks = envelope_blocks_for_payload(blob_size).ok_or_else(|| {
            InsertError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "blob is too large for the u32 pile-record span",
            ))
        })?;
        let record_size = ENVELOPE_HEADER_LEN
            .checked_add(blob_size)
            .and_then(|size| size.checked_add(padding))
            .ok_or_else(|| {
                InsertError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "blob pile-record size overflows usize",
                ))
            })?;
        let use_atomic = record_size <= ATOMIC_WRITE_LIMIT;

        if use_atomic {
            self.file.lock_shared()?;
        } else {
            // Oversized record: exclude other writers for the duration of
            // the multi-syscall append. Shared readers ([`refresh`]) block
            // until unlock, so they never observe a partially-written tail.
            self.file.lock()?;
        }
        let res = (|| {
            self.refresh_locked().map_err(InsertError::from)?;

            let handle: Inline<Handle<S>> = blob.get_handle();
            let hash: Inline<Hash<Blake3>> = handle.into();

            if let Some(entry) = self.blobs.get(&hash.raw).copied() {
                let record = indexed_blob_record(&self.mmap, self.applied_length, entry, &hash);
                let state = self.validations.state(
                    entry.record_offset,
                    &record.bytes,
                    &hash,
                    ValidationStrategy::Serial,
                );
                if matches!(state, ValidationState::Validated) {
                    return Ok(handle.transmute());
                }
            }
            let now_in_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
            let header =
                BlobHeaderEnvelope::new(span_blocks, now_in_ms as u64, blob_size as u64, hash);
            let actual_record_size = record_size;
            // post-pad is < 256.
            let zero_buf = [0u8; ENVELOPE_BLOCK_LEN];
            // Mark before entering the syscall: partial writes and later
            // read-back failures must still leave close responsible for the
            // bytes this handle may have appended.
            self.dirty = true;
            if use_atomic {
                let bufs = [
                    IoSlice::new(header.as_bytes()),
                    IoSlice::new(blob.bytes.as_ref()),
                    IoSlice::new(&zero_buf[..padding]),
                ];
                let written = self.file.write_vectored(&bufs)?;
                if written != actual_record_size {
                    return Err(InsertError::IoError(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "failed to write blob record",
                    )));
                }
            } else {
                // Separate `write_all` calls — payload dominates, so the extra
                // syscalls for header/padding are negligible. Any partial
                // completion after a crash is caught by `amputate`.
                self.file.write_all(header.as_bytes())?;
                self.file.write_all(blob.bytes.as_ref())?;
                if padding > 0 {
                    self.file.write_all(&zero_buf[..padding])?;
                }
            }

            loop {
                match self.apply_next().map_err(InsertError::from)? {
                    Some(Applied::Blob { hash: h }) => {
                        if h == hash {
                            break;
                        }
                    }
                    Some(Applied::Branch { .. }) => {}
                    Some(Applied::BranchTombstone { .. }) => {}
                    Some(Applied::WantAssert { .. }) => {}
                    Some(Applied::WantRetract { .. }) => {}
                    Some(Applied::Collection { .. }) => {}
                    Some(Applied::CollectionGossip { .. }) => {}
                    Some(Applied::LegacyCollectionV3) => {}
                    Some(Applied::Opaque) => {}
                    None => {
                        return Err(InsertError::IoError(std::io::Error::other(
                            "blob missing after write",
                        )));
                    }
                }
            }

            Ok(handle.transmute())
        })();
        let unlock_res = self.file.unlock();
        let handle = res?;
        unlock_res?;
        Ok(handle)
    }
}

impl PinStore for Pile {
    type PinsError = ReadError;
    // Pulling a head may require refreshing the pile which can fail; expose
    // the underlying `ReadError` so callers can surface refresh failures.
    type HeadError = ReadError;
    type UpdateError = PileWriteError;

    type ListIter<'a> = PileBranchStoreIter;

    fn pins<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::PinsError> {
        // Ensure newly appended records are applied before enumerating
        // branches so external writers are visible to callers.
        self.refresh()?;
        // Create an owned ordered iterator from the PATCH clone so the
        // returned iterator does not borrow from `self.branches`. This avoids
        // allocating a temporary Vec of ids while preserving tree-order.
        let cloned = self.branches.clone();
        let inner = cloned.into_iter_ordered();
        Ok(PileBranchStoreIter { inner })
    }

    fn head(&mut self, id: Id) -> Result<Option<Inline<Handle<SimpleArchive>>>, Self::HeadError> {
        // Ensure newly appended records are applied before returning the head.
        // This keeps callers up-to-date with any external writers that appended
        // to the pile file.
        self.refresh()?;
        Ok(self.branches.get(&id.into()).copied())
    }

    /// Updates the head of `id` to `new` if it matches `old`.
    ///
    /// This method does not verify that `new` refers to a blob stored in the pile,
    /// allowing piles to reference external data and serve as head-only stores.
    ///
    /// The update is written to the pile but is **not durable** until
    /// [`Pile::flush`] is called. Callers must explicitly flush to ensure
    /// pin updates survive crashes.
    ///
    /// After the header is written, the record is read back with `apply_next`
    /// while still holding the lock, ensuring the update is applied without an
    /// additional refresh pass.
    fn update(
        &mut self,
        id: Id,
        old: Option<Inline<Handle<SimpleArchive>>>,
        new: Option<Inline<Handle<SimpleArchive>>>,
    ) -> Result<super::PushResult, Self::UpdateError> {
        self.file.lock()?;
        let res = (|| {
            self.refresh_locked().map_err(PileWriteError::from)?;
            let current_hash = self.branches.get(&id.into()).copied();
            if current_hash != old {
                return Ok(PushResult::Conflict(current_hash));
            }

            // No-op short-circuit: if the requested head is already
            // what we have, return success without appending a record.
            // The pin table is logically a (id → head) map; a write
            // where new == current carries no information and would
            // just churn the append-only file. Steady-state gossip
            // rebroadcasts of unchanged heads (e.g. tracking-pin
            // re-publication at 30s ticks) hit this path heavily.
            if current_hash == new {
                return Ok(PushResult::Success());
            }

            // Enveloped branch/tombstone records: fixed 256-byte header, no data, so the
            // record is exactly one 256-byte unit — keeping a current pile
            // 256-aligned throughout (branches write under the exclusive lock).
            self.dirty = true;
            let (expected, write_res) = match new {
                Some(new) => {
                    let header = BranchHeaderEnvelope::new(id, new);
                    (ENVELOPE_HEADER_LEN, self.file.write(header.as_bytes()))
                }
                None => {
                    let header = BranchTombstoneHeaderEnvelope::new(id);
                    (ENVELOPE_HEADER_LEN, self.file.write(header.as_bytes()))
                }
            };
            let written = match write_res {
                Ok(n) => n,
                Err(e) => return Err(PileWriteError::IoError(e)),
            };
            if written != expected {
                return Err(PileWriteError::IoError(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write branch header",
                )));
            }
            match self.apply_next().map_err(PileWriteError::from)? {
                Some(Applied::Branch { id: bid, hash }) if matches!(new, Some(new) if bid == id && hash == new.into()) => {
                    Ok(PushResult::Success())
                }
                Some(Applied::BranchTombstone { id: bid }) if new.is_none() && bid == id => {
                    Ok(PushResult::Success())
                }
                Some(_) => Err(PileWriteError::IoError(std::io::Error::other(
                    "unexpected record after branch write",
                ))),
                None => Err(PileWriteError::IoError(std::io::Error::other(
                    "branch missing after write",
                ))),
            }
        })();
        let unlock_res = self.file.unlock();
        let out = res?;
        unlock_res?;
        Ok(out)
    }
}

/// Iterator over the LWW-resolved typed requests stored in the pile,
/// using the PATCH's ordered key iterator (byte order, deterministic).
pub struct PileWantIter {
    inner: crate::patch::PATCHIntoOrderedIterator<WANT_REQUEST_BYTES_LEN, IdentitySchema, ()>,
}

impl Iterator for PileWantIter {
    type Item = Result<WantRequest, PileWriteError>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.inner.next()?;
        Some(Ok(WantRequest::from_bytes(bytes).expect(
            "Pile only indexes structurally decoded canonical want requests",
        )))
    }
}

impl Pile {
    /// Append an enveloped typed want assertion or retraction.
    /// Mirrors [`PinStore::update`]'s write path:
    /// exclusive lock, refresh, no-op short-circuit when the LWW state already
    /// matches, a single fixed 256-byte header write (keeping a current pile
    /// 256-aligned), and an `apply_next` read-back while still holding the
    /// lock. Like branch updates, the record is **not durable** until
    /// [`Pile::flush`] is called.
    fn write_want_marker(
        &mut self,
        request: WantRequest,
        asserted: bool,
    ) -> Result<(), PileWriteError> {
        self.file.lock()?;
        let res = (|| {
            self.refresh_locked().map_err(PileWriteError::from)?;

            // No-op short-circuit: the wanted set is logically a per-request
            // LWW register; re-asserting the current state carries no
            // information and would just churn the append-only file.
            let key = request.to_bytes();
            let current = self.wants.get(&key).is_some();
            if current == asserted {
                return Ok(());
            }

            self.dirty = true;
            // Blob wants retain the historical physical marker. Otherwise an
            // older reader would see `legacy assert; typed retract` as
            // `assert; opaque` and resurrect the request. Operation wants are
            // independent of every legacy meaning and use the typed marker.
            let write_res = match request {
                WantRequest::Blob { handle } => self
                    .file
                    .write(WantHeaderEnvelope::new(handle, asserted).as_bytes()),
                WantRequest::Merge { .. } | WantRequest::Derive { .. } => self.file.write(
                    TypedWantHeaderEnvelope::new_operation(request, asserted)
                        .expect("collection-operation want must have a typed envelope")
                        .as_bytes(),
                ),
            };
            let written = write_res.map_err(PileWriteError::IoError)?;
            if written != ENVELOPE_HEADER_LEN {
                return Err(PileWriteError::IoError(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write typed want header",
                )));
            }
            match self.apply_next().map_err(PileWriteError::from)? {
                Some(Applied::WantAssert { request: actual }) if asserted && actual == request => {
                    Ok(())
                }
                Some(Applied::WantRetract { request: actual })
                    if !asserted && actual == request =>
                {
                    Ok(())
                }
                Some(_) => Err(PileWriteError::IoError(std::io::Error::other(
                    "unexpected record after typed want write",
                ))),
                None => Err(PileWriteError::IoError(std::io::Error::other(
                    "typed want marker missing after write",
                ))),
            }
        })();
        let unlock_res = self.file.unlock();
        res?;
        unlock_res?;
        Ok(())
    }
}

impl super::PinSnapshotSource for Pile {
    type PinSnapshotError = ReadError;

    fn snapshot_pin_heads(&mut self) -> Result<super::PinSnapshot, Self::PinSnapshotError> {
        // PATCH is persistent, so this is one cheap immutable snapshot. Keep
        // refresh here as the single strict path: failure is returned rather
        // than becoming a partial authorization view.
        self.refresh()?;
        Ok(self.branches.clone())
    }
}

impl WantStore for Pile {
    type WantError = PileWriteError;
    type WantIter<'a> = PileWantIter;

    /// Assert `request`; call [`Pile::flush`] to make it crash-durable.
    fn want(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.write_want_marker(request, true)
    }

    /// Retract `request` (last-writer-wins by log position).
    fn unwant(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.write_want_marker(request, false)
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        // Ensure newly appended records are applied before enumerating so
        // external writers are visible to callers (mirrors `pins`).
        self.refresh()?;
        let cloned = self.wants.clone();
        Ok(PileWantIter {
            inner: cloned.into_iter_ordered(),
        })
    }
}

impl crate::repo::BlobStoreMeta for PileReader {
    type MetaError = Infallible;

    fn metadata<S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<Option<crate::repo::BlobMetadata>, Self::MetaError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let hash: &Inline<Hash<Blake3>> = handle.as_transmute();
        let Some(entry) = self.blobs.get(&hash.raw) else {
            return Ok(None);
        };
        let entry = *entry;
        let record = indexed_blob_record(&self.mmap, self.covered_len, entry, hash);
        let state = self.validations.state(
            entry.record_offset,
            &record.bytes,
            hash,
            ValidationStrategy::ParallelIfLarge,
        );
        match state {
            ValidationState::Validated => Ok(Some(crate::repo::BlobMetadata {
                timestamp: record.timestamp,
                length: record.bytes.len() as u64,
            })),
            ValidationState::Invalid => Ok(None),
        }
    }
}

/// How a source pile's active wants participate in a retained rewrite.
///
/// Wants are demand markers, not ownership roots. Preserving them copies
/// the marker into the destination but does not retain or copy the requested
/// blob unless an explicit or strong-pin root reaches it independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WantRewritePolicy {
    /// Recreate every active want marker without promoting its target.
    Preserve,
    /// Omit want markers from the destination.
    Drop,
}

/// Deterministic accounting for one retained pile rewrite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PileRewriteStats {
    /// Exact number of resident blobs selected and copied.
    pub retained_blobs: usize,
    /// Number of active legacy strong-pin mappings recreated.
    pub strong_pins: usize,
    /// Number of want markers recreated.
    pub wants: usize,
    /// Number of grow-only collection-publication grants preserved.
    pub collection_gossips: usize,
}

/// Failure while copying one policy-selected pile state into another pile.
#[derive(Debug)]
#[non_exhaustive]
pub enum PileRewriteError {
    /// The source could not produce a coherent reader snapshot.
    Source(ReadError),
    /// The source contains opaque records, so a semantic rewrite could not
    /// prove that it would preserve their bytes and retention laws.
    OpaqueRecords {
        /// Number of opaque records observed in the source snapshot.
        count: usize,
    },
    /// A selected blob was absent, invalid, or could not be stored.
    Transfer(super::TransferError<Infallible, GetBlobError<Infallible>, InsertError>),
    /// A strong-pin mapping could not be appended to the destination.
    StrongPin(PileWriteError),
    /// The destination already maps a retained pin id to another head.
    StrongPinConflict {
        /// Conflicting pin id.
        id: Id,
        /// Head already present in the destination.
        current: Option<Inline<Handle<SimpleArchive>>>,
    },
    /// A preserved want marker could not be appended.
    Want(PileWriteError),
    /// An immutable collection-algebra record could not be appended.
    Collection(CollectionInsertError),
    /// The completed destination state could not be made durable.
    Flush(FlushError),
}

impl std::fmt::Display for PileRewriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => write!(f, "failed to snapshot source pile: {error}"),
            Self::OpaqueRecords { count } => write!(
                f,
                "refusing to rewrite a pile containing {count} opaque record(s)"
            ),
            Self::Transfer(error) => write!(f, "failed to copy a retained blob: {error}"),
            Self::StrongPin(error) => write!(f, "failed to recreate a strong pin: {error}"),
            Self::StrongPinConflict { id, current } => write!(
                f,
                "destination has conflicting strong pin {id:X} at {current:?}"
            ),
            Self::Want(error) => write!(f, "failed to recreate a want: {error}"),
            Self::Collection(error) => {
                write!(f, "failed to preserve a collection record: {error}")
            }
            Self::Flush(error) => write!(f, "failed to flush rewritten pile: {error}"),
        }
    }
}

impl Error for PileRewriteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Transfer(error) => Some(error),
            Self::StrongPin(error) | Self::Want(error) => Some(error),
            Self::Collection(error) => Some(error),
            Self::Flush(error) => Some(error),
            Self::StrongPinConflict { .. } | Self::OpaqueRecords { .. } => None,
        }
    }
}

impl Pile {
    /// Copy a policy-selected state into another append-only pile.
    ///
    /// `explicit` is normally produced by a higher-level policy such as
    /// collection resolution. This byte-copy boundary deliberately knows
    /// nothing about collection authorization and does not persist the policy;
    /// callers must recompute and supply its roots on every later rewrite. It
    /// additionally treats every active legacy strong-pin head as a recursive
    /// ownership root and recreates the exact pin mapping, allowing collection
    /// and branch models to coexist during migration.
    ///
    /// The source is refreshed once; blobs, strong pins, collection records,
    /// inert legacy collection headers, and wants are then
    /// taken from that coherent applied-prefix snapshot. Strictly verified V4
    /// commits retain their resident descriptor, data, and metadata recursively;
    /// an invalid commit authenticates none of its fields. A valid commit whose
    /// dependency is not resident is still copied as durable ground truth, but
    /// the absent dependency is not manufactured into a transfer root. Later
    /// synchronization may satisfy it. Merge and derive records are algebraic
    /// evidence rather than ownership edges. Byte-distinct legacy V3 collection
    /// headers are copied exactly but remain semantically inert. The destination
    /// may already contain identical blobs, records, headers, and strong-pin
    /// mappings, making retries idempotent, but a differently mapped pin or
    /// intrinsic-record collision is an error. Missing or invalid explicitly
    /// selected blobs still fail the rewrite rather than silently weakening the
    /// caller's retention policy. One final flush makes blobs and records durable
    /// in append order.
    pub fn rewrite_retained_into(
        &mut self,
        destination: &mut Pile,
        explicit: &super::RetentionRoots,
        wants: WantRewritePolicy,
    ) -> Result<PileRewriteStats, PileRewriteError> {
        let reader = self.reader().map_err(PileRewriteError::Source)?;
        if self.opaque_records != 0 {
            return Err(PileRewriteError::OpaqueRecords {
                count: self.opaque_records,
            });
        }
        let strong_pins = self.branches.clone();
        let collection_records = self.collection_records.clone();
        let collection_gossips = self.collection_gossips.clone();
        let legacy_collection_headers = self.legacy_collection_headers.clone();
        let source_wants = self.wants.clone();

        let mut roots = explicit.clone();
        for raw in &strong_pins {
            let head = *strong_pins
                .get(raw)
                .expect("pin key from snapshot must retain its value");
            roots.retain_recursive(head);
        }
        for record in collection_records.values() {
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            if commit.verify_strict().is_err() {
                // Structural decoding is deliberately weaker than authority:
                // preserve the immutable record below for diagnostics and
                // future replication, but let none of its attacker-controlled
                // fields affect local lifetime.
                continue;
            }

            // Unlike explicit policy roots, native COMMIT dependencies may be
            // absent on a partially synchronized node. Observe only this
            // coherent reader snapshot; do not issue a demand read, and do not
            // turn absence into a root which would make every later rewrite
            // fail. `reachable` already applies the same resident-only rule to
            // recursive descendants.
            let descriptor = Inline::<Handle<UnknownBlob>>::new(commit.collection().raw);
            let data = Inline::<Handle<UnknownBlob>>::new(commit.data().raw);
            let metadata = commit.metadata().transmute();
            for handle in [descriptor, data, metadata] {
                if reader
                    .contains_blob(handle)
                    .expect("PileReader residency lookup is infallible")
                {
                    roots.retain_recursive(handle);
                }
            }
        }
        let keep = roots.expanded(&reader);
        let retained_blobs = keep.len();

        for copied in super::transfer(&reader, destination, keep) {
            copied.map_err(PileRewriteError::Transfer)?;
        }

        for raw in &strong_pins {
            let id = Id::new(*raw).expect("Pile never stores a nil strong-pin id");
            let head = *strong_pins
                .get(raw)
                .expect("pin key from snapshot must retain its value");
            match destination
                .update(id, None, Some(head))
                .map_err(PileRewriteError::StrongPin)?
            {
                PushResult::Success() => {}
                PushResult::Conflict(Some(current)) if current == head => {}
                PushResult::Conflict(current) => {
                    return Err(PileRewriteError::StrongPinConflict { id, current });
                }
            }
        }

        for header in legacy_collection_headers {
            destination
                .preserve_legacy_collection_header(header)
                .map_err(PileRewriteError::Collection)?;
        }

        for record in collection_records.into_values() {
            destination
                .insert(record)
                .map_err(PileRewriteError::Collection)?;
        }
        for grant in &collection_gossips {
            destination
                .gossip(*grant)
                .map_err(PileRewriteError::Collection)?;
        }

        let mut preserved_wants = 0usize;
        if wants == WantRewritePolicy::Preserve {
            for bytes in source_wants.into_iter_ordered() {
                destination
                    .want(
                        WantRequest::from_bytes(bytes).expect(
                            "Pile only indexes structurally decoded canonical want requests",
                        ),
                    )
                    .map_err(PileRewriteError::Want)?;
                preserved_wants += 1;
            }
        }

        destination.flush().map_err(PileRewriteError::Flush)?;
        Ok(PileRewriteStats {
            retained_blobs,
            strong_pins: strong_pins.len() as usize,
            wants: preserved_wants,
            collection_gossips: collection_gossips.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ed25519_dalek::SigningKey;
    use rand::RngCore;
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;
    use tempfile;

    use crate::collection::{empty_metadata_handle, CollectionDescriptor, CollectionId};
    use crate::macros::entity;
    use crate::repo::lazy::Lazy;
    use crate::repo::yard::{Yard, YardCollectError, YardConfig, YardReclaimError};
    use crate::repo::{BlobStoreMeta, PushResult, RetentionRoots, StorageClose};
    use crate::trible::TribleSet;

    fn fresh_empty_pile_path(dir: &tempfile::TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::File::create(&path).unwrap();
        path
    }

    const TEST_UNKNOWN_KIND_A: RawId = [0xA5; 16];
    const TEST_UNKNOWN_KIND_B: RawId = [0x5A; 16];

    fn test_envelope_bytes(kind: RawId, span_blocks: u32, physical_len: usize) -> Vec<u8> {
        assert!(physical_len >= 36);
        let mut bytes = vec![0xC3; physical_len];
        bytes[..16].copy_from_slice(&MAGIC_MARKER_ENVELOPE);
        bytes[16..32].copy_from_slice(&kind);
        bytes[32..36].copy_from_slice(&span_blocks.to_le_bytes());
        bytes
    }

    fn append_test_bytes(path: &Path, bytes: &[u8]) {
        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    fn collection_test_id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn collection_test_hash(byte: u8) -> Inline<Hash<Blake3>> {
        Inline::new([byte; 32])
    }

    fn collection_test_collection(byte: u8) -> CollectionId {
        Inline::new([byte; 32])
    }

    fn collection_test_records() -> Vec<CollectionRecord> {
        let source = collection_test_collection(1);
        let target = collection_test_collection(2);
        let key = SigningKey::from_bytes(&[7; 32]);
        vec![
            CollectionRecord::Commit(CollectionCommit::sign(
                &key,
                source,
                collection_test_hash(6),
                empty_metadata_handle(),
            )),
            CollectionRecord::Merge(CollectionMerge::new(
                source,
                collection_test_hash(6),
                collection_test_hash(7),
                collection_test_hash(8),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                source,
                target,
                collection_test_hash(8),
                collection_test_hash(9),
            )),
        ]
    }

    fn collection_test_gossips() -> Vec<CollectionGossip> {
        vec![
            CollectionGossip::sign(
                &SigningKey::from_bytes(&[21; 32]),
                collection_test_collection(1),
            ),
            CollectionGossip::sign(
                &SigningKey::from_bytes(&[22; 32]),
                collection_test_collection(2),
            ),
        ]
    }

    fn fixed_collection_header(bytes: &[u8]) -> [u8; V3_HEADER_LEN] {
        bytes
            .try_into()
            .expect("collection headers are exactly one fixed pile header")
    }

    fn legacy_collection_test_headers() -> Vec<(LegacyCollectionRecordKindV3, [u8; V3_HEADER_LEN])>
    {
        vec![
            (
                LegacyCollectionRecordKindV3::Definition,
                fixed_collection_header(
                    CollectionDefinitionHeaderV3 {
                        magic_marker: MAGIC_MARKER_COLLECTION_DEFINITION_V3,
                        scope: [1; 16],
                        representation: [2; 16],
                        recipe: [3; 16],
                        reserved: [0; 192],
                    }
                    .as_bytes(),
                ),
            ),
            (
                LegacyCollectionRecordKindV3::Commit,
                fixed_collection_header(
                    CollectionCommitHeaderV3 {
                        magic_marker: MAGIC_MARKER_COLLECTION_COMMIT_V3,
                        collection: [4; 16],
                        data: [5; 32],
                        metadata: [6; 32],
                        public_key: [7; 32],
                        signature_r: [8; 32],
                        signature_s: [9; 32],
                        reserved: [0; 64],
                    }
                    .as_bytes(),
                ),
            ),
            (
                LegacyCollectionRecordKindV3::Merge,
                fixed_collection_header(
                    CollectionMergeHeaderV3 {
                        magic_marker: MAGIC_MARKER_COLLECTION_MERGE_V3,
                        collection: [10; 16],
                        low: [11; 32],
                        high: [12; 32],
                        result: [13; 32],
                        reserved: [0; 128],
                    }
                    .as_bytes(),
                ),
            ),
            (
                LegacyCollectionRecordKindV3::Derive,
                fixed_collection_header(
                    CollectionDeriveHeaderV3 {
                        magic_marker: MAGIC_MARKER_COLLECTION_DERIVE_V3,
                        source: [14; 16],
                        target: [15; 16],
                        input: [16; 32],
                        output: [17; 32],
                        reserved: [0; 144],
                    }
                    .as_bytes(),
                ),
            ),
        ]
    }

    fn legacy_collection_headers_at(
        path: &Path,
    ) -> Vec<(LegacyCollectionRecordKindV3, [u8; V3_HEADER_LEN])> {
        let mut records = PileRecords::open(path).unwrap();
        let mut found = Vec::new();
        while let Some(record) = records.next() {
            let record = record.unwrap();
            let PileRecordContent::LegacyCollectionV3 { kind } = record.content else {
                continue;
            };
            let header = records.bytes()[record.offset..record.offset + record.len]
                .try_into()
                .unwrap();
            found.push((kind, header));
        }
        found
    }

    fn sorted_collection_records(mut records: Vec<CollectionRecord>) -> Vec<CollectionRecord> {
        records.sort_by_key(CollectionRecord::id);
        records
    }

    fn invalidate_collection_commit(commit: CollectionCommit) -> CollectionCommit {
        let (signature_r, signature_s) = commit.signature();
        let mut forged_r = signature_r.raw;
        forged_r[0] ^= 1;
        let forged = CollectionCommit::from_parts(
            commit.collection(),
            commit.data(),
            commit.metadata(),
            commit.public_key(),
            Inline::new(forged_r),
            signature_s,
        );
        assert!(forged.verify_strict().is_err());
        forged
    }

    #[test]
    fn enveloped_collection_record_headers_are_fixed_zero_padded_and_roundtrip() {
        let records = collection_test_records();
        let expected = [
            (MAGIC_MARKER_COLLECTION_COMMIT_V4, 228usize),
            (MAGIC_MARKER_COLLECTION_MERGE_V4, 164usize),
            (MAGIC_MARKER_COLLECTION_DERIVE_V4, 164usize),
        ];

        for (record, (magic, reserved_start)) in records.into_iter().zip(expected) {
            let header = collection_record_header(&record);
            assert_eq!(header.len(), ENVELOPE_HEADER_LEN);
            assert_eq!(&header[..16], MAGIC_MARKER_ENVELOPE.as_slice());
            assert_eq!(&header[16..32], magic.as_slice());
            assert_eq!(
                u32::from_le_bytes(header[32..36].try_into().unwrap()),
                ENVELOPE_HEADER_BLOCKS
            );
            assert!(header[reserved_start..].iter().all(|byte| *byte == 0));

            let decoded = decode_record(&header, 0).unwrap();
            assert_eq!(decoded.len, ENVELOPE_HEADER_LEN);
            assert!(matches!(
                decoded.content,
                PileRecordContent::Collection { record: decoded } if decoded == record
            ));

            let mut nonzero_padding = header;
            nonzero_padding[reserved_start] = 1;
            assert!(matches!(
                decode_record(&nonzero_padding, 0),
                Err(ReadError::CorruptPile { valid_length: 0 })
            ));
        }
    }

    #[test]
    fn every_current_write_path_uses_the_generic_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "all-enveloped.pile");
        let mut pile = Pile::open(&path).unwrap();

        let blob_data = vec![0x42; ENVELOPE_BLOCK_LEN + 1];
        let blob = pile
            .put::<UnknownBlob, _>(Bytes::from_source(blob_data.clone()))
            .unwrap();

        let branch_id = Id::new([1; 16]).unwrap();
        let branch_head = Inline::<Handle<SimpleArchive>>::new([2; 32]);
        assert!(matches!(
            pile.update(branch_id, None, Some(branch_head)).unwrap(),
            PushResult::Success()
        ));
        assert!(matches!(
            pile.update(branch_id, Some(branch_head), None).unwrap(),
            PushResult::Success()
        ));

        let wanted = Inline::<Handle<UnknownBlob>>::new([5; 32]);
        pile.want(WantRequest::blob(wanted)).unwrap();
        pile.unwant(WantRequest::blob(wanted)).unwrap();

        let collection_records = collection_test_records();
        for record in &collection_records {
            pile.insert(*record).unwrap();
        }
        let collection_gossip = collection_test_gossips()[0];
        pile.gossip(collection_gossip).unwrap();
        pile.close().unwrap();

        let expected = [
            (MAGIC_MARKER_BLOB_V3, 3u32),
            (MAGIC_MARKER_BRANCH_V3, 1),
            (MAGIC_MARKER_BRANCH_TOMBSTONE_V3, 1),
            (MAGIC_MARKER_WEAK_PIN_V3, 1),
            (MAGIC_MARKER_WEAK_UNPIN_V3, 1),
            (MAGIC_MARKER_COLLECTION_COMMIT_V4, 1),
            (MAGIC_MARKER_COLLECTION_MERGE_V4, 1),
            (MAGIC_MARKER_COLLECTION_DERIVE_V4, 1),
            (MAGIC_MARKER_COLLECTION_GOSSIP_V1, 1),
        ];
        let mut records = PileRecords::open(&path).unwrap();
        let decoded = (&mut records).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(decoded.len(), expected.len());
        for (record, (kind, blocks)) in decoded.iter().zip(expected) {
            let raw = &records.bytes()[record.offset..record.offset + record.len];
            assert_eq!(&raw[..16], MAGIC_MARKER_ENVELOPE.as_slice());
            assert_eq!(&raw[16..32], kind.as_slice());
            assert_eq!(u32::from_le_bytes(raw[32..36].try_into().unwrap()), blocks);
            assert_eq!(record.len, blocks as usize * ENVELOPE_BLOCK_LEN);
        }
        let blob_raw = &records.bytes()[decoded[0].offset..decoded[0].offset + decoded[0].len];
        assert_eq!(&blob_raw[32..36], &3u32.to_le_bytes());
        assert_eq!(&blob_raw[44..52], &(blob_data.len() as u64).to_le_bytes());

        let mut reopened = Pile::open(&path).unwrap();
        let fetched: Blob<UnknownBlob> = reopened.reader().unwrap().get(blob).unwrap();
        assert_eq!(fetched.bytes.as_ref(), blob_data);
        assert_eq!(reopened.head(branch_id).unwrap(), None);
        assert!(reopened.wants().unwrap().next().is_none());
        assert_eq!(
            reopened
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            sorted_collection_records(collection_records)
        );
        assert_eq!(
            reopened
                .gossips()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![collection_gossip]
        );
        reopened.close().unwrap();
    }

    #[test]
    fn envelope_numeric_fields_are_canonical_little_endian() {
        let header = BlobHeaderEnvelope::new(
            0x0102_0304,
            0x1112_1314_1516_1718,
            0x2122_2324_2526_2728,
            collection_test_hash(9),
        );
        let bytes = header.as_bytes();
        assert_eq!(&bytes[32..36], &0x0102_0304u32.to_le_bytes());
        assert_eq!(&bytes[36..44], &0x1112_1314_1516_1718u64.to_le_bytes());
        assert_eq!(&bytes[44..52], &0x2122_2324_2526_2728u64.to_le_bytes());

        assert_eq!(envelope_blocks_for_payload(0), Some(1));
        assert_eq!(envelope_blocks_for_payload(255), Some(2));
        assert_eq!(envelope_blocks_for_payload(256), Some(2));
        assert_eq!(envelope_blocks_for_payload(257), Some(3));
        #[cfg(target_pointer_width = "64")]
        {
            let largest_payload = (u32::MAX as usize - 1) * ENVELOPE_BLOCK_LEN;
            assert_eq!(envelope_blocks_for_payload(largest_payload), Some(u32::MAX));
            assert_eq!(envelope_blocks_for_payload(largest_payload + 1), None);
        }
        assert_eq!(envelope_blocks_for_payload(usize::MAX), None);
    }

    #[test]
    fn legacy_v4_collection_headers_remain_readable_byte_exact() {
        for record in collection_test_records() {
            let mut header = [0u8; V3_HEADER_LEN];
            match record {
                CollectionRecord::Commit(commit) => {
                    let (signature_r, signature_s) = commit.signature();
                    header.copy_from_slice(
                        CollectionCommitHeaderV4 {
                            magic_marker: MAGIC_MARKER_COLLECTION_COMMIT_V4,
                            collection: commit.collection().raw,
                            data: commit.data().raw,
                            metadata: commit.metadata().raw,
                            public_key: commit.public_key().raw,
                            signature_r: signature_r.raw,
                            signature_s: signature_s.raw,
                            reserved: [0; 48],
                        }
                        .as_bytes(),
                    );
                }
                CollectionRecord::Merge(merge) => {
                    let (low, high) = merge.inputs();
                    header.copy_from_slice(
                        CollectionMergeHeaderV4 {
                            magic_marker: MAGIC_MARKER_COLLECTION_MERGE_V4,
                            collection: merge.collection().raw,
                            low: low.raw,
                            high: high.raw,
                            result: merge.result().raw,
                            reserved: [0; 112],
                        }
                        .as_bytes(),
                    );
                }
                CollectionRecord::Derive(derive) => {
                    let (input, output) = derive.mapping();
                    header.copy_from_slice(
                        CollectionDeriveHeaderV4 {
                            magic_marker: MAGIC_MARKER_COLLECTION_DERIVE_V4,
                            source: derive.source().raw,
                            target: derive.target().raw,
                            input: input.raw,
                            output: output.raw,
                            reserved: [0; 112],
                        }
                        .as_bytes(),
                    );
                }
            }

            let original = header;
            let decoded = decode_record(&header, 0).unwrap();
            assert_eq!(decoded.len, V3_HEADER_LEN);
            assert!(matches!(
                decoded.content,
                PileRecordContent::Collection { record: decoded } if decoded == record
            ));
            assert_eq!(header, original);
        }
    }

    #[test]
    fn opaque_envelopes_are_raw_visible_and_writers_cross_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "opaque-crossing.pile");
        let header_only = test_envelope_bytes(TEST_UNKNOWN_KIND_A, 1, ENVELOPE_BLOCK_LEN);
        let multi_block = test_envelope_bytes(TEST_UNKNOWN_KIND_B, 2, 2 * ENVELOPE_BLOCK_LEN);

        // Hold an already-refreshed writer while another descriptor appends
        // both future record kinds. Its next write must refresh across both.
        let mut pile = Pile::open(&path).unwrap();
        pile.refresh().unwrap();
        {
            let mut external = OpenOptions::new().append(true).open(&path).unwrap();
            external.write_all(&header_only).unwrap();
            external.write_all(&multi_block).unwrap();
            external.sync_all().unwrap();
        }
        let known_payload = b"known after opaque".to_vec();
        let known = pile
            .put::<UnknownBlob, _>(Bytes::from_source(known_payload.clone()))
            .unwrap();
        let branch_id = Id::new([9; 16]).unwrap();
        pile.update(branch_id, None, Some(known.transmute()))
            .unwrap();
        pile.close().unwrap();

        let mut records = PileRecords::open(&path).unwrap();
        let decoded = (&mut records).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(decoded.len(), 4);
        assert!(matches!(
            decoded[0],
            PileRecord {
                len: ENVELOPE_BLOCK_LEN,
                content: PileRecordContent::Opaque {
                    kind: TEST_UNKNOWN_KIND_A
                },
                ..
            }
        ));
        assert!(matches!(
            decoded[1],
            PileRecord {
                len,
                content: PileRecordContent::Opaque { kind: TEST_UNKNOWN_KIND_B },
                ..
            } if len == 2 * ENVELOPE_BLOCK_LEN
        ));
        assert!(matches!(decoded[2].content, PileRecordContent::Blob { .. }));
        assert!(matches!(
            decoded[3].content,
            PileRecordContent::Branch { .. }
        ));
        assert_eq!(&records.bytes()[..ENVELOPE_BLOCK_LEN], &header_only);
        assert_eq!(
            &records.bytes()[ENVELOPE_BLOCK_LEN..3 * ENVELOPE_BLOCK_LEN],
            &multi_block
        );

        let mut reopened = Pile::open(&path).unwrap();
        reopened.refresh().unwrap();
        assert_eq!(reopened.opaque_records, 2);
        let fetched: Blob<UnknownBlob> = reopened.reader().unwrap().get(known).unwrap();
        assert_eq!(fetched.bytes.as_ref(), known_payload);
        assert_eq!(reopened.head(branch_id).unwrap(), Some(known.transmute()));
        reopened.close().unwrap();
    }

    #[test]
    fn retired_local_cell_records_are_opaque_migration_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "retired-local-cells.pile");
        let mut unenveloped = [0u8; V3_HEADER_LEN];
        unenveloped[..16].copy_from_slice(&MAGIC_MARKER_LOCAL_CELL_V3);
        let enveloped = test_envelope_bytes(
            MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3,
            ENVELOPE_HEADER_BLOCKS,
            ENVELOPE_HEADER_LEN,
        );
        append_test_bytes(&path, &unenveloped);
        append_test_bytes(&path, &enveloped);

        let mut records = PileRecords::open(&path).unwrap();
        let decoded = (&mut records).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(matches!(
            decoded[0].content,
            PileRecordContent::Opaque {
                kind: MAGIC_MARKER_LOCAL_CELL_V3
            }
        ));
        assert!(matches!(
            decoded[1].content,
            PileRecordContent::Opaque {
                kind: MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3
            }
        ));
        assert_eq!(decoded[0].len, V3_HEADER_LEN);
        assert_eq!(decoded[1].len, ENVELOPE_HEADER_LEN);

        let mut pile = Pile::open(&path).unwrap();
        pile.refresh().unwrap();
        assert_eq!(pile.opaque_record_count().unwrap(), 2);
        pile.close().unwrap();
    }

    #[test]
    fn truncated_retired_unenveloped_cell_is_corrupt_not_an_applied_record() {
        for kind in [
            MAGIC_MARKER_LOCAL_CELL_V3,
            MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3,
        ] {
            for len in [16usize, 17, V3_HEADER_LEN - 1] {
                let mut bytes = vec![0u8; len];
                bytes[..16].copy_from_slice(&kind);
                assert!(matches!(
                    decode_record(&bytes, 37),
                    Err(ReadError::CorruptPile { valid_length: 37 })
                ));
            }
        }
    }

    #[test]
    fn retired_enveloped_cell_requires_its_historical_fixed_span() {
        for kind in [
            MAGIC_MARKER_LOCAL_CELL_V3,
            MAGIC_MARKER_LOCAL_CELL_TOMBSTONE_V3,
        ] {
            let bytes = test_envelope_bytes(kind, 2, 2 * ENVELOPE_BLOCK_LEN);
            assert!(matches!(
                decode_record(&bytes, 41),
                Err(ReadError::CorruptPile { valid_length: 41 })
            ));
        }
    }

    #[test]
    fn opaque_projection_preserves_lww_order_for_branches_and_wants() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "opaque-lww.pile");
        let opaque = test_envelope_bytes(TEST_UNKNOWN_KIND_A, 1, ENVELOPE_BLOCK_LEN);
        let mut pile = Pile::open(&path).unwrap();

        let branch_cleared = Id::new([11; 16]).unwrap();
        let branch_restored = Id::new([12; 16]).unwrap();
        let branch_head = Inline::<Handle<SimpleArchive>>::new([13; 32]);
        pile.update(branch_cleared, None, Some(branch_head))
            .unwrap();
        append_test_bytes(&path, &opaque);
        pile.update(branch_cleared, Some(branch_head), None)
            .unwrap();
        append_test_bytes(
            &path,
            BranchTombstoneHeaderEnvelope::new(branch_restored).as_bytes(),
        );
        append_test_bytes(&path, &opaque);
        pile.update(branch_restored, None, Some(branch_head))
            .unwrap();

        let want_retracted = Inline::<Handle<UnknownBlob>>::new([17; 32]);
        let want_restored = Inline::<Handle<UnknownBlob>>::new([18; 32]);
        pile.want(WantRequest::blob(want_retracted)).unwrap();
        append_test_bytes(&path, &opaque);
        pile.unwant(WantRequest::blob(want_retracted)).unwrap();
        append_test_bytes(
            &path,
            WantHeaderEnvelope::new(want_restored, false).as_bytes(),
        );
        append_test_bytes(&path, &opaque);
        pile.want(WantRequest::blob(want_restored)).unwrap();
        pile.close().unwrap();

        let mut reopened = Pile::open(&path).unwrap();
        reopened.refresh().unwrap();
        assert_eq!(reopened.opaque_record_count().unwrap(), 4);
        assert_eq!(reopened.head(branch_cleared).unwrap(), None);
        assert_eq!(reopened.head(branch_restored).unwrap(), Some(branch_head));
        let wants = reopened
            .wants()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(wants, vec![WantRequest::blob(want_restored)]);
        reopened.close().unwrap();
    }

    #[test]
    fn legacy_v3_and_enveloped_piles_concatenate_without_reframing() {
        let dir = tempfile::tempdir().unwrap();
        let legacy_path = fresh_empty_pile_path(&dir, "legacy-v3.pile");
        let current_path = fresh_empty_pile_path(&dir, "enveloped.pile");
        let merged_path = dir.path().join("mixed-cat.pile");

        let legacy_payload = b"legacy V3".to_vec();
        let legacy_handle =
            Blob::<UnknownBlob>::new(Bytes::from_source(legacy_payload.clone())).get_handle();
        append_v3_blob_candidate(&legacy_path, legacy_handle.into(), &legacy_payload, 17);

        let current_payload = b"current envelope".to_vec();
        let mut current = Pile::open(&current_path).unwrap();
        let current_handle = current
            .put::<UnknownBlob, _>(Bytes::from_source(current_payload.clone()))
            .unwrap();
        current.close().unwrap();

        let mut merged = std::fs::read(&legacy_path).unwrap();
        merged.extend_from_slice(&std::fs::read(&current_path).unwrap());
        std::fs::write(&merged_path, merged).unwrap();

        let mut pile = Pile::open(&merged_path).unwrap();
        pile.refresh().unwrap();
        let reader = pile.reader().unwrap();
        let legacy: Blob<UnknownBlob> = reader.get(legacy_handle).unwrap();
        let current: Blob<UnknownBlob> = reader.get(current_handle).unwrap();
        assert_eq!(legacy.bytes.as_ref(), legacy_payload);
        assert_eq!(current.bytes.as_ref(), current_payload);
        drop(reader);
        pile.close().unwrap();

        let mut records = PileRecords::open(&merged_path).unwrap();
        let decoded = (&mut records).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(
            &records.bytes()[decoded[0].offset..decoded[0].offset + 16],
            MAGIC_MARKER_BLOB_V3.as_slice()
        );
        assert_eq!(
            &records.bytes()[decoded[1].offset..decoded[1].offset + 16],
            MAGIC_MARKER_ENVELOPE.as_slice()
        );
    }

    #[test]
    fn lazy_pile_reads_reopens_and_appends_across_opaque_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "lazy-opaque.pile");
        std::fs::write(
            &path,
            test_envelope_bytes(TEST_UNKNOWN_KIND_A, 1, ENVELOPE_BLOCK_LEN),
        )
        .unwrap();

        let first_payload = b"first lazy blob".to_vec();
        let mut seed = Pile::open(&path).unwrap();
        let first = seed
            .put::<UnknownBlob, _>(Bytes::from_source(first_payload.clone()))
            .unwrap();
        seed.close().unwrap();

        let mut lazy: Lazy<Pile> = Lazy::new(Pile::open(&path).unwrap());
        let reader = BlobStore::reader(&mut lazy).unwrap();
        let first_read: Blob<UnknownBlob> = reader.get(first).unwrap();
        assert_eq!(first_read.bytes.as_ref(), first_payload);
        drop(reader);

        {
            let mut external = OpenOptions::new().append(true).open(&path).unwrap();
            external
                .write_all(&test_envelope_bytes(
                    TEST_UNKNOWN_KIND_B,
                    2,
                    2 * ENVELOPE_BLOCK_LEN,
                ))
                .unwrap();
        }
        let second_payload = b"second lazy blob".to_vec();
        let second = BlobStorePut::put::<UnknownBlob, _>(
            &mut lazy,
            Bytes::from_source(second_payload.clone()),
        )
        .unwrap();
        lazy.into_store().close().unwrap();

        let mut reopened: Lazy<Pile> = Lazy::new(Pile::open(&path).unwrap());
        let reader = BlobStore::reader(&mut reopened).unwrap();
        let first_read: Blob<UnknownBlob> = reader.get(first).unwrap();
        let second_read: Blob<UnknownBlob> = reader.get(second).unwrap();
        assert_eq!(first_read.bytes.as_ref(), first_payload);
        assert_eq!(second_read.bytes.as_ref(), second_payload);
        drop(reader);
        assert_eq!(reopened.store().opaque_record_count().unwrap(), 2);
        reopened.into_store().close().unwrap();
    }

    #[test]
    fn envelope_span_rejects_zero_maximum_truncation_and_kind_mismatch() {
        let complete_header = test_envelope_bytes(TEST_UNKNOWN_KIND_A, 1, ENVELOPE_HEADER_LEN);
        for truncated_at in [0usize, 1, 15, 16, 31, 32, 35, 36, 255] {
            assert!(matches!(
                decode_record(&complete_header[..truncated_at], 11),
                Err(ReadError::CorruptPile { valid_length: 11 })
            ));
        }

        for malformed in [
            test_envelope_bytes(TEST_UNKNOWN_KIND_A, 0, ENVELOPE_BLOCK_LEN),
            test_envelope_bytes(TEST_UNKNOWN_KIND_A, u32::MAX, ENVELOPE_BLOCK_LEN),
            test_envelope_bytes(TEST_UNKNOWN_KIND_A, 2, 2 * ENVELOPE_BLOCK_LEN - 1),
            test_envelope_bytes(MAGIC_MARKER_BRANCH_V3, 2, 2 * ENVELOPE_BLOCK_LEN),
        ] {
            assert!(matches!(
                decode_record(&malformed, 17),
                Err(ReadError::CorruptPile { valid_length: 17 })
            ));
        }

        let prefix_only = test_envelope_bytes(TEST_UNKNOWN_KIND_A, 1, 36);
        assert!(matches!(
            decode_record(&prefix_only, 23),
            Err(ReadError::CorruptPile { valid_length: 23 })
        ));

        let hash = collection_test_hash(7);
        let mut wrong_blob_span = vec![0u8; 2 * ENVELOPE_BLOCK_LEN];
        wrong_blob_span[..ENVELOPE_HEADER_LEN].copy_from_slice(
            BlobHeaderEnvelope::new(1, 42, (ENVELOPE_BLOCK_LEN + 1) as u64, hash).as_bytes(),
        );
        assert!(matches!(
            decode_record(&wrong_blob_span, 29),
            Err(ReadError::CorruptPile { valid_length: 29 })
        ));
    }

    #[test]
    fn amputation_crosses_complete_opaque_and_truncates_torn_opaque_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "opaque-amputation.pile");
        let complete = test_envelope_bytes(TEST_UNKNOWN_KIND_A, 2, 2 * ENVELOPE_BLOCK_LEN);
        let torn = test_envelope_bytes(TEST_UNKNOWN_KIND_B, 2, ENVELOPE_BLOCK_LEN);
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(&complete).unwrap();
            file.write_all(&torn).unwrap();
        }

        let mut pile = Pile::open(&path).unwrap();
        assert!(matches!(
            pile.refresh(),
            Err(ReadError::CorruptPile { valid_length })
                if valid_length == complete.len()
        ));
        pile.amputate().unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            complete.len() as u64
        );
        pile.refresh().unwrap();
        assert_eq!(pile.opaque_records, 1);
        pile.close().unwrap();

        let mut records = PileRecords::open(&path).unwrap();
        let only = records.next().unwrap().unwrap();
        assert_eq!(only.len, complete.len());
        assert!(matches!(
            only.content,
            PileRecordContent::Opaque {
                kind: TEST_UNKNOWN_KIND_A
            }
        ));
        assert!(records.next().is_none());
    }

    #[test]
    fn amputation_at_refuses_a_stale_boundary_before_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "boundary-guarded-amputation.pile");
        let complete = test_envelope_bytes(TEST_UNKNOWN_KIND_A, 2, 2 * ENVELOPE_BLOCK_LEN);
        let torn = test_envelope_bytes(TEST_UNKNOWN_KIND_B, 2, ENVELOPE_BLOCK_LEN);
        let mut bytes = complete.clone();
        bytes.extend_from_slice(&torn);
        std::fs::write(&path, &bytes).unwrap();

        let mut pile = Pile::open(&path).unwrap();
        assert!(matches!(
            pile.amputate_at(0),
            Err(ReadError::CorruptPile { valid_length })
                if valid_length == complete.len()
        ));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        assert!(pile.amputate_at(complete.len()).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), complete);
        pile.close().unwrap();
    }

    #[test]
    fn opaque_records_refuse_pile_and_yard_retention_before_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "opaque-source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "opaque-destination.pile");
        std::fs::write(
            &source_path,
            test_envelope_bytes(TEST_UNKNOWN_KIND_A, 1, ENVELOPE_BLOCK_LEN),
        )
        .unwrap();

        let mut source = Pile::open(&source_path).unwrap();
        let retained = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"possibly owned".to_vec()))
            .unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();
        destination
            .put::<UnknownBlob, _>(Bytes::from_source(b"sentinel".to_vec()))
            .unwrap();
        destination.flush().unwrap();
        let destination_before = std::fs::read(&destination_path).unwrap();

        assert!(matches!(
            source.rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            ),
            Err(PileRewriteError::OpaqueRecords { count: 1 })
        ));
        assert_eq!(
            std::fs::read(&destination_path).unwrap(),
            destination_before
        );
        let fetched: Blob<UnknownBlob> = source.reader().unwrap().get(retained).unwrap();
        assert_eq!(fetched.bytes.as_ref(), b"possibly owned");
        destination.close().unwrap();
        source.close().unwrap();

        // The fence is Yard-wide: an opaque record in the young generation
        // may own a known blob physically resident only in an older one.
        let old_path = fresh_empty_pile_path(&dir, "opaque-owned-old.pile");
        let mut old = Pile::open(&old_path).unwrap();
        let cross_generation = old
            .put::<UnknownBlob, _>(Bytes::from_source(
                b"possibly owned across generations".to_vec(),
            ))
            .unwrap();
        old.close().unwrap();
        let young_before = std::fs::read(&source_path).unwrap();
        let old_before = std::fs::read(&old_path).unwrap();
        let mut yard = Yard::open([&source_path, &old_path], YardConfig::default()).unwrap();
        assert!(matches!(
            yard.collect(&RetentionRoots::new()),
            Err(YardCollectError::OpaqueRecords { count: 1 })
        ));
        assert!(matches!(
            yard.compact(&RetentionRoots::new()),
            Err(YardCollectError::OpaqueRecords { count: 1 })
        ));
        assert!(matches!(
            yard.reclaim(),
            Err(YardReclaimError::OpaqueRecords { count: 1 })
        ));
        assert_eq!(std::fs::read(&source_path).unwrap(), young_before);
        assert_eq!(std::fs::read(&old_path).unwrap(), old_before);
        let fetched: Blob<UnknownBlob> = yard.reader().unwrap().get(retained).unwrap();
        assert_eq!(fetched.bytes.as_ref(), b"possibly owned");
        let fetched: Blob<UnknownBlob> = yard.reader().unwrap().get(cross_generation).unwrap();
        assert_eq!(fetched.bytes.as_ref(), b"possibly owned across generations");
        yard.close().unwrap();
    }

    #[test]
    fn legacy_v3_collection_headers_are_inert_and_preserved_by_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "legacy-source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "legacy-destination.pile");
        let expected = legacy_collection_test_headers();

        {
            let mut file = OpenOptions::new().append(true).open(&source_path).unwrap();
            for (_, header) in &expected {
                file.write_all(header).unwrap();
            }
        }

        assert_eq!(
            legacy_collection_headers_at(&source_path)
                .into_iter()
                .collect::<BTreeSet<_>>(),
            expected.iter().copied().collect::<BTreeSet<_>>()
        );

        let mut malformed = expected[0].1;
        malformed[64] = 1;
        assert!(matches!(
            decode_record(&malformed, 0),
            Err(ReadError::CorruptPile { valid_length: 0 })
        ));

        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();
        assert_eq!(source.records().unwrap().count(), 0);

        let stats = source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();
        assert_eq!(stats.retained_blobs, 0);
        assert_eq!(destination.records().unwrap().count(), 0);

        let rewritten = legacy_collection_headers_at(&destination_path);
        assert_eq!(
            rewritten.into_iter().collect::<BTreeSet<_>>(),
            expected.iter().copied().collect::<BTreeSet<_>>()
        );

        let once = std::fs::metadata(&destination_path).unwrap().len();
        source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();
        assert_eq!(std::fs::metadata(&destination_path).unwrap().len(), once);

        destination.close().unwrap();
        source.close().unwrap();
    }

    #[test]
    fn yard_reclaim_preserves_inert_legacy_v3_collection_headers() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "legacy-yard.pile");
        let expected = legacy_collection_test_headers();

        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            for (_, header) in &expected {
                file.write_all(header).unwrap();
            }
        }

        let mut yard = Yard::open([&path], YardConfig::default()).unwrap();
        yard.reclaim().unwrap();
        yard.close().unwrap();

        assert_eq!(
            legacy_collection_headers_at(&path)
                .into_iter()
                .collect::<BTreeSet<_>>(),
            expected.into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn native_collection_records_replay_in_intrinsic_id_order_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "collections.pile");
        let records = collection_test_records();
        let expected = sorted_collection_records(records.clone());

        let mut pile = Pile::open(&path).unwrap();
        for record in records {
            pile.insert(record).unwrap();
        }
        pile.close().unwrap();

        let mut reopened = Pile::open(&path).unwrap();
        let actual = reopened
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual, expected);
        reopened.close().unwrap();
    }

    #[test]
    fn native_collection_record_insert_is_physically_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "idempotent.pile");
        let record = collection_test_records()[0];
        let mut pile = Pile::open(&path).unwrap();

        pile.insert(record).unwrap();
        let once = std::fs::metadata(&path).unwrap().len();
        pile.insert(record).unwrap();
        let twice = std::fs::metadata(&path).unwrap().len();

        assert_eq!(once, ENVELOPE_HEADER_LEN as u64);
        assert_eq!(twice, once);
        assert_eq!(pile.records().unwrap().count(), 1);
        pile.close().unwrap();
    }

    #[test]
    fn native_collection_records_cat_as_an_order_independent_set_union() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = fresh_empty_pile_path(&dir, "a.pile");
        let path_b = fresh_empty_pile_path(&dir, "b.pile");
        let path_ab = dir.path().join("ab.pile");
        let path_ba = dir.path().join("ba.pile");
        let records = collection_test_records();

        let mut a = Pile::open(&path_a).unwrap();
        a.insert(records[0]).unwrap();
        a.insert(records[1]).unwrap();
        a.close().unwrap();
        let mut b = Pile::open(&path_b).unwrap();
        b.insert(records[0]).unwrap();
        b.insert(records[2]).unwrap();
        b.close().unwrap();

        let bytes_a = std::fs::read(&path_a).unwrap();
        let bytes_b = std::fs::read(&path_b).unwrap();
        let mut ab = bytes_a.clone();
        ab.extend_from_slice(&bytes_b);
        std::fs::write(&path_ab, ab).unwrap();
        let mut ba = bytes_b;
        ba.extend_from_slice(&bytes_a);
        std::fs::write(&path_ba, ba).unwrap();

        let expected = sorted_collection_records(records);
        for path in [&path_ab, &path_ba] {
            let mut pile = Pile::open(path).unwrap();
            let actual = pile
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(actual, expected);
            pile.close().unwrap();
        }
    }

    #[test]
    fn collection_selection_survives_insert_reopen_and_concatenation() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = fresh_empty_pile_path(&dir, "selection-a.pile");
        let path_b = fresh_empty_pile_path(&dir, "selection-b.pile");
        let path_ab = dir.path().join("selection-ab.pile");
        let path_ba = dir.path().join("selection-ba.pile");
        let source = collection_test_collection(1);
        let target = collection_test_collection(2);
        let input = collection_test_hash(8);
        let records = collection_test_records();
        let first = records[2];
        let conflicting = CollectionRecord::Derive(CollectionDerive::new(
            source,
            target,
            input,
            collection_test_hash(10),
        ));
        let unrelated = CollectionRecord::Derive(CollectionDerive::new(
            source,
            collection_test_collection(3),
            input,
            collection_test_hash(11),
        ));
        let exact = [CollectionRecordSelector::Operation(WantRequest::derive(
            source, target, input,
        ))]
        .into_iter()
        .collect();

        let mut a = Pile::open(&path_a).unwrap();
        for record in [records[0], records[1], first, unrelated] {
            a.insert(record).unwrap();
        }
        assert_eq!(a.select_records(&exact).unwrap(), vec![first]);
        a.close().unwrap();

        let mut reopened = Pile::open(&path_a).unwrap();
        assert_eq!(reopened.select_records(&exact).unwrap(), vec![first]);
        reopened.close().unwrap();

        let mut b = Pile::open(&path_b).unwrap();
        for record in [first, conflicting] {
            b.insert(record).unwrap();
        }
        b.close().unwrap();

        let bytes_a = std::fs::read(&path_a).unwrap();
        let bytes_b = std::fs::read(&path_b).unwrap();
        let mut ab = bytes_a.clone();
        ab.extend_from_slice(&bytes_b);
        std::fs::write(&path_ab, ab).unwrap();
        let mut ba = bytes_b;
        ba.extend_from_slice(&bytes_a);
        std::fs::write(&path_ba, ba).unwrap();

        let mut expected = vec![first, conflicting];
        expected.sort_unstable_by_key(CollectionRecord::id);
        for path in [&path_ab, &path_ba] {
            let mut pile = Pile::open(path).unwrap();
            assert_eq!(pile.select_records(&exact).unwrap(), expected);
            assert!(!pile.select_records(&exact).unwrap().contains(&unrelated));
            pile.close().unwrap();
        }
    }

    #[test]
    fn collection_gossips_replay_idempotently_and_cat_as_set_union() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = fresh_empty_pile_path(&dir, "gossip-a.pile");
        let path_b = fresh_empty_pile_path(&dir, "gossip-b.pile");
        let path_ab = dir.path().join("gossip-ab.pile");
        let path_ba = dir.path().join("gossip-ba.pile");
        let grants = collection_test_gossips();

        let mut a = Pile::open(&path_a).unwrap();
        a.gossip(grants[0]).unwrap();
        let once = std::fs::metadata(&path_a).unwrap().len();
        a.gossip(grants[0]).unwrap();
        assert_eq!(std::fs::metadata(&path_a).unwrap().len(), once);
        a.close().unwrap();

        let mut b = Pile::open(&path_b).unwrap();
        b.gossip(grants[1]).unwrap();
        b.close().unwrap();

        let bytes_a = std::fs::read(&path_a).unwrap();
        let bytes_b = std::fs::read(&path_b).unwrap();
        let mut ab = bytes_a.clone();
        ab.extend_from_slice(&bytes_b);
        std::fs::write(&path_ab, ab).unwrap();
        let mut ba = bytes_b;
        ba.extend_from_slice(&bytes_a);
        std::fs::write(&path_ba, ba).unwrap();

        let expected = grants.into_iter().collect::<BTreeSet<_>>();
        for path in [&path_ab, &path_ba] {
            let mut pile = Pile::open(path).unwrap();
            let actual = pile
                .gossips()
                .unwrap()
                .collect::<Result<BTreeSet<_>, _>>()
                .unwrap();
            assert_eq!(actual, expected);
            assert!(actual.iter().all(|grant| grant.verify_strict().is_ok()));
            pile.close().unwrap();
        }
    }

    #[test]
    fn native_collection_record_torn_tail_is_detected_and_amputated() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "torn.pile");
        let mut pile = Pile::open(&path).unwrap();
        pile.insert(collection_test_records()[0]).unwrap();
        pile.close().unwrap();

        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((ENVELOPE_HEADER_LEN - 1) as u64)
            .unwrap();
        let mut reopened = Pile::open(&path).unwrap();
        assert!(matches!(
            reopened.refresh(),
            Err(ReadError::CorruptPile { valid_length: 0 })
        ));
        reopened.amputate().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        reopened.close().unwrap();
    }

    #[test]
    fn retained_rewrite_composes_explicit_roots_strong_pins_and_wants() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "destination.pile");
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();

        let legacy_attachment = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"legacy attachment".to_vec()))
            .unwrap();
        let legacy_head = source
            .put::<SimpleArchive, _>(Blob::<SimpleArchive>::new(Bytes::from_source(
                legacy_attachment.raw.to_vec(),
            )))
            .unwrap();
        let pin_id = Id::new([9; 16]).unwrap();
        assert!(matches!(
            source.update(pin_id, None, Some(legacy_head)).unwrap(),
            PushResult::Success()
        ));

        let collection_attachment = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"collection attachment".to_vec()))
            .unwrap();
        let collection_data = source
            .put::<UnknownBlob, _>(Bytes::from_source(collection_attachment.raw.to_vec()))
            .unwrap();

        let obsolete_input = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"obsolete input".to_vec()))
            .unwrap();
        let collection_record = source
            .put::<UnknownBlob, _>(Bytes::from_source(obsolete_input.raw.to_vec()))
            .unwrap();
        let want_target = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"want target".to_vec()))
            .unwrap();
        source.want(WantRequest::blob(want_target)).unwrap();
        let orphan = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"orphan".to_vec()))
            .unwrap();
        source.flush().unwrap();

        let mut explicit = RetentionRoots::new();
        explicit.retain_recursive(collection_data);
        // Record hashes are algebraic descriptions, not ownership edges: the
        // record survives but its obsolete input does not.
        explicit.retain_direct(collection_record);

        let stats = source
            .rewrite_retained_into(&mut destination, &explicit, WantRewritePolicy::Preserve)
            .unwrap();
        assert_eq!(
            stats,
            PileRewriteStats {
                retained_blobs: 5,
                strong_pins: 1,
                wants: 1,
                collection_gossips: 0,
            }
        );

        let reader = destination.reader().unwrap();
        for retained in [
            legacy_attachment,
            legacy_head.transmute(),
            collection_attachment,
            collection_data,
            collection_record,
        ] {
            assert!(reader.get::<Blob<UnknownBlob>, _>(retained).is_ok());
        }
        for collected in [obsolete_input, want_target, orphan] {
            assert!(reader.get::<Blob<UnknownBlob>, _>(collected).is_err());
        }
        assert_eq!(destination.head(pin_id).unwrap(), Some(legacy_head));
        assert_eq!(
            destination
                .wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![WantRequest::blob(want_target)]
        );

        drop(reader);
        destination.close().unwrap();
        source.close().unwrap();
    }

    #[test]
    fn retained_rewrite_ignores_invalid_commit_owned_resident_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "invalid-commit-source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "invalid-commit-destination.pile");
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();

        let forged_data = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"forged ownership data".to_vec()))
            .unwrap();
        let metadata_facts: TribleSet = entity! {
            crate::metadata::tag: collection_test_id(20)
        }
        .into();
        let forged_metadata = source
            .put::<SimpleArchive, _>(metadata_facts.to_blob())
            .unwrap();
        let descriptor = CollectionDescriptor::new(
            collection_test_id(21),
            collection_test_id(22),
            collection_test_id(23),
        );
        let descriptor_handle = source
            .put::<SimpleArchive, _>(CollectionDescriptor::to_blob(&descriptor))
            .unwrap();
        assert_eq!(descriptor_handle, descriptor.handle());
        let invalid = invalidate_collection_commit(CollectionCommit::sign(
            &SigningKey::from_bytes(&[24; 32]),
            descriptor_handle,
            Inline::<Hash<Blake3>>::new(forged_data.raw),
            forged_metadata,
        ));
        let records = vec![CollectionRecord::Commit(invalid)];
        for record in records.iter().copied() {
            source.insert(record).unwrap();
        }
        source.flush().unwrap();

        let stats = source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();
        assert_eq!(stats.retained_blobs, 0);
        assert_eq!(
            destination
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            sorted_collection_records(records),
        );

        let reader = destination.reader().unwrap();
        assert!(matches!(
            reader.get::<Blob<UnknownBlob>, _>(forged_data),
            Err(GetBlobError::BlobNotFound)
        ));
        assert!(matches!(
            reader.get::<Blob<SimpleArchive>, _>(forged_metadata),
            Err(GetBlobError::BlobNotFound)
        ));
        assert!(matches!(
            reader.get::<Blob<SimpleArchive>, _>(descriptor_handle),
            Err(GetBlobError::BlobNotFound)
        ));

        drop(reader);
        destination.close().unwrap();
        source.close().unwrap();
    }

    #[test]
    fn retained_rewrite_preserves_valid_dangling_commit_without_demanding_dependencies() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "dangling-commit-source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "dangling-commit-destination.pile");
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();

        let missing_descriptor = collection_test_collection(25);
        let missing_data = collection_test_hash(28);
        let missing_metadata = Inline::<Handle<SimpleArchive>>::new([29; 32]);
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[30; 32]),
            missing_descriptor,
            missing_data,
            missing_metadata,
        );
        commit.verify_strict().unwrap();
        let records = vec![CollectionRecord::Commit(commit)];
        for record in records.iter().copied() {
            source.insert(record).unwrap();
        }
        source.flush().unwrap();

        let stats = source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();
        assert_eq!(stats.retained_blobs, 0);
        assert_eq!(
            destination
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            sorted_collection_records(records),
        );

        let reader = destination.reader().unwrap();
        assert!(!reader
            .contains_blob(Handle::<UnknownBlob>::from_hash(missing_data))
            .unwrap());
        assert!(!reader.contains_blob(missing_metadata).unwrap());
        assert!(!reader.contains_blob(missing_descriptor).unwrap());

        drop(reader);
        destination.close().unwrap();
        source.close().unwrap();
    }

    #[test]
    fn retained_rewrite_still_fails_loud_for_missing_explicit_root() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "missing-explicit-source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "missing-explicit-destination.pile");
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();
        let missing = Inline::<Handle<UnknownBlob>>::new([31; 32]);
        let mut explicit = RetentionRoots::new();
        explicit.retain_recursive(missing);

        let error = source
            .rewrite_retained_into(&mut destination, &explicit, WantRewritePolicy::Drop)
            .unwrap_err();
        assert!(matches!(
            error,
            PileRewriteError::Transfer(crate::repo::TransferError::Load(
                GetBlobError::BlobNotFound
            ))
        ));

        destination.close().unwrap();
        source.close().unwrap();
    }

    #[test]
    fn retained_rewrite_preserves_native_records_and_commit_owned_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = fresh_empty_pile_path(&dir, "collection-source.pile");
        let destination_path = fresh_empty_pile_path(&dir, "collection-destination.pile");
        let mut source = Pile::open(&source_path).unwrap();
        let mut destination = Pile::open(&destination_path).unwrap();

        let attachment = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"owned attachment".to_vec()))
            .unwrap();
        let data = source
            .put::<UnknownBlob, _>(Bytes::from_source(attachment.raw.to_vec()))
            .unwrap();
        let metadata = source
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        assert_eq!(metadata, empty_metadata_handle());
        let orphan = source
            .put::<UnknownBlob, _>(Bytes::from_source(b"unowned".to_vec()))
            .unwrap();

        let descriptor = CollectionDescriptor::new(
            collection_test_id(10),
            collection_test_id(11),
            collection_test_id(12),
        );
        let descriptor_handle = source
            .put::<SimpleArchive, _>(CollectionDescriptor::to_blob(&descriptor))
            .unwrap();
        assert_eq!(descriptor_handle, descriptor.handle());
        let key = SigningKey::from_bytes(&[13; 32]);
        let commit = CollectionCommit::sign(
            &key,
            descriptor_handle,
            Inline::<Hash<Blake3>>::new(data.raw),
            metadata,
        );
        commit.verify_strict().unwrap();
        let records = vec![
            CollectionRecord::Commit(commit),
            CollectionRecord::Merge(CollectionMerge::new(
                descriptor_handle,
                collection_test_hash(14),
                collection_test_hash(15),
                collection_test_hash(16),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                descriptor_handle,
                collection_test_collection(17),
                collection_test_hash(16),
                collection_test_hash(18),
            )),
        ];
        for record in records.iter().copied() {
            source.insert(record).unwrap();
        }
        source.flush().unwrap();

        let stats = source
            .rewrite_retained_into(
                &mut destination,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .unwrap();
        assert_eq!(stats.retained_blobs, 4);

        let actual_records = destination
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual_records, sorted_collection_records(records));
        let reader = destination.reader().unwrap();
        for retained in [
            attachment,
            data,
            metadata.transmute(),
            descriptor_handle.transmute(),
        ] {
            assert!(reader.get::<Blob<UnknownBlob>, _>(retained).is_ok());
        }
        assert!(reader.get::<Blob<UnknownBlob>, _>(orphan).is_err());

        drop(reader);
        destination.close().unwrap();
        source.close().unwrap();
    }

    fn append_v3_blob_candidate(
        path: &Path,
        hash: Inline<Hash<Blake3>>,
        payload: &[u8],
        timestamp: u64,
    ) -> usize {
        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        let record_offset = file.metadata().unwrap().len() as usize;
        let header = BlobHeaderV3::new(timestamp, payload.len() as u64, hash);
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
        file.write_all(&vec![0; block_post_pad(payload.len())])
            .unwrap();
        file.sync_all().unwrap();
        record_offset
    }

    #[test]
    fn index_entry_is_one_machine_word() {
        assert_eq!(
            std::mem::size_of::<IndexEntry>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn payload_validation_matches_and_rejects_around_parallel_threshold() {
        for strategy in [
            ValidationStrategy::Serial,
            ValidationStrategy::ParallelIfLarge,
        ] {
            for len in [
                PARALLEL_BLAKE3_THRESHOLD - 1,
                PARALLEL_BLAKE3_THRESHOLD,
                PARALLEL_BLAKE3_THRESHOLD + 1,
            ] {
                let bytes = Bytes::from_source(
                    (0..len)
                        .map(|position| (position.wrapping_mul(131) % 251) as u8)
                        .collect::<Vec<_>>(),
                );
                let expected = Hash::<Blake3>::digest(&bytes);

                assert!(matches!(
                    compute_validation_state(&bytes, &expected, strategy),
                    ValidationState::Validated
                ));

                let mut wrong = expected;
                wrong.raw[0] ^= 1;
                assert!(matches!(
                    compute_validation_state(&bytes, &wrong, strategy),
                    ValidationState::Invalid
                ));
            }
        }
    }

    #[test]
    fn payload_validation_keeps_the_first_cached_result() {
        let bytes = Bytes::from_source(vec![0x5A; PARALLEL_BLAKE3_THRESHOLD]);
        let expected = Hash::<Blake3>::digest(&bytes);
        let cache = ValidationCache::default();
        assert!(matches!(
            cache.state(7, &bytes, &expected, ValidationStrategy::Serial),
            ValidationState::Validated
        ));

        let mut wrong = expected;
        wrong.raw[0] ^= 1;
        assert!(matches!(
            cache.state(7, &bytes, &wrong, ValidationStrategy::ParallelIfLarge),
            ValidationState::Validated
        ));
        assert_eq!(cache.states.lock().unwrap().len(), 1);
    }

    #[test]
    fn replay_keeps_validation_cache_sparse_and_readers_share_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "sparse-validation.pile");

        let (first, second) = {
            let mut writer = Pile::open(&path).unwrap();
            let first = writer
                .put::<UnknownBlob, _>(Bytes::from_source(b"first".to_vec()))
                .unwrap();
            let second = writer
                .put::<UnknownBlob, _>(Bytes::from_source(b"second".to_vec()))
                .unwrap();
            writer.close().unwrap();
            (first, second)
        };

        let mut replay = Pile::open(&path).unwrap();
        replay.refresh().unwrap();
        assert!(replay.validations.states.lock().unwrap().is_empty());

        let reader = replay.reader().unwrap();
        let cloned = reader.clone();
        assert!(Arc::ptr_eq(
            &reader.validations.states,
            &cloned.validations.states
        ));
        let _: Blob<UnknownBlob> = reader.get(first).unwrap();
        assert_eq!(replay.validations.states.lock().unwrap().len(), 1);
        let _: Blob<UnknownBlob> = cloned.get(first).unwrap();
        assert_eq!(replay.validations.states.lock().unwrap().len(), 1);
        assert!(!replay
            .validations
            .states
            .lock()
            .unwrap()
            .contains_key(&replay.blobs.get(&second.raw).unwrap().record_offset));

        drop(reader);
        drop(cloned);
        replay.close().unwrap();
    }

    #[test]
    fn duplicate_validation_is_isolated_by_record_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "offset-validation.pile");
        let payload = b"target";
        let handle = Blob::<UnknownBlob>::new(Bytes::from_source(payload.to_vec())).get_handle();
        let hash: Inline<Hash<Blake3>> = handle.into();

        let first = append_v3_blob_candidate(&path, hash, b"bad-01", 1);
        let second = append_v3_blob_candidate(&path, hash, payload, 2);
        let _third = append_v3_blob_candidate(&path, hash, b"bad-03", 3);

        let mut pile = Pile::open(&path).unwrap();
        pile.refresh().unwrap();
        assert_eq!(pile.blobs.get(&hash.raw).unwrap().record_offset, second);
        let states = pile.validations.states.lock().unwrap();
        assert!(matches!(states.get(&first), Some(ValidationState::Invalid)));
        assert!(matches!(
            states.get(&second),
            Some(ValidationState::Validated)
        ));
        assert_eq!(states.len(), 2);
        drop(states);

        let reader = pile.reader().unwrap();
        let blob: Blob<UnknownBlob> = reader.get(handle).unwrap();
        assert_eq!(blob.bytes.as_ref(), payload);
        drop(reader);
        pile.close().unwrap();
    }

    #[test]
    fn all_invalid_duplicates_leave_the_last_candidate_lazy() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "all-invalid.pile");
        let expected = b"target";
        let handle = Blob::<UnknownBlob>::new(Bytes::from_source(expected.to_vec())).get_handle();
        let hash: Inline<Hash<Blake3>> = handle.into();

        let first = append_v3_blob_candidate(&path, hash, b"bad-01", 1);
        let second = append_v3_blob_candidate(&path, hash, b"bad-02", 2);
        let third = append_v3_blob_candidate(&path, hash, b"bad-03", 3);

        let mut pile = Pile::open(&path).unwrap();
        pile.refresh().unwrap();
        assert_eq!(pile.blobs.get(&hash.raw).unwrap().record_offset, third);
        let states = pile.validations.states.lock().unwrap();
        assert!(matches!(states.get(&first), Some(ValidationState::Invalid)));
        assert!(matches!(
            states.get(&second),
            Some(ValidationState::Invalid)
        ));
        assert!(!states.contains_key(&third));
        drop(states);

        let reader = pile.reader().unwrap();
        assert!(matches!(
            reader.get::<Blob<UnknownBlob>, UnknownBlob>(handle),
            Err(GetBlobError::ValidationError(_))
        ));
        assert!(matches!(
            pile.validations.states.lock().unwrap().get(&third),
            Some(ValidationState::Invalid)
        ));
        drop(reader);
        pile.close().unwrap();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_validation_dispatch_respects_threshold_and_current_pool() {
        let bytes = Bytes::from_source(vec![0xA5; PARALLEL_BLAKE3_THRESHOLD]);
        let expected = Hash::<Blake3>::digest(&bytes);

        let one_worker = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        one_worker.install(|| {
            assert!(!should_parallelize_validation(PARALLEL_BLAKE3_THRESHOLD));
            assert!(matches!(
                compute_validation_state(&bytes, &expected, ValidationStrategy::ParallelIfLarge),
                ValidationState::Validated
            ));
        });

        let two_workers = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        two_workers.install(|| {
            assert!(!should_parallelize_validation(
                PARALLEL_BLAKE3_THRESHOLD - 1
            ));
            assert!(should_parallelize_validation(PARALLEL_BLAKE3_THRESHOLD));
            assert!(matches!(
                compute_validation_state(&bytes, &expected, ValidationStrategy::ParallelIfLarge),
                ValidationState::Validated
            ));
            assert!(matches!(
                compute_validation_state(&bytes, &expected, ValidationStrategy::Serial),
                ValidationState::Validated
            ));
        });
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn large_reader_get_uses_parallel_validation() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "large-reader.pile");
        let payload = vec![0xC3; PARALLEL_BLAKE3_THRESHOLD + 17];

        let mut writer = Pile::open(&path).unwrap();
        let handle: Inline<Handle<UnknownBlob>> =
            writer.put(Bytes::from_source(payload.clone())).unwrap();
        writer.close().unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let mut replayed = Pile::open(&path).unwrap();
        let reader = replayed.reader().unwrap();
        pool.install(|| {
            let blob = reader
                .get::<Blob<UnknownBlob>, UnknownBlob>(handle)
                .unwrap();
            assert_eq!(blob.bytes.as_ref(), payload.as_slice());
        });
        drop(reader);
        replayed.close().unwrap();
    }

    #[test]
    fn open() {
        const RECORD_LEN: usize = 1 << 10; // 1k
        const RECORD_COUNT: usize = 1 << 12; // 4k

        let mut rng = rand::thread_rng();
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_pile = fresh_empty_pile_path(&tmp_dir, "test.pile");
        let mut pile: Pile = Pile::open(&tmp_pile).unwrap();

        (0..RECORD_COUNT).for_each(|_| {
            let mut record = Vec::with_capacity(RECORD_LEN);
            rng.fill_bytes(&mut record);

            let data: Blob<UnknownBlob> = Blob::new(Bytes::from_source(record));
            pile.put::<UnknownBlob, _>(data).unwrap();
        });

        pile.close().unwrap();

        let mut reopened: Pile = Pile::open(&tmp_pile).unwrap();
        reopened.amputate().unwrap();
        reopened.close().unwrap();
    }

    #[test]
    fn put_enveloped_256_aligned_roundtrip() {
        // Every current record is a 256-byte multiple with the data at a fixed
        // header offset, so plain `put` yields absolutely 256-aligned
        // (GPU-aliasable) data in a current pile.
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "v3.pile");
        // Sizes around the 64/256 boundaries to exercise the post-pad.
        let sizes = [1usize, 7, 33, 64, 100, 192, 255, 256, 257, 1000, 4096];
        let mut hashes = Vec::new();
        let mut datas: Vec<Vec<u8>> = Vec::new();
        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            for &sz in &sizes {
                let data: Vec<u8> = (0..sz).map(|i| (i % 251) as u8).collect();
                let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
                let h = pile.put::<UnknownBlob, _>(blob).unwrap();
                let hash: Inline<Hash<Blake3>> = h.into();
                hashes.push(hash);
                datas.push(data);
            }
            pile.close().unwrap();
        }
        // Reopen fresh — the scan rebuilds the index from enveloped records.
        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        for (hash, expected) in hashes.iter().zip(&datas) {
            let entry = *pile
                .blobs
                .get(&hash.raw)
                .expect("enveloped blob missing after reopen");
            let record = indexed_blob_record(&pile.mmap, pile.applied_length, entry, hash);
            assert_eq!(
                record.payload_offset % GPU_DATA_ALIGNMENT,
                0,
                "enveloped data offset {} not {GPU_DATA_ALIGNMENT}-aligned (size {})",
                record.payload_offset,
                expected.len()
            );
            assert_eq!(
                record.bytes.as_ref(),
                &expected[..],
                "enveloped roundtrip mismatch (size {})",
                expected.len()
            );
        }
        pile.close().unwrap();
    }

    /// The whole point of uniform current framing: `cat a.pile >> b.pile` is a valid merge —
    /// every record from both piles is found and byte-correct, the data stays
    /// 256-aligned, and `amputate()` does not truncate the concatenation as
    /// corrupt. This is what an offset-derived pad could never survive.
    #[test]
    fn enveloped_cat_merge_preserves_all_blobs_and_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path_a = fresh_empty_pile_path(&dir, "a.pile");
        let path_b = fresh_empty_pile_path(&dir, "b.pile");
        let sizes = [1usize, 33, 100, 256, 257, 1000, 4096];
        let mut handles: Vec<(Inline<Hash<Blake3>>, Vec<u8>)> = Vec::new();

        {
            let mut a: Pile = Pile::open(&path_a).unwrap();
            for (k, &sz) in sizes.iter().enumerate() {
                let data: Vec<u8> = (0..sz).map(|i| ((i + k) % 251) as u8).collect();
                let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
                let h: Inline<Hash<Blake3>> = a.put::<UnknownBlob, _>(blob).unwrap().into();
                handles.push((h, data));
            }
            a.close().unwrap();
        }
        {
            let mut b: Pile = Pile::open(&path_b).unwrap();
            for (k, &sz) in sizes.iter().enumerate() {
                // Distinct content so no hash collisions with pile A.
                let data: Vec<u8> = (0..sz).map(|i| ((i + k + 128) % 251) as u8).collect();
                let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
                let h: Inline<Hash<Blake3>> = b.put::<UnknownBlob, _>(blob).unwrap().into();
                handles.push((h, data));
            }
            b.close().unwrap();
        }

        // Each current pile is a whole number of 256-byte units — the precondition
        // that makes the appended pile land on a 256-aligned offset.
        assert_eq!(
            std::fs::metadata(&path_a).unwrap().len() % ENVELOPE_BLOCK_LEN as u64,
            0
        );
        assert_eq!(
            std::fs::metadata(&path_b).unwrap().len() % ENVELOPE_BLOCK_LEN as u64,
            0
        );

        // cat a.pile >> b.pile
        {
            let a_bytes = std::fs::read(&path_a).unwrap();
            let mut bf = std::fs::OpenOptions::new()
                .append(true)
                .open(&path_b)
                .unwrap();
            bf.write_all(&a_bytes).unwrap();
            bf.sync_all().unwrap();
        }
        let merged_len = std::fs::metadata(&path_b).unwrap().len();

        let mut merged: Pile = Pile::open(&path_b).unwrap();
        merged.amputate().unwrap();
        assert_eq!(
            std::fs::metadata(&path_b).unwrap().len(),
            merged_len,
            "cat-merged pile was truncated — cat is not a valid framed merge"
        );
        for (hash, expected) in &handles {
            let entry = *merged
                .blobs
                .get(&hash.raw)
                .expect("blob lost after cat-merge");
            let record = indexed_blob_record(&merged.mmap, merged.applied_length, entry, hash);
            assert_eq!(
                record.payload_offset % ENVELOPE_BLOCK_LEN,
                0,
                "post-cat data offset not 256-aligned"
            );
            assert_eq!(
                record.bytes.as_ref(),
                &expected[..],
                "blob bytes wrong after cat-merge"
            );
        }
        // Still 256-aligned, so it can be cat'd again.
        assert_eq!(
            std::fs::metadata(&path_b).unwrap().len() % ENVELOPE_BLOCK_LEN as u64,
            0
        );
        merged.close().unwrap();
    }

    /// Existing V1 piles remain readable unchanged by the current reader.
    #[test]
    fn v3_reader_still_reads_legacy_v1_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "legacy_v1.pile");
        let data = vec![9u8; 40];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle: Inline<Handle<UnknownBlob>> = blob.get_handle();
        let hash: Inline<Hash<Blake3>> = handle.into();
        // Hand-write a legacy V1 blob record: 64-byte header + data + 64-pad.
        {
            let header = BlobHeader::new(42, data.len() as u64, hash);
            let pad = padding_for_blob(data.len());
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(header.as_bytes()).unwrap();
            f.write_all(&data).unwrap();
            f.write_all(&vec![0u8; pad]).unwrap();
            f.sync_all().unwrap();
        }
        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        let reader = pile.reader().unwrap();
        let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
        assert_eq!(
            fetched.bytes.as_ref(),
            data.as_slice(),
            "legacy V1 blob not read by the current reader"
        );
        pile.close().unwrap();
    }

    #[test]
    fn recover_shrink() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 20]));
            pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.close().unwrap();
        }

        // Corrupt by removing some bytes from the end
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        let len = file.metadata().unwrap().len();
        file.set_len(len - 10).unwrap();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        pile.close().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn refresh_corrupt_reports_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 20]));
            pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.close().unwrap();
        }

        let file_len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(file_len - 10)
            .unwrap();

        let mut pile: Pile = Pile::open(&path).unwrap();
        match pile.refresh() {
            Err(ReadError::CorruptPile { valid_length }) => assert_eq!(valid_length, 0),
            other => panic!("unexpected result: {other:?}"),
        }
        pile.close().unwrap();
    }

    #[test]
    fn bounded_replay_stops_at_its_observed_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "bounded-refresh.pile");

        let (first, second) = {
            let mut writer = Pile::open(&path).unwrap();
            let first = writer
                .put::<UnknownBlob, _>(Bytes::from_source(b"first".to_vec()))
                .unwrap();
            let second = writer
                .put::<UnknownBlob, _>(Bytes::from_source(b"second".to_vec()))
                .unwrap();
            writer.close().unwrap();
            (first, second)
        };

        let first_end = {
            let mut records = PileRecords::open(&path).unwrap();
            let record = records.next().unwrap().unwrap();
            record.offset + record.len
        };

        let mut replay = Pile::open(&path).unwrap();
        assert!(matches!(
            replay.apply_next_bounded(first_end).unwrap(),
            Some(Applied::Blob { hash }) if hash.raw == first.raw
        ));
        assert!(replay.apply_next_bounded(first_end).unwrap().is_none());
        assert_eq!(replay.applied_length, first_end);
        assert!(replay.blobs.get(&first.raw).is_some());
        assert!(replay.blobs.get(&second.raw).is_none());

        replay.refresh().unwrap();
        assert!(replay.blobs.get(&second.raw).is_some());
        replay.close().unwrap();
    }

    #[test]
    fn unknown_magic_reports_unsupported_without_amputation() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 20]));
            pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.close().unwrap();
        }

        let valid_len = std::fs::metadata(&path).unwrap().len() as usize;
        let unknown_marker = [0xA5u8; 16];
        let mut unknown_record = [0u8; V3_HEADER_LEN];
        unknown_record[..16].copy_from_slice(&unknown_marker);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&unknown_record)
            .unwrap();
        let length_with_unknown_record = std::fs::metadata(&path).unwrap().len();

        let mut pile: Pile = Pile::open(&path).unwrap();
        assert!(matches!(
            pile.refresh(),
            Err(ReadError::UnsupportedRecord { offset, marker })
                if offset == valid_len && marker == unknown_marker
        ));
        assert!(matches!(
            pile.amputate(),
            Err(ReadError::UnsupportedRecord { offset, marker })
                if offset == valid_len && marker == unknown_marker
        ));
        pile.close().unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            length_with_unknown_record,
            "amputation must preserve a record whose marker this reader does not know"
        );
    }

    #[test]
    fn decoder_distinguishes_unknown_magic_from_truncated_known_record() {
        let unknown_marker = [0xA5u8; 16];
        assert!(matches!(
            decode_record(&unknown_marker, ENVELOPE_BLOCK_LEN),
            Err(ReadError::UnsupportedRecord { offset, marker })
                if offset == ENVELOPE_BLOCK_LEN && marker == unknown_marker
        ));

        assert!(matches!(
            decode_record(&MAGIC_MARKER_BLOB_V3, ENVELOPE_BLOCK_LEN),
            Err(ReadError::CorruptPile { valid_length })
                if valid_length == ENVELOPE_BLOCK_LEN
        ));
    }

    #[test]
    fn refresh_partial_header_reports_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 20]));
            pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.close().unwrap();
        }

        let file_len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(file_len + 8)
            .unwrap();

        let mut pile: Pile = Pile::open(&path).unwrap();
        match pile.refresh() {
            Err(ReadError::CorruptPile { valid_length }) => {
                assert_eq!(valid_length as u64, file_len)
            }
            other => panic!("unexpected result: {other:?}"),
        }
        pile.close().unwrap();
    }

    #[test]
    fn refresh_length_beyond_file_reports_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 20]));
            pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.close().unwrap();
        }

        use std::io::Seek;
        use std::io::SeekFrom;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(48)).unwrap();
        file.write_all(&(1_000_000u64).to_le_bytes()).unwrap();
        file.flush().unwrap();
        drop(file);

        let mut pile: Pile = Pile::open(&path).unwrap();
        match pile.refresh() {
            Err(ReadError::CorruptPile { valid_length }) => assert_eq!(valid_length, 0),
            other => panic!("unexpected result: {other:?}"),
        }
        pile.close().unwrap();
    }

    #[test]
    fn amputate_truncates_length_beyond_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 20]));
            pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.close().unwrap();
        }

        use std::io::Seek;
        use std::io::SeekFrom;
        use std::io::Write;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(48)).unwrap();
        file.write_all(&(1_000_000u64).to_le_bytes()).unwrap();
        file.flush().unwrap();
        drop(file);

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        pile.close().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn put_and_get_preserves_blob_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let data = vec![42u8; 100];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();

        {
            let reader = pile.reader().unwrap();
            let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
            assert_eq!(fetched.bytes.as_ref(), data.as_slice());
        }

        pile.close().unwrap();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        let reader = pile.reader().unwrap();
        let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
        assert_eq!(fetched.bytes.as_ref(), data.as_slice());
        pile.close().unwrap();
    }

    #[test]
    fn close_flushes_only_mutations_by_this_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");
        let mut observer = Pile::open(&path).unwrap();
        let mut writer = Pile::open(&path).unwrap();

        assert!(!observer.dirty);
        assert!(!writer.dirty);

        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![4u8; 32]));
        let handle = writer.put::<UnknownBlob, _>(blob).unwrap();
        assert!(writer.dirty);

        // Replaying an append made through another descriptor must not make a
        // read-only observer responsible for a whole-file sync.
        observer.refresh().unwrap();
        assert!(!observer.dirty);
        observer.flush_if_dirty().unwrap();
        assert!(!observer.dirty);

        writer.flush().unwrap();
        assert!(!writer.dirty);
        writer.flush().unwrap();
        assert!(!writer.dirty);

        let branch = Id::new([9u8; 16]).unwrap();
        writer
            .update(branch, None, Some(handle.transmute()))
            .unwrap();
        assert!(writer.dirty);
        writer.flush().unwrap();
        assert!(!writer.dirty);

        // Conflicts and logical no-ops append nothing.
        assert!(matches!(
            writer
                .update(branch, None, Some(handle.transmute()))
                .unwrap(),
            PushResult::Conflict(_)
        ));
        assert!(!writer.dirty);
        writer
            .update(branch, Some(handle.transmute()), Some(handle.transmute()))
            .unwrap();
        assert!(!writer.dirty);

        writer.want(WantRequest::blob(handle)).unwrap();
        assert!(writer.dirty);
        writer.flush().unwrap();
        assert!(!writer.dirty);
        writer.want(WantRequest::blob(handle)).unwrap();
        assert!(!writer.dirty);
        writer.unwant(WantRequest::blob(handle)).unwrap();
        assert!(writer.dirty);
        writer.flush().unwrap();
        assert!(!writer.dirty);

        observer.close().unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn iter_lists_all_blobs_handles() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blobs = vec![vec![1u8; 3], vec![2u8; 4], vec![3u8; 5]];
        let mut expected = HashMap::new();
        for data in blobs {
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
            let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
            expected.insert(handle, data);
        }
        pile.flush().unwrap();

        let reader = pile.reader().unwrap();
        for item in reader.iter() {
            let (handle, blob) = item.expect("infallible iteration");
            let data = expected.remove(&handle).unwrap();
            assert_eq!(blob.bytes.as_ref(), data.as_slice());
        }
        assert!(expected.is_empty());

        pile.close().unwrap();
    }

    #[test]
    fn blobs_diff_returns_only_new_handles() {
        use crate::repo::BlobStoreList;
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();

        // Stage three baseline blobs and snapshot the reader.
        let mut baseline_handles: HashSet<Inline<Handle<UnknownBlob>>> = HashSet::new();
        for data in [vec![1u8; 3], vec![2u8; 4], vec![3u8; 5]] {
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data));
            let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
            baseline_handles.insert(handle);
        }
        let baseline = pile.reader().unwrap();

        // Stage two more blobs after taking the baseline snapshot.
        let mut new_handles: HashSet<Inline<Handle<UnknownBlob>>> = HashSet::new();
        for data in [vec![4u8; 6], vec![5u8; 7]] {
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data));
            let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
            new_handles.insert(handle);
        }

        // Diff the current reader against the baseline.
        let current = pile.reader().unwrap();
        let diffed: HashSet<Inline<Handle<UnknownBlob>>> = current
            .blobs_diff(&baseline)
            .map(|r| r.expect("infallible diff iter").handle)
            .collect();

        // Diff should equal exactly the new blobs — none of the baseline ones.
        assert_eq!(diffed, new_handles);
        for h in &baseline_handles {
            assert!(!diffed.contains(h), "baseline blob leaked into diff");
        }

        // Round-trip sanity: diffing a reader against itself yields nothing.
        let empty: HashSet<_> = current
            .blobs_diff(&current)
            .map(|r| r.expect("infallible").handle)
            .collect();
        assert!(empty.is_empty());

        pile.close().unwrap();
    }

    #[test]
    fn metadata_reflects_length_and_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let data = vec![9u8; 10];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();

        let reader = pile.reader().unwrap();
        let metadata = reader.metadata(handle).unwrap().expect("metadata");
        assert_eq!(metadata.length, data.len() as u64);
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        assert!(metadata.timestamp >= before && metadata.timestamp <= after);
        pile.close().unwrap();
    }

    #[test]
    fn listing_reports_header_lengths() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let first = pile
            .put::<UnknownBlob, _>(Blob::new(Bytes::from_source(vec![1u8; 3])))
            .unwrap();
        let second = pile
            .put::<UnknownBlob, _>(Blob::new(Bytes::from_source(vec![2u8; 17])))
            .unwrap();
        let reader = pile.reader().unwrap();
        let listed: HashMap<_, _> = reader
            .blobs()
            .map(|result| {
                let info = result.expect("infallible listing");
                (info.handle, info.length)
            })
            .collect();

        assert_eq!(listed.get(&first), Some(&3));
        assert_eq!(listed.get(&second), Some(&17));
        pile.close().unwrap();
    }

    #[test]
    fn metadata_returns_none_for_unflushed_blob() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let reader = pile.reader().unwrap();

        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 4]));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();

        assert!(reader.metadata(handle).unwrap().is_none());

        pile.flush().unwrap();
        let reader = pile.reader().unwrap();
        assert!(reader.metadata(handle).unwrap().is_some());
        pile.close().unwrap();
    }

    #[test]
    fn blob_after_branch_is_clean() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();

        let branch_id = Id::new([1; 16]).unwrap();
        let head = Inline::<Handle<SimpleArchive>>::new([2; 32]);
        pile.update(branch_id, None, Some(head)).unwrap();

        let data = vec![3u8; 8];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();

        let stored: Blob<UnknownBlob> = pile.reader().unwrap().get(handle).unwrap();
        assert_eq!(stored.bytes.as_ref(), &data[..]);
        pile.close().unwrap();
    }

    #[test]
    fn insert_after_branch_preserves_head() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 5]));
        let handle1 = pile.put::<UnknownBlob, _>(blob1).unwrap();

        let branch_id = Id::new([1u8; 16]).unwrap();
        pile.update(branch_id, None, Some(handle1.transmute()))
            .unwrap();

        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![2u8; 5]));
        pile.put::<UnknownBlob, _>(blob2).unwrap();
        pile.close().unwrap();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        let head = pile.head(branch_id).unwrap();
        assert_eq!(head, Some(handle1.transmute()));
        pile.close().unwrap();
    }

    #[test]
    fn branch_update_survives_manual_flush() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let branch_id = Id::new([1u8; 16]).unwrap();

        let handle = {
            let mut pile: Pile = Pile::open(&path).unwrap();
            let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![3u8; 5]));
            let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
            pile.update(branch_id, None, Some(handle.transmute()))
                .unwrap();
            pile.flush().unwrap();
            std::mem::forget(pile);
            handle
        };

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        assert_eq!(pile.head(branch_id).unwrap(), Some(handle.transmute()));
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
        pile.close().unwrap();
    }

    #[test]
    fn branch_tombstone_removes_head_and_listing() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 5]));
        let h = pile.put::<UnknownBlob, _>(blob).unwrap();
        let branch_id = Id::new([7u8; 16]).unwrap();
        pile.update(branch_id, None, Some(h.transmute())).unwrap();
        pile.flush().unwrap();

        assert_eq!(pile.head(branch_id).unwrap(), Some(h.transmute()));

        pile.update(branch_id, Some(h.transmute()), None).unwrap();
        pile.flush().unwrap();

        assert_eq!(pile.head(branch_id).unwrap(), None);
        let branches: HashSet<Id> = pile.pins().unwrap().map(|r| r.unwrap()).collect();
        assert!(!branches.contains(&branch_id));
        pile.close().unwrap();
    }

    #[test]
    fn branch_update_detects_conflict() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 5]));
        let handle1 = pile.put::<UnknownBlob, _>(blob1).unwrap();

        let branch_id = Id::new([2u8; 16]).unwrap();
        pile.update(branch_id, None, Some(handle1.transmute()))
            .unwrap();

        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![2u8; 5]));
        let handle2 = pile.put::<UnknownBlob, _>(blob2).unwrap();
        pile.flush().unwrap();

        match pile
            .update(
                branch_id,
                Some(handle2.transmute()),
                Some(handle2.transmute()),
            )
            .unwrap()
        {
            PushResult::Conflict(current) => {
                assert_eq!(current, Some(handle1.transmute()));
            }
            other => panic!("unexpected result: {other:?}"),
        }
        assert_eq!(pile.head(branch_id).unwrap(), Some(handle1.transmute()));
        pile.close().unwrap();
    }

    #[test]
    fn branch_update_conflict_returns_current_head() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 5]));
        let handle1 = pile.put::<UnknownBlob, _>(blob1).unwrap();

        let branch_id = Id::new([1u8; 16]).unwrap();
        pile.update(branch_id, None, Some(handle1.transmute()))
            .unwrap();
        pile.flush().unwrap();

        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![2u8; 5]));
        let handle2 = pile.put::<UnknownBlob, _>(blob2).unwrap();

        let result = pile
            .update(
                branch_id,
                Some(handle2.transmute()),
                Some(handle2.transmute()),
            )
            .unwrap();
        match result {
            PushResult::Conflict(current) => assert_eq!(current, Some(handle1.transmute())),
            other => panic!("unexpected result: {other:?}"),
        }
        assert_eq!(pile.head(branch_id).unwrap(), Some(handle1.transmute()));
        pile.close().unwrap();
    }

    #[test]
    fn metadata_returns_length_and_timestamp() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![7u8; 32]));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.close().unwrap();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        let reader = pile.reader().unwrap();
        let meta = reader.metadata(handle).unwrap().expect("metadata");
        assert_eq!(meta.length, 32);
        assert!(meta.timestamp > 0);
        pile.close().unwrap();
    }

    #[test]
    fn iter_lists_all_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 4]));
        let h1 = pile.put::<UnknownBlob, _>(blob1).unwrap();
        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![2u8; 4]));
        let h2 = pile.put::<UnknownBlob, _>(blob2).unwrap();
        pile.flush().unwrap();

        let reader = pile.reader().unwrap();
        let handles: Vec<_> = reader
            .iter()
            .map(|res| res.expect("infallible iteration").0)
            .collect();
        assert!(handles.contains(&h1));
        assert!(handles.contains(&h2));
        assert_eq!(handles.len(), 2);
        pile.close().unwrap();
    }

    #[test]
    fn update_conflict_returns_current_head() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 5]));
        let h1 = pile.put::<UnknownBlob, _>(blob1).unwrap();
        let branch_id = Id::new([1u8; 16]).unwrap();
        pile.update(branch_id, None, Some(h1.transmute())).unwrap();
        pile.flush().unwrap();

        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![2u8; 5]));
        let h2 = pile.put::<UnknownBlob, _>(blob2).unwrap();
        pile.flush().unwrap();

        match pile.update(branch_id, Some(h2.transmute()), Some(h1.transmute())) {
            Ok(PushResult::Conflict(existing)) => {
                assert_eq!(existing, Some(h1.transmute()))
            }
            other => panic!("unexpected result: {other:?}"),
        }
        assert_eq!(pile.head(branch_id).unwrap(), Some(h1.transmute()));
        pile.close().unwrap();
    }

    #[test]
    fn refresh_errors_on_malformed_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 4]));
        pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();

        use std::io::Write;
        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            file.write_all(b"garbage").unwrap();
            file.sync_all().unwrap();
        }

        assert!(pile.refresh().is_err());
        pile.close().unwrap();
    }

    #[test]
    fn amputate_truncates_corrupt_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let data = vec![1u8; 4];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();

        use std::io::Write;
        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            file.write_all(b"garbage").unwrap();
            file.sync_all().unwrap();
        }

        pile.amputate().unwrap();

        // Blobs are written as enveloped records (fixed 256-byte header, padded to a
        // 256-byte multiple).
        let expected_len =
            (super::ENVELOPE_HEADER_LEN + data.len() + super::block_post_pad(data.len())) as u64;
        assert_eq!(std::fs::metadata(&path).unwrap().len(), expected_len);

        let reader = pile.reader().unwrap();
        let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
        assert_eq!(fetched.bytes.as_ref(), data.as_slice());
        pile.close().unwrap();
    }

    #[test]
    fn refresh_replaces_corrupt_blob_with_new_candidate() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile1: Pile = Pile::open(&path).unwrap();
        let mut pile2: Pile = Pile::open(&path).unwrap();

        let data = vec![1u8; 4];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile1.put(blob).unwrap();
        pile1.flush().unwrap();
        pile1.refresh().unwrap();

        // Corrupt the first enveloped blob's payload (the fixed header is 256 bytes).
        use std::io::Seek;
        use std::io::SeekFrom;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(ENVELOPE_HEADER_LEN as u64))
            .unwrap();
        file.write_all(&[9u8; 4]).unwrap();
        file.sync_all().unwrap();

        // Append a valid copy using the second pile which hasn't seen the first one.
        let blob_dup: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        pile2.put::<UnknownBlob, _>(blob_dup).unwrap();
        pile2.flush().unwrap();

        // Refresh the first pile; it should replace the corrupted blob with the new one.
        pile1.refresh().unwrap();
        let reader = pile1.reader().unwrap();
        let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
        assert_eq!(fetched.bytes.as_ref(), data.as_slice());
        pile1.close().unwrap();
        pile2.close().unwrap();
    }

    #[test]
    fn put_duplicate_blob_does_not_grow_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let data = vec![9u8; 32];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle1 = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();
        let len_after_first = std::fs::metadata(&path).unwrap().len();

        let blob_dup: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data));
        let handle2 = pile.put(blob_dup).unwrap();
        pile.flush().unwrap();
        let len_after_second = std::fs::metadata(&path).unwrap().len();

        assert_eq!(handle1, handle2);
        assert_eq!(len_after_first, len_after_second);
        pile.close().unwrap();
    }

    #[test]
    fn branch_update_conflict_returns_existing_head() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![1u8; 8]));
        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![2u8; 8]));
        let h1 = pile.put::<UnknownBlob, _>(blob1).unwrap();
        let h2 = pile.put::<UnknownBlob, _>(blob2).unwrap();
        pile.flush().unwrap();

        let branch_id = Id::new([3u8; 16]).unwrap();
        pile.update(branch_id, None, Some(h1.transmute())).unwrap();

        match pile.update(branch_id, Some(h2.transmute()), Some(h2.transmute())) {
            Ok(PushResult::Conflict(existing)) => {
                assert_eq!(existing, Some(h1.transmute()))
            }
            other => panic!("expected conflict, got {other:?}"),
        }
        assert_eq!(pile.head(branch_id).unwrap(), Some(h1.transmute()));
        pile.close().unwrap();
    }

    #[test]
    fn branch_update_noop_does_not_grow_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(vec![7u8; 8]));
        let h = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();

        let branch_id = Id::new([4u8; 16]).unwrap();
        pile.update(branch_id, None, Some(h.transmute())).unwrap();
        pile.flush().unwrap();
        let len_after_first = std::fs::metadata(&path).unwrap().len();

        match pile.update(branch_id, Some(h.transmute()), Some(h.transmute())) {
            Ok(PushResult::Success()) => {}
            other => panic!("expected no-op success, got {other:?}"),
        }
        pile.flush().unwrap();
        let len_after_noop = std::fs::metadata(&path).unwrap().len();

        assert_eq!(
            len_after_first, len_after_noop,
            "no-op branch update must not append a new record"
        );
        assert_eq!(pile.head(branch_id).unwrap(), Some(h.transmute()));
        pile.close().unwrap();
    }

    #[test]
    fn iterator_skips_missing_index_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(b"hello".as_slice()));
        let blob2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(b"world".as_slice()));
        let handle1 = pile.put::<UnknownBlob, _>(blob1).unwrap();
        let handle2 = pile.put::<UnknownBlob, _>(blob2).unwrap();
        pile.flush().unwrap();

        let mut reader = pile.reader().unwrap();
        let _full_patch = reader.blobs.clone();
        let hash1: Inline<Hash<Blake3>> = handle1.into();
        reader.blobs.remove(&hash1.raw);

        let mut iter = reader.iter();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| iter.next()));
        if let Ok(Some(Ok((h, _)))) = result {
            assert_eq!(h, handle2);
            assert!(iter.next().is_none());
        } else {
            assert!(cfg!(debug_assertions));
        }
        pile.close().unwrap();
    }

    #[test]
    fn metadata_reports_blob_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let mut pile: Pile = Pile::open(&path).unwrap();
        let data = vec![7u8; 16];
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();
        pile.flush().unwrap();

        let reader = pile.reader().unwrap();
        let meta = reader.metadata(handle).unwrap().expect("metadata");
        assert_eq!(meta.length, data.len() as u64);
        pile.close().unwrap();
    }

    /// Durable wants survive close + reopen; the scan rebuilds the
    /// LWW-resolved set from the on-disk markers.
    #[test]
    fn want_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        // A want may name a blob that is not present in the pile.
        let wanted: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![7u8; 21])).get_handle();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.want(WantRequest::blob(wanted)).unwrap();
        let pinned: HashSet<_> = pile.wants().unwrap().map(|r| r.unwrap()).collect();
        assert!(pinned.contains(&WantRequest::blob(wanted)));
        pile.close().unwrap();

        let mut reopened: Pile = Pile::open(&path).unwrap();
        reopened.amputate().unwrap();
        let pinned: HashSet<_> = reopened.wants().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(pinned.len(), 1);
        assert!(
            pinned.contains(&WantRequest::blob(wanted)),
            "want lost across reopen — restart amnesia"
        );
        reopened.close().unwrap();
    }

    #[test]
    fn typed_operation_wants_roundtrip_as_exact_enveloped_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "typed-wants.pile");
        let source = collection_test_collection(31);
        let target = collection_test_collection(32);
        let merge = WantRequest::merge(source, collection_test_hash(34), collection_test_hash(33));
        let derive = WantRequest::derive(source, target, collection_test_hash(35));

        let mut pile = Pile::open(&path).unwrap();
        pile.want(merge).unwrap();
        pile.want(derive).unwrap();
        pile.flush().unwrap();
        assert_eq!(
            pile.wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![merge, derive]
        );
        pile.close().unwrap();

        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            (2 * ENVELOPE_HEADER_LEN) as u64
        );
        let records = PileRecords::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(matches!(
            records[0].content,
            PileRecordContent::WantAssert { request } if request == merge
        ));
        assert!(matches!(
            records[1].content,
            PileRecordContent::WantAssert { request } if request == derive
        ));

        let mut reopened = Pile::open(&path).unwrap();
        reopened.refresh().unwrap();
        assert_eq!(
            reopened
                .wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![merge, derive]
        );
        reopened.close().unwrap();
    }

    #[test]
    fn typed_want_retraction_is_scoped_to_the_exact_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "typed-want-lww.pile");
        let source = collection_test_collection(41);
        let target = collection_test_collection(42);
        let input = collection_test_hash(43);
        let merge = WantRequest::merge(source, input, collection_test_hash(44));
        let derive = WantRequest::derive(source, target, input);

        let mut pile = Pile::open(&path).unwrap();
        pile.want(merge).unwrap();
        pile.want(derive).unwrap();
        pile.unwant(merge).unwrap();
        assert_eq!(
            pile.wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![derive]
        );
        pile.close().unwrap();

        let mut reopened = Pile::open(&path).unwrap();
        reopened.refresh().unwrap();
        assert_eq!(
            reopened
                .wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![derive]
        );
        reopened.close().unwrap();
    }

    #[test]
    fn blob_wants_keep_the_legacy_physical_kinds_for_old_reader_projection() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "blob-want-projection.pile");
        let handle = Inline::<Handle<UnknownBlob>>::new([47; 32]);

        let mut pile = Pile::open(&path).unwrap();
        pile.want(WantRequest::blob(handle)).unwrap();
        pile.unwant(WantRequest::blob(handle)).unwrap();
        pile.close().unwrap();

        let records = PileRecords::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(matches!(
            records[0].content,
            PileRecordContent::WeakPin { handle: actual } if actual == handle
        ));
        assert!(matches!(
            records[1].content,
            PileRecordContent::WeakUnpin { handle: actual } if actual == handle
        ));
    }

    #[test]
    fn typed_physical_want_markers_reject_blob_requests() {
        let request = WantRequest::blob(Inline::<Handle<UnknownBlob>>::new([48; 32]));
        assert!(TypedWantHeaderEnvelope::new_operation(request, true).is_none());

        // Also reject the representation at the trust boundary rather than
        // merely relying on our writer not to produce it.
        let encoded = request.to_bytes();
        let header = TypedWantHeaderEnvelope {
            envelope_marker: MAGIC_MARKER_ENVELOPE,
            record_kind: MAGIC_MARKER_WANT_ASSERT_V2,
            span_blocks: ENVELOPE_HEADER_BLOCKS.to_le_bytes(),
            request_kind: encoded[0],
            field_a: encoded[1..33].try_into().unwrap(),
            field_b: encoded[33..65].try_into().unwrap(),
            field_c: encoded[65..97].try_into().unwrap(),
            reserved: [0; 123],
        };
        assert!(matches!(
            decode_record(header.as_bytes(), 0),
            Err(ReadError::CorruptPile { valid_length: 0 })
        ));
    }

    /// LWW by log position: the last marker for a handle wins, both live and
    /// across a fresh scan of the on-disk record sequence.
    #[test]
    fn want_lww_last_writer_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let a: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![1u8; 9])).get_handle();
        let b: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![2u8; 9])).get_handle();

        let mut pile: Pile = Pile::open(&path).unwrap();
        // a: pin, unpin, pin — three real records; last writer says pinned.
        pile.want(WantRequest::blob(a)).unwrap();
        pile.unwant(WantRequest::blob(a)).unwrap();
        pile.want(WantRequest::blob(a)).unwrap();
        // b: pin then unpin — last writer says unpinned.
        pile.want(WantRequest::blob(b)).unwrap();
        pile.unwant(WantRequest::blob(b)).unwrap();

        let pinned: HashSet<_> = pile.wants().unwrap().map(|r| r.unwrap()).collect();
        assert!(pinned.contains(&WantRequest::blob(a)));
        assert!(!pinned.contains(&WantRequest::blob(b)));
        pile.close().unwrap();

        // The same resolution must fall out of a fresh log replay.
        let mut reopened: Pile = Pile::open(&path).unwrap();
        reopened.amputate().unwrap();
        let pinned: HashSet<_> = reopened.wants().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(pinned.len(), 1);
        assert!(pinned.contains(&WantRequest::blob(a)));
        assert!(!pinned.contains(&WantRequest::blob(b)));
        reopened.close().unwrap();
    }

    /// Re-asserting the current want state is a no-op append (mirrors the
    /// branch-update no-op rule): the LWW state carries no new information.
    #[test]
    fn want_noop_does_not_grow_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        let h: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![3u8; 5])).get_handle();

        let mut pile: Pile = Pile::open(&path).unwrap();
        // Unpinning a never-pinned handle records nothing.
        pile.unwant(WantRequest::blob(h)).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);

        pile.want(WantRequest::blob(h)).unwrap();
        let len_after_pin = std::fs::metadata(&path).unwrap().len();
        assert_eq!(len_after_pin, ENVELOPE_HEADER_LEN as u64);

        pile.want(WantRequest::blob(h)).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len_after_pin);
        pile.close().unwrap();
    }

    /// Mixed pile: a legacy V1 blob, enveloped blobs, branch records, and want
    /// markers interleaved — the scan walks every record kind cleanly and
    /// each index (blobs, branches, wants) resolves correctly.
    #[test]
    fn mixed_v1_enveloped_branch_and_weak_markers_interleave() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "mixed.pile");

        // Hand-write a legacy V1 blob record first (64-byte header + pad).
        let v1_data = vec![9u8; 40];
        let v1_blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(v1_data.clone()));
        let v1_handle: Inline<Handle<UnknownBlob>> = v1_blob.get_handle();
        {
            let v1_hash: Inline<Hash<Blake3>> = v1_handle.into();
            let header = BlobHeader::new(42, v1_data.len() as u64, v1_hash);
            let pad = padding_for_blob(v1_data.len());
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(header.as_bytes()).unwrap();
            f.write_all(&v1_data).unwrap();
            f.write_all(&vec![0u8; pad]).unwrap();
            f.sync_all().unwrap();
        }

        let branch_id = Id::new([5u8; 16]).unwrap();
        let want: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![11u8; 13])).get_handle();
        let retracted: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![12u8; 13])).get_handle();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();

        // Interleave: want, enveloped blob, branch head, want + unwant, then
        // another enveloped blob.
        pile.want(WantRequest::blob(want)).unwrap();
        let d1 = vec![1u8; 300];
        let b1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(d1.clone()));
        let h1 = pile.put::<UnknownBlob, _>(b1).unwrap();
        pile.update(branch_id, None, Some(h1.transmute())).unwrap();
        pile.want(WantRequest::blob(retracted)).unwrap();
        pile.unwant(WantRequest::blob(retracted)).unwrap();
        let d2 = vec![2u8; 77];
        let b2: Blob<UnknownBlob> = Blob::new(Bytes::from_source(d2.clone()));
        let h2 = pile.put::<UnknownBlob, _>(b2).unwrap();
        pile.close().unwrap();

        // Fresh scan must walk the whole interleaved sequence.
        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();

        let reader = pile.reader().unwrap();
        let got_v1: Blob<UnknownBlob> = reader.get(v1_handle).unwrap();
        assert_eq!(got_v1.bytes.as_ref(), v1_data.as_slice());
        let got1: Blob<UnknownBlob> = reader.get(h1).unwrap();
        assert_eq!(got1.bytes.as_ref(), d1.as_slice());
        let got2: Blob<UnknownBlob> = reader.get(h2).unwrap();
        assert_eq!(got2.bytes.as_ref(), d2.as_slice());
        drop(reader);

        assert_eq!(pile.head(branch_id).unwrap(), Some(h1.transmute()));

        let pinned: HashSet<_> = pile.wants().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(pinned.len(), 1);
        assert!(pinned.contains(&WantRequest::blob(want)));
        assert!(!pinned.contains(&WantRequest::blob(retracted)));
        pile.close().unwrap();
    }

    /// [`PileRecords`] walks a mixed legacy/current pile record-by-record: every
    /// record kind appears in log order, offsets tile the file exactly, blob
    /// payloads are addressable through `data_offset`/`data_len`, and an
    /// unknown-marker tail surfaces as `Err(UnsupportedRecord)` — never a
    /// silent stop.
    #[test]
    fn pile_records_walks_mixed_pile_and_fails_loud() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "records.pile");

        // Hand-write a legacy V1 blob record first (64-byte header + pad).
        let v1_data = vec![9u8; 40];
        let v1_blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(v1_data.clone()));
        let v1_handle: Inline<Handle<UnknownBlob>> = v1_blob.get_handle();
        {
            let v1_hash: Inline<Hash<Blake3>> = v1_handle.into();
            let header = BlobHeader::new(42, v1_data.len() as u64, v1_hash);
            let pad = padding_for_blob(v1_data.len());
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(header.as_bytes()).unwrap();
            f.write_all(&v1_data).unwrap();
            f.write_all(&vec![0u8; pad]).unwrap();
            f.sync_all().unwrap();
        }

        let branch_id = Id::new([5u8; 16]).unwrap();
        let want: Inline<Handle<UnknownBlob>> =
            Blob::<UnknownBlob>::new(Bytes::from_source(vec![11u8; 13])).get_handle();

        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        let d1 = vec![1u8; 300];
        let b1: Blob<UnknownBlob> = Blob::new(Bytes::from_source(d1.clone()));
        let h1 = pile.put::<UnknownBlob, _>(b1).unwrap();
        pile.update(branch_id, None, Some(h1.transmute())).unwrap();
        pile.want(WantRequest::blob(want)).unwrap();
        pile.unwant(WantRequest::blob(want)).unwrap();
        pile.update(branch_id, Some(h1.transmute()), None).unwrap();
        pile.close().unwrap();

        let mut records = PileRecords::open(&path).unwrap();
        let bytes = records.bytes().clone();
        let decoded: Vec<PileRecord> = (&mut records)
            .map(|r| r.expect("well-formed pile decodes cleanly"))
            .collect();

        // Records tile the file: each starts where the previous ended.
        let mut expected_offset = 0;
        for record in &decoded {
            assert_eq!(record.offset, expected_offset);
            expected_offset += record.len;
        }
        assert_eq!(expected_offset, bytes.len());

        // Exact sequence: V1 blob, enveloped blob, branch set, legacy-compatible
        // blob want, blob retraction, branch tombstone.
        assert_eq!(decoded.len(), 6);
        match decoded[0].content {
            PileRecordContent::Blob {
                timestamp,
                hash,
                data_offset,
                data_len,
            } => {
                assert_eq!(timestamp, 42);
                assert_eq!(hash, v1_handle.into());
                assert_eq!(data_offset, BLOB_HEADER_LEN);
                assert_eq!(&bytes[data_offset..data_offset + data_len], &v1_data[..]);
            }
            other => panic!("expected V1 blob record, got {other:?}"),
        }
        match decoded[1].content {
            PileRecordContent::Blob {
                hash,
                data_offset,
                data_len,
                ..
            } => {
                assert_eq!(hash, h1.into());
                assert_eq!(data_offset, decoded[1].offset + ENVELOPE_HEADER_LEN);
                assert_eq!(&bytes[data_offset..data_offset + data_len], &d1[..]);
            }
            other => panic!("expected enveloped blob record, got {other:?}"),
        }
        match decoded[2].content {
            PileRecordContent::Branch {
                branch_id: bid,
                head,
            } => {
                assert_eq!(bid, branch_id);
                assert_eq!(head, h1.transmute());
            }
            other => panic!("expected branch record, got {other:?}"),
        }
        match decoded[3].content {
            PileRecordContent::WeakPin { handle } => assert_eq!(handle, want),
            other => panic!("expected weak-pin record, got {other:?}"),
        }
        match decoded[4].content {
            PileRecordContent::WeakUnpin { handle } => assert_eq!(handle, want),
            other => panic!("expected weak-unpin record, got {other:?}"),
        }
        match decoded[5].content {
            PileRecordContent::BranchTombstone { branch_id: bid } => {
                assert_eq!(bid, branch_id)
            }
            other => panic!("expected branch tombstone record, got {other:?}"),
        }

        // An unknown unenveloped record marker is an error at its offset, then
        // the iterator ends. Its length is unknowable, so it must not be called
        // corruption.
        let unknown_offset = std::fs::metadata(&path).unwrap().len() as usize;
        let unknown_marker = [0xFFu8; 16];
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(&unknown_marker).unwrap();
            f.sync_all().unwrap();
        }
        let mut records = PileRecords::open(&path).unwrap();
        let mut ok = 0;
        let err = loop {
            match records.next() {
                Some(Ok(_)) => ok += 1,
                Some(Err(e)) => break e,
                None => panic!("iterator ended without reporting the corrupt tail"),
            }
        };
        assert_eq!(ok, 6);
        match err {
            ReadError::UnsupportedRecord { offset, marker } => {
                assert_eq!(offset, unknown_offset);
                assert_eq!(marker, unknown_marker);
            }
            other => panic!("expected UnsupportedRecord, got {other:?}"),
        }
        assert!(records.next().is_none(), "iterator must end after an error");
    }

    // recover_grow test removed as growth strategy no longer exists

    /// Exercise the `ATOMIC_WRITE_LIMIT` fallback: an oversized blob must
    /// still round-trip correctly through the exclusive-lock multi-write
    /// path. Marked `#[ignore]` because the test allocates ~1 GiB and
    /// writes ~2 GiB to disk; run explicitly with
    /// `cargo test --release -- --ignored put_and_get_oversized_blob`.
    #[test]
    #[ignore]
    fn put_and_get_oversized_blob() {
        let dir = tempfile::tempdir().unwrap();
        let path = fresh_empty_pile_path(&dir, "pile.pile");

        // Slightly over the threshold so we land in the non-atomic branch.
        let size = ATOMIC_WRITE_LIMIT + 1_024;
        let mut data = vec![0u8; size];
        // Sprinkle some non-trivial pattern so `Bytes` equality has teeth.
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(13).wrapping_add(7);
        }

        let mut pile: Pile = Pile::open(&path).unwrap();
        let blob: Blob<UnknownBlob> = Blob::new(Bytes::from_source(data.clone()));
        let handle = pile.put::<UnknownBlob, _>(blob).unwrap();

        {
            let reader = pile.reader().unwrap();
            let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
            assert_eq!(fetched.bytes.len(), size);
            assert_eq!(fetched.bytes.as_ref(), data.as_slice());
        }

        pile.close().unwrap();

        // Round-trip across open+amputate to ensure the on-disk record
        // is fully self-describing and recoverable.
        let mut pile: Pile = Pile::open(&path).unwrap();
        pile.amputate().unwrap();
        let reader = pile.reader().unwrap();
        let fetched: Blob<UnknownBlob> = reader.get(handle).unwrap();
        assert_eq!(fetched.bytes.as_ref(), data.as_slice());
        pile.close().unwrap();
    }
}
