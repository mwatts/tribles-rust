//! Jerky-backed succinct building blocks for the index blobs.
//!
//! **Current status:** scaffolding. This module proves the
//! `anybytes::ByteArea` → `jerky::Serializable` round-trip works
//! in this crate before we cut the full succinct `BM25Index`
//! blob over to it. Each component here is a self-contained
//! zerocopy-loadable view; later iterations compose them into
//! `SuccinctBM25Index::try_from_blob` alongside a proper
//! `try_from_blob` of the naive format.
//!
//! The module is gated behind the `succinct` feature so the
//! naive path stays compilable without `jerky`.
//!
//! # Why this exists
//!
//! The naive `BM25Index` stores `doc_lens` as `Vec<u32>` — four
//! bytes per doc regardless of how short the doc actually is. At
//! 100k docs that's 0.4 MiB, small — but the pattern generalizes.
//! Once `doc_lens` loads zero-copy through jerky's
//! `CompactVector`, the same recipe swaps in for the big levers:
//! postings (144 MiB at 100k docs), term table, HNSW neighbour
//! arrays. Get the mechanics right here with a tiny surface
//! area, then expand.
//!
//! ```
//! use triblespace_search::succinct::SuccinctDocLens;
//!
//! let lens = vec![3u32, 7, 1, 15, 2];
//! let (bytes, meta) = SuccinctDocLens::build(&lens).expect("build");
//! let view = SuccinctDocLens::from_bytes(meta, bytes).expect("load");
//!
//! assert_eq!(view.len(), 5);
//! assert_eq!(view.get(0), Some(3));
//! assert_eq!(view.get(3), Some(15));
//! assert_eq!(view.to_vec(), lens);
//! ```

use anybytes::{ByteArea, Bytes};
use jerky::int_vectors::compact_vector::CompactVectorMeta;
use jerky::int_vectors::{CompactVector, CompactVectorBuilder};
use triblespace::core::blob::{Blob, BlobSchema, ToBlob, TryFromBlob};
use triblespace::core::id::Id;
use triblespace::core::id_hex;
use triblespace::core::metadata::ConstId;
use triblespace::core::value::RawValue;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::bm25::BM25Index;
use crate::FORMAT_VERSION;

/// Byte-layout mirror of [`CompactVectorMeta`] that's safe to
/// serialize through our own blob format.
///
/// jerky's `CompactVectorMeta` is `FromBytes + KnownLayout +
/// Immutable` but not `IntoBytes` (it has no way to opt in from
/// downstream), so we can't `as_bytes` it directly. This mirror
/// owns the derives we need on our side, and the From/To impls
/// document + check the layout equivalence (both are `repr(C)`
/// with four `u64`-sized fields on 64-bit platforms).
#[derive(Debug, Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
struct CompactVectorMetaOnDisk {
    len: u64,
    width: u64,
    handle_offset: u64,
    handle_len: u64,
}

const _: () = assert!(
    std::mem::size_of::<CompactVectorMetaOnDisk>() == 32,
    "CompactVectorMetaOnDisk must be 32 bytes",
);
const _: () = assert!(
    std::mem::size_of::<CompactVectorMeta>() == 32,
    "CompactVectorMeta must be 32 bytes (assumes 64-bit target)",
);

impl From<CompactVectorMeta> for CompactVectorMetaOnDisk {
    fn from(m: CompactVectorMeta) -> Self {
        Self {
            len: m.len as u64,
            width: m.width as u64,
            handle_offset: m.handle.offset as u64,
            handle_len: m.handle.len as u64,
        }
    }
}

impl CompactVectorMetaOnDisk {
    /// Convert back to jerky's meta. Because `SectionHandle` has
    /// a private `_type: PhantomData<T>` we can't construct one
    /// with a struct literal — but its layout is identical to
    /// `(usize, usize)` and it's `#[repr(C)]` with a known layout,
    /// so we transmute through a matching byte layout.
    fn to_jerky(self) -> CompactVectorMeta {
        // Build a 32-byte scratch in the on-disk layout, then
        // `read_from_bytes` it into `CompactVectorMeta` — sound
        // because both are `repr(C)`, both are 32 bytes, and
        // both share the same field order on 64-bit targets.
        let disk_bytes = self.as_bytes();
        CompactVectorMeta::read_from_bytes(disk_bytes)
            .expect("CompactVectorMeta has same 32-byte repr(C) layout on 64-bit")
    }
}

/// Errors produced by the succinct building blocks.
#[derive(Debug)]
pub enum SuccinctDocLensError {
    /// Failure propagating out of `anybytes::ByteArea`.
    Bytes(std::io::Error),
    /// Failure propagating out of `jerky` (build or view).
    Jerky(jerky::error::Error),
    /// Declared row count does not match the byte length.
    SizeMismatch {
        /// Total bytes available.
        bytes: usize,
        /// Declared rows × row width.
        expected: usize,
    },
}

impl std::fmt::Display for SuccinctDocLensError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(e) => write!(f, "succinct: bytes error: {e}"),
            Self::Jerky(e) => write!(f, "succinct: jerky error: {e}"),
            Self::SizeMismatch { bytes, expected } => write!(
                f,
                "succinct: size mismatch: have {bytes} bytes, declared needs {expected}"
            ),
        }
    }
}

impl std::error::Error for SuccinctDocLensError {}

impl From<std::io::Error> for SuccinctDocLensError {
    fn from(e: std::io::Error) -> Self {
        Self::Bytes(e)
    }
}

impl From<jerky::error::Error> for SuccinctDocLensError {
    fn from(e: jerky::error::Error) -> Self {
        Self::Jerky(e)
    }
}

/// A zero-copy view over per-document length counts, bit-packed
/// via [`jerky::int_vectors::CompactVector`].
///
/// The bit-width is chosen at build time as `ceil(log2(max+1))`,
/// so short-doc corpora (common for wiki fragments) pay a fraction
/// of the `u32` cost.
#[derive(Debug)]
pub struct SuccinctDocLens {
    inner: CompactVector,
}

impl SuccinctDocLens {
    /// Build a succinct doc-lens table. Returns the frozen
    /// [`Bytes`] region and [`CompactVectorMeta`] needed to
    /// reconstruct a view with [`Self::from_bytes`].
    pub fn build(lens: &[u32]) -> Result<(Bytes, CompactVectorMeta), SuccinctDocLensError> {
        let width = required_width(lens);
        let mut area = ByteArea::new()?;
        let mut sections = area.sections();
        let mut builder = CompactVectorBuilder::with_capacity(lens.len(), width, &mut sections)?;
        builder.set_ints(0..lens.len(), lens.iter().map(|&n| n as usize))?;
        let cv = builder.freeze();
        let meta = cv.metadata();
        let bytes = area.freeze()?;
        Ok((bytes, meta))
    }

    /// Reconstruct a view from the frozen bytes + metadata.
    pub fn from_bytes(
        meta: CompactVectorMeta,
        bytes: Bytes,
    ) -> Result<Self, SuccinctDocLensError> {
        let inner = CompactVector::from_bytes(meta, bytes)?;
        Ok(Self { inner })
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Document length at position `i`, or `None` if out of range.
    pub fn get(&self, i: usize) -> Option<u32> {
        // get_int returns usize; doc_lens fit in u32 by construction.
        self.inner.get_int(i).map(|n| n as u32)
    }

    /// Collect into a `Vec<u32>` for ergonomic inspection.
    pub fn to_vec(&self) -> Vec<u32> {
        self.inner.to_vec().into_iter().map(|n| n as u32).collect()
    }

    /// Bit width per entry (log₂ of max value + 1).
    pub fn width(&self) -> usize {
        self.inner.metadata().width
    }
}

/// Number of bits needed to represent every entry in `lens`.
/// Always at least 1 so `CompactVectorBuilder::with_capacity`
/// accepts it (width ∈ `1..=64`).
fn required_width(lens: &[u32]) -> usize {
    let max = lens.iter().copied().max().unwrap_or(0);
    match max {
        0 => 1,
        _ => 32 - (max as u32).leading_zeros() as usize,
    }
}

/// Fixed-row-size table of bytes, zero-copy loaded from a
/// contiguous [`Bytes`] region.
///
/// Parameterized on the row width `N`. Used for BM25's doc-id
/// table (`N = 16`) and term table (`N = 32`). No jerky
/// compression — each row is already fixed-size and can be
/// sliced out of the raw bytes with a pointer, which is what
/// makes this "succinct" at all: the caller's `Bytes` handle is
/// the ground truth and no data is duplicated into structs.
///
/// The caller is responsible for sorting the table at build time
/// if they want binary-search lookups (see [`Self::binary_search`]).
///
/// # Build / view
///
/// ```
/// use triblespace_search::succinct::FixedBytesTable;
///
/// let rows = vec![[1u8; 16], [3u8; 16], [7u8; 16]];
/// let bytes = FixedBytesTable::<16>::build(&rows);
/// let view = FixedBytesTable::<16>::from_bytes(bytes, rows.len()).unwrap();
/// assert_eq!(view.binary_search(&[3u8; 16]), Ok(1));
/// assert_eq!(view.binary_search(&[5u8; 16]), Err(2));
/// ```
#[derive(Debug)]
pub struct FixedBytesTable<const N: usize> {
    bytes: Bytes,
    len: usize,
}

impl<const N: usize> FixedBytesTable<N> {
    /// Serialize `rows` into a flat `Bytes` buffer. The companion
    /// [`Self::from_bytes`] reconstructs a view; callers persist
    /// `rows.len()` out-of-band (e.g., in a BlobHeader).
    pub fn build(rows: &[[u8; N]]) -> Bytes {
        let mut buf = Vec::with_capacity(rows.len() * N);
        for row in rows {
            buf.extend_from_slice(row);
        }
        Bytes::from_source(buf)
    }

    /// View `bytes` as `len` rows of `N` bytes each.
    pub fn from_bytes(bytes: Bytes, len: usize) -> Result<Self, SuccinctDocLensError> {
        let expected = len.checked_mul(N).ok_or(SuccinctDocLensError::SizeMismatch {
            bytes: bytes.len(),
            expected: usize::MAX,
        })?;
        if bytes.len() < expected {
            return Err(SuccinctDocLensError::SizeMismatch {
                bytes: bytes.len(),
                expected,
            });
        }
        Ok(Self { bytes, len })
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Row at position `i`, or `None` if out of range. Zero-copy:
    /// returns a reference into the backing `Bytes`.
    pub fn get(&self, i: usize) -> Option<&[u8; N]> {
        if i >= self.len {
            return None;
        }
        let start = i * N;
        let end = start + N;
        // A byte array has trivial alignment, so the slice
        // directly coerces to `&[u8; N]`.
        self.bytes[start..end].try_into().ok()
    }

    /// Binary-search for `needle`. Returns `Ok(index)` if found,
    /// `Err(insertion_point)` otherwise. Requires the table to be
    /// sorted at build time.
    pub fn binary_search(&self, needle: &[u8; N]) -> std::result::Result<usize, usize> {
        let mut lo = 0;
        let mut hi = self.len;
        while lo < hi {
            let mid = (lo + hi) / 2;
            // `.expect` is safe: mid < hi ≤ self.len, and get()
            // validates against that.
            let row = self.get(mid).expect("binary search in-bounds");
            match row.as_slice().cmp(needle.as_slice()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }

    /// Collect into a `Vec<[u8; N]>` for tests / inspection.
    pub fn to_vec(&self) -> Vec<[u8; N]> {
        (0..self.len).map(|i| *self.get(i).unwrap()).collect()
    }
}

/// Per-term posting lists backed by jerky primitives.
///
/// Splits into two zero-copy regions:
/// - **`idx_bytes`** — carries the jerky `CompactVector`s for
///   `doc_idx` (per-posting document index, width
///   `ceil(log2(n_docs+1))`) and `offsets` (per-term cumulative
///   offset, width `ceil(log2(total+1))`). Shared `ByteArea`
///   means both load from the same frozen bytes.
/// - **`score_bytes`** — flat little-endian f32 scores, one per
///   posting in the same order as `doc_idx`. Kept separate (and
///   uncompressed) for now; a quantized variant is a future
///   optimization.
///
/// [`build`] returns both `Bytes` plus the [`SuccinctPostingsMeta`]
/// needed to rehydrate. The caller decides how the two regions
/// land in the final blob (typically: record their offsets +
/// lengths in the blob header, concatenate in the body).
///
/// [`build`]: Self::build
#[derive(Debug)]
pub struct SuccinctPostings {
    doc_idx: CompactVector,
    offsets: CompactVector,
    scores: Bytes,
    n_terms: usize,
}

/// Serialized layout metadata for [`SuccinctPostings`].
#[derive(Debug, Clone, Copy)]
pub struct SuccinctPostingsMeta {
    /// Meta for the `doc_idx` CompactVector (carved out of
    /// `idx_bytes`).
    pub doc_idx: CompactVectorMeta,
    /// Meta for the `offsets` CompactVector (carved out of
    /// `idx_bytes`).
    pub offsets: CompactVectorMeta,
    /// Number of terms (== `offsets.len() - 1`).
    pub n_terms: u64,
}

impl SuccinctPostings {
    /// Serialize `lists[t]` as the posting list for term index
    /// `t`. Returns `(idx_bytes, score_bytes, meta)`. `n_docs` is
    /// the total document count (sets the bit width for
    /// `doc_idx`).
    pub fn build(
        lists: &[Vec<(u32, f32)>],
        n_docs: u32,
    ) -> Result<(Bytes, Bytes, SuccinctPostingsMeta), SuccinctDocLensError> {
        let total: usize = lists.iter().map(|l| l.len()).sum();
        let doc_idx_width = width_for(n_docs as usize + 1);
        let offsets_width = width_for(total + 1);

        // ─ jerky-backed sections ─
        let mut area = ByteArea::new()?;
        let mut sections = area.sections();

        let mut doc_idx_b =
            CompactVectorBuilder::with_capacity(total, doc_idx_width, &mut sections)?;
        let mut pos = 0usize;
        for list in lists {
            for &(idx, _) in list {
                doc_idx_b.set_int(pos, idx as usize)?;
                pos += 1;
            }
        }
        let doc_idx = doc_idx_b.freeze();
        let doc_idx_meta = doc_idx.metadata();

        let mut offsets_b =
            CompactVectorBuilder::with_capacity(lists.len() + 1, offsets_width, &mut sections)?;
        offsets_b.set_int(0, 0)?;
        let mut cum = 0usize;
        for (i, list) in lists.iter().enumerate() {
            cum += list.len();
            offsets_b.set_int(i + 1, cum)?;
        }
        let offsets = offsets_b.freeze();
        let offsets_meta = offsets.metadata();

        // Drop sections + writer so area can freeze.
        drop((doc_idx, offsets));
        drop(sections);
        let idx_bytes = area.freeze()?;

        // ─ scores: flat f32 LE buffer, separate Bytes ─
        let mut score_buf = Vec::with_capacity(total * std::mem::size_of::<f32>());
        for list in lists {
            for &(_, score) in list {
                score_buf.extend_from_slice(&score.to_le_bytes());
            }
        }
        let score_bytes = Bytes::from_source(score_buf);

        let meta = SuccinctPostingsMeta {
            doc_idx: doc_idx_meta,
            offsets: offsets_meta,
            n_terms: lists.len() as u64,
        };
        Ok((idx_bytes, score_bytes, meta))
    }

    /// Reconstruct from metadata + both byte regions.
    pub fn from_bytes(
        meta: SuccinctPostingsMeta,
        idx_bytes: Bytes,
        score_bytes: Bytes,
    ) -> Result<Self, SuccinctDocLensError> {
        let doc_idx = CompactVector::from_bytes(meta.doc_idx, idx_bytes.clone())?;
        let offsets = CompactVector::from_bytes(meta.offsets, idx_bytes)?;
        let expected = doc_idx.len() * std::mem::size_of::<f32>();
        if score_bytes.len() < expected {
            return Err(SuccinctDocLensError::SizeMismatch {
                bytes: score_bytes.len(),
                expected,
            });
        }
        Ok(Self {
            doc_idx,
            offsets,
            scores: score_bytes,
            n_terms: meta.n_terms as usize,
        })
    }

    /// Number of terms.
    pub fn term_count(&self) -> usize {
        self.n_terms
    }

    /// Number of postings for term `t`. `None` if out of range.
    pub fn posting_count(&self, t: usize) -> Option<usize> {
        if t >= self.n_terms {
            return None;
        }
        let start = self.offsets.get_int(t)?;
        let end = self.offsets.get_int(t + 1)?;
        Some(end - start)
    }

    /// Iterate `(doc_idx, score)` postings for term `t`.
    pub fn postings_for(
        &self,
        t: usize,
    ) -> Option<impl Iterator<Item = (u32, f32)> + '_> {
        if t >= self.n_terms {
            return None;
        }
        let start = self.offsets.get_int(t)?;
        let end = self.offsets.get_int(t + 1)?;
        Some((start..end).map(move |i| {
            let idx = self.doc_idx.get_int(i).unwrap() as u32;
            let off = i * std::mem::size_of::<f32>();
            let score = f32::from_le_bytes(
                self.scores[off..off + 4].try_into().unwrap(),
            );
            (idx, score)
        }))
    }
}

/// Minimum bit width to represent the value `n` (or 1 if 0).
fn width_for(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

/// Zero-copy, jerky-backed BM25 index.
///
/// Same query surface as [`crate::bm25::BM25Index`], but postings
/// + doc_lens live in bit-packed [`CompactVector`]s and the
/// doc-id / term tables are sliced directly out of an
/// [`anybytes::Bytes`] region without copying.
///
/// For 100k wiki fragments the naive blob is ~157 MiB (91 % of
/// which is postings); this representation cuts the posting
/// `doc_idx` from 32 bits → `ceil(log2(n_docs))` bits — about
/// 2.1× on the postings table and 1.3× on the whole blob before
/// we touch score quantization.
///
/// Built via [`Self::from_naive`] (takes a fully-materialized
/// [`BM25Index`] and re-encodes it). A direct succinct builder
/// (skipping the naive intermediate) is a later optimization.
#[derive(Debug)]
pub struct SuccinctBM25Index {
    doc_ids: FixedBytesTable<16>,
    doc_lens: SuccinctDocLens,
    terms: FixedBytesTable<32>,
    postings: SuccinctPostings,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
}

impl SuccinctBM25Index {
    /// Re-encode a naive [`BM25Index`] into the succinct form.
    /// The postings + per-doc counts shrink; scores stay as raw
    /// f32 for now.
    pub fn from_naive(idx: &BM25Index) -> Result<Self, SuccinctDocLensError> {
        // doc_ids: flat 16-byte rows.
        let doc_id_rows: Vec<[u8; 16]> =
            (0..idx.doc_count()).map(|i| *idx.doc_ids()[i].as_ref()).collect();
        let doc_ids_bytes = FixedBytesTable::<16>::build(&doc_id_rows);
        let doc_ids =
            FixedBytesTable::<16>::from_bytes(doc_ids_bytes, doc_id_rows.len())?;

        // terms: flat 32-byte rows, sorted (idx.terms() guarantees).
        let term_rows: Vec<[u8; 32]> = idx.terms_slice().to_vec();
        let terms_bytes = FixedBytesTable::<32>::build(&term_rows);
        let terms = FixedBytesTable::<32>::from_bytes(terms_bytes, term_rows.len())?;

        // doc_lens: jerky CompactVector.
        let (doc_lens_bytes, doc_lens_meta) = SuccinctDocLens::build(idx.doc_lens())?;
        let doc_lens = SuccinctDocLens::from_bytes(doc_lens_meta, doc_lens_bytes)?;

        // postings: rebuild the per-term posting lists.
        let lists: Vec<Vec<(u32, f32)>> = (0..idx.term_count())
            .map(|t| idx.postings_for(t).to_vec())
            .collect();
        let (idx_bytes, score_bytes, postings_meta) =
            SuccinctPostings::build(&lists, idx.doc_count() as u32)?;
        let postings = SuccinctPostings::from_bytes(postings_meta, idx_bytes, score_bytes)?;

        Ok(Self {
            doc_ids,
            doc_lens,
            terms,
            postings,
            avg_doc_len: idx.avg_doc_len(),
            k1: idx.k1(),
            b: idx.b(),
        })
    }

    /// Number of documents.
    pub fn doc_count(&self) -> usize {
        self.doc_ids.len()
    }

    /// Number of distinct terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Average document length used at build time.
    pub fn avg_doc_len(&self) -> f32 {
        self.avg_doc_len
    }

    /// BM25 `k1` used at build time.
    pub fn k1(&self) -> f32 {
        self.k1
    }

    /// BM25 `b` used at build time.
    pub fn b(&self) -> f32 {
        self.b
    }

    /// Length of doc `i`, or `None` if out of range.
    pub fn doc_len(&self, i: usize) -> Option<u32> {
        self.doc_lens.get(i)
    }

    /// Number of documents containing `term`.
    pub fn doc_frequency(&self, term: &RawValue) -> usize {
        match self.terms.binary_search(term) {
            Ok(t) => self.postings.posting_count(t).unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Iterate `(doc_id, score)` postings for `term`. Empty if
    /// the term is absent.
    pub fn query_term<'a>(
        &'a self,
        term: &RawValue,
    ) -> Box<dyn Iterator<Item = (Id, f32)> + 'a> {
        match self.terms.binary_search(term) {
            Ok(t) => match self.postings.postings_for(t) {
                Some(iter) => Box::new(iter.map(move |(doc_idx, score)| {
                    let raw = self.doc_ids.get(doc_idx as usize).expect("doc_idx in range");
                    let id = Id::new(*raw).expect("non-nil doc_id");
                    (id, score)
                })),
                None => Box::new(std::iter::empty()),
            },
            Err(_) => Box::new(std::iter::empty()),
        }
    }

    /// Serialize to a self-contained blob. The layout:
    ///
    /// ```text
    /// [header               ] SUCCINCT_HEADER_LEN B
    /// [doc_ids              ] n_docs × 16 B
    /// [terms                ] n_terms × 32 B
    /// [doc_lens_bytes       ] variable (CompactVector body)
    /// [postings_idx_bytes   ] variable (doc_idx + offsets CompactVectors)
    /// [postings_score_bytes ] total_postings × 4 B
    /// ```
    ///
    /// The header carries all scalar parameters plus the three
    /// `CompactVectorMeta` structures (doc_lens, postings.doc_idx,
    /// postings.offsets) and the byte-offset/length of each
    /// section.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_docs = self.doc_count() as u64;
        let n_terms = self.term_count() as u64;

        // Capture component metas + bytes.
        let doc_lens_meta: CompactVectorMetaOnDisk =
            self.doc_lens.inner.metadata().into();
        let postings_doc_idx_meta: CompactVectorMetaOnDisk =
            self.postings.doc_idx.metadata().into();
        let postings_offsets_meta: CompactVectorMetaOnDisk =
            self.postings.offsets.metadata().into();

        // Reserve room for the header, then append sections.
        let doc_ids_bytes: Vec<u8> = (0..self.doc_count())
            .flat_map(|i| self.doc_ids.get(i).copied().unwrap_or([0u8; 16]))
            .collect();
        let terms_bytes: Vec<u8> = (0..self.term_count())
            .flat_map(|i| self.terms.get(i).copied().unwrap_or([0u8; 32]))
            .collect();
        // doc_lens: re-serialize via ByteArea so we own the bytes.
        let (doc_lens_region, _) =
            SuccinctDocLens::build(&self.doc_lens.to_vec()).expect("re-serialize");
        // postings: re-serialize via ByteArea as well. Easiest to
        // walk self.postings and round-trip through build().
        let lists: Vec<Vec<(u32, f32)>> = (0..self.term_count())
            .map(|t| self.postings.postings_for(t).unwrap().collect())
            .collect();
        let (postings_idx_region, postings_score_region, _) =
            SuccinctPostings::build(&lists, self.doc_count() as u32).expect("re-serialize");

        // Offsets inside the blob body (relative to end of header).
        let doc_ids_off = 0u64;
        let doc_ids_len = doc_ids_bytes.len() as u64;
        let terms_off = doc_ids_off + doc_ids_len;
        let terms_len = terms_bytes.len() as u64;
        let doc_lens_off = terms_off + terms_len;
        let doc_lens_len = doc_lens_region.len() as u64;
        let postings_idx_off = doc_lens_off + doc_lens_len;
        let postings_idx_len = postings_idx_region.len() as u64;
        let postings_score_off = postings_idx_off + postings_idx_len;
        let postings_score_len = postings_score_region.len() as u64;

        let body_len = postings_score_off + postings_score_len;
        let mut buf = Vec::with_capacity(SUCCINCT_HEADER_LEN + body_len as usize);

        // ── header ────────────────────────────────────────────
        buf.extend_from_slice(&SUCCINCT_MAGIC.to_le_bytes()); // 4
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes()); // 2
        buf.extend_from_slice(&0u16.to_le_bytes()); // reserved, 2
        buf.extend_from_slice(&self.avg_doc_len.to_le_bytes()); // 4
        buf.extend_from_slice(&self.k1.to_le_bytes()); // 4
        buf.extend_from_slice(&self.b.to_le_bytes()); // 4
        buf.extend_from_slice(&n_docs.to_le_bytes()); // 8
        buf.extend_from_slice(&n_terms.to_le_bytes()); // 8
        // doc_lens meta (32)
        buf.extend_from_slice(doc_lens_meta.as_bytes());
        // postings doc_idx meta (32)
        buf.extend_from_slice(postings_doc_idx_meta.as_bytes());
        // postings offsets meta (32)
        buf.extend_from_slice(postings_offsets_meta.as_bytes());
        // section offsets/lengths (10 × 8 = 80)
        buf.extend_from_slice(&doc_ids_off.to_le_bytes());
        buf.extend_from_slice(&doc_ids_len.to_le_bytes());
        buf.extend_from_slice(&terms_off.to_le_bytes());
        buf.extend_from_slice(&terms_len.to_le_bytes());
        buf.extend_from_slice(&doc_lens_off.to_le_bytes());
        buf.extend_from_slice(&doc_lens_len.to_le_bytes());
        buf.extend_from_slice(&postings_idx_off.to_le_bytes());
        buf.extend_from_slice(&postings_idx_len.to_le_bytes());
        buf.extend_from_slice(&postings_score_off.to_le_bytes());
        buf.extend_from_slice(&postings_score_len.to_le_bytes());
        debug_assert_eq!(buf.len(), SUCCINCT_HEADER_LEN);

        // ── body ──────────────────────────────────────────────
        buf.extend_from_slice(&doc_ids_bytes);
        buf.extend_from_slice(&terms_bytes);
        buf.extend_from_slice(&doc_lens_region);
        buf.extend_from_slice(&postings_idx_region);
        buf.extend_from_slice(&postings_score_region);
        buf
    }

    /// Reload from bytes previously produced by [`Self::to_bytes`].
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, SuccinctLoadError> {
        if bytes.len() < SUCCINCT_HEADER_LEN {
            return Err(SuccinctLoadError::ShortHeader);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != SUCCINCT_MAGIC {
            return Err(SuccinctLoadError::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(SuccinctLoadError::VersionMismatch(version));
        }
        // reserved [6..8]
        let avg_doc_len = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let k1 = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let b = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let n_docs = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
        let n_terms = u64::from_le_bytes(bytes[28..36].try_into().unwrap()) as usize;

        let doc_lens_meta = CompactVectorMetaOnDisk::read_from_bytes(&bytes[36..68])
            .map_err(|_| SuccinctLoadError::BadMeta("doc_lens"))?
            .to_jerky();
        let postings_doc_idx_meta =
            CompactVectorMetaOnDisk::read_from_bytes(&bytes[68..100])
                .map_err(|_| SuccinctLoadError::BadMeta("postings_doc_idx"))?
                .to_jerky();
        let postings_offsets_meta =
            CompactVectorMetaOnDisk::read_from_bytes(&bytes[100..132])
                .map_err(|_| SuccinctLoadError::BadMeta("postings_offsets"))?
                .to_jerky();

        // Section offsets/lengths.
        let read_u64 = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let doc_ids_off = read_u64(132) as usize;
        let doc_ids_len = read_u64(140) as usize;
        let terms_off = read_u64(148) as usize;
        let terms_len = read_u64(156) as usize;
        let doc_lens_off = read_u64(164) as usize;
        let doc_lens_len = read_u64(172) as usize;
        let postings_idx_off = read_u64(180) as usize;
        let postings_idx_len = read_u64(188) as usize;
        let postings_score_off = read_u64(196) as usize;
        let postings_score_len = read_u64(204) as usize;
        debug_assert_eq!(SUCCINCT_HEADER_LEN, 212);

        let body_start = SUCCINCT_HEADER_LEN;
        let body = &bytes[body_start..];
        let check = |end: usize, name: &'static str| -> Result<(), SuccinctLoadError> {
            if end > body.len() {
                Err(SuccinctLoadError::TruncatedSection(name))
            } else {
                Ok(())
            }
        };
        check(doc_ids_off + doc_ids_len, "doc_ids")?;
        check(terms_off + terms_len, "terms")?;
        check(doc_lens_off + doc_lens_len, "doc_lens")?;
        check(postings_idx_off + postings_idx_len, "postings_idx")?;
        check(postings_score_off + postings_score_len, "postings_score")?;

        // Wrap the whole body in a single Bytes region so we
        // share backing storage across sub-views.
        let body_bytes = Bytes::from_source(body.to_vec());
        let doc_ids_bytes = body_bytes.slice(doc_ids_off..doc_ids_off + doc_ids_len);
        let terms_bytes = body_bytes.slice(terms_off..terms_off + terms_len);
        let doc_lens_bytes = body_bytes.slice(doc_lens_off..doc_lens_off + doc_lens_len);
        let postings_idx_bytes =
            body_bytes.slice(postings_idx_off..postings_idx_off + postings_idx_len);
        let postings_score_bytes =
            body_bytes.slice(postings_score_off..postings_score_off + postings_score_len);

        let doc_ids = FixedBytesTable::<16>::from_bytes(doc_ids_bytes, n_docs)
            .map_err(|_| SuccinctLoadError::TruncatedSection("doc_ids"))?;
        let terms = FixedBytesTable::<32>::from_bytes(terms_bytes, n_terms)
            .map_err(|_| SuccinctLoadError::TruncatedSection("terms"))?;
        let doc_lens = SuccinctDocLens::from_bytes(doc_lens_meta, doc_lens_bytes)
            .map_err(|_| SuccinctLoadError::TruncatedSection("doc_lens"))?;
        let postings_meta = SuccinctPostingsMeta {
            doc_idx: postings_doc_idx_meta,
            offsets: postings_offsets_meta,
            n_terms: n_terms as u64,
        };
        let postings = SuccinctPostings::from_bytes(
            postings_meta,
            postings_idx_bytes,
            postings_score_bytes,
        )
        .map_err(|_| SuccinctLoadError::TruncatedSection("postings"))?;

        Ok(Self {
            doc_ids,
            doc_lens,
            terms,
            postings,
            avg_doc_len,
            k1,
            b,
        })
    }
}

/// Magic header tag for SuccinctBM25Index blobs.
const SUCCINCT_MAGIC: u32 = u32::from_le_bytes(*b"SB25");
/// Header length in bytes.
///
/// Layout: 4 magic + 2 version + 2 reserved + 4 avg_doc_len +
/// 4 k1 + 4 b + 8 n_docs + 8 n_terms + 3 × 32 CompactVectorMeta
/// + 10 × 8 section (offset, len) = 212.
const SUCCINCT_HEADER_LEN: usize = 212;

/// Errors loading a `SuccinctBM25Index` blob.
#[derive(Debug, Clone, PartialEq)]
pub enum SuccinctLoadError {
    /// Blob shorter than the fixed header.
    ShortHeader,
    /// Magic bytes don't match `"SB25"`.
    BadMagic,
    /// Unknown format version.
    VersionMismatch(u16),
    /// A declared section extends past the blob body.
    TruncatedSection(&'static str),
    /// A `CompactVectorMeta` in the header couldn't be parsed.
    BadMeta(&'static str),
}

impl std::fmt::Display for SuccinctLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortHeader => write!(f, "SB25 blob shorter than header"),
            Self::BadMagic => write!(f, "SB25 blob: magic mismatch"),
            Self::VersionMismatch(v) => {
                write!(f, "SB25 blob: version {v} (expected {FORMAT_VERSION})")
            }
            Self::TruncatedSection(name) => {
                write!(f, "SB25 blob: truncated section `{name}`")
            }
            Self::BadMeta(name) => write!(f, "SB25 blob: bad CompactVectorMeta `{name}`"),
        }
    }
}

impl std::error::Error for SuccinctLoadError {}

/// Content-addressed [`BlobSchema`] marker for the real succinct
/// BM25 blob format (SB25 / 212 B header + bit-packed body).
///
/// Schema id minted via `trible genid`:
/// `68C03764D04D05DF65E49589FBBA1441`. Distinct from
/// [`crate::bm25::SuccinctBM25Index`] (the legacy marker that
/// currently wraps the naive format during the succinct
/// transition).
pub enum SuccinctBM25Blob {}

impl ConstId for SuccinctBM25Blob {
    const ID: Id = id_hex!("68C03764D04D05DF65E49589FBBA1441");
}

impl BlobSchema for SuccinctBM25Blob {}

impl ToBlob<SuccinctBM25Blob> for &SuccinctBM25Index {
    fn to_blob(self) -> Blob<SuccinctBM25Blob> {
        Blob::new(Bytes::from_source(self.to_bytes()))
    }
}

impl ToBlob<SuccinctBM25Blob> for SuccinctBM25Index {
    fn to_blob(self) -> Blob<SuccinctBM25Blob> {
        (&self).to_blob()
    }
}

impl TryFromBlob<SuccinctBM25Blob> for SuccinctBM25Index {
    type Error = SuccinctLoadError;

    fn try_from_blob(blob: Blob<SuccinctBM25Blob>) -> Result<Self, Self::Error> {
        SuccinctBM25Index::try_from_bytes(blob.bytes.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_roundtrip() {
        let (bytes, meta) = SuccinctDocLens::build(&[]).unwrap();
        let view = SuccinctDocLens::from_bytes(meta, bytes).unwrap();
        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
        assert_eq!(view.get(0), None);
    }

    #[test]
    fn small_roundtrip() {
        let lens = vec![3u32, 0, 7, 1, 15];
        let (bytes, meta) = SuccinctDocLens::build(&lens).unwrap();
        let view = SuccinctDocLens::from_bytes(meta, bytes).unwrap();
        assert_eq!(view.len(), lens.len());
        for (i, &n) in lens.iter().enumerate() {
            assert_eq!(view.get(i), Some(n), "mismatch at {i}");
        }
        assert_eq!(view.to_vec(), lens);
    }

    #[test]
    fn out_of_range_is_none() {
        let (bytes, meta) = SuccinctDocLens::build(&[1u32, 2, 3]).unwrap();
        let view = SuccinctDocLens::from_bytes(meta, bytes).unwrap();
        assert_eq!(view.get(3), None);
        assert_eq!(view.get(99), None);
    }

    #[test]
    fn width_matches_max_value() {
        // max 15 -> 4 bits, max 16 -> 5 bits.
        assert_eq!(required_width(&[0, 15, 7, 3]), 4);
        assert_eq!(required_width(&[0, 16]), 5);
        // all zeros -> width 1 (CompactVector min).
        assert_eq!(required_width(&[0, 0, 0]), 1);
        // empty -> width 1.
        assert_eq!(required_width(&[]), 1);
    }

    #[test]
    fn large_lens_pack_correctly() {
        // Lengths up to 1_000_000 — 20 bits per entry. Round-trip
        // must preserve the full range.
        let lens: Vec<u32> = (0..200).map(|i| i * 5_000).collect();
        let (bytes, meta) = SuccinctDocLens::build(&lens).unwrap();
        let view = SuccinctDocLens::from_bytes(meta, bytes).unwrap();
        assert_eq!(view.to_vec(), lens);
        assert_eq!(view.width(), 20); // log2(995_000) rounded up.
    }

    #[test]
    fn bit_packing_beats_raw_u32() {
        // For a corpus where all docs are ≤255 tokens, each entry
        // packs into 8 bits instead of 32 — 4x smaller. Not an
        // exact assertion on bytes (CompactVector has small fixed
        // overhead), but the frozen bytes should be clearly
        // smaller than 4 * n.
        let lens: Vec<u32> = (0..1000).map(|i| (i % 200) as u32).collect();
        let (bytes, _meta) = SuccinctDocLens::build(&lens).unwrap();
        assert!(
            bytes.len() < lens.len() * 4,
            "succinct {} < naive {}",
            bytes.len(),
            lens.len() * 4
        );
    }

    #[test]
    fn fixed_table_16_roundtrip() {
        let rows = vec![[7u8; 16], [3u8; 16], [99u8; 16]];
        let bytes = FixedBytesTable::<16>::build(&rows);
        let view = FixedBytesTable::<16>::from_bytes(bytes, rows.len()).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some(&[7u8; 16]));
        assert_eq!(view.get(2), Some(&[99u8; 16]));
        assert_eq!(view.get(3), None);
    }

    #[test]
    fn fixed_table_32_roundtrip() {
        let rows = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let bytes = FixedBytesTable::<32>::build(&rows);
        let view = FixedBytesTable::<32>::from_bytes(bytes, rows.len()).unwrap();
        assert_eq!(view.to_vec(), rows);
    }

    #[test]
    fn fixed_table_binary_search_finds_term() {
        // Sorted table of 32-byte terms — what BM25's term table
        // wants.
        let mut rows = vec![
            [5u8; 32],
            [1u8; 32],
            [9u8; 32],
            [3u8; 32],
        ];
        rows.sort();
        let bytes = FixedBytesTable::<32>::build(&rows);
        let view = FixedBytesTable::<32>::from_bytes(bytes, rows.len()).unwrap();
        assert_eq!(view.binary_search(&[3u8; 32]), Ok(1));
        assert_eq!(view.binary_search(&[5u8; 32]), Ok(2));
        assert_eq!(view.binary_search(&[7u8; 32]), Err(3));
    }

    #[test]
    fn fixed_table_rejects_size_mismatch() {
        // 3 rows × 16 bytes = 48 bytes; claim 4 rows -> error.
        let rows = vec![[0u8; 16], [1u8; 16], [2u8; 16]];
        let bytes = FixedBytesTable::<16>::build(&rows);
        let err = FixedBytesTable::<16>::from_bytes(bytes, 4).unwrap_err();
        assert!(matches!(err, SuccinctDocLensError::SizeMismatch { .. }));
    }

    #[test]
    fn fixed_table_empty_is_empty() {
        let bytes = FixedBytesTable::<16>::build(&[]);
        let view = FixedBytesTable::<16>::from_bytes(bytes, 0).unwrap();
        assert!(view.is_empty());
        assert_eq!(view.get(0), None);
        assert_eq!(view.binary_search(&[0u8; 16]), Err(0));
    }

    #[test]
    fn postings_roundtrip_simple() {
        let lists = vec![
            vec![(0u32, 1.5f32), (3, 0.75), (7, 2.0)],
            vec![(1, 0.5), (2, 3.25)],
            vec![],
            vec![(4, 9.0)],
        ];
        let (idx_bytes, score_bytes, meta) =
            SuccinctPostings::build(&lists, 8).unwrap();
        let view =
            SuccinctPostings::from_bytes(meta, idx_bytes, score_bytes).unwrap();
        assert_eq!(view.term_count(), 4);
        assert_eq!(view.posting_count(0), Some(3));
        assert_eq!(view.posting_count(1), Some(2));
        assert_eq!(view.posting_count(2), Some(0));
        assert_eq!(view.posting_count(3), Some(1));
        assert_eq!(view.posting_count(4), None);

        for (t, expected) in lists.iter().enumerate() {
            let got: Vec<(u32, f32)> = view.postings_for(t).unwrap().collect();
            assert_eq!(&got, expected, "term {t}");
        }
    }

    #[test]
    fn postings_empty_corpus() {
        let (idx_bytes, score_bytes, meta) =
            SuccinctPostings::build(&[] as &[Vec<(u32, f32)>], 0).unwrap();
        let view =
            SuccinctPostings::from_bytes(meta, idx_bytes, score_bytes).unwrap();
        assert_eq!(view.term_count(), 0);
        assert!(view.postings_for(0).is_none());
    }

    #[test]
    fn succinct_bm25_matches_naive_on_sample() {
        use crate::bm25::BM25Builder;
        use crate::tokens::hash_tokens;
        use triblespace::core::id::Id;

        fn iid(byte: u8) -> Id {
            Id::new([byte; 16]).unwrap()
        }

        let mut b = BM25Builder::new();
        b.insert(iid(1), hash_tokens("the quick brown fox"));
        b.insert(iid(2), hash_tokens("the lazy brown dog"));
        b.insert(iid(3), hash_tokens("quick silver fox jumps"));
        b.insert(iid(4), hash_tokens("unrelated filler content"));
        let naive = b.build();
        let succinct = SuccinctBM25Index::from_naive(&naive).unwrap();

        assert_eq!(succinct.doc_count(), naive.doc_count());
        assert_eq!(succinct.term_count(), naive.term_count());
        assert_eq!(succinct.k1(), naive.k1());
        assert_eq!(succinct.b(), naive.b());
        assert!((succinct.avg_doc_len() - naive.avg_doc_len()).abs() < 1e-6);

        // Every stored term must produce identical postings.
        for term in naive.terms_slice() {
            let n: Vec<_> = naive.query_term(term).collect();
            let s: Vec<_> = succinct.query_term(term).collect();
            assert_eq!(
                n.len(),
                s.len(),
                "posting count mismatch for term {term:x?}"
            );
            for ((n_id, n_s), (s_id, s_s)) in n.iter().zip(s.iter()) {
                assert_eq!(n_id, s_id);
                assert!(
                    (n_s - s_s).abs() < 1e-6,
                    "score mismatch for {n_id:?}: naive={n_s} succinct={s_s}"
                );
            }
            assert_eq!(
                naive.doc_frequency(term),
                succinct.doc_frequency(term)
            );
        }

        // Missing term returns nothing.
        let missing = hash_tokens("banana")[0];
        assert!(succinct.query_term(&missing).next().is_none());
        assert_eq!(succinct.doc_frequency(&missing), 0);
    }

    #[test]
    fn succinct_bm25_empty_corpus() {
        use crate::bm25::BM25Builder;
        let naive = BM25Builder::new().build();
        let succinct = SuccinctBM25Index::from_naive(&naive).unwrap();
        assert_eq!(succinct.doc_count(), 0);
        assert_eq!(succinct.term_count(), 0);
        assert!(succinct.query_term(&[0u8; 32]).next().is_none());
    }

    fn build_succinct_sample() -> SuccinctBM25Index {
        use crate::bm25::BM25Builder;
        use crate::tokens::hash_tokens;
        use triblespace::core::id::Id;
        fn iid(byte: u8) -> Id {
            Id::new([byte; 16]).unwrap()
        }
        let mut b = BM25Builder::new().k1(1.4).b(0.72);
        b.insert(iid(1), hash_tokens("the quick brown fox"));
        b.insert(iid(2), hash_tokens("the lazy brown dog"));
        b.insert(iid(3), hash_tokens("quick silver fox jumps"));
        b.insert(iid(4), hash_tokens("completely unrelated filler content"));
        SuccinctBM25Index::from_naive(&b.build()).unwrap()
    }

    #[test]
    fn succinct_bm25_bytes_round_trip() {
        use crate::tokens::hash_tokens;
        let original = build_succinct_sample();
        let bytes = original.to_bytes();
        let reloaded = SuccinctBM25Index::try_from_bytes(&bytes).expect("valid blob");

        assert_eq!(reloaded.doc_count(), original.doc_count());
        assert_eq!(reloaded.term_count(), original.term_count());
        assert_eq!(reloaded.k1(), original.k1());
        assert_eq!(reloaded.b(), original.b());
        assert!((reloaded.avg_doc_len() - original.avg_doc_len()).abs() < 1e-6);

        // Every term's posting list must round-trip bit-identical.
        for word in ["the", "fox", "quick", "brown", "dog"] {
            let term = hash_tokens(word)[0];
            let a: Vec<_> = original.query_term(&term).collect();
            let b: Vec<_> = reloaded.query_term(&term).collect();
            assert_eq!(a.len(), b.len(), "term '{word}' count mismatch");
            for ((a_id, a_s), (b_id, b_s)) in a.iter().zip(b.iter()) {
                assert_eq!(a_id, b_id);
                assert!(
                    (a_s - b_s).abs() < 1e-6,
                    "term '{word}': score mismatch {a_s} vs {b_s}"
                );
            }
        }
    }

    #[test]
    fn succinct_bm25_empty_round_trip() {
        use crate::bm25::BM25Builder;
        let naive = BM25Builder::new().build();
        let idx = SuccinctBM25Index::from_naive(&naive).unwrap();
        let bytes = idx.to_bytes();
        let reloaded = SuccinctBM25Index::try_from_bytes(&bytes).expect("valid blob");
        assert_eq!(reloaded.doc_count(), 0);
        assert_eq!(reloaded.term_count(), 0);
    }

    #[test]
    fn succinct_bm25_rejects_short_header() {
        let err = SuccinctBM25Index::try_from_bytes(&[0u8; 10]).unwrap_err();
        assert_eq!(err, SuccinctLoadError::ShortHeader);
    }

    #[test]
    fn succinct_bm25_rejects_bad_magic() {
        let mut bytes = build_succinct_sample().to_bytes();
        bytes[0] = b'X';
        let err = SuccinctBM25Index::try_from_bytes(&bytes).unwrap_err();
        assert_eq!(err, SuccinctLoadError::BadMagic);
    }

    #[test]
    fn succinct_bm25_rejects_bad_version() {
        let mut bytes = build_succinct_sample().to_bytes();
        bytes[4] = 99;
        let err = SuccinctBM25Index::try_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, SuccinctLoadError::VersionMismatch(_)));
    }

    #[test]
    fn succinct_bm25_rejects_truncation() {
        let bytes = build_succinct_sample().to_bytes();
        let truncated = &bytes[..bytes.len() - 2];
        let err = SuccinctBM25Index::try_from_bytes(truncated).unwrap_err();
        assert!(matches!(err, SuccinctLoadError::TruncatedSection(_)));
    }

    #[test]
    fn succinct_bm25_blob_schema_round_trip() {
        use triblespace::core::blob::{ToBlob, TryFromBlob};
        let original = build_succinct_sample();
        let blob: triblespace::core::blob::Blob<SuccinctBM25Blob> =
            (&original).to_blob();
        let reloaded = SuccinctBM25Index::try_from_blob(blob).expect("valid blob");
        assert_eq!(reloaded.doc_count(), original.doc_count());
        assert_eq!(reloaded.term_count(), original.term_count());
    }

    #[test]
    fn succinct_bm25_blob_is_deterministic() {
        // Content-addressing guarantee: same corpus must produce
        // identical bytes across runs.
        let a = build_succinct_sample().to_bytes();
        let b = build_succinct_sample().to_bytes();
        assert_eq!(a, b);
    }

    #[test]
    fn succinct_blob_smaller_than_naive_at_scale() {
        // Same corpus through naive and succinct paths — succinct
        // should fit in fewer bytes once postings bit-pack.
        use crate::bm25::BM25Builder;
        use crate::tokens::hash_tokens;
        use triblespace::core::id::Id;

        // Build a corpus large enough that the bit-packing wins
        // dominate the per-blob fixed overhead (~212B header +
        // metas).
        let mut b = BM25Builder::new();
        for i in 1..=250u16 {
            let text = format!(
                "doc {i} contains the quick brown fox {}",
                i % 17
            );
            let id = Id::new([
                (i >> 8) as u8,
                (i & 0xff) as u8,
                0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
                if i == 0 { 1 } else { 0xaa },
            ])
            .unwrap();
            b.insert(id, hash_tokens(&text));
        }
        let naive = b.build();
        let succinct = SuccinctBM25Index::from_naive(&naive).unwrap();

        let naive_bytes = naive.to_bytes();
        let succinct_bytes = succinct.to_bytes();
        // The target is a real savings; at this scale we expect
        // the succinct blob to be strictly smaller.
        assert!(
            succinct_bytes.len() < naive_bytes.len(),
            "succinct {} should be < naive {}",
            succinct_bytes.len(),
            naive_bytes.len(),
        );
    }

    #[test]
    fn postings_scale_saves_space_vs_naive() {
        // 1000 docs × 3 postings each over 500 terms; with
        // log2(1001)=10-bit doc_idx + 10-bit offsets, the jerky
        // idx_bytes should be clearly smaller than a u32 doc_idx
        // + u32 offset stored naively (10 vs 32 bits).
        let mut lists = Vec::new();
        for t in 0..500 {
            let mut l = Vec::new();
            for j in 0..3 {
                l.push(((t * 3 + j) as u32 % 1000, 1.0 + j as f32));
            }
            lists.push(l);
        }
        let total: usize = lists.iter().map(|l| l.len()).sum();
        let (idx_bytes, score_bytes, _meta) =
            SuccinctPostings::build(&lists, 1000).unwrap();
        // Naive doc_idx alone would be 4 bytes × total_postings.
        // idx_bytes holds doc_idx + offsets.
        let naive_doc_idx = total * 4;
        let naive_offsets = (lists.len() + 1) * 4;
        assert!(
            idx_bytes.len() < naive_doc_idx + naive_offsets,
            "succinct idx {} < naive idx+offsets {}",
            idx_bytes.len(),
            naive_doc_idx + naive_offsets
        );
        // Scores are still f32, no win expected on that side.
        assert_eq!(score_bytes.len(), total * 4);
    }
}
