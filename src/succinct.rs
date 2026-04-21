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
use triblespace::core::id::Id;
use triblespace::core::value::RawValue;

use crate::bm25::BM25Index;

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
