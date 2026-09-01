//! Canonical row-local NVFP4 cosine collection.
//!
//! The persisted value is a set of independently quantized embedding rows,
//! keyed and ordered by the exact embedding blob handle.  A row owns its FP32
//! global scale; adding another row can therefore never requantize an existing
//! one.  That independence is what makes sorted set union a canonical,
//! associative, commutative, and idempotent collection join.
//!
//! Each row stores a primary NVFP4 reconstruction and a second NVFP4
//! reconstruction of its residual. A member is a gapless structure-of-arrays:
//! `handles[N] | q0_globals[N] | q0_e4m3_scales[N][ceil256(D)/16] |
//! q0_e2m1_codes[N][ceil256(D)/2] | q1_globals[N] |
//! q1_e4m3_scales[N][ceil256(D)/16] | q1_e2m1_codes[N][ceil256(D)/2] |
//! norm_f32[N] | error_f32[N] | N_u64 | D_u64`.
//! Integers and floats are little-endian; handles are strictly ascending;
//! negative FP4 zero is rejected. `norm_f32` is an upward-rounded norm of the
//! summed reconstruction. `error_f32` is one final upward-rounded
//! transform-and-two-stage-quantization L2 certificate.
//!
//! Approximation is confined to candidate discovery.  [`NvFp4CosineIndex`]
//! uses conservative error bounds and fetches original embedding blobs for
//! exact reranking, so [`NvFp4CosineIndex::top_k`] and
//! [`NvFp4CosineIndex::above`] retain exact cosine semantics.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap};
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use anybytes::{Bytes, View};
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, BlobEncoding, TryFromBlob};
use triblespace_core::collection::records::{mapping_algorithm, KIND_COLLECTION_MAPPING};
use triblespace_core::collection::{
    CollectionEncoding, CollectionMapping, CollectionOperationError, Cover, TryFromCover,
    TryFromCoverError,
};
use triblespace_core::id::{id_hex, ExclusiveId, Id};
use triblespace_core::inline::encodings::genid::GenId;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::encodings::iu256::U256BE;
use triblespace_core::inline::{Inline, IntoInline, TryFromInline};
use triblespace_core::macros::{attributes, entity};
use triblespace_core::metadata::{self, MetaDescribe};
use triblespace_core::repo::{BlobStoreGet, BlobStoreMeta};
use triblespace_core::trible::{Fragment, TribleSet, TRIBLE_LEN};

const HANDLE_LEN: usize = 32;
const FLOAT_LEN: usize = 4;
const FOOTER_LEN: usize = 16;
const QUANT_BLOCK: usize = 16;
const ROTATION_BLOCK: usize = 256;
const QUANT_STAGES: usize = 2;
const FP4_MAX: f64 = 6.0;
const FP8_MAX: f64 = 448.0;

// Stable marker for this exact byte and cosine recipe. Minted with
// `trible genid` on 2026-09-01. It is embedded in the derived encoding's
// identity together with E, so a recipe or exact embedding encoding change
// necessarily produces another collection encoding.
pub const NVFP4_COSINE_SET: Id = id_hex!("18F7786A9F916ADD0E06DF15FA818A2F");

// Stable identity for the SimpleArchive attribute-selection mapping. Minted
// with `trible genid` on 2026-09-01. The selected attribute, exact blob
// encoding, and dimension remain concrete mapping-instance parameters.
pub const EMBEDDING_ATTRIBUTE_TO_NVFP4: Id = id_hex!("E8732C5918436416D071C0BAEF4F883F");

attributes! {
    /// Logical embedding dimension selected by one concrete NVFP4 mapping.
    ///
    /// Anchor minted with `trible genid` on 2026-09-01:
    /// `96ED6826E7FE88F1906D8C634A187C93`.
    /// Existing `metadata::attribute` and `metadata::blob_encoding` carry the
    /// other two parameters; this is the sole new mapping-field vocabulary.
    "96ED6826E7FE88F1906D8C634A187C93" as nvfp4_dimension: U256BE;
}

/// Failure to decode, construct, or query a canonical NVFP4 cosine set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvFp4Error {
    message: String,
}

impl NvFp4Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for NvFp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NvFp4Error {}

/// Canonical row-local NVFP4 carrier for exact embedding encoding `E`.
pub struct NvFp4CosineSet<E: BlobEncoding>(PhantomData<E>);

struct NvFp4CosineRecipe;

impl MetaDescribe for NvFp4CosineRecipe {
    fn describe() -> Fragment {
        let id = NVFP4_COSINE_SET;
        entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "nvfp4-cosine-recipe",
            metadata::description: "Canonical row-local two-stage residual NVFP4 cosine carrier. Rows are ordered by exact embedding handle and independently normalized, deterministically rotated, block-scaled, quantized twice, and conservatively error-bounded. Join is set union by handle; exact source embeddings remain lazy reranking dependencies.",
            metadata::tag: metadata::KIND_TAG,
        }
    }
}

impl<E> MetaDescribe for NvFp4CosineSet<E>
where
    E: BlobEncoding,
{
    fn describe() -> Fragment {
        let mut description = entity! {
            metadata::tag: metadata::KIND_BLOB_ENCODING,
            metadata::tag*: <NvFp4CosineRecipe as MetaDescribe>::describe(),
            metadata::blob_encoding*: E::describe(),
        };
        let id = description.root().expect("rooted NVFP4 encoding");
        description += entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "nvfp4-cosine-set",
            metadata::description: "Typed canonical set of independently two-stage residual-NVFP4-quantized embedding rows. The exact embedding blob encoding participates in this encoding's intrinsic identity.",
        };
        description
    }
}

impl<E> BlobEncoding for NvFp4CosineSet<E> where E: BlobEncoding {}

/// One exact similarity result.
#[derive(Debug)]
pub struct SimilarityHit<E: BlobEncoding> {
    /// Exact source embedding blob.
    pub embedding: Inline<Handle<E>>,
    /// Exact deterministic cosine score accumulated in `f64`.
    pub score: f64,
}

impl<E: BlobEncoding> Copy for SimilarityHit<E> {}

impl<E: BlobEncoding> Clone for SimilarityHit<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: BlobEncoding> PartialEq for SimilarityHit<E> {
    fn eq(&self, other: &Self) -> bool {
        self.embedding == other.embedding && self.score.to_bits() == other.score.to_bits()
    }
}

#[derive(Clone, Debug)]
struct StageLayout {
    globals: std::ops::Range<usize>,
    block_scales: std::ops::Range<usize>,
    codes: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
struct Layout {
    rows: usize,
    dimension: usize,
    blocks_per_row: usize,
    codes_per_row: usize,
    stages: [StageLayout; QUANT_STAGES],
    norms: std::ops::Range<usize>,
    errors: std::ops::Range<usize>,
}

impl Layout {
    fn parse(bytes: &[u8]) -> Result<Self, NvFp4Error> {
        if bytes.len() < FOOTER_LEN {
            return Err(NvFp4Error::new("NVFP4 member is shorter than its footer"));
        }
        let footer = bytes.len() - FOOTER_LEN;
        let rows = read_u64(&bytes[footer..footer + 8], "row count")?;
        let dimension = read_u64(&bytes[footer + 8..], "dimension")?;
        let rows =
            usize::try_from(rows).map_err(|_| NvFp4Error::new("NVFP4 row count exceeds usize"))?;
        let dimension = usize::try_from(dimension)
            .map_err(|_| NvFp4Error::new("NVFP4 dimension exceeds usize"))?;
        if dimension == 0 {
            return Err(NvFp4Error::new("NVFP4 dimension must be positive"));
        }
        let physical_dimension = dimension
            .checked_add(ROTATION_BLOCK - 1)
            .map(|value| value / ROTATION_BLOCK * ROTATION_BLOCK)
            .ok_or_else(|| NvFp4Error::new("NVFP4 padded dimension overflows usize"))?;
        let blocks_per_row = physical_dimension / QUANT_BLOCK;
        let codes_per_row = physical_dimension / 2;

        let handles_end = rows
            .checked_mul(HANDLE_LEN)
            .ok_or_else(|| NvFp4Error::new("NVFP4 handle plane overflows usize"))?;
        let global_len = rows
            .checked_mul(FLOAT_LEN)
            .ok_or_else(|| NvFp4Error::new("NVFP4 global-scale plane overflows usize"))?;
        let scales_len = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| NvFp4Error::new("NVFP4 block-scale plane overflows usize"))?;
        let codes_len = rows
            .checked_mul(codes_per_row)
            .ok_or_else(|| NvFp4Error::new("NVFP4 code plane overflows usize"))?;
        let float_plane_len = rows
            .checked_mul(FLOAT_LEN)
            .ok_or_else(|| NvFp4Error::new("NVFP4 float plane overflows usize"))?;
        let mut cursor = handles_end;
        let mut next_stage = || -> Result<StageLayout, NvFp4Error> {
            let globals = take_plane(&mut cursor, global_len, "global-scale")?;
            let block_scales = take_plane(&mut cursor, scales_len, "block-scale")?;
            let codes = take_plane(&mut cursor, codes_len, "code")?;
            Ok(StageLayout {
                globals,
                block_scales,
                codes,
            })
        };
        let stages = [next_stage()?, next_stage()?];
        let norms = take_plane(&mut cursor, float_plane_len, "norm")?;
        let errors = take_plane(&mut cursor, float_plane_len, "error")?;
        if cursor != footer {
            return Err(NvFp4Error::new(format!(
                "NVFP4 member length {} does not match N={rows}, D={dimension}",
                bytes.len()
            )));
        }

        let layout = Self {
            rows,
            dimension,
            blocks_per_row,
            codes_per_row,
            stages,
            norms,
            errors,
        };
        layout.validate(bytes)?;
        Ok(layout)
    }

    fn validate(&self, bytes: &[u8]) -> Result<(), NvFp4Error> {
        let mut previous: Option<&[u8]> = None;
        for row in 0..self.rows {
            let handle = self.handle(bytes, row);
            if previous.is_some_and(|old| old >= handle) {
                return Err(NvFp4Error::new(
                    "NVFP4 embedding handles must be strictly increasing",
                ));
            }
            previous = Some(handle);
            for stage in 0..QUANT_STAGES {
                validate_nonnegative_f32(self.global(bytes, row, stage), "global scale")?;
                if self
                    .block_scales(bytes, row, stage)
                    .iter()
                    .any(|&scale| scale > 0x7e)
                {
                    return Err(NvFp4Error::new(
                        "NVFP4 block scale is not a finite nonnegative E4M3 value",
                    ));
                }
                if self.codes(bytes, row, stage).iter().any(|&pair| {
                    let low = pair & 0x0f;
                    let high = pair >> 4;
                    low == 0x08 || high == 0x08
                }) {
                    return Err(NvFp4Error::new(
                        "NVFP4 code plane contains noncanonical negative zero",
                    ));
                }
            }
            validate_nonnegative_f32(self.norm(bytes, row), "reconstruction norm")?;
            validate_nonnegative_f32(self.error(bytes, row), "error bound")?;
        }
        Ok(())
    }

    fn handle<'a>(&self, bytes: &'a [u8], row: usize) -> &'a [u8] {
        &bytes[row * HANDLE_LEN..(row + 1) * HANDLE_LEN]
    }

    fn global(&self, bytes: &[u8], row: usize, stage: usize) -> f32 {
        let offset = self.stages[stage].globals.start + row * FLOAT_LEN;
        read_f32(&bytes[offset..offset + FLOAT_LEN])
    }

    fn block_scales<'a>(&self, bytes: &'a [u8], row: usize, stage: usize) -> &'a [u8] {
        let start = self.stages[stage].block_scales.start + row * self.blocks_per_row;
        &bytes[start..start + self.blocks_per_row]
    }

    fn codes<'a>(&self, bytes: &'a [u8], row: usize, stage: usize) -> &'a [u8] {
        let start = self.stages[stage].codes.start + row * self.codes_per_row;
        &bytes[start..start + self.codes_per_row]
    }

    fn norm(&self, bytes: &[u8], row: usize) -> f32 {
        read_f32(&bytes[self.norms.start + row * FLOAT_LEN..][..FLOAT_LEN])
    }

    fn error(&self, bytes: &[u8], row: usize) -> f32 {
        read_f32(&bytes[self.errors.start + row * FLOAT_LEN..][..FLOAT_LEN])
    }
}

fn take_plane(
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> Result<std::ops::Range<usize>, NvFp4Error> {
    let start = *cursor;
    let end = start
        .checked_add(len)
        .ok_or_else(|| NvFp4Error::new(format!("NVFP4 {field} offset overflows usize")))?;
    *cursor = end;
    Ok(start..end)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QuantizedStage {
    global: [u8; FLOAT_LEN],
    block_scales: Vec<u8>,
    codes: Vec<u8>,
}

impl QuantizedStage {
    fn quantize(values: &[f64]) -> Result<Self, NvFp4Error> {
        debug_assert_eq!(values.len() % ROTATION_BLOCK, 0);
        let maximum = values.iter().fold(
            0.0f64,
            |old, &value| {
                if value.abs() > old {
                    value.abs()
                } else {
                    old
                }
            },
        );
        if maximum == 0.0 {
            return Ok(Self {
                global: 0.0f32.to_le_bytes(),
                block_scales: vec![0; values.len() / QUANT_BLOCK],
                codes: vec![0; values.len() / 2],
            });
        }

        let global = (maximum / (FP4_MAX * FP8_MAX)) as f32;
        if !global.is_finite() || global <= 0.0 {
            return Err(NvFp4Error::new(
                "embedding produced an invalid NVFP4 global scale",
            ));
        }
        let global64 = f64::from(global);
        let mut block_scales = Vec::with_capacity(values.len() / QUANT_BLOCK);
        let mut codes = Vec::with_capacity(values.len() / 2);

        for block in values.chunks_exact(QUANT_BLOCK) {
            let block_maximum =
                block.iter().fold(
                    0.0f64,
                    |old, &value| {
                        if value.abs() > old {
                            value.abs()
                        } else {
                            old
                        }
                    },
                );
            let scale = if block_maximum == 0.0 {
                0
            } else {
                encode_e4m3(block_maximum / (FP4_MAX * global64))
            };
            block_scales.push(scale);
            let reconstructed_scale = global64 * decode_e4m3(scale);

            for pair in block.chunks_exact(2) {
                let low = if reconstructed_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(pair[0] / reconstructed_scale)
                };
                let high = if reconstructed_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(pair[1] / reconstructed_scale)
                };
                codes.push(low | (high << 4));
            }
        }

        Ok(Self {
            global: global.to_le_bytes(),
            block_scales,
            codes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct EncodedRow {
    handle: [u8; HANDLE_LEN],
    stages: [QuantizedStage; QUANT_STAGES],
    norm: [u8; FLOAT_LEN],
    error: [u8; FLOAT_LEN],
}

impl EncodedRow {
    fn quantize(
        handle: [u8; HANDLE_LEN],
        embedding: &[f32],
        dimension: usize,
    ) -> Result<Self, NvFp4Error> {
        let normalized = normalized_embedding(embedding, dimension)?;
        let transformed = rotate_normalized(&normalized)?;
        let primary = QuantizedStage::quantize(&transformed)?;
        let primary_decoded = decode_stage(&primary);
        let residual: Vec<_> = transformed
            .iter()
            .zip(&primary_decoded)
            .map(|(&exact, &approximate)| exact - approximate)
            .collect();
        let correction = QuantizedStage::quantize(&residual)?;
        let correction_decoded = decode_stage(&correction);
        let reconstruction: Vec<_> = primary_decoded
            .iter()
            .zip(&correction_decoded)
            .map(|(&primary, &correction)| primary + correction)
            .collect();
        let quantization_residual = outward_l2(&transformed, &reconstruction)?;
        // `scaled_hadamard` has eight rounded butterfly stages. Its final
        // multiplication by 1/16 is an exact power-of-two scaling for the
        // finite normal values emitted by this encoding. Componentwise gamma₈
        // error and ||x||₁ <= 16||x||₂ therefore give this conservative
        // forward-transform allowance. Adding it to the measured rotated-space
        // quantization residual lets queries center their certificate on the
        // streamed decoded dot without assuming that the computed FWHT is
        // exactly orthogonal.
        let transform_allowance = transform_allowance(&normalized)?;
        let error = upward_f32(add_up_nonnegative(
            quantization_residual,
            transform_allowance,
        ))?;
        let norm = upward_f32(outward_norm(&reconstruction)?)?;
        Ok(Self {
            handle,
            stages: [primary, correction],
            norm: norm.to_le_bytes(),
            error: error.to_le_bytes(),
        })
    }
}

fn normalized_embedding(embedding: &[f32], dimension: usize) -> Result<Vec<f64>, NvFp4Error> {
    if embedding.len() != dimension {
        return Err(NvFp4Error::new(format!(
            "embedding has dimension {}, expected {dimension}",
            embedding.len(),
        )));
    }
    let mut norm_squared = 0.0f64;
    for &value in embedding {
        if !value.is_finite() {
            return Err(NvFp4Error::new("embedding coordinates must all be finite"));
        }
        let value = f64::from(value);
        norm_squared += value * value;
    }
    let norm = norm_squared.sqrt();
    if norm == 0.0 {
        return Ok(vec![0.0; dimension]);
    }
    Ok(embedding
        .iter()
        .map(|&value| f64::from(value) / norm)
        .collect())
}

fn rotate_normalized(normalized: &[f64]) -> Result<Vec<f64>, NvFp4Error> {
    let physical_dimension = normalized
        .len()
        .checked_add(ROTATION_BLOCK - 1)
        .map(|value| value / ROTATION_BLOCK * ROTATION_BLOCK)
        .ok_or_else(|| NvFp4Error::new("embedding padded dimension overflows usize"))?;
    let mut transformed = vec![0.0; physical_dimension];
    for (index, (&source, target)) in normalized.iter().zip(&mut transformed).enumerate() {
        *target = source * rotation_sign(index);
    }
    for block in transformed.chunks_exact_mut(ROTATION_BLOCK) {
        scaled_hadamard(block);
    }
    Ok(transformed)
}

fn scaled_hadamard(block: &mut [f64]) {
    debug_assert_eq!(block.len(), ROTATION_BLOCK);
    let mut width = 1;
    while width < ROTATION_BLOCK {
        for start in (0..ROTATION_BLOCK).step_by(width * 2) {
            for offset in 0..width {
                let low = block[start + offset];
                let high = block[start + width + offset];
                block[start + offset] = low + high;
                block[start + width + offset] = low - high;
            }
        }
        width *= 2;
    }
    for value in block {
        *value *= 1.0 / 16.0;
    }
}

fn rotation_sign(index: usize) -> f64 {
    if splitmix64(index as u64 ^ 0x6B3F_7B4C_DA9E_0673) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn transform_allowance(normalized: &[f64]) -> Result<f64, NvFp4Error> {
    Ok(multiply_up_nonnegative(
        multiply_up_nonnegative(16.0, roundoff_gamma(8)),
        outward_norm(normalized)?,
    ))
}

fn outward_l2(left: &[f64], right: &[f64]) -> Result<f64, NvFp4Error> {
    if left.len() != right.len() {
        return Err(NvFp4Error::new("L2 dimensions differ"));
    }
    let mut squared = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        let raw_difference = (left - right).abs();
        if raw_difference == 0.0 {
            continue;
        }
        let difference = next_up_f64(raw_difference);
        let term = next_up_f64(difference * difference);
        squared = next_up_f64(squared + term);
    }
    Ok(next_up_f64(squared.sqrt()))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn encode_e4m3(value: f64) -> u8 {
    let value = value.clamp(0.0, FP8_MAX);
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for raw in 0..=0x7e {
        let candidate = decode_e4m3(raw);
        let distance = (candidate - value).abs();
        if distance < best_distance
            || (distance == best_distance
                && ((raw & 1 == 0 && best & 1 != 0) || (raw & 1 == best & 1 && raw < best)))
        {
            best = raw;
            best_distance = distance;
        }
    }
    best
}

fn decode_e4m3(raw: u8) -> f64 {
    let exponent = (raw >> 3) & 0x0f;
    let mantissa = raw & 0x07;
    let (significand, power) = if exponent == 0 {
        (mantissa, -9)
    } else {
        (8 + mantissa, i32::from(exponent) - 10)
    };
    // Every finite nonnegative E4M3 value is a small integer times an exact
    // binary power. Constructing that power directly avoids a runtime `powi`
    // call in every scanned block while producing the identical f64 bits.
    let scale = f64::from_bits(((power + 1023) as u64) << (f64::MANTISSA_DIGITS - 1));
    f64::from(significand) * scale
}

fn encode_e2m1(value: f64) -> u8 {
    const POSITIVE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let negative = value.is_sign_negative();
    let magnitude = value.abs().min(FP4_MAX);
    let mut best = 0usize;
    let mut best_distance = f64::INFINITY;
    for (raw, candidate) in POSITIVE.into_iter().enumerate() {
        let distance = (candidate - magnitude).abs();
        if distance < best_distance
            || (distance == best_distance
                && ((raw & 1 == 0 && best & 1 != 0) || (raw & 1 == best & 1 && raw < best)))
        {
            best = raw;
            best_distance = distance;
        }
    }
    if best == 0 {
        0
    } else if negative {
        best as u8 | 0x08
    } else {
        best as u8
    }
}

const fn decode_e2m1(raw: u8) -> f64 {
    const POSITIVE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = POSITIVE[(raw & 0x07) as usize];
    if raw & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

const fn decoded_e2m1_pairs() -> [[f64; 2]; 256] {
    let mut pairs = [[0.0; 2]; 256];
    let mut raw = 0usize;
    while raw < pairs.len() {
        let packed = raw as u8;
        pairs[raw] = [decode_e2m1(packed & 0x0f), decode_e2m1(packed >> 4)];
        raw += 1;
    }
    pairs
}

const DECODED_E2M1_PAIRS: [[f64; 2]; 256] = decoded_e2m1_pairs();

fn upward_f32(value: f64) -> Result<f32, NvFp4Error> {
    let mut rounded = value as f32;
    if !rounded.is_finite() {
        return Err(NvFp4Error::new("NVFP4 error bound exceeds f32"));
    }
    if f64::from(rounded) < value {
        rounded = f32::from_bits(rounded.to_bits() + 1);
    }
    if !rounded.is_finite() {
        return Err(NvFp4Error::new("NVFP4 error bound exceeds f32"));
    }
    Ok(rounded)
}

fn encode_rows<E: BlobEncoding>(
    dimension: usize,
    mut rows: Vec<EncodedRow>,
) -> Result<Blob<NvFp4CosineSet<E>>, NvFp4Error> {
    if dimension == 0 {
        return Err(NvFp4Error::new("NVFP4 dimension must be positive"));
    }
    rows.sort_unstable_by_key(|row| row.handle);
    let physical_dimension = dimension
        .checked_add(ROTATION_BLOCK - 1)
        .map(|value| value / ROTATION_BLOCK * ROTATION_BLOCK)
        .ok_or_else(|| NvFp4Error::new("NVFP4 padded dimension overflows usize"))?;
    let blocks_per_row = physical_dimension / QUANT_BLOCK;
    let codes_per_row = physical_dimension / 2;
    let mut distinct: Vec<EncodedRow> = Vec::with_capacity(rows.len());
    for row in rows {
        if row.stages.iter().any(|stage| {
            stage.block_scales.len() != blocks_per_row || stage.codes.len() != codes_per_row
        }) {
            return Err(NvFp4Error::new(
                "NVFP4 row payload does not match its dimension",
            ));
        }
        if let Some(previous) = distinct.last() {
            if previous.handle == row.handle {
                if previous != &row {
                    return Err(NvFp4Error::new(
                        "one embedding handle has two different NVFP4 rows",
                    ));
                }
                continue;
            }
        }
        distinct.push(row);
    }

    let stage_width = FLOAT_LEN
        .checked_add(blocks_per_row)
        .and_then(|value| value.checked_add(codes_per_row))
        .ok_or_else(|| NvFp4Error::new("NVFP4 stage width overflows usize"))?;
    let row_width = stage_width
        .checked_mul(QUANT_STAGES)
        .and_then(|value| value.checked_add(HANDLE_LEN))
        .and_then(|value| value.checked_add(FLOAT_LEN))
        .and_then(|value| value.checked_add(FLOAT_LEN))
        .ok_or_else(|| NvFp4Error::new("NVFP4 row width overflows usize"))?;
    let capacity = distinct
        .len()
        .checked_mul(row_width)
        .and_then(|value| value.checked_add(FOOTER_LEN))
        .ok_or_else(|| NvFp4Error::new("NVFP4 member length overflows usize"))?;
    let mut bytes = Vec::with_capacity(capacity);
    for row in &distinct {
        bytes.extend_from_slice(&row.handle);
    }
    for stage in 0..QUANT_STAGES {
        for row in &distinct {
            bytes.extend_from_slice(&row.stages[stage].global);
        }
        for row in &distinct {
            bytes.extend_from_slice(&row.stages[stage].block_scales);
        }
        for row in &distinct {
            bytes.extend_from_slice(&row.stages[stage].codes);
        }
    }
    for row in &distinct {
        bytes.extend_from_slice(&row.norm);
    }
    for row in &distinct {
        bytes.extend_from_slice(&row.error);
    }
    bytes.extend_from_slice(
        &u64::try_from(distinct.len())
            .map_err(|_| NvFp4Error::new("NVFP4 row count exceeds u64"))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(
        &u64::try_from(dimension)
            .map_err(|_| NvFp4Error::new("NVFP4 dimension exceeds u64"))?
            .to_le_bytes(),
    );
    debug_assert_eq!(bytes.len(), capacity);
    let blob = Blob::new(Bytes::from_source(bytes));
    Layout::parse(blob.bytes.as_ref())?;
    Ok(blob)
}

fn owned_row(bytes: &[u8], layout: &Layout, row: usize) -> EncodedRow {
    EncodedRow {
        handle: layout
            .handle(bytes, row)
            .try_into()
            .expect("32-byte handle"),
        stages: std::array::from_fn(|stage| QuantizedStage {
            global: bytes[layout.stages[stage].globals.start + row * FLOAT_LEN..][..FLOAT_LEN]
                .try_into()
                .expect("four-byte global scale"),
            block_scales: layout.block_scales(bytes, row, stage).to_vec(),
            codes: layout.codes(bytes, row, stage).to_vec(),
        }),
        norm: bytes[layout.norms.start + row * FLOAT_LEN..][..FLOAT_LEN]
            .try_into()
            .expect("four-byte reconstruction norm"),
        error: bytes[layout.errors.start + row * FLOAT_LEN..][..FLOAT_LEN]
            .try_into()
            .expect("four-byte error bound"),
    }
}

fn rows_equal(
    left_bytes: &[u8],
    left_layout: &Layout,
    left_row: usize,
    right_bytes: &[u8],
    right_layout: &Layout,
    right_row: usize,
) -> bool {
    let stages_equal = (0..QUANT_STAGES).all(|stage| {
        let left_global = left_layout.stages[stage].globals.start + left_row * FLOAT_LEN;
        let right_global = right_layout.stages[stage].globals.start + right_row * FLOAT_LEN;
        left_bytes[left_global..left_global + FLOAT_LEN]
            == right_bytes[right_global..right_global + FLOAT_LEN]
            && left_layout.block_scales(left_bytes, left_row, stage)
                == right_layout.block_scales(right_bytes, right_row, stage)
            && left_layout.codes(left_bytes, left_row, stage)
                == right_layout.codes(right_bytes, right_row, stage)
    });
    let left_norm = left_layout.norms.start + left_row * FLOAT_LEN;
    let right_norm = right_layout.norms.start + right_row * FLOAT_LEN;
    let left_error = left_layout.errors.start + left_row * FLOAT_LEN;
    let right_error = right_layout.errors.start + right_row * FLOAT_LEN;
    left_layout.handle(left_bytes, left_row) == right_layout.handle(right_bytes, right_row)
        && stages_equal
        && left_bytes[left_norm..left_norm + FLOAT_LEN]
            == right_bytes[right_norm..right_norm + FLOAT_LEN]
        && left_bytes[left_error..left_error + FLOAT_LEN]
            == right_bytes[right_error..right_error + FLOAT_LEN]
}

fn join_members<E: BlobEncoding>(
    low: &Blob<NvFp4CosineSet<E>>,
    high: &Blob<NvFp4CosineSet<E>>,
    dimension: usize,
) -> Result<Blob<NvFp4CosineSet<E>>, NvFp4Error> {
    let low_layout = Layout::parse(low.bytes.as_ref())?;
    let high_layout = Layout::parse(high.bytes.as_ref())?;
    if low_layout.dimension != dimension || high_layout.dimension != dimension {
        return Err(NvFp4Error::new(format!(
            "NVFP4 join member dimension does not match descriptor {dimension}"
        )));
    }
    let mut rows = Vec::with_capacity(low_layout.rows + high_layout.rows);
    let mut low_row = 0;
    let mut high_row = 0;
    while low_row < low_layout.rows && high_row < high_layout.rows {
        match low_layout
            .handle(low.bytes.as_ref(), low_row)
            .cmp(high_layout.handle(high.bytes.as_ref(), high_row))
        {
            Ordering::Less => {
                rows.push(owned_row(low.bytes.as_ref(), &low_layout, low_row));
                low_row += 1;
            }
            Ordering::Greater => {
                rows.push(owned_row(high.bytes.as_ref(), &high_layout, high_row));
                high_row += 1;
            }
            Ordering::Equal => {
                let left = owned_row(low.bytes.as_ref(), &low_layout, low_row);
                let right = owned_row(high.bytes.as_ref(), &high_layout, high_row);
                if left != right {
                    return Err(NvFp4Error::new(
                        "one embedding handle has two different NVFP4 rows",
                    ));
                }
                rows.push(left);
                low_row += 1;
                high_row += 1;
            }
        }
    }
    while low_row < low_layout.rows {
        rows.push(owned_row(low.bytes.as_ref(), &low_layout, low_row));
        low_row += 1;
    }
    while high_row < high_layout.rows {
        rows.push(owned_row(high.bytes.as_ref(), &high_layout, high_row));
        high_row += 1;
    }
    encode_rows(dimension, rows)
}

#[derive(Clone, Debug)]
struct Member {
    bytes: Bytes,
    layout: Layout,
}

/// Lazy cover-aware query view over canonical NVFP4 members.
pub struct NvFp4CosineIndex<E: BlobEncoding> {
    members: Vec<Member>,
    dimension: usize,
    _encoding: PhantomData<E>,
}

/// Bound projection of one handle-valued SimpleArchive attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingAttributeToNvFp4<E: BlobEncoding> {
    attribute: Id,
    dimension: NonZeroUsize,
    _encoding: PhantomData<E>,
}

impl<E: BlobEncoding> EmbeddingAttributeToNvFp4<E> {
    /// Select `attribute` as typed `Handle<E>` values of exactly `dimension`.
    pub fn new(attribute: Id, dimension: usize) -> Result<Self, NvFp4Error> {
        let dimension = NonZeroUsize::new(dimension)
            .ok_or_else(|| NvFp4Error::new("embedding dimension must be positive"))?;
        u64::try_from(dimension.get())
            .map_err(|_| NvFp4Error::new("embedding dimension exceeds u64"))?;
        Ok(Self {
            attribute,
            dimension,
            _encoding: PhantomData,
        })
    }

    /// Selected source attribute.
    pub fn attribute(&self) -> Id {
        self.attribute
    }

    /// Exact logical embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension.get()
    }
}

struct EmbeddingAttributeToNvFp4Recipe;

impl MetaDescribe for EmbeddingAttributeToNvFp4Recipe {
    fn describe() -> Fragment {
        let id = EMBEDDING_ATTRIBUTE_TO_NVFP4;
        entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "embedding-attribute-to-nvfp4",
            metadata::description: "Canonical join-preserving projection from one selected Handle<E>-valued SimpleArchive attribute to NvFp4CosineSet<E>. Each distinct exact handle contributes one independently normalized, fixed-sign block-Hadamard-rotated, two-stage residual-NVFP4 row with an upward-rounded reconstruction norm and final L2 certificate.",
            metadata::tag: metadata::KIND_COLLECTION_MAPPING_ALGORITHM,
        }
    }
}

fn mapping_fragment<E: BlobEncoding>(attribute: Id, dimension: usize) -> Fragment {
    let attribute: Inline<GenId> = attribute.to_inline();
    entity! { _ @
        metadata::tag: KIND_COLLECTION_MAPPING,
        mapping_algorithm*: <EmbeddingAttributeToNvFp4Recipe as MetaDescribe>::describe(),
        metadata::attribute: attribute,
        metadata::blob_encoding*: E::describe(),
        nvfp4_dimension: dimension as u64,
    }
}

fn mapping_attribute(descriptor: &Fragment) -> Result<Id, CollectionOperationError> {
    let raw = triblespace_core::collection::descriptor::mapping_argument(
        descriptor.facts(),
        metadata::attribute.id(),
    )
    .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?
    .ok_or_else(|| {
        CollectionOperationError::Fatal("NVFP4 mapping is missing metadata::attribute".to_owned())
    })?;
    Inline::<GenId>::new(raw)
        .try_from_inline::<Id>()
        .map_err(|source| {
            CollectionOperationError::Fatal(format!(
                "NVFP4 mapping has an invalid metadata::attribute: {source:?}"
            ))
        })
}

fn mapping_embedding_encoding(descriptor: &Fragment) -> Result<Id, CollectionOperationError> {
    let raw = triblespace_core::collection::descriptor::mapping_argument(
        descriptor.facts(),
        metadata::blob_encoding.id(),
    )
    .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?
    .ok_or_else(|| {
        CollectionOperationError::Fatal(
            "NVFP4 mapping is missing metadata::blob_encoding".to_owned(),
        )
    })?;
    Inline::<GenId>::new(raw)
        .try_from_inline::<Id>()
        .map_err(|source| {
            CollectionOperationError::Fatal(format!(
                "NVFP4 mapping has an invalid metadata::blob_encoding: {source:?}"
            ))
        })
}

fn mapping_dimension(descriptor: &Fragment) -> Result<usize, CollectionOperationError> {
    mapping_dimension_facts(descriptor.facts())
}

fn mapping_dimension_facts(facts: &TribleSet) -> Result<usize, CollectionOperationError> {
    let raw =
        triblespace_core::collection::descriptor::mapping_argument(facts, nvfp4_dimension.id())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?
            .ok_or_else(|| {
                CollectionOperationError::Fatal(
                    "NVFP4 mapping is missing nvfp4_dimension".to_owned(),
                )
            })?;
    let dimension = u64::try_from_inline(&Inline::<U256BE>::new(raw)).map_err(|source| {
        CollectionOperationError::Fatal(format!(
            "NVFP4 mapping has an invalid dimension: {source:?}"
        ))
    })?;
    let dimension = usize::try_from(dimension).map_err(|_| {
        CollectionOperationError::Fatal("NVFP4 mapping dimension exceeds usize".to_owned())
    })?;
    if dimension == 0 {
        return Err(CollectionOperationError::Fatal(
            "NVFP4 mapping dimension must be positive".to_owned(),
        ));
    }
    Ok(dimension)
}

impl<E> CollectionMapping for EmbeddingAttributeToNvFp4<E>
where
    E: BlobEncoding,
    View<[f32]>: TryFromBlob<E>,
    <View<[f32]> as TryFromBlob<E>>::Error: fmt::Display + Send + Sync + 'static,
{
    type Source = SimpleArchive;
    type Target = NvFp4CosineSet<E>;

    fn fragment(&self) -> Fragment {
        mapping_fragment::<E>(self.attribute, self.dimension.get())
    }

    fn bind(_source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        let actual = triblespace_core::collection::descriptor::mapping_algorithm(target.facts())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?;
        if actual != Some(EMBEDDING_ATTRIBUTE_TO_NVFP4) {
            return Err(CollectionOperationError::Fatal(format!(
                "NVFP4 mapping algorithm {:?} does not match {EMBEDDING_ATTRIBUTE_TO_NVFP4:X}",
                actual.map(|id| format!("{id:X}")),
            )));
        }
        let actual_encoding = mapping_embedding_encoding(target)?;
        if actual_encoding != E::id() {
            return Err(CollectionOperationError::Fatal(format!(
                "NVFP4 mapping names embedding encoding {actual_encoding:X}, expected {:X}",
                E::id(),
            )));
        }
        let attribute = mapping_attribute(target)?;
        let dimension = mapping_dimension(target)?;
        Ok(Self {
            attribute,
            dimension: NonZeroUsize::new(dimension).expect("checked positive"),
            _encoding: PhantomData,
        })
    }

    fn map<R>(
        &self,
        source: &Blob<SimpleArchive>,
        reader: &R,
    ) -> Result<Blob<Self::Target>, CollectionOperationError>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        triblespace_core::collection::simplearchive_union::validate_element(source)
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?;

        let mut handles = BTreeSet::new();
        for raw in source.bytes.as_ref().chunks_exact(TRIBLE_LEN) {
            if raw[16..32] == self.attribute[..] {
                handles.insert(raw[32..64].try_into().expect("32-byte trible value"));
            }
        }

        let mut rows = Vec::with_capacity(handles.len());
        for raw in handles {
            let handle = Inline::<Handle<E>>::new(raw);
            let resident = reader
                .metadata(handle)
                .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?;
            if resident.is_none() {
                return Err(CollectionOperationError::MissingDependency(
                    Handle::<E>::to_hash(handle),
                ));
            }
            let blob: Blob<E> = reader
                .get(handle)
                .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?;
            let embedding = View::<[f32]>::try_from_blob(blob).map_err(|source| {
                CollectionOperationError::Fatal(format!(
                    "embedding {} cannot be decoded: {source}",
                    uppercase_hex(&raw),
                ))
            })?;
            rows.push(
                EncodedRow::quantize(raw, embedding.as_ref(), self.dimension.get())
                    .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?,
            );
        }
        encode_rows::<E>(self.dimension.get(), rows)
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))
    }
}

impl<E> CollectionEncoding for NvFp4CosineSet<E>
where
    E: BlobEncoding,
{
    fn validate_descriptor(descriptor: &Fragment) -> Result<(), CollectionOperationError> {
        mapping_dimension(descriptor).map(|_| ())
    }

    fn validate_member<R>(
        descriptor: &Fragment,
        member: &Blob<Self>,
        _reader: &R,
    ) -> Result<(), CollectionOperationError>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        let expected = mapping_dimension(descriptor)?;
        let layout = Layout::parse(member.bytes.as_ref())
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))?;
        if layout.dimension != expected {
            return Err(CollectionOperationError::Fatal(format!(
                "NVFP4 member dimension {} does not match descriptor {expected}",
                layout.dimension,
            )));
        }
        // Member admission validates the self-contained byte grammar only.
        // Replaying the deterministic mapping here would fetch and requantize
        // every exact source embedding, defeating both lazy reranking and
        // persisted derivation work. Locally mapped members are canonical by
        // construction. The network currently does not reuse unsigned remote
        // DERIVE equations; introducing that would require an independent
        // trust or recomputation boundary rather than stronger byte parsing.
        Ok(())
    }

    fn join_members<R>(
        descriptor: &Fragment,
        low: &Blob<Self>,
        high: &Blob<Self>,
        _reader: &R,
    ) -> Result<Blob<Self>, CollectionOperationError>
    where
        R: BlobStoreGet + BlobStoreMeta,
    {
        let expected = mapping_dimension(descriptor)?;
        join_members(low, high, expected)
            .map_err(|source| CollectionOperationError::Fatal(source.to_string()))
    }
}

impl<E: BlobEncoding> fmt::Debug for NvFp4CosineIndex<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NvFp4CosineIndex")
            .field("members", &self.members.len())
            .field("dimension", &self.dimension)
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    handle: [u8; HANDLE_LEN],
    upper: f64,
}

impl<E> NvFp4CosineIndex<E>
where
    E: BlobEncoding,
    View<[f32]>: TryFromBlob<E>,
    <View<[f32]> as TryFromBlob<E>>::Error: fmt::Display + Send + Sync + 'static,
{
    /// Logical embedding dimension shared by every member in the cover.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of physical cover segments retained by this lazy view.
    pub fn segment_count(&self) -> usize {
        self.members.len()
    }

    /// Exact top `k` cosine neighbours, ranked by score then handle.
    ///
    /// Candidate discovery scans the compact rows once. Original embeddings
    /// are fetched in descending certified-upper-bound order until the stored
    /// envelopes prove that no unseen row can enter the exact result.
    pub fn top_k<R>(
        &self,
        snapshot: &R,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SimilarityHit<E>>, NvFp4Error>
    where
        R: BlobStoreGet,
    {
        if k == 0 {
            return Ok(Vec::new());
        }
        let prepared = PreparedQuery::new(query, self.dimension)?;
        let mut candidates = self.candidates(&prepared)?;
        candidates.sort_unstable_by(|left, right| {
            right
                .upper
                .total_cmp(&left.upper)
                .then_with(|| left.handle.cmp(&right.handle))
        });
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let wanted = k.min(candidates.len());
        let mut ranked = Vec::with_capacity(wanted + 1);
        let mut checked = 0usize;
        let mut target = wanted;
        while checked < candidates.len() {
            let end = target.min(candidates.len());
            for candidate in &candidates[checked..end] {
                ranked.push(self.exact_hit(snapshot, &prepared.exact, candidate.handle)?);
            }
            sort_hits(&mut ranked);
            ranked.truncate(wanted);
            checked = end;

            let Some(unseen) = candidates.get(checked) else {
                break;
            };
            if ranked.len() == wanted && ranked[wanted - 1].score > unseen.upper {
                // Strict comparison preserves the secondary handle ordering
                // when an unseen exact score could tie the current boundary.
                break;
            }
            target = target.saturating_mul(2).max(checked.saturating_add(1));
        }
        Ok(ranked)
    }

    /// Every embedding whose exact cosine is at least `floor`.
    ///
    /// Only rows whose conservative upper bound can cross the threshold cause
    /// an exact blob fetch. Returned rows are ranked identically to `top_k`.
    pub fn above<R>(
        &self,
        snapshot: &R,
        query: &[f32],
        floor: f64,
    ) -> Result<Vec<SimilarityHit<E>>, NvFp4Error>
    where
        R: BlobStoreGet,
    {
        if floor.is_nan() {
            return Err(NvFp4Error::new("cosine floor must not be NaN"));
        }
        if floor > 1.0 {
            return Ok(Vec::new());
        }
        let prepared = PreparedQuery::new(query, self.dimension)?;
        let mut exact = Vec::new();
        for candidate in self.candidates(&prepared)? {
            if candidate.upper < floor {
                continue;
            }
            let hit = self.exact_hit(snapshot, &prepared.exact, candidate.handle)?;
            if hit.score >= floor {
                exact.push(hit);
            }
        }
        sort_hits(&mut exact);
        Ok(exact)
    }

    fn candidates(&self, query: &PreparedQuery) -> Result<Vec<Candidate>, NvFp4Error> {
        let mut candidates = Vec::new();
        let compact_gamma = dot_gamma(query.approximate.len());
        let division_gamma = roundoff_gamma(1);
        let exact_gamma = dot_gamma(self.dimension);
        let exact_query_norm_bound = add_up_nonnegative(query.approximate_norm, query.error);
        self.for_each_unique_row(|handle, member, row| {
            let bytes = member.bytes.as_ref();
            let reconstruction_norm = f64::from(member.layout.norm(bytes, row));
            let raw_row_error = f64::from(member.layout.error(bytes, row));
            // Let y be the canonical f64 sum of both decoded stages and n the
            // stored upward norm. The scan centers its cosine certificate on
            // c = y/n. Since ||y|| <= n,
            //   ||y - y/n|| = ||y|| |n - 1| / n <= |n - 1|.
            // Add that normalization displacement to the persisted ||x-y||
            // certificate, rounding the final sum outward.
            let normalization_displacement = if reconstruction_norm == 0.0 {
                0.0
            } else {
                absolute_difference_up(reconstruction_norm, 1.0)
            };
            let row_error = add_up_nonnegative(raw_row_error, normalization_displacement);
            let globals: [f64; QUANT_STAGES] =
                std::array::from_fn(|stage| f64::from(member.layout.global(bytes, row, stage)));
            let scales: [&[u8]; QUANT_STAGES] =
                std::array::from_fn(|stage| member.layout.block_scales(bytes, row, stage));
            let codes: [&[u8]; QUANT_STAGES] =
                std::array::from_fn(|stage| member.layout.codes(bytes, row, stage));
            let mut raw_dot_lanes = [0.0f64; QUANT_BLOCK / 2];
            let mut coordinate = 0;
            for block in 0..member.layout.blocks_per_row {
                let decoded_scales: [f64; QUANT_STAGES] =
                    std::array::from_fn(|stage| globals[stage] * decode_e4m3(scales[stage][block]));
                let code_start = block * (QUANT_BLOCK / 2);
                for (lane, code) in (code_start..code_start + QUANT_BLOCK / 2).enumerate() {
                    let primary = DECODED_E2M1_PAIRS[usize::from(codes[0][code])];
                    let correction = DECODED_E2M1_PAIRS[usize::from(codes[1][code])];
                    // These sums are the canonical reconstructed coordinates
                    // and exactly match construction-time decoding.
                    let low = primary[0] * decoded_scales[0] + correction[0] * decoded_scales[1];
                    let high = primary[1] * decoded_scales[0] + correction[1] * decoded_scales[1];
                    raw_dot_lanes[lane] += query.approximate[coordinate] * low;
                    raw_dot_lanes[lane] += query.approximate[coordinate + 1] * high;
                    coordinate += 2;
                }
            }
            debug_assert_eq!(coordinate, query.approximate.len());
            // Eight fixed lanes break the long scalar dependency chain. Each
            // product crosses a shallower reduction than the sequential
            // 2D-rounding model, so `compact_gamma` remains conservative.
            let mut raw_dot = raw_dot_lanes[0];
            for lane in &raw_dot_lanes[1..] {
                raw_dot += *lane;
            }

            let approximate = if reconstruction_norm == 0.0 {
                0.0
            } else {
                raw_dot / reconstruction_norm
            };
            if !approximate.is_finite() {
                return Err(NvFp4Error::new("NVFP4 approximate cosine is not finite"));
            }
            // Cauchy gives sum |q_i y_i| <= ||q|| ||y||. Dividing the
            // fixed-order dot by n >= ||y|| cancels the stored row norm, so
            // the dot-product allowance is gamma * ||q||. The division itself
            // contributes one separately certified rounding.
            let accumulation_error = multiply_up_nonnegative(compact_gamma, query.approximate_norm);
            let division_error = multiply_up_nonnegative(division_gamma, approximate.abs());
            let exact_row_norm_bound = add_up_nonnegative(1.0, row_error);
            // Exact reranking uses the same fixed-order normalized-coordinate
            // dot. Its possible upward accumulation error must be included too
            // when comparing that returned f64 score against this envelope.
            let exact_accumulation_error = multiply_up_nonnegative(
                multiply_up_nonnegative(exact_gamma, exact_query_norm_bound),
                exact_row_norm_bound,
            );
            let envelope = [
                multiply_up_nonnegative(query.error, exact_row_norm_bound),
                multiply_up_nonnegative(query.approximate_norm, row_error),
                accumulation_error,
                division_error,
                exact_accumulation_error,
            ]
            .into_iter()
            .fold(0.0, add_up_nonnegative);
            let upper = certified_cosine_upper(approximate, envelope);
            candidates.push(Candidate { handle, upper });
            Ok(())
        })?;
        Ok(candidates)
    }

    fn exact_hit<R>(
        &self,
        snapshot: &R,
        normalized_query: &[f64],
        raw: [u8; HANDLE_LEN],
    ) -> Result<SimilarityHit<E>, NvFp4Error>
    where
        R: BlobStoreGet,
    {
        let embedding = Inline::<Handle<E>>::new(raw);
        let blob: Blob<E> = snapshot.get(embedding).map_err(|source| {
            NvFp4Error::new(format!(
                "cannot fetch exact embedding {}: {source}",
                uppercase_hex(&raw),
            ))
        })?;
        let candidate = View::<[f32]>::try_from_blob(blob).map_err(|source| {
            NvFp4Error::new(format!(
                "cannot decode exact embedding {}: {source}",
                uppercase_hex(&raw),
            ))
        })?;
        if candidate.len() != self.dimension {
            return Err(NvFp4Error::new(format!(
                "exact embedding {} has dimension {}, expected {}",
                uppercase_hex(&raw),
                candidate.len(),
                self.dimension,
            )));
        }
        let score = exact_cosine_with_normalized_left(normalized_query, candidate.as_ref())?;
        Ok(SimilarityHit { embedding, score })
    }

    fn for_each_unique_row<F>(&self, mut visit: F) -> Result<(), NvFp4Error>
    where
        F: FnMut([u8; HANDLE_LEN], &Member, usize) -> Result<(), NvFp4Error>,
    {
        let mut heap = BinaryHeap::new();
        for (member, segment) in self.members.iter().enumerate() {
            if segment.layout.rows > 0 {
                let handle = segment
                    .layout
                    .handle(segment.bytes.as_ref(), 0)
                    .try_into()
                    .expect("32-byte handle");
                heap.push(Reverse((handle, member, 0usize)));
            }
        }

        let mut occurrences = Vec::new();
        while let Some(Reverse((handle, member, row))) = heap.pop() {
            occurrences.clear();
            occurrences.push((member, row));
            while heap
                .peek()
                .is_some_and(|Reverse((next, _, _))| next == &handle)
            {
                let Reverse((_, member, row)) = heap.pop().expect("peeked row");
                occurrences.push((member, row));
            }
            for &(other_member, other_row) in &occurrences[1..] {
                if !rows_equal(
                    self.members[member].bytes.as_ref(),
                    &self.members[member].layout,
                    row,
                    self.members[other_member].bytes.as_ref(),
                    &self.members[other_member].layout,
                    other_row,
                ) {
                    return Err(NvFp4Error::new(
                        "one embedding handle has conflicting rows across cover members",
                    ));
                }
            }
            visit(handle, &self.members[member], row)?;

            for &(member, row) in &occurrences {
                let next = row + 1;
                if next < self.members[member].layout.rows {
                    let next_handle = self.members[member]
                        .layout
                        .handle(self.members[member].bytes.as_ref(), next)
                        .try_into()
                        .expect("32-byte handle");
                    heap.push(Reverse((next_handle, member, next)));
                }
            }
        }
        Ok(())
    }
}

struct PreparedQuery {
    exact: Vec<f64>,
    approximate: Vec<f64>,
    approximate_norm: f64,
    error: f64,
}

impl PreparedQuery {
    fn new(query: &[f32], dimension: usize) -> Result<Self, NvFp4Error> {
        let exact = normalized_embedding(query, dimension)?;
        // The CPU path already scans reconstructed rows as f64, so quantizing
        // the query would add error without saving work. Keep the computed
        // rotated query exact and certify only its eight FWHT butterfly stages.
        let approximate = rotate_normalized(&exact)?;
        let approximate_norm = outward_norm(&approximate)?;
        let error = transform_allowance(&exact)?;
        Ok(Self {
            exact,
            approximate,
            approximate_norm,
            error,
        })
    }
}

fn outward_norm(values: &[f64]) -> Result<f64, NvFp4Error> {
    let mut squared = 0.0f64;
    for &value in values {
        if value == 0.0 {
            continue;
        }
        let magnitude = next_up_f64(value.abs());
        let term = next_up_f64(magnitude * magnitude);
        squared = next_up_f64(squared + term);
    }
    outward_sqrt(squared)
}

fn outward_sqrt(squared: f64) -> Result<f64, NvFp4Error> {
    if squared == 0.0 {
        return Ok(0.0);
    }
    let norm = next_up_f64(squared.sqrt());
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(NvFp4Error::new("NVFP4 reconstructed norm is not finite"))
    }
}

fn decode_stage(stage: &QuantizedStage) -> Vec<f64> {
    let global = f64::from(read_f32(&stage.global));
    let mut decoded = Vec::with_capacity(stage.codes.len() * 2);
    for (&scale, codes) in stage
        .block_scales
        .iter()
        .zip(stage.codes.chunks_exact(QUANT_BLOCK / 2))
    {
        let scale = global * decode_e4m3(scale);
        for &pair in codes {
            decoded.push(decode_e2m1(pair & 0x0f) * scale);
            decoded.push(decode_e2m1(pair >> 4) * scale);
        }
    }
    decoded
}

#[cfg(test)]
fn decode_encoded_row(row: &EncodedRow) -> Vec<f64> {
    let primary = decode_stage(&row.stages[0]);
    let correction = decode_stage(&row.stages[1]);
    primary
        .into_iter()
        .zip(correction)
        // This fixed-order f64 sum defines the canonical reconstruction. The
        // candidate scan reproduces the same sum, so its rounding is already
        // enclosed by the stored reconstruction residual rather than being a
        // new per-query uncertainty.
        .map(|(primary, correction)| primary + correction)
        .collect()
}

#[cfg(test)]
fn exact_cosine(left: &[f32], right: &[f32]) -> Result<f64, NvFp4Error> {
    if left.len() != right.len() {
        return Err(NvFp4Error::new(format!(
            "cosine dimensions differ: {} and {}",
            left.len(),
            right.len(),
        )));
    }
    let left = normalized_embedding(left, left.len())?;
    exact_cosine_with_normalized_left(&left, right)
}

fn exact_cosine_with_normalized_left(left: &[f64], right: &[f32]) -> Result<f64, NvFp4Error> {
    if left.len() != right.len() {
        return Err(NvFp4Error::new(format!(
            "cosine dimensions differ: {} and {}",
            left.len(),
            right.len(),
        )));
    }
    let right = normalized_embedding(right, right.len())?;
    let mut dot = 0.0f64;
    for (&left, &right) in left.iter().zip(&right) {
        dot += left * right;
    }
    Ok(clamp_cosine(dot))
}

fn clamp_cosine(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

fn certified_cosine_upper(center: f64, envelope: f64) -> f64 {
    clamp_cosine(next_up_f64(center + envelope))
}

fn sort_hits<E: BlobEncoding>(hits: &mut [SimilarityHit<E>]) {
    hits.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.embedding.raw.cmp(&right.embedding.raw))
    });
}

fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        value
    } else if value == -0.0 {
        f64::from_bits(1)
    } else if value >= 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        value
    } else if value == 0.0 {
        -f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn absolute_difference_up(left: f64, right: f64) -> f64 {
    let difference = (left - right).abs();
    if difference == 0.0 {
        0.0
    } else {
        next_up_f64(difference)
    }
}

fn add_up_nonnegative(left: f64, right: f64) -> f64 {
    debug_assert!(left >= 0.0 && right >= 0.0);
    next_up_f64(left + right)
}

fn multiply_up_nonnegative(left: f64, right: f64) -> f64 {
    debug_assert!(left >= 0.0 && right >= 0.0);
    if left == 0.0 || right == 0.0 {
        return 0.0;
    }
    next_up_f64(left * right)
}

/// Higham's gamma bound for `operations` sequential roundings, rounded outward.
fn roundoff_gamma(operations: usize) -> f64 {
    if operations as u128 > 1u128 << f64::MANTISSA_DIGITS {
        return f64::INFINITY;
    }
    let unit = f64::EPSILON / 2.0;
    let numerator = multiply_up_nonnegative(operations as f64, unit);
    if numerator >= 1.0 {
        return f64::INFINITY;
    }
    let denominator = next_down_f64(1.0 - numerator);
    if denominator <= 0.0 {
        f64::INFINITY
    } else {
        next_up_f64(numerator / denominator)
    }
}

/// Gamma bound for a fixed-order dot product.
///
/// Both multiplication and addition are counted. An impossibly large vector
/// for which the standard bound's denominator is no longer positive simply
/// receives an infinite error bound and therefore cannot be pruned.
fn dot_gamma(dimension: usize) -> f64 {
    dimension
        .checked_mul(2)
        .map(roundoff_gamma)
        .unwrap_or(f64::INFINITY)
}

fn uppercase_hex(raw: &[u8]) -> String {
    use std::fmt::Write;

    let mut rendered = String::with_capacity(raw.len() * 2);
    for byte in raw {
        write!(&mut rendered, "{byte:02X}").expect("write to String");
    }
    rendered
}

impl<E> TryFromCover<NvFp4CosineSet<E>> for NvFp4CosineIndex<E>
where
    E: BlobEncoding,
    View<[f32]>: TryFromBlob<E>,
    <View<[f32]> as TryFromBlob<E>>::Error: fmt::Display + Send + Sync + 'static,
{
    type Error = NvFp4Error;

    fn try_from_cover<R>(
        cover: &Cover<NvFp4CosineSet<E>>,
        snapshot: &R,
    ) -> Result<Self, TryFromCoverError<R::GetError<Infallible>, Self::Error>>
    where
        R: BlobStoreGet,
    {
        let descriptor_handle = cover.collection().handle();
        let descriptor_data = Handle::<SimpleArchive>::to_hash(descriptor_handle);
        let descriptor: Blob<SimpleArchive> =
            snapshot
                .get(descriptor_handle)
                .map_err(|source| TryFromCoverError::MemberGet {
                    member: descriptor_data,
                    source,
                })?;
        let facts = TribleSet::try_from_blob(descriptor)
            .map_err(|source| TryFromCoverError::View(NvFp4Error::new(source.to_string())))?;
        let dimension = mapping_dimension_facts(&facts)
            .map_err(|source| TryFromCoverError::View(NvFp4Error::new(source.to_string())))?;

        let mut members = Vec::with_capacity(cover.len());
        for handle in cover.members() {
            let member = Handle::<NvFp4CosineSet<E>>::to_hash(handle);
            let blob: Blob<NvFp4CosineSet<E>> = snapshot
                .get(handle)
                .map_err(|source| TryFromCoverError::MemberGet { member, source })?;
            let layout = Layout::parse(blob.bytes.as_ref()).map_err(TryFromCoverError::View)?;
            if layout.dimension != dimension {
                return Err(TryFromCoverError::View(NvFp4Error::new(format!(
                    "NVFP4 member dimension {} does not match descriptor {dimension}",
                    layout.dimension,
                ))));
            }
            members.push(Member {
                bytes: blob.bytes,
                layout,
            });
        }
        Ok(Self {
            members,
            dimension,
            _encoding: PhantomData,
        })
    }
}

fn read_u64(bytes: &[u8], field: &str) -> Result<u64, NvFp4Error> {
    let raw: [u8; 8] = bytes
        .try_into()
        .map_err(|_| NvFp4Error::new(format!("invalid NVFP4 {field}")))?;
    Ok(u64::from_le_bytes(raw))
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().expect("four-byte float field"))
}

fn validate_nonnegative_f32(value: f32, field: &str) -> Result<(), NvFp4Error> {
    if !value.is_finite() || value.is_sign_negative() {
        return Err(NvFp4Error::new(format!(
            "NVFP4 {field} must be finite and nonnegative"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::Embedding;
    use ed25519_dalek::SigningKey;
    use std::cell::Cell;
    use std::error::Error;
    use triblespace_core::attribute::Attribute;
    use triblespace_core::blob::IntoBlob;
    use triblespace_core::collection::{AdmissionPolicy, CollectionPolicy, CollectionStoreExt};
    use triblespace_core::inline::InlineEncoding;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStorePut, SnapshotSource};
    use triblespace_core::trible::Trible;

    struct Counting<'a, R> {
        inner: &'a R,
        gets: Cell<usize>,
    }

    impl<'a, R> Counting<'a, R> {
        fn new(inner: &'a R) -> Self {
            Self {
                inner,
                gets: Cell::new(0),
            }
        }

        fn gets(&self) -> usize {
            self.gets.get()
        }
    }

    impl<R: BlobStoreGet> BlobStoreGet for Counting<'_, R> {
        type GetError<E: Error + Send + Sync + 'static> = R::GetError<E>;

        fn get<T, S>(
            &self,
            handle: Inline<Handle<S>>,
        ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
        where
            S: BlobEncoding + 'static,
            T: TryFromBlob<S>,
            Handle<S>: InlineEncoding,
        {
            self.gets.set(self.gets.get() + 1);
            self.inner.get(handle)
        }
    }

    fn row(handle: u8, values: &[f32]) -> EncodedRow {
        EncodedRow::quantize([handle; HANDLE_LEN], values, values.len()).unwrap()
    }

    fn member(
        rows: impl IntoIterator<Item = EncodedRow>,
        dimension: usize,
    ) -> Blob<NvFp4CosineSet<Embedding>> {
        encode_rows(dimension, rows.into_iter().collect()).unwrap()
    }

    fn index(blob: &Blob<NvFp4CosineSet<Embedding>>) -> NvFp4CosineIndex<Embedding> {
        NvFp4CosineIndex {
            members: vec![Member {
                layout: Layout::parse(blob.bytes.as_ref()).unwrap(),
                bytes: blob.bytes.clone(),
            }],
            dimension: Layout::parse(blob.bytes.as_ref()).unwrap().dimension,
            _encoding: PhantomData,
        }
    }

    fn embedding_facts(
        attribute: Id,
        rows: impl IntoIterator<Item = (u8, Inline<Handle<Embedding>>)>,
    ) -> TribleSet {
        let mut facts = TribleSet::new();
        for (entity, embedding) in rows {
            let entity = Id::new([entity; 16]).unwrap();
            facts.insert(&Trible::force(&entity, &attribute, &embedding));
        }
        facts
    }

    #[test]
    fn e4m3_dyadic_decode_matches_reference_for_every_canonical_byte() {
        for raw in 0..=0x7e {
            let exponent = (raw >> 3) & 0x0f;
            let mantissa = raw & 0x07;
            let reference = if exponent == 0 {
                f64::from(mantissa) * 2f64.powi(-9)
            } else {
                (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7)
            };
            assert_eq!(
                decode_e4m3(raw).to_bits(),
                reference.to_bits(),
                "E4M3 byte {raw:#04x}",
            );
        }
    }

    #[test]
    fn packed_e2m1_pair_table_decodes_both_coordinates_bitwise() {
        const POSITIVE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let reference = |raw: u8| {
            let magnitude = POSITIVE[usize::from(raw & 0x07)];
            if raw & 0x08 == 0 {
                magnitude
            } else {
                -magnitude
            }
        };

        for raw in u8::MIN..=u8::MAX {
            let decoded = DECODED_E2M1_PAIRS[usize::from(raw)];
            assert_eq!(
                decoded[0].to_bits(),
                reference(raw & 0x0f).to_bits(),
                "low E2M1 nibble of {raw:#04x}",
            );
            assert_eq!(
                decoded[1].to_bits(),
                reference(raw >> 4).to_bits(),
                "high E2M1 nibble of {raw:#04x}",
            );
        }
    }

    #[test]
    fn canonical_rows_and_join_are_aci() {
        let a = row(1, &[1.0, 0.0, 0.0]);
        let b = row(2, &[0.0, 1.0, 0.0]);
        let c = row(3, &[0.0, 0.0, 1.0]);
        let ab = member([b.clone(), a.clone(), a.clone()], 3);
        let ba = member([a.clone(), b.clone()], 3);
        assert_eq!(ab.bytes.as_ref(), ba.bytes.as_ref());

        let bc = member([b, c.clone()], 3);
        let c = member([c], 3);
        let ab_bc = join_members(&ab, &bc, 3).unwrap();
        let bc_ab = join_members(&bc, &ab, 3).unwrap();
        assert_eq!(ab_bc.bytes.as_ref(), bc_ab.bytes.as_ref());

        let idempotent = join_members(&ab_bc, &ab_bc, 3).unwrap();
        assert_eq!(idempotent.bytes.as_ref(), ab_bc.bytes.as_ref());

        let left = join_members(&join_members(&ab, &bc, 3).unwrap(), &c, 3).unwrap();
        let right = join_members(&ab, &join_members(&bc, &c, 3).unwrap(), 3).unwrap();
        assert_eq!(left.bytes.as_ref(), right.bytes.as_ref());
    }

    #[test]
    fn mapping_is_a_join_homomorphism_with_overlap_and_empty() {
        const DIMENSION: usize = 3;
        let attribute = Attribute::<Handle<Embedding>>::named("nvfp4-homomorphism");
        let mapping =
            EmbeddingAttributeToNvFp4::<Embedding>::new(attribute.id(), DIMENSION).unwrap();
        let mut store = MemoryRepo::default();
        let first = store.put::<Embedding, _>(vec![1.0f32, 0.0, 0.0]).unwrap();
        let shared = store.put::<Embedding, _>(vec![0.0f32, 1.0, 0.0]).unwrap();
        let last = store.put::<Embedding, _>(vec![0.0f32, 0.0, 1.0]).unwrap();
        let snapshot = store.snapshot().unwrap();

        let left = embedding_facts(attribute.id(), [(1, first), (2, shared), (3, first)]);
        let right = embedding_facts(attribute.id(), [(4, shared), (5, last)]);
        let mut union = left.clone();
        union += right.clone();

        let mapped_left = mapping.map(&left.to_blob(), &snapshot).unwrap();
        let mapped_right = mapping.map(&right.to_blob(), &snapshot).unwrap();
        let mapped_union = mapping.map(&union.to_blob(), &snapshot).unwrap();
        let joined = join_members(&mapped_left, &mapped_right, DIMENSION).unwrap();
        assert_eq!(mapped_union.bytes.as_ref(), joined.bytes.as_ref());

        let empty = mapping.map(&TribleSet::new().to_blob(), &snapshot).unwrap();
        let with_empty = join_members(&mapped_left, &empty, DIMENSION).unwrap();
        assert_eq!(mapped_left.bytes.as_ref(), with_empty.bytes.as_ref());
    }

    #[test]
    fn lazy_view_and_candidate_scan_do_not_require_exact_sources() {
        const DIMENSION: usize = 3;
        let authority = SigningKey::from_bytes(&[73; 32]);
        let root = authority.verifying_key();
        let policy =
            CollectionPolicy::new(AdmissionPolicy::direct(root), AdmissionPolicy::direct(root));
        let attribute = Attribute::<Handle<Embedding>>::named("nvfp4-lazy-attachment");
        let mut source_store = MemoryRepo::default();
        let exact = source_store
            .put::<Embedding, _>(vec![1.0f32, 0.0, 0.0])
            .unwrap();
        let source = source_store
            .collection("nvfp4-lazy-source", policy.clone())
            .unwrap();
        let target = source_store
            .derive(
                source,
                EmbeddingAttributeToNvFp4::<Embedding>::new(attribute.id(), DIMENSION).unwrap(),
                policy,
            )
            .unwrap();
        source_store
            .commit(
                source,
                &authority,
                Fragment::from(embedding_facts(attribute.id(), [(1, exact)])),
            )
            .unwrap();
        let source_snapshot = source_store.snapshot().unwrap();
        let source_cover = source.admitted(&source_snapshot).unwrap();
        let target_cover = source_store
            .ensure::<EmbeddingAttributeToNvFp4<Embedding>>(target, &source_cover)
            .unwrap();
        let source_snapshot = source_store.snapshot().unwrap();

        // Copy only the target descriptor and compact member into a fresh
        // store. The exact embedding blob is deliberately absent.
        let descriptor: Blob<SimpleArchive> = source_snapshot.get(target.handle()).unwrap();
        let member_handle = target_cover.members().next().unwrap();
        let compact: Blob<NvFp4CosineSet<Embedding>> = source_snapshot.get(member_handle).unwrap();
        let mut sparse = MemoryRepo::default();
        assert_eq!(
            sparse.put::<SimpleArchive, _>(descriptor).unwrap(),
            target.handle(),
        );
        assert_eq!(
            sparse.put::<NvFp4CosineSet<Embedding>, _>(compact).unwrap(),
            member_handle,
        );
        let sparse = sparse.snapshot().unwrap();
        assert!(sparse.metadata(exact).unwrap().is_none());

        let counted = Counting::new(&sparse);
        let index = NvFp4CosineIndex::<Embedding>::try_from_cover(&target_cover, &counted).unwrap();
        assert_eq!(counted.gets(), 1 + target_cover.len());
        let prepared = PreparedQuery::new(&[1.0, 0.0, 0.0], DIMENSION).unwrap();
        assert_eq!(index.candidates(&prepared).unwrap().len(), 1);
        assert_eq!(counted.gets(), 1 + target_cover.len());
    }

    #[test]
    fn zero_row_has_stable_canonical_member_hash() {
        let blob = member([row(0x2a, &[0.0])], 1);
        assert_eq!(blob.bytes.len(), 352);
        assert_eq!(
            uppercase_hex(&blob.get_handle().raw),
            "305800D6C5020C39DBCC988FF4AC43B1D0302B5C4DCC5AAFEDF8D6611AFAEB1B",
        );
    }

    #[test]
    fn stored_error_encloses_the_full_padded_rotated_residual() {
        for dimension in [1, 3, 17, 255, 256, 257] {
            let values: Vec<_> = (0..dimension)
                .map(|index| {
                    let raw = splitmix64(index as u64 ^ dimension as u64);
                    (raw as i64 as f64 / i64::MAX as f64) as f32
                })
                .collect();
            let normalized = normalized_embedding(&values, dimension).unwrap();
            let encoded = row(1, &values);
            let transformed = rotate_normalized(&normalized).unwrap();
            let decoded = decode_encoded_row(&encoded);
            assert_eq!(transformed.len(), decoded.len());
            let measured = transformed
                .iter()
                .zip(&decoded)
                .map(|(&exact, &approximate)| {
                    let difference = exact - approximate;
                    difference * difference
                })
                .sum::<f64>()
                .sqrt();
            assert!(measured <= f64::from(read_f32(&encoded.error)));
            let measured_norm = decoded
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            assert!(measured_norm <= f64::from(read_f32(&encoded.norm)));
        }
    }

    #[test]
    fn residual_stage_strictly_reduces_reconstruction_error() {
        const DIMENSION: usize = 257;
        let values: Vec<_> = (0..DIMENSION)
            .map(|index| {
                let raw = splitmix64(index as u64 ^ 0xE873_2C59_1843_6416);
                (raw as i64 as f64 / i64::MAX as f64) as f32
            })
            .collect();
        let normalized = normalized_embedding(&values, DIMENSION).unwrap();
        let transformed = rotate_normalized(&normalized).unwrap();
        let encoded = row(1, &values);
        let primary = decode_stage(&encoded.stages[0]);
        let corrected = decode_encoded_row(&encoded);
        let primary_error = transformed
            .iter()
            .zip(&primary)
            .map(|(&exact, &approximate)| (exact - approximate).powi(2))
            .sum::<f64>()
            .sqrt();
        let corrected_error = transformed
            .iter()
            .zip(&corrected)
            .map(|(&exact, &approximate)| (exact - approximate).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            corrected_error < primary_error,
            "residual stage should improve {primary_error}, got {corrected_error}",
        );
    }

    #[test]
    fn candidate_upper_bounds_dominate_exact_scores() {
        const DIMENSION: usize = 37;
        let mut seed = 0xA762_09BC_909B_8B0Fu64;
        let mut next_vector = || {
            (0..DIMENSION)
                .map(|_| {
                    seed = splitmix64(seed);
                    let signed = (seed >> 11) as i64 - (1i64 << 52);
                    (signed as f64 / (1u64 << 52) as f64) as f32
                })
                .collect::<Vec<_>>()
        };
        let rows: Vec<_> = (1..=32)
            .map(|handle| {
                let values = next_vector();
                (handle, values)
            })
            .collect();
        let encoded = member(
            rows.iter().map(|(handle, values)| row(*handle, values)),
            DIMENSION,
        );
        let index = index(&encoded);

        for _ in 0..16 {
            let query = next_vector();
            let prepared = PreparedQuery::new(&query, DIMENSION).unwrap();
            let candidates = index.candidates(&prepared).unwrap();
            for (candidate, (_, exact)) in candidates.iter().zip(&rows) {
                let score = exact_cosine(&query, exact).unwrap();
                assert!(
                    score <= candidate.upper,
                    "exact {score} exceeded certified upper {}",
                    candidate.upper,
                );
            }
        }
    }

    #[test]
    fn certified_upper_uses_the_exact_rerankers_clamped_codomain() {
        let exact = exact_cosine(&[1.0], &[-1.0]).unwrap();
        assert_eq!(exact, -1.0);
        // A certificate for the unclamped floating dot may legitimately fall
        // below -1. The returned exact scorer raises that value to -1, so the
        // upper certificate must undergo the same clamp before pruning.
        assert_eq!(certified_cosine_upper(-1.5, 0.25), exact);
    }

    #[test]
    fn outward_f32_rejects_an_unrepresentable_finite_bound() {
        assert_eq!(upward_f32(f64::from(f32::MAX)).unwrap(), f32::MAX);
        assert!(upward_f32(next_up_f64(f64::from(f32::MAX))).is_err());
    }

    #[test]
    fn malformed_or_conflicting_members_are_rejected() {
        let one = row(1, &[1.0, 0.0]);
        let conflicting = row(1, &[0.0, 1.0]);
        assert!(encode_rows::<Embedding>(2, vec![one, conflicting]).is_err());

        let mut malformed = member([row(2, &[1.0, 1.0])], 2).bytes.as_ref().to_vec();
        malformed[0] = 3;
        assert!(Layout::parse(&malformed).is_ok());
        let last_code = Layout::parse(&malformed).unwrap().stages[0].codes.start;
        malformed[last_code] = 0x08;
        assert!(Layout::parse(&malformed).is_err());
    }
}
