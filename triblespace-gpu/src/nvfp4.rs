use std::fmt;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle as DeviceHandle;
use triblespace_core::blob::BlobEncoding;
use triblespace_search::nvfp4::{
    NvFp4CosineIndex, NvFp4DotScanner, NvFp4ScanQuery, NvFp4ScanSegment,
};

type CudaRuntime = cubecl::cuda::CudaRuntime;

const PLANE_SIZE: u32 = 32;
const THREADS: u32 = 256;

#[cube]
fn decode_e2m1(raw: u32) -> f64 {
    let magnitude = raw & 7u32;
    let exponent = (magnitude >> 1u32) & 3u32;
    let mantissa = magnitude & 1u32;
    let mut value = if exponent == 0u32 {
        f64::cast_from(mantissa) * 0.5f64
    } else {
        let mut scale = 0.5f64;
        let mut step = 1u32;
        while step < exponent {
            scale *= 2.0f64;
            step += 1u32;
        }
        f64::cast_from(2u32 + mantissa) * scale
    };
    if raw & 8u32 != 0u32 {
        value = -value;
    }
    value
}

#[cube]
fn decode_e4m3(raw: u32) -> f64 {
    let exponent = (raw >> 3u32) & 15u32;
    let mantissa = raw & 7u32;
    if exponent == 0u32 {
        f64::cast_from(mantissa) * 0.001953125f64
    } else {
        let mut scale = 0.0009765625f64;
        let mut step = 0u32;
        while step < exponent {
            scale *= 2.0f64;
            step += 1u32;
        }
        f64::cast_from(8u32 + mantissa) * scale
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
/// Decode and dot one canonical row per CUDA plane.
///
/// Coordinate reconstruction is bit-identical to the canonical CPU `f64`
/// path even if CUDA contracts a multiplication into the stage addition. A
/// binary32 global contributes at most 24 significand bits, canonical E4M3 at
/// most four, and E2M1 at most two, so every pre-add product is an exactly
/// representable dyadic with at most 30 bits. Their full exponent range also
/// lies inside `f64`. Consequently contraction still performs the same single
/// rounding at `primary + correction`. Only the subsequent query products and
/// their plane reduction may round differently; search encloses those with
/// its dimension-wide `gamma(2D)` certificate.
fn nvfp4_decode_dot(
    query: &Array<f64>,
    primary_globals: &Array<f32>,
    primary_scales: &Array<u8>,
    primary_codes: &Array<u8>,
    correction_globals: &Array<f32>,
    correction_scales: &Array<u8>,
    correction_codes: &Array<u8>,
    dots: &mut Array<f64>,
    rows: u32,
    blocks_per_row: u32,
    codes_per_row: u32,
    output_row_offset: u32,
) {
    let row = (ABSOLUTE_POS as u32) / PLANE_SIZE;
    if row < rows {
        let lane = UNIT_POS_PLANE;
        let primary_global = f64::cast_from(primary_globals[row as usize]);
        let correction_global = f64::cast_from(correction_globals[row as usize]);
        let mut partial = 0.0f64;
        let mut code = lane;
        while code < codes_per_row {
            let block = code / 8u32;
            let scale_index = row * blocks_per_row + block;
            let primary_scale =
                primary_global * decode_e4m3(u32::cast_from(primary_scales[scale_index as usize]));
            let correction_scale = correction_global
                * decode_e4m3(u32::cast_from(correction_scales[scale_index as usize]));
            let code_index = row * codes_per_row + code;
            let primary = u32::cast_from(primary_codes[code_index as usize]);
            let correction = u32::cast_from(correction_codes[code_index as usize]);
            let primary_low = decode_e2m1(primary & 15u32) * primary_scale;
            let primary_high = decode_e2m1(primary >> 4u32) * primary_scale;
            let correction_low = decode_e2m1(correction & 15u32) * correction_scale;
            let correction_high = decode_e2m1(correction >> 4u32) * correction_scale;
            let low = primary_low + correction_low;
            let high = primary_high + correction_high;
            let coordinate = code * 2u32;
            partial += query[coordinate as usize] * low;
            partial += query[(coordinate + 1u32) as usize] * high;
            code += PLANE_SIZE;
        }
        let dot = plane_sum(partial);
        if lane == 0u32 {
            dots[(output_row_offset + row) as usize] = dot;
        }
    }
}

#[derive(Clone)]
struct ResidentStage {
    globals: DeviceHandle,
    block_scales: DeviceHandle,
    codes: DeviceHandle,
}

#[derive(Clone)]
struct ResidentSegment {
    content_handle: [u8; 32],
    rows: u32,
    blocks_per_row: u32,
    codes_per_row: u32,
    stages: Option<[ResidentStage; 2]>,
}

/// Failure to construct or execute a resident CUDA NVFP4 candidate scan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NvFp4CudaError {
    /// A segment dimension does not fit the CUDA kernel's `u32` geometry.
    GeometryOverflow(&'static str),
    /// CUDA does not expose the warp contract used by one-row-per-plane scan.
    UnsupportedPlane { min: u32, max: u32 },
    /// The scanner was invoked with a different ordered segment sequence.
    SegmentMismatch { index: usize },
    /// Search supplied a query or output slice of an unexpected length.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A CUDA command or synchronization failed.
    Device(String),
}

impl fmt::Display for NvFp4CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeometryOverflow(what) => write!(formatter, "NVFP4 CUDA {what} exceeds u32"),
            Self::UnsupportedPlane { min, max } => write!(
                formatter,
                "NVFP4 CUDA scan needs 32-lane planes, device reports {min}..={max}",
            ),
            Self::SegmentMismatch { index } => {
                write!(
                    formatter,
                    "NVFP4 CUDA resident segment {index} does not match"
                )
            }
            Self::ShapeMismatch {
                what,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 CUDA {what} has length {actual}, expected {expected}",
            ),
            Self::Device(error) => write!(formatter, "NVFP4 CUDA operation failed: {error}"),
        }
    }
}

impl std::error::Error for NvFp4CudaError {}

/// CUDA-resident canonical NVFP4 planes for one immutable search index.
///
/// Construction uploads every physical segment once. Queries upload only the
/// prepared `f64` coordinates and read back one `f64` dot per physical row.
/// Candidate certificates, overlap deduplication, and exact reranking remain
/// in `triblespace-search` through [`NvFp4DotScanner`].
pub struct CudaNvFp4DotScanner {
    client: ComputeClient<CudaRuntime>,
    segments: Vec<ResidentSegment>,
    physical_dimension: usize,
    physical_rows: usize,
}

impl CudaNvFp4DotScanner {
    /// Upload the canonical compact planes of `index` to `device`.
    pub fn new<E: BlobEncoding>(
        index: &NvFp4CosineIndex<E>,
        device: &cubecl::cuda::CudaDevice,
    ) -> Result<Self, NvFp4CudaError> {
        use cubecl::ir::features::Plane;

        let client = CudaRuntime::client(device);
        let properties = client.properties();
        let (min, max) = (
            properties.hardware.plane_size_min,
            properties.hardware.plane_size_max,
        );
        if !properties.features.plane.contains(Plane::Ops) || min != PLANE_SIZE || max != PLANE_SIZE
        {
            return Err(NvFp4CudaError::UnsupportedPlane { min, max });
        }

        let source = index.scan_segments();
        let physical_dimension = source
            .first()
            .map(|segment| segment.codes_per_row() * 2)
            .unwrap_or(0);
        let mut physical_rows = 0u32;
        let mut segments = Vec::with_capacity(source.len());
        for segment in source {
            let rows = u32::try_from(segment.rows())
                .map_err(|_| NvFp4CudaError::GeometryOverflow("row count"))?;
            let blocks_per_row = u32::try_from(segment.blocks_per_row())
                .map_err(|_| NvFp4CudaError::GeometryOverflow("block count"))?;
            let codes_per_row = u32::try_from(segment.codes_per_row())
                .map_err(|_| NvFp4CudaError::GeometryOverflow("code count"))?;
            rows.checked_mul(PLANE_SIZE)
                .and_then(|threads| threads.checked_add(THREADS - 1))
                .ok_or(NvFp4CudaError::GeometryOverflow("thread count"))?;
            rows.checked_mul(blocks_per_row)
                .ok_or(NvFp4CudaError::GeometryOverflow("scale-plane index"))?;
            rows.checked_mul(codes_per_row)
                .ok_or(NvFp4CudaError::GeometryOverflow("code-plane index"))?;
            codes_per_row
                .checked_mul(2)
                .ok_or(NvFp4CudaError::GeometryOverflow("query-coordinate index"))?;
            physical_rows = physical_rows
                .checked_add(rows)
                .ok_or(NvFp4CudaError::GeometryOverflow("physical row count"))?;
            let stages = if rows == 0 {
                None
            } else {
                Some(segment.stages().map(|stage| {
                    let globals: Vec<_> = stage
                        .global_scale_bytes()
                        .chunks_exact(std::mem::size_of::<f32>())
                        .map(|raw| {
                            f32::from_le_bytes(raw.try_into().expect("validated f32 scale bytes"))
                        })
                        .collect();
                    ResidentStage {
                        globals: client.create_from_slice(f32::as_bytes(&globals)),
                        block_scales: client.create_from_slice(stage.block_scales()),
                        codes: client.create_from_slice(stage.codes()),
                    }
                }))
            };
            segments.push(ResidentSegment {
                content_handle: segment.content_handle(),
                rows,
                blocks_per_row,
                codes_per_row,
                stages,
            });
        }
        Ok(Self {
            client,
            segments,
            physical_dimension,
            physical_rows: physical_rows as usize,
        })
    }
}

impl NvFp4DotScanner for CudaNvFp4DotScanner {
    type Error = NvFp4CudaError;

    fn scan(
        &self,
        query: NvFp4ScanQuery<'_>,
        segments: &[NvFp4ScanSegment<'_>],
        dots: &mut [f64],
    ) -> Result<(), Self::Error> {
        if segments.len() != self.segments.len() {
            return Err(NvFp4CudaError::ShapeMismatch {
                what: "segment sequence",
                expected: self.segments.len(),
                actual: segments.len(),
            });
        }
        for (index, (resident, source)) in self.segments.iter().zip(segments).enumerate() {
            if resident.content_handle != source.content_handle() {
                return Err(NvFp4CudaError::SegmentMismatch { index });
            }
        }
        if dots.len() != self.physical_rows {
            return Err(NvFp4CudaError::ShapeMismatch {
                what: "output",
                expected: self.physical_rows,
                actual: dots.len(),
            });
        }
        if self.physical_rows == 0 {
            return Ok(());
        }
        let query = query.coordinates();
        if query.len() != self.physical_dimension {
            return Err(NvFp4CudaError::ShapeMismatch {
                what: "query",
                expected: self.physical_dimension,
                actual: query.len(),
            });
        }

        let query_handle = self.client.create_from_slice(f64::as_bytes(query));
        let cube_dim = CubeDim::new_1d(THREADS);
        let output_bytes = self
            .physical_rows
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(NvFp4CudaError::GeometryOverflow("output byte count"))?;
        let output = self.client.empty(output_bytes);
        let mut output_row_offset = 0u32;
        for segment in &self.segments {
            let Some(stages) = &segment.stages else {
                continue;
            };
            let threads = segment.rows as usize * PLANE_SIZE as usize;
            let dispatch = cubecl::calculate_cube_count_elemwise(&self.client, threads, cube_dim);
            unsafe {
                nvfp4_decode_dot::launch_unchecked::<CudaRuntime>(
                    &self.client,
                    dispatch,
                    cube_dim,
                    ArrayArg::from_raw_parts(query_handle.clone(), query.len()),
                    ArrayArg::from_raw_parts(stages[0].globals.clone(), segment.rows as usize),
                    ArrayArg::from_raw_parts(
                        stages[0].block_scales.clone(),
                        segment.rows as usize * segment.blocks_per_row as usize,
                    ),
                    ArrayArg::from_raw_parts(
                        stages[0].codes.clone(),
                        segment.rows as usize * segment.codes_per_row as usize,
                    ),
                    ArrayArg::from_raw_parts(stages[1].globals.clone(), segment.rows as usize),
                    ArrayArg::from_raw_parts(
                        stages[1].block_scales.clone(),
                        segment.rows as usize * segment.blocks_per_row as usize,
                    ),
                    ArrayArg::from_raw_parts(
                        stages[1].codes.clone(),
                        segment.rows as usize * segment.codes_per_row as usize,
                    ),
                    ArrayArg::from_raw_parts(output.clone(), self.physical_rows),
                    segment.rows,
                    segment.blocks_per_row,
                    segment.codes_per_row,
                    output_row_offset,
                )
            };
            output_row_offset += segment.rows;
        }

        let bytes = self
            .client
            .read_one(output)
            .map_err(|error| NvFp4CudaError::Device(format!("{error:?}")))?;
        if bytes.len() != output_bytes {
            return Err(NvFp4CudaError::ShapeMismatch {
                what: "readback bytes",
                expected: output_bytes,
                actual: bytes.len(),
            });
        }
        for (slot, chunk) in dots
            .iter_mut()
            .zip(bytes.chunks_exact(std::mem::size_of::<f64>()))
        {
            let raw: [u8; 8] = chunk.try_into().expect("eight-byte CUDA f64");
            *slot = f64::from_ne_bytes(raw);
        }
        Ok(())
    }
}

#[cfg(test)]
#[cube(launch_unchecked)]
fn decode_coordinate_extrema(
    primary_globals: &Array<f32>,
    primary_scales: &Array<u8>,
    primary_codes: &Array<u8>,
    correction_globals: &Array<f32>,
    correction_scales: &Array<u8>,
    correction_codes: &Array<u8>,
    coordinates: &mut Array<f64>,
) {
    let index = ABSOLUTE_POS;
    if index < coordinates.len() {
        let primary = decode_e2m1(u32::cast_from(primary_codes[index]))
            * (f64::cast_from(primary_globals[index])
                * decode_e4m3(u32::cast_from(primary_scales[index])));
        let correction = decode_e2m1(u32::cast_from(correction_codes[index]))
            * (f64::cast_from(correction_globals[index])
                * decode_e4m3(u32::cast_from(correction_scales[index])));
        coordinates[index] = primary + correction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use triblespace_core::attribute::Attribute;
    use triblespace_core::collection::{
        AdmissionPolicy, CollectionPolicy, CollectionStoreExt, TryFromCover,
    };
    use triblespace_core::id::Id;
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStorePut, SnapshotSource};
    use triblespace_core::trible::{Fragment, Trible, TribleSet};
    use triblespace_search::nvfp4::EmbeddingAttributeToNvFp4;
    use triblespace_search::schemas::Embedding;

    fn host_e2m1(raw: u8) -> f64 {
        const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let magnitude = MAGNITUDES[usize::from(raw & 7)];
        if raw & 8 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    fn host_e4m3(raw: u8) -> f64 {
        let exponent = (raw >> 3) & 15;
        let mantissa = raw & 7;
        if exponent == 0 {
            f64::from(mantissa) * 2f64.powi(-9)
        } else {
            f64::from(8 + mantissa) * 2f64.powi(i32::from(exponent) - 10)
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA CUDA device"]
    fn cuda_extremal_coordinate_decode_is_bit_identical() {
        let primary_globals = [
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::MAX,
        ];
        let primary_scales = [0x00, 0x01, 0x07, 0x38, 0x7e, 0x7e];
        let primary_codes = [0x00, 0x01, 0x07, 0x0f, 0x07, 0x0f];
        let correction_globals = [
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            0.5,
            f32::MAX,
            f32::MAX,
        ];
        let correction_scales = [0x7e, 0x07, 0x01, 0x38, 0x7e, 0x7e];
        let correction_codes = [0x00, 0x0f, 0x01, 0x07, 0x0f, 0x07];

        let client = CudaRuntime::client(&cubecl::cuda::CudaDevice::default());
        let output = client.empty(primary_globals.len() * std::mem::size_of::<f64>());
        let cube_dim = CubeDim::new_1d(32);
        unsafe {
            decode_coordinate_extrema::launch_unchecked::<CudaRuntime>(
                &client,
                cubecl::calculate_cube_count_elemwise(&client, primary_globals.len(), cube_dim),
                cube_dim,
                ArrayArg::from_raw_parts(
                    client.create_from_slice(f32::as_bytes(&primary_globals)),
                    primary_globals.len(),
                ),
                ArrayArg::from_raw_parts(
                    client.create_from_slice(&primary_scales),
                    primary_scales.len(),
                ),
                ArrayArg::from_raw_parts(
                    client.create_from_slice(&primary_codes),
                    primary_codes.len(),
                ),
                ArrayArg::from_raw_parts(
                    client.create_from_slice(f32::as_bytes(&correction_globals)),
                    correction_globals.len(),
                ),
                ArrayArg::from_raw_parts(
                    client.create_from_slice(&correction_scales),
                    correction_scales.len(),
                ),
                ArrayArg::from_raw_parts(
                    client.create_from_slice(&correction_codes),
                    correction_codes.len(),
                ),
                ArrayArg::from_raw_parts(output.clone(), primary_globals.len()),
            )
        };
        let bytes = client.read_one(output).unwrap();
        for (index, raw) in bytes.chunks_exact(std::mem::size_of::<f64>()).enumerate() {
            let actual = f64::from_ne_bytes(raw.try_into().unwrap());
            let primary = host_e2m1(primary_codes[index])
                * (f64::from(primary_globals[index]) * host_e4m3(primary_scales[index]));
            let correction = host_e2m1(correction_codes[index])
                * (f64::from(correction_globals[index]) * host_e4m3(correction_scales[index]));
            let expected = primary + correction;
            assert_eq!(actual.to_bits(), expected.to_bits(), "extreme case {index}");
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA CUDA device"]
    fn cuda_candidate_scan_matches_cpu_exact_results() {
        const DIMENSION: usize = 37;
        let authority = SigningKey::from_bytes(&[91; 32]);
        let root = authority.verifying_key();
        let policy =
            CollectionPolicy::new(AdmissionPolicy::direct(root), AdmissionPolicy::direct(root));
        let attribute = Attribute::<Handle<Embedding>>::named("nvfp4-cuda-parity");
        let mut store = MemoryRepo::default();
        let mut facts = TribleSet::new();
        for row in 1u8..=64 {
            let values: Vec<_> = (0..DIMENSION)
                .map(|coordinate| {
                    let phase = f32::from(row) * 0.37 + coordinate as f32 * 0.19;
                    phase.sin() + (phase * 0.31).cos()
                })
                .collect();
            let embedding = store.put::<Embedding, _>(values).unwrap();
            let entity = Id::new([row; 16]).unwrap();
            facts.insert(&Trible::force(&entity, &attribute.id(), &embedding));
        }
        let source = store
            .collection("nvfp4-cuda-source", policy.clone())
            .unwrap();
        let target = store
            .derive(
                source,
                EmbeddingAttributeToNvFp4::<Embedding>::new(attribute.id(), DIMENSION).unwrap(),
                policy,
            )
            .unwrap();
        store
            .commit(source, &authority, Fragment::from(facts))
            .unwrap();
        let snapshot = store.snapshot().unwrap();
        let source_cover = source.admitted(&snapshot).unwrap();
        let target_cover = store
            .ensure::<EmbeddingAttributeToNvFp4<Embedding>>(target, &source_cover)
            .unwrap();
        let snapshot = store.snapshot().unwrap();
        let index =
            NvFp4CosineIndex::<Embedding>::try_from_cover(&target_cover, &snapshot).unwrap();
        let scanner =
            CudaNvFp4DotScanner::new(&index, &cubecl::cuda::CudaDevice::default()).unwrap();

        for query_index in 0..8 {
            let query: Vec<_> = (0..DIMENSION)
                .map(|coordinate| {
                    let phase = query_index as f32 * 0.43 + coordinate as f32 * 0.23;
                    phase.cos() - (phase * 0.17).sin()
                })
                .collect();
            for k in [1, 3, 17, 64] {
                let cpu = index.top_k(&snapshot, &query, k).unwrap();
                let cuda = index
                    .top_k_with_scanner(&snapshot, &query, k, &scanner)
                    .unwrap();
                assert!(cpu == cuda, "top-{k} differs for query {query_index}");
            }
            for floor in [-0.5, 0.0, 0.5, 0.9] {
                let cpu = index.above(&snapshot, &query, floor).unwrap();
                let cuda = index
                    .above_with_scanner(&snapshot, &query, floor, &scanner)
                    .unwrap();
                assert!(
                    cpu == cuda,
                    "threshold {floor} differs for query {query_index}",
                );
            }
        }
    }
}
