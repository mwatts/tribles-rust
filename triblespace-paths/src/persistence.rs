use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use triblespace_core::blob::{Blob, BlobEncoding};
use triblespace_core::id::{ExclusiveId, Id};
use triblespace_core::id_hex;
use triblespace_core::inline::encodings::hash::{Blake3, Hash};
use triblespace_core::inline::Inline;
use triblespace_core::metadata::{self, MetaDescribe};
use triblespace_core::prelude::{attributes, entity};
use triblespace_core::trible::Fragment;

use crate::{Automaton, PathSummary, Step};

const HEADER_LEN: usize = 48;
const FINGERPRINT_VERSION: u32 = 1;
// The automaton wire domain separator is the bytes of the stable
// path-summary-v2 algorithm anchor `341216BFE738E2D82BFFF96F52E7FE06`, minted
// with `trible genid` on 2026-07-28 when canonical summaries were restricted
// to matched support plus nullable identity.
const AUTOMATON_FINGERPRINT_DOMAIN: [u8; 16] = [
    0x34, 0x12, 0x16, 0xbf, 0xe7, 0x38, 0xe2, 0xd8, 0x2b, 0xff, 0xf9, 0x6f, 0x52, 0xe7, 0xfe, 0x06,
];

attributes! {
    /// Canonical BLAKE3 fingerprint of the fixed path automaton. Minted with
    /// `trible genid` on 2026-07-28.
    "77DF5A905CCE3B0643BB02999F73BE4C" unsafe as pub path_automaton_fingerprint: Hash<Blake3>;
}

/// Canonical direct-product summary bytes for one fixed automaton.
pub struct PathSummaryBlob;

impl BlobEncoding for PathSummaryBlob {}

impl MetaDescribe for PathSummaryBlob {
    fn describe() -> Fragment {
        // Minted with `trible genid` on 2026-07-28 when canonical-domain
        // validation changed to matched support plus nullable identity.
        let id: Id = id_hex!("F15A8487F9372278E10F220DC37C2888");
        entity! { ExclusiveId::force_ref(&id) @
            metadata::name: "path-summary-v2",
            metadata::description: "Canonical fixed-header path summary: the sorted graph-term domain required by one fixed automaton followed by sorted direct product arcs. Nullable automata retain the complete supplied endpoint universe; the 48-byte zero-vertex form is the canonical empty summary.",
            metadata::tag: metadata::KIND_BLOB_ENCODING,
        }
    }
}

/// Canonical path-summary validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PathSummaryBlobError {
    /// The fixed header or payload length is invalid.
    BadLength,
    /// The blob belongs to a different fixed automaton.
    DifferentAutomaton,
    /// A non-nullable summary retained vertices outside matched-edge support.
    NoncanonicalDomain,
    /// Vertex values are not strictly increasing.
    VertexOrder,
    /// Product arcs are not strictly increasing.
    ArcOrder,
    /// A product ordinal lies outside the declared carrier.
    ArcOutOfBounds,
    /// An arc's source/destination state pair is absent from the automaton.
    InvalidStatePair,
    /// A length or carrier calculation overflowed its representation.
    CapacityOverflow,
}

impl fmt::Display for PathSummaryBlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::BadLength => "path-summary blob has an invalid length",
            Self::DifferentAutomaton => "path-summary blob belongs to a different automaton",
            Self::NoncanonicalDomain => {
                "a non-nullable path summary contains vertices outside matched-edge support"
            }
            Self::VertexOrder => "path-summary vertices are not strictly ordered",
            Self::ArcOrder => "path-summary arcs are not strictly ordered",
            Self::ArcOutOfBounds => "path-summary arc is outside the product carrier",
            Self::InvalidStatePair => "path-summary arc uses an impossible automaton state pair",
            Self::CapacityOverflow => "path-summary dimensions overflow their representation",
        };
        f.write_str(message)
    }
}

impl Error for PathSummaryBlobError {}

impl PathSummaryBlob {
    /// Encode one canonical constructional summary.
    ///
    /// Product arcs use full-domain `u32` ordinals on disk. A persisted
    /// nullable summary therefore still requires `|U| * |Q| <= u32::MAX`,
    /// even though materialization closes only the smaller matched support.
    /// The empty summary is the fixed 48-byte header with zero vertices and
    /// arcs; retaining it makes derivation a total join homomorphism.
    pub fn encode(summary: &PathSummary) -> Result<Blob<Self>, PathSummaryBlobError> {
        if !summary.has_canonical_domain() {
            return Err(PathSummaryBlobError::NoncanonicalDomain);
        }
        let vertex_count = u32::try_from(summary.vertices().len())
            .map_err(|_| PathSummaryBlobError::CapacityOverflow)?;
        checked_product_count(summary.vertices().len(), summary.automaton().state_count())?;
        let arcs = summary.ordinal_arcs().collect::<Vec<_>>();
        let arc_count =
            u64::try_from(arcs.len()).map_err(|_| PathSummaryBlobError::CapacityOverflow)?;
        let vertex_bytes = summary
            .vertices()
            .len()
            .checked_mul(32)
            .ok_or(PathSummaryBlobError::CapacityOverflow)?;
        let arc_bytes = arcs
            .len()
            .checked_mul(8)
            .ok_or(PathSummaryBlobError::CapacityOverflow)?;
        let capacity = HEADER_LEN
            .checked_add(vertex_bytes)
            .and_then(|length| length.checked_add(arc_bytes))
            .ok_or(PathSummaryBlobError::CapacityOverflow)?;
        let fingerprint = automaton_fingerprint(summary.automaton());

        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&fingerprint.raw);
        bytes.extend_from_slice(&summary.automaton().state_count().to_le_bytes());
        bytes.extend_from_slice(&vertex_count.to_le_bytes());
        bytes.extend_from_slice(&arc_count.to_le_bytes());
        for vertex in summary.vertices() {
            bytes.extend_from_slice(vertex);
        }
        for (source, target) in arcs {
            bytes.extend_from_slice(&source.to_le_bytes());
            bytes.extend_from_slice(&target.to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), capacity);
        Ok(Blob::new(bytes.into()))
    }

    /// Validate and decode canonical bytes against the expected automaton.
    pub fn decode(
        blob: Blob<Self>,
        automaton: &Automaton,
    ) -> Result<PathSummary, PathSummaryBlobError> {
        let bytes = blob.bytes.as_ref();
        let header = validate_header(bytes, automaton)?;
        let vertex_count = header.vertex_count;
        let vertex_bytes = header.vertex_bytes;
        let state_count = automaton.state_count();
        let product_count = header.product_count;

        let mut vertices = Vec::with_capacity(vertex_count);
        for chunk in bytes[HEADER_LEN..HEADER_LEN + vertex_bytes].chunks_exact(32) {
            vertices.push(chunk.try_into().expect("chunks_exact yields 32 bytes"));
        }
        if vertices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PathSummaryBlobError::VertexOrder);
        }

        let allowed_state_pairs = automaton
            .transitions()
            .iter()
            .map(|transition| (transition.from, transition.to))
            .collect::<BTreeSet<_>>();
        let mut arcs = Vec::with_capacity(header.arc_count);
        let mut previous = None;
        for chunk in bytes[HEADER_LEN + vertex_bytes..].chunks_exact(8) {
            let source = u32::from_le_bytes(chunk[..4].try_into().expect("four bytes"));
            let target = u32::from_le_bytes(chunk[4..].try_into().expect("four bytes"));
            if source as usize >= product_count || target as usize >= product_count {
                return Err(PathSummaryBlobError::ArcOutOfBounds);
            }
            if !allowed_state_pairs.contains(&(source % state_count, target % state_count)) {
                return Err(PathSummaryBlobError::InvalidStatePair);
            }
            let arc = (source, target);
            if previous.is_some_and(|previous| previous >= arc) {
                return Err(PathSummaryBlobError::ArcOrder);
            }
            previous = Some(arc);
            arcs.push(arc);
        }
        let summary = PathSummary::from_canonical_ordinals(automaton.clone(), vertices, arcs)
            .map_err(|_| PathSummaryBlobError::CapacityOverflow)?;
        if !summary.has_canonical_domain() {
            return Err(PathSummaryBlobError::NoncanonicalDomain);
        }
        Ok(summary)
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedHeader {
    vertex_count: usize,
    vertex_bytes: usize,
    arc_count: usize,
    product_count: usize,
}

fn validate_header(
    bytes: &[u8],
    automaton: &Automaton,
) -> Result<ValidatedHeader, PathSummaryBlobError> {
    if bytes.len() < HEADER_LEN {
        return Err(PathSummaryBlobError::BadLength);
    }
    let expected_fingerprint = automaton_fingerprint(automaton);
    if bytes[..32] != expected_fingerprint.raw {
        return Err(PathSummaryBlobError::DifferentAutomaton);
    }
    let state_count = read_u32(bytes, 32);
    if state_count != automaton.state_count() {
        return Err(PathSummaryBlobError::DifferentAutomaton);
    }
    let vertex_count = read_u32(bytes, 36) as usize;
    let arc_count =
        usize::try_from(read_u64(bytes, 40)).map_err(|_| PathSummaryBlobError::CapacityOverflow)?;
    let vertex_bytes = vertex_count
        .checked_mul(32)
        .ok_or(PathSummaryBlobError::CapacityOverflow)?;
    let arc_bytes = arc_count
        .checked_mul(8)
        .ok_or(PathSummaryBlobError::CapacityOverflow)?;
    let expected_length = HEADER_LEN
        .checked_add(vertex_bytes)
        .and_then(|length| length.checked_add(arc_bytes))
        .ok_or(PathSummaryBlobError::CapacityOverflow)?;
    if bytes.len() != expected_length {
        return Err(PathSummaryBlobError::BadLength);
    }
    let product_count = checked_product_count(vertex_count, state_count)?;
    Ok(ValidatedHeader {
        vertex_count,
        vertex_bytes,
        arc_count,
        product_count,
    })
}

fn checked_product_count(
    vertex_count: usize,
    state_count: u32,
) -> Result<usize, PathSummaryBlobError> {
    let product_count = vertex_count
        .checked_mul(state_count as usize)
        .ok_or(PathSummaryBlobError::CapacityOverflow)?;
    if product_count > u32::MAX as usize {
        return Err(PathSummaryBlobError::CapacityOverflow);
    }
    Ok(product_count)
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("checked header"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("checked header"),
    )
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn transition_wire(transition: &crate::Transition) -> Vec<u8> {
    let (opcode, labels): (u8, &[triblespace_core::id::RawId]) = match &transition.step {
        Step::Forward(label) => (0, std::slice::from_ref(label)),
        Step::Reverse(label) => (1, std::slice::from_ref(label)),
        Step::ForwardExcept(labels) => (2, labels),
        Step::ReverseExcept(labels) => (3, labels),
    };
    let mut bytes = Vec::with_capacity(17 + labels.len() * 16);
    push_u32(&mut bytes, transition.from);
    push_u32(&mut bytes, transition.to);
    bytes.push(opcode);
    push_u64(
        &mut bytes,
        u64::try_from(labels.len()).expect("label vector length fits in u64"),
    );
    for label in labels {
        bytes.extend_from_slice(label);
    }
    bytes
}

/// Stable wire fingerprint of a canonical fixed automaton.
pub fn automaton_fingerprint(automaton: &Automaton) -> Inline<Hash<Blake3>> {
    let initial = automaton.initial_states().collect::<Vec<_>>();
    let accepting = automaton.accepting_states().collect::<Vec<_>>();
    let mut transitions = automaton
        .transitions()
        .iter()
        .map(transition_wire)
        .collect::<Vec<_>>();
    transitions.sort_unstable();

    let mut wire = Vec::new();
    wire.extend_from_slice(&AUTOMATON_FINGERPRINT_DOMAIN);
    push_u32(&mut wire, FINGERPRINT_VERSION);
    push_u32(&mut wire, automaton.state_count());
    push_u32(
        &mut wire,
        u32::try_from(initial.len()).expect("automaton state count is u32"),
    );
    push_u32(
        &mut wire,
        u32::try_from(accepting.len()).expect("automaton state count is u32"),
    );
    push_u64(
        &mut wire,
        u64::try_from(transitions.len()).expect("transition vector length fits in u64"),
    );
    for state in initial {
        push_u32(&mut wire, state);
    }
    for state in accepting {
        push_u32(&mut wire, state);
    }
    for transition in transitions {
        wire.extend_from_slice(&transition);
    }
    Inline::new(Blake3::digest(&wire))
}

/// The path-summary law.
///
/// This names the law only. *Which* automaton a summary is over is an argument
/// carried on the collection descriptor as `path_automaton_fingerprint`, not
/// folded into this id: a digest of an automaton that is stored nowhere would
/// leave the collection's meaning unrecoverable from the pile.
pub const PATH_SUMMARY_RECIPE_V1: Id =
    triblespace_core::id_hex!("341216BFE738E2D82BFFF96F52E7FE06");

/// The path-summary law, as a describable type.
pub struct PathSummaryV1;

impl MetaDescribe for PathSummaryV1 {
    fn describe() -> Fragment {
        let id: Id = PATH_SUMMARY_RECIPE_V1;
        entity! {
            ExclusiveId::force_ref(&id) @
                metadata::name: "path-summary-v1",
                metadata::description: "Summary of the vertex pairs connected by a fixed regular path automaton, unioned across a collection's elements. Union is associative, commutative and idempotent, so summaries merge without regard to order. Takes one argument, carried as a trible on the collection descriptor: `path_automaton_fingerprint`, the digest of the epsilon-free automaton the summary is over. A different automaton is a different collection, because it answers a different question.",
                metadata::tag: metadata::KIND_COLLECTION_RECIPE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use triblespace_core::inline::RawInline;

    use crate::{GraphEdge, Transition};

    fn vertex(byte: u8) -> RawInline {
        [byte; 32]
    }

    fn label(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn edge(source: u8, attribute: u8, target: u8) -> GraphEdge {
        GraphEdge {
            source: vertex(source),
            attribute: label(attribute),
            target: vertex(target),
        }
    }

    fn plus(attribute: u8) -> Automaton {
        let attribute = label(attribute);
        Automaton::new(
            2,
            [0],
            [1],
            [
                Transition::new(0, 1, Step::Forward(attribute)),
                Transition::new(1, 1, Step::Forward(attribute)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn canonical_bytes_are_input_order_invariant_and_golden() {
        let automaton = Automaton::new(
            1,
            [0, 0],
            [0],
            [
                Transition::new(0, 0, Step::Forward(label(9))),
                Transition::new(0, 0, Step::Forward(label(9))),
            ],
        )
        .unwrap();
        let first = PathSummary::from_edges(automaton.clone(), [edge(1, 9, 2), edge(1, 9, 2)]);
        let second = PathSummary::from_edges(automaton, [edge(1, 9, 2)]);
        let first_blob = PathSummaryBlob::encode(&first).unwrap();
        let second_blob = PathSummaryBlob::encode(&second).unwrap();

        assert_eq!(first_blob.bytes, second_blob.bytes);
        assert_eq!(first_blob.get_handle(), second_blob.get_handle());
        assert_eq!(
            hex(first_blob.bytes.as_ref()),
            "5f73d0cf0230edf0512144e14a5e96132e661e1925556060b8036f217dc9b7f801000000020000000100000000000000010101010101010101010101010101010101010101010101010101010101010102020202020202020202020202020202020202020202020202020202020202020000000001000000"
        );

        let ordered = plus(7);
        let reversed = Automaton::new(
            2,
            [0],
            [1],
            [
                Transition::new(1, 1, Step::Forward(label(7))),
                Transition::new(0, 1, Step::Forward(label(7))),
            ],
        )
        .unwrap();
        assert_eq!(
            automaton_fingerprint(&ordered),
            automaton_fingerprint(&reversed)
        );
    }

    #[test]
    fn malformed_length_fingerprint_order_bounds_and_state_pair_are_rejected() {
        let automaton = plus(9);
        let summary = PathSummary::from_edges(automaton.clone(), [edge(1, 9, 2)]);
        let blob = PathSummaryBlob::encode(&summary).unwrap();

        let mut bad = blob.bytes.as_ref().to_vec();
        bad.pop();
        assert_eq!(
            PathSummaryBlob::decode(Blob::new(bad.into()), &automaton).unwrap_err(),
            PathSummaryBlobError::BadLength
        );

        let mut bad = blob.bytes.as_ref().to_vec();
        bad[0] ^= 1;
        assert_eq!(
            PathSummaryBlob::decode(Blob::new(bad.into()), &automaton).unwrap_err(),
            PathSummaryBlobError::DifferentAutomaton
        );

        let mut bad = blob.bytes.as_ref().to_vec();
        bad[48..80].copy_from_slice(&vertex(2));
        assert_eq!(
            PathSummaryBlob::decode(Blob::new(bad.into()), &automaton).unwrap_err(),
            PathSummaryBlobError::VertexOrder
        );

        let arc_offset = HEADER_LEN + 2 * 32;
        let mut bad = blob.bytes.as_ref().to_vec();
        bad[arc_offset + 4..arc_offset + 8].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(
            PathSummaryBlob::decode(Blob::new(bad.into()), &automaton).unwrap_err(),
            PathSummaryBlobError::ArcOutOfBounds
        );

        let mut bad = blob.bytes.as_ref().to_vec();
        bad[arc_offset + 4..arc_offset + 8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            PathSummaryBlob::decode(Blob::new(bad.into()), &automaton).unwrap_err(),
            PathSummaryBlobError::InvalidStatePair
        );

        let mut bad = blob.bytes.as_ref().to_vec();
        let first_arc = bad[arc_offset..arc_offset + 8].to_vec();
        bad[arc_offset + 8..arc_offset + 16].copy_from_slice(&first_arc);
        assert_eq!(
            PathSummaryBlob::decode(Blob::new(bad.into()), &automaton).unwrap_err(),
            PathSummaryBlobError::ArcOrder
        );
    }

    #[test]
    fn product_carrier_dimensions_are_checked_before_ordinal_lowering() {
        assert_eq!(checked_product_count(2, 3).unwrap(), 6);
        assert_eq!(
            checked_product_count(u32::MAX as usize, 2).unwrap_err(),
            PathSummaryBlobError::CapacityOverflow
        );
        assert_eq!(
            checked_product_count(usize::MAX, 2).unwrap_err(),
            PathSummaryBlobError::CapacityOverflow
        );
    }

    fn hex(bytes: &[u8]) -> String {
        const TABLE: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(TABLE[(byte >> 4) as usize] as char);
            out.push(TABLE[(byte & 0xf) as usize] as char);
        }
        out
    }
}
