//! Canonical, addressable byte form of BLAKE3 PATCH Merkle nodes.

use core::fmt;

const ALIGNMENT: usize = 32;
const HEADER_LEN: usize = ALIGNMENT;
const BRANCH_METADATA_LEN: usize = ALIGNMENT;
const CHILD_LEN: usize = 2 * ALIGNMENT;
const MAGIC: [u8; 24] = *b"triblespace.patch.node\0\0";
const VERSION: u16 = 3;
const LEAF_KIND: u8 = 0;
const BRANCH_KIND: u8 = 1;

/// Alignment of every field that may contain a child BLAKE3 digest.
pub const BLAKE3_MERKLE_NODE_ALIGNMENT: usize = ALIGNMENT;

/// Version encoded by [`Blake3MerkleNodeBlob`].
pub const BLAKE3_MERKLE_NODE_VERSION: u16 = VERSION;

/// One canonical child descriptor decoded from a branch node blob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Blake3MerkleNodeChild {
    edge: u8,
    leaf_count: u64,
    digest: [u8; 32],
}

impl Blake3MerkleNodeChild {
    /// Edge byte selecting this child from its parent branch.
    pub const fn edge(&self) -> u8 {
        self.edge
    }

    /// Exact number of leaves below this child.
    pub const fn leaf_count(&self) -> u64 {
        self.leaf_count
    }

    /// BLAKE3 address of the child's canonical node blob.
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

/// Borrowed, strictly validated byte form of one BLAKE3 PATCH Merkle node.
///
/// The encoded bytes are themselves the node preimage: ordinary
/// `blake3::hash(node.canonical_blob_bytes())` is its PATCH digest. Every
/// embedded child digest starts at a 32-byte boundary, so generic blob graph
/// traversal can discover child nodes without understanding PATCH.
///
/// Validation allocates nothing. Accessors borrow exact fields from the input,
/// and [`Self::children`] parses the fixed-width descriptors lazily.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Blake3MerkleNodeBlob<'a, const KEY_LEN: usize> {
    bytes: &'a [u8],
}

impl<'a, const KEY_LEN: usize> Blake3MerkleNodeBlob<'a, KEY_LEN> {
    /// Decode exactly one canonical version-3 Merkle node blob.
    ///
    /// Alternate spellings are rejected: lengths must be exact, all reserved
    /// and padding bytes must be zero, branch edges must be strictly ascending,
    /// and child counts must add up to the declared subtree count.
    pub fn decode(bytes: &'a [u8]) -> Result<Self, Blake3MerkleNodeDecodeError> {
        if bytes.len() < HEADER_LEN {
            return Err(Blake3MerkleNodeDecodeError::InvalidLength {
                expected: HEADER_LEN,
                actual: bytes.len(),
            });
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(Blake3MerkleNodeDecodeError::InvalidMagic);
        }

        let version = u16::from_le_bytes(bytes[24..26].try_into().expect("two-byte version"));
        if version != VERSION {
            return Err(Blake3MerkleNodeDecodeError::UnsupportedVersion(version));
        }
        let kind = bytes[26];
        if bytes[27] != 0 {
            return Err(Blake3MerkleNodeDecodeError::NonZeroReserved);
        }
        let encoded_key_len =
            u32::from_le_bytes(bytes[28..32].try_into().expect("four-byte key length"));
        if encoded_key_len as u64 != KEY_LEN as u64 {
            return Err(Blake3MerkleNodeDecodeError::KeyLengthMismatch {
                expected: KEY_LEN,
                encoded: encoded_key_len,
            });
        }

        match kind {
            LEAF_KIND => Self::validate_leaf(bytes)?,
            BRANCH_KIND => Self::validate_branch(bytes)?,
            other => return Err(Blake3MerkleNodeDecodeError::InvalidKind(other)),
        }
        Ok(Self { bytes })
    }

    fn validate_leaf(bytes: &[u8]) -> Result<(), Blake3MerkleNodeDecodeError> {
        let expected =
            leaf_encoded_len(KEY_LEN).ok_or(Blake3MerkleNodeDecodeError::LengthOverflow)?;
        require_exact_len(bytes, expected)?;

        let key_end = HEADER_LEN + KEY_LEN;
        if bytes[key_end..].iter().any(|byte| *byte != 0) {
            return Err(Blake3MerkleNodeDecodeError::NonZeroPadding);
        }
        Ok(())
    }

    fn validate_branch(bytes: &[u8]) -> Result<(), Blake3MerkleNodeDecodeError> {
        if bytes.len() < HEADER_LEN + BRANCH_METADATA_LEN {
            return Err(Blake3MerkleNodeDecodeError::InvalidLength {
                expected: HEADER_LEN + BRANCH_METADATA_LEN,
                actual: bytes.len(),
            });
        }
        let metadata = &bytes[HEADER_LEN..HEADER_LEN + BRANCH_METADATA_LEN];
        let end_depth =
            u32::from_le_bytes(metadata[..4].try_into().expect("four-byte prefix length"));
        if end_depth as u64 >= KEY_LEN as u64 {
            return Err(Blake3MerkleNodeDecodeError::InvalidPrefixLength {
                key_len: KEY_LEN,
                prefix_len: end_depth,
            });
        }
        let child_count =
            u32::from_le_bytes(metadata[4..8].try_into().expect("four-byte child count"));
        if !(2..=256).contains(&child_count) {
            return Err(Blake3MerkleNodeDecodeError::InvalidChildCount(child_count));
        }
        let leaf_count =
            u64::from_le_bytes(metadata[8..16].try_into().expect("eight-byte leaf count"));
        if metadata[16..].iter().any(|byte| *byte != 0) {
            return Err(Blake3MerkleNodeDecodeError::NonZeroReserved);
        }

        let prefix_len = end_depth as usize;
        let expected = branch_encoded_len(prefix_len, child_count as usize)
            .ok_or(Blake3MerkleNodeDecodeError::LengthOverflow)?;
        require_exact_len(bytes, expected)?;

        let prefix_start = HEADER_LEN + BRANCH_METADATA_LEN;
        let prefix_padded_len =
            aligned_len(prefix_len).ok_or(Blake3MerkleNodeDecodeError::LengthOverflow)?;
        let prefix_end = prefix_start + prefix_len;
        let children_start = prefix_start + prefix_padded_len;
        if bytes[prefix_end..children_start]
            .iter()
            .any(|byte| *byte != 0)
        {
            return Err(Blake3MerkleNodeDecodeError::NonZeroPadding);
        }

        let mut previous_edge = None;
        let mut child_leaf_sum = 0u64;
        for index in 0..child_count as usize {
            let start = children_start + index * CHILD_LEN;
            let descriptor = &bytes[start..start + ALIGNMENT];
            if descriptor[1..8].iter().any(|byte| *byte != 0)
                || descriptor[16..].iter().any(|byte| *byte != 0)
            {
                return Err(Blake3MerkleNodeDecodeError::NonZeroReserved);
            }

            let edge = descriptor[0];
            if let Some(previous) = previous_edge {
                if edge <= previous {
                    return Err(Blake3MerkleNodeDecodeError::ChildEdgesOutOfOrder {
                        previous,
                        current: edge,
                    });
                }
            }
            previous_edge = Some(edge);

            let child_leaf_count = u64::from_le_bytes(
                descriptor[8..16]
                    .try_into()
                    .expect("eight-byte child leaf count"),
            );
            if child_leaf_count == 0 {
                return Err(Blake3MerkleNodeDecodeError::ZeroChildLeafCount { index });
            }
            child_leaf_sum = child_leaf_sum
                .checked_add(child_leaf_count)
                .ok_or(Blake3MerkleNodeDecodeError::LeafCountOverflow)?;
        }
        if child_leaf_sum != leaf_count {
            return Err(Blake3MerkleNodeDecodeError::LeafCountMismatch {
                declared: leaf_count,
                children: child_leaf_sum,
            });
        }

        Ok(())
    }

    /// Return the exact validated bytes, which are already canonical.
    pub const fn canonical_blob_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// BLAKE3 content address of [`Self::canonical_blob_bytes`].
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(self.bytes).as_bytes()
    }

    /// Whether this node represents one complete key.
    pub fn is_leaf(&self) -> bool {
        self.bytes[26] == LEAF_KIND
    }

    /// Complete key for a leaf, or `None` for a branch.
    pub fn key(&self) -> Option<&'a [u8; KEY_LEN]> {
        self.is_leaf().then(|| {
            self.bytes[HEADER_LEN..HEADER_LEN + KEY_LEN]
                .try_into()
                .expect("validated leaf key width")
        })
    }

    /// Compressed tree-order prefix for a branch, or the full key for a leaf.
    pub fn prefix(&self) -> &'a [u8] {
        if self.is_leaf() {
            return &self.bytes[HEADER_LEN..HEADER_LEN + KEY_LEN];
        }
        let prefix_len = read_u32(&self.bytes[HEADER_LEN..HEADER_LEN + 4]) as usize;
        let start = HEADER_LEN + BRANCH_METADATA_LEN;
        &self.bytes[start..start + prefix_len]
    }

    /// Exact number of leaves below this node.
    pub fn leaf_count(&self) -> u64 {
        if self.is_leaf() {
            1
        } else {
            read_u64(&self.bytes[HEADER_LEN + 8..HEADER_LEN + 16])
        }
    }

    /// Lazily parse ascending branch children; a leaf returns an empty iterator.
    pub fn children(
        &self,
    ) -> impl ExactSizeIterator<Item = Blake3MerkleNodeChild>
           + DoubleEndedIterator
           + std::iter::FusedIterator
           + 'a {
        let children = if self.is_leaf() {
            &self.bytes[..0]
        } else {
            let prefix_len = read_u32(&self.bytes[HEADER_LEN..HEADER_LEN + 4]) as usize;
            let start = HEADER_LEN
                + BRANCH_METADATA_LEN
                + aligned_len(prefix_len).expect("validated prefix length");
            &self.bytes[start..]
        };
        children.chunks_exact(CHILD_LEN).map(decode_child)
    }
}

impl<const KEY_LEN: usize> fmt::Debug for Blake3MerkleNodeBlob<'_, KEY_LEN> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Blake3MerkleNodeBlob")
            .field("prefix", &self.prefix())
            .field("digest", &self.digest())
            .field("leaf_count", &self.leaf_count())
            .field("child_count", &self.children().len())
            .finish()
    }
}

fn decode_child(bytes: &[u8]) -> Blake3MerkleNodeChild {
    Blake3MerkleNodeChild {
        edge: bytes[0],
        leaf_count: read_u64(&bytes[8..16]),
        digest: bytes[ALIGNMENT..CHILD_LEN]
            .try_into()
            .expect("validated child digest width"),
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("validated four-byte integer"))
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("validated eight-byte integer"))
}

/// Why bytes failed strict version-3 PATCH Merkle-node decoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Blake3MerkleNodeDecodeError {
    /// The blob did not have the one exact length implied by its fields.
    InvalidLength { expected: usize, actual: usize },
    /// Length arithmetic overflowed the host address space.
    LengthOverflow,
    /// The common PATCH-node magic did not match.
    InvalidMagic,
    /// The version is not supported by this decoder.
    UnsupportedVersion(u16),
    /// The node-kind byte was neither leaf nor branch.
    InvalidKind(u8),
    /// The encoded key width did not match the decoder's const generic.
    KeyLengthMismatch { expected: usize, encoded: u32 },
    /// A branch prefix was not strictly shorter than a complete key.
    InvalidPrefixLength { key_len: usize, prefix_len: u32 },
    /// A branch did not contain between two and 256 children.
    InvalidChildCount(u32),
    /// Reserved bytes that canonical encoders set to zero were nonzero.
    NonZeroReserved,
    /// Alignment padding that canonical encoders set to zero was nonzero.
    NonZeroPadding,
    /// Branch edge descriptors were not strictly ascending.
    ChildEdgesOutOfOrder { previous: u8, current: u8 },
    /// A child descriptor claimed to contain no leaves.
    ZeroChildLeafCount { index: usize },
    /// Summing child leaf counts overflowed `u64`.
    LeafCountOverflow,
    /// Child leaf counts did not sum to the branch's declared count.
    LeafCountMismatch { declared: u64, children: u64 },
}

impl fmt::Display for Blake3MerkleNodeDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid canonical BLAKE3 PATCH Merkle node: {self:?}"
        )
    }
}

impl std::error::Error for Blake3MerkleNodeDecodeError {}

pub(super) trait CanonicalSink {
    fn write(&mut self, bytes: &[u8]);
}

impl CanonicalSink for blake3::Hasher {
    fn write(&mut self, bytes: &[u8]) {
        self.update(bytes);
    }
}

impl CanonicalSink for Vec<u8> {
    fn write(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

pub(super) fn encode_leaf(sink: &mut impl CanonicalSink, key: &[u8]) {
    encode_header(sink, LEAF_KIND, key.len());
    sink.write(key);
    write_zero_padding(sink, key.len());
}

pub(super) fn encode_branch_start(
    sink: &mut impl CanonicalSink,
    key_len: usize,
    end_depth: usize,
    child_count: usize,
    leaf_count: u64,
    prefix: impl IntoIterator<Item = u8>,
) {
    assert!(end_depth < key_len, "a PATCH branch prefix must be partial");
    // Branch editors recompute their aggregate before the caller collapses or
    // drops a transient zero- or one-child physical branch. Those bytes never
    // become an addressable logical node; the strict public decoder below
    // admits only the canonical 2..=256-child shape.
    assert!(
        child_count <= 256,
        "a PATCH branch has at most 256 children"
    );
    encode_header(sink, BRANCH_KIND, key_len);

    let mut metadata = [0u8; BRANCH_METADATA_LEN];
    metadata[..4].copy_from_slice(&to_u32(end_depth, "PATCH prefix length").to_le_bytes());
    metadata[4..8].copy_from_slice(&to_u32(child_count, "PATCH child count").to_le_bytes());
    metadata[8..16].copy_from_slice(&leaf_count.to_le_bytes());
    sink.write(&metadata);

    let mut block = [0u8; ALIGNMENT];
    let mut written = 0usize;
    for byte in prefix {
        assert!(written < end_depth, "PATCH prefix iterator was too long");
        block[written % ALIGNMENT] = byte;
        written += 1;
        if written % ALIGNMENT == 0 {
            sink.write(&block);
            block.fill(0);
        }
    }
    assert_eq!(written, end_depth, "PATCH prefix iterator was too short");
    if written % ALIGNMENT != 0 {
        sink.write(&block);
    }
}

pub(super) fn encode_child(
    sink: &mut impl CanonicalSink,
    edge: u8,
    leaf_count: u64,
    digest: [u8; 32],
) {
    assert!(leaf_count > 0, "a PATCH child contains at least one leaf");
    let mut descriptor = [0u8; ALIGNMENT];
    descriptor[0] = edge;
    descriptor[8..16].copy_from_slice(&leaf_count.to_le_bytes());
    sink.write(&descriptor);
    sink.write(&digest);
}

pub(super) fn leaf_encoded_len(key_len: usize) -> Option<usize> {
    HEADER_LEN.checked_add(aligned_len(key_len)?)
}

pub(super) fn branch_encoded_len(prefix_len: usize, child_count: usize) -> Option<usize> {
    HEADER_LEN
        .checked_add(BRANCH_METADATA_LEN)?
        .checked_add(aligned_len(prefix_len)?)?
        .checked_add(child_count.checked_mul(CHILD_LEN)?)
}

fn encode_header(sink: &mut impl CanonicalSink, kind: u8, key_len: usize) {
    let mut header = [0u8; HEADER_LEN];
    header[..MAGIC.len()].copy_from_slice(&MAGIC);
    header[24..26].copy_from_slice(&VERSION.to_le_bytes());
    header[26] = kind;
    header[28..32].copy_from_slice(&to_u32(key_len, "PATCH key width").to_le_bytes());
    sink.write(&header);
}

fn write_zero_padding(sink: &mut impl CanonicalSink, unaligned_len: usize) {
    let padding = padding_len(unaligned_len);
    if padding != 0 {
        sink.write(&[0; ALIGNMENT][..padding]);
    }
}

fn aligned_len(len: usize) -> Option<usize> {
    len.checked_add(padding_len(len))
}

const fn padding_len(len: usize) -> usize {
    (ALIGNMENT - len % ALIGNMENT) % ALIGNMENT
}

fn to_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{what} does not fit the v3 wire format"))
}

fn require_exact_len(bytes: &[u8], expected: usize) -> Result<(), Blake3MerkleNodeDecodeError> {
    if bytes.len() == expected {
        Ok(())
    } else {
        Err(Blake3MerkleNodeDecodeError::InvalidLength {
            expected,
            actual: bytes.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf<const KEY_LEN: usize>(key: [u8; KEY_LEN]) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_leaf(&mut bytes, &key);
        bytes
    }

    fn branch() -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_branch_start(&mut bytes, 16, 3, 2, 5, [1, 2, 3]);
        encode_child(&mut bytes, 7, 2, [11; 32]);
        encode_child(&mut bytes, 200, 3, [22; 32]);
        bytes
    }

    #[test]
    fn leaves_round_trip_across_inventory_key_widths() {
        fn check<const KEY_LEN: usize>(key: [u8; KEY_LEN]) {
            let bytes = leaf(key);
            assert_eq!(bytes.len() % ALIGNMENT, 0);
            let decoded = Blake3MerkleNodeBlob::<KEY_LEN>::decode(&bytes).unwrap();
            assert!(decoded.is_leaf());
            assert_eq!(decoded.key(), Some(&key));
            assert_eq!(decoded.leaf_count(), 1);
            assert_eq!(decoded.canonical_blob_bytes(), bytes);
            assert!(core::ptr::eq(
                decoded.canonical_blob_bytes().as_ptr(),
                bytes.as_ptr()
            ));
            assert_eq!(decoded.digest(), *blake3::hash(&bytes).as_bytes());
        }

        check([0x11; 16]);
        check([0x22; 32]);
        check([0x33; 64]);
    }

    #[test]
    fn branch_round_trips_and_places_child_hashes_on_aligned_boundaries() {
        let bytes = branch();
        let decoded = Blake3MerkleNodeBlob::<16>::decode(&bytes).unwrap();
        assert!(!decoded.is_leaf());
        assert_eq!(decoded.prefix(), [1, 2, 3]);
        assert_eq!(decoded.leaf_count(), 5);
        let children = decoded.children().collect::<Vec<_>>();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].edge(), 7);
        assert_eq!(children[1].edge(), 200);
        assert_eq!(decoded.canonical_blob_bytes(), bytes);

        let children_start = HEADER_LEN + BRANCH_METADATA_LEN + ALIGNMENT;
        assert_eq!(&bytes[children_start + 32..children_start + 64], &[11; 32]);
        assert_eq!(&bytes[children_start + 96..children_start + 128], &[22; 32]);
        assert_eq!((children_start + 32) % ALIGNMENT, 0);
        assert_eq!((children_start + 96) % ALIGNMENT, 0);
    }

    #[test]
    fn decoder_rejects_noncanonical_padding_order_and_counts() {
        let mut leaf_padding = leaf([7u8; 16]);
        *leaf_padding.last_mut().unwrap() = 1;
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&leaf_padding),
            Err(Blake3MerkleNodeDecodeError::NonZeroPadding)
        );

        let canonical = branch();
        let children_start = HEADER_LEN + BRANCH_METADATA_LEN + ALIGNMENT;

        let mut prefix_padding = canonical.clone();
        prefix_padding[HEADER_LEN + BRANCH_METADATA_LEN + 3] = 1;
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&prefix_padding),
            Err(Blake3MerkleNodeDecodeError::NonZeroPadding)
        );

        let mut descriptor_padding = canonical.clone();
        descriptor_padding[children_start + 1] = 1;
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&descriptor_padding),
            Err(Blake3MerkleNodeDecodeError::NonZeroReserved)
        );

        let mut unordered = canonical.clone();
        unordered[children_start + CHILD_LEN] = 7;
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&unordered),
            Err(Blake3MerkleNodeDecodeError::ChildEdgesOutOfOrder {
                previous: 7,
                current: 7,
            })
        );

        let mut zero_child = canonical.clone();
        zero_child[children_start + 8..children_start + 16].fill(0);
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&zero_child),
            Err(Blake3MerkleNodeDecodeError::ZeroChildLeafCount { index: 0 })
        );

        let mut mismatch = canonical.clone();
        mismatch[HEADER_LEN + 8..HEADER_LEN + 16].copy_from_slice(&6u64.to_le_bytes());
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&mismatch),
            Err(Blake3MerkleNodeDecodeError::LeafCountMismatch {
                declared: 6,
                children: 5,
            })
        );

        let mut overflow = canonical;
        overflow[children_start + 8..children_start + 16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&overflow),
            Err(Blake3MerkleNodeDecodeError::LeafCountOverflow)
        );
    }

    #[test]
    fn decoder_rejects_wrong_header_shape_and_exact_lengths() {
        let canonical = branch();

        let mut wrong_version = canonical.clone();
        wrong_version[24..26].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&wrong_version),
            Err(Blake3MerkleNodeDecodeError::UnsupportedVersion(2))
        );

        let mut wrong_key_len = canonical.clone();
        wrong_key_len[28..32].copy_from_slice(&32u32.to_le_bytes());
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&wrong_key_len),
            Err(Blake3MerkleNodeDecodeError::KeyLengthMismatch {
                expected: 16,
                encoded: 32,
            })
        );

        let mut invalid_kind = canonical.clone();
        invalid_kind[26] = 9;
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&invalid_kind),
            Err(Blake3MerkleNodeDecodeError::InvalidKind(9))
        );

        let mut invalid_depth = canonical.clone();
        invalid_depth[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&16u32.to_le_bytes());
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&invalid_depth),
            Err(Blake3MerkleNodeDecodeError::InvalidPrefixLength {
                key_len: 16,
                prefix_len: 16,
            })
        );

        let mut invalid_count = canonical.clone();
        invalid_count[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            Blake3MerkleNodeBlob::<16>::decode(&invalid_count),
            Err(Blake3MerkleNodeDecodeError::InvalidChildCount(1))
        );

        let mut trailing = canonical;
        trailing.push(0);
        assert!(matches!(
            Blake3MerkleNodeBlob::<16>::decode(&trailing),
            Err(Blake3MerkleNodeDecodeError::InvalidLength { .. })
        ));
    }
}
