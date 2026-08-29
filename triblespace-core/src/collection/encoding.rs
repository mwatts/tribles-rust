//! Canonical encodings and mappings for typed collections.
//!
//! Durable collection records carry representation-neutral content hashes.
//! This module restores their physical meaning at the API boundary without
//! inventing a second runtime planner:
//!
//! - a [`CollectionEncoding`] owns the canonical bytes, validation, and join
//!   within one collection;
//! - a [`CollectionMapping`] owns one parameterized, join-preserving
//!   conversion from a source encoding to a target encoding;
//! - [`Collection`] binds an encoding to one exact, content-addressed
//!   descriptor.
//!
//! Logical interpretation is deliberately separate in
//! [`TryFromCover`](super::TryFromCover). An interpretation may join every
//! physical member eagerly or retain an exact cover of mmap-backed shards.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::blob::{Blob, BlobEncoding, IntoBlob};
use crate::inline::encodings::hash::Handle;
use crate::inline::InlineEncoding;
use crate::metadata::MetaDescribe;
use crate::repo::{BlobStoreGet, BlobStoreMeta, BlobStorePut};
use crate::trible::Fragment;

use super::{descriptor, CollectionHandle, RecordDecodeError};

/// Failure of one exact canonical collection operation.
///
/// `Capacity` is reserved for deterministic geometry limits of the chosen
/// encoding. It must not describe transient allocation, I/O, or accelerator
/// failures. Malformed or noncanonical bytes are always `Fatal`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionOperationError {
    /// The operation or supplied bytes are invalid and another cover cannot
    /// repair it.
    Fatal(String),
    /// This exact encoding cannot hold the result, but a finer physical cover
    /// may still represent the same logical value.
    Capacity(String),
}

impl fmt::Display for CollectionOperationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(reason) | Self::Capacity(reason) => formatter.write_str(reason),
        }
    }
}

impl Error for CollectionOperationError {}

/// One fully attached physical member of a collection encoding.
///
/// Most encodings are monolithic, so their artifact is simply the root
/// [`Blob`] itself. Merkle encodings may instead retain the root, its resolved
/// dependencies, and any runtime reconstructed while validating that closure.
/// The transient attachment is deliberately not another durable record: its
/// root remains the sole collection-member identity.
pub trait CollectionArtifact<E>: Clone
where
    E: BlobEncoding,
    Handle<E>: InlineEncoding,
{
    /// Canonical root blob whose content handle names this member.
    fn root(&self) -> &Blob<E>;

    /// Publish the complete artifact closure, dependencies before root.
    ///
    /// Implementations may harmlessly re-put already resident dependencies;
    /// content addressing makes publication idempotent. Collection equations
    /// are always inserted by the caller only after every artifact has been
    /// published.
    fn publish<S>(&self, store: &mut S) -> Result<(), S::PutError>
    where
        S: BlobStorePut;
}

impl<E> CollectionArtifact<E> for Blob<E>
where
    E: BlobEncoding,
    Handle<E>: InlineEncoding,
{
    fn root(&self) -> &Blob<E> {
        self
    }

    fn publish<S>(&self, store: &mut S) -> Result<(), S::PutError>
    where
        S: BlobStorePut,
    {
        store.put::<E, _>(self.clone())?;
        Ok(())
    }
}

/// One canonical join-semilattice carried by a blob encoding.
///
/// The encoding is the physical shape *and* its intra-shape join law. This is
/// intentionally stronger than [`BlobEncoding`]: not every blob format is a
/// collection member, while every `CollectionEncoding` has one unambiguous
/// canonical join.
pub trait CollectionEncoding: BlobEncoding + MetaDescribe + Sized + 'static {
    /// Fully validated, closure-attached runtime form of one member.
    type Artifact: CollectionArtifact<Self>;

    /// Validate encoding-specific context carried by one descriptor.
    ///
    /// Most encodings need no context. An encoding whose canonical bytes are
    /// parameterized (for example a path summary over one automaton) validates
    /// only the shape information it needs here; the source-to-target mapping
    /// still owns the parameterized conversion itself.
    fn validate_descriptor(_descriptor: &Fragment) -> Result<(), CollectionOperationError> {
        Ok(())
    }

    /// Resolve and validate one member independently of its provenance.
    ///
    /// The root bytes have already passed the blob store's content-address
    /// boundary. A Merkle encoding may load children through `reader`; a
    /// monolithic encoding normally ignores it and returns the root blob.
    fn attach_member<R>(
        descriptor: &Fragment,
        member: Blob<Self>,
        reader: &R,
    ) -> Result<Self::Artifact, CollectionOperationError>
    where
        R: BlobStoreGet + BlobStoreMeta;

    /// Compute the exact canonical join of two members.
    fn join_members(
        descriptor: &Fragment,
        low: &Self::Artifact,
        high: &Self::Artifact,
    ) -> Result<Self::Artifact, CollectionOperationError>;
}

/// One parameterized mapping between collection encodings.
///
/// A concrete mapping binds the mapping fragment embedded in the target
/// descriptor, then maps physical source members. Canonical builders normally
/// derive the mapping entity id, but binding validates its algorithm and
/// parameters rather than its minting history. Implementations
/// must be a join homomorphism:
///
/// `map(a join b) = map(a) join map(b)`.
pub trait CollectionMapping<Source, Target>
where
    Source: CollectionEncoding,
    Target: CollectionEncoding,
{
    /// Bind and validate the concrete mapping named by the target descriptor.
    fn bind(source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError>
    where
        Self: Sized;

    /// Compute the canonical target image of one source member.
    fn map(&self, source: &Source::Artifact) -> Result<Target::Artifact, CollectionOperationError>;
}

/// A descriptor does not denote the encoding requested by its Rust type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionTypeError {
    /// The descriptor could not be decoded structurally.
    Malformed(RecordDecodeError),
    /// The descriptor names another canonical encoding.
    WrongEncoding {
        /// Encoding required by the Rust type.
        expected: crate::id::Id,
        /// Encoding carried by the descriptor.
        actual: crate::id::Id,
    },
    /// The descriptor names the right encoding but supplies invalid context.
    InvalidDescriptor(CollectionOperationError),
}

impl fmt::Display for CollectionTypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(source) => source.fmt(formatter),
            Self::WrongEncoding { expected, actual } => write!(
                formatter,
                "collection encoding {actual:X} does not match {expected:X}",
            ),
            Self::InvalidDescriptor(source) => {
                write!(formatter, "invalid collection encoding context: {source}")
            }
        }
    }
}

impl Error for CollectionTypeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Malformed(source) => Some(source),
            Self::InvalidDescriptor(source) => Some(source),
            Self::WrongEncoding { .. } => None,
        }
    }
}

/// Verify that one descriptor denotes `E`.
pub(crate) fn validate_descriptor_type<E>(
    descriptor_fragment: &Fragment,
) -> Result<(), CollectionTypeError>
where
    E: CollectionEncoding,
{
    descriptor::validate(descriptor_fragment.facts()).map_err(CollectionTypeError::Malformed)?;
    let actual = descriptor::representation(descriptor_fragment.facts())
        .map_err(CollectionTypeError::Malformed)?;
    let expected = E::id();
    if actual != expected {
        return Err(CollectionTypeError::WrongEncoding { expected, actual });
    }
    E::validate_descriptor(descriptor_fragment).map_err(CollectionTypeError::InvalidDescriptor)
}

/// One exact collection descriptor, typed by its canonical member encoding.
///
/// The store owns the descriptor bytes. This is only its cheap, cloneable
/// content address plus compile-time meaning; constructing it is restricted to
/// descriptor-validation boundaries.
pub struct Collection<E: CollectionEncoding> {
    handle: CollectionHandle,
    encoding: PhantomData<fn() -> E>,
}

impl<E: CollectionEncoding> Collection<E> {
    pub(crate) const fn from_handle(handle: CollectionHandle) -> Self {
        Self {
            handle,
            encoding: PhantomData,
        }
    }

    /// Validate and type one self-contained descriptor without storing it.
    ///
    /// Use [`CollectionStoreExt::collection`](super::CollectionStoreExt::collection)
    /// when the descriptor and its attachment closure should also be registered
    /// in a store.
    pub fn from_descriptor(descriptor_fragment: &Fragment) -> Result<Self, CollectionTypeError> {
        validate_descriptor_type::<E>(descriptor_fragment)?;
        let handle = IntoBlob::<crate::blob::encodings::simplearchive::SimpleArchive>::to_blob(
            descriptor_fragment,
        )
        .get_handle();
        Ok(Self::from_handle(handle))
    }

    /// Representation-neutral descriptor handle stored in dense records.
    pub const fn handle(self) -> CollectionHandle {
        self.handle
    }
}

impl<E: CollectionEncoding> Clone for Collection<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: CollectionEncoding> Copy for Collection<E> {}

impl<E: CollectionEncoding> fmt::Debug for Collection<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Collection")
            .field(&hex::encode_upper(self.handle.raw))
            .finish()
    }
}

impl<E: CollectionEncoding> PartialEq for Collection<E> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl<E: CollectionEncoding> Eq for Collection<E> {}

impl<E: CollectionEncoding> PartialOrd for Collection<E> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: CollectionEncoding> Ord for Collection<E> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.handle.cmp(&other.handle)
    }
}

impl<E: CollectionEncoding> std::hash::Hash for Collection<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

impl<E: CollectionEncoding> From<Collection<E>> for CollectionHandle {
    fn from(collection: Collection<E>) -> Self {
        collection.handle
    }
}
