//! Typed collection lattices above the representation-neutral record layer.
//!
//! Durable collection records deliberately carry bare content hashes: their
//! meaning is supplied by the descriptor they name.  This module restores that
//! meaning at the API boundary.  A [`CollectionLattice`] identifies the pair of
//! blob representation and join law, [`Collection`] binds that pair to one
//! exact descriptor handle, and higher layers may therefore expose typed
//! member handles without changing the pile or wire formats.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use crate::blob::{Blob, BlobEncoding, IntoBlob};
use crate::metadata::MetaDescribe;
use crate::trible::Fragment;

use super::{descriptor, CollectionHandle, RecordDecodeError};

/// Failure of one exact operation in a canonical collection lattice.
///
/// `Capacity` is reserved for deterministic geometry limits of the chosen
/// representation.  It must not describe transient allocation, I/O, or
/// accelerator failures.  Malformed or noncanonical bytes are always `Fatal`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionLatticeError {
    /// The operation or supplied bytes are invalid and another cover cannot
    /// repair it.
    Fatal(String),
    /// This exact representation cannot hold the result, but a finer physical
    /// cover may still represent the same logical value.
    Capacity(String),
}

impl fmt::Display for CollectionLatticeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(reason) | Self::Capacity(reason) => formatter.write_str(reason),
        }
    }
}

impl Error for CollectionLatticeError {}

/// One canonical join-semilattice carried by collection elements.
///
/// A recipe alone is not enough to identify a lattice: the SimpleArchive and
/// SuccinctArchive collections intentionally implement the same set-union law
/// using different canonical bytes.  Implementations therefore bind both an
/// [`Encoding`](Self::Encoding) and a [`Recipe`](Self::Recipe).
pub trait CollectionLattice: Sized + 'static {
    /// Canonical blob representation of one physical lattice member.
    type Encoding: BlobEncoding;

    /// Described law governing validation and join of those members.
    type Recipe: MetaDescribe;

    /// Validate recipe arguments carried by one concrete descriptor.
    ///
    /// The Rust type fixes the representation and law, while the descriptor
    /// remains the runtime value.  Recipes without arguments use this default.
    fn validate_arguments(_descriptor: &Fragment) -> Result<(), CollectionLatticeError> {
        Ok(())
    }

    /// Validate one member independently of its provenance.
    fn validate_member(
        descriptor: &Fragment,
        member: &Blob<Self::Encoding>,
    ) -> Result<(), CollectionLatticeError>;

    /// Compute the exact canonical join of two members.
    fn merge_members(
        descriptor: &Fragment,
        low: &Blob<Self::Encoding>,
        high: &Blob<Self::Encoding>,
    ) -> Result<Blob<Self::Encoding>, CollectionLatticeError>;
}

/// A canonical join homomorphism between two collection lattices.
///
/// Source and target validation and join live on the lattices themselves; the
/// mapping contributes only the operation that crosses between them.
pub trait CollectionHomomorphism<Source, Target>
where
    Source: CollectionLattice,
    Target: CollectionLattice,
{
    /// Bind runtime recipe arguments once at a typed descriptor boundary.
    fn bind(source: &Fragment, target: &Fragment) -> Result<Self, CollectionLatticeError>
    where
        Self: Sized;

    /// Compute the canonical target image of one source member.
    fn map(
        &self,
        source: &Blob<Source::Encoding>,
    ) -> Result<Blob<Target::Encoding>, CollectionLatticeError>;
}

/// A descriptor does not denote the lattice requested by its Rust type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollectionTypeError {
    /// The descriptor could not be decoded structurally.
    Malformed(RecordDecodeError),
    /// The descriptor names another element representation.
    WrongRepresentation {
        /// Representation required by the Rust lattice type.
        expected: crate::id::Id,
        /// Representation carried by the descriptor.
        actual: crate::id::Id,
    },
    /// The descriptor names another join law.
    WrongRecipe {
        /// Recipe required by the Rust lattice type.
        expected: crate::id::Id,
        /// Recipe carried by the descriptor.
        actual: crate::id::Id,
    },
    /// The descriptor names the right law but supplies invalid arguments.
    InvalidArguments(CollectionLatticeError),
}

impl fmt::Display for CollectionTypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(source) => source.fmt(formatter),
            Self::WrongRepresentation { expected, actual } => write!(
                formatter,
                "collection representation {actual:X} does not match {expected:X}",
            ),
            Self::WrongRecipe { expected, actual } => write!(
                formatter,
                "collection recipe {actual:X} does not match {expected:X}",
            ),
            Self::InvalidArguments(source) => {
                write!(formatter, "invalid collection recipe arguments: {source}")
            }
        }
    }
}

impl Error for CollectionTypeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Malformed(source) => Some(source),
            Self::InvalidArguments(source) => Some(source),
            Self::WrongRepresentation { .. } | Self::WrongRecipe { .. } => None,
        }
    }
}

/// Verify that one descriptor denotes `L`.
pub(crate) fn validate_descriptor_type<L>(
    descriptor_fragment: &Fragment,
) -> Result<(), CollectionTypeError>
where
    L: CollectionLattice,
{
    let actual_representation = descriptor::representation(descriptor_fragment.facts())
        .map_err(CollectionTypeError::Malformed)?;
    let expected_representation = <L::Encoding as MetaDescribe>::id();
    if actual_representation != expected_representation {
        return Err(CollectionTypeError::WrongRepresentation {
            expected: expected_representation,
            actual: actual_representation,
        });
    }

    let actual_recipe =
        descriptor::recipe(descriptor_fragment.facts()).map_err(CollectionTypeError::Malformed)?;
    let expected_recipe = <L::Recipe as MetaDescribe>::id();
    if actual_recipe != expected_recipe {
        return Err(CollectionTypeError::WrongRecipe {
            expected: expected_recipe,
            actual: actual_recipe,
        });
    }
    L::validate_arguments(descriptor_fragment).map_err(CollectionTypeError::InvalidArguments)?;
    Ok(())
}

/// One exact collection descriptor, typed by the lattice it denotes.
///
/// The store owns the descriptor bytes.  This is only its cheap, cloneable
/// content address plus compile-time meaning; constructing it is restricted to
/// descriptor-validation boundaries.
pub struct Collection<L: CollectionLattice> {
    handle: CollectionHandle,
    lattice: PhantomData<fn() -> L>,
}

impl<L: CollectionLattice> Collection<L> {
    pub(crate) const fn from_handle(handle: CollectionHandle) -> Self {
        Self {
            handle,
            lattice: PhantomData,
        }
    }

    /// Validate and type one self-contained descriptor without storing it.
    ///
    /// Use [`CollectionStoreExt::collection`](super::CollectionStoreExt::collection)
    /// when the descriptor and its attachment closure should also be registered
    /// in a store.
    pub fn from_descriptor(descriptor_fragment: &Fragment) -> Result<Self, CollectionTypeError> {
        validate_descriptor_type::<L>(descriptor_fragment)?;
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

impl<L: CollectionLattice> Clone for Collection<L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L: CollectionLattice> Copy for Collection<L> {}

impl<L: CollectionLattice> fmt::Debug for Collection<L> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Collection")
            .field(&hex::encode_upper(self.handle.raw))
            .finish()
    }
}

impl<L: CollectionLattice> PartialEq for Collection<L> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl<L: CollectionLattice> Eq for Collection<L> {}

impl<L: CollectionLattice> PartialOrd for Collection<L> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<L: CollectionLattice> Ord for Collection<L> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.handle.cmp(&other.handle)
    }
}

impl<L: CollectionLattice> std::hash::Hash for Collection<L> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

impl<L: CollectionLattice> From<Collection<L>> for CollectionHandle {
    fn from(collection: Collection<L>) -> Self {
        collection.handle
    }
}
