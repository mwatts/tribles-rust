//! Local scheduling policy for collection-scoped repair.
//!
//! The former global team inventory and blob-mirroring policy were removed at
//! the collection-host cutover. Collection authority is carried by each
//! repair request, while exact blob acquisition remains demand-driven.

/// Local direction policy for periodic collection repair.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReconcileDirection {
    /// Pull explicitly active collections and serve them to admitted readers.
    #[default]
    Bidirectional,
    /// Pull active collections without serving local collection state or data.
    ReadOnly,
    /// Serve active collections and public exact data without initiating repair.
    WriteOnly,
}

impl ReconcileDirection {
    /// Whether the local scheduler initiates repair pulls.
    pub const fn pulls(self) -> bool {
        !matches!(self, Self::WriteOnly)
    }

    /// Whether inbound admitted readers may receive collection state and exact
    /// public/bearer reads may be served.
    pub const fn serves(self) -> bool {
        !matches!(self, Self::ReadOnly)
    }
}

/// Local-only collection repair policy.
///
/// This value is never sent as authority and cannot widen a collection's READ
/// policy or disclosure boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReconcileQos {
    /// Whether this peer pulls, serves, or does both.
    pub direction: ReconcileDirection,
}
