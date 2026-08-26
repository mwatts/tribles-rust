//! Monotone binding between one physical store and one network team.
//!
//! A store scope is a local safety assertion, not network inventory. An
//! unbound store may be bound once; repeating the same assertion is
//! idempotent, while observing or attempting to add a different team fails
//! closed. Backends must preserve every assertion when their physical storage
//! is concatenated or rewritten so disagreement remains observable.

use std::error::Error;
use std::fmt;

use ed25519_dalek::VerifyingKey;

/// Failure while observing or asserting a store's unique team scope.
#[derive(Debug)]
pub enum StoreScopeError<E> {
    /// The storage backend could not observe or append its scope record.
    Backend(E),
    /// The store contains assertions for two different teams.
    Conflict {
        /// Lexicographically lower conflicting team key.
        first: VerifyingKey,
        /// Lexicographically higher conflicting team key.
        second: VerifyingKey,
    },
}

impl<E> StoreScopeError<E> {
    /// Construct a canonical conflict independent of observation order.
    pub fn conflict(a: VerifyingKey, b: VerifyingKey) -> Self {
        let (first, second) = if a.as_bytes() <= b.as_bytes() {
            (a, b)
        } else {
            (b, a)
        };
        Self::Conflict { first, second }
    }
}

impl<E: fmt::Display> fmt::Display for StoreScopeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(f, "failed to access store scope: {error}"),
            Self::Conflict { first, second } => write!(
                f,
                "store is bound to conflicting teams {} and {}",
                hex::encode_upper(first.as_bytes()),
                hex::encode_upper(second.as_bytes()),
            ),
        }
    }
}

impl<E: Error + 'static> Error for StoreScopeError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Conflict { .. } => None,
        }
    }
}

/// Storage capable of a single monotone team-scope assertion.
///
/// This trait deliberately exposes no rebind or removal operation. The scope
/// protects local network assembly and is neither authorization nor gossiped
/// inventory.
pub trait StoreScope {
    /// Backend error produced while observing or appending an assertion.
    type ScopeError: Error + Send + Sync + 'static;

    /// Observe the unique team key, `None` when unbound, or fail on conflict.
    fn store_scope(&mut self) -> Result<Option<VerifyingKey>, StoreScopeError<Self::ScopeError>>;

    /// Assert that this store belongs to `team`.
    ///
    /// The first assertion binds an unbound store. The same assertion is an
    /// idempotent no-op; a different assertion fails without replacing either
    /// value.
    fn bind_store_scope(
        &mut self,
        team: VerifyingKey,
    ) -> Result<(), StoreScopeError<Self::ScopeError>>;
}

impl<S> StoreScope for &mut S
where
    S: StoreScope + ?Sized,
{
    type ScopeError = S::ScopeError;

    fn store_scope(&mut self) -> Result<Option<VerifyingKey>, StoreScopeError<Self::ScopeError>> {
        (**self).store_scope()
    }

    fn bind_store_scope(
        &mut self,
        team: VerifyingKey,
    ) -> Result<(), StoreScopeError<Self::ScopeError>> {
        (**self).bind_store_scope(team)
    }
}
