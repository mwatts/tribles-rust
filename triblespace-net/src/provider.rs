//! Bounded, receiver-local hints for locating already-known immutable artifacts.
//!
//! This is not an artifact-discovery index: callers must already know the
//! [`ArtifactId`] they ask about. It answers only "who recently said they can
//! provide this exact artifact?" Exact transfer and validation remain separate
//! operations. Replica placement uses the network's bounded iterative XOR
//! routing; temporary partitions and stale leases remain soft unknowns, never
//! fabricated answers.

use std::collections::BTreeMap;
use std::time::Duration;

use ed25519_dalek::VerifyingKey;

use crate::clock::Mono;
use crate::transport::PeerId;

/// Bare physical identity of one data artifact.
pub type ArtifactId = [u8; 32];

/// Opaque team-scoped rendezvous key used to select provider-directory replicas.
pub(crate) type ProviderKey = [u8; 32];

/// Receiver-chosen lifetime of one provider hint.
pub(crate) const PROVIDER_LEASE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
/// Maximum fan-out returned for one artifact.
pub(crate) const MAX_PROVIDERS_PER_KEY: usize = 64;
const _: () = assert!(MAX_PROVIDERS_PER_KEY <= u8::MAX as usize);
/// Hard bound on all logical provider entries held by one process.
const MAX_PROVIDER_ENTRIES: usize = 65_536;

/// Derive a lookup key without exposing cross-team content overlap.
pub(crate) fn provider_key(team: VerifyingKey, artifact: ArtifactId) -> ProviderKey {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/provider-key/v1");
    hasher.update(team.as_bytes());
    hasher.update(&artifact);
    *hasher.finalize().as_bytes()
}

/// Soft provider state. Leases are deliberately local to the receiver: a
/// publisher cannot choose an absurd expiry or make timestamp-distinct values
/// accumulate forever.
pub(crate) struct ProviderDirectory {
    providers: BTreeMap<ProviderKey, BTreeMap<PeerId, Mono>>,
    provider_count: usize,
    limits: DirectoryLimits,
}

#[derive(Clone, Copy)]
struct DirectoryLimits {
    lease: Duration,
    per_key: usize,
    total: usize,
}

impl Default for ProviderDirectory {
    fn default() -> Self {
        Self {
            providers: BTreeMap::new(),
            provider_count: 0,
            limits: DirectoryLimits {
                lease: PROVIDER_LEASE_LIFETIME,
                per_key: MAX_PROVIDERS_PER_KEY,
                total: MAX_PROVIDER_ENTRIES,
            },
        }
    }
}

impl ProviderDirectory {
    /// Record or renew the authenticated caller as a provider.
    ///
    /// There is intentionally no claimed-provider argument in the wire
    /// protocol. `provider` comes from the authenticated connection identity.
    pub(crate) fn put(&mut self, key: ProviderKey, provider: PeerId, now: Mono) -> bool {
        self.prune_key(&key, now);

        if let Some(expires_at) = self
            .providers
            .get_mut(&key)
            .and_then(|providers| providers.get_mut(&provider))
        {
            *expires_at = now + self.limits.lease;
            return true;
        }

        if self
            .providers
            .get(&key)
            .is_some_and(|providers| providers.len() >= self.limits.per_key)
        {
            return false;
        }

        if self.provider_count >= self.limits.total {
            // Ordinary inserts and renewals touch one bounded bucket. The only
            // O(total) path runs at the hard capacity where it may free room.
            self.prune_expired(now);
            if self.provider_count >= self.limits.total {
                return false;
            }
        }

        self.providers
            .entry(key)
            .or_default()
            .insert(provider, now + self.limits.lease);
        self.provider_count += 1;
        true
    }

    /// Return all live hints in deterministic peer-id order.
    pub(crate) fn get(&mut self, key: ProviderKey, now: Mono) -> Vec<PeerId> {
        self.prune_key(&key, now);
        self.providers
            .get(&key)
            .map(|providers| providers.keys().copied().collect())
            .unwrap_or_default()
    }

    fn prune_key(&mut self, key: &ProviderKey, now: Mono) {
        let Some(providers) = self.providers.get_mut(key) else {
            return;
        };
        let before = providers.len();
        providers.retain(|_, expires_at| *expires_at > now);
        self.provider_count -= before - providers.len();
        if providers.is_empty() {
            self.providers.remove(key);
        }
    }

    fn prune_expired(&mut self, now: Mono) {
        self.providers.retain(|_, providers| {
            providers.retain(|_, expires_at| *expires_at > now);
            !providers.is_empty()
        });
        self.provider_count = self.providers.values().map(BTreeMap::len).sum();
    }

    #[cfg(test)]
    fn with_limits(lease: Duration, per_key: usize, total: usize) -> Self {
        Self {
            providers: BTreeMap::new(),
            provider_count: 0,
            limits: DirectoryLimits {
                lease,
                per_key,
                total,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn keys_are_deterministic_and_team_scoped() {
        let team_a = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let team_b = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let artifact = [3; 32];
        assert_eq!(
            provider_key(team_a, artifact),
            provider_key(team_a, artifact)
        );
        assert_ne!(
            provider_key(team_a, artifact),
            provider_key(team_b, artifact)
        );
        assert_ne!(
            provider_key(team_a, artifact),
            provider_key(team_a, [4; 32])
        );
    }

    #[test]
    fn one_teams_artifact_hint_is_invisible_under_another_team() {
        let now = crate::clock::mono_now();
        let team_a = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let team_b = SigningKey::from_bytes(&[2; 32]).verifying_key();
        let artifact = [3; 32];
        let provider = SigningKey::from_bytes(&[4; 32]).verifying_key().to_bytes();
        let mut directory = ProviderDirectory::default();

        assert!(directory.put(provider_key(team_a, artifact), provider, now));
        assert_eq!(
            directory.get(provider_key(team_a, artifact), now),
            vec![provider]
        );
        assert!(
            directory
                .get(provider_key(team_b, artifact), now)
                .is_empty()
        );
    }

    #[test]
    fn leases_expire_and_renew_without_duplicate_state() {
        let now = crate::clock::mono_now();
        let lease = Duration::from_secs(10);
        let mut directory = ProviderDirectory::with_limits(lease, 2, 2);
        let key = [1; 32];
        let provider = [2; 32];

        assert!(directory.put(key, provider, now));
        assert!(directory.put(key, provider, now + Duration::from_secs(5)));
        assert_eq!(directory.provider_count, 1);
        assert_eq!(
            directory.get(key, now + Duration::from_secs(14)),
            vec![provider]
        );
        assert!(directory.get(key, now + Duration::from_secs(15)).is_empty());
        assert_eq!(directory.provider_count, 0);
    }

    #[test]
    fn per_key_and_global_bounds_are_hard() {
        let now = crate::clock::mono_now();
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(10), 2, 3);

        assert!(directory.put([1; 32], [1; 32], now));
        assert!(directory.put([1; 32], [2; 32], now));
        assert!(!directory.put([1; 32], [3; 32], now));
        assert!(directory.put([2; 32], [3; 32], now));
        assert!(!directory.put([3; 32], [4; 32], now));
        assert_eq!(directory.provider_count, 3);

        // An expired entry is reclaimed on the otherwise-full insertion path.
        assert!(directory.put([3; 32], [4; 32], now + Duration::from_secs(10)));
        assert_eq!(directory.provider_count, 1);
    }
}
