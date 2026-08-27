//! Receiver-local provider covers for already-known immutable artifacts.
//!
//! One provider publishes at most one immutable shard for each first byte of
//! its team-scoped rendezvous keys. Shards are soft state: a matching Merkle
//! root renews one receiver-local lease, a changed root atomically replaces the
//! old shard, and an omitted prefix disappears when its old lease expires.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use ed25519_dalek::VerifyingKey;
use triblespace_core::patch::{Blake3Merkle, IdentitySchema, PATCH};

use crate::clock::Mono;
use crate::transport::PeerId;

/// Bare physical identity of one data artifact.
pub type ArtifactId = [u8; 32];

/// Opaque team-scoped rendezvous key for exact provider membership.
pub(crate) type ProviderKey = [u8; 32];
/// DHT target shared by every artifact in one provider-cover prefix.
pub(crate) type ProviderPrefixKey = [u8; 32];

/// Receiver-chosen lifetime of one provider-cover shard.
pub(crate) const PROVIDER_LEASE_LIFETIME: Duration = Duration::from_secs(24 * 60 * 60);
/// Maximum fan-out returned for one exact artifact lookup.
pub(crate) const MAX_PROVIDERS_PER_KEY: usize = 64;
const _: () = assert!(MAX_PROVIDERS_PER_KEY <= u8::MAX as usize);

/// A single shard body is bounded near 2 MiB. Uniform team-scoped provider
/// keys spread across 256 prefixes, so the mean shard reaches this limit only
/// around 16.7 million active offers; pathological skew is omitted explicitly.
pub(crate) const MAX_PROVIDER_SHARD_MEMBERS: usize = 1 << 16;
const MAX_PROVIDER_SHARDS: usize = 65_536;
const MAX_PROVIDER_MEMBERS: usize = 1 << 24;
/// Fair receiver-local share across the prefix shards this directory happens
/// to hold for one provider. This is not a publisher-cover ceiling.
const MAX_PROVIDER_MEMBERS_PER_PROVIDER: usize = 1 << 20;
/// Bound work performed by one exact directory lookup. A soft directory may
/// return fewer hints rather than monopolize an async worker on an adversarially
/// dense prefix; the per-prefix cursor makes repeated lookups cover every
/// candidate eventually.
const MAX_PROVIDER_SHARDS_SCANNED_PER_GET: usize = 256;
/// Bound opportunistic expiry reclamation performed by one RPC. Expired shards
/// which remain queued are filtered by their deadline and conservatively retain
/// capacity until a later call reclaims them.
const MAX_EXPIRED_PROVIDER_SHARDS_PER_CALL: usize = 64;

type ProviderPatch = PATCH<32, IdentitySchema, (), Blake3Merkle>;

/// Derive an exact lookup key without exposing cross-team content overlap.
pub(crate) fn provider_key(team: VerifyingKey, artifact: ArtifactId) -> ProviderKey {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/provider-key/v1");
    hasher.update(team.as_bytes());
    hasher.update(&artifact);
    *hasher.finalize().as_bytes()
}

/// Derive the DHT target for one fixed first-byte shard.
pub(crate) fn provider_prefix_key(team: VerifyingKey, prefix: u8) -> ProviderPrefixKey {
    let mut hasher = blake3::Hasher::new_derive_key("triblespace.net/provider-prefix/v1");
    hasher.update(team.as_bytes());
    hasher.update(&[prefix]);
    *hasher.finalize().as_bytes()
}

/// One canonical nonempty provider-cover shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProviderShard {
    prefix: u8,
    digest: [u8; 32],
    keys: Arc<[ProviderKey]>,
}

impl ProviderShard {
    pub(crate) fn prefix(&self) -> u8 {
        self.prefix
    }

    pub(crate) fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub(crate) fn count(&self) -> u32 {
        u32::try_from(self.keys.len()).expect("provider shard has a static u32 bound")
    }

    pub(crate) fn keys(&self) -> &[ProviderKey] {
        &self.keys
    }
}

/// Canonical publishable cover of a provider's active
/// `OFFER ∩ resident ∩ serving` artifacts.
#[derive(Clone, Default)]
pub(crate) struct ProviderCover {
    shards: BTreeMap<u8, ProviderShard>,
}

/// One active prefix which could not be represented within the bounded wire
/// body. It is diagnostic only and does not participate in cover identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OmittedProviderPrefix {
    pub(crate) prefix: u8,
    pub(crate) count: usize,
}

/// A cover build is exact for every represented prefix. Oversized prefixes are
/// reported separately so their previous leases can expire without suppressing
/// publication of neighboring valid prefixes.
#[derive(Default)]
pub(crate) struct ProviderCoverBuild {
    pub(crate) cover: ProviderCover,
    pub(crate) omitted: Vec<OmittedProviderPrefix>,
}

impl ProviderCover {
    pub(crate) fn from_artifacts(
        team: VerifyingKey,
        artifacts: impl IntoIterator<Item = ArtifactId>,
    ) -> ProviderCoverBuild {
        Self::from_artifacts_with_shard_limit(team, artifacts, MAX_PROVIDER_SHARD_MEMBERS)
    }

    pub(crate) fn from_artifacts_with_shard_limit(
        team: VerifyingKey,
        artifacts: impl IntoIterator<Item = ArtifactId>,
        shard_limit: usize,
    ) -> ProviderCoverBuild {
        debug_assert!(shard_limit > 0);
        let mut keys: Vec<_> = artifacts
            .into_iter()
            .map(|artifact| provider_key(team, artifact))
            .collect();
        keys.sort_unstable();
        keys.dedup();

        let full = ProviderPatch::from_keys(keys.iter().copied());
        let mut shards = BTreeMap::new();
        let mut omitted = Vec::new();
        let mut start = 0;
        while start < keys.len() {
            let prefix = keys[start][0];
            let mut end = start + 1;
            while end < keys.len() && keys[end][0] == prefix {
                end += 1;
            }
            if end - start > shard_limit {
                omitted.push(OmittedProviderPrefix {
                    prefix,
                    count: end - start,
                });
                start = end;
                continue;
            }
            let shard_keys: Arc<[ProviderKey]> = keys[start..end].to_vec().into();
            let node = full
                .merkle_node(&[prefix])
                .expect("a provider-cover prefix exists in the full PATCH");
            debug_assert_eq!(node.leaf_count(), shard_keys.len() as u64);
            shards.insert(
                prefix,
                ProviderShard {
                    prefix,
                    digest: node.digest(),
                    keys: shard_keys,
                },
            );
            start = end;
        }

        ProviderCoverBuild {
            cover: Self { shards },
            omitted,
        }
    }

    pub(crate) fn get(&self, prefix: u8) -> Option<&ProviderShard> {
        self.shards.get(&prefix)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u8, &ProviderShard)> {
        self.shards.iter()
    }

    pub(crate) fn same_membership(&self, other: &Self) -> bool {
        self.shards.len() == other.shards.len()
            && self.shards.iter().zip(&other.shards).all(
                |((left_prefix, left), (right_prefix, right))| {
                    left_prefix == right_prefix
                        && left.digest == right.digest
                        && left.keys.len() == right.keys.len()
                },
            )
    }

    #[cfg(test)]
    pub(crate) fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

/// Fully validated replacement built before the directory's atomic mutation.
pub(crate) struct ProviderShardCandidate {
    prefix: u8,
    digest: [u8; 32],
    keys: Box<[ProviderKey]>,
}

impl ProviderShardCandidate {
    pub(crate) fn validate(
        prefix: u8,
        digest: [u8; 32],
        count: u32,
        keys: Vec<ProviderKey>,
    ) -> Result<Self> {
        let count = usize::try_from(count).expect("u32 fits usize on supported platforms");
        if count == 0 || count > MAX_PROVIDER_SHARD_MEMBERS {
            bail!("provider-cover shard count is outside the supported bounds");
        }
        if keys.len() != count {
            bail!("provider-cover shard body count does not match its probe");
        }
        if keys.iter().any(|key| key[0] != prefix) {
            bail!("provider-cover shard contains a key outside its prefix");
        }
        if !keys.windows(2).all(|pair| pair[0] < pair[1]) {
            bail!("provider-cover shard keys are not strictly ascending");
        }
        let rebuilt = ProviderPatch::from_keys(keys.iter().copied())
            .merkle_root()
            .expect("a validated provider-cover shard is nonempty");
        if rebuilt != digest {
            bail!("provider-cover shard digest does not match its body");
        }
        Ok(Self {
            prefix,
            digest,
            keys: keys.into_boxed_slice(),
        })
    }

    fn count(&self) -> usize {
        self.keys.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderProbe {
    Known,
    Need,
    Full,
}

struct StoredProviderShard {
    digest: [u8; 32],
    keys: Box<[ProviderKey]>,
    expires_at: Mono,
}

/// Receiver-local soft directory. The primary map owns complete shards; the
/// prefix index only names candidate providers. Exact lookup checks the sorted
/// immutable shard body, avoiding a second copy of every membership and making
/// replacement and expiry proportional to shards rather than artifacts.
pub(crate) struct ProviderDirectory {
    shards: BTreeMap<(u8, PeerId), StoredProviderShard>,
    providers_by_prefix: BTreeMap<u8, BTreeSet<PeerId>>,
    lookup_cursor_by_prefix: BTreeMap<u8, PeerId>,
    deadlines: BTreeSet<(Mono, u8, PeerId)>,
    member_count: usize,
    members_by_provider: BTreeMap<PeerId, usize>,
    limits: DirectoryLimits,
}

#[derive(Clone, Copy)]
struct DirectoryLimits {
    lease: Duration,
    shards: usize,
    members: usize,
    members_per_provider: usize,
}

impl Default for ProviderDirectory {
    fn default() -> Self {
        Self {
            shards: BTreeMap::new(),
            providers_by_prefix: BTreeMap::new(),
            lookup_cursor_by_prefix: BTreeMap::new(),
            deadlines: BTreeSet::new(),
            member_count: 0,
            members_by_provider: BTreeMap::new(),
            limits: DirectoryLimits {
                lease: PROVIDER_LEASE_LIFETIME,
                shards: MAX_PROVIDER_SHARDS,
                members: MAX_PROVIDER_MEMBERS,
                members_per_provider: MAX_PROVIDER_MEMBERS_PER_PROVIDER,
            },
        }
    }
}

impl ProviderDirectory {
    /// Compare one advertised root. Matching state is renewed without
    /// touching any membership; changed state asks for the full canonical body.
    pub(crate) fn probe(
        &mut self,
        prefix: u8,
        digest: [u8; 32],
        count: u32,
        provider: PeerId,
        now: Mono,
    ) -> ProviderProbe {
        self.prune_expired(now);
        let count = usize::try_from(count).expect("u32 fits usize on supported platforms");
        if count == 0 || count > MAX_PROVIDER_SHARD_MEMBERS {
            return ProviderProbe::Full;
        }
        let key = (prefix, provider);
        if self
            .shards
            .get(&key)
            .is_some_and(|shard| shard.digest == digest && shard.keys.len() == count)
        {
            self.renew(key, now);
            return ProviderProbe::Known;
        }

        let old_count = self.shards.get(&key).map_or(0, |shard| shard.keys.len());
        let projected_members = self.member_count - old_count + count;
        let projected_provider_members = self
            .members_by_provider
            .get(&provider)
            .copied()
            .unwrap_or(0)
            - old_count
            + count;
        if (!self.shards.contains_key(&key) && self.shards.len() >= self.limits.shards)
            || projected_members > self.limits.members
            || projected_provider_members > self.limits.members_per_provider
        {
            ProviderProbe::Full
        } else {
            ProviderProbe::Need
        }
    }

    /// Atomically install one already-validated body. Capacity failure leaves
    /// the old live shard untouched.
    pub(crate) fn install(
        &mut self,
        candidate: ProviderShardCandidate,
        provider: PeerId,
        now: Mono,
    ) -> bool {
        self.prune_expired(now);
        let key = (candidate.prefix, provider);

        let old_count = self.shards.get(&key).map_or(0, |shard| shard.keys.len());
        let projected_members = self.member_count - old_count + candidate.count();
        let projected_provider_members = self
            .members_by_provider
            .get(&provider)
            .copied()
            .unwrap_or(0)
            - old_count
            + candidate.count();
        if (!self.shards.contains_key(&key) && self.shards.len() >= self.limits.shards)
            || projected_members > self.limits.members
            || projected_provider_members > self.limits.members_per_provider
        {
            return false;
        }

        let replacing = self.shards.contains_key(&key);
        if let Some(old) = self.shards.remove(&key) {
            self.deadlines.remove(&(old.expires_at, key.0, key.1));
            self.member_count -= old.keys.len();
            self.decrement_provider_members(provider, old.keys.len());
        }

        let expires_at = now + self.limits.lease;
        if !replacing {
            self.providers_by_prefix
                .entry(candidate.prefix)
                .or_default()
                .insert(provider);
        }
        self.member_count += candidate.count();
        *self.members_by_provider.entry(provider).or_default() += candidate.count();
        self.deadlines.insert((expires_at, key.0, key.1));
        self.shards.insert(
            key,
            StoredProviderShard {
                digest: candidate.digest,
                keys: candidate.keys,
                expires_at,
            },
        );
        true
    }

    /// Return bounded live providers for one exact rendezvous key.
    pub(crate) fn get(&mut self, key: ProviderKey, now: Mono) -> Vec<PeerId> {
        self.prune_expired(now);
        let prefix = key[0];
        let Some(providers) = self.providers_by_prefix.get(&prefix) else {
            return Vec::new();
        };
        let cursor = self
            .lookup_cursor_by_prefix
            .get(&prefix)
            .copied()
            .or_else(|| providers.last().copied())
            .expect("a retained prefix has at least one provider");
        let mut result = Vec::with_capacity(MAX_PROVIDERS_PER_KEY.min(providers.len()));
        let mut last_scanned = None;
        use std::ops::Bound::{Excluded, Unbounded};
        for provider in providers
            .range((Excluded(cursor), Unbounded))
            .chain(providers.range(..=cursor))
            .take(MAX_PROVIDER_SHARDS_SCANNED_PER_GET)
            .copied()
        {
            last_scanned = Some(provider);
            let Some(shard) = self.shards.get(&(prefix, provider)) else {
                continue;
            };
            if shard.expires_at > now && shard.keys.binary_search(&key).is_ok() {
                result.push(provider);
                if result.len() == MAX_PROVIDERS_PER_KEY {
                    break;
                }
            }
        }
        if let Some(last) = last_scanned {
            self.lookup_cursor_by_prefix.insert(prefix, last);
        }
        result
    }

    fn renew(&mut self, key: (u8, PeerId), now: Mono) {
        let shard = self
            .shards
            .get_mut(&key)
            .expect("renewed provider-cover shard exists");
        self.deadlines.remove(&(shard.expires_at, key.0, key.1));
        shard.expires_at = now + self.limits.lease;
        self.deadlines.insert((shard.expires_at, key.0, key.1));
    }

    fn prune_expired(&mut self, now: Mono) {
        for _ in 0..MAX_EXPIRED_PROVIDER_SHARDS_PER_CALL {
            let Some((expires_at, prefix, provider)) = self.deadlines.first().copied() else {
                break;
            };
            if expires_at > now {
                break;
            }
            self.deadlines.remove(&(expires_at, prefix, provider));
            let key = (prefix, provider);
            let Some(shard) = self.shards.remove(&key) else {
                continue;
            };
            self.member_count -= shard.keys.len();
            self.decrement_provider_members(provider, shard.keys.len());
            let remove_prefix = {
                let providers = self
                    .providers_by_prefix
                    .get_mut(&prefix)
                    .expect("stored shard contributes to its prefix index");
                providers.remove(&provider);
                providers.is_empty()
            };
            if remove_prefix {
                self.providers_by_prefix.remove(&prefix);
                self.lookup_cursor_by_prefix.remove(&prefix);
            }
        }
    }

    fn decrement_provider_members(&mut self, provider: PeerId, count: usize) {
        let remove = {
            let members = self
                .members_by_provider
                .get_mut(&provider)
                .expect("stored shard contributes to provider membership");
            *members -= count;
            *members == 0
        };
        if remove {
            self.members_by_provider.remove(&provider);
        }
    }

    #[cfg(test)]
    fn with_limits(
        lease: Duration,
        shards: usize,
        members: usize,
        members_per_provider: usize,
    ) -> Self {
        Self {
            limits: DirectoryLimits {
                lease,
                shards,
                members,
                members_per_provider,
            },
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn candidate(prefix: u8, suffixes: impl IntoIterator<Item = u32>) -> ProviderShardCandidate {
        let keys: Vec<_> = suffixes
            .into_iter()
            .map(|suffix| {
                let mut key = [0; 32];
                key[0] = prefix;
                key[28..].copy_from_slice(&suffix.to_be_bytes());
                key
            })
            .collect();
        let digest = ProviderPatch::from_keys(keys.iter().copied())
            .merkle_root()
            .unwrap();
        ProviderShardCandidate::validate(prefix, digest, keys.len() as u32, keys).unwrap()
    }

    fn install(
        directory: &mut ProviderDirectory,
        candidate: ProviderShardCandidate,
        provider: PeerId,
        now: Mono,
    ) -> bool {
        assert_eq!(
            directory.probe(
                candidate.prefix,
                candidate.digest,
                candidate.count() as u32,
                provider,
                now,
            ),
            ProviderProbe::Need
        );
        directory.install(candidate, provider, now)
    }

    #[test]
    fn keys_and_prefix_routes_are_deterministic_and_team_scoped() {
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
        assert_eq!(
            provider_prefix_key(team_a, 7),
            provider_prefix_key(team_a, 7)
        );
        assert_ne!(
            provider_prefix_key(team_a, 7),
            provider_prefix_key(team_b, 7)
        );
        assert_ne!(
            provider_prefix_key(team_a, 7),
            provider_prefix_key(team_a, 8)
        );
    }

    #[test]
    fn a_large_offer_set_collapses_to_at_most_256_publication_units() {
        let team = SigningKey::from_bytes(&[1; 32]).verifying_key();
        let build = ProviderCover::from_artifacts(
            team,
            (0..243_955_u64).map(|index| {
                let mut artifact = [0; 32];
                artifact[24..].copy_from_slice(&index.to_be_bytes());
                artifact
            }),
        );
        assert!(build.omitted.is_empty());
        let cover = build.cover;
        assert!(cover.shard_count() <= 256);
        assert_eq!(
            cover
                .iter()
                .map(|(_, shard)| shard.keys().len())
                .sum::<usize>(),
            243_955
        );
    }

    #[test]
    fn full_patch_prefix_digests_equal_standalone_shard_roots() {
        let team = SigningKey::from_bytes(&[5; 32]).verifying_key();
        let build = ProviderCover::from_artifacts(
            team,
            (0..1000_u64).map(|index| {
                let mut artifact = [0; 32];
                artifact[24..].copy_from_slice(&index.to_be_bytes());
                artifact
            }),
        );
        assert!(build.omitted.is_empty());
        let cover = build.cover;
        for (_, shard) in cover.iter() {
            assert_eq!(
                ProviderPatch::from_keys(shard.keys().iter().copied()).merkle_root(),
                Some(shard.digest())
            );
        }
    }

    #[test]
    fn same_root_renewal_does_not_rebuild_membership() {
        let now = crate::clock::mono_now();
        let provider = [2; 32];
        let candidate = candidate(7, 0..1000);
        let digest = candidate.digest;
        let count = candidate.count() as u32;
        let exact = candidate.keys[17];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(10), 4, 2000, 2000);

        assert!(install(&mut directory, candidate, provider, now));
        let prefix_index_before = directory.providers_by_prefix.clone();
        assert_eq!(
            directory.probe(7, digest, count, provider, now + Duration::from_secs(5)),
            ProviderProbe::Known
        );
        assert_eq!(directory.providers_by_prefix, prefix_index_before);
        assert_eq!(
            directory.get(exact, now + Duration::from_secs(14)),
            vec![provider]
        );
        assert!(
            directory
                .get(exact, now + Duration::from_secs(15))
                .is_empty()
        );
    }

    #[test]
    fn replacement_updates_exact_membership_and_failed_replacement_is_atomic() {
        let now = crate::clock::mono_now();
        let provider = [3; 32];
        let old = candidate(9, [1, 2]);
        let old_key = old.keys[0];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(10), 1, 3, 3);
        assert!(install(&mut directory, old, provider, now));

        let replacement = candidate(9, [2, 3]);
        let retained = replacement.keys[0];
        let added = replacement.keys[1];
        assert!(install(&mut directory, replacement, provider, now));
        assert!(directory.get(old_key, now).is_empty());
        assert_eq!(directory.get(retained, now), vec![provider]);
        assert_eq!(directory.get(added, now), vec![provider]);

        let too_large = candidate(9, [4, 5, 6, 7]);
        assert_eq!(
            directory.probe(
                too_large.prefix,
                too_large.digest,
                too_large.count() as u32,
                provider,
                now,
            ),
            ProviderProbe::Full
        );
        assert!(!directory.install(too_large, provider, now));
        assert_eq!(directory.get(retained, now), vec![provider]);
        assert_eq!(directory.get(added, now), vec![provider]);
    }

    #[test]
    fn validation_rejects_wrong_prefix_order_count_and_digest() {
        let a = candidate(1, [1]);
        let key = a.keys[0];
        assert!(ProviderShardCandidate::validate(2, a.digest, 1, vec![key]).is_err());
        assert!(ProviderShardCandidate::validate(1, a.digest, 2, vec![key]).is_err());
        assert!(ProviderShardCandidate::validate(1, [9; 32], 1, vec![key]).is_err());
        assert!(ProviderShardCandidate::validate(1, a.digest, 2, vec![key, key]).is_err());
    }

    #[test]
    fn one_provider_can_represent_more_than_65_536_members() {
        let now = crate::clock::mono_now();
        let provider = [4; 32];
        let mut directory = ProviderDirectory::default();
        let first = candidate(11, 0..32_768);
        let second = candidate(12, 0..32_769);
        let exact = second.keys[32_768];
        assert!(install(&mut directory, first, provider, now));
        assert!(install(&mut directory, second, provider, now));
        assert_eq!(directory.get(exact, now), vec![provider]);
        assert_eq!(directory.member_count, 65_537);
    }

    #[test]
    fn one_provider_cannot_consume_an_otherwise_available_global_directory() {
        let now = crate::clock::mono_now();
        let greedy = [7; 32];
        let peer = [8; 32];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(10), 8, 100, 4);
        assert!(install(&mut directory, candidate(1, 0..4), greedy, now));
        assert_eq!(
            directory.probe(2, candidate(2, [1]).digest, 1, greedy, now),
            ProviderProbe::Full
        );
        assert!(install(&mut directory, candidate(2, [1]), peer, now));
    }

    #[test]
    fn exact_lookup_rotates_across_more_than_64_stored_providers() {
        let now = crate::clock::mono_now();
        let exact = candidate(3, [1]).keys[0];
        let mut directory = ProviderDirectory::default();
        for byte in 1..=70 {
            assert!(install(&mut directory, candidate(3, [1]), [byte; 32], now,));
        }

        let first = directory.get(exact, now);
        let second = directory.get(exact, now);
        assert_eq!(first.len(), MAX_PROVIDERS_PER_KEY);
        assert_eq!(second.len(), MAX_PROVIDERS_PER_KEY);
        let seen: BTreeSet<_> = first.into_iter().chain(second).collect();
        assert_eq!(seen.len(), 70, "the result cap is not a storage cap");
    }

    #[test]
    fn sparse_lookup_scans_a_bounded_rotating_prefix_window() {
        let now = crate::clock::mono_now();
        let exact = candidate(3, [999]).keys[0];
        let mut directory = ProviderDirectory::default();
        let mut expected = [0; 32];
        expected[30..].copy_from_slice(&299_u16.to_be_bytes());
        for index in 0..300_u16 {
            let mut provider = [0; 32];
            provider[30..].copy_from_slice(&index.to_be_bytes());
            let suffix = if provider == expected { 999 } else { 1 };
            assert!(install(
                &mut directory,
                candidate(3, [suffix]),
                provider,
                now,
            ));
        }

        assert!(directory.get(exact, now).is_empty());
        assert_eq!(directory.get(exact, now), vec![expected]);
    }

    #[test]
    fn expiry_reclamation_is_bounded_and_unreclaimed_shards_are_not_returned() {
        let now = crate::clock::mono_now();
        let exact = candidate(3, [1]).keys[0];
        let mut directory = ProviderDirectory::with_limits(Duration::from_secs(1), 100, 100, 1);
        for byte in 1..=65 {
            assert!(install(&mut directory, candidate(3, [1]), [byte; 32], now,));
        }

        assert!(
            directory
                .get(exact, now + Duration::from_secs(1))
                .is_empty()
        );
        assert_eq!(directory.shards.len(), 1);
        assert!(
            directory
                .get(exact, now + Duration::from_secs(1))
                .is_empty()
        );
        assert!(directory.shards.is_empty());
    }
}
