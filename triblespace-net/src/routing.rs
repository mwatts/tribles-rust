//! Bounded, transport-independent XOR routing.
//!
//! The table distinguishes identities merely named by another peer from peers
//! that answered a direct, authenticated request.  The lookup machine likewise
//! makes no network decisions: it yields a bounded batch and waits for its
//! caller to report either an authenticated response or a failure.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::transport::PeerId;

/// An arbitrary point in the 256-bit XOR keyspace.
pub(crate) type RoutingKey = [u8; 32];
/// Number of peers retained in one XOR-distance bucket.
pub(crate) const K: usize = 20;
/// Maximum number of concurrent requests in one iterative lookup.
pub(crate) const ALPHA: usize = 3;
const BUCKET_COUNT: usize = std::mem::size_of::<PeerId>() * 8;
const ROUTING_CAPACITY: usize = BUCKET_COUNT * K;
const MAX_LOOKUP_QUERIES: usize = ROUTING_CAPACITY;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteState {
    /// The identity was configured locally or named by another peer.
    Candidate,
    /// The peer answered a direct request whose transport and capability were
    /// authenticated by the caller.
    Verified,
}

#[derive(Default)]
struct Bucket {
    entries: BTreeMap<PeerId, RouteState>,
}

/// A deterministic, hard-bounded Kademlia-style routing table.
///
/// Each peer occupies the bucket selected by the most significant bit in its
/// XOR distance from `local`.  A bucket retains at most [`K`] identities.
/// Verified peers take precedence over candidates; ties are resolved by XOR
/// distance from the local peer and then by identity.  Consequently inserting
/// the same evidence in a different order produces the same table. Explicit
/// local configuration is retained separately and is expected to be bounded
/// by the caller rather than by rules for hostile learned state.
pub(crate) struct RoutingTable {
    local: PeerId,
    /// Explicit local configuration is trusted provenance, not hostile learned
    /// state. It remains available even if the corresponding learned route is
    /// evicted or a connection attempt fails.
    configured: BTreeSet<PeerId>,
    /// Stored PEER/snapshot evidence eligible for periodic anti-entropy.
    /// DHT referrals never enter this set.
    sync: BTreeSet<PeerId>,
    buckets: [Bucket; BUCKET_COUNT],
}

impl RoutingTable {
    /// Start with locally configured bootstrap identities as unverified
    /// candidates.  Configuration does not manufacture proof of reachability.
    pub(crate) fn new<I>(local: PeerId, configured: I) -> Self
    where
        I: IntoIterator<Item = PeerId>,
    {
        let configured = configured
            .into_iter()
            .filter(|peer| *peer != local)
            .collect();
        Self {
            local,
            configured,
            sync: BTreeSet::new(),
            buckets: std::array::from_fn(|_| Bucket::default()),
        }
    }

    /// Remember an unverified identity without demoting an already verified
    /// route. Returns whether the identity survived the bucket bound.
    pub(crate) fn note_candidate(&mut self, peer: PeerId) -> bool {
        if self.configured.contains(&peer) {
            return true;
        }
        self.insert(peer, RouteState::Candidate)
    }

    /// Remember explicit stored PEER evidence as both a route and a periodic
    /// anti-entropy target. This provenance is distinct from DHT referrals.
    pub(crate) fn note_sync_candidate(&mut self, peer: PeerId) -> bool {
        if self.configured.contains(&peer) {
            return true;
        }
        self.sync.insert(peer);
        self.insert(peer, RouteState::Candidate)
    }

    /// Promote a peer only after the caller observed a direct authenticated
    /// response. Returns whether the identity survived the bucket bound.
    pub(crate) fn promote_authenticated(&mut self, peer: PeerId) -> bool {
        self.insert(peer, RouteState::Verified)
    }

    /// Remove failed learned evidence. Explicit local configuration survives
    /// and becomes an unverified candidate again.
    pub(crate) fn remove(&mut self, peer: PeerId) -> bool {
        let Some(bucket) = bucket_index(self.local, peer) else {
            return false;
        };
        self.buckets[bucket].entries.remove(&peer).is_some()
    }

    #[cfg(test)]
    pub(crate) fn state(&self, peer: PeerId) -> Option<RouteState> {
        let learned = bucket_index(self.local, peer)
            .and_then(|bucket| self.buckets[bucket].entries.get(&peer).copied());
        learned.or_else(|| {
            self.configured
                .contains(&peer)
                .then_some(RouteState::Candidate)
        })
    }

    #[cfg(test)]
    /// Number of unique configured and learned identities.
    pub(crate) fn len(&self) -> usize {
        self.all().len()
    }

    #[cfg(test)]
    pub(crate) fn learned_len(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.entries.len()).sum()
    }

    #[cfg(test)]
    pub(crate) fn configured_len(&self) -> usize {
        self.configured.len()
    }

    /// Return at most `limit` known identities ordered by XOR distance from
    /// `target`. Candidate status does not affect distance order.
    pub(crate) fn closest(&self, target: RoutingKey, limit: usize) -> Vec<PeerId> {
        let mut peers = self.all();
        peers.sort_unstable_by(|a, b| distance_cmp(target, *a, *b));
        peers.truncate(limit);
        peers
    }

    /// Like [`Self::closest`], but excludes identities that have never
    /// answered this process directly.
    pub(crate) fn closest_verified(&self, target: RoutingKey, limit: usize) -> Vec<PeerId> {
        let mut peers: Vec<_> =
            self.buckets
                .iter()
                .flat_map(|bucket| {
                    bucket.entries.iter().filter_map(|(peer, state)| {
                        (*state == RouteState::Verified).then_some(*peer)
                    })
                })
                .collect();
        peers.sort_unstable_by(|a, b| distance_cmp(target, *a, *b));
        peers.truncate(limit);
        peers
    }

    /// Explicit anti-entropy targets only: local configuration plus stored
    /// PEER/snapshot evidence, never identities learned solely through DHT.
    pub(crate) fn sync_candidates(&self) -> Vec<PeerId> {
        let mut peers = self.configured.clone();
        peers.extend(self.sync.iter().copied());
        peers.into_iter().collect()
    }

    /// Whether this node has any durable seed, synchronized peer evidence, or
    /// learned DHT route from which remote replication can reasonably be
    /// expected. Failed learned routes may leave their synchronized
    /// provenance behind, so an outage cannot silently turn a replicated
    /// publication into a singleton publication on its next retry.
    pub(crate) fn expects_remote(&self) -> bool {
        !self.configured.is_empty()
            || !self.sync.is_empty()
            || self.buckets.iter().any(|bucket| !bucket.entries.is_empty())
    }

    fn all(&self) -> Vec<PeerId> {
        let mut peers = self.configured.clone();
        for peer in self
            .buckets
            .iter()
            .flat_map(|bucket| bucket.entries.keys().copied())
        {
            peers.insert(peer);
        }
        peers.into_iter().collect()
    }

    fn insert(&mut self, peer: PeerId, state: RouteState) -> bool {
        let Some(index) = bucket_index(self.local, peer) else {
            return false;
        };
        let bucket = &mut self.buckets[index];
        bucket
            .entries
            .entry(peer)
            .and_modify(|old| {
                if state == RouteState::Verified {
                    *old = RouteState::Verified;
                }
            })
            .or_insert(state);

        if bucket.entries.len() > K {
            let mut ranked: Vec<_> = bucket
                .entries
                .iter()
                .map(|(peer, state)| (*peer, *state))
                .collect();
            ranked.sort_unstable_by(|(a, a_state), (b, b_state)| {
                route_rank(*a_state)
                    .cmp(&route_rank(*b_state))
                    .then_with(|| distance_cmp(self.local, *a, *b))
            });
            let retained: BTreeSet<_> = ranked.into_iter().take(K).map(|(peer, _)| peer).collect();
            bucket.entries.retain(|peer, _| retained.contains(peer));
        }
        bucket.entries.contains_key(&peer)
    }

    #[cfg(test)]
    fn bucket_len(&self, index: usize) -> usize {
        self.buckets[index].entries.len()
    }
}

fn route_rank(state: RouteState) -> u8 {
    match state {
        RouteState::Verified => 0,
        RouteState::Candidate => 1,
    }
}

fn bucket_index(local: PeerId, peer: PeerId) -> Option<usize> {
    for (byte_index, (&local, &peer)) in local.iter().zip(&peer).enumerate() {
        let distance = local ^ peer;
        if distance != 0 {
            let bit_in_byte = 7 - distance.leading_zeros() as usize;
            return Some((std::mem::size_of::<PeerId>() - 1 - byte_index) * 8 + bit_in_byte);
        }
    }
    None
}

pub(crate) fn distance_cmp(target: RoutingKey, a: PeerId, b: PeerId) -> Ordering {
    a.iter()
        .zip(b)
        .zip(target)
        .map(|((&a, b), target)| (a ^ target).cmp(&(b ^ target)))
        .find(|ordering| !ordering.is_eq())
        .unwrap_or_else(|| a.cmp(&b))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LookupState {
    Pending,
    InFlight,
    Responded,
}

/// A synchronous control plane for one iterative XOR lookup.
///
/// The caller may issue every peer returned by [`Self::next_batch`] in
/// parallel.  It must eventually report each as either an authenticated
/// response or a failure. Candidate replies never verify their named peers;
/// only the peer that directly answered is promoted.
pub(crate) struct IterativeLookup {
    local: PeerId,
    target: RoutingKey,
    shortlist: BTreeMap<PeerId, LookupState>,
    queried: BTreeSet<PeerId>,
    authenticated_responders: Vec<PeerId>,
}

impl IterativeLookup {
    pub(crate) fn new<I>(local: PeerId, target: RoutingKey, seeds: I) -> Self
    where
        I: IntoIterator<Item = PeerId>,
    {
        let mut lookup = Self {
            local,
            target,
            shortlist: BTreeMap::new(),
            queried: BTreeSet::new(),
            authenticated_responders: Vec::new(),
        };
        lookup.add_candidates(bounded_closest(target, local, seeds));
        lookup.trim_shortlist();
        lookup
    }

    /// Mark and return the next closest batch, never exceeding [`ALPHA`]
    /// concurrent requests. An empty batch with `is_finished() == false`
    /// means earlier requests are still in flight.
    pub(crate) fn next_batch(&mut self) -> Vec<PeerId> {
        if self.queried.len() >= MAX_LOOKUP_QUERIES {
            self.shortlist
                .retain(|_, state| *state == LookupState::InFlight);
            return Vec::new();
        }

        let in_flight = self
            .shortlist
            .values()
            .filter(|state| **state == LookupState::InFlight)
            .count();
        let available = ALPHA.saturating_sub(in_flight);
        if available == 0 {
            return Vec::new();
        }

        let mut pending: Vec<_> = self
            .shortlist
            .iter()
            .filter_map(|(peer, state)| (*state == LookupState::Pending).then_some(*peer))
            .collect();
        pending.sort_unstable_by(|a, b| distance_cmp(self.target, *a, *b));
        pending.truncate(available.min(MAX_LOOKUP_QUERIES.saturating_sub(self.queried.len())));
        for peer in &pending {
            self.shortlist.insert(*peer, LookupState::InFlight);
            self.queried.insert(*peer);
        }
        pending
    }

    /// Accept a bounded FIND_NODE reply from a peer in the current batch.
    ///
    /// `candidates` may itself be oversized or duplicate-heavy; only its
    /// deterministic [`K`] closest distinct identities are retained. The
    /// responding peer is promoted, while every identity it names remains a
    /// candidate until that identity answers directly in a later batch.
    pub(crate) fn record_authenticated_response<I>(
        &mut self,
        peer: PeerId,
        candidates: I,
        routes: &mut RoutingTable,
    ) -> bool
    where
        I: IntoIterator<Item = PeerId>,
    {
        if self.shortlist.get(&peer) != Some(&LookupState::InFlight) {
            return false;
        }

        self.shortlist.insert(peer, LookupState::Responded);
        self.authenticated_responders = bounded_closest(
            self.target,
            self.local,
            self.authenticated_responders
                .iter()
                .copied()
                .chain(std::iter::once(peer)),
        );
        routes.promote_authenticated(peer);
        let candidates = bounded_closest(self.target, self.local, candidates);
        for candidate in &candidates {
            routes.note_candidate(*candidate);
        }
        self.add_candidates(candidates);
        self.trim_shortlist();
        true
    }

    /// Complete one in-flight request as failed and remove its stale route.
    pub(crate) fn record_failure(&mut self, peer: PeerId, routes: &mut RoutingTable) -> bool {
        if self.shortlist.get(&peer) != Some(&LookupState::InFlight) {
            return false;
        }
        self.shortlist.remove(&peer);
        routes.remove(peer);
        self.trim_shortlist();
        true
    }

    pub(crate) fn is_finished(&self) -> bool {
        let has_in_flight = self
            .shortlist
            .values()
            .any(|state| *state == LookupState::InFlight);
        let has_pending = self
            .shortlist
            .values()
            .any(|state| *state == LookupState::Pending);
        !has_in_flight && (!has_pending || self.queried.len() >= MAX_LOOKUP_QUERIES)
    }

    /// The at-most-K peers closest to the target that answered this lookup
    /// directly and authenticated successfully. This lookup-local evidence is
    /// independent of later routing-table eviction.
    pub(crate) fn closest_authenticated_responders(&self) -> &[PeerId] {
        &self.authenticated_responders
    }

    #[cfg(test)]
    fn shortlist_len(&self) -> usize {
        self.shortlist.len()
    }

    fn add_candidates<I>(&mut self, candidates: I)
    where
        I: IntoIterator<Item = PeerId>,
    {
        for peer in candidates {
            if peer != self.local && !self.queried.contains(&peer) {
                self.shortlist.entry(peer).or_insert(LookupState::Pending);
            }
        }
    }

    fn trim_shortlist(&mut self) {
        let mut ranked: Vec<_> = self
            .shortlist
            .iter()
            .filter_map(|(peer, state)| (*state != LookupState::InFlight).then_some(*peer))
            .collect();
        ranked.sort_unstable_by(|a, b| distance_cmp(self.target, *a, *b));
        let retained: BTreeSet<_> = ranked.into_iter().take(K).collect();
        self.shortlist
            .retain(|peer, state| *state == LookupState::InFlight || retained.contains(peer));
    }
}

/// Select a bounded, order-independent set from an untrusted iterator without
/// first collecting the whole reply.
fn bounded_closest<I>(target: RoutingKey, local: PeerId, peers: I) -> Vec<PeerId>
where
    I: IntoIterator<Item = PeerId>,
{
    let mut closest = Vec::with_capacity(K);
    for peer in peers {
        if peer == local || closest.contains(&peer) {
            continue;
        }
        let position = closest
            .binary_search_by(|known| distance_cmp(target, *known, peer))
            .unwrap_or_else(|position| position);
        if position < K {
            closest.insert(position, peer);
            closest.truncate(K);
        }
    }
    closest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u16) -> PeerId {
        let mut id = [0; 32];
        id[30..].copy_from_slice(&n.to_be_bytes());
        id
    }

    fn drive_lookup(
        local: PeerId,
        target: RoutingKey,
        seeds: Vec<PeerId>,
        network: &BTreeMap<PeerId, Vec<PeerId>>,
    ) -> (RoutingTable, Vec<PeerId>) {
        let mut routes = RoutingTable::new(local, seeds.clone());
        let mut lookup = IterativeLookup::new(local, target, seeds);
        let mut contacted = Vec::new();
        while !lookup.is_finished() {
            let batch = lookup.next_batch();
            assert!(!batch.is_empty(), "driver has no abandoned in-flight work");
            for peer in batch {
                contacted.push(peer);
                if let Some(reply) = network.get(&peer) {
                    assert!(lookup.record_authenticated_response(
                        peer,
                        reply.iter().copied(),
                        &mut routes
                    ));
                } else {
                    assert!(lookup.record_failure(peer, &mut routes));
                }
            }
        }
        (routes, contacted)
    }

    #[test]
    fn bucket_indices_follow_xor_magnitude() {
        let local = [0; 32];
        assert_eq!(bucket_index(local, local), None);
        assert_eq!(bucket_index(local, id(1)), Some(0));
        assert_eq!(bucket_index(local, id(2)), Some(1));
        let mut high = [0; 32];
        high[0] = 0x80;
        assert_eq!(bucket_index(local, high), Some(255));
    }

    #[test]
    fn configured_routes_start_as_candidates_and_self_is_ignored() {
        let local = id(1);
        let table = RoutingTable::new(local, [local, id(2), id(2)]);
        assert_eq!(table.len(), 1);
        assert_eq!(table.configured_len(), 1);
        assert_eq!(table.learned_len(), 0);
        assert_eq!(table.state(local), None);
        assert_eq!(table.state(id(2)), Some(RouteState::Candidate));
        assert!(table.closest_verified(id(2), K).is_empty());
        assert_eq!(table.closest(id(2), K), vec![id(2)]);
    }

    #[test]
    fn durable_peer_provenance_keeps_remote_replication_expected_during_outage() {
        let local = id(1);
        let remote = id(2);
        let mut table = RoutingTable::new(local, []);
        assert!(!table.expects_remote());
        assert!(table.note_sync_candidate(remote));
        assert!(table.expects_remote());
        assert!(table.remove(remote));
        assert!(
            table.expects_remote(),
            "a failed route must not silently turn a replicated publication into a singleton"
        );
    }

    #[test]
    fn direct_response_promotes_only_the_responder() {
        let local = id(0);
        let seed = id(1);
        let named = id(2);
        let mut routes = RoutingTable::new(local, [seed]);
        let mut lookup = IterativeLookup::new(local, named, [seed]);
        assert_eq!(lookup.next_batch(), vec![seed]);
        assert!(lookup.record_authenticated_response(seed, [named], &mut routes));
        assert_eq!(routes.state(seed), Some(RouteState::Verified));
        assert_eq!(routes.state(named), Some(RouteState::Candidate));

        assert_eq!(lookup.next_batch(), vec![named]);
        assert!(lookup.record_authenticated_response(named, [], &mut routes));
        assert_eq!(routes.state(named), Some(RouteState::Verified));
        assert!(lookup.is_finished());
    }

    #[test]
    fn routing_bounds_are_hard_and_retention_is_order_independent() {
        let local = [0; 32];
        let mut peers = Vec::new();
        for n in 1..=10_000_u32 {
            let mut peer = *blake3::hash(&n.to_be_bytes()).as_bytes();
            if peer == local {
                peer[31] = 1;
            }
            peers.push(peer);
        }

        let verified: Vec<_> = peers.iter().step_by(7).copied().collect();
        let mut forward = RoutingTable::new(local, []);
        let mut reverse = RoutingTable::new(local, []);
        for peer in &peers {
            forward.note_candidate(*peer);
        }
        for peer in peers.iter().rev() {
            reverse.note_candidate(*peer);
        }
        for peer in &verified {
            forward.promote_authenticated(*peer);
        }
        for peer in verified.iter().rev() {
            reverse.promote_authenticated(*peer);
        }

        assert!(forward.learned_len() <= ROUTING_CAPACITY);
        assert!(reverse.learned_len() <= ROUTING_CAPACITY);
        for index in 0..BUCKET_COUNT {
            assert!(forward.bucket_len(index) <= K);
            assert!(reverse.bucket_len(index) <= K);
        }
        assert_eq!(
            forward.closest([0xff; 32], ROUTING_CAPACITY),
            reverse.closest([0xff; 32], ROUTING_CAPACITY)
        );
        assert_eq!(
            forward.closest_verified([0xff; 32], ROUTING_CAPACITY),
            reverse.closest_verified([0xff; 32], ROUTING_CAPACITY)
        );

        // Candidate insertion alone is likewise independent of observation order.
        let mut forward_candidates = RoutingTable::new(local, []);
        let mut reverse_candidates = RoutingTable::new(local, []);
        for peer in &peers {
            forward_candidates.note_candidate(*peer);
        }
        for peer in peers.iter().rev() {
            reverse_candidates.note_candidate(*peer);
        }
        assert_eq!(
            forward_candidates.closest([0xff; 32], ROUTING_CAPACITY),
            reverse_candidates.closest([0xff; 32], ROUTING_CAPACITY)
        );

        let removed = forward.closest([0xff; 32], 1)[0];
        assert!(forward.remove(removed));
        assert_eq!(forward.state(removed), None);
        assert!(!forward.remove(removed));
    }

    #[test]
    fn closest_is_deterministic_xor_order() {
        let table = RoutingTable::new(id(0), [id(7), id(2), id(5), id(1)]);
        assert_eq!(table.closest(id(4), 3), vec![id(5), id(7), id(1)]);
        assert_eq!(table.closest(id(4), 3), table.closest(id(4), 3));
    }

    #[test]
    fn oversized_duplicate_reply_is_bounded_and_order_independent() {
        let local = id(0);
        let seed = id(1);
        let target = id(600);
        let reply: Vec<_> = (2..=2_000).flat_map(|n| [id(n), id(n)]).collect();

        let mut forward_routes = RoutingTable::new(local, [seed]);
        let mut forward = IterativeLookup::new(local, target, [seed]);
        assert_eq!(forward.next_batch(), vec![seed]);
        assert!(forward.record_authenticated_response(
            seed,
            reply.iter().copied(),
            &mut forward_routes
        ));

        let mut reverse_routes = RoutingTable::new(local, [seed]);
        let mut reverse = IterativeLookup::new(local, target, [seed]);
        assert_eq!(reverse.next_batch(), vec![seed]);
        assert!(reverse.record_authenticated_response(
            seed,
            reply.iter().rev().copied(),
            &mut reverse_routes
        ));

        assert!(forward.shortlist_len() <= K);
        assert!(reverse.shortlist_len() <= K);
        assert_eq!(forward.next_batch(), reverse.next_batch());
    }

    #[test]
    fn sparse_line_discovers_the_target_progressively() {
        let mut network = BTreeMap::new();
        for n in 1..8 {
            network.insert(id(n), vec![id(n + 1)]);
        }
        network.insert(id(8), Vec::new());

        let (routes, contacted) = drive_lookup(id(0), id(8), vec![id(1)], &network);
        assert_eq!(contacted, (1..=8).map(id).collect::<Vec<_>>());
        assert_eq!(routes.state(id(8)), Some(RouteState::Verified));
    }

    #[test]
    fn sparse_ring_discovers_without_requerying() {
        let mut network = BTreeMap::new();
        for n in 1..=9 {
            let previous = if n == 1 { 9 } else { n - 1 };
            let next = if n == 9 { 1 } else { n + 1 };
            network.insert(id(n), vec![id(previous), id(next)]);
        }

        let (_, contacted) = drive_lookup(id(0), id(6), vec![id(1)], &network);
        assert!(contacted.contains(&id(6)));
        assert_eq!(
            contacted.len(),
            contacted.iter().copied().collect::<BTreeSet<_>>().len()
        );
        assert!(contacted.len() <= network.len());
    }

    #[test]
    fn batches_respect_alpha_and_failures_terminate() {
        let local = id(0);
        let seeds: Vec<_> = (1..=K as u16).map(id).collect();
        let mut routes = RoutingTable::new(local, seeds.iter().copied());
        let mut lookup = IterativeLookup::new(local, id(100), seeds);
        let mut contacted = 0;

        while !lookup.is_finished() {
            let batch = lookup.next_batch();
            assert!(!batch.is_empty());
            assert!(batch.len() <= ALPHA);
            contacted += batch.len();
            for peer in batch {
                assert!(lookup.record_failure(peer, &mut routes));
            }
        }
        assert_eq!(contacted, K);
        assert_eq!(routes.learned_len(), 0);
        assert_eq!(routes.configured_len(), K);
        assert_eq!(routes.len(), K);
        assert!((1..=K as u16).all(|n| routes.state(id(n)) == Some(RouteState::Candidate)));
    }

    #[test]
    fn configured_seed_survives_learned_eviction_and_failure() {
        let local = id(0);
        let configured = id(0xffff);
        let mut routes = RoutingTable::new(local, [configured]);

        routes.promote_authenticated(configured);
        assert_eq!(routes.state(configured), Some(RouteState::Verified));
        for n in 0x8000..0x8000 + K as u16 {
            routes.promote_authenticated(id(n));
        }
        assert!(
            !routes
                .closest_verified(configured, ROUTING_CAPACITY)
                .contains(&configured)
        );
        assert_eq!(routes.state(configured), Some(RouteState::Candidate));
        assert!(routes.closest(configured, K).contains(&configured));

        // A later failed retry can remove learned liveness evidence, but not
        // the explicit local instruction to use this peer as a bootstrap.
        assert!(routes.remove(id(0x8000)));
        assert!(routes.promote_authenticated(configured));
        assert_eq!(routes.state(configured), Some(RouteState::Verified));
        assert!(routes.remove(configured));
        assert_eq!(routes.state(configured), Some(RouteState::Candidate));
        assert!(routes.closest(configured, K).contains(&configured));
    }

    #[test]
    fn dht_referrals_never_become_anti_entropy_targets() {
        let local = [0; 32];
        let configured = [id(1)];
        let stored_peer = id(2);
        let dht_peer = id(3);
        let mut routes = RoutingTable::new(local, configured);

        assert!(routes.note_sync_candidate(stored_peer));
        assert!(routes.note_candidate(dht_peer));
        assert!(routes.promote_authenticated(dht_peer));
        assert_eq!(routes.sync_candidates(), vec![configured[0], stored_peer]);

        assert!(routes.remove(stored_peer));
        assert_eq!(routes.state(stored_peer), None);
        assert_eq!(routes.sync_candidates(), vec![configured[0], stored_peer]);
    }

    #[test]
    fn sync_provenance_survives_bucket_eviction_without_outranking_routes() {
        let local = [0; 32];
        let mut routes = RoutingTable::new(local, []);
        let mut sync_peers = Vec::new();
        for suffix in 1..=K + 1 {
            let mut peer = [0; 32];
            peer[0] = 0x80;
            peer[30..].copy_from_slice(&(suffix as u16).to_be_bytes());
            routes.note_sync_candidate(peer);
            sync_peers.push(peer);
        }
        for suffix in 100..100 + K {
            let mut peer = [0; 32];
            peer[0] = 0x80;
            peer[30..].copy_from_slice(&(suffix as u16).to_be_bytes());
            routes.promote_authenticated(peer);
        }

        assert_eq!(routes.closest_verified([0; 32], K).len(), K);
        assert_eq!(routes.sync_candidates(), sync_peers);
    }

    #[test]
    fn lookup_retains_target_near_authenticated_responders_after_route_eviction() {
        let local = id(0);
        let responder = id(0xffff);
        let target: RoutingKey = responder;
        let mut routes = RoutingTable::new(local, [responder]);
        let mut lookup = IterativeLookup::new(local, target, [responder]);

        assert_eq!(lookup.next_batch(), vec![responder]);
        assert!(lookup.record_authenticated_response(responder, [], &mut routes));
        assert_eq!(lookup.closest_authenticated_responders(), &[responder]);

        // All peers occupy the same local-distance bucket. The long-lived
        // table prefers the K routes nearer `local`, but lookup results remain
        // ranked by the arbitrary lookup target.
        for n in 0x8000..0x8000 + K as u16 {
            routes.promote_authenticated(id(n));
        }
        assert!(
            !routes
                .closest_verified(target, ROUTING_CAPACITY)
                .contains(&responder)
        );
        assert_eq!(lookup.closest_authenticated_responders(), &[responder]);
    }

    #[test]
    fn lookup_responder_result_is_bounded_and_target_ordered() {
        let local = id(0);
        let target = id(25);
        let mut routes = RoutingTable::new(local, [id(1)]);
        let mut lookup = IterativeLookup::new(local, target, [id(1)]);

        for n in 1..=25 {
            assert_eq!(lookup.next_batch(), vec![id(n)]);
            let next = (n < 25).then(|| id(n + 1));
            assert!(lookup.record_authenticated_response(id(n), next.into_iter(), &mut routes));
        }

        let responders = lookup.closest_authenticated_responders();
        assert_eq!(responders.len(), K);
        assert_eq!(responders[0], target);
        assert!(!responders.contains(&id(2)));
        assert!(
            responders
                .windows(2)
                .all(|pair| distance_cmp(target, pair[0], pair[1]).is_le())
        );
    }

    #[test]
    fn unsolicited_or_duplicate_completion_cannot_promote() {
        let local = id(0);
        let seed = id(1);
        let stranger = id(2);
        let mut routes = RoutingTable::new(local, [seed]);
        let mut lookup = IterativeLookup::new(local, stranger, [seed]);

        assert!(!lookup.record_authenticated_response(stranger, [], &mut routes));
        assert_eq!(routes.state(stranger), None);
        assert_eq!(lookup.next_batch(), vec![seed]);
        assert!(lookup.record_authenticated_response(seed, [stranger], &mut routes));
        assert!(!lookup.record_authenticated_response(seed, [], &mut routes));
    }
}
