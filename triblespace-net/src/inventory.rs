//! Authorized, componentwise inventory views for periodic anti-entropy.
//!
//! Inventory synchronization is a single protocol over four grow-only key
//! sets: [`InventoryComponent::Peer`],
//! [`InventoryComponent::CollectionRecord`],
//! [`InventoryComponent::CapabilityProof`], and [`InventoryComponent::Blob`].
//! A CONNECT proof admits a transport connection only. Before inventory data
//! is disclosed, the remote transport key must separately prove exact
//! [`ACTION_SYNC_TEAM`] authority for the team public key. The authorized team
//! fixes the only inventory; a client never supplies another scope or
//! turns local quality-of-service choices into authority.
//!
//! Full-team views preserve every structurally canonical collection record and
//! capability proof. Signature-invalid or otherwise semantically unusable
//! values remain inert evidence: inventory presence grants no authority and is
//! not proof of liveness, reachability, retention, or revocation state.

use anyhow::{Context, Result, bail};
use hifitime::Epoch;
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::capability::{
    CapabilityAtom, CapabilityMode, CapabilityProof, CapabilityProofBundle, CapabilityRequest,
    CapabilityResource, CapabilityValidity,
};
use triblespace_core::collection::{CollectionRead, CollectionRecord};
use triblespace_core::id::{Id, id_hex};
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{Blake3Merkle, Entry as PatchEntry, IdentitySchema, PATCH};
use triblespace_core::repo::peer::PeerEvidence;
use triblespace_core::repo::{
    BlobStoreGet, BlobStoreList, CapabilityProofRead, PeerRead, SnapshotSource,
};

/// Permission to synchronize one server-selected team inventory.
///
/// Minted on 2026-08-26 CEST with the exact command `trible genid`, whose
/// output was `8A421ADA00BAD095FA912070DC696EB1`.
///
/// The capability resource for a full-team inventory is the exact 32-byte team
/// public key. This action is independent of
/// [`crate::protocol::ACTION_CONNECT`]: CONNECT grants no inventory
/// disclosure or reconciliation authority.
pub const ACTION_SYNC_TEAM: Id = id_hex!("8A421ADA00BAD095FA912070DC696EB1");

/// Exact capability atom required to reconcile the full-team inventory.
pub fn sync_team_capability_atom(team: ed25519_dalek::VerifyingKey) -> CapabilityAtom {
    CapabilityAtom::new(
        ACTION_SYNC_TEAM.into(),
        CapabilityResource::new(team.to_bytes()),
    )
}

/// Fixed server-side key prefix beneath which one component is exposed.
///
/// Protocol prefixes and representative keys are always relative to this
/// base. A full-team PEER walk therefore starts at the existing global
/// `team || peer` PATCH subtree under `team` and carries only the 32-byte peer
/// suffix. The remaining components expose their complete trees from an empty
/// base. This convention makes every locator unambiguous without rebuilding a
/// second PEER key space.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum InventoryBasePrefix {
    /// Component root is the full PATCH root.
    Empty,
    /// PEER component root is the subtree under this exact team key.
    PeerTeam([u8; 32]),
}

impl InventoryBasePrefix {
    /// Exact fixed bytes prepended to every protocol-relative key.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::PeerTeam(team) => team,
        }
    }

    /// Convert a protocol-relative key to the canonical full PATCH key.
    pub(crate) fn absolute_key(
        self,
        component: InventoryComponent,
        relative: &[u8],
    ) -> Result<Vec<u8>> {
        if relative.len() != component.relative_key_len(self) {
            bail!(
                "inventory relative key has length {}; expected {} for {:?}",
                relative.len(),
                component.relative_key_len(self),
                component
            );
        }
        let mut absolute = Vec::with_capacity(component.key_len());
        absolute.extend_from_slice(self.as_bytes());
        absolute.extend_from_slice(relative);
        Ok(absolute)
    }

    /// Borrow the protocol-relative suffix of one canonical full PATCH key.
    pub(crate) fn relative_key<'a>(
        self,
        component: InventoryComponent,
        absolute: &'a [u8],
    ) -> Result<&'a [u8]> {
        if absolute.len() != component.key_len() {
            bail!(
                "inventory absolute key has length {}; expected {} for {:?}",
                absolute.len(),
                component.key_len(),
                component
            );
        }
        let base = self.as_bytes();
        if !absolute.starts_with(base) {
            bail!("inventory key is outside its authorized component base");
        }
        Ok(&absolute[base.len()..])
    }
}

/// Server-side policy for inventory authorization.
///
/// The team key is both the external capability trust root and the exact
/// full-team capability resource. The configured backing store is the
/// single-team boundary: PEER keys are intrinsically team-prefixed, while
/// collection-record, capability-proof, and blob identities carry no team
/// field and therefore expose the store's complete sets after authorization.
/// A host must never attach a multi-team store to this configuration. There is
/// no ambient replica-set identifier or separately selectable sync scope.
#[derive(Clone, Copy, Debug)]
pub struct InventoryServerConfig {
    team: ed25519_dalek::VerifyingKey,
}

impl InventoryServerConfig {
    /// Serve the complete structurally canonical inventory for `team` from a
    /// backing store dedicated to that team.
    pub const fn full_team(team: ed25519_dalek::VerifyingKey) -> Self {
        Self { team }
    }

    /// Exact team trust root and synchronization scope.
    pub const fn team(self) -> ed25519_dalek::VerifyingKey {
        self.team
    }

    /// Verify one connection's explicit inventory credential.
    ///
    /// The expected proof leaf is the authenticated transport peer. Successful
    /// verification happens once per connection; later manifest and node
    /// requests use the returned team session instead of resending authority.
    pub fn authorize(
        self,
        peer: ed25519_dalek::VerifyingKey,
        proof: &CapabilityProofBundle,
        now: Epoch,
    ) -> Result<AuthorizedInventorySession> {
        let verified = proof
            .verify(
                self.team,
                now,
                peer,
                CapabilityRequest::new(
                    sync_team_capability_atom(self.team),
                    CapabilityMode::Invoke,
                ),
            )
            .context("verify inventory reconciliation capability")?;
        Ok(AuthorizedInventorySession {
            team: self.team,
            validity: verified.effective_validity(),
        })
    }
}

/// Connection-local result of explicit inventory authorization.
///
/// A host must stop serving this session when its effective proof validity
/// expires. Reauthorization selects a fresh connection session; it never
/// silently falls back to an unauthenticated current snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorizedInventorySession {
    team: ed25519_dalek::VerifyingKey,
    validity: Option<CapabilityValidity>,
}

impl AuthorizedInventorySession {
    /// Exact team trust root and inventory scope.
    pub const fn team(self) -> ed25519_dalek::VerifyingKey {
        self.team
    }

    /// Fixed PATCH prefix selected for `component`.
    pub(crate) fn base_prefix(self, component: InventoryComponent) -> InventoryBasePrefix {
        component.base_prefix(self.team)
    }

    /// Whether the effective proof interval still contains `now`.
    pub fn is_current_at(self, now: Epoch) -> bool {
        self.validity.is_none_or(|validity| validity.contains(now))
    }
}

/// One Merkle-reconciled component of the inventory product.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum InventoryComponent {
    /// Canonical `PEER(team, peer)` routing evidence keyed by its 64-byte body.
    Peer = 1,
    /// Canonical collection records keyed by their 16-byte intrinsic ids.
    CollectionRecord = 2,
    /// Canonical complete proofs keyed by their 32-byte content identities.
    CapabilityProof = 3,
    /// Resident blobs keyed by their 32-byte BLAKE3 handles.
    Blob = 4,
}

impl InventoryComponent {
    /// Canonical manifest order.
    pub const ALL: [Self; 4] = [
        Self::Peer,
        Self::CollectionRecord,
        Self::CapabilityProof,
        Self::Blob,
    ];

    /// Decode the stable wire tag.
    pub fn from_byte(byte: u8) -> Result<Self> {
        match byte {
            1 => Ok(Self::Peer),
            2 => Ok(Self::CollectionRecord),
            3 => Ok(Self::CapabilityProof),
            4 => Ok(Self::Blob),
            _ => bail!("unknown inventory component {byte:#x}"),
        }
    }

    /// Canonical full PATCH key length.
    pub const fn key_len(self) -> usize {
        match self {
            Self::Peer => 64,
            Self::CollectionRecord => 16,
            Self::CapabilityProof | Self::Blob => 32,
        }
    }

    /// Wire-key length relative to the authorized component base.
    pub(crate) fn relative_key_len(self, base: InventoryBasePrefix) -> usize {
        self.key_len()
            .checked_sub(base.as_bytes().len())
            .expect("an authorized base cannot exceed its component key")
    }

    pub(crate) fn base_prefix(self, team: ed25519_dalek::VerifyingKey) -> InventoryBasePrefix {
        match self {
            Self::Peer => InventoryBasePrefix::PeerTeam(team.to_bytes()),
            Self::CollectionRecord | Self::CapabilityProof | Self::Blob => {
                InventoryBasePrefix::Empty
            }
        }
    }

    pub(crate) const fn index(self) -> usize {
        self as usize - 1
    }
}

/// Authenticated root advertised for one immutable component snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComponentManifest {
    component: InventoryComponent,
    leaf_count: u64,
    root: Option<[u8; 32]>,
}

impl ComponentManifest {
    pub(crate) fn new(
        component: InventoryComponent,
        leaf_count: u64,
        root: Option<[u8; 32]>,
    ) -> Self {
        debug_assert_eq!(root.is_none(), leaf_count == 0);
        Self {
            component,
            leaf_count,
            root,
        }
    }

    pub(crate) fn from_wire(
        component: InventoryComponent,
        leaf_count: u64,
        root: Option<[u8; 32]>,
    ) -> Result<Self> {
        if root.is_none() != (leaf_count == 0) {
            bail!("empty inventory root and leaf count disagree");
        }
        Ok(Self {
            component,
            leaf_count,
            root,
        })
    }

    /// Component described by this entry.
    pub const fn component(self) -> InventoryComponent {
        self.component
    }

    /// Authenticated number of leaves expected below the root.
    pub const fn leaf_count(self) -> u64 {
        self.leaf_count
    }

    /// Canonical PATCH Merkle root; `None` is the unique empty set.
    pub const fn root(self) -> Option<[u8; 32]> {
        self.root
    }
}

const GENERATION_DOMAIN: &[u8] = b"triblespace.inventory.generation\0";
const GENERATION_VERSION: u32 = 1;

/// Content-derived identity for one complete four-component manifest.
///
/// This binds the team, roots, and leaf counts served in one immutable view.
/// It is a cache identity, not evidence of global completeness.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InventoryGeneration([u8; 32]);

impl InventoryGeneration {
    fn derive(team: ed25519_dalek::VerifyingKey, components: &[ComponentManifest; 4]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(GENERATION_DOMAIN);
        hasher.update(&GENERATION_VERSION.to_be_bytes());
        hasher.update(&team.to_bytes());
        for component in components {
            hasher.update(&[component.component as u8]);
            hasher.update(&component.leaf_count.to_be_bytes());
            match component.root {
                None => {
                    hasher.update(&[0]);
                }
                Some(root) => {
                    hasher.update(&[1]);
                    hasher.update(&root);
                }
            }
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// Return the portable manifest-generation bytes.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// Root summary for the four authorized inventory components.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InventoryManifest {
    generation: InventoryGeneration,
    components: [ComponentManifest; 4],
}

impl InventoryManifest {
    pub(crate) fn new(
        team: ed25519_dalek::VerifyingKey,
        components: [ComponentManifest; 4],
    ) -> Self {
        let generation = InventoryGeneration::derive(team, &components);
        Self {
            generation,
            components,
        }
    }

    pub(crate) fn from_wire(
        team: ed25519_dalek::VerifyingKey,
        generation: InventoryGeneration,
        components: [ComponentManifest; 4],
    ) -> Result<Self> {
        for (expected, component) in InventoryComponent::ALL.into_iter().zip(components) {
            if component.component != expected {
                bail!("inventory manifest components are out of canonical order");
            }
        }
        let expected = InventoryGeneration::derive(team, &components);
        if generation != expected {
            bail!("inventory generation does not bind its team and component roots");
        }
        Ok(Self {
            generation,
            components,
        })
    }

    /// Content-derived identity of this exact manifest.
    pub const fn generation(self: &Self) -> InventoryGeneration {
        self.generation
    }

    /// Root entry for `component`.
    pub const fn component(&self, component: InventoryComponent) -> ComponentManifest {
        self.components[component.index()]
    }

    /// Entries in canonical component order.
    pub const fn components(&self) -> &[ComponentManifest; 4] {
        &self.components
    }
}

/// Local direction policy for periodic reconciliation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReconcileDirection {
    /// Pull remote inventories and serve the local inventory.
    #[default]
    Bidirectional,
    /// Pull remote inventories without serving local data.
    ReadOnly,
    /// Serve local data without pulling or exact-WANT fetching.
    WriteOnly,
}

impl ReconcileDirection {
    /// Whether the local scheduler should initiate authenticated walks.
    pub const fn pulls(self) -> bool {
        !matches!(self, Self::WriteOnly)
    }

    /// Whether authenticated peers may read the local inventory and blobs.
    pub const fn serves(self) -> bool {
        !matches!(self, Self::ReadOnly)
    }

    /// Whether an inbound reader becomes durable routing evidence locally.
    /// A write-only publisher has no reason to learn clients as pull routes.
    pub const fn admits_inbound_peer(self) -> bool {
        matches!(self, Self::Bidirectional)
    }
}

/// Local blob synchronization policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BlobReconcileMode {
    /// Skip blob inventory. Durable blob WANTs use team-scoped DHT provider
    /// lookup followed by authenticated `GET_BLOB`, independently of the walk.
    #[default]
    Demand,
    /// Traverse every blob key in the authorized inventory and fetch missing bytes.
    ///
    /// Mirror describes synchronization work, not retention. On an evicting
    /// store such as Yard it is best-effort cache residency: later eviction is
    /// allowed and a future sweep may refetch the blob. A durable mirror needs
    /// a separate non-evicting sink or explicit retention contract.
    Mirror,
}

/// Local-only reconciliation policy.
///
/// These values are never sent as authorization inputs and cannot widen a
/// server-selected inventory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReconcileQos {
    /// Whether this peer pulls, serves, or does both.
    pub direction: ReconcileDirection,
    /// Demand-driven or full-residency blob behavior.
    pub blobs: BlobReconcileMode,
}

impl ReconcileQos {
    /// Whether the local scheduler traverses this component.
    pub const fn traverses(self, component: InventoryComponent) -> bool {
        self.direction.pulls()
            && (!matches!(component, InventoryComponent::Blob)
                || matches!(self.blobs, BlobReconcileMode::Mirror))
    }
}

impl Default for ReconcileQos {
    fn default() -> Self {
        Self {
            direction: ReconcileDirection::Bidirectional,
            blobs: BlobReconcileMode::Demand,
        }
    }
}

/// Merkle inventory for canonical peer-routing evidence.
pub type PeerInventory = PATCH<64, IdentitySchema, (), Blake3Merkle>;
/// Merkle inventory for canonical collection records.
///
/// Values do not participate in PATCH hashing: the portable Merkle identity
/// remains exactly the set of intrinsic record ids. Keeping the canonical
/// body in the same immutable leaf lets a pinned walk resolve it without a
/// second snapshot or lookup structure.
pub type CollectionRecordInventory = PATCH<16, IdentitySchema, CollectionRecord, Blake3Merkle>;
/// Merkle inventory for canonical complete capability proofs.
///
/// As with records, the authenticated digest commits only to proof ids. The
/// value is frozen local serving state and is checked against its key during
/// construction and leaf resolution.
pub type CapabilityProofInventory = PATCH<32, IdentitySchema, CapabilityProof, Blake3Merkle>;
/// Merkle inventory for resident blob handles.
pub type BlobInventory = PATCH<32, IdentitySchema, (), Blake3Merkle>;

/// Build one key-only BLAKE3 inventory.
///
/// Keeping construction behind this helper isolates store observation from
/// PATCH's canonical bottom-up snapshot construction.
pub(crate) fn build_key_inventory<const KEY_LEN: usize>(
    keys: impl IntoIterator<Item = [u8; KEY_LEN]>,
) -> PATCH<KEY_LEN, IdentitySchema, (), Blake3Merkle> {
    PATCH::from_keys(keys)
}

pub(crate) fn build_record_inventory(
    records: impl IntoIterator<Item = CollectionRecord>,
) -> Result<CollectionRecordInventory> {
    let mut records: Vec<_> = records
        .into_iter()
        .map(|record| (record.id().raw(), record))
        .collect();
    records.sort_unstable_by_key(|(key, _)| *key);

    let mut inventory = CollectionRecordInventory::new();
    for (key, record) in records {
        if let Some(existing) = inventory.get(&key) {
            if existing != &record {
                bail!("collection record id collision while freezing inventory");
            }
            continue;
        }
        inventory.insert(&PatchEntry::with_value(&key, record));
    }
    Ok(inventory)
}

pub(crate) fn build_proof_inventory(
    proofs: impl IntoIterator<Item = CapabilityProof>,
) -> Result<CapabilityProofInventory> {
    // CapabilityProof::id() hashes the complete proof body. Cache it once;
    // asking the sort comparator to recompute it would make freezing cost
    // O(total proof bytes * log n).
    let mut proofs: Vec<_> = proofs
        .into_iter()
        .map(|proof| (proof.id().raw, proof))
        .collect();
    proofs.sort_unstable_by_key(|(key, _)| *key);

    let mut inventory = CapabilityProofInventory::new();
    for (key, proof) in proofs {
        if let Some(existing) = inventory.get(&key) {
            if existing != &proof {
                bail!("capability proof id collision while freezing inventory");
            }
            continue;
        }
        inventory.insert(&PatchEntry::with_value(&key, proof));
    }
    Ok(inventory)
}

/// Immutable authorized observation of all four inventory components.
///
/// [`InventorySnapshot::from_store`] freezes one coherent store snapshot before
/// deriving any component. Components remain independently addressable by
/// their Merkle roots without reopening the store or crossing that boundary.
pub struct InventorySnapshot<R> {
    team: ed25519_dalek::VerifyingKey,
    store_snapshot: R,
    blobs: BlobInventory,
    manifest: InventoryManifest,
}

impl InventorySnapshot<()> {
    /// Observe a single-team store and construct its fixed full inventory.
    ///
    /// The caller must dedicate `store` to `team`. Records, proofs, and blobs
    /// have content identities but no intrinsic team label, so authorizing this
    /// snapshot intentionally discloses their complete sets.
    pub fn from_store<S>(
        store: &mut S,
        team: ed25519_dalek::VerifyingKey,
    ) -> Result<InventorySnapshot<S::Snapshot>>
    where
        S: SnapshotSource,
        S::Snapshot: BlobStoreGet + BlobStoreList + CollectionRead + CapabilityProofRead + PeerRead,
    {
        let store_snapshot = store.snapshot().map_err(anyhow::Error::new)?;
        let peer_keys = {
            let iterator = store_snapshot.peers().map_err(anyhow::Error::new)?;
            iterator
                .map(|evidence| evidence.map(|evidence| *evidence.as_bytes()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::new)?
        };
        let records = {
            let iterator = store_snapshot.records().map_err(anyhow::Error::new)?;
            iterator
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::new)?
        };
        let proofs = {
            let iterator = store_snapshot.proofs().map_err(anyhow::Error::new)?;
            iterator
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::new)?
        };
        let blob_keys = store_snapshot
            .blobs()
            .map(|info| info.map(|info| info.handle.raw))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;
        InventorySnapshot::from_observation_parts(
            team,
            store_snapshot,
            peer_keys,
            records,
            proofs,
            blob_keys,
        )
    }
}

impl<R> InventorySnapshot<R>
where
    R: BlobStoreList,
{
    /// Build an inventory from already-observed component values.
    ///
    /// Backend ordering is ignored and duplicate identities collapse. Merkle
    /// digests authenticate keys only; record/proof bodies are frozen as
    /// associated PATCH values and checked against those keys, while blob
    /// bytes are retrieved from the pinned store snapshot only when requested.
    pub fn from_observation(
        team: ed25519_dalek::VerifyingKey,
        store_snapshot: R,
        peers: impl IntoIterator<Item = PeerEvidence>,
        records: impl IntoIterator<Item = CollectionRecord>,
        proofs: impl IntoIterator<Item = CapabilityProof>,
    ) -> Result<Self> {
        let peer_keys = peers.into_iter().map(|evidence| *evidence.as_bytes());
        let blob_keys = store_snapshot
            .blobs()
            .map(|info| info.map(|info| info.handle.raw))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(anyhow::Error::new)?;

        Self::from_observation_parts(team, store_snapshot, peer_keys, records, proofs, blob_keys)
    }

    fn from_observation_parts(
        team: ed25519_dalek::VerifyingKey,
        store_snapshot: R,
        peer_keys: impl IntoIterator<Item = [u8; 64]>,
        records: impl IntoIterator<Item = CollectionRecord>,
        proofs: impl IntoIterator<Item = CapabilityProof>,
        blob_keys: impl IntoIterator<Item = [u8; 32]>,
    ) -> Result<Self> {
        let peer_inventory = build_key_inventory(peer_keys);
        let record_inventory = build_record_inventory(records)?;
        let proof_inventory = build_proof_inventory(proofs)?;
        let blob_inventory = build_key_inventory(blob_keys);

        let peer_root = peer_inventory.merkle_node(team.as_bytes());

        let components = [
            ComponentManifest::new(
                InventoryComponent::Peer,
                peer_root.map_or(0, |node| node.leaf_count()),
                peer_root.map(|node| node.digest()),
            ),
            ComponentManifest::new(
                InventoryComponent::CollectionRecord,
                record_inventory.len(),
                record_inventory.merkle_root(),
            ),
            ComponentManifest::new(
                InventoryComponent::CapabilityProof,
                proof_inventory.len(),
                proof_inventory.merkle_root(),
            ),
            ComponentManifest::new(
                InventoryComponent::Blob,
                blob_inventory.len(),
                blob_inventory.merkle_root(),
            ),
        ];
        let manifest = InventoryManifest::new(team, components);

        Ok(Self {
            team,
            store_snapshot,
            blobs: blob_inventory,
            manifest,
        })
    }

    /// Exact team scope captured by this snapshot.
    pub const fn team(&self) -> ed25519_dalek::VerifyingKey {
        self.team
    }

    /// Exact four-component root manifest.
    pub const fn manifest(&self) -> &InventoryManifest {
        &self.manifest
    }

    /// Frozen store snapshot used to serve a blob named by this inventory.
    pub const fn store_snapshot(&self) -> &R {
        &self.store_snapshot
    }
}

impl<R> InventorySnapshot<R>
where
    R: BlobStoreGet,
{
    /// Read one exact blob only if it belongs to this pinned inventory.
    pub fn blob_bytes(&self, hash: [u8; 32]) -> Option<anybytes::Bytes> {
        if self.blobs.get(&hash).is_none() {
            return None;
        }
        self.store_snapshot
            .get::<anybytes::Bytes, UnknownBlob>(Inline::<Handle<UnknownBlob>>::new(hash))
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use hifitime::Epoch;
    use triblespace_core::blob::encodings::UnknownBlob;
    use triblespace_core::capability::{CapabilityClaim, CapabilityMode};
    use triblespace_core::collection::{
        CollectionCommit, CollectionData, CollectionRecord, CollectionStore,
    };
    use triblespace_core::inline::Inline;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStorePut, PeerStore};

    use super::*;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn proof(
        root: &SigningKey,
        leaf: &SigningKey,
        atom: CapabilityAtom,
        mode: CapabilityMode,
    ) -> CapabilityProofBundle {
        CapabilityProofBundle::issue_root(
            root,
            CapabilityClaim::root(atom, mode, None),
            leaf.verifying_key(),
        )
        .unwrap()
    }

    fn commit(author: &SigningKey, byte: u8) -> CollectionRecord {
        CollectionRecord::Commit(CollectionCommit::sign(
            author,
            Inline::new([0x31; 32]),
            CollectionData::new([byte; 32]),
            Inline::new([0x32; 32]),
        ))
    }

    #[test]
    fn reconcile_authority_is_exact_team_scoped_and_distinct_from_connect() {
        let team = key(1);
        let peer = key(2);
        let other_team = key(3);
        let server = InventoryServerConfig::full_team(team.verifying_key());
        let reconcile = proof(
            &team,
            &peer,
            sync_team_capability_atom(team.verifying_key()),
            CapabilityMode::Invoke,
        );

        let session = server
            .authorize(
                peer.verifying_key(),
                &reconcile,
                Epoch::from_tai_seconds(0.0),
            )
            .unwrap();
        assert_eq!(session.team(), team.verifying_key());

        // A principal allowed to invoke and delegate remains an invoker. Team
        // founders and delegating invitees use this mode in normal operation.
        let delegating_reconcile = proof(
            &team,
            &peer,
            sync_team_capability_atom(team.verifying_key()),
            CapabilityMode::InvokeAndDelegate,
        );
        server
            .authorize(
                peer.verifying_key(),
                &delegating_reconcile,
                Epoch::from_tai_seconds(0.0),
            )
            .unwrap();

        let delegate_only = proof(
            &team,
            &peer,
            sync_team_capability_atom(team.verifying_key()),
            CapabilityMode::Delegate,
        );
        assert!(
            server
                .authorize(
                    peer.verifying_key(),
                    &delegate_only,
                    Epoch::from_tai_seconds(0.0),
                )
                .is_err()
        );

        let connect = proof(
            &team,
            &peer,
            crate::protocol::connect_capability_atom(team.verifying_key()),
            CapabilityMode::Invoke,
        );
        assert!(
            server
                .authorize(peer.verifying_key(), &connect, Epoch::from_tai_seconds(0.0))
                .is_err()
        );

        let wrong_resource = proof(
            &team,
            &peer,
            sync_team_capability_atom(other_team.verifying_key()),
            CapabilityMode::Invoke,
        );
        assert!(
            server
                .authorize(
                    peer.verifying_key(),
                    &wrong_resource,
                    Epoch::from_tai_seconds(0.0)
                )
                .is_err()
        );
    }

    #[test]
    fn authorized_team_fixes_the_peer_inventory_base() {
        let team = key(1).verifying_key();
        let session = AuthorizedInventorySession {
            team,
            validity: None,
        };
        assert_eq!(
            session.base_prefix(InventoryComponent::Peer).as_bytes(),
            team.as_bytes()
        );
        assert!(
            session
                .base_prefix(InventoryComponent::CollectionRecord)
                .as_bytes()
                .is_empty()
        );
    }

    #[test]
    fn full_team_manifest_scopes_global_peer_tree_and_preserves_inert_evidence() {
        let team = key(1);
        let other_team = key(2);
        let peer = key(3);
        let second_peer = key(5);
        let author = key(4);
        let mut store = MemoryRepo::default();
        store
            .insert_peer(PeerEvidence::new(
                team.verifying_key(),
                peer.verifying_key(),
            ))
            .unwrap();
        store
            .insert_peer(PeerEvidence::new(
                team.verifying_key(),
                second_peer.verifying_key(),
            ))
            .unwrap();
        store
            .insert_peer(PeerEvidence::new(
                other_team.verifying_key(),
                peer.verifying_key(),
            ))
            .unwrap();

        let valid = commit(&author, 7);
        let mut invalid_bytes = valid.to_bytes();
        let last = invalid_bytes.len() - 1;
        invalid_bytes[last] ^= 0x80;
        let invalid = CollectionRecord::from_bytes(&invalid_bytes).unwrap();
        let CollectionRecord::Commit(invalid_commit) = invalid else {
            unreachable!("fixture is a commit")
        };
        assert!(invalid_commit.verify_strict().is_err());
        CollectionStore::insert(&mut store, invalid).unwrap();

        let snapshot = InventorySnapshot::from_store(&mut store, team.verifying_key()).unwrap();
        assert_eq!(
            snapshot
                .manifest()
                .component(InventoryComponent::Peer)
                .leaf_count(),
            2
        );
        assert_eq!(
            snapshot
                .manifest()
                .component(InventoryComponent::CollectionRecord)
                .leaf_count(),
            1
        );

        let base = InventoryComponent::Peer.base_prefix(team.verifying_key());
        assert_eq!(base.as_bytes(), team.verifying_key().as_bytes());
        assert_eq!(
            base.absolute_key(InventoryComponent::Peer, peer.verifying_key().as_bytes())
                .unwrap(),
            PeerEvidence::new(team.verifying_key(), peer.verifying_key()).to_bytes()
        );

        let mut team_only = MemoryRepo::default();
        // Reverse insertion order: the PEER root is independent of history and
        // of unrelated teams in the global PEER key space.
        team_only
            .insert_peer(PeerEvidence::new(
                team.verifying_key(),
                second_peer.verifying_key(),
            ))
            .unwrap();
        team_only
            .insert_peer(PeerEvidence::new(
                team.verifying_key(),
                peer.verifying_key(),
            ))
            .unwrap();
        let team_only_snapshot =
            InventorySnapshot::from_store(&mut team_only, team.verifying_key()).unwrap();
        assert_eq!(
            snapshot.manifest().component(InventoryComponent::Peer),
            team_only_snapshot
                .manifest()
                .component(InventoryComponent::Peer)
        );
    }

    #[test]
    fn component_roots_are_history_independent_and_isolate_blob_churn() {
        let team = key(1);
        let author = key(2);
        let record_a = commit(&author, 1);
        let record_b = commit(&author, 2);

        let mut left = MemoryRepo::default();
        CollectionStore::insert(&mut left, record_a).unwrap();
        CollectionStore::insert(&mut left, record_b).unwrap();
        let left_snapshot = InventorySnapshot::from_store(&mut left, team.verifying_key()).unwrap();

        let mut right = MemoryRepo::default();
        CollectionStore::insert(&mut right, record_b).unwrap();
        CollectionStore::insert(&mut right, record_a).unwrap();
        let right_snapshot =
            InventorySnapshot::from_store(&mut right, team.verifying_key()).unwrap();
        assert_eq!(left_snapshot.manifest(), right_snapshot.manifest());

        let mut with_blob = MemoryRepo::default();
        CollectionStore::insert(&mut with_blob, record_b).unwrap();
        CollectionStore::insert(&mut with_blob, record_a).unwrap();
        with_blob
            .put::<UnknownBlob, _>(anybytes::Bytes::from_source(b"resident".to_vec()))
            .unwrap();
        let blob_snapshot =
            InventorySnapshot::from_store(&mut with_blob, team.verifying_key()).unwrap();

        for component in [
            InventoryComponent::Peer,
            InventoryComponent::CollectionRecord,
            InventoryComponent::CapabilityProof,
        ] {
            assert_eq!(
                left_snapshot.manifest().component(component),
                blob_snapshot.manifest().component(component)
            );
        }
        assert_ne!(
            left_snapshot.manifest().component(InventoryComponent::Blob),
            blob_snapshot.manifest().component(InventoryComponent::Blob)
        );
        assert_ne!(
            left_snapshot.manifest().generation(),
            blob_snapshot.manifest().generation()
        );
    }

    #[test]
    fn demand_skips_only_blob_inventory_and_write_only_skips_all_walks() {
        let demand = ReconcileQos::default();
        assert!(demand.traverses(InventoryComponent::Peer));
        assert!(demand.traverses(InventoryComponent::CollectionRecord));
        assert!(demand.traverses(InventoryComponent::CapabilityProof));
        assert!(!demand.traverses(InventoryComponent::Blob));

        let mirror = ReconcileQos {
            blobs: BlobReconcileMode::Mirror,
            ..demand
        };
        assert!(mirror.traverses(InventoryComponent::Blob));

        let write_only = ReconcileQos {
            direction: ReconcileDirection::WriteOnly,
            ..mirror
        };
        assert!(
            InventoryComponent::ALL
                .into_iter()
                .all(|component| !write_only.traverses(component))
        );
    }
}
