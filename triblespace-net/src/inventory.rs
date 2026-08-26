//! Authorized, componentwise inventory views for periodic anti-entropy.
//!
//! Inventory synchronization is a single protocol over four grow-only key
//! sets: [`InventoryComponent::Peer`],
//! [`InventoryComponent::CollectionRecord`],
//! [`InventoryComponent::CapabilityProof`], and [`InventoryComponent::Blob`].
//! A CONNECT proof admits a transport connection only. Before inventory data
//! is disclosed, the remote transport key must separately prove exact
//! [`ACTION_SYNC_TEAM`] authority for the team public key. The server then
//! selects the projection and derives its [`AuthorizedViewId`]; a client never
//! supplies a view id or turns local quality-of-service choices into authority.
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
use triblespace_core::collection::{CollectionRecord, CollectionStore};
use triblespace_core::id::{Id, id_hex};
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::patch::{Blake3Merkle, Entry, IdentitySchema, PATCH};
use triblespace_core::repo::peer::PeerEvidence;
use triblespace_core::repo::{
    BlobStore, BlobStoreGet, BlobStoreList, CapabilityProofStore, PeerStore,
};

/// Permission to synchronize one server-selected inventory view of a team.
///
/// Minted on 2026-08-26 CEST with the exact command `trible genid`, whose
/// output was `8A421ADA00BAD095FA912070DC696EB1`.
///
/// The capability resource for a full-team view is the exact 32-byte team
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

/// A server-owned inventory projection.
///
/// Projection is an authorization decision, not a quality-of-service knob.
/// Only the complete team evidence store is implemented today. A future
/// projection with genuinely different disclosure semantics must receive its
/// own content-derived descriptor and authorization policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InventoryProjection {
    /// Every structurally canonical resident item for the team.
    FullTeam,
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
pub enum InventoryBasePrefix {
    /// Component root is the full PATCH root.
    Empty,
    /// PEER component root is the subtree under this exact team key.
    PeerTeam([u8; 32]),
}

impl InventoryBasePrefix {
    /// Exact fixed bytes prepended to every protocol-relative key.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::PeerTeam(team) => team,
        }
    }

    /// Convert a protocol-relative key to the canonical full PATCH key.
    pub fn absolute_key(self, component: InventoryComponent, relative: &[u8]) -> Result<Vec<u8>> {
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
    pub fn relative_key<'a>(
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

impl InventoryProjection {
    pub(crate) const fn wire_tag(self) -> u8 {
        match self {
            Self::FullTeam => 1,
        }
    }

    pub(crate) fn from_wire_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::FullTeam),
            _ => bail!("unknown authorized inventory projection {tag:#x}"),
        }
    }
}

const VIEW_ID_DOMAIN: &[u8] = b"triblespace.inventory.authorized-view\0";
const VIEW_ID_VERSION: u32 = 1;

/// Content-derived cache identity of one server-selected authorized view.
///
/// This is not a user-managed replica-set name and is never a capability
/// resource. The full-team identity is derived from the exact team public key
/// and the fixed projection semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AuthorizedViewId([u8; 32]);

impl AuthorizedViewId {
    fn derive(team: ed25519_dalek::VerifyingKey, projection: InventoryProjection) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(VIEW_ID_DOMAIN);
        hasher.update(&VIEW_ID_VERSION.to_be_bytes());
        hasher.update(&team.to_bytes());
        hasher.update(&[projection.wire_tag()]);
        Self(*hasher.finalize().as_bytes())
    }

    /// Return the portable cache-key bytes.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// The exact inventory projection selected by a server after authorization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorizedInventoryView {
    id: AuthorizedViewId,
    team: ed25519_dalek::VerifyingKey,
    projection: InventoryProjection,
}

impl AuthorizedInventoryView {
    pub(crate) fn full_team(team: ed25519_dalek::VerifyingKey) -> Self {
        let projection = InventoryProjection::FullTeam;
        Self {
            id: AuthorizedViewId::derive(team, projection),
            team,
            projection,
        }
    }

    /// Server-derived identity used to pin manifests and component snapshots.
    pub const fn id(self) -> AuthorizedViewId {
        self.id
    }

    /// Team whose evidence can appear in this view.
    pub const fn team(self) -> ed25519_dalek::VerifyingKey {
        self.team
    }

    /// Disclosure projection selected by the server.
    pub const fn projection(self) -> InventoryProjection {
        self.projection
    }

    /// Fixed PATCH prefix selected for `component`.
    ///
    /// The PEER base enforces team isolation structurally. Other full-team
    /// components use their complete roots.
    pub fn base_prefix(self, component: InventoryComponent) -> InventoryBasePrefix {
        match component {
            InventoryComponent::Peer => InventoryBasePrefix::PeerTeam(self.team.to_bytes()),
            InventoryComponent::CollectionRecord
            | InventoryComponent::CapabilityProof
            | InventoryComponent::Blob => InventoryBasePrefix::Empty,
        }
    }
}

/// Server-side policy for inventory authorization.
///
/// The team key is both the external capability trust root and the exact
/// full-team capability resource. There is no ambient replica-set identifier.
#[derive(Clone, Copy, Debug)]
pub struct InventoryServerConfig {
    team: ed25519_dalek::VerifyingKey,
}

impl InventoryServerConfig {
    /// Serve the complete structurally canonical inventory for `team`.
    pub const fn full_team(team: ed25519_dalek::VerifyingKey) -> Self {
        Self { team }
    }

    /// Exact team trust root and synchronization scope.
    pub const fn team(self) -> ed25519_dalek::VerifyingKey {
        self.team
    }

    /// Verify one connection's explicit inventory credential and select the
    /// authorized view.
    ///
    /// The expected proof leaf is the authenticated transport peer. Successful
    /// verification is intended to happen once per connection/view; later
    /// manifest and node requests use the returned session instead of
    /// resending authority on every operation.
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
        if verified.effective_mode() != CapabilityMode::Invoke {
            bail!("inventory reconciliation proof must end in exact invoke-only authority");
        }
        Ok(AuthorizedInventorySession {
            view: AuthorizedInventoryView::full_team(self.team),
            validity: verified.effective_validity(),
        })
    }
}

/// Connection-local result of explicit inventory authorization.
///
/// A host must stop serving this session when its effective proof validity
/// expires. Reauthorization selects a fresh connection/view session; it never
/// silently falls back to an unauthenticated current snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorizedInventorySession {
    view: AuthorizedInventoryView,
    validity: Option<CapabilityValidity>,
}

impl AuthorizedInventorySession {
    /// Server-selected view authorized for this connection.
    pub const fn view(self) -> AuthorizedInventoryView {
        self.view
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
    pub fn relative_key_len(self, base: InventoryBasePrefix) -> usize {
        self.key_len()
            .checked_sub(base.as_bytes().len())
            .expect("an authorized base cannot exceed its component key")
    }

    const fn index(self) -> usize {
        self as usize - 1
    }
}

const COMPONENT_TOKEN_DOMAIN: &[u8] = b"triblespace.inventory.component-token\0";
const COMPONENT_TOKEN_VERSION: u32 = 1;

/// View-scoped identity of one immutable component snapshot.
///
/// Requests pin this token exactly. A stale or evicted token is an ordinary
/// retry condition; servers must never substitute their current component.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InventoryToken([u8; 32]);

impl InventoryToken {
    fn derive(
        view: AuthorizedViewId,
        component: InventoryComponent,
        leaf_count: u64,
        root: Option<[u8; 32]>,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMPONENT_TOKEN_DOMAIN);
        hasher.update(&COMPONENT_TOKEN_VERSION.to_be_bytes());
        hasher.update(&view.0);
        hasher.update(&[component as u8]);
        hasher.update(&leaf_count.to_be_bytes());
        match root {
            None => {
                hasher.update(&[0]);
            }
            Some(root) => {
                hasher.update(&[1]);
                hasher.update(&root);
            }
        };
        Self(*hasher.finalize().as_bytes())
    }

    /// Return the exact portable token bytes.
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// Authenticated root advertised for one immutable component snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComponentManifest {
    component: InventoryComponent,
    token: InventoryToken,
    leaf_count: u64,
    root: Option<[u8; 32]>,
}

impl ComponentManifest {
    pub(crate) fn new(
        view: AuthorizedViewId,
        component: InventoryComponent,
        leaf_count: u64,
        root: Option<[u8; 32]>,
    ) -> Self {
        Self {
            component,
            token: InventoryToken::derive(view, component, leaf_count, root),
            leaf_count,
            root,
        }
    }

    pub(crate) fn from_wire(
        view: AuthorizedViewId,
        component: InventoryComponent,
        leaf_count: u64,
        root: Option<[u8; 32]>,
        token: InventoryToken,
    ) -> Result<Self> {
        let expected = InventoryToken::derive(view, component, leaf_count, root);
        if token != expected {
            bail!("inventory component token does not bind its advertised root");
        }
        Ok(Self {
            component,
            token,
            leaf_count,
            root,
        })
    }

    /// Component described by this entry.
    pub const fn component(self) -> InventoryComponent {
        self.component
    }

    /// View-scoped immutable snapshot token.
    pub const fn token(self) -> InventoryToken {
        self.token
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

/// Content-derived wake generation for one complete four-component manifest.
///
/// Gossip carries this value only as an untrusted hint to schedule an
/// authenticated manifest check. It is not completeness evidence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InventoryGeneration([u8; 32]);

impl InventoryGeneration {
    fn derive(view: AuthorizedViewId, components: &[ComponentManifest; 4]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(GENERATION_DOMAIN);
        hasher.update(&GENERATION_VERSION.to_be_bytes());
        hasher.update(&view.0);
        for component in components {
            hasher.update(&[component.component as u8]);
            hasher.update(&component.token.0);
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// Return the portable wake-generation bytes.
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
    view: AuthorizedViewId,
    generation: InventoryGeneration,
    components: [ComponentManifest; 4],
}

impl InventoryManifest {
    pub(crate) fn new(view: AuthorizedViewId, components: [ComponentManifest; 4]) -> Self {
        let generation = InventoryGeneration::derive(view, &components);
        Self {
            view,
            generation,
            components,
        }
    }

    pub(crate) fn from_wire(
        view: AuthorizedViewId,
        generation: InventoryGeneration,
        components: [ComponentManifest; 4],
    ) -> Result<Self> {
        for (expected, component) in InventoryComponent::ALL.into_iter().zip(components) {
            if component.component != expected {
                bail!("inventory manifest components are out of canonical order");
            }
        }
        let expected = InventoryGeneration::derive(view, &components);
        if generation != expected {
            bail!("inventory generation does not bind its component tokens");
        }
        Ok(Self {
            view,
            generation,
            components,
        })
    }

    /// Server-derived authorized view identity.
    pub const fn view(self: &Self) -> AuthorizedViewId {
        self.view
    }

    /// Untrusted-gossip wake value, authenticated here by the manifest frame.
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
    /// Pull remote inventories and publish local wake hints.
    #[default]
    Bidirectional,
    /// Pull remote inventories but suppress local wake publication.
    ReadOnly,
    /// Publish local wake hints but do not pull remote inventories.
    WriteOnly,
}

impl ReconcileDirection {
    /// Whether the local scheduler should initiate authenticated walks.
    pub const fn pulls(self) -> bool {
        !matches!(self, Self::WriteOnly)
    }

    /// Whether the local scheduler should publish wake hints.
    pub const fn publishes(self) -> bool {
        !matches!(self, Self::ReadOnly)
    }
}

/// Local blob synchronization policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BlobReconcileMode {
    /// Skip blob inventory. Durable blob WANTs use exact DHT discovery and
    /// `GET_BLOB` independently of the periodic inventory walk.
    #[default]
    Demand,
    /// Traverse every blob key in the authorized view and fetch missing bytes.
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
/// server-selected view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReconcileQos {
    /// Whether this peer pulls, publishes wake hints, or both.
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
pub type CollectionRecordInventory = PATCH<16, IdentitySchema, CollectionRecord, Blake3Merkle>;
/// Merkle inventory for canonical complete capability proofs.
pub type CapabilityProofInventory = PATCH<32, IdentitySchema, CapabilityProof, Blake3Merkle>;
/// Merkle inventory for resident blob handles, valued by untrusted length hint.
pub type BlobInventory = PATCH<32, IdentitySchema, u64, Blake3Merkle>;

/// Immutable authorized observation of all four inventory components.
///
/// Components receive independent content tokens. No atomicity across store
/// traits is implied or required: a later blob observation does not invalidate
/// an already-pinned record walk.
pub struct InventorySnapshot<R> {
    view: AuthorizedInventoryView,
    reader: R,
    peers: PeerInventory,
    records: CollectionRecordInventory,
    proofs: CapabilityProofInventory,
    blobs: BlobInventory,
    manifest: InventoryManifest,
}

impl InventorySnapshot<()> {
    /// Observe a store and construct the fixed full-team projection selected
    /// by `view`.
    pub fn from_store<S>(
        store: &mut S,
        view: AuthorizedInventoryView,
    ) -> Result<InventorySnapshot<S::Reader>>
    where
        S: BlobStore + CollectionStore + CapabilityProofStore + PeerStore,
    {
        let peers = {
            let iterator = store.peers().map_err(anyhow::Error::new)?;
            iterator
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::new)?
        };
        let records = {
            let iterator = store.records().map_err(anyhow::Error::new)?;
            iterator
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::new)?
        };
        let proofs = {
            let iterator = store.proofs().map_err(anyhow::Error::new)?;
            iterator
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(anyhow::Error::new)?
        };
        let reader = store.reader().map_err(anyhow::Error::new)?;
        InventorySnapshot::from_observation(view, reader, peers, records, proofs)
    }
}

impl<R> InventorySnapshot<R>
where
    R: BlobStoreGet + BlobStoreList,
{
    /// Build an inventory from already-observed component values.
    ///
    /// Backend ordering is ignored, duplicate identities collapse, and a body
    /// conflict under one intrinsic key is rejected. Blob listing metadata is
    /// not authoritative: only handles whose bytes can be retrieved and whose
    /// observed length matches are advertised.
    pub fn from_observation(
        view: AuthorizedInventoryView,
        reader: R,
        peers: impl IntoIterator<Item = PeerEvidence>,
        records: impl IntoIterator<Item = CollectionRecord>,
        proofs: impl IntoIterator<Item = CapabilityProof>,
    ) -> Result<Self> {
        let mut peer_inventory = PeerInventory::new();
        for evidence in peers {
            if evidence.team() == view.team {
                peer_inventory.insert(&Entry::new(evidence.as_bytes()));
            }
        }

        let mut record_inventory = CollectionRecordInventory::new();
        for record in records {
            let key = record.id().raw();
            if let Some(existing) = record_inventory.get(&key) {
                if existing != &record {
                    bail!(
                        "collection record id {} names conflicting canonical bytes",
                        hex::encode(key)
                    );
                }
                continue;
            }
            record_inventory.insert(&Entry::with_value(&key, record));
        }

        let mut proof_inventory = CapabilityProofInventory::new();
        for proof in proofs {
            let key = proof.id().raw;
            if let Some(existing) = proof_inventory.get(&key) {
                if existing != &proof {
                    bail!(
                        "capability proof id {} names conflicting canonical bytes",
                        hex::encode(key)
                    );
                }
                continue;
            }
            proof_inventory.insert(&Entry::with_value(&key, proof));
        }

        let mut blob_inventory = BlobInventory::new();
        for info in reader.blobs() {
            let info = info.map_err(anyhow::Error::new)?;
            let Ok(bytes) = reader.get::<anybytes::Bytes, UnknownBlob>(info.handle) else {
                continue;
            };
            if bytes.len() as u64 != info.length {
                continue;
            }
            let key = info.handle.raw;
            if let Some(existing) = blob_inventory.get(&key) {
                if *existing != info.length {
                    bail!(
                        "blob {} has conflicting resident lengths {} and {}",
                        hex::encode(key),
                        existing,
                        info.length
                    );
                }
                continue;
            }
            blob_inventory.insert(&Entry::with_value(&key, info.length));
        }

        let components = [
            ComponentManifest::new(
                view.id,
                InventoryComponent::Peer,
                peer_inventory.len(),
                peer_inventory.merkle_root(),
            ),
            ComponentManifest::new(
                view.id,
                InventoryComponent::CollectionRecord,
                record_inventory.len(),
                record_inventory.merkle_root(),
            ),
            ComponentManifest::new(
                view.id,
                InventoryComponent::CapabilityProof,
                proof_inventory.len(),
                proof_inventory.merkle_root(),
            ),
            ComponentManifest::new(
                view.id,
                InventoryComponent::Blob,
                blob_inventory.len(),
                blob_inventory.merkle_root(),
            ),
        ];
        let manifest = InventoryManifest::new(view.id, components);

        Ok(Self {
            view,
            reader,
            peers: peer_inventory,
            records: record_inventory,
            proofs: proof_inventory,
            blobs: blob_inventory,
            manifest,
        })
    }

    /// Authorized projection captured by this snapshot.
    pub const fn view(&self) -> AuthorizedInventoryView {
        self.view
    }

    /// Exact four-component root manifest.
    pub const fn manifest(&self) -> &InventoryManifest {
        &self.manifest
    }

    /// Frozen reader used to serve a blob named by this inventory.
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    pub(crate) const fn peers(&self) -> &PeerInventory {
        &self.peers
    }

    pub(crate) const fn records(&self) -> &CollectionRecordInventory {
        &self.records
    }

    pub(crate) const fn proofs(&self) -> &CapabilityProofInventory {
        &self.proofs
    }

    pub(crate) const fn blobs(&self) -> &BlobInventory {
        &self.blobs
    }

    /// Read one exact blob only if it belongs to this pinned inventory.
    pub fn blob_bytes(&self, hash: [u8; 32]) -> Option<anybytes::Bytes> {
        if self.blobs.get(&hash).is_none() {
            return None;
        }
        self.reader
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
    use triblespace_core::collection::{CollectionCommit, CollectionData, CollectionRecord};
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
        assert_eq!(session.view().team(), team.verifying_key());
        assert_eq!(session.view().projection(), InventoryProjection::FullTeam);

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
    fn server_derives_view_identity_from_projection_and_team() {
        let first = AuthorizedInventoryView::full_team(key(1).verifying_key());
        let same = AuthorizedInventoryView::full_team(key(1).verifying_key());
        let other = AuthorizedInventoryView::full_team(key(2).verifying_key());
        assert_eq!(first.id(), same.id());
        assert_ne!(first.id(), other.id());
    }

    #[test]
    fn full_team_snapshot_filters_peer_scope_but_preserves_inert_evidence() {
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

        let view = AuthorizedInventoryView::full_team(team.verifying_key());
        let snapshot = InventorySnapshot::from_store(&mut store, view).unwrap();
        assert_eq!(snapshot.peers().len(), 2);
        assert_eq!(snapshot.records().len(), 1);
        assert_eq!(
            snapshot
                .manifest()
                .component(InventoryComponent::CollectionRecord)
                .leaf_count(),
            1
        );

        let base = view.base_prefix(InventoryComponent::Peer);
        assert_eq!(base.as_bytes(), team.verifying_key().as_bytes());
        assert_eq!(
            base.absolute_key(InventoryComponent::Peer, peer.verifying_key().as_bytes())
                .unwrap(),
            PeerEvidence::new(team.verifying_key(), peer.verifying_key()).to_bytes()
        );

        let mut team_only = MemoryRepo::default();
        // Reverse insertion order: the view root is independent of history and
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
        let team_only_snapshot = InventorySnapshot::from_store(&mut team_only, view).unwrap();
        assert_eq!(
            snapshot.manifest().component(InventoryComponent::Peer),
            team_only_snapshot
                .manifest()
                .component(InventoryComponent::Peer)
        );
    }

    #[test]
    fn component_tokens_are_history_independent_and_isolate_blob_churn() {
        let team = key(1);
        let author = key(2);
        let view = AuthorizedInventoryView::full_team(team.verifying_key());
        let record_a = commit(&author, 1);
        let record_b = commit(&author, 2);

        let mut left = MemoryRepo::default();
        CollectionStore::insert(&mut left, record_a).unwrap();
        CollectionStore::insert(&mut left, record_b).unwrap();
        let left_snapshot = InventorySnapshot::from_store(&mut left, view).unwrap();

        let mut right = MemoryRepo::default();
        CollectionStore::insert(&mut right, record_b).unwrap();
        CollectionStore::insert(&mut right, record_a).unwrap();
        let right_snapshot = InventorySnapshot::from_store(&mut right, view).unwrap();
        assert_eq!(left_snapshot.manifest(), right_snapshot.manifest());

        let mut with_blob = MemoryRepo::default();
        CollectionStore::insert(&mut with_blob, record_b).unwrap();
        CollectionStore::insert(&mut with_blob, record_a).unwrap();
        with_blob
            .put::<UnknownBlob, _>(anybytes::Bytes::from_source(b"resident".to_vec()))
            .unwrap();
        let blob_snapshot = InventorySnapshot::from_store(&mut with_blob, view).unwrap();

        for component in [
            InventoryComponent::Peer,
            InventoryComponent::CollectionRecord,
            InventoryComponent::CapabilityProof,
        ] {
            assert_eq!(
                left_snapshot.manifest().component(component).token(),
                blob_snapshot.manifest().component(component).token()
            );
        }
        assert_ne!(
            left_snapshot
                .manifest()
                .component(InventoryComponent::Blob)
                .token(),
            blob_snapshot
                .manifest()
                .component(InventoryComponent::Blob)
                .token()
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
