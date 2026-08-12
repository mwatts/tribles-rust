//! Private node policy as one signer-owned, grow-only collection.
//!
//! Policy is operational state, but it is not mutable scalar state. Requests,
//! decisions, capability holdings, renewal versions, retractions, and delivery
//! acknowledgements are immutable facts committed to one collection owned by
//! the node's signing key. Versions form explicit
//! [`metadata::supersedes`](triblespace_core::metadata::supersedes) DAGs.
//! Independently-written pile copies can therefore be unioned without an
//! order-dependent last-writer-wins interpretation: a concurrent disagreement
//! remains an explicit fork and policy consumers fail closed until a later
//! version supersedes every conflicting head.
//!
//! Collection membership is private by default. This module never writes a
//! gossip marker and [`Peer`](crate::peer::Peer) deliberately does not expose a
//! `CollectionStore` implementation. Content-addressed capability blobs may be
//! advertised by the ordinary blob layer, but the signed collection records
//! that confer policy membership are not branch heads and are not announced as
//! policy.

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::collection::{Collection, CollectionStore};
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::encodings::time::NsTAIInterval;
use triblespace_core::macros::{entity, find, pattern};
use triblespace_core::metadata;
use triblespace_core::prelude::attributes;
use triblespace_core::prelude::inlineencodings::{ED25519PublicKey, GenId};
use triblespace_core::repo::{
    BlobStore, BlobStoreList, BlobStoreMeta, BlobStorePut, PinStore, StorageFlush,
};
use triblespace_core::trible::{Fragment, TribleSet};

attributes! {
    /// Legacy marker used by pre-collection policy pins. New policy code never
    /// writes it; branch gossip keeps recognizing it until old pins have been
    /// explicitly migrated or tombstoned.
    "3361F2DE0BD68BA8712EC5B9CCC7EF2A" unsafe as legacy_local_only_pin: GenId;

    /// Names the team whose current credential a version describes.
    "E1EE471B597A4142AD26CA1FED368D2F" unsafe as pub cap_for_team: ED25519PublicKey;
    /// Subject whose issued credential a policy version maintains.
    "384D8A994AF026BBD1329CAD7041E3B8" unsafe as pub policy_subject: ED25519PublicKey;
    /// Stable scope-root identity of an issued credential.
    "D67D3CB1562B27504892BF0ACB55EA8B" unsafe as pub policy_scope: GenId;
    /// Interval covered by the credential in this exact policy version.
    "AEF94EAB060C3D78AE373715885897C0" unsafe as pub policy_issued_at: NsTAIInterval;
    /// Handle of the capability blob in this exact policy version.
    "BF6B9C894E3CA2AB5FBCC12B925C9680" unsafe as pub policy_latest_cap: Handle<SimpleArchive>;
    /// Handle of the signature blob in this exact policy version.
    "5A72B59BF016C7024385B6976BD8AD0E" unsafe as pub policy_latest_sig: Handle<SimpleArchive>;
    /// Present only on a terminal policy version.
    "57C45D022B79C4D3A021AC0114D973EE" unsafe as pub policy_retracted_at: NsTAIInterval;
    /// Timestamp on an acknowledgement event for one exact policy version.
    "2E289E766CFD4F2554D430C31337BE2B" unsafe as pub policy_delivered_at: NsTAIInterval;

    /// Public key that sent a capability request.
    "3583BC29C2155717639FA7E9314CC8B9" unsafe as pub request_requester: ED25519PublicKey;
    /// Exact partial-capability blob supplied by the requester.
    "42903FA16A2913144A48072F575BB304" unsafe as pub request_partial_cap: Handle<SimpleArchive>;
    /// Timestamp on an immutable observation of a stable request.
    "8CC3155E937E416C8CFDC11630E9789E" unsafe as pub request_received_at: NsTAIInterval;
    /// APPROVED or REJECTED on an immutable decision event.
    "4D72D56FF30DA693679F08D629DA7574" unsafe as pub request_status: GenId;

    /// Capability blob held for one team in an exact team-cap version.
    "A2BBD772754BBB8EAFD7479F5A1249FD" unsafe as pub team_cap_handle: Handle<SimpleArchive>;
    /// Signature blob paired with `team_cap_handle`.
    "FAC14D0CAB23B1C7AC20D8CF1C843EBF" unsafe as pub team_sig_handle: Handle<SimpleArchive>;

    // Minted with `trible genid` on 2026-08-12 for the collection-native
    // policy schema.
    /// Stable request named by an observation event.
    "9231A8F3684B10D9782FD1FBED3E43B0" as pub request_observed: GenId;
    /// Stable request named by a decision event.
    "939EE0BAEB35531E794F0E4E005AF20E" as pub request_decides: GenId;
    /// Team root included in a renewal-policy track identity.
    "C111FC414B1BFA5B39C8AC0230D6CAB1" as pub policy_team_root: ED25519PublicKey;
    /// Exact policy version named by a delivery acknowledgement.
    "ECA4E42E6FA84DF3B87FECA69F26303E" as pub policy_acknowledges: GenId;
    /// Request whose approval caused an issued policy version.
    "BE1DD8EC542E0DEC4DD64DB189C85CF8" as pub policy_request: GenId;
}

/// Scope for the node's one private policy collection.
///
/// Minted with `trible genid` on 2026-08-12.
pub const POLICY_COLLECTION_SCOPE: Id =
    triblespace_core::id::id_hex!("8067402E88FE8DBBFA559F2212C2353D");

// Entity-kind ids, minted with `trible genid` on 2026-08-12.
const KIND_REQUEST: Id = triblespace_core::id::id_hex!("35C96F095C003B7CDAE9602276C8D125");
const KIND_REQUEST_OBSERVATION: Id =
    triblespace_core::id::id_hex!("7FC24CCB05D6B99A3CA49F6B5D237A1D");
const KIND_REQUEST_DECISION: Id = triblespace_core::id::id_hex!("4B2C21CC2A145CADA7FE254AF7203DF8");
const KIND_POLICY_VERSION: Id = triblespace_core::id::id_hex!("25EC6D8585681E74DAA91A3F2D5ADC5F");
const KIND_DELIVERY_ACK: Id = triblespace_core::id::id_hex!("2FA2CED1789C8A29FC2B2FF6E6DE48FD");
const KIND_TEAM_CAP_VERSION: Id = triblespace_core::id::id_hex!("53490AE586FDA93510D4BAFC4B487D2F");

/// Derived status for a request that has no decision event.
pub const STATUS_PENDING: Id = triblespace_core::id::id_hex!("08A49DEBF036B127CF60D8B33A7B9B31");
/// Immutable positive request decision.
pub const STATUS_APPROVED: Id = triblespace_core::id::id_hex!("6186747FD38D84D23BA82F3ABE6D9952");
/// Immutable negative request decision.
pub const STATUS_REJECTED: Id = triblespace_core::id::id_hex!("3E54420C1F7EECFCED83203FA749C912");

/// Failure to read, validate, or extend the local policy collection.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("{stage} failed: {detail}")]
    Storage { stage: &'static str, detail: String },
    #[error("malformed policy collection: {0}")]
    Malformed(&'static str),
    #[error("unknown request status {0:?}")]
    UnknownStatus(Id),
    #[error("policy entity {0:?} was not found")]
    NotFound(Id),
    #[error("{domain} has conflicting heads for {identity}")]
    Conflict {
        domain: &'static str,
        identity: String,
    },
    #[error("policy version {expected:?} is stale; current head is {current:?}")]
    Stale { expected: Id, current: Id },
    #[error("policy track ending at {0:?} is retracted and cannot be resumed")]
    Retracted(Id),
    #[error("request {request:?} belongs to a different subject")]
    RequestSubjectMismatch { request: Id },
}

fn storage_error(stage: &'static str, source: impl std::fmt::Display) -> PolicyError {
    PolicyError::Storage {
        stage,
        detail: source.to_string(),
    }
}

fn materialize<S>(store: &mut S, signing_key: &SigningKey) -> Result<TribleSet, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    Collection::new(&mut *store, POLICY_COLLECTION_SCOPE, signing_key.clone())
        .materialize()
        .map_err(|error| storage_error("materializing policy collection", error))
}

fn commit<S>(store: &mut S, signing_key: &SigningKey, fragment: Fragment) -> Result<(), PolicyError>
where
    S: BlobStorePut + CollectionStore + StorageFlush,
{
    Collection::new(&mut *store, POLICY_COLLECTION_SCOPE, signing_key.clone())
        .commit(fragment)
        .map(|_| ())
        .map_err(|error| storage_error("committing policy collection", error))
}

fn optional_one<T>(mut values: impl Iterator<Item = T>) -> Result<Option<T>, PolicyError> {
    match (values.next(), values.next()) {
        (None, None) => Ok(None),
        (Some(value), None) => Ok(Some(value)),
        _ => Err(PolicyError::Malformed("optional field is repeated")),
    }
}

/// Whether `pin_id` is a policy pin written by the pre-collection
/// implementation.
///
/// This is a staged-migration privacy guard, not a policy storage path. It
/// fails closed when the pin or its value cannot be read and never turns a
/// missing value into a durable want.
pub(crate) fn is_legacy_local_only_pin<S>(store: &mut S, pin_id: Id) -> bool
where
    S: BlobStore + PinStore,
{
    let head = match store.head(pin_id) {
        Ok(Some(head)) => head,
        Ok(None) => return false,
        Err(_) => return true,
    };
    let Ok(reader) = store.reader() else {
        return true;
    };
    if !reader.contains_blob(head).unwrap_or(false) {
        return true;
    }
    let Ok(value) =
        triblespace_core::repo::BlobStoreGet::get::<TribleSet, SimpleArchive>(&reader, head)
    else {
        return true;
    };
    find!(
        kind: Id,
        pattern!(&value, [{ _?entity @ legacy_local_only_pin: ?kind }])
    )
    .next()
    .is_some()
}

/// One stable capability request and its derived decision state.
#[derive(Clone, Debug)]
pub struct PendingRequest {
    pub id: Id,
    pub requester: VerifyingKey,
    pub partial_cap: Inline<Handle<SimpleArchive>>,
    pub received_at: Inline<NsTAIInterval>,
    pub status: Id,
}

fn request_fragment(
    requester: VerifyingKey,
    partial_cap: Inline<Handle<SimpleArchive>>,
) -> Fragment {
    entity! {
        metadata::tag: KIND_REQUEST,
        request_requester: requester,
        request_partial_cap: partial_cap,
    }
}

fn request_observation_fragment(request: Id, received_at: Inline<NsTAIInterval>) -> Fragment {
    entity! {
        metadata::tag: KIND_REQUEST_OBSERVATION,
        request_observed: request,
        request_received_at: received_at,
    }
}

fn request_decision_fragment(request: Id, status: Id) -> Result<Fragment, PolicyError> {
    if status != STATUS_APPROVED && status != STATUS_REJECTED {
        return Err(PolicyError::UnknownStatus(status));
    }
    Ok(entity! {
        metadata::tag: KIND_REQUEST_DECISION,
        request_decides: request,
        request_status: status,
    })
}

fn requests_from(meta: &TribleSet) -> Result<Vec<PendingRequest>, PolicyError> {
    let mut requests = BTreeMap::new();
    for (id, requester, partial_cap) in find!(
        (
            id: Id,
            requester: VerifyingKey,
            partial_cap: Inline<Handle<SimpleArchive>>,
        ),
        pattern!(meta, [{
            ?id @
            metadata::tag: KIND_REQUEST,
            request_requester: ?requester,
            request_partial_cap: ?partial_cap,
        }])
    ) {
        let canonical = request_fragment(requester, partial_cap)
            .root()
            .expect("request fragment has one root");
        if canonical != id {
            return Err(PolicyError::Malformed("request id is not intrinsic"));
        }
        let value = (requester, partial_cap);
        if requests.insert(id, value).is_some() {
            return Err(PolicyError::Malformed("request has repeated core fields"));
        }
    }

    let mut observations: BTreeMap<Id, Inline<NsTAIInterval>> = BTreeMap::new();
    for (observation, request, received_at) in find!(
        (
            observation: Id,
            request: Id,
            received_at: Inline<NsTAIInterval>,
        ),
        pattern!(meta, [{
            ?observation @
            metadata::tag: KIND_REQUEST_OBSERVATION,
            request_observed: ?request,
            request_received_at: ?received_at,
        }])
    ) {
        let canonical = request_observation_fragment(request, received_at)
            .root()
            .expect("observation fragment has one root");
        if canonical != observation {
            return Err(PolicyError::Malformed(
                "request observation id is not intrinsic",
            ));
        }
        if !requests.contains_key(&request) {
            return Err(PolicyError::Malformed(
                "observation names an unknown request",
            ));
        }
        observations
            .entry(request)
            .and_modify(|current| {
                if received_at.raw < current.raw {
                    *current = received_at;
                }
            })
            .or_insert(received_at);
    }

    let mut decisions: BTreeMap<Id, Id> = BTreeMap::new();
    for (decision, request, status) in find!(
        (decision: Id, request: Id, status: Id),
        pattern!(meta, [{
            ?decision @
            metadata::tag: KIND_REQUEST_DECISION,
            request_decides: ?request,
            request_status: ?status,
        }])
    ) {
        let canonical = request_decision_fragment(request, status)?
            .root()
            .expect("decision fragment has one root");
        if canonical != decision {
            return Err(PolicyError::Malformed(
                "request decision id is not intrinsic",
            ));
        }
        if !requests.contains_key(&request) {
            return Err(PolicyError::Malformed("decision names an unknown request"));
        }
        match decisions.insert(request, status) {
            Some(previous) if previous != status => {
                return Err(PolicyError::Conflict {
                    domain: "request decision",
                    identity: format!("{request:?}"),
                });
            }
            _ => {}
        }
    }

    requests
        .into_iter()
        .map(|(id, (requester, partial_cap))| {
            let received_at = observations
                .get(&id)
                .copied()
                .ok_or(PolicyError::Malformed("request has no observation"))?;
            Ok(PendingRequest {
                id,
                requester,
                partial_cap,
                received_at,
                status: decisions.get(&id).copied().unwrap_or(STATUS_PENDING),
            })
        })
        .collect()
}

/// Materialize all stable requests and their derived states.
pub fn list_pending_requests<S>(
    store: &mut S,
    signing_key: &SigningKey,
) -> Result<Vec<PendingRequest>, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    requests_from(&materialize(store, signing_key)?)
}

/// Record one authenticated wire request.
///
/// Request identity excludes receipt time, so retrying identical wire content
/// returns the same id. Each receipt remains an immutable observation.
pub fn record_pending_request<S>(
    store: &mut S,
    signing_key: &SigningKey,
    requester: VerifyingKey,
    partial_cap: Inline<Handle<SimpleArchive>>,
    received_at: Inline<NsTAIInterval>,
) -> Result<Id, PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    let request = request_fragment(requester, partial_cap);
    let request_id = request.root().expect("request fragment has one root");
    let mut fragment = request;
    fragment += request_observation_fragment(request_id, received_at);
    commit(store, signing_key, fragment)?;
    Ok(request_id)
}

fn decision_for(meta: &TribleSet, request_id: Id) -> Result<Id, PolicyError> {
    requests_from(meta)?
        .into_iter()
        .find(|request| request.id == request_id)
        .map(|request| request.status)
        .ok_or(PolicyError::NotFound(request_id))
}

/// Reject a pending request with one immutable decision event.
///
/// Approval intentionally has no symmetric status-only operation; use
/// [`approve_request_and_record_policy`] so approval and issuance are one
/// collection commit.
pub fn reject_pending_request<S>(
    store: &mut S,
    signing_key: &SigningKey,
    request_id: Id,
) -> Result<(), PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    let meta = materialize(store, signing_key)?;
    match decision_for(&meta, request_id)? {
        STATUS_PENDING => commit(
            store,
            signing_key,
            request_decision_fragment(request_id, STATUS_REJECTED)?,
        ),
        STATUS_REJECTED => Ok(()),
        _ => Err(PolicyError::Conflict {
            domain: "request decision",
            identity: format!("{request_id:?}"),
        }),
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct GrantKey {
    team_root: [u8; 32],
    subject: [u8; 32],
    scope: Id,
}

impl GrantKey {
    fn new(team_root: VerifyingKey, subject: VerifyingKey, scope: Id) -> Self {
        Self {
            team_root: team_root.to_bytes(),
            subject: subject.to_bytes(),
            scope,
        }
    }

    fn label(self) -> String {
        format!(
            "team={}, subject={}, scope={:?}",
            hex::encode(self.team_root),
            hex::encode(self.subject),
            self.scope
        )
    }
}

#[derive(Clone)]
struct PolicyVersion {
    id: Id,
    team_root: VerifyingKey,
    subject: VerifyingKey,
    scope: Id,
    issued_at: Inline<NsTAIInterval>,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
    retracted_at: Option<Inline<NsTAIInterval>>,
    request: Option<Id>,
    predecessors: Vec<Id>,
}

impl PolicyVersion {
    fn key(&self) -> GrantKey {
        GrantKey::new(self.team_root, self.subject, self.scope)
    }
}

fn policy_version_fragment(version: &PolicyVersion) -> Fragment {
    entity! {
        metadata::tag: KIND_POLICY_VERSION,
        policy_team_root: version.team_root,
        policy_subject: version.subject,
        policy_scope: version.scope,
        policy_issued_at: version.issued_at,
        policy_latest_cap: version.cap,
        policy_latest_sig: version.sig,
        policy_retracted_at?: version.retracted_at,
        policy_request?: version.request,
        metadata::supersedes*: version.predecessors.iter().copied(),
    }
}

fn policy_versions(meta: &TribleSet) -> Result<BTreeMap<Id, PolicyVersion>, PolicyError> {
    let mut versions = BTreeMap::new();
    for (id, team_root, subject, scope, issued_at, cap, sig) in find!(
        (
            id: Id,
            team_root: VerifyingKey,
            subject: VerifyingKey,
            scope: Id,
            issued_at: Inline<NsTAIInterval>,
            cap: Inline<Handle<SimpleArchive>>,
            sig: Inline<Handle<SimpleArchive>>,
        ),
        pattern!(meta, [{
            ?id @
            metadata::tag: KIND_POLICY_VERSION,
            policy_team_root: ?team_root,
            policy_subject: ?subject,
            policy_scope: ?scope,
            policy_issued_at: ?issued_at,
            policy_latest_cap: ?cap,
            policy_latest_sig: ?sig,
        }])
    ) {
        let retracted_at = optional_one(find!(
            value: Inline<NsTAIInterval>,
            pattern!(meta, [{ id @ policy_retracted_at: ?value }])
        ))?;
        let request = optional_one(find!(
            value: Id,
            pattern!(meta, [{ id @ policy_request: ?value }])
        ))?;
        let mut predecessors: Vec<Id> = find!(
            value: Id,
            pattern!(meta, [{ id @ metadata::supersedes: ?value }])
        )
        .collect();
        predecessors.sort();
        predecessors.dedup();
        let version = PolicyVersion {
            id,
            team_root,
            subject,
            scope,
            issued_at,
            cap,
            sig,
            retracted_at,
            request,
            predecessors,
        };
        let canonical = policy_version_fragment(&version)
            .root()
            .expect("policy version has one root");
        if canonical != id {
            return Err(PolicyError::Malformed("policy version id is not intrinsic"));
        }
        if versions.insert(id, version).is_some() {
            return Err(PolicyError::Malformed(
                "policy version has repeated core fields",
            ));
        }
    }
    validate_policy_dag(&versions)?;
    Ok(versions)
}

fn validate_policy_dag(versions: &BTreeMap<Id, PolicyVersion>) -> Result<(), PolicyError> {
    for version in versions.values() {
        for predecessor in &version.predecessors {
            let prior = versions.get(predecessor).ok_or(PolicyError::Malformed(
                "policy version has a dangling predecessor",
            ))?;
            if prior.key() != version.key() {
                return Err(PolicyError::Malformed(
                    "policy version supersedes a different policy track",
                ));
            }
        }
    }

    fn visit(
        id: Id,
        versions: &BTreeMap<Id, PolicyVersion>,
        visiting: &mut BTreeSet<Id>,
        visited: &mut BTreeSet<Id>,
    ) -> Result<(), PolicyError> {
        if visited.contains(&id) {
            return Ok(());
        }
        if !visiting.insert(id) {
            return Err(PolicyError::Malformed(
                "policy supersession graph contains a cycle",
            ));
        }
        for predecessor in &versions[&id].predecessors {
            visit(*predecessor, versions, visiting, visited)?;
        }
        visiting.remove(&id);
        visited.insert(id);
        Ok(())
    }

    let mut visited = BTreeSet::new();
    for id in versions.keys().copied() {
        visit(id, versions, &mut BTreeSet::new(), &mut visited)?;
    }
    Ok(())
}

fn policy_heads(
    versions: &BTreeMap<Id, PolicyVersion>,
) -> Result<BTreeMap<GrantKey, PolicyVersion>, PolicyError> {
    let superseded: BTreeSet<Id> = versions
        .values()
        .flat_map(|version| version.predecessors.iter().copied())
        .collect();
    let mut grouped: BTreeMap<GrantKey, Vec<PolicyVersion>> = BTreeMap::new();
    for version in versions.values().filter(|v| !superseded.contains(&v.id)) {
        grouped
            .entry(version.key())
            .or_default()
            .push(version.clone());
    }
    let all_keys: BTreeSet<GrantKey> = versions.values().map(PolicyVersion::key).collect();
    let mut heads = BTreeMap::new();
    for key in all_keys {
        let candidates = grouped.remove(&key).unwrap_or_default();
        if candidates.len() != 1 {
            return Err(PolicyError::Conflict {
                domain: "renewal-policy version DAG",
                identity: key.label(),
            });
        }
        heads.insert(key, candidates.into_iter().next().expect("len checked"));
    }
    Ok(heads)
}

fn delivery_acks(
    meta: &TribleSet,
    versions: &BTreeMap<Id, PolicyVersion>,
) -> Result<BTreeMap<Id, Inline<NsTAIInterval>>, PolicyError> {
    let mut acks = BTreeMap::new();
    for (ack, version, delivered_at) in find!(
        (
            ack: Id,
            version: Id,
            delivered_at: Inline<NsTAIInterval>,
        ),
        pattern!(meta, [{
            ?ack @
            metadata::tag: KIND_DELIVERY_ACK,
            policy_acknowledges: ?version,
            policy_delivered_at: ?delivered_at,
        }])
    ) {
        let canonical = entity! {
            metadata::tag: KIND_DELIVERY_ACK,
            policy_acknowledges: version,
            policy_delivered_at: delivered_at,
        }
        .root()
        .expect("delivery acknowledgement has one root");
        if canonical != ack {
            return Err(PolicyError::Malformed(
                "delivery acknowledgement id is not intrinsic",
            ));
        }
        if !versions.contains_key(&version) {
            return Err(PolicyError::Malformed(
                "delivery acknowledgement names an unknown policy version",
            ));
        }
        acks.entry(version)
            .and_modify(|current: &mut Inline<NsTAIInterval>| {
                if delivered_at.raw < current.raw {
                    *current = delivered_at;
                }
            })
            .or_insert(delivered_at);
    }
    Ok(acks)
}

/// One derived head of a renewal-policy version DAG.
#[derive(Clone, Debug)]
pub struct PolicyEntry {
    pub id: Id,
    pub team_root: VerifyingKey,
    pub subject: VerifyingKey,
    pub scope: Id,
    pub issued_at: Inline<NsTAIInterval>,
    pub latest_cap: Inline<Handle<SimpleArchive>>,
    pub latest_sig: Inline<Handle<SimpleArchive>>,
    pub retracted_at: Option<Inline<NsTAIInterval>>,
    pub delivered_at: Option<Inline<NsTAIInterval>>,
}

fn list_renewal_policy_from(meta: &TribleSet) -> Result<Vec<PolicyEntry>, PolicyError> {
    let versions = policy_versions(meta)?;
    let heads = policy_heads(&versions)?;
    let acks = delivery_acks(meta, &versions)?;
    Ok(heads
        .into_values()
        .map(|version| PolicyEntry {
            id: version.id,
            team_root: version.team_root,
            subject: version.subject,
            scope: version.scope,
            issued_at: version.issued_at,
            latest_cap: version.cap,
            latest_sig: version.sig,
            retracted_at: version.retracted_at,
            delivered_at: acks.get(&version.id).copied(),
        })
        .collect())
}

/// Enumerate the unique derived head of every renewal-policy track.
pub fn list_renewal_policy<S>(
    store: &mut S,
    signing_key: &SigningKey,
) -> Result<Vec<PolicyEntry>, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    list_renewal_policy_from(&materialize(store, signing_key)?)
}

/// Active policy heads whose exact version has not been acknowledged.
pub fn undelivered_entries<S>(
    store: &mut S,
    signing_key: &SigningKey,
) -> Result<Vec<PolicyEntry>, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    Ok(list_renewal_policy(store, signing_key)?
        .into_iter()
        .filter(|entry| entry.retracted_at.is_none() && entry.delivered_at.is_none())
        .collect())
}

/// Active policy heads whose credential expires within `renewal_window`.
pub fn renewable_within<S>(
    store: &mut S,
    signing_key: &SigningKey,
    renewal_window: hifitime::Duration,
) -> Result<Vec<PolicyEntry>, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    use triblespace_core::inline::TryFromInline;

    let cutoff = crate::clock::epoch_now() + renewal_window;
    Ok(list_renewal_policy(store, signing_key)?
        .into_iter()
        .filter(|entry| entry.retracted_at.is_none())
        .filter(|entry| {
            <(hifitime::Epoch, hifitime::Epoch)>::try_from_inline(&entry.issued_at)
                .map(|(_, upper)| upper <= cutoff)
                .unwrap_or(true)
        })
        .collect())
}

fn policy_candidate(
    meta: &TribleSet,
    team_root: VerifyingKey,
    subject: VerifyingKey,
    scope: Id,
    issued_at: Inline<NsTAIInterval>,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
    request: Option<Id>,
) -> Result<(Id, Option<Fragment>), PolicyError> {
    let versions = policy_versions(meta)?;
    let heads = policy_heads(&versions)?;
    let key = GrantKey::new(team_root, subject, scope);
    let predecessors = match heads.get(&key) {
        None => Vec::new(),
        Some(current) if current.retracted_at.is_some() => {
            return Err(PolicyError::Retracted(current.id));
        }
        Some(current)
            if current.issued_at == issued_at && current.cap == cap && current.sig == sig =>
        {
            return Ok((current.id, None));
        }
        Some(current) => vec![current.id],
    };
    let mut version = PolicyVersion {
        id: team_root_placeholder(),
        team_root,
        subject,
        scope,
        issued_at,
        cap,
        sig,
        retracted_at: None,
        request,
        predecessors,
    };
    let fragment = policy_version_fragment(&version);
    version.id = fragment.root().expect("policy version has one root");
    Ok((version.id, Some(fragment)))
}

// A non-nil temporary id used only while deriving an intrinsic entity id. The
// `id` field is not emitted by `policy_version_fragment`.
fn team_root_placeholder() -> Id {
    POLICY_COLLECTION_SCOPE
}

/// Append an issued credential to its `(team, subject, scope)` version DAG.
/// Exact replay is idempotent; a new credential supersedes the unique head.
pub fn record_policy_entry<S>(
    store: &mut S,
    signing_key: &SigningKey,
    team_root: VerifyingKey,
    subject: VerifyingKey,
    scope: Id,
    issued_at: Inline<NsTAIInterval>,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
) -> Result<Id, PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    let meta = materialize(store, signing_key)?;
    let (id, fragment) =
        policy_candidate(&meta, team_root, subject, scope, issued_at, cap, sig, None)?;
    if let Some(fragment) = fragment {
        commit(store, signing_key, fragment)?;
    }
    Ok(id)
}

/// Atomically approve a request and append the corresponding issued policy
/// version in one collection commit.
pub fn approve_request_and_record_policy<S>(
    store: &mut S,
    signing_key: &SigningKey,
    request_id: Id,
    team_root: VerifyingKey,
    subject: VerifyingKey,
    scope: Id,
    issued_at: Inline<NsTAIInterval>,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
) -> Result<Id, PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    let meta = materialize(store, signing_key)?;
    let request = requests_from(&meta)?
        .into_iter()
        .find(|request| request.id == request_id)
        .ok_or(PolicyError::NotFound(request_id))?;
    if request.requester != subject {
        return Err(PolicyError::RequestSubjectMismatch {
            request: request_id,
        });
    }

    let decision = match request.status {
        STATUS_PENDING => Some(request_decision_fragment(request_id, STATUS_APPROVED)?),
        STATUS_APPROVED => None,
        _ => {
            return Err(PolicyError::Conflict {
                domain: "request decision",
                identity: format!("{request_id:?}"),
            });
        }
    };
    let (version_id, version) = policy_candidate(
        &meta,
        team_root,
        subject,
        scope,
        issued_at,
        cap,
        sig,
        Some(request_id),
    )?;

    let mut fragment = Fragment::empty();
    if let Some(decision) = decision {
        fragment += decision;
    }
    if let Some(version) = version {
        fragment += version;
    }
    if !fragment.facts().is_empty() {
        commit(store, signing_key, fragment)?;
    }
    Ok(version_id)
}

/// Append a successor for the exact current policy head.
pub fn update_policy_entry<S>(
    store: &mut S,
    signing_key: &SigningKey,
    entry_id: Id,
    new_issued_at: Inline<NsTAIInterval>,
    new_cap: Inline<Handle<SimpleArchive>>,
    new_sig: Inline<Handle<SimpleArchive>>,
) -> Result<Id, PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    let meta = materialize(store, signing_key)?;
    let versions = policy_versions(&meta)?;
    let previous = versions
        .get(&entry_id)
        .cloned()
        .ok_or(PolicyError::NotFound(entry_id))?;
    let current = policy_heads(&versions)?
        .get(&previous.key())
        .cloned()
        .ok_or(PolicyError::NotFound(entry_id))?;
    if current.id != entry_id {
        return Err(PolicyError::Stale {
            expected: entry_id,
            current: current.id,
        });
    }
    if current.retracted_at.is_some() {
        return Err(PolicyError::Retracted(current.id));
    }
    if current.issued_at == new_issued_at && current.cap == new_cap && current.sig == new_sig {
        return Ok(current.id);
    }
    let next = PolicyVersion {
        id: team_root_placeholder(),
        team_root: current.team_root,
        subject: current.subject,
        scope: current.scope,
        issued_at: new_issued_at,
        cap: new_cap,
        sig: new_sig,
        retracted_at: None,
        request: current.request,
        predecessors: vec![current.id],
    };
    let fragment = policy_version_fragment(&next);
    let id = fragment.root().expect("policy version has one root");
    commit(store, signing_key, fragment)?;
    Ok(id)
}

/// Append a terminal successor to the exact current policy head.
pub fn retract_policy_entry<S>(
    store: &mut S,
    signing_key: &SigningKey,
    entry_id: Id,
) -> Result<Id, PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    use triblespace_core::inline::TryToInline;

    let meta = materialize(store, signing_key)?;
    let versions = policy_versions(&meta)?;
    let previous = versions
        .get(&entry_id)
        .cloned()
        .ok_or(PolicyError::NotFound(entry_id))?;
    let current = policy_heads(&versions)?
        .get(&previous.key())
        .cloned()
        .ok_or(PolicyError::NotFound(entry_id))?;
    if current.id != entry_id {
        return Err(PolicyError::Stale {
            expected: entry_id,
            current: current.id,
        });
    }
    if current.retracted_at.is_some() {
        return Ok(current.id);
    }
    let now = crate::clock::epoch_now();
    let retracted_at = (now, now)
        .try_to_inline()
        .map_err(|error| storage_error("encoding retraction time", error))?;
    let terminal = PolicyVersion {
        id: team_root_placeholder(),
        team_root: current.team_root,
        subject: current.subject,
        scope: current.scope,
        issued_at: current.issued_at,
        cap: current.cap,
        sig: current.sig,
        retracted_at: Some(retracted_at),
        request: current.request,
        predecessors: vec![current.id],
    };
    let fragment = policy_version_fragment(&terminal);
    let id = fragment.root().expect("policy version has one root");
    commit(store, signing_key, fragment)?;
    Ok(id)
}

/// Record acknowledgement of one exact policy version.
pub fn mark_policy_delivered<S>(
    store: &mut S,
    signing_key: &SigningKey,
    entry_id: Id,
) -> Result<(), PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    use triblespace_core::inline::TryToInline;

    let meta = materialize(store, signing_key)?;
    let versions = policy_versions(&meta)?;
    if !versions.contains_key(&entry_id) {
        return Err(PolicyError::NotFound(entry_id));
    }
    if delivery_acks(&meta, &versions)?.contains_key(&entry_id) {
        return Ok(());
    }
    let now = crate::clock::epoch_now();
    let delivered_at = (now, now)
        .try_to_inline()
        .map_err(|error| storage_error("encoding delivery time", error))?;
    commit(
        store,
        signing_key,
        entity! {
            metadata::tag: KIND_DELIVERY_ACK,
            policy_acknowledges: entry_id,
            policy_delivered_at: delivered_at,
        },
    )
}

/// Find the exact policy version authenticated by `(subject, signature)`.
pub fn find_policy_entry_by_subject_and_sig<S>(
    store: &mut S,
    signing_key: &SigningKey,
    subject: VerifyingKey,
    latest_sig: Inline<Handle<SimpleArchive>>,
) -> Result<Option<Id>, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    let meta = materialize(store, signing_key)?;
    let mut matches = policy_versions(&meta)?
        .into_values()
        .filter(|version| version.subject == subject && version.sig == latest_sig)
        .map(|version| version.id);
    match (matches.next(), matches.next()) {
        (None, None) => Ok(None),
        (Some(id), None) => Ok(Some(id)),
        _ => Err(PolicyError::Conflict {
            domain: "authenticated policy version",
            identity: hex::encode(latest_sig.raw),
        }),
    }
}

#[derive(Clone)]
struct TeamCapVersion {
    id: Id,
    team_root: VerifyingKey,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
    predecessors: Vec<Id>,
}

fn team_cap_fragment(version: &TeamCapVersion) -> Fragment {
    entity! {
        metadata::tag: KIND_TEAM_CAP_VERSION,
        cap_for_team: version.team_root,
        team_cap_handle: version.cap,
        team_sig_handle: version.sig,
        metadata::supersedes*: version.predecessors.iter().copied(),
    }
}

fn team_cap_heads(meta: &TribleSet) -> Result<BTreeMap<[u8; 32], TeamCapVersion>, PolicyError> {
    let mut versions = BTreeMap::new();
    for (id, team_root, cap, sig) in find!(
        (
            id: Id,
            team_root: VerifyingKey,
            cap: Inline<Handle<SimpleArchive>>,
            sig: Inline<Handle<SimpleArchive>>,
        ),
        pattern!(meta, [{
            ?id @
            metadata::tag: KIND_TEAM_CAP_VERSION,
            cap_for_team: ?team_root,
            team_cap_handle: ?cap,
            team_sig_handle: ?sig,
        }])
    ) {
        let mut predecessors: Vec<Id> = find!(
            value: Id,
            pattern!(meta, [{ id @ metadata::supersedes: ?value }])
        )
        .collect();
        predecessors.sort();
        predecessors.dedup();
        let version = TeamCapVersion {
            id,
            team_root,
            cap,
            sig,
            predecessors,
        };
        if team_cap_fragment(&version)
            .root()
            .expect("team-cap version has one root")
            != id
        {
            return Err(PolicyError::Malformed(
                "team-cap version id is not intrinsic",
            ));
        }
        if versions.insert(id, version).is_some() {
            return Err(PolicyError::Malformed(
                "team-cap version has repeated core fields",
            ));
        }
    }
    for version in versions.values() {
        for predecessor in &version.predecessors {
            let prior = versions.get(predecessor).ok_or(PolicyError::Malformed(
                "team-cap version has a dangling predecessor",
            ))?;
            if prior.team_root != version.team_root {
                return Err(PolicyError::Malformed(
                    "team-cap version supersedes a different team",
                ));
            }
        }
    }
    let superseded: BTreeSet<Id> = versions
        .values()
        .flat_map(|version| version.predecessors.iter().copied())
        .collect();
    let teams: BTreeSet<[u8; 32]> = versions
        .values()
        .map(|version| version.team_root.to_bytes())
        .collect();
    let mut heads = BTreeMap::new();
    for team in teams {
        let candidates: Vec<_> = versions
            .values()
            .filter(|version| {
                version.team_root.to_bytes() == team && !superseded.contains(&version.id)
            })
            .cloned()
            .collect();
        if candidates.len() != 1 {
            return Err(PolicyError::Conflict {
                domain: "team-cap version DAG",
                identity: hex::encode(team),
            });
        }
        heads.insert(team, candidates.into_iter().next().expect("len checked"));
    }
    Ok(heads)
}

/// Append the current credential for one team. Exact replay is idempotent.
pub fn set_team_cap<S>(
    store: &mut S,
    signing_key: &SigningKey,
    team_root: VerifyingKey,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
) -> Result<(), PolicyError>
where
    S: BlobStore + CollectionStore + StorageFlush,
    S::Reader: BlobStoreMeta,
{
    let meta = materialize(store, signing_key)?;
    let heads = team_cap_heads(&meta)?;
    let predecessor = match heads.get(&team_root.to_bytes()) {
        Some(current) if current.cap == cap && current.sig == sig => return Ok(()),
        Some(current) => vec![current.id],
        None => Vec::new(),
    };
    let version = TeamCapVersion {
        id: team_root_placeholder(),
        team_root,
        cap,
        sig,
        predecessors: predecessor,
    };
    commit(store, signing_key, team_cap_fragment(&version))
}

/// Read the unique current credential for `team_root`.
pub fn current_team_cap<S>(
    store: &mut S,
    signing_key: &SigningKey,
    team_root: VerifyingKey,
) -> Result<Option<(Inline<Handle<SimpleArchive>>, Inline<Handle<SimpleArchive>>)>, PolicyError>
where
    S: BlobStore + CollectionStore,
    S::Reader: BlobStoreMeta,
{
    Ok(team_cap_heads(&materialize(store, signing_key)?)?
        .get(&team_root.to_bytes())
        .map(|version| (version.cap, version.sig)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;
    use triblespace_core::blob::{Blob, IntoBlob};
    use triblespace_core::collection::CollectionStore;
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::inline::TryToInline;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStorePut, PinStore, PushResult};

    fn point_now() -> Inline<NsTAIInterval> {
        let now = hifitime::Epoch::now().expect("system time");
        (now, now).try_to_inline().expect("point interval")
    }

    fn handle(store: &mut MemoryRepo) -> Inline<Handle<SimpleArchive>> {
        let blob: Blob<SimpleArchive> = TribleSet::new().to_blob();
        store.put(blob).expect("put")
    }

    #[test]
    fn request_retry_has_one_stable_identity() {
        let key = SigningKey::generate(&mut OsRng);
        let mut store = MemoryRepo::default();
        let requester = SigningKey::generate(&mut OsRng).verifying_key();
        let partial = handle(&mut store);
        let received = point_now();

        let first = record_pending_request(&mut store, &key, requester, partial, received)
            .expect("first observation");
        let second = record_pending_request(&mut store, &key, requester, partial, received)
            .expect("replayed observation");
        assert_eq!(first, second);

        let listed = list_pending_requests(&mut store, &key).expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, first);
        assert_eq!(listed[0].status, STATUS_PENDING);
        assert!(store.pins().unwrap().next().is_none());
        assert!(store.records().unwrap().next().is_some());
    }

    #[test]
    fn collection_is_signer_owned() {
        let owner = SigningKey::generate(&mut OsRng);
        let stranger = SigningKey::generate(&mut OsRng);
        let mut store = MemoryRepo::default();
        let requester = SigningKey::generate(&mut OsRng).verifying_key();
        let partial = handle(&mut store);
        record_pending_request(&mut store, &owner, requester, partial, point_now()).unwrap();

        assert_eq!(list_pending_requests(&mut store, &owner).unwrap().len(), 1);
        assert!(
            list_pending_requests(&mut store, &stranger)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn approval_and_issued_version_share_one_commit() {
        let key = SigningKey::generate(&mut OsRng);
        let team = SigningKey::generate(&mut OsRng).verifying_key();
        let requester = SigningKey::generate(&mut OsRng).verifying_key();
        let mut store = MemoryRepo::default();
        let partial = handle(&mut store);
        let request =
            record_pending_request(&mut store, &key, requester, partial, point_now()).unwrap();
        let before = store.records().unwrap().count();
        let cap = handle(&mut store);
        let sig = handle(&mut store);
        let version = approve_request_and_record_policy(
            &mut store,
            &key,
            request,
            team,
            requester,
            *triblespace_core::id::ufoid(),
            point_now(),
            cap,
            sig,
        )
        .unwrap();
        assert_eq!(store.records().unwrap().count(), before + 1);
        assert_eq!(
            list_pending_requests(&mut store, &key).unwrap()[0].status,
            STATUS_APPROVED
        );
        assert_eq!(
            list_renewal_policy(&mut store, &key).unwrap()[0].id,
            version
        );
    }

    #[test]
    fn renewal_appends_a_successor_and_rejects_stale_updates() {
        let key = SigningKey::generate(&mut OsRng);
        let team = SigningKey::generate(&mut OsRng).verifying_key();
        let subject = SigningKey::generate(&mut OsRng).verifying_key();
        let mut store = MemoryRepo::default();
        let scope = *triblespace_core::id::ufoid();
        let first_cap = handle(&mut store);
        let first_sig = handle(&mut store);
        let first = record_policy_entry(
            &mut store,
            &key,
            team,
            subject,
            scope,
            point_now(),
            first_cap,
            first_sig,
        )
        .unwrap();
        let second_cap = handle(&mut store);
        let second_sig = handle(&mut store);
        let second =
            update_policy_entry(&mut store, &key, first, point_now(), second_cap, second_sig)
                .unwrap();
        assert_ne!(first, second);
        let stale_cap = handle(&mut store);
        let stale_sig = handle(&mut store);
        assert!(matches!(
            update_policy_entry(&mut store, &key, first, point_now(), stale_cap, stale_sig,),
            Err(PolicyError::Stale { .. })
        ));
    }

    #[test]
    fn team_caps_are_independent_version_tracks() {
        let key = SigningKey::generate(&mut OsRng);
        let team_a = SigningKey::generate(&mut OsRng).verifying_key();
        let team_b = SigningKey::generate(&mut OsRng).verifying_key();
        let mut store = MemoryRepo::default();
        let cap_a = handle(&mut store);
        let sig_a = handle(&mut store);
        let cap_b = handle(&mut store);
        let sig_b = handle(&mut store);
        set_team_cap(&mut store, &key, team_a, cap_a, sig_a).unwrap();
        set_team_cap(&mut store, &key, team_b, cap_b, sig_b).unwrap();
        assert_eq!(
            current_team_cap(&mut store, &key, team_a).unwrap(),
            Some((cap_a, sig_a))
        );
        assert_eq!(
            current_team_cap(&mut store, &key, team_b).unwrap(),
            Some((cap_b, sig_b))
        );
    }

    #[test]
    fn legacy_policy_pin_is_still_private() {
        let mut store = MemoryRepo::default();
        let marker = *triblespace_core::id::ufoid();
        let value: TribleSet = entity! {
            ExclusiveId::force_ref(&marker) @
            legacy_local_only_pin: POLICY_COLLECTION_SCOPE,
        }
        .into();
        let head = store.put(value).unwrap();
        let pin = *triblespace_core::id::ufoid();
        assert!(matches!(
            store.update(pin, None, Some(head)).unwrap(),
            PushResult::Success()
        ));
        assert!(is_legacy_local_only_pin(&mut store, pin));
    }
}
