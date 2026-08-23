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
//! that confer policy membership are not announced as public collection
//! evidence.

use std::collections::{BTreeMap, BTreeSet};
use triblespace_core::collection::reach;

use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::collection::records::CollectionName;
use triblespace_core::collection::{Collection, CollectionStore};
use triblespace_core::id::Id;
use triblespace_core::inline::Inline;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::encodings::time::NsTAIInterval;
use triblespace_core::macros::{entity, find, pattern};
use triblespace_core::metadata;
use triblespace_core::prelude::attributes;
use triblespace_core::prelude::inlineencodings::{ED25519PublicKey, GenId};
use triblespace_core::repo::{BlobStore, BlobStoreMeta, BlobStorePut, StorageFlush};
use triblespace_core::trible::{Fragment, TribleSet};

attributes! {
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
    /// Retraction time carried by a terminal version or its observation event.
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
    /// Stable terminal version named by an immutable retraction observation.
    "72244D02BFC88B514C64E37763AEB310" as pub policy_retraction_observes: GenId;
}

/// Name of the node's one private policy collection.
///
/// This replaced a minted scope id: the collection now says in the pile what
/// it is, instead of being recognisable only to someone holding this source.
/// Its team is the node's own key -- a team of one -- which is what a private
/// per-node collection means.
pub const POLICY_COLLECTION_NAME: &str = "policy";

/// The policy collection's name, validated.
pub fn policy_collection_name() -> CollectionName {
    CollectionName::new(POLICY_COLLECTION_NAME).expect("the policy collection name is legal")
}

/// A non-nil id used only as a placeholder while deriving an intrinsic entity
/// id; see [`team_root_placeholder`]. It is the id the policy collection's
/// scope used to be minted under, kept so the version-DAG ids it feeds do not
/// move.
///
/// Minted with `trible genid` on 2026-08-12.
const TEAM_ROOT_PLACEHOLDER: Id = triblespace_core::id::id_hex!("8067402E88FE8DBBFA559F2212C2353D");

// Entity-kind ids, minted with `trible genid` on 2026-08-12.
const KIND_REQUEST: Id = triblespace_core::id::id_hex!("35C96F095C003B7CDAE9602276C8D125");
const KIND_REQUEST_OBSERVATION: Id =
    triblespace_core::id::id_hex!("7FC24CCB05D6B99A3CA49F6B5D237A1D");
const KIND_REQUEST_DECISION: Id = triblespace_core::id::id_hex!("4B2C21CC2A145CADA7FE254AF7203DF8");
const KIND_POLICY_VERSION: Id = triblespace_core::id::id_hex!("25EC6D8585681E74DAA91A3F2D5ADC5F");
const KIND_DELIVERY_ACK: Id = triblespace_core::id::id_hex!("2FA2CED1789C8A29FC2B2FF6E6DE48FD");
const KIND_TEAM_CAP_VERSION: Id = triblespace_core::id::id_hex!("53490AE586FDA93510D4BAFC4B487D2F");
const KIND_POLICY_RETRACTION_OBSERVATION: Id =
    triblespace_core::id::id_hex!("5405154A77D3765D121FCCCA573090AA");

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
    Collection::new(
        &mut *store,
        &policy_collection_name(),
        signing_key.verifying_key(),
        signing_key.clone(),
        reach::private(),
    )
    .materialize()
    .map_err(|error| storage_error("materializing policy collection", error))
}

fn commit<S>(store: &mut S, signing_key: &SigningKey, fragment: Fragment) -> Result<(), PolicyError>
where
    S: BlobStorePut + CollectionStore + StorageFlush,
{
    let mut collection = Collection::new(
        &mut *store,
        &policy_collection_name(),
        signing_key.verifying_key(),
        signing_key.clone(),
        reach::private(),
    );
    collection
        .commit(fragment)
        .map_err(|error| storage_error("committing policy collection", error))?;
    collection
        .flush()
        .map_err(|error| storage_error("flushing policy collection", error))
}

fn optional_one<T>(mut values: impl Iterator<Item = T>) -> Result<Option<T>, PolicyError> {
    match (values.next(), values.next()) {
        (None, None) => Ok(None),
        (Some(value), None) => Ok(Some(value)),
        _ => Err(PolicyError::Malformed("optional field is repeated")),
    }
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

#[derive(Clone)]
struct RequestDecision {
    id: Id,
    request: Id,
    status: Id,
    predecessors: Vec<Id>,
}

fn request_decision_fragment(
    request: Id,
    status: Id,
    predecessors: &[Id],
) -> Result<Fragment, PolicyError> {
    if status != STATUS_APPROVED && status != STATUS_REJECTED {
        return Err(PolicyError::UnknownStatus(status));
    }
    Ok(entity! {
        metadata::tag: KIND_REQUEST_DECISION,
        request_decides: request,
        request_status: status,
        metadata::supersedes*: predecessors.iter().copied(),
    })
}

fn request_records_from(meta: &TribleSet) -> Result<BTreeMap<Id, PendingRequest>, PolicyError> {
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

    requests
        .into_iter()
        .map(|(id, (requester, partial_cap))| {
            let received_at = observations
                .get(&id)
                .copied()
                .ok_or(PolicyError::Malformed("request has no observation"))?;
            Ok((
                id,
                PendingRequest {
                    id,
                    requester,
                    partial_cap,
                    received_at,
                    status: STATUS_PENDING,
                },
            ))
        })
        .collect()
}

fn request_decisions(
    meta: &TribleSet,
    requests: &BTreeMap<Id, PendingRequest>,
) -> Result<BTreeMap<Id, RequestDecision>, PolicyError> {
    let mut decisions = BTreeMap::new();
    for (decision, request, status) in find!(
        (decision: Id, request: Id, status: Id),
        pattern!(meta, [{
            ?decision @
            metadata::tag: KIND_REQUEST_DECISION,
            request_decides: ?request,
            request_status: ?status,
        }])
    ) {
        let mut predecessors: Vec<Id> = find!(
            value: Id,
            pattern!(meta, [{ decision @ metadata::supersedes: ?value }])
        )
        .collect();
        predecessors.sort();
        predecessors.dedup();
        let canonical = request_decision_fragment(request, status, &predecessors)?
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
        if decisions
            .insert(
                decision,
                RequestDecision {
                    id: decision,
                    request,
                    status,
                    predecessors,
                },
            )
            .is_some()
        {
            return Err(PolicyError::Malformed(
                "request decision has repeated core fields",
            ));
        }
    }
    for decision in decisions.values() {
        for predecessor in &decision.predecessors {
            let prior = decisions.get(predecessor).ok_or(PolicyError::Malformed(
                "request decision has a dangling predecessor",
            ))?;
            if prior.request != decision.request {
                return Err(PolicyError::Malformed(
                    "request decision supersedes a different request",
                ));
            }
        }
    }
    Ok(decisions)
}

fn request_decision_frontiers(
    decisions: &BTreeMap<Id, RequestDecision>,
) -> BTreeMap<Id, Vec<RequestDecision>> {
    let superseded: BTreeSet<Id> = decisions
        .values()
        .flat_map(|decision| decision.predecessors.iter().copied())
        .collect();
    let mut frontiers: BTreeMap<Id, Vec<RequestDecision>> = BTreeMap::new();
    for decision in decisions
        .values()
        .filter(|decision| !superseded.contains(&decision.id))
    {
        frontiers
            .entry(decision.request)
            .or_default()
            .push(decision.clone());
    }
    for candidates in frontiers.values_mut() {
        candidates.sort_by_key(|decision| decision.id);
    }
    frontiers
}

fn requests_from(meta: &TribleSet) -> Result<Vec<PendingRequest>, PolicyError> {
    let mut requests = request_records_from(meta)?;
    let decisions = request_decisions(meta, &requests)?;
    for (request, candidates) in request_decision_frontiers(&decisions) {
        if candidates.len() != 1 {
            return Err(PolicyError::Conflict {
                domain: "request decision DAG",
                identity: format!("{request:?}"),
            });
        }
        requests
            .get_mut(&request)
            .expect("decision validation checked request")
            .status = candidates[0].status;
    }
    Ok(requests.into_values().collect())
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
    use triblespace_core::inline::TryToInline;

    let meta = materialize(store, signing_key)?;
    let requests = request_records_from(&meta)?;
    if !requests.contains_key(&request_id) {
        return Err(PolicyError::NotFound(request_id));
    }
    let decisions = request_decisions(&meta, &requests)?;
    let frontier = request_decision_frontiers(&decisions)
        .remove(&request_id)
        .unwrap_or_default();
    if frontier.len() == 1 && frontier[0].status == STATUS_REJECTED {
        return Ok(());
    }
    if frontier.len() == 1 && frontier[0].status == STATUS_APPROVED {
        return Err(PolicyError::Conflict {
            domain: "request decision",
            identity: format!("{request_id:?}"),
        });
    }
    let mut fragment = request_decision_fragment(
        request_id,
        STATUS_REJECTED,
        &frontier
            .iter()
            .map(|decision| decision.id)
            .collect::<Vec<_>>(),
    )?;

    // Resolving an approval/rejection fork toward rejection must also stop any
    // issuance authorized by the losing approval. Terminalize each affected
    // active policy track in the same collection commit, so no intermediate
    // state can renew or redispatch it.
    if !frontier.is_empty() {
        let now = crate::clock::epoch_now();
        let retracted_at = (now, now)
            .try_to_inline()
            .map_err(|error| storage_error("encoding rejection time", error))?;
        let versions = policy_versions_raw(&meta)?;
        for candidates in policy_frontiers(&versions).into_values() {
            if candidates
                .iter()
                .any(|version| version.request == Some(request_id))
                && candidates
                    .iter()
                    .any(|version| version.retracted_at.is_none())
            {
                let (_, terminal) = terminal_policy_fragment(&candidates, retracted_at);
                fragment += terminal;
            }
        }
    }
    commit(store, signing_key, fragment)
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

fn retraction_observation_fragment(version: Id, retracted_at: Inline<NsTAIInterval>) -> Fragment {
    entity! {
        metadata::tag: KIND_POLICY_RETRACTION_OBSERVATION,
        policy_retraction_observes: version,
        policy_retracted_at: retracted_at,
    }
}

fn retraction_observations(
    meta: &TribleSet,
) -> Result<BTreeMap<Id, Inline<NsTAIInterval>>, PolicyError> {
    let mut observations = BTreeMap::new();
    for (observation, version, retracted_at) in find!(
        (
            observation: Id,
            version: Id,
            retracted_at: Inline<NsTAIInterval>,
        ),
        pattern!(meta, [{
            ?observation @
            metadata::tag: KIND_POLICY_RETRACTION_OBSERVATION,
            policy_retraction_observes: ?version,
            policy_retracted_at: ?retracted_at,
        }])
    ) {
        if retraction_observation_fragment(version, retracted_at)
            .root()
            .expect("retraction observation has one root")
            != observation
        {
            return Err(PolicyError::Malformed(
                "policy retraction observation id is not intrinsic",
            ));
        }
        observations
            .entry(version)
            .and_modify(|current: &mut Inline<NsTAIInterval>| {
                if retracted_at.raw < current.raw {
                    *current = retracted_at;
                }
            })
            .or_insert(retracted_at);
    }
    Ok(observations)
}

fn terminal_policy_fragment(
    frontier: &[PolicyVersion],
    retracted_at: Inline<NsTAIInterval>,
) -> (Id, Fragment) {
    let template = frontier.first().expect("terminal frontier is nonempty");
    debug_assert!(
        frontier
            .iter()
            .all(|version| version.key() == template.key())
    );
    let terminal = PolicyVersion {
        id: team_root_placeholder(),
        team_root: template.team_root,
        subject: template.subject,
        scope: template.scope,
        issued_at: template.issued_at,
        cap: template.cap,
        sig: template.sig,
        // The terminal entity is stable across independent retractions. Wall
        // clock is recorded separately as a commutative observation.
        retracted_at: None,
        request: template.request,
        predecessors: frontier.iter().map(|version| version.id).collect(),
    };
    let mut fragment = policy_version_fragment(&terminal);
    let id = fragment.root().expect("policy version has one root");
    fragment += retraction_observation_fragment(id, retracted_at);
    (id, fragment)
}

fn policy_versions_raw(meta: &TribleSet) -> Result<BTreeMap<Id, PolicyVersion>, PolicyError> {
    let retractions = retraction_observations(meta)?;
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
        let embedded_retracted_at = optional_one(find!(
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
            retracted_at: embedded_retracted_at,
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
    for (version, retracted_at) in retractions {
        let current = versions.get_mut(&version).ok_or(PolicyError::Malformed(
            "retraction observation names an unknown policy version",
        ))?;
        current.retracted_at = Some(match current.retracted_at {
            Some(embedded) if embedded.raw <= retracted_at.raw => embedded,
            _ => retracted_at,
        });
    }
    validate_policy_dag(&versions)?;
    Ok(versions)
}

fn policy_versions(meta: &TribleSet) -> Result<BTreeMap<Id, PolicyVersion>, PolicyError> {
    let versions = policy_versions_raw(meta)?;
    validate_request_backed_versions(meta, &versions)?;
    Ok(versions)
}

fn validate_request_backed_versions(
    meta: &TribleSet,
    versions: &BTreeMap<Id, PolicyVersion>,
) -> Result<(), PolicyError> {
    let requests = request_records_from(meta)?;
    let decisions = request_decisions(meta, &requests)?;
    let decision_frontiers = request_decision_frontiers(&decisions);
    let active: BTreeSet<Id> = policy_frontiers(versions)
        .into_values()
        .flatten()
        .map(|version| version.id)
        .collect();
    for version in versions.values() {
        let Some(request_id) = version.request else {
            continue;
        };
        let request = requests.get(&request_id).ok_or(PolicyError::Malformed(
            "policy version names an unknown request",
        ))?;
        if request.requester != version.subject {
            return Err(PolicyError::Malformed(
                "policy version request belongs to a different subject",
            ));
        }
        if active.contains(&version.id) && version.retracted_at.is_none() {
            match decision_frontiers.get(&request_id).map(Vec::as_slice) {
                Some([decision]) if decision.status == STATUS_APPROVED => {}
                Some(decisions) if decisions.len() > 1 => {
                    return Err(PolicyError::Conflict {
                        domain: "request decision DAG",
                        identity: format!("{request_id:?}"),
                    });
                }
                _ => {
                    return Err(PolicyError::Conflict {
                        domain: "request-backed policy authorization",
                        identity: format!("{request_id:?}"),
                    });
                }
            }
        }
    }
    Ok(())
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
    let mut heads = BTreeMap::new();
    for (key, candidates) in policy_frontiers(versions) {
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

fn policy_frontiers(
    versions: &BTreeMap<Id, PolicyVersion>,
) -> BTreeMap<GrantKey, Vec<PolicyVersion>> {
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
    for candidates in grouped.values_mut() {
        candidates.sort_by_key(|version| version.id);
    }
    grouped
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

fn policy_candidate_from_versions(
    versions: &BTreeMap<Id, PolicyVersion>,
    team_root: VerifyingKey,
    subject: VerifyingKey,
    scope: Id,
    issued_at: Inline<NsTAIInterval>,
    cap: Inline<Handle<SimpleArchive>>,
    sig: Inline<Handle<SimpleArchive>>,
    request: Option<Id>,
) -> Result<(Id, Option<Fragment>), PolicyError> {
    let key = GrantKey::new(team_root, subject, scope);
    let frontier = policy_frontiers(versions).remove(&key).unwrap_or_default();
    let existing = versions.values().find(|version| {
        version.key() == key
            && version.issued_at == issued_at
            && version.cap == cap
            && version.sig == sig
            && version.request == request
            && version.retracted_at.is_none()
    });
    if let Some(existing) = existing {
        let selected_fork_head =
            frontier.len() > 1 && frontier.iter().any(|candidate| candidate.id == existing.id);
        if !selected_fork_head {
            return Ok((existing.id, None));
        }
    }
    if let Some(terminal) = frontier
        .iter()
        .find(|version| version.retracted_at.is_some())
    {
        return Err(PolicyError::Retracted(terminal.id));
    }
    let predecessors = frontier.into_iter().map(|version| version.id).collect();
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
    TEAM_ROOT_PLACEHOLDER
}

/// Append an issued credential to its `(team, subject, scope)` version DAG.
/// Exact replay is idempotent. A new credential supersedes every current head;
/// replaying one selected head of a fork explicitly converges on that value.
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
    let versions = policy_versions(&meta)?;
    let (id, fragment) = policy_candidate_from_versions(
        &versions, team_root, subject, scope, issued_at, cap, sig, None,
    )?;
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
    let requests = request_records_from(&meta)?;
    let request = requests
        .get(&request_id)
        .cloned()
        .ok_or(PolicyError::NotFound(request_id))?;
    if request.requester != subject {
        return Err(PolicyError::RequestSubjectMismatch {
            request: request_id,
        });
    }

    let decisions = request_decisions(&meta, &requests)?;
    let frontier = request_decision_frontiers(&decisions)
        .remove(&request_id)
        .unwrap_or_default();
    let exact_approval = || -> Result<Id, PolicyError> {
        // Approval is a one-shot transition, not an alternate renewal API.
        // An exact replay returns the already-recorded version even if the
        // track has since advanced; different issuance bytes must not
        // silently supersede the current credential.
        let versions = policy_versions_raw(&meta)?;
        let mut exact = versions.values().filter(|version| {
            version.request == Some(request_id)
                && version.team_root == team_root
                && version.subject == subject
                && version.scope == scope
                && version.issued_at == issued_at
                && version.cap == cap
                && version.sig == sig
                && version.retracted_at.is_none()
        });
        match (exact.next(), exact.next()) {
            (Some(version), None) => Ok(version.id),
            _ => Err(PolicyError::Conflict {
                domain: "request approval replay",
                identity: format!("{request_id:?}"),
            }),
        }
    };
    let decision_predecessors = match frontier.as_slice() {
        [] => Vec::new(),
        [only] if only.status == STATUS_APPROVED => return exact_approval(),
        [only] if only.status == STATUS_REJECTED => {
            return Err(PolicyError::Conflict {
                domain: "request decision",
                identity: format!("{request_id:?}"),
            });
        }
        _ => frontier.iter().map(|decision| decision.id).collect(),
    };
    let decision = request_decision_fragment(request_id, STATUS_APPROVED, &decision_predecessors)?;
    let versions = policy_versions_raw(&meta)?;
    let (version_id, version) = policy_candidate_from_versions(
        &versions,
        team_root,
        subject,
        scope,
        issued_at,
        cap,
        sig,
        Some(request_id),
    )?;

    let mut fragment = Fragment::empty();
    fragment += decision;
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
    if let Some(existing) = versions.values().find(|version| {
        version.key() == previous.key()
            && version.predecessors.contains(&entry_id)
            && version.retracted_at.is_none()
            && version.issued_at == new_issued_at
            && version.cap == new_cap
            && version.sig == new_sig
    }) {
        return Ok(existing.id);
    }
    let frontier = policy_frontiers(&versions)
        .remove(&previous.key())
        .unwrap_or_default();
    let [current] = frontier.as_slice() else {
        return Err(PolicyError::Conflict {
            domain: "renewal-policy version DAG",
            identity: previous.key().label(),
        });
    };
    if current.id != entry_id {
        return Err(PolicyError::Stale {
            expected: entry_id,
            current: current.id,
        });
    }
    let current = current.clone();
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
    let versions = policy_versions_raw(&meta)?;
    let previous = versions
        .get(&entry_id)
        .cloned()
        .ok_or(PolicyError::NotFound(entry_id))?;
    let frontier = policy_frontiers(&versions)
        .remove(&previous.key())
        .unwrap_or_default();
    if frontier.len() == 1 && frontier[0].retracted_at.is_some() {
        return Ok(frontier[0].id);
    }
    if frontier.is_empty() {
        return Err(PolicyError::NotFound(entry_id));
    }
    let now = crate::clock::epoch_now();
    let retracted_at = (now, now)
        .try_to_inline()
        .map_err(|error| storage_error("encoding retraction time", error))?;
    let (id, fragment) = terminal_policy_fragment(&frontier, retracted_at);
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
    let versions = policy_versions(&meta)?;
    let mut matches = versions
        .values()
        .filter(|version| version.subject == subject && version.sig == latest_sig)
        .map(|version| version.id);
    match (matches.next(), matches.next()) {
        (None, None) => Ok(None),
        (Some(id), None) => Ok(Some(id)),
        _ => {
            // Fork convergence can deliberately carry a selected credential
            // into a new multi-predecessor head, so the same signature may
            // occur in both historical and current versions. Prefer the one
            // unique current head; otherwise the acknowledgement is
            // genuinely ambiguous and must fail closed.
            let mut current = policy_heads(&versions)?
                .into_values()
                .filter(|version| version.subject == subject && version.sig == latest_sig)
                .map(|version| version.id);
            match (current.next(), current.next()) {
                (Some(id), None) => Ok(Some(id)),
                _ => Err(PolicyError::Conflict {
                    domain: "authenticated policy version",
                    identity: hex::encode(latest_sig.raw),
                }),
            }
        }
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

fn team_cap_versions(meta: &TribleSet) -> Result<BTreeMap<Id, TeamCapVersion>, PolicyError> {
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
    Ok(versions)
}

fn team_cap_heads_from_versions(
    versions: &BTreeMap<Id, TeamCapVersion>,
) -> Result<BTreeMap<[u8; 32], TeamCapVersion>, PolicyError> {
    let frontiers = team_cap_frontiers(versions);
    let mut heads = BTreeMap::new();
    for (team, candidates) in frontiers {
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

fn team_cap_frontiers(
    versions: &BTreeMap<Id, TeamCapVersion>,
) -> BTreeMap<[u8; 32], Vec<TeamCapVersion>> {
    let superseded: BTreeSet<Id> = versions
        .values()
        .flat_map(|version| version.predecessors.iter().copied())
        .collect();
    let mut frontiers: BTreeMap<[u8; 32], Vec<TeamCapVersion>> = BTreeMap::new();
    for version in versions
        .values()
        .filter(|version| !superseded.contains(&version.id))
    {
        frontiers
            .entry(version.team_root.to_bytes())
            .or_default()
            .push(version.clone());
    }
    for candidates in frontiers.values_mut() {
        candidates.sort_by_key(|version| version.id);
    }
    frontiers
}

fn team_cap_heads(meta: &TribleSet) -> Result<BTreeMap<[u8; 32], TeamCapVersion>, PolicyError> {
    team_cap_heads_from_versions(&team_cap_versions(meta)?)
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
    let versions = team_cap_versions(&meta)?;
    let frontier = team_cap_frontiers(&versions)
        .remove(&team_root.to_bytes())
        .unwrap_or_default();
    let existing = versions
        .values()
        .find(|version| version.team_root == team_root && version.cap == cap && version.sig == sig);
    if let Some(existing) = existing {
        let selected_fork_head =
            frontier.len() > 1 && frontier.iter().any(|candidate| candidate.id == existing.id);
        if !selected_fork_head {
            return Ok(());
        }
    }
    let predecessor = frontier.into_iter().map(|version| version.id).collect();
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
    use triblespace_core::inline::TryToInline;
    use triblespace_core::repo::BlobStorePut;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::pile::Pile;

    fn point_now() -> Inline<NsTAIInterval> {
        let now = hifitime::Epoch::now().expect("system time");
        (now, now).try_to_inline().expect("point interval")
    }

    fn handle(store: &mut MemoryRepo) -> Inline<Handle<SimpleArchive>> {
        let blob: Blob<SimpleArchive> = TribleSet::new().to_blob();
        store.put(blob).expect("put")
    }

    fn tagged_handle(store: &mut Pile, tag: Id) -> Inline<Handle<SimpleArchive>> {
        let value: TribleSet = entity! { metadata::tag: tag }.into();
        store.put(value).expect("put tagged archive")
    }

    fn merge_memory_repo(target: &mut MemoryRepo, mut source: MemoryRepo) {
        let records = source
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        target.blobs.union(source.blobs);
        for record in records {
            target.insert(record).unwrap();
        }
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
        let issued_at = point_now();
        let version = approve_request_and_record_policy(
            &mut store,
            &key,
            request,
            team,
            requester,
            *triblespace_core::id::ufoid(),
            issued_at,
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

        let scope = list_renewal_policy(&mut store, &key).unwrap()[0].scope;
        let replay = approve_request_and_record_policy(
            &mut store, &key, request, team, requester, scope, issued_at, cap, sig,
        )
        .unwrap();
        assert_eq!(replay, version);
        assert_eq!(store.records().unwrap().count(), before + 1);

        let different_cap = {
            let value: TribleSet = entity! { metadata::tag: *triblespace_core::id::ufoid() }.into();
            store.put(value).unwrap()
        };
        assert!(matches!(
            approve_request_and_record_policy(
                &mut store,
                &key,
                request,
                team,
                requester,
                scope,
                issued_at,
                different_cap,
                sig,
            ),
            Err(PolicyError::Conflict {
                domain: "request approval replay",
                ..
            })
        ));
        assert_eq!(store.records().unwrap().count(), before + 1);
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
        let second_issued_at = point_now();
        let second = update_policy_entry(
            &mut store,
            &key,
            first,
            second_issued_at,
            second_cap,
            second_sig,
        )
        .unwrap();
        assert_ne!(first, second);
        assert_eq!(
            update_policy_entry(
                &mut store,
                &key,
                first,
                second_issued_at,
                second_cap,
                second_sig,
            )
            .unwrap(),
            second
        );
        let stale_cap = handle(&mut store);
        let stale_sig = handle(&mut store);
        assert!(matches!(
            update_policy_entry(&mut store, &key, first, point_now(), stale_cap, stale_sig,),
            Err(PolicyError::Stale { .. })
        ));
        let terminal = retract_policy_entry(&mut store, &key, second).unwrap();
        assert_eq!(
            retract_policy_entry(&mut store, &key, second).unwrap(),
            terminal
        );
    }

    #[test]
    fn concatenating_policy_forks_fails_closed_then_converges_explicitly() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.pile");
        let left_path = dir.path().join("left.pile");
        let right_path = dir.path().join("right.pile");
        std::fs::File::create(&base_path).unwrap();

        let key = SigningKey::generate(&mut OsRng);
        let team = SigningKey::generate(&mut OsRng).verifying_key();
        let subject = SigningKey::generate(&mut OsRng).verifying_key();
        let scope = *triblespace_core::id::ufoid();
        let first = {
            let mut base = Pile::open(&base_path).unwrap();
            let cap = tagged_handle(&mut base, *triblespace_core::id::ufoid());
            let sig = tagged_handle(&mut base, *triblespace_core::id::ufoid());
            let first =
                record_policy_entry(&mut base, &key, team, subject, scope, point_now(), cap, sig)
                    .unwrap();
            base.close().unwrap();
            first
        };
        std::fs::copy(&base_path, &left_path).unwrap();
        std::fs::copy(&base_path, &right_path).unwrap();

        for path in [&left_path, &right_path] {
            let mut pile = Pile::open(path).unwrap();
            let cap = tagged_handle(&mut pile, *triblespace_core::id::ufoid());
            let sig = tagged_handle(&mut pile, *triblespace_core::id::ufoid());
            update_policy_entry(&mut pile, &key, first, point_now(), cap, sig).unwrap();
            pile.close().unwrap();
        }

        let left = std::fs::read(&left_path).unwrap();
        let right = std::fs::read(&right_path).unwrap();
        for (name, first_bytes, second_bytes) in
            [("left-right", &left, &right), ("right-left", &right, &left)]
        {
            let merged_path = dir.path().join(format!("{name}.pile"));
            let mut merged_bytes = first_bytes.to_vec();
            merged_bytes.extend_from_slice(second_bytes);
            std::fs::write(&merged_path, merged_bytes).unwrap();
            let mut merged = Pile::open(&merged_path).unwrap();
            assert!(matches!(
                list_renewal_policy(&mut merged, &key),
                Err(PolicyError::Conflict {
                    domain: "renewal-policy version DAG",
                    ..
                })
            ));
            let resolved_cap = tagged_handle(&mut merged, *triblespace_core::id::ufoid());
            let resolved_sig = tagged_handle(&mut merged, *triblespace_core::id::ufoid());
            let resolved = record_policy_entry(
                &mut merged,
                &key,
                team,
                subject,
                scope,
                point_now(),
                resolved_cap,
                resolved_sig,
            )
            .unwrap();
            let listed = list_renewal_policy(&mut merged, &key).unwrap();
            assert_eq!(listed.len(), 1);
            assert_eq!(listed[0].id, resolved);
            merged.close().unwrap();
        }
    }

    #[test]
    fn concurrent_approve_and_reject_fails_closed_until_explicit_rejection() {
        let key = SigningKey::generate(&mut OsRng);
        let team = SigningKey::generate(&mut OsRng).verifying_key();
        let requester = SigningKey::generate(&mut OsRng).verifying_key();
        let mut base = MemoryRepo::default();
        let partial = handle(&mut base);
        let request =
            record_pending_request(&mut base, &key, requester, partial, point_now()).unwrap();

        let mut approved = base.clone();
        let cap = handle(&mut approved);
        let sig = handle(&mut approved);
        approve_request_and_record_policy(
            &mut approved,
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
        let mut rejected = base;
        reject_pending_request(&mut rejected, &key, request).unwrap();

        merge_memory_repo(&mut approved, rejected);
        assert!(matches!(
            list_pending_requests(&mut approved, &key),
            Err(PolicyError::Conflict {
                domain: "request decision DAG",
                ..
            })
        ));
        assert!(matches!(
            list_renewal_policy(&mut approved, &key),
            Err(PolicyError::Conflict {
                domain: "request decision DAG",
                ..
            })
        ));
        reject_pending_request(&mut approved, &key, request).unwrap();
        assert_eq!(
            list_pending_requests(&mut approved, &key).unwrap()[0].status,
            STATUS_REJECTED
        );
        let policy = list_renewal_policy(&mut approved, &key).unwrap();
        assert_eq!(policy.len(), 1);
        assert!(policy[0].retracted_at.is_some());
        assert!(undelivered_entries(&mut approved, &key).unwrap().is_empty());
    }

    #[test]
    fn concurrent_retractions_converge_to_one_stable_terminal() {
        let key = SigningKey::generate(&mut OsRng);
        let team = SigningKey::generate(&mut OsRng).verifying_key();
        let subject = SigningKey::generate(&mut OsRng).verifying_key();
        let mut base = MemoryRepo::default();
        let cap = handle(&mut base);
        let sig = handle(&mut base);
        let first = record_policy_entry(
            &mut base,
            &key,
            team,
            subject,
            *triblespace_core::id::ufoid(),
            point_now(),
            cap,
            sig,
        )
        .unwrap();
        let mut left = base.clone();
        let mut right = base;
        let left_terminal = retract_policy_entry(&mut left, &key, first).unwrap();
        let right_terminal = retract_policy_entry(&mut right, &key, first).unwrap();
        assert_eq!(left_terminal, right_terminal);
        merge_memory_repo(&mut left, right);
        let listed = list_renewal_policy(&mut left, &key).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, left_terminal);
        assert!(listed[0].retracted_at.is_some());
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
    fn team_cap_fork_can_converge_on_a_selected_existing_head() {
        let key = SigningKey::generate(&mut OsRng);
        let team = SigningKey::generate(&mut OsRng).verifying_key();
        let mut base = MemoryRepo::default();
        let first_cap = handle(&mut base);
        let first_sig = handle(&mut base);
        set_team_cap(&mut base, &key, team, first_cap, first_sig).unwrap();

        let mut left = base.clone();
        let mut right = base;
        let left_cap = {
            let value: TribleSet = entity! { metadata::tag: *triblespace_core::id::ufoid() }.into();
            left.put(value).unwrap()
        };
        let left_sig = handle(&mut left);
        let right_cap = {
            let value: TribleSet = entity! { metadata::tag: *triblespace_core::id::ufoid() }.into();
            right.put(value).unwrap()
        };
        let right_sig = handle(&mut right);
        set_team_cap(&mut left, &key, team, left_cap, left_sig).unwrap();
        set_team_cap(&mut right, &key, team, right_cap, right_sig).unwrap();
        merge_memory_repo(&mut left, right);
        assert!(matches!(
            current_team_cap(&mut left, &key, team),
            Err(PolicyError::Conflict {
                domain: "team-cap version DAG",
                ..
            })
        ));

        set_team_cap(&mut left, &key, team, left_cap, left_sig).unwrap();
        assert_eq!(
            current_team_cap(&mut left, &key, team).unwrap(),
            Some((left_cap, left_sig))
        );
    }
}
