# Distributed Sync

The [`triblespace-net`](https://github.com/triblespace/triblespace-rs/tree/main/triblespace-net)
crate synchronizes one team's TribleSpace store over
[iroh](https://www.iroh.computer/). It has one authenticated direct protocol
with two narrow progress paths:

- bounded fair pairwise PATCH walks converge monotone store inventories; and
- a bounded XOR DHT locates providers whose durable local OFFER intersects
  their current resident Blob snapshot, for an already-known exact artifact
  handle.

Stored routing evidence and DHT referrals grant no authority. Every inventory
walk, provider operation, and exact read uses reciprocal CONNECT and SYNC_TEAM
authorization. Pairwise set union is the epidemic exchange itself; there is no
publisherless broadcast wake plane.

The user-visible surface is `Peer<S>`. It wraps a synchronous local store and
owns the asynchronous host. There is no remote mutable head, receipt RPC,
replica roster, or second authorization database.

Enable it through the facade crate's `net` feature:

```toml
[dependencies]
triblespace = { version = "x.y.z", features = ["net"] }
```

```rust,ignore
use triblespace::net::peer::{
    BlobReconcileMode, Peer, PeerConfig, ReconcileDirection, ReconcileQos,
};

let pile = triblespace::core::repo::pile::Pile::open(path)?;
let mut peer = Peer::new(pile, signing_key.clone(), PeerConfig {
    peers: vec![bootstrap_endpoint],
    team: team_root,
    connect_proof,
    sync_proof,
    qos: ReconcileQos {
        direction: ReconcileDirection::Bidirectional,
        blobs: BlobReconcileMode::Demand,
    },
})?;

peer.refresh();
```

`team` is simultaneously the external capability trust root, inventory scope,
and provider-key namespace. The backing store must be dedicated to this one
team. Only PEER evidence contains a team key intrinsically; collection
records, proofs, and blobs are content-addressed global forms. Attaching a
mixed-team store would therefore disclose all of those resident sets to any
caller authorized for the configured team.

## One four-component inventory

Synchronization is componentwise set union over four canonical inventories:

| Component | Canonical key | Meaning |
|---|---|---|
| PEER | `team_public_key || peer_public_key` | monotone routing-candidate evidence |
| Collection record | 16-byte intrinsic record ID | native `COMMIT`, `MERGE`, or `DERIVE` evidence |
| Capability proof | 32-byte proof identity | one complete native `K0 (S C K)+` proof |
| Blob | 32-byte BLAKE3 handle | resident content bytes |

The PEER walk is rooted at the configured team's subtree and carries only the
32-byte peer suffix on the wire. The other three components expose the complete
sets in the dedicated store. Collection and proof leaves carry their exact
canonical bodies; blob leaves carry keys and are transferred separately.

This inventory is structural evidence, not ambient authorization. A resident
proof does not become active merely because it synchronized. A collection
record may be structurally canonical while remaining semantically unusable to
a particular resolver. A PEER fact says only that an endpoint is a routing
candidate for the team: it proves no liveness, reachability, residency,
retention, or capability.

`Peer::refresh` is the mutable-store boundary. It drains verified network
events, inserts them monotonically, crosses one storage durability barrier,
and only then replaces the immutable snapshot served by the host. Thus remote
readers never observe a newly admitted batch before that batch is durable.
External appends to a file-backed pile are reobserved at this same boundary.
`Peer::try_refresh` performs the identical boundary and fail-closed serving
cleanup while returning a scope error to an operation that must not continue
against a physically conflicted store; ordinary scheduler loops may use the
unit-returning `refresh` surface and observe the warning instead.

Each immutable store snapshot is also its own component-aware local
invalidation token. `StoreSnapshot::changes_since` conservatively identifies
which of PEER, collection-record, capability-proof, and the observable Blob
view may have changed; backends without cheap classification report all
components. The Blob component includes membership, metadata, and
retrievability. Pile compares persistent PATCH root sharing, so an unrelated
WANT append is a no-op while same-handle backing replacement still reports a
Blob change. The host carries unchanged immutable PATCH components directly
into the next serving snapshot and enumerates only changed components. These
snapshots and change masks never cross the network. A semantic no-op refresh
still replaces the host's physical Blob-reader lease while retaining its
inventory tree, so compaction can retire old mmap and Yard generations without
inventing a fifth semantic change component.

## Two independent capabilities

Every useful connection proves two exact actions rooted at the same team key:

```text
CONNECT
subject  = authenticated transport peer key
action   = ACTION_CONNECT
resource = team public-key bytes
mode     = Invoke

SYNC_TEAM
subject  = authenticated transport peer key
action   = ACTION_SYNC_TEAM
resource = team public-key bytes
mode     = Invoke
```

The proofs are independent. They may have different ancestry and validity
bounds, and possession of one never implies the other. Invoke-and-delegate
authority satisfies an Invoke request; Delegate-only authority does not.

The first stream of every connection is `OP_AUTH` with the complete CONNECT
`CapabilityProofBundle`. Verification uses the externally configured team
root, the transport-authenticated caller key, and the current time. On success
the server returns its own bounded CONNECT bundle in that same response; the
client verifies its team, exact action/resource, current validity, and leaf
against the expected TLS-authenticated remote endpoint before pooling the
connection. A rejected connection closes. Both sides stop using a bounded
accepted connection when the corresponding effective inclusive CONNECT
validity interval expires.

The initiator's CONNECT bundle necessarily crosses before the receiver proves
its own CONNECT capability. That bundle is non-bearer evidence cryptographically
bound to the caller and exact delegate key. The client checks that the TLS
identity is the endpoint it intended to dial before sending even this first
proof, but the proof itself is not cryptographically bound to that receiver
key; this exchange does not claim credential confidentiality or zero knowledge.
It sends no SYNC proof, element identity, query, or data request until the
returned CONNECT proof verifies.

After mutual CONNECT, `INVENTORY_AUTH` presents the complete SYNC_TEAM bundle
once for that connection. The server returns its own bounded SYNC_TEAM bundle
only after accepting the caller; the client verifies it against the same team,
remote TLS endpoint, exact SYNC_TEAM atom, and current instant before issuing
any useful request. Successful reciprocal verification installs one
team-selected, validity-bounded session on each side. The client cannot
nominate another scope, and a second authorization attempt on the same
connection is rejected. Manifest, node, provider, blob-range, and known-hash
`GET_BLOB` requests all require this live session. Knowing a content hash is
not disclosure authority.

The client shares one pool across reconciliation and DHT operations and keeps
at most 64 fully CONNECT- and SYNC_TEAM-authorized sessions resident. A peer
enters the least-recently-used set only after both reciprocal exchanges
succeed. Capacity retirement releases the pool's ownership without actively
closing the shallow connection clones held by in-flight operations; a later
request redials and rechecks both proofs. Failure and expiry remove only the
exact session generation that observed them, so a late operation cannot evict
a newer redial of the same peer.

Neither handshake searches the store for a proof or fetches missing claims.
Each selected bundle contains its exact ordered proof and claim closure inline.
Stored proof presence remains evidence only.

## Merkle reconciliation

Each component is an ordered PATCH inventory with a canonical BLAKE3 Merkle
summary. A manifest contains the four component tags, leaf counts, and roots in
fixed order. Its generation hash binds the version, team key, and all four
entries. The generation is a cache identity, not an authorization token or
evidence of global completeness.

A puller compares remote summaries with its local trees and descends only
differing prefixes. Every node request states the expected component root,
prefix, count, and digest. Responses are bounded and checked against that
expectation. Collection and proof PATCHes retain canonical bodies as immutable
leaf values, but values do not participate in their Merkle digest: the
authenticated identity remains the exact key set. Construction and leaf
service both require every body to match its intrinsic key.

The manifest pins immutable **component** snapshots in a server cache bounded
to eight historical roots per component.
Unchanged non-blob roots reuse the same component `Arc`. The Blob tree is also
reused, but its small access-bearing wrapper is refreshed so an unchanged key
set cannot pin an obsolete mmap or Yard generation indefinitely. Snapshot
installation refreshes an already-pinned matching Blob root without implicitly
pinning an unrequested one. Record-only
churn therefore neither retains old blob-access snapshots nor duplicates the other
inventory trees.
Later node and blob-range requests repeat the exact root. If that snapshot has
expired or been evicted, the server returns `snapshot unavailable`; it never
splices bytes from its current state into the older walk. The client obtains a
fresh manifest and restarts that component. Mirror blob transfers use bounded
ranges and verify the complete BLAKE3 handle before admission.

The first installed local snapshot starts anti-entropy immediately. Thereafter
each 30-second period admits at most `K = 20` peers from a fair rotating cursor,
with at most eight live walks. The pending queue therefore never contains more
than one period's eligible budget and never carries backoff-delayed peers.
Slow sweeps do not accumulate catch-up periods, repeated snapshot installation
cannot manufacture work, and every configured or stored PEER candidate in an
eventually stable peer set is selected as the cursor cycles.

## Routing and discovery

`PeerConfig.peers` contains bootstrap endpoint IDs or tickets. A successful
authorized synchronization session and synchronized
`PEER(team, peer_public_key)` evidence can add routing candidates. PEER set
union carries those candidates transitively without creating a mutable roster.

An endpoint remains only a candidate until the two proof handshakes succeed.
Every returned payload is checked against its requested BLAKE3 handle.

The same routing table powers an iterative Kademlia-style XOR lookup with
256 buckets, `K = 20`, and `alpha = 3`. Configured and synchronized PEER
evidence are periodic anti-entropy targets; identities named only by DHT
referrals are not. A referred identity becomes verified routing state only
after a direct reciprocal capability-authenticated response.

Artifact publication has one production path. `ArtifactOfferStore` records
grow-only local willingness to serve `c`; `Peer::refresh` observes that policy
outside the coherent semantic `StoreSnapshot`, so an OFFER-only append does
not rebuild semantic inventory. The host intersects the complete OFFER
snapshot with the Blob keys in its current immutable serving snapshot. Only
that intersection is published, and only under a serving direction.

Newly active offers are announced immediately. The host derives a team-scoped
provider key `r = provider_key(team, c)` for every active artifact, builds one
canonical BLAKE3-Merkle PATCH, and groups its keys by the fixed first byte of
`r`. Publication therefore has at most 256 prefix shards regardless of the
number of offered artifacts. Each shard is routed under a separate
team-and-prefix DHT key to the `K` closest responsive nodes. A node with no
remote routing evidence treats its local shard leases as a sane success.
Once any configured, synchronized, or learned remote route exists, at least
one remote directory must accept the announcement: local self-insertion alone
does not turn an outage into an apparent replicated success. Failed or
capacity-rejected replication retries with bounded exponential backoff.

Publication first probes `(prefix, digest, count)`. An equal root renews the
receiver-local lease without transferring or walking membership. A changed
root requests one strictly ascending body of full 32-byte provider keys. The
receiver checks the authenticated provider identity, prefix, count, bounds,
ordering, and rebuilt PATCH digest before atomically replacing the shard and
its prefix-directory entry. Capacity or validation failure preserves the old
valid shard. Missing prefixes receive no imperative removal; their leases
simply expire. Expiry reclamation is bounded per request, while deadline checks
prevent unreclaimed stale shards from being returned. A pathologically
oversized single prefix is reported and omitted while neighboring exact shards
continue to publish and renew.

Successful prefix leases are renewed at half their lifetime. An ordered
due-time scheduler admits at most `alpha = 3` changed or due prefixes at once,
uses fair backoff, and retains successful deadlines across changed-root retry.
Cover reconstruction is keyed by the durable OFFER snapshot and the immutable
Blob-component root: peer, record, or proof churn reuses the existing cover
instead of rehashing every offer. Restart reobserves the durable OFFER snapshot.
An absent artifact, a cleared serving snapshot, or `ReadOnly` direction leaves
the offer dormant; if residency or a serving view later returns, normal
snapshot observation activates it. In-flight stale hints need no cancellation
protocol because receivers expire leases.

Receiver admission is bounded only by the aggregate directory weight
`(live shards, live memberships) <= (65,536, 2^24)`. It is work-conserving:
any provider may use capacity which is presently free, with no lower
per-provider ceiling. Replacement computes the weight after removing the old
shard and adding its candidate, so a same-weight replacement remains possible
at either exact boundary. A prefix rejected because the aggregate directory is
full remains soft unknown and retries after capacity expires. Admission is
therefore first-arrival: one authorized provider may occupy the whole bounded
directory until its leases expire. Principal fairness is intentionally outside
this soft discovery primitive.

A reader that already knows `c` performs iterative `FIND_NODE`, asks those
replicas for the corresponding team-and-prefix directory, and sends the raw
artifact handle in `PROVIDER_GET`. Each replica derives `r` itself and returns
only providers whose live sorted prefix shard contains that exact key. Both the
candidate scan and result fan-out are bounded and rotate across calls without
capping stored memberships. Because this is a soft directory, one lookup may
return fewer hints (including none) under adversarially dense occupancy; it
never returns a false membership, and repeated lookups advance through every
candidate. The reader then fetches from the returned providers.
Every provider operation and exact transfer uses the existing reciprocal
CONNECT and SYNC_TEAM authorization; transfer also checks that the received
bytes hash to `c`. A holder without an active resident OFFER is honestly
unavailable, and the implementation never falls back to probing every known
peer. OFFER itself grants no authority, retention, demand, or inventory
membership.

## Local quality of service

Direction controls which work this node performs; it is never sent as an
authorization claim:

- `Bidirectional` pulls remote inventories, serves local inventory and blobs,
  and may retain an authenticated inbound peer as PEER routing evidence.
- `ReadOnly` pulls and services local WANTs, but does not serve local inventory
  or blobs.
- `WriteOnly` serves local data, but never pulls, demand-fetches, or records
  inbound readers as pull routes.

The blob mode is independent:

- `Demand` skips the broad Blob inventory. A durable exact blob WANT uses the
  DHT provider path for that already-known handle.
- `Mirror` also walks the complete authorized Blob inventory and fetches every
  missing resident blob in bounded ranges.

Mirror describes synchronization work, not a retention promise. An evicting
store may discard mirrored bytes later; a durable mirror requires a
non-evicting sink or another explicit retention contract. Direction and blob
mode cannot widen the server-selected inventory or either capability.

## Durable WANTs

`WantStore` records local operational interest. Its canonical keys ask for an
exact blob, a matching native `MERGE`, or a matching native `DERIVE` record.
WANTs remain durable and are retried with bounded backoff; temporary
unreachability means “not obtained yet,” never “absent.”

Blob WANTs use explicit DHT provider discovery followed by exact authenticated
`GET_BLOB` in Demand mode. The received bytes are content-checked, landed, and
flushed before the WANT is counted as fulfilled.

Collection-operation WANTs need no network operation. The full team inventory
already converges all collection records, including conflicting valid
answers. The reconciler refreshes the peer and asks the local indexed store for
matching records. No local match leaves the WANT pending while periodic
inventory sweeps continue. Obtaining a receipt's result content is a separate
blob WANT.

## Lattice-aware exact collection reuse

Inventory synchronization converges immutable collection equations, but it
does not blindly mirror every artifact named by them. An exact derived
collection can instead ask `ensure_exact_derived` to reuse a physical cover
already materialized elsewhere:

1. the caller supplies one frozen, opaque source cover whose payload
   members and dependencies are resident locally;
2. the peer freezes target `MERGE` and `DERIVE` result handles from its current
   converged record view as speculative availability offers;
3. the core exact resolver first accepts any complete resident cover, then—if
   local bytes are insufficient—selects an exact antichain from those offers;
4. the network fetches only the selected handles with authenticated exact
   `GET_BLOB` requests and probes the same cover again;
5. an absent, malformed, or stale offer is removed and the resolver replans;
   when no offered cover remains, ordinary local `ensure_exact` construction
   is the fallback.

Offers never authorize source truth and never override a complete local cover.
Fetched target bytes are content-checked, representation-validated on the next
probe, and admitted only as local cache evidence. These speculative reads do
not create durable WANTs; callers that want long-lived residency must state
that policy separately.

Current `MERGE` and `DERIVE` equations are unsigned reproducible evidence. The
exact resolver therefore recomputes their canonical results from authenticated
source roots before trusting the offered identities. Remote reuse saves
bandwidth and avoids retaining redundant artifacts, but does not yet attest
away that derivation computation. A future cheaper exact witness may improve
compute reuse without weakening this boundary.

## Wire surface

All direct operations use
`PILE_SYNC_ALPN = "/triblespace/pile-sync/14"`. One QUIC stream carries one
strictly framed operation:

| Operation | Byte | Purpose |
|---|---:|---|
| `GET_BLOB` | `0x02` | read one exact current blob after both authorizations |
| `OP_AUTH` | `0x05` | exchange subject-bound CONNECT bundles; mandatory first stream only |
| `PROVIDER_PROBE` | `0x06` | compare or renew one authenticated provider-prefix root |
| `PROVIDER_GET` | `0x07` | read live hints for one already-known artifact |
| `INVENTORY_AUTH` | `0x08` | exchange SYNC_TEAM bundles and install one connection-local session |
| `INVENTORY_MANIFEST` | `0x09` | read the four ordered component roots and generation |
| `INVENTORY_NODE` | `0x0A` | read one expected-digest node from a pinned component |
| `INVENTORY_BLOB_RANGE` | `0x0B` | read at most one bounded range from a pinned Blob root |
| `FIND_NODE` | `0x0C` | read up to `K` directly verified routes nearest one XOR key |
| `PROVIDER_BODY` | `0x0D` | atomically install one changed canonical prefix body |

Provider-cover leases are bounded receiver-local soft state; there is no remote
semantic write, collection-evidence, operation-receipt, blob-child, custody, or
replica operation. Receivers admit strictly checked results through their own
local store boundary.

## CLI

The `trible` CLI selects both resident proofs explicitly:

```text
trible pile net identity [--key PATH]
    Initialize the key if needed and print this node's iroh identity.

trible pile net inventory <PILE>
    Print the bound team and exact generation, count, and PATCH root of every
    locally sampled /14 inventory component.

trible pile net status <PILE>
    --team-root HEX --connect-proof ID --sync-proof ID [--key PATH]
    Resolve both exact native bundles and verify their roots, actions,
    resources, leaves, Invoke authority, and current validity.

trible pile net sync <PILE>
    --team-root HEX --connect-proof ID --sync-proof ID
    [--peers ID_OR_TICKET,...] [--key PATH]
    [--direction bidirectional|read-only|write-only]
    [--blobs demand|mirror]
    [--duration SECS] [--quiescent-for SECS]
    Run foreground periodic team-inventory reconciliation.
```

`team create` issues founder CONNECT and SYNC_TEAM proofs and reports both
IDs. `team invite` extends two explicitly selected parent paths and writes one
versioned portable artifact. `team join` verifies both bundles against the
separately supplied team root and invitee key before one idempotent store
write. See [Capability Authorization](capability-auth.md) for the proof model.

`inventory` is local, read-only evidence over the same canonical manifest the
wire protocol reconciles. It derives the team from the pile's existing store
scope and fails when the pile is unbound, conflicted, or changes while the
snapshot is constructed. Equal generations prove equality of all four sampled
inventory sets without depending on append order or physical pile bytes. With
`sync --blobs demand`, blob divergence is intentional; compare the PEER,
collection-record, and capability-proof count/root pairs for exact structural
equality.
Even equal sampled generations are not a proof that every possible swarm
member has participated or that no later write can occur.

Without a lifecycle flag, `sync` runs until interrupted. `--duration` provides
a wall-clock bound. `--quiescent-for` is only a local observation that no
recent inventory admission or WANT fulfillment occurred; it is not a
distributed proof of convergence.

## Invariants worth retaining

- Synchronization is componentwise set union, never last-writer-wins state.
- The backing store is a single-team security boundary.
- CONNECT and SYNC_TEAM are reciprocal subject-bound proofs; SYNC_TEAM
  separately authorizes every disclosure.
- Proof presence, PEER evidence, DHT referrals, and provider hints grant no
  authority.
- Bounded fair pairwise PATCH reconciliation establishes epidemic progress;
  there is no separate wake plane.
- Every Merkle walk pins exact roots and fails closed when a snapshot is gone.
- Demand is explicit local interest; inventory observation creates no hidden
  WANT.
- Provider publication is exactly `OFFER ∩ resident Blob snapshot ∩ serving
  QoS`; at most 256 independently expiring prefix leases replace per-artifact
  state, and OFFER-only changes never rebuild synchronized inventory.
- Mirror residency is not retention.
- Operation answers are ordinary converged collection evidence, including
  conflicts.
- Temporary unreachability remains unknown, never absent.
