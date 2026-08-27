# Distributed Sync

The [`triblespace-net`](https://github.com/triblespace/triblespace-rs/tree/main/triblespace-net)
crate synchronizes one team's TribleSpace store over
[iroh](https://www.iroh.computer/). It has one data protocol and two
deliberately narrow discovery mechanisms:

- authenticated QUIC performs authorized inventory walks and exact blob reads;
- a team-derived gossip topic carries lossy generation wake hints.

Gossip neighbors and stored routing evidence grant no authority. Periodic
authenticated inventory sweeps are the correctness path, so dropped or
duplicated wake frames affect latency rather than convergence.

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
and deterministic gossip topic. The backing store must be dedicated to this
one team. Only PEER evidence contains a team key intrinsically; collection
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
root, the transport-authenticated caller key, and the current time. A rejected
connection closes. A bounded accepted connection also closes when the
effective inclusive CONNECT validity interval expires.

After CONNECT, `INVENTORY_AUTH` presents the complete SYNC_TEAM bundle once
for that connection. Successful verification installs one team-selected,
validity-bounded session. The client cannot nominate another scope, and a
second authorization attempt on the same connection is rejected. Manifest,
node, blob-range, and known-hash `GET_BLOB` requests all require this live
session. Knowing a content hash is not disclosure authority.

Neither handshake searches the store for a proof or fetches missing claims.
Each selected bundle contains its exact ordered proof and claim closure inline.
Stored proof presence remains evidence only.

## Merkle reconciliation

Each component is an ordered PATCH inventory with a canonical BLAKE3 Merkle
summary. A manifest contains the four component tags, leaf counts, and roots in
fixed order. Its generation hash binds the version, team key, and all four
entries. The generation is useful as a wake value and cache key, but it is not
an authorization token or evidence of global completeness.

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
reused, but its small reader-bearing wrapper is refreshed so an unchanged key
set cannot pin an obsolete mmap or Yard generation indefinitely. Snapshot
installation refreshes an already-pinned matching Blob root without implicitly
pinning an unrequested one. Record-only
churn therefore neither retains old blob readers nor duplicates the other
inventory trees.
Later node and blob-range requests repeat the exact root. If that snapshot has
expired or been evicted, the server returns `snapshot unavailable`; it never
splices bytes from its current state into the older walk. The client obtains a
fresh manifest and restarts that component. Mirror blob transfers use bounded
ranges and verify the complete BLAKE3 handle before admission.

Periodic sweeps compare every eligible route even if gossip is silent. A
gossip frame contains only a version, the exact team key, and a manifest
generation. It schedules an earlier authenticated check; the delivering mesh
neighbor is not presumed to be the publisher and is never inserted as a route.

## Routing and discovery

`PeerConfig.peers` contains bootstrap endpoint IDs or tickets. A successful
authorized synchronization session and synchronized
`PEER(team, peer_public_key)` evidence can add routing candidates. PEER set
union carries those candidates transitively without creating a mutable roster.

An endpoint remains only a candidate until the two proof handshakes succeed.
Every returned payload is checked against its requested BLAKE3 handle.

The team key derives the production gossip topic, preventing an authorized
store from accidentally rendezvousing on another team's mesh. Joining that
topic, observing a generation, or appearing as a gossip neighbor still grants
no transport or inventory authority.

## Local quality of service

Direction controls which work this node performs; it is never sent as an
authorization claim:

- `Bidirectional` pulls remote inventories, publishes wake hints, serves local
  inventory and blobs, and may retain an authenticated inbound peer as PEER
  routing evidence.
- `ReadOnly` pulls and services local WANTs, but neither publishes nor serves
  local inventory or blobs.
- `WriteOnly` publishes and serves local data, but never pulls,
  demand-fetches, or records inbound readers as pull routes.

The blob mode is independent:

- `Demand` skips the broad Blob inventory. A durable exact blob WANT tries the
  configured and learned authenticated routes.
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

Blob WANTs use exact authenticated `GET_BLOB` over known team routes in Demand
mode. The received bytes are content-checked, landed, and flushed before the
WANT is counted as fulfilled.

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

1. the caller supplies one frozen, authenticated source ticket whose source
   data and dependencies are resident locally;
2. the peer freezes target `MERGE` and `DERIVE` result handles from its current
   converged record view as speculative availability offers;
3. the core exact resolver first accepts any complete resident cover, then—if
   local bytes are insufficient—selects an exact antichain from those offers;
4. the network fetches only the selected handles with authenticated exact
   `GET_BLOB` requests and probes the same ticket again;
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
`PILE_SYNC_ALPN = "/triblespace/pile-sync/11"`. One QUIC stream carries one
strictly framed operation:

| Operation | Byte | Purpose |
|---|---:|---|
| `GET_BLOB` | `0x02` | read one exact current blob after both authorizations |
| `OP_AUTH` | `0x05` | present CONNECT bundle; mandatory first stream only |
| `INVENTORY_AUTH` | `0x08` | install the one connection-local SYNC_TEAM session |
| `INVENTORY_MANIFEST` | `0x09` | read the four ordered component roots and generation |
| `INVENTORY_NODE` | `0x0A` | read one expected-digest node from a pinned component |
| `INVENTORY_BLOB_RANGE` | `0x0B` | read at most one bounded range from a pinned Blob root |

There is no remote write, collection-evidence, operation-receipt, blob-child,
custody, or replica operation. Receivers admit strictly checked results through
their own local store boundary.

## CLI

The `trible` CLI selects both resident proofs explicitly:

```text
trible pile net identity [--key PATH]
    Initialize the key if needed and print this node's iroh identity.

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

Without a lifecycle flag, `sync` runs until interrupted. `--duration` provides
a wall-clock bound. `--quiescent-for` is only a local observation that no
recent inventory admission or WANT fulfillment occurred; it is not a
distributed proof of convergence.

## Invariants worth retaining

- Synchronization is componentwise set union, never last-writer-wins state.
- The backing store is a single-team security boundary.
- CONNECT admits transport; SYNC_TEAM separately authorizes every disclosure.
- Proof presence, PEER evidence, and gossip grant no authority.
- Gossip wakes reconciliation; periodic authenticated sweeps establish
  eventual progress.
- Every Merkle walk pins exact roots and fails closed when a snapshot is gone.
- Demand is explicit local interest; inventory observation creates no hidden
  WANT.
- Mirror residency is not retention.
- Operation answers are ordinary converged collection evidence, including
  conflicts.
- Temporary unreachability remains unknown, never absent.
