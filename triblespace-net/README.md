# triblespace-net

Authorized team inventory synchronization for TribleSpace over
[iroh](https://www.iroh.computer). A peer periodically reconciles four
monotone store components—PEER routing evidence, collection records,
capability proofs, and optionally blobs—through authenticated Merkle walks.
These bounded pairwise PATCH unions are the epidemic exchange itself; there is
no separate broadcast wake plane. Exact artifacts are located through the
authenticated DHT after explicit provider publication.

The user-facing surface is `Peer<S>`, a synchronous store wrapper backed by an
async host. `Peer::refresh` drains verified network events, crosses one storage
flush barrier, and only then replaces the immutable snapshot served to other
peers. No remote mutable head, receipt RPC, replica roster, or second authority
database is involved.

## Getting started

Most users should enable this crate through the facade crate's `net` feature:

```toml
[dependencies]
triblespace = { version = "0.47", features = ["net"] }
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

`team` is the exact Ed25519 trust root and inventory scope. The backing store
must be dedicated to that team because collection records, proofs, and blobs
have content identities but no intrinsic team label.

Every connection first presents a complete CONNECT proof bound to its
transport key, and the server returns its own bounded CONNECT proof in the same
round trip. Before any inventory or blob disclosure, the client and server
repeat that reciprocal exchange for SYNC_TEAM exactly once on the connection.
Both sides' proofs must be rooted at `team`, invoke their exact action for the
TLS-authenticated endpoint key, and be current. Startup rejects local proof
configuration for another endpoint, team, action, or validity interval.

The initiating CONNECT proof necessarily reaches the exact dialed TLS identity
before that endpoint proves CONNECT authority. It is non-bearer evidence bound
to the initiator's key and sent over TLS to the identity the client intended to
dial; the proof itself is not cryptographically bound to the receiver key. It
is not a confidential or zero-knowledge credential. No SYNC proof, element
identity, query, or data request is sent until the returned CONNECT proof
verifies; useful requests additionally wait for reciprocal SYNC_TEAM. Knowing
a team root or content hash, or holding PEER evidence, is not sufficient.

## Reconciliation policy

PEER, collection-record, and capability-proof inventories always converge when
pulling is enabled. Blob behavior is local policy:

- `Demand` skips the broad blob inventory. Durable blob WANTs locate an
  explicitly published provider through the DHT, then use exact,
  SYNC_TEAM-authorized `GET_BLOB`.
- `Mirror` also walks the complete authorized blob inventory and fetches
  missing bytes in bounded ranges. Mirroring is a synchronization policy, not
  a retention promise; an evicting store may discard bytes later.

Direction is also local policy:

- `Bidirectional` pulls and admits remote inventory, serves its local
  inventory, and learns authenticated inbound peers.
- `ReadOnly` pulls and services local WANTs but does not serve local data.
- `WriteOnly` serves local data but never pulls, demand-fetches, or admits
  inbound readers as local PEER evidence.

Configured endpoint addresses are bootstrap routes. Successful authorized
sessions and synchronized `PEER(team, peer)` evidence can add routing
candidates. Every 30-second period admits at most `K = 20` candidates from a
fair rotating cursor, with at most eight live sweeps. Slow or failed peers
therefore cannot create an unbounded pending scan, while every stored or
configured peer in an eventually stable set is revisited after backoff. A first
installed snapshot starts one period immediately; later snapshots cannot
manufacture extra budget.

Collection-operation WANTs need no special RPC. Since every native collection
record is already part of the team inventory, the reconciler refreshes the
peer and uses the store's indexed record selection to observe matching MERGE
or DERIVE receipts. Absence stays pending; conflicting answers remain in the
grow-only record set.

The full model, wire formats, authorization boundaries, and CLI surface live
in the book's [Distributed Sync](https://docs.rs/triblespace/latest/triblespace/)
chapter.

## Crate layout

- `peer` — synchronous store wrapper, durable admission, and Demand reads
- `inventory` — four canonical team-scoped PATCH inventories and QoS policy
- `inventory_reconcile` — root-pinned Merkle difference walker
- `inventory_wire` — bounded SYNC_TEAM authorization, node, and blob-range codecs
- `reconcile` — durable WANT observation and exact blob fulfillment
- `protocol` — CONNECT authentication and exact blob framing
- `host` — transport, snapshots, connection pool, and periodic scheduler
- `transport` — production iroh and deterministic simulation transports
- `identity` — persistent network signing-key handling
