# triblespace-net

Collection-scoped anti-entropy for TribleSpace over
[iroh](https://www.iroh.computer). A peer retains an immutable activation
overlay for each explicitly active collection. One repair stream reconciles
the exact product of two grow-only PATCHes: currently WRITE-admitted signed
COMMITs and portable WRITE-evidence bundles.

The user-facing surface is `Peer<S>`, a synchronous store wrapper backed by an
async host. `Peer::refresh` drains verified repair events, crosses one storage
flush barrier, and only then replaces the immutable snapshots served to other
peers. There is no global team inventory, remote mutable head, replica roster,
or separate authority database.

## Getting started

Networking is a downstream capability and is consumed directly rather than
through the core TribleSpace facade:

```toml
[dependencies]
triblespace = "0.47"
triblespace-net = "0.47"
```

```rust,ignore
use triblespace_net::peer::{
    Peer, PeerConfig, ReconcileDirection, ReconcileQos,
};

let pile = triblespace::core::repo::pile::Pile::open(path)?;
let mut peer = Peer::new(
    pile,
    signing_key,
    PeerConfig {
        peers: vec![bootstrap_endpoint],
        qos: ReconcileQos {
            direction: ReconcileDirection::Bidirectional,
            ..ReconcileQos::default()
        },
    },
)?;
peer.activate_collection(collection_handle);

loop {
    peer.refresh();
    std::thread::sleep(std::time::Duration::from_millis(100));
}
```

Activation is ephemeral process state. It writes no OFFER/GOSSIP marker and
does not create an ambient collection registry.

## Authority and disclosure

The QUIC/TLS connection authenticates endpoint identities but grants no team
or collection authority. Every collection repair request names exactly one
collection. The repair client presents its bounded READ(C) witness first. The
server verifies the TLS client before revealing a manifest or PATCH leaf; the
publisher itself needs no READ(C). Claims and proofs are non-secret
authorization certificates. A caller without READ(C) receives no collection
manifest, PATCH leaf, record, WRITE evidence, or root; merely knowing C grants
no disclosure.

DHT `FIND_NODE` and provider-directory operations use two independent opaque
namespaces. KDF(C) locates participants for READ(C)-authorized collection
repair. KDF(H) locates providers of exact resident content without naming a
collection. Each exact-content lease carries a token derived from H and the
provider endpoint, which a requester who knows H verifies before dialing.

The exact stream never sends H. The authenticated provider proves knowledge of
H first, bound to both TLS endpoint identities; only then does the requester
return its independently domain-separated proof. A false locator advertiser
therefore cannot make the requester disclose H or masquerade as a provider.
Returned bytes are accepted only when they hash to H. READ(C) is not consulted
by exact GET and remains exclusively the collection-repair disclosure boundary.

## Repair and wake

Periodic pairwise repair is authoritative anti-entropy. For each active
collection, the caller opens one bidirectional stream, presents READ(C), pins
the returned record and WRITE-evidence roots, and walks only missing PATCH
nodes. Complete proof bundles include their claim bytes, so landing later
WRITE evidence can activate an older record without a separate claim-fetch
protocol.

Production iroh peers also subscribe to stock `iroh-gossip` topics keyed by a
domain-separated one-way image of the 32-byte collection handle. A 177-byte
nonce-v3 wake contains only version, signed origin endpoint, separate opaque
semantic and payload roots, and a fresh nonce. Demand peers react only to the
semantic root; Full peers react to both. A mismatch schedules ordinary
READ-authorized repair from that signed origin; the wake itself carries no
authority or collection state. Missed or lagged wakes are harmless because
bounded sampled anti-entropy through leased signed wake origins remains active.

Direction is local policy:

- `Bidirectional` pulls active collections and serves admitted readers.
- `ReadOnly` pulls but does not serve local collection state or bearer data.
- `WriteOnly` serves admitted readers and bearer data but does not initiate
  repair or service local WANTs.

Configured endpoint addresses bootstrap gossip and DHT only. Repair targets
come from signed wake origins or endpoint-bound KDF(C) leases. Exact-content
targets come from KDF(H) leases. Unrelated configured peers never receive C or
its proofs.

## Exact content

`BlobReplication::Demand` keeps exact reads lazy. A bare durable
`WantRequest::Blob(H)` asks the reconciler to discover and obtain those exact
bytes through KDF(H). It needs no collection descriptor, activation, or READ
proof. `BlobReplication::Full` instead walks a third, stream-pinned
80-byte-key disclosure-forest PATCH inside an admitted collection-repair
session. Each key commits to depth, parent, aligned chunk index, and child
handle; the receiver accepts roots only from locally WRITE-admitted COMMITs and
descendants only after verifying the parent bytes. The same command and byte
budgets paginate large mirrors across ordinary repair sessions.

All exact requests share the one `Blob(H)` identity. A successful landing
satisfies the durable request locally; failed discovery leaves it pending.
Collection membership, proof state, and activation are irrelevant to that
exact-content operation.

The full model, wire formats, authorization boundaries, and CLI surface live
in the book's [Distributed Sync](https://docs.rs/triblespace/latest/triblespace/)
chapter.

## Crate layout

- `collection_activation` — per-collection record and WRITE-evidence PATCHes
- `collection_session` / `collection_wire` — one READ-authorized repair stream
- `patch_repair` — root-pinned Merkle difference walker
- `peer` — synchronous store wrapper, durable admission, and local WANT intent
- `reconcile` — durable WANT observation and reproducible-operation fulfillment
- `provider` / `routing` — bounded bearer provider directory and XOR routing
- `protocol` — public direct-operation framing
- `host` — immutable overlays, connection pool, wake bridge, and scheduler
- `transport` — production iroh and deterministic simulation transports
- `identity` — persistent network signing-key handling
