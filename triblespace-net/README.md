# triblespace-net

Collection-scoped anti-entropy for TribleSpace over
[iroh](https://www.iroh.computer). A peer retains an immutable activation
overlay for each explicitly active collection. One repair stream reconciles
the exact product of two grow-only PATCHes: native collection records and
portable WRITE-evidence bundles.

The user-facing surface is `Peer<S>`, a synchronous store wrapper backed by an
async host. `Peer::refresh` drains verified repair events, crosses one storage
flush barrier, and only then replaces the immutable snapshots served to other
peers. There is no global team inventory, remote mutable head, replica roster,
or separate authority database.

## Getting started

Most users should enable this crate through the facade crate's `net` feature:

```toml
[dependencies]
triblespace = { version = "0.47", features = ["net"] }
```

```rust,ignore
use triblespace::net::peer::{
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
collection and carries the caller's complete portable READ(C) proof bundles.
The server verifies those bundles against the collection descriptor before it
reveals the repair manifest or any PATCH leaf. An endpoint with WRITE(C) but
without READ(C) cannot learn collection records, WRITE evidence, or roots.

DHT `FIND_NODE`, provider-directory operations, and exact `GET_BLOB` by an
already-known immutable handle remain bearer/public mechanisms. Provider
advertising is built only from the store's collection disclosure snapshot, so
restricted material is never published as a provider hint merely because the
bytes are resident.

## Repair and wake

Periodic pairwise repair is authoritative anti-entropy. For each active
collection, the caller opens one bidirectional stream, presents READ(C), pins
the returned record and WRITE-evidence roots, and walks only missing PATCH
nodes. Complete proof bundles include their claim bytes, so landing later
WRITE evidence can activate an older record without a separate claim-fetch
protocol.

Production iroh peers also subscribe to stock `iroh-gossip` topics keyed
exactly by the 32-byte collection handle. A wake contains only a signed origin
endpoint and the opaque activation root. A root mismatch schedules ordinary
READ-authorized repair from that signed origin; the wake itself carries no
authority or collection state. Missed or lagged wakes are harmless because
the periodic repair path remains active.

Direction is local policy:

- `Bidirectional` pulls active collections and serves admitted readers.
- `ReadOnly` pulls but does not serve local collection state or bearer data.
- `WriteOnly` serves admitted readers and bearer data but does not initiate
  repair or service local WANTs.

Configured endpoint addresses are bootstrap repair targets. DHT referrals and
wake origins may provide transient routes, but no durable PEER record is
created or consumed by the network host.

## Exact content

Durable blob WANTs use the authenticated DHT to find policy-approved provider
hints and then fetch the exact known handle. Exact reads are independent of
collection repair: no broad blob inventory or mirror mode exists. Collection
operation WANTs observe matching records through the local indexed record
selection after `Peer::refresh`.

The full model, wire formats, authorization boundaries, and CLI surface live
in the book's [Distributed Sync](https://docs.rs/triblespace/latest/triblespace/)
chapter.

## Crate layout

- `collection_activation` — per-collection record and WRITE-evidence PATCHes
- `collection_session` / `collection_wire` — one READ-authorized repair stream
- `patch_repair` — root-pinned Merkle difference walker
- `peer` — synchronous store wrapper, durable admission, and exact WANT reads
- `reconcile` — durable WANT observation and exact blob fulfillment
- `provider` / `routing` — bounded bearer provider directory and XOR routing
- `protocol` — public direct-operation framing
- `host` — immutable overlays, connection pool, wake bridge, and scheduler
- `transport` — production iroh and deterministic simulation transports
- `identity` — persistent network signing-key handling
