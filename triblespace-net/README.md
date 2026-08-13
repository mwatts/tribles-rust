# triblespace-net

Distributed collection synchronization for TribleSpace over
[iroh](https://www.iroh.computer). Immutable signed collection evidence is
flooded through a team gossip mesh, content-addressed blobs are discovered
through a DHT, and exact reads travel over authenticated QUIC.

The user-facing surface is `Peer<S>`, a wrapper that gives a local store a
network presence without turning the synchronous storage traits into async
APIs. `Peer::refresh` admits pending evidence and publishes newly appended
records; the background host owns transport and periodically replays the
current durable publication frontier for late joiners.

## Getting started

Most users should enable `triblespace-net` through the facade crate's `net`
feature rather than depending on this crate directly:

```toml
[dependencies]
triblespace = { version = "0.47", features = ["net"] }
```

```rust,ignore
use triblespace::net::peer::{Peer, PeerConfig, SyncDirection};

let pile = triblespace::core::repo::pile::Pile::open(path)?;
let mut peer = Peer::new(pile, signing_key.clone(), PeerConfig {
    peers: vec![bootstrap_endpoint],
    gossip: true,
    team_root,
    self_cap,
    direction: SyncDirection::Bidirectional,
});

// Publish/admit external collection writes and drive lazy WANT fulfillment.
peer.refresh();
```

Gossip transports only a strictly verified publication grant plus its matching
signed `COMMIT`. It does **not** fetch the collection descriptor, data,
metadata, or attachments and does not manufacture a WANT. Those resources
remain independently content-addressed and are fetched only when local policy
asks for them.

The full model, wire formats, authorization boundaries, durable WANT behavior,
and CLI surface live in the book's
[Distributed Sync](https://docs.rs/triblespace/latest/triblespace/) chapter.

## Crate layout

- `peer` — synchronous store wrapper and collection-evidence admission
- `collection_sync` — prepare, authorize, and durably admit sparse evidence
- `collection_wire` — canonical dense codecs and exact collection RPCs
- `reconcile` — fulfillment of durable blob, merge, and derive WANTs
- `protocol` — authenticated read-only QUIC protocol
- `policy` — private capability lifecycle collections
- `transport` — production iroh and deterministic simulation transports
- `identity` — persistent network signing-key handling

`host` and `channel` are implementation details: the async transport loop and
the narrow bridge between it and `Peer`.
