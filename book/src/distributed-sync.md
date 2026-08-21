# Distributed Sync

The [`triblespace-net`](https://github.com/triblespace/triblespace-rs/tree/main/triblespace-net)
crate synchronizes TribleSpace's native collection algebra over
[iroh](https://www.iroh.computer/). Its three transport mechanisms have
separate jobs:

- a team gossip mesh discovers immutable signed collection evidence;
- a DHT discovers providers for content-addressed blobs; and
- authenticated QUIC answers exact blob, collection, and operation-receipt
  questions.

The user-visible surface is `Peer<S>`. It wraps a local store, owns the async
network host, and continues to expose synchronous storage traits. Network
activity does not introduce a remote mutable head or a distributed
compare-and-swap operation.

Enable it through the facade crate's `net` feature:

```toml
[dependencies]
triblespace = { version = "x.y.z", features = ["net"] }
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

peer.refresh();
```

## The synchronized object is evidence

A native collection is a grow-only algebra of signed `COMMIT` assertions and
reproducible `MERGE` and `DERIVE` equations. There is no distinguished head.
Concatenating two stores unions their evidence, so synchronization uses that
same operation rather than reconstructing a Git-like branch protocol.

One `COMMIT` is a 192-byte signed assertion containing:

- the canonical collection descriptor handle;
- the data identity;
- the metadata archive handle;
- the author's public key; and
- the author's Ed25519 signature.

The signature proves who authored the assertion. It does not force a receiver
to treat that author as ground truth. Author trust belongs to the resolver that
selects a collection view, where all relevant local policy is available.

Publication permission is a property of the collection, not a second record.
The descriptor's optional `collection_reach` attribute names a reach law, and
because a collection handle is the hash of its descriptor, a collection that
travels and one that stays put are *different collections*. There is nothing to
sign after the fact and nothing to forget: an author who commits into a
collection whose identity declares it public has published, and one who commits
into a collection that declares nothing has not.

A peer may relay a commit exactly when it can resolve that commit's collection
descriptor and that descriptor declares a reach law it implements. A descriptor
it cannot resolve is a refusal — permission it cannot read is permission it does
not have — and so is a law it does not recognise, which is why reach names a law
rather than carrying a boolean. `Collection::commit` writes its descriptor as a
commit dependency, so a publisher's own store always holds its own permission.

The receiving side does not consult descriptors. Admitting evidence a peer
handed you leaks nothing, and requiring residency there would contradict the
sparse-admission rule above. The guarantee holds transitively: a receiver that
later serves is a relay, and answers the question then.

The canonical evidence frame is the 192-byte commit. The mesh frame adds one
kind byte and an eight-byte anti-deduplication nonce, for 201 bytes in total.
The nonce changes transport identity for periodic replay but is not signed and
has no semantic meaning.

The receiver strictly verifies both signatures and their author/collection
correspondence before admitting the pair to its two grow-only stores. The mesh
carrier is only a relay; it is neither author nor authority. Duplicate delivery
is reduced by intrinsic commit identity, and one drained batch crosses one
storage flush barrier.

## Sparse discovery is deliberate

Gossiping a `COMMIT` does not transfer any referenced blob. In
particular, the receiver may know the exact handles of the descriptor, data,
metadata, and attachments while possessing none of their bytes. Admission also
does not create a WANT for any of them.

This separation is what lets a node cheaply learn a large global frontier and
then apply its own cache, trust, and derivation policy. Evidence answers “what
has been asserted?” A WANT answers “what work or content should this node
obtain?” Conflating them would make every observation an involuntary full
replica.

`Peer::refresh` performs both directions of the sparse exchange:

1. drain pending evidence frames, canonicalize and admit them;
2. update the host's immutable serving snapshot;
3. announce newly added blobs to the DHT; and
4. gossip newly visible relayable commits.

Construction drives one initial refresh, so existing evidence is published
without a separate startup ritual. The host later reads the current durable
snapshot every 30 seconds and replays the exact publication frontier with fresh
nonces. It does not maintain a second unbounded ledger mirror.

## Durable WANTs

`WantStore` records local operational interest. Its canonical request key has
three variants:

- `Blob(handle)` asks to obtain content and retain it while cache policy
  permits;
- `Merge(collection, low, high)` asks whether a matching native `MERGE`
  receipt exists; and
- `Derive(source, target, input)` asks whether a matching native `DERIVE`
  receipt exists.

The long-running `Reconciler` services these requests without deleting them.
An unavailable answer is normal: the WANT remains durable and retries with
bounded exponential backoff.

Blob fulfillment first tries configured or currently live team peers as
routing candidates, then falls back to DHT providers. A routing candidate is
not presumed to hold the blob; every returned payload must still match the
requested BLAKE3 handle.

Merge and derive questions have no result hash to feed into the DHT. They are
therefore probed against the explicitly configured authenticated peers. Every
exact native receipt returned before the deadline is unioned into the local
`CollectionStore`. A sweep only becomes complete when every configured peer
answered; partial answers remain useful evidence but do not turn temporary
unreachability into a proof of absence. Obtaining the result bytes is a
separate `Blob(result)` WANT.

## Authenticated read protocol

All point-to-point operations use `PILE_SYNC_ALPN =
"/triblespace/pile-sync/5"`. The first stream on a connection must be
`OP_AUTH`, which proves a capability chain back to the configured team root.
The remaining protocol is read-only:

| Operation | Question | Response |
|---|---|---|
| `GET_BLOB` | exact 32-byte content handle | bytes or missing |
| `CHILDREN` | exact parent handle | referenced present handles |
| `COLLECTION_EVIDENCE` | exact collection descriptor handle | canonical commits the descriptor permits |
| `COLLECTION_OPERATION_RECEIPTS` | exact merge/derive WANT key | canonical matching receipts |

`fetch_collection_evidence_from` returns strictly verified but inert sparse
evidence from one named peer. `reconcile_collection_from` adds an explicit
caller authorization phase and admits the whole accepted batch only after
transport and validation have completed. Neither operation fetches referenced
content.

Collection-native enumeration currently requires unrestricted read authority;
collection-scoped capabilities are future work. Blob reads may still be
restricted to the graph reachable from named local pins. For that reason the
read-only `PinSnapshotSource` remains part of capability evaluation even though
pins are no longer a synchronization protocol or a remote source of truth.

Capability request, issuance, renewal, and delivery state lives in private
signer-owned collections. Those collections declare no reach, so ordinary
evidence gossip cannot expose them — and because reach is part of a descriptor,
no later signature can change that without naming a different collection. Capability-chain blobs may be
fetched during mutual authentication and cached after verification.

## Direction policy

`PeerConfig::direction` controls participation without changing evidence
semantics:

- `Bidirectional` admits incoming evidence, publishes local evidence, and
  services WANTs;
- `ReadOnly` admits incoming evidence and may service WANTs, but publishes no
  local evidence or blob announcements; and
- `WriteOnly` publishes but discards incoming evidence and does not fetch.

These are runtime bandwidth policies, not durable retractions or alternate
collection modes. They apply to replication data only: capability requests,
deliveries, and delivery acknowledgements remain accepted so authorization can
still be established and renewed.

## CLI

The `trible` CLI exposes the same model:

```text
trible pile net identity [--key PATH]
    Print this node's iroh identity.

trible pile net status [--key PATH]
    Show the node id, team root, and local capability handle.

trible pile net sync <PILE> [--peers ...] [--key PATH]
    Run collection-evidence gossip and durable WANT reconciliation.
    --read-only suppresses publication; --write-only suppresses
    admission and fetching; --no-lazy disables WANT servicing.
```

The mesh topic is the team-root public key. Multi-user deployments supply
`TRIBLE_TEAM_ROOT` and `TRIBLE_TEAM_CAP`; a missing root falls back to a
single-user team-of-one. `--duration` and `--quiescent-for` bound a run, but
quiescence is only an observation that no recent event or fulfillment occurred,
not a distributed proof that every peer has converged.

## Invariants worth retaining

- Network discovery is set union, not last-writer-wins state.
- A collection's declared reach authorizes relay, not semantic trust.
- The relay carrying evidence is not its author or a presumed blob holder.
- Observing a commit never creates hidden content demand.
- Merge and derive receipts are evidence; their output blobs remain lazy.
- Every network payload is strictly framed and content/signature checked before
  it can affect local resolution.
- Temporary unreachability remains “unknown,” never “absent.”
