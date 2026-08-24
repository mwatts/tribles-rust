# Distributed Sync

The [`triblespace-net`](https://github.com/triblespace/triblespace-rs/tree/main/triblespace-net)
crate synchronizes TribleSpace's native collection algebra over
[iroh](https://www.iroh.computer/). Its three transport mechanisms have
separate jobs:

- an explicitly selected gossip mesh discovers immutable signed collection
  evidence;
- a DHT discovers providers for content-addressed blobs; and
- CONNECT-authenticated QUIC answers exact blob, collection, and
  operation-receipt questions.

The user-visible surface is `Peer<S>`. It wraps a local store, owns the async
network host, and continues to expose synchronous storage traits. Network
activity does not introduce a remote mutable head, a distributed
compare-and-swap operation, or a second authorization database.

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
    gossip_topic: Some(shared_topic),
    connect_root,
    connect_proof,
    direction: SyncDirection::Bidirectional,
});

peer.refresh();
```

`connect_proof` is a complete root-to-leaf blob-native capability proof whose
leaf invokes `ACTION_CONNECT` for `signing_key` on the exact 32 public-key
bytes of `connect_root`. The caller selects and builds that proof; the
transport sends exactly those bytes and never searches for authority
implicitly. `gossip_topic` is a separate rendezvous choice.

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
selects a collection view, where the relevant application policy and positive
authority observation are available.

Publication permission is part of the collection descriptor. Its optional
`collection_reach` attribute names a reach law, and because a collection
handle is the hash of its descriptor, a collection that travels and one that
stays put are different collections. An author who commits into a collection
whose identity declares public reach has published that occurrence; a commit
into a descriptor with no declared reach remains local unless some other
explicit law says otherwise.

A peer may relay a commit only when it can resolve that commit's exact
descriptor and the descriptor declares a reach law the peer implements. A
missing descriptor is a refusal, as is an unrecognized law. `Collection::commit`
writes the descriptor before the commit, so a publisher's own store contains
the permission it needs to interpret.

The receiving side does not require the descriptor before admitting sparse
evidence. Admission leaks no local data and requiring descriptor residency
would defeat sparse discovery. If that receiver later wants to relay the
commit, it must resolve and apply reach at that later boundary.

The gossip frame contains one strictly verified signed `CollectionCommit`, one
kind byte, and an eight-byte anti-deduplication nonce. The nonce gives periodic
replays a fresh transport identity but is not signed and has no semantic
meaning. Duplicate semantic delivery collapses by intrinsic commit ID, and one
drained batch crosses one storage flush barrier.

## Gossip is sparse and untrusted

Gossiping a `COMMIT` does not transfer any referenced blob. The receiver may
learn exact handles for the descriptor, data, metadata, and attachments while
possessing none of their bytes. Admission also does not manufacture a WANT.

Capability claims and signatures are ordinary content-addressed blobs, but
CONNECT carries the exact required ancestry inline. Authentication never
enumerates a collection or relies on a pre-auth network fetch.

Sparse discovery lets a node learn a large global frontier and then apply its
own cache, trust, authority, and derivation policy. Evidence answers “what has
been asserted?” A WANT answers “what content or computation should this node
obtain?” Treating the first as the second would turn observation into
involuntary full replication.

The application selects the gossip topic explicitly. It need not equal the
CONNECT trust root, and knowing or joining it is not authority. In particular,
a CONNECT capability does not grant gossip membership or make a gossip carrier
trusted. Safety comes from strict verification of each signed commit plus reach
checks at relay time, not from trusting the mesh participant that forwarded it.

`Peer::refresh` performs both sides of local sparse synchronization:

1. drain pending gossip frames, strictly verify, canonicalize, and admit them;
2. refresh the immutable snapshot served by direct RPC;
3. announce newly added blobs to the DHT; and
4. gossip newly visible commits whose descriptors permit relay.

Construction drives one initial refresh, so resident blobs and existing
evidence are published without a separate startup ritual. The host later
reads the current durable snapshot every 30 seconds and replays the relayable
commit frontier with fresh nonces. It does not retain a second unbounded ledger
mirror.

## Durable WANTs

`WantStore` records local operational interest. Its canonical request key has
three variants:

- `Blob(handle)` asks to obtain exact content;
- `Merge(collection, low, high)` asks whether a matching native `MERGE`
  receipt exists; and
- `Derive(target, input)` asks whether a matching native `DERIVE` receipt
  exists. The target descriptor already names the source collection and
  recipe.

The long-running `Reconciler` services these questions without deleting them.
An unavailable answer is normal: the WANT remains durable and retries with
bounded exponential backoff.

Blob fulfillment first tries configured or recently live team peers as routing
candidates, then falls back to DHT provider discovery. A candidate is not
presumed to hold the blob, and every returned payload must still hash to the
requested BLAKE3 handle.

Merge and derive questions have no result hash to feed into the DHT. They are
therefore probed against explicitly configured authenticated peers. Every exact
native receipt returned before the deadline is unioned into the local
`CollectionStore`. A sweep is complete only when every configured peer
answered; partial answers remain useful evidence but temporary unreachability
never becomes proof of absence. Obtaining a receipt's result bytes is a
separate `Blob(result)` WANT.

## Blob-native CONNECT authentication

All direct point-to-point operations use
`PILE_SYNC_ALPN = "/triblespace/pile-sync/7"`. The first stream on every
connection must be `OP_AUTH`. Its request carries a length-prefixed canonical
capability proof inline:

```text
OP_AUTH
proof_length:u32
proof: version:u8, count:u8,
       (claim_length:u16, signature_length:u16,
        claim:bytes, signature:bytes)*
```

The server obtains the caller's Ed25519 public key from the authenticated
transport and verifies this exact claim:

```text
subject  = transport peer key
action   = ACTION_CONNECT
resource = configured CONNECT trust-root public-key bytes
mode     = Invoke
```

The proof must start with a trust-root-signed occurrence, follow each exact
parent in order, preserve action and resource, and end at the claimed peer.
Every canonical claim and signature blob is verified directly from the stream
at the explicit current epoch. Empty, truncated, oversized, reordered,
malformed, expired, not-yet-valid, or claim-mismatched proofs are rejected. The
response is `AUTH_OK` (`0x00`) or `AUTH_REJECTED` (`0x01`); a rejected
connection is closed. A successful connection is also closed immediately after
the proof chain's effective inclusive upper bound, including when it is idle in
the shared connection pool.

There is no pre-auth exception, remote proof-fetch operation, ambient store
lookup, or renewal exchange. After a successful first stream, later streams on
that connection may use the read-only direct RPC surface while its proof
remains valid:

| Operation | Byte | Question | Response |
|---|---:|---|---|
| `GET_BLOB` | `0x02` | exact 32-byte content handle | bytes or missing |
| `CHILDREN` | `0x03` | exact parent blob handle | resident referenced handles |
| `OP_AUTH` | `0x05` | inline exact CONNECT proof | accept or reject; first stream only |
| `COLLECTION_EVIDENCE` | `0x06` | exact collection descriptor handle | relayable signed commits |
| `COLLECTION_OPERATION_RECEIPTS` | `0x07` | exact merge/derive WANT key | matching native receipts |

The direct protocol has no remote write operation.

CONNECT is transport admission, not a permission hierarchy. It grants no
`ACTION_WRITE`, generic `READ`, gossip, collection reach, semantic author
trust, custody, or retention. The host's current serving snapshot determines
which read-only answers exist; content hashes, commit signatures, collection
reach, application author selection, and local retention policy retain their
own independent boundaries.

`fetch_collection_evidence_from` returns strictly verified but inert sparse
evidence from one named authenticated peer. `reconcile_collection_from` adds
an explicit caller authorization phase and admits the complete accepted batch
only after transport and validation finish. Neither operation fetches
referenced content.

## Direction policy

`PeerConfig::direction` controls local participation without changing evidence
or authority semantics:

- `Bidirectional` admits incoming evidence, publishes local evidence and blob
  announcements, and may service WANTs;
- `ReadOnly` admits incoming evidence and may service WANTs, but publishes no
  local evidence or blob announcements; and
- `WriteOnly` publishes but discards incoming evidence and does not fetch.

These are runtime bandwidth policies, not grants, durable retractions, or
alternate collection modes. Every direct connection still presents the exact
configured CONNECT proof. Selecting `gossip_topic: None` similarly disables
topic participation without changing direct-RPC authority.

## CLI

The `trible` CLI makes team and proof selection explicit:

```text
trible pile net identity [--key PATH]
    Initialize the key if needed and print this node's iroh identity.

trible pile net status <PILE> --team-root HEX --grant ID [--key PATH]
    Load the existing key, resolve the exact accepted local CONNECT grant,
    reconstruct its portable ancestry, and report the proof step count.

trible pile net sync <PILE> --team-root HEX --grant ID
    [--peers ID_OR_TICKET,...] [--key PATH]
    Run collection-evidence gossip and durable WANT reconciliation.
    --read-only suppresses publication; --write-only suppresses admission
    and fetching; --no-lazy disables WANT servicing.
```

`status` and `sync` require all of the pile, team root, grant occurrence, and
existing signing key. A missing or differently owned grant, inert ancestry,
wrong action/resource, non-invoking leaf, absent key, or non-portable proof is
an error before networking starts. There is no environment-variable fallback,
all-zero grant sentinel, implicit team-of-one grant, or automatic key creation
on these two paths.

The mesh topic is an explicit application choice independent of the CONNECT
root. `--duration` and `--quiescent-for` can bound a run, but quiescence means
only that no recent network event or WANT fulfillment was observed. It is not
a distributed proof that every peer has converged.

## Invariants worth retaining

- Network discovery is set union, not last-writer-wins state.
- A collection's declared reach authorizes relay, not semantic trust.
- The gossip carrier is neither the commit author nor a presumed blob holder.
- Observing a commit never creates hidden content demand.
- Merge and derive receipts are evidence; their output blobs remain lazy.
- Direct RPC starts with one inline, exact, claim-directed CONNECT proof.
- CONNECT authenticates the session and grants no other action or storage
  policy.
- Every payload is strictly framed and content/signature checked before it can
  affect local resolution.
- Temporary unreachability remains “unknown,” never “absent.”
