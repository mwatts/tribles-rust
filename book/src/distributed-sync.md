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

`connect_proof` is a complete `CapabilityProofBundle` whose leaf invokes
`ACTION_CONNECT` for `signing_key` on the exact 32 public-key bytes of
`connect_root`. The caller selects that proof explicitly; the transport sends
the native `K0 (S C K)+` path and its ordered keyless claim blobs and never
searches for authority implicitly. `gossip_topic` is a separate rendezvous
choice.

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

Capability claims are ordinary content-addressed blobs and complete proofs are
native set records, but CONNECT carries the exact required proof and claims
inline. Authentication never performs a pre-auth proof lookup or blob fetch.

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

## Durable peer evidence

Core storage exposes one monotone topology primitive:
`PEER(team_public_key, peer_public_key)`. It is an unsigned, canonical
two-key fact stored through `PeerStore`; concatenating piles unions the set.
There is deliberately no inverse record. Its presence says only “this peer is
a routing candidate associated with this team.” It grants no capability,
proves no liveness or reachability, promises no resident data, and retains no
blob.

The current network host does not consume this set yet: configured endpoint
tickets and process-local successful custody neighbors remain its operational
inputs. Keeping the storage fact narrower than that behavior makes it a clean
foundation for a later unified transport with quality-of-service policies,
rather than silently introducing a second peer protocol in this change.

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

## Direct CONNECT authentication

All direct point-to-point operations use
`PILE_SYNC_ALPN = "/triblespace/pile-sync/9"`. The first stream on every
connection must be `OP_AUTH`. Its request carries one length-prefixed canonical
proof bundle inline:

```text
OP_AUTH
bundle_length:u32be
bundle:
    version:u8 = 1
    count:u8
    proof: K0 (S C K){count}
    repeat count times:
        claim_length:u16be
        claim:bytes
```

The server obtains the caller's Ed25519 public key from the authenticated
transport and verifies this exact claim:

```text
subject  = transport peer key
action   = ACTION_CONNECT
resource = configured CONNECT trust-root public-key bytes
mode     = Invoke
```

The proof root must equal the configured trust root and its final key must equal
the transport peer. Every strict signature binds issuer key, exact claim
handle, and delegate key. Ordered keyless claims must form one parent-claim
path whose exact atom, mode intersection, and inclusive validity intersection
satisfy the request at the explicit current epoch. Empty, truncated,
oversized, reordered, malformed, expired, not-yet-valid, or claim-mismatched
bundles are rejected. The response is `AUTH_OK` (`0x00`) or `AUTH_REJECTED`
(`0x01`); a rejected connection is closed. A successful connection is also
closed after the proof's effective inclusive upper bound, including while idle
in the shared connection pool.

There is no pre-auth exception, remote proof-fetch operation, ambient store
lookup, or renewal exchange. After a successful first stream, later streams on
that connection may use the read-only direct RPC surface while its proof
remains valid:

| Operation | Byte | Question | Response |
|---|---:|---|---|
| `GET_BLOB` | `0x02` | exact 32-byte content handle | bytes or missing |
| `CHILDREN` | `0x03` | exact parent blob handle | resident referenced handles |
| `OP_AUTH` | `0x05` | inline exact CONNECT proof bundle | accept or reject; first stream only |
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

## Custody replicas

Sparse public synchronization is deliberately not a backup protocol. A
separate custody lane converges the complete **resident semantic store** across
a proof-authorized neighbor graph:

```text
BlobStore × CollectionStore × CapabilityProofStore
```

The join is componentwise set union. Custody therefore copies every valid
resident blob, every canonical native collection record, and every canonical
native capability proof. It does not copy WANTs, historical pins, append
timestamps, framing padding, routing state, or retry state. Unknown opaque pile
records stop startup rather than being silently discarded; recognized retired
V4 `DERIVE` records are known inert computation and may be omitted.

This product is semantic, not a comparison of pile byte streams. Bucket
prefixes are the first byte of canonical blob, record, or proof identities;
they are not file prefixes, offsets, or append epochs. Snapshot construction
deduplicates and orders those identities canonically, so piles containing the
same three sets have the same inventory summary and generation even when their
records were appended, concatenated, or padded in different orders.

Custody uses ordinary Iroh reachability: explicit addresses when supplied,
direct and NAT-traversed paths when available, and encrypted relays when they
are not. An operator may bootstrap a node with either a bare endpoint identity
or an `EndpointTicket` carrying one or more route hints. Custody still exposes
only the pile-sync protocol: it does not join collection gossip or start the
blob-provider DHT. Reachability is not authority. The ordinary CONNECT proof
authenticates the transport, but it grants no custody access. Every summary,
page, and blob-range request additionally carries a complete proof for the
exact action
`ACTION_REPLICATE_STORE = D8453B974E15F5DF17B1B67A338B3EBD`, the exact
256-bit replica-set resource, the transport peer as leaf, and exact Invoke
mode.

Configured peers are bootstrap neighbors, not a global roster. After an
unknown peer completes CONNECT and one strictly framed summary request with a
valid REPLICATE proof, the server remembers that identity as a process-local
neighbor for the intersection of both proofs' validity intervals. It can then
pull the joiner's resident state on its next sweep. Consequently a joining
node needs to know only one reachable member of a connected replica graph;
ordinary componentwise union carries state transitively through the graph.
There is no durable peer list, membership enumeration, or peer-list gossip.

The bounded anti-entropy protocol adds three operations:

| Operation | Byte | Question | Response |
|---|---:|---|---|
| `REPLICA_SUMMARY` | `0x08` | all 256 first-byte buckets of all three components | count, byte total, and BLAKE3 digest per bucket; these derive the inventory-generation token |
| `REPLICA_PAGE` | `0x09` | summary generation, component, bucket, and exclusive cursor | one sorted bounded page plus a final marker |
| `REPLICA_BLOB` | `0x0A` | summary generation, exact handle, expected length, offset, and bounded maximum | at most one 1 MiB range or missing |

Collection-record IDs are extended with a zero suffix only for page ordering;
their canonical dense record bytes remain the transferred representation.
Proofs likewise travel in their canonical native body. Blob ranges land in an
anonymous file in a caller-selected pile-adjacent directory, are hashed once as
they arrive, and become a read-only mapping whose verified handle is retained
through destination admission.

Each sweep obtains immutable remote summaries, validates every completed page
stream against its advertised bucket digest, and admits all blobs from every
healthy peer before admitting any of their collection or proof evidence. Blob
writes cross durability barriers in bounded 64 MiB batches; record and proof
pages cross a barrier per page. A stale pooled connection is evicted and
redialed once within the same sweep; a still-failed peer remains incomplete and
retries on the next sweep while independent peers continue. Receive-file,
allocation, write, metadata, and mapping failures are local storage failures
and abort the sweep rather than masquerading as peer unavailability.
Each accepted summary derives a content identity from the complete summary and
places that exact immutable serving generation in a bounded process-local
cache. Every subsequent page and blob request must echo the generation.
Publishing a newer local snapshot therefore cannot splice its items into an
older advertised walk, while reconnecting peers and peers that observed the
same generation share one cached snapshot. A missing or evicted generation is
rejected and the client restarts from a fresh summary. The cache uses the
service's global connection-capacity bound for its number of indexed
generations. Eviction removes lookup eligibility, not necessarily the final
allocation immediately: up to the global in-flight-request limit may already
hold a clone or response bytes backed by an evicted snapshot, and those live
only until their existing request deadlines. The generation token is a
retry/cache key, never an authorization capability: every request still
verifies current REPLICATE authority independently.
Cancellation can abandon only network/page work: store mutations are
synchronous, temporary receive files close on drop, the transport shuts down
gracefully, and the pile is explicitly closed before the command exits.

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

trible pile net status <PILE> --team-root HEX --proof ID [--key PATH]
    Load the existing key, exact native proof, and its named claim blobs;
    verify CONNECT at the current time; and report the proof step count.

trible pile net sync <PILE> --team-root HEX --proof ID
    --gossip-topic HEX
    [--peers ID_OR_TICKET,...] [--key PATH]
    Run collection-evidence gossip and durable WANT reconciliation.
    --read-only suppresses publication; --write-only suppresses admission
    and fetching; --no-lazy disables WANT servicing.

trible team replica create|issue|join ...
    Provision and import invoke-only proofs under an independent offline
    replica root and exact replica-set resource.

trible pile net custody status <PILE> ...
    Validate bootstrap peer ids or tickets, both exact proofs, the receive
    directory, opaque-record fence, and complete local inventory without
    opening a socket.

trible pile net custody run <PILE> ...
    Run foreground custody anti-entropy over ordinary Iroh routing until
    SIGINT or SIGTERM, and print the live endpoint ticket.
```

Set `RUST_LOG=triblespace_net=info` when operating a custody node to record
both the initially selected Iroh path and any later direct/relay path
migration without changing route selection.

`status` and `sync` require the pile, CONNECT trust root, exact proof ID, and
existing signing key; `sync` additionally requires its independent gossip
topic. A missing proof or claim, different leaf, invalid ancestry, wrong
action/resource, non-invoking or currently invalid proof, or absent key is an
error before networking starts.

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
- Direct RPC starts with one inline, exact CONNECT proof bundle.
- CONNECT authenticates the session and grants no other action or storage
  policy.
- Full custody is a separately authorized product union over an authenticated,
  process-local neighbor graph; it never follows gossip or creates WANTs.
- Every payload is strictly framed and content/signature checked before it can
  affect local resolution.
- Temporary unreachability remains “unknown,” never “absent.”
