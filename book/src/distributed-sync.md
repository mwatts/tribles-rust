# Distributed Sync

`triblespace-net` synchronizes collections rather than exposing one ambient
store inventory. Its protocol follows the same decomposition as the collection
model:

1. stock `iroh-gossip` announces that an endpoint has a new opaque state for
   one collection;
2. one READ(C)-authorized PATCH walk repairs the exact evidence which can
   change that collection's value; and
3. exact blob handles fetch only the immutable bytes a resolver actually
   chooses through collection-scoped provider discovery and an authenticated
   collection session.

No global team, mutable roster, durable OFFER/GOSSIP bit, or second replicated
inventory is needed. The collection descriptor already states independent
READ and WRITE policy, and iroh authenticates the endpoint key on each direct
connection.

## Four independent capabilities

The boundaries are deliberately small:

```text
know C           -> join C's wake topic and learn (origin, opaque state root)
prove READ(C)    -> receive and repair C's activation evidence
know H           -> authorize the exact immutable bytes H
know C           -> discover a provider which proves READ(C) before H is revealed
satisfy WRITE(C) -> make a signed COMMIT active in C
```

`C` is the exact 32-byte collection descriptor handle. `H` is an exact blob
handle. Knowing either value is already unforgeable naming power, but they do
different jobs: `C` discovers a collection participant, while `H` is the bearer
capability for one exact immutable value. Before the requester reveals `H`, the
selected participant proves current READ(C); the provider directory never sees
either raw handle.

READ and WRITE are independent `AdmissionPolicy` values embedded in the
descriptor. Each is either `Open` or a canonical quorum over Ed25519 roots with
separate invoke and delegation thresholds. A derived collection chooses its
own policies. Source ancestry, routing knowledge, and possession of a blob do
not silently supply collection authority.

Local stores remain permissive grow-only ledgers. They may contain a COMMIT
whose signer does not currently satisfy WRITE(C), or a proof irrelevant to any
resident collection. Admission is applied when a snapshot is observed. Later
proof evidence may activate an old commit without rewriting or retracting it.

## The activation product

For one collection, exactly two grow-only sets can change admission:

- every structurally valid `COMMIT`, `MERGE`, and `DERIVE` record whose
  intrinsic collection is C; and
- every complete portable WRITE proof bundle relevant to C's WRITE-policy
  roots.

Each set is represented by an immutable BLAKE3-Merkle PATCH. Collection records
are keyed by their 16-byte intrinsic record ID; proof bundles are keyed by
their 32-byte proof ID and carry their exact ordered claim closure as the leaf
value. The opaque activation root commits to C, both PATCH roots, and both leaf
counts under a versioned domain.

This product matters. Synchronizing only collection records would miss the
case where a newly arrived proof activates an old COMMIT. Synchronizing a whole
proof store would disclose unrelated capability structure. The overlay is the
smallest exact state whose union can change C.

Unsigned MERGE and DERIVE records remain optional computation evidence. Their
presence never activates a payload on its own, and a reader validates an
equation before using it as a support-equivalent physical route.

## Opaque wakes over stock gossip

The `iroh-gossip` topic ID is a domain-separated one-way image of the collection
handle. Anyone who knows C can derive and join that topic, while generic gossip
routers do not learn raw C. There is no authorization handshake merely to hear
that something changed. The application payload is fixed width (177 bytes):

```text
version:u8 || endpoint_origin:32 || semantic_root:32 || payload_root:32 || nonce:16 || signature:64
```

The collection handle is not repeated in the envelope, but it is included in
the signature transcript. Replaying identical bytes on another collection
topic therefore fails verification. The origin is the same Ed25519 identity as
the iroh endpoint and tells receivers which peer can answer repair.

A wake contains no record, proof, blob handle, leaf count, component root, or
human-readable collection metadata. It is a latency hint, not durable evidence
and not authorization. The fresh nonce makes repeated change and neighbor
announcements distinct.
Stock gossip supplies bounded duplicate suppression and an efficient
dissemination tree; bounded leased wake-origin sampling supplies eventual
anti-entropy when a wake is delayed or missed.

## READ(C)-authorized exact repair

After observing a changed wake root—or when periodically sampling live signed
wake origins—a node opens one bidirectional collection-repair stream to that
origin. Its hello names C and carries the client's bounded READ proof forest.
The server admits the TLS-authenticated client before returning any manifest.
For `Open` READ the witness is empty. A WRITE-only publisher therefore needs
no READ authority merely to serve an authorized replica.

The server loads one immutable activation overlay for C and applies the
descriptor's exact READ policy at one instant. Rejection returns no manifest.
On admission it returns record, WRITE-evidence, and disclosure-forest PATCH
summaries plus the same opaque roots. The client may then walk only differing prefixes and receive missing leaf
bodies:

- canonical, currently WRITE-accountable `COMMIT` records;
- complete relevant WRITE proof bundles.

The disclosure forest uses unit-valued 80-byte keys
`depth || parent_H || aligned_index || child_H`. Roots are authenticated
descriptor/data/metadata handles from admitted COMMITs. A child becomes trusted
only after the receiver verifies its exact aligned occurrence in a trusted,
hash-verified parent. The remote PATCH is an availability oracle, never
authority for arbitrary handles. Demand peers ignore the payload wake root;
Full peers incrementally follow both roots under the same bounded session.

Every request pins the manifest's expected component root. The server serves
the whole stream from one immutable overlay lease, so responses cannot splice
two moments together and need no historical-root cache. The client validates
node summaries, intrinsic leaf keys, record bodies, proof signatures, and claim
handles before insertion.

The exact-blob stream uses C only for private rendezvous. The provider proves
READ(C) before the requester reveals H; possession of H then authorizes those
exact resident bytes. Bare WANT(H) is local retention intent and has no network
discovery promise. Full custody separately validates reachability through the
authenticated disclosure forest before retaining payloads.

Repair is one-way pull. Two peers converge by each eventually pulling after a
wake or periodic sweep. This keeps authorization and failure local to one
stream while set union makes direction irrelevant to the final value. A node
which holds no overlay for C answers unavailable rather than exposing a global
inventory.

## Blob transfer is lazy and bearer-addressed

Activation repair does not fetch descriptor dependencies, payloads, metadata,
attachments, or derived artifacts. It transfers the lattice evidence needed to
decide what exists. A resolver can then select the cheapest resident
support-equivalent cover and request only the missing immutable handles that
matter to that computation.

Knowledge of a full content hash H is the read capability for those exact
bytes. Discovery is nevertheless collection-scoped: the requester discovers
participants through opaque KDF(C), verifies that a candidate proves READ(C)
before revealing H, and then requests those exact resident bytes by H. The
provider does not perform a second collection-membership check on H: the
requester need not prove READ(C) merely to exercise a bearer handle it already
holds. Fetching does not assert
collection membership, activate a commit, or create a durable WANT unless the
caller chooses to record one.

Provider discovery never sends C to directory nodes. Both publication and
lookup derive one full-width rendezvous key:

```text
provider_key = BLAKE3-KDF("triblespace.net/collection-provider-key/v1", C)
```

Providers renew one soft lease per active collection for that opaque key at
nearby XOR-DHT nodes. Each lease carries an endpoint-binding token derived
independently from C and the provider id; clients verify it before dialing. A
lookup returns endpoint candidates, never bytes or authority. The directory is
bounded soft state: routing and provider leases may disappear without changing
the collection or local retention.

Autonomous provider publication follows active serving policy rather than a
durable OFFER record. It is O(collections), not O(resident blobs). There is no
global per-H directory and possession of a detached H does not promise global
discovery; a caller needs a C route or an already-known provider.

## Routing is process state

Initial peers come from `PeerConfig` or the CLI. A verified wake origin and DHT
referrals may become live routing candidates, but there is no synchronized
PEER roster and no durable peer record in the current protocol. Liveness,
backoff, connection pooling, DHT buckets, and provider leases are operational
soft state; restarting may forget them without losing semantic data.

One connection pool is shared by collection repair and bearer/DHT operations.
Iroh's transport authentication binds each connection to its endpoint ID.
There is no generic AUTH or SYNC_TEAM exchange: collection evidence is gated by
READ(C). Exact bytes are gated by H, with the collection route additionally
requiring the provider to prove READ(C) before the requester reveals H.

## Lattice-aware sparse replication

The network does not force every replica to mirror every blob. Collection
records expose the same lattice known to local maintenance:

```text
COMMIT(C, a)       COMMIT(C, b)
       \             /
        MERGE(C, a, b, c)

DERIVE(D, c, d)
```

A node can repair the small activation overlay and use its resident exact merge
and derivation results while planning a cover. Missing derived results are
computed by the ordinary local `ensure` path: unsigned MERGE/DERIVE outputs are
not remote publication authority and are not reused over the network in this
release. Evidence and computation still converge by union; no central
scheduler or query planner is required.

Durable WANT remains orthogonal operational policy. Bare
`WantRequest::Blob(H)` is local-only, while `BlobInCollection(C,H)` names the
exact route through which the reconciler discovers `H`. For every pending
route, it loads and validates C's descriptor policy from the same coherent
store snapshot used to observe the WANT, then performs provider discovery under
KDF(C). C need not be active or configured on the requester, and this lookup
does not activate it. If C's descriptor is absent or malformed, the WANT stays
pending; the reconciler neither guesses another collection nor falls back to a
configured peer.
Exact routes remain distinct intents even though local presence of `H`
satisfies all of them. `Merge(C,a,b)` and `Derive(D,input)` let one process
state demand while a network or worker process fulfills it. WANT grants no
READ, WRITE, retention, or membership semantics.

A bare blob WANT does not carry C and therefore triggers no network discovery.
For a routed WANT, the provider proves READ(C) under the validated resident
policy before the requester reveals bearer H. Multiple routes for one H share
one fetch budget, and a successful durable landing satisfies every route plus
any separate bare intent for H. The DHT never publishes or queries KDF(H).

## Wire surface

Protocol version 20 keeps the direct operation set narrow:

| Operation | Code | Meaning |
|---|---:|---|
| `GET_BLOB` | `0x02` | low-level exact bearer transport after scoped discovery |
| `PROVIDER_PUT` | `0x06` | renew this endpoint's opaque provider lease |
| `PROVIDER_GET` | `0x07` | obtain bounded candidates for one opaque key |
| `FIND_NODE` | `0x0C` | iterative XOR-DHT routing step |
| `COLLECTION_REPAIR` | `0x0D` | receiver-authorized semantic and Full PATCH repair |
| `COLLECTION_BLOB` | `0x0E` | provider-proved READ(C), then exact bearer H fetch |

There is deliberately no store manifest, global inventory authorization,
push-broadcast record, receipt RPC, remote mutable head, or unpublish operation.

The CLI selects explicit collections and bootstrap peers:

```text
trible pile net sync DATA.pile \
    --collection COLLECTION_HANDLE [--collection COLLECTION_HANDLE ...] \
    [--peers ENDPOINT_TICKET ...] [--direction bidirectional|read-only|write-only] \
    [--payload demand|full]
```

Direction and serving choices are local QoS. They do not participate in
collection identity and cannot change which evidence is semantically valid.

## Convergence and failure model

- Concatenation, local insertion, and remote repair all perform set union.
- Duplicate records and proof bundles collapse by intrinsic identity.
- A missed wake only adds latency; periodic repair still converges connected
  readers.
- An invalid wake, record, proof, PATCH node, or blob fails that input and
  cannot retract previously accepted evidence.
- Missing blobs leave a semantic cover known but not yet materializable.
- A DHT miss says only that no live provider was found; it says nothing about
  whether H or its collection exists.
- Concurrent writers and offline replicas reconverge without preserving pile
  byte order.

The result is one elemental loop: gossip says *where to ask*, READ(C)-gated
PATCH repair says *what changed*, and bearer content addressing retrieves only
the immutable bytes the local lattice resolver decides to use.
