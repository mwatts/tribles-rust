# Distributed Sync

`triblespace-net` synchronizes collections rather than exposing one ambient
store inventory. Its protocol follows the same decomposition as the collection
model:

1. stock `iroh-gossip` announces that an endpoint has a new opaque state for
   one collection;
2. one READ(C)-authorized PATCH walk repairs the exact evidence which can
   change that collection's value; and
3. bearer blob handles fetch only the immutable bytes a resolver actually
   chooses, with an opaque DHT provider directory for openly disclosed data.

No global team, mutable roster, durable OFFER/GOSSIP bit, or second replicated
inventory is needed. The collection descriptor already states independent
READ and WRITE policy, and iroh authenticates the endpoint key on each direct
connection.

## Four independent capabilities

The boundaries are deliberately small:

```text
know C           -> join C's wake topic and learn (origin, opaque state root)
prove READ(C)    -> receive and repair C's activation evidence
know H           -> request immutable blob H from a provider which serves it
satisfy WRITE(C) -> make a signed COMMIT active in C
```

`C` is the exact 32-byte collection descriptor handle. `H` is an exact blob
handle. Knowing either value is already unforgeable naming power, but they do
different jobs: `C` discovers state change without revealing content, while
`H` is the bearer capability for one immutable value.

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

The collection handle is the exact `iroh-gossip` topic ID. Anyone who knows C
may join that topic; there is no authorization handshake merely to hear that
something changed. The application payload is fixed width:

```text
version:u8 || endpoint_origin:32 || activation_root:32 || signature:64
```

The collection handle is not repeated in the envelope, but it is included in
the signature transcript. Replaying identical bytes on another collection
topic therefore fails verification. The origin is the same Ed25519 identity as
the iroh endpoint and tells receivers which peer can answer repair.

A wake contains no record, proof, blob handle, leaf count, component root, or
human-readable collection metadata. It is a latency hint, not durable evidence
and not authorization. Stock gossip supplies bounded duplicate suppression and
an efficient dissemination tree; periodic anti-entropy remains the source of
eventual convergence when wakes are delayed or missed.

## READ(C)-authorized exact repair

After observing a changed wake root—or on a periodic configured-peer sweep—a
node opens one bidirectional collection-repair stream to the origin. The hello
names C and carries a bounded portable READ proof forest for the caller's
TLS-authenticated endpoint key. For `Open` READ the forest is empty.

The server loads one immutable activation overlay for C and applies the
descriptor's exact READ policy at one instant. Rejection returns no manifest.
On admission it returns the two PATCH summaries and the same opaque activation
root. The client then walks only differing prefixes and receives missing leaf
bodies:

- canonical `COMMIT`, `MERGE`, and `DERIVE` records; and
- complete relevant WRITE proof bundles.

Every request pins the manifest's expected component root. The server serves
the whole stream from one immutable overlay lease, so responses cannot splice
two moments together and need no historical-root cache. The client validates
node summaries, intrinsic leaf keys, record bodies, proof signatures, and claim
handles before insertion.

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

`GET_BLOB(H)` is bearer-addressed: knowledge of the full content hash is the
read capability for those exact bytes. The receiver verifies the returned
bytes against H before storing them. Fetching does not assert collection
membership, activate a commit, or create a durable WANT unless the caller
chooses to record one.

Provider discovery never sends H to directory nodes. Both publication and
lookup derive one full-width rendezvous key:

```text
provider_key = BLAKE3-KDF("triblespace.net/provider-key/v2", H)
```

Providers renew soft leases for that opaque key at nearby XOR-DHT nodes. A
lookup returns endpoint candidates, never bytes or authority. The directory is
bounded soft state: routing and provider leases may disappear without changing
the collection or local retention.

Autonomous provider publication is derived from collection disclosure rather
than a durable OFFER record. Only payload and closure handles admitted through
an `Open` READ policy are published globally, and only while the local serving
QoS permits it. Restricted collections can still exchange already-known H
with authorized peers, but do not advertise their handles through the global
directory. This separation avoids making the replication topology mirror
private access-control structure.

## Routing is process state

Initial peers come from `PeerConfig` or the CLI. A verified wake origin and DHT
referrals may become live routing candidates, but there is no synchronized
PEER roster and no durable peer record in the current protocol. Liveness,
backoff, connection pooling, DHT buckets, and provider leases are operational
soft state; restarting may forget them without losing semantic data.

One connection pool is shared by collection repair and bearer/DHT operations.
Iroh's transport authentication binds each connection to its endpoint ID.
There is no generic AUTH or SYNC_TEAM exchange: useful collection bytes are
gated by READ(C), while exact blob bytes are gated by H.

## Lattice-aware sparse replication

The network does not force every replica to mirror every blob. Collection
records expose the same lattice known to local maintenance:

```text
COMMIT(C, a)       COMMIT(C, b)
       \             /
        MERGE(C, a, b, c)

DERIVE(D, c, d)
```

A node can repair the small activation overlay, inspect which exact merge and
derivation results peers already know, and fetch the best available cover
instead of downloading both inputs and recomputing an identical result. If no
useful artifact is resident, the ordinary local `ensure` path computes it and
publishes the same canonical equation. Evidence and computation converge by
union; no central scheduler or query planner is required.

Durable WANT remains orthogonal local policy. `WantRequest::Blob(H)`,
`Merge(C,a,b)`, and `Derive(D,input)` let one process state demand while a
network or worker process fulfills it. WANT grants no READ, WRITE, retention,
or membership semantics.

## Wire surface

Protocol version 17 keeps the direct operation set narrow:

| Operation | Code | Meaning |
|---|---:|---|
| `GET_BLOB` | `0x02` | fetch one bearer handle H |
| `PROVIDER_PUT` | `0x06` | renew this endpoint's opaque provider lease |
| `PROVIDER_GET` | `0x07` | obtain bounded candidates for one opaque key |
| `FIND_NODE` | `0x0C` | iterative XOR-DHT routing step |
| `COLLECTION_REPAIR` | `0x0D` | READ(C)-authorized activation PATCH repair |

There is deliberately no store manifest, global inventory authorization,
push-broadcast record, receipt RPC, remote mutable head, or unpublish operation.

The CLI selects explicit collections and bootstrap peers:

```text
trible pile net sync DATA.pile \
    --collection COLLECTION_HANDLE [--collection COLLECTION_HANDLE ...] \
    [--peers ENDPOINT_TICKET ...] [--direction pull|push|bidirectional]
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
