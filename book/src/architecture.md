# Architecture Overview

TribleSpace is an embedded knowledge graph whose storage and distribution model
is a small join algebra. Facts, blobs, and publication records are immutable;
independent stores combine by union. The architecture follows from that choice
rather than wrapping an ordinary mutable database in a replication protocol.

## The load-bearing principles

### Content addressing

Every blob is named by a hash of its bytes. Identical values deduplicate,
readers can validate integrity without trusting storage, and a reference has
the same meaning in memory, in a pile, or on an object store. Handles fit in a
trible's 32-byte value slot, so descriptions, datasets, metadata, and large
attachments all use the same reference primitive.

### Monotone evidence

A `TribleSet` is a set of immutable facts. A `CollectionStore` is a set of
immutable algebra records. Their merge is set union, which is associative,
commutative, and idempotent. Concatenating independently written piles therefore
cannot create a last-writer-wins conflict or change the meaning of an existing
record.

This is the practical consequence of the [CALM
principle](https://arxiv.org/abs/1901.01930): monotone conclusions do not need a
coordination protocol. Application-level change is represented explicitly as
new facts, version links, or successor DAGs rather than by overwriting an
ambient current value.

### Authority and computation are separate

A signed `COMMIT` says that an author places one element in a collection. An
unsigned `MERGE` or `DERIVE` says that reproducible computation connected known
elements. The former is irreducible authority; the latter is replaceable cache
evidence which a reader validates under the collection's recipe.

That separation is why a materialized index does not become ground truth merely
because it is convenient, and why collecting an accelerator does not erase the
committed facts from which it can be rebuilt.

Who may make that signed assertion can be proven without making storage a
policy oracle. A capability-guarded descriptor names an external trust root,
while the facade owns explicit presentations pairing an expected leaf key with
a complete `K0 (S C K)+` proof bundle. The native proof binds each issuer,
exact keyless claim handle, and delegate key; the ordered claim blobs carry the
action/resource, mode, validity, and parent-claim restrictions. Ordinary
collection operations verify the resulting meet directly at one clock
instant. Holding a signing key or finding a proof in storage grants nothing by
itself.

## Architectural layers

```text
┌──────────────────────────────────────────────────┐
│ Application                                      │
│ entity! · Fragment · find! · pattern!            │
├──────────────────────────────────────────────────┤
│ Collection facades                               │
│ publish, select exact tickets, materialize views │
├──────────────────────────────────────────────────┤
│ Collection algebra                              │
│ signed COMMIT · checked MERGE · checked DERIVE   │
├──────────────────────────────────────────────────┤
│ Storage                                          │
│ CollectionStore · BlobStore · WantStore          │
│ CapabilityProofStore · PeerStore                  │
├──────────────────────────────────────────────────┤
│ Data and representations                         │
│ TribleSet/PATCH · SimpleArchive · SuccinctArchive│
└──────────────────────────────────────────────────┘
```

The boundaries are deliberately narrow. Query constraints do not know how
bytes were published. A collection recipe does not decide replication policy.
A blob store does not infer authority from a handle it happens to contain.

## Tribles, sets, and fragments

A [`Trible`](https://docs.rs/triblespace/latest/triblespace/trible/struct.Trible.html)
is a fixed 64-byte entity–attribute–value fact. The attribute determines how
the value's 32 bytes are interpreted. A [`TribleSet`](https://docs.rs/triblespace/latest/triblespace/trible/struct.TribleSet.html)
stores each fact once while maintaining the six entity/attribute/value
permutations needed by the query engine. The underlying persistent adaptive
tries make cloned sets cheap and union, intersection, and difference structural
operations.

A `Fragment` is the publication unit applications normally construct. It
carries:

- ordinary facts;
- descriptive metafacts;
- exported intrinsic entity IDs; and
- the content-addressed attachments referenced by either fact set.

Fragments compose with `+=`. The `entity!` macro derives an ID when no explicit
subject is supplied and inserts encoded blob payloads into the fragment's
attachment store. This keeps provenance and required bytes together without
mixing schema descriptions into ordinary application queries.

## Blob storage

`BlobStorePut`, `BlobStoreGet`, `BlobStoreMeta`, and `BlobStoreList` describe
small independent capabilities rather than one all-or-nothing database trait.
The main backends are:

- `MemoryRepo` for process-local work and tests;
- `Pile` for one append-only, memory-mapped file; and
- `ObjectStoreRemote` for S3-compatible storage.

Content addressing makes storage placement a physical concern. A missing local
blob is not a semantic retraction: another node may still provide the exact
bytes later.

## Collections are self-describing lattices

A collection descriptor is an ordinary `TribleSet`, encoded as a canonical
`SimpleArchive`. Its content handle is the `CollectionHandle`. A root descriptor
normally states:

- a human-readable name within a public-key namespace;
- an optional capability trust root;
- the element representation;
- the join recipe; and
- a reach law governing permissionless relay.

A derived descriptor names its source collection, the homomorphism recipe, and
its own optional capability trust root. Trust never inherits from the source.
Descriptions of the representation and recipe travel in the same archive, so a
record naming the descriptor remains interpretable without a separate registry
entry.

`CollectionStore` contains three native record kinds:

| Record | Meaning | Dense payload |
|---|---|---:|
| `COMMIT(C, x, metadata, author, signature)` | The author asserts `x` as an independent member of `C`. | 192 bytes |
| `MERGE(C, a, b, c)` | Under `C`'s join law, `a ⊔ b = c`. | 128 bytes |
| `DERIVE(T, a, b)` | The homomorphism named by target `T` maps source element `a` to target element `b`. | 96 bytes |

All three have intrinsic IDs derived from their exact canonical payload. A
repeat insert is a no-op. `COMMIT` is signed because its assertion cannot be
recomputed; `MERGE` and `DERIVE` are unsigned because correctness comes from
the recipe and exact bytes, not the identity of the machine that performed the
work.

The algebra has no distinguished head. Several commits coexist, and the value
of a selected collection view is the join of its admitted members. This makes a
commit the atomic publication boundary without inventing a mutable register
above it.

## Publishing and observing

`Collection<S>` owns a storage backend, one canonical descriptor, one signing
key, and one explicit admission policy. Open admission accepts every strictly
verified signer and omits the descriptor's capability trust-root fact.
Capability admission writes that exact trust root and retains owned
`CapabilityPresentation`s in the facade. `Collection::commit(fragment)`
observes the clock once, verifies every presentation's bundle for its expected
leaf and exact `ACTION_WRITE` atom, and requires its signing key among those
leaves before storing the descriptor, attachments, canonical data archive,
canonical metadata archive, and signed native commit record. It performs no
ambient proof lookup and does not flush implicitly; callers choose durability
cadence with `Collection::flush` or the backend's explicit close operation.

Reads are exact about what they observed, not magical about global time:

- `ticket()` verifies explicit presentations and returns every exact verified
  commit by the resulting subjects; this reads native records but not the
  selected commits' data or metadata blobs;
- `snapshot()` carries materialized facts, that exact admission-selected
  commit set, and the target blob reader which validated them; and
- exact-ticket facades let another consumer materialize a caller-selected
  multi-author commit frontier without holding a publishing key.

Each call observes one known prefix of an append-only store. A concurrent
commit may appear now or on the next call, but a snapshot never
combines facts from one admission frontier with commits from another. The
facade's local signing key does not narrow reads: all explicitly presented
subjects participate, or every strict signer in open mode.

## Derived physical representations

The same collection can be projected into representations optimized for a
particular task. A canonical SuccinctArchive collection, a Rank9 sidecar
collection, or a regular-path summary is a derived lattice whose recipe is a
join homomorphism:

```text
f(a ⊔ b) = f(a) ⊔ f(b)
```

That law permits either route through the evidence graph: merge source shards
and derive once, derive individual shards and merge their images, or reuse any
validated mixture already present. Exact tickets keep the logical authority
fixed while a resolver chooses a resident physical cover. Missing derived
artifacts are cache misses, not missing facts.

## WANT is operational, not semantic

A `WantStore` records local interest in obtaining a blob or discovering a
particular merge/derive result. WANTs do not add collection members, authorize
authors, retain all referenced data, or force another node to perform work.
They are durable coordination-free questions which a reconciler may satisfy by
fetching content or unioning a matching native equation into the local store.

Keeping WANT orthogonal prevents metadata convergence from becoming
involuntary blob mirroring. A peer can learn its authorized team's collection
record frontier, then decide which blobs and derived representations are
useful locally.

## Peer evidence is topology, not authority

`PeerStore` holds positive `PEER(team_public_key, peer_public_key)` routing
facts as another grow-only set. A fact is only a candidate edge for discovery:
it does not authorize the peer, prove that it is live or reachable, promise
that it stores any content, or retain a blob. There is no retraction record.
That weak monotone meaning lets independently learned topology converge by
union without coupling transport policy to capability verification.

## Storage and synchronization compose by union

`Pile` stores blobs, native collection records, capability proofs, peer
evidence, and WANT records in one
append-only log. `ObjectStoreRemote` places immutable collection records under
content-derived object keys. The network layer uses authenticated Merkle walks
to union one team's PEER evidence, collection records, proofs, and optionally
blobs. Bounded fair pairwise PATCH reconciliation is the epidemic exchange,
while an authenticated DHT locates explicitly published providers of an exact
artifact handle. Merge/derive questions are answered from the converged local
record index. In every case convergence means unioning evidence; it does not
mean electing a winner.

Legacy branch and pin records remain decodable only so old piles can be
inspected, conservatively retained, and explicitly migrated. They are not part
of the current publication or authorization model.
