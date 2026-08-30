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
evidence which a reader validates under the collection's encoding and mapping.

That separation is why a materialized index does not become ground truth merely
because it is convenient, and why collecting an accelerator does not erase the
committed facts from which it can be rebuilt.

Who may make that signed assertion can be proven without making storage a
policy oracle. A capability-guarded descriptor names an external trust root,
while the facade discovers resident `K0 (S C K)+` proofs rooted there. The
native proof binds each issuer,
exact keyless claim handle, and delegate key; the ordered claim blobs carry the
action/resource, mode, validity, and parent-claim restrictions. Ordinary
collection operations verify the resulting meet directly at one clock
instant against exact `ACTION_WRITE` on the descriptor. Merely finding an
unverified or irrelevant proof in storage grants nothing.

## Architectural layers

```text
┌──────────────────────────────────────────────────┐
│ Application                                      │
│ entity! · Fragment · find! · pattern!            │
├──────────────────────────────────────────────────┤
│ Collection facades                               │
│ publish, select exact covers, materialize views  │
├──────────────────────────────────────────────────┤
│ Collection algebra                              │
│ signed COMMIT · checked MERGE · checked DERIVE   │
├──────────────────────────────────────────────────┤
│ Storage                                          │
│ CollectionStore · BlobStore · WantStore          │
│ ArtifactOfferStore · CapabilityProofStore         │
│ PeerStore                                        │
├──────────────────────────────────────────────────┤
│ Data and representations                         │
│ TribleSet/PATCH · SimpleArchive · SuccinctArchive│
└──────────────────────────────────────────────────┘
```

The boundaries are deliberately narrow. Query constraints do not know how
bytes were published. A collection encoding or mapping does not decide
replication policy. A blob store does not infer authority from a handle it
happens to contain.

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

- a human-readable UTF-8 name;
- one mandatory descriptor-local authority;
- the canonical member encoding; and
- a reach law governing permissionless relay.

A derived descriptor replaces the name with its source collection and one
concrete mapping entity. Canonical builders derive that entity's id, while
readers preserve the substitution rule by validating its algorithm and
parameters instead of its minting history. The entity names a mapping algorithm
and carries its concrete parameters as ordinary tribles. The target encoding and
mandatory authority remain local to the derived descriptor; authority never
inherits from the source. Descriptions of the encoding and mapping algorithm
travel in the same archive, so a record naming the descriptor remains
interpretable without a separate registry entry.

`CollectionStore` contains three native record kinds:

| Record | Meaning | Dense payload |
|---|---|---:|
| `COMMIT(C, x, metadata, author, signature)` | The author asserts `x` as an independent member of `C`. | 192 bytes |
| `MERGE(C, a, b, c)` | Under `C`'s join law, `a ⊔ b = c`. | 128 bytes |
| `DERIVE(T, a, b)` | The mapping named by target `T` maps source element `a` to target element `b`. | 96 bytes |

All three have intrinsic IDs derived from their exact canonical payload. A
repeat insert is a no-op. `COMMIT` is signed because its assertion cannot be
recomputed; `MERGE` and `DERIVE` are unsigned because correctness comes from
the encoding or mapping plus exact bytes, not the identity of the machine that
performed the work.

The algebra has no distinguished head. Several commits coexist, and the value
of a selected collection view is the join of its admitted members. This makes a
commit the atomic publication boundary without inventing a mutable register
above it.

## Publishing and observing

The collection value is its canonical descriptor handle. The descriptor
carries one mandatory local authority, and the storage backend owns I/O and
durability. `store.collection(descriptor_fragment)` registers and offers the
descriptor's complete attachment closure. `store.commit(collection, signer,
fragment)` then loads and structurally validates that descriptor and publishes attachments,
canonical data, canonical metadata, and the signed native record in dependency
order. Local publication performs no authorization check and no implicit
flush: authorization governs which resident claims another operation admits,
not what a process may append to its own store.

Reads are exact about what they observed, not magical about global time:

- `store.snapshot()` freezes blob bytes, collection records, capability proofs,
  and peer evidence from one coherent known prefix;
- `collection.admitted(&snapshot)` applies the descriptor authority and
  resident delegation proofs to obtain one semantic `Cover<E>` without
  fetching member data;
- `cover.resolve(&snapshot)` selects a resident support-equivalent physical
  decomposition; and
- `V::try_from_cover(&physical, &snapshot)` reconstructs the logical value
  through that same immutable observation. `collection.read(&snapshot)` is the
  convenience form of the final three steps.

Cover identity is the collection descriptor plus distinct payload handles.
Signer, signature, and metadata claims currently known to the store remain
queryable, but no claim is required for replay, and another claim over the same
payload does not change the cover or repeat data work.

Each store snapshot observes one known prefix of an append-only store. A
concurrent commit may appear now or on the next call, but one observation never
combines records from one prefix with payloads from another. The
descriptor authority and every delegate authorized by a valid resident proof
participate; unauthorized commits remain inert.

## Derived physical representations

The same logical data can be projected into encodings optimized for a
particular task. A canonical SuccinctArchive collection or a regular-path
summary is connected to its source by a mapping which is a join homomorphism:

```text
f(a ⊔ b) = f(a) ⊔ f(b)
```

That law permits either route through the evidence graph: merge source shards
and derive once, derive individual shards and merge their images, or reuse any
validated mixture already present. An opaque source cover fixes the logical
value while a resolver chooses a target cover with equal support.
Different target covers may denote the same join, and every lattice position
uses the same `Cover<E>` shape while retaining its own typed member handles.
Missing derived artifacts are cache misses, not missing facts.

Route freedom belongs to the input collection expected by the mapping rather
than to a flag on `Cover`: ordinary Succinct construction may reuse any
validated equal-support route. Rank9 acceleration is another ordinary derived
collection. Its members are ordinary
`Blob<Rank9AcceleratedSuccinctArchiveBlob>` values whose first 32 bytes name the
exact portable `SuccinctArchiveBlob` child carrying their source rows. The root
and named child together are a complete source-bound accelerated encoding, not
a separate sidecar or runtime artifact.

Raw Succinct members own the directly materializable join. Rank9 roots do not
define another physical join: maintenance compacts raw members first, then the
ordinary raw-to-accelerated `DERIVE` maps that exact source blob to its query
index. Thus raw cover `{a, b}` maps to `{f(a), f(b)}` even when resident
evidence also proves `a join b = c`; only explicit raw compaction to cover
`{c}` changes the accelerated cover to `{f(c)}`. Exact derivation stores the
selected source before the target root and semantic record. A cover-aware view
follows the embedded handle through its store snapshot, validates the exact raw/index
pair, and only then builds the transient query runtime. A root whose raw child
is absent is nonresident; ensuring reconstructs the exact accelerated image
from that raw source.

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

## OFFER is local service intent

An `ArtifactOfferStore` is the positive, grow-only set of artifact handles this
store is willing to serve. OFFER is deliberately weaker than both residency
and WANT: it neither proves that bytes are currently present nor requests that
they be fetched. It grants no authority, contributes no collection evidence,
does not join synchronized inventory, and never becomes a garbage-collection
root. This narrow meaning lets store-local publication intent survive restart
and conservative rewriting without turning policy into ownership.

## Peer evidence is topology, not authority

`PeerStore` holds positive `PEER(team_public_key, peer_public_key)` routing
facts as another grow-only set. A fact is only a candidate edge for discovery:
it does not authorize the peer, prove that it is live or reachable, promise
that it stores any content, or retain a blob. There is no retraction record.
That weak monotone meaning lets independently learned topology converge by
union without coupling transport policy to capability verification.

## Storage and synchronization compose by union

`Pile` stores blobs, native collection records, capability proofs, peer
evidence, OFFER records, and WANT records in one
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
