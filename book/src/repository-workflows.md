# Collection Workflows

TribleSpace publishes data into self-describing grow-only collections. A
collection has no mutable head and no privileged linear history: independent
signed commits coexist, replicas combine records by set union, and stored
merge or derivation equations preserve reusable physical work.

## Vocabulary

- **`Fragment`** — facts, descriptive metafacts, exported IDs, and referenced
  blob attachments produced as one composable value.
- **`BlobStore`** — immutable content-addressed bytes.
- **`CollectionStore`** — a grow-only set of native `COMMIT`, `MERGE`, and
  `DERIVE` records.
- **Collection descriptor** — a canonical `SimpleArchive` which describes a
  collection's anchor, member encoding, and independent READ and WRITE
  admission policies. A derived descriptor
  additionally links one concrete mapping entity carrying its algorithm and
  concrete parameters. The descriptor's content handle is the
  `CollectionHandle`.
- **`Collection<E>`** — a cheap descriptor handle whose
  `CollectionEncoding` type `E` owns the canonical member bytes and join.
  Constructing it validates that the runtime descriptor names `E`.
- **`Cover<E>`** — one typed collection identity plus a PATCH of distinct
  `Handle<E>` members selected for one read or derivation.
  Signatures, authors, and metadata remain queryable provenance, but are not
  coordinates of the value.
- **`TryFromCover<E>`** — reconstruction of one logical view from selected
  physical members. This is the payload-decoding boundary: a view may join
  eagerly or retain mmap-backed shards and query their union lazily.
- **WANT** — an orthogonal local request for content or existing computation;
  it is neither collection membership nor authority.

`MemoryRepo`, `Pile`, and the storage composition wrappers implement both the
blob and native collection surfaces. A collection is its descriptor handle;
the store remains the sole owner of I/O, durability, and lifetime.

An existing descriptor is opened through the same frozen read boundary used
for later observation:

```rust,ignore
let snapshot = storage.snapshot()?;
let models = Collection::<SimpleArchive>::open(&snapshot, collection_handle)?;
```

`open` fetches and validates the canonical descriptor and checks that its
member encoding is `SimpleArchive`. It never registers, rewrites, or otherwise
mutates the store.

## Publish a root collection

Register the descriptor once, then pass its returned handle to store
operations:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode,
    CapabilityProofBundle, CapabilityResource,
};
use triblespace::core::{
    blob::encodings::simplearchive::SimpleArchive,
    collection::{AdmissionPolicy, CollectionPolicy, ACTION_WRITE},
    repo::CapabilityProofStore,
};
use triblespace::prelude::*;

let team_key = SigningKey::generate(&mut OsRng);
let writer = SigningKey::generate(&mut OsRng);
let team = team_key.verifying_key();
let writer_subject = writer.verifying_key();
let mut storage = MemoryRepo::default();
let models = storage.collection(
    "models",
    CollectionPolicy::new(
        AdmissionPolicy::direct(team),
        AdmissionPolicy::direct(team),
    ),
)?;
let atom = CapabilityAtom::new(
    CapabilityAction::new(ACTION_WRITE),
    CapabilityResource::from(models.handle()),
);
let bundle = CapabilityProofBundle::issue_root(
    &team_key,
    CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
    writer_subject,
)?;
let (proof, claims) = bundle.into_parts();
for claim in claims {
    storage.put::<SimpleArchive, _>(claim)?;
}
storage.insert_proof(proof)?;

let commit = storage.commit(
    models,
    &writer,
    entity! { metadata::name: "first-model" },
)?;
let snapshot = storage.snapshot()?;
let cover = models.admitted(&snapshot)?;
assert!(cover.contains(Handle::<SimpleArchive>::from_hash(commit.data())));
storage.flush()?;
```

Local publication deliberately performs no authorization check: the local
store is a grow-only claim ledger, not an access-control boundary. Observation
loads the independent policies from the descriptor. A policy root is admitted
directly; every other author needs enough resident proof paths for exact
`ACTION_WRITE` on this descriptor. Each operation observes the clock once and
verifies every matching proof. Invalid, expired, irrelevant, or incomplete
candidate evidence grants nothing; inability to enumerate the proof store
remains an error.

READ and WRITE are explicit because both participate in collection identity.
Either may be `Open` or a canonical quorum over capability roots, with
independent invoke and delegation thresholds. Derived collections state their
own policies rather than inheriting ambient authority from a source or a
network-wide team scope.

### What publication writes

One `store.commit(collection, signer, fragment)` performs these semantic steps:

1. consume the fragment once into attachments, facts, and metafacts;
2. store the fragment's attachments;
3. encode and store the facts as the canonical `SimpleArchive` member;
4. encode and store metafacts as the mandatory canonical metadata
   `SimpleArchive`;
5. insert a signed `COMMIT` naming the already typed collection, data, and
   metadata handles.

Dependencies precede the record which gives them authority. Publication does
not flush implicitly. Call `flush()` at the application's chosen durability
boundary or explicitly close the backend. Repeating the same fragment with the
same identity produces the same intrinsic record and is a no-op; different
commits coexist.

`COMMIT` is deliberately a source operation over authored `Fragment` values.
Other collection encodings enter the lattice through reproducible `DERIVE` and
`MERGE` records rather than alternative signed leaf formats.

Importers which must validate additional artifacts before making the source
commit visible use the same path with an explicit pause before step 5:

```rust,ignore
let prepared = PreparedCollectionCommit::from_fragment(candidate);
let mut staged = prepared.stage_for(&mut storage, models, &signer)?;

// Dependencies are resident, but COMMIT is still withheld. Any validation or
// reproducible DERIVE/MERGE publication can use this exact store now.
validate_candidate(staged.store_mut())?;

let commit = staged.finalize()?; // the sole signed COMMIT insertion
```

Preparation is store-free and dropping either a prepared or staged value never
publishes a commit. `stage_for` accepts `Collection<SimpleArchive>`, not a raw
handle or a reconstructed descriptor fragment.

## The native algebra

The collection descriptor is the only collection-control structure represented
as a trible archive. The algebra records are fixed-width native records:

```text
COMMIT(collection, data, metadata, author, signature)  // 192 bytes
MERGE(collection, low, high, result)                   // 128 bytes
DERIVE(target, input, output)                          //  96 bytes
```

`COMMIT` is a signed exogenous assertion: no machine can recompute whether an
author intended to publish a member. `MERGE` is an exact join equation within
one collection. `DERIVE` is one observation of the mapping linked by its
target descriptor; that descriptor already names its source, mapping
algorithm, and concrete mapping parameters.

Merge inputs are canonically ordered and every record has an intrinsic ID over
its kind and exact payload. `CollectionStore::insert` therefore implements set
insertion rather than an update. Concatenating stores unions evidence.

Unsigned equations are materialized computation, not authority. Publishing a
`MERGE` or `DERIVE` records work which has already been performed; warm
resolution follows that equation without executing the join or mapping again.
Equation trust belongs at the store/synchronization boundary. Blob residency
is independent: an absent result is a cache miss and cannot suppress an
available explicit cover member.

Local publication remains unconditional. A publisher which needs to predict
whether an authority-aware observation will admit a signer can freeze a store
snapshot and call `collection.writer_is_admitted(&snapshot, signer)`: it checks
the descriptor WRITE policy and resident exact WRITE evidence without scanning
collection commits or publishing anything.

## Known-prefix snapshots and covers

`store.snapshot()` freezes one immutable observation containing blob bytes,
collection records, capability proofs, and peer evidence from the same known
prefix. A collection then performs admission against that observation:

```rust,ignore
let snapshot = store.snapshot()?;
let admitted = collection.admitted(&snapshot)?;
let physical = admitted.resolve(&snapshot)?;
let value = V::try_from_cover(&physical, &snapshot)?;
```

`admitted` is the semantic COMMIT frontier: it verifies the descriptor's exact
WRITE policy and forms a
`Cover<E>` from distinct payload handles signed by the admitted subjects.
`resolve` may select a resident support-equivalent decomposition using stored
`MERGE` and `DERIVE` evidence. `TryFromCover<E>` then constructs the logical
value solely from that selected physical cover and the same frozen store
snapshot. `collection.read::<V, _>(&snapshot)` is the convenience form of these
three steps. For a `SimpleArchive`, `V = TribleSet`; for a
`SuccinctArchiveBlob`, `V` may be an mmap-backed union retaining selected
shards.

Consumers which need the exact strictly verified COMMIT roots selected during
admission use `collection.admitted_with_commits(&snapshot)`; later claims over
the same payload remain broader provenance rather than retroactive roots.

This is a coherent **known-prefix** observation, not a global latest
transaction. A concurrent immutable insert may appear on this call or a later
call. Every cover member nevertheless has an admitted signed assertion in that
snapshot. Admission does not fetch payload bytes; physical resolution selects a
resident support-equivalent cover, and the requested typed view decodes its
members.

Admission does not fetch or materialize member blobs. Keep the returned cover
when another component will select or build a representation:

```rust,ignore
let snapshot = storage.snapshot()?;
let cover = models.admitted(&snapshot)?;
let physical = cover.resolve(&snapshot)?;
let facts = TribleSet::try_from_cover(&physical, &snapshot)?;
```

Exact replay does not need a publishing key, re-run admission, or retain any
signed commit or metadata. The opaque cover itself names the exact descriptor
and payload identities. Use `cover.commits(&snapshot)` when currently
resident authorship and metadata provenance matters; zero commits is a valid
answer and does not invalidate replay. Commits whose data handles are absent
from the cover remain inert. Replaying an opaque payload frontier still uses a
single store snapshot for resolution and member reads.

## Reuse merge work without changing meaning

A logical collection value is the join of a cover's members. It does not need
one monolithic blob. A resolver may choose members consisting of committed
payloads and stored merge results:

```text
    a       b       c           explicit payloads
     \     /        |
      a⊔b           |           reusable MERGE result
        \           /
         (a⊔b)⊔c                logical collection value
```

Distinct covers can have the same support: `{a, b}` and `{a⊔b}` are different
PATCH sets, but the stored `MERGE` equation records that they denote the same
join. This is useful for LSM-like maintenance: small commits remain
independently attributable, while deterministic merges amortize reads into
larger canonical shards. A selected target cover is replaceable computation,
never a second history or a new authority root.

## Derive another representation

Suppose `f` is a canonical join homomorphism, represented in the API by a
`CollectionMapping` whose associated types are `Source` and `Target`:

```text
f(a ⊔ b) = f(a) ⊔ f(b)
```

Then a resolver may derive a merged source once, derive leaves separately and
merge their images, or reuse any stored mixture already present. `DERIVE`
records expose those reusable edges across collection lattices. Newly executed
joins and mappings publish every successful result and equation, even when a
later planning or storage step fails or selects another route. Publication is
operation-ordered rather than phase-batched, so a failure leaves the complete
successful prefix addressable instead of stranding its blobs without their
equations. Canonical joins, mappings, and logical cover views receive one
frozen store snapshot and may resolve immutable dependencies named by their
inputs; unrelated resident blobs are never ambient semantic input.

The SuccinctArchive facade applies this model as two ordinary derivations:

```text
SimpleArchive --DERIVE--> SuccinctArchiveBlob --DERIVE-->
    Rank9AcceleratedSuccinctArchiveBlob
```

```rust,ignore
use triblespace::core::collection::succinctarchive_union::{
    RawToRank9AcceleratedMapping, SimpleToSuccinctMapping,
    SuccinctArchiveCollection,
};

let source = storage.collection("models", source_policy)?;
let raw = storage.derive(source, SimpleToSuccinctMapping, raw_policy)?;
let accelerated = storage.derive(raw, RawToRank9AcceleratedMapping, accelerated_policy)?;
let succinct = SuccinctArchiveCollection::new(source, raw, accelerated);

let archive = succinct.ensure(&mut storage, &cover)?;
let same_archive = succinct.attach(&mut storage, &cover)?;

// The same algebra edges are available directly on storage.
let raw_cover = storage.ensure::<SimpleToSuccinctMapping>(raw, &cover)?;
let accelerated_cover =
    storage.ensure::<RawToRank9AcceleratedMapping>(accelerated, &raw_cover)?;
```

- `attach` is read-only, performs no collection algebra, and requires a
  complete resident physical cover.
- `ensure` is the singular construction and maintenance path. It completes the
  raw projection, deterministically carries colliding raw target members by
  serialized-size tier, then ensures a support-equivalent accelerated cover and
  returns its query view.

At each target lattice node, `ensure` reuses the resident result first. If the
result is absent, it joins the two corresponding target children when both are
resident. A capacity-terminal target join falls through to the corresponding
resident source node. If that source node is absent, or its
mapping reports a capacity boundary, planning reuses any already complete lower
target cover and otherwise descends to the source children. It never creates a
source merge merely as a planning shortcut. A source join is materialized only
when the selected target join names its exact result as an immutable
representation dependency. Every source or target `MERGE` and cross-lattice
`DERIVE` it actually computes is stored with its equation, including useful
work completed before a later capacity or fatal result.

The maintenance policy has no knob: a raw target member belongs to
`floor(log2(max(1, serialized_len)))`, and the lowest two content handles in
the lowest colliding tier are carried first. A capacity-limited encoding may
leave a collision stable; otherwise the resulting cover has at
most one member per tier. Pairwise-disjoint carries in one tier share a
deterministic semantic plan, but each output is constructed against a cheap
fresh store snapshot and published immediately. The exact per-point planner is
re-entered before another tier is selected. This avoids a full semantic
re-probe per pair without retaining a tier of newly generated bytes in memory.

Every position uses the same `Cover<E>` shape, but its typed handles cannot be
mixed across representations. `Cover<SimpleArchive>` contains only
`Handle<SimpleArchive>`; `Cover<SuccinctArchiveBlob>` contains only
`Handle<SuccinctArchiveBlob>`; the second stage uses
`Handle<Rank9AcceleratedSuccinctArchiveBlob>`. Stored `MERGE` equations define
support-equivalent routes; `Cover` carries no route-mode bit. Ordinary raw
Succinct derivation follows the resident-node priority above while preserving
support equal to the source cover. The accelerated stage resolves the ordinary
derived lattice over the raw cover selected upstream. Its cover-aware view
reads each embedded raw handle through the store snapshot and validates the
exact raw/index pair before constructing the query runtime. There is no
separate member-image mode.

None of them signs a replacement root, advances a head, flushes implicitly, or
adds a special manifest. [Regular-path summaries](regular-path-indexes.md) and
Rank9 acceleration both use the same collection algebra. The accelerated
encoding is a Merkle root whose first 32 bytes name its exact portable raw
child. It is also a full lattice: resident accelerated children `A(a)` and
`A(b)` join canonically to `A(a ⊔ b)`. The named raw result must be resident
before that accelerated result is published. If it is absent, the generic
storage executor first publishes the ordinary source `MERGE(a,b,c)`, then
retries and publishes `MERGE(A(a),A(b),A(c))`. Each operation emits one blob.
The commuting-square law implies `DERIVE(c,A(c))`, so that redundant edge need
not be stored explicitly. Physical-cover resolution excludes an accelerated
member whose named raw child is unavailable and retries a finer
support-equivalent route; the typed view repeats the raw/index check only as a
defensive boundary for callers that construct a physical cover directly.

## WANT missing content or computation

Sparse evidence discovery deliberately does not fetch commit dependencies.
`WantStore` records operational interest with three request shapes:

- `Blob(handle)` — obtain those exact bytes;
- `Merge(collection, low, high)` — discover an existing matching merge result;
  and
- `Derive(target, input)` — discover an existing matching derivation; the
  target descriptor already names the source collection and concrete mapping.

`Blob(H)` is the only exact-content identity. A reconciler may satisfy it from
local workers or discover providers under opaque KDF(H), without activating or
even naming a collection. The provider proves H first and the requester second,
with both proofs bound to the authenticated endpoints; H itself is never sent,
and landed bytes must hash to H. The answer to an operation WANT is the
ordinary native equation; obtaining its result bytes is a separate blob WANT.
A WANT grants no collection authority and does not change the value of any
collection.

## Migrate a legacy branch explicitly

Old piles may contain signed commit DAGs and mutable pin records. Current
readers retain an immutable `PinSnapshot` and legacy decoders so operators can
inspect and migrate that evidence without restoring the old publication API.

```text
trible pile migrate data.pile branch-to-collection \
  --branch legacy-events \
  --collection-name events \
  --signing-key ./writer.key
```

The command is deliberately same-pile: source commit blobs must already be
resident. It freezes the selected legacy head, validates the complete reachable
DAG, and converts each authored node into a native commit using its exact
`repo::content` and `metadata::archive` handles. A missing metadata archive maps
to the canonical empty archive. Contentless merge wrappers are validated but do
not become members.

With no further options the target descriptor gives the migration signing key
one-root direct READ and WRITE policies. The resulting commits are therefore
admitted directly by ordinary collection admission against a store snapshot.

The migration-only `--authority` option instead uses another trust root for
both direct policies:

```text
trible pile migrate data.pile branch-to-collection \
  --branch legacy-events \
  --collection-name events \
  --authority <64-hex-character-ed25519-public-key> \
  --signing-key ./writer.key
```

Local publication remains unconditional, so this form still writes commits
signed by the migration key. A later read admits them only when the store holds
enough exact root-to-signer `ACTION_WRITE` evidence for the resulting
descriptor handle. The migration command does not invent, scan for, or store
that delegation.

The complete source DAG and every prepared target element are validated before
the target descriptor, dependency, or commit is published. Storage failures
remain backend errors; authorization is deliberately deferred to reads rather
than treated as permission to append locally.

Legacy wrapper parents, messages, timestamps, authors, and signatures are not
silently reinterpreted as application metadata. Two source nodes with identical
data and semantic metadata map to one intrinsic native commit. Re-running with
the same collection identity and key is idempotent.

Migration is the only reason application-facing tooling needs to name a legacy
branch. New code publishes directly to collections.

## Operational invariants

- Persist dependencies before the record that makes them meaningful.
- Treat a cover's payload identities as semantic ground truth. Select a
  complete resident support-equivalent physical cover, then let the requested
  typed view decode exactly those members. Signed commits and metadata remain
  lazy provenance queried separately.
- Treat stored unsigned equations as reusable materialized LSM work. Never
  replay algebra merely to trust a local equation; apply future trust/quorum
  policy at record admission instead.
- Persist every successful join or mapping. Yard/GC policy alone decides when
  its result bytes leave local storage.
- Keep admission, retention, and WANT policy orthogonal.
- Carry exact covers across derivation boundaries instead of asking for an
  ambient “latest”.
- Flush at explicit application durability boundaries.
- Merge stores by union; never choose meaning from append order.

These rules are sufficient for both low-latency single-process use and sparse
distributed collection maintenance without introducing a second execution
model.
