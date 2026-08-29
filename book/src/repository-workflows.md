# Collection Workflows

TribleSpace publishes data into self-describing grow-only collections. A
collection has no mutable head and no privileged linear history: independent
signed commits coexist, replicas combine records by set union, and validated
merge or derivation equations provide reusable physical work.

## Vocabulary

- **`Fragment`** — facts, descriptive metafacts, exported IDs, and referenced
  blob attachments produced as one composable value.
- **`BlobStore`** — immutable content-addressed bytes.
- **`CollectionStore`** — a grow-only set of native `COMMIT`, `MERGE`, and
  `DERIVE` records.
- **Collection descriptor** — a canonical `SimpleArchive` which describes a
  collection's anchor, member encoding, and reach law. A derived descriptor
  additionally links one concrete mapping entity carrying its algorithm and
  concrete parameters. The descriptor's content handle is the
  `CollectionHandle`.
- **`Collection<E>`** — a cheap descriptor handle whose
  `CollectionEncoding` type `E` owns the canonical member bytes, validation,
  and join. Constructing it validates that the runtime descriptor names `E`.
- **`Cover<E>`** — one typed collection identity plus a PATCH of distinct
  `Handle<E>` members selected for one read or derivation.
  Signatures, authors, and metadata remain queryable provenance, but are not
  coordinates of the value.
- **`TryFromCover<E>`** — reconstruction of one logical view from validated
  physical members. A view may join eagerly or retain mmap-backed shards and
  query their union lazily.
- **WANT** — an orthogonal local request for content or existing computation;
  it is neither collection membership nor authority.
- **OFFER** — positive local willingness to serve an artifact; it is neither
  residency, demand, synchronized inventory, retention, nor authority.

`MemoryRepo`, `Pile`, and the storage composition wrappers implement both the
blob and native collection surfaces. A collection is its descriptor handle;
the store remains the sole owner of I/O, durability, and lifetime.

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
    collection::{reach, simplearchive_union, ACTION_WRITE},
    repo::CapabilityProofStore,
};
use triblespace::prelude::*;

let team_key = SigningKey::generate(&mut OsRng);
let writer = SigningKey::generate(&mut OsRng);
let team = team_key.verifying_key();
let writer_subject = writer.verifying_key();
let mut storage = MemoryRepo::default();
let models = storage.collection::<SimpleArchive>(simplearchive_union::descriptor(
    "models",
    team,
    reach::private(),
))?;
let atom = CapabilityAtom::new(
    CapabilityAction::new(ACTION_WRITE),
    CapabilityResource::from(models),
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
let snapshot = storage.snapshot::<TribleSet, _>(models)?;
assert!(snapshot
    .cover()
    .contains(Handle::<SimpleArchive>::from_hash(commit.data())));
storage.flush()?;
```

Local publication deliberately performs no authorization check: the local
store is a grow-only claim ledger, not an access-control boundary. Observation
loads the mandatory authority from the descriptor. Commits self-signed by that
authority are admitted directly; every delegated author needs a resident proof
for exact `ACTION_WRITE` on this descriptor. Each operation observes the clock
once and verifies every matching proof. Invalid, expired, irrelevant, or
incomplete candidate evidence grants nothing; inability to enumerate the proof
store remains an error.

The reach argument is explicit because it participates in collection identity.
`reach::private()` declares no permissionless relay; `reach::public()` states
that any holder may relay verified commits. A derived collection states its
own reach independently of its source. Reach is distinct from explicit
team-store synchronization: SYNC_TEAM authority admits the complete inventory
of a store already dedicated to that team, including records for private
descriptors.

### What publication writes

One `store.commit(collection, signer, fragment)` performs these semantic steps:

1. fetch and structurally validate the already registered descriptor;
2. store the fragment's attachments;
3. encode the value using `E`'s canonical member encoding;
4. encode metafacts as the mandatory canonical metadata `SimpleArchive`;
5. durably offer those dependencies; and
6. insert a signed `COMMIT` naming the descriptor, data, and metadata handles.

Dependencies precede the record which gives them authority. Publication does
not flush implicitly. Call `flush()` at the application's chosen durability
boundary or explicitly close the backend. Repeating the same fragment with the
same identity produces the same intrinsic record and is a no-op; different
commits coexist.

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

Unsigned equations are replaceable computation, not authority. A resolver
admits them only when the target encoding, declared mapping algorithm and
parameters, and content identities validate. An invalid or unavailable result is a cache miss and
cannot suppress an explicit cover member.

## Known-prefix snapshots and covers

`store.snapshot(collection)` observes one clock instant, discovers resident
capability proofs rooted at the descriptor authority, and verifies them against
the exact collection write request before discovering commits by the resulting
subjects.
Their distinct data handles form a `Cover<E>`. It opens one target reader and
constructs a logical view from only that payload set. The returned
`Snapshot<E, V, R>` keeps the logical view `V`, exact cover, and reader together
so downstream code cannot accidentally pair one logical frontier with a
different physical view. For a
`SimpleArchive`, `V = TribleSet`; for a `SuccinctArchiveBlob`, `V` may be
an mmap-backed `UnionArchive` retaining all selected shards.

This is a coherent **known-prefix** observation, not a global latest
transaction. A concurrent immutable insert may appear on this call or a later
call. Every cover member was nevertheless admitted and its payload validated
for the returned snapshot, or the call fails instead of returning a partial
set.

`store.cover(collection)` performs the same admission check and
record discovery, but does not fetch or materialize member blobs. Freeze a
cover when another component will select or build a representation:

```rust,ignore
let cover = storage.cover(models)?;
let facts = storage.materialize(&cover)?;
```

Exact replay does not need a publishing key, re-run admission, or retain any
signed commit or metadata. The opaque cover itself names the exact descriptor
and payload members to validate. Use `store.claims(&cover)` when currently
resident authorship and metadata provenance matters; zero claims is a valid
answer and does not invalidate replay. Commits whose data handles are absent
from the cover remain inert. Call `snapshot` when admission and
materialization should be one coherent operation; call `materialize(&cover)`
when replaying an opaque payload frontier.

## Reuse merge work without changing meaning

A logical collection value is the join of a cover's members. It does not need
one monolithic blob. A resolver may choose members consisting of committed
payloads and validated merge results:

```text
    a       b       c           explicit payloads
     \     /        |
      a⊔b           |           reusable MERGE result
        \           /
         (a⊔b)⊔c                logical collection value
```

Distinct covers can have the same support: `{a, b}` and `{a⊔b}` are different
PATCH sets, but the validated `MERGE` equation proves that they denote the same
join. This is useful for LSM-like maintenance: small commits remain
independently attributable, while deterministic merges amortize reads into
larger canonical shards. A selected target cover is replaceable computation,
never a second history or a new authority root.

## Derive another representation

Suppose `f` is a canonical join homomorphism, represented in the API by a
`CollectionMapping<Source, Target>`:

```text
f(a ⊔ b) = f(a) ⊔ f(b)
```

Then a resolver may derive a merged source once, derive leaves separately and
merge their images, or reuse any validated mixture already present. `DERIVE`
records expose those reusable edges across collection lattices.

The SuccinctArchive facade applies this model as two ordinary derivations:

```text
SimpleArchive --DERIVE--> SuccinctArchiveBlob
               --DERIVE--> Rank9AcceleratedSuccinctArchiveBlob
```

```rust,ignore
use triblespace::core::collection::succinctarchive_union::SuccinctArchiveCollection;

let succinct = SuccinctArchiveCollection::new(
    "models",
    team,
    reach::private(), // source reach, and therefore source identity
    team,
    reach::private(), // target reach
);

let archive = succinct.ensure_exact(&mut storage, &cover)?;
let same_archive = succinct.attach_exact(&mut storage, &cover)?;
let compact_archive = succinct.compact_exact(&mut storage, &cover)?;
```

- `attach_exact` is read-only and requires a complete valid resident cover.
- `ensure_exact` reuses valid equations, computes missing canonical images, and
  publishes dependencies before new records.
- `compact_exact` deterministically compacts the raw target cover, then ensures
  the matching accelerated cover and returns its query view.

Every position uses the same `Cover<E>` shape, but its typed handles cannot be
mixed across representations. `Cover<SimpleArchive>` contains only
`Handle<SimpleArchive>`; `Cover<SuccinctArchiveBlob>` contains only
`Handle<SuccinctArchiveBlob>`; the second stage uses
`Handle<Rank9AcceleratedSuccinctArchiveBlob>`. The target descriptor and its
bound `CollectionMapping<Source, Target>` determine route freedom. Ordinary raw
Succinct derivation may choose any cheapest validated route whose support
equals the source cover. The accelerated stage maps the exact raw cover selected
upstream, while its attached artifact retains each root, raw child, and query
runtime together. Exactness is a property of the mapping, not a mode bit or an
untyped hash convention.

None of them signs a replacement root, advances a head, flushes implicitly, or
adds a special manifest. [Regular-path summaries](regular-path-indexes.md) and
Rank9 acceleration both use the same collection algebra. The accelerated
encoding is a Merkle root whose first 32 bytes name its exact portable raw
child. Its join operates on complete attached artifacts, so mapping and then
joining produces the same root as joining the raw inputs and mapping once.
Publication writes child before root before the ordinary `DERIVE` or `MERGE`;
an incomplete closure is merely a nonresident route which `ensure_exact` can
reconstruct.

## WANT missing content or computation

Sparse evidence discovery deliberately does not fetch commit dependencies.
`WantStore` records local operational interest with three request shapes:

- `Blob(handle)` — obtain exact bytes;
- `Merge(collection, low, high)` — discover an existing matching merge result;
  and
- `Derive(target, input)` — discover an existing matching derivation; the
  target descriptor already names the source collection and concrete mapping.

A reconciler may satisfy those questions from local workers or peers. The
answer to an operation WANT is the ordinary native equation; obtaining its
result bytes is a separate blob WANT. A WANT grants no authority and does not
change the value of any collection.

## OFFER local service intent

`ArtifactOfferStore` durably records a grow-only set of handles this store is
willing to serve. `offer_all` is the primary operation so callers can publish a
whole successful workflow batch through one backend boundary; repeated handles
and already-known offers are idempotent. `offers_snapshot` is a cheap immutable
deterministic observation for local service policy.

OFFER is intentionally not a second collection or network inventory. It says
nothing about authority, reach, current residency, retention, or demand. A
conservative rewrite carries the marker forward but may collect the artifact,
leaving dormant intent that becomes effective if identical content returns.

Piles populated before OFFER was part of normal publication can recover this
local intent explicitly:

```text
trible pile migrate data.pile seed-artifact-offers --dry-run
trible pile migrate data.pile seed-artifact-offers
```

The command freezes native collection records, then observes one resident-blob
and one existing-OFFER snapshot. A strictly signed COMMIT contributes the
resident conservative closure of its collection descriptor, data, and
metadata, including resident attachments. MERGE contributes only its resident
descriptor and result; DERIVE contributes only its resident target descriptor
and output. Inputs are reproducible provenance rather than serving intent.
Invalid commits are inert, missing references are counted without creating
WANTs, corrupt selected content fails the run before any new OFFER is written,
and unrelated resident blobs are never scanned. Re-running is idempotent.

This is intentionally a dedicated operator action rather than part of generic
schema migration: it states willingness to serve historical artifacts. It does
not add collection evidence, bind a team, require a cover, or turn OFFER into
a garbage-collection root.

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

With no further options the target descriptor's mandatory authority is the
migration signing key. The resulting commits are therefore admitted directly
by ordinary `cover` and `snapshot` calls.

To register the migrated collection under a different authority, name that
trust root explicitly:

```text
trible pile migrate data.pile branch-to-collection \
  --branch legacy-events \
  --collection-name events \
  --authority <64-hex-character-ed25519-public-key> \
  --signing-key ./writer.key
```

Local publication remains unconditional, so this form still writes commits
signed by the migration key. A later read admits them only when the store holds
an exact root-to-signer `ACTION_WRITE` proof for the resulting
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
- Treat an opaque cover's payload members as mandatory ground truth; fail loud
  when its descriptor or data is absent or invalid. Signed commits and metadata
  are unnecessary for replay and remain lazy provenance queried separately.
- Treat unsigned equations as optional, freshly validated cache evidence.
- Keep reach, trust, retention, and WANT policy orthogonal.
- Carry exact covers across derivation boundaries instead of asking for an
  ambient “latest”.
- Flush at explicit application durability boundaries.
- Merge stores by union; never choose meaning from append order.

These rules are sufficient for both low-latency single-process use and sparse
distributed collection maintenance without introducing a second execution
model.
