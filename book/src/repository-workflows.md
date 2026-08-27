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
  collection's anchor, element representation, join recipe, and reach law. Its
  content handle is the `CollectionHandle`.
- **Ticket** — the exact byte-identical set of signed commits selected as
  admitted ground truth for one read or derivation.
- **WANT** — an orthogonal local request for content or existing computation;
  it is neither collection membership nor authority.
- **OFFER** — positive local willingness to serve an artifact; it is neither
  residency, demand, synchronized inventory, retention, nor authority.

`MemoryRepo`, `Pile`, and the storage composition wrappers implement both the
blob and native collection surfaces. `ObjectStoreRemote` exposes the equivalent
asynchronous primitives and can be adapted to the synchronous collection facade
through `Blocking`.

## Publish a root collection

`Collection<S>` owns one backend, canonical descriptor, and signing identity:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::capability::{
    CapabilityAction, CapabilityAtom, CapabilityClaim, CapabilityMode,
    CapabilityProofBundle, CapabilityResource,
};
use triblespace::core::collection::{reach, simplearchive_union, ACTION_WRITE};
use triblespace::prelude::*;

let team_key = SigningKey::generate(&mut OsRng);
let writer = SigningKey::generate(&mut OsRng);
let team = team_key.verifying_key();
let writer_subject = writer.verifying_key();
let name = CollectionName::new("models")?;
let mut storage = MemoryRepo::default();
let descriptor =
    simplearchive_union::descriptor(&name, team, Some(team), reach::private());
let target = descriptor.facts().clone().to_blob().get_handle();
let atom = CapabilityAtom::new(
    CapabilityAction::new(ACTION_WRITE),
    CapabilityResource::from(target),
);
let proof = CapabilityProofBundle::issue_root(
    &team_key,
    CapabilityClaim::root(atom, CapabilityMode::Invoke, None),
    writer_subject,
)?;
let mut models = Collection::new(
    storage,
    &name,
    team,
    writer,
    reach::private(),
    CollectionAdmission::capability(
        team,
        vec![CapabilityPresentation::new(writer_subject, proof)],
    ),
);

let commit = models.commit(entity! { metadata::name: "first-model" })?;
let snapshot = models.snapshot()?;
assert_eq!(snapshot.commits(), &[commit]);
models.flush()?;
let storage = models.into_storage();
```

Capability admission stores the trust root in the descriptor and retains each
expected leaf plus complete proof bundle in the facade. Each operation observes
the clock once and verifies the bundle against that root, leaf, and exact
`ACTION_WRITE` request for this descriptor before touching storage. Ordinary
reads admit every explicitly presented leaf, so a foreign author remains
visible when its proof is supplied. Invalid explicit evidence fails loud; an
empty presentation set is a valid policy that admits nobody.
`CollectionAdmission::Open` is the deliberate alternative which admits every
strictly verified signer.

The reach argument is explicit because it participates in collection identity.
`reach::private()` declares no permissionless relay; `reach::public()` states
that any holder may relay verified commits. A derived collection states its
own reach independently of its source. Reach is distinct from explicit
team-store synchronization: SYNC_TEAM authority admits the complete inventory
of a store already dedicated to that team, including records for private
descriptors.

### What publication writes

One `Collection::commit(fragment)` performs these semantic steps:

1. verify the facade's explicit admission evidence at one clock instant and
   require the local signer to be admitted for exact `WRITE`;
2. canonicalize and store the collection descriptor;
3. store the fragment's attachments;
4. encode facts as the canonical data `SimpleArchive`;
5. encode metafacts as the mandatory canonical metadata `SimpleArchive`; and
6. insert a signed `COMMIT` naming those exact three handles.

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
one collection. `DERIVE` is one observation of the join homomorphism described
by its target descriptor; the target already names its source and recipe.

Merge inputs are canonically ordered and every record has an intrinsic ID over
its kind and exact payload. `CollectionStore::insert` therefore implements set
insertion rather than an update. Concatenating stores unions evidence.

Unsigned equations are replaceable computation, not authority. A resolver
admits them only when the declared recipe and content identities validate. An
invalid or unavailable result is a cache miss and cannot suppress a signed
leaf.

## Known-prefix snapshots and exact tickets

`Collection::snapshot()` observes one clock instant, verifies every explicit
capability presentation, then discovers the exact strictly verified commits by
the resulting subjects. Open admission accepts every strict signer instead. It
opens one target reader and materializes only that admitted set. It returns
facts, commits, and reader together so downstream code cannot accidentally pair
one logical frontier with a different physical view.

This is a coherent **known-prefix** observation, not a global latest
transaction. A concurrent immutable insert may appear on this call or a later
call. Every selected commit is nevertheless present and valid in the returned
snapshot, or the call fails instead of returning a partial set.

`ticket()` performs the same admission check and target-record discovery, but
does not fetch or materialize the target collection's data and metadata blobs.
Freeze a ticket when another component will select or build a representation:

```rust,ignore
let ticket = models.ticket()?;

let source = SimpleArchiveCollection::new(
    name.clone(),
    team,
    models.admission().trust_root(),
    reach::private(),
);
let snapshot = source.snapshot_exact(models.storage_mut(), &ticket)?;
```

An exact-ticket facade does not need the publishing key. Ticket members may
have different authors, but each must byte-match one resident strictly verified
record for the exact descriptor. Commits in storage but absent from the ticket
remain inert. Live membership policy is not supplied as an ambient callback:
ordinary `ticket`, `snapshot`, and `materialize` all enforce the facade's same
explicit admission value.

## Reuse merge work without changing meaning

A logical collection value is the join of the selected committed leaves. It
does not need one monolithic blob. A resolver may choose an exact resident
cover consisting of leaves and validated merge results:

```text
    a       b       c           signed leaves
     \     /        |
      a⊔b           |           reusable MERGE result
        \           /
         (a⊔b)⊔c                logical collection value
```

Several physical covers can represent the same logical value. This is useful
for LSM-like maintenance: small commits remain independently attributable,
while deterministic merges amortize reads into larger canonical shards. The
cover is an optimization chosen under an exact ticket, never a second history
or a new authority root.

## Derive another representation

Suppose `f` is a canonical join homomorphism:

```text
f(a ⊔ b) = f(a) ⊔ f(b)
```

Then a resolver may derive a merged source once, derive leaves separately and
merge their images, or reuse any validated mixture already present. `DERIVE`
records expose those reusable edges across collection lattices.

The raw SuccinctArchive facade applies this model directly:

```rust,ignore
use triblespace::core::collection::succinctarchive_union::SuccinctArchiveCollection;

let trust_root = models.admission().trust_root();
let succinct = SuccinctArchiveCollection::new(
    name.clone(),
    team,
    trust_root,
    reach::private(), // source reach, and therefore source identity
    trust_root,
    reach::private(), // target reach
);

let archive = succinct.ensure_exact(models.storage_mut(), &ticket)?;
let same_archive = succinct.attach_exact(models.storage_mut(), &ticket)?;
let compact_archive = succinct.compact_exact(models.storage_mut(), &ticket)?;
```

- `attach_exact` is read-only and requires a complete valid resident cover.
- `ensure_exact` reuses valid equations, computes missing canonical images, and
  publishes dependencies before new records.
- `compact_exact` performs explicit deterministic tiered merges under the same
  ticket.

None of them signs a replacement root, advances a head, flushes implicitly, or
adds a special manifest. Rank9 fibers and [regular-path
summaries](regular-path-indexes.md) use the same collection algebra.

## WANT missing content or computation

Sparse evidence discovery deliberately does not fetch commit dependencies.
`WantStore` records local operational interest with three request shapes:

- `Blob(handle)` — obtain exact bytes;
- `Merge(collection, low, high)` — discover an existing matching merge result;
  and
- `Derive(target, input)` — discover an existing matching derivation; the
  target descriptor already names the source collection and recipe.

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
not add collection evidence, bind a team, require a ticket, or turn OFFER into
a garbage-collection root.

## Migrate a legacy branch explicitly

Old piles may contain signed commit DAGs and mutable pin records. Current
readers retain an immutable `PinSnapshot` and legacy decoders so operators can
inspect and migrate that evidence without restoring the old publication API.

```text
trible pile migrate data.pile branch-to-collection \
  --branch legacy-events \
  --collection-name events \
  --namespace <64-hex-character-ed25519-public-key> \
  --signing-key ./writer.key
```

The command is deliberately same-pile: source commit blobs must already be
resident. It freezes the selected legacy head, validates the complete reachable
DAG, and converts each authored node into a native commit using its exact
`repo::content` and `metadata::archive` handles. A missing metadata archive maps
to the canonical empty archive. Contentless merge wrappers are validated but do
not become members.

`--namespace` contributes only to the target descriptor's identity. With no
further options the target uses `CollectionAdmission::Open`; the namespace key
does not implicitly authorize, admit, or identify the signing key.

To migrate into a capability-guarded target, name its independent trust root
and the signing key's exact local proof:

```text
trible pile migrate data.pile branch-to-collection \
  --branch legacy-events \
  --collection-name events \
  --namespace <64-hex-character-ed25519-public-key> \
  --authority <64-hex-character-ed25519-public-key> \
  --proof <64-hex-character-proof-id> \
  --signing-key ./writer.key
```

The command loads that one native proof by its BLAKE3 ID, loads only the claim
blobs named by the proof, and verifies its trust root, signer leaf,
`ACTION_WRITE`, exact target descriptor resource, and minimum Invoke mode at
one shared instant. It does not choose a path by scanning keys or claims.
`--proof` therefore requires `--authority`.

The one bootstrap case is explicit: when the migration signing key is itself
`--authority`, the proof may be omitted. The command issues one deterministic
root `WRITE`/Invoke claim and proof in memory, verifies it through the same
boundary, then stores the claim blob and native proof. A delegated signer must
supply a proof.

The complete source DAG and target admission are validated before any target
descriptor, dependency, claim blob, proof record, or commit is published. An
authorization failure therefore leaves the target untouched.

Legacy wrapper parents, messages, timestamps, authors, and signatures are not
silently reinterpreted as application metadata. Two source nodes with identical
data and semantic metadata map to one intrinsic native commit. Re-running with
the same collection identity and key is idempotent.

Migration is the only reason application-facing tooling needs to name a legacy
branch. New code publishes directly to collections.

## Operational invariants

- Persist dependencies before the record that makes them meaningful.
- Treat signed selected commits as mandatory ground truth; fail loud when their
  descriptor, data, or metadata is absent or invalid.
- Treat unsigned equations as optional, freshly validated cache evidence.
- Keep reach, trust, retention, and WANT policy orthogonal.
- Carry exact tickets across derivation boundaries instead of asking for an
  ambient “latest”.
- Flush at explicit application durability boundaries.
- Merge stores by union; never choose meaning from append order.

These rules are sufficient for both low-latency single-process use and sparse
distributed collection maintenance without introducing a second execution
model.
