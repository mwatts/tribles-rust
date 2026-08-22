# Repository Workflows

Working with a Tribles repository feels familiar to Git users, but the types
make data ownership and lifecycle explicit. Keep the following vocabulary in
mind when exploring the API:

* **Repository** – top-level object that tracks history through `BlobStore`
  and `BranchStore` implementations.
* **Workspace** – mutable view of a branch, similar to Git's working directory
  and index combined. Workspaces buffer commits and custom blobs until you push
  them back to the repository.
* **BlobStore** – storage backend for commits and payload blobs.
* **BranchStore** – records branch metadata and head pointers.

Both stores can be in memory, on disk or backed by a remote service. The
examples in `examples/repo.rs` and `examples/workspace.rs` showcase these APIs
and are a great place to start if you are comfortable with Git but new to
Tribles.

## Publishing an append-only collection

Applications that only need to publish independent facts do not need to mint a
branch or select a mutable head. `Collection<S>` is the narrow publication
facade for that case. It combines a storage backend, one canonical collection
descriptor, and a signing key; every call to `Collection::commit(Fragment)`
publishes one independent signed membership assertion:

```rust,ignore
use triblespace::prelude::{Collection, CollectionName, Fragment};

// `storage` implements BlobStorePut + CollectionStore.
// A root collection is anchored by its `name` within a `team`, named by that
// team's root public key. A single-node application is a team of one and
// says so by passing its own verifying key.
let name = CollectionName::new("models")?;
let team = signing_key.verifying_key();
let mut collection = Collection::new(storage, &name, team, signing_key);
let commit = collection.commit(fragment)?;
// Optional: choose an explicit durability boundary for one or many commits.
collection.flush()?;
let snapshot = collection.snapshot()?;
let facts = snapshot.facts();
let exact_commits = snapshot.commits();
let storage = collection.into_storage();
```

When only the source authority is needed, `Collection::ticket()` performs one
deterministic native-record discovery pass and returns the facade's exact,
strictly verified own commits without opening a blob reader or materializing
facts. Use `snapshot()` instead when facts, commits, and their validating reader
must come from one coherent observed prefix; separate `ticket()` and
`materialize()` calls are independent known-prefix observations.

A collection descriptor is an ordinary `TribleSet` stored as a canonical
`SimpleArchive`. There is no wrapper type: it is the facts of one `entity!`,
naming an **anchor** -- a **name** and a **team** on a root, or the **source**
collection a derivation is computed from -- together with its blob
**representation**, its algebraic **recipe**, and any arguments that recipe
takes. It also embeds the representation's and the recipe's own descriptions,
so the descriptor states what they are rather than only naming them. Its
32-byte content handle is the `CollectionHandle`, and that handle comes from
storing the blob rather than from hashing a descriptor nobody kept. `COMMIT`, `MERGE`, and `DERIVE`
are native typed algebra records rather than trible sets or blobs. Their exact
dense payloads are 192, 128, and 128 bytes respectively, and carry descriptor
handles directly, so any claim can resolve and verify its own collection
semantics through the ordinary blob store. There is no separate definition
record or registry whose synchronization could make an otherwise complete
claim ambiguous. `Collection::new` constructs the canonical
`SimpleArchive`-union descriptor for the supplied name and team. The descriptor
is the only collection-control structure represented as a `SimpleArchive`.

The fragment remains self-contained across the publication boundary: its facts
become the collection's canonical `SimpleArchive` data element, its metafacts
become the commit's canonical metadata archive, and attachments from its shared
blob store are copied alongside those two archives. Publication writes all
dependencies, including the descriptor blob, before inserting the signed
commit record, but deliberately performs no implicit durability flush. Call
`Collection::flush` at the cadence the application requires, or rely on the
storage's explicit close operation. Identical retries are idempotent, while
distinct commits coexist; there is no branch head, CAS retry, or implied
"latest" member.

When the backend also provides a blob reader with metadata lookup,
`Collection::materialize()` returns the complete known union of commits signed
by the facade's own key for this exact named collection. Commits signed by
other keys are not admitted. A failed signature authenticates none of its
record fields and is therefore an inert diagnostic, never an owner-attributed
veto. Every strictly verified own commit is ground truth: its descriptor, data
archive, and exact metadata archive must all validate or the read fails instead
of silently returning a partial set. Valid resident `MERGE` records may provide
a compact physical cover. Validation is the intersection of two reachability
walks: backwards from resident result identities (through nonresident
intermediates when needed), then forwards from authenticated leaves. Optional
result bytes are exact-checked against their computed canonical identity before
a tentative physical cover is accepted. Missing, corrupt, invalid, ungrounded,
or irrelevant
unsigned merge evidence is merely a cache miss, so it cannot erase committed
leaves or drive demand-born blob fetches.

This read boundary assumes that collection records have already passed the
deployment's admission policy. It bounds blob authority, not arbitrary CPU
work: each distinct admitted equation may still require canonical union
validation. A network service must therefore authenticate and bound record
admission (or retain immutable validation receipts) before placing untrusted
claims in its durable `CollectionStore`; otherwise both validation CPU and
temporary resource use scale with the admitted equation graph.

Materialization has the same observed-prefix concurrency contract as native
record listing. It first discovers one deterministic record view and then opens
a blob-reader snapshot. A concurrent commit not observed during that discovery
appears on a later call; all own commits that were observed are included or the
call returns an error. This is deliberately not a global "latest" transaction.
`Collection::snapshot()` carries the materialized facts, that exact authorized
commit set, and the reader used for validation together. Physically visible
blobs from a later commit remain inert unless its signed record belongs to the
returned commit set, so derived-index builders cannot accidentally pair an old
fact view with a newer source ticket. `Collection::materialize()` is the facts-
only projection and retains its empty-collection fast path without opening a
reader.

`Pile`, `MemoryRepo`, and the storage composition wrappers implement the native
`CollectionStore` surface. `ObjectStoreRemote` exposes the corresponding async
surface and can be used with `Collection` through `Blocking`. Under the
configured object-store prefix, each record is a create-only object at
`collection-records/<record-id>`. Its bytes are a one-byte, layout-versioned
variant tag followed by the exact dense typed payload. A future payload layout
receives a new tag rather than reinterpreting an existing one. Listing is an
observed monotone view rather than a global snapshot: a concurrent immutable
insert may appear on this list or the next, but every observed object is
decoded, has its record ID recomputed from its semantic kind and dense payload,
and is checked against the ID in its path. Descriptor blobs use the ordinary
blob namespace; `CollectionStore` contains only `COMMIT`, `MERGE`, and
`DERIVE`.

This collection path coexists with the branch-oriented `Repository` and
`Workspace` APIs documented below. Native collection records are not pin
updates; choosing the headless path does not change the semantics of existing
branches.

### Migrating one legacy branch directly

`trible pile migrate <pile> branch-to-collection` converts one legacy
`Repository` branch in place. It is deliberately a same-pile operation: the
source commit blobs must already be in the pile that will receive the native
records. The collection identity and authority policy are never inferred from
the branch wrapper. Supply all three explicitly:

```text
trible pile migrate data.pile branch-to-collection \
  --branch legacy-events \
  --collection-name events \
  --team-root <64-hex-character-ed25519-public-key> \
  --signing-key ./writer.key
```

The command resolves `--branch` as an active 32-character hex branch id, or as
an exact branch name when no such id is active. Both current UTF8String names
and the older inline ShortString names are understood; duplicate names are
rejected instead of guessed. Before its first append, it freezes the selected
pin head and validates the complete reachable commit DAG through one later
append-only blob snapshot, including canonical wrapper/content/archive bytes,
authored content signatures, parent availability, and the current or
historical intrinsic identity of contentless canonical merges.
Older authored commits may have random wrapper subjects; a unique subject and
unique mapped `content` and `metadata::archive` fields are required, but
today's intrinsic subject derivation is not retroactively imposed on them.
Discarded wrapper annotations do not become migration preconditions.

Each authored node becomes `COMMIT(collection, repo::content,
metadata::archive)`, re-signed by the supplied target key. An absent
`metadata::archive` becomes the canonical empty archive. Repository wrapper
parents, timestamps, messages, authors and signatures do not become collection
metadata; contentless merge nodes are validated ancestry and are skipped. It
follows that two source nodes with the same exact content and semantic metadata
map to one native commit. The printed mapping makes that many-to-one collapse
visible. Re-running with the same collection identity and signing key is
idempotent: content-addressed dependencies and already-present native records
are no-ops.

The generic migration validates the two archive roots and their authentication,
not application-specific meanings of 32-byte values inside their facts. It
copies nothing because source and target are the same pile: any resident
attachment closure remains resident and becomes reachable through the native
commit, while a faculty that requires particular attachments must validate
that schema before migration.

## Attach an exact direct fact view

Read-only consumers that already hold an immutable commit ticket do not need
the publishing key. `SimpleArchiveCollection` fixes the same canonical
descriptor from the collection's name and team, and accepts the complete signed
records as a set:

```rust,ignore
use triblespace::prelude::{CollectionName, SimpleArchiveCollection};

let models = SimpleArchiveCollection::new(CollectionName::new("models")?, team);
let facts = models.attach_exact(&mut storage, &ticket)?;
let snapshot = models.snapshot_exact(&mut storage, &ticket)?;
```

Ticket members may have different authors. Byte-identical repeats collapse,
and the snapshot returns the canonical intrinsic-id-sorted unique commit set.
Each member must name this descriptor, byte-match a strictly verified stored
record, and have resident valid descriptor, data, and mandatory metadata
blobs. Commits present in storage but omitted from the ticket remain inert.
Same-descriptor `MERGE` evidence may supply a smaller exact physical cover but
cannot widen the authorized union. `attach_exact` and `snapshot_exact` never
write; the latter carries the one reader used to validate and materialize the
selected record observation.

## Attach an exact derived query view

Derived query artifacts can use the signed source commits themselves as a
frozen authority ticket. The native raw-Succinct facade has no repository or
pin dependency:

```rust,ignore
use triblespace::core::collection::succinctarchive_union::SuccinctArchiveCollection;

let succinct = SuccinctArchiveCollection::new(name.clone(), team);

// `ticket` is the byte-identical set of signed SimpleArchive commits selected
// by the caller. Completion publishes only reproducible unsigned evidence.
let archive = succinct.ensure_exact(&mut storage, &ticket)?;

// A read-only consumer can instead require a complete resident cover.
let same_archive = succinct.attach_exact(&mut storage, &ticket)?;

// Explicit maintenance compacts colliding canonical byte-size tiers. It is
// never scheduled by attachment and still uses the same frozen ticket.
let compact_archive = succinct.compact_exact(&mut storage, &ticket)?;
```

All three methods discover only the commits named by the ticket, verify their
signatures and exact stored bytes, and admit unsigned source merges, target
merges, and derivations only when their canonical equations validate. The
returned `UnionArchive` preserves the deterministic physical shard cover.
`ensure_exact` first probes for a complete cover, derives only the necessary
resident source members, publishes descriptors before all output blobs and all
outputs before `DERIVE` records, drops the old reader before writing, and then
re-admits through a fresh reader. It neither signs a new root nor flushes the
store. An empty ticket performs no storage I/O and receives one process-local
empty query shard; a signed commit whose source data happens to be empty still
has ordinary persisted derivation provenance. If one selected source upper is
too large for the fixed target representation, ensuring excludes that physical
member and globally recomputes the source cover under the same reader and
resolved semantics. Successful images are reused across replans, while only
the final feasible cover is published. A signed source leaf that has no finer
representable cover returns an explicit unrepresentable-cover error and writes
nothing.

Once that raw cover is fixed, the facade attaches one source-bound Rank9
sidecar per selected member. These fibers live in a separate collection
descriptor whose recipe pins the raw/sidecar format, builder version, native
pointer width, and byte order. The four version-1 recipe ids distinguish
32/64-bit and little/big-endian builds; the active compilation target selects
one. This target collection is the lifted image `i(a)` of the raw lattice, with
`i(a) join i(b) = i(a join b)`. An ordinary raw-to-Rank9 `DERIVE` therefore
states the true homomorphism, even though this lifecycle never materializes a
Rank9 `MERGE`.

`attach_exact` treats those records and blobs as optional cache evidence. It
scans the ledger once, accepts a unique claimed output only after fresh hashes,
the embedded raw-source handle, and `SuccinctArchive::from_blob_pair` all
agree, and falls back to a transient Rank9 rebuild for missing, corrupt,
mismatched, or ambiguous evidence without writing. `ensure_exact` computes the
one canonical sidecar for each incomplete raw member, puts both descriptors and
all missing sidecars before new claims, then opens a fresh reader and strictly
checks the exact expected raw endpoints, sidecars, source headers, and
`DERIVE`s. A complete probe writes nothing. If a canonical claim survives but
its endpoint does not, repair stores the endpoint without appending a duplicate
claim. Persisted Rank9 validation is linear and does not build another Rank9
index; reconstructing a query runtime still allocates its runtime arena and
views, while transient fallback additionally allocates and builds the missing
index bytes.

`compact_exact` first performs ordinary exact completion, then groups the
admitted canonical raw target members into fixed dyadic serialized-byte tiers.
It repeatedly joins the two lowest content handles in the lowest colliding
tier. If that fixed representation cannot encode a pair, the lower member is
retired for the round while the higher member remains eligible for the next
deterministic pair. Every attempt therefore shrinks the active set, and a
capacity-stable result may deliberately retain members in the same tier. A
round is staged completely before publication: if no join succeeds, it writes
nothing; otherwise the target descriptor and every canonical result blob are
put before any topologically ordered unsigned `MERGE` record. No flush,
manifest, receipt, level record, retention root, or new authority is implied.
A fresh exact attachment checks each published round under the same ticket.
For any observed cover, repeated or concurrent calls choose the same
content-addressed results and intrinsic record IDs, making retries idempotent.

Unsigned equations remain reproducible cache evidence rather than durable
receipts or authority. Their intermediate blobs need not remain resident:
attachment walks backwards from resident source and target results, then
reconstructs the finite candidate graph forwards from authenticated source
leaves. Canonical intermediates live only in use-counted scratch. The selected
physical artifacts are freshly hashed and representation-validated; an invalid
optional upper artifact is removed and the cover is recomputed from valid lower
members. This reconstruction writes no blobs or records and requires no new
retention roots.

The facade never schedules compaction in the background. Rank9 fibers add no
signed root, receipt, manifest, retention root, durability flush, policy knob,
or target-side compactor; explicit compaction first selects the raw cover and
then builds fibers only for those selected members. The raw format's per-shard
`u32::MAX` row and domain limits still apply. Capacity is a typed construction
outcome: source completion falls back to finer exact members, while target
maintenance returns a deterministic capacity-stable colliding cover when no
further representable join exists. Persisted malformed or noncanonical bytes
remain fatal and never acquire this fallback meaning.

## Opening a repository

Repositories are constructed from any storage that implements the appropriate
traits. The choice largely depends on your deployment scenario:

1. Pick or compose a storage backend (see [Storage Backends and
   Composition](#storage-backends-and-composition)).
2. Create a signing key for the identity that will author commits.
3. Call `Repository::new(storage, signing_key, commit_metadata)` to obtain a handle.
   Pass `TribleSet::new()` for `commit_metadata` when you do not need custom
   metadata on commits.

Most applications perform the above steps once during start-up and then reuse
the resulting `Repository`. If initialization may fail (for example when opening
an on-disk pile), bubble the error to the caller so the process can retry or
surface a helpful message to operators.

## Storage Backends and Composition

`Repository` accepts any storage that implements both the `BlobStore` and
`BranchStore` traits, so you can combine backends to fit your deployment. The
crate ships with a few ready-made options:

- [`MemoryRepo`](../src/repo/memoryrepo.rs) stores everything in memory and is
  ideal for tests or short-lived tooling where persistence is optional.
- [`Pile`](../src/repo/pile.rs) persists blobs and branch metadata in a single
  append-only file. It is the default choice for durable local repositories and
  integrates with the pile tooling described in [Pile Format](pile-format.md).
- [`ObjectStoreRemote`](../src/repo/objectstore.rs) connects to
  [`object_store`](https://docs.rs/object_store/latest/object_store/) endpoints
  (S3, local filesystems, etc.). It keeps all repository data in the remote
  service and is useful when you want a shared blob store without running a
  dedicated server.
- [`HybridStore`](../src/repo/hybridstore.rs) lets you split responsibilities,
  e.g. storing blobs on disk while keeping branch heads in memory or another
  backend. Any combination that satisfies the trait bounds works.

Backends that need explicit shutdown can implement `StorageClose`. When the
repository type exposes that trait bound you can call `repo.close()?` to flush
and release resources instead of relying on `Drop` to run at an unknown time.
This is especially handy for automation where the process may terminate soon
after completing a task.

```rust,ignore
use triblespace::core::repo::hybridstore::HybridStore;
use triblespace::core::repo::memoryrepo::MemoryRepo;
use triblespace::core::repo::objectstore::ObjectStoreRemote;
use triblespace::core::repo::Repository;
use triblespace::core::inline::encodings::hash::Blake3;
use url::Url;

let blob_remote: ObjectStoreRemote<Blake3> =
    ObjectStoreRemote::with_url(&Url::parse("s3://bucket/prefix")?)?;
let branch_store = MemoryRepo::default();
let storage = HybridStore::new(blob_remote, branch_store);
let mut repo = Repository::new(storage, signing_key, TribleSet::new())?;

// Work with repo as usual …
// repo.close()?; // if the underlying storage supports StorageClose
```

## Branching

A branch records a line of history and carries the metadata that identifies who
controls updates to that history. Creating one writes initial metadata to the
underlying store and returns an [`ExclusiveId`](../src/id.rs) guarding the
branch head. Dereference that ID when you need a plain [`Id`](../src/id.rs) for
queries or workspace operations.

Typical steps for working on a branch look like:

1. Create a repository backed by blob and branch stores via `Repository::new`.
2. Initialize or look up a branch ID with helpers like
   `Repository::create_branch`. When interacting with an existing branch call
   `Repository::pull` directly.
3. Commit changes in the workspace using `Workspace::commit`.
4. Push the workspace with `Repository::push` (or handle conflicts manually via
   `Repository::try_push`) to publish those commits.

The example below demonstrates bootstrapping a new branch and opening multiple
workspaces on it. Each workspace holds its own staging area, so remember to push
before sharing work or starting another task.


```rust,ignore
let mut repo = Repository::new(pile, SigningKey::generate(&mut OsRng), TribleSet::new())?;
let branch_id = repo.create_branch("main", None).expect("create branch");

let mut ws = repo.pull(*branch_id).expect("pull branch");
let mut ws2 = repo.pull(ws.branch_id()).expect("open branch");
```

After committing changes you can push the workspace back. `push` will retry on
contention and attempt to merge, while `try_push` performs a single attempt and
returns `Ok(Some(conflict_ws))` when the branch head moved. Choose the latter
when you need explicit conflict handling:

```rust,ignore
ws.commit(change, "initial commit");
repo.push(&mut ws)?;
```

### Managing signing identities

The key passed to `Repository::new` becomes the default signing identity for
branch metadata and commits. Collaborative projects often need to switch
between multiple authors or assign a dedicated key to automation. You can
adjust the active identity in three ways:

* `Repository::set_signing_key` replaces the repository's default key. Subsequent
  calls to helpers such as `Repository::create_branch` or `Repository::pull` use the new
  key for any commits created from those workspaces.
* `Repository::create_branch_with_key` signs a branch's metadata with an explicit
  key, allowing each branch to advertise the author responsible for updating it.
* `Repository::pull_with_key` opens a workspace that will sign its future commits
  with the provided key, regardless of the repository default.

The snippet below demonstrates giving an automation bot its own identity while
letting a human collaborator keep theirs:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::repo::Repository;

let alice = SigningKey::generate(&mut OsRng);
let automation = SigningKey::generate(&mut OsRng);

// Assume `pile` was opened earlier, e.g. via `Pile::open` as shown in previous sections.
let mut repo = Repository::new(pile, alice.clone(), TribleSet::new())?;

// Create a dedicated branch for the automation pipeline using its key.
let automation_branch = repo
    .create_branch_with_key("automation", None, automation.clone())?
    .release();

// Point automation jobs at their dedicated identity by default.
repo.set_signing_key(automation.clone());
let mut bot_ws = repo.pull(automation_branch)?;

// Humans can opt into their own signing identity even while automation remains
// the repository default.
let mut human_ws = repo.pull_with_key(automation_branch, alice.clone())?;
```

`human_ws` and `bot_ws` now operate on the same branch but will sign their
commits with different keys. This pattern is useful when rotating credentials or
running scheduled jobs under a service identity while preserving authorship in
the history. You can swap identities at any time; existing workspaces keep the
key they were created with until you explicitly call
`Repository::set_signing_key`.

## Inspecting History

You can explore previous commits using `Workspace::checkout` which returns a
`Checkout` (which derefs to `TribleSet` and also tracks the `CommitSet`) with the
union of the specified commit contents. Passing a single
commit returns just that commit. To include its history you can use the
`ancestors` helper. Commit ranges are supported for convenience. The expression
`a..b` yields every commit reachable from `b` that is not reachable from `a`,
treating missing endpoints as empty (`..b`) or the current `HEAD` (`a..` and
`..`). These selectors compose with filters, so you can slice history to only
the entities you care about.

```rust,ignore
let history = ws.checkout(commit_a..commit_b)?;
let full = ws.checkout(ancestors(commit_b))?;
```

The [`history_of`](../src/repo.rs) helper builds on the `filter` selector to
retrieve only the commits affecting a specific entity. Commit selectors are
covered in more detail in the next chapter:

```rust,ignore
let entity_changes = ws.checkout(history_of(my_entity))?;
```

## Working with Custom Blobs

Workspaces keep a private blob store that mirrors the repository's backing
store. This makes it easy to stage large payloads alongside the trible sets you
plan to commit. The [`Workspace::put`](../src/repo.rs) helper stores any type
implementing [`ToBlob`](triblespace::core::blob::ToBlob) and returns a typed handle you can
embed like any other value. Handles are `Copy`, so you can commit them and reuse
them to fetch the blob later.

The example below stages a quote and an archived `TribleSet`, commits both, then
retrieves them again with strongly typed and raw views. In practice you might
use this pattern to attach schema migrations, binary artifacts, or other payloads
that should travel with the commit:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::blob::Blob;
use triblespace::core::examples::{self, literature};
use triblespace::prelude::*;
use triblespace::core::repo::{self, memoryrepo::MemoryRepo, Repository};
use blobencodings::{UTF8String, SimpleArchive};

let storage = MemoryRepo::default();
let mut repo = Repository::new(storage, SigningKey::generate(&mut OsRng), TribleSet::new())?;
let branch_id = repo.create_branch("main", None).expect("create branch");
let mut ws = repo.pull(*branch_id).expect("pull branch");

// `entity!{}` auto-puts blob payloads into the workspace's blob
// store — the value side of a `Handle<S>`-typed field becomes the
// content-addressed handle that lives in the trible.
//
// When you also need the handle in hand (to read back, log, share,
// or reuse across multiple entities), call `ws.put` explicitly.
let quote_handle: Inline<Handle<UTF8String>> =
    ws.put("Fear is the mind-killer".to_owned());
let archive_handle: Inline<Handle<SimpleArchive>> =
    ws.put(&examples::dataset());

let mut change = entity! {
    literature::title: "Dune (annotated)",
    literature::quote: quote_handle.clone(),
};
change += entity! { repo::content: archive_handle.clone() };

ws.commit(change, "Attach annotated dataset");
// Single-attempt push. Use `push` to let the repository merge and retry automatically.
repo.try_push(&mut ws).expect("try_push");

// Fetch the staged blobs back with the desired representation.
let restored_quote: String = ws
    .get(quote_handle)
    .expect("load quote");
let restored_set: TribleSet = ws
    .get(archive_handle)
    .expect("load dataset");
let archive_bytes: Blob<SimpleArchive> = ws
    .get(archive_handle)
    .expect("load raw blob");
std::fs::write("dataset.car", archive_bytes.bytes.as_ref()).expect("persist archive");
```

Rust infers the blob encoding for both `put` and `get` from the handles and the
assignment context, so the calls stay concise without explicit turbofish
annotations.

Blobs staged this way stay local to the workspace until you push the commit.
`Workspace::get` searches the workspace-local store first and falls back to the
repository if necessary, so the handles remain valid after you publish the
commit. This round trip lets you persist logs, archives, or other auxiliary
files next to your structured data without inventing a separate storage
channel.

## Merging and Conflict Handling

When pushing a workspace another client might have already updated the branch.
There are two ways to handle this:

- `Repository::try_push` — a single-attempt push that uploads local blobs and
  attempts a CAS update once. If the branch advanced concurrently it returns
  `Ok(Some(conflict_ws))` so callers can merge and retry explicitly:

```rust,ignore
ws.commit(content, "codex-turn");
let mut current_ws = ws;
while let Some(mut incoming) = repo.try_push(&mut current_ws)? {
    // Merge the local staged changes into the incoming workspace and retry.
    incoming.merge(&mut current_ws)?;
    current_ws = incoming;
}
```

- `Repository::push` — a convenience wrapper that performs the merge-and-retry
  loop for you. Call this when you prefer the repository to handle conflicts
  automatically; it either succeeds (returns `Ok(())`) or returns an error.

```rust,ignore
ws.commit(content, "codex-turn");
repo.push(&mut ws)?; // will internally merge and retry until success
```

> **Troubleshooting:** `Workspace::merge` succeeds only when both workspaces
> share a blob store. Merging a workspace pulled from a different pile or
> remote returns `MergeError::DifferentRepos`. Decide which repository will own
> the combined history, transfer the other branch's reachable blobs into it with
> `repo::transfer(reachable(...))`, create a branch for that imported head, and
> merge locally once both workspaces target the same store.

After a successful push the branch may have advanced further than the head
supplied, because the repository refreshes its view after releasing the lock.
An error indicating a corrupted pile does not necessarily mean the push failed;
the update might have been written before the corruption occurred.

This snippet is taken from [`examples/workspace.rs`](../examples/workspace.rs).
The [`examples/repo.rs`](../examples/repo.rs) example demonstrates the same
pattern with two separate workspaces. The returned `Workspace` already contains
the remote commits, so after merging your changes you push that new workspace to
continue.

## Typical CLI Usage

There is a small command line front-end in the
[`trible`](https://github.com/triblespace/trible) repository. It exposes push
and merge operations over simple commands and follows the same API presented in
the examples. The tool is currently experimental and may lag behind the library,
but it demonstrates how repository operations map onto a CLI.

## Diagram

A simplified view of the push/merge cycle:

```text

        ┌───────────┐         pull          ┌───────────┐
        | local ws  |◀───────────────────── |   repo    |
        └─────┬─────┘                       └───────────┘
              │
              │ commit
              │                                                                      
              ▼                                   
        ┌───────────┐         push          ┌───────────┐
        │  local ws │ ─────────────────────▶│   repo    │
        └─────┬─────┘                       └─────┬─────┘
              │                                   │
              │ merge                             │ conflict?
              └──────▶┌─────────────┐◀────────────┘
                      │ conflict ws │       
                      └───────┬─────┘
                              │             ┌───────────┐
                              └────────────▶|   repo    │
                                     push   └───────────┘
   
```

Each push either succeeds or returns a workspace containing the other changes.
Merging incorporates your commits and the process repeats until no conflicts
remain.

### Troubleshooting push, branch, and pull failures

`Repository::push`, `Repository::create_branch`, and `Repository::pull` surface
errors from the underlying blob and branch stores. These APIs intentionally do
not hide storage issues, because diagnosing an I/O failure or a corrupt commit
usually requires operator intervention. The table below lists the error variants
along with common causes and remediation steps.

| API | Error variant | Likely causes and guidance |
| --- | --- | --- |
| `Repository::push` | `PushError::StorageBranches` | Enumerating branch metadata in the backing store failed. Check connectivity and credentials for the branch store (for example, the object-store bucket, filesystem directory, or HTTP endpoint). |
| `Repository::push` | `PushError::StorageReader` | Creating a blob reader failed before any transfer started. The blob store may be offline, misconfigured, or returning permission errors. |
| `Repository::push` | `PushError::StorageGet` | Fetching existing commit metadata failed. The underlying store returned an error or the metadata blob could not be decoded, which often signals corruption or truncated uploads. Inspect the referenced blob in the store to confirm it exists and is readable. |
| `Repository::push` | `PushError::StoragePut` | Uploading new content or metadata blobs failed. Look for transient network failures, insufficient space, or rejected writes in the blob store logs. On local `Pile` stores backed by `writev`, very large single records can fail with `EINVAL` (for example when total iovec bytes exceed platform syscall limits). Split oversized payloads into semantic chunks (with a manifest/root record) before retrying. |
| `Repository::push` | `PushError::BranchUpdate` | Updating the branch head failed. Many backends implement optimistic compare-and-swap semantics; stale heads or concurrent writers therefore surface here as update errors. Refresh the workspace and retry after resolving any store-side errors. |
| `Repository::push` | `PushError::BadBranchMetadata` | The branch metadata could not be parsed. Inspect the stored metadata blobs for corruption or manual edits and repair them before retrying the push. |
| Branch creation APIs | `BranchError::StorageReader` | Creating a blob reader failed. Treat this like `PushError::StorageReader`: verify the blob store connectivity and credentials. |
| Branch creation APIs | `BranchError::StorageGet` | Reading branch metadata during initialization failed. Check for corrupted metadata blobs or connectivity problems. |
| Branch creation APIs | `BranchError::StoragePut` | Persisting branch metadata failed. Inspect store logs for rejected writes or quota issues. |
| Branch creation APIs | `BranchError::BranchHead` | Retrieving the current head of the branch failed. This usually points to an unavailable branch store or inconsistent metadata. |
| Branch creation APIs | `BranchError::BranchUpdate` | Updating the branch entry failed. Resolve branch-store errors and ensure no other writers are racing the update before retrying. |
| Branch creation APIs | `BranchError::AlreadyExists` | A branch with the requested name already exists. Choose a different name or delete the existing branch before recreating it. |
| Branch creation APIs | `BranchError::BranchNotFound` | The specified base branch does not exist. Verify the branch identifier and that the base branch has not been deleted. |
| `Repository::pull` | `PullError::BranchNotFound` | The branch is missing from the repository. Check the branch name/ID and confirm that it has not been removed. |
| `Repository::pull` | `PullError::BranchStorage` | Accessing the branch store failed. This mirrors `BranchError::BranchHead` and usually indicates an unavailable or misconfigured backend. |
| `Repository::pull` | `PullError::BlobReader` | Creating a blob reader failed before commits could be fetched. Ensure the blob store is reachable and that the credentials grant read access. |
| `Repository::pull` | `PullError::BlobStorage` | Reading commit or metadata blobs failed. Investigate missing objects, network failures, or permission problems in the blob store. |
| `Repository::pull` | `PullError::BadBranchMetadata` | The branch metadata is malformed. Inspect and repair the stored metadata before retrying the pull. |

## Remote Stores

Remote deployments use the [`ObjectStoreRemote`](../src/repo/objectstore.rs)
backend to speak to any service supported by the
[`object_store`](https://docs.rs/object_store/latest/object_store/) crate (S3,
Google Cloud Storage, Azure Blob Storage, HTTP-backed stores, the local
filesystem, and the in-memory `memory:///` adapter). `ObjectStoreRemote`
implements both `BlobStore` and `BranchStore`, so the rest of the repository API
continues to work unchanged – the only difference is the URL you pass to
`with_url`.

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::prelude::*;
use triblespace::core::repo::objectstore::ObjectStoreRemote;
use triblespace::core::repo::Repository;
use triblespace::core::inline::encodings::hash::Blake3;
use url::Url;

fn open_remote_repo(raw_url: &str) -> anyhow::Result<()> {
    let url = Url::parse(raw_url)?;
    let storage = ObjectStoreRemote::<Blake3>::with_url(&url)?;
    let mut repo = Repository::new(storage, SigningKey::generate(&mut OsRng), TribleSet::new())?;

    let branch_id = repo.create_branch("main", None)?;
    let mut ws = repo.pull(*branch_id)?;
    ws.commit(TribleSet::new(), "initial commit");

    while let Some(mut incoming) = repo.try_push(&mut ws)? {
        incoming.merge(&mut ws)?;
        ws = incoming;
    }

    Ok(())
}
```

`ObjectStoreRemote` writes directly through to the backing service. It
implements `StorageClose`, but the implementation is a no-op, so dropping the
repository handle is usually sufficient. Call `repo.close()` if you prefer an
explicit shutdown step.

Credential configuration follows the `object_store` backend you select. For
example, S3 endpoints consume AWS access keys or IAM roles, while
`memory:///foo` provides a purely in-memory store for local testing. Once the
URL resolves, repositories backed by piles and remote stores share the same
workflow APIs.

## Attaching a Foreign History (merge-import)

Sometimes you want to graft an existing branch from another pile into your
current repository without rewriting its commits. Tribles supports a
conservative, schema‑agnostic import followed by a single merge commit:

1. Copy all reachable blobs from the source branch head into the target pile
   by streaming the `reachable` walker into `repo::transfer`. The traversal
   scans every 32‑byte aligned chunk and enqueues any candidate that
   dereferences in the source.
2. Create a single merge commit that has two parents: your current branch head
   and the imported head. No content is attached to the merge; it simply ties
   the DAGs together.

This yields a faithful attachment of the foreign history — commits and their
content are copied verbatim, and a one‑off merge connects both histories.

The `trible` CLI exposes this as:

```sh
trible branch merge-import \
  --from-pile /path/to/src.pile --from-name source-branch \
  --to-pile   /path/to/dst.pile --to-name   self
```

Internally this uses the `reachable` walker in combination with
`repo::transfer` plus `Workspace::merge_commit`. Because the traversal scans
aligned 32‑byte chunks, it is forward‑compatible with new formats as long as
embedded handles remain 32‑aligned.

> **Sidebar — Choosing a copy routine**
> - `repo::transfer` pairs the reachability walker (or any other iterator you
>   provide) with targeted copies, returning `(old_handle, new_handle)` pairs
>   for the supplied handles. Feed it the `reachable` iterator when you only
>   want live blobs, the output of
>   [`potential_handles`](https://docs.rs/triblespace/latest/triblespace/repo/fn.potential_handles.html)
>   when scanning metadata, or the `.handle` values projected from
>   `BlobInfo` items returned by `BlobStoreList::blobs()` when duplicating an
>   entire store.
> - `MemoryBlobStore::keep` (and other `BlobStoreKeep` implementations) retain
>   whichever handles you stream to them, making it easy to drop unreachable
>   blobs once you've walked your roots.
>
> Reachable copy keeps imports minimal; the transfer helper lets you rewrite
> specific handles while duplicating data into another store.

### Programmatic example (Rust)

The same flow can be used directly from Rust when you have two piles on disk and
want to attach the history of one branch to another:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::prelude::*;
use triblespace::core::repo::{self, pile::Pile, Repository};
use triblespace::core::inline::encodings::hash::Blake3;
use triblespace::core::inline::encodings::hash::Handle;

fn merge_import_example(
    src_path: &std::path::Path,
    src_branch_id: triblespace::id::Id,
    dst_path: &std::path::Path,
    dst_branch_id: triblespace::id::Id,
) -> anyhow::Result<()> {
    // 1) Open source (read) and destination (write) piles. `refresh`
    //    loads the existing records and fails loud on a corrupt tail
    //    (repair is a separate, explicit step: `Pile::amputate` /
    //    `trible pile amputate`).
    let mut src = Pile::open(src_path)?;
    src.refresh()?;
    let mut dst = Pile::open(dst_path)?;
    dst.refresh()?;

    // 2) Resolve source head commit handle
    let src_head: Inline<Handle<blobencodings::SimpleArchive>> =
        src.head(src_branch_id)?.ok_or_else(|| anyhow::anyhow!("source head not found"))?;

    // 3) Conservatively copy all reachable blobs from source → destination
    let reader = src.reader()?;
    let mapping: Vec<_> = repo::transfer(
        &reader,
        &mut dst,
        repo::reachable(&reader, [src_head.transmute()]),
    )
    .collect::<Result<_, _>>()?;
    eprintln!("copied {} reachable blobs", mapping.len());

    // 4) Attach via a single merge commit in the destination branch
    let mut repo = Repository::new(dst, SigningKey::generate(&mut OsRng), TribleSet::new())?;
    let mut ws = repo.pull(dst_branch_id)?;
    ws.merge_commit(src_head)?; // parents = { current HEAD, src_head }

    // 5) Push with standard conflict resolution
    while let Some(mut incoming) = repo.try_push(&mut ws)? {
        incoming.merge(&mut ws)?;
        ws = incoming;
    }

    drop(ws);
    repo.close()?;
    drop(reader);
    src.close()?;
    Ok(())
}
```

## Optional telemetry sink

The facade crate exposes an optional `telemetry` feature that turns `tracing`
spans into TribleSpace commits. This is useful for profiling services, import
pipelines, or long-running agents while keeping telemetry noise in a dedicated
pile.

```rust,ignore
use triblespace::telemetry::Telemetry;

let _guard = Telemetry::install_global_from_env("archive import");
```

Set `TELEMETRY_PILE` and a `TELEMETRY_COLLECTION_NAME` — lowercase ASCII
letters, digits and `-`, starting with a letter, at most 32 bytes — to enable
the sink. The sink generates its own key per session, so its collection is a
team of one rooted at that key. Every flushed batch becomes
an independent signed collection commit carrying its telemetry schema as
metafacts; no mutable branch head or compare-and-set retry is involved. You can
tune batching via `TELEMETRY_FLUSH_MS`.
