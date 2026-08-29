# Glossary

This chapter collects the core terms that appear throughout the book. Skim it
when you encounter unfamiliar terminology or need a refresher on how concepts
relate to one another in TribleSpace.

### Action
An uninterpreted 128-bit identifier naming one exact operation. Actions do not
form a hierarchy and never imply one another. For example, `ACTION_WRITE` and
`ACTION_CONNECT` are separate atoms even when they concern collections
associated with the same team.

### Attribute
A property that describes some aspect of an entity. Attributes occupy the
middle position in a trible and carry the `InlineEncoding` (or blob-handle
encoding) that interprets and validates the value. Modules mint them with the
`attributes!` macro, so they behave like detached struct fields: each attribute
remains independently typed even when many are combined to describe the same
entity, preserving its individual semantics. Provide an explicit 128-bit id in
the macro when you need a canonical column shared across crates or languages;
omit the literal to derive a deterministic id from the attribute name and
encoding (the macro wraps the name + encoding id in an `entity!{}` fragment and
takes the root for you), which is handy for short-lived or internal attributes.

### Blob
An immutable chunk of binary data addressed by the hash of its contents. Blobs
store payloads that do not fit in the fixed 32-byte value slot—long strings,
media assets, archived `TribleSet`s, commit metadata, and other large
artifacts. Each blob is tagged with a `BlobEncoding` so applications can decode it
back into native types.

### Blob Store
An abstraction that persists immutable content-addressed blobs. Implementations
back local piles, in-memory collections, or remote object stores while exposing
small capability traits for insertion, retrieval, metadata, and enumeration.

### Capability Claim
A closed canonical, keyless `SimpleArchive` naming one exact action/resource
atom, invoke/delegate mode, optional parent **claim** handle, and optional
inclusive validity interval. A root claim has no parent; each later claim names
its exact semantic predecessor. Principal keys and signatures live in the
native proof, not in the claim.

### Capability Proof
A canonical native `K0 (S C K)+` byte string. Each strict Ed25519 signature
binds its issuer key, exact claim handle, and delegate key. Its BLAKE3 digest is
the proof ID used for exact physical lookup. Verification also receives the
ordered claim blobs, external trust root, expected leaf, explicit instant, and
exact request; authority is the meet of the claims, never a consequence of
proof presence.

### Capability Proof Bundle
A complete portable capability proof plus the exact claim blobs it names in
root-to-leaf order. Invites, CONNECT, and SYNC_TEAM authorization carry this
bounded self-contained form, so verification needs no pre-auth fetch or
ambient lookup.

### Capability Proof Store
A grow-only native set of canonical capability proofs. It supports
deterministic enumeration and exact lookup by proof ID, but no discovery by key
or claim. Storing a proof preserves evidence and can root resident claim blobs;
it does not make the proof authorized.

### Commit
A signed native collection membership assertion. A `CollectionCommit` names
the exact collection descriptor, data element, mandatory metadata archive, and
author. Its intrinsic record ID is derived from the canonical 192-byte payload.
Commits are independent leaves rather than snapshots in a parent chain.

### Capability Presentation
One owned `CapabilityProofBundle` paired with the exact leaf key the caller
expects it to establish. The expectation prevents a valid prefix or proof for
another principal from silently becoming an admission decision.

### Collection
A self-describing grow-only join semilattice. Signed commits introduce members;
validated merge records describe joins within the lattice; derivation records
map elements into another collection through a canonical homomorphism. A
collection has no distinguished head.

### Cover
One exact point in a collection lattice, represented by the collection
descriptor identity and a PATCH set of distinct payload handles. Signatures,
authors, and metadata are optional provenance fibers queryable from the store,
not part of cover identity or required for replay, so several claims over
identical data collapse to one member. Distinct covers may have the same
support: a validated merge can prove that `{a, b}` and `{a⊔b}` denote the same
join. Cover construction is opaque;
admission and validated collection algebra produce them rather than accepting
caller-forged hash sets.

### Collection Admission
The read-time signer decision performed by `store.cover` and
`store.snapshot`. The descriptor authority is admitted directly; each
additional expected leaf must carry an explicit proof bundle which verifies at
one clock instant against that authority and the exact
`ACTION_WRITE`/collection atom. An empty presentation set therefore admits the
descriptor authority alone.

### Collection Descriptor
A canonical `SimpleArchive` describing a collection's UTF-8 root name or exact
derived source, mandatory descriptor-local authority, element representation,
join recipe, and reach law. Its content handle is the `CollectionHandle`, so
every native record which names a collection can resolve its meaning through
the ordinary blob store. A derived descriptor states its own authority and
never inherits one through its source.

### Collection Store
A grow-only set of native `COMMIT`, `MERGE`, and `DERIVE` records. Insertion is
idempotent by intrinsic record ID; combining two stores is set union.

### Collection Snapshot
One coherent known-prefix observation containing materialized facts, the exact
payload `Cover` which names them, and the blob reader which validated their
dependencies.

### CONNECT
The exact `ACTION_CONNECT` atom used by `triblespace-net` to authenticate a
direct-RPC session. Its resource is the team's exact 32-byte trust-root public
key, and its claimed subject must equal the transport peer key. CONNECT grants
no WRITE, generic READ, inventory disclosure, collection reach, semantic
trust, or retention authority. Protocol v12 exchanges complete subject-bound
proof bundles on the connection's first `OP_AUTH` stream. A separate reciprocal
SYNC_TEAM exchange for the same team and endpoints must authorize inventory and
blob reads.

### Constraint
The trait that every query operator implements. Its methods—`variables`,
`estimate`, `propose`, `confirm`, `satisfied`, and `influence`—let the Atreides
solver navigate the search space without a separate planner. `propose` and
`confirm` take a *frontier*: a whole batch of parent bindings, of which a
single binding is the width-1 case. Constraints are stateless: every method
receives the bindings it needs as a parameter, so the engine can backtrack,
batch, and split without telling anyone.
Estimates guide variable ordering and never change results; `confirm` may only
kill candidates, never add or revive them. Custom data sources and application
predicates participate in queries by implementing this trait.

### Entity
The first position in a trible. Entities identify the subject making a
statement and group the attributes asserted about it. They are represented by
stable identifiers so multiple facts about the same subject cohere.

In practice you pick an identifier policy:
- **Extrinsic ids** (for example `ufoid`, `fucid`, `genid`) track a conceptual
  subject across edits and versions. Use these when you intend to accumulate
  additional facts over time.
- **Intrinsic ids** (content-derived hashes) are recomputed from the entity's
  asserted fields. The `entity!` macro uses this policy when you omit the
  explicit `id @` prefix (or when you write `_ @`), so identical records unify
  naturally.

Ownership policies and schemas determine who may mint new facts for a given
identifier.

### Fragment
A self-contained bundle of exported IDs, content facts, descriptive metafacts,
and one content-addressed blob store shared by both fact sets. `entity!` and
import pipelines return fragments; `entity!` carries descriptions for the
attributes that actually emitted facts. Fragments compose via `+=` without
mixing descriptions into ordinary queries. Use `Fragment::root()` to extract
derived IDs, `Fragment::empty()` to start accumulation, and spread (`*`) to pass
child fragments into parent entities, giving Merkle trees for free.

### Derive
An unsigned exact equation mapping one source element into a derived collection.
The target descriptor names both source and recipe, so the record needs only
the target, input, and output identities. Derivations are reproducible cache
evidence, not authority.

### Merge
An unsigned exact equation `a ⊔ b = c` inside one collection. A validated merge
result can replace its inputs in a physical cover without changing the logical
value or creating new authority.

### PATCH
The **Persistent Adaptive Trie with Cuckoo-compression and Hash-maintenance**.
A single PATCH stores one ordering of a trible set in a 256-ary trie whose
nodes use byte-oriented cuckoo hash tables and copy-on-write semantics. A
`TribleSet` maintains six PATCH instances — one per permutation of entity,
attribute, and value. Shared leaves keep permutations deduplicated, rolling
hashes let set operations skip unchanged branches, and queries only visit the
segments relevant to their bindings, further described in
[the deep-dive chapter](deep-dive/patch.md).

### Pile
An append-only collection of blobs, native collection records, native
capability proofs, positive peer-routing evidence, OFFER records, and WANT records stored in
one file. Piles are memory
mapped, recoverable after interrupted appends, and mergeable by byte
concatenation. Legacy pin records remain decodable only for conservative
retention and explicit migration.

### Encoding
The byte-layout contract for a typed value. Encodings assign language-agnostic
meaning to the raw bytes — they are not the concrete Rust types — so any
implementation that understands the encoding can interpret the payloads
consistently. **Inline encodings** map the fixed 32-byte payload of a trible to
native types; **blob encodings** describe arbitrarily long payloads so tribles
referencing those blobs stay portable. The corresponding traits are
`InlineEncoding` and `BlobEncoding`.

### Team Root
The external Ed25519 key expected as `K0` in a team's direct capability proofs.
Its exact 32 public bytes are the resource for both CONNECT and SYNC_TEAM, the
scope prefix for the four-component inventory, and the namespace for provider
DHT keys. These shared bytes do not conflate authority: routing knowledge or
holding one proof never implies the other proof or any data access. Keeping the
secret offline after bootstrap is operational practice, not a one-use rule:
anyone holding it can issue another independent root proof.

### SYNC_TEAM
The exact `ACTION_SYNC_TEAM` atom used by `triblespace-net` to authorize one
team inventory session after CONNECT. Invoke authority for the authenticated
transport key permits disclosure of that team's PEER, collection-record,
capability-proof, and blob inventory. It is connection-local, validity-bounded,
and independent of routing evidence, DHT participation, and local
Demand/Mirror or direction policy.

### Trible
A three-part tuple of entity, attribute, and value stored in a fixed 64-byte
layout. Tribles capture atomic facts, and query engines compose them into joins
and higher-order results.

### TribleSpace
The storage model which organises tribles across blobs, PATCHes, and native
collections. It emphasizes immutable content-addressed data, monotone set
semantics, and reproducible derived representations.

### Inline
The third position in a trible. Values store a fixed 32-byte payload interpreted
through the attribute’s schema. They often embed identifiers for related
entities or handles referencing larger blobs.

### WANT
A durable local request for a blob or for an existing merge/derive result.
WANT is operational policy: it neither adds collection authority nor changes a
collection's logical value.

### OFFER

A durable positive local willingness to serve one content-addressed artifact.
OFFER forms a grow-only set and has no retraction. It grants no authority,
requests no bytes, contributes no collection or synchronized-inventory
evidence, proves no residency, and retains no blob.
