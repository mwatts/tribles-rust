# Glossary

This chapter collects the core terms that appear throughout the book. Skim it
when you encounter unfamiliar terminology or need a refresher on how concepts
relate to one another in TribleSpace.

### Action
An uninterpreted 128-bit identifier naming one exact operation in the positive
authority kernel. Actions do not form a hierarchy and never imply one another.
For example, `ACTION_WRITE` and `ACTION_CONNECT` are separate atoms even when
they concern collections associated with the same team.

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

### Authority Collection
The canonical public `SimpleArchive`-union collection named `authority` for one
team root. Its content handle is `authority::collection(team_root)`. Signed
grant occurrences are ordinary commits in this grow-only collection, with
canonical empty metadata; there is no separate mutable membership store.

### Authority Grant
One atomic positive authority statement carried by one signed authority
collection commit. The outer commit signer is the issuer and its intrinsic
commit ID is the grant occurrence ID. The canonical grant names one direct
subject key, one exact resource collection handle, one exact action, optional
exact parent occurrence, and explicit invoke/delegate uses. Independent
accepted grants are alternatives under set union.

### Authority Proof
A self-contained, root-to-leaf sequence of authority commits paired with their
exact canonical grant archives. Verification checks the team-root anchor,
signatures, data identities, exact parent links, issuer/subject binding,
unchanged action and resource, and delegation at every step, then checks the
leaf against the caller's exact subject/action/resource/mode claim. This
claim-directed final check prevents a valid truncated prefix from authorizing
the intended descendant.

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

### Commit
A signed native collection membership assertion. A `CollectionCommit` names
the exact collection descriptor, data element, mandatory metadata archive, and
author. Its intrinsic record ID is derived from the canonical 192-byte payload.
Commits are independent leaves rather than snapshots in a parent chain.

### Collection
A self-describing grow-only join semilattice. Signed commits introduce members;
validated merge records describe joins within the lattice; derivation records
map elements into another collection through a canonical homomorphism. A
collection has no distinguished head.

### Collection Descriptor
A canonical `SimpleArchive` describing a collection's anchor, element
representation, join recipe, and reach law. Its content handle is the
`CollectionHandle`, so every native record which names a collection can resolve
its meaning through the ordinary blob store.

### Collection Store
A grow-only set of native `COMMIT`, `MERGE`, and `DERIVE` records. Insertion is
idempotent by intrinsic record ID; combining two stores is set union.

### Collection Snapshot
One coherent known-prefix observation containing materialized facts, the exact
verified commit set which authorized them, and the blob reader which validated
their dependencies.

### CONNECT
The exact `ACTION_CONNECT` atom used by `triblespace-net` to authenticate a
direct-RPC session. Its resource is the team's
`authority::collection(team_root)`, and its claimed subject must equal the
transport peer key. CONNECT grants no WRITE, generic READ, gossip, collection
reach, semantic trust, custody, or retention authority. Protocol v6 carries
the complete claim-directed proof inline on the connection's first `OP_AUTH`
stream.

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
An append-only collection of blobs, native collection records, and WANT records
stored in one file. Piles are memory mapped, recoverable after interrupted
appends, and mergeable by byte concatenation. Legacy pin records remain
decodable only for conservative retention and explicit migration.

### Encoding
The byte-layout contract for a typed value. Encodings assign language-agnostic
meaning to the raw bytes — they are not the concrete Rust types — so any
implementation that understands the encoding can interpret the payloads
consistently. **Inline encodings** map the fixed 32-byte payload of a trible to
native types; **blob encodings** describe arbitrarily long payloads so tribles
referencing those blobs stay portable. The corresponding traits are
`InlineEncoding` and `BlobEncoding`.

### Team Root
The Ed25519 key whose public half identifies a team and its one canonical
[authority collection](#authority-collection). A no-parent authority grant is
grounded only when its collection commit is signed by this root. Keeping the
secret offline after bootstrap is operational practice, not a one-use rule in
the algebra: anyone holding it can publish another independent root grant.
`PeerConfig.team_root` fixes both the CONNECT proof anchor and, when gossip is
enabled, the team topic ID.

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

### Ticket
The exact canonical set of signed commits selected as authority for one
materialization or derivation. A ticket is explicit continuation state and can
be diffed by intrinsic commit ID without walking a commit chain.

### WANT
A durable local request for a blob or for an existing merge/derive result.
WANT is operational policy: it neither adds collection authority nor changes a
collection's logical value.
