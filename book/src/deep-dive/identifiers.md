# Identifiers for Distributed Systems

Distributed systems are assembled from independently authored pieces of data.
Keeping those pieces addressable requires names that survive replication,
concurrent edits, and local conventions. We have found it useful to categorize
identifier schemes along two axes:

|                | **Abstract**                | **Semantic**      |
|----------------|-----------------------------|-------------------|
| **Intrinsic**  | Hash, Signature             | Embeddings        |
| **Extrinsic**  | UUID, UFOID, FUCID, PubKey  | Names, DOI, URL   |

- **Rows — derivability.** An *intrinsic* identifier can be recomputed
  from the entity alone: anyone holding the bytes can produce the same
  id independently. An *extrinsic* identifier is assigned separately
  from the entity — the entity carries no hint of what its id should be.
- **Columns — content encoding.** An *abstract* identifier is opaque:
  its bits carry no readable meaning about the entity. A *semantic*
  identifier encodes meaning (words, codes, URL paths) that humans or
  machines can consume as signal without a lookup.

The axes are independent — every cell is populated — but the quadrants
have different structural properties (see
[Quadrant Properties](#quadrant-properties) below). Classifying an
identifier along both axes makes its trade-offs explicit and clarifies
when a workflow needs to combine multiple schemes.

## Abstract vs. Semantic Identifiers

### Semantic identifiers

Semantic identifiers (names, URLs, descriptive labels, embeddings) carry meaning
about the thing they reference. That context makes them convenient for humans
and for search workflows where the identifier itself narrows the scope of
possible matches. Their usefulness comes with caveats:

- Semantics drift. A name that once felt accurate can become misleading as an
  entity evolves.
- Semantic identifiers are rarely unique. They work best as entry points into a
  richer dataset rather than as the source of truth for identity.
- Without scoping, semantics clash. Two communities can reasonably reuse the
  same word for different concepts, so the identifier must be tied to a
  namespace, deployment, or audience.

Distributed systems reconcile those tensions by mapping many local semantic
names to a persistent abstract identifier. Users remain free to adopt whatever
terminology makes sense to them while the system maintains a shared canonical
identity.

Embeddings deserve a special mention. They encode meaning in a machine-friendly
form that can be compared for similarity instead of exact equality. That makes
them great for recommendations and clustering but still unsuitable as primary
identifiers: two distinct entities can legitimately share similar embeddings,
and embeddings can change whenever the underlying model is retrained.

### Abstract identifiers

Abstract identifiers (UUIDs, UFOIDs, FUCIDs, hashes, signatures) strip all
meaning away in favor of uniqueness. They can be minted without coordination,
usually by drawing from a high-entropy space and trusting probability to keep
collisions effectively impossible. Abstract identifiers shine when you need:

- Stable handles that survive across replicas and through refactors.
- Globally unique names without a centralized registrar.
- Cheap, constant-time generation so every component can allocate identifiers on
  demand.

Because they carry no inherent semantics, abstract identifiers are almost always
paired with richer metadata. They provide the skeleton that keeps references
consistent while semantic identifiers supply the narrative that humans consume.

## Intrinsic vs. Extrinsic Identifiers

The intrinsic/extrinsic axis captures whether an identifier can be recomputed
from the entity itself or whether it is assigned externally.

### Intrinsic identifiers

Intrinsic identifiers (cryptographic hashes, digital signatures, content-based
addresses) are derived from the bytes they describe. They function as
fingerprints: if two values share the same intrinsic identifier then they are
bit-for-bit identical. This property gives us:

- Immutability. Changing the content produces a different identifier, which
  immediately signals tampering or corruption.
- Self-validation. Replicas can verify received data locally instead of trusting
  a third party.
- Stronger adversarial guarantees. Because an attacker must find collisions
  deliberately, intrinsic identifiers rely on cryptographic strength rather than
  purely statistical rarity.

### Extrinsic identifiers

Extrinsic identifiers (names, URLs, DOIs, UUIDs, UFOIDs, FUCIDs) are assigned by
policy instead of by content. They track a conceptual entity as it evolves
through versions, formats, or migrations. In other words, extrinsic identifiers
carry the "story" of a thing while intrinsic identifiers nail down individual
revisions.

Thinking about the classic ship of Theseus thought experiment makes the
distinction concrete: the restored ship and the reconstructed ship share the
same extrinsic identity (they are both "Theseus' ship") but have different
intrinsic identities because their planks differ.

### A note on signatures and public keys

Signatures are a special form of hash: a digest over `(content,
private_key)`. Used as identifiers they sit in the intrinsic-abstract
cell — anyone holding the content and the author's public key can
recompute and verify.

Public keys used as *actor identities* (identifying a person, agent, or
service) are a different animal: a pubkey doesn't derive from the
person it identifies — it's assigned by whoever generated the keypair.
That makes pubkey-as-identity extrinsic abstract, essentially a
high-entropy UUID with a bonus cryptographic capability.

## Quadrant Properties

Classifying along two independent axes leaves four quadrants, and each
has a structural property worth calling out because it constrains
system design.

**Extrinsic + Semantic ⇒ an authority.** If an identifier carries
meaning and *can't* be derived from the entity, someone had to assign
that meaning. Semantic content is low-entropy by nature — the space of
sensible names is small, so two parties minting independently will
often collide — and avoiding collisions requires coordination. That
coordinator is the authority, whether explicit (DOI registrar, ICANN,
ISBN agency) or implicit (a cultural namespace, an organization's
wiki). There is no decentralized extrinsic-semantic scheme.

**Extrinsic + Abstract can be fully decentralized.** A 128-bit random
identifier collides only statistically. Two parties minting UUIDs (or
UFOIDs, or pubkeys) independently will never see each other's values
in practice, so no coordinator is needed — the decentralization is
paid for in entropy, not consensus.

**Intrinsic + anything is decentralized by construction.** The entity
is the authority: any two parties with the same bytes produce the same
id (for hashes) or the same neighborhood in embedding space (for
embeddings). No registrar exists, because none is needed.

**The upshot:** if you want a decentralized naming system, you have to
give up one of *extrinsic* or *semantic*. Every centralized naming
system in practice (domains, ISBNs, DOIs, academic affiliations)
occupies the extrinsic-semantic quadrant because that's where authority
is structurally required.

## Picking entity ids in code

TribleSpace lets you mix intrinsic and extrinsic identifiers depending on what
you are modeling. The `entity!` macro mirrors that split:

```rust
use triblespace::examples::literature;
use triblespace::prelude::*;

// Intrinsic identity (default): the id is derived deterministically from the
// attribute/value pairs, so identical record literals unify.
let record = entity! {
    literature::firstname: "Frank",
    literature::lastname: "Herbert",
};

// Extrinsic identity: supply an id expression when you want a stable subject
// whose facts can evolve across edits and commits.
let alice = ufoid();
let subject = entity! { &alice @
    literature::firstname: "Frank",
};

// `_ @` is an explicit synonym for the intrinsic form.
let also_intrinsic = entity! { _ @ literature::firstname: "Frank" };

// Optional facts: use `?:` with an `Option<T>` value to omit missing data
// without resorting to branching.
let maybe_alias: Option<&str> = None;
let with_optional = entity! { _ @
    literature::firstname: "Frank",
    literature::alias?: maybe_alias,
};

// Repeated facts: use `*:` with an `IntoIterator<Item = T>` to emit multiple
// facts for the same attribute.
let aliases = ["Frank", "F.H."];
let with_repeated = entity! { _ @
    literature::firstname: "Frank",
    literature::alias*: aliases,
};
```

## Embeddings as Semantic Intrinsic Identifiers

Embeddings blur our neat taxonomy. They are intrinsic because they are computed
from the underlying data, yet they are overtly semantic because similar content
produces nearby points in the embedding space. That duality makes them powerful
for discovery:

- Systems can exchange embeddings as a "lingua franca" without exposing raw
  documents.
- Expensive feature extraction can happen once and power many downstream
  indexes, decentralizing search infrastructure.
- Embeddings let us compare otherwise incomparable artifacts (for example, a
  caption and an illustration) by projecting them into a shared space.

Despite those advantages, embeddings should still point at a durable abstract
identifier rather than act as the identifier. Collisions are expected, model
updates can shift the space, and floating-point representations can lose
determinism across hardware.

## High-Entropy Identifiers

For a truly distributed system, the creation of identifiers must avoid the bottlenecks and overhead associated
with a central coordinating authority. At the same time, we must ensure that these identifiers are unique.  

To guarantee uniqueness, we use abstract identifiers containing a large amount of entropy, making collisions
statistically irrelevant. However, the entropy requirements differ based on the type of identifier:
- **Extrinsic abstract identifiers** need enough entropy to prevent accidental collisions in normal operation.
- **Intrinsic abstract identifiers** must also resist adversarial forging attempts, requiring significantly higher entropy.  

From an information-theoretic perspective, the length of an identifier determines the maximum amount of
entropy it can encode. For example, a 128-bit identifier can represent \( 2^{128} \) unique values, which is
sufficient to make collisions statistically negligible even for large-scale systems.  

For intrinsic identifiers, 256 bits is widely considered sufficient when modern cryptographic hash functions
(e.g., SHA-256) are used. These hash functions provide strong guarantees of collision resistance, preimage
resistance, and second-preimage resistance. Even in the event of weaknesses being discovered in a specific
algorithm, it is more practical to adopt a new hash function than to increase the bit size of identifiers.  

Additionally, future advances such as quantum computing are unlikely to undermine this length. Grover's algorithm
would halve the effective security of a 256-bit hash, reducing it to \( 2^{128} \) operations—still infeasible with
current or theoretical technology. As a result, 256 bits remains a future-proof choice for intrinsic identifiers.  

Such 256-bit intrinsic identifiers are represented by the types
[`Hash`](crate::value::schemas::hash::Hash) and
[`Handle`](crate::value::schemas::hash::Handle).  

Not every workflow needs cryptographic strength. We therefore ship three
high-entropy abstract identifier families—**RNGID, UFOID, and FUCID**—that keep
128 bits of global uniqueness while trading off locality, compressibility, and
predictability to suit different scenarios.

## Comparison of Identifier Types

|                        | [RNGID](crate::id::rngid::rngid) | [UFOID](crate::id::ufoid::ufoid) | [FUCID](crate::id::fucid::fucid) |
|------------------------|----------------------------------|----------------------------------|----------------------------------|
| Global entropy         | 128 bits                        | 96 bits random + timestamp       | 128 bits                         |
| Locality               | None                            | High (time-ordered)              | High (monotonic counter)         |
| Compression friendliness | None                          | Low                              | High                             |
| Predictability         | None                            | Low (reveals mint time)          | High (per-source sequence)       |

"Predictability" here is a tradeoff axis, not a quality: higher
predictability enables tighter compression and cache-friendly scans, but
reveals mint metadata (time or source) and is therefore unsuitable
whenever adversarial unpredictability matters. For those cases prefer
RNGID's fully random bits, or step up to a 256-bit cryptographic
[`Hash`](crate::value::schemas::hash::Hash).

## Example: Scientific Publishing

A published paper plays three distinct identifier roles — the
`.html`/`.pdf` artifact, the "same paper across revisions" grouping,
and the human-readable label in a citation — that the quadrant framing
suggests pulling apart.

- **Artifacts** belong in the *intrinsic-abstract* quadrant: identify
  each PDF by a cryptographic hash of its bytes. Any two parties
  referencing the same digest are looking at bit-for-bit identical
  content, and verification is self-contained.
- **Revision groupings** belong in the *extrinsic-abstract* quadrant:
  mint a UFOID/FUCID when the paper is first created, and attach every
  later revision's content hash to it. The id is stable across rewrites
  and decentralizable (no registrar needed, just enough entropy).
- **Human-readable labels** — citation keys, abbreviations, working
  titles — are semantic, contextual, and deliberately local. They
  don't carry system identity; they're just display strings that point
  at one of the above.

DOIs land in the *extrinsic-semantic* quadrant: a publisher prefix, a
journal stem, often a human-recognizable slug — all assigned by a
registrar. By the [quadrant properties](#quadrant-properties) above
that quadrant *structurally* requires a central authority, for
collision avoidance (low semantic entropy) and for resolution
(translating the id to a concrete artifact). DOIs-as-centralized isn't
a design flaw; it's the price of the quadrant. You can't have
decentralization *and* extrinsic semantic content at once.

What triblespace recommends isn't "replace DOIs with something better
in the same quadrant" — it's to *decompose* the role DOIs try to play
into identifiers that each live in a quadrant they fit: content hashes
for artifact identity, abstract extrinsic ids for revision grouping,
and semantic labels (including DOIs, when you need citation
compatibility with the outside world) as per-context references on top.

## ID Ownership

In distributed systems, consistency requires monotonicity due to the
[CALM principle](https://arxiv.org/abs/1901.01930) ("Consistency As
Logical Monotonicity" — any program that only grows its state can be
eventually consistent without coordination; anything that can retract
state requires coordination).
However, this is not necessary for single-writer systems. By assigning
each ID an owner, we ensure that only the current owner can write new
information about an entity associated with that ID. This allows for
fine-grained synchronization and concurrency control.

To create a transaction, you can uniquely own all entities involved and write new data for them
simultaneously. Since there can only be one owner for each ID at any given time, you can be
confident that no other information has been written about the entities in question.

By default, all minted `ExclusiveId`s are associated with the thread they are dropped from.
These IDs can be found in queries via the `local_ids` function.

Once the IDs are back in scope you can either work with them directly as
[`ExclusiveId`](crate::id::ExclusiveId)s or move them into an explicit
[`IdOwner`](crate::id::IdOwner) for a longer lived transaction.  The example
below shows both approaches in action:

```rust
use triblespace::examples::literature;
use triblespace::prelude::*;

let mut kb = TribleSet::new();
{
    let isaac = ufoid();
    let jules = ufoid();
    kb += entity! { &isaac @
        literature::firstname: "Isaac",
        literature::lastname: "Asimov",
    };
    kb += entity! { &jules @
        literature::firstname: "Jules",
        literature::lastname: "Verne",
    };
} // `isaac` and `jules` fall back to this thread's implicit IdOwner here.

let mut txn_owner = IdOwner::new();
let mut updates = TribleSet::new();

for (author, name) in find!(
    (author: ExclusiveId, name: String),
    and!(
        local_ids(author),
        pattern!(&kb, [{
            ?author @ literature::firstname: ?name
        }])
    )
) {
    // `author` is an ExclusiveId borrowed from the implicit thread owner.
    let author_id = txn_owner.insert(author);

    {
        let borrowed = txn_owner
            .borrow(&author_id)
            .expect("the ID was inserted above");
        updates += entity! { &borrowed @ literature::lastname: name.clone() };
    } // `borrowed` drops here and returns the ID to `txn_owner`.
}
```

The `entity!` macro accepts `ExclusiveId`s by value or reference, so you can
pass either an owned guard or a borrowed one.

Sometimes you want to compare two attributes without exposing the comparison
variable outside the pattern. Prefixing the binding with `_?`, such as
`_?name`, allocates a scoped variable local to the macro invocation. Both
`pattern!` and `pattern_changes!` will reuse the same generated query variable
whenever the `_?` form appears again, letting you express equality constraints
inline without touching the outer [`find!`](crate::query::find) signature.

Binding the variable as an [`ExclusiveId`](crate::id::ExclusiveId) means the
closure that [`find!`](crate::query::find) installs will run the
[`TryFromValue`](crate::value::TryFromValue) implementation for `ExclusiveId`.
The conversion invokes [`Id::acquire`](crate::id::Id::acquire) and would silently
skip the row if the current thread did not own the identifier (filter
semantics).  The
[`local_ids`](crate::query::local_ids) constraint keeps the query safe by only
enumerating IDs already owned by this thread, so no rows are filtered in
practice.  In the example we immediately
move the acquired guard into `txn_owner`, enabling subsequent calls to
[`IdOwner::borrow`](crate::id::IdOwner::borrow) that yield
[`OwnedId`](crate::id::OwnedId)s.  Dropping an `OwnedId` automatically returns
the identifier to its owner so you can borrow it again later.  If you only need
the ID for a quick update you can skip the explicit owner entirely, bind the
variable as a plain [`Id`](crate::id::Id), and call
[`Id::acquire`](crate::id::Id::acquire) when exclusive access is required.

### Ownership and Eventual Consistency

While a simple grow set (like the commit histories backing a branch)
already constitutes a conflict-free replicated data type (CRDT), it is
also limited in expressiveness. To provide richer semantics while
guaranteeing conflict-free mergeability we allow only "owned" IDs to be
used in the `entity` position of newly generated triples. As owned IDs
are [`Send`] but not [`Sync`] owning a set of them essentially
constitutes a single-writer transaction domain, allowing for some
non-monotonic operations like `if-does-not-exist` over the set of
contained entities. Note that this does not make operations that would
break CALM (consistency as logical monotonicity) safe — e.g. `delete`.
