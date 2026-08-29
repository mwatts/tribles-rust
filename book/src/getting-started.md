# Getting Started

This chapter publishes and queries a small native collection. It assumes Rust
is installed and you are comfortable running `cargo` commands.

## 1. Add dependencies

```bash
cargo new tribles-demo
cd tribles-demo
cargo add triblespace ed25519-dalek rand
```

`triblespace` supplies the data model, stores, collection operations, and query
macros. `ed25519-dalek` and `rand` create the publishing identity used in this
example.

## 2. Declare attributes

Attributes carry the encoding of their value. Shared attributes should use a
stable explicit anchor; the encoding participates in the resulting identity.
Omit the anchor only for local prototypes whose identity may follow their name.

```rust,ignore
mod literature {
    use triblespace::prelude::*;
    use triblespace::prelude::blobencodings::UTF8String;
    use triblespace::prelude::inlineencodings::{GenId, Handle, ShortString};

    attributes! {
        "A74AA63539354CDA47F387A4C3A8D54C" as pub title: ShortString;
        "6A03BAF6CFB822F04DA164ADAAEB53F6" as pub quote: Handle<UTF8String>;
        "8F180883F9FD5F787E9E0AF0DF5866B9" as pub author: GenId;
        "0DBB530B37B966D137C50B943700EDB2" as pub firstname: ShortString;
        "6BAA463FD4EAF45F6A103DB9433E4545" as pub lastname: ShortString;
        "D2D1B857AC92CEAA45C0737147CA417E" as pub alias: ShortString;
    }
}
```

Use `trible genid` when minting a new published anchor. The literal-pinning
`"HEX_ID" unsafe as ...` spelling is only for preserving an already-published
attribute's exact historical bytes when its old identity cannot be re-derived.

## 3. Register a collection

A root collection is identified by the content handle of its descriptor. The
descriptor carries its UTF-8 name, mandatory authority, representation, join
recipe, and reach law. The descriptor itself is an ordinary self-contained
`Fragment`; its canonical content handle is the collection value:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::collection::{reach, simplearchive_union};
use triblespace::prelude::*;

let key = SigningKey::generate(&mut OsRng);
let mut storage = MemoryRepo::default();
let descriptor = simplearchive_union::descriptor(
    "library",
    key.verifying_key(),
    reach::private(),
);
let library = storage.collection(descriptor)?;
```

`reach::private()` declares no permissionless relay. Use `reach::public()` only
when the collection's identity should state that any holder may relay its
verified commits. Reach does not partition an explicitly authorized team
inventory: a store attached to `triblespace-net` is dedicated to one team, and
SYNC_TEAM reconciliation includes every resident collection record.

Local publication is unconditional: any process may append a structurally
valid self-signed commit to its own store. Reading applies authority. Commits by
the descriptor authority are admitted automatically; delegated authors become
visible only when the caller presents a root-to-leaf proof for exact
`ACTION_WRITE` on this descriptor. Invalid explicit evidence fails loud.

## 4. Build a self-contained fragment

With no explicit subject prefix, `entity!` derives the entity ID from the
canonical set of emitted fields. `Fragment::root()` returns that exported ID,
which can be referenced by another entity. Long strings become blobs and remain
attached to the fragment automatically.

```rust,ignore
let author = entity! {
    literature::firstname: "Frank",
    literature::lastname: "Herbert",
};
let author_id = author.root().expect("intrinsic author id");

let book = entity! {
    literature::title: "Dune",
    literature::author: &author_id,
    literature::quote: "I must not fear. Fear is the mind-killer.",
};

let mut import = author;
import += book;
```

A fragment has four coordinated channels: facts, descriptive metafacts,
exported IDs, and one shared attachment store. `+=` unions all four, so no
parallel manifest or manual blob-staging step is needed.

## 5. Publish independent commits

```rust,ignore
let first = storage.commit(library, &key, import)?;

// A later fact about the same entity is another independent member.
let second = storage.commit(library, &key, entity! {
    &author_id @ literature::alias: "Francis",
})?;

assert_ne!(first.id(), second.id());
```

Publication exact-validates the registered descriptor, then writes fragment
attachments, data, and metadata before inserting the signed `COMMIT` record.
It performs no permission check and no implicit flush. The commit is the atomic
assertion. There is no mutable head to advance: both records remain members and
the collection value is their union.

Repeating byte-identical input produces the same record ID and is idempotent.
Distinct input produces another coexisting member. Application-level
supersession or versioning is represented in the facts when a domain needs it;
append order is never an implicit winner.

## 6. Read one coherent snapshot

```rust,ignore
let snapshot = storage.snapshot(library, &[])?;
let title = "Dune";

for (first, last, quote) in find!(
    (first: String, last: String, quote),
    pattern!(snapshot.facts(), [
        { _?author @
            literature::firstname: ?first,
            literature::lastname: ?last
        },
        { _?book @
            literature::title: title,
            literature::author: _?author,
            literature::quote: ?quote
        }
    ])
) {
    let quote: View<str> = snapshot.reader().get(quote)?;
    println!("'{}'\n - from {title} by {first} {last}.", quote.as_ref());
}
```

`snapshot()` admits the descriptor authority plus explicitly supplied
delegated presentations at one clock instant. It opens one target blob-reader
view and materializes facts solely from the resulting exact payload cover. The
returned `Snapshot<SimpleArchiveUnion, TribleSet, R>` keeps facts, its typed
`Cover<SimpleArchiveUnion>`, and reader together. A concurrent commit may
appear on this call or a later call, but physically visible blobs from an
unobserved commit cannot leak into the snapshot's admitted set.

Use `storage.cover(library, presentations)` when only the exact payload
frontier is needed. It verifies presentations and scans native collection
records, but it does not fetch or materialize member blobs. Duplicate signed
claims for one payload collapse to one cover member; authorship, signatures,
and metadata currently known to the store are reported separately by
`storage.claims(&cover)`, which may validly return no claims after construction.
The cover is the continuation passed to derived representations such as
SuccinctArchive or path-index collections.

## 7. Choose durability explicitly

`store.commit` performs no implicit flush. For a memory store that makes
no difference. For a pile or remote backend, choose the durability boundary
that matches the application:

```rust,ignore
storage.commit(library, &key, batch_a)?;
storage.commit(library, &key, batch_b)?;
storage.flush()?;
```

Amortizing one flush over several commits does not weaken their logical
identity or change merge semantics. Flushing and closing remain operations of
the chosen backend rather than collection policy.

## What to remember

- `entity!` builds intrinsic entities and carries required blobs.
- `Fragment` is the self-contained publication value.
- `Collection<L>` is a descriptor handle statically bound to its member
  encoding and join law; the store owns all I/O.
- `store.commit` publishes one signed, independent member without conflating
  local storage with network authorization.
- `store.snapshot` returns one coherent known-prefix view admitted by the
  descriptor authority and explicit delegated proofs.
- Replicas converge by unioning records; they never elect a branch head.
- Derived indexes are reproducible collection images, not alternate authority.

Continue with [Collection Workflows](repository-workflows.md) for the native
record algebra, exact covers, migration from legacy piles, and derived
collection maintenance.
