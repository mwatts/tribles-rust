# Getting Started

This chapter publishes and queries a small native collection. It assumes Rust
is installed and you are comfortable running `cargo` commands.

## 1. Add dependencies

```bash
cargo new tribles-demo
cd tribles-demo
cargo add triblespace ed25519-dalek rand
```

`triblespace` supplies the data model, stores, collection facade, and query
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

## 3. Open a collection

A root collection is identified by the content handle of its descriptor. The
descriptor names the collection within a team and states its representation,
join recipe, and reach law. A single-user process is simply a team of one:

```rust,ignore
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::collection::reach;
use triblespace::prelude::*;

let key = SigningKey::generate(&mut OsRng);
let team = key.verifying_key();
let name = CollectionName::new("library")?;
let mut library = Collection::new(
    MemoryRepo::default(),
    &name,
    team,
    key,
    reach::private(),
);
```

`reach::private()` means that network peers do not proactively gossip this
collection. Use `reach::public()` only when the collection's identity should
state that any holder may relay its verified commits.

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
let first = library.commit(import)?;

// A later fact about the same entity is another independent member.
let second = library.commit(entity! {
    &author_id @ literature::alias: "Francis",
})?;

assert_ne!(first.id(), second.id());
```

Publication writes the descriptor, data archive, metadata archive, and fragment
attachments before inserting the signed `COMMIT` record. The commit is the
atomic assertion. There is no mutable head to advance: both records remain
members and the collection value is their union.

Repeating byte-identical input produces the same record ID and is idempotent.
Distinct input produces another coexisting member. Application-level
supersession or versioning is represented in the facts when a domain needs it;
append order is never an implicit winner.

## 6. Read one coherent snapshot

```rust,ignore
let snapshot = library.snapshot()?;
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

`snapshot()` discovers one exact verified commit set, opens one blob-reader
view, and materializes facts solely from those commits. The returned
`CollectionSnapshot` keeps all three together. A concurrent commit may appear
on this call or a later call, but physically visible blobs from an unobserved
commit cannot leak into the snapshot's authority set.

Use `ticket()` when only the exact commit frontier is needed. It reads native
records without opening a blob view or materializing facts, which is useful for
feeding derived representations such as SuccinctArchive or path-index
collections.

## 7. Choose durability explicitly

`Collection::commit` performs no implicit flush. For a memory store that makes
no difference. For a pile or remote backend, choose the durability boundary
that matches the application:

```rust,ignore
library.commit(batch_a)?;
library.commit(batch_b)?;
library.flush()?;
```

Amortizing one flush over several commits does not weaken their logical
identity or change merge semantics. Consume the facade with `into_storage()`
when another component needs the backend, or call `close()` where the backend
supports explicit close.

## What to remember

- `entity!` builds intrinsic entities and carries required blobs.
- `Fragment` is the self-contained publication value.
- `Collection::commit` publishes one signed, independent member.
- `Collection::snapshot` returns a coherent known-prefix view.
- Replicas converge by unioning records; they never elect a branch head.
- Derived indexes are reproducible collection images, not alternate authority.

Continue with [Collection Workflows](repository-workflows.md) for the native
record algebra, exact tickets, migration from legacy piles, and derived
collection maintenance.
