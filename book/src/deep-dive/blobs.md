# Blobs

Blobs are immutable byte sequences used whenever data does not fit in a
trible's fixed 256-bit value slot. `BlobEncoding` gives those bytes a portable
meaning, just as `InlineEncoding` interprets the value slot.

## Handles, encodings, and stores

A blob's handle combines its content hash with a compile-time encoding. The
same bytes have the same content identity, while the encoding tells a caller
how to validate and decode them. Handles fit inline in tribles, so large values
remain ordinary typed graph edges.

Storage is split into small traits:

- `BlobStorePut` inserts encoded values and returns their handles;
- `BlobStoreGet` resolves typed handles;
- `BlobStoreMeta` reports residency metadata without fetching bytes; and
- `BlobStoreList` enumerates stored objects where a backend supports it.

`MemoryRepo`, `Pile`, and remote object stores implement the relevant
capabilities. Content addressing makes repeat insertion idempotent and allows
caches to copy bytes without coordinating their names.

## Fragments carry their attachments

The usual application path does not put each blob manually. `entity!` accepts a
Rust value for a handle-valued attribute, encodes it, and puts the resulting
bytes into the returned fragment's shared attachment store:

```rust,ignore
use triblespace::prelude::*;
use triblespace::prelude::blobencodings::UTF8String;
use triblespace::prelude::inlineencodings::Handle;

attributes! {
    // Local prototype: derive the attribute from its name and encoding.
    pub body: Handle<UTF8String>;
}

let article = entity! {
    body: "A long string which lives in a content-addressed blob.",
};

assert_eq!(article.facts().len(), 1);
```

Composing fragments with `+=` unions their facts, metafacts, exported IDs, and
attachments. `Collection::commit(fragment)` copies those attachments before it
publishes the signed commit which refers to them. There is no separate staging
manifest to keep in sync.

## Put and get explicitly when needed

Low-level code can still work directly with a store:

```rust,ignore
use triblespace::prelude::*;
use triblespace::prelude::blobencodings::UTF8String;
use triblespace::prelude::inlineencodings::Handle;

let mut store = MemoryBlobStore::new();
let handle: Inline<Handle<UTF8String>> = store.put("Fear is the mind-killer.")?;
let reader = store.reader()?;
let value: View<str> = reader.get(handle)?;
assert_eq!(value.as_ref(), "Fear is the mind-killer.");
```

Explicit insertion is useful for importers, representation builders, and code
which must know a handle before constructing the referring entity.

## Archives are blobs too

Canonical `SimpleArchive` encodes a `TribleSet` as sorted, duplicate-free
64-byte rows. Collection descriptors, commit data, and commit metadata use this
representation. The handle of a descriptor archive is therefore the
collection's identity; the handle of a data archive is one collection element.

SuccinctArchive and other query-oriented formats are also typed blobs. Their
collection recipes define canonical merge and derivation operations, so a
validated derived artifact can be cached or forgotten without changing the
authority of the signed source commits.

## Conservative references

The generic retention walker scans blob bytes in aligned 32-byte chunks and
checks whether a chunk names a resident blob. This may retain an accidental
extra object, but it does not omit a real inline handle. Signed collection
commits recursively retain their resident descriptor, data, metadata, and
attachment closure. Unsigned merge and derivation equations do not create
strong ownership roots.

This division keeps tribles compact, blobs verifiable, and publication
self-contained while letting physical storage and cache policy evolve
independently.
