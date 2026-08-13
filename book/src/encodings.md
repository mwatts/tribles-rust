# Encodings

TribleSpace stores data in strongly typed values and blobs. An *encoding*
describes the language‑agnostic byte layout for these types: [`Inline`]s always
occupy exactly 32&nbsp;bytes while [`Blob`]s may be any length. Encodings translate
those raw bytes to concrete application types and decouple persisted data from a
particular implementation. This separation lets you refactor to new libraries or
frameworks without rewriting what's already stored or coordinating live
migrations. The crate ships with a collection of ready‑made encodings located in
[`triblespace::core::inline::encodings`](https://docs.rs/triblespace/latest/triblespace/core/inline/encodings/index.html) and
[`triblespace::core::blob::encodings`](https://docs.rs/triblespace/latest/triblespace/core/blob/encodings/index.html).

When data crosses the FFI boundary or is consumed by a different language, the
encoding is the contract both sides agree on. Consumers only need to understand
the byte layout and identifier to read the data—they never have to link against
your Rust types. Likewise, the Rust side can evolve its internal
representations—add helper methods, change struct layouts, or introduce new
types—without invalidating existing datasets.

### Why 32 bytes?

Storing arbitrary Rust types requires a portable representation. Instead of
human‑readable identifiers like RDF's URIs, Tribles uses a fixed 32‑byte array
for all values. This size provides enough entropy to embed intrinsic
identifiers—typically cryptographic hashes—when a value references data stored
elsewhere in a blob. Keeping the width constant avoids platform‑specific
encoding concerns and makes it easy to reason about memory usage.

### Conversion traits

Conversion goes through the `Encodes<Source>` trait, which lives **on the
encoding** (the encoding is the impl target; the source is the trait parameter).
This is the same direction as std's `From<T>` — and for the same reason: it
trivially satisfies Rust's orphan rule, so you can write
`impl Encodes<SomeForeignType> for MyLocalEncoding` without any "trait
position 0" gymnastics.

The ergonomic source-side methods `.to_inline()` / `.to_blob()` /
`.into_encoded()` are auto-derived blanket implementations — users never
implement them directly, the same way you never implement `Into<T>` in Rust:

```text
User implements:     Auto-derived via blanket:
  Encodes<T> for S     IntoEncoded<S> for T  (+ IntoInline / IntoBlob aliases)
```

For fallible conversions where the error type is part of the contract (parsing
a hex string into a hash, validating a timestamp range, rejecting reserved
bits), use `TryToInline` / `TryFromInline` — kept as separate traits because the
error type is per‑source.

```rust
use triblespace::core::inline::encodings::shortstring::ShortString;
use triblespace::core::inline::{TryFromInline, TryToInline, Inline};

struct Username(String);

impl TryToInline<ShortString> for Username {
    type Error = &'static str;

    fn try_to_inline(self) -> Result<Inline<ShortString>, Self::Error> {
        if self.0.is_empty() {
            Err("username must not be empty")
        } else {
            self.0
                .as_str()
                .try_to_inline()
                .map_err(|_| "username too long or contains NULs")
        }
    }
}

impl TryFromInline<'_, ShortString> for Username {
    type Error = &'static str;

    fn try_from_inline(value: &Inline<ShortString>) -> Result<Self, Self::Error> {
        String::try_from_inline(value)
            .map(Username)
            .map_err(|_| "invalid utf-8 or too long")
    }
}
```

### Encoding identifiers

Every encoding declares a unique 128‑bit identifier, accessible via the
`MetaDescribe::id` method (for example, `ShortString::id()`).
Persisting these IDs keeps serialized data self describing so other tooling can
make sense of the payload without linking against your Rust types. Dynamic
language bindings (like the Python crate) inspect the stored encoding identifier
to choose the correct decoder, while internal metadata stored inside Trible
Space can use the same IDs to describe which encoding governs a value, blob, or
hash protocol.

Identifiers also make it possible to derive deterministic attribute IDs when you
ingest external formats. Wrap the source field name in an entity-core fragment —
`Attribute::<S>::from(entity!{ metadata::name: <name handle>, metadata::value_encoding: <S as MetaDescribe>::id() })` —
to combine the encoding ID with the source field name and produce a stable
attribute so re-importing the same data always targets the same column.
The `attributes!` macro offers three identity origins. Omitting the literal
derives identity from `(name, encoding)`, which is useful for quick experiments
or source-shaped internal attributes. `"HEX_ANCHOR" as name: Encoding` derives
identity from `(anchor, encoding)`, which is the preferred form for attributes
shared across binaries or languages: the Rust name can change freely, while a
type change truthfully creates a different column. The exceptional
`"HEX_ID" unsafe as name: Encoding` form uses the literal bytes verbatim. It is
for preserving an already-published identity and carries the unchecked
obligation that the encoding still agrees with all rows under that id.

## Built‑in inline encodings

The crate provides the following inline encodings out of the box:
- `GenId` &ndash; an abstract 128 bit identifier.
- `ShortString` &ndash; a UTF-8 string up to 32 bytes.
- `U256BE` / `U256LE` &ndash; 256-bit unsigned integers.
- `I256BE` / `I256LE` &ndash; 256-bit signed integers.
- `R256BE` / `R256LE` &ndash; 256-bit rational numbers.
- `ROrd256` &ndash; exact rationals whose bytes sort in numeric order.
- `F64` &ndash; IEEE-754 double-precision floating point number (little-endian).
- `F256BE` / `F256LE` &ndash; 256-bit floating point numbers.
- `Hash` and `Handle` &ndash; cryptographic digests and blob handles (see [`hash.rs`](../src/inline/encodings/hash.rs)).
- `ED25519RComponent`, `ED25519SComponent` and `ED25519PublicKey` &ndash; signature fields and keys.
- `NsTAIInterval` to encode time intervals.
- `Boolean` &ndash; all-zero for false, all-0xFF for true.
- `LineLocation` &ndash; a `(start_line, start_col, end_line, end_col)` span encoded as four big-endian u64 values.
- `RangeU128` &ndash; a half-open `(start, end)` range of two big-endian u128 values.
- `RangeInclusiveU128` &ndash; an inclusive `(start, end)` range of two big-endian u128 values.
- `UnknownInline` as a fallback when no specific encoding is known.

```rust
# use triblespace::prelude::*;
use triblespace::core::metadata::MetaDescribe;
use triblespace::core::inline::encodings::shortstring::ShortString;
use triblespace::core::inline::{IntoInline, InlineEncoding};

let v: Inline<ShortString> = "hi".to_inline();
let raw_bytes = v.raw; // Persist alongside the encoding's metadata id.
let encoding_id = ShortString::id(); // derived via describe(&mut scratch).root()
```

## Built‑in blob encodings

The crate also ships with these blob encodings:

- `LongString` for arbitrarily long UTF‑8 strings.
- `RawBytes` for opaque file-backed byte payloads.
- `SimpleArchive` which stores a raw sequence of tribles.
- `SuccinctArchiveBlob` which stores the [`SuccinctArchive` index
  type](https://docs.rs/triblespace/latest/triblespace/core/blob/encodings/succinctarchive/struct.SuccinctArchive.html)
  for offline queries. It contains only deterministic Ring/wavelet data and EOF
  metadata. `SuccinctArchiveRank9IndexBlob` is the separately
  content-addressed, source-bound native Rank9/select accelerator; its first 32
  bytes identify the exact raw archive it indexes. The `SuccinctArchive`
  helper exposes high-level iterators, returns both artifacts with
  `to_blob_pair`, and attaches an existing pair with `from_blob_pair`.
  `SuccinctArchiveBlob::build_from_simple_archive` derives the canonical raw
  artifact without constructing query indexes, while
  `SuccinctArchiveBlob::merge` computes an exact-validated raw set union with
  no runtime or Rank9 attachment.
- `WasmCode` for WebAssembly bytecode stored as a blob.
- `UnknownBlob` for data of unknown type.

```rust
use triblespace::core::metadata::MetaDescribe;
use triblespace::core::blob::encodings::longstring::LongString;
use triblespace::core::blob::{Blob, BlobEncoding, IntoBlob};

let b: Blob<LongString> = "example".to_blob();
let encoding_id = LongString::id(); // derived via describe(&mut scratch).root()
```

Both value and blob encodings can emit optional discovery metadata. Calling
`MetaDescribe::describe` returns a rooted `Fragment` (exporting the encoding id)
whose facts tag the encoding entity with `metadata::KIND_INLINE_ENCODING` or
`metadata::KIND_BLOB_ENCODING` and may attach a `metadata::name` and
`metadata::description` (LongString handles). Persist the description blobs
alongside the metadata tribles if you want the text to remain readable.

## Choosing the right encoding

When defining an attribute, the encoding determines how the 32-byte value slot is
interpreted. Use this decision tree to pick the right one:

```text
What are you storing?
│
├─ A reference to another entity?
│  └─ GenId
│
├─ A tag, category, or enum-like classifier?
│  └─ metadata::tag (GenId) — tags are entities with their own ID.
│     Use metadata::name to give them a human-readable label.
│     ⚠ Do NOT define a separate ShortString tag attribute —
│     use the canonical metadata::tag and mint tag IDs.
│
├─ A short label or display name?
│  ├─ Fits in 32 bytes (≤32 UTF-8 bytes)?
│  │  └─ ShortString
│  └─ Longer text?
│     └─ Handle<LongString>  (blob)
│
├─ A number?
│  ├─ Integer
│  │  ├─ Fits in 64 bits? → U256BE (zero-extended) or custom u64 encoding
│  │  └─ Needs full 256 bits? → U256BE / I256BE
│  ├─ Floating point
│  │  ├─ Standard double? → F64
│  │  └─ Extended precision? → F256BE
│  └─ Rational? → R256
│     ⚠ Canonical (reduced) but NOT numerically ordered:
│     comparison hits the numerator first, so 1/1 sorts
│     before 2/3. Use a fixed-scale integer if the index
│     has to answer "greater than x" — or ROrd256 if you
│     need exact division AND numeric range queries.
│
├─ A timestamp or time range?
│  └─ NsTAIInterval
│
├─ A cryptographic value?
│  ├─ Content hash? → Hash<Blake3>
│  ├─ Reference to a blob? → Handle<BlobEncoding>
│  └─ Signature? → ED25519RComponent / ED25519SComponent / ED25519PublicKey
│
├─ A file or binary payload?
│  └─ Handle<RawBytes>  (blob)
│
├─ A large structured dataset?
│  └─ Handle<SimpleArchive>  (blob, stores a TribleSet)
│
└─ Something else?
   ├─ Fits in 32 bytes? → define a custom InlineEncoding
   └─ Larger? → define a custom BlobEncoding + use Handle
```

**Rules of thumb:**
- If two values should be joinable (appear in the same query variable), they must
  share an encoding. Choose the most specific encoding that covers both uses.
- Prefer `ShortString` over `LongString` when the text fits — inline values avoid
  a blob lookup.
- Use `GenId` for relationships between entities. Never store entity references as
  strings.
- When in doubt between an inline encoding and a blob, ask: "will I ever want to
  query or join on this directly?" If yes, it should be inline. If it's opaque
  content you just retrieve, use a blob handle.

## Exact rationals: `R256` vs `ROrd256`

Indexes compare the 32 stored bytes, so an index range is only a *value* range
when the encoding's byte order matches its numeric order. `R256` does not have
that property. It stores the numerator in the first 16 bytes and the denominator
in the last 16, so bytewise comparison reads the numerator first and `1/1` sorts
below `2/3` even though `1 > 2/3`. Its big-endian variant gives a stable portable
layout, not a numerically meaningful one.

`ROrd256` is the sibling encoding that does sort numerically, while staying
exact and canonical.

**How.** The Stern–Brocot tree is a binary search tree over the rationals, so
its in-order traversal is numerically sorted and a root path (`L`/`R`) with a
terminator sorting between `L` and `R` compares lexicographically in numeric
order. A raw path is not width-bounded — `1/1000000` is 999999 left branches
deep — but the continued fraction `[a0; a1, a2, …]` is exactly the run-length
encoding of that path, so it carries the identical order in O(log min(p,q))
terms. Comparison of continued fractions alternates (`a0` ascending, `a1`
descending, …), so each term is written in a prefix-free order-preserving code
and bit-complemented at odd positions, which turns the alternation back into
plain lexicographic comparison. A terminated fraction behaves like a `+∞` term
at the next position, so the terminator is a run of the pad value that sorts
above every code — and complementing it at odd positions correctly turns that
`+∞` into `−∞`.

**Canonical.** The Euclidean algorithm never emits the alternative `[…, n-1, 1]`
spelling, so exactly one byte string exists per value and intrinsic ids stay
stable. Bytes that claim a trailing `1` are rejected by `validate`.

**Representable subset.** Encoding costs roughly `2·log2(max(|p|, q))` bits and
fails with a typed `OrderedRatioError::OutOfDomain` rather than rounding:

| input | fits |
|---|---|
| any `p/q` with `max(\|p\|, q) ≤ 2^104` | always (guaranteed) |
| any `i128` integer, any `1/n` | always |
| random 96-bit `p` and `q` | always in practice |
| random 120-bit `p` and `q` | ~99% |
| random 127-bit `p` and `q` | ~26% |

The guaranteed bound is set by long continued fractions of small-but-not-one
terms; the smallest value that does *not* fit needs `max(|p|, q) > 2^104.7`.
Counter-intuitively, Fibonacci ratios — the *longest* continued fractions — are
the cheapest, because a term of `1` costs a single bit, so every Fibonacci ratio
representable in `i128` encodes comfortably.

**Cost.** Ordering is free at query time (it is `memcmp` on bytes the index
already compares); you pay for it on write. Encoding runs a Euclidean expansion
and decoding a continuant recurrence, both O(number of terms):

| | `ROrd256` | `R256BE` | `R256LE` |
|---|---|---|---|
| encode, 64-bit `p/q` | 290 ns | 157 ns | 1.6 ns |
| encode, worst case | 850 ns | 270 ns | 1.6 ns |
| decode, 64-bit `p/q` | 176 ns | 157 ns | 157 ns |

`R256LE`'s encode is two `to_le_bytes` and nothing else, so relative to a raw
two-limb store `ROrd256` is two orders of magnitude slower to encode. Against
`R256BE` — which canonicalizes with a gcd — it is under 2× for typical values,
and decoding is comparable either way because `R256`'s own canonicality check
also runs a gcd. Run `cargo bench -p triblespace-core --bench ordered_rational`
to reproduce.

**Choose `ROrd256`** when you need exactness *and* numeric range queries on the
same column: exact empirical rates `k/n` where the denominator varies, exact
probabilities or thresholds, ratios that arise from division and are then
filtered by magnitude. The alternative — an `R256` column plus a parallel `F64`
sort key — needs two attributes that can drift apart, and its range answers are
inexact at the boundary, because the bound itself (`1/3`, say) is not a float.
`ROrd256` makes the index answer the exact answer.

**Choose `R256`** for everything else. It is simpler, encodes in nanoseconds, and
covers the full `i128 × i128` box rather than a subset of it.

**Choose neither for money.** Amounts have a fixed scale, are not divided at
storage, and already sort numerically as scaled integers. Use `Currency<C>`,
which is exactly that.

## Defining new encodings

Custom formats implement [`InlineEncoding`] or [`BlobEncoding`]. A unique identifier
serves as the encoding ID. The example below defines a little-endian `u64`
inline encoding and a simple blob encoding for arbitrary bytes.

```rust,ignore
{{#include ../../examples/custom_schema.rs:custom_schema}}
```

See [`examples/custom_schema.rs`](https://github.com/triblespace/triblespace-rs/blob/main/examples/custom_schema.rs) for the full
source.

### Versioning and evolution

Schemas form part of your persistence contract. When evolving them consider the
following guidelines:

1. **Prefer additive changes.** Introduce a new encoding identifier when breaking
   compatibility. Consumers can continue to read the legacy data while new
   writers use the replacement ID.
2. **Annotate data with migration paths.** Store both the encoding ID and a
   logical version number if the consumer needs to know which rules to apply.
   `UnknownInline`/`UnknownBlob` allow you to safely defer decoding until a newer
   binary is available.
3. **Keep validation centralized.** Place invariants in your encoding
   conversions so migrations cannot accidentally create invalid values.

By keeping encoding identifiers alongside stored values and blobs you can roll out
new representations incrementally: ship readers that understand both IDs, update
your import pipelines, and finally switch writers once everything recognizes the
replacement encoding.

## Inline formatters (WASM)

Binary formats are great for portability and performance, but they can be
painful to inspect if you don’t know the encoding ahead of time. TribleSpace
supports an optional encoding-level formatter mechanism: an inline encoding can point
to a small sandboxed WebAssembly module that turns its raw 32 bytes into a
human-readable string.

The formatter is stored as a blob (`blobencodings::WasmCode`) and referenced from
the encoding identifier entity via the metadata attribute `metadata::value_formatter`.

The built-in runner lives behind the `wasm` feature flag (enabled by default in
the `triblespace` facade crate) and uses `wasmi` with tight limits (fuel, memory
pages, output size). Modules must not import anything and use the following
minimal ABI:

- `memory` (linear memory)
- `format(w0: i64, w1: i64, w2: i64, w3: i64) -> i64`

The `format` arguments are the raw 32 bytes split into 4×8-byte chunks
(little-endian). The return value packs the output pointer and output length:

- Success returns `(output_len << 32) | output_ptr` with `output_ptr != 0`.
- Failure returns `(error_code << 32) | 0` (i.e. `output_ptr == 0`).

The core crate can optionally ship built-in formatters for its built-in value
encodings. Enable the `wasm` feature to have
`MetaDescribe::describe` (which is fallible) attach `metadata::value_formatter` entries for the
standard encodings. This feature requires the `wasm32-unknown-unknown` Rust
target at build time because the bundled formatters are compiled to WebAssembly
via the `#[value_formatter]` proc macro.
