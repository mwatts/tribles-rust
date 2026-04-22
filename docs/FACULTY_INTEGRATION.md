# Using triblespace-search from a faculty

This crate is built to be used from the
[faculties](https://github.com/triblespace/faculties)
rust-script ecosystem: small, self-contained scripts with
`rust-script` shebangs that read and write a pile.

The pattern below is the distilled version of what
`cargo run --example compose_bm25_and_pattern` does, rewritten
as a runnable faculty that maintains a BM25 index over wiki
fragments in the caller's pile.

A runnable, self-contained version of the same pattern ships
as `cargo run --example faculty_wiki_search` — it uses a
tempdir pile, seeds a handful of fragments, runs refresh, then
issues a few queries. If the pseudo-code below drifts, that
example is the source of truth.

## Example: `wiki_search.rs`

```rust,ignore
#!/usr/bin/env -S rust-script
//! Build or refresh a BM25 index over wiki fragments in a pile
//! and answer queries against it. One-shot command:
//!
//!   wiki_search.rs --pile ./self.pile refresh
//!   wiki_search.rs --pile ./self.pile query "quick brown fox"
//!
//! ```cargo
//! [dependencies]
//! triblespace = "0.36"
//! triblespace-search = "0.0"
//! clap = { version = "4", features = ["derive"] }
//! ```

use clap::Parser;
use std::path::PathBuf;

use triblespace::core::find;
use triblespace::core::id::Id;
use triblespace::core::repo::{BlobStoreGet, BlobStorePut};
use triblespace::macros::pattern;

use triblespace_search::bm25::BM25Builder;
use triblespace_search::succinct::{SuccinctBM25Blob, SuccinctBM25Index};
use triblespace_search::tokens::hash_tokens;

// Assume a `wiki` namespace is already in the pile, providing:
//   wiki::title:   ShortString                   (fragment title)
//   wiki::body:    Handle<Blake3, LongString>    (typst body)
//   wiki::index:   Handle<Blake3, SuccinctBM25Blob>  (current-index handle)
mod wiki { /* ... */ }

#[derive(Parser)]
enum Cmd {
    /// Rebuild the BM25 index from current fragments and replace
    /// the handle stored under `wiki::index`.
    Refresh,
    /// Query the current index and print matching fragments.
    Query { text: String },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open the pile. (Code that differs only in pile-open
    // plumbing is left as stubs; see other faculties for the
    // real invocation.)
    let (mut pile, kb) = open_pile_and_kb()?;

    match Cmd::parse() {
        Cmd::Refresh => {
            // Walk every (fragment_id, body) pair. No shadow
            // datamodel — the pile is the source of truth.
            //
            // `BM25Builder::new()` gives the default
            // `BM25Builder<GenId, WordHash>` — entity-id doc
            // keys, Blake3-hashed text tokens. Pass `&id` and the
            // `ToValue<GenId>` impl handles the conversion;
            // `hash_tokens` returns `Vec<Value<WordHash>>` so
            // terms are typed end-to-end.
            let mut builder = BM25Builder::new();
            for (id, body) in find!(
                (id: Id, body: String),
                pattern!(&kb, [{ ?id @ wiki::body: ?body }])
            ) {
                builder.insert(&id, hash_tokens(&body));
            }
            // `.build()` goes direct to succinct — no separate
            // naive intermediate.
            let idx = builder.build();
            let handle = pile.put::<SuccinctBM25Blob, _>(&idx)?;
            // Persist the handle as a single-attribute trible
            // under a well-known anchor id (see below), or in
            // branch metadata.
            persist_index_handle(&mut pile, handle)?;
        }
        Cmd::Query { text } => {
            let handle = load_current_index_handle(&kb)?;
            let reader = pile.reader()?;
            // The type annotation on the get line picks the
            // D=GenId, T=WordHash defaults; an index built with
            // different schemas would spell them here too (e.g.
            // `SuccinctBM25Index<ShortString, WordHash>`).
            let idx: SuccinctBM25Index =
                reader.get::<SuccinctBM25Index, SuccinctBM25Blob>(handle)?;
            let q = hash_tokens(&text);
            // `.query_multi_ids` is the GenId-keyed convenience
            // that decodes the posting keys back to `Id`.
            for (id, score) in idx.query_multi_ids(&q).into_iter().take(10) {
                let title = lookup_title(&kb, id).unwrap_or_default();
                println!("{score:6.2}  {id}  {title}");
            }
        }
    }
    Ok(())
}
```

## Choosing schemas

`BM25Builder<D, T>` / `SuccinctBM25Index<D, T>` is generic over
the doc-key schema `D` and the term schema `T`. Defaults are
`<GenId, WordHash>` — what the skeleton above uses. Other
shapes are explicit:

| Use case | Type |
|---|---|
| Body-text search (the example) | `BM25Builder<GenId, WordHash>` |
| Title-keyed search | `BM25Builder<ShortString, WordHash>` |
| Phrase search (bigram terms) | `BM25Builder<GenId, BigramHash>` |
| Prefix / fuzzy (n-gram terms) | `BM25Builder<GenId, NgramHash>` |
| Entity co-occurrence ("which docs cite X?") | `BM25Builder<GenId, GenId>` |

Each tokenizer outputs a distinct term schema —
`hash_tokens` → `WordHash`, `bigram_tokens` → `BigramHash`,
`ngram_tokens` → `NgramHash` — so the compiler refuses to feed
the wrong flavour into the wrong index. Indexes that need
multiple tokenizer flavours become multiple indexes joined
via the query engine (`and!` / `or!`); see
`examples/phrase_search.rs` for the two-index pattern.

## Pattern: rebuild-and-replace, no mutation

triblespace-search indexes are content-addressed and immutable
by design. The `Refresh` command:

1. Reads every fragment from the pile (never pre-materializes
   into a separate store — see the "No shadow datamodels" rule
   in CLAUDE.md).
2. Builds a fresh `SuccinctBM25Index`.
3. `put`s the blob — the returned handle is the index's hash.
4. Replaces the reference in branch metadata (or a trible
   attribute under a stable id).

If nothing changed between refreshes, step 3 returns the same
handle because content-addressing: the pile's blob-dedup layer
stores it once. That's free caching.

## Pattern: query-time composition with `find!`

The BM25 constraint plugs into the same `find!` / `and!` /
`pattern!` engine as everything else. The typed term makes the
composition transparent — `hash_tokens("typst")[0]` is a
`Value<WordHash>`, which is exactly what
`idx.docs_and_scores(..)` wants on a `<GenId, WordHash>` index:

```rust,ignore
let rows: Vec<(Id, f32)> = find!(
    (doc: Id, score: f32),
    and!(
        idx.docs_and_scores(doc, score, hash_tokens("typst")[0]),
        pattern!(&kb, [{ ?doc @ wiki::tag: &some_tag }])
    )
).collect();
```

This is the "find docs with `typst` in the body AND tagged X"
query, running through a single engine pass. The engine picks
the cheaper side to iterate (`estimate()`) — either the BM25
posting list for `typst` or the tag index — and confirms with
the other.

See `examples/compose_bm25_and_pattern.rs` and
`examples/compose_hnsw_and_pattern.rs` for the full runnable
versions with a concrete KB.

## Pattern: score tolerance, not strict equality

When binding `score` as a `Variable<F32LE>` in a `find!`, scores
are quantized (u16, scale stored in SB25 header). The
constraint's equality check widens to
`score_tolerance() = max_score / 65534` automatically via the
`BM25Queryable` trait — callers writing raw equality checks
against a score from another source should use that tolerance
too:

```rust,ignore
let tol = idx.score_tolerance();
if (observed - expected).abs() <= tol { /* match */ }
```

## Open questions for faculty authors

- **Where does the handle live?** Branch metadata (one reference
  per branch), a trible under a stable id (pile-scoped), or a
  commit attribute (version-tied)? All three work; pick based
  on how often the index refreshes vs. how often branches move.

- **How big does the index get?** For the 100k-fragment target,
  the SB25 blob is ~86 MiB (naive would be ~157 MiB). See
  `cargo run --release --example blob_sizes_at_scale` for the
  actual numbers on a corpus your size.

- **When to embed?** BM25 is the default for text search. Layer
  in HNSW (`SuccinctHNSWIndex`) only once the caller has an
  embedding schema they're willing to commit to — the
  embeddings are the caller's data, not this crate's.

## Schema-rotation migrations

`triblespace-search` identifies each blob format by a
`BlobSchema` `ConstId`. A breaking byte-layout change rotates
that ID. Understanding what that *doesn't* do is important:

### `ConstId` is metadata, not a runtime type guard

`SuccinctBM25Blob::ID` is used by the describe/introspection
machinery and by derivations like `Handle<H, T>::ID =
hash(H::ID, T::ID)`. It is **not** checked by
`try_from_blob` — that dispatches on the Rust type
parameter. After an ID rotation:

- `SuccinctBM25Blob` is the same Rust type with the same
  `TryFromBlob` impl.
- `Blob<SuccinctBM25Blob>` is the same Rust type.
- Bytes in the pile under an old handle are the same bytes.
- `reader.get::<SuccinctBM25Index, SuccinctBM25Blob>(h)`
  returns those bytes and happily tries to parse them. If
  the old-format and new-format byte layouts happen to
  alias cleanly, you'll decode wrong values with no error.
  If they don't, you'll get a `TruncatedSection` —
  eventually, at runtime, not at compile time.

So the rotation **doesn't enforce anything on its own.** It's
a signal that old bytes and new code aren't compatible — the
caller has to act on that signal by rotating the identity
that the pile *does* use for dispatch.

### What the caller has to rotate

If you have an attribute whose value type references one of
our blob schemas — e.g.

```rust
struct WikiBm25Handle;
impl ConstId for WikiBm25Handle { const ID: Id = id_hex!("…"); }
// value type: Handle<Blake3, SuccinctBM25Blob>
```

the attribute `ConstId` is your stable contract for
"triples under this attribute hold *this* kind of value."
When we rotate `SuccinctBM25Blob::ID`, the Rust type the
attribute's value resolves to has changed underneath, but
the attribute's own ID hasn't — so old triples are still
under the same attribute, silently pointing at blobs whose
bytes no longer match what the code thinks they are.

Migration recipe:

1. `trible genid` → new attribute id, e.g. `wiki::bm25-v2`.
2. Rebuild the index against the new schema, `put` the
   blob, write the handle under the new attribute.
3. Transition readers to the new attribute. The old
   attribute + its stored triples become inert (any binary
   using them would have to keep the old crate version
   pinned anyway, because the old blob schema doesn't
   survive the rotation).
4. Once no reader references the old attribute, `pile keep`
   sweeps the orphaned blobs at the next consolidation.

Attributes whose value type is a plain `ShortString`,
`LongString`, `GenId`, etc. (anything not parameterized on a
crate-owned `BlobSchema`) are unaffected by our schema
rotations — only handles to *our* blob types transitively
depend on our IDs.
