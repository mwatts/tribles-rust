# Using triblespace-search from a faculty

This crate is built to be used from the
[faculties](https://github.com/triblespace/faculties)
rust-script ecosystem: small, self-contained scripts with
`rust-script` shebangs that read and write a pile.

The pattern below is the distilled version of what
`cargo run --example compose_bm25_and_pattern` does, rewritten
as a runnable faculty that maintains a BM25 index over wiki
fragments in the caller's pile.

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
use triblespace::core::query::Variable;
use triblespace::core::repo::{BlobStoreGet, BlobStorePut};
use triblespace::core::value::schemas::genid::GenId;
use triblespace::macros::pattern;

use triblespace_search::bm25::BM25Builder;
use triblespace_search::succinct::{SuccinctBM25Blob, SuccinctBM25Index};
use triblespace_search::tokens::hash_tokens;

// Assume a `wiki` namespace is already in the pile, providing:
//   wiki::title:   ShortString   (fragment title)
//   wiki::body:    LongString    (typst body)
//   wiki::index:   SuccinctBM25Blob  (current-index handle)
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
            // Query every (fragment_id, body) pair. No shadow
            // datamodel — the pile is the source of truth.
            let mut builder = BM25Builder::new();
            for (id, body) in find!(
                (id: Id, body: String),
                pattern!(&kb, [{ ?id @ wiki::body: ?body }])
            ) {
                builder.insert(id, hash_tokens(&body));
            }
            let idx = SuccinctBM25Index::from_naive(&builder.build())?;
            let handle = pile.put::<SuccinctBM25Blob, _>(&idx)?;
            // Persist the handle as a single-attribute fragment
            // under a well-known id, or as branch metadata.
            persist_index_handle(&mut pile, handle)?;
        }
        Cmd::Query { text } => {
            let handle = load_current_index_handle(&kb)?;
            let reader = pile.reader()?;
            let idx: SuccinctBM25Index =
                reader.get::<SuccinctBM25Index, SuccinctBM25Blob>(handle)?;
            let q = hash_tokens(&text);
            for (id, score) in idx.query_multi(&q).into_iter().take(10) {
                // Cross-reference back to the pile for display.
                let title = lookup_title(&kb, id).unwrap_or_default();
                println!("{score:6.2}  {id}  {title}");
            }
        }
    }
    Ok(())
}
```

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
`pattern!` engine as everything else:

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
that ID. If your crate has minted an attribute whose value
type references one of our schemas — e.g.

```rust
struct WikiBm25Handle;
impl ConstId for WikiBm25Handle { const ID: Id = id_hex!("…"); }
// value type: Handle<Blake3, SuccinctBM25Blob>
```

then rolling forward to a new version of this crate means:

1. **Types re-derive automatically.** `Handle<Blake3,
   SuccinctBM25Blob>::ID` cascades from our new schema id
   on recompile — Rust's `ConstId` derivation for `Handle`
   is `hash(H::ID, T::ID)`. The type system catches
   callers that try to mix old-schema handles with
   new-schema readers at compile time.

2. **Stored triples do not.** Triples you already wrote
   under the old attribute still point at *old-format*
   blobs. The old attribute now implicitly claims to hold
   new-format blobs (semantically — the on-disk triple is
   unchanged), so those triples are effectively orphaned:
   readable only by a binary pinned to the old crate
   version. Re-ingest is the clean path.

Migration recipe for any attribute that holds a handle to
one of our schemas:

1. `trible genid` → new attribute id, e.g. `wiki::bm25-v2`.
2. Rebuild the index against the new schema, `put` the
   blob, write the handle under the new attribute.
3. Transition readers to the new attribute. Once no reader
   references the old attribute, `pile keep` will sweep
   the old blobs the next time you consolidate.

Attributes whose value type is a plain `ShortString`,
`LongString`, `GenId`, etc. (anything not parameterized on a
crate-owned `BlobSchema`) are unaffected by our schema
rotations — only handles to *our* blob types transitively
depend on our IDs.
