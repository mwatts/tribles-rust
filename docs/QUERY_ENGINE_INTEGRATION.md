# Query engine integration

How BM25 posting-list constraints and HNSW similarity plug into
the triblespace query engine as first-class `Constraint`s. Most
of this is shipped; sections marked **(planned)** describe the
still-open roadmap.

## Goal

A user writes a `find!` that composes BM25, HNSW, and arbitrary
`pattern!` clauses in one engine pass:

```rust
let hits: Vec<(Id,)> = find!(
    (paper: Id),
    temp!(
        (anchor, emb),
        and!(
            anchor.is(query_handle),
            bm25.docs_containing(paper, graph_term),
            pattern!(&kb, [{ ?paper @ attrs::paper_embedding: ?emb }]),
            hnsw_view.similar(anchor, emb, 0.8),
        )
    )
)
.collect();
```

— and the engine honours its normal cardinality-driven join
reordering, `or!` composition, `exists!` short-circuiting, etc.
Search stops being a bolt-on retrieval API and becomes part of
the query algebra. See `examples/hybrid_search.rs` for the full
runnable form.

## What a `Constraint` needs to provide

From `triblespace::core::query::Constraint`:

1. **Variables it touches** — returned as a `VariableSet` so
   the engine knows which assignments the constraint constrains.
2. **Cardinality estimate** per bound-variable combination. The
   engine uses this to reorder joins — smallest estimated
   result-set wins.
3. **Propose values for one variable given the others bound.**
   For BM25: propose `doc` given a pinned `term`; propose
   `score` given `doc + term`; etc.
4. **Confirm a fully-bound tuple.** Given all variables bound,
   yes or no.

## BM25 as a `Constraint`

Two constraint shapes, both generic over `(D, T)`:

### `docs_containing(doc, term)`

```rust
let c: DocsContainingTerm<'_, _, D> = idx.docs_containing(doc, term);
```

One variable (`doc: Variable<D>`), one pinned `term: Value<T>`.
The engine can propose entity ids whose posting list includes
the term, or confirm a bound doc against that posting list.

**Cardinality** = `doc_frequency(term)` — one posting-list
length, no guesswork.

### `docs_and_scores(doc, score, term)`

```rust
let c: BM25ScoredPostings<'_, _, D> =
    idx.docs_and_scores(doc, score, term);
```

Two variables (`doc`, `score: Variable<F32LE>`), one pinned
term. Every posting `(doc, score)` becomes a candidate tuple;
the engine can project either side or drive both.

Score dedupes by bit-pattern so two docs sharing a BM25 score
don't expand into a Cartesian cross. A `BM25Queryable::score_tolerance()`
method lets quantized backends widen the equality check
(lossless naive returns `f32::EPSILON`; succinct returns
`max_score / 65534`).

### Reverse lookup: doc bound, term free *(planned)*

Rare: "what terms are in this doc." Requires a per-doc inverted
structure. V1 doesn't expose this — the primary join direction is
term → docs. If a caller needs the reverse, they walk the term
table themselves and filter.

### Shared `BM25Queryable` trait

Both the naive `BM25Index<D, T>` and the succinct
`SuccinctBM25Index<D, T>` implement `BM25Queryable`. The
constraints are generic over it — same constraint types, same
`find!` integration, either backend.

## HNSW as a `Constraint`

### Binary relation: `similar(a, b, score_floor)`

Two variables (`a, b: Variable<Handle<Blake3, Embedding>>`) and
one fixed cosine threshold. Produced by the `similar` method on
any attached view:

```rust
let c: Similar<'_, _> = view.similar(a, b, score_floor);
```

**Semantics:** `similar(a, b, floor)` holds iff both handles
exist in the pile's blob store and `cosine(*a, *b) ≥ floor`.
The relation is symmetric (cosine is symmetric). Operationally,
at least one of `a` / `b` must be bound so the engine can walk
the HNSW graph from that side; the other side then enumerates
candidates above the floor.

### Why threshold, not top-k

Top-k is an *operational* choice — the caller decided to keep
N results after running the walk. Threshold is *semantic* — "a
cosine of ≥ 0.8 means the vectors are close enough for this
query". Semantics compose through the engine; operational
knobs don't. If a caller needs top-k, they walk on their side
and slice.

### Why fixed score (not a variable)

Score being free would force the index to report every visited
node during ef-search with its cosine, and the engine would
join on quantized scores. The caller almost never gets a
meaningful `score` from somewhere else in the query — and if
they want the exact similarity for a specific `(a, b)` pair,
fetching both embedding blobs and computing cosine directly is
one line and unaffected by u16 quantization.

### Cardinality

- One side bound: exact — run the walk, return the candidate
  count.
- Neither side bound: `usize::MAX` — the engine should order
  other constraints first. (Returning `None` would flag the
  variable as unconstrained.)

### Shared `SimilaritySearch` trait

`AttachedHNSWIndex`, `AttachedFlatIndex`, and
`AttachedSuccinctHNSWIndex` all implement `SimilaritySearch`
with two methods: `neighbours_above(handle, floor)` and
`cosine_between(a, b)`. Both are infallible at the trait
boundary — fetch failures fail-open as "no match" (empty vec /
`None`) because the engine's propose/confirm hooks have no
error channel.

## Combinators callers actually write

These compose naturally from the primitive constraints:

```rust
// Hybrid: title mentions 'graph' AND embedding close to query.
find!(
    (paper: Id),
    temp!((anchor, emb),
        and!(
            anchor.is(query_handle),
            bm25.docs_containing(paper, graph_term),
            pattern!(&kb, [{ ?paper @ attrs::paper_embedding: ?emb }]),
            hnsw.similar(anchor, emb, 0.8),
        )
    ),
)

// "Fragments citing X that also mention 'typst'."
find!(
    (doc: Id),
    and!(
        bm25.docs_containing(doc, id_as_term(x)),
        bm25.docs_containing(doc, hash_tokens("typst")[0]),
    ),
)

// "Similar to query, restricted to kind tag."
find!(
    (doc: Id),
    temp!((anchor, emb),
        and!(
            anchor.is(query_handle),
            pattern!(&kb, [
                { ?doc @ metadata::tag: &kind },
                { ?doc @ attrs::doc_embedding: ?emb },
            ]),
            hnsw.similar(anchor, emb, 0.7),
        )
    ),
)
```

The library does not provide `top_k` / `sort_by_score` in the
engine. Ordering is operational — callers collect the iterator
and slice. Matches the "unordered queries" tenet.

## Handle resolution

The constraint borrows from a specific index value (naive or
reloaded from a blob). Typical flow:

```rust
let handle: Value<Handle<Blake3, SuccinctBM25Blob>> =
    load_current_index_handle(&kb)?;
let reader = pile.reader()?;
let idx: SuccinctBM25Index =
    reader.get::<SuccinctBM25Index, SuccinctBM25Blob>(handle)?;
let c = idx.docs_containing(doc, term);
```

`idx` owns the data; `c` borrows it for the duration of the
query pass. A later rebuild produces a new handle; the next
query picks it up by loading the updated handle.

## Open questions

1. **Higher-level `bm25_query!` macro.** A single-token
   constraint (`docs_containing`) and a per-term scored one
   (`docs_and_scores`) are the primitives. A
   `bm25_query(doc, "some text", score)` macro that tokenises,
   runs `docs_and_scores` per term, sums scores through the
   engine, and projects `doc + score` would be the natural
   next step. Open: sum semantics (max / sum / OR-combined
   score) and how it plays with score dedupe.
2. **Reverse lookups.** See the BM25 section; currently
   unindexed, reported via doc-side walk if ever needed.
3. **Async / deferred index loading.** Large blobs are
   mmap-backed via `anybytes::Bytes` already; a `Bytes::view`
   failure happens at load time, not at constraint use time.

## Non-goals (v1)

- `top_k` / `sort_by_score` combinators. Callers slice.
- Hybrid score as a first-class bound variable. Callers write
  the linear combination in Rust.
- Live incremental updates. Rebuild-and-replace only.
- Cross-language bindings. Rust-native only.
