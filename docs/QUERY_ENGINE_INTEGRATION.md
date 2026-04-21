# Query engine integration — design sketch

How `bm25(?doc, ?term, ?score)` and `similar(?doc, vec, ?score)`
plug into the triblespace query engine as first-class constraints.
Not yet implemented; this doc is the target design so the next
few iterations (and future contributors) have a fixed aim.

## Goal

A user should be able to write:

```rust
for (doc, score) in find!(
    (doc: Id, score: f32),
    pattern!(&space, [
        { doc @ wiki::content: _ },          // filter: has content
        bm25.contains(&doc, query_term, &score, 0.1),
    ]),
) { ... }
```

and have the engine honor its normal cardinality-driven join
reordering, `or!` composition, `exists!` short-circuiting, etc.
BM25 and vector search stop being a bolt-on retrieval API and
become part of the query algebra.

## What a `Constraint` needs to provide

From `triblespace_core::query::constraint::Constraint`:

1. **Variables it touches.** Reported as `VariableSet` so the
   engine knows which variable assignments the constraint
   constrains.
2. **Cardinality estimate** per bound-variable combination. The
   engine uses this to reorder joins — the smallest estimated
   result-set constraint goes first.
3. **Propose values for one variable given the others are
   bound.** This is the core — for BM25, propose `doc` given a
   bound `term`, or propose `term` given a bound `doc`, etc.
4. **Confirm a fully-bound tuple.** Given all variables bound,
   yes or no.

## BM25 as a `Constraint`

Three variables: `(doc: Id, term: Value, score: f32)`.

### Forward lookup: term bound, doc + score free

Common case: "find all docs matching this term".

```
for (doc, score) in index.query_term(&term) {
    propose(doc, score);
}
```

Cardinality: `index.doc_frequency(&term)` — one posting list
length, no guesswork.

### Reverse lookup: doc bound, term + score free

Rare but supported: "what terms are in this doc". Requires a
per-doc inverted structure. In the naive build we walk the full
posting table (O(total_postings)); in the succinct build we
store a second wavelet matrix indexed by doc. V1 can stub this
branch and log "unindexed access pattern" so we notice if
callers hit it.

### Full confirm: doc, term, score all bound

Query `index.query_term(&term)`, scan for the doc, compare
score with tolerance. O(doc_frequency(term)) per check, but
confirms are rare in join plans.

### Score threshold

Callers typically want "score > threshold" — expose as a
constructor parameter on the constraint:

```rust
bm25.contains(&doc, &term, &score, threshold: 0.1)
```

Threshold becomes part of cardinality estimation (how many
postings exceed `t`). V1 approximates as "all postings"; later
we can keep per-term score histograms for tighter estimates.

## HNSW as a `Constraint`

Vector search is a bit different — the "query" is a vector, not
a variable. Shape:

```rust
similar(&doc, query_vec, &score, k: 10)
```

Variables: `(doc, score)`. The `query_vec` is a *parameter*
baked into the constraint at construction time, not a bound
variable. Cardinality = `k`.

Propose: run `index.similar(query_vec, k)`, stream
`(doc, score)` pairs.

## Combinators callers will actually write

These fall out of the above primitive constraints:

```rust
// Hybrid: BM25 + vector, linear combination.
bm25(doc, term, bm25_score) and
similar(doc, vec, cos_score) and
let combined = alpha * bm25_score + (1 - alpha) * cos_score

// "Fragments citing X that also mention 'typst'"
bm25(doc, id_as_term(X), _) and
bm25(doc, hash_token("typst"), _) and
pattern!([{ doc @ metadata::tag: TAG_VERSION }])

// "Top-5 by similarity to `vec`, restricted to `kind` tag"
similar(doc, vec, score, k=50) and
pattern!([{ doc @ metadata::tag: &kind }])
// then caller sorts by score desc, truncates to 5
```

The library does not provide a `top_k` / `sort_by_score`
combinator; callers collect the `(doc, score)` iterator and
take what they need. Keeping ordering out of the engine matches
the "unordered queries" tenet.

## Handle resolution

The constraint needs to be constructed against a *specific*
index blob. The flow:

```rust
let handle: Handle<SuccinctBM25Index> = /* from branch metadata,
                                           commit metadata, or
                                           a plain trible */;
let blob = pile.get::<SuccinctBM25Index>(handle)?;
let idx = BM25Index::try_from_blob(blob)?;
let cst = idx.contains(&doc_var, &term_var, &score_var);
```

`BM25Index` owns the data; `cst` borrows. Lifetime:
`cst` lives for at most one query pass.

For long-lived agent processes that keep an index resident, a
convenience `Arc<BM25Index>` constraint can share the data
across many queries without per-query allocation.

## Rebuild-and-replace within a query plan

If a query plan references an index handle from branch metadata
and the handle updates mid-plan — doesn't happen: constraints
resolve their `Arc<BM25Index>` once at `find!` entry. A later
rebuild produces a new handle; the next query picks it up.

## Open questions worth flagging

1. **Score as f32 in a 256-bit-wide engine.** triblespace's
   query engine bundles values as 32-byte things. Encoding an
   f32 score as a `Value<F32LE>` (zero-padded) is fine but the
   schema type needs minting — another `trible genid`.
2. **Cardinality for reverse lookups.** Without a per-doc
   inverted structure, `term`-given-`doc` proposals have no
   tight estimate. V1 can use a loose upper bound (avg_doc_len
   as a proxy) and accept that the engine may reorder
   sub-optimally for those rare cases.
3. **Threaded build / parallel propose.** The proposer is
   inherently sequential per-constraint, but the underlying
   index lookups could be threaded across terms. Out of scope
   for v1; note for profiling later.
4. **Async / deferred index loading.** A large index blob
   should be mmap-backed; `anybytes::Bytes` already supports
   this. The constraint constructor is where we'd verify the
   blob is resident (`Bytes::view`) and fail fast otherwise.

## Implementation order

1. Mint a `Value<F32LE>` schema for score bundling.
2. Implement `BM25Index::contains(&self, doc, term, score, threshold) -> impl Constraint`.
3. Unit tests: posting-list → iterator → constraint
   enumeration parity.
4. `similar(&vec, k)` constraint for FlatIndex, same pattern.
5. Integration test: faculty-sized corpus (few thousand
   fragments), demonstrate a 2-clause `find!` query that uses
   both BM25 and a `pattern!` filter.

## Non-goals (v1)

- A `top_k` combinator. Callers slice.
- Hybrid-score-as-a-first-class-query. Callers write the
  combinator in Rust.
- Live incremental updates. Rebuild-and-replace only.
- Cross-language (Python, C) bindings. Rust-native only.
