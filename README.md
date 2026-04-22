# triblespace-search

Content-addressed BM25 + HNSW indexes on top of
[triblespace](https://github.com/triblespace/triblespace-rs) piles.

Two blob types, loaded zero-copy via [anybytes] and [jerky]:

- **`SuccinctBM25Index`** (SB25 blob, schema id
  `68C03764D04D05DF65E49589FBBA1441`) — lexical / associative
  retrieval. Terms are 32-byte triblespace `Value`s, so the
  index handles text search, entity co-occurrence, and tag
  weighting with the same schema. Postings bit-packed via jerky
  `CompactVector`.
- **`SuccinctHNSWIndex`** (SH25 blob, schema id
  `A96890DE5F85A4F2285C365549B21BC2`) — approximate cosine
  similarity over caller-supplied embedding handles. Graph
  stored as per-(layer, node) CSR in two jerky `CompactVector`s.
  Nodes are `Handle<Blake3, Embedding>` values; the caller's
  doc-to-embedding mapping is a trible they own, not a shadow
  datamodel inside the index.

Both indexes are rebuilt-and-replaced (no mutation). The resulting
blob handle is persisted wherever the caller likes — branch
metadata, commit metadata, a plain trible, or an in-memory cache.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design.

[anybytes]: https://github.com/triblespace/anybytes
[jerky]: https://github.com/triblespace/jerky

## Status

**Pre-alpha.** Public API shapes are still settling. Version 0.0.0
until the first non-scaffold release. The design is frozen; the
naive-then-succinct implementation order is the open work item.

### What works today

* **`BM25Index`** (naive in-memory): build + single- and multi-
  term query, content-addressed byte serialization, two
  triblespace `Constraint`s — `docs_containing` (just `doc`)
  and `docs_and_scores` (`doc` + `score` as bound
  `Variable<GenId>` + `Variable<F32LE>`).
* **`SuccinctBM25Index`**: jerky-backed zero-copy view — doc
  ids via `FixedBytesTable<16>`, terms via `FixedBytesTable<32>`,
  doc-lengths + postings via `CompactVector`. Same query
  surface; SB25 blob format lands directly in a pile via
  `ToBlob`/`TryFromBlob`.
* **`FlatIndex`**: brute-force exact cosine baseline. Same
  `similar(a, b, score_floor)` constraint as HNSW — useful for
  ground truth and small corpora.
* **`HNSWIndex`** (naive Malkov & Yashunin 2018) with
  deterministic level sampling, ef-search, byte serialization.
  Validated at 1 000 handles / 32-dim against `FlatIndex` at
  ≥ 70 % above-threshold recall.
* **`SuccinctHNSWIndex`**: jerky-backed zero-copy view — a
  `FixedBytesTable<32>` of embedding handles plus a CSR graph
  encoded as two `CompactVector`s. Nodes IS the handle; the
  caller's doc → embedding mapping lives in their tribles, not
  here.
* **Binary-relation similarity constraint** `similar(a, b,
  score_floor)` produced by the `similar()` method on any
  attached view. `a` and `b` are `Variable<Handle<Blake3,
  Embedding>>`; `score_floor` is a fixed cosine threshold.
  Callers who need the exact score fetch both embeddings and
  compute cosine directly — no u16 quantization.
* **Shared constraint trait** `SimilaritySearch` (HNSW, Flat,
  SuccinctHNSW) + `BM25Queryable` (naive + succinct BM25) — the
  same constraints work against either backend.
* **`tokens::hash_tokens`**: opt-in whitespace + lowercase +
  Blake3 tokenizer producing 32-byte term values.
* **`tokens::ngram_tokens`**: character n-gram tokenizer (n
  namespaced into the hash) for prefix / typo matching.
  Compose with `hash_tokens` to get both exact and fuzzy
  matching through a single BM25 index.
* **`tokens::code_tokens`**: identifier tokenizer — splits on
  camelCase, `snake_case`, digit boundaries, and acronyms
  (`HTMLParser` → `html`, `parser`). Lowercased output hashes
  the same as `hash_tokens`, so code and prose can share one
  index.
* **`tokens::bigram_tokens`**: word-level bigram tokenizer
  namespaced into `"2w:"` so bigrams and single-word hashes
  coexist in one index. Compose with `hash_tokens` to answer
  both single-word and phrase queries — `bigram_tokens("quick
  brown")` hashes only the ordered pair, so a doc matches iff
  the two words appear adjacently.
* **`schemas::F32LE`**: `ValueSchema` for packing `f32` scores
  into 32-byte `Value<F32LE>`s. Used by the scored BM25
  constraint.
* Eight runnable examples:
  - `query_demo` — text search, multi-term OR, value-as-term
    citation search.
  - `compose_bm25_and_pattern` — BM25 + `pattern!` over a
    `TribleSet` in one `find!`.
  - `multi_term_bm25_search` — higher-level `bm25_query`
    constraint joined with a `pattern!` filter, projecting the
    summed BM25 score as a bound engine variable.
  - `compose_hnsw_and_pattern` — similarity + `pattern!`
    composition via the binary `Similar` relation.
  - `hybrid_search` — BM25 + similarity + `pattern!` in one
    `find!`; both filters active simultaneously.
  - `blob_sizes_at_scale` — naive vs. SB25 blob size + parallel
    build speedup at 1k / 5k / 10k / 50k docs.
  - `query_latency` — p50/p99 latency for BM25 queries and
    HNSW threshold walks.
  - `phrase_search` — `hash_tokens` + `bigram_tokens` in two
    typed indexes; same corpus answers single-word and phrase
    queries.
* 154 tests across unit, scale (1k-doc equivalence +
  naive-vs-SB25 size guard), engine-integration
  (`IntersectionConstraint` joins + `find!` / `pattern!`
  composition + `find!` over both succinct paths), and
  doctests.

### What's next

* A higher-level `bm25_query(?doc, "typst links", ?score)`
  macro that tokenizes the query string and aggregates per-term
  scores through the engine.
* Wavelet-matrix HNSW graph per DESIGN.md's RING plan (no
  current win under forward-only traversal; see
  `docs/HNSW_GRAPH_ENCODING.md`).

See
[`docs/DESIGN.md`](docs/DESIGN.md),
[`docs/QUERY_ENGINE_INTEGRATION.md`](docs/QUERY_ENGINE_INTEGRATION.md),
[`docs/HNSW_GRAPH_ENCODING.md`](docs/HNSW_GRAPH_ENCODING.md),
and
[`docs/FACULTY_INTEGRATION.md`](docs/FACULTY_INTEGRATION.md) for
the rust-script faculty consumption pattern.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
