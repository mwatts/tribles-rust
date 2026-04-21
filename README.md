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
  `7AFE59E7F895B23F05452FF7919E12E4`) — approximate k-nearest-
  neighbour over caller-supplied embeddings. Graph stored as
  per-(layer, node) CSR in two jerky `CompactVector`s.

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
* **`FlatIndex`**: brute-force k-NN baseline + two `Constraint`s.
  Useful for ground truth and small corpora.
* **`HNSWIndex`** (naive Malkov & Yashunin 2018) with
  deterministic level sampling, ef-search, byte serialization,
  two `Constraint`s parallel to FlatIndex's. Validated at
  1 000 docs / 32-dim against `FlatIndex` at ≥ 70 % top-10
  recall.
* **`SuccinctHNSWIndex`**: jerky-backed zero-copy view — doc
  ids + flat f32 vectors + CSR graph via two `CompactVector`s.
  SH25 blob format; same `similar()` output as the naive
  index bit-for-bit.
* **Shared constraint traits** `BM25Queryable` + `HNSWQueryable`
  — the same `Constraint` implementations work against naive or
  succinct indexes through the engine.
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
* **`schemas::F32LE`**: `ValueSchema` for packing `f32` scores
  into 32-byte `Value<F32LE>`s. Used by the scored BM25
  constraint.
* Four runnable examples:
  - `query_demo` — text search, multi-term OR, value-as-term
    citation search.
  - `compose_bm25_and_pattern` — BM25 + `pattern!` over a
    `TribleSet` in one `find!`.
  - `compose_hnsw_and_pattern` — vector similarity + `pattern!`
    composition.
  - `blob_sizes_at_scale` — naive vs. SB25 blob size at 1k /
    5k / 10k docs.
* 146 tests across unit, scale (1k-doc equivalence +
  naive-vs-SB25 size guard), engine-integration
  (`IntersectionConstraint` joins + `find!` / `pattern!`
  composition + `find!` over both succinct paths), and
  doctests.

### What's next

* Replace the naive blob format with `SuccinctBM25Index` end-
  to-end (drop the transitional `bm25::SuccinctBM25Index` marker).
* Quantized BM25 scores (u16 `CompactVector` instead of raw
  f32) — cuts postings to ~2 bytes per entry.
* Wavelet-matrix HNSW graph per DESIGN.md's RING plan.
* Additional token helpers (phrase rewriting, code-aware
  splitting).

See
[`docs/DESIGN.md`](docs/DESIGN.md),
[`docs/QUERY_ENGINE_INTEGRATION.md`](docs/QUERY_ENGINE_INTEGRATION.md),
and
[`docs/HNSW_GRAPH_ENCODING.md`](docs/HNSW_GRAPH_ENCODING.md).

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
