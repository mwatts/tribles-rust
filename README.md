# triblespace-search

Content-addressed BM25 + HNSW indexes on top of
[triblespace](https://github.com/triblespace/triblespace-rs) piles.

Two blob types, loaded zero-copy via [anybytes] and (eventually)
[jerky]:

- **`SuccinctBM25Index`** — lexical / associative retrieval. Terms
  are 32-byte triblespace `Value`s, so the index handles text
  search, entity co-occurrence, and tag weighting with the same
  schema.
- **`SuccinctHNSWIndex`** — approximate k-nearest-neighbour over
  caller-supplied embeddings.

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

* **`BM25Index`**: in-memory build + single- and multi-term
  query, content-addressed byte serialization, `BlobSchema`
  impl, two triblespace `Constraint`s — `docs_containing` (just
  `doc`) and `docs_and_scores` (`doc` + `score` as bound
  `Variable<GenId>` and `Variable<F32LE>`).
* **`FlatIndex`**: brute-force k-NN baseline. Build + cosine
  top-k query, byte serialization, two `Constraint`s —
  `similar_constraint` (just `doc`) and `similar_with_scores`
  (`doc` + `score`). Useful for ground truth and small corpora.
* **`HNSWIndex`**: layered-graph approximate k-NN (Malkov &
  Yashunin 2018) with deterministic level sampling, ef-search,
  byte serialization, `BlobSchema` impl, two `Constraint`s
  parallel to FlatIndex's. Validated at 1 000 docs / 32-dim
  against `FlatIndex` ground truth at ≥ 70% top-10 recall.
* **`tokens::hash_tokens`**: opt-in whitespace + lowercase +
  Blake3 tokenizer producing 32-byte term values.
* **`tokens::ngram_tokens`**: character n-gram tokenizer (n
  namespaced into the hash) for prefix / typo matching.
  Compose with `hash_tokens` to get both exact and fuzzy
  matching through a single BM25 index.
* **`schemas::F32LE`**: `ValueSchema` for packing `f32` scores
  into 32-byte `Value<F32LE>`s. Used by the scored BM25
  constraint.
* One runnable example (`cargo run --example query_demo`)
  covering text search, multi-term OR-queries, and the
  value-as-term citation-search trick.
* 98 tests across unit, scale (1k-doc), engine-integration
  (`IntersectionConstraint` joins + `find!`/`pattern!` composition),
  and doctests.

### What's next

* Jerky-backed succinct blobs (wavelet matrices for term-table
  + posting lists; per-layer wavelet matrices for HNSW
  neighbour graphs) via the `jerky::Serializable` pattern. Same
  API, smaller and faster.
* Additional token helpers (phrase rewriting, code-aware
  splitting).

See [`docs/DESIGN.md`](docs/DESIGN.md) and
[`docs/QUERY_ENGINE_INTEGRATION.md`](docs/QUERY_ENGINE_INTEGRATION.md).

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
