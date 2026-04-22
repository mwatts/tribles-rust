# Changelog

All notable changes to `triblespace-search`. The crate is
pre-alpha and version-locked at `0.0.0` until the first
non-scaffold release — entries below capture the shape of the
crate as it is today.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
dates are commit dates rather than release dates.

## Unreleased / pre-alpha

### Blob types (the shipped surface)

- **`SuccinctBM25Index`** (blob schema `SuccinctBM25Blob`, id
  `68C03764D04D05DF65E49589FBBA1441`) — SB25 wire format, 236 B
  header + bit-packed body via jerky. Carries doc ids (flat),
  terms (sorted flat), doc-lens (`CompactVector`), and postings
  (three `CompactVector`s in one `ByteArea`: `doc_idx`,
  cumulative `offsets`, and u16-quantized `scores` scaled by a
  header-stored `max_score`). At 100 k docs the blob is ~86
  MiB, roughly half the naive byte-format.
- **`SuccinctHNSWIndex`** (blob schema `SuccinctHNSWBlob`, id
  `7AFE59E7F895B23F05452FF7919E12E4`) — SH25 wire format, 152 B
  header + body: doc ids, flat f32 vectors (zero-copy viewed as
  `&[f32]` via `anybytes::View`), and a CSR-shaped graph built
  from two jerky `CompactVector`s. Same greedy + ef-search as
  the naive `HNSWIndex`, producing bit-identical top-k results
  at 1 k scale.
- Both blob types implement `ToBlob` / `TryFromBlob` against
  their respective schemas, and survive a real
  `triblespace::core::repo::BlobStorePut` / `BlobStoreGet`
  round-trip (see `tests/pile_roundtrip.rs`).

### Query engine integration

- `BM25Queryable` + `HNSWQueryable` traits generalize the four
  BM25 / HNSW constraint types over both naive and succinct
  index representations. `find!` / `pattern!` / `and!` queries
  work unmodified against either.
- `BM25Queryable::score_tolerance()` lets the constraint's
  score-equality check widen for quantized indexes
  automatically — lossless naive path keeps `f32::EPSILON`,
  `SuccinctBM25Index` returns `max_score / 65534`.
- `BM25ScoredPostings`, `SimilarToVectorScored`, and
  `SimilarToVectorHNSWScored` dedupe their score proposals by
  bit-pattern to avoid Cartesian blow-up when multiple docs
  share a score. Regression-locked at 1 k scale.

### Build-side

- `BM25Builder::build()` goes direct to `SuccinctBM25Index` in a
  single pass: sorts + dedups keys into `CompressedUniverse`
  first, then accumulates tf and scores keyed by universe code
  from the start. No insertion-order → universe-code remap, no
  per-term resort pass.
- `HNSWBuilder::build()` returns `SuccinctHNSWIndex` for the
  same "one blessed build method" ergonomic. Unlike BM25, there's
  no redundant-work win — HNSW still goes through the naive
  intermediate internally (necessary because levels are revealed
  incrementally, see node-major-vs-layer-major discussion) —
  but the public API now mirrors BM25's, and
  `SuccinctHNSWIndex::from_naive` remains available for callers
  who already hold a naive index.
- `HNSWBuilder::build_naive()` exposes the naive reference
  index (same ergonomics as `BM25Builder::build_naive`).
- Naive / oracle types (`BM25Index`, `HNSWIndex`, `FlatIndex`,
  `FlatBuilder`, `AttachedHNSWIndex`, `AttachedFlatIndex`) now
  live at `triblespace_search::testing::*`. The types are still
  physically declared in `bm25` / `hnsw` modules — their original
  paths are `#[doc(hidden)]` so rustdoc only surfaces the
  `testing::` path, signalling "reference-only, not a production
  API." The builders themselves (`BM25Builder`, `HNSWBuilder`)
  stay public at their canonical paths; the re-exports cover the
  naive forms they produce via `build_naive()`.
- Naive `to_bytes` / `try_from_bytes` on `BM25Index` /
  `HNSWIndex` / `FlatIndex` deleted along with their
  `BM25LoadError` / `HNSWLoadError` / `FlatLoadError` types
  (~400 LOC gone). The naive indexes are reference oracles
  only; persistence is always through the succinct forms.
  `byte_size()` accessors preserve the "succinct < naive
  baseline" regression guard without materializing bytes.
- `BM25Builder::build_naive()` / `build_naive_with_threads(n)`
  keep the naive insertion-order [`BM25Index`] available as a
  correctness oracle (score comparisons in tests) and for
  benchmarking the scoring loop in isolation from jerky
  packing. The naive path still supports sharded tf accumulation
  via `std::thread::scope`; byte-identical output across
  {1, 2, 3, 4, 8} threads.
- `SuccinctBM25Index::from_naive` retired — callers that had
  `SuccinctBM25Index::from_naive(&b.build())` collapse to
  `b.build()`.
- `HNSWBuilder` with deterministic level sampling
  (`.with_seed(u64)`).

### Tokenizers (`tokens::*`)

- `hash_tokens` — whitespace + lowercase + Blake3.
- `ngram_tokens(n)` — character n-grams, `n` namespaced into
  the hash. Compose with `hash_tokens` for prefix / fuzzy
  matching.
- `code_tokens` — camelCase / snake_case / acronym / digit
  splitter. Lowercased output shares term-space with
  `hash_tokens`.
- `bigram_tokens` — word-level bigrams, `"2w:"` namespace +
  `\0` word-boundary delimiter. Compose with `hash_tokens` for
  phrase-aware retrieval.

### Schemas

- `schemas::F32LE` — 32-byte `ValueSchema` for `f32` scores,
  used by the scored BM25 + similarity constraints.

### Examples (runnable)

- `query_demo`: text search + multi-term OR + value-as-term
  citation search.
- `compose_bm25_and_pattern`: BM25 + `pattern!` over a
  TribleSet in one `find!` / `and!`.
- `compose_hnsw_and_pattern`: vector search + `pattern!`
  composition.
- `blob_sizes_at_scale`: naive vs. SB25 blob sizes + parallel
  build speedup at 1 k / 5 k / 10 k / 50 k docs.
- `query_latency`: single-term + multi-term + HNSW latency
  p50/avg/p99 on 10 k / 50 k × 32 corpora.

### Docs

- `docs/DESIGN.md` — full design + 100 k worked example with
  measured build / size / latency numbers.
- `docs/QUERY_ENGINE_INTEGRATION.md` — constraint-trait surface.
- `docs/HNSW_GRAPH_ENCODING.md` — why the shipped CSR is
  the right HNSW graph encoding (not RING wavelet matrix) under
  current forward-only traversal, with the query patterns that
  would flip the decision.
- `docs/FACULTY_INTEGRATION.md` — worked `wiki_search.rs`
  rust-script template for consuming from faculties.

### Tests

159+ tests across unit, engine-integration (`find!`/`pattern!`
composition + regression guards), scale (1 k-doc equivalence +
size-regression + score-quantization top-10 preservation), and
`BlobStore` put/get round-trip. All pass.

## Not yet shipped

- Wavelet-matrix term table (would shrink the 9.6 MiB at 100 k;
  correctness-first is winning).
- RING-style wavelet matrix on the HNSW neighbour column (no
  win for forward-only traversal; see
  `docs/HNSW_GRAPH_ENCODING.md`).
- Direct `SuccinctBM25Index` builder that skips the naive
  intermediate (memory win at large build-time scale).
- Vector quantization for HNSW embeddings — intentionally
  caller-owned via the embedding schema.
- Published release / git push to
  [github.com/triblespace/triblespace-search](https://github.com/triblespace/triblespace-search)
  — awaiting JP's authorization.
