# triblespace-search — design

Two content-addressed index blobs that sit on top of a triblespace
pile: one for BM25-style lexical / associative retrieval, one for
approximate nearest-neighbour search over embeddings. Both follow
the same invariants:

1. **Content-addressed.** Same corpus → same blob hash. Rebuilds
   are free when nothing has changed; same content embedded with
   the same model yields the same blob everywhere in the pile.
2. **Rebuild-and-replace, no mutation.** `build(corpus) -> Handle`
   returns a fresh blob. The caller persists the handle wherever
   it belongs (branch metadata, commit metadata, a plain trible).
   Old index blobs stop being referenced on rebuild; squash
   reclaims them eventually.
3. **Zero-copy views via jerky.** The blob is a self-contained
   byte buffer; a `try_from_blob` produces a view that holds an
   `anybytes::Bytes` backing and answers queries without copying.
4. **Unordered-query shape.** Both indexes expose their query
   primitive as a triblespace constraint where the score is a
   bound variable:
     `bm25(?doc, ?term, ?score)`
     `similar(?doc, query_vec, ?score)`
   Callers combine with `and!` / `or!` / filters / sorts in the
   normal query engine.

## Term is a `Value`

BM25 in `triblespace-search` is not text-specific. Callers supply
terms as 32-byte `Value`s; the library provides a
`hash_tokens(&str) -> Vec<Value>` helper that Blake3-hashes
tokenized words but never forces it on the schema. Downstream uses:

| Term source                       | What this gets you                    |
| :-------------------------------- | :------------------------------------ |
| `hash(word)`                      | Classic text search.                  |
| entity `Id`                       | "Docs mentioning this person."        |
| tag `Id`                          | Tag-weighted search.                  |
| `hash(n-gram)`                    | Phrase search via query rewrite.      |
| fragment `Id`                     | "Docs citing this fragment."          |

The BM25 index is therefore a general `(doc: Id, term: Value, score)`
relation with IDF and length-normalized scoring baked in at build
time.

## `SuccinctBM25Index` — SB25 blob layout

Self-contained blob, zero-copy via `anybytes::Bytes`, bit-packed
via jerky. Schema id: `68C03764D04D05DF65E49589FBBA1441` (see
`succinct::SuccinctBM25Blob`).

```
[header              ] 280 B (fixed)
  magic                   u32    ; "SB25"
  version                 u16    ; FORMAT_VERSION = 3
  reserved                u16
  avg_doc_len             f32    ; for length normalization
  k1                      f32    ; BM25 tuning (default 1.5)
  b                       f32    ; BM25 tuning (default 0.75)
  max_score               f32    ; u16-quantization scale
  reserved                u32
  n_docs                  u64
  n_terms                 u64
  doc_lens_meta           32 B   ; CompactVectorMetaOnDisk
  postings_doc_idx_meta   32 B   ; CompactVectorMetaOnDisk
  postings_offsets_meta   32 B   ; CompactVectorMetaOnDisk
  postings_scores_meta    32 B   ; CompactVectorMetaOnDisk
  keys_meta               40 B   ; CompressedUniverseMetaOnDisk
  (section_offset, section_len) × 4 = 64 B
  tail pad                 4 B   ; so body starts at offset 280,
                                  ; keeping section offsets 8-aligned

[keys                ] variable         ; CompressedUniverse view:
                                        ; 4-byte fragment dictionary
                                        ; (sorted, deduped) + DACs-byte
                                        ; codes, one per unique key.
                                        ; `keys.access(code)` decodes
                                        ; the 32-byte RawValue.
[terms               ] n_terms × 32 B  ; sorted RawValue table
[doc_lens            ] variable         ; jerky CompactVector body
                                        ; width = ceil(log2(max_len + 1))
                                        ; indexed by universe-code order
[postings            ] variable         ; three jerky CompactVectors in
                                        ; one ByteArea:
                                        ;   doc_idx (width log2(n_docs+1),
                                        ;     stores universe codes, not
                                        ;     insertion indexes)
                                        ;   offsets (width log2(total+1))
                                        ;   scores  (width 16, u16-quantized)
```

Every body section starts on an 8-byte boundary; the header
carries a 4-byte tail pad to land the first body section on an
8-aligned absolute offset. `CompactVector` reinterprets its
backing buffer as `[u64]`, so misalignment would panic at load
time.

The four `CompactVectorMetaOnDisk` structs are a
[`zerocopy::IntoBytes`]-deriveable mirror of jerky's
`CompactVectorMeta` (jerky's upstream derives only `FromBytes`).
Four `u64` fields, 32 bytes each, `#[repr(C)]`. The
`CompressedUniverseMetaOnDisk` is the same trick for
`triblespace::core::blob::schemas::succinctarchive::CompressedUniverseMeta`
— five `u64` fields, 40 bytes, `#[repr(C)]`. Static asserts in
`succinct.rs` lock the size and layout equivalence.

Lookup algorithm:
1. Binary-search the term in `terms` (FixedBytesTable) → term
   index *t*.
2. Read `(offsets[t], offsets[t+1])` from the postings offsets
   CompactVector.
3. For each *i* in that range, read `doc_idx[i]` (a
   `CompressedUniverse` code) from the postings doc_idx
   CompactVector and `score[i]` from the quantized score
   section; decode the external key via
   `keys.access(doc_idx)`.

### What's already compressed (as of the current impl)

- `doc_lens` → bit-packed to `ceil(log2(max_len + 1))` bits.
  At 100k docs with avg_doc_len ≈ 180 and max ≈ 1024, ~10 bits
  instead of 32 — 3.2× savings on that section.
- `postings.doc_idx` → bit-packed to `ceil(log2(n_docs + 1))`.
  At 100k docs, 17 bits instead of 32 — 1.9× savings.
- `postings.offsets` → bit-packed likewise.
- `postings.scores` → u16-quantized via global `max_score`
  scale. Half-bucket error bound = `max_score / 2 × 65535`;
  `score_tolerance` on the index returns the full bucket
  `max_score / 65534` for constraint equality. 2× savings
  vs. f32 on the score section, top-10 preservation verified
  at 1k scale.

### What's still flat (deliberately)

- `terms` — 32 bytes each (Blake3 hash). We tried fragment-
  dictionary compression here (phase 2a) and it **grew** the
  section: Blake3 hashes have maximum entropy, so the 4-byte
  fragment dictionary overhead exceeded any code-length win.
  See `tests/scale_smoke.rs` and the phase 2a revert in git
  history for the actual numbers.

### What keys-side compression bought us

- `keys` is now a `CompressedUniverse` (Phase 2b). Measured via
  `cargo run --release --example blob_sizes_at_scale`:

  | corpus           | keys section vs 32 B flat |
  | :--------------- | :------------------------: |
  | scattered GenIds |        0.74×–0.81×         |
  | 11-byte-prefix   |        0.29×–0.32×         |

  "Scattered" is the pseudo-random `id_from_u64` with 16 trailing
  random bytes (worst-ish case — only the leading 16 zero bytes
  are shared). "Correlated" shares an 11-byte prefix and varies
  only the last 5 — simulates "one session of entity ids minted
  from a shared namespace seed."
- Whole-blob ratio moves too, but modestly: 0.48×→0.42× at 1 k
  docs with correlated keys; ~0.01×–0.02× improvement at 50 k
  because postings dominate the denominator.
- The architectural win is type-level: `keys.access(code)` goes
  through the same universe plumbing as every other `Value`
  table in the stack; range / prefix / membership queries over
  the keys universe compose for free.
### Open compression directions

- **Delta-encoded posting doc_idx** — posting lists are now
  universe-code-sorted (Phase 2b), so consecutive deltas
  compress further via Simple16 / ELF / VByte. Roughly halves
  the `doc_idx` section at Heaps-law corpora. This is the
  next-biggest win to chase — postings dominate the blob.
- **Non-uniform score quantization** — current u16 quantization
  uses a linear global `max_score` scale. A log-space or per-
  term scale would preserve more precision in the high-df
  (common-term) tail at the cost of a bigger header. Only
  worth it if ranking drift bites at larger corpora.
- **Wavelet matrix on the term table** — would let rank/select
  queries hit terms without a linear-compare binary search.
  For identification-only lookups the current FixedBytesTable
  binary search is competitive; this unlocks range queries
  over terms (useful for n-gram prefix scans).

## `SuccinctHNSWIndex` — SH25 blob layout

Self-contained blob, zero-copy via `anybytes::Bytes`. Schema id:
`7AFE59E7F895B23F05452FF7919E12E4` (see
`succinct::SuccinctHNSWBlob`).

```
[header              ] 152 B (fixed)
  magic                   u32    ; "SH25"
  version                 u16
  reserved                u16
  dim                     u32
  m                       u16    ; max neighbours on non-zero layers
  m0                      u16    ; max neighbours on layer 0
  max_level               u8
  reserved                u8
  has_entry_point         u8
  reserved                u8
  entry_point             u32
  n_nodes                 u64
  n_layers                u64
  graph_neighbours_meta   32 B   ; CompactVectorMetaOnDisk
  graph_offsets_meta      32 B   ; CompactVectorMetaOnDisk
  (section_offset, section_len) × 3 = 48 B

[doc_ids             ] n_nodes × 16 B
[vectors             ] n_nodes × dim × 4 B    ; flat f32 LE,
                                              ; L2-normalized at insert
[graph_bytes         ] variable               ; two CompactVectors in one
                                              ; ByteArea:
                                              ;   neighbours (width log2(n+1))
                                              ;   offsets    (width log2(E+1))
```

`graph_bytes` packs neighbour lists across all `(layer, node)`
pairs into a flat CSR: `offsets[L·(n+1) + i]` gives the start of
node *i*'s neighbour list on layer *L* inside `neighbours`. Nodes
absent from layer *L* encode as empty slices — search walks stay
correct because an empty neighbour list is a dead end, and the
search always enters from the top-level entry point.

Query algorithm (standard Malkov-Yashunin search):
1. Start at `entry_point` on `max_level`.
2. Greedy-descend layer-by-layer down to 1.
3. On layer 0, ef-width beam search, return top-k.

The succinct path re-implements the greedy + ef-search against
the bit-packed graph; see `SuccinctHNSWIndex::similar` in
`src/succinct.rs`.

### What's already compressed

- Graph `neighbours` → `ceil(log2(n_nodes + 1))` bits per
  neighbour index (17 bits at 100k nodes vs. 32 bits raw).
- Graph `offsets` → `ceil(log2(total_edges + 1))` bits per
  offset, which for `M=16` / `M0=32` averages similar savings.

### What's still flat

- `doc_ids` — 16-byte natural size.
- `vectors` — raw f32. Caller-owned data; compression is the
  caller's decision via their embedding schema choice (the
  crate itself stays agnostic).

### Open compression directions

- **Wavelet matrix on the neighbour column** — would give
  `rank` / `select` for free, which opens up reverse-neighbour
  lookup and range-filtered similarity. At current forward-only
  HNSW traversal the CSR `CompactVector` layout is both smaller
  and faster (no rank/select auxiliary overhead). See
  [`docs/HNSW_GRAPH_ENCODING.md`](HNSW_GRAPH_ENCODING.md) for
  the full tradeoff analysis and the query patterns that would
  flip the decision.
- **Vector quantization** — the caller owns the embedding
  schema. Future work: a `SuccinctHNSWBlob` variant (or a
  separate schema id) that stores `[u8; dim]` quantized vectors
  with per-component min/max scaling, or PQ codes. Keeps the
  graph untouched.

## Query engine integration

Both indexes expose their query as a `triblespace::Constraint`
implementation. A caller loads the blob once (cheap — mmap-backed
`anybytes::Bytes`) and produces a constraint by binding the index
handle + query term or vector.

```rust
let bm25 = BM25Index::try_from_blob(&blob)?;
let hnsw = HNSWIndex::try_from_blob(&blob)?;

for (doc, score) in find!(
    (doc: Id, score: f32),
    pattern!(&space, [
        { doc @ wiki::content: _ },
        bm25.contains(&doc, &term, &score, threshold: 0.1),
    ]),
) { ... }
```

Scores are bound variables; top-k is expressed by sorting results
in the caller (or via a future `top_k` combinator).

## What lives where

| Concern                       | Crate                   |
| :---------------------------- | :---------------------- |
| `Value`, `Id`, `TribleSet`    | triblespace             |
| Blob byte buffers (mmap)      | anybytes                |
| Succinct primitives           | jerky                   |
| BlobSchema + constraints      | **triblespace-search**  |
| Tokenizers (opt-in helpers)   | **triblespace-search**  |
| Caller-supplied embeddings    | downstream              |

`triblespace-search` does not depend on any embedding library.
Callers bring their own embeddings (local MiniLM via fastembed,
API-based Voyage/OpenAI, or anything that produces `f32` vectors
of a fixed dimensionality) and insert them into the pile under
an `Embedding<const D: usize>` schema they control.

## Non-goals (v1)

- Mutable updates. Rebuild is the only update path.
- Distributed/sharded indexes. Single-node first; sharding lives
  above the index API if/when it matters.
- Language-aware tokenization. `hash_tokens` is intentionally
  minimal; callers with real NLP needs tokenize themselves.
- Score combinations across BM25 + HNSW (hybrid search). Caller
  writes a `bm25(...) + alpha * similar(...)` combinator in the
  query.

## Worked example: 100 000 wiki fragments

Sizing exercise for the canonical downstream: indexing a Liora
pile of ≈ 100 k typst wiki fragments, average ≈ 180 words each
(≈ 300 raw tokens with punctuation). Numbers are back-of-envelope
for the *naive* (current) layout — the jerky succinct pass will
shrink the term-heavy sections.

### BM25 — size estimate

Assume after `hash_tokens`:
- `n_docs = 100 000`
- `avg_doc_len ≈ 180` unique tokens per doc after trim/dedup
- distinct terms across corpus `n_terms ≈ 300 000` (Heaps' law
  with β ≈ 0.5, k ≈ 30 for English-ish text)
- total postings `≈ 100 000 × 180 = 18 000 000` entries

Two columns: the naive `BM25Index::to_bytes` format (scaffold,
kept for testing self-consistency) and the landed SB25 format
(`SuccinctBM25Index::to_bytes`) with bit-packing + score
quantization.

| Section            | Per-entry | Count      | Naive bytes | SB25 bytes  |
| :----------------- | --------: | ---------: | ----------: | ----------: |
| header             | —         | —          |       32 B  |     280 B   |
| keys               |    32 B   | 100 000    |   3.2 MiB   | ~1.5–3.2 MiB|
| doc_lens           |     4 B   | 100 000    |   0.4 MiB   | ~0.12 MiB   |
| terms (sorted)     |    32 B   | 300 000    |   9.6 MiB   |  9.6 MiB    |
| postings_offsets   |     4 B   | 300 001    |   1.2 MiB   | ~0.6 MiB    |
| postings.doc_idx   |     4 B   | 18 000 000 |    72 MiB   | ~38 MiB     |
| postings.score     |     4 B   | 18 000 000 |    72 MiB   |    36 MiB   |
| **Total**          |           |            | **~159 MiB**| **~86 MiB** |

Every row computed the same way: the bit-packed sections use
`ceil(log2(n + 1))` bits per entry (doc_idx → 17 bits ≈ 2.12 B;
doc_lens at max ≈ 1024 → 10 bits ≈ 1.25 B; offsets at 18M max →
25 bits ≈ 3.1 B), and u16-quantized scores drop from 4 B to 2 B.

The `keys` range covers the fragment-dictionary compression
spread: near-worst-case (random 32-byte values, no shared 4-byte
fragments) ≈ raw 3.2 MiB plus a small DACs overhead; typical
GenId-keyed corpora with 16 bytes of zero padding and structured
trible bytes compress toward ~1.5 MiB. Neither end moves the
blob total noticeably — keys are ~2 % of the 86 MiB.

The **postings dominate** at 85 %+ of either blob. SB25's bit-
packed `doc_idx` plus u16 scores halves that section — the rest
of the footprint (keys, terms, docs_lens) is already as small as
the data allows without additional structure.

Term table is the second-largest chunk (9.6 MiB of 32-byte
Blake3 hashes). Phase 2a tried fragment-dictionary compression
here and it made the section **bigger** — maximum-entropy hashes
have no shared 4-byte fragments for the dictionary to exploit.
Left uncompressed.

### BM25 — build time

Build is O(total postings) with hashmap bookkeeping: `18 M`
insertions into the `HashMap<RawValue, HashMap<u32, u32>>` tf
table, then a sort over 300 k term hashes (32-byte compare).
On current laptop hardware:
- Hash-tokenize 100 k fragments × 180 tokens ≈ 18 M Blake3 hashes.
  Blake3 is ~3 GB/s on short inputs → ~0.5 s.
- Hashmap inserts: ~100 ns each × 18 M ≈ 1.8 s.
- Term sort: 300 k × log₂(300 k) × 32-byte compare ≈ 50 ms.
- Score computation: 18 M FMA-ish float ops ≈ 50 ms.

So **~3 s single-threaded** for the full corpus.
`BM25Builder::build_with_threads(n)` shards docs across `n`
scoped threads (std::thread::scope, no rayon dep) and merges
per-shard tf maps at the end. Observed speedups at 4 threads
on a laptop: ~1.2× at 10 k docs, ~1.3× at 50 k — the merge
cost stays serial and caps the win. Byte-identical output vs.
single-threaded. A term-partitioned variant would push further
but needs a routing hash per insert; filed as future work when
build-time actually bites.

### BM25 — query latency

`cargo run --release --example query_latency` on current laptop
hardware (10 k / 50 k docs with Zipf-ish vocab):

| Corpus              | Path   | p50     | avg     | p99     |
| :------------------ | :----- | ------: | ------: | ------: |
| 10 k × 64 tokens    | naive  |  125 ns |  162 ns |  459 ns |
| 10 k × 64 tokens    | SB25   |  875 ns | 1.06 µs | 4.04 µs |
| 50 k × 96 tokens    | naive  |  292 ns |  417 ns | 2.29 µs |
| 50 k × 96 tokens    | SB25   | 2.00 µs | 2.82 µs | 6.21 µs |

Single-term queries stay comfortably under 3 µs on the succinct
path even at 50 k docs — the original design estimate was a
generous upper bound. The naive path is ~6-7× faster than SB25
(flat `Vec` read + pre-baked `f32` vs. bit-unpacking a
`CompactVector` + dequantizing a u16 score); SB25 trades that
latency for ~2× smaller blobs on disk.

3-term `query_multi` (OR of independent posting lists):
~207 µs p50 at 50 k docs, dominated by the aggregation
`HashMap<Id, f32>`. For latency-critical multi-term queries an
`and!` via the engine is cheaper — merge-join size is
`min(|postings|)`, not `sum(|postings|)`.

### HNSW — size estimate

At `n = 100 000`, `dim = 384` (MiniLM), `M = 16`, M0 = 32:
- `doc_ids`: 100 k × 16 B = 1.6 MiB
- `embeddings`: 100 k × 384 × 4 B = **147 MiB** (dominates)
- `node_levels`: 100 k × 1 B = 0.1 MiB
- `neighbour_offsets`: 100 k × 8 × 4 B ≈ 3.2 MiB (avg ~8 layers
  at `log_M(100k) ≈ 4`, padded for uniform indexing)
- `neighbour_array`: ~100 k × M0 × 4 B ≈ 13 MiB (mostly layer 0)
- **Total ~165 MiB**, 89 % of which is raw f32 embeddings.

Jerky-succinct pass only helps the graph (`neighbour_array` +
`neighbour_offsets`) — the embeddings are the caller's data and
compressing them (f16, int8, PQ) is a separate design decision
the caller makes when choosing the `Embedding<const D: usize>`
schema. Graph-only savings are ≈ 10 MiB → ≈ 3 MiB (wavelet
matrix on source × target pairs), so the big-picture BM25+HNSW
blob for 100 k frags sits around 200 MiB and the succinct pass
takes it to ~180 MiB — embedding dominance means the compression
win is mostly on the BM25 side.

### HNSW — query latency

`query_latency` example on 5 k / 10 k × 32-dim vectors, top-10 +
ef=50:

| Corpus          | Path  | p50    | avg    | p99    |
| :-------------- | :---- | -----: | -----: | -----: |
| 5 k × 32        | naive | 54 µs  | 54 µs  | 62 µs  |
| 5 k × 32        | SH25  | 56 µs  | 57 µs  | 66 µs  |
| 10 k × 32       | naive | 55 µs  | 55 µs  | 60 µs  |
| 10 k × 32       | SH25  | 57 µs  | 57 µs  | 67 µs  |

SH25 sits within ~4 % of the naive latency. The previous
~1.4× gap (per-component `f32::from_le_bytes` + `Vec` alloc
per vector read) closed once `SuccinctHNSWIndex` started
caching an `anybytes::View<[f32]>` over the whole vectors
section. `vector(i)` is now `&view[i*dim .. (i+1)*dim]` with
no allocation and no decode loop.

### Takeaways

- Naive BM25 blob is ~1.5 KiB per doc — already shippable as a
  scaffold; SB25 halves that.
- Postings are the biggest lever; bit-packing + u16 scores
  already claimed it. The next step (delta-encoded `doc_idx`,
  wavelet-matrix term table) is incremental, not transformative.
- For HNSW, the interesting compression sits in *caller-owned*
  embedding bytes. This crate's succinct pass is about graph
  compactness + rank/select speed, not bulk size.
- At these scales a single-node mmap-backed blob is fine; the
  "distributed indexes" non-goal holds even at 1 M docs.
