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

## `SuccinctBM25Index` blob layout

Single self-contained blob, loaded zero-copy via jerky/anybytes.

```
[header              ] 32 B
  magic                   u32   ; "BM25"
  version                 u16
  reserved                u16
  n_docs                  u32
  n_terms                 u32
  avg_doc_len             f32   ; for length normalization
  k1                      f32   ; BM25 tuning (default 1.5)
  b                       f32   ; BM25 tuning (default 0.75)

[doc_id_table        ] n_docs × 16 B
  doc_id                  [u8; 16]   ; entity Ids in index-local order

[doc_len_table       ] n_docs × u32
  doc_len                 u32   ; token count (raw, pre-norm)

[term_table          ] n_terms × 32 B
  term_value              [u8; 32]   ; sorted ascending for binary search

[postings_offsets    ] (n_terms + 1) × u64
  cumulative offsets into the postings byte array

[postings            ] variable
  for each term, a sorted-doc-id posting list with per-doc score:
    doc_idx                jerky-encoded u32
    score                  f16 or quantized u16
```

Lookup algorithm:
1. Binary-search `term_value` in `term_table` → term index *t*.
2. Read postings slice `[offsets[t] .. offsets[t+1]]`.
3. Iterate `(doc_idx, score)` pairs; join with `doc_id_table` to
   get external `Id`s.

Write-only cost: one binary search per query term, one linear walk
per term's posting list. No random access beyond that.

### Later compression

The initial implementation stores `term_table` as a flat sorted
array and postings as packed `(doc_idx, score)` pairs. A
jerky-backed version swaps in a **wavelet matrix on `term_id`**
for rank/select over terms and on `doc_idx` for posting lookup,
plus ELF/Simple16 compression on score deltas. The API stays
identical.

The swap will follow jerky's own `Serializable` pattern:

```rust
// Example shape — mirrors jerky/src/serialization.rs.
#[repr(C)]
#[derive(zerocopy::FromBytes, KnownLayout, Immutable)]
pub struct BM25IndexMeta {
    // Fixed-size offsets + counts into the Bytes region.
    n_docs: u32,
    n_terms: u32,
    term_table_offset: u64,
    postings_offset: u64,
    // ... etc.
}

impl Serializable for BM25Index {
    type Meta = BM25IndexMeta;
    type Error = BM25LoadError;
    fn metadata(&self) -> Self::Meta { ... }
    fn from_bytes(meta: Self::Meta, bytes: Bytes) -> Result<Self, Self::Error> {
        // Slice `bytes` using offsets from `meta`, no copies.
    }
}
```

The blob body will be `[meta_bytes | arena_bytes]` with the meta
prefix at a fixed size so `try_from_bytes` can `zerocopy::from_bytes`
the prefix, wrap the tail as a `Bytes`, and build the view in
constant time.

## `SuccinctHNSWIndex` blob layout

Unlabeled-edge HNSW, inspired by SuccinctArchive's RING index but
saving 3x the wavelet matrices since HNSW edges carry no labels
(just node ↔ node).

```
[header              ] 64 B
  magic                   u32   ; "HNSW"
  version                 u16
  n_nodes                 u32
  n_layers                u8
  dim                     u32
  entry_point             u32   ; index of the top-layer entry node
  M                       u16   ; max neighbours per layer (non-zero level)
  M0                      u16   ; max neighbours on layer 0
  ef_construction         u16

[doc_id_table        ] n_nodes × 16 B
  doc_id                  [u8; 16]

[embeddings          ] n_nodes × dim × f32
  raw vectors (f32 for v1; f16 or PQ codes later)

[layer_directory     ] n_layers × 4 B
  layer_size              u32   ; number of nodes present on this layer

[layer_graphs        ] per layer, a wavelet matrix over
                       (source, neighbour) pairs — source is the
                       node's local index on that layer, neighbour
                       is its global node id. Rank/select gives
                       O(log n) neighbour iteration.
```

Query algorithm (standard HNSW greedy search):
1. Start at `entry_point` on the top layer.
2. Greedy-descend: at each layer, iterate current node's
   neighbours, keep the closest to the query. Walk until no
   improvement.
3. On layer 0, do ef-width beam search, return top-k.

### Later compression

V1 stores embeddings as raw `f32` and layer graphs via a simple
CSR (sources offset + dense neighbour array). The jerky-backed
version replaces CSR with one wavelet matrix per layer for
rank-select neighbour enumeration, and may quantize embeddings
(f16, int8, or PQ).

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

Naive layout (current `to_bytes`):
| Section             | Per-entry | Count         | Bytes         |
| :------------------ | --------: | ------------: | ------------: |
| header              | —         | —             |            32 |
| doc_ids             |     16 B  | 100 000       |       1.6 MiB |
| doc_lens            |      4 B  | 100 000       |       0.4 MiB |
| terms (sorted)      |     32 B  | 300 000       |       9.6 MiB |
| postings_offsets    |      4 B  | 300 001       |       1.2 MiB |
| postings            |      8 B  | 18 000 000    |       144 MiB |
| **Total**           |           |               |   **~157 MiB**|

The **postings table dominates** — 91 % of the blob. That's where
the jerky succinct pass has the biggest lever: a wavelet matrix
over `doc_idx` plus quantized (`f16`) scores would cut it to
`≈ 18 M × 4 B ≈ 72 MiB` even without delta coding, or
`≈ 18 M × (log₂(100k) / 8 + 2) B ≈ 36 MiB` with bit-packed
doc_idx. Rough target: 40–60 MiB for the full blob.

Term table is the second-largest chunk (9.6 MiB of 32-byte hashes).
A wavelet matrix over byte-quantized prefixes + flat tail trades
some rank/select cost for compression, but 10 MiB is small enough
that the naive version is fine until someone complains.

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

So **~3 s single-threaded** for the full corpus. A multi-threaded
builder (shard by doc, merge the per-shard tf maps) would cut
this to ~1 s on an 8-core laptop. Not yet implemented.

### BM25 — query latency

Single-term query:
- Binary-search `terms` (32-byte compare): ~19 iterations × ~100 ns
  = ~2 μs.
- Scan posting list: average `18 M / 300 k = 60` postings per
  term, each 8 B contiguous — ~0.5 μs of memory-bandwidth-bound
  scan.
- Total: **~3 μs per term**, before join overhead.

Intersection of two terms (`find! and!`): query engine joins the
smaller posting list against the larger; real cost is
O(min(|A|, |B|)) per the usual merge-join. At 60 postings per
term, a 2-term `and!` is still comfortably sub-10 μs.

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

### Takeaways

- Naive BM25 blob is ~1.5 KiB per doc — already shippable.
- Postings table is where succinctness earns its keep (3–4×
  expected).
- For HNSW, the interesting compression sits in *caller-owned*
  embedding bytes. This crate's succinct pass is about graph
  compactness + rank/select speed, not bulk size.
- At these scales a single-node mmap-backed blob is fine; the
  "distributed indexes" non-goal holds even at 1 M docs.
