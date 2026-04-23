# HNSW graph encoding — why CSR, and when wavelet matrix would win

The design doc (`DESIGN.md`) originally pointed at
SuccinctArchive's RING wavelet-matrix plan for the HNSW layer
graph, with a "~3× savings vs trible-graph encoding" note. The
shipped `SuccinctGraph` uses two jerky `CompactVector`s in a
flat CSR layout instead. This doc explains the reasoning so
the wavelet plan doesn't get reinvented unnecessarily, and
notes where a wavelet matrix actually would buy something.

## What RING does for SuccinctArchive

SuccinctArchive encodes a TribleSet (subject, predicate, object
triples) using three wavelet matrices — one per column — plus
rank/select support. The wavelet machinery lets it answer
arbitrary "given two of {s, p, o}, find all matching third"
queries in `O(log n)` time without materializing intermediate
tables. That's the justification for the 3× factor: you pay for
three wavelet matrices because the query surface needs all three
projections.

## HNSW edges are unlabeled

HNSW's layer graph is a simple `(source, neighbour)` pair. There
is no predicate column — an edge is just "node A's neighbour on
layer L is node B". One column's worth of structure.

So the 3× factor never applied to HNSW. At most, a wavelet
matrix over the single neighbour column would save ~1× (the same
as the cost of not having one).

## Forward traversal doesn't need rank/select

The HNSW search loop:
1. Greedy descent on upper layers → at each step, read the
   current node's neighbour list and pick the closest to the
   query.
2. ef-search on layer 0 → expand a frontier by walking
   neighbour lists and tracking the best `ef` candidates.

Both phases need exactly **sequential neighbour enumeration**
for a given `(source, layer)`. No rank queries, no
"neighbours in byte range [a, b]", no "how often does node X
appear as a neighbour before position p".

A CSR layout answers sequential neighbour enumeration in
`O(k)` for `k` neighbours, where each neighbour is one
bit-packed CompactVector read (`O(1)` amortized). A wavelet
matrix answers the same query in `O(k log n)`. For HNSW's hot
path, CSR is strictly faster *and* smaller (no rank/select
auxiliary structure).

## When wavelet matrix *would* be the right move

Specific future queries that would justify the extra cost:

- **Reverse-neighbour lookup.** "Which nodes link to node X at
  layer L?" Needed if we ever want to implement online
  neighbour pruning or graph surgery without rebuilding. A
  wavelet matrix over the neighbour column answers this via
  `select(X, k)` for `k = 1..rank(end, X)` in `O(log n)` per
  result. CSR has to scan every node's list.

- **Range-constrained neighbour queries.** "Node X's neighbours
  whose doc-id falls in a given range." Useful if HNSW ever
  grows a bucket-level filter (e.g., namespace-restricted
  similarity). Wavelet-range is `O(log n)` per match; CSR is
  `O(k)`.

- **Compressed neighbour column via BWT-like ordering.** If the
  neighbour column is re-ordered to group identical values,
  the wavelet matrix's bitmaps compress heavily. For a
  small-world graph with high-degree hubs this could beat CSR
  on size. For HNSW's balanced `M=16, M0=32` layout, degrees
  are bounded and the win is marginal.

## The 2-ring variant we actually built

After the original CSR-vs-wavelet analysis above, JP pointed
out a result we'd missed in Arroyuelo et al. *The Ring* (ACM
TODS 2024, §4.4): for the **fixed-predicate / 2-column case**
(unlabeled graphs — exactly us), a single wavelet tree on the
high-endpoint column plus a run bitmap on the low-endpoint
column encodes the graph in **`n log m + O(n + m)`** bits vs
CSR's **`2n log m + O(n + m)`**. Each undirected edge is
stored once as `(low, high)` with `low ≤ high`; "enumerate
neighbours of `v`" unions two prefix-range reads.

We implemented this as `src/ring.rs::RingGraph` and
benchmarked it against `SuccinctGraph` head-to-head. The
original "wavelet matrix is bigger" intuition was wrong
because it was priced against a full 3-ring; the 2-ring
skips one of the wavelet matrices entirely.

## Measured numbers (commits c846e77, 55137c5)

Single layer, synthetic HNSW-shaped graph (each node picks
`M0` random neighbours symmetrically):

| n | M0 | CSR blob | Ring blob | size ratio |
|---:|---:|---:|---:|---:|
| 1 000 | 32 | 51 KiB | 28 KiB | **0.54×** |
| 10 000 | 32 | 714 KiB | 371 KiB | **0.52×** |
| 100 000 | 32 | 8.5 MiB | 4.4 MiB | **0.51×** |
| 100 000 | 16 | 4.4 MiB | 2.2 MiB | **0.50×** |

The ring hits the theoretical `½` bound at every scale. The
lack of runs in HNSW neighbour columns doesn't matter — the
gain comes from storing each edge once, not from entropy
compression.

**Per-neighbour enumeration cost** (microbench, same graph):

| backend | 1k/M0=32 | 100k/M0=32 |
|---|---:|---:|
| CSR (`CompactVector::get` × deg) | 81 ns | 130 ns |
| Ring (`WM::access` / `select` + `rank0`) | 8.2 µs | 21.4 µs |

~100–165× slower per-neighbour — wavelet-tree access is
`O(log m)` bit traversals, each with its own rank9 directory
lookup.

## End-to-end HNSW query latency (commit 55137c5)

`ring_vs_csr_hnsw` runs the same greedy-descent + ef-search
walk against both backends, distance eval through
`BlobCache<MemoryBlobStore, Embedding>` — the production
query path:

| n | dim | CSR walk avg | Ring walk avg | latency ratio |
|---:|---:|---:|---:|---:|
| 1 000 | 32 | 114 µs | 489 µs | 4.30× |
| 10 000 | 32 | 277 µs | 996 µs | 3.59× |
| 10 000 | 128 | 483 µs | 1.25 ms | 2.58× |
| 50 000 | 128 | 700 µs | 2.04 ms | 2.91× |

Top-10 overlap between the two backends drops from 10/10 at
1k to 6/10 at 50k. Not a bug — the graph topology is
identical, but ring emits neighbours in a different order
(low-side-first via contiguous range, then high-side via
`select`-walk), and HNSW's approximate ef-search resolves
ties against a min-heap that's sensitive to iteration order.
The final top-k can differ slightly between backends on the
same query.

### Why the ratio shrinks with corpus size

At large `n` each query's distance-eval cost grows
proportional to `visited_nodes × avg_degree × dim` while the
graph-traversal cost grows proportional to
`visited_nodes × log n` (ring) or `visited_nodes × M` (CSR).
For high-dim embeddings and many visited nodes, the distance
side eats more of the budget and the ring's log-factor
slowdown matters less in relative terms. For cold piles
where each `BlobCache::get` hits disk (ms-scale), the ratio
converges toward 1×.

## Decision

**Ship CSR as the default backend.** The ~3× query regression
for in-memory / warm-cache workloads is too steep for
general-purpose use, and at 1B corpus scale the graph is ~4 %
of total storage (embeddings dominate at 90 %+) — so even a
clean 2× graph-size win is ~2 % of total footprint. That's
not the right lever.

**Keep `RingGraph` alongside as a documented primitive.**
Callers whose pile lives on disk / network (cold fetches
dominate query time) or whose *branch-metadata* size is a
hard constraint can compose it themselves. We've tested it
across scales and round-trip, so the building block is
stable. If a real deployment pattern surfaces, we'd add a
`SuccinctRingHNSWIndex` variant that uses `RingGraph` in
place of `SuccinctGraph` — mechanical change, mostly new
blob-schema id.

**Lever to pull first** at 1B scale is embedding
quantization (`EmbeddingI8`, PQ, bfloat16). At dim=384+ the
embeddings are 90 %+ of total storage; a 4–16× shrink there
dwarfs any graph-encoding choice.

## When the ring would flip

- **Online neighbour pruning / graph surgery** — reverse
  lookup ("who points to X") is O(log m) via `select` on
  the wavelet tree; on CSR it requires scanning every
  node's list. If we ever add incremental `update()`
  without full rebuild, ring starts earning its weight.
- **Range-filtered similarity** — "nearest neighbours of X
  whose doc-id falls in [a, b]" — wavelet-tree 2D range
  query. No plans for this yet.
- **Branch metadata persistence** — if the index blob
  itself is what we're optimizing for (e.g., many versions
  stored per branch, each paying full graph size), the 2×
  saving amortizes across versions.

## See also

- `src/succinct.rs` — `SuccinctGraph` (CSR, default).
- `src/ring.rs` — `RingGraph` (2-ring, opt-in primitive).
- `examples/ring_vs_csr.rs` — microbench.
- `examples/ring_vs_csr_hnsw.rs` — end-to-end HNSW bench.
- jerky's `char_sequences::WaveletMatrix` — backs
  `RingGraph::high_column`.
