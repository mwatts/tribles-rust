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

## Current size numbers

At 100 k nodes, `M0 = 32`, `dim = 384`, per the worked example
in `DESIGN.md`:

| Section             | CSR (current) | Wavelet matrix |
|:--------------------|--------------:|---------------:|
| neighbours column   |      ~6 MiB   |    ~8 MiB (+rank/select) |
| offsets column      |      ~1.6 MiB |    N/A          |
| **total graph**     |      **~7.6 MiB** | **~8 MiB** |

Wavelet matrices have a fixed rank/select overhead (≈ 25 % on
top of the raw bits for `Rank9SelIndex`). That overhead is what
makes them bigger than CSR at this scale. They'd start to win
only if the neighbour sequence has exploitable structure
(long runs of the same hub node), which doesn't happen in a
well-connected HNSW.

## Decision

- Keep CSR via two `CompactVector`s in `SuccinctGraph`.
- Treat the RING/wavelet-matrix note in the original
  `DESIGN.md` as rationale-that-didn't-survive-contact-with-
  the-query-pattern.
- Revisit if/when we add reverse-neighbour lookup or a
  range-filtered similarity constraint.

## See also

- `src/succinct.rs` — `SuccinctGraph` implementation.
- `docs/DESIGN.md` — SH25 blob layout; references this file
  for the graph-encoding reasoning.
- jerky's `char_sequences::WaveletMatrix` — the tool we'd
  reach for if one of the future queries above lands.
