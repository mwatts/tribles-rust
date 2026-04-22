//! Approximate nearest-neighbour search over caller-supplied
//! embeddings.
//!
//! [`HNSWIndex`] is the naive layered-graph implementation
//! (Malkov & Yashunin 2018). It's the builder + in-memory
//! representation; convert to [`crate::succinct::SuccinctHNSWIndex`]
//! and use `SuccinctHNSWBlob` for the content-addressed on-pile
//! form.
//!
//! [`FlatIndex`] is the brute-force exact cosine baseline — useful
//! for ≤ 100k docs, for ground-truth recall checks, and for
//! doctest examples without the graph build overhead.
//!
//! # Build and query
//!
//! ```
//! # use triblespace::core::blob::MemoryBlobStore;
//! # use triblespace::core::id::Id;
//! # use triblespace::core::repo::BlobStore;
//! # use triblespace::core::value::schemas::hash::Blake3;
//! # use triblespace_search::hnsw::FlatBuilder;
//! # use triblespace_search::schemas::put_embedding;
//! let mut store = MemoryBlobStore::<Blake3>::new();
//! let h1 = put_embedding::<_, Blake3>(&mut store, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
//! let h2 = put_embedding::<_, Blake3>(&mut store, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
//! let h3 = put_embedding::<_, Blake3>(&mut store, vec![0.9, 0.1, 0.0, 0.0]).unwrap();
//!
//! let mut b = FlatBuilder::new(4);
//! b.insert_id(Id::new([1; 16]).unwrap(), h1);
//! b.insert_id(Id::new([2; 16]).unwrap(), h2);
//! b.insert_id(Id::new([3; 16]).unwrap(), h3);
//! let idx = b.build();
//!
//! let reader = store.reader().unwrap();
//! let query = vec![1.0, 0.0, 0.0, 0.0];
//! let hits = idx.attach(&reader).similar_ids(&query, 2).unwrap();
//! assert_eq!(hits.len(), 2);
//! // doc 1 is an exact match, doc 3 nearly so.
//! assert_eq!(hits[0].0, Id::new([1; 16]).unwrap());
//! assert_eq!(hits[1].0, Id::new([3; 16]).unwrap());
//! ```

use triblespace::core::id::{Id, RawId};
use triblespace::core::value::schemas::hash::{Blake3, Handle};
use triblespace::core::value::{RawValue, Value, ValueSchema};

use crate::schemas::Embedding;

// ── HNSW blob byte format ────────────────────────────────────────────
//
// No magic bytes, no version field: the blob-level type
// (a typed `BlobSchema` / handle on the pile side, or the
// `HNSWIndex::try_from_bytes` entry point itself) is the
// identity. A breaking format change mints a new schema ID
// and therefore a new type, so the compiler polices it.

// ── Proper HNSW graph (layered, approximate k-NN) ─────────────────

/// Per-node state during build: vector lives inline so graph
/// construction can compute distances without touching a blob
/// store. `build()` strips the vector and produces
/// [`HNSWIndexNode`].
#[derive(Debug)]
struct HNSWNode {
    vector: Vec<f32>,
    #[allow(dead_code)]
    level: u8,
    neighbors: Vec<Vec<u32>>,
}

/// Post-build per-node state. No vector — queries resolve
/// embeddings through a caller-supplied blob store via the
/// parallel `handles` table.
#[derive(Debug)]
struct HNSWIndexNode {
    #[allow(dead_code)]
    level: u8,
    neighbors: Vec<Vec<u32>>,
}

/// Builder for a proper layered-graph HNSW index.
///
/// Implements the incremental insert from Malkov & Yashunin
/// (2018) with the standard level-sampling + ef-search + simple
/// neighbour-selection heuristic. Parameters follow the paper's
/// defaults unless overridden on the builder.
pub struct HNSWBuilder {
    dim: usize,
    m: u16,
    m0: u16,
    ef_construction: u16,
    /// Level-sampling multiplier `m_L = 1 / ln(M)`.
    level_mult: f32,
    /// SplitMix64 state for deterministic level sampling.
    rng: u64,
    /// Per-node state, inclusive of the inline vector used for
    /// graph-construction distance computations. The vectors
    /// get stripped when `build()` consumes the builder —
    /// they don't survive into `HNSWIndex`.
    nodes: Vec<HNSWNode>,
    keys: Vec<RawValue>,
    /// Content-addressed handle for each node's embedding.
    /// Parallel index with `keys` and `nodes`; the final
    /// `HNSWIndex` keeps only this table for query-time
    /// resolution.
    handles: Vec<Value<Handle<Blake3, Embedding>>>,
    entry_point: Option<u32>,
    max_level: u8,
}

impl HNSWBuilder {
    /// Create a fresh builder with `dim`-dimensional vectors and
    /// default HNSW parameters (`M = 16`, `M0 = 2*M = 32`,
    /// `ef_construction = 200`). The deterministic PRNG seed
    /// starts at `0xC0FFEE_HNSW`; override via [`with_seed`] for
    /// reproducible but differently-ordered builds.
    ///
    /// [`with_seed`]: Self::with_seed
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "HNSWBuilder: dim must be > 0");
        let m = 16u16;
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction: 200,
            level_mult: 1.0 / (m as f32).ln(),
            rng: 0xC0FFEEu64,
            nodes: Vec::new(),
            keys: Vec::new(),
            handles: Vec::new(),
            entry_point: None,
            max_level: 0,
        }
    }

    /// Override `M` (max neighbours on non-zero layers). `M0`
    /// defaults to `2 * M` unless overridden separately.
    pub fn m(mut self, m: u16) -> Self {
        assert!(m >= 2, "HNSWBuilder: M must be ≥ 2");
        self.m = m;
        self.m0 = m * 2;
        self.level_mult = 1.0 / (m as f32).ln();
        self
    }

    /// Override `M0` (max neighbours on layer 0). Must be ≥ M.
    pub fn m0(mut self, m0: u16) -> Self {
        assert!(m0 >= self.m, "HNSWBuilder: M0 must be ≥ M");
        self.m0 = m0;
        self
    }

    /// Override `ef_construction` (search width during insert).
    pub fn ef_construction(mut self, ef: u16) -> Self {
        assert!(ef >= 1, "HNSWBuilder: ef_construction must be ≥ 1");
        self.ef_construction = ef;
        self
    }

    /// Override the level-sampling PRNG seed for reproducibility
    /// across runs.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = seed;
        self
    }

    /// Sample a new node level from `⌊-ln(U) * m_L⌋`.
    fn sample_level(&mut self) -> u8 {
        // SplitMix64 step.
        self.rng = self.rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map to uniform (0, 1] so `ln` is defined.
        let u = ((z >> 11) as f64 / (1u64 << 53) as f64).max(f64::MIN_POSITIVE);
        let l = (-u.ln() * self.level_mult as f64).floor() as i32;
        l.clamp(0, u8::MAX as i32) as u8
    }

    /// Insert a document keyed by `key`, referenced by
    /// `handle` (a content-addressed pointer to the
    /// [`Embedding`] blob), with the raw `vec` supplied for
    /// build-time distance computations. The builder keeps the
    /// vector in RAM during graph construction and strips it at
    /// [`build`]; the final [`HNSWIndex`] only carries the
    /// handle, so embeddings live in the pile's blob store and
    /// dedupe across indexes. `key` is any 32-byte triblespace
    /// value (GenId / ShortString / tag / composite — see
    /// [`insert_id`] / [`insert_value`] for convenience
    /// wrappers). The vector is L2-normalized in place before
    /// distance computation, so the index treats its metric as
    /// cosine similarity; the stored `handle` is expected to
    /// point at an already-normalized embedding (the
    /// [`put_embedding`] helper normalizes before put).
    ///
    /// [`build`]: Self::build
    /// [`insert_id`]: Self::insert_id
    /// [`insert_value`]: Self::insert_value
    /// [`put_embedding`]: crate::schemas::put_embedding
    pub fn insert(
        &mut self,
        key: RawValue,
        handle: Value<Handle<Blake3, Embedding>>,
        mut vec: Vec<f32>,
    ) -> Result<(), DimMismatch> {
        if vec.len() != self.dim {
            return Err(DimMismatch {
                expected: self.dim,
                got: vec.len(),
            });
        }
        normalize(&mut vec);
        let new_level = self.sample_level();
        let new_idx = self.nodes.len() as u32;

        // Descend from entry_point down to new_level + 1 using
        // greedy 1-step search.
        let mut curr = self.entry_point;
        if let Some(mut cnode) = curr {
            for lvl in ((new_level + 1)..=self.max_level).rev() {
                cnode = self.greedy_search_layer(&vec, cnode, lvl);
            }
            curr = Some(cnode);
        }

        // Allocate the new node before connecting so neighbour
        // indexes are stable.
        self.nodes.push(HNSWNode {
            vector: vec.clone(),
            level: new_level,
            neighbors: vec![Vec::new(); new_level as usize + 1],
        });
        self.keys.push(key);
        self.handles.push(handle);

        // Connect from new_level down to 0.
        if let Some(start) = curr {
            let mut entry = start;
            for lvl in (0..=new_level.min(self.max_level)).rev() {
                let cap = if lvl == 0 { self.m0 } else { self.m } as usize;
                let candidates = self.search_layer(&vec, entry, self.ef_construction as usize, lvl);
                let selected = Self::select_neighbours(&candidates, cap);

                // Bidirectional edges.
                for &n in &selected {
                    self.nodes[new_idx as usize].neighbors[lvl as usize].push(n);
                    self.nodes[n as usize].neighbors[lvl as usize].push(new_idx);
                }
                // Prune the new node's layer-list and the new
                // neighbours' lists to the layer cap.
                self.prune_neighbours(new_idx, lvl, cap);
                for &n in &selected {
                    self.prune_neighbours(n, lvl, cap);
                }

                // Pick the best candidate as entry for the next
                // (lower) layer.
                if let Some((best, _)) = candidates
                    .iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                {
                    entry = *best;
                }
            }
        }

        if new_level > self.max_level || self.entry_point.is_none() {
            self.max_level = new_level;
            self.entry_point = Some(new_idx);
        }
        Ok(())
    }

    /// Convenience: insert keyed by a triblespace [`Id`].
    pub fn insert_id(
        &mut self,
        doc_id: Id,
        handle: Value<Handle<Blake3, Embedding>>,
        vec: Vec<f32>,
    ) -> Result<(), DimMismatch> {
        let mut raw = [0u8; 32];
        let id_bytes: &RawId = doc_id.as_ref();
        raw[16..32].copy_from_slice(id_bytes);
        self.insert(raw, handle, vec)
    }

    /// Convenience: insert keyed by a typed [`Value<S>`].
    pub fn insert_value<S: ValueSchema>(
        &mut self,
        key: Value<S>,
        handle: Value<Handle<Blake3, Embedding>>,
        vec: Vec<f32>,
    ) -> Result<(), DimMismatch> {
        self.insert(key.raw, handle, vec)
    }

    /// Consume the builder and produce a succinct HNSW index,
    /// ready to `put` into a pile or query directly. This is
    /// the production path — the naive in-memory [`HNSWIndex`]
    /// is kept only as a reference oracle (see
    /// [`build_naive`][Self::build_naive]).
    pub fn build(self) -> crate::succinct::SuccinctHNSWIndex {
        crate::succinct::SuccinctHNSWIndex::from_naive(&self.build_naive())
            .expect("from_naive cannot fail on a valid HNSWIndex built by HNSWBuilder")
    }

    /// Naive layered-graph reference index. Strips the inline
    /// build-time vectors — only the handles survive; embeddings
    /// are resolved at query time through the caller-supplied
    /// blob store. Kept public as a correctness oracle for tests
    /// validating the succinct form, and as an intermediate when
    /// callers already hold a naive index and want
    /// [`SuccinctHNSWIndex::from_naive`][crate::succinct::SuccinctHNSWIndex::from_naive]
    /// directly. Most callers want [`build`][Self::build].
    pub fn build_naive(self) -> HNSWIndex {
        let nodes: Vec<HNSWIndexNode> = self
            .nodes
            .into_iter()
            .map(|n| HNSWIndexNode {
                level: n.level,
                neighbors: n.neighbors,
            })
            .collect();
        HNSWIndex {
            dim: self.dim,
            m: self.m,
            m0: self.m0,
            nodes,
            keys: self.keys,
            handles: self.handles,
            entry_point: self.entry_point,
            max_level: self.max_level,
        }
    }

    // ── HNSW primitives (shared with the immutable index) ────────

    /// Walk greedily to the node with minimum distance to `q` on
    /// `layer` starting from `entry`. O(neighbours_on_layer)
    /// per step. Used for intermediate layers during both insert
    /// and search.
    fn greedy_search_layer(&self, q: &[f32], entry: u32, layer: u8) -> u32 {
        let mut curr = entry;
        let mut curr_dist = cosine_dist(q, &self.nodes[curr as usize].vector);
        loop {
            let mut changed = false;
            let node = &self.nodes[curr as usize];
            let Some(neigh) = node.neighbors.get(layer as usize) else {
                return curr;
            };
            for &n in neigh {
                let d = cosine_dist(q, &self.nodes[n as usize].vector);
                if d < curr_dist {
                    curr_dist = d;
                    curr = n;
                    changed = true;
                }
            }
            if !changed {
                return curr;
            }
        }
    }

    /// Standard HNSW layer ef-search. Returns a list of
    /// `(node_idx, distance)` pairs, up to `ef` of them.
    fn search_layer(&self, q: &[f32], entry: u32, ef: usize, layer: u8) -> Vec<(u32, f32)> {
        use std::collections::BinaryHeap;

        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        visited.insert(entry);
        let d0 = cosine_dist(q, &self.nodes[entry as usize].vector);
        let mut candidates: BinaryHeap<MinDist> = BinaryHeap::new();
        candidates.push(MinDist {
            idx: entry,
            dist: d0,
        });
        let mut results: BinaryHeap<MaxDist> = BinaryHeap::new();
        results.push(MaxDist {
            idx: entry,
            dist: d0,
        });
        while let Some(c) = candidates.pop() {
            let farthest = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && results.len() >= ef {
                break;
            }
            let node = &self.nodes[c.idx as usize];
            let Some(neigh) = node.neighbors.get(layer as usize) else {
                continue;
            };
            for &n in neigh {
                if !visited.insert(n) {
                    continue;
                }
                let d = cosine_dist(q, &self.nodes[n as usize].vector);
                let farthest = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                if d < farthest || results.len() < ef {
                    candidates.push(MinDist { idx: n, dist: d });
                    results.push(MaxDist { idx: n, dist: d });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        results.into_iter().map(|m| (m.idx, m.dist)).collect()
    }

    /// Pick the `cap` closest candidates. The paper's simple
    /// heuristic — good enough for typical embedding spaces and
    /// the simplest thing to unit-test. The "extended" heuristic
    /// that considers inter-candidate distances can swap in
    /// later behind the same function signature.
    fn select_neighbours(candidates: &[(u32, f32)], cap: usize) -> Vec<u32> {
        let mut sorted: Vec<&(u32, f32)> = candidates.iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.into_iter().take(cap).map(|&(i, _)| i).collect()
    }

    /// Trim `node`'s layer-`layer` neighbour list to `cap`
    /// entries, keeping the closest by distance.
    fn prune_neighbours(&mut self, node: u32, layer: u8, cap: usize) {
        // Borrow-checker dance: snapshot the neighbour ids and
        // the node's vector so we can score against `self.nodes`
        // without holding a mut-borrow on the list.
        let list_snapshot: Vec<u32> = self.nodes[node as usize].neighbors[layer as usize].clone();
        if list_snapshot.len() <= cap {
            // Already small enough; just dedupe in place.
            let list = &mut self.nodes[node as usize].neighbors[layer as usize];
            list.sort_unstable();
            list.dedup();
            return;
        }
        let q = self.nodes[node as usize].vector.clone();
        let mut scored: Vec<(u32, f32)> = list_snapshot
            .iter()
            .map(|&n| (n, cosine_dist(&q, &self.nodes[n as usize].vector)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let list = &mut self.nodes[node as usize].neighbors[layer as usize];
        list.clear();
        list.extend(scored.into_iter().take(cap).map(|(i, _)| i));
        list.sort_unstable();
        list.dedup();
    }
}

/// Immutable approximate-nearest-neighbour index built via
/// [`HNSWBuilder`]. Query performance is sub-linear in the
/// corpus size (O(log n · degree) typical) — the trade-off is
/// a larger up-front build cost than [`FlatIndex`] and slightly
/// approximate recall.
#[derive(Debug)]
pub struct HNSWIndex {
    dim: usize,
    m: u16,
    m0: u16,
    /// Post-build per-node state. Neighbour lists survive; the
    /// vectors were stripped — distance evaluations resolve
    /// handles through a caller-supplied blob store.
    nodes: Vec<HNSWIndexNode>,
    keys: Vec<RawValue>,
    handles: Vec<Value<Handle<Blake3, Embedding>>>,
    entry_point: Option<u32>,
    max_level: u8,
}

impl HNSWIndex {
    /// Vector dimensionality configured at build time.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.keys.len()
    }
    /// Max neighbours per non-zero layer.
    pub fn m(&self) -> u16 {
        self.m
    }
    /// Max neighbours on layer 0.
    pub fn m0(&self) -> u16 {
        self.m0
    }
    /// Highest layer a node was inserted at.
    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    /// Internal keys table: `keys()[i]` is the stored 32-byte
    /// [`RawValue`] for internal node index `i`. Exposed so
    /// succinct re-encoders can snapshot the table without
    /// roundtripping through `similar()`.
    pub fn keys(&self) -> &[RawValue] {
        &self.keys
    }

    /// Level node `i` was sampled into.
    pub fn node_level(&self, i: usize) -> Option<u8> {
        self.nodes.get(i).map(|n| n.level)
    }

    /// Neighbours of node `i` on `layer`. Empty slice if the
    /// node wasn't inserted at that layer.
    pub fn node_neighbours(&self, i: usize, layer: u8) -> &[u32] {
        self.nodes
            .get(i)
            .and_then(|n| n.neighbors.get(layer as usize))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// The stored embedding-handle table. Paired index-wise
    /// with [`keys`]; `handles()[i]` points at the blob that
    /// holds the embedding for internal node `i`.
    ///
    /// [`keys`]: Self::keys
    pub fn handles(&self) -> &[Value<Handle<Blake3, Embedding>>] {
        &self.handles
    }

    /// Current entry-point node index (the last inserted node
    /// at `max_level`), or `None` if the index is empty.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Attach a blob store to this index, returning a queryable
    /// view. The typical load path:
    ///
    /// ```ignore
    /// let idx: HNSWIndex = reader.get::<_, HNSWBlob>(handle)?;
    /// let view = idx.attach(&reader);
    /// view.similar(&query, k, ef)?;
    /// ```
    ///
    /// The view wraps `store` in an internal
    /// [`BlobCache`][c] keyed on `Handle<Blake3, Embedding>`,
    /// so the HNSW walk's repeated visits to the same node
    /// (greedy + ef-search revisit neighbours) deserialize
    /// each embedding at most once per view. `B: Clone` is
    /// required because the cache owns the store; the
    /// typical readers (`MemoryBlobStoreReader`, pile
    /// readers) are cheap-clone (just a `ReadHandle`
    /// refcount bump).
    ///
    /// [c]: triblespace::core::blob::BlobCache
    pub fn attach<'a, B>(&'a self, store: &B) -> AttachedHNSWIndex<'a, B>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3> + Clone,
    {
        AttachedHNSWIndex {
            index: self,
            cache: triblespace::core::blob::BlobCache::new(store.clone()),
        }
    }

    /// Theoretical size of the naive flat-array serialization in
    /// bytes — kept as a baseline to regression-check that the
    /// succinct HNSW blob actually saves space.
    ///
    /// Layout: 24 B header + `n_nodes × 32 B` keys + `n_nodes ×
    /// 32 B` handles + `n_nodes × 1 B` levels + per-node offset
    /// table (`(max_level + 2) × 4 B` stride) + total neighbours
    /// × 4 B.
    pub fn byte_size(&self) -> usize {
        let n = self.nodes.len();
        let entries_per_node = (self.max_level as usize) + 2;
        let total_neighbours: usize = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|l| l.len()).sum::<usize>())
            .sum();
        24 + n * 32 + n * 32 + n + n * entries_per_node * 4 + total_neighbours * 4
    }
}

/// A [`HNSWIndex`] paired with the blob store its handles
/// resolve against — produced by [`HNSWIndex::attach`]. All
/// `similar_*` methods live here; the bare [`HNSWIndex`] only
/// exposes metadata and the blob format.
///
/// The view owns a [`BlobCache`][c] over the provided store,
/// specialized to `(Embedding, View<[f32]>)`. HNSW graph walks
/// revisit neighbour nodes repeatedly — the cache collapses
/// those into a single blob-fetch + deserialize per node per
/// view lifetime.
///
/// [c]: triblespace::core::blob::BlobCache
pub struct AttachedHNSWIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<Blake3>,
{
    index: &'a HNSWIndex,
    cache: triblespace::core::blob::BlobCache<B, Blake3, Embedding, anybytes::View<[f32]>>,
}

impl<'a, B> AttachedHNSWIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<Blake3>,
{
    /// The inner index (back-reference for metadata queries).
    pub fn index(&self) -> &HNSWIndex {
        self.index
    }

    /// Approximate top-k nearest neighbours to `query` under
    /// cosine similarity. `ef` tunes the search width (larger =
    /// better recall at higher cost); pass `None` to default to
    /// `k`.
    ///
    /// Handle lookups happen on every distance evaluation along
    /// the HNSW walk. Wrap the store in a
    /// [`BlobCache`][c] before attaching when the same view will
    /// be queried repeatedly — it amortizes the per-handle
    /// deserialize across queries.
    ///
    /// [c]: triblespace::core::blob::BlobCache
    pub fn similar(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(RawValue, f32)>, B::GetError<anybytes::view::ViewError>> {
        if query.len() != self.index.dim || k == 0 {
            return Ok(Vec::new());
        }
        let Some(entry) = self.index.entry_point else {
            return Ok(Vec::new());
        };
        let mut q = query.to_vec();
        normalize(&mut q);
        let ef = ef.unwrap_or(k).max(k);

        let mut curr = entry;
        for lvl in (1..=self.index.max_level).rev() {
            curr = self.greedy_search_layer(&q, curr, lvl)?;
        }
        let candidates = self.search_layer(&q, curr, ef, 0)?;
        let mut ranked: Vec<(RawValue, f32)> = candidates
            .into_iter()
            .map(|(i, dist)| (self.index.keys[i as usize], 1.0 - dist))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(k);
        Ok(ranked)
    }

    /// [`similar`] with GenId-typed keys decoded back to
    /// [`Id`]; other schemas are dropped.
    ///
    /// [`similar`]: Self::similar
    pub fn similar_ids(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(Id, f32)>, B::GetError<anybytes::view::ViewError>> {
        Ok(self
            .similar(query, k, ef)?
            .into_iter()
            .filter_map(|(raw, s)| {
                if raw[0..16] != [0u8; 16] {
                    return None;
                }
                let id_bytes: RawId = raw[16..32].try_into().ok()?;
                Id::new(id_bytes).map(|id| (id, s))
            })
            .collect())
    }

    fn dist_to(
        &self,
        q: &[f32],
        i: u32,
    ) -> Result<f32, B::GetError<anybytes::view::ViewError>> {
        let handle = self.index.handles[i as usize];
        let view = self.cache.get(handle)?;
        Ok(cosine_dist(q, view.as_ref().as_ref()))
    }

    fn greedy_search_layer(
        &self,
        q: &[f32],
        entry: u32,
        layer: u8,
    ) -> Result<u32, B::GetError<anybytes::view::ViewError>> {
        let mut curr = entry;
        let mut curr_dist = self.dist_to(q, curr)?;
        loop {
            let mut changed = false;
            let node = &self.index.nodes[curr as usize];
            let Some(neigh) = node.neighbors.get(layer as usize) else {
                return Ok(curr);
            };
            let neigh = neigh.clone();
            for n in neigh {
                let d = self.dist_to(q, n)?;
                if d < curr_dist {
                    curr_dist = d;
                    curr = n;
                    changed = true;
                }
            }
            if !changed {
                return Ok(curr);
            }
        }
    }

    fn search_layer(
        &self,
        q: &[f32],
        entry: u32,
        ef: usize,
        layer: u8,
    ) -> Result<Vec<(u32, f32)>, B::GetError<anybytes::view::ViewError>> {
        use std::collections::BinaryHeap;
        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        visited.insert(entry);
        let d0 = self.dist_to(q, entry)?;
        let mut candidates: BinaryHeap<MinDist> = BinaryHeap::new();
        candidates.push(MinDist {
            idx: entry,
            dist: d0,
        });
        let mut results: BinaryHeap<MaxDist> = BinaryHeap::new();
        results.push(MaxDist {
            idx: entry,
            dist: d0,
        });
        while let Some(c) = candidates.pop() {
            let farthest = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && results.len() >= ef {
                break;
            }
            let neigh = {
                let node = &self.index.nodes[c.idx as usize];
                let Some(neigh) = node.neighbors.get(layer as usize) else {
                    continue;
                };
                neigh.clone()
            };
            for n in neigh {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.dist_to(q, n)?;
                let farthest = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                if d < farthest || results.len() < ef {
                    candidates.push(MinDist { idx: n, dist: d });
                    results.push(MaxDist { idx: n, dist: d });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        Ok(results.into_iter().map(|m| (m.idx, m.dist)).collect())
    }
}

/// Cosine distance = 1 - dot(a, b) for pre-normalized vectors.
pub(crate) fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

/// Min-heap wrapper: smaller distance = higher priority.
#[derive(Clone, Copy)]
struct MinDist {
    idx: u32,
    dist: f32,
}
impl PartialEq for MinDist {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for MinDist {}
impl PartialOrd for MinDist {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinDist {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // Invert so BinaryHeap (max-heap) behaves as min-heap
        // over distance.
        o.dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Max-heap wrapper: larger distance = higher priority (for
/// evicting the farthest).
#[derive(Clone, Copy)]
struct MaxDist {
    idx: u32,
    dist: f32,
}
impl PartialEq for MaxDist {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for MaxDist {}
impl PartialOrd for MaxDist {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxDist {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Caller tried to insert a vector whose length disagrees with
/// the index's configured dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimMismatch {
    pub expected: usize,
    pub got: usize,
}

impl std::fmt::Display for DimMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "embedding dimensionality mismatch: expected {}, got {}",
            self.expected, self.got
        )
    }
}

impl std::error::Error for DimMismatch {}

/// L2-normalize `v` in place. Zero vectors are left untouched.
pub(crate) fn normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    if norm_sq > 0.0 {
        let inv = 1.0 / norm_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Dot product. Assumes both slices have equal length.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Builder for a flat (brute-force) k-NN index.
///
/// All vectors are L2-normalized at insert time so the distance
/// metric at query time is exact cosine similarity (`dot(q, v) =
/// cos(q, v)` for unit vectors). Pre-normalizing moves the
/// division into the build pass and keeps the query hot path a
/// single dot product per doc.
pub struct FlatBuilder {
    dim: usize,
    keys: Vec<RawValue>,
    handles: Vec<Value<Handle<Blake3, Embedding>>>,
}

impl FlatBuilder {
    /// Start a fresh builder. `dim` is the expected embedding
    /// length — stored in the index and checked against the
    /// query vector at query time.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "FlatBuilder: dim must be > 0");
        Self {
            dim,
            keys: Vec::new(),
            handles: Vec::new(),
        }
    }

    /// Insert a `(key, handle)` pair. `key` is any 32-byte
    /// triblespace value (GenId / ShortString / tag / composite
    /// — see [`insert_id`] / [`insert_value`] for typed
    /// wrappers); `handle` points at an [`Embedding`] blob in
    /// the pile's blob store. The builder stores neither the
    /// raw vector nor any copy of it — the pile owns the
    /// embedding and content-addresses it, so two indexes that
    /// embed the same entity share storage.
    ///
    /// Use [`crate::schemas::put_embedding`] to put + normalize
    /// + get a handle in one step.
    ///
    /// [`insert_id`]: Self::insert_id
    /// [`insert_value`]: Self::insert_value
    pub fn insert(&mut self, key: RawValue, handle: Value<Handle<Blake3, Embedding>>) {
        self.keys.push(key);
        self.handles.push(handle);
    }

    /// Convenience: insert keyed by a triblespace [`Id`].
    pub fn insert_id(&mut self, doc_id: Id, handle: Value<Handle<Blake3, Embedding>>) {
        let mut raw = [0u8; 32];
        let id_bytes: &RawId = doc_id.as_ref();
        raw[16..32].copy_from_slice(id_bytes);
        self.insert(raw, handle);
    }

    /// Convenience: insert keyed by a typed [`Value<S>`].
    pub fn insert_value<S: ValueSchema>(
        &mut self,
        key: Value<S>,
        handle: Value<Handle<Blake3, Embedding>>,
    ) {
        self.insert(key.raw, handle);
    }

    /// Consume the builder and produce a flat index.
    pub fn build(self) -> FlatIndex {
        FlatIndex {
            dim: self.dim,
            keys: self.keys,
            handles: self.handles,
        }
    }

    /// Number of vectors inserted so far.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// `true` if no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Configured embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Brute-force k-NN index.
///
/// Stores `(key, handle)` pairs — the embedding blobs live in
/// the pile's blob store, content-addressed. `similar()`
/// resolves handles through a caller-supplied
/// [`BlobStoreGet`][g] at query time, so two indexes that
/// embed the same entity share storage.
///
/// Scores are cosine similarity in `[-1, 1]` **iff** the
/// stored embeddings are L2-normalized (the convention — see
/// [`Embedding`]'s docs). `similar()` L2-normalizes the query
/// itself so the dot product reads back as cosine.
///
/// [g]: triblespace::core::repo::BlobStoreGet
#[derive(Debug, Clone)]
pub struct FlatIndex {
    dim: usize,
    keys: Vec<RawValue>,
    handles: Vec<Value<Handle<Blake3, Embedding>>>,
}

impl FlatIndex {
    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed documents.
    pub fn doc_count(&self) -> usize {
        self.keys.len()
    }

    /// The stored 32-byte keys table. `keys()[i]` is the
    /// [`RawValue`] for internal index `i`.
    pub fn keys(&self) -> &[RawValue] {
        &self.keys
    }

    /// The stored embedding-handle table. Paired index-wise
    /// with [`keys`].
    ///
    /// [`keys`]: Self::keys
    pub fn handles(&self) -> &[Value<Handle<Blake3, Embedding>>] {
        &self.handles
    }

    /// Attach a blob store to this index, returning a queryable
    /// view. This is the one-liner typical load path pairs with:
    ///
    /// ```ignore
    /// let idx: FlatIndex = reader.get::<_, FlatBlob>(handle)?;
    /// let view = idx.attach(&reader);
    /// view.similar(&query, k)?;
    /// ```
    ///
    /// The view wraps `store` in an internal
    /// [`BlobCache`][c] keyed on `Handle<Blake3, Embedding>`.
    /// Flat's brute-force scan visits each handle only once
    /// per query, so the cache's real payoff is across
    /// repeated queries against the same view — identical
    /// queries reuse the cached deserializations instead of
    /// re-fetching. `B: Clone` so the cache can own the store;
    /// typical readers are cheap-clone.
    ///
    /// [c]: triblespace::core::blob::BlobCache
    pub fn attach<'a, B>(&'a self, store: &B) -> AttachedFlatIndex<'a, B>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3> + Clone,
    {
        AttachedFlatIndex {
            index: self,
            cache: triblespace::core::blob::BlobCache::new(store.clone()),
        }
    }
}

/// A [`FlatIndex`] paired with the blob store its handles
/// resolve against — produced by [`FlatIndex::attach`].
///
/// Owns a [`BlobCache`][c] over the store, specialized to
/// `(Embedding, View<[f32]>)`. Dropping the view drops the
/// cache; the underlying store is unaffected.
///
/// [c]: triblespace::core::blob::BlobCache
pub struct AttachedFlatIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<Blake3>,
{
    index: &'a FlatIndex,
    cache: triblespace::core::blob::BlobCache<B, Blake3, Embedding, anybytes::View<[f32]>>,
}

impl<'a, B> AttachedFlatIndex<'a, B>
where
    B: triblespace::core::repo::BlobStoreGet<Blake3>,
{
    /// The inner index (back-reference, in case the caller
    /// wants metadata methods like `doc_count` without going
    /// through the view).
    pub fn index(&self) -> &FlatIndex {
        self.index
    }

    /// Return the top `k` documents by cosine similarity to
    /// `query`. The query is L2-normalized before scoring;
    /// stored embeddings are expected to be L2-normalized
    /// already (see [`Embedding`]'s convention doc).
    ///
    /// Returns fewer than `k` results if the index has fewer
    /// docs, and an empty vec on dim mismatch. Handle-fetch
    /// failures propagate via `B::GetError`.
    pub fn similar(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(RawValue, f32)>, B::GetError<anybytes::view::ViewError>> {
        if query.len() != self.index.dim || k == 0 {
            return Ok(Vec::new());
        }
        let mut q = query.to_vec();
        crate::schemas::l2_normalize(&mut q);

        let mut heap: std::collections::BinaryHeap<MinScored> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for (i, key) in self.index.keys.iter().enumerate() {
            let handle = self.index.handles[i];
            let view = self.cache.get(handle)?;
            let score = dot(&q, view.as_ref().as_ref());
            heap.push(MinScored { key: *key, score });
            if heap.len() > k {
                heap.pop();
            }
        }
        let mut out: Vec<(RawValue, f32)> = heap.into_iter().map(|m| (m.key, m.score)).collect();
        out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Convenience: [`similar`] with GenId-typed keys decoded
    /// back to [`Id`]; other schemas are dropped.
    ///
    /// [`similar`]: Self::similar
    pub fn similar_ids(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(Id, f32)>, B::GetError<anybytes::view::ViewError>> {
        Ok(self
            .similar(query, k)?
            .into_iter()
            .filter_map(|(raw, s)| {
                if raw[0..16] != [0u8; 16] {
                    return None;
                }
                let id_bytes: RawId = raw[16..32].try_into().ok()?;
                Id::new(id_bytes).map(|id| (id, s))
            })
            .collect())
    }
}

impl FlatIndex {
    /// Theoretical size of the naive flat-array serialization in
    /// bytes — baseline for comparing against more compressed
    /// forms. `24` B header + 64 B per doc (32 B key + 32 B
    /// handle); embeddings live in the pile's blob store and
    /// aren't counted here.
    pub fn byte_size(&self) -> usize {
        24 + self.keys.len() * 64
    }
}

/// A `(score, key)` wrapper whose `Ord` impl inverts score so
/// pushing into a max-heap of capacity `k` yields a min-heap
/// over scores — the top-k retention trick.
#[derive(Clone, Copy)]
struct MinScored {
    key: RawValue,
    score: f32,
}

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for MinScored {}

impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Invert so `BinaryHeap` behaves as a min-heap over score.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::repo::BlobStore;
    use triblespace::core::value::schemas::hash::Blake3;

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    /// Test helper: `Value<GenId>` form of `id(byte)` — the
    /// 32-byte representation that HNSWBuilder::insert_id
    /// stores and FlatIndex::similar returns.
    fn id_key(byte: u8) -> RawValue {
        let mut raw = [0u8; 32];
        let id = id(byte);
        let id_bytes: &RawId = id.as_ref();
        raw[16..32].copy_from_slice(id_bytes);
        raw
    }

    /// Put `vec` into `store` as a normalized [`Embedding`] blob
    /// and return the handle.
    fn put_emb(
        store: &mut MemoryBlobStore<Blake3>,
        vec: Vec<f32>,
    ) -> Value<Handle<Blake3, Embedding>> {
        crate::schemas::put_embedding::<_, Blake3>(store, vec).unwrap()
    }

    /// Test sugar: put `vec` into `store` and insert the (id,
    /// handle, vec) triple into `builder`. Returns the handle
    /// for anyone who needs it.
    fn hnsw_insert(
        builder: &mut HNSWBuilder,
        store: &mut MemoryBlobStore<Blake3>,
        doc_id: Id,
        vec: Vec<f32>,
    ) -> Result<Value<Handle<Blake3, Embedding>>, DimMismatch> {
        let h = put_emb(store, vec.clone());
        builder.insert_id(doc_id, h, vec)?;
        Ok(h)
    }

    /// Build a [`FlatIndex`] from `(id, vec)` pairs. Returns
    /// the index AND the store — the writer must live for the
    /// reader to remain valid (reft_light ReadHandles are
    /// backed by the writer's allocation).
    fn build_flat(
        dim: usize,
        entries: &[(Id, Vec<f32>)],
    ) -> (FlatIndex, MemoryBlobStore<Blake3>) {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = FlatBuilder::new(dim);
        for (doc, vec) in entries {
            let h = put_emb(&mut store, vec.clone());
            b.insert_id(*doc, h);
        }
        (b.build(), store)
    }

    /// Take a stable reader from an existing store. Test
    /// sugar — `store.reader().unwrap()` unwrapped for brevity.
    fn reader_of(store: &mut MemoryBlobStore<Blake3>) -> <MemoryBlobStore<Blake3> as BlobStore<Blake3>>::Reader {
        store.reader().unwrap()
    }

    #[test]
    fn exact_match_is_top() {
        let (idx, mut store) = build_flat(
            3,
            &[
                (id(1), vec![1.0, 0.0, 0.0]),
                (id(2), vec![0.0, 1.0, 0.0]),
                (id(3), vec![0.0, 0.0, 1.0]),
            ],
        );
        let hits = idx.attach(&reader_of(&mut store)).similar(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id_key(1));
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ranked_by_similarity() {
        let (idx, mut store) = build_flat(
            2,
            &[
                (id(1), vec![1.0, 0.0]),
                (id(2), vec![0.9, 0.1]),
                (id(3), vec![0.0, 1.0]),
            ],
        );
        let hits = idx.attach(&reader_of(&mut store)).similar(&[1.0, 0.0], 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, id_key(1));
        assert_eq!(hits[1].0, id_key(2));
        assert_eq!(hits[2].0, id_key(3));
        // Scores are monotonically non-increasing.
        assert!(hits[0].1 >= hits[1].1 && hits[1].1 >= hits[2].1);
    }

    #[test]
    fn normalizes_input_vectors() {
        // Two inputs that are parallel but scaled differently
        // must yield identical scores against any query — the
        // `put_embedding` helper normalizes before putting, so
        // they produce the *same* handle (dedup).
        let (idx, mut store) = build_flat(
            2,
            &[(id(1), vec![3.0, 0.0]), (id(2), vec![100.0, 0.0])],
        );
        let hits = idx.attach(&reader_of(&mut store)).similar(&[1.0, 0.0], 2).unwrap();
        assert!((hits[0].1 - hits[1].1).abs() < 1e-5);
    }

    #[test]
    fn empty_index_is_queryable() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let idx = FlatBuilder::new(4).build();
        let reader = store.reader().unwrap();
        assert_eq!(idx.attach(&reader).similar(&[0.0; 4], 3).unwrap(), vec![]);
    }

    #[test]
    fn k_zero_returns_empty() {
        let (idx, mut store) = build_flat(2, &[(id(1), vec![1.0, 0.0])]);
        assert!(idx.attach(&reader_of(&mut store)).similar(&[1.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn wrong_dim_query_returns_empty() {
        let (idx, mut store) = build_flat(3, &[(id(1), vec![1.0, 0.0, 0.0])]);
        assert!(idx.attach(&reader_of(&mut store)).similar(&[1.0, 0.0], 1).unwrap().is_empty()); // dim 2 vs 3
    }

    #[test]
    fn k_larger_than_corpus_truncates() {
        let (idx, mut store) = build_flat(
            2,
            &[(id(1), vec![1.0, 0.0]), (id(2), vec![0.0, 1.0])],
        );
        let hits = idx.attach(&reader_of(&mut store)).similar(&[1.0, 0.0], 10).unwrap();
        assert_eq!(hits.len(), 2);
    }

    fn sample_flat() -> (FlatIndex, MemoryBlobStore<Blake3>) {
        build_flat(
            3,
            &[
                (id(1), vec![1.0, 0.0, 0.0]),
                (id(2), vec![0.0, 1.0, 0.0]),
                (id(3), vec![0.5, 0.5, 0.0]),
            ],
        )
    }

    #[test]
    fn flat_byte_size_matches_formula() {
        let (idx, _) = sample_flat();
        assert_eq!(idx.byte_size(), 24 + idx.doc_count() * 64);
    }

    // ── HNSW tests ────────────────────────────────────────────────

    #[test]
    fn hnsw_empty_index_is_queryable() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let idx = HNSWBuilder::new(4).build();
        assert_eq!(idx.doc_count(), 0);
        assert!(idx.attach(&store.reader().unwrap()).similar(&[0.0; 4], 3, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn hnsw_single_doc_returns_itself() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3);
        hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0, 0.0]).unwrap();
        let idx = b.build();
        let hits = idx.attach(&store.reader().unwrap()).similar(&[1.0, 0.0, 0.0], 1, None)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id_key(1));
        assert!((hits[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hnsw_ranks_similar_vectors_highest() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(2);
        hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(2), vec![0.9, 0.1]).unwrap();
        hnsw_insert(&mut b, &mut store, id(3), vec![0.0, 1.0]).unwrap();
        let idx = b.build();
        let hits = idx.attach(&store.reader().unwrap()).similar(&[1.0, 0.0], 3, None)
            .unwrap();
        assert_eq!(hits[0].0, id_key(1));
        assert_eq!(hits[1].0, id_key(2));
        assert_eq!(hits[2].0, id_key(3));
    }

    #[test]
    fn hnsw_recall_matches_flat_on_small_corpus() {
        // Build both indexes over the same vectors, run the same
        // queries, confirm HNSW's top-3 overlaps ≥ 2 with flat's
        // exact top-3. (With small corpora HNSW may lose one due
        // to the level-sampling quantum, but shouldn't be wildly
        // different.)
        use crate::hnsw::FlatBuilder;
        let mut rng = 0xBABE_u64;
        let next = |r: &mut u64| {
            *r = r.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = *r;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        let dim = 16;
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut flat_b = FlatBuilder::new(dim);
        let mut hnsw_b = HNSWBuilder::new(dim).with_seed(42);
        for i in 0..200 {
            let vec: Vec<f32> = (0..dim)
                .map(|_| (next(&mut rng) as i32 as f32) / (i32::MAX as f32))
                .collect();
            let dx = id_from_u64((i + 1) as u64);
            let h = put_emb(&mut store, vec.clone());
            flat_b.insert_id(dx, h);
            hnsw_b.insert_id(dx, h, vec).unwrap();
        }
        let flat = flat_b.build();
        let hnsw = hnsw_b.build();
        let reader = store.reader().unwrap();

        // Five random queries.
        let mut total_overlap = 0;
        let mut total_k = 0;
        for _ in 0..5 {
            let q: Vec<f32> = (0..dim)
                .map(|_| (next(&mut rng) as i32 as f32) / (i32::MAX as f32))
                .collect();
            let truth: std::collections::HashSet<RawValue> = flat
                .attach(&reader)
                .similar(&q, 10)
                .unwrap()
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            let got: std::collections::HashSet<RawValue> = hnsw
                .attach(&reader)
                .similar(&q, 10, Some(50))
                .unwrap()
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            total_overlap += truth.intersection(&got).count();
            total_k += 10;
        }
        // Expect recall ≥ 0.7 on this dataset with ef_search=50.
        // Looser bound catches blatantly broken HNSW impls while
        // tolerating the algorithm's inherent approximate-ness.
        let recall = total_overlap as f32 / total_k as f32;
        assert!(recall >= 0.7, "HNSW recall {recall:.2} below 0.7 threshold");
    }

    fn id_from_u64(n: u64) -> Id {
        let mut raw = [0u8; 16];
        raw[..8].copy_from_slice(&n.max(1).to_le_bytes());
        raw[8..].copy_from_slice(&n.max(1).wrapping_mul(0x9E3779B9).to_le_bytes());
        Id::new(raw).unwrap()
    }

    #[test]
    fn hnsw_deterministic_seed_reproduces_structure() {
        let build = |seed: u64| {
            let mut store = MemoryBlobStore::<Blake3>::new();
            let mut b = HNSWBuilder::new(3).with_seed(seed);
            for i in 1..=20u8 {
                let v = vec![
                    (i as f32) / 20.0,
                    ((i as f32) * 2.0) % 1.0,
                    ((i as f32) * 3.0) % 1.0,
                ];
                let h = put_emb(&mut store, v.clone());
                b.insert_id(Id::new([i; 16]).unwrap(), h, v).unwrap();
            }
            (b.build(), store)
        };
        let (a, mut a_store) = build(123);
        let (b, mut b_store) = build(123);
        assert_eq!(a.doc_count(), b.doc_count());
        assert_eq!(a.max_level(), b.max_level());
        let q = vec![0.5, 0.3, 0.1];
        let ra = a.attach(&a_store.reader().unwrap()).similar(&q, 5, None).unwrap();
        let rb = b.attach(&b_store.reader().unwrap()).similar(&q, 5, None).unwrap();
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(rb.iter()) {
            assert_eq!(x.0, y.0);
        }
    }

    #[test]
    fn hnsw_dim_mismatch_rejected_at_insert() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3);
        // `hnsw_insert` normalizes+puts inside; the dim check
        // fires from `insert_id` because we pass a mismatched
        // `vec` to the builder.
        let err = hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0]).unwrap_err();
        assert_eq!(err.expected, 3);
        assert_eq!(err.got, 2);
    }

    #[test]
    fn hnsw_wrong_dim_query_returns_empty() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3);
        hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0, 0.0]).unwrap();
        let idx = b.build();
        assert!(idx.attach(&store.reader().unwrap()).similar(&[1.0, 0.0], 3, None)
            .unwrap()
            .is_empty());
    }

    fn sample_hnsw() -> (HNSWIndex, MemoryBlobStore<Blake3>) {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3).with_seed(42);
        hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(2), vec![0.9, 0.1, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(3), vec![0.0, 1.0, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(4), vec![0.0, 0.0, 1.0]).unwrap();
        (b.build_naive(), store)
    }

    #[test]
    fn hnsw_byte_size_positive_and_growing() {
        let (idx, _) = sample_hnsw();
        let small = idx.byte_size();
        assert!(small > 0);
        // Same corpus plus one doc must be strictly larger in the
        // naive layout — 32 B key + 32 B handle + 1 B level +
        // whatever neighbours land on the new node.
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3).with_seed(19);
        hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(2), vec![0.0, 1.0, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(3), vec![0.5, 0.5, 0.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(4), vec![0.0, 0.0, 1.0]).unwrap();
        hnsw_insert(&mut b, &mut store, id(5), vec![0.2, 0.3, 0.5]).unwrap();
        let larger = b.build_naive().byte_size();
        assert!(larger > small);
    }
}
