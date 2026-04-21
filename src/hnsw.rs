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
//! let hits = idx.similar_ids(&query, 2, &reader).unwrap();
//! assert_eq!(hits.len(), 2);
//! // doc 1 is an exact match, doc 3 nearly so.
//! assert_eq!(hits[0].0, Id::new([1; 16]).unwrap());
//! assert_eq!(hits[1].0, Id::new([3; 16]).unwrap());
//! ```

use triblespace::core::id::{Id, RawId};
use triblespace::core::value::schemas::hash::{Blake3, Handle};
use triblespace::core::value::{RawValue, Value, ValueSchema};

use crate::schemas::Embedding;
use crate::FORMAT_VERSION;

// ── HNSW blob byte format ────────────────────────────────────────────

const HNSW_MAGIC: u32 = u32::from_le_bytes(*b"HNSW");
const HNSW_HEADER_LEN: usize = 32;

/// Errors produced by [`HNSWIndex::try_from_bytes`].
#[derive(Debug, Clone, PartialEq)]
pub enum HNSWLoadError {
    ShortHeader,
    BadMagic,
    VersionMismatch(u16),
    TruncatedSection(&'static str),
    /// A stored neighbour index is `>= n_nodes`.
    OutOfRangeNeighbour(u32),
    /// A per-node neighbour count was larger than the corpus.
    ImpossibleNeighbourCount(u32),
}

impl std::fmt::Display for HNSWLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortHeader => write!(f, "HNSW blob shorter than header"),
            Self::BadMagic => write!(f, "HNSW blob: magic mismatch"),
            Self::VersionMismatch(v) => {
                write!(f, "HNSW blob: version {v} (expected {})", FORMAT_VERSION)
            }
            Self::TruncatedSection(name) => {
                write!(f, "HNSW blob: truncated section `{name}`")
            }
            Self::OutOfRangeNeighbour(i) => {
                write!(f, "HNSW blob: neighbour index {i} ≥ n_nodes")
            }
            Self::ImpossibleNeighbourCount(n) => {
                write!(
                    f,
                    "HNSW blob: node has {n} neighbours but corpus is smaller"
                )
            }
        }
    }
}

impl std::error::Error for HNSWLoadError {}

// Byte format for the flat k-NN blob. A separate magic from the
// eventual proper-HNSW blob ("FLAT" vs "HNSW") — they're
// different on-disk shapes (flat = no layers, no graph).
const FLAT_MAGIC: u32 = u32::from_le_bytes(*b"FLAT");
const FLAT_HEADER_LEN: usize = 32;

/// Errors produced by [`FlatIndex::try_from_bytes`].
#[derive(Debug, Clone, PartialEq)]
pub enum FlatLoadError {
    ShortHeader,
    BadMagic,
    VersionMismatch(u16),
    TruncatedSection(&'static str),
}

impl std::fmt::Display for FlatLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortHeader => write!(f, "FLAT blob shorter than header"),
            Self::BadMagic => write!(f, "FLAT blob: magic mismatch"),
            Self::VersionMismatch(v) => {
                write!(f, "FLAT blob: version {v} (expected {})", FORMAT_VERSION)
            }
            Self::TruncatedSection(n) => write!(f, "FLAT blob: truncated section `{n}`"),
        }
    }
}

impl std::error::Error for FlatLoadError {}

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

    /// Consume the builder and produce an immutable [`HNSWIndex`].
    /// Strips the inline build-time vectors — only the handles
    /// survive. Embeddings are resolved at query time through
    /// the caller-supplied blob store.
    pub fn build(self) -> HNSWIndex {
        // Strip vectors; keep only neighbour lists and per-node
        // level metadata for succinct encoding downstream.
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

    /// Approximate top-k nearest neighbours to `query` under
    /// cosine similarity, resolving embedding handles through
    /// `store`. `ef` tunes the search width (larger = better
    /// recall at higher cost); pass `None` to default to `k`.
    ///
    /// Handle lookups happen on every distance evaluation along
    /// the HNSW walk. Wrap `store` in a
    /// [`BlobCache`][c] when the same index will be queried
    /// repeatedly — it amortizes the per-handle deserialize
    /// across queries.
    ///
    /// [c]: triblespace::core::blob::BlobCache
    pub fn similar<B>(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        store: &B,
    ) -> Result<Vec<(RawValue, f32)>, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        if query.len() != self.dim || k == 0 {
            return Ok(Vec::new());
        }
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        let mut q = query.to_vec();
        normalize(&mut q);
        let ef = ef.unwrap_or(k).max(k);

        // Greedy descent from max_level down to 1.
        let mut curr = entry;
        for lvl in (1..=self.max_level).rev() {
            curr = self.greedy_search_layer(&q, curr, lvl, store)?;
        }
        // ef-search on layer 0.
        let candidates = self.search_layer(&q, curr, ef, 0, store)?;
        let mut ranked: Vec<(RawValue, f32)> = candidates
            .into_iter()
            .map(|(i, dist)| {
                // Convert distance back to similarity (cos = 1 - dist).
                (self.keys[i as usize], 1.0 - dist)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(k);
        Ok(ranked)
    }

    /// Convenience: [`similar`] but decode each key as a
    /// triblespace [`Id`] (assuming the index was keyed via
    /// [`HNSWBuilder::insert_id`] / `Value<GenId>`). Returns
    /// `None` for stored keys that aren't valid GenIds
    /// (non-zero leading 16 bytes or a nil tail).
    ///
    /// [`similar`]: Self::similar
    pub fn similar_ids<B>(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        store: &B,
    ) -> Result<Vec<(Id, f32)>, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        Ok(self
            .similar(query, k, ef, store)?
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

    /// Serialize the index to a self-contained little-endian
    /// byte buffer. Layout:
    ///
    /// ```text
    /// [32 B header]
    ///   magic u32 = "HNSW"
    ///   version u16
    ///   reserved u16
    ///   n_nodes u32
    ///   dim u32
    ///   M u16, M0 u16
    ///   max_level u8, has_entry u8, reserved 2 B
    ///   entry_point u32
    ///   reserved 4 B
    /// [n_nodes × 32 B keys]
    /// [n_nodes × 32 B handles] — Value<Handle<Blake3, Embedding>>
    /// [n_nodes × 1 B node_level]
    /// [n_nodes × layer_count+1 × 4 B per-layer neighbour offsets
    ///   — cumulative, starts at 0]
    /// [total_neighbours × 4 B neighbour indices]
    /// ```
    ///
    /// Embeddings are NOT inline — the `handles` section
    /// references blobs in the pile's blob store. Query time
    /// resolves handles through a caller-supplied
    /// `BlobStoreGet` (see [`similar`]).
    ///
    /// [`similar`]: Self::similar
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_nodes = self.nodes.len() as u32;
        let has_entry: u8 = self.entry_point.is_some() as u8;
        let entry = self.entry_point.unwrap_or(0);
        let dim = self.dim as u32;

        let mut buf = Vec::new();
        // Header.
        buf.extend_from_slice(&HNSW_MAGIC.to_le_bytes()); // 4
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes()); // 2
        buf.extend_from_slice(&0u16.to_le_bytes()); // 2 reserved
        buf.extend_from_slice(&n_nodes.to_le_bytes()); // 4
        buf.extend_from_slice(&dim.to_le_bytes()); // 4
        buf.extend_from_slice(&self.m.to_le_bytes()); // 2
        buf.extend_from_slice(&self.m0.to_le_bytes()); // 2
        buf.push(self.max_level); // 1
        buf.push(has_entry); // 1
        buf.extend_from_slice(&[0u8; 2]); // 2 reserved
        buf.extend_from_slice(&entry.to_le_bytes()); // 4
        buf.extend_from_slice(&[0u8; 4]); // 4 reserved → 32
        debug_assert_eq!(buf.len(), HNSW_HEADER_LEN);

        // keys
        for key in &self.keys {
            buf.extend_from_slice(key);
        }
        // handles
        for handle in &self.handles {
            buf.extend_from_slice(&handle.raw);
        }
        // node levels
        for node in &self.nodes {
            buf.push(node.level);
        }
        // per-node neighbour offsets + flat neighbour array.
        // Per node we store `level + 2` offsets (one per layer
        // 0..=level, plus a tail offset); neighbour indices live
        // in a single trailing flat array.
        let max_layer_count = (self.max_level as usize) + 1;
        let total_neighbours: u64 = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|l| l.len() as u64).sum::<u64>())
            .sum();

        // Offsets: for each node, (level + 2) u32s. Layout puts
        // them all first so loading can stream them without
        // back-patching.
        let mut neighbour_bytes: Vec<u8> = Vec::with_capacity(total_neighbours as usize * 4);
        for node in &self.nodes {
            let mut running: u32 = 0;
            buf.extend_from_slice(&running.to_le_bytes());
            for layer in &node.neighbors {
                running += layer.len() as u32;
                buf.extend_from_slice(&running.to_le_bytes());
                for &n in layer {
                    neighbour_bytes.extend_from_slice(&n.to_le_bytes());
                }
            }
            // Pad the node's offset table to max_layer_count + 1
            // entries so each node has the same offset stride —
            // simplifies deserialization indexing.
            let actual_entries = node.neighbors.len() + 1; // level..=level inclusive + tail
            let padding = (max_layer_count + 1).saturating_sub(actual_entries);
            for _ in 0..padding {
                buf.extend_from_slice(&running.to_le_bytes());
            }
        }
        buf.extend(neighbour_bytes);
        buf
    }

    /// Reload an index previously produced by [`to_bytes`].
    ///
    /// [`to_bytes`]: Self::to_bytes
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, HNSWLoadError> {
        use HNSWLoadError as E;
        if bytes.len() < HNSW_HEADER_LEN {
            return Err(E::ShortHeader);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != HNSW_MAGIC {
            return Err(E::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(E::VersionMismatch(version));
        }
        let n_nodes = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let m = u16::from_le_bytes(bytes[16..18].try_into().unwrap());
        let m0 = u16::from_le_bytes(bytes[18..20].try_into().unwrap());
        let max_level = bytes[20];
        let has_entry = bytes[21] != 0;
        // bytes[22..24] reserved
        let entry_raw = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        // bytes[28..32] reserved

        let mut pos = HNSW_HEADER_LEN;
        // keys
        let keys_end = pos + n_nodes * 32;
        if bytes.len() < keys_end {
            return Err(E::TruncatedSection("keys"));
        }
        let mut keys: Vec<RawValue> = Vec::with_capacity(n_nodes);
        for chunk in bytes[pos..keys_end].chunks_exact(32) {
            let raw: RawValue = chunk.try_into().unwrap();
            keys.push(raw);
        }
        pos = keys_end;

        // handles
        let handles_end = pos + n_nodes * 32;
        if bytes.len() < handles_end {
            return Err(E::TruncatedSection("handles"));
        }
        let mut handles: Vec<Value<Handle<Blake3, Embedding>>> = Vec::with_capacity(n_nodes);
        for chunk in bytes[pos..handles_end].chunks_exact(32) {
            let raw: RawValue = chunk.try_into().unwrap();
            handles.push(Value::new(raw));
        }
        pos = handles_end;
        let _ = dim; // dim stays in the header as metadata; no inline vectors to size.

        // levels
        let levels_end = pos + n_nodes;
        if bytes.len() < levels_end {
            return Err(E::TruncatedSection("levels"));
        }
        let levels: Vec<u8> = bytes[pos..levels_end].to_vec();
        pos = levels_end;

        // per-node offset tables (size = max_layer_count + 1
        // u32 entries per node).
        let max_layer_count = (max_level as usize) + 1;
        let entries_per_node = max_layer_count + 1;
        let offsets_end = pos + n_nodes * entries_per_node * 4;
        if bytes.len() < offsets_end {
            return Err(E::TruncatedSection("neighbour_offsets"));
        }
        let all_offsets: Vec<u32> = bytes[pos..offsets_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        pos = offsets_end;

        // flat neighbour indices
        let remaining = &bytes[pos..];
        let all_neighbours: Vec<u32> = remaining
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // Rebuild per-node HNSWIndexNode structs.
        let mut nodes: Vec<HNSWIndexNode> = Vec::with_capacity(n_nodes);
        let mut neighbour_cursor: u32 = 0;
        for i in 0..n_nodes {
            let level = levels[i];
            let n_layers = level as usize + 1;
            let mut layer_lists: Vec<Vec<u32>> = Vec::with_capacity(n_layers);
            let offset_block = &all_offsets[i * entries_per_node..(i + 1) * entries_per_node];
            for layer_idx in 0..n_layers {
                let start = offset_block[layer_idx];
                let end = offset_block[layer_idx + 1];
                if start > end {
                    return Err(E::TruncatedSection("neighbour_offsets"));
                }
                let mut list: Vec<u32> = Vec::with_capacity((end - start) as usize);
                for offset in start..end {
                    let neighbour_pos = neighbour_cursor + (offset - start);
                    let Some(&n) = all_neighbours.get(neighbour_pos as usize) else {
                        return Err(E::TruncatedSection("neighbours"));
                    };
                    if n as usize >= n_nodes {
                        return Err(E::OutOfRangeNeighbour(n));
                    }
                    list.push(n);
                }
                layer_lists.push(list);
            }
            let last = offset_block[n_layers];
            let first = offset_block[0];
            let span = last - first;
            if span as usize > n_nodes * n_layers {
                return Err(E::ImpossibleNeighbourCount(span));
            }
            neighbour_cursor += span;
            nodes.push(HNSWIndexNode {
                level,
                neighbors: layer_lists,
            });
        }

        Ok(Self {
            dim,
            m,
            m0,
            nodes,
            keys,
            handles,
            entry_point: if has_entry { Some(entry_raw) } else { None },
            max_level,
        })
    }

    /// Fetch node `i`'s embedding from `store` and dot-product
    /// it with the normalized query. Every distance evaluation
    /// in the HNSW walk routes through this helper.
    fn dist_to<B>(
        &self,
        q: &[f32],
        i: u32,
        store: &B,
    ) -> Result<f32, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        let handle = self.handles[i as usize];
        let view: anybytes::View<[f32]> =
            store.get::<anybytes::View<[f32]>, Embedding>(handle)?;
        Ok(cosine_dist(q, view.as_ref()))
    }

    fn greedy_search_layer<B>(
        &self,
        q: &[f32],
        entry: u32,
        layer: u8,
        store: &B,
    ) -> Result<u32, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        let mut curr = entry;
        let mut curr_dist = self.dist_to(q, curr, store)?;
        loop {
            let mut changed = false;
            // Defensive bounds: a later insert can leave a stub
            // entry_point that never received layer-L neighbour
            // lists yet. Bail out cleanly instead of panicking.
            let node = &self.nodes[curr as usize];
            let Some(neigh) = node.neighbors.get(layer as usize) else {
                return Ok(curr);
            };
            let neigh = neigh.clone();
            for n in neigh {
                let d = self.dist_to(q, n, store)?;
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

    fn search_layer<B>(
        &self,
        q: &[f32],
        entry: u32,
        ef: usize,
        layer: u8,
        store: &B,
    ) -> Result<Vec<(u32, f32)>, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        use std::collections::BinaryHeap;
        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        visited.insert(entry);
        let d0 = self.dist_to(q, entry, store)?;
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
                let node = &self.nodes[c.idx as usize];
                let Some(neigh) = node.neighbors.get(layer as usize) else {
                    continue;
                };
                neigh.clone()
            };
            for n in neigh {
                if !visited.insert(n) {
                    continue;
                }
                let d = self.dist_to(q, n, store)?;
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

    /// Return the top `k` documents by cosine similarity to
    /// `query`, resolving embedding handles through `store`.
    ///
    /// The query is L2-normalized before scoring; stored
    /// embeddings are expected to be L2-normalized already
    /// (see [`Embedding`]'s convention doc).
    ///
    /// Returns fewer than `k` results if the index has fewer
    /// docs, and an empty vec on dim mismatch. Handle-fetch
    /// failures propagate via `B::GetError`.
    pub fn similar<B>(
        &self,
        query: &[f32],
        k: usize,
        store: &B,
    ) -> Result<Vec<(RawValue, f32)>, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        if query.len() != self.dim || k == 0 {
            return Ok(Vec::new());
        }
        let mut q = query.to_vec();
        crate::schemas::l2_normalize(&mut q);

        let mut heap: std::collections::BinaryHeap<MinScored> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for (i, key) in self.keys.iter().enumerate() {
            let handle = self.handles[i];
            eprintln!("similar: try handle {:02x?}", handle.raw);
            let view: anybytes::View<[f32]> = store.get::<anybytes::View<[f32]>, Embedding>(handle)?;
            let score = dot(&q, view.as_ref());
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
    pub fn similar_ids<B>(
        &self,
        query: &[f32],
        k: usize,
        store: &B,
    ) -> Result<Vec<(Id, f32)>, B::GetError<anybytes::view::ViewError>>
    where
        B: triblespace::core::repo::BlobStoreGet<Blake3>,
    {
        Ok(self
            .similar(query, k, store)?
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
    /// Serialize to a self-contained little-endian byte buffer.
    ///
    /// Layout:
    /// ```text
    /// [32 B header]
    ///   magic u32 = "FLAT"
    ///   version u16
    ///   reserved u16
    ///   n_docs u32
    ///   dim u32
    ///   reserved 12 B
    /// [n_docs × 32 B keys]
    /// [n_docs × 32 B handles] — Value<Handle<Blake3, Embedding>>
    /// ```
    ///
    /// Embeddings are NOT stored in this blob — they live in
    /// the pile's blob store, referenced by handle. The blob
    /// here is tiny: header + 64 bytes per doc.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n_docs = self.keys.len() as u32;
        let dim = self.dim as u32;

        let mut buf = Vec::with_capacity(FLAT_HEADER_LEN + self.keys.len() * 64);

        buf.extend_from_slice(&FLAT_MAGIC.to_le_bytes()); // 4
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes()); // 2
        buf.extend_from_slice(&0u16.to_le_bytes()); // 2 reserved
        buf.extend_from_slice(&n_docs.to_le_bytes()); // 4
        buf.extend_from_slice(&dim.to_le_bytes()); // 4
        buf.extend_from_slice(&[0u8; 16]); // 16 reserved → total 32
        debug_assert_eq!(buf.len(), FLAT_HEADER_LEN);

        for key in &self.keys {
            buf.extend_from_slice(key);
        }
        for handle in &self.handles {
            buf.extend_from_slice(&handle.raw);
        }
        buf
    }

    /// Reload an index previously produced by [`to_bytes`].
    ///
    /// [`to_bytes`]: Self::to_bytes
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, FlatLoadError> {
        use FlatLoadError as E;

        if bytes.len() < FLAT_HEADER_LEN {
            return Err(E::ShortHeader);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != FLAT_MAGIC {
            return Err(E::BadMagic);
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(E::VersionMismatch(version));
        }
        let n_docs = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

        let mut pos = FLAT_HEADER_LEN;
        let keys_end = pos + n_docs * 32;
        if bytes.len() < keys_end {
            return Err(E::TruncatedSection("keys"));
        }
        let mut keys: Vec<RawValue> = Vec::with_capacity(n_docs);
        for chunk in bytes[pos..keys_end].chunks_exact(32) {
            let raw: RawValue = chunk.try_into().unwrap();
            keys.push(raw);
        }
        pos = keys_end;

        let handles_end = pos + n_docs * 32;
        if bytes.len() < handles_end {
            return Err(E::TruncatedSection("handles"));
        }
        let mut handles: Vec<Value<Handle<Blake3, Embedding>>> = Vec::with_capacity(n_docs);
        for chunk in bytes[pos..handles_end].chunks_exact(32) {
            let raw: RawValue = chunk.try_into().unwrap();
            handles.push(Value::new(raw));
        }

        Ok(Self { dim, keys, handles })
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
        let hits = idx.similar(&[1.0, 0.0, 0.0], 1, &reader_of(&mut store)).unwrap();
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
        let hits = idx.similar(&[1.0, 0.0], 3, &reader_of(&mut store)).unwrap();
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
        let hits = idx.similar(&[1.0, 0.0], 2, &reader_of(&mut store)).unwrap();
        assert!((hits[0].1 - hits[1].1).abs() < 1e-5);
    }

    #[test]
    fn empty_index_is_queryable() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let idx = FlatBuilder::new(4).build();
        let reader = store.reader().unwrap();
        assert_eq!(idx.similar(&[0.0; 4], 3, &reader).unwrap(), vec![]);
    }

    #[test]
    fn k_zero_returns_empty() {
        let (idx, mut store) = build_flat(2, &[(id(1), vec![1.0, 0.0])]);
        assert!(idx.similar(&[1.0, 0.0], 0, &reader_of(&mut store)).unwrap().is_empty());
    }

    #[test]
    fn wrong_dim_query_returns_empty() {
        let (idx, mut store) = build_flat(3, &[(id(1), vec![1.0, 0.0, 0.0])]);
        assert!(idx.similar(&[1.0, 0.0], 1, &reader_of(&mut store)).unwrap().is_empty()); // dim 2 vs 3
    }

    #[test]
    fn k_larger_than_corpus_truncates() {
        let (idx, mut store) = build_flat(
            2,
            &[(id(1), vec![1.0, 0.0]), (id(2), vec![0.0, 1.0])],
        );
        let hits = idx.similar(&[1.0, 0.0], 10, &reader_of(&mut store)).unwrap();
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
    fn flat_bytes_round_trip() {
        let (original, mut store) = sample_flat();
        let reloaded = FlatIndex::try_from_bytes(&original.to_bytes()).expect("valid blob");
        assert_eq!(reloaded.dim(), original.dim());
        assert_eq!(reloaded.doc_count(), original.doc_count());

        // Query results must match exactly — handles round-trip
        // unchanged, and the same store resolves them both.
        let q = vec![1.0, 0.0, 0.0];
        let a = original.similar(&q, 3, &reader_of(&mut store)).unwrap();
        let b = reloaded.similar(&q, 3, &reader_of(&mut store)).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.0, y.0);
            assert!((x.1 - y.1).abs() < 1e-6);
        }
    }

    #[test]
    fn flat_blob_is_deterministic() {
        let (a, _) = sample_flat();
        let (b, _) = sample_flat();
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn flat_short_header_rejected() {
        let err = FlatIndex::try_from_bytes(&[0; 8]).unwrap_err();
        assert_eq!(err, FlatLoadError::ShortHeader);
    }

    #[test]
    fn flat_bad_magic_rejected() {
        let (idx, _store) = sample_flat();
        let mut bytes = idx.to_bytes();
        bytes[0] = b'X';
        let err = FlatIndex::try_from_bytes(&bytes).unwrap_err();
        assert_eq!(err, FlatLoadError::BadMagic);
    }

    #[test]
    fn flat_truncation_rejected() {
        let (idx, _store) = sample_flat();
        let bytes = idx.to_bytes();
        let err = FlatIndex::try_from_bytes(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(matches!(err, FlatLoadError::TruncatedSection(_)));
    }

    // ── HNSW tests ────────────────────────────────────────────────

    #[test]
    fn hnsw_empty_index_is_queryable() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let idx = HNSWBuilder::new(4).build();
        assert_eq!(idx.doc_count(), 0);
        assert!(idx
            .similar(&[0.0; 4], 3, None, &store.reader().unwrap())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn hnsw_single_doc_returns_itself() {
        let mut store = MemoryBlobStore::<Blake3>::new();
        let mut b = HNSWBuilder::new(3);
        hnsw_insert(&mut b, &mut store, id(1), vec![1.0, 0.0, 0.0]).unwrap();
        let idx = b.build();
        let hits = idx
            .similar(&[1.0, 0.0, 0.0], 1, None, &store.reader().unwrap())
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
        let hits = idx
            .similar(&[1.0, 0.0], 3, None, &store.reader().unwrap())
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
                .similar(&q, 10, &reader)
                .unwrap()
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            let got: std::collections::HashSet<RawValue> = hnsw
                .similar(&q, 10, Some(50), &reader)
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
        let ra = a.similar(&q, 5, None, &a_store.reader().unwrap()).unwrap();
        let rb = b.similar(&q, 5, None, &b_store.reader().unwrap()).unwrap();
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
        assert!(idx
            .similar(&[1.0, 0.0], 3, None, &store.reader().unwrap())
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
        (b.build(), store)
    }

    #[test]
    fn hnsw_empty_bytes_round_trip() {
        let idx = HNSWBuilder::new(3).build();
        let bytes = idx.to_bytes();
        let reloaded = HNSWIndex::try_from_bytes(&bytes).expect("valid blob");
        assert_eq!(reloaded.doc_count(), 0);
        assert_eq!(reloaded.dim(), 3);
    }

    #[test]
    fn hnsw_bytes_round_trip_preserves_queries() {
        let (idx, mut store) = sample_hnsw();
        let bytes = idx.to_bytes();
        let reloaded = HNSWIndex::try_from_bytes(&bytes).expect("valid blob");
        assert_eq!(reloaded.doc_count(), idx.doc_count());
        assert_eq!(reloaded.dim(), idx.dim());
        assert_eq!(reloaded.m(), idx.m());
        assert_eq!(reloaded.m0(), idx.m0());
        assert_eq!(reloaded.max_level(), idx.max_level());

        let q = vec![1.0, 0.0, 0.0];
        let reader = store.reader().unwrap();
        let a = idx.similar(&q, 4, Some(10), &reader).unwrap();
        let b = reloaded.similar(&q, 4, Some(10), &reader).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.0, y.0);
            assert!((x.1 - y.1).abs() < 1e-5);
        }
    }

    #[test]
    fn hnsw_bytes_are_deterministic() {
        let (a, _) = sample_hnsw();
        let (b, _) = sample_hnsw();
        assert_eq!(a.to_bytes(), b.to_bytes());
    }

    #[test]
    fn hnsw_short_header_rejected() {
        let err = HNSWIndex::try_from_bytes(&[0; 10]).unwrap_err();
        assert_eq!(err, HNSWLoadError::ShortHeader);
    }

    #[test]
    fn hnsw_bad_magic_rejected() {
        let (idx, _) = sample_hnsw();
        let mut bytes = idx.to_bytes();
        bytes[0] = b'X';
        let err = HNSWIndex::try_from_bytes(&bytes).unwrap_err();
        assert_eq!(err, HNSWLoadError::BadMagic);
    }
}
