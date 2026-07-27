use std::cmp::Reverse;
use std::collections::BinaryHeap;

use triblespace_core::inline::RawInline;
use triblespace_core::trible::Trible;

use crate::{Automaton, GraphEdge, PathError, PathSummary};

#[derive(Clone, Debug)]
struct Csr {
    offsets: Vec<usize>,
    values: Vec<u32>,
}

impl Csr {
    fn row(&self, ordinal: usize) -> &[u32] {
        &self.values[self.offsets[ordinal]..self.offsets[ordinal + 1]]
    }

    fn transpose(&self, rows: usize, offset_count: usize) -> Self {
        let mut counts = vec![0usize; rows];
        for &target in &self.values {
            counts[target as usize] += 1;
        }
        let mut offsets = Vec::with_capacity(offset_count);
        offsets.push(0);
        for count in counts {
            offsets.push(offsets.last().copied().unwrap() + count);
        }
        let mut next = offsets[..rows].to_vec();
        let mut values = vec![0u32; self.values.len()];
        for source in 0..rows {
            for &target in self.row(source) {
                let slot = &mut next[target as usize];
                values[*slot] = source as u32;
                *slot += 1;
            }
        }
        Self { offsets, values }
    }
}

/// Exact accepted endpoint relation for one summary snapshot.
///
/// The forward CSR is the canonical denotation. Reverse fibers and the three
/// domain views are derived from it. The constructional [`PathSummary`] is
/// retained so independently built indexes can still be merged exactly.
#[derive(Clone, Debug)]
pub struct PathIndex {
    summary: PathSummary,
    forward: Csr,
    reverse: Csr,
    starts: Vec<u32>,
    ends: Vec<u32>,
    diagonal: Vec<u32>,
}

impl PathIndex {
    /// Builds an index directly from graph edges.
    pub fn from_edges(
        automaton: Automaton,
        edges: impl IntoIterator<Item = GraphEdge>,
    ) -> Result<Self, PathError> {
        Self::from_summary(PathSummary::from_edges(automaton, edges))
    }

    /// Builds an index directly from tribles.
    pub fn from_tribles<'a>(
        automaton: Automaton,
        tribles: impl IntoIterator<Item = &'a Trible>,
    ) -> Result<Self, PathError> {
        Self::from_summary(PathSummary::from_tribles(automaton, tribles))
    }

    /// Materializes one exact endpoint relation from a constructional summary.
    pub fn from_summary(summary: PathSummary) -> Result<Self, PathError> {
        let vertex_count = summary.vertices.len();
        if vertex_count > u32::MAX as usize {
            return Err(PathError::TooManyVertices {
                count: vertex_count,
            });
        }
        let offset_count = vertex_count
            .checked_add(1)
            .ok_or(PathError::CapacityOverflow)?;
        let nullable = summary.automaton.accepts_empty();
        let active_storage;
        let active_vertices = if nullable {
            active_storage = summary.active_vertices();
            active_storage.as_slice()
        } else {
            debug_assert!(summary.has_canonical_domain());
            summary.vertices.as_slice()
        };
        let active = materialize_active_relation(&summary, active_vertices)?;
        let forward = expand_relation(&summary.vertices, active_vertices, active, nullable)?;

        let starts = (0..vertex_count)
            .filter(|&source| !forward.row(source).is_empty())
            .map(|source| source as u32)
            .collect();
        let diagonal = (0..vertex_count)
            .filter(|&source| forward.row(source).binary_search(&(source as u32)).is_ok())
            .map(|source| source as u32)
            .collect();
        let reverse = forward.transpose(vertex_count, offset_count);
        let ends = (0..vertex_count)
            .filter(|&target| !reverse.row(target).is_empty())
            .map(|target| target as u32)
            .collect();

        Ok(Self {
            summary,
            forward,
            reverse,
            starts,
            ends,
            diagonal,
        })
    }

    /// Exact constructional summary retained by this index.
    pub fn summary(&self) -> &PathSummary {
        &self.summary
    }

    /// Fixed automaton defining this relation.
    pub fn automaton(&self) -> &Automaton {
        self.summary.automaton()
    }

    /// Number of graph terms in the canonical endpoint domain.
    pub fn vertex_count(&self) -> usize {
        self.summary.vertices.len()
    }

    /// Number of distinct accepted endpoint pairs.
    pub fn accepted_pair_count(&self) -> usize {
        self.forward.values.len()
    }

    /// Whether the automaton accepts a path from `source` to `target`.
    pub fn contains(&self, source: &RawInline, target: &RawInline) -> bool {
        let (Ok(source), Ok(target)) = (
            self.summary.vertices.binary_search(source),
            self.summary.vertices.binary_search(target),
        ) else {
            return false;
        };
        self.forward
            .row(source)
            .binary_search(&(target as u32))
            .is_ok()
    }

    /// Sorted, duplicate-free accepted endpoint pairs.
    pub fn accepted_pairs(&self) -> impl Iterator<Item = (RawInline, RawInline)> + '_ {
        (0..self.vertex_count()).flat_map(move |source| {
            let source_value = self.summary.vertices[source];
            self.forward
                .row(source)
                .iter()
                .map(move |&target| (source_value, self.summary.vertices[target as usize]))
        })
    }

    /// Sorted accepted targets for one source.
    pub fn reachable_from<'a>(
        &'a self,
        source: &RawInline,
    ) -> impl Iterator<Item = RawInline> + 'a {
        self.values(self.forward_ordinals(source))
    }

    /// Sorted accepted sources for one target.
    pub fn reaching<'a>(&'a self, target: &RawInline) -> impl Iterator<Item = RawInline> + 'a {
        self.values(self.reverse_ordinals(target))
    }

    /// Sorted sources having at least one accepted target.
    pub fn starts(&self) -> impl Iterator<Item = RawInline> + '_ {
        self.values(&self.starts)
    }

    /// Sorted targets having at least one accepted source.
    pub fn ends(&self) -> impl Iterator<Item = RawInline> + '_ {
        self.values(&self.ends)
    }

    /// Sorted vertices accepted from themselves.
    pub fn diagonal(&self) -> impl Iterator<Item = RawInline> + '_ {
        self.values(&self.diagonal)
    }

    /// Closes the canonical union of two retained summaries.
    pub fn merge(&self, other: &Self) -> Result<Self, PathError> {
        Self::from_summary(self.summary.merge(&other.summary)?)
    }

    /// Closes the canonical union of any nonempty index collection.
    pub fn merge_all<'a>(indexes: impl IntoIterator<Item = &'a Self>) -> Result<Self, PathError> {
        let summaries = indexes
            .into_iter()
            .map(|index| &index.summary)
            .collect::<Vec<_>>();
        Self::from_summary(PathSummary::merge_all(summaries)?)
    }

    pub(crate) fn values<'a>(
        &'a self,
        ordinals: &'a [u32],
    ) -> impl Iterator<Item = RawInline> + 'a {
        ordinals
            .iter()
            .map(|&ordinal| self.summary.vertices[ordinal as usize])
    }

    pub(crate) fn value(&self, ordinal: u32) -> RawInline {
        self.summary.vertices[ordinal as usize]
    }

    pub(crate) fn forward_ordinals(&self, source: &RawInline) -> &[u32] {
        self.summary
            .vertices
            .binary_search(source)
            .ok()
            .map(|source| self.forward.row(source))
            .unwrap_or(&[])
    }

    pub(crate) fn reverse_ordinals(&self, target: &RawInline) -> &[u32] {
        self.summary
            .vertices
            .binary_search(target)
            .ok()
            .map(|target| self.reverse.row(target))
            .unwrap_or(&[])
    }

    pub(crate) fn starts_ordinals(&self) -> &[u32] {
        &self.starts
    }

    pub(crate) fn ends_ordinals(&self) -> &[u32] {
        &self.ends
    }

    pub(crate) fn diagonal_ordinals(&self) -> &[u32] {
        &self.diagonal
    }

    pub(crate) fn ordinal_in(&self, ordinals: &[u32], value: &RawInline) -> bool {
        self.summary
            .vertices
            .binary_search(value)
            .is_ok_and(|ordinal| ordinals.binary_search(&(ordinal as u32)).is_ok())
    }
}

/// Close only the endpoints that occur in a direct product arc. Every
/// positive-length accepted path lies in this carrier. A nullable automaton's
/// identity over the larger supplied domain is added by [`expand_relation`].
fn materialize_active_relation(
    summary: &PathSummary,
    vertices: &[RawInline],
) -> Result<Csr, PathError> {
    let vertex_count = vertices.len();
    let offset_count = vertex_count
        .checked_add(1)
        .ok_or(PathError::CapacityOverflow)?;
    let state_count = summary.automaton.state_count() as usize;
    let product_count = vertex_count
        .checked_mul(state_count)
        .ok_or(PathError::CapacityOverflow)?;
    if product_count > u32::MAX as usize {
        return Err(PathError::ProductCarrierTooLarge {
            vertices: vertex_count,
            states: summary.automaton.state_count(),
        });
    }

    let mut adjacency = vec![Vec::<u32>::new(); product_count];
    for (source, target) in summary.ordinal_arcs_in(vertices) {
        adjacency[source as usize].push(target);
    }
    for targets in &mut adjacency {
        targets.sort_unstable();
        targets.dedup();
    }

    let (component_of, component_count) = strongly_connected_components(&adjacency);
    let condensation = condensation(&adjacency, &component_of, component_count);
    let topological = topological_order(&condensation);

    let row_words = vertex_count.div_ceil(u64::BITS as usize);
    let reach_words = component_count
        .checked_mul(row_words)
        .ok_or(PathError::CapacityOverflow)?;
    let mut accepting_reach = vec![0u64; reach_words];
    for (product, &component) in component_of.iter().enumerate() {
        let state = (product % state_count) as u32;
        if summary.automaton.is_accepting(state) {
            let component = component as usize;
            set_bit(
                &mut accepting_reach[component * row_words..(component + 1) * row_words],
                product / state_count,
            );
        }
    }
    for &component in topological.iter().rev() {
        for &successor in &condensation[component as usize] {
            union_component_rows(
                &mut accepting_reach,
                row_words,
                component as usize,
                successor as usize,
            );
        }
    }

    let mut offsets = Vec::with_capacity(offset_count);
    let mut values = Vec::new();
    let mut row = vec![0u64; row_words];
    offsets.push(0);
    for source in 0..vertex_count {
        row.fill(0);
        for initial in summary.automaton.initial_states() {
            let product = source * state_count + initial as usize;
            let component = component_of[product] as usize;
            let component_row =
                &accepting_reach[component * row_words..(component + 1) * row_words];
            for (target, source) in row.iter_mut().zip(component_row) {
                *target |= source;
            }
        }
        values.extend(
            set_bits(&row)
                .take_while(|&target| target < vertex_count)
                .map(|target| {
                    u32::try_from(target).expect("active vertex count was checked against u32::MAX")
                }),
        );
        offsets.push(values.len());
    }
    Ok(Csr { offsets, values })
}

fn expand_relation(
    vertices: &[RawInline],
    active_vertices: &[RawInline],
    active: Csr,
    add_identity: bool,
) -> Result<Csr, PathError> {
    if vertices == active_vertices {
        return Ok(active);
    }

    let active_to_full = active_vertices
        .iter()
        .map(|vertex| {
            vertices
                .binary_search(vertex)
                .expect("active path vertices belong to the summary domain")
        })
        .collect::<Vec<_>>();
    let added_identity = if add_identity {
        vertices.len().saturating_sub(active_vertices.len())
    } else {
        0
    };
    let capacity = active
        .values
        .len()
        .checked_add(added_identity)
        .ok_or(PathError::CapacityOverflow)?;
    let mut offsets = Vec::with_capacity(
        vertices
            .len()
            .checked_add(1)
            .ok_or(PathError::CapacityOverflow)?,
    );
    let mut values = Vec::with_capacity(capacity);
    let mut active_source = 0usize;
    offsets.push(0);
    for source in 0..vertices.len() {
        if active_to_full.get(active_source).copied() == Some(source) {
            let mut identity_written = false;
            for &active_target in active.row(active_source) {
                let target = active_to_full[active_target as usize];
                if add_identity && !identity_written && source < target {
                    values.push(source as u32);
                    identity_written = true;
                }
                values.push(target as u32);
                identity_written |= source == target;
            }
            if add_identity && !identity_written {
                values.push(source as u32);
            }
            active_source += 1;
        } else if add_identity {
            values.push(source as u32);
        }
        offsets.push(values.len());
    }
    debug_assert_eq!(active_source, active_vertices.len());
    Ok(Csr { offsets, values })
}

fn strongly_connected_components(adjacency: &[Vec<u32>]) -> (Vec<u32>, usize) {
    let mut reverse = vec![Vec::new(); adjacency.len()];
    for (source, targets) in adjacency.iter().enumerate() {
        for &target in targets {
            reverse[target as usize].push(source as u32);
        }
    }

    let mut seen = vec![false; adjacency.len()];
    let mut postorder = Vec::with_capacity(adjacency.len());
    for root in 0..adjacency.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root as u32, 0usize)];
        while let Some((node, next_edge)) = stack.last_mut() {
            if *next_edge < adjacency[*node as usize].len() {
                let target = adjacency[*node as usize][*next_edge];
                *next_edge += 1;
                if !seen[target as usize] {
                    seen[target as usize] = true;
                    stack.push((target, 0));
                }
            } else {
                postorder.push(*node);
                stack.pop();
            }
        }
    }

    let mut component_of = vec![u32::MAX; adjacency.len()];
    let mut component_count = 0u32;
    for &root in postorder.iter().rev() {
        if component_of[root as usize] != u32::MAX {
            continue;
        }
        component_of[root as usize] = component_count;
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            for &predecessor in &reverse[node as usize] {
                if component_of[predecessor as usize] == u32::MAX {
                    component_of[predecessor as usize] = component_count;
                    stack.push(predecessor);
                }
            }
        }
        component_count += 1;
    }
    (component_of, component_count as usize)
}

fn condensation(adjacency: &[Vec<u32>], component_of: &[u32], count: usize) -> Vec<Vec<u32>> {
    let mut result = vec![Vec::new(); count];
    for (source, targets) in adjacency.iter().enumerate() {
        let source_component = component_of[source];
        for &target in targets {
            let target_component = component_of[target as usize];
            if source_component != target_component {
                result[source_component as usize].push(target_component);
            }
        }
    }
    for targets in &mut result {
        targets.sort_unstable();
        targets.dedup();
    }
    result
}

fn topological_order(adjacency: &[Vec<u32>]) -> Vec<u32> {
    let mut indegree = vec![0usize; adjacency.len()];
    for targets in adjacency {
        for &target in targets {
            indegree[target as usize] += 1;
        }
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(component, _)| Reverse(component as u32))
        .collect::<BinaryHeap<_>>();
    let mut result = Vec::with_capacity(adjacency.len());
    while let Some(Reverse(component)) = ready.pop() {
        result.push(component);
        for &target in &adjacency[component as usize] {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                ready.push(Reverse(target));
            }
        }
    }
    debug_assert_eq!(result.len(), adjacency.len());
    result
}

fn union_component_rows(rows: &mut [u64], width: usize, target: usize, source: usize) {
    for word in 0..width {
        let source_word = rows[source * width + word];
        rows[target * width + word] |= source_word;
    }
}

fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / u64::BITS as usize] |= 1u64 << (bit % u64::BITS as usize);
}

fn set_bits(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words
        .iter()
        .copied()
        .enumerate()
        .flat_map(|(word_index, mut word)| {
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                Some(word_index * u64::BITS as usize + bit)
            })
        })
}
