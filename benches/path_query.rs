use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use triblespace::prelude::*;

mod bench_social {
    use triblespace::prelude::*;
    attributes! {
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" as follows: valueschemas::GenId;
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" as likes: valueschemas::GenId;
    }
}

/// Build a linear chain: n0 → n1 → n2 → ... → n_{len-1}
fn build_chain(len: usize) -> (TribleSet, Vec<ExclusiveId>) {
    let nodes: Vec<ExclusiveId> = (0..len).map(|_| fucid()).collect();
    let mut set = TribleSet::new();
    for i in 0..len - 1 {
        set += entity! { &nodes[i] @ bench_social::follows: &nodes[i + 1] };
    }
    (set, nodes)
}

/// Build a binary tree of depth d: each node follows two children.
fn build_tree(depth: usize) -> (TribleSet, ExclusiveId) {
    let root = fucid();
    let mut set = TribleSet::new();
    let mut frontier = vec![root.id];
    for _ in 0..depth {
        let mut next = Vec::new();
        for parent in &frontier {
            let left = fucid();
            let right = fucid();
            set += entity! { ExclusiveId::force_ref(parent) @ bench_social::follows: &left };
            set += entity! { ExclusiveId::force_ref(parent) @ bench_social::follows: &right };
            next.push(left.id);
            next.push(right.id);
        }
        frontier = next;
    }
    (set, root)
}

/// Build a graph with many nodes but only a small connected component reachable
/// from the start. Tests whether binding-aware estimation avoids scanning all nodes.
fn build_sparse(total_nodes: usize, reachable_len: usize) -> (TribleSet, ExclusiveId) {
    // Reachable chain.
    let (mut set, chain) = build_chain(reachable_len);
    let start = ExclusiveId::force(chain[0].id);
    // Unreachable noise: disconnected nodes with random edges among themselves.
    let noise: Vec<ExclusiveId> = (0..total_nodes - reachable_len).map(|_| fucid()).collect();
    for i in 0..noise.len().saturating_sub(1) {
        set += entity! { &noise[i] @ bench_social::follows: &noise[i + 1] };
    }
    (set, start)
}

fn bench_path_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("path/chain/follows+");
    for len in [10, 50, 100, 500] {
        let (set, nodes) = build_chain(len);
        let start_val = nodes[0].id.to_value();
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                let results: Vec<_> = find!(
                    (s: Value<_>, e: Value<_>),
                    and!(s.is(start_val), path!(set.clone(), s bench_social::follows+ e))
                )
                .collect();
                black_box(results)
            })
        });
    }
    group.finish();
}

fn bench_path_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("path/tree/follows+");
    for depth in [3, 5, 7, 9] {
        let (set, root) = build_tree(depth);
        let start_val = root.to_value();
        let expected = (1 << (depth + 1)) - 2; // 2^(d+1) - 2 reachable nodes
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, _| {
            b.iter(|| {
                let results: Vec<_> = find!(
                    (s: Value<_>, e: Value<_>),
                    and!(s.is(start_val), path!(set.clone(), s bench_social::follows+ e))
                )
                .collect();
                assert_eq!(results.len(), expected);
                black_box(results)
            })
        });
    }
    group.finish();
}

fn bench_path_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("path/sparse/follows+");
    // Fixed reachable chain of 10, increasing noise.
    for total in [100, 1_000, 10_000] {
        let (set, start) = build_sparse(total, 10);
        let start_val = start.to_value();
        group.bench_with_input(BenchmarkId::from_parameter(total), &total, |b, _| {
            b.iter(|| {
                let results: Vec<_> = find!(
                    (s: Value<_>, e: Value<_>),
                    and!(s.is(start_val), path!(set.clone(), s bench_social::follows+ e))
                )
                .collect();
                assert_eq!(results.len(), 9); // 10-node chain, 9 reachable from start
                black_box(results)
            })
        });
    }
    group.finish();
}

fn bench_path_alternation(c: &mut Criterion) {
    let mut group = c.benchmark_group("path/chain/(follows|likes)+");
    for len in [10, 50, 100] {
        let nodes: Vec<ExclusiveId> = (0..len).map(|_| fucid()).collect();
        let mut set = TribleSet::new();
        for i in 0..len - 1 {
            if i % 2 == 0 {
                set += entity! { &nodes[i] @ bench_social::follows: &nodes[i + 1] };
            } else {
                set += entity! { &nodes[i] @ bench_social::likes: &nodes[i + 1] };
            }
        }
        let start_val = nodes[0].id.to_value();
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                let results: Vec<_> = find!(
                    (s: Value<_>, e: Value<_>),
                    and!(
                        s.is(start_val),
                        path!(set.clone(), s (bench_social::follows | bench_social::likes)+ e)
                    )
                )
                .collect();
                black_box(results)
            })
        });
    }
    group.finish();
}

fn bench_path_conjunctive(c: &mut Criterion) {
    let mut group = c.benchmark_group("path/conjunctive");
    // Path query combined with a triple pattern constraint —
    // tests whether the path constraint participates in variable ordering.
    for len in [10, 50, 100] {
        let nodes: Vec<ExclusiveId> = (0..len).map(|_| fucid()).collect();
        let mut set = TribleSet::new();
        for i in 0..len - 1 {
            set += entity! { &nodes[i] @ bench_social::follows: &nodes[i + 1] };
        }
        // Add a "likes" edge from the last node to a target.
        let target = fucid();
        set += entity! { &nodes[len - 1] @ bench_social::likes: &target };
        let start_val = nodes[0].id.to_value();
        let target_val = target.to_value();

        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                // "Find nodes reachable via follows+ from start that also like target"
                let results: Vec<_> = find!(
                    (s: Value<_>, mid: Value<_>, e: Value<_>),
                    and!(
                        s.is(start_val),
                        e.is(target_val),
                        path!(set.clone(), s bench_social::follows+ mid),
                        pattern!(&set, [{ ?mid @ bench_social::likes: ?e }])
                    )
                )
                .collect();
                assert_eq!(results.len(), 1);
                black_box(results)
            })
        });
    }
    group.finish();
}

criterion_group!(
    path_benches,
    bench_path_chain,
    bench_path_tree,
    bench_path_sparse,
    bench_path_alternation,
    bench_path_conjunctive,
);
criterion_main!(path_benches);
