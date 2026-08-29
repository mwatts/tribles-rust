//! End-to-end incremental-query maintenance over growing exact covers.
//!
//! The two arms maintain the same application result set over source-identical
//! stores. Both retain a `SuccinctArchiveView`; only their query strategy
//! differs:
//!
//! - `full` re-runs the complete query and replaces the result set.
//! - `incremental` obtains one exact payload-support delta, runs
//!   `pattern_changes!`, and extends the result set.
//!
//! Every timed observation includes exact-view admission and application-side
//! `BTreeSet` maintenance. Source publication, cover discovery, fixture
//! construction, and the seed view are outside timing. Each measured run uses
//! fresh independent stores and advances through every fixed-size commit;
//! geometric checkpoints only control which observations are reported.
//! Strong equality checks run outside each timer but between observations, so
//! results describe an invariant-heavy warm trace rather than isolated
//! cache-cold calls.
//!
//! The fixture deliberately projects every query variable and gives every book
//! a unique binding. This makes the two accumulated result sets comparable. It
//! does not assert that `pattern_changes!` generally computes a projected set
//! difference: hidden witnesses and repeated commit support can legitimately
//! re-emit projected tuples.
//!
//! Usage:
//!
//! ```text
//! cargo bench --bench incremental_collection_queries -- \
//!   [--commits 64] [--books-per-commit 256] [--warmup 1] [--iters 4]
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use triblespace::core::collection::simplearchive_union::SimpleArchiveUnion;
use triblespace::core::collection::succinctarchive_union::{
    SuccinctArchiveCollection, SuccinctArchiveView,
};
use triblespace::core::collection::{
    reach, simplearchive_union, CollectionStoreExt, FactCover, SimpleArchiveCollection,
};
use triblespace::core::examples::literature;
use triblespace::prelude::*;

type Entity = Inline<inlineencodings::GenId>;
type Title = Inline<inlineencodings::ShortString>;
type Row = (Entity, Entity, Title);

#[derive(Clone)]
struct Fixture {
    store: MemoryRepo,
    seed_cover: FactCover,
    covers: Vec<FactCover>,
    expected_batches: Vec<Vec<Row>>,
    simple: SimpleArchiveCollection,
    succinct: SuccinctArchiveCollection,
}

fn benchmark_name() -> &'static str {
    "incremental-query-benchmark"
}

fn benchmark_key() -> SigningKey {
    SigningKey::from_bytes(&[0x49; 32])
}

fn build_fixture(commits: usize, books_per_commit: usize) -> Fixture {
    let signing_key = benchmark_key();
    let authority = signing_key.verifying_key();
    let name = benchmark_name();
    let mut store = MemoryRepo::default();
    let collection = store
        .collection::<SimpleArchiveUnion>(simplearchive_union::descriptor(
            name,
            authority,
            reach::private(),
        ))
        .expect("register benchmark collection");

    let author = entity! {
        literature::firstname: "Frank",
        literature::lastname: "Herbert",
    };
    let author_id = author.root().expect("intrinsic author id");
    store
        .commit(collection, &signing_key, author)
        .expect("publish seed author");
    let seed_cover = store.cover(collection, &[]).expect("freeze seed cover");
    assert_eq!(seed_cover.len(), 1);

    let mut covers = Vec::with_capacity(commits);
    let mut expected_batches = Vec::with_capacity(commits);
    for commit in 0..commits {
        let mut fragment = Fragment::empty();
        let mut expected = Vec::with_capacity(books_per_commit);
        for book in 0..books_per_commit {
            let ordinal = commit
                .checked_mul(books_per_commit)
                .and_then(|base| base.checked_add(book))
                .expect("fixture ordinal fits usize");
            let title = format!("Book {ordinal:020}");
            let entity = entity! {
                literature::title: title.as_str(),
                literature::author: &author_id,
            };
            let book_id = entity.root().expect("intrinsic book id");
            expected.push((
                author_id.to_inline(),
                book_id.to_inline(),
                title.as_str().to_inline(),
            ));
            fragment += entity;
        }
        store
            .commit(collection, &signing_key, fragment)
            .expect("publish book commit");
        covers.push(store.cover(collection, &[]).expect("freeze exact cover"));
        expected_batches.push(expected);
    }

    let simple = SimpleArchiveCollection::new(name, authority, reach::private());
    let succinct = SuccinctArchiveCollection::new(
        name,
        authority,
        reach::private(),
        authority,
        reach::private(),
    );

    Fixture {
        store,
        seed_cover,
        covers,
        expected_batches,
        simple,
        succinct,
    }
}

struct FullState {
    store: MemoryRepo,
    view: SuccinctArchiveView,
    results: BTreeSet<Row>,
}

impl FullState {
    fn seeded(fixture: &Fixture) -> Self {
        let mut store = fixture.store.clone();
        let mut view = fixture.succinct.exact_view();
        let seed = view
            .ensure(&mut store, &fixture.seed_cover)
            .expect("admit seed view");
        assert_eq!(seed.iter().count(), 2, "seed contains the author facts");
        Self {
            store,
            view,
            results: BTreeSet::new(),
        }
    }

    fn observe(&mut self, cover: &FactCover) -> Step {
        let start = Instant::now();
        let full = self
            .view
            .ensure(&mut self.store, cover)
            .expect("advance full-query view");
        let mut raw_rows = 0usize;
        let mut next = BTreeSet::new();
        for row in find!(
            (author: Entity, book: Entity, title: Title),
            pattern!(&full, [
                { ?author @ literature::firstname: "Frank" },
                { ?book @ literature::author: ?author, literature::title: ?title }
            ])
        ) {
            raw_rows += 1;
            next.insert(row);
        }
        let distinct_rows = next.len();
        self.results = next;
        black_box(self.results.len());
        Step {
            elapsed: start.elapsed(),
            raw_rows,
            distinct_rows,
        }
    }
}

struct IncrementalState {
    store: MemoryRepo,
    view: SuccinctArchiveView,
    checkpoint: FactCover,
    results: BTreeSet<Row>,
}

impl IncrementalState {
    fn seeded(fixture: &Fixture) -> Self {
        let mut store = fixture.store.clone();
        let mut view = fixture.succinct.exact_view();
        let seed = view
            .ensure(&mut store, &fixture.seed_cover)
            .expect("admit seed view");
        assert_eq!(seed.iter().count(), 2, "seed contains the author facts");
        Self {
            store,
            view,
            checkpoint: fixture.seed_cover.clone(),
            results: BTreeSet::new(),
        }
    }

    fn observe(&mut self, fixture: &Fixture, cover: &FactCover) -> Step {
        let start = Instant::now();
        let added = cover
            .additions_since(&self.checkpoint)
            .expect("cover grows monotonically");
        assert_eq!(added.len(), 1, "one payload is observed per step");
        let full = self
            .view
            .ensure(&mut self.store, cover)
            .expect("advance incremental full view");
        let changed = fixture
            .simple
            .attach_exact(&mut self.store, &added)
            .expect("attach exact support delta");

        let mut raw_rows = 0usize;
        let mut batch = BTreeSet::new();
        for row in find!(
            (author: Entity, book: Entity, title: Title),
            pattern_changes!(&full, &changed, [
                { ?author @ literature::firstname: "Frank" },
                { ?book @ literature::author: ?author, literature::title: ?title }
            ])
        ) {
            raw_rows += 1;
            batch.insert(row);
        }
        let distinct_rows = batch.len();
        self.results.extend(batch);
        self.checkpoint = cover.clone();
        black_box(self.results.len());
        Step {
            elapsed: start.elapsed(),
            raw_rows,
            distinct_rows,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Arm {
    Full,
    Incremental,
}

#[derive(Clone, Copy, Debug)]
struct Step {
    elapsed: Duration,
    raw_rows: usize,
    distinct_rows: usize,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    arm: Arm,
    commits: usize,
    total_results: usize,
    step: Step,
}

struct Run {
    samples: Vec<Sample>,
    results: BTreeSet<Row>,
}

fn geometric_checkpoints(commits: usize) -> BTreeSet<usize> {
    let mut checkpoints = BTreeSet::new();
    let mut next = 1usize;
    while next < commits {
        checkpoints.insert(next);
        next = next.saturating_mul(2);
    }
    checkpoints.insert(commits);
    checkpoints
}

fn run_full(fixture: &Fixture, checkpoints: &BTreeSet<usize>) -> Run {
    let mut state = FullState::seeded(fixture);
    let mut expected = BTreeSet::new();
    let mut samples = Vec::with_capacity(checkpoints.len());
    for (index, (cover, batch)) in fixture
        .covers
        .iter()
        .zip(&fixture.expected_batches)
        .enumerate()
    {
        expected.extend(batch.iter().cloned());
        let step = state.observe(cover);
        assert_eq!(step.raw_rows, expected.len());
        assert_eq!(step.distinct_rows, expected.len());
        assert_eq!(state.results, expected);
        assert_eq!(state.view.cover(), Some(cover));
        let commits = index + 1;
        if checkpoints.contains(&commits) {
            samples.push(Sample {
                arm: Arm::Full,
                commits,
                total_results: expected.len(),
                step,
            });
        }
    }
    Run {
        samples,
        results: state.results,
    }
}

fn run_incremental(fixture: &Fixture, checkpoints: &BTreeSet<usize>) -> Run {
    let mut state = IncrementalState::seeded(fixture);
    let mut expected = BTreeSet::new();
    let mut samples = Vec::with_capacity(checkpoints.len());
    for (index, (cover, batch)) in fixture
        .covers
        .iter()
        .zip(&fixture.expected_batches)
        .enumerate()
    {
        expected.extend(batch.iter().cloned());
        let step = state.observe(fixture, cover);
        assert_eq!(step.raw_rows, batch.len());
        assert_eq!(step.distinct_rows, batch.len());
        assert_eq!(state.results, expected);
        assert_eq!(&state.checkpoint, cover);
        assert_eq!(state.view.cover(), Some(cover));
        let commits = index + 1;
        if checkpoints.contains(&commits) {
            samples.push(Sample {
                arm: Arm::Incremental,
                commits,
                total_results: expected.len(),
                step,
            });
        }
    }
    Run {
        samples,
        results: state.results,
    }
}

#[derive(Default)]
struct Aggregate {
    elapsed_ns: Vec<u128>,
    raw_rows: Option<usize>,
    distinct_rows: Option<usize>,
    total_results: Option<usize>,
}

impl Aggregate {
    fn push(&mut self, sample: Sample) {
        self.elapsed_ns.push(sample.step.elapsed.as_nanos());
        assert_eq_or_init(&mut self.raw_rows, sample.step.raw_rows);
        assert_eq_or_init(&mut self.distinct_rows, sample.step.distinct_rows);
        assert_eq_or_init(&mut self.total_results, sample.total_results);
    }
}

fn assert_eq_or_init(slot: &mut Option<usize>, value: usize) {
    match slot {
        Some(expected) => assert_eq!(*expected, value),
        None => *slot = Some(value),
    }
}

fn median(values: &[u128]) -> u128 {
    let mut values = values.to_vec();
    values.sort_unstable();
    let upper = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[upper - 1] + (values[upper] - values[upper - 1]) / 2
    } else {
        values[upper]
    }
}

fn parse_usize(args: &[String], index: &mut usize, option: &str) -> usize {
    *index += 1;
    args.get(*index)
        .unwrap_or_else(|| panic!("{option} needs an integer"))
        .parse()
        .unwrap_or_else(|_| panic!("{option} needs an integer"))
}

fn main() {
    let mut commits = 64usize;
    let mut books_per_commit = 256usize;
    let mut warmup = 1usize;
    let mut iterations = 4usize;
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--commits" => commits = parse_usize(&args, &mut index, "--commits"),
            "--books-per-commit" => {
                books_per_commit = parse_usize(&args, &mut index, "--books-per-commit")
            }
            "--warmup" => warmup = parse_usize(&args, &mut index, "--warmup"),
            "--iters" => iterations = parse_usize(&args, &mut index, "--iters"),
            "--bench" => {}
            other => panic!("unknown option {other:?}"),
        }
        index += 1;
    }
    assert!(commits > 0, "--commits must be nonzero");
    assert!(books_per_commit > 0, "--books-per-commit must be nonzero");
    assert!(iterations > 0, "--iters must be nonzero");

    let fixture = build_fixture(commits, books_per_commit);
    let checkpoints = geometric_checkpoints(commits);
    let mut aggregates = BTreeMap::<(usize, Arm), Aggregate>::new();
    for iteration in 0..warmup + iterations {
        let (full, incremental) = if iteration.is_multiple_of(2) {
            (
                run_full(&fixture, &checkpoints),
                run_incremental(&fixture, &checkpoints),
            )
        } else {
            let incremental = run_incremental(&fixture, &checkpoints);
            let full = run_full(&fixture, &checkpoints);
            (full, incremental)
        };
        assert_eq!(full.results, incremental.results);
        if iteration >= warmup {
            for sample in full.samples.into_iter().chain(incremental.samples) {
                aggregates
                    .entry((sample.commits, sample.arm))
                    .or_default()
                    .push(sample);
            }
        }
    }

    println!("incremental collection-query maintenance");
    println!("  commits              : {commits}");
    println!("  books per commit     : {books_per_commit}");
    println!("  discarded warm runs : {warmup}");
    println!("  measured whole runs  : {iterations}");
    println!(
        "\n{:>7} {:>9} {:>12} {:>12} {:>9} {:>12} {:>12}",
        "commits", "results", "full-ms", "incr-ms", "speedup", "full-rows", "delta-rows"
    );
    for checkpoint in checkpoints {
        let full = &aggregates[&(checkpoint, Arm::Full)];
        let incremental = &aggregates[&(checkpoint, Arm::Incremental)];
        let full_ns = median(&full.elapsed_ns);
        let incremental_ns = median(&incremental.elapsed_ns);
        println!(
            "{:>7} {:>9} {:>12.3} {:>12.3} {:>8.2}x {:>12} {:>12}",
            checkpoint,
            full.total_results.expect("full result count"),
            full_ns as f64 / 1_000_000.0,
            incremental_ns as f64 / 1_000_000.0,
            full_ns as f64 / incremental_ns.max(1) as f64,
            full.raw_rows.expect("full raw rows"),
            incremental.raw_rows.expect("incremental raw rows"),
        );
    }
    println!(
        "\nEach row is median latency for one commit observation; both arms include exact-view admission and application BTreeSet maintenance."
    );
}
