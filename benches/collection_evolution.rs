//! Evolving-ticket maintenance benchmark for canonical Succinct collections.
//!
//! This benchmark compares the two real public maintenance operations on
//! geometrically growing exact tickets:
//!
//! - `ensure_exact`: canonical raw Succinct derivation plus Rank9 fibers.
//! - `compact_exact`: the same exact completion plus deterministic dyadic
//!   raw-target compaction and Rank9 fibers for the selected cover.
//!
//! Each operation gets an independent warm store, a source-identical cold
//! store with no derived evidence, and an immediate unchanged warm no-op. The
//! source commits are appended outside the timers. Store deltas quantify new
//! durable state avoided by an evolved store; a bench-local algebra wrapper
//! then performs one untimed, zero-write raw admission to expose algebra work
//! that store totals hide.
//! No scan or diagnostic touches a measured store before its first timed call
//! at a checkpoint, and the immediate no-op remains adjacent to that call.
//!
//! Final relations are materialized outside the timers into canonical
//! contiguous SimpleArchive bytes. Their cached content handles must match the
//! exact source prefix and the corresponding warm/cold/no-op arms. Physical
//! cover identities are reported but need not match: optional evidence may
//! choose another exact cover without changing the logical relation.
//!
//! Usage:
//!
//! ```text
//! cargo bench --bench collection_evolution -- \
//!   [--commits 64] [--rows-per-commit 1024] [--warmup 1] [--iters 4]
//! ```

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::{
    OrderedUniverse, SuccinctArchiveBlob, UnionArchive,
};
use triblespace_core::blob::Blob;
use triblespace_core::collection::exact_derived::{
    ExactAlgebraError, ExactDerivedAlgebra, ExactDerivedCollection,
};
use triblespace_core::collection::reach;
use triblespace_core::collection::succinctarchive_union::SuccinctArchiveCollection;
use triblespace_core::collection::{
    CollectionAdmission, CollectionCommit, CollectionHandle, CollectionRecord, CollectionStore,
};
use triblespace_core::inline::Encodes;
use triblespace_core::prelude::*;
use triblespace_core::repo::{ArtifactOfferStore, BlobStore, BlobStoreList, StoreRevision};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AlgebraCalls {
    validate_source: u64,
    validate_target: u64,
    join_source: u64,
    derive: u64,
    join_target: u64,
    input_bytes: u64,
}

struct CountingAlgebra<'a> {
    inner: &'a SuccinctArchiveCollection,
    calls: Cell<AlgebraCalls>,
}

impl<'a> CountingAlgebra<'a> {
    fn new(inner: &'a SuccinctArchiveCollection) -> Self {
        Self {
            inner,
            calls: Cell::new(AlgebraCalls::default()),
        }
    }

    fn bump(&self, update: impl FnOnce(&mut AlgebraCalls)) {
        let mut calls = self.calls.get();
        update(&mut calls);
        self.calls.set(calls);
    }
}

impl ExactDerivedAlgebra<SimpleArchive, SuccinctArchiveBlob> for CountingAlgebra<'_> {
    fn validate_source(
        &self,
        descriptor: &Fragment,
        source: &Blob<SimpleArchive>,
    ) -> Result<(), ExactAlgebraError> {
        self.bump(|calls| {
            calls.validate_source += 1;
            calls.input_bytes += source.bytes.len() as u64;
        });
        self.inner.validate_source(descriptor, source)
    }

    fn validate_target(
        &self,
        descriptor: &Fragment,
        target: &Blob<SuccinctArchiveBlob>,
    ) -> Result<(), ExactAlgebraError> {
        self.bump(|calls| {
            calls.validate_target += 1;
            calls.input_bytes += target.bytes.len() as u64;
        });
        self.inner.validate_target(descriptor, target)
    }

    fn join_source(
        &self,
        low: &Blob<SimpleArchive>,
        high: &Blob<SimpleArchive>,
    ) -> Result<Blob<SimpleArchive>, ExactAlgebraError> {
        self.bump(|calls| {
            calls.join_source += 1;
            calls.input_bytes += (low.bytes.len() + high.bytes.len()) as u64;
        });
        self.inner.join_source(low, high)
    }

    fn derive(
        &self,
        source: &Blob<SimpleArchive>,
    ) -> Result<Blob<SuccinctArchiveBlob>, ExactAlgebraError> {
        self.bump(|calls| {
            calls.derive += 1;
            calls.input_bytes += source.bytes.len() as u64;
        });
        self.inner.derive(source)
    }

    fn join_target(
        &self,
        low: &Blob<SuccinctArchiveBlob>,
        high: &Blob<SuccinctArchiveBlob>,
    ) -> Result<Blob<SuccinctArchiveBlob>, ExactAlgebraError> {
        self.bump(|calls| {
            calls.join_target += 1;
            calls.input_bytes += (low.bytes.len() + high.bytes.len()) as u64;
        });
        self.inner.join_target(low, high)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct StoreShape {
    blobs: u64,
    blob_bytes: u64,
    offers: u64,
    commits: u64,
    source_merges: u64,
    raw_derives: u64,
    raw_merges: u64,
    rank9_derives: u64,
    rank9_merges: u64,
    other_records: u64,
}

impl StoreShape {
    fn plus(self, other: Self) -> Self {
        Self {
            blobs: self.blobs + other.blobs,
            blob_bytes: self.blob_bytes + other.blob_bytes,
            offers: self.offers + other.offers,
            commits: self.commits + other.commits,
            source_merges: self.source_merges + other.source_merges,
            raw_derives: self.raw_derives + other.raw_derives,
            raw_merges: self.raw_merges + other.raw_merges,
            rank9_derives: self.rank9_derives + other.rank9_derives,
            rank9_merges: self.rank9_merges + other.rank9_merges,
            other_records: self.other_records + other.other_records,
        }
    }

    fn difference(self, before: Self) -> Self {
        Self {
            blobs: self.blobs - before.blobs,
            blob_bytes: self.blob_bytes - before.blob_bytes,
            offers: self.offers - before.offers,
            commits: self.commits - before.commits,
            source_merges: self.source_merges - before.source_merges,
            raw_derives: self.raw_derives - before.raw_derives,
            raw_merges: self.raw_merges - before.raw_merges,
            rank9_derives: self.rank9_derives - before.rank9_derives,
            rank9_merges: self.rank9_merges - before.rank9_merges,
            other_records: self.other_records - before.other_records,
        }
    }
}

struct Collections {
    source: CollectionHandle,
    raw: CollectionHandle,
    rank9: CollectionHandle,
}

fn store_shape(store: &mut MemoryRepo, collections: &Collections) -> StoreShape {
    let mut shape = StoreShape::default();
    for record in store.records().expect("enumerate collection records") {
        match record.expect("MemoryRepo collection records are infallible") {
            CollectionRecord::Commit(commit) if commit.collection() == collections.source => {
                shape.commits += 1;
            }
            CollectionRecord::Merge(merge) if merge.collection() == collections.source => {
                shape.source_merges += 1;
            }
            CollectionRecord::Derive(derive) if derive.target() == collections.raw => {
                shape.raw_derives += 1;
            }
            CollectionRecord::Merge(merge) if merge.collection() == collections.raw => {
                shape.raw_merges += 1;
            }
            CollectionRecord::Derive(derive) if derive.target() == collections.rank9 => {
                shape.rank9_derives += 1;
            }
            CollectionRecord::Merge(merge) if merge.collection() == collections.rank9 => {
                shape.rank9_merges += 1;
            }
            _ => shape.other_records += 1,
        }
    }

    let reader = store.reader().expect("snapshot MemoryRepo blobs");
    for info in reader.blobs() {
        let info = info.expect("MemoryRepo blob listing is infallible");
        shape.blobs += 1;
        shape.blob_bytes += info.length;
    }
    shape.offers = store
        .offers_snapshot()
        .expect("MemoryRepo offer snapshot is infallible")
        .len();
    shape
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RelationIdentity {
    rows: u64,
    hash: [u8; 32],
}

fn relation_identity(rows: impl IntoIterator<Item = Trible>) -> RelationIdentity {
    let set: TribleSet = rows.into_iter().collect();
    relation_identity_set(&set)
}

fn relation_identity_set(set: &TribleSet) -> RelationIdentity {
    // SimpleArchive is one canonical contiguous EAV sequence. Blob::new has
    // already hashed the complete buffer, so reuse its cached content handle.
    let archive = SimpleArchive::encode(set);
    RelationIdentity {
        rows: set.len() as u64,
        hash: archive.get_handle().raw,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CoverIdentity {
    members: u64,
    bytes: u64,
    hash: [u8; 32],
}

fn cover_identity(
    cover: &triblespace_core::collection::exact_derived::ExactCover<SuccinctArchiveBlob>,
) -> CoverIdentity {
    let mut hasher = blake3::Hasher::new();
    let mut bytes = 0u64;
    for (data, blob) in cover.members() {
        hasher.update(&data.raw);
        bytes += blob.bytes.len() as u64;
    }
    CoverIdentity {
        members: cover.len() as u64,
        bytes,
        hash: *hasher.finalize().as_bytes(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Ensure,
    Compact,
}

impl Operation {
    fn execute(
        self,
        succinct: &SuccinctArchiveCollection,
        store: &mut MemoryRepo,
        ticket: &[CollectionCommit],
    ) -> UnionArchive<OrderedUniverse> {
        match self {
            Self::Ensure => succinct
                .ensure_exact(store, ticket)
                .expect("ensure exact Succinct collection"),
            Self::Compact => succinct
                .compact_exact(store, ticket)
                .expect("compact exact Succinct collection"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Arm {
    EnsureWarm,
    EnsureCold,
    EnsureNoop,
    CompactWarm,
    CompactCold,
    CompactNoop,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::EnsureWarm => "ensure-warm",
            Self::EnsureCold => "ensure-cold",
            Self::EnsureNoop => "ensure-noop",
            Self::CompactWarm => "compact-warm",
            Self::CompactCold => "compact-cold",
            Self::CompactNoop => "compact-noop",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    arm: Arm,
    commits: usize,
    total_rows: u64,
    basis_rows: u64,
    elapsed: Duration,
    work: StoreShape,
    algebra_calls: AlgebraCalls,
    relation: RelationIdentity,
    cover: CoverIdentity,
}

struct TimedOperation {
    elapsed: Duration,
    union: UnionArchive<OrderedUniverse>,
}

struct RunContext<'a> {
    ticket: &'a [CollectionCommit],
    total_rows: u64,
    newly_supported_rows: u64,
    expected: RelationIdentity,
    succinct: &'a SuccinctArchiveCollection,
    exact: &'a ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob>,
    collections: &'a Collections,
}

fn time_operation(
    operation: Operation,
    store: &mut MemoryRepo,
    ticket: &[CollectionCommit],
    succinct: &SuccinctArchiveCollection,
) -> TimedOperation {
    let start = Instant::now();
    let union = operation.execute(succinct, store, ticket);
    let elapsed = start.elapsed();
    black_box(union.segment_count());
    TimedOperation { elapsed, union }
}

fn diagnose_raw(
    store: &mut MemoryRepo,
    ticket: &[CollectionCommit],
    succinct: &SuccinctArchiveCollection,
    exact: &ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob>,
) -> (AlgebraCalls, CoverIdentity) {
    // This is outside the timer and must be a zero-write operation: it measures
    // the scratch proof graph needed to re-admit the exact raw cover now held
    // by this arm. The public Rank9 phase remains represented only by timing
    // and its durable DERIVE/blob delta because its algebra is intentionally
    // private to the facade.
    let diagnostic_before = store
        .store_revision()
        .expect("snapshot pre-diagnostic store revision");
    let offers_before = store
        .offers_snapshot()
        .expect("snapshot pre-diagnostic artifact offers");
    let algebra = CountingAlgebra::new(succinct);
    let raw_cover = exact
        .ensure_exact(store, ticket, &algebra)
        .expect("diagnose complete raw exact cover");
    let algebra_calls = algebra.calls.get();
    let cover = cover_identity(&raw_cover);
    let diagnostic_after = store
        .store_revision()
        .expect("snapshot post-diagnostic store revision");
    assert!(
        diagnostic_after == diagnostic_before,
        "post-operation raw algebra diagnostic wrote sync-visible storage"
    );
    assert_eq!(
        store
            .offers_snapshot()
            .expect("snapshot post-diagnostic artifact offers"),
        offers_before,
        "post-operation raw algebra diagnostic published an artifact offer"
    );
    (algebra_calls, cover)
}

fn finish_sample(
    arm: Arm,
    context: &RunContext<'_>,
    basis_rows: u64,
    timed: TimedOperation,
    work: StoreShape,
    diagnostic: (AlgebraCalls, CoverIdentity),
) -> Sample {
    let (algebra_calls, cover) = diagnostic;
    let relation = relation_identity(timed.union.iter());
    Sample {
        arm,
        commits: context.ticket.len(),
        total_rows: context.total_rows,
        basis_rows,
        elapsed: timed.elapsed,
        work,
        algebra_calls,
        relation,
        cover,
    }
}

fn run_warm_pair(
    operation: Operation,
    store: &mut MemoryRepo,
    context: &RunContext<'_>,
    before: StoreShape,
) -> ([Sample; 2], StoreShape) {
    let (warm_arm, noop_arm) = match operation {
        Operation::Ensure => (Arm::EnsureWarm, Arm::EnsureNoop),
        Operation::Compact => (Arm::CompactWarm, Arm::CompactNoop),
    };
    let timed_warm = time_operation(operation, store, context.ticket, context.succinct);
    let revision_after_warm = store
        .store_revision()
        .expect("snapshot revision between warm and no-op calls");
    let offers_after_warm = store
        .offers_snapshot()
        .expect("snapshot offers between warm and no-op calls");

    // Keep these public calls adjacent. In particular, do not materialize the
    // first result or replay the exact proof before timing the unchanged call.
    let timed_noop = time_operation(operation, store, context.ticket, context.succinct);
    let revision_after_noop = store
        .store_revision()
        .expect("snapshot revision after no-op call");
    let offers_after_noop = store
        .offers_snapshot()
        .expect("snapshot offers after no-op call");
    assert!(
        revision_after_noop == revision_after_warm,
        "an unchanged public operation changed sync-visible storage"
    );
    assert_eq!(
        offers_after_noop, offers_after_warm,
        "an unchanged public operation published an artifact offer"
    );

    let after = store_shape(store, context.collections);
    let warm_work = after.difference(before);
    let diagnostic = diagnose_raw(store, context.ticket, context.succinct, context.exact);
    let warm = finish_sample(
        warm_arm,
        context,
        context.newly_supported_rows,
        timed_warm,
        warm_work,
        diagnostic,
    );
    assert_eq!(warm.relation, context.expected);

    let noop = finish_sample(
        noop_arm,
        context,
        context.total_rows,
        timed_noop,
        StoreShape::default(),
        diagnostic,
    );
    assert_eq!(noop.relation, context.expected);
    ([warm, noop], after)
}

fn run_cold(
    operation: Operation,
    arm: Arm,
    store: &mut MemoryRepo,
    context: &RunContext<'_>,
    before: StoreShape,
) -> Sample {
    let timed = time_operation(operation, store, context.ticket, context.succinct);
    let after = store_shape(store, context.collections);
    let diagnostic = diagnose_raw(store, context.ticket, context.succinct, context.exact);
    let cold = finish_sample(
        arm,
        context,
        context.total_rows,
        timed,
        after.difference(before),
        diagnostic,
    );
    assert_eq!(cold.relation, context.expected);
    cold
}

fn run_operation_family(
    iteration: usize,
    operation: Operation,
    warm_store: &mut MemoryRepo,
    cold_store: &mut MemoryRepo,
    context: &RunContext<'_>,
    baselines: [StoreShape; 2],
) -> (Vec<Sample>, StoreShape) {
    let [warm_before, cold_before] = baselines;
    let cold_arm = match operation {
        Operation::Ensure => Arm::EnsureCold,
        Operation::Compact => Arm::CompactCold,
    };
    let (warm_pair, warm_after, cold) = if iteration.is_multiple_of(2) {
        let (warm_pair, warm_after) = run_warm_pair(operation, warm_store, context, warm_before);
        let cold = run_cold(operation, cold_arm, cold_store, context, cold_before);
        (warm_pair, warm_after, cold)
    } else {
        let cold = run_cold(operation, cold_arm, cold_store, context, cold_before);
        let (warm_pair, warm_after) = run_warm_pair(operation, warm_store, context, warm_before);
        (warm_pair, warm_after, cold)
    };
    (vec![warm_pair[0], cold, warm_pair[1]], warm_after)
}

fn geometric_checkpoints(commits: usize) -> Vec<usize> {
    let mut checkpoints = Vec::new();
    let mut next = 1usize;
    while next < commits {
        checkpoints.push(next);
        next = next.saturating_mul(2);
    }
    checkpoints.push(commits);
    checkpoints
}

fn make_chunk(commit: usize, rows: usize) -> TribleSet {
    let mut chunk = TribleSet::new();
    for row in 0..rows {
        let ordinal = (commit as u64)
            .checked_mul(rows as u64)
            .and_then(|base| base.checked_add(row as u64))
            .expect("benchmark ordinal fits u64");
        let mut raw = [0u8; 64];
        raw[..8].copy_from_slice(&(ordinal + 1).to_be_bytes());
        raw[8..16].copy_from_slice(&0xE001_0000_0000_0001u64.to_be_bytes());
        raw[16..24].copy_from_slice(&0xA001_0000_0000_0001u64.to_be_bytes());
        raw[24..32].copy_from_slice(&(commit as u64 + 1).to_be_bytes());
        raw[32..40].copy_from_slice(&ordinal.rotate_left(17).to_be_bytes());
        raw[40..48].copy_from_slice(&ordinal.wrapping_mul(31).to_be_bytes());
        raw[48..56].copy_from_slice(&(commit as u64).to_be_bytes());
        raw[56..64].copy_from_slice(&(row as u64).to_be_bytes());
        chunk.insert(&Trible::force_raw(raw).expect("non-nil entity and attribute"));
    }
    chunk
}

fn benchmark_name() -> CollectionName {
    CollectionName::new("evolving-succinct-benchmark").expect("legal collection name")
}

fn benchmark_namespace() -> ed25519_dalek::VerifyingKey {
    SigningKey::from_bytes(&[0x71; 32]).verifying_key()
}

fn new_source_collection() -> Collection<MemoryRepo> {
    Collection::new(
        MemoryRepo::default(),
        &benchmark_name(),
        benchmark_namespace(),
        SigningKey::from_bytes(&[0x71; 32]),
        reach::private(),
        CollectionAdmission::Open,
    )
}

fn publish_same_chunk(chunk: &TribleSet, collections: &mut [&mut Collection<MemoryRepo>]) {
    let mut expected = None;
    for collection in collections {
        let commit = collection
            .commit(Fragment::from(chunk.clone()))
            .expect("publish source commit");
        match expected {
            None => expected = Some(commit),
            Some(expected) => {
                assert_eq!(
                    commit, expected,
                    "identical source publications must converge"
                )
            }
        }
    }
}

fn run_iteration(
    iteration: usize,
    chunks: &[TribleSet],
    checkpoints: &[usize],
    succinct: &SuccinctArchiveCollection,
) -> Vec<Sample> {
    let mut source_accounting = new_source_collection();
    let mut cold_ensure_source = new_source_collection();
    let mut cold_compact_source = new_source_collection();
    let mut warm_ensure = new_source_collection();
    let mut warm_compact = new_source_collection();
    let exact = ExactDerivedCollection::<SimpleArchive, SuccinctArchiveBlob>::new(
        succinct.source_descriptor(),
        succinct.descriptor(),
    );
    let collections = Collections {
        source: succinct.source_collection(),
        raw: succinct.collection(),
        rank9: succinct.rank9_collection(),
    };

    let mut published = 0usize;
    let mut previous_rows = 0u64;
    let mut expected = TribleSet::new();
    let mut samples = Vec::with_capacity(checkpoints.len() * 6);
    let mut ensure_derived_shape = StoreShape::default();
    let mut compact_derived_shape = StoreShape::default();
    for &checkpoint in checkpoints {
        for chunk in &chunks[published..checkpoint] {
            expected.union(chunk.clone());
            publish_same_chunk(
                chunk,
                &mut [
                    &mut source_accounting,
                    &mut cold_ensure_source,
                    &mut cold_compact_source,
                    &mut warm_ensure,
                    &mut warm_compact,
                ],
            );
        }
        published = checkpoint;

        let ticket = source_accounting
            .ticket()
            .expect("freeze accounting source ticket");
        assert_eq!(ticket.len(), checkpoint);
        let source_shape = store_shape(source_accounting.storage_mut(), &collections);

        let total_rows = expected.len() as u64;
        let newly_supported_rows = total_rows - previous_rows;
        previous_rows = total_rows;
        let expected_identity = relation_identity_set(&expected);
        let context = RunContext {
            ticket: &ticket,
            total_rows,
            newly_supported_rows,
            expected: expected_identity,
            succinct,
            exact: &exact,
            collections: &collections,
        };

        let mut cold_ensure = cold_ensure_source.storage().clone();
        let mut cold_compact = cold_compact_source.storage().clone();
        let ensure_before = source_shape.plus(ensure_derived_shape);
        let compact_before = source_shape.plus(compact_derived_shape);
        if iteration.is_multiple_of(2) {
            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Ensure,
                warm_ensure.storage_mut(),
                &mut cold_ensure,
                &context,
                [ensure_before, source_shape],
            );
            samples.extend(family);
            ensure_derived_shape = warm_after.difference(source_shape);

            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Compact,
                warm_compact.storage_mut(),
                &mut cold_compact,
                &context,
                [compact_before, source_shape],
            );
            samples.extend(family);
            compact_derived_shape = warm_after.difference(source_shape);
        } else {
            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Compact,
                warm_compact.storage_mut(),
                &mut cold_compact,
                &context,
                [compact_before, source_shape],
            );
            samples.extend(family);
            compact_derived_shape = warm_after.difference(source_shape);

            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Ensure,
                warm_ensure.storage_mut(),
                &mut cold_ensure,
                &context,
                [ensure_before, source_shape],
            );
            samples.extend(family);
            ensure_derived_shape = warm_after.difference(source_shape);
        }
    }
    samples
}

#[derive(Default)]
struct Aggregate {
    elapsed_ns: Vec<u128>,
    work: Option<StoreShape>,
    algebra_calls: Option<AlgebraCalls>,
    total_rows: u64,
    basis_rows: u64,
    relation: Option<RelationIdentity>,
    cover: Option<CoverIdentity>,
}

impl Aggregate {
    fn push(&mut self, sample: Sample) {
        self.elapsed_ns.push(sample.elapsed.as_nanos());
        match self.work {
            None => self.work = Some(sample.work),
            Some(expected) => assert_eq!(expected, sample.work, "store work changed across runs"),
        }
        match self.algebra_calls {
            None => self.algebra_calls = Some(sample.algebra_calls),
            Some(expected) => assert_eq!(
                expected, sample.algebra_calls,
                "algebra work changed across runs",
            ),
        }
        if self.total_rows == 0 {
            self.total_rows = sample.total_rows;
            self.basis_rows = sample.basis_rows;
            self.relation = Some(sample.relation);
            self.cover = Some(sample.cover);
        } else {
            assert_eq!(self.total_rows, sample.total_rows);
            assert_eq!(self.basis_rows, sample.basis_rows);
            assert_eq!(self.relation, Some(sample.relation));
            assert_eq!(self.cover, Some(sample.cover));
        }
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

fn short_hash(hash: &[u8; 32]) -> String {
    let mut output = String::with_capacity(12);
    for byte in &hash[..6] {
        write!(&mut output, "{byte:02X}").expect("writing to String is infallible");
    }
    output
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
    let mut rows_per_commit = 1_024usize;
    let mut warmup = 1usize;
    let mut iterations = 4usize;
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--commits" => commits = parse_usize(&args, &mut index, "--commits"),
            "--rows-per-commit" => {
                rows_per_commit = parse_usize(&args, &mut index, "--rows-per-commit")
            }
            "--warmup" => warmup = parse_usize(&args, &mut index, "--warmup"),
            "--iters" => iterations = parse_usize(&args, &mut index, "--iters"),
            "--bench" => {}
            other => panic!("unknown option {other:?}"),
        }
        index += 1;
    }
    assert!(commits > 0, "--commits must be nonzero");
    assert!(rows_per_commit > 0, "--rows-per-commit must be nonzero");
    assert!(iterations > 0, "--iters must be nonzero");

    let checkpoints = geometric_checkpoints(commits);
    let chunks: Vec<_> = (0..commits)
        .map(|commit| make_chunk(commit, rows_per_commit))
        .collect();
    let succinct = SuccinctArchiveCollection::new(
        benchmark_name(),
        benchmark_namespace(),
        None,
        reach::private(),
        None,
        reach::private(),
    );
    println!(
        "config   : commits={commits} rows/commit={rows_per_commit} warmup={warmup} iters={iterations} checkpoints={checkpoints:?}"
    );
    println!(
        "timing   : source publication and all diagnostics excluded; median of {iterations} whole runs"
    );
    println!(
        "cold     : derived-evidence cold, not CPU-cache cold; warm/cold order alternates by measured run"
    );

    for iteration in 0..warmup {
        black_box(run_iteration(iteration, &chunks, &checkpoints, &succinct));
    }

    let mut aggregates = BTreeMap::<(usize, Arm), Aggregate>::new();
    for iteration in 0..iterations {
        for sample in run_iteration(iteration, &chunks, &checkpoints, &succinct) {
            aggregates
                .entry((sample.commits, sample.arm))
                .or_default()
                .push(sample);
        }
    }

    let arms = [
        Arm::EnsureWarm,
        Arm::EnsureCold,
        Arm::EnsureNoop,
        Arm::CompactWarm,
        Arm::CompactCold,
        Arm::CompactNoop,
    ];
    println!(
        "\n{:>7} {:>13} {:>11} {:>14} {:>10} {:>8}",
        "commits", "arm", "median-ms", "ns/basis-row", "basis-rows", "cover",
    );
    for &checkpoint in &checkpoints {
        for arm in arms {
            let aggregate = &aggregates[&(checkpoint, arm)];
            let elapsed = median(&aggregate.elapsed_ns);
            println!(
                "{:>7} {:>13} {:>11.3} {:>14.1} {:>10} {:>8}",
                checkpoint,
                arm.label(),
                elapsed as f64 / 1_000_000.0,
                elapsed as f64 / aggregate.basis_rows.max(1) as f64,
                aggregate.basis_rows,
                aggregate.cover.expect("cover identity").members,
            );
        }
    }

    println!(
        "\nwork columns: +B=blobs, +bytes=blob payload, +O=offers, +D=raw derives, +M=raw merges, +R=Rank9 derives; algebra=vs/vt/d/js/jt calls and cumulative argument MiB in one untimed zero-write raw re-admission"
    );
    println!(
        "{:>7} {:>13} {:>4} {:>10} {:>4} {:>4} {:>4} {:>4} {:>26} {:>10}",
        "commits",
        "arm",
        "+B",
        "+bytes",
        "+O",
        "+D",
        "+M",
        "+R",
        "algebra vs/vt/d/js/jt",
        "arg-MiB",
    );
    for &checkpoint in &checkpoints {
        for arm in arms {
            let aggregate = &aggregates[&(checkpoint, arm)];
            let work = aggregate.work.expect("store work");
            let calls = aggregate.algebra_calls.expect("algebra calls");
            println!(
                "{:>7} {:>13} {:>4} {:>10} {:>4} {:>4} {:>4} {:>4} {:>26} {:>10.2}",
                checkpoint,
                arm.label(),
                work.blobs,
                work.blob_bytes,
                work.offers,
                work.raw_derives,
                work.raw_merges,
                work.rank9_derives,
                format!(
                    "{}/{}/{}/{}/{}",
                    calls.validate_source,
                    calls.validate_target,
                    calls.derive,
                    calls.join_source,
                    calls.join_target,
                ),
                calls.input_bytes as f64 / (1024.0 * 1024.0),
            );
            assert_eq!(work.commits, 0, "measured operation wrote a COMMIT");
            assert_eq!(
                work.source_merges, 0,
                "measured operation compacted the source collection",
            );
            assert_eq!(work.rank9_merges, 0, "Rank9 fibers must remain one-to-one");
            assert_eq!(work.other_records, 0, "unclassified record write");
        }
    }

    println!("\nidentity : canonical logical EAV equality verified against every source prefix");
    for &checkpoint in &checkpoints {
        let relation = aggregates[&(checkpoint, Arm::EnsureWarm)]
            .relation
            .expect("relation identity");
        for arm in arms {
            assert_eq!(aggregates[&(checkpoint, arm)].relation, Some(relation));
        }
        let ensure_warm = aggregates[&(checkpoint, Arm::EnsureWarm)]
            .cover
            .expect("ensure warm cover");
        let ensure_cold = aggregates[&(checkpoint, Arm::EnsureCold)]
            .cover
            .expect("ensure cold cover");
        let compact_warm = aggregates[&(checkpoint, Arm::CompactWarm)]
            .cover
            .expect("compact warm cover");
        let compact_cold = aggregates[&(checkpoint, Arm::CompactCold)]
            .cover
            .expect("compact cold cover");
        println!(
            "  commits={checkpoint:<7} rows={:<9} logical={} ensure-physical={} ({}/{}) compact-physical={} ({}/{})",
            relation.rows,
            short_hash(&relation.hash),
            if ensure_warm.hash == ensure_cold.hash {
                "same"
            } else {
                "different"
            },
            ensure_warm.members,
            ensure_cold.members,
            if compact_warm.hash == compact_cold.hash {
                "same"
            } else {
                "different"
            },
            compact_warm.members,
            compact_cold.members,
        );
    }
}
