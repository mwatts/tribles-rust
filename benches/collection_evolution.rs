//! Evolving-cover maintenance benchmark for canonical Succinct collections.
//!
//! This benchmark compares the three real public maintenance operations on
//! geometrically growing exact covers:
//!
//! - `ensure_exact`: canonical raw Succinct derivation plus Rank9 fibers.
//! - `exact_view().ensure`: retain already-admitted immutable shards and run
//!   ordinary exact admission only over newly signed support.
//! - `compact_exact`: the same exact completion plus deterministic dyadic
//!   raw-target compaction and Rank9 fibers for the selected cover.
//!
//! Stateless operations get an independent warm store, a source-identical cold
//! store with no derived evidence, and an immediate unchanged warm no-op. The
//! maintained view gets its own evolving store and immediate no-op. Source
//! commits are appended outside the timers. Store deltas quantify new durable
//! state; a bench-local mapping wrapper performs one untimed, zero-write
//! raw fresh attachment for stateless arms to expose replay work that store
//! totals hide. The maintained view reports the canonical projection calls
//! made during its timed observation, so continuation reuse is measured rather
//! than inferred from durable writes. It is intentionally not replayed after
//! timing: that would measure the stateless operation it exists to avoid, not
//! its actual work.
//! No scan or diagnostic touches a measured store before its first timed call
//! at a checkpoint, and the immediate no-op remains adjacent to that call.
//!
//! Final relations are materialized outside the timers into canonical
//! contiguous SimpleArchive bytes. Their cached content handles must match the
//! exact source prefix and the corresponding warm/cold/no-op arms. Stateless
//! raw-cover identities and maintained-view segment counts are reported, but
//! physical covers need not match: optional evidence may choose another exact
//! cover without changing the logical relation.
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
use triblespace_core::collection::exact_derived::ExactDerivedCollection;
use triblespace_core::collection::reach;
use triblespace_core::collection::succinctarchive_union::{
    rank9_mapping_fragment, SimpleToSuccinctMapping, SuccinctArchiveCollection,
    SuccinctArchiveView, SuccinctArchiveViewWork,
};
use triblespace_core::collection::{
    simplearchive_union, Collection, CollectionHandle, CollectionMapping, CollectionOperationError,
    CollectionRecord, CollectionStore, CollectionStoreExt, Cover, CoverAttachment,
    MappingEvidenceStore, MappingHandle,
};
use triblespace_core::inline::Encodes;
use triblespace_core::prelude::*;
use triblespace_core::repo::{ArtifactOfferStore, BlobStore, BlobStoreList, StoreRevision};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MappingCalls {
    derive: u64,
    input_bytes: u64,
}

thread_local! {
    static MAPPING_CALLS: Cell<MappingCalls> =
        const { Cell::new(MappingCalls { derive: 0, input_bytes: 0 }) };
}

fn reset_mapping_calls() {
    MAPPING_CALLS.set(MappingCalls::default());
}

fn mapping_calls() -> MappingCalls {
    MAPPING_CALLS.get()
}

struct CountingSuccinctMapping {
    inner: SimpleToSuccinctMapping,
}

impl CollectionMapping<SimpleArchive, SuccinctArchiveBlob> for CountingSuccinctMapping {
    fn bind(source: &Fragment, target: &Fragment) -> Result<Self, CollectionOperationError> {
        Ok(Self {
            inner: SimpleToSuccinctMapping::bind(source, target)?,
        })
    }

    fn map(
        &self,
        source: &Blob<SimpleArchive>,
    ) -> Result<Blob<SuccinctArchiveBlob>, CollectionOperationError> {
        MAPPING_CALLS.with(|slot| {
            let mut calls = slot.get();
            calls.derive += 1;
            calls.input_bytes += source.bytes.len() as u64;
            slot.set(calls);
        });
        self.inner.map(source)
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
    rank9_evidence: u64,
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
            rank9_evidence: self.rank9_evidence + other.rank9_evidence,
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
            rank9_evidence: self.rank9_evidence - before.rank9_evidence,
            other_records: self.other_records - before.other_records,
        }
    }
}

struct Collections {
    source: CollectionHandle,
    raw: CollectionHandle,
    rank9_mapping: MappingHandle,
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
            CollectionRecord::Derive(derive) if derive.collection() == collections.raw => {
                shape.raw_derives += 1;
            }
            CollectionRecord::Merge(merge) if merge.collection() == collections.raw => {
                shape.raw_merges += 1;
            }
            _ => shape.other_records += 1,
        }
    }

    for evidence in store.evidence().expect("enumerate mapping evidence") {
        let evidence = evidence.expect("MemoryRepo mapping evidence is infallible");
        if evidence.mapping() == collections.rank9_mapping {
            shape.rank9_evidence += 1;
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

fn cover_identity(cover: &CoverAttachment<SuccinctArchiveBlob>) -> CoverIdentity {
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
        cover: &Cover<SimpleArchive>,
    ) -> UnionArchive<OrderedUniverse> {
        match self {
            Self::Ensure => succinct
                .ensure_exact(store, cover)
                .expect("ensure exact Succinct collection"),
            Self::Compact => succinct
                .compact_exact(store, cover)
                .expect("compact exact Succinct collection"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Arm {
    EnsureWarm,
    EnsureCold,
    EnsureNoop,
    ViewAdvance,
    ViewNoop,
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
            Self::ViewAdvance => "view-advance",
            Self::ViewNoop => "view-noop",
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
    diagnostic: Diagnostic,
    relation: RelationIdentity,
    cover_members: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Diagnostic {
    StatelessReplay {
        calls: MappingCalls,
        cover: CoverIdentity,
    },
    ViewActual(SuccinctArchiveViewWork),
}

struct TimedOperation {
    elapsed: Duration,
    union: UnionArchive<OrderedUniverse>,
}

struct RunContext<'a> {
    cover: &'a Cover<SimpleArchive>,
    total_rows: u64,
    newly_supported_rows: u64,
    expected: RelationIdentity,
    succinct: &'a SuccinctArchiveCollection,
    exact: &'a ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob, CountingSuccinctMapping>,
    collections: &'a Collections,
}

fn time_operation(
    operation: Operation,
    store: &mut MemoryRepo,
    cover: &Cover<SimpleArchive>,
    succinct: &SuccinctArchiveCollection,
) -> TimedOperation {
    let start = Instant::now();
    let union = operation.execute(succinct, store, cover);
    let elapsed = start.elapsed();
    black_box(union.segment_count());
    TimedOperation { elapsed, union }
}

fn diagnose_raw(
    store: &mut MemoryRepo,
    cover: &Cover<SimpleArchive>,
    exact: &ExactDerivedCollection<SimpleArchive, SuccinctArchiveBlob, CountingSuccinctMapping>,
) -> (MappingCalls, CoverIdentity) {
    // This is outside the timer and must be a zero-write operation: it measures
    // the scratch proof graph needed to revalidate the exact raw cover now held
    // by this arm. The public Rank9 phase remains represented only by timing
    // and its durable mapping-evidence/blob delta because fiber construction is
    // intentionally private to the facade.
    let diagnostic_before = store
        .store_revision()
        .expect("snapshot pre-diagnostic store revision");
    let offers_before = store
        .offers_snapshot()
        .expect("snapshot pre-diagnostic artifact offers");
    reset_mapping_calls();
    let raw_cover = exact
        .ensure_exact(store, cover)
        .expect("diagnose complete raw exact cover");
    let projection_calls = mapping_calls();
    let cover = cover_identity(&raw_cover);
    let diagnostic_after = store
        .store_revision()
        .expect("snapshot post-diagnostic store revision");
    assert!(
        diagnostic_after == diagnostic_before,
        "post-operation raw mapping diagnostic wrote sync-visible storage"
    );
    assert_eq!(
        store
            .offers_snapshot()
            .expect("snapshot post-diagnostic artifact offers"),
        offers_before,
        "post-operation raw mapping diagnostic published an artifact offer"
    );
    (projection_calls, cover)
}

fn finish_sample(
    arm: Arm,
    context: &RunContext<'_>,
    basis_rows: u64,
    timed: TimedOperation,
    work: StoreShape,
    diagnostic: Diagnostic,
) -> Sample {
    let cover_members = timed.union.segment_count() as u64;
    let relation = relation_identity(timed.union.iter());
    Sample {
        arm,
        commits: context.cover.len(),
        total_rows: context.total_rows,
        basis_rows,
        elapsed: timed.elapsed,
        work,
        diagnostic,
        relation,
        cover_members,
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
    let timed_warm = time_operation(operation, store, context.cover, context.succinct);
    let revision_after_warm = store
        .store_revision()
        .expect("snapshot revision between warm and no-op calls");
    let offers_after_warm = store
        .offers_snapshot()
        .expect("snapshot offers between warm and no-op calls");

    // Keep these public calls adjacent. In particular, do not materialize the
    // first result or replay the exact proof before timing the unchanged call.
    let timed_noop = time_operation(operation, store, context.cover, context.succinct);
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
    let diagnostic = diagnose_raw(store, context.cover, context.exact);
    let warm = finish_sample(
        warm_arm,
        context,
        context.newly_supported_rows,
        timed_warm,
        warm_work,
        Diagnostic::StatelessReplay {
            calls: diagnostic.0,
            cover: diagnostic.1,
        },
    );
    assert_eq!(warm.relation, context.expected);

    let noop = finish_sample(
        noop_arm,
        context,
        context.total_rows,
        timed_noop,
        StoreShape::default(),
        Diagnostic::StatelessReplay {
            calls: diagnostic.0,
            cover: diagnostic.1,
        },
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
    let timed = time_operation(operation, store, context.cover, context.succinct);
    let after = store_shape(store, context.collections);
    let diagnostic = diagnose_raw(store, context.cover, context.exact);
    let cold = finish_sample(
        arm,
        context,
        context.total_rows,
        timed,
        after.difference(before),
        Diagnostic::StatelessReplay {
            calls: diagnostic.0,
            cover: diagnostic.1,
        },
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

fn time_exact_view(
    view: &mut SuccinctArchiveView,
    store: &mut MemoryRepo,
    cover: &Cover<SimpleArchive>,
) -> TimedOperation {
    let start = Instant::now();
    let union = view
        .ensure(store, cover)
        .expect("advance maintained exact Succinct view");
    let elapsed = start.elapsed();
    black_box(union.segment_count());
    TimedOperation { elapsed, union }
}

fn run_exact_view_pair(
    view: &mut SuccinctArchiveView,
    store: &mut MemoryRepo,
    context: &RunContext<'_>,
    before: StoreShape,
) -> ([Sample; 2], StoreShape) {
    let timed_advance = time_exact_view(view, store, context.cover);
    let advance_work = view
        .last_work()
        .expect("successful view advance records actual work");
    assert_eq!(advance_work.cover_members, context.cover.len());
    assert_eq!(
        advance_work.processed_members + advance_work.reused_members,
        context.cover.len(),
        "view support accounting must cover the exact payload set",
    );
    let revision_after_advance = store
        .store_revision()
        .expect("snapshot revision between view advance and no-op");
    let offers_after_advance = store
        .offers_snapshot()
        .expect("snapshot offers between view advance and no-op");

    let timed_noop = time_exact_view(view, store, context.cover);
    let noop_work = view
        .last_work()
        .expect("successful view no-op records actual work");
    assert_eq!(
        noop_work,
        SuccinctArchiveViewWork {
            cover_members: context.cover.len(),
            processed_members: 0,
            reused_members: context.cover.len(),
            ..SuccinctArchiveViewWork::default()
        },
        "an identical exact-view cover must not replay raw proof work",
    );
    assert!(
        store
            .store_revision()
            .expect("snapshot revision after view no-op")
            == revision_after_advance,
        "an unchanged exact view changed sync-visible storage",
    );
    assert_eq!(
        store
            .offers_snapshot()
            .expect("snapshot offers after view no-op"),
        offers_after_advance,
        "an unchanged exact view published an artifact offer",
    );

    let after = store_shape(store, context.collections);
    let advance = finish_sample(
        Arm::ViewAdvance,
        context,
        context.newly_supported_rows,
        timed_advance,
        after.difference(before),
        Diagnostic::ViewActual(advance_work),
    );
    let noop = finish_sample(
        Arm::ViewNoop,
        context,
        context.total_rows,
        timed_noop,
        StoreShape::default(),
        Diagnostic::ViewActual(noop_work),
    );
    assert_eq!(advance.relation, context.expected);
    assert_eq!(noop.relation, context.expected);
    ([advance, noop], after)
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

fn benchmark_name() -> &'static str {
    "evolving-succinct-benchmark"
}

fn benchmark_authority() -> ed25519_dalek::VerifyingKey {
    SigningKey::from_bytes(&[0x71; 32]).verifying_key()
}

fn new_source_store(source: Collection<SimpleArchive>) -> MemoryRepo {
    let mut store = MemoryRepo::default();
    let registered = store
        .collection(simplearchive_union::descriptor(
            benchmark_name(),
            benchmark_authority(),
            reach::private(),
        ))
        .expect("register benchmark source collection");
    assert_eq!(registered, source);
    store
}

fn publish_same_chunk(
    chunk: &TribleSet,
    source: Collection<SimpleArchive>,
    signing_key: &SigningKey,
    stores: &mut [&mut MemoryRepo],
) {
    let mut expected = None;
    for store in stores {
        let commit = store
            .commit(source, signing_key, Fragment::from(chunk.clone()))
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
    let mut exact_view = succinct.exact_view();
    let exact =
        ExactDerivedCollection::<SimpleArchive, SuccinctArchiveBlob, CountingSuccinctMapping>::new(
            succinct.source_descriptor(),
            succinct.descriptor(),
        )
        .expect("bind measured raw Succinct projection");
    let source = exact.source_collection();
    let collections = Collections {
        source: source.handle(),
        raw: succinct.collection(),
        rank9_mapping: rank9_mapping_fragment()
            .facts()
            .clone()
            .to_blob()
            .get_handle(),
    };
    let signing_key = SigningKey::from_bytes(&[0x71; 32]);
    let mut source_accounting = new_source_store(source);
    let mut cold_ensure_source = new_source_store(source);
    let mut cold_compact_source = new_source_store(source);
    let mut warm_ensure = new_source_store(source);
    let mut exact_view_source = new_source_store(source);
    let mut warm_compact = new_source_store(source);

    let mut published = 0usize;
    let mut previous_rows = 0u64;
    let mut expected = TribleSet::new();
    let mut samples = Vec::with_capacity(checkpoints.len() * 8);
    let mut ensure_derived_shape = StoreShape::default();
    let mut view_derived_shape = StoreShape::default();
    let mut compact_derived_shape = StoreShape::default();
    for &checkpoint in checkpoints {
        for chunk in &chunks[published..checkpoint] {
            expected.union(chunk.clone());
            publish_same_chunk(
                chunk,
                source,
                &signing_key,
                &mut [
                    &mut source_accounting,
                    &mut cold_ensure_source,
                    &mut cold_compact_source,
                    &mut warm_ensure,
                    &mut exact_view_source,
                    &mut warm_compact,
                ],
            );
        }
        published = checkpoint;

        let cover = source_accounting
            .cover(source, &[])
            .expect("freeze accounting source cover");
        assert_eq!(cover.len(), checkpoint);
        let source_shape = store_shape(&mut source_accounting, &collections);

        let total_rows = expected.len() as u64;
        let newly_supported_rows = total_rows - previous_rows;
        previous_rows = total_rows;
        let expected_identity = relation_identity_set(&expected);
        let context = RunContext {
            cover: &cover,
            total_rows,
            newly_supported_rows,
            expected: expected_identity,
            succinct,
            exact: &exact,
            collections: &collections,
        };

        let mut cold_ensure = cold_ensure_source.clone();
        let mut cold_compact = cold_compact_source.clone();
        let ensure_before = source_shape.plus(ensure_derived_shape);
        let view_before = source_shape.plus(view_derived_shape);
        let compact_before = source_shape.plus(compact_derived_shape);
        if iteration.is_multiple_of(2) {
            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Ensure,
                &mut warm_ensure,
                &mut cold_ensure,
                &context,
                [ensure_before, source_shape],
            );
            samples.extend(family);
            ensure_derived_shape = warm_after.difference(source_shape);

            let (pair, after) = run_exact_view_pair(
                &mut exact_view,
                &mut exact_view_source,
                &context,
                view_before,
            );
            samples.extend(pair);
            view_derived_shape = after.difference(source_shape);

            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Compact,
                &mut warm_compact,
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
                &mut warm_compact,
                &mut cold_compact,
                &context,
                [compact_before, source_shape],
            );
            samples.extend(family);
            compact_derived_shape = warm_after.difference(source_shape);

            let (pair, after) = run_exact_view_pair(
                &mut exact_view,
                &mut exact_view_source,
                &context,
                view_before,
            );
            samples.extend(pair);
            view_derived_shape = after.difference(source_shape);

            let (family, warm_after) = run_operation_family(
                iteration,
                Operation::Ensure,
                &mut warm_ensure,
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
    diagnostic: Option<Diagnostic>,
    total_rows: u64,
    basis_rows: u64,
    relation: Option<RelationIdentity>,
    cover_members: u64,
}

impl Aggregate {
    fn push(&mut self, sample: Sample) {
        self.elapsed_ns.push(sample.elapsed.as_nanos());
        match self.work {
            None => self.work = Some(sample.work),
            Some(expected) => assert_eq!(expected, sample.work, "store work changed across runs"),
        }
        match self.diagnostic {
            None => self.diagnostic = Some(sample.diagnostic),
            Some(expected) => assert_eq!(
                expected, sample.diagnostic,
                "diagnostic work changed across runs",
            ),
        }
        if self.total_rows == 0 {
            self.total_rows = sample.total_rows;
            self.basis_rows = sample.basis_rows;
            self.relation = Some(sample.relation);
            self.cover_members = sample.cover_members;
        } else {
            assert_eq!(self.total_rows, sample.total_rows);
            assert_eq!(self.basis_rows, sample.basis_rows);
            assert_eq!(self.relation, Some(sample.relation));
            assert_eq!(self.cover_members, sample.cover_members);
        }
    }
}

fn raw_cover(aggregate: &Aggregate) -> CoverIdentity {
    match aggregate.diagnostic.expect("diagnostic observation") {
        Diagnostic::StatelessReplay { cover, .. } => cover,
        Diagnostic::ViewActual(_) => panic!("view arm has no stateless raw-cover replay"),
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
        benchmark_authority(),
        reach::private(),
        benchmark_authority(),
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
        Arm::ViewAdvance,
        Arm::ViewNoop,
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
                aggregate.cover_members,
            );
        }
    }

    println!(
        "\nwork columns: +B=blobs, +bytes=blob payload, +O=offers, +D=raw derives, +M=raw merges, +R=Rank9 mapping evidence; maps=canonical source-to-target mapping calls and cumulative input MiB (stateless=replayed after timing, view=actual timed observation); support=admitted/reused commits for views"
    );
    println!(
        "{:>7} {:>13} {:>4} {:>10} {:>4} {:>4} {:>4} {:>4} {:>15} {:>26} {:>10}",
        "commits",
        "arm",
        "+B",
        "+bytes",
        "+O",
        "+D",
        "+M",
        "+R",
        "support",
        "projection maps",
        "arg-MiB",
    );
    for &checkpoint in &checkpoints {
        for arm in arms {
            let aggregate = &aggregates[&(checkpoint, arm)];
            let work = aggregate.work.expect("store work");
            let (support, calls, argument_mib) =
                match aggregate.diagnostic.expect("diagnostic observation") {
                    Diagnostic::StatelessReplay { calls, .. } => (
                        "replay/full".to_owned(),
                        calls.derive.to_string(),
                        format!("{:.2}", calls.input_bytes as f64 / (1024.0 * 1024.0)),
                    ),
                    Diagnostic::ViewActual(work) => (
                        format!("{}/{}", work.processed_members, work.reused_members),
                        work.derive.to_string(),
                        format!("{:.2}", work.input_bytes as f64 / (1024.0 * 1024.0)),
                    ),
                };
            println!(
                "{:>7} {:>13} {:>4} {:>10} {:>4} {:>4} {:>4} {:>4} {:>15} {:>26} {:>10}",
                checkpoint,
                arm.label(),
                work.blobs,
                work.blob_bytes,
                work.offers,
                work.raw_derives,
                work.raw_merges,
                work.rank9_evidence,
                support,
                calls,
                argument_mib,
            );
            assert_eq!(work.commits, 0, "measured operation wrote a COMMIT");
            assert_eq!(
                work.source_merges, 0,
                "measured operation compacted the source collection",
            );
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
        let ensure_warm = raw_cover(&aggregates[&(checkpoint, Arm::EnsureWarm)]);
        let ensure_cold = raw_cover(&aggregates[&(checkpoint, Arm::EnsureCold)]);
        let compact_warm = raw_cover(&aggregates[&(checkpoint, Arm::CompactWarm)]);
        let compact_cold = raw_cover(&aggregates[&(checkpoint, Arm::CompactCold)]);
        let view_members = aggregates[&(checkpoint, Arm::ViewAdvance)].cover_members;
        println!(
            "  commits={checkpoint:<7} rows={:<9} logical={} ensure-physical={} ({}/{}) view-members={} compact-physical={} ({}/{})",
            relation.rows,
            short_hash(&relation.hash),
            if ensure_warm.hash == ensure_cold.hash {
                "same"
            } else {
                "different"
            },
            ensure_warm.members,
            ensure_cold.members,
            view_members,
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
