//! Faculty-shaped comparison of legacy `Repository` branches and native
//! `Collection`s over the same `Pile` implementation and fact workload.
//!
//! Unlike a revision-to-revision benchmark, this keeps compiler, PATCH, pile
//! parser, hashing, and `SimpleArchive` implementation constant. The variable
//! is the persisted coordination model: a CAS branch over a commit ancestry or
//! an unordered set of signed collection commits.
//!
//! The harness deliberately separates fresh-handle pile opening, pile refresh
//! and validation, semantic materialization on both fresh and already-open
//! handles, query execution over an already-materialized set, and an actual
//! subprocess-shaped command. "Cold" here means a fresh process or `Pile`
//! handle; the OS page cache is not flushed.
//!
//! Write rows use the same durability cadence: both models preserve ordered
//! publication in memory and make the entire measured batch durable when the
//! pile closes. Neither ordinary path hides per-commit durability barriers.
//!
//! Run a representative quick matrix:
//!
//! ```text
//! cargo bench -p triblespace-core --bench faculty_collection_storage -- --quick
//! ```
//!
//! Run the default matrix or select cases and repetitions:
//!
//! ```text
//! cargo bench -p triblespace-core --bench faculty_collection_storage -- \
//!   --cases 256/1,4096/16,4096/256,65536/256,65536/1024 --reps 21
//! ```

use triblespace_core::collection::reach;
use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use tempfile::{tempdir, TempDir};
use triblespace_core::collection::records::CollectionName;
use triblespace_core::collection::{
    discover_collection_records, Collection, CollectionStore, VerifyingKey,
};
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::genid::GenId;
use triblespace_core::inline::InlineEncoding;
use triblespace_core::metadata;
use triblespace_core::prelude::{find, pattern, Trible, TribleSet};
use triblespace_core::repo::pile::{Pile, PileRecordContent, PileRecords};
use triblespace_core::repo::Repository;
use triblespace_core::trible::Fragment;

const KEY_BYTES: [u8; 32] = [0x5A; 32];
const APPEND_BATCH: usize = 32;

#[derive(Clone, Copy, Debug)]
struct Case {
    facts: usize,
    commits: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct RecordCounts {
    bytes: u64,
    records: usize,
    blobs: usize,
    branches: usize,
    collections: usize,
    other: usize,
}

#[derive(Debug)]
struct Fixture {
    _dir: TempDir,
    legacy: PathBuf,
    collection: PathBuf,
    branch: Id,
    name: CollectionName,
    expected: TribleSet,
}

#[derive(Clone, Copy, Debug)]
struct Summary {
    median: Duration,
    p10: Duration,
    p90: Duration,
    mean: Duration,
}

#[derive(Clone, Copy, Debug)]
struct ValueSummary {
    median: f64,
    p10: f64,
    p90: f64,
    mean: f64,
}

#[derive(Debug)]
struct Args {
    cases: Vec<Case>,
    reps: usize,
    append_reps: usize,
}

fn main() {
    if env::args().nth(1).as_deref() == Some("__child") {
        child_command();
        return;
    }
    if env::args().nth(1).as_deref() == Some("__profile") {
        profile_command();
        return;
    }
    let args = parse_args();
    eprintln!(
        "faculty_collection_storage: release={} arch={} os={} reps={} append_reps={} (fresh handles, warm OS cache)",
        !cfg!(debug_assertions),
        env::consts::ARCH,
        env::consts::OS,
        args.reps,
        args.append_reps,
    );
    println!(
        "facts,commits,model,pile_bytes,records,blobs,branches,collection_records,metric,unit,median,p10,p90,mean"
    );

    for case in args.cases {
        let fixture = build_fixture(case);
        let legacy_counts = record_counts(&fixture.legacy);
        let collection_counts = record_counts(&fixture.collection);

        validate_fixture(&fixture);

        bench_read_matrix(
            case,
            "legacy",
            &fixture.legacy,
            legacy_counts,
            args.reps,
            || legacy_materialize(&fixture.legacy, fixture.branch),
        );
        bench_read_matrix(
            case,
            "collection",
            &fixture.collection,
            collection_counts,
            args.reps,
            || collection_materialize(&fixture.collection, &fixture.name),
        );

        emit(
            case,
            "legacy",
            legacy_counts,
            "hot_semantic_materialize",
            hot_legacy_materialize(&fixture.legacy, fixture.branch, args.reps),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "hot_semantic_materialize",
            hot_collection_materialize(&fixture.collection, &fixture.name, args.reps),
        );

        emit(
            case,
            "legacy",
            legacy_counts,
            "fresh_handle_pull",
            sample(args.reps, || {
                legacy_pull_only(&fixture.legacy, fixture.branch)
            }),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "fresh_handle_record_discovery",
            sample(args.reps, || collection_discovery_only(&fixture.collection)),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "hot_record_enumeration_raw",
            hot_collection_enumeration(&fixture.collection, args.reps),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "hot_record_discovery",
            hot_collection_discovery(&fixture.collection, args.reps),
        );

        let legacy_facts = legacy_materialize(&fixture.legacy, fixture.branch);
        let collection_facts = collection_materialize(&fixture.collection, &fixture.name);
        emit(
            case,
            "legacy",
            legacy_counts,
            "query_materialized",
            sample(args.reps, || query_all(black_box(&legacy_facts))),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "query_materialized",
            sample(args.reps, || query_all(black_box(&collection_facts))),
        );

        // A real faculty invocation also pays process and Rust-runtime startup.
        // Use this very executable as the child so code generation stays
        // identical to the in-process comparison.
        let executable = env::current_exe().expect("benchmark executable path");
        let (legacy_cold, legacy_rss) = sample_subprocess(args.reps, || {
            subprocess_query(
                &executable,
                "legacy",
                &fixture.legacy,
                &format!("{:X}", fixture.branch),
            )
        });
        let (collection_cold, collection_rss) = sample_subprocess(args.reps, || {
            subprocess_query(
                &executable,
                "collection",
                &fixture.collection,
                fixture.name.as_str(),
            )
        });
        emit(
            case,
            "legacy",
            legacy_counts,
            "cold_subprocess_query",
            legacy_cold,
        );
        emit_value(
            case,
            "legacy",
            legacy_counts,
            "cold_subprocess_peak_rss",
            "bytes",
            legacy_rss,
        );
        emit(
            case,
            "collection",
            collection_counts,
            "cold_subprocess_query",
            collection_cold,
        );
        emit_value(
            case,
            "collection",
            collection_counts,
            "cold_subprocess_peak_rss",
            "bytes",
            collection_rss,
        );

        emit(
            case,
            "legacy",
            legacy_counts,
            "append_one_command",
            sample_measured(args.append_reps, || {
                append_one_clone_legacy(&fixture.legacy, fixture.branch, case.facts)
            }),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "append_one_command",
            sample_measured(args.append_reps, || {
                append_one_clone_collection(&fixture.collection, &fixture.name, case.facts)
            }),
        );

        emit(
            case,
            "legacy",
            legacy_counts,
            "append_32_open",
            sample_measured(args.append_reps, || {
                append_many_clone_legacy(&fixture.legacy, fixture.branch, case.facts, APPEND_BATCH)
            }),
        );
        emit(
            case,
            "collection",
            collection_counts,
            "append_32_open",
            sample_measured(args.append_reps, || {
                append_many_clone_collection(
                    &fixture.collection,
                    &fixture.name,
                    case.facts,
                    APPEND_BATCH,
                )
            }),
        );
    }
}

fn parse_args() -> Args {
    let mut cases = vec![
        Case {
            facts: 256,
            commits: 1,
        },
        Case {
            facts: 4_096,
            commits: 1,
        },
        Case {
            facts: 4_096,
            commits: 16,
        },
        Case {
            facts: 4_096,
            commits: 256,
        },
        Case {
            facts: 65_536,
            commits: 1,
        },
        Case {
            facts: 65_536,
            commits: 16,
        },
        Case {
            facts: 65_536,
            commits: 256,
        },
        Case {
            facts: 65_536,
            commits: 1_024,
        },
    ];
    let mut reps = 21;
    let mut append_reps = 7;
    let mut argv = env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--quick" => {
                cases = vec![
                    Case {
                        facts: 256,
                        commits: 1,
                    },
                    Case {
                        facts: 4_096,
                        commits: 1,
                    },
                    Case {
                        facts: 4_096,
                        commits: 16,
                    },
                    Case {
                        facts: 4_096,
                        commits: 256,
                    },
                ];
                reps = 9;
                append_reps = 3;
            }
            "--cases" => cases = parse_cases(&argv.next().expect("--cases requires a value")),
            "--reps" => {
                reps = argv
                    .next()
                    .expect("--reps requires a value")
                    .parse()
                    .unwrap()
            }
            "--append-reps" => {
                append_reps = argv
                    .next()
                    .expect("--append-reps requires a value")
                    .parse()
                    .unwrap()
            }
            "--help" | "-h" => {
                println!(
                    "usage: faculty_collection_storage [--quick] [--cases FACTS/COMMITS,...] [--reps N] [--append-reps N]"
                );
                std::process::exit(0);
            }
            // `cargo bench` passes libtest-style flags even to harness-free
            // targets on some toolchains. They do not change this executable.
            "--bench" => {}
            other if other.starts_with("--profile-time") => {}
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(reps > 0 && append_reps > 0);
    Args {
        cases,
        reps,
        append_reps,
    }
}

fn child_command() {
    let mut args = env::args().skip(2);
    let model = args.next().expect("child model");
    let path = PathBuf::from(args.next().expect("child pile path"));
    // The selector is a branch id for the legacy model and a collection name
    // for the collection model; each side parses what it needs.
    let selector = args.next().expect("child branch id or collection name");
    let facts = match model.as_str() {
        "legacy" => legacy_materialize(&path, Id::from_hex(&selector).expect("child branch id")),
        "collection" => collection_materialize(
            &path,
            &CollectionName::new(&selector).expect("child collection name"),
        ),
        _ => panic!("unknown child model {model}"),
    };
    let count = query_all(&facts);
    println!("{count},{}", peak_rss_bytes());
}

fn profile_command() {
    let mut args = env::args().skip(2);
    let model = args.next().expect("profile model");
    let facts: usize = args.next().expect("profile facts").parse().unwrap();
    let commits: usize = args.next().expect("profile commits").parse().unwrap();
    let seconds: f64 = args.next().expect("profile seconds").parse().unwrap();
    let fixture = build_fixture(Case { facts, commits });
    eprintln!("READY pid={}", std::process::id());
    std::thread::sleep(Duration::from_secs(2));
    let started = Instant::now();
    let mut iterations = 0_u64;
    while started.elapsed().as_secs_f64() < seconds {
        match model.as_str() {
            "legacy" => black_box(legacy_materialize(&fixture.legacy, fixture.branch)),
            "collection" => black_box(collection_materialize(&fixture.collection, &fixture.name)),
            _ => panic!("unknown profile model {model}"),
        };
        iterations += 1;
    }
    eprintln!("DONE iterations={iterations}");
}

fn subprocess_query(executable: &Path, model: &str, pile: &Path, selector: &str) -> (usize, u64) {
    let output = Command::new(executable)
        .arg("__child")
        .arg(model)
        .arg(pile)
        .arg(selector)
        .output()
        .expect("spawn child command");
    assert!(
        output.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = String::from_utf8(output.stdout).unwrap();
    let (count, rss) = output.trim().split_once(',').expect("child count,rss");
    (count.parse().unwrap(), rss.parse().unwrap())
}

#[cfg(target_os = "macos")]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    // Darwin reports ru_maxrss in bytes.
    unsafe { usage.assume_init().ru_maxrss as u64 }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    // Linux and the common BSD interfaces report ru_maxrss in KiB.
    unsafe { usage.assume_init().ru_maxrss as u64 * 1024 }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

fn parse_cases(raw: &str) -> Vec<Case> {
    raw.split(',')
        .map(|case| {
            let (facts, commits) = case
                .split_once('/')
                .unwrap_or_else(|| panic!("case must be FACTS/COMMITS: {case}"));
            let case = Case {
                facts: facts.parse().unwrap(),
                commits: commits.parse().unwrap(),
            };
            assert!(case.facts > 0);
            assert!(case.commits > 0 && case.commits <= case.facts);
            case
        })
        .collect()
}

fn fixed_id(namespace: u64, ordinal: usize) -> Id {
    let mut raw = [0_u8; 16];
    raw[..8].copy_from_slice(&namespace.to_be_bytes());
    raw[8..].copy_from_slice(&(ordinal as u64 + 1).to_be_bytes());
    Id::new(raw).expect("nonzero deterministic id")
}

fn fact(ordinal: usize) -> Trible {
    let entity = fixed_id(1, ordinal);
    let value = GenId::inline_from(fixed_id(2, ordinal));
    Trible::force(&entity, &metadata::tag.id(), &value)
}

fn fact_set(start: usize, len: usize) -> TribleSet {
    let mut facts = TribleSet::new();
    for ordinal in start..start + len {
        facts.insert(&fact(ordinal));
    }
    facts
}

fn fragments(total: usize, commits: usize) -> Vec<Fragment> {
    let base = total / commits;
    let remainder = total % commits;
    let mut start = 0;
    (0..commits)
        .map(|index| {
            let len = base + usize::from(index < remainder);
            let fragment = Fragment::from(fact_set(start, len));
            start += len;
            fragment
        })
        .collect()
}

fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&KEY_BYTES)
}

/// The benchmark collection's name, and its team of one: the same key that
/// signs every commit here.
fn collection_name() -> CollectionName {
    CollectionName::new("faculty-benchmark").expect("legal collection name")
}

fn collection_team() -> VerifyingKey {
    signing_key().verifying_key()
}

fn open_refreshed(path: &Path) -> Pile {
    let mut pile = Pile::open(path).expect("open pile");
    pile.refresh().expect("refresh pile");
    pile
}

fn build_fixture(case: Case) -> Fixture {
    let dir = tempdir().expect("fixture tempdir");
    let legacy = dir.path().join("legacy.pile");
    let collection = dir.path().join("collection.pile");
    File::create(&legacy).unwrap();
    File::create(&collection).unwrap();

    let expected = fact_set(0, case.facts);
    let fragments = fragments(case.facts, case.commits);

    let pile = open_refreshed(&legacy);
    let mut repo = Repository::new(pile, signing_key(), TribleSet::new()).unwrap();
    let branch = *repo.create_branch("faculty-benchmark", None).unwrap();
    for (index, fragment) in fragments.iter().cloned().enumerate() {
        let mut workspace = repo.pull(branch).unwrap();
        workspace.commit(fragment, &format!("faculty commit {index}"));
        repo.push(&mut workspace).unwrap();
    }
    repo.close().unwrap();

    let name = collection_name();
    let pile = open_refreshed(&collection);
    let mut collection_facade = Collection::new(pile, &name, collection_team(), signing_key(), reach::private());
    for fragment in fragments {
        collection_facade.commit(fragment).unwrap();
    }
    collection_facade.close().unwrap();

    Fixture {
        _dir: dir,
        legacy,
        collection,
        branch,
        name,
        expected,
    }
}

fn validate_fixture(fixture: &Fixture) {
    let legacy = legacy_materialize(&fixture.legacy, fixture.branch);
    let collection = collection_materialize(&fixture.collection, &fixture.name);
    assert_eq!(legacy, fixture.expected, "legacy fixture changed semantics");
    assert_eq!(
        collection, fixture.expected,
        "collection fixture changed semantics"
    );
    assert_eq!(legacy, collection, "storage models disagree");
    assert_eq!(query_all(&legacy), fixture.expected.len());
    assert_eq!(query_all(&collection), fixture.expected.len());
}

fn legacy_materialize(path: &Path, branch: Id) -> TribleSet {
    let pile = open_refreshed(path);
    let mut repo = Repository::new(pile, signing_key(), TribleSet::new()).unwrap();
    let facts = repo
        .pull(branch)
        .unwrap()
        .checkout(..)
        .unwrap()
        .into_facts();
    repo.close().unwrap();
    facts
}

fn collection_materialize(path: &Path, name: &CollectionName) -> TribleSet {
    let pile = open_refreshed(path);
    let mut collection = Collection::new(pile, name, collection_team(), signing_key(), reach::private());
    let facts = collection.materialize().unwrap();
    collection.close().unwrap();
    facts
}

fn legacy_pull_only(path: &Path, branch: Id) -> usize {
    let pile = open_refreshed(path);
    let mut repo = Repository::new(pile, signing_key(), TribleSet::new()).unwrap();
    let workspace = repo.pull(branch).unwrap();
    let present = black_box(workspace.head().is_some()) as usize;
    drop(workspace);
    repo.close().unwrap();
    present
}

fn collection_discovery_only(path: &Path) -> usize {
    let mut pile = open_refreshed(path);
    let count = black_box(
        discover_collection_records(&mut pile)
            .unwrap()
            .commits()
            .len(),
    );
    pile.close().unwrap();
    count
}

fn collection_enumeration_only(path: &Path) -> usize {
    let mut pile = open_refreshed(path);
    let count = pile.records().unwrap().count();
    pile.close().unwrap();
    black_box(count)
}

fn hot_legacy_materialize(path: &Path, branch: Id, reps: usize) -> Summary {
    let pile = open_refreshed(path);
    let mut repo = Repository::new(pile, signing_key(), TribleSet::new()).unwrap();
    let summary = sample(reps, || {
        repo.pull(branch)
            .unwrap()
            .checkout(..)
            .unwrap()
            .into_facts()
            .len()
    });
    repo.close().unwrap();
    summary
}

fn hot_collection_materialize(path: &Path, name: &CollectionName, reps: usize) -> Summary {
    let pile = open_refreshed(path);
    let mut collection = Collection::new(pile, name, collection_team(), signing_key(), reach::private());
    let summary = sample(reps, || collection.materialize().unwrap().len());
    collection.close().unwrap();
    summary
}

fn hot_collection_enumeration(path: &Path, reps: usize) -> Summary {
    let mut pile = open_refreshed(path);
    let summary = sample(reps, || pile.records().unwrap().count());
    pile.close().unwrap();
    summary
}

fn hot_collection_discovery(path: &Path, reps: usize) -> Summary {
    let mut pile = open_refreshed(path);
    let summary = sample(reps, || {
        discover_collection_records(&mut pile)
            .unwrap()
            .commits()
            .len()
    });
    pile.close().unwrap();
    summary
}

fn query_all(facts: &TribleSet) -> usize {
    find!(
        (entity: Id, tag: Id),
        pattern!(facts, [{ ?entity @ metadata::tag: ?tag }])
    )
    .count()
}

fn bench_read_matrix<F>(
    case: Case,
    model: &str,
    path: &Path,
    counts: RecordCounts,
    reps: usize,
    mut semantic: F,
) where
    F: FnMut() -> TribleSet,
{
    emit(
        case,
        model,
        counts,
        "open_map_only",
        sample(reps, || {
            let pile = Pile::open(path).unwrap();
            pile.close().unwrap();
        }),
    );
    if model == "collection" {
        emit(
            case,
            model,
            counts,
            "fresh_handle_record_enumeration_raw",
            sample(reps, || collection_enumeration_only(path)),
        );
    }
    emit(
        case,
        model,
        counts,
        "refresh_validation",
        sample(reps, || {
            let mut pile = Pile::open(path).unwrap();
            pile.refresh().unwrap();
            pile.close().unwrap();
        }),
    );
    emit(
        case,
        model,
        counts,
        "fresh_handle_materialize",
        sample(reps, || {
            let facts = semantic();
            black_box(facts.len())
        }),
    );
    emit(
        case,
        model,
        counts,
        "fresh_handle_query",
        sample(reps, || {
            let facts = semantic();
            black_box(query_all(&facts))
        }),
    );
}

fn clone_fixture(source: &Path) -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let destination = dir.path().join("sample.pile");
    #[cfg(target_os = "macos")]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let source_c = CString::new(source.as_os_str().as_bytes()).unwrap();
        let destination_c = CString::new(destination.as_os_str().as_bytes()).unwrap();
        // `clonefile` is copy-on-write on APFS and happens outside every timed
        // region. Fall back to a byte copy when the filesystem cannot clone.
        let cloned = unsafe { libc::clonefile(source_c.as_ptr(), destination_c.as_ptr(), 0) } == 0;
        if !cloned {
            fs::copy(source, &destination).unwrap();
        }
    }
    #[cfg(not(target_os = "macos"))]
    fs::copy(source, &destination).unwrap();
    (dir, destination)
}

fn append_one_clone_legacy(source: &Path, branch: Id, ordinal: usize) -> Duration {
    let (_dir, path) = clone_fixture(source);
    let started = Instant::now();
    let pile = open_refreshed(&path);
    let mut repo = Repository::new(pile, signing_key(), TribleSet::new()).unwrap();
    let mut workspace = repo.pull(branch).unwrap();
    workspace.commit(Fragment::from(fact_set(ordinal, 1)), "append one");
    repo.push(&mut workspace).unwrap();
    repo.close().unwrap();
    started.elapsed()
}

fn append_one_clone_collection(source: &Path, name: &CollectionName, ordinal: usize) -> Duration {
    let (_dir, path) = clone_fixture(source);
    let started = Instant::now();
    let pile = open_refreshed(&path);
    let mut collection = Collection::new(pile, name, collection_team(), signing_key(), reach::private());
    collection
        .commit(Fragment::from(fact_set(ordinal, 1)))
        .unwrap();
    collection.close().unwrap();
    started.elapsed()
}

fn append_many_clone_legacy(source: &Path, branch: Id, ordinal: usize, count: usize) -> Duration {
    let (_dir, path) = clone_fixture(source);
    let started = Instant::now();
    let pile = open_refreshed(&path);
    let mut repo = Repository::new(pile, signing_key(), TribleSet::new()).unwrap();
    for offset in 0..count {
        let mut workspace = repo.pull(branch).unwrap();
        workspace.commit(
            Fragment::from(fact_set(ordinal + offset, 1)),
            "append repeated",
        );
        repo.push(&mut workspace).unwrap();
    }
    repo.close().unwrap();
    started.elapsed()
}

fn append_many_clone_collection(
    source: &Path,
    name: &CollectionName,
    ordinal: usize,
    count: usize,
) -> Duration {
    let (_dir, path) = clone_fixture(source);
    let started = Instant::now();
    let pile = open_refreshed(&path);
    let mut collection = Collection::new(pile, name, collection_team(), signing_key(), reach::private());
    for offset in 0..count {
        collection
            .commit(Fragment::from(fact_set(ordinal + offset, 1)))
            .unwrap();
    }
    collection.close().unwrap();
    started.elapsed()
}

fn record_counts(path: &Path) -> RecordCounts {
    let mut result = RecordCounts {
        bytes: fs::metadata(path).unwrap().len(),
        ..Default::default()
    };
    for record in PileRecords::open(path).unwrap() {
        let record = record.unwrap();
        result.records += 1;
        match record.content {
            PileRecordContent::Blob { .. } => result.blobs += 1,
            PileRecordContent::Branch { .. } | PileRecordContent::BranchTombstone { .. } => {
                result.branches += 1
            }
            PileRecordContent::Collection { .. } => result.collections += 1,
            _ => result.other += 1,
        }
    }
    result
}

fn sample<T, F>(reps: usize, mut operation: F) -> Summary
where
    F: FnMut() -> T,
{
    // One untimed warmup makes allocator and page-fault startup less dominant.
    black_box(operation());
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        black_box(operation());
        samples.push(started.elapsed());
    }
    summarize(samples)
}

fn sample_measured<F>(reps: usize, mut operation: F) -> Summary
where
    F: FnMut() -> Duration,
{
    // Fixture cloning is intentionally outside the duration returned by the
    // operation, so append numbers contain only the command-shaped storage
    // operation and not benchmark reset machinery.
    black_box(operation());
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        samples.push(black_box(operation()));
    }
    summarize(samples)
}

fn sample_subprocess<F>(reps: usize, mut operation: F) -> (Summary, ValueSummary)
where
    F: FnMut() -> (usize, u64),
{
    black_box(operation());
    let mut durations = Vec::with_capacity(reps);
    let mut rss = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        let (count, peak_rss) = operation();
        black_box(count);
        durations.push(started.elapsed());
        rss.push(peak_rss);
    }
    (summarize(durations), summarize_values(rss))
}

fn summarize(mut samples: Vec<Duration>) -> Summary {
    samples.sort_unstable();
    let len = samples.len();
    let sum: Duration = samples.iter().copied().sum();
    Summary {
        median: samples[len / 2],
        p10: samples[(len - 1) / 10],
        p90: samples[((len - 1) * 9) / 10],
        mean: sum / len as u32,
    }
}

fn summarize_values(mut samples: Vec<u64>) -> ValueSummary {
    samples.sort_unstable();
    let len = samples.len();
    let sum: u128 = samples.iter().map(|value| *value as u128).sum();
    ValueSummary {
        median: samples[len / 2] as f64,
        p10: samples[(len - 1) / 10] as f64,
        p90: samples[((len - 1) * 9) / 10] as f64,
        mean: sum as f64 / len as f64,
    }
}

fn emit(case: Case, model: &str, counts: RecordCounts, metric: &str, summary: Summary) {
    println!(
        "{},{},{},{},{},{},{},{},{},us,{:.3},{:.3},{:.3},{:.3}",
        case.facts,
        case.commits,
        model,
        counts.bytes,
        counts.records,
        counts.blobs,
        counts.branches,
        counts.collections,
        metric,
        micros(summary.median),
        micros(summary.p10),
        micros(summary.p90),
        micros(summary.mean),
    );
}

fn emit_value(
    case: Case,
    model: &str,
    counts: RecordCounts,
    metric: &str,
    unit: &str,
    summary: ValueSummary,
) {
    println!(
        "{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
        case.facts,
        case.commits,
        model,
        counts.bytes,
        counts.records,
        counts.blobs,
        counts.branches,
        counts.collections,
        metric,
        unit,
        summary.median,
        summary.p10,
        summary.p90,
        summary.mean,
    );
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}
