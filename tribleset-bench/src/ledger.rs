//! Results ledger: canonical telemetry sessions/spans plus bench
//! outcome entities, written DIRECTLY (no tracing) through the pinned
//! native-collection LEDGER dependency — measurement I/O never depends
//! on the era of the subject.
//!
//! DISCIPLINE: this module uses ONLY the ledger's umbrella surface
//! (`triblespace::prelude`, `triblespace::core::…`). The umbrella
//! macros expand to absolute `::triblespace::core` paths, which in
//! this crate resolve to the ledger; the core-macro flavor (what the
//! subject-side modules use) expands to `::triblespace_core`, which
//! resolves to the SUBJECT's core and must never appear here.
//!
//! The telemetry schema ids are declared byte-for-byte identical to
//! `june-on-tip/src/telemetry.rs` — the minted ids are the contract;
//! GORBIE's telemetry-viewer renders the axis.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{anyhow, bail, Context, Result};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::collection::{
    discover_collection_records, resolve_collection_semantics, simplearchive_union, Collection,
    CollectionClaimValidation, CollectionData, CollectionHandle, CollectionName,
    CollectionValidationRequest, SimpleArchiveCollection,
};
use triblespace::core::metadata;
use triblespace::core::repo::pile::Pile;
use triblespace::prelude::blobencodings::LongString;
use triblespace::prelude::inlineencodings::{Handle, ShortString};
use triblespace::prelude::*;

/// Canonical telemetry attributes — byte-for-byte the ids of
/// `triblespace::telemetry::schema` (read from
/// `june-on-tip/src/telemetry.rs`).
pub mod tele {
    use triblespace::prelude::blobencodings::LongString;
    use triblespace::prelude::inlineencodings::{GenId, Handle, ShortString, U256BE};
    use triblespace::prelude::*;

    attributes! {
        "3E062AA7E3554C8F2DB94883CE639BFE" unsafe as pub session: GenId;
        "146E5AA2F7CB3D8B654BC7742A13CAB3" unsafe as pub parent: GenId;
        "CCB0147D20C4C6FCAC0E3D87FAFF71D1" unsafe as pub name: Handle<LongString>;
        "8A4BE2C4D0E90D2B9EE0E1A07ECA2CFA" unsafe as pub category: ShortString;
        "E11A84A30CC112650DC860B66B8BD8A9" unsafe as pub begin_ns: U256BE;
        "2786FA563372FB6EF469EC7710719A49" unsafe as pub end_ns: U256BE;
        "7593602383D0B0D21BBE382A67E5BD9F" unsafe as pub duration_ns: U256BE;
        "7E96DD9A0B5002796B645ED25F5E99AC" unsafe as pub source: Handle<LongString>;
    }
}

/// Bench decorations — minted for this suite (`trible genid`, never
/// guessed): session provenance (commit/engine/config) and per-measure
/// outcome entities (of_run/workload/outcome/rows).
pub mod bench {
    use triblespace::prelude::blobencodings::LongString;
    use triblespace::prelude::inlineencodings::{GenId, Handle, ShortString, U256BE};
    use triblespace::prelude::*;

    attributes! {
        /// Subject git rev (short=12) the session measured.
        "2C96F6429B3E772B15A0AB630C2B394F" unsafe as pub commit: ShortString;
        /// Engine label (--label) naming the subject on the axis.
        "2C899A2497B9565328A42A44996BD6A1" unsafe as pub engine: ShortString;
        /// Full run configuration: CLI + dataset + suite crate version.
        "8A3D02A290208D39DC18C69FAF38F1E1" unsafe as pub config: Handle<LongString>;
        /// Outcome entity -> its session.
        "75342A5BCA3BAD27285C5B76DB22CFCF" unsafe as pub of_run: GenId;
        /// Outcome entity -> measure key (e.g. "harkonnen/F5/total").
        "81ADFDA915ABA850EE23FEE3B88FC02F" unsafe as pub workload: Handle<LongString>;
        /// signal | skip:<reason> | panic:<reason> | gate_fail:<reason>.
        "5ACAF4FD8D71F0205694F646520707B5" unsafe as pub outcome: ShortString;
        /// Result cardinality of the measure, where meaningful.
        "B5A378BDC1A7F1C4576B2DC6902B5995" unsafe as pub rows: U256BE;
    }
}

/// Tag id of a telemetry session entity.
pub static KIND_SESSION: LazyLock<Id> =
    LazyLock::new(|| Id::from_hex("2701F7019B865D461F0169B1303026D6").expect("kind_session id"));
/// Tag id of a telemetry span entity.
pub static KIND_SPAN: LazyLock<Id> =
    LazyLock::new(|| Id::from_hex("0AF9FEB9A2BFEB1BE8A8229829181085").expect("kind_span id"));
/// Name the benchmark-results collection is known by within its team.
pub static RESULTS_COLLECTION_NAME: LazyLock<CollectionName> = LazyLock::new(|| {
    CollectionName::new("tribleset-bench-results").expect("results collection name")
});

/// Published root key of the team the benchmark-results collection belongs to.
///
/// A root collection is anchored by a name *and* a team, so the suite needs a
/// team key that is the same on every machine and in every run — each run signs
/// its commit with a throwaway key, and `verify` authorizes every strictly
/// self-signed commit in the collection, so this key gates nothing and there is
/// no secret to protect. It only has to pick out one collection.
///
/// It is derived from the id this suite already minted for its results
/// (`F6D99F76BC15E78C0BBD44F9D28A0C0A`, the extrinsic *scope* back when that was
/// how a root was anchored) rather than minting a second constant, so the
/// results collection carries its own provenance forward under the new anchor.
/// Result piles written under the scope anchor are not reachable through it;
/// they predate the naming migration.
pub static RESULTS_COLLECTION_TEAM: LazyLock<VerifyingKey> = LazyLock::new(|| {
    let minted = Id::from_hex("F6D99F76BC15E78C0BBD44F9D28A0C0A").expect("results collection id");
    let mut seed = [0u8; 32];
    seed[..16].copy_from_slice(&minted.raw());
    seed[16..].copy_from_slice(&minted.raw());
    SigningKey::from_bytes(&seed).verifying_key()
});

/// The read-side facade naming the results collection.
fn results_collection() -> SimpleArchiveCollection {
    SimpleArchiveCollection::new(RESULTS_COLLECTION_NAME.clone(), *RESULTS_COLLECTION_TEAM)
}

/// Clip a string to a ShortString-safe payload: first line, NULs
/// stripped, at most 32 bytes on a char boundary.
fn clip32(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").replace('\0', "");
    let mut end = line.len().min(32);
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    line[..end].to_owned()
}

/// Complete schema description stored beside every results element.
///
/// `entity!` already carries descriptions for the attributes it uses. Adding
/// the two tag entities and the full attribute namespaces here also makes a
/// partial run fragment intelligible without depending on Rust source or on a
/// separate repository metadata commit.
fn ledger_metadata() -> Fragment {
    let mut description = tele::describe();
    description += bench::describe();
    description += entity! { ExclusiveId::force_ref(&*KIND_SESSION) @
        metadata::name: "telemetry_session",
        metadata::description:
            "A benchmark session grouping raw timing spans and measure outcomes.",
        metadata::tag: metadata::KIND_TAG,
    };
    description += entity! { ExclusiveId::force_ref(&*KIND_SPAN) @
        metadata::name: "telemetry_span",
        metadata::description:
            "One raw measured iteration with begin, end, and duration in nanoseconds.",
        metadata::tag: metadata::KIND_TAG,
    };
    description
}

/// One open results collection. All facts and their attachments accumulate in
/// a self-contained fragment; `finish` publishes one independent signed commit
/// and closes the pile.
pub struct ResultsLedger {
    collection: Collection<Pile>,
    session: Id,
    pending: Fragment,
}

impl ResultsLedger {
    /// Open (creating the file as needed) and start a session entity decorated
    /// with the bench provenance attributes.
    pub fn open(path: &Path, commit: &str, label: &str, config: &str) -> Result<Self> {
        if !path.exists() {
            std::fs::OpenOptions::new()
                .create_new(true)
                .append(true)
                .open(path)
                .with_context(|| format!("create results pile {}", path.display()))?;
        }
        let mut pile =
            Pile::open(path).map_err(|e| anyhow!("open results pile {}: {e:?}", path.display()))?;
        pile.refresh()
            .map_err(|e| anyhow!("load results pile: {e:?}"))?;
        let collection = Collection::new(
            pile,
            &RESULTS_COLLECTION_NAME,
            *RESULTS_COLLECTION_TEAM,
            SigningKey::generate(&mut OsRng),
        );

        let session_owner = genid();
        let session = *session_owner;
        let mut pending = Fragment::empty();
        let name_handle = pending.put("tribleset-bench".to_string());
        let config_handle = pending.put(config.to_string());
        let commit_short = clip32(commit);
        let label_short = clip32(label);
        pending += entity! { &session_owner @
            metadata::tag: *KIND_SESSION,
            tele::category: "session",
            tele::name: name_handle,
            tele::begin_ns: 0u64,
            bench::commit: commit_short.as_str(),
            bench::engine: label_short.as_str(),
            bench::config: config_handle,
        };

        Ok(Self {
            collection,
            session,
            pending,
        })
    }

    /// The session entity id (for logging).
    pub fn session(&self) -> Id {
        self.session
    }

    /// Record one measured iteration as a telemetry span.
    pub fn span(&mut self, name: &str, begin_ns: u64, duration_ns: u64) {
        let span_owner = genid();
        let name_handle = self.pending.put(name.to_string());
        self.pending += entity! { &span_owner @
            metadata::tag: *KIND_SPAN,
            tele::session: self.session,
            tele::category: "bench",
            tele::name: name_handle,
            tele::begin_ns: begin_ns,
            tele::end_ns: begin_ns + duration_ns,
            tele::duration_ns: duration_ns,
        };
    }

    /// Record a per-measure outcome entity.
    pub fn outcome(&mut self, workload: &str, outcome: &str, rows: Option<u64>) {
        let outcome_owner = genid();
        let workload_handle = self.pending.put(workload.to_string());
        let outcome_short = clip32(outcome);
        self.pending += entity! { &outcome_owner @
            bench::of_run: self.session,
            bench::workload: workload_handle,
            bench::outcome: outcome_short.as_str(),
        };
        if let Some(r) = rows {
            self.pending += entity! { &outcome_owner @ bench::rows: r };
        }
    }

    /// Close the session, publish one independent collection commit, and close
    /// the pile. The fragment carries every long-string attachment referenced
    /// by its facts and its full schema description in commit metadata.
    pub fn finish(mut self, end_ns: u64) -> Result<()> {
        let session_ref = ExclusiveId::force_ref(&self.session);
        self.pending += entity! { session_ref @
            tele::end_ns: end_ns,
            tele::duration_ns: end_ns,
        };
        self.pending.describe_with(ledger_metadata());
        self.collection
            .commit(std::mem::take(&mut self.pending))
            .map_err(|e| anyhow!("publish results collection commit: {e:?}"))?;
        self.collection
            .close()
            .map_err(|e| anyhow!("close results pile: {e:?}"))?;
        Ok(())
    }
}

fn load_collection_archive<R>(
    reader: &R,
    data: CollectionData,
) -> std::result::Result<Blob<SimpleArchive>, String>
where
    R: BlobStoreGet,
{
    let handle: Inline<Handle<SimpleArchive>> = data.transmute();
    reader
        .get(handle)
        .map_err(|e| format!("read collection element {data:?}: {e:?}"))
}

/// Read the descriptor a collection handle names, as ordinary facts.
///
/// A descriptor is a `TribleSet` and the handle is the hash of its archive, so
/// rehashing the bytes is the whole check: bytes that hash to the handle we
/// computed from our own recipe *are* our descriptor, and there is nothing
/// further to compare.
fn load_collection_descriptor<R>(
    reader: &R,
    collection: CollectionHandle,
) -> std::result::Result<TribleSet, String>
where
    R: BlobStoreGet,
{
    let blob: Blob<SimpleArchive> = reader
        .get(collection)
        .map_err(|e| format!("read collection descriptor {collection:?}: {e:?}"))?;
    let canonical = Blob::<SimpleArchive>::new(blob.bytes.clone());
    if canonical.get_handle() != collection {
        return Err(format!(
            "collection descriptor bytes do not match handle {collection:?}"
        ));
    }
    TribleSet::try_from_blob(canonical)
        .map_err(|e| format!("decode collection descriptor {collection:?}: {e:?}"))
}

/// The acceptance instrument: reopen a results pile read-only, discover and
/// resolve every valid signed commit in the deterministic results collection,
/// unite its set-valued elements, and print session + span + outcome counts.
///
/// The pile itself is the trust scope, matching the old local results-branch
/// reader: every strictly self-signed commit in this exact collection is
/// authorized. No commit is selected as a mutable or latest head.
pub fn verify(path: &Path) -> Result<()> {
    let mut pile =
        Pile::open(path).map_err(|e| anyhow!("open results pile {}: {e:?}", path.display()))?;
    pile.refresh()
        .map_err(|e| anyhow!("load results pile: {e:?}"))?;
    let discovered = discover_collection_records(&mut pile)
        .map_err(|e| anyhow!("discover results collection records: {e:?}"))?;
    if !discovered.diagnostics().is_empty() {
        bail!(
            "results pile contains invalid signed collection records: {:?}",
            discovered.diagnostics()
        );
    }

    let facade = results_collection();
    let descriptor = facade.descriptor();
    let collection = facade.collection();
    let reader = pile.reader().map_err(|e| anyhow!("pile reader: {e:?}"))?;
    // Presence is the match: the handle was computed from the canonical recipe
    // here, and the loader rehashes what the pile returned against it.
    load_collection_descriptor(&reader, collection).map_err(|e| {
        anyhow!(
            "results collection {} is not described in {}: {e}",
            RESULTS_COLLECTION_NAME.as_str(),
            path.display()
        )
    })?;

    let authorized: BTreeSet<Id> = discovered
        .commits()
        .iter()
        .filter(|commit| commit.collection() == collection)
        .map(|commit| commit.id())
        .collect();
    if authorized.is_empty() {
        bail!(
            "results collection {} has no signed commits",
            RESULTS_COLLECTION_NAME.as_str()
        );
    }

    // The results collection is a set of independent signed commits; nothing
    // derives from it and nothing is derived into it, so the lineage the
    // resolver consults is empty.
    let lineage = BTreeMap::new();
    let resolution = resolve_collection_semantics(&discovered, &lineage, &authorized, |request| {
        let verdict = match request {
            CollectionValidationRequest::Commit { claim } if claim.collection() == collection => {
                let blob = load_collection_archive(&reader, claim.data())?;
                match simplearchive_union::validate_commit(&descriptor, claim, &blob) {
                    Ok(()) => CollectionClaimValidation::Accepted,
                    Err(error) => CollectionClaimValidation::Rejected(error),
                }
            }
            CollectionValidationRequest::Merge { claim } if claim.collection() == collection => {
                let (low, high) = claim.inputs();
                let low = load_collection_archive(&reader, low)?;
                let high = load_collection_archive(&reader, high)?;
                let result = load_collection_archive(&reader, claim.result())?;
                match simplearchive_union::validate_merge(&descriptor, claim, &low, &high, &result)
                {
                    Ok(()) => CollectionClaimValidation::Accepted,
                    Err(error) => CollectionClaimValidation::Rejected(error),
                }
            }
            // This reader owns one concrete SimpleArchive-union collection.
            // Unrelated records and mappings need another collection's
            // authorization/recipe policy and therefore remain ineligible.
            CollectionValidationRequest::Commit { .. }
            | CollectionValidationRequest::Merge { .. }
            | CollectionValidationRequest::Derive { .. } => CollectionClaimValidation::Pending,
        };
        Ok::<_, String>(verdict)
    })
    .map_err(|e| anyhow!("resolve results collection: {e}"))?;

    let unresolved: Vec<Id> = authorized
        .difference(resolution.admitted_claims())
        .copied()
        .collect();
    if !unresolved.is_empty() {
        bail!(
            "results collection has unresolved commits {unresolved:?}; rejected={:?}",
            resolution.rejected()
        );
    }

    let mut facts = TribleSet::new();
    for data in resolution
        .semantics()
        .members(collection)
        .into_iter()
        .flatten()
    {
        let handle: Inline<Handle<SimpleArchive>> = data.transmute();
        let set: TribleSet = reader
            .get(handle)
            .map_err(|e| anyhow!("read admitted results element {data:?}: {e:?}"))?;
        facts += set;
    }
    let commits = authorized.len();

    let kind_session: Id = *KIND_SESSION;
    let kind_span: Id = *KIND_SPAN;

    let sessions: Vec<Id> = find!(
        (s: Id),
        pattern!(&facts, [{ ?s @ metadata::tag: kind_session }])
    )
    .map(|(s,)| s)
    .collect();

    let span_count = find!(
        (s: Id),
        pattern!(&facts, [{ ?s @ metadata::tag: kind_span }])
    )
    .count();

    println!(
        "verify   : {} — {} signed commits, {} tribles in the results collection",
        path.display(),
        commits,
        facts.len()
    );
    println!("sessions : {}", sessions.len());
    for (s, c, eng, cfg) in find!(
        (
            s: Id,
            c: Inline<ShortString>,
            eng: Inline<ShortString>,
            cfg: Inline<Handle<LongString>>
        ),
        pattern!(&facts, [{ ?s @ bench::commit: ?c, bench::engine: ?eng, bench::config: ?cfg }])
    ) {
        let commit: String = c
            .try_from_inline()
            .map_err(|e| anyhow!("commit decode: {e:?}"))?;
        let engine: String = eng
            .try_from_inline()
            .map_err(|e| anyhow!("engine decode: {e:?}"))?;
        let config: anybytes::View<str> =
            reader.get(cfg).map_err(|e| anyhow!("config blob: {e:?}"))?;
        println!("  {s:X}  commit={commit} engine={engine}");
        println!("    config: {}", config.as_ref());
    }

    println!("spans    : {span_count}");
    // Project the span id too: find! heads have SET semantics, so a
    // name-only head would collapse the per-iteration spans to one row
    // per distinct name.
    //
    // The runner deliberately stores raw per-iteration observations and
    // never aggregates; reading a duration out of them is the viewer's
    // job, and this is the suite's own minimal viewer. `min` leads the
    // summary on purpose: on a contended machine the fastest observed
    // iteration is the least contaminated estimate of the work itself,
    // while the spread against `max` shows how much interference the
    // run actually absorbed.
    //
    // Keyed by (session, name), never by name alone: a results pile
    // accumulates every run ever made against it, so collapsing on the
    // name would silently average one rung's arm into another's — the
    // exact confusion a comparative arm exists to avoid.
    let mut span_times: std::collections::BTreeMap<(Id, String), Vec<u64>> = Default::default();
    for (_s, run, n, d) in find!(
        (s: Id, run: Id, n: Inline<Handle<LongString>>, d: u64),
        pattern!(&facts, [{ ?s @
            metadata::tag: kind_span,
            tele::session: ?run,
            tele::name: ?n,
            tele::duration_ns: ?d
        }])
    ) {
        let name: anybytes::View<str> = reader
            .get(n)
            .map_err(|e| anyhow!("span name blob: {e:?}"))?;
        span_times
            .entry((run, name.as_ref().to_owned()))
            .or_default()
            .push(d);
    }
    let mut current: Option<Id> = None;
    for ((run, name), times) in &mut span_times {
        if current != Some(*run) {
            current = Some(*run);
            println!("  session {run:X}");
            println!(
                "  {:<45}{:>4}{:>12}{:>12}{:>12}",
                "span", "n", "min ms", "median ms", "max ms"
            );
        }
        times.sort_unstable();
        let ms = |ns: u64| ns as f64 / 1e6;
        println!(
            "  {name:<45}{:>4}{:>12.3}{:>12.3}{:>12.3}",
            times.len(),
            ms(times[0]),
            ms(times[times.len() / 2]),
            ms(times[times.len() - 1]),
        );
    }

    // Optional rows per outcome entity (the engine is monotone; the
    // optional join happens here in Rust).
    let mut rows_of: std::collections::HashMap<Id, u64> = Default::default();
    for (o, r) in find!(
        (o: Id, r: u64),
        pattern!(&facts, [{ ?o @ bench::rows: ?r }])
    ) {
        rows_of.insert(o, r);
    }

    let mut outcome_rows: Vec<(String, String, Option<u64>)> = Vec::new();
    for (o, w, v) in find!(
        (o: Id, w: Inline<Handle<LongString>>, v: Inline<ShortString>),
        pattern!(&facts, [{ ?o @ bench::workload: ?w, bench::outcome: ?v }])
    ) {
        let workload: anybytes::View<str> =
            reader.get(w).map_err(|e| anyhow!("workload blob: {e:?}"))?;
        let outcome: String = v
            .try_from_inline()
            .map_err(|e| anyhow!("outcome decode: {e:?}"))?;
        outcome_rows.push((
            workload.as_ref().to_owned(),
            outcome,
            rows_of.get(&o).copied(),
        ));
    }
    println!("outcomes : {}", outcome_rows.len());
    let mut histogram: std::collections::BTreeMap<(String, String), usize> = Default::default();
    for (workload, outcome, _) in &outcome_rows {
        let group = workload.split('/').next().unwrap_or(workload).to_owned();
        *histogram.entry((group, outcome.clone())).or_default() += 1;
    }
    for ((group, outcome), count) in &histogram {
        println!("  {group:<14} {outcome:<28} x{count}");
    }
    outcome_rows.sort();
    let mut any_rows = false;
    for (workload, outcome, rows) in &outcome_rows {
        if let Some(n) = rows {
            if !any_rows {
                println!("rows     :");
                any_rows = true;
            }
            println!("  {workload:<45} {outcome:<10} rows={n}");
        }
    }

    pile.close()
        .map_err(|e| anyhow!("close results pile: {e:?}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_collection_roundtrip_accumulates_runs_with_metadata_and_attachments() {
        let path = std::env::temp_dir().join(format!("tribleset-bench-ledger-{:X}.pile", *genid()));

        let mut ledger = ResultsLedger::open(
            &path,
            "0123456789abcdef",
            "native-collection-test",
            "argv: test | suite: tribleset-bench test",
        )
        .unwrap();
        ledger.span("fixture/total", 11, 13);
        ledger.outcome("fixture/total", "signal", Some(7));
        ledger.finish(29).unwrap();

        let mut second = ResultsLedger::open(
            &path,
            "fedcba9876543210",
            "native-collection-test-2",
            "argv: test-2 | suite: tribleset-bench test",
        )
        .unwrap();
        second.outcome("fixture/other", "skip:test", None);
        second.finish(3).unwrap();

        verify(&path).unwrap();

        let mut pile = Pile::open(&path).unwrap();
        let discovered = discover_collection_records(&mut pile).unwrap();
        let facade = results_collection();
        let collection = facade.collection();
        let reader = pile.reader().unwrap();
        assert_eq!(
            load_collection_descriptor(&reader, collection).unwrap(),
            facade.descriptor().into_facts()
        );
        let commits: Vec<_> = discovered
            .commits()
            .iter()
            .filter(|commit| commit.collection() == collection)
            .collect();
        assert_eq!(commits.len(), 2);

        let mut facts = TribleSet::new();
        for commit in commits {
            let metadata: TribleSet = reader.get(commit.metadata()).unwrap();
            assert!(!metadata.is_empty());
            let data_handle: Inline<Handle<SimpleArchive>> = commit.data().transmute();
            let run: TribleSet = reader.get(data_handle).unwrap();
            facts += run;
        }
        assert_eq!(
            find!(
                session: Id,
                pattern!(&facts, [{ ?session @ metadata::tag: *KIND_SESSION }])
            )
            .count(),
            2
        );
        let configs: BTreeSet<String> = find!(
            cfg: Inline<Handle<LongString>>,
            pattern!(&facts, [{ bench::config: ?cfg }])
        )
        .map(|config| {
            let config: anybytes::View<str> = reader.get(config).unwrap();
            config.as_ref().to_owned()
        })
        .collect();
        assert_eq!(configs.len(), 2);
        assert!(configs
            .iter()
            .all(|config| config.contains("suite: tribleset-bench test")));

        drop(reader);
        pile.close().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
