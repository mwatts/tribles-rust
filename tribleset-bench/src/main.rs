//! tribleset-bench — engine-version-agnostic benchmark suite runner.
//!
//! The SUBJECT engine (renamed dep `subject`, repointed per rev by
//! bench.sh) runs the workload; results land as canonical telemetry
//! sessions/spans plus per-measure outcome entities, written through
//! the stable LEDGER dep (`triblespace` 0.47) on the minted results
//! branch. Explicit stopwatch (`std::time::Instant`), one span per
//! measured iteration (warmups unmeasured), NO aggregation in the
//! runner — raw observations only; statistics are the viewer's job.
//!
//! Groups:
//! - `ladder/checkout/total` — `Workspace::checkout` of the first k
//!   commits of the `--data` pile's branch at the `--rung` target.
//! - `arch/build_ram/total` — `SuccinctArchive<OrderedUniverse>` build
//!   over the checked-out set.
//! - `harkonnen/F{1..5}/{ttfr,total}` — the R1 adversarial fixtures; F3
//!   (oasis) and F5 (diamond) run everywhere, F1/F2/F4 are rpq-gated.
//! - `harkonnen/F{6..15}/…` — the R2 white-box fixtures, one engine
//!   decision each. All run everywhere except F10, which is gpu-gated
//!   because it reads the routing threshold out of `triblespace-gpu`.
//! - `sparqloscope/<query>/total` — the vendored TRANSLATED registry;
//!   without a wd Dataset every query records SKIP "dataset absent"
//!   (the census still lands in the pile).
//!
//! Panics in any measure are caught (`quiet_catch`) and recorded as
//! `panic:<reason>` outcomes; the run continues.
//!
//! !!! Always point `--data` at a clonefile copy (`cp -c`) of a
//! dataset pile — the checkout phase's `Repository::new` appends a
//! commit-metadata record to the pile file on open.

use std::time::Instant;

use subject::core::prelude::TribleSet;

mod fixtures;
mod ledger;

#[path = "../queries/wd_schema.rs"]
mod wd_schema;

#[path = "../queries/sparqloscope.rs"]
mod queries;

struct Cfg {
    data: Option<std::path::PathBuf>,
    branch: Option<String>,
    rung: usize,
    results: Option<std::path::PathBuf>,
    label: Option<String>,
    iters: usize,
    warmup: usize,
    build_iters: usize,
    build_warmup: usize,
    verify: Option<std::path::PathBuf>,
}

fn parse_size(s: &str) -> Option<usize> {
    let (num, mul) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1_000),
        'M' => (&s[..s.len() - 1], 1_000_000),
        'G' => (&s[..s.len() - 1], 1_000_000_000),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|n| n * mul)
}

fn usage() -> ! {
    eprintln!(
        "usage: tribleset-bench --results <pile> --label <engine label> \
         [--data <pile> --branch <name> --rung <N>] \
         [--iters N] [--warmup N] [--build-iters N] [--build-warmup N]\n\
         \x20      tribleset-bench --verify <pile>\n\
         Sizes accept k/M/G suffixes. --data must be a clonefile copy \
         (cp -c) of a dataset pile."
    );
    std::process::exit(2);
}

fn parse_cfg() -> Cfg {
    let mut cfg = Cfg {
        data: None,
        branch: None,
        rung: 1_000_000,
        results: None,
        label: None,
        iters: 12,
        warmup: 3,
        build_iters: 8,
        build_warmup: 2,
        verify: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    fn take<'a>(args: &'a [String], i: &mut usize) -> &'a str {
        *i += 1;
        args.get(*i)
            .unwrap_or_else(|| {
                eprintln!("{} needs an argument", args[*i - 1]);
                std::process::exit(2);
            })
            .as_str()
    }
    fn take_size(args: &[String], i: &mut usize) -> usize {
        let raw = take(args, i);
        parse_size(raw).unwrap_or_else(|| {
            eprintln!("{} needs a size argument, got {raw:?}", args[*i - 1]);
            std::process::exit(2);
        })
    }
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => cfg.data = Some(take(&args, &mut i).into()),
            "--branch" => cfg.branch = Some(take(&args, &mut i).to_owned()),
            "--rung" => cfg.rung = take_size(&args, &mut i),
            "--results" => cfg.results = Some(take(&args, &mut i).into()),
            "--label" => cfg.label = Some(take(&args, &mut i).to_owned()),
            "--iters" => cfg.iters = take_size(&args, &mut i),
            "--warmup" => cfg.warmup = take_size(&args, &mut i),
            "--build-iters" => cfg.build_iters = take_size(&args, &mut i),
            "--build-warmup" => cfg.build_warmup = take_size(&args, &mut i),
            "--verify" => cfg.verify = Some(take(&args, &mut i).into()),
            other => {
                eprintln!("unrecognized arg {other:?}");
                usage();
            }
        }
        i += 1;
    }
    cfg
}

/// The subject's git rev (short=12), read at runtime from the checkout
/// the `subject` dependency points at (the bench.sh-managed
/// `subjects/current` symlink).
fn subject_commit() -> String {
    let subject_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/subjects/current");
    match std::process::Command::new("git")
        .args(["-C", subject_dir, "rev-parse", "--short=12", "HEAD"])
        .output()
    {
        Ok(out) if out.status.success() => {
            String::from_utf8_lossy(&out.stdout).trim().to_owned()
        }
        _ => {
            eprintln!("note     : could not read the subject git rev from {subject_dir}");
            "unknown".to_owned()
        }
    }
}

/// One measure being sampled across iterations: raw spans plus the
/// panic/identity state that decides its outcome.
struct Measure {
    name: &'static str,
    spans: Vec<(u64, u64)>,
    panicked: Option<String>,
    ident: Option<usize>,
    gate: Option<String>,
}

impl Measure {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            spans: Vec::new(),
            panicked: None,
            ident: None,
            gate: None,
        }
    }

    /// One guarded timed call. Skips slots that already panicked (the
    /// panic is deterministic), records the span when warmed up, and
    /// checks cross-iteration workload identity on the row count.
    fn iterate(&mut self, recording: bool, base: &Instant, f: impl FnOnce() -> usize) {
        if self.panicked.is_some() {
            return;
        }
        let begin_ns = base.elapsed().as_nanos() as u64;
        let t = Instant::now();
        match fixtures::quiet_catch(f) {
            Ok(rows) => {
                if recording {
                    self.spans.push((begin_ns, t.elapsed().as_nanos() as u64));
                }
                match self.ident {
                    None => self.ident = Some(rows),
                    Some(expected) if expected != rows => {
                        self.gate.get_or_insert(format!(
                            "identity: saw {rows} rows, expected {expected}"
                        ));
                    }
                    _ => {}
                }
            }
            Err(msg) => self.panicked = Some(msg),
        }
    }

    /// Gate the identity count against a fixed expectation.
    fn expect_rows(&mut self, expected: usize) {
        if self.panicked.is_some() || self.gate.is_some() {
            return;
        }
        if let Some(n) = self.ident {
            if n != expected {
                self.gate = Some(format!("rows: saw {n}, expected {expected}"));
            }
        }
    }

    /// Write spans + the outcome entity; print one console line.
    fn emit(self, led: &mut ledger::ResultsLedger, rows_meaningful: bool) {
        for (begin_ns, duration_ns) in &self.spans {
            led.span(self.name, *begin_ns, *duration_ns);
        }
        let (outcome, rows) = match (&self.panicked, &self.gate) {
            (Some(msg), _) => (format!("panic:{msg}"), None),
            (None, Some(gate)) => (format!("gate_fail:{gate}"), None),
            (None, None) => (
                "signal".to_owned(),
                if rows_meaningful {
                    self.ident.map(|n| n as u64)
                } else {
                    None
                },
            ),
        };
        led.outcome(self.name, &outcome, rows);
        match rows {
            Some(n) => println!("  {:<32} {outcome} ({} spans, {n} rows)", self.name, self.spans.len()),
            None => println!("  {:<32} {outcome} ({} spans)", self.name, self.spans.len()),
        }
    }
}

/// One measure of an R2 fixture: what to call on the built set, whether
/// its row count is a cardinality (vs. a TTFR sentinel), and the exact
/// count its construction predicts.
struct R2Measure {
    name: &'static str,
    /// Whether `rows` is meaningful telemetry (false for TTFR probes,
    /// which only ever report 0 or 1).
    rows_meaningful: bool,
    /// The gate. `None` only where a fixture's construction genuinely
    /// does not pin a count.
    expect: Option<usize>,
    run: fn(&TribleSet) -> usize,
}

/// Build one R2 fixture (panic-guarded, once) and iterate every measure
/// over it, gating each on its expected row count. A panic in the
/// builder is recorded against every measure of the fixture, matching
/// how the F3/F5 pair is handled.
fn run_r2(
    led: &mut ledger::ResultsLedger,
    warmup: usize,
    iters: usize,
    base: &Instant,
    build: impl FnOnce() -> TribleSet,
    measures: &[R2Measure],
) {
    let built = match fixtures::quiet_catch(build) {
        Err(msg) => {
            for m in measures {
                led.outcome(m.name, &format!("panic:{msg}"), None);
                println!("  {:<32} panic ({msg})", m.name);
            }
            return;
        }
        Ok(set) => set,
    };
    let mut running: Vec<Measure> = measures.iter().map(|m| Measure::new(m.name)).collect();
    for i in 0..(warmup + iters) {
        let recording = i >= warmup;
        for (state, spec) in running.iter_mut().zip(measures.iter()) {
            state.iterate(recording, base, || (spec.run)(&built));
        }
    }
    for (mut state, spec) in running.into_iter().zip(measures.iter()) {
        if let Some(expected) = spec.expect {
            state.expect_rows(expected);
        }
        state.emit(led, spec.rows_meaningful);
    }
}

fn main() {
    let cfg = parse_cfg();

    if let Some(path) = &cfg.verify {
        if let Err(e) = ledger::verify(path) {
            eprintln!("verify failed: {e:?}");
            std::process::exit(1);
        }
        return;
    }

    let (Some(results), Some(label)) = (&cfg.results, &cfg.label) else {
        eprintln!("--results and --label are required for a bench run");
        usage();
    };

    let commit = subject_commit();
    let config = format!(
        "argv: {} | data: {} branch: {} rung: {} | iters: {} warmup: {} build_iters: {} build_warmup: {} | suite: tribleset-bench {}",
        std::env::args().skip(1).collect::<Vec<_>>().join(" "),
        cfg.data
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into()),
        cfg.branch.as_deref().unwrap_or("auto"),
        cfg.rung,
        cfg.iters,
        cfg.warmup,
        cfg.build_iters,
        cfg.build_warmup,
        env!("CARGO_PKG_VERSION"),
    );

    println!("subject  : {commit} ({label})");
    println!("config   : {config}");

    let suite_start = Instant::now();
    let base = Instant::now();
    let mut led = match ledger::ResultsLedger::open(results, &commit, label, &config) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cannot open results ledger: {e:?}");
            std::process::exit(1);
        }
    };
    println!("session  : {:X}", led.session());

    // -- ladder + arch -----------------------------------------------------
    let dataset = match &cfg.data {
        None => {
            println!("  {:<32} SKIP (no --data)", "ladder/checkout/total");
            led.outcome("ladder/checkout/total", "skip:no-data", None);
            println!("  {:<32} SKIP (no --data)", "arch/build_ram/total");
            led.outcome("arch/build_ram/total", "skip:no-data", None);
            None
        }
        Some(path) => {
            match fixtures::quiet_catch(|| {
                fixtures::pile_checkout(
                    path,
                    cfg.branch.as_deref(),
                    cfg.rung,
                    cfg.build_iters,
                    cfg.build_warmup,
                    &base,
                )
            }) {
                Ok(Ok((set, spans, tribles))) => {
                    for (begin_ns, duration_ns) in &spans {
                        led.span("ladder/checkout/total", *begin_ns, *duration_ns);
                    }
                    led.outcome("ladder/checkout/total", "signal", Some(tribles as u64));
                    println!(
                        "  {:<32} signal ({} spans, {tribles} tribles)",
                        "ladder/checkout/total",
                        spans.len()
                    );
                    Some(set)
                }
                Ok(Err(gate)) => {
                    led.outcome("ladder/checkout/total", &format!("gate_fail:{gate}"), None);
                    println!("  {:<32} gate_fail ({gate})", "ladder/checkout/total");
                    None
                }
                Err(msg) => {
                    led.outcome("ladder/checkout/total", &format!("panic:{msg}"), None);
                    println!("  {:<32} panic ({msg})", "ladder/checkout/total");
                    None
                }
            }
        }
    };
    if let Some(set) = &dataset {
        /// Above this trible count the RAM archive build is skipped
        /// (the portable_bench --max-ram default).
        const MAX_RAM: usize = 20_000_000;
        if set.len() > MAX_RAM {
            led.outcome("arch/build_ram/total", "skip:max-ram", None);
            println!(
                "  {:<32} SKIP ({} tribles > max-ram {MAX_RAM})",
                "arch/build_ram/total",
                set.len()
            );
        } else {
            let mut m = Measure::new("arch/build_ram/total");
            for i in 0..(cfg.build_warmup + cfg.build_iters) {
                let recording = i >= cfg.build_warmup;
                m.iterate(recording, &base, || {
                    let arch = fixtures::build_archive(set);
                    drop(arch);
                    set.len()
                });
            }
            m.emit(&mut led, true);
        }
    }

    // -- harkonnen ---------------------------------------------------------
    #[cfg(not(feature = "rpq"))]
    for name in [
        "harkonnen/F1/ttfr",
        "harkonnen/F1/total",
        "harkonnen/F2/ttfr",
        "harkonnen/F2/total",
        "harkonnen/F4/ttfr",
        "harkonnen/F4/total",
    ] {
        led.outcome(name, "skip:rpq", None);
        println!("  {name:<32} SKIP (rpq: no regular-path constraint)");
    }

    match fixtures::quiet_catch(|| {
        (
            fixtures::build_oasis(fixtures::OASIS_K, fixtures::OASIS_FAN, fixtures::OASIS_DEATHS),
            fixtures::build_diamond(fixtures::DIAMOND_N),
        )
    }) {
        Err(msg) => {
            for name in [
                "harkonnen/F3/ttfr",
                "harkonnen/F3/total",
                "harkonnen/F5/ttfr",
                "harkonnen/F5/total",
            ] {
                led.outcome(name, &format!("panic:{msg}"), None);
                println!("  {name:<32} panic ({msg})");
            }
        }
        Ok(((oasis, _oasis_start), diamond)) => {
            let mut f3_ttfr = Measure::new("harkonnen/F3/ttfr");
            let mut f3_total = Measure::new("harkonnen/F3/total");
            let mut f5_ttfr = Measure::new("harkonnen/F5/ttfr");
            let mut f5_total = Measure::new("harkonnen/F5/total");
            for i in 0..(cfg.warmup + cfg.iters) {
                let recording = i >= cfg.warmup;
                f3_ttfr.iterate(recording, &base, || fixtures::f3_ttfr(&oasis));
                f3_total.iterate(recording, &base, || fixtures::f3_total(&oasis));
                f5_ttfr.iterate(recording, &base, || fixtures::f5_ttfr(&diamond));
                f5_total.iterate(recording, &base, || fixtures::f5_total(&diamond));
            }
            f3_total.expect_rows(fixtures::F3_EXPECTED_ROWS);
            f5_total.expect_rows(fixtures::F5_EXPECTED_ROWS);
            f3_ttfr.emit(&mut led, false);
            f3_total.emit(&mut led, true);
            f5_ttfr.emit(&mut led, false);
            f5_total.emit(&mut led, true);
        }
    }

    // -- harkonnen R2 (F6..F15) --------------------------------------------
    // One fixture at a time: each builder runs once, its measures share
    // the built set, and every measure carries the exact row count its
    // construction derives (see the fixture docs for each derivation).
    run_r2(
        &mut led,
        cfg.warmup,
        cfg.iters,
        &base,
        fixtures::build_union_fan,
        &[R2Measure {
            name: "harkonnen/F6/total",
            rows_meaningful: true,
            expect: Some(fixtures::F6_EXPECTED_ROWS),
            run: fixtures::f6_total,
        }],
    );
    run_r2(
        &mut led,
        cfg.warmup,
        cfg.iters,
        &base,
        fixtures::build_hub_skew,
        &[R2Measure {
            name: "harkonnen/F7/total",
            rows_meaningful: true,
            expect: Some(fixtures::F7_EXPECTED_ROWS),
            run: fixtures::f7_total,
        }],
    );
    run_r2(
        &mut led,
        cfg.warmup,
        cfg.iters,
        &base,
        fixtures::build_witness_multiplicity,
        &[
            R2Measure {
                name: "harkonnen/F8/bag",
                rows_meaningful: true,
                expect: Some(fixtures::F8_EXPECTED_BAG_ROWS),
                run: fixtures::f8_bag,
            },
            R2Measure {
                name: "harkonnen/F8/distinct",
                rows_meaningful: true,
                expect: Some(fixtures::F8_EXPECTED_DISTINCT_ROWS),
                run: fixtures::f8_distinct,
            },
        ],
    );
    run_r2(
        &mut led,
        cfg.warmup,
        cfg.iters,
        &base,
        fixtures::build_mask_sparse,
        &[R2Measure {
            name: "harkonnen/F9/sparse",
            rows_meaningful: true,
            expect: Some(fixtures::F9_SPARSE_EXPECTED_ROWS),
            run: fixtures::f9_total,
        }],
    );
    run_r2(
        &mut led,
        cfg.warmup,
        cfg.iters,
        &base,
        fixtures::build_mask_dense,
        &[R2Measure {
            name: "harkonnen/F9/dense",
            rows_meaningful: true,
            expect: Some(fixtures::F9_DENSE_EXPECTED_ROWS),
            run: fixtures::f9_total,
        }],
    );

    #[cfg(not(feature = "gpu"))]
    for name in ["harkonnen/F10/below", "harkonnen/F10/above"] {
        led.outcome(name, "skip:gpu", None);
        println!("  {name:<32} SKIP (gpu: no triblespace-gpu on the subject)");
    }
    #[cfg(feature = "gpu")]
    {
        run_r2(
            &mut led,
            cfg.warmup,
            cfg.iters,
            &base,
            || fixtures::build_gpu_boundary(false, fixtures::F10_BELOW),
            &[R2Measure {
                name: "harkonnen/F10/below",
                rows_meaningful: true,
                expect: Some(fixtures::F10_BELOW),
                run: |set| fixtures::f10_total(false, set),
            }],
        );
        run_r2(
            &mut led,
            cfg.warmup,
            cfg.iters,
            &base,
            || fixtures::build_gpu_boundary(true, fixtures::F10_ABOVE),
            &[R2Measure {
                name: "harkonnen/F10/above",
                rows_meaningful: true,
                expect: Some(fixtures::F10_ABOVE),
                run: |set| fixtures::f10_total(true, set),
            }],
        );
    }

    // -- sparqloscope ------------------------------------------------------
    // No wd Dataset loader is vendored (the pile manifest schema and
    // loaders stay in sparqloscope-bench, and no wd dataset exists on
    // this machine), so the whole registry records SKIP — the census
    // itself is the deliverable and must land in the pile.
    let (mut engine_kind, mut fold_kind, mut periphery_kind) = (0usize, 0usize, 0usize);
    for t in queries::TRANSLATED {
        match t.kind {
            queries::Kind::Engine => engine_kind += 1,
            queries::Kind::Fold => fold_kind += 1,
            queries::Kind::Periphery => periphery_kind += 1,
        }
        led.outcome(
            &format!("sparqloscope/{}/total", t.name),
            "skip:dataset-absent",
            None,
        );
    }
    for name in queries::SKIPPED_PATHS {
        led.outcome(&format!("sparqloscope/{name}/total"), "skip:rpq", None);
    }
    println!(
        "  sparqloscope census              {} dataset-absent ({engine_kind} engine / {fold_kind} fold / {periphery_kind} periphery) + {} rpq",
        queries::TRANSLATED.len(),
        queries::SKIPPED_PATHS.len()
    );

    // -- close -------------------------------------------------------------
    let end_ns = base.elapsed().as_nanos() as u64;
    if let Err(e) = led.finish(end_ns) {
        eprintln!("cannot finish results session: {e:?}");
        std::process::exit(1);
    }
    println!(
        "done     : suite ran {:.2}s, results in {}",
        suite_start.elapsed().as_secs_f64(),
        results.display()
    );
}
