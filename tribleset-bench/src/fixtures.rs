//! Subject-side measurement material, vendored from
//! `june-on-tip/benches/portable_bench.rs` (working tree of
//! 2026-07-27) and adapted to the `subject` dependency (the engine
//! under test). Everything in this module runs against the SUBJECT:
//! imports go through `subject::core::…` / the core prelude, whose
//! macros expand to `$crate` or `::triblespace_core` paths — both
//! resolve to the subject's core, never to the results ledger.
//!
//! Vendored pieces:
//! - `quiet_catch` — panics are an outcome, not a crash.
//! - the Harkonnen R1 adversarial fixtures (`Ids`, `r1_schema`,
//!   `build_chain`/`build_oasis`/`build_khop`/`build_diamond`; minted
//!   attribute ids from 2026-07-19 preserved byte-for-byte). The
//!   path-shaped fixtures F1/F2/F4 (chain, ring, k-hop) are
//!   `rpq`-gated like the transitive-path queries: without a
//!   regular-path constraint on the subject they cannot be queried,
//!   so their builders compile only under `--features rpq` and the
//!   runner records SKIP outcomes for their measures.
//! - the F3 (oasis-last) and F5 (two-route diamond) fixture queries.
//! - `commit_chain` + `pile_checkout` — the ladder phase: checkout of
//!   the first k commits of a dataset pile's data branch, where k is
//!   derived from a cumulative-trible rung target.

use std::time::Instant;

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

use subject::core::blob::encodings::longstring::LongString;
use subject::core::blob::encodings::simplearchive::SimpleArchive;
use subject::core::blob::encodings::succinctarchive::{OrderedUniverse, SuccinctArchive};
use subject::core::inline::encodings::hash::Handle;
use subject::core::metadata;
use subject::core::prelude::inlineencodings::GenId;
use subject::core::prelude::*;
use subject::core::repo::pile::Pile;
use subject::core::repo::{self, Repository};

// ---------------------------------------------------------------------------
// Crash isolation
// ---------------------------------------------------------------------------

/// `catch_unwind` with the default panic hook silenced around the call
/// (expected panics at hostile subject revs must not spam stderr — hook
/// saved and restored) and the payload reduced to the first line of its
/// message.
pub fn quiet_catch<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    out.map_err(|payload| {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else {
            "non-string panic payload"
        };
        msg.lines().next().unwrap_or("").to_owned()
    })
}

// ---------------------------------------------------------------------------
// Harkonnen R1 adversarial fixtures (vendored; ids minted 2026-07-19,
// never invented).
// ---------------------------------------------------------------------------

mod r1_schema {
    // mp/msrc/khop keep their minted ids although the path!-based
    // fixtures that consume them (F1/F2/F4) are rpq-gated.
    #![allow(dead_code)]
    use subject::core::prelude::*;

    attributes! {
        // metronome / ring edge
        "277A42231FD9D42DD50D789D8F9E8661" as mp: inlineencodings::GenId;
        // multi-source marker (K>1 eager-cohort control)
        "0F64BC179033DB2703C65E7DBBAA9AD3" as msrc: inlineencodings::GenId;
        // oasis: type marker, p edge, q edge
        "A0C25A0F02E2D5232269F274761B2AB1" as otype: inlineencodings::GenId;
        "831EA731FB6C91252CDDC4FC399DC975" as op: inlineencodings::GenId;
        "2B3A5EF282FED1F652A2C182E116C28C" as oq: inlineencodings::GenId;
        // thin k-hop functional chain edge
        "EE09E63B176F818960267C5041CA6C92" as khop: inlineencodings::GenId;
        // diamond (reconvergence-capture) route attributes
        "E73DC5D12C49394D3C6D883A152E57C9" as da: inlineencodings::GenId;
        "C41A8C9EC883E09D34C86F87C15EA965" as db: inlineencodings::GenId;
    }
}

/// Default fixture sizes (the portable_bench defaults; F5's expected
/// row count derives from `DIAMOND_N`).
pub const OASIS_K: usize = 4_000;
pub const OASIS_FAN: usize = 32;
pub const OASIS_DEATHS: usize = 20;
pub const DIAMOND_N: usize = 256;

/// F3's expected total row count: exactly one complete op->oq path
/// (the oasis).
pub const F3_EXPECTED_ROWS: usize = 1;

/// F5's expected total row count: per entity pair (one per route),
/// route 0 contributes 1 da-edge x 4 db-edges = 4 rows and route 1
/// contributes 1 x 2 = 2 rows (the three same-target da inserts
/// dedupe), so 6 rows per `n_per_route` step.
pub const F5_EXPECTED_ROWS: usize = 6 * DIAMOND_N;

/// Deterministic UFOID-shaped ids (shared locality prefix, splitmix
/// suffix) so succinct-backend value order — and therefore exploration
/// order — is reproducible across runs and machines.
struct Ids {
    next: u64,
}

impl Ids {
    fn new() -> Self {
        Self { next: 1 }
    }

    fn splitmix64(mut v: u64) -> u64 {
        v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
        v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        v ^ (v >> 31)
    }

    fn mint(&mut self) -> ExclusiveId {
        let c = self.next;
        self.next += 1;
        let mut raw = [0u8; 16];
        raw[..4].copy_from_slice(&0xD46B_0001u32.to_be_bytes());
        raw[4..12].copy_from_slice(&Self::splitmix64(c).to_be_bytes());
        raw[12..].copy_from_slice(&Self::splitmix64(c ^ 0xD1B5_4A32).to_be_bytes()[..4]);
        ExclusiveId::force(Id::new(raw).expect("nonzero prefix"))
    }

    /// Mint with a chosen leading suffix byte so a fixture can pin
    /// where a value lands in sorted-universe order.
    fn mint_ordered(&mut self, order: u8) -> ExclusiveId {
        let c = self.next;
        self.next += 1;
        let mut raw = [0u8; 16];
        raw[..4].copy_from_slice(&0xD46B_0001u32.to_be_bytes());
        raw[4] = order;
        raw[5..12].copy_from_slice(&Self::splitmix64(c).to_be_bytes()[..7]);
        raw[12..].copy_from_slice(&Self::splitmix64(c ^ 0x5EED_5EED).to_be_bytes()[..4]);
        ExclusiveId::force(Id::new(raw).expect("nonzero prefix"))
    }
}

/// F1/F2 — metronome chain and ring. rpq-gated: the fixture queries
/// are transitive closures, inexpressible without a regular-path
/// constraint on the subject.
#[cfg(feature = "rpq")]
pub fn build_chain(n: usize, ring: bool, sources: usize) -> (TribleSet, Id) {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let nodes: Vec<ExclusiveId> = (0..n).map(|_| ids.mint()).collect();
    for w in nodes.windows(2) {
        set += entity! { &w[0] @ r1_schema::mp: &w[1] };
    }
    if ring {
        set += entity! { &nodes[n - 1] @ r1_schema::mp: &nodes[0] };
    }
    for s in nodes.iter().take(sources.max(1)) {
        set += entity! { s @ r1_schema::msrc: s };
    }
    let start: Id = *nodes[0];
    (set, start)
}

/// F3 — oasis-last: `k` typed entities; the single oasis (order byte
/// 0x00, explored LAST) owns the only complete op->oq path; the first
/// `deaths` entities in exploration order have no `op` edge (cheap
/// deaths); every other entity fans `fan` junk op-edges (expensive
/// depth-2 refutations).
pub fn build_oasis(k: usize, fan: usize, deaths: usize) -> (TribleSet, Id) {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let oasis = ids.mint_ordered(0x00);
    let y_star = ids.mint_ordered(0x01);
    let z = ids.mint_ordered(0x02);
    set += entity! { &oasis @ r1_schema::otype: &oasis };
    set += entity! { &oasis @ r1_schema::op: &y_star };
    set += entity! { &y_star @ r1_schema::oq: &z };
    for i in 0..k {
        let order = 0xFF - ((i % 0x80) as u8);
        let e = ids.mint_ordered(order.max(0x03));
        set += entity! { &e @ r1_schema::otype: &e };
        if i >= deaths {
            for _ in 0..fan {
                let junk = ids.mint_ordered(0x7F);
                set += entity! { &e @ r1_schema::op: &junk };
            }
        }
    }
    let start: Id = *oasis;
    (set, start)
}

/// F4 — thin functional k-hop chain from a constant. rpq-gated like
/// F1/F2.
#[cfg(feature = "rpq")]
pub fn build_khop(k: usize) -> (TribleSet, Id) {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let nodes: Vec<ExclusiveId> = (0..=k).map(|_| ids.mint()).collect();
    for w in nodes.windows(2) {
        set += entity! { &w[0] @ r1_schema::khop: &w[1] };
    }
    let start: Id = *nodes[0];
    (set, start)
}

/// F5 — two-route diamond for reconvergence capture: two populations
/// prefer opposite orders of (da, db) then share identical
/// continuations; the eager solver merges them maximally, a width-1
/// sprint historically reenters.
pub fn build_diamond(n_per_route: usize) -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    for route in 0..2usize {
        for _ in 0..n_per_route {
            let e = ids.mint();
            let x = ids.mint();
            let y = ids.mint();
            let (fat, thin) = if route == 0 { (3usize, 1usize) } else { (1, 3) };
            for _ in 0..thin {
                set += entity! { &e @ r1_schema::da: &x };
            }
            for _ in 0..fat {
                let alt = ids.mint();
                set += entity! { &e @ r1_schema::db: &alt };
            }
            set += entity! { &e @ r1_schema::db: &y };
        }
    }
    set
}

// ---------------------------------------------------------------------------
// F3/F5 fixture queries (the pattern!-join measures; TTFR = first-row
// latency via `.next()`, total = full drain via `.count()`).
// ---------------------------------------------------------------------------

/// F3 arm-to-first-row: rows drained (0 or 1).
pub fn f3_ttfr(oasis: &TribleSet) -> usize {
    find!(
        (e: Inline<GenId>, y: Inline<GenId>, z: Inline<GenId>),
        and!(
            pattern!(oasis, [{ ?e @ r1_schema::otype: ?e }]),
            pattern!(oasis, [{ ?e @ r1_schema::op: ?y }]),
            pattern!(oasis, [{ ?y @ r1_schema::oq: ?z }]),
        )
    )
    .next()
    .map_or(0, |_| 1)
}

/// F3 full drain: total row count (expected [`F3_EXPECTED_ROWS`]).
pub fn f3_total(oasis: &TribleSet) -> usize {
    find!(
        (e: Inline<GenId>, y: Inline<GenId>, z: Inline<GenId>),
        and!(
            pattern!(oasis, [{ ?e @ r1_schema::otype: ?e }]),
            pattern!(oasis, [{ ?e @ r1_schema::op: ?y }]),
            pattern!(oasis, [{ ?y @ r1_schema::oq: ?z }]),
        )
    )
    .count()
}

/// F5 arm-to-first-row: rows drained (0 or 1).
pub fn f5_ttfr(diamond: &TribleSet) -> usize {
    find!(
        (e: Inline<GenId>, x: Inline<GenId>, y: Inline<GenId>),
        and!(
            pattern!(diamond, [{ ?e @ r1_schema::da: ?x }]),
            pattern!(diamond, [{ ?e @ r1_schema::db: ?y }]),
        )
    )
    .next()
    .map_or(0, |_| 1)
}

/// F5 full drain: total row count (expected [`F5_EXPECTED_ROWS`]).
pub fn f5_total(diamond: &TribleSet) -> usize {
    find!(
        (e: Inline<GenId>, x: Inline<GenId>, y: Inline<GenId>),
        and!(
            pattern!(diamond, [{ ?e @ r1_schema::da: ?x }]),
            pattern!(diamond, [{ ?e @ r1_schema::db: ?y }]),
        )
    )
    .count()
}

// ---------------------------------------------------------------------------
// Arch phase: RAM succinct-archive build.
// ---------------------------------------------------------------------------

/// One `SuccinctArchive<OrderedUniverse>` build from the checked-out
/// set (the `arch/build_ram` measure).
pub fn build_archive(set: &TribleSet) -> SuccinctArchive<OrderedUniverse> {
    set.into()
}

// ---------------------------------------------------------------------------
// Ladder phase: pile checkout (rung -> k mapping + Workspace::checkout).
// ---------------------------------------------------------------------------

type CommitHandle = Inline<Handle<SimpleArchive>>;

/// Walk a linear branch parents-first (oldest-first). Uses only
/// `repo::parent` facts.
fn commit_chain(
    reader: &subject::core::repo::pile::PileReader,
    head: CommitHandle,
) -> Vec<(CommitHandle, TribleSet)> {
    let mut chain = Vec::new();
    let mut cursor = Some(head);
    while let Some(handle) = cursor {
        let meta: TribleSet = reader.get(handle).expect("read commit metadata");
        let parents: Vec<CommitHandle> = find!(
            (p: Inline<Handle<SimpleArchive>>),
            pattern!(&meta, [{ repo::parent: ?p }])
        )
        .map(|(p,)| p)
        .collect();
        chain.push((handle, meta));
        cursor = match parents[..] {
            [] => None,
            [p] => Some(p),
            _ => panic!("merge commit in data branch (expected a linear chain)"),
        };
    }
    chain.reverse();
    chain
}

/// Open the pile, resolve the data branch, map the rung to k commits,
/// and measure `Workspace::checkout` of those commits — one span per
/// warmed iteration, timestamps against `base`.
///
/// !!! `Repository::new` appends one commit-metadata blob record to
/// the pile file on open — always point the runner at a clonefile copy
/// (`cp -c` on APFS, free) of the dataset pile, never at the original.
///
/// Returns the checked-out set (prefix-carved for sub-first-chunk
/// rungs), the per-iteration spans as `(begin_ns, duration_ns)`, and
/// the identity trible count. `Err` = workload-identity gate failure;
/// panics (API drift at hostile revs) escape to the caller's
/// [`quiet_catch`].
pub fn pile_checkout(
    path: &std::path::Path,
    branch: Option<&str>,
    rung: usize,
    iters: usize,
    warmup: usize,
    base: &Instant,
) -> Result<(TribleSet, Vec<(u64, u64)>, usize), String> {
    let mut pile = Pile::open(path).expect("open pile");
    pile.refresh().expect("load pile records");
    let reader = pile.reader().expect("pile reader");

    // Resolve branches by metadata::name. With `branch`, exact match;
    // else auto-pick the single branch not named "manifest".
    let branch_ids: Vec<Id> = pile
        .pins()
        .expect("list branches")
        .collect::<Result<Vec<_>, _>>()
        .expect("list branches");
    let mut named: Vec<(Id, String, TribleSet)> = Vec::new();
    for id in branch_ids {
        let Ok(Some(meta_handle)) = pile.head(id) else { continue };
        let Ok(meta): Result<TribleSet, _> = reader.get(meta_handle) else { continue };
        let handles: Vec<Inline<Handle<LongString>>> = find!(
            (n: Inline<Handle<LongString>>),
            pattern!(&meta, [{ metadata::name: ?n }])
        )
        .map(|(n,)| n)
        .collect();
        let [h] = handles[..] else { continue };
        let Ok(name): Result<anybytes::View<str>, _> = reader.get(h) else { continue };
        named.push((id, name.as_ref().to_owned(), meta));
    }
    let (branch_id, branch_name, branch_meta) = match branch {
        Some(want) => named
            .into_iter()
            .find(|(_, n, _)| n == want)
            .unwrap_or_else(|| panic!("no branch named {want:?} in pile")),
        None => {
            let mut data: Vec<_> = named.into_iter().filter(|(_, n, _)| n != "manifest").collect();
            match data.len() {
                1 => data.remove(0),
                n => panic!(
                    "cannot auto-pick a data branch ({n} non-manifest branches: {:?}) — pass --branch",
                    data.iter().map(|(_, n, _)| n.clone()).collect::<Vec<_>>()
                ),
            }
        }
    };

    let heads: Vec<CommitHandle> = find!(
        (c: Inline<Handle<SimpleArchive>>),
        pattern!(&branch_meta, [{ repo::head: ?c }])
    )
    .map(|(c,)| c)
    .collect();
    let [head] = heads[..] else { panic!("branch {branch_name:?} has no unique head commit") };
    let chain = commit_chain(&reader, head);

    // Rung -> k: one walk, per-commit tribles = SimpleArchive blob
    // length / 64.
    let mut handles: Vec<CommitHandle> = Vec::new();
    let mut cum: Vec<usize> = Vec::new();
    let mut total = 0usize;
    for (handle, meta) in &chain {
        let contents: Vec<CommitHandle> = find!(
            (c: Inline<Handle<SimpleArchive>>),
            pattern!(meta, [{ repo::content: ?c }])
        )
        .map(|(c,)| c)
        .collect();
        let [content] = contents[..] else { continue }; // skip empty commits
        let blob: Blob<SimpleArchive> = reader.get(content).expect("read content blob");
        total += blob.bytes.len() / 64;
        handles.push(*handle);
        cum.push(total);
    }
    assert!(!handles.is_empty(), "branch {branch_name:?} has no content commits");
    let (k, carve) = if rung < cum[0] {
        (1, Some(rung))
    } else {
        match cum.iter().position(|&c| c >= rung) {
            Some(idx) => (idx + 1, None),
            None => {
                println!(
                    "note     : rung {rung} exceeds pile total ~{total} tribles; using the full chain"
                );
                (handles.len(), None)
            }
        }
    };
    println!(
        "rung     : target {rung} -> k={k}/{} commits (~{} cumulative tribles{}) on branch {branch_name:?}",
        handles.len(),
        cum[k - 1],
        carve.map(|n| format!(", carving a {n}-trible sorted prefix")).unwrap_or_default()
    );

    // Workspace::checkout, one span per warmed iteration.
    let mut repo = Repository::new(pile, SigningKey::generate(&mut OsRng), TribleSet::new())
        .expect("create repository view");
    let mut ws = repo.pull(branch_id).expect("pull branch");
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut out: Option<TribleSet> = None;
    let mut ident: Option<usize> = None;
    let mut gate: Option<String> = None;
    for i in 0..(warmup + iters) {
        let recording = i >= warmup;
        let begin_ns = base.elapsed().as_nanos() as u64;
        let t = Instant::now();
        let co = ws.checkout(&handles[..k]).expect("checkout");
        let mut set = co.into_facts();
        if let Some(n) = carve {
            let mut prefix = TribleSet::new();
            for t in set.iter().take(n) {
                prefix.insert(t);
            }
            set = prefix;
        }
        if recording {
            spans.push((begin_ns, t.elapsed().as_nanos() as u64));
        }
        match ident {
            None => ident = Some(set.len()),
            Some(expected) if expected != set.len() => {
                gate = Some(format!(
                    "checkout-identity: iter {i} saw {} tribles, expected {expected}",
                    set.len()
                ));
            }
            _ => {}
        }
        out = Some(set);
    }
    drop(ws);
    repo.close().expect("close pile");
    if let Some(g) = gate {
        return Err(g);
    }
    Ok((out.expect("at least one iteration"), spans, ident.unwrap_or(0)))
}
