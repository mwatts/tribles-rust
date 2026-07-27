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
//!
//! Native to this crate (not vendored):
//! - the Harkonnen R2 white-box fixtures F6..F15 (attribute ids minted
//!   2026-07-27). Each one isolates ONE engine decision changed in the
//!   Term-native / propose-confirm work of this week, so a regression
//!   gets a name instead of a vibe. None is path-shaped, so none is
//!   rpq-gated; only F10 (which reads the GPU routing threshold out of
//!   `triblespace-gpu`) is feature-gated.

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
// Raw engine protocol surface — needed only by the R2 fixtures: F11's
// hand-written `Constraint` wrapper, F12's programmatic chain, and F13's
// hand-rolled variable context.
use subject::core::query::{
    Binding, Candidates, Constraint, ProposalBuffer, ProposeCursor, VariableContext, VariableId,
    VariableSet,
};
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

// ---------------------------------------------------------------------------
// Harkonnen R2 white-box fixtures (F6..F15).
//
// R1 (F1..F5) probed the engine as a black box: adversarial *shapes*
// whose cost profile exposed whatever the solver did. R2 is the
// complement — each fixture isolates exactly ONE engine decision changed
// in this week's Term-native / propose-confirm work, with a deterministic
// construction and an EXACT expected row count derived from the
// construction (never a magic number). A regression then gets a name.
//
// Every builder is deterministic: fixed anchors, `Ids`' splitmix walk,
// no RNG. Sizes are deliberately modest — the whole suite must stay
// inside seconds on a machine that is already busy — and each fixture's
// doc records what its size buys.
// ---------------------------------------------------------------------------

mod r2_schema {
    use subject::core::prelude::*;

    attributes! {
        // F6 — the eight `or!` arm attributes (k = F6_ARMS).
        "84AAEA2CDB0F31C9926D5BA55DCB646B" as u0: inlineencodings::GenId;
        "A784BFAFA148EFF689FA757C2A95EA2C" as u1: inlineencodings::GenId;
        "2AAB1B3A425B16D9680BF525CB0A9496" as u2: inlineencodings::GenId;
        "1A6306A5E7A469266D3F6BC3B4F0F830" as u3: inlineencodings::GenId;
        "5478E2F71D3C8FA4FCFD72D653C2F050" as u4: inlineencodings::GenId;
        "595533D46DA73D4D9EDAE23C69DA8B28" as u5: inlineencodings::GenId;
        "BBB163B3E6F55B2040940AD229F012D7" as u6: inlineencodings::GenId;
        "B358920E7B50BC2334BDBA2B9EE95D44" as u7: inlineencodings::GenId;
        // F7 — hub skew: in-edge (source -> node) and out-edge (node -> target).
        "0E64D47BDE5A9CCE0F41C6384BC7935F" as hs: inlineencodings::GenId;
        "D88DE6AC745F82E035551D9D2625F976" as ht: inlineencodings::GenId;
        // F8 — witness multiplicity: the two hidden-variable fans.
        "52A53A202DFC13F7074E0A697ABAB30C" as wa: inlineencodings::GenId;
        "236FC28BA8AA72FDE8E5723B403A0BD2" as wb: inlineencodings::GenId;
        // F9 — mask density: proposer side and confirmer side.
        "8DC793B45ADE8B81ADE789E8313BF974" as ma: inlineencodings::GenId;
        "B0E12EFD7CF941497603F09DEDF89760" as mb: inlineencodings::GenId;
        // F10 — GPU confirm-batch threshold: proposer side and confirmer side.
        "3573F33EBA6E94CBAFFD27FFBAA4A0F8" as ga: inlineencodings::GenId;
        "3E9767FEE8A1EE68C58F0419817D5140" as gb: inlineencodings::GenId;
        // F11 — lying estimates: the small source and the large source.
        "7503DE9B6CA70780D9096223E4DD1A08" as la: inlineencodings::GenId;
        "EE3CC3EBE8F1E540DBC396C985E282F2" as lb: inlineencodings::GenId;
        // F12 — deep chain: the single functional hop edge.
        "7A7F3D3A4EBFB4FC617D063261FB592C" as hop: inlineencodings::GenId;
        // F13 — constant pressure: five string-valued slots per entity.
        "FCE112E22C1592CC487EF5320D9E25D7" as c0: inlineencodings::ShortString;
        "B57751A976F6908E53938CF80A615DCC" as c1: inlineencodings::ShortString;
        "8C72F6255184F8FA9BDBD15574F540A8" as c2: inlineencodings::ShortString;
        "ED5816955F19EFA0C6BBA2C2F7BA77ED" as c3: inlineencodings::ShortString;
        "61C46CDF6E77A00A3B566C5157F57EFB" as c4: inlineencodings::ShortString;
        // F14 — widening ramp: the wide root and the selective confirmer.
        "6389E8FE7CA25BE81A2BAFEF79C8EDBC" as w1: inlineencodings::GenId;
        "ED3F65982ED22D4008A510A19D4798A8" as w2: inlineencodings::GenId;
        // F15 — union dedup pressure: the two overlapping arms.
        "32697C0766C902A3A6BEA17656631E5F" as ua: inlineencodings::GenId;
        "CBF362B65F450B457B32C05C5198868F" as ub: inlineencodings::GenId;
    }
}

/// Locality prefix of every R2 *anchor* — a well-known id a builder and
/// its query both name without plumbing a value through a return type.
/// Distinct from `Ids`' bulk prefix (`0xD46B0001`) so anchors can never
/// collide with generated filler.
const ANCHOR_TAG: u32 = 0xD46B_0002;

/// The anchor registry (`n` -> role). Keeping every anchor in ONE
/// numbering makes accidental reuse across fixtures impossible to write
/// by mistake:
///
/// | `n`      | role |
/// |----------|------|
/// | 0..=7    | F6 arm hub value (the literal each arm matches) |
/// | 8..=15   | F6 arm decoy value (same attribute, wrong literal) |
/// | 16, 17   | F9 proposer root, confirmer probe |
/// | 18, 19   | F10 below-threshold root, probe |
/// | 20, 21   | F10 above-threshold root, probe |
/// | 22, 23   | F11 small source root, large source root |
/// | 24, 25   | F14 wide root, selective probe |
/// | 26       | F15 shared hub value |
/// | 100..    | F13 constant-pressure entities (`100 + i`) |
fn anchor(n: u64) -> ExclusiveId {
    let mut raw = [0u8; 16];
    raw[..4].copy_from_slice(&ANCHOR_TAG.to_be_bytes());
    raw[4..12].copy_from_slice(&Ids::splitmix64(n).to_be_bytes());
    raw[12..].copy_from_slice(&Ids::splitmix64(n ^ 0xA5A5_5A5A).to_be_bytes()[..4]);
    ExclusiveId::force(Id::new(raw).expect("nonzero prefix"))
}

// ---------------------------------------------------------------------------
// F6 — union fan.
//
// INTERROGATES: the `or!` aligned-arms path, and dead-variant gating.
//
// Before the Term-native fold every attribute constant and literal value
// became a fresh hidden variable, so two structurally identical
// `pattern!` invocations never declared the same variable set and
// `UnionConstraint::new` panicked — the documented `or!(pattern!(..),
// pattern!(..))` form was dead code. The fold makes constants Terms
// below the variable layer, so all k arms here declare exactly `{e}`
// even though each carries a DIFFERENT attribute constant AND a
// DIFFERENT literal value. This is the synthetic twin of sparqloscope
// q3.
//
// Dead-variant gating: with `?e` the only variable, every arm's
// `satisfied()` is exact the moment `?e` binds (entity, attribute and
// value all pinned), so k-1 arms are *provably* dead at confirm time.
// The decoy entities below make the literal load-bearing: they carry the
// right attribute with the wrong value and must contribute no rows.
// ---------------------------------------------------------------------------

/// F6: number of `or!` arms — one per minted arm attribute.
pub const F6_ARMS: usize = 8;
/// F6: matching entities per arm.
pub const F6_PER_ARM: usize = 256;
/// F6: decoy entities per arm (right attribute, wrong literal).
pub const F6_DECOYS_PER_ARM: usize = 32;

/// F6 expected rows. Each of the `F6_ARMS * F6_PER_ARM` matching
/// entities carries exactly one arm edge, so it satisfies exactly one
/// arm and yields exactly one binding of the single query variable
/// `?e`; the `F6_ARMS * F6_DECOYS_PER_ARM` decoys satisfy no arm.
/// 8 * 256 = 2048.
pub const F6_EXPECTED_ROWS: usize = F6_ARMS * F6_PER_ARM;

/// The literal value arm `j` matches.
fn f6_hub(j: usize) -> ExclusiveId {
    anchor(j as u64)
}

/// The literal value arm `j`'s decoys carry instead.
fn f6_decoy(j: usize) -> ExclusiveId {
    anchor(8 + j as u64)
}

/// One arm-`j` edge. The attribute must be a literal constant at each
/// call site (that is the whole point of the fixture), so the arm index
/// is dispatched rather than indexed.
fn f6_edge(j: usize, e: &ExclusiveId, v: &ExclusiveId) -> Fragment {
    match j {
        0 => entity! { e @ r2_schema::u0: v },
        1 => entity! { e @ r2_schema::u1: v },
        2 => entity! { e @ r2_schema::u2: v },
        3 => entity! { e @ r2_schema::u3: v },
        4 => entity! { e @ r2_schema::u4: v },
        5 => entity! { e @ r2_schema::u5: v },
        6 => entity! { e @ r2_schema::u6: v },
        7 => entity! { e @ r2_schema::u7: v },
        _ => unreachable!("F6 has exactly {F6_ARMS} arms"),
    }
}

/// F6 builder: per arm, `F6_PER_ARM` entities pointing at that arm's hub
/// value and `F6_DECOYS_PER_ARM` entities pointing at its decoy value.
pub fn build_union_fan() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    for j in 0..F6_ARMS {
        let hub = f6_hub(j);
        let decoy = f6_decoy(j);
        for _ in 0..F6_PER_ARM {
            let e = ids.mint();
            set += f6_edge(j, &e, &hub);
        }
        for _ in 0..F6_DECOYS_PER_ARM {
            let e = ids.mint();
            set += f6_edge(j, &e, &decoy);
        }
    }
    set
}

/// F6 full drain: total row count (expected [`F6_EXPECTED_ROWS`]).
pub fn f6_total(set: &TribleSet) -> usize {
    find!(
        (e: Inline<GenId>),
        or!(
            pattern!(set, [{ ?e @ r2_schema::u0: &f6_hub(0) }]),
            pattern!(set, [{ ?e @ r2_schema::u1: &f6_hub(1) }]),
            pattern!(set, [{ ?e @ r2_schema::u2: &f6_hub(2) }]),
            pattern!(set, [{ ?e @ r2_schema::u3: &f6_hub(3) }]),
            pattern!(set, [{ ?e @ r2_schema::u4: &f6_hub(4) }]),
            pattern!(set, [{ ?e @ r2_schema::u5: &f6_hub(5) }]),
            pattern!(set, [{ ?e @ r2_schema::u6: &f6_hub(6) }]),
            pattern!(set, [{ ?e @ r2_schema::u7: &f6_hub(7) }]),
        )
    )
    .count()
}

// ---------------------------------------------------------------------------
// F7 — hub skew.
//
// INTERROGATES: cardinality-estimate robustness and dynamic variable
// ordering under an extreme degree distribution.
//
// `IntersectionConstraint` picks its proposer by minimum estimate and
// re-sorts the unbound variables whenever a binding invalidates an
// estimate (`push_next_variable`). A single node with a 20 000x degree
// makes the *average* estimate a lie: any plan chosen from global
// cardinality alone is wrong for one of the two populations. The row
// count is invariant under every legal plan, so it gates correctness
// while the spans record what the skew cost.
// ---------------------------------------------------------------------------

/// F7: uniform nodes (in-degree 1, out-degree 1).
pub const F7_UNIFORM: usize = 1_000;
/// F7: out-degree of the single hub node (in-degree 1).
pub const F7_HUB_FANOUT: usize = 20_000;

/// F7 expected rows. The join `?x -hs-> ?h -ht-> ?y` yields, per middle
/// node, in-degree x out-degree rows: the hub contributes
/// `1 * F7_HUB_FANOUT`, each uniform node `1 * 1`.
/// 20 000 + 1 000 = 21 000.
pub const F7_EXPECTED_ROWS: usize = F7_HUB_FANOUT + F7_UNIFORM;

/// F7 builder: one hub with `F7_HUB_FANOUT` out-edges plus `F7_UNIFORM`
/// degree-1 nodes; every middle node has exactly one in-edge, so the
/// skew lives entirely in the out-degree.
pub fn build_hub_skew() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();

    let hub = ids.mint();
    let hub_src = ids.mint();
    set += entity! { &hub_src @ r2_schema::hs: &hub };
    for _ in 0..F7_HUB_FANOUT {
        let y = ids.mint();
        set += entity! { &hub @ r2_schema::ht: &y };
    }

    for _ in 0..F7_UNIFORM {
        let h = ids.mint();
        let x = ids.mint();
        let y = ids.mint();
        set += entity! { &x @ r2_schema::hs: &h };
        set += entity! { &h @ r2_schema::ht: &y };
    }
    set
}

/// F7 full drain: total row count (expected [`F7_EXPECTED_ROWS`]).
pub fn f7_total(set: &TribleSet) -> usize {
    find!(
        (x: Inline<GenId>, h: Inline<GenId>, y: Inline<GenId>),
        and!(
            pattern!(set, [{ ?x @ r2_schema::hs: ?h }]),
            pattern!(set, [{ ?h @ r2_schema::ht: ?y }]),
        )
    )
    .count()
}

// ---------------------------------------------------------------------------
// F8 — witness multiplicity.
//
// INTERROGATES: the bag-semantics contract.
//
// The engine is a bag of COMPLETE bindings: `Query::next` emits a row
// every time the unbound set empties, with no projection dedup, so
// hidden (`temp!`) variables multiply the visible head. Dedup belongs to
// the consumer. This fixture pins both halves of that contract at once
// with the two-query idiom — the same constraint drained as a bag and
// collected into a set — and is the fixture that fails loudly if head
// dedup is ever reintroduced inside the engine (the bag measure would
// collapse onto the distinct one).
// ---------------------------------------------------------------------------

/// F8: entities carrying both fans.
pub const F8_ENTITIES: usize = 256;
/// F8: `wa` out-degree per entity (hidden witness `?m`).
pub const F8_WA_DEG: usize = 8;
/// F8: `wb` out-degree per entity (hidden witness `?n`).
pub const F8_WB_DEG: usize = 8;

/// F8 expected BAG rows: per entity every `(?m, ?n)` pair is a distinct
/// complete binding, so `F8_WA_DEG * F8_WB_DEG` = 64 rows per entity.
/// 256 * 8 * 8 = 16 384 — exactly 64x the projected-distinct count.
pub const F8_EXPECTED_BAG_ROWS: usize = F8_ENTITIES * F8_WA_DEG * F8_WB_DEG;

/// F8 expected DISTINCT rows: the head is `?e` alone and every entity
/// has both fans, so the client-side set has one member per entity.
/// 256.
pub const F8_EXPECTED_DISTINCT_ROWS: usize = F8_ENTITIES;

/// F8 builder: `F8_ENTITIES` entities, each with `F8_WA_DEG` `wa` edges
/// and `F8_WB_DEG` `wb` edges to fresh targets.
pub fn build_witness_multiplicity() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    for _ in 0..F8_ENTITIES {
        let e = ids.mint();
        for _ in 0..F8_WA_DEG {
            let m = ids.mint();
            set += entity! { &e @ r2_schema::wa: &m };
        }
        for _ in 0..F8_WB_DEG {
            let n = ids.mint();
            set += entity! { &e @ r2_schema::wb: &n };
        }
    }
    set
}

/// F8 bag drain: one row per complete `(e, m, n)` binding, projected to
/// `?e` (expected [`F8_EXPECTED_BAG_ROWS`]).
pub fn f8_bag(set: &TribleSet) -> usize {
    find!(
        (e: Inline<GenId>),
        temp!(
            (m, n),
            and!(
                pattern!(set, [{ ?e @ r2_schema::wa: ?m }]),
                pattern!(set, [{ ?e @ r2_schema::wb: ?n }]),
            )
        )
    )
    .count()
}

/// F8 client-side distinct: the SAME constraint, deduplicated by the
/// consumer (expected [`F8_EXPECTED_DISTINCT_ROWS`]).
pub fn f8_distinct(set: &TribleSet) -> usize {
    let seen: std::collections::HashSet<[u8; 32]> = find!(
        (e: Inline<GenId>),
        temp!(
            (m, n),
            and!(
                pattern!(set, [{ ?e @ r2_schema::wa: ?m }]),
                pattern!(set, [{ ?e @ r2_schema::wb: ?n }]),
            )
        )
    )
    .map(|(e,)| e.raw)
    .collect();
    seen.len()
}

// ---------------------------------------------------------------------------
// F9 — mask density extremes.
//
// INTERROGATES: the engine-owned liveness design (`ProposalBuffer` +
// `Candidates`), generalizing the F5 lesson.
//
// Confirmers kill entries in place and nothing is ever compacted, so the
// engine scans over dead entries. That is a deliberate trade: no
// compaction cost, no index invalidation, one word per entry writable
// without contention. Its two worst cases are the two variants here — a
// mask that kills nearly everything (a long scan over corpses to reach
// each survivor) and a mask that kills nothing (pure overhead on the
// liveness words).
//
// Both variants are built so the two sides carry IDENTICAL cardinality
// (`F9_REGION` entries each). That matters: `IntersectionConstraint`
// always lets the *lowest*-estimate child propose, so a merely-small
// confirmer would simply become the proposer and the mask would never
// run. Equal estimates force a genuine mask over a full-size region and
// make selectivity a property of the data, not of the plan.
// ---------------------------------------------------------------------------

/// F9: candidates on each side of both variants.
pub const F9_REGION: usize = 20_000;
/// F9 sparse: every `F9_SPARSE_STRIDE`-th candidate survives.
pub const F9_SPARSE_STRIDE: usize = 1_000;

/// F9 sparse expected rows: the confirmer shares exactly the candidates
/// at CONSTRUCTION indices `0, F9_SPARSE_STRIDE, 2*F9_SPARSE_STRIDE,
/// ...` — `Ids::mint`'s splitmix suffix scatters those through the
/// proposal's value order, so the survivors are spread across the region
/// rather than clustered, which is what makes this a density test and
/// not a prefix test. 20 000 / 1 000 = 20 survivors (a 99.9% kill
/// rate).
pub const F9_SPARSE_EXPECTED_ROWS: usize = F9_REGION / F9_SPARSE_STRIDE;

/// F9 dense expected rows: both sides carry the same `F9_REGION` values,
/// so the mask kills nothing. 20 000.
pub const F9_DENSE_EXPECTED_ROWS: usize = F9_REGION;

/// F9 proposer root (the `ma` side).
fn f9_root() -> ExclusiveId {
    anchor(16)
}

/// F9 confirmer probe (the `mb` side).
fn f9_probe() -> ExclusiveId {
    anchor(17)
}

/// F9 sparse builder: both sides hold `F9_REGION` values but only every
/// `F9_SPARSE_STRIDE`-th one coincides.
pub fn build_mask_sparse() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let root = f9_root();
    let probe = f9_probe();
    for i in 0..F9_REGION {
        let v = ids.mint();
        set += entity! { &root @ r2_schema::ma: &v };
        if i % F9_SPARSE_STRIDE == 0 {
            set += entity! { &probe @ r2_schema::mb: &v };
        } else {
            let miss = ids.mint();
            set += entity! { &probe @ r2_schema::mb: &miss };
        }
    }
    set
}

/// F9 dense builder: both sides hold exactly the same `F9_REGION`
/// values, so no candidate is ever killed.
pub fn build_mask_dense() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let root = f9_root();
    let probe = f9_probe();
    for _ in 0..F9_REGION {
        let v = ids.mint();
        set += entity! { &root @ r2_schema::ma: &v };
        set += entity! { &probe @ r2_schema::mb: &v };
    }
    set
}

/// F9 full drain (either variant): total row count.
pub fn f9_total(set: &TribleSet) -> usize {
    find!(
        (v: Inline<GenId>),
        and!(
            pattern!(set, [{ &f9_root() @ r2_schema::ma: ?v }]),
            pattern!(set, [{ &f9_probe() @ r2_schema::mb: ?v }]),
        )
    )
    .count()
}

// ---------------------------------------------------------------------------
// F10 — GPU confirm-batch threshold boundary.
//
// INTERROGATES: routing hysteresis around `DEFAULT_MIN_CONFIRM_BATCH`,
// and threshold rot.
//
// `triblespace-gpu` routes a confirm region to the device only when it
// holds at least `DEFAULT_MIN_CONFIRM_BATCH` live candidates; smaller
// regions run the canonical CPU probes. The threshold is a MEASURED
// crossover (an M4 Max sweep recorded in its doc comment), which means
// it is exactly the kind of constant that silently rots. This fixture
// straddles it: two runs whose only difference is one candidate on
// either side of the boundary. The constant is READ from the engine
// (never copied here), so moving it moves the fixture.
//
// Gate on ROWS only — the answer must be the region size on both sides,
// which is the property routing may never change. Timings across the
// boundary are the interesting signal but are not a gate.
// ---------------------------------------------------------------------------

/// F10: the routing threshold, read from the engine so this fixture
/// cannot drift away from it.
#[cfg(feature = "gpu")]
pub const F10_THRESHOLD: usize = subject::gpu::DEFAULT_MIN_CONFIRM_BATCH;

/// F10: candidates just BELOW the routing threshold (CPU probes).
#[cfg(feature = "gpu")]
pub const F10_BELOW: usize = F10_THRESHOLD - 1;

/// F10: candidates just ABOVE the routing threshold (device confirm).
#[cfg(feature = "gpu")]
pub const F10_ABOVE: usize = F10_THRESHOLD + 1;

/// F10 root of the region of size `n`.
#[cfg(feature = "gpu")]
fn f10_root(above: bool) -> ExclusiveId {
    anchor(if above { 20 } else { 18 })
}

/// F10 probe of the region of size `n`.
#[cfg(feature = "gpu")]
fn f10_probe(above: bool) -> ExclusiveId {
    anchor(if above { 21 } else { 19 })
}

/// F10 builder: a fully-live region of exactly `n` candidates —
/// proposer and confirmer carry the same `n` values, so every candidate
/// survives and the region handed to `confirm` is `n` entries wide.
#[cfg(feature = "gpu")]
pub fn build_gpu_boundary(above: bool, n: usize) -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let root = f10_root(above);
    let probe = f10_probe(above);
    for _ in 0..n {
        let v = ids.mint();
        set += entity! { &root @ r2_schema::ga: &v };
        set += entity! { &probe @ r2_schema::gb: &v };
    }
    set
}

/// F10 full drain: total row count, expected to equal the region size
/// (`F10_BELOW` / `F10_ABOVE`) on both sides of the boundary.
#[cfg(feature = "gpu")]
pub fn f10_total(above: bool, set: &TribleSet) -> usize {
    find!(
        (v: Inline<GenId>),
        and!(
            pattern!(set, [{ &f10_root(above) @ r2_schema::ga: ?v }]),
            pattern!(set, [{ &f10_probe(above) @ r2_schema::gb: ?v }]),
        )
    )
    .count()
}

// ---------------------------------------------------------------------------
// F11 — lying estimates.
//
// INTERROGATES: graceful degradation of dynamic variable ordering and of
// the propose/confirm role choice when `Constraint::estimate` is wrong.
//
// The trait explicitly licenses inexact estimates ("the estimate need
// not be exact — it guides variable ordering, not correctness"). This
// fixture cashes that promise: the SAME query is run three times, once
// honestly and twice through a wrapper whose `estimate` is wrong by
// `F11_LIE_FACTOR` in each direction, forwarding every other method
// verbatim. Both lies flip which side proposes and which side confirms.
// All three must produce the same rows; only the spans may differ.
// ---------------------------------------------------------------------------

/// Test-only [`Constraint`] wrapper that scales the wrapped estimate by
/// `mul / div` and forwards every other method unchanged. This is the
/// only place in the suite that implements `Constraint` by hand; it
/// exists so a fixture can hand the planner a known-wrong number.
pub struct LyingEstimate<C> {
    inner: C,
    mul: usize,
    div: usize,
}

impl<C> LyingEstimate<C> {
    /// Report `factor`x MORE candidates than the wrapped constraint has.
    pub fn over(inner: C, factor: usize) -> Self {
        Self {
            inner,
            mul: factor,
            div: 1,
        }
    }

    /// Report `factor`x FEWER candidates than the wrapped constraint has.
    pub fn under(inner: C, factor: usize) -> Self {
        Self {
            inner,
            mul: 1,
            div: factor,
        }
    }
}

impl<'a, C: Constraint<'a>> Constraint<'a> for LyingEstimate<C> {
    fn variables(&self) -> VariableSet {
        self.inner.variables()
    }

    /// The lie. Clamped to at least 1 so the planner's `ilog2` bucketing
    /// still sees a well-formed magnitude.
    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        self.inner
            .estimate(variable, binding)
            .map(|e| (e.saturating_mul(self.mul) / self.div).max(1))
    }

    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut ProposalBuffer) {
        self.inner.propose(variable, binding, proposals)
    }

    fn propose_chunk(
        &self,
        variable: VariableId,
        binding: &Binding,
        cursor: &mut ProposeCursor,
        budget: usize,
        proposals: &mut ProposalBuffer,
    ) -> bool {
        self.inner
            .propose_chunk(variable, binding, cursor, budget, proposals)
    }

    fn confirm(&self, variable: VariableId, binding: &Binding, cands: &mut Candidates<'_>) {
        self.inner.confirm(variable, binding, cands)
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        self.inner.satisfied(binding)
    }

    fn influence(&self, variable: VariableId) -> VariableSet {
        self.inner.influence(variable)
    }
}

/// F11: candidates on the small (honest proposer) side.
pub const F11_SMALL: usize = 200;
/// F11: candidates on the large side.
pub const F11_LARGE: usize = 10_000;
/// F11: how far the wrapper's estimate is off, in each direction.
pub const F11_LIE_FACTOR: usize = 100;

/// F11 expected rows (identical for all three plans): the small side's
/// values are a subset of the large side's, so the intersection is the
/// small side. 200.
pub const F11_EXPECTED_ROWS: usize = F11_SMALL;

/// F11 small source root (`la`).
fn f11_small_root() -> ExclusiveId {
    anchor(22)
}

/// F11 large source root (`lb`).
fn f11_large_root() -> ExclusiveId {
    anchor(23)
}

/// F11 builder: the large side holds `F11_LARGE` values of which the
/// first `F11_SMALL` are exactly the small side's.
///
/// Honest estimates: 200 vs 10 000, so the small side proposes.
/// `over` (small x100 = 20 000 > 10 000) and `under` (large /100 = 100 <
/// 200) both hand the proposer role to the large side instead.
pub fn build_lying_estimates() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let small = f11_small_root();
    let large = f11_large_root();
    for i in 0..F11_LARGE {
        let v = ids.mint();
        set += entity! { &large @ r2_schema::lb: &v };
        if i < F11_SMALL {
            set += entity! { &small @ r2_schema::la: &v };
        }
    }
    set
}

/// F11 honest plan: the true estimates, small side proposes.
pub fn f11_truth(set: &TribleSet) -> usize {
    find!(
        (v: Inline<GenId>),
        and!(
            pattern!(set, [{ &f11_small_root() @ r2_schema::la: ?v }]),
            pattern!(set, [{ &f11_large_root() @ r2_schema::lb: ?v }]),
        )
    )
    .count()
}

/// F11 over-estimate plan: the small side claims `F11_LIE_FACTOR`x MORE
/// candidates than it has.
pub fn f11_over(set: &TribleSet) -> usize {
    find!(
        (v: Inline<GenId>),
        and!(
            LyingEstimate::over(
                pattern!(set, [{ &f11_small_root() @ r2_schema::la: ?v }]),
                F11_LIE_FACTOR
            ),
            pattern!(set, [{ &f11_large_root() @ r2_schema::lb: ?v }]),
        )
    )
    .count()
}

/// F11 under-estimate plan: the large side claims `F11_LIE_FACTOR`x
/// FEWER candidates than it has.
pub fn f11_under(set: &TribleSet) -> usize {
    find!(
        (v: Inline<GenId>),
        and!(
            pattern!(set, [{ &f11_small_root() @ r2_schema::la: ?v }]),
            LyingEstimate::under(
                pattern!(set, [{ &f11_large_root() @ r2_schema::lb: ?v }]),
                F11_LIE_FACTOR
            ),
        )
    )
    .count()
}

// ---------------------------------------------------------------------------
// F12 — deep chain.
//
// INTERROGATES: search-stack depth and the approach to the 128-variable
// ceiling.
//
// `Binding` is a fixed `[RawInline; 128]`, `VariableContext` asserts
// `next_index < 128`, and the search stack / unbound list are
// `ArrayVec<_, 128>`. A 60-hop chain allocates 61 variables — just under
// half the budget, deep enough that every level of the DFS stack is
// exercised and cheap enough to stay in the suite's time envelope. The
// chain is built programmatically through `TriblePattern::pattern` (not
// 60 lines of `pattern!`), which keeps the hop count a knob.
// ---------------------------------------------------------------------------

/// F12: hops in each chain (variables allocated = `F12_HOPS + 1`).
pub const F12_HOPS: usize = 60;
/// F12: complete chains of exactly `F12_HOPS` edges.
pub const F12_CHAINS: usize = 2;
/// F12: decoy chains, half the required length — they force the search
/// to unwind from deep in the stack without ever completing.
pub const F12_DECOYS: usize = 8;

/// F12 expected rows: a `hop` edge is functional (out-degree 1) and the
/// chains are disjoint, so a path of exactly `F12_HOPS` edges exists
/// once per complete chain, starting at its head; the decoy chains are
/// `F12_HOPS / 2` edges long and complete none. 2.
pub const F12_EXPECTED_ROWS: usize = F12_CHAINS;

/// F12 builder: `F12_CHAINS` disjoint chains of `F12_HOPS` edges plus
/// `F12_DECOYS` half-length chains.
pub fn build_deep_chain() -> TribleSet {
    let mut ids = Ids::new();
    let mut set = TribleSet::new();
    let chain = |set: &mut TribleSet, ids: &mut Ids, edges: usize| {
        let mut prev = ids.mint();
        for _ in 0..edges {
            let next = ids.mint();
            *set += entity! { &prev @ r2_schema::hop: &next };
            prev = next;
        }
    };
    for _ in 0..F12_CHAINS {
        chain(&mut set, &mut ids, F12_HOPS);
    }
    for _ in 0..F12_DECOYS {
        chain(&mut set, &mut ids, F12_HOPS / 2);
    }
    set
}

/// F12 full drain: total row count (expected [`F12_EXPECTED_ROWS`]).
///
/// The head projects the two ends of the chain; the `F12_HOPS - 1`
/// interior variables are allocated from the `find!` context inside the
/// constraint expression, exactly as `temp!` would, so the query really
/// does carry `F12_HOPS + 1` variables.
pub fn f12_total(set: &TribleSet) -> usize {
    find!(
        (head: Inline<GenId>, tail: Inline<GenId>),
        {
            let hop: Inline<GenId> = IntoInline::to_inline(r2_schema::hop.id());
            let mut constraints: Vec<Box<dyn Constraint + Send + Sync>> = Vec::new();
            let mut prev: Variable<GenId> = head;
            for i in 0..F12_HOPS {
                let next: Variable<GenId> = if i + 1 == F12_HOPS {
                    tail
                } else {
                    __local_find_context!().next_variable()
                };
                constraints.push(Box::new(set.pattern(prev, hop, next)));
                prev = next;
            }
            std::sync::Arc::new(IntersectionConstraint::new(constraints))
        }
    )
    .count()
}
