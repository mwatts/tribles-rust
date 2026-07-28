use core::panic;

use smallvec::SmallVec;

use crate::id::id_from_value;
use crate::id::id_into_value;
use crate::id::ID_LEN;
use crate::query::Binding;
use crate::query::Constraint;
use crate::query::Frontier;
use crate::query::RawTerm;
use crate::query::Term;
use crate::query::VariableId;
use crate::query::VariableSet;
use crate::trible::TribleSet;
use crate::inline::encodings::genid::GenId;
use crate::inline::InlineEncoding;
use crate::inline::RawInline;
use crate::inline::INLINE_LEN;
use crate::query::Candidates;
use crate::query::ProposalBuffer;

/// Batch size at which this source stops probing in frontier order and
/// probes in **index order** instead.
///
/// A batched `propose`/`confirm` is N covering-index lookups for N
/// parent bindings. Taken in frontier order those are N independent
/// descents from the root of a PATCH — pointer chasing, one cache miss
/// per level per probe, nothing shared between consecutive probes. The
/// keys are byte arrays and the PATCH is ordered on exactly those bytes
/// (a prefix passed to `has_prefix`/`infixes` is in *tree* order, so a
/// lexicographic sort of the prefixes is the tree's own descent order),
/// so sorting the batch's keys first turns N random descents into one
/// ordered walk: consecutive probes share their upper path, which stays
/// in cache, and the leaves are visited in address order the prefetcher
/// can follow.
///
/// Sorting buys two further things that are not about locality at all:
///
/// * **Duplicate keys collapse.** Several frontier rows routinely
///   project to the *same* key — a join whose parents fan in, or a
///   pattern with no bound position at all, where every row's key is
///   empty. Sorted, they are adjacent, so the index is walked once and
///   the result is fanned out to each row's segment.
/// * **Duplicate candidates collapse.** That fan-out puts the same
///   value under several rows of one key-run, and a sorted run presents
///   them adjacently, so the confirming probe is memoised across them.
///
/// This is the boundary between the two strategies, and the only thing
/// that separates them: both paths run the same code over a permutation
/// of the batch, and differ solely in whether that permutation is
/// sorted. Set it to `usize::MAX` to measure the plain frontier-order
/// loop.
const SORTED_PROBE_MIN: usize = 2;

/// Kills every entry named by `order` whose value fails `keep`, skipping
/// entries that are already dead — [`Candidates::retain`] over a
/// permutation instead of the region's own order.
///
/// The verdict is memoised across *adjacent equal values*, which costs
/// one 32-byte compare and pays for itself whenever the permutation is
/// sorted: a key-run fanned out over several frontier rows carries each
/// candidate once per row, and sorted they arrive back to back.
#[inline]
fn retain_at(cands: &mut Candidates<'_>, order: &[u32], mut keep: impl FnMut(&RawInline) -> bool) {
    let mut memo: Option<(RawInline, bool)> = None;
    for &i in order {
        let i = i as usize;
        if !cands.is_live(i) {
            continue;
        }
        let value = cands.values()[i];
        let verdict = match memo {
            Some((seen, verdict)) if seen == value => verdict,
            _ => {
                let verdict = keep(&value);
                memo = Some((value, verdict));
                verdict
            }
        };
        if !verdict {
            cands.kill(i);
        }
    }
}

/// Kills every entry named by `order` — the [`Candidates::kill_all`] of
/// a permutation, used where a row's own bound positions are malformed.
#[inline]
fn kill_at(cands: &mut Candidates<'_>, order: &[u32]) {
    for &i in order {
        cands.kill(i as usize);
    }
}

/// A triple-pattern lookup against a [`TribleSet`].
///
/// Created by [`TribleSet::pattern`](crate::query::TriblePattern::pattern)
/// (typically via the [`pattern!`](crate::pattern) macro). Each position —
/// entity, attribute, value — is a [`Term`]: a variable to solve for or a
/// constant pinned at construction. The constraint uses the six covering
/// indexes (EAV, EVA, AEV, AVE, VEA, VAE) to provide tight estimates and
/// fast proposals regardless of which positions are bound; a constant
/// position simply enters that dispatch as bound from the start.
///
/// When all three positions have values, [`satisfied`](Constraint::satisfied)
/// checks whether the triple exists in the set, enabling composite
/// constraints to prune dead branches early.
pub struct TribleSetConstraint {
    term_e: RawTerm,
    term_a: RawTerm,
    term_v: RawTerm,
    set: TribleSet,
}

impl TribleSetConstraint {
    /// Creates a triple-pattern constraint over `set` for the given
    /// entity, attribute, and value terms.
    pub fn new<V: InlineEncoding>(
        e: impl Into<Term<GenId>>,
        a: impl Into<Term<GenId>>,
        v: impl Into<Term<V>>,
        set: TribleSet,
    ) -> Self {
        TribleSetConstraint {
            term_e: e.into().erase(),
            term_a: a.into().erase(),
            term_v: v.into().erase(),
            set,
        }
    }
}

impl TribleSetConstraint {
    /// Kills the entries `order` names — indices into `cands` — whose
    /// value is inconsistent with `binding`.
    ///
    /// `order` is a permutation of some part of the region rather than a
    /// range, because the region spans a whole [`Frontier`] and the
    /// caller decides in which order the covering index is probed. Every
    /// entry it names must belong to a row whose bound positions equal
    /// `binding`'s; the caller establishes that by grouping the region by
    /// probe key.
    fn confirm_at(
        &self,
        variable: VariableId,
        binding: &Binding,
        cands: &mut Candidates<'_>,
        order: &[u32],
    ) {
        let e_var = self.term_e.is_var(variable);
        let a_var = self.term_a.is_var(variable);
        let v_var = self.term_v.is_var(variable);

        if !e_var && !a_var && !v_var {
            return;
        }

        let e_bound = if let Some(e) = self.term_e.position_value(binding) {
            let Some(e) = id_from_value(e) else {
                kill_at(cands, order);
                return;
            };
            Some(e)
        } else {
            None
        };
        let a_bound = if let Some(a) = self.term_a.position_value(binding) {
            let Some(a) = id_from_value(a) else {
                kill_at(cands, order);
                return;
            };
            Some(a)
        } else {
            None
        };
        let v_bound = self.term_v.position_value(binding);

        match (e_bound, a_bound, v_bound, e_var, a_var, v_var) {
            (None, None, None, true, false, false) => retain_at(cands, order, |value| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                self.set.eav.has_prefix(&id)
            }),
            (None, None, None, false, true, false) => retain_at(cands, order, |value| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                self.set.aev.has_prefix(&id)
            }),
            (None, None, None, false, false, true) => {
                retain_at(cands, order, |value| self.set.vea.has_prefix(value))
            }
            (Some(e), None, None, false, true, false) => retain_at(cands, order, |value| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                let mut prefix = [0u8; ID_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e[..]);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&id);
                self.set.eav.has_prefix(&prefix)
            }),
            (Some(e), None, None, false, false, true) => retain_at(cands, order, |value| {
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e[..]);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(value);
                self.set.eva.has_prefix(&prefix)
            }),
            (None, Some(a), None, true, false, false) => retain_at(cands, order, |value| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                let mut prefix = [0u8; ID_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a[..]);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&id);
                self.set.aev.has_prefix(&prefix)
            }),
            (None, Some(a), None, false, false, true) => retain_at(cands, order, |value| {
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a[..]);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(value);
                self.set.ave.has_prefix(&prefix)
            }),
            (None, None, Some(v), true, false, false) => retain_at(cands, order, |value| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                let mut prefix = [0u8; INLINE_LEN + ID_LEN];
                prefix[0..INLINE_LEN].copy_from_slice(&v[..]);
                prefix[INLINE_LEN..INLINE_LEN + ID_LEN].copy_from_slice(&id);
                self.set.vea.has_prefix(&prefix)
            }),
            (None, None, Some(v), false, true, false) => retain_at(cands, order, |value| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                let mut prefix = [0u8; INLINE_LEN + ID_LEN];
                prefix[0..INLINE_LEN].copy_from_slice(&v[..]);
                prefix[INLINE_LEN..INLINE_LEN + ID_LEN].copy_from_slice(&id);
                self.set.vae.has_prefix(&prefix)
            }),
            (None, Some(a), Some(v), true, false, false) => retain_at(cands, order, |value: &[u8; 32]| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                let mut prefix = [0u8; ID_LEN + INLINE_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(v);
                prefix[ID_LEN + INLINE_LEN..ID_LEN + INLINE_LEN + ID_LEN].copy_from_slice(&id);
                self.set.ave.has_prefix(&prefix)
            }),
            (Some(e), None, Some(v), false, true, false) => retain_at(cands, order, |value: &[u8; 32]| {
                let Some(id) = id_from_value(value) else {
                    return false;
                };
                let mut prefix = [0u8; ID_LEN + INLINE_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(v);
                prefix[ID_LEN + INLINE_LEN..ID_LEN + INLINE_LEN + ID_LEN].copy_from_slice(&id);
                self.set.eva.has_prefix(&prefix)
            }),
            (Some(e), Some(a), None, false, false, true) => retain_at(cands, order, |value: &[u8; 32]| {
                let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&a);
                prefix[ID_LEN + ID_LEN..ID_LEN + ID_LEN + INLINE_LEN].copy_from_slice(value);
                self.set.eav.has_prefix(&prefix)
            }),

            // Same-Variable arms. The proposal value plays two roles
            // (e and v, or e and a, or a and v); we build a full
            // 64-byte trible key from each proposal and check
            // `has_prefix` against the appropriate index.
            (_, Some(a), _, true, false, true) => retain_at(cands, order, |value| {
                // pattern(x, a, x): proposal is both entity and value.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&a[..]);
                prefix[ID_LEN + ID_LEN..].copy_from_slice(&id_into_value(&id));
                self.set.eav.has_prefix(&prefix)
            }),
            (_, None, _, true, false, true) => retain_at(cands, order, |value| {
                // pattern(x, ?, x): proposal is entity == value, any attr.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN..].copy_from_slice(&id_into_value(&id));
                self.set.eva.has_prefix(&prefix)
            }),
            (_, _, Some(v), true, true, false) => retain_at(cands, order, |value| {
                // pattern(x, x, v): proposal is entity == attribute.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN + ID_LEN..].copy_from_slice(&v[..]);
                self.set.eav.has_prefix(&prefix)
            }),
            (_, _, None, true, true, false) => retain_at(cands, order, |value| {
                // pattern(x, x, ?): proposal is entity == attribute, any v.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&id);
                self.set.eav.has_prefix(&prefix)
            }),
            (Some(e), _, _, false, true, true) => retain_at(cands, order, |value| {
                // pattern(e, x, x): proposal is attribute == value.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN + ID_LEN..].copy_from_slice(&id_into_value(&id));
                self.set.eav.has_prefix(&prefix)
            }),
            (None, _, _, false, true, true) => retain_at(cands, order, |value| {
                // pattern(?, x, x): proposal is attribute == value, any e.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN..].copy_from_slice(&id_into_value(&id));
                self.set.ave.has_prefix(&prefix)
            }),
            (_, _, _, true, true, true) => retain_at(cands, order, |value| {
                // pattern(x, x, x): proposal plays all three roles.
                let Some(id) = id_from_value(value) else { return false; };
                let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&id);
                prefix[ID_LEN + ID_LEN..].copy_from_slice(&id_into_value(&id));
                self.set.eav.has_prefix(&prefix)
            }),
            _ => panic!("invalid trible constraint state"),
        }
    }

    fn propose_row(&self, variable: VariableId, binding: &Binding, proposals: &mut ProposalBuffer) {
        let e_var = self.term_e.is_var(variable);
        let a_var = self.term_a.is_var(variable);
        let v_var = self.term_v.is_var(variable);

        if !e_var && !a_var && !v_var {
            return;
        }

        let e_bound = if let Some(e) = self.term_e.position_value(binding) {
            let Some(e) = id_from_value(e) else {
                return;
            };
            Some(e)
        } else {
            None
        };
        let a_bound = if let Some(a) = self.term_a.position_value(binding) {
            let Some(a) = id_from_value(a) else {
                return;
            };
            Some(a)
        } else {
            None
        };
        let v_bound = self.term_v.position_value(binding);

        match (e_bound, a_bound, v_bound, e_var, a_var, v_var) {
            // Distinct-position combinations: the queried variable
            // appears in exactly one trible slot. Drive enumeration
            // from the most selective covering index.
            (None, None, None, true, false, false) => {
                self.set.eav.infixes(&[0; 0], &mut |e: &[u8; 16]| {
                    proposals.push(id_into_value(e))
                });
            }
            (None, None, None, false, true, false) => {
                self.set.aev.infixes(&[0; 0], &mut |a: &[u8; 16]| {
                    proposals.push(id_into_value(a))
                });
            }
            (None, None, None, false, false, true) => {
                self.set
                    .vea
                    .infixes(&[0; 0], &mut |&v: &[u8; 32]| proposals.push(v));
            }

            (Some(e), None, None, false, true, false) => {
                self.set
                    .eav
                    .infixes(&e, &mut |a: &[u8; 16]| proposals.push(id_into_value(a)));
            }
            (Some(e), None, None, false, false, true) => {
                self.set
                    .eva
                    .infixes(&e, &mut |&v: &[u8; 32]| proposals.push(v));
            }

            (None, Some(a), None, true, false, false) => {
                self.set
                    .aev
                    .infixes(&a, &mut |e: &[u8; 16]| proposals.push(id_into_value(e)));
            }
            (None, Some(a), None, false, false, true) => {
                self.set
                    .ave
                    .infixes(&a, &mut |&v: &[u8; 32]| proposals.push(v));
            }

            (None, None, Some(v), true, false, false) => {
                self.set
                    .vea
                    .infixes(v, &mut |e: &[u8; 16]| proposals.push(id_into_value(e)));
            }
            (None, None, Some(v), false, true, false) => {
                self.set
                    .vae
                    .infixes(v, &mut |a: &[u8; 16]| proposals.push(id_into_value(a)));
            }
            (None, Some(a), Some(v), true, false, false) => {
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a[..]);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(&v[..]);
                self.set.ave.infixes(&prefix, &mut |e: &[u8; 16]| {
                    proposals.push(id_into_value(e))
                });
            }
            (Some(e), None, Some(v), false, true, false) => {
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e[..]);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(&v[..]);
                self.set.eva.infixes(&prefix, &mut |a: &[u8; 16]| {
                    proposals.push(id_into_value(a))
                });
            }
            (Some(e), Some(a), None, false, false, true) => {
                let mut prefix = [0u8; ID_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e[..]);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&a[..]);
                self.set
                    .eav
                    .infixes(&prefix, &mut |&v: &[u8; 32]| proposals.push(v));
            }

            // Same-Variable arms. The covering indexes already
            // dedup; the equality constraint between two positions
            // is enforced inline via `has_prefix`. No HashSet — the
            // index walk pays the dedup cost once.
            (_, Some(a), _, true, false, true) => {
                // pattern(x, a, x) — entity equals value, attr bound.
                self.set.aev.infixes(&a, &mut |e: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                    prefix[0..ID_LEN].copy_from_slice(e);
                    prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&a[..]);
                    prefix[ID_LEN + ID_LEN..].copy_from_slice(&id_into_value(e));
                    if self.set.eav.has_prefix(&prefix) {
                        proposals.push(id_into_value(e));
                    }
                });
            }
            (_, None, _, true, false, true) => {
                // pattern(x, ?, x) — entity equals value, attr free.
                // Enumerate distinct entities; keep those with ∃ a . (e, a, e).
                self.set.eav.infixes(&[0; 0], &mut |e: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                    prefix[0..ID_LEN].copy_from_slice(e);
                    prefix[ID_LEN..].copy_from_slice(&id_into_value(e));
                    if self.set.eva.has_prefix(&prefix) {
                        proposals.push(id_into_value(e));
                    }
                });
            }
            (_, _, Some(v), true, true, false) => {
                // pattern(x, x, v) — entity equals attribute, value bound.
                self.set.vae.infixes(v, &mut |a: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                    prefix[0..ID_LEN].copy_from_slice(a);
                    prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(a);
                    prefix[ID_LEN + ID_LEN..].copy_from_slice(&v[..]);
                    if self.set.eav.has_prefix(&prefix) {
                        proposals.push(id_into_value(a));
                    }
                });
            }
            (_, _, None, true, true, false) => {
                // pattern(x, x, ?) — entity equals attribute, value free.
                self.set.aev.infixes(&[0; 0], &mut |a: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + ID_LEN];
                    prefix[0..ID_LEN].copy_from_slice(a);
                    prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(a);
                    if self.set.eav.has_prefix(&prefix) {
                        proposals.push(id_into_value(a));
                    }
                });
            }
            (Some(e), _, _, false, true, true) => {
                // pattern(e, x, x) — attribute equals value, entity bound.
                self.set.eav.infixes(&e, &mut |a: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                    prefix[0..ID_LEN].copy_from_slice(&e[..]);
                    prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(a);
                    prefix[ID_LEN + ID_LEN..].copy_from_slice(&id_into_value(a));
                    if self.set.eav.has_prefix(&prefix) {
                        proposals.push(id_into_value(a));
                    }
                });
            }
            (None, _, _, false, true, true) => {
                // pattern(?, x, x) — attribute equals value, entity free.
                self.set.aev.infixes(&[0; 0], &mut |a: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                    prefix[0..ID_LEN].copy_from_slice(a);
                    prefix[ID_LEN..].copy_from_slice(&id_into_value(a));
                    if self.set.ave.has_prefix(&prefix) {
                        proposals.push(id_into_value(a));
                    }
                });
            }
            (_, _, _, true, true, true) => {
                // pattern(x, x, x) — all three positions share one
                // Variable. Enumerate distinct entities; keep those
                // with (e, e, id_into_value(e)) in the set.
                self.set.eav.infixes(&[0; 0], &mut |e: &[u8; 16]| {
                    let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                    prefix[0..ID_LEN].copy_from_slice(e);
                    prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(e);
                    prefix[ID_LEN + ID_LEN..].copy_from_slice(&id_into_value(e));
                    if self.set.eav.has_prefix(&prefix) {
                        proposals.push(id_into_value(e));
                    }
                });
            }
            _ => panic!("TribleSetConstraint: unreachable position-bound combo"),
        }
    }

    /// Whether `variable` occupies any position of this pattern — the
    /// relevance check every protocol method opens with, hoisted so the
    /// batched entry points can skip building a probe-key matrix for a
    /// variable they have no opinion about.
    fn touches(&self, variable: VariableId) -> bool {
        self.term_e.is_var(variable)
            || self.term_a.is_var(variable)
            || self.term_v.is_var(variable)
    }

    /// Appends the bytes of every position this constraint reads under
    /// `binding` — the bound ones and the constants, in e-a-v order — to
    /// `out`. This is the row's **probe key**.
    ///
    /// Two rows with the same key are indistinguishable to
    /// [`propose_row`](Self::propose_row) and
    /// [`confirm_at`](Self::confirm_at): both dispatch on *which*
    /// positions have a value, which a [`Frontier`] shares by
    /// construction, and read nothing else from the binding. So the key
    /// is a complete summary of a row for this source's purposes — equal
    /// keys may be answered once, and the key's byte order is the order
    /// the covering index wants to be probed in.
    ///
    /// Every row of a frontier writes the same number of bytes, so the
    /// keys form a fixed-stride matrix.
    fn write_probe_key(&self, binding: &Binding, out: &mut SmallVec<[u8; 128]>) {
        for term in [&self.term_e, &self.term_a, &self.term_v] {
            if let Some(value) = term.position_value(binding) {
                out.extend_from_slice(value);
            }
        }
    }

    /// The frontier's probe keys as a fixed-stride matrix, plus the row
    /// permutation that visits them in key order (or in frontier order,
    /// below [`SORTED_PROBE_MIN`]).
    fn probe_keys(&self, frontier: &Frontier<'_>) -> (SmallVec<[u8; 128]>, usize, Vec<u32>) {
        let rows = frontier.len();
        let mut keys: SmallVec<[u8; 128]> = SmallVec::new();
        for row in 0..rows {
            self.write_probe_key(&frontier.row(row), &mut keys);
        }
        let stride = if rows == 0 { 0 } else { keys.len() / rows };
        let mut order: Vec<u32> = (0..rows as u32).collect();
        if stride != 0 && rows >= SORTED_PROBE_MIN {
            // Ties break on the row number, so the permutation is a
            // deterministic function of the frontier rather than of the
            // sort's internal choices.
            order.sort_unstable_by(|&a, &b| {
                let a = a as usize;
                let b = b as usize;
                keys[a * stride..(a + 1) * stride]
                    .cmp(&keys[b * stride..(b + 1) * stride])
                    .then(a.cmp(&b))
            });
        }
        (keys, stride, order)
    }
}

impl<'a> Constraint<'a> for TribleSetConstraint {

    /// Returns the set of variable positions (constant positions are
    /// invisible to the engine).
    fn variables(&self) -> VariableSet {
        let mut variables = VariableSet::new_empty();
        self.term_e.add_to(&mut variables);
        self.term_a.add_to(&mut variables);
        self.term_v.add_to(&mut variables);
        variables
    }

    /// Uses the covering indexes (EAV, EVA, AEV, AVE, VEA, VAE) to
    /// count matching entries via `segmented_len`. The index chosen
    /// depends on which of the other two positions are already bound,
    /// giving tight estimates regardless of access pattern.
    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        let e_var = self.term_e.is_var(variable);
        let a_var = self.term_a.is_var(variable);
        let v_var = self.term_v.is_var(variable);

        if !e_var && !a_var && !v_var {
            return None;
        }

        let e_bound = if let Some(e) = self.term_e.position_value(binding) {
            let Some(e) = id_from_value(e) else {
                return Some(0);
            };
            Some(e)
        } else {
            None
        };
        let a_bound = if let Some(a) = self.term_a.position_value(binding) {
            let Some(a) = id_from_value(a) else {
                return Some(0);
            };
            Some(a)
        } else {
            None
        };
        let v_bound = self.term_v.position_value(binding);

        Some(match (e_bound, a_bound, v_bound, e_var, a_var, v_var) {
            // Legal distinct-position combinations (queried var
            // appears in exactly one trible position).
            (None, None, None, true, false, false) => self.set.eav.segmented_len(&[0; 0]),
            (None, None, None, false, true, false) => self.set.aev.segmented_len(&[0; 0]),
            (None, None, None, false, false, true) => self.set.vea.segmented_len(&[0; 0]),
            (Some(e), None, None, false, true, false) => {
                let mut prefix = [0u8; ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e[..]);
                self.set.eav.segmented_len(&prefix)
            }
            (Some(e), None, None, false, false, true) => {
                let mut prefix = [0u8; ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e[..]);
                self.set.eva.segmented_len(&prefix)
            }
            (None, Some(a), None, true, false, false) => {
                let mut prefix = [0u8; ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a[..]);
                self.set.aev.segmented_len(&prefix)
            }
            (None, Some(a), None, false, false, true) => {
                let mut prefix = [0u8; ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a[..]);
                self.set.ave.segmented_len(&prefix)
            }
            (None, None, Some(v), true, false, false) => {
                let mut prefix = [0u8; INLINE_LEN];
                prefix[0..INLINE_LEN].copy_from_slice(&v[..]);
                self.set.vea.segmented_len(&prefix)
            }
            (None, None, Some(v), false, true, false) => {
                let mut prefix = [0u8; INLINE_LEN];
                prefix[0..INLINE_LEN].copy_from_slice(&v[..]);
                self.set.vae.segmented_len(&prefix)
            }
            (None, Some(a), Some(v), true, false, false) => {
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&a);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(v);
                self.set.ave.segmented_len(&prefix)
            }
            (Some(e), None, Some(v), false, true, false) => {
                let mut prefix = [0u8; ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e);
                prefix[ID_LEN..ID_LEN + INLINE_LEN].copy_from_slice(v);
                self.set.eva.segmented_len(&prefix)
            }
            (Some(e), Some(a), None, false, false, true) => {
                let mut prefix = [0u8; ID_LEN + ID_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&a);
                self.set.eav.segmented_len(&prefix)
            }

            // Same-Variable in two positions. Conservative upper
            // bounds via covering-index `segmented_len` — the
            // actual count would require a `has_prefix` check per
            // candidate, which the planner doesn't need: any tight
            // upper bound drives variable-ordering decisions just
            // as well. `propose` does the real per-candidate work.
            (_, Some(a), _, true, false, true) => {
                // e == v (self-edge), attribute bound.
                let mut prefix = [0u8; ID_LEN];
                prefix.copy_from_slice(&a[..]);
                self.set.aev.segmented_len(&prefix)
            }
            (_, None, _, true, false, true) => {
                // e == v, attribute free.
                self.set.eav.segmented_len(&[0; 0])
            }
            (_, _, Some(v), true, true, false) => {
                // e == a, value bound.
                let mut prefix = [0u8; INLINE_LEN];
                prefix.copy_from_slice(&v[..]);
                self.set.vae.segmented_len(&prefix)
            }
            (_, _, None, true, true, false) => {
                // e == a, value free.
                self.set.aev.segmented_len(&[0; 0])
            }
            (Some(e), _, _, false, true, true) => {
                // a == v, entity bound.
                let mut prefix = [0u8; ID_LEN];
                prefix.copy_from_slice(&e[..]);
                self.set.eav.segmented_len(&prefix)
            }
            (None, _, _, false, true, true) => {
                // a == v, entity free.
                self.set.aev.segmented_len(&[0; 0])
            }
            (_, _, _, true, true, true) => {
                // pattern(x, x, x) — all three positions share one
                // Variable. Conservative upper bound: distinct
                // entities in the set.
                self.set.eav.segmented_len(&[0; 0])
            }
            _ => panic!("TribleSetConstraint: unreachable position-bound combo"),
        } as usize)
    }

    /// Enumerates matching values for every row of the batch: N covering
    /// index walks for N parent bindings, into one segmented buffer.
    ///
    /// The bound *set* is shared across the frontier, so every row takes
    /// the same covering index and differs only in its prefix bytes —
    /// there is no per-row re-plan, just N prefixes. Those prefixes are
    /// visited in **key order** rather than frontier order (see
    /// [`SORTED_PROBE_MIN`]), which makes the walks an ordered sweep of
    /// the PATCH instead of N descents from its root, and lets rows that
    /// share a prefix be answered once and fanned out. Segment order
    /// follows the probe order; a proposer may visit rows in any order,
    /// and each row's candidates still arrive contiguously under its own
    /// tag.
    fn propose(
        &self,
        variable: VariableId,
        frontier: &Frontier<'_>,
        proposals: &mut ProposalBuffer,
    ) {
        let rows = frontier.len();
        if rows == 0 || !self.touches(variable) {
            return;
        }
        let (keys, stride, order) = self.probe_keys(frontier);
        let key = |row: u32| {
            let row = row as usize;
            &keys[row * stride..(row + 1) * stride]
        };

        let mut shared: Vec<RawInline> = Vec::new();
        let mut run_start = 0;
        while run_start < rows {
            let lead = order[run_start];
            let mut run_end = run_start + 1;
            while run_end < rows && key(order[run_end]) == key(lead) {
                run_end += 1;
            }

            let base = proposals.len();
            proposals.open(lead);
            self.propose_row(variable, &frontier.row(lead as usize), proposals);
            if run_end - run_start > 1 {
                // The remaining rows of the run have the same prefix, so
                // they have the same candidates: copy rather than walk
                // the index again.
                shared.clear();
                shared.extend_from_slice(&proposals[base..]);
                for &row in &order[run_start + 1..run_end] {
                    proposals.open(row);
                    proposals.extend_from_slice(&shared);
                }
            }
            run_start = run_end;
        }
    }

    /// Retains only proposals whose combined key (their own row's bound
    /// positions + the proposed value) has a matching prefix in the
    /// appropriate index.
    ///
    /// The region spans the whole batch, so it is walked in **probe
    /// order**: grouped by probe key — coarser than by parent tag, since
    /// distinct rows that agree on this constraint's positions confirm
    /// identically — and, within a group, in value order, which is the
    /// order the covering index is laid out in. Below
    /// [`SORTED_PROBE_MIN`] the region is walked in its own order
    /// instead, which is the same grouping the tags already carry.
    fn confirm(&self, variable: VariableId, frontier: &Frontier<'_>, cands: &mut Candidates<'_>) {
        let entries = cands.len();
        if entries == 0 || frontier.is_empty() || !self.touches(variable) {
            return;
        }
        let mut keys: SmallVec<[u8; 128]> = SmallVec::new();
        for row in 0..frontier.len() {
            self.write_probe_key(&frontier.row(row), &mut keys);
        }
        let stride = keys.len() / frontier.len();
        let key = |row: u32| {
            let row = row as usize;
            &keys[row * stride..(row + 1) * stride]
        };
        // The tags are read after the region turns mutable, so take a
        // copy of them rather than holding a borrow across the kills.
        let parents: SmallVec<[u32; 64]> = SmallVec::from_slice(cands.parents());

        let mut order: SmallVec<[u32; 64]> = (0..entries as u32).collect();
        if entries >= SORTED_PROBE_MIN {
            let values = cands.values();
            order.sort_unstable_by(|&a, &b| {
                key(parents[a as usize])
                    .cmp(key(parents[b as usize]))
                    .then_with(|| values[a as usize].cmp(&values[b as usize]))
                    .then(a.cmp(&b))
            });
        }

        let mut run_start = 0;
        while run_start < entries {
            let lead = parents[order[run_start] as usize];
            let mut run_end = run_start + 1;
            while run_end < entries && key(parents[order[run_end] as usize]) == key(lead) {
                run_end += 1;
            }
            let binding = frontier.row(lead as usize);
            self.confirm_at(variable, &binding, cands, &order[run_start..run_end]);
            run_start = run_end;
        }
    }

    /// When all three positions have values (bound or constant), checks
    /// whether the triple exists in the EAV index. Returns `true`
    /// optimistically when any position is still unbound.
    fn satisfied(&self, binding: &Binding) -> bool {
        let e = self.term_e.position_value(binding);
        let a = self.term_a.position_value(binding);
        let v = self.term_v.position_value(binding);
        match (e, a, v) {
            (Some(e_raw), Some(a_raw), Some(v_raw)) => {
                let Some(e) = id_from_value(e_raw) else {
                    return false;
                };
                let Some(a) = id_from_value(a_raw) else {
                    return false;
                };
                let mut prefix = [0u8; ID_LEN + ID_LEN + INLINE_LEN];
                prefix[0..ID_LEN].copy_from_slice(&e);
                prefix[ID_LEN..ID_LEN + ID_LEN].copy_from_slice(&a);
                prefix[ID_LEN + ID_LEN..].copy_from_slice(v_raw);
                self.set.eav.has_prefix(&prefix)
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::find;
    use crate::id::rngid;
    use crate::query::TriblePattern;
    use crate::query::Variable;
    use crate::trible::Trible;
    use crate::trible::TribleSet;
    use crate::inline::encodings::UnknownInline;
    use crate::inline::Inline;

    #[test]
    fn constant() {
        let mut set = TribleSet::new();
        set.insert(&Trible::new(
            &rngid(),
            &rngid(),
            &Inline::<UnknownInline>::new([0; 32]),
        ));

        let q = find! {
            (e: Inline<_>, a: Inline<_>, v: Inline<_>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        };
        let r: Vec<_> = q.collect();

        assert_eq!(1, r.len())
    }

    #[test]
    fn self_edge_pattern_e_eq_v() {
        // Verify `pattern(x, a, x)` (same Variable in entity and
        // value positions) enumerates self-edge entities without
        // panicking. Adds 3 self-edges and 2 non-self tribles for
        // the same attribute; the query should return exactly 3.
        use crate::inline::encodings::genid::GenId;
        use crate::and;

        // Helper: encode a 16-byte id as a GenId-style Inline value
        // (32 bytes: upper 16 zero, lower 16 = id).
        fn id_as_inline(id: &[u8; 16]) -> Inline<GenId> {
            let mut bytes = [0u8; 32];
            bytes[16..32].copy_from_slice(id);
            Inline::<GenId>::new(bytes)
        }

        let mut set = TribleSet::new();
        let a = rngid();
        let self1 = rngid();
        let self2 = rngid();
        let self3 = rngid();
        let other = rngid();

        // 3 self-edges: x has attribute a with value x
        for x in [&self1, &self2, &self3] {
            set.insert(&Trible::new(x, &a, &id_as_inline(x)));
        }
        // 2 non-self tribles with the same attribute
        set.insert(&Trible::new(&self1, &a, &id_as_inline(&other)));
        set.insert(&Trible::new(&other, &a, &id_as_inline(&self2)));

        // Free attribute: count all self-edges
        let q = find! {
            (x: Inline<GenId>, attr: Inline<GenId>),
            set.pattern(x, attr, x)
        };
        let r: Vec<_> = q.collect();
        assert_eq!(3, r.len(), "expected 3 self-edges, got {}", r.len());

        // Bound attribute: should still be 3 since only attribute a
        // appears in our self-edges
        let q = find! {
            (x: Inline<GenId>, attr: Inline<GenId>),
            and!(
                attr.is(id_as_inline(&a)),
                set.pattern(x, attr, x)
            )
        };
        let r: Vec<_> = q.collect();
        assert_eq!(3, r.len(), "expected 3 self-edges with bound attr, got {}", r.len());
    }

    #[test]
    fn entity_attr_dup_pattern() {
        // `pattern(x, x, v)` — entity equals attribute.
        use crate::inline::encodings::genid::GenId;

        fn id_as_inline(id: &[u8; 16]) -> Inline<GenId> {
            let mut bytes = [0u8; 32];
            bytes[16..32].copy_from_slice(id);
            Inline::<GenId>::new(bytes)
        }

        let mut set = TribleSet::new();
        // Two entities that double as their own attributes.
        let dup1 = rngid();
        let dup2 = rngid();
        let other = rngid();
        let v1 = rngid();
        let v2 = rngid();

        set.insert(&Trible::new(&dup1, &dup1, &id_as_inline(&v1)));
        set.insert(&Trible::new(&dup2, &dup2, &id_as_inline(&v2)));
        // Non-dup tribles
        set.insert(&Trible::new(&dup1, &other, &id_as_inline(&v1)));
        set.insert(&Trible::new(&other, &dup1, &id_as_inline(&v1)));

        let q = find! {
            (x: Inline<GenId>, val: Inline<GenId>),
            set.pattern(x, x, val)
        };
        let r: Vec<_> = q.collect();
        assert_eq!(2, r.len(), "expected 2 entity-attr dups, got {}", r.len());
    }

    #[test]
    fn attr_value_dup_pattern() {
        // `pattern(e, x, x)` — attribute equals value.
        use crate::inline::encodings::genid::GenId;

        fn id_as_inline(id: &[u8; 16]) -> Inline<GenId> {
            let mut bytes = [0u8; 32];
            bytes[16..32].copy_from_slice(id);
            Inline::<GenId>::new(bytes)
        }

        let mut set = TribleSet::new();
        let dup1 = rngid(); // attribute id (and value id)
        let dup2 = rngid();
        let other_attr = rngid();
        let e1 = rngid();
        let e2 = rngid();
        let e3 = rngid();

        // attribute equals value tribles
        set.insert(&Trible::new(&e1, &dup1, &id_as_inline(&dup1)));
        set.insert(&Trible::new(&e2, &dup2, &id_as_inline(&dup2)));
        // Non-dup: different value
        set.insert(&Trible::new(&e3, &dup1, &id_as_inline(&dup2)));
        // Non-dup: attribute differs from value's id portion
        set.insert(&Trible::new(&e3, &other_attr, &id_as_inline(&dup1)));

        let q = find! {
            (e: Inline<GenId>, x: Inline<GenId>),
            set.pattern(e, x, x)
        };
        let r: Vec<_> = q.collect();
        assert_eq!(2, r.len(), "expected 2 attr-value dups, got {}", r.len());
    }

    #[test]
    fn all_three_same_pattern() {
        // `pattern(x, x, x)` — entity, attribute, and value all
        // share one Variable. The natural Wikidata meta-class
        // example: Q35120 (entity) is itself, instances-of itself.
        // Here: 2 entities that fully self-assert (e == a, value
        // encodes e) and several near-misses that share two of
        // the three roles.
        use crate::inline::encodings::genid::GenId;

        fn id_as_inline(id: &[u8; 16]) -> Inline<GenId> {
            let mut bytes = [0u8; 32];
            bytes[16..32].copy_from_slice(id);
            Inline::<GenId>::new(bytes)
        }

        let mut set = TribleSet::new();
        let xxx1 = rngid();
        let xxx2 = rngid();
        let other = rngid();

        // 2 full triples: (x, x, x)
        set.insert(&Trible::new(&xxx1, &xxx1, &id_as_inline(&xxx1)));
        set.insert(&Trible::new(&xxx2, &xxx2, &id_as_inline(&xxx2)));
        // Near-miss: e == a but value differs
        set.insert(&Trible::new(&xxx1, &xxx1, &id_as_inline(&other)));
        // Near-miss: e == v but attribute differs
        set.insert(&Trible::new(&xxx2, &other, &id_as_inline(&xxx2)));
        // Near-miss: a == v but entity differs
        set.insert(&Trible::new(&other, &xxx1, &id_as_inline(&xxx1)));

        let q = find! {
            (x: Inline<GenId>),
            set.pattern(x, x, x)
        };
        let r: Vec<_> = q.collect();
        assert_eq!(2, r.len(), "expected 2 self-self-self triples, got {}", r.len());
    }
}
