use proptest::collection::vec;
use proptest::prelude::*;
use std::collections::HashSet;
use triblespace_core::id::rngid;
use triblespace_core::prelude::*;
use triblespace_core::query::{
    Binding, BindingStore, Candidates, Constraint, ContainsConstraint, Frontier, ProposalBuffer, TriblePattern, Variable, VariableContext,
};
use triblespace_core::trible::{Fragment, Trible};
use triblespace_core::inline::encodings::genid::GenId;
use triblespace_core::inline::encodings::UnknownInline;

mod test_ns {
    use triblespace_core::prelude::*;
    attributes! {
        "BB00000000000000BB00000000000001" as pub link: inlineencodings::GenId;
        "BB00000000000000BB00000000000002" as pub label: inlineencodings::ShortString;
        "BB00000000000000BB00000000000003" as pub other_link: inlineencodings::GenId;
    }
}

fn arb_trible() -> impl Strategy<Value = Trible> {
    (
        prop::array::uniform16(1u8..=255),
        prop::array::uniform16(1u8..=255),
        prop::array::uniform32(any::<u8>()),
    )
        .prop_map(|(e, a, v)| {
            let mut data = [0u8; 64];
            data[0..16].copy_from_slice(&e);
            data[16..32].copy_from_slice(&a);
            data[32..64].copy_from_slice(&v);
            Trible::force_raw(data).expect("non-nil e and a")
        })
}

fn arb_tribleset(max: usize) -> impl Strategy<Value = TribleSet> {
    vec(arb_trible(), 1..max).prop_map(|tribles| {
        let mut set = TribleSet::new();
        for t in &tribles {
            set.insert(t);
        }
        set
    })
}

proptest! {
    // ── TribleSetConstraint: estimate accuracy ─────────────────────────

    #[test]
    fn estimate_entity_count_matches_actual(set in arb_tribleset(20)) {
        let mut ctx = VariableContext::new();
        let e = ctx.next_variable();
        let a = ctx.next_variable();
        let v: Variable<UnknownInline> = ctx.next_variable();
        let constraint = set.pattern(e, a, v);

        let frontier = Frontier::default();
        let estimate = constraint.estimate(e.index, &Binding::default()).unwrap();

        // Estimate should be >= actual distinct entity count
        let mut proposals = ProposalBuffer::new();
        constraint.propose(e.index, &frontier, &mut proposals);
        prop_assert!(estimate >= proposals.len(),
            "estimate {} < actual proposals {}", estimate, proposals.len());
    }

    #[test]
    fn propose_entity_all_in_set(set in arb_tribleset(20)) {
        let mut ctx = VariableContext::new();
        let e = ctx.next_variable();
        let a = ctx.next_variable();
        let v: Variable<UnknownInline> = ctx.next_variable();
        let constraint = set.pattern(e, a, v);

        let frontier = Frontier::default();
        let mut proposals = ProposalBuffer::new();
        constraint.propose(e.index, &frontier, &mut proposals);

        // Every proposed entity must appear in at least one trible
        for entity_raw in &proposals {
            let found = set.iter().any(|t| &t.data[0..16] == &entity_raw[16..32]);
            prop_assert!(found,
                "proposed entity not found in any trible");
        }
    }

    #[test]
    fn find_returns_only_existing_triples(set in arb_tribleset(15)) {
        let results: Vec<_> = find!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        ).collect();

        // Every result triple must exist in the set
        for (e, a, v) in &results {
            let found = set.iter().any(|t| {
                &t.data[0..16] == &e.raw[16..32]
                    && &t.data[16..32] == &a.raw[16..32]
                    && &t.data[32..64] == &v.raw[..]
            });
            prop_assert!(found, "query result not in set");
        }

        // And the count matches
        prop_assert_eq!(results.len(), set.len(),
            "result count {} != set size {}", results.len(), set.len());
    }

    // ── Satisfied: consistency ──────────────────────────────────────────

    #[test]
    fn satisfied_true_for_existing_triple(set in arb_tribleset(10)) {
        // Pick the first trible and bind all three variables
        if let Some(t) = set.iter().next() {
            let mut ctx = VariableContext::new();
            let e = ctx.next_variable();
            let a = ctx.next_variable();
            let v: Variable<UnknownInline> = ctx.next_variable();
            let constraint = set.pattern(e, a, v);

            let mut binding = BindingStore::new();
            let mut e_val = [0u8; 32];
            e_val[16..32].copy_from_slice(&t.data[0..16]);
            binding.bind(e.index, &e_val);
            let mut a_val = [0u8; 32];
            a_val[16..32].copy_from_slice(&t.data[16..32]);
            binding.bind(a.index, &a_val);
            binding.bind(v.index, &t.data[32..64].try_into().unwrap());

            prop_assert!(constraint.satisfied(&binding.view()),
                "existing triple should satisfy constraint");
        }
    }

    #[test]
    fn satisfied_false_for_absent_triple(
        set in arb_tribleset(5),
        fake in arb_trible()
    ) {
        // If the fake trible is NOT in the set, satisfied should be false
        if !set.contains(&fake) {
            let mut ctx = VariableContext::new();
            let e = ctx.next_variable();
            let a = ctx.next_variable();
            let v: Variable<UnknownInline> = ctx.next_variable();
            let constraint = set.pattern(e, a, v);

            let mut binding = BindingStore::new();
            let mut e_val = [0u8; 32];
            e_val[16..32].copy_from_slice(&fake.data[0..16]);
            binding.bind(e.index, &e_val);
            let mut a_val = [0u8; 32];
            a_val[16..32].copy_from_slice(&fake.data[16..32]);
            binding.bind(a.index, &a_val);
            binding.bind(v.index, &fake.data[32..64].try_into().unwrap());

            prop_assert!(!constraint.satisfied(&binding.view()),
                "absent triple should not satisfy constraint");
        }
    }

    // ── IntersectionConstraint: tighter than either child ──────────────

    // ── Fragment algebra ─────────────────────────────────────────────

    #[test]
    fn fragment_union_commutative(
        a_tribles in vec(arb_trible(), 1..5),
        b_tribles in vec(arb_trible(), 1..5),
    ) {
        let id_a = rngid();
        let id_b = rngid();
        let mut set_a = TribleSet::new();
        for t in &a_tribles { set_a.insert(t); }
        let mut set_b = TribleSet::new();
        for t in &b_tribles { set_b.insert(t); }
        let frag_a = Fragment::rooted(*id_a, set_a);
        let frag_b = Fragment::rooted(*id_b, set_b);

        let ab = frag_a.clone() + frag_b.clone();
        let ba = frag_b + frag_a;
        prop_assert_eq!(ab, ba);
    }

    #[test]
    fn fragment_root_preserved(tribles in vec(arb_trible(), 1..5)) {
        let id = rngid();
        let mut set = TribleSet::new();
        for t in &tribles { set.insert(t); }
        let frag = Fragment::rooted(*id, set);
        prop_assert_eq!(frag.root(), Some(*id));
    }

    #[test]
    fn fragment_facts_deref_consistent(tribles in vec(arb_trible(), 1..10)) {
        let id = rngid();
        let mut set = TribleSet::new();
        for t in &tribles { set.insert(t); }
        let frag = Fragment::rooted(*id, set.clone());
        // Deref to TribleSet should give same len
        prop_assert_eq!(frag.len(), set.len());
        prop_assert_eq!(frag.facts(), &set);
    }

    #[test]
    fn fragment_union_accumulates_exports(
        a_tribles in vec(arb_trible(), 1..3),
        b_tribles in vec(arb_trible(), 1..3),
    ) {
        let id_a = rngid();
        let id_b = rngid();
        let mut set_a = TribleSet::new();
        for t in &a_tribles { set_a.insert(t); }
        let mut set_b = TribleSet::new();
        for t in &b_tribles { set_b.insert(t); }
        let frag_a = Fragment::rooted(*id_a, set_a);
        let frag_b = Fragment::rooted(*id_b, set_b);

        let merged = frag_a + frag_b;
        let exports: Vec<_> = merged.exports().collect();
        if *id_a != *id_b {
            prop_assert_eq!(exports.len(), 2);
        }
        prop_assert!(exports.contains(&*id_a));
        prop_assert!(exports.contains(&*id_b));
    }

    // ── IntersectionConstraint: tighter than either child ──────────────

    #[test]
    fn intersection_no_larger_than_either(
        a in arb_tribleset(10),
        b in arb_tribleset(10)
    ) {
        let inter = a.intersect(&b);
        let inter_results: Vec<_> = find!(
            (e: Inline<_>, a_v: Inline<_>, v: Inline<UnknownInline>),
            inter.pattern(e, a_v, v as Variable<UnknownInline>)
        ).collect();
        let a_results: Vec<_> = find!(
            (e: Inline<_>, a_v: Inline<_>, v: Inline<UnknownInline>),
            a.pattern(e, a_v, v as Variable<UnknownInline>)
        ).collect();
        let b_results: Vec<_> = find!(
            (e: Inline<_>, a_v: Inline<_>, v: Inline<UnknownInline>),
            b.pattern(e, a_v, v as Variable<UnknownInline>)
        ).collect();
        prop_assert!(inter_results.len() <= a_results.len());
        prop_assert!(inter_results.len() <= b_results.len());
    }

    // ── find! fundamentals ──────────────────────────────────────────

    #[test]
    fn find_is_deterministic(set in arb_tribleset(15)) {
        let results1: Vec<_> = find!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        ).collect();
        let results2: Vec<_> = find!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        ).collect();
        prop_assert_eq!(results1, results2,
            "same query on same set should be deterministic");
    }

    #[test]
    fn find_no_duplicates(set in arb_tribleset(15)) {
        let results: Vec<_> = find!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        ).collect();
        let unique: HashSet<_> = results.iter().collect();
        prop_assert_eq!(results.len(), unique.len(),
            "query should not produce duplicate results");
    }

    // ── ConstantConstraint protocol ────────────────────────────────────

    #[test]
    fn constant_constraint_always_proposes_one(val in prop::array::uniform32(any::<u8>())) {
        use triblespace_core::query::constantconstraint::ConstantConstraint;

        let c = ConstantConstraint::new(
            Variable::<UnknownInline>::new(0),
            Inline::<UnknownInline>::new(val),
        );
        let frontier = Frontier::default();

        prop_assert_eq!(c.estimate(0, &Binding::default()), Some(1));

        let mut proposals = ProposalBuffer::new();
        c.propose(0, &frontier, &mut proposals);
        prop_assert_eq!(proposals.len(), 1);
        prop_assert_eq!(proposals[0], val);
    }

    #[test]
    fn constant_constraint_confirms_matching_only(
        constant in prop::array::uniform32(any::<u8>()),
        candidate in prop::array::uniform32(any::<u8>()),
    ) {
        use triblespace_core::query::constantconstraint::ConstantConstraint;

        let c = ConstantConstraint::new(
            Variable::<UnknownInline>::new(0),
            Inline::<UnknownInline>::new(constant),
        );
        let frontier = Frontier::default();

        let mut proposals = ProposalBuffer::new();
        proposals.push(candidate);
        c.confirm(0, &frontier, &mut proposals.region(0));

        if constant == candidate {
            prop_assert_eq!(proposals.count_live(0), 1);
        } else {
            prop_assert_eq!(proposals.count_live(0), 0);
        }
    }

    // ── exists! ────────────────────────────────────────────────────────

    #[test]
    fn exists_consistent_with_find(set in arb_tribleset(10)) {
        let has_results = find!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        ).next().is_some();
        let exists_result = exists!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            set.pattern(e, a, v as Variable<UnknownInline>)
        );
        prop_assert_eq!(has_results, exists_result);
    }

    #[test]
    fn exists_empty_set_is_false(_dummy in 0..1u8) {
        let empty = TribleSet::new();
        let result = exists!(
            (e: Inline<_>, a: Inline<_>, v: Inline<UnknownInline>),
            empty.pattern(e, a, v as Variable<UnknownInline>)
        );
        prop_assert!(!result);
    }

    // ── and! (intersection constraint) ─────────────────────────────────

    #[test]
    fn and_with_hashset_filters(
        labels in vec("[a-z]{1,6}", 3..8),
    ) {
        let mut set = TribleSet::new();
        for label in &labels {
            let e = rngid();
            set += entity! { &e @ test_ns::label: label.as_str() };
        }

        // Pick a subset to allow
        let allowed: HashSet<String> = labels.iter().take(2).cloned().collect();

        let all: Vec<String> = find!(
            label: String,
            pattern!(&set, [{ test_ns::label: ?label }])
        ).collect();

        let filtered: Vec<String> = find!(
            label: String,
            and!(
                allowed.has(label),
                pattern!(&set, [{ test_ns::label: ?label }])
            )
        ).collect();

        // Filtered must be a subset of all
        for label in &filtered {
            prop_assert!(all.contains(label));
        }
        // And only contain allowed values
        for label in &filtered {
            prop_assert!(allowed.contains(label),
                "{:?} not in allowed set", label);
        }
    }

    // ── or! (union constraint) ─────────────────────────────────────────

    #[test]
    fn or_superset_of_both(
        a in arb_tribleset(8),
        b in arb_tribleset(8),
    ) {
        // or! at the raw constraint level: both branches share variables
        let a_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            a.pattern(e, attr, v as Variable<UnknownInline>)
        ).collect();
        let b_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            b.pattern(e, attr, v as Variable<UnknownInline>)
        ).collect();
        let or_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            or!(
                a.pattern(e, attr, v as Variable<UnknownInline>),
                b.pattern(e, attr, v as Variable<UnknownInline>)
            )
        ).collect();

        // or! must contain everything from a
        for triple in &a_results {
            prop_assert!(or_results.contains(triple),
                "or! missing a result from set a");
        }
        // and everything from b
        for triple in &b_results {
            prop_assert!(or_results.contains(triple),
                "or! missing a result from set b");
        }
        // and nothing extra (since union of disjoint sets)
        prop_assert!(or_results.len() <= a_results.len() + b_results.len());
    }


    // ── EqualityConstraint ──────────────────────────────────────────

    // ── Union distributes over queries ───────────────────────────────

    #[test]
    fn query_union_equals_union_of_queries(
        a in arb_tribleset(8),
        b in arb_tribleset(8),
    ) {
        // query(A ∪ B) ⊇ query(A) ∪ query(B)
        // (equality holds for full scans)
        let union = a.clone() + b.clone();

        let mut union_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            union.pattern(e, attr, v as Variable<UnknownInline>)
        ).collect();

        let a_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            a.pattern(e, attr, v as Variable<UnknownInline>)
        ).collect();

        let b_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            b.pattern(e, attr, v as Variable<UnknownInline>)
        ).collect();

        // Merge and deduplicate the individual results
        let mut merged: Vec<_> = a_results.into_iter().chain(b_results).collect();
        merged.sort();
        merged.dedup();
        union_results.sort();

        prop_assert_eq!(union_results, merged,
            "query(A ∪ B) should equal query(A) ∪ query(B)");
    }

    // ── ignore! hides variables without breaking joins ─────────────────

    #[test]
    fn ignore_hides_entity_but_join_works(
        names in vec("[a-z]{1,6}", 2..6),
    ) {
        let hub = rngid();
        let mut set = TribleSet::new();
        set += entity! { &hub @ test_ns::label: "hub" };

        for name in &names {
            let e = rngid();
            set += entity! { &e @ test_ns::label: name.as_str(), test_ns::link: &hub };
        }

        // Without ignore!: get both name and entity
        let full_results: Vec<(Inline<_>, String)> = find!(
            (entity: Inline<_>, name: String),
            pattern!(&set, [
                { ?entity @ test_ns::label: ?name, test_ns::link: _?target },
                { _?target @ test_ns::label: "hub" }
            ])
        ).collect();

        // With temp! (equivalent of ignore for our purposes): get just name
        let name_only: Vec<String> = find!(
            name: String,
            pattern!(&set, [
                { _?entity @ test_ns::label: ?name, test_ns::link: _?target },
                { _?target @ test_ns::label: "hub" }
            ])
        ).collect();

        // Both should find the same names
        let mut full_names: Vec<String> = full_results.into_iter().map(|(_, n)| n).collect();
        let mut names_only_sorted = name_only.clone();
        full_names.sort();
        names_only_sorted.sort();
        prop_assert_eq!(full_names, names_only_sorted,
            "hiding entity variable should not affect join results");

        // And should find all expected names
        for name in &names {
            prop_assert!(name_only.contains(name),
                "missing {:?}", name);
        }
    }

    // ── Intersect query equals and! of queries ─────────────────────────

    #[test]
    fn intersect_query_equals_and_of_queries(
        a in arb_tribleset(8),
        b in arb_tribleset(8),
    ) {
        let intersect = a.intersect(&b);

        let mut intersect_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            intersect.pattern(e, attr, v as Variable<UnknownInline>)
        ).collect();

        // and! of two patterns on different sets = intersection of results
        let mut and_results: Vec<_> = find!(
            (e: Inline<_>, attr: Inline<_>, v: Inline<UnknownInline>),
            and!(
                a.pattern(e, attr, v as Variable<UnknownInline>),
                b.pattern(e, attr, v as Variable<UnknownInline>)
            )
        ).collect();

        intersect_results.sort();
        and_results.sort();
        prop_assert_eq!(intersect_results, and_results,
            "query(A ∩ B) should equal and!(query(A), query(B))");
    }

    // ── SortedSlice ─────────────────────────────────────────────────

    #[test]
    fn sorted_slice_same_as_hashset(
        values in proptest::collection::hash_set("[a-z]{1,6}", 1..15),
    ) {
        use triblespace_core::query::sortedsliceconstraint::SortedSlice;
        use triblespace_core::inline::encodings::shortstring::ShortString;

        let hash: HashSet<String> = values;
        let mut sorted_vals: Vec<String> = hash.iter().cloned().collect();
        sorted_vals.sort();
        let slice = SortedSlice::new(&sorted_vals).unwrap();

        let mut hash_results: Vec<Inline<ShortString>> = find!(
            v: Inline<ShortString>,
            hash.has(v)
        ).collect();

        let mut slice_results: Vec<Inline<ShortString>> = find!(
            v: Inline<ShortString>,
            slice.has(v)
        ).collect();

        hash_results.sort();
        slice_results.sort();
        prop_assert_eq!(hash_results, slice_results,
            "SortedSlice should produce same results as HashSet");
    }

    #[test]
    fn sorted_slice_rejects_unsorted(_dummy in 0..1u8) {
        use triblespace_core::query::sortedsliceconstraint::SortedSlice;
        let data = ["c", "a", "b"];
        prop_assert!(SortedSlice::new(&data).is_err());
    }

    #[test]
    fn sorted_slice_accepts_sorted(len in 0..20usize) {
        use triblespace_core::query::sortedsliceconstraint::SortedSlice;
        let data: Vec<String> = (0..len).map(|i| format!("{i:04}")).collect();
        prop_assert!(SortedSlice::new(&data).is_ok());
    }

    #[test]
    fn mut_slice_has_sorts_and_matches_sorted_slice(
        values in proptest::collection::hash_set("[a-z]{1,6}", 1..15),
    ) {
        // `&mut [T]` (and anything that derefs to one) should sort on
        // `.has()` and produce the same rows as a pre-sorted `SortedSlice`.
        use triblespace_core::query::sortedsliceconstraint::SortedSlice;
        use triblespace_core::inline::encodings::shortstring::ShortString;

        let mut shuffled: Vec<String> = values.into_iter().collect();
        // Scramble deterministically so we have something to sort.
        shuffled.sort_by(|a, b| b.cmp(a));
        let mut sorted = shuffled.clone();
        sorted.sort();

        let presorted = SortedSlice::new(&sorted).unwrap();
        let mut expected: Vec<Inline<ShortString>> = find!(
            v: Inline<ShortString>,
            presorted.has(v)
        ).collect();

        // &mut [T] — direct impl path.
        let mut actual_slice: Vec<Inline<ShortString>> = find!(
            v: Inline<ShortString>,
            (&mut shuffled[..]).has(v)
        ).collect();

        // Reshuffle and exercise &mut Vec<T> — should reach the impl via DerefMut.
        shuffled.sort_by(|a, b| b.cmp(a));
        let mut actual_vec: Vec<Inline<ShortString>> = find!(
            v: Inline<ShortString>,
            (&mut shuffled).has(v)
        ).collect();

        expected.sort();
        actual_slice.sort();
        actual_vec.sort();
        prop_assert_eq!(&expected, &actual_slice,
            "&mut [T]::has should sort in place and produce the same rows as SortedSlice");
        prop_assert_eq!(&expected, &actual_vec,
            "&mut Vec<T>::has should route to &mut [T] via DerefMut and match");
    }

    // ── EqualityConstraint ──────────────────────────────────────────

    #[test]
    fn equality_constraint_propose_mirrors_peer(
        val in prop::array::uniform32(any::<u8>()),
    ) {
        use triblespace_core::query::equalityconstraint::EqualityConstraint;

        let eq = EqualityConstraint::new(0, 1);
        let mut binding = BindingStore::new();
        binding.bind(0, &val);

        // With peer bound, estimate should be 1
        prop_assert_eq!(eq.estimate(1, &binding.view()), Some(1));

        // Propose should yield the peer's value
        let mut proposals = ProposalBuffer::new();
        eq.propose(1, &binding.frontier(), &mut proposals);
        prop_assert_eq!(proposals.len(), 1);
        prop_assert_eq!(proposals[0], val);
    }

    #[test]
    fn equality_constraint_confirm_filters(
        peer_val in prop::array::uniform32(any::<u8>()),
        other_val in prop::array::uniform32(any::<u8>()),
    ) {
        use triblespace_core::query::equalityconstraint::EqualityConstraint;

        let eq = EqualityConstraint::new(0, 1);
        let mut binding = BindingStore::new();
        binding.bind(0, &peer_val);

        let mut proposals = ProposalBuffer::new();
        proposals.push(peer_val);
        proposals.push(other_val);
        eq.confirm(1, &binding.frontier(), &mut proposals.region(0));

        if peer_val == other_val {
            prop_assert_eq!(proposals.count_live(0), 2); // both match
        } else {
            prop_assert_eq!(proposals.count_live(0), 1);
            prop_assert!(proposals.is_live(0));
        }
    }

    #[test]
    fn equality_constraint_satisfied_both_bound(
        a_val in prop::array::uniform32(any::<u8>()),
        b_val in prop::array::uniform32(any::<u8>()),
    ) {
        use triblespace_core::query::equalityconstraint::EqualityConstraint;

        let eq = EqualityConstraint::new(0, 1);
        let mut binding = BindingStore::new();
        binding.bind(0, &a_val);
        binding.bind(1, &b_val);

        prop_assert_eq!(eq.satisfied(&binding.view()), a_val == b_val);
    }

    #[test]
    fn equality_constraint_satisfied_partial(_dummy in 0..1u8) {
        use triblespace_core::query::equalityconstraint::EqualityConstraint;

        let eq = EqualityConstraint::new(0, 1);

        // Neither bound — optimistically true
        prop_assert!(eq.satisfied(&Binding::default()));

        // One bound — optimistically true
        let mut binding = BindingStore::new();
        binding.bind(0, &[42; 32]);
        prop_assert!(eq.satisfied(&binding.view()));
    }

    #[test]
    fn equality_constraint_symmetric(
        val in prop::array::uniform32(any::<u8>()),
    ) {
        use triblespace_core::query::equalityconstraint::EqualityConstraint;

        let eq = EqualityConstraint::new(0, 1);

        // Bind a=val, propose for b → val
        let mut binding_a = BindingStore::new();
        binding_a.bind(0, &val);
        let mut props_b = ProposalBuffer::new();
        eq.propose(1, &binding_a.frontier(), &mut props_b);

        // Bind b=val, propose for a → val
        let mut binding_b = BindingStore::new();
        binding_b.bind(1, &val);
        let mut props_a = ProposalBuffer::new();
        eq.propose(0, &binding_b.frontier(), &mut props_a);

        prop_assert_eq!(&props_a[..], &props_b[..]);
    }

    #[test]
    fn variableset_union_commutative(a_bits: u128, b_bits: u128) {
        let a = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(a_bits) };
        let b = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(b_bits) };
        prop_assert_eq!(a.union(b), b.union(a));
    }

    #[test]
    fn variableset_intersect_commutative(a_bits: u128, b_bits: u128) {
        let a = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(a_bits) };
        let b = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(b_bits) };
        prop_assert_eq!(a.intersect(b), b.intersect(a));
    }

    #[test]
    fn variableset_demorgan_union(a_bits: u128, b_bits: u128) {
        // ¬(A ∪ B) = ¬A ∩ ¬B
        let a = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(a_bits) };
        let b = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(b_bits) };
        prop_assert_eq!(
            a.union(b).complement(),
            a.complement().intersect(b.complement())
        );
    }

    #[test]
    fn variableset_demorgan_intersect(a_bits: u128, b_bits: u128) {
        // ¬(A ∩ B) = ¬A ∪ ¬B
        let a = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(a_bits) };
        let b = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(b_bits) };
        prop_assert_eq!(
            a.intersect(b).complement(),
            a.complement().union(b.complement())
        );
    }

    #[test]
    fn variableset_subtract_is_intersect_complement(a_bits: u128, b_bits: u128) {
        // A \ B = A ∩ ¬B
        let a = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(a_bits) };
        let b = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(b_bits) };
        prop_assert_eq!(
            a.subtract(b),
            a.intersect(b.complement())
        );
    }

    #[test]
    fn variableset_count_matches_drain(bits: u128) {
        let vs = unsafe { std::mem::transmute::<u128, triblespace_core::query::VariableSet>(bits) };
        let count = vs.count();
        let mut copy = vs;
        let mut drained = 0;
        while copy.drain_next_ascending().is_some() {
            drained += 1;
        }
        prop_assert_eq!(count, drained);
    }

    // ── Binding set/get/unset ──────────────────────────────────────────

    #[test]
    fn binding_set_get_roundtrip(idx in 0..128usize, value: [u8; 32]) {
        let mut binding = BindingStore::new();
        binding.bind(idx, &value);
        let got = binding.view().get(idx);
        prop_assert_eq!(got, Some(&value));
    }

    #[test]
    fn binding_unset_removes(idx in 0..128usize, value: [u8; 32]) {
        let mut binding = BindingStore::new();
        binding.bind(idx, &value);
        binding.unset(idx);
        prop_assert_eq!(binding.view().get(idx), None);
    }

    #[test]
    fn binding_independent_variables(
        i in 0..64usize,
        j in 64..128usize,
        vi: [u8; 32],
        vj: [u8; 32],
    ) {
        let mut binding = BindingStore::new();
        binding.bind(i, &vi);
        binding.bind(j, &vj);
        prop_assert_eq!(binding.view().get(i), Some(&vi));
        prop_assert_eq!(binding.view().get(j), Some(&vj));
        binding.unset(i);
        prop_assert_eq!(binding.view().get(i), None);
        prop_assert_eq!(binding.view().get(j), Some(&vj)); // j unaffected
    }

}

/// Test-only source that records the *width* of every frontier it is
/// proposed over — the thing the batched protocol exists to make large.
struct WidthObserver {
    variable: usize,
    values: Vec<[u8; 32]>,
    widths: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
}

impl<'a> Constraint<'a> for WidthObserver {
    fn variables(&self) -> triblespace_core::query::VariableSet {
        let mut set = triblespace_core::query::VariableSet::new_empty();
        set.set(self.variable);
        set
    }

    fn estimate(&self, variable: usize, _binding: &Binding) -> Option<usize> {
        (variable == self.variable).then_some(self.values.len())
    }

    fn propose(
        &self,
        variable: usize,
        frontier: &triblespace_core::query::Frontier<'_>,
        proposals: &mut ProposalBuffer,
    ) {
        if variable != self.variable {
            return;
        }
        self.widths.lock().unwrap().push(frontier.len());
        for row in 0..frontier.len() {
            proposals.open(row as u32);
            proposals.extend_from_slice(&self.values);
        }
    }

    fn confirm(
        &self,
        variable: usize,
        _frontier: &triblespace_core::query::Frontier<'_>,
        cands: &mut Candidates<'_>,
    ) {
        if variable == self.variable {
            cands.retain(|v| self.values.binary_search(v).is_ok());
        }
    }
}

/// A two-variable query over `values`, run at the given frontier width.
fn cross_product_rows(
    values: &[[u8; 32]],
    width: usize,
    widths: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
) -> Vec<([u8; 32], [u8; 32])> {
    let outer = WidthObserver {
        variable: 0,
        values: values.to_vec(),
        widths: std::sync::Arc::clone(&widths),
    };
    let inner = WidthObserver {
        variable: 1,
        values: values.to_vec(),
        widths,
    };
    triblespace_core::query::Query::new(
        triblespace_core::query::intersectionconstraint::IntersectionConstraint::new(vec![
            Box::new(outer) as Box<dyn Constraint + Send + Sync>,
            Box::new(inner),
        ]),
        |binding: &Binding| Some((*binding.get(0)?, *binding.get(1)?)),
    )
    .with_frontier_width(width)
    .collect()
}

proptest! {
    /// The batch width is an execution choice, not a semantic one: a
    /// frontier of one (the pre-batching shape) and a wide frontier must
    /// produce the very same bag of rows.
    #[test]
    fn frontier_width_preserves_the_bag(
        seed_values in vec(prop::array::uniform32(any::<u8>()), 1..24),
    ) {
        let mut values: Vec<[u8; 32]> = seed_values;
        values.sort_unstable();
        values.dedup();

        let narrow_widths = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let wide_widths = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut narrow = cross_product_rows(&values, 1, std::sync::Arc::clone(&narrow_widths));
        let mut wide = cross_product_rows(&values, 4096, std::sync::Arc::clone(&wide_widths));

        prop_assert_eq!(narrow.len(), values.len() * values.len());
        narrow.sort_unstable();
        wide.sort_unstable();
        prop_assert_eq!(narrow, wide);

        // ... and the width actually reached the second level, which is the
        // whole point: with a frontier of one, every deeper propose sees a
        // single parent binding.
        prop_assert!(narrow_widths.lock().unwrap().iter().all(|&w| w == 1));
        if values.len() > 1 {
            prop_assert!(wide_widths.lock().unwrap().iter().any(|&w| w > 1));
        }
    }
}

#[test]
fn deep_levels_see_a_wide_frontier() {
    // Two levels of 40 values each: at width 1 the inner level is proposed
    // over one binding 40 times; batched, it is proposed over all 40 at
    // once.
    let values: Vec<[u8; 32]> = (0..40u32)
        .map(|i| {
            let mut v = [0u8; 32];
            v[28..32].copy_from_slice(&i.to_be_bytes());
            v
        })
        .collect();

    let widths = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let rows = cross_product_rows(&values, 4096, std::sync::Arc::clone(&widths));
    assert_eq!(rows.len(), 40 * 40);

    let widths = widths.lock().unwrap();
    assert_eq!(widths.as_slice(), &[1, 40], "root then one 40-wide expansion");
}

#[test]
fn frontier_stats_report_an_unfragmented_batch() {
    let values: Vec<[u8; 32]> = (0..8u32)
        .map(|i| {
            let mut v = [0u8; 32];
            v[28..32].copy_from_slice(&i.to_be_bytes());
            v
        })
        .collect();

    let widths = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let outer = WidthObserver {
        variable: 0,
        values: values.clone(),
        widths: std::sync::Arc::clone(&widths),
    };
    let inner = WidthObserver {
        variable: 1,
        values: values.clone(),
        widths,
    };
    let query = triblespace_core::query::Query::new(
        triblespace_core::query::intersectionconstraint::IntersectionConstraint::new(vec![
            Box::new(outer) as Box<dyn Constraint + Send + Sync>,
            Box::new(inner),
        ]),
        |binding: &Binding| Some((*binding.get(0)?, *binding.get(1)?)),
    );
    let stats = query.stats();
    let rows: Vec<_> = query.collect();

    assert_eq!(rows.len(), 64);
    // Root expansion (1 row) plus one 8-row expansion; both agreed on the
    // variable to bind, so the batch never fragmented.
    assert_eq!(stats.expansions(), 2);
    assert_eq!(stats.rows(), 1 + 8);
    assert_eq!(stats.variable_groups(), 2);
    assert_eq!(stats.mean_variable_groups(), 1.0);
}

/// A source whose estimate for its own variable depends on the *value*
/// bound to variable 0 — so the rows of one frontier disagree about which
/// variable to bind next, and the batch has to fragment.
struct Skewed {
    variable: usize,
    values: Vec<[u8; 32]>,
    /// The low-byte parity of variable 0's value that makes this source
    /// look cheap.
    cheap_parity: u8,
}

impl<'a> Constraint<'a> for Skewed {
    fn variables(&self) -> triblespace_core::query::VariableSet {
        let mut set = triblespace_core::query::VariableSet::new_empty();
        set.set(self.variable);
        set
    }

    fn estimate(&self, variable: usize, binding: &Binding) -> Option<usize> {
        if variable != self.variable {
            return None;
        }
        Some(match binding.get(0) {
            Some(anchor) if anchor[31] % 2 == self.cheap_parity => 1,
            // Expensive both when the anchor has the wrong parity and
            // before it is bound at all, so the root binds variable 0
            // first and the disagreement appears one level down.
            _ => 4096,
        })
    }

    /// Binding variable 0 is what moves this source's estimate, and the
    /// engine only refreshes what `influence` names.
    fn influence(&self, variable: usize) -> triblespace_core::query::VariableSet {
        if variable == 0 {
            let mut set = triblespace_core::query::VariableSet::new_empty();
            set.set(self.variable);
            set
        } else {
            let mut set = self.variables();
            set.unset(variable);
            set
        }
    }

    fn propose(
        &self,
        variable: usize,
        frontier: &triblespace_core::query::Frontier<'_>,
        proposals: &mut ProposalBuffer,
    ) {
        if variable != self.variable {
            return;
        }
        for row in 0..frontier.len() {
            proposals.open(row as u32);
            proposals.extend_from_slice(&self.values);
        }
    }

    fn confirm(
        &self,
        variable: usize,
        _frontier: &triblespace_core::query::Frontier<'_>,
        cands: &mut Candidates<'_>,
    ) {
        if variable == self.variable {
            cands.retain(|v| self.values.binary_search(v).is_ok());
        }
    }
}

fn value(tag: u8, i: u32) -> [u8; 32] {
    let mut v = [0u8; 32];
    v[0] = tag;
    v[27..31].copy_from_slice(&i.to_be_bytes());
    v[31] = i as u8;
    v
}

fn skewed_rows(width: usize) -> (Vec<([u8; 32], [u8; 32], [u8; 32])>, (u64, u64)) {
    // Four anchors, alternating parity, so the frontier at depth 1 splits
    // evenly between "bind ?b next" and "bind ?c next".
    let anchors: Vec<[u8; 32]> = (0..4u32).map(|i| value(0xA0, i)).collect();
    let bs: Vec<[u8; 32]> = (0..3u32).map(|i| value(0xB0, i)).collect();
    let cs: Vec<[u8; 32]> = (0..3u32).map(|i| value(0xC0, i)).collect();

    let root = WidthObserver {
        variable: 0,
        values: anchors,
        widths: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
    };
    let query = triblespace_core::query::Query::new(
        triblespace_core::query::intersectionconstraint::IntersectionConstraint::new(vec![
            Box::new(root) as Box<dyn Constraint + Send + Sync>,
            Box::new(Skewed {
                variable: 1,
                values: bs,
                cheap_parity: 0,
            }),
            Box::new(Skewed {
                variable: 2,
                values: cs,
                cheap_parity: 1,
            }),
        ]),
        |binding: &Binding| Some((*binding.get(0)?, *binding.get(1)?, *binding.get(2)?)),
    )
    .with_frontier_width(width);
    let stats = query.stats();
    let mut rows: Vec<_> = query.collect();
    rows.sort_unstable();
    (rows, (stats.expansions(), stats.variable_groups()))
}

#[test]
fn a_fragmented_frontier_keeps_every_row() {
    let (narrow, narrow_stats) = skewed_rows(1);
    let (wide, wide_stats) = skewed_rows(4096);

    // 4 anchors x 3 x 3.
    assert_eq!(narrow.len(), 36);
    assert_eq!(narrow, wide, "batch width must not change the bag");

    // A frontier of one can never fragment: one row, one group.
    assert_eq!(narrow_stats.0, narrow_stats.1);
    // The wide run really did hit the split path — the depth-1 frontier of
    // four rows disagreed, so it cost more groups than expansions.
    assert!(
        wide_stats.1 > wide_stats.0,
        "expected a fragmented expansion, saw {} groups over {} expansions",
        wide_stats.1,
        wide_stats.0
    );
}
