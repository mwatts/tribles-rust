use proptest::prelude::*;
use proptest::collection::vec;
use triblespace_core::id::rngid;
use triblespace_core::prelude::*;

mod test_ns {
    use triblespace_core::prelude::*;
    attributes! {
        "CC00000000000000CC00000000000001" as pub name: valueschemas::ShortString;
        "CC00000000000000CC00000000000002" as pub link: valueschemas::GenId;
        "CC00000000000000CC00000000000003" as pub count: valueschemas::U256BE;
    }
}

proptest! {
    // ── entity! query round-trip ───────────────────────────────────────

    #[test]
    fn entity_name_queryable(name in "[a-z]{1,8}") {
        let e = rngid();
        let set: TribleSet = entity! { &e @ test_ns::name: name.as_str() }.into();

        let results: Vec<String> = find!(
            n: String,
            pattern!(&set, [{ test_ns::name: ?n }])
        ).collect();

        prop_assert_eq!(results.len(), 1);
        prop_assert_eq!(&results[0], &name);
    }

    #[test]
    fn entity_link_queryable(_dummy in 0..1u8) {
        let src = rngid();
        let dst = rngid();
        let set: TribleSet = entity! { &src @ test_ns::link: &dst }.into();

        let results: Vec<Value<_>> = find!(
            target: Value<_>,
            pattern!(&set, [{ test_ns::link: ?target }])
        ).collect();

        prop_assert_eq!(results.len(), 1);
        prop_assert_eq!(results[0], (&dst).to_value());
    }

    #[test]
    fn entity_multiple_attrs_queryable(name in "[a-z]{1,8}") {
        let e = rngid();
        let other = rngid();
        let set: TribleSet = entity! { &e @
            test_ns::name: name.as_str(),
            test_ns::link: &other
        }.into();

        // Query name
        let names: Vec<String> = find!(
            n: String,
            pattern!(&set, [{ test_ns::name: ?n }])
        ).collect();
        prop_assert_eq!(names.len(), 1);
        prop_assert_eq!(&names[0], &name);

        // Query link
        let links: Vec<Value<_>> = find!(
            target: Value<_>,
            pattern!(&set, [{ test_ns::link: ?target }])
        ).collect();
        prop_assert_eq!(links.len(), 1);
        prop_assert_eq!(links[0], (&other).to_value());
    }

    #[test]
    fn entity_union_preserves_all(
        names in vec("[a-z]{1,6}", 2..8),
    ) {
        let mut set = TribleSet::new();
        for name in &names {
            let e = rngid();
            set += entity! { &e @ test_ns::name: name.as_str() };
        }

        let results: Vec<String> = find!(
            n: String,
            pattern!(&set, [{ test_ns::name: ?n }])
        ).collect();

        // Every name we inserted should be queryable
        for name in &names {
            prop_assert!(results.contains(name),
                "missing name {:?}", name);
        }
        // Count should match (each entity has exactly one name)
        prop_assert_eq!(results.len(), names.len());
    }

    // ── Multi-entity join ──────────────────────────────────────────────

    #[test]
    fn join_finds_linked_names(
        src_name in "[a-z]{1,6}",
        dst_name in "[a-z]{1,6}",
    ) {
        let src = rngid();
        let dst = rngid();
        let mut set = TribleSet::new();
        set += entity! { &src @ test_ns::name: src_name.as_str(), test_ns::link: &dst };
        set += entity! { &dst @ test_ns::name: dst_name.as_str() };

        // Join: find names of entities linked from src
        let results: Vec<String> = find!(
            name: String,
            pattern!(&set, [
                { _?source @ test_ns::name: src_name.as_str(), test_ns::link: _?target },
                { _?target @ test_ns::name: ?name }
            ])
        ).collect();

        prop_assert_eq!(results.len(), 1);
        prop_assert_eq!(&results[0], &dst_name);
    }

    // ── Fragment root ──────────────────────────────────────────────────

    #[test]
    fn entity_fragment_has_root(name in "[a-z]{1,8}") {
        let e = rngid();
        let frag = entity! { &e @ test_ns::name: name.as_str() };
        prop_assert_eq!(frag.root(), Some(*e));
    }

    #[test]
    fn entity_fragment_facts_match_tribleset(name in "[a-z]{1,8}") {
        let e = rngid();
        let frag = entity! { &e @ test_ns::name: name.as_str() };
        let set: TribleSet = frag.clone().into();
        prop_assert_eq!(frag.facts(), &set);
    }

    // ── temp! creates variables scoped to the expression ───────────────

    #[test]
    fn temp_enforces_join_without_projecting(
        src_name in "[a-z]{1,6}",
        dst_name in "[a-z]{1,6}",
    ) {
        let src = rngid();
        let dst = rngid();
        let decoy = rngid();

        let mut set = TribleSet::new();
        set += entity! { &src @ test_ns::name: src_name.as_str(), test_ns::link: &dst };
        set += entity! { &dst @ test_ns::name: dst_name.as_str() };
        // Decoy: has a name but isn't linked from src
        set += entity! { &decoy @ test_ns::name: "decoy" };

        // temp! creates a join variable that doesn't leak into results
        let results: Vec<String> = find!(
            name: String,
            temp!((target), pattern!(&set, [
                { _?source @ test_ns::name: src_name.as_str(), test_ns::link: ?target },
                { ?target @ test_ns::name: ?name }
            ]))
        ).collect();

        // Should find dst_name but not "decoy"
        prop_assert_eq!(results.len(), 1);
        prop_assert_eq!(&results[0], &dst_name);
    }

    // ── _? local variables enforce equality ────────────────────────────

    #[test]
    fn self_referencing_entity_via_desugaring(
        names in vec("[a-z]{1,6}", 2..6),
    ) {
        let mut set = TribleSet::new();
        let mut self_linker = None;

        for (i, name) in names.iter().enumerate() {
            let e = rngid();
            set += entity! { &e @ test_ns::name: name.as_str() };
            if i == 0 {
                set += entity! { &e @ test_ns::link: &e };
                self_linker = Some(name.clone());
            } else {
                let other = rngid();
                set += entity! { &e @ test_ns::link: &other };
            }
        }

        // _?e in both entity and value positions — the macro now
        // desugars this to a fresh variable + EqualityConstraint.
        let results: Vec<String> = find!(
            name: String,
            pattern!(&set, [
                { _?e @ test_ns::name: ?name, test_ns::link: _?e }
            ])
        ).collect();

        prop_assert_eq!(results.len(), 1,
            "expected 1 self-linker, got {:?}", results);
        prop_assert_eq!(&results[0], self_linker.as_ref().unwrap());
    }

    #[test]
    fn local_var_enforces_join_across_entities(
        names in vec("[a-z]{1,6}", 2..6),
    ) {
        let mut set = TribleSet::new();
        let hub = rngid();
        set += entity! { &hub @ test_ns::name: "hub" };

        // Create entities that link to hub
        for name in &names {
            let e = rngid();
            set += entity! { &e @ test_ns::name: name.as_str(), test_ns::link: &hub };
        }
        // Create a decoy that links to something else
        let decoy = rngid();
        let other = rngid();
        set += entity! { &decoy @ test_ns::name: "decoy", test_ns::link: &other };

        // _?target joins across two entity clauses
        let results: Vec<String> = find!(
            name: String,
            pattern!(&set, [
                { _?source @ test_ns::name: ?name, test_ns::link: _?target },
                { _?target @ test_ns::name: "hub" }
            ])
        ).collect();

        // Should find all names that link to hub, but not "decoy"
        for name in &names {
            prop_assert!(results.contains(name),
                "missing {:?} from join results", name);
        }
        prop_assert!(!results.contains(&"decoy".to_string()),
            "decoy should not appear in results");
    }
}
