use proptest::prelude::*;
use proptest::collection::vec;
use triblespace_core::id::rngid;
use triblespace_core::prelude::*;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::Repository;
use triblespace_core::value::schemas::hash::Blake3;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

mod test_ns {
    use triblespace_core::prelude::*;
    attributes! {
        "DD00000000000000DD00000000000001" as pub label: valueschemas::ShortString;
    }
}

proptest! {
    // ── Workspace commit + checkout round-trip ─────────────────────────

    #[test]
    fn commit_checkout_roundtrip(
        labels in vec("[a-z]{1,8}", 1..10),
    ) {
        let storage = MemoryRepo::default();
        let mut repo = Repository::new(
            storage,
            SigningKey::generate(&mut OsRng),
            TribleSet::new(),
        ).unwrap();
        let branch_id = repo.create_branch("test", None).expect("create branch");
        let mut ws = repo.pull(*branch_id).expect("pull");

        // Commit data
        let mut data = TribleSet::new();
        for label in &labels {
            let e = rngid();
            data += entity! { &e @ test_ns::label: label.as_str() };
        }
        ws.commit(data.clone(), "test commit");

        // Checkout and verify
        let checkout = ws.checkout(..).expect("checkout");
        prop_assert_eq!(checkout.facts().len(), data.len(),
            "checkout should contain all committed tribles");

        // Query should return all labels
        let mut found: Vec<String> = find!(
            label: String,
            pattern!(&checkout, [{ test_ns::label: ?label }])
        ).collect();
        let mut expected: Vec<String> = labels.clone();
        found.sort();
        expected.sort();
        prop_assert_eq!(found, expected);
    }

    #[test]
    fn multiple_commits_accumulate(
        batch1 in vec("[a-z]{1,6}", 1..5),
        batch2 in vec("[a-z]{1,6}", 1..5),
    ) {
        let storage = MemoryRepo::default();
        let mut repo = Repository::new(
            storage,
            SigningKey::generate(&mut OsRng),
            TribleSet::new(),
        ).unwrap();
        let branch_id = repo.create_branch("test", None).expect("create branch");
        let mut ws = repo.pull(*branch_id).expect("pull");

        // First commit
        let mut data1 = TribleSet::new();
        for label in &batch1 {
            let e = rngid();
            data1 += entity! { &e @ test_ns::label: label.as_str() };
        }
        ws.commit(data1.clone(), "batch 1");

        // Second commit
        let mut data2 = TribleSet::new();
        for label in &batch2 {
            let e = rngid();
            data2 += entity! { &e @ test_ns::label: label.as_str() };
        }
        ws.commit(data2.clone(), "batch 2");

        // Full checkout should contain both batches
        let checkout = ws.checkout(..).expect("checkout");
        let expected_len = data1.len() + data2.len();
        prop_assert_eq!(checkout.facts().len(), expected_len);

        // All labels from both batches should be queryable
        let found: Vec<String> = find!(
            label: String,
            pattern!(&checkout, [{ test_ns::label: ?label }])
        ).collect();
        for label in batch1.iter().chain(batch2.iter()) {
            prop_assert!(found.contains(label),
                "missing {:?}", label);
        }
    }

    #[test]
    fn push_then_pull_preserves_data(
        labels in vec("[a-z]{1,8}", 1..8),
    ) {
        let storage = MemoryRepo::default();
        let mut repo = Repository::new(
            storage,
            SigningKey::generate(&mut OsRng),
            TribleSet::new(),
        ).unwrap();
        let branch_id = repo.create_branch("test", None).expect("create branch");
        let mut ws = repo.pull(*branch_id).expect("pull");

        let mut data = TribleSet::new();
        for label in &labels {
            let e = rngid();
            data += entity! { &e @ test_ns::label: label.as_str() };
        }
        ws.commit(data, "commit");
        repo.push(&mut ws).expect("push");

        // Fresh pull should see the same data
        let mut ws2 = repo.pull(*branch_id).expect("pull2");
        let checkout = ws2.checkout(..).expect("checkout");

        let mut found: Vec<String> = find!(
            label: String,
            pattern!(&checkout, [{ test_ns::label: ?label }])
        ).collect();
        let mut expected: Vec<String> = labels;
        found.sort();
        expected.sort();
        prop_assert_eq!(found, expected,
            "push then pull should preserve all data");
    }

    #[test]
    fn checkout_commits_tracks_seen(
        labels in vec("[a-z]{1,8}", 1..5),
    ) {
        let storage = MemoryRepo::default();
        let mut repo = Repository::new(
            storage,
            SigningKey::generate(&mut OsRng),
            TribleSet::new(),
        ).unwrap();
        let branch_id = repo.create_branch("test", None).expect("create branch");
        let mut ws = repo.pull(*branch_id).expect("pull");

        let mut data = TribleSet::new();
        for label in &labels {
            let e = rngid();
            data += entity! { &e @ test_ns::label: label.as_str() };
        }
        ws.commit(data, "commit");

        let checkout = ws.checkout(..).expect("checkout");
        // commits() should be non-empty after a checkout with data
        prop_assert!(!checkout.commits().is_empty(),
            "checkout should track the commit");
    }
}
