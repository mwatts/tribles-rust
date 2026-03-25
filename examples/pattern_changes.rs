use crate::entity;
use crate::pattern_changes;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use triblespace::core::repo::memoryrepo::MemoryRepo;
use triblespace::core::repo::Repository;
use triblespace::prelude::*;

pub mod literature {
    use triblespace::prelude::*;

    attributes! {
        "8F180883F9FD5F787E9E0AF0DF5866B9" as author: valueschemas::GenId;
        "0DBB530B37B966D137C50B943700EDB2" as firstname: valueschemas::ShortString;
        "6BAA463FD4EAF45F6A103DB9433E4545" as lastname: valueschemas::ShortString;
        "A74AA63539354CDA47F387A4C3A8D54C" as title: valueschemas::ShortString;
    }
}

fn main() {
    // ANCHOR: pattern_changes_example
    let storage = MemoryRepo::default();
    let mut repo = Repository::new(storage, SigningKey::generate(&mut OsRng), TribleSet::new()).unwrap();
    let branch_id = repo.create_branch("main", None).expect("branch");

    // ── commit initial data ──────────────────────────────────────────
    let shakespeare = ufoid();
    let hamlet = ufoid();
    let mut ws = repo.pull(*branch_id).expect("pull");
    let mut initial = TribleSet::new();
    initial += entity! { &shakespeare @ literature::firstname: "William", literature::lastname: "Shakespeare" };
    initial += entity! { &hamlet @ literature::title: "Hamlet", literature::author: &shakespeare };
    ws.commit(initial, "initial");
    repo.push(&mut ws).unwrap();

    // ── first checkout: load everything ──────────────────────────────
    // `changed` and `full` start as the same full checkout.
    let mut changed = repo.pull(*branch_id).expect("pull").checkout(..).expect("checkout");
    let mut full = changed.facts().clone();

    // On the first iteration, everything is "new".
    let all_titles: Vec<String> = find!(
        title: String,
        pattern_changes!(&full, &changed, [
            { _?book @ literature::title: ?title }
        ])
    )
    .collect();
    assert_eq!(all_titles, vec!["Hamlet".to_string()]);

    // ── simulate an external update ──────────────────────────────────
    let macbeth = ufoid();
    let mut ws = repo.pull(*branch_id).expect("pull");
    ws.commit(
        entity! { &macbeth @ literature::title: "Macbeth", literature::author: &shakespeare },
        "add Macbeth",
    );
    repo.push(&mut ws).unwrap();

    // ── incremental update ───────────────────────────────────────────
    // Pull fresh, checkout only commits since our last head.
    changed = repo.pull(*branch_id).expect("pull").checkout(changed.head()..).expect("delta");
    full += &changed;

    // Only Macbeth shows up — Hamlet was in the previous checkout.
    let new_titles: Vec<String> = find!(
        title: String,
        pattern_changes!(&full, &changed, [
            { _?book @ literature::title: ?title }
        ])
    )
    .collect();
    assert_eq!(new_titles, vec!["Macbeth".to_string()]);
    println!("New titles: {new_titles:?}");
    // ANCHOR_END: pattern_changes_example
}
