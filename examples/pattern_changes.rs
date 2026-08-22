use crate::entity;
use crate::pattern_changes;
use triblespace::prelude::*;

pub mod literature {
    use triblespace::prelude::*;

    attributes! {
        "8F180883F9FD5F787E9E0AF0DF5866B9" unsafe as author: inlineencodings::GenId;
        "0DBB530B37B966D137C50B943700EDB2" unsafe as firstname: inlineencodings::ShortString;
        "6BAA463FD4EAF45F6A103DB9433E4545" unsafe as lastname: inlineencodings::ShortString;
        "A74AA63539354CDA47F387A4C3A8D54C" unsafe as title: inlineencodings::ShortString;
    }
}

fn main() {
    // ANCHOR: pattern_changes_example
    // `pattern_changes!` needs only the complete known set and the newly
    // observed delta. How those sets arrived is deliberately orthogonal.
    let mut initial = entity! { literature::firstname: "Frank", literature::lastname: "Herbert" };
    let herbert = initial.root().expect("intrinsic author identity");
    initial += entity! { literature::title: "Dune", literature::author: &herbert };

    // On first observation, the whole set is new.
    let mut changed = initial.into_facts();
    let mut full = changed.clone();

    let all_titles: Vec<String> = find!(
        title: String,
        pattern_changes!(&full, &changed, [
            { _?author @ literature::firstname: "Frank" },
            { _?book @ literature::author: _?author, literature::title: ?title }
        ])
    )
    .collect();
    assert_eq!(all_titles, vec!["Dune".to_string()]);

    // A later collection observation contributes one monotonic delta.
    changed = entity! {
        literature::title: "Dune Messiah",
        literature::author: &herbert,
    }
    .into_facts();
    full += changed.clone();

    // Only Dune Messiah shows up — Dune was in the previous observation.
    let new_titles: Vec<String> = find!(
        title: String,
        pattern_changes!(&full, &changed, [
            { _?author @ literature::firstname: "Frank" },
            { _?book @ literature::author: _?author, literature::title: ?title }
        ])
    )
    .collect();
    assert_eq!(new_titles, vec!["Dune Messiah".to_string()]);
    println!("New titles: {new_titles:?}");
    // ANCHOR_END: pattern_changes_example
}
