use tribles::prelude::valueschemas::*;
use tribles::prelude::*;

use fake::faker::name::raw::*;
use fake::locales::*;
use fake::Fake;

NS! {
    pub namespace knights {
        "39E2D06DBCD9CB96DE5BC46F362CFF31" as loves: GenId;
        "7D4F339CC4AE0BBA2765F34BE1D108EF" as name: ShortString;
        "3E0C58AC884072EA6429BB00A1BA1DA4" as title: ShortString;
    }
}

fn main() {
    let mut kb = TribleSet::new();
    (0..1000000).for_each(|_| {
        let lover_a = ufoid();
        let lover_b = ufoid();

        kb.union(knights::entity!({
            name: Name(EN).fake::<String>(),
            loves: &lover_b
        }));
        kb.union(knights::entity!({
            name: Name(EN).fake::<String>(),
            loves: &lover_a
        }));
    });

    let mut data_kb = TribleSet::new();

    let romeo = ufoid();
    let juliet = ufoid();

    data_kb.union(knights::entity!(&juliet, {
        name: "Juliet",
        loves: &romeo
    }));
    data_kb.union(knights::entity!(&romeo, {
        name: "Romeo",
        loves: &juliet
    }));

    kb.union(data_kb);

    loop {
        for _r in find!(
            ctx,
            (juliet: Value<_>, name: Value<_>),
            knights::pattern!(ctx, &kb, [
            {name: ("Romeo"),
            loves: juliet},
            {juliet @
                name: name
            }])
        ) {
            coz::progress!();
        }
    }
}
