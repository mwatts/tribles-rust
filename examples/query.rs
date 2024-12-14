use fake::faker::lorem::en::Sentence;
use fake::faker::lorem::en::Word;
use tribles::prelude::blobschemas::*;
use tribles::prelude::valueschemas::*;
use tribles::prelude::*;

use fake::faker::name::raw::*;
use fake::locales::*;
use fake::Fake;

NS! {
    pub namespace literature {
        "8F180883F9FD5F787E9E0AF0DF5866B9" as author: GenId;
        "0DBB530B37B966D137C50B943700EDB2" as firstname: ShortString;
        "6BAA463FD4EAF45F6A103DB9433E4545" as lastname: ShortString;
        "A74AA63539354CDA47F387A4C3A8D54C" as title: ShortString;
        "76AE5012877E09FF0EE0868FE9AA0343" as height: FR256;
        "6A03BAF6CFB822F04DA164ADAAEB53F6" as quote: Handle<Blake3, LongString>;
    }
}

fn main() {
    let mut kb = TribleSet::new();
    let mut blobs = BlobSet::new();
    (0..1000000).for_each(|_| {
        let author = fucid();
        let book = fucid();
        kb += literature::entity!(&author, {
            firstname: FirstName(EN).fake::<String>(),
            lastname: LastName(EN).fake::<String>(),
        });
        kb += literature::entity!(&book, {
            author: &author,
            title: Word().fake::<String>(),
            quote: blobs.insert(Sentence(5..25).fake::<String>())
        });
    });

    let _result: Vec<_> = find!(
    (author: Value<_>, title: Value<_>, quote: Value<_>),
    literature::pattern!(&kb, [
    {author @
        firstname: ("Frank"),
        lastname: ("Herbert")},
    { author: author,
        title: title,
        quote: quote
    }]))
    .collect();
}
