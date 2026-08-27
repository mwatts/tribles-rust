#![doc = include_str!("../README.md")]
extern crate self as triblespace;

pub use triblespace_core::arrayvec;

pub use triblespace_core as core;

/// Distributed sync via iroh: [`net::peer::Peer<S>`] wraps any store with
/// authenticated PATCH anti-entropy, durable wants, and DHT-located
/// content-addressed blob fetching. Gated behind the `net` feature.
#[cfg(feature = "net")]
pub use triblespace_net as net;

/// Content-addressed BM25 + HNSW search indexes on triblespace piles.
/// See [`search::bm25`], [`search::hnsw`], [`search::constraint`] for the
/// query-engine integration. Gated behind the `search` feature.
#[cfg(feature = "search")]
pub use triblespace_search as search;

/// Opt-in WGPU acceleration for succinct-archive construction and batched
/// query confirmation. Gated behind the `gpu` feature, which also enables the
/// parallel query executor.
#[cfg(feature = "gpu")]
pub use triblespace_gpu as gpu;

pub mod macros {
    pub use triblespace_core::macros::id_hex;
    pub use triblespace_macros::{
        attributes, entity, exists, find, pattern, pattern_changes, value_formatter,
    };
}

pub mod prelude {
    pub use crate::macros::{
        attributes, entity, exists, find, id_hex, pattern, pattern_changes, value_formatter,
    };
    pub use triblespace_core::prelude::*;
}

#[cfg(feature = "telemetry")]
pub mod telemetry;

#[cfg(kani)]
#[path = "../proofs/mod.rs"]
mod proofs;

#[cfg(doctest)]
mod book_doctests {
    #[doc = include_str!("../book/src/query-language.md")]
    pub struct QueryLanguage;
    #[doc = include_str!("../book/src/macro-cookbook.md")]
    pub struct MacroCookbook;
    #[doc = include_str!("../book/src/patterns-and-recipes.md")]
    pub struct PatternsAndRecipes;
    #[doc = include_str!("../book/src/encodings.md")]
    pub struct Schemas;
    #[doc = include_str!("../book/src/repository-workflows.md")]
    pub struct RepositoryWorkflows;
    #[doc = include_str!("../book/src/getting-started.md")]
    pub struct GettingStarted;
    #[doc = include_str!("../book/src/incremental-queries.md")]
    pub struct IncrementalQueries;
    #[doc = include_str!("../book/src/query-engine.md")]
    pub struct QueryEngine;
    #[doc = include_str!("../book/src/importing-data-formats.md")]
    pub struct ImportingDataFormats;
    #[doc = include_str!("../book/src/garbage-collection.md")]
    pub struct GarbageCollection;
    #[doc = include_str!("../book/src/pile-blob-metadata.md")]
    pub struct PileBlobMetadata;
    #[doc = include_str!("../book/src/pile-format.md")]
    pub struct PileFormat;
}

// Keep the README example here so the facade crate exercises the public API as
// consumers see it while `triblespace-core` stays lean for proc-macro usage.
#[cfg(test)]
mod readme_example {
    use crate::prelude::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    mod literature {
        use crate::prelude::blobencodings::UTF8String;
        use crate::prelude::inlineencodings::{GenId, Handle, ShortString, R256};
        use crate::prelude::*;

        attributes! {
            /// The title of a work.
            ///
            /// Small doc paragraph used in the book examples.
            "A74AA63539354CDA47F387A4C3A8D54C" unsafe as pub title: ShortString;

            /// A quote from a work.
            "6A03BAF6CFB822F04DA164ADAAEB53F6" unsafe as pub quote: Handle<UTF8String>;

            /// The author of a work.
            "8F180883F9FD5F787E9E0AF0DF5866B9" unsafe as pub author: GenId;

            /// The first name of an author.
            "0DBB530B37B966D137C50B943700EDB2" unsafe as pub firstname: ShortString;

            /// The last name of an author.
            "6BAA463FD4EAF45F6A103DB9433E4545" unsafe as pub lastname: ShortString;

            /// The number of pages in the work.
            "FCCE870BECA333D059D5CD68C43B98F0" unsafe as pub page_count: R256;

        }
    }

    #[test]
    fn readme_example() -> Result<(), Box<dyn std::error::Error>> {
        use crate::core::collection::{reach, CollectionAdmission};

        let storage = MemoryRepo::default();
        let key = SigningKey::generate(&mut OsRng);
        let namespace = key.verifying_key();
        let name = CollectionName::new("library")?;
        let mut library = Collection::new(
            storage,
            &name,
            namespace,
            key,
            reach::private(),
            CollectionAdmission::Open,
        );

        let mut initial = entity! {
            literature::firstname: "Frank",
            literature::lastname: "Herbert",
        };
        let herbert = initial.root().expect("intrinsic author identity");

        let quote = initial.put("I must not fear. Fear is the mind-killer.");
        initial += entity! {
            literature::title: "Dune",
            literature::author: &herbert,
            literature::quote: quote,
        };

        library.commit(initial)?;

        let snapshot = library.snapshot()?;
        let catalog = snapshot.facts();
        let title = "Dune";

        for (f, l, quote) in find!(
            (first: String, last: String, quote),
            pattern!(&catalog, [
                { _?author @
                    literature::firstname: ?first,
                    literature::lastname: ?last
                },
                { _?book @
                    literature::title: title,
                    literature::author: _?author,
                    literature::quote: ?quote
                }
            ])
        ) {
            let quote: View<str> = snapshot.reader().get(quote)?;
            let quote = quote.as_ref();
            println!("'{quote}'\n - from {title} by {f} {l}.");
        }

        // Independent commits coexist and materialize by set union. There is
        // no mutable head to race and no conflict-resolution loop to write.
        library.commit(entity! {
            literature::firstname: "Francis",
            literature::lastname: "Bacon",
        })?;
        library.commit(entity! {
            literature::firstname: "Franklin",
            literature::lastname: "Roosevelt",
        })?;

        let catalog = library.materialize()?;
        let mut names: Vec<String> = find!(
            first: String,
            pattern!(&catalog, [{ _?author @ literature::firstname: ?first }])
        )
        .collect();
        names.sort();
        assert_eq!(names, ["Francis", "Frank", "Franklin"]);

        Ok(())
    }
}
