//! tribleset-bench — engine-version-agnostic benchmark suite runner.
//!
//! Stub stage: mounts the vendored query modules against the `subject`
//! dependency and prints the registry census. The measuring runner and
//! the ledger writer land in the next commit.

#[path = "../queries/wd_schema.rs"]
mod wd_schema;

#[path = "../queries/sparqloscope.rs"]
mod queries;

fn main() {
    println!(
        "tribleset-bench {}: {} translated sparqloscope queries, {} rpq-gated",
        env!("CARGO_PKG_VERSION"),
        queries::TRANSLATED.len(),
        queries::SKIPPED_PATHS.len(),
    );
    for t in queries::TRANSLATED {
        println!("  {:<55} {:?}", t.name, t.kind);
    }
    for name in queries::SKIPPED_PATHS {
        println!("  {name:<55} SKIP (rpq)");
    }
}
