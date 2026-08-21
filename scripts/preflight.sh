#!/usr/bin/env bash
set -euo pipefail

# Move to repository root
cd "$(dirname "$0")/.."

# Ensure rustfmt is installed
rustup component add rustfmt

# Ensure mdBook is installed
if ! command -v mdbook >/dev/null 2>&1; then
    cargo install mdbook
fi

# Run formatting check, tests, and build the book
cargo fmt -- --check
# `sim` gates three integration binaries — sim_collection_wire, sim_collection_gossip
# and sim_lazy — behind `#![cfg(feature = "sim")]`. No crate in the workspace enables
# it, so a bare `cargo test` compiled them EMPTY and printed "test result: ok" with
# 0 passed / 0 failed. 31 tests, including every collection-transfer test, never ran;
# the gate was green because it was empty. Found 2026-08-21.
cargo test --features sim
./scripts/build_book.sh
