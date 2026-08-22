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

# Run formatting check, tests, and build the book. The repository root is both
# a package and a workspace, so `--all` is required to check every member.
cargo fmt --all -- --check
# `sim` gates three integration binaries — sim_collection_wire, sim_collection_gossip
# and sim_lazy — behind `#![cfg(feature = "sim")]`. No crate in the workspace enables
# it, so a bare `cargo test` compiled them EMPTY and printed "test result: ok" with
# 0 passed / 0 failed. 31 tests, including every collection-transfer test, never ran;
# the gate was green because it was empty. Found 2026-08-21.
#
# The feature lives on `triblespace-net`, not on the root `triblespace` package, so
# it must be named package-qualified: a bare `--features sim` resolves against the
# root package and cargo refuses with "does not contain this feature" before running
# a single test — a gate that fails to start is no better than one that starts empty.
cargo test --workspace --features triblespace-net/sim
./scripts/build_book.sh
