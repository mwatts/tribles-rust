#!/usr/bin/env bash
set -euo pipefail

# Move to repository root
cd "$(dirname "$0")/.."

# Run only the tests for quick iteration.
#
# `--features triblespace-net/sim` is not optional here either: without it
# sim_collection_wire, sim_collection_gossip and sim_lazy compile empty and
# report "ok" with 0 passed / 0 failed. The feature is package-qualified
# because it lives on triblespace-net, not on the root package.
cargo test --workspace --features triblespace-net/sim
