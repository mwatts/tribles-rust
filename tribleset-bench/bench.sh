#!/bin/sh
# bench.sh — THE single command: bench any subject rev or checkout.
#
#   ./bench.sh <rev-or-path> [runner args…]
#
#   <rev-or-path>  a git rev in ../triblespace-rs (a detached worktree
#                  is created/reused under ./subjects/<rev12>), or a
#                  path to any triblespace checkout.
#   runner args    forwarded verbatim to the tribleset-bench binary
#                  (--results/--label/--data/…; run with none to see
#                  usage).
#
# MECHANISM (dep repointing): the `subject` dependency in Cargo.toml is
# a path dep on the `subjects/current` symlink; this script points the
# symlink at the requested checkout and runs `cargo run --release`.
# Cargo `[patch]`/`--config` patching was considered and rejected: a
# patch section replaces a *source* (registry or git URL) and cannot
# retarget a path dependency, and declaring the subject as a registry
# dep so it becomes patchable would let cargo unify it with the results
# LEDGER (also package `triblespace`) — one crate instance instead of
# the two the suite's design requires. The symlink keeps the manifest
# static, needs no cargo nightly features, and cargo fingerprints the
# resolved target, so switching subjects triggers exactly the rebuild
# it should.
#
# The default target (fresh clone, before any rev was requested) is
# ../../triblespace-rs; `./bench.sh ../triblespace-rs …` recreates it.

set -eu
cd "$(dirname "$0")"

[ $# -ge 1 ] || {
    echo "usage: ./bench.sh <rev-or-path> [runner args…]" >&2
    exit 2
}
arg="$1"
shift

mkdir -p subjects
if [ -d "$arg" ]; then
    target=$(cd "$arg" && pwd)
else
    rev=$(git -C ../triblespace-rs rev-parse --verify --short=12 "$arg^{commit}") || {
        echo "bench.sh: '$arg' is neither a directory nor a rev in ../triblespace-rs" >&2
        exit 2
    }
    wt="subjects/$rev"
    if [ ! -d "$wt" ]; then
        git -C ../triblespace-rs worktree add --detach "$(pwd)/$wt" "$rev"
    fi
    target=$(cd "$wt" && pwd)
fi

ln -sfn "$target" subjects/current
echo "bench.sh : subject -> $target" >&2

exec cargo run --release -- "$@"
