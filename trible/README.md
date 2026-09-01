# Trible CLI

Trible CLI is a friendly companion for exploring and managing
[Tribles](https://github.com/triblespace/tribles-rust) and TribleSpace piles
from the command line.

This crate tracks `triblespace` releases (major/minor), and may ship independent patch releases.

## Installation

```bash
cargo install trible
```

Or, for local development:

```bash
cargo install --path .
```

## Quick Start

1. Create a new pile to hold your data:

   ```bash
   trible pile create demo.pile
   ```

2. Add a file as a blob. This command prints a handle for the stored blob:

   ```bash
   echo "hello" > greeting.txt
   trible pile blob put demo.pile greeting.txt
   ```

3. List the blobs in the pile to confirm the handle:

   ```bash
   trible pile blob list demo.pile
   ```

4. Retrieve the blob using its handle:

   ```bash
   trible pile blob get demo.pile <HANDLE> copy.txt
   ```

The file `copy.txt` now contains the original contents of `greeting.txt`.

## Usage

Run `trible <COMMAND>` to invoke a subcommand.

### Generate identifiers

- `genid` — generate a random identifier.

### Generate shell completions

- `completion <SHELL>` — output a completion script for `bash`, `zsh`, or `fish`.

### Work with piles

- `pile create <PATH>` — initialize an empty pile, creating parent directories as needed.
- `pile diagnose check <PILE>` — verify pile integrity.
- `pile diagnose locate-hash <PILE> <HANDLE>` — scan raw pile bytes and report where a handle appears (blob header vs payload references).
- `pile compact <SOURCE> --into <DESTINATION>` — conservatively repack into a
  fresh pile. Every distinct valid blob and every distinct native collection,
  proof, and peer record remains; active WANTs, legacy pins, and the unique
  store scope are projected once. Blob records receive fresh timestamps,
  corrupt physical occurrences disappear when another occurrence validates,
  and known semantically inert retired records are dropped. Opaque kinds are
  refused. Quiesce writers for an exact whole-file result because a late append
  may remain outside the valid observed prefix. On Unix the destination starts
  no broader than mode 0600, then receives the source permissions through its
  open handle after rewriting. On post-create failure the command attempts to
  remove the destination and reports cleanup failure.
- `pile migrate <PILE> list` — list known migrations and whether they are needed for this pile.
- `pile migrate <PILE> run [MIGRATION]` — run migrations (all by default). Pass `--dry-run` to preview changes.

Legacy piles can be migrated directly into native collections:

```bash
trible pile migrate <PILE> branch-to-collection \
  --branch <LEGACY_BRANCH> \
  --collection-name <NAME> \
  [--authority <TRUST_ROOT>] \
  --signing-key <KEY_PATH>
```

This is an explicit compatibility operation over an immutable legacy pin
snapshot. `--authority` selects the direct trust root for both the target's
READ and WRITE policies and defaults to the migration signer. Choosing another
root does not block local publication, but the resulting commits remain
inadmissible until the pile contains an exact WRITE proof for their signer.
The command validates the full frozen legacy closure before registering the
target, preserves every existing fact and metadata entity id, and does not
create, advance, merge, or delete branches.

#### Blobs

- `pile blob list [--metadata] <PILE>` — list stored blob handles. Pass `--metadata` to include timestamps and sizes.
- `pile blob put <PILE> <FILE>` — store a file as a blob and print its handle.
- `pile blob get <PILE> <HANDLE> <OUTPUT>` — extract a blob by handle.
- `pile blob inspect <PILE> <HANDLE>` — display metadata for a stored blob.

#### Collections

A collection is identified by the blake3 handle of its canonical descriptor
blob — a `SimpleArchive` naming its root or derived source, independent READ
and WRITE admission policies, representation, and concrete mapping, together
with the encoding and mapping-algorithm descriptions.
`pile blob inspect` sees only the encoded blob; these subcommands decode it.

- `pile collection init <PILE> <NAME> [--key PATH]` — register a canonical
  named `SimpleArchive` descriptor whose direct READ and WRITE root is an
  existing durable signing key, then print its exact handle. It stores no
  synthetic commit and is idempotent.
- `pile collection list [--metadata] <PILE>` — one row per distinct collection the pile's commit / merge / derive records reference, with the decoded name/source anchor, READ and WRITE policies, representation, and mapping algorithm (known representation and algorithm ids are named). Pass `--metadata` for per-collection record counts and the descriptor blob's size and storage timestamp.
- `pile collection show <PILE> <HANDLE>` — decode one descriptor, its anchor,
  READ and WRITE policies, representation, mapping, and referencing record
  counts. The handle is accepted with or without the `blake3:` prefix.
- `pile collection grant-read <PILE> <COLLECTION> <RECIPIENT> [--key PATH]` —
  issue one deterministic, unbounded READ/Invoke proof from a configured READ
  root to an Ed25519 public key. Claims are persisted before the proof;
  replaying the exact command is idempotent.
- `pile collection grant-write <PILE> <COLLECTION> <RECIPIENT> [--key PATH]` —
  issue the symmetric deterministic WRITE/Invoke proof from a configured WRITE
  root to an author key.

### Distributed pile sync

Built on `triblespace-net` (authenticated iroh QUIC, collection-scoped PATCH
anti-entropy, stock-gossip wakeups, and DHT provider lookup). Opening a
transport connection grants no collection authority. Each repair request names
one exact collection and carries the caller's portable READ(C) proof bundles;
the server checks them before disclosing a manifest or PATCH leaf.

- `pile net identity [--key PATH]` — print this node's iroh identity (auto-generates a key if missing).
- `pile net sync <PILE> --collection HANDLE [--collection HANDLE ...] [--peers ID_OR_TICKET,...] [--key PATH] [--direction bidirectional|read-only|write-only]` — activate the named collections and run periodic repair. `read-only` pulls but does not serve, while `write-only` serves admitted readers but does not pull or service local WANTs. `--duration SECS` and `--quiescent-for SECS` provide optional process-lifecycle bounds.

The exact repair state is the product of the collection's native record PATCH
and its portable WRITE-evidence PATCH. A later WRITE proof can therefore
activate an older commit without inventing a second synchronization protocol.
Production peers subscribe to stock `iroh-gossip` topics keyed by the
domain-separated image of the collection handle; signed opaque-root
mismatches accelerate ordinary repair, while periodic anti-entropy remains
authoritative.

DHT routing, provider lookup, and direct GET by a known immutable handle are
collection-independent bearer mechanisms. Every served resident blob may
publish an opaque KDF(H) lease with an H-bound endpoint token. Direct GET sends
only that locator: the provider proves H first, the requester second, both
proofs bind the authenticated endpoints, and returned bytes must hash to H.
Collection READ(C) remains exclusively the admission boundary for collection
anti-entropy and Full repair. The network host neither uses nor writes durable
team/PEER routing state.

### Work with remote stores

#### Blobs

- `store blob list <URL>` — list objects at a remote store.
- `store blob put <URL> <FILE>` — upload a file to a remote store and print its handle.
- `store blob get <URL> <HANDLE> <OUTPUT>` — download a blob from a remote store.
- `store blob forget <URL> <HANDLE>` — remove an object from a remote store.
- `store blob inspect <URL> <HANDLE>` — display metadata for a remote blob.

See `INVENTORY.md` for notes on possible cleanup and future functionality.

## Development

Command implementations live in `src/cli/`, with pile, collection, migration,
network, and remote-blob modules. Contributions are always welcome!
