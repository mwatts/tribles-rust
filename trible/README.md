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
- `pile migrate <PILE> list` — list known migrations and whether they are needed for this pile.
- `pile migrate <PILE> run [MIGRATION]` — run migrations (all by default). Pass `--dry-run` to preview changes.

Legacy piles can be migrated directly into native collections:

```bash
trible pile migrate <PILE> branch-to-collection \
  --branch <LEGACY_BRANCH> \
  --collection-name <NAME> \
  --team-root <PUBLIC_KEY> \
  --signing-key <KEY_PATH>
```

This is an explicit compatibility operation over an immutable legacy pin
snapshot. The current CLI does not create, advance, merge, or delete branches.

#### Blobs

- `pile blob list [--metadata] <PILE>` — list stored blob handles. Pass `--metadata` to include timestamps and sizes.
- `pile blob put <PILE> <FILE>` — store a file as a blob and print its handle.
- `pile blob get <PILE> <HANDLE> <OUTPUT>` — extract a blob by handle.
- `pile blob inspect <PILE> <HANDLE>` — display metadata for a stored blob.

#### Collections

A collection is identified by the blake3 handle of its canonical descriptor
blob — a `SimpleArchive` naming its anchor, representation, join recipe, and
reach law, together with the representation and recipe descriptions.
`pile blob inspect` sees only the encoded blob; these subcommands decode it.

- `pile collection list [--metadata] <PILE>` — one row per distinct collection the pile's commit / merge / derive records reference, with the decoded scope, representation, and recipe (known representation and recipe ids are named). Pass `--metadata` for per-collection record counts and the descriptor blob's size and storage timestamp.
- `pile collection show <PILE> <HANDLE>` — decode one descriptor, its anchor,
  representation, recipe, reach law, and referencing record counts. The handle
  is accepted with or without the `blake3:` prefix.

### Distributed pile sync

Built on `triblespace-net` (iroh QUIC + DHT + gossip). Every connection
authenticates with an exact, positive CONNECT grant chain rooted at the team's
public key; see *Team authority* below. Runtime configuration is explicit: the
local pile must contain the accepted chain named by `--team-root` and
`--grant`.

- `pile net identity [--key PATH]` — print this node's iroh identity (auto-generates a key if missing).
- `pile net status <PILE> --team-root HEX --grant ID [--key PATH]` — resolve the exact local CONNECT grant, reconstruct its ancestry proof, and print the configuration the node would present during authentication.
- `pile net sync <PILE> --team-root HEX --grant ID [--peers ID,...] [--key PATH]` — long-running collection-evidence sync on the team's gossip mesh. The mesh is identified by the team root directly. The exact accepted grant must invoke CONNECT for the local key; no ambient environment fallback or sentinel exists. Collection records converge by set union, while referenced blobs and operation receipts stay lazy until requested by durable wants. Use `--read-only` or `--write-only` for directional operation and `--no-lazy` to suppress want reconciliation.

### Team authority

Team authority is one public, grow-only collection of positive signed grant
occurrences. A grant names one direct subject key, one exact resource, one
action, an invoke/delegate mode, and optionally one exact delegating parent.
There is no permission hierarchy, expiry, retraction, pending workflow, or
mutable membership head in this kernel. A team root signs the founder grant;
every invite carries the exact bounded parent chain needed to validate and
import it.

- `team create --pile PATH [--key KEY_PATH]` — mint a team root, publish an explicit founder CONNECT grant with invocation and delegation, and print the root public key, offline root secret, and founder grant id.
- `team invite --pile PATH --team-root HEX --parent ID --key ISSUER --invitee HEX [--delegate] --out FILE` — prove that ISSUER owns the exact delegating CONNECT parent, publish the child, and write a portable public proof bundle. Without `--delegate`, the child may connect but cannot invite.
- `team join --pile PATH --key INVITEE --invite FILE` — verify that the self-contained bundle grants CONNECT to INVITEE on the exact team authority collection, then import its descriptor, grant data, and signed commits idempotently.
- `team list --pile PATH --team-root HEX` — resolve and print accepted grants plus inert-candidate diagnostics.
- `team show --pile PATH --team-root HEX --grant ID` — print the exact accepted root-to-leaf ancestry of one grant occurrence.

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
network, team, and remote-blob modules. Contributions are always welcome!
