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
snapshot. Here `--team-root` is deliberately both the target name namespace
and its optional capability authority, matching the current named-collection
facade. The current CLI does not create, advance, merge, or delete branches.

#### Blobs

- `pile blob list [--metadata] <PILE>` — list stored blob handles. Pass `--metadata` to include timestamps and sizes.
- `pile blob put <PILE> <FILE>` — store a file as a blob and print its handle.
- `pile blob get <PILE> <HANDLE> <OUTPUT>` — extract a blob by handle.
- `pile blob inspect <PILE> <HANDLE>` — display metadata for a stored blob.

#### Collections

A collection is identified by the blake3 handle of its canonical descriptor
blob — a `SimpleArchive` naming its root (`name` + public-key namespace) or
derived source, optional local capability authority, representation, join
recipe, and reach law, together with the representation and recipe
descriptions.
`pile blob inspect` sees only the encoded blob; these subcommands decode it.

- `pile collection list [--metadata] <PILE>` — one row per distinct collection the pile's commit / merge / derive records reference, with the decoded namespace/source anchor, local authority, representation, and recipe (known representation and recipe ids are named). Pass `--metadata` for per-collection record counts and the descriptor blob's size and storage timestamp.
- `pile collection show <PILE> <HANDLE>` — decode one descriptor, its anchor,
  local authority, representation, recipe, reach law, and referencing record
  counts. The handle is accepted with or without the `blake3:` prefix.

### Distributed pile sync

Built on `triblespace-net` (iroh QUIC + DHT + gossip). Every connection
authenticates with an exact CONNECT capability proof rooted at the team's
public key; see *Team capabilities* below. Runtime configuration is explicit:
the local pile must contain the exact claim/signature chain named by
`--team-root` and `--credential`. Authentication identity and gossip
rendezvous are independent inputs.

- `pile net identity [--key PATH]` — print this node's iroh identity (auto-generates a key if missing).
- `pile net status <PILE> --team-root HEX --credential HANDLE [--key PATH]` — load the designated credential by exact blob handle, reconstruct and verify its CONNECT proof for the local key at the current time, and print the authentication configuration.
- `pile net sync <PILE> --team-root HEX --credential HANDLE --gossip-topic HEX [--peers ID,...] [--key PATH]` — long-running collection-evidence sync. `--team-root` selects the CONNECT trust root and exact resource; the separately required `--gossip-topic` selects rendezvous without inference or fallback. The designated credential must invoke CONNECT for the local key. Collection records converge by set union, while referenced blobs and operation receipts stay lazy until requested by durable wants. Use `--read-only` or `--write-only` for directional operation and `--no-lazy` to suppress want reconciliation.

### Team capabilities

A team is identified by an Ed25519 trust-root public key, whose exact 32 bytes
are also its CONNECT resource. Each capability step is one canonical claim blob
and one signature blob. A claim names one subject, exact action/resource atom,
invoke/delegate mode, optional parent signature handle, and optional inclusive
validity interval. Proofs follow those handles from one designated leaf; there
is no authority collection, resolver, global membership registry, or list
operation. An invite carries the complete bounded root-to-leaf proof needed for
standalone verification.

- `team create --pile PATH [--key KEY_PATH] [--valid-from RFC3339 --valid-until RFC3339]` — mint a team root, issue the founder CONNECT in `invoke+delegate` mode, store the exact claim/signature blobs, and print the root public key, offline root secret, and founder credential handle.
- `team invite --pile PATH --team-root HEX --parent HANDLE --key ISSUER --invitee HEX [--delegate] [--valid-from RFC3339 --valid-until RFC3339] --out FILE` — load one exact resident parent, verify its delegation capability at the current time, issue and store the child blobs, and write a portable proof bundle. Without `--delegate`, the child may connect but cannot invite.
- `team join --pile PATH --key INVITEE --invite FILE` — verify the bundle's exact root, subject, CONNECT atom, minimum invoke mode, and current validity before storing its claim/signature blobs idempotently. It prints the accepted leaf credential.
- `team show --pile PATH --team-root HEX --credential HANDLE` — load, verify, and print the exact root-to-leaf ancestry of one designated credential.

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
