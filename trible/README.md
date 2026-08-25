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
  --namespace <PUBLIC_KEY> \
  --signing-key <KEY_PATH>
```

This is an explicit compatibility operation over an immutable legacy pin
snapshot. `--namespace` names the target but grants no authority; omission of
authorization flags selects explicitly open admission. For controlled
admission, pass `--authority <TRUST_ROOT> --proof <PROOF_ID>`. The exact native
proof and its named claim blobs must prove WRITE/Invoke for the signing key and
target before target bytes are staged. The proof may be omitted only when the
signing key is the authority root, in which case the command mints, stores, and
prints the exact root proof. The current CLI does not create, advance, merge,
or delete branches.

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
the local pile must contain the native proof selected by `--proof` and every
claim blob it names. Authentication identity and gossip rendezvous are
independent inputs.

- `pile net identity [--key PATH]` — print this node's iroh identity (auto-generates a key if missing).
- `pile net status <PILE> --team-root HEX --proof ID [--key PATH]` — load the exact native proof and its named claims, verify CONNECT for the local key at the current time, and print the authentication configuration.
- `pile net sync <PILE> --team-root HEX --proof ID --gossip-topic HEX [--peers ID,...] [--key PATH]` — long-running collection-evidence sync. `--team-root` selects the CONNECT trust root and exact resource; the separately required `--gossip-topic` selects rendezvous without inference or fallback. The proof must invoke CONNECT for the local key. Collection records converge by set union, while referenced blobs and operation receipts stay lazy until requested by durable wants. Use `--read-only` or `--write-only` for directional operation and `--no-lazy` to suppress want reconciliation.

### Team capabilities

A team is identified by an Ed25519 trust-root public key, whose exact 32 bytes
are also its CONNECT resource. Semantic restrictions are keyless canonical
claim blobs linked by parent claim handles. Principal delegation is one native
`K0 (S C K)+` proof whose signatures bind issuer key, exact claim handle, and
delegate key. Its BLAKE3 digest is the proof ID used for exact lookup. An invite
carries the complete proof and ordered claims required for standalone
verification.

- `team create --pile PATH [--key KEY_PATH] [--root-key ROOT_KEY_PATH] [--valid-from RFC3339 --valid-until RFC3339]` — initialize a durable offline team-root key, issue the founder CONNECT in `invoke+delegate` mode, store its claim and native proof, and print the root public key, root-key path, and founder proof ID.
- `team invite --pile PATH --team-root HEX --parent-proof ID --key ISSUER --invitee HEX [--delegate] [--valid-from RFC3339 --valid-until RFC3339] --out FILE` — load one exact resident proof, verify its current delegation capability, extend it for the invitee, and write a portable proof bundle. Without `--delegate`, the child may connect but cannot invite.
- `team join --pile PATH --team-root HEX --key INVITEE --invite FILE` — verify the bundle against the externally supplied team root, expected leaf, exact CONNECT atom, minimum invoke mode, and current validity before storing its claims and native proof idempotently. It prints the accepted proof ID.
- `team show --pile PATH --team-root HEX --proof ID` — load, verify, and print one exact proof and its claim ancestry.

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
