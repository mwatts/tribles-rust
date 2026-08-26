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

Built on `triblespace-net` (authenticated iroh QUIC plus gossip wake hints). A node presents two
independent exact proofs rooted at the team public key: CONNECT admits the
transport connection and SYNC_TEAM authorizes disclosure and reconciliation of
that team's inventory. The local pile must contain both selected native proofs
and every claim blob they name. The team root also deterministically selects
the gossip topic; gossip carries only lossy generation wake hints.

- `pile net identity [--key PATH]` — print this node's iroh identity (auto-generates a key if missing).
- `pile net status <PILE> --team-root HEX --connect-proof ID --sync-proof ID [--key PATH]` — load both exact native bundles, verify CONNECT and SYNC_TEAM for the local key at the current time, and print their IDs and step counts.
- `pile net sync <PILE> --team-root HEX --connect-proof ID --sync-proof ID [--peers ID_OR_TICKET,...] [--key PATH] [--direction bidirectional|read-only|write-only] [--blobs demand|mirror]` — run authorized periodic anti-entropy. PEER evidence, native collection records, and capability proofs converge by set union. `demand` (the default) fetches blobs only for durable WANTs; `mirror` also walks the complete blob inventory. `read-only` pulls but does not advertise or serve local data, while `write-only` advertises and serves but never pulls or demand-fetches. `--duration SECS` and `--quiescent-for SECS` provide optional process-lifecycle bounds.

The pile is one team-scoped store: records, proofs, and blobs do not carry a
separate team label. Configured peers are bootstrap routes, and successful
authorized synchronization carries monotone `PEER(team, peer)` routing
evidence. Lost gossip does not prevent convergence because periodic direct
sweeps remain the correctness path.

### Team capabilities

A team is identified by an Ed25519 trust-root public key, whose exact 32 bytes
are the resource for two distinct capabilities: CONNECT admits transport and
SYNC_TEAM permits inventory disclosure and reconciliation. Semantic
restrictions are keyless canonical claim blobs linked by parent claim handles.
Each independent `K0 (S C K)+` proof binds issuer key, exact claim handle, and
delegate key; its BLAKE3 digest is the exact lookup id. One versioned invite
artifact packages the bounded CONNECT and SYNC_TEAM proof bundles in fixed
order so joining remains one operation without conflating their authority.

- `team create --pile PATH [--key KEY_PATH] [--root-key ROOT_KEY_PATH] [--valid-from RFC3339 --valid-until RFC3339]` — initialize a durable offline team-root key, issue and store founder CONNECT and SYNC_TEAM proofs in `invoke+delegate` mode, and print both exact proof IDs.
- `team invite --pile PATH --team-root HEX --connect-parent-proof ID --sync-parent-proof ID --key ISSUER --invitee HEX [--delegate] [--valid-from RFC3339 --valid-until RFC3339] --out FILE` — verify and extend both exact resident delegation chains, then write one portable invite artifact. Without `--delegate`, the child may connect and synchronize but cannot invite.
- `team join --pile PATH --team-root HEX --key INVITEE --invite FILE` — validate both bundles against the externally supplied root, expected leaf, exact action, minimum invoke mode, and current validity before the first idempotent store write. It prints both accepted proof IDs.
- `team show --pile PATH --team-root HEX --proof ID` — load, verify, and print one exact CONNECT or SYNC_TEAM proof and its claim ancestry.

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
