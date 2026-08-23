# Positive Authority and CONNECT

TribleSpace authority is a public, grow-only set of positive grant
occurrences. Each occurrence is an ordinary signed collection commit: the
commit signer is the issuer, the commit's intrinsic ID is the grant occurrence
ID, and the committed `SimpleArchive` contains one exact grant atom.

The kernel has no mutable membership record, negative grant, permission
hierarchy, expiry, renewal, or retraction. Independent valid grants combine by
set union. Delegation follows exact parent occurrences, so portable evidence is
a concrete root-to-leaf chain rather than a lookup into ambient policy.

## One public authority collection per team

A team is rooted in one Ed25519 public key. That key determines a canonical
collection descriptor named `authority` with:

- `SimpleArchive` elements;
- set union as its join law; and
- public reach, so signed grant evidence may be relayed.

The descriptor's content handle is
`authority::collection(team_root)`. There is exactly one such authority
collection for a team root. It is an ordinary native collection, not a new
storage primitive or a mutable policy table.

Every accepted authority record is a `CollectionCommit` in that collection.
Its data must be exactly one canonical grant entity and its metadata must be
the canonical empty archive. A grant names:

| Field | Meaning |
|---|---|
| subject | one direct Ed25519 public-key principal |
| resource | one exact collection descriptor handle |
| action | one exact, uninterpreted action ID |
| parent | zero or one exact grant occurrence ID |
| invoke | whether the subject may perform the action |
| delegate | whether the subject may issue a child for the same action and resource |

Invocation and delegation are independent uses. A grant may carry invocation,
delegation, or both, but never neither. Actions do not imply one another: a
grant for one action is not a weaker or stronger spelling of another action.
Resources are equally exact; there is no ambient resource namespace or
wildcard.

`triblespace-core` defines `ACTION_WRITE` for authorizing a subject to
contribute signed commits to one exact collection. `triblespace-net` defines a
separate `ACTION_CONNECT`. A grant for either one says nothing about the
other.

## Positive fixed-point resolution

Authority resolution observes one known prefix of the team's authority
collection and computes its least positive fixed point.

A no-parent occurrence is accepted only when its collection commit is signed
by the team root. A child occurrence is accepted only when all of the
following hold:

1. its exact named parent occurrence is already accepted;
2. the child commit signer is the parent grant's subject;
3. the child preserves the parent's exact action and resource; and
4. the parent grants delegation.

The child's direct subject and invoke/delegate mode are its own explicit
fields. Several independently grounded grants are alternatives: any one exact
accepted occurrence can establish the requested authority.

Malformed or incomplete candidates remain inert instead of poisoning the
whole observation. Resolution reports diagnostics for invalid signatures,
non-canonical metadata or grant data, unavailable data blobs, wrong root
issuers, signer/subject mismatches, changed actions or resources,
non-delegating parents, and unresolved parents. Adding a previously missing
parent or data blob can ground more grants on a later observation; an already
accepted grant is never removed by set growth.

## Claim-directed portable proofs

An `AuthorityProof` carries the exact accepted ancestry needed for one claim.
Its steps are ordered root to leaf. Each step contains:

- the complete 192-byte signed authority `CollectionCommit`; and
- the canonical grant archive whose content identity that commit names.

Verification is standalone. It strictly verifies every commit, recomputes and
checks every adjacent data identity, requires the canonical team authority
collection and empty metadata, and enforces the exact parent, issuer, action,
resource, and delegation rules at every step.

Verification is also claim-directed. After validating the chain, the verifier
compares its leaf with the caller's required subject, action, resource, and
minimum invoke/delegate mode. A valid prefix proves its own leaf; it cannot be
mistaken for proof of a descendant that the caller expected. A grant carrying
both uses satisfies a claim requiring invocation alone or delegation alone.

The portable wire codec is versioned and bounded. Version 1 encodes:

```text
version:u8
step_count:u8
repeat step_count times:
    commit:192
    grant_data_length:u16
    grant_data:bytes
```

The one-byte count bounds a transport proof to 255 steps. Each grant archive
is bounded by the seven tribles in the canonical delegated shape. Decoding
checks the complete framing before allocating step payloads. These are
transport bounds, not an expiry or a depth limit in the authority algebra.

Invite bundles prepend the 32-byte team-root public key to the same proof
bytes. They are public, self-contained evidence. Possessing a bundle does not
let another key use it because verification binds the leaf subject to the
claim, and CONNECT authentication binds that claim to the transport peer's
Ed25519 key.

## CONNECT authenticates direct RPC only

`ACTION_CONNECT` authorizes one exact subject to establish an authenticated
direct-RPC connection for one team. Its required resource is always that
team's authority collection:

```text
subject = transport peer's Ed25519 key
action = ACTION_CONNECT
resource = authority::collection(team_root)
required mode = Invoke
```

That atom grants no `WRITE`, generic `READ`, gossip membership, collection
reach, blob custody, retention, or semantic trust. Collection reach still
decides which signed commits a holder may proactively relay. Author admission
still belongs to the resolver selecting a collection view. Gossip remains a
sparse evidence transport, and local WANT/retention policy remains local.
After CONNECT admits a session, the endpoint may answer its configured
read-only RPC surface; that disclosure is a property of the host's serving
snapshot, not a READ grant carried by CONNECT.

The direct protocol uses ALPN `/triblespace/pile-sync/6`. The first stream on
every connection must be `OP_AUTH` (`0x05`) carrying the complete bounded proof
inline. The server verifies an exact CONNECT claim for the TLS peer and replies
with `AUTH_OK` or `AUTH_REJECTED`. Only after success are the read-only direct
RPC operations served on later streams.

There is no pre-auth proof-fetch operation and no ambient proof lookup. A
caller must already possess the complete proof it presents. `OP_AUTH` cannot
appear again later on the connection.

## Team CLI

The `trible team` surface has five commands. All authority evidence lives in
the supplied pile.

```text
trible team create --pile PATH [--key KEY_PATH]
```

Creates a fresh team-root key, initializes the founder key at its conventional
path if needed, and publishes a root-signed founder CONNECT grant with both
invocation and delegation. It prints the team-root public key, the team-root
secret, and the founder grant ID. The root secret is not a mutable membership
database and is not written to the pile or a key file; capture it from the
command output and store it offline because anyone holding it can publish
independent root grants for that team.

```text
trible team invite --pile PATH --team-root HEX --parent ID \
    --key ISSUER_KEY --invitee HEX [--delegate] --out FILE
```

Loads an existing issuer key and requires `--parent` to be an accepted exact
CONNECT grant whose subject is that issuer and whose mode permits delegation.
It publishes a child occurrence and writes a self-contained invite bundle.
Without `--delegate`, the child invokes CONNECT only. With `--delegate`, it
both invokes and may issue another child.

```text
trible team join --pile PATH --key INVITEE_KEY --invite FILE
```

Loads the invitee's existing key, verifies the bundle against an exact CONNECT
claim for that key, and idempotently imports the authority descriptor, empty
metadata archive, grant data archives, and signed commits.

```text
trible team list --pile PATH --team-root HEX
```

Prints accepted occurrences in intrinsic commit-ID order and the diagnostics
for inert candidates.

```text
trible team show --pile PATH --team-root HEX --grant ID
```

Prints one accepted occurrence's exact ancestry from the root grant to the
selected leaf. An inert or absent occurrence is an error rather than a partial
chain.

A complete bootstrap is explicit:

```bash
# Founder
trible pile create founder.pile
trible team create --pile founder.pile --key founder.key
# Save the printed team root and founder grant ID.

# Invitee creates only its local transport key.
trible pile create member.pile
trible pile net identity --key member.key

# Founder publishes the child and writes a portable bundle.
trible team invite \
  --pile founder.pile \
  --team-root <TEAM_ROOT> \
  --parent <FOUNDER_GRANT_ID> \
  --key founder.key \
  --invitee <MEMBER_PUBLIC_KEY> \
  --out member.invite

# Transfer member.invite by any ordinary file channel, then import it.
trible team join \
  --pile member.pile \
  --key member.key \
  --invite member.invite
```

`pile net status` and `pile net sync` then require the pile, team root, and
exact accepted grant explicitly:

```bash
trible pile net status member.pile \
  --key member.key --team-root <TEAM_ROOT> --grant <MEMBER_GRANT_ID>

trible pile net sync member.pile \
  --key member.key --team-root <TEAM_ROOT> --grant <MEMBER_GRANT_ID> \
  --peers <FOUNDER_ENDPOINT>
```

Both commands load an existing key, resolve that exact local occurrence,
confirm it invokes CONNECT for the key, reconstruct its ancestry, and reject a
proof that exceeds the transport bounds. There is no environment-variable
fallback, all-zero sentinel, implicit team-of-one credential, or automatic key
creation on these paths. `pile net identity` is the explicit key-initialization
command.

## Deliberate boundary: no removal inside an epoch

Positive authority is monotone. A valid occurrence does not expire and cannot
be retracted, renewed, denied by an admin hierarchy, or hidden behind a pending
approval workflow. The kernel also has no distinguished "current membership"
head whose replacement could invalidate earlier proofs.

Durable removal is therefore an epoch change outside this kernel: move to a
successor team, collection, or key epoch and enforce the cutoff at that new
epoch's external admission boundary. Merely issuing a new subject key under
the same still-served team root does not invalidate an old CONNECT proof. That
cost is explicit. It is what keeps authority proof verification local,
portable, and invariant under pile concatenation.
