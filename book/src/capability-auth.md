# Blob-Native Capabilities and CONNECT

TribleSpace capabilities are exact signed claim trees. Each claim and each
signature is an ordinary canonical `SimpleArchive` blob, named by its Blake3
content handle. A credential is simply the 32-byte handle of the leaf signature
blob.

There is no authority collection, policy resolver, membership scan, or global
credential registry. A caller designates one leaf handle, loads only the blobs
that exact ancestry names, and verifies one explicit claim against an external
trust root and instant.

## Claims, signatures, and credentials

A capability claim names:

| Field | Meaning |
|---|---|
| subject | one direct Ed25519 public-key principal |
| resource | one exact opaque 32-byte resource |
| action | one exact, uninterpreted action ID |
| mode | `Invoke`, `Delegate`, or `InvokeAndDelegate` |
| parent | zero or one exact parent signature-blob handle |
| validity | zero or one inclusive lower/upper TAI interval |

Invocation and delegation are independent uses. A mode satisfies a requested
mode only when it contains every requested use. Actions do not imply one
another, and resources have no wildcard or ambient namespace.

A signature blob names the exact claim-blob handle, signer public key, and
Ed25519 signature over those claim bytes. A root claim has no parent and must be
signed by the externally supplied trust root. A child names the immediately
preceding signature blob and must be signed by that parent's subject.

Claims and signatures are stored through the ordinary blob API. Storing a
proof adds no membership occurrence or mutable head: inserting the same bytes
again has the same handles. The leaf signature handle is therefore enough to
identify one credential without a separate index.

## Exact reconstruction and verification

`CapabilityProof::load` starts from one leaf credential. It loads the exact
signature blob, follows that signature's exact claim handle, then follows the
claim's exact parent signature handle until it reaches a root. It neither lists
storage nor searches for alternative grants. Missing, malformed,
wrongly-addressed, or cyclic evidence is an error.

Loading proves only that the named blobs can be reconstructed. Admission is
`CapabilityProof::verify_claim`, whose caller supplies:

- the external trust-root public key;
- the exact verification instant; and
- the expected leaf subject, action/resource atom, and minimum mode.

Verification hashes every blob from its bytes, parses closed canonical shapes,
and strictly checks every signature. It also requires:

1. the first claim to have no parent and its signature to come from the trust
   root;
2. every child to name the immediately preceding signature blob;
3. every child signature to come from its parent's subject;
4. every parent to carry delegation and contain the child's mode;
5. the exact action/resource atom to remain unchanged; and
6. the explicit instant to lie inside every bounded inclusive validity
   interval.

The verified capability reports the intersection of all bounded intervals as
its effective validity. `None` means every step is unbounded. A valid proof
prefix proves only its own leaf; it cannot substitute for a descendant the
caller expected.

## Portable proofs and invite bundles

A portable `CapabilityProof` is the root-to-leaf sequence of exact claim and
signature blobs. The network codec is versioned and bounded. Version 2 encodes:

```text
version:u8
step_count:u8
repeat step_count times:
    claim_length:u16
    signature_length:u16
    claim:bytes
    signature:bytes
```

The one-byte count bounds one transport proof to 255 steps. Claim and signature
lengths are bounded by their largest canonical archive shapes, and decoding
validates the complete frame before allocating step blobs. These are carrier
bounds; the capability algebra and exact-handle loader do not impose a depth
limit.

A team invite prepends the team's 32-byte trust-root public key to those proof
bytes. The bundle is public and self-contained. Possessing it does not let a
different key use it because verification binds the leaf subject to the
caller's expected key.

## CONNECT authenticates direct RPC only

For a team, the CONNECT resource is the trust-root public key's exact 32 bytes:

```text
subject = transport peer's Ed25519 key
action = ACTION_CONNECT
resource = team trust-root public-key bytes
required mode = Invoke
```

CONNECT grants no collection `WRITE`, generic `READ`, gossip membership,
collection reach, blob custody, retention, or semantic trust. It authorizes the
configured direct-RPC surface only. Collection reach still controls proactive
relay, and local WANT/retention policy remains local.

`PeerConfig` keeps authentication and rendezvous separate:

- `connect_root` is the external trust root and exact CONNECT resource;
- `connect_proof` is the already selected complete proof for this peer; and
- `gossip_topic` is an independent optional 32-byte application choice.

The direct protocol uses ALPN `/triblespace/pile-sync/7`. The first stream on
every connection must be `OP_AUTH` (`0x05`) carrying the complete bounded proof
inline. The server verifies CONNECT for the authenticated transport key at the
explicit current epoch and replies with `AUTH_OK` or `AUTH_REJECTED`. A bounded
successful session is closed after the proof's effective inclusive upper
bound.

There is no pre-auth proof fetch or ambient proof lookup. A caller already
possesses the exact proof it presents, and `OP_AUTH` cannot be repeated later
on the connection.

## Team CLI

The `trible team` surface has four commands. All claim/signature evidence lives
as ordinary blobs in the supplied pile.

```text
trible team create --pile PATH [--key KEY_PATH]
    [--valid-from RFC3339 --valid-until RFC3339]
```

Creates a fresh team-root key, initializes the founder key at its conventional
path if needed, issues a root-signed founder CONNECT credential in
`InvokeAndDelegate` mode, and stores its exact blobs. It prints the team-root
public key, offline root secret, and 64-hex-character founder credential. The
root secret is not written to the pile or a key file; anyone holding it can
issue an independent root credential for the team.

```text
trible team invite --pile PATH --team-root HEX --parent HANDLE \
    --key ISSUER_KEY --invitee HEX [--delegate] \
    [--valid-from RFC3339 --valid-until RFC3339] --out FILE
```

Loads exactly the designated parent credential from the pile and verifies that
the issuer holds CONNECT delegation at the current time. It signs the child,
stores the extended proof's exact blobs, writes a self-contained bundle, and
prints the child's leaf credential. Without `--delegate`, the child carries
invocation only; with it, the child carries invocation and delegation. Validity
bounds are optional, inclusive, and must be supplied as a pair.

```text
trible team join --pile PATH --key INVITEE_KEY --invite FILE
```

Loads the invitee's existing key and verifies the bundle's exact root, subject,
CONNECT atom, minimum invocation mode, and current validity before writing
anything. It then stores every claim/signature blob idempotently and prints the
accepted leaf credential.

```text
trible team show --pile PATH --team-root HEX --credential HANDLE
```

Loads one designated credential by exact handle, verifies it at the current
time, and prints its root-to-leaf ancestry. There is deliberately no team-wide
`list`: ordinary blob storage is not a global credential registry.

A complete bootstrap is explicit:

```bash
# Founder
trible pile create founder.pile
trible team create --pile founder.pile --key founder.key
# Save the printed team root and founder credential.

# Invitee creates only its local transport key.
trible pile create member.pile
trible pile net identity --key member.key

# Founder issues the child and writes a portable bundle.
trible team invite \
  --pile founder.pile \
  --team-root <TEAM_ROOT> \
  --parent <FOUNDER_CREDENTIAL> \
  --key founder.key \
  --invitee <MEMBER_PUBLIC_KEY> \
  --out member.invite

# Transfer member.invite by any ordinary file channel, then import it.
trible team join \
  --pile member.pile \
  --key member.key \
  --invite member.invite
```

`pile net status` requires the exact trust root and credential. `sync` also
requires an independent gossip topic; it is never inferred from authorization:

```bash
trible pile net status member.pile \
  --key member.key \
  --team-root <TEAM_ROOT> \
  --credential <MEMBER_CREDENTIAL>

trible pile net sync member.pile \
  --key member.key \
  --team-root <TEAM_ROOT> \
  --credential <MEMBER_CREDENTIAL> \
  --gossip-topic <32_BYTE_HEX_TOPIC> \
  --peers <FOUNDER_ENDPOINT>
```

Both commands load the existing key and exact local proof, verify CONNECT for
that key at the current time, and reject evidence outside the transport bounds.
There is no environment-variable fallback, all-zero sentinel, inferred topic,
implicit team-of-one credential, or automatic key creation on these paths.
`pile net identity` is the explicit key-initialization command.

## Validity is not revocation

Optional inclusive bounds let a credential be not-yet-valid or expire. They do
not create mutable policy, renewal, retraction, negative grants, an admin
hierarchy, or a distinguished current-membership head. An unbounded valid
credential remains valid under the same served trust root.

Ending authority before a signed upper bound is therefore an epoch or serving
policy change outside this kernel: move to a successor team/root/key epoch or
stop serving the old trust root. That explicit cost keeps verification local,
portable, claim-directed, and independent of storage enumeration.
