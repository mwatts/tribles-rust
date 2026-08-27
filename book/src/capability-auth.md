# Direct Capability Proofs

TribleSpace authorization is one direct, portable proof. Semantic restrictions
live in content-addressed claim blobs; principals and signatures live in a
compact native proof:

```text
K0 (S0 C0 K1) (S1 C1 K2) ... (Sn Cn Kn+1)
```

`K0` is an externally chosen Ed25519 trust root. Each `Si` is a signature by
`Ki`, `Ci` is the exact BLAKE3 handle of a claim blob, and `Ki+1` is the next
principal. The verifier also receives the expected final key. Nothing is
inferred from possession, storage enumeration, append order, or a mutable
membership head.

## Keyless claims

A `CapabilityClaim` is one closed canonical `SimpleArchive` containing:

| Field | Meaning |
|---|---|
| action | one exact, uninterpreted 128-bit operation ID |
| resource | one exact opaque 32-byte resource identity |
| mode | `Invoke`, `Delegate`, or `InvokeAndDelegate` |
| parent | zero or one exact parent **claim** handle |
| validity | zero or one inclusive TAI interval |

Claims contain no public key and no signature. A root claim has no parent; each
later claim names the immediately preceding claim. The same semantic claim DAG
can therefore be used in distinct principal paths without changing claim
identity.

Actions and resources are exact atoms. There are no wildcards, implicit action
hierarchies, or ambient resource namespaces in the kernel. Applications define
the conversion from a concrete resource to its 32-byte identity.

## The native proof

The canonical proof body is `K0 (S C K)+`: one 32-byte root followed by one or
more 128-byte edges. There is no count, padding, or alternate field ordering in
the body. Every edge signature covers:

```text
"triblespace.capability.proof-edge\0"
|| 1:u32be
|| issuer_key
|| claim_handle
|| delegate_key
```

Binding both keys and the exact claim handle prevents key substitution,
cross-claim replay, and path splicing. Ed25519 verification is strict. The
proof ID is BLAKE3 over the complete canonical body, so the same proof has one
stable lookup identity in memory, a pile, or another store.

`CapabilityProofStore` is a grow-only set of these native proofs. It supports
insertion, deterministic enumeration, and exact lookup by proof ID. It does
not search by key or claim and it grants no authority merely because a proof
is present.

## Verification is a meet

`CapabilityProofBundle::verify` takes four explicit boundary values:

- the external trust-root key;
- the expected leaf key, normally authenticated by the transport or named by
  the collection commit;
- the exact verification instant; and
- the requested action/resource atom and minimum mode.

It then checks the native signatures, hashes and parses the ordered claim
blobs, and evaluates the root-to-leaf restrictions:

1. the first claim has no parent;
2. each later claim names the previous claim handle;
3. the effective parent mode contains `Delegate` before another edge follows;
4. every action/resource atom is exactly equal;
5. modes combine by bit intersection; and
6. bounded validity intervals combine by inclusive intersection and contain
   the supplied instant.

This is attenuation by meet, not a syntactic “child must be narrower” rule. A
child that repeats a wider mode cannot restore a bit removed earlier; it simply
adds no restriction for that bit. An empty atom, mode, or validity meet rejects
the proof. A valid prefix cannot stand in for a descendant because verification
also checks the expected leaf key.

The result reports the effective atom, mode, validity interval, leaf claim,
leaf key, and proof ID. A holder may extend it only when its effective mode
still delegates, the signing key equals the verified leaf, and the child names
the exact leaf claim.

## Portable bundles

A `CapabilityProofBundle` carries the native proof together with the exact
claim blobs in root-to-leaf order. Its bounded version-1 transport form is:

```text
version:u8 = 1
step_count:u8
proof: 32 + step_count * 128 bytes
repeat step_count times:
    claim_length:u16be
    claim:bytes
```

The count is nonzero and at most 255. Claim lengths are bounded by the closed
canonical claim shape; decoders reject truncation, trailing bytes, noncanonical
lengths, malformed keys, and oversized outer frames before treating the bundle
as evidence. The bundle is self-contained for one verification round trip.
Possessing it does not authorize a different key because the proof and caller
both bind the expected leaf.

## Storage and lifetime

Claims are ordinary blobs. The native proof record is the direct lifetime edge
for its claim closure: conservative collection preserves every canonical proof
record, and a proof whose signatures verify makes each resident claim handle an
exact direct root. Every ancestral claim is already named explicitly by the
proof, so the collector does not scan opaque claim values or follow parent
handles recursively. A missing claim remains missing and can be fetched later;
an invalidly signed proof roots no blob. Trust-root selection and semantic
claim verification remain caller responsibilities, not garbage-collector
policy.

There is no second retention collection. Storing a proof and its claims is
enough. The storage layer publishes claim blobs before the proof record so an
observer never mistakes a partially written local bundle for complete local
evidence.

## CONNECT

For direct TribleSpace RPC, the requested atom is:

```text
action   = ACTION_CONNECT
resource = configured trust-root public-key bytes
mode     = Invoke
leaf     = authenticated transport peer key
```

Protocol v12 carries one length-prefixed `CapabilityProofBundle` in the first
`OP_AUTH` stream and returns the server's own bounded bundle after the success
status. Each side supplies its configured team root, one current TAI instant,
and the other endpoint's TLS-authenticated key to verification. The client
checks that the connection identity is the endpoint it intended to dial before
sending its proof. Rejection closes the connection; a bounded accepted session
is discarded after either effective inclusive upper validity bound.

This first exchange is mutual authentication, not credential secrecy. The
initiator's proof necessarily reaches the dialed TLS endpoint before that
endpoint proves its own CONNECT capability. Because the proof is bound to the
initiator's exact key it is non-bearer evidence. The client sends it over TLS
to the identity it intended to dial, but the proof itself is not
cryptographically bound to that receiver key; the protocol is neither
confidential nor zero knowledge at the capability-bundle layer. No later proof,
element identity, query, or data request crosses until the reciprocal CONNECT
proof verifies.

CONNECT admits only the transport connection. It grants no collection
`ACTION_WRITE`, generic read policy, inventory disclosure, semantic trust,
retention, or blob availability. A second exact atom,
`ACTION_SYNC_TEAM(team_root)` in Invoke mode for the same transport key, must
be exchanged once through `INVENTORY_AUTH` before manifests, nodes, provider
operations, blob ranges, or even a known-hash `GET_BLOB` may be served. The
client verifies the returned server proof against the same TLS endpoint before
sending any useful request. These two proofs may have different delegation
paths and validity bounds. The team root also derives the gossip topic, but
receiving a wake frame grants neither CONNECT nor SYNC_TEAM authority.

There is no pre-auth fetch. The presenter sends each complete bundle inline.
After both authorizations, collection records converge independently of blob
policy: a `COMMIT` does not pull its referenced blobs in Demand mode, and an
observed handle does not create a WANT.

## WANT and gossip boundaries

Capability proof records are durable local set evidence, but the capability
layer defines no proof-specific gossip or WANT. Authorized team inventory may
union resident proofs like any other inert evidence; that still does not make
them active authority. Portable bundles are transferred explicitly when they
must authorize an operation, such as in an invite or one of the two connection
handshakes. Claim blobs use the ordinary authorized blob transport and may be
requested by their exact handles when local policy wants them.

This separation is intentional:

- proof presence is not authority;
- gossip rendezvous is not team membership;
- WANT is local demand, not authorization; and
- blob availability is not semantic validity.

## Team bootstrap

A team is named by its Ed25519 trust-root public key. `team create` issues two
root claims for the founder's key—CONNECT and SYNC_TEAM, both initially in
invoke-and-delegate mode—stores their claims and proofs, and reports both proof
IDs. The root key is created as a private durable file (or loaded from
`--root-key`) so it can remain offline without depending on terminal output for
recovery.

`team invite` loads one exact parent proof for each action, verifies current
delegation, extends both paths for the invitee, and writes one versioned
portable artifact. `team join` requires the expected team-root public key
separately, verifies both bundles against that external root and the invitee
key, then inserts all claims and both native proofs in one idempotent operation.
The artifact therefore never gets to nominate its own authority. `team show`
loads one exact proof ID and its named claims; it never scans for a roster or
chooses among paths implicitly.

Validity bounds are optional monotone restrictions, not mutable revocation.
Ending an unexpired grant requires changing the served trust root or another
explicit application epoch. That cost keeps the kernel local, portable, and
independent of a second authorization database.
