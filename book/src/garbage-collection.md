# Garbage Collection and Forgetting

Stores grow as signed commits, reproducible derived artifacts, and user blobs
accumulate. Because every blob is content addressed and immutable, nothing is
overwritten in place. A node periodically _forgets_ local bytes which its
retention policy no longer owns.

Forgetting is deliberately conservative. It only removes local copies, so
re-synchronising from a peer or satisfying a later WANT may restore them.
Forgetting therefore complements the monotone model: an assertion is not
semantically retracted merely because one node drops a physical copy.

The main challenge is distinguishing authority from useful-but-reproducible
cache work. A digest mentioned in a record is not automatically a lifetime
edge.

## Direct and Recursive Policy Roots

Not every hash written inside a retained blob is an ownership edge. Collection
ledger records are the important counterexample: a `MERGE` names its inputs and
result to state an algebraic equation, but retaining that record should not by
itself pin every historical physical input forever. `RetentionRoots` therefore
has two sorts:

- a **direct** root retains exactly the named blob; and
- a **recursive** root retains the blob and all resident descendants found by
  conservative traversal.

Strong collection retention follows signed ground truth, independently of
current WRITE admission. For every strictly verified native `COMMIT`, the
descriptor, signed data, and metadata handles are recursive roots, so all of
their resident attachments remain owned. This conservatism matters because a
later positive grant may activate an already-resident commit. The descriptor is
the canonical collection anchor, representation, optional source mapping, and
independent READ/WRITE policy description encoded as a `SimpleArchive`; its
32-byte content handle is the `CollectionHandle` carried by the commit. The native commit record
is preserved by `CollectionStore` rather than represented as a blob root.
Planning an explicitly selected authoritative view fails if any required
descriptor, data, or metadata blob is absent.

Unsigned `MERGE` and `DERIVE` records are reproducible cache work. They add no
strong roots merely because their equations are present; their named inputs,
results, and otherwise-unowned descriptor blobs may all be collected.
Conservative Pile and Yard rewrites preserve the equation records themselves
as immutable ledger evidence, but that preservation creates no blob ownership
edge. All successful computations are stored first; Yard/GC alone later chooses
which materializations remain resident. This boundary also means the strong
planner needs neither a requested-view set nor validation-verdict machinery:
admitted strictly verified commits determine the collections that are retained.

Rank9 acceleration uses the same rule through ordinary collection records.
`Rank9AcceleratedSuccinctArchiveBlob` is a Merkle root whose first 32 bytes name
its portable raw `SuccinctArchiveBlob` child. A raw-to-accelerated `DERIVE` may
survive a conservative ledger rewrite, but the unsigned equation still
manufactures no ownership. Raw Succinct and Rank9-accelerated members are both
directly joinable. A Rank9 join depends on the exact raw union being resident;
when storage materializes that raw `MERGE`, the accelerated `MERGE` can follow.
If policy retains the accelerated root, ordinary child traversal also retains
its named raw child.

Read-only materialization accepts only a complete resident source-bound member.
Cover resolution checks the encoding's required representation closure before
selecting a root, so an accelerated root with a missing raw child is unavailable
and a complete finer cover may replace it. The cover-aware view still loads the
embedded raw child through the same snapshot and validates the exact raw/index
pair before constructing a transient query runtime. Normal `ensure` publication
stores every dependency before the member that names it and publishes the
corresponding ordinary `MERGE` or `DERIVE` last. No accelerator-specific
retention relation or hidden root is involved.

The resulting roots compose with both storage paths. Yard's `collect` and
`compact` accept explicit policy roots in addition to the native collection
roots they discover. Both Yard collection and `Pile::rewrite_retained_into`
strictly verify a native `COMMIT` signature before its fields can add implicit
roots. They
preserve every immutable record, including invalid and partially synchronized
records, but recursively retain only dependencies named by valid commits which
are resident in the relevant Pile snapshot or live in the Yard. An absent
dependency therefore remains available for later synchronization instead of
permanently poisoning local retention. Caller-supplied `RetentionRoots` keep
their existing backend semantics; in particular, a retained Pile rewrite still
fails loud when an explicitly selected blob is absent.

For migration safety, a retained Pile rewrite also recreates the exact immutable
legacy pin snapshot it observed. A resident pin head is a recursive root; a
dangling pin remains dangling and its mapping is still recreated without
manufacturing or demanding the absent blob. That is physical preservation, not
a current publication or retention API. Legacy V3 collection records are
different: their
16-byte definition identities predate descriptor handles, so they are
preserved byte-for-byte as
inert physical evidence but grant no current collection authority and own no
blobs. Capability proofs have their own direct native root. Conservative
rewrites preserve every canonical proof record. When the proof's native
`K0 (S C K)+` signatures verify, each resident claim handle in that proof is a
direct root. The proof already enumerates the complete ancestry, so claim
contents are deliberately not scanned for child handles. Missing claims remain
missing, and an invalidly signed proof roots nothing. This signature check
establishes only a safe local lifetime edge: semantic authority still requires
an external trust root, expected leaf, instant, and exact request.
WANT records are an explicit rewrite choice. Preserving them copies each exact
grow-only request but does not promote a requested blob to an ownership root;
dropping them omits the markers entirely. Yard collection may trim evictable
blob demand from its in-memory budget, and reclaim then records only the
surviving set. No negative record is appended, so operation wants remain
durable and stale pile concatenation cannot retract demand.

`RetentionRoots` is deliberately a pure, ephemeral plan rather than a retained
collection registry. A caller selecting one semantic view must rediscover its
records, apply its explicit open or capability admission, and supply a fresh
plan for the commits it selected. Ordinary Pile and Yard rewrites do not use
that narrower admission decision: they independently apply the conservative
rule above, preserving every native record and recursively retaining the
resident descriptor, data, and metadata closure of every strictly verified
current `COMMIT`.

Opaque records form a harder boundary: their bytes have a known span and
ordinary replay can safely project them away, but the reader cannot know
whether their former or future semantics own known blobs. This includes unknown
generic-envelope kinds and both recognized encodings of the retired local-cell
experiment. `Pile::rewrite_retained_into` and Yard collection, compaction, and
reclaim therefore refuse before changing destination bytes or live sets when
any opaque record is present. Upgrade to tooling that understands an unknown
kind, or explicitly migrate retired records, before performing destructive
retention.

## Conservative Reachability

Canonical archives contain fixed 64-byte tribles whose value half is one
32-byte inline value. The generic walker treats aligned 32-byte chunks as
candidate handles and follows those which name resident blobs. This may keep an
accidental extra blob, but with 256-bit handles it will not plausibly discard a
real reference. Opaque attachments normally behave as leaves because their
chunks do not name another resident object.

The retention procedure is therefore:

1. enumerate native collection and capability-proof records from one observed
   store view;
2. strictly verify each commit and each proof's direct signatures before their
   fields gain retention authority;
3. add the resident descriptor, data, and metadata of valid commits as
   recursive roots, and each resident claim explicitly named by a validly
   signed proof as a direct root;
4. add caller-selected direct or recursive policy roots;
5. recursively mark resident candidate handles; and
6. rewrite or evict everything outside that conservative live set while
   preserving the immutable record ledger.

`MERGE` and `DERIVE` records remain useful even when their result bytes are
absent: a later resolver can request the exact recorded result from peers or,
if no copy remains, recompute and republish it. Existing resident results are
never recomputed merely to prove their equations.

## Operational Tips

- **Schedule forgetting deliberately.** Trigger it after large merges or
  imports rather than on every commit so you amortize the walk over meaningful
  changes.
- **Watch available storage.** Because forgetting only affects the local node,
  replicating from a peer may temporarily reintroduce forgotten blobs. Consider
  monitoring disk usage and budgeting headroom for such bursts.
- **Keep a safety margin.** If you are unsure whether a handle should be
  retained, include it in the root set. Collisions between 32-byte handles are
  effectively impossible, so cautious root selection simply preserves anything
  that might be referenced.

The essential safety rule is asymmetric: uncertainty keeps bytes. Collection
can sacrifice space, never recoverability of an explicitly owned resident
closure.
