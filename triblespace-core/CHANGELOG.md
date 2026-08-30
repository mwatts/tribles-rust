# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Make every collection member an ordinary typed `Blob<E>`.
  `CollectionEncoding` validates that blob and may expose one directly
  materializable canonical join; encodings whose physical compaction belongs
  in another lattice keep multi-member covers instead. `CollectionMapping`
  maps blobs to blobs, and exact derivation stores a selected source member
  before its target image and ordinary `DERIVE` record.

- Add direct typed collection encodings, covers, and logical cover
  views. `CollectionEncoding` attaches canonical validation and an optional
  directly materializable join to the member encoding; `Collection<E>` and
  `Cover<E>` retain that encoding through the public API; signed commits accept
  authored `Fragment` values in `SimpleArchive` source collections, while
  typed materialization works for non-`SimpleArchive` collections. Derived
  descriptors link a concrete mapping entity carrying its algorithm and
  concrete parameters, while exact derived lifecycles bind one
  `CollectionMapping<Source, Target>` whose law is a join homomorphism.

- Add top-level `capability`, a direct authorization kernel. Keyless canonical
  `SimpleArchive` claims carry one exact action/resource atom, invoke/delegate
  mode, optional inclusive validity interval, and optional parent claim handle.
  Canonical native proofs use `K0 (S C K)+`; every strict Ed25519 signature
  binds issuer key, exact claim handle, and delegate key, and BLAKE3 over the
  complete proof is its stable identity. A bounded portable bundle carries the
  exact ordered claims. Verification takes an external trust root, expected
  leaf, explicit instant, and request, then computes the claims' atom, mode,
  and validity meet without ambient lookup.
- Add `CapabilityProofStore` to `MemoryRepo`, `Pile`, and `Yard` as a native
  grow-only proof set with deterministic enumeration and exact proof-ID lookup.
  Pile records use bounded canonical framing. Conservative collection preserves
  proof records and treats every claim named by a signature-valid proof as an
  exact direct root. It never scans opaque claim values as child handles, and
  proof presence alone grants no authority.

### Changed

- Clean-cutover the unpublished Rank9 API and identities.
  `Rank9AcceleratedSuccinctArchiveBlob` is now an ABI-qualified Merkle-root
  `CollectionEncoding` whose first 32 bytes name its portable raw
  `SuccinctArchiveBlob` child. `RawToRank9AcceleratedMappingV1` maps raw members
  through ordinary `DERIVE` records. Raw Succinct members remain directly
  joinable; accelerated roots have no direct `MERGE`, so exact maintenance
  compacts raw first and derives the matching query-ready encoding. Cover-aware
  views pull the named child through the immutable store snapshot and validate
  the complete raw/index pair. The former `Rank9MappingV1`/`RANK9_MAPPING_V1`,
  intermediate `Rank9SidecarMappingV1*`, old blob names, and their obsolete id
  family have no compatibility aliases. The separate mapping-evidence record
  kind and store surface were removed after a scan found no live records in
  known piles; derived artifacts may be recomputed.

- Treat handles cached by typed `Blob` values returned from `BlobStoreGet` as
  trusted content identities. Collection operations retain structural and
  semantic validation at real ingress boundaries while deleting duplicate
  rehashes and post-write rereads of values produced or loaded in-process.

- Replace split Reader and revision APIs with `SnapshotSource` and one coherent
  immutable `StoreSnapshot`. Blob access, collection records, capability
  proofs, and PEER evidence are frozen together. `changes_since` defaults to
  conservative full invalidation, while `MemoryRepo`, `Pile`, and `Yard`
  compare persistent component PATCHes directly. `BLOBS` covers membership,
  metadata, and retrievability; Pile's lineage-local root-sharing comparison
  catches same-handle backing replacement while ignoring unrelated appended
  records, even though semantic PATCH equality intentionally hashes keys, not
  attached storage offsets. Yard retention planning now uses one snapshot for
  opaque-record refusal, live membership, commits, and proofs, and physical
  rewrites preserve peer evidence alongside collections, proofs, and offers.

## [0.41.4] - 2026-05-17

Lock-step bump alongside the trailing-dot-leak +
connection-reuse fixes in `triblespace-net` / `trible`. No
source changes in `triblespace-core`. See the workspace
[`../CHANGELOG.md`](../CHANGELOG.md) for the full release notes.

## [0.41.3] - 2026-05-17

Lock-step bump alongside the trailing-dot relay-URL fix in
`triblespace-net` / `trible`. No source changes in
`triblespace-core`. See the workspace
[`../CHANGELOG.md`](../CHANGELOG.md) for the full release notes.

## [0.41.2] - 2026-05-17

Lock-step bump alongside the address-symmetry work in
`triblespace-net` / `trible`. No source changes in
`triblespace-core`. See the workspace
[`../CHANGELOG.md`](../CHANGELOG.md) for the full release notes.

## [0.41.1] - 2026-05-17

Lock-step bump alongside the EndpointTicket-everywhere work
in `triblespace-net` / `trible`. No source changes in
`triblespace-core`. See the workspace
[`../CHANGELOG.md`](../CHANGELOG.md) for the full release notes.

## [0.41.0] - 2026-05-16

Lock-step bump alongside the iroh 0.98 family upgrade in
`triblespace-net`. No source changes in `triblespace-core`.
See the workspace [`../CHANGELOG.md`](../CHANGELOG.md) for
the full release notes.

## [0.39.0] - 2026-05-11

The canonical-attribute-id + bounded-path-estimation release.
See the workspace [`../CHANGELOG.md`](../CHANGELOG.md) for the
full release notes on dynamic-name attribute id derivation,
the IRI BlobEncoding, `metadata::iri`, `Attribute::from_iri`, the
`MemoryBlobStore::union` structural merge, and the `Workspace`
`local_blobs → staged` rename.

### Path-query: bounded-depth closure estimation
- **`estimate_from`'s closure-fallback no longer full-materialises**
  the result set
  (`triblespace-core/src/query/regularpathconstraint.rs`). The
  previous fallback ran `eval_from(set, body, start).len()` —
  paying the full cost of computing the closure just to measure
  its size. The new `bounded_eval_from` helper caps closure BFS
  at `RPQ_ESTIMATE_DEPTH = 5` levels, matching Karalis et al.
  ESWC 2024 §4.3's "default estimation": bounded depth →
  bounded estimate cost, sufficient for variable ordering.
  Non-closure expressions don't consume depth; the bound only
  fires on Plus/Star iteration steps. Nested closures multiply
  (`Plus(Plus(q))` runs the inner Plus to depth 5 for each of
  the outer's 5 steps — `O(depth^k)` for closure-nesting
  depth `k`), which the doc comment flags. Shallow estimation
  (the constant-time per-attribute count from the segmented
  index) was already in place; this commit closes the remaining
  gap where shallow doesn't apply.

## [0.38.0] - 2026-05-07

Lock-step bump alongside the team-rooted-gossip release in
`triblespace-net` / `trible`. No source changes in
`triblespace-core`. See the workspace
[`../CHANGELOG.md`](../CHANGELOG.md) for the full release notes.

## [0.37.0] - 2026-05-06

First per-crate CHANGELOG. Earlier `triblespace-core` releases
are documented at the workspace level in
[`../CHANGELOG.md`](../CHANGELOG.md).

### Added
- **`PathOp::Optional` (`(p)?`) primitive** in the path-query
  language. `Optional(p)` matches zero-or-one applications of
  `p`; semantically `Union(Identity, p)` but recognised inline
  so the zero-step branch reuses the bound start node directly
  instead of materialising every node as an `Identity`
  candidate. Same shape as the `Star` arm but with the zero-
  step alone (no transitive frontier). Plus a `from_postfix`-
  time normalisation pass that distributes `Optional` and
  `Union` out of `Concat` via the standard rewrites
  (`a / b? ↔ a | (a / b)`, `(a | b) / c ↔ (a / c) | (b / c)`,
  etc.) — without it, the typical WDBench shape
  `Concat(Attr, Optional(Attr))` (`p / q?`) would hit the
  `build_constraint` `unreachable!()` arm. Macro syntax in
  `path!` (`(p)?`) is the follow-up; until then callers
  construct `PathOp::Optional` postfix-style via
  `RegularPathConstraint::new`. Two proptests cover the
  standalone `(p)?` boundary case and the `p / p?` Concat-
  with-Optional case post-normalisation.
- **`PathOp::Inverse` (`^p`) primitive** in the path-query
  language. `^attr` reverses the direction of an attribute
  edge (VAE-index lookup yielding entity bytes, mirroring the
  existing forward `eval_attr` / EAV-index path). Compound
  expressions push down via the standard reversal rewrites
  (`^(a/b) ↔ ^b/^a`, `^(a+) ↔ (^a)+`); double negation
  (`^^a → a`) cancels at `from_postfix`-time. Macro syntax in
  `path!` (`^p`) is the follow-up; until then callers
  construct `PathOp::Inverse` postfix-style via
  `RegularPathConstraint::new`. Two proptests cover
  standalone `^link` and `(^p / p)+` (mid-path inverse inside
  a Plus loop).
- **`Universe::search_range(min, max) -> Range<usize>`**, plus
  the underlying `search_lower(v)` / `search_upper(v)`
  primitives. `O(log n)` half-open code range over a monotonic
  universe; default impls fall through to a binary search via
  `Universe::access`. Implementations with a flat sorted slice
  override to skip the virtual-call overhead.
- **`SuccinctArchive::value_in_range`** constraint exploits
  the new universe primitive: `O(log n + K)` proposals over
  range-bounded values, where `K` is the number of distinct
  in-range codes that actually appear on the indexed axis.
  Composable with `pattern!` / `find!` / `and!`. Combined with
  `enumerate_in_range` (the bounded variant of
  `enumerate_domain`), it gives the engine a real range-query
  primitive without scanning the full value column.
- **`repo::capability` runnable doctests** on every primary
  public function: `build_capability`, `verify_chain`,
  `build_revocation`, `extract_revocation_pairs`,
  `VerifiedCapability` (covering `permissions`,
  `granted_branches`, `grants_read`, `grants_read_on`).

### Changed
- **`SuccinctArchive`'s value-axis enumeration** routes
  range-bounded queries through `Universe::search_range`
  rather than enumerating the full domain and post-filtering.
  Same result; `O(log n + K)` instead of `O(n)`.
- **Workspace doc warnings cleaned** — 9 stale intra-doc-link
  warnings in `Universe` trait method docs and the
  `succinctarchive` module fixed (`[Self::search]`,
  `[Self::access]`, `[Self::search_lower]`,
  `[Self::enumerate_domain]` etc.). `cargo doc -p
  triblespace-core --no-deps` is now warning-free.
