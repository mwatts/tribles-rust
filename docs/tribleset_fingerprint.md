# TribleSet Fingerprints

`TribleSetFingerprint` is an O(1), opaque cache token derived from the PATCH
root aggregate used for `TribleSet` equality. Before crossing the public API,
that internal XOR aggregate passes through a domain-separated SipHash-2-4 PRF
under a second process-random key. `as_u128`, `Debug`, and `Hash` therefore
expose only the blinded token, never PATCH's linear maintenance value.

Equal sets produce equal tokens within a process. A token match is still a
128-bit cache hint rather than proof of equality, and tokens intentionally
change across runs. `TribleSetFingerprint::EMPTY` remains the distinguished
empty-set value.

Use this fingerprint for UI or in-memory caches where you want to skip rebuilding
work derived from a `TribleSet`. If you need a persistent identifier that is
stable across processes, derive a `Handle<Blake3, SimpleArchive>` from the
canonical `SimpleArchive` representation instead (at O(n) cost).
