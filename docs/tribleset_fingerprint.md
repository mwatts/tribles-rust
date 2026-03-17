# TribleSet Fingerprints

`TribleSetFingerprint` is an O(1) cache key derived from the PATCH root hash used
for `TribleSet` equality. It is stable within a process but not across runs
because PATCH uses a randomized hash key.

Use this fingerprint for UI or in-memory caches where you want to skip rebuilding
work derived from a `TribleSet`. If you need a persistent identifier that is
stable across processes, derive a `Handle<Blake3, SimpleArchive>` from the
canonical `SimpleArchive` representation instead (at O(n) cost).
