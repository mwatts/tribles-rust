# tribleset-bench

`tribleset-bench` measures an engine checkout and writes canonical benchmark
telemetry. The optional performance-fingerprint notebook is a reader for the
separate demand-curve `observations.tsv` receipt; it never runs inside the
measured process.

## Performance fingerprint

Run the embedded demonstration:

```sh
cargo run --manifest-path tribleset-bench/Cargo.toml \
  --features fingerprint-notebook \
  --bin performance-fingerprint -- --demo
```

Open a demand-curve receipt (a directory is resolved to
`observations.tsv`):

```sh
cargo run --manifest-path tribleset-bench/Cargo.toml \
  --features fingerprint-notebook \
  --bin performance-fingerprint -- path/to/observations.tsv
```

`PERFORMANCE_FINGERPRINT_TSV` is the environment-variable alternative. GORBIE
can render a deterministic headless receipt with its ordinary capture flags:

```sh
cargo run --manifest-path tribleset-bench/Cargo.toml \
  --features fingerprint-notebook \
  --bin performance-fingerprint -- --demo --headless \
  --out-dir /tmp/performance-fingerprint
```

The adapter groups rows into a subject
`(engine, engine_variant, backend, substrate, parallelism)` and preserves the
full subject × query-shape × scale × demand matrix. Header aliases normalize
older names such as `commit`, `storage`, `device`, `mode`, `query`, `limit`,
and `duration_ns` in memory; the source receipt is not migrated.

The primary curve is `c(k) = median T(k) / k`. `k=1` is time to first result.
`full` remains a distinct terminal iterator-exhaustion demand even when its
cardinality equals a numeric limit. Falling, flat, and rising curves therefore
read as amortizing, linear, and superlinear respectively. Construction has no
per-result divisor and appears only in the exact-value matrix.

Only valid measured cells enter plots. Every matrix cell is rendered as one of
`measured`, `missing`, `unsupported`, `error`, or `cardinality mismatch`.
Identity records validate full-drain cardinality and digest agreement across
subjects. When the canonical `abba_position` and `repetition` fields are
present, each measured cell must cover the subject's complete observed slot
grid and each full drain must carry the complete identity-position grid;
partial data stays visibly `missing` instead of entering a median. Rayon
construction and numeric limits at or beyond an agreed full cardinality are
protocol omissions, so the adapter marks them `unsupported` instead of
inventing missing measurements.

### Receipt boundary

The raw TSV is canonical. This first notebook intentionally has no persistent
schema of its own. Its expected axes are the union of parseable receipt rows,
so a shape or scale absent from the entire file cannot yet be reconstructed.
Likewise, a producer that aborts before writing a structured status can be
shown as incomplete, but its exact stderr/exit cause is unavailable. A future
manifest/invocation receipt can supply those two kinds of negative evidence
without changing the visualization model.
