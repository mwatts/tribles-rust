# triblespace-search

Content-addressed BM25 + HNSW indexes on top of
[triblespace](https://github.com/triblespace/triblespace-rs) piles.

Two blob types, loaded zero-copy via [anybytes] and (eventually)
[jerky]:

- **`SuccinctBM25Index`** — lexical / associative retrieval. Terms
  are 32-byte triblespace `Value`s, so the index handles text
  search, entity co-occurrence, and tag weighting with the same
  schema.
- **`SuccinctHNSWIndex`** — approximate k-nearest-neighbour over
  caller-supplied embeddings.

Both indexes are rebuilt-and-replaced (no mutation). The resulting
blob handle is persisted wherever the caller likes — branch
metadata, commit metadata, a plain trible, or an in-memory cache.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design.

[anybytes]: https://github.com/triblespace/anybytes
[jerky]: https://github.com/triblespace/jerky

## Status

**Pre-alpha.** Public API shapes are still settling. Version 0.0.0
until the first non-scaffold release.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
