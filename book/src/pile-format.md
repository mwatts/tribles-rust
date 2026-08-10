# Pile Format

The on-disk pile keeps blobs, native collection records, pins, local cells,
and wants in one append-only file. The write-ahead log *is* the database: all indices are
reconstructed from the bytes already stored on disk. This design avoids
background compaction, manifest management, or auxiliary metadata while still
providing a durable content-addressed store for local repositories. The pile
file is memory mapped for fast, zero-copy reads and can be safely shared
between threads because existing bytes are never mutated—once data is
validated it remains stable.

While large databases often avoid `mmap` due to pitfalls with partial writes and
page cache thrashing [[1](https://db.cs.cmu.edu/mmap-cidr2022/)], the pile's
narrow usage pattern keeps these failure modes manageable. Appends happen
sequentially and validation walks new bytes before readers observe them, so the
memory map never exposes half-written records.

## Record model: uniform 256-byte records (V3)

Every record the pile writes today — blob, native collection definition,
collection commit, collection merge, collection derive, branch (pin) head,
branch tombstone, local-cell value, local-cell tombstone, want assertion, or
want retraction — uses the **V3** layout:
a fixed **256-byte header**, followed (for blobs) by the payload, padded so the
whole record is a **256-byte multiple**. Want records retain their historical
weak-pin/weak-unpin magic markers for byte compatibility; those are physical
format names, not the public storage model. This uniformity is load-bearing:

- **Position independence.** Blob data starts at the constant
  `record_start + 256`; there is no offset-derived padding. A record means
  the same thing at any offset, so records survive relocation and
  `cat a.pile >> b.pile` is a valid merge of two piles.
- **Alignment for free.** Because every record is a 256-byte multiple, a
  pure-V3 pile stays 256-aligned throughout under the atomic lock-free
  append — every blob's payload lands on a 256-byte boundary, satisfying GPU
  storage-buffer binding requirements (CUDA / Metal
  `min_storage_buffer_offset_alignment`) for zero-copy aliasing.
- **Cache-friendly headers.** Each header begins on a cache-line boundary and
  admits safe typed views with the `zerocopy` crate.

Reserved header bytes are zeroed and are **not** part of the content hash;
per-record metadata belongs in tribles, not in the header, so identical bytes
never fork into distinct blobs.

The reader still accepts the original **V1** records (64-byte-aligned blob,
branch, and tombstone layouts — see [Legacy V1 records](#legacy-v1-records)),
so piles written before V3 read byte-identical with no migration step. New
writes are always V3. The skew direction to watch is the other one: **binaries
from before V3 treat V3 records as unknown and fail loud with
`ReadError::CorruptPile`** — they do not truncate anything. When an old binary
reports corruption on a pile a newer binary wrote, the fix is to upgrade the
binary, never to "repair" the pile.

## Design Rationale

This format emphasizes **simplicity** over sophisticated on-disk structures.
Appending new records rather than rewriting existing data keeps corruption
windows small and avoids complicated page management. Storing everything in a
single file makes a pile easy to back up, replicate over simple transports, or
merge by concatenation, while still allowing it to be memory mapped for fast
reads. Internally the pile tracks an `applied_length` watermark; offsets below
this boundary are known-good and only the tail beyond it is rescanned when
refreshing state.

## Operational workflow

1. **Open the file.** `Pile::open` builds the struct around a `File` handle
   and `memmap2` mapping. It does not read any records yet (and it does not
   create missing files — create the file explicitly for a fresh pile).
2. **Load and validate.** `refresh` acquires a shared lock, walks bytes beyond
   `applied_length`, and rebuilds the blob, collection-record, and pin indices
   in memory. It **fails
   loud** on a corrupt or torn tail (`ReadError::CorruptPile { valid_length }`)
   and never mutates the file. Callers rarely need to invoke it directly:
   `reader`, `records`, `pins`, `head`, `update`, `cell`, and `set_cell` call
   `refresh` internally
   before they inspect or apply records, so external writers are visible
   without a standalone scan.
3. **Amputate only when asked to.** `amputate` is the explicit, opt-in repair
   path: it re-runs validation under an exclusive lock and truncates the file
   back to the last valid record, discarding a torn tail left by a crash. It
   is deliberately **not** part of the normal open sequence — implicit repair
   under version skew is a silent data-loss hazard (an old binary would "eat"
   every newer-format record past the first one it misreads as corruption).
   The `trible pile amputate <path>` command wraps it for operators.
4. **Append new records.** `put` (through the `BlobStorePut` trait),
   `CollectionStore::insert`, local-cell replacement, and pin update helpers
   extend the file. Each
   append immediately feeds the bytes back through the record scanner so
   in-memory indices stay synchronised without waiting for a manual `refresh`.
   Blob records use a single `write_vectored` call; fixed-width collection and
   pin records use one append of their 256-byte header.
   Records larger than ~1&nbsp;GiB can't be appended in a single atomic
   `writev` because kernel `write_vectored` calls cap at `INT_MAX` bytes on
   macOS and `MAX_RW_COUNT` (~2&nbsp;GiB) on Linux. In that case `put` takes
   an exclusive file lock and issues plain `write_all` calls — still
   append-only, still repairable by an explicit `amputate` if a crash leaves a
   partial tail, but serialised against other writers for the duration of the
   append.
5. **Read through a snapshot.** `reader` clones the memory map and PATCH
   indices into a `PileReader`, yielding iterators and metadata lookups that
   can execute without further locking.

This lifecycle keeps pile usage predictable: open → operate (operations
refresh as they run) → hand out read-only readers. If a process wants to scan
for new appends between operations (for example, a background monitor that is
not issuing `reader` or pin calls), it can explicitly call `refresh` to pick up
external writers without blocking them for long. If corruption is ever
reported, surface it to the operator; truncating is a decision, not a default.

## Immutability Assumptions

A pile is treated as an immutable append-only log. Once a record sits below a
process's applied offset, its bytes are assumed permanent. The implementation
does not guard against mutations; modifying existing bytes is undefined
behavior. Only the tail beyond the applied offset might hide a partial append
after a crash, so validation and repair only operate on that region. Each
record's validation state is cached for the lifetime of the process under this
assumption, avoiding repeated hash verification for frequently accessed blobs.

Hash verification only happens when blobs are read. Opening even a very large
pile is therefore fast while still catching corruption before data is used.

Every record begins with a 16&nbsp;byte magic marker that identifies its kind.
The sections below illustrate the layout of each type.

## Usage

A pile typically lives as a `.pile` file on disk. Repositories open it through
`Pile::open` and load it with `refresh` (directly or via the first operation
that refreshes internally). Multiple threads may share the same handle thanks
to internal synchronisation, making a pile a convenient durable store for
local development. Blob appends use a single `O_APPEND` write. Each handle
remembers the last offset it processed and, after appending, scans any gap left
by concurrent writes before advancing this `applied_length`. Writers may race
and duplicate blobs, but content addressing keeps the data consistent. Each
handle tracks hashes of pending appends separately so repeated writes are
deduplicated until a `refresh`. Pin updates only record the referenced hash and
do not verify that the corresponding blob exists in the pile, so a pile may act
as a head-only store when blob data resides elsewhere.

```rust,ignore
use std::error::Error;
use std::path::PathBuf;

use anybytes::Bytes;
use triblespace::prelude::*;
use triblespace::core::repo::pile::ReadError;
use triblespace::core::repo::BlobStoreMeta;

fn add_blob(bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from("data.pile");
    let mut pile = Pile::open(&path)?;
    // Load and validate the existing records. This FAILS LOUD on a corrupt
    // or torn tail and never mutates the file. Repair is a separate,
    // explicit decision (`Pile::amputate` / `trible pile amputate`), typically
    // made by an operator after checking that the binary isn't simply older
    // than the pile's records.
    match pile.refresh() {
        Ok(()) => {}
        Err(err @ ReadError::CorruptPile { .. }) => return Err(err.into()),
        Err(other) => return Err(other.into()),
    }

    // Insert a blob and obtain a handle pointing at the on-disk bytes.
    let handle = pile.put(Bytes::from_source(bytes.to_vec()))?;

    // Readers operate on a snapshot cloned from the pile's mmap.
    let reader = pile.reader()?;
    if let Some(meta) = reader.metadata(handle)? {
        println!("stored {} bytes at {}", meta.length, meta.timestamp);
    }
    drop(reader);
    pile.close()?;
    Ok(())
}
```

This pattern illustrates the typical flow: open, load with `refresh`, rely on
the built-in refreshes performed by `reader` and pin helpers, mutate via
`put`, then hand the `PileReader` snapshot to read-only consumers. Updating
pin heads requires a brief critical section—`flush → refresh → lock →
refresh → append → unlock`—so a caller observes a consistent head even when
multiple processes contend for the same file descriptor. `refresh` acquires a
shared lock so it cannot race with an explicit `amputate`, which takes an
exclusive lock before truncating a corrupted tail.

Filesystems lacking atomic `write`/`vwrite` appends—such as some network or
FUSE-based implementations—cannot safely host multiple writers for records
below the `~1&nbsp;GiB` atomic-write threshold and are not supported in that
mode. (Records above the threshold use the exclusive-lock fallback and don't
rely on filesystem atomicity.) Using an atomicity-lacking filesystem for
small records risks pile corruption.

## Bounded refresh snapshots

Replay snapshots the observed file length once per refresh and decodes exactly
that bounded prefix. Shared-lock atomic writers may append after the snapshot;
those records are intentionally picked up by the next refresh. Post-write
readback still observes the live length while looking for the caller's own
record. This avoids a metadata syscall per record without weakening exact
torn-tail offsets or amputation's exclusive retry.

`PileReader` receives one persistent PATCH snapshot when it is created. Later
refreshes can extend the pile's copy without changing existing readers, and
`blobs_diff` can compare two snapshots through PATCH's structurally shared set
difference instead of enumerating either complete index.

Tools that need the raw log rather than the collapsed state—reflogs,
consolidation, forensics—should use
[`PileRecords`](../../src/repo/pile.rs), an iterator over every record in a
pile file in log order. It shares its decoder with the replay path described
above, so it understands every record format ever written; do not hand-roll a
parser against the layouts documented in this chapter. An unknown or
truncated record is reported as an error, never skipped.

## Blob Records

```text
            ┌────16 byte───┐┌8 byte┐┌8 byte┐┌────────────32 byte───────────┐┌───192 byte───┐
          ┌ ┌──────────────┐┌──────┐┌──────┐┌──────────────────────────────┐┌──────────────┐
 header   │ │ blob marker  ││ time ││length││             hash             ││  reserved 0s │
 (256 B)  └ └──────────────┘└──────┘└──────┘└──────────────────────────────┘└──────────────┘
            ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
 payload    │        bytes, post-padded so the record is a 256-byte multiple             │
            └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

Each blob record carries:

- **Magic marker** – identifies the record kind.
- **Timestamp** – milliseconds since the Unix epoch when the append occurred.
- **Payload length** – the unpadded byte length of the blob.
- **Hash** – the digest produced by the pile's hash protocol (BLAKE3 by
  default) and used as the blob handle.
- **Reserved** – zeroed padding to the fixed 256-byte header length; not part
  of the content hash.

The payload follows at `record_start + 256` and is post-padded to the next
256-byte boundary. The [Pile Blob Metadata](./pile-blob-metadata.md) chapter
explains how to query these fields through the `PileReader` API.

## Native Collection Records

`CollectionStore` is a grow-only set of canonical collection-calculus records.
The pile stores its four record kinds directly as fixed 256-byte V3 records;
they are **not blob records**, have no following payload, and carry no
insertion timestamp. They are also distinct from mutable branch pins and local cells
described below: collection records have no head, tombstone, or
last-writer-wins update. Their logical key is the record's intrinsic entity ID.

The magic markers below identify the compact pile representation. They are
wire-format markers, not the `metadata::tag` IDs found in the equivalent
canonical `SimpleArchive` entities.

| Kind | V3 magic marker | Exact byte layout |
|---|---|---|
| Definition | `3BE108504E4F5242FB24AA72D6D94CE1` | `0..16` marker, `16..32` scope ID, `32..48` representation ID, `48..64` recipe ID, `64..256` reserved zeros |
| Commit | `BB758AA6F79FBFC4D1958592A8956777` | `0..16` marker, `16..32` collection ID, `32..64` data digest, `64..96` metadata handle, `96..128` Ed25519 public key, `128..160` signature R, `160..192` signature S, `192..256` reserved zeros |
| Merge | `CC0108AC1DF4F335AFA856A529C42BE9` | `0..16` marker, `16..32` collection ID, `32..64` lower input digest, `64..96` higher input digest, `96..128` result digest, `128..256` reserved zeros |
| Derive | `07ECF056F6F015D94389FFF21F851480` | `0..16` marker, `16..32` source collection ID, `32..48` target collection ID, `48..80` input digest, `80..112` output digest, `112..256` reserved zeros |

Every reserved byte must be zero; a nonzero reserved byte makes replay fail as
corrupt rather than silently assigning meaning to a format extension. Merge
inputs are stored in lexicographic digest order (`low <= high`), so swapping
the two operands cannot create a second representation of the same
commutative equation.

The intrinsic record ID is deliberately absent from these headers. On replay,
the decoder reconstructs the exact canonical one-root entity from the stored
fields and its collection-record kind tag, then derives the root ID from that
fact set. For a commit this reconstruction includes the public key and both
signature components. Consequently the compact pile header and the canonical
`SimpleArchive` form identify the same semantic record without trusting a
separately stored key.

Pile replay keeps the records in intrinsic-ID order. Re-inserting an identical
record is an idempotent success; a different record reconstructing to the same
ID is reported as a collision. Concatenating piles therefore gives set-union
semantics for collection records: append order and duplicate copies do not
change the discovered collection calculus. This order-independent behavior is
specific to collection records and does not turn the last-writer-wins pin log
into a set.

## Pin Records (branch head / tombstone)

```text
            ┌────16 byte───┐┌────16 byte───┐┌────────────32 byte───────────┐┌───192 byte───┐
          ┌ ┌──────────────┐┌──────────────┐┌──────────────────────────────┐┌──────────────┐
 head     │ │ branch marker││   branch id  ││             hash             ││  reserved 0s │
 (256 B)  └ └──────────────┘└──────────────┘└──────────────────────────────┘└──────────────┘

            ┌────16 byte───┐┌────16 byte───┐┌──────────────224 byte────────────────────────┐
          ┌ ┌──────────────┐┌──────────────┐┌──────────────────────────────────────────────┐
 tombstone│ │ tomb marker  ││   branch id  ││                 reserved 0s                  │
 (256 B)  └ └──────────────┘└──────────────┘└──────────────────────────────────────────────┘
```

Pin-head records map a pin (branch) identifier to the hash of a blob; a
tombstone retracts the mapping. Appends are intentionally lightweight: the
pile does not check whether the referenced blob exists locally, allowing
deployments that store heads on disk while serving blob contents from a remote
store.

## Local Cell Records

```text
            ┌────16 byte───┐┌────16 byte───┐┌────────────32 byte───────────┐┌───192 byte───┐
          ┌ ┌──────────────┐┌──────────────┐┌──────────────────────────────┐┌──────────────┐
 value    │ │ cell marker  ││   cell id    ││      SimpleArchive handle    ││  reserved 0s │
 (256 B)  └ └──────────────┘└──────────────┘└──────────────────────────────┘└──────────────┘

            ┌────16 byte───┐┌────16 byte───┐┌──────────────224 byte────────────────────────┐
          ┌ ┌──────────────┐┌──────────────┐┌──────────────────────────────────────────────┐
 clear    │ │ clear marker ││   cell id    ││                 reserved 0s                  │
 (256 B)  └ └──────────────┘└──────────────┘└──────────────────────────────────────────────┘
```

Local cells are named last-writer-wins operational values. Their V3 markers are
`24264FA9EE46A1ACC0E024AE69774B09` (replace) and
`4FE372AE868D22A44DED7A60D579B651` (clear), minted with `trible genid` on
2026-08-10. A clear is material even in a pile that has not observed an older
value, so concatenating that pile after an older one still suppresses the old
cell.

Cells are deliberately not branches: they have no compare-and-swap guard,
history, enumeration API, collection authority, or gossip surface. They are
also not wants. A current cell value is instead a recursive **local operational
retention root**, allowing queryable policy stored in ordinary
`SimpleArchive` blobs to survive collection without asserting that it belongs
to any published collection.

## Want Records

```text
            ┌────16 byte───┐┌────────────32 byte───────────┐┌────────────208 byte──────────┐
          ┌ ┌──────────────┐┌──────────────────────────────┐┌──────────────────────────────┐
 want     │ │assert marker ││         blob handle          ││          reserved 0s         │
 (256 B)  └ └──────────────┘└──────────────────────────────┘└──────────────────────────────┘
```

A want assertion (and its retraction counterpart, using the same layout with a
different marker) is keyed by **blob handle** — per-blob and anonymous, with no
pin ID. Assertions and retractions resolve last-writer-wins per handle. The
resulting [`WantStore`](https://docs.rs/triblespace-core/latest/triblespace_core/repo/trait.WantStore.html)
state is independent from mutable branches and local policy cells: a pile may use wants for
fetch-on-demand and bounded cache retention without using branches at all.
Because wants are durable records, reopening a pile reconstructs the current
wanted set. The implementation keeps the original weak-pin/weak-unpin marker
IDs solely so existing piles continue to decode byte-for-byte.

## Legacy V1 records

Piles written before V3 contain 64-byte-aligned records: a 64-byte blob header
(marker, timestamp, length, hash) followed by a payload padded to a 64-byte
boundary, and 64-byte branch / tombstone records. The reader recognises the V1
markers and reads these records byte-identical; they are never rewritten. V1
had no want records.

## Recovery

`refresh` scans an existing file to ensure every header uses a known marker
and that the whole record fits. It does not verify any hashes. If a truncated
or unknown block is found the function reports the number of bytes that were
valid so far using `ReadError::CorruptPile` — and leaves the file untouched.

If the file shrinks between scans into data that has already been applied, the
process aborts immediately. Previously returned `Bytes` handles would dangle
and continuing could cause undefined behavior, so truncation into validated
data is treated as unrecoverable.

`refresh` holds a shared file lock while scanning. This prevents a concurrent
`amputate` call from truncating the file out from under the reader.

The `amputate` helper is the explicit, destructive repair path: it re-runs the same
validation under an exclusive lock and truncates the file to the valid length
if corruption is encountered, discarding incomplete data left by an
interrupted write. Run it deliberately (e.g. via `trible pile amputate <path>`)
— never as a routine part of opening — and only once you know the "corruption"
isn't just an older binary meeting newer record kinds. Hash verification
happens lazily only when individual blobs are loaded so that opening a large
pile remains fast.

For more details on interacting with a pile see the [`Pile` struct
documentation](https://docs.rs/triblespace/latest/triblespace/repo/pile/struct.Pile.html).

[1]: https://db.cs.cmu.edu/mmap-cidr2022/ "The Case Against Memory-Mapped I/O"
