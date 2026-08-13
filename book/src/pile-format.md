# Pile Format

The on-disk pile keeps blobs, native collection records, pins, wants, and
collection-publication grants in one append-only file. The write-ahead log *is*
the database: all indices are
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

## Record model: generic envelope and uniform 256-byte framing

Every record the pile writes today begins with the same fixed **256-byte
envelope header**, followed (for blobs) by the payload, padded so the whole
record is a **256-byte multiple**:

| Offset | Width | Field |
|---:|---:|---|
| `0..16` | 16 | Generic envelope marker `E5A95E5D8A0BBA8782E46B9C9E73B313` |
| `16..32` | 16 | Semantic record-kind ID |
| `32..36` | 4 | Total record span in 256-byte blocks, unsigned little-endian |
| `36..256` | 220 | Kind-specific body and zeroed reserved bytes |

The envelope marker was minted with `trible genid` on 2026-08-11. Record kinds
reuse the existing current V3 blob/branch/want markers and V4 collection
markers; no semantic IDs were reminted. A collection descriptor itself remains
an ordinary blob, not a fourth collection-record kind. Want records likewise
retain their historical weak-pin/weak-unpin kind IDs; those are physical format
names, not the public storage model.

The span includes the header. Zero is invalid; decoders perform checked
`span * 256` arithmetic and require that the complete record fit in the
observed pile prefix. A header-only record therefore has span 1. A blob has
span `1 + ceil(payload_length / 256)`, and the decoder requires the generic
span and blob-specific byte length to agree exactly. A `u32` block count keeps
the prefix compact while permitting a single record of almost 1&nbsp;TiB.

The common framing is load-bearing:

- **Position independence.** Blob data starts at the constant
  `record_start + 256`; there is no offset-derived padding. A record means
  the same thing at any offset, so records survive relocation and
  `cat a.pile >> b.pile` is a valid merge of two piles.
- **Alignment for free.** Because every newly written record is a 256-byte
  multiple, a pile composed entirely of current 256-byte-framed records stays
  aligned under the atomic lock-free append. Every blob payload in such a pile
  lands on a 256-byte boundary, satisfying GPU storage-buffer binding
  requirements (CUDA / Metal `min_storage_buffer_offset_alignment`) for
  zero-copy aliasing.
- **Cache-friendly headers.** Each header begins on a cache-line boundary and
  admits safe typed views with the `zerocopy` crate.

Reserved kind-body bytes are zeroed and are **not** part of the content hash;
per-record metadata belongs in tribles, not in the header, so identical bytes
never fork into distinct blobs.

Unknown kinds inside this envelope decode as opaque records. Normal pile replay
semantically skips them and continues with subsequent known records;
`PileRecords` still exposes their exact offset, length, kind, and raw bytes.
This is a forgetful projection: any future kind introduced under this envelope
must remain independent of the meaning of known records. In particular, it may
not change the validity or effect of a known record, constrain an old writer's
otherwise-valid append, or make an existing record depend on a companion
record of the new kind. Such an extension—or any other extension whose absence
cannot conservatively mean “no effect”—requires a new generic envelope marker
instead.

Concatenation is associative ordered composition, not universally commutative:
branches and wants are right-biased last-writer-wins logs. Opaque filtering is
sound because it leaves the relative order of every known record unchanged;
only collection records additionally collapse to order-independent set union.

The reader still accepts original **V1** records (64-byte-aligned blob, branch,
and tombstone layouts), unenveloped **V3** records, and unenveloped **V4**
collection records byte-for-byte. An unknown *unenveloped* marker still reports
`ReadError::UnsupportedRecord { offset, marker }`, because its boundary is
unknowable. Older binaries predating the generic envelope reject its marker;
upgrade them rather than trying to repair the pile.

The envelope deliberately has no checksum or complemented length. It detects
torn/truncated appends through its bounds and kind-specific checks, and it
solves version-skew framing; it is not intended to diagnose arbitrary header
bit rot. For example, a corrupted kind can look like an opaque kind, while a
corrupted but still in-bounds span can cover later bytes. Blob payload integrity
remains protected by its content hash.

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
   in memory. It **fails loud** on a corrupt or torn record
   (`ReadError::CorruptPile { valid_length }`). It skips bounded unknown
   envelope kinds as opaque records and distinguishes an unknown legacy marker
   as `ReadError::UnsupportedRecord { offset, marker }`. It never mutates the
   file. Callers rarely need to invoke it directly:
   `reader`, `records`, `pins`, `head`, and `update` call `refresh` internally
   before they inspect or apply records, so external writers are visible
   without a standalone scan.
3. **Amputate only when asked to.** `amputate` is the explicit, opt-in repair
   path: it re-runs validation under an exclusive lock and truncates the file
   back to the last valid record, discarding a torn record left by a
   crash. It crosses complete opaque envelopes and may truncate a torn opaque
   tail at its known start. It refuses `UnsupportedRecord` without modifying
   the file because an unknown unenveloped record's boundary is unknowable. It
   is deliberately **not** part of the normal open sequence. The
   `trible pile amputate <path>` command wraps it for operators.
4. **Append new records.** `put` (through the `BlobStorePut` trait),
   `CollectionStore::insert`, and pin update helpers extend the file. Each
   append immediately feeds the bytes back through the record scanner so
   in-memory indices stay synchronised without waiting for a manual `refresh`.
   Blob records use a single `write_vectored` call; fixed-width collection and
   pin records use one append of their 256-byte envelope header.
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

Every newly written record begins with the generic marker, kind ID, and span
described above. The sections below illustrate each kind-specific body.

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
    // or torn record and never mutates the file. Unknown envelope kinds are
    // skipped as opaque; unknown legacy markers remain unsupported.
    match pile.refresh() {
        Ok(()) => {}
        Err(err @ ReadError::UnsupportedRecord { .. }) => return Err(err.into()),
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
parser against the layouts documented in this chapter. An unknown envelope kind
is yielded as `PileRecordContent::Opaque` with its declared boundary; callers
can preserve its exact bytes through the iterator's raw file view. An unknown
unenveloped marker is reported as `UnsupportedRecord`, while a malformed or
truncated record is reported as `CorruptPile`.

For an operator-facing view of one exact boundary, use
`trible pile diagnose record-at <pile> <offset>`. The command is read-only: it
walks the same canonical decoder from byte zero, rejects offsets that land
inside a record, and prints the physical marker, classification, known span,
next offset, and the fields the current reader can safely decode. In
particular, an unsupported unenveloped marker asks for a newer reader and never
suggests amputation; only a malformed or torn known record presents the
explicit destructive-repair command.

Semantic Pile and Yard reads may continue across opaque records, but destructive
retention is different: `Pile::rewrite_retained_into`, Yard collection,
compaction, and reclaim refuse before mutation when any opaque record is
present. An older reader cannot know whether the unknown kind owns a known
blob, so silently omitting it—or collecting its dependencies—would be unsafe.

## Blob Records

| Offset | Width | Field |
|---:|---:|---|
| `0..16` | 16 | Generic envelope marker |
| `16..32` | 16 | Blob kind `9C33EEB525065A62EAEC4BE43DCC355A` |
| `32..36` | 4 | Total 256-byte-block span, little-endian |
| `36..44` | 8 | Timestamp in Unix milliseconds, little-endian |
| `44..52` | 8 | Exact unpadded payload byte length, little-endian |
| `52..84` | 32 | BLAKE3 payload hash |
| `84..256` | 172 | Reserved zeros |
| `256..` | variable | Payload and post-padding to the declared span |

Each blob record carries:

- **Record kind** – identifies blob semantics inside the generic envelope.
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

`CollectionStore` is a grow-only set of typed collection-calculus records:
signed `COMMIT` assertions and unsigned `MERGE` and `DERIVE` equations. The
pile stores these three kinds directly as fixed one-block enveloped records.
Their pile record-kind markers retain the V4 values. They are
**not blob records**, have no following payload, and carry no insertion
timestamp. They are also distinct from mutable branch pins and wants:
collection records have no head, tombstone, or
last-writer-wins update. Their logical key is a content-derived 16-byte record
ID; a collection record is not a trible entity.

The collection itself is identified by a canonical `SimpleArchive` descriptor
containing `(scope, representation, recipe)`. Its 32-byte blob handle is the
sole `CollectionId`. Records carry this handle directly; there is no definition
record or registry. Consequently a transferred claim names the exact descriptor
bytes needed to interpret it, using the ordinary blob store.

The magic markers below identify the compact pile representation. They are
storage-envelope markers, distinct both from the stable semantic kind IDs used
in record-ID/signature domains and from the one-byte versioned tags used by
generic dense record stores. There is no equivalent `SimpleArchive` form for
these algebra records.

| Kind | V4 kind ID | Kind-specific byte layout after the common prefix |
|---|---|---|
| Commit | `CBF2CF97D52A3486E16C12D70D397C66` | `36..68` descriptor handle, `68..100` data digest, `100..132` metadata handle, `132..164` Ed25519 public key, `164..196` signature R, `196..228` signature S, `228..256` reserved zeros |
| Merge | `9F5D028D4C423620D6957A5F726FA727` | `36..68` descriptor handle, `68..100` lower input digest, `100..132` higher input digest, `132..164` result digest, `164..256` reserved zeros |
| Derive | `ECFB2EE90ED8042244F7BAC704454BB9` | `36..68` source descriptor handle, `68..100` target descriptor handle, `100..132` input digest, `132..164` output digest, `164..256` reserved zeros |

Every reserved byte must be zero; a nonzero reserved byte makes replay fail as
corrupt rather than silently assigning meaning to a format extension. Merge
inputs are stored in lexicographic digest order (`low <= high`), so swapping
the two operands cannot create a second representation of the same
commutative equation.

The record ID is deliberately absent from these headers. On replay, the
decoder reconstructs the record's exact dense typed payload: 192 bytes for a
commit and 128 bytes for a merge or derive. It hashes a domain separator,
record-ID version, stable semantic kind ID, and every payload byte with BLAKE3,
then uses the digest's final 16 bytes as the record ID. For a commit the payload
includes the public key and both signature components. Consequently the pile
header and the typed dense codec identify the same semantic record without
trusting a separately stored key.

Pile replay keeps the records in record-ID order. Re-inserting an identical
record is an idempotent success; a different record reconstructing to the same
ID is reported as a collision. Concatenating piles therefore gives set-union
semantics for collection records: append order and duplicate copies do not
change the discovered collection calculus. This order-independent behavior is
specific to collection records and grow-only publication grants; it does not
turn the last-writer-wins pin log into a set.

## Collection Publication Grants

`CollectionGossipStore` is separate from `CollectionStore`. A grant is an
author-signed, irrevocable protocol permission to redistribute that author's
strictly verified and locally admitted `COMMIT`s in one exact `CollectionId`.
It is not a fourth collection-calculus record, does not retain any blob, and
does not require the descriptor or commits to be present when inserted. Thus a
grant arriving before or after its data has the same meaning under pile
concatenation.

The kind ID `9BB5B1F4D6FD8FB850B494C2CF51B5CA` was minted with `trible genid` on
2026-08-12. Its one-block envelope body is:

| Offset | Field |
|---:|---|
| `36..68` | collection descriptor handle |
| `68..100` | author Ed25519 public key |
| `100..132` | signature R component |
| `132..164` | signature S component |
| `164..256` | reserved zeros |

The signed transcript is domain-separated and binds the kind, version, author,
and descriptor handle. Physical storage preserves even structurally decoded
invalid witnesses so imported hostile evidence cannot corrupt a whole pile;
consumers grant publication permission only after strict signature
verification. Logical authorization is existential over `(author,
CollectionId)` while storage retains byte-distinct witnesses in deterministic
order. There is deliberately no tombstone or `ungossip`: once another holder
has observed permission, local mutation cannot take it back. Runtime policy may
still choose not to operate a relay.

### Legacy unenveloped V4 collection records

Before the generic envelope, the same three V4 kind IDs occupied bytes
`0..16`, followed immediately by the semantic fields. Those exact 256-byte
records remain readable and reconstruct the same current collection records.
They are never rewritten in place; newly inserted records use the envelope.

| Kind | Legacy unenveloped byte layout |
|---|---|
| Commit | `0..16` kind, `16..48` descriptor, `48..80` data, `80..112` metadata, `112..144` public key, `144..176` signature R, `176..208` signature S, `208..256` zeros |
| Merge | `0..16` kind, `16..48` descriptor, `48..80` low, `80..112` high, `112..144` result, `144..256` zeros |
| Derive | `0..16` kind, `16..48` source descriptor, `48..80` target descriptor, `80..112` input, `112..144` output, `144..256` zeros |

### Legacy V3 collection records

V3 encoded a collection by a separate definition record with a 16-byte
intrinsic entity ID. Its V1 commit signature transcript and equations therefore
do not identify the current descriptor-handle semantics. The reader recognizes
all four old markers so it can validate record boundaries and preserve their
bytes during conservative rewrites, but treats them as inert physical evidence:
they never enter `CollectionStore`, assert membership, or retain blobs.

| Legacy kind | V3 magic marker | Exact byte layout |
|---|---|---|
| Definition | `3BE108504E4F5242FB24AA72D6D94CE1` | `0..16` marker, `16..32` scope ID, `32..48` representation ID, `48..64` recipe ID, `64..256` reserved zeros |
| Commit | `BB758AA6F79FBFC4D1958592A8956777` | `0..16` marker, `16..32` definition ID, `32..64` data digest, `64..96` metadata handle, `96..128` Ed25519 public key, `128..160` signature R, `160..192` signature S, `192..256` reserved zeros |
| Merge | `CC0108AC1DF4F335AFA856A529C42BE9` | `0..16` marker, `16..32` definition ID, `32..64` lower input digest, `64..96` higher input digest, `96..128` result digest, `128..256` reserved zeros |
| Derive | `07ECF056F6F015D94389FFF21F851480` | `0..16` marker, `16..32` source definition ID, `32..48` target definition ID, `48..80` input digest, `80..112` output digest, `112..256` reserved zeros |

## Pin Records (branch head / tombstone)

| Kind | Kind ID | Kind-specific body after the common prefix |
|---|---|---|
| Head | `AC363D04AFE1AF17B39581B1E23021D7` | `36..52` branch ID, `52..84` hash, `84..256` reserved zeros |
| Tombstone | `D0CBA0C8EAAB4C0C73121C3205671E4F` | `36..52` branch ID, `52..256` reserved zeros |

Pin-head records map a pin (branch) identifier to the hash of a blob; a
tombstone retracts the mapping. Appends are intentionally lightweight: the
pile does not check whether the referenced blob exists locally, allowing
deployments that store heads on disk while serving blob contents from a remote
store.

## Retired Local Cell Records

| Kind | Kind ID | Kind-specific body after the common prefix |
|---|---|---|
| Replace | `24264FA9EE46A1ACC0E024AE69774B09` | `36..52` cell ID, `52..84` `SimpleArchive` handle, `84..256` reserved zeros |
| Clear | `4FE372AE868D22A44DED7A60D579B651` | `36..52` cell ID, `52..256` reserved zeros |

These markers belonged to an experimental named last-writer-wins value API.
That API and its writers were removed before release: a whole-value replacement
was not invariant under pile concatenation and made independently edited policy
silently order-dependent. The markers are retired permanently and must never be
assigned new meaning.

Current readers recognize both the enveloped form above and the fixed-width
unenveloped V3 form solely to preserve migration evidence. They expose either
form as `PileRecordContent::Opaque`, do not project a value into repository
state, and do not treat its referenced archive as a retention root. Raw tooling
through `PileRecords` can still copy or explicitly migrate the exact bytes.
Because their former ownership semantics are no longer interpreted,
`Pile::rewrite_retained_into` and Yard collection/reclaim refuse a destructive
rewrite while any such record remains.

## Want Records

| Kind | Kind ID | Kind-specific body after the common prefix |
|---|---|---|
| Assert | `8F3EEFEDECD491F63F6EAAA5FD6F3D5E` | `36..68` blob handle, `68..256` reserved zeros |
| Retract | `2D76662DFF0187EC36A8C90B12BB8B0D` | `36..68` blob handle, `68..256` reserved zeros |

A want assertion (and its retraction counterpart, using the same layout with a
different marker) is keyed by **blob handle** — per-blob and anonymous, with no
pin ID. Assertions and retractions resolve last-writer-wins per handle. The
resulting [`WantStore`](https://docs.rs/triblespace-core/latest/triblespace_core/repo/trait.WantStore.html)
state is independent from mutable branches and native collections: a pile may use wants for
fetch-on-demand and bounded cache retention without using branches at all.
Because wants are durable records, reopening a pile reconstructs the current
wanted set. The implementation keeps the original weak-pin/weak-unpin marker
IDs solely so existing piles continue to decode byte-for-byte.

## Legacy unenveloped records

Unenveloped V3 blob, branch, and want records place their kind ID directly in
`0..16`. Their semantic bodies begin at byte 16 rather than byte 36: a V3 blob
stores timestamp at `16..24`, byte length at `24..32`, and hash at `32..64`;
branch IDs occupy `16..32`; branch values occupy `32..64`; and want handles
occupy `16..48`. All have a 256-byte header and remain readable byte-for-byte.
The two retired local-cell markers are the one deliberate exception to the
unknown-unenveloped rule: their historical 256-byte boundary is known, so the
reader crosses them and exposes them as opaque migration evidence. Their former
cell ID occupied `16..32`, and a replacement's archive handle occupied
`32..64`. The legacy V3 and V4 collection layouts are listed above.

Piles written before V3 contain 64-byte-aligned V1 records: a 64-byte blob
header (marker, timestamp, length, hash) followed by a payload padded to a
64-byte boundary, and 64-byte branch / tombstone records. The reader recognises
the V1 markers and reads these records byte-identical; they are never rewritten.
V1 had no want records.

## Recovery

`refresh` scans an existing file to ensure every record fits. It does not verify
blob hashes. A malformed or truncated known or enveloped record reports the
number of bytes that were valid so far using `ReadError::CorruptPile`. A
complete unknown envelope kind is structurally accepted and semantically
skipped; an unknown unenveloped marker reports its bytes and offset using
`ReadError::UnsupportedRecord`, since the reader cannot infer that record's
length. The retired cell markers are recognized as fixed 256-byte opaque
records rather than guessed. Both errors leave the file untouched, and the
reader never guesses any other legacy record length.

If the file shrinks between scans into data that has already been applied, the
process aborts immediately. Previously returned `Bytes` handles would dangle
and continuing could cause undefined behavior, so truncation into validated
data is treated as unrecoverable.

`refresh` holds a shared file lock while scanning. This prevents a concurrent
`amputate` call from truncating the file out from under the reader.

The `amputate` helper is the explicit, destructive repair path: it re-runs the
same validation under an exclusive lock and truncates the file to the valid
length if corruption is encountered, discarding incomplete data left by an
interrupted write. It crosses complete opaque envelopes, truncates a torn one
at its start, and propagates `UnsupportedRecord` for unknown unenveloped
markers without truncating. Run it deliberately (e.g. via
`trible pile amputate <path>`)—never as a routine part of opening. Hash
verification happens lazily only when individual blobs are loaded so that
opening a large pile remains fast.

For more details on interacting with a pile see the [`Pile` struct
documentation](https://docs.rs/triblespace/latest/triblespace/repo/pile/struct.Pile.html).

[1]: https://db.cs.cmu.edu/mmap-cidr2022/ "The Case Against Memory-Mapped I/O"
