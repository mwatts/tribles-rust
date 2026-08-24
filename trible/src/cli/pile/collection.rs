//! `trible pile collection …` — a collection-aware view of a pile.
//!
//! A collection is identified by the blake3 handle of its *descriptor blob*,
//! not by the entity id inside that blob. The descriptor is an ordinary
//! canonical `SimpleArchive` of one intrinsic entity: the
//! `KIND_COLLECTION_DESCRIPTOR` tag, an anchor — `name` + `namespace` for a
//! root, `source` for a derivation — plus an optional local capability
//! authority, the blob `representation`, the join `recipe`, and whatever
//! arguments its recipe carries.
//!
//! Without this module the only way to look at one was
//! `pile blob inspect <PILE> blake3:<HEX>`, which reports "256 bytes, Binary"
//! and nothing else. Here the facts are read with the same
//! [`descriptor`](triblespace_core::collection::descriptor) queries resolution
//! and retention use, so the CLI view can never drift from the semantics.
//!
//! Names are what the listing leads with. A root carries the name it is known
//! by inside its namespace, and that name — not the 64 hex characters of its
//! descriptor handle — is what an operator came to read. Every subcommand that
//! takes a collection therefore accepts either spelling; the two can never be
//! confused, because a name is `[a-z0-9-]{1,32}` starting with a letter and a
//! handle is 64 hex characters.

use anyhow::{anyhow, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::SuccinctArchiveBlob;
use triblespace_core::blob::Blob;
use triblespace_core::blob::TryFromBlob;
use triblespace_core::collection::descriptor;
use triblespace_core::collection::records::{CollectionHandle, CollectionName, CollectionRecord};
use triblespace_core::collection::store::CollectionStore;
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::hash::{Blake3, Hash};
use triblespace_core::inline::Inline;
use triblespace_core::metadata::MetaDescribe;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use triblespace_core::trible::TribleSet;

use super::open_refreshed;

/// Hex characters shown for a handle or key when the full value is not asked
/// for. Sixteen is far past the point where two collections in one pile
/// collide, and short enough that a row stays one terminal line.
const ABBREV: usize = 16;

#[derive(Parser)]
pub enum Command {
    /// List every collection the pile references, named ones first.
    ///
    /// A collection is "referenced" when some commit, merge, or derive record
    /// in the pile names it. Roots are listed by the name they carry, then
    /// derivations by their source, then any collection whose descriptor
    /// claims no anchor or is not in the pile at all — those are never
    /// silently dropped, because a pile that has forgotten what a collection
    /// is called still has its records.
    List {
        /// Path to the pile file to inspect.
        path: PathBuf,
        /// Only list named roots, hiding derivations and anchorless
        /// collections.
        #[arg(long)]
        named: bool,
        /// Also show the descriptor blob's size and storage timestamp.
        #[arg(long)]
        metadata: bool,
        /// Print handles, namespace keys, and authority keys in full.
        #[arg(long)]
        long: bool,
    },
    /// Fully decode one collection descriptor.
    ///
    /// Prints the descriptor handle (the collection identity), the intrinsic
    /// entity id inside the archive, the decoded anchor / representation /
    /// recipe, every trible in the archive, and how many records in this pile
    /// reference the collection.
    Show {
        /// Path to the pile file to read.
        pile: PathBuf,
        /// Collection name, or descriptor handle with or without the
        /// `blake3:` prefix.
        collection: String,
    },
    /// List the commit, merge, and derive records that name one collection.
    ///
    /// This is the record stream itself, in the store's deterministic
    /// intrinsic-id order — not a commit chain, because a collection has no
    /// head to walk back from. Signatures are verified as they are printed.
    Log {
        /// Path to the pile file to read.
        pile: PathBuf,
        /// Collection name, or descriptor handle with or without the
        /// `blake3:` prefix.
        collection: String,
        /// Maximum records to print; `0` prints all of them.
        #[arg(long, default_value_t = 25)]
        limit: usize,
        /// Print handles and keys in full instead of abbreviated.
        #[arg(long)]
        long: bool,
    },
}

pub fn run(cmd: Command) -> Result<()> {
    match cmd {
        Command::List {
            path,
            named,
            metadata,
            long,
        } => run_list(path, named, metadata, long),
        Command::Show { pile, collection } => run_show(pile, collection),
        Command::Log {
            pile,
            collection,
            limit,
            long,
        } => run_log(pile, collection, limit, long),
    }
}

/// Parse a collection handle, accepting both `blake3:HEX` and a bare `HEX`.
///
/// `pile blob inspect` rejects the bare form with `BadProtocol`. Collection
/// handles get copied out of record dumps and log lines in both shapes, so
/// this entry point normalizes rather than nitpicks.
fn parse_collection_handle(handle: &str) -> Result<CollectionHandle> {
    use triblespace::prelude::TryToInline;

    let trimmed = handle.trim();
    let owned;
    let normalized = if trimmed.contains(':') {
        trimmed
    } else {
        owned = format!("blake3:{trimmed}");
        owned.as_str()
    };
    let hash: Inline<Hash<Blake3>> = normalized.try_to_inline().map_err(|e| {
        anyhow!(
            "parse collection handle {handle:?}: {e:?} (expected `blake3:<64 hex>` or bare hex)"
        )
    })?;
    Ok(hash.into())
}

fn handle_hex(handle: CollectionHandle) -> String {
    hex::encode(handle.raw)
}

fn abbrev(text: &str, long: bool) -> String {
    if long || text.len() <= ABBREV {
        text.to_owned()
    } else {
        format!("{}…", &text[..ABBREV])
    }
}

/// Resolve the blob-representation id against the schemas that actually
/// implement a collection kind. There is no id-keyed registry to consult, so
/// this asks each known `MetaDescribe` schema for its own id rather than
/// inventing a table of literals.
fn representation_name(id: Id) -> Option<&'static str> {
    if id == <SimpleArchive as MetaDescribe>::id() {
        Some("SimpleArchive")
    } else if id == <SuccinctArchiveBlob as MetaDescribe>::id() {
        Some("SuccinctArchiveBlob")
    } else {
        None
    }
}

/// Resolve a recipe id against the union recipes declared in core.
fn recipe_name(id: Id) -> Option<&'static str> {
    use triblespace_core::collection::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1;
    use triblespace_core::collection::succinctarchive_union::{
        RANK9_LIFTED_UNION_RECIPE_V1_32_BE, RANK9_LIFTED_UNION_RECIPE_V1_32_LE,
        RANK9_LIFTED_UNION_RECIPE_V1_64_BE, RANK9_LIFTED_UNION_RECIPE_V1_64_LE,
    };

    if id == TRIBLE_SET_UNION_RECIPE_V1 {
        Some("TRIBLE_SET_UNION_RECIPE_V1")
    } else if id == RANK9_LIFTED_UNION_RECIPE_V1_32_LE {
        Some("RANK9_LIFTED_UNION_RECIPE_V1_32_LE")
    } else if id == RANK9_LIFTED_UNION_RECIPE_V1_32_BE {
        Some("RANK9_LIFTED_UNION_RECIPE_V1_32_BE")
    } else if id == RANK9_LIFTED_UNION_RECIPE_V1_64_LE {
        Some("RANK9_LIFTED_UNION_RECIPE_V1_64_LE")
    } else if id == RANK9_LIFTED_UNION_RECIPE_V1_64_BE {
        Some("RANK9_LIFTED_UNION_RECIPE_V1_64_BE")
    } else {
        None
    }
}

fn named_id(id: Id, name: Option<&'static str>) -> String {
    match name {
        Some(name) => format!("{id:X} ({name})"),
        None => format!("{id:X}"),
    }
}

/// The short column form: the schema's name if this binary implements it, and
/// the bare id if it does not. An unknown id is not an error — a descriptor
/// may name a recipe some other reader owns.
fn short_named_id(
    id: Result<Id, impl std::fmt::Debug>,
    name: fn(Id) -> Option<&'static str>,
) -> String {
    match id {
        Ok(id) => name(id)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{id:X}")),
        Err(_) => "<unreadable>".to_owned(),
    }
}

/// How many records of each kind name one collection.
#[derive(Default, Clone, Copy)]
struct Refs {
    commits: usize,
    merges: usize,
    /// Derives naming this collection as their source.
    derives_from: usize,
    /// Derives naming this collection as their target.
    derives_into: usize,
}

impl Refs {
    fn total(&self) -> usize {
        self.commits + self.merges + self.derives_from + self.derives_into
    }
}

/// Walk every collection record in the pile and tally which collections they
/// name. Merges and derives are included: a collection that only ever appears
/// as a derive target is still a collection this pile references.
fn referenced_collections(pile: &mut Pile) -> Result<BTreeMap<CollectionHandle, Refs>> {
    let mut refs: BTreeMap<CollectionHandle, Refs> = BTreeMap::new();
    let records = pile
        .records()
        .map_err(|e| anyhow!("enumerate collection records: {e:?}"))?;
    for record in records {
        let record = record.map_err(|e| anyhow!("decode collection record: {e:?}"))?;
        match record {
            CollectionRecord::Commit(commit) => {
                refs.entry(commit.collection()).or_default().commits += 1;
            }
            CollectionRecord::Merge(merge) => {
                refs.entry(merge.collection()).or_default().merges += 1;
            }
            CollectionRecord::Derive(derive) => {
                refs.entry(derive.target()).or_default().derives_into += 1;
            }
        }
    }
    Ok(refs)
}

/// The descriptor facts, or why they could not be read.
enum Fields {
    Decoded(TribleSet),
    Missing,
    Undecodable(String),
}

impl Fields {
    fn load<R: BlobStoreGet>(reader: &R, handle: CollectionHandle) -> Self {
        let blob: Blob<SimpleArchive> = match reader.get(handle) {
            Ok(blob) => blob,
            Err(_) => return Fields::Missing,
        };
        match <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob) {
            Ok(facts) => Fields::Decoded(facts),
            Err(e) => Fields::Undecodable(format!("{e:?}")),
        }
    }

    fn facts(&self) -> Option<&TribleSet> {
        match self {
            Fields::Decoded(facts) => Some(facts),
            _ => None,
        }
    }
}

/// One collection the pile references: its identity, what its descriptor says,
/// and how many records name it.
///
/// The descriptor stays a bare `TribleSet` and is read with the same
/// `descriptor::*` queries the library uses. Nothing here caches a decoded
/// field beside it.
struct Enumerated {
    handle: CollectionHandle,
    refs: Refs,
    fields: Fields,
}

/// What a descriptor is anchored to. A root is named within a namespace, a
/// derivation is anchored by the collection it derives from, and a descriptor
/// that claims neither is neither — the listing says so rather than demanding
/// one.
enum Anchor {
    Root {
        /// `Err` when the stored name is not a legal collection name.
        name: Result<CollectionName, String>,
        namespace: Option<Result<String, String>>,
    },
    Derived(CollectionHandle),
    /// Anchored by the opaque scope id that naming replaced.
    RetiredScope,
    Bare,
    Unreadable(String),
}

/// The anchor a root carried before roots were named.
///
/// `collection_scope` was a minted opaque id. It discriminated roots
/// correctly and told a reader nothing: every faculty kept its scope as a hex
/// constant in its own source, so "which collection is this?" was answerable
/// only by someone holding the code. That is the complaint naming exists to
/// answer, and the library deleted the attribute along with the semantics
/// when `collection_name` + `collection_team` first replaced it. Current
/// descriptors use the distinct `collection_namespace` anchor instead.
///
/// A pile keeps what it was written with, so a reader still meets it — 46 of
/// the 69 collections in a live pile are anchored this way. Recognizing the
/// id here is the display layer naming a constant it does not implement,
/// exactly as `recipe_name` names a recipe it does not run: it lets the
/// listing say *why* a row has no name instead of reporting the absence as if
/// the descriptor were empty. Nothing here gives the attribute meaning again.
///
/// The value is not invented: it is the literal from the declaration deleted
/// in commit 0ea2a50b, itself minted with `trible genid` on 2026-08-07.
const RETIRED_COLLECTION_SCOPE: [u8; 16] = [
    0xD3, 0x41, 0x88, 0x73, 0xC7, 0x03, 0x92, 0xE3, 0xAD, 0xAA, 0x05, 0xC0, 0x0E, 0x11, 0xA5, 0x83,
];

fn carries_retired_scope(facts: &TribleSet) -> bool {
    facts
        .iter()
        .any(|trible| <[u8; 16]>::from(*trible.a()) == RETIRED_COLLECTION_SCOPE)
}

fn anchor(fields: &Fields) -> Anchor {
    let facts = match fields {
        Fields::Decoded(facts) => facts,
        Fields::Missing => return Anchor::Unreadable("descriptor blob not in pile".to_owned()),
        Fields::Undecodable(e) => {
            return Anchor::Unreadable(format!("descriptor undecodable: {e}"))
        }
    };
    if let Some(source) = descriptor::source(facts) {
        return Anchor::Derived(source);
    }
    match descriptor::name(facts) {
        Some(name) => Anchor::Root {
            name: name.map_err(|e| e.to_string()),
            namespace: descriptor::namespace(facts).map(|namespace| {
                namespace
                    .map(|namespace| hex::encode_upper(namespace.to_bytes()))
                    .map_err(|e| e.to_string())
            }),
        },
        None if carries_retired_scope(facts) => Anchor::RetiredScope,
        None => Anchor::Bare,
    }
}

/// Sort order for the listing: named roots by name, then derivations by
/// source, then everything with nothing to be called. Ties break on the
/// handle so a listing is stable across runs.
fn sort_key(row: &Enumerated) -> (u8, String, [u8; 32]) {
    match anchor(&row.fields) {
        Anchor::Root { name: Ok(name), .. } => (0, name.as_str().to_owned(), row.handle.raw),
        Anchor::Root { name: Err(_), .. } => (1, String::new(), row.handle.raw),
        Anchor::Derived(source) => (2, handle_hex(source), row.handle.raw),
        Anchor::RetiredScope => (3, String::new(), row.handle.raw),
        Anchor::Bare => (4, String::new(), row.handle.raw),
        Anchor::Unreadable(_) => (5, String::new(), row.handle.raw),
    }
}

/// Every collection the pile references, descriptors decoded, in listing
/// order.
fn enumerate(pile: &mut Pile) -> Result<Vec<Enumerated>> {
    let refs = referenced_collections(pile)?;
    let reader = pile.reader().map_err(|e| anyhow!("pile reader: {e:?}"))?;
    let mut rows: Vec<Enumerated> = refs
        .into_iter()
        .map(|(handle, refs)| Enumerated {
            handle,
            refs,
            fields: Fields::load(&reader, handle),
        })
        .collect();
    rows.sort_by_key(sort_key);
    Ok(rows)
}

/// Resolve what an operator typed to a collection in this pile.
///
/// A name and a handle can never be confused: a name is at most 32 bytes of
/// `[a-z0-9-]` starting with a letter, and a handle is 64 hex characters. So
/// the dispatch needs no flag — whichever spelling parses is the one meant.
fn resolve(rows: &[Enumerated], reference: &str) -> Result<CollectionHandle> {
    let reference = reference.trim();
    let Ok(name) = CollectionName::new(reference) else {
        return parse_collection_handle(reference);
    };

    let matches: Vec<CollectionHandle> = rows
        .iter()
        .filter(|row| {
            matches!(anchor(&row.fields), Anchor::Root { name: Ok(found), .. } if found == name)
        })
        .map(|row| row.handle)
        .collect();

    match matches.as_slice() {
        [only] => Ok(*only),
        [] => {
            let known: Vec<String> = rows
                .iter()
                .filter_map(|row| match anchor(&row.fields) {
                    Anchor::Root { name: Ok(name), .. } => Some(name.as_str().to_owned()),
                    _ => None,
                })
                .collect();
            if known.is_empty() {
                Err(anyhow!(
                    "no collection named {name} in this pile; it references no named collections \
                     at all. `pile collection list` shows what it does reference"
                ))
            } else {
                Err(anyhow!(
                    "no collection named {name} in this pile; it has: {}",
                    known.join(", ")
                ))
            }
        }
        many => Err(anyhow!(
            "{} collections in this pile are named {name} — a name identifies a collection only \
             within one namespace, and these disagree. Pass one of the handles instead:\n{}",
            many.len(),
            many.iter()
                .map(|handle| format!("  {}", handle_hex(*handle)))
                .collect::<Vec<_>>()
                .join("\n")
        )),
    }
}

enum Align {
    Left,
    Right,
}

/// Print a table whose columns are exactly as wide as their widest cell.
fn print_table(headers: &[&str], aligns: &[Align], rows: &[Vec<String>], indent: &str) {
    if rows.is_empty() {
        return;
    }
    let mut widths: Vec<usize> = headers
        .iter()
        .map(|header| header.chars().count())
        .collect();
    for row in rows {
        for (column, cell) in row.iter().enumerate() {
            widths[column] = widths[column].max(cell.chars().count());
        }
    }
    let render = |cells: &[String]| {
        let mut line = String::from(indent);
        for (column, cell) in cells.iter().enumerate() {
            let last = column + 1 == cells.len();
            let pad = widths[column].saturating_sub(cell.chars().count());
            match aligns[column] {
                Align::Left => {
                    line.push_str(cell);
                    if !last {
                        line.push_str(&" ".repeat(pad));
                    }
                }
                Align::Right => {
                    line.push_str(&" ".repeat(pad));
                    line.push_str(cell);
                }
            }
            if !last {
                line.push_str("  ");
            }
        }
        line
    };
    let header_cells: Vec<String> = headers.iter().map(|header| header.to_string()).collect();
    println!("{}", render(&header_cells));
    for row in rows {
        println!("{}", render(row));
    }
}

fn format_timestamp(millis: u64) -> String {
    use chrono::{DateTime, Utc};
    use std::time::{Duration, UNIX_EPOCH};

    let dt = UNIX_EPOCH + Duration::from_millis(millis);
    DateTime::<Utc>::from(dt).to_rfc3339()
}

fn run_list(path: PathBuf, named_only: bool, metadata: bool, long: bool) -> Result<()> {
    let mut pile = open_refreshed(&path)?;
    let res = (|| -> Result<()> {
        let rows = enumerate(&mut pile)?;
        if rows.is_empty() {
            println!("(no collections referenced by pile {})", path.display());
            return Ok(());
        }
        let reader = pile.reader().map_err(|e| anyhow!("pile reader: {e:?}"))?;

        // A row's trailing columns are the same question for every section, so
        // they are built once here.
        let tail = |row: &Enumerated| -> Vec<String> {
            let mut cells = vec![
                row.refs.total().to_string(),
                abbrev(&handle_hex(row.handle), long),
                match row.fields.facts().and_then(descriptor::authority) {
                    Some(Ok(authority)) => abbrev(&hex::encode_upper(authority.to_bytes()), long),
                    Some(Err(error)) => format!("<invalid: {error}>"),
                    None => "-".to_owned(),
                },
                match row.fields.facts() {
                    Some(facts) => {
                        short_named_id(descriptor::representation(facts), representation_name)
                    }
                    None => "-".to_owned(),
                },
                match row.fields.facts() {
                    Some(facts) => short_named_id(descriptor::recipe(facts), recipe_name),
                    None => "-".to_owned(),
                },
            ];
            if metadata {
                match reader.metadata(row.handle) {
                    Ok(Some(meta)) => {
                        cells.push(meta.length.to_string());
                        cells.push(format_timestamp(meta.timestamp));
                    }
                    Ok(None) => {
                        cells.push("-".to_owned());
                        cells.push("absent".to_owned());
                    }
                    Err(e) => {
                        cells.push("-".to_owned());
                        cells.push(format!("metadata error ({e:?})"));
                    }
                }
            }
            cells
        };

        let mut named: Vec<Vec<String>> = Vec::new();
        let mut derived: Vec<Vec<String>> = Vec::new();
        let mut anchorless: Vec<(String, Vec<String>)> = Vec::new();
        let mut scoped = 0usize;
        for row in &rows {
            match anchor(&row.fields) {
                Anchor::Root { name, namespace } => {
                    let mut cells = vec![
                        match &name {
                            Ok(name) => name.as_str().to_owned(),
                            Err(e) => format!("<invalid: {e}>"),
                        },
                        match &namespace {
                            Some(Ok(namespace)) => abbrev(namespace, long),
                            Some(Err(e)) => format!("<invalid: {e}>"),
                            None => "-".to_owned(),
                        },
                    ];
                    cells.extend(tail(row));
                    named.push(cells);
                }
                Anchor::Derived(source) => {
                    let mut cells = vec![abbrev(&handle_hex(source), long)];
                    cells.extend(tail(row));
                    derived.push(cells);
                }
                // A descriptor that decoded but claims no anchor needs no
                // explanation beyond the section heading. One that could not
                // be read at all does, and says so in its own column.
                Anchor::RetiredScope => {
                    scoped += 1;
                    anchorless.push(("pre-naming scope anchor".to_owned(), tail(row)));
                }
                Anchor::Bare => {
                    anchorless.push((String::new(), tail(row)));
                }
                Anchor::Unreadable(why) => {
                    anchorless.push((why, tail(row)));
                }
            }
        }

        // Name only the categories this pile actually has. "0 derived" is
        // noise in a summary whose job is to say what is here.
        let summary: Vec<String> = [
            (named.len(), "named"),
            (derived.len(), "derived"),
            (scoped, "pre-naming"),
            (anchorless.len() - scoped, "without a readable anchor"),
        ]
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .map(|(count, what)| format!("{count} {what}"))
        .collect();
        println!(
            "collections in {}: {} ({})",
            path.display(),
            rows.len(),
            summary.join(", ")
        );

        let mut tail_headers: Vec<&str> = vec![
            "RECORDS",
            "COLLECTION",
            "AUTHORITY",
            "REPRESENTATION",
            "RECIPE",
        ];
        let mut tail_aligns = vec![
            Align::Right,
            Align::Left,
            Align::Left,
            Align::Left,
            Align::Left,
        ];
        if metadata {
            tail_headers.extend(["BYTES", "STORED"]);
            tail_aligns.extend([Align::Right, Align::Left]);
        }

        if !named.is_empty() {
            println!();
            let mut headers = vec!["NAME", "NAMESPACE"];
            headers.extend(tail_headers.iter().copied());
            let mut aligns = vec![Align::Left, Align::Left];
            aligns.extend(tail_aligns.iter().map(|align| match align {
                Align::Left => Align::Left,
                Align::Right => Align::Right,
            }));
            print_table(&headers, &aligns, &named, "  ");
        }

        if named_only {
            let hidden: Vec<String> = [(derived.len(), "derived"), (anchorless.len(), "unnamed")]
                .into_iter()
                .filter(|(count, _)| *count > 0)
                .map(|(count, what)| format!("{count} {what}"))
                .collect();
            if !hidden.is_empty() {
                println!();
                println!("  ({} hidden by --named)", hidden.join(", "));
            }
            return Ok(());
        }

        if !derived.is_empty() {
            println!();
            println!("derived from another collection:");
            let mut headers = vec!["SOURCE"];
            headers.extend(tail_headers.iter().copied());
            let mut aligns = vec![Align::Left];
            aligns.extend(tail_aligns.iter().map(|align| match align {
                Align::Left => Align::Left,
                Align::Right => Align::Right,
            }));
            print_table(&headers, &aligns, &derived, "  ");
        }

        if !anchorless.is_empty() {
            println!();
            println!("without a name:");
            // The note column earns its width only if some row has a note;
            // otherwise every cell would repeat the heading.
            let noted = anchorless.iter().any(|(note, _)| !note.is_empty());
            let mut headers: Vec<&str> = Vec::new();
            let mut aligns: Vec<Align> = Vec::new();
            if noted {
                headers.push("NOTE");
                aligns.push(Align::Left);
            }
            headers.extend(tail_headers.iter().copied());
            aligns.extend(tail_aligns.iter().map(|align| match align {
                Align::Left => Align::Left,
                Align::Right => Align::Right,
            }));
            let rows: Vec<Vec<String>> = anchorless
                .into_iter()
                .map(|(note, cells)| {
                    if noted {
                        std::iter::once(note).chain(cells).collect()
                    } else {
                        cells
                    }
                })
                .collect();
            print_table(&headers, &aligns, &rows, "  ");
        }

        Ok(())
    })();
    let close_res = pile.close().map_err(|e| anyhow!("pile close: {e:?}"));
    res.and(close_res)
}

fn run_show(path: PathBuf, reference: String) -> Result<()> {
    let mut pile = open_refreshed(&path)?;
    let res = (|| -> Result<()> {
        let rows = enumerate(&mut pile)?;
        let handle = resolve(&rows, &reference)?;
        let reader = pile.reader().map_err(|e| anyhow!("pile reader: {e:?}"))?;
        let blob: Blob<SimpleArchive> = reader
            .get(handle)
            .map_err(|e| anyhow!("read descriptor blob {}: {e:?}", handle_hex(handle)))?;

        println!("collection: blake3:{}", handle_hex(handle));
        if let Ok(Some(meta)) = reader.metadata(handle) {
            println!(
                "descriptor blob: {} bytes, stored {}",
                meta.length,
                format_timestamp(meta.timestamp)
            );
        } else {
            println!("descriptor blob: {} bytes", blob.bytes.len());
        }

        let descriptor = <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob.clone())
            .map_err(|e| anyhow!("decode collection descriptor: {e:?}"))?;
        println!("entity id:      {:X}", descriptor::entity(&descriptor)?);
        match anchor(&Fields::Decoded(descriptor.clone())) {
            Anchor::Root { name, namespace } => {
                match name {
                    Ok(name) => println!("name:           {name}"),
                    Err(e) => println!("name:           <invalid: {e}>"),
                }
                match namespace {
                    Some(Ok(namespace)) => println!("namespace:      {namespace}"),
                    Some(Err(e)) => println!("namespace:      <invalid: {e}>"),
                    None => println!("namespace:      <none>"),
                }
            }
            Anchor::Derived(source) => {
                println!("source:         {}", handle_hex(source));
            }
            Anchor::RetiredScope => println!(
                "anchor:         retired `collection_scope` — this descriptor predates naming, \n\
                 \x20               so it has no name to be listed under"
            ),
            Anchor::Bare => println!("anchor:         <none>"),
            Anchor::Unreadable(why) => println!("anchor:         <{why}>"),
        }
        match descriptor::authority(&descriptor) {
            Some(Ok(authority)) => println!(
                "authority:      {}",
                hex::encode_upper(authority.to_bytes())
            ),
            Some(Err(error)) => println!("authority:      <invalid: {error}>"),
            None => println!("authority:      <none>"),
        }
        println!(
            "representation: {}",
            named_id(
                descriptor::representation(&descriptor)?,
                representation_name(descriptor::representation(&descriptor)?)
            )
        );
        println!(
            "recipe:         {}",
            named_id(
                descriptor::recipe(&descriptor)?,
                recipe_name(descriptor::recipe(&descriptor)?)
            )
        );

        let facts: TribleSet = reader
            .get::<TribleSet, SimpleArchive>(handle)
            .map_err(|e| anyhow!("unarchive descriptor: {e:?}"))?;
        println!("tribles:        {}", facts.len());
        for trible in facts.iter() {
            println!(
                "  {:X} {:X} {}",
                trible.e(),
                trible.a(),
                hex::encode_upper(&trible.data[32..64])
            );
        }

        let counts = rows
            .iter()
            .find(|row| row.handle == handle)
            .map(|row| row.refs)
            .unwrap_or_default();
        println!(
            "records:        {} (commits={} merges={} derives-from={} derives-into={})",
            counts.total(),
            counts.commits,
            counts.merges,
            counts.derives_from,
            counts.derives_into,
        );
        Ok(())
    })();
    let close_res = pile.close().map_err(|e| anyhow!("pile close: {e:?}"));
    res.and(close_res)
}

/// Does this record name that collection?
///
/// A commit and a merge name the collection they are of; a derive names the
/// collection it produces. This is the same question `referenced_collections`
/// tallies, asked one record at a time.
fn names_collection(record: &CollectionRecord, collection: CollectionHandle) -> bool {
    match record {
        CollectionRecord::Commit(commit) => commit.collection() == collection,
        CollectionRecord::Merge(merge) => merge.collection() == collection,
        CollectionRecord::Derive(derive) => derive.target() == collection,
    }
}

fn run_log(path: PathBuf, reference: String, limit: usize, long: bool) -> Result<()> {
    let mut pile = open_refreshed(&path)?;
    let res = (|| -> Result<()> {
        let rows = enumerate(&mut pile)?;
        let handle = resolve(&rows, &reference)?;
        let row = rows
            .iter()
            .find(|row| row.handle == handle)
            .ok_or_else(|| {
                anyhow!(
                    "no record in this pile names collection {}",
                    handle_hex(handle)
                )
            })?;

        print!("collection: blake3:{}", handle_hex(handle));
        match anchor(&row.fields) {
            Anchor::Root { name: Ok(name), .. } => print!("  ({name})"),
            Anchor::Derived(source) => print!("  (derived from {})", handle_hex(source)),
            _ => {}
        }
        println!();
        println!(
            "records: {} (commits={} merges={} derives-from={} derives-into={})",
            row.refs.total(),
            row.refs.commits,
            row.refs.merges,
            row.refs.derives_from,
            row.refs.derives_into,
        );
        println!();

        let short = |bytes: [u8; 32]| abbrev(&hex::encode(bytes), long);
        let mut printed = 0usize;
        let mut skipped = 0usize;
        let records = pile
            .records()
            .map_err(|e| anyhow!("enumerate collection records: {e:?}"))?;
        for record in records {
            let record = record.map_err(|e| anyhow!("decode collection record: {e:?}"))?;
            if !names_collection(&record, handle) {
                continue;
            }
            if limit != 0 && printed == limit {
                skipped += 1;
                continue;
            }
            printed += 1;
            match record {
                CollectionRecord::Commit(commit) => {
                    let signature = match commit.verify_strict() {
                        Ok(()) => "ok".to_owned(),
                        Err(e) => format!("INVALID ({e})"),
                    };
                    println!(
                        "commit  {:X}  data={}  meta={}  signer={}  signature={signature}",
                        commit.id(),
                        short(commit.data().raw),
                        short(commit.metadata().raw),
                        abbrev(&hex::encode_upper(commit.public_key().raw), long),
                    );
                }
                CollectionRecord::Merge(merge) => {
                    let (low, high) = merge.inputs();
                    println!(
                        "merge   {:X}  low={}  high={}  result={}",
                        merge.id(),
                        short(low.raw),
                        short(high.raw),
                        short(merge.result().raw),
                    );
                }
                CollectionRecord::Derive(derive) => {
                    let (input, output) = derive.mapping();
                    println!(
                        "derive  {:X}  input={}  output={}",
                        derive.id(),
                        short(input.raw),
                        short(output.raw),
                    );
                }
            }
        }
        if skipped > 0 {
            println!();
            println!("… {skipped} more (pass --limit 0 for all of them)");
        }
        Ok(())
    })();
    let close_res = pile.close().map_err(|e| anyhow!("pile close: {e:?}"));
    res.and(close_res)
}

/// Every collection a set of records names, without the descriptor lookups.
/// Exposed for tests that want the enumeration independent of blob presence.
#[cfg(test)]
fn referenced_ids(records: &[CollectionRecord]) -> std::collections::BTreeSet<CollectionHandle> {
    let mut out = std::collections::BTreeSet::new();
    for record in records {
        match record {
            CollectionRecord::Commit(commit) => {
                out.insert(commit.collection());
            }
            CollectionRecord::Merge(merge) => {
                out.insert(merge.collection());
            }
            CollectionRecord::Derive(derive) => {
                out.insert(derive.target());
            }
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use std::collections::BTreeSet;
    use triblespace_core::blob::{IntoBlob, MemoryBlobStore};
    use triblespace_core::collection::reach;
    use triblespace_core::collection::records::CollectionName;
    use triblespace_core::collection::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1;
    use triblespace_core::id::fucid;
    use triblespace_core::repo::BlobStorePut;

    /// A descriptor built in-process must round-trip through exactly the path
    /// `show` uses: store the blob, address it by its own handle, read it back
    /// as a `Blob<SimpleArchive>`, and decode. This pins the identity rule the
    /// whole subcommand rests on — the collection id is the hash of the
    /// descriptor blob, never the entity id inside it.
    #[test]
    fn show_decodes_a_descriptor_addressed_by_its_blob_handle() {
        let name = CollectionName::new("inspected").unwrap();
        let namespace = SigningKey::from_bytes(&[7; 32]).verifying_key();
        let authority = SigningKey::from_bytes(&[8; 32]).verifying_key();
        let representation = <SimpleArchive as MetaDescribe>::id();
        let fragment = descriptor::naming(
            &name,
            namespace,
            Some(authority),
            representation,
            TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        );
        let entity_id = fragment.root().expect("the descriptor has one root");

        let mut store = MemoryBlobStore::new();
        let handle: CollectionHandle = store
            .put::<SimpleArchive, _>(fragment.into_facts().to_blob())
            .expect("store descriptor");
        assert_ne!(
            handle.raw[..16],
            <[u8; 16]>::from(entity_id)[..],
            "identity is the blob hash, not the intrinsic entity id"
        );

        let reader = store.reader().expect("reader");
        let blob: Blob<SimpleArchive> = reader.get(handle).expect("read descriptor blob");
        assert_eq!(blob.bytes.len(), 384, "six tribles at 64 bytes each");

        let decoded = <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob)
            .expect("decode descriptor");
        assert_eq!(descriptor::name(&decoded).unwrap().unwrap(), name);
        assert_eq!(descriptor::namespace(&decoded).unwrap().unwrap(), namespace);
        assert_eq!(descriptor::authority(&decoded).unwrap().unwrap(), authority);
        assert_eq!(
            descriptor::representation(&decoded).unwrap(),
            representation
        );
        assert_eq!(
            descriptor::recipe(&decoded).unwrap(),
            TRIBLE_SET_UNION_RECIPE_V1
        );
        assert_eq!(descriptor::entity(&decoded).unwrap(), entity_id);

        // The trible dump `show` prints comes from the same bytes.
        let facts: TribleSet = reader
            .get::<TribleSet, SimpleArchive>(handle)
            .expect("unarchive descriptor");
        assert_eq!(facts.len(), 6);
        let entities: BTreeSet<Id> = facts.iter().map(|t| *t.e()).collect();
        assert_eq!(
            entities,
            BTreeSet::from([entity_id]),
            "all six tribles hang off one intrinsic entity"
        );
    }

    /// `show` must accept the handle in both shapes; `blob inspect` rejects
    /// the bare form with `BadProtocol`.
    #[test]
    fn handles_parse_with_and_without_the_blake3_prefix() {
        let hex = "1c1362fbde47aacdfe3ec872a61b5ff270ef57c30f97ef36511adb1e3536edd2";
        let prefixed = parse_collection_handle(&format!("blake3:{hex}")).expect("prefixed");
        let bare = parse_collection_handle(hex).expect("bare");
        let padded = parse_collection_handle(&format!("  {hex}\n")).expect("padded");
        assert_eq!(prefixed, bare);
        assert_eq!(prefixed, padded);
        assert_eq!(handle_hex(prefixed), hex);

        assert!(parse_collection_handle("sha256:00").is_err());
        assert!(parse_collection_handle("not-hex").is_err());
    }

    /// Enumeration must see collections named by merges and by *both* sides
    /// of a derive, not just commit targets.
    #[test]
    fn enumeration_covers_merges_and_derive_targets() {
        use triblespace_core::collection::records::{CollectionDerive, CollectionMerge};

        fn collection(byte: u8) -> CollectionHandle {
            Inline::new([byte; 32])
        }
        fn data(byte: u8) -> Inline<Hash<Blake3>> {
            Inline::new([byte; 32])
        }

        let records = vec![
            CollectionRecord::Merge(CollectionMerge::new(
                collection(1),
                data(10),
                data(11),
                data(12),
            )),
            CollectionRecord::Derive(CollectionDerive::new(collection(3), data(20), data(21))),
        ];

        assert_eq!(
            referenced_ids(&records),
            // A derive names only its target now: the source is what the
            // target's descriptor says, so enumeration no longer learns a
            // collection from the derive that points into it.
            BTreeSet::from([collection(1), collection(3)])
        );
    }

    #[test]
    fn known_recipe_and_representation_ids_resolve_to_names() {
        assert_eq!(
            recipe_name(TRIBLE_SET_UNION_RECIPE_V1),
            Some("TRIBLE_SET_UNION_RECIPE_V1")
        );
        assert_eq!(
            representation_name(<SimpleArchive as MetaDescribe>::id()),
            Some("SimpleArchive")
        );
        assert_eq!(recipe_name(*fucid()), None);
        assert!(named_id(
            TRIBLE_SET_UNION_RECIPE_V1,
            recipe_name(TRIBLE_SET_UNION_RECIPE_V1)
        )
        .contains("6D64C5F4B9E9B73F57C5F8702AB7FE45"));
    }

    /// Build one descriptor's decoded fields the way `list` sees them.
    fn root(name: &str, namespace_seed: u8) -> Fields {
        let name = CollectionName::new(name).expect("legal name");
        let namespace = SigningKey::from_bytes(&[namespace_seed; 32]).verifying_key();
        Fields::Decoded(
            descriptor::naming(
                &name,
                namespace,
                Some(namespace),
                <SimpleArchive as MetaDescribe>::id(),
                TRIBLE_SET_UNION_RECIPE_V1,
                reach::private(),
            )
            .into_facts(),
        )
    }

    fn row(handle_byte: u8, fields: Fields) -> Enumerated {
        Enumerated {
            handle: Inline::new([handle_byte; 32]),
            refs: Refs::default(),
            fields,
        }
    }

    /// The listing leads with names, so the order has to be the name order —
    /// not the handle order the tally happens to accumulate in. Anything
    /// without a name sorts after everything with one.
    #[test]
    fn named_roots_sort_by_name_ahead_of_everything_unnamed() {
        let mut rows = vec![
            row(9, Fields::Missing),
            row(3, root("wiki", 7)),
            row(4, Fields::Decoded(TribleSet::new())),
            row(1, root("compass", 7)),
        ];
        rows.sort_by_key(sort_key);
        let order: Vec<u8> = rows.iter().map(|row| row.handle.raw[0]).collect();
        assert_eq!(order, vec![1, 3, 4, 9], "compass, wiki, bare, missing");
    }

    /// A collection name and a descriptor handle are disjoint spellings: a
    /// name is at most 32 bytes of `[a-z0-9-]` starting with a letter, and a
    /// handle is 64 hex characters, which fails both the length and the start
    /// rule. So one argument can carry either without a flag to say which.
    #[test]
    fn a_handle_can_never_be_mistaken_for_a_name() {
        let hex = "1c1362fbde47aacdfe3ec872a61b5ff270ef57c30f97ef36511adb1e3536edd2";
        assert!(CollectionName::new(hex).is_err());
        assert!(CollectionName::new(&format!("blake3:{hex}")).is_err());

        let rows = vec![row(3, root("wiki", 7))];
        assert_eq!(
            resolve(&rows, hex).expect("resolves as a handle"),
            parse_collection_handle(hex).expect("parses as a handle")
        );
        assert_eq!(
            resolve(&rows, "wiki").expect("resolves as a name"),
            rows[0].handle
        );
    }

    /// A name identifies a collection within one namespace, so two namespaces
    /// may both have a `wiki` in one pile. That is not a lookup failure to
    /// paper over with a first match — it is an ambiguity the operator has to
    /// break.
    #[test]
    fn a_name_shared_by_two_namespaces_refuses_to_resolve() {
        let rows = vec![row(3, root("wiki", 7)), row(5, root("wiki", 9))];
        let err = resolve(&rows, "wiki").expect_err("ambiguous");
        let text = err.to_string();
        assert!(text.contains("2 collections"), "{text}");
        assert!(text.contains(&handle_hex(rows[0].handle)), "{text}");
        assert!(text.contains(&handle_hex(rows[1].handle)), "{text}");
    }

    /// An unknown name must name what the pile does have, because the whole
    /// reason to type a name is not knowing the handle.
    #[test]
    fn an_unknown_name_reports_the_names_that_exist() {
        let rows = vec![row(3, root("wiki", 7)), row(1, root("compass", 7))];
        let text = resolve(&rows, "memory").expect_err("absent").to_string();
        assert!(text.contains("wiki"), "{text}");
        assert!(text.contains("compass"), "{text}");
    }

    /// The 46 unnamed collections in a live pile are not descriptors that say
    /// nothing — they carry the opaque scope anchor naming replaced. Reporting
    /// that as "no anchor" would describe the reader's vocabulary rather than
    /// the descriptor, and would hide the very reason naming exists.
    #[test]
    fn a_pre_naming_descriptor_is_reported_as_scoped_not_as_empty() {
        use triblespace_core::metadata;
        use triblespace_core::prelude::entity;
        use triblespace_core::trible::Fragment;

        // The retired attribute is gone from the library, so the fixture is
        // assembled from raw parts the same way a pile's bytes are.
        let bare: Fragment = entity! {
            metadata::tag: triblespace_core::collection::records::KIND_COLLECTION_DESCRIPTOR,
        };
        let bare = bare.into_facts();
        assert!(!carries_retired_scope(&bare));
        assert!(matches!(anchor(&Fields::Decoded(bare)), Anchor::Bare));

        let named = root("wiki", 7);
        let Fields::Decoded(named) = named else {
            unreachable!("decoded")
        };
        assert!(
            !carries_retired_scope(&named),
            "a named root carries no scope"
        );
    }

    /// A derive names only its target, and `log` filters records by the same
    /// question the tally counts, so the two can never disagree about which
    /// records belong to a collection.
    #[test]
    fn log_filters_records_by_the_same_question_the_tally_counts() {
        use triblespace_core::collection::records::{CollectionDerive, CollectionMerge};

        fn collection(byte: u8) -> CollectionHandle {
            Inline::new([byte; 32])
        }
        fn data(byte: u8) -> Inline<Hash<Blake3>> {
            Inline::new([byte; 32])
        }

        let records = vec![
            CollectionRecord::Merge(CollectionMerge::new(
                collection(1),
                data(10),
                data(11),
                data(12),
            )),
            CollectionRecord::Derive(CollectionDerive::new(collection(3), data(20), data(21))),
        ];

        for handle in referenced_ids(&records) {
            let named: Vec<&CollectionRecord> = records
                .iter()
                .filter(|record| names_collection(record, handle))
                .collect();
            assert_eq!(
                named.len(),
                1,
                "each fixture record names exactly one collection"
            );
        }
        assert!(!names_collection(&records[0], collection(3)));
        assert!(!names_collection(&records[1], collection(1)));
    }
}
