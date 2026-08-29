//! `trible pile collection …` — a collection-aware view of a pile.
//!
//! A collection is identified by the blake3 handle of its *descriptor blob*,
//! not by the entity id inside that blob. The descriptor is an ordinary
//! canonical `SimpleArchive` of one intrinsic entity: the
//! `KIND_COLLECTION_DESCRIPTOR` tag, an anchor — `name` for a root, `source`
//! for a derivation — plus its mandatory capability authority, the blob
//! `representation`, and — for a derivation — its content-derived `mapping`,
//! the mapping algorithm, and whatever parameters that mapping carries.
//!
//! Without this module the only way to look at one was
//! `pile blob inspect <PILE> blake3:<HEX>`, which reports "256 bytes, Binary"
//! and nothing else. Here the facts are read with the same
//! [`descriptor`](triblespace_core::collection::descriptor) queries resolution
//! and retention use, so the CLI view can never drift from the semantics.
//!
//! Names are what the listing leads with. A root carries the name it is known
//! by under its authority, and that name — not the 64 hex characters of its
//! descriptor handle — is what an operator came to read. Every subcommand that
//! takes a collection therefore accepts either spelling. `blake3:` and `name:`
//! prefixes disambiguate the unusual case where an arbitrary UTF-8 name itself
//! looks like a bare handle.

use anyhow::{anyhow, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::{
    Rank9AcceleratedSuccinctArchiveBlob, SuccinctArchiveBlob,
};
use triblespace_core::blob::encodings::utf8string::UTF8String;
use triblespace_core::blob::Blob;
use triblespace_core::blob::TryFromBlob;
use triblespace_core::collection::descriptor;
use triblespace_core::collection::records::{CollectionHandle, CollectionRecord};
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
        /// Print handles and authority keys in full.
        #[arg(long)]
        long: bool,
    },
    /// Fully decode one collection descriptor.
    ///
    /// Prints the descriptor handle (the collection identity), the intrinsic
    /// entity id inside the archive, the decoded anchor / representation /
    /// mapping, every trible in the archive, and how many records in this pile
    /// reference the collection.
    Show {
        /// Path to the pile file to read.
        pile: PathBuf,
        /// Collection name, or descriptor handle. Use `name:` or `blake3:`
        /// to disambiguate a name that itself looks like a handle.
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
        /// Collection name, or descriptor handle. Use `name:` or `blake3:`
        /// to disambiguate a name that itself looks like a handle.
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
    } else if id == <Rank9AcceleratedSuccinctArchiveBlob as MetaDescribe>::id() {
        Some("Rank9AcceleratedSuccinctArchiveBlob")
    } else {
        None
    }
}

/// Resolve a mapping-algorithm id against the algorithms declared in core.
fn mapping_algorithm_name(id: Id) -> Option<&'static str> {
    use triblespace_core::collection::lww_register::REGISTER_COORDINATES_MAPPING_V1;
    use triblespace_core::collection::observed_union::OBSERVE_STATES_MAPPING_V1;
    use triblespace_core::collection::succinctarchive_union::{
        RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_BE, RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_LE,
        RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_BE, RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_LE,
        SIMPLE_TO_SUCCINCT_MAPPING_V1,
    };

    if id == SIMPLE_TO_SUCCINCT_MAPPING_V1 {
        Some("SIMPLE_TO_SUCCINCT_MAPPING_V1")
    } else if id == RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_LE {
        Some("RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_LE")
    } else if id == RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_BE {
        Some("RAW_TO_RANK9_ACCELERATED_MAPPING_V1_32_BE")
    } else if id == RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_LE {
        Some("RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_LE")
    } else if id == RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_BE {
        Some("RAW_TO_RANK9_ACCELERATED_MAPPING_V1_64_BE")
    } else if id == OBSERVE_STATES_MAPPING_V1 {
        Some("OBSERVE_STATES_MAPPING_V1")
    } else if id == REGISTER_COORDINATES_MAPPING_V1 {
        Some("REGISTER_COORDINATES_MAPPING_V1")
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
/// may name an encoding or mapping algorithm some other reader owns.
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

/// The short column form for an optional id. Root collections have no mapping,
/// while malformed descriptors remain visibly different from that valid
/// absence.
fn short_optional_id(id: Result<Option<Id>, impl std::fmt::Debug>, long: bool) -> String {
    match id {
        Ok(Some(id)) => abbrev(&format!("{id:X}"), long),
        Ok(None) => "-".to_owned(),
        Err(_) => "<unreadable>".to_owned(),
    }
}

/// The short column form for an optional id whose known values have names.
fn short_optional_named_id(
    id: Result<Option<Id>, impl std::fmt::Debug>,
    name: fn(Id) -> Option<&'static str>,
) -> String {
    match id {
        Ok(Some(id)) => name(id)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{id:X}")),
        Ok(None) => "-".to_owned(),
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
                refs.entry(derive.collection()).or_default().derives_into += 1;
            }
        }
    }
    Ok(refs)
}

/// The descriptor facts, or why they could not be read.
enum Fields {
    Decoded {
        facts: TribleSet,
        name: Result<Option<String>, String>,
    },
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
            Ok(facts) => {
                let name = match descriptor::name(&facts) {
                    Ok(Some(handle)) => reader
                        .get::<Blob<UTF8String>, UTF8String>(handle)
                        .map_err(|error| format!("read collection name attachment: {error}"))
                        .and_then(|name| {
                            std::str::from_utf8(&name.bytes)
                                .map(|name| Some(name.to_owned()))
                                .map_err(|error| {
                                    format!("decode collection name attachment: {error}")
                                })
                        }),
                    Ok(None) => Ok(None),
                    Err(error) => Err(error.to_string()),
                };
                Fields::Decoded { facts, name }
            }
            Err(e) => Fields::Undecodable(format!("{e:?}")),
        }
    }

    fn facts(&self) -> Option<&TribleSet> {
        match self {
            Fields::Decoded { facts, .. } => Some(facts),
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

/// What a descriptor is anchored to. A root is named under its authority and
/// a derivation is anchored by the collection it derives from.
enum Anchor {
    Root(Result<String, String>),
    Derived(CollectionHandle),
    Unreadable(String),
}

fn anchor(fields: &Fields) -> Anchor {
    let (facts, loaded_name) = match fields {
        Fields::Decoded { facts, name } => (facts, name),
        Fields::Missing => return Anchor::Unreadable("descriptor blob not in pile".to_owned()),
        Fields::Undecodable(e) => {
            return Anchor::Unreadable(format!("descriptor undecodable: {e}"))
        }
    };
    match descriptor::source(facts) {
        Ok(Some(source)) => {
            if matches!(descriptor::name(facts), Ok(Some(_))) {
                return Anchor::Unreadable(
                    "descriptor carries both collection_name and collection_source".to_owned(),
                );
            }
            return Anchor::Derived(source);
        }
        Ok(None) => {}
        Err(error) => return Anchor::Unreadable(error.to_string()),
    }
    match loaded_name {
        Ok(Some(name)) => Anchor::Root(Ok(name.clone())),
        Ok(None) => Anchor::Unreadable(
            "descriptor carries neither collection_name nor collection_source".to_owned(),
        ),
        Err(error) => Anchor::Root(Err(error.clone())),
    }
}

/// Sort order for the listing: named roots by name, then derivations by
/// source, then everything with nothing to be called. Ties break on the
/// handle so a listing is stable across runs.
fn sort_key(row: &Enumerated) -> (u8, String, [u8; 32]) {
    match anchor(&row.fields) {
        Anchor::Root(Ok(name)) => (0, name, row.handle.raw),
        Anchor::Root(Err(_)) => (1, String::new(), row.handle.raw),
        Anchor::Derived(source) => (2, handle_hex(source), row.handle.raw),
        Anchor::Unreadable(_) => (3, String::new(), row.handle.raw),
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
/// Explicit prefixes are authoritative. For an unprefixed reference, an exact
/// name wins unless it also parses as a different handle, in which case the
/// operator must disambiguate.
fn resolve(rows: &[Enumerated], reference: &str) -> Result<CollectionHandle> {
    let reference = reference.trim();
    if reference.starts_with("blake3:") {
        return parse_collection_handle(reference);
    }
    let (name, explicit_name) = match reference.strip_prefix("name:") {
        Some(name) => (name, true),
        None => (reference, false),
    };

    let matches: Vec<CollectionHandle> = rows
        .iter()
        .filter(|row| matches!(anchor(&row.fields), Anchor::Root(Ok(found)) if found == name))
        .map(|row| row.handle)
        .collect();

    if !explicit_name {
        if let Ok(handle) = parse_collection_handle(reference) {
            match matches.as_slice() {
                [] => return Ok(handle),
                [named] if *named == handle => return Ok(handle),
                _ => {
                    return Err(anyhow!(
                        "{reference:?} is both a collection name and a bare handle; use \
                         `name:{reference}` or `blake3:{reference}`"
                    ))
                }
            }
        }
    }

    match matches.as_slice() {
        [only] => Ok(*only),
        [] => {
            let known: Vec<String> = rows
                .iter()
                .filter_map(|row| match anchor(&row.fields) {
                    Anchor::Root(Ok(name)) => Some(name),
                    _ => None,
                })
                .collect();
            if known.is_empty() {
                Err(anyhow!(
                    "no collection named {name:?} in this pile; it references no named collections \
                     at all. `pile collection list` shows what it does reference"
                ))
            } else {
                Err(anyhow!(
                    "no collection named {name:?} in this pile; it has: {}",
                    known.join(", ")
                ))
            }
        }
        many => Err(anyhow!(
            "{} collections in this pile are named {name:?} under different authorities. Pass one \
             of the handles instead:\n{}",
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
                match row.fields.facts().map(descriptor::authority) {
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
                    Some(facts) => short_optional_id(descriptor::mapping(facts), long),
                    None => "-".to_owned(),
                },
                match row.fields.facts() {
                    Some(facts) => short_optional_named_id(
                        descriptor::mapping_algorithm(facts),
                        mapping_algorithm_name,
                    ),
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
        for row in &rows {
            match anchor(&row.fields) {
                Anchor::Root(name) => {
                    let mut cells = vec![match &name {
                        Ok(name) => name.clone(),
                        Err(e) => format!("<invalid: {e}>"),
                    }];
                    cells.extend(tail(row));
                    named.push(cells);
                }
                Anchor::Derived(source) => {
                    let mut cells = vec![abbrev(&handle_hex(source), long)];
                    cells.extend(tail(row));
                    derived.push(cells);
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
            (anchorless.len(), "without a readable anchor"),
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
            "MAPPING",
            "ALGORITHM",
        ];
        let mut tail_aligns = vec![
            Align::Right,
            Align::Left,
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
            let mut headers = vec!["NAME"];
            headers.extend(tail_headers.iter().copied());
            let mut aligns = vec![Align::Left];
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
        let fields = Fields::load(&reader, handle);
        match anchor(&fields) {
            Anchor::Root(Ok(name)) => println!("name:           {name}"),
            Anchor::Root(Err(e)) => println!("name:           <invalid: {e}>"),
            Anchor::Derived(source) => {
                println!("source:         {}", handle_hex(source));
            }
            Anchor::Unreadable(why) => println!("anchor:         <{why}>"),
        }
        match descriptor::authority(&descriptor) {
            Ok(authority) => println!(
                "authority:      {}",
                hex::encode_upper(authority.to_bytes())
            ),
            Err(error) => println!("authority:      <invalid: {error}>"),
        }
        println!(
            "representation: {}",
            named_id(
                descriptor::representation(&descriptor)?,
                representation_name(descriptor::representation(&descriptor)?)
            )
        );
        match descriptor::mapping(&descriptor)? {
            Some(mapping) => println!("mapping:        {mapping:X}"),
            None => println!("mapping:        <none>"),
        }
        match descriptor::mapping_algorithm(&descriptor)? {
            Some(algorithm) => println!(
                "mapping algo:   {}",
                named_id(algorithm, mapping_algorithm_name(algorithm))
            ),
            None => println!("mapping algo:   <none>"),
        }

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
        CollectionRecord::Derive(derive) => derive.collection() == collection,
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
            Anchor::Root(Ok(name)) => print!("  ({name})"),
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
                    let (input, output) = (derive.input(), derive.output());
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
                out.insert(derive.collection());
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
    use triblespace_core::collection::succinctarchive_union::SIMPLE_TO_SUCCINCT_MAPPING_V1;
    use triblespace_core::repo::BlobStorePut;

    /// A descriptor built in-process must round-trip through exactly the path
    /// `show` uses: store the blob, address it by its own handle, read it back
    /// as a `Blob<SimpleArchive>`, and decode. This pins the identity rule the
    /// whole subcommand rests on — the collection id is the hash of the
    /// descriptor blob, never the entity id inside it.
    #[test]
    fn show_decodes_a_descriptor_addressed_by_its_blob_handle() {
        let authority = SigningKey::from_bytes(&[8; 32]).verifying_key();
        let representation = <SimpleArchive as MetaDescribe>::id();
        let fragment = descriptor::naming("inspected", authority, representation, reach::private());
        let entity_id = fragment.root().expect("the descriptor has one root");

        let mut store = MemoryBlobStore::new();
        let expected_name = store
            .put::<UTF8String, _>("inspected".to_owned())
            .expect("store descriptor name");
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
        assert_eq!(blob.bytes.len(), 256, "four tribles at 64 bytes each");

        let decoded = <TribleSet as TryFromBlob<SimpleArchive>>::try_from_blob(blob)
            .expect("decode descriptor");
        assert_eq!(descriptor::name(&decoded).unwrap().unwrap(), expected_name);
        assert_eq!(descriptor::authority(&decoded).unwrap(), authority);
        assert_eq!(
            descriptor::representation(&decoded).unwrap(),
            representation
        );
        assert_eq!(descriptor::mapping(&decoded).unwrap(), None);
        assert_eq!(descriptor::mapping_algorithm(&decoded).unwrap(), None);
        assert_eq!(descriptor::entity(&decoded).unwrap(), entity_id);

        // The trible dump `show` prints comes from the same bytes.
        let facts: TribleSet = reader
            .get::<TribleSet, SimpleArchive>(handle)
            .expect("unarchive descriptor");
        assert_eq!(facts.len(), 4);
        let entities: BTreeSet<Id> = facts.iter().map(|t| *t.e()).collect();
        assert_eq!(
            entities,
            BTreeSet::from([entity_id]),
            "all descriptor tribles hang off one intrinsic entity"
        );

        assert!(matches!(
            anchor(&Fields::load(&reader, handle)),
            Anchor::Root(Ok(name)) if name == "inspected"
        ));
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
    fn known_mapping_algorithm_and_representation_ids_resolve_to_names() {
        assert_eq!(
            mapping_algorithm_name(SIMPLE_TO_SUCCINCT_MAPPING_V1),
            Some("SIMPLE_TO_SUCCINCT_MAPPING_V1")
        );
        assert_eq!(
            representation_name(<SimpleArchive as MetaDescribe>::id()),
            Some("SimpleArchive")
        );
        let unknown = Id::new([0xFF; 16]).unwrap();
        assert_eq!(mapping_algorithm_name(unknown), None);
        assert!(named_id(
            SIMPLE_TO_SUCCINCT_MAPPING_V1,
            mapping_algorithm_name(SIMPLE_TO_SUCCINCT_MAPPING_V1)
        )
        .contains("9C8CFEB097B0A336E09D506E8DD361C2"));
    }

    /// Build one descriptor's decoded fields the way `list` sees them.
    fn root(name: &str, authority_seed: u8) -> Fields {
        let authority = SigningKey::from_bytes(&[authority_seed; 32]).verifying_key();
        Fields::Decoded {
            facts: descriptor::naming(
                name,
                authority,
                <SimpleArchive as MetaDescribe>::id(),
                reach::private(),
            )
            .into_facts(),
            name: Ok(Some(name.to_owned())),
        }
    }

    fn anchorless(facts: TribleSet) -> Fields {
        Fields::Decoded {
            facts,
            name: Ok(None),
        }
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
            row(4, anchorless(TribleSet::new())),
            row(1, root("compass", 7)),
        ];
        rows.sort_by_key(sort_key);
        let order: Vec<u8> = rows.iter().map(|row| row.handle.raw[0]).collect();
        assert_eq!(order, vec![1, 3, 4, 9], "compass, wiki, malformed, missing");
    }

    /// UTF-8 names can look exactly like handles. Explicit prefixes resolve
    /// that real ambiguity without restricting what a collection may be named.
    #[test]
    fn explicit_prefixes_disambiguate_a_hex_shaped_name() {
        let hex = "1c1362fbde47aacdfe3ec872a61b5ff270ef57c30f97ef36511adb1e3536edd2";
        let rows = vec![row(3, root(hex, 7)), row(4, root("wiki", 7))];
        assert!(resolve(&rows, hex).is_err(), "bare spelling is ambiguous");
        assert_eq!(
            resolve(&rows, &format!("blake3:{hex}")).expect("explicit handle"),
            parse_collection_handle(hex).expect("parses as a handle")
        );
        assert_eq!(
            resolve(&rows, &format!("name:{hex}")).expect("explicit name"),
            rows[0].handle
        );
        assert_eq!(resolve(&rows, "wiki").unwrap(), rows[1].handle);
    }

    /// A name is scoped by descriptor authority, so two authorities may both
    /// have a `wiki`. Lookup must not paper that over with a first match.
    #[test]
    fn a_name_shared_by_two_authorities_refuses_to_resolve() {
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

    /// Current descriptors require exactly one root name or derived source.
    /// Historical shapes remain visible as malformed, not silently promoted
    /// through a compatibility interpretation.
    #[test]
    fn a_descriptor_without_a_current_anchor_is_reported_as_malformed() {
        use triblespace_core::metadata;
        use triblespace_core::prelude::entity;

        let bare = entity! {
            metadata::tag: triblespace_core::collection::records::KIND_COLLECTION_DESCRIPTOR,
        }
        .into_facts();
        assert!(matches!(
            anchor(&anchorless(bare)),
            Anchor::Unreadable(message) if message.contains("neither collection_name nor collection_source")
        ));
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
