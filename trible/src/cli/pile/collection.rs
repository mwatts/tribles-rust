//! `trible pile collection …` — a collection-aware view of a pile.
//!
//! A collection is identified by the blake3 handle of its *descriptor blob*,
//! not by the entity id inside that blob. The descriptor is a small canonical
//! `SimpleArchive` carrying exactly four tribles on one intrinsic entity: the
//! `KIND_COLLECTION_DESCRIPTOR` tag plus the extrinsic `scope`, the blob
//! `representation`, and the join `recipe`.
//!
//! Without this module the only way to look at one was
//! `pile blob inspect <PILE> blake3:<HEX>`, which reports "256 bytes, Binary"
//! and nothing else. Here the descriptor is decoded with
//! [`CollectionDescriptor::decode`] — the same decoder resolution and
//! retention use — so the CLI view can never drift from the semantics.

use anyhow::{anyhow, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;

use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::encodings::succinctarchive::SuccinctArchiveBlob;
use triblespace_core::blob::Blob;
use triblespace_core::collection::records::{
    CollectionDescriptor, CollectionHandle, CollectionRecord,
};
use triblespace_core::collection::store::CollectionStore;
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::hash::{Blake3, Hash};
use triblespace_core::inline::Inline;
use triblespace_core::metadata::MetaDescribe;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::{BlobStore, BlobStoreGet, BlobStoreMeta};
use triblespace_core::trible::TribleSet;

use super::open_refreshed;

#[derive(Parser)]
pub enum Command {
    /// List every distinct collection the pile references, one row each.
    ///
    /// A collection is "referenced" when some commit, merge, or derive
    /// record in the pile names it. The descriptor blob itself may or may
    /// not be present; rows whose descriptor is missing or undecodable say
    /// so instead of being silently dropped.
    List {
        /// Path to the pile file to inspect.
        path: PathBuf,
        /// Also show per-collection record counts and the descriptor blob's
        /// storage timestamp and size.
        #[arg(long)]
        metadata: bool,
    },
    /// Fully decode one collection descriptor.
    ///
    /// Prints the descriptor handle (the collection identity), the intrinsic
    /// entity id inside the archive, the decoded scope / representation /
    /// recipe, every trible in the archive, and how many records in this
    /// pile reference the collection.
    Show {
        /// Path to the pile file to read.
        pile: PathBuf,
        /// Descriptor handle, with or without the `blake3:` prefix.
        handle: String,
    },
}

pub fn run(cmd: Command) -> Result<()> {
    match cmd {
        Command::List { path, metadata } => run_list(path, metadata),
        Command::Show { pile, handle } => run_show(pile, handle),
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
        anyhow!("parse collection handle {handle:?}: {e:?} (expected `blake3:<64 hex>` or bare hex)")
    })?;
    Ok(hash.into())
}

fn handle_hex(handle: CollectionHandle) -> String {
    hex::encode(handle.raw)
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
                refs.entry(derive.source()).or_default().derives_from += 1;
                refs.entry(derive.target()).or_default().derives_into += 1;
            }
        }
    }
    Ok(refs)
}

/// The three decoded fields, or why they could not be decoded.
enum Fields {
    Decoded(CollectionDescriptor),
    Missing,
    Undecodable(String),
}

impl Fields {
    fn load<R: BlobStoreGet>(reader: &R, handle: CollectionHandle) -> Self {
        let blob: Blob<SimpleArchive> = match reader.get(handle) {
            Ok(blob) => blob,
            Err(_) => return Fields::Missing,
        };
        match CollectionDescriptor::decode(&blob) {
            Ok(descriptor) => Fields::Decoded(descriptor),
            Err(e) => Fields::Undecodable(format!("{e:?}")),
        }
    }
}

fn run_list(path: PathBuf, metadata: bool) -> Result<()> {
    let mut pile = open_refreshed(&path)?;
    let res = (|| -> Result<()> {
        let refs = referenced_collections(&mut pile)?;
        if refs.is_empty() {
            println!("(no collections referenced by pile {})", path.display());
            return Ok(());
        }

        let reader = pile.reader().map_err(|e| anyhow!("pile reader: {e:?}"))?;
        println!("collections: {}", refs.len());
        for (handle, counts) in &refs {
            print!("  {}", handle_hex(*handle));
            match Fields::load(&reader, *handle) {
                Fields::Decoded(descriptor) => {
                    print!(
                        "  scope={:X}  representation={}  recipe={}",
                        descriptor.scope()?,
                        named_id(
                            descriptor.representation()?,
                            representation_name(descriptor.representation()?)
                        ),
                        named_id(descriptor.recipe()?, recipe_name(descriptor.recipe()?)),
                    );
                }
                Fields::Missing => print!("  <descriptor blob not in pile>"),
                Fields::Undecodable(e) => print!("  <descriptor undecodable: {e}>"),
            }
            println!();

            if metadata {
                println!(
                    "      records={} (commits={} merges={} derives-from={} derives-into={})",
                    counts.total(),
                    counts.commits,
                    counts.merges,
                    counts.derives_from,
                    counts.derives_into,
                );
                match reader.metadata(*handle) {
                    Ok(Some(meta)) => {
                        println!(
                            "      descriptor blob: {} bytes, stored {}",
                            meta.length,
                            format_timestamp(meta.timestamp)
                        );
                    }
                    Ok(None) => println!("      descriptor blob: absent"),
                    Err(e) => println!("      descriptor blob: metadata error ({e:?})"),
                }
            }
        }
        Ok(())
    })();
    let close_res = pile.close().map_err(|e| anyhow!("pile close: {e:?}"));
    res.and(close_res)
}

fn format_timestamp(millis: u64) -> String {
    use chrono::{DateTime, Utc};
    use std::time::{Duration, UNIX_EPOCH};

    let dt = UNIX_EPOCH + Duration::from_millis(millis);
    DateTime::<Utc>::from(dt).to_rfc3339()
}

fn run_show(path: PathBuf, handle: String) -> Result<()> {
    let handle = parse_collection_handle(&handle)?;
    let mut pile = open_refreshed(&path)?;
    let res = (|| -> Result<()> {
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

        let descriptor = CollectionDescriptor::decode(&blob)
            .map_err(|e| anyhow!("decode collection descriptor: {e:?}"))?;
        println!("entity id:      {:X}", descriptor.entity_id()?);
        println!("scope:          {:X}", descriptor.scope()?);
        println!(
            "representation: {}",
            named_id(
                descriptor.representation()?,
                representation_name(descriptor.representation()?)
            )
        );
        println!(
            "recipe:         {}",
            named_id(descriptor.recipe()?, recipe_name(descriptor.recipe()?))
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

        // The pile has already been replayed, so the reference tally is a
        // walk over an in-memory map rather than a second file scan.
        let refs = referenced_collections(&mut pile)?;
        let counts = refs.get(&handle).copied().unwrap_or_default();
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
                out.insert(derive.source());
                out.insert(derive.target());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use triblespace_core::blob::MemoryBlobStore;
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
        let scope = *fucid();
        let representation = <SimpleArchive as MetaDescribe>::id();
        let descriptor = CollectionDescriptor::naming(scope, representation, TRIBLE_SET_UNION_RECIPE_V1);

        let mut store = MemoryBlobStore::new();
        let handle: CollectionHandle = store
            .put::<SimpleArchive, _>(descriptor.to_blob())
            .expect("store descriptor");
        assert_eq!(
            handle, descriptor.handle(),
            "the stored blob's handle is the collection identity"
        );
        assert_ne!(
            handle.raw[..16],
            <[u8; 16]>::from(descriptor.entity_id().unwrap())[..],
            "identity is the blob hash, not the intrinsic entity id"
        );

        let reader = store.reader().expect("reader");
        let blob: Blob<SimpleArchive> = reader.get(handle).expect("read descriptor blob");
        assert_eq!(blob.bytes.len(), 256, "four tribles at 64 bytes each");

        let decoded = CollectionDescriptor::decode(&blob).expect("decode descriptor");
        assert_eq!(decoded.scope().unwrap(), scope);
        assert_eq!(decoded.representation().unwrap(), representation);
        assert_eq!(decoded.recipe().unwrap(), TRIBLE_SET_UNION_RECIPE_V1);
        assert_eq!(decoded.entity_id().unwrap(), descriptor.entity_id().unwrap());

        // The trible dump `show` prints comes from the same bytes.
        let facts: TribleSet = reader
            .get::<TribleSet, SimpleArchive>(handle)
            .expect("unarchive descriptor");
        assert_eq!(facts.len(), 4);
        let entities: BTreeSet<Id> = facts.iter().map(|t| *t.e()).collect();
        assert_eq!(
            entities,
            BTreeSet::from([descriptor.entity_id().unwrap()]),
            "all four tribles hang off one intrinsic entity"
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
    fn enumeration_covers_merges_and_both_derive_sides() {
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
            CollectionRecord::Derive(CollectionDerive::new(
                collection(2),
                collection(3),
                data(20),
                data(21),
            )),
        ];

        assert_eq!(
            referenced_ids(&records),
            BTreeSet::from([collection(1), collection(2), collection(3)])
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
        assert!(named_id(TRIBLE_SET_UNION_RECIPE_V1, recipe_name(TRIBLE_SET_UNION_RECIPE_V1))
            .contains("6D64C5F4B9E9B73F57C5F8702AB7FE45"));
    }
}
