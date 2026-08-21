use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use triblespace::prelude::*;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::BlobStoreMeta;
use triblespace_core::repo::PushResult;
use triblespace_core::trible::TribleSet;

type NameHandle = Inline<Handle<blobencodings::LongString>>;
type BranchMetaHandle = Inline<Handle<blobencodings::SimpleArchive>>;

mod legacy_branch_metadata {
    use super::*;

    // Legacy branch-name attribute (ShortString) used by older triblespace versions.
    attributes! {
        "2E26F8BA886495A8DF04ACF0ED3ACBD4" unsafe as legacy_name: inlineencodings::ShortString;
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Migration {
    #[value(name = "branch-metadata-name")]
    BranchMetadataName,
    #[value(name = "record-kind-descriptions")]
    RecordKindDescriptions,
}

#[derive(Parser, Debug)]
pub enum Command {
    /// List known migrations and whether they are needed for this pile.
    List,
    /// Re-encode a whole pile into the current record framing.
    ///
    /// Only the markers released in v0.46.4 are a compatibility commitment.
    /// Anything written between that release and the current framing is read
    /// once, here, and rewritten; a reframed pile needs none of those decoders
    /// again.
    Reframe {
        /// Destination pile. Must not already exist.
        #[arg(long = "into")]
        into: PathBuf,
    },
    /// Run migrations (all by default, or a single named migration).
    Run {
        /// Optional migration name. If omitted, run all migrations in order.
        #[arg(value_enum)]
        migration: Option<Migration>,
        /// Show what would change without mutating the pile.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// Do not rename duplicate branches (useful for forensic inspection).
        #[arg(long, default_value_t = false)]
        no_rename_duplicates: bool,
    },
}

pub fn run(pile_path: PathBuf, cmd: Command) -> Result<()> {
    match cmd {
        Command::List => list_migrations(&pile_path),
        Command::Reframe { into } => reframe(&pile_path, &into),
        Command::Run {
            migration,
            dry_run,
            no_rename_duplicates,
        } => {
            let rename_duplicates = !no_rename_duplicates;
            match migration {
                None => {
                    migrate_branch_metadata_name(&pile_path, dry_run, rename_duplicates)?;
                    migrate_record_kind_descriptions(&pile_path, dry_run)?;
                }
                Some(Migration::BranchMetadataName) => {
                    migrate_branch_metadata_name(&pile_path, dry_run, rename_duplicates)?;
                }
                Some(Migration::RecordKindDescriptions) => {
                    migrate_record_kind_descriptions(&pile_path, dry_run)?;
                }
            }
            Ok(())
        }
    }
}

fn list_migrations(pile_path: &PathBuf) -> Result<()> {
    let mut pile = super::open_refreshed(pile_path)?;
    let res = (|| -> Result<(), anyhow::Error> {
        let reader = pile.reader().context("pile reader")?;

        let mut missing_name = 0usize;
        // Branches the migration CANNOT fix, tracked separately from the ones
        // it can so the report never implies a repair it will not perform.
        let mut indeterminate_name = 0usize;
        let mut unreadable_meta = 0usize;
        let mut duplicate_names: HashMap<String, usize> = HashMap::new();

        for bid in pile.pins().context("list branches")? {
            let bid = bid.context("branch id")?;
            let Some(meta_handle) = pile.head(bid).context("branch head")? else {
                continue;
            };

            let meta: TribleSet =
                match reader.get::<TribleSet, blobencodings::SimpleArchive>(meta_handle) {
                    Ok(meta) => meta,
                    Err(_) => {
                        // Previously `continue` — silently dropping the branch
                        // from the audit, so a pile with unreadable branch
                        // metadata reported "ok". An audit that cannot read a
                        // branch must say so, not omit it.
                        unreadable_meta += 1;
                        continue;
                    }
                };

            if !has_unique_name(&meta, bid) {
                // The legacy name is what the migration reads, so a branch is
                // only *migratable* when it has one.
                if legacy_branch_name(&meta, bid)
                    .context("read legacy branch name")?
                    .is_some()
                {
                    missing_name += 1;
                } else {
                    // No unique modern name AND no legacy name to migrate
                    // from: either the metadata carries two names or it
                    // carries none. The old code counted this nowhere and
                    // printed nothing, so the branches most in need of repair
                    // were the ones the report stayed silent about.
                    indeterminate_name += 1;
                }
            }

            if let Some(name) =
                load_branch_name(&reader, &meta, bid).context("decode branch name")?
            {
                *duplicate_names.entry(name).or_insert(0) += 1;
            }
        }

        let duplicates = duplicate_names.values().filter(|v| **v > 1).count();

        println!("Known migrations:");
        if missing_name == 0 {
            // "ok" is reserved for a pile with nothing wrong. With
            // indeterminate or unreadable branches present the migration has
            // nothing to DO, which is not the same claim.
            if indeterminate_name == 0 && unreadable_meta == 0 {
                println!("- branch-metadata-name: ok");
            } else {
                println!("- branch-metadata-name: nothing to migrate");
            }
        } else {
            println!("- branch-metadata-name: needed ({missing_name} branch(es))");
        }
        if duplicates > 0 {
            println!(
                "  note: {duplicates} duplicate branch name(s) detected (run migration to auto-rename)"
            );
        }
        // Reported separately from `missing_name` because these are NOT
        // fixed by running the migration — they need `pile reid` or metadata
        // repair. Folding them into the migratable count would promise a
        // repair that does not happen.
        if indeterminate_name > 0 {
            println!(
                "  warning: {indeterminate_name} branch(es) have no determinable name \
                 (no unique metadata::name and no legacy name); the migration cannot \
                 fix these"
            );
        }
        if unreadable_meta > 0 {
            println!(
                "  warning: {unreadable_meta} branch(es) have unreadable metadata and \
                 were not audited"
            );
        }

        let (present, missing) = record_kind_description_census(&reader)?;
        let total = present + missing;
        if missing == 0 {
            println!("- record-kind-descriptions: ok ({total} blob(s) resident)");
        } else {
            println!(
                "- record-kind-descriptions: needed ({missing} of {total} description \
                 blob(s) missing)"
            );
        }
        Ok(())
    })();

    let close_res = pile.close().map_err(|e| anyhow::anyhow!("{e:?}"));
    res.and(close_res)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct BranchInfo {
    branch_id: Id,
    meta_handle: BranchMetaHandle,
    meta_entity: Id,
    name: Option<String>,
    has_head: bool,
    meta: TribleSet,
}

fn migrate_branch_metadata_name(
    pile_path: &PathBuf,
    dry_run: bool,
    rename_duplicates: bool,
) -> Result<()> {
    // The migration rewrites this pile in place, but opening it must still be
    // fail-loud: a corrupt tail is amputated explicitly (`trible pile amputate`),
    // never as a silent side effect of running a migration.
    let mut pile = super::open_refreshed(pile_path)?;

    let res = (|| -> Result<(), anyhow::Error> {
        let reader = pile.reader().context("pile reader")?;
        let iter = pile.pins().context("list branches")?;

        let mut branches: Vec<BranchInfo> = Vec::new();
        for bid in iter {
            let bid = bid.context("branch id")?;
            let Some(meta_handle) = pile.head(bid).context("branch head")? else {
                continue;
            };

            let meta: TribleSet =
                match reader.get::<TribleSet, blobencodings::SimpleArchive>(meta_handle) {
                    Ok(meta) => meta,
                    Err(_) => continue,
                };

            let Ok(meta_entity) = triblespace_core::repo::branch::branch_entity(&meta, bid) else {
                // Not a branch metadata blob we recognize; skip.
                continue;
            };

            let head_count = find!(
                head: BranchMetaHandle,
                pattern!(&meta, [{ meta_entity @ triblespace_core::repo::head: ?head }])
            )
            .count();
            // Name migration is orthogonal to head repair. Preserve every
            // scoped head fact byte-for-byte; `has_head` is only a preference
            // when choosing which duplicate name to keep.
            let has_head = head_count >= 1;

            let name = load_branch_name(&reader, &meta, bid).context("decode branch name")?;

            branches.push(BranchInfo {
                branch_id: bid,
                meta_handle,
                meta_entity,
                name,
                has_head,
                meta,
            });
        }

        let mut migrated = 0usize;
        for info in branches.iter_mut() {
            let needs_name = !has_unique_name(&info.meta, info.branch_id);
            if !needs_name {
                continue;
            }

            let legacy_name = legacy_branch_name(&info.meta, info.branch_id)
                .context("read legacy branch name")?;
            let Some(legacy_name) = legacy_name else {
                continue;
            };

            if dry_run {
                println!(
                    "Would migrate branch {:X}: add metadata::name = {legacy_name:?}",
                    info.branch_id
                );
                continue;
            }

            let name_handle: NameHandle = pile
                .put::<blobencodings::LongString, _>(legacy_name.clone())
                .context("store branch name blob")?;

            let new_meta = rewrite_branch_meta(&info.meta, info.meta_entity, name_handle);
            let new_meta_handle: BranchMetaHandle = pile
                .put(new_meta.clone())
                .context("store updated branch metadata")?;

            match pile
                .update(
                    info.branch_id,
                    Some(info.meta_handle),
                    Some(new_meta_handle),
                )
                .map_err(|e| anyhow!("update branch {:X}: {e:?}", info.branch_id))?
            {
                PushResult::Success() => {
                    info.meta_handle = new_meta_handle;
                    info.meta = new_meta;
                    info.name = Some(legacy_name);
                    migrated += 1;
                }
                PushResult::Conflict(_) => {
                    anyhow::bail!(
                        "branch {:X} advanced concurrently; rerun migration",
                        info.branch_id
                    );
                }
            }
        }

        let mut renamed = 0usize;
        if rename_duplicates {
            renamed =
                rename_duplicate_branch_names(&mut pile, &branches, dry_run).context("dedupe")?;
        }

        if dry_run {
            println!("Dry run complete.");
        } else {
            println!("Migrated {migrated} branch metadata blobs.");
            if rename_duplicates {
                println!("Renamed {renamed} duplicate branch(es).");
            }
        }
        Ok(())
    })();

    let close_res = pile.close().map_err(|e| anyhow::anyhow!("{e:?}"));
    res.and(close_res)?;
    Ok(())
}

fn has_unique_name(meta: &TribleSet, branch_id: Id) -> bool {
    let Ok(branch_entity) = triblespace_core::repo::branch::branch_entity(meta, branch_id) else {
        return false;
    };
    let mut names = find!(
        handle: NameHandle,
        pattern!(meta, [{ branch_entity @ triblespace_core::metadata::name: ?handle }])
    );
    names.next().is_some() && names.next().is_none()
}

fn legacy_branch_name(meta: &TribleSet, branch_id: Id) -> Result<Option<String>> {
    let Ok(branch_entity) = triblespace_core::repo::branch::branch_entity(meta, branch_id) else {
        return Ok(None);
    };
    let mut names = find!(
        name: String,
        pattern!(meta, [{ branch_entity @ legacy_branch_metadata::legacy_name: ?name }])
    );
    let Some(name) = names.next() else {
        return Ok(None);
    };
    if names.next().is_some() {
        return Ok(None);
    }
    Ok(Some(name))
}

fn load_branch_name(
    reader: &impl BlobStoreGet,
    meta: &TribleSet,
    branch_id: Id,
) -> Result<Option<String>> {
    let Ok(branch_entity) = triblespace_core::repo::branch::branch_entity(meta, branch_id) else {
        return Ok(None);
    };
    let mut names = find!(
        handle: NameHandle,
        pattern!(meta, [{ branch_entity @ triblespace_core::metadata::name: ?handle }])
    );

    let Some(handle) = names.next() else {
        return legacy_branch_name(meta, branch_id);
    };
    if names.next().is_some() {
        return Ok(None);
    }

    let view: View<str> = reader
        .get(handle)
        .map_err(|err| anyhow!("read branch name blob: {err:?}"))?;
    Ok(Some(view.as_ref().to_string()))
}

fn rewrite_branch_meta(meta: &TribleSet, meta_entity: Id, name_handle: NameHandle) -> TribleSet {
    let mut out = TribleSet::new();
    let name_attr = triblespace_core::metadata::name.id();
    let legacy_attr = legacy_branch_metadata::legacy_name.id();
    for t in meta.iter() {
        if t.e() == &meta_entity && (t.a() == &name_attr || t.a() == &legacy_attr) {
            continue;
        }
        out.insert(t);
    }
    out += entity! { ExclusiveId::force_ref(&meta_entity) @ triblespace_core::metadata::name: name_handle };
    out
}

fn rename_duplicate_branch_names(
    pile: &mut Pile,
    branches: &[BranchInfo],
    dry_run: bool,
) -> Result<usize> {
    let mut by_name: HashMap<&str, Vec<&BranchInfo>> = HashMap::new();
    for info in branches {
        let Some(name) = info.name.as_deref() else {
            continue;
        };
        by_name.entry(name).or_default().push(info);
    }

    let reader = pile.reader().context("pile reader")?;

    let mut renamed = 0usize;
    for (name, items) in by_name {
        if items.len() < 2 {
            continue;
        }

        // Choose the canonical branch to keep the name. Prefer non-empty branches
        // (those with a commit head), then prefer the most recently updated branch
        // metadata blob as a stable tie-breaker.
        let mut best: Option<(&BranchInfo, u64)> = None;
        for info in &items {
            let ts = reader
                .metadata(info.meta_handle)
                .ok()
                .flatten()
                .map(|m| m.timestamp)
                .unwrap_or(0);
            match best {
                None => best = Some((info, ts)),
                Some((cur, cur_ts)) => {
                    let better = match (cur.has_head, info.has_head) {
                        (false, true) => true,
                        (true, false) => false,
                        _ => ts > cur_ts,
                    };
                    if better {
                        best = Some((info, ts));
                    }
                }
            }
        }
        let Some((canonical, _)) = best else {
            continue;
        };

        for orphan in items
            .into_iter()
            .filter(|i| i.branch_id != canonical.branch_id)
        {
            let suffix = format!("{:X}", orphan.branch_id);
            let prefix_len = 8.min(suffix.len());
            let new_name = format!("{name}--orphan-{}", &suffix[..prefix_len]);

            if dry_run {
                println!(
                    "Would rename duplicate branch {:X} {name:?} -> {new_name:?} (kept {:X})",
                    orphan.branch_id, canonical.branch_id
                );
                continue;
            }

            let name_handle: NameHandle = pile
                .put::<blobencodings::LongString, _>(new_name.clone())
                .context("store renamed branch name blob")?;

            let meta: TribleSet = reader
                .get::<TribleSet, blobencodings::SimpleArchive>(orphan.meta_handle)
                .context("read duplicate branch metadata")?;

            let new_meta = rewrite_branch_meta(&meta, orphan.meta_entity, name_handle);
            let new_meta_handle: BranchMetaHandle = pile
                .put(new_meta.clone())
                .context("store renamed branch metadata")?;

            match pile
                .update(
                    orphan.branch_id,
                    Some(orphan.meta_handle),
                    Some(new_meta_handle),
                )
                .map_err(|e| anyhow!("update branch {:X}: {e:?}", orphan.branch_id))?
            {
                PushResult::Success() => {
                    renamed += 1;
                }
                PushResult::Conflict(_) => {
                    anyhow::bail!(
                        "branch {:X} advanced concurrently while renaming; rerun migration",
                        orphan.branch_id
                    );
                }
            }
        }
    }

    Ok(renamed)
}


/// Count how many of this reader's record-kind description blobs the pile
/// already holds.
///
/// The census is what makes a re-run honest: a pile that already carries the
/// descriptions reports nothing to do rather than repeating its whole worklist.
fn record_kind_description_census(
    reader: &(impl BlobStoreGet + triblespace_core::repo::BlobStoreList),
) -> Result<(usize, usize)> {
    let mut present = 0usize;
    let mut missing = 0usize;
    for blob in triblespace_core::repo::pile::description_blobs() {
        let handle = blob.get_handle();
        if reader
            .contains_blob(handle)
            .map_err(|err| anyhow!("residency lookup: {err:?}"))?
        {
            present += 1;
        } else {
            missing += 1;
        }
    }
    Ok((present, missing))
}

/// Make every record kind this binary writes resolvable inside the pile.
///
/// A record kind is the 32-byte handle of a description archive. That makes it
/// resolvable in principle; storing the archives makes it resolvable *here*,
/// so a reader holding nothing but the file can say what any record in it is.
fn migrate_record_kind_descriptions(pile_path: &PathBuf, dry_run: bool) -> Result<()> {
    let mut pile = super::open_refreshed(pile_path)?;

    let res = (|| -> Result<(), anyhow::Error> {
        let reader = pile.reader().context("pile reader")?;
        let (present, missing) = record_kind_description_census(&reader)?;
        drop(reader);

        if missing == 0 {
            println!(
                "record-kind-descriptions: nothing to do ({present} description blob(s) \
                 already resident)."
            );
            return Ok(());
        }
        if dry_run {
            println!(
                "Would store {missing} record-kind description blob(s) ({present} already \
                 resident)."
            );
            return Ok(());
        }

        // Content addressing makes this idempotent, so the already-resident
        // ones cost nothing and the count above is a report, not a plan the
        // write path has to honour.
        let stored = pile
            .publish_record_kind_descriptions()
            .map_err(|err| anyhow!("store record-kind descriptions: {err:?}"))?;
        println!("Published {stored} record-kind description blob(s) ({missing} were missing).");
        Ok(())
    })();

    let close_res = pile.close().map_err(|e| anyhow::anyhow!("{e:?}"));
    res.and(close_res)?;
    Ok(())
}


/// Re-encode `pile_path` into a fresh pile at `destination`.
///
/// The destination is created empty and must not already exist: a reframe that
/// appended into an existing file would produce exactly the mixed framing it
/// exists to eliminate.
///
/// Every commit in the result is verified afterwards. A signature covers a
/// domain-separated transcript over the record's fields rather than the bytes
/// of its frame, so re-encoding cannot invalidate one — but that is a claim
/// about two layers, and the cheap way to be sure is to check rather than to
/// reason.
fn reframe(pile_path: &PathBuf, destination: &PathBuf) -> Result<()> {
    if destination.exists() {
        anyhow::bail!(
            "destination {} already exists; reframe writes a fresh pile",
            destination.display()
        );
    }
    std::fs::File::create(destination)
        .with_context(|| format!("create {}", destination.display()))?;

    let mut out = Pile::open(destination).map_err(|e| anyhow!("open destination: {e:?}"))?;
    let res = (|| -> Result<(), anyhow::Error> {
        let stats = triblespace_core::repo::pile::reframe_into(pile_path, &mut out)
            .map_err(|e| anyhow!("reframe: {e}"))?;
        println!(
            "Reframed into {}:\n  blobs: {}\n  pin updates: {}\n  wants: {}\n  \
             collection records: {}\n  dropped inert records: {}",
            destination.display(),
            stats.blobs,
            stats.pin_updates,
            stats.wants,
            stats.collection_records,
            stats.dropped_inert,
        );

        let (mut checked, mut invalid) = (0usize, 0usize);
        for record in out.records().map_err(|e| anyhow!("read records: {e:?}"))? {
            let record = record.map_err(|e| anyhow!("read record: {e:?}"))?;
            if let triblespace_core::collection::CollectionRecord::Commit(commit) = record {
                checked += 1;
                if commit.verify_strict().is_err() {
                    invalid += 1;
                }
            }
        }
        println!("  commits verified: {checked} (signature-invalid {invalid})");
        if invalid > 0 {
            anyhow::bail!(
                "{invalid} commit(s) failed strict verification after reframing; \
                 keep the source pile and report this"
            );
        }
        Ok(())
    })();

    let close_res = out.close().map_err(|e| anyhow::anyhow!("{e:?}"));
    res.and(close_res)?;
    Ok(())
}
