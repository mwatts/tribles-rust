use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use triblespace_core::collection::CollectionStore;
use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::{BlobStore, BlobStoreGet};

mod branch_to_collection;

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Migration {
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
    /// Re-sign one verified legacy branch as native collection commits.
    ///
    /// This is deliberately a same-pile migration: one frozen pin observation
    /// selects the head, a later append-only blob snapshot validates everything
    /// it reaches, and only then are native records appended to that pile. A
    /// An omitted authority makes admission explicitly open. An authority-root
    /// signer may bootstrap its own exact WRITE proof; any other signer must
    /// designate an existing exact proof.
    BranchToCollection {
        /// Legacy branch to migrate, by exact name or 32-hex-character id.
        #[arg(long)]
        branch: String,
        /// Immutable name of the target root collection.
        #[arg(long)]
        collection_name: String,
        /// Public-key namespace used only to name the target collection.
        #[arg(long)]
        namespace: String,
        /// Optional capability trust root. Omit for explicitly open admission.
        #[arg(long)]
        authority: Option<String>,
        /// Exact WRITE proof id. Requires --authority; omit only when the
        /// signing key itself is the authority root.
        #[arg(long, requires = "authority")]
        proof: Option<String>,
        /// Durable target signing-key file (64-hex-character seed).
        #[arg(long)]
        signing_key: PathBuf,
    },
    /// Run migrations (all by default, or a single named migration).
    Run {
        /// Optional migration name. If omitted, run all migrations in order.
        #[arg(value_enum)]
        migration: Option<Migration>,
        /// Show what would change without mutating the pile.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

pub fn run(pile_path: PathBuf, cmd: Command) -> Result<()> {
    match cmd {
        Command::List => list_migrations(&pile_path),
        Command::Reframe { into } => reframe(&pile_path, &into),
        Command::BranchToCollection {
            branch,
            collection_name,
            namespace,
            authority,
            proof,
            signing_key,
        } => branch_to_collection::run(
            pile_path,
            branch,
            collection_name,
            namespace,
            authority,
            proof,
            signing_key,
        ),
        Command::Run { migration, dry_run } => {
            match migration {
                None | Some(Migration::RecordKindDescriptions) => {
                    migrate_record_kind_descriptions(&pile_path, dry_run)?
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
        let (present, missing) = record_kind_description_census(&reader)?;
        let total = present + missing;

        println!("Known migrations:");
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
             capability proofs: {}\n  collection records: {}\n  dropped inert records: {}",
            destination.display(),
            stats.blobs,
            stats.pin_updates,
            stats.wants,
            stats.capability_proofs,
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
