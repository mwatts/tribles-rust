use anyhow::{anyhow, Result};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

use triblespace_core::repo::pile::{Pile, ReadError};

pub mod blob;
pub mod branch;
mod diagnose;
mod extract;
mod merge;
mod migrate;
pub mod net;
pub mod pin;
mod reid;
mod signing;
mod squash;

#[derive(Parser)]
pub enum PileCommand {
    /// Operations on branches stored in a pile file. Branches are
    /// the named-pin specialization that holds a commit-chain head;
    /// `branch list` filters to those and shows commit-aware info.
    /// For the generic pin view (all pins regardless of role), see
    /// `pile pin`.
    Branch {
        #[command(subcommand)]
        cmd: branch::Command,
    },
    /// Operations on the legacy pin storage primitive. Branches and unnamed
    /// legacy/application pins show up here; private node policy lives in a
    /// signer-owned collection. For the branch-specific view, see `pile branch`.
    Pin {
        #[command(subcommand)]
        cmd: pin::Command,
    },
    /// Operations on blobs stored in a pile file.
    Blob {
        #[command(subcommand)]
        cmd: blob::Command,
    },
    /// Provision durable signing identities for pile-backed writers.
    SigningKey {
        #[command(subcommand)]
        cmd: signing::Command,
    },
    /// Merge source branch heads into a target branch.
    Merge {
        /// Path to the pile file to modify
        pile: PathBuf,
        /// Target branch id (hex)
        target: String,
        /// Source branch id(s) (hex)
        #[arg(num_args = 1..)]
        sources: Vec<String>,
        /// Optional signing key path. The file should contain a 64-char hex seed.
        #[arg(long)]
        signing_key: Option<PathBuf>,
    },
    /// Create a new empty pile file.
    ///
    /// This is mainly a cross-platform convenience; a plain `touch` on
    /// Unix-like systems achieves the same result.
    Create {
        /// Path to the pile file to create
        path: PathBuf,
    },
    /// Diagnostic helpers for inspecting and repairing piles.
    Diagnose {
        #[command(subcommand)]
        cmd: diagnose::Command,
    },
    /// DESTRUCTIVE: truncate a pile at its first malformed or torn record,
    /// deleting everything after it.
    ///
    /// This explicit repair entry point loads every structurally complete
    /// record and cuts the file back to the last valid boundary—everything
    /// past that point is permanently destroyed. Complete opaque envelopes
    /// are crossed, and a torn opaque envelope can be cut at its known start;
    /// unknown unenveloped markers are refused because their length is
    /// unknowable and may indicate a newer pile format. This is last-resort
    /// surgery for a torn append left by a crashed write: back the file up
    /// first, inspect the reported boundary with `trible pile diagnose
    /// record-at`, confirm the tail is genuinely torn, and only then run this
    /// by hand with that exact boundary.
    Amputate {
        /// Path to the pile file to amputate (TRUNCATED in place)
        path: PathBuf,
        /// Exact byte offset to truncate to.
        ///
        /// This must equal the boundary reported by the current reader. The
        /// explicit value prevents an old tool's generic repair suggestion
        /// from becoming an unexamined destructive command.
        #[arg(long, value_name = "BYTE_OFFSET")]
        truncate_to: usize,
    },
    /// Migrate legacy pile metadata to the current schemas.
    Migrate {
        /// Path to the pile file to modify
        pile: PathBuf,
        #[command(subcommand)]
        cmd: migrate::Command,
    },
    /// Distributed pile sync over iroh (p2p QUIC connections).
    Net {
        #[command(subcommand)]
        cmd: net::Command,
    },
    /// Squash all branch histories into single commits in a new pile.
    ///
    /// For each branch, the full accumulated content and metadata are
    /// checked out and written as a single commit. Only blobs reachable
    /// from the squashed content are copied. The result is a minimal
    /// pile with clean commit timestamps and no orphaned data.
    Squash {
        /// Source pile file
        source: PathBuf,
        /// Destination pile file (will be created)
        dest: PathBuf,
        /// Only include these branches (by name or hex ID). If omitted, all branches are included.
        #[arg(long)]
        include: Vec<String>,
        /// Exclude these branches (by name or hex ID).
        #[arg(long)]
        exclude: Vec<String>,
        /// Optional signing key path
        #[arg(long)]
        signing_key: Option<PathBuf>,
    },
    /// Stream one branch's commit chain into a fresh pile.
    ///
    /// The scalable single-branch alternative to `squash`: the branch's
    /// content is never materialized. Commits are walked oldest → newest
    /// and each content delta blob is copied as raw bytes into the
    /// destination, where a fresh commit is minted per original commit
    /// (preserving messages and per-commit deltas). Peak memory stays
    /// proportional to one commit's blob references, so this works on
    /// piles far larger than RAM. Prints a per-commit ladder table with
    /// running cumulative trible counts.
    Extract {
        /// Source pile file
        source: PathBuf,
        /// Destination pile file (will be created)
        dest: PathBuf,
        /// Branch to extract (name or hex id)
        #[arg(long)]
        branch: String,
    },
    /// Re-id every branch into a new pile, preserving names + full history.
    ///
    /// Each branch keeps its name and head commit, but receives a freshly
    /// minted branch id; the full reachable blob graph is copied unchanged
    /// (unlike `squash`, which collapses history). Use this to de-alias two
    /// piles that share branch ids before `cat` + `branch consolidate
    /// --by-name`.
    Reid {
        /// Source pile file
        source: PathBuf,
        /// Destination pile file (will be created)
        dest: PathBuf,
        /// Optional signing key path
        #[arg(long)]
        signing_key: Option<PathBuf>,
    },
}

/// Turn a pile read failure into an operator-facing diagnostic without
/// suggesting destructive repair for an unsupported record marker.
pub(crate) fn pile_read_error(path: &Path, err: ReadError) -> anyhow::Error {
    match err {
        err @ ReadError::UnsupportedRecord { .. } => anyhow!(
            "pile {} contains a record format unsupported by this binary ({err}); this is likely \
             version skew. Upgrade trible to a reader that recognizes the marker. The pile was \
             left unchanged",
            path.display()
        ),
        err @ ReadError::CorruptPile { .. } => anyhow!(
            "pile {} has a malformed or incomplete known record ({err}); this reader cannot \
             prove that the remaining bytes are a disposable torn write. The pile was left \
             unchanged. Inspect the reported boundary with `trible pile diagnose record-at {} \
             <BYTE_OFFSET>`. Only after making a backup and independently confirming that every \
             byte from that boundary onward may be destroyed, run `trible pile amputate {} \
             --truncate-to <BYTE_OFFSET>`",
            path.display(),
            path.display(),
            path.display()
        ),
        err => anyhow!("read pile {}: {err}", path.display()),
    }
}

/// Open a pile and load its records via `refresh`, failing loud on a corrupt,
/// torn, or unsupported tail without modifying it. Deliberate repair of
/// genuine corruption stays a separate, boundary-confirmed
/// `trible pile amputate <path> --truncate-to <byte-offset>` step.
pub(crate) fn open_refreshed(path: &Path) -> Result<Pile> {
    let mut pile = Pile::open(path).map_err(|e| anyhow!("open pile {}: {e:?}", path.display()))?;
    if let Err(err) = pile.refresh() {
        let _ = pile.close();
        return Err(pile_read_error(path, err));
    }
    Ok(pile)
}

pub fn run(cmd: PileCommand) -> Result<()> {
    match cmd {
        PileCommand::Branch { cmd } => branch::run(cmd),
        PileCommand::Pin { cmd } => pin::run(cmd),
        PileCommand::Blob { cmd } => blob::run(cmd),
        PileCommand::SigningKey { cmd } => signing::run(cmd),
        PileCommand::Merge {
            pile,
            target,
            sources,
            signing_key,
        } => merge::run(pile, target, sources, signing_key),
        PileCommand::Create { path } => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }

            // Pile::open no longer auto-creates files (v0.32.1), so we
            // explicitly touch the path first. Fine if the file already
            // exists — fs::File::create truncates empty-or-not, and
            // piles are append-only so an empty file is the initial
            // state.
            fs::File::create(&path)?;

            let pile: Pile = Pile::open(&path)?;
            // Explicit close makes the empty pile durable and avoids Drop warnings.
            pile.close().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            Ok(())
        }
        PileCommand::Net { cmd } => net::run(cmd),
        PileCommand::Diagnose { cmd } => diagnose::run(cmd),
        PileCommand::Amputate { path, truncate_to } => {
            let mut pile = Pile::open(&path)?;
            // Boundary comparison and truncation happen under the same
            // exclusive file lock. A preflight `refresh` here would leave a
            // check-then-act race with another repair process.
            let amputated = match pile.amputate_at(truncate_to) {
                Ok(amputated) => amputated,
                Err(ReadError::CorruptPile { valid_length }) => {
                    let _ = pile.close();
                    return Err(anyhow!(
                        "refusing destructive repair: --truncate-to {truncate_to} does not match \
                         the current reader's boundary {valid_length}; the pile was left unchanged"
                    ));
                }
                Err(err) => {
                    let _ = pile.close();
                    return Err(pile_read_error(&path, err));
                }
            };
            pile.close()
                .map_err(|e| anyhow::anyhow!("close pile: {e:?}"))?;
            if amputated {
                println!(
                    "{}: amputated tail at confirmed boundary {truncate_to}",
                    path.display()
                );
            } else {
                println!("{}: already valid", path.display());
            }
            Ok(())
        }
        PileCommand::Migrate { pile, cmd } => migrate::run(pile, cmd),
        PileCommand::Squash {
            source,
            dest,
            include,
            exclude,
            signing_key,
        } => squash::run(source, dest, signing_key, include, exclude),
        PileCommand::Extract {
            source,
            dest,
            branch,
        } => extract::run(source, dest, branch),
        PileCommand::Reid {
            source,
            dest,
            signing_key,
        } => reid::run(source, dest, signing_key),
    }
}
