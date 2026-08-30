use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use triblespace_core::blob::encodings::UnknownBlob;
use triblespace_core::blob::Blob;
use triblespace_core::collection::{CollectionRead, CollectionRecord};
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::{Inline, INLINE_LEN};
use triblespace_core::repo::pile::{Pile, PileSnapshot};
use triblespace_core::repo::{
    ArtifactHandle, ArtifactOfferSnapshot, ArtifactOfferStore, BlobStoreGet, BlobStoreList,
    SnapshotSource,
};

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
    /// it reaches, and only then are native records appended to that pile. The
    /// target authority defaults to the migration signer. Choosing a different
    /// authority is allowed, but admission of the resulting commits then needs
    /// a resident proof granting the signer exact WRITE access to the
    /// collection.
    BranchToCollection {
        /// Legacy branch to migrate, by exact name or 32-hex-character id.
        #[arg(long)]
        branch: String,
        /// Immutable name of the target root collection.
        #[arg(long)]
        collection_name: String,
        /// Capability trust root. Defaults to the migration signer.
        #[arg(long)]
        authority: Option<String>,
        /// Durable target signing-key file (64-hex-character seed).
        #[arg(long)]
        signing_key: PathBuf,
    },
    /// Seed durable local service intent for resident artifacts already
    /// published by native collection records.
    ///
    /// This is an explicit one-time bridge for piles populated before normal
    /// publication wrote OFFER records. It observes one frozen record view and
    /// one resident-blob view, never scans unrelated blobs, creates no WANTs,
    /// and does not make the selected artifacts garbage-collection roots.
    SeedArtifactOffers {
        /// Show the exact frozen census without appending OFFER records.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
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
            authority,
            signing_key,
        } => branch_to_collection::run(pile_path, branch, collection_name, authority, signing_key),
        Command::SeedArtifactOffers { dry_run } => seed_artifact_offers(&pile_path, dry_run),
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

/// Bound command-side working batches independently of pile record framing.
///
/// `ArtifactOfferStore::offer_all` remains the durable bulk boundary. Keeping
/// this internal avoids turning migration mechanics into a tuning surface.
const ARTIFACT_OFFER_SEED_BATCH_SIZE: usize = 4_096;

#[derive(Debug, Default, Eq, PartialEq)]
struct ArtifactOfferSeedPlan {
    records: usize,
    valid_commits: usize,
    invalid_commits: usize,
    merges: usize,
    derives: usize,
    missing: BTreeSet<ArtifactHandle>,
    candidates: BTreeSet<ArtifactHandle>,
    novel: Vec<ArtifactHandle>,
}

impl ArtifactOfferSeedPlan {
    fn missing_count(&self) -> usize {
        self.missing.len()
    }

    fn already_offered_count(&self) -> usize {
        self.candidates.len().saturating_sub(self.novel.len())
    }
}

/// Explicitly recover local OFFER intent for artifacts published before OFFER
/// was part of the normal publication boundary.
///
/// Collection records and resident blobs are frozen together. The operational
/// offer view is then sampled once, so concurrent appenders can only become
/// work for a later idempotent invocation. All candidate payloads are
/// content-validated before the first append; a corrupt resident candidate
/// therefore fails loud and is never newly offered.
fn seed_artifact_offers(pile_path: &PathBuf, dry_run: bool) -> Result<()> {
    let mut pile = super::open_refreshed(pile_path)?;

    let res = (|| -> Result<(), anyhow::Error> {
        let snapshot = pile
            .snapshot()
            .context("freeze native collection and blob view")?;
        let records = snapshot
            .records()
            .context("freeze native collection records")?
            .collect::<Result<Vec<_>, _>>()
            .context("read frozen native collection records")?;
        let offers = pile
            .offers_snapshot()
            .context("freeze artifact offer view")?;
        let plan = plan_artifact_offers(records, &snapshot, &offers)?;

        print_artifact_offer_seed_plan(&plan, dry_run);
        if dry_run || plan.novel.is_empty() {
            return Ok(());
        }

        for batch in plan.novel.chunks(ARTIFACT_OFFER_SEED_BATCH_SIZE) {
            pile.offer_all(batch.iter().copied())
                .context("append seeded artifact offers")?;
        }
        println!(
            "seed-artifact-offers: seeded {} previously unoffered artifact(s).",
            plan.novel.len()
        );
        Ok(())
    })();

    let close_res = pile.close().map_err(|error| anyhow!("close pile: {error}"));
    res.and(close_res)?;
    Ok(())
}

fn print_artifact_offer_seed_plan(plan: &ArtifactOfferSeedPlan, dry_run: bool) {
    let prefix = if dry_run {
        "seed-artifact-offers dry run"
    } else {
        "seed-artifact-offers"
    };
    println!(
        "{prefix}: {} native record(s): {} valid COMMIT, {} invalid COMMIT, {} MERGE, {} DERIVE.",
        plan.records, plan.valid_commits, plan.invalid_commits, plan.merges, plan.derives,
    );
    println!(
        "{prefix}: {} resident candidate artifact(s), {} already offered, {} missing reference(s) skipped; {} new OFFER(s).",
        plan.candidates.len(),
        plan.already_offered_count(),
        plan.missing_count(),
        plan.novel.len(),
    );
}

fn plan_artifact_offers(
    records: Vec<CollectionRecord>,
    snapshot: &PileSnapshot,
    offers: &ArtifactOfferSnapshot,
) -> Result<ArtifactOfferSeedPlan> {
    let mut plan = ArtifactOfferSeedPlan {
        records: records.len(),
        ..ArtifactOfferSeedPlan::default()
    };
    let mut recursive_roots = BTreeSet::new();
    let mut direct_roots = BTreeSet::new();

    for record in records {
        match record {
            CollectionRecord::Commit(commit) => {
                if commit.verify_strict().is_err() {
                    plan.invalid_commits += 1;
                    continue;
                }
                plan.valid_commits += 1;
                recursive_roots.insert(commit.collection().transmute());
                recursive_roots.insert(Handle::<UnknownBlob>::from_hash(commit.data()));
                recursive_roots.insert(commit.metadata().transmute());
            }
            CollectionRecord::Merge(merge) => {
                plan.merges += 1;
                direct_roots.insert(merge.collection().transmute());
                direct_roots.insert(Handle::<UnknownBlob>::from_hash(merge.result()));
            }
            CollectionRecord::Derive(derive) => {
                plan.derives += 1;
                let (_, output) = (derive.input(), derive.output());
                direct_roots.insert(derive.collection().transmute());
                direct_roots.insert(Handle::<UnknownBlob>::from_hash(output));
            }
        }
    }

    let mut queued = BTreeSet::new();
    let mut queue = VecDeque::new();
    for handle in recursive_roots {
        if resident(snapshot, handle)? {
            queued.insert(handle);
            queue.push_back(handle);
        } else {
            plan.missing.insert(handle);
        }
    }

    while let Some(handle) = queue.pop_front() {
        let blob = validate_candidate(snapshot, handle)?;
        plan.candidates.insert(handle);

        // This is intentionally the canonical default `BlobChildren`
        // traversal: non-overlapping aligned 32-byte values, with a trailing
        // partial value ignored. We spell it out here because BlobChildren's
        // convenient default suppresses a corrupt-parent load error, while a
        // migration must validate the entire plan before its first OFFER.
        for chunk in blob.bytes.as_ref().chunks_exact(INLINE_LEN) {
            let mut raw = [0u8; INLINE_LEN];
            raw.copy_from_slice(chunk);
            let child = Inline::<Handle<UnknownBlob>>::new(raw);
            if !queued.contains(&child) && resident(snapshot, child)? {
                queued.insert(child);
                queue.push_back(child);
            }
        }
    }

    for handle in direct_roots {
        if plan.candidates.contains(&handle) {
            continue;
        }
        if !resident(snapshot, handle)? {
            plan.missing.insert(handle);
            continue;
        }
        validate_candidate(snapshot, handle)?;
        plan.candidates.insert(handle);
    }

    plan.novel = plan
        .candidates
        .iter()
        .copied()
        .filter(|handle| !offers.contains(*handle))
        .collect();
    Ok(plan)
}

fn resident(snapshot: &PileSnapshot, handle: ArtifactHandle) -> Result<bool> {
    snapshot
        .contains_blob(handle)
        .map_err(|error| anyhow!("inspect artifact {}: {error}", artifact_hex(handle)))
}

fn validate_candidate(
    snapshot: &PileSnapshot,
    handle: ArtifactHandle,
) -> Result<Blob<UnknownBlob>> {
    snapshot
        .get::<Blob<UnknownBlob>, UnknownBlob>(handle)
        .map_err(|error| {
            anyhow!(
                "refusing to offer corrupt resident artifact {}: {error}",
                artifact_hex(handle)
            )
        })
}

fn artifact_hex(handle: ArtifactHandle) -> String {
    hex::encode_upper(handle.raw)
}

fn list_migrations(pile_path: &PathBuf) -> Result<()> {
    let mut pile = super::open_refreshed(pile_path)?;
    let res = (|| -> Result<(), anyhow::Error> {
        let snapshot = pile.snapshot().context("pile snapshot")?;
        let (present, missing) = record_kind_description_census(&snapshot)?;
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
        let snapshot = pile.snapshot().context("pile snapshot")?;
        let (present, missing) = record_kind_description_census(&snapshot)?;

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
        let snapshot = out.snapshot().context("snapshot reframed pile")?;
        for record in snapshot
            .records()
            .map_err(|e| anyhow!("read records: {e:?}"))?
        {
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

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    use ed25519_dalek::SigningKey;
    use tempfile::TempDir;
    use triblespace_core::blob::{Blob, Bytes};
    use triblespace_core::collection::{
        CollectionCommit, CollectionDerive, CollectionMerge, CollectionRecord, CollectionStore,
    };
    use triblespace_core::inline::encodings::hash::{Blake3, Hash};
    use triblespace_core::repo::{ArtifactOfferStore, BlobStorePut, WantStore};

    use super::*;

    fn fresh_pile(directory: &TempDir, name: &str) -> PathBuf {
        let path = directory.path().join(name);
        std::fs::File::create(&path).unwrap();
        path
    }

    fn put(pile: &mut Pile, bytes: impl Into<Vec<u8>>) -> ArtifactHandle {
        pile.put::<UnknownBlob, _>(Blob::<UnknownBlob>::new(Bytes::from_source(bytes.into())))
            .unwrap()
    }

    fn data(handle: ArtifactHandle) -> Inline<Hash<Blake3>> {
        Inline::new(handle.raw)
    }

    fn invalid_signature(mut commit: CollectionCommit) -> CollectionCommit {
        let mut bytes = commit.to_bytes();
        *bytes.last_mut().unwrap() ^= 0x80;
        commit = CollectionCommit::from_bytes(bytes);
        assert!(commit.verify_strict().is_err());
        commit
    }

    fn offers(path: &PathBuf) -> BTreeSet<ArtifactHandle> {
        let mut pile = Pile::open(path).unwrap();
        pile.refresh().unwrap();
        let result = pile.offers_snapshot().unwrap().iter().collect();
        pile.close().unwrap();
        result
    }

    #[test]
    fn seed_selects_only_published_resident_artifacts_with_record_specific_ownership() {
        let directory = tempfile::tempdir().unwrap();
        let path = fresh_pile(&directory, "seed.pile");
        let mut pile = Pile::open(&path).unwrap();

        let descriptor_child = put(&mut pile, b"commit descriptor child".to_vec());
        let descriptor = put(&mut pile, descriptor_child.raw.to_vec());
        let data_child = put(&mut pile, b"commit data child".to_vec());
        let trailing_partial_child = put(&mut pile, b"trailing partial child".to_vec());
        let mut data_payload = data_child.raw.to_vec();
        data_payload.extend_from_slice(&trailing_partial_child.raw[..17]);
        let commit_data = put(&mut pile, data_payload);
        let metadata_child = put(&mut pile, b"commit metadata child".to_vec());
        let metadata = put(&mut pile, metadata_child.raw.to_vec());
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[7; 32]),
            descriptor.transmute(),
            data(commit_data),
            metadata.transmute(),
        );
        pile.insert(CollectionRecord::Commit(commit)).unwrap();

        let merge_descriptor_child = put(&mut pile, b"merge descriptor child".to_vec());
        let merge_descriptor = put(&mut pile, merge_descriptor_child.raw.to_vec());
        let merge_result_child = put(&mut pile, b"merge result child".to_vec());
        let merge_result = put(&mut pile, merge_result_child.raw.to_vec());
        let merge_low = put(&mut pile, b"merge low".to_vec());
        let merge_high = put(&mut pile, b"merge high".to_vec());
        pile.insert(CollectionRecord::Merge(CollectionMerge::new(
            merge_descriptor.transmute(),
            data(merge_low),
            data(merge_high),
            data(merge_result),
        )))
        .unwrap();

        let derive_target_child = put(&mut pile, b"derive target child".to_vec());
        let derive_target = put(&mut pile, derive_target_child.raw.to_vec());
        let derive_output_child = put(&mut pile, b"derive output child".to_vec());
        let derive_output = put(&mut pile, derive_output_child.raw.to_vec());
        let derive_input = put(&mut pile, b"derive input".to_vec());
        pile.insert(CollectionRecord::Derive(CollectionDerive::new(
            derive_target.transmute(),
            data(derive_input),
            data(derive_output),
        )))
        .unwrap();

        let invalid_child = put(&mut pile, b"invalid commit child".to_vec());
        let invalid_data = put(&mut pile, invalid_child.raw.to_vec());
        let invalid_descriptor = put(&mut pile, b"invalid descriptor".to_vec());
        let invalid_metadata = put(&mut pile, b"invalid metadata".to_vec());
        let invalid = invalid_signature(CollectionCommit::sign(
            &SigningKey::from_bytes(&[8; 32]),
            invalid_descriptor.transmute(),
            data(invalid_data),
            invalid_metadata.transmute(),
        ));
        pile.insert(CollectionRecord::Commit(invalid)).unwrap();

        let orphan = put(&mut pile, b"unrelated resident orphan".to_vec());
        pile.offer(commit_data).unwrap();
        pile.close().unwrap();

        let length_before_dry_run = std::fs::metadata(&path).unwrap().len();
        seed_artifact_offers(&path, true).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            length_before_dry_run
        );
        assert_eq!(offers(&path), BTreeSet::from([commit_data]));

        seed_artifact_offers(&path, false).unwrap();
        let expected = BTreeSet::from([
            descriptor_child,
            descriptor,
            data_child,
            commit_data,
            metadata_child,
            metadata,
            merge_descriptor,
            merge_result,
            derive_target,
            derive_output,
        ]);
        assert_eq!(offers(&path), expected);

        let excluded = [
            merge_descriptor_child,
            merge_result_child,
            merge_low,
            merge_high,
            derive_target_child,
            derive_output_child,
            derive_input,
            trailing_partial_child,
            invalid_child,
            invalid_data,
            invalid_descriptor,
            invalid_metadata,
            orphan,
        ];
        let actual = offers(&path);
        for handle in excluded {
            assert!(!actual.contains(&handle), "unexpected offer for {handle:?}");
        }

        let length_after_first_seed = std::fs::metadata(&path).unwrap().len();
        seed_artifact_offers(&path, false).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            length_after_first_seed,
            "re-running the seed must append no duplicate OFFER records"
        );
    }

    #[test]
    fn seed_skips_missing_references_without_creating_wants() {
        let directory = tempfile::tempdir().unwrap();
        let path = fresh_pile(&directory, "missing.pile");
        let mut pile = Pile::open(&path).unwrap();
        let resident_data = put(&mut pile, b"resident commit data".to_vec());
        let missing_descriptor = Inline::new([0x31; 32]);
        let missing_metadata = Inline::new([0x32; 32]);
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[9; 32]),
            missing_descriptor,
            data(resident_data),
            missing_metadata,
        );
        pile.insert(CollectionRecord::Commit(commit)).unwrap();
        pile.close().unwrap();

        seed_artifact_offers(&path, false).unwrap();
        assert_eq!(offers(&path), BTreeSet::from([resident_data]));

        let mut pile = Pile::open(&path).unwrap();
        assert_eq!(pile.wants().unwrap().count(), 0);
        pile.close().unwrap();
    }

    #[test]
    fn corrupt_candidate_fails_before_any_offer_is_appended() {
        let directory = tempfile::tempdir().unwrap();
        let path = fresh_pile(&directory, "corrupt.pile");
        let corrupt_payload = b"candidate payload unique to corruption test".to_vec();
        let mut pile = Pile::open(&path).unwrap();
        let descriptor = put(&mut pile, b"valid direct descriptor".to_vec());
        let result = put(&mut pile, corrupt_payload.clone());
        pile.insert(CollectionRecord::Merge(CollectionMerge::new(
            descriptor.transmute(),
            Inline::new([0x41; 32]),
            Inline::new([0x42; 32]),
            data(result),
        )))
        .unwrap();
        pile.close().unwrap();

        let mut bytes = Vec::new();
        OpenOptions::new()
            .read(true)
            .open(&path)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        let payload_offset = bytes
            .windows(corrupt_payload.len())
            .position(|window| window == corrupt_payload)
            .expect("unique payload must be in pile");
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(payload_offset as u64)).unwrap();
        file.write_all(b"C").unwrap();
        file.sync_all().unwrap();

        let error = seed_artifact_offers(&path, false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("refusing to offer corrupt resident artifact"),
            "{error:#}"
        );
        assert!(offers(&path).is_empty());
    }
}
