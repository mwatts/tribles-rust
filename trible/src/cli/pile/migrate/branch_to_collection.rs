//! Direct legacy `Repository` branch to native collection migration.
//!
//! Validation and publication are separate phases. The first phase freezes the
//! selected pin head, opens one later append-only blob snapshot which contains
//! everything that head can name, validates every reachable commit, and
//! prepares every authored native commit entirely in memory. Only after that
//! succeeds may the second phase append dependencies and final `COMMIT`
//! records to the same pile.

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace_core::attribute::Attribute;
use triblespace_core::blob::encodings::longstring::LongString;
use triblespace_core::blob::encodings::simplearchive::SimpleArchive;
use triblespace_core::blob::{Blob, IntoBlob};
use triblespace_core::collection::records::CollectionName;
use triblespace_core::collection::Reach;
use triblespace_core::collection::simplearchive_union::{self, PreparedCollectionCommit};
use triblespace_core::collection::CollectionCommit;
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::{Inline, InlineEncoding};
use triblespace_core::metadata;
use triblespace_core::repo::pile::{Pile, PileReader};
use triblespace_core::repo::{self, BlobStore, BlobStoreGet, CommitHandle, PinSnapshotSource};
use triblespace_core::trible::TribleSet;

use super::super::signing::load_signing_key;

type ArchiveHandle = Inline<Handle<SimpleArchive>>;
type NameHandle = Inline<Handle<LongString>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MigrationReport {
    branch: Id,
    head: Option<CommitHandle>,
    reachable: usize,
    authored: usize,
    contentless_merges: usize,
    unique_targets: usize,
}

pub(super) fn run(
    pile_path: PathBuf,
    branch: String,
    collection_name: String,
    team_root: String,
    signing_key: PathBuf,
) -> Result<()> {
    let name = CollectionName::new(&collection_name)
        .map_err(|error| anyhow!("invalid target collection name {collection_name:?}: {error}"))?;
    let team = parse_team_root(&team_root)?;
    let signer = load_signing_key(&Some(signing_key))?;

    let mut pile = super::super::open_refreshed(&pile_path)?;
    let result = migrate(&mut pile, &branch, &name, team, &signer);
    let close = pile.close().map_err(|error| anyhow!("close pile: {error}"));
    let (report, mappings) = result?;
    close?;
    print_report(
        &pile_path,
        &name,
        team,
        signer.verifying_key(),
        report,
        &mappings,
    );
    Ok(())
}

fn migrate(
    pile: &mut Pile,
    branch_reference: &str,
    name: &CollectionName,
    team: VerifyingKey,
    signer: &SigningKey,
) -> Result<(MigrationReport, Vec<(CommitHandle, CollectionCommit)>)> {
    // Freeze the mutable names first, then take one append-only blob view.
    // A concurrent append may enter the later reader, but cannot change the
    // selected head; every handle reachable from that frozen head predates it.
    // Native appends later remap `pile`, while this reader keeps the validated
    // legacy bytes alive.
    let pins = pile
        .snapshot_pin_heads()
        .context("snapshot active legacy branch pins")?;
    let reader = pile.reader().context("snapshot legacy pile blobs")?;
    let (branch, branch_meta) = resolve_branch(&reader, &pins, branch_reference)?;
    let head = validate_branch_head(&reader, branch, &branch_meta)?;
    // Migrated history stays put. A legacy branch carried no notion of reach,
    // so declaring one here would be inventing a decision on the user's behalf
    // about data they wrote before the question existed -- and it would change
    // the collection's handle, which is the one thing a migration must let the
    // owner predict. `Reach::Private` writes no reach attribute at all, so the
    // descriptor is byte-identical to what this migration produced before the
    // attribute existed. Publishing migrated material stays a deliberate
    // re-commit into a differently-named collection.
    let descriptor = simplearchive_union::descriptor(name, team, Reach::Private);
    let (reachable, contentless_merges, prepared) = match head {
        Some(head) => prepare_reachable(&reader, head, &descriptor)?,
        None => (0, 0, Vec::new()),
    };
    let authored = prepared.len();

    // Preparation above performs no I/O. Once every reachable node has
    // passed, publish each direct source mapping. Exact repeats naturally
    // converge through the content-addressed blob and collection stores.
    let mut mappings = Vec::with_capacity(authored);
    for (source, prepared) in prepared {
        let staged = prepared
            .stage(pile, signer)
            .map_err(|error| anyhow!("stage native collection commit: {error}"))?;
        let commit = staged
            .finalize()
            .map_err(|error| anyhow!("finalize native collection commit: {error}"))?;
        mappings.push((source, commit));
    }
    mappings.sort_unstable_by_key(|(source, _)| source.raw);

    let unique_targets = mappings
        .iter()
        .map(|(_, target)| target.id())
        .collect::<BTreeSet<_>>()
        .len();
    let report = MigrationReport {
        branch,
        head,
        reachable,
        authored,
        contentless_merges,
        unique_targets,
    };
    Ok((report, mappings))
}

fn parse_team_root(text: &str) -> Result<VerifyingKey> {
    let bytes = hex::decode(text.trim()).context("team root must be hexadecimal")?;
    let bytes: [u8; 32] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        anyhow!(
            "team root must be exactly 32 bytes (64 hex characters), found {} bytes",
            bytes.len()
        )
    })?;
    VerifyingKey::from_bytes(&bytes).context("team root is not a valid Ed25519 public key")
}

fn resolve_branch(
    reader: &PileReader,
    pins: &repo::PinSnapshot,
    reference: &str,
) -> Result<(Id, TribleSet)> {
    if let Ok(id) = parse_branch_id(reference) {
        let raw: [u8; 16] = id.into();
        if let Some(handle) = pins.get(&raw).copied() {
            let facts = read_archive(reader, handle, "legacy branch metadata")?.1;
            repo::branch::branch_entity(&facts, id)
                .map_err(|_| anyhow!("pin {id:X} does not contain one branch metadata subject"))?;
            return Ok((id, facts));
        }
    }

    let wanted: NameHandle = reference.to_owned().to_blob().get_handle();
    let mut matches = Vec::new();
    for raw in pins.iter_ordered() {
        let id = Id::new(*raw).expect("pin snapshot contains a nil id");
        let handle = *pins.get(raw).expect("iterated pin has a value");
        let (_, facts) = read_archive(reader, handle, "legacy branch metadata")?;

        let Ok(subject) = repo::branch::branch_entity(&facts, id) else {
            continue;
        };
        let matches_name = {
            let mut current_names = facts
                .iter()
                .filter(|fact| fact.e() == &subject && fact.a() == &metadata::name.id())
                .map(|fact| *fact.v::<Handle<LongString>>());
            let current = current_names.next();
            if current_names.next().is_some() {
                continue;
            }
            match current {
                Some(name) => name == wanted,
                None => super::legacy_branch_name(&facts, id)?.as_deref() == Some(reference),
            }
        };
        if matches_name {
            matches.push((id, facts));
        }
    }
    match matches.len() {
        0 => bail!("no active legacy branch named {reference:?}"),
        1 => Ok(matches.pop().expect("one branch match")),
        count => bail!("{count} active legacy branches are named {reference:?}; use an id"),
    }
}

fn parse_branch_id(text: &str) -> Result<Id> {
    let bytes = hex::decode(text.trim()).context("not a branch id")?;
    let bytes: [u8; 16] = bytes.try_into().map_err(|_| anyhow!("not a branch id"))?;
    Id::new(bytes).ok_or_else(|| anyhow!("branch id cannot be nil"))
}

fn validate_branch_head(
    reader: &PileReader,
    branch: Id,
    facts: &TribleSet,
) -> Result<Option<CommitHandle>> {
    let subject = repo::branch::branch_entity(facts, branch)
        .map_err(|_| anyhow!("legacy branch {branch:X} has no unique metadata subject"))?;
    let head = one_value(facts, subject, &repo::head, "branch head")?;
    if let Some(head) = head {
        let (blob, _) = read_archive(reader, head, "legacy branch head commit")?;
        repo::branch::verify(branch, blob, facts.clone())
            .map_err(|_| anyhow!("legacy branch {branch:X} head signature is invalid"))?;
    }
    Ok(head)
}

fn prepare_reachable(
    reader: &PileReader,
    head: CommitHandle,
    descriptor: &triblespace_core::trible::Fragment,
) -> Result<(usize, usize, Vec<(CommitHandle, PreparedCollectionCommit)>)> {
    let empty_metadata: Blob<SimpleArchive> = TribleSet::new().to_blob();
    // Pile reads verify each content address. A reference cycle would require
    // a BLAKE3 fixed point or collision, so set reachability needs no separate
    // active-path cycle state.
    let mut seen = HashSet::new();
    let mut stack = vec![head];
    let mut contentless_merges = 0;
    let mut prepared = Vec::new();

    while let Some(handle) = stack.pop() {
        if !seen.insert(handle) {
            continue;
        }

        let (_, facts) = read_archive(reader, handle, "legacy commit wrapper")?;
        let Some(first) = facts.iter().next() else {
            bail!("legacy commit {} has an empty wrapper", handle_hex(handle));
        };
        let subject = *first.e();
        if facts.iter().any(|fact| fact.e() != &subject) {
            bail!(
                "legacy commit {} must contain exactly one wrapper subject",
                handle_hex(handle)
            );
        }

        let content = one_value(&facts, subject, &repo::content, "content")?;
        let metadata = one_value(&facts, subject, &metadata::archive, "metadata archive")?;
        let parents: Vec<CommitHandle> = facts
            .iter()
            .filter(|fact| fact.a() == &repo::parent.id())
            .map(|fact| *fact.v::<Handle<SimpleArchive>>())
            .collect();
        stack.extend(parents.iter().copied());

        if let Some(content) = content {
            let data = read_blob(reader, content, "legacy commit content")?;
            repo::commit::verify(data.clone(), facts).map_err(|_| {
                anyhow!(
                    "legacy authored commit {} has an invalid content signature",
                    handle_hex(handle)
                )
            })?;
            let metadata = match metadata {
                Some(handle) => read_blob(reader, handle, "legacy commit metadata archive")?,
                None => empty_metadata.clone(),
            };
            let commit = simplearchive_union::prepare_commit(descriptor, &data, &metadata)
                .map_err(|error| anyhow!("prepare native collection commit: {error}"))?;
            prepared.push((handle, commit));
        } else {
            validate_contentless_merge(&facts, subject, handle, &parents)?;
            contentless_merges += 1;
        }
    }

    Ok((seen.len(), contentless_merges, prepared))
}

fn validate_contentless_merge(
    facts: &TribleSet,
    subject: Id,
    handle: CommitHandle,
    parents: &[CommitHandle],
) -> Result<()> {
    let only_parents = facts
        .iter()
        .all(|fact| fact.e() == &subject && fact.a() == &repo::parent.id());
    let current =
        triblespace_core::macros::entity! { repo::parent*: parents.iter().copied() }.root();
    let historical = triblespace_core::trible::intrinsic_entity_id_v1(
        parents
            .iter()
            .map(|parent| (repo::parent.id(), parent.raw))
            .collect(),
    );
    if parents.len() < 2 || !only_parents || (current != Some(subject) && historical != subject) {
        bail!(
            "contentless legacy commit {} is not a canonical merge",
            handle_hex(handle)
        );
    }
    Ok(())
}

fn read_archive(
    reader: &PileReader,
    handle: ArchiveHandle,
    what: &str,
) -> Result<(Blob<SimpleArchive>, TribleSet)> {
    let blob: Blob<SimpleArchive> = reader
        .get(handle)
        .with_context(|| format!("read {what} {}", handle_hex(handle)))?;
    let facts = blob
        .clone()
        .try_from_blob()
        .with_context(|| format!("decode canonical {what} {}", handle_hex(handle)))?;
    Ok((blob, facts))
}

fn read_blob(
    reader: &PileReader,
    handle: ArchiveHandle,
    what: &str,
) -> Result<Blob<SimpleArchive>> {
    reader
        .get(handle)
        .with_context(|| format!("read {what} {}", handle_hex(handle)))
}

fn one_value<V: InlineEncoding>(
    facts: &TribleSet,
    subject: Id,
    attribute: &Attribute<V>,
    field: &str,
) -> Result<Option<Inline<V>>> {
    let mut values = facts
        .iter()
        .filter(|fact| fact.e() == &subject && fact.a() == &attribute.id())
        .map(|fact| *fact.v::<V>());
    let first = values.next();
    if values.next().is_some() {
        bail!("legacy wrapper subject {subject:X} has repeated {field}");
    }
    Ok(first)
}

fn handle_hex(handle: ArchiveHandle) -> String {
    hex::encode_upper(handle.raw)
}

fn print_report(
    pile_path: &PathBuf,
    name: &CollectionName,
    team: VerifyingKey,
    signer: VerifyingKey,
    report: MigrationReport,
    mappings: &[(CommitHandle, CollectionCommit)],
) {
    println!("same-pile migration: {}", pile_path.display());
    println!("source branch: {:X}", report.branch);
    println!(
        "source head: {}",
        report
            .head
            .map(handle_hex)
            .unwrap_or_else(|| "<none>".to_owned())
    );
    println!("collection name: {name}");
    println!("team root: {}", hex::encode_upper(team.to_bytes()));
    println!("target signer: {}", hex::encode_upper(signer.to_bytes()));
    println!("SOURCE COMMIT                                                     TARGET COMMIT");
    for (source, target) in mappings {
        println!("{}  {:X}", handle_hex(*source), target.id());
    }
    println!(
        "validated {} reachable node(s): {} authored, {} canonical contentless merge(s) skipped",
        report.reachable, report.authored, report.contentless_merges
    );
    println!(
        "{} source authored node(s) -> {} unique native COMMIT(s) ({} many-to-one collapse(s)); replay is idempotent",
        report.authored,
        report.unique_targets,
        report.authored.saturating_sub(report.unique_targets),
    );
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use std::fs;

    use ed25519_dalek::Signer;
    use tempfile::NamedTempFile;
    use triblespace_core::collection::{Collection, CollectionStore};
    use triblespace_core::id::ExclusiveId;
    use triblespace_core::repo::{BlobStorePut, PinStore, Repository};

    use super::*;

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn fact(byte: u8) -> TribleSet {
        let id = Id::new([byte; 16]).unwrap();
        triblespace_core::macros::entity! {
            ExclusiveId::force_ref(&id) @
                metadata::tag: metadata::KIND_MULTI,
        }
        .into_facts()
    }

    fn legacy_fixture(path: &std::path::Path) -> Result<Id> {
        let pile = Pile::open(path)?;
        let mut repository = Repository::new(pile, key(1), fact(9))?;
        let branch = *repository
            .create_branch("legacy", None)
            .map_err(|error| anyhow!("failed to create legacy branch: {error:?}"))?;

        let mut workspace = repository
            .pull(branch)
            .map_err(|error| anyhow!("failed to pull legacy branch: {error:?}"))?;
        workspace.commit(fact(1), "first wrapper");
        repository
            .push(&mut workspace)
            .map_err(|error| anyhow!("failed to push first legacy commit: {error:?}"))?;

        // Same exact content and repo-wide metadata, but another wrapper
        // message and parent: these two source commits must collapse.
        workspace.commit(fact(1), "second wrapper");
        repository
            .push(&mut workspace)
            .map_err(|error| anyhow!("failed to push second legacy commit: {error:?}"))?;

        let mut left = repository
            .pull(branch)
            .map_err(|error| anyhow!("failed to pull left fork: {error:?}"))?;
        let mut right = repository
            .pull(branch)
            .map_err(|error| anyhow!("failed to pull right fork: {error:?}"))?;
        left.commit(fact(2), "left");
        right.commit(fact(3), "right");
        repository
            .push(&mut left)
            .map_err(|error| anyhow!("failed to push left fork: {error:?}"))?;
        repository
            .push(&mut right)
            .map_err(|error| anyhow!("failed to push right fork: {error:?}"))?;
        drop((workspace, left, right));

        let mut pile = repository.into_storage();
        pile.flush()?;
        pile.close()?;
        Ok(branch)
    }

    #[test]
    fn migration_is_exact_many_to_one_and_replay_idempotent() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let branch = legacy_fixture(&path)?;
        let name = CollectionName::new("events").unwrap();
        let team = key(2).verifying_key();
        let signer = key(3);

        let mut pile = super::super::super::open_refreshed(&path)?;
        let (first, first_map) = migrate(&mut pile, "legacy", &name, team, &signer)?;
        assert_eq!(first.branch, branch);
        assert_eq!(first.reachable, 5);
        assert_eq!(first.authored, 4);
        assert_eq!(first.contentless_merges, 1);
        assert_eq!(first.unique_targets, 3);
        assert_eq!(first_map.len(), 4);
        assert_eq!(
            first_map
                .iter()
                .map(|(_, target)| target.id())
                .collect::<BTreeSet<_>>()
                .len(),
            3
        );

        let reader = pile.reader()?;
        let empty_metadata = TribleSet::new().to_blob().get_handle();
        let mut expected_union = TribleSet::new();
        let mut expected_metadata = None;
        for (source, target) in &first_map {
            let (_, wrapper) = read_archive(&reader, *source, "expected source wrapper")?;
            let subject = *wrapper.iter().next().expect("authored wrapper subject").e();
            let data = one_value(&wrapper, subject, &repo::content, "content")?
                .expect("authored source content");
            let metadata = one_value(&wrapper, subject, &metadata::archive, "metadata archive")?
                .unwrap_or(empty_metadata);
            assert_eq!(target.data().raw, data.raw);
            assert_eq!(target.metadata(), metadata);
            expected_metadata.get_or_insert(metadata);

            let facts: TribleSet = reader
                .get(data)
                .map_err(|error| anyhow!("read expected legacy content: {error}"))?;
            expected_union += facts;
        }
        let expected_metadata = expected_metadata.expect("repository fixture carries metadata");
        assert!(first_map
            .iter()
            .all(|(_, target)| target.metadata() == expected_metadata));
        drop(reader);
        let materialized = Collection::new(&mut pile, &name, team, signer.clone(), Reach::Private)
            .materialize()
            .map_err(|error| anyhow!("materialize migrated collection: {error}"))?;
        assert_eq!(materialized, expected_union);

        pile.flush()?;
        let first_len = fs::metadata(&path)?.len();
        let (second, second_map) =
            migrate(&mut pile, &format!("{branch:X}"), &name, team, &signer)?;
        pile.flush()?;
        assert_eq!(second, first);
        assert_eq!(
            first_map
                .iter()
                .map(|(source, target)| (source.raw, target.id()))
                .collect::<Vec<_>>(),
            second_map
                .iter()
                .map(|(source, target)| (source.raw, target.id()))
                .collect::<Vec<_>>()
        );
        assert_eq!(fs::metadata(&path)?.len(), first_len);
        assert_eq!(pile.records()?.collect::<Result<Vec<_>, _>>()?.len(), 3);
        pile.close()?;
        Ok(())
    }

    #[test]
    fn authored_empty_and_absent_metadata_survive_while_merge_is_skipped() -> Result<()> {
        let file = NamedTempFile::new()?;
        let mut pile = Pile::open(file.path())?;
        let author = key(8);

        let empty: Blob<SimpleArchive> = TribleSet::new().to_blob();
        pile.put::<SimpleArchive, _>(empty.clone())?;
        let empty_wrapper =
            repo::commit::commit_metadata(&author, [], None, Some(empty.clone()), None);
        let empty_commit = pile.put::<SimpleArchive, _>(empty_wrapper)?;

        let data: Blob<SimpleArchive> = fact(9).to_blob();
        pile.put::<SimpleArchive, _>(data.clone())?;
        let data_wrapper =
            repo::commit::commit_metadata(&author, [], None, Some(data.clone()), None);
        let data_commit = pile.put::<SimpleArchive, _>(data_wrapper)?;

        let merge_wrapper =
            repo::commit::commit_metadata(&author, [empty_commit, data_commit], None, None, None);
        let merge_blob: Blob<SimpleArchive> = merge_wrapper.to_blob();
        pile.put::<SimpleArchive, _>(merge_blob.clone())?;

        let branch = Id::new([0xD8; 16]).unwrap();
        let branch_name = pile.put::<LongString, _>("empty-and-merge".to_owned())?;
        let branch_meta =
            repo::branch::branch_metadata(&author, branch, branch_name, Some(merge_blob));
        let branch_meta = pile.put::<SimpleArchive, _>(branch_meta)?;
        assert!(matches!(
            pile.update(branch, None, Some(branch_meta))?,
            repo::PushResult::Success()
        ));

        let collection_name = CollectionName::new("empty-preserved").unwrap();
        let team = key(10).verifying_key();
        let signer = key(11);
        let (report, mappings) = migrate(
            &mut pile,
            "empty-and-merge",
            &collection_name,
            team,
            &signer,
        )?;

        assert_eq!(report.reachable, 3);
        assert_eq!(report.authored, 2);
        assert_eq!(report.contentless_merges, 1);
        assert_eq!(mappings.len(), 2);
        let empty_target = mappings
            .iter()
            .find(|(source, _)| *source == empty_commit)
            .map(|(_, target)| *target)
            .expect("authored empty commit has a mapping");
        assert_eq!(empty_target.data().raw, empty.get_handle().raw);
        assert_eq!(empty_target.metadata(), empty.get_handle());
        let data_target = mappings
            .iter()
            .find(|(source, _)| *source == data_commit)
            .map(|(_, target)| *target)
            .expect("authored data commit has a mapping");
        assert_eq!(data_target.data().raw, data.get_handle().raw);
        assert_eq!(data_target.metadata(), empty.get_handle());

        let materialized = Collection::new(&mut pile, &collection_name, team, signer, Reach::Private)
            .materialize()
            .map_err(|error| anyhow!("materialize authored-empty fixture: {error}"))?;
        assert_eq!(materialized, fact(9));
        pile.close()?;
        Ok(())
    }

    #[test]
    fn legacy_hex_shaped_short_name_falls_back_from_absent_id() -> Result<()> {
        let file = NamedTempFile::new()?;
        let mut pile = Pile::open(file.path())?;
        let branch = Id::new([0xE9; 16]).unwrap();
        let hex_name = "ABABABABABABABABABABABABABABABAB";
        let legacy_meta = triblespace_core::macros::entity! {
            repo::branch: branch,
            super::super::legacy_branch_metadata::legacy_name: hex_name,
        }
        .into_facts();
        let legacy_meta = pile.put::<SimpleArchive, _>(legacy_meta)?;
        assert!(matches!(
            pile.update(branch, None, Some(legacy_meta))?,
            repo::PushResult::Success()
        ));

        let pins = pile.snapshot_pin_heads()?;
        let reader = pile.reader()?;
        let (resolved, _) = resolve_branch(&reader, &pins, hex_name)?;
        assert_eq!(resolved, branch);
        pile.close()?;
        Ok(())
    }

    #[test]
    fn unrelated_nonresident_current_name_does_not_block_named_migration() -> Result<()> {
        let file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();
        let intended = legacy_fixture(&path)?;
        let mut pile = super::super::super::open_refreshed(&path)?;

        let unrelated = Id::new([0xFA; 16]).unwrap();
        let nonresident_name: NameHandle = "never-stored-name".to_owned().to_blob().get_handle();
        let unrelated_meta = repo::branch::branch_unsigned(unrelated, nonresident_name, None);
        let unrelated_meta = pile.put::<SimpleArchive, _>(unrelated_meta)?;
        assert!(matches!(
            pile.update(unrelated, None, Some(unrelated_meta))?,
            repo::PushResult::Success()
        ));

        let (report, mappings) = migrate(
            &mut pile,
            "legacy",
            &CollectionName::new("name-resolution-target").unwrap(),
            key(12).verifying_key(),
            &key(13),
        )?;
        assert_eq!(report.branch, intended);
        assert!(!mappings.is_empty());
        pile.close()?;
        Ok(())
    }

    #[test]
    fn authored_random_wrapper_subject_remains_valid_legacy_input() -> Result<()> {
        let file = NamedTempFile::new()?;
        let mut pile = Pile::open(file.path())?;
        let author = key(4);
        let content: Blob<SimpleArchive> = fact(4).to_blob();
        let content_handle = pile.put(content.clone())?;
        let signature = author.sign(&content.bytes);
        let subject = Id::new([0xA5; 16]).unwrap();
        let wrapper = triblespace_core::macros::entity! {
            ExclusiveId::force_ref(&subject) @
                repo::content: content_handle,
                triblespace_core::attestation::signed_by: author.verifying_key(),
                triblespace_core::attestation::signature_r: signature,
                triblespace_core::attestation::signature_s: signature,
        }
        .into_facts();
        let handle = pile.put::<SimpleArchive, _>(wrapper)?;
        let reader = pile.reader()?;

        let (_, wrapper) = read_archive(&reader, handle, "random-subject wrapper")?;
        assert_eq!(
            one_value(&wrapper, subject, &repo::content, "content")?,
            Some(content_handle)
        );
        let descriptor = simplearchive_union::descriptor(
            &CollectionName::new("random-subject").unwrap(),
            key(14).verifying_key(),
            Reach::Private,
        );
        let (reachable, merges, prepared) = prepare_reachable(&reader, handle, &descriptor)?;
        assert_eq!((reachable, merges, prepared.len()), (1, 0, 1));
        pile.close()?;
        Ok(())
    }

    #[test]
    fn invalid_reachable_signature_writes_no_collection_record() -> Result<()> {
        let file = NamedTempFile::new()?;
        let mut pile = Pile::open(file.path())?;
        let author = key(5);
        let valid_content: Blob<SimpleArchive> = fact(4).to_blob();
        pile.put::<SimpleArchive, _>(valid_content.clone())?;
        let valid_wrapper =
            repo::commit::commit_metadata(&author, [], None, Some(valid_content), None);
        let valid_parent = pile.put::<SimpleArchive, _>(valid_wrapper)?;

        let content: Blob<SimpleArchive> = fact(5).to_blob();
        let content_handle = pile.put(content.clone())?;
        let wrong_signature = author.sign(b"not the content archive");
        let subject = Id::new([0xB6; 16]).unwrap();
        let wrapper = triblespace_core::macros::entity! {
            ExclusiveId::force_ref(&subject) @
                repo::content: content_handle,
                repo::parent: valid_parent,
                triblespace_core::attestation::signed_by: author.verifying_key(),
                triblespace_core::attestation::signature_r: wrong_signature,
                triblespace_core::attestation::signature_s: wrong_signature,
        }
        .into_facts();
        let wrapper_blob: Blob<SimpleArchive> = wrapper.to_blob();
        pile.put::<SimpleArchive, _>(wrapper_blob.clone())?;

        let branch = Id::new([0xC7; 16]).unwrap();
        let name = pile.put::<LongString, _>("bad".to_owned())?;
        let branch_meta = repo::branch::branch_metadata(&author, branch, name, Some(wrapper_blob));
        let branch_meta = pile.put::<SimpleArchive, _>(branch_meta)?;
        assert!(matches!(
            pile.update(branch, None, Some(branch_meta))?,
            repo::PushResult::Success()
        ));

        let error = migrate(
            &mut pile,
            "bad",
            &CollectionName::new("target").unwrap(),
            key(6).verifying_key(),
            &key(7),
        )
        .expect_err("bad authored signature must reject the whole migration");
        assert!(error.to_string().contains("invalid content signature"));
        assert!(pile.records()?.collect::<Result<Vec<_>, _>>()?.is_empty());
        pile.close()?;
        Ok(())
    }
}
