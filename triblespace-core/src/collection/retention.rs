//! Strong retention roots for authorized collection commits.
//!
//! A signed, locally authorized [`super::CollectionCommit`] is durable ground
//! truth. Its collection definition and commit record are direct roots; its
//! data and metadata are recursive roots which own their resident attachments.
//!
//! Unsigned `MERGE` and `DERIVE` records are reproducible cache work, not
//! authority. They therefore add no strong roots, whether their equations are
//! active, accepted-but-ungrounded, or merely present in the store. A separate
//! cache policy may retain selected equations and materializations without
//! weakening this ground-truth boundary.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::UnknownBlob;
use crate::blob::BlobEncoding;
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding};
use crate::repo::{BlobStoreMeta, RetentionRoots};

use super::{CollectionData, CollectionResolution, DiscoveredCollectionRecords};

/// A collection retention plan could not prove that every required blob stays
/// available.
#[derive(Debug)]
pub enum CollectionRetentionError<MetadataError> {
    /// An admitted commit's collection has no resident canonical definition.
    MissingDefinition {
        /// Intrinsic collection id.
        collection: Id,
    },
    /// Storage metadata lookup failed while establishing residency.
    Metadata {
        /// Handle whose residency could not be established.
        handle: Inline<Handle<UnknownBlob>>,
        /// Backend failure.
        source: MetadataError,
    },
    /// An admitted commit's signed data is absent.
    MissingCommitData {
        /// Intrinsic commit-record id.
        commit: Id,
        /// Missing signed data blob.
        data: CollectionData,
    },
    /// An admitted commit's signed metadata is absent.
    MissingCommitMetadata {
        /// Intrinsic commit-record id.
        commit: Id,
        /// Missing metadata blob.
        metadata: Inline<Handle<SimpleArchive>>,
    },
}

impl<MetadataError: fmt::Display> fmt::Display for CollectionRetentionError<MetadataError> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDefinition { collection } => {
                write!(f, "collection {collection:X} has no canonical definition")
            }
            Self::Metadata { handle, source } => write!(
                f,
                "failed to inspect collection retention handle {}: {source}",
                hex::encode_upper(handle.raw),
            ),
            Self::MissingCommitData { commit, data } => write!(
                f,
                "admitted commit {commit:X} has missing signed data {}",
                hex::encode_upper(data.raw),
            ),
            Self::MissingCommitMetadata { commit, metadata } => write!(
                f,
                "admitted commit {commit:X} has missing metadata {}",
                hex::encode_upper(metadata.raw),
            ),
        }
    }
}

impl<MetadataError> Error for CollectionRetentionError<MetadataError>
where
    MetadataError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Metadata { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Plan strong roots for every admitted commit in `resolution`.
///
/// `resolution` must be the result of the caller's explicit commit-signer
/// authorization and claim-validation policy over `records`. Unauthorized
/// commits are absent from [`CollectionResolution::admitted_claims`], so they
/// cannot retain anything. Admitted unsigned equations are deliberately
/// ignored: validation and activation do not turn cache work into authority.
///
/// Every admitted commit retains its referenced collection definition and
/// canonical commit record directly, plus its signed data and metadata
/// recursively. Planning fails if any required definition, data, or metadata
/// blob is absent instead of manufacturing a root for unavailable ground
/// truth.
///
/// The returned roots are a pure result, not a persisted retained-scope
/// registry. A collector must rediscover, authorize, resolve, and plan again on
/// each later pass. A legacy pin must not be removed until some higher layer
/// durably owns that recurring policy.
pub fn plan_collection_retention<D, R>(
    records: &DiscoveredCollectionRecords,
    resolution: &CollectionResolution<D>,
    reader: &R,
) -> Result<RetentionRoots, CollectionRetentionError<<R as BlobStoreMeta>::MetaError>>
where
    R: BlobStoreMeta + ?Sized,
{
    let admitted = resolution.admitted_claims();
    let definitions: BTreeMap<_, _> = records
        .definitions()
        .iter()
        .map(|definition| (definition.id(), definition))
        .collect();
    let mut roots = RetentionRoots::new();

    for claim in records.commits() {
        if !admitted.contains(&claim.id()) {
            continue;
        }

        let definition = definitions.get(&claim.collection()).ok_or(
            CollectionRetentionError::MissingDefinition {
                collection: claim.collection(),
            },
        )?;
        let definition_handle = definition.to_blob().get_handle();
        if !require_resident(reader, definition_handle)? {
            return Err(CollectionRetentionError::MissingDefinition {
                collection: claim.collection(),
            });
        }
        roots.retain_direct(definition_handle);
        roots.retain_direct(claim.to_blob().get_handle());

        let data_handle = Handle::<UnknownBlob>::from_hash(claim.data());
        if !require_resident(reader, data_handle)? {
            return Err(CollectionRetentionError::MissingCommitData {
                commit: claim.id(),
                data: claim.data(),
            });
        }
        roots.retain_recursive(data_handle);

        if !require_resident(reader, claim.metadata())? {
            return Err(CollectionRetentionError::MissingCommitMetadata {
                commit: claim.id(),
                metadata: claim.metadata(),
            });
        }
        roots.retain_recursive(claim.metadata());
    }

    Ok(roots)
}

fn require_resident<R, S>(
    reader: &R,
    handle: Inline<Handle<S>>,
) -> Result<bool, CollectionRetentionError<<R as BlobStoreMeta>::MetaError>>
where
    R: BlobStoreMeta + ?Sized,
    S: BlobEncoding + 'static,
    Handle<S>: InlineEncoding,
{
    reader
        .metadata(handle)
        .map(|entry| entry.is_some())
        .map_err(|source| CollectionRetentionError::Metadata {
            handle: handle.transmute(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::longstring::LongString;
    use crate::blob::{Blob, IntoBlob, MemoryBlobStore};
    use crate::collection::simplearchive_union::{self, SimpleArchiveUnionValidationError};
    use crate::collection::{
        discover_collection_records, resolve_collection_semantics, CollectionClaimValidation,
        CollectionCommit, CollectionDerive, CollectionMerge, CollectionValidationRequest,
    };
    use crate::inline::encodings::hash::{Blake3, Hash};
    use crate::macros::entity;
    use crate::metadata;
    use crate::repo::{BlobStore, BlobStoreGet};
    use crate::trible::{Trible, TribleSet, TRIBLE_LEN};

    fn id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn row(entity: u8, attribute: u8, value: u8) -> [u8; TRIBLE_LEN] {
        let mut row = [value; TRIBLE_LEN];
        row[..16].fill(entity);
        row[16..32].fill(attribute);
        row
    }

    fn archive(rows: impl IntoIterator<Item = [u8; TRIBLE_LEN]>) -> Blob<SimpleArchive> {
        let mut facts = TribleSet::new();
        for row in rows {
            facts.insert(&Trible::force_raw(row).unwrap());
        }
        facts.to_blob()
    }

    fn data(blob: &Blob<SimpleArchive>) -> CollectionData {
        Inline::<Hash<Blake3>>::new(Blake3::digest(&blob.bytes))
    }

    fn load_archive<R: BlobStoreGet>(
        reader: &R,
        data: CollectionData,
    ) -> Option<Blob<SimpleArchive>> {
        reader.get(Handle::<SimpleArchive>::from_hash(data)).ok()
    }

    fn validate_union<R: BlobStoreGet>(
        reader: &R,
        durable: &BTreeSet<Id>,
        request: CollectionValidationRequest<'_>,
    ) -> Result<CollectionClaimValidation<SimpleArchiveUnionValidationError>, Infallible> {
        if durable.contains(&request.claim_id()) {
            return Ok(CollectionClaimValidation::Accepted);
        }

        let verdict = match request {
            CollectionValidationRequest::Commit { definition, claim } => {
                let Some(blob) = load_archive(reader, claim.data()) else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                match simplearchive_union::validate_commit(definition, claim, &blob) {
                    Ok(()) => CollectionClaimValidation::Accepted,
                    Err(error) => CollectionClaimValidation::Rejected(error),
                }
            }
            CollectionValidationRequest::Merge { definition, claim } => {
                let (low, high) = claim.inputs();
                let (Some(low), Some(high), Some(result)) = (
                    load_archive(reader, low),
                    load_archive(reader, high),
                    load_archive(reader, claim.result()),
                ) else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                match simplearchive_union::validate_merge(definition, claim, &low, &high, &result) {
                    Ok(()) => CollectionClaimValidation::Accepted,
                    Err(error) => CollectionClaimValidation::Rejected(error),
                }
            }
            CollectionValidationRequest::Derive { .. } => CollectionClaimValidation::Pending,
        };
        Ok(verdict)
    }

    fn validate_union_and_derives<R: BlobStoreGet>(
        reader: &R,
        request: CollectionValidationRequest<'_>,
    ) -> Result<CollectionClaimValidation<SimpleArchiveUnionValidationError>, Infallible> {
        match request {
            CollectionValidationRequest::Derive { .. } => Ok(CollectionClaimValidation::Accepted),
            other => validate_union(reader, &BTreeSet::new(), other),
        }
    }

    fn insert_record_fixture(
        store: &mut MemoryBlobStore,
        definition: &super::super::CollectionDefinition,
        data: Blob<SimpleArchive>,
        metadata: Blob<SimpleArchive>,
        key: &SigningKey,
    ) -> CollectionCommit {
        let commit = CollectionCommit::sign(
            key,
            definition.id(),
            self::data(&data),
            metadata.get_handle(),
        );
        store.insert(super::super::CollectionDefinition::to_blob(definition));
        store.insert(data);
        store.insert(metadata);
        store.insert(CollectionCommit::to_blob(&commit));
        commit
    }

    #[test]
    fn authorized_commits_are_exact_strong_roots_and_keep_attachments() {
        let definition = simplearchive_union::definition(id(1));
        let key = SigningKey::from_bytes(&[7; 32]);
        let content_text: Blob<LongString> = "retained content".to_owned().to_blob();
        let content_text_handle = content_text.get_handle();
        let metadata_text: Blob<LongString> = "retained metadata".to_owned().to_blob();
        let metadata_text_handle = metadata_text.get_handle();
        let orphan: Blob<LongString> = "orphan".to_owned().to_blob();
        let orphan_handle = orphan.get_handle();

        let content = entity! { metadata::name: content_text_handle }
            .into_facts()
            .to_blob();
        let metadata = entity! { metadata::description: metadata_text_handle }
            .into_facts()
            .to_blob();

        let mut store = MemoryBlobStore::new();
        store.insert(content_text);
        store.insert(metadata_text);
        store.insert(orphan);
        let content_handle = content.get_handle();
        let metadata_handle = metadata.get_handle();
        let definition_handle =
            super::super::CollectionDefinition::to_blob(&definition).get_handle();
        let commit = insert_record_fixture(&mut store, &definition, content, metadata, &key);
        let commit_handle = CollectionCommit::to_blob(&commit).get_handle();

        let reader = store.reader().unwrap();
        let records = discover_collection_records(&reader).unwrap();
        let authorized = BTreeSet::from([commit.id()]);
        let resolution = resolve_collection_semantics(&records, &authorized, |request| {
            validate_union(&reader, &BTreeSet::new(), request)
        })
        .unwrap();
        let roots = plan_collection_retention(&records, &resolution, &reader).unwrap();
        let direct: BTreeSet<_> = roots.direct().collect();
        assert_eq!(
            direct,
            BTreeSet::from([definition_handle.transmute(), commit_handle.transmute()])
        );
        let recursive: BTreeSet<_> = roots.recursive().collect();
        assert_eq!(
            recursive,
            BTreeSet::from([content_handle.transmute(), metadata_handle.transmute()])
        );
        let keep = roots.expanded(&reader);

        store.keep(keep);
        let reader = store.reader().unwrap();
        assert!(reader
            .get::<Blob<SimpleArchive>, _>(definition_handle)
            .is_ok());
        assert!(reader.get::<Blob<SimpleArchive>, _>(commit_handle).is_ok());
        assert!(reader
            .get::<Blob<SimpleArchive>, _>(Handle::from_hash(commit.data()))
            .is_ok());
        assert!(reader
            .get::<Blob<SimpleArchive>, _>(commit.metadata())
            .is_ok());
        assert!(reader
            .get::<Blob<LongString>, _>(content_text_handle)
            .is_ok());
        assert!(reader
            .get::<Blob<LongString>, _>(metadata_text_handle)
            .is_ok());
        assert!(reader.get::<Blob<LongString>, _>(orphan_handle).is_err());
    }

    #[test]
    fn unsigned_equations_add_no_strong_roots_when_grounded_or_ungrounded() {
        let source = simplearchive_union::definition(id(1));
        let target = simplearchive_union::definition(id(2));
        let source_definition = super::super::CollectionDefinition::to_blob(&source);
        let target_definition = super::super::CollectionDefinition::to_blob(&target);
        let key = SigningKey::from_bytes(&[7; 32]);
        let left = archive([row(1, 1, 1)]);
        let right = archive([row(2, 1, 2)]);
        let empty_metadata = TribleSet::new().to_blob();

        let mut store = MemoryBlobStore::new();
        let first = insert_record_fixture(
            &mut store,
            &source,
            left.clone(),
            empty_metadata.clone(),
            &key,
        );
        let second = insert_record_fixture(
            &mut store,
            &source,
            right.clone(),
            empty_metadata.clone(),
            &key,
        );
        store.insert(target_definition.clone());

        let active_merge_result = simplearchive_union::join(&left, &right).unwrap();
        let active_merge = CollectionMerge::new(
            source.id(),
            data(&left),
            data(&right),
            data(&active_merge_result),
        );
        let active_derive_output = archive([row(3, 1, 3)]);
        let active_derive = CollectionDerive::new(
            source.id(),
            target.id(),
            data(&active_merge_result),
            data(&active_derive_output),
        );

        let orphan_left = archive([row(4, 1, 4)]);
        let orphan_right = archive([row(5, 1, 5)]);
        let orphan_merge_result = simplearchive_union::join(&orphan_left, &orphan_right).unwrap();
        let orphan_merge = CollectionMerge::new(
            source.id(),
            data(&orphan_left),
            data(&orphan_right),
            data(&orphan_merge_result),
        );
        let orphan_derive_output = archive([row(6, 1, 6)]);
        let orphan_derive = CollectionDerive::new(
            source.id(),
            target.id(),
            data(&orphan_merge_result),
            data(&orphan_derive_output),
        );

        for blob in [
            active_merge_result.clone(),
            active_derive_output.clone(),
            orphan_left.clone(),
            orphan_right.clone(),
            orphan_merge_result.clone(),
            orphan_derive_output.clone(),
        ] {
            store.insert(blob);
        }
        for record in [&active_merge, &orphan_merge] {
            store.insert(CollectionMerge::to_blob(record));
        }
        for record in [&active_derive, &orphan_derive] {
            store.insert(CollectionDerive::to_blob(record));
        }

        let reader = store.reader().unwrap();
        let records = discover_collection_records(&reader).unwrap();

        let unauthorized = resolve_collection_semantics(&records, &BTreeSet::new(), |request| {
            validate_union_and_derives(&reader, request)
        })
        .unwrap();
        assert!(unauthorized.admitted_claims().contains(&active_merge.id()));
        assert!(unauthorized.admitted_claims().contains(&active_derive.id()));
        let empty = plan_collection_retention(&records, &unauthorized, &reader).unwrap();
        assert_eq!(empty.direct().len(), 0);
        assert_eq!(empty.recursive().len(), 0);

        let authorized = BTreeSet::from([first.id(), second.id()]);
        let resolution = resolve_collection_semantics(&records, &authorized, |request| {
            validate_union_and_derives(&reader, request)
        })
        .unwrap();
        for active in [active_merge.id(), active_derive.id()] {
            assert!(resolution.admitted_claims().contains(&active));
            assert!(!resolution.activation_pending().contains(&active));
        }
        for orphan in [orphan_merge.id(), orphan_derive.id()] {
            assert!(resolution.admitted_claims().contains(&orphan));
            assert!(resolution.activation_pending().contains(&orphan));
        }

        let roots = plan_collection_retention(&records, &resolution, &reader).unwrap();
        let direct: BTreeSet<_> = roots.direct().collect();
        assert_eq!(
            direct,
            BTreeSet::from([
                source_definition.get_handle().transmute(),
                CollectionCommit::to_blob(&first).get_handle().transmute(),
                CollectionCommit::to_blob(&second).get_handle().transmute(),
            ])
        );
        let recursive: BTreeSet<_> = roots.recursive().collect();
        assert_eq!(
            recursive,
            BTreeSet::from([
                left.get_handle().transmute(),
                right.get_handle().transmute(),
                empty_metadata.get_handle().transmute(),
            ])
        );

        let keep = roots.expanded(&reader);
        assert!(!keep.contains(&target_definition.get_handle().transmute()));
        for record in [&active_merge, &orphan_merge] {
            assert!(!keep.contains(&CollectionMerge::to_blob(record).get_handle().transmute()));
        }
        for record in [&active_derive, &orphan_derive] {
            assert!(!keep.contains(&CollectionDerive::to_blob(record).get_handle().transmute()));
        }
        for cache_blob in [
            &active_merge_result,
            &active_derive_output,
            &orphan_left,
            &orphan_right,
            &orphan_merge_result,
            &orphan_derive_output,
        ] {
            assert!(!keep.contains(&cache_blob.get_handle().transmute()));
        }
    }

    #[test]
    fn missing_required_commit_ground_truth_is_rejected() {
        let definition = simplearchive_union::definition(id(1));
        let key = SigningKey::from_bytes(&[7; 32]);
        let content = archive([row(1, 1, 1)]);
        let metadata = archive([row(2, 1, 2)]);
        let definition_blob = super::super::CollectionDefinition::to_blob(&definition);
        let commit =
            CollectionCommit::sign(&key, definition.id(), data(&content), metadata.get_handle());
        let commit_blob = CollectionCommit::to_blob(&commit);

        let mut complete = MemoryBlobStore::new();
        for blob in [
            definition_blob.clone(),
            content.clone(),
            metadata.clone(),
            commit_blob.clone(),
        ] {
            complete.insert(blob);
        }
        let complete_reader = complete.reader().unwrap();
        let records = discover_collection_records(&complete_reader).unwrap();
        let resolution =
            resolve_collection_semantics(&records, &BTreeSet::from([commit.id()]), |request| {
                validate_union(&complete_reader, &BTreeSet::new(), request)
            })
            .unwrap();
        plan_collection_retention(&records, &resolution, &complete_reader).unwrap();

        let mut missing_definition = MemoryBlobStore::new();
        for blob in [content.clone(), metadata.clone(), commit_blob.clone()] {
            missing_definition.insert(blob);
        }
        let reader = missing_definition.reader().unwrap();
        assert!(matches!(
            plan_collection_retention(&records, &resolution, &reader),
            Err(CollectionRetentionError::MissingDefinition { collection })
                if collection == definition.id()
        ));

        let mut missing_data = MemoryBlobStore::new();
        for blob in [
            definition_blob.clone(),
            metadata.clone(),
            commit_blob.clone(),
        ] {
            missing_data.insert(blob);
        }
        let reader = missing_data.reader().unwrap();
        assert!(matches!(
            plan_collection_retention(&records, &resolution, &reader),
            Err(CollectionRetentionError::MissingCommitData { commit: id, data: missing })
                if id == commit.id() && missing == data(&content)
        ));

        let mut missing_metadata = MemoryBlobStore::new();
        for blob in [definition_blob, content, commit_blob] {
            missing_metadata.insert(blob);
        }
        let reader = missing_metadata.reader().unwrap();
        assert!(matches!(
            plan_collection_retention(&records, &resolution, &reader),
            Err(CollectionRetentionError::MissingCommitMetadata {
                commit: id,
                metadata: missing,
            }) if id == commit.id() && missing == metadata.get_handle()
        ));
    }
}
