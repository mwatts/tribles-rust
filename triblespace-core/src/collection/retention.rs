//! Strong retention roots for authorized collection commits.
//!
//! A signed, locally authorized [`super::CollectionCommit`] is durable ground
//! truth. Its native collection records are retained by the collection-store
//! rewrite policy; its data and metadata are recursive blob roots which own
//! their resident attachments.
//!
//! Unsigned `MERGE` and `DERIVE` records are reproducible cache work, not
//! authority. They therefore add no strong roots, whether their equations are
//! active, accepted-but-ungrounded, or merely present in the store. A separate
//! cache policy may retain selected equations and materializations without
//! weakening this ground-truth boundary.

use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::UnknownBlob;
use crate::blob::BlobEncoding;
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding};
use crate::repo::{BlobStoreMeta, RetentionRoots};

use super::{CollectionData, CollectionHandle, CollectionResolution, DiscoveredCollectionRecords};

/// A collection retention plan could not prove that every required blob stays
/// available.
#[derive(Debug)]
pub enum CollectionRetentionError<MetadataError> {
    /// An admitted commit's canonical collection descriptor blob is absent.
    MissingDescriptor {
        /// Canonical collection-descriptor handle.
        collection: CollectionHandle,
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
            Self::MissingDescriptor { collection } => {
                write!(
                    f,
                    "collection descriptor {} is not resident",
                    hex::encode_upper(collection.raw),
                )
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
/// Every admitted commit requires its referenced collection descriptor blob
/// to be resident and retains the descriptor, signed data, and metadata
/// recursively. Native commit records are not blobs and therefore do not
/// appear in the returned roots. Planning fails if any required blob is absent
/// instead of manufacturing a root for unavailable ground truth.
///
/// The returned roots are a pure result, not a persisted retained-scope
/// registry. A collector must rediscover, authorize, resolve, and plan again on
/// each later pass. Ordinary rewrites remain conservative and preserve every
/// native record plus the resident closure owned by every strictly verified
/// `COMMIT`; applying this narrower authorized plan is an explicit local
/// forgetting operation, never an implicit publication side effect.
pub fn plan_collection_retention<D, R>(
    records: &DiscoveredCollectionRecords,
    resolution: &CollectionResolution<D>,
    reader: &R,
) -> Result<RetentionRoots, CollectionRetentionError<<R as BlobStoreMeta>::MetaError>>
where
    R: BlobStoreMeta + ?Sized,
{
    let admitted = resolution.admitted_claims();
    let mut roots = RetentionRoots::new();

    for claim in records.commits() {
        if !admitted.contains(&claim.id()) {
            continue;
        }

        if !require_resident(reader, claim.collection())? {
            return Err(CollectionRetentionError::MissingDescriptor {
                collection: claim.collection(),
            });
        }
        roots.retain_recursive(claim.collection());

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

    fn lineage_from_derives(
        _records: &DiscoveredCollectionRecords,
    ) -> std::collections::BTreeMap<CollectionHandle, CollectionHandle> {
        // These retention tests build a single source -> target derivation
        // over scopes 1 and 2; a derive no longer names its source, so the
        // lineage is stated here the way a descriptor would state it.
        std::collections::BTreeMap::from([(
            simplearchive_union::descriptor(id(2)).handle(),
            simplearchive_union::descriptor(id(1)).handle(),
        )])
    }
    use super::*;

    use std::collections::BTreeSet;
    use std::convert::Infallible;

    use ed25519_dalek::SigningKey;

    use crate::blob::encodings::longstring::LongString;
    use crate::blob::{Blob, IntoBlob, MemoryBlobStore};
    use crate::collection::simplearchive_union::{self, SimpleArchiveUnionValidationError};
    use crate::collection::{
        discover_collection_records, resolve_collection_semantics, CollectionClaimValidation,
        CollectionCommit, CollectionDerive, CollectionDescriptor, CollectionHandle, CollectionMerge,
        CollectionRecord, CollectionStore, CollectionValidationRequest,
    };
    use crate::inline::encodings::hash::{Blake3, Hash};
    use crate::macros::entity;
    use crate::metadata;
    use crate::repo::{memoryrepo::MemoryRepo, BlobStore, BlobStoreGet, BlobStoreKeep};
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

    fn load_descriptor<R: BlobStoreGet>(
        reader: &R,
        collection: CollectionHandle,
    ) -> Option<CollectionDescriptor> {
        let blob: Blob<SimpleArchive> = reader.get(collection).ok()?;
        let blob = Blob::<SimpleArchive>::new(blob.bytes.clone());
        (blob.get_handle() == collection)
            .then(|| CollectionDescriptor::decode(&blob).ok())
            .flatten()
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
            CollectionValidationRequest::Commit { claim } => {
                let Some(descriptor) = load_descriptor(reader, claim.collection()) else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                let Some(blob) = load_archive(reader, claim.data()) else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                match simplearchive_union::validate_commit(&descriptor, claim, &blob) {
                    Ok(()) => CollectionClaimValidation::Accepted,
                    Err(error) => CollectionClaimValidation::Rejected(error),
                }
            }
            CollectionValidationRequest::Merge { claim } => {
                let Some(descriptor) = load_descriptor(reader, claim.collection()) else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                let (low, high) = claim.inputs();
                let (Some(low), Some(high), Some(result)) = (
                    load_archive(reader, low),
                    load_archive(reader, high),
                    load_archive(reader, claim.result()),
                ) else {
                    return Ok(CollectionClaimValidation::Pending);
                };
                match simplearchive_union::validate_merge(&descriptor, claim, &low, &high, &result)
                {
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
        store: &mut MemoryRepo,
        descriptor: &CollectionDescriptor,
        data: Blob<SimpleArchive>,
        metadata: Blob<SimpleArchive>,
        key: &SigningKey,
    ) -> CollectionCommit {
        let commit = CollectionCommit::sign(
            key,
            descriptor.handle(),
            self::data(&data),
            metadata.get_handle(),
        );
        store.blobs.insert(descriptor.to_blob());
        store.blobs.insert(data);
        store.blobs.insert(metadata);
        CollectionStore::insert(store, CollectionRecord::Commit(commit)).unwrap();
        commit
    }

    #[test]
    fn authorized_commits_are_exact_strong_roots_and_keep_attachments() {
        let descriptor = simplearchive_union::descriptor(id(1));
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

        let mut store = MemoryRepo::default();
        store.blobs.insert(content_text);
        store.blobs.insert(metadata_text);
        store.blobs.insert(orphan);
        let content_handle = content.get_handle();
        let metadata_handle = metadata.get_handle();
        let commit = insert_record_fixture(&mut store, &descriptor, content, metadata, &key);

        let records = discover_collection_records(&mut store).unwrap();
        let reader = store.reader().unwrap();
        let authorized = BTreeSet::from([commit.id()]);
        let resolution = resolve_collection_semantics(&records, &lineage_from_derives(&records), &authorized, |request| {
            validate_union(&reader, &BTreeSet::new(), request)
        })
        .unwrap();
        let roots = plan_collection_retention(&records, &resolution, &reader).unwrap();
        assert_eq!(roots.direct().len(), 0);
        let recursive: BTreeSet<_> = roots.recursive().collect();
        assert_eq!(
            recursive,
            BTreeSet::from([
                descriptor.handle().transmute(),
                content_handle.transmute(),
                metadata_handle.transmute(),
            ])
        );
        let keep = roots.expanded(&reader);

        store.keep(keep);
        let retained_records = discover_collection_records(&mut store).unwrap();
        assert_eq!(retained_records.commits(), &[commit]);
        let reader = store.reader().unwrap();
        let retained_descriptor: Blob<SimpleArchive> = reader.get(descriptor.handle()).unwrap();
        assert_eq!(
            CollectionDescriptor::decode(&retained_descriptor).unwrap(),
            descriptor
        );
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
        let source = simplearchive_union::descriptor(id(1));
        let target = simplearchive_union::descriptor(id(2));
        let key = SigningKey::from_bytes(&[7; 32]);
        let left = archive([row(1, 1, 1)]);
        let right = archive([row(2, 1, 2)]);
        let empty_metadata = TribleSet::new().to_blob();

        let mut store = MemoryRepo::default();
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
        store.blobs.insert(CollectionDescriptor::to_blob(&target));

        let active_merge_result = simplearchive_union::join(&left, &right).unwrap();
        let active_merge = CollectionMerge::new(
            source.handle(),
            data(&left),
            data(&right),
            data(&active_merge_result),
        );
        let active_derive_output = archive([row(3, 1, 3)]);
        let active_derive = CollectionDerive::new(
            target.handle(),
            data(&active_merge_result),
            data(&active_derive_output),
        );

        let orphan_left = archive([row(4, 1, 4)]);
        let orphan_right = archive([row(5, 1, 5)]);
        let orphan_merge_result = simplearchive_union::join(&orphan_left, &orphan_right).unwrap();
        let orphan_merge = CollectionMerge::new(
            source.handle(),
            data(&orphan_left),
            data(&orphan_right),
            data(&orphan_merge_result),
        );
        let orphan_derive_output = archive([row(6, 1, 6)]);
        let orphan_derive = CollectionDerive::new(
            target.handle(),
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
            store.blobs.insert(blob);
        }
        for record in [&active_merge, &orphan_merge] {
            CollectionStore::insert(&mut store, CollectionRecord::Merge(*record)).unwrap();
        }
        for record in [&active_derive, &orphan_derive] {
            CollectionStore::insert(&mut store, CollectionRecord::Derive(*record)).unwrap();
        }

        let records = discover_collection_records(&mut store).unwrap();
        let reader = store.reader().unwrap();

        let unauthorized = resolve_collection_semantics(&records, &lineage_from_derives(&records), &BTreeSet::new(), |request| {
            validate_union_and_derives(&reader, request)
        })
        .unwrap();
        assert!(unauthorized.admitted_claims().contains(&active_merge.id()));
        assert!(unauthorized.admitted_claims().contains(&active_derive.id()));
        let empty = plan_collection_retention(&records, &unauthorized, &reader).unwrap();
        assert_eq!(empty.direct().len(), 0);
        assert_eq!(empty.recursive().len(), 0);

        let authorized = BTreeSet::from([first.id(), second.id()]);
        let resolution = resolve_collection_semantics(&records, &lineage_from_derives(&records), &authorized, |request| {
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
        assert_eq!(roots.direct().len(), 0);
        let recursive: BTreeSet<_> = roots.recursive().collect();
        assert_eq!(
            recursive,
            BTreeSet::from([
                source.handle().transmute(),
                left.get_handle().transmute(),
                right.get_handle().transmute(),
                empty_metadata.get_handle().transmute(),
            ])
        );

        let keep = roots.expanded(&reader);
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
        let descriptor = simplearchive_union::descriptor(id(1));
        let key = SigningKey::from_bytes(&[7; 32]);
        let content = archive([row(1, 1, 1)]);
        let metadata = archive([row(2, 1, 2)]);
        let commit = CollectionCommit::sign(
            &key,
            descriptor.handle(),
            data(&content),
            metadata.get_handle(),
        );

        let mut complete = MemoryRepo::default();
        CollectionStore::insert(&mut complete, CollectionRecord::Commit(commit)).unwrap();
        complete
            .blobs
            .insert(CollectionDescriptor::to_blob(&descriptor));
        complete.blobs.insert(content.clone());
        complete.blobs.insert(metadata.clone());
        let records = discover_collection_records(&mut complete).unwrap();
        let complete_reader = complete.reader().unwrap();
        let resolution =
            resolve_collection_semantics(&records, &lineage_from_derives(&records), &BTreeSet::from([commit.id()]), |request| {
                validate_union(&complete_reader, &BTreeSet::new(), request)
            })
            .unwrap();
        plan_collection_retention(&records, &resolution, &complete_reader).unwrap();

        let mut missing_descriptor = MemoryBlobStore::new();
        missing_descriptor.insert(content.clone());
        missing_descriptor.insert(metadata.clone());
        let reader = missing_descriptor.reader().unwrap();
        assert!(matches!(
            plan_collection_retention(&records, &resolution, &reader),
            Err(CollectionRetentionError::MissingDescriptor { collection })
                if collection == descriptor.handle()
        ));

        let mut missing_data = MemoryBlobStore::new();
        missing_data.insert(CollectionDescriptor::to_blob(&descriptor));
        missing_data.insert(metadata.clone());
        let reader = missing_data.reader().unwrap();
        assert!(matches!(
            plan_collection_retention(&records, &resolution, &reader),
            Err(CollectionRetentionError::MissingCommitData { commit: id, data: missing })
                if id == commit.id() && missing == data(&content)
        ));

        let mut missing_metadata = MemoryBlobStore::new();
        missing_metadata.insert(CollectionDescriptor::to_blob(&descriptor));
        missing_metadata.insert(content);
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
