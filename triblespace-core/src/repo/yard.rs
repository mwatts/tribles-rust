//! Generational collection of piles for lazy-retention blob storage and a
//! generation-independent native collection-record union.
//!
//! A [`Yard`](crate::repo::yard::Yard) keeps an ordered young-to-old sequence of [`Pile`](crate::repo::pile::Pile)
//! generations. Writes land in the youngest generation, reads search the union
//! of each generation's live PATCH set, and retention/compaction update those
//! PATCH sets without changing Pile's append-only storage contract. Call
//! [`Yard::reclaim`](crate::repo::yard::Yard::reclaim) after collection when the logically evicted blobs should
//! also be physically removed from disk.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anybytes::Bytes;

use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, BlobEncoding, IntoBlob, TryFromBlob};
use crate::collection::{CollectionRecord, CollectionRecordSelector, CollectionStore};
use crate::id::Id;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding, INLINE_LEN};
use crate::patch::{Entry, IdentitySchema, PATCH};

use super::pile::{
    CollectionInsertError, GetBlobError, InsertError, Pile, PileReader, PileWriteError, ReadError,
};
use super::{
    transfer, BlobChildren, BlobInfo, BlobStore, BlobStoreGet, BlobStoreList, BlobStorePut,
    RetentionRoots, StorageClose, TransferError, WantRequest, WantStore,
};

type HandleSet = PATCH<INLINE_LEN, IdentitySchema>;
type WantIndex = PATCH<INLINE_LEN, IdentitySchema, WantEntry>;

#[derive(Debug, Clone, Copy)]
struct WantEntry {
    last_used: u64,
}

#[derive(Debug, Default)]
struct WantState {
    /// Blob requests alone carry LRU cache semantics.
    wants: WantIndex,
    /// Operation requests are durable questions, never blob-retention roots
    /// and never subject to the resident-blob cache budget.
    operations: BTreeSet<WantRequest>,
    clock: u64,
}

impl WantState {
    fn want(&mut self, request: WantRequest) {
        match request {
            WantRequest::Blob { handle } => {
                self.clock = self.clock.wrapping_add(1).max(1);
                let entry = Entry::with_value(
                    &handle.raw,
                    WantEntry {
                        last_used: self.clock,
                    },
                );
                self.wants.replace(&entry);
            }
            WantRequest::Merge { .. } | WantRequest::Derive { .. } => {
                self.operations.insert(request);
            }
        }
    }

    fn unwant(&mut self, request: WantRequest) {
        match request {
            WantRequest::Blob { handle } => self.wants.remove(&handle.raw),
            WantRequest::Merge { .. } | WantRequest::Derive { .. } => {
                self.operations.remove(&request);
            }
        }
    }

    fn requests(&self) -> Vec<WantRequest> {
        let mut requests: Vec<_> = (&self.wants)
            .into_iter()
            .map(|raw| WantRequest::blob(Inline::<Handle<UnknownBlob>>::new(*raw)))
            .chain(self.operations.iter().copied())
            .collect();
        requests.sort();
        requests
    }

    fn trim_to_present_budget(&mut self, present: &HandleSet, budget: usize) -> HandleSet {
        let mut candidates = Vec::new();
        for raw in &self.wants {
            if present.get(raw).is_some() {
                let entry = *self
                    .wants
                    .get(raw)
                    .expect("key from PATCH iterator must resolve in the same PATCH");
                candidates.push((*raw, entry.last_used));
            }
        }

        candidates.sort_by_key(|(_, last_used)| Reverse(*last_used));

        let mut retained = WantIndex::new();
        let mut handles = HandleSet::new();
        for (raw, last_used) in candidates.into_iter().take(budget) {
            retained.replace(&Entry::with_value(&raw, WantEntry { last_used }));
            handles.insert(&Entry::new(&raw));
        }
        self.wants = retained;
        handles
    }
}

#[derive(Debug, Clone, Copy)]
pub struct YardConfig {
    /// Maximum number of wanted blobs retained in the young cache.
    pub want_budget: usize,
    /// Strong survivor budget for the youngest level.
    pub strong_level_budget: usize,
    /// Per-level strong budget multiplier.
    pub fanout: usize,
}

impl Default for YardConfig {
    fn default() -> Self {
        Self {
            want_budget: 1024,
            strong_level_budget: 1024,
            fanout: 10,
        }
    }
}

#[derive(Debug)]
struct Segment {
    path: PathBuf,
    pile: Option<Pile>,
    live: HandleSet,
}

impl Segment {
    fn pile_mut(&mut self) -> &mut Pile {
        self.pile
            .as_mut()
            .expect("yard segment pile already closed")
    }
}

/// A generation (tier): an ordered list of segments. The youngest segment is
/// the active write target; reads union across all segments. (Today every
/// generation holds exactly one segment; multi-segment tiers land next.)
#[derive(Debug)]
struct Generation {
    segments: Vec<Segment>,
}

impl Generation {
    fn one(segment: Segment) -> Self {
        Self {
            segments: vec![segment],
        }
    }

    /// The active write segment — the youngest in the tier.
    fn active_mut(&mut self) -> &mut Segment {
        self.segments
            .last_mut()
            .expect("yard generation has no segment")
    }

    /// Total live blobs across the tier's segments.
    fn live_len(&self) -> usize {
        self.segments.iter().map(|s| s.live.len() as usize).sum()
    }
}

/// Generational, LSM-style collection of piles.
#[derive(Debug)]
pub struct Yard {
    generations: Vec<Generation>,
    config: YardConfig,
    want_state: Arc<Mutex<WantState>>,
}

impl Yard {
    fn opaque_record_count(&mut self) -> Result<usize, ReadError> {
        let mut count = 0usize;
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                count = count
                    .checked_add(segment.pile_mut().opaque_record_count()?)
                    .expect("yard opaque-record count overflow");
            }
        }
        Ok(count)
    }

    /// Create a fresh yard, truncating/creating one pile file per generation.
    pub fn create<P>(
        paths: impl IntoIterator<Item = P>,
        config: YardConfig,
    ) -> Result<Self, YardOpenError>
    where
        P: AsRef<Path>,
    {
        let mut generations = Vec::new();
        for path in paths {
            let path = path.as_ref().to_path_buf();
            File::create(&path).map_err(YardOpenError::Io)?;
            let pile = Pile::open(&path).map_err(|err| YardOpenError::Pile {
                path: path.clone(),
                err,
            })?;
            generations.push(Generation::one(Segment {
                path,
                pile: Some(pile),
                live: HandleSet::new(),
            }));
        }
        if generations.is_empty() {
            return Err(YardOpenError::NoGenerations);
        }
        Ok(Self {
            generations,
            config,
            want_state: Arc::new(Mutex::new(WantState::default())),
        })
    }

    /// Open an existing yard and treat all blobs in each pile as live.
    ///
    /// Fails loud on corruption: a generation pile with an invalid tail
    /// surfaces as [`YardOpenError::Pile`] naming the file, and **nothing is
    /// truncated**. Repair is an explicit opt-in via [`Yard::amputate`]
    /// (mirroring [`Pile::refresh`] vs [`Pile::amputate`]).
    ///
    /// The wanted set is rebuilt from the durable markers in the single young
    /// operational log. Wants found in an older generation are rejected: each
    /// pile exposes only its locally collapsed LWW set, so cross-file ordering
    /// cannot be reconstructed soundly.
    pub fn open<P>(
        paths: impl IntoIterator<Item = P>,
        config: YardConfig,
    ) -> Result<Self, YardOpenError>
    where
        P: AsRef<Path>,
    {
        Self::open_impl(paths, config, false)
    }

    /// Open an existing yard, **amputating** each generation pile first:
    /// every generation file is **TRUNCATED at its first invalid record,
    /// destroying everything after it**, exactly like [`Pile::amputate`].
    /// This is the explicit opt-in counterpart to the fail-loud [`Yard::open`] — reach for it only after `open` reported
    /// corruption and losing the invalid tail is acceptable.
    pub fn amputate<P>(
        paths: impl IntoIterator<Item = P>,
        config: YardConfig,
    ) -> Result<Self, YardOpenError>
    where
        P: AsRef<Path>,
    {
        Self::open_impl(paths, config, true)
    }

    fn open_impl<P>(
        paths: impl IntoIterator<Item = P>,
        config: YardConfig,
        repair: bool,
    ) -> Result<Self, YardOpenError>
    where
        P: AsRef<Path>,
    {
        let mut generations = Vec::new();
        for path in paths {
            let path = path.as_ref().to_path_buf();
            let mut pile = Pile::open(&path).map_err(|err| YardOpenError::Pile {
                path: path.clone(),
                err,
            })?;
            let load = if repair {
                pile.amputate()
            } else {
                pile.refresh()
            };
            load.map_err(|err| YardOpenError::Pile {
                path: path.clone(),
                err,
            })?;
            let reader = pile.reader().map_err(|err| YardOpenError::Pile {
                path: path.clone(),
                err,
            })?;
            let live = collect_list(reader.blobs()).map_err(YardOpenError::List)?;
            generations.push(Generation::one(Segment {
                path,
                pile: Some(pile),
                live,
            }));
        }
        if generations.is_empty() {
            return Err(YardOpenError::NoGenerations);
        }
        // Wants are a single young-generation operational log. Combining the
        // already-collapsed sets of several generations would falsely revive
        // an old assertion after a young retraction, so reject that ambiguous
        // state instead of inventing cross-pile append order.
        let mut want_state = WantState::default();
        for (level, generation) in generations.iter_mut().enumerate() {
            // Generation::one is the only constructor today. If multi-segment
            // tiers land, wants must remain in one ordered active-segment log
            // rather than unioning several locally collapsed LWW sets.
            debug_assert_eq!(generation.segments.len(), 1);
            for segment in &mut generation.segments {
                let requests = segment
                    .pile_mut()
                    .wants()
                    .map_err(update_err_io)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(update_err_io)?;
                if level != 0 && !requests.is_empty() {
                    return Err(YardOpenError::WantsOutsideYoungGeneration { level });
                }
                for request in requests {
                    want_state.want(request);
                }
            }
        }
        Ok(Self {
            generations,
            config,
            want_state: Arc::new(Mutex::new(want_state)),
        })
    }

    /// Number of generations in young-to-old order.
    pub fn generation_count(&self) -> usize {
        self.generations.len()
    }

    /// Number of live blobs in a generation.
    pub fn generation_len(&self, level: usize) -> Option<usize> {
        self.generations.get(level).map(|g| g.live_len())
    }

    /// Returns whether a live handle is currently associated with `level`.
    pub fn contains_in_generation<S>(&self, level: usize, handle: Inline<Handle<S>>) -> bool
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let handle: Inline<Handle<UnknownBlob>> = handle.transmute();
        self.generations
            .get(level)
            .is_some_and(|g| g.segments.iter().any(|s| s.live.get(&handle.raw).is_some()))
    }

    /// Re-append the surviving want markers to the young generation's
    /// pile. A pile rewrite ([`reclaim_generation`]) transfers only live
    /// blobs, so it drops the want marker records along with the dead
    /// bytes; whenever the young pile is rewritten the current wanted set must
    /// be re-recorded (surviving wants re-recorded, evicted ones dropped —
    /// eviction already removed them from the in-memory set).
    fn rerecord_want_markers(&mut self) -> Result<(), std::io::Error> {
        let wants: Vec<WantRequest> = {
            let want_state = self.want_state.lock().expect("want mutex poisoned");
            want_state.requests()
        };
        let pile = self.generations[0].active_mut().pile_mut();
        for request in wants {
            pile.want(request).map_err(|err| match err {
                PileWriteError::IoError(io) => io,
            })?;
        }
        pile.flush().map_err(|err| match err {
            super::pile::FlushError::IoError(io) => io,
        })?;
        Ok(())
    }

    /// Recompute the keep set with explicit policy roots and logically collect
    /// cold wants and orphans.
    ///
    /// The supplied roots are strong for this pass. Direct roots retain only
    /// themselves; recursive roots retain their resident descendants. Callers
    /// must supply the same policy on every later collection pass for the
    /// corresponding data to remain live. Strictly verified native collection
    /// commits add their resident data and metadata as recursive roots;
    /// invalid commits authenticate nothing, dangling dependencies remain for
    /// later synchronization, and unsigned equations add no roots. Pass an
    /// empty [`RetentionRoots`] explicitly when verified commits are the only
    /// desired strong roots.
    pub fn collect(&mut self, retention: &RetentionRoots) -> Result<(), YardCollectError> {
        let opaque_records = self
            .opaque_record_count()
            .map_err(|err| YardCollectError::Reader(YardReaderError::Pile(err)))?;
        if opaque_records != 0 {
            return Err(YardCollectError::OpaqueRecords {
                count: opaque_records,
            });
        }
        let retention = self
            .retention_with_native_commits(retention)
            .map_err(YardCollectError::CollectionRecords)?;
        let reader = self.reader().map_err(YardCollectError::Reader)?;
        let strong_keep = self.strong_keep_set(&reader, &retention);
        let present = reader.live_set();
        let want_keep = self
            .want_state
            .lock()
            .expect("want mutex poisoned")
            .trim_to_present_budget(&present, self.config.want_budget);

        let mut keep = strong_keep;
        keep.union(want_keep);
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                segment.live = segment.live.intersect(&keep);
            }
        }
        Ok(())
    }

    /// Run one compaction pass with explicit policy roots.
    ///
    /// Strong survivors descend when a level exceeds its strong budget. The
    /// whole surviving tier moves together; wanted survivors remain evictable
    /// under the want budget after they descend. Pass an empty
    /// [`RetentionRoots`] explicitly when verified commits are the only
    /// desired strong roots.
    pub fn compact(&mut self, retention: &RetentionRoots) -> Result<(), YardCollectError> {
        self.collect(retention)?;
        let last = self.generations.len().saturating_sub(1);
        let mut dumped = Vec::new();

        {
            let retention = self
                .retention_with_native_commits(retention)
                .map_err(YardCollectError::CollectionRecords)?;
            let reader = self.reader().map_err(YardCollectError::Reader)?;
            let strong_keep = self.strong_keep_set(&reader, &retention);

            for level in 0..last {
                let strong_here = self.generations[level].segments[0]
                    .live
                    .intersect(&strong_keep);
                if strong_here.len() as usize <= self.strong_budget_for(level) {
                    continue;
                }

                // Overflow: dump the whole tier down — strong *and* wanted
                // survivors. `collect(retention)` above already dropped dead, so the
                // segment's `live` is exactly the survivors. Wanted content descends to
                // use space in lower tiers rather than being pinned to the
                // youngest generation; it stays evictable everywhere and is
                // dropped by the want budget under pressure.
                let movers = self.generations[level].segments[0].live.clone();
                let handles: Vec<_> = movers
                    .clone()
                    .into_iter()
                    .map(Inline::<Handle<UnknownBlob>>::new)
                    .collect();

                let mut copied = Vec::new();
                {
                    let target = self.generations[level + 1].active_mut().pile_mut();
                    for result in transfer(&reader, target, handles.clone()) {
                        let (source, _target) = result.map_err(YardCollectError::Transfer)?;
                        copied.push(source);
                    }
                }

                {
                    let target = self.generations[level + 1].active_mut();
                    for source in copied {
                        target.live.insert(&Entry::new(&source.raw));
                    }
                }

                for raw in movers {
                    self.generations[level].segments[0].live.remove(&raw);
                }

                // Make the moved blobs durable in the target before the source
                // pile is recycled below, so a crash can't drop content that
                // would briefly live in neither place.
                self.generations[level + 1]
                    .active_mut()
                    .pile_mut()
                    .flush()
                    .map_err(YardCollectError::Flush)?;
                dumped.push(level);
            }
        }

        // Fold reclamation into the merge: each dumped tier is now empty, so
        // recycle its segment in place (crash-safe write-empty + atomic rename)
        // rather than leaving dead bytes for a separate reclaim() pass.
        for level in dumped {
            self.reclaim_segment(level, 0)
                .map_err(YardCollectError::Reclaim)?;
            // The rewrite dropped the young pile's want markers along
            // with its dead bytes; re-record the surviving wanted set.
            if level == 0 {
                self.rerecord_want_markers()
                    .map_err(YardCollectError::WantMarkers)?;
            }
        }

        self.collect(retention)
    }

    /// Physically rewrite each generation's pile to contain only its live set.
    ///
    /// Collection and compaction are logical operations: they update each
    /// generation's live PATCH set, so evicted blobs stop being readable through
    /// Yard readers, but they do not mutate the underlying append-only pile
    /// files. `reclaim` is the explicit physical step. For each generation it
    /// writes the current live handles and every native collection record to a
    /// sibling temporary pile, closes both piles, atomically renames the
    /// temporary file over the original on the same filesystem, and reopens
    /// the generation.
    pub fn reclaim(&mut self) -> Result<(), YardReclaimError> {
        let opaque_records = self.opaque_record_count().map_err(YardReclaimError::Pile)?;
        if opaque_records != 0 {
            return Err(YardReclaimError::OpaqueRecords {
                count: opaque_records,
            });
        }
        for level in 0..self.generations.len() {
            for index in 0..self.generations[level].segments.len() {
                self.reclaim_segment(level, index)?;
            }
            // The rewrite dropped the young pile's want markers along
            // with its dead bytes; re-record the surviving wanted set so the
            // wants stay durable (evicted ones are simply not re-recorded).
            if level == 0 {
                self.rerecord_want_markers()
                    .map_err(YardReclaimError::WantMarkers)?;
            }
        }
        Ok(())
    }

    /// Rewrite the segment at `(level, index)` down to its live set via
    /// [`reclaim_generation`]. If the rewrite fails, reopen the generation
    /// file as-is (fail-loud: [`Pile::refresh`], no repair, no truncation)
    /// so the yard stays usable and the rewrite error propagates. If even
    /// the reopen fails — for example the file is corrupt — both errors
    /// propagate together via [`YardReclaimError::Reopen`] and the segment
    /// is left closed.
    fn reclaim_segment(&mut self, level: usize, index: usize) -> Result<(), YardReclaimError> {
        let segment = &mut self.generations[level].segments[index];
        let path = segment.path.clone();
        let temp_path = reclaim_temp_path(&path, level);
        let live = segment.live.clone();
        let pile = segment
            .pile
            .take()
            .expect("yard segment pile already closed");

        match reclaim_generation(&path, &temp_path, &live, pile) {
            Ok(pile) => {
                self.generations[level].segments[index].pile = Some(pile);
                Ok(())
            }
            Err(primary) => {
                let reopen = Pile::open(&path).and_then(|mut pile| {
                    pile.refresh()?;
                    Ok(pile)
                });
                match reopen {
                    Ok(pile) => {
                        self.generations[level].segments[index].pile = Some(pile);
                        Err(primary)
                    }
                    Err(err) => Err(YardReclaimError::Reopen {
                        path,
                        primary: Box::new(primary),
                        err,
                    }),
                }
            }
        }
    }

    fn strong_budget_for(&self, level: usize) -> usize {
        let multiplier = self.config.fanout.max(1).saturating_pow(level as u32);
        self.config.strong_level_budget.saturating_mul(multiplier)
    }

    /// Add the resident ownership edges carried by strictly verified native
    /// commits. Invalid signatures authenticate no fields. A valid commit is
    /// preserved as a native record even when one of its dependencies is not
    /// live locally, but only its live descriptor, data, and metadata become
    /// recursive roots; unsigned merge/derive equations are evidence only and
    /// never own their inputs.
    fn retention_with_native_commits(
        &mut self,
        retention: &RetentionRoots,
    ) -> Result<RetentionRoots, YardCollectionRecordsError> {
        let mut combined = retention.clone();
        let records = self.records()?.collect::<Result<Vec<_>, _>>()?;
        for record in records {
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            if commit.verify_strict().is_err() {
                continue;
            }

            let descriptor = Inline::<Handle<UnknownBlob>>::new(commit.collection().raw);
            let data = Inline::<Handle<UnknownBlob>>::new(commit.data().raw);
            let metadata = commit.metadata().transmute();
            for handle in [descriptor, data, metadata] {
                let live = self.generations.iter().any(|generation| {
                    generation
                        .segments
                        .iter()
                        .any(|segment| segment.live.get(&handle.raw).is_some())
                });
                if live {
                    combined.retain_recursive(handle);
                }
            }
        }
        Ok(combined)
    }

    fn strong_keep_set(&self, reader: &YardReader, retention: &RetentionRoots) -> HandleSet {
        let mut keep = HandleSet::new();
        // Explicit policy roots remain strong even if the same handle or an
        // owned descendant also has a stale want marker.
        for handle in retention.expanded(reader) {
            keep.insert(&Entry::new(&handle.raw));
        }
        keep
    }

    #[cfg(test)]
    fn put_in_generation<S, T>(
        &mut self,
        level: usize,
        item: T,
    ) -> Result<Inline<Handle<S>>, InsertError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let handle = self.generations[level]
            .active_mut()
            .pile_mut()
            .put::<S, T>(item)?;
        let unknown: Inline<Handle<UnknownBlob>> = handle.transmute();
        self.generations[level]
            .active_mut()
            .live
            .insert(&Entry::new(&unknown.raw));
        Ok(handle)
    }
}

/// Deterministic owned snapshot of the native collection records visible
/// across all yard generations.
pub struct YardCollectionRecordIter {
    inner: std::collections::btree_map::IntoValues<Id, CollectionRecord>,
}

impl Iterator for YardCollectionRecordIter {
    type Item = Result<CollectionRecord, YardCollectionRecordsError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(Ok)
    }
}

/// Failure while replaying the native collection-record union of a yard.
#[derive(Debug)]
pub enum YardCollectionRecordsError {
    /// One generation could not refresh or decode its pile.
    Pile(ReadError),
    /// Two generations presented different canonical records under one
    /// intrinsic id.
    IdCollision { id: Id },
}

impl fmt::Display for YardCollectionRecordsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pile(error) => write!(f, "failed to replay yard collection records: {error}"),
            Self::IdCollision { id } => {
                write!(f, "collection record id {id:X} names different fields")
            }
        }
    }
}

impl Error for YardCollectionRecordsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pile(error) => Some(error),
            Self::IdCollision { .. } => None,
        }
    }
}

impl CollectionStore for Yard {
    type RecordsError = YardCollectionRecordsError;
    type InsertError = CollectionInsertError;
    type RecordIter<'a> = YardCollectionRecordIter;

    fn records<'a>(&'a mut self) -> Result<Self::RecordIter<'a>, Self::RecordsError> {
        let mut records = BTreeMap::new();
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                let replay = segment
                    .pile_mut()
                    .records()
                    .map_err(YardCollectionRecordsError::Pile)?;
                for result in replay {
                    let record = result.map_err(YardCollectionRecordsError::Pile)?;
                    let id = record.id();
                    match records.get(&id) {
                        Some(existing) if existing != &record => {
                            return Err(YardCollectionRecordsError::IdCollision { id });
                        }
                        Some(_) => {}
                        None => {
                            records.insert(id, record);
                        }
                    }
                }
            }
        }
        Ok(YardCollectionRecordIter {
            inner: records.into_values(),
        })
    }

    fn select_records(
        &mut self,
        selectors: &BTreeSet<CollectionRecordSelector>,
    ) -> Result<Vec<CollectionRecord>, Self::RecordsError> {
        if selectors.is_empty() {
            return Ok(Vec::new());
        }
        let mut records = BTreeMap::new();
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                let selected = segment
                    .pile_mut()
                    .select_records(selectors)
                    .map_err(YardCollectionRecordsError::Pile)?;
                for record in selected {
                    let id = record.id();
                    match records.get(&id) {
                        Some(existing) if existing != &record => {
                            return Err(YardCollectionRecordsError::IdCollision { id });
                        }
                        Some(_) => {}
                        None => {
                            records.insert(id, record);
                        }
                    }
                }
            }
        }
        Ok(records.into_values().collect())
    }

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.generations[0].active_mut().pile_mut().insert(record)
    }
}

impl WantStore for Yard {
    type WantError = PileWriteError;

    type WantIter<'a> = std::vec::IntoIter<Result<WantRequest, PileWriteError>>;

    /// Assert a want for a blob: refresh its LRU recency in memory AND persist a
    /// want marker to the young generation's pile, so the want survives
    /// a restart ([`Yard::open`] reloads it).
    ///
    /// Automatic lazy reads call this only on a miss. Explicit callers may
    /// also want an already-resident blob; that assertion is persisted and
    /// makes the resident copy subject to the yard's bounded want policy.
    fn want(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.generations[0].active_mut().pile_mut().want(request)?;
        self.want_state
            .lock()
            .expect("want mutex poisoned")
            .want(request);
        Ok(())
    }

    /// Retract a want: remove it from the in-memory want state and
    /// persist a want-retraction marker to the young generation's pile
    /// (last-writer-wins against any earlier want marker).
    fn unwant(&mut self, request: WantRequest) -> Result<(), Self::WantError> {
        self.generations[0]
            .active_mut()
            .pile_mut()
            .unwant(request)?;
        self.want_state
            .lock()
            .expect("want mutex poisoned")
            .unwant(request);
        Ok(())
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        let items: Vec<Result<WantRequest, PileWriteError>> = {
            let want_state = self.want_state.lock().expect("want mutex poisoned");
            want_state.requests().into_iter().map(Ok).collect()
        };
        Ok(items.into_iter())
    }
}

impl Drop for Yard {
    fn drop(&mut self) {
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                if let Some(pile) = segment.pile.take() {
                    let _ = pile.close();
                }
            }
        }
    }
}

impl BlobStorePut for Yard {
    type PutError = InsertError;

    fn put<S, T>(&mut self, item: T) -> Result<Inline<Handle<S>>, Self::PutError>
    where
        S: BlobEncoding + 'static,
        T: IntoBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let handle = self.generations[0]
            .active_mut()
            .pile_mut()
            .put::<S, T>(item)?;
        let unknown: Inline<Handle<UnknownBlob>> = handle.transmute();
        self.generations[0]
            .active_mut()
            .live
            .insert(&Entry::new(&unknown.raw));
        Ok(handle)
    }
}

impl BlobStore for Yard {
    type Reader = YardReader;
    type ReaderError = YardReaderError;

    fn reader(&mut self) -> Result<Self::Reader, Self::ReaderError> {
        let mut generations = Vec::new();
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                generations.push(YardGenerationReader {
                    reader: segment.pile_mut().reader().map_err(YardReaderError::Pile)?,
                    live: segment.live.clone(),
                });
            }
        }
        Ok(YardReader {
            generations,
            want_state: self.want_state.clone(),
        })
    }
}

impl super::StorageFlush for Yard {
    type Error = super::pile::FlushError;

    /// Flush every open generation pile. Want markers and fresh
    /// writes land in the young generation, but older generations can
    /// hold unsynced rewrites from `reclaim`/`compact`, so sync them all.
    fn flush(&mut self) -> Result<(), Self::Error> {
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                if let Some(pile) = segment.pile.as_mut() {
                    pile.flush()?;
                }
            }
        }
        Ok(())
    }
}

impl StorageClose for Yard {
    type Error = YardCloseError;

    fn close(mut self) -> Result<(), Self::Error> {
        for generation in &mut self.generations {
            for segment in &mut generation.segments {
                if let Some(pile) = segment.pile.take() {
                    pile.close().map_err(YardCloseError::Pile)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct YardGenerationReader {
    reader: PileReader,
    live: HandleSet,
}

impl PartialEq for YardGenerationReader {
    fn eq(&self, other: &Self) -> bool {
        self.reader == other.reader && self.live == other.live
    }
}

impl Eq for YardGenerationReader {}

/// Read-only Yard snapshot.
#[derive(Debug, Clone)]
pub struct YardReader {
    generations: Vec<YardGenerationReader>,
    want_state: Arc<Mutex<WantState>>,
}

impl YardReader {
    fn live_set(&self) -> HandleSet {
        let mut live = HandleSet::new();
        for generation in &self.generations {
            live.union(generation.live.clone());
        }
        live
    }

    /// Union read across generations (young -> old) that does NOT mint a
    /// demand-born want on a miss; returns `None` on a clean miss.
    /// Speculative / structural reads (reference discovery via
    /// `children`) use this so they never pollute the wanted set with
    /// wants for non-existent hashes. The public `get` layers the
    /// demand-born want on top of it.
    fn get_local<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Option<Result<T, YardGetError<<T as TryFromBlob<S>>::Error>>>
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        let unknown: Inline<Handle<UnknownBlob>> = handle.transmute();
        for generation in &self.generations {
            if generation.live.get(&unknown.raw).is_none() {
                continue;
            }
            match generation.reader.get::<T, S>(handle) {
                Ok(value) => return Some(Ok(value)),
                Err(GetBlobError::BlobNotFound) => continue,
                Err(err) => return Some(Err(YardGetError::Pile(err))),
            }
        }
        None
    }
}

impl PartialEq for YardReader {
    fn eq(&self, other: &Self) -> bool {
        self.generations == other.generations
    }
}

impl Eq for YardReader {}

impl BlobStoreGet for YardReader {
    type GetError<E: Error + Send + Sync + 'static> = YardGetError<E>;

    fn get<T, S>(
        &self,
        handle: Inline<Handle<S>>,
    ) -> Result<T, Self::GetError<<T as TryFromBlob<S>>::Error>>
    where
        S: BlobEncoding + 'static,
        T: TryFromBlob<S>,
        Handle<S>: InlineEncoding,
    {
        match self.get_local::<T, S>(handle) {
            Some(result) => result,
            None => {
                // An *intentional* read that missed is a demand-born
                // "want" — mint the want so the sync daemon can fetch
                // it. Speculative scans use `get_local` and never land here.
                self.want_state
                    .lock()
                    .expect("want mutex poisoned")
                    .want(WantRequest::blob(handle));
                Err(YardGetError::NotFound)
            }
        }
    }
}

impl BlobChildren for YardReader {
    fn children(&self, handle: Inline<Handle<UnknownBlob>>) -> Vec<Inline<Handle<UnknownBlob>>> {
        // Structural scan: use the non-minting read so reference
        // discovery never floods the wanted set with speculative wants. Wanted
        // cache policy is intentionally absent here: callers such as explicit
        // retention need the complete resident ownership graph.
        let Some(Ok(blob)) = self.get_local::<Blob<UnknownBlob>, UnknownBlob>(handle) else {
            return Vec::new();
        };
        let bytes = blob.bytes.as_ref();
        let mut result = Vec::new();
        let mut offset = 0usize;
        while offset + INLINE_LEN <= bytes.len() {
            let mut raw = [0u8; INLINE_LEN];
            raw.copy_from_slice(&bytes[offset..offset + INLINE_LEN]);

            let candidate = Inline::<Handle<UnknownBlob>>::new(raw);
            if matches!(self.get_local::<Bytes, UnknownBlob>(candidate), Some(Ok(_))) {
                result.push(candidate);
            }
            offset += INLINE_LEN;
        }
        result
    }
}

impl BlobStoreList for YardReader {
    type Iter<'a> = YardListIter;
    type Err = Infallible;

    fn blobs(&self) -> Self::Iter<'_> {
        YardListIter {
            inner: self.live_set().into_iter(),
            generations: self.generations.clone(),
        }
    }

    fn contains_blob<S>(&self, handle: Inline<Handle<S>>) -> Result<bool, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let handle: Inline<Handle<UnknownBlob>> = handle.transmute();
        Ok(self
            .generations
            .iter()
            .any(|generation| generation.live.get(&handle.raw).is_some()))
    }

    fn blob_info<S>(&self, handle: Inline<Handle<S>>) -> Result<Option<BlobInfo>, Self::Err>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let handle: Inline<Handle<UnknownBlob>> = handle.transmute();
        Ok(self.generations.iter().find_map(|generation| {
            generation
                .live
                .get(&handle.raw)
                .and_then(|_| generation.reader.unvalidated_blob_info(handle))
        }))
    }
}

pub struct YardListIter {
    inner: crate::patch::PATCHIntoIterator<INLINE_LEN, IdentitySchema, ()>,
    generations: Vec<YardGenerationReader>,
}

impl Iterator for YardListIter {
    type Item = Result<BlobInfo, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        let handle = Inline::<Handle<UnknownBlob>>::new(self.inner.next()?);
        let info = self
            .generations
            .iter()
            .find_map(|generation| generation.reader.unvalidated_blob_info(handle))
            .expect("live Yard handle must resolve in one generation snapshot");
        Some(Ok(info))
    }
}

fn update_err_io(err: PileWriteError) -> YardOpenError {
    match err {
        PileWriteError::IoError(io) => YardOpenError::Io(io),
    }
}

fn collect_list<E>(iter: impl IntoIterator<Item = Result<BlobInfo, E>>) -> Result<HandleSet, E> {
    let mut set = HandleSet::new();
    for result in iter {
        let info = result?;
        set.insert(&Entry::new(&info.handle.raw));
    }
    Ok(set)
}

fn reclaim_generation(
    path: &Path,
    temp_path: &Path,
    live: &HandleSet,
    mut old_pile: Pile,
) -> Result<Pile, YardReclaimError> {
    let opaque_records = old_pile
        .opaque_record_count()
        .map_err(YardReclaimError::Pile)?;
    if opaque_records != 0 {
        return Err(YardReclaimError::OpaqueRecords {
            count: opaque_records,
        });
    }

    match fs::remove_file(temp_path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(YardReclaimError::Io(err)),
    }

    let collection_records = old_pile
        .records()
        .map_err(YardReclaimError::Pile)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(YardReclaimError::Pile)?;
    let reader = old_pile.reader().map_err(YardReclaimError::Pile)?;
    File::create(temp_path).map_err(YardReclaimError::Io)?;
    let mut new_pile = Pile::open(temp_path).map_err(YardReclaimError::Pile)?;
    let handles: Vec<_> = live
        .clone()
        .into_iter()
        .map(Inline::<Handle<UnknownBlob>>::new)
        .collect();

    for result in transfer(&reader, &mut new_pile, handles) {
        result.map_err(YardReclaimError::Transfer)?;
    }
    old_pile
        .preserve_legacy_collection_headers_into(&mut new_pile)
        .map_err(YardReclaimError::CollectionRecord)?;
    for record in collection_records {
        new_pile
            .insert(record)
            .map_err(YardReclaimError::CollectionRecord)?;
    }
    new_pile.close().map_err(YardReclaimError::Close)?;
    drop(reader);
    old_pile.close().map_err(YardReclaimError::Close)?;
    fs::rename(temp_path, path).map_err(YardReclaimError::Io)?;

    let mut reopened = Pile::open(path).map_err(YardReclaimError::Pile)?;
    // The rewritten pile was just written and closed by us; fail loud on
    // any validation error rather than repair-truncating it.
    reopened.refresh().map_err(YardReclaimError::Pile)?;
    Ok(reopened)
}

fn reclaim_temp_path(path: &Path, level: usize) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "generation".into());
    path.with_file_name(format!(
        ".{file_name}.reclaim-{}-{level}.tmp",
        std::process::id()
    ))
}

#[derive(Debug)]
pub enum YardOpenError {
    NoGenerations,
    /// Durable wants must live only in the young operational generation;
    /// collapsed per-generation sets cannot reconstruct cross-file LWW order.
    WantsOutsideYoungGeneration {
        level: usize,
    },
    Io(std::io::Error),
    /// A generation pile failed to open or validate. A
    /// [`ReadError::CorruptPile`] here means the named generation file has
    /// an invalid tail; nothing was truncated — repair explicitly with
    /// [`Yard::amputate`] if losing the tail is acceptable.
    Pile {
        /// The generation pile file that failed.
        path: PathBuf,
        /// The underlying pile error.
        err: ReadError,
    },
    List(GetBlobError<Infallible>),
}

impl fmt::Display for YardOpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGenerations => write!(f, "yard requires at least one generation"),
            Self::WantsOutsideYoungGeneration { level } => write!(
                f,
                "yard generation {level} contains wants; wants must live only in generation 0"
            ),
            Self::Io(err) => write!(f, "failed to create yard pile file: {err}"),
            Self::Pile { path, err } => {
                write!(
                    f,
                    "failed to open yard generation pile {}: {err}",
                    path.display()
                )
            }
            Self::List(err) => write!(f, "failed to list yard pile: {err}"),
        }
    }
}

impl Error for YardOpenError {}

#[derive(Debug)]
pub enum YardReaderError {
    Pile(ReadError),
}

impl fmt::Display for YardReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pile(err) => write!(f, "failed to read yard generation: {err}"),
        }
    }
}

impl Error for YardReaderError {}

#[derive(Debug)]
pub enum YardGetError<E: Error> {
    NotFound,
    Pile(GetBlobError<E>),
}

impl<E: Error> fmt::Display for YardGetError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "blob not found in yard"),
            Self::Pile(err) => write!(f, "yard generation read failed: {err}"),
        }
    }
}

impl<E: Error + 'static> Error for YardGetError<E> {}

#[derive(Debug)]
#[non_exhaustive]
pub enum YardCollectError {
    Reader(YardReaderError),
    /// At least one generation contains opaque records. Collection cannot know
    /// whether they own otherwise-unrooted blobs, so it refuses before
    /// changing any generation's live set.
    OpaqueRecords {
        /// Total opaque records found across all generations.
        count: usize,
    },
    CollectionRecords(YardCollectionRecordsError),
    Transfer(TransferError<Infallible, YardGetError<Infallible>, InsertError>),
    Flush(super::pile::FlushError),
    Reclaim(YardReclaimError),
    WantMarkers(std::io::Error),
}

impl fmt::Display for YardCollectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reader(err) => write!(f, "failed to create yard reader: {err}"),
            Self::OpaqueRecords { count } => write!(
                f,
                "refusing to collect a yard containing {count} opaque record(s)"
            ),
            Self::CollectionRecords(err) => {
                write!(f, "failed to replay yard collection records: {err}")
            }
            Self::Transfer(err) => write!(f, "failed to compact yard generation: {err}"),
            Self::Flush(err) => write!(f, "failed to flush yard generation pile: {err}"),
            Self::Reclaim(err) => {
                write!(f, "failed to recycle compacted yard generation: {err}")
            }
            Self::WantMarkers(err) => {
                write!(f, "failed to re-record want markers: {err}")
            }
        }
    }
}

impl Error for YardCollectError {}

#[derive(Debug)]
pub enum YardCloseError {
    Pile(super::pile::FlushError),
}

impl fmt::Display for YardCloseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pile(err) => write!(f, "failed to close yard pile: {err}"),
        }
    }
}

impl Error for YardCloseError {}

#[derive(Debug)]
#[non_exhaustive]
pub enum YardReclaimError {
    Io(std::io::Error),
    Pile(ReadError),
    /// One or more generations contain opaque records. Reclaim cannot infer
    /// their retention semantics and refuses before replacing any file.
    OpaqueRecords {
        /// Number of opaque records found by the refusing scan.
        count: usize,
    },
    Transfer(TransferError<Infallible, GetBlobError<Infallible>, InsertError>),
    CollectionRecord(CollectionInsertError),
    Close(super::pile::FlushError),
    WantMarkers(std::io::Error),
    /// A generation rewrite failed (`primary`) and the subsequent
    /// fail-loud reopen of the generation file also failed (`err`). The
    /// segment is left closed; nothing was truncated.
    Reopen {
        /// The generation pile file that could not be reopened.
        path: PathBuf,
        /// The rewrite error that triggered the reopen.
        primary: Box<YardReclaimError>,
        /// The reopen/validation error.
        err: ReadError,
    },
}

impl fmt::Display for YardReclaimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to replace yard generation pile: {err}"),
            Self::Pile(err) => write!(f, "failed to read yard generation pile: {err}"),
            Self::OpaqueRecords { count } => write!(
                f,
                "refusing to reclaim a yard containing {count} opaque record(s)"
            ),
            Self::Transfer(err) => write!(f, "failed to copy live yard blobs: {err}"),
            Self::CollectionRecord(err) => {
                write!(f, "failed to copy a yard collection record: {err}")
            }
            Self::Close(err) => write!(f, "failed to close yard generation pile: {err}"),
            Self::WantMarkers(err) => {
                write!(f, "failed to re-record want markers: {err}")
            }
            Self::Reopen { path, primary, err } => write!(
                f,
                "failed to reopen yard generation pile {} after failed rewrite ({primary}): {err}",
                path.display()
            ),
        }
    }
}

impl Error for YardReclaimError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::encodings::rawbytes::RawBytes;
    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::collection::descriptor::{identity_for_tests, named_for_tests};
    use crate::collection::{
        empty_metadata_handle, CollectionCommit, CollectionDerive, CollectionMerge,
    };
    use crate::trible::TribleSet;
    use ed25519_dalek::SigningKey;
    use std::collections::BTreeSet;

    fn yard_with_paths(
        generations: usize,
        config: YardConfig,
    ) -> (tempfile::TempDir, Vec<PathBuf>, Yard) {
        let dir = tempfile::tempdir().unwrap();
        let paths = (0..generations)
            .map(|i| dir.path().join(format!("gen-{i}.pile")))
            .collect::<Vec<_>>();
        let yard = Yard::create(paths.clone(), config).unwrap();
        (dir, paths, yard)
    }

    fn yard_with(generations: usize, config: YardConfig) -> (tempfile::TempDir, Yard) {
        let (dir, _paths, yard) = yard_with_paths(generations, config);
        (dir, yard)
    }

    fn raw_blob(bytes: &'static [u8]) -> Bytes {
        Bytes::from_source(bytes.to_vec())
    }

    fn pin_id(byte: u8) -> Id {
        Id::new([byte; 16]).unwrap()
    }

    fn merge_record(tag: u8) -> CollectionRecord {
        let descriptor = named_for_tests(
            &format!("tagged-{tag}"),
            pin_id(tag.wrapping_add(1)),
            pin_id(tag.wrapping_add(2)),
        );
        CollectionRecord::Merge(CollectionMerge::new(
            identity_for_tests(&descriptor),
            Inline::new([tag.wrapping_add(3); 32]),
            Inline::new([tag.wrapping_add(4); 32]),
            Inline::new([tag.wrapping_add(5); 32]),
        ))
    }

    fn invalidate_collection_commit(commit: CollectionCommit) -> CollectionCommit {
        let (signature_r, signature_s) = commit.signature();
        let mut forged_r = signature_r.raw;
        forged_r[0] ^= 1;
        let forged = CollectionCommit::from_parts(
            commit.collection(),
            commit.data(),
            commit.metadata(),
            commit.public_key(),
            Inline::new(forged_r),
            signature_s,
        );
        assert!(forged.verify_strict().is_err());
        forged
    }

    fn get_raw(
        reader: &YardReader,
        handle: Inline<Handle<RawBytes>>,
    ) -> Result<Bytes, YardGetError<Infallible>> {
        reader.get::<Bytes, RawBytes>(handle)
    }

    fn pile_blob_count(path: &Path) -> usize {
        let mut pile = Pile::open(path).unwrap();
        pile.refresh().unwrap();
        let reader = pile.reader().unwrap();
        let count = reader.blobs().collect::<Result<Vec<_>, _>>().unwrap().len();
        drop(reader);
        pile.close().unwrap();
        count
    }

    #[test]
    fn generation_union_read_finds_older_generation() {
        let (_dir, mut yard) = yard_with(2, YardConfig::default());
        let old = yard
            .put_in_generation::<RawBytes, _>(1, raw_blob(b"old generation"))
            .unwrap();

        let reader = yard.reader().unwrap();

        assert_eq!(get_raw(&reader, old).unwrap(), raw_blob(b"old generation"));
        let info = reader
            .blobs()
            .find_map(|result| {
                let info = result.unwrap();
                (info.handle.raw == old.raw).then_some(info)
            })
            .expect("older generation is listed");
        assert_eq!(info.length, b"old generation".len() as u64);
    }

    #[test]
    fn collection_records_form_a_deterministic_generation_union_and_write_young() {
        let config = YardConfig::default();
        let (_dir, paths, mut yard) = yard_with_paths(2, config);
        let first = merge_record(21);
        let second = merge_record(27);
        let third = merge_record(33);

        yard.generations[1]
            .active_mut()
            .pile_mut()
            .insert(first)
            .unwrap();
        yard.generations[1]
            .active_mut()
            .pile_mut()
            .insert(second)
            .unwrap();
        yard.generations[0]
            .active_mut()
            .pile_mut()
            .insert(first)
            .unwrap();
        yard.insert(third).unwrap();

        let mut expected = vec![first, second, third];
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(
            yard.records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            expected
        );
        let young = yard.generations[0]
            .active_mut()
            .pile_mut()
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(young.contains(&third));

        yard.close().unwrap();
        let mut reopened = Yard::open(&paths, config).unwrap();
        assert_eq!(
            reopened
                .records()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            expected
        );
        reopened.close().unwrap();
    }

    #[test]
    fn collection_selection_unions_generations_without_choosing_an_output() {
        let config = YardConfig::default();
        let (_dir, paths, mut yard) = yard_with_paths(2, config);
        let target = Inline::new([42; 32]);
        let input = Inline::new([43; 32]);
        let first =
            CollectionRecord::Derive(CollectionDerive::new(target, input, Inline::new([44; 32])));
        let conflicting =
            CollectionRecord::Derive(CollectionDerive::new(target, input, Inline::new([45; 32])));
        let unrelated = CollectionRecord::Derive(CollectionDerive::new(
            Inline::new([46; 32]),
            input,
            Inline::new([47; 32]),
        ));
        yard.generations[1]
            .active_mut()
            .pile_mut()
            .insert(first)
            .unwrap();
        yard.generations[0]
            .active_mut()
            .pile_mut()
            .insert(first)
            .unwrap();
        yard.generations[0]
            .active_mut()
            .pile_mut()
            .insert(conflicting)
            .unwrap();
        yard.generations[0]
            .active_mut()
            .pile_mut()
            .insert(unrelated)
            .unwrap();
        let selectors = [CollectionRecordSelector::Operation(WantRequest::derive(
            target, input,
        ))]
        .into_iter()
        .collect();
        let mut expected = vec![first, conflicting];
        expected.sort_unstable_by_key(CollectionRecord::id);

        assert_eq!(yard.select_records(&selectors).unwrap(), expected);
        yard.close().unwrap();

        let mut reopened = Yard::open(&paths, config).unwrap();
        assert_eq!(reopened.select_records(&selectors).unwrap(), expected);
        assert!(!reopened
            .select_records(&selectors)
            .unwrap()
            .contains(&unrelated));
        reopened.close().unwrap();
    }

    #[test]
    fn native_commits_root_owned_blobs_and_reclaim_preserves_every_record_kind() {
        let (dir, mut yard) = yard_with(1, YardConfig::default());
        let attachment = yard
            .put::<RawBytes, _>(raw_blob(b"commit-owned attachment"))
            .unwrap();
        let data = yard
            .put::<RawBytes, _>(Bytes::from_source(attachment.raw.to_vec()))
            .unwrap();
        let metadata = yard
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        assert_eq!(metadata, empty_metadata_handle());
        let equation_only = yard
            .put::<RawBytes, _>(raw_blob(b"mentioned only by unsigned equations"))
            .unwrap();

        let descriptor = named_for_tests("retained", pin_id(32), pin_id(33));
        let collection = yard
            .put::<SimpleArchive, _>(crate::blob::IntoBlob::<SimpleArchive>::to_blob(
                descriptor.into_facts(),
            ))
            .unwrap();
        let key = SigningKey::from_bytes(&[34; 32]);
        let commit = CollectionCommit::sign(&key, collection, Inline::new(data.raw), metadata);
        commit.verify_strict().unwrap();
        let records = vec![
            CollectionRecord::Commit(commit),
            CollectionRecord::Merge(CollectionMerge::new(
                collection,
                Inline::new(equation_only.raw),
                Inline::new([35; 32]),
                Inline::new([36; 32]),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                identity_for_tests(&named_for_tests("derived", pin_id(38), pin_id(39))),
                Inline::new([36; 32]),
                Inline::new(equation_only.raw),
            )),
        ];
        for record in records.iter().copied() {
            yard.insert(record).unwrap();
        }

        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();
        assert!(reader.get::<Bytes, RawBytes>(attachment).is_ok());
        assert!(reader.get::<Bytes, RawBytes>(data).is_ok());
        assert!(reader
            .get::<Blob<SimpleArchive>, SimpleArchive>(metadata)
            .is_ok());
        assert!(reader
            .get::<Blob<SimpleArchive>, SimpleArchive>(collection)
            .is_ok());
        assert!(reader.get::<Bytes, RawBytes>(equation_only).is_err());
        drop(reader);

        yard.reclaim().unwrap();
        assert_eq!(pile_blob_count(&dir.path().join("gen-0.pile")), 4);
        let actual = yard
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut expected = records;
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(actual, expected);
    }

    #[test]
    fn invalid_native_commit_cannot_keep_resident_yard_blobs_live() {
        let (dir, mut yard) = yard_with(1, YardConfig::default());
        let forged_data = yard
            .put::<RawBytes, _>(raw_blob(b"invalid commit data"))
            .unwrap();
        let forged_metadata = yard
            .put::<SimpleArchive, _>(TribleSet::new().to_blob())
            .unwrap();
        let descriptor = named_for_tests("forged", pin_id(39), pin_id(40));
        let collection = yard
            .put::<SimpleArchive, _>(crate::blob::IntoBlob::<SimpleArchive>::to_blob(
                descriptor.into_facts(),
            ))
            .unwrap();
        let invalid = invalidate_collection_commit(CollectionCommit::sign(
            &SigningKey::from_bytes(&[41; 32]),
            collection,
            Inline::new(forged_data.raw),
            forged_metadata,
        ));
        let records = vec![CollectionRecord::Commit(invalid)];
        for record in records.iter().copied() {
            yard.insert(record).unwrap();
        }

        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();
        assert!(!reader.contains_blob(forged_data).unwrap());
        assert!(!reader.contains_blob(forged_metadata).unwrap());
        drop(reader);

        yard.reclaim().unwrap();
        assert_eq!(pile_blob_count(&dir.path().join("gen-0.pile")), 0);
        let mut actual = yard
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        actual.sort_by_key(CollectionRecord::id);
        let mut expected = records;
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(actual, expected);
    }

    #[test]
    fn valid_dangling_native_commit_survives_yard_collection_and_reclaim() {
        let (dir, mut yard) = yard_with(1, YardConfig::default());
        let descriptor = named_for_tests("dangling", pin_id(43), pin_id(44));
        let collection = yard
            .put::<SimpleArchive, _>(crate::blob::IntoBlob::<SimpleArchive>::to_blob(
                descriptor.into_facts(),
            ))
            .unwrap();
        let missing_data = Inline::new([45; 32]);
        let missing_metadata = Inline::<Handle<SimpleArchive>>::new([46; 32]);
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[47; 32]),
            collection,
            missing_data,
            missing_metadata,
        );
        commit.verify_strict().unwrap();
        let records = vec![CollectionRecord::Commit(commit)];
        for record in records.iter().copied() {
            yard.insert(record).unwrap();
        }

        yard.collect(&RetentionRoots::new()).unwrap();
        yard.reclaim().unwrap();
        assert_eq!(pile_blob_count(&dir.path().join("gen-0.pile")), 1);
        let mut actual = yard
            .records()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        actual.sort_by_key(CollectionRecord::id);
        let mut expected = records;
        expected.sort_by_key(CollectionRecord::id);
        assert_eq!(actual, expected);
    }

    #[test]
    fn explicit_keep_and_want_evict_gc() {
        let (_dir, mut yard) = yard_with(
            1,
            YardConfig {
                want_budget: 0,
                ..YardConfig::default()
            },
        );
        let strong = yard.put::<RawBytes, _>(raw_blob(b"strong")).unwrap();
        // demand-born wanted: wanted while absent, then fetched, then LRU-
        // evicted under a zero budget — a genuine cache eviction, not an
        // orphan sweep.
        let wanted = Blob::<RawBytes>::new(raw_blob(b"wanted")).get_handle();
        yard.want(WantRequest::blob(wanted)).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"wanted")).unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_recursive(strong);
        yard.collect(&roots).unwrap();
        let reader = yard.reader().unwrap();

        assert_eq!(get_raw(&reader, strong).unwrap(), raw_blob(b"strong"));
        assert!(matches!(
            get_raw(&reader, wanted),
            Err(YardGetError::NotFound)
        ));
    }

    #[test]
    fn explicit_retention_distinguishes_owned_and_descriptive_edges() {
        let (_dir, mut yard) = yard_with(
            1,
            YardConfig {
                want_budget: 0,
                ..YardConfig::default()
            },
        );
        let owned_child =
            Blob::<UnknownBlob>::new(Bytes::from_source(b"owned child".to_vec())).get_handle();
        // Even an evictable cache want cannot veto an explicit policy
        // root: collection retention has already decided this edge is owned.
        yard.want(WantRequest::blob(owned_child)).unwrap();
        yard.put::<UnknownBlob, _>(Bytes::from_source(b"owned child".to_vec()))
            .unwrap();
        let owned_parent = yard
            .put::<UnknownBlob, _>(Bytes::from_source(owned_child.raw.to_vec()))
            .unwrap();

        let described_input = yard
            .put::<UnknownBlob, _>(Bytes::from_source(b"described input".to_vec()))
            .unwrap();
        let ledger_record = yard
            .put::<UnknownBlob, _>(Bytes::from_source(described_input.raw.to_vec()))
            .unwrap();
        let orphan = yard
            .put::<UnknownBlob, _>(Bytes::from_source(b"orphan".to_vec()))
            .unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_recursive(owned_parent);
        roots.retain_direct(ledger_record);
        yard.collect(&roots).unwrap();
        let reader = yard.reader().unwrap();

        for retained in [owned_parent, owned_child, ledger_record] {
            assert!(reader.get::<Blob<UnknownBlob>, _>(retained).is_ok());
        }
        for collected in [described_input, orphan] {
            assert!(matches!(
                reader.get::<Blob<UnknownBlob>, _>(collected),
                Err(YardGetError::NotFound)
            ));
        }
    }

    #[test]
    fn hole_safe_walk_prunes_wanted_absent_child() {
        let (_dir, mut yard) = yard_with(1, YardConfig::default());
        let absent =
            Blob::<UnknownBlob>::new(Bytes::from_source(b"not stored".to_vec())).get_handle();
        let parent = yard
            .put::<UnknownBlob, _>(Bytes::from_source(absent.raw.to_vec()))
            .unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_recursive(parent);
        yard.want(WantRequest::blob(absent)).unwrap();

        yard.collect(&roots).unwrap();
        let reader = yard.reader().unwrap();

        assert!(reader.get::<Blob<UnknownBlob>, UnknownBlob>(parent).is_ok());
        assert!(matches!(
            reader.get::<Blob<UnknownBlob>, UnknownBlob>(absent),
            Err(YardGetError::NotFound)
        ));
    }

    #[test]
    fn compaction_tenures_retained_and_lets_wants_descend() {
        let (_dir, mut yard) = yard_with(
            3,
            YardConfig {
                want_budget: 10,
                strong_level_budget: 0,
                fanout: 1,
            },
        );
        let strong = yard.put::<RawBytes, _>(raw_blob(b"tenured")).unwrap();
        // `wanted` is demand-born: wanted while absent, then fetched, so it is
        // a genuine cache entry — not a resident downgrade, which no-ops.
        let wanted = Blob::<RawBytes>::new(raw_blob(b"cache")).get_handle();
        yard.want(WantRequest::blob(wanted)).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"cache")).unwrap();
        let mut roots = RetentionRoots::new();
        roots.retain_recursive(strong);

        yard.compact(&roots).unwrap();

        // With a zero strong budget everything overflows downward; wanted now
        // rides the flow to the bottom alongside strong (it is not pinned to
        // the youngest generation), and stays there because it is within the
        // want budget.
        assert!(!yard.contains_in_generation(0, strong));
        assert!(!yard.contains_in_generation(1, strong));
        assert!(yard.contains_in_generation(2, strong));
        assert!(!yard.contains_in_generation(0, wanted));
        assert!(!yard.contains_in_generation(1, wanted));
        assert!(yard.contains_in_generation(2, wanted));
    }

    #[test]
    fn compact_recycles_dumped_generations_without_a_separate_reclaim() {
        let (_dir, paths, mut yard) = yard_with_paths(
            2,
            YardConfig {
                want_budget: 0,
                strong_level_budget: 0,
                fanout: 1,
            },
        );
        // A strong blob lands in gen 0 and, with a zero budget, overflows on
        // compaction — the whole of gen 0 dumps into gen 1.
        let strong = yard
            .put::<RawBytes, _>(Bytes::from_source(vec![b'S'; 512]))
            .unwrap();
        let mut roots = RetentionRoots::new();
        roots.retain_recursive(strong);
        // Dead bytes physically present in gen 0, so there is genuinely
        // something for the merge to reclaim.
        let _dead = yard
            .put::<RawBytes, _>(Bytes::from_source(vec![b'D'; 4096]))
            .unwrap();
        assert_eq!(pile_blob_count(&paths[0]), 2);
        let strong_before = {
            let reader = yard.reader().unwrap();
            get_raw(&reader, strong).unwrap()
        };

        yard.compact(&roots).unwrap();

        // No separate reclaim(): the merge itself recycled gen 0's pile, so it
        // is physically empty, while the live blob moved down to gen 1 and
        // stays readable.
        assert_eq!(pile_blob_count(&paths[0]), 0);
        assert!(yard.contains_in_generation(1, strong));
        let reader = yard.reader().unwrap();
        assert_eq!(get_raw(&reader, strong).unwrap(), strong_before);
    }

    #[test]
    fn reclaim_rewrites_generation_to_live_blobs_only() {
        let (_dir, paths, mut yard) = yard_with_paths(
            1,
            YardConfig {
                want_budget: 0,
                ..YardConfig::default()
            },
        );

        let live = yard
            .put::<RawBytes, _>(Bytes::from_source(vec![b'L'; 512]))
            .unwrap();
        let evicted = yard
            .put::<RawBytes, _>(Bytes::from_source(vec![b'E'; 4096]))
            .unwrap();

        let mut roots = RetentionRoots::new();
        roots.retain_recursive(live);
        yard.collect(&roots).unwrap();
        let before_size = fs::metadata(&paths[0]).unwrap().len();
        let before_count = pile_blob_count(&paths[0]);
        let before_reader = yard.reader().unwrap();
        let live_before = get_raw(&before_reader, live).unwrap();

        assert!(matches!(
            get_raw(&before_reader, evicted),
            Err(YardGetError::NotFound)
        ));
        assert_eq!(before_count, 2);

        yard.reclaim().unwrap();

        let after_size = fs::metadata(&paths[0]).unwrap().len();
        let after_count = pile_blob_count(&paths[0]);
        let after_reader = yard.reader().unwrap();

        assert!(after_size < before_size);
        assert_eq!(after_count, 1);
        assert_eq!(get_raw(&after_reader, live).unwrap(), live_before);
        assert!(matches!(
            get_raw(&after_reader, evicted),
            Err(YardGetError::NotFound)
        ));

        let mut fresh_pile = Pile::open(&paths[0]).unwrap();
        fresh_pile.refresh().unwrap();
        let fresh_reader = fresh_pile.reader().unwrap();
        assert_eq!(
            fresh_reader.get::<Bytes, RawBytes>(live).unwrap(),
            live_before
        );
        assert!(matches!(
            fresh_reader.get::<Bytes, RawBytes>(evicted),
            Err(GetBlobError::BlobNotFound)
        ));
        drop(fresh_reader);
        fresh_pile.close().unwrap();

        yard.reclaim().unwrap();
        assert_eq!(fs::metadata(&paths[0]).unwrap().len(), after_size);
        assert_eq!(pile_blob_count(&paths[0]), after_count);
    }

    /// The amnesia regression: wants are durable pile records, so
    /// reopening a yard rebuilds the want state instead of resetting it.
    #[test]
    fn yard_open_reloads_wants() {
        let (_dir, paths, mut yard) = yard_with_paths(2, YardConfig::default());

        // A pure want: asserted while absent, never fetched.
        let want = Blob::<RawBytes>::new(raw_blob(b"still wanted after restart")).get_handle();
        yard.want(WantRequest::blob(want)).unwrap();
        // A demand-fetched cache entry: wanted while absent, then put.
        let cached = Blob::<RawBytes>::new(raw_blob(b"cached")).get_handle();
        yard.want(WantRequest::blob(cached)).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"cached")).unwrap();
        // A retracted want must stay retracted across restart (LWW).
        let retracted = Blob::<RawBytes>::new(raw_blob(b"changed my mind")).get_handle();
        yard.want(WantRequest::blob(retracted)).unwrap();
        yard.unwant(WantRequest::blob(retracted)).unwrap();

        drop(yard); // closes (and flushes) the generation piles

        let mut reopened = Yard::open(paths, YardConfig::default()).unwrap();
        let wanted: BTreeSet<_> = reopened
            .wants()
            .unwrap()
            .map(|result| match result.unwrap() {
                WantRequest::Blob { handle } => handle.raw,
                _ => panic!("test only inserted blob requests"),
            })
            .collect();
        assert!(
            wanted.contains(&want.raw),
            "wanted want lost across restart — the amnesia bug"
        );
        assert!(
            wanted.contains(&cached.raw),
            "wanted cache-retention marker lost across restart"
        );
        assert!(
            !wanted.contains(&retracted.raw),
            "want retraction did not stick across restart"
        );

        // The reloaded want still works as a retention marker: the
        // cached blob survives collection under the default budget.
        reopened.collect(&RetentionRoots::new()).unwrap();
        let reader = reopened.reader().unwrap();
        assert_eq!(get_raw(&reader, cached).unwrap(), raw_blob(b"cached"));
    }

    /// A young-pile rewrite (reclaim) must not drop the durable wanted set:
    /// surviving want markers are re-recorded into the rewritten pile.
    #[test]
    fn want_markers_survive_reclaim() {
        let (_dir, paths, mut yard) = yard_with_paths(1, YardConfig::default());

        let want = Blob::<RawBytes>::new(raw_blob(b"wanted, absent")).get_handle();
        yard.want(WantRequest::blob(want)).unwrap();
        let cached = Blob::<RawBytes>::new(raw_blob(b"cached blob")).get_handle();
        yard.want(WantRequest::blob(cached)).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"cached blob")).unwrap();

        // Rewrite the young pile: only live blobs are transferred, so the
        // marker records are dropped — and must be re-recorded.
        yard.reclaim().unwrap();

        drop(yard);
        let mut reopened = Yard::open(paths, YardConfig::default()).unwrap();
        let wanted: BTreeSet<_> = reopened
            .wants()
            .unwrap()
            .map(|result| match result.unwrap() {
                WantRequest::Blob { handle } => handle.raw,
                _ => panic!("test only inserted blob requests"),
            })
            .collect();
        assert!(
            wanted.contains(&want.raw),
            "want marker lost by reclaim rewrite"
        );
        assert!(
            wanted.contains(&cached.raw),
            "cache marker lost by reclaim rewrite"
        );
        let reader = reopened.reader().unwrap();
        assert_eq!(get_raw(&reader, cached).unwrap(), raw_blob(b"cached blob"));
    }

    #[test]
    fn operation_wants_survive_reclaim_without_retaining_their_digest_fields() {
        let config = YardConfig {
            want_budget: 0,
            ..YardConfig::default()
        };
        let (_dir, paths, mut yard) = yard_with_paths(1, config);
        let input_blob = yard
            .put::<RawBytes, _>(raw_blob(b"an operation input digest is not a blob root"))
            .unwrap();
        let source = Inline::new([51; INLINE_LEN]);
        let target = Inline::new([52; INLINE_LEN]);
        let input = Inline::new(input_blob.raw);
        let merge = WantRequest::merge(source, input, Inline::new([53; INLINE_LEN]));
        let derive = WantRequest::derive(target, input);
        yard.want(merge).unwrap();
        yard.want(derive).unwrap();

        yard.collect(&RetentionRoots::new()).unwrap();
        assert!(!yard.contains_in_generation(0, input_blob));
        yard.reclaim().unwrap();
        drop(yard);

        let mut reopened = Yard::open(paths, config).unwrap();
        assert_eq!(
            reopened
                .wants()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![merge, derive]
        );
        assert!(!reopened.contains_in_generation(0, input_blob));
    }

    #[test]
    fn yard_rejects_wants_stranded_in_an_old_generation() {
        let (_dir, paths, yard) = yard_with_paths(2, YardConfig::default());
        drop(yard);

        let request =
            WantRequest::derive(Inline::new([62; INLINE_LEN]), Inline::new([63; INLINE_LEN]));
        let mut old = Pile::open(&paths[1]).unwrap();
        old.want(request).unwrap();
        old.close().unwrap();

        assert!(matches!(
            Yard::open(paths, YardConfig::default()),
            Err(YardOpenError::WantsOutsideYoungGeneration { level: 1 })
        ));
    }

    /// The fail-loud posture: opening a yard whose generation pile has a
    /// corrupt tail must surface the corruption (naming the file) WITHOUT
    /// truncating anything; `Yard::amputate` is the explicit opt-in repair.
    #[test]
    fn open_fails_loud_on_corrupt_generation_without_truncating() {
        use std::io::Write;

        let (_dir, paths, mut yard) = yard_with_paths(1, YardConfig::default());
        let live = yard.put::<RawBytes, _>(raw_blob(b"survivor")).unwrap();
        drop(yard); // closes (and flushes) the generation pile

        // Tear the tail before a complete record marker lands.
        {
            let mut file = fs::OpenOptions::new().append(true).open(&paths[0]).unwrap();
            file.write_all(&[0xFF; 8]).unwrap();
            file.sync_all().unwrap();
        }
        let corrupt_len = fs::metadata(&paths[0]).unwrap().len();

        // Fail-loud open: the corruption propagates, names the file, and
        // the file is NOT truncated.
        match Yard::open(paths.clone(), YardConfig::default()) {
            Err(YardOpenError::Pile { path, err }) => {
                assert_eq!(path, paths[0]);
                assert!(
                    matches!(err, ReadError::CorruptPile { .. }),
                    "expected CorruptPile, got: {err}"
                );
            }
            other => panic!("expected fail-loud corrupt open, got {other:?}"),
        }
        assert_eq!(
            fs::metadata(&paths[0]).unwrap().len(),
            corrupt_len,
            "fail-loud open must not truncate the generation pile"
        );

        // Explicit repair: amputate truncates the invalid tail and the
        // valid prefix stays readable.
        let mut repaired = Yard::amputate(paths.clone(), YardConfig::default()).unwrap();
        assert!(fs::metadata(&paths[0]).unwrap().len() < corrupt_len);
        let reader = repaired.reader().unwrap();
        assert_eq!(get_raw(&reader, live).unwrap(), raw_blob(b"survivor"));
    }
}
