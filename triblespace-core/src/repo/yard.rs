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
use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anybytes::Bytes;

use crate::blob::encodings::UnknownBlob;
use crate::blob::{Blob, BlobEncoding, IntoBlob, TryFromBlob};
use crate::collection::{CollectionRecord, CollectionStore};
use crate::id::{Id, RawId};
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding, INLINE_LEN};
use crate::local_cell::LocalCellStore;
use crate::patch::{Entry, IdentitySchema, PATCH};

use crate::prelude::blobencodings::SimpleArchive;

use super::pile::{
    CollectionInsertError, GetBlobError, InsertError, Pile, PileReader, PileWriteError, ReadError,
};
use super::{
    transfer, BlobChildren, BlobInfo, BlobStore, BlobStoreGet, BlobStoreList, BlobStorePut,
    PinStore, PushResult, RetentionRoots, StorageClose, TransferError, WantStore,
};

type HandleSet = PATCH<INLINE_LEN, IdentitySchema>;
type StrongPins = PATCH<16, IdentitySchema, Inline<Handle<UnknownBlob>>>;
type LocalCells = PATCH<16, IdentitySchema, Inline<Handle<SimpleArchive>>>;
type WantIndex = PATCH<INLINE_LEN, IdentitySchema, WantEntry>;

#[derive(Debug, Clone, Copy)]
struct WantEntry {
    last_used: u64,
}

#[derive(Debug, Default)]
struct WantState {
    wants: WantIndex,
    clock: u64,
}

impl WantState {
    fn want(&mut self, handle: Inline<Handle<UnknownBlob>>) {
        self.clock = self.clock.wrapping_add(1).max(1);
        let entry = Entry::with_value(
            &handle.raw,
            WantEntry {
                last_used: self.clock,
            },
        );
        self.wants.replace(&entry);
    }

    fn unwant(&mut self, raw: &[u8; INLINE_LEN]) {
        self.wants.remove(raw);
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
    strong_pins: StrongPins,
    /// LWW local-cell view reconstructed old-to-young across every pile.
    cells: LocalCells,
    want_state: Arc<Mutex<WantState>>,
}

impl Yard {
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
            strong_pins: StrongPins::new(),
            cells: LocalCells::new(),
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
    /// The wanted set is rebuilt from the durable want markers found
    /// in the generation piles (old to young, so the young generation's
    /// markers override older ones), fixing the restart amnesia the previous
    /// in-memory-only want state had.
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
        // Reload the durable wants. Iterate old -> young so a young
        // marker (re-)pins last and wins the LRU recency slot; each pile's
        // own set is already LWW-resolved by its log order. (In practice
        // markers are only ever written to the young generation's pile.)
        let mut want_state = WantState::default();
        for generation in generations.iter_mut().rev() {
            for segment in &mut generation.segments {
                for marker in segment.pile_mut().wants().map_err(update_err_io)? {
                    want_state.want(marker.map_err(update_err_io)?);
                }
            }
        }
        let mut yard = Self {
            generations,
            config,
            strong_pins: StrongPins::new(),
            cells: LocalCells::new(),
            want_state: Arc::new(Mutex::new(want_state)),
        };
        yard.refresh_cells()
            .map_err(|error| YardOpenError::Io(error.into()))?;
        Ok(yard)
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

    /// Strongly pin a blob as the current head for `pin`.
    pub fn pin_strong<S>(&mut self, pin: Id, handle: Inline<Handle<S>>)
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let handle: Inline<Handle<UnknownBlob>> = handle.transmute();
        let raw: RawId = pin.into();
        self.strong_pins.replace(&Entry::with_value(&raw, handle));
    }

    /// Remove a strong pin.
    pub fn unpin_strong(&mut self, pin: Id) {
        let raw: RawId = pin.into();
        self.strong_pins.remove(&raw);
    }

    /// Rebuild the local-cell view from every generation. Generations are
    /// stored young-to-old, so replay proceeds in reverse; a younger value or
    /// tombstone then wins exactly as concatenated pile records would.
    fn refresh_cells(&mut self) -> Result<(), ReadError> {
        let mut cells = LocalCells::new();
        for generation in self.generations.iter_mut().rev() {
            for segment in &mut generation.segments {
                let (values, tombstones) = segment.pile_mut().local_cell_snapshot()?;
                for raw in &tombstones {
                    cells.remove(raw);
                }
                for raw in &values {
                    let value = *values
                        .get(raw)
                        .expect("cell key from pile snapshot must retain its value");
                    cells.replace(&Entry::with_value(raw, value));
                }
            }
        }
        self.cells = cells;
        Ok(())
    }

    /// Re-append the surviving want markers to the young generation's
    /// pile. A pile rewrite ([`reclaim_generation`]) transfers only live
    /// blobs, so it drops the want marker records along with the dead
    /// bytes; whenever the young pile is rewritten the current wanted set must
    /// be re-recorded (surviving wants re-recorded, evicted ones dropped —
    /// eviction already removed them from the in-memory set).
    fn rerecord_want_markers(&mut self) -> Result<(), std::io::Error> {
        let wants: Vec<Inline<Handle<UnknownBlob>>> = {
            let want_state = self.want_state.lock().expect("want mutex poisoned");
            (&want_state.wants)
                .into_iter()
                .map(|raw| Inline::<Handle<UnknownBlob>>::new(*raw))
                .collect()
        };
        let pile = self.generations[0].active_mut().pile_mut();
        for handle in wants {
            pile.want(handle).map_err(|err| match err {
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
    /// empty [`RetentionRoots`] explicitly when those commits and legacy strong
    /// pins are the only desired strong roots.
    pub fn collect(&mut self, retention: &RetentionRoots) -> Result<(), YardCollectError> {
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
    /// [`RetentionRoots`] explicitly when legacy strong pins are the only
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
    /// live locally, but only live data and metadata become recursive roots;
    /// unsigned merge/derive equations are evidence only and never own their
    /// inputs.
    fn retention_with_native_commits(
        &mut self,
        retention: &RetentionRoots,
    ) -> Result<RetentionRoots, YardCollectionRecordsError> {
        self.refresh_cells()
            .map_err(YardCollectionRecordsError::Pile)?;
        let mut combined = retention.clone();
        let records = self.records()?.collect::<Result<Vec<_>, _>>()?;
        for record in records {
            let CollectionRecord::Commit(commit) = record else {
                continue;
            };
            if commit.verify_strict().is_err() {
                continue;
            }

            let data = Inline::<Handle<UnknownBlob>>::new(commit.data().raw);
            let metadata = commit.metadata().transmute();
            for handle in [data, metadata] {
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
        for raw in &self.cells {
            let value = *self
                .cells
                .get(raw)
                .expect("cell key from yard snapshot must retain its value");
            combined.retain_recursive(value);
        }
        Ok(combined)
    }

    fn strong_keep_set(&self, reader: &YardReader, retention: &RetentionRoots) -> HandleSet {
        let wants = self
            .want_state
            .lock()
            .expect("want mutex poisoned")
            .wants
            .clone();
        let roots: Vec<_> = (&self.strong_pins)
            .into_iter()
            .filter_map(|pin| self.strong_pins.get(pin).copied())
            .collect();

        let mut keep = HandleSet::new();
        let mut queue = VecDeque::from(roots);
        while let Some(handle) = queue.pop_front() {
            // Wants veto legacy strong-pin ownership and prune that whole
            // subtree. This policy belongs to the collector, not to the
            // structural BlobChildren view used by explicit retention.
            if wants.get(&handle.raw).is_some() || keep.get(&handle.raw).is_some() {
                continue;
            }
            keep.insert(&Entry::new(&handle.raw));
            queue.extend(reader.children(handle));
        }
        // Explicit policy roots are not cache wants. They remain strong
        // even if the same handle or an owned descendant also has a stale want
        // marker.
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

    fn insert(&mut self, record: CollectionRecord) -> Result<(), Self::InsertError> {
        self.generations[0].active_mut().pile_mut().insert(record)
    }
}

impl LocalCellStore for Yard {
    type CellError = PileWriteError;

    fn cell(&mut self, id: Id) -> Result<Option<Inline<Handle<SimpleArchive>>>, Self::CellError> {
        self.refresh_cells().map_err(PileWriteError::from)?;
        Ok(self.cells.get(&id.into()).copied())
    }

    fn set_cell(
        &mut self,
        id: Id,
        value: Option<Inline<Handle<SimpleArchive>>>,
    ) -> Result<(), Self::CellError> {
        self.refresh_cells().map_err(PileWriteError::from)?;
        self.generations[0]
            .active_mut()
            .pile_mut()
            .set_cell(id, value)?;
        match value {
            Some(value) => self.cells.replace(&Entry::with_value(&id.into(), value)),
            None => self.cells.remove(&id.into()),
        }
        Ok(())
    }
}

impl PinStore for Yard {
    type PinsError = Infallible;
    type HeadError = Infallible;
    type UpdateError = Infallible;

    type ListIter<'a> = std::vec::IntoIter<Result<Id, Infallible>>;

    fn pins<'a>(&'a mut self) -> Result<Self::ListIter<'a>, Self::PinsError> {
        // Byte-ordered (PATCH tree order) for deterministic iteration,
        // mirroring Pile's PATCH-backed `pins`.
        let ids: Vec<Result<Id, Infallible>> = self
            .strong_pins
            .clone()
            .into_iter_ordered()
            .map(|raw| Ok(Id::new(raw).expect("nil pin id in yard strong pins")))
            .collect();
        Ok(ids.into_iter())
    }

    fn head(&mut self, id: Id) -> Result<Option<Inline<Handle<SimpleArchive>>>, Self::HeadError> {
        let raw: RawId = id.into();
        Ok(self.strong_pins.get(&raw).copied().map(Inline::transmute))
    }

    fn update(
        &mut self,
        id: Id,
        old: Option<Inline<Handle<SimpleArchive>>>,
        new: Option<Inline<Handle<SimpleArchive>>>,
    ) -> Result<PushResult, Self::UpdateError> {
        let raw: RawId = id.into();
        let current: Option<Inline<Handle<SimpleArchive>>> =
            self.strong_pins.get(&raw).copied().map(Inline::transmute);
        if current != old {
            return Ok(PushResult::Conflict(current));
        }
        match new {
            Some(new) => self.pin_strong(id, new),
            None => self.unpin_strong(id),
        }
        Ok(PushResult::Success())
    }
}

impl WantStore for Yard {
    type WantError = PileWriteError;

    type WantIter<'a> = std::vec::IntoIter<Result<Inline<Handle<UnknownBlob>>, PileWriteError>>;

    /// Assert a want for a blob: refresh its LRU recency in memory AND persist a
    /// want marker to the young generation's pile, so the want survives
    /// a restart ([`Yard::open`] reloads it).
    ///
    /// Automatic lazy reads call this only on a miss. Explicit callers may
    /// also want an already-resident blob; that assertion is persisted and
    /// makes the resident copy subject to the yard's bounded want policy.
    fn want<S>(&mut self, handle: Inline<Handle<S>>) -> Result<(), Self::WantError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let handle: Inline<Handle<UnknownBlob>> = handle.transmute();
        self.generations[0]
            .active_mut()
            .pile_mut()
            .want::<UnknownBlob>(handle)?;
        self.want_state
            .lock()
            .expect("want mutex poisoned")
            .want(handle);
        Ok(())
    }

    /// Retract a want: remove it from the in-memory want state and
    /// persist a want-retraction marker to the young generation's pile
    /// (last-writer-wins against any earlier want marker).
    fn unwant<S>(&mut self, handle: Inline<Handle<S>>) -> Result<(), Self::WantError>
    where
        S: BlobEncoding + 'static,
        Handle<S>: InlineEncoding,
    {
        let handle: Inline<Handle<UnknownBlob>> = handle.transmute();
        self.generations[0]
            .active_mut()
            .pile_mut()
            .unwant::<UnknownBlob>(handle)?;
        self.want_state
            .lock()
            .expect("want mutex poisoned")
            .unwant(&handle.raw);
        Ok(())
    }

    fn wants<'a>(&'a mut self) -> Result<Self::WantIter<'a>, Self::WantError> {
        let items: Vec<Result<Inline<Handle<UnknownBlob>>, PileWriteError>> = {
            let want_state = self.want_state.lock().expect("want mutex poisoned");
            (&want_state.wants)
                .into_iter()
                .map(|raw| Ok(Inline::<Handle<UnknownBlob>>::new(*raw)))
                .collect()
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
                    .want(handle.transmute());
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
            .find_map(|generation| generation.reader.blob_info(handle))
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
    let (local_cells, local_cell_tombstones) = old_pile
        .local_cell_snapshot()
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
    for record in collection_records {
        new_pile
            .insert(record)
            .map_err(YardReclaimError::CollectionRecord)?;
    }
    for raw in &local_cell_tombstones {
        let id = Id::new(*raw).expect("Pile never stores a nil local-cell id");
        new_pile
            .set_cell(id, None)
            .map_err(YardReclaimError::LocalCell)?;
    }
    for raw in &local_cells {
        let id = Id::new(*raw).expect("Pile never stores a nil local-cell id");
        let value = *local_cells
            .get(raw)
            .expect("cell key from pile snapshot must retain its value");
        new_pile
            .set_cell(id, Some(value))
            .map_err(YardReclaimError::LocalCell)?;
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
pub enum YardCollectError {
    Reader(YardReaderError),
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
pub enum YardReclaimError {
    Io(std::io::Error),
    Pile(ReadError),
    Transfer(TransferError<Infallible, GetBlobError<Infallible>, InsertError>),
    CollectionRecord(CollectionInsertError),
    LocalCell(PileWriteError),
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
            Self::Transfer(err) => write!(f, "failed to copy live yard blobs: {err}"),
            Self::CollectionRecord(err) => {
                write!(f, "failed to copy a yard collection record: {err}")
            }
            Self::LocalCell(err) => write!(f, "failed to copy a yard local cell: {err}"),
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
    use crate::collection::{
        empty_metadata_handle, CollectionCommit, CollectionDefinition, CollectionDerive,
        CollectionMerge,
    };
    use crate::trible::TribleSet;
    use ed25519_dalek::SigningKey;
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
        let first = CollectionRecord::Definition(CollectionDefinition::new(
            pin_id(21),
            pin_id(22),
            pin_id(23),
        ));
        let second = CollectionRecord::Definition(CollectionDefinition::new(
            pin_id(24),
            pin_id(25),
            pin_id(26),
        ));
        let third = CollectionRecord::Definition(CollectionDefinition::new(
            pin_id(27),
            pin_id(28),
            pin_id(29),
        ));

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

        let definition = CollectionDefinition::new(pin_id(31), pin_id(32), pin_id(33));
        let key = SigningKey::from_bytes(&[34; 32]);
        let commit = CollectionCommit::sign(&key, definition.id(), Inline::new(data.raw), metadata);
        commit.verify_strict().unwrap();
        let records = vec![
            CollectionRecord::Definition(definition),
            CollectionRecord::Commit(commit),
            CollectionRecord::Merge(CollectionMerge::new(
                definition.id(),
                Inline::new(equation_only.raw),
                Inline::new([35; 32]),
                Inline::new([36; 32]),
            )),
            CollectionRecord::Derive(CollectionDerive::new(
                definition.id(),
                pin_id(37),
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
        assert!(reader.get::<Bytes, RawBytes>(equation_only).is_err());
        drop(reader);

        yard.reclaim().unwrap();
        assert_eq!(pile_blob_count(&dir.path().join("gen-0.pile")), 3);
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
        let definition = CollectionDefinition::new(pin_id(38), pin_id(39), pin_id(40));
        let invalid = invalidate_collection_commit(CollectionCommit::sign(
            &SigningKey::from_bytes(&[41; 32]),
            definition.id(),
            Inline::new(forged_data.raw),
            forged_metadata,
        ));
        let records = vec![
            CollectionRecord::Definition(definition),
            CollectionRecord::Commit(invalid),
        ];
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
        let definition = CollectionDefinition::new(pin_id(42), pin_id(43), pin_id(44));
        let missing_data = Inline::new([45; 32]);
        let missing_metadata = Inline::<Handle<SimpleArchive>>::new([46; 32]);
        let commit = CollectionCommit::sign(
            &SigningKey::from_bytes(&[47; 32]),
            definition.id(),
            missing_data,
            missing_metadata,
        );
        commit.verify_strict().unwrap();
        let records = vec![
            CollectionRecord::Definition(definition),
            CollectionRecord::Commit(commit),
        ];
        for record in records.iter().copied() {
            yard.insert(record).unwrap();
        }

        yard.collect(&RetentionRoots::new()).unwrap();
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
    fn strong_keep_and_want_evict_gc() {
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
        yard.want(wanted).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"wanted")).unwrap();

        yard.pin_strong(pin_id(1), strong);
        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();

        assert_eq!(get_raw(&reader, strong).unwrap(), raw_blob(b"strong"));
        assert!(matches!(
            get_raw(&reader, wanted),
            Err(YardGetError::NotFound)
        ));
    }

    #[test]
    fn want_veto_overrides_strong_reachability() {
        let (_dir, mut yard) = yard_with(
            1,
            YardConfig {
                want_budget: 0,
                ..YardConfig::default()
            },
        );
        // `child` enters the cache the demand-born way: wanted while
        // absent (the want), then fetched. It is reachable from a strong
        // parent, yet the wanted veto still makes it evictable.
        let child = Blob::<UnknownBlob>::new(Bytes::from_source(b"child".to_vec())).get_handle();
        yard.want(child).unwrap();
        yard.put::<UnknownBlob, _>(Bytes::from_source(b"child".to_vec()))
            .unwrap();
        let parent = yard
            .put::<UnknownBlob, _>(Bytes::from_source(child.raw.to_vec()))
            .unwrap();

        yard.pin_strong(pin_id(2), parent);
        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();

        assert!(reader.get::<Blob<UnknownBlob>, UnknownBlob>(parent).is_ok());
        assert!(matches!(
            reader.get::<Blob<UnknownBlob>, UnknownBlob>(child),
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
        yard.want(owned_child).unwrap();
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

        yard.pin_strong(pin_id(3), parent);
        yard.want(absent).unwrap();

        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();

        assert!(reader.get::<Blob<UnknownBlob>, UnknownBlob>(parent).is_ok());
        assert!(matches!(
            reader.get::<Blob<UnknownBlob>, UnknownBlob>(absent),
            Err(YardGetError::NotFound)
        ));
    }

    #[test]
    fn compaction_tenures_strong_and_lets_wants_descend() {
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
        yard.want(wanted).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"cache")).unwrap();
        yard.pin_strong(pin_id(4), strong);

        yard.compact(&RetentionRoots::new()).unwrap();

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
        yard.pin_strong(pin_id(7), strong);
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

        yard.compact(&RetentionRoots::new()).unwrap();

        // No separate reclaim(): the merge itself recycled gen 0's pile, so it
        // is physically empty, while the live blob moved down to gen 1 and
        // stays readable.
        assert_eq!(pile_blob_count(&paths[0]), 0);
        assert!(yard.contains_in_generation(1, strong));
        let reader = yard.reader().unwrap();
        assert_eq!(get_raw(&reader, strong).unwrap(), strong_before);
    }

    #[test]
    fn superseded_strong_head_becomes_droppable() {
        let (_dir, mut yard) = yard_with(1, YardConfig::default());
        let old = yard.put::<RawBytes, _>(raw_blob(b"old")).unwrap();
        let pin = pin_id(5);

        yard.pin_strong(pin, old);
        yard.collect(&RetentionRoots::new()).unwrap();
        assert_eq!(
            get_raw(&yard.reader().unwrap(), old).unwrap(),
            raw_blob(b"old")
        );

        let new = yard.put::<RawBytes, _>(raw_blob(b"new")).unwrap();
        yard.pin_strong(pin, new);
        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();

        assert!(matches!(get_raw(&reader, old), Err(YardGetError::NotFound)));
        assert_eq!(get_raw(&reader, new).unwrap(), raw_blob(b"new"));
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

        yard.pin_strong(pin_id(6), live);
        yard.collect(&RetentionRoots::new()).unwrap();
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
        yard.want(want).unwrap();
        // A demand-fetched cache entry: wanted while absent, then put.
        let cached = Blob::<RawBytes>::new(raw_blob(b"cached")).get_handle();
        yard.want(cached).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"cached")).unwrap();
        // A retracted want must stay retracted across restart (LWW).
        let retracted = Blob::<RawBytes>::new(raw_blob(b"changed my mind")).get_handle();
        yard.want(retracted).unwrap();
        yard.unwant(retracted).unwrap();

        drop(yard); // closes (and flushes) the generation piles

        let mut reopened = Yard::open(paths, YardConfig::default()).unwrap();
        let wanted: BTreeSet<_> = reopened.wants().unwrap().map(|r| r.unwrap().raw).collect();
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
    /// surviving pins are re-recorded into the rewritten pile.
    #[test]
    fn want_markers_survive_reclaim() {
        let (_dir, paths, mut yard) = yard_with_paths(1, YardConfig::default());

        let want = Blob::<RawBytes>::new(raw_blob(b"wanted, absent")).get_handle();
        yard.want(want).unwrap();
        let cached = Blob::<RawBytes>::new(raw_blob(b"cached blob")).get_handle();
        yard.want(cached).unwrap();
        yard.put::<RawBytes, _>(raw_blob(b"cached blob")).unwrap();

        // Rewrite the young pile: only live blobs are transferred, so the
        // marker records are dropped — and must be re-recorded.
        yard.reclaim().unwrap();

        drop(yard);
        let mut reopened = Yard::open(paths, YardConfig::default()).unwrap();
        let wanted: BTreeSet<_> = reopened.wants().unwrap().map(|r| r.unwrap().raw).collect();
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
    fn local_cell_value_is_a_recursive_root_and_survives_reclaim() {
        let (_dir, paths, mut yard) = yard_with_paths(
            2,
            YardConfig {
                want_budget: 0,
                strong_level_budget: 0,
                fanout: 2,
            },
        );
        let cell_id = pin_id(87);
        let value = yard
            .put::<SimpleArchive, _>(TribleSet::new())
            .expect("put cell value");
        let orphan = yard
            .put::<RawBytes, _>(raw_blob(b"unowned"))
            .expect("put orphan");
        yard.set_cell(cell_id, Some(value)).unwrap();

        yard.collect(&RetentionRoots::new()).unwrap();
        let reader = yard.reader().unwrap();
        assert!(reader.get::<TribleSet, SimpleArchive>(value).is_ok());
        assert!(reader.get::<Bytes, RawBytes>(orphan).is_err());
        drop(reader);

        yard.reclaim().unwrap();
        yard.close().unwrap();

        let mut reopened = Yard::open(paths, YardConfig::default()).unwrap();
        assert_eq!(reopened.cell(cell_id).unwrap(), Some(value));
        assert!(reopened
            .reader()
            .unwrap()
            .get::<TribleSet, SimpleArchive>(value)
            .is_ok());
        assert_eq!(reopened.pins().unwrap().count(), 0);
        reopened.close().unwrap();
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

        // Corrupt the tail: append garbage that is not a valid record.
        {
            let mut file = fs::OpenOptions::new().append(true).open(&paths[0]).unwrap();
            file.write_all(&[0xFF; 64]).unwrap();
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

    /// Yard's PinStore impl: CAS semantics over the in-memory strong pins.
    #[test]
    fn yard_pinstore_cas_update() {
        let (_dir, mut yard) = yard_with(1, YardConfig::default());
        let h1 = yard.put::<RawBytes, _>(raw_blob(b"one")).unwrap();
        let h2 = yard.put::<RawBytes, _>(raw_blob(b"two")).unwrap();
        let pin = pin_id(9);

        assert!(matches!(
            yard.update(pin, None, Some(h1.transmute())).unwrap(),
            PushResult::Success()
        ));
        assert_eq!(yard.head(pin).unwrap(), Some(h1.transmute()));
        match yard
            .update(pin, Some(h2.transmute()), Some(h2.transmute()))
            .unwrap()
        {
            PushResult::Conflict(current) => assert_eq!(current, Some(h1.transmute())),
            other => panic!("expected conflict, got {other:?}"),
        }
        let ids: Vec<_> = yard.pins().unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(ids, vec![pin]);
        assert!(matches!(
            yard.update(pin, Some(h1.transmute()), None).unwrap(),
            PushResult::Success()
        ));
        assert_eq!(yard.head(pin).unwrap(), None);
    }

    mod dst {
        use super::*;

        const GENERATIONS: usize = 4;
        const SEEDS: u64 = 50;
        const STEPS: usize = 64;
        const PIN_COUNT: usize = 8;

        type RawHandle = [u8; INLINE_LEN];

        #[derive(Debug, Clone)]
        struct Model {
            handles: Vec<RawHandle>,
            bytes: BTreeMap<RawHandle, Vec<u8>>,
            absent: Vec<RawHandle>,
        }

        impl Model {
            fn new() -> Self {
                Self {
                    handles: Vec::new(),
                    bytes: BTreeMap::new(),
                    absent: Vec::new(),
                }
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct FinalState {
            live_by_generation: Vec<Vec<RawHandle>>,
            readable: Vec<RawHandle>,
        }

        #[derive(Clone, Copy, Debug)]
        enum WantMode {
            YoungOnly,
            AnyKnownHandle,
        }

        #[derive(Clone, Copy, Debug)]
        struct SplitMix64 {
            state: u64,
        }

        impl SplitMix64 {
            fn new(seed: u64) -> Self {
                Self { state: seed }
            }

            fn next_u64(&mut self) -> u64 {
                self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }

            fn index(&mut self, len: usize) -> usize {
                (self.next_u64() as usize) % len
            }

            fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
                self.next_u64() % denominator < numerator
            }

            fn fill(&mut self, bytes: &mut [u8]) {
                for chunk in bytes.chunks_mut(8) {
                    let random = self.next_u64().to_le_bytes();
                    chunk.copy_from_slice(&random[..chunk.len()]);
                }
            }
        }

        fn unknown(raw: RawHandle) -> Inline<Handle<UnknownBlob>> {
            Inline::<Handle<UnknownBlob>>::new(raw)
        }

        fn pin_id(index: usize) -> Id {
            Id::new([(index as u8).wrapping_add(1); 16]).unwrap()
        }

        fn live_sets(yard: &Yard) -> Vec<BTreeSet<RawHandle>> {
            yard.generations
                .iter()
                .map(|generation| {
                    generation
                        .segments
                        .iter()
                        .flat_map(|s| s.live.clone().into_iter())
                        .collect()
                })
                .collect()
        }

        fn live_union(yard: &Yard) -> BTreeSet<RawHandle> {
            live_sets(yard).into_iter().flatten().collect()
        }

        fn wants(yard: &Yard) -> BTreeMap<RawHandle, u64> {
            let want_state = yard.want_state.lock().expect("want mutex poisoned");
            want_state
                .wants
                .clone()
                .into_iter()
                .map(|raw| {
                    let entry = want_state
                        .wants
                        .get(&raw)
                        .expect("want key resolves")
                        .last_used;
                    (raw, entry)
                })
                .collect()
        }

        fn strong_roots(yard: &Yard) -> Vec<RawHandle> {
            (&yard.strong_pins)
                .into_iter()
                .filter_map(|pin| yard.strong_pins.get(pin).copied())
                .map(|handle| handle.raw)
                .collect()
        }

        fn budgeted_wants(
            wanted: &BTreeMap<RawHandle, u64>,
            present: &BTreeSet<RawHandle>,
            budget: usize,
        ) -> BTreeSet<RawHandle> {
            let mut candidates = wanted
                .iter()
                .filter(|(raw, _)| present.contains(*raw))
                .map(|(raw, last_used)| (*raw, *last_used))
                .collect::<Vec<_>>();
            candidates.sort_by_key(|(_, last_used)| Reverse(*last_used));
            candidates
                .into_iter()
                .take(budget)
                .map(|(raw, _)| raw)
                .collect()
        }

        fn child_chunks(bytes: &[u8]) -> impl Iterator<Item = RawHandle> + '_ {
            bytes.chunks_exact(INLINE_LEN).map(|chunk| {
                let mut raw = [0u8; INLINE_LEN];
                raw.copy_from_slice(chunk);
                raw
            })
        }

        fn model_strong_keep(
            roots: &[RawHandle],
            present: &BTreeSet<RawHandle>,
            wanted: &BTreeSet<RawHandle>,
            model: &Model,
        ) -> BTreeSet<RawHandle> {
            let mut queue = VecDeque::new();
            for root in roots {
                if !wanted.contains(root) {
                    queue.push_back(*root);
                }
            }

            let mut keep = BTreeSet::new();
            while let Some(raw) = queue.pop_front() {
                if !keep.insert(raw) || !present.contains(&raw) {
                    continue;
                }

                let Some(bytes) = model.bytes.get(&raw) else {
                    continue;
                };

                for child in child_chunks(bytes) {
                    if !wanted.contains(&child)
                        && present.contains(&child)
                        && model.bytes.contains_key(&child)
                        && !keep.contains(&child)
                    {
                        queue.push_back(child);
                    }
                }
            }

            keep
        }

        fn expected_live_after_collect(yard: &Yard, model: &Model) -> BTreeSet<RawHandle> {
            let present = live_union(yard);
            let wants_with_lru = wants(yard);
            let wanted = wants_with_lru.keys().copied().collect::<BTreeSet<_>>();
            let strong_keep = model_strong_keep(&strong_roots(yard), &present, &wanted, model);
            let want_keep = budgeted_wants(&wants_with_lru, &present, yard.config.want_budget);

            present
                .into_iter()
                .filter(|raw| strong_keep.contains(raw) || want_keep.contains(raw))
                .collect()
        }

        fn assert_readable_bytes(
            reader: &YardReader,
            raw: RawHandle,
            expected: &[u8],
            seed: u64,
            step: usize,
        ) {
            let actual = reader
                .get_local::<Bytes, UnknownBlob>(unknown(raw))
                .unwrap_or_else(|| {
                    panic!("seed {seed} step {step}: live handle {raw:02X?} was not readable")
                })
                .unwrap_or_else(|err| {
                    panic!("seed {seed} step {step}: live handle {raw:02X?} errored: {err}")
                });
            assert_eq!(
                actual.as_ref(),
                expected,
                "seed {seed} step {step}: readable bytes changed for {raw:02X?}"
            );
        }

        fn assert_general_invariants(yard: &mut Yard, model: &Model, seed: u64, step: usize) {
            let reader = yard.reader().unwrap();
            let live = live_union(yard);
            let wanted = wants(yard).keys().copied().collect::<BTreeSet<_>>();
            let strong_keep = model_strong_keep(&strong_roots(yard), &live, &wanted, model);

            for raw in strong_keep.intersection(&live) {
                let expected = model
                    .bytes
                    .get(raw)
                    .unwrap_or_else(|| panic!("seed {seed} step {step}: unknown live handle"));
                assert_readable_bytes(&reader, *raw, expected, seed, step);
            }

            if let Some(raw) = wanted.intersection(&strong_keep).next() {
                panic!("seed {seed} step {step}: want {raw:02X?} leaked into strong keep");
            }

            for raw in &live {
                let expected = model.bytes.get(raw).unwrap_or_else(|| {
                    panic!("seed {seed} step {step}: live set has unknown blob")
                });
                assert_readable_bytes(&reader, *raw, expected, seed, step);
                let _ = reader.children(unknown(*raw));
            }

            for raw in model.bytes.keys().filter(|raw| !live.contains(*raw)) {
                assert!(
                    reader
                        .get_local::<Bytes, UnknownBlob>(unknown(*raw))
                        .is_none(),
                    "seed {seed} step {step}: non-live handle {raw:02X?} was readable"
                );
            }

            for raw in &model.absent {
                assert!(
                    reader
                        .get_local::<Bytes, UnknownBlob>(unknown(*raw))
                        .is_none(),
                    "seed {seed} step {step}: absent handle {raw:02X?} became readable"
                );
                assert!(
                    reader.children(unknown(*raw)).is_empty(),
                    "seed {seed} step {step}: absent handle {raw:02X?} had children"
                );
            }
        }

        fn assert_exact_collect_result(
            yard: &mut Yard,
            expected: &BTreeSet<RawHandle>,
            model: &Model,
            seed: u64,
            step: usize,
        ) {
            let actual = live_union(yard);
            assert_eq!(
                &actual, expected,
                "seed {seed} step {step}: live union after collection did not equal keep set"
            );
            assert_general_invariants(yard, model, seed, step);
        }

        fn snapshot_readable(yard: &mut Yard) -> BTreeMap<RawHandle, Vec<u8>> {
            let reader = yard.reader().unwrap();
            live_union(yard)
                .into_iter()
                .filter_map(|raw| {
                    reader
                        .get_local::<Bytes, UnknownBlob>(unknown(raw))
                        .map(|result| (raw, result.unwrap().as_ref().to_vec()))
                })
                .collect()
        }

        fn assert_reclaim_preserved(
            yard: &mut Yard,
            before: &BTreeMap<RawHandle, Vec<u8>>,
            model: &Model,
            seed: u64,
            step: usize,
        ) {
            let reader = yard.reader().unwrap();
            let live = live_union(yard);
            for (raw, bytes) in before {
                assert!(
                    live.contains(raw),
                    "seed {seed} step {step}: reclaim removed live handle {raw:02X?}"
                );
                assert_readable_bytes(&reader, *raw, bytes, seed, step);
            }
            for raw in model.bytes.keys().filter(|raw| !live.contains(*raw)) {
                assert!(
                    reader
                        .get_local::<Bytes, UnknownBlob>(unknown(*raw))
                        .is_none(),
                    "seed {seed} step {step}: reclaim exposed non-live handle {raw:02X?}"
                );
            }
        }

        fn fresh_absent_handle(rng: &mut SplitMix64, model: &mut Model) -> RawHandle {
            let mut bytes = vec![0u8; 48];
            rng.fill(&mut bytes);
            let handle = Blob::<UnknownBlob>::new(Bytes::from_source(bytes)).get_handle();
            model.absent.push(handle.raw);
            handle.raw
        }

        fn choose_known_or_absent(rng: &mut SplitMix64, model: &mut Model) -> RawHandle {
            if !model.handles.is_empty() && rng.chance(3, 4) {
                model.handles[rng.index(model.handles.len())]
            } else {
                fresh_absent_handle(rng, model)
            }
        }

        fn choose_want_target(
            yard: &Yard,
            rng: &mut SplitMix64,
            model: &mut Model,
            mode: WantMode,
        ) -> RawHandle {
            match mode {
                WantMode::AnyKnownHandle => choose_known_or_absent(rng, model),
                WantMode::YoungOnly => {
                    let young = live_sets(yard)
                        .first()
                        .into_iter()
                        .flat_map(|set| set.iter())
                        .copied()
                        .collect::<Vec<_>>();
                    if !young.is_empty() && rng.chance(3, 4) {
                        young[rng.index(young.len())]
                    } else {
                        fresh_absent_handle(rng, model)
                    }
                }
            }
        }

        fn put_fresh_blob(
            yard: &mut Yard,
            model: &mut Model,
            rng: &mut SplitMix64,
            seed: u64,
            step: usize,
        ) {
            let mut bytes = Vec::new();
            let mut unique = [0u8; INLINE_LEN];
            unique[..8].copy_from_slice(&seed.to_le_bytes());
            unique[8..16].copy_from_slice(&(step as u64).to_le_bytes());
            unique[16..24].copy_from_slice(&rng.next_u64().to_le_bytes());
            unique[24..32].copy_from_slice(&rng.next_u64().to_le_bytes());
            bytes.extend_from_slice(&unique);

            let child_count = if model.handles.is_empty() {
                0
            } else {
                rng.index(4)
            };
            for _ in 0..child_count {
                let child = choose_known_or_absent(rng, model);
                bytes.extend_from_slice(&child);
            }

            let noise_len = rng.index(17);
            let mut noise = vec![0u8; noise_len];
            rng.fill(&mut noise);
            bytes.extend_from_slice(&noise);

            let blob = Blob::<UnknownBlob>::new(Bytes::from_source(bytes.clone()));
            let expected = blob.get_handle();
            let handle = if rng.chance(2, 3) {
                yard.put::<UnknownBlob, _>(blob).unwrap()
            } else {
                let level = rng.index(GENERATIONS);
                yard.put_in_generation::<UnknownBlob, _>(level, blob)
                    .unwrap()
            };
            assert_eq!(handle.raw, expected.raw);

            model.bytes.entry(handle.raw).or_insert(bytes);
            if !model.handles.contains(&handle.raw) {
                model.handles.push(handle.raw);
            }
        }

        fn run_one(seed: u64, want_mode: WantMode) -> FinalState {
            let (_dir, mut yard) = yard_with(
                GENERATIONS,
                YardConfig {
                    want_budget: 3,
                    strong_level_budget: 2,
                    fanout: 2,
                },
            );
            let mut rng = SplitMix64::new(seed);
            let mut model = Model::new();

            for step in 0..STEPS {
                match rng.index(9) {
                    0 | 1 => put_fresh_blob(&mut yard, &mut model, &mut rng, seed, step),
                    2 => {
                        if !model.handles.is_empty() {
                            let pin = pin_id(rng.index(PIN_COUNT));
                            let raw = model.handles[rng.index(model.handles.len())];
                            yard.pin_strong(pin, unknown(raw));
                        }
                    }
                    3 => yard.unpin_strong(pin_id(rng.index(PIN_COUNT))),
                    4 => {
                        let raw = choose_want_target(&yard, &mut rng, &mut model, want_mode);
                        yard.want(unknown(raw)).unwrap();
                    }
                    5 => {
                        let raw = choose_known_or_absent(&mut rng, &mut model);
                        let reader = yard.reader().unwrap();
                        let result = reader.get::<Bytes, UnknownBlob>(unknown(raw));
                        if !live_union(&yard).contains(&raw) {
                            assert!(
                                matches!(result, Err(YardGetError::NotFound)),
                                "seed {seed} step {step}: absent get did not miss cleanly"
                            );
                        }
                    }
                    6 => {
                        let expected = expected_live_after_collect(&yard, &model);
                        yard.collect(&RetentionRoots::new()).unwrap();
                        assert_exact_collect_result(&mut yard, &expected, &model, seed, step);
                    }
                    7 => {
                        let expected = expected_live_after_collect(&yard, &model);
                        yard.compact(&RetentionRoots::new()).unwrap();
                        assert_exact_collect_result(&mut yard, &expected, &model, seed, step);
                    }
                    8 => {
                        let before = snapshot_readable(&mut yard);
                        yard.reclaim().unwrap();
                        assert_reclaim_preserved(&mut yard, &before, &model, seed, step);
                    }
                    _ => unreachable!(),
                }

                assert_general_invariants(&mut yard, &model, seed, step);
            }

            let reader = yard.reader().unwrap();
            let mut live_by_generation = live_sets(&yard)
                .into_iter()
                .map(|set| set.into_iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            for generation in &mut live_by_generation {
                generation.sort();
            }
            let mut readable = live_union(&yard)
                .into_iter()
                .filter(|raw| {
                    reader
                        .get_local::<Bytes, UnknownBlob>(unknown(*raw))
                        .is_some()
                })
                .collect::<Vec<_>>();
            readable.sort();

            FinalState {
                live_by_generation,
                readable,
            }
        }

        #[test]
        fn seeded_yard_property_sequences() {
            for seed in 0..SEEDS {
                run_one(0xC0DE_0000_0000_0000 ^ seed, WantMode::YoungOnly);
            }
        }

        #[test]
        fn seeded_yard_property_sequences_are_deterministic() {
            for seed in [0, 13, 49] {
                let seed = 0xD57D_0000_0000_0000 ^ seed;
                assert_eq!(
                    run_one(seed, WantMode::YoungOnly),
                    run_one(seed, WantMode::YoungOnly),
                    "seed {seed} diverged"
                );
            }
        }

        #[test]
        fn seeded_yard_property_sequences_include_resident_wants() {
            // Exercise explicit wants for absent, young, and already-tenured
            // blobs. The model treats every assertion uniformly and verifies
            // the same bounded cache policy after collection.
            for seed in [0, 2, 7, 13, 31, 49] {
                run_one(0xC0DE_0000_0000_0000 ^ seed, WantMode::AnyKnownHandle);
            }
        }

        #[test]
        fn explicit_want_on_resident_blob_is_recorded_and_budgeted() {
            let (_dir, mut yard) = yard_with(
                3,
                YardConfig {
                    want_budget: 0,
                    strong_level_budget: 0,
                    fanout: 1,
                },
            );
            let tenured = yard
                .put::<UnknownBlob, _>(Bytes::from_source(b"tenured then wanted".to_vec()))
                .unwrap();

            yard.pin_strong(pin_id(0), tenured);
            yard.compact(&RetentionRoots::new()).unwrap();
            assert!(yard.contains_in_generation(2, tenured));

            // Explicit interest is recorded even though the bytes are already
            // present. With a zero want budget, that makes the previously
            // strong resident eligible for eviction on the next collection.
            yard.want(tenured).unwrap();
            assert_eq!(
                yard.wants()
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
                vec![tenured]
            );
            yard.collect(&RetentionRoots::new()).unwrap();

            assert!(
                !yard.contains_in_generation(2, tenured),
                "zero want budget must evict explicitly wanted resident content"
            );
        }
    }
}
