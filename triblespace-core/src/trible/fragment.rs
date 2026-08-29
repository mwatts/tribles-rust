use std::ops::{Add, AddAssign, Deref};

use crate::blob::{BlobEncoding, IntoBlob, MemoryBlobStore};
use crate::id::Id;
use crate::id::RawId;
use crate::inline::encodings::hash::Handle;
use crate::inline::Inline;
use crate::metadata::Describe;
use crate::patch::Entry;
use crate::patch::PATCH;

use super::Trible;
use super::TribleSet;

/// A rooted (or multi-root) fragment of a knowledge graph.
///
/// A fragment is a [`TribleSet`] plus a (possibly empty) set of "exported" entity
/// ids that act as entry points into the contained facts, plus the
/// [`MemoryBlobStore`] holding any bytes the contained facts reference
/// by handle. Exports are not privileged in the graph model itself;
/// they are simply the ids the producer wants to hand back to the
/// caller as the fragment's interface.
///
/// The embedded blob store is what makes a Fragment *self-contained*:
/// handles in either fact set can reference bytes that the fragment carries
/// with itself. An empty `MemoryBlobStore` is structurally a single PATCH-root
/// pointer — fragments without blobs pay essentially zero overhead.
///
/// Alongside its content facts, a fragment carries a separate set of
/// **metafacts** describing the attributes used by that content. Keeping the
/// two sets separate means ordinary content queries never see schema records,
/// while publication can archive both halves atomically. One shared blob store
/// backs handles from either set: content addressing deduplicates equal bytes,
/// and a fragment remains one indivisible ownership unit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Fragment {
    exports: PATCH<16>,
    facts: TribleSet,
    metafacts: TribleSet,
    blobs: MemoryBlobStore,
}

impl Fragment {
    /// Creates an empty fragment with no exports and no facts.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates a fragment that exports a single root id, with the
    /// given facts and an empty blob store.
    pub fn rooted(root: Id, facts: TribleSet) -> Self {
        let mut exports = PATCH::<16>::new();
        let raw: RawId = root.into();
        exports.insert(&Entry::new(&raw));
        Self {
            exports,
            facts,
            metafacts: TribleSet::new(),
            blobs: MemoryBlobStore::new(),
        }
    }

    /// Creates a fragment with the given exported ids and an empty blob store.
    ///
    /// Export ids are canonicalized as a set (duplicates are ignored). Empty
    /// exports are allowed.
    pub fn new<I>(exports: I, facts: TribleSet) -> Self
    where
        I: IntoIterator<Item = Id>,
    {
        let mut export_set = PATCH::<16>::new();
        for id in exports {
            let raw: RawId = id.into();
            export_set.insert(&Entry::new(&raw));
        }
        Self {
            exports: export_set,
            facts,
            metafacts: TribleSet::new(),
            blobs: MemoryBlobStore::new(),
        }
    }

    /// Creates a fragment with no exports, holding the given facts and
    /// blob store. Useful when re-wrapping the tail of a destructured
    /// fragment (e.g. inside `Spread::spread`) where the exports have
    /// already been consumed.
    pub fn from_facts_and_blobs(facts: TribleSet, blobs: MemoryBlobStore) -> Self {
        Self {
            exports: PATCH::<16>::new(),
            facts,
            metafacts: TribleSet::new(),
            blobs,
        }
    }

    /// Creates an unrooted fragment from all three carried data channels.
    ///
    /// This is the inverse of [`Fragment::into_parts`] minus the exports and is
    /// primarily useful when composition consumes a fragment's exported ids.
    pub fn from_parts(facts: TribleSet, metafacts: TribleSet, blobs: MemoryBlobStore) -> Self {
        Self {
            exports: PATCH::<16>::new(),
            facts,
            metafacts,
            blobs,
        }
    }

    /// Creates a fragment that exports a single root id, with the given facts
    /// and blob store but no metafacts. Producers that also have descriptions
    /// should use [`Fragment::rooted_from_parts`].
    pub fn rooted_with_blobs(root: Id, facts: TribleSet, blobs: MemoryBlobStore) -> Self {
        Self::rooted_from_parts(root, facts, TribleSet::new(), blobs)
    }

    /// Creates a rooted fragment from content, description, and their shared
    /// blob store.
    pub fn rooted_from_parts(
        root: Id,
        facts: TribleSet,
        metafacts: TribleSet,
        blobs: MemoryBlobStore,
    ) -> Self {
        let mut exports = PATCH::<16>::new();
        let raw: RawId = root.into();
        exports.insert(&Entry::new(&raw));
        Self {
            exports,
            facts,
            metafacts,
            blobs,
        }
    }

    /// Insert a blob into the fragment's local blob store and return the
    /// content-addressed handle that references it.
    ///
    /// Use this when you want a Fragment to be self-contained — every
    /// handle in its facts has its bytes available without consulting
    /// an external blob store. Idempotent under content addressing:
    /// putting the same bytes twice returns the same handle and
    /// doesn't grow the store.
    pub fn put<S, T>(&mut self, item: T) -> Inline<Handle<S>>
    where
        S: BlobEncoding,
        T: IntoBlob<S>,
    {
        self.blobs.insert(item.to_blob())
    }

    /// Returns the exported ids for this fragment, in deterministic (lexicographic) order.
    pub fn exports(&self) -> impl Iterator<Item = Id> + '_ {
        self.exports
            .iter_ordered()
            .map(|raw| Id::new(*raw).expect("export ids are non-nil"))
    }

    /// Returns the single exported id if this fragment is rooted.
    pub fn root(&self) -> Option<Id> {
        if self.exports.len() == 1 {
            let raw = self
                .exports
                .iter_ordered()
                .next()
                .expect("len() == 1 implies a first element exists");
            Some(Id::new(*raw).expect("export ids are non-nil"))
        } else {
            None
        }
    }

    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    /// Mutable access to the fragment's facts, for producers that
    /// accumulate tribles directly (e.g. importers inserting per-row
    /// facts alongside `put`-ing the blobs those facts reference).
    pub fn facts_mut(&mut self) -> &mut TribleSet {
        &mut self.facts
    }

    /// Returns the schema and other descriptive facts carried alongside the
    /// content.
    pub fn metafacts(&self) -> &TribleSet {
        &self.metafacts
    }

    /// Mutable access for importers that discover attributes while ingesting
    /// data.
    pub fn metafacts_mut(&mut self) -> &mut TribleSet {
        &mut self.metafacts
    }

    /// Promotes another fragment into this fragment's description channel.
    ///
    /// Both of the description fragment's fact sets become metafacts here;
    /// its exports are intentionally ignored, and its blobs join the one
    /// shared store. This is the runtime-schema counterpart to the metadata
    /// emitted automatically by `entity!`.
    pub fn describe_with(&mut self, description: Fragment) {
        let (_, facts, metafacts, blobs) = description.into_parts();
        self.metafacts += facts;
        self.metafacts += metafacts;
        self.blobs.union(blobs);
    }

    /// Borrow the fragment's local blob store.
    pub fn blobs(&self) -> &MemoryBlobStore {
        &self.blobs
    }

    /// Mutable access to the fragment's local blob store, for
    /// producers that need to merge an existing store in bulk
    /// (`blobs_mut().union(other)`) rather than `put` items one at
    /// a time.
    pub fn blobs_mut(&mut self) -> &mut MemoryBlobStore {
        &mut self.blobs
    }

    pub fn into_facts(self) -> TribleSet {
        self.facts
    }

    /// Consume the fragment, yielding only its metafacts.
    pub fn into_metafacts(self) -> TribleSet {
        self.metafacts
    }

    /// Consume the fragment, yielding its content facts and shared blob store.
    ///
    /// Exports and metafacts are dropped. Because the store is shared, it may
    /// conservatively include bytes referenced only by the discarded
    /// metafacts. Use [`Fragment::into_parts`] when publishing both channels.
    pub fn into_facts_and_blobs(self) -> (TribleSet, MemoryBlobStore) {
        (self.facts, self.blobs)
    }

    /// Consume the complete fragment into exports, facts, metafacts, and the
    /// blob store shared by both fact sets.
    pub fn into_parts(self) -> (PATCH<16>, TribleSet, TribleSet, MemoryBlobStore) {
        (self.exports, self.facts, self.metafacts, self.blobs)
    }
}

impl Describe for Fragment {
    /// Returns this fragment's description as an ordinary fragment.
    ///
    /// Metafacts become the description's content facts. The shared blob store
    /// is retained conservatively because it owns attachments referenced by
    /// both the content and its description; this lets a generic publication
    /// path stage the complete closure before consuming the original value.
    fn describe(&self) -> Fragment {
        Fragment::from_facts_and_blobs(self.metafacts.clone(), self.blobs.clone())
    }
}

impl Deref for Fragment {
    type Target = TribleSet;

    fn deref(&self) -> &Self::Target {
        &self.facts
    }
}

impl<'a> IntoIterator for &'a Fragment {
    type Item = &'a Trible;
    type IntoIter = super::tribleset::TribleSetIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.facts.iter()
    }
}

impl AddAssign for Fragment {
    fn add_assign(&mut self, rhs: Self) {
        self.facts += rhs.facts;
        self.metafacts += rhs.metafacts;
        self.exports.union(rhs.exports);
        self.blobs.union(rhs.blobs);
    }
}

impl AddAssign<TribleSet> for Fragment {
    /// Facts-only merge — does not touch exports or blobs.
    fn add_assign(&mut self, rhs: TribleSet) {
        self.facts += rhs;
    }
}

impl Add for Fragment {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<TribleSet> for Fragment {
    type Output = Self;

    fn add(mut self, rhs: TribleSet) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign<Fragment> for TribleSet {
    fn add_assign(&mut self, rhs: Fragment) {
        self.union(rhs.facts);
    }
}

impl Add<Fragment> for TribleSet {
    type Output = Self;

    fn add(mut self, rhs: Fragment) -> Self::Output {
        self += rhs;
        self
    }
}

/// Promote a bare `TribleSet` into an undescribed fragment.
///
/// This remains available for the legacy repository surface while collection
/// publication migrates to concrete, self-describing `Fragment` values. New
/// producers should prefer `entity!` or an explicit fragment constructor so
/// metadata is not omitted accidentally.
impl From<TribleSet> for Fragment {
    fn from(facts: TribleSet) -> Self {
        Self::from_facts_and_blobs(facts, MemoryBlobStore::new())
    }
}

impl From<Fragment> for TribleSet {
    fn from(value: Fragment) -> Self {
        value.facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macros::entity;
    use crate::metadata;

    #[test]
    fn describe_promotes_metafacts_and_keeps_the_attachment_closure() {
        let fragment = entity! { _ @
            metadata::description: "fragment content",
        };
        assert!(!fragment.metafacts().is_empty());
        assert!(!fragment.blobs().is_empty());

        let description = Describe::describe(&fragment);

        assert_eq!(description.facts(), fragment.metafacts());
        assert!(description.metafacts().is_empty());
        assert_eq!(description.exports().count(), 0);
        assert_eq!(description.blobs(), fragment.blobs());
    }
}
