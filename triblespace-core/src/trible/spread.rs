use crate::id::Id;

use super::Fragment;
use super::TribleSet;

/// Trait for types that can be "spread" into an `entity!` repeated attribute.
///
/// A spread produces an iterator of attribute values, plus an optional
/// set of extra facts that are merged into the entity's result fragment.
///
/// Plain iterators return an empty extra-facts set. A [`Fragment`] returns
/// its exported ids as the values and its contained facts as the extras.
pub trait Spread {
    /// The type of each yielded value.
    type Item;
    /// The iterator type returned by [`spread`](Spread::spread).
    type Iter: IntoIterator<Item = Self::Item>;
    /// Decomposes the value into an iterator of items and extra facts to merge.
    fn spread(self) -> (Self::Iter, TribleSet);
}

impl<I: IntoIterator> Spread for I {
    type Item = I::Item;
    type Iter = I;
    fn spread(self) -> (Self::Iter, TribleSet) {
        (self, TribleSet::new())
    }
}

impl Spread for Fragment {
    type Item = Id;
    type Iter = std::vec::IntoIter<Id>;
    fn spread(self) -> (Self::Iter, TribleSet) {
        let (exports, facts) = self.into_parts();
        let ids: Vec<Id> = exports
            .iter_ordered()
            .map(|raw| Id::new(*raw).expect("export ids are non-nil"))
            .collect();
        (ids.into_iter(), facts)
    }
}
