use crate::query::Binding;
use crate::query::Constraint;
use crate::query::Variable;
use crate::query::VariableId;
use crate::query::VariableSet;
use crate::trible::TribleSet;
use crate::value::RawValue;
use crate::value::Value;
use crate::value::ValueSchema;
use crate::value::VALUE_LEN;
/// A value-range-aware constraint that uses the TribleSet's AVE index
/// to propose only values in a byte-lexicographic range.
///
/// When proposing for the value variable with the attribute bound, it
/// calls `infixes_range` on the AVE index — the trie skips entire
/// subtrees outside the range. This makes range queries O(k + pruned)
/// instead of O(n).
///
/// Create via [`TribleSet::value_in_range`]:
///
/// ```rust,ignore
/// find!((id: Id, ts: Value<NsTAIInterval>),
///     and!(
///         pattern!(data, [{ ?id @ exec::requested_at: ?ts }]),
///         data.value_in_range(ts, min_ts, max_ts),
///     )
/// )
/// ```
pub struct TribleSetRangeConstraint {
    variable_v: VariableId,
    min: RawValue,
    max: RawValue,
    set: TribleSet,
}

impl TribleSetRangeConstraint {
    pub fn new<V: ValueSchema>(
        variable_v: Variable<V>,
        min: Value<V>,
        max: Value<V>,
        set: TribleSet,
    ) -> Self {
        TribleSetRangeConstraint {
            variable_v: variable_v.index,
            min: min.raw,
            max: max.raw,
            set,
        }
    }
}

impl<'a> Constraint<'a> for TribleSetRangeConstraint {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.variable_v)
    }

    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if variable != self.variable_v {
            return None;
        }
        // Use count_range on the VEA index for an accurate estimate.
        // This counts leaves in the byte range using cached branch counts,
        // visiting only boundary nodes — O(depth × branching) not O(n).
        let count = self
            .set
            .vea
            .count_range::<0, VALUE_LEN>(&[0u8; 0], &self.min, &self.max);
        Some(count.min(usize::MAX as u64) as usize)
    }

    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable != self.variable_v {
            return;
        }
        // Scan the VEA index for all values in range.
        // VEA tree order: V(32) → E(16) → A(16).
        // With empty prefix, infixes_range on V(32 bytes) gives us all
        // values in [min, max]. The trie prunes branches outside the range.
        self.set
            .vea
            .infixes_range::<0, VALUE_LEN, _>(&[0u8; 0], &self.min, &self.max, |v| {
                proposals.push(*v);
            });
    }

    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if variable == self.variable_v {
            proposals.retain(|v| *v >= self.min && *v <= self.max);
        }
    }

    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.variable_v) {
            Some(v) => *v >= self.min && *v <= self.max,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::valueschemas::R256BE;
    use crate::prelude::*;

    attributes! {
        "BB00000000000000BB00000000000000" as range_test_score: R256BE;
    }

    #[test]
    fn value_in_range_proposes_correctly() {
        let e1 = ufoid();
        let e2 = ufoid();
        let e3 = ufoid();
        let e4 = ufoid();

        let v10: Value<R256BE> = 10i128.to_value();
        let v50: Value<R256BE> = 50i128.to_value();
        let v90: Value<R256BE> = 90i128.to_value();
        let v100: Value<R256BE> = 100i128.to_value();

        let mut data = TribleSet::new();
        data += entity! { &e1 @ range_test_score: v10 };
        data += entity! { &e2 @ range_test_score: v50 };
        data += entity! { &e3 @ range_test_score: v90 };
        data += entity! { &e4 @ range_test_score: v100 };

        // Without range: all 4 results.
        let all: Vec<Value<R256BE>> = find!(
            v: Value<R256BE>,
            pattern!(&data, [{ range_test_score: ?v }])
        )
        .collect();
        assert_eq!(all.len(), 4);

        // With value_in_range [20..=95]: only v50 and v90.
        let min: Value<R256BE> = 20i128.to_value();
        let max: Value<R256BE> = 95i128.to_value();
        let mut filtered: Vec<Value<R256BE>> = find!(
            v: Value<R256BE>,
            and!(
                pattern!(&data, [{ range_test_score: ?v }]),
                data.value_in_range(v, min, max),
            )
        )
        .collect();
        filtered.sort();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0], v50);
        assert_eq!(filtered[1], v90);

        // Boundary: exact match on min and max.
        let min_exact: Value<R256BE> = 50i128.to_value();
        let max_exact: Value<R256BE> = 90i128.to_value();
        let mut exact: Vec<Value<R256BE>> = find!(
            v: Value<R256BE>,
            and!(
                pattern!(&data, [{ range_test_score: ?v }]),
                data.value_in_range(v, min_exact, max_exact),
            )
        )
        .collect();
        exact.sort();
        assert_eq!(exact.len(), 2);
        assert_eq!(exact[0], v50);
        assert_eq!(exact[1], v90);

        // Empty range: no results.
        let min_empty: Value<R256BE> = 91i128.to_value();
        let max_empty: Value<R256BE> = 99i128.to_value();
        let empty: Vec<Value<R256BE>> = find!(
            v: Value<R256BE>,
            and!(
                pattern!(&data, [{ range_test_score: ?v }]),
                data.value_in_range(v, min_empty, max_empty),
            )
        )
        .collect();
        assert_eq!(empty.len(), 0);
    }
}
