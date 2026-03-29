use crate::id::id_into_value;
use crate::id::ID_LEN;
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
use crate::value::schemas::genid::GenId;

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
/// find!((id: Id, ts: Value<OrderedNsTAIInterval>),
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
        let count = self.set.vea.count_range::<0, VALUE_LEN>(
            &[0u8; 0],
            &self.min,
            &self.max,
        );
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
        self.set.vea.infixes_range::<0, VALUE_LEN, _>(
            &[0u8; 0],
            &self.min,
            &self.max,
            |v| {
                proposals.push(*v);
            },
        );
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
