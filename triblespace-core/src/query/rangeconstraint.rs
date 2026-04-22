use super::*;

/// Restricts a variable's raw value to a byte-lexicographic range.
///
/// This constraint only **confirms** — it never proposes candidates.
/// Use it with [`and!`](crate::and) alongside a constraint that does
/// propose (e.g. a [`pattern!`](crate::macros::pattern)):
///
/// ```rust,ignore
/// find!((id: Id, ts: Value<NsTAIInterval>),
///     and!(
///         pattern!(data, [{ ?id @ exec::requested_at: ?ts }]),
///         value_range(ts, min_ts, max_ts),
///     )
/// )
/// ```
///
/// The estimate returns `usize::MAX` so the intersection sorts this
/// constraint last — the tighter TribleSet constraint proposes first,
/// then this range constraint filters.
pub struct ValueRange {
    variable: VariableId,
    min: RawValue,
    max: RawValue,
}

impl ValueRange {
    /// Create a range constraint on `variable` with inclusive bounds.
    pub fn new<T: ValueSchema>(variable: Variable<T>, min: Value<T>, max: Value<T>) -> Self {
        ValueRange {
            variable: variable.index,
            min: min.raw,
            max: max.raw,
        }
    }
}

/// Convenience function to create a [`ValueRange`] constraint.
pub fn value_range<T: ValueSchema>(
    variable: Variable<T>,
    min: Value<T>,
    max: Value<T>,
) -> ValueRange {
    ValueRange::new(variable, min, max)
}

impl<'a> Constraint<'a> for ValueRange {
    fn variables(&self) -> VariableSet {
        VariableSet::new_singleton(self.variable)
    }

    /// Returns `usize::MAX` so the intersection never chooses this
    /// constraint as the proposer — it only confirms.
    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.variable == variable {
            Some(usize::MAX)
        } else {
            None
        }
    }

    /// Does not propose — the paired TribleSet constraint handles proposals.
    fn propose(&self, _variable: VariableId, _binding: &Binding, _proposals: &mut Vec<RawValue>) {
        // Intentionally empty: this constraint only confirms.
    }

    /// Retains only proposals whose raw bytes fall within [min, max] inclusive.
    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.variable == variable {
            proposals.retain(|v| *v >= self.min && *v <= self.max);
        }
    }

    /// Returns `false` when the variable is bound to a value outside the range.
    fn satisfied(&self, binding: &Binding) -> bool {
        match binding.get(self.variable) {
            Some(v) => *v >= self.min && *v <= self.max,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::valueschemas::R256;
    use crate::prelude::*;

    attributes! {
        "AA00000000000000AA00000000000000" as test_score: R256;
    }

    #[test]
    fn value_range_filters_correctly() {
        let e1 = ufoid();
        let e2 = ufoid();
        let e3 = ufoid();

        let v10: Value<R256> = 10i128.to_value();
        let v50: Value<R256> = 50i128.to_value();
        let v90: Value<R256> = 90i128.to_value();

        let mut data = TribleSet::new();
        data += entity! { &e1 @ test_score: v10 };
        data += entity! { &e2 @ test_score: v50 };
        data += entity! { &e3 @ test_score: v90 };

        // Without range: all 3 results.
        let all: Vec<Value<R256>> = find!(
            v: Value<R256>,
            pattern!(&data, [{ test_score: ?v }])
        )
        .collect();
        assert_eq!(all.len(), 3);

        // With range [20..80]: only v50.
        let min: Value<R256> = 20i128.to_value();
        let max: Value<R256> = 80i128.to_value();
        let filtered: Vec<Value<R256>> = find!(
            v: Value<R256>,
            and!(
                pattern!(&data, [{ test_score: ?v }]),
                value_range(v, min, max),
            )
        )
        .collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], v50);
    }
}
