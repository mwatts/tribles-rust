#![cfg(kani)]

use crate::id::{ExclusiveId, Id, ID_LEN};
use crate::inline::encodings::UnknownInline;
use crate::inline::{Inline, INLINE_LEN};
use crate::patch::Entry;
use crate::trible::Trible;
use kani::BoundedArbitrary;

/// Ensures the generated identifier is never nil by rejecting the sentinel.
fn non_nil_raw_id() -> [u8; ID_LEN] {
    let raw: [u8; ID_LEN] = kani::any();
    // Rule out the nil sentinel without biasing the remaining population.
    kani::assume(raw != [0u8; ID_LEN]);
    raw
}

/// Generate a bounded identifier suitable for use in Kani harnesses.
///
/// The value is guaranteed to be non-nil so it can be promoted to an
/// [`ExclusiveId`] when needed.
pub fn bounded_id() -> Id {
    Id::new(non_nil_raw_id()).expect("non-nil ids are always valid")
}

/// Generate a bounded, writeable identifier for harnesses that require
/// entity ownership.
pub fn bounded_exclusive_id() -> ExclusiveId {
    ExclusiveId::force(bounded_id())
}

/// Produce a value with a reduced search space for harnesses that only care
/// about byte-level behaviour.
pub fn bounded_unknown_value() -> Inline<UnknownInline> {
    let raw: [u8; INLINE_LEN] = kani::any();
    // Restrict the value to the lower nibble of each byte to keep the state
    // space manageable for symbolic execution while still covering a wide
    // range of patterns.
    let raw = raw.map(|byte| byte & 0x0F);
    Inline::new(raw)
}

/// Construct a single [`Trible`] using bounded identifiers and value bytes.
pub fn bounded_trible() -> Trible {
    let entity = bounded_exclusive_id();
    let attribute = bounded_id();
    let value = bounded_unknown_value();
    Trible::new(&entity, &attribute, &value)
}

/// Generate a bounded key for PATCH based structures by restricting each byte
/// to the lower nibble.
pub fn bounded_patch_key<const KEY_LEN: usize>() -> [u8; KEY_LEN] {
    let raw: [u8; KEY_LEN] = kani::any();
    raw.map(|byte| byte & 0x0F)
}

/// Produce a shareable PATCH entry with an empty payload along with the key
/// bytes used to construct it.
pub fn bounded_patch_entry<const KEY_LEN: usize>() -> ([u8; KEY_LEN], Entry<KEY_LEN>) {
    let key = bounded_patch_key::<KEY_LEN>();
    let entry = Entry::new(&key);
    (key, entry)
}

/// Produce a shareable PATCH entry with a bounded payload generated via the
/// [`BoundedArbitrary`] trait, returning both the key and entry so harnesses can
/// interact with the PATCH APIs that accept raw keys.
pub fn bounded_patch_entry_with_value<const KEY_LEN: usize, V, const MAX: usize>(
) -> ([u8; KEY_LEN], Entry<KEY_LEN, V>)
where
    V: BoundedArbitrary,
{
    let key = bounded_patch_key::<KEY_LEN>();
    let value = V::bounded_any::<MAX>();
    let entry = Entry::with_value(&key, value);
    (key, entry)
}
