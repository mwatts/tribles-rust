//! Reading one collection descriptor.
//!
//! A descriptor is an ordinary [`TribleSet`]: the facts of one
//! [`entity!`](crate::macros::entity), stored as a
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive)
//! blob whose handle is the collection identity. There is no wrapper type, so
//! reading one is an ordinary query over ordinary facts.
//!
//! Nothing here validates shape. A descriptor may carry attributes this binary
//! has never heard of: they are arguments to a recipe it does not implement,
//! they travel through untouched, and classifying them is nobody's business
//! but that recipe's. By convention a root carries
//! [`collection_name`] and [`collection_team`] while a derived collection
//! carries [`collection_source`] instead, but that is what the recipes agree
//! on, not something enforced here.

use ed25519_dalek::VerifyingKey;

use crate::id::Id;
use crate::inline::encodings::genid::GenId;
use crate::inline::encodings::shortstring::ShortString;
use crate::inline::{Inline, IntoInline, RawInline};
use crate::metadata;
use crate::prelude::{entity, find, pattern};
use crate::query::TriblePattern;
use crate::temp;
use crate::trible::{Fragment, TribleSet};

use super::records::{
    collection_name, collection_recipe, collection_representation, collection_source,
    collection_team, CollectionHandle, CollectionName, RecordDecodeError,
    KIND_COLLECTION_DESCRIPTOR,
};

/// Build a root descriptor that names its representation and recipe without
/// describing them.
///
/// A collection kind normally writes its own descriptor as a visible
/// `entity!` beside the recipe that reads it; see
/// [`simplearchive_union::descriptor`](crate::collection::simplearchive_union::descriptor),
/// which additionally embeds both self-descriptions so a stranger holding the
/// one blob can say what the collection is. This is the bare generic form,
/// for callers holding only ids.
pub fn naming(
    name: &CollectionName,
    team: VerifyingKey,
    representation: Id,
    recipe: Id,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.as_str(),
        collection_team: team,
        collection_representation: representation,
        collection_recipe: recipe,
    }
}

/// The entity the descriptor's own attributes hang off.
///
/// A descriptor archive holds more than one entity: the descriptor, plus the
/// embedded self-descriptions of its representation and its recipe. The
/// descriptor is the one tagged [`KIND_COLLECTION_DESCRIPTOR`].
///
/// This is not the collection identity. That is the handle of the stored
/// descriptor blob.
///
/// For the decoded path only: a descriptor read back out of a blob is a bare
/// `TribleSet`, so finding its root means looking for the tag. A caller that
/// *built* the descriptor already holds the root and should use
/// [`Fragment::root`](crate::trible::Fragment::root) rather than pay this scan
/// to recover something it never lost.
pub fn entity(facts: &TribleSet) -> Result<Id, RecordDecodeError> {
    exactly_one(
        find!(
            (e: Id),
            pattern!(facts, [{ ?e @ metadata::tag: KIND_COLLECTION_DESCRIPTOR }])
        )
        .map(|(e,)| e),
        "metadata::tag",
    )
}

/// Blob representation carried by the elements of this collection.
pub fn representation(facts: &TribleSet) -> Result<Id, RecordDecodeError> {
    exactly_one(
        find!(
            (v: Id?),
            pattern!(facts, [{ _?e @ collection_representation: ?v }])
        )
        .map(|(v,)| v),
        "collection_representation",
    )?
    .map_err(|_| RecordDecodeError::InvalidId("collection_representation"))
}

/// Canonical recipe governing construction and merge for this collection.
///
/// This names the *law*. Its arguments, if any, are further attributes on the
/// same entity; see [`argument`].
pub fn recipe(facts: &TribleSet) -> Result<Id, RecordDecodeError> {
    exactly_one(
        find!((v: Id?), pattern!(facts, [{ _?e @ collection_recipe: ?v }])).map(|(v,)| v),
        "collection_recipe",
    )?
    .map_err(|_| RecordDecodeError::InvalidId("collection_recipe"))
}

/// The collection this one derives from, if it derives from one.
///
/// A root has no source and answers `None`; that is not a failure, it is what
/// being a root means.
pub fn source(facts: &TribleSet) -> Option<CollectionHandle> {
    find!(
        (v: CollectionHandle),
        pattern!(facts, [{ _?e @ collection_source: ?v }])
    )
    .map(|(v,)| v)
    .next()
}

/// The name a root collection is known by within its team.
///
/// A derived collection has no name of its own and answers `None`: its anchor
/// is its source, and its name is whatever that source is called.
pub fn name(facts: &TribleSet) -> Option<Result<CollectionName, RecordDecodeError>> {
    let raw: Inline<ShortString> = find!(
        (v: Inline<ShortString>),
        pattern!(facts, [{ _?e @ collection_name: ?v }])
    )
    .map(|(v,)| v)
    .next()?;
    Some(
        raw.try_from_inline::<String>()
            .map_err(|_| RecordDecodeError::InvalidId("collection_name"))
            .and_then(|text| {
                CollectionName::new(&text)
                    .map_err(|_| RecordDecodeError::InvalidId("collection_name"))
            }),
    )
}

/// Root public key of the team a root collection belongs to.
///
/// A derived collection answers `None` and inherits its team from its source,
/// transitively.
pub fn team(facts: &TribleSet) -> Option<Result<VerifyingKey, RecordDecodeError>> {
    let key = find!(
        (v: VerifyingKey?),
        pattern!(facts, [{ _?e @ collection_team: ?v }])
    )
    .map(|(v,)| v)
    .next()?;
    Some(key.map_err(|_| RecordDecodeError::InvalidId("collection_team")))
}

/// Look up one recipe argument by attribute.
///
/// The attribute is a runtime id because the recipe that minted it is the only
/// thing that knows what it means; this reads the raw bytes and hands them
/// back for that recipe to interpret.
pub fn argument(facts: &TribleSet, attribute: Id) -> Option<RawInline> {
    let attribute: Inline<GenId> = attribute.to_inline();
    find!(
        (v: Inline<GenId>),
        temp!((e), facts.pattern::<GenId>(e, attribute, v))
    )
    .map(|(v,)| v.raw)
    .next()
}

fn exactly_one<T>(
    mut rows: impl Iterator<Item = T>,
    field: &'static str,
) -> Result<T, RecordDecodeError> {
    let Some(value) = rows.next() else {
        return Err(RecordDecodeError::MissingField(field));
    };
    if rows.next().is_some() {
        return Err(RecordDecodeError::RepeatedField(field));
    }
    Ok(value)
}

/// A root descriptor under one fixed test team, named by `name`.
///
/// Tests overwhelmingly want "some collection, distinct from that other one".
/// Spelling that as a name under a fixed team keeps the distinction readable
/// in the test itself, which is the whole reason a name replaced an opaque
/// scope id.
#[cfg(test)]
pub(crate) fn named_for_tests(name: &str, representation: Id, recipe: Id) -> Fragment {
    let team = ed25519_dalek::SigningKey::from_bytes(&[0xAA; 32]).verifying_key();
    naming(
        &CollectionName::new(name).expect("test collection name"),
        team,
        representation,
        recipe,
    )
}

/// Content identity of a descriptor, for tests that have no store.
///
/// The record algebra -- which commits a ticket admits, which merges compose,
/// what a derive equates -- is a property of records alone, and its tests
/// build them without a pile. There is no `put` to take a handle from, so
/// there is nothing to forget to write and no phantom to create: the
/// descriptor is an input to the algebra rather than a thing in a store.
///
/// This is why it is `cfg(test)` rather than public. Production code always
/// has somewhere to put the descriptor, and takes the handle from what `put`
/// hands back -- because a handle computed beside a store instead of by it can
/// name a collection whose descriptor was never written, leaving records that
/// reference something nothing can decode.
#[cfg(test)]
pub(crate) fn identity_for_tests(descriptor: &Fragment) -> CollectionHandle {
    crate::blob::IntoBlob::<crate::blob::encodings::simplearchive::SimpleArchive>::to_blob(
        descriptor.facts().clone(),
    )
    .get_handle()
}
