//! Reading one collection descriptor.
//!
//! A descriptor is an ordinary [`TribleSet`]: the facts of one
//! [`entity!`](crate::macros::entity), stored as a
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive)
//! blob whose handle is the collection identity. There is no wrapper type, so
//! reading one is an ordinary query over ordinary facts.
//!
//! A descriptor may carry attributes this binary has never heard of: they are
//! arguments to a recipe it does not implement, they travel through untouched,
//! and classifying them is nobody's business but that recipe's. A root carries
//! [`collection_name`] while a derived collection carries
//! [`collection_source`] instead. Both carry exactly one local
//! [`collection_authority`]; authority is never inferred by walking the source
//! chain. Readers first locate the one tagged descriptor entity, then bind
//! every field lookup to that exact entity so embedded descriptions cannot
//! accidentally satisfy descriptor shape.

use ed25519_dalek::VerifyingKey;

use itertools::Itertools;

use crate::blob::encodings::utf8string::UTF8String;
use crate::id::Id;
use crate::inline::encodings::genid::GenId;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding, IntoInline, RawInline};
use crate::metadata;
use crate::prelude::{entity, find, pattern};
use crate::query::TriblePattern;
use crate::trible::{Fragment, TribleSet};

// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use super::reach;
use super::records::{
    collection_authority, collection_name, collection_reach, collection_recipe,
    collection_representation, collection_source, CollectionHandle, RecordDecodeError,
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
///
/// `reach` is a fragment rather than a flag, and it is required rather than
/// defaulted. Required, because reach used to be a separate signed grant that
/// production code never minted, so the normal outcome of publishing was that
/// nothing replicated and nothing complained; an argument cannot be
/// forgotten. A fragment, because what it exports is what gets declared and
/// what it carries rides along into the same blob -- so
/// [`reach::private`](crate::collection::reach::private) exports nothing and
/// writes nothing, and a future law with arguments needs no change to this
/// signature to state them.
pub fn naming(
    name: &str,
    authority: VerifyingKey,
    representation: Id,
    recipe: Id,
    reach: Fragment,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.to_owned(),
        collection_authority: authority,
        collection_representation: representation,
        collection_recipe: recipe,
        collection_reach*: reach,
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
    let descriptor = entity(facts)?;
    exactly_one(
        find!(
            (v: Id?),
            pattern!(facts, [{ descriptor @ collection_representation: ?v }])
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
    let descriptor = entity(facts)?;
    exactly_one(
        find!((v: Id?), pattern!(facts, [{ descriptor @ collection_recipe: ?v }])).map(|(v,)| v),
        "collection_recipe",
    )?
    .map_err(|_| RecordDecodeError::InvalidId("collection_recipe"))
}

/// The collection this one derives from, if it derives from one.
///
/// A root has no source and answers `None`; that is not a failure, it is what
/// being a root means.
pub fn source(facts: &TribleSet) -> Result<Option<CollectionHandle>, RecordDecodeError> {
    let descriptor = entity(facts)?;
    at_most_one(
        find!(
            (v: CollectionHandle),
            pattern!(facts, [{ descriptor @ collection_source: ?v }])
        )
        .map(|(v,)| v),
        "collection_source",
    )
}

/// Handle of the UTF-8 name carried by a root collection.
///
/// A derived collection has no name of its own and answers `None`: its anchor
/// is its source, and its name is whatever that source is called.
pub fn name(facts: &TribleSet) -> Result<Option<Inline<Handle<UTF8String>>>, RecordDecodeError> {
    let descriptor = entity(facts)?;
    at_most_one(
        find!(
            (v: Inline<Handle<UTF8String>>),
            pattern!(facts, [{ descriptor @ collection_name: ?v }])
        )
        .map(|(v,)| v),
        "collection_name",
    )
}

/// External capability trust root declared by this descriptor.
///
/// Exactly one authority row must occur in the complete descriptor archive,
/// and it must hang from the one tagged descriptor entity. Looking up the row
/// globally before checking its subject makes a smuggled authority on an
/// embedded description fail closed instead of being ignored. A derived
/// descriptor names its authority directly; source walking never supplies it.
pub fn authority(facts: &TribleSet) -> Result<VerifyingKey, RecordDecodeError> {
    let raw: Inline<crate::inline::encodings::ed25519::ED25519PublicKey> =
        exactly_one_descriptor_inline(facts, &collection_authority, "collection_authority")?;
    raw.try_from_inline::<VerifyingKey>()
        .map_err(|_| RecordDecodeError::InvalidId("collection_authority"))
}

/// Look up one recipe argument by attribute.
///
/// The attribute is a runtime id because the recipe that minted it is the only
/// thing that knows what it means; this reads the raw bytes and hands them
/// back for that recipe to interpret.
pub fn argument(facts: &TribleSet, attribute: Id) -> Result<Option<RawInline>, RecordDecodeError> {
    let descriptor: Inline<GenId> = entity(facts)?.to_inline();
    let attribute: Inline<GenId> = attribute.to_inline();
    at_most_one(
        find!(
            (v: Inline<GenId>),
            facts.pattern::<GenId>(descriptor, attribute, v)
        )
        .map(|(v,)| v.raw),
        "recipe argument",
    )
}

/// Decode one required single-valued field and require that it belongs to the
/// exact tagged descriptor entity.
fn exactly_one_descriptor_inline<S: InlineEncoding>(
    facts: &TribleSet,
    attribute: &crate::attribute::Attribute<S>,
    field: &'static str,
) -> Result<Inline<S>, RecordDecodeError> {
    let descriptor = entity(facts)?;
    let fact = exactly_one(
        facts.iter().filter(|fact| fact.a() == &attribute.id()),
        field,
    )?;
    if fact.e() != &descriptor {
        return Err(RecordDecodeError::FieldOnWrongEntity(field));
    }
    Ok(*fact.v::<S>())
}

/// `Itertools::exactly_one`, saying which field the rows came from.
///
/// The question is the same one the rest of the crate asks by that name; only
/// the answer differs, because a decoder has to name the attribute it was
/// reading. `ExactlyOneError` carries the leftover iterator, which is empty
/// exactly when there was no first row at all -- that is how the two failures
/// are told apart here.
fn exactly_one<T>(
    rows: impl Iterator<Item = T>,
    field: &'static str,
) -> Result<T, RecordDecodeError> {
    rows.exactly_one().map_err(|mut leftover| {
        if leftover.next().is_some() {
            RecordDecodeError::RepeatedField(field)
        } else {
            RecordDecodeError::MissingField(field)
        }
    })
}

/// Decode an optional single-valued field without accepting an arbitrary
/// first match from malformed input.
fn at_most_one<T>(
    mut rows: impl Iterator<Item = T>,
    field: &'static str,
) -> Result<Option<T>, RecordDecodeError> {
    let Some(first) = rows.next() else {
        return Ok(None);
    };
    if rows.next().is_some() {
        return Err(RecordDecodeError::RepeatedField(field));
    }
    Ok(Some(first))
}

/// A root descriptor under one fixed test authority, named by `name`.
///
/// Tests overwhelmingly want "some collection, distinct from that other one".
/// Spelling that as a name under a fixed authority keeps the distinction
/// readable in the test itself.
#[cfg(test)]
pub(crate) fn named_for_tests(name: &str, representation: Id, recipe: Id) -> Fragment {
    let root = ed25519_dalek::SigningKey::from_bytes(&[0xAA; 32]).verifying_key();
    naming(name, root, representation, recipe, reach::private())
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

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::blob::encodings::utf8string::UTF8String;
    use crate::collection::records::{collection_authority, collection_source};
    use crate::collection::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1;
    use crate::inline::encodings::ed25519::ED25519PublicKey;
    use crate::metadata;
    use crate::repo::{BlobStore, BlobStoreGet};
    use crate::trible::Trible;

    use anybytes::View;

    fn team_key(byte: u8) -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[byte; 32]).verifying_key()
    }

    fn root(name: &str, authority: ed25519_dalek::VerifyingKey) -> Fragment {
        naming(
            name,
            authority,
            <SimpleArchive as crate::metadata::MetaDescribe>::id(),
            TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        )
    }

    fn derived(source_of: &Fragment, authority: ed25519_dalek::VerifyingKey) -> Fragment {
        crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
            collection_source: identity_for_tests(source_of),
            collection_authority: authority,
        }
    }

    #[test]
    fn roots_carry_a_mandatory_local_authority() {
        let trust_root = team_key(4);
        let fragment = root("ledger", trust_root);

        assert_eq!(authority(fragment.facts()), Ok(trust_root));
    }

    #[test]
    fn derived_authority_is_local_and_never_inherited() {
        let root_authority = team_key(5);
        let derived_authority = team_key(6);
        let base = root("ledger", root_authority);
        let governed = derived(&base, derived_authority);

        assert_eq!(authority(base.facts()), Ok(root_authority));
        assert_eq!(authority(governed.facts()), Ok(derived_authority));
    }

    #[test]
    fn authority_participates_in_descriptor_identity() {
        let first = root("ledger", team_key(8));
        let second = root("ledger", team_key(9));

        assert_ne!(identity_for_tests(&first), identity_for_tests(&second));
        assert_eq!(first.facts().len(), second.facts().len());
    }

    #[test]
    fn name_is_unbounded_utf8_and_its_attachment_stays_with_the_fragment() {
        let expected = "a root collection name deliberately longer than thirty-two bytes 🦊";
        let fragment = root(expected, team_key(10));
        let handle = name(fragment.facts())
            .expect("valid descriptor")
            .expect("root descriptor has a name");

        let mut blobs = fragment.blobs().clone();
        let reader = blobs.reader().expect("memory blob reader");
        let actual: View<str> = reader
            .get::<View<str>, UTF8String>(handle)
            .expect("the name bytes travel with the descriptor fragment");
        assert_eq!(&*actual, expected);
    }

    #[test]
    fn duplicate_optional_name_is_rejected() {
        let mut fragment = root("ledger", team_key(10));
        let descriptor = fragment.root().expect("descriptor root");
        let second = fragment.put::<UTF8String, _>("another name".to_owned());
        fragment
            .facts_mut()
            .insert(&Trible::force(&descriptor, &collection_name.id(), &second));

        assert_eq!(
            name(fragment.facts()),
            Err(RecordDecodeError::RepeatedField("collection_name"))
        );
    }

    #[test]
    fn duplicate_optional_source_is_rejected() {
        let base = root("ledger", team_key(10));
        let mut fragment = derived(&base, team_key(11));
        let descriptor = fragment.root().expect("descriptor root");
        let second = Inline::<Handle<SimpleArchive>>::new([0x42; 32]);
        fragment.facts_mut().insert(&Trible::force(
            &descriptor,
            &collection_source.id(),
            &second,
        ));

        assert_eq!(
            source(fragment.facts()),
            Err(RecordDecodeError::RepeatedField("collection_source"))
        );
    }

    #[test]
    fn missing_authority_is_rejected() {
        let fragment = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
        };
        assert_eq!(
            authority(fragment.facts()),
            Err(RecordDecodeError::MissingField("collection_authority"))
        );
    }

    #[test]
    fn duplicate_authority_is_rejected() {
        let mut fragment = root("ledger", team_key(11));
        let descriptor = fragment.root().expect("descriptor root");
        let second: Inline<ED25519PublicKey> = team_key(12).to_inline();
        fragment.facts_mut().insert(&Trible::force(
            &descriptor,
            &collection_authority.id(),
            &second,
        ));

        assert_eq!(
            authority(fragment.facts()),
            Err(RecordDecodeError::RepeatedField("collection_authority"))
        );
    }

    #[test]
    fn malformed_authority_is_rejected() {
        let mut fragment = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
        };
        let descriptor = fragment.root().expect("descriptor root");
        let mut raw = [0_u8; 32];
        raw[0] = 2;
        let malformed = Inline::<ED25519PublicKey>::new(raw);
        fragment.facts_mut().insert(&Trible::force(
            &descriptor,
            &collection_authority.id(),
            &malformed,
        ));

        assert_eq!(
            authority(fragment.facts()),
            Err(RecordDecodeError::InvalidId("collection_authority"))
        );
    }

    #[test]
    fn off_entity_authority_is_rejected() {
        let mut fragment = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
        };
        let other = crate::id::id_hex!("13131313131313131313131313131313");
        let key: Inline<ED25519PublicKey> = team_key(13).to_inline();
        fragment
            .facts_mut()
            .insert(&Trible::force(&other, &collection_authority.id(), &key));

        assert_eq!(
            authority(fragment.facts()),
            Err(RecordDecodeError::FieldOnWrongEntity(
                "collection_authority"
            ))
        );
    }

    /// Declaring reach is a rename, which is the entire point.
    ///
    /// The public handle is pinned for the same reason the absent one above
    /// is, and against the same kind of witness: it was captured at commit
    /// 5b32ca5d, when reach was a two-variant Rust enum whose `declared()`
    /// fed an optional attribute. Stating reach as a fragment spread into the
    /// same attribute has to write the same one row, and this is what says so
    /// -- a builder that agreed with the old one only about *absence* would
    /// still have quietly renamed every collection that travels.
    #[test]
    fn declaring_reach_makes_a_different_collection() {
        let team = team_key(9);
        let private =
            crate::collection::simplearchive_union::descriptor("ledger", team, reach::private());
        let public =
            crate::collection::simplearchive_union::descriptor("ledger", team, reach::public());

        assert_ne!(identity_for_tests(&private), identity_for_tests(&public));
        assert_eq!(private.facts().len() + 1, public.facts().len());
    }

    /// A descriptor answers whether it travels, and silence is a refusal.
    #[test]
    fn reach_is_read_from_the_descriptor_and_absence_refuses() {
        let team = team_key(10);

        let private =
            crate::collection::simplearchive_union::descriptor("ledger", team, reach::private());
        assert_eq!(reach::declared(private.facts()), None);
        assert!(!reach::travels(private.facts()));

        let public =
            crate::collection::simplearchive_union::descriptor("ledger", team, reach::public());
        assert_eq!(reach::declared(public.facts()), Some(reach::PUBLIC));
        assert!(reach::travels(public.facts()));
    }

    /// A reach law this binary does not implement is a refusal, not a guess.
    ///
    /// This is the property a boolean could not have had. A future mode --
    /// some subset of a team -- reaching an older reader must not be read as
    /// "public" merely because it is not "absent".
    #[test]
    fn an_unknown_reach_law_does_not_travel() {
        let unknown = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
            collection_reach: crate::id::id_hex!("44444444444444444444444444444444"),
        };
        assert_eq!(
            reach::declared(unknown.facts()),
            Some(crate::id::id_hex!("44444444444444444444444444444444"))
        );
        assert!(!reach::travels(unknown.facts()));
    }

    /// Two declarations are not a majority vote.
    ///
    /// A descriptor asserting both `reach::PUBLIC` and something else has not
    /// stated a reach, and the tie is broken closed rather than by picking the
    /// permissive row.
    #[test]
    fn a_descriptor_declaring_two_reaches_declares_none() {
        let e =
            crate::id::ExclusiveId::force(crate::id::id_hex!("55555555555555555555555555555555"));
        let mut facts = TribleSet::new();
        facts += crate::prelude::entity! { &e @
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
            collection_reach: reach::PUBLIC,
        }
        .into_facts();
        assert!(reach::travels(&facts));

        facts += crate::prelude::entity! { &e @
            collection_reach: crate::id::id_hex!("44444444444444444444444444444444"),
        }
        .into_facts();
        assert_eq!(reach::declared(&facts), None);
        assert!(!reach::travels(&facts));
    }

    /// A derived collection declares its own reach and inherits nothing.
    ///
    /// Both directions matter. A public index over a private source would
    /// leak the source's shape if reach were inherited downward, and a
    /// deliberately published aggregate over private inputs would be
    /// impossible if reach were inherited upward. Authority follows the same
    /// local-descriptor rule rather than walking the source chain.
    #[test]
    fn a_derived_collection_declares_reach_independently_of_its_source() {
        let team = team_key(11);

        let private_root =
            crate::collection::simplearchive_union::descriptor("ledger", team, reach::private());
        let private_root_handle = identity_for_tests(&private_root);

        // A public derivation of a private source.
        let public_index = crate::collection::succinctarchive_union::descriptor(
            private_root_handle,
            team,
            reach::public(),
        );
        assert!(!reach::travels(private_root.facts()));
        assert!(reach::travels(public_index.facts()));

        let public_root =
            crate::collection::simplearchive_union::descriptor("ledger", team, reach::public());
        let public_root_handle = identity_for_tests(&public_root);

        // And a private derivation of a public source.
        let private_index = crate::collection::succinctarchive_union::descriptor(
            public_root_handle,
            team,
            reach::private(),
        );
        assert!(reach::travels(public_root.facts()));
        assert!(!reach::travels(private_index.facts()));

        // Reading reach never walks `collection_source`, so an absent source
        // descriptor cannot change the answer.
        let orphan = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
            collection_source: identity_for_tests(&public_root),
        };
        assert!(!reach::travels(orphan.facts()));
    }
}
