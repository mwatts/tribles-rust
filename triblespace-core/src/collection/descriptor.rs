//! Reading one collection descriptor.
//!
//! A descriptor is an ordinary [`TribleSet`]: the facts of one
//! [`entity!`](crate::macros::entity), stored as a
//! [`SimpleArchive`](crate::blob::encodings::simplearchive::SimpleArchive)
//! blob whose handle is the collection identity. The descriptor names its
//! encoding directly, and reading one is an ordinary query over ordinary
//! facts.
//!
//! A root carries [`collection_name`] while a derived collection carries
//! [`collection_source`] and one concrete [`collection_mapping`]
//! instance instead. Mapping parameters hang from that mapping entity, not
//! from the collection descriptor, so the conversion remains independently
//! identifiable and queryable. Both kinds carry exactly one local
//! [`collection_authority`]; authority is never inferred by walking the source
//! chain. Readers first locate the one tagged descriptor entity, then bind
//! every field lookup to that exact entity so embedded descriptions cannot
//! accidentally satisfy descriptor shape.

use ed25519_dalek::VerifyingKey;

use itertools::Itertools;

use crate::blob::encodings::simplearchive::SimpleArchive;
use crate::blob::encodings::utf8string::UTF8String;
use crate::blob::encodings::UnknownBlob;
use crate::blob::Blob;
use crate::id::Id;
use crate::inline::encodings::genid::GenId;
use crate::inline::encodings::hash::Handle;
use crate::inline::{Inline, InlineEncoding, IntoInline, RawInline};
use crate::metadata;
use crate::prelude::{entity, find, pattern};
use crate::query::TriblePattern;
use crate::repo::{BlobStorePut, SnapshotSource};
use crate::trible::{Fragment, TribleSet};

// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use super::reach;
use super::records::{
    collection_authority, collection_mapping, collection_name, collection_reach,
    collection_representation, collection_source, mapping_algorithm as mapping_algorithm_attribute,
    CollectionHandle, RecordDecodeError, KIND_COLLECTION_DESCRIPTOR, KIND_COLLECTION_MAPPING,
};

/// Retired `collection_recipe` attribute, minted with `trible genid` on
/// 2026-08-07. It remains only as a rejection marker: accepting an old recipe
/// descriptor as a recipe-free encoding descriptor would silently reinterpret
/// its identity and laws.
const OBSOLETE_COLLECTION_RECIPE: Id = crate::id::id_hex!("5D338C58D897B969BE1AE0956CCFE301");

/// Store one descriptor archive and every blob carried by its self-contained
/// Fragment, returning the canonical descriptor handle.
///
/// The descriptor identity covers only its fact archive. Names and embedded
/// self-descriptions may reference separate blobs, so publishing facts alone
/// would leave a descriptor whose shape validates but whose descriptions
/// cannot be read. Callers normally pass an [`OfferCapture`](crate::repo::OfferCapture)
/// so the complete closure is advertised at the following semantic record.
pub(crate) fn put_closure<S>(
    store: &mut S,
    descriptor: &Fragment,
) -> Result<CollectionHandle, S::PutError>
where
    S: BlobStorePut,
{
    let mut blobs = descriptor.blobs().clone();
    let mut embedded: Vec<Blob<UnknownBlob>> = blobs
        .snapshot()
        .expect("MemoryBlobStore::snapshot is infallible")
        .into_iter()
        .map(|(_, blob)| blob)
        .collect();
    embedded.sort_unstable_by_key(|blob| blob.get_handle().raw);
    for blob in embedded {
        store.put::<UnknownBlob, _>(blob)?;
    }
    store.put::<SimpleArchive, _>(descriptor.facts().clone())
}

/// Build a root descriptor that names its encoding without describing it.
///
/// A collection encoding normally writes its own descriptor as a visible
/// `entity!`; see
/// [`simplearchive_union::descriptor`](crate::collection::simplearchive_union::descriptor),
/// which additionally embeds the encoding's self-description so a stranger
/// holding the one blob can say what the collection is. This is the bare
/// generic form, for callers holding only ids.
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
    reach: Fragment,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.to_owned(),
        collection_authority: authority,
        collection_representation: representation,
        collection_reach*: reach,
    }
}

/// Build a derived descriptor around one concrete mapping instance.
///
/// The mapping Fragment is spread into the same descriptor archive. Its root
/// is linked from the descriptor and all algorithm descriptions, parameters,
/// and attachments therefore travel with the collection identity.
pub fn deriving(
    source: CollectionHandle,
    authority: VerifyingKey,
    representation: Id,
    mapping: Fragment,
    reach: Fragment,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_source: source,
        collection_authority: authority,
        collection_representation: representation,
        collection_mapping*: mapping,
        collection_reach*: reach,
    }
}

/// The entity the descriptor's own attributes hang off.
///
/// A descriptor archive holds more than one entity: the descriptor, plus the
/// embedded self-descriptions of its encoding and mapping. The
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

/// Concrete mapping instance carried by a derived collection.
///
/// Root collections answer `None`. A derived collection names exactly one
/// mapping entity. Canonical builders normally derive its id from its concrete
/// parameters, but readers accept the equivalent extrinsic-id substitution.
pub fn mapping(facts: &TribleSet) -> Result<Option<Id>, RecordDecodeError> {
    let descriptor = entity(facts)?;
    at_most_one(
        find!(
            (v: Id?),
            pattern!(facts, [{ descriptor @ collection_mapping: ?v }])
        )
        .map(|(v,)| v),
        "collection_mapping",
    )?
    .map(|value| value.map_err(|_| RecordDecodeError::InvalidId("collection_mapping")))
    .transpose()
}

/// Algorithm named by the concrete mapping instance, if this is derived.
pub fn mapping_algorithm(facts: &TribleSet) -> Result<Option<Id>, RecordDecodeError> {
    let Some(mapping) = mapping(facts)? else {
        return Ok(None);
    };
    let kind: Inline<GenId> = KIND_COLLECTION_MAPPING.to_inline();
    if !facts.iter().any(|fact| {
        fact.e() == &mapping && fact.a() == &metadata::tag.id() && fact.v::<GenId>() == &kind
    }) {
        return Err(RecordDecodeError::MissingField(
            "mapping metadata::tag KIND_COLLECTION_MAPPING",
        ));
    }
    exactly_one(
        find!(
            (v: Id?),
            pattern!(facts, [{ mapping @ mapping_algorithm_attribute: ?v }])
        )
        .map(|(v,)| v),
        "mapping_algorithm",
    )?
    .map(Some)
    .map_err(|_| RecordDecodeError::InvalidId("mapping_algorithm"))
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

/// Validate the representation-independent shape shared by every collection
/// descriptor and return its local authority.
///
/// A root is named and has no source mapping. A derived collection is unnamed
/// and carries both its source and one concrete mapping. Encoding-specific
/// context is deliberately left to [`CollectionEncoding::validate_descriptor`]
/// at the typed boundary.
pub fn validate(facts: &TribleSet) -> Result<VerifyingKey, RecordDecodeError> {
    let descriptor = entity(facts)?;
    if facts
        .iter()
        .any(|fact| fact.e() == &descriptor && fact.a() == &OBSOLETE_COLLECTION_RECIPE)
    {
        return Err(RecordDecodeError::ObsoleteField("collection_recipe"));
    }
    representation(facts)?;
    let name = name(facts)?;
    let source = source(facts)?;
    let mapping = mapping(facts)?;
    match (name, source, mapping) {
        (Some(_), None, None) => {}
        (None, Some(_), Some(_)) => {
            mapping_algorithm(facts)?;
        }
        (None, None, None) => {
            return Err(RecordDecodeError::MissingField(
                "collection_name or collection_source with collection_mapping",
            ));
        }
        _ => {
            return Err(RecordDecodeError::RepeatedField(
                "collection shape (root name or derived source/mapping)",
            ));
        }
    }
    authority(facts)
}

/// Look up one descriptor argument by attribute.
///
/// This remains useful for reach laws whose arguments belong to the
/// descriptor itself. Source-to-target parameters use [`mapping_argument`].
pub fn argument(facts: &TribleSet, attribute: Id) -> Result<Option<RawInline>, RecordDecodeError> {
    argument_on(facts, entity(facts)?, attribute, "descriptor argument")
}

/// Look up one concrete mapping parameter by attribute.
pub fn mapping_argument(
    facts: &TribleSet,
    attribute: Id,
) -> Result<Option<RawInline>, RecordDecodeError> {
    let Some(mapping) = mapping(facts)? else {
        return Ok(None);
    };
    argument_on(facts, mapping, attribute, "mapping argument")
}

fn argument_on(
    facts: &TribleSet,
    subject: Id,
    attribute: Id,
    field: &'static str,
) -> Result<Option<RawInline>, RecordDecodeError> {
    let subject: Inline<GenId> = subject.to_inline();
    let attribute: Inline<GenId> = attribute.to_inline();
    at_most_one(
        find!(
            (v: Inline<GenId>),
            facts.pattern::<GenId>(subject, attribute, v)
        )
        .map(|(v,)| v.raw),
        field,
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
pub(crate) fn named_for_tests(name: &str, representation: Id) -> Fragment {
    let root = ed25519_dalek::SigningKey::from_bytes(&[0xAA; 32]).verifying_key();
    naming(name, root, representation, reach::private())
}

/// Content identity of a descriptor, for tests that have no store.
///
/// The record algebra -- which payloads a cover admits, which merges compose,
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
    use crate::inline::encodings::ed25519::ED25519PublicKey;
    use crate::metadata;
    use crate::repo::BlobStoreGet;
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
    fn retired_recipe_descriptors_are_not_reinterpreted_as_encoding_descriptors() {
        let mut fragment = root("legacy", team_key(12));
        let descriptor = fragment.root().expect("descriptor root");
        let value: Inline<GenId> = crate::collection::records::KIND_COLLECTION_MAPPING.to_inline();
        fragment.facts_mut().insert(&Trible::force(
            &descriptor,
            &OBSOLETE_COLLECTION_RECIPE,
            &value,
        ));

        assert_eq!(
            validate(fragment.facts()),
            Err(RecordDecodeError::ObsoleteField("collection_recipe"))
        );
        assert!(matches!(
            crate::collection::Collection::<SimpleArchive>::from_descriptor(&fragment),
            Err(crate::collection::CollectionTypeError::Malformed(
                RecordDecodeError::ObsoleteField("collection_recipe")
            ))
        ));
    }

    #[test]
    fn name_is_unbounded_utf8_and_its_attachment_stays_with_the_fragment() {
        let expected = "a root collection name deliberately longer than thirty-two bytes 🦊";
        let fragment = root(expected, team_key(10));
        let handle = name(fragment.facts())
            .expect("valid descriptor")
            .expect("root descriptor has a name");

        let mut blobs = fragment.blobs().clone();
        let reader = blobs.snapshot().expect("memory blob snapshot");
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
