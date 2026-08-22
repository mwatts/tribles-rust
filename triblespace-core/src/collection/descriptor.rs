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

use std::fmt;

use ed25519_dalek::VerifyingKey;

use itertools::Itertools;

use crate::id::Id;
use crate::inline::encodings::genid::GenId;
use crate::inline::encodings::shortstring::ShortString;
use crate::inline::{Inline, IntoInline, RawInline};
use crate::metadata;
use crate::prelude::{entity, find, pattern};
use crate::query::TriblePattern;
use crate::temp;
use crate::trible::{Fragment, TribleSet};

// Reach arrives here as a builder argument; only the tests name a
// particular one.
#[cfg(test)]
use super::reach;
use super::records::{
    collection_name, collection_reach, collection_recipe, collection_representation,
    collection_source, collection_team, CollectionHandle, CollectionName, RecordDecodeError,
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
    name: &CollectionName,
    team: VerifyingKey,
    representation: Id,
    recipe: Id,
    reach: Fragment,
) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.as_str(),
        collection_team: team,
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

/// How deep a team-root walk will follow [`collection_source`].
///
/// Not a safety bound against cycles -- see [`team_root`], which explains why
/// there cannot be one -- but against a chain long enough that walking it is
/// itself the attack. Real derivation pipelines stack a handful of
/// representations deep.
pub const MAX_DERIVATION_DEPTH: usize = 64;

/// Why a collection's team root could not be established.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TeamRootError {
    /// A descriptor on the chain is not available to the lookup.
    ///
    /// This is not a verdict about the collection. The team root is simply not
    /// knowable yet, exactly as an absent blob makes a claim pending rather
    /// than invalid.
    NotResident(CollectionHandle),
    /// A descriptor decoded, but the field the walk needed did not.
    Invalid {
        /// Descriptor holding the unreadable field.
        collection: CollectionHandle,
        /// What went wrong reading it.
        source: RecordDecodeError,
    },
    /// The chain is longer than [`MAX_DERIVATION_DEPTH`].
    TooDeep {
        /// Descriptor the walk gave up on.
        collection: CollectionHandle,
    },
    /// The chain reached a root that anchors to no team.
    ///
    /// A root with neither a source nor a team has no authority to inherit, so
    /// there is no owner to report for anything derived from it.
    NoTeam(CollectionHandle),
}

impl fmt::Display for TeamRootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotResident(collection) => write!(
                f,
                "collection descriptor {} is not resident",
                hex::encode_upper(collection.raw),
            ),
            Self::Invalid { collection, source } => write!(
                f,
                "collection descriptor {} is invalid: {source}",
                hex::encode_upper(collection.raw),
            ),
            Self::TooDeep { collection } => write!(
                f,
                "derivation chain through {} is deeper than {MAX_DERIVATION_DEPTH}",
                hex::encode_upper(collection.raw),
            ),
            Self::NoTeam(collection) => write!(
                f,
                "root collection descriptor {} names no team",
                hex::encode_upper(collection.raw),
            ),
        }
    }
}

impl std::error::Error for TeamRootError {}

/// Walk [`collection_source`] to the root and read the team it anchors to.
///
/// A root carries its own [`collection_team`]; a derived collection carries
/// only its source and inherits the team transitively. Asking who owns a
/// derived collection therefore means walking to that root, which is what
/// `trible pile collection show` does to report a derivation's team. `lookup`
/// answers with a descriptor's facts, or `None` when the descriptor is not
/// available locally.
///
/// **The walk terminates by construction, and no cycle is representable.** A
/// descriptor is named by the hash of its own bytes, and `collection_source`
/// holds such a name, so a descriptor can only ever point at bytes that
/// already existed when it was written. A cycle would require a descriptor
/// containing its own hash -- a preimage. This is not a rule anyone enforces;
/// content addressing makes the derivation graph acyclic in the same way it
/// makes a Merkle tree acyclic. [`MAX_DERIVATION_DEPTH`] therefore guards
/// against absurd length, not against looping.
///
/// The cost is one small archive fetch per hop, and a real pipeline is one to
/// three hops deep. The case worth planning for is not depth but residency: a
/// missing source descriptor makes the team root *unknown*, which a caller
/// should treat as pending rather than as a rejection.
pub fn team_root(
    collection: CollectionHandle,
    mut lookup: impl FnMut(CollectionHandle) -> Option<TribleSet>,
) -> Result<(CollectionHandle, VerifyingKey), TeamRootError> {
    let mut current = collection;
    for _ in 0..MAX_DERIVATION_DEPTH {
        let facts = lookup(current).ok_or(TeamRootError::NotResident(current))?;
        if let Some(team) = team(&facts) {
            let team = team.map_err(|source| TeamRootError::Invalid {
                collection: current,
                source,
            })?;
            return Ok((current, team));
        }
        match source(&facts) {
            Some(next) => current = next,
            None => return Err(TeamRootError::NoTeam(current)),
        }
    }
    Err(TeamRootError::TooDeep {
        collection: current,
    })
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
        reach::private(),
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::blob::encodings::simplearchive::SimpleArchive;
    use crate::collection::records::collection_source;
    use crate::collection::simplearchive_union::TRIBLE_SET_UNION_RECIPE_V1;
    use crate::metadata;

    fn store(descriptors: &[Fragment]) -> BTreeMap<CollectionHandle, TribleSet> {
        descriptors
            .iter()
            .map(|fragment| (identity_for_tests(fragment), fragment.facts().clone()))
            .collect()
    }

    fn team_key(byte: u8) -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[byte; 32]).verifying_key()
    }

    fn root(name: &str, team: ed25519_dalek::VerifyingKey) -> Fragment {
        naming(
            &CollectionName::new(name).unwrap(),
            team,
            <SimpleArchive as crate::metadata::MetaDescribe>::id(),
            TRIBLE_SET_UNION_RECIPE_V1,
            reach::private(),
        )
    }

    fn derived(source_of: &Fragment) -> Fragment {
        crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
            collection_source: identity_for_tests(source_of),
        }
    }

    #[test]
    fn a_root_is_its_own_team_root() {
        let team = team_key(3);
        let fragment = root("ledger", team);
        let resident = store(&[fragment.clone()]);
        assert_eq!(
            team_root(identity_for_tests(&fragment), |handle| resident
                .get(&handle)
                .cloned()),
            Ok((identity_for_tests(&fragment), team))
        );
    }

    #[test]
    fn a_derived_collection_inherits_its_team_through_the_source_chain() {
        let team = team_key(4);
        let base = root("ledger", team);
        let mid = derived(&base);
        let top = derived(&mid);
        let resident = store(&[base.clone(), mid, top.clone()]);

        // Three hops, one small archive fetch each, and the answer is the
        // root's team rather than anything the derived descriptors say.
        assert_eq!(
            team_root(identity_for_tests(&top), |handle| resident
                .get(&handle)
                .cloned()),
            Ok((identity_for_tests(&base), team))
        );
    }

    #[test]
    fn a_missing_source_descriptor_makes_the_team_root_unknown_not_wrong() {
        let base = root("ledger", team_key(5));
        let mid = derived(&base);
        // The root is absent: its team is not knowable, which is a different
        // answer from "this collection has no team".
        let resident = store(&[mid.clone()]);
        assert_eq!(
            team_root(identity_for_tests(&mid), |handle| resident
                .get(&handle)
                .cloned()),
            Err(TeamRootError::NotResident(identity_for_tests(&base)))
        );
    }

    #[test]
    fn a_chain_that_reaches_no_team_says_so() {
        let orphan = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
        };
        let resident = store(&[orphan.clone()]);
        assert_eq!(
            team_root(identity_for_tests(&orphan), |handle| resident
                .get(&handle)
                .cloned()),
            Err(TeamRootError::NoTeam(identity_for_tests(&orphan)))
        );
    }

    #[test]
    fn an_absurdly_long_chain_is_abandoned_rather_than_walked() {
        // A cycle is not representable -- a descriptor is named by the hash of
        // its own bytes, so pointing at itself would need a preimage -- but a
        // very long chain is, and the walk gives up rather than paying for it.
        let team = team_key(6);
        let mut fragments = vec![root("ledger", team)];
        for _ in 0..MAX_DERIVATION_DEPTH + 2 {
            let next = derived(fragments.last().unwrap());
            fragments.push(next);
        }
        let resident = store(&fragments);
        let deepest = identity_for_tests(fragments.last().unwrap());
        assert!(matches!(
            team_root(deepest, |handle| resident.get(&handle).cloned()),
            Err(TeamRootError::TooDeep { .. })
        ));

        // The same chain, walked from inside the bound, still resolves.
        let shallow = identity_for_tests(&fragments[3]);
        assert_eq!(
            team_root(shallow, |handle| resident.get(&handle).cloned()),
            Ok((identity_for_tests(&fragments[0]), team))
        );
    }

    /// A descriptor that declares no reach hashes exactly as it did before the
    /// attribute existed.
    ///
    /// These four handles were captured from this crate at commit e72b20db,
    /// the last one built before `collection_reach` was declared. They are
    /// pinned rather than recomputed because recomputing them would prove
    /// nothing: the question is whether today's builders agree with a version
    /// of the code that is no longer here to ask.
    ///
    /// This is the whole reason the attribute is optional. `self.pile` holds
    /// 69 collections of which only 23 are named, the other 46 being the same
    /// data under earlier descriptor shapes; a mandatory field would have made
    /// it 138.
    #[test]
    fn a_descriptor_without_reach_keeps_the_identity_it_had_before_reach_existed() {
        let team = team_key(0xAA);
        let name = CollectionName::new("ledger").unwrap();

        let bare = naming(
            &name,
            team,
            crate::id::id_hex!("11111111111111111111111111111111"),
            crate::id::id_hex!("22222222222222222222222222222222"),
            reach::private(),
        );
        assert_eq!(
            hex::encode_upper(identity_for_tests(&bare).raw),
            "9D413F35934206B916104EE38F03E40E0D1F3AEE2332E00F61FB23B12B422F15"
        );

        let root = crate::collection::simplearchive_union::descriptor(&name, team, reach::private());
        let root_handle = identity_for_tests(&root);
        assert_eq!(
            hex::encode_upper(root_handle.raw),
            "8D492ED32C9A96F6F1B6EED0ED565376F8A9CA6C7073A155C8F84783747B465E"
        );

        let succinct =
            crate::collection::succinctarchive_union::descriptor(root_handle, reach::private());
        assert_eq!(
            hex::encode_upper(identity_for_tests(&succinct).raw),
            "C1E12A2FB1CA64CC38D138039F13C9F99DC9AE1A9F73FC610D4201E0FBB84052"
        );

        let observed = crate::collection::observed_union::descriptor(
            root_handle,
            crate::id::id_hex!("33333333333333333333333333333333"),
            reach::private(),
        );
        assert_eq!(
            hex::encode_upper(identity_for_tests(&observed).raw),
            "CCE515D8FE869D2709DC328894FDEBEDBE287FC1E69640C0AE2F71424B721F33"
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
        let name = CollectionName::new("ledger").unwrap();
        let private =
            crate::collection::simplearchive_union::descriptor(&name, team, reach::private());
        let public =
            crate::collection::simplearchive_union::descriptor(&name, team, reach::public());

        assert_ne!(identity_for_tests(&private), identity_for_tests(&public));
        assert_eq!(private.facts().len() + 1, public.facts().len());
        assert_eq!(
            hex::encode_upper(identity_for_tests(&public).raw),
            "719258222ECBADFF790DCC7EE1A1ABA1C7F7A4E35641C0D653D954E66ED78ED0"
        );
    }

    /// A descriptor answers whether it travels, and silence is a refusal.
    #[test]
    fn reach_is_read_from_the_descriptor_and_absence_refuses() {
        let team = team_key(10);
        let name = CollectionName::new("ledger").unwrap();

        let private =
            crate::collection::simplearchive_union::descriptor(&name, team, reach::private());
        assert_eq!(reach::declared(private.facts()), None);
        assert!(!reach::travels(private.facts()));

        let public = crate::collection::simplearchive_union::descriptor(&name, team, reach::public());
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
        let e = crate::id::ExclusiveId::force(crate::id::id_hex!(
            "55555555555555555555555555555555"
        ));
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
    /// impossible if reach were inherited upward. Contrast
    /// [`team_root`], which *does* walk, because ownership genuinely is
    /// inherited.
    #[test]
    fn a_derived_collection_declares_reach_independently_of_its_source() {
        let team = team_key(11);
        let name = CollectionName::new("ledger").unwrap();

        let private_root =
            crate::collection::simplearchive_union::descriptor(&name, team, reach::private());
        let private_root_handle = identity_for_tests(&private_root);

        // A public derivation of a private source.
        let public_index =
            crate::collection::succinctarchive_union::descriptor(private_root_handle, reach::public());
        assert!(!reach::travels(private_root.facts()));
        assert!(reach::travels(public_index.facts()));

        let public_root =
            crate::collection::simplearchive_union::descriptor(&name, team, reach::public());
        let public_root_handle = identity_for_tests(&public_root);

        // And a private derivation of a public source.
        let private_index = crate::collection::succinctarchive_union::descriptor(
            public_root_handle,
            reach::private(),
        );
        assert!(reach::travels(public_root.facts()));
        assert!(!reach::travels(private_index.facts()));

        // Reading reach never walks `collection_source`, so an absent source
        // descriptor cannot change the answer -- unlike `team_root`, which
        // would report the chain unresolvable here.
        let orphan = crate::prelude::entity! {
            metadata::tag: super::KIND_COLLECTION_DESCRIPTOR,
            collection_source: identity_for_tests(&public_root),
        };
        assert!(!reach::travels(orphan.facts()));
    }
}
