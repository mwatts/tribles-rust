//! Four-channel [`Fragment`] semantics: exports, facts, metafacts, and one
//! content-addressed blob store shared by both fact sets.

use anybytes::View;
use triblespace_core::blob::encodings::utf8string::UTF8String;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::prelude::inlineencodings::GenId;
use triblespace_core::prelude::*;

attributes! {
    /// Data-side text whose declaration documentation lives in metafacts.
    fm_required: Handle<UTF8String>;
    /// An optional data-side text field.
    fm_optional: Handle<UTF8String>;
    /// A repeated data-side text field.
    fm_repeated: Handle<UTF8String>;
    /// Text carried by a child fragment spread into its parent.
    fm_child_text: Handle<UTF8String>;
    /// Links from a parent to the roots exported by a spread fragment.
    fm_children: GenId;
}

fn description_count(fragment: &Fragment, attribute: Id) -> usize {
    find!(
        (usage: Id),
        pattern!(
            fragment.metafacts(),
            [{ ?usage @ triblespace_core::metadata::attribute: attribute }]
        )
    )
    .count()
}

fn describes(fragment: &Fragment, attribute: Id) -> bool {
    description_count(fragment, attribute) > 0
}

#[test]
fn entity_carries_descriptions_for_each_attribute_that_emits_a_fact() {
    let optional = Some("optional payload");
    let fragment = entity! {
        fm_required: "required payload",
        fm_optional?: optional,
        fm_repeated*: ["first repeated payload", "second repeated payload"],
    };

    assert!(describes(&fragment, fm_required.id()));
    assert!(describes(&fragment, fm_optional.id()));
    assert!(describes(&fragment, fm_repeated.id()));
}

#[test]
fn absent_optional_and_empty_repeated_attributes_do_not_add_descriptions() {
    let absent: Option<&str> = None;
    let empty: Vec<&str> = Vec::new();
    let fragment = entity! {
        fm_required: "the one present field",
        fm_optional?: absent,
        fm_repeated*: empty,
    };

    assert!(describes(&fragment, fm_required.id()));
    assert!(!describes(&fragment, fm_optional.id()));
    assert!(!describes(&fragment, fm_repeated.id()));
}

#[test]
fn fragment_merge_deduplicates_descriptions_and_unions_the_shared_store() {
    let mut left = entity! { fm_required: "left payload" };
    let right = entity! { fm_required: "right payload" };

    assert_eq!(left.metafacts(), right.metafacts());
    let description = left.metafacts().clone();
    let separate_blob_count = left.blobs().len() + right.blobs().len();

    left += right;

    assert_eq!(left.metafacts(), &description);
    assert_eq!(description_count(&left, fm_required.id()), 1);
    assert!(
        left.blobs().len() < separate_blob_count,
        "metadata blobs common to both fragments should deduplicate"
    );

    let handles: Vec<Inline<Handle<UTF8String>>> = find!(
        (value: Inline<Handle<UTF8String>>),
        pattern!(&left, [{ fm_required: ?value }])
    )
    .map(|(value,)| value)
    .collect();
    assert_eq!(handles.len(), 2);

    let mut store = left.blobs().clone();
    let reader = store.reader().expect("shared blob-store reader");
    let mut values: Vec<String> = handles
        .into_iter()
        .map(|handle| {
            let value: View<str> = reader
                .get::<View<str>, UTF8String>(handle)
                .expect("each merged data handle resolves");
            value.to_string()
        })
        .collect();
    values.sort();
    assert_eq!(values, ["left payload", "right payload"]);
}

#[test]
fn spreading_a_fragment_preserves_the_child_metafacts() {
    let child = entity! { fm_child_text: "nested payload" };
    let child_id = child.root().expect("child is rooted");

    let parent = entity! { fm_children*: child };

    assert!(describes(&parent, fm_children.id()));
    assert!(describes(&parent, fm_child_text.id()));

    let nested: Vec<Inline<Handle<UTF8String>>> = find!(
        (value: Inline<Handle<UTF8String>>),
        pattern!(&parent, [{ child_id @ fm_child_text: ?value }])
    )
    .map(|(value,)| value)
    .collect();
    assert_eq!(nested.len(), 1);

    let mut store = parent.blobs().clone();
    let reader = store.reader().expect("parent blob-store reader");
    let value: View<str> = reader
        .get::<View<str>, UTF8String>(nested[0])
        .expect("the spread child blob resolves from the parent store");
    assert_eq!(&*value, "nested payload");
}

#[test]
fn describe_with_promotes_both_description_fact_sets_without_changing_content() {
    let mut content = entity! { fm_required: "content payload" };
    let description = entity! { fm_optional: "runtime description payload" };

    let root_before = content.root();
    let facts_before = content.facts().clone();
    let mut expected_metafacts = content.metafacts().clone();
    expected_metafacts += description.facts().clone();
    expected_metafacts += description.metafacts().clone();

    content.describe_with(description);

    assert_eq!(content.root(), root_before);
    assert_eq!(content.facts(), &facts_before);
    assert_eq!(content.metafacts(), &expected_metafacts);
}

#[test]
fn declaration_description_does_not_participate_in_entity_identity() {
    let declared = entity! { fm_required: "identity payload" };

    // `Attribute::named` has the same identity core as the bare-name arm of
    // `attributes!`, but it does not carry this declaration site's usage and
    // documentation record.
    let runtime_attribute = Attribute::<Handle<UTF8String>>::named("fm_required");
    assert_eq!(runtime_attribute.id(), fm_required.id());
    let minimally_described = entity! { runtime_attribute: "identity payload" };

    assert_eq!(declared.root(), minimally_described.root());
    assert_eq!(declared.facts(), minimally_described.facts());
    assert_ne!(declared.metafacts(), minimally_described.metafacts());
}

#[test]
fn derived_attribute_identity_fragment_carries_its_name_blob() {
    let attribute = fm_required.id();
    let fragment = fm_required.fragment();
    let name_handle = find!(
        (value: Inline<Handle<UTF8String>>),
        pattern!(fragment, [{ attribute @
            triblespace_core::metadata::name: ?value
        }])
    )
    .map(|(value,)| value)
    .next()
    .expect("the derived attribute identity has a name");

    let mut store = fragment.blobs().clone();
    let reader = store.reader().expect("attribute blob-store reader");
    let name: View<str> = reader
        .get::<View<str>, UTF8String>(name_handle)
        .expect("the identity fragment carries its own name bytes");
    assert_eq!(&*name, "fm_required");
}

#[test]
fn data_and_metadata_handles_resolve_from_the_same_blob_store() {
    let fragment = entity! { fm_required: "data-side payload" };
    let root = fragment.root().expect("entity is rooted");

    let data_handle = find!(
        (value: Inline<Handle<UTF8String>>),
        pattern!(&fragment, [{ root @ fm_required: ?value }])
    )
    .map(|(value,)| value)
    .next()
    .expect("data handle is present");

    let metadata_handle = find!(
        (value: Inline<Handle<UTF8String>>),
        pattern!(
            fragment.metafacts(),
            [{ _?usage @
                triblespace_core::metadata::attribute: fm_required.id(),
                triblespace_core::metadata::description: ?value
            }]
        )
    )
    .map(|(value,)| value)
    .next()
    .expect("declaration documentation handle is present in metafacts");

    let mut store = fragment.blobs().clone();
    let reader = store.reader().expect("shared blob-store reader");
    let data: View<str> = reader
        .get::<View<str>, UTF8String>(data_handle)
        .expect("data handle resolves from the fragment store");
    let documentation: View<str> = reader
        .get::<View<str>, UTF8String>(metadata_handle)
        .expect("metadata handle resolves from that same store");

    assert_eq!(&*data, "data-side payload");
    assert_eq!(
        &*documentation,
        "Data-side text whose declaration documentation lives in metafacts."
    );
}
