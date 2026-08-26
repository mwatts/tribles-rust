use triblespace_core::patch::{Blake3Merkle, Entry, IdentitySchema, PATCH};

#[test]
fn public_merkle_view_is_value_opaque_and_cursor_bounded() {
    struct SecretValue(u8);

    let left = Entry::<4, SecretValue, Blake3Merkle>::with_value(&[1, 2, 3, 4], SecretValue(4));
    let right = Entry::<4, SecretValue, Blake3Merkle>::with_value(&[1, 2, 9, 0], SecretValue(9));
    let mut patch = PATCH::<4, IdentitySchema, SecretValue, Blake3Merkle>::new();
    patch.insert(&right);
    patch.insert(&left);

    let node = patch.merkle_node(&[1, 2]).expect("shared prefix");
    assert_eq!(node.prefix(), &[1, 2]);
    assert_eq!(node.leaf_count(), 2);
    assert_eq!(
        node.children().map(|(edge, _)| edge).collect::<Vec<_>>(),
        [3, 9]
    );
    assert_eq!(
        node.items_after(Some(&[1, 2, 3, 4]), 1).collect::<Vec<_>>(),
        [[1, 2, 9, 0]]
    );
    assert_eq!(left.value().0, 4);
    assert_eq!(right.value().0, 9);
}
