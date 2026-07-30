use triblespace::core::patch::{Entry, IdentitySchema, PATCH};

#[test]
fn borrowing_iter_handles_a_branch_at_one_byte_capacity() {
    let keys = [[0u8], [1u8]];
    let mut patch: PATCH<1, IdentitySchema> = PATCH::new();
    for key in &keys {
        patch.insert(&Entry::new(key));
    }

    assert_eq!(patch.branch_histogram()[0].0, 1);
    let mut iter = patch.iter();
    assert_eq!(iter.size_hint(), (2, Some(2)));
    let mut actual: Vec<_> = iter.by_ref().copied().collect();
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
    actual.sort_unstable();
    assert_eq!(actual, keys.to_vec());
}

#[test]
fn borrowing_iter_uses_all_key_len_branch_frames() {
    let keys = [[0u8, 0], [0, 1], [1, 0]];
    let mut patch: PATCH<2, IdentitySchema> = PATCH::new();
    for key in &keys {
        patch.insert(&Entry::new(key));
    }

    let histogram = patch.branch_histogram();
    assert_eq!(histogram[0].0, 1);
    assert_eq!(histogram[1].0, 1);
    let mut actual: Vec<_> = patch.iter().copied().collect();
    actual.sort_unstable();
    assert_eq!(actual, keys.to_vec());
}

#[test]
fn borrowing_iter_needs_no_stack_frame_for_a_zero_length_key() {
    let key = [];
    let mut patch: PATCH<0, IdentitySchema> = PATCH::new();
    {
        let mut iter = patch.iter();
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    patch.insert(&Entry::new(&key));
    let mut iter = patch.iter();
    assert_eq!(iter.len(), 1);
    assert_eq!(iter.size_hint(), (1, Some(1)));
    assert_eq!(iter.next(), Some(&key));
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}
