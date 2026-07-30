use triblespace_core::trible::{Trible, TribleSet};

#[derive(Copy, Clone, Default)]
struct Combination([u128; 2]);

impl Combination {
    fn singleton(index: usize) -> Self {
        let mut words = [0; 2];
        words[index / 128] = 1u128 << (index % 128);
        Self(words)
    }

    fn xor_assign(&mut self, other: Self) {
        self.0[0] ^= other.0[0];
        self.0[1] ^= other.0[1];
    }

    fn contains(self, index: usize) -> bool {
        self.0[index / 128] & (1u128 << (index % 128)) != 0
    }
}

fn dependency(vectors: &[u128]) -> Combination {
    let mut basis: [Option<(u128, Combination)>; 128] = [None; 128];

    'input: for (index, &input) in vectors.iter().enumerate() {
        assert_ne!(input, 0);
        let mut vector = input;
        let mut combination = Combination::singleton(index);

        loop {
            if vector == 0 {
                return combination;
            }
            let pivot = (u128::BITS - 1 - vector.leading_zeros()) as usize;
            if let Some((basis_vector, basis_combination)) = basis[pivot] {
                vector ^= basis_vector;
                combination.xor_assign(basis_combination);
            } else {
                basis[pivot] = Some((vector, combination));
                continue 'input;
            }
        }
    }

    panic!("more than 128 vectors must have a GF(2) dependency");
}

fn fixture_trible(nonce: u32) -> Trible {
    let mut raw = [0u8; 64];
    raw[..4].copy_from_slice(&nonce.to_be_bytes());
    raw[16] = 0xa5;
    raw[32..36].copy_from_slice(&nonce.to_le_bytes());
    Trible::force_raw(raw).expect("fixture entity and attribute are non-nil")
}

fn singleton_fingerprint(trible: &Trible) -> u128 {
    let mut set = TribleSet::new();
    set.insert(trible);
    set.fingerprint()
        .as_u128()
        .expect("a singleton is not the empty set")
}

#[test]
fn chosen_singleton_dependency_does_not_cross_the_public_fingerprint_boundary() {
    let candidates: Vec<_> = (1..=512)
        .filter_map(|nonce| {
            let trible = fixture_trible(nonce);
            let fingerprint = singleton_fingerprint(&trible);
            (fingerprint != 0).then_some((trible, fingerprint))
        })
        .take(129)
        .collect();
    assert_eq!(candidates.len(), 129);

    let vectors: Vec<_> = candidates
        .iter()
        .map(|(_, fingerprint)| *fingerprint)
        .collect();
    let relation = dependency(&vectors);
    let support: Vec<_> = (0..vectors.len())
        .filter(|&index| relation.contains(index))
        .collect();
    assert!(support.len() >= 2);
    assert_eq!(
        support.iter().fold(0, |xor, &index| xor ^ vectors[index]),
        0
    );

    let common = fixture_trible(10_000);
    let mut left = TribleSet::new();
    let mut right = TribleSet::new();
    left.insert(&common);
    right.insert(&common);
    for (position, &index) in support.iter().enumerate() {
        if position % 2 == 0 {
            left.insert(&candidates[index].0);
        } else {
            right.insert(&candidates[index].0);
        }
    }

    // Establish semantic inequality without trusting fingerprint-based
    // TribleSet::eq: this witness was placed only in the left partition.
    let left_only = &candidates[support[0]].0;
    assert!(left.contains(left_only));
    assert!(!right.contains(left_only));
    assert!(left.len() >= 2 && right.len() >= 2);

    // Before public blinding, the relation above is a relation among PATCH's
    // raw leaf vectors. Both Branch roots therefore collide here. With the
    // nonlinear export boundary, it is merely a relation among opaque tokens
    // and conveys no corresponding relation among the internal aggregates.
    assert_ne!(left.fingerprint(), right.fingerprint());
    assert_ne!(left, right);
}
