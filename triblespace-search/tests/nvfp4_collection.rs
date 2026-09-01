use ed25519_dalek::SigningKey;

use triblespace_core::attribute::Attribute;
use triblespace_core::collection::{
    AdmissionPolicy, CollectionPolicy, CollectionStoreExt, TryFromCover,
};
use triblespace_core::id::Id;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::Inline;
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{BlobStorePut, SnapshotSource};
use triblespace_core::trible::{Fragment, Trible, TribleSet};

use triblespace_search::nvfp4::{EmbeddingAttributeToNvFp4, NvFp4CosineIndex};
use triblespace_search::schemas::Embedding;

fn direct_policy(authority: &SigningKey) -> CollectionPolicy {
    let root = authority.verifying_key();
    CollectionPolicy::new(AdmissionPolicy::direct(root), AdmissionPolicy::direct(root))
}

#[test]
fn simplearchive_mapping_lazy_view_and_exact_queries_compose() {
    let authority = SigningKey::from_bytes(&[91; 32]);
    let policy = direct_policy(&authority);
    let attribute = Attribute::<Handle<Embedding>>::named("nvfp4-test-embedding");
    let mut store = MemoryRepo::default();

    let positive = store.put::<Embedding, _>(vec![1.0f32, 0.0, 0.0]).unwrap();
    let diagonal = store.put::<Embedding, _>(vec![1.0f32, 1.0, 0.0]).unwrap();
    let negative = store.put::<Embedding, _>(vec![-1.0f32, 0.0, 0.0]).unwrap();

    let mut facts = TribleSet::new();
    for (entity, embedding) in [
        (1, positive),
        (2, diagonal),
        (3, negative),
        // Projection has set semantics by exact embedding handle.
        (4, positive),
    ] {
        let entity = Id::new([entity; 16]).unwrap();
        facts.insert(&Trible::force(&entity, &attribute.id(), &embedding));
    }

    let source = store.collection("nvfp4-source", policy.clone()).unwrap();
    let target = store
        .derive(
            source,
            EmbeddingAttributeToNvFp4::<Embedding>::new(attribute.id(), 3).unwrap(),
            policy,
        )
        .unwrap();
    store
        .commit(source, &authority, Fragment::from(facts))
        .unwrap();

    let snapshot = store.snapshot().unwrap();
    let source_cover = source.admitted(&snapshot).unwrap();
    let target_cover = store
        .ensure::<EmbeddingAttributeToNvFp4<Embedding>>(target, &source_cover)
        .unwrap();
    let snapshot = store.snapshot().unwrap();
    let index = NvFp4CosineIndex::<Embedding>::try_from_cover(&target_cover, &snapshot).unwrap();

    assert_eq!(index.dimension(), 3);
    assert_eq!(index.segment_count(), 1);
    let top = index.top_k(&snapshot, &[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].embedding, positive);
    assert_eq!(top[0].score, 1.0);
    assert_eq!(top[1].embedding, diagonal);
    assert!(top[1].score > 0.7 && top[1].score < 0.71);

    let above = index.above(&snapshot, &[1.0, 0.0, 0.0], 0.7).unwrap();
    assert_eq!(
        above
            .iter()
            .map(|hit| hit.embedding)
            .collect::<Vec<Inline<Handle<Embedding>>>>(),
        vec![positive, diagonal],
    );
}
