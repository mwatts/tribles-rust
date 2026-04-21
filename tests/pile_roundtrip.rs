//! End-to-end smoke test: does a succinct index actually
//! survive a real `BlobStore::put` / `BlobStoreGet::get` cycle?
//!
//! The in-crate BlobSchema tests exercise `to_blob` /
//! `try_from_blob` directly. This test closes the last-mile
//! loop: go through the triblespace repo traits so we know the
//! handle-typed API works unmodified with SB25 + SH25 blobs.
//!
//! Uses `MemoryBlobStore` rather than an on-disk pile because
//! the test is about the API chain, not file I/O — the
//! pile-backed write path exercises the same traits.

use triblespace::core::blob::MemoryBlobStore;
use triblespace::core::id::Id;
use triblespace::core::repo::{BlobStoreGet, BlobStorePut};

use triblespace_search::bm25::BM25Builder;
use triblespace_search::hnsw::HNSWBuilder;
use triblespace_search::succinct::{
    SuccinctBM25Blob, SuccinctBM25Index, SuccinctHNSWBlob, SuccinctHNSWIndex,
};
use triblespace_search::tokens::hash_tokens;

fn iid(byte: u8) -> Id {
    Id::new([byte; 16]).unwrap()
}

#[test]
fn succinct_bm25_survives_blob_store_roundtrip() {
    // Build a small index.
    let mut b = BM25Builder::new();
    b.insert_id(iid(1), hash_tokens("the quick brown fox"));
    b.insert_id(iid(2), hash_tokens("the lazy brown dog"));
    b.insert_id(iid(3), hash_tokens("quick silver fox jumps"));
    let naive = b.build();
    let original = SuccinctBM25Index::from_naive(&naive).unwrap();

    // Put → handle.
    let mut store = MemoryBlobStore::<triblespace::core::value::schemas::hash::Blake3>::new();
    let handle = store
        .put::<SuccinctBM25Blob, _>(&original)
        .expect("put should succeed");

    // Get → reloaded view.
    let reader = <MemoryBlobStore<_> as triblespace::core::repo::BlobStore<_>>::reader(&mut store)
        .expect("reader");
    let reloaded: SuccinctBM25Index = reader
        .get::<SuccinctBM25Index, SuccinctBM25Blob>(handle)
        .expect("get should succeed");

    // Same corpus descriptors.
    assert_eq!(reloaded.doc_count(), original.doc_count());
    assert_eq!(reloaded.term_count(), original.term_count());
    assert_eq!(reloaded.k1(), original.k1());
    assert_eq!(reloaded.b(), original.b());
    assert!((reloaded.avg_doc_len() - original.avg_doc_len()).abs() < 1e-6);

    // Same query answer for "fox".
    let fox = hash_tokens("fox")[0];
    let a: Vec<_> = original.query_term(&fox).collect();
    let r: Vec<_> = reloaded.query_term(&fox).collect();
    assert_eq!(a.len(), r.len());
    let tol = reloaded.score_tolerance().max(1e-5);
    for ((a_id, a_s), (r_id, r_s)) in a.iter().zip(r.iter()) {
        assert_eq!(a_id, r_id);
        assert!(
            (a_s - r_s).abs() <= tol,
            "score drift after pile round-trip: {a_s} vs {r_s} > tol {tol}"
        );
    }
}

#[test]
fn succinct_hnsw_survives_blob_store_roundtrip() {
    // Build a small HNSW index.
    let mut b = HNSWBuilder::new(4).with_seed(9);
    for i in 1..=12u8 {
        let f = i as f32;
        let v = vec![f.sin(), f.cos(), (f * 0.5).sin(), (f * 0.3).cos()];
        b.insert(iid(i), v).unwrap();
    }
    let naive = b.build();
    let original = SuccinctHNSWIndex::from_naive(&naive).unwrap();

    // Put → handle.
    let mut store = MemoryBlobStore::<triblespace::core::value::schemas::hash::Blake3>::new();
    let handle = store
        .put::<SuccinctHNSWBlob, _>(&original)
        .expect("put should succeed");

    // Get → reloaded view.
    let reader = <MemoryBlobStore<_> as triblespace::core::repo::BlobStore<_>>::reader(&mut store)
        .expect("reader");
    let reloaded: SuccinctHNSWIndex = reader
        .get::<SuccinctHNSWIndex, SuccinctHNSWBlob>(handle)
        .expect("get should succeed");

    assert_eq!(reloaded.doc_count(), original.doc_count());
    assert_eq!(reloaded.dim(), original.dim());
    assert_eq!(reloaded.max_level(), original.max_level());

    // Same top-3 for the same query.
    let q = vec![0.5, -0.2, 0.3, 0.7];
    let a = original.similar(&q, 3, Some(10));
    let r = reloaded.similar(&q, 3, Some(10));
    assert_eq!(a.len(), r.len());
    for ((a_id, a_s), (r_id, r_s)) in a.iter().zip(r.iter()) {
        assert_eq!(a_id, r_id);
        assert!(
            (a_s - r_s).abs() < 1e-5,
            "score drift after pile round-trip: {a_s} vs {r_s}"
        );
    }
}
