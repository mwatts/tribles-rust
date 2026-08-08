//! [`Bm25Rollup`]: an [`IndexKind`] whose segments are persisted
//! succinct BM25 indexes over a branch's message-content tribles.
//!
//! # The waste this removes
//!
//! Lexical archive search (`archive search`) used to persist ONE
//! monolithic BM25 index and rebuild-and-replace it wholesale on
//! every `archive index` run: a fresh index entity minted each time,
//! the whole corpus re-tokenised, the old index left as orphaned
//! exhaust. [`Bm25Rollup`] persists exact-typed artifacts on inclusive
//! commit ranges instead. [`append_range`] appends a logical source range,
//! including certified-empty projections, and size-tiered compaction bounds
//! the read fan-out while preserving the exact DAG cover.
//!
//! [`SuccinctRollup`]: triblespace_core::repo::index_home::SuccinctRollup
//! [`HnswRollup`]: crate::index_hnsw::HnswRollup
//! [`append_range`]: triblespace_core::repo::index_home::append_range
//!
//! # Where the text lives
//!
//! The source view passed to [`IndexKind::build`] carries
//! `message -> Handle<LongString>` content tribles under a caller-named
//! attribute; the message *text* is a separate content-addressed blob
//! in the pile. So — like [`HnswRollup`] and its embedding handles —
//! `Bm25Rollup` holds a blob reader to resolve those handles into the
//! strings [`crate::tokens::hash_tokens`] tokenises. The reader is used
//! only by [`build`](IndexKind::build); merge operates directly on the
//! persisted succinct segments, and [`attach`](IndexKind::attach) is
//! zero-copy (it decodes only the stored succinct blob).
//!
//! # Multi-segment query semantics
//!
//! A selected LSM cover can hold several segments. Persisted postings carry
//! exact raw term frequencies. [`SuccinctBM25Cover`] joins their document
//! carrier and document lengths once, then derives global IDF and BM25 scores
//! from only the queried postings. Consequently the same logical corpus has
//! identical scores and ranking under every physical cover shape. Callers
//! serving repeated queries should retain that resident cover; [`query_across`]
//! is the one-shot convenience path.
//!
//! [`query_across`]: crate::index_bm25::query_across
//! [`merge`]: IndexKind::merge

use std::collections::HashMap;

use anybytes::View;

use triblespace_core::blob::encodings::longstring::LongString;
use triblespace_core::blob::{Blob, IntoBlob, TryFromBlob};
use triblespace_core::id::{ExclusiveId, Id};
use triblespace_core::inline::encodings::genid::GenId;
use triblespace_core::inline::encodings::hash::Handle;
use triblespace_core::inline::{Inline, RawInline};
use triblespace_core::metadata;
use triblespace_core::prelude::{entity, pattern};
use triblespace_core::repo::index_home::{ArtifactError, IndexKind};
use triblespace_core::repo::{BlobStoreGet, BlobStorePut};
use triblespace_core::trible::{Fragment, TribleSet};

use crate::bm25::BM25Builder;
use crate::index_schema::{index_source_attribute, seg_bm25};
pub use crate::succinct::Bm25TuningMismatch;
use crate::succinct::{SuccinctBM25Blob, SuccinctBM25Cover, SuccinctBM25Index};
use crate::tokens::WordHash;

/// The document-key / term schemas of the BM25 segments this kind
/// builds: entity-keyed documents, word-hash terms — the classic
/// text-search shape and the one `archive search` uses.
type Seg = SuccinctBM25Index<GenId, WordHash>;

/// An [`IndexKind`] whose segments are [`SuccinctBM25Index`]es over the
/// `Handle<LongString>` content a branch's entities point at, keyed by
/// entity id.
///
/// Parameterised by the blob reader `R` used to resolve those content handles
/// into text during [`build`](IndexKind::build). Merge operates directly on
/// exact frequencies in the persisted segments. Attach and query need no
/// content reader — the stored succinct index is self-contained (terms are
/// hashed at build time).
#[derive(Clone)]
pub struct Bm25Rollup<R> {
    reader: R,
    content_attr: Id,
}

impl<R> Bm25Rollup<R> {
    /// A rollup that indexes the text behind the `Handle<LongString>`
    /// values stored under `content_attr`, resolving them through
    /// `reader`.
    pub fn new(reader: R, content_attr: Id) -> Self {
        Self {
            reader,
            content_attr,
        }
    }

    /// Stable kind id — minted via `trible genid`
    /// (`881C9D0DAC43814CB4E80897E420B67B`). Distinct from
    /// `SuccinctRollup`'s and `HnswRollup`'s so all three kinds'
    /// manifests coexist in one branch-head tribleset.
    pub const KIND_ID_HEX: &'static str = "881C9D0DAC43814CB4E80897E420B67B";
}

impl<R> Bm25Rollup<R>
where
    R: BlobStoreGet,
{
    /// Resolve one content handle into its text. A range is a completion
    /// certificate, so an unreadable source handle fails the build instead of
    /// silently publishing an incomplete projection.
    fn text_of(&self, h: Inline<Handle<LongString>>) -> Result<String, ArtifactError> {
        let view: View<str> = self
            .reader
            .get::<View<str>, LongString>(h)
            .map_err(|error| Box::new(error) as ArtifactError)?;
        Ok(view.as_ref().to_owned())
    }

    /// Build a succinct BM25 blob from an iterator of `(doc_key,
    /// tokens)` rows. Used by `build` and by materialized-oracle tests for
    /// the streaming merge.
    fn build_blob<I>(&self, rows: I) -> Blob<SuccinctBM25Blob>
    where
        I: IntoIterator<Item = (Inline<GenId>, Vec<Inline<WordHash>>)>,
    {
        let mut builder: BM25Builder<GenId, WordHash> = BM25Builder::new();
        for (key, tokens) in rows {
            builder.insert(key, tokens);
        }
        let idx: Seg = builder.build();
        (&idx).to_blob()
    }
}

impl<R> IndexKind for Bm25Rollup<R>
where
    R: BlobStoreGet,
{
    type Segment = Seg;
    type PreparedArtifact = Blob<SuccinctBM25Blob>;
    type StoredArtifact = Inline<Handle<SuccinctBM25Blob>>;

    fn recipe_fragment(&self) -> Fragment {
        let algorithm = Id::from_hex(Self::KIND_ID_HEX).expect("valid algorithm id");
        entity! { _ @
            metadata::tag: algorithm,
            index_source_attribute: self.content_attr,
        }
    }

    fn build(&self, source: &TribleSet) -> Result<Vec<Self::PreparedArtifact>, ArtifactError> {
        // Extract `entity -> Handle<LongString>` tribles under our
        // content attribute and tokenise each resolved string. An entity can
        // carry several content values in one commit. Treat those values as a
        // monotone union: for each term keep the largest frequency seen in
        // any value. `max` makes the result independent of trible order,
        // retains terms from every value, and keeps exact duplicates
        // idempotent instead of lengthening the document.
        let mut docs: HashMap<RawInline, HashMap<RawInline, u32>> = HashMap::new();
        for t in source.iter().filter(|t| t.a() == &self.content_attr) {
            let key: Inline<GenId> = triblespace_core::inline::IntoInline::to_inline(t.e());
            let handle: Inline<Handle<LongString>> = *t.v::<Handle<LongString>>();
            let text = self.text_of(handle)?;

            let mut value_tfs: HashMap<RawInline, u32> = HashMap::new();
            for term in crate::tokens::hash_tokens(&text) {
                *value_tfs.entry(term.raw).or_default() += 1;
            }
            let doc_tfs = docs.entry(key.raw).or_default();
            for (term, tf) in value_tfs {
                doc_tfs
                    .entry(term)
                    .and_modify(|old| *old = (*old).max(tf))
                    .or_insert(tf);
            }
        }

        let mut rows: Vec<(Inline<GenId>, Vec<Inline<WordHash>>)> = docs
            .into_iter()
            .map(|(key, tfs)| {
                let mut tfs: Vec<(RawInline, u32)> = tfs.into_iter().collect();
                tfs.sort_unstable_by_key(|&(term, _)| term);
                let tokens = tfs
                    .into_iter()
                    .flat_map(|(term, tf)| {
                        std::iter::repeat(Inline::<WordHash>::new(term)).take(tf as usize)
                    })
                    .collect();
                (Inline::<GenId>::new(key), tokens)
            })
            .collect();
        rows.sort_unstable_by_key(|(key, _)| key.raw);
        if rows.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![self.build_blob(rows)])
        }
    }

    fn put<S: BlobStorePut>(
        &self,
        storage: &mut S,
        artifact: Self::PreparedArtifact,
    ) -> Result<Self::StoredArtifact, ArtifactError> {
        storage
            .put(artifact)
            .map_err(|error| Box::new(error) as ArtifactError)
    }

    fn emit(&self, entity: Id, artifact: &Self::StoredArtifact) -> TribleSet {
        entity! { ExclusiveId::force_ref(&entity) @ seg_bm25: *artifact }.into_facts()
    }

    fn parse<B: BlobStoreGet>(
        &self,
        _reader: &B,
        facts: &TribleSet,
        entity: Id,
    ) -> Result<Vec<Self::StoredArtifact>, ArtifactError> {
        Ok(triblespace_core::find!(
            handle: Inline<Handle<SuccinctBM25Blob>>,
            pattern!(facts, [{ entity @ seg_bm25: ?handle }])
        )
        .collect())
    }

    fn attach<B: BlobStoreGet>(
        &self,
        reader: &B,
        artifact: &Self::StoredArtifact,
    ) -> Result<Self::Segment, ArtifactError> {
        let blob: Blob<SuccinctBM25Blob> = reader
            .get(*artifact)
            .map_err(|error| Box::new(error) as ArtifactError)?;
        SuccinctBM25Index::try_from_blob(blob).map_err(|error| Box::new(error) as ArtifactError)
    }

    fn merge(
        &self,
        segments: &[Self::Segment],
    ) -> Result<Vec<Self::PreparedArtifact>, ArtifactError> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }
        // The exact join validates one shared scoring recipe at its own
        // boundary and retains all duplicate-key content without a
        // corpus-sized token-bag intermediate.
        let merged = SuccinctBM25Index::try_merge_segments(segments)?;
        if merged.doc_count() == 0 {
            Ok(Vec::new())
        } else {
            Ok(vec![(&merged).to_blob()])
        }
    }
}

/// One-shot rank of a logical corpus represented by several attached BM25
/// artifacts.
///
/// This constructs a [`SuccinctBM25Cover`] and is therefore ideal for a
/// one-off query or correctness boundary. A server should construct and retain
/// the cover itself so its all-postings document-length pass is amortised.
pub fn query_across(
    segments: &[Seg],
    terms: &[Inline<WordHash>],
) -> Result<Vec<(Inline<GenId>, f32)>, Bm25TuningMismatch> {
    Ok(SuccinctBM25Cover::new(segments)?.query_multi(terms))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use anybytes::Bytes;
    use triblespace_core::blob::encodings::longstring::LongString;
    use triblespace_core::blob::Blob;
    use triblespace_core::id::{fucid, Id};
    use triblespace_core::inline::encodings::hash::Handle;
    use triblespace_core::inline::Inline;
    use triblespace_core::prelude::{attributes, entity};
    use triblespace_core::repo::index_home::{append_stored_range, IndexKind, Manifest};
    use triblespace_core::repo::index_range::CommitRange;
    use triblespace_core::repo::memoryrepo::MemoryRepo;
    use triblespace_core::repo::{BlobStore, BlobStorePut};
    use triblespace_core::trible::TribleSet;

    use super::*;
    use crate::index_schema::seg_bm25;
    use crate::tokens::hash_tokens;

    attributes! {
        "155F694D45E9135AEBBE3FDAE750A69F" as content: Handle<LongString>;
        "882E48C941C34CA9B27E708A808AEE1C" as alternate_content: Handle<LongString>;
    }

    fn commit(byte: u8) -> triblespace_core::repo::CommitHandle {
        Inline::new([byte; 32])
    }

    fn stage(storage: &mut MemoryRepo, attribute: Id, document: Id, text: &str) -> TribleSet {
        let handle: Inline<Handle<LongString>> = storage.put(text.to_owned()).unwrap();
        let mut source = TribleSet::new();
        source.insert(&triblespace_core::trible::Trible::new(
            triblespace_core::id::ExclusiveId::force_ref(&document),
            &attribute,
            &handle,
        ));
        source
    }

    fn decode(blob: Blob<SuccinctBM25Blob>) -> Seg {
        SuccinctBM25Index::try_from_blob(blob).unwrap()
    }

    fn reload(segment: &Seg) -> Seg {
        decode(Blob::new(segment.bytes.clone()))
    }

    fn build_segment(kind: &Bm25Rollup<impl BlobStoreGet>, source: &TribleSet) -> Seg {
        decode(kind.build(source).unwrap().pop().unwrap())
    }

    fn merge_segment(kind: &Bm25Rollup<impl BlobStoreGet>, segments: &[Seg]) -> Seg {
        decode(kind.merge(segments).unwrap().pop().unwrap())
    }

    fn synthetic(n: usize) -> Vec<(Id, String)> {
        const VOCAB: &[&str] = &[
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "memory", "pile",
            "trible", "index", "search", "rollup", "segment", "merge",
        ];
        let mut rng = 0xC0FFEE_u64;
        let mut next = || {
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut value = rng;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
            value ^ (value >> 31)
        };
        (0..n)
            .map(|_| {
                let len = 4 + (next() % 12) as usize;
                let words: Vec<_> = (0..len)
                    .map(|_| VOCAB[(next() as usize) % VOCAB.len()])
                    .collect();
                (*fucid(), words.join(" "))
            })
            .collect()
    }

    fn stage_many(storage: &mut MemoryRepo, pairs: &[(Id, String)]) -> TribleSet {
        let mut source = TribleSet::new();
        for (document, text) in pairs {
            source += stage(storage, content.id(), *document, text);
        }
        source
    }

    fn oracle_ranked(table: &[(Id, String)], query: &str) -> Vec<(RawInline, f32)> {
        let mut builder: BM25Builder<GenId, WordHash> = BM25Builder::new();
        for (document, text) in table {
            builder.insert(document, hash_tokens(text));
        }
        builder
            .build()
            .query_multi(&hash_tokens(query))
            .into_iter()
            .map(|(document, score)| (document.raw, score))
            .collect()
    }

    #[derive(Clone, Copy)]
    struct MergeRng(u64);

    impl MergeRng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_133111EB);
            value ^ (value >> 31)
        }
    }

    fn merge_doc(ordinal: u64) -> Inline<GenId> {
        let mut raw = [0u8; 32];
        raw[0] = 1;
        raw[24..].copy_from_slice(&ordinal.to_be_bytes());
        Inline::new(raw)
    }

    fn merge_term(ordinal: u64) -> Inline<WordHash> {
        let mut raw = [0u8; 32];
        raw[..8].copy_from_slice(&ordinal.to_be_bytes());
        raw[8..16].copy_from_slice(&ordinal.rotate_left(13).to_be_bytes());
        raw[16..24].copy_from_slice(&ordinal.rotate_left(29).to_be_bytes());
        raw[24..].copy_from_slice(&ordinal.rotate_left(47).to_be_bytes());
        Inline::new(raw)
    }

    fn materialized_max_union(segments: &[Seg], k1: f32, b: f32) -> Seg {
        let mut union: HashMap<RawInline, HashMap<RawInline, u32>> = HashMap::new();
        for segment in segments {
            for (key, tokens) in segment.reconstruct_docs() {
                let mut source_tfs: HashMap<RawInline, u32> = HashMap::new();
                for term in tokens {
                    *source_tfs.entry(term).or_default() += 1;
                }
                let merged_tfs = union.entry(key).or_default();
                for (term, frequency) in source_tfs {
                    merged_tfs
                        .entry(term)
                        .and_modify(|old| *old = (*old).max(frequency))
                        .or_insert(frequency);
                }
            }
        }

        let mut rows: Vec<_> = union.into_iter().collect();
        rows.sort_unstable_by_key(|(key, _)| *key);
        let mut builder: BM25Builder<GenId, WordHash> = BM25Builder::new().k1(k1).b(b);
        for (key, frequencies) in rows {
            let mut frequencies: Vec<_> = frequencies.into_iter().collect();
            frequencies.sort_unstable_by_key(|(term, _)| *term);
            let terms = frequencies.into_iter().flat_map(|(term, frequency)| {
                std::iter::repeat_n(Inline::<WordHash>::new(term), frequency as usize)
            });
            builder.insert(Inline::<GenId>::new(key), terms);
        }
        builder.build()
    }

    fn adversarial_cover_leaves() -> Vec<Seg> {
        let alpha = merge_term(1);
        let beta = merge_term(2);
        let gamma = merge_term(3);
        let common = merge_term(4);
        let rare = merge_term(5);

        let leaf = |rows: Vec<(u64, Vec<Inline<WordHash>>)>| {
            let mut builder: BM25Builder<GenId, WordHash> = BM25Builder::new();
            for (document, terms) in rows {
                builder.insert(merge_doc(document), terms);
            }
            builder.build()
        };

        vec![
            leaf(vec![
                (1, Vec::new()),
                (
                    2,
                    std::iter::repeat_n(alpha, 65_537).chain([beta]).collect(),
                ),
                (3, vec![common, common]),
            ]),
            leaf(vec![
                (2, std::iter::repeat_n(alpha, 65_539).collect()),
                (2, vec![beta, beta, beta, gamma]),
                (4, vec![rare, common]),
            ]),
            leaf(vec![
                (3, vec![common, beta, beta]),
                (5, vec![alpha, gamma]),
                (6, Vec::new()),
            ]),
            leaf(vec![
                (1, vec![gamma]),
                (4, vec![rare, rare, rare, rare]),
                (7, vec![common]),
            ]),
            leaf(vec![
                (5, vec![alpha, alpha, beta]),
                (7, vec![common]),
                (8, vec![gamma, rare]),
            ]),
        ]
    }

    fn ranked_bits(rows: Vec<(Inline<GenId>, f32)>) -> Vec<(RawInline, u32)> {
        rows.into_iter()
            .map(|(document, score)| (document.raw, score.to_bits()))
            .collect()
    }

    type ExpectedRankings = Vec<(Vec<Inline<WordHash>>, Vec<(RawInline, u32)>)>;

    fn assert_cover_matches(segments: &[Seg], expected: &ExpectedRankings, expected_docs: usize) {
        let cover = SuccinctBM25Cover::new(segments).unwrap();
        assert_eq!(cover.doc_count(), expected_docs);
        for (query, ranking) in expected {
            assert_eq!(ranked_bits(cover.query_multi(query)), *ranking);
            assert_eq!(
                ranked_bits(query_across(segments, query).unwrap()),
                *ranking
            );
        }
    }

    #[test]
    fn single_artifact_equals_monolithic_oracle() {
        let pairs = synthetic(120);
        let mut storage = MemoryRepo::default();
        let source = stage_many(&mut storage, &pairs);
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let artifact = kind.build(&source).unwrap().pop().unwrap();
        let reloaded = Blob::<SuccinctBM25Blob>::new(artifact.bytes.clone());
        let segment = decode(reloaded);
        assert_eq!(segment.doc_count(), pairs.len());

        for query in [
            "alpha",
            "memory search",
            "rollup segment merge",
            "theta zeta",
        ] {
            let got: HashMap<_, _> =
                query_across(std::slice::from_ref(&segment), &hash_tokens(query))
                    .unwrap()
                    .into_iter()
                    .map(|(document, score)| (document.raw, score))
                    .collect();
            let expected: HashMap<_, _> = oracle_ranked(&pairs, query).into_iter().collect();
            assert_eq!(got.len(), expected.len(), "query `{query}` hit count");
            for (document, expected_score) in expected {
                let score = got[&document];
                assert_eq!(score.to_bits(), expected_score.to_bits());
            }
        }
    }

    #[test]
    fn build_unions_repeated_content_values_by_max_tf() {
        let mut storage = MemoryRepo::default();
        let shared = *fucid();
        let mut source = stage(
            &mut storage,
            content.id(),
            shared,
            "alpha alpha first_value",
        );
        source += stage(
            &mut storage,
            content.id(),
            shared,
            "alpha beta second_value",
        );
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let segment = build_segment(&kind, &source);
        assert_eq!(segment.doc_count(), 1);
        assert_eq!(segment.doc_len(0), Some(5));
        for term in ["alpha", "beta", "first_value", "second_value"] {
            assert_eq!(segment.query_multi(&hash_tokens(term)).len(), 1);
        }
    }

    #[test]
    fn merge_matches_monolithic_document_sets() {
        let mut storage = MemoryRepo::default();
        let first = synthetic(60);
        let second = synthetic(60);
        let source_a = stage_many(&mut storage, &first);
        let source_b = stage_many(&mut storage, &second);
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let merged = merge_segment(
            &kind,
            &[
                build_segment(&kind, &source_a),
                build_segment(&kind, &source_b),
            ],
        );
        let mut union = first;
        union.extend(second);
        assert_eq!(merged.doc_count(), union.len());
        for query in ["memory pile", "alpha beta gamma", "index search rollup"] {
            let got: HashSet<_> = query_across(std::slice::from_ref(&merged), &hash_tokens(query))
                .unwrap()
                .into_iter()
                .map(|(document, _)| document.raw)
                .collect();
            let expected: HashSet<_> = oracle_ranked(&union, query)
                .into_iter()
                .map(|(document, _)| document)
                .collect();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn bounded_merge_is_max_union_order_independent_and_idempotent() {
        let mut storage = MemoryRepo::default();
        let shared = *fucid();
        let first_only = *fucid();
        let second_only = *fucid();
        let source_a = stage_many(
            &mut storage,
            &[
                (shared, "alpha alpha first_owner".into()),
                (first_only, "gamma stable".into()),
            ],
        );
        let source_b = stage_many(
            &mut storage,
            &[
                (shared, "shadow_only beta".into()),
                (second_only, "beta delta".into()),
            ],
        );
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let left = build_segment(&kind, &source_a);
        let right = build_segment(&kind, &source_b);
        let defaults: BM25Builder<GenId, WordHash> = BM25Builder::new();
        let expected =
            materialized_max_union(&[reload(&left), reload(&right)], defaults.k1, defaults.b);
        let direct = merge_segment(&kind, &[reload(&left), reload(&right)]);
        let reversed = merge_segment(&kind, &[reload(&right), reload(&left)]);
        assert_eq!(direct.bytes.as_ref(), expected.bytes.as_ref());
        assert_eq!(direct.bytes.as_ref(), reversed.bytes.as_ref());

        let duplicate = merge_segment(&kind, &[reload(&left), left]);
        assert_eq!(duplicate.doc_count(), 2);
        let shared_key: Inline<GenId> = triblespace_core::inline::IntoInline::to_inline(&shared);
        let code = duplicate
            .document_keys()
            .position(|key| key == shared_key)
            .unwrap();
        assert_eq!(duplicate.doc_len(code), Some(3));
    }

    #[test]
    fn randomized_high_tf_merge_matches_materialized_max_union() {
        const SEGMENTS: usize = 5;
        const DOCS_PER_SEGMENT: usize = 36;
        const SHARED_DOCS: usize = 15;
        const VOCAB: u64 = 41;

        let mut segments = Vec::new();
        for segment in 0..SEGMENTS {
            let mut rng = MergeRng(0xB25_0A11 ^ segment as u64);
            let mut builder: BM25Builder<GenId, WordHash> = BM25Builder::new();
            for local in 0..DOCS_PER_SEGMENT {
                let ordinal = if local < SHARED_DOCS {
                    local
                } else {
                    SHARED_DOCS + segment * (DOCS_PER_SEGMENT - SHARED_DOCS) + local - SHARED_DOCS
                };
                let mut terms = Vec::new();
                for slot in 0..12 {
                    let term = merge_term(rng.next() % VOCAB);
                    let mut frequency = 1 + (rng.next() % 9) as usize;
                    if (segment + local + slot) % 43 == 0 {
                        frequency = 257 + (rng.next() % 1_300) as usize;
                    }
                    terms.extend(std::iter::repeat_n(term, frequency));
                }
                if local == 0 {
                    terms.extend(std::iter::repeat_n(merge_term(0), 300 + segment * 700));
                }
                builder.insert(merge_doc((ordinal + 1) as u64), terms);
            }
            segments.push(builder.build());
        }

        let defaults: BM25Builder<GenId, WordHash> = BM25Builder::new();
        let expected = materialized_max_union(&segments, defaults.k1, defaults.b);
        let merged = SuccinctBM25Index::try_merge_segments(&segments).unwrap();
        assert_eq!(merged.bytes.as_ref(), expected.bytes.as_ref());
        let left = SuccinctBM25Index::try_merge_segments(&segments[..2]).unwrap();
        let right = SuccinctBM25Index::try_merge_segments(&segments[2..]).unwrap();
        let grouped = SuccinctBM25Index::try_merge_segments(&[left, right]).unwrap();
        assert_eq!(merged.bytes.as_ref(), grouped.bytes.as_ref());
        segments.reverse();
        let reversed = SuccinctBM25Index::try_merge_segments(&segments).unwrap();
        assert_eq!(merged.bytes.as_ref(), reversed.bytes.as_ref());
        segments.push(reload(&segments[0]));
        let duplicated = SuccinctBM25Index::try_merge_segments(&segments).unwrap();
        assert_eq!(merged.bytes.as_ref(), duplicated.bytes.as_ref());
    }

    #[test]
    fn resident_cover_is_invariant_under_arbitrary_lsm_shapes() {
        let queries = vec![
            vec![merge_term(1)],
            vec![merge_term(2), merge_term(3)],
            // Query multiplicity is part of the bag-of-words query, even
            // though corpus duplicates join idempotently.
            vec![merge_term(4), merge_term(4), merge_term(5)],
            vec![merge_term(99)],
        ];
        let canonical = SuccinctBM25Index::try_merge_segments(&adversarial_cover_leaves()).unwrap();
        let expected: ExpectedRankings = queries
            .into_iter()
            .map(|query| {
                let ranking = ranked_bits(canonical.query_multi(&query));
                (query, ranking)
            })
            .collect();
        let expected_docs = canonical.doc_count();
        assert_eq!(expected_docs, 8);

        // Raw leaves, including an exact duplicate leaf: duplicate physical
        // coverage must not inflate either frequencies or document lengths.
        let mut raw = adversarial_cover_leaves();
        raw.push(reload(&raw[0]));
        assert_cover_matches(&raw, &expected, expected_docs);

        // A deliberately uneven partial compaction.
        let mut leaves = adversarial_cover_leaves();
        let tail = leaves.split_off(3);
        let right = SuccinctBM25Index::try_merge_segments(&tail).unwrap();
        let left_pair = SuccinctBM25Index::try_merge_segments(&leaves[..2]).unwrap();
        let partial = vec![left_pair, reload(&leaves[2]), right];
        assert_cover_matches(&partial, &expected, expected_docs);

        // Random merge trees exercise every intermediate cover, not only the
        // final compacted singleton. Physical order is perturbed by
        // swap-removal along the way.
        for seed in 0..12 {
            let mut rng = MergeRng(0xC07E_5A11 ^ seed);
            let mut cover = adversarial_cover_leaves();
            assert_cover_matches(&cover, &expected, expected_docs);
            while cover.len() > 1 {
                let left = (rng.next() as usize) % cover.len();
                let a = cover.swap_remove(left);
                let right = (rng.next() as usize) % cover.len();
                let b = cover.swap_remove(right);
                cover.push(SuccinctBM25Index::try_merge_segments(&[a, b]).unwrap());
                assert_cover_matches(&cover, &expected, expected_docs);
            }
            assert_eq!(cover[0].bytes.as_ref(), canonical.bytes.as_ref());
        }
    }

    #[test]
    fn cover_rejects_mixed_scoring_recipes() {
        let mut left: BM25Builder<GenId, WordHash> = BM25Builder::new().k1(1.2).b(0.75);
        left.insert(merge_doc(1), [merge_term(1)]);
        let mut right: BM25Builder<GenId, WordHash> = BM25Builder::new().k1(1.5).b(0.75);
        right.insert(merge_doc(2), [merge_term(1)]);
        let segments = [left.build(), right.build()];

        assert!(SuccinctBM25Cover::new(&segments).is_err());
        assert!(query_across(&segments, &[merge_term(1)]).is_err());
        let error = SuccinctBM25Index::try_merge_segments(&segments).unwrap_err();
        assert!(error.downcast_ref::<Bm25TuningMismatch>().is_some());
    }

    #[test]
    fn typed_fact_roundtrip_attaches_and_queries() {
        let mut storage = MemoryRepo::default();
        let document = *fucid();
        let source = stage(&mut storage, content.id(), document, "alpha beta alpha");
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let artifact = kind.build(&source).unwrap().pop().unwrap();
        let stored = kind.put(&mut storage, artifact).unwrap();
        let range_entity = *fucid();
        let facts = kind.emit(range_entity, &stored);

        assert!(facts.iter().all(|fact| fact.a() == &seg_bm25.id()));
        let reader = storage.reader().unwrap();
        assert_eq!(
            kind.parse(&reader, &facts, range_entity).unwrap(),
            vec![stored]
        );
        let attached = kind.attach(&reader, &stored).unwrap();
        let hits: HashSet<_> = query_across(&[attached], &hash_tokens("alpha"))
            .unwrap()
            .into_iter()
            .map(|(key, _)| key.raw)
            .collect();
        let key: Inline<GenId> = triblespace_core::inline::IntoInline::to_inline(&document);
        assert_eq!(hits, HashSet::from([key.raw]));
    }

    #[test]
    fn canonical_empty_projection_and_merge_have_no_artifacts() {
        let mut storage = MemoryRepo::default();
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        assert!(kind.build(&TribleSet::new()).unwrap().is_empty());
        assert!(kind.merge(&[]).unwrap().is_empty());

        let unrelated = entity! { _ @ alternate_content: storage.put::<LongString, _>("x".to_owned()).unwrap() }
            .into_facts();
        assert!(kind.build(&unrelated).unwrap().is_empty());
    }

    #[test]
    fn unreadable_source_content_fails_the_range_build() {
        let mut storage = MemoryRepo::default();
        let document = *fucid();
        let missing = Inline::<Handle<LongString>>::new([0xA5; 32]);
        let mut source = TribleSet::new();
        source.insert(&triblespace_core::trible::Trible::new(
            triblespace_core::id::ExclusiveId::force_ref(&document),
            &content.id(),
            &missing,
        ));
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        assert!(kind.build(&source).is_err());
    }

    #[test]
    fn recipe_identity_depends_on_source_but_not_reader() {
        let mut left_store = MemoryRepo::default();
        let mut right_store = MemoryRepo::default();
        let left = Bm25Rollup::new(left_store.reader().unwrap(), content.id());
        let same = Bm25Rollup::new(right_store.reader().unwrap(), content.id());
        let other = Bm25Rollup::new(right_store.reader().unwrap(), alternate_content.id());

        assert_eq!(left.recipe_fragment().root(), same.recipe_fragment().root());
        assert_ne!(
            left.recipe_fragment().root(),
            other.recipe_fragment().root()
        );
    }

    #[test]
    fn parameter_distinct_bm25_recipes_coexist_in_one_manifest_set() {
        let mut storage = MemoryRepo::default();
        let document = *fucid();
        let source_a = stage(&mut storage, content.id(), document, "alpha");
        let source_b = stage(&mut storage, alternate_content.id(), document, "beta");
        let reader = storage.reader().unwrap();
        let kind_a = Bm25Rollup::new(reader.clone(), content.id());
        let kind_b = Bm25Rollup::new(reader, alternate_content.id());
        let artifact_a = kind_a.build(&source_a).unwrap().pop().unwrap();
        let artifact_b = kind_b.build(&source_b).unwrap().pop().unwrap();
        let stored_a = kind_a.put(&mut storage, artifact_a).unwrap();
        let stored_b = kind_b.put(&mut storage, artifact_b).unwrap();
        let mut branch_set = TribleSet::new();

        append_stored_range(
            &mut storage,
            &kind_a,
            CommitRange::leaf(commit(1)),
            vec![stored_a],
            &mut branch_set,
        )
        .unwrap();
        append_stored_range(
            &mut storage,
            &kind_b,
            CommitRange::leaf(commit(1)),
            vec![stored_b],
            &mut branch_set,
        )
        .unwrap();

        let reader = storage.reader().unwrap();
        let manifest_a = Manifest::from_tribles(&branch_set, &reader, &kind_a).unwrap();
        let manifest_b = Manifest::from_tribles(&branch_set, &reader, &kind_b).unwrap();
        assert_ne!(manifest_a.recipe(), manifest_b.recipe());
        assert_eq!(manifest_a.ranges().len(), 1);
        assert_eq!(manifest_b.ranges().len(), 1);
        assert_eq!(manifest_a.ranges()[0].artifacts(), &[stored_a]);
        assert_eq!(manifest_b.ranges()[0].artifacts(), &[stored_b]);
    }

    #[test]
    fn repeated_typed_facts_are_physical_artifacts_and_bad_bytes_fail_attach() {
        let mut storage = MemoryRepo::default();
        let source_a = stage(&mut storage, content.id(), *fucid(), "alpha");
        let source_b = stage(&mut storage, content.id(), *fucid(), "beta");
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let stored_a = kind
            .put(&mut storage, kind.build(&source_a).unwrap().pop().unwrap())
            .unwrap();
        let stored_b = kind
            .put(&mut storage, kind.build(&source_b).unwrap().pop().unwrap())
            .unwrap();
        let entity = *fucid();
        let mut facts = kind.emit(entity, &stored_a);
        facts += kind.emit(entity, &stored_b);
        let reader = storage.reader().unwrap();
        let parsed: HashSet<_> = kind
            .parse(&reader, &facts, entity)
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(parsed, HashSet::from([stored_a, stored_b]));

        let malformed = Blob::<SuccinctBM25Blob>::new(Bytes::from(vec![0u8; 8]));
        let malformed_handle = storage.put(malformed).unwrap();
        let reader = storage.reader().unwrap();
        assert!(kind.attach(&reader, &malformed_handle).is_err());
    }

    #[test]
    fn typed_merge_preserves_document_union() {
        let mut storage = MemoryRepo::default();
        let first = stage(&mut storage, content.id(), *fucid(), "alpha");
        let second = stage(&mut storage, content.id(), *fucid(), "beta");
        let kind = Bm25Rollup::new(storage.reader().unwrap(), content.id());
        let left = decode(kind.build(&first).unwrap().pop().unwrap());
        let right = decode(kind.build(&second).unwrap().pop().unwrap());
        let merged = decode(kind.merge(&[left, right]).unwrap().pop().unwrap());
        assert_eq!(merged.doc_count(), 2);
        assert_eq!(
            query_across(&[merged], &hash_tokens("alpha beta"))
                .unwrap()
                .len(),
            2
        );
    }
}
