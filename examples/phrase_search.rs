//! Phrase-aware BM25 retrieval via composed tokenizers.
//!
//! Indexes each doc with both `hash_tokens` (single-word terms)
//! and `bigram_tokens` (ordered word pairs). The same index then
//! answers:
//! - Single-word queries via the hash-tokens half.
//! - Phrase queries via the bigram half — only docs that contain
//!   the phrase's consecutive word pairs rank highly.
//!
//! ```sh
//! cargo run --example phrase_search
//! ```

use triblespace::core::id::Id;
use triblespace::core::value::RawValue;
use triblespace_search::bm25::BM25Builder;
use triblespace_search::succinct::SuccinctBM25Index;
use triblespace_search::tokens::{bigram_tokens, hash_tokens};

fn id(byte: u8) -> Id {
    Id::new([byte; 16]).expect("non-nil")
}

/// Index a doc with single-word + bigram tokens combined.
fn phrase_tokens(text: &str) -> Vec<RawValue> {
    let mut v = hash_tokens(text);
    v.extend(bigram_tokens(text));
    v
}

fn main() {
    // A corpus where the word "fox" appears in all four docs,
    // but only two have the phrase "quick brown" as an adjacent
    // pair, and only one has "brown fox" adjacent.
    let corpus = [
        (id(1), "the quick brown fox jumps"),
        (id(2), "a quick silver fox"),
        (id(3), "the brown fox runs fast"),
        (id(4), "quick fox and brown dog"),
    ];

    let mut b = BM25Builder::new();
    for (doc_id, text) in &corpus {
        b.insert_id(*doc_id, phrase_tokens(text));
    }
    let idx = SuccinctBM25Index::from_naive(&b.build()).unwrap();
    println!(
        "index: {} docs, {} terms (single-words + bigrams)\n",
        idx.doc_count(),
        idx.term_count()
    );

    // 1. Single-word query: "fox" hits every doc.
    let word = hash_tokens("fox");
    let hits: Vec<_> = idx.query_term_ids(&word[0]).collect();
    println!("single-word query 'fox':");
    for (d, s) in &hits {
        let text = corpus
            .iter()
            .find(|(i, _)| i == d)
            .map(|(_, t)| *t)
            .unwrap_or("?");
        println!("  {d}  score={s:.3}  {text}");
    }
    assert_eq!(hits.len(), 4);

    // 2. Phrase query: "quick brown" — one bigram. Matches docs
    //    that contain those two words in adjacent order.
    let phrase = bigram_tokens("quick brown");
    assert_eq!(phrase.len(), 1);
    let hits: Vec<_> = idx.query_term_ids(&phrase[0]).collect();
    println!("\nphrase query 'quick brown' (adjacent only):");
    for (d, s) in &hits {
        let text = corpus
            .iter()
            .find(|(i, _)| i == d)
            .map(|(_, t)| *t)
            .unwrap_or("?");
        println!("  {d}  score={s:.3}  {text}");
    }
    // doc 1: "...quick brown fox..." ✓
    // doc 4: "quick fox and brown dog" — quick and brown are
    //        NOT adjacent, so no match.
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, id(1));

    // 3. Phrase query: "brown fox" — matches docs 1 and 3.
    let phrase = bigram_tokens("brown fox");
    let hits: Vec<_> = idx.query_term_ids(&phrase[0]).collect();
    println!("\nphrase query 'brown fox':");
    for (d, s) in &hits {
        let text = corpus
            .iter()
            .find(|(i, _)| i == d)
            .map(|(_, t)| *t)
            .unwrap_or("?");
        println!("  {d}  score={s:.3}  {text}");
    }
    assert_eq!(hits.len(), 2);
    let ids: Vec<_> = hits.iter().map(|(d, _)| *d).collect();
    assert!(ids.contains(&id(1)));
    assert!(ids.contains(&id(3)));

    // 4. Longer phrase: two bigrams must both match. Combined
    //    BM25 score via query_multi.
    println!("\nlonger phrase query 'the quick brown' (chains two bigrams):");
    let mut multi = bigram_tokens("the quick brown");
    // Also include the words, so the single-word half
    // contributes weight.
    multi.extend(hash_tokens("the quick brown"));
    // Run through naive since SuccinctBM25Index doesn't yet expose
    // query_multi — sum per-term would produce the same ranking.
    let mut acc: std::collections::HashMap<Id, f32> = Default::default();
    for term in &multi {
        for (d, s) in idx.query_term_ids(term) {
            *acc.entry(d).or_default() += s;
        }
    }
    let mut ranked: Vec<_> = acc.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (d, s) in ranked.iter().take(3) {
        let text = corpus
            .iter()
            .find(|(i, _)| i == d)
            .map(|(_, t)| *t)
            .unwrap_or("?");
        println!("  {d}  score={s:.3}  {text}");
    }
    assert_eq!(
        ranked[0].0,
        id(1),
        "doc 1 has both 'the quick' AND 'quick brown' bigrams"
    );
}
