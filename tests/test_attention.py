import numpy as np

from attention.sentence_attention import (
    split_sentences, extract_sentences, facet_attention, _softmax,
)


def test_softmax_sums_to_one():
    w = _softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[2] > w[0]  # larger logit -> larger weight


def test_split_sentences_filters_and_dedupes():
    text = "Built retrieval systems. short. | Built retrieval systems. | Shipped a ranking model to production users."
    s = split_sentences(text)
    assert "short." not in s                      # too short, dropped
    assert len(s) == 2                             # duplicate collapsed
    assert any("ranking model" in x for x in s)


def test_extract_sentences_pulls_from_summary_and_roles():
    cand = {
        "profile": {"headline": "Senior ML Engineer building search",
                    "summary": "I built an embeddings retrieval pipeline serving millions of users."},
        "career_history": [
            {"description": "Designed the NDCG evaluation harness and ran A/B tests on ranking changes."},
        ],
    }
    sents = extract_sentences(cand)
    assert any("retrieval" in s.lower() for s in sents)
    assert any("ndcg" in s.lower() for s in sents)


def test_facet_attention_prefers_relevant_sentence():
    # Facet query vector; three sentence vectors, the last one aligned with it.
    facet = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sent_embs = np.array([
        [0.0, 1.0, 0.0],   # irrelevant
        [0.0, 0.0, 1.0],   # irrelevant
        [1.0, 0.0, 0.0],   # exact match
    ], dtype=np.float32)
    attn, maxsim, idx = facet_attention(sent_embs, facet, temperature=0.1)
    assert idx == 2                    # picks the aligned sentence as evidence
    assert maxsim > 0.99
    assert attn > 0.5                  # attention concentrates on the match


def test_facet_attention_empty():
    facet = np.array([1.0, 0.0], dtype=np.float32)
    attn, maxsim, idx = facet_attention(np.zeros((0, 2), dtype=np.float32), facet)
    assert attn == 0.0 and idx == -1
