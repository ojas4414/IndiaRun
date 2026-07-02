"""Query-conditioned attention re-ranking over candidate sentences.

Architecture: this is the *re-rank* half of a standard retrieve-then-rerank
pipeline. FAISS gives us a shortlist via a single document-level embedding
(fast, high recall). Here we look *inside* each shortlisted profile at the
sentence level so a single strong sentence -- e.g. "built a hybrid retrieval
system serving 2M users" -- is not averaged away by document pooling.

The mechanism is scaled dot-product attention (Vaswani et al. 2017) used as
*cross*-attention:

    query  = a JD facet embedding                 (what we are looking for)
    keys   = candidate sentence embeddings         (the evidence)
    values = the same sentence similarities

    scores  = sentences . facet                    (cosine; embeddings are L2-normalized)
    weights = softmax(scores / temperature)        (the attention distribution)
    relevance(facet) = weights . scores            (attention-pooled evidence)

As temperature -> 0 this converges to hard MaxSim (ColBERT-style late
interaction); higher temperature averages across several supporting
sentences. The highest-attention sentence is surfaced as human-readable
evidence for the reasoning string.

The sentence embedder is transformer self-attention; this module adds the
cross-attention pooling layer on top of it.
"""
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Sentence boundaries: end punctuation, or the " | " / newline separators the
# profile text uses between roles and fields.
_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*\|\s*|\n+")
_MIN_LEN = 20      # drop fragments shorter than this many chars
_MAX_LEN = 320     # truncate very long sentences
_MAX_SENTENCES = 16  # cap per candidate to bound cost / stay deterministic


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    out, seen = [], set()
    for raw in _SPLIT.split(text):
        s = raw.strip()
        if len(s) < _MIN_LEN:
            continue
        s = s[:_MAX_LEN]
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def extract_sentences(candidate: Dict[str, Any]) -> List[str]:
    """Collect candidate evidence sentences from summary + role descriptions."""
    profile = candidate.get("profile", {})
    chunks: List[str] = []
    if profile.get("headline"):
        chunks.append(profile["headline"])
    if profile.get("summary"):
        chunks.append(profile["summary"])
    for role in candidate.get("career_history", []):
        desc = role.get("description", "")
        if desc:
            chunks.append(desc)

    sents: List[str] = []
    for c in chunks:
        sents.extend(split_sentences(c))
    return sents[:_MAX_SENTENCES]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def facet_attention(sent_embs: np.ndarray, facet: np.ndarray,
                    temperature: float = 0.1) -> Tuple[float, float, int]:
    """One facet against a candidate's sentences.

    Returns (attention_pooled_relevance, max_similarity, argmax_sentence_idx).
    `sent_embs` is (n, d) L2-normalized; `facet` is (d,) L2-normalized.
    """
    if sent_embs.shape[0] == 0:
        return 0.0, 0.0, -1
    sims = sent_embs @ facet                      # (n,) cosine similarities
    weights = _softmax(sims / max(temperature, 1e-6))
    attn = float(weights @ sims)                  # attention-pooled relevance
    idx = int(np.argmax(sims))
    return attn, float(sims[idx]), idx


class SentenceAttentionScorer:
    """Scores a candidate's textual evidence against the JD facets.

    The heavy sentence embedding is done in one batch via `score_batch`, so the
    cost is ~linear in total sentences across the shortlist (a couple of
    thousand candidates), well inside the ranking time budget.
    """

    def __init__(self, jd_requirements: Dict[str, Any], model=None,
                 temperature: float = 0.1):
        self.temperature = temperature
        facets = jd_requirements.get("evidence_facets", [])
        self.facet_names = [f["name"] for f in facets]
        self.facet_weights = np.array([f.get("weight", 1.0) for f in facets],
                                      dtype=np.float32)
        self._facet_queries = [f["query"] for f in facets]

        # Contrastive "anti-fit" facets: their attention is subtracted, so a
        # sentence that merely name-drops the right words (e.g. SEO articles
        # that "ranked in search") is cancelled out by a strong negative match.
        neg = jd_requirements.get("negative_facets", [])
        self.neg_names = [f["name"] for f in neg]
        self._neg_queries = [f["query"] for f in neg]

        self._model = model
        self._facet_embs: Optional[np.ndarray] = None
        self._neg_embs: Optional[np.ndarray] = None

    # -- model / facet setup ----------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        if self._facet_embs is None:
            self._facet_embs = np.asarray(
                self._model.encode(self._facet_queries, normalize_embeddings=True),
                dtype=np.float32,
            )
            if self._neg_queries:
                self._neg_embs = np.asarray(
                    self._model.encode(self._neg_queries, normalize_embeddings=True),
                    dtype=np.float32,
                )
            else:
                self._neg_embs = np.zeros((0, self._facet_embs.shape[1]), dtype=np.float32)

    # -- scoring -----------------------------------------------------------
    def score_batch(self, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Embed all sentences across the shortlist once, then compute per
        candidate: an aggregate facet score in [0, 1] and the best evidence
        sentence. Keyed by candidate_id."""
        self._ensure_model()

        # Flatten sentences, remembering ownership spans.
        all_sents: List[str] = []
        spans: List[Tuple[str, int, int, List[str]]] = []  # (cid, start, end, sents)
        for cand in candidates:
            sents = extract_sentences(cand)
            start = len(all_sents)
            all_sents.extend(sents)
            spans.append((cand.get("candidate_id"), start, len(all_sents), sents))

        if all_sents:
            embs = np.asarray(
                self._model.encode(all_sents, batch_size=256,
                                   normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
        else:
            embs = np.zeros((0, self._facet_embs.shape[1]), dtype=np.float32)

        results: Dict[str, Dict[str, Any]] = {}
        wsum = float(self.facet_weights.sum()) or 1.0
        for cid, start, end, sents in spans:
            sub = embs[start:end]
            results[cid] = self._score_one(sub, sents, wsum)
        return results

    def score_one(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience single-candidate path (used in tests / previews)."""
        self._ensure_model()
        sents = extract_sentences(candidate)
        if sents:
            embs = np.asarray(
                self._model.encode(sents, normalize_embeddings=True,
                                   show_progress_bar=False),
                dtype=np.float32,
            )
        else:
            embs = np.zeros((0, self._facet_embs.shape[1]), dtype=np.float32)
        wsum = float(self.facet_weights.sum()) or 1.0
        return self._score_one(embs, sents, wsum)

    def _score_one(self, sent_embs: np.ndarray, sents: List[str],
                   wsum: float) -> Dict[str, Any]:
        if sent_embs.shape[0] == 0 or self._facet_embs is None:
            return {"score": 0.0, "evidence": None, "evidence_facet": None,
                    "facet_scores": {}}

        # Strongest anti-fit (negative) facet attention, used as a margin.
        neg_max = 0.0
        if self._neg_embs is not None and self._neg_embs.shape[0] > 0:
            for j in range(self._neg_embs.shape[0]):
                attn_n, _, _ = facet_attention(sent_embs, self._neg_embs[j],
                                               self.temperature)
                neg_max = max(neg_max, attn_n)

        facet_scores: Dict[str, float] = {}
        best_facet_i, best_facet_val, best_sent_idx = -1, -1.0, -1
        weighted = 0.0
        for i, name in enumerate(self.facet_names):
            attn, maxsim, idx = facet_attention(sent_embs, self._facet_embs[i],
                                                self.temperature)
            # Margin: a facet only counts to the extent it beats the best
            # anti-fit match, so sentences that merely name-drop keywords
            # (and match a negative facet just as strongly) contribute nothing.
            contrib = max(0.0, attn - neg_max)
            facet_scores[name] = round(attn, 4)
            weighted += self.facet_weights[i] * contrib
            # Track the strongest facet (by weighted max similarity) for evidence.
            if maxsim * self.facet_weights[i] > best_facet_val:
                best_facet_val = maxsim * self.facet_weights[i]
                best_facet_i, best_sent_idx = i, idx

        # Map the margin-weighted relevance into [0, 1].
        score = max(0.0, min((weighted / wsum) / 0.15, 1.0))

        evidence = sents[best_sent_idx] if 0 <= best_sent_idx < len(sents) else None
        evidence_facet = self.facet_names[best_facet_i] if best_facet_i >= 0 else None
        return {"score": round(score, 4), "evidence": evidence,
                "evidence_facet": evidence_facet, "facet_scores": facet_scores}
