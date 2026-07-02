# Sentence-level attention re-ranking

This document explains the query-conditioned attention layer in
`attention/sentence_attention.py` — what it does, the math, and the design
decisions behind it. It's written so the design can be defended end to end.

## Why it exists

The retrieval stage embeds each candidate profile into **one** vector and does
FAISS ANN search against the JD vector. That's fast and high-recall, but a
single document vector is a *mean* of everything in the profile — so one strong
sentence ("built a hybrid retrieval system serving 2M users") gets averaged in
with paragraphs of unrelated text. The signal we care about most is diluted.

The fix is the standard **retrieve-then-rerank** pattern: keep the cheap
document-level retrieval for recall, then look *inside* each shortlisted
profile at the sentence level for precision.

## The mechanism: cross-attention pooling

The sentence embedder (`all-MiniLM-L6-v2`) already uses transformer
**self-attention** internally to produce each sentence embedding. On top of
that we add a **cross-attention** pooling layer, using scaled dot-product
attention (Vaswani et al., 2017) with the JD facet as the query:

```
query  q  = a JD facet embedding          (what we're looking for)
keys   K  = candidate sentence embeddings  (n × d)
values    = the sentence similarities themselves

scores  = K · q                 # cosine sim (all embeddings are L2-normalized)
weights = softmax(scores / τ)   # the attention distribution over sentences
relevance(facet) = weights · scores   # attention-pooled evidence
```

- `τ` (temperature) controls sharpness. As `τ → 0` this converges to hard
  **MaxSim** — the ColBERT late-interaction operator (take the single best
  sentence). Larger `τ` averages over several supporting sentences. We use
  `τ = 0.1`, close to MaxSim but slightly smoothed.
- The **argmax** sentence per facet is surfaced as human-readable *evidence* in
  the reasoning string — the exact sentence the layer "fired on".

## JD facets

Instead of one JD query we use a handful of **facets** (in
`jd_requirements.json → evidence_facets`), each a natural-language description
of one thing the role needs: production retrieval, vector-DB ops, ranking
evaluation, shipped-at-scale, LLM fine-tuning, distributed systems. Each facet
is scored independently and weighted; the "must-have" facets weigh more than
the "nice-to-have" ones. This mirrors how the JD itself is structured
("things you absolutely need" vs "things we'd like").

## Contrastive facets (the important part)

**Failure mode we found:** attention alone is fooled by sentence-level keyword
stuffing. A *Marketing Manager* whose profile says *"wrote SEO articles that
ranked on the first page of search for AI/ML topics"* scores highly on the
retrieval facet — the sentence is genuinely *semantically* near "search /
ranking", it just isn't real retrieval engineering.

**Fix:** contrastive (anti-fit) facets in `negative_facets` — marketing/SEO
content, pure research, framework tutorials, generic admin. Each facet's
contribution is a **margin** over the strongest negative match:

```
contrib(facet) = max(0, attn_pos(facet) − max_j attn_neg(j))
score = Σ w_i · contrib_i  /  Σ w_i     (then scaled into [0, 1])
```

So a sentence that matches a positive facet *and* an anti-fit facet equally
nets to zero. Only evidence that is closer to the role than to the anti-fit
patterns counts. This is a small contrastive-learning idea applied at scoring
time, and it directly targets the "keyword trap" the challenge is built around.

## Where it sits in the pipeline

1. FAISS retrieves the top ~2000 by document similarity.
2. Hard disqualifiers (off-domain title, consulting-only, CV/speech-only, …)
   drop candidates **before** attention runs — so the anti-fit roles never even
   reach this layer.
3. Deterministic feature scoring (skill graph, trajectory DP, behavioural
   signals, semantic sim) ranks the survivors.
4. Attention re-ranks the **top ~600 survivors** (bounds the sentence-embedding
   cost to a few thousand sentences — well inside the 5-minute budget).
5. Attention contributes **15%** of the final score. It's deliberately a
   *minority* signal: the skill graph and trajectory are the primary keyword-trap
   defenses; attention refines ordering and, above all, provides concrete
   evidence for the reasoning.

## Properties

- **Deterministic.** CPU inference on a fixed model + fixed math ⇒ identical
  output on identical input.
- **Offline.** The model is baked into the Docker image at build time; no
  network at rank time.
- **Bounded cost.** Only the top ~600 survivors are sentence-embedded, capped at
  16 sentences each.
- **Fail-safe.** Attention only *adds* a bonus; if it's disabled
  (`DISABLE_ATTENTION=1`) or scores zero, the ranking degrades gracefully to the
  feature-based order.
