# Approach — AI Candidate-Ranking System

*One-page walkthrough for reviewers. Deeper math is in [docs/attention.md](docs/attention.md);
numbers are in [evaluation/report.md](evaluation/report.md) and
[evaluation/adversarial_report.md](evaluation/adversarial_report.md).*

## Thesis: the LLM should understand, never decide

Letting an LLM rank 100k candidates is non-deterministic, slow, and can't run
offline in 5 minutes on CPU. So we split **understanding** (offline, embeddings)
from **ranking** (deterministic scoring). `rank.py` makes zero LLM/network calls
and produces byte-identical output on identical input.

## Pipeline (retrieve → rerank → disambiguate)

```
candidates.jsonl (100k)
   │  precompute (no time limit): honeypot filter → sentence-transformer embeddings → FAISS
   ▼
Stage 1  RETRIEVAL      FAISS ANN over one doc embedding → top-2000 funnel (recall)
Stage 2  FEATURE RERANK skill-graph + trajectory-DP + behavioral + semantic − disqualifiers
Stage 3  ATTENTION      query-conditioned sentence attention over top-600 (precision + evidence)
Stage 4  TWINS          cluster near-duplicate profiles; most-available twin leads
   ▼
top-100 → deterministic reasoning → submission.csv (≤5 min, CPU, offline)
```

## How each trap in the brief is handled

| Trap (from the brief) | Defense |
|---|---|
| **Keyword stuffing** | Skill-graph + trajectory carry 57% of weight; semantic/keywords are minority. Off-domain titles are hard-disqualified. Verified by the adversarial test. |
| **Plain-language Tier-5s** (real fits, no buzzwords) | Semantic retrieval + sentence attention surface production evidence even without keyword matches. |
| **Behavioral twins** (identical on paper) | Stage-4 twin clustering breaks the tie by availability and demotes duplicates. |
| **~80 honeypots** (impossible profiles) | Date/tenure impossibility checks filter them at precompute (52 caught; 0 in top-100, well under the 10% DQ bar). |
| **Consulting-only / CV-speech-only / title-chasers / no-recent-code** | Explicit disqualifier rules from the JD. |

## Two things that make this more than a cosine-similarity submission

1. **Contrastive-facet attention.** Sentence-level attention alone is fooled by
   sentence-level keyword stuffing ("SEO articles that ranked in search"). We add
   anti-fit facets and a margin, so evidence that matches a negative facet as
   strongly as a positive one nets to zero. (We found this failure empirically —
   see the commit history / docs.)
2. **We measured it.** An offline harness reports NDCG/MRR/MAP vs. a transparent
   recruiter rubric **and** label-free trap metrics, plus a per-component
   ablation and an **adversarial keyword-stuffing** test. The weights were tuned
   from this harness, not guessed — e.g. we found semantic similarity is a
   *recall* signal (near-inert in final ordering) and shifted its weight to the
   trajectory signal that the ablation proved dominant.

## Guarantees

- **Deterministic** — `make determinism` ranks twice and diffs (identical).
- **Offline** — the model is baked into the Docker image; ranking sets
  `TRANSFORMERS_OFFLINE=1`.
- **In budget** — ranks the full 100k in ~40s on CPU (limit: 5 min).
- **Reproducible** — `docker compose up --build` runs precompute → rank → validate.

## Honest limitations

- The `gold.py` rubric is weak supervision and partly overlaps with the ranker's
  own features, so NDCG/MRR should be read as *agreement with an explicit
  rubric*, not an oracle. The label-free metrics are the load-bearing evidence.
- Attention contributes only 15% of the score; its main value is the evidence
  sentence in each reasoning, not raw NDCG. This is a deliberate trade for
  explainability + availability, which the rubric does not score but the JD
  requires.
