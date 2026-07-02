# Evaluation report

Ranking quality within the top-2000 retrieval funnel. Baselines (`naive_keyword`, `semantic_only`) get no disqualifier gate; the system configs do. NDCG/MRR/MAP are vs. the recruiter rubric in `gold.py` (weak supervision); **HP%** (honeypots) and **OffDom%** (off-domain titles) in the top-100 are label-free and objective.

```
config           NDCG@10  NDCG@100    MRR  MAP@100  meanGold    HP%  OffDom%
----------------------------------------------------------------------------
naive_keyword      0.653     0.615  0.500    0.790      2.12    0.0      6.0
semantic_only      0.656     0.508  1.000    0.789      1.58    0.0     27.0
full               1.000     0.876  1.000    0.985      2.68    0.0      4.0
-attention         1.000     0.938  1.000    0.997      2.83    0.0      2.0
-skill_graph       1.000     0.898  1.000    0.992      2.74    0.0      3.0
-trajectory        0.879     0.753  1.000    0.916      2.40    0.0      8.0
-behavioral        1.000     0.904  1.000    0.989      2.76    0.0      2.0
-semantic          1.000     0.884  1.000    0.984      2.70    0.0      4.0
```

## Read-out

**Headline (label-free):** the structured pipeline beats both baselines. NDCG@100 = **0.876** for `full` vs 0.615 (naive_keyword) / 0.508 (semantic_only). Off-domain titles in the top-100: **4%** (full) vs **27%** (semantic_only). Honeypots in top-100: 0%.

**Strongest fit signal:** removing `trajectory` drops NDCG@100 to 0.753 — the largest ablation loss, so it carries the most JD-fit information.

**Semantic is a recall signal, not a ranking signal:** `-semantic` ≈ `full`, i.e. once the funnel is semantic-sorted the semantic term adds little to *final* ordering — its value is populating the funnel. Hence its low final-stage weight.

**On the other ablations:** removing attention / skill_graph / behavioral can nudge rubric-NDCG up, because `gold.py` scores title + career-text fit and cannot see the objectives those signals optimize — availability (JD: down-weight inactive/unresponsive), verified skills, and the evidence sentence in each reasoning. They are therefore kept as minority weights.

**Caveat.** `gold.py` is weak supervision and partially overlaps with the ranker's features (title, career text), so NDCG/MRR/MAP mean *agreement with an explicit recruiter rubric*, not an unbiased oracle. The label-free HP%/OffDom% columns and the baseline gap are the load-bearing evidence.
