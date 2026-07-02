"""Standard offline ranking metrics: NDCG@k, MRR, MAP@k.

`ranking` is a list of relevance grades in ranked order (index 0 = rank 1).
`relevant` treats grade >= `rel_threshold` as a relevant hit.
"""
from typing import List
import math


def dcg_at_k(gains: List[float], k: int) -> float:
    total = 0.0
    for i, g in enumerate(gains[:k]):
        total += (2 ** g - 1) / math.log2(i + 2)  # discount log2(rank+1)
    return total


def ndcg_at_k(ranking: List[float], k: int) -> float:
    """Normalized DCG. `ranking` = relevance grades in the produced order."""
    ideal = sorted(ranking, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranking, k) / idcg


def mrr(ranking: List[float], rel_threshold: float = 2.0) -> float:
    """Reciprocal rank of the first relevant item."""
    for i, g in enumerate(ranking):
        if g >= rel_threshold:
            return 1.0 / (i + 1)
    return 0.0


def map_at_k(ranking: List[float], k: int, rel_threshold: float = 2.0) -> float:
    """Mean average precision for a single ranking (i.e. average precision)."""
    hits, precisions = 0, []
    for i, g in enumerate(ranking[:k]):
        if g >= rel_threshold:
            hits += 1
            precisions.append(hits / (i + 1))
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)
