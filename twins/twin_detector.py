"""Behavioral-twin detection and disambiguation.

The challenge deliberately plants "behavioral twins": candidates who look
essentially identical on paper (same role, same experience, same skills) and
differ only in their behavioral signals. A ranker that scores paper features
will stack both twins together; the intended behaviour is to break the tie with
availability/engagement and to avoid flooding the top-100 with near-duplicates.

We detect twins by a two-level scheme:
  1. Coarse bucket by (role category, rounded years of experience) — cheap.
  2. Within a bucket, merge candidates whose canonical skill sets have
     Jaccard >= threshold into the same twin cluster.

Disambiguation is then applied by the ranker: within a cluster the most
available candidate (behavioural score) leads, and the remaining twins receive
a graduated demotion (an MMR-style diversity penalty) so identical profiles do
not occupy consecutive slots.
"""
from typing import List, Dict, Any, FrozenSet

from trajectory.dp_aligner import _title_token
from skill_graph.graph_builder import normalize_skill


class TwinDetector:
    def __init__(self, jaccard_threshold: float = 0.85):
        self.jac = jaccard_threshold

    def _skillset(self, cand: Dict[str, Any]) -> FrozenSet[str]:
        return frozenset(
            c for c in (normalize_skill(s.get("name", ""))
                        for s in cand.get("skills", [])) if c
        )

    def _bucket_key(self, cand: Dict[str, Any]):
        p = cand.get("profile", {})
        return (_title_token(p.get("current_title", "")),
                round(p.get("years_of_experience", 0) or 0))

    @staticmethod
    def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def cluster(self, candidates: List[Dict[str, Any]]) -> List[int]:
        """Assign each candidate a cluster id. Candidates sharing a cluster id
        (with more than one member) are behavioral twins.

        `candidates` is a list of records each holding a 'cand' dict.
        """
        buckets: Dict[Any, List[int]] = {}
        for i, rec in enumerate(candidates):
            buckets.setdefault(self._bucket_key(rec["cand"]), []).append(i)

        cluster_of = [-1] * len(candidates)
        next_id = 0
        for _, idxs in buckets.items():
            sets = [self._skillset(candidates[i]["cand"]) for i in idxs]
            local = [-1] * len(idxs)
            for a in range(len(idxs)):
                if local[a] != -1:
                    continue
                local[a] = next_id
                for b in range(a + 1, len(idxs)):
                    if local[b] == -1 and self._jaccard(sets[a], sets[b]) >= self.jac:
                        local[b] = next_id
                next_id += 1
            for pos, i in enumerate(idxs):
                cluster_of[i] = local[pos]
        return cluster_of
