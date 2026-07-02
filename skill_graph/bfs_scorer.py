from collections import deque
from typing import Dict, List, Any, Set, Tuple
from .graph_builder import SkillGraph, normalize_skill


class BFSScorer:
    def __init__(self, jd_requirements: Dict[str, Any]):
        self.graph = SkillGraph()
        # Normalize the JD skills to canonical nodes once, up front.
        self.req_skills = self._norm_list(jd_requirements.get("required_skills", []))
        self.pref_skills = self._norm_list(jd_requirements.get("preferred_skills", []))

    @staticmethod
    def _norm_list(items: List[str]) -> List[str]:
        seen, out = set(), []
        for it in items:
            c = normalize_skill(it)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def score_skills(self, candidate: Dict[str, Any]) -> float:
        """Transferability score from candidate skills to JD skills via
        exact-on-canonical match + bounded BFS over adjacent skills.

        Required skill:  dist 0 -> 1.0, dist 1 -> 0.5, dist 2 -> 0.25
        Preferred skill: dist 0 -> 0.5, dist 1 -> 0.25

        A small boost is added for JD skills the candidate has *verified* via
        a high Redrob skill-assessment score.
        """
        cand_canon = self._candidate_canon(candidate)
        if not cand_canon:
            return 0.0

        max_possible = len(self.req_skills) * 1.0 + len(self.pref_skills) * 0.5
        if max_possible == 0:
            return 1.0

        assessed = self._assessed_canon(candidate)  # canonical -> score(0..100)

        total = 0.0
        for rs in self.req_skills:
            dist = self._bfs_min_distance(cand_canon, rs)
            if dist == 0:
                total += 1.0
            elif dist == 1:
                total += 0.5
            elif dist == 2:
                total += 0.25
            # Verified-skill boost: reward assessment scores > 70 on a match
            if dist == 0 and assessed.get(rs, 0) > 70:
                total += 0.25

        for ps in self.pref_skills:
            dist = self._bfs_min_distance(cand_canon, ps)
            if dist == 0:
                total += 0.5
            elif dist == 1:
                total += 0.25
            if dist == 0 and assessed.get(ps, 0) > 70:
                total += 0.15

        return min(total / max_possible, 1.0)

    @staticmethod
    def _candidate_canon(candidate: Dict[str, Any]) -> Set[str]:
        out = set()
        for s in candidate.get("skills", []):
            c = normalize_skill(s.get("name", ""))
            if c:
                out.add(c)
        return out

    @staticmethod
    def _assessed_canon(candidate: Dict[str, Any]) -> Dict[str, float]:
        scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
        out: Dict[str, float] = {}
        for name, val in scores.items():
            c = normalize_skill(name)
            if c:
                out[c] = max(out.get(c, 0.0), float(val))
        return out

    def _bfs_min_distance(self, start_nodes: Set[str], target: str) -> int:
        """Minimum graph distance from any candidate skill to the target,
        capped at 2. Returns 999 if unreachable within 2 hops."""
        if target in start_nodes:
            return 0

        queue = deque((n, 0) for n in start_nodes)
        visited = set(start_nodes)
        while queue:
            node, dist = queue.popleft()
            if dist >= 2:
                continue
            for nb in self.graph.get_neighbors(node):
                if nb == target:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return 999
