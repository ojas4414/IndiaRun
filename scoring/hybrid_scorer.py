import os
from typing import List, Dict, Any, Tuple
from embeddings.faiss_index import CandidateIndex
from skill_graph.bfs_scorer import BFSScorer
from trajectory.dp_aligner import TrajectoryScorer
from behavioral.signal_scorer import BehavioralScorer
from filters.disqualifiers import DisqualifierFilter
from security.injection_defense import InjectionDefender
from attention.sentence_attention import SentenceAttentionScorer
from twins.twin_detector import TwinDetector


class HybridScorer:
    """Retrieve-then-rerank ranker.

    Stage 1 (retrieval): FAISS ANN over a single document embedding -> wide
    shortlist. Stage 2 (rerank): deterministic feature scoring (skill graph,
    trajectory DP, behavioural signals, semantic sim) minus disqualifiers.
    Stage 3 (attention rerank): query-conditioned sentence attention over the
    strongest survivors, so a single strong sentence isn't averaged away.
    Stage 4 (twin disambiguation): near-duplicate "behavioral twin" profiles
    are clustered; the most available twin leads and the rest are demoted.
    """

    # Per-position demotion applied to behavioral twins after the leader.
    TWIN_DEMOTION = 0.05

    # Weight budget sums to 1.0. Tuned via evaluation/evaluate.py: trajectory is
    # the dominant JD-fit signal, so it carries the most weight; the semantic
    # term is largely a *recall* signal (its value is populating the funnel, not
    # final ordering), so it is down-weighted here; behavioural + attention are
    # kept as minority signals for availability and explainability, which the
    # rubric does not score but the JD explicitly requires.
    W_SKILL = 0.22
    W_TRAJ = 0.35
    W_BEHAV = 0.18
    W_SEM = 0.10
    W_ATTN = 0.15

    def __init__(self, jd_requirements: Dict[str, Any], index_path: str,
                 ids_path: str, jd_embedding_path: str,
                 use_attention: bool = True, attention_shortlist: int = 600,
                 dedupe_twins: bool = True):
        self.jd_reqs = jd_requirements

        self.index = CandidateIndex(dim=384)
        self.index.load(index_path, ids_path)

        import numpy as np
        self.jd_embedding = np.load(jd_embedding_path)

        self.skill_scorer = BFSScorer(jd_requirements)
        self.traj_scorer = TrajectoryScorer(jd_requirements)
        self.behav_scorer = BehavioralScorer(jd_requirements)
        self.disqualifiers = DisqualifierFilter(jd_requirements)
        self.defender = InjectionDefender()

        # Attention re-ranker (can be disabled for the time-tight fallback).
        env_off = os.environ.get("DISABLE_ATTENTION", "").lower() in ("1", "true", "yes")
        self.use_attention = use_attention and not env_off
        self.attention_shortlist = attention_shortlist
        self.attn_scorer = SentenceAttentionScorer(jd_requirements) if self.use_attention else None

        # Behavioral-twin disambiguation.
        self.dedupe_twins = dedupe_twins
        self.twin_detector = TwinDetector() if dedupe_twins else None
        self.twin_clusters_found = 0  # populated by score_all, for reporting

        # Populated by score_all: cid -> {evidence, evidence_facet, facet_scores}.
        self.attention_info: Dict[str, Dict[str, Any]] = {}

    def _score_semantic(self, top_k_retrieval: int = 2000) -> List[Tuple[str, float]]:
        return self.index.search(self.jd_embedding, top_k=top_k_retrieval)

    def score_all(self, candidates_dict: Dict[str, Dict[str, Any]],
                  top_k_retrieval: int = 2000) -> List[Tuple[str, float]]:
        """Return the top-100 (candidate_id, score), score-descending, ties
        broken by ascending candidate_id."""
        semantic_matches = self._score_semantic(top_k_retrieval=top_k_retrieval)

        # --- Stage 2: deterministic feature scoring over survivors ---------
        survivors: List[Dict[str, Any]] = []
        for cid, sem_score in semantic_matches:
            cand = candidates_dict.get(cid)
            if cand is None:
                continue

            penalty = self.disqualifiers.check_disqualifiers(cand)
            if penalty >= 5.0:  # hard disqualifier -> drop entirely
                continue

            behav = self.behav_scorer.score_behavioral(cand)
            base = (
                self.skill_scorer.score_skills(cand) * self.W_SKILL +
                self.traj_scorer.score_trajectory(cand) * self.W_TRAJ +
                behav * self.W_BEHAV +
                sem_score * self.W_SEM -
                penalty
            )
            survivors.append({"cid": cid, "cand": cand, "base": base,
                              "behav": behav, "attn": 0.0, "twin_penalty": 0.0})

        # --- Stage 3: attention re-rank the strongest survivors ------------
        if self.use_attention and survivors:
            survivors.sort(key=lambda r: r["base"], reverse=True)
            shortlist = survivors[: self.attention_shortlist]
            attn = self.attn_scorer.score_batch([r["cand"] for r in shortlist])
            for r in shortlist:
                info = attn.get(r["cid"], {})
                r["attn"] = info.get("score", 0.0)
                self.attention_info[r["cid"]] = info
        # survivors outside the shortlist keep attn = 0 (already lower base).

        # pre-twin fit score (used to order twins within a cluster)
        for r in survivors:
            r["fit"] = r["base"] + self.W_ATTN * r["attn"]

        # --- Stage 4: behavioral-twin disambiguation -----------------------
        if self.dedupe_twins and survivors:
            self._apply_twin_demotion(survivors)

        # --- Final scoring + deterministic ordering ------------------------
        # Round to the same 4 decimals the CSV writes, so the tie-break here
        # matches what the validator sees (two distinct floats that round to
        # the same string must be ordered by candidate_id ascending).
        scored = [(r["cid"], round(r["fit"] - r["twin_penalty"], 4))
                  for r in survivors]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:100]

    def _apply_twin_demotion(self, survivors: List[Dict[str, Any]]):
        """Cluster near-duplicate profiles; within a cluster the most available
        candidate leads and later twins get a graduated demotion so identical
        profiles do not stack in consecutive slots."""
        from collections import defaultdict
        clusters = self.twin_detector.cluster(survivors)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, cl in enumerate(clusters):
            groups[cl].append(i)

        self.twin_clusters_found = sum(1 for g in groups.values() if len(g) > 1)
        for idxs in groups.values():
            if len(idxs) <= 1:
                continue
            # Lead with the most available twin; tie-break by fit then id.
            idxs.sort(key=lambda i: (survivors[i]["behav"], survivors[i]["fit"],
                                     -sum(map(ord, survivors[i]["cid"]))),
                      reverse=True)
            for position, i in enumerate(idxs):
                survivors[i]["twin_penalty"] = self.TWIN_DEMOTION * position
