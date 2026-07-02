"""A transparent 'recruiter rubric' for weak-supervision relevance labels.

There is no ground-truth label set shipped with the challenge, so we encode
the JD's own hiring rules as a labelling function and grade each candidate
0-3. This is *weak supervision*: it is imperfect and partially overlaps with
the ranker's own features (title, career text), so NDCG/MRR against it should
be read as "how well does the ranking agree with an explicit recruiter rubric"
-- not as an unbiased oracle. The label-free trap metrics in evaluate.py
(honeypot %, off-domain %) are the more objective signals.

Grades:
  0  disqualified / not a fit (honeypot, off-domain title, consulting-only)
  1  adjacent (engineering/data background, weak domain evidence)
  2  relevant (AI/ML role or clear production retrieval/ranking evidence)
  3  strong (relevant role AND shipped-to-production retrieval/ranking + in band)
"""
from typing import Dict, Any

from trajectory.dp_aligner import _title_token

_PROD_EVIDENCE = ("retrieval", "ranking", "recommend", "embedding", "semantic search",
                  "vector", "ndcg", "relevance", "learning to rank", "search")
_SHIPPED = ("production", "deployed", "shipped", "users", "at scale", "latency",
            "serving", "a/b", "throughput", "real-time")


def _profile_text(candidate: Dict[str, Any]) -> str:
    parts = [candidate.get("profile", {}).get("summary", "")]
    for r in candidate.get("career_history", []):
        parts.append(r.get("description", ""))
    return " ".join(parts).lower()


def gold_relevance(candidate: Dict[str, Any], jd: Dict[str, Any],
                   is_honeypot: bool, disq_penalty: float) -> int:
    # Hard-zero cases.
    if is_honeypot or disq_penalty >= 5.0:
        return 0

    title = candidate.get("profile", {}).get("current_title", "").lower()
    if any(ir.lower() in title for ir in jd.get("invalid_roles", [])):
        return 0

    text = _profile_text(candidate)
    has_domain = any(kw in text for kw in _PROD_EVIDENCE)
    has_shipped = any(kw in text for kw in _SHIPPED)
    prod_evidence = has_domain and has_shipped

    tok = _title_token(title)
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    in_band = 5 <= yoe <= 9

    # Base grade from role category.
    if tok == "ai":
        grade = 2
    elif tok == "data":
        grade = 1
    elif tok == "swe":
        grade = 1 if has_domain else 0
    else:
        grade = 0

    # Production retrieval/ranking evidence bumps up.
    if prod_evidence and grade >= 1:
        grade += 1
    # Out-of-band experience is a soft negative.
    if not in_band and grade > 0:
        grade -= 1

    return max(0, min(grade, 3))
