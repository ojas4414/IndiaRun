from typing import Dict, Any, List

# Role categories used for sequence alignment. We collapse noisy free-text
# titles onto a small alphabet so we can align a candidate's career path
# against the ideal growth trajectory toward a senior AI/ML role.
_AI_KW = ("ai engineer", "ml engineer", "machine learning", "applied scientist",
          "ranking engineer", "search engineer", "nlp engineer", "research engineer",
          "ai specialist", "deep learning")
_DATA_KW = ("data engineer", "data scientist", "analytics engineer", "data analyst",
            "analytics")
_SWE_KW = ("software engineer", "backend", "front end", "frontend", "full stack",
           "fullstack", "developer", "devops", "cloud engineer", "platform engineer",
           "sre", "qa engineer", "mobile developer")


def _title_token(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in _AI_KW):
        return "ai"
    if any(k in t for k in _DATA_KW):
        return "data"
    if any(k in t for k in _SWE_KW):
        return "swe"
    return "irrelevant"


# The ideal progression: engineering foundation -> data/ML -> AI/ML specialist.
_IDEAL_PATH = ["swe", "data", "ai"]

# Substitution scores for aligning a candidate token (rows) to an ideal token.
_SUB = {
    ("ai", "ai"): 2, ("data", "data"): 2, ("swe", "swe"): 2,
    ("ai", "data"): 1, ("data", "ai"): 1,
    ("data", "swe"): 1, ("swe", "data"): 1,
    ("ai", "swe"): 0, ("swe", "ai"): 0,
}
_GAP = -1
_IRRELEVANT = -1  # aligning an irrelevant role against anything


def _sub_score(a: str, b: str) -> int:
    if a == "irrelevant" or b == "irrelevant":
        return _IRRELEVANT
    return _SUB.get((a, b), 0)


def _needleman_wunsch(seq: List[str], ideal: List[str]) -> int:
    """Classic global alignment DP. Returns the optimal alignment score."""
    n, m = len(seq), len(ideal)
    # dp[i][j] = best score aligning seq[:i] with ideal[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + _sub_score(seq[i - 1], ideal[j - 1])
            up = dp[i - 1][j] + _GAP
            left = dp[i][j - 1] + _GAP
            dp[i][j] = max(diag, up, left)
    return dp[n][m]


class TrajectoryScorer:
    def __init__(self, jd_requirements: Dict[str, Any]):
        self.jd_reqs = jd_requirements
        self.ideal_roles = [r.lower() for r in jd_requirements.get("ideal_roles", [])]
        self.invalid_roles = [r.lower() for r in jd_requirements.get("invalid_roles", [])]

    def _ordered_titles(self, career: List[Dict[str, Any]]) -> List[str]:
        """Career titles ordered oldest -> newest by start_date."""
        def key(role):
            return role.get("start_date") or ""
        return [r.get("title", "") for r in sorted(career, key=key)]

    def score_trajectory(self, candidate: Dict[str, Any]) -> float:
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        current_title = profile.get("current_title", "").lower()

        # Hard gate: a clearly off-domain current role earns nothing here.
        if any(ir in current_title for ir in self.invalid_roles):
            return 0.0
        if not career:
            return 0.1

        # 1) DP sequence alignment (0.40) -------------------------------
        tokens = [_title_token(t) for t in self._ordered_titles(career)]
        raw = _needleman_wunsch(tokens, _IDEAL_PATH)
        # Normalize: best achievable is 2 per ideal step; worst is bounded below.
        best = 2 * len(_IDEAL_PATH)
        worst = _GAP * (len(tokens) + len(_IDEAL_PATH))
        align = (raw - worst) / (best - worst) if best != worst else 0.0
        align = max(0.0, min(align, 1.0))
        score = 0.40 * align

        # 2) Current-title fit (0.30) -----------------------------------
        if any(ir in current_title for ir in self.ideal_roles):
            score += 0.30
        elif _title_token(current_title) in ("ai", "data"):
            score += 0.20
        elif "engineer" in current_title or "developer" in current_title:
            score += 0.10

        # 3) YOE alignment (0.20) ---------------------------------------
        yoe = profile.get("years_of_experience", 0)
        ideal_min, ideal_max = self.jd_reqs.get("ideal_experience", (6, 8))
        acc_min, acc_max = self.jd_reqs.get("experience_range", (5, 9))
        if ideal_min <= yoe <= ideal_max:
            score += 0.20
        elif acc_min <= yoe <= acc_max:
            score += 0.10

        # 4) Production signals (0.10) ----------------------------------
        prod_kw = ("shipped", "deployed", "production", "at scale", "latency",
                   "throughput", "serving", "infrastructure")
        if any(any(pk in (r.get("description", "").lower()) for pk in prod_kw) for r in career):
            score += 0.10

        # 5) Title-chaser penalty ---------------------------------------
        # Frequent short stints (avg tenure < 18 months across >=3 roles).
        durations = [r.get("duration_months", 0) for r in career if r.get("duration_months")]
        if len(durations) >= 3:
            avg_tenure = sum(durations) / len(durations)
            if avg_tenure < 12:
                score -= 0.20
            elif avg_tenure < 18:
                score -= 0.10

        return max(0.0, min(score, 1.0))

    def is_title_chaser(self, candidate: Dict[str, Any]) -> bool:
        career = candidate.get("career_history", [])
        durations = [r.get("duration_months", 0) for r in career if r.get("duration_months")]
        if len(durations) >= 3:
            return (sum(durations) / len(durations)) < 18
        return False
