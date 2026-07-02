from typing import Dict, Any
from skill_graph.graph_builder import normalize_skill

# Skills that signal a pure CV / speech / robotics focus (a JD disqualifier).
_CV_SPEECH = {
    "image classification", "object detection", "yolo", "cnn", "opencv",
    "computer vision", "gans", "diffusion models", "speech recognition", "asr",
    "tts", "reinforcement learning", "robotics",
}
# The retrieval / ranking / LLM canonical nodes this role actually wants.
_CORE = {"embeddings", "sentence-transformers", "information-retrieval", "vector-search",
         "semantic-search", "vector-database", "faiss", "rag", "hybrid-search",
         "ranking-evaluation", "learning-to-rank", "llm", "fine-tuning"}


class DisqualifierFilter:
    def __init__(self, jd_requirements: Dict[str, Any]):
        self.dq = jd_requirements.get("disqualifiers", {})
        self.consulting_firms = [f.lower() for f in self.dq.get("consulting_only", [])]
        self.invalid_roles = [r.lower() for r in jd_requirements.get("invalid_roles", [])]

    def check_disqualifiers(self, candidate: Dict[str, Any]) -> float:
        """Penalty to subtract. Values >= 5 are treated as hard disqualifiers
        by the hybrid scorer."""
        penalty = 0.0
        career = candidate.get("career_history", [])
        profile = candidate.get("profile", {})
        current_title = profile.get("current_title", "").lower()

        if not career:
            return 10.0

        # 1) Off-domain current role.
        if any(ir in current_title for ir in self.invalid_roles):
            return 10.0

        # 2) Consulting-only career.
        if self.consulting_firms:
            if all(any(cf in (r.get("company", "").lower()) for cf in self.consulting_firms)
                   for r in career):
                penalty += 10.0

        # 3) Title-chaser: frequent short stints.
        if self.dq.get("title_chaser_pattern", False):
            durs = [r.get("duration_months", 0) for r in career if r.get("duration_months")]
            if len(durs) >= 3 and (sum(durs) / len(durs)) < 18:
                penalty += 2.0

        # 4) Pure CV / speech / robotics with no retrieval/ranking depth.
        if self.dq.get("cv_speech_robotics_only", False):
            canon = {normalize_skill(s.get("name", "")) for s in candidate.get("skills", [])}
            assessed = {a.strip().lower()
                        for a in candidate.get("redrob_signals", {})
                                          .get("skill_assessment_scores", {})}
            cv_focus = any(a in _CV_SPEECH for a in assessed) or bool(canon & _CV_SPEECH)
            if cv_focus and not (canon & _CORE):
                penalty += 5.0

        # 5) Pure research without production.
        if self.dq.get("pure_research_no_production", False):
            if "research" in current_title and "engineer" not in current_title:
                prod = ("shipped", "deployed", "production")
                if not any(any(pk in (r.get("description", "").lower()) for pk in prod)
                           for r in career):
                    penalty += 5.0

        # 6) Senior/architect with no recent hands-on code (soft).
        if self.dq.get("no_recent_code", False):
            if "architect" in current_title or "director" in current_title:
                penalty += 0.5

        return penalty
