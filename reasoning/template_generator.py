from typing import Dict, Any, List
from skill_graph.graph_builder import normalize_skill

# Canonical nodes grouped by the JD theme they speak to, so we can name the
# right part of the job when a candidate matches.
_RETRIEVAL = {"embeddings", "sentence-transformers", "openai-embeddings", "bge", "e5",
              "vector-search", "semantic-search", "information-retrieval", "hybrid-search",
              "rag", "vector-database", "faiss", "pinecone", "weaviate", "qdrant",
              "milvus", "pgvector", "chroma", "elasticsearch", "opensearch"}
_RANKING = {"ranking-evaluation", "ndcg", "mrr", "map", "learning-to-rank",
            "neural-ranking", "a-b-testing", "offline-online-correlation"}
_LLM = {"llm", "fine-tuning", "lora", "qlora", "peft", "langchain", "llamaindex",
        "prompt-engineering"}


def _pick(pool: List[str], seed: int) -> str:
    return pool[seed % len(pool)]


class ReasoningGenerator:
    def __init__(self, jd_requirements: Dict[str, Any]):
        self.jd_reqs = jd_requirements
        self.notice_pref = jd_requirements.get("notice_period_preference_days", 30)
        self.locations = [l.lower() for l in jd_requirements.get("location_preference", [])]
        self.req_canon = {normalize_skill(s) for s in jd_requirements.get("required_skills", [])}
        self.pref_canon = {normalize_skill(s) for s in jd_requirements.get("preferred_skills", [])}
        self.ideal_roles = [r.lower() for r in jd_requirements.get("ideal_roles", [])]

    # -- fact extraction ---------------------------------------------------
    def _matched_skills(self, candidate: Dict[str, Any]):
        """Return (display_names, matched_canon_set) for JD-relevant skills the
        candidate actually lists, preserving their own display spelling."""
        names, canon = [], set()
        for s in candidate.get("skills", []):
            disp = s.get("name", "")
            c = normalize_skill(disp)
            if c and (c in self.req_canon or c in self.pref_canon):
                if c not in canon:
                    canon.add(c)
                    names.append(disp)
        return names, canon

    def _theme(self, canon: set) -> str:
        if canon & _RETRIEVAL:
            return "embeddings & retrieval"
        if canon & _RANKING:
            return "ranking & evaluation"
        if canon & _LLM:
            return "LLM fine-tuning"
        return "AI/ML"

    def _concerns(self, candidate, matched_canon) -> List[str]:
        profile = candidate.get("profile", {})
        sig = candidate.get("redrob_signals", {})
        out = []

        notice = sig.get("notice_period_days")
        if notice is not None and notice > self.notice_pref:
            out.append(f"notice period is {notice} days (prefers <={self.notice_pref})")

        rr = sig.get("recruiter_response_rate")
        if rr is not None and rr < 0.4:
            out.append(f"low recruiter response rate ({rr:.0%})")

        loc = profile.get("location", "")
        if loc and self.locations and not any(l in loc.lower() for l in self.locations):
            tail = "open to relocating" if sig.get("willing_to_relocate") else "relocation unconfirmed"
            out.append(f"based in {loc} ({tail})")

        title = profile.get("current_title", "").lower()
        if not any(ir in title for ir in self.ideal_roles):
            out.append("current title isn't a direct AI/ML role, so relevance rests on skills")

        missing_req = self.req_canon - matched_canon - {None}
        if len(missing_req) >= max(1, int(0.6 * len(self.req_canon))):
            out.append("several core retrieval/eval skills aren't evidenced")

        return out

    @staticmethod
    def _clean_evidence(text: str, limit: int = 160) -> str:
        t = " ".join(text.split())            # collapse whitespace/newlines
        if len(t) > limit:
            t = t[:limit].rsplit(" ", 1)[0] + "..."
        return t.strip(' "')

    # -- main --------------------------------------------------------------
    def generate(self, candidate: Dict[str, Any], score: float, rank: int,
                 evidence: str = None) -> str:
        profile = candidate.get("profile", {})
        sig = candidate.get("redrob_signals", {})
        title = profile.get("current_title", "Professional")
        yoe = profile.get("years_of_experience", 0) or 0

        names, canon = self._matched_skills(candidate)
        theme = self._theme(canon)
        seed = sum(ord(c) for c in candidate.get("candidate_id", "")) + rank

        # Opener varies by rank band and candidate seed.
        if rank <= 10:
            opener = _pick(["Top-tier fit", "Standout candidate", "Excellent match",
                            "Strong hire signal"], seed)
        elif rank <= 50:
            opener = _pick(["Solid fit", "Credible match", "Promising profile",
                            "Good alignment"], seed)
        else:
            opener = _pick(["Worth a look", "Borderline fit", "Possible fit",
                            "Depth option"], seed)

        # Experience phrasing varies.
        exp = _pick([
            f"{title} with {yoe:.1f} yrs experience",
            f"currently {title} ({yoe:.1f} yrs in the field)",
            f"{yoe:.1f} years of work as {title}",
            f"{title}, ~{yoe:.0f} yrs experience",
        ], seed + 1)

        parts = [f"{opener} — {exp}."]

        # Specific skills + JD connection.
        if names:
            shown = ", ".join(names[:4])
            connect = _pick([
                f"Hands-on with {shown}, mapping onto the role's {theme} work.",
                f"Directly relevant skills ({shown}) line up with the JD's {theme} needs.",
                f"Brings {shown} — the core of the {theme} stack this role centres on.",
            ], seed + 2)
            parts.append(connect)

            # Highlight a verified (assessed) skill for extra specificity.
            assessed = sig.get("skill_assessment_scores", {}) or {}
            verified = [(k, v) for k, v in assessed.items()
                        if v > 70 and normalize_skill(k) in canon]
            if verified:
                k, v = max(verified, key=lambda kv: kv[1])
                parts.append(f"Assessment-verified in {k} ({v:.0f}/100).")
        else:
            parts.append("No JD-core AI/retrieval skills surfaced; relevance is indirect.")

        # Quote the sentence the attention layer fired on (concrete evidence).
        if evidence:
            ev = self._clean_evidence(evidence)
            if ev:
                lead = _pick(["Evidence:", "From their profile:", "Notably:"], seed + 3)
                parts.append(f'{lead} "{ev}"')

        # A positive behavioural signal, when present.
        rr = sig.get("recruiter_response_rate")
        gh = sig.get("github_activity_score")
        if rank <= 50 and rr is not None and rr >= 0.6:
            parts.append(f"Responsive to recruiters ({rr:.0%}).")
        elif gh is not None and gh >= 70:
            parts.append(f"Active on GitHub (activity score {gh:.0f}).")

        # Honest concern(s) — always name at least one when one exists.
        concerns = self._concerns(candidate, canon)
        if concerns:
            c = concerns[0]
            if rank <= 10:
                parts.append(f"Note: {c}, but the skill match offsets it.")
            elif rank <= 50:
                parts.append(f"Caveat: {c}.")
            else:
                extra = f" Also {concerns[1]}." if len(concerns) > 1 else ""
                parts.append(f"Concern: {c}.{extra}")

        return " ".join(parts)
