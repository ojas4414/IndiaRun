import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from security.injection_defense import InjectionDefender


def build_jd_text(jd_reqs: Dict) -> str:
    """Flatten the JD requirements into the text embedded against candidates.
    Shared by precompute.py (full-pool run) and sandbox_app.py (ad-hoc demo)
    so both embed the JD identically.
    """
    return (
        "Ideal roles: " + ", ".join(jd_reqs.get("ideal_roles", [])) + " | " +
        "Required: " + ", ".join(jd_reqs.get("required_skills", [])) + " | " +
        "Preferred: " + ", ".join(jd_reqs.get("preferred_skills", [])) + " | " +
        f"Domain: {jd_reqs.get('domain', '')} | Seniority: {jd_reqs.get('seniority', '')} | " +
        "Culture: " + ", ".join(jd_reqs.get("culture_signals", []))
    )


class CandidateEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # We can optionally start a multi-process pool for faster CPU encoding
        self.pool = None
        self.defender = InjectionDefender()

    def start_pool(self):
        self.pool = self.model.start_multi_process_pool()
        
    def stop_pool(self):
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)

    def extract_text(self, candidate: dict) -> str:
        """
        Combines headline, summary, skills, and career descriptions 
        into a single text document for embedding.
        """
        san = self.defender.sanitize
        parts = []

        # Profile
        profile = candidate.get("profile", {})
        if profile.get("headline"):
            parts.append(f"Headline: {san(profile['headline'])}")
        if profile.get("summary"):
            parts.append(f"Summary: {san(profile['summary'])}")

        # Skills
        skills = candidate.get("skills", [])
        if skills:
            skill_names = [san(s.get("name")) for s in skills if s.get("name")]
            parts.append(f"Skills: {', '.join(skill_names)}")

        # Career History
        career = candidate.get("career_history", [])
        if career:
            roles = []
            for role in career:
                title = san(role.get("title", ""))
                desc = san(role.get("description", ""))
                if title or desc:
                    roles.append(f"{title}: {desc}")
            if roles:
                parts.append("Experience: " + " | ".join(roles))

        return "\n".join(parts)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Computes embeddings for a batch of texts.
        """
        if self.pool:
            return self.model.encode_multi_process(texts, self.pool, batch_size=256, normalize_embeddings=True)
        return self.model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)

    def embed_jd(self, jd_text: str) -> np.ndarray:
        return self.model.encode([jd_text], normalize_embeddings=True)[0]
