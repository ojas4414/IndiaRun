from typing import Dict, Any
from datetime import datetime

_TODAY = datetime(2026, 7, 2)


class BehavioralScorer:
    def __init__(self, jd_requirements: Dict[str, Any]):
        self.notice_pref = jd_requirements.get("notice_period_preference_days", 30)
        self.locations = [l.lower() for l in jd_requirements.get("location_preference", [])]

    def score_behavioral(self, candidate: Dict[str, Any]) -> float:
        """Behavioural fit from the Redrob signal set + location. Range [0, 1]."""
        sig = candidate.get("redrob_signals", {})
        if not sig:
            return 0.0

        score = 0.0

        # Engagement / reliability
        score += sig.get("recruiter_response_rate", 0) * 0.18
        score += sig.get("interview_completion_rate", 0) * 0.12

        # Notice period
        notice = sig.get("notice_period_days", 90)
        if notice <= self.notice_pref:
            score += 0.10
        elif notice <= 60:
            score += 0.05
        elif notice > 90:
            score -= 0.05

        # Availability / intent
        if sig.get("open_to_work_flag", False):
            score += 0.08

        # GitHub activity (0..100) — the JD values people who "write code".
        gh = sig.get("github_activity_score", 0) or 0
        score += min(gh / 100.0, 1.0) * 0.12

        # Trust / verification
        if sig.get("verified_email") and sig.get("verified_phone"):
            score += 0.05
        if sig.get("linkedin_connected"):
            score += 0.02

        # Network + completeness (lightly weighted, saturating)
        cc = sig.get("connection_count", 0) or 0
        score += min(cc / 1000.0, 1.0) * 0.05
        score += (sig.get("profile_completeness_score", 0) or 0) / 100.0 * 0.05

        # Recency of activity
        active = self._parse(sig.get("last_active_date"))
        if active:
            days = (_TODAY - active).days
            if days < 30:
                score += 0.10
            elif days < 90:
                score += 0.05
            elif days > 180:
                score -= 0.10

        # Location fit vs JD preference
        loc = candidate.get("profile", {}).get("location", "").lower()
        if self.locations:
            if any(l in loc for l in self.locations):
                score += 0.10
            elif sig.get("willing_to_relocate"):
                score += 0.04

        return max(0.0, min(score, 1.0))

    @staticmethod
    def _parse(date_str):
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return None
