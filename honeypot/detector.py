from typing import Dict, Any, Tuple, Optional
from datetime import datetime

# Reference "today" for the challenge (dataset frozen mid-2026).
_TODAY = datetime(2026, 7, 2)


def _parse(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _months_between(a: datetime, b: datetime) -> float:
    return (b - a).days / 30.44


class HoneypotDetector:
    """Rejects impossible / fabricated profiles before they enter the index."""

    def check_candidate(self, candidate_dict: Dict[str, Any]) -> Tuple[bool, str]:
        skills = candidate_dict.get("skills", [])
        profile = candidate_dict.get("profile", {})
        career = candidate_dict.get("career_history", [])

        # Rule 1: many "expert" skills claimed with 0 months of practice.
        expert_zero = sum(
            1 for s in skills
            if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
        )
        if len(skills) >= 5 and expert_zero >= 5:
            return True, f"Honeypot: {expert_zero} expert skills with 0 duration"

        # Per-role date sanity checks.
        earliest_start: Optional[datetime] = None
        for role in career:
            dur = role.get("duration_months", 0) or 0

            # Rule 2: absurd single-role tenure (>50 years).
            if dur > 600:
                return True, "Honeypot: impossible role duration (>50y)"

            start = _parse(role.get("start_date"))
            end = _parse(role.get("end_date")) or (_TODAY if role.get("is_current") else None)

            if start:
                if earliest_start is None or start < earliest_start:
                    earliest_start = start
                # Rule 3: start date in the future.
                if start > _TODAY:
                    return True, "Honeypot: role starts in the future"
                # Rule 4: end before start.
                if end and end < start:
                    return True, "Honeypot: role ends before it starts"
                # Rule 5: claimed tenure far exceeds the actual date span
                # (e.g. "8 years at a company that only spans 3 years").
                if end is not None:
                    span = _months_between(start, end)
                    if dur > span + 6:  # >6 month slack for rounding
                        return True, (
                            f"Honeypot: tenure {dur}mo exceeds date span "
                            f"{span:.0f}mo"
                        )

        # Rule 6: claims more total experience than the career timeline allows.
        yoe = profile.get("years_of_experience", 0) or 0
        if earliest_start is not None:
            max_possible_years = _months_between(earliest_start, _TODAY) / 12.0
            if yoe > max_possible_years + 1.0:
                return True, (
                    f"Honeypot: {yoe}y experience but career began only "
                    f"{max_possible_years:.1f}y ago"
                )

        # Rule 7: substantial claimed experience with essentially no history.
        total_career_years = sum(r.get("duration_months", 0) for r in career) / 12.0
        if yoe >= 5 and total_career_years < 0.5 and len(career) <= 1:
            return True, "Honeypot: high YOE with no meaningful career history"

        return False, ""
