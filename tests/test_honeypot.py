import pytest
from honeypot.detector import HoneypotDetector

def test_honeypot_expert_zero_duration():
    detector = HoneypotDetector()
    
    cand = {
        "candidate_id": "CAND_1",
        "skills": [
            {"name": "A", "proficiency": "expert", "duration_months": 0},
            {"name": "B", "proficiency": "expert", "duration_months": 0},
            {"name": "C", "proficiency": "expert", "duration_months": 0},
            {"name": "D", "proficiency": "expert", "duration_months": 0},
            {"name": "E", "proficiency": "expert", "duration_months": 0},
        ]
    }
    
    is_honeypot, reason = detector.check_candidate(cand)
    assert is_honeypot == True
    assert "expert skills with 0 duration" in reason

def test_clean_candidate():
    detector = HoneypotDetector()
    
    cand = {
        "candidate_id": "CAND_2",
        "skills": [
            {"name": "A", "proficiency": "expert", "duration_months": 48},
            {"name": "B", "proficiency": "intermediate", "duration_months": 12},
        ],
        "profile": {"years_of_experience": 5.0},
        "career_history": [{"duration_months": 60}]
    }
    
    is_honeypot, _ = detector.check_candidate(cand)
    assert is_honeypot == False
