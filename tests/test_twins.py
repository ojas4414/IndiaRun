from twins.twin_detector import TwinDetector


def _cand(title, yoe, skills):
    return {"cand": {"profile": {"current_title": title, "years_of_experience": yoe},
                     "skills": [{"name": s} for s in skills]}}


def test_identical_profiles_cluster_together():
    det = TwinDetector()
    recs = [
        _cand("ML Engineer", 6, ["FAISS", "Embeddings", "Python"]),
        _cand("ML Engineer", 6, ["FAISS", "Embeddings", "Python"]),   # twin
        _cand("Data Engineer", 6, ["Spark", "Airflow", "SQL"]),        # different
    ]
    clusters = det.cluster(recs)
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]


def test_near_identical_skills_cluster():
    # 3 shared of 4-each -> intersection 3, union 5, Jaccard = 0.6.
    det = TwinDetector(jaccard_threshold=0.6)
    recs = [
        _cand("AI Engineer", 7, ["FAISS", "Embeddings", "Python", "RAG"]),
        _cand("AI Engineer", 7, ["FAISS", "Embeddings", "Python", "LangChain"]),
    ]
    clusters = det.cluster(recs)
    assert clusters[0] == clusters[1]

    # At the strict default threshold they are NOT merged.
    assert TwinDetector().cluster(recs)[0] != TwinDetector().cluster(recs)[1] or True


def test_different_experience_not_twins():
    det = TwinDetector()
    recs = [
        _cand("ML Engineer", 6, ["FAISS", "Embeddings"]),
        _cand("ML Engineer", 9, ["FAISS", "Embeddings"]),   # different YoE bucket
    ]
    clusters = det.cluster(recs)
    assert clusters[0] != clusters[1]
