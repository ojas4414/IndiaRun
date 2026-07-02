from evaluation.metrics import ndcg_at_k, mrr, map_at_k, dcg_at_k


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k([3, 2, 1, 0], 4) == 1.0        # already ideal
    assert ndcg_at_k([0, 1, 2, 3], 4) < 1.0          # reversed is worse


def test_ndcg_monotonic_improvement():
    good = ndcg_at_k([3, 3, 0, 0], 4)
    bad = ndcg_at_k([0, 0, 3, 3], 4)
    assert good > bad


def test_mrr():
    assert mrr([0, 0, 2, 0]) == 1 / 3
    assert mrr([2, 0, 0]) == 1.0
    assert mrr([0, 1, 1]) == 0.0        # nothing >= threshold 2


def test_map_at_k():
    # relevant at ranks 1 and 3: precisions 1/1 and 2/3 -> avg
    assert abs(map_at_k([2, 0, 2, 0], 4) - (1.0 + 2 / 3) / 2) < 1e-9
    assert map_at_k([0, 0, 0], 3) == 0.0


def test_dcg_gain_formula():
    # single grade-3 item at rank 1: (2^3-1)/log2(2) = 7
    assert abs(dcg_at_k([3], 1) - 7.0) < 1e-9
