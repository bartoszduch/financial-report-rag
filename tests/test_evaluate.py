from src.evaluate import recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b", "c"], {"b"}, 3) == 1.0
