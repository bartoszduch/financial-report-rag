import numpy as np
import pytest

from src.models import DocumentChunk
from src.vector_store import VectorStore


def test_search_and_persistence(tmp_path):
    store = VectorStore(2, "dummy-model")
    chunks = [
        DocumentChunk("c1", "report.pdf", "Revenue", [1]),
        DocumentChunk("c2", "report.pdf", "Costs", [2]),
    ]
    store.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), chunks)

    matches = store.search(np.array([1.0, 0.0], dtype=np.float32), 1)
    assert matches[0][0] == 0

    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, "dummy-model")
    assert loaded.chunks[0].chunk_id == "c1"


def test_incompatible_model(tmp_path):
    store = VectorStore(2, "model-a")
    store.save(tmp_path)
    with pytest.raises(ValueError, match="embedding model"):
        VectorStore.load(tmp_path, "model-b")
