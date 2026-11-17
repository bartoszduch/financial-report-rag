import numpy as np

from src.config import Settings
from src.models import DocumentChunk
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


class DummyEmbedder:
    def encode_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


class DummyProvider:
    def __init__(self, answer="Revenue was $10 million [Source 1]."):
        self.answer = answer

    def generate(self, prompt):
        return self.answer


def make_store():
    store = VectorStore(2, "dummy-model")
    store.add(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [DocumentChunk("c1", "report.pdf", "Revenue was $10 million.", [3])],
    )
    return store


def test_grounded_answer_and_sources():
    result = RAGPipeline(make_store(), DummyEmbedder(), DummyProvider(), Settings(top_k=1)).query(
        "What was revenue?"
    )
    assert result.answer.endswith("[Source 1].")
    assert result.sources[0].chunk.page_numbers == [3]
    assert result.citations == [1]


def test_fallback_for_unsupported_answer():
    provider = DummyProvider("The stock could reach $1 million.")
    result = RAGPipeline(make_store(), DummyEmbedder(), provider, Settings(top_k=1)).query(
        "Unsupported forecast?"
    )
    assert result.answer == "Insufficient information in the provided document."
