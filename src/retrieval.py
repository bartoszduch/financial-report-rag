from src.embeddings import Embedder
from src.models import RetrievalResult
from src.vector_store import VectorStore


def retrieve(store: VectorStore, embedder: Embedder, question: str, top_k: int) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_embedding = embedder.encode_query(question)
    matches = store.search(query_embedding, top_k)
    return [
        RetrievalResult(chunk=store.chunks[position], score=score, rank=rank)
        for rank, (position, score) in enumerate(matches, start=1)
    ]
