import re
import time

from src.config import Settings
from src.embeddings import Embedder
from src.llm_provider import LLMProvider
from src.models import QueryResult
from src.prompts import FALLBACK, build_prompt
from src.retrieval import retrieve
from src.vector_store import VectorStore


def validate_citations(answer: str, source_count: int) -> list[int]:
    identifiers = [int(value) for value in re.findall(r"\[Source (\d+)\]", answer)]
    valid = sorted(set(value for value in identifiers if 1 <= value <= source_count))
    invalid = any(value > source_count or value < 1 for value in identifiers)
    if invalid:
        return valid
    return valid


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        provider: LLMProvider,
        settings: Settings,
    ):
        self.store = store
        self.embedder = embedder
        self.provider = provider
        self.settings = settings

    def query(self, question: str) -> QueryResult:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = time.perf_counter()

        retrieval_started = time.perf_counter()
        sources = retrieve(self.store, self.embedder, question, self.settings.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        threshold = self.settings.similarity_threshold
        insufficient = not sources or (threshold > 0 and max(s.score for s in sources) < threshold)
        if insufficient:
            answer = FALLBACK
        else:
            prompt = build_prompt(question, sources)
            llm_started = time.perf_counter()
            answer = self.provider.generate(prompt)
            llm_ms = (time.perf_counter() - llm_started) * 1000

            if not re.search(r"\[Source \d+\]", answer):
                answer = FALLBACK
            citations = validate_citations(answer, len(sources))
            if any(
                int(value) > len(sources)
                for value in re.findall(r"\[Source (\d+)\]", answer)
            ):
                answer = FALLBACK

        llm_ms = locals().get("llm_ms", 0.0)
        citations = validate_citations(answer, len(sources)) if answer != FALLBACK else []
        total_ms = (time.perf_counter() - started) * 1000
        return QueryResult(
            answer=answer,
            sources=sources,
            citations=citations,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=total_ms,
        )
