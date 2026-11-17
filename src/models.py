from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    document: str
    page_number: int
    text: str


@dataclass
class DocumentChunk:
    chunk_id: str
    document: str
    text: str
    page_numbers: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    rank: int


@dataclass
class QueryResult:
    answer: str
    sources: list[RetrievalResult]
    citations: list[int]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float

