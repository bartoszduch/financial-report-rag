from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.chunking import SlidingWindowChunker
from src.config import Settings
from src.embeddings import SentenceTransformerEmbedder
from src.evaluate import run_evaluation
from src.llm_provider import GroqProvider
from src.pdf_parser import PDFParseError, parse_pdf
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

app = FastAPI(title="Financial Report RAG Assistant", version="0.1.0")
settings = Settings()
settings.validate_retrieval_settings()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    rank: int
    score: float
    document: str
    page_numbers: list[int]
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    sources: list[SourceResponse]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float


class EvaluateRequest(BaseModel):
    dataset_path: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF uploads are supported")
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "PDF upload exceeds configured size limit")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
        temp.write(content)
        temp_path = Path(temp.name)
    try:
        pages = parse_pdf(temp_path)
        chunks = SlidingWindowChunker(settings.chunk_size, settings.chunk_overlap).chunk(pages)
        for chunk in chunks:
            chunk.document = file.filename
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        vectors = embedder.encode_documents([chunk.text for chunk in chunks])
        store = VectorStore(vectors.shape[1], settings.embedding_model)
        store.add(vectors, chunks)
        store.save(settings.index_dir)
        return {"document": file.filename, "chunks": len(chunks)}
    except PDFParseError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        store = VectorStore.load(settings.index_dir, settings.embedding_model)
        provider = GroqProvider(settings.groq_api_key, settings.groq_model)
        effective = settings.model_copy(deep=True)
        if request.top_k is not None:
            effective.top_k = request.top_k
        result = RAGPipeline(store, embedder, provider, effective).query(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(404, "No persisted index found; ingest a PDF first") from exc
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise HTTPException(503, str(exc)) from exc
        raise
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        sources=[
            SourceResponse(
                rank=s.rank,
                score=s.score,
                document=s.chunk.document,
                page_numbers=s.chunk.page_numbers,
                chunk_id=s.chunk.chunk_id,
            )
            for s in result.sources
        ],
        retrieval_latency_ms=result.retrieval_latency_ms,
        llm_latency_ms=result.llm_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    base = Path("data/evaluation").resolve()
    dataset = (base / request.dataset_path).resolve()
    if dataset.parent != base or not dataset.is_file():
        raise HTTPException(404, "Evaluation dataset not found")
    try:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        store = VectorStore.load(settings.index_dir, settings.embedding_model)

        def retrieval_fn(question: str, top_k: int):
            from src.retrieval import retrieve

            return [item.chunk for item in retrieve(store, embedder, question, top_k)]

        return run_evaluation(dataset, retrieval_fn, embedder=embedder)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
