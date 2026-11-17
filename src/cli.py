import json
import shutil
import time
from pathlib import Path

import typer

from src.chunking import SlidingWindowChunker
from src.config import Settings
from src.embeddings import SentenceTransformerEmbedder
from src.evaluate import run_evaluation
from src.llm_provider import GroqProvider
from src.pdf_parser import PDFParseError, parse_pdf
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

app = typer.Typer(help="Financial Report RAG Assistant")


def _embedding(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings.embedding_model)


@app.command()
def ingest(
    pdf: Path = typer.Option(..., "--pdf"),
    index: Path = typer.Option(Path("storage"), "--index"),
):
    settings = Settings(index_dir=index)
    settings.validate_retrieval_settings()
    started = time.perf_counter()
    try:
        pages = parse_pdf(pdf)
        chunks = SlidingWindowChunker(settings.chunk_size, settings.chunk_overlap).chunk(pages)
        embedder = _embedding(settings)
        vectors = embedder.encode_documents([chunk.text for chunk in chunks])
        store = VectorStore(vectors.shape[1], settings.embedding_model)
        store.add(vectors, chunks)
        store.save(index)
    except (PDFParseError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Ingested {len(chunks)} chunks from {pdf.name} in {time.perf_counter() - started:.2f}s")


@app.command()
def ask(
    question: str = typer.Option(..., "--question"),
    index: Path = typer.Option(Path("storage"), "--index"),
    top_k: int = typer.Option(None, "--top-k"),
):
    settings = Settings(index_dir=index)
    if top_k is not None:
        settings.top_k = top_k
    settings.validate_retrieval_settings()
    try:
        embedder = _embedding(settings)
        store = VectorStore.load(index, settings.embedding_model)
        provider = GroqProvider(settings.groq_api_key, settings.groq_model)
        result = RAGPipeline(store, embedder, provider, settings).query(question)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result.answer)
    if result.sources:
        typer.echo("\nSources:")
        for source in result.sources:
            pages = ", ".join(str(page) for page in source.chunk.page_numbers)
            typer.echo(f"[Source {source.rank}] {source.chunk.document}, pages {pages}, score={source.score:.4f}")


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., "--dataset"),
    index: Path = typer.Option(Path("storage"), "--index"),
):
    settings = Settings(index_dir=index)
    embedder = _embedding(settings)
    store = VectorStore.load(index, settings.embedding_model)

    def retrieval_fn(question: str, top_k: int):
        from src.retrieval import retrieve

        return [item.chunk for item in retrieve(store, embedder, question, top_k)]

    report = run_evaluation(dataset, retrieval_fn, embedder=embedder)
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
