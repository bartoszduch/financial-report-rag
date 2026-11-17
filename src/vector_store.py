import json
from pathlib import Path
from typing import Any

import numpy as np

from src.models import DocumentChunk


class VectorStore:
    def __init__(self, dimension: int, model_name: str):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required") from exc
        self.dimension = dimension
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected embeddings with dimension {self.dimension}")
        if vectors.shape[0] != len(chunks):
            raise ValueError("embedding count must match chunk count")
        if vectors.shape[0] == 0:
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index.ntotal == 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(f"query dimension must be {self.dimension}")
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        import faiss

        scores, positions = self.index.search(query, min(top_k, self.index.ntotal))
        return [(int(position), float(score)) for position, score in zip(positions[0], scores[0]) if position >= 0]

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "index.faiss"
        metadata_path = directory / "index_metadata.json"
        faiss.write_index(self.index, str(index_path))
        metadata: dict[str, Any] = {
            "embedding_model": self.model_name,
            "dimension": self.dimension,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document": c.document,
                    "text": c.text,
                    "page_numbers": c.page_numbers,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return index_path, metadata_path

    @classmethod
    def load(cls, directory: str | Path, embedding_model: str) -> "VectorStore":
        import faiss

        directory = Path(directory)
        index_path = directory / "index.faiss"
        metadata_path = directory / "index_metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("A complete persisted index was not found")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["embedding_model"] != embedding_model:
            raise ValueError(
                f"index uses embedding model {metadata['embedding_model']}, "
                f"but {embedding_model} was requested"
            )
        store = cls(int(metadata["dimension"]), embedding_model)
        store.index = faiss.read_index(str(index_path))
        store.chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                document=c["document"],
                text=c["text"],
                page_numbers=c["page_numbers"],
                metadata=c.get("metadata", {}),
            )
            for c in metadata["chunks"]
        ]
        if store.index.d != store.dimension or store.index.ntotal != len(store.chunks):
            raise ValueError("persisted index metadata is inconsistent")
        return store
