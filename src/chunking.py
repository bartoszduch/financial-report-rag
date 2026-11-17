from abc import ABC, abstractmethod
import hashlib

from src.models import DocumentChunk, DocumentPage


class Chunker(ABC):
    @abstractmethod
    def chunk(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        raise NotImplementedError


class SlidingWindowChunker(Chunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        if not pages:
            return []

        document = pages[0].document
        chunks: list[DocumentChunk] = []
        words: list[str] = []
        page_by_word: list[int] = []
        for page in pages:
            for word in page.text.split():
                words.append(word)
                page_by_word.append(page.page_number)

        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            text = " ".join(words[start:end])
            source_pages = sorted(set(page_by_word[start:end]))
            digest = hashlib.sha256(
                f"{document}:{start}:{text}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document}:{digest}",
                    document=document,
                    text=text,
                    page_numbers=source_pages,
                    metadata={"word_start": start, "word_end": end - 1},
                )
            )
            if end == len(words):
                break
            start = end - self.overlap
        return chunks
