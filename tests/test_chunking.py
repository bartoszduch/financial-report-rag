from src.chunking import SlidingWindowChunker
from src.models import DocumentPage


def test_chunk_overlap_and_page_metadata():
    pages = [
        DocumentPage("report.pdf", 1, " ".join(f"w{i}" for i in range(300))),
        DocumentPage("report.pdf", 2, " ".join(f"p{i}" for i in range(300))),
    ]
    chunks = SlidingWindowChunker(500, 100).chunk(pages)

    assert len(chunks) == 2
    assert chunks[0].text.split()[-100:] == chunks[1].text.split()[:100]
    assert set(chunks[0].page_numbers) == {1, 2}
    assert chunks[0].chunk_id == SlidingWindowChunker(500, 100).chunk(pages)[0].chunk_id


def test_invalid_chunk_parameters():
    import pytest

    with pytest.raises(ValueError):
        SlidingWindowChunker(100, 100)
