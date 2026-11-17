import time
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.models import DocumentPage


class PDFParseError(ValueError):
    pass


def parse_pdf(path: str | Path) -> list[DocumentPage]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise PDFParseError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PDFParseError("Only PDF files are supported")

    started = time.perf_counter()
    try:
        reader = PdfReader(str(pdf_path), strict=False)
    except (PdfReadError, OSError) as exc:
        raise PDFParseError(f"Corrupted or unreadable PDF: {exc}") from exc

    pages: list[DocumentPage] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PDFParseError(f"Unable to extract page {number}: {exc}") from exc
        normalized = " ".join(text.split())
        if normalized:
            pages.append(DocumentPage(pdf_path.name, number, normalized))

    elapsed = time.perf_counter() - started
    if not pages:
        raise PDFParseError(
            f"No usable text found in {pdf_path.name}; the PDF may be image-only/scanned "
            f"(processing took {elapsed:.3f}s)"
        )
    return pages
