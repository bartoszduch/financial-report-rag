from pypdf import PdfWriter
import pytest

from src.pdf_parser import PDFParseError, parse_pdf


def test_parse_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": "Test report"})
    with path.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(PDFParseError, match="No usable text"):
        parse_pdf(path)


def test_corrupted_pdf(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(PDFParseError, match="Corrupted|No usable"):
        parse_pdf(path)
