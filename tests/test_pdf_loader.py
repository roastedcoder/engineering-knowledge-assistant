import pymupdf
import pytest

from app.ingestion.pdf_loader import PDFLoader


@pytest.fixture
def sample_pdf(tmp_path):
    file_path = tmp_path / "sample.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Redis Architecture")
    page.insert_text((72, 100), "Redis is an in-memory data store.")
    pdf.save(file_path)
    pdf.close()

    return file_path


def test_load_pdf_file(sample_pdf):
    documents = PDFLoader().load(str(sample_pdf))

    assert len(documents) == 1
    assert "Redis Architecture" in documents[0].content
    assert documents[0].metadata["source"] == "sample.pdf"
    assert documents[0].metadata["file_type"] == "pdf"
    assert documents[0].metadata["page"] == 1


def test_load_missing_pdf_file():
    with pytest.raises(FileNotFoundError):
        PDFLoader().load("documents/missing.pdf")


def test_reject_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Some content")

    with pytest.raises(ValueError):
        PDFLoader().load(str(file_path))
        