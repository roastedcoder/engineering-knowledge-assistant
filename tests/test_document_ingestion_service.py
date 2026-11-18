import pymupdf
import pytest

from app.ingestion.document_ingestion_service import DocumentIngestionService


@pytest.fixture
def sample_markdown(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Redis\nRedis is an in-memory database.")
    return file_path


@pytest.fixture
def sample_pdf(tmp_path):
    file_path = tmp_path / "sample.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Redis Architecture")
    pdf.save(file_path)
    pdf.close()

    return file_path


def test_load_markdown(sample_markdown):
    documents = DocumentIngestionService().load(str(sample_markdown))

    assert len(documents) == 1
    assert documents[0].metadata["file_type"] == "markdown"


def test_load_pdf(sample_pdf):
    documents = DocumentIngestionService().load(str(sample_pdf))

    assert len(documents) == 1
    assert documents[0].metadata["file_type"] == "pdf"


def test_reject_unsupported_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Some content")

    with pytest.raises(ValueError):
        DocumentIngestionService().load(str(file_path))

def test_load_directory_processes_supported_files(tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()

    # Create Markdown file
    markdown_file = directory / "redis.md"
    markdown_file.write_text("# Redis\nRedis is an in-memory database.")

    # Create PDF file
    pdf_file = directory / "architecture.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Redis Architecture")
    pdf.save(pdf_file)
    pdf.close()

    # Create unsupported file
    (directory / "notes.txt").write_text("Ignore this file")

    documents = DocumentIngestionService().load_directory(str(directory))

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "architecture.pdf"
    assert documents[1].metadata["source"] == "redis.md"