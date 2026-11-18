import pytest

from app.ingestion.markdown_loader import MarkdownLoader


def test_load_markdown_file(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Redis\nRedis is an in-memory database.")

    document = MarkdownLoader().load(str(file_path))

    assert document.content == "# Redis\nRedis is an in-memory database."
    assert document.metadata["source"] == "sample.md"
    assert document.metadata["file_type"] == "markdown"


def test_load_missing_markdown_file():
    with pytest.raises(FileNotFoundError):
        MarkdownLoader().load("documents/missing.md")


def test_reject_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Some content")

    with pytest.raises(ValueError):
        MarkdownLoader().load(str(file_path))