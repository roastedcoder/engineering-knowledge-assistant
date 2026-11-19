
import pytest

from app.ingestion.document_tracker import DocumentTracker


def test_detects_new_documents(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    (documents_dir / "redis.md").write_text("Redis documentation")

    tracker = DocumentTracker(
        manifest_path=str(tmp_path / "manifest.json")
    )

    result = tracker.scan(str(documents_dir))

    assert result["added"] == ["redis.md"]
    assert result["modified"] == []
    assert result["deleted"] == []
    assert result["unchanged"] == []


def test_detects_modified_documents(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    document = documents_dir / "redis.md"
    document.write_text("Redis documentation")

    tracker = DocumentTracker(
        manifest_path=str(tmp_path / "manifest.json")
    )

    first_scan = tracker.scan(str(documents_dir))
    tracker.save_manifest(first_scan["manifest"])

    document.write_text("Updated Redis documentation")

    second_scan = tracker.scan(str(documents_dir))

    assert second_scan["modified"] == ["redis.md"]
    assert second_scan["added"] == []


def test_detects_unchanged_documents(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    (documents_dir / "redis.md").write_text("Redis documentation")

    tracker = DocumentTracker(
        manifest_path=str(tmp_path / "manifest.json")
    )

    first_scan = tracker.scan(str(documents_dir))
    tracker.save_manifest(first_scan["manifest"])

    second_scan = tracker.scan(str(documents_dir))

    assert second_scan["unchanged"] == ["redis.md"]
    assert second_scan["added"] == []
    assert second_scan["modified"] == []


def test_detects_deleted_documents(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    document = documents_dir / "redis.md"
    document.write_text("Redis documentation")

    tracker = DocumentTracker(
        manifest_path=str(tmp_path / "manifest.json")
    )

    first_scan = tracker.scan(str(documents_dir))
    tracker.save_manifest(first_scan["manifest"])

    document.unlink()

    second_scan = tracker.scan(str(documents_dir))

    assert second_scan["deleted"] == ["redis.md"]


def test_ignores_unsupported_files(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    (documents_dir / "notes.txt").write_text("Some notes")
    (documents_dir / "redis.md").write_text("Redis documentation")

    tracker = DocumentTracker(
        manifest_path=str(tmp_path / "manifest.json")
    )

    result = tracker.scan(str(documents_dir))

    assert result["added"] == ["redis.md"]


def test_rejects_missing_directory(tmp_path):
    tracker = DocumentTracker(
        manifest_path=str(tmp_path / "manifest.json")
    )

    with pytest.raises(FileNotFoundError):
        tracker.scan(str(tmp_path / "missing"))
