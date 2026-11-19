
from unittest.mock import Mock

from app.ingestion.incremental_ingestion_service import (
    IncrementalIngestionService,
)
from app.models import Chunk, Document
from app.retrieval.faiss_store import FAISSStore


def test_processes_only_new_documents(tmp_path):

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    (documents_dir / "redis.md").write_text("Redis documentation")

    tracker = Mock()
    tracker.scan.return_value = {
        "added": ["redis.md"],
        "modified": [],
        "deleted": [],
        "unchanged": [],
        "manifest": {"redis.md": "hash-1"},
    }

    ingestion = Mock()
    ingestion.load.return_value = [
        Document(
            content="Redis documentation",
            metadata={"source": "redis.md"},
        )
    ]

    chunker = Mock()
    chunker.split.return_value = [
        Chunk(
            content="Redis documentation",
            metadata={"source": "redis.md"},
        )
    ]

    embedding = Mock()
    embedding.embed_texts.return_value = [[1.0, 0.0, 0.0]]

    persistence = Mock()

    service = IncrementalIngestionService(
        tracker=tracker,
        ingestion_service=ingestion,
        chunker=chunker,
        embedding_service=embedding,
        persistence=persistence,
    )

    store = FAISSStore(dimension=3)

    result = service.update(str(documents_dir), store)

    assert result["added"] == ["redis.md"]
    assert result["new_chunks"] == 1
    assert result["total_chunks"] == 1

    ingestion.load.assert_called_once()
    embedding.embed_texts.assert_called_once()
    persistence.save.assert_called_once()
    tracker.save_manifest.assert_called_once()


def test_does_not_generate_embeddings_when_nothing_changed():

    tracker = Mock()
    tracker.scan.return_value = {
        "added": [],
        "modified": [],
        "deleted": [],
        "unchanged": ["redis.md"],
        "manifest": {"redis.md": "hash-1"},
    }

    ingestion = Mock()
    chunker = Mock()
    embedding = Mock()
    persistence = Mock()

    service = IncrementalIngestionService(
        tracker=tracker,
        ingestion_service=ingestion,
        chunker=chunker,
        embedding_service=embedding,
        persistence=persistence,
    )

    store = FAISSStore(dimension=3)

    result = service.update("documents", store)

    assert result["new_chunks"] == 0
    assert result["total_chunks"] == 0

    ingestion.load.assert_not_called()
    embedding.embed_texts.assert_not_called()
    persistence.save.assert_not_called()

    tracker.save_manifest.assert_called_once()
