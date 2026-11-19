
from app.ingestion.document_tracker import DocumentTracker
from app.ingestion.document_ingestion_service import DocumentIngestionService
from app.chunking.text_chunker import TextChunker
from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.faiss_persistence import FAISSPersistence


class IncrementalIngestionService:

    def __init__(
        self,
        tracker: DocumentTracker,
        ingestion_service: DocumentIngestionService,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        persistence: FAISSPersistence,
    ):
        self.tracker = tracker
        self.ingestion_service = ingestion_service
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.persistence = persistence

    def update(
        self,
        directory_path: str,
        vector_store: FAISSStore,
    ) -> dict:

        # 1. Detect document changes
        changes = self.tracker.scan(directory_path)

        added = changes["added"]
        modified = changes["modified"]
        deleted = changes["deleted"]

        changed_files = added + modified

        # 2. Remove old chunks for modified and deleted documents
        files_to_remove = modified + deleted

        for file_name in files_to_remove:
            vector_store.remove_by_source(file_name)

        # 3. Load only newly added or modified documents
        new_chunks = []

        for file_name in changed_files:
            file_path = f"{directory_path}/{file_name}"

            documents = self.ingestion_service.load(file_path)

            for document in documents:
                new_chunks.extend(
                    self.chunker.split(document)
                )

        # 4. Generate embeddings only for new chunks
        if new_chunks:

            embeddings = self.embedding_service.embed_texts(
                [chunk.content for chunk in new_chunks]
            )

            vector_store.add(new_chunks, embeddings)

        # 5. Persist updated index
        if changed_files or deleted:
            self.persistence.save(vector_store)

        # 6. Save manifest after index persistence succeeds
        self.tracker.save_manifest(changes["manifest"])

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "unchanged": changes["unchanged"],
            "new_chunks": len(new_chunks),
            "total_chunks": len(vector_store.chunks),
        }
