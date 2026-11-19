
import pickle
from pathlib import Path

import faiss

from app.retrieval.faiss_store import FAISSStore


class FAISSPersistence:

    def __init__(self, storage_dir: str = "storage/faiss"):
        self.storage_dir = Path(storage_dir)

    def save(self, store: FAISSStore) -> None:

        # Create the storage directory if it doesn't exist
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save the FAISS vector index
        index_path = self.storage_dir / "index.faiss"
        faiss.write_index(store.index, str(index_path))

        # Save the corresponding chunks
        chunks_path = self.storage_dir / "chunks.pkl"

        with open(chunks_path, "wb") as file:
            pickle.dump(store.chunks, file)


    def load(self) -> FAISSStore:

        index_path = self.storage_dir / "index.faiss"
        chunks_path = self.storage_dir / "chunks.pkl"

        # Check whether both files exist
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "FAISS index or chunks file not found"
            )

        # Load the FAISS index
        index = faiss.read_index(str(index_path))

        # Load the corresponding chunks
        with open(chunks_path, "rb") as file:
            chunks = pickle.load(file)

        # Create a new FAISSStore and restore its state
        store = FAISSStore(dimension=index.d)

        store.index = index
        store.chunks = chunks

        return store
