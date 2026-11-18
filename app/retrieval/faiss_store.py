import faiss
import numpy as np

from app.models import Chunk


class FAISSStore:

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with dimension {self.dimension}"
            )

        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.index.ntotal == 0:
            return []

        query = np.asarray([query_embedding], dtype="float32")

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query embedding dimension {self.dimension}"
            )

        faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position == -1:
                continue

            results.append({
                "chunk": self.chunks[index_position],
                "score": float(score),
            })

        return results