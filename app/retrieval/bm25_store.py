
import re

from rank_bm25 import BM25Okapi

from app.models import Chunk


class BM25Store:

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.index = None

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        self.chunks.extend(chunks)

        tokenized_documents = [
            self._tokenize(chunk.content)
            for chunk in self.chunks
        ]

        self.index = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not query.strip() or self.index is None:
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self.index.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index_position in ranked_indices:
            score = float(scores[index_position])

            if score <= 0:
                continue

            results.append({
                "chunk": self.chunks[index_position],
                "score": score,
            })

            if len(results) == top_k:
                break

        return results
