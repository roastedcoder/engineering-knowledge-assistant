
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 3,
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not results:
            return []

        pairs = [
            (query, result["chunk"].content)
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_results.append({
                "chunk": result["chunk"],
                "score": float(score),
            })

        reranked_results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return reranked_results[:top_k]
