
class HybridRetriever:

    def __init__(
        self,
        faiss_store,
        bm25_store,
        rrf_k: int = 60,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self.rrf_k = rrf_k
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store

    def reciprocal_rank_fusion(
        self,
        *ranked_lists: list[dict],
    ) -> list[dict]:

        rrf_scores = {}
        chunks_by_content = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                chunk = result["chunk"]
                chunk_key = chunk.content

                score = 1 / (self.rrf_k + rank)

                rrf_scores[chunk_key] = (
                    rrf_scores.get(chunk_key, 0) + score
                )

                chunks_by_content[chunk_key] = chunk

        ranked_chunks = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for chunk_key, score in ranked_chunks:
            results.append({
                "chunk": chunks_by_content[chunk_key],
                "score": score,
            })

        return results

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        faiss_results = self.faiss_store.search(query_embedding, top_k)
        bm25_results = self.bm25_store.search(query, top_k)
        reciprocal_ranked_results = self.reciprocal_rank_fusion(faiss_results, bm25_results)
        return reciprocal_ranked_results[:top_k]