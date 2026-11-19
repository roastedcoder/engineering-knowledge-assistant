def recall_at_k(
    retrieved_results: list[dict],
    relevant_sources: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_sources:
        raise ValueError("relevant_sources cannot be empty")

    top_results = retrieved_results[:k]

    retrieved_sources = {
        result["chunk"].metadata["source"]
        for result in top_results
    }

    relevant = set(relevant_sources)

    found = retrieved_sources.intersection(relevant)

    return len(found) / len(relevant)