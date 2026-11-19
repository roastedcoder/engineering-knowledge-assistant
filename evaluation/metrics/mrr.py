def reciprocal_rank(
    retrieved_results: list[dict],
    relevant_sources: list[str],
) -> float:
    relevant = set(relevant_sources)

    for rank, result in enumerate(retrieved_results, start=1):
        source = result["chunk"].metadata["source"]

        if source in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    all_results: list[list[dict]],
    all_relevant_sources: list[list[str]],
) -> float:
    if len(all_results) != len(all_relevant_sources):
        raise ValueError("Results and relevant sources must have equal length")

    if not all_results:
        raise ValueError("Evaluation dataset cannot be empty")

    reciprocal_ranks = [
        reciprocal_rank(results, sources)
        for results, sources in zip(all_results, all_relevant_sources)
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)