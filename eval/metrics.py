"""Evaluation metrics: Hit Rate@k, MRR and numeric accuracy within a tolerance."""


def hit_rate_at_k(
    retrieved: list[list[int]], relevant: list[set[int]], k: int
) -> float:
    """Fraction of queries with at least one relevant chunk in the top-k."""
    raise NotImplementedError


def mean_reciprocal_rank(
    retrieved: list[list[int]], relevant: list[set[int]]
) -> float:
    """Mean of 1/rank of the first relevant chunk per query."""
    raise NotImplementedError


def numeric_accuracy(
    predicted: list[float], expected: list[float], rel_tol: float = 0.01
) -> float:
    """Fraction of predictions within a relative tolerance of the ground truth."""
    raise NotImplementedError
