"""Hybrid retrieval: fuse dense and sparse rankings with Reciprocal Rank Fusion."""


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse several ranked lists of chunk ids into one list of (chunk_id, rrf_score)."""
    raise NotImplementedError


def hybrid_search(query: str, k: int = 10) -> list[tuple[int, float]]:
    """Run dense and BM25 search and return the RRF-fused top-k."""
    raise NotImplementedError
