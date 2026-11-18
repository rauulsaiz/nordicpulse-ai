"""Dense retrieval: text embeddings stored in a FAISS index."""

from pathlib import Path
from typing import Any


def build_index(chunks: list[dict[str, Any]], index_dir: Path) -> None:
    """Embed the chunks and persist a FAISS index under index_dir."""
    raise NotImplementedError


def search(query: str, index_dir: Path, k: int = 10) -> list[tuple[int, float]]:
    """Return the top-k (chunk_id, score) pairs for the query."""
    raise NotImplementedError
