"""Split parsed Markdown into chunks carrying metadata (company, year, page)."""

from typing import Any


def chunk_markdown(
    markdown: str, company: str, year: int, max_tokens: int = 512
) -> list[dict[str, Any]]:
    """Return chunks as dicts with text plus company, year and page metadata."""
    raise NotImplementedError
