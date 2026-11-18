"""Custom MCP server exposing search_filings, extract_kpis and calculate_financial_ratios."""

from typing import Any


def search_filings(
    query: str, company: str | None = None, year: int | None = None, k: int = 5
) -> list[dict[str, Any]]:
    """Tool: hybrid search over the filings, optionally filtered by company and year."""
    raise NotImplementedError


def extract_kpis(company: str, year: int, kpi_names: list[str]) -> dict[str, Any]:
    """Tool: extract the requested KPIs for a company and year as a KPIReport dict."""
    raise NotImplementedError


def calculate_financial_ratios(
    company: str, year: int, ratios: list[str]
) -> dict[str, float]:
    """Tool: compute the requested financial ratios for a company and year."""
    raise NotImplementedError


def main() -> None:
    """Register the tools and start the MCP server."""
    raise NotImplementedError
