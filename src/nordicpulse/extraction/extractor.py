"""LLM-based extraction of financial and ESG KPIs into validated KPIReport objects."""

from nordicpulse.extraction.schemas import KPIReport


def extract_kpis(company: str, year: int, kpi_names: list[str]) -> KPIReport:
    """Retrieve relevant chunks and extract the requested KPIs with an LLM."""
    raise NotImplementedError
