"""Pydantic models that form the project's data contract for extracted KPIs."""

from pydantic import BaseModel


class KPIValue(BaseModel):
    """A single KPI value exactly as reported, with provenance."""

    value: float
    unit: str
    currency: str | None
    period: str
    definition_as_reported: str
    source_page: int
    source_quote: str


class KPIReport(BaseModel):
    """All KPIs extracted from one company's annual report for one year."""

    company: str
    year: int
    kpis: dict[str, KPIValue]
