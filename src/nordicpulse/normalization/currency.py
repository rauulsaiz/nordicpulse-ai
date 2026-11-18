"""Deterministic currency and scale conversion (no LLM involved)."""


def convert_currency(
    value: float, from_currency: str, to_currency: str, period: str
) -> float:
    """Convert an amount between currencies using the FX rates in data/fx/."""
    raise NotImplementedError


def convert_scale(value: float, from_scale: str, to_scale: str) -> float:
    """Convert between scales such as thousands, millions and billions."""
    raise NotImplementedError
