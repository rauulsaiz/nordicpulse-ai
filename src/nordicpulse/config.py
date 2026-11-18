"""Project configuration: loading config/companies.yaml and resolving project paths."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
COMPANIES_YAML: Path = CONFIG_DIR / "companies.yaml"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FX_DIR: Path = DATA_DIR / "fx"
INDEXES_DIR: Path = PROJECT_ROOT / "indexes"
MANIFEST_PATH: Path = RAW_DIR / "manifest.json"


def project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def load_companies(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the company/report definitions from companies.yaml.

    Each company dict has: id, name, reporting_currency and reports
    (a mapping of int year -> PDF URL).
    """
    with (path or COMPANIES_YAML).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["companies"]
