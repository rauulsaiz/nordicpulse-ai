"""Download annual-report PDFs listed in companies.yaml into data/raw/.

Run with: python -m nordicpulse.ingestion.download [--force]
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader

from nordicpulse.config import MANIFEST_PATH, RAW_DIR, load_companies

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = (15, 120)  # (connect, read) seconds
CHUNK_SIZE = 1024 * 256


def pdf_path(dest_dir: Path, company_id: str, year: int) -> Path:
    """Return the local path data/raw/{company_id}_{year}.pdf."""
    return dest_dir / f"{company_id}_{year}.pdf"


def validate_pdf(path: Path) -> int:
    """Check the file starts with %PDF and return its page count (raises ValueError if invalid)."""
    with path.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise ValueError("file does not start with %PDF")
    try:
        return len(PdfReader(str(path)).pages)
    except Exception as exc:  # pypdf raises several error types
        raise ValueError(f"pypdf could not read the file: {exc}") from exc


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download_report(company: str, year: int, url: str, dest_dir: Path) -> Path:
    """Stream a single report PDF to dest_dir and return its local path.

    The download goes to a .part file first and is renamed only after validation.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = pdf_path(dest_dir, company, year)
    part = final.with_suffix(".pdf.part")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}
    try:
        with requests.get(url, headers=headers, timeout=TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            with part.open("wb") as fh:
                for chunk in resp.iter_content(CHUNK_SIZE):
                    fh.write(chunk)
        validate_pdf(part)
        part.replace(final)
    finally:
        part.unlink(missing_ok=True)
    return final


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load the existing manifest entries (empty list if none)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def write_manifest(entries: list[dict[str, Any]], path: Path = MANIFEST_PATH) -> None:
    """Write manifest entries sorted by company and year."""
    entries = sorted(entries, key=lambda e: (e["company_id"], e["year"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _entry(company_id: str, year: int, url: str, path: Path, n_pages: int,
           downloaded_at: str) -> dict[str, Any]:
    return {
        "company_id": company_id,
        "year": year,
        "url": url,
        "path": path.relative_to(RAW_DIR.parent.parent).as_posix(),
        "sha256": sha256_of(path),
        "n_pages": n_pages,
        "bytes": path.stat().st_size,
        "downloaded_at": downloaded_at,
    }


def download_all(dest_dir: Path = RAW_DIR, force: bool = False) -> list[dict[str, Any]]:
    """Download every report in companies.yaml; return one result row per report.

    Each row has the manifest fields plus "name" and "status"
    (downloaded / skipped / failed) and "error" when failed.
    """
    manifest = {(e["company_id"], e["year"]): e for e in load_manifest()}
    rows: list[dict[str, Any]] = []

    for company in load_companies():
        for year, url in company["reports"].items():
            year = int(year)
            cid = company["id"]
            target = pdf_path(dest_dir, cid, year)
            row: dict[str, Any] = {"company_id": cid, "name": company["name"], "year": year,
                                   "url": url}
            try:
                if target.exists() and not force:
                    try:
                        n_pages = validate_pdf(target)
                    except ValueError:
                        n_pages = None  # invalid leftover: fall through to re-download
                    if n_pages is not None:
                        old = manifest.get((cid, year))
                        when = old["downloaded_at"] if old else datetime.fromtimestamp(
                            target.stat().st_mtime, timezone.utc).isoformat()
                        entry = _entry(cid, year, url, target, n_pages, when)
                        manifest[(cid, year)] = entry
                        rows.append({**entry, "name": company["name"], "status": "skipped"})
                        continue
                print(f"Downloading {company['name']} {year} ...", flush=True)
                download_report(cid, year, url, dest_dir)
                n_pages = validate_pdf(target)
                when = datetime.now(timezone.utc).isoformat()
                entry = _entry(cid, year, url, target, n_pages, when)
                manifest[(cid, year)] = entry
                rows.append({**entry, "name": company["name"], "status": "downloaded"})
            except Exception as exc:
                rows.append({**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    write_manifest(list(manifest.values()))
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a summary table: company, year, pages, MB and status."""
    header = f"{'Company':<16}{'Year':<6}{'Pages':>6}{'MB':>9}  Status"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        pages = r.get("n_pages", "-")
        mb = f"{r['bytes'] / 1024 / 1024:.2f}" if "bytes" in r else "-"
        print(f"{r['name']:<16}{r['year']:<6}{pages:>6}{mb:>9}  {r['status']}")
    for r in rows:
        if r["status"] == "failed":
            print(f"\nFAILED {r['name']} {r['year']}: {r['error']}\n  {r['url']}")


def main() -> None:
    """Command-line entry point."""
    sys.stdout.reconfigure(encoding="utf-8")  # company names contain non-ASCII (Ørsted)
    parser = argparse.ArgumentParser(description="Download annual-report PDFs.")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    print_summary(download_all(force=args.force))


if __name__ == "__main__":
    main()
