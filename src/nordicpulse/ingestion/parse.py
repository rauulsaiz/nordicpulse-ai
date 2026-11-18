"""Parse annual-report PDFs with PyMuPDF (plain text) and Docling (layout + tables).

Run with:
    python -m nordicpulse.ingestion.parse [--ids vestas_2025 ...] [--force]

For each report id ``{company_id}_{year}`` listed in config/companies.yaml, reads
``data/raw/{id}.pdf`` and writes:

    data/processed/pymupdf/{id}.jsonl   one line per page: {"page": n, "text": ...} (1-based)
    data/processed/docling/{id}.md      Docling Markdown export
    data/processed/docling/{id}.json    full Docling document (provenance + page numbers)

A report whose three outputs already exist is skipped unless --force is given, so an
interrupted batch job can simply be resubmitted. Every file is written to a temporary
path and renamed at the end, so a killed job never leaves a half-written output that
would be mistaken for a finished one.

At the end, data/processed/processing_summary.csv is (re)written with one row per report.

Heavy dependencies (docling, torch, pymupdf) are imported lazily so that ``--help``
works without them installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nordicpulse.config import PROCESSED_DIR, RAW_DIR, load_companies

log = logging.getLogger("nordicpulse.parse")

PYMUPDF_DIR: Path = PROCESSED_DIR / "pymupdf"
DOCLING_DIR: Path = PROCESSED_DIR / "docling"
SUMMARY_CSV: Path = PROCESSED_DIR / "processing_summary.csv"

SUMMARY_FIELDS = ["id", "n_pages", "device", "docling_seconds", "pymupdf_seconds",
                  "n_tables_docling", "status"]


@dataclass
class ReportResult:
    """One row of processing_summary.csv."""

    id: str
    n_pages: int | None = None
    device: str | None = None
    docling_seconds: float | None = None
    pymupdf_seconds: float | None = None
    n_tables_docling: int | None = None
    status: str = "pending"  # ok / skipped / failed / missing_pdf


# --------------------------------------------------------------------------- paths


def report_ids() -> list[str]:
    """Return all report ids ``{company_id}_{year}`` in companies.yaml order."""
    return [f"{c['id']}_{int(year)}" for c in load_companies() for year in c["reports"]]


def output_paths(report_id: str) -> dict[str, Path]:
    """Return the three output paths of a report."""
    return {
        "pymupdf": PYMUPDF_DIR / f"{report_id}.jsonl",
        "markdown": DOCLING_DIR / f"{report_id}.md",
        "json": DOCLING_DIR / f"{report_id}.json",
    }


def is_done(report_id: str) -> bool:
    """True if all three outputs of a report already exist."""
    return all(p.exists() for p in output_paths(report_id).values())


def _write_atomic(path: Path, text: str) -> None:
    """Write text to path via a temporary file + rename (no partial outputs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- device


def detect_device() -> str:
    """Return "cuda" if torch sees a GPU, else "cpu", and log what was found."""
    try:
        import torch
    except ImportError:
        log.warning("torch is not importable; Docling will run on CPU")
        return "cpu"
    if torch.cuda.is_available():
        log.info("CUDA available: %d device(s), using %s (torch %s, CUDA %s)",
                 torch.cuda.device_count(), torch.cuda.get_device_name(0),
                 torch.__version__, torch.version.cuda)
        return "cuda"
    log.warning("CUDA NOT available (torch %s); Docling will run on CPU", torch.__version__)
    return "cpu"


# --------------------------------------------------------------------------- PyMuPDF


def parse_pymupdf(pdf: Path, out_path: Path) -> int:
    """Extract plain text per page to JSONL (pages numbered from 1); return the page count."""
    import pymupdf

    lines: list[str] = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            lines.append(json.dumps({"page": i, "text": page.get_text("text")},
                                    ensure_ascii=False))
        n_pages = doc.page_count
    _write_atomic(out_path, "\n".join(lines) + "\n")
    return n_pages


# --------------------------------------------------------------------------- Docling


def build_docling_converter(device: str) -> Any:
    """Build a DocumentConverter for PDFs: no OCR, table structure on, on the given device."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # AcceleratorOptions moved between Docling releases; support both locations.
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    except ImportError:
        from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

    num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    accel_device = AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU

    pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=num_threads,
                                                              device=accel_device)
    log.info("Docling pipeline: do_ocr=False, do_table_structure=True, device=%s, threads=%d",
             accel_device.value, num_threads)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def parse_docling(converter: Any, pdf: Path, md_path: Path, json_path: Path) -> tuple[int, int]:
    """Convert a PDF with Docling, save Markdown + full JSON; return (n_pages, n_tables)."""
    result = converter.convert(pdf)
    doc = result.document
    # export_to_dict keeps provenance (page_no + bbox) for every text item and table.
    _write_atomic(md_path, doc.export_to_markdown())
    _write_atomic(json_path, json.dumps(doc.export_to_dict(), ensure_ascii=False))
    return doc.num_pages(), len(doc.tables)


# --------------------------------------------------------------------------- summary


def _row_from_outputs(report_id: str, previous: dict[str, dict[str, str]]) -> ReportResult:
    """Build a summary row for a skipped report, reusing timings from a previous summary."""
    row = ReportResult(id=report_id, status="skipped")
    prev = previous.get(report_id)
    if prev:
        row.device = prev.get("device") or None
        row.docling_seconds = float(prev["docling_seconds"]) if prev.get("docling_seconds") else None
        row.pymupdf_seconds = float(prev["pymupdf_seconds"]) if prev.get("pymupdf_seconds") else None
    paths = output_paths(report_id)
    try:
        with paths["pymupdf"].open(encoding="utf-8") as fh:
            row.n_pages = sum(1 for line in fh if line.strip())
        with paths["json"].open(encoding="utf-8") as fh:
            row.n_tables_docling = len(json.load(fh).get("tables", []))
    except Exception as exc:  # summary is best-effort for skipped reports
        log.warning("%s: could not read existing outputs for summary: %s", report_id, exc)
    return row


def load_previous_summary(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load an existing summary CSV keyed by id (empty dict if absent)."""
    path = path or SUMMARY_CSV
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def write_summary(results: list[ReportResult], previous: dict[str, dict[str, str]],
                  all_ids: list[str], path: Path | None = None) -> None:
    """Write the summary CSV; reports not processed in this run keep their previous row."""
    path = path or SUMMARY_CSV
    current = {r.id: asdict(r) for r in results}
    rows = [current.get(rid) or previous[rid] for rid in all_ids if rid in current or rid in previous]
    for r in rows:
        for key in ("docling_seconds", "pymupdf_seconds"):
            if isinstance(r.get(key), float):
                r[key] = f"{r[key]:.1f}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


# --------------------------------------------------------------------------- driver


def process_report(report_id: str, device: str, get_converter: Any) -> ReportResult:
    """Run PyMuPDF and Docling on one report; errors are logged and reported as status=failed."""
    row = ReportResult(id=report_id, device=device)
    pdf = RAW_DIR / f"{report_id}.pdf"
    if not pdf.exists():
        log.error("%s: PDF not found at %s (run nordicpulse.ingestion.download first)",
                  report_id, pdf)
        row.status = "missing_pdf"
        return row
    paths = output_paths(report_id)
    try:
        t0 = time.perf_counter()
        row.n_pages = parse_pymupdf(pdf, paths["pymupdf"])
        row.pymupdf_seconds = time.perf_counter() - t0
        log.info("%s: PyMuPDF done, %d pages in %.1f s", report_id, row.n_pages,
                 row.pymupdf_seconds)

        converter = get_converter()
        log.info("%s: Docling started on %s", report_id, device)
        t0 = time.perf_counter()
        docling_pages, row.n_tables_docling = parse_docling(converter, pdf, paths["markdown"],
                                                            paths["json"])
        row.docling_seconds = time.perf_counter() - t0
        log.info("%s: Docling done, %d pages, %d tables in %.1f s (%.2f s/page) on %s",
                 report_id, docling_pages, row.n_tables_docling, row.docling_seconds,
                 row.docling_seconds / max(docling_pages, 1), device)
        if docling_pages != row.n_pages:
            log.warning("%s: page count mismatch PyMuPDF=%d Docling=%d", report_id,
                        row.n_pages, docling_pages)
        row.status = "ok"
    except Exception:
        log.exception("%s: FAILED", report_id)
        row.status = "failed"
    return row


def run(ids: list[str] | None = None, force: bool = False) -> list[ReportResult]:
    """Process the selected reports (all by default) and write the summary CSV."""
    all_ids = report_ids()
    selected = ids or all_ids
    unknown = sorted(set(selected) - set(all_ids))
    if unknown:
        raise SystemExit(f"Unknown report ids: {unknown}. Valid ids: {all_ids}")

    previous = load_previous_summary()
    todo = [rid for rid in selected if force or not is_done(rid)]
    log.info("Reports selected: %d, to process: %d, already done: %d%s",
             len(selected), len(todo), len(selected) - len(todo), " (--force)" if force else "")

    device = detect_device() if todo else "n/a"
    converter_cache: dict[str, Any] = {}

    def get_converter() -> Any:
        # Built once and lazily: model loading is slow and unnecessary if all are skipped.
        if "c" not in converter_cache:
            t0 = time.perf_counter()
            converter_cache["c"] = build_docling_converter(device)
            log.info("Docling converter ready in %.1f s", time.perf_counter() - t0)
        return converter_cache["c"]

    results: list[ReportResult] = []
    for rid in selected:
        if rid not in todo:
            log.info("%s: all outputs exist, skipping (use --force to redo)", rid)
            results.append(_row_from_outputs(rid, previous))
            continue
        results.append(process_report(rid, device, get_converter))
        # Rewrite the summary after every report so progress survives a killed job.
        write_summary(results, previous, all_ids)

    write_summary(results, previous, all_ids)
    log.info("Summary written to %s", SUMMARY_CSV)
    for r in results:
        log.info("  %-20s %-11s pages=%s tables=%s docling=%s s", r.id, r.status, r.n_pages,
                 r.n_tables_docling,
                 f"{r.docling_seconds:.1f}" if r.docling_seconds is not None else "-")
    return results


def main() -> None:
    """Command-line entry point. Exits with code 1 if any report failed."""
    parser = argparse.ArgumentParser(description="Parse report PDFs with PyMuPDF and Docling.")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="report ids to process, e.g. vestas_2025 (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="reprocess reports even if all outputs exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    results = run(ids=args.ids, force=args.force)
    if any(r.status in ("failed", "missing_pdf") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
