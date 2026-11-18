"""Tests for the CUDA-failure handling of nordicpulse.ingestion.parse.

They use fake Docling converters, so neither torch, docling nor a GPU is needed.
"""

from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import nordicpulse.ingestion.parse as parse

KERNEL_MSG = "CUDA error: no kernel image is available for execution on the device"


class _Status(Enum):  # stands in for docling's ConversionStatus
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


class AcceleratorError(RuntimeError):  # same class name as torch.AcceleratorError
    pass


class OutOfMemoryError(RuntimeError):  # same class name as torch.OutOfMemoryError
    pass


def _result(status: _Status, messages: list[str]) -> SimpleNamespace:
    doc = SimpleNamespace(export_to_markdown=lambda: "# md", export_to_dict=lambda: {},
                          num_pages=lambda: 2, tables=[])
    errors = [SimpleNamespace(error_message=m) for m in messages]
    return SimpleNamespace(status=status, errors=errors, document=doc)


class FakeConverter:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls = 0

    def convert(self, pdf: Path, raises_on_error: bool = True) -> SimpleNamespace:
        assert raises_on_error is False
        self.calls += 1
        return self.result


def test_is_cuda_error() -> None:
    assert parse.is_cuda_error(AcceleratorError(KERNEL_MSG))
    assert parse.is_cuda_error(RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED"))
    assert parse.is_cuda_error(KERNEL_MSG)
    wrapped = RuntimeError("layout model failed")
    wrapped.__cause__ = AcceleratorError(KERNEL_MSG)
    assert parse.is_cuda_error(wrapped)
    assert not parse.is_cuda_error(OutOfMemoryError("CUDA out of memory. Tried to allocate"))
    assert not parse.is_cuda_error(ValueError("table structure mismatch"))


def test_parse_docling_raises_gpu_error_on_cuda_page_errors(tmp_path: Path) -> None:
    conv = FakeConverter(_result(_Status.PARTIAL_SUCCESS, [KERNEL_MSG] * 3))
    with pytest.raises(parse.GpuError, match="3 CUDA error"):
        parse.parse_docling(conv, tmp_path / "x.pdf", tmp_path / "x.md", tmp_path / "x.json")
    assert not (tmp_path / "x.md").exists()  # nothing saved for a GPU failure


def test_parse_docling_saves_partial_success_without_cuda_errors(tmp_path: Path) -> None:
    conv = FakeConverter(_result(_Status.PARTIAL_SUCCESS, ["page 7: table model timeout"]))
    assert parse.parse_docling(conv, tmp_path / "x.pdf", tmp_path / "x.md",
                               tmp_path / "x.json") == (2, 0)
    assert (tmp_path / "x.md").exists() and (tmp_path / "x.json").exists()


def test_parse_docling_raises_on_failure(tmp_path: Path) -> None:
    conv = FakeConverter(_result(_Status.FAILURE, ["broken pdf"]))
    with pytest.raises(RuntimeError, match="FAILURE"):
        parse.parse_docling(conv, tmp_path / "x.pdf", tmp_path / "x.md", tmp_path / "x.json")


@pytest.fixture
def fake_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Three reports with PDFs present; outputs and summary under tmp_path."""
    ids = ["a_2024", "b_2024", "c_2024"]
    raw = tmp_path / "raw"
    raw.mkdir()
    for rid in ids:
        (raw / f"{rid}.pdf").write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(parse, "RAW_DIR", raw)
    monkeypatch.setattr(parse, "PYMUPDF_DIR", tmp_path / "pymupdf")
    monkeypatch.setattr(parse, "DOCLING_DIR", tmp_path / "docling")
    monkeypatch.setattr(parse, "SUMMARY_CSV", tmp_path / "summary.csv")
    monkeypatch.setattr(parse, "report_ids", lambda: ids)
    monkeypatch.setattr(parse, "detect_device", lambda require_gpu=False: "cuda")
    monkeypatch.setattr(parse, "parse_pymupdf", lambda pdf, out: 2)
    return ids


def test_run_stops_at_first_cuda_failure(fake_project: list[str],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    conv = FakeConverter(_result(_Status.PARTIAL_SUCCESS, [KERNEL_MSG]))
    monkeypatch.setattr(parse, "build_docling_converter", lambda device: conv)
    with pytest.raises(parse.GpuError):
        parse.run()
    assert conv.calls == 1  # b_2024 and c_2024 were never attempted
    summary = (parse.SUMMARY_CSV).read_text(encoding="utf-8")
    assert "a_2024" in summary and "failed" in summary and "b_2024" not in summary


def test_run_continues_after_non_cuda_failure(fake_project: list[str],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    conv = FakeConverter(_result(_Status.FAILURE, ["broken pdf"]))
    monkeypatch.setattr(parse, "build_docling_converter", lambda device: conv)
    results = parse.run()
    assert conv.calls == 3
    assert [r.status for r in results] == ["failed"] * 3


def test_require_gpu_without_torch_or_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert parse.detect_device(require_gpu=False) == "cpu"
    with pytest.raises(parse.GpuError):
        parse.detect_device(require_gpu=True)
