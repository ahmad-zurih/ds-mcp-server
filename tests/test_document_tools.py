"""Tests for the document / file-intelligence tools.

Real-extraction tests are guarded with ``pytest.importorskip`` so the suite
stays green whether or not the optional ``[documents]`` / ``[ocr]`` /
``[profiling]`` extras are installed. The graceful error paths (missing file,
wrong extension, missing library) are always exercised.
"""
from __future__ import annotations

import os

import pytest

from ds_mcp_server._tools import document_tools as dt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_check_file_missing(self):
        assert dt._check_file("/no/such/file.pdf").startswith("Error: file not found")

    def test_check_file_wrong_extension(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("hi")
        err = dt._check_file(str(f), (".pdf",))
        assert err is not None
        assert "unsupported file type" in err

    def test_check_file_ok(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("hi")
        assert dt._check_file(str(f), (".txt",)) is None

    def test_truncate(self):
        out = dt._truncate("x" * 100, 10)
        assert out.startswith("x" * 10)
        assert "truncated" in out

    def test_parse_page_range(self):
        assert dt._parse_page_range("1,3,5-6", 10) == [0, 2, 4, 5]
        assert dt._parse_page_range(None, 3) == [0, 1, 2]
        # Out-of-range values are ignored.
        assert dt._parse_page_range("99", 3) == []


# ---------------------------------------------------------------------------
# Missing-file / wrong-type error paths (no optional deps required)
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_read_pdf_missing(self):
        assert dt.read_pdf_impl("/nope.pdf").startswith("Error: file not found")

    def test_read_docx_wrong_ext(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert "unsupported file type" in dt.read_docx_impl(str(f))

    def test_read_excel_missing(self):
        assert dt.read_excel_sheets_impl("/nope.xlsx").startswith("Error: file not found")

    def test_ocr_missing_file(self):
        assert dt.ocr_image_impl("/nope.png").startswith("Error: file not found")

    def test_summarize_missing(self):
        assert dt.summarize_document_impl("/nope.txt").startswith("Error: file not found")


# ---------------------------------------------------------------------------
# summarize_document — plain text needs no optional dependency
# ---------------------------------------------------------------------------


class TestSummarizeText:
    def test_summarize_txt(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("The quick brown fox. " * 50)
        out = dt.summarize_document_impl(str(f))
        assert "doc.txt" in out
        assert "quick brown fox" in out

    def test_summarize_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   ")
        out = dt.summarize_document_impl(str(f))
        assert "No extractable text" in out


# ---------------------------------------------------------------------------
# Real extraction (guarded by importorskip)
# ---------------------------------------------------------------------------


class TestPdf:
    def test_read_pdf(self, tmp_path):
        pytest.importorskip("reportlab")
        pytest.importorskip("pypdf")
        from reportlab.pdfgen import canvas

        pdf = tmp_path / "hello.pdf"
        c = canvas.Canvas(str(pdf))
        c.drawString(100, 750, "Hello PDF World")
        c.save()

        out = dt.read_pdf_impl(str(pdf))
        assert "Hello PDF World" in out


class TestDocx:
    def test_read_docx(self, tmp_path):
        docx = pytest.importorskip("docx")

        path = tmp_path / "doc.docx"
        d = docx.Document()
        d.add_paragraph("Paragraph one.")
        d.add_paragraph("Paragraph two.")
        d.save(str(path))

        out = dt.read_docx_impl(str(path))
        assert "Paragraph one." in out
        assert "Paragraph two." in out


class TestExcel:
    def test_read_excel_sheets(self, tmp_path):
        pytest.importorskip("openpyxl")
        import pandas as pd

        path = tmp_path / "book.xlsx"
        with pd.ExcelWriter(str(path)) as writer:
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(
                writer, sheet_name="First", index=False
            )
            pd.DataFrame({"x": [9]}).to_excel(writer, sheet_name="Second", index=False)

        out = dt.read_excel_sheets_impl(str(path))
        assert "First" in out
        assert "Second" in out


class TestOcrGraceful:
    def test_ocr_reports_missing_dependency_or_runs(self, tmp_path):
        Image = pytest.importorskip("PIL.Image")
        img = tmp_path / "blank.png"
        Image.new("RGB", (20, 20), "white").save(str(img))

        out = dt.ocr_image_impl(str(img))
        # Either tesseract isn't installed (clear message) or OCR ran (string).
        assert isinstance(out, str) and out
