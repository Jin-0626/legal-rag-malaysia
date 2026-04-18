"""PDF text extraction helpers for real law-document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


class PdfIngestionError(RuntimeError):
    """Raised when a law PDF cannot be converted into usable text."""


@dataclass(frozen=True)
class ExtractedPdfPage:
    """Structured page text that preserves line boundaries for chunking."""

    page_number: int
    lines: list[str]
    text: str


@dataclass(frozen=True)
class LawDocumentText:
    """Structured text extracted from one law PDF."""

    document_id: str
    title: str
    source_path: str
    pages: list[ExtractedPdfPage]
    full_text: str


def extract_law_document_text(pdf_path: Path) -> LawDocumentText:
    """Extract structured text from a PDF while preserving line structure."""

    path = Path(pdf_path)
    if not path.exists():
        raise PdfIngestionError(f"Law PDF not found: {path}")

    try:
        with fitz.open(path) as document:
            pages = [_extract_page(page) for page in document]
    except (fitz.FileDataError, ValueError) as exc:
        raise PdfIngestionError(f"Unable to open PDF for extraction: {path}") from exc

    non_empty_pages = [page for page in pages if page.lines]
    if not non_empty_pages:
        raise PdfIngestionError(f"PDF extraction produced no usable text: {path}")

    full_text = "\n\n".join(page.text for page in non_empty_pages)
    return LawDocumentText(
        document_id=path.stem.lower().replace(" ", "_"),
        title=path.stem,
        source_path=str(path),
        pages=non_empty_pages,
        full_text=full_text,
    )


def _extract_page(page: fitz.Page) -> ExtractedPdfPage:
    raw_text = page.get_text("text")
    normalized_lines = [line.rstrip() for line in raw_text.splitlines() if line.strip()]

    return ExtractedPdfPage(
        page_number=page.number + 1,
        lines=normalized_lines,
        text="\n".join(normalized_lines),
    )
