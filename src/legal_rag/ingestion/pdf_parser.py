"""Backward-compatible imports for PDF parsing helpers."""

from .parse_pdf import (
    ExtractedPdfPage,
    LawDocumentText,
    PdfIngestionError,
    extract_law_document_text,
)

__all__ = [
    "ExtractedPdfPage",
    "LawDocumentText",
    "PdfIngestionError",
    "extract_law_document_text",
]
