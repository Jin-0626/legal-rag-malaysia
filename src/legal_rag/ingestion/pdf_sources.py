"""PDF discovery helpers for future law ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LawPdfSource:
    """Flat, serializable metadata describing a source law PDF."""

    document_id: str
    title: str
    path: str


def discover_law_pdfs(root: Path) -> list[LawPdfSource]:
    """Return all PDFs under a root directory in a stable order."""

    pdf_paths = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    return [
        LawPdfSource(
            document_id=path.stem.lower().replace(" ", "_"),
            title=path.stem,
            path=str(path),
        )
        for path in pdf_paths
    ]
