"""Typed settings for production modules and law PDF ingestion paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LegalRAGSettings:
    """Project paths and defaults used by modular pipeline components."""

    project_root: Path
    data_dir: Path
    raw_law_pdfs_dir: Path
    processed_dir: Path
    embeddings_dir: Path
    chunk_size_words: int = 250
    chunk_overlap_words: int = 40


def build_settings(project_root: Path | None = None) -> LegalRAGSettings:
    """Build settings with directories prepared for real law PDF ingestion."""

    root = project_root or Path(__file__).resolve().parents[3]
    data_dir = root / "data"
    raw_law_pdfs_dir = data_dir / "raw_law_pdfs"
    processed_dir = data_dir / "processed"
    embeddings_dir = data_dir / "embeddings"

    for directory in (raw_law_pdfs_dir, processed_dir, embeddings_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return LegalRAGSettings(
        project_root=root,
        data_dir=data_dir,
        raw_law_pdfs_dir=raw_law_pdfs_dir,
        processed_dir=processed_dir,
        embeddings_dir=embeddings_dir,
    )
