"""Chunk metadata models used across retrieval pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    """Minimal chunk record with section-aware metadata."""

    chunk_id: str
    document_id: str
    section_heading: str
    section_id: str
    subsection_id: Optional[str]
    paragraph_id: Optional[str]
    text: str
    source_path: str
    act_title: str = ""
    act_number: str = ""
    source_file: str = ""
    chunk_index: int = 0
