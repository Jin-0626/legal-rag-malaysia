"""End-to-end law PDF ingestion and flat chunk export helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from legal_rag.chunking.models import Chunk
from legal_rag.chunking.section_chunker import chunk_legal_text
from legal_rag.ingestion.parse_pdf import LawDocumentText, extract_law_document_text

ACT_NUMBER_PATTERN = re.compile(r"\bAct\s+\d+[A-Z]?\b")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
ACT_YEAR_PATTERN = re.compile(r"^ACT\s+\d{4}$")


def ingest_law_pdf_to_chunks(
    pdf_path: Path,
    max_words: int = 250,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Parse a law PDF, normalize extracted text, and return legal-aware chunks."""

    document = extract_law_document_text(pdf_path)
    normalized_text = normalize_law_document_text(document)
    if not normalized_text.strip():
        return []

    return chunk_legal_text(
        document_id=document.document_id,
        text=normalized_text,
        source_path=document.source_path,
        act_title=derive_act_title(document),
        act_number=derive_act_number(document),
        max_words=max_words,
        overlap_words=overlap_words,
    )


def normalize_law_document_text(document: LawDocumentText) -> str:
    """Normalize extracted page lines into chunker-ready statute text."""

    act_title = derive_act_title(document)
    act_number = derive_act_number(document)
    normalized_lines: list[str] = []
    skipping_contents = False

    for page in document.pages:
        for line in page.lines:
            cleaned_line = WHITESPACE_PATTERN.sub(" ", line).strip()
            if cleaned_line:
                if cleaned_line == "ARRANGEMENT OF SECTIONS":
                    skipping_contents = True
                    continue

                if skipping_contents:
                    if cleaned_line.startswith("ENACTED"):
                        skipping_contents = False
                    else:
                        continue

                if _is_running_header_or_footer(cleaned_line, act_title, act_number):
                    continue

                normalized_lines.append(cleaned_line)
        normalized_lines.append("")

    return "\n".join(normalized_lines).strip()


def derive_act_title(document: LawDocumentText) -> str:
    """Derive a human-readable act title from extracted text."""

    first_page_lines = document.pages[0].lines if document.pages else []
    for index, line in enumerate(first_page_lines):
        cleaned = line.strip()
        if ACT_YEAR_PATTERN.fullmatch(cleaned):
            title_prefix = _collect_title_prefix(first_page_lines, index)
            if title_prefix:
                return f"{title_prefix} {cleaned.title()}"

    for index, line in enumerate(first_page_lines):
        cleaned = line.strip()
        if not cleaned:
            continue
        if ACT_NUMBER_PATTERN.fullmatch(cleaned):
            continue
        if PART_OR_DIVISION_PATTERN.fullmatch(cleaned):
            continue
        if cleaned.isupper() and len(cleaned.split()) <= 6:
            continue
        if cleaned.startswith("LAWS OF MALAYSIA") or cleaned.startswith("REPRINT"):
            continue
        if cleaned.isdigit():
            continue
        return cleaned

    return document.title.replace("-", " ").title()


def derive_act_number(document: LawDocumentText) -> str:
    """Extract the act number from the first few pages when present."""

    for page in document.pages[:5]:
        for line in page.lines:
            match = ACT_NUMBER_PATTERN.fullmatch(line.strip())
            if match:
                return match.group(0)
    return ""


def chunk_to_record(chunk: Chunk) -> dict[str, Any]:
    """Convert a chunk into a flat serializable export record."""

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "act_title": chunk.act_title,
        "act_number": chunk.act_number,
        "section_heading": chunk.section_heading,
        "section_id": chunk.section_id,
        "subsection_id": chunk.subsection_id,
        "paragraph_id": chunk.paragraph_id,
        "source_file": chunk.source_file,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
    }


def export_chunks_to_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    """Write flat chunk records to JSONL for downstream retrieval ingestion."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk_to_record(chunk), ensure_ascii=False) + "\n")


PART_OR_DIVISION_PATTERN = re.compile(r"^(Part|PART|Division)\b")


def _collect_title_prefix(lines: list[str], act_year_index: int) -> str:
    candidates: list[str] = []
    for reverse_index in range(act_year_index - 1, -1, -1):
        cleaned = lines[reverse_index].strip()
        if not cleaned or cleaned.isdigit():
            continue
        if cleaned.startswith("LAWS OF MALAYSIA") or cleaned == "REPRINT":
            continue
        if ACT_NUMBER_PATTERN.search(cleaned):
            continue
        if PART_OR_DIVISION_PATTERN.fullmatch(cleaned):
            continue
        if not cleaned.isupper():
            break
        candidates.insert(0, cleaned.title())
    return " ".join(candidates).strip()


def _is_running_header_or_footer(line: str, act_title: str, act_number: str) -> bool:
    if line.isdigit():
        return True
    if line in {"Laws of Malaysia", "LAWS OF MALAYSIA", "REPRINT"}:
        return True
    if act_number and line == act_number:
        return True
    if act_title and line == act_title:
        return True
    return False
