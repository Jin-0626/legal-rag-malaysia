"""Legal-unit-aware chunking for Malaysian legal text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .models import Chunk

UNIT_LABELS = {
    "section": "Section",
    "article": "Article",
    "perkara": "Perkara",
}

EXPLICIT_UNIT_PATTERNS = {
    "section": re.compile(r"^Section\s+(?P<unit_id>\d+[A-Z]?)\b(?P<heading>.*)$"),
    "article": re.compile(r"^Article\s+(?P<unit_id>\d+[A-Z]?)\b(?P<heading>.*)$"),
    "perkara": re.compile(r"^Perkara\s+(?P<unit_id>\d+[A-Z]?)\b(?P<heading>.*)$"),
}
NUMBERED_UNIT_PATTERN = re.compile(r"^(?P<unit_id>\d+[A-Z]?)\.\s*(?P<body>.*)$")
TOP_LEVEL_BODY_PATTERN = re.compile(r"^(?P<unit_id>\d+[A-Z]?)\.\s*\((?P<subsection_id>\d+[A-Z]?)\)\s*(?P<body>.*)$")
SUBSECTION_PATTERN = re.compile(r"^\((?P<subsection_id>\d+[A-Z]?)\)\s*(?P<body>.*)$")
PARAGRAPH_PATTERN = re.compile(r"^\((?P<paragraph_id>[a-zA-Z]+)\)\s*(?P<body>.*)$")

HIERARCHY_PATTERNS = (
    re.compile(r"^(Part|PART)\s+[A-Z0-9IVXLC]+$"),
    re.compile(r"^Division\s+\d+[A-Z]?$", re.IGNORECASE),
    re.compile(r"^(Chapter|CHAPTER)\s+[A-Z0-9IVXLC]+$"),
    re.compile(r"^(Schedule|SCHEDULE)\b"),
    re.compile(r"^(Bahagian|BAHAGIAN)\s+[A-Z0-9IVXLC]+$"),
    re.compile(r"^(Bab|BAB)\s+[A-Z0-9IVXLC]+(?:—.*)?$"),
    re.compile(r"^(Jadual|JADUAL)\b"),
)


@dataclass(frozen=True)
class _LegalUnit:
    unit_type: str
    unit_id: str
    section_heading: str
    section_id: str
    subsection_id: str | None
    paragraph_id: str | None
    text: str


@dataclass(frozen=True)
class SectionBoundaryLeak:
    chunk_id: str
    assigned_section_id: str
    found_section_id: str
    line: str
    assigned_unit_type: str = "section"
    found_unit_type: str = "section"


def chunk_legal_text(
    document_id: str,
    text: str,
    source_path: str,
    act_title: str = "",
    act_number: str = "",
    max_words: int = 250,
    overlap_words: int = 40,
    unit_type_hint: str | None = None,
) -> list[Chunk]:
    """Chunk legal text by top-level legal unit, subsection, and paragraph boundaries."""

    inferred_unit_type = unit_type_hint or _infer_unit_type(text=text, source_path=source_path, act_title=act_title)
    units = _extract_legal_units(text=text, unit_type_hint=inferred_unit_type)
    if not units:
        return []

    chunks: list[Chunk] = []
    current_units: list[_LegalUnit] = []
    current_word_count = 0
    chunk_index = 0

    for unit in units:
        if current_units and unit.section_id != current_units[0].section_id:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    units=current_units,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    act_title=act_title,
                    act_number=act_number,
                )
            )
            chunk_index += 1
            current_units = []
            current_word_count = 0

        unit_word_count = len(unit.text.split())
        split_threshold = max_words if not (unit.subsection_id or unit.paragraph_id) else max_words * 3
        if unit_word_count >= split_threshold:
            if current_units:
                chunks.append(
                    _build_chunk(
                        document_id=document_id,
                        units=current_units,
                        source_path=source_path,
                        chunk_index=chunk_index,
                        act_title=act_title,
                        act_number=act_number,
                    )
                )
                chunk_index += 1
                current_units = _overlap_units(current_units, overlap_words)
                current_word_count = _word_count(current_units)

            oversized_chunks = _split_large_unit(
                document_id=document_id,
                unit=unit,
                source_path=source_path,
                act_title=act_title,
                act_number=act_number,
                max_words=max_words,
                overlap_words=overlap_words,
                chunk_index_start=chunk_index,
            )
            chunks.extend(oversized_chunks)
            chunk_index += len(oversized_chunks)
            current_units = []
            current_word_count = 0
            continue

        prospective_count = current_word_count + unit_word_count
        if current_units and prospective_count > max_words:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    units=current_units,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    act_title=act_title,
                    act_number=act_number,
                )
            )
            chunk_index += 1
            current_units = _overlap_units(current_units, overlap_words)
            current_word_count = _word_count(current_units)

        current_units.append(unit)
        current_word_count += unit_word_count

    if current_units:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                units=current_units,
                source_path=source_path,
                chunk_index=chunk_index,
                act_title=act_title,
                act_number=act_number,
            )
        )

    return chunks


def find_section_boundary_leaks(chunks: list[Chunk]) -> list[SectionBoundaryLeak]:
    """Detect chunks that contain a top-level legal unit marker for a different unit."""

    leaks: list[SectionBoundaryLeak] = []
    for chunk in chunks:
        chunk_unit_type = chunk.unit_type or "section"
        seen_units: set[tuple[str, str]] = set()
        for raw_line in chunk.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            found = _line_unit_match(line=line, default_unit_type=chunk_unit_type)
            if (
                found
                and found[0] == chunk_unit_type
                and found[1] != chunk.section_id
                and found not in seen_units
            ):
                seen_units.add(found)
                leaks.append(
                    SectionBoundaryLeak(
                        chunk_id=chunk.chunk_id,
                        assigned_section_id=chunk.section_id,
                        found_section_id=found[1],
                        line=line,
                        assigned_unit_type=chunk_unit_type,
                        found_unit_type=found[0],
                    )
                )
    return leaks


def chunk_section_text(
    document_id: str,
    section_heading: str,
    text: str,
    source_path: str,
    act_title: str = "",
    act_number: str = "",
    max_words: int = 250,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Chunk one legal unit while preserving subsection and paragraph integrity."""

    normalized_heading = section_heading.strip()
    if not normalized_heading or not text.strip():
        return []

    unit_type = _infer_unit_type_from_heading(normalized_heading) or _infer_unit_type(
        text=normalized_heading,
        source_path=source_path,
        act_title=act_title,
    )
    section_text = f"{normalized_heading}\n{text.strip()}"
    return chunk_legal_text(
        document_id=document_id,
        text=section_text,
        source_path=source_path,
        act_title=act_title,
        act_number=act_number,
        max_words=max_words,
        overlap_words=overlap_words,
        unit_type_hint=unit_type,
    )


def _extract_legal_units(text: str, unit_type_hint: str) -> list[_LegalUnit]:
    current_unit_type = unit_type_hint
    current_section_id: str | None = None
    current_section_heading = ""
    current_subsection_id: str | None = None
    current_paragraph_id: str | None = None
    current_lines: list[str] = []
    units: list[_LegalUnit] = []
    pending_section_heading: str | None = None

    def flush_current_unit() -> None:
        nonlocal current_lines
        if not current_section_id or not current_lines:
            current_lines = []
            return

        normalized_text = "\n".join(line.strip() for line in current_lines if line.strip())
        if not normalized_text:
            current_lines = []
            return

        units.append(
            _LegalUnit(
                unit_type=current_unit_type,
                unit_id=current_section_id,
                section_heading=current_section_heading,
                section_id=current_section_id,
                subsection_id=current_subsection_id,
                paragraph_id=current_paragraph_id,
                text=normalized_text,
            )
        )
        current_lines = []

    lines = [raw_line.strip() for raw_line in text.splitlines() if raw_line.strip()]

    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else None

        explicit_match = _explicit_unit_match(line)
        if explicit_match and explicit_match[0] == unit_type_hint:
            flush_current_unit()
            current_unit_type = explicit_match[0]
            current_section_id = explicit_match[1]
            current_section_heading = explicit_match[2]
            current_subsection_id = None
            current_paragraph_id = None
            current_lines = [line]
            pending_section_heading = None
            continue

        numbered_unit_match = NUMBERED_UNIT_PATTERN.match(line)
        if numbered_unit_match and _looks_like_unit_start(
            line=line,
            pending_section_heading=pending_section_heading,
            current_section_id=current_section_id,
        ):
            flush_current_unit()
            current_section_id = numbered_unit_match.group("unit_id")
            current_section_heading = _build_numbered_unit_heading(
                unit_type=current_unit_type,
                unit_id=current_section_id,
                pending_heading=pending_section_heading,
            )
            current_subsection_id = _extract_subsection_id(numbered_unit_match.group("body"))
            current_paragraph_id = None
            current_lines = [current_section_heading, line]
            pending_section_heading = None
            continue

        subsection_match = SUBSECTION_PATTERN.match(line)
        if subsection_match and current_section_id:
            if current_lines == [current_section_heading]:
                current_subsection_id = subsection_match.group("subsection_id")
                current_paragraph_id = None
                current_lines.append(line)
                continue

            flush_current_unit()
            current_subsection_id = subsection_match.group("subsection_id")
            current_paragraph_id = None
            current_lines = [line]
            pending_section_heading = None
            continue

        paragraph_match = PARAGRAPH_PATTERN.match(line)
        if paragraph_match and current_section_id:
            if current_subsection_id:
                current_lines.append(line)
            else:
                flush_current_unit()
                current_paragraph_id = paragraph_match.group("paragraph_id")
                current_lines = [line]
            pending_section_heading = None
            continue

        if _can_be_unit_heading(line, next_line):
            pending_section_heading = line
            continue

        pending_section_heading = None
        current_lines.append(line)

    flush_current_unit()
    return units


def _build_chunk(
    document_id: str,
    units: list[_LegalUnit],
    source_path: str,
    chunk_index: int,
    act_title: str,
    act_number: str,
) -> Chunk:
    first_unit = units[0]
    subsection_id = first_unit.subsection_id if all(
        unit.subsection_id == first_unit.subsection_id for unit in units
    ) else None
    paragraph_id = first_unit.paragraph_id if all(
        unit.paragraph_id == first_unit.paragraph_id for unit in units
    ) else None

    return Chunk(
        chunk_id=f"{document_id}:{first_unit.section_id}:{chunk_index}",
        document_id=document_id,
        section_heading=first_unit.section_heading,
        section_id=first_unit.section_id,
        subsection_id=subsection_id,
        paragraph_id=paragraph_id,
        text="\n".join(unit.text for unit in units),
        source_path=source_path,
        act_title=act_title,
        act_number=act_number,
        source_file=Path(source_path).name,
        chunk_index=chunk_index,
        unit_type=first_unit.unit_type,
        unit_id=first_unit.unit_id,
    )


def _split_large_unit(
    document_id: str,
    unit: _LegalUnit,
    source_path: str,
    act_title: str,
    act_number: str,
    max_words: int,
    overlap_words: int,
    chunk_index_start: int,
) -> list[Chunk]:
    words = unit.text.split()
    step = max(1, max_words - overlap_words)
    chunks: list[Chunk] = []

    for offset, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:{unit.section_id}:{chunk_index_start + offset}",
                document_id=document_id,
                section_heading=unit.section_heading,
                section_id=unit.section_id,
                subsection_id=unit.subsection_id,
                paragraph_id=unit.paragraph_id,
                text=chunk_text,
                source_path=source_path,
                act_title=act_title,
                act_number=act_number,
                source_file=Path(source_path).name,
                chunk_index=chunk_index_start + offset,
                unit_type=unit.unit_type,
                unit_id=unit.unit_id,
            )
        )

    return chunks


def _overlap_units(units: list[_LegalUnit], overlap_words: int) -> list[_LegalUnit]:
    if overlap_words <= 0:
        return []

    overlapped: list[_LegalUnit] = []
    running_words = 0
    target_subsection_id = units[-1].subsection_id
    for unit in reversed(units):
        unit_words = len(unit.text.split())
        if (
            overlapped
            and target_subsection_id is not None
            and unit.subsection_id != target_subsection_id
        ):
            break
        if overlapped and running_words + unit_words > overlap_words:
            break
        overlapped.insert(0, unit)
        running_words += unit_words
        if running_words >= overlap_words:
            break

    return overlapped


def _word_count(units: list[_LegalUnit]) -> int:
    return sum(len(unit.text.split()) for unit in units)


def _can_be_unit_heading(line: str, next_line: str | None) -> bool:
    if not next_line or not NUMBERED_UNIT_PATTERN.match(next_line):
        return False
    if any(pattern.match(line) for pattern in HIERARCHY_PATTERNS):
        return False
    if _explicit_unit_match(line) or line in {"Section", "Article", "Perkara"}:
        return False
    if re.fullmatch(r"\d+", line):
        return False
    if line.endswith((".", ";", ":", ",")):
        return False
    return True


def _looks_like_unit_start(
    line: str,
    pending_section_heading: str | None,
    current_section_id: str | None,
) -> bool:
    if pending_section_heading:
        return True
    if current_section_id is None:
        return True
    return not line.startswith(f"{current_section_id}.")


def _build_numbered_unit_heading(unit_type: str, unit_id: str, pending_heading: str | None) -> str:
    heading = f"{UNIT_LABELS.get(unit_type, 'Section')} {unit_id}"
    if pending_heading:
        heading = f"{heading} {pending_heading}"
    return heading


def _extract_subsection_id(section_body: str) -> str | None:
    subsection_match = SUBSECTION_PATTERN.match(section_body.strip())
    if subsection_match:
        return subsection_match.group("subsection_id")
    return None


def _explicit_unit_match(line: str) -> tuple[str, str, str] | None:
    stripped = line.strip()
    for unit_type, pattern in EXPLICIT_UNIT_PATTERNS.items():
        match = pattern.match(stripped)
        if match:
            unit_id = match.group("unit_id")
            heading = _build_numbered_unit_heading(unit_type=unit_type, unit_id=unit_id, pending_heading=match.group("heading").strip() or None)
            return unit_type, unit_id, heading
    return None


def _line_unit_match(line: str, default_unit_type: str) -> tuple[str, str] | None:
    explicit = _explicit_unit_match(line)
    if explicit:
        return explicit[0], explicit[1]

    numbered_match = TOP_LEVEL_BODY_PATTERN.match(line) or NUMBERED_UNIT_PATTERN.match(line)
    if numbered_match:
        return default_unit_type, numbered_match.group("unit_id")

    return None


def _infer_unit_type(text: str, source_path: str, act_title: str) -> str:
    lowered = "\n".join([text[:2000], source_path, act_title]).lower()
    if "perlembagaan" in lowered or "perkara" in lowered or "bahagian" in lowered:
        return "perkara"
    if "constitution" in lowered or "article" in lowered:
        return "article"
    return "section"


def _infer_unit_type_from_heading(heading: str) -> str | None:
    lowered = heading.lower()
    for unit_type, label in UNIT_LABELS.items():
        if lowered.startswith(label.lower()):
            return unit_type
    return None
