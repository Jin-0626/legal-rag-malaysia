"""End-to-end law PDF ingestion and flat chunk export helpers."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from legal_rag.chunking.models import Chunk
from legal_rag.chunking.section_chunker import chunk_legal_text
from legal_rag.ingestion.parse_pdf import LawDocumentText, extract_law_document_text

ACT_NUMBER_PATTERN = re.compile(r"\bAct\s+(?:\d+[A-Z]?|A\d+)\b", re.IGNORECASE)
ORDER_NUMBER_PATTERN = re.compile(r"\bP\.U\.\s*\([A-Z]\)\s*\d+\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
ACT_YEAR_PATTERN = re.compile(r"^ACT\s+\d{4}$")
BODY_START_PATTERN = re.compile(r"^\d+[A-Z]?\.\s*\((\d+[A-Z]?)\)")
NUMBERED_PROVISION_PATTERN = re.compile(r"^\d+[A-Z]?\.\s*\S")
NUMBERED_PROVISION_ONLY_PATTERN = re.compile(r"^\d+[A-Z]?\.$")
EXPLICIT_UNIT_HEADING_PATTERN = re.compile(r"^(Section|Article|Perkara)\s+\d+[A-Z]?\b", re.IGNORECASE)
HIERARCHY_CONTEXT_PATTERN = re.compile(
    r"^(Part|PART|Division|Chapter|CHAPTER|Schedule|SCHEDULE|Bahagian|BAHAGIAN|Bab|BAB|Jadual|JADUAL)\b"
)
UPPER_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z \-&,()]+$")
UPPER_TITLE_LINE_PATTERN = re.compile(r"^[A-Z][A-Z0-9 \-&,()/.'’]+$")
DATE_ONLY_PATTERN = re.compile(r"^\d{1,2}\s+[A-Za-z]+\s+\d{4}$")
TITLE_NOTICE_PATTERN = re.compile(
    r"^(as at\s+\d{1,2}\s+[a-z]+\s+\d{4}|incorporating all amendments|online version|text of reprint|reprint|published by|under the authority|this text is|commissioner of law revision|percetakan nasional|jabatan peguam negara|attorney general.?s chambers|warta kerajaan persekutuan|federal government|gazette|disiarkan oleh|unannotated statutes of malaysia)",
    re.IGNORECASE,
)
TITLE_HEADER_SKIP_PATTERN = re.compile(
    r"^(laws of malaysia|act\s+\d+[a-z]?|p\.u\.\s*\([a-z]\)\s*\d+|online version of updated|online version of updated text of reprint|text of reprint|reprint|federal constitution|perlembagaan persekutuan)$",
    re.IGNORECASE,
)
CONTENTS_MARKER_PATTERN = re.compile(r"(arrangement of sections|arrangement of articles|contents)", re.IGNORECASE)
TABLE_OF_CONTENTS_ENTRY_PATTERN = re.compile(
    r"^(Section\s+\d+[A-Z]?\b|\d+[A-Z]?\.\S?|(Chapter|Division|Part)\b|SECTION$)",
    re.IGNORECASE,
)
HEADER_NOISE_PREFIX_PATTERN = re.compile(r"^(?:ACT\s+[A-Z0-9]+[,./\\-]*|P\.U\.\s*\([A-Z]\)\s*\d+[,./\\-]*)+", re.IGNORECASE)
STANDARD_LEGAL_ALIASES = (
    {
        "match_terms": ("akta kerja 1955", "employment act 1955"),
        "act_title": "Employment Act 1955",
        "aliases": ("Employment Act 1955", "Akta Kerja 1955", "Act 265"),
    },
    {
        "match_terms": ("federal constitution",),
        "act_title": "Federal Constitution",
        "aliases": ("Federal Constitution", "Constitution of Malaysia"),
    },
    {
        "match_terms": ("perlembagaan persekutuan",),
        "act_title": "Perlembagaan Persekutuan",
        "aliases": ("Perlembagaan Persekutuan",),
    },
    {
        "match_terms": ("personal data protection act 2010", "pdpa"),
        "act_title": "Personal Data Protection Act 2010",
        "aliases": ("Personal Data Protection Act 2010", "PDPA", "Act 709"),
    },
    {
        "match_terms": ("act a1727", "personal data protection amendment act 2024"),
        "act_title": "Personal Data Protection (Amendment) Act 2024",
        "aliases": ("Personal Data Protection (Amendment) Act 2024", "PDPA Amendment Act 2024", "Act A1727"),
    },
    {
        "match_terms": ("employees provident fund act 1991", "empoyees provident fund act 1991", "act 452"),
        "act_title": "Employees Provident Fund Act 1991",
        "aliases": ("Employees Provident Fund Act 1991", "EPF Act 1991", "Act 452"),
    },
    {
        "match_terms": ("minimum wages order 2024", "perintah gaji minimum 2024", "p u a 376", "pua 376"),
        "act_title": "Minimum Wages Order 2024",
        "aliases": ("Minimum Wages Order 2024", "Perintah Gaji Minimum 2024", "P.U. (A) 376"),
    },
    {
        "match_terms": ("factories and machinery act 1967", "act 139"),
        "act_title": "Factories and Machinery Act 1967",
        "aliases": ("Factories and Machinery Act 1967", "Act 139"),
    },
    {
        "match_terms": ("sale of goods act 1957", "act 382"),
        "act_title": "Sale of Goods Act 1957",
        "aliases": ("Sale of Goods Act 1957", "Act 382"),
    },
    {
        "match_terms": ("employment insurance system act 2017", "act 800"),
        "act_title": "Employment Insurance System Act 2017",
        "aliases": ("Employment Insurance System Act 2017", "Act 800"),
    },
    {
        "match_terms": ("consumer protection act 1999", "act 599"),
        "act_title": "Consumer Protection Act 1999",
        "aliases": ("Consumer Protection Act 1999", "Act 599"),
    },
    {
        "match_terms": ("national land code", "act 828"),
        "act_title": "National Land Code",
        "aliases": ("National Land Code", "Act 828"),
    },
    {
        "match_terms": ("penal code", "act 574", "akta 574"),
        "act_title": "Penal Code",
        "aliases": ("Penal Code", "Act 574", "Akta 574"),
    },
    {
        "match_terms": ("industrial relations act 1967", "act 177"),
        "act_title": "Industrial Relations Act 1967",
        "aliases": ("Industrial Relations Act 1967", "Act 177"),
    },
    {
        "match_terms": ("police act 1967", "act 344"),
        "act_title": "Police Act 1967",
        "aliases": ("Police Act 1967", "Act 344"),
    },
    {
        "match_terms": ("civil law act 1956", "act 67"),
        "act_title": "Civil Law Act 1956",
        "aliases": ("Civil Law Act 1956", "Act 67"),
    },
    {
        "match_terms": ("contracts act 1950", "act 136"),
        "act_title": "Contracts Act 1950",
        "aliases": ("Contracts Act 1950", "Act 136"),
    },
    {
        "match_terms": ("income tax act 1967", "act 53"),
        "act_title": "Income Tax Act 1967",
        "aliases": ("Income Tax Act 1967", "Act 53"),
    },
    {
        "match_terms": ("occupational safety and health act 1994", "act 514"),
        "act_title": "Occupational Safety and Health Act 1994",
        "aliases": ("Occupational Safety and Health Act 1994", "Act 514", "OSHA 1994"),
    },
    {
        "match_terms": ("road transport act 1987", "act 333"),
        "act_title": "Road Transport Act 1987",
        "aliases": ("Road Transport Act 1987", "Act 333"),
    },
    {
        "match_terms": ("holidays act 1951", "act 369"),
        "act_title": "Holidays Act 1951",
        "aliases": ("Holidays Act 1951", "Act 369"),
    },
    {
        "match_terms": ("law reform marriage and divorce act 1976", "law reform marriage and divorce", "act 164"),
        "act_title": "Law Reform (Marriage and Divorce) Act 1976",
        "aliases": ("Law Reform (Marriage and Divorce) Act 1976", "Act 164"),
    },
    {
        "match_terms": ("children and young persons employment act 1966", "act 350"),
        "act_title": "Children and Young Persons (Employment) Act 1966",
        "aliases": ("Children and Young Persons (Employment) Act 1966", "Act 350"),
    },
)
CANONICAL_DOCUMENT_METADATA = STANDARD_LEGAL_ALIASES


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

    chunks = chunk_legal_text(
        document_id=document.document_id,
        text=normalized_text,
        source_path=document.source_path,
        act_title=derive_act_title(document),
        act_number=derive_act_number(document),
        max_words=max_words,
        overlap_words=overlap_words,
        unit_type_hint=derive_unit_type(document),
    )
    document_aliases = derive_document_aliases(document)
    return [replace(chunk, document_aliases=document_aliases) for chunk in chunks]


def normalize_law_document_text(document: LawDocumentText) -> str:
    """Normalize extracted page lines into chunker-ready statute text."""

    act_title = derive_act_title(document)
    act_number = derive_act_number(document)
    normalized_lines: list[str] = []
    pre_body_buffer: list[str] = []
    started_body = False

    for page in document.pages:
        if _is_editorial_note_page(page.lines):
            continue
        if not started_body and _is_contents_or_index_page(page.lines):
            continue

        page_lines = _prepare_page_lines(page.lines)
        for line_index, line in enumerate(page_lines):
            cleaned_line = _normalize_extracted_line(line)
            if not cleaned_line:
                continue

            if _is_running_header_or_footer(cleaned_line, act_title, act_number):
                continue

            if not started_body:
                next_line = page_lines[line_index + 1] if line_index + 1 < len(page_lines) else None
                if _looks_like_body_start(cleaned_line, next_line):
                    normalized_lines.extend(_body_context_lines(pre_body_buffer))
                    normalized_lines.append(cleaned_line)
                    started_body = True
                    pre_body_buffer = []
                    continue

                pre_body_buffer.append(cleaned_line)
                pre_body_buffer = pre_body_buffer[-8:]
                continue

            normalized_lines.append(cleaned_line)
        normalized_lines.append("")

    return "\n".join(normalized_lines).strip()


def derive_act_title(document: LawDocumentText) -> str:
    """Derive a human-readable act title from extracted text."""

    metadata = _canonical_document_metadata(document)
    if metadata is not None:
        return metadata["act_title"]

    first_page_lines = document.pages[0].lines if document.pages else []
    structured_title = _extract_structured_title(first_page_lines)
    if structured_title:
        return structured_title

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
        if _is_bad_title_candidate(cleaned):
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

    return _normalize_filename_title(document.title)


def derive_act_number(document: LawDocumentText) -> str:
    """Extract the act number from the first few pages when present."""

    for page in document.pages[:5]:
        for line in page.lines:
            cleaned_line = WHITESPACE_PATTERN.sub(" ", line).strip()
            match = ACT_NUMBER_PATTERN.fullmatch(cleaned_line) or ORDER_NUMBER_PATTERN.fullmatch(cleaned_line)
            if match:
                return match.group(0)
    return ""


def derive_unit_type(document: LawDocumentText) -> str:
    """Infer the top-level legal unit type for chunk metadata."""

    lowered = document.title.lower()
    if "perlembagaan" in lowered:
        return "perkara"
    if "constitution" in lowered:
        return "article"
    return "section"


def derive_document_aliases(document: LawDocumentText) -> tuple[str, ...]:
    """Return canonical document aliases for retrieval-aware matching."""

    metadata = _canonical_document_metadata(document)
    if metadata is not None:
        return tuple(metadata["aliases"])

    aliases = {derive_act_title(document)}
    act_number = derive_act_number(document)
    if act_number:
        aliases.add(act_number)

    cleaned_title = _normalize_filename_title(document.title)
    if _is_alias_usable(cleaned_title):
        aliases.add(cleaned_title)

    cleaned_stem = _normalize_filename_title(Path(document.source_path).stem)
    if _is_alias_usable(cleaned_stem):
        aliases.add(cleaned_stem)

    return tuple(sorted(alias for alias in aliases if alias))


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
        "unit_type": chunk.unit_type,
        "unit_id": chunk.unit_id or chunk.section_id,
        "document_aliases": list(chunk.document_aliases),
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


def _looks_like_body_start(line: str, next_line: str | None = None) -> bool:
    if BODY_START_PATTERN.match(line):
        return True
    if NUMBERED_PROVISION_PATTERN.match(line):
        if next_line and not _looks_like_unit_body_followup(next_line.strip()):
            return False
        return True
    if NUMBERED_PROVISION_ONLY_PATTERN.match(line) and next_line and re.match(r"^\(\d+[A-Z]?\)", next_line.strip()):
        return True
    if EXPLICIT_UNIT_HEADING_PATTERN.match(line):
        if next_line and not _looks_like_unit_body_followup(next_line.strip()):
            return False
        return True
    if line.startswith("ENACTED by the Parliament of Malaysia as follows:"):
        return True
    return False


def _body_context_lines(lines: list[str]) -> list[str]:
    context: list[str] = []
    for line in lines[-4:]:
        if HIERARCHY_CONTEXT_PATTERN.match(line):
            context.append(line)
            continue
        if (
            TABLE_OF_CONTENTS_ENTRY_PATTERN.match(line)
            or line in {"Long Title", "LIST OF AMENDMENTS", "SECTION"}
            or line.startswith("Unannotated Statutes of Malaysia")
            or line.startswith("Page ")
            or line.startswith("[")
            or "]" in line
        ):
            continue
        if line.startswith("An Act "):
            continue
        if UPPER_HEADING_PATTERN.fullmatch(line) and len(line.split()) <= 12:
            context.append(line)
            continue
        if not line.endswith((".", ";", ":")) and len(line.split()) <= 16:
            context.append(line)
    deduped: list[str] = []
    for line in context:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return deduped


def _is_editorial_note_page(lines: list[str]) -> bool:
    top_lines = [line.strip() for line in lines[:8] if line.strip()]
    if not top_lines:
        return False
    note_markers = {"NOTE:", "NOTES", "CATATAN"}
    return any(line in note_markers for line in top_lines)


def _is_contents_or_index_page(lines: list[str]) -> bool:
    cleaned_lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in lines if line.strip()]
    if not cleaned_lines:
        return False
    top_block = " ".join(cleaned_lines[:12])
    if CONTENTS_MARKER_PATTERN.search(top_block):
        return True

    toc_like_count = sum(1 for line in cleaned_lines[:40] if TABLE_OF_CONTENTS_ENTRY_PATTERN.match(line))
    short_line_count = sum(1 for line in cleaned_lines[:40] if len(line.split()) <= 8)
    if "SECTION" in cleaned_lines[:20] and toc_like_count >= 4 and short_line_count >= 6:
        return True
    if toc_like_count >= 12 and short_line_count >= 16:
        return True
    return False


def _canonical_document_metadata(document: LawDocumentText) -> dict[str, object] | None:
    first_page_lines = document.pages[0].lines if document.pages else []
    normalized_candidates = {
        _normalize_metadata_text(document.title),
        _normalize_metadata_text(Path(document.source_path).stem),
        _normalize_metadata_text(" ".join(first_page_lines[:20])),
    }
    derived_number = derive_act_number(document)
    if derived_number:
        normalized_candidates.add(_normalize_metadata_text(derived_number))
    for entry in CANONICAL_DOCUMENT_METADATA:
        if any(term in candidate for candidate in normalized_candidates for term in entry["match_terms"]):
            return entry
    return None


def _normalize_metadata_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _prepare_page_lines(lines: list[str]) -> list[str]:
    prepared: list[str] = []
    index = 0
    while index < len(lines):
        current = WHITESPACE_PATTERN.sub(" ", lines[index]).strip()
        next_line = WHITESPACE_PATTERN.sub(" ", lines[index + 1]).strip() if index + 1 < len(lines) else ""
        if NUMBERED_PROVISION_ONLY_PATTERN.match(current) and next_line.startswith("("):
            prepared.append(f"{current} {next_line}")
            index += 2
            continue
        prepared.append(current)
        index += 1
    return prepared


def _normalize_extracted_line(line: str) -> str:
    cleaned = WHITESPACE_PATTERN.sub(" ", line).strip()
    cleaned = HEADER_NOISE_PREFIX_PATTERN.sub("", cleaned).lstrip(" ,./\\-")
    if cleaned.startswith("Unannotated Statutes of Malaysia") or cleaned.startswith("Page "):
        return ""
    return cleaned


def _extract_structured_title(lines: list[str]) -> str:
    title_lines: list[str] = []
    capture_after_reference = False

    for raw_line in lines[:25]:
        cleaned = WHITESPACE_PATTERN.sub(" ", raw_line).strip()
        if not cleaned:
            continue
        if _is_bad_title_candidate(cleaned):
            if title_lines:
                break
            continue
        if ACT_NUMBER_PATTERN.fullmatch(cleaned) or re.fullmatch(r"P\.U\.\s*\([A-Z]\)\s*\d+", cleaned, re.IGNORECASE):
            capture_after_reference = True
            continue
        if not capture_after_reference and TITLE_HEADER_SKIP_PATTERN.match(cleaned):
            continue
        if not capture_after_reference and not UPPER_TITLE_LINE_PATTERN.fullmatch(cleaned):
            continue
        if UPPER_TITLE_LINE_PATTERN.fullmatch(cleaned):
            title_lines.append(cleaned)
            if len(title_lines) >= 3:
                break
            continue
        if title_lines:
            break

    normalized = _normalize_structured_title(title_lines)
    if normalized:
        return normalized
    return ""


def _normalize_structured_title(lines: list[str]) -> str:
    if not lines:
        return ""

    cleaned_lines = [line.strip(" /") for line in lines if line.strip(" /")]
    if not cleaned_lines:
        return ""

    if len(cleaned_lines) >= 2 and "ORDER" in cleaned_lines[-1]:
        candidate = cleaned_lines[-1]
    else:
        candidate = " ".join(cleaned_lines)

    candidate = re.sub(r"\s+", " ", candidate).strip(" /")
    candidate = re.sub(r"\b(Section|Article|Perkara)$", "", candidate, flags=re.IGNORECASE).strip()
    if not candidate:
        return ""
    if candidate.isupper():
        return candidate.title()
    return candidate


def _normalize_filename_title(title: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", title)
    cleaned = re.sub(r"^\d{8}\s+", "", cleaned)
    cleaned = re.sub(r"\b\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"\(\d+\)", "", cleaned)
    cleaned = re.sub(r"_\d+$", "", cleaned)
    cleaned = re.sub(r"\b(?:english|bi|reprint version [\d.]+)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    return cleaned.title()


def _is_bad_title_candidate(line: str) -> bool:
    cleaned = line.strip()
    if not cleaned:
        return True
    if cleaned.isdigit() or DATE_ONLY_PATTERN.fullmatch(cleaned):
        return True
    if TITLE_NOTICE_PATTERN.match(cleaned):
        return True
    return False


def _is_alias_usable(alias: str) -> bool:
    cleaned = alias.strip()
    if not cleaned:
        return False
    if any(character in cleaned for character in {"_", "&"}):
        return False
    if re.match(r"^\d{8}\b", cleaned):
        return False
    if len(cleaned) > 60 and "Act" not in cleaned and "Order" not in cleaned:
        return False
    if cleaned.isupper() and len(cleaned.split()) > 3:
        return False
    return True


def _looks_like_unit_body_followup(line: str) -> bool:
    if not line:
        return False
    if re.match(r"^\(\d+[A-Z]?\)", line):
        return True
    if NUMBERED_PROVISION_PATTERN.match(line) or EXPLICIT_UNIT_HEADING_PATTERN.match(line):
        return False
    if line in {"SECTION", "Long Title", "LIST OF AMENDMENTS"}:
        return False
    if UPPER_HEADING_PATTERN.fullmatch(line) and len(line.split()) <= 6:
        return False
    return True
