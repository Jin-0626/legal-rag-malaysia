"""Rule-based Gold Set V2 candidate generation for Malaysian legal retrieval."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

HIERARCHY_LINE_PATTERN = re.compile(
    r"^(?P<label>Part|PART|Chapter|CHAPTER|Division|Bahagian|BAHAGIAN|Bab|BAB|Jadual|JADUAL|Schedule|SCHEDULE)\b(?P<body>.*)$"
)
UNIT_HEADING_PREFIX_PATTERN = re.compile(r"^(Section|Article|Perkara)\s+\d+[A-Z]?\s*", re.IGNORECASE)
GENERIC_HEADING_BODIES = {
    "",
    "section",
    "article",
    "perkara",
    "chapter",
    "division",
    "part",
    "bahagian",
    "bab",
    "jadual",
    "schedule",
}
NOISY_HEADING_PATTERN = re.compile(r"(laws of malaysia|reprint|cetakan semula|online version|updated text)", re.IGNORECASE)
DEFINITION_TERMS = ("interpretation", "definition", "definitions", "tafsiran", "maksud")
OBLIGATION_TERMS = ("duties", "duty", "obligation", "obligations", "requirement", "requirements")
CAPABILITY_TERMS = ("rights", "right", "powers", "power", "appeal", "appeals", "application", "offence", "offences")
CATEGORY_PRIORITY = (
    "direct_lookup",
    "heading_lookup",
    "definition",
    "capability",
    "obligation",
    "rights",
    "hierarchy",
    "bilingual",
    "amendment",
    "gazette_order",
)


@dataclass(frozen=True)
class UnitRecord:
    chunk_id: str
    document_id: str
    act_title: str
    act_number: str
    document_aliases: tuple[str, ...]
    source_file: str
    unit_type: str
    unit_id: str
    section_heading: str
    subsection_id: str | None
    paragraph_id: str | None
    language: str
    document_kind: str
    heading_body: str
    hierarchy_line: str | None
    text: str


@dataclass(frozen=True)
class GeneratedGoldCandidate:
    query: str
    expected_doc: str | None
    expected_unit: str | None
    unit_type: str | None
    category: str
    language: str
    source_chunk_id: str | None
    generation_method: str
    confidence: float
    needs_review: bool
    expected_act_title: str | None = None
    expected_section_id: str | None = None


def load_unit_records(processed_dir: Path) -> list[UnitRecord]:
    """Load one representative chunk per legal unit from processed JSONL exports."""

    records_by_unit: dict[tuple[str, str, str], dict] = {}
    for path in sorted(Path(processed_dir).glob("*.jsonl")):
        if path.name.endswith(".smoke.jsonl"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                key = (
                    payload["document_id"],
                    payload.get("unit_type", "section"),
                    payload.get("unit_id", payload["section_id"]),
                )
                if key not in records_by_unit or int(payload.get("chunk_index", 0)) < int(
                    records_by_unit[key].get("chunk_index", 0)
                ):
                    records_by_unit[key] = payload

    units: list[UnitRecord] = []
    for payload in records_by_unit.values():
        heading_body = _heading_body(payload.get("section_heading", ""))
        hierarchy_line = _extract_hierarchy_line(payload.get("text", ""))
        act_title = payload.get("act_title", "").strip()
        units.append(
            UnitRecord(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                act_title=act_title,
                act_number=payload.get("act_number", ""),
                document_aliases=tuple(payload.get("document_aliases", [])),
                source_file=payload.get("source_file", ""),
                unit_type=payload.get("unit_type", "section"),
                unit_id=payload.get("unit_id", payload["section_id"]),
                section_heading=payload.get("section_heading", "").strip(),
                subsection_id=payload.get("subsection_id"),
                paragraph_id=payload.get("paragraph_id"),
                language=_detect_language(payload),
                document_kind=_detect_document_kind(payload),
                heading_body=heading_body,
                hierarchy_line=hierarchy_line,
                text=payload.get("text", ""),
            )
        )
    return sorted(units, key=lambda item: (item.act_title, item.unit_type, _unit_sort_key(item.unit_id)))


def build_gold_set_v2_candidates(processed_dir: Path) -> list[GeneratedGoldCandidate]:
    """Generate conservative, reviewable candidate gold-set entries from processed chunks."""

    units = load_unit_records(processed_dir)
    candidates: list[GeneratedGoldCandidate] = []
    for unit in units:
        candidates.extend(_generate_positive_candidates(unit))
    candidates.extend(_generate_negative_candidates(units))
    deduped = _dedupe_candidates(candidates)
    return _balance_candidates(deduped)


def write_gold_set_candidates(candidates: list[GeneratedGoldCandidate], output_path: Path) -> None:
    """Write generated candidates to JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")


def format_gold_set_summary(candidates: list[GeneratedGoldCandidate]) -> str:
    """Build a human-review summary of the generated candidate set."""

    by_doc = Counter(candidate.expected_doc or "negative/none" for candidate in candidates)
    by_category = Counter(candidate.category for candidate in candidates)
    by_language = Counter(candidate.language for candidate in candidates)
    by_review = Counter("needs_review" if candidate.needs_review else "high_confidence" for candidate in candidates)

    lines = [
        "# Gold Set V2 Candidate Summary",
        "",
        f"- Total generated candidates: {len(candidates)}",
        f"- High-confidence candidates: {by_review['high_confidence']}",
        f"- Needs-review candidates: {by_review['needs_review']}",
        "",
        "## Counts By Document",
    ]
    for document, count in sorted(by_doc.items()):
        lines.append(f"- {document}: {count}")

    lines.extend(["", "## Counts By Category"])
    for category, count in sorted(by_category.items()):
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Counts By Language"])
    for language, count in sorted(by_language.items()):
        lines.append(f"- {language}: {count}")

    lines.extend(["", "## Sample Queries"])
    for candidate in candidates[:20]:
        lines.append(
            f"- [{candidate.category}/{candidate.language}] {candidate.query} "
            f"(doc={candidate.expected_doc or 'none'}, unit={candidate.expected_unit or 'none'}, "
            f"confidence={candidate.confidence:.2f}, review={candidate.needs_review})"
        )
    return "\n".join(lines) + "\n"


def _generate_positive_candidates(unit: UnitRecord) -> list[GeneratedGoldCandidate]:
    if not _is_high_value_unit(unit):
        return []

    candidates: list[GeneratedGoldCandidate] = []
    primary_doc_name = _primary_doc_name(unit)
    primary_alias = _primary_alias(unit)
    label = _unit_label(unit.unit_type, unit.language)

    candidates.extend(
        [
            _candidate(
                query=_direct_lookup_query(unit, primary_doc_name),
                unit=unit,
                category="direct_lookup",
                language=unit.language,
                generation_method="direct_lookup_template",
                confidence=0.92,
                needs_review=False,
            ),
            _candidate(
                query=f"{label} {unit.unit_id} {primary_alias}",
                unit=unit,
                category="direct_lookup",
                language=unit.language,
                generation_method="compact_lookup_template",
                confidence=0.88,
                needs_review=False,
            ),
        ]
    )

    if unit.heading_body:
        candidates.extend(_heading_lookup_candidates(unit, primary_doc_name, primary_alias))
        candidates.extend(_category_candidates_from_heading(unit, primary_doc_name, primary_alias))

    if unit.hierarchy_line:
        candidates.extend(_hierarchy_candidates(unit, primary_doc_name))

    if unit.document_kind == "amendment":
        candidates.extend(_amendment_candidates(unit))

    if unit.document_kind in {"gazette", "order"}:
        candidates.extend(_gazette_candidates(unit))

    if unit.language == "ms":
        candidates.extend(_bilingual_candidates(unit, primary_doc_name))

    return candidates


def _heading_lookup_candidates(unit: UnitRecord, primary_doc_name: str, primary_alias: str) -> list[GeneratedGoldCandidate]:
    if not _is_heading_body_usable(unit.heading_body):
        return []
    if unit.language == "ms":
        query = f"{_unit_label(unit.unit_type, 'ms')} manakah berkaitan {unit.heading_body.lower()}?"
    else:
        query = f"Which {_unit_label(unit.unit_type, 'en').lower()} covers {unit.heading_body} in the {primary_doc_name}?"
    return [
        _candidate(
            query=query,
            unit=unit,
            category="heading_lookup",
            language=unit.language,
            generation_method="heading_lookup_template",
            confidence=0.84,
            needs_review=False,
        ),
        _candidate(
            query=f"What does {primary_alias} say about {unit.heading_body}?" if unit.language == "en" else f"Apakah yang dinyatakan oleh {primary_alias} tentang {unit.heading_body.lower()}?",
            unit=unit,
            category="heading_lookup",
            language=unit.language,
            generation_method="heading_topic_template",
            confidence=0.78,
            needs_review=False,
        ),
    ]


def _category_candidates_from_heading(unit: UnitRecord, primary_doc_name: str, primary_alias: str) -> list[GeneratedGoldCandidate]:
    heading_lower = unit.heading_body.lower()
    candidates: list[GeneratedGoldCandidate] = []
    if any(term in heading_lower for term in DEFINITION_TERMS):
        candidates.extend(
            [
                _candidate(
                    query=(
                        f"What is defined under {unit.heading_body} in {_unit_label(unit.unit_type, 'en')} {unit.unit_id} of the {primary_doc_name}?"
                        if unit.language == "en"
                        else f"Apakah yang ditakrifkan di bawah {unit.heading_body.lower()} dalam {_unit_label(unit.unit_type, 'ms')} {unit.unit_id} {primary_doc_name}?"
                    ),
                    unit=unit,
                    category="definition",
                    language=unit.language,
                    generation_method="definition_heading_template",
                    confidence=0.85,
                    needs_review=False,
                ),
                _candidate(
                    query=(
                        f"What does {primary_alias} define in {_unit_label(unit.unit_type, 'en')} {unit.unit_id}?"
                        if unit.language == "en"
                        else f"Apakah maksud {_unit_label(unit.unit_type, 'ms')} {unit.unit_id} dalam {primary_alias}?"
                    ),
                    unit=unit,
                    category="definition",
                    language=unit.language,
                    generation_method="definition_unit_template",
                    confidence=0.82,
                    needs_review=False,
                ),
            ]
        )
    if any(term in heading_lower for term in OBLIGATION_TERMS):
        candidates.append(
            _candidate(
                query=f"What duties are imposed under {primary_alias}?" if unit.language == "en" else f"Apakah kewajipan di bawah {primary_alias}?",
                unit=unit,
                category="obligation",
                language=unit.language,
                generation_method="obligation_heading_template",
                confidence=0.72,
                needs_review=False,
            )
        )
    if any(term in heading_lower for term in CAPABILITY_TERMS):
        candidates.extend(_capability_queries(unit, primary_alias))
    return candidates


def _hierarchy_candidates(unit: UnitRecord, primary_doc_name: str) -> list[GeneratedGoldCandidate]:
    label, _, topic = _parse_hierarchy_line(unit.hierarchy_line)
    if not label or not topic or not _is_heading_body_usable(unit.heading_body):
        return []
    query = (
        f"Which {label} covers {unit.heading_body} in the {primary_doc_name}?"
        if unit.language == "en"
        else f"{label} manakah berkaitan {unit.heading_body.lower()}?"
    )
    return [
        _candidate(
            query=query,
            unit=unit,
            category="hierarchy",
            language=unit.language,
            generation_method="hierarchy_context_template",
            confidence=0.46,
            needs_review=True,
        )
    ]


def _amendment_candidates(unit: UnitRecord) -> list[GeneratedGoldCandidate]:
    return [
        _candidate(
            query=f"What changes were introduced in {unit.act_title}?",
            unit=unit,
            category="amendment",
            language=unit.language,
            generation_method="amendment_document_template",
            confidence=0.38,
            needs_review=True,
        ),
        _candidate(
            query=f"What is amended by {unit.act_title}?",
            unit=unit,
            category="amendment",
            language=unit.language,
            generation_method="amendment_scope_template",
            confidence=0.34,
            needs_review=True,
        ),
    ]


def _gazette_candidates(unit: UnitRecord) -> list[GeneratedGoldCandidate]:
    return [
        _candidate(
            query=f"What does {unit.act_title} order?",
            unit=unit,
            category="gazette_order",
            language=unit.language,
            generation_method="gazette_order_template",
            confidence=0.32,
            needs_review=True,
        )
    ]


def _bilingual_candidates(unit: UnitRecord, primary_doc_name: str) -> list[GeneratedGoldCandidate]:
    if unit.unit_type != "perkara":
        return []
    return [
        _candidate(
            query=f"Apakah maksud Perkara {unit.unit_id} dalam {primary_doc_name}?",
            unit=unit,
            category="bilingual",
            language="ms",
            generation_method="bm_direct_lookup_template",
            confidence=0.86,
            needs_review=False,
        )
    ]


def _capability_queries(unit: UnitRecord, primary_alias: str) -> list[GeneratedGoldCandidate]:
    heading_lower = unit.heading_body.lower()
    query = None
    if "appeal" in heading_lower:
        query = f"Who can appeal under {_unit_label(unit.unit_type, 'en')} {unit.unit_id} of {primary_alias}?"
        category = "capability"
    elif "application" in heading_lower:
        query = f"To whom does {_unit_label(unit.unit_type, 'en')} {unit.unit_id} of {primary_alias} apply?"
        category = "capability"
    elif "power" in heading_lower or "powers" in heading_lower:
        query = f"What powers are provided in {_unit_label(unit.unit_type, 'en')} {unit.unit_id} of {primary_alias}?"
        category = "capability"
    elif "right" in heading_lower or "rights" in heading_lower:
        query = f"What rights are recognized in {_unit_label(unit.unit_type, 'en')} {unit.unit_id} of {primary_alias}?"
        category = "rights"
    elif "offence" in heading_lower or "offences" in heading_lower:
        query = f"What offences are created in {_unit_label(unit.unit_type, 'en')} {unit.unit_id} of {primary_alias}?"
        category = "capability"
    else:
        return []
    return [
        _candidate(
            query=query,
            unit=unit,
            category=category,
            language=unit.language,
            generation_method="capability_heading_template",
            confidence=0.74,
            needs_review=False,
        )
    ]


def _generate_negative_candidates(units: list[UnitRecord]) -> list[GeneratedGoldCandidate]:
    high_value_units = [unit for unit in units if _is_high_value_unit(unit)]
    by_doc: dict[str, list[UnitRecord]] = defaultdict(list)
    for unit in high_value_units:
        by_doc[unit.act_title].append(unit)

    negatives: list[GeneratedGoldCandidate] = []
    documents = sorted(by_doc)
    if not documents:
        return negatives

    for index, document in enumerate(documents):
        units_for_doc = by_doc[document]
        if not units_for_doc:
            continue
        unit = units_for_doc[min(1, len(units_for_doc) - 1)]
        wrong_doc = documents[(index + 1) % len(documents)]
        label = _unit_label(unit.unit_type, unit.language)
        negatives.append(
            GeneratedGoldCandidate(
                query=f"What does {label} 999 of the {document} say?" if unit.language == "en" else f"Apakah kandungan {label} 999 dalam {document}?",
                expected_doc=None,
                expected_unit=None,
                unit_type=unit.unit_type,
                category="negative",
                language=unit.language,
                source_chunk_id=None,
                generation_method="synthetic_invalid_unit_template",
                confidence=0.2,
                needs_review=True,
                expected_act_title=None,
                expected_section_id=None,
            )
        )
        if _is_heading_body_usable(unit.heading_body):
            negatives.append(
                GeneratedGoldCandidate(
                    query=(
                        f"Which {label.lower()} covers {unit.heading_body} in the {wrong_doc}?"
                        if unit.language == "en"
                        else f"{label} manakah berkaitan {unit.heading_body.lower()} dalam {wrong_doc}?"
                    ),
                    expected_doc=None,
                    expected_unit=None,
                    unit_type=unit.unit_type,
                    category="negative",
                    language=unit.language,
                    source_chunk_id=None,
                    generation_method="synthetic_wrong_document_template",
                    confidence=0.24,
                    needs_review=True,
                    expected_act_title=None,
                    expected_section_id=None,
                )
            )
    return negatives


def _dedupe_candidates(candidates: list[GeneratedGoldCandidate]) -> list[GeneratedGoldCandidate]:
    deduped: dict[str, GeneratedGoldCandidate] = {}
    for candidate in candidates:
        key = _normalize_query(candidate.query)
        current = deduped.get(key)
        if current is None or candidate.confidence > current.confidence:
            deduped[key] = candidate
    return list(deduped.values())


def _balance_candidates(candidates: list[GeneratedGoldCandidate]) -> list[GeneratedGoldCandidate]:
    positives = [candidate for candidate in candidates if candidate.category != "negative"]
    negatives = [candidate for candidate in candidates if candidate.category == "negative"]

    balanced: list[GeneratedGoldCandidate] = []
    by_doc_category: dict[str, dict[str, list[GeneratedGoldCandidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in positives:
        doc_key = candidate.expected_doc or "none"
        by_doc_category[doc_key][candidate.category].append(candidate)

    for doc_key, category_map in by_doc_category.items():
        for category in category_map:
            category_map[category].sort(key=lambda item: (-item.confidence, item.query))

        doc_count = 0
        while doc_count < 24:
            added_in_round = False
            for category in CATEGORY_PRIORITY:
                bucket = category_map.get(category)
                if not bucket:
                    continue
                category_items = [item for item in balanced if (item.expected_doc or "none") == doc_key and item.category == category]
                if len(category_items) >= 6:
                    continue
                balanced.append(bucket.pop(0))
                doc_count += 1
                added_in_round = True
                if doc_count >= 24:
                    break
            if not added_in_round:
                break

    negatives = sorted(negatives, key=lambda item: (item.language, item.query))[: max(8, len(balanced) // 6)]
    final_candidates = balanced + negatives
    final_candidates.sort(
        key=lambda item: (
            item.expected_doc or "zzz-negative",
            item.category,
            item.language,
            item.query,
        )
    )
    return final_candidates


def _candidate(
    *,
    query: str,
    unit: UnitRecord,
    category: str,
    language: str,
    generation_method: str,
    confidence: float,
    needs_review: bool,
) -> GeneratedGoldCandidate:
    return GeneratedGoldCandidate(
        query=query.strip(),
        expected_doc=unit.act_title,
        expected_unit=unit.unit_id,
        unit_type=unit.unit_type,
        category=category,
        language=language,
        source_chunk_id=unit.chunk_id,
        generation_method=generation_method,
        confidence=confidence,
        needs_review=needs_review,
        expected_act_title=unit.act_title,
        expected_section_id=unit.unit_id,
    )


def _direct_lookup_query(unit: UnitRecord, primary_doc_name: str) -> str:
    label = _unit_label(unit.unit_type, unit.language)
    if unit.language == "ms":
        return f"Apakah kandungan {label} {unit.unit_id} dalam {primary_doc_name}?"
    return f"What does {label} {unit.unit_id} of the {primary_doc_name} say?"


def _unit_label(unit_type: str, language: str) -> str:
    if unit_type == "article":
        return "Article"
    if unit_type == "perkara":
        return "Perkara"
    if language == "ms":
        return "Seksyen"
    return "Section"


def _primary_doc_name(unit: UnitRecord) -> str:
    return unit.act_title


def _primary_alias(unit: UnitRecord) -> str:
    aliases = [alias for alias in unit.document_aliases if alias and alias != unit.act_title]
    if unit.language == "ms":
        for alias in aliases:
            if _looks_malay(alias):
                return alias
    for alias in aliases:
        if len(alias) <= 10 or any(keyword in alias for keyword in ("Act", "PDPA", "Akta")):
            return alias
    return unit.act_title


def _detect_language(payload: dict) -> str:
    title = f"{payload.get('act_title', '')} {payload.get('section_heading', '')}".lower()
    if any(term in title for term in ("perlembagaan", "perkara", "bahagian", "hak", "kebebasan", "akta")):
        return "ms"
    return "en"


def _detect_document_kind(payload: dict) -> str:
    title = f"{payload.get('act_title', '')} {payload.get('source_file', '')}".lower()
    if any(term in title for term in ("amendment", "pindaan")) or re.search(r"\bact[- ]?a\d+\b", title):
        return "amendment"
    if any(term in title for term in ("p.u.", "gazette", "order", "perintah")):
        return "gazette"
    if any(term in title for term in ("constitution", "perlembagaan")):
        return "constitution"
    return "act"


def _heading_body(section_heading: str) -> str:
    body = UNIT_HEADING_PREFIX_PATTERN.sub("", section_heading.strip())
    body = re.sub(r"\s+", " ", body).strip(" -—:;")
    return body


def _extract_hierarchy_line(text: str) -> str | None:
    for line in text.splitlines()[:4]:
        cleaned = line.strip()
        if HIERARCHY_LINE_PATTERN.match(cleaned):
            return cleaned
    return None


def _parse_hierarchy_line(line: str | None) -> tuple[str | None, str | None, str | None]:
    if not line:
        return None, None, None
    match = HIERARCHY_LINE_PATTERN.match(line.strip())
    if not match:
        return None, None, None
    label = match.group("label").title()
    body = re.sub(r"^[\s—:.-]+", "", match.group("body")).strip()
    return label, body, body


def _is_high_value_unit(unit: UnitRecord) -> bool:
    if not unit.act_title or NOISY_HEADING_PATTERN.search(unit.act_title):
        return False
    if not unit.section_heading or NOISY_HEADING_PATTERN.search(unit.section_heading):
        return False
    if unit.document_kind == "gazette" and len(unit.text.split()) < 20:
        return False
    return True


def _is_heading_body_usable(heading_body: str) -> bool:
    if not heading_body:
        return False
    lowered = heading_body.lower().strip()
    if lowered in GENERIC_HEADING_BODIES:
        return False
    if len(lowered) < 4:
        return False
    if NOISY_HEADING_PATTERN.search(lowered):
        return False
    digit_ratio = sum(character.isdigit() for character in lowered) / max(len(lowered), 1)
    if digit_ratio > 0.25:
        return False
    weird_ratio = sum(not (character.isalnum() or character.isspace() or character in "-&,/") for character in lowered) / max(len(lowered), 1)
    return weird_ratio < 0.12


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def _looks_malay(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("akta", "perlembagaan", "perkara", "bahagian", "hak", "kebebasan"))


def _unit_sort_key(unit_id: str) -> tuple[int, str]:
    digits = "".join(character for character in unit_id if character.isdigit())
    return (int(digits) if digits else 10_000, unit_id)
