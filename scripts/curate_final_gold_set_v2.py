"""Curate the reviewed Final Gold Set V2 from generated and manual entries."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from legal_rag.evaluation.goldset_generator import GeneratedGoldCandidate, UnitRecord, load_unit_records


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "data" / "evaluation" / "gold_set_v2_candidates.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "final_gold_set_v2.jsonl"
SUMMARY_PATH = ROOT / "data" / "evaluation" / "final_gold_set_v2_summary.md"

SELECTED_CANDIDATE_SPECS = [
    {"source_query": "What does Section 1 of the Employment Act 1955 say?"},
    {"source_query": "What does Section 10 of the Employment Act 1955 say?"},
    {
        "source_query": "Apakah kandungan Perkara 10 dalam Federal Constitution?",
        "query": "Apakah kandungan Perkara 10 dalam Perlembagaan Persekutuan?",
    },
    {"source_query": "What does Section 1 of the Personal Data Protection Act 2010 say?"},
    {"source_query": "What does Section 10 of the Consumer Protection Act 1999 say?"},
    {"source_query": "What does Section 10 of the Contracts Act 1950 say?"},
    {"source_query": "What does Section 103 of the Income Tax Act 1967 say?"},
    {"source_query": "What does Section 2 of the Minimum Wages Order 2024 say?"},
    {"source_query": "What does Section 10 of the Personal Data Protection (Amendment) Act 2024 say?"},
    {"source_query": "What does Section 12 of the Children and Young Persons (Employment) Act 1966 say?"},
    {
        "source_query": "Which section covers Annual leave in the Employment Act 1955?",
        "query": "Which section of the Employment Act 1955 deals with annual leave?",
    },
    {
        "source_query": "Which section covers Appeals in the Employment Act 1955?",
        "query": "Which section of the Employment Act 1955 deals with appeals?",
    },
    {
        "source_query": "Perkara manakah berkaitan agama bagi persekutuan?",
        "query": "Perkara manakah berkaitan agama bagi Persekutuan?",
    },
    {
        "source_query": "Which section covers Access Principle in the Personal Data Protection Act 2010?",
        "query": "Which section of the PDPA sets out the Access Principle?",
    },
    {
        "source_query": "Which section covers Accounts and audit in the Personal Data Protection Act 2010?",
        "query": "Which section of the PDPA deals with accounts and audit?",
    },
    {
        "source_query": "Which section covers Application in the Consumer Protection Act 1999?",
        "query": "Which section of the Consumer Protection Act 1999 deals with application?",
    },
    {
        "source_query": "Which section covers Acceptance must be absolute in the Contracts Act 1950?",
        "query": "Which section of the Contracts Act 1950 says acceptance must be absolute?",
    },
    {
        "source_query": "Which section covers Nama dan permulaan kuat kuasa in the Minimum Wages Order 2024?",
        "query": "Which section of the Minimum Wages Order 2024 deals with citation and commencement?",
    },
    {
        "source_query": "Which section covers Pembatalan in the Minimum Wages Order 2024?",
        "query": "Which section of the Minimum Wages Order 2024 deals with revocation?",
    },
    {
        "source_query": "Which section covers Amendment of section 4 in the Personal Data Protection (Amendment) Act 2024?",
        "query": "Which section of the PDPA Amendment Act 2024 amends section 4 of the principal Act?",
    },
    {
        "source_query": "What is defined under Interpretation in Section 2 of the Employment Act 1955?",
    },
    {
        "source_query": "Apakah yang ditakrifkan di bawah tafsiran bahagian x dalam Perkara 148 Federal Constitution?",
        "query": "Apakah yang ditakrifkan di bawah tafsiran Bahagian X dalam Perlembagaan Persekutuan?",
    },
    {
        "source_query": "What is defined under Interpretation in Section 4 of the Personal Data Protection Act 2010?",
    },
    {
        "source_query": "What is defined under Interpretation in Section 3 of the Consumer Protection Act 1999?",
    },
    {
        "source_query": "What is defined under Interpretation in Section 2 of the Contracts Act 1950?",
    },
    {
        "source_query": "What is defined under Interpretation in Section 2 of the Income Tax Act 1967?",
    },
    {
        "source_query": "What is defined under Interpretation in Section 2 of the Employees Provident Fund Act 1991?",
    },
    {
        "source_query": "What does Act 67 define in Section 2?",
        "query": "What is defined in section 2 of the Civil Law Act 1956?",
    },
    {
        "source_query": "To whom does Section 1 of Akta Kerja 1955 apply?",
        "query": "Who does section 1 of the Employment Act 1955 apply to?",
    },
    {
        "source_query": "What powers are provided in Article 121 of Federal Constitution?",
        "query": "What powers does Article 121 of the Federal Constitution confer?",
    },
    {
        "source_query": "To whom does Section 2 of PDPA apply?",
        "query": "Who is covered by section 2 of the PDPA?",
    },
    {
        "source_query": "What offences are created in Section 137 of Act 599?",
        "query": "What offences are created by section 137 of the Consumer Protection Act 1999?",
    },
    {
        "source_query": "What powers are provided in Section 135 of Act 53?",
        "query": "What powers does section 135 of the Income Tax Act 1967 confer?",
    },
    {
        "source_query": "What rights are recognized in Section 81 of Akta Kerja 1955?",
        "query": "What right does section 81 of the Employment Act 1955 give an employee?",
    },
    {
        "source_query": "What rights are recognized in Article 13 of Federal Constitution?",
        "query": "What right is protected by Article 13 of the Federal Constitution?",
    },
    {
        "source_query": "What rights are recognized in Section 43 of PDPA?",
        "query": "What right does section 43 of the PDPA provide?",
    },
    {
        "source_query": "What rights are recognized in Section 39 of Act 599?",
        "query": "What consumer right is addressed in section 39 of the Consumer Protection Act 1999?",
    },
    {
        "source_query": "What duties are imposed under Akta Kerja 1955?",
        "query": "What duties does the Employment Act 1955 impose?",
    },
    {
        "source_query": "What duties are imposed under Act 514?",
        "query": "What duties does the Occupational Safety and Health Act 1994 impose?",
    },
    {
        "source_query": "What duties are imposed under PDPA?",
        "query": "What duties does the PDPA impose?",
    },
    {
        "source_query": "Apakah maksud Perkara 1 dalam Federal Constitution?",
        "query": "Apakah maksud Perkara 1 dalam Perlembagaan Persekutuan?",
    },
    {
        "source_query": "Apakah maksud Perkara 10 dalam Federal Constitution?",
        "query": "Apakah maksud Perkara 10 dalam Perlembagaan Persekutuan?",
    },
    {"source_query": "What does Article 999 of the Federal Constitution say?"},
    {"source_query": "What does Section 999 of the Employment Act 1955 say?"},
    {"source_query": "What does Section 999 of the Personal Data Protection Act 2010 say?"},
    {"source_query": "What does Section 999 of the Minimum Wages Order 2024 say?"},
    {"source_query": "What does Section 999 of the Consumer Protection Act 1999 say?"},
]

MANUAL_ENTRIES = [
    {
        "query": "Which section begins Part II of the Employment Act 1955?",
        "expected_doc": "Employment Act 1955",
        "expected_unit": "5",
        "unit_type": "section",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Which section begins Part III of the Employment Act 1955?",
        "expected_doc": "Employment Act 1955",
        "expected_unit": "15",
        "unit_type": "section",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Which Article begins Part II on Fundamental Liberties in the Federal Constitution?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "5",
        "unit_type": "article",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Which Article begins Part III on Citizenship in the Federal Constitution?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "14",
        "unit_type": "article",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Which section begins Part II of the Personal Data Protection Act 2010?",
        "expected_doc": "Personal Data Protection Act 2010",
        "expected_unit": "4",
        "unit_type": "section",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Which section begins Division 2 under Part II of the Personal Data Protection Act 2010?",
        "expected_doc": "Personal Data Protection Act 2010",
        "expected_unit": "12",
        "unit_type": "section",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Which section opens Part XIA of the Consumer Protection Act 1999?",
        "expected_doc": "Consumer Protection Act 1999",
        "expected_unit": "84",
        "unit_type": "section",
        "category": "hierarchy",
        "language": "en",
    },
    {
        "query": "Perkara manakah memulakan Bahagian II tentang Kebebasan Asasi dalam Perlembagaan Persekutuan?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "5",
        "unit_type": "perkara",
        "category": "hierarchy",
        "language": "ms",
    },
    {
        "query": "Apakah kandungan Perkara 8 dalam Perlembagaan Persekutuan?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "8",
        "unit_type": "perkara",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Apakah maksud Perkara 160 dalam Perlembagaan Persekutuan?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "160",
        "unit_type": "perkara",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Apakah hak yang dilindungi oleh Perkara 13 Perlembagaan Persekutuan?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "13",
        "unit_type": "perkara",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Apakah kebebasan yang disentuh oleh Perkara 10 dalam Perlembagaan Persekutuan?",
        "expected_doc": "Federal Constitution",
        "expected_unit": "10",
        "unit_type": "perkara",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Apakah kandungan Seksyen 2 Akta Kerja 1955?",
        "expected_doc": "Employment Act 1955",
        "expected_unit": "2",
        "unit_type": "section",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955?",
        "expected_doc": "Employment Act 1955",
        "expected_unit": "60E",
        "unit_type": "section",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Kepada siapakah Akta Perlindungan Data Peribadi 2010 terpakai dalam urus niaga komersial?",
        "expected_doc": "Personal Data Protection Act 2010",
        "expected_unit": "2",
        "unit_type": "section",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "Seksyen manakah memperkenalkan hak pemindahan data dalam Akta Pindaan PDPA 2024?",
        "expected_doc": "Personal Data Protection (Amendment) Act 2024",
        "expected_unit": "9",
        "unit_type": "section",
        "category": "bilingual",
        "language": "ms",
    },
    {
        "query": "When does the Personal Data Protection (Amendment) Act 2024 come into force?",
        "expected_doc": "Personal Data Protection (Amendment) Act 2024",
        "expected_unit": "1",
        "unit_type": "section",
        "category": "amendment",
        "language": "en",
    },
    {
        "query": "Which section of Act A1727 makes the general amendment to the principal Act?",
        "expected_doc": "Personal Data Protection (Amendment) Act 2024",
        "expected_unit": "2",
        "unit_type": "section",
        "category": "amendment",
        "language": "en",
    },
    {
        "query": "Which section of Act A1727 amends section 4 of the PDPA?",
        "expected_doc": "Personal Data Protection (Amendment) Act 2024",
        "expected_unit": "3",
        "unit_type": "section",
        "category": "amendment",
        "language": "en",
    },
    {
        "query": "Which section of Act A1727 inserts a new Division 1A into Part II of the PDPA?",
        "expected_doc": "Personal Data Protection (Amendment) Act 2024",
        "expected_unit": "6",
        "unit_type": "section",
        "category": "amendment",
        "language": "en",
    },
    {
        "query": "Which section of Act A1727 introduces the right to data portability?",
        "expected_doc": "Personal Data Protection (Amendment) Act 2024",
        "expected_unit": "9",
        "unit_type": "section",
        "category": "amendment",
        "language": "en",
    },
    {
        "query": "When does the Minimum Wages Order 2024 come into force?",
        "expected_doc": "Minimum Wages Order 2024",
        "expected_unit": "1",
        "unit_type": "section",
        "category": "gazette_order",
        "language": "en",
    },
    {
        "query": "Who is excluded from the Minimum Wages Order 2024?",
        "expected_doc": "Minimum Wages Order 2024",
        "expected_unit": "2",
        "unit_type": "section",
        "category": "gazette_order",
        "language": "en",
    },
    {
        "query": "What minimum wage rates apply from 1 February 2025 under the Minimum Wages Order 2024?",
        "expected_doc": "Minimum Wages Order 2024",
        "expected_unit": "3",
        "unit_type": "section",
        "category": "gazette_order",
        "language": "en",
    },
    {
        "query": "What minimum wage rates apply from 1 August 2025 under the Minimum Wages Order 2024?",
        "expected_doc": "Minimum Wages Order 2024",
        "expected_unit": "5",
        "unit_type": "section",
        "category": "gazette_order",
        "language": "en",
    },
    {
        "query": "Which earlier order is revoked by the Minimum Wages Order 2024?",
        "expected_doc": "Minimum Wages Order 2024",
        "expected_unit": "6",
        "unit_type": "section",
        "category": "gazette_order",
        "language": "en",
    },
]


def main() -> None:
    generated_candidates = _load_generated_candidates(CANDIDATE_PATH)
    selected_candidates = [_build_selected_payload(spec, generated_candidates) for spec in SELECTED_CANDIDATE_SPECS]
    units = load_unit_records(PROCESSED_DIR)
    unit_lookup = _build_unit_lookup(units)
    manual_payloads = [_build_manual_payload(entry, unit_lookup) for entry in MANUAL_ENTRIES]
    final_payloads = _dedupe_payloads(selected_candidates + manual_payloads)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for payload in final_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = _build_summary(
        final_payloads=final_payloads,
        generated_candidates=generated_candidates,
        selected_candidate_count=len(selected_candidates),
        manual_supplement_count=len(manual_payloads),
    )
    SUMMARY_PATH.write_text(summary, encoding="utf-8")


def _load_generated_candidates(path: Path) -> dict[str, GeneratedGoldCandidate]:
    by_query: dict[str, GeneratedGoldCandidate] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            candidate = GeneratedGoldCandidate(**payload)
            by_query[candidate.query] = candidate
    return by_query


def _build_selected_payload(
    spec: dict[str, str], generated_candidates: dict[str, GeneratedGoldCandidate]
) -> dict[str, object]:
    candidate = generated_candidates[spec["source_query"]]
    return {
        "query": spec.get("query", candidate.query),
        "expected_doc": candidate.expected_doc,
        "expected_unit": candidate.expected_unit,
        "unit_type": candidate.unit_type,
        "category": candidate.category,
        "language": candidate.language,
        "source_chunk_id": candidate.source_chunk_id,
        "generation_method": "curated_from_candidate",
        "confidence": candidate.confidence,
        "needs_review": False,
        "expected_act_title": candidate.expected_act_title,
        "expected_section_id": candidate.expected_section_id,
    }


def _build_unit_lookup(units: list[UnitRecord]) -> dict[tuple[str, str, str], UnitRecord]:
    lookup: dict[tuple[str, str, str], UnitRecord] = {}
    for unit in units:
        key = (unit.act_title, unit.unit_type, unit.unit_id)
        if key not in lookup:
            lookup[key] = unit
    return lookup


def _build_manual_payload(
    entry: dict[str, object], unit_lookup: dict[tuple[str, str, str], UnitRecord]
) -> dict[str, object]:
    key = (
        str(entry["expected_doc"]),
        str(entry["unit_type"]),
        str(entry["expected_unit"]),
    )
    unit = unit_lookup[key]
    return {
        "query": entry["query"],
        "expected_doc": entry["expected_doc"],
        "expected_unit": entry["expected_unit"],
        "unit_type": entry["unit_type"],
        "category": entry["category"],
        "language": entry["language"],
        "source_chunk_id": unit.chunk_id,
        "generation_method": "manual_curation",
        "confidence": 1.0,
        "needs_review": False,
        "expected_act_title": entry["expected_doc"],
        "expected_section_id": entry["expected_unit"],
    }


def _dedupe_payloads(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for payload in payloads:
        query = str(payload["query"]).strip()
        if query in seen:
            continue
        seen.add(query)
        deduped.append(payload)
    return deduped


def _build_summary(
    *,
    final_payloads: list[dict[str, object]],
    generated_candidates: dict[str, GeneratedGoldCandidate],
    selected_candidate_count: int,
    manual_supplement_count: int,
) -> str:
    by_category = Counter(str(payload["category"]) for payload in final_payloads)
    by_document = Counter(str(payload["expected_doc"] or "negative/none") for payload in final_payloads)
    by_language = Counter(str(payload["language"]) for payload in final_payloads)
    excluded_count = len(generated_candidates) - selected_candidate_count

    lines = [
        "# Final Gold Set V2 Summary",
        "",
        f"- Final benchmark size: {len(final_payloads)}",
        f"- Selected from auto-generated candidate pool: {selected_candidate_count}",
        f"- Manual supplements added: {manual_supplement_count}",
        f"- Removed auto-generated candidates: {excluded_count}",
        "",
        "## Counts By Category",
    ]
    for category, count in sorted(by_category.items()):
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Counts By Document"])
    for document, count in sorted(by_document.items()):
        lines.append(f"- {document}: {count}")

    lines.extend(["", "## Counts By Language"])
    for language, count in sorted(by_language.items()):
        lines.append(f"- {language}: {count}")

    lines.extend(
        [
            "",
            "## Curation Notes",
            "- Low-confidence amendment and gazette templates from the generator were replaced with grounded section-level queries tied to the refreshed corpus.",
            "- Malay-language coverage was manually expanded so bilingual evaluation is no longer limited to a handful of constitutional prompts.",
            "- Hierarchy coverage was curated manually because the generated hierarchy candidates were often too noisy or overly dependent on OCR-fragmented headings.",
            "- Core lookup, heading, definition, capability, rights, obligation, and negative probes still mostly come from the reviewed auto-generated candidate pool.",
            "",
            "## Category Balancing Decisions",
            "- Direct and heading lookups were capped to avoid letting templated section queries dominate the benchmark.",
            "- Capability, rights, and obligation prompts were kept as natural legal-user questions instead of raw template wording where possible.",
            "- Amendment and gazette/order queries were intentionally boosted because those document families are important in the expanded corpus but under-produced by the generator.",
            "- Negative queries were kept simple and deterministic so retrieval failure is easy to interpret during evaluation.",
            "",
            "## Sample Queries",
        ]
    )
    for payload in final_payloads[:20]:
        lines.append(
            f"- [{payload['category']}/{payload['language']}] {payload['query']} "
            f"(doc={payload['expected_doc']}, unit={payload['expected_unit']})"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
