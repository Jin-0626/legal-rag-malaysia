"""Process raw law PDFs that are missing from the current processed corpus."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from legal_rag.config.settings import build_settings
from legal_rag.ingestion import PdfIngestionError, export_chunks_to_jsonl, extract_law_document_text, ingest_law_pdf_to_chunks
from legal_rag.ingestion.chunk_export import derive_act_number, derive_act_title, derive_document_aliases, derive_unit_type
from legal_rag.chunking.section_chunker import find_section_boundary_leaks

REPORT_PATH = Path("data/evaluation/processed_corpus_gap_report.json")
BAD_METADATA_PATTERN = re.compile(
    r"^(as at\s+\d|incorporating all amendments|online version|text of reprint|reprint|published by|4 disember 2024|4 december 2024)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawPdfInspection:
    pdf_path: Path
    act_title: str
    act_number: str
    unit_type: str
    aliases: tuple[str, ...]


def main() -> None:
    settings = build_settings()
    processed_inventory = _load_processed_inventory(settings.processed_dir)
    raw_pdfs = sorted(settings.raw_law_pdfs_dir.glob("*.pdf"))

    inspections: list[RawPdfInspection] = []
    failures: list[dict[str, object]] = []
    for pdf_path in raw_pdfs:
        try:
            inspections.append(_inspect_raw_pdf(pdf_path))
        except Exception as exc:  # pragma: no cover - exercised by live batch run
            failures.append(
                {
                    "source_file": pdf_path.name,
                    "status": "inspection_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    duplicate_skips: list[dict[str, object]] = []
    selected_for_processing: list[RawPdfInspection] = []
    grouped: dict[str, list[RawPdfInspection]] = defaultdict(list)
    for inspection in inspections:
        grouped[_normalize_key(inspection.act_title)].append(inspection)

    processed_title_keys = {_normalize_key(item["act_title"]) for item in processed_inventory}
    processed_stems = {item["processed_file_stem"] for item in processed_inventory}

    for title_key, group in sorted(grouped.items()):
        chosen = _choose_canonical_source(group)
        duplicates = [item for item in group if item != chosen]
        for duplicate in duplicates:
            duplicate_skips.append(
                {
                    "source_file": duplicate.pdf_path.name,
                    "reason": "duplicate_raw_pdf",
                    "canonical_source": chosen.pdf_path.name,
                    "act_title": duplicate.act_title,
                }
            )

        if title_key in processed_title_keys or chosen.pdf_path.stem in processed_stems:
            duplicate_skips.append(
                {
                    "source_file": chosen.pdf_path.name,
                    "reason": "already_processed",
                    "act_title": chosen.act_title,
                }
            )
            continue

        selected_for_processing.append(chosen)

    processed_results: list[dict[str, object]] = []
    for inspection in selected_for_processing:
        processed_results.append(_process_pdf(inspection, settings.processed_dir, settings.chunk_size_words, settings.chunk_overlap_words))

    after_inventory = _load_processed_inventory(settings.processed_dir)
    report = {
        "raw_pdf_count": len(raw_pdfs),
        "processed_document_count_before": len(processed_inventory),
        "processed_document_count_after": len(after_inventory),
        "newly_added_processed_docs": [item["source_file"] for item in processed_results if item["status"] == "processed"],
        "failed_docs": failures + [item for item in processed_results if item["status"] != "processed"],
        "duplicate_docs_skipped": duplicate_skips,
        "before_inventory": processed_inventory,
        "after_inventory": after_inventory,
        "new_document_validations": processed_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Raw PDFs: {len(raw_pdfs)}")
    print(f"Processed before: {len(processed_inventory)}")
    print(f"Processed after: {len(after_inventory)}")
    print(f"Newly processed: {len(report['newly_added_processed_docs'])}")
    print(f"Failed docs: {len(report['failed_docs'])}")
    print(f"Duplicate or already processed skips: {len(duplicate_skips)}")
    print(f"Report written to {REPORT_PATH}")


def _inspect_raw_pdf(pdf_path: Path) -> RawPdfInspection:
    document = extract_law_document_text(pdf_path)
    return RawPdfInspection(
        pdf_path=pdf_path,
        act_title=derive_act_title(document),
        act_number=derive_act_number(document),
        unit_type=derive_unit_type(document),
        aliases=derive_document_aliases(document),
    )


def _load_processed_inventory(processed_dir: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        if path.name.endswith(".smoke.jsonl"):
            continue
        first_line = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), None)
        if not first_line:
            continue
        payload = json.loads(first_line)
        inventory.append(
            {
                "processed_file": path.name,
                "processed_file_stem": path.stem,
                "act_title": payload.get("act_title", path.stem),
                "unit_type": payload.get("unit_type", "section"),
                "source_file": payload.get("source_file", ""),
            }
        )
    return inventory


def _choose_canonical_source(group: list[RawPdfInspection]) -> RawPdfInspection:
    return sorted(
        group,
        key=lambda item: (
            _duplicate_penalty(item.pdf_path.stem),
            len(item.pdf_path.stem),
            item.pdf_path.name.lower(),
        ),
    )[0]


def _duplicate_penalty(stem: str) -> tuple[int, int, int]:
    lowered = stem.lower()
    duplicate_marker = 1 if re.search(r"(\(\d+\)|_\d+$|\bcopy\b)", lowered) else 0
    noisy_marker = 1 if any(term in lowered for term in ("reprint-version", "updated", "online version")) else 0
    dated_prefix = 1 if re.match(r"^\d{8}", lowered) else 0
    return duplicate_marker, noisy_marker, dated_prefix


def _process_pdf(
    inspection: RawPdfInspection,
    processed_dir: Path,
    max_words: int,
    overlap_words: int,
) -> dict[str, object]:
    try:
        chunks = ingest_law_pdf_to_chunks(
            inspection.pdf_path,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    except (PdfIngestionError, RuntimeError, ValueError) as exc:
        return {
            "source_file": inspection.pdf_path.name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not chunks:
        return {
            "source_file": inspection.pdf_path.name,
            "status": "failed",
            "error": "No chunks produced",
        }

    output_path = processed_dir / f"{inspection.pdf_path.stem}.jsonl"
    export_chunks_to_jsonl(chunks, output_path)

    leaks = find_section_boundary_leaks(chunks)
    unit_counts = Counter(chunk.unit_type for chunk in chunks)
    dominant_unit_type = unit_counts.most_common(1)[0][0] if unit_counts else inspection.unit_type
    metadata_title = chunks[0].act_title
    return {
        "source_file": inspection.pdf_path.name,
        "processed_file": output_path.name,
        "status": "processed",
        "chunk_count": len(chunks),
        "detected_unit_type": dominant_unit_type,
        "unit_type_counts": dict(unit_counts),
        "mixed_unit_types": len(unit_counts) > 1,
        "leakage_count": len(leaks),
        "canonical_metadata": _looks_canonical(metadata_title),
        "act_title": metadata_title,
        "aliases": list(chunks[0].document_aliases),
    }


def _looks_canonical(title: str) -> bool:
    normalized = title.strip()
    if not normalized:
        return False
    return BAD_METADATA_PATTERN.match(normalized) is None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
if __name__ == "__main__":
    main()
