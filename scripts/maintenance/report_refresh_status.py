"""Generate reproducible index and Gold Set refresh summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EVALUATION_DIR = DATA_DIR / "evaluation"

UNDERREPRESENTED_CATEGORY_THRESHOLD = 15
LOW_CANDIDATE_DOC_THRESHOLD = 18


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def filename_like_alias(alias: str) -> bool:
    lowered = alias.lower()
    return (
        ".pdf" in lowered
        or ".jsonl" in lowered
        or "\\" in alias
        or "/" in alias
        or lowered.endswith(".smoke")
    )


def build_index_summary() -> dict[str, Any]:
    processed_files = sorted(
        path
        for path in PROCESSED_DIR.glob("*.jsonl")
        if not path.name.endswith(".smoke.jsonl")
    )
    skipped = []

    for path in sorted(PROCESSED_DIR.iterdir()):
        if path.suffix != ".jsonl":
            skipped.append(
                {"name": path.name, "reason": "non_jsonl_processed_artifact"}
            )
        elif path.name.endswith(".smoke.jsonl"):
            skipped.append({"name": path.name, "reason": "smoke_artifact_excluded"})

    counts_by_document: Counter[str] = Counter()
    counts_by_processed_file: dict[str, int] = {}
    metadata_anomalies: list[dict[str, Any]] = []
    mixed_unit_documents: dict[str, Counter[str]] = defaultdict(Counter)

    for path in processed_files:
        records = load_jsonl(path)
        counts_by_processed_file[path.name] = len(records)
        if not records:
            continue

        document_title = records[0].get("act_title") or path.stem
        counts_by_document[document_title] += len(records)

        aliases = records[0].get("document_aliases") or []
        alias_issues = [
            alias for alias in aliases if not alias or filename_like_alias(str(alias))
        ]
        if alias_issues:
            metadata_anomalies.append(
                {
                    "document": document_title,
                    "issues": ["filename_like_aliases_present"],
                    "aliases": aliases,
                }
            )

        for record in records:
            unit_type = (record.get("unit_type") or "unknown").strip().lower()
            mixed_unit_documents[document_title][unit_type] += 1

    mixed_unit_report = [
        {"document": document, "unit_type_counts": dict(counter)}
        for document, counter in mixed_unit_documents.items()
        if len(counter) > 1
    ]

    combined_vector_path = EMBEDDINGS_DIR / "legal-corpus.vectors.jsonl"
    combined_vector_records = (
        count_jsonl_rows(combined_vector_path) if combined_vector_path.exists() else 0
    )

    return {
        "processed_docs_used": len(processed_files),
        "processed_files_used": [path.name for path in processed_files],
        "total_chunks_indexed": sum(counts_by_processed_file.values()),
        "counts_by_document": dict(sorted(counts_by_document.items())),
        "counts_by_processed_file": counts_by_processed_file,
        "documents_skipped": skipped,
        "combined_vector_records": combined_vector_records,
        "metadata_anomalies": metadata_anomalies,
        "mixed_unit_documents": mixed_unit_report,
    }


def build_index_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Index Refresh Summary",
        "",
        f"- Processed docs used: {summary['processed_docs_used']}",
        f"- Total chunks indexed: {summary['total_chunks_indexed']}",
        f"- Combined vector records: {summary['combined_vector_records']}",
        "",
        "## Counts by document",
    ]
    for document, count in summary["counts_by_document"].items():
        lines.append(f"- {document}: {count}")

    lines.extend(["", "## Skipped documents"])
    for skipped in summary["documents_skipped"]:
        lines.append(f"- {skipped['name']}: {skipped['reason']}")

    if summary["mixed_unit_documents"]:
        lines.extend(["", "## Mixed unit typing"])
        for item in summary["mixed_unit_documents"]:
            lines.append(f"- {item['document']}: {item['unit_type_counts']}")

    if summary["metadata_anomalies"]:
        lines.extend(["", "## Metadata anomalies"])
        for item in summary["metadata_anomalies"]:
            lines.append(f"- {item['document']}: {', '.join(item['issues'])}")

    return "\n".join(lines) + "\n"


def build_gold_set_stats(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(
        str(candidate.get("category") or "unknown") for candidate in candidates
    )
    document_counts = Counter(
        str(candidate.get("expected_doc") or "negative/none") for candidate in candidates
    )
    language_counts = Counter(
        str(candidate.get("language") or "unknown") for candidate in candidates
    )

    return {
        "total_candidates": len(candidates),
        "counts_by_category": dict(sorted(category_counts.items())),
        "counts_by_document": dict(sorted(document_counts.items())),
        "counts_by_language": dict(sorted(language_counts.items())),
    }


def build_refresh_comparison(
    current_candidates: list[dict[str, Any]], baseline_path: Path
) -> dict[str, Any]:
    baseline: dict[str, Any] = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    before = baseline.get("before", {})
    after = build_gold_set_stats(current_candidates)

    before_documents = set(before.get("counts_by_document", {}))
    after_documents = set(after.get("counts_by_document", {}))

    low_candidate_documents = {
        document: count
        for document, count in after["counts_by_document"].items()
        if document != "negative/none" and count < LOW_CANDIDATE_DOC_THRESHOLD
    }
    underrepresented_categories = {
        category: count
        for category, count in after["counts_by_category"].items()
        if count < UNDERREPRESENTED_CATEGORY_THRESHOLD
    }

    index_summary = build_index_summary()

    return {
        "before": before,
        "after": after,
        "candidate_delta": after["total_candidates"] - before.get("total_candidates", 0),
        "newly_represented_documents": sorted(after_documents - before_documents),
        "gazette_order_present": after["counts_by_category"].get("gazette_order", 0) > 0,
        "low_candidate_documents": dict(sorted(low_candidate_documents.items())),
        "underrepresented_categories": dict(sorted(underrepresented_categories.items())),
        "metadata_anomalies": index_summary["metadata_anomalies"],
        "mixed_unit_documents": index_summary["mixed_unit_documents"],
    }


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    index_summary = build_index_summary()
    comparison_baseline_path = EVALUATION_DIR / "gold_set_v2_refresh_comparison.json"
    current_candidates = load_jsonl(EVALUATION_DIR / "gold_set_v2_candidates.jsonl")
    comparison = build_refresh_comparison(current_candidates, comparison_baseline_path)

    index_summary_path = EVALUATION_DIR / "index_refresh_summary.json"
    index_summary_md_path = EVALUATION_DIR / "index_refresh_summary.md"
    comparison_path = EVALUATION_DIR / "gold_set_v2_refresh_comparison.json"

    index_summary_path.write_text(
        json.dumps(index_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index_summary_md_path.write_text(
        build_index_summary_markdown(index_summary), encoding="utf-8"
    )
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"index_summary={index_summary_path}")
    print(f"index_summary_md={index_summary_md_path}")
    print(f"comparison={comparison_path}")


if __name__ == "__main__":
    main()
