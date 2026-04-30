from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_GOLD_PATH = ROOT / "data" / "evaluation" / "final_gold_set_v2.jsonl"
REPORT_PATH = ROOT / "data" / "evaluation" / "final_gold_set_v2_report.md"


def test_final_gold_set_v2_has_expected_shape() -> None:
    rows = [json.loads(line) for line in FINAL_GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert 70 <= len(rows) <= 80
    assert len({row["query"] for row in rows}) == len(rows)

    categories = {row["category"] for row in rows}
    assert {
        "direct_lookup",
        "heading_lookup",
        "definition",
        "capability",
        "rights",
        "obligation",
        "bilingual",
        "negative",
        "hierarchy",
        "amendment",
        "gazette_order",
    }.issubset(categories)

    languages = {row["language"] for row in rows}
    assert {"en", "ms"}.issubset(languages)

    documents = {row["expected_doc"] for row in rows if row["expected_doc"]}
    assert {
        "Employment Act 1955",
        "Federal Constitution",
        "Personal Data Protection Act 2010",
        "Personal Data Protection (Amendment) Act 2024",
    }.issubset(documents)


def test_final_gold_set_report_mentions_benchmark_results() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert "Final Gold Set V2 Retrieval Report" in report
    assert "Total queries: 73" in report
    assert "### hybrid_rerank" in report
    assert "- hit@1: 1.000" in report
