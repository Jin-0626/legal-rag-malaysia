"""Diagnose reranker behavior on Final Gold Set V2 failure clusters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag.config.settings import build_settings
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.in_memory import (
    _build_query_context,
    _cross_reference_density_penalty,
    _definition_heading_boost,
    _definition_mismatch_penalty,
    _heading_match_score,
    _position_bias,
    _rerank_document_boost,
    _should_apply_rerank,
    filter_and_prerank_candidates,
    rerank_candidates,
)
from legal_rag.retrieval.vector_store import JsonlVectorStore

FOCUS_CATEGORIES = {"heading_lookup", "bilingual", "amendment"}


def main() -> None:
    settings = build_settings()
    gold_path = settings.data_dir / "evaluation" / "final_gold_set_v2.jsonl"
    vector_store_path = settings.embeddings_dir / "legal-corpus.vectors.jsonl"
    report_json_path = settings.data_dir / "evaluation" / "reranker_diagnostics_report.json"
    report_md_path = settings.data_dir / "evaluation" / "reranker_diagnostics_report.md"

    benchmark = _load_jsonl(gold_path)
    vector_store = JsonlVectorStore(vector_store_path)
    embedder = OllamaEmbedder()

    cases: list[dict[str, Any]] = []
    summary = {
        "focus_queries": 0,
        "triggered_queries": 0,
        "correct_present_before_rerank": 0,
        "hybrid_top1_correct": 0,
        "rerank_top1_correct": 0,
        "filtered_rerank_top1_correct": 0,
        "hybrid_only_wins": 0,
        "rerank_only_wins": 0,
        "filtered_rerank_only_wins": 0,
    }

    for record in benchmark:
        category = str(record["category"])
        expected_doc = record.get("expected_doc")
        expected_unit = record.get("expected_unit")
        if expected_doc is None or expected_unit is None:
            continue
        if category not in FOCUS_CATEGORIES:
            continue

        query = record["query"]
        context = _build_query_context(query)
        hybrid_results = vector_store.search(query, embedder, top_k=10, mode="hybrid")
        reranked_results = rerank_candidates(query, hybrid_results, top_k=10)
        filtered_candidates = filter_and_prerank_candidates(query, hybrid_results, top_k=10)
        filtered_reranked_results = rerank_candidates(query, filtered_candidates, top_k=10)

        hybrid_hit = _find_rank(hybrid_results, expected_doc, str(expected_unit))
        rerank_hit = _find_rank(reranked_results, expected_doc, str(expected_unit))
        filtered_hit = _find_rank(filtered_reranked_results, expected_doc, str(expected_unit))
        case = {
            "query": query,
            "category": category,
            "expected_doc": expected_doc,
            "expected_unit": str(expected_unit),
            "rerank_triggered": _should_apply_rerank(context),
            "correct_present_before_rerank": hybrid_hit is not None,
            "hybrid_hit_rank": hybrid_hit,
            "rerank_hit_rank": rerank_hit,
            "filtered_rerank_hit_rank": filtered_hit,
            "hybrid_top_k": [_candidate_payload(context, result) for result in hybrid_results[:5]],
            "rerank_top_k": [_candidate_payload(context, result) for result in reranked_results[:5]],
            "filtered_rerank_top_k": [_candidate_payload(context, result) for result in filtered_reranked_results[:5]],
        }
        cases.append(case)

        summary["focus_queries"] += 1
        summary["triggered_queries"] += int(case["rerank_triggered"])
        summary["correct_present_before_rerank"] += int(case["correct_present_before_rerank"])
        summary["hybrid_top1_correct"] += int(hybrid_hit == 1)
        summary["rerank_top1_correct"] += int(rerank_hit == 1)
        summary["filtered_rerank_top1_correct"] += int(filtered_hit == 1)
        summary["hybrid_only_wins"] += int(hybrid_hit == 1 and rerank_hit != 1)
        summary["rerank_only_wins"] += int(rerank_hit == 1 and hybrid_hit != 1)
        summary["filtered_rerank_only_wins"] += int(filtered_hit == 1 and hybrid_hit != 1)

    payload = {"summary": summary, "cases": cases}
    report_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(_format_markdown(payload), encoding="utf-8")

    print(f"focus_queries={summary['focus_queries']}")
    print(f"triggered_queries={summary['triggered_queries']}")
    print(f"report_json={report_json_path}")
    print(f"report_md={report_md_path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_rank(results, expected_doc: str, expected_unit: str) -> int | None:
    for index, result in enumerate(results, 1):
        if result.chunk.act_title == expected_doc and str(result.chunk.unit_id) == expected_unit:
            return index
    return None


def _candidate_payload(context, result) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "act_title": chunk.act_title,
        "unit_id": chunk.unit_id,
        "section_heading": chunk.section_heading,
        "score": round(result.score, 4),
        "features": {
            "heading_match": round(_heading_match_score(context, chunk), 4),
            "definition_bias": round(_definition_heading_boost(context, chunk), 4),
            "position_bias": round(_position_bias(chunk), 4),
            "document_boost": round(_rerank_document_boost(context, chunk), 4),
            "cross_reference_penalty": round(_cross_reference_density_penalty(chunk), 4),
            "definition_mismatch_penalty": round(_definition_mismatch_penalty(context, chunk), 4),
        },
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Reranker Diagnostics",
        "",
        "## Summary",
        "",
        f"- Focus queries: {summary['focus_queries']}",
        f"- Triggered queries: {summary['triggered_queries']}",
        f"- Correct candidate present before rerank: {summary['correct_present_before_rerank']}",
        f"- Hybrid top-1 correct: {summary['hybrid_top1_correct']}",
        f"- Hybrid rerank top-1 correct: {summary['rerank_top1_correct']}",
        f"- Hybrid filtered rerank top-1 correct: {summary['filtered_rerank_top1_correct']}",
        f"- Hybrid-only wins: {summary['hybrid_only_wins']}",
        f"- Rerank-only wins: {summary['rerank_only_wins']}",
        f"- Filtered-rerank-only wins: {summary['filtered_rerank_only_wins']}",
        "",
        "## Cases",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"### {case['query']}")
        lines.append("")
        lines.append(f"- Category: {case['category']}")
        lines.append(f"- Expected: {case['expected_doc']} / {case['expected_unit']}")
        lines.append(f"- Triggered: {case['rerank_triggered']}")
        lines.append(f"- Correct in hybrid top-k: {case['correct_present_before_rerank']}")
        lines.append(f"- Hybrid hit rank: {case['hybrid_hit_rank']}")
        lines.append(f"- Rerank hit rank: {case['rerank_hit_rank']}")
        lines.append(f"- Filtered rerank hit rank: {case['filtered_rerank_hit_rank']}")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
