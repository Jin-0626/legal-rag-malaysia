"""Evaluate all retrieval modes on the curated Final Gold Set V2 benchmark."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from legal_rag.config.settings import build_settings
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder
from legal_rag.evaluation.goldset_generator import load_unit_records
from legal_rag.graph import build_legal_graph
from legal_rag.retrieval import SearchMode
from legal_rag.retrieval.in_memory import search_embedded_entries
from legal_rag.retrieval.vector_store import JsonlVectorStore, chunk_from_record
from legal_rag.workflows import graph_supported_search


BASELINE_MODES: tuple[SearchMode, ...] = (
    "lexical",
    "embedding",
    "hybrid",
    "hybrid_rerank",
    "hybrid_filtered_rerank",
)
GRAPH_MODES: tuple[str, ...] = (
    "graph_supported",
    "hybrid_plus_graph",
    "hybrid_plus_graph_with_graph_rerank",
)
MODES: tuple[str, ...] = BASELINE_MODES + GRAPH_MODES


def main() -> None:
    settings = build_settings()
    gold_path = settings.data_dir / "evaluation" / "final_gold_set_v2.jsonl"
    vector_store_path = settings.embeddings_dir / "legal-corpus.vectors.jsonl"
    report_json_path = settings.data_dir / "evaluation" / "final_gold_set_v2_report.json"
    report_md_path = settings.data_dir / "evaluation" / "final_gold_set_v2_report.md"

    benchmark = _load_benchmark(gold_path)
    embedder = OllamaEmbedder()
    vector_store = JsonlVectorStore(vector_store_path)
    entries = _load_embedded_entries(vector_store)
    graph = build_legal_graph([entry.chunk for entry in entries])
    alias_lookup = _build_alias_lookup(settings.processed_dir)

    report: dict[str, Any] = {
        "gold_path": str(gold_path),
        "vector_store_path": str(vector_store_path),
        "total_queries": len(benchmark),
        "modes": {},
    }

    for mode in MODES:
        report["modes"][mode] = _evaluate_mode(
            benchmark=benchmark,
            entries=entries,
            embedder=embedder,
            graph=graph,
            alias_lookup=alias_lookup,
            mode=mode,
            top_k=3,
        )

    report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md_path.write_text(_format_markdown_report(report), encoding="utf-8")

    print(f"queries={len(benchmark)}")
    for mode in MODES:
        summary = report["modes"][mode]["overall"]
        print(
            f"{mode}: hit@1={summary['hit_at_1']:.3f} "
            f"hit@3={summary['hit_at_3']:.3f} "
            f"wrong_section_rate={summary['wrong_section_rate']:.3f} "
            f"wrong_document_rate={summary['wrong_document_rate']:.3f} "
            f"negative_no_answer_accuracy={summary['negative_no_answer_accuracy']:.3f}"
        )
    print(f"report_json={report_json_path}")
    print(f"report_md={report_md_path}")


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_embedded_entries(vector_store: JsonlVectorStore) -> list[EmbeddedChunk]:
    entries: list[EmbeddedChunk] = []
    for record in vector_store.load_records():
        entries.append(
            EmbeddedChunk(
                chunk=chunk_from_record(
                    {
                        "chunk_id": record.chunk_id,
                        "document_id": record.document_id,
                        "act_title": record.act_title,
                        "act_number": record.act_number,
                        "section_heading": record.section_heading,
                        "section_id": record.section_id,
                        "unit_type": record.unit_type,
                        "unit_id": record.unit_id,
                        "subsection_id": record.subsection_id,
                        "paragraph_id": record.paragraph_id,
                        "source_file": record.source_file,
                        "source_path": record.source_path,
                        "chunk_index": record.chunk_index,
                        "document_aliases": record.document_aliases,
                        "text": record.text,
                    }
                ),
                embedding=record.embedding,
            )
        )
    return entries


def _build_alias_lookup(processed_dir: Path) -> list[tuple[str, str]]:
    alias_pairs: set[tuple[str, str]] = set()
    for unit in load_unit_records(processed_dir):
        alias_pairs.add((unit.act_title.lower(), unit.act_title))
        for alias in unit.document_aliases:
            alias_pairs.add((alias.lower(), unit.act_title))
    # Prefer longer aliases first so short names like "Act 53" do not win too early.
    return sorted(alias_pairs, key=lambda item: (-len(item[0]), item[0]))


def _evaluate_mode(
    *,
    benchmark: list[dict[str, Any]],
    entries: list[EmbeddedChunk],
    embedder: OllamaEmbedder,
    graph: Any,
    alias_lookup: list[tuple[str, str]],
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    positive_queries = 0
    negative_queries = 0
    hit_at_1_count = 0
    hit_at_3_count = 0
    wrong_section_count = 0
    wrong_document_count = 0
    negative_no_answer_count = 0

    category_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    language_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    document_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_buckets: Counter[str] = Counter()

    for record in benchmark:
        query = record["query"]
        category = str(record["category"])
        language = str(record["language"])
        expected_doc = record.get("expected_doc")
        expected_unit = record.get("expected_unit")
        is_negative = expected_doc is None or expected_unit is None or category == "negative"

        if mode in GRAPH_MODES:
            results = graph_supported_search(
                entries=entries,
                embedder=embedder,
                graph=graph,
                query=query,
                top_k=top_k,
                mode=mode,
            )
        else:
            results = search_embedded_entries(entries, query, embedder, top_k=top_k, mode=mode)
        match_payloads = [
            {
                "rank": index + 1,
                "chunk_id": result.chunk.chunk_id,
                "act_title": result.chunk.act_title,
                "unit_type": result.chunk.unit_type,
                "unit_id": result.chunk.unit_id or result.chunk.section_id,
                "section_heading": result.chunk.section_heading,
                "score": result.score,
            }
            for index, result in enumerate(results)
        ]
        top_1 = match_payloads[0] if match_payloads else None

        if is_negative:
            negative_queries += 1
            referenced_doc = _infer_referenced_doc(query, alias_lookup)
            no_answer_success = _negative_no_answer_success(results, referenced_doc)
            negative_no_answer_count += int(no_answer_success)
            failure_bucket = None if no_answer_success else "no_answer_failure"
            case = {
                "query": query,
                "category": category,
                "language": language,
                "expected_doc": None,
                "expected_unit": None,
                "is_negative": True,
                "referenced_doc": referenced_doc,
                "hit_at_1": None,
                "hit_at_3": None,
                "wrong_document": None,
                "wrong_unit_in_right_document": None,
                "no_answer_success": no_answer_success,
                "failure_bucket": failure_bucket,
                "actual_top_1": top_1,
                "matches": match_payloads,
            }
        else:
            positive_queries += 1
            hit_at_1 = bool(
                top_1
                and top_1["act_title"] == expected_doc
                and str(top_1["unit_id"]) == str(expected_unit)
            )
            hit_at_3 = any(
                match["act_title"] == expected_doc and str(match["unit_id"]) == str(expected_unit)
                for match in match_payloads[:top_k]
            )
            wrong_document = bool(top_1 and top_1["act_title"] != expected_doc)
            wrong_unit_in_right_document = bool(
                top_1
                and top_1["act_title"] == expected_doc
                and str(top_1["unit_id"]) != str(expected_unit)
            )
            hit_at_1_count += int(hit_at_1)
            hit_at_3_count += int(hit_at_3)
            wrong_document_count += int(wrong_document)
            wrong_section_count += int(wrong_unit_in_right_document)
            failure_bucket = _classify_failure_bucket(
                category=category,
                hit_at_1=hit_at_1,
                top_1=top_1,
                expected_doc=str(expected_doc),
            )
            case = {
                "query": query,
                "category": category,
                "language": language,
                "expected_doc": expected_doc,
                "expected_unit": expected_unit,
                "is_negative": False,
                "referenced_doc": None,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "wrong_document": wrong_document,
                "wrong_unit_in_right_document": wrong_unit_in_right_document,
                "no_answer_success": None,
                "failure_bucket": failure_bucket,
                "actual_top_1": top_1,
                "matches": match_payloads,
            }

        if case["failure_bucket"]:
            failure_buckets[str(case["failure_bucket"])] += 1

        category_groups[category].append(case)
        language_groups[language].append(case)
        document_groups[str(expected_doc or "negative/none")].append(case)
        cases.append(case)

    overall = {
        "total_queries": len(benchmark),
        "positive_queries": positive_queries,
        "negative_queries": negative_queries,
        "hit_at_1": _ratio(hit_at_1_count, positive_queries),
        "hit_at_3": _ratio(hit_at_3_count, positive_queries),
        "wrong_section_rate": _ratio(wrong_section_count, positive_queries),
        "wrong_document_rate": _ratio(wrong_document_count, positive_queries),
        "negative_no_answer_accuracy": _ratio(negative_no_answer_count, negative_queries),
        "failure_bucket_counts": dict(sorted(failure_buckets.items())),
    }

    return {
        "overall": overall,
        "by_category": _summarize_groups(category_groups),
        "by_language": _summarize_groups(language_groups),
        "by_document": _summarize_groups(document_groups),
        "failed_cases": [case for case in cases if case["failure_bucket"] is not None],
        "cases": cases,
    }


def _infer_referenced_doc(query: str, alias_lookup: list[tuple[str, str]]) -> str | None:
    lowered = query.lower()
    for alias, act_title in alias_lookup:
        if alias and alias in lowered:
            return act_title
    return None


def _negative_no_answer_success(results: list[Any], referenced_doc: str | None) -> bool:
    if referenced_doc is None:
        return len(results) == 0
    return all(result.chunk.act_title != referenced_doc for result in results)


def _classify_failure_bucket(
    *,
    category: str,
    hit_at_1: bool,
    top_1: dict[str, Any] | None,
    expected_doc: str,
) -> str | None:
    if hit_at_1:
        return None
    if category == "hierarchy":
        return "hierarchy_failure"
    if category == "amendment":
        return "amendment_failure"
    if category == "gazette_order":
        return "gazette_failure"
    if top_1 is None:
        return "no_answer_failure"
    if top_1["act_title"] != expected_doc:
        return "wrong_document"
    return "wrong_unit_in_right_document"


def _summarize_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group_name, cases in sorted(groups.items()):
        positive_cases = [case for case in cases if not case["is_negative"]]
        negative_cases = [case for case in cases if case["is_negative"]]
        summary[group_name] = {
            "total_queries": len(cases),
            "positive_queries": len(positive_cases),
            "negative_queries": len(negative_cases),
            "hit_at_1": _ratio(sum(1 for case in positive_cases if case["hit_at_1"]), len(positive_cases)),
            "hit_at_3": _ratio(sum(1 for case in positive_cases if case["hit_at_3"]), len(positive_cases)),
            "wrong_section_rate": _ratio(
                sum(1 for case in positive_cases if case["wrong_unit_in_right_document"]),
                len(positive_cases),
            ),
            "wrong_document_rate": _ratio(
                sum(1 for case in positive_cases if case["wrong_document"]),
                len(positive_cases),
            ),
            "negative_no_answer_accuracy": _ratio(
                sum(1 for case in negative_cases if case["no_answer_success"]),
                len(negative_cases),
            ),
            "failure_bucket_counts": dict(
                sorted(Counter(str(case["failure_bucket"]) for case in cases if case["failure_bucket"]).items())
            ),
        }
    return summary


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _format_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Final Gold Set V2 Retrieval Report",
        "",
        f"- Benchmark: `{report['gold_path']}`",
        f"- Vector store: `{report['vector_store_path']}`",
        f"- Total queries: {report['total_queries']}",
        "- Note: `negative_no_answer_accuracy` is a retrieval-side proxy. A negative query counts as correct only when top-k results avoid surfacing the referenced document as a false answer.",
        "",
        "## Overall Results",
    ]

    for mode in MODES:
        overall = report["modes"][mode]["overall"]
        lines.extend(
            [
                f"### {mode}",
                f"- hit@1: {overall['hit_at_1']:.3f}",
                f"- hit@3: {overall['hit_at_3']:.3f}",
                f"- wrong-section rate: {overall['wrong_section_rate']:.3f}",
                f"- wrong-document rate: {overall['wrong_document_rate']:.3f}",
                f"- no-answer accuracy for negative queries: {overall['negative_no_answer_accuracy']:.3f}",
                f"- failure buckets: {overall['failure_bucket_counts']}",
                "",
            ]
        )

    best_mode = max(
        MODES,
        key=lambda mode: (
            report["modes"][mode]["overall"]["hit_at_1"],
            report["modes"][mode]["overall"]["hit_at_3"],
        ),
    )
    lines.extend(
        [
            "## Best Mode",
            f"- Best hit@1 on this benchmark: `{best_mode}`",
            "",
            "## Category Breakdown",
        ]
    )
    for mode in MODES:
        lines.append(f"### {mode}")
        for category, metrics in report["modes"][mode]["by_category"].items():
            lines.append(
                f"- {category}: count={metrics['total_queries']}, "
                f"hit@1={metrics['hit_at_1']:.3f}, hit@3={metrics['hit_at_3']:.3f}, "
                f"wrong_doc={metrics['wrong_document_rate']:.3f}, "
                f"wrong_unit={metrics['wrong_section_rate']:.3f}, "
                f"negative_no_answer={metrics['negative_no_answer_accuracy']:.3f}"
            )
        lines.append("")

    lines.append("## Language Breakdown")
    for mode in MODES:
        lines.append(f"### {mode}")
        for language, metrics in report["modes"][mode]["by_language"].items():
            lines.append(
                f"- {language}: count={metrics['total_queries']}, "
                f"hit@1={metrics['hit_at_1']:.3f}, hit@3={metrics['hit_at_3']:.3f}, "
                f"wrong_doc={metrics['wrong_document_rate']:.3f}, "
                f"wrong_unit={metrics['wrong_section_rate']:.3f}, "
                f"negative_no_answer={metrics['negative_no_answer_accuracy']:.3f}"
            )
        lines.append("")

    lines.append("## Key Failures")
    for mode in MODES:
        failed_cases = report["modes"][mode]["failed_cases"]
        lines.append(f"### {mode}")
        if not failed_cases:
            lines.append("- No failed cases.")
            lines.append("")
            continue
        for case in failed_cases[:12]:
            actual = case["actual_top_1"]
            if actual is None:
                actual_summary = "none"
            else:
                actual_summary = f"{actual['act_title']} {actual['unit_id']} ({actual['section_heading']})"
            lines.append(
                f"- [{case['failure_bucket']}] {case['query']} "
                f"(expected={case['expected_doc'] or case['referenced_doc']}/{case['expected_unit'] or 'none'}, "
                f"actual_top_1={actual_summary})"
            )
        if len(failed_cases) > 12:
            lines.append(f"- ... and {len(failed_cases) - 12} more failed cases in JSON.")
        lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
