"""Evaluate lexical, embedding, and hybrid retrieval variants against a legal gold set."""

from __future__ import annotations

import json

from legal_rag.config.settings import build_settings
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval import JsonlVectorStore, SearchMode, evaluate_retrieval, load_gold_queries


def main() -> None:
    settings = build_settings()
    gold_path = settings.data_dir / "evaluation" / "hybrid_retrieval_gold.jsonl"
    vector_store_path = settings.embeddings_dir / "legal-corpus.vectors.jsonl"
    output_path = settings.data_dir / "evaluation" / "hybrid_retrieval_report.json"

    gold_queries = load_gold_queries(gold_path)
    vector_store = JsonlVectorStore(vector_store_path)
    embedder = OllamaEmbedder()

    report = {
        "gold_path": str(gold_path),
        "vector_store_path": str(vector_store_path),
        "modes": {},
    }
    for mode in ("lexical", "embedding", "hybrid", "hybrid_rerank", "hybrid_filtered_rerank"):
        summary = evaluate_retrieval(
            gold_queries=gold_queries,
            vector_store=vector_store,
            embedder=embedder,
            top_k=3,
            mode=mode,
        )
        direct_cases = [case for case in summary.cases if case.query_type == "direct_lookup"]
        direct_top1 = 0.0
        if direct_cases:
            direct_top1 = sum(1 for case in direct_cases if case.hit_at_1) / len(direct_cases)
        report["modes"][mode] = {
            "total_queries": summary.total_queries,
            "hit_at_1": summary.hit_at_1,
            "hit_at_3": summary.hit_at_3,
            "wrong_section_rate": summary.wrong_section_rate,
            "direct_lookup_top_1": direct_top1,
            "error_breakdown": summary.error_breakdown,
            "cases": [
                {
                    "query": case.query,
                    "query_type": case.query_type,
                    "expected_act_title": case.expected_act_title,
                    "expected_section_id": case.expected_section_id,
                    "expected_subsection_id": case.expected_subsection_id,
                    "hit_at_1": case.hit_at_1,
                    "hit_at_3": case.hit_at_3,
                    "miss_category": case.miss_category,
                    "matches": [
                        {
                            "rank": match.rank,
                            "chunk_id": match.chunk_id,
                            "act_title": match.act_title,
                            "section_id": match.section_id,
                            "subsection_id": match.subsection_id,
                            "score": match.score,
                            "act_match": match.act_match,
                            "section_match": match.section_match,
                            "subsection_match": match.subsection_match,
                            "target_match": match.target_match,
                        }
                        for match in case.matches
                    ],
                }
                for case in summary.cases
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"queries={len(gold_queries)}")
    for mode in ("lexical", "embedding", "hybrid", "hybrid_rerank", "hybrid_filtered_rerank"):
        summary = report["modes"][mode]
        print(
            f"{mode}: hit@1={summary['hit_at_1']:.3f} "
            f"hit@3={summary['hit_at_3']:.3f} "
            f"direct_lookup@1={summary['direct_lookup_top_1']:.3f} "
            f"wrong_section_rate={summary['wrong_section_rate']:.3f}"
        )
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
