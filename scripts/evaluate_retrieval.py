"""Evaluate retrieval quality against a small gold set of statute queries."""

from __future__ import annotations

from pathlib import Path

from legal_rag.config.settings import build_settings
from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval import JsonlVectorStore, evaluate_retrieval, load_gold_queries, write_evaluation_summary


def main() -> None:
    settings = build_settings()
    gold_path = settings.data_dir / "evaluation" / "retrieval_gold.jsonl"
    vector_store_path = settings.embeddings_dir / "personal-data-protection-act-2010.smoke.vectors.jsonl"
    output_path = settings.data_dir / "evaluation" / "retrieval_report.json"

    gold_queries = load_gold_queries(gold_path)
    vector_store = JsonlVectorStore(vector_store_path)
    embedder = OllamaEmbedder()
    summary = evaluate_retrieval(gold_queries, vector_store, embedder, top_k=3)
    write_evaluation_summary(summary, output_path)

    print(f"queries={summary.total_queries}")
    print(f"hit@1={summary.hit_at_1:.3f}")
    print(f"hit@3={summary.hit_at_3:.3f}")
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
