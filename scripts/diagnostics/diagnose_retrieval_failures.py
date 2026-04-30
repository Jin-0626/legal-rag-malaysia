"""Diagnose remaining retrieval misses with candidate-pool and rerank attribution."""

from __future__ import annotations

import json
from dataclasses import asdict

from legal_rag.config.settings import build_settings
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder
from legal_rag.retrieval import JsonlVectorStore, evaluate_retrieval, load_gold_queries
from legal_rag.retrieval.in_memory import (
    FILTER_CANDIDATE_POOL,
    RERANK_CANDIDATE_POOL,
    _build_query_context,
    _combine_scores,
    _cosine_similarity,
    _cross_reference_density_penalty,
    _definition_heading_boost,
    _document_match_score,
    _exact_unit_match_boost,
    _heading_match_score,
    _heading_recall_boost,
    _lexical_similarity,
    _position_bias,
    _subsection_match_boost,
    _should_apply_rerank,
    filter_and_prerank_candidates,
    rerank_candidates,
    search_embedded_entries,
)
from legal_rag.retrieval.vector_store import chunk_from_record


def main() -> None:
    settings = build_settings()
    gold_path = settings.data_dir / "evaluation" / "hybrid_retrieval_gold.jsonl"
    vector_store_path = settings.embeddings_dir / "legal-corpus.vectors.jsonl"
    output_path = settings.data_dir / "evaluation" / "retrieval_failure_attribution.json"

    gold_queries = load_gold_queries(gold_path)
    gold_lookup = {query.query: query for query in gold_queries}
    vector_store = JsonlVectorStore(vector_store_path)
    embedder = OllamaEmbedder()
    summary = evaluate_retrieval(
        gold_queries=gold_queries,
        vector_store=vector_store,
        embedder=embedder,
        top_k=3,
        mode="hybrid_filtered_rerank",
    )
    records = vector_store.load_records()
    entries = [
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
        for record in records
    ]

    report = {
        "gold_path": str(gold_path),
        "vector_store_path": str(vector_store_path),
        "mode": summary.mode,
        "failed_queries": [],
    }

    for case in summary.cases:
        if case.hit_at_1:
            continue
        gold_query = gold_lookup[case.query]
        context = _build_query_context(gold_query.query)
        query_embedding = embedder.embed([gold_query.query])[0]
        scored_entries = []
        for entry in entries:
            chunk = entry.chunk
            lexical_score = _lexical_similarity(context.query_tokens, chunk)
            semantic_score = max(0.0, _cosine_similarity(query_embedding, entry.embedding))
            document_score = _document_match_score(context, chunk)
            subsection_boost = _subsection_match_boost(context, chunk)
            definition_boost = _definition_heading_boost(context, chunk)
            hybrid_score = _combine_scores(
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                document_score=document_score,
                subsection_boost=subsection_boost,
                definition_boost=definition_boost,
                heading_recall_boost=_heading_recall_boost(context, chunk),
                mode="hybrid",
            )
            scored_entries.append((entry, hybrid_score))
        scored_entries.sort(key=lambda item: (-item[1], item[0].chunk.chunk_index, item[0].chunk.chunk_id))

        expected_entries = [
            (entry, score)
            for entry, score in scored_entries
            if entry.chunk.act_title == gold_query.expected_act_title
            and entry.chunk.section_id == gold_query.expected_section_id
            and (
                gold_query.expected_subsection_id is None
                or entry.chunk.subsection_id == gold_query.expected_subsection_id
            )
        ]
        expected_entry, expected_hybrid_score = expected_entries[0]
        hybrid_pool = search_embedded_entries(
            entries=entries,
            query=gold_query.query,
            embedder=embedder,
            top_k=FILTER_CANDIDATE_POOL,
            mode="hybrid",
        )
        filtered_pool = filter_and_prerank_candidates(
            query=gold_query.query,
            candidates=hybrid_pool,
            top_k=RERANK_CANDIDATE_POOL,
        )
        rerank_applied = _should_apply_rerank(context)
        reranked = rerank_candidates(gold_query.query, filtered_pool, top_k=3)
        top_chunk = reranked[0].chunk if reranked else None
        final_top_match = case.matches[0] if case.matches else None

        report["failed_queries"].append(
            {
                "query": gold_query.query,
                "rerank_applied_in_live_path": rerank_applied,
                "expected": {
                    "act_title": gold_query.expected_act_title,
                    "section_id": gold_query.expected_section_id,
                    "subsection_id": gold_query.expected_subsection_id,
                    "chunk_id": expected_entry.chunk.chunk_id,
                },
                "expected_in_hybrid_pool": _result_rank(hybrid_pool, expected_entry.chunk.chunk_id),
                "expected_survived_filtering": _result_rank(filtered_pool, expected_entry.chunk.chunk_id),
                "final_top_1": asdict(final_top_match) if final_top_match is not None else None,
                "hypothetical_top_1_after_filter_rerank": {
                    "chunk_id": top_chunk.chunk_id if top_chunk else None,
                    "act_title": top_chunk.act_title if top_chunk else None,
                    "section_id": top_chunk.section_id if top_chunk else None,
                },
                "expected_features": _feature_breakdown(context, expected_entry, expected_hybrid_score),
                "top_1_features": _feature_breakdown(
                    context,
                    next(entry for entry, _ in scored_entries if entry.chunk.chunk_id == top_chunk.chunk_id),
                    next(score for entry, score in scored_entries if entry.chunk.chunk_id == top_chunk.chunk_id),
                )
                if top_chunk
                else None,
                "final_case": asdict(case),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"failed_queries={len(report['failed_queries'])}")
    print(f"report={output_path}")


def _result_rank(results, chunk_id: str) -> int | None:
    for index, result in enumerate(results, start=1):
        if result.chunk.chunk_id == chunk_id:
            return index
    return None


def _feature_breakdown(context, entry: EmbeddedChunk, hybrid_score: float) -> dict[str, object]:
    chunk = entry.chunk
    lexical_score = _lexical_similarity(context.query_tokens, chunk)
    heading_match = _heading_match_score(context, chunk)
    document_score = _document_match_score(context, chunk)
    definition_bias = _definition_heading_boost(context, chunk)
    exact_unit_boost = _exact_unit_match_boost(context, chunk)
    cross_reference_penalty = _cross_reference_density_penalty(chunk)
    position_bias = _position_bias(chunk)
    rerank_score = (
        (hybrid_score * 0.35)
        + (heading_match * 1.4)
        + (definition_bias * 1.6)
        + (_heading_recall_boost(context, chunk) * 1.2)
        + (position_bias * 0.9)
        + (exact_unit_boost * 1.6)
        + (document_score * 2.0)
        - (cross_reference_penalty * 1.1)
    )
    return {
        "chunk_id": chunk.chunk_id,
        "act_title": chunk.act_title,
        "section_id": chunk.section_id,
        "subsection_id": chunk.subsection_id,
        "heading": chunk.section_heading,
        "lexical_score": lexical_score,
        "heading_match": heading_match,
        "document_score": document_score,
        "definition_bias": definition_bias,
        "heading_recall_boost": _heading_recall_boost(context, chunk),
        "exact_unit_boost": exact_unit_boost,
        "cross_reference_penalty": cross_reference_penalty,
        "position_bias": position_bias,
        "hybrid_score": hybrid_score,
        "rerank_score_without_index_penalty": rerank_score,
    }


if __name__ == "__main__":
    main()
