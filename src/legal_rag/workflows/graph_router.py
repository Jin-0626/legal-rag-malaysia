"""Deterministic router for hybrid-vs-graph legal retrieval workflows."""

from __future__ import annotations

from typing import Literal

from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder
from legal_rag.graph.legal_graph import LegalGraph, search_graph
from legal_rag.retrieval.in_memory import (
    FILTER_CANDIDATE_POOL,
    RERANK_CANDIDATE_POOL,
    GraphSignal,
    RetrievalResult,
    filter_and_prerank_candidates,
    rerank_candidates,
    search_embedded_entries,
)

GraphMode = Literal["graph_supported", "hybrid_plus_graph", "hybrid_plus_graph_with_graph_rerank"]


def graph_supported_search(
    *,
    entries: list[EmbeddedChunk],
    embedder: OllamaEmbedder,
    graph: LegalGraph,
    query: str,
    top_k: int = 3,
    mode: GraphMode = "graph_supported",
) -> list[RetrievalResult]:
    """Route supported graph queries to graph retrieval and fall back safely otherwise."""

    hybrid_results = search_embedded_entries(entries, query, embedder, top_k=max(top_k, 6), mode="hybrid")
    if _looks_like_hierarchy_query(query):
        if mode == "hybrid_plus_graph_with_graph_rerank":
            return search_embedded_entries(entries, query, embedder, top_k=top_k, mode="hybrid_filtered_rerank")
        return hybrid_results[:top_k]
    graph_results = search_graph(graph, query, top_k=max(top_k, 6))

    if not graph_results:
        if mode == "hybrid_plus_graph_with_graph_rerank":
            return search_embedded_entries(entries, query, embedder, top_k=top_k, mode="hybrid_filtered_rerank")
        return hybrid_results[:top_k]
    if hybrid_results and graph_results[0].chunk.document_id != hybrid_results[0].chunk.document_id:
        if mode == "hybrid_plus_graph_with_graph_rerank":
            return search_embedded_entries(entries, query, embedder, top_k=top_k, mode="hybrid_filtered_rerank")
        return hybrid_results[:top_k]
    if mode == "graph_supported":
        return [RetrievalResult(chunk=result.chunk, score=result.score) for result in graph_results[:top_k]]
    target_document_id = hybrid_results[0].chunk.document_id if hybrid_results else graph_results[0].chunk.document_id
    safe_graph_results = [result for result in graph_results if result.chunk.document_id == target_document_id]
    if mode == "hybrid_plus_graph_with_graph_rerank":
        return _graph_rerank_results(query, entries, embedder, hybrid_results, safe_graph_results, top_k=top_k)
    return _merge_results(
        hybrid_results,
        safe_graph_results,
        top_k=top_k,
    )


def _merge_results(
    hybrid_results: list[RetrievalResult],
    graph_results: list,
    *,
    top_k: int,
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for result in hybrid_results:
        merged[result.chunk.chunk_id] = result
    for result in graph_results:
        existing = merged.get(result.chunk.chunk_id)
        graph_weighted = RetrievalResult(chunk=result.chunk, score=result.score + 5.0)
        if existing is None or graph_weighted.score > existing.score:
            merged[result.chunk.chunk_id] = graph_weighted
    return sorted(
        merged.values(),
        key=lambda result: (-result.score, result.chunk.chunk_index, result.chunk.chunk_id),
    )[:top_k]


def _graph_rerank_results(
    query: str,
    entries: list[EmbeddedChunk],
    embedder: OllamaEmbedder,
    hybrid_results: list[RetrievalResult],
    graph_results: list,
    *,
    top_k: int,
) -> list[RetrievalResult]:
    base_results = search_embedded_entries(
        entries,
        query,
        embedder,
        top_k=max(top_k, FILTER_CANDIDATE_POOL),
        mode="hybrid",
    )
    filtered = filter_and_prerank_candidates(query, base_results, top_k=RERANK_CANDIDATE_POOL)
    merged_candidates = _merge_results(filtered, graph_results, top_k=RERANK_CANDIDATE_POOL)
    graph_signals = {
        result.chunk.chunk_id: GraphSignal(
            graph_support=True,
            path_strength=_graph_path_strength(result.reason),
            rank_position=index,
            reason=result.reason,
        )
        for index, result in enumerate(graph_results, start=1)
    }
    if hybrid_results:
        hybrid_document_id = hybrid_results[0].chunk.document_id
        graph_signals = {
            chunk_id: signal
            for chunk_id, signal in graph_signals.items()
            if any(candidate.chunk.chunk_id == chunk_id and candidate.chunk.document_id == hybrid_document_id for candidate in merged_candidates)
        }
    return rerank_candidates(query, merged_candidates, top_k=top_k, graph_signals=graph_signals)


def _graph_path_strength(reason: str) -> float:
    if "amendment linkage" in reason:
        return 2.4
    if "explicit reference" in reason:
        return 2.0
    if "hierarchy" in reason:
        return 1.6
    return 1.0


def _looks_like_hierarchy_query(query: str) -> bool:
    lowered = query.lower()
    return any(label in lowered for label in ("part ", "chapter ", "division ", "bahagian ", "bab ", "schedule ", "jadual "))
