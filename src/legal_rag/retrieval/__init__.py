"""Retrieval scaffolding for initial legal search flows."""

from .evaluation import (
    GoldMatch,
    GoldQuery,
    MISS_CATEGORIES,
    RetrievalEvaluationCase,
    RetrievalEvaluationSummary,
    evaluate_retrieval,
    load_gold_queries,
    write_evaluation_summary,
)
from .in_memory import EmbeddingRetriever, RetrievalResult, SimpleRetriever, SimpleVectorIndex
from .vector_store import JsonlVectorStore, StoredVectorRecord, chunk_from_record, load_chunk_records

__all__ = [
    "EmbeddingRetriever",
    "GoldMatch",
    "GoldQuery",
    "JsonlVectorStore",
    "MISS_CATEGORIES",
    "RetrievalResult",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationSummary",
    "SimpleRetriever",
    "SimpleVectorIndex",
    "StoredVectorRecord",
    "chunk_from_record",
    "evaluate_retrieval",
    "load_gold_queries",
    "load_chunk_records",
    "write_evaluation_summary",
]
