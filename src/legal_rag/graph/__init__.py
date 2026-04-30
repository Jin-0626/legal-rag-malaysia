"""Minimal legal structure graph utilities for targeted GraphRAG retrieval."""

from .legal_graph import (
    GraphRetrievalResult,
    HierarchyNode,
    LegalGraph,
    UnitNode,
    build_legal_graph,
    search_graph,
)

__all__ = [
    "GraphRetrievalResult",
    "HierarchyNode",
    "LegalGraph",
    "UnitNode",
    "build_legal_graph",
    "search_graph",
]
