"""Hybrid legal retrieval with exact-unit, lexical, and embedding ranking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from legal_rag.chunking.models import Chunk
from legal_rag.embeddings.embedder import EmbeddedChunk, OllamaEmbedder

SearchMode = Literal["embedding", "lexical", "hybrid", "hybrid_rerank", "hybrid_filtered_rerank"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
UNIT_QUERY_PATTERN = re.compile(r"\b(section|article|perkara)\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
HIERARCHY_QUERY_PATTERN = re.compile(r"\b(part|chapter|bahagian|bab|division)\s+([A-Za-z0-9IVXLCDM]+)\b", re.IGNORECASE)
HIERARCHY_LINE_PATTERN = re.compile(
    r"^(part|chapter|bahagian|bab|division)\s+([A-Za-z0-9IVXLCDM]+)\s*[:.\-]?\s*$",
    re.IGNORECASE,
)
PARENTHETICAL_SUFFIX_PATTERN = re.compile(r"\s*\([^)]*\)")
PUA_PATTERN = re.compile(r"\bp\.?\s*u\.?\s*\(?a\)?\b", re.IGNORECASE)
SUBSECTION_QUERY_PATTERN = re.compile(
    r"\bsubsection\s*\(?(\d+[A-Za-z]?)\)?\b|\((\d+[A-Za-z]?)\)",
    re.IGNORECASE,
)
CROSS_REFERENCE_PATTERN = re.compile(
    r"\b(section|sections|article|articles|perkara|perkara-perkara)\s+\d+[A-Za-z]?\b",
    re.IGNORECASE,
)
DOCUMENT_STOPWORDS = {
    "the",
    "of",
    "and",
    "act",
    "akta",
    "laws",
    "law",
    "malaysia",
    "reprint",
    "cetakan",
    "semula",
    "constitution",
    "federal",
    "perlembagaan",
}
QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "for",
    "under",
    "what",
    "does",
    "do",
    "is",
    "are",
    "be",
    "by",
    "in",
    "on",
    "who",
    "when",
    "where",
    "which",
}
DEFINITION_CUES = (
    "definition",
    "defined",
    "define",
    "meaning of",
    "mean",
    "maksud",
    "interpretation",
    "tafsiran",
    "ditakrifkan",
)
OBLIGATION_TEXT_PATTERN = re.compile(r"\b(shall|must|required|obliged)\b", re.IGNORECASE)
RERANK_CANDIDATE_POOL = 10
FILTER_CANDIDATE_POOL = 16
MAX_WEAK_EMBEDDING_CANDIDATES = 2
RERANK_CONFIDENCE_MARGIN = 0.4
RERANK_CUES = ("titled", "definition", "define", "defined", "interpretation", "maksud", "apakah")
STRICT_DEFINITION_CUES = (
    "definition",
    "define",
    "defined",
    "interpretation",
    "maksud",
    "mean",
    "tafsiran",
    "ditakrifkan",
)
OBLIGATION_CUES = ("what does", "what must", "must ")
CAPABILITY_CUES = ("who can", "who may", "who is entitled to")
LEGAL_ACTION_CUES = ("appeal", "apply", "claim")
DOCUMENT_HINT_CUES = (" act ", "akta", "constitution", "perlembagaan", "pdpa", "federal")
HEADING_LOOKUP_CUES = (
    "which section of",
    "which article of",
    "which perkara",
    "deals with",
    "sets out",
    "covers",
    "berkaitan",
    "seksyen manakah",
    "perkara manakah",
    "apakah kandungan",
    "terpakai",
)
EMPLOYMENT_AGREEMENT_CORE_CUES = (
    "employment agreement",
    "employment contract",
    "contract of service",
    "terms and conditions",
)
EMPLOYMENT_AGREEMENT_ADVICE_CUES = (
    "before signing",
    "signing",
    "protect myself",
    "protect yourself",
    "need to check",
    "should i check",
    "what should i check",
    "employee rights",
)
EMPLOYMENT_AGREEMENT_NARROW_PENALTY_TERMS = (
    "foreign employee",
    "foreign domestic employee",
    "director general",
)
AMENDMENT_QUERY_CUES = (
    "amendment",
    "amends",
    "amend ",
    "principal act",
    "act a",
    "come into force",
    "introduces",
    "new section",
    "pindaan",
)
LEXICAL_NORMALIZATION = {
    "appeals": "appeal",
    "appeal": "appeal",
    "applies": "apply",
    "apply": "apply",
    "application": "apply",
    "scope": "scope",
    "define": "define",
    "defined": "define",
    "definition": "define",
    "interpret": "interpret",
    "interpretation": "interpret",
    "require": "require",
    "requires": "require",
    "required": "require",
    "requirement": "require",
    "duty": "obligation",
    "duties": "obligation",
    "obligation": "obligation",
    "obligations": "obligation",
    "seksyen": "section",
    "perkara": "article",
    "akta": "act",
    "terpakai": "apply",
    "pindaan": "amendment",
    "komersial": "commercial",
    "cuti": "leave",
    "tahunan": "annual",
    "tafsiran": "interpret",
    "ditakrifkan": "define",
    "dikuatkuasakan": "commencement",
    "dikuatkuasa": "commencement",
    "dikuatkuasakan": "commencement",
    "dikecualikan": "exclude",
    "pengecualian": "exclude",
    "introduces": "introduce",
    "introduced": "introduce",
    "introduction": "introduce",
    "changes": "amendment",
    "changed": "amendment",
    "amends": "amendment",
    "amend": "amendment",
    "excluded": "exclude",
    "exclusion": "exclude",
    "wages": "wage",
    "rates": "rate",
}
RECALL_FAMILY_TERMS = {
    "appeal",
    "apply",
    "scope",
    "define",
    "interpret",
    "require",
    "obligation",
    "amendment",
    "commencement",
    "exclude",
    "order",
    "wage",
    "rate",
}


@dataclass(frozen=True)
class RetrievalResult:
    """Scored retrieval result for one chunk."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class QueryContext:
    """Normalized retrieval hints parsed from a legal query."""

    raw_query: str
    normalized_query: str
    compact_query: str
    query_tokens: tuple[str, ...]
    unit_type: str | None
    unit_id: str | None
    subsection_id: str | None
    is_definition_query: bool
    is_hierarchy_query: bool
    hierarchy_label: str | None
    hierarchy_id: str | None
    is_obligation_query: bool
    is_employment_agreement_query: bool


@dataclass(frozen=True)
class CandidateFeatures:
    """Lightweight legal-aware features used for filtering and reranking."""

    result: RetrievalResult
    lexical_score: float
    heading_match: float
    document_score: float
    cross_reference_penalty: float
    definition_bias: float
    exact_unit_boost: float
    weak_embedding_match: bool


@dataclass(frozen=True)
class GraphSignal:
    """Optional graph-derived support signal for a rerank candidate."""

    graph_support: bool
    path_strength: float
    rank_position: int
    reason: str


class SimpleVectorIndex:
    """Store embedded chunks in memory and search them with hybrid ranking."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.embedder = embedder
        self._entries: list[EmbeddedChunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._entries.extend(self.embedder.embed_chunks(chunks))

    def search(self, query: str, top_k: int = 3, mode: SearchMode = "hybrid") -> list[RetrievalResult]:
        return search_embedded_entries(
            entries=self._entries,
            query=query,
            embedder=self.embedder,
            top_k=top_k,
            mode=mode,
        )


class SimpleRetriever:
    """Compatibility wrapper around the embedding-based in-memory vector index."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.index = SimpleVectorIndex(embedder)

    def add(self, chunks: list[Chunk]) -> None:
        """Add embedded chunks to the current in-memory index."""

        self.index.add(chunks)

    def search(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 3,
        mode: SearchMode = "hybrid",
    ) -> list[RetrievalResult]:
        """Rebuild the in-memory index from chunks and search with hybrid ranking."""

        self.index = SimpleVectorIndex(self.index.embedder)
        self.index.add(chunks)
        return self.index.search(query, top_k=top_k, mode=mode)


class EmbeddingRetriever(SimpleRetriever):
    """Explicit embedding-based retriever alias for downstream clarity."""


def search_embedded_entries(
    entries: list[EmbeddedChunk],
    query: str,
    embedder: OllamaEmbedder,
    top_k: int = 3,
    mode: SearchMode = "hybrid",
) -> list[RetrievalResult]:
    """Rank embedded chunks using exact-unit, lexical, and semantic signals."""

    if not query.strip() or not entries:
        return []

    if mode in {"hybrid_rerank", "hybrid_filtered_rerank"}:
        context = _build_query_context(query)
        if _is_impossible_unit_lookup(entries, context):
            return []
        base_results = search_embedded_entries(
            entries=entries,
            query=query,
            embedder=embedder,
            top_k=max(top_k, FILTER_CANDIDATE_POOL if mode == "hybrid_filtered_rerank" else RERANK_CANDIDATE_POOL),
            mode="hybrid",
        )
        if not _should_apply_rerank(context):
            return base_results[:top_k]
        if mode == "hybrid_filtered_rerank":
            base_results = filter_and_prerank_candidates(query, base_results, top_k=RERANK_CANDIDATE_POOL)
        if context.is_definition_query and not any(
            _definition_heading_boost(context, candidate.chunk) > 0.0 for candidate in base_results
        ):
            return base_results[:top_k]
        if _should_skip_rerank(context, base_results):
            return base_results[:top_k]
        return rerank_candidates(query, base_results, top_k=top_k)

    context = _build_query_context(query)
    if _is_impossible_unit_lookup(entries, context):
        return []
    query_embedding: list[float] | None = None
    if mode in {"embedding", "hybrid"}:
        query_embedding = embedder.embed([query])[0]

    exact_results = _search_exact_unit_matches(entries, context, top_k=top_k, mode=mode)
    if exact_results:
        referenced_chunk = _infer_referenced_chunk(entries, context)
        if referenced_chunk is not None:
            exact_results = _refine_referenced_document_ranking(context, exact_results, referenced_chunk)
        return exact_results

    hierarchy_results = _search_hierarchy_candidates(entries, context, top_k=top_k, mode=mode)
    if hierarchy_results:
        return hierarchy_results

    scored: list[RetrievalResult] = []
    for entry in entries:
        lexical_score = _lexical_similarity(context.query_tokens, entry.chunk)
        semantic_score = 0.0
        if query_embedding is not None:
            semantic_score = max(0.0, _cosine_similarity(query_embedding, entry.embedding))
        document_score = _document_match_score(context, entry.chunk)
        subsection_boost = _subsection_match_boost(context, entry.chunk)
        definition_boost = _definition_heading_boost(context, entry.chunk)
        heading_recall_boost = _heading_recall_boost(context, entry.chunk)
        hierarchy_boost = _hierarchy_match_boost(context, entry.chunk)
        obligation_boost = _obligation_text_boost(context, entry.chunk)
        obligation_heading_boost = _obligation_heading_boost(context, entry.chunk)
        obligation_internal_score = _obligation_internal_score(context, entry.chunk)
        obligation_definition_penalty = _obligation_definition_penalty(context, entry.chunk)
        employment_agreement_score = _employment_agreement_score(context, entry.chunk)
        score = _combine_scores(
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            document_score=document_score,
            subsection_boost=subsection_boost,
            definition_boost=definition_boost,
            heading_recall_boost=heading_recall_boost,
            hierarchy_boost=hierarchy_boost,
            obligation_boost=obligation_boost,
            obligation_heading_boost=obligation_heading_boost,
            obligation_internal_score=obligation_internal_score,
            obligation_definition_penalty=obligation_definition_penalty,
            employment_agreement_score=employment_agreement_score,
            mode=mode,
        )
        if score <= 0.0:
            continue
        scored.append(RetrievalResult(chunk=entry.chunk, score=score))

    ranked = _sort_results(scored)
    referenced_chunk = _infer_referenced_chunk(entries, context)
    if referenced_chunk is not None:
        ranked = _refine_referenced_document_ranking(context, ranked, referenced_chunk)
    if context.is_obligation_query and referenced_chunk is not None:
        ranked = _prefer_referenced_document(context, ranked, referenced_chunk)
    if mode == "hybrid":
        ranked = _ensure_recall_candidates(context, ranked, top_k=top_k)
    return ranked[:top_k]


def rerank_candidates(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 3,
    graph_signals: dict[str, GraphSignal] | None = None,
) -> list[RetrievalResult]:
    """Rerank already-retrieved candidates using lightweight legal-specific signals."""

    if not candidates:
        return []

    context = _build_query_context(query)
    lowered = context.raw_query.lower()
    is_heading_lookup = _looks_like_heading_lookup_query(lowered)
    is_amendment_query = _looks_like_amendment_query(lowered)
    candidate_weight = 0.5 if (is_heading_lookup or is_amendment_query) else 0.35
    heading_weight = 2.2 if is_heading_lookup else 1.4
    position_weight = 0.25 if (is_heading_lookup or is_amendment_query) else 0.9
    cross_reference_weight = 0.45 if is_amendment_query else 1.1
    reranked: list[RetrievalResult] = []
    has_definition_heading = any(
        _definition_heading_boost(context, candidate.chunk) > 0.0 for candidate in candidates
    )
    for index, candidate in enumerate(candidates):
        chunk = candidate.chunk
        heading_match = _heading_match_score(context, chunk)
        definition_bias = _definition_heading_boost(context, chunk)
        position_bias = _position_bias(chunk)
        exact_unit_boost = _exact_unit_match_boost(context, chunk)
        document_boost = _rerank_document_boost(context, chunk)
        cross_reference_penalty = _cross_reference_density_penalty(chunk)
        definition_mismatch_penalty = _definition_mismatch_penalty(context, chunk)
        amendment_boost = _amendment_rerank_boost(lowered, chunk) if is_amendment_query else 0.0
        graph_boost = _graph_signal_boost(
            context,
            signal=graph_signals.get(chunk.chunk_id) if graph_signals else None,
        )
        rerank_score = (
            (candidate.score * candidate_weight)
            + (heading_match * heading_weight)
            + (definition_bias * 1.6)
            + (position_bias * position_weight)
            + (exact_unit_boost * 1.6)
            + (document_boost * 2.0)
            + amendment_boost
            + graph_boost
            - (cross_reference_penalty * cross_reference_weight)
            - (definition_mismatch_penalty * 2.0)
            - (index * 0.01)
        )
        if has_definition_heading and context.is_definition_query:
            if definition_bias > 0.0:
                rerank_score += 2.0
            else:
                rerank_score -= 0.75
        reranked.append(RetrievalResult(chunk=chunk, score=rerank_score))
    reranked = _sort_results(reranked)
    reranked = _refine_same_document_heading_order(context, reranked)
    return reranked[:top_k]


def _graph_signal_boost(context: QueryContext, signal: GraphSignal | None) -> float:
    if signal is None or not signal.graph_support:
        return 0.0
    boost = signal.path_strength
    boost += max(0.0, 1.0 - (signal.rank_position * 0.15))
    if "amendment linkage" in signal.reason:
        boost += 1.1
    if "explicit reference" in signal.reason:
        boost += 0.9
    if "hierarchy" in signal.reason:
        boost += 0.8
    if context.is_hierarchy_query and "hierarchy" in signal.reason:
        boost += 0.6
    return boost


def filter_and_prerank_candidates(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = RERANK_CANDIDATE_POOL,
) -> list[RetrievalResult]:
    """Trim noisy hybrid candidates before reranking using legal-aware heuristics."""

    if not candidates:
        return []

    context = _build_query_context(query)
    features = [_candidate_features(context, candidate) for candidate in candidates]
    strong_document_intent = any(feature.document_score >= 1.0 for feature in features)
    has_heading_candidates = any(feature.heading_match > 0.0 for feature in features)
    has_definition_heading = any(feature.definition_bias > 0.0 for feature in features)
    matching_document_count = sum(1 for feature in features if feature.document_score >= 1.0)
    weak_embedding_kept = 0
    filtered: list[RetrievalResult] = []

    for feature in _sort_candidate_features(features, context):
        if strong_document_intent and feature.document_score == 0.0 and matching_document_count >= 2:
            if feature.heading_match <= 0.0 and feature.exact_unit_boost <= 0.0:
                continue
        if context.is_definition_query and has_definition_heading:
            if feature.definition_bias <= 0.0 and feature.heading_match <= 0.0 and feature.document_score <= 0.0:
                continue
        if has_heading_candidates and feature.heading_match <= 0.0 and feature.lexical_score < 0.2:
            if feature.document_score <= 0.0 and feature.exact_unit_boost <= 0.0:
                if feature.weak_embedding_match:
                    if weak_embedding_kept >= MAX_WEAK_EMBEDDING_CANDIDATES:
                        continue
                    weak_embedding_kept += 1
                else:
                    continue
        filtered.append(
            RetrievalResult(
                chunk=feature.result.chunk,
                score=_candidate_pre_rank_score(context, feature),
            )
        )
        if len(filtered) >= top_k:
            break

    if filtered:
        return _sort_results(filtered)[:top_k]
    return _sort_results(candidates)[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


def _build_query_context(query: str) -> QueryContext:
    normalized_surface_query = _normalize_query_surface(query)
    normalized_query = _normalize_text(normalized_surface_query)
    compact_query = normalized_query.replace(" ", "")
    unit_match = UNIT_QUERY_PATTERN.search(normalized_surface_query)
    hierarchy_match = HIERARCHY_QUERY_PATTERN.search(normalized_surface_query)
    subsection_match = SUBSECTION_QUERY_PATTERN.search(normalized_surface_query)
    subsection_id = None
    if subsection_match is not None:
        subsection_id = subsection_match.group(1) or subsection_match.group(2)
    lowered_query = query.lower()
    return QueryContext(
        raw_query=query,
        normalized_query=normalized_query,
        compact_query=compact_query,
        query_tokens=tuple(_tokenize_query(query)),
        unit_type=unit_match.group(1).lower() if unit_match else None,
        unit_id=(unit_match.group(2).upper() if unit_match else None),
        subsection_id=subsection_id.upper() if subsection_id else None,
        is_definition_query=_looks_like_definition_query(query),
        is_hierarchy_query=hierarchy_match is not None,
        hierarchy_label=hierarchy_match.group(1).lower() if hierarchy_match else None,
        hierarchy_id=hierarchy_match.group(2).upper() if hierarchy_match else None,
        is_obligation_query=_looks_like_obligation_query(lowered_query),
        is_employment_agreement_query=_looks_like_employment_agreement_query(lowered_query),
    )


def _search_exact_unit_matches(
    entries: list[EmbeddedChunk],
    context: QueryContext,
    top_k: int,
    mode: SearchMode,
) -> list[RetrievalResult]:
    if mode == "embedding":
        return []
    if not _should_use_exact_unit_lookup(context):
        return []
    if context.unit_type is None or context.unit_id is None:
        return []

    matches = [
        entry.chunk
        for entry in entries
        if entry.chunk.unit_type.lower() == context.unit_type
        and (entry.chunk.unit_id or entry.chunk.section_id).upper() == context.unit_id
    ]
    if not matches:
        return []

    ranked = []
    for chunk in matches:
        document_score = _document_match_score(context, chunk)
        subsection_boost = _subsection_match_boost(context, chunk)
        lexical_score = _lexical_similarity(context.query_tokens, chunk)
        definition_boost = _definition_heading_boost(context, chunk)
        heading_recall_boost = _heading_recall_boost(context, chunk)
        hierarchy_boost = _hierarchy_match_boost(context, chunk)
        obligation_boost = _obligation_text_boost(context, chunk)
        obligation_heading_boost = _obligation_heading_boost(context, chunk)
        obligation_internal_score = _obligation_internal_score(context, chunk)
        obligation_definition_penalty = _obligation_definition_penalty(context, chunk)
        # Exact legal-unit hits dominate the final ordering.
        score = (
            10.0
            + (document_score * 2.0)
            + subsection_boost
            + definition_boost
            + heading_recall_boost
            + hierarchy_boost
            + obligation_boost
            + obligation_heading_boost
            + obligation_internal_score
            - obligation_definition_penalty
            + lexical_score
        )
        ranked.append(RetrievalResult(chunk=chunk, score=score))
    return _sort_results(ranked)[:top_k]


def _search_hierarchy_candidates(
    entries: list[EmbeddedChunk],
    context: QueryContext,
    top_k: int,
    mode: SearchMode,
) -> list[RetrievalResult]:
    if mode == "embedding" or not context.is_hierarchy_query:
        return []
    referenced_chunk = _infer_referenced_chunk(entries, context)
    if referenced_chunk is None:
        return []

    ordered_chunks = sorted(
        (
            entry.chunk
            for entry in entries
            if entry.chunk.document_id == referenced_chunk.document_id
            and entry.chunk.source_path == referenced_chunk.source_path
        ),
        key=lambda chunk: (chunk.chunk_index, chunk.chunk_id),
    )
    if not ordered_chunks:
        return []

    begin_intent = any(
        term in context.raw_query.lower()
        for term in ("begin", "begins", "start", "starts", "open", "opens", "memulakan")
    )
    ranked_by_id: dict[str, RetrievalResult] = {}
    for index, chunk in enumerate(ordered_chunks):
        if not _chunk_contains_hierarchy(context, chunk):
            continue
        chunk_starts_with_heading = _chunk_starts_with_unit_heading(chunk)
        hierarchy_reliability = _hierarchy_marker_reliability(chunk)
        base_score = (
            7.0
            + (_document_match_score(context, chunk) * 3.0)
            + _lexical_similarity(context.query_tokens, chunk)
            + hierarchy_reliability
        )
        if begin_intent:
            base_score += 1.5
            if chunk_starts_with_heading:
                base_score -= 2.0
            if _is_compact_hierarchy_tail(chunk):
                base_score -= 1.0
            if chunk.unit_type.lower() == "section":
                base_score += 1.0
            if chunk.unit_type.lower() in {"article", "perkara"}:
                base_score -= 1.5
        _store_best_result(ranked_by_id, RetrievalResult(chunk=chunk, score=base_score))

        if context.is_definition_query:
            for scoped_chunk in _hierarchy_scope_chunks(ordered_chunks, index):
                scoped_score = _hierarchy_definition_scope_score(context, scoped_chunk)
                if scoped_score <= 0.0:
                    continue
                _store_best_result(
                    ranked_by_id,
                    RetrievalResult(
                        chunk=scoped_chunk,
                        score=(
                            8.5
                            + (_document_match_score(context, scoped_chunk) * 3.0)
                            + _lexical_similarity(context.query_tokens, scoped_chunk)
                            + scoped_score
                        ),
                    ),
                )

        if not begin_intent:
            continue
        successor = _next_unit_chunk(ordered_chunks, index, chunk)
        if successor is None:
            continue
        successor_score = (
            6.0
            + (_document_match_score(context, successor) * 3.0)
            + _lexical_similarity(context.query_tokens, successor)
            + _document_internal_position_bonus(successor)
        )
        if chunk_starts_with_heading:
            successor_score += 0.5 if _is_compact_hierarchy_tail(chunk) else 1.25
        elif _is_compact_hierarchy_tail(chunk):
            successor_score += 3.0
        else:
            successor_score += 0.75
        if chunk.unit_type.lower() in {"article", "perkara"}:
            successor_score += 1.5
        successor_score -= _cross_reference_density_penalty(chunk) * 1.5
        _store_best_result(ranked_by_id, RetrievalResult(chunk=successor, score=successor_score))

    return _sort_results(list(ranked_by_id.values()))[:top_k]


def _combine_scores(
    *,
    lexical_score: float,
    semantic_score: float,
    document_score: float,
    subsection_boost: float,
    definition_boost: float,
    heading_recall_boost: float,
    hierarchy_boost: float,
    obligation_boost: float,
    obligation_heading_boost: float,
    obligation_internal_score: float,
    obligation_definition_penalty: float,
    employment_agreement_score: float,
    mode: SearchMode,
) -> float:
    if mode == "lexical":
        return (
            lexical_score
            + (document_score * 0.8)
            + subsection_boost
            + definition_boost
            + heading_recall_boost
            + hierarchy_boost
            + obligation_boost
            + obligation_heading_boost
            + obligation_internal_score
            - obligation_definition_penalty
            + (employment_agreement_score * 1.6)
        )
    if mode == "embedding":
        return (
            semantic_score
            + (document_score * 0.15)
            + subsection_boost
            + definition_boost
            + (heading_recall_boost * 0.1)
            + (hierarchy_boost * 0.2)
            + (obligation_boost * 0.2)
            + (obligation_heading_boost * 0.2)
            + (obligation_internal_score * 0.2)
            - (obligation_definition_penalty * 0.2)
            + (employment_agreement_score * 0.2)
        )
    return (
        (lexical_score * 0.8)
        + (semantic_score * 0.2)
        + (document_score * 0.8)
        + subsection_boost
        + definition_boost
        + (heading_recall_boost * 1.2)
        + (hierarchy_boost * 1.6)
        + (obligation_boost * 1.2)
        + (obligation_heading_boost * 1.6)
        + (obligation_internal_score * 1.4)
        - (obligation_definition_penalty * 1.2)
        + (employment_agreement_score * 1.8)
    )


def _lexical_similarity(query_tokens: tuple[str, ...], chunk: Chunk) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(chunk.text, QUERY_STOPWORDS))
    heading_tokens = set(_tokenize(chunk.section_heading, QUERY_STOPWORDS))
    if not text_tokens and not heading_tokens:
        return 0.0

    overlap = sum(1 for token in query_tokens if token in text_tokens)
    heading_overlap = sum(1 for token in query_tokens if token in heading_tokens)
    coverage = overlap / len(query_tokens)
    heading_coverage = heading_overlap / len(query_tokens)
    heading_bonus = 0.0
    if heading_coverage >= 0.5:
        heading_bonus += 0.75
    return (coverage * 0.5) + (heading_coverage * 2.5) + heading_bonus


def _document_match_score(context: QueryContext, chunk: Chunk) -> float:
    if not context.normalized_query:
        return 0.0

    score = 0.0
    query_tokens = set(context.query_tokens)
    for alias in _document_aliases(chunk):
        if not alias:
            continue
        alias_tokens = set(_tokenize(alias, DOCUMENT_STOPWORDS))
        if alias_tokens and alias_tokens.issubset(query_tokens):
            score = max(score, 1.0)
        normalized_alias = _normalize_text(alias)
        if normalized_alias and normalized_alias in context.normalized_query:
            score = max(score, 1.0)
        compact_alias = normalized_alias.replace(" ", "")
        if compact_alias and compact_alias in context.compact_query:
            score = max(score, 1.0)
        if alias_tokens and query_tokens:
            overlap_ratio = len(alias_tokens & query_tokens) / len(alias_tokens)
            score = max(score, overlap_ratio)
    return score


def _document_aliases(chunk: Chunk) -> set[str]:
    aliases = {
        chunk.act_title,
        chunk.document_id,
        chunk.source_file,
        Path(chunk.source_path).stem,
    }
    aliases.update(chunk.document_aliases)
    expanded_aliases = set(aliases)
    for alias in list(aliases):
        expanded_aliases.update(_document_alias_variants(alias))
    title_tokens = _tokenize(chunk.act_title, DOCUMENT_STOPWORDS)
    if title_tokens:
        expanded_aliases.add("".join(token[0] for token in title_tokens))
    return {alias for alias in expanded_aliases if alias}


def _document_alias_variants(alias: str) -> set[str]:
    alpha_tokens = [token for token in TOKEN_PATTERN.findall(alias) if token.isalpha()]
    if len(alpha_tokens) < 2:
        return set()
    variants = {alias}
    stripped = alias
    while True:
        candidate = PARENTHETICAL_SUFFIX_PATTERN.sub("", stripped).strip(" -_")
        if candidate == stripped:
            break
        variants.add(candidate)
        stripped = candidate
    compact_without_years = " ".join(
        token
        for token in TOKEN_PATTERN.findall(stripped)
        if not token.isdigit() and not re.fullmatch(r"\d{4}", token)
    )
    if compact_without_years:
        variants.add(compact_without_years)
    return {variant.strip() for variant in variants if variant.strip()}


def _subsection_match_boost(context: QueryContext, chunk: Chunk) -> float:
    if context.subsection_id is None:
        return 0.0
    if chunk.subsection_id and chunk.subsection_id.upper() == context.subsection_id:
        return 0.5
    return 0.0


def _heading_match_score(context: QueryContext, chunk: Chunk) -> float:
    heading_tokens = set(_tokenize(chunk.section_heading, QUERY_STOPWORDS))
    if not heading_tokens or not context.query_tokens:
        return 0.0
    overlap_ratio = len(heading_tokens & set(context.query_tokens)) / len(heading_tokens)
    if any(token in heading_tokens for token in {"interpretation", "definition", "tafsiran"}) and context.is_definition_query:
        overlap_ratio += 0.5
    return overlap_ratio


def _heading_alignment_score(context: QueryContext, chunk: Chunk) -> float:
    intents = _normalized_heading_intents(context)
    if not intents:
        return 0.0

    heading_core = _heading_core_text(chunk.section_heading)
    if not heading_core:
        return 0.0

    heading_tokens = set(_tokenize(heading_core, QUERY_STOPWORDS))
    text = _normalize_text(chunk.text[:500])
    score = 0.0

    for intent in intents:
        synonyms = _intent_synonyms(intent)
        exact_match = any(heading_core == synonym for synonym in synonyms)
        starts_with_match = any(heading_core.startswith(synonym + " ") for synonym in synonyms)
        contains_match = any(synonym in heading_core for synonym in synonyms)
        text_match = any(synonym in text for synonym in synonyms)

        if exact_match:
            score += 3.2
        elif starts_with_match:
            score += 2.2
        elif contains_match:
            score += 1.2
        elif text_match:
            score += 0.4

        if intent == "commencement":
            if "short title and commencement" in heading_core or "citation and commencement" in heading_core:
                score += 1.2
            elif "general amendment" in heading_core:
                score -= 1.0
        if intent == "application":
            if heading_core == "application":
                score += 1.2
            elif "application of" in heading_core:
                score -= 0.4
        if intent == "appeal" and heading_core in {"appeal", "appeals"}:
            score += 0.9
        if intent == "annual leave" and heading_core == "annual leave":
            score += 1.0
        if intent == "revocation" and heading_core == "revocation":
            score += 1.0
        if intent == "interpretation":
            if any(term in heading_core for term in ("interpretation", "tafsiran")):
                score += 0.8
            if context.is_hierarchy_query and context.hierarchy_id is not None:
                hierarchy_id = context.hierarchy_id.lower()
                if hierarchy_id in text or hierarchy_id in heading_core:
                    score += 0.8
        if intent == "introduction":
            if "new section" in heading_core:
                score += 1.2
            if "data portability" in heading_core or "data portability" in text:
                score += 1.0
            if "general amendment" in heading_core:
                score -= 1.1

    return score


def _heading_recall_boost(context: QueryContext, chunk: Chunk) -> float:
    heading_tokens = set(_tokenize(chunk.section_heading, QUERY_STOPWORDS))
    if not heading_tokens or not context.query_tokens:
        return 0.0
    recall_overlap = heading_tokens & set(context.query_tokens) & RECALL_FAMILY_TERMS
    if recall_overlap:
        return 1.0
    return 0.0


def _hierarchy_match_boost(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_hierarchy_query or context.hierarchy_label is None or context.hierarchy_id is None:
        return 0.0
    hierarchy_lines = _extract_hierarchy_lines(chunk.text)
    for label, identifier in hierarchy_lines:
        if label == context.hierarchy_label and identifier == context.hierarchy_id:
            boost = 3.5
            if any(term in context.raw_query.lower() for term in ("begin", "begins", "start", "starts", "open", "opens", "memulakan")):
                boost += 1.0
            return boost
    return 0.0


def _same_document_specificity_score(context: QueryContext, chunk: Chunk) -> float:
    intents = _normalized_heading_intents(context)
    if not intents:
        return 0.0

    heading_core = _heading_core_text(chunk.section_heading)
    heading_tokens = _tokenize(heading_core, QUERY_STOPWORDS)
    if not heading_tokens:
        return 0.0

    token_count = len(heading_tokens)
    comma_count = heading_core.count(",")
    conjunction_count = sum(heading_tokens.count(token) for token in ("and", "or"))
    score = 0.0

    if any(intent in {"appeal", "application", "revocation", "annual leave"} for intent in intents):
        if token_count <= 2:
            score += 0.9
        elif token_count <= 4:
            score += 0.35
        else:
            score -= 0.25
        score -= comma_count * 0.35
        if any(term in heading_core for term in (" and ", " overtime", " holiday", " sick leave", "other written law")):
            score -= 0.6

    if "commencement" in intents:
        if any(term in heading_core for term in ("short title and commencement", "citation and commencement")):
            score += 0.8
        if token_count > 4:
            score -= 0.2

    if "introduction" in intents and "general amendment" in heading_core:
        score -= 0.9

    if context.is_definition_query and context.is_hierarchy_query and context.hierarchy_id is not None:
        hierarchy_id = context.hierarchy_id.lower()
        text = _normalize_text(chunk.text[:500])
        if hierarchy_id in text or hierarchy_id in heading_core:
            score += 0.6

    return score


def _obligation_text_boost(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_obligation_query:
        return 0.0
    matches = len(OBLIGATION_TEXT_PATTERN.findall(chunk.text))
    if matches == 0:
        return 0.0
    return min(1.5, 0.5 + (matches * 0.15))


def _obligation_heading_boost(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_obligation_query:
        return 0.0
    heading_tokens = set(_tokenize(chunk.section_heading, QUERY_STOPWORDS))
    if "obligation" in heading_tokens:
        return 2.5
    raw_heading_tokens = set(TOKEN_PATTERN.findall(chunk.section_heading.lower()))
    if {"duty", "duties"} & raw_heading_tokens:
        return 2.5
    return 0.0


def _obligation_internal_score(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_obligation_query:
        return 0.0
    lowered_heading = chunk.section_heading.lower()
    heading_tokens = set(TOKEN_PATTERN.findall(lowered_heading))
    score = _document_internal_position_bonus(chunk)
    if "general duties" in lowered_heading:
        score += 1.2
    if {"general", "duty", "duties"} <= heading_tokens or {"general", "obligation"} <= heading_tokens:
        score += 0.8
    if {"employer", "employers"} & heading_tokens:
        score += 0.6
    if {"employee", "employees"} & heading_tokens and not {"employer", "employers"} & heading_tokens:
        score -= 0.2
    if {"manufacturer", "supplier", "designer", "occupier", "licensee"} & heading_tokens:
        score -= 0.4
    return score


def _obligation_definition_penalty(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_obligation_query:
        return 0.0
    heading = chunk.section_heading.lower()
    if "interpretation" in heading or "definition" in heading or "tafsiran" in heading:
        return 1.0
    return 0.0


def _candidate_features(context: QueryContext, candidate: RetrievalResult) -> CandidateFeatures:
    chunk = candidate.chunk
    lexical_score = _lexical_similarity(context.query_tokens, chunk)
    heading_match = _heading_match_score(context, chunk)
    document_score = _rerank_document_boost(context, chunk)
    cross_reference_penalty = _cross_reference_density_penalty(chunk)
    definition_bias = _definition_heading_boost(context, chunk)
    exact_unit_boost = _exact_unit_match_boost(context, chunk)
    weak_embedding_match = lexical_score < 0.2 and heading_match <= 0.0
    return CandidateFeatures(
        result=candidate,
        lexical_score=lexical_score,
        heading_match=heading_match,
        document_score=document_score,
        cross_reference_penalty=cross_reference_penalty,
        definition_bias=definition_bias,
        exact_unit_boost=exact_unit_boost,
        weak_embedding_match=weak_embedding_match,
    )


def _sort_candidate_features(
    features: list[CandidateFeatures],
    context: QueryContext,
) -> list[CandidateFeatures]:
    return sorted(
        features,
        key=lambda feature: (
            -_candidate_pre_rank_score(context, feature),
            feature.result.chunk.chunk_index,
            feature.result.chunk.chunk_id,
        ),
    )


def _candidate_pre_rank_score(context: QueryContext, feature: CandidateFeatures) -> float:
    score = (
        (feature.result.score * 0.45)
        + (feature.lexical_score * 2.4)
        + (feature.heading_match * 2.8)
        + (feature.document_score * 3.0)
        + (feature.definition_bias * 1.5)
        + (feature.exact_unit_boost * 1.5)
        - (feature.cross_reference_penalty * 1.5)
        + (_employment_agreement_score(context, feature.result.chunk) * 2.0)
    )
    if context.is_definition_query and feature.definition_bias <= 0.0:
        score -= 1.0
    if feature.weak_embedding_match:
        score -= 0.5
    return score


def _position_bias(chunk: Chunk) -> float:
    unit_key = chunk.unit_id or chunk.section_id
    numeric_part = "".join(character for character in unit_key if character.isdigit())
    if not numeric_part:
        return 0.0
    number = int(numeric_part)
    if number <= 10:
        return 2.0
    if number <= 25:
        return 0.8
    if number <= 50:
        return 0.2
    return 0.0


def _refine_same_document_heading_order(
    context: QueryContext,
    ranked: list[RetrievalResult],
) -> list[RetrievalResult]:
    if not ranked or not _should_apply_same_document_heading_refinement(context):
        return ranked

    focus_chunk = ranked[0].chunk
    focus_indexes = [
        index
        for index, result in enumerate(ranked)
        if _same_document(result.chunk, focus_chunk)
    ]
    if len(focus_indexes) < 2:
        return ranked

    focus_results = [ranked[index] for index in focus_indexes]
    if not any(_heading_alignment_score(context, result.chunk) != 0.0 for result in focus_results):
        return ranked

    refined_focus = sorted(
        focus_results,
        key=lambda result: (
            -(
                (result.score * 0.35)
                + (_heading_alignment_score(context, result.chunk) * 2.5)
                + (_same_document_specificity_score(context, result.chunk) * 1.5)
            ),
            result.chunk.chunk_index,
            result.chunk.chunk_id,
        ),
    )
    refined = list(ranked)
    for index, replacement in zip(focus_indexes, refined_focus, strict=False):
        refined[index] = replacement
    return refined


def _exact_unit_match_boost(context: QueryContext, chunk: Chunk) -> float:
    if context.unit_type is None or context.unit_id is None:
        return 0.0
    if chunk.unit_type.lower() == context.unit_type and (chunk.unit_id or chunk.section_id).upper() == context.unit_id:
        return 1.0
    return 0.0


def _cross_reference_density_penalty(chunk: Chunk) -> float:
    references = len(CROSS_REFERENCE_PATTERN.findall(chunk.text))
    words = max(len(chunk.text.split()), 1)
    density = references / words
    if references <= 1:
        return 0.0
    return min(1.0, density * 40.0)


def _rerank_document_boost(context: QueryContext, chunk: Chunk) -> float:
    score = _document_match_score(context, chunk)
    alpha_title_tokens = [
        token for token in _tokenize(chunk.act_title, DOCUMENT_STOPWORDS) if token.isalpha()
    ]
    if alpha_title_tokens:
        acronym = "".join(token[0] for token in alpha_title_tokens)
        if len(acronym) >= 3 and acronym and acronym in context.compact_query:
            score = max(score, 3.0)
    return score


def _should_apply_rerank(context: QueryContext) -> bool:
    if context.unit_type is not None and context.unit_id is not None:
        return False
    if context.is_hierarchy_query and not _has_strict_definition_intent(context.raw_query):
        return False
    if _has_strict_definition_intent(context.raw_query):
        return True
    lowered = context.raw_query.lower()
    has_document_hint = _query_has_document_hint(lowered)
    if _looks_like_amendment_query(lowered):
        return True
    if _looks_like_heading_lookup_query(lowered) and (
        has_document_hint or _looks_like_gazette_order_query(context)
    ):
        return True
    if any(cue in lowered for cue in RERANK_CUES if cue != "apakah"):
        return True
    if _looks_like_obligation_query(lowered) and has_document_hint:
        return True
    if _looks_like_capability_query(lowered) and (
        has_document_hint or _looks_like_legal_action_query(lowered)
    ):
        return True
    if _looks_like_legal_action_query(lowered) and has_document_hint:
        return True
    return False


def _should_skip_rerank(context: QueryContext, candidates: list[RetrievalResult]) -> bool:
    if len(candidates) < 2:
        return True
    lowered = context.raw_query.lower()
    if not (
        _looks_like_heading_lookup_query(lowered)
        or _looks_like_amendment_query(lowered)
        or context.is_hierarchy_query
        or context.is_obligation_query
    ):
        return False
    if _should_apply_same_document_heading_refinement(context):
        top_chunk = candidates[0].chunk
        same_document_candidates = [
            candidate for candidate in candidates if _same_document(candidate.chunk, top_chunk)
        ]
        if len(same_document_candidates) >= 2:
            return False
    return (candidates[0].score - candidates[1].score) >= RERANK_CONFIDENCE_MARGIN


def _should_apply_same_document_heading_refinement(context: QueryContext) -> bool:
    lowered = context.raw_query.lower()
    if _has_strict_definition_intent(context.raw_query):
        return True
    if _looks_like_amendment_query(lowered):
        return True
    if _looks_like_heading_lookup_query(lowered):
        return True
    return False


def _has_strict_definition_intent(query: str) -> bool:
    lowered = query.lower()
    return any(cue in lowered for cue in STRICT_DEFINITION_CUES)


def _looks_like_definition_query(query: str) -> bool:
    lowered = query.lower()
    if any(cue in lowered for cue in DEFINITION_CUES):
        return True
    if "what is" in lowered and any(cue in lowered for cue in ("meaning", "mean", "definition", "defined")):
        return True
    if "what does" in lowered and any(cue in lowered for cue in ("define", "mean", "meaning")):
        return True
    if "how does" in lowered and "define" in lowered:
        return True
    return False


def _looks_like_obligation_query(lowered_query: str) -> bool:
    if any(cue in lowered_query for cue in OBLIGATION_CUES) and any(
        token in lowered_query for token in ("require", "must", "duty", "obligation")
    ):
        return True
    return any(token in lowered_query for token in (" duty ", " duties ", " obligation ", " obligations ", " must ", " required "))


def _looks_like_employment_agreement_query(lowered_query: str) -> bool:
    has_core_phrase = any(cue in lowered_query for cue in EMPLOYMENT_AGREEMENT_CORE_CUES)
    has_advice_phrase = any(cue in lowered_query for cue in EMPLOYMENT_AGREEMENT_ADVICE_CUES)
    has_token_pattern = (
        "employment" in lowered_query
        and ("agreement" in lowered_query or "contract" in lowered_query)
        and any(token in lowered_query for token in ("check", "sign", "protect", "rights", "terms"))
    )
    return (has_core_phrase and has_advice_phrase) or has_token_pattern


def _looks_like_capability_query(lowered_query: str) -> bool:
    return any(cue in lowered_query for cue in CAPABILITY_CUES)


def _looks_like_legal_action_query(lowered_query: str) -> bool:
    return any(cue in lowered_query for cue in LEGAL_ACTION_CUES) and any(
        token in lowered_query
        for token in ("under", "under the", "pursuant", "right", "entitled", "may", "can", "who")
    )


def _looks_like_heading_lookup_query(lowered_query: str) -> bool:
    return any(cue in lowered_query for cue in HEADING_LOOKUP_CUES)


def _looks_like_amendment_query(lowered_query: str) -> bool:
    return any(cue in lowered_query for cue in AMENDMENT_QUERY_CUES)


def _amendment_rerank_boost(lowered_query: str, chunk: Chunk) -> float:
    lowered_text = f"{chunk.section_heading.lower()} {chunk.text.lower()}"
    lowered_heading = chunk.section_heading.lower()
    score = 0.0
    if "come into force" in lowered_query or "commencement" in lowered_query:
        if any(term in lowered_text for term in ("commencement", "come into operation", "short title and commencement")):
            score += 1.0
    if "introduc" in lowered_query and any(term in lowered_text for term in ("introduc", "new section", "insert")):
        score += 0.7
    if ("introduc" in lowered_query or "data portability" in lowered_query) and "new section" in lowered_heading:
        score += 0.8
    if "data portability" in lowered_query and "data portability" in lowered_text:
        score += 1.1
    section_match = re.search(r"section\s+(\d+[a-z]?)", lowered_query)
    if section_match is not None and f"section {section_match.group(1)}" in lowered_text:
        score += 0.8
    if "principal act" in lowered_query and "principal act" in lowered_text:
        score += 0.5
    return score


def _query_has_document_hint(lowered_query: str) -> bool:
    padded = f" {lowered_query} "
    return any(cue in padded for cue in DOCUMENT_HINT_CUES)


def _normalized_heading_intents(context: QueryContext) -> tuple[str, ...]:
    lowered = context.raw_query.lower()
    normalized = context.normalized_query
    intents: list[str] = []

    if "annual leave" in normalized or "cuti tahunan" in lowered:
        intents.append("annual leave")
    if "revoc" in lowered or "revok" in lowered:
        intents.append("revocation")
    if "come into force" in lowered or "comes into force" in lowered or "comes into operation" in lowered:
        intents.append("commencement")
    if "commencement" in lowered or "dikuatkuasa" in lowered or "dikuatkuasakan" in lowered:
        intents.append("commencement")
    if "appeal" in context.query_tokens:
        intents.append("appeal")
    if "apply" in context.query_tokens and "exclude" not in context.query_tokens:
        intents.append("application")
    if context.is_definition_query:
        intents.append("interpretation")
    if (
        "introduc" in lowered
        or "data portability" in normalized
        or "pemindahan data" in lowered
    ):
        intents.append("introduction")

    seen: set[str] = set()
    ordered: list[str] = []
    for intent in intents:
        if intent in seen:
            continue
        seen.add(intent)
        ordered.append(intent)
    return tuple(ordered)


def _intent_synonyms(intent: str) -> tuple[str, ...]:
    if intent == "appeal":
        return ("appeal", "appeals")
    if intent == "application":
        return ("application", "apply", "applies")
    if intent == "revocation":
        return ("revocation", "revoke", "revoked")
    if intent == "annual leave":
        return ("annual leave",)
    if intent == "commencement":
        return ("commencement", "come into operation", "come into force", "citation and commencement")
    if intent == "interpretation":
        return ("interpretation", "definition", "tafsiran")
    if intent == "introduction":
        return ("new section", "data portability", "introduce", "insert")
    return (intent,)


def _definition_heading_boost(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_definition_query:
        return 0.0
    heading = chunk.section_heading.lower()
    if "interpretation" in heading or "definition" in heading or "tafsiran" in heading:
        return 2.0
    return 0.0


def _definition_mismatch_penalty(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_definition_query:
        return 0.0
    heading = chunk.section_heading.lower()
    if "interpretation" in heading or "definition" in heading or "tafsiran" in heading:
        return 0.0
    return 1.0


def _sort_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    return sorted(
        results,
        key=lambda item: (
            -item.score,
            item.chunk.chunk_index,
            item.chunk.chunk_id,
        ),
    )


def _prefer_referenced_document(
    context: QueryContext,
    ranked: list[RetrievalResult],
    referenced_chunk: Chunk,
) -> list[RetrievalResult]:
    if not ranked:
        return ranked
    if not any(
        result.chunk.document_id == referenced_chunk.document_id
        and result.chunk.source_path == referenced_chunk.source_path
        for result in ranked
    ):
        return ranked
    return sorted(
        ranked,
        key=lambda result: (
            not (
                result.chunk.document_id == referenced_chunk.document_id
                and result.chunk.source_path == referenced_chunk.source_path
            ),
            -(
                result.score
                + (_obligation_internal_score(context, result.chunk) * 1.5)
                + (_document_match_score(context, result.chunk) * 2.0)
            ),
            result.chunk.chunk_index,
            result.chunk.chunk_id,
        ),
    )


def _normalize_text(text: str) -> str:
    tokens = [_normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]
    return " ".join(tokens)


def _heading_core_text(heading: str) -> str:
    core = re.sub(r"^(section|article|perkara)\s+\d+[a-z]?\s*", "", heading.strip(), flags=re.IGNORECASE)
    return _normalize_text(core).strip()


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = [_normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]
    return [token for token in tokens if token not in stopwords]


def _tokenize_query(text: str) -> list[str]:
    normalized_query_text = _normalize_query_surface(text)
    tokens = _tokenize(normalized_query_text, QUERY_STOPWORDS)
    expanded_tokens = list(tokens)
    for token in _expand_query_tokens(normalized_query_text, tokens):
        normalized_token = _normalize_token(token)
        if normalized_token not in expanded_tokens:
            expanded_tokens.append(normalized_token)
    return expanded_tokens


def _normalize_token(token: str) -> str:
    lowered = token.lower()
    return LEXICAL_NORMALIZATION.get(lowered, lowered)


def _expand_query_tokens(query_text: str, tokens: list[str]) -> list[str]:
    lowered_query = _normalize_query_surface(query_text).lower()
    token_set = set(tokens)
    expansions: list[str] = []
    if "appeal" in token_set:
        expansions.extend(["appeal", "right", "appeal"])
    if "apply" in token_set or ("personal" in token_set and "commercial" in token_set):
        expansions.extend(["apply", "scope", "apply"])
    if "define" in token_set or "interpret" in token_set:
        expansions.extend(["define", "interpret"])
    if "require" in token_set:
        expansions.extend(["require"])
    if "obligation" in token_set or any(token in token_set for token in ("must", "required")):
        expansions.extend(["shall", "must", "required", "duty", "duties", "obligation"])
    if {"annual", "leave"} <= token_set:
        expansions.extend(["annual", "leave"])
    if ("commercial" in token_set and ("urus" in token_set or "niaga" in token_set)) or "transactions" in token_set:
        expansions.extend(["commercial", "transactions", "transaction", "apply", "scope"])
    if ("data" in token_set and "pemindahan" in token_set) or "data portability" in lowered_query:
        expansions.extend(["data", "portability", "introduce", "new", "section", "right"])
    if "amendment" in token_set or "principal" in token_set:
        expansions.extend(["amendment", "principal", "act"])
    if "come into force" in lowered_query or "commencement" in token_set:
        expansions.extend(["commencement", "operation"])
    if "order" in token_set or "pua" in token_set or ("minimum" in token_set and "wage" in token_set):
        expansions.extend(["order", "minimum", "wage"])
        if "exclude" in token_set:
            expansions.extend(["non", "application"])
        if "apply" in token_set and any(month in token_set for month in MONTH_TOKENS):
            expansions.extend(["effect", "rate", "wage", "minimum"])
    if _looks_like_employment_agreement_query(lowered_query):
        expansions.extend(
            [
                "contract",
                "service",
                "written",
                "termination",
                "notice",
                "wage",
                "deduction",
                "working",
                "hours",
                "annual",
                "leave",
                "sick",
                "rest",
                "favourable",
                "conditions",
                "trade",
                "union",
                "rights",
            ]
        )
    return expansions


MONTH_TOKENS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _normalize_query_surface(text: str) -> str:
    normalized = PUA_PATTERN.sub(" pua order ", text)
    replacements = {
        "akta kerja": "employment act",
        "akta perlindungan data peribadi": "personal data protection act pdpa",
        "akta pindaan pdpa": "pdpa amendment act act a1727",
        "personal data protection (amendment) act 2024": "personal data protection amendment act act a1727",
        "urus niaga komersial": "commercial transactions",
        "pemindahan data": "data portability",
        "berkuat kuasa": "commencement",
        "come into force": "commencement",
        "with effect from": "effect",
    }
    lowered = normalized.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return lowered


def _ensure_recall_candidates(
    context: QueryContext,
    ranked: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    if len(ranked) <= top_k:
        return ranked
    current = list(ranked[:top_k])
    current_ids = {result.chunk.chunk_id for result in current}
    strong_document_intent = any(_document_match_score(context, result.chunk) >= 1.0 for result in ranked)
    if not strong_document_intent:
        return current
    if context.is_hierarchy_query:
        hierarchy_candidate = next(
            (
                result
                for result in ranked
                if result.chunk.chunk_id not in current_ids
                and _document_match_score(context, result.chunk) >= 1.0
                and _hierarchy_match_boost(context, result.chunk) > 0.0
            ),
            None,
        )
        if hierarchy_candidate is not None:
            current[-1] = hierarchy_candidate
            current = _sort_results(current)
            current_ids = {result.chunk.chunk_id for result in current}
    if context.is_obligation_query:
        obligation_candidate = next(
            (
                result
                for result in ranked
                if result.chunk.chunk_id not in current_ids
                and _document_match_score(context, result.chunk) >= 1.0
                and _obligation_text_boost(context, result.chunk) > 0.0
                and _obligation_definition_penalty(context, result.chunk) <= 0.0
            ),
            None,
        )
        if obligation_candidate is not None:
            current[-1] = obligation_candidate
            current = _sort_results(current)
            current_ids = {result.chunk.chunk_id for result in current}
    heading_candidate = next(
        (
            result
            for result in ranked
            if result.chunk.chunk_id not in current_ids
            and _document_match_score(context, result.chunk) >= 1.0
            and _heading_recall_boost(context, result.chunk) > 0.0
        ),
        None,
    )
    if heading_candidate is None:
        return current
    current[-1] = heading_candidate
    return _sort_results(current)


def _refine_referenced_document_ranking(
    context: QueryContext,
    ranked: list[RetrievalResult],
    referenced_chunk: Chunk,
) -> list[RetrievalResult]:
    refined: list[RetrievalResult] = []
    for result in ranked:
        chunk = result.chunk
        score = result.score
        if _same_document(chunk, referenced_chunk):
            score += 0.75
            score += _referenced_document_query_boost(context, chunk)
        else:
            score -= _out_of_document_penalty(context, chunk, referenced_chunk)
        refined.append(RetrievalResult(chunk=chunk, score=score))
    return _sort_results(refined)


def _same_document(left: Chunk, right: Chunk) -> bool:
    return left.document_id == right.document_id and left.source_path == right.source_path


def _referenced_document_query_boost(context: QueryContext, chunk: Chunk) -> float:
    score = 0.0
    score += _gazette_order_document_boost(context, chunk)
    score += _amendment_document_boost(context, chunk)
    score += _heading_intent_score(context, chunk)
    return score


def _out_of_document_penalty(
    context: QueryContext,
    chunk: Chunk,
    referenced_chunk: Chunk,
) -> float:
    penalty = 0.35 if _query_has_document_hint(context.raw_query.lower()) else 0.0
    if context.unit_type is not None and context.unit_id is not None:
        chunk_unit_id = (chunk.unit_id or chunk.section_id).upper()
        if chunk.unit_type.lower() == context.unit_type and chunk_unit_id == context.unit_id:
            penalty += 2.0
    penalty += _incidental_reference_penalty(chunk, referenced_chunk)
    return penalty


def _incidental_reference_penalty(chunk: Chunk, referenced_chunk: Chunk) -> float:
    alias_hits = 0
    normalized_heading = _normalize_text(chunk.section_heading)
    normalized_text = _normalize_text(chunk.text[:400])
    for alias in _document_aliases(referenced_chunk):
        alias_tokens = _tokenize(alias, DOCUMENT_STOPWORDS)
        if len(alias_tokens) < 2:
            continue
        normalized_alias = _normalize_text(alias)
        if not normalized_alias:
            continue
        if normalized_alias in normalized_heading:
            alias_hits += 1
            continue
        if normalized_alias in normalized_text:
            alias_hits += 1
    if alias_hits == 0:
        return 0.0
    return min(2.5, 1.25 + (alias_hits * 0.4))


def _gazette_order_document_boost(context: QueryContext, chunk: Chunk) -> float:
    if not _looks_like_gazette_order_query(context):
        return 0.0
    lowered_query = context.raw_query.lower()
    lowered_heading = chunk.section_heading.lower()
    lowered_text = chunk.text.lower()
    score = 0.0
    is_exclusion_query = "exclude" in context.query_tokens
    if not is_exclusion_query and any(token in context.query_tokens for token in ("rate", "wage", "minimum")):
        if any(term in lowered_heading for term in ("rate", "wage", "with effect from")):
            score += 1.6
        if "non-application" in lowered_heading:
            score -= 1.8
        month_hits = [month for month in MONTH_TOKENS if month in lowered_query]
        if month_hits and any(month in lowered_heading or month in lowered_text for month in month_hits):
            score += 1.0
    if is_exclusion_query:
        if "non-application" in lowered_heading or "does not apply" in lowered_text:
            score += 1.8
        if any(term in lowered_heading for term in ("rate", "wage", "with effect from")):
            score -= 1.8
        if any(term in lowered_heading for term in ("short title", "commencement", "interpretation")):
            score -= 0.5
    if "revok" in lowered_query:
        if "revocation" in lowered_heading or "revoked" in lowered_text:
            score += 2.4
        if "order" in lowered_heading and "revocation" not in lowered_heading:
            score -= 0.2
    return score


def _amendment_document_boost(context: QueryContext, chunk: Chunk) -> float:
    lowered_query = context.raw_query.lower()
    if not _looks_like_amendment_query(lowered_query):
        return 0.0
    lowered_heading = chunk.section_heading.lower()
    lowered_text = chunk.text.lower()
    score = 0.0
    if "amendment" in lowered_heading or "amendment of section" in lowered_heading:
        score += 0.8
    section_refs = re.findall(r"section\s+(\d+[a-z]?)", lowered_query)
    if section_refs:
        target_section = section_refs[-1].lower()
        amendment_heading_match = re.search(r"amendment of section\s+(\d+[a-z]?)", lowered_heading)
        if amendment_heading_match is not None:
            if amendment_heading_match.group(1) == target_section:
                score += 2.2
            else:
                score -= 1.8
        elif f"section {target_section}" in lowered_text:
            score += 1.0
    if "introduc" in lowered_query or "data portability" in lowered_query:
        if "data portability" in lowered_heading or "data portability" in lowered_text:
            score += 2.2
        if "new section" in lowered_heading or "new section" in lowered_text:
            score += 1.8
        if "amendment of section" in lowered_heading and "data portability" not in lowered_text:
            score -= 1.2
    return score


def _looks_like_gazette_order_query(context: QueryContext) -> bool:
    lowered_query = context.raw_query.lower()
    return (
        "order" in context.query_tokens
        or "pua" in context.query_tokens
        or "p.u." in lowered_query
        or ("minimum" in context.query_tokens and "wage" in context.query_tokens)
    )


def _heading_intent_score(context: QueryContext, chunk: Chunk) -> float:
    lowered_query = context.raw_query.lower()
    normalized_query = context.normalized_query
    lowered_heading = chunk.section_heading.lower()
    lowered_text = chunk.text.lower()
    score = 0.0
    is_exclusion_query = "exclude" in context.query_tokens
    is_rate_query = any(token in context.query_tokens for token in ("rate", "wage", "minimum")) and not is_exclusion_query
    is_citation_query = any(
        phrase in lowered_query
        for phrase in ("citation and commencement", "short title and commencement", "citation", "commencement")
    )
    is_revocation_query = "revok" in lowered_query
    is_introduction_query = (
        "introduc" in lowered_query
        or "data portability" in lowered_query
        or "introduce" in context.query_tokens
        or "portability" in context.query_tokens
        or "data portability" in normalized_query
    )

    if context.is_definition_query:
        if any(term in lowered_heading for term in ("interpretation", "definition", "tafsiran")):
            score += 1.8
        else:
            score -= 0.6
        if context.is_hierarchy_query and context.hierarchy_id is not None:
            hierarchy_token = context.hierarchy_id.lower()
            if hierarchy_token in lowered_heading or hierarchy_token in lowered_text:
                score += 0.8

    if is_exclusion_query:
        if "non-application" in lowered_heading or "does not apply" in lowered_text:
            score += 2.4
        if any(term in lowered_heading for term in ("with effect from", "rate", "wage", "revocation")):
            score -= 1.4

    if is_rate_query:
        if any(term in lowered_heading for term in ("rate", "wage", "with effect from")):
            score += 1.4
        if "non-application" in lowered_heading:
            score -= 1.0

    if is_citation_query:
        if any(term in lowered_heading for term in ("citation and commencement", "short title and commencement")):
            score += 2.8
        elif any(term in lowered_heading for term in ("commencement", "citation", "short title")):
            score += 1.6
        if any(term in lowered_heading for term in ("with effect from", "rate", "wage", "revocation")):
            score -= 1.6

    if is_revocation_query:
        if "revocation" in lowered_heading or "revoked" in lowered_text:
            score += 2.6
        elif any(term in lowered_heading for term in ("with effect from", "rate", "wage")):
            score -= 1.0

    if "appeal" in context.query_tokens:
        if "appeal" in lowered_heading or "appeals" in lowered_heading:
            score += 1.5
        elif "appeal" not in lowered_text:
            score -= 0.5

    if "acceptance" in context.query_tokens:
        if "acceptance" in lowered_heading or "acceptance" in lowered_text:
            score += 2.0
            if "absolute" in lowered_text or "absolute" in lowered_heading:
                score += 0.8
        elif "obligation" in lowered_heading:
            score -= 0.6

    if "apply" in context.query_tokens and "exclude" not in context.query_tokens:
        if "application" in lowered_heading or "applies" in lowered_text:
            score += 1.2
        if "non-application" in lowered_heading:
            score -= 0.6

    if is_introduction_query:
        if "new section" in lowered_heading or "new section" in lowered_text:
            score += 3.4
        if "data portability" in lowered_heading or "data portability" in lowered_text:
            score += 3.0
        if "new division" in lowered_heading or "new division" in lowered_text:
            score += 0.8
        if "amendment of section" in lowered_heading and "data portability" not in lowered_text:
            score -= 2.2

    score += _employment_agreement_score(context, chunk)
    return score


def _employment_agreement_score(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_employment_agreement_query:
        return 0.0

    lowered_heading = chunk.section_heading.lower()
    lowered_text = chunk.text.lower()
    score = 0.0

    if "employment act" in (chunk.act_title or "").lower():
        score += 0.25

    high_value_heading_terms = {
        "contract of service": 2.6,
        "contracts to be in writing": 3.0,
        "termination": 2.0,
        "notice of termination": 2.8,
        "wage": 1.8,
        "deduction": 1.8,
        "hours of work": 1.8,
        "annual leave": 1.9,
        "sick leave": 1.9,
        "rest day": 1.8,
        "more favourable conditions": 2.0,
        "trade union": 1.5,
        "discrimination": 1.2,
        "flexible working arrangement": 1.3,
    }
    for term, weight in high_value_heading_terms.items():
        if term in lowered_heading:
            score += weight
        elif term in lowered_text:
            score += weight * 0.35

    if "foreign" not in context.query_tokens:
        for term in EMPLOYMENT_AGREEMENT_NARROW_PENALTY_TERMS:
            if term in lowered_heading:
                score -= 2.4 if "foreign" in term else 1.0
            elif term in lowered_text:
                score -= 0.8 if "foreign" in term else 0.3

    if lowered_heading.strip() in {"section 9 employment", "section 95 employment"}:
        score -= 0.9

    return score


def _extract_hierarchy_lines(text: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        cleaned = raw_line.strip()
        match = HIERARCHY_LINE_PATTERN.match(cleaned)
        if match is None:
            continue
        lines.append((match.group(1).lower(), match.group(2).upper()))
    return lines


def _chunk_contains_hierarchy(context: QueryContext, chunk: Chunk) -> bool:
    if context.hierarchy_label is None or context.hierarchy_id is None:
        return False
    return any(
        label == context.hierarchy_label and identifier == context.hierarchy_id
        for label, identifier in _extract_hierarchy_lines(chunk.text)
    )


def _hierarchy_scope_chunks(ordered_chunks: list[Chunk], anchor_index: int) -> list[Chunk]:
    scope: list[Chunk] = []
    seen_units: set[str] = set()
    for candidate in ordered_chunks[anchor_index + 1 :]:
        if _extract_hierarchy_lines(candidate.text):
            break
        unit_key = candidate.unit_id or candidate.section_id or candidate.chunk_id
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)
        scope.append(candidate)
    return scope


def _hierarchy_definition_scope_score(context: QueryContext, chunk: Chunk) -> float:
    if not context.is_definition_query or not context.is_hierarchy_query:
        return 0.0
    lowered_heading = chunk.section_heading.lower()
    lowered_text = chunk.text.lower()
    score = 0.0

    if any(term in lowered_heading for term in ("interpretation", "definition", "tafsiran")):
        score += 2.5
    elif any(term in lowered_text for term in ("interpretation", "definition", "tafsiran")):
        score += 0.6

    hierarchy_id = (context.hierarchy_id or "").lower()
    hierarchy_label = (context.hierarchy_label or "").lower()
    if hierarchy_id and hierarchy_label:
        hierarchy_phrase = f"{hierarchy_label} {hierarchy_id}"
        if hierarchy_phrase in lowered_heading:
            score += 1.6
        elif hierarchy_phrase in lowered_text[:600]:
            score += 0.9

    if any(term in lowered_heading for term in ("part", "bahagian", "chapter", "bab")):
        score += 0.4
    if any(term in lowered_heading for term in ("interpretation of", "tafsiran")):
        score += 0.6
    return score


def _chunk_starts_with_unit_heading(chunk: Chunk) -> bool:
    heading = chunk.section_heading.strip()
    return bool(heading) and chunk.text.lstrip().lower().startswith(heading.lower())


def _document_internal_position_bonus(chunk: Chunk) -> float:
    if chunk.chunk_index <= 25:
        return 1.0
    if chunk.chunk_index <= 75:
        return 0.5
    if chunk.chunk_index <= 150:
        return 0.15
    return 0.0


def _hierarchy_marker_reliability(chunk: Chunk) -> float:
    reliability = _document_internal_position_bonus(chunk)
    reliability -= _cross_reference_density_penalty(chunk) * 2.0
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    tail_lines = lines[-4:]
    if any(HIERARCHY_LINE_PATTERN.match(line) for line in tail_lines):
        reliability += 0.75
    return reliability


def _is_compact_hierarchy_tail(chunk: Chunk) -> bool:
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    if len(lines) > 6:
        return False
    return any(HIERARCHY_LINE_PATTERN.match(line) for line in lines[-3:])


def _next_unit_chunk(ordered_chunks: list[Chunk], current_index: int, current_chunk: Chunk) -> Chunk | None:
    current_unit = current_chunk.unit_id or current_chunk.section_id
    for candidate in ordered_chunks[current_index + 1 :]:
        candidate_unit = candidate.unit_id or candidate.section_id
        if candidate_unit != current_unit:
            return candidate
    return None


def _store_best_result(results: dict[str, RetrievalResult], candidate: RetrievalResult) -> None:
    existing = results.get(candidate.chunk.chunk_id)
    if existing is None or candidate.score > existing.score:
        results[candidate.chunk.chunk_id] = candidate


def _is_impossible_unit_lookup(entries: list[EmbeddedChunk], context: QueryContext) -> bool:
    if context.unit_type is None or context.unit_id is None:
        return False
    requested_number = _numeric_unit_value(context.unit_id)
    if requested_number is None:
        return False
    referenced_chunk = _infer_referenced_chunk(entries, context)
    if referenced_chunk is None:
        return False
    max_unit_number = None
    for entry in entries:
        chunk = entry.chunk
        if chunk.document_id != referenced_chunk.document_id or chunk.source_path != referenced_chunk.source_path:
            continue
        if chunk.unit_type.lower() != context.unit_type:
            continue
        candidate_number = _numeric_unit_value(chunk.unit_id or chunk.section_id)
        if candidate_number is None:
            continue
        max_unit_number = candidate_number if max_unit_number is None else max(max_unit_number, candidate_number)
    if max_unit_number is None:
        return False
    return requested_number > max_unit_number


def _infer_referenced_document(entries: list[EmbeddedChunk], context: QueryContext) -> str | None:
    referenced_chunk = _infer_referenced_chunk(entries, context)
    if referenced_chunk is None:
        return None
    return referenced_chunk.act_title


def _infer_referenced_chunk(entries: list[EmbeddedChunk], context: QueryContext) -> Chunk | None:
    best_chunk = None
    best_score = 0.0
    seen: dict[tuple[str, str], Chunk] = {}
    for entry in entries:
        key = (entry.chunk.document_id, entry.chunk.source_path)
        if key not in seen:
            seen[key] = entry.chunk
    for chunk in seen.values():
        score = _document_match_score(context, chunk) + _document_reference_tiebreak_bonus(context, chunk)
        if score > best_score:
            best_score = score
            best_chunk = chunk
    if best_score >= 1.0:
        return best_chunk
    return None


def _should_use_exact_unit_lookup(context: QueryContext) -> bool:
    if context.unit_type is None or context.unit_id is None:
        return False
    lowered_query = context.raw_query.lower()
    if _looks_like_amendment_query(lowered_query) and any(
        cue in lowered_query
        for cue in ("which section", "which article", "seksyen manakah", "perkara manakah")
    ):
        return False
    return True


def _document_reference_tiebreak_bonus(context: QueryContext, chunk: Chunk) -> float:
    lowered_query = context.raw_query.lower()
    title = chunk.act_title.lower()
    score = 0.0
    aliases = " ".join(alias.lower() for alias in _document_aliases(chunk))
    if any(term in lowered_query for term in ("amendment", "pindaan", "act a")):
        if "amendment" in title or "pindaan" in title or re.search(r"\bact\s+a\d+\b", aliases):
            score += 0.6
    elif _should_use_exact_unit_lookup(context):
        if "amendment" in title or "pindaan" in title:
            score -= 1.2
        else:
            score += 0.2
    return score


def _numeric_unit_value(unit_id: str | None) -> int | None:
    if not unit_id:
        return None
    digits = "".join(character for character in unit_id if character.isdigit())
    if not digits:
        return None
    return int(digits)
