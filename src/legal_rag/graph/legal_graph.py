"""Deterministic legal structure graph for hierarchy, amendment, and reference queries."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from legal_rag.chunking.models import Chunk
UNIT_REF_PATTERN = re.compile(r"\b(section|article|perkara)\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
HIERARCHY_PATTERN = re.compile(
    r"^(?P<label>Part|PART|Chapter|CHAPTER|Division|DIVISION|Bahagian|BAHAGIAN|Bab|BAB|Schedule|SCHEDULE|Jadual|JADUAL)\s+(?P<id>[A-Za-z0-9IVXLCDM]+)\s*[:.\-]?\s*(?P<title>.*)$"
)
AMENDMENT_OF_PATTERN = re.compile(r"\bamendment of (section|article|perkara)\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
NEW_SECTION_PATTERN = re.compile(r"\bnew (section|article|perkara)\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
INSERT_AFTER_PATTERN = re.compile(
    r"\binserting after (section|article|perkara)\s+(\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)
COMMENCEMENT_PATTERN = re.compile(r"\b(commencement|comes into operation|mula berkuat kuasa)\b", re.IGNORECASE)
AMENDMENT_QUERY_PATTERN = re.compile(r"\b(amendment|amends|amend|principal act|pindaan|new section|introduc)\b", re.IGNORECASE)
REFERENCE_QUERY_PATTERN = re.compile(
    r"\b(refer|refers|referred|reference|cross-reference|under section|under article|under perkara)\b",
    re.IGNORECASE,
)
HIERARCHY_QUERY_PATTERN = re.compile(r"\b(part|chapter|division|bahagian|bab|schedule|jadual)\s+([A-Za-z0-9IVXLCDM]+)\b", re.IGNORECASE)
QUERY_TOPIC_STOPWORDS = {
    "which",
    "what",
    "does",
    "say",
    "about",
    "of",
    "the",
    "in",
    "under",
    "begins",
    "begin",
    "starts",
    "start",
    "covers",
    "cover",
    "deals",
    "deal",
    "with",
    "section",
    "article",
    "perkara",
    "part",
    "chapter",
    "division",
    "bahagian",
    "bab",
    "schedule",
    "jadual",
}


@dataclass(frozen=True)
class HierarchyNode:
    """One hierarchy marker, such as Part II or Bahagian II, inside a document."""

    node_id: str
    document_id: str
    label: str
    identifier: str
    title: str
    chunk_index: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnitNode:
    """One representative top-level legal unit."""

    node_id: str
    document_id: str
    unit_type: str
    unit_id: str
    chunk: Chunk


@dataclass(frozen=True)
class GraphRetrievalResult:
    """Explainable graph result paired to a source chunk."""

    chunk: Chunk
    score: float
    reason: str


@dataclass
class LegalGraph:
    """Compact in-memory legal graph built from processed legal chunks."""

    documents: dict[str, str] = field(default_factory=dict)
    document_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    units_by_document: dict[str, list[UnitNode]] = field(default_factory=lambda: defaultdict(list))
    unit_lookup: dict[tuple[str, str, str], UnitNode] = field(default_factory=dict)
    hierarchy_nodes: dict[str, HierarchyNode] = field(default_factory=dict)
    hierarchy_contains: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    next_unit: dict[str, str] = field(default_factory=dict)
    refers_to: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_refers_to: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    amends: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    inserts: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    commences: set[str] = field(default_factory=set)


def build_legal_graph(chunks: list[Chunk]) -> LegalGraph:
    """Build a conservative legal structure graph from chunk metadata and text."""

    graph = LegalGraph()
    representative_units: dict[tuple[str, str, str], Chunk] = {}
    for chunk in sorted(chunks, key=lambda item: (item.document_id, item.chunk_index, item.chunk_id)):
        unit_type = (chunk.unit_type or "section").lower()
        unit_id = (chunk.unit_id or chunk.section_id).upper()
        key = (chunk.document_id, unit_type, unit_id)
        existing = representative_units.get(key)
        if existing is None or chunk.chunk_index < existing.chunk_index:
            representative_units[key] = chunk

    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in representative_units.values():
        chunks_by_document[chunk.document_id].append(chunk)

    principal_doc_lookup = _build_principal_doc_lookup(representative_units.values())

    for document_id, document_chunks in chunks_by_document.items():
        ordered_chunks = sorted(document_chunks, key=lambda item: (item.chunk_index, item.chunk_id))
        graph.documents[document_id] = ordered_chunks[0].act_title
        graph.document_aliases[document_id] = _document_aliases(ordered_chunks[0])

        current_hierarchy: dict[str, str] = {}
        unit_nodes: list[UnitNode] = []

        for chunk in ordered_chunks:
            unit_type = (chunk.unit_type or "section").lower()
            unit_id = (chunk.unit_id or chunk.section_id).upper()
            node_id = _unit_node_id(document_id, unit_type, unit_id)
            unit_node = UnitNode(node_id=node_id, document_id=document_id, unit_type=unit_type, unit_id=unit_id, chunk=chunk)
            unit_nodes.append(unit_node)
            graph.unit_lookup[(document_id, unit_type, unit_id)] = unit_node

            pre_unit_hierarchies, post_unit_hierarchies = _extract_hierarchy_nodes(chunk)
            for hierarchy in pre_unit_hierarchies:
                graph.hierarchy_nodes[hierarchy.node_id] = hierarchy
                current_hierarchy[hierarchy.label] = hierarchy.node_id

            for hierarchy_node_id in current_hierarchy.values():
                graph.hierarchy_contains[hierarchy_node_id].append(node_id)

            for hierarchy in post_unit_hierarchies:
                graph.hierarchy_nodes[hierarchy.node_id] = hierarchy
                current_hierarchy[hierarchy.label] = hierarchy.node_id

            _attach_reference_edges(graph, unit_node, principal_doc_lookup)

        graph.units_by_document[document_id] = unit_nodes
        for current, following in zip(unit_nodes, unit_nodes[1:], strict=False):
            graph.next_unit[current.node_id] = following.node_id

    return graph


def search_graph(graph: LegalGraph, query: str, top_k: int = 3) -> list[GraphRetrievalResult]:
    """Return explainable graph hits for supported legal-structure queries."""

    if not query.strip():
        return []

    supported_query = _classify_graph_query(query)
    if supported_query == "hierarchy":
        return _search_hierarchy(graph, query, top_k=top_k)
    if supported_query == "amendment":
        return _search_amendment(graph, query, top_k=top_k)
    if supported_query == "reference":
        return _search_cross_reference(graph, query, top_k=top_k)
    return []


def _search_hierarchy(graph: LegalGraph, query: str, top_k: int) -> list[GraphRetrievalResult]:
    match = HIERARCHY_QUERY_PATTERN.search(query)
    if match is None:
        return []
    requested_label = match.group(1).lower()
    requested_id = match.group(2).upper()
    document_id = _infer_document_id(graph, query)

    ranked_hierarchies: list[tuple[float, HierarchyNode]] = []
    for hierarchy in graph.hierarchy_nodes.values():
        if document_id is not None and hierarchy.document_id != document_id:
            continue
        score = 0.0
        if hierarchy.label == requested_label:
            score += 4.0
        if hierarchy.identifier == requested_id:
            score += 4.0
        score += _token_overlap_score(_hierarchy_query_topic_tokens(query), _tokenize(hierarchy.title)) * 2.0
        if score > 0.0:
            ranked_hierarchies.append((score, hierarchy))

    ranked_hierarchies.sort(key=lambda item: (-item[0], item[1].chunk_index, item[1].node_id))
    results: list[GraphRetrievalResult] = []
    for score, hierarchy in ranked_hierarchies:
        first_unit_node = _first_unit_for_hierarchy(graph, hierarchy.node_id)
        if first_unit_node is None:
            continue
        results.append(
            GraphRetrievalResult(
                chunk=first_unit_node.chunk,
                score=10.0 + score,
                reason=f"graph: hierarchy {hierarchy.label} {hierarchy.identifier}",
            )
        )
        if len(results) >= top_k:
            break
    return results


def _search_amendment(graph: LegalGraph, query: str, top_k: int) -> list[GraphRetrievalResult]:
    document_id = _infer_document_id(graph, query)
    referenced_units = [(label.lower(), unit_id.upper()) for label, unit_id in UNIT_REF_PATTERN.findall(query)]
    query_tokens = _tokenize(query)
    lowered_query = query.lower()
    is_commencement_query = any(
        phrase in lowered_query
        for phrase in ("come into force", "comes into force", "into force", "commencement", "operation")
    )
    is_general_amendment_query = "general amendment" in lowered_query or (
        "principal act" in lowered_query and not referenced_units and "introduc" not in lowered_query
    )
    results: list[GraphRetrievalResult] = []

    for document_units in graph.units_by_document.values():
        for unit_node in document_units:
            if document_id is not None and unit_node.document_id != document_id:
                continue
            score = 0.0
            heading_tokens = _tokenize(unit_node.chunk.section_heading)
            if unit_node.node_id in graph.amends or unit_node.node_id in graph.inserts:
                score += 3.0
            for target_type, target_id in referenced_units:
                matched_target = False
                for edge_target in graph.amends.get(unit_node.node_id, set()):
                    if edge_target.endswith(f":{target_type}:{target_id}"):
                        score += 6.0
                        matched_target = True
                for edge_target in graph.inserts.get(unit_node.node_id, set()):
                    if edge_target.endswith(f":{target_type}:{target_id}"):
                        score += 6.5
                        matched_target = True
                if not matched_target and target_id in heading_tokens:
                    score += 0.5
            score += _token_overlap_score(query_tokens, heading_tokens)
            if "new" in query_tokens and "new" in _tokenize(unit_node.chunk.section_heading):
                score += 2.0
            if is_commencement_query and unit_node.node_id in graph.commences:
                score += 4.0
            if is_commencement_query and "commencement" in heading_tokens:
                score += 3.0
            if is_general_amendment_query and "general" in heading_tokens and "amendment" in heading_tokens:
                score += 7.0
            if score <= 0.0:
                continue
            results.append(
                GraphRetrievalResult(
                    chunk=unit_node.chunk,
                    score=10.0 + score,
                    reason="graph: amendment linkage",
                )
            )

    return _sort_graph_results(results)[:top_k]


def _search_cross_reference(graph: LegalGraph, query: str, top_k: int) -> list[GraphRetrievalResult]:
    document_id = _infer_document_id(graph, query)
    unit_matches = [(label.lower(), unit_id.upper()) for label, unit_id in UNIT_REF_PATTERN.findall(query)]
    if not unit_matches:
        return []

    results: list[GraphRetrievalResult] = []
    for target_type, target_id in unit_matches:
        if document_id is not None:
            target_node = graph.unit_lookup.get((document_id, target_type, target_id))
            if target_node is None:
                continue
            referring = graph.reverse_refers_to.get(target_node.node_id, set())
        else:
            referring = {
                source_node_id
                for (doc_id, unit_type, unit_id), target_node in graph.unit_lookup.items()
                if unit_type == target_type and unit_id == target_id
                for source_node_id in graph.reverse_refers_to.get(target_node.node_id, set())
            }
        for source_node_id in referring:
            source = _unit_node_by_id(graph, source_node_id)
            if source is None:
                continue
            results.append(
                GraphRetrievalResult(
                    chunk=source.chunk,
                    score=8.0 + _token_overlap_score(_tokenize(query), _tokenize(source.chunk.section_heading)),
                    reason=f"graph: explicit reference to {target_type} {target_id}",
                )
            )
    return _sort_graph_results(results)[:top_k]


def _classify_graph_query(query: str) -> str | None:
    lowered = query.lower()
    if HIERARCHY_QUERY_PATTERN.search(lowered):
        return "hierarchy"
    if AMENDMENT_QUERY_PATTERN.search(lowered):
        return "amendment"
    if REFERENCE_QUERY_PATTERN.search(lowered):
        return "reference"
    return None


def _extract_hierarchy_nodes(chunk: Chunk) -> tuple[list[HierarchyNode], list[HierarchyNode]]:
    pre_unit_nodes: list[HierarchyNode] = []
    post_unit_nodes: list[HierarchyNode] = []
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        match = HIERARCHY_PATTERN.match(line)
        if match is None:
            continue
        title = match.group("title").strip()
        if not title and index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            if not UNIT_REF_PATTERN.match(next_line):
                title = next_line
        label = match.group("label").lower()
        identifier = match.group("id").upper()
        node_id = _hierarchy_node_id(chunk.document_id, label, identifier)
        hierarchy_node = HierarchyNode(
            node_id=node_id,
            document_id=chunk.document_id,
            label=label,
            identifier=identifier,
            title=title,
            chunk_index=chunk.chunk_index,
            aliases=_document_aliases(chunk),
        )
        if index == 0:
            pre_unit_nodes.append(hierarchy_node)
        else:
            post_unit_nodes.append(hierarchy_node)
    return pre_unit_nodes, post_unit_nodes


def _attach_reference_edges(
    graph: LegalGraph,
    unit_node: UnitNode,
    principal_doc_lookup: dict[str, str],
) -> None:
    chunk = unit_node.chunk
    text = f"{chunk.section_heading}\n{chunk.text}"

    for ref_type, ref_id in UNIT_REF_PATTERN.findall(text):
        normalized_type = ref_type.lower()
        normalized_id = ref_id.upper()
        if normalized_type == unit_node.unit_type and normalized_id == unit_node.unit_id:
            continue
        target = graph.unit_lookup.get((unit_node.document_id, normalized_type, normalized_id))
        if target is not None:
            graph.refers_to[unit_node.node_id].add(target.node_id)
            graph.reverse_refers_to[target.node_id].add(unit_node.node_id)

    amendment_match = AMENDMENT_OF_PATTERN.search(chunk.section_heading)
    if amendment_match is not None:
        target_type = amendment_match.group(1).lower()
        target_id = amendment_match.group(2).upper()
        target_document_id = principal_doc_lookup.get(unit_node.document_id, unit_node.document_id)
        graph.amends[unit_node.node_id].add(_unit_node_id(target_document_id, target_type, target_id))

    new_section_match = NEW_SECTION_PATTERN.search(chunk.section_heading)
    if new_section_match is not None:
        target_type = new_section_match.group(1).lower()
        target_id = new_section_match.group(2).upper()
        target_document_id = principal_doc_lookup.get(unit_node.document_id, unit_node.document_id)
        graph.inserts[unit_node.node_id].add(_unit_node_id(target_document_id, target_type, target_id))

    after_match = INSERT_AFTER_PATTERN.search(chunk.text)
    if after_match is not None:
        target_type = after_match.group(1).lower()
        target_id = after_match.group(2).upper()
        target_document_id = principal_doc_lookup.get(unit_node.document_id, unit_node.document_id)
        graph.refers_to[unit_node.node_id].add(_unit_node_id(target_document_id, target_type, target_id))

    if COMMENCEMENT_PATTERN.search(text):
        graph.commences.add(unit_node.node_id)


def _build_principal_doc_lookup(chunks: list[Chunk]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    by_document = {chunk.document_id: chunk for chunk in chunks}
    for document_id, chunk in by_document.items():
        title = chunk.act_title.lower()
        if "amendment" not in title and "pindaan" not in title:
            continue
        principal_title = re.sub(r"\s*\((amendment|pindaan)\)\s*", " ", chunk.act_title, flags=re.IGNORECASE)
        principal_title = re.sub(r"\s+Act\s+A\d+\b", "", principal_title, flags=re.IGNORECASE).strip()
        best_document_id = None
        best_score = 0
        principal_tokens = set(_tokenize(principal_title))
        for candidate_id, candidate_chunk in by_document.items():
            if candidate_id == document_id:
                continue
            candidate_title = candidate_chunk.act_title.lower()
            candidate_tokens = set(_tokenize(candidate_title))
            overlap = len(principal_tokens & candidate_tokens)
            if overlap > best_score:
                best_score = overlap
                best_document_id = candidate_id
        if best_document_id is not None and best_score >= 2:
            lookup[document_id] = best_document_id
    return lookup


def _infer_document_id(graph: LegalGraph, query: str) -> str | None:
    lowered = query.lower()
    ranked: list[tuple[int, str]] = []
    for document_id, aliases in graph.document_aliases.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower and alias_lower in lowered:
                ranked.append((len(alias_lower), document_id))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def _first_unit_for_hierarchy(graph: LegalGraph, hierarchy_node_id: str) -> UnitNode | None:
    candidates = graph.hierarchy_contains.get(hierarchy_node_id, [])
    if not candidates:
        return None
    first_unit_id = min(
        candidates,
        key=lambda node_id: (_unit_node_by_id(graph, node_id).chunk.chunk_index if _unit_node_by_id(graph, node_id) else 10**9, node_id),
    )
    return _unit_node_by_id(graph, first_unit_id)


def _unit_node_by_id(graph: LegalGraph, node_id: str) -> UnitNode | None:
    for unit_nodes in graph.units_by_document.values():
        for unit_node in unit_nodes:
            if unit_node.node_id == node_id:
                return unit_node
    return None


def _hierarchy_query_topic_tokens(query: str) -> list[str]:
    tokens = []
    for token in _tokenize(query):
        if token in QUERY_TOPIC_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _document_aliases(chunk: Chunk) -> tuple[str, ...]:
    aliases = [chunk.act_title, *chunk.document_aliases]
    deduped: list[str] = []
    for alias in aliases:
        cleaned = alias.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _token_overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / max(len(left_set), 1)


def _sort_graph_results(results: list[GraphRetrievalResult]) -> list[GraphRetrievalResult]:
    best_by_chunk: dict[str, GraphRetrievalResult] = {}
    for result in results:
        existing = best_by_chunk.get(result.chunk.chunk_id)
        if existing is None or result.score > existing.score:
            best_by_chunk[result.chunk.chunk_id] = result
    return sorted(
        best_by_chunk.values(),
        key=lambda result: (-result.score, result.chunk.chunk_index, result.chunk.chunk_id),
    )


def _unit_node_id(document_id: str, unit_type: str, unit_id: str) -> str:
    return f"{document_id}:{unit_type}:{unit_id}"


def _hierarchy_node_id(document_id: str, label: str, identifier: str) -> str:
    return f"{document_id}:hierarchy:{label}:{identifier}"
