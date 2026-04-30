"""Lightweight retrieval evaluation harness for legal section-level targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from legal_rag.embeddings.embedder import OllamaEmbedder
from legal_rag.retrieval.in_memory import SearchMode
from legal_rag.retrieval.vector_store import JsonlVectorStore

MISS_CATEGORIES = (
    "wrong_act",
    "wrong_section",
    "right_section_wrong_subsection",
    "no_hit_in_top_k",
)


@dataclass(frozen=True)
class GoldQuery:
    """Expected section-level retrieval target for one evaluation query."""

    query: str
    expected_act_title: str
    expected_section_id: str
    expected_subsection_id: str | None = None
    query_type: str = "general"


@dataclass(frozen=True)
class GoldMatch:
    """Per-result match information against a gold query."""

    rank: int
    chunk_id: str
    act_title: str
    section_id: str
    subsection_id: str | None
    score: float
    act_match: bool
    section_match: bool
    subsection_match: bool
    target_match: bool


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """Detailed evaluation output for one gold query."""

    query: str
    query_type: str
    expected_act_title: str
    expected_section_id: str
    expected_subsection_id: str | None
    hit_at_1: bool
    hit_at_3: bool
    miss_category: str | None
    matches: list[GoldMatch]


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """Aggregate retrieval evaluation metrics and detailed cases."""

    mode: str
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    wrong_section_rate: float
    error_breakdown: dict[str, int]
    cases: list[RetrievalEvaluationCase]


def load_gold_queries(jsonl_path: Path) -> list[GoldQuery]:
    """Load gold retrieval queries from JSONL."""

    path = Path(jsonl_path)
    queries: list[GoldQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            queries.append(
                GoldQuery(
                    query=payload["query"],
                    expected_act_title=payload["expected_act_title"],
                    expected_section_id=str(payload["expected_section_id"]),
                    expected_subsection_id=payload.get("expected_subsection_id"),
                    query_type=payload.get("query_type", "general"),
                )
            )
    return queries


def evaluate_retrieval(
    gold_queries: list[GoldQuery],
    vector_store: JsonlVectorStore,
    embedder: OllamaEmbedder,
    top_k: int = 3,
    mode: SearchMode = "hybrid",
) -> RetrievalEvaluationSummary:
    """Evaluate retrieval quality against section-level gold targets."""

    cases: list[RetrievalEvaluationCase] = []
    hit_at_1_count = 0
    hit_at_3_count = 0
    error_breakdown = {category: 0 for category in MISS_CATEGORIES}

    for gold_query in gold_queries:
        results = vector_store.search(gold_query.query, embedder, top_k=top_k, mode=mode)
        matches = [
            _build_match(gold_query=gold_query, result=result, rank=index + 1)
            for index, result in enumerate(results)
        ]
        hit_at_1 = bool(matches and matches[0].target_match)
        hit_at_3 = any(match.target_match for match in matches[:3])
        miss_category = None if hit_at_3 else _classify_miss(matches)
        hit_at_1_count += int(hit_at_1)
        hit_at_3_count += int(hit_at_3)
        if miss_category is not None:
            error_breakdown[miss_category] += 1
        cases.append(
            RetrievalEvaluationCase(
                query=gold_query.query,
                query_type=gold_query.query_type,
                expected_act_title=gold_query.expected_act_title,
                expected_section_id=gold_query.expected_section_id,
                expected_subsection_id=gold_query.expected_subsection_id,
                hit_at_1=hit_at_1,
                hit_at_3=hit_at_3,
                miss_category=miss_category,
                matches=matches,
            )
        )

    total_queries = len(gold_queries)
    if total_queries == 0:
        return RetrievalEvaluationSummary(
            mode=mode,
            total_queries=0,
            hit_at_1=0.0,
            hit_at_3=0.0,
            wrong_section_rate=0.0,
            error_breakdown=error_breakdown,
            cases=[],
        )

    wrong_section_count = error_breakdown["wrong_section"]
    return RetrievalEvaluationSummary(
        mode=mode,
        total_queries=total_queries,
        hit_at_1=hit_at_1_count / total_queries,
        hit_at_3=hit_at_3_count / total_queries,
        wrong_section_rate=wrong_section_count / total_queries,
        error_breakdown=error_breakdown,
        cases=cases,
    )


def write_evaluation_summary(summary: RetrievalEvaluationSummary, output_path: Path) -> None:
    """Write evaluation summary and detailed cases to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": summary.mode,
        "total_queries": summary.total_queries,
        "hit_at_1": summary.hit_at_1,
        "hit_at_3": summary.hit_at_3,
        "wrong_section_rate": summary.wrong_section_rate,
        "error_breakdown": summary.error_breakdown,
        "cases": [
            {
                **asdict(case),
                "matches": [asdict(match) for match in case.matches],
            }
            for case in summary.cases
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _classify_miss(matches: list[GoldMatch]) -> str:
    if not matches:
        return "no_hit_in_top_k"
    if any(match.act_match and match.section_match and not match.subsection_match for match in matches):
        return "right_section_wrong_subsection"
    if any(match.act_match and not match.section_match for match in matches):
        return "wrong_section"
    if all(not match.act_match for match in matches):
        return "wrong_act"
    return "no_hit_in_top_k"


def _build_match(gold_query: GoldQuery, result, rank: int) -> GoldMatch:
    chunk = result.chunk
    act_match = chunk.act_title == gold_query.expected_act_title
    section_match = chunk.section_id == gold_query.expected_section_id
    if gold_query.expected_subsection_id is None:
        subsection_match = True
    else:
        subsection_match = chunk.subsection_id == gold_query.expected_subsection_id

    return GoldMatch(
        rank=rank,
        chunk_id=chunk.chunk_id,
        act_title=chunk.act_title,
        section_id=chunk.section_id,
        subsection_id=chunk.subsection_id,
        score=result.score,
        act_match=act_match,
        section_match=section_match,
        subsection_match=subsection_match,
        target_match=act_match and section_match and subsection_match,
    )
