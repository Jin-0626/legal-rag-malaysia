"""Full-corpus rebuild and validation helpers for processed legal chunks."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_rag.chunking.models import Chunk
from legal_rag.chunking.section_chunker import find_section_boundary_leaks
from legal_rag.config.settings import LegalRAGSettings
from legal_rag.graph import build_legal_graph
from legal_rag.ingestion.chunk_export import export_chunks_to_jsonl, ingest_law_pdf_to_chunks
from legal_rag.retrieval.vector_store import load_chunk_records

TITLE_NOISE_PATTERN = re.compile(
    r"(as at\s+\d{1,2}\s+\w+\s+\d{4}|online version|text of reprint|reprint|published by|warta kerajaan|gazette)",
    re.IGNORECASE,
)
FILENAMEISH_TITLE_PATTERN = re.compile(r"[_]|^\d{8}\b|[A-Z]{6,}_[A-Z0-9_]+")
ALIAS_NOISE_PATTERN = re.compile(r"[_]|^\d{8}\b|mys\d+|f\d{6,}", re.IGNORECASE)


def snapshot_processed_corpus(processed_dir: Path) -> dict[str, Any]:
    """Summarize the currently exported processed corpus."""

    processed_path = Path(processed_dir)
    jsonl_paths = sorted(processed_path.glob("*.jsonl"))
    documents: list[dict[str, Any]] = []
    total_chunks = 0
    unit_type_counter: Counter[str] = Counter()

    for jsonl_path in jsonl_paths:
        chunks = load_chunk_records(jsonl_path)
        unit_types = sorted({chunk.unit_type for chunk in chunks if chunk.unit_type})
        act_title = chunks[0].act_title if chunks else ""
        documents.append(
            {
                "file_name": jsonl_path.name,
                "document_id": chunks[0].document_id if chunks else jsonl_path.stem,
                "act_title": act_title,
                "chunk_count": len(chunks),
                "unit_types": unit_types,
            }
        )
        total_chunks += len(chunks)
        unit_type_counter.update(unit_types or ["unknown"])

    other_files = sorted(
        path.name
        for path in processed_path.iterdir()
        if path.is_file() and path.suffix.lower() != ".jsonl"
    )

    return {
        "document_count": len(documents),
        "total_chunks": total_chunks,
        "documents": documents,
        "unit_type_distribution": dict(sorted(unit_type_counter.items())),
        "other_files": other_files,
    }


def archive_processed_artifacts(processed_dir: Path, archive_root: Path) -> dict[str, Any]:
    """Move current processed artifacts into a timestamped archive directory."""

    processed_path = Path(processed_dir)
    archive_base = Path(archive_root)
    archive_name = datetime.now(timezone.utc).strftime("rebuild_%Y%m%dT%H%M%SZ")
    archive_dir = archive_base / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved_files: list[str] = []
    for path in sorted(processed_path.iterdir()):
        if not path.is_file():
            continue
        destination = archive_dir / path.name
        shutil.move(str(path), destination)
        moved_files.append(path.name)

    return {
        "archive_dir": str(archive_dir),
        "moved_files": moved_files,
    }


def collect_document_metadata_issues(pdf_path: Path, chunks: list[Chunk]) -> list[str]:
    """Flag conservative metadata quality issues for one rebuilt document."""

    issues: list[str] = []
    if not chunks:
        return ["No chunks were produced."]

    act_titles = {chunk.act_title.strip() for chunk in chunks if chunk.act_title.strip()}
    document_ids = {chunk.document_id.strip() for chunk in chunks if chunk.document_id.strip()}
    unit_types = {chunk.unit_type.strip().lower() for chunk in chunks if chunk.unit_type.strip()}
    alias_set = {alias.strip() for chunk in chunks for alias in chunk.document_aliases if alias.strip()}

    if len(act_titles) != 1:
        issues.append(f"Inconsistent act_title values: {sorted(act_titles)}")
    else:
        act_title = next(iter(act_titles))
        if TITLE_NOISE_PATTERN.search(act_title):
            issues.append(f"act_title contains title-noise text: {act_title}")
        if FILENAMEISH_TITLE_PATTERN.search(act_title):
            issues.append(f"act_title looks filename-derived: {act_title}")

    if len(document_ids) != 1:
        issues.append(f"Inconsistent document_id values: {sorted(document_ids)}")

    if len(unit_types) != 1:
        issues.append(f"Mixed unit types detected: {sorted(unit_types)}")

    if not alias_set:
        issues.append("Missing document aliases.")
    else:
        noisy_aliases = sorted(alias for alias in alias_set if ALIAS_NOISE_PATTERN.search(alias))
        if noisy_aliases:
            issues.append(f"Filename-like or noisy aliases detected: {noisy_aliases}")

    expected_source = pdf_path.name
    wrong_source_files = sorted({chunk.source_file for chunk in chunks if chunk.source_file != expected_source})
    if wrong_source_files:
        issues.append(f"Unexpected source_file values: {wrong_source_files}")

    missing_unit_ids = [chunk.chunk_id for chunk in chunks if not (chunk.unit_id or chunk.section_id)]
    if missing_unit_ids:
        issues.append(f"Chunks missing unit_id/section_id: {missing_unit_ids[:5]}")

    return issues


def validate_graph_consistency(chunks: list[Chunk]) -> dict[str, Any]:
    """Check that graph nodes and edge references stay internally consistent."""

    graph = build_legal_graph(chunks)
    unit_node_ids = {
        unit_node.node_id
        for unit_nodes in graph.units_by_document.values()
        for unit_node in unit_nodes
    }
    hierarchy_node_ids = set(graph.hierarchy_nodes)
    orphan_edges: list[dict[str, str]] = []
    informational_edges: list[dict[str, str]] = []

    for source_id, target_id in graph.next_unit.items():
        if source_id not in unit_node_ids or target_id not in unit_node_ids:
            orphan_edges.append({"edge_type": "NEXT_UNIT", "source": source_id, "target": target_id})

    for source_id, targets in graph.refers_to.items():
        for target_id in targets:
            if source_id not in unit_node_ids or target_id not in unit_node_ids:
                orphan_edges.append({"edge_type": "REFERS_TO", "source": source_id, "target": target_id})

    for target_id, sources in graph.reverse_refers_to.items():
        for source_id in sources:
            if target_id not in unit_node_ids or source_id not in unit_node_ids:
                orphan_edges.append({"edge_type": "REVERSE_REFERS_TO", "source": source_id, "target": target_id})

    for source_id, targets in graph.amends.items():
        for target_id in targets:
            if source_id not in unit_node_ids or target_id not in unit_node_ids:
                orphan_edges.append({"edge_type": "AMENDS", "source": source_id, "target": target_id})

    for source_id, targets in graph.inserts.items():
        for target_id in targets:
            if source_id not in unit_node_ids:
                orphan_edges.append({"edge_type": "INSERTS", "source": source_id, "target": target_id})
            elif target_id not in unit_node_ids:
                informational_edges.append(
                    {"edge_type": "INSERTS_UNMATERIALIZED_TARGET", "source": source_id, "target": target_id}
                )

    for hierarchy_id, contained_units in graph.hierarchy_contains.items():
        for unit_id in contained_units:
            if hierarchy_id not in hierarchy_node_ids or unit_id not in unit_node_ids:
                orphan_edges.append({"edge_type": "CONTAINS", "source": hierarchy_id, "target": unit_id})

    return {
        "document_count": len(graph.documents),
        "unit_node_count": len(unit_node_ids),
        "hierarchy_node_count": len(hierarchy_node_ids),
        "refers_to_edge_count": sum(len(targets) for targets in graph.refers_to.values()),
        "amends_edge_count": sum(len(targets) for targets in graph.amends.values()),
        "inserts_edge_count": sum(len(targets) for targets in graph.inserts.values()),
        "orphan_edge_count": len(orphan_edges),
        "orphan_edges": orphan_edges[:50],
        "informational_edge_count": len(informational_edges),
        "informational_edges": informational_edges[:50],
    }


def compare_corpus_snapshots(previous: dict[str, Any], current_documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare previous and current corpus summaries."""

    previous_docs = {doc["file_name"]: doc for doc in previous.get("documents", [])}
    current_docs = {doc["file_name"]: doc for doc in current_documents}

    added = sorted(name for name in current_docs if name not in previous_docs)
    removed = sorted(name for name in previous_docs if name not in current_docs)
    changed: list[dict[str, Any]] = []
    for name in sorted(current_docs):
        if name not in previous_docs:
            continue
        previous_doc = previous_docs[name]
        current_doc = current_docs[name]
        if previous_doc.get("chunk_count") != current_doc.get("chunk_count") or previous_doc.get("unit_types") != current_doc.get("unit_types"):
            changed.append(
                {
                    "file_name": name,
                    "previous_chunk_count": previous_doc.get("chunk_count"),
                    "current_chunk_count": current_doc.get("chunk_count"),
                    "previous_unit_types": previous_doc.get("unit_types"),
                    "current_unit_types": current_doc.get("unit_types"),
                }
            )

    return {
        "previous_document_count": previous.get("document_count", 0),
        "current_document_count": len(current_documents),
        "previous_total_chunks": previous.get("total_chunks", 0),
        "current_total_chunks": sum(int(doc["chunk_count"]) for doc in current_documents),
        "added_documents": added,
        "removed_documents": removed,
        "changed_documents": changed,
        "previous_other_files": previous.get("other_files", []),
    }


def write_corpus_report(report: dict[str, Any], output_path: Path) -> None:
    """Persist the processed-corpus rebuild report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_processed_corpus(settings: LegalRAGSettings) -> dict[str, Any]:
    """Rebuild the processed corpus from raw PDFs and emit a validation report."""

    raw_pdf_paths = sorted(settings.raw_law_pdfs_dir.glob("*.pdf"))
    previous_snapshot = snapshot_processed_corpus(settings.processed_dir)
    archive_info = archive_processed_artifacts(settings.processed_dir, settings.processed_dir / "archive")

    processed_documents: list[dict[str, Any]] = []
    failed_documents: list[dict[str, str]] = []
    metadata_issues: list[dict[str, Any]] = []
    leakage_summary: list[dict[str, Any]] = []
    all_chunks: list[Chunk] = []
    unit_type_distribution: Counter[str] = Counter()

    for pdf_path in raw_pdf_paths:
        try:
            chunks = ingest_law_pdf_to_chunks(
                pdf_path,
                max_words=settings.chunk_size_words,
                overlap_words=settings.chunk_overlap_words,
            )
            output_path = settings.processed_dir / f"{pdf_path.stem}.jsonl"
            export_chunks_to_jsonl(chunks, output_path)
        except Exception as exc:  # pragma: no cover - exercised by real corpus rebuild
            failed_documents.append({"source_pdf": pdf_path.name, "error": str(exc)})
            continue

        leaks = find_section_boundary_leaks(chunks)
        issues = collect_document_metadata_issues(pdf_path, chunks)
        doc_unit_types = sorted({chunk.unit_type for chunk in chunks if chunk.unit_type})
        act_title = chunks[0].act_title if chunks else pdf_path.stem

        processed_documents.append(
            {
                "file_name": output_path.name,
                "source_pdf": pdf_path.name,
                "document_id": chunks[0].document_id if chunks else pdf_path.stem,
                "act_title": act_title,
                "chunk_count": len(chunks),
                "unit_types": doc_unit_types,
                "leakage_count": len(leaks),
                "metadata_issue_count": len(issues),
            }
        )
        leakage_summary.append(
            {
                "file_name": output_path.name,
                "source_pdf": pdf_path.name,
                "leakage_count": len(leaks),
                "examples": [asdict(leak) for leak in leaks[:10]],
            }
        )
        if issues:
            metadata_issues.append(
                {
                    "file_name": output_path.name,
                    "source_pdf": pdf_path.name,
                    "issues": issues,
                }
            )

        all_chunks.extend(chunks)
        unit_type_distribution.update(doc_unit_types or ["unknown"])

    graph_consistency = validate_graph_consistency(all_chunks)
    comparison = compare_corpus_snapshots(previous_snapshot, processed_documents)
    report = {
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "raw_pdf_count": len(raw_pdf_paths),
        "processed_document_count": len(processed_documents),
        "total_chunks": sum(document["chunk_count"] for document in processed_documents),
        "documents_processed": processed_documents,
        "processed_document_list": [document["file_name"] for document in processed_documents],
        "failed_documents": failed_documents,
        "unit_type_distribution": dict(sorted(unit_type_distribution.items())),
        "leakage_summary": leakage_summary,
        "metadata_issues": metadata_issues,
        "graph_consistency": graph_consistency,
        "archive_info": archive_info,
        "previous_snapshot": previous_snapshot,
        "comparison_with_previous": comparison,
    }
    return report
