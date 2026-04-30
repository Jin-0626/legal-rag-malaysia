"""Generate conservative Gold Set V2 candidates from processed legal chunks."""

from __future__ import annotations

from legal_rag.config.settings import build_settings
from legal_rag.evaluation import (
    build_gold_set_v2_candidates,
    format_gold_set_summary,
    write_gold_set_candidates,
)


def main() -> None:
    settings = build_settings()
    candidates = build_gold_set_v2_candidates(settings.processed_dir)
    jsonl_path = settings.data_dir / "evaluation" / "gold_set_v2_candidates.jsonl"
    summary_path = settings.data_dir / "evaluation" / "gold_set_v2_summary.md"

    write_gold_set_candidates(candidates, jsonl_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(format_gold_set_summary(candidates), encoding="utf-8")

    print(f"candidates={len(candidates)}")
    print(f"jsonl={jsonl_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
