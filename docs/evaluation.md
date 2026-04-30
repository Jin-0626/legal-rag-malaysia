# Evaluation

## Durable Assets

Benchmark assets are kept in `data/evaluation/`, including:

- `final_gold_set_v2.jsonl`
- `final_gold_set_v2_report.json`
- `final_gold_set_v2_report.md`
- `gold_set_v2_candidates.jsonl`
- `index_refresh_summary.json`
- `index_refresh_summary.md`

## Benchmark of Record

Final Gold Set V2 currently reports, for the best reranked mode:

- hit@1: `1.000`
- hit@3: `1.000`
- wrong-document rate: `0.000`
- wrong-section rate: `0.000`
- negative/no-answer accuracy: `1.000`

These figures are backed by `data/evaluation/final_gold_set_v2_report.md`.

Metrics should be re-run after corpus changes, chunking changes, retrieval changes, benchmark revisions, or any rebuild of generated processed/vector artifacts.

## Reproduce

```powershell
.\venv_new\Scripts\python.exe scripts/rebuild_all.py
```

To run only the benchmark on an already rebuilt corpus and vector index:

```powershell
$env:PYTHONPATH='src'
.\venv_new\Scripts\python.exe scripts/evaluate_final_gold_set_v2.py
```

## Gold Set Generation

```powershell
$env:PYTHONPATH='src'
.\venv_new\Scripts\python.exe scripts/generate_gold_set_v2.py
```

## Gold Set Curation

```powershell
$env:PYTHONPATH='src'
.\venv_new\Scripts\python.exe scripts/curate_final_gold_set_v2.py
```

## Notes

- `negative_no_answer_accuracy` is a retrieval-side proxy, not a full answer-synthesis hallucination metric.
- Production changes should preserve the benchmark artifacts and avoid changing evaluation assumptions silently.
