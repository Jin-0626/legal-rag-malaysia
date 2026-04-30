# Data Layout

This repository treats `data/processed/` and `data/embeddings/` as generated artifacts.

Source-controlled data kept in the repo:

- `data/raw_law_pdfs/` for the raw corpus inputs
- `data/evaluation/final_gold_set_v2.jsonl`
- `data/evaluation/final_gold_set_v2_report.json`
- `data/evaluation/final_gold_set_v2_report.md`
- `data/evaluation/gold_set_v2_candidates.jsonl`
- `data/evaluation/index_refresh_summary.json`
- `data/evaluation/index_refresh_summary.md`

Generated data that should be rebuilt locally:

- `data/processed/*.jsonl`
- `data/embeddings/*.vectors.jsonl`
- `data/embeddings/chroma.sqlite3`
- temporary or diagnostic evaluation dumps

To rebuild the generated data from raw inputs:

```powershell
.\venv_new\Scripts\python.exe scripts\rebuild_all.py
```
