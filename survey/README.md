Survey evidence pipeline
========================

This directory contains reproducible scripts for the survey evidence counts.
Do not hand-edit generated files under `out/`.

Workflow:

1. Configure API keys, if available, in `.secrets.env`.
2. Run `python survey/harvest/run_all.py` to populate `raw_exports/`.
3. Run `python survey/scripts/01_ingest.py` to deduplicate metadata.
4. Fill `ledger.csv`; rerun `02_screen_report.py`.
5. Fill `extraction.csv`; rerun `03_extract_report.py`, then figures/tables/checks.
