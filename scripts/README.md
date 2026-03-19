# Scripts Index

This directory is the canonical home for research entrypoints, rebuild helpers, and maintenance scripts.

## Transition Note

The repository still contains a flat `scripts/` layout from earlier phases. This document is the source of truth for script ownership while the physical migration continues. New scripts should go into the lane directories introduced below, not into the top level of the repository.

## Lane Directories

- `scripts/research/` - active research automation and canonical report generation
- `scripts/maintenance/` - setup, rebuild, migration, and environment helpers
- `scripts/legacy/` - retired or superseded entrypoints kept for history

## Current Script Classification

### Active research: German/Kempten

- `acquire_sources.py`
- `damage_evolution.py`
- `extract_characters.py`
- `extract_ornaments.py`
- `formal_stats.py`
- `generate_discord_summary.py`
- `generate_final_report.py`
- `generate_pdf_report.py`
- `match_character_sorts.py`
- `prepare_proof_images.py`
- `scan_greenman_all.py`
- `research/generate_manual_review_ledger.py`
- `research/generate_greenman_review_sheet.py`

### Canonical research: Shakespeare lane

- `research/generate_shakespeare_canonical_artifacts.py`
- `maintenance/rebuild_shakespeare_lane.py`

### Historical and diagnostic Shakespeare scripts

- `compare_page.py`
- `download_aspley_sonnets.py`
- `download_folger_sonnets.py`
- `generate_forensic_pdf.py`
- `generate_variance_atlas.py`
- `isolate_sonnets.py`
- `register_and_extract.py`
- `register_page_images.py`
- `run_forensic_scan_v3.py`
- `sonnet_census.py`
- `sonnet_census_prealigned.py`
- `verify_anomalies.py`

### Maintenance and rebuild

- `setup_env.py`
- `test_visualizer_integration.py`
- `maintenance/rebuild_german_lane.py`
- `maintenance/run_negative_control.py`

### Legacy or archived

- `audit_body_anomalies.py`
- `audit_restorations.py`
- `generate_matrix_svg.py`
- `run_pepe_monte_carlo.py`
- `verify_prophetic_services.py`
- `visualize_matrix.py`

## Rules

- Do not add new research scripts at repo root.
- If a script produces a report that should be shareable, pair it with a canonical output path and document it in `docs/REPO_INDEX.md`.
- When a script becomes obsolete, move it into the archive lane or document it as superseded instead of silently leaving it as an unexplained orphan.
