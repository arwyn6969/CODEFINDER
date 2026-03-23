# Consolidation Retention Lock

Status date: 2026-03-23

This document freezes the initial keep/legacy/prune-later classification for the consolidation pass. Nothing should be moved or removed unless it fits this contract.

## Runtime Contract

- Canonical backend entrypoint: `app.api.main:app`
- Canonical public API prefix: `/api/*`
- Compatibility API prefix retained during consolidation: `/api/v1/*`
- Internal legacy exploratory surface: `/api/research/*`

## Directory Classification

| Path | Status | Working rule |
| --- | --- | --- |
| `app/`, `frontend/`, `tests/`, `alembic/` | Keep | Supported product/runtime surface. |
| `docs/` | Keep | Durable contributor docs, contracts, manuscripts, and methods. |
| `scripts/` | Keep | Approved home for research, rebuild, maintenance, and legacy entrypoints. |
| `data/` | Keep | Managed local store for corpora and databases. Keep raw source corpora unless explicitly superseded. |
| `reports/shareable/`, `reports/final_report/`, `reports/shakespeare/`, `reports/manual_review/`, `reports/negative_control/` | Keep | Canonical or control outputs tied to the active lanes. |
| `frontend/src/pages/research/`, `app/api/routes/research.py`, `scripts/legacy/`, `archive/research_scripts/` | Internal legacy | Preserve, label clearly, do not advertise as the primary product surface. |
| `archive/` | Archive | Historical audit material only. Not part of the active roadmap. |
| `demo_results/`, `experiments/`, `logs/`, `temp/`, `uploads/` | Prune later | Treat as local/generated unless a specific item is promoted into a documented lane. |
| Root-level one-off scripts and generated artifacts | Prune later | Move into `scripts/legacy/` or `archive/` before any deletion. |

## Do-Not-Delete List

- `README.md`
- `docs/REPO_CONTRACT.md`
- `docs/REPO_INDEX.md`
- `docs/CODEFINDER_ROADMAP.md`
- `docs/REBUILD_AND_RETENTION_MANIFEST.md`
- `docs/BOOK_HISTORY_MANUSCRIPT.md`
- `docs/BOOK_HISTORY_METHODS_APPENDIX.md`
- `docs/SHAKESPEARE_MANUSCRIPT.md`
- `docs/SHAKESPEARE_METHODS_APPENDIX.md`
- `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
- `docs/forensic_methodology_critique.md`
- `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`
- `reports/final_report/summary.txt`
- `reports/shakespeare/summary.txt`
- `reports/shakespeare/package_manifest.json`
- `reports/shakespeare/page_equivalence_manifest.json`
- `reports/manual_review/manual_review_ledger.json`
- `reports/negative_control/negative_control_memo.md`
- `data/sources/config.yaml`
- `data/sources/folger_sonnets_1609/source_metadata.json`
- `data/sources/folger_sonnets_1609_aspley/source_metadata.json`

## Initial Consolidation Rules

- Prefer move or archive over deletion.
- Do not remove raw source corpora during this pass.
- Do not remove canonical report families during this pass.
- Remove rebuildable generated artifacts only after their rebuild path is documented.
- If a root-level file is preserved for history, it belongs in `scripts/legacy/`, `docs/legacy/`, or `archive/`, not at repo root.
