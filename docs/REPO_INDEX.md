# Repo Index

Use this document as the first-stop map for the repository.

## Start Here

| Need | Open this |
| --- | --- |
| Run the backend | `app.api.main:app` |
| Run the full local stack | `run_dashboard.sh` |
| Understand repo rules | `docs/REPO_CONTRACT.md` |
| Understand current architecture | `docs/architecture.md` |
| See active roadmap phases | `docs/CODEFINDER_ROADMAP.md` |
| See rebuild and retention rules | `docs/REBUILD_AND_RETENTION_MANIFEST.md` |
| Read the current German/Kempten manuscript | `docs/BOOK_HISTORY_MANUSCRIPT.md` |
| Read German/Kempten methods and reproducibility | `docs/BOOK_HISTORY_METHODS_APPENDIX.md` |
| Assess Shakespeare cleanup status | `docs/SHAKESPEARE_INTERNAL_SUMMARY.md` |

## Active, Secondary, and Archived Lanes

### Product lane

- Source: `app/`, `frontend/`, `tests/`
- Canonical runtime: `app.api.main:app`
- Canonical API surface: `/api/*`
- Purpose: keep the application runnable and the supported research workflows reachable

### German/Kempten lane

- Source: `scripts/`, `data/sources/`, `docs/BOOK_HISTORY_MANUSCRIPT.md`, `docs/BOOK_HISTORY_METHODS_APPENDIX.md`
- Shareable output: `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- Research archive outputs:
  - `reports/final_report/summary.txt`
  - `reports/final_report/final_report.html`
  - `reports/final_report/CODEFINDER_Forensic_Report.pdf`
- Status: active research priority

### Shakespeare lane

- Source: legacy and transitional scripts in `scripts/`, older report folders in `reports/`, `docs/forensic_methodology_critique.md`, and `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
- Status: secondary cleanup track
- Internal source of truth: `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
- External-safe package: not yet available

### Archive lane

- Source: `archive/`, `docs/RESEARCH_COMPENDIUM.md`
- Status: preserved, not active
- Rule: keep for history and audit only; do not present as current project claims

## Directory Map

| Path | Purpose |
| --- | --- |
| `app/` | FastAPI application, models, services, templates |
| `frontend/` | React application and build tooling |
| `scripts/` | Research, rebuild, and maintenance entrypoints |
| `tests/` | Backend tests and targeted research verification |
| `docs/` | Durable contracts, architecture docs, manuscripts, and critiques |
| `reports/` | Generated reports, proof images, and summaries |
| `data/` | Local corpora, source config, and forensic databases |
| `archive/` | Retired exploratory scripts and archived project material |

## Canonical Outputs By Lane

| Lane | Shareable | Archive / control |
| --- | --- | --- |
| German/Kempten | `reports/shareable/CODEFINDER_Discord_Summary.pdf` | `reports/final_report/summary.txt`, `docs/BOOK_HISTORY_MANUSCRIPT.md`, `docs/BOOK_HISTORY_METHODS_APPENDIX.md` |
| Shakespeare | None until cleanup completes | `docs/SHAKESPEARE_INTERNAL_SUMMARY.md` plus `docs/forensic_methodology_critique.md` |
| Exploratory archive | None | `docs/RESEARCH_COMPENDIUM.md`, `archive/` |

## Things Not To Treat As Canonical

- Root-level one-off Python scripts outside `app/` and `scripts/`
- Older report folders that predate the March 7, 2026 German/Kempten cleanup
- `docs/RESEARCH_COMPENDIUM.md` for any external audience
- Any outward-facing report that mixes stale dates, stale source metadata, or legacy verdict language

## Working Rules

- Add new research automation under `scripts/`, not at repo root.
- Regenerate artifacts instead of editing report outputs by hand.
- Use `reports/README.md` and `data/README.md` before adding new large local artifacts.
- Treat the German/Kempten lane as the current source of truth for active historical-print research.
