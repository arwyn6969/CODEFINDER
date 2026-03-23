# Repo Index

Use this document as the first-stop map for the repository.

## Start Here

| Need | Open this |
| --- | --- |
| Understand current branch status and handoff state | `docs/DEVELOPER_HANDOFF.md` |
| Run the backend | `app.api.main:app` |
| Run the full local stack | `run_dashboard.sh` |
| Understand repo rules | `docs/REPO_CONTRACT.md` |
| Review cleanup retention decisions | `docs/CONSOLIDATION_RETENTION_LOCK.md` |
| Understand current architecture | `docs/architecture.md` |
| See active roadmap phases | `docs/CODEFINDER_ROADMAP.md` |
| See rebuild and retention rules | `docs/REBUILD_AND_RETENTION_MANIFEST.md` |
| Read the current German/Kempten manuscript | `docs/BOOK_HISTORY_MANUSCRIPT.md` |
| Read German/Kempten methods and reproducibility | `docs/BOOK_HISTORY_METHODS_APPENDIX.md` |
| Read the Shakespeare manuscript | `docs/SHAKESPEARE_MANUSCRIPT.md` |
| Read Shakespeare methods and reproducibility | `docs/SHAKESPEARE_METHODS_APPENDIX.md` |
| Assess Shakespeare lane status | `docs/SHAKESPEARE_INTERNAL_SUMMARY.md` |

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

- Source: `scripts/research/generate_shakespeare_canonical_artifacts.py`, `scripts/maintenance/rebuild_shakespeare_lane.py`, `reports/shakespeare/`, `docs/SHAKESPEARE_MANUSCRIPT.md`, `docs/SHAKESPEARE_METHODS_APPENDIX.md`, `docs/forensic_methodology_critique.md`
- Status: secondary canonical track
- Internal source of truth: `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
- External-safe package: available under `reports/shakespeare/` plus `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`

### Internal legacy exploratory lane

- Source: `app/api/routes/research.py`, `frontend/src/pages/research/`, `scripts/legacy/`, `archive/research_scripts/`
- Status: internal legacy
- Scope: preserved ELS, gematria, prophetic, and geographic-style exploratory tooling
- Rule: keep runnable and documented, but do not treat it as the main product surface or the current research priority

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
| `docs/legacy/` | Preserved historical docs that are no longer source-of-truth |
| `reports/` | Generated reports, proof images, and summaries |
| `data/` | Local corpora, source config, and forensic databases |
| `archive/` | Retired exploratory scripts and archived project material |

## Canonical Outputs By Lane

| Lane | Shareable | Archive / control |
| --- | --- | --- |
| German/Kempten | `reports/shareable/CODEFINDER_Discord_Summary.pdf` | `reports/final_report/summary.txt`, `docs/BOOK_HISTORY_MANUSCRIPT.md`, `docs/BOOK_HISTORY_METHODS_APPENDIX.md` |
| Shakespeare | `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf` | `reports/shakespeare/summary.txt`, `docs/SHAKESPEARE_MANUSCRIPT.md`, `docs/SHAKESPEARE_METHODS_APPENDIX.md`, `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`, `docs/forensic_methodology_critique.md` |
| Exploratory archive | None | `docs/RESEARCH_COMPENDIUM.md`, `archive/` |

## Things Not To Treat As Canonical

- Root-level one-off Python scripts outside `app/` and `scripts/`
- Legacy docs or generated artifacts left at repo root after they have an approved home
- Older report folders that predate the March 7, 2026 German/Kempten cleanup
- `docs/RESEARCH_COMPENDIUM.md` for any external audience
- Any outward-facing report that mixes stale dates, stale source metadata, or legacy verdict language

## Working Rules

- Add new research automation under `scripts/`, not at repo root.
- Regenerate artifacts instead of editing report outputs by hand.
- Use `reports/README.md` and `data/README.md` before adding new large local artifacts.
- Treat the German/Kempten lane as the current source of truth for active historical-print research.
