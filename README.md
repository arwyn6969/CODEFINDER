# CODEFINDER

CODEFINDER is a single repository with two active purposes:

1. A FastAPI + React application for document ingestion, OCR-driven analysis, and report delivery.
2. A historical-print research workflow, currently centered on the German/Kempten corpus.

This repository is no longer positioned as a general-purpose cipher or prophetic-analysis sandbox. Exploratory gematria, ELS, and related material is preserved for audit history, but it is not part of the active roadmap.

## Active Lanes

| Lane | Status | Scope | Canonical outputs |
| --- | --- | --- | --- |
| Product | Active | FastAPI backend, React frontend, API routes under `/api/*` | Running app, API docs, tests |
| German/Kempten research | Active priority | Four-source early modern German/Latin print comparison | `reports/shareable/CODEFINDER_Discord_Summary.pdf`, `reports/final_report/summary.txt`, `docs/BOOK_HISTORY_MANUSCRIPT.md`, `docs/BOOK_HISTORY_METHODS_APPENDIX.md` |
| Shakespeare research | Secondary canonical | Wright/Aspley witness normalization and diagnostic comparison | `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`, `reports/shakespeare/summary.txt`, `docs/SHAKESPEARE_MANUSCRIPT.md`, `docs/SHAKESPEARE_METHODS_APPENDIX.md` |
| Exploratory archive | Archived | Gematria, ELS, prophetic, and related exploratory work | Preserved for history only; not safe to circulate as active findings |

## Runtime Contract

- Canonical backend entrypoint: `app.api.main:app`
- Public API surface: `/api/*`
- API docs: `http://localhost:8000/api/docs`
- Health endpoint: `http://localhost:8000/api/health`
- Frontend workspace: `frontend/`
- One-click local launcher: `./run_dashboard.sh`

Legacy routes or legacy research files may still exist in the tree for compatibility or auditability, but they are not the source of truth for current repo identity.

## Quick Start

### Local Python + Node

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm start
```

### Docker

```bash
docker-compose up -d
```

## What To Read First

- `docs/DEVELOPER_HANDOFF.md` - current branch picture, project stage, and next steps for the next developer
- `docs/REPO_INDEX.md` - where active code, reports, archives, and generated artifacts live
- `docs/REPO_CONTRACT.md` - locked runtime, organization, and documentation rules
- `docs/CODEFINDER_ROADMAP.md` - current implementation phases and remaining blockers
- `docs/REBUILD_AND_RETENTION_MANIFEST.md` - canonical rebuild paths and artifact retention rules
- `docs/architecture.md` - current product and research architecture

## Research Outputs

### German/Kempten

- Shareable summary: `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- Research archive summary: `reports/final_report/summary.txt`
- External-facing manuscript draft: `docs/BOOK_HISTORY_MANUSCRIPT.md`
- Methods and reproducibility appendix: `docs/BOOK_HISTORY_METHODS_APPENDIX.md`

### Shakespeare

- Shareable summary: `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`
- Research archive summary: `reports/shakespeare/summary.txt`
- External-facing manuscript draft: `docs/SHAKESPEARE_MANUSCRIPT.md`
- Methods and reproducibility appendix: `docs/SHAKESPEARE_METHODS_APPENDIX.md`
- Control document and lane index: `docs/forensic_methodology_critique.md`, `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`

### Archived exploratory work

- Internal notebook only: `docs/RESEARCH_COMPENDIUM.md`
- Historical archive scripts: `archive/`

## Repository Layout

```text
app/         FastAPI application and service layer
frontend/    React application
scripts/     Research, rebuild, and maintenance entrypoints
tests/       Pytest suite and targeted research checks
docs/        Contracts, architecture, roadmaps, and research writing
reports/     Generated reports and human-readable summaries
data/        Local corpora, source configuration, and forensic databases
archive/     Preserved exploratory and retired research material
```

Root-level one-off Python files outside `app/` and `scripts/` are retained for history or transition work. They are not the canonical place to add new functionality.

## Testing

```bash
pytest -q
cd frontend && npm run build
```

The GitHub Actions workflow currently checks backend tests on Python `3.9` and `3.11`, plus a frontend production build on Node `18`.

## Contribution Rules

- Preserve the runtime contract around `app.api.main:app` and `/api/*`.
- Treat German/Kempten as the active research lane.
- Do not present Shakespeare or archived exploratory outputs as settled external findings.
- Do not add new research scripts at repo root.
- Prefer regenerating report artifacts from scripts rather than editing outputs by hand.

## License

This project is licensed under the MIT License. See `LICENSE`.
