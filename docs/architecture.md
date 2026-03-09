# CODEFINDER Architecture

This repository is a single monorepo with two active responsibilities:

1. Product runtime: a FastAPI backend and React frontend for document-oriented analysis workflows.
2. Historical-print research: a set of reproducible pipelines and reports, with German/Kempten as the active research lane.

## Lane Overview

| Lane | Status | Primary locations | Notes |
| --- | --- | --- | --- |
| Product | Active | `app/`, `frontend/`, `tests/` | Canonical runtime stays `app.api.main:app` with API routes under `/api/*`. |
| German/Kempten research | Active priority | `scripts/`, `docs/BOOK_HISTORY_MANUSCRIPT.md`, `docs/BOOK_HISTORY_METHODS_APPENDIX.md`, `reports/shareable/`, `reports/final_report/` | Current shareable and archive outputs must stay aligned. |
| Shakespeare research | Secondary cleanup | `scripts/`, `docs/forensic_methodology_critique.md`, legacy report folders in `reports/` | Not yet a single external-safe package. |
| Exploratory archive | Archived | `archive/`, `docs/RESEARCH_COMPENDIUM.md` | Preserved for audit history; not part of the active roadmap. |

## Runtime Architecture

```mermaid
graph TD
    Frontend["React frontend<br/>frontend/"] -->|"HTTP /api/*"| API["FastAPI app<br/>app.api.main:app"]
    API --> Services["Application services<br/>app/services/"]
    API --> DB["PostgreSQL or SQLite"]
    Services --> Files["Local corpora, reports, proof images"]
    ResearchScripts["Research scripts<br/>scripts/"] --> Files
    ResearchScripts --> Reports["Generated reports<br/>reports/"]
    ResearchDocs["Research docs<br/>docs/"] --> Reports
```

### Runtime contract

- Backend entrypoint: `uvicorn app.api.main:app`
- Public docs: `/api/docs`
- Health endpoint: `/api/health`
- Frontend dev server: `frontend/` on port `3000`
- Docker stack: backend, frontend, PostgreSQL

Existing compatibility routes may remain in code, but the contract above is the one contributors should treat as stable.

## Research Architecture

### German/Kempten pipeline

```mermaid
graph LR
    Sources["German/Kempten source corpora<br/>data/sources/"] --> OCR["OCR and extraction<br/>extract_characters.py / extract_ornaments.py"]
    OCR --> Matching["Character and ornament comparison<br/>match_character_sorts.py / scan_greenman_all.py"]
    Matching --> Stats["Formal statistics and chronology work<br/>formal_stats.py / damage_evolution.py"]
    Stats --> Reports["Archive and shareable reports<br/>generate_final_report.py / generate_pdf_report.py / generate_discord_summary.py"]
    Reports --> Manuscript["Interpretation and methods docs<br/>BOOK_HISTORY_MANUSCRIPT.md / BOOK_HISTORY_METHODS_APPENDIX.md"]
```

Active evidentiary base:

- `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- `reports/final_report/summary.txt`
- `docs/BOOK_HISTORY_MANUSCRIPT.md`
- `docs/BOOK_HISTORY_METHODS_APPENDIX.md`

Known blockers before the German lane is fully stable:

- Manual validation of top character-sort matches
- Negative-control corpus processed through the same pipeline
- Corrected source chronology for damage analysis
- Final wording pass so every outward-facing claim matches the cleaned artifacts

### Shakespeare lane

The Shakespeare work remains important but secondary. It is currently a cleanup lane, not a finished external package. The control document is `docs/forensic_methodology_critique.md`, and any future shareable output must explicitly supersede the contradictory older report cluster.

## Storage Model

- `data/` is the managed local store for source corpora, configuration, and forensic databases.
- `reports/` is a generated artifact store, not a long-term hand-edited document area.
- `docs/` holds durable human-readable contracts, manuscripts, and reproducibility material.
- `archive/` preserves retired or exploratory work without advertising it as active functionality.

See `docs/REPO_INDEX.md`, `docs/REPO_CONTRACT.md`, `reports/README.md`, and `data/README.md` for the operating rules attached to that storage model.
