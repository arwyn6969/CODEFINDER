# Repo Contract

Effective date: 2026-03-07

This document locks the operating rules for CODEFINDER as a combined product and historical-print research repository.

## 1. Runtime Contract

- The canonical backend entrypoint is `app.api.main:app`.
- The canonical public API surface lives under `/api/*`.
- Public docs live at `/api/docs`.
- Health checks live at `/api/health`.
- `run_dashboard.sh` is the one-click local launcher, but it must continue to resolve to the same canonical backend and frontend workspaces.

## 2. Research Output Contract

### German/Kempten

- One canonical shareable output:
  - `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- One canonical archive package:
  - `reports/final_report/summary.txt`
  - `reports/final_report/final_report.html`
  - `reports/final_report/CODEFINDER_Forensic_Report.pdf`
  - `docs/BOOK_HISTORY_MANUSCRIPT.md`
  - `docs/BOOK_HISTORY_METHODS_APPENDIX.md`
- No outward-facing German report may be generated from stale metadata or wording that exceeds the cleaned evidence.

### Shakespeare

- One canonical shareable output:
  - `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`
- One canonical archive package:
  - `reports/shakespeare/summary.txt`
  - `reports/shakespeare/shakespeare_archive.html`
  - `reports/shakespeare/CODEFINDER_Shakespeare_Report.pdf`
  - `reports/shakespeare/package_manifest.json`
  - `reports/shakespeare/page_equivalence_manifest.json`
  - `docs/SHAKESPEARE_MANUSCRIPT.md`
  - `docs/SHAKESPEARE_METHODS_APPENDIX.md`
- `docs/forensic_methodology_critique.md` remains the control document for deciding what survives into future stronger claims.
- No outward-facing Shakespeare report may imply typographic identity or same-object proof beyond the current external-safe diagnostic evidence.

### Archive

- Archived exploratory material is preserved for internal reference only.
- Archived work may not be presented in README-level messaging or as active roadmap functionality.

## 3. Organization Contract

- The repo remains a single monorepo.
- The active lanes are:
  - Product
  - German/Kempten research
  - Shakespeare canonical research
  - Archive
- New research scripts do not belong at repo root.
- `scripts/` is the only approved home for new research entrypoints, rebuild helpers, and maintenance scripts.
- Root-level one-off Python files are transition or legacy material unless they are promoted into a documented lane.

## 4. Documentation Contract

- `README.md` describes the active repo identity only.
- `docs/REPO_INDEX.md` is the first-stop navigation document for contributors.
- `docs/architecture.md` explains the current runtime and research architecture without marketing archived exploratory work as an active subsystem.
- `docs/RESEARCH_COMPENDIUM.md` remains an internal notebook only.

## 5. Storage And Artifact Contract

- `data/` is a managed local store, not a general dump.
- `reports/` is a generated artifact store, not a hand-edited source-of-truth directory.
- Durable interpretation belongs in `docs/`.
- Large generated artifacts should remain non-default in git whenever possible; canonical summaries and control documents should be tracked instead.
- When an artifact can be regenerated from code, prefer storing the generator and a lightweight manifest over storing bulk output.

## 6. Exit Criteria For The Current Cleanup

The repo-level cleanup represented by this contract is considered complete only when:

- A new contributor can tell what is active, what is secondary, and what is archived without opening legacy research reports.
- German/Kempten shareable and archive outputs agree on dates, claims, and limitations.
- Shakespeare shareable and archive outputs agree on corpus counts, claim level, and limitations.
- Artifact churn is reduced so `git status` is mostly code and doc changes rather than bulk generated files.
