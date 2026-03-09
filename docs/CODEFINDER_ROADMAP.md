# CODEFINDER Roadmap

Status date: 2026-03-07

This roadmap turns the repo-wide cleanup plan into a working sequence. It is intentionally narrower than the historical sprawl already present in the repository.

## Direction

- One repo, clearer lanes
- Dual-track overall: product plus research
- German/Kempten is the active research priority
- Shakespeare is the secondary cleanup track
- Exploratory gematria, ELS, cipher, and prophetic material is preserved but removed from the active roadmap

## Phase Status

| Phase | Status | Scope |
| --- | --- | --- |
| Phase 1 | In progress | Lock repo identity, lane labeling, runtime contract, and top-level docs |
| Phase 2 | In progress | Finish German/Kempten to a stable shareable-plus-archive standard |
| Phase 3 | Not started | Rebuild Shakespeare into one sober canonical package |
| Phase 4 | Started | Reduce script and artifact sprawl, define storage and rebuild rules |
| Phase 5 | Not started | Audit and narrow the product surface around supported workflows |

## Phase 1: Lock Repo Identity And Lanes

Implemented:

- `README.md` rewritten around product plus historical-print research
- `docs/REPO_INDEX.md` added as contributor navigation
- `docs/REPO_CONTRACT.md` added as the locked operating contract
- `docs/architecture.md` rewritten to match the active repo shape
- `archive/`, `data/`, `reports/`, and `scripts/` now have explicit lane/storage documentation

Still to do:

- Continue removing outdated top-level messaging from legacy user-facing docs where needed
- Audit route descriptions and product copy for language inherited from the older "ancient text / prophetic" framing

## Phase 2: Finish German/Kempten

Already in place:

- Shareable summary: `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- Archive summary and PDF package under `reports/final_report/`
- External manuscript draft: `docs/BOOK_HISTORY_MANUSCRIPT.md`
- Methods appendix: `docs/BOOK_HISTORY_METHODS_APPENDIX.md`

Remaining blockers:

- Manual validation of the top character-sort matches
- Negative-control corpus processed through the same pipeline
- Corrected source chronology for damage evolution
- Final wording review across all German/Kempten outward-facing outputs

Stable means:

- Source metadata, generated artifacts, shareable summary, and manuscript language all agree

## Phase 3: Rebuild Shakespeare

Required outcomes:

- One canonical Wright/Aspley summary replaces the contradictory legacy report cluster
- Clear separation between reusable methods, tentative findings, and archived claims
- A future shareable Shakespeare package is created only after that consolidation

Control document:

- `docs/forensic_methodology_critique.md`
- `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`

## Phase 4: Reduce Operational Noise

Implemented or started:

- Local noise directories can now be ignored cleanly
- Script lanes are documented under `scripts/`
- `data/` and `reports/` now have explicit storage rules
- Rebuild and retention rules are now centralized in `docs/REBUILD_AND_RETENTION_MANIFEST.md`

Still to do:

- Continue moving active script ownership into documented lanes
- Reduce tracked generated artifacts where practical
- Add reproducible rebuild paths for each active report family

## Phase 5: Stabilize Product Surface

Required outcomes:

- Frontend and API surface match the supported roadmap
- Important verification scripts become tests where feasible
- CI remains focused on backend runtime health, frontend build health, and a small research smoke-check set

## Current Deliverable Priority Order

1. Keep the FastAPI and frontend runtime stable.
2. Finish German/Kempten validation work.
3. Consolidate Shakespeare into one sober package.
4. Continue artifact and script cleanup.
5. Only then expand product features again.
