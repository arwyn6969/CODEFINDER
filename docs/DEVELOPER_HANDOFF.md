# Developer Handoff

Status date: 2026-03-19

This is the fast handoff document for the next developer. Read this first if you need to understand the live branch, the divergent branch, the current development stage, and what should happen next.

## Executive Summary

CODEFINDER is in a consolidation and stabilization phase, not a greenfield build phase. The repo now has a clearer identity:

- one product lane: FastAPI backend plus React frontend;
- one active research priority: German/Kempten;
- one secondary canonical research lane: Shakespeare;
- one archived lane: older exploratory gematria, ELS, and prophetic material kept for history only.

The current local work is a Shakespeare-lane consolidation pass. It turns that lane from "cleanup in progress" into a canonical, externally safe packet with a rebuild script, controlled artifacts, supporting docs, and a targeted test.

## Branch Situation

### Active branch

`main` is the active branch and should remain the default integration line.

Facts:

- Local branch: `main`
- Tracked remote: `origin/main`
- Last shared remote commit before this handoff package: `294b0be` (`2026-03-09`) - `Summarize recent repo changes`

Use `main` as the source of truth for ongoing work after this handoff.

### Divergent side branch

`origin/cursor/project-deep-dive-review-0316` is a divergent side branch, not a peer to merge casually.

Facts:

- Remote branch head: `d37dffb` (`2025-12-09`) - `Add pytest-cov and CI troubleshooting guide`
- Merge base with `origin/main`: `0067b62` (`2025-08-10`) - `Slim publish: initial code without large artifacts`
- It diverged before the current repo direction, lane cleanup, and Shakespeare/German canonical packaging work

Working rule:

- treat `origin/cursor/project-deep-dive-review-0316` as a historical review branch;
- do not merge it wholesale into `main`;
- only cherry-pick specific commits from it if a clearly wanted CI or docs improvement is identified and re-reviewed in the context of current `main`.

## What This Handoff Push Contains

This handoff package is centered on the Shakespeare lane and repo-level documentation that describes it.

### New Shakespeare canonical artifacts

- `reports/shakespeare/summary.txt`
- `reports/shakespeare/shakespeare_archive.html`
- `reports/shakespeare/CODEFINDER_Shakespeare_Report.pdf`
- `reports/shakespeare/package_manifest.json`
- `reports/shakespeare/page_equivalence_manifest.json`
- `reports/shakespeare/comparison/*`
- `reports/shakespeare/manual_review/*`
- `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`

### New Shakespeare rebuild and verification entrypoints

- `scripts/maintenance/rebuild_shakespeare_lane.py`
- `scripts/research/generate_shakespeare_canonical_artifacts.py`
- `tests/test_shakespeare_completion_artifacts.py`

### Updated repo and source-of-truth docs

- `README.md`
- `docs/REPO_INDEX.md`
- `docs/REPO_CONTRACT.md`
- `docs/CODEFINDER_ROADMAP.md`
- `docs/REBUILD_AND_RETENTION_MANIFEST.md`
- `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
- `docs/SHAKESPEARE_MANUSCRIPT.md`
- `docs/SHAKESPEARE_METHODS_APPENDIX.md`
- `reports/README.md`
- `scripts/README.md`

### Updated source metadata

- `data/sources/config.yaml`
- `data/sources/folger_sonnets_1609/source_metadata.json`
- `data/sources/folger_sonnets_1609_aspley/source_metadata.json`

The important change in those metadata files is that the current local Shakespeare corpus is described honestly as a `2000px` JPG cache, while still noting that native/full IIIF reacquisition remains possible later.

## Current Development Stage

The repo is between research cleanup and operational stabilization. It is not ready for broad new feature growth yet.

### Phase readout

| Phase | Status | Meaning now |
| --- | --- | --- |
| Phase 1 | In progress | Repo identity and contributor-facing docs are much clearer, but legacy messaging still exists in pockets |
| Phase 2 | In progress | German/Kempten is still the active research priority, but it still has validation blockers |
| Phase 3 | In progress | Shakespeare now has a canonical packet, but it is still a cautious diagnostic package, not a final claim |
| Phase 4 | Started | Script/artifact sprawl is being reduced, but more cleanup is still needed |
| Phase 5 | Not started | Product-surface narrowing and supported-workflow cleanup still remain ahead |

### Practical stage summary

- Product runtime: should stay stable, but it is not the current expansion priority.
- German/Kempten lane: still the primary research lane and still needs validation work before it should be treated as fully stable.
- Shakespeare lane: newly consolidated into a controlled package, but intentionally limited to provisional computational evidence and diagnostic comparison.
- Repo operations: improving, but still not fully normalized.

## What Is Canonical Right Now

### Product

- Backend entrypoint: `app.api.main:app`
- Public API surface: `/api/*`
- Frontend workspace: `frontend/`

### German/Kempten

- Shareable output: `reports/shareable/CODEFINDER_Discord_Summary.pdf`
- Archive/manuscript outputs:
  - `reports/final_report/summary.txt`
  - `docs/BOOK_HISTORY_MANUSCRIPT.md`
  - `docs/BOOK_HISTORY_METHODS_APPENDIX.md`

### Shakespeare

- Shareable output: `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`
- Archive/manuscript outputs:
  - `reports/shakespeare/summary.txt`
  - `docs/SHAKESPEARE_MANUSCRIPT.md`
  - `docs/SHAKESPEARE_METHODS_APPENDIX.md`
- Internal control docs:
  - `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
  - `docs/forensic_methodology_critique.md`

### Not canonical

- old Shakespeare legacy report clusters outside `reports/shakespeare/`;
- root-level one-off research scripts as a place for new work;
- archived exploratory gematria, ELS, and prophetic outputs as active claims.

## Key Constraints For The Next Developer

### 1. Do not reopen branch strategy

Keep working from `main`. The divergent cursor branch is reference material at best.

### 2. Do not overclaim the Shakespeare lane

The canonical Shakespeare package is deliberately cautious. It does not support "proven typographic identity", "same object", or similar language. Treat it as a disciplined checkpoint, not a solved case.

### 3. Keep German/Kempten as the active research priority

If you need to choose between advancing Shakespeare polish and closing German/Kempten blockers, the roadmap says German/Kempten comes first.

### 4. Continue reducing repo noise

Prefer documented rebuild scripts, targeted tests, and controlled output folders over ad hoc root-level scripts and manual artifact edits.

## Recommended Next Steps

1. Keep `main` as the integration branch and close this handoff push there.
2. Finish German/Kempten validation work: manual match review, negative-control processing, chronology correction, and final wording review.
3. For Shakespeare, improve or replace the partial sonnet-opening map before making any stronger equivalence claims.
4. Decide whether a native-resolution Shakespeare rerun is needed if later work depends on image-level claims rather than diagnostic comparison.
5. Require specialist bibliographical review before upgrading any Shakespeare row beyond design-level similarity.
6. Continue Phase 4 cleanup by shrinking script/artifact sprawl and turning important verification checks into tests where feasible.
7. Start Phase 5 only after the repo lanes and evidence surfaces are stable enough that the supported product workflows are clear.

## First Files To Read

If someone new is taking over, this is the shortest useful reading order:

1. `README.md`
2. `docs/REPO_INDEX.md`
3. `docs/CODEFINDER_ROADMAP.md`
4. `docs/REPO_CONTRACT.md`
5. `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`
6. `docs/DEVELOPER_HANDOFF.md`

## Rebuild And Verification Commands

### Shakespeare lane

```bash
./.venv/bin/python scripts/maintenance/rebuild_shakespeare_lane.py
pytest -q tests/test_shakespeare_completion_artifacts.py
```

### Product runtime

```bash
uvicorn app.api.main:app --reload --port 8000
cd frontend && npm start
```

## Bottom Line

The repo is no longer "figure out what this project is." That part is mostly solved. The current stage is "stabilize the cleaned structure, finish the primary German/Kempten work, and keep Shakespeare disciplined and reproducible." The next developer should treat this handoff as a consolidation checkpoint, not as permission to broaden claims or reopen old branch history.
