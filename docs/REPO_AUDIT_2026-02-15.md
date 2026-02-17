# CODEFINDER Repo Audit
> **Date**: 2026-02-15
> **Scope**: Runtime consistency, documentation accuracy, repository hygiene, and test health.

## Executive Summary
The project is functionally healthy (tests pass) but structurally noisy. The main risks were route-entrypoint drift and stale documentation. This audit pass standardizes runtime behavior around `app.api.main:app`, restores endpoint consistency, and updates core docs to reduce onboarding and execution errors.

## What Was Verified

### Runtime and Routing
- Canonical API app: `app/api/main.py`
- Compatibility entrypoint: `app/main.py` now defers to canonical app
- Frontend expects `/api/*` endpoints
- Canonical app now mounts:
  - `/api/auth`
  - `/api/documents`
  - `/api/analysis`
  - `/api/patterns`
  - `/api/search`
  - `/api/reports`
  - `/api/visualizations`
  - `/api/research`
  - `/api/relationships`
  - `/api/ws`
- Legacy aliases are retained under `/api/v1/*` for backward compatibility.

### Test Health
- Latest run: `679` collected, `668` passed, `11` skipped, `0` failed.

### Repository Scale Signals
- Tracked files: `273`
- Working-tree status entries at audit time: `115`
- Top-level Python scripts: `54`
- `reports/`: `~5.3G`, `~295k` files
- `data/`: `~1.9G`

## Structural Risks

1. Artifact sprawl
- Large generated datasets and reports are mixed with product code.
- This creates high cognitive load and noisy git status.

2. Script proliferation
- Many one-off analysis scripts at repo root make ownership and execution paths unclear.

3. Documentation drift
- Existing docs had conflicting route prefixes and stale commands.

4. Non-collected verification scripts
- `verify_*.py` scripts under `tests/` are not collected by default.

## Remediation Applied in This Pass

1. Runtime unification
- Canonicalized operation around `app.api.main:app`.
- Added missing research/relationships routers to canonical app.
- Added legacy `/api/v1/*` aliases for compatibility.
- Added `/docs -> /api/docs` and `/health -> /api/health` compatibility routes.

2. Launcher fix
- Updated `run_dashboard.sh` to launch `app.api.main:app` and print `/api/docs` URL.

3. Documentation updates
- Updated `README.md` and `CODEFINDER_USER_GUIDE.md` for:
  - correct API URLs
  - correct frontend command (`npm start`)
  - docker service naming alignment
  - current test snapshot

4. Baseline refresh
- Replaced stale `TEST_BASELINE.md` with current verified baseline.

5. Route-compatibility guardrails
- Added `tests/test_api_route_compatibility.py` to lock canonical and legacy route behavior.

## Recommended Next Cleanup Phases

### Phase 1 (Safe, high ROI)
- Finalize `.gitignore` strategy for generated report outputs.
- Keep only durable human-readable report summaries in git.

### Phase 2 (Medium effort)
- Move root-level one-off scripts into namespaced folders:
  - `scripts/research/`
  - `scripts/maintenance/`
  - `scripts/legacy/`
- Add a short index file documenting script purpose and expected inputs/outputs.

### Phase 3 (Quality hardening)
- Convert critical `verify_*.py` checks into collected tests.
- Add API smoke tests for research/relationship endpoints in canonical runtime.
- Address deprecation warnings (`from_orm`, `Query.get`, asyncio loop scope).

## Success Criteria for Ongoing Hygiene
- Single documented runtime entrypoint remains canonical.
- Docs and scripts use one endpoint contract (`/api/*`).
- Generated artifacts are excluded by default from version control.
- New research scripts are added only under namespaced script directories.
