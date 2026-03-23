# Legacy Exploratory Lab Runbook

The Legacy Exploratory Lab is the preserved internal lane for ELS, gematria, prophetic, cipher, and geographic-style exploratory tooling.

It is intentionally:

- reachable,
- authenticated,
- auditable,
- secondary to the main product and historical-print research lanes.

It is intentionally not:

- the default product workflow,
- an external-facing evidence packet,
- a signal to revive speculative claims as current project positioning.

## Supported Internal Flows

- `POST /api/research/gematria`
  Returns typed gematria results under `results`, with optional pattern persistence.
- `POST /api/research/transliterate`
  Returns Hebrew candidate terms for ELS-oriented searches.
- `POST /api/research/els`
  Returns normalized ELS matches with `start_index`, `end_index`, `location`, `skip`, `direction`, and `term`.
- `POST /api/research/els/visualize`
  Returns a typed matrix payload for the Legacy Lab modal viewer.
- `GET /api/research/geometry/{document_id}`
  Returns a legacy geometry summary built from stored pattern coordinates and BardCode-style coordinate extraction.

The React surface for this lane lives under `frontend/src/pages/research/`.

## Auth Expectations

- Legacy Lab routes remain authenticated.
- Local demo auth continues to work when `ENABLE_DEMO_AUTH=true`.
- Collected smoke tests obtain a token through `/api/auth/login` and then hit the research endpoints with bearer auth.

## Geometry Notes

- The geometry summary is intentionally conservative.
- It uses stored pattern coordinates as its source of truth.
- It reports explicit `status` and `warnings` for no-data and partial-analysis cases instead of silently returning a fully empty “success” shape.
- Coordinate candidates and historical-site matches are internal exploratory artifacts, not external claims.

## Bootstrap And Smoke Path

Use the maintenance script below to bootstrap the Python environment, run the targeted Legacy Lab backend checks, and verify the frontend build:

```bash
./scripts/maintenance/bootstrap_legacy_lab.sh
```

That script currently does three things:

1. Creates or reuses `.venv` and installs `requirements.txt`.
2. Runs the targeted Legacy Lab backend tests.
3. Builds the frontend with `npm run build`.

## Out Of Scope

- Repositioning the repo around exploratory/profhetic claims.
- Expanding the legacy lane before the main product and active research lanes are stable.
- Treating archived notebooks or one-off legacy reports as canonical evidence.
