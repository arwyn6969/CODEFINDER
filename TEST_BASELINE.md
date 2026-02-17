# CODEFINDER Test Baseline
> **Last Verified**: 2026-02-15 (local)
> **Purpose**: Ground-truth snapshot of test health and high-risk technical gaps.

---

## 1. Verification Snapshot

### Runtime Entry Point
- **Canonical app**: `app.api.main:app`
- **Compatibility entrypoint**: `app.main:app` (shim to canonical app)

### Test Collection and Result
- **Command**: `pytest -q`
- **Collected**: `679`
- **Passed**: `668`
- **Skipped**: `11`
- **Failed**: `0`

### Environment Notes from Latest Run
- Python runtime reported via local environment: `3.9.x`
- Warnings still present (non-failing):
  - `pytest-asyncio` loop-scope deprecation warning
  - `pydantic` v2 class-config deprecation warning
  - SQLAlchemy `Query.get()` legacy warning in some services

---

## 2. Current Test Layout

### Collected by `pytest.ini`
- `testpaths = tests`
- `python_files = test_*.py`

### Impact
- Files named `verify_*.py` are **not** collected by default and should be treated as ad-hoc verification scripts unless renamed or explicitly invoked.

---

## 3. High-Risk Gaps Remaining

1. API surface consistency
- Canonical `/api/*` routes are active.
- Legacy `/api/v1/*` aliases are maintained for compatibility.
- Keep all new feature development targeting `/api/*` only.

2. Research feature contract drift risk
- Research/relationship endpoints are now mounted in canonical app.
- Keep frontend service paths aligned with backend route registration.

3. Deprecation cleanup backlog
- Replace deprecated `from_orm` usage with `model_validate` where practical.
- Replace SQLAlchemy `Query.get()` with `Session.get()`.
- Set explicit `pytest-asyncio` loop scope in config.

---

## 4. Practical Next Testing Steps

1. Compatibility smoke tests are now present in `tests/test_api_route_compatibility.py` for:
- legacy docs/health aliases
- `/api/v1/auth/login`
- `/api` and `/api/v1` research/relationship route availability

2. Convert critical `verify_*.py` flows into collected tests (`test_*.py`).

3. Add a CI smoke check that fails on route-registration drift between frontend expected paths and backend mounted prefixes.

---

## 5. Quick Commands

```bash
# Full suite
pytest -q

# Verbose
pytest -v

# One module
pytest tests/test_api_endpoints.py -v

# Markers
pytest -m unit
pytest -m integration
pytest -m "not slow"
```
