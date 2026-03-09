# Rebuild And Retention Manifest

Status date: 2026-03-08

This manifest defines what should be kept as canonical repo-tracked outputs and how the active report families are rebuilt.

## Canonical Rebuild Entry Points

### German/Kempten

- Full report family rebuild: `./.venv/bin/python scripts/maintenance/rebuild_german_lane.py`
- Manual review ledger only: `./.venv/bin/python scripts/research/generate_manual_review_ledger.py`
- Greenman review sheet only: `./.venv/bin/python scripts/research/generate_greenman_review_sheet.py`

### Negative control

- Repo-local control run: `./.venv/bin/python scripts/maintenance/run_negative_control.py`

## Retention Rules

### Track in git by default

- Durable docs in `docs/`
- Canonical summaries and control artifacts
- Lightweight JSON or text manifests needed to explain a claim
- Rebuild entrypoints and tests

### Prefer local-only / generated

- Bulk crop images in `reports/crops/`
- Large PDF and HTML families that can be regenerated
- One-off debug exports, intermediate overlays, and cache-like artifacts
- Negative-control reruns that are useful for local review but not part of the canonical repo narrative

## Canonical Report Families

| Family | Canonical tracked artifacts | Rebuild path |
| --- | --- | --- |
| German shareable | `reports/shareable/CODEFINDER_Discord_Summary.pdf` | `scripts/maintenance/rebuild_german_lane.py` |
| German archive | `reports/final_report/summary.txt` and companion HTML/PDF | `scripts/maintenance/rebuild_german_lane.py` |
| German manual review | `reports/manual_review/manual_review_ledger.md`, `reports/manual_review/greenman_review.md`, `reports/manual_review/greenman_review_sheet.png` | `scripts/research/generate_manual_review_ledger.py`, `scripts/research/generate_greenman_review_sheet.py` |
| Negative control | `reports/negative_control/negative_control_memo.md` | `scripts/maintenance/run_negative_control.py` |
| Shakespeare internal control | `docs/SHAKESPEARE_INTERNAL_SUMMARY.md` | documentation-only synthesis from surviving artifacts |

## Operational Defaults

- Do not hand-edit generated report content when a rebuild script exists.
- Do not reuse Shakespeare legacy outputs without first checking `docs/SHAKESPEARE_INTERNAL_SUMMARY.md`.
- Do not promote damage-evolution output into outward-facing claims unless chronology-sensitive metadata has been reconciled and rerun.
- When removing generated artifacts from git tracking, keep the local files and preserve the rebuild command in this manifest.
