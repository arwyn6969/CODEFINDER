# Shakespeare Canonical Lane Index

> Status: Internal lane index for the canonical Shakespeare packet
> Date: March 9, 2026
> Role: Points contributors to the canonical Shakespeare package and keeps older report clusters in their historical place

## Current Status

The Shakespeare lane now has a canonical package, but it remains an external-safe diagnostic packet rather than a finished attribution case. The correct reading is cautious: the lane preserves useful computational evidence and a normalized witness inventory, while explicitly stating that source-equivalence constraints remain unresolved.

## Canonical Shakespeare Package

Use these artifacts first:

- `docs/SHAKESPEARE_MANUSCRIPT.md`
- `docs/SHAKESPEARE_METHODS_APPENDIX.md`
- `reports/shakespeare/summary.txt`
- `reports/shakespeare/package_manifest.json`
- `reports/shakespeare/page_equivalence_manifest.json`
- `reports/shakespeare/comparison/whole_book_comparison.json`
- `reports/shakespeare/comparison/high_confidence_comparison.json`
- `reports/shakespeare/comparison/page_variance_diagnostic.json`
- `reports/shakespeare/manual_review/manual_review_ledger.json`
- `reports/shakespeare/manual_review/manual_review_summary.md`
- `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`

## What Survives From The Older Cluster

The following older artifacts still matter, but only as inputs to the new packet:

- `docs/forensic_methodology_critique.md`
- `reports/page_forensics/page_comparison.json`
- `reports/full_sonnet_mapping.json`
- `reports/scan_wright_fixed/statistics.json`
- `reports/scan_aspley_fixed/statistics.json`
- `reports/wright_80conf/statistics.json`
- `reports/aspley_80conf/statistics.json`

## Reading Rules

- Treat `reports/shakespeare/page_equivalence_manifest.json` as the structural source of truth for the local Shakespeare corpus.
- Treat the normalized Shakespeare comparison JSONs under `reports/shakespeare/comparison/` as the canonical quantitative outputs.
- Treat the manual-review labels as narrow interpretive tags, not as a substitute for bibliographical judgement.
- Treat design similarity as distinct from same-object proof.

## Archived Or Superseded Claims

These claims do not survive into the canonical packet:

- that the Wright and Aspley witnesses are already proven typographically identical;
- that the null hypothesis has been accepted and the alternative rejected;
- that OCR-only explanations settle the full variance picture;
- that the legacy page-forensics prose report remains a safe external summary.

## Legacy Outputs To Keep In Historical Context

The following folders remain preserved, but they are no longer the canonical Shakespeare surface:

- `reports/page_forensics/`
- `reports/wright_vs_aspley_comparison/`
- `reports/wright_vs_aspley_80conf/`
- `reports/scan_wright_fixed/`
- `reports/scan_aspley_fixed/`
- `reports/wright_80conf/`
- `reports/aspley_80conf/`
- `reports/full_sonnet_mapping_report.md`

## If Shakespeare Is Reopened For A Deeper Rerun

The next higher bar is not new prose. It is tighter evidence:

1. improve or complete the sonnet-level page map;
2. rerun extraction from a documented resolution if stronger image-level claims are needed;
3. keep page-variance outputs diagnostic until page equivalence improves;
4. require explicit manual and bibliographical review before any stronger typographic claim is made.
