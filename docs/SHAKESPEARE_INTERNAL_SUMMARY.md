# Shakespeare Internal Canonical Summary

> Status: Internal control packet only
> Audience: Repository contributors and internal research review
> Role: Supersedes the contradictory Wright/Aspley report cluster as the single Shakespeare source of truth

## Current Status

The Shakespeare lane is not an external-facing research package. It remains a secondary cleanup track whose purpose is to preserve useful methods, identify which findings survive scrutiny, and quarantine stale claims that should not be reused.

This document supersedes older Shakespeare prose reports that treated the lane as settled.

## Canonical Inputs

Use these artifacts as the only approved starting points for Shakespeare cleanup:

- `docs/forensic_methodology_critique.md`
- `reports/wright_vs_aspley_comparison/comparison_summary.json`
- `reports/wright_vs_aspley_80conf/comparison_summary.json`
- `reports/page_forensics/page_comparison.json`
- `reports/scan_wright_fixed/statistics.json`
- `reports/scan_aspley_fixed/statistics.json`
- `reports/wright_80conf/statistics.json`
- `reports/aspley_80conf/statistics.json`
- `reports/full_sonnet_mapping.json`

## Reusable Methods And Data

- The methodology critique remains valid as a control document. It correctly rejects the earlier habit of inferring typographic identity from averages alone.
- The Wright/Aspley comparison summaries remain useful as inventory artifacts because they document source coverage mismatch and OCR-count deltas.
- The page-forensics JSON remains useful as a diagnostic artifact because it shows that page-by-page variance is widespread and cannot be hand-waved away.
- The surviving scan statistics remain useful as extraction baselines, not as attribution proof.

## Tentative Findings Worth Preserving

- Source coverage mismatch is real and unresolved in the older cluster. The comparison summaries still show a 53-page Wright source against a 67-page Aspley source, with a 14-page delta driven by different digitization coverage.
- OCR-count deltas are large enough that simplistic "same text, same type, just scan quality" conclusions are not trustworthy without better controls.
- Long-s and ligature counts diverge materially across the old runs and therefore remain signals worth rechecking under a cleaned workflow.
- The page-forensics dataset reports `40` significant pages out of `53`, which is too much variance to dismiss without more careful alignment and per-page validation.
- Full sonnet mapping remains incomplete. The current mapping summary reports only `4` Wright and `16` Aspley finds in the surviving JSON, which means the alignment tooling is not yet mature enough to support strong claims.

## Claims That Are Archived And Should Not Survive

- Any claim that the Wright and Aspley witnesses are already proven typographically identical.
- Any claim that the null hypothesis has been accepted and the alternative hypothesis rejected.
- Any claim that the observed differences are fully explained by OCR quality without explicit controls.
- Any prose report that treats the Shakespeare lane as a finished attribution case.

These archived claims remain preserved for audit history only.

## Superseded Legacy Outputs

Treat the following as historical-only unless a future cleanup explicitly regenerates them from a controlled workflow:

- `reports/page_forensics/FORENSIC_ANALYSIS_REPORT.md`
- `reports/wright_vs_aspley_comparison/`
- `reports/wright_vs_aspley_80conf/`
- `reports/scan_wright_fixed/`
- `reports/scan_aspley_fixed/`
- `reports/wright_80conf/`
- `reports/aspley_80conf/`

## If Shakespeare Is Reactivated Later

The next cleanup pass should do these things in order:

1. Normalize source coverage so the compared corpora are page- and witness-equivalent.
2. Rebuild extraction and comparison under one consistent OCR and persistence workflow.
3. Add explicit control comparisons before attributing variance to scan quality.
4. Generate pair-level visual review artifacts before any prose verdict.
5. Only then decide whether the lane deserves a cautious shareable summary.

Until that happens, Shakespeare is considered complete enough only as an internal control packet.
