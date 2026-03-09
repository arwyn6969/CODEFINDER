# Negative Control Comparison Memo

- Control source: `folger_iiif_aspley`
- Database: `data/negative_control.db`
- Output directory: `reports/negative_control`
- Extracted page scope: `all available pages`
- Sort metric: `sort_metric_v1_0_30_40_30` (cosine=0.30, avg_fingerprint=0.40, dimension=0.30)
- Extracted characters in control source: `3339`
- Greenman matches in control scan: `0`
- Median pairwise sort average: `0.661`
- Max pairwise sort average: `0.718`
- Publication-grade status: `REJECTED`

## Sort similarity against the German corpus

| Pair | Avg score | Characters compared |
| --- | ---: | ---: |
| `bsb_munich_10057380 vs folger_iiif_aspley` | 0.718 | 45 |
| `folger_iiif_aspley vs gdz_goettingen_ppn777246686` | 0.646 | 23 |
| `folger_iiif_aspley vs google_books_tractatus_brevis` | 0.634 | 13 |
| `folger_iiif_aspley vs hab_wolfenbuettel_178_1_theol_1s` | 0.676 | 32 |

## Statistical checks involving the control source

| Pair | KS width | KS height | Chi-squared | Bootstrap mean |
| --- | --- | --- | --- | --- |
| `bsb_munich_10057380 vs folger_iiif_aspley` | DIFFERENT | DIFFERENT | DIFFERENT | 0.981 |
| `folger_iiif_aspley vs gdz_goettingen_ppn777246686` | DIFFERENT | DIFFERENT | DIFFERENT | 0.961 |
| `folger_iiif_aspley vs google_books_tractatus_brevis` | DIFFERENT | DIFFERENT | DIFFERENT | 0.976 |
| `folger_iiif_aspley vs hab_wolfenbuettel_178_1_theol_1s` | DIFFERENT | DIFFERENT | DIFFERENT | 0.828 |

## Acceptance checks

- `0` Greenman matches: `PASS`
- All KS and chi-squared verdicts `DIFFERENT`: `PASS`
- Median pairwise sort average `< 0.60`: `FAIL`
- No pairwise sort average exceeds `0.65`: `FAIL`

## Interpretation

This repo-local control is intended to estimate how much similarity the current workflow can produce on an unrelated seventeenth-century print already present in the workspace.
In the current run it yields zero Greenman matches, but the character-sort averages remain comparatively high; that means it is useful as an internal stress test, not as a clean publication-grade negative control.
Treat it as a baseline check, and plan to replace it with a deliberately chosen unrelated seventeenth-century comparator before scholarly submission.