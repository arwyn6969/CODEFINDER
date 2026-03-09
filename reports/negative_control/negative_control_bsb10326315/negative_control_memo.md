# Negative Control Comparison Memo

- Control source: `negative_control_bsb10326315`
- Database: `data/negative_control_bsb10326315.db`
- Output directory: `reports/negative_control/negative_control_bsb10326315`
- Extracted page scope: `first 61 pages`
- Sort metric: `sort_metric_v1_0_30_40_30` (cosine=0.30, avg_fingerprint=0.40, dimension=0.30)
- Extracted characters in control source: `3144`
- Greenman matches in control scan: `0`
- Median pairwise sort average: `0.626`
- Max pairwise sort average: `0.676`
- Publication-grade status: `REJECTED`

## Sort similarity against the German corpus

| Pair | Avg score | Characters compared |
| --- | ---: | ---: |
| `bsb_munich_10057380 vs negative_control_bsb10326315` | 0.676 | 20 |
| `gdz_goettingen_ppn777246686 vs negative_control_bsb10326315` | 0.605 | 16 |
| `google_books_tractatus_brevis vs negative_control_bsb10326315` | 0.579 | 12 |
| `hab_wolfenbuettel_178_1_theol_1s vs negative_control_bsb10326315` | 0.647 | 18 |

## Statistical checks involving the control source

| Pair | KS width | KS height | Chi-squared | Bootstrap mean |
| --- | --- | --- | --- | --- |
| `bsb_munich_10057380 vs negative_control_bsb10326315` | DIFFERENT | DIFFERENT | DIFFERENT | 0.845 |
| `gdz_goettingen_ppn777246686 vs negative_control_bsb10326315` | DIFFERENT | DIFFERENT | DIFFERENT | 0.802 |
| `google_books_tractatus_brevis vs negative_control_bsb10326315` | DIFFERENT | DIFFERENT | DIFFERENT | 0.843 |
| `hab_wolfenbuettel_178_1_theol_1s vs negative_control_bsb10326315` | DIFFERENT | DIFFERENT | DIFFERENT | 0.985 |

## Acceptance checks

- `0` Greenman matches: `PASS`
- All KS and chi-squared verdicts `DIFFERENT`: `PASS`
- Median pairwise sort average `< 0.60`: `FAIL`
- No pairwise sort average exceeds `0.65`: `FAIL`

## Interpretation

This run does not satisfy the current publication-grade thresholds for a negative control.
Preserve it as audit history, but do not cite it as the accepted scholarly comparator.