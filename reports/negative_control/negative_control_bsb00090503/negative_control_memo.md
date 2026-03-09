# Negative Control Comparison Memo

- Control source: `negative_control_bsb00090503`
- Database: `data/negative_control_bsb00090503.db`
- Output directory: `reports/negative_control/negative_control_bsb00090503`
- Extracted page scope: `first 61 pages`
- Sort metric: `sort_metric_v1_0_30_40_30` (cosine=0.30, avg_fingerprint=0.40, dimension=0.30)
- Extracted characters in control source: `705`
- Greenman matches in control scan: `0`
- Median pairwise sort average: `0.647`
- Max pairwise sort average: `0.691`
- Publication-grade status: `REJECTED`

## Sort similarity against the German corpus

| Pair | Avg score | Characters compared |
| --- | ---: | ---: |
| `bsb_munich_10057380 vs negative_control_bsb00090503` | 0.691 | 14 |
| `gdz_goettingen_ppn777246686 vs negative_control_bsb00090503` | 0.611 | 12 |
| `google_books_tractatus_brevis vs negative_control_bsb00090503` | 0.584 | 10 |
| `hab_wolfenbuettel_178_1_theol_1s vs negative_control_bsb00090503` | 0.684 | 14 |

## Statistical checks involving the control source

| Pair | KS width | KS height | Chi-squared | Bootstrap mean |
| --- | --- | --- | --- | --- |
| `bsb_munich_10057380 vs negative_control_bsb00090503` | DIFFERENT | DIFFERENT | DIFFERENT | 0.842 |
| `gdz_goettingen_ppn777246686 vs negative_control_bsb00090503` | DIFFERENT | DIFFERENT | DIFFERENT | 0.889 |
| `google_books_tractatus_brevis vs negative_control_bsb00090503` | DIFFERENT | DIFFERENT | DIFFERENT | 0.846 |
| `hab_wolfenbuettel_178_1_theol_1s vs negative_control_bsb00090503` | DIFFERENT | DIFFERENT | DIFFERENT | 0.708 |

## Acceptance checks

- `0` Greenman matches: `PASS`
- All KS and chi-squared verdicts `DIFFERENT`: `PASS`
- Median pairwise sort average `< 0.60`: `FAIL`
- No pairwise sort average exceeds `0.65`: `FAIL`

## Interpretation

This run does not satisfy the current publication-grade thresholds for a negative control.
Preserve it as audit history, but do not cite it as the accepted scholarly comparator.