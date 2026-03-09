# Sort Metric Calibration Report

This report tests the production sort metric against the fixed German/Kempten positive corpus and the preserved negative-control corpus in `0.05` weight steps.

## Baseline production metric

- Version: `sort_metric_v1_0_30_40_30`
- Weights: `cosine=0.30, avg_fingerprint=0.40, dimension=0.30`

### Baseline external negative-control status

| Control | External | Median pair avg | Max pair avg | Greenman | Distributional | Accepted |
| --- | --- | ---: | ---: | --- | --- | --- |
| `folger_iiif_aspley` | `no` | 0.661 | 0.718 | `PASS` | `PASS` | `FAIL` |
| `negative_control_bsb10222478` | `yes` | 0.636 | 0.701 | `PASS` | `PASS` | `FAIL` |
| `negative_control_bsb10326315` | `yes` | 0.626 | 0.676 | `PASS` | `PASS` | `FAIL` |
| `negative_control_bsb00090503` | `yes` | 0.647 | 0.691 | `PASS` | `PASS` | `FAIL` |

## Calibration outcome

- Weight combinations evaluated: `231`
- Constraint-satisfying combinations: `178`
- External controls accepted by any combination: `0`
- Chosen weights: `none`

No eligible weight set accepted any external negative control while keeping all six German pairs above `0.60` and within `0.08` of baseline.
Best non-accepting attempt: `cosine=0.00, avg_fingerprint=0.75, dimension=0.25`
- Weakest German pair under that attempt: `0.603`
- Strongest negative-control pair under that attempt: `0.663`

Conclusion: source choice is not the only blocker. Under the current component set and acceptance rules, metric or threshold insufficiency remains.

## Best-attempt negative-control table

| Control | External | Median pair avg | Max pair avg | Accepted |
| --- | --- | ---: | ---: | --- |
| `folger_iiif_aspley` | `no` | 0.646 | 0.663 | `FAIL` |
| `negative_control_bsb10222478` | `yes` | 0.627 | 0.656 | `FAIL` |
| `negative_control_bsb10326315` | `yes` | 0.610 | 0.633 | `FAIL` |
| `negative_control_bsb00090503` | `yes` | 0.605 | 0.636 | `FAIL` |
