# Negative-Control Search Log

This log records the publication-grade negative-control candidates tested against the German/Kempten corpus after the March 8, 2026 manual-review pass.

Acceptance rules for a publication-grade control:

- `0` Greenman matches
- all KS and chi-squared verdicts `DIFFERENT`
- median pairwise sort average `< 0.60`
- no pairwise sort average exceeds `0.65`

## Candidate outcomes

| Control source | Scope | Characters | Greenman | Median sort avg | Max sort avg | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `negative_control_bsb10222478` | first `61` pages | 4,047 | 0 | 0.636 | 0.701 | Rejected |
| `negative_control_bsb10326315` | first `61` pages | 3,144 | 0 | 0.626 | 0.676 | Rejected |
| `negative_control_bsb00090503` | first `61` pages | 705 | 0 | 0.647 | 0.691 | Rejected |

## Interpretation

All three BSB candidates satisfy the ornament and formal-statistics conditions, but all three fail the current sort-similarity thresholds. The failure mode is consistent: the present sort metric still treats these unrelated early modern witnesses as too close to the Kempten corpus to serve as a publication-grade control.

## Metric calibration outcome

The current production sort metric was then calibrated in `0.05` weight steps using:

- positives: the six canonical German/Kempten source pairs
- negatives: `folger_iiif_aspley` plus the three rejected BSB controls

Canonical calibration artifact:

- `reports/negative_control/sort_metric_calibration.md`

Result:

- `231` weight combinations evaluated
- `178` combinations kept all six German pairs above `0.60` and within `0.08` of baseline
- `0` external negative controls accepted by any eligible combination

Best non-accepting formula:

- `cosine=0.00, avg_fingerprint=0.75, dimension=0.25`
- weakest German pair remained `0.603`
- strongest external negative-control pair still reached `0.656`

Interpretation:

- source choice is not the only blocker
- under the current component set and acceptance rules, metric or threshold insufficiency remains

## Bounded fallback source-search pass

After the calibration failure, one bounded non-BSB search pass was completed and recorded in:

- `reports/negative_control/fallback_source_search.md`

The bounded pass did not yield a new screenable publication-grade control:

1. GDZ candidate `PPN832871516` reached candidate-identification stage, but the GDZ manifest host reset connections from the workspace during acquisition.
2. HAB candidate `ti-kapsel-2-13s` was rejected at metadata review because the bibliographic description identifies it as `1689`.
3. HAB candidate `229-19-theol-6s` was rejected at metadata review because the bibliographic description identifies it as `1662`.

Current repo status after this pass:

- the internal Shakespeare stress test remains available at `reports/negative_control/negative_control_memo.md`
- three BSB candidates are preserved as rejected audit history in namespaced folders
- no publication-grade negative control has been accepted yet
- the calibration sweep and bounded fallback search both point to the current sort discriminator as a live blocker

## Next selection guidance

The next pass should prioritize the metric before source expansion:

- tighten or redesign the sort discriminator so unrelated witnesses separate more cleanly
- only then resume source search with a materially different early modern roman/italic witness
