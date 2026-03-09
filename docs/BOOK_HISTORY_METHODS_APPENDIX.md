# Methods and Reproducibility Appendix for the Book-History Draft

> Status: Companion appendix to `docs/BOOK_HISTORY_MANUSCRIPT.md`
> Date: March 8, 2026
> Scope: Corpus inventory, pipeline parameters, remediation log, validation requirements, and claim-to-artifact mapping

## 1. Corpus Inventory

| Source ID | Short citation | Holding institution / identifier | Acquisition mode | Pages | Extracted chars |
| --- | --- | --- | --- | ---: | ---: |
| `bsb_munich_10057380` | Hans Sachs, *Sehr Herrliche, Schoene und Warhaffte Gedicht. 4...* (1616) | Bayerische Staatsbibliothek, `bsb10057380`, shelfmark `4 P.o.germ. 176 i-4` | IIIF manifest | 782 | 15,320 |
| `gdz_goettingen_ppn777246686` | Jacobus Ruelichius, *Vita, Non Vita* (1609) | GDZ, `PPN777246686` | IIIF manifest | 60 | 1,979 |
| `google_books_tractatus_brevis` | Antonio degli Albizzi, *Tractatus brevis* (1613) | Google Books, `uThoAAAAcAAJ` | PDF page extraction | 97 | 1,843 |
| `hab_wolfenbuettel_178_1_theol_1s` | Antonio Albizzi, *Exercitationum theologicarum pars prima* (1616) | HAB Wolfenbuettel, `178-1-theol-1s` | HTTP image scrape | 298 | 8,571 |
| `TOTAL` | Corpus total | - | - | 1,237 | 27,713 |

### Bibliographical notes

- The BSB item is locally documented through the IIIF manifest as a Hans Sachs volume printed in Kempten by Christoff Krausen in 1616.
- The GDZ item is locally documented through the manifest as Jacobus Ruelichius, *Vita, Non Vita* (Kempten: Krause, 1609).
- The HAB item was previously under-described in local config. The local title-page image identifies it as an Antonio Albizzi theological volume printed in Kempten by Christophoro Kraus in 1616.
- The Google Books item remains described through the local source config and record id as Antonio degli Albizzi, *Tractatus brevis: continens decem principia doctrinae christianae* (1613).
- Printer names appear in variant catalog forms (`Kraus`, `Krause`, `Krausen`). Authority control should be normalized before submission.

## 2. Processing Configuration

### OCR extraction

- Script: `scripts/extract_characters.py`
- OCR model: `frk+deu+eng`
- Page segmentation mode: `6`
- OCR engine mode: `1`
- Minimum OCR confidence: `50`
- Default page normalization height: `2400`
- Character scope later used in sort matching: A-Z, a-z, eszett/`ss`, `a/o/u` umlaut forms, and long-s where available

### Ornament extraction

- Script: `scripts/extract_ornaments.py`
- Service: `app/services/ornament_extractor.py`
- Preprocessing: grayscale conversion, Otsu binarization, morphological closing, contour detection
- Minimum ornament candidate area: `5000` pixels
- Maximum aspect ratio: `4.0`
- Minimum ink density: `0.15`
- Purpose: identify candidate ornamental blocks for later comparison

### Foliate-head scan

- Script: `scripts/scan_greenman_all.py`
- Reference method: SIFT feature matching plus aggregate block fingerprint comparison
- Lowe ratio threshold: `0.7`
- Minimum good SIFT matches: `50`
- Fingerprint threshold: `0.90`
- Candidate viability filter: reject candidates outside configured area bounds and reject oversized page-scale artifacts
- Current verified output: one GDZ candidate on page `9`

### Character-sort matching

- Script: `scripts/match_character_sorts.py`
- Minimum instances per source to compare a character: `3`
- Fingerprint match threshold: `0.55`
- Pairwise report verdict threshold: averages above `0.6` labelled `SIMILAR FORMS`
- Current scholarly interpretation: useful indicator of similar printed forms, not proof of identical physical sorts

### Formal statistics

- Script: `scripts/formal_stats.py`
- Tests used: Kolmogorov-Smirnov, chi-squared, Mann-Whitney U, bootstrap
- Bootstrap iterations: `1000`
- Reporting standard: read distributional tests and bootstrap together, not bootstrap in isolation

### Damage evolution

- Script: `scripts/damage_evolution.py`
- Current output after chronology correction: `INSUFFICIENT`, with `1/5` metrics increasing in the corrected `1609 -> 1613 -> 1616 -> 1616` source order
- Current scholarly status: diagnostic only and excluded from the external verdict

## 3. March 7, 2026 Remediation Log

The March 7 rerun was not a simple rerun. It required several correctness fixes before the outputs became internally coherent enough for a scholarly draft.

1. Report generators were repointed from stale corpus wiring to the German/Fraktur corpus database so that corpus totals and evidentiary summaries described the right dataset.
2. The foliate-head scan was hardened against false positives by restoring the `0.90` fingerprint threshold and filtering out page-sized ornament candidates.
3. `scripts/extract_characters.py` was aligned to its actual OCR stack so the default model and the script documentation both use `frk+deu+eng`.
4. Proof-image preparation was pointed back at the German corpus and its crop handling was corrected so the report illustrations came from the right sources.
5. Page-number parsing was corrected for GDZ and BSB naming patterns, which affected rerun consistency and page-level references.
6. Zero-hit OCR pages were saved cleanly without leaving stale sort-image artifacts behind, improving rerun hygiene.
7. Cached ornament candidates were loaded from the correct directory during the foliate-head pass.
8. Proof-image crop resolution was corrected so image evidence in the report was not degraded by the wrong sizing assumptions.
9. The character-sort workflow was tightened to the intended Latin/German alphabet instead of drifting into irrelevant symbols.
10. The BSB extraction pass was rerun cleanly after sharded processing resolved earlier extraction incompleteness.

## 4. Current Evidentiary Position

The present stack supports the following restrained claims:

- The corpus inventory is now internally coherent at `1,237` pages and `27,713` characters.
- Character-sort similarity is consistently high enough to justify further material comparison.
- Formal distributional tests still show that the sources are not statistically identical.
- Foliate-head evidence is source-specific under the present thresholds.
- Damage evolution has now been rerun against corrected local source dates and remains unsuitable as chronological evidence.

The stack does not support the following claims:

- "The same woodblock has been demonstrated across multiple sources."
- "Bootstrap similarity proves shared physical sorts."
- "The corpus has already passed expert bibliographical validation."
- "Chronological wear has been established across the four sources."

## 5. Manual Validation Protocol

Before external submission or circulation to specialists, the next review phase should follow a fixed manual protocol.

1. For each high-similarity source pair, inspect the top `10-20` individual character matches from the sort report.
2. Review image pairs at high zoom, not just aggregate scores.
3. Classify each reviewed item as `same design`, `possible same sort/block`, or `inconclusive`.
4. For ornaments, inspect the verified GDZ foliate-head candidate against at least one negative-control ornament that the pipeline ranked below threshold.
5. Record the page, bounding box, and rationale for each expert judgment.
6. Separate "same type design" from "same physical sort" in all manual annotations.

Current completion status:

- Manual-review ledger completed: `reports/manual_review/manual_review_ledger.md`
- Structured export files completed: `reports/manual_review/manual_review_ledger.csv` and `reports/manual_review/manual_review_ledger.json`
- Canonical closeout summary completed: `reports/manual_review/manual_review_summary.md`
- Pairwise contact sheets generated for all six source pairs under `reports/manual_review/sheets/`
- Foliate-head review note generated: `reports/manual_review/greenman_review.md` with companion JSON and review sheet
- Current review totals: `29` rows `same design`, `2` rows `possible same sort/block`, `29` rows `inconclusive`

Working definitions for that review:

- `same design`: the letterform or ornament design is convincingly similar, but the evidence does not isolate a shared physical object.
- `possible same sort/block`: design similarity plus localized wear, contour, or damage correspondences that could reflect a shared physical matrix.
- `inconclusive`: the image pair is too noisy, too sparse, or too resolution-dependent to support either of the above judgments.

## 6. Negative-Control Search Status

The present draft still lacks an accepted publication-grade negative-control corpus result in the book-history packet. That remains the main unresolved blocker before serious scholarly circulation.

A repo-local control workflow remains available at `scripts/maintenance/run_negative_control.py`. In addition, three BSB candidates have now been processed as `61`-page slices and preserved as rejected audit history:

1. `negative_control_bsb10222478`
2. `negative_control_bsb10326315`
3. `negative_control_bsb00090503`

All three rejected BSB candidates satisfy the ornament and formal-statistics conditions:

- `0` foliate-head matches
- all KS and chi-squared verdicts `DIFFERENT`

All three fail the current sort-similarity conditions:

- median pairwise sort averages remain between `0.626` and `0.647`
- maximum pairwise sort averages remain between `0.676` and `0.701`

A dedicated calibration pass was then run over the three current score components (`cosine_similarity`, `avg_fingerprint_score`, `dimension_similarity`) in `0.05` weight steps:

- calibration artifact: `reports/negative_control/sort_metric_calibration.md`
- grid size: `231` weight combinations
- eligible combinations that preserved the German corpus constraints: `178`
- external controls accepted by any eligible combination: `0`

Best non-accepting calibration attempt:

- weights: `cosine=0.00, avg_fingerprint=0.75, dimension=0.25`
- weakest German pair: `0.603`
- strongest external negative-control pair: `0.656`

This means the current blocker is not reducible to source choice alone; the present component set and acceptance rules still leave the sort discriminator too permissive.

A bounded fallback search was then documented in `reports/negative_control/fallback_source_search.md`:

1. GDZ candidate `PPN832871516` reached candidate-identification stage, but the manifest host reset connections from the workspace during acquisition.
2. HAB candidate `ti-kapsel-2-13s` was rejected at metadata review because the bibliographic description identifies it as `1689`.
3. HAB candidate `229-19-theol-6s` was rejected at metadata review because the bibliographic description identifies it as `1662`.

Canonical log for those attempts:

- `reports/negative_control/negative_control_search_log.md`

Implication for the current draft:

- negative-control searching is now documented and auditable
- the draft is stronger than before because rejected controls are preserved, not forgotten
- the draft still does not have a publication-grade accepted comparator
- the calibration sweep now shows that metric insufficiency is an explicit part of the remaining blocker

## 7. Claim-to-Artifact Map

| Claim in manuscript | Artifact | Path | Use note |
| --- | --- | --- | --- |
| Corpus totals equal 1,237 pages and 27,713 chars | Final summary | `reports/final_report/summary.txt` | Primary inventory reference |
| One verified foliate-head candidate exists in GDZ only | Match manifest | `reports/greenman_scan/matches.json` | Source, page, bbox, and fingerprint data |
| Foliate-head result should remain source-specific | Foliate-head HTML report | `reports/greenman_scan/greenman_report.html` | Visual context and thresholded report |
| Sort similarity scores range from 0.621 to 0.709 | Sort report | `reports/character_sort_match/sort_report.html` | Pairwise summary and per-character detail |
| All KS and chi-squared pairwise tests are `DIFFERENT` | Statistics report | `reports/statistical_analysis/stats_report.html` | Distributional caution against overclaiming |
| Bootstrap similarity remains high | Statistics report | `reports/statistical_analysis/stats_report.html` | Read with the distributional tests, not alone |
| Damage report is diagnostic only after chronology correction | Damage report | `reports/damage_evolution/damage_report.html` | Keep out of the external verdict unless a later rerun changes the result |
| Manual review completed with conservative results | Manual-review summary | `reports/manual_review/manual_review_summary.md` | `29` same-design, `2` provisional same-sort/block, `29` inconclusive |
| Three BSB control candidates were rejected under the current thresholds | Negative-control search log | `reports/negative_control/negative_control_search_log.md` | Audit trail for `negative_control_bsb10222478`, `negative_control_bsb10326315`, and `negative_control_bsb00090503` |
| No `0.05`-step calibration formula resolves the current negative-control blocker | Calibration report | `reports/negative_control/sort_metric_calibration.md` | Shows `231` tested weight combinations and zero eligible external-control acceptances |
| External claim should remain provisional | Final report package | `reports/final_report/CODEFINDER_Forensic_Report.pdf` and `reports/final_report/final_report.html` | Current narrative synthesis |
| Manual review can proceed from fixed artifacts | Manual-review ledger | `reports/manual_review/manual_review_ledger.md` and `reports/manual_review/sheets/` | Review the top pairwise character matches at image level |
| Ornament control review is fixed to one verified match and one control candidate | Foliate-head review note | `reports/manual_review/greenman_review.md`, `reports/manual_review/greenman_review.json`, and `reports/manual_review/greenman_review_sheet.png` | Source-specific ornament review artifact for manual validation |

## 8. Caption-Ready Figure Notes

These captions are written to control over-interpretation in a scholarly draft.

- Figure 1. Selected early text-bearing pages from the four-source comparison set. These images establish visual context and acquisition comparability; they do not by themselves imply common material origin.
- Figure 2. Page context and extracted crop for the verified GDZ foliate-head candidate. This figure demonstrates one source-specific match only and should not be cited as evidence of corpus-wide woodblock reuse.
- Figure 3. Supplemental SIFT overlay for the verified GDZ foliate-head candidate. This figure is diagnostic support for Figure 2 and should not be used as an independent identity claim.
- Figure 4. Reviewed character-pair exemplars from the manual-review ledger. These panels distinguish `possible same sort/block` from `same design` and should be read as cautious image-level evidence only.
- Figure 5. Pairwise average character-form similarity by source pair. These values indicate strong formal resemblance in printed letterforms, but they do not independently prove identical physical sorts.
- Figure 6. Manual-review outcomes by source pair. The review balance shows that the current packet is dominated by `same design` and `inconclusive` outcomes rather than repeated same-object judgments.
- Figure 7. Bootstrap mean similarity with `95%` confidence intervals. High bootstrap similarity should be read alongside divergent distributional tests, not as standalone proof of shared material origin.
- Figure 8. Formal-test verdict matrix. The juxtaposition of KS, chi-squared, and Mann-Whitney readings is intended to prevent a simplistic same/different interpretation.

## 9. Submission Readiness Checklist

- [ ] Normalize all printer names and shelfmarks against library catalog records.
- [ ] Accept a true publication-grade negative-control source.
- [x] Document the current metric-calibration failure against the fixed positive/negative corpus.
- [x] Complete manual review of top sort matches and the verified GDZ ornament candidate.
- [x] Correct source dates in the damage-evolution workflow and rerun that report.
- [ ] Replace approximate or locally inferred metadata with catalog-confirmed citations where available.
- [ ] Ensure every strong sentence in the manuscript maps to one artifact in Section 7.

## 10. Selected Scholarship and Source Records

- Philip Gaskell, *A New Introduction to Bibliography* (Oxford: Clarendon Press, 1972).
- D. F. McKenzie, *Bibliography and the Sociology of Texts* (London: British Library, 1999).
- Hazel Wilkinson, James Briggs, and Dirk Gorissen, "Computer Vision and the Creation of a Database of Printers' Ornaments," *Digital Humanities Quarterly* 15, no. 1 (2021).
- Simone Maghenzani and Massimo Firpo, "Antonio degli Albizzi and Lutheran Propaganda in Early Seventeenth-Century Italy," *Journal of Ecclesiastical History* 73, no. 2 (2022): 275-307.
- BSB record: `https://mdz-nbn-resolving.de/details:bsb10057380`
- GDZ record: `http://resolver.sub.uni-goettingen.de/purl?PPN777246686`
- HAB record: `http://diglib.hab.de/drucke/178-1-theol-1s/start.htm`
- Google Books record: `https://books.google.com/books?id=uThoAAAAcAAJ`
