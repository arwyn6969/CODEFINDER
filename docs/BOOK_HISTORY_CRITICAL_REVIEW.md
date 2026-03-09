# Critical Review of the March 8, 2026 Kempten Draft

> Scope: `docs/BOOK_HISTORY_MANUSCRIPT.md`, `docs/BOOK_HISTORY_METHODS_APPENDIX.md`, `reports/final_report/`, `reports/shareable/`, `reports/proof_images/`, and the Greenman/manual-review artifacts.
> Audience: skeptical external book-history readers.
> Goal: red-team critique plus an actionable revision brief.

## Executive Assessment

The manuscript itself is materially stronger than the downstream report package built around it. `docs/BOOK_HISTORY_MANUSCRIPT.md` is cautious, keeps the strongest claim provisional, and repeatedly distinguishes similar printed forms from identical physical sorts or blocks. The main publication risks sit elsewhere:

1. the image package is not yet persuasive enough for skeptical external readers;
2. the Greenman/foliate-head evidence is presented with more confidence in the PDF/shareable layer than the underlying artifact set can bear;
3. several public-facing figures are generated from convenience selections rather than argument-led selections.

The result is a split signal. The manuscript reads like a careful computational finding-aid. The PDF/shareable layer sometimes reads like a more settled evidentiary packet. That mismatch is the main thing to fix before wider circulation.

## Ranked Findings

### 1. The Greenman figure package is not publication-safe in its current form

This is the highest-risk issue because it sits on the weakest evidentiary line in the whole packet. The manuscript says the ornament evidence is source-specific and does not support multi-source woodblock reuse. That caution is correct. The figure layer muddies it.

- The verified result set contains exactly one accepted match, in GDZ only: `reports/greenman_scan/matches.json`.
- The review-sheet generator places a reference crop, a verified GDZ crop, and a below-threshold control side by side: `scripts/research/generate_greenman_review_sheet.py`.
- The result is visually confusing because the reference and verified panels are near-indistinguishable at review scale. Even if that is computationally expected, it does not teach an outside reader what the claimed evidentiary step actually is.
- The public PDF escalates the language further. `scripts/generate_pdf_report.py` says SIFT lines "demonstrate physical identity" and that high-density matches support a "strong local identification." That phrasing is too strong for a single-source, reference-driven match.
- The Discord/shareable PDF compounds the problem by captioning the left panel as a "Reference crop" even though the embedded file is `reports/proof_images/greenman_crop_gdz.jpg`, i.e. the verified GDZ crop selected for public proof-image use: `scripts/generate_discord_summary.py`.

### 2. The character proof figures do not currently prove what the captions say they prove

This is a more serious issue than simple aesthetics. The proof-image generator does not select argument-relevant character pairs. It selects the first available crop for each character/source combination.

- `scripts/prepare_proof_images.py` uses `LIMIT 1` with no score ordering or pairwise review linkage for the public character composites.
- The resulting images are tiny strips such as `189x40`, `58x40`, and `121x40`, which are not adequate for serif, contour, or wear inspection by a skeptical reader.
- The PDF then says "Identical type produces identical character shapes, serifs, and proportions" and presents these strips as "actual character crops compared." That overstates both what is shown and how the crops were selected.
- The manual-review contact sheets are better evidence than the public proof strips because they at least preserve rank, score, page, and bbox context.

### 3. Figure 1 uses inconsistent and sometimes non-representative source pages

The source-page montage looks authoritative, but the selection logic is ad hoc.

- `scripts/prepare_proof_images.py` hard-codes `000_0001.jpg` for BSB, `001_.jpg` for GDZ, `00003.jpg` for HAB, and `page-07.jpg` for Google Books, then falls back to the first JPEG if those paths miss.
- The actual BSB image in the public packet is essentially a sparse endpaper/date page, not a useful specimen of the printed forms under discussion.
- GDZ and HAB are title-page-like images. Google Books is not aligned to the same document function as the other two.
- The caption "Representative pages from each of the four source publications" is therefore too neutral. The current montage is not representative in a controlled bibliographical sense.

### 4. The downstream PDF/report wording is more assertive than the manuscript

The paper and the report package are not presently matched for claim discipline.

- The manuscript says the results are a computational finding aid, not a substitute for analytical bibliography.
- The final PDF says SIFT lines demonstrate physical identity and that identical type produces identical forms.
- The shareable summary is somewhat better, but still includes a caption mismatch around the Greenman crop/match pairing.

If the manuscript is the canonical external draft, the downstream outputs need to inherit its caution, not quietly outrun it.

### 5. The negative-control blocker remains real and must stay visible

This is not a writing flaw; it is a genuine methodological constraint that the manuscript is right to foreground.

- `reports/negative_control/negative_control_search_log.md` shows three external BSB candidates rejected under the present sort thresholds.
- `reports/negative_control/sort_metric_calibration.md` shows `231` weight combinations tested and zero eligible external-control acceptances.
- This means the current sort discriminator is still too permissive for publication-grade control work.

The manuscript handles this well. The risk is not under-disclosure inside the draft. The risk is that an external reader will see the polished figures first and underestimate how live this blocker still is.

### 6. Source metadata normalization is improved but not finished

The repo is candid about this, and the external research confirms it.

- The BSB record clearly identifies the Hans Sachs volume, the 1616 date, the call number `4 P.o.germ. 176 i-4`, and the Kempten printing line under Christoff Krausen.
- The Google Books record confirms `Tractatus brevis: continens decem principia doctrinae christianae`, Antonio Albizzi, publisher line `Kraus, 1613`, and original from the Bavarian State Library.
- The HAB digital library menu metadata gives a fuller catalog description than the current local shorthand and explicitly reads `Campidoni : Kraus, 1616`.

This does not break the draft, but it does mean the corpus table still carries some locally normalized phrasing that should be reconciled against catalog records before submission.

## Claim-to-Evidence Matrix

| Strong claim in the manuscript | Status | Evidence trail | Review note |
| --- | --- | --- | --- |
| The cleaned rerun processed `1,237` pages and `27,713` extracted characters across four sources. | supported | `reports/final_report/summary.txt`; `docs/BOOK_HISTORY_METHODS_APPENDIX.md` section 1 | Cleanly supported and repeated consistently across the packet. |
| One verified Greenman candidate exists in GDZ only under current thresholds. | supported | `reports/greenman_scan/matches.json`; `reports/manual_review/greenman_review.md` | Supported as a result-state claim. The presentation layer is weaker than the underlying count claim. |
| The current Greenman evidence does not support multi-source woodblock reuse. | supported | absence of accepted non-GDZ matches in `reports/greenman_scan/matches.json`; Appendix section 4 | Correctly cautious and should stay that way. |
| Character-sort averages remain in the `0.621-0.709` range. | supported | `reports/final_report/summary.txt`; `reports/final_report/final_report.html`; Appendix section 7 | Numerically consistent across local artifacts. |
| Manual review of `60` rows produced `29` same-design, `2` possible same-sort/block, `29` inconclusive. | supported | `reports/manual_review/manual_review_summary.md`; `reports/manual_review/manual_review_ledger.md` | Strong internal support. Still not a substitute for external specialist review. |
| Formal KS and chi-squared tests remain `DIFFERENT` across all source pairs while bootstrap similarity remains high. | supported | `reports/final_report/final_report.html`; `reports/statistical_analysis/stats_report.html`; Appendix section 7 | This is one of the better-supported balancing claims in the packet. |
| The damage-evolution rerun is `INSUFFICIENT`, with only `1/5` metrics increasing chronologically. | supported | `reports/damage_evolution/damage_report.html`; Appendix section 2 | Supported and correctly treated as diagnostic only. |
| No publication-grade negative control has yet been accepted. | supported | `reports/negative_control/negative_control_search_log.md`; `reports/negative_control/sort_metric_calibration.md` | Strongly supported and should remain prominent. |
| The evidence is best understood as consistent with shared or transferred printing materials, but not sufficient to prove identical physical sorts or blocks. | partially supported | cumulative reading of sort report, manual review, Greenman scan, and control failure | Defensible as a cautious synthesis, but still an interpretive synthesis rather than a direct demonstration. |
| The corpus may reflect a shared printing network centered on Kempten and the Kraus/Krause/Krausen shop tradition. | needs specialist confirmation | manuscript discussion plus catalog metadata | Plausible, but this is where computational evidence and bibliographical history meet. It needs external book-historical review. |
| The paper should be read as a computational finding aid, not a substitute for analytical bibliography. | supported | manuscript framing; DHQ article on printers' ornaments; Appendix section 4 | This is the right framing and should be imported into every downstream artifact. |

## External Research Notes

### Official source records

- BSB official record: [MDZ / bsb10057380](https://www.digitale-sammlungen.de/en/details/bsb10057380)
  - Confirms the long Hans Sachs title, the 1616 date, the holding institution, call number `4 P.o.germ. 176 i-4`, and a publication statement that includes both Augsburg distribution and Kempten printing by Christoff Krausen.
- GDZ official record: [GDZ / PPN777246686](https://gdz.sub.uni-goettingen.de/id/PPN777246686)
  - The host resets intermittently from this workspace, but the official record URL and the repo's preserved title-page image remain aligned with the local description of `Vita, Non Vita` and the 1609 GDZ witness.
- HAB official record family: [HAB digital object](https://diglib.hab.de/drucke/178-1-theol-1s/start.htm?image=00001)
  - The old frameset viewer is fragile, but the official menu metadata exposed by the site gives a fuller catalog line ending with `Campidoni : Kraus, 1616`. That should be reconciled against the shorter local label before submission.
- Google Books official record: [Google Books / uThoAAAAcAAJ](https://books.google.com/books?id=uThoAAAAcAAJ)
  - Confirms the title `Tractatus brevis: continens decem principia doctrinae christianae`, author Antonio Albizzi, publisher line `Kraus, 1613`, and original from the Bavarian State Library.

### Methodological scholarship

- Hazel Wilkinson, James Briggs, and Dirk Gorissen, [Computer Vision and the Creation of a Database of Printers' Ornaments](https://www.digitalhumanities.org/dhq/vol/15/1/000491/000491.html)
  - This is the closest methodological analogue in the current packet.
  - The article supports the manuscript's strongest instinct: computer vision can extend bibliographical work, but it does not remove the need for domain judgment about repetition, copying, reuse, and attribution.

### Terminology note

- Inference from the packet plus the ornament literature: `Greenman` is vivid but may be too interpretive unless the paper is making an iconographic claim.
- The manuscript already uses the parenthetical `foliate head`.
- For external readers, `foliate-head ornament` or `foliate-head device` is probably the safer lead term, with `Greenman` retained only as a working nickname if needed.

## Revision Brief

### Immediate prose fixes

1. Keep the manuscript's caution as the canonical tone and revise the downstream PDF/shareable language to match it.
2. Replace phrases like "demonstrate physical identity" and "Identical type produces identical character shapes" with language about strong local similarity, ranked comparison, and manual-review prompts.
3. Add one plain-language sentence early in the paper explaining what the reader should understand by `same design`, `possible same sort/block`, and `inconclusive`.
4. Change the lead ornament label from `Greenman` to `foliate-head ornament` in the main prose, with `Greenman` retained as an internal shorthand if desired.

### Immediate figure fixes

1. Replace Figure 1 with aligned page functions across all four witnesses: either all title pages or all text-bearing openings.
2. Replace the current public Greenman crop with a contextual page image showing the bounding box, then use a second inset for the crop itself.
3. Move the dense SIFT overlay to appendix/supplement status or redraw it with a handful of annotated correspondences rather than a full field of lines.
4. Replace the current character strips with crops taken from the reviewed top-ranked pairs, not arbitrary `LIMIT 1` selections.
5. If the Greenman review sheet is circulated, relabel it so the reader can see exactly what is reference, what is accepted candidate, what is below-threshold control, and why those panels are expected to look similar.

### Deeper blockers that should remain visible

1. The paper still lacks an accepted publication-grade negative control.
2. The current sort discriminator still needs redesign or tightening before source expansion.
3. The strongest image-level claims still need external bibliographical review.
4. HAB and, to a lesser extent, other source metadata still need catalog-normalized citation cleanup.

## Bottom Line

The manuscript is close to being a credible cautious discussion draft. The figure package is not. The right next step is not a dramatic rewrite of the paper's core caution; it is to bring the public evidence layer up to the same standard of restraint and precision that the manuscript already mostly maintains.
