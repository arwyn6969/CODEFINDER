# Provisional Computational Evidence for Shared Printing Materials in Four Early Seventeenth-Century Kempten Imprints

> Status: External-facing working draft for book-history readers
> Date: March 8, 2026
> Claim level: Provisional computational evidence, not a settled attribution claim

## Abstract

This study tests whether a small computational comparison can help prioritize bibliographical review across four early seventeenth-century imprints associated with Kempten and the Kraus/Krause/Krausen printing network. A cleaned rerun of the current pipeline processed 1,237 page images and 27,713 extracted characters across four sources. Character-sort comparison returns consistently high pairwise similarity scores across the available corpus slices (0.621-0.709), while bootstrap resampling likewise shows high mean similarity (0.798-0.982). At the same time, formal distributional tests remain divergent across all source pairs, and ornament evidence is materially weaker than sort evidence: the current foliate-head ornament scan, referred to in some project notes as the Greenman scan, verifies one candidate in the GDZ source only and does not support a claim of multi-source woodblock reuse. The present evidence is therefore best understood as consistent with shared or transferred printing materials, especially at the level of type design and shop practice, but not sufficient to prove reuse of identical physical sorts or blocks. The results should be read as a computational finding aid for further bibliographical inspection, not as a substitute for manual analytical bibliography.

## Research Question and Historiographical Context

The narrow question of this draft is whether computational comparison can identify evidence consistent with shared printing materials across a small corpus of Kraus/Krause-related imprints. The aim is not to replace analytical bibliography with automation. Rather, it is to use OCR-derived character inventories, ornament detection, and image-level comparison to identify where a bibliographer should look next.

That framing matters. Classical bibliography still requires attention to material evidence, printing-house practice, and the social conditions of textual production (Gaskell; McKenzie). Recent digital work on printers' ornaments has shown that computer vision can usefully extend, but not displace, established bibliographical method (Wilkinson, Briggs, and Gorissen). The present draft follows that narrower logic: it offers a computational triage layer for a corpus whose bibliographical connections already merit attention because multiple items are associated with Kempten and printers recorded as Christoph Kraus, Krause, or Krausen.

## Corpus and Provenance

The analysis uses four digital facsimiles currently held in the local CODEFINDER corpus.

| Source ID | Bibliographic description | Holding institution / record | Pages | Extracted chars | Provenance note |
| --- | --- | --- | ---: | ---: | --- |
| `bsb_munich_10057380` | Hans Sachs, *Sehr Herrliche, Schoene und Warhaffte Gedicht. 4, Das Vierdt Poetisch Buch...* (Augsburg; printed in Kempten by Christoff Krausen, 1616) | Bayerische Staatsbibliothek, Munich, shelfmark `4 P.o.germ. 176 i-4`; digital id `bsb10057380` | 782 | 15,320 | IIIF manifest and library metadata available locally |
| `gdz_goettingen_ppn777246686` | Jacobus Ruelichius, *Vita, Non Vita* (Kempten: Krause, 1609) | Goettinger Digitalisierungszentrum, record `PPN777246686` | 60 | 1,979 | IIIF manifest available locally |
| `google_books_tractatus_brevis` | Antonio degli Albizzi, *Tractatus brevis: continens decem principia doctrinae christianae* (Kempten, 1613) | Google Books, id `uThoAAAAcAAJ` | 97 | 1,843 | Local page images extracted from PDF facsimile |
| `hab_wolfenbuettel_178_1_theol_1s` | Antonio Albizzi, *Exercitationum theologicarum pars prima* (Kempten: Christophoro Kraus, 1616) | Herzog August Bibliothek, Wolfenbuettel, shelfmark `178-1-theol-1s` | 298 | 8,571 | Title normalized from local title-page image; full catalog normalization still needed |

These sources are not identical in acquisition path or image quality. BSB and GDZ arrive through IIIF manifests, HAB through HTTP image scraping, and Google Books through local extraction from a downloaded PDF. That heterogeneity is not a minor detail; it is one of the main reasons the conclusions below remain provisional.

## Methods

The current pipeline uses Tesseract OCR with the combined `frk+deu+eng` model, page normalization to 2400 px height, page segmentation mode `6`, and a minimum OCR confidence threshold of `50`. Ornament extraction uses grayscale conversion, Otsu binarization, contour detection, a minimum candidate area of `5000` pixels, a maximum aspect ratio of `4.0`, and a minimum ink-density threshold of `0.15`. The foliate-head ornament scan then compares ornament candidates to a fixed reference crop using SIFT feature matching plus aggregate block fingerprinting, with Lowe's ratio test at `0.7`, a minimum of `50` good SIFT matches, and a fingerprint threshold of `0.90`. Oversized, page-scale candidates are excluded.

Character-sort matching compares repeated letterforms across sources for the Latin alphabet plus selected German characters (eszett/`ss`, umlauted vowels, and long-s where available). A character enters comparison only when at least two sources contain at least three instances each. The matching report aggregates centroid similarity, sampled fingerprint comparison, and dimension similarity; the working threshold above `0.6` is treated here as `SIMILAR FORMS`. In this manuscript, that label indicates strong similarity in printed forms, not proof of identical physical sorts.

The manual-review labels used below are deliberately narrow. `same design` means that two forms plausibly derive from the same overall letterform design without establishing object identity; `possible same sort/block` marks a provisional same-object suspicion worth specialist inspection; and `inconclusive` marks pairs that cannot be read confidently at current scan quality.

Formal statistical testing adds Kolmogorov-Smirnov tests for dimension distributions, chi-squared tests for character-frequency distributions, Mann-Whitney U tests for non-parametric dimension comparison, and 1000 bootstrap resamples for similarity estimates. Those outputs are read together, not in isolation.

## Results

### 1. Inventory and Corpus Totals

The cleaned rerun processed 1,237 pages and 27,713 extracted characters. The BSB corpus dominates the page count, while the GDZ and Google Books slices remain comparatively small. Any cross-source reading must therefore account for uneven sample size.

### 2. Foliate-Head Ornament Findings

The ornament evidence is currently source-specific. Under the tightened thresholds now in use, the scan verifies one foliate-head ornament candidate in the GDZ source on page `9` (`009_1.jpg`). The current run does not verify corresponding foliate-head matches in HAB or BSB. Two earlier GDZ false positives were removed because their bounding boxes spanned most of the page and therefore could not represent discrete ornaments.

This matters interpretively. The present artifact set supports the statement "one verified candidate exists in GDZ under current thresholds." It does not support the stronger statement "the same woodblock recurs across the corpus."

### 3. Character-Sort Matching

Character-sort similarity is the strongest component of the current evidentiary stack. Across the pairwise report, average scores remain in the `0.621-0.709` range:

| Pair | Average similarity | Characters compared |
| --- | ---: | ---: |
| BSB vs GDZ | 0.688 | 25 |
| BSB vs Google Books | 0.663 | 13 |
| BSB vs HAB | 0.709 | 33 |
| GDZ vs Google Books | 0.621 | 12 |
| GDZ vs HAB | 0.669 | 18 |
| Google Books vs HAB | 0.660 | 13 |

These are meaningful similarities, especially given the mixed acquisition paths and languages in the corpus. But they do not by themselves resolve whether we are seeing the same physical sorts, closely related type designs, or a common shop aesthetic reproduced across separate casting histories.

Manual review of the current top `10` matches for each of the six source pairs has now been completed through the fixed ledger in `reports/manual_review/`. Across `60` reviewed rows, `29` were judged `same design`, `2` were judged `possible same sort/block`, and `29` remained `inconclusive`. That result supports the restrained reading of the sort report: the strongest evidence is for recurring design similarity, with only a small minority of image pairs warranting even provisional same-object suspicion. Put differently, the manual ledger presently supports many design-level resemblances, very few candidate same-object leads, and a large unresolved remainder.

### 4. Formal Statistical Tests

The formal tests complicate any attempt to make a simple sameness claim. All pairwise Kolmogorov-Smirnov comparisons in the current report return `DIFFERENT` for width and height distributions, and all chi-squared tests of character-frequency distributions likewise return `DIFFERENT`. Bootstrap similarity remains high, with mean values from `0.798` to `0.982`, and the Mann-Whitney effect sizes are mostly negligible. The best reading of this combination is not "the corpus is identical," but rather "some forms are strongly similar even though the distributions are not."

That distinction is central for a book-history audience. Bootstrap similarity can indicate robust formal resemblance; it cannot by itself establish common material origin.

### 5. Damage Evolution

The damage-evolution report has now been rerun against corrected local source dates: GDZ `1609`, Google Books `1613`, and both BSB and HAB `1616`. Under that corrected ordering, the current output is `INSUFFICIENT`, with only `1/5` metrics increasing chronologically.

That result is still useful, but mainly as a brake on overclaiming. At present the damage artifact does not strengthen the main argument; it narrows it. The report remains diagnostic only and should stay out of any external verdict language unless a later rerun on reconciled, catalog-confirmed metadata produces a materially different result.

## Discussion

The evidence is most persuasive when framed modestly. The character-sort report suggests strong cross-source similarity in printed forms, and the provenance data make it plausible that those similarities emerge from a shared printing network centered on Kempten and the Kraus/Krause/Krausen shop tradition. That is enough to justify further manual bibliographical work.

It is not enough to collapse distinct explanations into one. At least four interpretations remain in play:

1. Shared physical sorts circulated across the corpus.
2. Different instances of the same or closely related type design were used across associated jobs.
3. OCR normalization and scan heterogeneity exaggerate visual similarity for some characters.
4. Textual and linguistic differences alter character-frequency distributions even when material practices overlap.

The current results fit best with a restrained version of the first two explanations together: the corpus may preserve shared or transferred materials, but the stronger claim of identical sorts or multi-source woodblock reuse has not yet been demonstrated.

## Limitations and Validation Priorities

Three limitations block a stronger conclusion.

First, the corpus is heterogeneous in acquisition quality, scale, and metadata completeness. Second, the current study still lacks an accepted publication-grade negative control. A repo-local stress test using `folger_iiif_aspley` remains useful as internal caution only, and three BSB candidates (`negative_control_bsb10222478`, `negative_control_bsb10326315`, and `negative_control_bsb00090503`) have now been processed as `61`-page slices. All three yield zero foliate-head matches and `DIFFERENT` formal statistics, but all three fail the present sort thresholds, with median pairwise averages between `0.626` and `0.647` and maximum pairwise averages between `0.676` and `0.701`. A subsequent `0.05`-step calibration sweep across the current sort components also failed to identify any weight set that preserved all six German pairs above `0.60` while allowing even one external control to clear the acceptance rules, and a bounded GDZ/HAB fallback search did not surface a new screenable comparator. Third, the strongest image-level similarities have now been manually reviewed in a fixed ledger, but that internal review still needs specialist bibliographical confirmation before any stronger claim is made.

The next validation round should therefore do the following:

1. Have a specialist bibliographer review the completed `60`-row ledger and the `2` provisional `possible same sort/block` cases before any stronger material claim is circulated.
2. Keep the GDZ foliate-head candidate paired with the below-threshold BSB control documented in `reports/manual_review/greenman_review.md`; the current ornament result remains source-specific.
3. Tighten or redesign the current sort discriminator before widening source search further, because the preserved BSB control attempts and the completed calibration sweep both show that source choice alone is not resolving the false-similarity problem.
4. Keep damage chronology diagnostic-only unless a later rerun on reconciled metadata changes the result materially.

## Conclusion

The cleaned March 7, 2026 rerun supports a provisional shared-materials hypothesis across four Kempten-associated imprints. The strongest evidence lies in character-sort similarity and in the bibliographical plausibility of a shared printing network. The weakest evidence lies in the current foliate-head ornament comparison, which verifies one GDZ candidate only, and in the current damage-ordering report, which still depends on unresolved date metadata.

This is therefore a plausible computational draft for scholarly discussion, not a finished attribution argument. A skeptical book historian should be able to read the present claim as promising, provisional, and materially grounded, while still insisting on specialist analytical bibliography and a stronger accepted negative control before any stronger statement is made.

## Selected References

- Philip Gaskell, *A New Introduction to Bibliography* (Oxford: Clarendon Press, 1972).
- D. F. McKenzie, *Bibliography and the Sociology of Texts* (London: British Library, 1999).
- Hazel Wilkinson, James Briggs, and Dirk Gorissen, "Computer Vision and the Creation of a Database of Printers' Ornaments," *Digital Humanities Quarterly* 15, no. 1 (2021).
- Simone Maghenzani and Massimo Firpo, "Antonio degli Albizzi and Lutheran Propaganda in Early Seventeenth-Century Italy," *Journal of Ecclesiastical History* 73, no. 2 (2022): 275-307.

## Digital Source Records

- BSB Munich record: `https://mdz-nbn-resolving.de/details:bsb10057380`
- GDZ record: `http://resolver.sub.uni-goettingen.de/purl?PPN777246686`
- HAB record: `http://diglib.hab.de/drucke/178-1-theol-1s/start.htm`
- Google Books record: `https://books.google.com/books?id=uThoAAAAcAAJ`

For the reproducibility appendix, artifact map, and validation checklist, see `docs/BOOK_HISTORY_METHODS_APPENDIX.md`.
