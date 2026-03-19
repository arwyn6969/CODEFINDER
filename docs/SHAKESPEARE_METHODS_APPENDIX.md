# Methods and Reproducibility Appendix for the Shakespeare Packet

> Status: Companion appendix to `docs/SHAKESPEARE_MANUSCRIPT.md`
> Date: March 9, 2026
> Scope: External-safe corpus inventory, normalization rules, surviving diagnostics, remediation log, validation requirements, and claim-to-artifact mapping

## 1. Corpus Inventory

| Source ID | Short citation | Holding institution / identifier | Acquisition mode | Local images | Shared-sequence status |
| --- | --- | --- | --- | ---: | --- |
| `folger_sonnets_1609` | *Shake-speares sonnets* (`STC 22353a`; Wright) | Folger Shakespeare Library | IIIF manifest -> local `2000px` JPG cache | 53 | Entire witness participates in the shared `53`-image sequence |
| `folger_sonnets_1609_aspley` | *Shake-speares sonnets* (`STC 22353`; Aspley) | Folger Shakespeare Library | IIIF manifest -> local `2000px` JPG cache | 67 | First `53` images participate in the shared sequence; `14` remain unmatched extras |
| `TOTAL` | Local witness inventory | - | - | 120 | Shared sequence = `53` paired images |

### Bibliographical notes

- The Shakespeare packet is intentionally limited to the two local Folger witnesses already stored in `data/sources/`.
- The Wright witness is the seller-imprint Wright issue (`STC 22353a`); the Aspley witness is the seller-imprint Aspley issue (`STC 22353`).
- The current local corpus caches both witnesses as `2000px` JPG downloads, even though native/full IIIF reacquisition remains available from the same manifests.
- The canonical page-equivalence manifest classifies `41` shared pairs as included folio-content images, `12` shared pairs as paratext-heavy control images, and `14` Aspley images as unmatched extras.
- No third Shakespeare witness is introduced in this pass.

## 2. Processing Configuration

### Source verification and normalization

- Canonical source metadata: `data/sources/config.yaml`
- Witness metadata records:
  - `data/sources/folger_sonnets_1609/source_metadata.json`
  - `data/sources/folger_sonnets_1609_aspley/source_metadata.json`
- Canonical page-normalization artifact: `reports/shakespeare/page_equivalence_manifest.json`
- Classification rules:
  - `included`: shared-sequence image with a signed folio opening or mixed folio/edge content
  - `excluded_paratext`: shared-sequence image dominated by cover, endleaf, or spine material
  - `unmatched`: source-specific extra capture outside the shared `53`-image sequence

### Whole-witness OCR inventory

- Source artifacts:
  - `reports/scan_wright_fixed/statistics.json`
  - `reports/scan_aspley_fixed/statistics.json`
  - `reports/scan_wright_fixed/character_frequency.csv`
  - `reports/scan_aspley_fixed/character_frequency.csv`
- Canonical normalized output: `reports/shakespeare/comparison/whole_book_comparison.json`
- Present role: witness-inventory comparison only; includes the unmatched Aspley extras and shared-sequence paratext

### High-confidence matched comparison

- Source artifacts:
  - `reports/wright_80conf/statistics.json`
  - `reports/aspley_80conf/statistics.json`
  - `reports/wright_80conf/character_frequency.csv`
  - `reports/aspley_80conf/character_frequency.csv`
- Canonical normalized output: `reports/shakespeare/comparison/high_confidence_comparison.json`
- Present role: the most useful surviving quantitative comparison in the packet, but still not proof of typographic identity
- Current totals: Wright `38,909` characters versus Aspley `48,142`, with long-s counts `811` and `1,305`

### Page-variance diagnostic

- Source artifact: `reports/page_forensics/page_comparison.json`
- Canonical normalized output: `reports/shakespeare/comparison/page_variance_diagnostic.json`
- Present role: diagnostic only, because sequence pairing does not yet guarantee complete sonnet-level equivalence across every page pair

### Sonnet-opening map

- Source artifact: `reports/full_sonnet_mapping.json`
- Present canonical reading: partial surviving map only (`4` Wright detections, `16` Aspley detections, `1` same-page match)
- Explicitly not canonical: `reports/full_sonnet_mapping_report.md`, which preserves contradictory legacy prose

### Manual review

- Canonical ledger: `reports/shakespeare/manual_review/manual_review_ledger.json`
- Human-readable companions:
  - `reports/shakespeare/manual_review/manual_review_ledger.md`
  - `reports/shakespeare/manual_review/manual_review_summary.md`
- Approved labels:
  - `same design`
  - `possible same sort/block`
  - `inconclusive`
- Current package result: `2` `same design`, `0` `possible same sort/block`, `4` `inconclusive`

### Rebuild entrypoint

- Canonical rebuild command: `./.venv/bin/python scripts/maintenance/rebuild_shakespeare_lane.py`
- Generator script: `scripts/research/generate_shakespeare_canonical_artifacts.py`
- Generated outputs:
  - `reports/shakespeare/summary.txt`
  - `reports/shakespeare/shakespeare_archive.html`
  - `reports/shakespeare/CODEFINDER_Shakespeare_Report.pdf`
  - `reports/shareable/CODEFINDER_Shakespeare_Summary.pdf`
  - comparison JSONs, package manifest, page-equivalence manifest, and manual-review artifacts under `reports/shakespeare/`

## 3. March 9, 2026 Remediation Log

The Shakespeare cleanup was not treated as a prose rewrite. It required several structural corrections first.

1. The local Shakespeare metadata was reconciled to the actual current corpus: both witnesses are cached locally as `2000px` JPG downloads.
2. A new canonical page-equivalence manifest was created so the lane now states explicitly which images are included, excluded paratext, or unmatched.
3. Canonical Shakespeare comparison JSONs were regenerated with bibliographically explicit source labels and populated character-delta tables; the placeholder delta rows in the older comparison outputs are no longer the canonical surface.
4. A new manual-review ledger and summary were created for the Shakespeare lane using the same controlled label vocabulary as the German lane.
5. A canonical archive summary, package manifest, archive HTML/PDF, and shareable PDF were generated under a dedicated Shakespeare namespace.
6. Repo-level docs and rebuild/retention manifests were updated so Shakespeare now has a first-class canonical packet instead of a control memo plus a contradictory legacy report cluster.

## 4. Current Evidentiary Position

The canonical Shakespeare packet supports the following restrained statements:

- the local Folger corpus now has a normalized witness inventory and an explicit shared-sequence spine;
- whole-witness and high-confidence OCR inventories remain materially asymmetric across the two witnesses;
- the surviving sonnet-opening map is still too partial to stabilize every paired page as content-equivalent;
- the legacy page-variance diagnostic remains useful as a warning signal, not as a verdict;
- the manual review supports design-level continuity in a small number of rows, but no row currently warrants same-object language.

The packet does **not** support these stronger claims:

- that the witnesses are already proven typographically identical;
- that the null hypothesis has been accepted and the alternative rejected;
- that OCR quality alone fully explains all surviving variance;
- that any current page overlay proves a settled print-state conclusion.

## 5. Manual Validation Protocol

The current packet should be validated in this order:

1. Confirm the page-equivalence manifest against the local witness directories and ensure no image is classified more than once.
2. Review the sole same-page sonnet-opening match (page `9`, Sonnet `2`) as a design-level control only.
3. Revisit the top page-variance rows (`17`, `18`, `33`, `41`) under the new diagnostic framing, not the older verdict framing.
4. Review the Aspley-only extra capture on page `56` as source-shape evidence rather than as a paired page.
5. Treat any move from `same design` to `possible same sort/block` as requiring explicit visual and bibliographical justification.

## 6. Claim-to-Artifact Map

| Claim or wording | Canonical artifact | Allowed reading |
| --- | --- | --- |
| “The packet contains `53` shared-sequence image pairs.” | `reports/shakespeare/page_equivalence_manifest.json` | structural corpus fact |
| “The Aspley witness has `14` unmatched extra captures.” | `reports/shakespeare/page_equivalence_manifest.json` | structural corpus fact |
| “High-confidence OCR totals remain asymmetric.” | `reports/shakespeare/comparison/high_confidence_comparison.json` | quantitative observation only |
| “The page-variance output is diagnostic, not dispositive.” | `reports/shakespeare/comparison/page_variance_diagnostic.json` | interpretive limit |
| “Only `1` same-page sonnet-opening match survives in the canonical map.” | `reports/full_sonnet_mapping.json` and `reports/shakespeare/page_equivalence_manifest.json` | caution on equivalence |
| “No Shakespeare row is currently rated `possible same sort/block`.” | `reports/shakespeare/manual_review/manual_review_summary.md` | manual-review limit |

## 7. Caption-Ready Figure Notes

- `reports/page_forensics/deep_analysis/page_18_summary.png`
  - Use only as a diagnostic illustration of how strongly page-level overlays can diverge.
  - Do not caption it as proof of typographic difference or proof of OCR-only noise.
- `reports/page_forensics/page_variance_visualization.png`
  - Use only if the caption states that the figure reflects the legacy shared-sequence diagnostic, not full sonnet-level equivalence.
- `reports/page_forensics/outlier_summary.txt`
  - Treat as a ranking aid for review priorities, not as a publication verdict table.

## 8. Submission Readiness Checklist

- Page-equivalence manifest and package manifest agree on witness totals and unmatched extras.
- Manuscript, appendix, archive summary, and shareable summary all use the same claim level.
- Canonical Shakespeare outputs do not reuse retired overclaim language.
- Manual-review counts match the ledger exactly.
- Rebuild command reproduces the canonical Shakespeare artifact family from the local corpus.
- Any future stronger claim is blocked until either sonnet-level equivalence improves materially or a documented rerun supersedes the current packet.

## 9. Selected Scholarship and Source Records

- Philip Gaskell, *A New Introduction to Bibliography*.
- D. F. McKenzie, *Bibliography and the Sociology of Texts*.
- Folger Shakespeare Library digital record for `STC 22353a`.
- Folger Shakespeare Library digital record for `STC 22353`.
- Wright manifest: `https://digitalcollections.folger.edu/node/70076/manifest`
- Aspley manifest: `https://digitalcollections.folger.edu/node/29467/manifest`
