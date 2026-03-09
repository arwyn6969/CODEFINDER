# Figure Audit for the March 8, 2026 Kempten Packet

> Scope: figures embedded or staged for `reports/final_report/CODEFINDER_Forensic_Report.pdf`, `reports/shareable/CODEFINDER_Discord_Summary.pdf`, and the cited manual-review image artifacts.
> Actions: `keep`, `replace`, `remove`, `caption rewrite`.

## Summary

The current image set has three recurring problems:

1. selection by convenience rather than by argument;
2. captions that overstate what the image can show;
3. inconsistent page function and scale across compared witnesses.

The public packet should use fewer images, chosen more deliberately, with stronger contextual captions.

## Audit Table

| Asset | Current use | Claim supported | Provenance status | Readability status | Action | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| `reports/proof_images/source_page_bsb_munich.jpg` | Figure 1 in the forensic PDF | "Representative page" for BSB witness | weak | low | replace | This is effectively a sparse endpaper/date-style page, not a useful representative text specimen for comparison with the other witnesses. |
| `reports/proof_images/source_page_gdz_gottingen.jpg` | Figure 1 in the forensic PDF | "Representative page" for GDZ witness | acceptable | medium | caption rewrite | The image itself is useful, but the figure should say these are selected witness pages, not neutral representative pages. |
| `reports/proof_images/source_page_hab_wolfenbuttel.jpg` | Figure 1 in the forensic PDF | "Representative page" for HAB witness | acceptable | medium | caption rewrite | This works as a witness-identification image, but it should be aligned with comparable page function across the whole four-image set. |
| `reports/proof_images/source_page_google_books.jpg` | Figure 1 in the forensic PDF | "Representative page" for Google Books witness | weak | medium | replace | The selected Google Books page is not aligned to the same page function as GDZ/HAB and reads as an arbitrary insertion rather than a controlled comparison. |
| `reports/proof_images/greenman_crop_gdz.jpg` | Figure 2 in the forensic PDF; left panel in the shareable PDF | verified GDZ ornament candidate | acceptable | medium | replace | The crop is real, but by itself it lacks page context and becomes misleading in the shareable packet where the adjacent caption implies a reference crop. |
| `reports/proof_images/sift_match_gdz_goettingen_ppn777246686_p9_205_530.jpg` | Figure 3 in the forensic PDF; right panel in the shareable PDF | SIFT support for the verified GDZ match | acceptable | low | replace | The overlay is too dense to read and the caption overclaims by treating the line field as proof of physical identity. |
| `reports/proof_images/char_comparison_A.png` | Figure 4 component in the forensic PDF | similar printed forms for `A` | weak | low | replace | This strip is generated from arbitrary `LIMIT 1` crops and is too small to sustain the claim attached to it. |
| `reports/proof_images/char_comparison_B.png` | Figure 4 component in the forensic PDF | similar printed forms for `B` | weak | low | replace | Same issue as above, compounded by especially small width. |
| `reports/proof_images/char_comparison_d.png` | Figure 4 component in the forensic PDF | similar printed forms for `d` | weak | low | replace | If `d` is one of the best reviewed candidates, it should be shown from the manual-review ranked pair, not from convenience-selected crops. |
| `reports/proof_images/char_comparison_e.png` | Figure 4 component in the forensic PDF | similar printed forms for `e` | weak | low | replace | The figure does not encode rank, pair, page, or bbox context and therefore cannot be critically assessed by a reader. |
| `reports/proof_images/char_comparison_g.png` | Figure 4 component in the forensic PDF | similar printed forms for `g` | weak | low | replace | Same problem: tiny, arbitrary, and disconnected from the reviewed evidence ledger. |
| `reports/manual_review/greenman_review_sheet.png` | cited manual-review artifact in the appendix | reference vs verified candidate vs control | acceptable | medium | replace | Useful as internal audit evidence, but the reference and verified panels are too visually close for external readers without stronger labels and context. |
| `reports/manual_review/sheets/bsb_munich_10057380__gdz_goettingen_ppn777246686.png` | cited manual-review contact sheet | top reviewed pairwise character evidence for this source pair | strong | medium | keep | This is a good internal audit artifact because it preserves rank, score, page, bbox, and manual-assessment slots. |
| `reports/manual_review/sheets/bsb_munich_10057380__google_books_tractatus_brevis.png` | cited manual-review contact sheet | top reviewed pairwise character evidence for this source pair | strong | medium | keep | Same rationale; keep as appendix/supporting evidence, not as a main-text figure. |
| `reports/manual_review/sheets/bsb_munich_10057380__hab_wolfenbuettel_178_1_theol_1s.png` | cited manual-review contact sheet | top reviewed pairwise character evidence for this source pair | strong | medium | keep | Strong as internal review evidence; too dense for main-text publication use. |
| `reports/manual_review/sheets/gdz_goettingen_ppn777246686__google_books_tractatus_brevis.png` | cited manual-review contact sheet | top reviewed pairwise character evidence for this source pair | strong | medium | keep | Keep as appendix/supporting sheet only. |
| `reports/manual_review/sheets/gdz_goettingen_ppn777246686__hab_wolfenbuettel_178_1_theol_1s.png` | cited manual-review contact sheet | top reviewed pairwise character evidence for this source pair | strong | medium | keep | Strong internal support and arguably better than the public character strips. |
| `reports/manual_review/sheets/google_books_tractatus_brevis__hab_wolfenbuettel_178_1_theol_1s.png` | cited manual-review contact sheet | top reviewed pairwise character evidence for this source pair | strong | medium | keep | Keep as appendix/supporting sheet only. |

## Caption and Use Notes

### Figure 1 replacement rule

Use one of these two patterns only:

1. all title pages, if the goal is witness identification and provenance;
2. all text-bearing pages from comparable book positions, if the goal is typographic atmosphere.

Do not mix endpapers, title pages, and arbitrary body pages under a neutral "representative pages" caption.

### Greenman / foliate-head rule

- In the public packet, the ornament figure should show page context first, crop second.
- If a match overlay is kept, it should be simplified or moved to supplemental status.
- Any caption should say `source-specific verified candidate in GDZ only`, not imply cross-source recurrence.

### Character-comparison rule

- Public figures should be pulled from the reviewed top-ranked pairs in `reports/manual_review/manual_review_ledger.md`.
- Each image should carry page and source labels.
- If a figure is meant to suggest `possible same sort/block`, it should say that explicitly and explain why the judgment remains provisional.

## Recommended Replacement Set

If the packet is reduced to a publication-safe minimum, the main-text figure set should be:

1. a corrected witness-identification figure with aligned page functions;
2. a GDZ page-context image with the foliate-head bbox marked, plus a crop inset;
3. one simplified comparison figure built from the two reviewed `possible same sort/block` rows (`d` and `b`) in `bsb_munich_10057380` vs `gdz_goettingen_ppn777246686`;
4. one table or compact chart summarizing pairwise sort ranges and the manual-review totals.

Everything else can live in appendix or repository supplement status.
