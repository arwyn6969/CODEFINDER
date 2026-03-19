# Provisional Computational Evidence and Source-Equivalence Constraints in the Wright and Aspley Sonnets Witnesses

> Status: External-facing working draft for Shakespeare lane readers
> Date: March 9, 2026
> Claim level: External-safe provisional computational evidence and diagnostic comparison, not a settled attribution claim

## Abstract

This packet revisits the Shakespeare lane through two local Folger witnesses of the 1609 *Shake-speares sonnets*: the Wright issue (`STC 22353a`) and the Aspley issue (`STC 22353`). The immediate goal is narrower than the older report cluster. Rather than proving typographic identity, the packet asks what can be said safely once the surviving artifacts are normalized, contradiction-prone prose is retired, and the witness relationship is made explicit. The resulting corpus contains `120` local witness images, but only `53` of those images form a shared folio-sequence spine. Within that spine, `41` paired images contain signed folio content and `12` are paratext-heavy control images; the Aspley witness also carries `14` unmatched extra captures. High-confidence OCR totals remain asymmetric (`38,909` Wright, `48,142` Aspley), the legacy page-variance diagnostic still flags `40/53` paired images, and the surviving sonnet-opening map remains partial (`4` Wright detections, `16` Aspley detections, `1` same-page match). The cleaned packet therefore supports only a cautious reading: it preserves useful computational evidence for witness comparison, but source-equivalence constraints remain too strong for a final claim of typographic identity or shared physical type.

## Research Question and Historiographical Context

The question of this draft is not whether computation can replace bibliography. It is whether the current Shakespeare lane, once cleaned and normalized, yields evidence that deserves cautious external circulation at all. That is a lower bar than the language used in earlier internal reports, and intentionally so.

The older Shakespeare cluster overstated what page-aligned OCR differences could prove. The present packet instead treats computational outputs as a triage layer for bibliographical review. In that framing, the first responsibility is to state what the witnesses are, how the local corpus is shaped, and where the evidence becomes unstable. Classical analytical bibliography still depends on material evidence, print-shop context, and witness control. The computational outputs below are useful only insofar as they preserve those limits.

## Corpus and Provenance

The canonical Shakespeare packet is limited to the two local Folger witnesses already present in the repository.

| Source ID | Bibliographic description | Holding institution / record | Local images | Shared-sequence status | Provenance note |
| --- | --- | --- | ---: | --- | --- |
| `folger_sonnets_1609` | *Shake-speares sonnets : neuer before imprinted* (`STC 22353a`; seller imprint Wright) | Folger Shakespeare Library, [digital record](https://digitalcollections.folger.edu/bib169144-164315) | 53 | Entire local witness forms the shared sequence | Current local cache is a `2000px` JPG download from the Wright manifest |
| `folger_sonnets_1609_aspley` | *Shake-speares sonnets. : Neuer before imprinted* (`STC 22353`; seller imprint Aspley) | Folger Shakespeare Library, [digital record](https://digitalcollections.folger.edu/node/29467) | 67 | First `53` images participate in the shared sequence; `14` images are unmatched extras | Current local cache is a `2000px` JPG download from the Aspley manifest |

The normalized corpus therefore contains three distinct layers:

1. the Wright witness as a `53`-image local corpus;
2. the Aspley witness as a `67`-image local corpus;
3. a shared `53`-image folio-sequence spine used for page-paired diagnostics.

Within the shared sequence, `41` pairs are classified as included folio-content images and `12` pairs as paratext-heavy control images. The extra Aspley images are preserved, but they are not forced into the paired spine.

## Methods

The present packet does not treat every older Shakespeare artifact as equally trustworthy. Instead it rebuilds the lane from a small set of normalized inputs:

- source metadata and local image inventories for the two Folger witnesses;
- the new page-equivalence manifest in `reports/shakespeare/page_equivalence_manifest.json`;
- the surviving whole-witness scan statistics in `reports/scan_wright_fixed/` and `reports/scan_aspley_fixed/`;
- the surviving `80%` confidence scan statistics in `reports/wright_80conf/` and `reports/aspley_80conf/`;
- the legacy page-variance diagnostic in `reports/page_forensics/page_comparison.json`;
- the surviving partial sonnet-opening map in `reports/full_sonnet_mapping.json`;
- the new Shakespeare manual-review ledger in `reports/shakespeare/manual_review/`.

Two comparison levels are reported. The whole-witness comparison summarizes the current local corpora as stored, which means the Wright totals reflect `53` images and the Aspley totals reflect `67`. The matched high-confidence comparison is narrower: it uses the surviving `80%` confidence scan outputs anchored to the shared `53`-image spine. Neither comparison is treated as proof of shared physical type.

The manual-review labels follow the same narrow vocabulary now used in the German lane. `same design` marks a plausible design-level resemblance; `possible same sort/block` would mark a provisional same-object suspicion; `inconclusive` marks rows that remain too unstable for stronger interpretation. In the current Shakespeare packet, no row reaches `possible same sort/block`.

## Results

### 1. Inventory and corpus totals

The whole-witness local scan totals remain asymmetric:

| Witness | Local images | Characters | Unique chars | Avg confidence | Long-s count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wright | 53 | 89,236 | 110 | 63.11 | 593 |
| Aspley | 67 | 99,657 | 110 | 64.44 | 1,005 |

Those totals are useful as witness inventories, but they are not the main comparison layer because they mix shared-sequence material with the unmatched Aspley extras.

The higher-confidence matched packet is more informative for cross-witness comparison:

| Witness | Pages scanned | Characters | Unique chars | Avg confidence | Long-s count | Ligatures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wright | 53 | 38,909 | 75 | 90.30 | 811 | 1,890 |
| Aspley | 53 | 48,142 | 81 | 91.32 | 1,305 | 2,277 |

Even there, however, the asymmetry remains substantial enough that the packet cannot simply assume page- or sonnet-level equivalence.

### 2. Page-equivalence normalization and sonnet mapping

The strongest structural correction in the new lane is the explicit page-equivalence manifest. It makes three facts clear:

1. The Wright witness contributes `53` local images and the Aspley witness `67`.
2. Only the first `53` Aspley images belong to the shared folio-sequence spine.
3. The surviving sonnet-opening map remains fragmentary: `4` Wright detections, `16` Aspley detections, and only `1` same-page match.

That surviving same-page match is useful as a design-level continuity check, but not as a warrant for global claims about typographic identity across the packet.

### 3. OCR comparison and character deltas

The normalized comparison JSONs now use explicit bibliographical witness labels and populated character-delta tables. At the whole-witness level the largest positive deltas favor the Aspley witness in common lowercase forms such as `e`, `a`, `i`, and `t`, while the pipe character (`|`) swings strongly the other way. At the matched high-confidence level the strongest absolute deltas again involve common lowercase forms plus the long-s count (`811` Wright, `1,305` Aspley). These are meaningful inventory differences, but they do not by themselves resolve whether the driver is image quality, extraction behavior, witness state, page equivalence, or some combination of all four.

### 4. Page-variance diagnostics

The legacy page-forensics output remains worth preserving, but only as a diagnostic layer. It still flags `40` of `53` shared-sequence pairs as significant, with some of the largest deltas on pages `18`, `33`, `17`, and `41`. That result now has a narrower reading than it once did. It shows that page-paired OCR inventories vary materially across the shared sequence. It does not, on its own, prove typographic difference, compositor difference, or OCR-only noise.

### 5. Manual review

The new Shakespeare manual-review ledger intentionally stays small and conservative. It reviews `6` targeted rows:

- `2` rows are labelled `same design`;
- `0` rows are labelled `possible same sort/block`;
- `4` rows remain `inconclusive`.

This is the right proportion for the current packet. The manual review preserves useful leads without pretending that the evidence has already crossed into same-object proof.

## Discussion

The Shakespeare lane is now more professional precisely because it says less. The repo no longer needs to pretend that page-level OCR asymmetry, by itself, settles the Wright/Aspley relationship. Instead it can say something narrower and defensible: the local packet contains a coherent pair of Folger witnesses, a normalized page spine, and several surviving computational diagnostics worth preserving for further work. It also contains unresolved source-equivalence problems that block a stronger claim.

That outcome is not a failure. It is what a cleaned research packet should look like when the evidence remains mixed. The value of the present packet lies in auditability: a contributor can now see which artifacts are canonical, which are merely diagnostic, and which older claims have been retired.

## Limitations and Validation Priorities

The present packet remains limited in at least four ways.

First, the current local corpus is a `2000px` JPG cache for both witnesses, not a native-resolution rerun. Second, the surviving sonnet-opening map is still partial and therefore cannot stabilize every shared-sequence page as a content-equivalent pair. Third, the matched high-confidence comparison preserves strong asymmetries in character totals and long-s counts even after sequence normalization. Fourth, the manual review remains a computational audit aid rather than a specialist bibliographical judgement.

The next validation priorities are therefore clear:

1. complete or materially improve the sonnet-level page map;
2. rerun extraction from the same manifests at a documented resolution if stronger image-level claims are needed;
3. keep the current page-variance outputs diagnostic unless a tighter content-equivalence layer is established;
4. subject the most informative rows to specialist bibliographical review before any stronger statement about witness identity is made.

## Conclusion

The Shakespeare lane now has a canonical package, but not a final verdict. The cleaned packet supports provisional computational evidence and diagnostic comparison across the Wright and Aspley witnesses. It does not support the stronger language found in the older report cluster about proven typographic identity, accepted null hypotheses, or conclusively explained variance. In its present form, the packet is best used as a disciplined research checkpoint and a safer starting point for any later Shakespeare rerun.

## Selected References

- Gaskell, Philip. *A New Introduction to Bibliography*. Oxford: Clarendon Press, 1972.
- McKenzie, D. F. *Bibliography and the Sociology of Texts*. London: British Library, 1999.
- Folger Shakespeare Library. Digital record for *Shake-speares sonnets : neuer before imprinted* (`STC 22353a`).
- Folger Shakespeare Library. Digital record for *Shake-speares sonnets. : Neuer before imprinted* (`STC 22353`).

## Digital Source Records

- Wright witness: `https://digitalcollections.folger.edu/bib169144-164315`
- Wright manifest: `https://digitalcollections.folger.edu/node/70076/manifest`
- Aspley witness: `https://digitalcollections.folger.edu/node/29467`
- Aspley manifest: `https://digitalcollections.folger.edu/node/29467/manifest`
