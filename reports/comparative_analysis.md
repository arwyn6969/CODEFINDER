# CODEFINDER: Comparative Analysis Report

## Full Census Results (Enhanced Pipeline)

Both editions were scanned with the enhanced Tesseract + Visual Ligature Analysis pipeline.

| Metric | Wright (STC 22353a) | Aspley (STC 22353) | Delta |
|--------|---------------------|---------------------|-------|
| **Pages Scanned** | 53 | 67 | +14 |
| **Total Characters** | 64,692 | 71,405 | +6,713 |
| **Unique Characters** | 84 | 85 | +1 |
| **Long-s (ſ) Detected** | 1,549 | 2,025 | +476 |
| **Ligatures Detected** | 1,801 | 1,216 | -585 |
| **Scan Duration** | 61.4s | 97.1s | +35.7s |

## Analysis

### Character Count Discrepancy
The Aspley edition has ~6,700 more characters because it has 14 more pages (67 vs 53). This is expected.

### Long-s Detection
Both editions show significant Long-s counts (previously near zero before the visual analysis fix). The Aspley count is higher (+476), which correlates with its higher page count.

**Ratio Check:**
- Wright: 1,549 Long-s / 64,692 chars = **2.39%**
- Aspley: 2,025 Long-s / 71,405 chars = **2.84%**

The ratios are similar, suggesting consistent detection rates. The slight Aspley increase may be due to better scan quality.

### Ligature Discrepancy (Interesting Finding)
Wright shows *more* ligatures (1,801 vs 1,216) despite having fewer pages. This could indicate:
1. **Scan Quality**: Wright scans have better ligature bridge visibility.
2. **Detection Threshold**: The Aspley images are larger (remember the 0.64 scale factor), which might affect the bridge zone analysis.
3. **True Variance**: Less likely for identical type-blocks.

**Next Steps**: Investigate by comparing ligature detection on matched pages (e.g., Page 11 of both editions).

## Geometric Proof Reference
See the [Geometric Proof](file:///Users/arwynhughes/Documents/CODEFINDER_PUBLISH/reports/geometric_proof.md) for visual confirmation of typographical identity.
