# CODEFINDER: Final Proof Report

## Verdict: **INCONCLUSIVE**

Moderate evidence, requires further investigation.

---

## Acceptance Thresholds
| Metric | Threshold | Observed |
|--------|-----------|----------|
| Mean Residual (RMSE) | < 6.0 px | 5.38 px |
| Max Residual | < 15.0 px | 15.0 px |
| Min Page Coverage | > 25.0% | (per page) |
| Min Matches/Page | > 300 | (for validity) |

## Aggregate Statistics
- **Valid Pages Analyzed**: 32
- **Pages Passed**: 21 (65.6%)
- **Pages Failed**: 11
- **Total Character Matches**: 22,785
- **Mean RMSE**: 5.38 ± 1.34 px

---

## Per-Page Results

| Page | Matches | Mean Res. | Max Res. | Coverage | Verdict |
|------|---------|-----------|----------|----------|---------|
| 9 | 620 | 5.02 px | 14.96 px | 79.3% | ✅ PASS |
| 10 | 430 | 5.26 px | 14.92 px | 33.9% | ✅ PASS |
| 11 | 876 | 4.75 px | 14.98 px | 39.1% | ✅ PASS |
| 12 | 934 | 4.13 px | 14.97 px | 44.8% | ✅ PASS |
| 13 | 1001 | 3.94 px | 14.92 px | 44.8% | ✅ PASS |
| 14 | 1048 | 3.98 px | 14.94 px | 45.9% | ✅ PASS |
| 15 | 761 | 4.07 px | 14.96 px | 41.8% | ✅ PASS |
| 16 | 588 | 5.71 px | 14.94 px | 40.3% | ✅ PASS |
| 17 | 529 | 6.13 px | 14.97 px | 45.9% | ❌ FAIL |
| 18 | 791 | 7.62 px | 14.92 px | 33.6% | ❌ FAIL |
| 19 | 902 | 5.57 px | 15.0 px | 38.4% | ✅ PASS |
| 20 | 505 | 6.55 px | 14.9 px | 45.2% | ❌ FAIL |
| 21 | 856 | 4.14 px | 14.99 px | 47.8% | ✅ PASS |
| 24 | 836 | 5.69 px | 14.9 px | 40.5% | ✅ PASS |
| 25 | 415 | 5.85 px | 14.73 px | 38.7% | ✅ PASS |
| 26 | 659 | 6.58 px | 15.0 px | 27.9% | ❌ FAIL |
| 27 | 947 | 3.34 px | 14.92 px | 55.2% | ✅ PASS |
| 28 | 941 | 4.02 px | 14.75 px | 44.8% | ✅ PASS |
| 29 | 1129 | 2.51 px | 14.98 px | 47.2% | ✅ PASS |
| 30 | 405 | 6.36 px | 14.97 px | 26.2% | ❌ FAIL |
| 31 | 772 | 4.12 px | 14.99 px | 39.2% | ✅ PASS |
| 32 | 1057 | 3.29 px | 14.99 px | 48.8% | ✅ PASS |
| 33 | 606 | 6.86 px | 14.98 px | 33.4% | ❌ FAIL |
| 35 | 894 | 6.01 px | 15.0 px | 39.0% | ❌ FAIL |
| 37 | 568 | 5.81 px | 14.95 px | 33.8% | ✅ PASS |
| 39 | 516 | 5.69 px | 14.93 px | 24.2% | ✅ PASS |
| 40 | 537 | 8.85 px | 15.0 px | 25.1% | ❌ FAIL |
| 42 | 332 | 6.66 px | 14.99 px | 16.6% | ❌ FAIL |
| 43 | 310 | 6.13 px | 14.95 px | 30.2% | ❌ FAIL |
| 44 | 530 | 5.45 px | 14.99 px | 40.4% | ✅ PASS |
| 45 | 591 | 6.07 px | 14.93 px | 32.9% | ❌ FAIL |
| 46 | 899 | 5.93 px | 14.97 px | 42.0% | ✅ PASS |

---

## Excluded Pages
Pages with insufficient data for valid analysis:

| Page | Reason |
|------|--------|
| 1 | insufficient_matches (1) |
| 2 | insufficient_matches (2) |
| 3 | ransac_failed |
| 5 | ransac_failed |
| 6 | insufficient_matches (1) |
| 7 | insufficient_matches (37) |
| 8 | insufficient_matches (1) |
| 22 | insufficient_matches (2) |
| 23 | insufficient_matches (4) |
| 34 | insufficient_matches (2) |
| ... | (7 more) |

---

## Conclusion

Based on exhaustive per-character analysis of 32 valid pages 
with 22,785 character matches, the null hypothesis 
(H₀: Wright and Aspley editions are typographically identical) is **INCONCLUSIVE**.

**Evidence**:
- Mean character position error: 5.38 pixels
- 65.6% of analyzed pages pass acceptance thresholds
- Observed variance is consistent with expected printing/scanning artifacts

---

*Generated: 2026-02-05T19:36:30.044491*
