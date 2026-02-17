# Forensic Analysis Methodology Critique

## 📋 Document Purpose
This document critically examines the forensic analysis methodology used to compare the Wright (STC 22353a) and Aspley (STC 22353) editions of Shakespeare's Sonnets.

## ❌ Errors Made

### 1. Premature Conclusion
We concluded "both editions use the same type face" based on normalized averages showing <2px differences. This was wrong because:
- **Averages mask variance** - Two distributions can have identical means but different shapes
- **Normalization assumes uniform scale** - But character-level ratios show this isn't true
- **No statistical significance testing** - We didn't calculate p-values or confidence intervals

### 2. Attribution Bias
We attributed all differences to "OCR quality" without evidence:
- Wright has 38,909 high-confidence characters
- Aspley has 48,142 high-confidence characters
- We assumed this was "better scans" - but didn't prove it

### 3. Ignored Non-Uniform Ratios
If differences were purely OCR quality, ALL categories would increase by the same ratio. But:

| Category | Ratio | Expected (if uniform) |
|----------|-------|----------------------|
| digit | 1.539 | 1.24 |
| long_s | 1.261 | 1.24 |
| lowercase | 1.043 | 1.24 |
| other | 0.851 | 1.24 |

The spread (0.688) is too large to explain with OCR variance alone.

### 4. Specific Character Anomalies Ignored

| Character | Wright | Aspley | Ratio | Significance |
|-----------|--------|--------|-------|--------------|
| J (capital) | 2 | 15 | 7.5x | J barely used in 1609 |
| 5 | 4 | 15 | 3.75x | Digit disparity |
| ' (straight) | 96 | 18 | 0.19x | Apostrophe convention |
| ' (curly) | 9 | 23 | 2.56x | Opposite pattern |

These cannot be explained by "better scans."

## ✅ Correct Forensic Methodology

### 1. Source Validation
Before comparing, we must validate that we're comparing equivalent material:
- [ ] Are both editions from the same print run?
- [ ] Are page counts identical (excluding endpapers)?
- [ ] Are there known textual variants between STC 22353 and 22353a?

### 2. Normalization Protocol
- [ ] Measure physical dimensions of original books
- [ ] Calculate actual DPI of each scan
- [ ] Normalize all measurements to mm or points, not pixels

### 3. Statistical Rigor
For each character:
- [ ] Calculate mean, median, mode, and standard deviation
- [ ] Plot distribution histograms
- [ ] Apply Kolmogorov-Smirnov test to compare distributions
- [ ] Identify outliers (>2σ from mean)

### 4. Control Comparisons
- [ ] Compare characters we KNOW should be identical (e.g., same word on same page)
- [ ] Use this to calibrate OCR variance baseline
- [ ] Any variance beyond baseline = real typographical difference

### 5. Page-Level Analysis
- [ ] Extract character counts per page
- [ ] Identify pages with unusual variance
- [ ] Cross-reference with known compositor stints

### 6. Visual Verification
For any candidate difference:
- [ ] Extract character images from both editions
- [ ] Side-by-side visual comparison at high zoom
- [ ] Domain expert review

## 📊 Data We Actually Have

### Character Geometry (80% confidence)
- Wright: 38,909 characters, 75 unique types
- Aspley: 48,142 characters, 75 unique types

### Key Differences Requiring Investigation

1. **Apostrophe Pattern Reversal**
   - Wright prefers straight quote (96 vs 9)
   - Aspley prefers curly quote (23 vs 18)
   - This could indicate different compositor habits or fonts

2. **Long-s vs Regular-s Ratio**
   - Wright: 811 long-s, 994 regular-s (ratio 0.82)
   - Aspley: 1305 long-s, 1214 regular-s (ratio 1.07)
   - Aspley has MORE long-s - counter to modernization trend

3. **J Character**
   - Capital J was not standardized in 1609
   - Wright has 2 instances, Aspley has 15
   - Are these real 'J' or misread 'I'?

4. **Digit Variance**
   - Digits 4, 5 show high variance
   - Could relate to sonnet numbers or page numbers
   - Need page-specific analysis

## 🔬 Next Steps

1. **Build Per-Page Analysis**
   - Modify scanner to output page-level character counts
   - Compare page N (Wright) vs page N (Aspley) directly

2. **Extract Character Image Pairs**
   - For each anomalous character, extract images from both editions
   - Enable visual comparison

3. **Apply Statistical Tests**
   - Calculate if differences are statistically significant
   - Use bootstrap resampling for confidence intervals

4. **Cross-Reference Scholarly Literature**
   - What do scholars say about 22353 vs 22353a differences?
   - Are our findings consistent with known variants?

## 🎯 Hypothesis to Test

**Null Hypothesis (H₀)**: Both editions were printed from identical type with no typographical differences; all observed variance is OCR/scan quality.

**Alternative Hypothesis (H₁)**: The editions contain real typographical differences indicating different print states, compositor habits, or stop-press corrections.

Current evidence suggests we should **reject H₀** based on:
- Non-uniform category ratios
- Specific character anomalies (J, apostrophes)
- Long-s pattern differences

---

*Document created: 2026-02-05*
*Status: Critical methodology review*
