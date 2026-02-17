# FEASIBILITY STUDY: Next Steps for Forensic Print Block Analysis
## Compiled: February 5, 2026

---

## 🔍 CURRENT PIPELINE AUDIT: WEAK POINTS IDENTIFIED

### Critical Issues Found:

| Issue | Severity | Impact | Root Cause |
|-------|----------|--------|------------|
| **Empty Database** | 🔴 Critical | No persistent character data | OCR runs but doesn't save to SQLite |
| **No Character Geometry Storage** | 🔴 Critical | Can't do sort-level comparison | Geometry extracted but discarded |
| **Image Resolution Mismatch** | 🟠 High | Aspley 2879px vs Wright 1296px height | Different source quality |
| **Tesseract OCR Limitations** | 🟠 High | ~80% confidence ceiling | No specialized historical font training |
| **No Bounding Box Persistence** | 🟠 High | Can't extract individual characters | Data exists during scan, not saved |

### Pipeline Weak Points (In Order):

```
1. SOURCE IMAGES ──────────────────────────────────────────────
   ├─ Wright: 2000x1296 (lower resolution)
   └─ Aspley: 2000x2879 (higher resolution) ←── MISMATCH
   
2. OCR ENGINE ─────────────────────────────────────────────────
   ├─ Using Tesseract 4/5 with default English model
   ├─ No historical font training (Pica, Long-s, ligatures)
   └─ Confidence threshold at 80% (still noisy)
   
3. DATA STORAGE ───────────────────────────────────────────────  ← CRITICAL GAP
   ├─ Database tables EMPTY
   ├─ Character instances not persisted
   └─ Bounding boxes only in memory during scan
   
4. ANALYSIS LAYER ──────────────────────────────────────────────
   ├─ Page-level comparison working ✓
   ├─ Statistical significance working ✓
   └─ Sort-level comparison NOT POSSIBLE (no character images)
```

---

## 📊 OPTION ANALYSIS

### Option 1: High-Resolution Sort Extraction

**Goal**: Extract individual character images for visual comparison

**Feasibility**: ⭐⭐⭐⭐ (4/5)

**Why This Matters**:
- The ONLY way to definitively compare typefaces is visual inspection
- Can reveal damaged sorts, different type founts, or compositor habits
- Academic gold standard for bibliographical analysis

**Sub-Tasks Breakdown**:

| Task | Difficulty | Dependencies | Time Est. |
|------|------------|--------------|-----------|
| 1.1 Fix database schema for character storage | 3/10 | None | 30 min |
| 1.2 Modify OCR pipeline to save bounding boxes | 5/10 | 1.1 | 1 hr |
| 1.3 Create character image extractor | 4/10 | 1.2 | 45 min |
| 1.4 Run extraction on both editions | 2/10 | 1.3 | 2 hrs (automated) |
| 1.5 Create side-by-side character comparison tool | 5/10 | 1.4 | 1 hr |
| 1.6 Generate character comparison reports | 4/10 | 1.5 | 1 hr |

**Total Estimated Time**: 6-7 hours
**Maximum Task Difficulty**: 5/10 ✅

**Blockers**: None - all tools exist, just need integration

---

### Option 2: Cross-Reference with Bibliographical Scholarship

**Goal**: Validate findings against published research

**Feasibility**: ⭐⭐⭐ (3/5)

**Why This Matters**:
- Confirms if our analysis aligns with expert consensus
- May reveal things we're missing
- Provides academic credibility

**Sub-Tasks Breakdown**:

| Task | Difficulty | Dependencies | Time Est. |
|------|------------|--------------|-----------|
| 2.1 Research 1609 Quarto variants in literature | 3/10 | Web search | 1 hr |
| 2.2 Identify key bibliographical studies | 2/10 | 2.1 | 30 min |
| 2.3 Extract specific claims about type/variants | 4/10 | 2.2 | 1 hr |
| 2.4 Map claims to our data | 6/10 | 2.3 + our data | 2 hrs |
| 2.5 Generate validation report | 4/10 | 2.4 | 1 hr |

**Total Estimated Time**: 5-6 hours
**Maximum Task Difficulty**: 6/10 ✅

**Blockers**: 
- Requires access to academic sources (may be paywalled)
- Subjective interpretation needed

---

### Option 3: Apply Methodology to Different Sources

**Goal**: Test methodology on more copies/editions

**Feasibility**: ⭐⭐⭐⭐⭐ (5/5)

**Why This Matters**:
- Validates our methodology works generally
- May find ACTUAL variants in different copies
- Creates reusable forensic tooling

**Sub-Tasks Breakdown**:

| Task | Difficulty | Dependencies | Time Est. |
|------|------------|--------------|-----------|
| 3.1 Identify additional Sonnet copies (BL, Huntington) | 2/10 | Web search | 30 min |
| 3.2 Download/access additional IIIF sources | 4/10 | 3.1 | 1 hr |
| 3.3 Run existing pipeline on new sources | 2/10 | 3.2 | Automated |
| 3.4 Compare results across 3+ copies | 5/10 | 3.3 | 2 hrs |
| 3.5 Document methodology for other texts | 4/10 | 3.4 | 1 hr |

**Total Estimated Time**: 5-6 hours
**Maximum Task Difficulty**: 5/10 ✅

**Blockers**:
- Dependent on source availability
- May need institutional access

---

### Option 4: OCR QUALITY IMPROVEMENT (ROOT CAUSE FIX) 

**Goal**: Address the fundamental weak point - OCR accuracy

**Feasibility**: ⭐⭐⭐⭐ (4/5)

**Why This Matters**:
- 31.6% character count variance is primarily OCR-driven
- Better OCR = better analysis everywhere
- Fixes root cause rather than symptoms

**Sub-Tasks Breakdown**:

| Task | Difficulty | Dependencies | Time Est. |
|------|------------|--------------|-----------|
| 4.1 Implement Gemini Vision OCR adapter | 5/10 | API key | 1 hr |
| 4.2 Compare Tesseract vs Gemini on sample pages | 3/10 | 4.1 | 1 hr |
| 4.3 Train/fine-tune for historical fonts (optional) | 8/10 | Tesseract expertise | 4+ hrs |
| 4.4 Normalize image resolution before OCR | 4/10 | PIL | 45 min |
| 4.5 Re-run analysis with improved OCR | 2/10 | 4.1-4.4 | Automated |

**Total Estimated Time**: 4-6 hours (excluding 4.3)
**Maximum Task Difficulty**: 5/10 (if skipping training) ✅

**Blockers**:
- Task 4.3 (font training) is complex - SKIP or defer
- Gemini API costs

---

## 🎯 RECOMMENDATION: PRIORITIZED ROADMAP

Based on **ROI (Return on Investment)** and **fixing root causes**:

### Phase 1: Fix Data Storage Gap (IMMEDIATE)
**Priority**: 🔴 Critical
**Why**: Without persistent data, we rerun OCR constantly

| Step | Task | Complexity |
|------|------|------------|
| 1 | Create proper database schema | 3/10 |
| 2 | Modify scanner to persist character data | 5/10 |
| 3 | Add bounding box storage | 4/10 |
| 4 | Run one complete scan to populate DB | 2/10 |

**Outcome**: Foundational data layer for all future analysis

---

### Phase 2: High-Resolution Sort Extraction (THIS SESSION)
**Priority**: 🟠 High
**Why**: Enables visual verification of statistical findings

| Step | Task | Complexity |
|------|------|------------|
| 1 | Create character image extractor | 4/10 |
| 2 | Extract top 10 outlier characters | 3/10 |
| 3 | Generate side-by-side comparison sheets | 5/10 |
| 4 | Visual verification of J, quotes, digits | 2/10 |

**Outcome**: Definitive visual evidence for/against type variation

---

### Phase 3: OCR Quality Normalization (NEXT SESSION)
**Priority**: 🟡 Medium
**Why**: Reduces noise in all downstream analysis

| Step | Task | Complexity |
|------|------|------------|
| 1 | Normalize image resolution | 4/10 |
| 2 | Implement Gemini Vision as OCR option | 5/10 |
| 3 | A/B test Tesseract vs Gemini | 3/10 |
| 4 | Choose best engine, re-analyze | 2/10 |

**Outcome**: Higher confidence character detection

---

### Phase 4: Expand to Other Sources (FUTURE)
**Priority**: 🟢 Low (but important for validation)

---

## 📈 EFFORT VS. IMPACT MATRIX

```
                        HIGH IMPACT
                            │
        Phase 2             │            Phase 1
   (Sort Extraction)        │         (Data Storage)
   ●─────────────────       │       ────────────────●
   Medium Effort            │            Low Effort
                            │
 ───────────────────────────┼───────────────────────────
                            │
        Phase 3             │            Phase 4
   (OCR Improvement)        │        (More Sources)
   ●─────────────────       │       ────────────────●
   Medium Effort            │            Low Effort
                            │
                        LOW IMPACT
```

---

## ✅ IMMEDIATE ACTION PLAN

**Do NOW (in order)**:

1. **Fix Database Storage** (30 min, Complexity 3/10)
   - Create tables for characters, pages, sources
   - Modify existing scanner pipeline to persist data
   
2. **Extract Character Images** (1 hr, Complexity 4/10)
   - Focus on statistically significant characters:
     - Capital J (Wright: 2, Aspley: 13 - 6.5x ratio)
     - Straight quote (major outlier)
     - Capital O (2.2x ratio)
     - Digits (1.84x category ratio)

3. **Generate Visual Comparison Sheets** (45 min, Complexity 5/10)
   - Side-by-side of each outlier character
   - Human reviewer can make final determination

**Defer**:
- Historical font training (8/10 complexity - break down later if needed)
- Additional source acquisition (dependent on availability)

---

*Want me to proceed with Phase 1 (Data Storage Fix) and Phase 2 (Sort Extraction) now?*
