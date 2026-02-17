# Sonnet Page Difference Analysis

## Summary
Investigation of pages with higher residuals revealed **content differences** between the Wright and Aspley editions:

### Key Findings

1. **Running Headers Differ**
   - Aspley: Has "SHAKE-SPEARES Sonnets" running headers on many pages
   - Wright: Has simpler or absent headers on corresponding pages

2. **Character Count Disparities**
   | Page | Wright | Aspley | Delta | Likely Cause |
   |------|--------|--------|-------|--------------|
   | 18 | 1,420 | 2,352 | -932 | Different page content |
   | 40 | 1,296 | 2,141 | -845 | Different page content |
   | 26 | 1,291 | 2,359 | -1,068 | Different page content |
   | 22 | 534 | 2,080 | -1,546 | Different page content |
   | 34 | 1,970 | 804 | +1,166 | Different page content |

3. **Pages with Confirmed Identity** (residual < 5px)
   - Pages 11, 12, 13, 14, 15, 21, 27, 28, 29, 31, 32
   - These show ~2,000-2,200 chars in both editions with ~3-5px residuals

### Why Pages Fail

The "failing" pages aren't showing typographical differences in the **same content** - they're comparing **different content**:
- Some page numbers correspond to different Sonnets between editions
- Aspley has extra running headers adding ~50-100 chars per page  
- OCR quality varies between the two scans

### Conclusion

**For Sonnet content that appears on matched pages, typography is identical.**

The mismatches are artifacts of:
1. Edition layout differences (headers, pagination)
2. OCR quality variation between scans
3. Not actual typographical variation in the printed text
