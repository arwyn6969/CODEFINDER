# Document Scanning Protocol for Ink Dot Analysis

## Purpose

This protocol ensures scanned documents are captured at sufficient resolution for **sub-character ink dot analysis**. Follow these guidelines when re-scanning historical documents for the CODEFINDER project.

---

## Resolution Requirements

| Parameter | Recommended | Minimum | Notes |
|-----------|-------------|---------|-------|
| **Resolution** | 1200 DPI | 600 DPI | Native optical, not interpolated |
| **Color Depth** | 16-bit grayscale | 8-bit grayscale | 48-bit color for ink analysis |
| **Format** | TIFF (uncompressed) | PNG | Avoid JPEG compression |
| **Compression** | None | LZW lossless | Never use lossy compression |

> **Note**: The Folger IIIF source already provides ~7000×4700 px images (~300-400 DPI). For most ink texture analysis, these are adequate. Physical re-scans should only be pursued if higher resolution is specifically needed.

---

## Hardware Requirements

### Scanner
- Flatbed scanner with true optical resolution ≥1200 DPI
- Glass platen large enough for document
- No automatic image enhancement

### Lighting
- Uniform illumination across scan bed
- Optional: Raking light (15-30° angle) to reveal ink texture
- No glare or hot spots

### Calibration
- IT8 or similar color calibration target
- Run calibration before each scanning session

---

## Scanning Procedure

### Pre-Scan Checklist
- [ ] Scanner warmed up (10+ minutes)
- [ ] Calibration target scanned
- [ ] Scanner glass cleaned (lint-free cloth)
- [ ] Document positioned squarely
- [ ] Preview scan checked for alignment

### Scan Settings
```
Resolution: 1200 DPI
Color Mode: Grayscale 16-bit (or Color 48-bit)
Output Format: TIFF
Compression: None
Auto-enhance: OFF
Sharpening: OFF
Descreening: OFF
```

### Per-Page Procedure
1. Position document flat on scanner bed
2. Preview scan to check alignment
3. Adjust crop to include ~5mm border margin
4. Execute full-resolution scan
5. Verify output file is not corrupted
6. Log page number and scan timestamp

---

## Areas of Interest for Ink Analysis

When scanning, prioritize capturing these regions at highest quality:

| Area | Why It Matters |
|------|----------------|
| **Text body** | Primary analysis targets for dot detection |
| **Character edges** | Ink pooling is most visible here |
| **Page margins** | May contain compositor's marks |
| **Decorative elements** | Different pressure patterns |
| **Page numbers/headers** | Compositor identification |

---

## Post-Scan Verification

### Immediate Checks
- [ ] File size consistent (large = uncompressed)
- [ ] Resolution metadata correct (check EXIF/TIFF tags)
- [ ] No visible motion blur or scanner artifacts
- [ ] Full document visible without cropping

### Quality Metrics
```bash
# Check image properties
identify -verbose scan_001.tiff | grep -E "Resolution|Depth|Type"

# Expected output:
#   Resolution: 1200x1200
#   Depth: 16-bit
#   Type: Grayscale
```

---

## File Naming Convention

```
{source}_{page:03d}_{date}.tiff

Examples:
- folger_sonnets_001_20260205.tiff
- folger_sonnets_052_20260205.tiff
```

---

## Integration with CODEFINDER

After scanning, import images using the existing IIIF-style workflow:

```bash
# Place scans in data directory
cp *.tiff data/sources/high_res_scans/

# Run the scanner with new source
python3 sonnet_print_block_scanner.py --source data/sources/high_res_scans/
```

The `InkDotAnalyzer` can then be run on extracted characters:

```bash
python3 -m app.services.ink_dot_analyzer reports/digital_type_case/ --output reports/ink_dot_analysis/
```

---

## Appendix: Resolution Comparison

| Source | Approx. DPI | Ink Dot Suitability |
|--------|-------------|---------------------|
| Folger IIIF (native) | 300-400 | ⚠️ Adequate for basic analysis |
| Standard PDF | 150-200 | ❌ Too low |
| Physical re-scan @ 600 | 600 | ✅ Good |
| Physical re-scan @ 1200 | 1200 | ✅ Excellent |
