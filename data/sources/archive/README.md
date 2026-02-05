# Archived Shakespeare Sonnets Sources

This directory contains legacy/archived source files that have been superseded
by higher-quality sources but are retained for comparative analysis.

## Contents

### SONNETS_QUARTO_1609_NET.pdf
- **Status**: Archived (2026-02-05)
- **Provenance**: Unknown (likely Internet Archive)
- **Why Archived**: Unknown provenance; replaced by Folger Library IIIF source
- **Use Case**: Historical comparison with new Folger source

## Primary Source

The primary source is now the **Folger Shakespeare Library Digital Collections**:
- Catalog: https://digitalcollections.folger.edu/bib169144-164315
- Call Number: STC 22353a
- Resolution: ~7000×4700 pixels per page
- Rights: Public Domain (US)

To use the new source:
```bash
python scripts/download_folger_sonnets.py
```

## Source Comparison

If you need to compare analysis results between sources:
```bash
python scripts/compare_sources.py \
    --source1 reports/sonnet_print_block_analysis/ \
    --source2 reports/sonnet_print_block_analysis_folger/
```
