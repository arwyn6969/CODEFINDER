#!/usr/bin/env python3
"""
Per-Page Forensic Comparison Tool

Performs rigorous page-by-page comparison between two editions,
identifying pages with significant character count differences
that may indicate real typographical variants.

This addresses the methodological flaw of comparing only aggregate totals.
"""

import argparse
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
from datetime import datetime

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class PageStats:
    """Statistics for a single page."""
    page_number: int
    total_chars: int = 0
    high_conf_chars: int = 0  # >80% confidence
    avg_confidence: float = 0.0
    long_s_count: int = 0
    char_breakdown: Dict[str, int] = None
    
    def __post_init__(self):
        if self.char_breakdown is None:
            self.char_breakdown = {}


def extract_page_stats(image_path: str, page_num: int, min_confidence: float = 80.0) -> PageStats:
    """Extract character statistics from a single page image."""
    if not OCR_AVAILABLE:
        return PageStats(page_number=page_num)
    
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    stats = PageStats(page_number=page_num, char_breakdown={})
    confidence_sum = 0
    conf_count = 0
    
    for i, text in enumerate(data['text']):
        if not text.strip():
            continue
        
        conf = float(data['conf'][i])
        if conf < 0:  # Invalid confidence
            continue
        
        confidence_sum += conf
        conf_count += 1
        
        # Count characters
        for char in text:
            stats.total_chars += 1
            if conf >= min_confidence:
                stats.high_conf_chars += 1
                stats.char_breakdown[char] = stats.char_breakdown.get(char, 0) + 1
                if char == 'ſ':
                    stats.long_s_count += 1
    
    stats.avg_confidence = confidence_sum / conf_count if conf_count else 0
    return stats


def compare_page_pair(
    wright_stats: PageStats, 
    aspley_stats: PageStats
) -> Dict:
    """Compare statistics between corresponding pages from two editions."""
    
    char_delta = aspley_stats.high_conf_chars - wright_stats.high_conf_chars
    
    # Calculate per-character deltas
    all_chars = set(wright_stats.char_breakdown.keys()) | set(aspley_stats.char_breakdown.keys())
    char_deltas = {}
    for char in all_chars:
        w_count = wright_stats.char_breakdown.get(char, 0)
        a_count = aspley_stats.char_breakdown.get(char, 0)
        if w_count > 0 or a_count > 0:
            delta = a_count - w_count
            if abs(delta) > 0:  # Only track differences
                char_deltas[char] = {
                    'wright': w_count,
                    'aspley': a_count,
                    'delta': delta
                }
    
    # Determine significance
    # We expect some variance due to OCR - use standard deviation approach
    if wright_stats.high_conf_chars > 0:
        ratio = aspley_stats.high_conf_chars / wright_stats.high_conf_chars
    else:
        ratio = float('inf') if aspley_stats.high_conf_chars > 0 else 1.0
    
    # Flag as significant if >20% difference or >100 character delta
    is_significant = abs(char_delta) > 100 or (ratio < 0.8 or ratio > 1.2)
    
    return {
        'page': wright_stats.page_number,
        'wright_chars': wright_stats.high_conf_chars,
        'aspley_chars': aspley_stats.high_conf_chars,
        'delta': char_delta,
        'ratio': ratio,
        'wright_long_s': wright_stats.long_s_count,
        'aspley_long_s': aspley_stats.long_s_count,
        'long_s_delta': aspley_stats.long_s_count - wright_stats.long_s_count,
        'significant': is_significant,
        'char_deltas': char_deltas
    }


def run_comparison(
    wright_source: str,
    aspley_source: str,
    output_dir: str,
    page_range: Tuple[int, int] = None,
    min_confidence: float = 80.0
) -> List[Dict]:
    """Run page-by-page comparison between two editions."""
    
    wright_path = Path(wright_source)
    aspley_path = Path(aspley_source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    wright_images = sorted(wright_path.glob("*.jpg"))
    aspley_images = sorted(aspley_path.glob("*.jpg"))
    
    print(f"Wright images: {len(wright_images)}")
    print(f"Aspley images: {len(aspley_images)}")
    
    if page_range:
        start, end = page_range
        wright_images = wright_images[start:end]
        aspley_images = aspley_images[start:end]
    
    # Ensure same number of pages
    min_pages = min(len(wright_images), len(aspley_images))
    wright_images = wright_images[:min_pages]
    aspley_images = aspley_images[:min_pages]
    
    print(f"Comparing {min_pages} pages...")
    print()
    
    results = []
    significant_pages = []
    
    for i, (w_img, a_img) in enumerate(zip(wright_images, aspley_images)):
        page_num = i + 1
        
        if page_num % 10 == 0:
            print(f"  Processing page {page_num}/{min_pages}...")
        
        w_stats = extract_page_stats(str(w_img), page_num, min_confidence)
        a_stats = extract_page_stats(str(a_img), page_num, min_confidence)
        
        comparison = compare_page_pair(w_stats, a_stats)
        results.append(comparison)
        
        if comparison['significant']:
            significant_pages.append(comparison)
    
    # Generate reports
    print()
    print("=" * 70)
    print("PAGE-BY-PAGE FORENSIC COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(f"Total pages compared: {len(results)}")
    print(f"Significant differences found: {len(significant_pages)}")
    print()
    
    if significant_pages:
        print("PAGES WITH SIGNIFICANT DIFFERENCES:")
        print("-" * 70)
        print(f"{'Page':>5} {'Wright':>10} {'Aspley':>10} {'Delta':>10} {'Ratio':>8} {'Long-s Δ':>10}")
        print("-" * 70)
        for p in significant_pages:
            print(f"{p['page']:>5} {p['wright_chars']:>10} {p['aspley_chars']:>10} {p['delta']:>+10} {p['ratio']:>8.2f} {p['long_s_delta']:>+10}")
    
    # Save detailed results
    with open(output_path / "page_comparison.json", 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'wright_source': str(wright_source),
            'aspley_source': str(aspley_source),
            'min_confidence': min_confidence,
            'total_pages': len(results),
            'significant_pages': len(significant_pages),
            'results': results
        }, f, indent=2)
    
    # Save CSV summary
    with open(output_path / "page_comparison.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Page', 'Wright Chars', 'Aspley Chars', 'Delta', 'Ratio', 
                        'Wright Long-s', 'Aspley Long-s', 'Long-s Delta', 'Significant'])
        for r in results:
            writer.writerow([
                r['page'], r['wright_chars'], r['aspley_chars'], r['delta'],
                f"{r['ratio']:.3f}", r['wright_long_s'], r['aspley_long_s'],
                r['long_s_delta'], r['significant']
            ])
    
    print()
    print(f"Detailed results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Page-by-page forensic comparison")
    parser.add_argument("--wright", required=True, help="Path to Wright edition images")
    parser.add_argument("--aspley", required=True, help="Path to Aspley edition images")
    parser.add_argument("--output", default="reports/page_forensics", help="Output directory")
    parser.add_argument("--pages", default=None, help="Page range to compare (e.g., 7-47)")
    parser.add_argument("--min-confidence", type=float, default=80.0, help="Minimum OCR confidence")
    
    args = parser.parse_args()
    
    page_range = None
    if args.pages:
        parts = args.pages.split('-')
        page_range = (int(parts[0]) - 1, int(parts[1]))
    
    run_comparison(
        args.wright,
        args.aspley,
        args.output,
        page_range,
        args.min_confidence
    )


if __name__ == "__main__":
    main()
