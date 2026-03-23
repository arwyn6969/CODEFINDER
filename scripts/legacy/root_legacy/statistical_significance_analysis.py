#!/usr/bin/env python3
"""
Statistical Significance Analysis for Character Variants

Applies statistical methods (z-scores, chi-squared) to determine which
character differences between Wright and Aspley editions are statistically
significant vs. normal OCR variance.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def load_all_page_data(json_path: str) -> List[Dict]:
    """Load all page comparison data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['results']

def aggregate_character_counts(results: List[Dict]) -> Tuple[Dict, Dict]:
    """Aggregate character counts across all pages for both editions."""
    
    wright_totals = defaultdict(int)
    aspley_totals = defaultdict(int)
    
    # Track page-level data for variance calculation
    wright_by_page = defaultdict(list)
    aspley_by_page = defaultdict(list)
    
    for r in results:
        for char, data in r.get('char_deltas', {}).items():
            wright_totals[char] += data['wright']
            aspley_totals[char] += data['aspley']
            wright_by_page[char].append(data['wright'])
            aspley_by_page[char].append(data['aspley'])
        
        # Add characters that appear in only one edition
        # (char_deltas only contains diffs, need to check for zeros)
    
    return dict(wright_totals), dict(aspley_totals)

def calculate_z_score(w_count: int, a_count: int, total_w: int, total_a: int) -> float:
    """
    Calculate z-score for the difference in character proportions.
    Uses pooled proportion for two-sample z-test.
    """
    if total_w == 0 or total_a == 0:
        return 0.0
    
    # Proportions
    p1 = w_count / total_w
    p2 = a_count / total_a
    
    # Pooled proportion
    p_pool = (w_count + a_count) / (total_w + total_a)
    
    # Standard error
    if p_pool == 0 or p_pool == 1:
        return 0.0
    
    se = math.sqrt(p_pool * (1 - p_pool) * (1/total_w + 1/total_a))
    
    if se == 0:
        return 0.0
    
    # Z-score
    z = (p1 - p2) / se
    
    return z

def calculate_effect_size(w_count: int, a_count: int) -> float:
    """Calculate effect size (ratio of counts)."""
    if w_count == 0:
        return float('inf') if a_count > 0 else 0.0
    return a_count / w_count

def run_statistical_analysis(results: List[Dict], output_dir: str):
    """Run comprehensive statistical analysis."""
    
    output_path = Path(output_dir) / 'statistical_analysis'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Aggregate counts
    wright_totals, aspley_totals = aggregate_character_counts(results)
    
    # Get all characters
    all_chars = set(wright_totals.keys()) | set(aspley_totals.keys())
    
    # Total character counts
    total_wright = sum(wright_totals.values())
    total_aspley = sum(aspley_totals.values())
    
    print("="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Wright vs Aspley Edition Character Comparison")
    print("="*80)
    print(f"\nTotal Wright characters: {total_wright:,}")
    print(f"Total Aspley characters: {total_aspley:,}")
    print(f"Overall ratio: {total_aspley/total_wright:.3f}")
    print()
    
    # Calculate statistics for each character
    char_stats = []
    
    for char in all_chars:
        w_count = wright_totals.get(char, 0)
        a_count = aspley_totals.get(char, 0)
        
        z_score = calculate_z_score(w_count, a_count, total_wright, total_aspley)
        effect_size = calculate_effect_size(w_count, a_count)
        delta = a_count - w_count
        
        # Determine significance (|z| > 1.96 for p < 0.05, |z| > 2.58 for p < 0.01)
        if abs(z_score) > 3.29:  # p < 0.001
            significance = "🔴 CRITICAL (p<0.001)"
        elif abs(z_score) > 2.58:  # p < 0.01
            significance = "🟠 HIGH (p<0.01)"
        elif abs(z_score) > 1.96:  # p < 0.05
            significance = "🟡 MODERATE (p<0.05)"
        else:
            significance = "🟢 NOT SIGNIFICANT"
        
        char_stats.append({
            'char': char,
            'wright': w_count,
            'aspley': a_count,
            'delta': delta,
            'z_score': z_score,
            'effect_size': effect_size,
            'significance': significance
        })
    
    # Sort by absolute z-score
    char_stats.sort(key=lambda x: abs(x['z_score']), reverse=True)
    
    # Print top significant characters
    print("\nTOP 25 STATISTICALLY SIGNIFICANT CHARACTER DIFFERENCES:")
    print("-"*100)
    print(f"{'Char':<8}{'Wright':<10}{'Aspley':<10}{'Delta':<10}{'Z-Score':<12}{'Ratio':<10}{'Significance'}")
    print("-"*100)
    
    significant_chars = []
    for stat in char_stats[:25]:
        char = stat['char']
        display_char = repr(char) if not char.isprintable() or len(char) != 1 else f"'{char}'"
        
        effect_str = f"{stat['effect_size']:.2f}" if stat['effect_size'] != float('inf') else "∞"
        
        print(f"{display_char:<8}{stat['wright']:<10}{stat['aspley']:<10}{stat['delta']:+<10}"
              f"{stat['z_score']:+.3f}     {effect_str:<10}{stat['significance']}")
        
        if "CRITICAL" in stat['significance'] or "HIGH" in stat['significance']:
            significant_chars.append(stat)
    
    # ============ Create Visualization ============
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Character-Level Statistical Analysis: Wright vs Aspley', fontsize=14, fontweight='bold')
    
    # Filter to characters with enough samples for meaningful analysis
    analysis_chars = [s for s in char_stats if s['wright'] > 10 or s['aspley'] > 10]
    
    # 1. Z-Score Distribution
    ax1 = axes[0, 0]
    z_scores = [s['z_score'] for s in analysis_chars]
    chars = [s['char'] if s['char'].isprintable() else '?' for s in analysis_chars]
    colors = ['#E74C3C' if abs(z) > 2.58 else '#F4D03F' if abs(z) > 1.96 else '#27AE60' for z in z_scores]
    
    # Sort by z-score for better visualization
    sorted_indices = np.argsort(z_scores)[::-1]
    
    # Take top and bottom 20
    top_bottom = list(sorted_indices[:15]) + list(sorted_indices[-15:])
    
    z_subset = [z_scores[i] for i in top_bottom]
    char_subset = [chars[i] for i in top_bottom]
    color_subset = [colors[i] for i in top_bottom]
    
    y_pos = np.arange(len(z_subset))
    ax1.barh(y_pos, z_subset, color=color_subset, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(char_subset, fontsize=9)
    ax1.axvline(x=1.96, color='orange', linestyle='--', label='p<0.05')
    ax1.axvline(x=-1.96, color='orange', linestyle='--')
    ax1.axvline(x=2.58, color='red', linestyle='--', label='p<0.01')
    ax1.axvline(x=-2.58, color='red', linestyle='--')
    ax1.set_xlabel('Z-Score')
    ax1.set_title('Z-Score by Character (Top/Bottom 15)')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Effect Size (Ratio) Distribution
    ax2 = axes[0, 1]
    effect_sizes = [s['effect_size'] if s['effect_size'] != float('inf') else 3 for s in analysis_chars]
    effect_colors = ['#9B59B6' if e > 2 or e < 0.5 else '#3498DB' for e in effect_sizes]
    
    ax2.hist(effect_sizes, bins=30, color='#3498DB', edgecolor='black', alpha=0.7)
    ax2.axvline(x=1.0, color='black', linewidth=2, label='Equal (1:1)')
    ax2.axvline(x=1.237, color='red', linestyle='--', label=f'Overall Ratio ({total_aspley/total_wright:.3f})')
    ax2.set_xlabel('Aspley/Wright Ratio')
    ax2.set_ylabel('Character Count')
    ax2.set_title('Distribution of Character Ratios')
    ax2.legend()
    ax2.set_xlim(0, 3)
    
    # 3. Wright vs Aspley Scatter
    ax3 = axes[1, 0]
    wright_counts = [s['wright'] for s in analysis_chars]
    aspley_counts = [s['aspley'] for s in analysis_chars]
    sig_colors = ['#E74C3C' if abs(s['z_score']) > 2.58 else '#3498DB' for s in analysis_chars]
    
    ax3.scatter(wright_counts, aspley_counts, c=sig_colors, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add perfect correlation line
    max_val = max(max(wright_counts), max(aspley_counts))
    ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect 1:1')
    ax3.plot([0, max_val], [0, max_val * 1.237], 'r-', linewidth=1, alpha=0.5, label='Overall Ratio')
    
    # Label significant outliers
    for s in char_stats[:10]:
        if s['wright'] > 10 or s['aspley'] > 10:
            char_label = s['char'] if s['char'].isprintable() else '?'
            ax3.annotate(char_label, (s['wright'], s['aspley']), fontsize=8, ha='center')
    
    ax3.set_xlabel('Wright Count')
    ax3.set_ylabel('Aspley Count')
    ax3.set_title('Character Counts: Wright vs Aspley')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Category-Level Analysis
    ax4 = axes[1, 1]
    
    # Categorize characters
    categories = {
        'lowercase': [],
        'uppercase': [],
        'punctuation': [],
        'digit': [],
        'special': []
    }
    
    for s in char_stats:
        char = s['char']
        if char.islower():
            categories['lowercase'].append(s)
        elif char.isupper():
            categories['uppercase'].append(s)
        elif char.isdigit():
            categories['digit'].append(s)
        elif char in '.,;:!?\'"()-':
            categories['punctuation'].append(s)
        else:
            categories['special'].append(s)
    
    cat_names = []
    cat_ratios = []
    cat_colors = []
    
    for cat_name, cat_chars in categories.items():
        if cat_chars:
            w_total = sum(c['wright'] for c in cat_chars)
            a_total = sum(c['aspley'] for c in cat_chars)
            if w_total > 0:
                ratio = a_total / w_total
                cat_names.append(cat_name)
                cat_ratios.append(ratio)
                
                if abs(ratio - 1.237) < 0.1:
                    cat_colors.append('#27AE60')  # Normal
                elif ratio > 1.237:
                    cat_colors.append('#E74C3C')  # Aspley higher
                else:
                    cat_colors.append('#3498DB')  # Wright higher
    
    x_pos = np.arange(len(cat_names))
    ax4.bar(x_pos, cat_ratios, color=cat_colors, edgecolor='black')
    ax4.axhline(y=1.0, color='black', linewidth=2, label='Equal')
    ax4.axhline(y=1.237, color='red', linestyle='--', label='Overall Ratio')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(cat_names, rotation=45, ha='right')
    ax4.set_ylabel('Aspley/Wright Ratio')
    ax4.set_title('Ratio by Character Category')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    vis_path = output_path / 'statistical_analysis.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nVisualization saved: {vis_path}")
    
    # ============ Save detailed report ============
    report_path = output_path / 'statistical_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE REPORT\n")
        f.write("Wright (STC 22353a) vs Aspley (STC 22353)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Wright characters: {total_wright:,}\n")
        f.write(f"Total Aspley characters: {total_aspley:,}\n")
        f.write(f"Overall ratio: {total_aspley/total_wright:.4f}\n")
        f.write(f"Overall difference: {total_aspley - total_wright:+,}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        f.write("If the character ratio matches the overall ratio (1.237), variance is\n")
        f.write("explained by OCR quality differences, NOT typographical variation.\n")
        f.write("Characters with ratios significantly different from 1.237 indicate\n")
        f.write("potential REAL typographical differences worth investigation.\n\n")
        
        f.write("CRITICAL FINDINGS (|z| > 2.58, p < 0.01):\n")
        f.write("-"*80 + "\n")
        for s in char_stats:
            if abs(s['z_score']) > 2.58:
                char = s['char']
                display_char = repr(char) if not char.isprintable() else char
                ratio = s['effect_size'] if s['effect_size'] != float('inf') else "∞"
                f.write(f"  {display_char}: Wright={s['wright']}, Aspley={s['aspley']}, ")
                f.write(f"Z={s['z_score']:+.2f}, Ratio={ratio}\n")
    
    print(f"Report saved: {report_path}")
    
    return char_stats, significant_chars

def main():
    json_path = "reports/page_forensics/page_comparison.json"
    output_dir = "reports/page_forensics"
    
    print("Loading page data...")
    results = load_all_page_data(json_path)
    
    print(f"Analyzing {len(results)} pages...")
    
    char_stats, significant = run_statistical_analysis(results, output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Total significant characters (p<0.01): {len([s for s in char_stats if abs(s['z_score']) > 2.58])}")

if __name__ == "__main__":
    main()
