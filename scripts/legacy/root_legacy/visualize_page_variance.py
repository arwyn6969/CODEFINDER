#!/usr/bin/env python3
"""
Page Variance Visualization Tool

Creates visual charts showing the character count variance between
Wright and Aspley editions across all pages.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def load_comparison_data(json_path: str) -> dict:
    """Load the page comparison JSON data."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_variance_chart(data: dict, output_dir: str):
    """Create a bar chart showing character count variance by page."""
    
    results = data['results']
    pages = [r['page'] for r in results]
    wright_chars = [r['wright_chars'] for r in results]
    aspley_chars = [r['aspley_chars'] for r in results]
    ratios = [r['ratio'] if r['ratio'] != float('inf') else 3.0 for r in results]
    significant = [r['significant'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle('Wright vs Aspley Edition: Page-by-Page Forensic Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ============ CHART 1: Character Counts Side by Side ============
    ax1 = axes[0]
    x = np.arange(len(pages))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, wright_chars, width, label='Wright (STC 22353a)', 
                    color='#2E86AB', alpha=0.85)
    bars2 = ax1.bar(x + width/2, aspley_chars, width, label='Aspley (STC 22353)', 
                    color='#A23B72', alpha=0.85)
    
    ax1.set_xlabel('Page Number', fontsize=11)
    ax1.set_ylabel('Character Count (80%+ confidence)', fontsize=11)
    ax1.set_title('Character Counts Per Page', fontsize=13, fontweight='bold')
    ax1.set_xticks(x[::2])  # Show every other page number
    ax1.set_xticklabels([str(pages[i]) for i in range(0, len(pages), 2)], fontsize=9)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # ============ CHART 2: Ratio with Significance Threshold ============
    ax2 = axes[1]
    
    # Color bars by significance
    colors = ['#E74C3C' if s else '#27AE60' for s in significant]
    bars = ax2.bar(x, ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=2, label='1:1 Ratio (Identical)')
    ax2.axhline(y=1.2, color='orange', linestyle='--', linewidth=1.5, label='Significance Threshold (+20%)')
    ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5)
    
    ax2.set_xlabel('Page Number', fontsize=11)
    ax2.set_ylabel('Aspley / Wright Ratio', fontsize=11)
    ax2.set_title('Character Count Ratio (Aspley ÷ Wright)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([str(pages[i]) for i in range(0, len(pages), 2)], fontsize=9)
    ax2.set_ylim(0, 2.5)
    
    # Legend for significance
    sig_patch = mpatches.Patch(color='#E74C3C', alpha=0.8, label='Significant Difference')
    nosig_patch = mpatches.Patch(color='#27AE60', alpha=0.8, label='Within Normal Variance')
    ax2.legend(handles=[sig_patch, nosig_patch], loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # ============ CHART 3: Delta (Difference) with Outliers Highlighted ============
    ax3 = axes[2]
    
    deltas = [r['delta'] for r in results]
    
    # Identify extreme outliers (top 10 by absolute delta)
    sorted_indices = sorted(range(len(deltas)), key=lambda i: abs(deltas[i]), reverse=True)[:10]
    outlier_set = set(sorted_indices)
    
    colors3 = ['#9B59B6' if i in outlier_set else '#3498DB' for i in range(len(deltas))]
    bars3 = ax3.bar(x, deltas, color=colors3, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add labels for top outliers
    for i in sorted_indices[:5]:
        ax3.annotate(f'Pg {pages[i]}\n({deltas[i]:+d})', 
                     xy=(x[i], deltas[i]), 
                     xytext=(0, 10 if deltas[i] > 0 else -20),
                     textcoords='offset points',
                     ha='center', fontsize=8, fontweight='bold',
                     color='#9B59B6')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axhline(y=100, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax3.axhline(y=-100, color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Page Number', fontsize=11)
    ax3.set_ylabel('Character Delta (Aspley - Wright)', fontsize=11)
    ax3.set_title('Character Count Difference Per Page', fontsize=13, fontweight='bold')
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([str(pages[i]) for i in range(0, len(pages), 2)], fontsize=9)
    
    outlier_patch = mpatches.Patch(color='#9B59B6', alpha=0.8, label='Top 10 Outliers')
    normal_patch = mpatches.Patch(color='#3498DB', alpha=0.8, label='Other Pages')
    ax3.legend(handles=[outlier_patch, normal_patch], loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'page_variance_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")
    return str(output_path)

def create_outlier_summary(data: dict, output_dir: str):
    """Create a summary of the top outlier pages."""
    
    results = data['results']
    
    # Sort by absolute delta
    sorted_results = sorted(results, key=lambda r: abs(r['delta']), reverse=True)
    
    print("\n" + "="*70)
    print("TOP 10 FORENSIC OUTLIER PAGES")
    print("="*70)
    print(f"{'Rank':<6}{'Page':<8}{'Wright':<10}{'Aspley':<10}{'Delta':<10}{'Ratio':<10}{'Priority'}")
    print("-"*70)
    
    priorities = []
    for i, r in enumerate(sorted_results[:10]):
        ratio = r['ratio'] if r['ratio'] != float('inf') else '∞'
        if isinstance(ratio, float):
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = ratio
            
        # Calculate priority based on both delta and ratio
        if r['ratio'] == float('inf') or r['ratio'] == 0:
            priority = "BINDING/BLANK"
        elif abs(r['delta']) > 600:
            priority = "🔴 CRITICAL"
        elif abs(r['delta']) > 400:
            priority = "🟠 HIGH"
        elif abs(r['delta']) > 200:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"
        
        print(f"{i+1:<6}{r['page']:<8}{r['wright_chars']:<10}{r['aspley_chars']:<10}{r['delta']:+<10}{ratio_str:<10}{priority}")
        priorities.append((r['page'], r['delta'], priority))
    
    # Save summary to file
    output_path = Path(output_dir) / 'outlier_summary.txt'
    with open(output_path, 'w') as f:
        f.write("FORENSIC OUTLIER PAGE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("Pages ranked by absolute character count difference (Aspley - Wright)\n\n")
        f.write(f"{'Rank':<6}{'Page':<8}{'Wright':<10}{'Aspley':<10}{'Delta':<12}{'Ratio':<10}{'Priority'}\n")
        f.write("-" * 70 + "\n")
        
        for i, r in enumerate(sorted_results[:20]):
            ratio = r['ratio'] if r['ratio'] != float('inf') else '∞'
            if isinstance(ratio, float):
                ratio_str = f"{ratio:.2f}"
            else:
                ratio_str = ratio
            f.write(f"{i+1:<6}{r['page']:<8}{r['wright_chars']:<10}{r['aspley_chars']:<10}{r['delta']:+<12}{ratio_str:<10}\n")
    
    print(f"\nSummary saved to: {output_path}")
    return priorities[:5]  # Return top 5 for next phase

def main():
    json_path = "reports/page_forensics/page_comparison.json"
    output_dir = "reports/page_forensics"
    
    print("Loading page comparison data...")
    data = load_comparison_data(json_path)
    
    print(f"Total pages analyzed: {data['total_pages']}")
    print(f"Significant differences: {data['significant_pages']}")
    
    print("\nCreating variance visualization...")
    create_variance_chart(data, output_dir)
    
    top_outliers = create_outlier_summary(data, output_dir)
    
    return top_outliers

if __name__ == "__main__":
    main()
