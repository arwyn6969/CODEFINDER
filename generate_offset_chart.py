#!/usr/bin/env python3
"""
Visual Page-Offset Chart Generator
===================================
Creates a visual chart showing page offset per Sonnet between Wright and Aspley.

Usage:
    python3 generate_offset_chart.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def main():
    # Load mapping data
    with open("reports/full_sonnet_mapping.json") as f:
        data = json.load(f)
    
    wright = data['wright']
    aspley = data['aspley']
    
    # Calculate offsets
    sonnets = []
    offsets = []
    colors = []
    
    for num in range(1, 155):
        w_page = wright.get(str(num), {}).get('page')
        a_page = aspley.get(str(num), {}).get('page')
        
        if w_page and a_page:
            offset = a_page - w_page
            sonnets.append(num)
            offsets.append(offset)
            
            # Color by offset magnitude
            if offset == 0:
                colors.append('#2ecc71')  # Green - same page
            elif offset > 0:
                colors.append('#e74c3c')  # Red - Aspley later
            else:
                colors.append('#3498db')  # Blue - Aspley earlier
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Chart 1: Scatter plot of offsets
    ax1.scatter(sonnets, offsets, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax1.set_xlabel('Sonnet # (1-154)', fontsize=12)
    ax1.set_ylabel('Page Offset (Aspley Page - Wright Page)', fontsize=12)
    ax1.set_title(f'Page Offset by Sonnet: {len(sonnets)} Sonnets Matched in Both Editions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 160)
    ax1.set_xticks([1, 25, 50, 75, 100, 125, 154])
    
    # Add note
    ax1.text(0.02, 0.02, f'Note: Only {len(sonnets)}/154 Sonnets detected\nvia first-line matching (OCR limitations)', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Same Page (offset=0)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Aspley Later (+offset)')
    blue_patch = mpatches.Patch(color='#3498db', label='Aspley Earlier (-offset)')
    ax1.legend(handles=[green_patch, red_patch, blue_patch], loc='upper right')
    
    # Chart 2: Histogram of offsets
    ax2.hist(offsets, bins=range(min(offsets)-5, max(offsets)+5, 2), 
             edgecolor='black', alpha=0.7, color='#9b59b6')
    ax2.axvline(x=0, color='#2ecc71', linestyle='--', linewidth=2, label='Same Page')
    ax2.set_xlabel('Page Offset', fontsize=12)
    ax2.set_ylabel('Number of Sonnets', fontsize=12)
    ax2.set_title('Distribution of Page Offsets', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics annotation
    same_page = offsets.count(0)
    stats_text = f"Total Sonnets Mapped: {len(sonnets)}\n"
    stats_text += f"Same Page: {same_page} ({100*same_page/len(sonnets):.1f}%)\n"
    stats_text += f"Offset Range: {min(offsets)} to +{max(offsets)}"
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path("reports/page_offset_chart.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    
    # Also create a summary text file
    summary = "PAGE OFFSET ANALYSIS\n"
    summary += "=" * 50 + "\n\n"
    summary += f"Sonnets with same page number: {same_page}\n"
    summary += f"Sonnets with Aspley LATER: {len([o for o in offsets if o > 0])}\n"
    summary += f"Sonnets with Aspley EARLIER: {len([o for o in offsets if o < 0])}\n"
    summary += f"\nOffset range: {min(offsets)} to +{max(offsets)}\n"
    summary += f"Mean offset: {sum(offsets)/len(offsets):.1f}\n"
    
    summary += "\n\nSonnets on SAME PAGE:\n"
    for i, (s, o) in enumerate(zip(sonnets, offsets)):
        if o == 0:
            w_pg = wright[str(s)]['page']
            summary += f"  Sonnet {s} - Page {w_pg}\n"
    
    with open("reports/page_offset_summary.txt", 'w') as f:
        f.write(summary)
    print("Summary saved to: reports/page_offset_summary.txt")


if __name__ == "__main__":
    main()
