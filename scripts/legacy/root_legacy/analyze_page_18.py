#!/usr/bin/env python3
"""
Page 18 Deep Forensic Analysis

Creates side-by-side comparison and overlay analysis of Page 18 (D1 verso / D2 recto)
which shows the highest character count variance between Wright and Aspley editions.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

def load_page_details(json_path: str, page_num: int) -> dict:
    """Load detailed character breakdown for a specific page."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for result in data['results']:
        if result['page'] == page_num:
            return result
    return None

def create_page_comparison(wright_path: str, aspley_path: str, page_num: int, output_dir: str):
    """Create side-by-side comparison of the specified page."""
    
    # Load images
    wright_img = Image.open(wright_path).convert('RGB')
    aspley_img = Image.open(aspley_path).convert('RGB')
    
    # Get dimensions
    w_width, w_height = wright_img.size
    a_width, a_height = aspley_img.size
    
    print(f"Wright Page {page_num}: {w_width}x{w_height}")
    print(f"Aspley Page {page_num}: {a_width}x{a_height}")
    
    # Resize Aspley to match Wright dimensions for proper comparison
    if (a_width, a_height) != (w_width, w_height):
        aspley_resized = aspley_img.resize((w_width, w_height), Image.Resampling.LANCZOS)
    else:
        aspley_resized = aspley_img
    
    # ============ 1. Side-by-Side Comparison ============
    side_by_side = Image.new('RGB', (w_width * 2 + 50, w_height + 100), color='white')
    
    # Add images
    side_by_side.paste(wright_img, (0, 80))
    side_by_side.paste(aspley_resized, (w_width + 50, 80))
    
    # Add labels
    draw = ImageDraw.Draw(side_by_side)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.text((w_width//2 - 150, 20), "WRIGHT (STC 22353a)", fill='#2E86AB', font=font)
    draw.text((w_width + 50 + w_width//2 - 150, 20), "ASPLEY (STC 22353)", fill='#A23B72', font=font)
    
    side_by_side_path = Path(output_dir) / f'page_{page_num}_side_by_side.png'
    side_by_side.save(side_by_side_path, quality=95)
    print(f"Saved: {side_by_side_path}")
    
    # ============ 2. Grayscale and Difference Analysis ============
    wright_gray = wright_img.convert('L')
    aspley_gray = aspley_resized.convert('L')
    
    wright_arr = np.array(wright_gray, dtype=np.float32)
    aspley_arr = np.array(aspley_gray, dtype=np.float32)
    
    # Calculate difference
    diff_arr = np.abs(wright_arr - aspley_arr)
    
    # Statistics
    mean_diff = np.mean(diff_arr)
    max_diff = np.max(diff_arr)
    std_diff = np.std(diff_arr)
    
    # Pixels with significant difference (>50 grayscale levels)
    significant_pixels = np.sum(diff_arr > 50)
    total_pixels = diff_arr.size
    significant_pct = (significant_pixels / total_pixels) * 100
    
    print(f"\nDifference Statistics:")
    print(f"  Mean pixel difference: {mean_diff:.2f}")
    print(f"  Max pixel difference: {max_diff:.2f}")
    print(f"  Standard deviation: {std_diff:.2f}")
    print(f"  Significant pixels (>50): {significant_pixels:,} ({significant_pct:.2f}%)")
    
    # Enhanced difference image
    diff_enhanced = np.clip(diff_arr * 3, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_enhanced)
    
    diff_path = Path(output_dir) / f'page_{page_num}_difference.png'
    diff_img.save(diff_path)
    print(f"Saved: {diff_path}")
    
    # ============ 3. Color-Coded Overlay ============
    # Red = Wright only, Cyan = Aspley only, Gray = Common
    color_overlay = Image.new('RGB', (w_width, w_height))
    color_arr = np.zeros((w_height, w_width, 3), dtype=np.uint8)
    
    # Invert so text is bright
    wright_inv = 255 - wright_arr
    aspley_inv = 255 - aspley_arr
    
    # Red channel = Wright text
    color_arr[:, :, 0] = np.clip(wright_inv, 0, 255).astype(np.uint8)
    # Green channel = common (min of both)
    color_arr[:, :, 1] = np.minimum(wright_inv, aspley_inv).astype(np.uint8)
    # Blue channel = Aspley text
    color_arr[:, :, 2] = np.clip(aspley_inv, 0, 255).astype(np.uint8)
    
    color_overlay = Image.fromarray(color_arr)
    
    overlay_path = Path(output_dir) / f'page_{page_num}_color_overlay.png'
    color_overlay.save(overlay_path)
    print(f"Saved: {overlay_path}")
    
    # ============ 4. Analysis Summary Image ============
    # Create summary with all views
    summary_height = 800
    summary = Image.new('RGB', (w_width * 2, summary_height), color='#1a1a2e')
    
    # Resize for thumbnails
    thumb_height = 600
    thumb_width = int((thumb_height / w_height) * w_width)
    
    wright_thumb = wright_img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    aspley_thumb = aspley_resized.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
    
    # Calculate offsets to center
    left_offset = (w_width - thumb_width) // 2
    right_offset = w_width + left_offset
    
    summary.paste(wright_thumb, (left_offset, 120))
    summary.paste(aspley_thumb, (right_offset, 120))
    
    draw = ImageDraw.Draw(summary)
    
    # Title
    draw.text((w_width - 250, 20), f"PAGE {page_num} FORENSIC ANALYSIS", fill='white', font=font)
    
    # Labels
    draw.text((left_offset + thumb_width//2 - 120, 80), "WRIGHT (997 chars)", fill='#2E86AB', font=small_font)
    draw.text((right_offset + thumb_width//2 - 120, 80), "ASPLEY (1705 chars)", fill='#A23B72', font=small_font)
    
    # Stats bar
    stats_y = 740
    draw.rectangle([(0, stats_y - 20), (w_width * 2, summary_height)], fill='#16213e')
    draw.text((50, stats_y), f"DELTA: +708 chars | RATIO: 1.71x | Mean Pixel Diff: {mean_diff:.1f} | Significant Pixels: {significant_pct:.1f}%", fill='#e94560', font=small_font)
    
    summary_path = Path(output_dir) / f'page_{page_num}_summary.png'
    summary.save(summary_path)
    print(f"Saved: {summary_path}")
    
    return {
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'std_diff': std_diff,
        'significant_pct': significant_pct
    }

def analyze_character_breakdown(page_data: dict, output_dir: str, page_num: int):
    """Analyze and visualize the character-level differences."""
    
    char_deltas = page_data.get('char_deltas', {})
    
    if not char_deltas:
        print("No character delta data available")
        return
    
    # Sort by absolute delta
    sorted_chars = sorted(char_deltas.items(), key=lambda x: abs(x[1]['delta']), reverse=True)
    
    print(f"\nTOP CHARACTER DIFFERENCES (Page {page_num}):")
    print("-" * 60)
    print(f"{'Char':<8}{'Wright':<10}{'Aspley':<10}{'Delta':<10}{'% Change'}")
    print("-" * 60)
    
    for char, data in sorted_chars[:15]:
        display_char = repr(char) if len(char) == 1 and not char.isprintable() else f"'{char}'"
        w_count = data['wright']
        a_count = data['aspley']
        delta = data['delta']
        pct_change = ((a_count - w_count) / w_count * 100) if w_count > 0 else float('inf')
        
        if pct_change == float('inf'):
            pct_str = "NEW"
        else:
            pct_str = f"{pct_change:+.1f}%"
        
        print(f"{display_char:<8}{w_count:<10}{a_count:<10}{delta:+<10}{pct_str}")
    
    return sorted_chars

def main():
    page_num = 18  # Highest variance page
    
    output_dir = Path("reports/page_forensics/deep_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths (page 18 = file index 17, 0-indexed)
    wright_path = "data/sources/folger_sonnets_1609/018_d1_verso_d2_recto.jpg"
    aspley_files = sorted(Path("data/sources/folger_sonnets_1609_aspley").glob("*.jpg"))
    aspley_path = str(aspley_files[17])  # 18th file (0-indexed)
    
    print(f"Wright source: {wright_path}")
    print(f"Aspley source: {aspley_path}")
    print()
    
    # Load page data
    json_path = "reports/page_forensics/page_comparison.json"
    page_data = load_page_details(json_path, page_num)
    
    if page_data:
        print(f"Page {page_num} Summary:")
        print(f"  Wright chars: {page_data['wright_chars']}")
        print(f"  Aspley chars: {page_data['aspley_chars']}")
        print(f"  Delta: {page_data['delta']:+d}")
        print(f"  Ratio: {page_data['ratio']:.3f}")
        print()
    
    # Create visual comparison
    stats = create_page_comparison(wright_path, aspley_path, page_num, str(output_dir))
    
    # Analyze character breakdown
    if page_data:
        analyze_character_breakdown(page_data, str(output_dir), page_num)
    
    print(f"\n✅ Deep analysis complete. Check: {output_dir}")

if __name__ == "__main__":
    main()
